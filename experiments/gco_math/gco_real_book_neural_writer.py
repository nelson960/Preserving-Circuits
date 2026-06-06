#!/usr/bin/env python3
"""Train a neural GCO residual writer on real-book residual dynamics.

This experiment removes the deterministic write controller from the previous
capacity-frontier probes. A trainable writer sees real residual geometry and an
error/write target, then learns a sparse route and write vector from outcome
loss only:

    absolute mode:
        h_new = W_base k_new + u phi(k_new)
        h_old = W_base K_old + u phi(K_old)
        phi(k) = softplus((r^T k - b) / temperature) * temperature

    relative mode:
        h_new = W_base k_new + u
        h_old = W_base K_old + u psi(K_old, k_new)
        psi(k_old, k_new) = softplus((r^T k_old - r^T k_new - m) / temperature) * temperature

    rank1 mode:
        Delta W = u a^T
        h_new = (W_base + Delta W) k_new
        h_old = (W_base + Delta W) K_old
        a = k_new / ||k_new||^2 + q_perp
        q_perp^T k_new = 0, therefore a^T k_new = 1

The writer is not given action labels and it does not use the closed-form
protected write solve. It is trained to reduce new residual error while keeping
old residual damage low.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm.auto import tqdm

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

GCO_DIR = Path(__file__).resolve().parent
if str(GCO_DIR) not in sys.path:
    sys.path.append(str(GCO_DIR))

from gco_learned_route_constructor import (  # noqa: E402
    mse_columns,
    parse_int_list,
    resolve_device,
    scalar,
    set_seed,
)
from gco_real_book_capacity_frontier import (  # noqa: E402
    CapacityCase,
    build_capacity_cases,
    collect_examples,
)
from gco_residual_route_growth import (  # noqa: E402
    ResidualExample,
    instantiate_base_model,
    load_chunks,
    overlap_bucket,
    require_finite_float,
    require_finite_tensor,
    resolve_dtype,
    resolve_layer_index,
    ridge_base_weight,
    select_chunks,
)


@dataclass(frozen=True)
class WriterEval:
    split: str
    old_count: int
    overlap_bucket: str
    case_count: int
    new_gain_fraction: float
    old_damage_mean: float
    old_damage_max: float
    route_new_activation: float
    route_old_activation_rms: float
    write_gate: float
    write_norm: float
    loss: float


class NeuralResidualWriter(nn.Module):
    def __init__(
        self,
        *,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        temperature: float,
        writer_mode: str,
        write_anchor: str,
        write_residual_scale: float,
    ) -> None:
        super().__init__()
        if key_dim <= 0:
            raise ValueError("key_dim must be positive.")
        if value_dim <= 0:
            raise ValueError("value_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("--hidden-dim must be positive.")
        if temperature <= 0.0:
            raise ValueError("--route-temperature must be positive.")
        if writer_mode not in {"absolute", "relative", "rank1"}:
            raise ValueError("--writer-mode must be 'absolute', 'relative', or 'rank1'.")
        if write_anchor not in {"none", "error"}:
            raise ValueError("--write-anchor must be 'none' or 'error'.")
        if write_residual_scale < 0.0:
            raise ValueError("--write-residual-scale must be non-negative.")
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.temperature = temperature
        self.writer_mode = writer_mode
        self.write_anchor = write_anchor
        self.write_residual_scale = write_residual_scale
        input_dim = 3 * key_dim + 2 * value_dim + 6
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.direction_head = nn.Linear(hidden_dim, key_dim)
        self.threshold_head = nn.Linear(hidden_dim, 1)
        self.write_head = nn.Linear(hidden_dim, value_dim)
        self.write_gate_head = nn.Linear(hidden_dim, 1)

    def features(
        self,
        *,
        w_base: torch.Tensor,
        old_keys: torch.Tensor,
        old_values: torch.Tensor,
        old_usage: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if old_usage.ndim != 1 or old_usage.shape[0] != old_keys.shape[1]:
            raise ValueError(f"old_usage must match old key count, got {old_usage.shape}, {old_keys.shape}.")
        usage_mass = old_usage.sum()
        if scalar(usage_mass.detach()) <= eps:
            raise RuntimeError("Cannot build neural-writer features with near-zero old usage mass.")
        usage_weights = old_usage / usage_mass
        old_key_mean = old_keys.mean(dim=1, keepdim=True)
        old_value_mean = old_values.mean(dim=1, keepdim=True)
        usage_key_mean = old_keys @ usage_weights.reshape(-1, 1)
        base_new = w_base @ k_new
        error = v_new - base_new
        overlap = torch.clamp((k_new.T @ old_keys).max(), min=0.0, max=1.0)
        old_count = torch.tensor([[float(old_keys.shape[1])]], device=k_new.device, dtype=k_new.dtype)
        scalars = torch.cat(
            [
                overlap.reshape(1, 1),
                (1.0 - overlap.square()).reshape(1, 1),
                old_usage.mean().reshape(1, 1),
                old_usage.max().reshape(1, 1),
                old_count,
                torch.linalg.vector_norm(error).reshape(1, 1),
            ],
            dim=1,
        )
        feature = torch.cat(
            [
                k_new.T,
                old_key_mean.T,
                usage_key_mean.T,
                v_new.T,
                error.T,
                scalars,
            ],
            dim=1,
        )
        require_finite_tensor("neural_writer_features", feature)
        return feature

    def forward(
        self,
        *,
        w_base: torch.Tensor,
        old_keys: torch.Tensor,
        old_values: torch.Tensor,
        old_usage: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.features(
            w_base=w_base,
            old_keys=old_keys,
            old_values=old_values,
            old_usage=old_usage,
            k_new=k_new,
            v_new=v_new,
            eps=eps,
        )
        hidden = self.net(feature)
        raw_direction = self.direction_head(hidden)
        norm = torch.linalg.vector_norm(raw_direction, dim=1, keepdim=True)
        if scalar(norm.detach().min()) <= eps:
            raise RuntimeError("Neural writer produced a near-zero route direction.")
        direction = raw_direction / norm
        threshold = self.threshold_head(hidden)
        raw_write = self.write_head(hidden).T
        if self.write_anchor == "none":
            write_vector = raw_write
            write_gate = torch.zeros((1, 1), device=k_new.device, dtype=k_new.dtype)
        elif self.write_anchor == "error":
            error = v_new - w_base @ k_new
            write_gate = torch.sigmoid(self.write_gate_head(hidden)).T
            write_vector = write_gate * error + self.write_residual_scale * torch.tanh(raw_write)
        else:
            raise RuntimeError(f"Unhandled write anchor {self.write_anchor!r}.")
        old_projection = direction @ old_keys
        new_projection = direction @ k_new
        if self.writer_mode == "absolute":
            z_old = F.softplus((old_projection - threshold) / self.temperature) * self.temperature
            z_new = F.softplus((new_projection - threshold) / self.temperature) * self.temperature
        elif self.writer_mode == "relative":
            margin = F.softplus(threshold)
            z_old = F.softplus((old_projection - new_projection - margin) / self.temperature) * self.temperature
            z_new = torch.ones_like(new_projection)
        elif self.writer_mode == "rank1":
            raw_key = raw_direction.T
            key_norm_sq = (k_new * k_new).sum().clamp_min(eps)
            projected = ((raw_key.T @ k_new) / key_norm_sq) * k_new
            orthogonal_key = raw_key - projected
            write_key = k_new / key_norm_sq + orthogonal_key
            z_old = write_key.T @ old_keys
            z_new = write_key.T @ k_new
            direction = F.normalize(write_key.T, dim=-1)
        else:
            raise RuntimeError(f"Unhandled writer mode {self.writer_mode!r}.")
        require_finite_tensor("neural_writer_direction", direction)
        require_finite_tensor("neural_writer_threshold", threshold)
        require_finite_tensor("neural_writer_write_vector", write_vector)
        require_finite_tensor("neural_writer_z_old", z_old)
        require_finite_tensor("neural_writer_z_new", z_new)
        require_finite_tensor("neural_writer_write_gate", write_gate)
        return write_vector, z_old, z_new, direction, threshold, write_gate


def positive_float(name: str, value: float) -> None:
    require_finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    require_finite_float(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def writer_loss_for_case(
    *,
    writer: NeuralResidualWriter,
    case: CapacityCase,
    base_ridge: float,
    old_damage_weight: float,
    max_damage_weight: float,
    route_leak_weight: float,
    write_norm_weight: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w_base = ridge_base_weight(case.old_keys, case.old_values, base_ridge)
    write_vector, z_old, z_new, _direction, _threshold, write_gate = writer(
        w_base=w_base,
        old_keys=case.old_keys,
        old_values=case.old_values,
        old_usage=case.old_usage,
        k_new=case.k_new,
        v_new=case.v_new,
        eps=eps,
    )
    old_before = mse_columns(w_base @ case.old_keys, case.old_values).detach()
    new_before = mse_columns(w_base @ case.k_new, case.v_new).mean().detach().clamp_min(eps)
    old_after = mse_columns(w_base @ case.old_keys + write_vector @ z_old, case.old_values)
    new_after = mse_columns(w_base @ case.k_new + write_vector @ z_new, case.v_new).mean()
    damage = torch.relu(old_after - old_before)
    old_scale = old_before.mean().detach().clamp_min(eps)
    write_norm = torch.linalg.vector_norm(write_vector)
    route_leak = z_old.square().mean()
    loss = (
        new_after / new_before
        + old_damage_weight * damage.mean() / old_scale
        + max_damage_weight * damage.max() / old_scale
        + route_leak_weight * route_leak
        + write_norm_weight * write_norm.square()
    )
    require_finite_tensor("neural_writer_loss", loss)
    metrics = {
        "loss": loss.detach(),
        "new_before": new_before.detach(),
        "new_after": new_after.detach(),
        "old_damage_mean": damage.mean().detach(),
        "old_damage_max": damage.max().detach(),
        "route_new_activation": z_new.mean().detach(),
        "route_old_activation_rms": torch.sqrt(z_old.square().mean()).detach(),
        "write_gate": write_gate.mean().detach(),
        "write_norm": write_norm.detach(),
    }
    return loss, metrics


def evaluate_cases(
    *,
    writer: NeuralResidualWriter,
    cases: Sequence[CapacityCase],
    split: str,
    base_ridge: float,
    old_damage_weight: float,
    max_damage_weight: float,
    route_leak_weight: float,
    write_norm_weight: float,
    eps: float,
) -> list[WriterEval]:
    if not cases:
        raise ValueError("Cannot evaluate an empty case list.")
    grouped: dict[tuple[int, str], list[dict[str, float]]] = {}
    writer.eval()
    with torch.no_grad():
        for case in cases:
            _loss, metrics = writer_loss_for_case(
                writer=writer,
                case=case,
                base_ridge=base_ridge,
                old_damage_weight=old_damage_weight,
                max_damage_weight=max_damage_weight,
                route_leak_weight=route_leak_weight,
                write_norm_weight=write_norm_weight,
                eps=eps,
            )
            new_gain = (metrics["new_before"] - metrics["new_after"]) / metrics["new_before"].clamp_min(eps)
            bucket = overlap_bucket(case.protected_overlap_ratio)
            grouped.setdefault((case.old_count, bucket), []).append(
                {
                    "new_gain_fraction": scalar(new_gain),
                    "old_damage_mean": scalar(metrics["old_damage_mean"]),
                    "old_damage_max": scalar(metrics["old_damage_max"]),
                    "route_new_activation": scalar(metrics["route_new_activation"]),
                    "route_old_activation_rms": scalar(metrics["route_old_activation_rms"]),
                    "write_gate": scalar(metrics["write_gate"]),
                    "write_norm": scalar(metrics["write_norm"]),
                    "loss": scalar(metrics["loss"]),
                }
            )
    output: list[WriterEval] = []
    for (old_count, bucket), items in sorted(grouped.items()):
        def avg(name: str) -> float:
            values = [item[name] for item in items]
            for value in values:
                require_finite_float(name, value)
            return float(sum(values) / len(values))

        output.append(
            WriterEval(
                split=split,
                old_count=old_count,
                overlap_bucket=bucket,
                case_count=len(items),
                new_gain_fraction=avg("new_gain_fraction"),
                old_damage_mean=avg("old_damage_mean"),
                old_damage_max=avg("old_damage_max"),
                route_new_activation=avg("route_new_activation"),
                route_old_activation_rms=avg("route_old_activation_rms"),
                write_gate=avg("write_gate"),
                write_norm=avg("write_norm"),
                loss=avg("loss"),
            )
        )
    return output


def print_eval(rows: Sequence[WriterEval]) -> None:
    print("\nReadable neural-writer summary")
    print("-" * 128)
    print("split old overlap       n  gain old_dmg max_dmg r_new r_old gate write_norm loss")
    print("-" * 128)
    for row in rows:
        print(
            f"{row.split:<5} "
            f"{row.old_count:3d} "
            f"{row.overlap_bucket:>11} "
            f"{row.case_count:3d} "
            f"{row.new_gain_fraction:5.3f} "
            f"{row.old_damage_mean:7.3g} "
            f"{row.old_damage_max:7.3g} "
            f"{row.route_new_activation:5.3f} "
            f"{row.route_old_activation_rms:5.3f} "
            f"{row.write_gate:5.3f} "
            f"{row.write_norm:10.3g} "
            f"{row.loss:7.3f}"
        )
    print("-" * 128)
    print("\nWhat to look for:")
    print("  train gain rising means the neural writer is learning the write.")
    print("  eval gain rising means the learned write rule transfers to held-out real residuals.")
    print("  old_dmg/max_dmg show whether the learned write damages protected old residual behavior.")
    print("  r_new high with r_old low means the route is not just going silent.")
    print("  gate shows how strongly the error-anchored writer uses the current residual error.")


def collect_real_examples_for_chunks(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    chunks: Sequence[dict[str, object]],
    layer_index: int,
    device: torch.device,
) -> list[ResidualExample]:
    return collect_examples(
        model=model,
        tokenizer=tokenizer,
        chunks=chunks,
        layer_index=layer_index,
        max_seq_len=args.max_seq_len,
        window_stride=args.window_stride,
        sequences_per_chunk=args.sequences_per_chunk,
        examples_per_chunk=args.examples_per_chunk,
        min_grad_norm=args.min_grad_norm,
        device=device,
        eps=args.eps,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-real-book-neural-writer-seed0.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--old-start", type=int, default=0)
    parser.add_argument("--old-chunk-count", type=int, default=2)
    parser.add_argument("--train-new-start", type=int, default=2)
    parser.add_argument("--train-new-chunk-count", type=int, default=1)
    parser.add_argument("--eval-new-start", type=int, default=3)
    parser.add_argument("--eval-new-chunk-count", type=int, default=1)
    parser.add_argument("--sequences-per-chunk", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=64)
    parser.add_argument("--examples-per-chunk", type=int, default=160)
    parser.add_argument("--min-grad-norm", type=float, default=1e-8)
    parser.add_argument("--old-counts", type=str, default="160,240")
    parser.add_argument("--cases-per-old-count", type=int, default=32)
    parser.add_argument("--old-bank-mode", choices=["global", "nearest"], default="global")
    parser.add_argument("--base-ridge", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--writer-mode", choices=["absolute", "relative", "rank1"], default="relative")
    parser.add_argument("--write-anchor", choices=["none", "error"], default="error")
    parser.add_argument("--write-residual-scale", type=float, default=0.1)
    parser.add_argument("--route-temperature", type=float, default=0.05)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--old-damage-weight", type=float, default=5.0)
    parser.add_argument("--max-damage-weight", type=float, default=2.0)
    parser.add_argument("--route-leak-weight", type=float, default=0.0)
    parser.add_argument("--write-norm-weight", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    positive_float("base_ridge", args.base_ridge)
    positive_float("lr", args.lr)
    positive_float("route_temperature", args.route_temperature)
    nonnegative_float("write_residual_scale", args.write_residual_scale)
    nonnegative_float("old_damage_weight", args.old_damage_weight)
    nonnegative_float("max_damage_weight", args.max_damage_weight)
    nonnegative_float("route_leak_weight", args.route_leak_weight)
    nonnegative_float("write_norm_weight", args.write_norm_weight)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    old_chunks = select_chunks(chunks, start=args.old_start, count=args.old_chunk_count, name="old chunks")
    train_new_chunks = select_chunks(
        chunks,
        start=args.train_new_start,
        count=args.train_new_chunk_count,
        name="train new chunks",
    )
    eval_new_chunks = select_chunks(
        chunks,
        start=args.eval_new_start,
        count=args.eval_new_chunk_count,
        name="eval new chunks",
    )
    old_ids = {str(chunk["chunk_id"]) for chunk in old_chunks}
    train_new_ids = {str(chunk["chunk_id"]) for chunk in train_new_chunks}
    eval_new_ids = {str(chunk["chunk_id"]) for chunk in eval_new_chunks}
    if old_ids & train_new_ids:
        raise ValueError(f"Old and train-new chunks overlap: {sorted(old_ids & train_new_ids)}")
    if old_ids & eval_new_ids:
        raise ValueError(f"Old and eval-new chunks overlap: {sorted(old_ids & eval_new_ids)}")
    if train_new_ids & eval_new_ids:
        raise ValueError(f"Train-new and eval-new chunks overlap: {sorted(train_new_ids & eval_new_ids)}")

    model = instantiate_base_model(args, tokenizer.get_vocab_size(), device)
    model.to(dtype=dtype)
    layer_index = resolve_layer_index(args.layer_index, len(model.blocks))
    old_counts = parse_int_list(args.old_counts)
    max_old_count = max(old_counts)

    print("GCO REAL-BOOK NEURAL WRITER")
    print("=" * 112)
    print("Question:")
    print("  Can a trainable writer learn safe residual writes from real residual geometry?")
    print("\nLearned write:")
    if args.writer_mode == "absolute":
        print("  h_new = W_base k_new + u phi(k_new)")
        print("  h_old = W_base K_old + u phi(K_old)")
        print("  phi(k) = softplus((r^T k - b) / T) * T")
    elif args.writer_mode == "relative":
        print("  h_new = W_base k_new + u")
        print("  h_old = W_base K_old + u psi(K_old, k_new)")
        print("  psi(k_old, k_new) = softplus((r^T k_old - r^T k_new - m) / T) * T")
    elif args.writer_mode == "rank1":
        print("  Delta W = u a^T")
        print("  h_new = (W_base + Delta W) k_new")
        print("  h_old = (W_base + Delta W) K_old")
        print("  a = k_new / ||k_new||^2 + q_perp, with q_perp^T k_new = 0")
    else:
        raise RuntimeError(f"Unhandled writer mode {args.writer_mode!r}.")
    print("\nResidual source:")
    print(
        f"  layer_index={layer_index}, old_chunks={sorted(old_ids)}, "
        f"train_new_chunks={sorted(train_new_ids)}, eval_new_chunks={sorted(eval_new_ids)}, "
        f"old_bank_mode={args.old_bank_mode}"
    )
    print("=" * 112)

    old_examples = collect_real_examples_for_chunks(
        args=args,
        model=model,
        tokenizer=tokenizer,
        chunks=old_chunks,
        layer_index=layer_index,
        device=device,
    )
    if max_old_count > len(old_examples):
        raise ValueError(f"Max old_count={max_old_count} exceeds captured old examples {len(old_examples)}.")
    train_new_examples = collect_real_examples_for_chunks(
        args=args,
        model=model,
        tokenizer=tokenizer,
        chunks=train_new_chunks,
        layer_index=layer_index,
        device=device,
    )
    eval_new_examples = collect_real_examples_for_chunks(
        args=args,
        model=model,
        tokenizer=tokenizer,
        chunks=eval_new_chunks,
        layer_index=layer_index,
        device=device,
    )
    train_cases = build_capacity_cases(
        old_examples=old_examples,
        new_examples=train_new_examples,
        old_counts=old_counts,
        max_cases_per_old_count=args.cases_per_old_count,
        old_bank_mode=args.old_bank_mode,
        eps=args.eps,
    )
    eval_cases = build_capacity_cases(
        old_examples=old_examples,
        new_examples=eval_new_examples,
        old_counts=old_counts,
        max_cases_per_old_count=args.cases_per_old_count,
        old_bank_mode=args.old_bank_mode,
        eps=args.eps,
    )
    key_dim = train_cases[0].k_new.shape[0]
    value_dim = train_cases[0].v_new.shape[0]
    writer = NeuralResidualWriter(
        key_dim=key_dim,
        value_dim=value_dim,
        hidden_dim=args.hidden_dim,
        temperature=args.route_temperature,
        writer_mode=args.writer_mode,
        write_anchor=args.write_anchor,
        write_residual_scale=args.write_residual_scale,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr)

    trace: list[dict[str, float | int]] = []
    progress = tqdm(range(1, args.train_steps + 1), desc="train neural writer", unit="step")
    for step in progress:
        writer.train()
        case = train_cases[(step - 1) % len(train_cases)]
        loss, metrics = writer_loss_for_case(
            writer=writer,
            case=case,
            base_ridge=args.base_ridge,
            old_damage_weight=args.old_damage_weight,
            max_damage_weight=args.max_damage_weight,
            route_leak_weight=args.route_leak_weight,
            write_norm_weight=args.write_norm_weight,
            eps=args.eps,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            new_gain = (metrics["new_before"] - metrics["new_after"]) / metrics["new_before"].clamp_min(args.eps)
            row = {
                "step": step,
                "loss": scalar(metrics["loss"]),
                "new_gain_fraction": scalar(new_gain),
                "old_damage_mean": scalar(metrics["old_damage_mean"]),
                "old_damage_max": scalar(metrics["old_damage_max"]),
                "route_new_activation": scalar(metrics["route_new_activation"]),
                "route_old_activation_rms": scalar(metrics["route_old_activation_rms"]),
                "write_gate": scalar(metrics["write_gate"]),
                "write_norm": scalar(metrics["write_norm"]),
            }
            trace.append(row)
            progress.set_postfix(
                loss=f"{row['loss']:.5f}",
                gain=f"{row['new_gain_fraction']:.5f}",
                r_new=f"{row['route_new_activation']:.5f}",
                r_old=f"{row['route_old_activation_rms']:.5f}",
                gate=f"{row['write_gate']:.5f}",
            )
            tqdm.write(
                "step={step:6d} loss={loss:.5f} gain={new_gain_fraction:.5f} "
                "old_dmg={old_damage_mean:.6g} max_dmg={old_damage_max:.6g} "
                "r_new={route_new_activation:.5f} r_old={route_old_activation_rms:.5f} "
                "gate={write_gate:.5f} "
                "write_norm={write_norm:.5f}".format(**row)
            )

    train_eval = evaluate_cases(
        writer=writer,
        cases=train_cases,
        split="train",
        base_ridge=args.base_ridge,
        old_damage_weight=args.old_damage_weight,
        max_damage_weight=args.max_damage_weight,
        route_leak_weight=args.route_leak_weight,
        write_norm_weight=args.write_norm_weight,
        eps=args.eps,
    )
    heldout_eval = evaluate_cases(
        writer=writer,
        cases=eval_cases,
        split="eval",
        base_ridge=args.base_ridge,
        old_damage_weight=args.old_damage_weight,
        max_damage_weight=args.max_damage_weight,
        route_leak_weight=args.route_leak_weight,
        write_norm_weight=args.write_norm_weight,
        eps=args.eps,
    )
    rows = train_eval + heldout_eval
    print_eval(rows)
    result = {
        "config": {
            "base_model_path": str(args.base_model_path),
            "tokenizer_path": str(args.tokenizer_path),
            "chunks_path": str(args.chunks_path),
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "layer_index": layer_index,
            "old_chunk_ids": sorted(old_ids),
            "train_new_chunk_ids": sorted(train_new_ids),
            "eval_new_chunk_ids": sorted(eval_new_ids),
            "old_counts": old_counts,
            "old_bank_mode": args.old_bank_mode,
            "cases_per_old_count": args.cases_per_old_count,
            "base_ridge": args.base_ridge,
            "hidden_dim": args.hidden_dim,
            "writer_mode": args.writer_mode,
            "write_anchor": args.write_anchor,
            "write_residual_scale": args.write_residual_scale,
            "route_temperature": args.route_temperature,
            "train_steps": args.train_steps,
            "lr": args.lr,
            "old_damage_weight": args.old_damage_weight,
            "max_damage_weight": args.max_damage_weight,
            "route_leak_weight": args.route_leak_weight,
            "write_norm_weight": args.write_norm_weight,
        },
        "question": "Can a trainable writer learn safe residual writes from real residual geometry?",
        "old_example_count": len(old_examples),
        "train_new_example_count": len(train_new_examples),
        "eval_new_example_count": len(eval_new_examples),
        "train_case_count": len(train_cases),
        "eval_case_count": len(eval_cases),
        "trace": trace,
        "aggregate": [asdict(row) for row in rows],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nwrote_json={args.output_json}")
    return result


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
