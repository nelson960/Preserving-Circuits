"""Bridge-window + plastic-adapter continual-learning diagnostic.

This script tests a minimal version of:

    behavior anchors + bridge windows + plastic adapter + consolidation

The experiment has two phases:

1. Plastic phase:
   Freeze the fitted 100-word base model and train a temporary residual adapter
   on the new 100 words, old behavior anchors, and old->new bridge windows.

2. Consolidation phase:
   Train a fresh copy of the base model core to absorb the adapter behavior
   while still preserving old output anchors. The final checkpoint contains only
   the consolidated core model, not the temporary adapter.

This is a diagnostic path test, not the final optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_native_scratch_transformer import GCONativeTransformer, NativeGCOConfig
from experiments.gco_math.gco_prepare_tiny_cl_base import build_lm_windows, evaluate_model, load_chunks
from experiments.gco_math.gco_tiny_cl_behavior_budget_sweep import (
    output_kl_and_agreement,
    select_old_probe_indices,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    collect_logits,
    distillation_kl,
    geometry_report,
    load_checkpoint,
    make_optimizer,
    mean_layer_metric,
    old_margin_loss,
    set_only_native_weights_trainable,
    target_margins_from_logits,
    trainable_weight_parameters,
)
from experiments.gco_math.gco_visualize_tiny_geometry_drift import build_windows, word_span


class FinalResidualAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int, scale: float) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale must be positive and finite, got {scale}.")
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.scale = float(scale)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"Adapter input must be [batch, seq, d_model], got {h.shape}.")
        return h + self.scale * self.up(F.gelu(self.down(h)))

    def penalty(self) -> torch.Tensor:
        return self.down.weight.square().mean() + self.up.weight.square().mean()


class AdapterWrappedTransformer(nn.Module):
    def __init__(self, base: GCONativeTransformer, adapter: FinalResidualAdapter) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [batch, seq], got {tokens.shape}.")
        batch, seq_len = tokens.shape
        if seq_len > self.base.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.base.max_seq_len}.")
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
        h = self.base.token_embedding(tokens) + self.base.position_embedding(positions)
        for block in self.base.blocks:
            h = block(h)
        h = self.base.ln_f(h)
        h = self.adapter(h)
        return self.base.lm_head(h)


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def build_bridge_lm_windows(
    token_ids: list[int],
    *,
    boundary_token_index: int,
    seq_len: int,
    stride: int,
    max_windows: int,
    radius_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_int("seq_len", seq_len)
    positive_int("stride", stride)
    positive_int("max_windows", max_windows)
    nonnegative_int("radius_tokens", radius_tokens)
    nonnegative_int("boundary_token_index", boundary_token_index)
    if boundary_token_index <= 0 or boundary_token_index >= len(token_ids):
        raise ValueError(
            f"boundary_token_index must lie inside the token sequence, got {boundary_token_index} "
            f"for length {len(token_ids)}."
        )
    if len(token_ids) < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} tokens, got {len(token_ids)}.")
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    starts: list[int] = []
    last_start = len(token_ids) - seq_len - 1
    for start in range(0, last_start + 1, stride):
        input_end = start + seq_len - 1
        target_end = start + seq_len
        crosses_boundary = start < boundary_token_index <= target_end
        near_boundary = abs(start - boundary_token_index) <= radius_tokens or abs(input_end - boundary_token_index) <= radius_tokens
        if not crosses_boundary and not near_boundary:
            continue
        window = token_ids[start : start + seq_len + 1]
        if len(window) != seq_len + 1:
            raise RuntimeError(f"Bridge window length mismatch at start={start}: {len(window)}.")
        inputs.append(window[:-1])
        targets.append(window[1:])
        starts.append(start)
        if len(inputs) >= max_windows:
            break
    if not inputs:
        raise RuntimeError(
            "No bridge windows were built. Increase --bridge-radius-tokens or check the old/new boundary."
        )
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(starts, dtype=torch.long),
    )


def sample_indices(total: int, count: int, *, generator: torch.Generator) -> torch.Tensor:
    positive_int("total", total)
    positive_int("count", count)
    return torch.randint(low=0, high=total, size=(count,), generator=generator, device=torch.device("cpu"))


def ce_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = model(inputs.to(device))
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(device).reshape(-1))


def trainable_adapter_parameters(adapter: FinalResidualAdapter) -> list[torch.nn.Parameter]:
    params = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("Adapter has no trainable parameters.")
    return params


def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def train_plastic_adapter(
    *,
    args: argparse.Namespace,
    wrapped: AdapterWrappedTransformer,
    adapter: FinalResidualAdapter,
    data: dict[str, torch.Tensor],
    old_probe_indices: torch.Tensor,
    teacher_old_logits: torch.Tensor,
    teacher_old_margins: torch.Tensor,
    device: torch.device,
) -> list[dict[str, float]]:
    freeze_model(wrapped.base)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    optimizer = make_optimizer(args, trainable_adapter_parameters(adapter))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 101)

    old_inputs = data["old_inputs"]
    old_targets = data["old_targets"]
    new_inputs = data["new_inputs"]
    new_targets = data["new_targets"]
    bridge_inputs = data["bridge_inputs"]
    bridge_targets = data["bridge_targets"]
    probe_count = int(old_probe_indices.numel())
    selected_old_inputs = old_inputs[old_probe_indices] if probe_count > 0 else old_inputs[:0]
    selected_old_targets = old_targets[old_probe_indices] if probe_count > 0 else old_targets[:0]
    selected_teacher_logits = teacher_old_logits[old_probe_indices] if probe_count > 0 else teacher_old_logits[:0]
    selected_teacher_margins = teacher_old_margins[old_probe_indices] if probe_count > 0 else teacher_old_margins[:0]

    trace: list[dict[str, float]] = []
    for epoch in range(1, args.plastic_epochs + 1):
        wrapped.train()
        permutation = torch.randperm(new_inputs.shape[0], generator=generator)
        totals = {"loss": 0.0, "new": 0.0, "bridge": 0.0, "kl": 0.0, "margin": 0.0, "adapter": 0.0}
        batches = 0
        pbar = tqdm(
            range(0, new_inputs.shape[0], args.batch_size),
            desc=f"plastic epoch {epoch}/{args.plastic_epochs}",
        )
        for start in pbar:
            new_indices = permutation[start : start + args.batch_size]
            if new_indices.numel() <= 0:
                raise RuntimeError(f"Empty new batch at start={start}.")
            batch_size = int(new_indices.numel())
            bridge_indices = sample_indices(bridge_inputs.shape[0], batch_size, generator=generator)

            optimizer.zero_grad(set_to_none=True)
            new_logits = wrapped(new_inputs[new_indices].to(device))
            new_loss = F.cross_entropy(new_logits.reshape(-1, new_logits.shape[-1]), new_targets[new_indices].to(device).reshape(-1))
            bridge_logits = wrapped(bridge_inputs[bridge_indices].to(device))
            bridge_loss = F.cross_entropy(
                bridge_logits.reshape(-1, bridge_logits.shape[-1]),
                bridge_targets[bridge_indices].to(device).reshape(-1),
            )
            if probe_count > 0:
                old_batch_indices = sample_indices(probe_count, batch_size, generator=generator)
                old_logits = wrapped(selected_old_inputs[old_batch_indices].to(device))
                kl = distillation_kl(
                    old_logits,
                    selected_teacher_logits[old_batch_indices].to(device),
                    temperature=args.distill_temperature,
                )
                margin = old_margin_loss(
                    old_logits,
                    selected_old_targets[old_batch_indices].to(device),
                    selected_teacher_margins[old_batch_indices].to(device),
                    margin_slack=args.margin_slack,
                )
            else:
                kl = new_loss.new_zeros(())
                margin = new_loss.new_zeros(())
            adapter_penalty = adapter.penalty()
            loss = (
                new_loss
                + args.lambda_bridge * bridge_loss
                + args.lambda_kl * kl
                + args.lambda_margin * margin
                + args.lambda_adapter * adapter_penalty
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_adapter_parameters(adapter), args.grad_clip)
            optimizer.step()

            values = {
                "loss": float(loss.detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "bridge": float(bridge_loss.detach().cpu()),
                "kl": float(kl.detach().cpu()),
                "margin": float(margin.detach().cpu()),
                "adapter": float(adapter_penalty.detach().cpu()),
            }
            for key, value in values.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({key: f"{value:.3g}" for key, value in values.items() if key != "adapter"})
        if batches <= 0:
            raise RuntimeError(f"Plastic epoch {epoch} saw zero batches.")
        row = {key: value / float(batches) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        trace.append(row)
        print(
            "plastic epoch={:4d} loss={:.5f} new={:.5f} bridge={:.5f} kl={:.5f} margin={:.5f}".format(
                epoch,
                row["loss"],
                row["new"],
                row["bridge"],
                row["kl"],
                row["margin"],
            )
        )
    return trace


def train_consolidated_core(
    *,
    args: argparse.Namespace,
    core: GCONativeTransformer,
    adapter_teacher: AdapterWrappedTransformer,
    data: dict[str, torch.Tensor],
    old_probe_indices: torch.Tensor,
    teacher_old_logits: torch.Tensor,
    teacher_old_margins: torch.Tensor,
    device: torch.device,
) -> list[dict[str, float]]:
    set_only_native_weights_trainable(core)
    freeze_model(adapter_teacher)
    adapter_teacher.eval()
    optimizer = make_optimizer(args, trainable_weight_parameters(core))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 202)

    old_inputs = data["old_inputs"]
    old_targets = data["old_targets"]
    new_inputs = data["new_inputs"]
    new_targets = data["new_targets"]
    bridge_inputs = data["bridge_inputs"]
    bridge_targets = data["bridge_targets"]
    full_inputs = data["full_inputs"]
    probe_count = int(old_probe_indices.numel())
    selected_old_inputs = old_inputs[old_probe_indices] if probe_count > 0 else old_inputs[:0]
    selected_old_targets = old_targets[old_probe_indices] if probe_count > 0 else old_targets[:0]
    selected_teacher_logits = teacher_old_logits[old_probe_indices] if probe_count > 0 else teacher_old_logits[:0]
    selected_teacher_margins = teacher_old_margins[old_probe_indices] if probe_count > 0 else teacher_old_margins[:0]

    trace: list[dict[str, float]] = []
    for epoch in range(1, args.consolidation_epochs + 1):
        core.train()
        permutation = torch.randperm(new_inputs.shape[0], generator=generator)
        totals = {"loss": 0.0, "new": 0.0, "bridge": 0.0, "anchor_kl": 0.0, "margin": 0.0, "adapter_kl": 0.0}
        batches = 0
        pbar = tqdm(
            range(0, new_inputs.shape[0], args.batch_size),
            desc=f"consolidate epoch {epoch}/{args.consolidation_epochs}",
        )
        for start in pbar:
            new_indices = permutation[start : start + args.batch_size]
            if new_indices.numel() <= 0:
                raise RuntimeError(f"Empty new batch at start={start}.")
            batch_size = int(new_indices.numel())
            bridge_indices = sample_indices(bridge_inputs.shape[0], batch_size, generator=generator)
            full_indices = sample_indices(full_inputs.shape[0], batch_size, generator=generator)

            batch_new_inputs = new_inputs[new_indices].to(device)
            batch_new_targets = new_targets[new_indices].to(device)
            batch_bridge_inputs = bridge_inputs[bridge_indices].to(device)
            batch_bridge_targets = bridge_targets[bridge_indices].to(device)
            batch_full_inputs = full_inputs[full_indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            new_logits = core(batch_new_inputs)
            bridge_logits = core(batch_bridge_inputs)
            new_loss = F.cross_entropy(new_logits.reshape(-1, new_logits.shape[-1]), batch_new_targets.reshape(-1))
            bridge_loss = F.cross_entropy(bridge_logits.reshape(-1, bridge_logits.shape[-1]), batch_bridge_targets.reshape(-1))
            with torch.no_grad():
                adapter_new_logits = adapter_teacher(batch_new_inputs)
                adapter_bridge_logits = adapter_teacher(batch_bridge_inputs)
                adapter_full_logits = adapter_teacher(batch_full_inputs)
            adapter_kl = (
                distillation_kl(new_logits, adapter_new_logits, temperature=args.distill_temperature)
                + distillation_kl(bridge_logits, adapter_bridge_logits, temperature=args.distill_temperature)
                + distillation_kl(core(batch_full_inputs), adapter_full_logits, temperature=args.distill_temperature)
            ) / 3.0

            if probe_count > 0:
                old_batch_indices = sample_indices(probe_count, batch_size, generator=generator)
                old_logits = core(selected_old_inputs[old_batch_indices].to(device))
                anchor_kl = distillation_kl(
                    old_logits,
                    selected_teacher_logits[old_batch_indices].to(device),
                    temperature=args.distill_temperature,
                )
                margin = old_margin_loss(
                    old_logits,
                    selected_old_targets[old_batch_indices].to(device),
                    selected_teacher_margins[old_batch_indices].to(device),
                    margin_slack=args.margin_slack,
                )
            else:
                anchor_kl = new_loss.new_zeros(())
                margin = new_loss.new_zeros(())
            loss = (
                new_loss
                + args.lambda_bridge * bridge_loss
                + args.lambda_kl * anchor_kl
                + args.lambda_margin * margin
                + args.lambda_adapter_distill * adapter_kl
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(core), args.grad_clip)
            optimizer.step()

            values = {
                "loss": float(loss.detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "bridge": float(bridge_loss.detach().cpu()),
                "anchor_kl": float(anchor_kl.detach().cpu()),
                "margin": float(margin.detach().cpu()),
                "adapter_kl": float(adapter_kl.detach().cpu()),
            }
            for key, value in values.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({key: f"{value:.3g}" for key, value in values.items()})
        if batches <= 0:
            raise RuntimeError(f"Consolidation epoch {epoch} saw zero batches.")
        row = {key: value / float(batches) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        trace.append(row)
        print(
            "consolidate epoch={:4d} loss={:.5f} new={:.5f} bridge={:.5f} anchor_kl={:.5f} adapter_kl={:.5f}".format(
                epoch,
                row["loss"],
                row["new"],
                row["bridge"],
                row["anchor_kl"],
                row["adapter_kl"],
            )
        )
    return trace


def evaluate_all(
    *,
    model: nn.Module,
    data: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    return {
        "old": evaluate_model(model, data["old_inputs"], data["old_targets"], batch_size=batch_size, device=device),
        "new": evaluate_model(model, data["new_inputs"], data["new_targets"], batch_size=batch_size, device=device),
        "bridge": evaluate_model(model, data["bridge_inputs"], data["bridge_targets"], batch_size=batch_size, device=device),
        "full": evaluate_model(model, data["full_inputs"], data["full_targets"], batch_size=batch_size, device=device),
    }


def build_data(args: argparse.Namespace, seq_len: int) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.chunk_index >= len(chunks):
        raise ValueError(f"chunk_index={args.chunk_index} outside chunk count {len(chunks)}.")
    text = str(chunks[args.chunk_index]["text"])
    old_text = word_span(text, args.old_word_start, args.old_word_count)
    new_text = word_span(text, args.new_word_start, args.new_word_count)
    full_text = word_span(text, args.old_word_start, args.old_word_count + args.new_word_count)
    old_token_ids = tokenizer.encode(old_text).ids
    new_token_ids = tokenizer.encode(new_text).ids
    full_token_ids = tokenizer.encode(full_text).ids
    boundary_token_index = len(old_token_ids)
    old_inputs, old_targets = build_lm_windows(
        old_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows
    )
    new_inputs, new_targets = build_lm_windows(
        new_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows
    )
    full_inputs, full_targets = build_lm_windows(
        full_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows
    )
    bridge_inputs, bridge_targets, bridge_starts = build_bridge_lm_windows(
        full_token_ids,
        boundary_token_index=boundary_token_index,
        seq_len=seq_len,
        stride=args.stride,
        max_windows=args.max_bridge_windows,
        radius_tokens=args.bridge_radius_tokens,
    )
    full_geometry_windows, _positions = build_windows(
        full_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.geometry_max_windows
    )
    return {
        "chunks": chunks,
        "old_text": old_text,
        "new_text": new_text,
        "old_inputs": old_inputs,
        "old_targets": old_targets,
        "new_inputs": new_inputs,
        "new_targets": new_targets,
        "full_inputs": full_inputs,
        "full_targets": full_targets,
        "bridge_inputs": bridge_inputs,
        "bridge_targets": bridge_targets,
        "bridge_starts": bridge_starts,
        "full_geometry_windows": full_geometry_windows,
        "old_token_count": len(old_token_ids),
        "new_token_count": len(new_token_ids),
        "full_token_count": len(full_token_ids),
        "boundary_token_index": boundary_token_index,
        "seq_len": seq_len,
    }


def validate_args(args: argparse.Namespace) -> None:
    for name in ["base_checkpoint", "target_checkpoint", "tokenizer_path", "chunks_path"]:
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    nonnegative_int("chunk_index", args.chunk_index)
    nonnegative_int("old_word_start", args.old_word_start)
    positive_int("old_word_count", args.old_word_count)
    nonnegative_int("new_word_start", args.new_word_start)
    positive_int("new_word_count", args.new_word_count)
    if args.old_word_start + args.old_word_count != args.new_word_start:
        raise ValueError(
            "This diagnostic expects adjacent old and new spans: "
            f"old=[{args.old_word_start},{args.old_word_start + args.old_word_count}) "
            f"new_start={args.new_word_start}."
        )
    positive_int("stride", args.stride)
    positive_int("max_windows", args.max_windows)
    positive_int("max_bridge_windows", args.max_bridge_windows)
    nonnegative_int("bridge_radius_tokens", args.bridge_radius_tokens)
    positive_int("geometry_max_windows", args.geometry_max_windows)
    positive_int("old_probe_count", args.old_probe_count)
    positive_int("plastic_epochs", args.plastic_epochs)
    positive_int("consolidation_epochs", args.consolidation_epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("adapter_rank", args.adapter_rank)
    positive_float("adapter_scale", args.adapter_scale)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    nonnegative_float("lambda_kl", args.lambda_kl)
    nonnegative_float("lambda_margin", args.lambda_margin)
    nonnegative_float("lambda_bridge", args.lambda_bridge)
    nonnegative_float("lambda_adapter", args.lambda_adapter)
    nonnegative_float("lambda_adapter_distill", args.lambda_adapter_distill)
    positive_float("distill_temperature", args.distill_temperature)
    nonnegative_float("margin_slack", args.margin_slack)
    positive_float("grad_clip", args.grad_clip)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    base100, base_checkpoint = load_checkpoint(args.base_checkpoint, device)
    base200, target_checkpoint = load_checkpoint(args.target_checkpoint, device)
    if base_checkpoint["model_config"] != target_checkpoint["model_config"]:
        raise ValueError(
            "Base and target checkpoint specs differ. "
            f"base={base_checkpoint['model_config']} target={target_checkpoint['model_config']}"
        )
    freeze_model(base100)
    freeze_model(base200)
    base100.eval()
    base200.eval()

    seq_len = int(base_checkpoint["model_config"]["max_seq_len"])
    d_model = int(base_checkpoint["model_config"]["d_model"])
    data = build_data(args, seq_len)

    teacher_old_logits = collect_logits(base100, data["old_inputs"], batch_size=args.eval_batch_size, device=device)
    teacher_old_margins = target_margins_from_logits(teacher_old_logits, data["old_targets"])
    if args.old_probe_count > data["old_inputs"].shape[0]:
        raise ValueError(
            f"old_probe_count={args.old_probe_count} exceeds old windows={data['old_inputs'].shape[0]}."
        )
    old_probe_indices = select_old_probe_indices(
        count=args.old_probe_count,
        mode=args.old_probe_selection,
        teacher_logits=teacher_old_logits,
        teacher_margins=teacher_old_margins,
        old_targets=data["old_targets"],
        seed=args.seed,
    )
    if int(old_probe_indices.numel()) != args.old_probe_count:
        raise RuntimeError(
            f"Probe selector returned {old_probe_indices.numel()} indices for requested count={args.old_probe_count}."
        )

    target_metrics = evaluate_all(model=base200, data=data, batch_size=args.eval_batch_size, device=device)
    initial_metrics = evaluate_all(model=base100, data=data, batch_size=args.eval_batch_size, device=device)

    adapter_base, _adapter_checkpoint = load_checkpoint(args.base_checkpoint, device)
    freeze_model(adapter_base)
    adapter = FinalResidualAdapter(d_model=d_model, rank=args.adapter_rank, scale=args.adapter_scale).to(device)
    wrapped = AdapterWrappedTransformer(adapter_base, adapter).to(device)

    print("TINY BRIDGE + PLASTIC ADAPTER + CONSOLIDATION")
    print("=" * 112)
    print("Final saved model is a consolidated core. The temporary adapter is not required after consolidation.")
    print(
        f"device={device} old_probes={args.old_probe_count}/{data['old_inputs'].shape[0]} "
        f"selection={args.old_probe_selection} bridge_windows={data['bridge_inputs'].shape[0]} "
        f"adapter_rank={args.adapter_rank} adapter_scale={args.adapter_scale:g}"
    )
    print(
        "initial old={:.5f}/{:.4f} new={:.5f}/{:.4f} bridge={:.5f}/{:.4f} full={:.5f}/{:.4f}".format(
            initial_metrics["old"]["loss"],
            initial_metrics["old"]["token_accuracy"],
            initial_metrics["new"]["loss"],
            initial_metrics["new"]["token_accuracy"],
            initial_metrics["bridge"]["loss"],
            initial_metrics["bridge"]["token_accuracy"],
            initial_metrics["full"]["loss"],
            initial_metrics["full"]["token_accuracy"],
        )
    )
    print(
        "target200 old={:.5f}/{:.4f} new={:.5f}/{:.4f} bridge={:.5f}/{:.4f} full={:.5f}/{:.4f}".format(
            target_metrics["old"]["loss"],
            target_metrics["old"]["token_accuracy"],
            target_metrics["new"]["loss"],
            target_metrics["new"]["token_accuracy"],
            target_metrics["bridge"]["loss"],
            target_metrics["bridge"]["token_accuracy"],
            target_metrics["full"]["loss"],
            target_metrics["full"]["token_accuracy"],
        )
    )

    plastic_trace = train_plastic_adapter(
        args=args,
        wrapped=wrapped,
        adapter=adapter,
        data=data,
        old_probe_indices=old_probe_indices,
        teacher_old_logits=teacher_old_logits,
        teacher_old_margins=teacher_old_margins,
        device=device,
    )
    plastic_metrics = evaluate_all(model=wrapped, data=data, batch_size=args.eval_batch_size, device=device)

    consolidated_core, _consolidation_checkpoint = load_checkpoint(args.base_checkpoint, device)
    consolidation_trace = train_consolidated_core(
        args=args,
        core=consolidated_core,
        adapter_teacher=wrapped,
        data=data,
        old_probe_indices=old_probe_indices,
        teacher_old_logits=teacher_old_logits,
        teacher_old_margins=teacher_old_margins,
        device=device,
    )
    consolidated_metrics = evaluate_all(model=consolidated_core, data=data, batch_size=args.eval_batch_size, device=device)

    final_geometry = geometry_report(
        current=consolidated_core,
        base100=base100,
        base200=base200,
        windows=data["full_geometry_windows"],
        device=device,
    )
    final_geometry_mean = {
        "current_to_base100_drift_rel": mean_layer_metric(final_geometry, "current_to_base100_drift_rel"),
        "current_to_base100_cka": mean_layer_metric(final_geometry, "current_to_base100_cka"),
        "current_to_base200_drift_rel": mean_layer_metric(final_geometry, "current_to_base200_drift_rel"),
        "current_to_base200_cka": mean_layer_metric(final_geometry, "current_to_base200_cka"),
        "base100_to_base200_drift_rel": mean_layer_metric(final_geometry, "base100_to_base200_drift_rel"),
        "base100_to_base200_cka": mean_layer_metric(final_geometry, "base100_to_base200_cka"),
        "base200_closeness_gain": mean_layer_metric(final_geometry, "base200_closeness_gain"),
        "base200_cka_gain": mean_layer_metric(final_geometry, "base200_cka_gain"),
    }
    base100_logits = {
        "old": teacher_old_logits,
    }
    base200_logits = {
        "old": collect_logits(base200, data["old_inputs"], batch_size=args.eval_batch_size, device=device),
        "new": collect_logits(base200, data["new_inputs"], batch_size=args.eval_batch_size, device=device),
        "bridge": collect_logits(base200, data["bridge_inputs"], batch_size=args.eval_batch_size, device=device),
        "full": collect_logits(base200, data["full_inputs"], batch_size=args.eval_batch_size, device=device),
    }
    final_logits = {
        "old": collect_logits(consolidated_core, data["old_inputs"], batch_size=args.eval_batch_size, device=device),
        "new": collect_logits(consolidated_core, data["new_inputs"], batch_size=args.eval_batch_size, device=device),
        "bridge": collect_logits(consolidated_core, data["bridge_inputs"], batch_size=args.eval_batch_size, device=device),
        "full": collect_logits(consolidated_core, data["full_inputs"], batch_size=args.eval_batch_size, device=device),
    }
    behavior_match = {
        "old_to_base100": output_kl_and_agreement(
            current_logits=final_logits["old"],
            teacher_logits=base100_logits["old"],
            temperature=args.distill_temperature,
        ),
        "old_to_base200": output_kl_and_agreement(
            current_logits=final_logits["old"],
            teacher_logits=base200_logits["old"],
            temperature=args.distill_temperature,
        ),
        "new_to_base200": output_kl_and_agreement(
            current_logits=final_logits["new"],
            teacher_logits=base200_logits["new"],
            temperature=args.distill_temperature,
        ),
        "bridge_to_base200": output_kl_and_agreement(
            current_logits=final_logits["bridge"],
            teacher_logits=base200_logits["bridge"],
            temperature=args.distill_temperature,
        ),
        "full_to_base200": output_kl_and_agreement(
            current_logits=final_logits["full"],
            teacher_logits=base200_logits["full"],
            temperature=args.distill_temperature,
        ),
    }

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "base_checkpoint": str(args.base_checkpoint),
            "target_checkpoint": str(args.target_checkpoint),
            "model_state_dict": consolidated_core.state_dict(),
            "native_gco_config": asdict(NativeGCOConfig(**base_checkpoint["native_gco_config"])),
            "model_config": base_checkpoint["model_config"],
            "adapter_state_dict": adapter.state_dict(),
            "source": {
                "chunks_path": str(args.chunks_path),
                "chunk_index": args.chunk_index,
                "chunk_id": str(data["chunks"][args.chunk_index]["chunk_id"]),
                "old_word_start": args.old_word_start,
                "old_word_count": args.old_word_count,
                "new_word_start": args.new_word_start,
                "new_word_count": args.new_word_count,
                "old_probe_count": args.old_probe_count,
                "old_probe_indices": old_probe_indices.tolist(),
                "bridge_starts": data["bridge_starts"].tolist(),
                "boundary_token_index": data["boundary_token_index"],
            },
            "metrics": {
                "initial": initial_metrics,
                "target200": target_metrics,
                "plastic": plastic_metrics,
                "consolidated": consolidated_metrics,
                "final_geometry_mean": final_geometry_mean,
                "behavior_match": behavior_match,
            },
        },
        args.output_checkpoint,
    )
    result = {
        "question": "Can bridge windows plus a temporary plastic adapter improve the behavior-preserving CL path?",
        "base_checkpoint": str(args.base_checkpoint),
        "target_checkpoint": str(args.target_checkpoint),
        "output_checkpoint": str(args.output_checkpoint),
        "model_config": base_checkpoint["model_config"],
        "source": {
            "chunks_path": str(args.chunks_path),
            "chunk_index": args.chunk_index,
            "chunk_id": str(data["chunks"][args.chunk_index]["chunk_id"]),
            "old_word_start": args.old_word_start,
            "old_word_count": args.old_word_count,
            "new_word_start": args.new_word_start,
            "new_word_count": args.new_word_count,
            "old_probe_count": args.old_probe_count,
            "old_probe_selection": args.old_probe_selection,
            "old_probe_indices": old_probe_indices.tolist(),
            "old_window_count": int(data["old_inputs"].shape[0]),
            "new_window_count": int(data["new_inputs"].shape[0]),
            "bridge_window_count": int(data["bridge_inputs"].shape[0]),
            "full_window_count": int(data["full_inputs"].shape[0]),
            "bridge_starts": data["bridge_starts"].tolist(),
            "boundary_token_index": data["boundary_token_index"],
            "seq_len": seq_len,
            "stride": args.stride,
        },
        "optimizer": {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "grad_clip": args.grad_clip,
        },
        "loss_weights": {
            "lambda_kl": args.lambda_kl,
            "lambda_margin": args.lambda_margin,
            "lambda_bridge": args.lambda_bridge,
            "lambda_adapter": args.lambda_adapter,
            "lambda_adapter_distill": args.lambda_adapter_distill,
            "distill_temperature": args.distill_temperature,
            "margin_slack": args.margin_slack,
        },
        "adapter": {
            "rank": args.adapter_rank,
            "scale": args.adapter_scale,
            "plastic_epochs": args.plastic_epochs,
            "consolidation_epochs": args.consolidation_epochs,
        },
        "metrics": {
            "initial": initial_metrics,
            "target200": target_metrics,
            "plastic": plastic_metrics,
            "consolidated": consolidated_metrics,
            "final_geometry": final_geometry,
            "final_geometry_mean": final_geometry_mean,
            "behavior_match": behavior_match,
        },
        "plastic_trace": plastic_trace,
        "consolidation_trace": consolidation_trace,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nTINY BRIDGE ADAPTER CONSOLIDATION SUMMARY")
    print("=" * 112)
    for label, metrics in [("initial", initial_metrics), ("plastic", plastic_metrics), ("consolidated", consolidated_metrics), ("target200", target_metrics)]:
        print(
            "{:<12} old={:.5f}/{:.4f} new={:.5f}/{:.4f} bridge={:.5f}/{:.4f} full={:.5f}/{:.4f}".format(
                label,
                metrics["old"]["loss"],
                metrics["old"]["token_accuracy"],
                metrics["new"]["loss"],
                metrics["new"]["token_accuracy"],
                metrics["bridge"]["loss"],
                metrics["bridge"]["token_accuracy"],
                metrics["full"]["loss"],
                metrics["full"]["token_accuracy"],
            )
        )
    print(
        "geometry to_base200 rel={:.4f} cka={:.4f} closeness_gain={:+.4f} cka_gain={:+.4f}".format(
            final_geometry_mean["current_to_base200_drift_rel"],
            final_geometry_mean["current_to_base200_cka"],
            final_geometry_mean["base200_closeness_gain"],
            final_geometry_mean["base200_cka_gain"],
        )
    )
    print(
        "KL old_to_base100={:.5f} new_to_base200={:.5f} bridge_to_base200={:.5f} full_to_base200={:.5f}".format(
            behavior_match["old_to_base100"]["kl"],
            behavior_match["new_to_base200"]["kl"],
            behavior_match["bridge_to_base200"]["kl"],
            behavior_match["full_to_base200"]["kl"],
        )
    )
    print(f"wrote_checkpoint={args.output_checkpoint}")
    print(f"wrote_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        default=Path("model/checkpoints/gco-tiny-cl-base-200w-samespec-seed0.pt"),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("model/checkpoints/gco-tiny-cl-bridge-adapter-consolidated-100to200-seed0.pt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("model/analysis/gco-tiny-cl-bridge-adapter-consolidated-100to200-seed0.json"),
    )
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--old-word-start", type=int, default=0)
    parser.add_argument("--old-word-count", type=int, default=100)
    parser.add_argument("--new-word-start", type=int, default=100)
    parser.add_argument("--new-word-count", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--max-bridge-windows", type=int, default=128)
    parser.add_argument("--bridge-radius-tokens", type=int, default=64)
    parser.add_argument("--geometry-max-windows", type=int, default=128)
    parser.add_argument("--old-probe-count", type=int, default=64)
    parser.add_argument(
        "--old-probe-selection",
        choices=["uniform", "random", "low-margin", "high-loss"],
        default="uniform",
    )
    parser.add_argument("--plastic-epochs", type=int, default=200)
    parser.add_argument("--consolidation-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-margin", type=float, default=0.1)
    parser.add_argument("--lambda-bridge", type=float, default=1.0)
    parser.add_argument("--lambda-adapter", type=float, default=1e-4)
    parser.add_argument("--lambda-adapter-distill", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--margin-slack", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
