#!/usr/bin/env python3
"""Math-first tests for the Geometric Continual Optimizer.

This script intentionally avoids memory slots, external retrieval systems, and
task-specific controllers. It tests the central GCO claim on a small two-layer
linear network:

1. Does predicted activation drift correlate with actual forgetting?
2. Does projecting the gradient away from protected anchor Jacobians reduce
   forgetting while retaining usable gradient for the new task?
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class Task:
    basis: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class Params:
    w1: torch.Tensor
    w2: torch.Tensor


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor
    y: torch.Tensor


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown device: {name}")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        if device.type == "mps":
            raise RuntimeError("MPS does not support float64 for this experiment. Use --dtype float32.")
        return torch.float64
    raise ValueError(f"Unknown dtype: {name}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def randn(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=dtype)


def orthonormal_matrix(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q, r = torch.linalg.qr(randn((n, n), device=device, dtype=dtype))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def make_tasks(
    input_dim: int,
    output_dim: int,
    rank: int,
    overlap: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Task, Task]:
    if rank > input_dim // 2 and overlap < 1.0:
        raise ValueError("--rank must be <= input_dim / 2 when overlap < 1.0.")
    if not (0.0 <= overlap <= 1.0):
        raise ValueError("--overlap must be in [0, 1].")

    full_basis = orthonormal_matrix(input_dim, device=device, dtype=dtype)
    basis_1 = full_basis[:, :rank]

    shared = int(round(overlap * rank))
    shared = max(0, min(rank, shared))
    unique = rank - shared

    if unique == 0:
        basis_2 = basis_1.clone()
    else:
        basis_2 = torch.cat(
            [basis_1[:, :shared], full_basis[:, rank : rank + unique]],
            dim=1,
        )
        basis_2, _ = torch.linalg.qr(basis_2)

    target_1 = randn((rank, output_dim), device=device, dtype=dtype) / math.sqrt(rank)
    target_2 = randn((rank, output_dim), device=device, dtype=dtype) / math.sqrt(rank)
    return Task(basis_1, target_1), Task(basis_2, target_2)


def sample_task(
    task: Task,
    n: int,
    *,
    input_noise: float,
    output_noise: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Batch:
    rank = task.basis.shape[1]
    coeff = randn((n, rank), device=device, dtype=dtype)
    x = coeff @ task.basis.T
    if input_noise > 0:
        x = x + input_noise * randn(x.shape, device=device, dtype=dtype)
    y = coeff @ task.target
    if output_noise > 0:
        y = y + output_noise * randn(y.shape, device=device, dtype=dtype)
    return Batch(x=x, y=y)


def init_params(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Params:
    w1 = randn((hidden_dim, input_dim), device=device, dtype=dtype) / math.sqrt(input_dim)
    w2 = randn((output_dim, hidden_dim), device=device, dtype=dtype) / math.sqrt(hidden_dim)
    return Params(w1=w1, w2=w2)


def clone_params(params: Params) -> Params:
    return Params(w1=params.w1.clone(), w2=params.w2.clone())


def forward(params: Params, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden = x @ params.w1.T
    output = hidden @ params.w2.T
    return output, hidden


def per_sample_mse(params: Params, batch: Batch) -> torch.Tensor:
    pred, _ = forward(params, batch.x)
    return ((pred - batch.y) ** 2).mean(dim=1)


def mse(params: Params, batch: Batch) -> float:
    return float(per_sample_mse(params, batch).mean().detach().cpu())


def gradients(params: Params, batch: Batch, *, train_output: bool) -> Params:
    pred, hidden = forward(params, batch.x)
    err = 2.0 * (pred - batch.y) / pred.numel()
    grad_w2 = err.T @ hidden
    grad_hidden = err @ params.w2
    grad_w1 = grad_hidden.T @ batch.x
    if not train_output:
        grad_w2 = torch.zeros_like(grad_w2)
    return Params(w1=grad_w1, w2=grad_w2)


def add_params(params: Params, delta: Params) -> Params:
    return Params(w1=params.w1 + delta.w1, w2=params.w2 + delta.w2)


def scale_params(params: Params, scalar: float) -> Params:
    return Params(w1=params.w1 * scalar, w2=params.w2 * scalar)


def flatten(params: Params) -> torch.Tensor:
    return torch.cat([params.w1.reshape(-1), params.w2.reshape(-1)])


def unflatten(vector: torch.Tensor, template: Params) -> Params:
    w1_numel = template.w1.numel()
    w1 = vector[:w1_numel].reshape_as(template.w1)
    w2 = vector[w1_numel:].reshape_as(template.w2)
    return Params(w1=w1, w2=w2)


def l2_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector).detach().cpu())


def train_task1(
    params: Params,
    batch: Batch,
    *,
    steps: int,
    lr: float,
    train_output: bool,
) -> Params:
    current = clone_params(params)
    for _ in range(steps):
        grad = gradients(current, batch, train_output=train_output)
        current = add_params(current, scale_params(grad, -lr))
    return current


def anchor_jacobian_rows(
    params: Params,
    anchor_x: torch.Tensor,
    *,
    layer: str,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if layer not in {"hidden", "output", "both"}:
        raise ValueError(f"Unknown protected layer: {layer}")

    rows: List[torch.Tensor] = []
    n = anchor_x.shape[0]
    input_dim = params.w1.shape[1]
    hidden_dim = params.w1.shape[0]
    output_dim = params.w2.shape[0]
    p = params.w1.numel() + params.w2.numel()

    if weights is None:
        weights = torch.ones(n, device=anchor_x.device, dtype=anchor_x.dtype)

    for idx in range(n):
        q = anchor_x[idx]
        weight = torch.sqrt(weights[idx])
        if layer in {"hidden", "both"}:
            for h_idx in range(hidden_dim):
                row = torch.zeros(p, device=anchor_x.device, dtype=anchor_x.dtype)
                start = h_idx * input_dim
                row[start : start + input_dim] = q
                rows.append(weight * row)

        if layer in {"output", "both"}:
            hidden = params.w1 @ q
            for out_idx in range(output_dim):
                row = torch.zeros(p, device=anchor_x.device, dtype=anchor_x.dtype)
                w1_part = torch.outer(params.w2[out_idx], q).reshape(-1)
                w2_part = torch.zeros_like(params.w2).reshape(-1)
                start = out_idx * hidden_dim
                w2_part[start : start + hidden_dim] = hidden
                row[: params.w1.numel()] = w1_part
                row[params.w1.numel() :] = w2_part
                rows.append(weight * row)

    if not rows:
        return torch.zeros((0, p), device=anchor_x.device, dtype=anchor_x.dtype)
    return torch.stack(rows, dim=0)


def project_gradient(grad_flat: torch.Tensor, a: torch.Tensor, damping: float) -> torch.Tensor:
    if a.shape[0] == 0:
        return grad_flat
    eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    rhs = a @ grad_flat
    coeff = torch.linalg.solve(a @ a.T + damping * eye, rhs)
    return grad_flat - a.T @ coeff


def linearized_hidden_drift(params: Params, anchor_x: torch.Tensor, delta: Params) -> torch.Tensor:
    del params
    drift = anchor_x @ delta.w1.T
    return (drift**2).sum(dim=1)


def linearized_output_drift(params: Params, anchor_x: torch.Tensor, delta: Params) -> torch.Tensor:
    hidden = anchor_x @ params.w1.T
    delta_hidden = anchor_x @ delta.w1.T
    drift = delta_hidden @ params.w2.T + hidden @ delta.w2.T
    return (drift**2).sum(dim=1)


def actual_output_change(params: Params, anchor_x: torch.Tensor, delta: Params) -> torch.Tensor:
    before, _ = forward(params, anchor_x)
    after, _ = forward(add_params(params, delta), anchor_x)
    return ((after - before) ** 2).sum(dim=1)


def positive_loss_delta(params: Params, anchor_batch: Batch, delta: Params) -> torch.Tensor:
    before = per_sample_mse(params, anchor_batch)
    after = per_sample_mse(add_params(params, delta), anchor_batch)
    return torch.clamp(after - before, min=0.0)


def rank_1d(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float64)
    return ranks


def correlation(x_values: Sequence[float], y_values: Sequence[float], *, rank: bool) -> Optional[float]:
    if len(x_values) < 3:
        return None
    x = torch.tensor(x_values, dtype=torch.float64)
    y = torch.tensor(y_values, dtype=torch.float64)
    if rank:
        x = rank_1d(x)
        y = rank_1d(y)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return None
    return float((x @ y / denom).item())


def select_anchor_indices(
    risk: torch.Tensor,
    *,
    tau_protect: float,
    protect_top_k: int,
) -> torch.Tensor:
    eligible = torch.nonzero(risk > tau_protect, as_tuple=False).reshape(-1)
    if protect_top_k <= 0 or eligible.numel() <= protect_top_k:
        return eligible
    values = risk[eligible]
    top_local = torch.topk(values, k=protect_top_k).indices
    return eligible[top_local]


def stage1_drift_probe(
    params: Params,
    task2: Task,
    anchor_batch: Batch,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    input_noise: float,
    output_noise: float,
    train_output: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Optional[float]]:
    hidden_pred: List[float] = []
    output_pred: List[float] = []
    actual_forgetting: List[float] = []
    actual_output: List[float] = []

    for _ in range(steps):
        batch = sample_task(
            task2,
            batch_size,
            input_noise=input_noise,
            output_noise=output_noise,
            device=device,
            dtype=dtype,
        )
        grad = gradients(params, batch, train_output=train_output)
        delta = scale_params(grad, -lr)

        h_drift = linearized_hidden_drift(params, anchor_batch.x, delta)
        y_drift = linearized_output_drift(params, anchor_batch.x, delta)
        loss_delta = positive_loss_delta(params, anchor_batch, delta)
        output_change = actual_output_change(params, anchor_batch.x, delta)

        hidden_pred.extend(float(v) for v in h_drift.detach().cpu())
        output_pred.extend(float(v) for v in y_drift.detach().cpu())
        actual_forgetting.extend(float(v) for v in loss_delta.detach().cpu())
        actual_output.extend(float(v) for v in output_change.detach().cpu())

    return {
        "hidden_vs_forgetting_pearson": correlation(hidden_pred, actual_forgetting, rank=False),
        "hidden_vs_forgetting_spearman": correlation(hidden_pred, actual_forgetting, rank=True),
        "output_vs_forgetting_pearson": correlation(output_pred, actual_forgetting, rank=False),
        "output_vs_forgetting_spearman": correlation(output_pred, actual_forgetting, rank=True),
        "hidden_vs_output_change_pearson": correlation(hidden_pred, actual_output, rank=False),
        "output_pred_vs_output_change_pearson": correlation(output_pred, actual_output, rank=False),
    }


def method_protect_layer(method: str) -> Optional[str]:
    if method == "sgd":
        return None
    if method == "gco_hidden":
        return "hidden"
    if method == "gco_output":
        return "output"
    if method == "gco_both":
        return "both"
    raise ValueError(f"Unknown method: {method}")


def train_task2_method(
    initial: Params,
    task2: Task,
    anchor_batch: Batch,
    task1_eval: Batch,
    task2_eval: Batch,
    *,
    method: str,
    steps: int,
    batch_size: int,
    lr: float,
    damping: float,
    tau_protect: float,
    protect_top_k: int,
    input_noise: float,
    output_noise: float,
    train_output: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    current = clone_params(initial)
    layer = method_protect_layer(method)
    ratios: List[float] = []
    protected_counts: List[int] = []

    task1_before = mse(current, task1_eval)
    task2_before = mse(current, task2_eval)

    for _ in range(steps):
        batch = sample_task(
            task2,
            batch_size,
            input_noise=input_noise,
            output_noise=output_noise,
            device=device,
            dtype=dtype,
        )
        grad = gradients(current, batch, train_output=train_output)
        grad_flat = flatten(grad)
        if layer is None:
            update_flat = -lr * grad_flat
            ratios.append(1.0)
            protected_counts.append(0)
        else:
            raw_delta = scale_params(grad, -lr)
            raw_delta_flat = flatten(raw_delta)
            if layer == "hidden":
                risk = linearized_hidden_drift(current, anchor_batch.x, raw_delta)
            elif layer == "output":
                risk = linearized_output_drift(current, anchor_batch.x, raw_delta)
            elif layer == "both":
                risk = linearized_hidden_drift(current, anchor_batch.x, raw_delta)
                risk = risk + linearized_output_drift(current, anchor_batch.x, raw_delta)
            else:
                raise AssertionError("unreachable")

            selected = select_anchor_indices(
                risk,
                tau_protect=tau_protect,
                protect_top_k=protect_top_k,
            )
            selected_x = anchor_batch.x[selected]
            a = anchor_jacobian_rows(current, selected_x, layer=layer)
            safe_grad = project_gradient(grad_flat, a, damping=damping)
            update_flat = -lr * safe_grad
            grad_norm = torch.linalg.vector_norm(grad_flat)
            ratio = torch.linalg.vector_norm(safe_grad) / (grad_norm + 1e-12)
            ratios.append(float(ratio.detach().cpu()))
            protected_counts.append(int(selected.numel()))

            if not torch.isfinite(update_flat).all():
                raise FloatingPointError(f"Non-finite update produced by {method}.")
            if not torch.isfinite(raw_delta_flat).all():
                raise FloatingPointError("Non-finite raw delta produced.")

        current = add_params(current, unflatten(update_flat, current))

    task1_after = mse(current, task1_eval)
    task2_after = mse(current, task2_eval)
    return {
        "task1_before": task1_before,
        "task1_after": task1_after,
        "task2_before": task2_before,
        "task2_after": task2_after,
        "forgetting_delta": task1_after - task1_before,
        "task2_gain": task2_before - task2_after,
        "gradient_ratio_mean": float(sum(ratios) / len(ratios)),
        "protected_anchors_mean": float(sum(protected_counts) / len(protected_counts)),
    }


def run_seed(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    set_seed(seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    task1, task2 = make_tasks(
        args.input_dim,
        args.output_dim,
        args.rank,
        args.overlap,
        device=device,
        dtype=dtype,
    )
    initial = init_params(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        device=device,
        dtype=dtype,
    )
    task1_train = sample_task(
        task1,
        args.task1_train_size,
        input_noise=args.input_noise,
        output_noise=args.output_noise,
        device=device,
        dtype=dtype,
    )
    task1_eval = sample_task(
        task1,
        args.eval_size,
        input_noise=args.input_noise,
        output_noise=0.0,
        device=device,
        dtype=dtype,
    )
    task2_eval = sample_task(
        task2,
        args.eval_size,
        input_noise=args.input_noise,
        output_noise=0.0,
        device=device,
        dtype=dtype,
    )

    task1_model = train_task1(
        initial,
        task1_train,
        steps=args.task1_steps,
        lr=args.task1_lr,
        train_output=not args.freeze_output,
    )
    anchor_source = sample_task(
        task1,
        args.num_anchors,
        input_noise=args.input_noise,
        output_noise=0.0,
        device=device,
        dtype=dtype,
    )

    drift_probe = stage1_drift_probe(
        task1_model,
        task2,
        anchor_source,
        steps=args.probe_steps,
        batch_size=args.batch_size,
        lr=args.task2_lr,
        input_noise=args.input_noise,
        output_noise=args.output_noise,
        train_output=not args.freeze_output,
        device=device,
        dtype=dtype,
    )

    methods = {}
    for method in ("sgd", "gco_hidden", "gco_output", "gco_both"):
        methods[method] = train_task2_method(
            task1_model,
            task2,
            anchor_source,
            task1_eval,
            task2_eval,
            method=method,
            steps=args.task2_steps,
            batch_size=args.batch_size,
            lr=args.task2_lr,
            damping=args.damping,
            tau_protect=args.tau_protect,
            protect_top_k=args.protect_top_k,
            input_noise=args.input_noise,
            output_noise=args.output_noise,
            train_output=not args.freeze_output,
            device=device,
            dtype=dtype,
        )

    return {
        "seed": seed,
        "task1_loss_after_pretrain": mse(task1_model, task1_eval),
        "task2_loss_before_continual": mse(task1_model, task2_eval),
        "drift_probe": drift_probe,
        "methods": methods,
    }


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))


def aggregate(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    drift_keys = list(reports[0]["drift_probe"].keys())  # type: ignore[index, union-attr]
    method_names = list(reports[0]["methods"].keys())  # type: ignore[index, union-attr]

    drift_summary = {}
    for key in drift_keys:
        drift_summary[key] = {
            "mean": mean_std(report["drift_probe"][key] for report in reports)[0],  # type: ignore[index]
            "std": mean_std(report["drift_probe"][key] for report in reports)[1],  # type: ignore[index]
        }

    method_summary = {}
    for method in method_names:
        metric_names = list(reports[0]["methods"][method].keys())  # type: ignore[index]
        method_summary[method] = {}
        for metric in metric_names:
            mean, std = mean_std(report["methods"][method][metric] for report in reports)  # type: ignore[index]
            method_summary[method][metric] = {"mean": mean, "std": std}

    return {
        "drift_probe": drift_summary,
        "methods": method_summary,
    }


def fmt_mean_std(stats: Dict[str, float]) -> str:
    return f"{stats['mean']:.4f} +/- {stats['std']:.4f}"


def print_summary(report: Dict[str, object]) -> None:
    summary = report["summary"]  # type: ignore[index]
    print("\nGCO LINEAR STAGE 1/2 SUMMARY")
    print("=" * 116)
    print(
        f"seeds={report['seed_count']} input_dim={report['config']['input_dim']} "
        f"hidden_dim={report['config']['hidden_dim']} output_dim={report['config']['output_dim']} "
        f"rank={report['config']['rank']} overlap={report['config']['overlap']}"
    )
    print("-" * 116)
    print("Stage 1: predicted drift vs actual forgetting")
    print("-" * 116)
    for key, stats in summary["drift_probe"].items():  # type: ignore[index, union-attr]
        print(f"{key:38s} {fmt_mean_std(stats)}")
    print("-" * 116)
    print("Stage 2: sequential task-2 learning after task-1 pretrain")
    print("-" * 116)
    header = (
        f"{'method':16s} {'task1_after':>18s} {'task2_after':>18s} "
        f"{'forgetting':>18s} {'task2_gain':>18s} {'grad_ratio':>18s}"
    )
    print(header)
    print("-" * 116)
    for method, stats in summary["methods"].items():  # type: ignore[index, union-attr]
        print(
            f"{method:16s} "
            f"{fmt_mean_std(stats['task1_after']):>18s} "
            f"{fmt_mean_std(stats['task2_after']):>18s} "
            f"{fmt_mean_std(stats['forgetting_delta']):>18s} "
            f"{fmt_mean_std(stats['task2_gain']):>18s} "
            f"{fmt_mean_std(stats['gradient_ratio_mean']):>18s}"
        )
    print("=" * 116)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")

    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--overlap", type=float, default=0.5)

    parser.add_argument("--task1-train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--num-anchors", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-noise", type=float, default=0.0)
    parser.add_argument("--output-noise", type=float, default=0.0)

    parser.add_argument("--task1-steps", type=int, default=1200)
    parser.add_argument("--task1-lr", type=float, default=0.05)
    parser.add_argument("--task2-steps", type=int, default=250)
    parser.add_argument("--task2-lr", type=float, default=0.02)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--freeze-output", action="store_true")

    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--tau-protect", type=float, default=0.0)
    parser.add_argument("--protect-top-k", type=int, default=0)

    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def args_to_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "rank": args.rank,
        "overlap": args.overlap,
        "task1_train_size": args.task1_train_size,
        "eval_size": args.eval_size,
        "num_anchors": args.num_anchors,
        "batch_size": args.batch_size,
        "input_noise": args.input_noise,
        "output_noise": args.output_noise,
        "task1_steps": args.task1_steps,
        "task1_lr": args.task1_lr,
        "task2_steps": args.task2_steps,
        "task2_lr": args.task2_lr,
        "probe_steps": args.probe_steps,
        "freeze_output": args.freeze_output,
        "damping": args.damping,
        "tau_protect": args.tau_protect,
        "protect_top_k": args.protect_top_k,
        "device": args.device,
        "dtype": args.dtype,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    if args.num_anchors <= 0:
        raise ValueError("--num-anchors must be positive.")
    if args.damping <= 0:
        raise ValueError("--damping must be positive.")

    reports = []
    for i in range(args.seed_count):
        seed = args.seed_offset + i
        print(f"running_seed={seed}")
        reports.append(run_seed(args, seed))

    report = {
        "experiment": "gco_linear_stage1_stage2",
        "seed_count": args.seed_count,
        "config": args_to_config(args),
        "summary": aggregate(reports),
        "seeds": reports,
    }
    print_summary(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
