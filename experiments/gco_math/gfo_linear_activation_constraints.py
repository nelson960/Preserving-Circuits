#!/usr/bin/env python3
"""Linear GFO experiment with activation-representation constraints.

This is the clean mathematical version of the first GFO tests. It treats a
two-layer linear neural network as the model and protects a chosen hidden-state
representation:

    z_i = Psi_l(h_l(q_i; theta_store))

At each continual update it computes:

    raw update:       delta_raw = -eta grad L_new
    predicted drift:  ||r_i(theta) + J_i delta_raw||^2
    protected set:    anchors whose weighted predicted violation is high
    safe update:      projection of delta_raw onto activation constraints

No external memory slots, no symbolic controller, no task-specific write rules.
The only memory is a bank of activation anchors.
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
class Batch:
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class ModelShape:
    input_dim: int
    hidden_dim: int
    output_dim: int

    @property
    def w1_numel(self) -> int:
        return self.hidden_dim * self.input_dim

    @property
    def w2_numel(self) -> int:
        return self.output_dim * self.hidden_dim

    @property
    def numel(self) -> int:
        return self.w1_numel + self.w2_numel


@dataclass(frozen=True)
class AnchorBank:
    x: torch.Tensor
    hidden_z: torch.Tensor
    output_z: torch.Tensor
    weights: torch.Tensor
    eps: torch.Tensor


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
    if rank <= 0:
        raise ValueError("--rank must be positive.")
    if rank > input_dim:
        raise ValueError("--rank must be <= --input-dim.")
    if rank > input_dim // 2 and overlap < 1.0:
        raise ValueError("--rank must be <= input_dim / 2 when --overlap < 1.")
    if not (0.0 <= overlap <= 1.0):
        raise ValueError("--overlap must be in [0, 1].")

    full_basis = orthonormal_matrix(input_dim, device=device, dtype=dtype)
    basis_1 = full_basis[:, :rank]

    shared = max(0, min(rank, int(round(overlap * rank))))
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
    coeff = randn((n, task.basis.shape[1]), device=device, dtype=dtype)
    x = coeff @ task.basis.T
    if input_noise > 0:
        x = x + input_noise * randn(x.shape, device=device, dtype=dtype)
    y = coeff @ task.target
    if output_noise > 0:
        y = y + output_noise * randn(y.shape, device=device, dtype=dtype)
    return Batch(x=x, y=y)


def init_flat(shape: ModelShape, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w1 = randn((shape.hidden_dim, shape.input_dim), device=device, dtype=dtype) / math.sqrt(shape.input_dim)
    w2 = randn((shape.output_dim, shape.hidden_dim), device=device, dtype=dtype) / math.sqrt(shape.hidden_dim)
    return torch.cat([w1.reshape(-1), w2.reshape(-1)])


def unpack(theta: torch.Tensor, shape: ModelShape) -> Tuple[torch.Tensor, torch.Tensor]:
    if theta.numel() != shape.numel:
        raise ValueError(f"theta has {theta.numel()} elements, expected {shape.numel}.")
    w1 = theta[: shape.w1_numel].reshape(shape.hidden_dim, shape.input_dim)
    w2 = theta[shape.w1_numel :].reshape(shape.output_dim, shape.hidden_dim)
    return w1, w2


def forward(theta: torch.Tensor, shape: ModelShape, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w1, w2 = unpack(theta, shape)
    hidden = x @ w1.T
    output = hidden @ w2.T
    return output, hidden


def normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + eps)


def anchor_representation(
    h: torch.Tensor,
    *,
    mode: str,
    norm_scale: float,
    eps: float,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    if mode == "direction":
        return h / (norm + eps)
    if mode == "norm":
        return torch.log(norm + eps)
    if mode == "full":
        return h
    if mode == "direction_norm":
        direction = h / (norm + eps)
        log_norm = norm_scale * torch.log(norm + eps)
        return torch.cat([direction, log_norm], dim=-1)
    raise ValueError(f"Unknown anchor representation: {mode}")


def layer_representation(
    theta: torch.Tensor,
    shape: ModelShape,
    x: torch.Tensor,
    layer: str,
    eps: float,
    mode: str,
    norm_scale: float,
) -> torch.Tensor:
    output, hidden = forward(theta, shape, x)
    if layer == "hidden":
        return anchor_representation(hidden, mode=mode, norm_scale=norm_scale, eps=eps)
    if layer == "output":
        return anchor_representation(output, mode=mode, norm_scale=norm_scale, eps=eps)
    raise ValueError(f"Unknown layer: {layer}")


def mse(theta: torch.Tensor, shape: ModelShape, batch: Batch) -> torch.Tensor:
    pred, _ = forward(theta, shape, batch.x)
    return ((pred - batch.y) ** 2).mean()


def per_sample_mse(theta: torch.Tensor, shape: ModelShape, batch: Batch) -> torch.Tensor:
    pred, _ = forward(theta, shape, batch.x)
    return ((pred - batch.y) ** 2).mean(dim=1)


def loss_grad(theta: torch.Tensor, shape: ModelShape, batch: Batch, *, freeze_output: bool) -> torch.Tensor:
    theta_var = theta.detach().clone().requires_grad_(True)
    loss = mse(theta_var, shape, batch)
    grad = torch.autograd.grad(loss, theta_var)[0]
    if freeze_output:
        grad = grad.clone()
        grad[shape.w1_numel :] = 0.0
    if not torch.isfinite(grad).all():
        raise FloatingPointError("Non-finite gradient.")
    return grad.detach()


def train_task1(
    theta: torch.Tensor,
    shape: ModelShape,
    batch: Batch,
    *,
    steps: int,
    lr: float,
    freeze_output: bool,
) -> torch.Tensor:
    current = theta.detach().clone()
    for _ in range(steps):
        grad = loss_grad(current, shape, batch, freeze_output=freeze_output)
        current = current - lr * grad
    return current.detach()


def build_anchor_bank(
    theta: torch.Tensor,
    shape: ModelShape,
    anchor_batch: Batch,
    *,
    anchor_eps: float,
    anchor_representation_mode: str,
    norm_scale: float,
    drift_tolerance: float,
    importance: float,
) -> AnchorBank:
    if not (0.0 < importance <= 1.0):
        raise ValueError("--anchor-importance must be in (0, 1].")
    if drift_tolerance <= 0:
        raise ValueError("--drift-tolerance must be positive.")
    return AnchorBank(
        x=anchor_batch.x.detach().clone(),
        hidden_z=layer_representation(
            theta,
            shape,
            anchor_batch.x,
            "hidden",
            anchor_eps,
            anchor_representation_mode,
            norm_scale,
        )
        .detach()
        .clone(),
        output_z=layer_representation(
            theta,
            shape,
            anchor_batch.x,
            "output",
            anchor_eps,
            anchor_representation_mode,
            norm_scale,
        )
        .detach()
        .clone(),
        weights=torch.full(
            (anchor_batch.x.shape[0],),
            importance,
            device=anchor_batch.x.device,
            dtype=anchor_batch.x.dtype,
        ),
        eps=torch.full(
            (anchor_batch.x.shape[0],),
            drift_tolerance,
            device=anchor_batch.x.device,
            dtype=anchor_batch.x.dtype,
        ),
    )


def layer_anchor_z(bank: AnchorBank, layer: str) -> torch.Tensor:
    if layer == "hidden":
        return bank.hidden_z
    if layer == "output":
        return bank.output_z
    raise ValueError(f"Unknown layer: {layer}")


def anchor_jacobian(
    theta: torch.Tensor,
    shape: ModelShape,
    q: torch.Tensor,
    layer: str,
    anchor_eps: float,
    anchor_representation_mode: str,
    norm_scale: float,
) -> torch.Tensor:
    q_batched = q.unsqueeze(0)

    def fn(theta_in: torch.Tensor) -> torch.Tensor:
        return layer_representation(
            theta_in,
            shape,
            q_batched,
            layer,
            anchor_eps,
            anchor_representation_mode,
            norm_scale,
        ).reshape(-1)

    return torch.autograd.functional.jacobian(fn, theta, create_graph=False, strict=True).detach()


def all_anchor_jacobians(
    theta: torch.Tensor,
    shape: ModelShape,
    bank: AnchorBank,
    layer: str,
    anchor_eps: float,
    anchor_representation_mode: str,
    norm_scale: float,
) -> List[torch.Tensor]:
    return [
        anchor_jacobian(
            theta,
            shape,
            q,
            layer,
            anchor_eps,
            anchor_representation_mode,
            norm_scale,
        )
        for q in bank.x
    ]


def predicted_drift_by_layer(
    theta: torch.Tensor,
    shape: ModelShape,
    bank: AnchorBank,
    delta: torch.Tensor,
    layer: str,
    anchor_eps: float,
    anchor_representation_mode: str,
    norm_scale: float,
    jacobians: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    current_rep = layer_representation(
        theta,
        shape,
        bank.x,
        layer,
        anchor_eps,
        anchor_representation_mode,
        norm_scale,
    )
    residual = current_rep - layer_anchor_z(bank, layer)
    if jacobians is None:
        jacobians = all_anchor_jacobians(
            theta,
            shape,
            bank,
            layer,
            anchor_eps,
            anchor_representation_mode,
            norm_scale,
        )
    rows = []
    for idx, jac in enumerate(jacobians):
        pred = residual[idx] + jac @ delta
        rows.append(torch.dot(pred, pred))
    return torch.stack(rows)


def exact_drift_by_layer(
    theta: torch.Tensor,
    shape: ModelShape,
    bank: AnchorBank,
    layer: str,
    anchor_eps: float,
    anchor_representation_mode: str,
    norm_scale: float,
) -> torch.Tensor:
    current_rep = layer_representation(
        theta,
        shape,
        bank.x,
        layer,
        anchor_eps,
        anchor_representation_mode,
        norm_scale,
    )
    residual = current_rep - layer_anchor_z(bank, layer)
    return (residual**2).sum(dim=1)


def positive_forgetting(theta: torch.Tensor, shape: ModelShape, batch: Batch, delta: torch.Tensor) -> torch.Tensor:
    before = per_sample_mse(theta, shape, batch)
    after = per_sample_mse(theta + delta, shape, batch)
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


def select_protected(
    predicted_drift: torch.Tensor,
    weights: torch.Tensor,
    eps: torch.Tensor,
    *,
    tau_collision: float,
    protect_top_k: int,
) -> torch.Tensor:
    violation = torch.clamp(predicted_drift - eps, min=0.0)
    score = weights * violation
    eligible = torch.nonzero(score > tau_collision, as_tuple=False).reshape(-1)
    if protect_top_k <= 0 or eligible.numel() <= protect_top_k:
        return eligible
    top_local = torch.topk(score[eligible], k=protect_top_k).indices
    return eligible[top_local]


def build_constraint_system(
    theta: torch.Tensor,
    shape: ModelShape,
    bank: AnchorBank,
    selected: torch.Tensor,
    layers: Sequence[str],
    *,
    anchor_eps: float,
    anchor_representation_mode: str,
    norm_scale: float,
    constraint_mode: str,
    delta_raw: torch.Tensor,
    tolerance_tiny: float,
    broken_anchor_policy: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    rows: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    active_anchor_count = 0
    for anchor_idx in selected.tolist():
        q = bank.x[anchor_idx]
        sqrt_w = torch.sqrt(bank.weights[anchor_idx])
        anchor_had_constraint = False
        for layer in layers:
            jac = anchor_jacobian(
                theta,
                shape,
                q,
                layer,
                anchor_eps,
                anchor_representation_mode,
                norm_scale,
            )
            current = layer_representation(
                theta,
                shape,
                q.unsqueeze(0),
                layer,
                anchor_eps,
                anchor_representation_mode,
                norm_scale,
            ).reshape(-1)
            stored = layer_anchor_z(bank, layer)[anchor_idx]
            residual = current - stored
            if constraint_mode == "tangent":
                rows.append(sqrt_w * jac)
                target = torch.zeros(jac.shape[0], device=theta.device, dtype=theta.dtype)
                anchor_had_constraint = True
            elif constraint_mode == "restore":
                rows.append(sqrt_w * jac)
                target = -sqrt_w * (current - stored)
                anchor_had_constraint = True
            elif constraint_mode == "tolerance":
                predicted = residual + jac @ delta_raw
                predicted_norm_sq = torch.dot(predicted, predicted)
                current_norm_sq = torch.dot(residual, residual)
                eps_i = bank.eps[anchor_idx]
                if predicted_norm_sq <= eps_i:
                    continue
                if current_norm_sq > eps_i and predicted_norm_sq < current_norm_sq:
                    if broken_anchor_policy == "allow_improving":
                        continue
                    if broken_anchor_policy != "project_boundary":
                        raise ValueError(f"Unknown broken anchor policy: {broken_anchor_policy}")
                predicted_norm = torch.sqrt(predicted_norm_sq)
                if predicted_norm < tolerance_tiny:
                    boundary_target = torch.zeros_like(predicted)
                else:
                    boundary_target = torch.sqrt(eps_i) * predicted / (predicted_norm + tolerance_tiny)
                rows.append(sqrt_w * jac)
                target = sqrt_w * (boundary_target - residual)
                anchor_had_constraint = True
            else:
                raise ValueError(f"Unknown constraint mode: {constraint_mode}")
            targets.append(target)
        if anchor_had_constraint:
            active_anchor_count += 1

    if not rows:
        return (
            torch.zeros((0, theta.numel()), device=theta.device, dtype=theta.dtype),
            torch.zeros((0,), device=theta.device, dtype=theta.dtype),
            0,
        )
    return torch.cat(rows, dim=0), torch.cat(targets, dim=0), active_anchor_count


def affine_project(delta_raw: torch.Tensor, a: torch.Tensor, b: torch.Tensor, damping: float) -> torch.Tensor:
    if a.shape[0] == 0:
        return delta_raw
    eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    correction_rhs = a @ delta_raw - b
    coeff = torch.linalg.solve(a @ a.T + damping * eye, correction_rhs)
    delta = delta_raw - a.T @ coeff
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Non-finite projected update.")
    return delta


def method_layers(method: str) -> Sequence[str]:
    if method == "sgd":
        return []
    if method == "gfo_hidden":
        return ["hidden"]
    if method == "gfo_output":
        return ["output"]
    if method == "gfo_both":
        return ["hidden", "output"]
    raise ValueError(f"Unknown method: {method}")


def stage1_probe(
    theta: torch.Tensor,
    shape: ModelShape,
    task2: Task,
    anchor_eval: Batch,
    bank: AnchorBank,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    input_noise: float,
    output_noise: float,
    freeze_output: bool,
    anchor_eps: float,
    anchor_representation_mode: str,
    norm_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Optional[float]]:
    hidden_pred: List[float] = []
    output_pred: List[float] = []
    combined_pred: List[float] = []
    forget: List[float] = []

    for _ in range(steps):
        batch = sample_task(
            task2,
            batch_size,
            input_noise=input_noise,
            output_noise=output_noise,
            device=device,
            dtype=dtype,
        )
        grad = loss_grad(theta, shape, batch, freeze_output=freeze_output)
        delta = -lr * grad
        hidden_drift = predicted_drift_by_layer(
            theta,
            shape,
            bank,
            delta,
            "hidden",
            anchor_eps,
            anchor_representation_mode,
            norm_scale,
        )
        output_drift = predicted_drift_by_layer(
            theta,
            shape,
            bank,
            delta,
            "output",
            anchor_eps,
            anchor_representation_mode,
            norm_scale,
        )
        forgetting = positive_forgetting(theta, shape, anchor_eval, delta)

        hidden_pred.extend(float(v) for v in hidden_drift.detach().cpu())
        output_pred.extend(float(v) for v in output_drift.detach().cpu())
        combined_pred.extend(float(v) for v in (hidden_drift + output_drift).detach().cpu())
        forget.extend(float(v) for v in forgetting.detach().cpu())

    return {
        "hidden_rep_drift_vs_forgetting_pearson": correlation(hidden_pred, forget, rank=False),
        "hidden_rep_drift_vs_forgetting_spearman": correlation(hidden_pred, forget, rank=True),
        "output_rep_drift_vs_forgetting_pearson": correlation(output_pred, forget, rank=False),
        "output_rep_drift_vs_forgetting_spearman": correlation(output_pred, forget, rank=True),
        "combined_rep_drift_vs_forgetting_pearson": correlation(combined_pred, forget, rank=False),
        "combined_rep_drift_vs_forgetting_spearman": correlation(combined_pred, forget, rank=True),
    }


def train_task2_method(
    theta_initial: torch.Tensor,
    shape: ModelShape,
    task2: Task,
    bank: AnchorBank,
    task1_eval: Batch,
    task2_eval: Batch,
    *,
    method: str,
    steps: int,
    batch_size: int,
    lr: float,
    damping: float,
    tau_collision: float,
    protect_top_k: int,
    constraint_mode: str,
    input_noise: float,
    output_noise: float,
    freeze_output: bool,
    anchor_eps: float,
    anchor_representation_mode: str,
    norm_scale: float,
    tolerance_tiny: float,
    broken_anchor_policy: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    theta = theta_initial.detach().clone()
    layers = method_layers(method)
    task1_before = float(mse(theta, shape, task1_eval).detach().cpu())
    task2_before = float(mse(theta, shape, task2_eval).detach().cpu())

    update_ratios: List[float] = []
    protected_counts: List[int] = []
    hidden_drift_after: List[float] = []
    output_drift_after: List[float] = []

    for _ in range(steps):
        batch = sample_task(
            task2,
            batch_size,
            input_noise=input_noise,
            output_noise=output_noise,
            device=device,
            dtype=dtype,
        )
        grad = loss_grad(theta, shape, batch, freeze_output=freeze_output)
        delta_raw = -lr * grad

        if method == "sgd":
            delta = delta_raw
            selected = torch.empty((0,), device=theta.device, dtype=torch.long)
        else:
            drift = torch.zeros(bank.x.shape[0], device=theta.device, dtype=theta.dtype)
            for layer in layers:
                drift = drift + predicted_drift_by_layer(
                    theta,
                    shape,
                    bank,
                    delta_raw,
                    layer,
                    anchor_eps,
                    anchor_representation_mode,
                    norm_scale,
                )
            selected = select_protected(
                drift,
                bank.weights,
                bank.eps,
                tau_collision=tau_collision,
                protect_top_k=protect_top_k,
            )
            a, b, active_anchor_count = build_constraint_system(
                theta,
                shape,
                bank,
                selected,
                layers,
                anchor_eps=anchor_eps,
                anchor_representation_mode=anchor_representation_mode,
                norm_scale=norm_scale,
                constraint_mode=constraint_mode,
                delta_raw=delta_raw,
                tolerance_tiny=tolerance_tiny,
                broken_anchor_policy=broken_anchor_policy,
            )
            delta = affine_project(delta_raw, a, b, damping)

        raw_norm = torch.linalg.vector_norm(delta_raw)
        update_ratios.append(float((torch.linalg.vector_norm(delta) / (raw_norm + 1e-12)).detach().cpu()))
        if method == "sgd":
            protected_counts.append(0)
        elif constraint_mode == "tolerance":
            protected_counts.append(active_anchor_count)
        else:
            protected_counts.append(int(selected.numel()))
        theta = theta + delta

        hidden_drift_after.append(
            float(
                exact_drift_by_layer(
                    theta,
                    shape,
                    bank,
                    "hidden",
                    anchor_eps,
                    anchor_representation_mode,
                    norm_scale,
                )
                .mean()
                .detach()
                .cpu()
            )
        )
        output_drift_after.append(
            float(
                exact_drift_by_layer(
                    theta,
                    shape,
                    bank,
                    "output",
                    anchor_eps,
                    anchor_representation_mode,
                    norm_scale,
                )
                .mean()
                .detach()
                .cpu()
            )
        )

    task1_after = float(mse(theta, shape, task1_eval).detach().cpu())
    task2_after = float(mse(theta, shape, task2_eval).detach().cpu())
    return {
        "task1_before": task1_before,
        "task1_after": task1_after,
        "task2_before": task2_before,
        "task2_after": task2_after,
        "forgetting_delta": task1_after - task1_before,
        "task2_gain": task2_before - task2_after,
        "update_ratio_mean": sum(update_ratios) / len(update_ratios),
        "protected_anchors_mean": sum(protected_counts) / len(protected_counts),
        "hidden_anchor_drift_mean": sum(hidden_drift_after) / len(hidden_drift_after),
        "output_anchor_drift_mean": sum(output_drift_after) / len(output_drift_after),
    }


def run_seed(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    set_seed(seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    shape = ModelShape(args.input_dim, args.hidden_dim, args.output_dim)

    task1, task2 = make_tasks(
        args.input_dim,
        args.output_dim,
        args.rank,
        args.overlap,
        device=device,
        dtype=dtype,
    )
    theta0 = init_flat(shape, device=device, dtype=dtype)
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
    theta1 = train_task1(
        theta0,
        shape,
        task1_train,
        steps=args.task1_steps,
        lr=args.task1_lr,
        freeze_output=args.freeze_output,
    )
    anchor_batch = sample_task(
        task1,
        args.num_anchors,
        input_noise=args.input_noise,
        output_noise=0.0,
        device=device,
        dtype=dtype,
    )
    bank = build_anchor_bank(
        theta1,
        shape,
        anchor_batch,
        anchor_eps=args.anchor_eps,
        anchor_representation_mode=args.anchor_representation,
        norm_scale=args.norm_scale,
        drift_tolerance=args.drift_tolerance,
        importance=args.anchor_importance,
    )

    drift_probe = stage1_probe(
        theta1,
        shape,
        task2,
        anchor_batch,
        bank,
        steps=args.probe_steps,
        batch_size=args.batch_size,
        lr=args.task2_lr,
        input_noise=args.input_noise,
        output_noise=args.output_noise,
        freeze_output=args.freeze_output,
        anchor_eps=args.anchor_eps,
        anchor_representation_mode=args.anchor_representation,
        norm_scale=args.norm_scale,
        device=device,
        dtype=dtype,
    )

    methods = {}
    for method in ("sgd", "gfo_hidden", "gfo_output", "gfo_both"):
        methods[method] = train_task2_method(
            theta1,
            shape,
            task2,
            bank,
            task1_eval,
            task2_eval,
            method=method,
            steps=args.task2_steps,
            batch_size=args.batch_size,
            lr=args.task2_lr,
            damping=args.damping,
            tau_collision=args.tau_collision,
            protect_top_k=args.protect_top_k,
            constraint_mode=args.constraint_mode,
            input_noise=args.input_noise,
            output_noise=args.output_noise,
            freeze_output=args.freeze_output,
            anchor_eps=args.anchor_eps,
            anchor_representation_mode=args.anchor_representation,
            norm_scale=args.norm_scale,
            tolerance_tiny=args.tolerance_tiny,
            broken_anchor_policy=args.broken_anchor_policy,
            device=device,
            dtype=dtype,
        )

    return {
        "seed": seed,
        "task1_loss_after_pretrain": float(mse(theta1, shape, task1_eval).detach().cpu()),
        "task2_loss_before_continual": float(mse(theta1, shape, task2_eval).detach().cpu()),
        "drift_probe": drift_probe,
        "methods": methods,
    }


def mean_std(values: Iterable[Optional[float]]) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    tensor = torch.tensor(vals, dtype=torch.float64)
    return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False))}


def aggregate(seed_reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not seed_reports:
        raise ValueError("Cannot aggregate empty seed report list.")
    drift_keys = list(seed_reports[0]["drift_probe"].keys())  # type: ignore[index, union-attr]
    method_names = list(seed_reports[0]["methods"].keys())  # type: ignore[index, union-attr]

    drift_summary = {
        key: mean_std(report["drift_probe"][key] for report in seed_reports)  # type: ignore[index]
        for key in drift_keys
    }
    method_summary = {}
    for method in method_names:
        metrics = list(seed_reports[0]["methods"][method].keys())  # type: ignore[index]
        method_summary[method] = {
            metric: mean_std(report["methods"][method][metric] for report in seed_reports)  # type: ignore[index]
            for metric in metrics
        }
    return {"drift_probe": drift_summary, "methods": method_summary}


def fmt(stats: Dict[str, float]) -> str:
    return f"{stats['mean']:.4f} +/- {stats['std']:.4f}"


def print_summary(report: Dict[str, object]) -> None:
    summary = report["summary"]  # type: ignore[index]
    config = report["config"]  # type: ignore[index]
    print("\nGFO LINEAR ACTIVATION-CONSTRAINT SUMMARY")
    print("=" * 124)
    print(
        f"seeds={report['seed_count']} input={config['input_dim']} hidden={config['hidden_dim']} "
        f"output={config['output_dim']} rank={config['rank']} overlap={config['overlap']} "
        f"representation={config['anchor_representation']} constraint={config['constraint_mode']}"
    )
    print("-" * 124)
    print("Stage 1: predicted anchor-representation drift vs actual forgetting")
    print("-" * 124)
    for key, stats in summary["drift_probe"].items():  # type: ignore[index, union-attr]
        print(f"{key:48s} {fmt(stats)}")
    print("-" * 124)
    print("Stage 2: task-2 learning under activation constraints")
    print("-" * 124)
    print(
        f"{'method':14s} {'task1_after':>18s} {'task2_after':>18s} {'forgetting':>18s} "
        f"{'task2_gain':>18s} {'update_ratio':>18s} {'protected':>14s}"
    )
    print("-" * 124)
    for method, stats in summary["methods"].items():  # type: ignore[index, union-attr]
        print(
            f"{method:14s} "
            f"{fmt(stats['task1_after']):>18s} "
            f"{fmt(stats['task2_after']):>18s} "
            f"{fmt(stats['forgetting_delta']):>18s} "
            f"{fmt(stats['task2_gain']):>18s} "
            f"{fmt(stats['update_ratio_mean']):>18s} "
            f"{fmt(stats['protected_anchors_mean']):>14s}"
        )
    print("=" * 124)


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--num-anchors", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-noise", type=float, default=0.0)
    parser.add_argument("--output-noise", type=float, default=0.0)

    parser.add_argument("--task1-steps", type=int, default=1200)
    parser.add_argument("--task1-lr", type=float, default=0.05)
    parser.add_argument("--task2-steps", type=int, default=200)
    parser.add_argument("--task2-lr", type=float, default=0.02)
    parser.add_argument("--probe-steps", type=int, default=160)
    parser.add_argument("--freeze-output", action="store_true")

    parser.add_argument("--anchor-eps", type=float, default=1e-8)
    parser.add_argument(
        "--anchor-representation",
        choices=["direction", "norm", "full", "direction_norm"],
        default="full",
    )
    parser.add_argument("--norm-scale", type=float, default=1.0)
    parser.add_argument("--anchor-importance", type=float, default=1.0)
    parser.add_argument("--drift-tolerance", type=float, default=1e-4)
    parser.add_argument("--tau-collision", type=float, default=0.0)
    parser.add_argument("--protect-top-k", type=int, default=0)
    parser.add_argument("--constraint-mode", choices=["tangent", "restore", "tolerance"], default="restore")
    parser.add_argument("--tolerance-tiny", type=float, default=1e-12)
    parser.add_argument(
        "--broken-anchor-policy",
        choices=["allow_improving", "project_boundary"],
        default="allow_improving",
    )
    parser.add_argument("--damping", type=float, default=1e-3)

    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> Dict[str, object]:
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
        "anchor_eps": args.anchor_eps,
        "anchor_representation": args.anchor_representation,
        "norm_scale": args.norm_scale,
        "anchor_importance": args.anchor_importance,
        "drift_tolerance": args.drift_tolerance,
        "tau_collision": args.tau_collision,
        "protect_top_k": args.protect_top_k,
        "constraint_mode": args.constraint_mode,
        "tolerance_tiny": args.tolerance_tiny,
        "broken_anchor_policy": args.broken_anchor_policy,
        "damping": args.damping,
        "device": args.device,
        "dtype": args.dtype,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    if args.num_anchors <= 0:
        raise ValueError("--num-anchors must be positive.")
    if args.task1_steps <= 0 or args.task2_steps <= 0:
        raise ValueError("--task1-steps and --task2-steps must be positive.")
    if args.task1_lr <= 0 or args.task2_lr <= 0:
        raise ValueError("--task1-lr and --task2-lr must be positive.")
    if args.anchor_eps <= 0:
        raise ValueError("--anchor-eps must be positive.")
    if args.norm_scale <= 0:
        raise ValueError("--norm-scale must be positive.")
    if args.damping <= 0:
        raise ValueError("--damping must be positive.")
    if args.tolerance_tiny <= 0:
        raise ValueError("--tolerance-tiny must be positive.")
    if args.tau_collision < 0:
        raise ValueError("--tau-collision must be non-negative.")
    if args.drift_tolerance <= 0:
        raise ValueError("--drift-tolerance must be positive.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    reports = []
    for i in range(args.seed_count):
        seed = args.seed_offset + i
        print(f"running_seed={seed}")
        reports.append(run_seed(args, seed))

    report = {
        "experiment": "gfo_linear_activation_constraints",
        "seed_count": args.seed_count,
        "config": config_from_args(args),
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
