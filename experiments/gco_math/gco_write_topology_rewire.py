#!/usr/bin/env python3
"""Topology-rewiring write test for GCO.

This experiment starts from the failure found by the budgeted orthogonal-room
test:

    protected writes preserve old behavior, but when free room is small the
    required update norm becomes too large and the new behavior is only
    partially learned.

Here we test whether growing dormant routes can recover write capacity.

Model:

    base output:      y = W k
    grown route:      y = W k + U phi(k)

Old protected behavior:

    W K_old = V_old

New behavior:

    W k_new + U phi(k_new) -> v_new

Compared methods:

    raw_budget          unconstrained rank-one write into W, clipped by budget
    protected_budget    protected least-squares write into W, clipped by budget
    linear_rewire       grow dormant linear routes phi(k) = R k
    relu_rewire         grow dormant gated routes phi(k) = relu(R k - b)
    constructive_relu   grow one gate directly separating k_new from K_old

The key question:

    Can route growth recover new behavior gain without old behavior damage when
    protected weight writing is budget-limited?
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True)
class MethodResult:
    method: str
    free_room_ratio: float
    protected_overlap_ratio: float
    new_mse_before: float
    new_mse_after: float
    new_gain: float
    new_gain_fraction: float
    old_mse_before: float
    old_mse_after: float
    old_damage: float
    update_norm_unclipped: float
    update_norm_after_budget: float
    budget_scale: float
    selected_route_count: int
    selected_new_activation_mean: float | None
    selected_old_activation_rms_mean: float | None
    selected_score_mean: float | None


@dataclass(frozen=True)
class SweepCase:
    seed: int
    key_dim: int
    value_dim: int
    old_count: int
    requested_overlap: float
    free_room_ratio: float
    protected_overlap_ratio: float
    results: list[MethodResult]


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float.")
    return values


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
            raise RuntimeError("MPS does not support float64 for this experiment.")
        return torch.float64
    raise ValueError(f"Unknown dtype: {name}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def randn(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=dtype)


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def normalize_columns(x: torch.Tensor, eps: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=0, keepdim=True)
    if bool((norm <= eps).any().detach().cpu()):
        raise RuntimeError("Cannot normalize a zero-length column.")
    return x / norm


def normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    if bool((norm <= eps).any().detach().cpu()):
        raise RuntimeError("Cannot normalize a zero-length row.")
    return x / norm


def orthonormal_columns(rows: int, cols: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if cols > rows:
        raise ValueError(f"Cannot build {cols} orthonormal columns in {rows} dimensions.")
    q, r = torch.linalg.qr(randn((rows, cols), device=device, dtype=dtype), mode="reduced")
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def apply_budget(delta: torch.Tensor, max_update_norm: float) -> tuple[torch.Tensor, float, float, float]:
    if max_update_norm <= 0:
        raise ValueError("--max-update-norm must be positive.")
    raw_norm = scalar(torch.linalg.matrix_norm(delta))
    if raw_norm <= max_update_norm:
        return delta, raw_norm, raw_norm, 1.0
    scale = max_update_norm / (raw_norm + 1e-12)
    budgeted = delta * scale
    return budgeted, raw_norm, scalar(torch.linalg.matrix_norm(budgeted)), scale


def mse_columns(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=0)


def make_case(
    *,
    key_dim: int,
    value_dim: int,
    old_count: int,
    overlap: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    if key_dim <= 1:
        raise ValueError("--key-dim must be greater than 1.")
    if value_dim <= 0:
        raise ValueError("--value-dim must be positive.")
    if old_count <= 0:
        raise ValueError("--old-count values must be positive.")
    if old_count > key_dim:
        raise ValueError("--old-count must be <= --key-dim.")
    if not (0.0 <= overlap <= 1.0):
        raise ValueError("--overlaps values must be in [0, 1].")

    set_seed(seed)
    old_keys = orthonormal_columns(key_dim, old_count, device=device, dtype=dtype)
    old_values = randn((value_dim, old_count), device=device, dtype=dtype) / math.sqrt(value_dim)
    p_old = old_keys @ old_keys.T
    identity = torch.eye(key_dim, device=device, dtype=dtype)
    p_free = identity - p_old

    old_component = old_keys @ normalize_columns(randn((old_count, 1), device=device, dtype=dtype), eps)
    free_dim = key_dim - old_count
    if overlap < 1.0 and free_dim <= 0:
        raise ValueError("Requested overlap < 1.0 but no free complement exists.")
    if free_dim > 0:
        random_free = p_free @ randn((key_dim, 1), device=device, dtype=dtype)
        free_component = normalize_columns(random_free, eps)
    else:
        free_component = torch.zeros((key_dim, 1), device=device, dtype=dtype)

    k_new = math.sqrt(overlap) * old_component
    if overlap < 1.0:
        k_new = k_new + math.sqrt(1.0 - overlap) * free_component
    k_new = normalize_columns(k_new, eps)
    v_new = randn((value_dim, 1), device=device, dtype=dtype) / math.sqrt(value_dim)

    protected = scalar(torch.linalg.vector_norm(p_old @ k_new) ** 2 / (torch.linalg.vector_norm(k_new) ** 2 + eps))
    free = scalar(torch.linalg.vector_norm(p_free @ k_new) ** 2 / (torch.linalg.vector_norm(k_new) ** 2 + eps))
    return old_keys, old_values, k_new, v_new, p_old, p_free, protected, free


def base_weight(old_keys: torch.Tensor, old_values: torch.Tensor) -> torch.Tensor:
    return old_values @ old_keys.T


def raw_weight_delta(w_base: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, eps: float) -> torch.Tensor:
    error = v_new - w_base @ k_new
    denom = scalar(k_new.T @ k_new)
    if denom <= eps:
        raise RuntimeError("New key has near-zero norm.")
    return error @ k_new.T / denom


def protected_weight_delta(
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    lambda_protect: float,
    lambda_ridge: float,
) -> torch.Tensor:
    if lambda_protect < 0:
        raise ValueError("--lambda-protect must be non-negative.")
    if lambda_ridge <= 0:
        raise ValueError("--lambda-ridge must be positive.")
    key_dim = k_new.shape[0]
    identity = torch.eye(key_dim, device=k_new.device, dtype=k_new.dtype)
    system = k_new @ k_new.T + lambda_protect * (old_keys @ old_keys.T) + lambda_ridge * identity
    right = torch.linalg.solve(system, k_new)
    error = v_new - w_base @ k_new
    return error @ right.T


def evaluate_weight_delta(
    *,
    method: str,
    w_base: torch.Tensor,
    delta_w: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    free_room: float,
    protected_overlap: float,
    max_update_norm: float,
) -> MethodResult:
    budgeted, raw_norm, budgeted_norm, scale = apply_budget(delta_w, max_update_norm)
    old_before = scalar(mse_columns(w_base @ old_keys, old_values).mean())
    new_before = scalar(mse_columns(w_base @ k_new, v_new).mean())
    w_after = w_base + budgeted
    old_after = scalar(mse_columns(w_after @ old_keys, old_values).mean())
    new_after = scalar(mse_columns(w_after @ k_new, v_new).mean())
    new_gain = new_before - new_after
    return MethodResult(
        method=method,
        free_room_ratio=free_room,
        protected_overlap_ratio=protected_overlap,
        new_mse_before=new_before,
        new_mse_after=new_after,
        new_gain=new_gain,
        new_gain_fraction=new_gain / (new_before + 1e-12),
        old_mse_before=old_before,
        old_mse_after=old_after,
        old_damage=old_after - old_before,
        update_norm_unclipped=raw_norm,
        update_norm_after_budget=budgeted_norm,
        budget_scale=scale,
        selected_route_count=0,
        selected_new_activation_mean=None,
        selected_old_activation_rms_mean=None,
        selected_score_mean=None,
    )


def route_write(
    *,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    lambda_old: float,
    lambda_ridge: float,
) -> torch.Tensor:
    if lambda_old < 0:
        raise ValueError("--route-lambda-old must be non-negative.")
    if lambda_ridge <= 0:
        raise ValueError("--route-lambda-ridge must be positive.")
    route_count = z_new.shape[0]
    identity = torch.eye(route_count, device=z_new.device, dtype=z_new.dtype)
    system = z_new @ z_new.T + lambda_old * (z_old @ z_old.T) + lambda_ridge * identity
    coeff = torch.linalg.solve(system, z_new)
    error = v_new - w_base @ k_new
    return error @ coeff.T


def evaluate_route_delta(
    *,
    method: str,
    w_base: torch.Tensor,
    u_delta: torch.Tensor,
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    free_room: float,
    protected_overlap: float,
    max_update_norm: float,
    route_scores: torch.Tensor,
) -> MethodResult:
    budgeted, raw_norm, budgeted_norm, scale = apply_budget(u_delta, max_update_norm)
    old_before = scalar(mse_columns(w_base @ old_keys, old_values).mean())
    new_before = scalar(mse_columns(w_base @ k_new, v_new).mean())
    old_after = scalar(mse_columns(w_base @ old_keys + budgeted @ z_old, old_values).mean())
    new_after = scalar(mse_columns(w_base @ k_new + budgeted @ z_new, v_new).mean())
    new_gain = new_before - new_after
    new_abs = torch.abs(z_new.reshape(-1))
    old_rms = torch.sqrt((z_old**2).mean(dim=1))
    return MethodResult(
        method=method,
        free_room_ratio=free_room,
        protected_overlap_ratio=protected_overlap,
        new_mse_before=new_before,
        new_mse_after=new_after,
        new_gain=new_gain,
        new_gain_fraction=new_gain / (new_before + 1e-12),
        old_mse_before=old_before,
        old_mse_after=old_after,
        old_damage=old_after - old_before,
        update_norm_unclipped=raw_norm,
        update_norm_after_budget=budgeted_norm,
        budget_scale=scale,
        selected_route_count=z_new.shape[0],
        selected_new_activation_mean=scalar(new_abs.mean()),
        selected_old_activation_rms_mean=scalar(old_rms.mean()),
        selected_score_mean=scalar(route_scores.mean()),
    )


def select_routes(
    *,
    z_old_all: torch.Tensor,
    z_new_all: torch.Tensor,
    route_count: int,
    score_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if route_count <= 0:
        raise ValueError("--route-count must be positive.")
    if route_count > z_new_all.shape[0]:
        raise ValueError("--route-count must be <= --feature-count.")
    old_energy = (z_old_all**2).mean(dim=1)
    new_energy = z_new_all.reshape(-1) ** 2
    scores = new_energy / (old_energy + score_eps)
    selected = torch.topk(scores, k=route_count).indices
    return z_old_all[selected], z_new_all[selected], scores[selected]


def linear_features(
    *,
    key_dim: int,
    feature_count: int,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    routes = normalize_rows(randn((feature_count, key_dim), device=device, dtype=dtype), eps)
    return routes @ old_keys, routes @ k_new


def relu_features(
    *,
    key_dim: int,
    feature_count: int,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    bias_range: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bias_range <= 0:
        raise ValueError("--bias-range must be positive.")
    routes = normalize_rows(randn((feature_count, key_dim), device=device, dtype=dtype), eps)
    bias = torch.empty((feature_count, 1), device=device, dtype=dtype).uniform_(-bias_range, bias_range)
    return torch.relu(routes @ old_keys - bias), torch.relu(routes @ k_new - bias)


def constructive_relu_features(
    *,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    route = normalize_rows(k_new.T, eps)
    old_pre = route @ old_keys
    new_pre = route @ k_new
    max_old = old_pre.max(dim=1, keepdim=True).values
    bias = 0.5 * (new_pre + max_old)
    z_old = torch.relu(old_pre - bias)
    z_new = torch.relu(new_pre - bias)
    old_energy = (z_old**2).mean(dim=1)
    new_energy = z_new.reshape(-1) ** 2
    score = new_energy / (old_energy + eps)
    return z_old, z_new, score


def run_case(
    *,
    key_dim: int,
    value_dim: int,
    old_count: int,
    overlap: float,
    seed: int,
    feature_count: int,
    route_count: int,
    bias_range: float,
    lambda_protect: float,
    lambda_ridge: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
    max_update_norm: float,
    score_eps: float,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> SweepCase:
    old_keys, old_values, k_new, v_new, _, _, protected_overlap, free_room = make_case(
        key_dim=key_dim,
        value_dim=value_dim,
        old_count=old_count,
        overlap=overlap,
        seed=seed,
        device=device,
        dtype=dtype,
        eps=eps,
    )
    w_base = base_weight(old_keys, old_values)

    raw = evaluate_weight_delta(
        method="raw_budget",
        w_base=w_base,
        delta_w=raw_weight_delta(w_base, k_new, v_new, eps),
        old_keys=old_keys,
        old_values=old_values,
        k_new=k_new,
        v_new=v_new,
        free_room=free_room,
        protected_overlap=protected_overlap,
        max_update_norm=max_update_norm,
    )
    protected = evaluate_weight_delta(
        method="protected_budget",
        w_base=w_base,
        delta_w=protected_weight_delta(
            w_base,
            old_keys,
            k_new,
            v_new,
            lambda_protect=lambda_protect,
            lambda_ridge=lambda_ridge,
        ),
        old_keys=old_keys,
        old_values=old_values,
        k_new=k_new,
        v_new=v_new,
        free_room=free_room,
        protected_overlap=protected_overlap,
        max_update_norm=max_update_norm,
    )

    lin_old_all, lin_new_all = linear_features(
        key_dim=key_dim,
        feature_count=feature_count,
        old_keys=old_keys,
        k_new=k_new,
        device=device,
        dtype=dtype,
        eps=eps,
    )
    lin_old, lin_new, lin_scores = select_routes(
        z_old_all=lin_old_all,
        z_new_all=lin_new_all,
        route_count=route_count,
        score_eps=score_eps,
    )
    lin_u = route_write(
        w_base=w_base,
        old_keys=old_keys,
        k_new=k_new,
        v_new=v_new,
        z_old=lin_old,
        z_new=lin_new,
        lambda_old=route_lambda_old,
        lambda_ridge=route_lambda_ridge,
    )
    linear = evaluate_route_delta(
        method="linear_rewire",
        w_base=w_base,
        u_delta=lin_u,
        z_old=lin_old,
        z_new=lin_new,
        old_keys=old_keys,
        old_values=old_values,
        k_new=k_new,
        v_new=v_new,
        free_room=free_room,
        protected_overlap=protected_overlap,
        max_update_norm=max_update_norm,
        route_scores=lin_scores,
    )

    relu_old_all, relu_new_all = relu_features(
        key_dim=key_dim,
        feature_count=feature_count,
        old_keys=old_keys,
        k_new=k_new,
        bias_range=bias_range,
        device=device,
        dtype=dtype,
        eps=eps,
    )
    relu_old, relu_new, relu_scores = select_routes(
        z_old_all=relu_old_all,
        z_new_all=relu_new_all,
        route_count=route_count,
        score_eps=score_eps,
    )
    relu_u = route_write(
        w_base=w_base,
        old_keys=old_keys,
        k_new=k_new,
        v_new=v_new,
        z_old=relu_old,
        z_new=relu_new,
        lambda_old=route_lambda_old,
        lambda_ridge=route_lambda_ridge,
    )
    relu = evaluate_route_delta(
        method="relu_rewire",
        w_base=w_base,
        u_delta=relu_u,
        z_old=relu_old,
        z_new=relu_new,
        old_keys=old_keys,
        old_values=old_values,
        k_new=k_new,
        v_new=v_new,
        free_room=free_room,
        protected_overlap=protected_overlap,
        max_update_norm=max_update_norm,
        route_scores=relu_scores,
    )

    constructive_old, constructive_new, constructive_scores = constructive_relu_features(
        old_keys=old_keys,
        k_new=k_new,
        eps=eps,
    )
    constructive_u = route_write(
        w_base=w_base,
        old_keys=old_keys,
        k_new=k_new,
        v_new=v_new,
        z_old=constructive_old,
        z_new=constructive_new,
        lambda_old=route_lambda_old,
        lambda_ridge=route_lambda_ridge,
    )
    constructive = evaluate_route_delta(
        method="constructive_relu",
        w_base=w_base,
        u_delta=constructive_u,
        z_old=constructive_old,
        z_new=constructive_new,
        old_keys=old_keys,
        old_values=old_values,
        k_new=k_new,
        v_new=v_new,
        free_room=free_room,
        protected_overlap=protected_overlap,
        max_update_norm=max_update_norm,
        route_scores=constructive_scores,
    )

    return SweepCase(
        seed=seed,
        key_dim=key_dim,
        value_dim=value_dim,
        old_count=old_count,
        requested_overlap=overlap,
        free_room_ratio=free_room,
        protected_overlap_ratio=protected_overlap,
        results=[raw, protected, linear, relu, constructive],
    )


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def fmt(value: float | None, width: int = 8) -> str:
    if value is None:
        return "None".rjust(width)
    return f"{value:{width}.4g}"


def print_explanation() -> None:
    print()
    print("GCO TOPOLOGY-REWIRING WRITE TEST")
    print("=" * 104)
    print("Question:")
    print("  When protected weight writing is budget-limited, can growing dormant routes")
    print("  recover new behavior without damaging old behavior?")
    print()
    print("Base model:")
    print("  y = W k")
    print()
    print("Rewired model:")
    print("  y = W k + U phi(k)")
    print()
    print("Route types:")
    print("  linear_rewire: phi(k) = R k")
    print("  relu_rewire:   phi(k) = relu(R k - b)")
    print("  constructive:  phi(k) = relu(k_new^T k - midpoint(max_old, new))")
    print()
    print("A good rewire has high activation on k_new and low activation on K_old.")
    print("=" * 104)


def print_summary(cases: list[SweepCase]) -> None:
    grouped: dict[tuple[int, float, str], dict[str, list[float]]] = {}
    for case in cases:
        for result in case.results:
            key = (case.old_count, case.requested_overlap, result.method)
            bucket = grouped.setdefault(
                key,
                {
                    "free": [],
                    "gain_frac": [],
                    "old_damage": [],
                    "update_norm": [],
                    "scale": [],
                    "new_act": [],
                    "old_rms": [],
                    "score": [],
                },
            )
            bucket["free"].append(result.free_room_ratio)
            bucket["gain_frac"].append(result.new_gain_fraction)
            bucket["old_damage"].append(result.old_damage)
            bucket["update_norm"].append(result.update_norm_after_budget)
            bucket["scale"].append(result.budget_scale)
            if result.selected_new_activation_mean is not None:
                bucket["new_act"].append(result.selected_new_activation_mean)
            if result.selected_old_activation_rms_mean is not None:
                bucket["old_rms"].append(result.selected_old_activation_rms_mean)
            if result.selected_score_mean is not None:
                bucket["score"].append(result.selected_score_mean)

    print()
    print("Readable aggregate summary")
    print("-" * 130)
    print(
        "old overlap free  method             gain_frac old_damage  upd_norm  scale  "
        "new_act old_rms route_score"
    )
    print("-" * 130)
    for (old_count, overlap, method), bucket in sorted(grouped.items()):
        print(
            f"{old_count:3d} {overlap:7.2f} {mean(bucket['free']):4.2f}  "
            f"{method:<18s} "
            f"{mean(bucket['gain_frac']):9.3f} "
            f"{mean(bucket['old_damage']):10.3g} "
            f"{mean(bucket['update_norm']):9.3g} "
            f"{mean(bucket['scale']):6.3f} "
            f"{fmt(mean(bucket['new_act']) if bucket['new_act'] else None)} "
            f"{fmt(mean(bucket['old_rms']) if bucket['old_rms'] else None)} "
            f"{fmt(mean(bucket['score']) if bucket['score'] else None, width=10)}"
        )
    print("-" * 130)
    print()
    print("How to read this:")
    print("  gain_frac near 1.0 means the new write was fitted.")
    print("  old_damage near 0 means old behavior was preserved.")
    print("  scale below 1.0 means the route/write exceeded the update budget.")
    print("  new_act high and old_rms low means a grown route separates new from old.")
    print("  If rewire beats protected_budget at low free room without old_damage, topology helped.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=16)
    parser.add_argument("--old-counts", type=str, default="8,24,48")
    parser.add_argument("--overlaps", type=str, default="0.0,0.25,0.5,0.75,0.9,0.97")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--feature-count", type=int, default=4096)
    parser.add_argument("--route-count", type=int, default=4)
    parser.add_argument("--bias-range", type=float, default=0.5)
    parser.add_argument("--lambda-protect", type=float, default=100.0)
    parser.add_argument("--lambda-ridge", type=float, default=1e-5)
    parser.add_argument("--route-lambda-old", type=float, default=100.0)
    parser.add_argument("--route-lambda-ridge", type=float, default=1e-5)
    parser.add_argument("--max-update-norm", type=float, default=2.0)
    parser.add_argument("--score-eps", type=float, default=1e-8)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/analysis/gco-write-topology-rewire-seed0-5seed.json"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive.")
    if args.feature_count <= 0:
        raise ValueError("--feature-count must be positive.")
    if args.route_count <= 0:
        raise ValueError("--route-count must be positive.")
    if args.route_count > args.feature_count:
        raise ValueError("--route-count must be <= --feature-count.")
    if args.max_update_norm <= 0:
        raise ValueError("--max-update-norm must be positive.")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    old_counts = parse_int_list(args.old_counts)
    overlaps = parse_float_list(args.overlaps)

    cases: list[SweepCase] = []
    for seed_offset in range(args.seeds):
        case_seed = args.seed + seed_offset
        for old_count in old_counts:
            for overlap in overlaps:
                cases.append(
                    run_case(
                        key_dim=args.key_dim,
                        value_dim=args.value_dim,
                        old_count=old_count,
                        overlap=overlap,
                        seed=case_seed,
                        feature_count=args.feature_count,
                        route_count=args.route_count,
                        bias_range=args.bias_range,
                        lambda_protect=args.lambda_protect,
                        lambda_ridge=args.lambda_ridge,
                        route_lambda_old=args.route_lambda_old,
                        route_lambda_ridge=args.route_lambda_ridge,
                        max_update_norm=args.max_update_norm,
                        score_eps=args.score_eps,
                        eps=args.eps,
                        device=device,
                        dtype=dtype,
                    )
                )

    print_explanation()
    print_summary(cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "gco_write_topology_rewire",
        "config": {
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "old_counts": old_counts,
            "overlaps": overlaps,
            "seed": args.seed,
            "seeds": args.seeds,
            "feature_count": args.feature_count,
            "route_count": args.route_count,
            "bias_range": args.bias_range,
            "lambda_protect": args.lambda_protect,
            "lambda_ridge": args.lambda_ridge,
            "route_lambda_old": args.route_lambda_old,
            "route_lambda_ridge": args.route_lambda_ridge,
            "max_update_norm": args.max_update_norm,
            "device": str(device),
            "dtype": str(dtype),
        },
        "cases": [
            {
                **{key: value for key, value in asdict(case).items() if key != "results"},
                "results": [asdict(result) for result in case.results],
            }
            for case in cases
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote_json={args.output}")


if __name__ == "__main__":
    main()
