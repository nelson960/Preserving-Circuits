#!/usr/bin/env python3
"""Learned route-constructor test for GCO topology growth.

The previous topology experiment showed that a hand-constructed gated route can
recover write capacity:

    phi(k) = relu(r_new^T k - b_new)

with r_new aligned to k_new and b_new above the maximum old activation. This
script asks the next question:

    Can a small neural reasoner learn to construct that route from geometry?

The reasoner receives:

    k_new
    mean(K_old)
    max projection of K_old onto k_new
    free_room_ratio
    protected_overlap_ratio

and predicts:

    r_hat
    b_hat

Evaluation compares:

    protected_budget      protected write into W only
    constructive_relu     oracle geometric gate
    learned_relu          reasoner-predicted gate

No symbolic labels are used for the action. The learning target is geometric:
activate on the new key, stay silent on old keys, and allow a budgeted write.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConstructorEval:
    method: str
    old_count: int
    overlap: float
    free_room_ratio: float
    protected_overlap_ratio: float
    new_gain_fraction: float
    old_damage: float
    update_norm_unclipped: float
    update_norm_after_budget: float
    budget_scale: float
    new_activation: float
    old_activation_rms: float
    old_activation_max: float
    direction_cosine_to_new: float
    threshold: float


class RouteConstructor(nn.Module):
    def __init__(
        self,
        key_dim: int,
        hidden_dim: int,
        *,
        direction_anchor: str,
        residual_scale: float,
        threshold_bias: float,
    ) -> None:
        super().__init__()
        if key_dim <= 0:
            raise ValueError("key_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if direction_anchor not in {"none", "new_key"}:
            raise ValueError("--direction-anchor must be 'none' or 'new_key'.")
        if residual_scale < 0:
            raise ValueError("--residual-scale must be non-negative.")
        self.key_dim = key_dim
        self.direction_anchor = direction_anchor
        self.residual_scale = residual_scale
        input_dim = 2 * key_dim + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.direction_head = nn.Linear(hidden_dim, key_dim)
        self.threshold_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.threshold_head.bias, threshold_bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(features)
        raw_direction = self.direction_head(hidden)
        if self.direction_anchor == "new_key":
            base_direction = features[:, : self.key_dim]
            direction = F.normalize(base_direction + self.residual_scale * torch.tanh(raw_direction), dim=-1)
        else:
            direction = F.normalize(raw_direction, dim=-1)
        threshold = self.threshold_head(hidden)
        return direction, threshold


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


def orthonormal_columns(rows: int, cols: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if cols > rows:
        raise ValueError(f"Cannot build {cols} orthonormal columns in {rows} dimensions.")
    q, r = torch.linalg.qr(randn((rows, cols), device=device, dtype=dtype), mode="reduced")
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def make_case(
    *,
    key_dim: int,
    value_dim: int,
    old_count: int,
    overlap: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    if old_count > key_dim:
        raise ValueError("--old-count must be <= --key-dim.")
    if not (0.0 <= overlap <= 1.0):
        raise ValueError("--overlaps values must be in [0, 1].")

    old_keys = orthonormal_columns(key_dim, old_count, device=device, dtype=dtype)
    old_values = randn((value_dim, old_count), device=device, dtype=dtype) / math.sqrt(value_dim)
    p_old = old_keys @ old_keys.T
    p_free = torch.eye(key_dim, device=device, dtype=dtype) - p_old

    old_component = old_keys @ normalize_columns(randn((old_count, 1), device=device, dtype=dtype), eps)
    free_dim = key_dim - old_count
    if overlap < 1.0 and free_dim <= 0:
        raise ValueError("Requested overlap < 1.0 but no free complement exists.")
    if free_dim > 0:
        free_component = normalize_columns(p_free @ randn((key_dim, 1), device=device, dtype=dtype), eps)
    else:
        free_component = torch.zeros((key_dim, 1), device=device, dtype=dtype)

    k_new = math.sqrt(overlap) * old_component
    if overlap < 1.0:
        k_new = k_new + math.sqrt(1.0 - overlap) * free_component
    k_new = normalize_columns(k_new, eps)
    v_new = randn((value_dim, 1), device=device, dtype=dtype) / math.sqrt(value_dim)

    protected = scalar(torch.linalg.vector_norm(p_old @ k_new) ** 2 / (torch.linalg.vector_norm(k_new) ** 2 + eps))
    free = scalar(torch.linalg.vector_norm(p_free @ k_new) ** 2 / (torch.linalg.vector_norm(k_new) ** 2 + eps))
    return old_keys, old_values, k_new, v_new, protected, free


def make_features(old_keys: torch.Tensor, k_new: torch.Tensor, free: float, protected: float) -> torch.Tensor:
    direction = F.normalize(k_new.T, dim=-1)
    old_mean = old_keys.mean(dim=1, keepdim=True).T
    old_projection_max = (direction @ old_keys).max(dim=1, keepdim=True).values
    scalars = torch.tensor(
        [[free, protected, scalar(old_projection_max.squeeze())]],
        device=k_new.device,
        dtype=k_new.dtype,
    )
    return torch.cat([direction, old_mean, scalars], dim=1)


def oracle_route(old_keys: torch.Tensor, k_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    direction = F.normalize(k_new.T, dim=-1)
    old_pre = direction @ old_keys
    new_pre = direction @ k_new
    threshold = 0.5 * (old_pre.max(dim=1, keepdim=True).values + new_pre)
    return direction, threshold


def route_activations(direction: torch.Tensor, threshold: torch.Tensor, old_keys: torch.Tensor, k_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_old = torch.relu(direction @ old_keys - threshold)
    z_new = torch.relu(direction @ k_new - threshold)
    return z_old, z_new


def soft_route_activations(
    direction: torch.Tensor,
    threshold: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("--train-temperature must be positive.")
    z_old = F.softplus((direction @ old_keys - threshold) / temperature) * temperature
    z_new = F.softplus((direction @ k_new - threshold) / temperature) * temperature
    return z_old, z_new


def base_weight(old_keys: torch.Tensor, old_values: torch.Tensor) -> torch.Tensor:
    return old_values @ old_keys.T


def mse_columns(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=0)


def apply_budget(delta: torch.Tensor, max_update_norm: float) -> tuple[torch.Tensor, float, float, float]:
    raw_norm = scalar(torch.linalg.matrix_norm(delta))
    if raw_norm <= max_update_norm:
        return delta, raw_norm, raw_norm, 1.0
    scale = max_update_norm / (raw_norm + 1e-12)
    budgeted = delta * scale
    return budgeted, raw_norm, scalar(torch.linalg.matrix_norm(budgeted)), scale


def apply_budget_tensor(delta: torch.Tensor, max_update_norm: float, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_norm = torch.linalg.matrix_norm(delta)
    scale = torch.clamp(max_update_norm / (raw_norm + eps), max=1.0)
    return delta * scale, raw_norm, scale


def protected_weight_delta(
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    lambda_protect: float,
    lambda_ridge: float,
) -> torch.Tensor:
    key_dim = k_new.shape[0]
    identity = torch.eye(key_dim, device=k_new.device, dtype=k_new.dtype)
    system = k_new @ k_new.T + lambda_protect * (old_keys @ old_keys.T) + lambda_ridge * identity
    right = torch.linalg.solve(system, k_new)
    error = v_new - w_base @ k_new
    return error @ right.T


def route_delta(
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    *,
    lambda_old: float,
    lambda_ridge: float,
) -> torch.Tensor:
    route_dim = z_new.shape[0]
    identity = torch.eye(route_dim, device=z_new.device, dtype=z_new.dtype)
    system = z_new @ z_new.T + lambda_old * (z_old @ z_old.T) + lambda_ridge * identity
    coeff = torch.linalg.solve(system, z_new)
    error = v_new - w_base @ k_new
    return error @ coeff.T


def utility_training_loss_for_case(
    *,
    model: RouteConstructor,
    key_dim: int,
    value_dim: int,
    old_count: int,
    overlap: float,
    max_update_norm: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
    old_weight: float,
    budget_weight: float,
    old_gate_weight: float,
    train_temperature: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    old_keys, old_values, k_new, v_new, protected, free = make_case(
        key_dim=key_dim,
        value_dim=value_dim,
        old_count=old_count,
        overlap=overlap,
        device=device,
        dtype=dtype,
        eps=eps,
    )
    w_base = base_weight(old_keys, old_values)
    features = make_features(old_keys, k_new, free, protected)
    direction, threshold = model(features)
    z_old, z_new = soft_route_activations(direction, threshold, old_keys, k_new, train_temperature)
    delta = route_delta(
        w_base,
        old_keys,
        k_new,
        v_new,
        z_old,
        z_new,
        lambda_old=route_lambda_old,
        lambda_ridge=route_lambda_ridge,
    )
    budgeted, raw_norm, budget_scale = apply_budget_tensor(delta, max_update_norm, eps)
    new_before = mse_columns(w_base @ k_new, v_new).mean().detach()
    new_after = mse_columns(w_base @ k_new + budgeted @ z_new, v_new).mean()
    old_after = mse_columns(w_base @ old_keys + budgeted @ z_old, old_values).mean()
    budget_penalty = torch.relu(raw_norm / max_update_norm - 1.0).pow(2)
    old_gate_penalty = (z_old**2).mean()
    normalized_new_after = new_after / (new_before + eps)
    loss = (
        normalized_new_after
        + old_weight * old_after
        + budget_weight * budget_penalty
        + old_gate_weight * old_gate_penalty
    )
    metrics = {
        "new_after_fraction": normalized_new_after.detach(),
        "old_damage": old_after.detach(),
        "budget_penalty": budget_penalty.detach(),
        "old_gate": old_gate_penalty.detach(),
        "budget_scale": budget_scale.detach(),
        "new_activation": z_new.detach().mean(),
        "old_activation": z_old.detach().mean(),
    }
    return loss, metrics


def evaluate_protected(
    *,
    method: str,
    old_count: int,
    overlap: float,
    free: float,
    protected: float,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    max_update_norm: float,
    lambda_protect: float,
    lambda_ridge: float,
) -> ConstructorEval:
    delta = protected_weight_delta(
        w_base,
        old_keys,
        k_new,
        v_new,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    budgeted, raw_norm, budgeted_norm, scale = apply_budget(delta, max_update_norm)
    old_before = scalar(mse_columns(w_base @ old_keys, old_values).mean())
    new_before = scalar(mse_columns(w_base @ k_new, v_new).mean())
    w_after = w_base + budgeted
    old_after = scalar(mse_columns(w_after @ old_keys, old_values).mean())
    new_after = scalar(mse_columns(w_after @ k_new, v_new).mean())
    return ConstructorEval(
        method=method,
        old_count=old_count,
        overlap=overlap,
        free_room_ratio=free,
        protected_overlap_ratio=protected,
        new_gain_fraction=(new_before - new_after) / (new_before + 1e-12),
        old_damage=old_after - old_before,
        update_norm_unclipped=raw_norm,
        update_norm_after_budget=budgeted_norm,
        budget_scale=scale,
        new_activation=0.0,
        old_activation_rms=0.0,
        old_activation_max=0.0,
        direction_cosine_to_new=0.0,
        threshold=0.0,
    )


def evaluate_route(
    *,
    method: str,
    old_count: int,
    overlap: float,
    free: float,
    protected: float,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    direction: torch.Tensor,
    threshold: torch.Tensor,
    max_update_norm: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
) -> ConstructorEval:
    z_old, z_new = route_activations(direction, threshold, old_keys, k_new)
    delta = route_delta(
        w_base,
        old_keys,
        k_new,
        v_new,
        z_old,
        z_new,
        lambda_old=route_lambda_old,
        lambda_ridge=route_lambda_ridge,
    )
    budgeted, raw_norm, budgeted_norm, scale = apply_budget(delta, max_update_norm)
    old_before = scalar(mse_columns(w_base @ old_keys, old_values).mean())
    new_before = scalar(mse_columns(w_base @ k_new, v_new).mean())
    old_after = scalar(mse_columns(w_base @ old_keys + budgeted @ z_old, old_values).mean())
    new_after = scalar(mse_columns(w_base @ k_new + budgeted @ z_new, v_new).mean())
    k_dir = F.normalize(k_new.T, dim=-1)
    return ConstructorEval(
        method=method,
        old_count=old_count,
        overlap=overlap,
        free_room_ratio=free,
        protected_overlap_ratio=protected,
        new_gain_fraction=(new_before - new_after) / (new_before + 1e-12),
        old_damage=old_after - old_before,
        update_norm_unclipped=raw_norm,
        update_norm_after_budget=budgeted_norm,
        budget_scale=scale,
        new_activation=scalar(z_new.squeeze()),
        old_activation_rms=scalar(torch.sqrt((z_old**2).mean())),
        old_activation_max=scalar(z_old.max()),
        direction_cosine_to_new=scalar((direction * k_dir).sum(dim=1).mean()),
        threshold=scalar(threshold.mean()),
    )


def training_batch(
    *,
    key_dim: int,
    old_count_choices: list[int],
    overlap_choices: list[float],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    target_directions: list[torch.Tensor] = []
    target_thresholds: list[torch.Tensor] = []
    old_keys_blocks: list[torch.Tensor] = []
    for _ in range(batch_size):
        old_count = old_count_choices[int(torch.randint(0, len(old_count_choices), ()).item())]
        overlap = overlap_choices[int(torch.randint(0, len(overlap_choices), ()).item())]
        old_keys, _, k_new, _, protected, free = make_case(
            key_dim=key_dim,
            value_dim=1,
            old_count=old_count,
            overlap=overlap,
            device=device,
            dtype=dtype,
            eps=eps,
        )
        direction, threshold = oracle_route(old_keys, k_new)
        features.append(make_features(old_keys, k_new, free, protected))
        target_directions.append(direction)
        target_thresholds.append(threshold)
        old_keys_blocks.append(old_keys)

    return (
        torch.cat(features, dim=0),
        torch.cat(target_directions, dim=0),
        torch.cat(target_thresholds, dim=0),
        torch.tensor([block.shape[1] for block in old_keys_blocks], device=device),
    )


def train_constructor(
    *,
    model: RouteConstructor,
    key_dim: int,
    old_count_choices: list[int],
    overlap_choices: list[float],
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, steps + 1):
        features, direction_target, threshold_target, _ = training_batch(
            key_dim=key_dim,
            old_count_choices=old_count_choices,
            overlap_choices=overlap_choices,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            eps=eps,
        )
        direction, threshold = model(features)
        direction_loss = 1.0 - (direction * direction_target).sum(dim=1).mean()
        threshold_loss = F.mse_loss(threshold, threshold_target)
        loss = direction_loss + threshold_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "loss": scalar(loss),
                    "direction_loss": scalar(direction_loss),
                    "threshold_loss": scalar(threshold_loss),
                }
            )
    return history


def train_constructor_utility(
    *,
    model: RouteConstructor,
    key_dim: int,
    value_dim: int,
    old_count_choices: list[int],
    overlap_choices: list[float],
    steps: int,
    batch_size: int,
    lr: float,
    max_update_norm: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
    old_weight: float,
    budget_weight: float,
    old_gate_weight: float,
    train_temperature: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, steps + 1):
        losses: list[torch.Tensor] = []
        metric_sums: dict[str, torch.Tensor] = {}
        for _ in range(batch_size):
            old_count = old_count_choices[int(torch.randint(0, len(old_count_choices), ()).item())]
            overlap = overlap_choices[int(torch.randint(0, len(overlap_choices), ()).item())]
            loss, metrics = utility_training_loss_for_case(
                model=model,
                key_dim=key_dim,
                value_dim=value_dim,
                old_count=old_count,
                overlap=overlap,
                max_update_norm=max_update_norm,
                route_lambda_old=route_lambda_old,
                route_lambda_ridge=route_lambda_ridge,
                old_weight=old_weight,
                budget_weight=budget_weight,
                old_gate_weight=old_gate_weight,
                train_temperature=train_temperature,
                device=device,
                dtype=dtype,
                eps=eps,
            )
            losses.append(loss)
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value
        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            item = {"step": step, "loss": scalar(batch_loss)}
            for name, total in metric_sums.items():
                item[name] = scalar(total / batch_size)
            history.append(item)
    return history


def run_eval(
    *,
    model: RouteConstructor,
    key_dim: int,
    value_dim: int,
    old_counts: list[int],
    overlaps: list[float],
    seed: int,
    seeds: int,
    max_update_norm: float,
    lambda_protect: float,
    lambda_ridge: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> list[ConstructorEval]:
    results: list[ConstructorEval] = []
    model.eval()
    with torch.no_grad():
        for seed_offset in range(seeds):
            set_seed(seed + seed_offset)
            for old_count in old_counts:
                for overlap in overlaps:
                    old_keys, old_values, k_new, v_new, protected, free = make_case(
                        key_dim=key_dim,
                        value_dim=value_dim,
                        old_count=old_count,
                        overlap=overlap,
                        device=device,
                        dtype=dtype,
                        eps=eps,
                    )
                    w_base = base_weight(old_keys, old_values)
                    results.append(
                        evaluate_protected(
                            method="protected_budget",
                            old_count=old_count,
                            overlap=overlap,
                            free=free,
                            protected=protected,
                            w_base=w_base,
                            old_keys=old_keys,
                            old_values=old_values,
                            k_new=k_new,
                            v_new=v_new,
                            max_update_norm=max_update_norm,
                            lambda_protect=lambda_protect,
                            lambda_ridge=lambda_ridge,
                        )
                    )
                    oracle_direction, oracle_threshold = oracle_route(old_keys, k_new)
                    results.append(
                        evaluate_route(
                            method="constructive_relu",
                            old_count=old_count,
                            overlap=overlap,
                            free=free,
                            protected=protected,
                            w_base=w_base,
                            old_keys=old_keys,
                            old_values=old_values,
                            k_new=k_new,
                            v_new=v_new,
                            direction=oracle_direction,
                            threshold=oracle_threshold,
                            max_update_norm=max_update_norm,
                            route_lambda_old=route_lambda_old,
                            route_lambda_ridge=route_lambda_ridge,
                        )
                    )
                    features = make_features(old_keys, k_new, free, protected)
                    learned_direction, learned_threshold = model(features)
                    results.append(
                        evaluate_route(
                            method="learned_relu",
                            old_count=old_count,
                            overlap=overlap,
                            free=free,
                            protected=protected,
                            w_base=w_base,
                            old_keys=old_keys,
                            old_values=old_values,
                            k_new=k_new,
                            v_new=v_new,
                            direction=learned_direction,
                            threshold=learned_threshold,
                            max_update_norm=max_update_norm,
                            route_lambda_old=route_lambda_old,
                            route_lambda_ridge=route_lambda_ridge,
                        )
                    )
    return results


def aggregate(results: list[ConstructorEval]) -> list[dict[str, float | int | str]]:
    buckets: dict[tuple[int, float, str], list[ConstructorEval]] = {}
    for result in results:
        buckets.setdefault((result.old_count, result.overlap, result.method), []).append(result)

    rows: list[dict[str, float | int | str]] = []
    for (old_count, overlap, method), bucket in sorted(buckets.items()):
        def avg(name: str) -> float:
            return sum(float(getattr(item, name)) for item in bucket) / len(bucket)

        rows.append(
            {
                "old_count": old_count,
                "overlap": overlap,
                "method": method,
                "free_room_ratio": avg("free_room_ratio"),
                "new_gain_fraction": avg("new_gain_fraction"),
                "old_damage": avg("old_damage"),
                "update_norm_after_budget": avg("update_norm_after_budget"),
                "budget_scale": avg("budget_scale"),
                "new_activation": avg("new_activation"),
                "old_activation_rms": avg("old_activation_rms"),
                "old_activation_max": avg("old_activation_max"),
                "direction_cosine_to_new": avg("direction_cosine_to_new"),
                "threshold": avg("threshold"),
            }
        )
    return rows


def print_explanation() -> None:
    print()
    print("GCO LEARNED ROUTE-CONSTRUCTOR TEST")
    print("=" * 112)
    print("Question:")
    print("  Can a small neural reasoner learn to grow the gated route that separates")
    print("  a new key from old protected keys?")
    print()
    print("Route:")
    print("  phi(k) = relu(r_hat^T k - b_hat)")
    print()
    print("Good route:")
    print("  activates on k_new, stays silent on K_old, and enables a budgeted write.")
    print("=" * 112)


def print_summary(rows: list[dict[str, float | int | str]], history: list[dict[str, float]]) -> None:
    if history:
        print()
        print("Training trace")
        print("-" * 112)
        for item in history:
            metric_text = " ".join(
                f"{name}={value:.5f}"
                for name, value in item.items()
                if name != "step"
            )
            print(f"step={item['step']:>6d} {metric_text}")
    print()
    print("Readable aggregate summary")
    print("-" * 140)
    print(
        "old overlap free  method              gain_frac old_damage  upd_norm scale  "
        "new_act old_rms old_max cos_new threshold"
    )
    print("-" * 140)
    for row in rows:
        print(
            f"{int(row['old_count']):3d} "
            f"{float(row['overlap']):7.2f} "
            f"{float(row['free_room_ratio']):4.2f}  "
            f"{str(row['method']):<19s} "
            f"{float(row['new_gain_fraction']):9.3f} "
            f"{float(row['old_damage']):10.3g} "
            f"{float(row['update_norm_after_budget']):8.3g} "
            f"{float(row['budget_scale']):5.3f} "
            f"{float(row['new_activation']):7.3f} "
            f"{float(row['old_activation_rms']):7.3f} "
            f"{float(row['old_activation_max']):7.3f} "
            f"{float(row['direction_cosine_to_new']):7.3f} "
            f"{float(row['threshold']):9.3f}"
        )
    print("-" * 140)
    print()
    print("What to look for:")
    print("  learned_relu close to constructive_relu means the route-constructor learned the grow rule.")
    print("  old_damage near 0 means the learned route is isolated from protected keys.")
    print("  cos_new near 1 means the learned direction aligns to the new key.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=16)
    parser.add_argument("--old-counts", type=str, default="8,24,48")
    parser.add_argument("--overlaps", type=str, default="0.0,0.25,0.5,0.75,0.9,0.97")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-objective", choices=("supervised", "utility"), default="supervised")
    parser.add_argument("--direction-anchor", choices=("none", "new_key"), default="none")
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--threshold-bias", type=float, default=0.5)
    parser.add_argument("--max-update-norm", type=float, default=2.0)
    parser.add_argument("--lambda-protect", type=float, default=100.0)
    parser.add_argument("--lambda-ridge", type=float, default=1e-5)
    parser.add_argument("--route-lambda-old", type=float, default=100.0)
    parser.add_argument("--route-lambda-ridge", type=float, default=1e-5)
    parser.add_argument("--utility-old-weight", type=float, default=100.0)
    parser.add_argument("--utility-budget-weight", type=float, default=0.1)
    parser.add_argument("--utility-old-gate-weight", type=float, default=10.0)
    parser.add_argument("--train-temperature", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/analysis/gco-learned-route-constructor-seed0.json"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.eval_seeds <= 0:
        raise ValueError("--eval-seeds must be positive.")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_update_norm <= 0:
        raise ValueError("--max-update-norm must be positive.")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    old_counts = parse_int_list(args.old_counts)
    overlaps = parse_float_list(args.overlaps)
    set_seed(args.seed)

    model = RouteConstructor(
        args.key_dim,
        args.hidden_dim,
        direction_anchor=args.direction_anchor,
        residual_scale=args.residual_scale,
        threshold_bias=args.threshold_bias,
    ).to(device=device, dtype=dtype)
    if args.train_objective == "supervised":
        history = train_constructor(
            model=model,
            key_dim=args.key_dim,
            old_count_choices=old_counts,
            overlap_choices=overlaps,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            dtype=dtype,
            eps=args.eps,
        )
    elif args.train_objective == "utility":
        history = train_constructor_utility(
            model=model,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            old_count_choices=old_counts,
            overlap_choices=overlaps,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            max_update_norm=args.max_update_norm,
            route_lambda_old=args.route_lambda_old,
            route_lambda_ridge=args.route_lambda_ridge,
            old_weight=args.utility_old_weight,
            budget_weight=args.utility_budget_weight,
            old_gate_weight=args.utility_old_gate_weight,
            train_temperature=args.train_temperature,
            device=device,
            dtype=dtype,
            eps=args.eps,
        )
    else:
        raise ValueError(f"Unknown train objective: {args.train_objective}")
    results = run_eval(
        model=model,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        old_counts=old_counts,
        overlaps=overlaps,
        seed=args.seed,
        seeds=args.eval_seeds,
        max_update_norm=args.max_update_norm,
        lambda_protect=args.lambda_protect,
        lambda_ridge=args.lambda_ridge,
        route_lambda_old=args.route_lambda_old,
        route_lambda_ridge=args.route_lambda_ridge,
        device=device,
        dtype=dtype,
        eps=args.eps,
    )
    rows = aggregate(results)

    print_explanation()
    print_summary(rows, history)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "gco_learned_route_constructor",
        "config": {
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "old_counts": old_counts,
            "overlaps": overlaps,
            "seed": args.seed,
            "eval_seeds": args.eval_seeds,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "train_objective": args.train_objective,
            "direction_anchor": args.direction_anchor,
            "residual_scale": args.residual_scale,
            "threshold_bias": args.threshold_bias,
            "max_update_norm": args.max_update_norm,
            "lambda_protect": args.lambda_protect,
            "lambda_ridge": args.lambda_ridge,
            "route_lambda_old": args.route_lambda_old,
            "route_lambda_ridge": args.route_lambda_ridge,
            "utility_old_weight": args.utility_old_weight,
            "utility_budget_weight": args.utility_budget_weight,
            "utility_old_gate_weight": args.utility_old_gate_weight,
            "train_temperature": args.train_temperature,
            "device": str(device),
            "dtype": str(dtype),
        },
        "training_history": history,
        "summary": rows,
        "results": [asdict(result) for result in results],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote_json={args.output}")


if __name__ == "__main__":
    main()
