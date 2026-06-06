#!/usr/bin/env python3
"""Real-book GCO capacity-frontier and controlled-forgetting test.

This experiment asks a narrower question than full continual training:

    When a new residual write arrives, how much safe write gain remains as the
    protected old residual bank grows, and can usage-based decay recover gain?

It uses real residual states from the trained real-book transformer. There is no
synthetic key space. The write is a closed-form local residual update:

    minimize_Delta ||(W + Delta) k_new - v_new||^2
                 + lambda * ||Delta K_old D^(1/2)||_F^2
                 + ridge * ||Delta||_F^2

which gives:

    Delta = e k_new^T (k_new k_new^T + lambda K_old D K_old^T + ridge I)^-1

where e = v_new - W k_new and D is the anchor protection diagonal.

Policies compared:
    full_protect       D = I
    soft_usage_decay   D_i follows current real-data activation usage
    quantile_decay     low-usage anchors get reduced protection
    usefulness_decay   release only if low-use, low-damage, conflicting, replaceable
    adaptive_decay     optimizes D_i for the current write objective
    constrained_adaptive_decay optimizes D_i under explicit old-damage budgets
    rewire_on_constrained_failure tries a sparse residual route only when constrained writing stalls
    no_protect         D = 0, destructive upper bound on new learning
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

GCO_DIR = Path(__file__).resolve().parent
if str(GCO_DIR) not in sys.path:
    sys.path.append(str(GCO_DIR))

from gco_learned_route_constructor import (  # noqa: E402
    apply_budget,
    apply_budget_tensor,
    mse_columns,
    parse_float_list,
    parse_int_list,
    resolve_device,
    scalar,
    set_seed,
)
from gco_residual_route_growth import (  # noqa: E402
    ResidualExample,
    capture_residual_examples_for_chunk,
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
class CapacityCase:
    old_keys: torch.Tensor
    old_values: torch.Tensor
    old_usage: torch.Tensor
    k_new: torch.Tensor
    v_new: torch.Tensor
    old_count: int
    protected_overlap_ratio: float
    free_room_ratio: float
    new_chunk_id: str
    new_position: int
    new_token_loss: float
    new_grad_norm: float


@dataclass(frozen=True)
class CapacityEval:
    policy: str
    release_scale: float
    old_count: int
    overlap_bucket: str
    protected_overlap_ratio: float
    free_room_ratio: float
    mean_protection: float
    min_protection: float
    max_protection: float
    usage_mean: float
    usage_min: float
    usage_max: float
    effective_rank: float
    occupied_rank_fraction: float
    new_gain_fraction: float
    old_damage_mean: float
    old_damage_max: float
    protected_damage_weighted: float | None
    released_damage_weighted: float | None
    release_score_mean: float
    release_score_max: float
    raw_new_gain_fraction: float
    write_capacity_ratio: float
    raw_update_norm: float
    safe_update_norm: float
    safe_update_ratio: float
    budget_scale: float
    rewire_applied: float
    route_new_activation: float
    route_old_activation_rms: float


def positive_float(name: str, value: float) -> None:
    require_finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def bounded_float(name: str, value: float, lower: float, upper: float) -> None:
    require_finite_float(name, value)
    if not (lower <= value <= upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}.")


def collect_examples(
    *,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    chunks: Sequence[dict[str, object]],
    layer_index: int,
    max_seq_len: int,
    window_stride: int,
    sequences_per_chunk: int,
    examples_per_chunk: int,
    min_grad_norm: float,
    device: torch.device,
    eps: float,
) -> list[ResidualExample]:
    examples: list[ResidualExample] = []
    for chunk in chunks:
        examples.extend(
            capture_residual_examples_for_chunk(
                model=model,
                tokenizer=tokenizer,
                chunk=chunk,
                layer_index=layer_index,
                max_seq_len=max_seq_len,
                window_stride=window_stride,
                sequences_per_chunk=sequences_per_chunk,
                max_examples=examples_per_chunk,
                min_grad_norm=min_grad_norm,
                device=device,
                eps=eps,
            )
        )
    examples.sort(key=lambda item: item.score, reverse=True)
    if not examples:
        raise RuntimeError("No residual examples were captured.")
    return examples


def build_usage_scores(old_key_matrix: torch.Tensor, context_key_matrix: torch.Tensor) -> torch.Tensor:
    if old_key_matrix.ndim != 2 or context_key_matrix.ndim != 2:
        raise ValueError("old_key_matrix and context_key_matrix must be matrices.")
    if old_key_matrix.shape[0] != context_key_matrix.shape[0]:
        raise ValueError(
            "old/context key dimensions differ: "
            f"old={old_key_matrix.shape}, context={context_key_matrix.shape}."
        )
    similarities = old_key_matrix.T @ context_key_matrix
    positive = torch.clamp(similarities, min=0.0)
    usage = positive.square().max(dim=1).values
    require_finite_tensor("usage_scores", usage)
    return usage


def select_old_bank(
    *,
    old_examples: Sequence[ResidualExample],
    k_new: torch.Tensor,
    old_count: int,
    mode: str,
) -> list[int]:
    if old_count <= 0:
        raise ValueError("old_count must be positive.")
    if old_count > len(old_examples):
        raise ValueError(f"old_count={old_count} exceeds available old examples {len(old_examples)}.")
    if mode == "global":
        return list(range(old_count))
    if mode == "nearest":
        old_matrix = torch.stack([item.key for item in old_examples], dim=1)
        similarities = (k_new.reshape(1, -1) @ old_matrix).squeeze(0)
        return [int(index) for index in torch.topk(similarities, k=old_count, largest=True).indices.detach().cpu()]
    raise ValueError(f"Unknown --old-bank-mode {mode!r}.")


def build_capacity_cases(
    *,
    old_examples: Sequence[ResidualExample],
    new_examples: Sequence[ResidualExample],
    old_counts: Sequence[int],
    max_cases_per_old_count: int,
    old_bank_mode: str,
    eps: float,
) -> list[CapacityCase]:
    if max_cases_per_old_count <= 0:
        raise ValueError("--cases-per-old-count must be positive.")
    if not old_examples:
        raise ValueError("old_examples cannot be empty.")
    if not new_examples:
        raise ValueError("new_examples cannot be empty.")
    context_keys = torch.stack([item.key for item in new_examples], dim=1)
    cases: list[CapacityCase] = []
    for old_count in old_counts:
        for new_example in list(new_examples[:max_cases_per_old_count]):
            selected_indices = select_old_bank(
                old_examples=old_examples,
                k_new=new_example.key,
                old_count=old_count,
                mode=old_bank_mode,
            )
            old_keys = torch.stack([old_examples[index].key for index in selected_indices], dim=1)
            old_values = torch.stack([old_examples[index].value for index in selected_indices], dim=1)
            old_usage = build_usage_scores(old_keys, context_keys)
            max_overlap = scalar(torch.clamp((new_example.key.reshape(1, -1) @ old_keys).max(), min=0.0, max=1.0))
            protected = float(max_overlap * max_overlap)
            free = float(max(0.0, 1.0 - protected))
            if free < -eps or protected < -eps:
                raise RuntimeError("Invalid free/protected capacity ratio.")
            cases.append(
                CapacityCase(
                    old_keys=old_keys,
                    old_values=old_values,
                    old_usage=old_usage,
                    k_new=new_example.key.reshape(-1, 1),
                    v_new=new_example.value.reshape(-1, 1),
                    old_count=old_count,
                    protected_overlap_ratio=protected,
                    free_room_ratio=free,
                    new_chunk_id=new_example.chunk_id,
                    new_position=int(new_example.position),
                    new_token_loss=float(new_example.token_loss),
                    new_grad_norm=float(new_example.grad_norm),
                )
            )
    if not cases:
        raise RuntimeError("No capacity cases were built.")
    return cases


def protection_weights_for_policy(
    *,
    policy: str,
    usage: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    raw_damage: torch.Tensor,
    decay_floor: float,
    decay_power: float,
    decay_quantile: float,
    usefulness_usage_power: float,
    usefulness_damage_power: float,
    usefulness_conflict_power: float,
    usefulness_replacement_power: float,
    release_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if usage.ndim != 1:
        raise ValueError("usage must be a vector.")
    if raw_damage.ndim != 1:
        raise ValueError("raw_damage must be a vector.")
    if old_keys.ndim != 2 or k_new.ndim != 2:
        raise ValueError("old_keys and k_new must be matrices.")
    if k_new.shape[1] != 1:
        raise ValueError(f"k_new must have one column, got {k_new.shape}.")
    if old_keys.shape[1] != usage.shape[0] or old_keys.shape[1] != raw_damage.shape[0]:
        raise ValueError(
            "old_keys, usage, and raw_damage disagree on anchor count: "
            f"old_keys={old_keys.shape}, usage={usage.shape}, raw_damage={raw_damage.shape}."
        )
    bounded_float("decay_floor", decay_floor, 0.0, 1.0)
    positive_float("decay_power", decay_power)
    bounded_float("decay_quantile", decay_quantile, 0.0, 1.0)
    positive_float("usefulness_usage_power", usefulness_usage_power)
    positive_float("usefulness_damage_power", usefulness_damage_power)
    positive_float("usefulness_conflict_power", usefulness_conflict_power)
    positive_float("usefulness_replacement_power", usefulness_replacement_power)
    positive_float("release_scale", release_scale)
    release_score = torch.zeros_like(usage)
    if policy == "full_protect":
        weights = torch.ones_like(usage)
    elif policy == "no_protect":
        weights = torch.zeros_like(usage)
        release_score = torch.ones_like(usage)
    elif policy == "soft_usage_decay":
        usage_max = scalar(usage.max())
        if usage_max <= eps:
            raise RuntimeError("Cannot compute soft usage decay: maximum usage is near zero.")
        normalized = torch.clamp(usage / usage_max, min=0.0, max=1.0)
        release_score = (1.0 - normalized).pow(decay_power)
        weights = 1.0 - (1.0 - decay_floor) * release_score
    elif policy == "quantile_decay":
        threshold = torch.quantile(usage, decay_quantile)
        release_score = (usage <= threshold).to(dtype=usage.dtype)
        weights = 1.0 - (1.0 - decay_floor) * release_score
    elif policy == "usefulness_decay":
        usage_max = scalar(usage.max())
        if usage_max <= eps:
            raise RuntimeError("Cannot compute usefulness decay: maximum usage is near zero.")
        usage_norm = torch.clamp(usage / usage_max, min=0.0, max=1.0)
        positive_damage = torch.clamp(raw_damage, min=0.0)
        damage_max = scalar(positive_damage.max())
        if damage_max <= eps:
            damage_norm = torch.zeros_like(positive_damage)
        else:
            damage_norm = torch.clamp(positive_damage / damage_max, min=0.0, max=1.0)
        conflict = torch.clamp((old_keys.T @ k_new).squeeze(1).square(), min=0.0, max=1.0)
        pairwise = torch.clamp(old_keys.T @ old_keys, min=0.0).square()
        if pairwise.shape[0] > 1:
            pairwise = pairwise.clone()
            pairwise.fill_diagonal_(0.0)
            replacement = (pairwise * usage_norm.unsqueeze(0)).max(dim=1).values
            replacement_max = scalar(replacement.max())
            if replacement_max <= eps:
                replacement_norm = torch.zeros_like(replacement)
            else:
                replacement_norm = torch.clamp(replacement / replacement_max, min=0.0, max=1.0)
        else:
            replacement_norm = torch.zeros_like(usage)
        release_score = (
            (1.0 - usage_norm).pow(usefulness_usage_power)
            * (1.0 - damage_norm).pow(usefulness_damage_power)
            * conflict.pow(usefulness_conflict_power)
            * replacement_norm.pow(usefulness_replacement_power)
        )
        release_score = torch.clamp(release_scale * release_score, min=0.0, max=1.0)
        weights = 1.0 - (1.0 - decay_floor) * release_score
    else:
        raise ValueError(f"Unknown protection policy {policy!r}.")
    require_finite_tensor(f"protection_weights_{policy}", weights)
    require_finite_tensor(f"release_score_{policy}", release_score)
    if bool(((weights < 0.0) | (weights > 1.0)).any().detach().cpu()):
        raise RuntimeError(f"Protection weights for {policy} escaped [0, 1].")
    if bool(((release_score < 0.0) | (release_score > 1.0)).any().detach().cpu()):
        raise RuntimeError(f"Release scores for {policy} escaped [0, 1].")
    return weights, release_score


def weighted_protected_delta(
    *,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    protection_weights: torch.Tensor,
    lambda_protect: float,
    lambda_ridge: float,
) -> torch.Tensor:
    if protection_weights.ndim != 1:
        raise ValueError("protection_weights must be a vector.")
    if protection_weights.shape[0] != old_keys.shape[1]:
        raise ValueError(
            "Protection weight count does not match old key count: "
            f"weights={protection_weights.shape}, old_keys={old_keys.shape}."
        )
    positive_float("lambda_ridge", lambda_ridge)
    if lambda_protect < 0.0:
        raise ValueError("--lambda-protect must be non-negative.")
    key_dim = k_new.shape[0]
    identity = torch.eye(key_dim, device=k_new.device, dtype=k_new.dtype)
    weighted_old_keys = old_keys * torch.sqrt(protection_weights.clamp_min(0.0)).unsqueeze(0)
    system = (
        k_new @ k_new.T
        + lambda_protect * (weighted_old_keys @ weighted_old_keys.T)
        + lambda_ridge * identity
    )
    right = torch.linalg.solve(system, k_new)
    error = v_new - w_base @ k_new
    delta = error @ right.T
    require_finite_tensor("weighted_protected_delta", delta)
    return delta


def effective_rank_from_keys(old_keys: torch.Tensor, protection_weights: torch.Tensor, eps: float) -> float:
    weighted = old_keys * torch.sqrt(protection_weights.clamp_min(0.0)).unsqueeze(0)
    singular_values = torch.linalg.svdvals(weighted)
    positive = singular_values[singular_values > eps]
    if positive.numel() == 0:
        return 0.0
    probabilities = positive / positive.sum()
    entropy = -(probabilities * torch.log(probabilities.clamp_min(eps))).sum()
    value = scalar(torch.exp(entropy))
    require_finite_float("effective_rank", value)
    return value


def weighted_mean_or_none(values: torch.Tensor, weights: torch.Tensor, eps: float) -> float | None:
    mass = scalar(weights.sum())
    if mass <= eps:
        return None
    result = scalar((values * weights).sum() / weights.sum())
    require_finite_float("weighted_mean", result)
    return result


def normalize_by_max(values: torch.Tensor, eps: float) -> torch.Tensor:
    maximum = scalar(values.max())
    if maximum <= eps:
        return torch.zeros_like(values)
    return torch.clamp(values / maximum, min=0.0, max=1.0)


def replacement_scores(old_keys: torch.Tensor, usage_norm: torch.Tensor, eps: float) -> torch.Tensor:
    if old_keys.ndim != 2:
        raise ValueError("old_keys must be a matrix.")
    if usage_norm.ndim != 1 or usage_norm.shape[0] != old_keys.shape[1]:
        raise ValueError(f"usage_norm must match old key count, got usage={usage_norm.shape}, old={old_keys.shape}.")
    pairwise = torch.clamp(old_keys.T @ old_keys, min=0.0).square()
    if pairwise.shape[0] <= 1:
        return torch.zeros_like(usage_norm)
    pairwise = pairwise.clone()
    pairwise.fill_diagonal_(0.0)
    replacement = (pairwise * usage_norm.unsqueeze(0)).max(dim=1).values
    return normalize_by_max(replacement, eps)


def adaptive_protection_weights(
    *,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    old_usage: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    raw_damage: torch.Tensor,
    initial_weights: torch.Tensor,
    max_update_norm: float,
    lambda_protect: float,
    lambda_ridge: float,
    adaptive_steps: int,
    adaptive_lr: float,
    adaptive_old_weight: float,
    adaptive_max_damage_weight: float,
    adaptive_release_weight: float,
    adaptive_max_damage_multiple: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if adaptive_steps <= 0:
        raise ValueError("--adaptive-steps must be positive.")
    positive_float("adaptive_lr", adaptive_lr)
    if adaptive_old_weight < 0.0:
        raise ValueError("--adaptive-old-weight must be non-negative.")
    if adaptive_max_damage_weight < 0.0:
        raise ValueError("--adaptive-max-damage-weight must be non-negative.")
    if adaptive_release_weight < 0.0:
        raise ValueError("--adaptive-release-weight must be non-negative.")
    positive_float("adaptive_max_damage_multiple", adaptive_max_damage_multiple)
    if initial_weights.shape != old_usage.shape:
        raise ValueError(f"initial_weights shape {initial_weights.shape} must match old_usage {old_usage.shape}.")

    usage_norm = normalize_by_max(old_usage.detach(), eps)
    raw_damage_positive = torch.clamp(raw_damage.detach(), min=0.0)
    damage_norm = normalize_by_max(raw_damage_positive, eps)
    replacement_norm = replacement_scores(old_keys.detach(), usage_norm, eps)
    keep_pressure = usage_norm * damage_norm * (1.0 - replacement_norm)
    if scalar(keep_pressure.max()) <= eps:
        keep_pressure = torch.clamp(usage_norm + damage_norm, min=0.0, max=1.0)
    keep_mass = keep_pressure.sum().detach().clamp_min(eps)
    damage_scale = raw_damage_positive.mean().detach().clamp_min(eps)
    old_before_per = mse_columns(w_base @ old_keys, old_values).detach()
    new_before = mse_columns(w_base @ k_new, v_new).mean().detach().clamp_min(eps)

    init = torch.clamp(initial_weights.detach(), min=eps, max=1.0 - eps)
    logits = torch.log(init / (1.0 - init)).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=adaptive_lr)

    for _step in range(adaptive_steps):
        protection = torch.sigmoid(logits)
        delta = weighted_protected_delta(
            w_base=w_base,
            old_keys=old_keys,
            k_new=k_new,
            v_new=v_new,
            protection_weights=protection,
            lambda_protect=lambda_protect,
            lambda_ridge=lambda_ridge,
        )
        budgeted, _raw_norm, _scale = apply_budget_tensor(delta, max_update_norm, eps)
        new_after = mse_columns(w_base @ k_new + budgeted @ k_new, v_new).mean()
        old_after_per = mse_columns(w_base @ old_keys + budgeted @ old_keys, old_values)
        damage_per = torch.relu(old_after_per - old_before_per)
        release = 1.0 - protection
        normalized_new_after = new_after / new_before
        protected_damage = (damage_per * keep_pressure).sum() / (keep_mass * damage_scale)
        max_damage = damage_per.max() / damage_scale
        max_damage_penalty = torch.relu(max_damage - adaptive_max_damage_multiple).pow(2)
        release_penalty = (release * keep_pressure).sum() / keep_mass
        loss = (
            normalized_new_after
            + adaptive_old_weight * protected_damage
            + adaptive_max_damage_weight * max_damage_penalty
            + adaptive_release_weight * release_penalty
        )
        require_finite_tensor("adaptive_decay_loss", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    protection = torch.sigmoid(logits.detach())
    release_score = 1.0 - protection
    require_finite_tensor("adaptive_decay_protection", protection)
    require_finite_tensor("adaptive_decay_release", release_score)
    return protection, release_score


def damage_profile_for_weights(
    *,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    protection_weights: torch.Tensor,
    max_update_norm: float,
    lambda_protect: float,
    lambda_ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = weighted_protected_delta(
        w_base=w_base,
        old_keys=old_keys,
        k_new=k_new,
        v_new=v_new,
        protection_weights=protection_weights,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    budgeted, _raw_norm, _budgeted_norm, _budget_scale = apply_budget(delta, max_update_norm)
    old_before_per = mse_columns(w_base @ old_keys, old_values)
    old_after_per = mse_columns(w_base @ old_keys + budgeted @ old_keys, old_values)
    new_after = mse_columns(w_base @ k_new + budgeted @ k_new, v_new).mean()
    damage_per = torch.relu(old_after_per - old_before_per)
    require_finite_tensor("damage_profile_damage", damage_per)
    require_finite_tensor("damage_profile_new_after", new_after)
    return damage_per.detach(), new_after.detach(), budgeted.detach()


def constrained_adaptive_protection_weights(
    *,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    old_usage: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    raw_damage: torch.Tensor,
    initial_weights: torch.Tensor,
    reference_weights: torch.Tensor,
    max_update_norm: float,
    lambda_protect: float,
    lambda_ridge: float,
    adaptive_steps: int,
    adaptive_lr: float,
    constrained_violation_weight: float,
    constrained_release_weight: float,
    constrained_mean_budget_multiple: float,
    constrained_max_budget_multiple: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if adaptive_steps <= 0:
        raise ValueError("--adaptive-steps must be positive.")
    positive_float("adaptive_lr", adaptive_lr)
    positive_float("constrained_violation_weight", constrained_violation_weight)
    if constrained_release_weight < 0.0:
        raise ValueError("--constrained-release-weight must be non-negative.")
    positive_float("constrained_mean_budget_multiple", constrained_mean_budget_multiple)
    positive_float("constrained_max_budget_multiple", constrained_max_budget_multiple)
    if initial_weights.shape != old_usage.shape:
        raise ValueError(f"initial_weights shape {initial_weights.shape} must match old_usage {old_usage.shape}.")
    if reference_weights.shape != old_usage.shape:
        raise ValueError(f"reference_weights shape {reference_weights.shape} must match old_usage {old_usage.shape}.")

    reference_damage, _reference_new_after, _reference_delta = damage_profile_for_weights(
        w_base=w_base,
        old_keys=old_keys,
        old_values=old_values,
        k_new=k_new,
        v_new=v_new,
        protection_weights=reference_weights,
        max_update_norm=max_update_norm,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    full_damage, _full_new_after, _full_delta = damage_profile_for_weights(
        w_base=w_base,
        old_keys=old_keys,
        old_values=old_values,
        k_new=k_new,
        v_new=v_new,
        protection_weights=torch.ones_like(reference_weights),
        max_update_norm=max_update_norm,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    mean_budget = (
        constrained_mean_budget_multiple
        * torch.maximum(reference_damage.mean(), full_damage.mean()).detach().clamp_min(eps)
    )
    max_budget = (
        constrained_max_budget_multiple
        * torch.maximum(reference_damage.max(), full_damage.max()).detach().clamp_min(eps)
    )
    usage_norm = normalize_by_max(old_usage.detach(), eps)
    raw_damage_positive = torch.clamp(raw_damage.detach(), min=0.0)
    damage_norm = normalize_by_max(raw_damage_positive, eps)
    replacement_norm = replacement_scores(old_keys.detach(), usage_norm, eps)
    keep_pressure = usage_norm * damage_norm * (1.0 - replacement_norm)
    if scalar(keep_pressure.max()) <= eps:
        keep_pressure = torch.clamp(usage_norm + damage_norm, min=0.0, max=1.0)
    keep_mass = keep_pressure.sum().detach().clamp_min(eps)
    old_before_per = mse_columns(w_base @ old_keys, old_values).detach()
    new_before = mse_columns(w_base @ k_new, v_new).mean().detach().clamp_min(eps)

    init = torch.clamp(initial_weights.detach(), min=eps, max=1.0 - eps)
    logits = torch.log(init / (1.0 - init)).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=adaptive_lr)

    for _step in range(adaptive_steps):
        protection = torch.sigmoid(logits)
        delta = weighted_protected_delta(
            w_base=w_base,
            old_keys=old_keys,
            k_new=k_new,
            v_new=v_new,
            protection_weights=protection,
            lambda_protect=lambda_protect,
            lambda_ridge=lambda_ridge,
        )
        budgeted, _raw_norm, _scale = apply_budget_tensor(delta, max_update_norm, eps)
        new_after = mse_columns(w_base @ k_new + budgeted @ k_new, v_new).mean()
        old_after_per = mse_columns(w_base @ old_keys + budgeted @ old_keys, old_values)
        damage_per = torch.relu(old_after_per - old_before_per)
        mean_violation = torch.relu(damage_per.mean() / mean_budget - 1.0).pow(2)
        max_violation = torch.relu(damage_per.max() / max_budget - 1.0).pow(2)
        release = 1.0 - protection
        release_penalty = (release * keep_pressure).sum() / keep_mass
        loss = (
            new_after / new_before
            + constrained_violation_weight * (mean_violation + max_violation)
            + constrained_release_weight * release_penalty
        )
        require_finite_tensor("constrained_adaptive_decay_loss", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    protection = torch.sigmoid(logits.detach())
    release_score = 1.0 - protection
    require_finite_tensor("constrained_adaptive_decay_protection", protection)
    require_finite_tensor("constrained_adaptive_decay_release", release_score)
    return protection, release_score


def soft_route_activations(
    *,
    direction: torch.Tensor,
    threshold: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_float("rewire_temperature", temperature)
    if direction.ndim != 2 or direction.shape[0] != 1:
        raise ValueError(f"direction must have shape [1, d], got {direction.shape}.")
    if threshold.shape != (1, 1):
        raise ValueError(f"threshold must have shape [1, 1], got {threshold.shape}.")
    if old_keys.ndim != 2 or k_new.ndim != 2 or k_new.shape[1] != 1:
        raise ValueError(f"old_keys and k_new must be [d, n] and [d, 1], got {old_keys.shape}, {k_new.shape}.")
    z_old = F.softplus((direction @ old_keys - threshold) / temperature) * temperature
    z_new = F.softplus((direction @ k_new - threshold) / temperature) * temperature
    require_finite_tensor("soft_rewire_z_old", z_old)
    require_finite_tensor("soft_rewire_z_new", z_new)
    return z_old, z_new


def route_delta_from_activations(
    *,
    w_base: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    lambda_old: float,
    lambda_ridge: float,
) -> torch.Tensor:
    if lambda_old < 0.0:
        raise ValueError("--rewire-lambda-old must be non-negative.")
    positive_float("rewire_lambda_ridge", lambda_ridge)
    if z_old.ndim != 2 or z_new.ndim != 2:
        raise ValueError(f"z_old and z_new must be matrices, got {z_old.shape}, {z_new.shape}.")
    if z_old.shape[0] != z_new.shape[0] or z_new.shape[1] != 1:
        raise ValueError(f"Route activations have incompatible shapes: z_old={z_old.shape}, z_new={z_new.shape}.")
    route_dim = z_new.shape[0]
    identity = torch.eye(route_dim, device=z_new.device, dtype=z_new.dtype)
    system = z_new @ z_new.T + lambda_old * (z_old @ z_old.T) + lambda_ridge * identity
    coeff = torch.linalg.solve(system, z_new)
    error = v_new - w_base @ k_new
    delta = error @ coeff.T
    require_finite_tensor("rewire_route_delta", delta)
    return delta


def optimize_rewire_route(
    *,
    case: CapacityCase,
    policy_label: str,
    release_scale: float,
    constrained_eval: CapacityEval,
    max_update_norm: float,
    lambda_ridge: float,
    base_ridge: float,
    decay_floor: float,
    decay_power: float,
    decay_quantile: float,
    usefulness_usage_power: float,
    usefulness_damage_power: float,
    usefulness_conflict_power: float,
    usefulness_replacement_power: float,
    lambda_protect: float,
    rewire_steps: int,
    rewire_lr: float,
    rewire_temperature: float,
    rewire_lambda_old: float,
    rewire_lambda_ridge: float,
    rewire_violation_weight: float,
    rewire_old_gate_weight: float,
    rewire_mean_budget_multiple: float,
    rewire_max_budget_multiple: float,
    eps: float,
) -> CapacityEval:
    if rewire_steps <= 0:
        raise ValueError("--rewire-steps must be positive.")
    positive_float("rewire_lr", rewire_lr)
    positive_float("rewire_temperature", rewire_temperature)
    if rewire_lambda_old < 0.0:
        raise ValueError("--rewire-lambda-old must be non-negative.")
    positive_float("rewire_lambda_ridge", rewire_lambda_ridge)
    positive_float("rewire_violation_weight", rewire_violation_weight)
    if rewire_old_gate_weight < 0.0:
        raise ValueError("--rewire-old-gate-weight must be non-negative.")
    positive_float("rewire_mean_budget_multiple", rewire_mean_budget_multiple)
    positive_float("rewire_max_budget_multiple", rewire_max_budget_multiple)

    w_base = ridge_base_weight(case.old_keys, case.old_values, base_ridge)
    zero_protection = torch.zeros_like(case.old_usage)
    raw_delta = weighted_protected_delta(
        w_base=w_base,
        old_keys=case.old_keys,
        k_new=case.k_new,
        v_new=case.v_new,
        protection_weights=zero_protection,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    raw_budgeted, _raw_delta_norm, _raw_budgeted_norm, _raw_budget_scale = apply_budget(raw_delta, max_update_norm)
    old_before_per = mse_columns(w_base @ case.old_keys, case.old_values).detach()
    raw_old_after_per = mse_columns(w_base @ case.old_keys + raw_budgeted @ case.old_keys, case.old_values)
    raw_damage_per = raw_old_after_per - old_before_per
    reference_weights, _reference_release_score = protection_weights_for_policy(
        policy="quantile_decay",
        usage=case.old_usage,
        old_keys=case.old_keys,
        k_new=case.k_new,
        raw_damage=raw_damage_per,
        decay_floor=decay_floor,
        decay_power=decay_power,
        decay_quantile=decay_quantile,
        usefulness_usage_power=usefulness_usage_power,
        usefulness_damage_power=usefulness_damage_power,
        usefulness_conflict_power=usefulness_conflict_power,
        usefulness_replacement_power=usefulness_replacement_power,
        release_scale=1.0,
        eps=eps,
    )
    reference_damage, _reference_new_after, _reference_delta = damage_profile_for_weights(
        w_base=w_base,
        old_keys=case.old_keys,
        old_values=case.old_values,
        k_new=case.k_new,
        v_new=case.v_new,
        protection_weights=reference_weights,
        max_update_norm=max_update_norm,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    full_damage, _full_new_after, _full_delta = damage_profile_for_weights(
        w_base=w_base,
        old_keys=case.old_keys,
        old_values=case.old_values,
        k_new=case.k_new,
        v_new=case.v_new,
        protection_weights=torch.ones_like(reference_weights),
        max_update_norm=max_update_norm,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    mean_budget = (
        rewire_mean_budget_multiple
        * torch.maximum(reference_damage.mean(), full_damage.mean()).detach().clamp_min(eps)
    )
    max_budget = (
        rewire_max_budget_multiple
        * torch.maximum(reference_damage.max(), full_damage.max()).detach().clamp_min(eps)
    )
    new_before = mse_columns(w_base @ case.k_new, case.v_new).mean().detach().clamp_min(eps)

    direction_norm = torch.linalg.vector_norm(case.k_new.T)
    if scalar(direction_norm) <= eps:
        raise RuntimeError("Cannot initialize rewire route from a near-zero new key.")
    direction0 = (case.k_new.T / direction_norm).detach()
    old_pre0 = direction0 @ case.old_keys
    new_pre0 = direction0 @ case.k_new
    threshold0 = 0.5 * (old_pre0.max(dim=1, keepdim=True).values + new_pre0)
    raw_direction = direction0.clone().requires_grad_(True)
    threshold = threshold0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw_direction, threshold], lr=rewire_lr)

    for _step in range(rewire_steps):
        route_norm = torch.linalg.vector_norm(raw_direction, dim=1, keepdim=True)
        if scalar(route_norm.detach().min()) <= eps:
            raise RuntimeError("Rewire route direction collapsed to near zero during optimization.")
        direction = raw_direction / route_norm
        z_old, z_new = soft_route_activations(
            direction=direction,
            threshold=threshold,
            old_keys=case.old_keys,
            k_new=case.k_new,
            temperature=rewire_temperature,
        )
        delta = route_delta_from_activations(
            w_base=w_base,
            k_new=case.k_new,
            v_new=case.v_new,
            z_old=z_old,
            z_new=z_new,
            lambda_old=rewire_lambda_old,
            lambda_ridge=rewire_lambda_ridge,
        )
        budgeted, _route_raw_norm, _route_budget_scale = apply_budget_tensor(delta, max_update_norm, eps)
        new_after = mse_columns(w_base @ case.k_new + budgeted @ z_new, case.v_new).mean()
        old_after_per = mse_columns(w_base @ case.old_keys + budgeted @ z_old, case.old_values)
        damage_per = torch.relu(old_after_per - old_before_per)
        mean_violation = torch.relu(damage_per.mean() / mean_budget - 1.0).pow(2)
        max_violation = torch.relu(damage_per.max() / max_budget - 1.0).pow(2)
        old_gate_penalty = z_old.square().mean()
        loss = (
            new_after / new_before
            + rewire_violation_weight * (mean_violation + max_violation)
            + rewire_old_gate_weight * old_gate_penalty
        )
        require_finite_tensor("rewire_route_loss", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    route_norm = torch.linalg.vector_norm(raw_direction.detach(), dim=1, keepdim=True)
    if scalar(route_norm.min()) <= eps:
        raise RuntimeError("Final rewire route direction is near zero.")
    direction = raw_direction.detach() / route_norm
    final_threshold = threshold.detach()
    z_old, z_new = soft_route_activations(
        direction=direction,
        threshold=final_threshold,
        old_keys=case.old_keys,
        k_new=case.k_new,
        temperature=rewire_temperature,
    )
    route_delta = route_delta_from_activations(
        w_base=w_base,
        k_new=case.k_new,
        v_new=case.v_new,
        z_old=z_old,
        z_new=z_new,
        lambda_old=rewire_lambda_old,
        lambda_ridge=rewire_lambda_ridge,
    )
    route_raw_norm = scalar(torch.linalg.vector_norm(route_delta))
    budgeted, _safe_raw_norm, _budgeted_norm, budget_scale = apply_budget(route_delta, max_update_norm)
    raw_norm = scalar(torch.linalg.vector_norm(raw_delta))
    raw_new_after = scalar(mse_columns(w_base @ case.k_new + raw_budgeted @ case.k_new, case.v_new).mean())
    raw_new_gain_fraction = (scalar(new_before) - raw_new_after) / (scalar(new_before) + eps)
    old_after_per = mse_columns(w_base @ case.old_keys + budgeted @ z_old, case.old_values)
    new_after = scalar(mse_columns(w_base @ case.k_new + budgeted @ z_new, case.v_new).mean())
    new_gain_fraction = (scalar(new_before) - new_after) / (scalar(new_before) + eps)
    damage_per = old_after_per - old_before_per
    result = CapacityEval(
        policy=policy_label,
        release_scale=release_scale,
        old_count=case.old_count,
        overlap_bucket=overlap_bucket(case.protected_overlap_ratio),
        protected_overlap_ratio=case.protected_overlap_ratio,
        free_room_ratio=case.free_room_ratio,
        mean_protection=constrained_eval.mean_protection,
        min_protection=constrained_eval.min_protection,
        max_protection=constrained_eval.max_protection,
        usage_mean=constrained_eval.usage_mean,
        usage_min=constrained_eval.usage_min,
        usage_max=constrained_eval.usage_max,
        effective_rank=constrained_eval.effective_rank,
        occupied_rank_fraction=constrained_eval.occupied_rank_fraction,
        new_gain_fraction=new_gain_fraction,
        old_damage_mean=scalar(damage_per.mean()),
        old_damage_max=scalar(damage_per.max()),
        protected_damage_weighted=None,
        released_damage_weighted=None,
        release_score_mean=constrained_eval.release_score_mean,
        release_score_max=constrained_eval.release_score_max,
        raw_new_gain_fraction=raw_new_gain_fraction,
        write_capacity_ratio=new_gain_fraction / (raw_new_gain_fraction + eps),
        raw_update_norm=raw_norm,
        safe_update_norm=route_raw_norm,
        safe_update_ratio=route_raw_norm / (raw_norm + eps),
        budget_scale=budget_scale,
        rewire_applied=1.0,
        route_new_activation=scalar(z_new.mean()),
        route_old_activation_rms=scalar(torch.sqrt(z_old.square().mean())),
    )
    for name, value in asdict(result).items():
        if isinstance(value, float):
            require_finite_float(name, value)
    return result


def evaluate_case(
    *,
    case: CapacityCase,
    policy: str,
    policy_label: str,
    release_scale: float,
    max_update_norm: float,
    lambda_protect: float,
    lambda_ridge: float,
    base_ridge: float,
    decay_floor: float,
    decay_power: float,
    decay_quantile: float,
    usefulness_usage_power: float,
    usefulness_damage_power: float,
    usefulness_conflict_power: float,
    usefulness_replacement_power: float,
    adaptive_steps: int,
    adaptive_lr: float,
    adaptive_old_weight: float,
    adaptive_max_damage_weight: float,
    adaptive_release_weight: float,
    adaptive_max_damage_multiple: float,
    constrained_violation_weight: float,
    constrained_release_weight: float,
    constrained_mean_budget_multiple: float,
    constrained_max_budget_multiple: float,
    rewire_trigger_gain: float,
    rewire_steps: int,
    rewire_lr: float,
    rewire_temperature: float,
    rewire_lambda_old: float,
    rewire_lambda_ridge: float,
    rewire_violation_weight: float,
    rewire_old_gate_weight: float,
    rewire_mean_budget_multiple: float,
    rewire_max_budget_multiple: float,
    eps: float,
) -> CapacityEval:
    if policy == "rewire_on_constrained_failure":
        bounded_float("rewire_trigger_gain", rewire_trigger_gain, 0.0, 1.0)
        constrained_eval = evaluate_case(
            case=case,
            policy="constrained_adaptive_decay",
            policy_label="constrained_for_rewire",
            release_scale=release_scale,
            max_update_norm=max_update_norm,
            lambda_protect=lambda_protect,
            lambda_ridge=lambda_ridge,
            base_ridge=base_ridge,
            decay_floor=decay_floor,
            decay_power=decay_power,
            decay_quantile=decay_quantile,
            usefulness_usage_power=usefulness_usage_power,
            usefulness_damage_power=usefulness_damage_power,
            usefulness_conflict_power=usefulness_conflict_power,
            usefulness_replacement_power=usefulness_replacement_power,
            adaptive_steps=adaptive_steps,
            adaptive_lr=adaptive_lr,
            adaptive_old_weight=adaptive_old_weight,
            adaptive_max_damage_weight=adaptive_max_damage_weight,
            adaptive_release_weight=adaptive_release_weight,
            adaptive_max_damage_multiple=adaptive_max_damage_multiple,
            constrained_violation_weight=constrained_violation_weight,
            constrained_release_weight=constrained_release_weight,
            constrained_mean_budget_multiple=constrained_mean_budget_multiple,
            constrained_max_budget_multiple=constrained_max_budget_multiple,
            rewire_trigger_gain=rewire_trigger_gain,
            rewire_steps=rewire_steps,
            rewire_lr=rewire_lr,
            rewire_temperature=rewire_temperature,
            rewire_lambda_old=rewire_lambda_old,
            rewire_lambda_ridge=rewire_lambda_ridge,
            rewire_violation_weight=rewire_violation_weight,
            rewire_old_gate_weight=rewire_old_gate_weight,
            rewire_mean_budget_multiple=rewire_mean_budget_multiple,
            rewire_max_budget_multiple=rewire_max_budget_multiple,
            eps=eps,
        )
        if constrained_eval.new_gain_fraction >= rewire_trigger_gain:
            return replace(
                constrained_eval,
                policy=policy_label,
                rewire_applied=0.0,
                route_new_activation=0.0,
                route_old_activation_rms=0.0,
            )
        return optimize_rewire_route(
            case=case,
            policy_label=policy_label,
            release_scale=release_scale,
            constrained_eval=constrained_eval,
            max_update_norm=max_update_norm,
            lambda_ridge=lambda_ridge,
            base_ridge=base_ridge,
            decay_floor=decay_floor,
            decay_power=decay_power,
            decay_quantile=decay_quantile,
            usefulness_usage_power=usefulness_usage_power,
            usefulness_damage_power=usefulness_damage_power,
            usefulness_conflict_power=usefulness_conflict_power,
            usefulness_replacement_power=usefulness_replacement_power,
            lambda_protect=lambda_protect,
            rewire_steps=rewire_steps,
            rewire_lr=rewire_lr,
            rewire_temperature=rewire_temperature,
            rewire_lambda_old=rewire_lambda_old,
            rewire_lambda_ridge=rewire_lambda_ridge,
            rewire_violation_weight=rewire_violation_weight,
            rewire_old_gate_weight=rewire_old_gate_weight,
            rewire_mean_budget_multiple=rewire_mean_budget_multiple,
            rewire_max_budget_multiple=rewire_max_budget_multiple,
            eps=eps,
        )

    w_base = ridge_base_weight(case.old_keys, case.old_values, base_ridge)
    zero_protection = torch.zeros_like(case.old_usage)
    raw_delta = weighted_protected_delta(
        w_base=w_base,
        old_keys=case.old_keys,
        k_new=case.k_new,
        v_new=case.v_new,
        protection_weights=zero_protection,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    raw_budgeted, _raw_delta_norm, _raw_budgeted_norm, _raw_budget_scale = apply_budget(raw_delta, max_update_norm)
    old_before_per = mse_columns(w_base @ case.old_keys, case.old_values)
    raw_old_after_per = mse_columns(w_base @ case.old_keys + raw_budgeted @ case.old_keys, case.old_values)
    raw_damage_per = raw_old_after_per - old_before_per
    base_policy = "usefulness_decay" if policy in {"adaptive_decay", "constrained_adaptive_decay"} else policy
    protection_weights, release_score = protection_weights_for_policy(
        policy=base_policy,
        usage=case.old_usage,
        old_keys=case.old_keys,
        k_new=case.k_new,
        raw_damage=raw_damage_per,
        decay_floor=decay_floor,
        decay_power=decay_power,
        decay_quantile=decay_quantile,
        usefulness_usage_power=usefulness_usage_power,
        usefulness_damage_power=usefulness_damage_power,
        usefulness_conflict_power=usefulness_conflict_power,
        usefulness_replacement_power=usefulness_replacement_power,
        release_scale=release_scale,
        eps=eps,
    )
    if policy == "adaptive_decay":
        protection_weights, release_score = adaptive_protection_weights(
            w_base=w_base,
            old_keys=case.old_keys,
            old_values=case.old_values,
            old_usage=case.old_usage,
            k_new=case.k_new,
            v_new=case.v_new,
            raw_damage=raw_damage_per,
            initial_weights=protection_weights,
            max_update_norm=max_update_norm,
            lambda_protect=lambda_protect,
            lambda_ridge=lambda_ridge,
            adaptive_steps=adaptive_steps,
            adaptive_lr=adaptive_lr,
            adaptive_old_weight=adaptive_old_weight,
            adaptive_max_damage_weight=adaptive_max_damage_weight,
            adaptive_release_weight=adaptive_release_weight,
            adaptive_max_damage_multiple=adaptive_max_damage_multiple,
            eps=eps,
        )
    elif policy == "constrained_adaptive_decay":
        reference_weights, _reference_release_score = protection_weights_for_policy(
            policy="quantile_decay",
            usage=case.old_usage,
            old_keys=case.old_keys,
            k_new=case.k_new,
            raw_damage=raw_damage_per,
            decay_floor=decay_floor,
            decay_power=decay_power,
            decay_quantile=decay_quantile,
            usefulness_usage_power=usefulness_usage_power,
            usefulness_damage_power=usefulness_damage_power,
            usefulness_conflict_power=usefulness_conflict_power,
            usefulness_replacement_power=usefulness_replacement_power,
            release_scale=1.0,
            eps=eps,
        )
        protection_weights, release_score = constrained_adaptive_protection_weights(
            w_base=w_base,
            old_keys=case.old_keys,
            old_values=case.old_values,
            old_usage=case.old_usage,
            k_new=case.k_new,
            v_new=case.v_new,
            raw_damage=raw_damage_per,
            initial_weights=protection_weights,
            reference_weights=reference_weights,
            max_update_norm=max_update_norm,
            lambda_protect=lambda_protect,
            lambda_ridge=lambda_ridge,
            adaptive_steps=adaptive_steps,
            adaptive_lr=adaptive_lr,
            constrained_violation_weight=constrained_violation_weight,
            constrained_release_weight=constrained_release_weight,
            constrained_mean_budget_multiple=constrained_mean_budget_multiple,
            constrained_max_budget_multiple=constrained_max_budget_multiple,
            eps=eps,
        )
    safe_delta = weighted_protected_delta(
        w_base=w_base,
        old_keys=case.old_keys,
        k_new=case.k_new,
        v_new=case.v_new,
        protection_weights=protection_weights,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    raw_norm = scalar(torch.linalg.vector_norm(raw_delta))
    safe_norm = scalar(torch.linalg.vector_norm(safe_delta))
    safe_update_ratio = safe_norm / (raw_norm + eps)
    budgeted, _safe_raw_norm, _budgeted_norm, budget_scale = apply_budget(safe_delta, max_update_norm)
    new_before = scalar(mse_columns(w_base @ case.k_new, case.v_new).mean())
    raw_new_after = scalar(mse_columns(w_base @ case.k_new + raw_budgeted @ case.k_new, case.v_new).mean())
    raw_new_gain_fraction = (new_before - raw_new_after) / (new_before + eps)
    w_after = w_base + budgeted
    old_after_per = mse_columns(w_after @ case.old_keys, case.old_values)
    new_after = scalar(mse_columns(w_after @ case.k_new, case.v_new).mean())
    new_gain_fraction = (new_before - new_after) / (new_before + eps)
    damage_per = old_after_per - old_before_per
    old_damage_mean = scalar(damage_per.mean())
    old_damage_max = scalar(damage_per.max())
    released_weights = 1.0 - protection_weights
    effective_rank = effective_rank_from_keys(case.old_keys, protection_weights, eps)
    occupied_rank_fraction = effective_rank / float(case.old_keys.shape[0])
    result = CapacityEval(
        policy=policy_label,
        release_scale=release_scale,
        old_count=case.old_count,
        overlap_bucket=overlap_bucket(case.protected_overlap_ratio),
        protected_overlap_ratio=case.protected_overlap_ratio,
        free_room_ratio=case.free_room_ratio,
        mean_protection=scalar(protection_weights.mean()),
        min_protection=scalar(protection_weights.min()),
        max_protection=scalar(protection_weights.max()),
        usage_mean=scalar(case.old_usage.mean()),
        usage_min=scalar(case.old_usage.min()),
        usage_max=scalar(case.old_usage.max()),
        effective_rank=effective_rank,
        occupied_rank_fraction=occupied_rank_fraction,
        new_gain_fraction=(new_before - new_after) / (new_before + eps),
        old_damage_mean=old_damage_mean,
        old_damage_max=old_damage_max,
        protected_damage_weighted=weighted_mean_or_none(damage_per, protection_weights, eps),
        released_damage_weighted=weighted_mean_or_none(damage_per, released_weights, eps),
        release_score_mean=scalar(release_score.mean()),
        release_score_max=scalar(release_score.max()),
        raw_new_gain_fraction=raw_new_gain_fraction,
        write_capacity_ratio=new_gain_fraction / (raw_new_gain_fraction + eps),
        raw_update_norm=raw_norm,
        safe_update_norm=safe_norm,
        safe_update_ratio=safe_update_ratio,
        budget_scale=budget_scale,
        rewire_applied=0.0,
        route_new_activation=0.0,
        route_old_activation_rms=0.0,
    )
    for name, value in asdict(result).items():
        if isinstance(value, float):
            require_finite_float(name, value)
    return result


def aggregate(rows: Sequence[CapacityEval]) -> list[dict[str, float | int | str | None]]:
    if not rows:
        raise ValueError("Cannot aggregate empty result rows.")
    groups: dict[tuple[int, str, str], list[CapacityEval]] = {}
    for row in rows:
        groups.setdefault((row.old_count, row.overlap_bucket, row.policy), []).append(row)
    output: list[dict[str, float | int | str | None]] = []
    for (old_count, bucket, policy), items in sorted(groups.items()):
        def avg(name: str) -> float:
            values = [float(getattr(item, name)) for item in items]
            for value in values:
                require_finite_float(name, value)
            return float(sum(values) / len(values))

        def avg_optional(name: str) -> float | None:
            values = [getattr(item, name) for item in items]
            present = [float(value) for value in values if value is not None]
            if not present:
                return None
            for value in present:
                require_finite_float(name, value)
            return float(sum(present) / len(present))

        output.append(
            {
                "old_count": old_count,
                "overlap_bucket": bucket,
                "policy": policy,
                "release_scale": avg("release_scale"),
                "case_count": len(items),
                "protected_overlap_ratio": avg("protected_overlap_ratio"),
                "free_room_ratio": avg("free_room_ratio"),
                "mean_protection": avg("mean_protection"),
                "usage_mean": avg("usage_mean"),
                "effective_rank": avg("effective_rank"),
                "occupied_rank_fraction": avg("occupied_rank_fraction"),
                "new_gain_fraction": avg("new_gain_fraction"),
                "old_damage_mean": avg("old_damage_mean"),
                "old_damage_max": avg("old_damage_max"),
                "protected_damage_weighted": avg_optional("protected_damage_weighted"),
                "released_damage_weighted": avg_optional("released_damage_weighted"),
                "release_score_mean": avg("release_score_mean"),
                "release_score_max": avg("release_score_max"),
                "raw_new_gain_fraction": avg("raw_new_gain_fraction"),
                "write_capacity_ratio": avg("write_capacity_ratio"),
                "safe_update_ratio": avg("safe_update_ratio"),
                "budget_scale": avg("budget_scale"),
                "rewire_applied": avg("rewire_applied"),
                "route_new_activation": avg("route_new_activation"),
                "route_old_activation_rms": avg("route_old_activation_rms"),
            }
        )
    return output


def format_optional(value: float | int | str | None, width: int = 10) -> str:
    if value is None:
        return "n/a".rjust(width)
    if isinstance(value, str):
        return value.rjust(width)
    return f"{float(value):{width}.3g}"


def print_summary(rows: Sequence[dict[str, float | int | str | None]]) -> None:
    print("\nReadable capacity-frontier summary")
    print("-" * 200)
    print(
        "old overlap       n  policy                           gain old_dmg max_dmg prot_dmg rel_dmg "
        "gain_ratio upd_ratio mean_P rel_mean rel_max eff_rank occ_rank budget rw r_new r_old"
    )
    print("-" * 200)
    for row in rows:
        print(
            f"{int(row['old_count']):3d} "
            f"{str(row['overlap_bucket']):>11} "
            f"{int(row['case_count']):3d} "
            f"{str(row['policy']):<32} "
            f"{float(row['new_gain_fraction']):5.3f} "
            f"{float(row['old_damage_mean']):7.3g} "
            f"{float(row['old_damage_max']):7.3g} "
            f"{format_optional(row['protected_damage_weighted'], 8)} "
            f"{format_optional(row['released_damage_weighted'], 8)} "
            f"{float(row['write_capacity_ratio']):10.3f} "
            f"{float(row['safe_update_ratio']):10.3f} "
            f"{float(row['mean_protection']):6.3f} "
            f"{float(row['release_score_mean']):8.3f} "
            f"{float(row['release_score_max']):7.3f} "
            f"{float(row['effective_rank']):8.3f} "
            f"{float(row['occupied_rank_fraction']):8.3f} "
            f"{float(row['budget_scale']):6.3f} "
            f"{float(row['rewire_applied']):3.2f} "
            f"{float(row['route_new_activation']):5.3f} "
            f"{float(row['route_old_activation_rms']):5.3f}"
        )
    print("-" * 200)
    print("\nWhat to look for:")
    print("  gain high means the new residual write still has room.")
    print("  old_dmg/max_dmg low means old residual behavior stayed stable.")
    print("  gain_ratio near 0 means plasticity death: protection blocks useful new writing.")
    print("  upd_ratio is constrained update norm / raw update norm; read it with gain_ratio.")
    print("  rw is how often the sparse rewire branch fired; r_new/r_old show new activation and old leakage.")
    print("  usefulness_decay beating full_protect on gain with similar protected damage means reasoned decay recovered capacity.")
    print("  released damage is allowed to rise only if those anchors truly have low current usage.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-real-book-capacity-frontier-seed0.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--old-start", type=int, default=0)
    parser.add_argument("--old-chunk-count", type=int, default=2)
    parser.add_argument("--new-start", type=int, default=2)
    parser.add_argument("--new-chunk-count", type=int, default=1)
    parser.add_argument("--sequences-per-chunk", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=64)
    parser.add_argument("--examples-per-chunk", type=int, default=160)
    parser.add_argument("--min-grad-norm", type=float, default=1e-8)
    parser.add_argument("--old-counts", type=str, default="8,24,48,96")
    parser.add_argument("--cases-per-old-count", type=int, default=96)
    parser.add_argument("--old-bank-mode", choices=["global", "nearest"], default="global")
    parser.add_argument(
        "--policies",
        type=str,
        default="full_protect,soft_usage_decay,quantile_decay,usefulness_decay,adaptive_decay,constrained_adaptive_decay,no_protect",
    )
    parser.add_argument("--max-update-norm", type=float, default=2.0)
    parser.add_argument("--base-ridge", type=float, default=1e-3)
    parser.add_argument("--lambda-protect", type=float, default=10.0)
    parser.add_argument("--lambda-ridge", type=float, default=1e-3)
    parser.add_argument("--decay-floor", type=float, default=0.05)
    parser.add_argument("--decay-power", type=float, default=1.0)
    parser.add_argument("--decay-quantile", type=float, default=0.35)
    parser.add_argument("--usefulness-usage-power", type=float, default=1.0)
    parser.add_argument("--usefulness-damage-power", type=float, default=1.0)
    parser.add_argument("--usefulness-conflict-power", type=float, default=1.0)
    parser.add_argument("--usefulness-replacement-power", type=float, default=1.0)
    parser.add_argument("--usefulness-release-scales", type=str, default="1.0")
    parser.add_argument("--adaptive-steps", type=int, default=40)
    parser.add_argument("--adaptive-lr", type=float, default=0.08)
    parser.add_argument("--adaptive-old-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-max-damage-weight", type=float, default=0.25)
    parser.add_argument("--adaptive-release-weight", type=float, default=0.05)
    parser.add_argument("--adaptive-max-damage-multiple", type=float, default=0.05)
    parser.add_argument("--constrained-violation-weight", type=float, default=25.0)
    parser.add_argument("--constrained-release-weight", type=float, default=0.02)
    parser.add_argument("--constrained-mean-budget-multiple", type=float, default=1.0)
    parser.add_argument("--constrained-max-budget-multiple", type=float, default=1.0)
    parser.add_argument("--rewire-trigger-gain", type=float, default=0.25)
    parser.add_argument("--rewire-steps", type=int, default=80)
    parser.add_argument("--rewire-lr", type=float, default=0.05)
    parser.add_argument("--rewire-temperature", type=float, default=0.05)
    parser.add_argument("--rewire-lambda-old", type=float, default=20.0)
    parser.add_argument("--rewire-lambda-ridge", type=float, default=1e-3)
    parser.add_argument("--rewire-violation-weight", type=float, default=50.0)
    parser.add_argument("--rewire-old-gate-weight", type=float, default=5.0)
    parser.add_argument("--rewire-mean-budget-multiple", type=float, default=1.0)
    parser.add_argument("--rewire-max-budget-multiple", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser


def parse_policies(text: str) -> list[str]:
    policies = [item.strip() for item in text.split(",") if item.strip()]
    if not policies:
        raise ValueError("--policies must contain at least one policy.")
    allowed = {
        "full_protect",
        "soft_usage_decay",
        "quantile_decay",
        "usefulness_decay",
        "adaptive_decay",
        "constrained_adaptive_decay",
        "rewire_on_constrained_failure",
        "no_protect",
    }
    for policy in policies:
        if policy not in allowed:
            raise ValueError(f"Unknown policy {policy!r}; allowed={sorted(allowed)}.")
    return policies


def policy_label(policy: str, release_scale: float, scale_count: int) -> str:
    if (
        policy
        in {"usefulness_decay", "adaptive_decay", "constrained_adaptive_decay", "rewire_on_constrained_failure"}
        and scale_count > 1
    ):
        if policy == "usefulness_decay":
            prefix = "usefulness"
        elif policy == "adaptive_decay":
            prefix = "adaptive"
        elif policy == "constrained_adaptive_decay":
            prefix = "constrained"
        else:
            prefix = "rewire"
        return f"{prefix}_x{release_scale:g}"
    return policy


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    positive_float("max_update_norm", args.max_update_norm)
    positive_float("base_ridge", args.base_ridge)
    positive_float("lambda_ridge", args.lambda_ridge)
    if args.lambda_protect < 0.0:
        raise ValueError("--lambda-protect must be non-negative.")
    bounded_float("rewire_trigger_gain", args.rewire_trigger_gain, 0.0, 1.0)
    if args.rewire_steps <= 0:
        raise ValueError("--rewire-steps must be positive.")
    positive_float("rewire_lr", args.rewire_lr)
    positive_float("rewire_temperature", args.rewire_temperature)
    if args.rewire_lambda_old < 0.0:
        raise ValueError("--rewire-lambda-old must be non-negative.")
    positive_float("rewire_lambda_ridge", args.rewire_lambda_ridge)
    positive_float("rewire_violation_weight", args.rewire_violation_weight)
    if args.rewire_old_gate_weight < 0.0:
        raise ValueError("--rewire-old-gate-weight must be non-negative.")
    positive_float("rewire_mean_budget_multiple", args.rewire_mean_budget_multiple)
    positive_float("rewire_max_budget_multiple", args.rewire_max_budget_multiple)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    old_chunks = select_chunks(chunks, start=args.old_start, count=args.old_chunk_count, name="old chunks")
    new_chunks = select_chunks(chunks, start=args.new_start, count=args.new_chunk_count, name="new chunks")
    old_chunk_ids = {str(chunk["chunk_id"]) for chunk in old_chunks}
    new_chunk_ids = {str(chunk["chunk_id"]) for chunk in new_chunks}
    if old_chunk_ids & new_chunk_ids:
        raise ValueError(f"Old and new chunk selections overlap: {sorted(old_chunk_ids & new_chunk_ids)}")
    model = instantiate_base_model(args, tokenizer.get_vocab_size(), device)
    model.to(dtype=dtype)
    layer_index = resolve_layer_index(args.layer_index, len(model.blocks))
    old_counts = parse_int_list(args.old_counts)
    policies = parse_policies(args.policies)
    usefulness_release_scales = parse_float_list(args.usefulness_release_scales)
    for release_scale in usefulness_release_scales:
        positive_float("usefulness_release_scale", release_scale)

    print("GCO REAL-BOOK CAPACITY FRONTIER")
    print("=" * 112)
    print("Question:")
    print("  As old protected residual anchors fill the space, how much safe write gain remains?")
    print("  Does usage-based decay recover gain without damaging still-protected anchors?")
    print("\nWrite objective:")
    print("  min_Delta ||(W+Delta)k_new-v_new||^2 + lambda||Delta K_old D^(1/2)||_F^2 + ridge||Delta||_F^2")
    print("\nResidual source:")
    print(
        f"  layer_index={layer_index}, old_chunks={sorted(old_chunk_ids)}, "
        f"new_chunks={sorted(new_chunk_ids)}, old_bank_mode={args.old_bank_mode}"
    )
    print("=" * 112)

    old_examples = collect_examples(
        model=model,
        tokenizer=tokenizer,
        chunks=old_chunks,
        layer_index=layer_index,
        max_seq_len=args.max_seq_len,
        window_stride=args.window_stride,
        sequences_per_chunk=args.sequences_per_chunk,
        examples_per_chunk=args.examples_per_chunk,
        min_grad_norm=args.min_grad_norm,
        device=device,
        eps=args.eps,
    )
    new_examples = collect_examples(
        model=model,
        tokenizer=tokenizer,
        chunks=new_chunks,
        layer_index=layer_index,
        max_seq_len=args.max_seq_len,
        window_stride=args.window_stride,
        sequences_per_chunk=args.sequences_per_chunk,
        examples_per_chunk=args.examples_per_chunk,
        min_grad_norm=args.min_grad_norm,
        device=device,
        eps=args.eps,
    )
    max_old_count = max(old_counts)
    if max_old_count > len(old_examples):
        raise ValueError(f"Max old_count={max_old_count} exceeds captured old examples {len(old_examples)}.")
    cases = build_capacity_cases(
        old_examples=old_examples,
        new_examples=new_examples,
        old_counts=old_counts,
        max_cases_per_old_count=args.cases_per_old_count,
        old_bank_mode=args.old_bank_mode,
        eps=args.eps,
    )

    eval_rows: list[CapacityEval] = []
    for case in cases:
        for policy in policies:
            release_scales = (
                usefulness_release_scales
                if policy
                in {"usefulness_decay", "adaptive_decay", "constrained_adaptive_decay", "rewire_on_constrained_failure"}
                else [1.0]
            )
            for release_scale in release_scales:
                eval_rows.append(
                    evaluate_case(
                        case=case,
                        policy=policy,
                        policy_label=policy_label(policy, release_scale, len(release_scales)),
                        release_scale=release_scale,
                        max_update_norm=args.max_update_norm,
                        lambda_protect=args.lambda_protect,
                        lambda_ridge=args.lambda_ridge,
                        base_ridge=args.base_ridge,
                        decay_floor=args.decay_floor,
                        decay_power=args.decay_power,
                        decay_quantile=args.decay_quantile,
                        usefulness_usage_power=args.usefulness_usage_power,
                        usefulness_damage_power=args.usefulness_damage_power,
                        usefulness_conflict_power=args.usefulness_conflict_power,
                        usefulness_replacement_power=args.usefulness_replacement_power,
                        adaptive_steps=args.adaptive_steps,
                        adaptive_lr=args.adaptive_lr,
                        adaptive_old_weight=args.adaptive_old_weight,
                        adaptive_max_damage_weight=args.adaptive_max_damage_weight,
                        adaptive_release_weight=args.adaptive_release_weight,
                        adaptive_max_damage_multiple=args.adaptive_max_damage_multiple,
                        constrained_violation_weight=args.constrained_violation_weight,
                        constrained_release_weight=args.constrained_release_weight,
                        constrained_mean_budget_multiple=args.constrained_mean_budget_multiple,
                        constrained_max_budget_multiple=args.constrained_max_budget_multiple,
                        rewire_trigger_gain=args.rewire_trigger_gain,
                        rewire_steps=args.rewire_steps,
                        rewire_lr=args.rewire_lr,
                        rewire_temperature=args.rewire_temperature,
                        rewire_lambda_old=args.rewire_lambda_old,
                        rewire_lambda_ridge=args.rewire_lambda_ridge,
                        rewire_violation_weight=args.rewire_violation_weight,
                        rewire_old_gate_weight=args.rewire_old_gate_weight,
                        rewire_mean_budget_multiple=args.rewire_mean_budget_multiple,
                        rewire_max_budget_multiple=args.rewire_max_budget_multiple,
                        eps=args.eps,
                    )
                )
    aggregate_rows = aggregate(eval_rows)
    print_summary(aggregate_rows)
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
            "old_chunk_ids": sorted(old_chunk_ids),
            "new_chunk_ids": sorted(new_chunk_ids),
            "old_counts": old_counts,
            "old_bank_mode": args.old_bank_mode,
            "policies": policies,
            "cases_per_old_count": args.cases_per_old_count,
            "max_update_norm": args.max_update_norm,
            "base_ridge": args.base_ridge,
            "lambda_protect": args.lambda_protect,
            "lambda_ridge": args.lambda_ridge,
            "decay_floor": args.decay_floor,
            "decay_power": args.decay_power,
            "decay_quantile": args.decay_quantile,
            "usefulness_usage_power": args.usefulness_usage_power,
            "usefulness_damage_power": args.usefulness_damage_power,
            "usefulness_conflict_power": args.usefulness_conflict_power,
            "usefulness_replacement_power": args.usefulness_replacement_power,
            "usefulness_release_scales": usefulness_release_scales,
            "adaptive_steps": args.adaptive_steps,
            "adaptive_lr": args.adaptive_lr,
            "adaptive_old_weight": args.adaptive_old_weight,
            "adaptive_max_damage_weight": args.adaptive_max_damage_weight,
            "adaptive_release_weight": args.adaptive_release_weight,
            "adaptive_max_damage_multiple": args.adaptive_max_damage_multiple,
            "constrained_violation_weight": args.constrained_violation_weight,
            "constrained_release_weight": args.constrained_release_weight,
            "constrained_mean_budget_multiple": args.constrained_mean_budget_multiple,
            "constrained_max_budget_multiple": args.constrained_max_budget_multiple,
            "rewire_trigger_gain": args.rewire_trigger_gain,
            "rewire_steps": args.rewire_steps,
            "rewire_lr": args.rewire_lr,
            "rewire_temperature": args.rewire_temperature,
            "rewire_lambda_old": args.rewire_lambda_old,
            "rewire_lambda_ridge": args.rewire_lambda_ridge,
            "rewire_violation_weight": args.rewire_violation_weight,
            "rewire_old_gate_weight": args.rewire_old_gate_weight,
            "rewire_mean_budget_multiple": args.rewire_mean_budget_multiple,
            "rewire_max_budget_multiple": args.rewire_max_budget_multiple,
        },
        "question": "How does safe residual write capacity change as protected old anchors grow, and can usage decay recover capacity?",
        "old_example_count": len(old_examples),
        "new_example_count": len(new_examples),
        "case_count": len(cases),
        "aggregate": aggregate_rows,
        "eval_records": [asdict(row) for row in eval_rows],
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
