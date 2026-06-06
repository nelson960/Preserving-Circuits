#!/usr/bin/env python3
"""Scratch-train a native GCO transformer with no external optimizer.

This is the first native GCO training loop:

    forward pass
    cross-entropy loss
    backward pass gives local error signals
    each GCO module updates itself from activation, gradient, pressure, topology
    optional same-batch outcome credit measures whether the committed edit helped

There is no torch.optim optimizer. Gradients are used as error information, not
as the update rule. Optional canary windows are held out from each chunk and
used only as behavior diagnostics; they never drive updates.

For a GCO linear matrix W:

    M_t = norm_99(|y_t|^T |x_t|)
    H_t = beta_H H_{t-1} + (1-beta_H)(|G_t| * M_t)
    U_t = beta_U U_{t-1} + (1-beta_U)M_t
    H_norm = H_t / max(H_t)
    P_t = warmup(t) sigmoid(gamma(H_norm - mu))

    G_write = G_t * A_t * M_t
    G_safe = G_write - row_pressure * proj_W(G_write)
    W_{t+1} = decay_unused(W_t) - eta G_safe
    A_{t+1} = clamp(A_t + grow_t - prune_t, 0, 1)

When --outcome-credit-mode same_batch is enabled:

    U_outcome =
      (L_before - L_after) / |L_before|
      - c_edit ||Delta W_safe||
      - c_capacity ||K_write||_0
      - c_rewire ||K_rewire||_0
      - c_forget ||K_forget||_0

The reasoner receives this measured utility as gate credit for the write routes
it actually selected. This keeps pressure as state, not as the teacher.
Developmental maturity suppresses protect/forget/compress during scratch
learning, while failed-write pressure is required before rewire can act.

State labels are pressure regions, not permanent proofs:

    sculpting   P < tau_hardening
    hardening   tau_hardening <= P < tau_crystalline
    crystalline P >= tau_crystalline

Because H and U decay, a circuit can harden when evidence repeats and soften
when the data stops using it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

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

from gco_learned_route_constructor import resolve_device, scalar, set_seed  # noqa: E402
from gco_residual_route_growth import load_chunks, require_finite_float, require_finite_tensor, resolve_dtype  # noqa: E402

ROUTE_EVIDENCE_NAMES = (
    "activation_use",
    "error_pressure",
    "route_frequency",
    "route_recency",
    "pressure_gate",
    "formation",
    "protection_state",
    "decay_state",
    "direct_write_basis",
    "protected_capacity",
    "free_capacity",
    "plastic_capacity",
    "obsolete_capacity",
    "capacity_balance",
    "recurrent_state",
    "topology",
)
REASONER_FEATURE_COUNT = len(ROUTE_EVIDENCE_NAMES)
REASONER_GATE_COUNT = 5
REASONER_ROLE_COUNT = 7
WRITE_GATE_INDEX = 0
PROTECT_GATE_INDEX = 1
REWIRE_GATE_INDEX = 2
FORGET_GATE_INDEX = 3
COMPRESS_GATE_INDEX = 4
SSR_OBSERVATION_COUNT = len(ROUTE_EVIDENCE_NAMES)


@dataclass(frozen=True)
class NativeGCOConfig:
    reasoner_policy: str
    write_mode: str
    direct_write_error_scale: str
    min_active_topology: float
    lr: float
    direct_write_ridge: float
    direct_write_protect: float
    beta_pressure: float
    beta_formation: float
    beta_protection: float
    beta_decay: float
    beta_usage: float
    beta_capacity: float
    beta_recurrent: float
    reasoner_lr: float
    reasoner_weight_decay: float
    reasoner_update_clip: float
    internal_gate_lr_scale: float
    internal_write_gate_lr_scale: float
    internal_protect_gate_lr_scale: float
    internal_rewire_gate_lr_scale: float
    internal_forget_gate_lr_scale: float
    internal_compress_gate_lr_scale: float
    state_space_reasoner_dim: int
    state_space_reasoner_lr: float
    state_space_value_lr: float
    state_space_weight_decay: float
    state_space_update_clip: float
    state_space_init_std: float
    state_space_credit_beta: float
    state_space_write_blend: float
    state_space_protect_blend: float
    state_space_rewire_blend: float
    state_space_forget_blend: float
    state_space_compress_blend: float
    state_space_priority_blend: float
    outcome_credit_lr: float
    outcome_formation_lr: float
    outcome_baseline_beta: float
    outcome_failure_scale: float
    route_credit_scale: float
    route_credit_logit_clip: float
    route_credit_warmup_steps: int
    route_formation_scale: float
    route_formation_logit_clip: float
    formation_weight_mix: float
    formation_row_mix: float
    formation_col_mix: float
    formation_module_mix: float
    formation_multiscale_pooling: str
    positive_utility_write_floor: float
    protect_old_route_floor: float
    protect_collision_strength: float
    hardening_exit_threshold: float
    hardening_protection_strength: float
    structural_protect_strength: float
    outcome_edit_cost: float
    outcome_capacity_cost: float
    outcome_rewire_cost: float
    outcome_forget_cost: float
    failed_write_beta: float
    gamma: float
    mu: float
    warmup_steps: int
    recency_tau: float
    grow_lr: float
    prune_lr: float
    forget_lr: float
    max_step_norm: float
    init_topology: float
    hardening_threshold: float
    crystalline_threshold: float
    pathway_percentile: float
    eps: float


@dataclass(frozen=True)
class GCOModuleStats:
    name: str
    pressure_mean: float
    pressure_max: float
    pathway_mean: float
    pathway_max: float
    error_pressure_mean: float
    error_pressure_max: float
    formation_pressure_mean: float
    formation_pressure_max: float
    formation_effective_mean: float
    formation_effective_max: float
    formation_row_mean: float
    formation_row_max: float
    formation_col_mean: float
    formation_col_max: float
    formation_module_mean: float
    formation_module_max: float
    hardening_latch_mean: float
    hardening_latch_max: float
    ssr_row_state_norm_mean: float
    ssr_row_state_norm_max: float
    ssr_col_state_norm_mean: float
    ssr_col_state_norm_max: float
    ssr_module_state_norm_mean: float
    ssr_module_state_norm_max: float
    ssr_write_gate_mean: float
    ssr_write_gate_max: float
    ssr_protect_gate_mean: float
    ssr_protect_gate_max: float
    ssr_reliability_mean: float
    ssr_reliability_max: float
    ssr_collision_mean: float
    ssr_collision_max: float
    ssr_protect_eff_mean: float
    ssr_protect_eff_max: float
    ssr_rewire_gate_mean: float
    ssr_rewire_gate_max: float
    ssr_forget_gate_mean: float
    ssr_forget_gate_max: float
    ssr_compress_gate_mean: float
    ssr_compress_gate_max: float
    ssr_gain_pred_mean: float
    ssr_gain_pred_max: float
    ssr_capacity_cost_mean: float
    ssr_capacity_cost_max: float
    ssr_forget_safe_mean: float
    ssr_forget_safe_max: float
    ssr_priority_mean: float
    ssr_priority_max: float
    ssr_value_pred_mean: float
    ssr_value_pred_max: float
    ssr_credit_mean: float
    ssr_credit_max: float
    ssr_td_error_abs_mean: float
    ssr_td_error_abs_max: float
    ssr_update_norm: float
    write_pressure_mean: float
    write_pressure_max: float
    protect_need_mean: float
    protect_need_max: float
    protection_pressure_mean: float
    protection_pressure_max: float
    structural_protection_mean: float
    structural_protection_max: float
    structural_input_protection_mean: float
    structural_input_protection_max: float
    direct_write_protect_effective: float
    decay_pressure_mean: float
    decay_pressure_max: float
    direct_write_basis_mean: float
    direct_write_basis_max: float
    route_age_mean: float
    route_age_max: float
    route_recency_mean: float
    route_recency_max: float
    recurrent_state_mean: float
    recurrent_state_abs_mean: float
    recurrent_state_delta_mean: float
    protected_capacity_mean: float
    free_capacity_mean: float
    plastic_capacity_mean: float
    obsolete_capacity_mean: float
    reasoner_role_entropy_mean: float
    reasoner_role_max_share_mean: float
    reasoner_gate_utility_mean: float
    reasoner_gate_error_abs_mean: float
    reasoner_weight_norm: float
    reasoner_update_norm: float
    developmental_maturity: float
    control_phase_gate: float
    failed_write_signal: float
    sculpting_fraction: float
    hardening_fraction: float
    crystalline_fraction: float
    active_sculpting_fraction: float
    active_hardening_fraction: float
    active_crystalline_fraction: float
    active_hardening_latch_fraction: float
    topology_mean: float
    topology_active_fraction: float
    topology_grow_mean: float
    topology_grow_max: float
    topology_prune_mean: float
    topology_prune_max: float
    topology_delta_abs_mean: float
    usage_mean: float
    usage_max: float
    row_pressure_mean: float
    row_pressure_max: float
    write_gate_raw_mean: float
    protect_gate_raw_mean: float
    rewire_gate_raw_mean: float
    forget_gate_raw_mean: float
    compress_gate_raw_mean: float
    write_gate_mean: float
    protect_gate_mean: float
    rewire_gate_mean: float
    forget_gate_mean: float
    compress_gate_mean: float
    write_edit_fraction: float
    protect_edit_fraction: float
    rewire_edit_fraction: float
    forget_edit_fraction: float
    compress_edit_fraction: float
    write_edit_count: float
    protect_edit_count: float
    rewire_edit_count: float
    forget_edit_count: float
    compress_edit_count: float
    write_score_mass: float
    protect_score_mass: float
    rewire_score_mass: float
    forget_score_mass: float
    compress_score_mass: float
    raw_write_norm: float
    direct_write_norm: float
    safe_direction_norm: float
    safe_direction_ratio: float
    projection_removed_ratio: float
    step_scale: float
    safe_update_norm: float
    total_weight_delta_norm: float
    selected_weight_delta_norm: float
    unselected_weight_delta_norm: float
    unselected_weight_delta_max: float
    selected_topology_delta_norm: float
    unselected_topology_delta_norm: float
    unselected_topology_delta_max: float
    forget_rate_mean: float
    forget_rate_max: float
    inactive_mean: float
    inactive_max: float
    weight_norm: float


@dataclass(frozen=True)
class GCOOutcomeCreditStats:
    name: str
    selected_count: float
    eligibility_mass: float
    eligibility_max_share: float
    advantage: float
    route_advantage_mean: float
    route_advantage_abs_mean: float
    route_advantage_max: float
    route_formation_utility_mean: float
    route_formation_utility_max: float
    utility_target: float
    utility_target_min: float
    utility_target_max: float
    write_gate_mean: float
    protect_target_mean: float
    protect_target_min: float
    protect_target_max: float
    protect_gate_mean: float
    protect_error_abs_mean: float
    reliability_target_mean: float
    reliability_target_min: float
    reliability_target_max: float
    reliability_pred_mean: float
    reliability_error_abs_mean: float
    collision_target_mean: float
    collision_target_min: float
    collision_target_max: float
    collision_pred_mean: float
    collision_error_abs_mean: float
    gate_error_abs_mean: float
    update_norm: float


def positive_float(name: str, value: float) -> None:
    require_finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    require_finite_float(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def bounded_float(name: str, value: float, lower: float, upper: float) -> None:
    require_finite_float(name, value)
    if not (lower <= value <= upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}.")


def normalize_pathway(pathway: torch.Tensor, *, percentile: float, eps: float, name: str) -> torch.Tensor:
    if pathway.ndim != 2:
        raise ValueError(f"{name} pathway must be a matrix, got {pathway.shape}.")
    require_finite_tensor(name, pathway)
    bounded_float("pathway_percentile", percentile, 0.0, 1.0)
    if pathway.numel() == 0:
        raise ValueError(f"{name} pathway is empty.")
    scale = torch.quantile(pathway.reshape(-1), percentile).clamp_min(eps)
    result = (pathway / scale).clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_normalized", result)
    return result


def stack_route_evidence(evidence: dict[str, torch.Tensor], *, name: str) -> torch.Tensor:
    missing = [key for key in ROUTE_EVIDENCE_NAMES if key not in evidence]
    if missing:
        raise RuntimeError(f"{name} missing route evidence fields: {missing}.")
    extra = sorted(set(evidence).difference(ROUTE_EVIDENCE_NAMES))
    if extra:
        raise RuntimeError(f"{name} unknown route evidence fields: {extra}.")
    first = evidence[ROUTE_EVIDENCE_NAMES[0]]
    if first.ndim != 2:
        raise ValueError(f"{name} evidence tensors must be matrices, got {first.shape}.")
    tensors: list[torch.Tensor] = []
    for key in ROUTE_EVIDENCE_NAMES:
        tensor = evidence[key]
        if tensor.shape != first.shape:
            raise ValueError(f"{name} evidence {key} shape mismatch: {tensor.shape} vs {first.shape}.")
        require_finite_tensor(f"{name}_{key}", tensor)
        tensors.append(tensor.clamp(0.0, 1.0))
    feature_stack = torch.stack(tensors, dim=-1)
    require_finite_tensor(f"{name}_feature_stack", feature_stack)
    return feature_stack


def selected_route_trace_rows(
    module_name: str,
    context: dict[str, object] | None,
    *,
    limit: int,
) -> list[dict[str, object]]:
    if limit < 0:
        raise ValueError(f"{module_name} route trace limit must be non-negative, got {limit}.")
    if context is None or limit == 0:
        return []
    required = {
        "features",
        "indices",
        "rows",
        "cols",
        "eligibility",
        "state_before",
        "state_after",
        "write_gate_raw",
        "write_gate_eff",
        "protect_gate",
        "rewire_gate",
        "forget_gate",
        "compress_gate",
        "ssr_rewire_gate",
        "ssr_forget_gate",
        "ssr_compress_gate",
        "ssr_gain_pred",
        "ssr_capacity_cost",
        "ssr_forget_safe",
        "ssr_priority",
        "write_priority_control",
        "write_score",
        "protect_score",
        "rewire_score",
        "forget_score",
        "compress_score",
        "weight_before",
        "weight_after",
        "weight_delta",
        "topology_before",
        "topology_after",
        "topology_delta",
        "raw_write",
        "safe_write",
    }
    missing = required.difference(context)
    if missing:
        raise RuntimeError(f"{module_name} route trace context missing keys: {sorted(missing)}.")

    tensors: dict[str, torch.Tensor] = {}
    for key in required:
        value = context[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{module_name} route trace context {key} must be a tensor, got {type(value).__name__}.")
        require_finite_tensor(f"{module_name}_route_trace_{key}", value)
        tensors[key] = value.detach().cpu()

    features = tensors["features"]
    if features.ndim != 2 or features.shape[1] != REASONER_FEATURE_COUNT:
        raise ValueError(
            f"{module_name} route trace features expected [n,{REASONER_FEATURE_COUNT}], got {features.shape}."
        )
    row_count = min(limit, int(features.shape[0]))
    rows: list[dict[str, object]] = []
    for item_index in range(row_count):
        evidence = {
            field: float(features[item_index, field_index].item())
            for field_index, field in enumerate(ROUTE_EVIDENCE_NAMES)
        }
        state_before = float(tensors["state_before"][item_index].item())
        state_after = float(tensors["state_after"][item_index].item())
        weight_before = float(tensors["weight_before"][item_index].item())
        weight_after = float(tensors["weight_after"][item_index].item())
        topology_before = float(tensors["topology_before"][item_index].item())
        topology_after = float(tensors["topology_after"][item_index].item())
        rows.append(
            {
                "module": module_name,
                "route_index": int(tensors["indices"][item_index].item()),
                "row": int(tensors["rows"][item_index].item()),
                "col": int(tensors["cols"][item_index].item()),
                "evidence": evidence,
                "state_before": state_before,
                "state_after": state_after,
                "state_delta": state_after - state_before,
                "gates": {
                    "write_raw": float(tensors["write_gate_raw"][item_index].item()),
                    "write": float(tensors["write_gate_eff"][item_index].item()),
                    "protect": float(tensors["protect_gate"][item_index].item()),
                    "rewire": float(tensors["rewire_gate"][item_index].item()),
                    "forget": float(tensors["forget_gate"][item_index].item()),
                    "compress": float(tensors["compress_gate"][item_index].item()),
                },
                "scores": {
                    "write": float(tensors["write_score"][item_index].item()),
                    "protect": float(tensors["protect_score"][item_index].item()),
                    "rewire": float(tensors["rewire_score"][item_index].item()),
                    "forget": float(tensors["forget_score"][item_index].item()),
                    "compress": float(tensors["compress_score"][item_index].item()),
                },
                "reasoner_predictions": {
                    "rewire_gate": float(tensors["ssr_rewire_gate"][item_index].item()),
                    "forget_gate": float(tensors["ssr_forget_gate"][item_index].item()),
                    "compress_gate": float(tensors["ssr_compress_gate"][item_index].item()),
                    "gain": float(tensors["ssr_gain_pred"][item_index].item()),
                    "capacity_cost": float(tensors["ssr_capacity_cost"][item_index].item()),
                    "forget_safe": float(tensors["ssr_forget_safe"][item_index].item()),
                    "priority": float(tensors["ssr_priority"][item_index].item()),
                    "write_priority_control": float(tensors["write_priority_control"][item_index].item()),
                },
                "eligibility": float(tensors["eligibility"][item_index].item()),
                "weight_before": weight_before,
                "weight_after": weight_after,
                "weight_delta": weight_after - weight_before,
                "topology_before": topology_before,
                "topology_after": topology_after,
                "topology_delta": topology_after - topology_before,
                "raw_write": float(tensors["raw_write"][item_index].item()),
                "safe_write": float(tensors["safe_write"][item_index].item()),
            }
        )
    return rows


def format_route_trace_rows(rows: Sequence[dict[str, object]]) -> str:
    parts: list[str] = []
    for row in rows:
        evidence = row["evidence"]
        gates = row["gates"]
        scores = row["scores"]
        predictions = row["reasoner_predictions"]
        if (
            not isinstance(evidence, dict)
            or not isinstance(gates, dict)
            or not isinstance(scores, dict)
            or not isinstance(predictions, dict)
        ):
            raise TypeError("Route trace row evidence/gates/scores/predictions must be dictionaries.")
        parts.append(
            "{module}[{row},{col}] "
            "act={activation_use:.2f} err={error_pressure:.2f} F={formation:.2f} "
            "P={protection_state:.2f} D={decay_state:.2f} free={free_capacity:.2f} "
            "wg={write_gate:.2f} pg={protect_gate:.2f} "
            "gain/cap/pri={gain:.2f}/{capacity:.2f}/{priority:.2f} ws={write_score:.2f} "
            "elig={eligibility:.2g} dW={weight_delta:+.2e} dA={topology_delta:+.2e} "
            "dS={state_delta:+.2e}".format(
                module=row["module"],
                row=row["row"],
                col=row["col"],
                activation_use=evidence["activation_use"],
                error_pressure=evidence["error_pressure"],
                formation=evidence["formation"],
                protection_state=evidence["protection_state"],
                decay_state=evidence["decay_state"],
                free_capacity=evidence["free_capacity"],
                write_gate=gates["write"],
                protect_gate=gates["protect"],
                gain=predictions["gain"],
                capacity=predictions["capacity_cost"],
                priority=predictions["priority"],
                write_score=scores["write"],
                eligibility=row["eligibility"],
                weight_delta=row["weight_delta"],
                topology_delta=row["topology_delta"],
                state_delta=row["state_delta"],
            )
        )
    return " | ".join(parts)


def fixed_geometric_controls(
    *,
    M: torch.Tensor,
    error_norm: torch.Tensor,
    direct_write_basis: torch.Tensor,
    F_effective: torch.Tensor,
    P_state: torch.Tensor,
    D_state: torch.Tensor,
    U_norm: torch.Tensor,
    route_recency: torch.Tensor,
    old_route_strength: torch.Tensor,
    free_capacity: torch.Tensor,
    plastic_capacity: torch.Tensor,
    obsolete_capacity: torch.Tensor,
    protected_capacity: torch.Tensor,
    A: torch.Tensor,
    failed_write_signal: torch.Tensor,
    cfg: NativeGCOConfig,
    name: str,
) -> dict[str, torch.Tensor]:
    tensors = {
        "M": M,
        "error_norm": error_norm,
        "direct_write_basis": direct_write_basis,
        "F_effective": F_effective,
        "P_state": P_state,
        "D_state": D_state,
        "U_norm": U_norm,
        "route_recency": route_recency,
        "old_route_strength": old_route_strength,
        "free_capacity": free_capacity,
        "plastic_capacity": plastic_capacity,
        "obsolete_capacity": obsolete_capacity,
        "protected_capacity": protected_capacity,
        "A": A,
    }
    for key, tensor in tensors.items():
        if tensor.shape != M.shape:
            raise ValueError(f"{name} fixed reasoner {key} shape mismatch: {tensor.shape} vs {M.shape}.")
        require_finite_tensor(f"{name}_fixed_reasoner_{key}", tensor)

    collision_raw = (old_route_strength * direct_write_basis * M).clamp(0.0, 1.0)
    collision = normalize_pathway(
        collision_raw,
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_fixed_collision",
    )

    reuse_compatibility = (F_effective * U_norm * route_recency * (1.0 - P_state)).clamp(0.0, 1.0)
    novel_write = (
        direct_write_basis
        * error_norm
        * free_capacity
        * plastic_capacity
        * (1.0 - P_state)
        * (1.0 - D_state)
    ).clamp(0.0, 1.0)
    reuse_write = (
        direct_write_basis
        * error_norm
        * reuse_compatibility
        * (1.0 - collision)
    ).clamp(0.0, 1.0)
    write_evidence = torch.maximum(novel_write, reuse_write).clamp(0.0, 1.0)
    write_gate = normalize_pathway(
        write_evidence,
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_fixed_write_gate",
    )

    stable_dependency = (old_route_strength * U_norm * route_recency * (1.0 - error_norm)).clamp(0.0, 1.0)
    protect_evidence = torch.maximum(stable_dependency, collision).clamp(0.0, 1.0)
    protect_gate = normalize_pathway(
        protect_evidence,
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_fixed_protect_gate",
    )

    capacity_pressure = (
        (1.0 - free_capacity)
        * (protected_capacity + obsolete_capacity).clamp(0.0, 1.0)
        * (1.0 - plastic_capacity)
    ).clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_fixed_capacity_pressure", capacity_pressure)

    forget_candidate = (
        D_state
        * (1.0 - M)
        * (1.0 - U_norm)
        * (1.0 - route_recency)
        * (1.0 - old_route_strength)
    ).clamp(0.0, 1.0)
    forget_gate = (
        normalize_pathway(
            forget_candidate,
            percentile=cfg.pathway_percentile,
            eps=cfg.eps,
            name=f"{name}_fixed_forget_gate",
        )
        * capacity_pressure
    ).clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_fixed_forget_gate_capacity_gated", forget_gate)

    forget_evidence = (forget_candidate * capacity_pressure).clamp(0.0, 1.0)

    forget_safe = normalize_pathway(
        forget_evidence,
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_fixed_forget_safe",
    )

    rewire_evidence = (
        failed_write_signal
        * collision
        * torch.maximum(free_capacity, obsolete_capacity)
        * (1.0 - protect_gate)
    ).clamp(0.0, 1.0)
    rewire_gate = normalize_pathway(
        rewire_evidence,
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_fixed_rewire_gate",
    )

    compress_candidate = (
        P_state
        * U_norm
        * route_recency
        * (1.0 - error_norm)
        * (1.0 - collision)
    ).clamp(0.0, 1.0)
    compress_evidence = (compress_candidate * capacity_pressure).clamp(0.0, 1.0)
    compress_gate = (
        normalize_pathway(
            compress_candidate,
            percentile=cfg.pathway_percentile,
            eps=cfg.eps,
            name=f"{name}_fixed_compress_gate",
        )
        * capacity_pressure
    ).clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_fixed_compress_gate_capacity_gated", compress_gate)

    capacity_cost = normalize_pathway(
        capacity_pressure,
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_fixed_capacity_cost",
    )
    gain_pred = normalize_pathway(
        write_evidence,
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_fixed_gain",
    )
    priority = normalize_pathway(
        (write_evidence * (1.0 - collision) * torch.maximum(free_capacity, reuse_compatibility)).clamp(0.0, 1.0),
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_fixed_priority",
    )
    reliability = torch.maximum(stable_dependency, reuse_compatibility).clamp(0.0, 1.0)
    value_pred = (gain_pred - capacity_cost - collision).clamp(-1.0, 1.0)

    controls = {
        "write_gate": write_gate,
        "protect_gate": protect_gate,
        "rewire_gate": rewire_gate,
        "forget_gate": forget_gate,
        "compress_gate": compress_gate,
        "gain_pred": gain_pred,
        "capacity_cost": capacity_cost,
        "forget_safe": forget_safe,
        "priority": priority,
        "collision": collision,
        "reliability": reliability,
        "protect_eff": (protect_gate * reliability * collision).clamp(0.0, 1.0),
        "value_pred": value_pred,
    }
    for key, value in controls.items():
        require_finite_tensor(f"{name}_fixed_control_{key}", value)
    return controls


def native_reasoner_parameter(*shape: int, init_std: float = 0.0) -> nn.Parameter:
    nonnegative_float("native_reasoner_init_std", init_std)
    if init_std > 0.0:
        tensor = torch.empty(*shape)
        nn.init.normal_(tensor, mean=0.0, std=init_std)
    else:
        tensor = torch.zeros(*shape)
    return nn.Parameter(tensor, requires_grad=False)


def multiscale_formation_components(
    evidence: torch.Tensor,
    activity: torch.Tensor,
    *,
    pooling: str,
    eps: float,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if evidence.ndim != 2 or activity.ndim != 2:
        raise ValueError(f"{name} evidence/activity must be matrices, got {evidence.shape} and {activity.shape}.")
    if evidence.shape != activity.shape:
        raise ValueError(f"{name} evidence/activity shape mismatch: {evidence.shape} vs {activity.shape}.")
    require_finite_tensor(f"{name}_evidence", evidence)
    require_finite_tensor(f"{name}_activity", activity)
    clipped_evidence = evidence.clamp(0.0, 1.0)
    clipped_activity = activity.clamp(0.0, 1.0)
    if pooling == "mean":
        row_den = clipped_activity.sum(dim=1, keepdim=True).clamp_min(eps)
        col_den = clipped_activity.sum(dim=0, keepdim=True).clamp_min(eps)
        module_den = clipped_activity.sum().reshape(1, 1).clamp_min(eps)
        weighted = clipped_evidence * clipped_activity
        row = (weighted.sum(dim=1, keepdim=True) / row_den).clamp(0.0, 1.0)
        col = (weighted.sum(dim=0, keepdim=True) / col_den).clamp(0.0, 1.0)
        module = (weighted.sum().reshape(1, 1) / module_den).clamp(0.0, 1.0)
    elif pooling == "max":
        row = clipped_evidence.max(dim=1, keepdim=True).values
        col = clipped_evidence.max(dim=0, keepdim=True).values
        module = clipped_evidence.max().reshape(1, 1)
    else:
        raise ValueError(f"{name} unknown formation pooling mode: {pooling!r}.")
    require_finite_tensor(f"{name}_row", row)
    require_finite_tensor(f"{name}_col", col)
    require_finite_tensor(f"{name}_module", module)
    return row, col, module


def combine_multiscale_formation(
    F_state: torch.Tensor,
    F_row: torch.Tensor,
    F_col: torch.Tensor,
    F_module: torch.Tensor,
    cfg: NativeGCOConfig,
    *,
    name: str,
) -> torch.Tensor:
    if F_state.ndim != 2:
        raise ValueError(f"{name} F_state must be a matrix, got {F_state.shape}.")
    if F_row.shape != (F_state.shape[0], 1):
        raise ValueError(f"{name} F_row shape mismatch: expected {(F_state.shape[0], 1)}, got {F_row.shape}.")
    if F_col.shape != (1, F_state.shape[1]):
        raise ValueError(f"{name} F_col shape mismatch: expected {(1, F_state.shape[1])}, got {F_col.shape}.")
    if F_module.shape != (1, 1):
        raise ValueError(f"{name} F_module shape mismatch: expected {(1, 1)}, got {F_module.shape}.")
    total = cfg.formation_weight_mix + cfg.formation_row_mix + cfg.formation_col_mix + cfg.formation_module_mix
    positive_float(f"{name}_formation_mix_total", total)
    combined = (
        cfg.formation_weight_mix * F_state
        + cfg.formation_row_mix * F_row
        + cfg.formation_col_mix * F_col
        + cfg.formation_module_mix * F_module
    ) / total
    combined = combined.clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_formation_effective", combined)
    return combined


def reduce_route_observations(
    feature_stack: torch.Tensor,
    activity: torch.Tensor,
    *,
    pooling: str,
    eps: float,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if feature_stack.ndim != 3:
        raise ValueError(f"{name} feature_stack must have shape [out,in,features], got {feature_stack.shape}.")
    if feature_stack.shape[-1] != SSR_OBSERVATION_COUNT:
        raise ValueError(
            f"{name} expected {SSR_OBSERVATION_COUNT} observation features, got {feature_stack.shape[-1]}."
        )
    if activity.shape != feature_stack.shape[:2]:
        raise ValueError(f"{name} activity shape mismatch: {activity.shape} vs {feature_stack.shape[:2]}.")
    require_finite_tensor(f"{name}_feature_stack", feature_stack)
    require_finite_tensor(f"{name}_activity", activity)
    clipped_features = feature_stack.clamp(0.0, 1.0)
    clipped_activity = activity.clamp(0.0, 1.0)
    if pooling == "mean":
        row_den = clipped_activity.sum(dim=1, keepdim=True).clamp_min(eps)
        col_den = clipped_activity.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(eps)
        module_den = clipped_activity.sum().reshape(1, 1).clamp_min(eps)
        weighted = clipped_features * clipped_activity.unsqueeze(-1)
        row_obs = weighted.sum(dim=1) / row_den
        col_obs = weighted.sum(dim=0) / col_den
        module_obs = weighted.sum(dim=(0, 1), keepdim=False).reshape(1, -1) / module_den
    elif pooling == "max":
        row_obs = clipped_features.max(dim=1).values
        col_obs = clipped_features.max(dim=0).values
        module_obs = clipped_features.reshape(-1, SSR_OBSERVATION_COUNT).max(dim=0).values.reshape(1, -1)
    else:
        raise ValueError(f"{name} unknown route observation pooling mode: {pooling!r}.")
    row_obs = row_obs.clamp(0.0, 1.0)
    col_obs = col_obs.clamp(0.0, 1.0)
    module_obs = module_obs.clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_row_obs", row_obs)
    require_finite_tensor(f"{name}_col_obs", col_obs)
    require_finite_tensor(f"{name}_module_obs", module_obs)
    return row_obs, col_obs, module_obs


def state_space_route_step(
    route_state: torch.Tensor,
    route_obs: torch.Tensor,
    route_credit: torch.Tensor,
    state_state_weights: torch.Tensor,
    state_obs_weights: torch.Tensor,
    state_credit_weights: torch.Tensor,
    state_bias: torch.Tensor,
    write_head: torch.Tensor,
    write_bias: torch.Tensor,
    protect_head: torch.Tensor,
    protect_bias: torch.Tensor,
    reliability_head: torch.Tensor,
    reliability_bias: torch.Tensor,
    collision_head: torch.Tensor,
    collision_bias: torch.Tensor,
    value_head: torch.Tensor,
    value_bias: torch.Tensor,
    *,
    name: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if route_state.ndim != 2 or route_obs.ndim != 2 or route_credit.ndim != 2:
        raise ValueError(
            f"{name} route_state/obs/credit must be matrices, got "
            f"{route_state.shape}, {route_obs.shape}, {route_credit.shape}."
        )
    if route_state.shape[0] != route_obs.shape[0] or route_state.shape[0] != route_credit.shape[0]:
        raise ValueError(
            f"{name} route count mismatch: state={route_state.shape}, obs={route_obs.shape}, credit={route_credit.shape}."
        )
    if route_obs.shape[1] != SSR_OBSERVATION_COUNT:
        raise ValueError(f"{name} route_obs expected {SSR_OBSERVATION_COUNT} features, got {route_obs.shape[1]}.")
    state_dim = route_state.shape[1]
    if route_credit.shape[1] != 1:
        raise ValueError(f"{name} route_credit must have shape [n,1], got {route_credit.shape}.")
    if state_state_weights.shape != (state_dim, state_dim):
        raise ValueError(f"{name} state_state_weights shape mismatch: {state_state_weights.shape}.")
    if state_obs_weights.shape != (state_dim, SSR_OBSERVATION_COUNT):
        raise ValueError(f"{name} state_obs_weights shape mismatch: {state_obs_weights.shape}.")
    if state_credit_weights.shape != (state_dim,):
        raise ValueError(f"{name} state_credit_weights shape mismatch: {state_credit_weights.shape}.")
    if state_bias.shape != (state_dim,):
        raise ValueError(f"{name} state_bias shape mismatch: {state_bias.shape}.")
    z_dim = state_dim + SSR_OBSERVATION_COUNT
    if (
        write_head.shape != (z_dim,)
        or protect_head.shape != (z_dim,)
        or reliability_head.shape != (z_dim,)
        or collision_head.shape != (z_dim,)
        or value_head.shape != (z_dim,)
    ):
        raise ValueError(
            f"{name} write/protect/reliability/collision/value head shape mismatch: "
            f"{write_head.shape}, {protect_head.shape}, {reliability_head.shape}, "
            f"{collision_head.shape}, {value_head.shape}, z={z_dim}."
        )
    if (
        write_bias.shape != (1,)
        or protect_bias.shape != (1,)
        or reliability_bias.shape != (1,)
        or collision_bias.shape != (1,)
        or value_bias.shape != (1,)
    ):
        raise ValueError(
            f"{name} write/protect/reliability/collision/value bias shape mismatch: "
            f"{write_bias.shape}, {protect_bias.shape}, {reliability_bias.shape}, "
            f"{collision_bias.shape}, {value_bias.shape}."
        )
    require_finite_tensor(f"{name}_route_state", route_state)
    require_finite_tensor(f"{name}_route_obs", route_obs)
    require_finite_tensor(f"{name}_route_credit", route_credit)
    next_state = torch.tanh(
        route_state @ state_state_weights.transpose(0, 1)
        + route_obs @ state_obs_weights.transpose(0, 1)
        + route_credit.clamp(-1.0, 1.0) * state_credit_weights.reshape(1, -1)
        + state_bias.reshape(1, -1)
    )
    route_state.copy_(next_state)
    route_z = torch.cat([route_state, route_obs], dim=1)
    write_logit = route_z @ write_head + write_bias[0]
    protect_logit = route_z @ protect_head + protect_bias[0]
    reliability_logit = route_z @ reliability_head + reliability_bias[0]
    collision_logit = route_z @ collision_head + collision_bias[0]
    value_pred = route_z @ value_head + value_bias[0]
    require_finite_tensor(f"{name}_route_z", route_z)
    require_finite_tensor(f"{name}_write_logit", write_logit)
    require_finite_tensor(f"{name}_protect_logit", protect_logit)
    require_finite_tensor(f"{name}_reliability_logit", reliability_logit)
    require_finite_tensor(f"{name}_collision_logit", collision_logit)
    require_finite_tensor(f"{name}_value_pred", value_pred)
    return (
        route_z,
        write_logit,
        torch.sigmoid(write_logit).clamp(0.0, 1.0),
        protect_logit,
        torch.sigmoid(protect_logit).clamp(0.0, 1.0),
        reliability_logit,
        torch.sigmoid(reliability_logit).clamp(0.0, 1.0),
        collision_logit,
        torch.sigmoid(collision_logit).clamp(0.0, 1.0),
        value_pred,
    )


def state_space_sigmoid_head(
    route_z: torch.Tensor,
    head: torch.Tensor,
    bias: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route_z.ndim != 2:
        raise ValueError(f"{name} route_z must be a matrix, got {route_z.shape}.")
    if head.shape != (route_z.shape[1],):
        raise ValueError(f"{name} head shape mismatch: expected {(route_z.shape[1],)}, got {head.shape}.")
    if bias.shape != (1,):
        raise ValueError(f"{name} bias shape mismatch: expected {(1,)}, got {bias.shape}.")
    require_finite_tensor(f"{name}_route_z", route_z)
    require_finite_tensor(f"{name}_head", head)
    require_finite_tensor(f"{name}_bias", bias)
    logit = route_z @ head + bias[0]
    value = torch.sigmoid(logit).clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_logit", logit)
    require_finite_tensor(f"{name}_value", value)
    return logit, value


def combine_state_space_surface(
    row_value: torch.Tensor,
    col_value: torch.Tensor,
    module_value: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if row_value.ndim != 1 or col_value.ndim != 1 or module_value.shape != (1,):
        raise ValueError(
            f"{name} expected row/col/module vectors with module [1], got "
            f"{row_value.shape}, {col_value.shape}, {module_value.shape}."
        )
    surface = (row_value.reshape(-1, 1) + col_value.reshape(1, -1) + module_value.reshape(1, 1)) / 3.0
    surface = surface.clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_surface", surface)
    return surface


def dynamic_edit_mask(score: torch.Tensor, *, eps: float, name: str) -> tuple[torch.Tensor, int, float]:
    require_finite_tensor(name, score)
    if score.ndim != 2:
        raise ValueError(f"{name} score must be a matrix, got {score.shape}.")
    if score.numel() <= 0:
        raise ValueError(f"{name} score is empty.")
    clipped = score.clamp(0.0, 1.0)
    score_mass = scalar(clipped.sum())
    if score_mass <= eps:
        return torch.zeros_like(clipped), 0, score_mass
    edit_count = min(clipped.numel(), max(1, int(math.ceil(score_mass))))
    flat = clipped.reshape(-1)
    values, indices = torch.topk(flat, k=edit_count, largest=True, sorted=False)
    if scalar(values.max()) <= eps:
        return torch.zeros_like(clipped), 0, score_mass
    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(0, indices, 1.0)
    mask = mask_flat.reshape_as(clipped)
    require_finite_tensor(f"{name}_mask", mask)
    return mask, edit_count, score_mass


@torch.no_grad()
def apply_outcome_credit_to_reasoner(
    name: str,
    F_state: torch.Tensor,
    F_row: torch.Tensor,
    F_col: torch.Tensor,
    F_module: torch.Tensor,
    SSR_row_state: torch.Tensor,
    SSR_col_state: torch.Tensor,
    SSR_module_state: torch.Tensor,
    SSR_row_credit: torch.Tensor,
    SSR_col_credit: torch.Tensor,
    SSR_module_credit: torch.Tensor,
    SSR_last_td_error_abs: torch.Tensor,
    SSR_last_update_norm: torch.Tensor,
    reasoner_state_weights: torch.Tensor,
    reasoner_state_bias: torch.Tensor,
    reasoner_gate_weights: torch.Tensor,
    reasoner_gate_state_weights: torch.Tensor,
    reasoner_gate_bias: torch.Tensor,
    ssr_write_head: torch.Tensor,
    ssr_write_bias: torch.Tensor,
    ssr_protect_head: torch.Tensor,
    ssr_protect_bias: torch.Tensor,
    ssr_reliability_head: torch.Tensor,
    ssr_reliability_bias: torch.Tensor,
    ssr_collision_head: torch.Tensor,
    ssr_collision_bias: torch.Tensor,
    ssr_value_head: torch.Tensor,
    ssr_value_bias: torch.Tensor,
    context: dict[str, torch.Tensor] | None,
    cfg: NativeGCOConfig,
    utility: float,
    advantage: float,
    total_eligibility_mass: float,
    total_selected_count: float,
    step: int,
) -> GCOOutcomeCreditStats:
    require_finite_float(f"{name}_outcome_utility", utility)
    require_finite_float(f"{name}_outcome_advantage", advantage)
    nonnegative_float(f"{name}_outcome_total_eligibility_mass", total_eligibility_mass)
    nonnegative_float(f"{name}_outcome_total_selected_count", total_selected_count)
    if step <= 0:
        raise ValueError(f"{name} outcome credit step must be positive, got {step}.")
    if context is None:
        return GCOOutcomeCreditStats(
            name=name,
            selected_count=0.0,
            eligibility_mass=0.0,
            eligibility_max_share=0.0,
            advantage=advantage,
            route_advantage_mean=0.0,
            route_advantage_abs_mean=0.0,
            route_advantage_max=0.0,
            route_formation_utility_mean=0.0,
            route_formation_utility_max=0.0,
            utility_target=0.0,
            utility_target_min=0.0,
            utility_target_max=0.0,
            write_gate_mean=0.0,
            protect_target_mean=0.0,
            protect_target_min=0.0,
            protect_target_max=0.0,
            protect_gate_mean=0.0,
            protect_error_abs_mean=0.0,
            reliability_target_mean=0.0,
            reliability_target_min=0.0,
            reliability_target_max=0.0,
            reliability_pred_mean=0.0,
            reliability_error_abs_mean=0.0,
            collision_target_mean=0.0,
            collision_target_min=0.0,
            collision_target_max=0.0,
            collision_pred_mean=0.0,
            collision_error_abs_mean=0.0,
            gate_error_abs_mean=0.0,
            update_norm=0.0,
        )
    required = {"features", "state", "write_gate", "indices", "eligibility"}
    missing = required.difference(context)
    if missing:
        raise RuntimeError(f"{name} outcome credit context missing keys: {sorted(missing)}.")
    features = context["features"]
    state = context["state"]
    write_gate = context["write_gate"]
    indices = context["indices"]
    eligibility = context["eligibility"]
    if features.ndim != 2 or features.shape[1] != REASONER_FEATURE_COUNT:
        raise ValueError(f"{name} outcome features shape mismatch: {features.shape}.")
    if state.ndim != 1 or write_gate.ndim != 1:
        raise ValueError(f"{name} outcome state/write_gate must be vectors, got {state.shape} and {write_gate.shape}.")
    if indices.ndim != 1:
        raise ValueError(f"{name} outcome indices must be a vector, got {indices.shape}.")
    if eligibility.ndim != 1:
        raise ValueError(f"{name} outcome eligibility must be a vector, got {eligibility.shape}.")
    if (
        features.shape[0] != state.shape[0]
        or features.shape[0] != write_gate.shape[0]
        or features.shape[0] != indices.shape[0]
        or features.shape[0] != eligibility.shape[0]
    ):
        raise ValueError(
            f"{name} outcome context length mismatch: features={features.shape}, "
            f"state={state.shape}, write_gate={write_gate.shape}, "
            f"indices={indices.shape}, eligibility={eligibility.shape}."
        )
    selected_count = features.shape[0]
    if selected_count <= 0:
        raise RuntimeError(f"{name} outcome credit context has no selected entries.")
    require_finite_tensor(f"{name}_outcome_features", features)
    require_finite_tensor(f"{name}_outcome_state", state)
    require_finite_tensor(f"{name}_outcome_write_gate", write_gate)
    require_finite_tensor(f"{name}_outcome_eligibility", eligibility)
    if scalar(eligibility.min()) < 0.0:
        raise ValueError(f"{name} outcome eligibility contains negative values.")
    eligibility_mass = scalar(eligibility.sum())
    if eligibility_mass <= cfg.eps or total_eligibility_mass <= cfg.eps or total_selected_count <= cfg.eps:
        return GCOOutcomeCreditStats(
            name=name,
            selected_count=float(selected_count),
            eligibility_mass=eligibility_mass,
            eligibility_max_share=0.0,
            advantage=advantage,
            route_advantage_mean=0.0,
            route_advantage_abs_mean=0.0,
            route_advantage_max=0.0,
            route_formation_utility_mean=0.0,
            route_formation_utility_max=0.0,
            utility_target=0.0,
            utility_target_min=0.0,
            utility_target_max=0.0,
            write_gate_mean=scalar(write_gate.mean()),
            protect_target_mean=0.0,
            protect_target_min=0.0,
            protect_target_max=0.0,
            protect_gate_mean=0.0,
            protect_error_abs_mean=0.0,
            reliability_target_mean=0.0,
            reliability_target_min=0.0,
            reliability_target_max=0.0,
            reliability_pred_mean=0.0,
            reliability_error_abs_mean=0.0,
            collision_target_mean=0.0,
            collision_target_min=0.0,
            collision_target_max=0.0,
            collision_pred_mean=0.0,
            collision_error_abs_mean=0.0,
            gate_error_abs_mean=0.0,
            update_norm=0.0,
        )
    if F_state.ndim != 2:
        raise ValueError(f"{name} F_state must be a matrix, got {F_state.shape}.")
    if F_row.shape != (F_state.shape[0], 1):
        raise ValueError(f"{name} F_row shape mismatch: expected {(F_state.shape[0], 1)}, got {F_row.shape}.")
    if F_col.shape != (1, F_state.shape[1]):
        raise ValueError(f"{name} F_col shape mismatch: expected {(1, F_state.shape[1])}, got {F_col.shape}.")
    if F_module.shape != (1, 1):
        raise ValueError(f"{name} F_module shape mismatch: expected {(1, 1)}, got {F_module.shape}.")
    if scalar(indices.max().to(dtype=torch.float32)) >= float(F_state.numel()) or scalar(indices.min().to(dtype=torch.float32)) < 0.0:
        raise ValueError(f"{name} outcome indices out of range for F_state size {F_state.numel()}.")
    advantage_tensor = torch.as_tensor(advantage, device=features.device, dtype=features.dtype)
    total_eligibility_tensor = torch.as_tensor(total_eligibility_mass, device=features.device, dtype=features.dtype)
    total_selected_tensor = torch.as_tensor(total_selected_count, device=features.device, dtype=features.dtype)
    global_eligibility_share = eligibility / total_eligibility_tensor.clamp_min(cfg.eps)
    route_advantage = advantage_tensor * global_eligibility_share * total_selected_tensor
    credit_warmup = min(1.0, float(step) / float(max(1, cfg.route_credit_warmup_steps)))
    credit_logits = (route_advantage / cfg.route_credit_scale).clamp(
        -cfg.route_credit_logit_clip,
        cfg.route_credit_logit_clip,
    )
    credit_logits = credit_logits * torch.as_tensor(credit_warmup, device=features.device, dtype=features.dtype)
    target = (0.5 + 0.5 * torch.tanh(credit_logits)).clamp(0.0, 1.0)
    if utility > 0.0:
        floor_tensor = torch.as_tensor(cfg.positive_utility_write_floor, device=features.device, dtype=features.dtype)
        target = torch.maximum(target, floor_tensor).clamp(0.0, 1.0)
    positive_utility_tensor = torch.as_tensor(max(0.0, utility), device=features.device, dtype=features.dtype)
    route_formation_raw = positive_utility_tensor * global_eligibility_share * total_selected_tensor
    formation_logits = (route_formation_raw / cfg.route_formation_scale).clamp(
        0.0,
        cfg.route_formation_logit_clip,
    )
    formation_logits = formation_logits * torch.as_tensor(credit_warmup, device=features.device, dtype=features.dtype)
    positive_route_utility = torch.tanh(formation_logits).clamp(0.0, 1.0)
    if cfg.outcome_formation_lr > 0.0 and scalar(positive_route_utility.max()) > 0.0:
        flat_F = F_state.reshape(-1)
        selected_F = flat_F[indices]
        flat_F[indices] = (
            selected_F + cfg.outcome_formation_lr * positive_route_utility * (1.0 - selected_F)
        ).clamp(0.0, 1.0)
        credit_matrix = torch.zeros_like(F_state)
        credit_matrix.reshape(-1).scatter_(0, indices, positive_route_utility)
        selected_activity = torch.zeros_like(F_state)
        selected_activity.reshape(-1).scatter_(0, indices, 1.0)
        row_credit, col_credit, module_credit = multiscale_formation_components(
            credit_matrix,
            selected_activity,
            pooling=cfg.formation_multiscale_pooling,
            eps=cfg.eps,
            name=f"{name}_outcome_formation",
        )
        F_row.add_(cfg.outcome_formation_lr * row_credit * (1.0 - F_row)).clamp_(0.0, 1.0)
        F_col.add_(cfg.outcome_formation_lr * col_credit * (1.0 - F_col)).clamp_(0.0, 1.0)
        F_module.add_(cfg.outcome_formation_lr * module_credit * (1.0 - F_module)).clamp_(0.0, 1.0)
    route_weight = (eligibility / torch.as_tensor(eligibility_mass, device=features.device, dtype=features.dtype).clamp_min(cfg.eps)).clamp_min(0.0)
    route_weight = route_weight / route_weight.mean().clamp_min(cfg.eps)
    if cfg.state_space_reasoner_dim > 0:
        required_ssr = {
            "ssr_rows",
            "ssr_cols",
            "ssr_row_z",
            "ssr_col_z",
            "ssr_module_z",
            "ssr_write_gate",
            "ssr_protect_gate",
            "ssr_protect_target",
            "ssr_reliability",
            "ssr_reliability_target",
            "ssr_collision",
            "ssr_collision_target",
            "ssr_value_pred",
        }
        missing_ssr = required_ssr.difference(context)
        if missing_ssr:
            raise RuntimeError(f"{name} SSR outcome context missing keys: {sorted(missing_ssr)}.")
        ssr_rows = context["ssr_rows"]
        ssr_cols = context["ssr_cols"]
        ssr_row_z = context["ssr_row_z"]
        ssr_col_z = context["ssr_col_z"]
        ssr_module_z = context["ssr_module_z"]
        ssr_write_gate = context["ssr_write_gate"]
        ssr_protect_gate = context["ssr_protect_gate"]
        ssr_protect_target = context["ssr_protect_target"]
        ssr_reliability = context["ssr_reliability"]
        ssr_reliability_target = context["ssr_reliability_target"]
        ssr_collision = context["ssr_collision"]
        ssr_collision_target = context["ssr_collision_target"]
        ssr_value_pred = context["ssr_value_pred"]
        z_dim = cfg.state_space_reasoner_dim + SSR_OBSERVATION_COUNT
        if ssr_rows.shape != indices.shape or ssr_cols.shape != indices.shape:
            raise ValueError(f"{name} SSR row/col index shape mismatch: {ssr_rows.shape}, {ssr_cols.shape}, {indices.shape}.")
        if ssr_row_z.shape != (selected_count, z_dim):
            raise ValueError(f"{name} SSR row z shape mismatch: expected {(selected_count, z_dim)}, got {ssr_row_z.shape}.")
        if ssr_col_z.shape != (selected_count, z_dim):
            raise ValueError(f"{name} SSR col z shape mismatch: expected {(selected_count, z_dim)}, got {ssr_col_z.shape}.")
        if ssr_module_z.shape != (selected_count, z_dim):
            raise ValueError(f"{name} SSR module z shape mismatch: expected {(selected_count, z_dim)}, got {ssr_module_z.shape}.")
        if (
            ssr_write_gate.shape != (selected_count,)
            or ssr_protect_gate.shape != (selected_count,)
            or ssr_protect_target.shape != (selected_count,)
            or ssr_reliability.shape != (selected_count,)
            or ssr_reliability_target.shape != (selected_count,)
            or ssr_collision.shape != (selected_count,)
            or ssr_collision_target.shape != (selected_count,)
            or ssr_value_pred.shape != (selected_count,)
        ):
            raise ValueError(
                f"{name} SSR gate/value shape mismatch: "
                f"write={ssr_write_gate.shape}, protect={ssr_protect_gate.shape}, "
                f"target={ssr_protect_target.shape}, reliability={ssr_reliability.shape}, "
                f"reliability_target={ssr_reliability_target.shape}, collision={ssr_collision.shape}, "
                f"collision_target={ssr_collision_target.shape}, value={ssr_value_pred.shape}, n={selected_count}."
            )
        require_finite_tensor(f"{name}_ssr_write_gate", ssr_write_gate)
        require_finite_tensor(f"{name}_ssr_protect_gate", ssr_protect_gate)
        require_finite_tensor(f"{name}_ssr_protect_target", ssr_protect_target)
        require_finite_tensor(f"{name}_ssr_reliability", ssr_reliability)
        require_finite_tensor(f"{name}_ssr_reliability_target", ssr_reliability_target)
        require_finite_tensor(f"{name}_ssr_collision", ssr_collision)
        require_finite_tensor(f"{name}_ssr_collision_target", ssr_collision_target)
        require_finite_tensor(f"{name}_ssr_value_pred", ssr_value_pred)
        if cfg.state_space_reasoner_lr > 0.0 or cfg.state_space_value_lr > 0.0:
            ssr_z_mean = (ssr_row_z + ssr_col_z + ssr_module_z) / 3.0
            ssr_gate_error = target - ssr_write_gate
            ssr_gate_grad = ssr_gate_error * ssr_write_gate * (1.0 - ssr_write_gate) * route_weight
            ssr_write_delta = torch.einsum("n,nz->z", ssr_gate_grad, ssr_z_mean) / float(selected_count)
            ssr_write_bias_delta = ssr_gate_grad.mean().reshape_as(ssr_write_bias)
            ssr_protect_gate_error = ssr_protect_target - ssr_protect_gate
            ssr_protect_gate_grad = (
                ssr_protect_gate_error * ssr_protect_gate * (1.0 - ssr_protect_gate) * route_weight
            )
            ssr_protect_delta = torch.einsum("n,nz->z", ssr_protect_gate_grad, ssr_z_mean) / float(selected_count)
            ssr_protect_bias_delta = ssr_protect_gate_grad.mean().reshape_as(ssr_protect_bias)
            ssr_reliability_error = ssr_reliability_target - ssr_reliability
            ssr_reliability_grad = ssr_reliability_error * ssr_reliability * (1.0 - ssr_reliability) * route_weight
            ssr_reliability_delta = torch.einsum("n,nz->z", ssr_reliability_grad, ssr_z_mean) / float(selected_count)
            ssr_reliability_bias_delta = ssr_reliability_grad.mean().reshape_as(ssr_reliability_bias)
            ssr_collision_error = ssr_collision_target - ssr_collision
            ssr_collision_grad = ssr_collision_error * ssr_collision * (1.0 - ssr_collision) * route_weight
            ssr_collision_delta = torch.einsum("n,nz->z", ssr_collision_grad, ssr_z_mean) / float(selected_count)
            ssr_collision_bias_delta = ssr_collision_grad.mean().reshape_as(ssr_collision_bias)
            ssr_td_error = route_advantage - ssr_value_pred
            ssr_value_grad = ssr_td_error * route_weight
            ssr_value_delta = torch.einsum("n,nz->z", ssr_value_grad, ssr_z_mean) / float(selected_count)
            ssr_value_bias_delta = ssr_value_grad.mean().reshape_as(ssr_value_bias)
            ssr_update_norm_tensor = (
                torch.linalg.vector_norm(ssr_write_delta)
                + torch.linalg.vector_norm(ssr_write_bias_delta)
                + torch.linalg.vector_norm(ssr_protect_delta)
                + torch.linalg.vector_norm(ssr_protect_bias_delta)
                + torch.linalg.vector_norm(ssr_reliability_delta)
                + torch.linalg.vector_norm(ssr_reliability_bias_delta)
                + torch.linalg.vector_norm(ssr_collision_delta)
                + torch.linalg.vector_norm(ssr_collision_bias_delta)
                + torch.linalg.vector_norm(ssr_value_delta)
                + torch.linalg.vector_norm(ssr_value_bias_delta)
            )
            ssr_update_scale = torch.clamp(
                torch.as_tensor(cfg.state_space_update_clip, device=features.device, dtype=features.dtype)
                / ssr_update_norm_tensor.clamp_min(cfg.eps),
                max=1.0,
            )
            ssr_write_decay = 1.0 - cfg.state_space_reasoner_lr * cfg.state_space_weight_decay
            ssr_value_decay = 1.0 - cfg.state_space_value_lr * cfg.state_space_weight_decay
            ssr_write_head.mul_(ssr_write_decay).add_(
                ssr_write_delta,
                alpha=cfg.state_space_reasoner_lr * scalar(ssr_update_scale),
            )
            ssr_write_bias.mul_(ssr_write_decay).add_(
                ssr_write_bias_delta,
                alpha=cfg.state_space_reasoner_lr * scalar(ssr_update_scale),
            )
            ssr_protect_head.mul_(ssr_write_decay).add_(
                ssr_protect_delta,
                alpha=cfg.state_space_reasoner_lr * scalar(ssr_update_scale),
            )
            ssr_protect_bias.mul_(ssr_write_decay).add_(
                ssr_protect_bias_delta,
                alpha=cfg.state_space_reasoner_lr * scalar(ssr_update_scale),
            )
            ssr_reliability_head.mul_(ssr_write_decay).add_(
                ssr_reliability_delta,
                alpha=cfg.state_space_reasoner_lr * scalar(ssr_update_scale),
            )
            ssr_reliability_bias.mul_(ssr_write_decay).add_(
                ssr_reliability_bias_delta,
                alpha=cfg.state_space_reasoner_lr * scalar(ssr_update_scale),
            )
            ssr_collision_head.mul_(ssr_write_decay).add_(
                ssr_collision_delta,
                alpha=cfg.state_space_reasoner_lr * scalar(ssr_update_scale),
            )
            ssr_collision_bias.mul_(ssr_write_decay).add_(
                ssr_collision_bias_delta,
                alpha=cfg.state_space_reasoner_lr * scalar(ssr_update_scale),
            )
            ssr_value_head.mul_(ssr_value_decay).add_(
                ssr_value_delta,
                alpha=cfg.state_space_value_lr * scalar(ssr_update_scale),
            )
            ssr_value_bias.mul_(ssr_value_decay).add_(
                ssr_value_bias_delta,
                alpha=cfg.state_space_value_lr * scalar(ssr_update_scale),
            )
            SSR_last_td_error_abs.copy_(ssr_td_error.abs().mean().reshape_as(SSR_last_td_error_abs))
            SSR_last_update_norm.copy_((ssr_update_norm_tensor * ssr_update_scale).reshape_as(SSR_last_update_norm))
        if cfg.state_space_credit_beta < 1.0:
            row_credit_target = torch.zeros_like(SSR_row_credit)
            row_credit_count = torch.zeros_like(SSR_row_credit)
            col_credit_target = torch.zeros_like(SSR_col_credit)
            col_credit_count = torch.zeros_like(SSR_col_credit)
            route_credit_values = torch.maximum(
                torch.maximum(positive_route_utility, ssr_protect_target),
                torch.maximum(ssr_reliability_target, ssr_collision_target),
            ).reshape(-1, 1)
            row_credit_target.index_add_(0, ssr_rows, route_credit_values)
            row_credit_count.index_add_(0, ssr_rows, torch.ones_like(route_credit_values))
            col_credit_target.index_add_(0, ssr_cols, route_credit_values)
            col_credit_count.index_add_(0, ssr_cols, torch.ones_like(route_credit_values))
            row_credit_target = row_credit_target / row_credit_count.clamp_min(1.0)
            col_credit_target = col_credit_target / col_credit_count.clamp_min(1.0)
            module_credit_target = route_credit_values.mean().reshape_as(SSR_module_credit)
            SSR_row_credit.mul_(cfg.state_space_credit_beta).add_(row_credit_target, alpha=1.0 - cfg.state_space_credit_beta)
            SSR_col_credit.mul_(cfg.state_space_credit_beta).add_(col_credit_target, alpha=1.0 - cfg.state_space_credit_beta)
            SSR_module_credit.mul_(cfg.state_space_credit_beta).add_(module_credit_target, alpha=1.0 - cfg.state_space_credit_beta)
            SSR_row_credit.clamp_(-1.0, 1.0)
            SSR_col_credit.clamp_(-1.0, 1.0)
            SSR_module_credit.clamp_(-1.0, 1.0)
    if cfg.state_space_reasoner_dim > 0:
        protect_target_stats = ssr_protect_target
        protect_gate_stats = ssr_protect_gate
        protect_error_stats = (ssr_protect_target - ssr_protect_gate).abs()
        reliability_target_stats = ssr_reliability_target
        reliability_pred_stats = ssr_reliability
        reliability_error_stats = (ssr_reliability_target - ssr_reliability).abs()
        collision_target_stats = ssr_collision_target
        collision_pred_stats = ssr_collision
        collision_error_stats = (ssr_collision_target - ssr_collision).abs()
    else:
        protect_target_stats = torch.zeros_like(write_gate)
        protect_gate_stats = torch.zeros_like(write_gate)
        protect_error_stats = torch.zeros_like(write_gate)
        reliability_target_stats = torch.zeros_like(write_gate)
        reliability_pred_stats = torch.zeros_like(write_gate)
        reliability_error_stats = torch.zeros_like(write_gate)
        collision_target_stats = torch.zeros_like(write_gate)
        collision_pred_stats = torch.zeros_like(write_gate)
        collision_error_stats = torch.zeros_like(write_gate)
    if cfg.outcome_credit_lr == 0.0:
        return GCOOutcomeCreditStats(
            name=name,
            selected_count=float(selected_count),
            eligibility_mass=eligibility_mass,
            eligibility_max_share=scalar(global_eligibility_share.max()),
            advantage=advantage,
            route_advantage_mean=scalar(route_advantage.mean()),
            route_advantage_abs_mean=scalar(route_advantage.abs().mean()),
            route_advantage_max=scalar(route_advantage.max()),
            route_formation_utility_mean=scalar(positive_route_utility.mean()),
            route_formation_utility_max=scalar(positive_route_utility.max()),
            utility_target=scalar(target.mean()),
            utility_target_min=scalar(target.min()),
            utility_target_max=scalar(target.max()),
            write_gate_mean=scalar(write_gate.mean()),
            protect_target_mean=scalar(protect_target_stats.mean()),
            protect_target_min=scalar(protect_target_stats.min()),
            protect_target_max=scalar(protect_target_stats.max()),
            protect_gate_mean=scalar(protect_gate_stats.mean()),
            protect_error_abs_mean=scalar(protect_error_stats.mean()),
            reliability_target_mean=scalar(reliability_target_stats.mean()),
            reliability_target_min=scalar(reliability_target_stats.min()),
            reliability_target_max=scalar(reliability_target_stats.max()),
            reliability_pred_mean=scalar(reliability_pred_stats.mean()),
            reliability_error_abs_mean=scalar(reliability_error_stats.mean()),
            collision_target_mean=scalar(collision_target_stats.mean()),
            collision_target_min=scalar(collision_target_stats.min()),
            collision_target_max=scalar(collision_target_stats.max()),
            collision_pred_mean=scalar(collision_pred_stats.mean()),
            collision_error_abs_mean=scalar(collision_error_stats.mean()),
            gate_error_abs_mean=0.0,
            update_norm=0.0,
        )

    gate_error = target - write_gate
    gate_local_grad = gate_error * write_gate * (1.0 - write_gate) * route_weight
    gate_delta = torch.einsum("n,nf->f", gate_local_grad, features) / float(selected_count)
    gate_state_delta = (gate_local_grad * state).mean()
    gate_bias_delta = gate_local_grad.mean()
    state_grad = gate_local_grad * reasoner_gate_state_weights[WRITE_GATE_INDEX] * (1.0 - state * state)
    state_delta = torch.einsum("n,nf->f", state_grad, features) / float(selected_count)
    state_bias_delta = state_grad.mean().reshape_as(reasoner_state_bias)
    update_norm_tensor = (
        torch.linalg.vector_norm(gate_delta)
        + torch.linalg.vector_norm(gate_state_delta.reshape(1))
        + torch.linalg.vector_norm(gate_bias_delta.reshape(1))
        + torch.linalg.vector_norm(state_delta)
        + torch.linalg.vector_norm(state_bias_delta)
    )
    update_scale = torch.clamp(
        torch.as_tensor(cfg.reasoner_update_clip, device=features.device, dtype=features.dtype)
        / update_norm_tensor.clamp_min(cfg.eps),
        max=1.0,
    )
    decay = 1.0 - cfg.outcome_credit_lr * cfg.reasoner_weight_decay
    reasoner_gate_weights.mul_(decay)
    reasoner_gate_state_weights.mul_(decay)
    reasoner_gate_bias.mul_(decay)
    reasoner_state_weights.mul_(decay)
    reasoner_state_bias.mul_(decay)
    scaled_lr = cfg.outcome_credit_lr * scalar(update_scale)
    reasoner_gate_weights[WRITE_GATE_INDEX].add_(gate_delta, alpha=scaled_lr)
    reasoner_gate_state_weights[WRITE_GATE_INDEX].add_(gate_state_delta, alpha=scaled_lr)
    reasoner_gate_bias[WRITE_GATE_INDEX].add_(gate_bias_delta, alpha=scaled_lr)
    reasoner_state_weights.add_(state_delta, alpha=scaled_lr)
    reasoner_state_bias.add_(state_bias_delta, alpha=scaled_lr)
    stats = GCOOutcomeCreditStats(
        name=name,
        selected_count=float(selected_count),
        eligibility_mass=eligibility_mass,
        eligibility_max_share=scalar(global_eligibility_share.max()),
        advantage=advantage,
        route_advantage_mean=scalar(route_advantage.mean()),
        route_advantage_abs_mean=scalar(route_advantage.abs().mean()),
        route_advantage_max=scalar(route_advantage.max()),
        route_formation_utility_mean=scalar(positive_route_utility.mean()),
        route_formation_utility_max=scalar(positive_route_utility.max()),
        utility_target=scalar(target.mean()),
        utility_target_min=scalar(target.min()),
        utility_target_max=scalar(target.max()),
        write_gate_mean=scalar(write_gate.mean()),
        protect_target_mean=scalar(protect_target_stats.mean()),
        protect_target_min=scalar(protect_target_stats.min()),
        protect_target_max=scalar(protect_target_stats.max()),
        protect_gate_mean=scalar(protect_gate_stats.mean()),
        protect_error_abs_mean=scalar(protect_error_stats.mean()),
        reliability_target_mean=scalar(reliability_target_stats.mean()),
        reliability_target_min=scalar(reliability_target_stats.min()),
        reliability_target_max=scalar(reliability_target_stats.max()),
        reliability_pred_mean=scalar(reliability_pred_stats.mean()),
        reliability_error_abs_mean=scalar(reliability_error_stats.mean()),
        collision_target_mean=scalar(collision_target_stats.mean()),
        collision_target_min=scalar(collision_target_stats.min()),
        collision_target_max=scalar(collision_target_stats.max()),
        collision_pred_mean=scalar(collision_pred_stats.mean()),
        collision_error_abs_mean=scalar(collision_error_stats.mean()),
        gate_error_abs_mean=scalar(gate_error.abs().mean()),
        update_norm=scalar(update_norm_tensor * update_scale),
    )
    for key, value in asdict(stats).items():
        if isinstance(value, float):
            require_finite_float(f"{name}_outcome_{key}", value)
    return stats


class GCOLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cfg: NativeGCOConfig, *, name: str) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"GCOLinear dimensions must be positive, got in={in_dim}, out={out_dim}.")
        self.cfg = cfg
        self.name = name
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.register_buffer("A", torch.full((out_dim, in_dim), cfg.init_topology))
        self.register_buffer("H", torch.zeros(out_dim, in_dim))
        self.register_buffer("F_state", torch.zeros(out_dim, in_dim))
        self.register_buffer("F_row", torch.zeros(out_dim, 1))
        self.register_buffer("F_col", torch.zeros(1, in_dim))
        self.register_buffer("F_module", torch.zeros(1, 1))
        self.register_buffer("H_latch", torch.zeros(out_dim, in_dim))
        self.register_buffer("SSR_row_state", torch.zeros(out_dim, cfg.state_space_reasoner_dim))
        self.register_buffer("SSR_col_state", torch.zeros(in_dim, cfg.state_space_reasoner_dim))
        self.register_buffer("SSR_module_state", torch.zeros(1, cfg.state_space_reasoner_dim))
        self.register_buffer("SSR_row_credit", torch.zeros(out_dim, 1))
        self.register_buffer("SSR_col_credit", torch.zeros(in_dim, 1))
        self.register_buffer("SSR_module_credit", torch.zeros(1, 1))
        self.register_buffer("SSR_last_td_error_abs", torch.zeros(1))
        self.register_buffer("SSR_last_update_norm", torch.zeros(1))
        self.register_buffer("P_state", torch.zeros(out_dim, in_dim))
        self.register_buffer("D_state", torch.zeros(out_dim, in_dim))
        self.register_buffer("U", torch.zeros(out_dim, in_dim))
        self.register_buffer("C", torch.zeros(out_dim, in_dim))
        self.register_buffer("S", torch.zeros(out_dim, in_dim))
        self.register_buffer("Age", torch.zeros(out_dim, in_dim))
        self.reasoner_state_weights = native_reasoner_parameter(REASONER_FEATURE_COUNT)
        self.reasoner_state_bias = native_reasoner_parameter(1)
        self.reasoner_gate_weights = native_reasoner_parameter(REASONER_GATE_COUNT, REASONER_FEATURE_COUNT)
        self.reasoner_gate_state_weights = native_reasoner_parameter(REASONER_GATE_COUNT)
        self.reasoner_gate_bias = native_reasoner_parameter(REASONER_GATE_COUNT)
        self.reasoner_role_weights = native_reasoner_parameter(REASONER_ROLE_COUNT, REASONER_FEATURE_COUNT)
        self.reasoner_role_state_weights = native_reasoner_parameter(REASONER_ROLE_COUNT)
        self.reasoner_role_bias = native_reasoner_parameter(REASONER_ROLE_COUNT)
        z_dim = cfg.state_space_reasoner_dim + SSR_OBSERVATION_COUNT
        self.ssr_state_state_weights = native_reasoner_parameter(
            cfg.state_space_reasoner_dim,
            cfg.state_space_reasoner_dim,
            init_std=cfg.state_space_init_std,
        )
        self.ssr_state_obs_weights = native_reasoner_parameter(
            cfg.state_space_reasoner_dim,
            SSR_OBSERVATION_COUNT,
            init_std=cfg.state_space_init_std,
        )
        self.ssr_state_credit_weights = native_reasoner_parameter(cfg.state_space_reasoner_dim, init_std=cfg.state_space_init_std)
        self.ssr_state_bias = native_reasoner_parameter(cfg.state_space_reasoner_dim)
        self.ssr_write_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_write_bias = native_reasoner_parameter(1)
        self.ssr_protect_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_protect_bias = native_reasoner_parameter(1)
        self.ssr_reliability_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_reliability_bias = native_reasoner_parameter(1)
        self.ssr_collision_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_collision_bias = native_reasoner_parameter(1)
        self.ssr_value_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_value_bias = native_reasoner_parameter(1)
        self.ssr_rewire_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_rewire_bias = native_reasoner_parameter(1)
        self.ssr_forget_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_forget_bias = native_reasoner_parameter(1)
        self.ssr_compress_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_compress_bias = native_reasoner_parameter(1)
        self.ssr_gain_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_gain_bias = native_reasoner_parameter(1)
        self.ssr_capacity_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_capacity_bias = native_reasoner_parameter(1)
        self.ssr_forget_safe_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_forget_safe_bias = native_reasoner_parameter(1)
        self.ssr_priority_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_priority_bias = native_reasoner_parameter(1)
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._y_for_grad: torch.Tensor | None = None
        self._outcome_credit_context: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.W.shape[1]:
            raise ValueError(f"{self.name} expected input dim {self.W.shape[1]}, got {x.shape[-1]}.")
        y = F.linear(x, self.W * self.A)
        self._x = x.detach()
        self._y = y.detach()
        self._y_for_grad = y
        if y.requires_grad:
            y.retain_grad()
        return y

    def pathway(self) -> torch.Tensor:
        if self._x is None or self._y is None:
            raise RuntimeError(f"{self.name}.gco_step called before forward.")
        x = self._x.reshape(-1, self._x.shape[-1]).abs()
        y = self._y.reshape(-1, self._y.shape[-1]).abs()
        if x.shape[0] <= 0:
            raise ValueError(f"{self.name} cannot build pathway from empty activation batch.")
        raw = y.T @ x / float(x.shape[0])
        return normalize_pathway(
            raw,
            percentile=self.cfg.pathway_percentile,
            eps=self.cfg.eps,
            name=f"{self.name}_pathway",
        )

    def direct_write_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._x is None or self._y_for_grad is None:
            raise RuntimeError(f"{self.name}.direct_write_tensors called before forward.")
        if self._y_for_grad.grad is None:
            raise RuntimeError(f"{self.name}.direct_write_tensors called before backward produced output error.")
        x = self._x.reshape(-1, self._x.shape[-1])
        error = self._y_for_grad.grad.detach().reshape(-1, self._y_for_grad.shape[-1])
        if x.shape[0] != error.shape[0]:
            raise ValueError(f"{self.name} direct write batch mismatch: X={x.shape}, E={error.shape}.")
        require_finite_tensor(f"{self.name}_direct_input", x)
        require_finite_tensor(f"{self.name}_direct_output_error", error)
        return x, error

    @torch.no_grad()
    def gco_step(self, step: int, *, developmental_maturity: float, failed_write_signal: float) -> GCOModuleStats:
        direct_input: torch.Tensor | None = None
        direct_output_error: torch.Tensor | None = None
        if self.cfg.write_mode == "direct":
            direct_input, direct_output_error = self.direct_write_tensors()
        stats, outcome_credit_context = gco_matrix_step(
            self.name,
            self.W,
            self.A,
            self.H,
            self.F_state,
            self.F_row,
            self.F_col,
            self.F_module,
            self.H_latch,
            self.SSR_row_state,
            self.SSR_col_state,
            self.SSR_module_state,
            self.SSR_row_credit,
            self.SSR_col_credit,
            self.SSR_module_credit,
            self.SSR_last_td_error_abs,
            self.SSR_last_update_norm,
            self.P_state,
            self.D_state,
            self.U,
            self.C,
            self.S,
            self.Age,
            self.reasoner_state_weights,
            self.reasoner_state_bias,
            self.reasoner_gate_weights,
            self.reasoner_gate_state_weights,
            self.reasoner_gate_bias,
            self.reasoner_role_weights,
            self.reasoner_role_state_weights,
            self.reasoner_role_bias,
            self.ssr_state_state_weights,
            self.ssr_state_obs_weights,
            self.ssr_state_credit_weights,
            self.ssr_state_bias,
            self.ssr_write_head,
            self.ssr_write_bias,
            self.ssr_protect_head,
            self.ssr_protect_bias,
            self.ssr_reliability_head,
            self.ssr_reliability_bias,
            self.ssr_collision_head,
            self.ssr_collision_bias,
            self.ssr_value_head,
            self.ssr_value_bias,
            self.ssr_rewire_head,
            self.ssr_rewire_bias,
            self.ssr_forget_head,
            self.ssr_forget_bias,
            self.ssr_compress_head,
            self.ssr_compress_bias,
            self.ssr_gain_head,
            self.ssr_gain_bias,
            self.ssr_capacity_head,
            self.ssr_capacity_bias,
            self.ssr_forget_safe_head,
            self.ssr_forget_safe_bias,
            self.ssr_priority_head,
            self.ssr_priority_bias,
            self.pathway(),
            self.cfg,
            step,
            developmental_maturity=developmental_maturity,
            failed_write_signal=failed_write_signal,
            direct_input=direct_input,
            direct_output_error=direct_output_error,
            direct_write_direction=None,
        )
        self._outcome_credit_context = outcome_credit_context
        return stats

    @torch.no_grad()
    def outcome_credit_eligibility_sum(self) -> float:
        if self._outcome_credit_context is None:
            return 0.0
        eligibility = self._outcome_credit_context.get("eligibility")
        if eligibility is None:
            raise RuntimeError(f"{self.name} outcome context is missing eligibility.")
        require_finite_tensor(f"{self.name}_outcome_eligibility_sum", eligibility)
        return scalar(eligibility.sum())

    @torch.no_grad()
    def outcome_credit_selected_count(self) -> float:
        if self._outcome_credit_context is None:
            return 0.0
        eligibility = self._outcome_credit_context.get("eligibility")
        if eligibility is None:
            raise RuntimeError(f"{self.name} outcome context is missing eligibility.")
        return float(eligibility.shape[0])

    @torch.no_grad()
    def selected_route_traces(self, *, limit: int) -> list[dict[str, object]]:
        return selected_route_trace_rows(self.name, self._outcome_credit_context, limit=limit)

    @torch.no_grad()
    def apply_outcome_credit(
        self,
        *,
        utility: float,
        advantage: float,
        total_eligibility_mass: float,
        total_selected_count: float,
        step: int,
    ) -> GCOOutcomeCreditStats:
        stats = apply_outcome_credit_to_reasoner(
            self.name,
            self.F_state,
            self.F_row,
            self.F_col,
            self.F_module,
            self.SSR_row_state,
            self.SSR_col_state,
            self.SSR_module_state,
            self.SSR_row_credit,
            self.SSR_col_credit,
            self.SSR_module_credit,
            self.SSR_last_td_error_abs,
            self.SSR_last_update_norm,
            self.reasoner_state_weights,
            self.reasoner_state_bias,
            self.reasoner_gate_weights,
            self.reasoner_gate_state_weights,
            self.reasoner_gate_bias,
            self.ssr_write_head,
            self.ssr_write_bias,
            self.ssr_protect_head,
            self.ssr_protect_bias,
            self.ssr_reliability_head,
            self.ssr_reliability_bias,
            self.ssr_collision_head,
            self.ssr_collision_bias,
            self.ssr_value_head,
            self.ssr_value_bias,
            self._outcome_credit_context,
            self.cfg,
            utility,
            advantage,
            total_eligibility_mass,
            total_selected_count,
            step,
        )
        self._outcome_credit_context = None
        return stats


class GCOEmbedding(nn.Module):
    def __init__(self, count: int, dim: int, cfg: NativeGCOConfig, *, name: str) -> None:
        super().__init__()
        if count <= 0 or dim <= 0:
            raise ValueError(f"GCOEmbedding dimensions must be positive, got count={count}, dim={dim}.")
        self.cfg = cfg
        self.name = name
        self.W = nn.Parameter(torch.empty(count, dim))
        nn.init.normal_(self.W, mean=0.0, std=0.02)
        self.register_buffer("A", torch.full((count, dim), cfg.init_topology))
        self.register_buffer("H", torch.zeros(count, dim))
        self.register_buffer("F_state", torch.zeros(count, dim))
        self.register_buffer("F_row", torch.zeros(count, 1))
        self.register_buffer("F_col", torch.zeros(1, dim))
        self.register_buffer("F_module", torch.zeros(1, 1))
        self.register_buffer("H_latch", torch.zeros(count, dim))
        self.register_buffer("SSR_row_state", torch.zeros(count, cfg.state_space_reasoner_dim))
        self.register_buffer("SSR_col_state", torch.zeros(dim, cfg.state_space_reasoner_dim))
        self.register_buffer("SSR_module_state", torch.zeros(1, cfg.state_space_reasoner_dim))
        self.register_buffer("SSR_row_credit", torch.zeros(count, 1))
        self.register_buffer("SSR_col_credit", torch.zeros(dim, 1))
        self.register_buffer("SSR_module_credit", torch.zeros(1, 1))
        self.register_buffer("SSR_last_td_error_abs", torch.zeros(1))
        self.register_buffer("SSR_last_update_norm", torch.zeros(1))
        self.register_buffer("P_state", torch.zeros(count, dim))
        self.register_buffer("D_state", torch.zeros(count, dim))
        self.register_buffer("U", torch.zeros(count, dim))
        self.register_buffer("C", torch.zeros(count, dim))
        self.register_buffer("S", torch.zeros(count, dim))
        self.register_buffer("Age", torch.zeros(count, dim))
        self.reasoner_state_weights = native_reasoner_parameter(REASONER_FEATURE_COUNT)
        self.reasoner_state_bias = native_reasoner_parameter(1)
        self.reasoner_gate_weights = native_reasoner_parameter(REASONER_GATE_COUNT, REASONER_FEATURE_COUNT)
        self.reasoner_gate_state_weights = native_reasoner_parameter(REASONER_GATE_COUNT)
        self.reasoner_gate_bias = native_reasoner_parameter(REASONER_GATE_COUNT)
        self.reasoner_role_weights = native_reasoner_parameter(REASONER_ROLE_COUNT, REASONER_FEATURE_COUNT)
        self.reasoner_role_state_weights = native_reasoner_parameter(REASONER_ROLE_COUNT)
        self.reasoner_role_bias = native_reasoner_parameter(REASONER_ROLE_COUNT)
        z_dim = cfg.state_space_reasoner_dim + SSR_OBSERVATION_COUNT
        self.ssr_state_state_weights = native_reasoner_parameter(
            cfg.state_space_reasoner_dim,
            cfg.state_space_reasoner_dim,
            init_std=cfg.state_space_init_std,
        )
        self.ssr_state_obs_weights = native_reasoner_parameter(
            cfg.state_space_reasoner_dim,
            SSR_OBSERVATION_COUNT,
            init_std=cfg.state_space_init_std,
        )
        self.ssr_state_credit_weights = native_reasoner_parameter(cfg.state_space_reasoner_dim, init_std=cfg.state_space_init_std)
        self.ssr_state_bias = native_reasoner_parameter(cfg.state_space_reasoner_dim)
        self.ssr_write_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_write_bias = native_reasoner_parameter(1)
        self.ssr_protect_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_protect_bias = native_reasoner_parameter(1)
        self.ssr_reliability_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_reliability_bias = native_reasoner_parameter(1)
        self.ssr_collision_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_collision_bias = native_reasoner_parameter(1)
        self.ssr_value_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_value_bias = native_reasoner_parameter(1)
        self.ssr_rewire_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_rewire_bias = native_reasoner_parameter(1)
        self.ssr_forget_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_forget_bias = native_reasoner_parameter(1)
        self.ssr_compress_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_compress_bias = native_reasoner_parameter(1)
        self.ssr_gain_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_gain_bias = native_reasoner_parameter(1)
        self.ssr_capacity_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_capacity_bias = native_reasoner_parameter(1)
        self.ssr_forget_safe_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_forget_safe_bias = native_reasoner_parameter(1)
        self.ssr_priority_head = native_reasoner_parameter(z_dim, init_std=cfg.state_space_init_std)
        self.ssr_priority_bias = native_reasoner_parameter(1)
        self._ids: torch.Tensor | None = None
        self._y_for_grad: torch.Tensor | None = None
        self._outcome_credit_context: dict[str, torch.Tensor] | None = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dtype != torch.long:
            raise ValueError(f"{self.name} ids must be torch.long, got {ids.dtype}.")
        if ids.numel() <= 0:
            raise ValueError(f"{self.name} received empty ids.")
        if int(ids.min().detach().cpu()) < 0 or int(ids.max().detach().cpu()) >= self.W.shape[0]:
            raise ValueError(f"{self.name} ids out of range for embedding count {self.W.shape[0]}.")
        self._ids = ids.detach()
        y = F.embedding(ids, self.W * self.A)
        self._y_for_grad = y
        if y.requires_grad:
            y.retain_grad()
        return y

    def pathway(self) -> torch.Tensor:
        if self._ids is None:
            raise RuntimeError(f"{self.name}.gco_step called before forward.")
        counts = torch.bincount(self._ids.reshape(-1), minlength=self.W.shape[0]).to(device=self.W.device, dtype=self.W.dtype)
        if scalar(counts.sum()) <= self.cfg.eps:
            raise RuntimeError(f"{self.name} has zero token count in pathway.")
        row_usage = (counts / counts.sum()).reshape(-1, 1)
        pathway = row_usage.expand_as(self.W)
        require_finite_tensor(f"{self.name}_pathway", pathway)
        return pathway.clamp(0.0, 1.0)

    def direct_write_direction(self) -> torch.Tensor:
        if self._ids is None or self._y_for_grad is None:
            raise RuntimeError(f"{self.name}.direct_write_direction called before forward.")
        if self._y_for_grad.grad is None:
            raise RuntimeError(f"{self.name}.direct_write_direction called before backward produced output error.")
        ids = self._ids.reshape(-1)
        error = self._y_for_grad.grad.detach().reshape(-1, self.W.shape[1])
        if ids.shape[0] != error.shape[0]:
            raise ValueError(f"{self.name} direct embedding write mismatch: ids={ids.shape}, error={error.shape}.")
        if self.cfg.direct_write_error_scale == "mean":
            scaled_error = error
        elif self.cfg.direct_write_error_scale == "sample_count":
            scaled_error = error * float(ids.shape[0])
        else:
            raise ValueError(f"{self.name} unknown direct write error scale: {self.cfg.direct_write_error_scale!r}.")
        direction = torch.zeros_like(self.W)
        direction.index_add_(0, ids, scaled_error)
        counts = torch.bincount(ids, minlength=self.W.shape[0]).to(device=self.W.device, dtype=self.W.dtype).reshape(-1, 1)
        direction = direction / (counts + self.cfg.direct_write_ridge).clamp_min(self.cfg.eps)
        require_finite_tensor(f"{self.name}_direct_write_direction", direction)
        return direction

    @torch.no_grad()
    def gco_step(self, step: int, *, developmental_maturity: float, failed_write_signal: float) -> GCOModuleStats:
        direct_write_direction: torch.Tensor | None = None
        if self.cfg.write_mode == "direct":
            direct_write_direction = self.direct_write_direction()
        stats, outcome_credit_context = gco_matrix_step(
            self.name,
            self.W,
            self.A,
            self.H,
            self.F_state,
            self.F_row,
            self.F_col,
            self.F_module,
            self.H_latch,
            self.SSR_row_state,
            self.SSR_col_state,
            self.SSR_module_state,
            self.SSR_row_credit,
            self.SSR_col_credit,
            self.SSR_module_credit,
            self.SSR_last_td_error_abs,
            self.SSR_last_update_norm,
            self.P_state,
            self.D_state,
            self.U,
            self.C,
            self.S,
            self.Age,
            self.reasoner_state_weights,
            self.reasoner_state_bias,
            self.reasoner_gate_weights,
            self.reasoner_gate_state_weights,
            self.reasoner_gate_bias,
            self.reasoner_role_weights,
            self.reasoner_role_state_weights,
            self.reasoner_role_bias,
            self.ssr_state_state_weights,
            self.ssr_state_obs_weights,
            self.ssr_state_credit_weights,
            self.ssr_state_bias,
            self.ssr_write_head,
            self.ssr_write_bias,
            self.ssr_protect_head,
            self.ssr_protect_bias,
            self.ssr_reliability_head,
            self.ssr_reliability_bias,
            self.ssr_collision_head,
            self.ssr_collision_bias,
            self.ssr_value_head,
            self.ssr_value_bias,
            self.ssr_rewire_head,
            self.ssr_rewire_bias,
            self.ssr_forget_head,
            self.ssr_forget_bias,
            self.ssr_compress_head,
            self.ssr_compress_bias,
            self.ssr_gain_head,
            self.ssr_gain_bias,
            self.ssr_capacity_head,
            self.ssr_capacity_bias,
            self.ssr_forget_safe_head,
            self.ssr_forget_safe_bias,
            self.ssr_priority_head,
            self.ssr_priority_bias,
            self.pathway(),
            self.cfg,
            step,
            developmental_maturity=developmental_maturity,
            failed_write_signal=failed_write_signal,
            direct_input=None,
            direct_output_error=None,
            direct_write_direction=direct_write_direction,
        )
        self._outcome_credit_context = outcome_credit_context
        return stats

    @torch.no_grad()
    def outcome_credit_eligibility_sum(self) -> float:
        if self._outcome_credit_context is None:
            return 0.0
        eligibility = self._outcome_credit_context.get("eligibility")
        if eligibility is None:
            raise RuntimeError(f"{self.name} outcome context is missing eligibility.")
        require_finite_tensor(f"{self.name}_outcome_eligibility_sum", eligibility)
        return scalar(eligibility.sum())

    @torch.no_grad()
    def outcome_credit_selected_count(self) -> float:
        if self._outcome_credit_context is None:
            return 0.0
        eligibility = self._outcome_credit_context.get("eligibility")
        if eligibility is None:
            raise RuntimeError(f"{self.name} outcome context is missing eligibility.")
        return float(eligibility.shape[0])

    @torch.no_grad()
    def selected_route_traces(self, *, limit: int) -> list[dict[str, object]]:
        return selected_route_trace_rows(self.name, self._outcome_credit_context, limit=limit)

    @torch.no_grad()
    def apply_outcome_credit(
        self,
        *,
        utility: float,
        advantage: float,
        total_eligibility_mass: float,
        total_selected_count: float,
        step: int,
    ) -> GCOOutcomeCreditStats:
        stats = apply_outcome_credit_to_reasoner(
            self.name,
            self.F_state,
            self.F_row,
            self.F_col,
            self.F_module,
            self.SSR_row_state,
            self.SSR_col_state,
            self.SSR_module_state,
            self.SSR_row_credit,
            self.SSR_col_credit,
            self.SSR_module_credit,
            self.SSR_last_td_error_abs,
            self.SSR_last_update_norm,
            self.reasoner_state_weights,
            self.reasoner_state_bias,
            self.reasoner_gate_weights,
            self.reasoner_gate_state_weights,
            self.reasoner_gate_bias,
            self.ssr_write_head,
            self.ssr_write_bias,
            self.ssr_protect_head,
            self.ssr_protect_bias,
            self.ssr_reliability_head,
            self.ssr_reliability_bias,
            self.ssr_collision_head,
            self.ssr_collision_bias,
            self.ssr_value_head,
            self.ssr_value_bias,
            self._outcome_credit_context,
            self.cfg,
            utility,
            advantage,
            total_eligibility_mass,
            total_selected_count,
            step,
        )
        self._outcome_credit_context = None
        return stats


@torch.no_grad()
def gco_matrix_step(
    name: str,
    W: torch.nn.Parameter,
    A: torch.Tensor,
    H: torch.Tensor,
    F_state: torch.Tensor,
    F_row: torch.Tensor,
    F_col: torch.Tensor,
    F_module: torch.Tensor,
    H_latch: torch.Tensor,
    SSR_row_state: torch.Tensor,
    SSR_col_state: torch.Tensor,
    SSR_module_state: torch.Tensor,
    SSR_row_credit: torch.Tensor,
    SSR_col_credit: torch.Tensor,
    SSR_module_credit: torch.Tensor,
    SSR_last_td_error_abs: torch.Tensor,
    SSR_last_update_norm: torch.Tensor,
    P_state: torch.Tensor,
    D_state: torch.Tensor,
    U: torch.Tensor,
    C: torch.Tensor,
    S: torch.Tensor,
    Age: torch.Tensor,
    reasoner_state_weights: torch.Tensor,
    reasoner_state_bias: torch.Tensor,
    reasoner_gate_weights: torch.Tensor,
    reasoner_gate_state_weights: torch.Tensor,
    reasoner_gate_bias: torch.Tensor,
    reasoner_role_weights: torch.Tensor,
    reasoner_role_state_weights: torch.Tensor,
    reasoner_role_bias: torch.Tensor,
    ssr_state_state_weights: torch.Tensor,
    ssr_state_obs_weights: torch.Tensor,
    ssr_state_credit_weights: torch.Tensor,
    ssr_state_bias: torch.Tensor,
    ssr_write_head: torch.Tensor,
    ssr_write_bias: torch.Tensor,
    ssr_protect_head: torch.Tensor,
    ssr_protect_bias: torch.Tensor,
    ssr_reliability_head: torch.Tensor,
    ssr_reliability_bias: torch.Tensor,
    ssr_collision_head: torch.Tensor,
    ssr_collision_bias: torch.Tensor,
    ssr_value_head: torch.Tensor,
    ssr_value_bias: torch.Tensor,
    ssr_rewire_head: torch.Tensor,
    ssr_rewire_bias: torch.Tensor,
    ssr_forget_head: torch.Tensor,
    ssr_forget_bias: torch.Tensor,
    ssr_compress_head: torch.Tensor,
    ssr_compress_bias: torch.Tensor,
    ssr_gain_head: torch.Tensor,
    ssr_gain_bias: torch.Tensor,
    ssr_capacity_head: torch.Tensor,
    ssr_capacity_bias: torch.Tensor,
    ssr_forget_safe_head: torch.Tensor,
    ssr_forget_safe_bias: torch.Tensor,
    ssr_priority_head: torch.Tensor,
    ssr_priority_bias: torch.Tensor,
    M: torch.Tensor,
    cfg: NativeGCOConfig,
    step: int,
    *,
    developmental_maturity: float,
    failed_write_signal: float,
    direct_input: torch.Tensor | None,
    direct_output_error: torch.Tensor | None,
    direct_write_direction: torch.Tensor | None,
) -> tuple[GCOModuleStats, dict[str, torch.Tensor] | None]:
    if W.grad is None:
        raise RuntimeError(f"{name}.gco_step called before backward produced a gradient.")
    bounded_float("developmental_maturity", developmental_maturity, 0.0, 1.0)
    bounded_float("failed_write_signal", failed_write_signal, 0.0, 1.0)
    maturity_tensor = torch.as_tensor(developmental_maturity, device=W.device, dtype=W.dtype)
    control_phase_gate = maturity_tensor
    failed_write_tensor = torch.as_tensor(failed_write_signal, device=W.device, dtype=W.dtype)
    if (
        W.shape != A.shape
        or W.shape != H.shape
        or W.shape != F_state.shape
        or W.shape != H_latch.shape
        or W.shape != P_state.shape
        or W.shape != D_state.shape
        or W.shape != U.shape
        or W.shape != C.shape
        or W.shape != S.shape
        or W.shape != Age.shape
        or W.shape != M.shape
    ):
        raise ValueError(
            f"{name} shape mismatch: W={W.shape}, A={A.shape}, H={H.shape}, U={U.shape}, "
            f"F={F_state.shape}, H_latch={H_latch.shape}, P_state={P_state.shape}, D={D_state.shape}, "
            f"C={C.shape}, S={S.shape}, Age={Age.shape}, M={M.shape}."
        )
    if F_row.shape != (W.shape[0], 1):
        raise ValueError(f"{name} F_row shape mismatch: expected {(W.shape[0], 1)}, got {F_row.shape}.")
    if F_col.shape != (1, W.shape[1]):
        raise ValueError(f"{name} F_col shape mismatch: expected {(1, W.shape[1])}, got {F_col.shape}.")
    if F_module.shape != (1, 1):
        raise ValueError(f"{name} F_module shape mismatch: expected {(1, 1)}, got {F_module.shape}.")
    ssr_dim = cfg.state_space_reasoner_dim
    if SSR_row_state.shape != (W.shape[0], ssr_dim):
        raise ValueError(f"{name} SSR_row_state shape mismatch: expected {(W.shape[0], ssr_dim)}, got {SSR_row_state.shape}.")
    if SSR_col_state.shape != (W.shape[1], ssr_dim):
        raise ValueError(f"{name} SSR_col_state shape mismatch: expected {(W.shape[1], ssr_dim)}, got {SSR_col_state.shape}.")
    if SSR_module_state.shape != (1, ssr_dim):
        raise ValueError(f"{name} SSR_module_state shape mismatch: expected {(1, ssr_dim)}, got {SSR_module_state.shape}.")
    if SSR_row_credit.shape != (W.shape[0], 1):
        raise ValueError(f"{name} SSR_row_credit shape mismatch: expected {(W.shape[0], 1)}, got {SSR_row_credit.shape}.")
    if SSR_col_credit.shape != (W.shape[1], 1):
        raise ValueError(f"{name} SSR_col_credit shape mismatch: expected {(W.shape[1], 1)}, got {SSR_col_credit.shape}.")
    if SSR_module_credit.shape != (1, 1):
        raise ValueError(f"{name} SSR_module_credit shape mismatch: expected {(1, 1)}, got {SSR_module_credit.shape}.")
    if SSR_last_td_error_abs.shape != (1,) or SSR_last_update_norm.shape != (1,):
        raise ValueError(
            f"{name} SSR last metric shape mismatch: {SSR_last_td_error_abs.shape}, {SSR_last_update_norm.shape}."
        )
    z_dim = ssr_dim + SSR_OBSERVATION_COUNT
    if ssr_state_state_weights.shape != (ssr_dim, ssr_dim):
        raise ValueError(f"{name} ssr_state_state_weights shape mismatch: {ssr_state_state_weights.shape}.")
    if ssr_state_obs_weights.shape != (ssr_dim, SSR_OBSERVATION_COUNT):
        raise ValueError(f"{name} ssr_state_obs_weights shape mismatch: {ssr_state_obs_weights.shape}.")
    if ssr_state_credit_weights.shape != (ssr_dim,):
        raise ValueError(f"{name} ssr_state_credit_weights shape mismatch: {ssr_state_credit_weights.shape}.")
    if ssr_state_bias.shape != (ssr_dim,):
        raise ValueError(f"{name} ssr_state_bias shape mismatch: {ssr_state_bias.shape}.")
    if (
        ssr_write_head.shape != (z_dim,)
        or ssr_protect_head.shape != (z_dim,)
        or ssr_reliability_head.shape != (z_dim,)
        or ssr_collision_head.shape != (z_dim,)
        or ssr_value_head.shape != (z_dim,)
    ):
        raise ValueError(
            f"{name} ssr write/protect/reliability/collision/value head shape mismatch: "
            f"{ssr_write_head.shape}, {ssr_protect_head.shape}, {ssr_reliability_head.shape}, "
            f"{ssr_collision_head.shape}, {ssr_value_head.shape}."
        )
    if (
        ssr_write_bias.shape != (1,)
        or ssr_protect_bias.shape != (1,)
        or ssr_reliability_bias.shape != (1,)
        or ssr_collision_bias.shape != (1,)
        or ssr_value_bias.shape != (1,)
    ):
        raise ValueError(
            f"{name} ssr write/protect/reliability/collision/value bias shape mismatch: "
            f"{ssr_write_bias.shape}, {ssr_protect_bias.shape}, {ssr_reliability_bias.shape}, "
            f"{ssr_collision_bias.shape}, {ssr_value_bias.shape}."
        )
    G = W.grad.detach()
    require_finite_tensor(f"{name}_grad", G)
    require_finite_tensor(f"{name}_pathway", M)
    if reasoner_state_weights.shape != (REASONER_FEATURE_COUNT,):
        raise ValueError(f"{name} reasoner_state_weights shape mismatch: {reasoner_state_weights.shape}.")
    if reasoner_state_bias.shape != (1,):
        raise ValueError(f"{name} reasoner_state_bias shape mismatch: {reasoner_state_bias.shape}.")
    if reasoner_gate_weights.shape != (REASONER_GATE_COUNT, REASONER_FEATURE_COUNT):
        raise ValueError(f"{name} reasoner_gate_weights shape mismatch: {reasoner_gate_weights.shape}.")
    if reasoner_gate_state_weights.shape != (REASONER_GATE_COUNT,):
        raise ValueError(f"{name} reasoner_gate_state_weights shape mismatch: {reasoner_gate_state_weights.shape}.")
    if reasoner_gate_bias.shape != (REASONER_GATE_COUNT,):
        raise ValueError(f"{name} reasoner_gate_bias shape mismatch: {reasoner_gate_bias.shape}.")
    if reasoner_role_weights.shape != (REASONER_ROLE_COUNT, REASONER_FEATURE_COUNT):
        raise ValueError(f"{name} reasoner_role_weights shape mismatch: {reasoner_role_weights.shape}.")
    if reasoner_role_state_weights.shape != (REASONER_ROLE_COUNT,):
        raise ValueError(f"{name} reasoner_role_state_weights shape mismatch: {reasoner_role_state_weights.shape}.")
    if reasoner_role_bias.shape != (REASONER_ROLE_COUNT,):
        raise ValueError(f"{name} reasoner_role_bias shape mismatch: {reasoner_role_bias.shape}.")

    error_pressure = G.abs() * M
    H.mul_(cfg.beta_pressure).add_(error_pressure * control_phase_gate, alpha=1.0 - cfg.beta_pressure)
    U.mul_(cfg.beta_usage).add_(M, alpha=1.0 - cfg.beta_usage)
    Age.add_(1.0).mul_(1.0 - M.clamp(0.0, 1.0))
    route_recency = torch.exp(-Age / cfg.recency_tau)
    warm = min(1.0, float(step) / float(max(1, cfg.warmup_steps)))
    H_norm_pressure = (H / H.max().clamp_min(cfg.eps)).clamp(0.0, 1.0)
    P = warm * torch.sigmoid(cfg.gamma * (H_norm_pressure - cfg.mu))

    usage_max = U.max().clamp_min(cfg.eps)
    U_norm = (U / usage_max).clamp(0.0, 1.0)

    error_norm = (error_pressure / error_pressure.max().clamp_min(cfg.eps)).clamp(0.0, 1.0)
    P_control_state = (P_state * control_phase_gate).clamp(0.0, 1.0)
    D_control_state = (D_state * control_phase_gate).clamp(0.0, 1.0)
    H_latch_control = (H_latch * control_phase_gate).clamp(0.0, 1.0)
    old_route_strength = torch.maximum(P_control_state, H_latch_control * U_norm * route_recency).clamp(0.0, 1.0)
    structural_route_prev = (H_latch_control * U_norm * route_recency).clamp(0.0, 1.0)
    current_input_need = M.max(dim=0).values.clamp(0.0, 1.0)
    structural_input_protect = (
        structural_route_prev.mean(dim=0) * (1.0 - current_input_need)
    ).clamp(0.0, 1.0)
    input_protect = (
        old_route_strength.mean(dim=0)
        + cfg.structural_protect_strength * structural_input_protect
    ).clamp(0.0, 1.0)
    direct_write_protect_effective = cfg.direct_write_protect * input_protect.mean()

    if cfg.write_mode == "gradient":
        effective_write_direction = G * A * M
    elif cfg.write_mode == "direct":
        if direct_write_direction is not None:
            if direct_write_direction.shape != W.shape:
                raise ValueError(
                    f"{name} direct_write_direction shape mismatch: "
                    f"direction={direct_write_direction.shape}, W={W.shape}."
                )
            effective_write_direction = direct_write_direction.to(device=W.device, dtype=W.dtype)
        else:
            if direct_input is None or direct_output_error is None:
                raise RuntimeError(f"{name} direct write mode requires direct_input and direct_output_error.")
            if direct_input.ndim != 2 or direct_output_error.ndim != 2:
                raise ValueError(
                    f"{name} direct tensors must be matrices, got X={direct_input.shape}, E={direct_output_error.shape}."
                )
            if direct_input.shape[0] != direct_output_error.shape[0]:
                raise ValueError(
                    f"{name} direct tensor sample mismatch: X={direct_input.shape}, E={direct_output_error.shape}."
                )
            if direct_input.shape[1] != W.shape[1] or direct_output_error.shape[1] != W.shape[0]:
                raise ValueError(
                    f"{name} direct tensor feature mismatch: X={direct_input.shape}, E={direct_output_error.shape}, "
                    f"W={W.shape}."
                )
            direct_x = direct_input.to(device=W.device, dtype=W.dtype)
            direct_error = direct_output_error.to(device=W.device, dtype=W.dtype)
            sample_count = direct_x.shape[0]
            if sample_count <= 0:
                raise ValueError(f"{name} direct write saw zero samples.")
            if cfg.direct_write_error_scale == "mean":
                direct_error_scaled = direct_error
            elif cfg.direct_write_error_scale == "sample_count":
                direct_error_scaled = direct_error * float(sample_count)
            else:
                raise ValueError(f"{name} unknown direct write error scale: {cfg.direct_write_error_scale!r}.")
            x_columns = direct_x.transpose(0, 1)
            error_columns = direct_error_scaled.transpose(0, 1)
            gram = (x_columns @ x_columns.transpose(0, 1)) / float(sample_count)
            rhs = (error_columns @ x_columns.transpose(0, 1)) / float(sample_count)
            eye = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
            protected_penalty = torch.diag(input_protect)
            solve_matrix = gram + cfg.direct_write_protect * protected_penalty + cfg.direct_write_ridge * eye
            require_finite_tensor(f"{name}_direct_solve_matrix", solve_matrix)
            require_finite_tensor(f"{name}_direct_rhs", rhs)
            effective_write_direction = torch.linalg.solve(solve_matrix, rhs.transpose(0, 1)).transpose(0, 1)
        require_finite_tensor(f"{name}_direct_effective_write_direction", effective_write_direction)
        effective_write_direction = effective_write_direction / A.clamp_min(cfg.min_active_topology)
    else:
        raise ValueError(f"{name} unknown write mode: {cfg.write_mode!r}.")
    require_finite_tensor(f"{name}_effective_write_direction", effective_write_direction)

    direct_write_basis = normalize_pathway(
        (effective_write_direction * A).abs() * M,
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_write_basis",
    )

    formation_evidence = (
        direct_write_basis * M * error_norm * (1.0 - P_control_state) * (1.0 - D_control_state)
    ).clamp(0.0, 1.0)
    F_state.mul_(cfg.beta_formation).add_(formation_evidence, alpha=1.0 - cfg.beta_formation).clamp_(0.0, 1.0)
    row_evidence, col_evidence, module_evidence = multiscale_formation_components(
        formation_evidence,
        M,
        pooling=cfg.formation_multiscale_pooling,
        eps=cfg.eps,
        name=f"{name}_step_formation",
    )
    F_row.mul_(cfg.beta_formation).add_(row_evidence, alpha=1.0 - cfg.beta_formation).clamp_(0.0, 1.0)
    F_col.mul_(cfg.beta_formation).add_(col_evidence, alpha=1.0 - cfg.beta_formation).clamp_(0.0, 1.0)
    F_module.mul_(cfg.beta_formation).add_(module_evidence, alpha=1.0 - cfg.beta_formation).clamp_(0.0, 1.0)
    F_effective = combine_multiscale_formation(F_state, F_row, F_col, F_module, cfg, name=name)
    F_control = (F_effective * control_phase_gate).clamp(0.0, 1.0)
    hardening_enter = F_control >= cfg.hardening_threshold
    hardening_exit = F_control < cfg.hardening_exit_threshold
    H_latch.masked_fill_(hardening_enter, 1.0)
    H_latch.masked_fill_(hardening_exit, 0.0)
    require_finite_tensor(f"{name}_hardening_latch", H_latch)
    H_latch_control = (H_latch * control_phase_gate).clamp(0.0, 1.0)
    protection_evidence = (
        control_phase_gate * F_effective * M * U_norm * route_recency * (1.0 - error_norm)
    ).clamp(0.0, 1.0)
    hardening_protection_evidence = (
        H_latch_control * U_norm * route_recency * (1.0 - error_norm)
    ).clamp(0.0, 1.0)
    if cfg.hardening_protection_strength > 0.0:
        protection_evidence = (
            protection_evidence + cfg.hardening_protection_strength * hardening_protection_evidence
        ).clamp(0.0, 1.0)
    decay_evidence = (
        control_phase_gate
        * (1.0 - M)
        * (1.0 - U_norm)
        * (1.0 - route_recency)
        * (1.0 - F_effective)
        * (1.0 - P_control_state)
    ).clamp(0.0, 1.0)
    P_state.mul_(cfg.beta_protection).add_(protection_evidence, alpha=1.0 - cfg.beta_protection).clamp_(0.0, 1.0)
    D_state.mul_(cfg.beta_decay).add_(decay_evidence, alpha=1.0 - cfg.beta_decay).clamp_(0.0, 1.0)

    P_control_state = (P_state * control_phase_gate).clamp(0.0, 1.0)
    D_control_state = (D_state * control_phase_gate).clamp(0.0, 1.0)
    H_latch_control = (H_latch * control_phase_gate).clamp(0.0, 1.0)
    old_route_strength = torch.maximum(P_control_state, H_latch_control * U_norm * route_recency).clamp(0.0, 1.0)
    protected_capacity = (P_control_state * U_norm).clamp(0.0, 1.0)
    obsolete_capacity = (D_control_state * (1.0 - M) * (1.0 - U_norm) * (1.0 - route_recency)).clamp(0.0, 1.0)
    plastic_capacity = (F_effective + ((1.0 - P_control_state) * M)).clamp(0.0, 1.0)
    free_capacity = (((1.0 - A) * (1.0 - P_control_state)) + obsolete_capacity * A).clamp(0.0, 1.0)
    C.mul_(cfg.beta_capacity).add_(protected_capacity - obsolete_capacity, alpha=1.0 - cfg.beta_capacity).clamp_(-1.0, 1.0)

    route_evidence = {
        "activation_use": M,
        "error_pressure": error_norm,
        "route_frequency": U_norm,
        "route_recency": route_recency,
        "pressure_gate": P,
        "formation": F_effective,
        "protection_state": P_state,
        "decay_state": D_state,
        "direct_write_basis": direct_write_basis,
        "protected_capacity": protected_capacity,
        "free_capacity": free_capacity,
        "plastic_capacity": plastic_capacity,
        "obsolete_capacity": obsolete_capacity,
        "capacity_balance": (C + 1.0) * 0.5,
        "recurrent_state": (S + 1.0) * 0.5,
        "topology": A,
    }
    feature_stack = stack_route_evidence(
        route_evidence,
        name=f"{name}_route_evidence",
    )

    recurrent_positive_evidence = (
        M + error_norm + U_norm + route_recency + F_effective + direct_write_basis + free_capacity + plastic_capacity
    ) / 8.0
    recurrent_negative_evidence = (P_state + D_state + protected_capacity + obsolete_capacity) / 4.0
    recurrent_evidence = (recurrent_positive_evidence - recurrent_negative_evidence).clamp(-1.0, 1.0)
    if cfg.reasoner_policy == "fixed_geometric":
        state_drive = recurrent_evidence
    elif cfg.reasoner_policy == "learned":
        learned_state_drive = torch.tensordot(feature_stack, reasoner_state_weights, dims=([-1], [0])) + reasoner_state_bias[0]
        state_drive = (recurrent_evidence + learned_state_drive).clamp(-1.0, 1.0)
    else:
        raise ValueError(f"{name} unknown reasoner policy: {cfg.reasoner_policy!r}.")
    S_before = S.detach().clone()
    S.copy_(torch.tanh(cfg.beta_recurrent * S + (1.0 - cfg.beta_recurrent) * state_drive))
    recurrent_delta = (S - S_before).abs()

    gate_logits = (
        torch.einsum("...f,gf->...g", feature_stack, reasoner_gate_weights)
        + S.unsqueeze(-1) * reasoner_gate_state_weights
        + reasoner_gate_bias
    )
    gates = torch.sigmoid(gate_logits)
    write_gate_raw = gates[..., 0].clamp(0.0, 1.0)
    protect_gate_raw = gates[..., 1].clamp(0.0, 1.0)
    rewire_gate_raw = gates[..., 2].clamp(0.0, 1.0)
    forget_gate_raw = gates[..., 3].clamp(0.0, 1.0)
    compress_gate_raw = gates[..., 4].clamp(0.0, 1.0)
    if cfg.state_space_reasoner_dim > 0:
        ssr_row_obs, ssr_col_obs, ssr_module_obs = reduce_route_observations(
            feature_stack,
            M,
            pooling=cfg.formation_multiscale_pooling,
            eps=cfg.eps,
            name=f"{name}_ssr_observation",
        )
        (
            ssr_row_z,
            ssr_row_logit,
            ssr_row_gate,
            ssr_row_protect_logit,
            ssr_row_protect_gate,
            ssr_row_reliability_logit,
            ssr_row_reliability,
            ssr_row_collision_logit,
            ssr_row_collision,
            ssr_row_value,
        ) = state_space_route_step(
            SSR_row_state,
            ssr_row_obs,
            SSR_row_credit,
            ssr_state_state_weights,
            ssr_state_obs_weights,
            ssr_state_credit_weights,
            ssr_state_bias,
            ssr_write_head,
            ssr_write_bias,
            ssr_protect_head,
            ssr_protect_bias,
            ssr_reliability_head,
            ssr_reliability_bias,
            ssr_collision_head,
            ssr_collision_bias,
            ssr_value_head,
            ssr_value_bias,
            name=f"{name}_ssr_row",
        )
        (
            ssr_col_z,
            ssr_col_logit,
            ssr_col_gate,
            ssr_col_protect_logit,
            ssr_col_protect_gate,
            ssr_col_reliability_logit,
            ssr_col_reliability,
            ssr_col_collision_logit,
            ssr_col_collision,
            ssr_col_value,
        ) = state_space_route_step(
            SSR_col_state,
            ssr_col_obs,
            SSR_col_credit,
            ssr_state_state_weights,
            ssr_state_obs_weights,
            ssr_state_credit_weights,
            ssr_state_bias,
            ssr_write_head,
            ssr_write_bias,
            ssr_protect_head,
            ssr_protect_bias,
            ssr_reliability_head,
            ssr_reliability_bias,
            ssr_collision_head,
            ssr_collision_bias,
            ssr_value_head,
            ssr_value_bias,
            name=f"{name}_ssr_col",
        )
        (
            ssr_module_z,
            ssr_module_logit,
            ssr_module_gate,
            ssr_module_protect_logit,
            ssr_module_protect_gate,
            ssr_module_reliability_logit,
            ssr_module_reliability,
            ssr_module_collision_logit,
            ssr_module_collision,
            ssr_module_value,
        ) = state_space_route_step(
            SSR_module_state,
            ssr_module_obs,
            SSR_module_credit,
            ssr_state_state_weights,
            ssr_state_obs_weights,
            ssr_state_credit_weights,
            ssr_state_bias,
            ssr_write_head,
            ssr_write_bias,
            ssr_protect_head,
            ssr_protect_bias,
            ssr_reliability_head,
            ssr_reliability_bias,
            ssr_collision_head,
            ssr_collision_bias,
            ssr_value_head,
            ssr_value_bias,
            name=f"{name}_ssr_module",
        )
        _, ssr_row_rewire = state_space_sigmoid_head(
            ssr_row_z,
            ssr_rewire_head,
            ssr_rewire_bias,
            name=f"{name}_ssr_row_rewire",
        )
        _, ssr_col_rewire = state_space_sigmoid_head(
            ssr_col_z,
            ssr_rewire_head,
            ssr_rewire_bias,
            name=f"{name}_ssr_col_rewire",
        )
        _, ssr_module_rewire = state_space_sigmoid_head(
            ssr_module_z,
            ssr_rewire_head,
            ssr_rewire_bias,
            name=f"{name}_ssr_module_rewire",
        )
        _, ssr_row_forget = state_space_sigmoid_head(
            ssr_row_z,
            ssr_forget_head,
            ssr_forget_bias,
            name=f"{name}_ssr_row_forget",
        )
        _, ssr_col_forget = state_space_sigmoid_head(
            ssr_col_z,
            ssr_forget_head,
            ssr_forget_bias,
            name=f"{name}_ssr_col_forget",
        )
        _, ssr_module_forget = state_space_sigmoid_head(
            ssr_module_z,
            ssr_forget_head,
            ssr_forget_bias,
            name=f"{name}_ssr_module_forget",
        )
        _, ssr_row_compress = state_space_sigmoid_head(
            ssr_row_z,
            ssr_compress_head,
            ssr_compress_bias,
            name=f"{name}_ssr_row_compress",
        )
        _, ssr_col_compress = state_space_sigmoid_head(
            ssr_col_z,
            ssr_compress_head,
            ssr_compress_bias,
            name=f"{name}_ssr_col_compress",
        )
        _, ssr_module_compress = state_space_sigmoid_head(
            ssr_module_z,
            ssr_compress_head,
            ssr_compress_bias,
            name=f"{name}_ssr_module_compress",
        )
        _, ssr_row_gain = state_space_sigmoid_head(
            ssr_row_z,
            ssr_gain_head,
            ssr_gain_bias,
            name=f"{name}_ssr_row_gain",
        )
        _, ssr_col_gain = state_space_sigmoid_head(
            ssr_col_z,
            ssr_gain_head,
            ssr_gain_bias,
            name=f"{name}_ssr_col_gain",
        )
        _, ssr_module_gain = state_space_sigmoid_head(
            ssr_module_z,
            ssr_gain_head,
            ssr_gain_bias,
            name=f"{name}_ssr_module_gain",
        )
        _, ssr_row_capacity = state_space_sigmoid_head(
            ssr_row_z,
            ssr_capacity_head,
            ssr_capacity_bias,
            name=f"{name}_ssr_row_capacity",
        )
        _, ssr_col_capacity = state_space_sigmoid_head(
            ssr_col_z,
            ssr_capacity_head,
            ssr_capacity_bias,
            name=f"{name}_ssr_col_capacity",
        )
        _, ssr_module_capacity = state_space_sigmoid_head(
            ssr_module_z,
            ssr_capacity_head,
            ssr_capacity_bias,
            name=f"{name}_ssr_module_capacity",
        )
        _, ssr_row_forget_safe = state_space_sigmoid_head(
            ssr_row_z,
            ssr_forget_safe_head,
            ssr_forget_safe_bias,
            name=f"{name}_ssr_row_forget_safe",
        )
        _, ssr_col_forget_safe = state_space_sigmoid_head(
            ssr_col_z,
            ssr_forget_safe_head,
            ssr_forget_safe_bias,
            name=f"{name}_ssr_col_forget_safe",
        )
        _, ssr_module_forget_safe = state_space_sigmoid_head(
            ssr_module_z,
            ssr_forget_safe_head,
            ssr_forget_safe_bias,
            name=f"{name}_ssr_module_forget_safe",
        )
        _, ssr_row_priority = state_space_sigmoid_head(
            ssr_row_z,
            ssr_priority_head,
            ssr_priority_bias,
            name=f"{name}_ssr_row_priority",
        )
        _, ssr_col_priority = state_space_sigmoid_head(
            ssr_col_z,
            ssr_priority_head,
            ssr_priority_bias,
            name=f"{name}_ssr_col_priority",
        )
        _, ssr_module_priority = state_space_sigmoid_head(
            ssr_module_z,
            ssr_priority_head,
            ssr_priority_bias,
            name=f"{name}_ssr_module_priority",
        )
        ssr_write_logit = (
            ssr_row_logit.reshape(-1, 1) + ssr_col_logit.reshape(1, -1) + ssr_module_logit.reshape(1, 1)
        ) / 3.0
        ssr_write_gate = torch.sigmoid(ssr_write_logit).clamp(0.0, 1.0)
        ssr_protect_logit = (
            ssr_row_protect_logit.reshape(-1, 1)
            + ssr_col_protect_logit.reshape(1, -1)
            + ssr_module_protect_logit.reshape(1, 1)
        ) / 3.0
        ssr_protect_gate = torch.sigmoid(ssr_protect_logit).clamp(0.0, 1.0)
        ssr_reliability_logit = (
            ssr_row_reliability_logit.reshape(-1, 1)
            + ssr_col_reliability_logit.reshape(1, -1)
            + ssr_module_reliability_logit.reshape(1, 1)
        ) / 3.0
        ssr_reliability = torch.sigmoid(ssr_reliability_logit).clamp(0.0, 1.0)
        ssr_collision_logit = (
            ssr_row_collision_logit.reshape(-1, 1)
            + ssr_col_collision_logit.reshape(1, -1)
            + ssr_module_collision_logit.reshape(1, 1)
        ) / 3.0
        ssr_collision = torch.sigmoid(ssr_collision_logit).clamp(0.0, 1.0)
        ssr_protect_eff = (ssr_protect_gate * ssr_reliability * ssr_collision).clamp(0.0, 1.0)
        ssr_value_pred = (
            ssr_row_value.reshape(-1, 1) + ssr_col_value.reshape(1, -1) + ssr_module_value.reshape(1, 1)
        ) / 3.0
        ssr_rewire_gate = combine_state_space_surface(
            ssr_row_rewire,
            ssr_col_rewire,
            ssr_module_rewire,
            name=f"{name}_ssr_rewire",
        )
        ssr_forget_gate = combine_state_space_surface(
            ssr_row_forget,
            ssr_col_forget,
            ssr_module_forget,
            name=f"{name}_ssr_forget",
        )
        ssr_compress_gate = combine_state_space_surface(
            ssr_row_compress,
            ssr_col_compress,
            ssr_module_compress,
            name=f"{name}_ssr_compress",
        )
        ssr_gain_pred = combine_state_space_surface(
            ssr_row_gain,
            ssr_col_gain,
            ssr_module_gain,
            name=f"{name}_ssr_gain",
        )
        ssr_capacity_cost = combine_state_space_surface(
            ssr_row_capacity,
            ssr_col_capacity,
            ssr_module_capacity,
            name=f"{name}_ssr_capacity",
        )
        ssr_forget_safe = combine_state_space_surface(
            ssr_row_forget_safe,
            ssr_col_forget_safe,
            ssr_module_forget_safe,
            name=f"{name}_ssr_forget_safe",
        )
        ssr_priority = combine_state_space_surface(
            ssr_row_priority,
            ssr_col_priority,
            ssr_module_priority,
            name=f"{name}_ssr_priority",
        )
        write_gate = (
            (1.0 - cfg.state_space_write_blend) * write_gate_raw
            + cfg.state_space_write_blend * ssr_write_gate
        ).clamp(0.0, 1.0)
        hand_protect_gate = (protect_gate_raw * maturity_tensor).clamp(0.0, 1.0)
        protect_gate = (
            (1.0 - cfg.state_space_protect_blend) * hand_protect_gate
            + cfg.state_space_protect_blend * ssr_protect_eff
        ).clamp(0.0, 1.0)
    else:
        ssr_row_obs = torch.zeros(W.shape[0], SSR_OBSERVATION_COUNT, device=W.device, dtype=W.dtype)
        ssr_col_obs = torch.zeros(W.shape[1], SSR_OBSERVATION_COUNT, device=W.device, dtype=W.dtype)
        ssr_module_obs = torch.zeros(1, SSR_OBSERVATION_COUNT, device=W.device, dtype=W.dtype)
        ssr_row_z = torch.zeros(W.shape[0], SSR_OBSERVATION_COUNT, device=W.device, dtype=W.dtype)
        ssr_col_z = torch.zeros(W.shape[1], SSR_OBSERVATION_COUNT, device=W.device, dtype=W.dtype)
        ssr_module_z = torch.zeros(1, SSR_OBSERVATION_COUNT, device=W.device, dtype=W.dtype)
        ssr_row_gate = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        ssr_col_gate = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)
        ssr_module_gate = torch.zeros(1, device=W.device, dtype=W.dtype)
        ssr_row_protect_gate = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        ssr_col_protect_gate = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)
        ssr_module_protect_gate = torch.zeros(1, device=W.device, dtype=W.dtype)
        ssr_row_reliability = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        ssr_col_reliability = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)
        ssr_module_reliability = torch.zeros(1, device=W.device, dtype=W.dtype)
        ssr_row_collision = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        ssr_col_collision = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)
        ssr_module_collision = torch.zeros(1, device=W.device, dtype=W.dtype)
        ssr_row_value = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        ssr_col_value = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)
        ssr_module_value = torch.zeros(1, device=W.device, dtype=W.dtype)
        ssr_write_gate = torch.zeros_like(write_gate_raw)
        ssr_protect_gate = torch.zeros_like(protect_gate_raw)
        ssr_reliability = torch.zeros_like(protect_gate_raw)
        ssr_collision = torch.zeros_like(protect_gate_raw)
        ssr_protect_eff = torch.zeros_like(protect_gate_raw)
        ssr_value_pred = torch.zeros_like(write_gate_raw)
        ssr_rewire_gate = torch.zeros_like(rewire_gate_raw)
        ssr_forget_gate = torch.zeros_like(forget_gate_raw)
        ssr_compress_gate = torch.zeros_like(compress_gate_raw)
        ssr_gain_pred = torch.zeros_like(write_gate_raw)
        ssr_capacity_cost = torch.zeros_like(write_gate_raw)
        ssr_forget_safe = torch.zeros_like(forget_gate_raw)
        ssr_priority = torch.zeros_like(write_gate_raw)
        write_gate = write_gate_raw
        protect_gate = (protect_gate_raw * maturity_tensor).clamp(0.0, 1.0)
    if cfg.reasoner_policy == "fixed_geometric":
        fixed_controls = fixed_geometric_controls(
            M=M,
            error_norm=error_norm,
            direct_write_basis=direct_write_basis,
            F_effective=F_effective,
            P_state=P_control_state,
            D_state=D_control_state,
            U_norm=U_norm,
            route_recency=route_recency,
            old_route_strength=old_route_strength,
            free_capacity=free_capacity,
            plastic_capacity=plastic_capacity,
            obsolete_capacity=obsolete_capacity,
            protected_capacity=protected_capacity,
            A=A,
            failed_write_signal=failed_write_tensor,
            cfg=cfg,
            name=name,
        )
        sculpt_write_gate = normalize_pathway(
            (direct_write_basis * error_norm * M).clamp(0.0, 1.0),
            percentile=cfg.pathway_percentile,
            eps=cfg.eps,
            name=f"{name}_sculpt_write_gate",
        )
        write_gate = (
            (1.0 - control_phase_gate) * sculpt_write_gate
            + control_phase_gate * fixed_controls["write_gate"]
        ).clamp(0.0, 1.0)
        protect_gate = (control_phase_gate * fixed_controls["protect_gate"]).clamp(0.0, 1.0)
        rewire_gate = (control_phase_gate * fixed_controls["rewire_gate"]).clamp(0.0, 1.0)
        forget_gate = (control_phase_gate * fixed_controls["forget_gate"]).clamp(0.0, 1.0)
        compress_gate = (control_phase_gate * fixed_controls["compress_gate"]).clamp(0.0, 1.0)
        ssr_write_gate = write_gate
        ssr_protect_gate = protect_gate
        ssr_rewire_gate = rewire_gate
        ssr_forget_gate = forget_gate
        ssr_compress_gate = compress_gate
        ssr_gain_pred = fixed_controls["gain_pred"]
        ssr_capacity_cost = fixed_controls["capacity_cost"]
        ssr_forget_safe = fixed_controls["forget_safe"]
        ssr_priority = fixed_controls["priority"]
        ssr_collision = fixed_controls["collision"]
        ssr_reliability = fixed_controls["reliability"]
        ssr_protect_eff = fixed_controls["protect_eff"]
        ssr_value_pred = fixed_controls["value_pred"]
        write_priority_control = ((1.0 - control_phase_gate) + control_phase_gate * ssr_priority).clamp(0.0, 1.0)
        write_gate_raw = write_gate
        protect_gate_raw = protect_gate
        rewire_gate_raw = rewire_gate
        forget_gate_raw = forget_gate
        compress_gate_raw = compress_gate
    elif cfg.reasoner_policy == "learned":
        base_rewire_gate = (rewire_gate_raw * failed_write_tensor).clamp(0.0, 1.0)
        ssr_rewire_eff = (ssr_rewire_gate * ssr_collision * failed_write_tensor).clamp(0.0, 1.0)
        rewire_gate = (
            (1.0 - cfg.state_space_rewire_blend) * base_rewire_gate
            + cfg.state_space_rewire_blend * ssr_rewire_eff
        ).clamp(0.0, 1.0)
        base_forget_gate = (forget_gate_raw * maturity_tensor).clamp(0.0, 1.0)
        ssr_forget_eff = (ssr_forget_gate * ssr_forget_safe * maturity_tensor).clamp(0.0, 1.0)
        forget_gate = (
            (1.0 - cfg.state_space_forget_blend) * base_forget_gate
            + cfg.state_space_forget_blend * ssr_forget_eff
        ).clamp(0.0, 1.0)
        base_compress_gate = (compress_gate_raw * maturity_tensor).clamp(0.0, 1.0)
        ssr_compress_eff = (ssr_compress_gate * ssr_capacity_cost * maturity_tensor).clamp(0.0, 1.0)
        compress_gate = (
            (1.0 - cfg.state_space_compress_blend) * base_compress_gate
            + cfg.state_space_compress_blend * ssr_compress_eff
        ).clamp(0.0, 1.0)
        ssr_write_priority = (ssr_gain_pred * ssr_priority * (1.0 - ssr_collision)).clamp(0.0, 1.0)
        write_priority_control = (
            (1.0 - cfg.state_space_priority_blend)
            + cfg.state_space_priority_blend * ssr_write_priority
        ).clamp(0.0, 1.0)
    else:
        raise ValueError(f"{name} unknown reasoner policy: {cfg.reasoner_policy!r}.")

    write_pressure = normalize_pathway(
        (effective_write_direction.abs() * write_gate).clamp_min(0.0),
        percentile=cfg.pathway_percentile,
        eps=cfg.eps,
        name=f"{name}_write_pressure",
    )
    stable_protect_need = (cfg.protect_old_route_floor * old_route_strength).clamp(0.0, 1.0)
    collision_protect_need = (
        cfg.protect_collision_strength * old_route_strength * write_pressure * M
    ).clamp(0.0, 1.0)
    protect_need = torch.maximum(stable_protect_need, collision_protect_need).clamp(0.0, 1.0)
    require_finite_tensor(f"{name}_protect_need", protect_need)

    role_logits = (
        torch.einsum("...f,rf->...r", feature_stack, reasoner_role_weights)
        + S.unsqueeze(-1) * reasoner_role_state_weights
        + reasoner_role_bias
    )
    role_belief = torch.softmax(role_logits, dim=-1)
    role_entropy = -(role_belief * role_belief.clamp_min(cfg.eps).log()).sum(dim=-1) / math.log(float(REASONER_ROLE_COUNT))

    continual_write_utility = (
        direct_write_basis * (error_norm + F_effective).clamp(0.0, 1.0) * plastic_capacity * (1.0 - P_control_state)
    ).clamp(0.0, 1.0)
    sculpt_write_utility = (direct_write_basis * error_norm * M).clamp(0.0, 1.0)
    write_utility = (
        (1.0 - control_phase_gate) * sculpt_write_utility
        + control_phase_gate * continual_write_utility
    ).clamp(0.0, 1.0)
    gate_utility = torch.stack(
        [
            write_utility,
            protect_need,
            (error_norm * free_capacity * (1.0 - P_state) + D_state * free_capacity).clamp(0.0, 1.0),
            (D_state * (1.0 - M) * (1.0 - error_norm) * (1.0 - F_effective) * (1.0 - P_state)).clamp(0.0, 1.0),
            (P_state * U_norm * route_recency * (1.0 - error_norm)).clamp(0.0, 1.0),
        ],
        dim=-1,
    )
    role_utility = torch.stack(
        [
            ((1.0 - U_norm) * (1.0 - M) * (1.0 - error_norm)).clamp(0.0, 1.0),
            (error_norm * (1.0 - U_norm) * (1.0 - M)).clamp(0.0, 1.0),
            F_effective.clamp(0.0, 1.0),
            (F_effective * U_norm * M).clamp(0.0, 1.0),
            P_state.clamp(0.0, 1.0),
            (P_state * error_norm).clamp(0.0, 1.0),
            D_state.clamp(0.0, 1.0),
        ],
        dim=-1,
    )
    role_target = role_utility / role_utility.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)

    if cfg.reasoner_policy == "fixed_geometric":
        fixed_gate_surface = torch.stack([write_gate, protect_gate, rewire_gate, forget_gate, compress_gate], dim=-1)
        gate_error = gate_utility - fixed_gate_surface
        role_belief = role_target
        role_entropy = -(role_belief * role_belief.clamp_min(cfg.eps).log()).sum(dim=-1) / math.log(
            float(REASONER_ROLE_COUNT)
        )
        reasoner_update_norm_tensor = torch.zeros((), device=W.device, dtype=W.dtype)
        reasoner_update_scale = torch.ones((), device=W.device, dtype=W.dtype)
    elif cfg.reasoner_policy == "learned":
        gate_error = gate_utility - gates
        role_error = role_target - role_belief
        gate_lr_scale = cfg.internal_gate_lr_scale * torch.as_tensor(
            [
                cfg.internal_write_gate_lr_scale,
                cfg.internal_protect_gate_lr_scale,
                cfg.internal_rewire_gate_lr_scale,
                cfg.internal_forget_gate_lr_scale,
                cfg.internal_compress_gate_lr_scale,
            ],
            device=W.device,
            dtype=W.dtype,
        )
        gate_local_grad = gate_error * gates * (1.0 - gates) * gate_lr_scale
        role_local_grad = role_error
        feature_flat = feature_stack.reshape(-1, REASONER_FEATURE_COUNT)
        gate_grad_flat = gate_local_grad.reshape(-1, REASONER_GATE_COUNT)
        role_grad_flat = role_local_grad.reshape(-1, REASONER_ROLE_COUNT)
        state_grad = (
            (gate_local_grad * reasoner_gate_state_weights).sum(dim=-1)
            + (role_local_grad * reasoner_role_state_weights).sum(dim=-1)
        ) * (1.0 - S * S)
        state_grad_flat = state_grad.reshape(-1)
        reasoner_gate_delta = torch.einsum("ng,nf->gf", gate_grad_flat, feature_flat) / float(feature_flat.shape[0])
        reasoner_gate_state_delta = (gate_local_grad * S.unsqueeze(-1)).reshape(-1, REASONER_GATE_COUNT).mean(dim=0)
        reasoner_gate_bias_delta = gate_grad_flat.mean(dim=0)
        reasoner_role_delta = torch.einsum("nr,nf->rf", role_grad_flat, feature_flat) / float(feature_flat.shape[0])
        reasoner_role_state_delta = (role_local_grad * S.unsqueeze(-1)).reshape(-1, REASONER_ROLE_COUNT).mean(dim=0)
        reasoner_role_bias_delta = role_grad_flat.mean(dim=0)
        reasoner_state_delta = (state_grad_flat.unsqueeze(-1) * feature_flat).mean(dim=0)
        reasoner_state_bias_delta = state_grad_flat.mean().reshape_as(reasoner_state_bias)

        update_tensors = [
            reasoner_gate_delta,
            reasoner_gate_state_delta,
            reasoner_gate_bias_delta,
            reasoner_role_delta,
            reasoner_role_state_delta,
            reasoner_role_bias_delta,
            reasoner_state_delta,
            reasoner_state_bias_delta,
        ]
        reasoner_update_norm_tensor = sum(torch.linalg.vector_norm(item) for item in update_tensors)
        reasoner_update_scale = torch.clamp(
            torch.as_tensor(cfg.reasoner_update_clip, device=W.device, dtype=W.dtype)
            / reasoner_update_norm_tensor.clamp_min(cfg.eps),
            max=1.0,
        )
        decay = 1.0 - cfg.reasoner_lr * cfg.reasoner_weight_decay
        reasoner_gate_weights.mul_(decay).add_(
            reasoner_gate_delta,
            alpha=cfg.reasoner_lr * scalar(reasoner_update_scale),
        )
        reasoner_gate_state_weights.mul_(decay).add_(
            reasoner_gate_state_delta,
            alpha=cfg.reasoner_lr * scalar(reasoner_update_scale),
        )
        reasoner_gate_bias.mul_(decay).add_(
            reasoner_gate_bias_delta,
            alpha=cfg.reasoner_lr * scalar(reasoner_update_scale),
        )
        reasoner_role_weights.mul_(decay).add_(
            reasoner_role_delta,
            alpha=cfg.reasoner_lr * scalar(reasoner_update_scale),
        )
        reasoner_role_state_weights.mul_(decay).add_(
            reasoner_role_state_delta,
            alpha=cfg.reasoner_lr * scalar(reasoner_update_scale),
        )
        reasoner_role_bias.mul_(decay).add_(
            reasoner_role_bias_delta,
            alpha=cfg.reasoner_lr * scalar(reasoner_update_scale),
        )
        reasoner_state_weights.mul_(decay).add_(
            reasoner_state_delta,
            alpha=cfg.reasoner_lr * scalar(reasoner_update_scale),
        )
        reasoner_state_bias.mul_(decay).add_(
            reasoner_state_bias_delta,
            alpha=cfg.reasoner_lr * scalar(reasoner_update_scale),
        )
    else:
        raise ValueError(f"{name} unknown reasoner policy: {cfg.reasoner_policy!r}.")

    write_score = (write_gate * write_utility * write_priority_control).clamp(0.0, 1.0)
    protect_score = (protect_gate * gate_utility[..., 1]).clamp(0.0, 1.0)
    rewire_score = (rewire_gate * gate_utility[..., 2]).clamp(0.0, 1.0)
    forget_score = (forget_gate * gate_utility[..., 3]).clamp(0.0, 1.0)
    compress_score = (compress_gate * gate_utility[..., 4]).clamp(0.0, 1.0)
    write_mask, write_edit_count, write_score_mass = dynamic_edit_mask(
        write_score,
        eps=cfg.eps,
        name=f"{name}_write_score",
    )
    protect_mask, protect_edit_count, protect_score_mass = dynamic_edit_mask(
        protect_score,
        eps=cfg.eps,
        name=f"{name}_protect_score",
    )
    rewire_mask, rewire_edit_count, rewire_score_mass = dynamic_edit_mask(
        rewire_score,
        eps=cfg.eps,
        name=f"{name}_rewire_score",
    )
    forget_mask, forget_edit_count, forget_score_mass = dynamic_edit_mask(
        forget_score,
        eps=cfg.eps,
        name=f"{name}_forget_score",
    )
    compress_mask, compress_edit_count, compress_score_mass = dynamic_edit_mask(
        compress_score,
        eps=cfg.eps,
        name=f"{name}_compress_score",
    )

    grow = cfg.grow_lr * rewire_score * rewire_mask * (1.0 - A)
    prune = cfg.prune_lr * forget_score * forget_mask * A
    A_before = A.detach().clone()
    A.add_(grow - prune).clamp_(0.0, 1.0)
    topology_delta = A - A_before
    topology_delta_abs = topology_delta.abs()
    selected_topology_mask = ((rewire_mask + forget_mask) > 0.0).to(dtype=W.dtype)
    unselected_topology_mask = 1.0 - selected_topology_mask

    direct_write_norm = torch.linalg.matrix_norm(effective_write_direction)
    G_write = effective_write_direction * write_gate * write_mask
    current_route_need = torch.maximum(M, write_mask).clamp(0.0, 1.0)
    structural_protection = (
        cfg.structural_protect_strength
        * maturity_tensor
        * H_latch
        * U_norm
        * route_recency
        * (1.0 - current_route_need)
    ).clamp(0.0, 1.0)
    protect_field = (protect_gate * protect_mask * protect_need + structural_protection).clamp(0.0, 1.0)
    row_pressure = (protect_field * M).sum(dim=1, keepdim=True) / M.sum(dim=1, keepdim=True).clamp_min(cfg.eps)
    selected_weight_basis = W * write_mask
    row_dot = (G_write * selected_weight_basis).sum(dim=1, keepdim=True)
    row_norm = (selected_weight_basis * selected_weight_basis).sum(dim=1, keepdim=True).clamp_min(cfg.eps)
    projection = (row_dot / row_norm) * selected_weight_basis
    protected_projection = row_pressure * projection
    G_safe = (G_write - protected_projection) * write_mask

    raw_write_norm = torch.linalg.matrix_norm(G_write)
    protected_projection_norm = torch.linalg.matrix_norm(protected_projection)
    safe_norm = torch.linalg.matrix_norm(G_safe)
    scale = torch.clamp(
        torch.as_tensor(cfg.max_step_norm, device=W.device, dtype=W.dtype) / safe_norm.clamp_min(cfg.eps),
        max=1.0,
    )
    delta = -cfg.lr * scale * G_safe

    inactive = ((1.0 - M) * (1.0 - U_norm) * (1.0 - route_recency) * (1.0 - F_effective) * (1.0 - P_state)).clamp(
        0.0,
        1.0,
    )
    forget = (cfg.forget_lr * forget_score * forget_mask).clamp(0.0, 1.0)
    W_before = W.detach().clone()
    W.mul_(1.0 - forget).add_(delta)
    weight_delta = W - W_before
    selected_weight_mask = ((write_mask + forget_mask) > 0.0).to(dtype=W.dtype)
    unselected_weight_mask = 1.0 - selected_weight_mask
    W.grad = None

    crystalline = (P_state >= cfg.crystalline_threshold).to(dtype=W.dtype)
    hardening = ((H_latch >= 0.5) & (P_state < cfg.crystalline_threshold)).to(dtype=W.dtype)
    sculpting = (1.0 - torch.maximum(hardening, crystalline)).clamp(0.0, 1.0)
    active_mass = M.sum().clamp_min(cfg.eps)
    reasoner_weight_norm_tensor = (
        torch.linalg.vector_norm(reasoner_state_weights)
        + torch.linalg.vector_norm(reasoner_state_bias)
        + torch.linalg.matrix_norm(reasoner_gate_weights)
        + torch.linalg.vector_norm(reasoner_gate_state_weights)
        + torch.linalg.vector_norm(reasoner_gate_bias)
        + torch.linalg.matrix_norm(reasoner_role_weights)
        + torch.linalg.vector_norm(reasoner_role_state_weights)
        + torch.linalg.vector_norm(reasoner_role_bias)
    )
    stats = GCOModuleStats(
        name=name,
        pressure_mean=scalar(P.mean()),
        pressure_max=scalar(P.max()),
        pathway_mean=scalar(M.mean()),
        pathway_max=scalar(M.max()),
        error_pressure_mean=scalar(error_pressure.mean()),
        error_pressure_max=scalar(error_pressure.max()),
        formation_pressure_mean=scalar(F_state.mean()),
        formation_pressure_max=scalar(F_state.max()),
        formation_effective_mean=scalar(F_effective.mean()),
        formation_effective_max=scalar(F_effective.max()),
        formation_row_mean=scalar(F_row.mean()),
        formation_row_max=scalar(F_row.max()),
        formation_col_mean=scalar(F_col.mean()),
        formation_col_max=scalar(F_col.max()),
        formation_module_mean=scalar(F_module.mean()),
        formation_module_max=scalar(F_module.max()),
        hardening_latch_mean=scalar(H_latch.mean()),
        hardening_latch_max=scalar(H_latch.max()),
        ssr_row_state_norm_mean=scalar(torch.linalg.vector_norm(SSR_row_state, dim=1).mean()),
        ssr_row_state_norm_max=scalar(torch.linalg.vector_norm(SSR_row_state, dim=1).max()),
        ssr_col_state_norm_mean=scalar(torch.linalg.vector_norm(SSR_col_state, dim=1).mean()),
        ssr_col_state_norm_max=scalar(torch.linalg.vector_norm(SSR_col_state, dim=1).max()),
        ssr_module_state_norm_mean=scalar(torch.linalg.vector_norm(SSR_module_state, dim=1).mean()),
        ssr_module_state_norm_max=scalar(torch.linalg.vector_norm(SSR_module_state, dim=1).max()),
        ssr_write_gate_mean=scalar(ssr_write_gate.mean()),
        ssr_write_gate_max=scalar(ssr_write_gate.max()),
        ssr_protect_gate_mean=scalar(ssr_protect_gate.mean()),
        ssr_protect_gate_max=scalar(ssr_protect_gate.max()),
        ssr_reliability_mean=scalar(ssr_reliability.mean()),
        ssr_reliability_max=scalar(ssr_reliability.max()),
        ssr_collision_mean=scalar(ssr_collision.mean()),
        ssr_collision_max=scalar(ssr_collision.max()),
        ssr_protect_eff_mean=scalar(ssr_protect_eff.mean()),
        ssr_protect_eff_max=scalar(ssr_protect_eff.max()),
        ssr_rewire_gate_mean=scalar(ssr_rewire_gate.mean()),
        ssr_rewire_gate_max=scalar(ssr_rewire_gate.max()),
        ssr_forget_gate_mean=scalar(ssr_forget_gate.mean()),
        ssr_forget_gate_max=scalar(ssr_forget_gate.max()),
        ssr_compress_gate_mean=scalar(ssr_compress_gate.mean()),
        ssr_compress_gate_max=scalar(ssr_compress_gate.max()),
        ssr_gain_pred_mean=scalar(ssr_gain_pred.mean()),
        ssr_gain_pred_max=scalar(ssr_gain_pred.max()),
        ssr_capacity_cost_mean=scalar(ssr_capacity_cost.mean()),
        ssr_capacity_cost_max=scalar(ssr_capacity_cost.max()),
        ssr_forget_safe_mean=scalar(ssr_forget_safe.mean()),
        ssr_forget_safe_max=scalar(ssr_forget_safe.max()),
        ssr_priority_mean=scalar(ssr_priority.mean()),
        ssr_priority_max=scalar(ssr_priority.max()),
        ssr_value_pred_mean=scalar(ssr_value_pred.mean()),
        ssr_value_pred_max=scalar(ssr_value_pred.max()),
        ssr_credit_mean=scalar((SSR_row_credit.mean() + SSR_col_credit.mean() + SSR_module_credit.mean()) / 3.0),
        ssr_credit_max=scalar(torch.maximum(torch.maximum(SSR_row_credit.max(), SSR_col_credit.max()), SSR_module_credit.max())),
        ssr_td_error_abs_mean=scalar(SSR_last_td_error_abs.mean()),
        ssr_td_error_abs_max=scalar(SSR_last_td_error_abs.max()),
        ssr_update_norm=scalar(SSR_last_update_norm.mean()),
        write_pressure_mean=scalar(write_pressure.mean()),
        write_pressure_max=scalar(write_pressure.max()),
        protect_need_mean=scalar(protect_need.mean()),
        protect_need_max=scalar(protect_need.max()),
        protection_pressure_mean=scalar(P_state.mean()),
        protection_pressure_max=scalar(P_state.max()),
        structural_protection_mean=scalar(structural_protection.mean()),
        structural_protection_max=scalar(structural_protection.max()),
        structural_input_protection_mean=scalar(structural_input_protect.mean()),
        structural_input_protection_max=scalar(structural_input_protect.max()),
        direct_write_protect_effective=scalar(direct_write_protect_effective),
        decay_pressure_mean=scalar(D_state.mean()),
        decay_pressure_max=scalar(D_state.max()),
        direct_write_basis_mean=scalar(direct_write_basis.mean()),
        direct_write_basis_max=scalar(direct_write_basis.max()),
        route_age_mean=scalar(Age.mean()),
        route_age_max=scalar(Age.max()),
        route_recency_mean=scalar(route_recency.mean()),
        route_recency_max=scalar(route_recency.max()),
        recurrent_state_mean=scalar(S.mean()),
        recurrent_state_abs_mean=scalar(S.abs().mean()),
        recurrent_state_delta_mean=scalar(recurrent_delta.mean()),
        protected_capacity_mean=scalar(protected_capacity.mean()),
        free_capacity_mean=scalar(free_capacity.mean()),
        plastic_capacity_mean=scalar(plastic_capacity.mean()),
        obsolete_capacity_mean=scalar(obsolete_capacity.mean()),
        reasoner_role_entropy_mean=scalar(role_entropy.mean()),
        reasoner_role_max_share_mean=scalar(role_belief.max(dim=-1).values.mean()),
        reasoner_gate_utility_mean=scalar(gate_utility.mean()),
        reasoner_gate_error_abs_mean=scalar(gate_error.abs().mean()),
        reasoner_weight_norm=scalar(reasoner_weight_norm_tensor),
        reasoner_update_norm=scalar(reasoner_update_norm_tensor * reasoner_update_scale),
        developmental_maturity=developmental_maturity,
        control_phase_gate=scalar(control_phase_gate),
        failed_write_signal=failed_write_signal,
        sculpting_fraction=scalar(sculpting.mean()),
        hardening_fraction=scalar(hardening.mean()),
        crystalline_fraction=scalar(crystalline.mean()),
        active_sculpting_fraction=scalar((sculpting * M).sum() / active_mass),
        active_hardening_fraction=scalar((hardening * M).sum() / active_mass),
        active_crystalline_fraction=scalar((crystalline * M).sum() / active_mass),
        active_hardening_latch_fraction=scalar((H_latch * M).sum() / active_mass),
        topology_mean=scalar(A.mean()),
        topology_active_fraction=scalar((A > 0.05).to(dtype=W.dtype).mean()),
        topology_grow_mean=scalar(grow.mean()),
        topology_grow_max=scalar(grow.max()),
        topology_prune_mean=scalar(prune.mean()),
        topology_prune_max=scalar(prune.max()),
        topology_delta_abs_mean=scalar(topology_delta_abs.mean()),
        usage_mean=scalar(U.mean()),
        usage_max=scalar(U.max()),
        row_pressure_mean=scalar(row_pressure.mean()),
        row_pressure_max=scalar(row_pressure.max()),
        write_gate_raw_mean=scalar(write_gate_raw.mean()),
        protect_gate_raw_mean=scalar(protect_gate_raw.mean()),
        rewire_gate_raw_mean=scalar(rewire_gate_raw.mean()),
        forget_gate_raw_mean=scalar(forget_gate_raw.mean()),
        compress_gate_raw_mean=scalar(compress_gate_raw.mean()),
        write_gate_mean=scalar(write_gate.mean()),
        protect_gate_mean=scalar(protect_gate.mean()),
        rewire_gate_mean=scalar(rewire_gate.mean()),
        forget_gate_mean=scalar(forget_gate.mean()),
        compress_gate_mean=scalar(compress_gate.mean()),
        write_edit_fraction=float(write_edit_count / W.numel()),
        protect_edit_fraction=float(protect_edit_count / W.numel()),
        rewire_edit_fraction=float(rewire_edit_count / W.numel()),
        forget_edit_fraction=float(forget_edit_count / W.numel()),
        compress_edit_fraction=float(compress_edit_count / W.numel()),
        write_edit_count=float(write_edit_count),
        protect_edit_count=float(protect_edit_count),
        rewire_edit_count=float(rewire_edit_count),
        forget_edit_count=float(forget_edit_count),
        compress_edit_count=float(compress_edit_count),
        write_score_mass=float(write_score_mass),
        protect_score_mass=float(protect_score_mass),
        rewire_score_mass=float(rewire_score_mass),
        forget_score_mass=float(forget_score_mass),
        compress_score_mass=float(compress_score_mass),
        raw_write_norm=scalar(raw_write_norm),
        direct_write_norm=scalar(direct_write_norm),
        safe_direction_norm=scalar(safe_norm),
        safe_direction_ratio=scalar(safe_norm / raw_write_norm.clamp_min(cfg.eps)),
        projection_removed_ratio=scalar(protected_projection_norm / raw_write_norm.clamp_min(cfg.eps)),
        step_scale=scalar(scale),
        safe_update_norm=scalar(torch.linalg.matrix_norm(delta)),
        total_weight_delta_norm=scalar(torch.linalg.matrix_norm(weight_delta)),
        selected_weight_delta_norm=scalar(torch.linalg.matrix_norm(weight_delta * selected_weight_mask)),
        unselected_weight_delta_norm=scalar(torch.linalg.matrix_norm(weight_delta * unselected_weight_mask)),
        unselected_weight_delta_max=scalar((weight_delta * unselected_weight_mask).abs().max()),
        selected_topology_delta_norm=scalar(torch.linalg.matrix_norm(topology_delta * selected_topology_mask)),
        unselected_topology_delta_norm=scalar(torch.linalg.matrix_norm(topology_delta * unselected_topology_mask)),
        unselected_topology_delta_max=scalar((topology_delta * unselected_topology_mask).abs().max()),
        forget_rate_mean=scalar(forget.mean()),
        forget_rate_max=scalar(forget.max()),
        inactive_mean=scalar(inactive.mean()),
        inactive_max=scalar(inactive.max()),
        weight_norm=scalar(torch.linalg.matrix_norm(W)),
    )
    for key, value in asdict(stats).items():
        if isinstance(value, float):
            require_finite_float(f"{name}_{key}", value)
    eligibility_matrix = (write_mask * write_gate * G_safe.abs()).clamp_min(0.0)
    selected_write = (write_mask > 0.0) & (eligibility_matrix > cfg.eps)
    if bool(selected_write.any()):
        selected_indices = torch.nonzero(selected_write.reshape(-1), as_tuple=False).flatten()
        selected_rows = torch.div(selected_indices, W.shape[1], rounding_mode="floor")
        selected_cols = selected_indices.remainder(W.shape[1])
        outcome_credit_context = {
            "features": feature_stack[selected_write].detach().clone(),
            "state": S[selected_write].detach().clone(),
            "write_gate": write_gate_raw[selected_write].detach().clone(),
            "write_gate_raw": write_gate_raw[selected_write].detach().clone(),
            "write_gate_eff": write_gate[selected_write].detach().clone(),
            "protect_gate": protect_gate[selected_write].detach().clone(),
            "rewire_gate": rewire_gate[selected_write].detach().clone(),
            "forget_gate": forget_gate[selected_write].detach().clone(),
            "compress_gate": compress_gate[selected_write].detach().clone(),
            "ssr_rewire_gate": ssr_rewire_gate[selected_write].detach().clone(),
            "ssr_forget_gate": ssr_forget_gate[selected_write].detach().clone(),
            "ssr_compress_gate": ssr_compress_gate[selected_write].detach().clone(),
            "ssr_gain_pred": ssr_gain_pred[selected_write].detach().clone(),
            "ssr_capacity_cost": ssr_capacity_cost[selected_write].detach().clone(),
            "ssr_forget_safe": ssr_forget_safe[selected_write].detach().clone(),
            "ssr_priority": ssr_priority[selected_write].detach().clone(),
            "write_priority_control": write_priority_control[selected_write].detach().clone(),
            "write_score": write_score[selected_write].detach().clone(),
            "protect_score": protect_score[selected_write].detach().clone(),
            "rewire_score": rewire_score[selected_write].detach().clone(),
            "forget_score": forget_score[selected_write].detach().clone(),
            "compress_score": compress_score[selected_write].detach().clone(),
            "indices": selected_indices.detach().clone(),
            "rows": selected_rows.detach().clone(),
            "cols": selected_cols.detach().clone(),
            "eligibility": eligibility_matrix[selected_write].detach().clone(),
            "state_before": S_before[selected_write].detach().clone(),
            "state_after": S[selected_write].detach().clone(),
            "weight_before": W_before[selected_write].detach().clone(),
            "weight_after": W[selected_write].detach().clone(),
            "weight_delta": weight_delta[selected_write].detach().clone(),
            "topology_before": A_before[selected_write].detach().clone(),
            "topology_after": A[selected_write].detach().clone(),
            "topology_delta": topology_delta[selected_write].detach().clone(),
            "raw_write": G_write[selected_write].detach().clone(),
            "safe_write": G_safe[selected_write].detach().clone(),
        }
        if cfg.state_space_reasoner_dim > 0:
            outcome_credit_context.update(
                {
                    "ssr_rows": selected_rows.detach().clone(),
                    "ssr_cols": selected_cols.detach().clone(),
                    "ssr_row_z": ssr_row_z[selected_rows].detach().clone(),
                    "ssr_col_z": ssr_col_z[selected_cols].detach().clone(),
                    "ssr_module_z": ssr_module_z.expand(selected_indices.shape[0], -1).detach().clone(),
                    "ssr_write_gate": ssr_write_gate.reshape(-1)[selected_indices].detach().clone(),
                    "ssr_protect_gate": ssr_protect_gate.reshape(-1)[selected_indices].detach().clone(),
                    "ssr_protect_target": gate_utility[..., PROTECT_GATE_INDEX].reshape(-1)[selected_indices].detach().clone(),
                    "ssr_reliability": ssr_reliability.reshape(-1)[selected_indices].detach().clone(),
                    "ssr_reliability_target": old_route_strength.reshape(-1)[selected_indices].detach().clone(),
                    "ssr_collision": ssr_collision.reshape(-1)[selected_indices].detach().clone(),
                    "ssr_collision_target": collision_protect_need.reshape(-1)[selected_indices].detach().clone(),
                    "ssr_value_pred": ssr_value_pred.reshape(-1)[selected_indices].detach().clone(),
                }
            )
    else:
        outcome_credit_context = None
    return stats, outcome_credit_context


class GCONativeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cfg: NativeGCOConfig, *, prefix: str) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = GCOLinear(d_model, d_model, cfg, name=f"{prefix}.q")
        self.k = GCOLinear(d_model, d_model, cfg, name=f"{prefix}.k")
        self.v = GCOLinear(d_model, d_model, cfg, name=f"{prefix}.v")
        self.o = GCOLinear(d_model, d_model, cfg, name=f"{prefix}.o")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _dim = x.shape
        q = self.q(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(float(self.head_dim))
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        return self.o(out)


class GCONativeMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, cfg: NativeGCOConfig, *, prefix: str) -> None:
        super().__init__()
        self.fc1 = GCOLinear(d_model, d_ff, cfg, name=f"{prefix}.fc1")
        self.fc2 = GCOLinear(d_ff, d_model, cfg, name=f"{prefix}.fc2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GCONativeBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, cfg: NativeGCOConfig, *, layer_index: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = GCONativeSelfAttention(d_model, n_heads, cfg, prefix=f"blocks.{layer_index}.attn")
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = GCONativeMLP(d_model, d_ff, cfg, prefix=f"blocks.{layer_index}.mlp")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GCONativeTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        cfg: NativeGCOConfig,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        self.max_seq_len = max_seq_len
        self.token_embedding = GCOEmbedding(vocab_size, d_model, cfg, name="token_embedding")
        self.position_embedding = GCOEmbedding(max_seq_len, d_model, cfg, name="position_embedding")
        self.blocks = nn.ModuleList(
            [GCONativeBlock(d_model, n_heads, d_ff, cfg, layer_index=index) for index in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False)
        self.lm_head = GCOLinear(d_model, vocab_size, cfg, name="lm_head")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [batch, seq], got {tokens.shape}.")
        batch, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}.")
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
        h = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.ln_f(h))

    def gco_modules(self) -> list[GCOLinear | GCOEmbedding]:
        modules: list[GCOLinear | GCOEmbedding] = []
        for module in self.modules():
            if isinstance(module, (GCOLinear, GCOEmbedding)):
                modules.append(module)
        return modules

    def zero_gco_grads(self) -> None:
        for module in self.gco_modules():
            module.W.grad = None

    @torch.no_grad()
    def gco_step(
        self,
        step: int,
        *,
        developmental_maturity: float,
        failed_write_signal: float,
    ) -> list[GCOModuleStats]:
        return [
            module.gco_step(
                step,
                developmental_maturity=developmental_maturity,
                failed_write_signal=failed_write_signal,
            )
            for module in self.gco_modules()
        ]

    @torch.no_grad()
    def apply_outcome_credit(self, *, utility: float, advantage: float, step: int) -> list[GCOOutcomeCreditStats]:
        modules = self.gco_modules()
        total_eligibility_mass = sum(module.outcome_credit_eligibility_sum() for module in modules)
        total_selected_count = sum(module.outcome_credit_selected_count() for module in modules)
        require_finite_float("total_outcome_eligibility_mass", total_eligibility_mass)
        require_finite_float("total_outcome_selected_count", total_selected_count)
        return [
            module.apply_outcome_credit(
                utility=utility,
                advantage=advantage,
                total_eligibility_mass=total_eligibility_mass,
                total_selected_count=total_selected_count,
                step=step,
            )
            for module in modules
        ]

    @torch.no_grad()
    def selected_route_traces(self, *, limit: int) -> list[dict[str, object]]:
        if limit < 0:
            raise ValueError(f"selected route trace limit must be non-negative, got {limit}.")
        rows: list[dict[str, object]] = []
        remaining = limit
        for module in self.gco_modules():
            if remaining <= 0:
                break
            module_rows = module.selected_route_traces(limit=remaining)
            rows.extend(module_rows)
            remaining = limit - len(rows)
        return rows


def aggregate_stats(stats: Sequence[GCOModuleStats]) -> dict[str, float]:
    if not stats:
        raise ValueError("Cannot aggregate empty GCO stats.")

    def avg(name: str) -> float:
        values = [float(getattr(item, name)) for item in stats]
        for value in values:
            require_finite_float(name, value)
        return float(sum(values) / len(values))

    return {
        "pressure_mean": avg("pressure_mean"),
        "pressure_max": max(float(item.pressure_max) for item in stats),
        "pathway_mean": avg("pathway_mean"),
        "pathway_max": max(float(item.pathway_max) for item in stats),
        "error_pressure_mean": avg("error_pressure_mean"),
        "error_pressure_max": max(float(item.error_pressure_max) for item in stats),
        "formation_pressure_mean": avg("formation_pressure_mean"),
        "formation_pressure_max": max(float(item.formation_pressure_max) for item in stats),
        "formation_effective_mean": avg("formation_effective_mean"),
        "formation_effective_max": max(float(item.formation_effective_max) for item in stats),
        "formation_row_mean": avg("formation_row_mean"),
        "formation_row_max": max(float(item.formation_row_max) for item in stats),
        "formation_col_mean": avg("formation_col_mean"),
        "formation_col_max": max(float(item.formation_col_max) for item in stats),
        "formation_module_mean": avg("formation_module_mean"),
        "formation_module_max": max(float(item.formation_module_max) for item in stats),
        "hardening_latch_mean": avg("hardening_latch_mean"),
        "hardening_latch_max": max(float(item.hardening_latch_max) for item in stats),
        "ssr_row_state_norm_mean": avg("ssr_row_state_norm_mean"),
        "ssr_row_state_norm_max": max(float(item.ssr_row_state_norm_max) for item in stats),
        "ssr_col_state_norm_mean": avg("ssr_col_state_norm_mean"),
        "ssr_col_state_norm_max": max(float(item.ssr_col_state_norm_max) for item in stats),
        "ssr_module_state_norm_mean": avg("ssr_module_state_norm_mean"),
        "ssr_module_state_norm_max": max(float(item.ssr_module_state_norm_max) for item in stats),
        "ssr_write_gate_mean": avg("ssr_write_gate_mean"),
        "ssr_write_gate_max": max(float(item.ssr_write_gate_max) for item in stats),
        "ssr_protect_gate_mean": avg("ssr_protect_gate_mean"),
        "ssr_protect_gate_max": max(float(item.ssr_protect_gate_max) for item in stats),
        "ssr_reliability_mean": avg("ssr_reliability_mean"),
        "ssr_reliability_max": max(float(item.ssr_reliability_max) for item in stats),
        "ssr_collision_mean": avg("ssr_collision_mean"),
        "ssr_collision_max": max(float(item.ssr_collision_max) for item in stats),
        "ssr_protect_eff_mean": avg("ssr_protect_eff_mean"),
        "ssr_protect_eff_max": max(float(item.ssr_protect_eff_max) for item in stats),
        "ssr_rewire_gate_mean": avg("ssr_rewire_gate_mean"),
        "ssr_rewire_gate_max": max(float(item.ssr_rewire_gate_max) for item in stats),
        "ssr_forget_gate_mean": avg("ssr_forget_gate_mean"),
        "ssr_forget_gate_max": max(float(item.ssr_forget_gate_max) for item in stats),
        "ssr_compress_gate_mean": avg("ssr_compress_gate_mean"),
        "ssr_compress_gate_max": max(float(item.ssr_compress_gate_max) for item in stats),
        "ssr_gain_pred_mean": avg("ssr_gain_pred_mean"),
        "ssr_gain_pred_max": max(float(item.ssr_gain_pred_max) for item in stats),
        "ssr_capacity_cost_mean": avg("ssr_capacity_cost_mean"),
        "ssr_capacity_cost_max": max(float(item.ssr_capacity_cost_max) for item in stats),
        "ssr_forget_safe_mean": avg("ssr_forget_safe_mean"),
        "ssr_forget_safe_max": max(float(item.ssr_forget_safe_max) for item in stats),
        "ssr_priority_mean": avg("ssr_priority_mean"),
        "ssr_priority_max": max(float(item.ssr_priority_max) for item in stats),
        "ssr_value_pred_mean": avg("ssr_value_pred_mean"),
        "ssr_value_pred_max": max(float(item.ssr_value_pred_max) for item in stats),
        "ssr_credit_mean": avg("ssr_credit_mean"),
        "ssr_credit_max": max(float(item.ssr_credit_max) for item in stats),
        "ssr_td_error_abs_mean": avg("ssr_td_error_abs_mean"),
        "ssr_td_error_abs_max": max(float(item.ssr_td_error_abs_max) for item in stats),
        "ssr_update_norm": avg("ssr_update_norm"),
        "write_pressure_mean": avg("write_pressure_mean"),
        "write_pressure_max": max(float(item.write_pressure_max) for item in stats),
        "protect_need_mean": avg("protect_need_mean"),
        "protect_need_max": max(float(item.protect_need_max) for item in stats),
        "protection_pressure_mean": avg("protection_pressure_mean"),
        "protection_pressure_max": max(float(item.protection_pressure_max) for item in stats),
        "structural_protection_mean": avg("structural_protection_mean"),
        "structural_protection_max": max(float(item.structural_protection_max) for item in stats),
        "structural_input_protection_mean": avg("structural_input_protection_mean"),
        "structural_input_protection_max": max(float(item.structural_input_protection_max) for item in stats),
        "direct_write_protect_effective": avg("direct_write_protect_effective"),
        "decay_pressure_mean": avg("decay_pressure_mean"),
        "decay_pressure_max": max(float(item.decay_pressure_max) for item in stats),
        "direct_write_basis_mean": avg("direct_write_basis_mean"),
        "direct_write_basis_max": max(float(item.direct_write_basis_max) for item in stats),
        "route_age_mean": avg("route_age_mean"),
        "route_age_max": max(float(item.route_age_max) for item in stats),
        "route_recency_mean": avg("route_recency_mean"),
        "route_recency_max": max(float(item.route_recency_max) for item in stats),
        "recurrent_state_mean": avg("recurrent_state_mean"),
        "recurrent_state_abs_mean": avg("recurrent_state_abs_mean"),
        "recurrent_state_delta_mean": avg("recurrent_state_delta_mean"),
        "protected_capacity_mean": avg("protected_capacity_mean"),
        "free_capacity_mean": avg("free_capacity_mean"),
        "plastic_capacity_mean": avg("plastic_capacity_mean"),
        "obsolete_capacity_mean": avg("obsolete_capacity_mean"),
        "reasoner_role_entropy_mean": avg("reasoner_role_entropy_mean"),
        "reasoner_role_max_share_mean": avg("reasoner_role_max_share_mean"),
        "reasoner_gate_utility_mean": avg("reasoner_gate_utility_mean"),
        "reasoner_gate_error_abs_mean": avg("reasoner_gate_error_abs_mean"),
        "reasoner_weight_norm": avg("reasoner_weight_norm"),
        "reasoner_update_norm": avg("reasoner_update_norm"),
        "developmental_maturity": avg("developmental_maturity"),
        "control_phase_gate": avg("control_phase_gate"),
        "failed_write_signal": avg("failed_write_signal"),
        "sculpting_fraction": avg("sculpting_fraction"),
        "hardening_fraction": avg("hardening_fraction"),
        "crystalline_fraction": avg("crystalline_fraction"),
        "active_sculpting_fraction": avg("active_sculpting_fraction"),
        "active_hardening_fraction": avg("active_hardening_fraction"),
        "active_crystalline_fraction": avg("active_crystalline_fraction"),
        "active_hardening_latch_fraction": avg("active_hardening_latch_fraction"),
        "topology_mean": avg("topology_mean"),
        "topology_active_fraction": avg("topology_active_fraction"),
        "topology_grow_mean": avg("topology_grow_mean"),
        "topology_grow_max": max(float(item.topology_grow_max) for item in stats),
        "topology_prune_mean": avg("topology_prune_mean"),
        "topology_prune_max": max(float(item.topology_prune_max) for item in stats),
        "topology_delta_abs_mean": avg("topology_delta_abs_mean"),
        "usage_mean": avg("usage_mean"),
        "usage_max": max(float(item.usage_max) for item in stats),
        "row_pressure_mean": avg("row_pressure_mean"),
        "row_pressure_max": max(float(item.row_pressure_max) for item in stats),
        "write_gate_raw_mean": avg("write_gate_raw_mean"),
        "protect_gate_raw_mean": avg("protect_gate_raw_mean"),
        "rewire_gate_raw_mean": avg("rewire_gate_raw_mean"),
        "forget_gate_raw_mean": avg("forget_gate_raw_mean"),
        "compress_gate_raw_mean": avg("compress_gate_raw_mean"),
        "write_gate_mean": avg("write_gate_mean"),
        "protect_gate_mean": avg("protect_gate_mean"),
        "rewire_gate_mean": avg("rewire_gate_mean"),
        "forget_gate_mean": avg("forget_gate_mean"),
        "compress_gate_mean": avg("compress_gate_mean"),
        "write_edit_fraction": avg("write_edit_fraction"),
        "protect_edit_fraction": avg("protect_edit_fraction"),
        "rewire_edit_fraction": avg("rewire_edit_fraction"),
        "forget_edit_fraction": avg("forget_edit_fraction"),
        "compress_edit_fraction": avg("compress_edit_fraction"),
        "write_edit_count": avg("write_edit_count"),
        "protect_edit_count": avg("protect_edit_count"),
        "rewire_edit_count": avg("rewire_edit_count"),
        "forget_edit_count": avg("forget_edit_count"),
        "compress_edit_count": avg("compress_edit_count"),
        "write_score_mass": avg("write_score_mass"),
        "protect_score_mass": avg("protect_score_mass"),
        "rewire_score_mass": avg("rewire_score_mass"),
        "forget_score_mass": avg("forget_score_mass"),
        "compress_score_mass": avg("compress_score_mass"),
        "raw_write_norm": avg("raw_write_norm"),
        "direct_write_norm": avg("direct_write_norm"),
        "safe_direction_norm": avg("safe_direction_norm"),
        "safe_direction_ratio": avg("safe_direction_ratio"),
        "projection_removed_ratio": avg("projection_removed_ratio"),
        "step_scale": avg("step_scale"),
        "safe_update_norm": avg("safe_update_norm"),
        "total_weight_delta_norm": avg("total_weight_delta_norm"),
        "selected_weight_delta_norm": avg("selected_weight_delta_norm"),
        "unselected_weight_delta_norm": avg("unselected_weight_delta_norm"),
        "unselected_weight_delta_max": max(float(item.unselected_weight_delta_max) for item in stats),
        "selected_topology_delta_norm": avg("selected_topology_delta_norm"),
        "unselected_topology_delta_norm": avg("unselected_topology_delta_norm"),
        "unselected_topology_delta_max": max(float(item.unselected_topology_delta_max) for item in stats),
        "forget_rate_mean": avg("forget_rate_mean"),
        "forget_rate_max": max(float(item.forget_rate_max) for item in stats),
        "inactive_mean": avg("inactive_mean"),
        "inactive_max": max(float(item.inactive_max) for item in stats),
        "weight_norm": avg("weight_norm"),
    }


def aggregate_outcome_credit_stats(stats: Sequence[GCOOutcomeCreditStats]) -> dict[str, float]:
    if not stats:
        raise ValueError("Cannot aggregate empty outcome credit stats.")
    selected_total = sum(float(item.selected_count) for item in stats)
    require_finite_float("outcome_credit_selected_count", selected_total)

    def weighted_avg(name: str) -> float:
        if selected_total <= 0.0:
            return 0.0
        value = sum(float(getattr(item, name)) * float(item.selected_count) for item in stats) / selected_total
        require_finite_float(name, value)
        return float(value)

    update_norm = sum(float(item.update_norm) for item in stats)
    require_finite_float("outcome_credit_update_norm", update_norm)
    return {
        "outcome_credit_selected_count": selected_total,
        "outcome_credit_eligibility_mass": sum(float(item.eligibility_mass) for item in stats),
        "outcome_credit_eligibility_max_share": max(float(item.eligibility_max_share) for item in stats),
        "outcome_credit_advantage": weighted_avg("advantage"),
        "outcome_credit_route_advantage_mean": weighted_avg("route_advantage_mean"),
        "outcome_credit_route_advantage_abs_mean": weighted_avg("route_advantage_abs_mean"),
        "outcome_credit_route_advantage_max": max(float(item.route_advantage_max) for item in stats),
        "outcome_credit_route_formation_utility_mean": weighted_avg("route_formation_utility_mean"),
        "outcome_credit_route_formation_utility_max": max(float(item.route_formation_utility_max) for item in stats),
        "outcome_credit_utility_target": weighted_avg("utility_target"),
        "outcome_credit_utility_target_min": min(float(item.utility_target_min) for item in stats),
        "outcome_credit_utility_target_max": max(float(item.utility_target_max) for item in stats),
        "outcome_credit_write_gate_mean": weighted_avg("write_gate_mean"),
        "outcome_credit_protect_target_mean": weighted_avg("protect_target_mean"),
        "outcome_credit_protect_target_min": min(float(item.protect_target_min) for item in stats),
        "outcome_credit_protect_target_max": max(float(item.protect_target_max) for item in stats),
        "outcome_credit_protect_gate_mean": weighted_avg("protect_gate_mean"),
        "outcome_credit_protect_error_abs_mean": weighted_avg("protect_error_abs_mean"),
        "outcome_credit_reliability_target_mean": weighted_avg("reliability_target_mean"),
        "outcome_credit_reliability_target_min": min(float(item.reliability_target_min) for item in stats),
        "outcome_credit_reliability_target_max": max(float(item.reliability_target_max) for item in stats),
        "outcome_credit_reliability_pred_mean": weighted_avg("reliability_pred_mean"),
        "outcome_credit_reliability_error_abs_mean": weighted_avg("reliability_error_abs_mean"),
        "outcome_credit_collision_target_mean": weighted_avg("collision_target_mean"),
        "outcome_credit_collision_target_min": min(float(item.collision_target_min) for item in stats),
        "outcome_credit_collision_target_max": max(float(item.collision_target_max) for item in stats),
        "outcome_credit_collision_pred_mean": weighted_avg("collision_pred_mean"),
        "outcome_credit_collision_error_abs_mean": weighted_avg("collision_error_abs_mean"),
        "outcome_credit_gate_error_abs_mean": weighted_avg("gate_error_abs_mean"),
        "outcome_credit_update_norm": update_norm,
    }


def hottest_modules(stats: Sequence[GCOModuleStats], *, limit: int) -> list[dict[str, object]]:
    if limit <= 0:
        return []
    ranked = sorted(stats, key=lambda item: item.pressure_max, reverse=True)
    return [asdict(item) for item in ranked[:limit]]


def format_hot_modules(modules: Sequence[dict[str, object]]) -> str:
    parts: list[str] = []
    for item in modules:
        parts.append(
            "{name}:P={pressure_max:.2f},aH={active_hardening_fraction:.2f},"
            "aC={active_crystalline_fraction:.2f},F={formation_pressure_mean:.2f},"
            "Pr={protection_pressure_mean:.2f},D={decay_pressure_mean:.2f},S={recurrent_state_mean:.2f},"
            "wg={write_gate_mean:.2f},fg={forget_gate_mean:.2f},"
            "K={write_edit_count:.0f}/{rewire_edit_count:.0f}/{forget_edit_count:.0f},"
            "rE={reasoner_gate_error_abs_mean:.2f},AΔ={topology_delta_abs_mean:.1e}".format(**item)
        )
    return " | ".join(parts)


def windows_from_text(text: str, tokenizer: Tokenizer, *, max_seq_len: int, stride: int, max_windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if max_seq_len < 3:
        raise ValueError("--max-seq-len must be at least 3.")
    if stride <= 0:
        raise ValueError("--window-stride must be positive.")
    if max_windows <= 0:
        raise ValueError("--max-windows-per-chunk must be positive.")
    token_ids = tokenizer.encode(text).ids
    if len(token_ids) < max_seq_len:
        raise ValueError(f"Text produced {len(token_ids)} tokens, fewer than max_seq_len={max_seq_len}.")
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, len(token_ids) - max_seq_len, stride):
        window = token_ids[start : start + max_seq_len]
        inputs.append(window[:-1])
        targets.append(window[1:])
        if len(inputs) >= max_windows:
            break
    if not inputs:
        raise RuntimeError("No token windows were built.")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def split_train_canary_windows(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    canary_windows: int,
    canary_selection: str,
) -> dict[str, torch.Tensor | None]:
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError(f"Window tensors must be matrices, got inputs={inputs.shape}, targets={targets.shape}.")
    if inputs.shape != targets.shape:
        raise ValueError(f"Input/target window shape mismatch: inputs={inputs.shape}, targets={targets.shape}.")
    if canary_windows < 0:
        raise ValueError("--canary-windows-per-chunk must be non-negative.")
    if canary_windows == 0:
        return {"inputs": inputs, "targets": targets, "canary_inputs": None, "canary_targets": None}
    if inputs.shape[0] <= canary_windows:
        raise ValueError(
            f"Cannot reserve {canary_windows} canary windows from only {inputs.shape[0]} windows; "
            "increase --max-windows-per-chunk or reduce --canary-windows-per-chunk."
        )
    if canary_selection == "tail":
        canary_indices = torch.arange(inputs.shape[0] - canary_windows, inputs.shape[0], dtype=torch.long)
    elif canary_selection == "interleaved":
        canary_indices = torch.div(
            torch.arange(canary_windows, dtype=torch.long) * inputs.shape[0],
            canary_windows,
            rounding_mode="floor",
        )
    else:
        raise ValueError(f"Unknown canary selection mode: {canary_selection!r}.")
    unique_count = int(torch.unique(canary_indices).numel())
    if unique_count != canary_windows:
        raise RuntimeError(
            f"Canary selection produced duplicate indices: requested={canary_windows}, unique={unique_count}."
        )
    train_mask = torch.ones(inputs.shape[0], dtype=torch.bool)
    train_mask[canary_indices] = False
    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten()
    if train_indices.numel() <= 0:
        raise RuntimeError("Canary split removed every training window.")
    return {
        "inputs": inputs[train_indices],
        "targets": targets[train_indices],
        "canary_inputs": inputs[canary_indices],
        "canary_targets": targets[canary_indices],
    }


def build_chunk_dataset(
    chunks: Sequence[dict[str, object]],
    tokenizer: Tokenizer,
    *,
    max_seq_len: int,
    stride: int,
    max_windows: int,
    canary_windows: int,
    canary_selection: str,
) -> list[dict[str, object]]:
    dataset: list[dict[str, object]] = []
    for chunk in chunks:
        if "chunk_id" not in chunk or "text" not in chunk:
            raise ValueError("Every chunk must contain chunk_id and text.")
        inputs, targets = windows_from_text(
            str(chunk["text"]),
            tokenizer,
            max_seq_len=max_seq_len,
            stride=stride,
            max_windows=max_windows + canary_windows,
        )
        split = split_train_canary_windows(
            inputs,
            targets,
            canary_windows=canary_windows,
            canary_selection=canary_selection,
        )
        dataset.append({"chunk_id": str(chunk["chunk_id"]), **split})
    if not dataset:
        raise RuntimeError("No chunks selected for native GCO training.")
    return dataset


@torch.no_grad()
def token_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(f"Logits/targets shape mismatch: logits={logits.shape}, targets={targets.shape}.")
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).to(dtype=torch.float32)
    return scalar(correct.mean())


@torch.no_grad()
def evaluate_metrics(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0.0
    total_target_logprob = 0.0
    total_target_margin = 0.0
    min_target_margin: float | None = None
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        batch_targets = targets[start : start + batch_size].to(device)
        logits = model(batch_inputs)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = batch_targets.reshape(-1)
        if flat_logits.shape[-1] < 2:
            raise ValueError("Canary/eval margin requires vocab size of at least 2.")
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
        total_loss += scalar(loss)
        total_count += int(batch_targets.numel())
        total_correct += scalar((logits.argmax(dim=-1) == batch_targets).to(dtype=torch.float32).sum())
        target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        top_values, top_indices = torch.topk(flat_logits, k=2, dim=-1)
        competitor_logits = torch.where(top_indices[:, 0] == flat_targets, top_values[:, 1], top_values[:, 0])
        target_margins = target_logits - competitor_logits
        target_logprobs = F.log_softmax(flat_logits, dim=-1).gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        total_target_logprob += scalar(target_logprobs.sum())
        total_target_margin += scalar(target_margins.sum())
        batch_min_margin = scalar(target_margins.min())
        min_target_margin = batch_min_margin if min_target_margin is None else min(min_target_margin, batch_min_margin)
    if total_count <= 0:
        raise RuntimeError("Evaluation saw zero target tokens.")
    if min_target_margin is None:
        raise RuntimeError("Evaluation did not compute any target margins.")
    metrics = {
        "loss": total_loss / float(total_count),
        "token_accuracy": total_correct / float(total_count),
        "target_logprob_mean": total_target_logprob / float(total_count),
        "target_margin_mean": total_target_margin / float(total_count),
        "target_margin_min": min_target_margin,
    }
    for key, value in metrics.items():
        require_finite_float(f"eval_{key}", value)
    return metrics


def canary_drift_row(
    *,
    chunk_id: str,
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, object]:
    loss_delta = metrics["loss"] - baseline["loss"]
    margin_delta = metrics["target_margin_mean"] - baseline["target_margin_mean"]
    logprob_delta = metrics["target_logprob_mean"] - baseline["target_logprob_mean"]
    health = math.exp(-max(0.0, loss_delta) - max(0.0, -margin_delta))
    row: dict[str, object] = {
        "chunk_id": chunk_id,
        "loss": metrics["loss"],
        "token_accuracy": metrics["token_accuracy"],
        "target_logprob_mean": metrics["target_logprob_mean"],
        "target_margin_mean": metrics["target_margin_mean"],
        "target_margin_min": metrics["target_margin_min"],
        "baseline_loss": baseline["loss"],
        "baseline_token_accuracy": baseline["token_accuracy"],
        "baseline_target_logprob_mean": baseline["target_logprob_mean"],
        "baseline_target_margin_mean": baseline["target_margin_mean"],
        "loss_delta_from_canary_baseline": loss_delta,
        "target_logprob_delta_from_canary_baseline": logprob_delta,
        "target_margin_delta_from_canary_baseline": margin_delta,
        "canary_health": health,
    }
    for key, value in row.items():
        if isinstance(value, float):
            require_finite_float(f"canary_{chunk_id}_{key}", value)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-native-scratch-transformer-seed0.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--window-stride", type=int, default=48)
    parser.add_argument("--max-windows-per-chunk", type=int, default=64)
    parser.add_argument("--canary-windows-per-chunk", type=int, default=0)
    parser.add_argument("--canary-selection", choices=["tail", "interleaved"], default="tail")
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=5)
    parser.add_argument("--epochs-per-chunk", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--write-mode", choices=["gradient", "direct"], default="gradient")
    parser.add_argument("--reasoner-policy", choices=["fixed_geometric", "learned"], default="fixed_geometric")
    parser.add_argument("--direct-write-error-scale", choices=["mean", "sample_count"], default="mean")
    parser.add_argument("--min-active-topology", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--direct-write-ridge", type=float, default=1e-2)
    parser.add_argument("--direct-write-protect", type=float, default=1e-1)
    parser.add_argument("--beta-pressure", type=float, default=0.98)
    parser.add_argument("--beta-formation", type=float, default=0.98)
    parser.add_argument("--beta-protection", type=float, default=0.98)
    parser.add_argument("--beta-decay", type=float, default=0.98)
    parser.add_argument("--beta-usage", type=float, default=0.98)
    parser.add_argument("--beta-capacity", type=float, default=0.98)
    parser.add_argument("--beta-recurrent", type=float, default=0.98)
    parser.add_argument("--reasoner-lr", type=float, default=0.0)
    parser.add_argument("--reasoner-weight-decay", type=float, default=1e-4)
    parser.add_argument("--reasoner-update-clip", type=float, default=1.0)
    parser.add_argument("--internal-gate-lr-scale", type=float, default=1.0)
    parser.add_argument("--internal-write-gate-lr-scale", type=float, default=1.0)
    parser.add_argument("--internal-protect-gate-lr-scale", type=float, default=1.0)
    parser.add_argument("--internal-rewire-gate-lr-scale", type=float, default=1.0)
    parser.add_argument("--internal-forget-gate-lr-scale", type=float, default=1.0)
    parser.add_argument("--internal-compress-gate-lr-scale", type=float, default=1.0)
    parser.add_argument("--state-space-reasoner-dim", type=int, default=0)
    parser.add_argument("--state-space-reasoner-lr", type=float, default=0.0)
    parser.add_argument("--state-space-value-lr", type=float, default=0.0)
    parser.add_argument("--state-space-weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-space-update-clip", type=float, default=1.0)
    parser.add_argument("--state-space-init-std", type=float, default=0.02)
    parser.add_argument("--state-space-credit-beta", type=float, default=0.95)
    parser.add_argument("--state-space-write-blend", type=float, default=0.0)
    parser.add_argument("--state-space-protect-blend", type=float, default=0.0)
    parser.add_argument("--state-space-rewire-blend", type=float, default=0.0)
    parser.add_argument("--state-space-forget-blend", type=float, default=0.0)
    parser.add_argument("--state-space-compress-blend", type=float, default=0.0)
    parser.add_argument("--state-space-priority-blend", type=float, default=0.0)
    parser.add_argument("--outcome-credit-mode", choices=["none", "same_batch"], default="none")
    parser.add_argument("--outcome-credit-lr", type=float, default=0.0)
    parser.add_argument("--outcome-formation-lr", type=float, default=0.0)
    parser.add_argument("--outcome-baseline-beta", type=float, default=0.95)
    parser.add_argument("--outcome-failure-scale", type=float, default=1e-3)
    parser.add_argument("--route-credit-scale", type=float, default=1e-5)
    parser.add_argument("--route-credit-logit-clip", type=float, default=2.0)
    parser.add_argument("--route-credit-warmup-steps", type=int, default=100)
    parser.add_argument("--route-formation-scale", type=float, default=1e-5)
    parser.add_argument("--route-formation-logit-clip", type=float, default=2.0)
    parser.add_argument("--formation-weight-mix", type=float, default=1.0)
    parser.add_argument("--formation-row-mix", type=float, default=0.0)
    parser.add_argument("--formation-col-mix", type=float, default=0.0)
    parser.add_argument("--formation-module-mix", type=float, default=0.0)
    parser.add_argument("--formation-multiscale-pooling", choices=["mean", "max"], default="max")
    parser.add_argument("--positive-utility-write-floor", type=float, default=0.5)
    parser.add_argument("--protect-old-route-floor", type=float, default=0.25)
    parser.add_argument("--protect-collision-strength", type=float, default=1.0)
    parser.add_argument("--hardening-exit-threshold", type=float, default=0.2)
    parser.add_argument("--hardening-protection-strength", type=float, default=0.0)
    parser.add_argument("--structural-protect-strength", type=float, default=0.0)
    parser.add_argument("--outcome-edit-cost", type=float, default=0.0)
    parser.add_argument("--outcome-capacity-cost", type=float, default=0.0)
    parser.add_argument("--outcome-rewire-cost", type=float, default=0.0)
    parser.add_argument("--outcome-forget-cost", type=float, default=0.0)
    parser.add_argument("--maturity-source", choices=["accuracy", "loss_drop", "max_accuracy_loss_drop"], default="max_accuracy_loss_drop")
    parser.add_argument("--failed-write-beta", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=12.0)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--recency-tau", type=float, default=200.0)
    parser.add_argument("--grow-lr", type=float, default=1e-2)
    parser.add_argument("--prune-lr", type=float, default=5e-4)
    parser.add_argument("--forget-lr", type=float, default=1e-5)
    parser.add_argument("--max-step-norm", type=float, default=1.0)
    parser.add_argument("--init-topology", type=float, default=0.5)
    parser.add_argument("--hardening-threshold", type=float, default=0.35)
    parser.add_argument("--crystalline-threshold", type=float, default=0.75)
    parser.add_argument("--pathway-percentile", type=float, default=0.99)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--hot-modules", type=int, default=3)
    parser.add_argument("--route-trace-limit", type=int, default=0)
    parser.add_argument("--eval-after-chunk", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_float("lr", args.lr)
    positive_float("direct_write_ridge", args.direct_write_ridge)
    nonnegative_float("direct_write_protect", args.direct_write_protect)
    bounded_float("beta_pressure", args.beta_pressure, 0.0, 1.0)
    bounded_float("beta_formation", args.beta_formation, 0.0, 1.0)
    bounded_float("beta_protection", args.beta_protection, 0.0, 1.0)
    bounded_float("beta_decay", args.beta_decay, 0.0, 1.0)
    bounded_float("beta_usage", args.beta_usage, 0.0, 1.0)
    bounded_float("beta_capacity", args.beta_capacity, 0.0, 1.0)
    bounded_float("beta_recurrent", args.beta_recurrent, 0.0, 1.0)
    nonnegative_float("reasoner_lr", args.reasoner_lr)
    nonnegative_float("reasoner_weight_decay", args.reasoner_weight_decay)
    positive_float("reasoner_update_clip", args.reasoner_update_clip)
    nonnegative_float("internal_gate_lr_scale", args.internal_gate_lr_scale)
    nonnegative_float("internal_write_gate_lr_scale", args.internal_write_gate_lr_scale)
    nonnegative_float("internal_protect_gate_lr_scale", args.internal_protect_gate_lr_scale)
    nonnegative_float("internal_rewire_gate_lr_scale", args.internal_rewire_gate_lr_scale)
    nonnegative_float("internal_forget_gate_lr_scale", args.internal_forget_gate_lr_scale)
    nonnegative_float("internal_compress_gate_lr_scale", args.internal_compress_gate_lr_scale)
    if args.state_space_reasoner_dim < 0:
        raise ValueError("--state-space-reasoner-dim must be non-negative.")
    nonnegative_float("state_space_reasoner_lr", args.state_space_reasoner_lr)
    nonnegative_float("state_space_value_lr", args.state_space_value_lr)
    nonnegative_float("state_space_weight_decay", args.state_space_weight_decay)
    positive_float("state_space_update_clip", args.state_space_update_clip)
    nonnegative_float("state_space_init_std", args.state_space_init_std)
    bounded_float("state_space_credit_beta", args.state_space_credit_beta, 0.0, 1.0)
    bounded_float("state_space_write_blend", args.state_space_write_blend, 0.0, 1.0)
    bounded_float("state_space_protect_blend", args.state_space_protect_blend, 0.0, 1.0)
    bounded_float("state_space_rewire_blend", args.state_space_rewire_blend, 0.0, 1.0)
    bounded_float("state_space_forget_blend", args.state_space_forget_blend, 0.0, 1.0)
    bounded_float("state_space_compress_blend", args.state_space_compress_blend, 0.0, 1.0)
    bounded_float("state_space_priority_blend", args.state_space_priority_blend, 0.0, 1.0)
    if args.state_space_reasoner_dim == 0 and (
        args.state_space_reasoner_lr > 0.0
        or args.state_space_value_lr > 0.0
        or args.state_space_write_blend > 0.0
        or args.state_space_protect_blend > 0.0
        or args.state_space_rewire_blend > 0.0
        or args.state_space_forget_blend > 0.0
        or args.state_space_compress_blend > 0.0
        or args.state_space_priority_blend > 0.0
    ):
        raise ValueError(
            "--state-space-reasoner-dim must be positive when SSR learning or SSR gate blending is enabled."
        )
    if args.state_space_reasoner_lr * args.state_space_weight_decay >= 1.0:
        raise ValueError("--state-space-reasoner-lr * --state-space-weight-decay must be less than 1.")
    if args.state_space_value_lr * args.state_space_weight_decay >= 1.0:
        raise ValueError("--state-space-value-lr * --state-space-weight-decay must be less than 1.")
    if args.reasoner_lr * args.reasoner_weight_decay >= 1.0:
        raise ValueError("--reasoner-lr * --reasoner-weight-decay must be less than 1.")
    if args.reasoner_policy == "fixed_geometric":
        if args.reasoner_lr > 0.0:
            raise ValueError("--reasoner-policy fixed_geometric requires --reasoner-lr 0.")
        if args.state_space_reasoner_dim != 0:
            raise ValueError("--reasoner-policy fixed_geometric requires --state-space-reasoner-dim 0.")
        if args.state_space_reasoner_lr > 0.0 or args.state_space_value_lr > 0.0:
            raise ValueError(
                "--reasoner-policy fixed_geometric disables SSR learning; "
                "--state-space-reasoner-lr and --state-space-value-lr must be 0."
            )
        if (
            args.state_space_write_blend > 0.0
            or args.state_space_protect_blend > 0.0
            or args.state_space_rewire_blend > 0.0
            or args.state_space_forget_blend > 0.0
            or args.state_space_compress_blend > 0.0
            or args.state_space_priority_blend > 0.0
        ):
            raise ValueError("--reasoner-policy fixed_geometric disables SSR blends; all --state-space-*-blend values must be 0.")
        if args.outcome_credit_lr > 0.0 or args.outcome_formation_lr > 0.0:
            raise ValueError(
                "--reasoner-policy fixed_geometric disables online outcome learning; "
                "--outcome-credit-lr and --outcome-formation-lr must be 0."
            )
    nonnegative_float("outcome_credit_lr", args.outcome_credit_lr)
    nonnegative_float("outcome_formation_lr", args.outcome_formation_lr)
    bounded_float("outcome_baseline_beta", args.outcome_baseline_beta, 0.0, 1.0)
    positive_float("outcome_failure_scale", args.outcome_failure_scale)
    positive_float("route_credit_scale", args.route_credit_scale)
    positive_float("route_credit_logit_clip", args.route_credit_logit_clip)
    if args.route_credit_warmup_steps <= 0:
        raise ValueError("--route-credit-warmup-steps must be positive.")
    positive_float("route_formation_scale", args.route_formation_scale)
    positive_float("route_formation_logit_clip", args.route_formation_logit_clip)
    nonnegative_float("formation_weight_mix", args.formation_weight_mix)
    nonnegative_float("formation_row_mix", args.formation_row_mix)
    nonnegative_float("formation_col_mix", args.formation_col_mix)
    nonnegative_float("formation_module_mix", args.formation_module_mix)
    positive_float(
        "formation_mix_total",
        args.formation_weight_mix + args.formation_row_mix + args.formation_col_mix + args.formation_module_mix,
    )
    bounded_float("positive_utility_write_floor", args.positive_utility_write_floor, 0.0, 1.0)
    bounded_float("protect_old_route_floor", args.protect_old_route_floor, 0.0, 1.0)
    nonnegative_float("protect_collision_strength", args.protect_collision_strength)
    bounded_float("hardening_exit_threshold", args.hardening_exit_threshold, 0.0, 1.0)
    nonnegative_float("hardening_protection_strength", args.hardening_protection_strength)
    nonnegative_float("structural_protect_strength", args.structural_protect_strength)
    nonnegative_float("outcome_edit_cost", args.outcome_edit_cost)
    nonnegative_float("outcome_capacity_cost", args.outcome_capacity_cost)
    nonnegative_float("outcome_rewire_cost", args.outcome_rewire_cost)
    nonnegative_float("outcome_forget_cost", args.outcome_forget_cost)
    if args.outcome_credit_mode == "none" and args.outcome_credit_lr != 0.0:
        raise ValueError("--outcome-credit-lr must be 0 when --outcome-credit-mode is none.")
    if args.outcome_credit_lr * args.reasoner_weight_decay >= 1.0:
        raise ValueError("--outcome-credit-lr * --reasoner-weight-decay must be less than 1.")
    positive_float("eps", args.eps)
    bounded_float("min_active_topology", args.min_active_topology, args.eps, 1.0)
    bounded_float("failed_write_beta", args.failed_write_beta, 0.0, 1.0)
    positive_float("gamma", args.gamma)
    bounded_float("mu", args.mu, 0.0, 1.0)
    if args.warmup_steps <= 0:
        raise ValueError("--warmup-steps must be positive.")
    positive_float("recency_tau", args.recency_tau)
    nonnegative_float("grow_lr", args.grow_lr)
    nonnegative_float("prune_lr", args.prune_lr)
    nonnegative_float("forget_lr", args.forget_lr)
    positive_float("max_step_norm", args.max_step_norm)
    bounded_float("init_topology", args.init_topology, 0.0, 1.0)
    bounded_float("hardening_threshold", args.hardening_threshold, 0.0, 1.0)
    bounded_float("crystalline_threshold", args.crystalline_threshold, 0.0, 1.0)
    if args.hardening_exit_threshold >= args.hardening_threshold:
        raise ValueError("--hardening-exit-threshold must be lower than --hardening-threshold.")
    if args.hardening_threshold >= args.crystalline_threshold:
        raise ValueError("--hardening-threshold must be lower than --crystalline-threshold.")
    bounded_float("pathway_percentile", args.pathway_percentile, 0.0, 1.0)
    if args.epochs_per_chunk <= 0:
        raise ValueError("--epochs-per-chunk must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.canary_windows_per_chunk < 0:
        raise ValueError("--canary-windows-per-chunk must be non-negative.")
    if args.max_windows_per_chunk <= 0:
        raise ValueError("--max-windows-per-chunk must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.hot_modules < 0:
        raise ValueError("--hot-modules must be non-negative.")
    if args.route_trace_limit < 0:
        raise ValueError("--route-trace-limit must be non-negative.")


def select_training_chunks(chunks: Sequence[dict[str, object]], *, start: int, count: int) -> list[dict[str, object]]:
    if start < 0:
        raise ValueError("--chunk-start must be non-negative.")
    if count <= 0:
        raise ValueError("--chunk-count must be positive.")
    end = start + count
    if end > len(chunks):
        raise ValueError(f"Requested chunks [{start}, {end}) but only {len(chunks)} chunks exist.")
    return list(chunks[start:end])


def developmental_maturity_from_metrics(
    *,
    source: str,
    first_loss: float,
    current_loss: float,
    token_accuracy: float,
    eps: float,
) -> float:
    require_finite_float("first_loss", first_loss)
    require_finite_float("current_loss", current_loss)
    require_finite_float("token_accuracy", token_accuracy)
    positive_float("eps", eps)
    bounded_float("token_accuracy", token_accuracy, 0.0, 1.0)
    loss_drop = max(0.0, first_loss - current_loss) / max(abs(first_loss), eps)
    bounded_float("loss_drop_maturity", min(1.0, loss_drop), 0.0, 1.0)
    if source == "accuracy":
        maturity = token_accuracy
    elif source == "loss_drop":
        maturity = loss_drop
    elif source == "max_accuracy_loss_drop":
        maturity = max(token_accuracy, loss_drop)
    else:
        raise ValueError(f"Unknown maturity source: {source!r}.")
    maturity = min(1.0, max(0.0, maturity))
    require_finite_float("developmental_maturity", maturity)
    return maturity


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    chunks = select_training_chunks(load_chunks(args.chunks_path), start=args.chunk_start, count=args.chunk_count)
    dataset = build_chunk_dataset(
        chunks,
        tokenizer,
        max_seq_len=args.max_seq_len,
        stride=args.window_stride,
        max_windows=args.max_windows_per_chunk,
        canary_windows=args.canary_windows_per_chunk,
        canary_selection=args.canary_selection,
    )
    cfg = NativeGCOConfig(
        reasoner_policy=args.reasoner_policy,
        write_mode=args.write_mode,
        direct_write_error_scale=args.direct_write_error_scale,
        min_active_topology=args.min_active_topology,
        lr=args.lr,
        direct_write_ridge=args.direct_write_ridge,
        direct_write_protect=args.direct_write_protect,
        beta_pressure=args.beta_pressure,
        beta_formation=args.beta_formation,
        beta_protection=args.beta_protection,
        beta_decay=args.beta_decay,
        beta_usage=args.beta_usage,
        beta_capacity=args.beta_capacity,
        beta_recurrent=args.beta_recurrent,
        reasoner_lr=args.reasoner_lr,
        reasoner_weight_decay=args.reasoner_weight_decay,
        reasoner_update_clip=args.reasoner_update_clip,
        internal_gate_lr_scale=args.internal_gate_lr_scale,
        internal_write_gate_lr_scale=args.internal_write_gate_lr_scale,
        internal_protect_gate_lr_scale=args.internal_protect_gate_lr_scale,
        internal_rewire_gate_lr_scale=args.internal_rewire_gate_lr_scale,
        internal_forget_gate_lr_scale=args.internal_forget_gate_lr_scale,
        internal_compress_gate_lr_scale=args.internal_compress_gate_lr_scale,
        state_space_reasoner_dim=args.state_space_reasoner_dim,
        state_space_reasoner_lr=args.state_space_reasoner_lr,
        state_space_value_lr=args.state_space_value_lr,
        state_space_weight_decay=args.state_space_weight_decay,
        state_space_update_clip=args.state_space_update_clip,
        state_space_init_std=args.state_space_init_std,
        state_space_credit_beta=args.state_space_credit_beta,
        state_space_write_blend=args.state_space_write_blend,
        state_space_protect_blend=args.state_space_protect_blend,
        state_space_rewire_blend=args.state_space_rewire_blend,
        state_space_forget_blend=args.state_space_forget_blend,
        state_space_compress_blend=args.state_space_compress_blend,
        state_space_priority_blend=args.state_space_priority_blend,
        outcome_credit_lr=args.outcome_credit_lr,
        outcome_formation_lr=args.outcome_formation_lr,
        outcome_baseline_beta=args.outcome_baseline_beta,
        outcome_failure_scale=args.outcome_failure_scale,
        route_credit_scale=args.route_credit_scale,
        route_credit_logit_clip=args.route_credit_logit_clip,
        route_credit_warmup_steps=args.route_credit_warmup_steps,
        route_formation_scale=args.route_formation_scale,
        route_formation_logit_clip=args.route_formation_logit_clip,
        formation_weight_mix=args.formation_weight_mix,
        formation_row_mix=args.formation_row_mix,
        formation_col_mix=args.formation_col_mix,
        formation_module_mix=args.formation_module_mix,
        formation_multiscale_pooling=args.formation_multiscale_pooling,
        positive_utility_write_floor=args.positive_utility_write_floor,
        protect_old_route_floor=args.protect_old_route_floor,
        protect_collision_strength=args.protect_collision_strength,
        hardening_exit_threshold=args.hardening_exit_threshold,
        hardening_protection_strength=args.hardening_protection_strength,
        structural_protect_strength=args.structural_protect_strength,
        outcome_edit_cost=args.outcome_edit_cost,
        outcome_capacity_cost=args.outcome_capacity_cost,
        outcome_rewire_cost=args.outcome_rewire_cost,
        outcome_forget_cost=args.outcome_forget_cost,
        failed_write_beta=args.failed_write_beta,
        gamma=args.gamma,
        mu=args.mu,
        warmup_steps=args.warmup_steps,
        recency_tau=args.recency_tau,
        grow_lr=args.grow_lr,
        prune_lr=args.prune_lr,
        forget_lr=args.forget_lr,
        max_step_norm=args.max_step_norm,
        init_topology=args.init_topology,
        hardening_threshold=args.hardening_threshold,
        crystalline_threshold=args.crystalline_threshold,
        pathway_percentile=args.pathway_percentile,
        eps=args.eps,
    )
    model = GCONativeTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        cfg=cfg,
    ).to(device=device, dtype=dtype)

    print("NATIVE GCO SCRATCH TRANSFORMER")
    print("=" * 112)
    print("No torch.optim optimizer is used. Backward gradients are error signals only.")
    print(
        f"device={device} vocab={vocab_size} d_model={args.d_model} layers={args.n_layers} "
        f"heads={args.n_heads} d_ff={args.d_ff}"
    )
    print(
        f"reasoner_policy={args.reasoner_policy} "
        f"write_mode={args.write_mode} direct_ridge={args.direct_write_ridge:g} "
        f"direct_protect={args.direct_write_protect:g} direct_error_scale={args.direct_write_error_scale} "
        f"min_active_topology={args.min_active_topology:g}"
    )
    print(
        f"outcome_credit={args.outcome_credit_mode} outcome_lr={args.outcome_credit_lr:g} "
        f"failure_scale={args.outcome_failure_scale:g} "
        f"route_credit_scale={args.route_credit_scale:g} route_clip={args.route_credit_logit_clip:g} "
        f"route_warmup={args.route_credit_warmup_steps} "
        f"route_formation_scale={args.route_formation_scale:g} route_formation_clip={args.route_formation_logit_clip:g} "
        f"formation_mix=w{args.formation_weight_mix:g}/r{args.formation_row_mix:g}/c{args.formation_col_mix:g}/m{args.formation_module_mix:g} "
        f"formation_pool={args.formation_multiscale_pooling} "
        f"write_floor={args.positive_utility_write_floor:g} "
        f"protect_floor={args.protect_old_route_floor:g} "
        f"protect_collision={args.protect_collision_strength:g} "
        f"hardening_exit={args.hardening_exit_threshold:g} "
        f"hardening_protect={args.hardening_protection_strength:g} "
        f"struct_protect={args.structural_protect_strength:g} "
        f"internal_gate_scale={args.internal_gate_lr_scale:g} "
        f"gate_scales w/p/r/f/c="
        f"{args.internal_write_gate_lr_scale:g}/{args.internal_protect_gate_lr_scale:g}/"
        f"{args.internal_rewire_gate_lr_scale:g}/{args.internal_forget_gate_lr_scale:g}/"
        f"{args.internal_compress_gate_lr_scale:g} "
        f"ssr_dim={args.state_space_reasoner_dim} ssr_lr={args.state_space_reasoner_lr:g} "
        f"ssr_value_lr={args.state_space_value_lr:g} "
        f"ssr_blend w/p/r/f/c/pri="
        f"{args.state_space_write_blend:g}/{args.state_space_protect_blend:g}/"
        f"{args.state_space_rewire_blend:g}/{args.state_space_forget_blend:g}/"
        f"{args.state_space_compress_blend:g}/{args.state_space_priority_blend:g} "
        f"maturity_source={args.maturity_source} "
        f"route_trace_limit={args.route_trace_limit}"
    )
    print(
        "phases: sculpting=control gate near 0, continual-learning=control gate near 1; "
        "diagnostics: hardening latch enters F*ctrl>={:.2f}, exits F*ctrl<{:.2f}, crystalline=protect>={:.2f}".format(
            args.hardening_threshold,
            args.hardening_exit_threshold,
            args.crystalline_threshold,
        )
    )
    print("=" * 112)

    trace: list[dict[str, object]] = []
    chunk_evals: list[dict[str, object]] = []
    canary_evals: list[dict[str, object]] = []
    canary_baselines: dict[str, dict[str, float]] = {}
    eval_loss_history: dict[str, list[float]] = {}
    global_step = 0
    last_summary: dict[str, float] | None = None
    last_outcome_summary: dict[str, float] | None = None
    first_train_loss: float | None = None
    previous_logged_loss: float | None = None
    failed_write_ema = 0.0
    outcome_utility_baseline = 0.0
    for chunk_index, item in enumerate(dataset, start=1):
        chunk_id = str(item["chunk_id"])
        inputs = item["inputs"]
        targets = item["targets"]
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("Dataset inputs and targets must be tensors.")
        canary_inputs = item.get("canary_inputs")
        canary_count = canary_inputs.shape[0] if isinstance(canary_inputs, torch.Tensor) else 0
        print(f"\nchunk {chunk_index}/{len(dataset)}: {chunk_id} windows={inputs.shape[0]} canaries={canary_count}")
        for epoch in range(1, args.epochs_per_chunk + 1):
            permutation = torch.randperm(inputs.shape[0])
            progress = tqdm(
                range(0, inputs.shape[0], args.batch_size),
                desc=f"{chunk_id} epoch {epoch}/{args.epochs_per_chunk}",
                unit="batch",
            )
            for start in progress:
                global_step += 1
                idx = permutation[start : start + args.batch_size]
                batch_inputs = inputs[idx].to(device)
                batch_targets = targets[idx].to(device)
                model.train()
                model.zero_gco_grads()
                logits = model(batch_inputs)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
                require_finite_tensor("native_gco_loss", loss)
                loss_value = scalar(loss)
                batch_token_accuracy = token_accuracy_from_logits(logits, batch_targets)
                if first_train_loss is None:
                    first_train_loss = loss_value
                developmental_maturity = developmental_maturity_from_metrics(
                    source=args.maturity_source,
                    first_loss=first_train_loss,
                    current_loss=loss_value,
                    token_accuracy=batch_token_accuracy,
                    eps=args.eps,
                )
                loss.backward()
                stats = model.gco_step(
                    global_step,
                    developmental_maturity=developmental_maturity,
                    failed_write_signal=failed_write_ema,
                )
                summary = aggregate_stats(stats)
                should_log = global_step == 1 or global_step % args.log_every == 0
                selected_route_traces = (
                    model.selected_route_traces(limit=args.route_trace_limit) if should_log else []
                )
                outcome_summary: dict[str, float] = {
                    "outcome_loss_after_update": loss_value,
                    "outcome_loss_reduction": 0.0,
                    "outcome_loss_reduction_relative": 0.0,
                    "outcome_utility": 0.0,
                    "outcome_utility_baseline_before": outcome_utility_baseline,
                    "outcome_advantage": 0.0,
                    "outcome_utility_baseline_after": outcome_utility_baseline,
                    "outcome_credit_selected_count": 0.0,
                    "outcome_credit_eligibility_mass": 0.0,
                    "outcome_credit_eligibility_max_share": 0.0,
                    "outcome_credit_advantage": 0.0,
                    "outcome_credit_route_advantage_mean": 0.0,
                    "outcome_credit_route_advantage_abs_mean": 0.0,
                    "outcome_credit_route_advantage_max": 0.0,
                    "outcome_credit_route_formation_utility_mean": 0.0,
                    "outcome_credit_route_formation_utility_max": 0.0,
                    "outcome_credit_utility_target": 0.0,
                    "outcome_credit_utility_target_min": 0.0,
                    "outcome_credit_utility_target_max": 0.0,
                    "outcome_credit_write_gate_mean": 0.0,
                    "outcome_credit_protect_target_mean": 0.0,
                    "outcome_credit_protect_target_min": 0.0,
                    "outcome_credit_protect_target_max": 0.0,
                    "outcome_credit_protect_gate_mean": 0.0,
                    "outcome_credit_protect_error_abs_mean": 0.0,
                    "outcome_credit_reliability_target_mean": 0.0,
                    "outcome_credit_reliability_target_min": 0.0,
                    "outcome_credit_reliability_target_max": 0.0,
                    "outcome_credit_reliability_pred_mean": 0.0,
                    "outcome_credit_reliability_error_abs_mean": 0.0,
                    "outcome_credit_collision_target_mean": 0.0,
                    "outcome_credit_collision_target_min": 0.0,
                    "outcome_credit_collision_target_max": 0.0,
                    "outcome_credit_collision_pred_mean": 0.0,
                    "outcome_credit_collision_error_abs_mean": 0.0,
                    "outcome_credit_gate_error_abs_mean": 0.0,
                    "outcome_credit_update_norm": 0.0,
                    "failed_write_ema_after": failed_write_ema,
                }
                if args.outcome_credit_mode == "same_batch":
                    model.eval()
                    with torch.no_grad():
                        post_logits = model(batch_inputs)
                        post_loss = F.cross_entropy(
                            post_logits.reshape(-1, vocab_size),
                            batch_targets.reshape(-1),
                        )
                    require_finite_tensor("native_gco_post_update_loss", post_loss)
                    post_loss_value = scalar(post_loss)
                    loss_reduction = loss_value - post_loss_value
                    relative_loss_reduction = loss_reduction / max(abs(loss_value), args.eps)
                    outcome_utility = (
                        relative_loss_reduction
                        - args.outcome_edit_cost * summary["safe_update_norm"]
                        - args.outcome_capacity_cost * summary["write_edit_fraction"]
                        - args.outcome_rewire_cost * summary["rewire_edit_fraction"]
                        - args.outcome_forget_cost * summary["forget_edit_fraction"]
                    )
                    require_finite_float("native_gco_outcome_utility", outcome_utility)
                    outcome_utility_baseline_before = outcome_utility_baseline
                    outcome_advantage = outcome_utility - outcome_utility_baseline_before
                    require_finite_float("native_gco_outcome_advantage", outcome_advantage)
                    outcome_credit_stats = model.apply_outcome_credit(
                        utility=outcome_utility,
                        advantage=outcome_advantage,
                        step=global_step,
                    )
                    failed_write_pressure = min(1.0, max(0.0, -outcome_utility / args.outcome_failure_scale))
                    failed_write_ema = (
                        args.failed_write_beta * failed_write_ema
                        + (1.0 - args.failed_write_beta) * failed_write_pressure
                    )
                    require_finite_float("failed_write_ema", failed_write_ema)
                    outcome_utility_baseline = (
                        args.outcome_baseline_beta * outcome_utility_baseline
                        + (1.0 - args.outcome_baseline_beta) * outcome_utility
                    )
                    require_finite_float("outcome_utility_baseline", outcome_utility_baseline)
                    outcome_summary = {
                        "outcome_loss_after_update": post_loss_value,
                        "outcome_loss_reduction": loss_reduction,
                        "outcome_loss_reduction_relative": relative_loss_reduction,
                        "outcome_utility": outcome_utility,
                        "outcome_utility_baseline_before": outcome_utility_baseline_before,
                        "outcome_advantage": outcome_advantage,
                        "outcome_utility_baseline_after": outcome_utility_baseline,
                        "failed_write_ema_after": failed_write_ema,
                        **aggregate_outcome_credit_stats(outcome_credit_stats),
                    }
                    for key, value in outcome_summary.items():
                        require_finite_float(key, value)
                    model.train()
                last_outcome_summary = outcome_summary
                last_summary = summary
                if should_log:
                    module_stats = [asdict(item) for item in stats]
                    hot_modules = hottest_modules(stats, limit=args.hot_modules)
                    row: dict[str, object] = {
                        "step": global_step,
                        "chunk_id": chunk_id,
                        "loss": loss_value,
                        "batch_token_accuracy": batch_token_accuracy,
                        "loss_delta_from_first": loss_value - first_train_loss,
                        "loss_delta_from_previous_log": 0.0
                        if previous_logged_loss is None
                        else loss_value - previous_logged_loss,
                        **summary,
                        **outcome_summary,
                        "module_stats": module_stats,
                        "hot_modules": hot_modules,
                        "selected_route_traces": selected_route_traces,
                    }
                    trace.append(row)
                    previous_logged_loss = loss_value
                    tqdm.write(
                        "step={step:6d} chunk={chunk_id} loss={loss:.4f} "
                        "acc={batch_token_accuracy:.3f} "
                        "dL0={loss_delta_from_first:+.4f} dLlog={loss_delta_from_previous_log:+.4f} "
                        "P={pressure_mean:.3f}/{pressure_max:.3f} "
                        "S/H/C={sculpting_fraction:.2f}/{hardening_fraction:.2f}/{crystalline_fraction:.2f} "
                        "active={active_sculpting_fraction:.2f}/{active_hardening_fraction:.2f}/{active_crystalline_fraction:.2f} "
                        "Hlat={active_hardening_latch_fraction:.2f} "
                        "A={topology_mean:.3f} U={usage_mean:.3g} "
                        "mature={developmental_maturity:.3f} ctrl={control_phase_gate:.3f} fail={failed_write_signal:.3f} "
                        "upd={safe_update_norm:.3g}".format(**row)
                    )
                    tqdm.write(
                        "  mechanics path={pathway_mean:.3g}/{pathway_max:.3g} "
                        "errP={error_pressure_mean:.3g}/{error_pressure_max:.3g} "
                        "basis={direct_write_basis_mean:.3g}/{direct_write_basis_max:.3g} "
                        "write={direct_write_norm:.3g}/{raw_write_norm:.3g}->{safe_direction_norm:.3g} "
                        "safe={safe_direction_ratio:.2f} proj={projection_removed_ratio:.2f} scale={step_scale:.2f} "
                        "WΔ={total_weight_delta_norm:.2e}/sel{selected_weight_delta_norm:.2e}/unsel{unselected_weight_delta_norm:.2e} "
                        "AΔ=sel{selected_topology_delta_norm:.2e}/unsel{unselected_topology_delta_norm:.2e} "
                        "rewire=g{topology_grow_mean:.1e}/p{topology_prune_mean:.1e}/d{topology_delta_abs_mean:.1e} "
                        "forget={forget_rate_mean:.1e}/{forget_rate_max:.1e} "
                        "inactive={inactive_mean:.2f}/{inactive_max:.2f}".format(**row)
                    )
                    tqdm.write(
                        "  state age={route_age_mean:.2f}/{route_age_max:.0f} "
                        "rec={route_recency_mean:.2f}/{route_recency_max:.2f} "
                        "FPD={formation_pressure_mean:.3f}/{protection_pressure_mean:.3f}/{decay_pressure_mean:.3f} "
                        "Feff={formation_effective_mean:.3f}/{formation_effective_max:.3f} "
                        "Fms={formation_row_mean:.3f}/{formation_col_mean:.3f}/{formation_module_mean:.3f} "
                        "Hlat={hardening_latch_mean:.3f}/{hardening_latch_max:.3f} "
                        "SSRnorm={ssr_row_state_norm_mean:.3f}/{ssr_col_state_norm_mean:.3f}/{ssr_module_state_norm_mean:.3f} "
                        "SSRgateW={ssr_write_gate_mean:.3f}/{ssr_write_gate_max:.3f} "
                        "SSRgateP={ssr_protect_gate_mean:.3f}/{ssr_protect_gate_max:.3f} "
                        "SSRgateR/F/C={ssr_rewire_gate_mean:.3f}/{ssr_forget_gate_mean:.3f}/{ssr_compress_gate_mean:.3f} "
                        "SSRrel={ssr_reliability_mean:.3f}/{ssr_reliability_max:.3f} "
                        "SSRcol={ssr_collision_mean:.3f}/{ssr_collision_max:.3f} "
                        "SSReff={ssr_protect_eff_mean:.3f}/{ssr_protect_eff_max:.3f} "
                        "SSRgain/cap/fs/pri={ssr_gain_pred_mean:.3f}/{ssr_capacity_cost_mean:.3f}/{ssr_forget_safe_mean:.3f}/{ssr_priority_mean:.3f} "
                        "SSRval={ssr_value_pred_mean:+.3g}/{ssr_value_pred_max:+.3g} "
                        "SSRcred={ssr_credit_mean:.3g}/{ssr_credit_max:.3g} "
                        "SSRtd={ssr_td_error_abs_mean:.3g}/{ssr_td_error_abs_max:.3g} "
                        "SSRupd={ssr_update_norm:.3g} "
                        "Wpress={write_pressure_mean:.3f}/{write_pressure_max:.3f} "
                        "Pneed={protect_need_mean:.3f}/{protect_need_max:.3f} "
                        "struct={structural_protection_mean:.3f}/{structural_protection_max:.3f} "
                        "Pin={structural_input_protection_mean:.3f}/{structural_input_protection_max:.3f} "
                        "lambdaP={direct_write_protect_effective:.3g} "
                        "S={recurrent_state_mean:.2f}|{recurrent_state_abs_mean:.2f}/d{recurrent_state_delta_mean:.2e} "
                        "cap protected/free/plastic/obsolete="
                        "{protected_capacity_mean:.3f}/{free_capacity_mean:.3f}/"
                        "{plastic_capacity_mean:.3f}/{obsolete_capacity_mean:.3f} "
                        "gates_eff w/p/r/f/c="
                        "{write_gate_mean:.3f}/{protect_gate_mean:.3f}/"
                        "{rewire_gate_mean:.3f}/{forget_gate_mean:.3f}/{compress_gate_mean:.3f} "
                        "gates_raw w/p/r/f/c="
                        "{write_gate_raw_mean:.3f}/{protect_gate_raw_mean:.3f}/"
                        "{rewire_gate_raw_mean:.3f}/{forget_gate_raw_mean:.3f}/{compress_gate_raw_mean:.3f} "
                        "edits w/p/r/f/c="
                        "{write_edit_fraction:.3f}/{protect_edit_fraction:.3f}/"
                        "{rewire_edit_fraction:.3f}/{forget_edit_fraction:.3f}/{compress_edit_fraction:.3f} "
                        "mass={write_score_mass:.1f}/{protect_score_mass:.1f}/"
                        "{rewire_score_mass:.1f}/{forget_score_mass:.1f}/{compress_score_mass:.1f} "
                        "reasoner roleH/max={reasoner_role_entropy_mean:.3f}/{reasoner_role_max_share_mean:.3f} "
                        "util/err={reasoner_gate_utility_mean:.3f}/{reasoner_gate_error_abs_mean:.3f} "
                        "rW/d={reasoner_weight_norm:.3g}/{reasoner_update_norm:.3g}".format(**row)
                    )
                    if hot_modules:
                        tqdm.write(f"  hot {format_hot_modules(hot_modules)}")
                    if selected_route_traces:
                        tqdm.write(f"  routes {format_route_trace_rows(selected_route_traces)}")
                    if args.outcome_credit_mode == "same_batch":
                        tqdm.write(
                            "  outcome L_after={outcome_loss_after_update:.4f} "
                            "dL={outcome_loss_reduction:+.4g} rel={outcome_loss_reduction_relative:+.3g} "
                            "U={outcome_utility:+.3g} base={outcome_utility_baseline_before:+.3g}->{outcome_utility_baseline_after:+.3g} "
                            "adv={outcome_advantage:+.3g} target={outcome_credit_utility_target_min:.3f}/{outcome_credit_utility_target:.3f}/{outcome_credit_utility_target_max:.3f} "
                            "selected={outcome_credit_selected_count:.0f} elig={outcome_credit_eligibility_mass:.3g} "
                            "max_share={outcome_credit_eligibility_max_share:.3g} "
                            "rAdv={outcome_credit_route_advantage_mean:+.3g}/{outcome_credit_route_advantage_abs_mean:.3g}/{outcome_credit_route_advantage_max:+.3g} "
                            "form={outcome_credit_route_formation_utility_mean:.3g}/{outcome_credit_route_formation_utility_max:.3g} "
                            "gateW={outcome_credit_write_gate_mean:.3f} "
                            "gateP={outcome_credit_protect_gate_mean:.3f} "
                            "targetP={outcome_credit_protect_target_min:.3f}/{outcome_credit_protect_target_mean:.3f}/{outcome_credit_protect_target_max:.3f} "
                            "targetR={outcome_credit_reliability_target_min:.3f}/{outcome_credit_reliability_target_mean:.3f}/{outcome_credit_reliability_target_max:.3f} "
                            "predR={outcome_credit_reliability_pred_mean:.3f} "
                            "targetC={outcome_credit_collision_target_min:.3f}/{outcome_credit_collision_target_mean:.3f}/{outcome_credit_collision_target_max:.3f} "
                            "predC={outcome_credit_collision_pred_mean:.3f} "
                            "errW={outcome_credit_gate_error_abs_mean:.3f} "
                            "errP={outcome_credit_protect_error_abs_mean:.3f} "
                            "errR={outcome_credit_reliability_error_abs_mean:.3f} "
                            "errC={outcome_credit_collision_error_abs_mean:.3f} "
                            "rΔ={outcome_credit_update_norm:.3g} "
                            "fail_after={failed_write_ema_after:.3f}".format(**row)
                        )
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    acc=f"{batch_token_accuracy:.3f}",
                    P=f"{summary['pressure_max']:.2f}",
                    aS=f"{summary['active_sculpting_fraction']:.2f}",
                    aH=f"{summary['active_hardening_fraction']:.2f}",
                    aC=f"{summary['active_crystalline_fraction']:.2f}",
                    hL=f"{summary['active_hardening_latch_fraction']:.2f}",
                    m=f"{summary['developmental_maturity']:.2f}",
                    fail=f"{summary['failed_write_signal']:.2f}",
                    A=f"{summary['topology_mean']:.2f}",
                    out=f"{outcome_summary['outcome_loss_reduction']:+.2g}",
                )
        if args.eval_after_chunk:
            seen: list[dict[str, object]] = []
            for eval_item in dataset[:chunk_index]:
                eval_inputs = eval_item["inputs"]
                eval_targets = eval_item["targets"]
                if not isinstance(eval_inputs, torch.Tensor) or not isinstance(eval_targets, torch.Tensor):
                    raise TypeError("Evaluation inputs and targets must be tensors.")
                eval_metrics = evaluate_metrics(
                    model,
                    eval_inputs,
                    eval_targets,
                    batch_size=args.batch_size,
                    device=device,
                )
                eval_chunk_id = str(eval_item["chunk_id"])
                history = eval_loss_history.setdefault(eval_chunk_id, [])
                previous_eval_loss = history[-1] if history else None
                history.append(eval_metrics["loss"])
                seen.append(
                    {
                        "chunk_id": eval_chunk_id,
                        "loss": eval_metrics["loss"],
                        "token_accuracy": eval_metrics["token_accuracy"],
                        "loss_delta_from_previous_eval": 0.0
                        if previous_eval_loss is None
                        else eval_metrics["loss"] - previous_eval_loss,
                    }
                )
            chunk_evals.append({"after_chunk": chunk_id, "seen_chunk_losses": seen})
            print(
                "seen_chunk_metrics: "
                + ", ".join(
                    f"{row['chunk_id']}=loss{row['loss']:.4f}/acc{row['token_accuracy']:.3f}/"
                    f"d{row['loss_delta_from_previous_eval']:+.4f}"
                    for row in seen
                )
            )

        if args.canary_windows_per_chunk > 0:
            canary_rows: list[dict[str, object]] = []
            for eval_item in dataset[:chunk_index]:
                eval_canary_inputs = eval_item.get("canary_inputs")
                eval_canary_targets = eval_item.get("canary_targets")
                if not isinstance(eval_canary_inputs, torch.Tensor) or not isinstance(eval_canary_targets, torch.Tensor):
                    raise TypeError("Canary inputs and targets must be tensors when canaries are enabled.")
                eval_chunk_id = str(eval_item["chunk_id"])
                metrics = evaluate_metrics(
                    model,
                    eval_canary_inputs,
                    eval_canary_targets,
                    batch_size=args.batch_size,
                    device=device,
                )
                baseline = canary_baselines.setdefault(eval_chunk_id, dict(metrics))
                canary_rows.append(canary_drift_row(chunk_id=eval_chunk_id, metrics=metrics, baseline=baseline))
            canary_evals.append({"after_chunk": chunk_id, "canaries": canary_rows})
            print(
                "canary_metrics: "
                + ", ".join(
                    f"{row['chunk_id']}=loss{row['loss']:.4f}/acc{row['token_accuracy']:.3f}/"
                    f"dL{row['loss_delta_from_canary_baseline']:+.4f}/"
                    f"m{row['target_margin_mean']:+.3f}/"
                    f"dM{row['target_margin_delta_from_canary_baseline']:+.3f}/"
                    f"h{row['canary_health']:.3f}"
                    for row in canary_rows
                )
            )

    if last_summary is None:
        raise RuntimeError("Native GCO training completed without any update steps.")
    final_stats = last_summary
    result = {
        "config": {
            "tokenizer_path": str(args.tokenizer_path),
            "chunks_path": str(args.chunks_path),
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "window_stride": args.window_stride,
            "max_windows_per_chunk": args.max_windows_per_chunk,
            "canary_windows_per_chunk": args.canary_windows_per_chunk,
            "canary_selection": args.canary_selection,
            "chunk_ids": [str(item["chunk_id"]) for item in dataset],
            "epochs_per_chunk": args.epochs_per_chunk,
            "batch_size": args.batch_size,
            "hot_modules": args.hot_modules,
            "route_trace_limit": args.route_trace_limit,
            "outcome_credit_mode": args.outcome_credit_mode,
            "maturity_source": args.maturity_source,
            **asdict(cfg),
        },
        "question": "Can a transformer train from scratch with native GCO self-updates and visible state transitions?",
        "trace": trace,
        "chunk_evals": chunk_evals,
        "canary_evals": canary_evals,
        "first_train_loss": first_train_loss,
        "final_stats": final_stats,
        "final_outcome_summary": last_outcome_summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("\nNATIVE GCO SCRATCH SUMMARY")
    print("=" * 112)
    print(
        "final P={pressure_mean:.3f}/{pressure_max:.3f} "
        "S/H/C={sculpting_fraction:.2f}/{hardening_fraction:.2f}/{crystalline_fraction:.2f} "
        "active={active_sculpting_fraction:.2f}/{active_hardening_fraction:.2f}/{active_crystalline_fraction:.2f} "
        "Hlat={active_hardening_latch_fraction:.2f} "
        "A={topology_mean:.3f} U={usage_mean:.3g} "
        "mature={developmental_maturity:.3f} ctrl={control_phase_gate:.3f} fail={failed_write_signal:.3f} "
        "update={safe_update_norm:.3g}".format(**final_stats)
    )
    print(
        "final mechanics path={pathway_mean:.3g}/{pathway_max:.3g} "
        "errP={error_pressure_mean:.3g}/{error_pressure_max:.3g} "
        "basis={direct_write_basis_mean:.3g}/{direct_write_basis_max:.3g} "
        "write={direct_write_norm:.3g}/{raw_write_norm:.3g}->{safe_direction_norm:.3g} "
        "safe={safe_direction_ratio:.2f} proj={projection_removed_ratio:.2f} scale={step_scale:.2f} "
        "W_delta={total_weight_delta_norm:.2e}/selected{selected_weight_delta_norm:.2e}/"
        "unselected{unselected_weight_delta_norm:.2e} "
        "A_delta=selected{selected_topology_delta_norm:.2e}/unselected{unselected_topology_delta_norm:.2e} "
        "rewire=g{topology_grow_mean:.1e}/p{topology_prune_mean:.1e}/d{topology_delta_abs_mean:.1e} "
        "forget={forget_rate_mean:.1e}/{forget_rate_max:.1e} "
        "inactive={inactive_mean:.2f}/{inactive_max:.2f}".format(**final_stats)
    )
    print(
        "final state age={route_age_mean:.2f}/{route_age_max:.0f} "
        "rec={route_recency_mean:.2f}/{route_recency_max:.2f} "
        "FPD={formation_pressure_mean:.3f}/{protection_pressure_mean:.3f}/{decay_pressure_mean:.3f} "
        "Feff={formation_effective_mean:.3f}/{formation_effective_max:.3f} "
        "Fms={formation_row_mean:.3f}/{formation_col_mean:.3f}/{formation_module_mean:.3f} "
        "Hlat={hardening_latch_mean:.3f}/{hardening_latch_max:.3f} "
        "SSRnorm={ssr_row_state_norm_mean:.3f}/{ssr_col_state_norm_mean:.3f}/{ssr_module_state_norm_mean:.3f} "
        "SSRgateW={ssr_write_gate_mean:.3f}/{ssr_write_gate_max:.3f} "
        "SSRgateP={ssr_protect_gate_mean:.3f}/{ssr_protect_gate_max:.3f} "
        "SSRgateR/F/C={ssr_rewire_gate_mean:.3f}/{ssr_forget_gate_mean:.3f}/{ssr_compress_gate_mean:.3f} "
        "SSRrel={ssr_reliability_mean:.3f}/{ssr_reliability_max:.3f} "
        "SSRcol={ssr_collision_mean:.3f}/{ssr_collision_max:.3f} "
        "SSReff={ssr_protect_eff_mean:.3f}/{ssr_protect_eff_max:.3f} "
        "SSRgain/cap/fs/pri={ssr_gain_pred_mean:.3f}/{ssr_capacity_cost_mean:.3f}/{ssr_forget_safe_mean:.3f}/{ssr_priority_mean:.3f} "
        "SSRval={ssr_value_pred_mean:+.3g}/{ssr_value_pred_max:+.3g} "
        "SSRcred={ssr_credit_mean:.3g}/{ssr_credit_max:.3g} "
        "SSRtd={ssr_td_error_abs_mean:.3g}/{ssr_td_error_abs_max:.3g} "
        "SSRupd={ssr_update_norm:.3g} "
        "Wpress={write_pressure_mean:.3f}/{write_pressure_max:.3f} "
        "Pneed={protect_need_mean:.3f}/{protect_need_max:.3f} "
        "struct={structural_protection_mean:.3f}/{structural_protection_max:.3f} "
        "Pin={structural_input_protection_mean:.3f}/{structural_input_protection_max:.3f} "
        "lambdaP={direct_write_protect_effective:.3g} "
        "S={recurrent_state_mean:.2f}|{recurrent_state_abs_mean:.2f}/d{recurrent_state_delta_mean:.2e} "
        "cap protected/free/plastic/obsolete="
        "{protected_capacity_mean:.3f}/{free_capacity_mean:.3f}/"
        "{plastic_capacity_mean:.3f}/{obsolete_capacity_mean:.3f} "
        "gates_eff w/p/r/f/c="
        "{write_gate_mean:.3f}/{protect_gate_mean:.3f}/"
        "{rewire_gate_mean:.3f}/{forget_gate_mean:.3f}/{compress_gate_mean:.3f} "
        "gates_raw w/p/r/f/c="
        "{write_gate_raw_mean:.3f}/{protect_gate_raw_mean:.3f}/"
        "{rewire_gate_raw_mean:.3f}/{forget_gate_raw_mean:.3f}/{compress_gate_raw_mean:.3f} "
        "edits w/p/r/f/c="
        "{write_edit_fraction:.3f}/{protect_edit_fraction:.3f}/"
        "{rewire_edit_fraction:.3f}/{forget_edit_fraction:.3f}/{compress_edit_fraction:.3f} "
        "mass={write_score_mass:.1f}/{protect_score_mass:.1f}/"
        "{rewire_score_mass:.1f}/{forget_score_mass:.1f}/{compress_score_mass:.1f} "
        "reasoner roleH/max={reasoner_role_entropy_mean:.3f}/{reasoner_role_max_share_mean:.3f} "
        "util/err={reasoner_gate_utility_mean:.3f}/{reasoner_gate_error_abs_mean:.3f} "
        "rW/d={reasoner_weight_norm:.3g}/{reasoner_update_norm:.3g}".format(**final_stats)
    )
    if args.outcome_credit_mode == "same_batch":
        if last_outcome_summary is None:
            raise RuntimeError("Outcome credit mode was enabled but no outcome summary was recorded.")
        print(
            "final outcome L_after={outcome_loss_after_update:.4f} "
            "dL={outcome_loss_reduction:+.4g} rel={outcome_loss_reduction_relative:+.3g} "
            "U={outcome_utility:+.3g} base={outcome_utility_baseline_before:+.3g}->{outcome_utility_baseline_after:+.3g} "
            "adv={outcome_advantage:+.3g} target={outcome_credit_utility_target_min:.3f}/{outcome_credit_utility_target:.3f}/{outcome_credit_utility_target_max:.3f} "
            "selected={outcome_credit_selected_count:.0f} elig={outcome_credit_eligibility_mass:.3g} "
            "max_share={outcome_credit_eligibility_max_share:.3g} "
            "rAdv={outcome_credit_route_advantage_mean:+.3g}/{outcome_credit_route_advantage_abs_mean:.3g}/{outcome_credit_route_advantage_max:+.3g} "
            "form={outcome_credit_route_formation_utility_mean:.3g}/{outcome_credit_route_formation_utility_max:.3g} "
            "gateW={outcome_credit_write_gate_mean:.3f} "
            "gateP={outcome_credit_protect_gate_mean:.3f} "
            "targetP={outcome_credit_protect_target_min:.3f}/{outcome_credit_protect_target_mean:.3f}/{outcome_credit_protect_target_max:.3f} "
            "targetR={outcome_credit_reliability_target_min:.3f}/{outcome_credit_reliability_target_mean:.3f}/{outcome_credit_reliability_target_max:.3f} "
            "predR={outcome_credit_reliability_pred_mean:.3f} "
            "targetC={outcome_credit_collision_target_min:.3f}/{outcome_credit_collision_target_mean:.3f}/{outcome_credit_collision_target_max:.3f} "
            "predC={outcome_credit_collision_pred_mean:.3f} "
            "errW={outcome_credit_gate_error_abs_mean:.3f} "
            "errP={outcome_credit_protect_error_abs_mean:.3f} "
            "errR={outcome_credit_reliability_error_abs_mean:.3f} "
            "errC={outcome_credit_collision_error_abs_mean:.3f} r_delta={outcome_credit_update_norm:.3g} "
            "fail_after={failed_write_ema_after:.3f}".format(
                **last_outcome_summary
            )
        )
    print(f"wrote_json={args.output_json}")
    return result


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
