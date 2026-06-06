"""Run the CL phase from a fitted tiny base using reasoned native GCO writes.

This script loads the tiny base checkpoint and protected geometry anchors from
``gco_prepare_tiny_cl_base.py``. It then trains on a new real-text slice with
native GCO updates only. Each update is treated as a virtual trial:

1. Snapshot the model.
2. Apply a proposed GCO write on new data.
3. Measure new improvement, old probe damage, and old anchor drift.
4. Keep the write only if total utility is above threshold; otherwise restore.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_native_scratch_transformer import (
    GCOEmbedding,
    GCOLinear,
    GCONativeTransformer,
    NativeGCOConfig,
    aggregate_stats,
)
from experiments.gco_math.gco_prepare_tiny_cl_base import (
    build_lm_windows,
    capture_geometry_anchors,
    evaluate_model,
    load_chunks,
)


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


def finite_float(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")


def bounded_float(name: str, value: float, low: float, high: float) -> None:
    finite_float(name, value)
    if value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}.")


def word_span(text: str, start: int, count: int) -> str:
    nonnegative_int("new_word_start", start)
    positive_int("new_word_count", count)
    words = text.split()
    end = start + count
    if end > len(words):
        raise ValueError(f"Requested word span [{start}, {end}) but text has only {len(words)} words.")
    return " ".join(words[start:end])


def snapshot_state(model: GCONativeTransformer) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def restore_state(model: GCONativeTransformer, snapshot: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(snapshot, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State restore mismatch: missing={missing}, unexpected={unexpected}.")


def instantiate_model(checkpoint: dict[str, Any], cfg: NativeGCOConfig, device: torch.device) -> GCONativeTransformer:
    model_config = checkpoint["model_config"]
    required = {"vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"}
    missing = required.difference(model_config)
    if missing:
        raise RuntimeError(f"Checkpoint model_config missing fields: {sorted(missing)}.")
    model = GCONativeTransformer(
        vocab_size=int(model_config["vocab_size"]),
        d_model=int(model_config["d_model"]),
        n_layers=int(model_config["n_layers"]),
        n_heads=int(model_config["n_heads"]),
        d_ff=int(model_config["d_ff"]),
        max_seq_len=int(model_config["max_seq_len"]),
        cfg=cfg,
    ).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing_keys}, unexpected={unexpected_keys}.")
    return model


def build_cl_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> NativeGCOConfig:
    if "native_gco_config" not in checkpoint:
        raise RuntimeError("Checkpoint missing native_gco_config.")
    cfg = NativeGCOConfig(**checkpoint["native_gco_config"])
    return replace(
        cfg,
        reasoner_policy="fixed_geometric",
        write_mode="direct",
        lr=args.gco_lr,
        max_step_norm=args.max_step_norm,
        direct_write_ridge=args.direct_write_ridge,
        direct_write_protect=args.direct_write_protect,
        protect_old_route_floor=args.protect_old_route_floor,
        protect_collision_strength=args.protect_collision_strength,
        grow_lr=args.grow_lr,
        prune_lr=args.prune_lr,
        forget_lr=args.forget_lr,
        reasoner_lr=0.0,
        state_space_reasoner_dim=0,
        state_space_reasoner_lr=0.0,
        state_space_value_lr=0.0,
        outcome_credit_lr=0.0,
        outcome_formation_lr=0.0,
    )


def module_map(model: GCONativeTransformer) -> dict[str, GCOLinear | GCOEmbedding]:
    modules = {module.name: module for module in model.gco_modules()}
    if not modules:
        raise RuntimeError("Model has no GCO modules.")
    return modules


@torch.no_grad()
def seed_old_geometry(
    model: GCONativeTransformer,
    anchors: dict[str, Any],
    *,
    protect_scale: float,
    normalize: bool,
    device: torch.device,
) -> dict[str, float]:
    nonnegative_float("anchor_protect_scale", protect_scale)
    modules = module_map(model)
    anchor_modules = anchors["anchor_bank"]["modules"]
    seeded = 0
    max_protection = 0.0
    mean_protection = 0.0
    for name, entry in anchor_modules.items():
        if name not in modules:
            raise RuntimeError(f"Anchor module {name!r} does not exist in loaded model.")
        module = modules[name]
        pathway = entry["pathway_mean"].to(device=device, dtype=module.W.dtype)
        if pathway.shape != module.W.shape:
            raise ValueError(f"{name} anchor pathway shape {pathway.shape} != module shape {module.W.shape}.")
        if normalize:
            pathway = pathway / pathway.max().clamp_min(module.cfg.eps)
        protection = (pathway * protect_scale).clamp(0.0, 1.0)
        module.P_state.copy_(protection)
        module.H_latch.copy_(protection)
        module.U.copy_(pathway.clamp(0.0, 1.0))
        module.F_state.copy_(pathway.clamp(0.0, 1.0))
        module.F_row.copy_(pathway.max(dim=1, keepdim=True).values.clamp(0.0, 1.0))
        module.F_col.copy_(pathway.max(dim=0, keepdim=True).values.clamp(0.0, 1.0))
        module.F_module.fill_(float(pathway.max().detach().cpu()))
        module.D_state.zero_()
        module.C.copy_(protection)
        module.S.zero_()
        module.Age.zero_()
        seeded += 1
        max_protection = max(max_protection, float(protection.max().detach().cpu()))
        mean_protection += float(protection.mean().detach().cpu())
    if seeded <= 0:
        raise RuntimeError("No geometry anchors were seeded.")
    return {
        "seeded_module_count": float(seeded),
        "mean_seeded_protection": mean_protection / float(seeded),
        "max_seeded_protection": max_protection,
    }


@torch.no_grad()
def anchor_drift(
    model: GCONativeTransformer,
    anchors: dict[str, Any],
    old_inputs: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    top_routes: int,
) -> dict[str, float]:
    current = capture_geometry_anchors(
        model,
        old_inputs,
        batch_size=batch_size,
        device=device,
        top_routes=top_routes,
        store_activation_snapshots=False,
    )
    pathway_drifts: list[float] = []
    activation_drifts: list[float] = []
    top_route_drifts: list[float] = []
    for name, old_entry in anchors["anchor_bank"]["modules"].items():
        current_entry = current["modules"].get(name)
        if current_entry is None:
            raise RuntimeError(f"Current anchor capture missing module {name!r}.")
        old_pathway = old_entry["pathway_mean"].to(dtype=torch.float32)
        new_pathway = current_entry["pathway_mean"].to(dtype=torch.float32)
        old_activation = old_entry["activation_mean"].to(dtype=torch.float32)
        new_activation = current_entry["activation_mean"].to(dtype=torch.float32)
        pathway_drifts.append(float(torch.linalg.vector_norm(new_pathway - old_pathway) / old_pathway.norm().clamp_min(1e-8)))
        activation_drifts.append(
            float(torch.linalg.vector_norm(new_activation - old_activation) / old_activation.norm().clamp_min(1e-8))
        )
        old_top = old_entry["top_routes"]
        rows = old_top["rows"].to(dtype=torch.long)
        cols = old_top["cols"].to(dtype=torch.long)
        old_values = old_top["values"].to(dtype=torch.float32)
        new_values = new_pathway[rows, cols]
        top_route_drifts.append(float((new_values - old_values).abs().mean()))
    if not pathway_drifts or not activation_drifts or not top_route_drifts:
        raise RuntimeError("Anchor drift calculation produced no module drifts.")
    return {
        "anchor_pathway_drift_mean": float(sum(pathway_drifts) / len(pathway_drifts)),
        "anchor_pathway_drift_max": float(max(pathway_drifts)),
        "anchor_activation_drift_mean": float(sum(activation_drifts) / len(activation_drifts)),
        "anchor_activation_drift_max": float(max(activation_drifts)),
        "anchor_top_route_abs_delta_mean": float(sum(top_route_drifts) / len(top_route_drifts)),
        "anchor_top_route_abs_delta_max": float(max(top_route_drifts)),
    }


@torch.no_grad()
def batch_loss(model: GCONativeTransformer, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    model.eval()
    logits = model(inputs.to(device))
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.to(device).reshape(-1))
    return float(loss.detach().cpu())


@torch.no_grad()
def top_fraction_mask(score: torch.Tensor, fraction: float, eps: float, *, name: str) -> torch.Tensor:
    bounded_float(f"{name}_fraction", fraction, 0.0, 1.0)
    positive_float(f"{name}_eps", eps)
    flat_score = score.reshape(-1)
    if flat_score.numel() <= 0:
        raise RuntimeError(f"{name} cannot select from an empty score tensor.")
    positive = flat_score > eps
    if fraction <= 0.0 or not bool(positive.any().detach().cpu()):
        return torch.zeros_like(score)
    requested = int(math.ceil(float(flat_score.numel()) * fraction))
    k = min(max(1, requested), int(positive.to(dtype=torch.long).sum().detach().cpu()))
    indices = torch.topk(flat_score, k=k, largest=True).indices
    mask_flat = torch.zeros_like(flat_score)
    mask_flat[indices] = 1.0
    return mask_flat.reshape_as(score)


def utility_from_measurements(
    *,
    new_loss_drop: float,
    old_loss_damage: float,
    old_baseline_loss_damage: float,
    old_margin_damage: float,
    anchor_pathway_drift_mean: float,
    anchor_activation_drift_mean: float,
    args: argparse.Namespace,
) -> float:
    return (
        new_loss_drop
        - args.old_loss_weight * old_loss_damage
        - args.old_baseline_loss_weight * old_baseline_loss_damage
        - args.old_margin_weight * old_margin_damage
        - args.anchor_pathway_weight * anchor_pathway_drift_mean
        - args.anchor_activation_weight * anchor_activation_drift_mean
    )


@torch.no_grad()
def capture_rewire_second_trial_proposal(
    model: GCONativeTransformer,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    proposal: dict[str, torch.Tensor] = {}
    module_count = 0
    route_count = 0
    selected_count = 0.0
    score_mass = 0.0
    growth_mass = 0.0
    isolation_mass = 0.0
    delta_abs_mass = 0.0
    growth_max = 0.0
    isolation_max = 0.0
    delta_abs_max = 0.0
    for module in model.gco_modules():
        if module.W.grad is None:
            raise RuntimeError(f"{module.name} has no gradient for rewire second-trial proposal.")
        pathway = module.pathway()
        if isinstance(module, GCOEmbedding):
            write_signal = module.direct_write_direction().abs()
        else:
            direct_input, direct_output_error = module.direct_write_tensors()
            sample_count = direct_input.shape[0]
            if sample_count <= 0:
                raise RuntimeError(f"{module.name} saw zero samples while building rewire second-trial proposal.")
            write_signal = (direct_output_error.transpose(0, 1) @ direct_input / float(sample_count)).abs()
        if write_signal.shape != module.A.shape:
            raise RuntimeError(f"{module.name} write-signal shape mismatch: {write_signal.shape} != {module.A.shape}.")
        protected = torch.maximum(torch.maximum(module.P_state, module.H_latch), module.C).clamp(0.0, 1.0)
        free_capacity = (1.0 - protected).clamp(0.0, 1.0)
        growth_room = (1.0 - module.A).clamp(0.0, 1.0)
        active_room = module.A.clamp(0.0, 1.0)
        growth_score = (write_signal * pathway * free_capacity * growth_room).clamp_min(0.0)
        isolation_score = (write_signal * pathway * protected * active_room).clamp_min(0.0)
        if growth_score.shape != module.A.shape or isolation_score.shape != module.A.shape:
            raise RuntimeError(
                f"{module.name} rewire proposal shape mismatch: "
                f"growth={growth_score.shape}, isolation={isolation_score.shape}, A={module.A.shape}."
            )
        flat_score = (growth_score + isolation_score).reshape(-1)
        route_count += int(flat_score.numel())
        module_count += 1
        if flat_score.numel() <= 0:
            raise RuntimeError(f"{module.name} has no routes for rewire second-trial proposal.")
        if args.enable_rewire_second_trial:
            growth_mask = top_fraction_mask(
                growth_score,
                args.rewire_second_trial_route_fraction,
                args.rewire_second_trial_eps,
                name=f"{module.name}_rewire_growth",
            )
            isolation_mask = top_fraction_mask(
                isolation_score,
                args.rewire_second_trial_isolation_route_fraction,
                args.rewire_second_trial_eps,
                name=f"{module.name}_rewire_isolation",
            )
            growth_norm = growth_score / growth_score.max().clamp_min(args.rewire_second_trial_eps)
            isolation_norm = isolation_score / isolation_score.max().clamp_min(args.rewire_second_trial_eps)
            growth = (args.rewire_second_trial_growth * growth_mask * growth_norm * growth_room).clamp(0.0, 1.0)
            isolation = (
                args.rewire_second_trial_isolation * isolation_mask * isolation_norm * active_room
            ).clamp(0.0, 1.0)
        else:
            growth = torch.zeros_like(module.A)
            isolation = torch.zeros_like(module.A)
        delta = (growth - isolation).clamp(-1.0, 1.0)
        proposal[module.name] = delta.detach().clone()
        nonzero_delta = delta.abs() > args.rewire_second_trial_eps
        selected_count += float(nonzero_delta.to(dtype=torch.float32).sum().detach().cpu())
        score_mass += float(flat_score.sum().detach().cpu())
        growth_mass += float(growth.sum().detach().cpu())
        isolation_mass += float(isolation.sum().detach().cpu())
        delta_abs_mass += float(delta.abs().sum().detach().cpu())
        growth_max = max(growth_max, float(growth.max().detach().cpu()))
        isolation_max = max(isolation_max, float(isolation.max().detach().cpu()))
        delta_abs_max = max(delta_abs_max, float(delta.abs().max().detach().cpu()))
    if module_count <= 0 or route_count <= 0:
        raise RuntimeError("Rewire second-trial proposal saw no GCO modules.")
    return proposal, {
        "rewire_second_trial_module_count": float(module_count),
        "rewire_second_trial_route_count": float(route_count),
        "rewire_second_trial_proposal_score_mass": score_mass,
        "rewire_second_trial_growth_mass": growth_mass,
        "rewire_second_trial_isolation_mass": isolation_mass,
        "rewire_second_trial_delta_abs_mass": delta_abs_mass,
        "rewire_second_trial_growth_mean": growth_mass / float(route_count),
        "rewire_second_trial_isolation_mean": isolation_mass / float(route_count),
        "rewire_second_trial_delta_abs_mean": delta_abs_mass / float(route_count),
        "rewire_second_trial_growth_max": growth_max,
        "rewire_second_trial_isolation_max": isolation_max,
        "rewire_second_trial_delta_abs_max": delta_abs_max,
        "rewire_second_trial_growth_fraction": selected_count / float(route_count),
        "rewire_second_trial_delta_fraction": selected_count / float(route_count),
    }


@torch.no_grad()
def apply_rewire_second_trial_proposal(
    model: GCONativeTransformer,
    proposal: dict[str, torch.Tensor],
) -> None:
    modules = module_map(model)
    missing = set(modules).difference(proposal)
    unexpected = set(proposal).difference(modules)
    if missing or unexpected:
        raise RuntimeError(
            f"Rewire second-trial proposal mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    for name, growth in proposal.items():
        module = modules[name]
        if growth.shape != module.A.shape:
            raise RuntimeError(f"{name} proposal shape mismatch: {growth.shape} != {module.A.shape}.")
        module.A.add_(growth.to(device=module.A.device, dtype=module.A.dtype)).clamp_(0.0, 1.0)


@torch.no_grad()
def route_change_summary(
    model: GCONativeTransformer,
    anchors: dict[str, Any],
    before_snapshot: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    modules = module_map(model)
    anchor_modules = anchors["anchor_bank"]["modules"]
    total_routes = 0
    old_routes = 0
    reserve_routes = 0
    old_weight_delta_mass = 0.0
    reserve_weight_delta_mass = 0.0
    old_weight_delta_count = 0
    reserve_weight_delta_count = 0
    reserve_opened_count = 0
    reserve_opened_mass = 0.0
    old_topology_delta_mass = 0.0
    reserve_topology_delta_mass = 0.0
    for name, module in modules.items():
        if name not in anchor_modules:
            raise RuntimeError(f"Route summary missing anchor module {name!r}.")
        weight_key = f"{name}.W"
        topology_key = f"{name}.A"
        if weight_key not in before_snapshot or topology_key not in before_snapshot:
            raise RuntimeError(f"Snapshot missing keys for {name}: {weight_key}, {topology_key}.")
        old_topology = anchor_modules[name]["topology"].to(device=module.A.device, dtype=module.A.dtype)
        if old_topology.shape != module.A.shape:
            raise RuntimeError(f"{name} old topology shape mismatch: {old_topology.shape} != {module.A.shape}.")
        old_mask = old_topology >= args.route_active_threshold
        reserve_mask = ~old_mask
        before_weight = before_snapshot[weight_key].to(device=module.W.device, dtype=module.W.dtype)
        before_topology = before_snapshot[topology_key].to(device=module.A.device, dtype=module.A.dtype)
        weight_delta = (module.W.detach() - before_weight).abs()
        topology_delta = module.A.detach() - before_topology
        topology_delta_abs = topology_delta.abs()
        opened = (module.A.detach() > old_topology + args.route_delta_eps) & reserve_mask
        total_routes += int(module.A.numel())
        old_routes += int(old_mask.to(dtype=torch.long).sum().detach().cpu())
        reserve_routes += int(reserve_mask.to(dtype=torch.long).sum().detach().cpu())
        old_weight_delta_mass += float((weight_delta * old_mask.to(dtype=weight_delta.dtype)).sum().detach().cpu())
        reserve_weight_delta_mass += float((weight_delta * reserve_mask.to(dtype=weight_delta.dtype)).sum().detach().cpu())
        old_weight_delta_count += int(((weight_delta > args.route_delta_eps) & old_mask).to(dtype=torch.long).sum().detach().cpu())
        reserve_weight_delta_count += int(
            ((weight_delta > args.route_delta_eps) & reserve_mask).to(dtype=torch.long).sum().detach().cpu()
        )
        reserve_opened_count += int(opened.to(dtype=torch.long).sum().detach().cpu())
        reserve_opened_mass += float(((module.A.detach() - old_topology).clamp_min(0.0) * reserve_mask).sum().detach().cpu())
        old_topology_delta_mass += float((topology_delta_abs * old_mask.to(dtype=topology_delta_abs.dtype)).sum().detach().cpu())
        reserve_topology_delta_mass += float(
            (topology_delta_abs * reserve_mask.to(dtype=topology_delta_abs.dtype)).sum().detach().cpu()
        )
    if total_routes <= 0:
        raise RuntimeError("Route summary saw zero routes.")
    weight_total = old_weight_delta_mass + reserve_weight_delta_mass
    topology_total = old_topology_delta_mass + reserve_topology_delta_mass
    return {
        "route_total_count": float(total_routes),
        "old_route_fraction": float(old_routes) / float(total_routes),
        "reserve_route_fraction": float(reserve_routes) / float(total_routes),
        "old_route_weight_delta_mass": old_weight_delta_mass,
        "reserve_route_weight_delta_mass": reserve_weight_delta_mass,
        "old_route_weight_delta_fraction": old_weight_delta_mass / max(weight_total, args.route_delta_eps),
        "reserve_route_weight_delta_fraction": reserve_weight_delta_mass / max(weight_total, args.route_delta_eps),
        "old_route_update_fraction": float(old_weight_delta_count) / float(max(1, old_routes)),
        "reserve_route_update_fraction": float(reserve_weight_delta_count) / float(max(1, reserve_routes)),
        "opened_route_fraction": float(reserve_opened_count) / float(max(1, reserve_routes)),
        "opened_route_mass": reserve_opened_mass,
        "old_route_topology_delta_fraction": old_topology_delta_mass / max(topology_total, args.route_delta_eps),
        "reserve_route_topology_delta_fraction": reserve_topology_delta_mass / max(topology_total, args.route_delta_eps),
    }


def execute_gco_trial(
    model: GCONativeTransformer,
    *,
    step: int,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    old_inputs: torch.Tensor,
    old_targets: torch.Tensor,
    old_baseline: dict[str, float],
    old_before: dict[str, float],
    new_loss_before: float,
    anchors: dict[str, Any],
    args: argparse.Namespace,
    failed_write_signal: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, torch.Tensor], dict[str, float]]:
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(batch_inputs.to(device))
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_targets.to(device).reshape(-1))
    loss.backward()
    proposal, proposal_summary = capture_rewire_second_trial_proposal(model, args)
    stats = model.gco_step(step, developmental_maturity=args.control_phase_gate, failed_write_signal=failed_write_signal)
    summary = aggregate_stats(stats)

    new_loss_after = batch_loss(model, batch_inputs, batch_targets, device)
    old_after = evaluate_model(model, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    drift = anchor_drift(model, anchors, old_inputs, batch_size=args.eval_batch_size, device=device, top_routes=args.top_routes)

    new_loss_drop = new_loss_before - new_loss_after
    old_loss_damage = max(0.0, old_after["loss"] - old_before["loss"])
    old_baseline_loss_damage = max(0.0, old_after["loss"] - old_baseline["loss"])
    old_margin_damage = max(0.0, old_before["target_margin_mean"] - old_after["target_margin_mean"])
    utility = utility_from_measurements(
        new_loss_drop=new_loss_drop,
        old_loss_damage=old_loss_damage,
        old_baseline_loss_damage=old_baseline_loss_damage,
        old_margin_damage=old_margin_damage,
        anchor_pathway_drift_mean=drift["anchor_pathway_drift_mean"],
        anchor_activation_drift_mean=drift["anchor_activation_drift_mean"],
        args=args,
    )
    row = {
        "step": float(step),
        "utility": utility,
        "new_loss_before": new_loss_before,
        "new_loss_after": new_loss_after,
        "new_loss_drop": new_loss_drop,
        "old_loss_before": old_before["loss"],
        "old_loss_after": old_after["loss"],
        "old_loss_damage": old_loss_damage,
        "old_baseline_loss_damage": old_baseline_loss_damage,
        "old_accuracy_after": old_after["token_accuracy"],
        "old_margin_after": old_after["target_margin_mean"],
        "old_margin_damage": old_margin_damage,
        **drift,
        **proposal_summary,
        **summary,
    }
    for key, value in row.items():
        finite_float(key, value)
    return row, proposal, proposal_summary


def should_try_rewire_second_trial(row: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    if not args.enable_rewire_second_trial:
        return {
            "rewire_second_trial_triggered": 0.0,
            "rewire_second_trial_trigger_rejected": 0.0,
            "rewire_second_trial_trigger_old_loss": 0.0,
            "rewire_second_trial_trigger_old_margin": 0.0,
            "rewire_second_trial_trigger_anchor_path": 0.0,
            "rewire_second_trial_trigger_anchor_activation": 0.0,
        }
    if row["rewire_second_trial_delta_abs_mass"] <= args.rewire_second_trial_eps:
        return {
            "rewire_second_trial_triggered": 0.0,
            "rewire_second_trial_trigger_rejected": 0.0,
            "rewire_second_trial_trigger_old_loss": 0.0,
            "rewire_second_trial_trigger_old_margin": 0.0,
            "rewire_second_trial_trigger_anchor_path": 0.0,
            "rewire_second_trial_trigger_anchor_activation": 0.0,
        }
    if row["new_loss_drop"] < args.rewire_second_trial_min_new_loss_drop:
        return {
            "rewire_second_trial_triggered": 0.0,
            "rewire_second_trial_trigger_rejected": 0.0,
            "rewire_second_trial_trigger_old_loss": 0.0,
            "rewire_second_trial_trigger_old_margin": 0.0,
            "rewire_second_trial_trigger_anchor_path": 0.0,
            "rewire_second_trial_trigger_anchor_activation": 0.0,
        }
    rejected = row["utility"] < args.commit_threshold
    old_loss = row["old_loss_damage"] > args.rewire_second_trial_old_loss_damage
    old_margin = row["old_margin_damage"] > args.rewire_second_trial_old_margin_damage
    anchor_path = row["anchor_pathway_drift_mean"] > args.rewire_second_trial_anchor_pathway_drift
    anchor_activation = row["anchor_activation_drift_mean"] > args.rewire_second_trial_anchor_activation_drift
    triggered = rejected or old_loss or old_margin or anchor_path or anchor_activation
    return {
        "rewire_second_trial_triggered": float(1.0 if triggered else 0.0),
        "rewire_second_trial_trigger_rejected": float(1.0 if rejected else 0.0),
        "rewire_second_trial_trigger_old_loss": float(1.0 if old_loss else 0.0),
        "rewire_second_trial_trigger_old_margin": float(1.0 if old_margin else 0.0),
        "rewire_second_trial_trigger_anchor_path": float(1.0 if anchor_path else 0.0),
        "rewire_second_trial_trigger_anchor_activation": float(1.0 if anchor_activation else 0.0),
    }


def finalize_trial_row(
    row: dict[str, float],
    *,
    accepted: bool,
    selected_trial: float,
    failed_write_signal: float,
    trigger_summary: dict[str, float],
    direct_row: dict[str, float],
    rewire_row: dict[str, float] | None,
) -> dict[str, float]:
    next_failed = failed_write_signal
    out = dict(row)
    out["accepted"] = float(1.0 if accepted else 0.0)
    out["selected_trial"] = selected_trial
    out["failed_write_signal_after"] = next_failed
    out.update(trigger_summary)
    out["direct_trial_utility"] = direct_row["utility"]
    out["direct_trial_new_loss_drop"] = direct_row["new_loss_drop"]
    out["direct_trial_old_loss_damage"] = direct_row["old_loss_damage"]
    out["direct_trial_old_margin_damage"] = direct_row["old_margin_damage"]
    out["direct_trial_anchor_pathway_drift_mean"] = direct_row["anchor_pathway_drift_mean"]
    out["direct_trial_anchor_activation_drift_mean"] = direct_row["anchor_activation_drift_mean"]
    if rewire_row is None:
        out["rewire_second_trial_utility"] = 0.0
        out["rewire_second_trial_new_loss_drop"] = 0.0
        out["rewire_second_trial_old_loss_damage"] = 0.0
        out["rewire_second_trial_old_margin_damage"] = 0.0
        out["rewire_second_trial_anchor_pathway_drift_mean"] = 0.0
        out["rewire_second_trial_anchor_activation_drift_mean"] = 0.0
        out["rewire_second_trial_selected"] = 0.0
    else:
        out["rewire_second_trial_utility"] = rewire_row["utility"]
        out["rewire_second_trial_new_loss_drop"] = rewire_row["new_loss_drop"]
        out["rewire_second_trial_old_loss_damage"] = rewire_row["old_loss_damage"]
        out["rewire_second_trial_old_margin_damage"] = rewire_row["old_margin_damage"]
        out["rewire_second_trial_anchor_pathway_drift_mean"] = rewire_row["anchor_pathway_drift_mean"]
        out["rewire_second_trial_anchor_activation_drift_mean"] = rewire_row["anchor_activation_drift_mean"]
        out["rewire_second_trial_selected"] = float(1.0 if selected_trial == 2.0 else 0.0)
    for key, value in out.items():
        finite_float(key, value)
    return out


def train_step_with_virtual_trial(
    model: GCONativeTransformer,
    *,
    step: int,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    old_inputs: torch.Tensor,
    old_targets: torch.Tensor,
    old_baseline: dict[str, float],
    anchors: dict[str, Any],
    args: argparse.Namespace,
    failed_write_signal: float,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    snapshot = snapshot_state(model)
    old_before = evaluate_model(model, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    new_loss_before = batch_loss(model, batch_inputs, batch_targets, device)

    direct_row, proposal, direct_proposal_summary = execute_gco_trial(
        model,
        step=step,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        old_inputs=old_inputs,
        old_targets=old_targets,
        old_baseline=old_baseline,
        old_before=old_before,
        new_loss_before=new_loss_before,
        anchors=anchors,
        args=args,
        failed_write_signal=failed_write_signal,
        device=device,
    )
    direct_state = snapshot_state(model)
    trigger_summary = should_try_rewire_second_trial(direct_row, args)
    trigger_summary.update(
        {
            "rewire_second_trial_applied_growth_mass": direct_proposal_summary[
                "rewire_second_trial_growth_mass"
            ],
            "rewire_second_trial_applied_isolation_mass": direct_proposal_summary[
                "rewire_second_trial_isolation_mass"
            ],
            "rewire_second_trial_applied_delta_abs_mass": direct_proposal_summary[
                "rewire_second_trial_delta_abs_mass"
            ],
            "rewire_second_trial_applied_growth_mean": direct_proposal_summary[
                "rewire_second_trial_growth_mean"
            ],
            "rewire_second_trial_applied_isolation_mean": direct_proposal_summary[
                "rewire_second_trial_isolation_mean"
            ],
            "rewire_second_trial_applied_delta_abs_mean": direct_proposal_summary[
                "rewire_second_trial_delta_abs_mean"
            ],
            "rewire_second_trial_applied_growth_max": direct_proposal_summary[
                "rewire_second_trial_growth_max"
            ],
            "rewire_second_trial_applied_isolation_max": direct_proposal_summary[
                "rewire_second_trial_isolation_max"
            ],
            "rewire_second_trial_applied_delta_abs_max": direct_proposal_summary[
                "rewire_second_trial_delta_abs_max"
            ],
            "rewire_second_trial_applied_growth_fraction": direct_proposal_summary[
                "rewire_second_trial_growth_fraction"
            ],
            "rewire_second_trial_applied_delta_fraction": direct_proposal_summary[
                "rewire_second_trial_delta_fraction"
            ],
            "rewire_second_trial_applied_score_mass": direct_proposal_summary[
                "rewire_second_trial_proposal_score_mass"
            ],
        }
    )
    rewire_row: dict[str, float] | None = None

    if trigger_summary["rewire_second_trial_triggered"] > 0.5:
        restore_state(model, snapshot)
        apply_rewire_second_trial_proposal(model, proposal)
        rewire_row, _, _ = execute_gco_trial(
            model,
            step=step,
            batch_inputs=batch_inputs,
            batch_targets=batch_targets,
            old_inputs=old_inputs,
            old_targets=old_targets,
            old_baseline=old_baseline,
            old_before=old_before,
            new_loss_before=new_loss_before,
            anchors=anchors,
            args=args,
            failed_write_signal=failed_write_signal,
            device=device,
        )
        rewire_ok = rewire_row["utility"] >= args.commit_threshold
        direct_ok = direct_row["utility"] >= args.commit_threshold
        rewire_better = rewire_row["utility"] >= direct_row["utility"] + args.rewire_second_trial_min_utility_gain
        if rewire_ok and (rewire_better or not direct_ok):
            accepted = True
            selected_trial = 2.0
            selected_row = rewire_row
        elif direct_ok:
            restore_state(model, direct_state)
            accepted = True
            selected_trial = 1.0
            selected_row = direct_row
        else:
            restore_state(model, snapshot)
            accepted = False
            selected_trial = 0.0
            selected_row = direct_row if direct_row["utility"] >= rewire_row["utility"] else rewire_row
    else:
        accepted = direct_row["utility"] >= args.commit_threshold
        if not accepted:
            restore_state(model, snapshot)
            selected_trial = 0.0
        else:
            selected_trial = 1.0
        selected_row = direct_row

    next_failed = args.failed_write_beta * failed_write_signal + (1.0 - args.failed_write_beta) * (0.0 if accepted else 1.0)
    row = finalize_trial_row(
        selected_row,
        accepted=accepted,
        selected_trial=selected_trial,
        failed_write_signal=next_failed,
        trigger_summary=trigger_summary,
        direct_row=direct_row,
        rewire_row=rewire_row,
    )
    row.update(route_change_summary(model, anchors, snapshot, args))
    for key, value in row.items():
        finite_float(key, value)
    return row, next_failed


def validate_args(args: argparse.Namespace) -> None:
    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint does not exist: {args.base_checkpoint}")
    if not args.anchors_path.exists():
        raise FileNotFoundError(f"Anchors file does not exist: {args.anchors_path}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file does not exist: {args.tokenizer_path}")
    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {args.chunks_path}")
    nonnegative_int("new_word_start", args.new_word_start)
    positive_int("new_word_count", args.new_word_count)
    positive_int("max_windows", args.max_windows)
    positive_int("epochs", args.epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("top_routes", args.top_routes)
    positive_float("gco_lr", args.gco_lr)
    positive_float("max_step_norm", args.max_step_norm)
    positive_float("direct_write_ridge", args.direct_write_ridge)
    nonnegative_float("direct_write_protect", args.direct_write_protect)
    nonnegative_float("protect_old_route_floor", args.protect_old_route_floor)
    nonnegative_float("protect_collision_strength", args.protect_collision_strength)
    nonnegative_float("grow_lr", args.grow_lr)
    nonnegative_float("prune_lr", args.prune_lr)
    nonnegative_float("forget_lr", args.forget_lr)
    if not (0.0 <= args.control_phase_gate <= 1.0):
        raise ValueError(f"control_phase_gate must be in [0,1], got {args.control_phase_gate}.")
    nonnegative_float("anchor_protect_scale", args.anchor_protect_scale)
    nonnegative_float("old_loss_weight", args.old_loss_weight)
    nonnegative_float("old_baseline_loss_weight", args.old_baseline_loss_weight)
    nonnegative_float("old_margin_weight", args.old_margin_weight)
    nonnegative_float("anchor_pathway_weight", args.anchor_pathway_weight)
    nonnegative_float("anchor_activation_weight", args.anchor_activation_weight)
    finite_float("commit_threshold", args.commit_threshold)
    if not (0.0 <= args.failed_write_beta <= 1.0):
        raise ValueError(f"failed_write_beta must be in [0,1], got {args.failed_write_beta}.")
    bounded_float("rewire_second_trial_route_fraction", args.rewire_second_trial_route_fraction, 0.0, 1.0)
    bounded_float(
        "rewire_second_trial_isolation_route_fraction",
        args.rewire_second_trial_isolation_route_fraction,
        0.0,
        1.0,
    )
    nonnegative_float("rewire_second_trial_growth", args.rewire_second_trial_growth)
    nonnegative_float("rewire_second_trial_isolation", args.rewire_second_trial_isolation)
    nonnegative_float("rewire_second_trial_min_new_loss_drop", args.rewire_second_trial_min_new_loss_drop)
    nonnegative_float("rewire_second_trial_old_loss_damage", args.rewire_second_trial_old_loss_damage)
    nonnegative_float("rewire_second_trial_old_margin_damage", args.rewire_second_trial_old_margin_damage)
    nonnegative_float("rewire_second_trial_anchor_pathway_drift", args.rewire_second_trial_anchor_pathway_drift)
    nonnegative_float("rewire_second_trial_anchor_activation_drift", args.rewire_second_trial_anchor_activation_drift)
    finite_float("rewire_second_trial_min_utility_gain", args.rewire_second_trial_min_utility_gain)
    positive_float("rewire_second_trial_eps", args.rewire_second_trial_eps)
    bounded_float("route_active_threshold", args.route_active_threshold, 0.0, 1.0)
    positive_float("route_delta_eps", args.route_delta_eps)
    if (
        args.enable_rewire_second_trial
        and args.rewire_second_trial_growth > 0.0
        and args.rewire_second_trial_route_fraction <= 0.0
    ):
        raise ValueError("rewire_second_trial_route_fraction must be > 0 when second-trial rewiring is enabled.")
    if args.enable_rewire_second_trial and args.rewire_second_trial_growth <= 0.0:
        if args.rewire_second_trial_isolation <= 0.0:
            raise ValueError(
                "At least one of rewire_second_trial_growth or rewire_second_trial_isolation must be > 0 "
                "when second-trial rewiring is enabled."
            )
    if args.enable_rewire_second_trial and args.rewire_second_trial_isolation > 0.0:
        if args.rewire_second_trial_isolation_route_fraction <= 0.0:
            raise ValueError(
                "rewire_second_trial_isolation_route_fraction must be > 0 when isolation rewiring is enabled."
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    anchors = torch.load(args.anchors_path, map_location="cpu", weights_only=False)
    cfg = build_cl_config(checkpoint, args)
    model = instantiate_model(checkpoint, cfg, device)
    seed_summary = seed_old_geometry(
        model,
        anchors,
        protect_scale=args.anchor_protect_scale,
        normalize=args.normalize_anchor_protection,
        device=device,
    )

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.new_chunk_index < 0 or args.new_chunk_index >= len(chunks):
        raise ValueError(f"new_chunk_index={args.new_chunk_index} outside chunk count {len(chunks)}.")
    new_text = word_span(str(chunks[args.new_chunk_index]["text"]), args.new_word_start, args.new_word_count)
    token_ids = tokenizer.encode(new_text).ids
    model_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    new_inputs, new_targets = build_lm_windows(
        token_ids,
        seq_len=model_seq_len,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    old_inputs = anchors["probe_inputs"].to(dtype=torch.long)
    old_targets = anchors["probe_targets"].to(dtype=torch.long)
    old_baseline = anchors["final_metrics"]
    old_initial = evaluate_model(model, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    new_initial = evaluate_model(model, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
    initial_drift = anchor_drift(model, anchors, old_inputs, batch_size=args.eval_batch_size, device=device, top_routes=args.top_routes)

    print("GCO TINY CL REASONED WRITE")
    print("=" * 112)
    print(
        f"device={device} base={args.base_checkpoint} anchors={args.anchors_path} "
        f"ctrl={args.control_phase_gate:g} gco_lr={args.gco_lr:g}"
    )
    print(
        f"old_loss={old_initial['loss']:.6f} old_acc={old_initial['token_accuracy']:.4f} "
        f"new_loss={new_initial['loss']:.6f} new_words=[{args.new_word_start},{args.new_word_start + args.new_word_count}) "
        f"new_windows={new_inputs.shape[0]}"
    )
    print(
        "seeded modules={seeded_module_count:.0f} meanP={mean_seeded_protection:.4f} maxP={max_seeded_protection:.4f} "
        "initial_anchor_drift path={anchor_pathway_drift_mean:.4g} act={anchor_activation_drift_mean:.4g}".format(
            **seed_summary,
            **initial_drift,
        )
    )

    trace: list[dict[str, float]] = []
    failed_write_signal = 0.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(new_inputs.shape[0])
        accepted_count = 0
        rejected_count = 0
        rewire_attempted_count = 0
        rewire_selected_count = 0
        pbar = tqdm(range(0, new_inputs.shape[0], args.batch_size), desc=f"cl epoch {epoch}/{args.epochs}")
        for start in pbar:
            global_step += 1
            indices = permutation[start : start + args.batch_size]
            row, failed_write_signal = train_step_with_virtual_trial(
                model,
                step=global_step,
                batch_inputs=new_inputs[indices],
                batch_targets=new_targets[indices],
                old_inputs=old_inputs,
                old_targets=old_targets,
                old_baseline=old_baseline,
                anchors=anchors,
                args=args,
                failed_write_signal=failed_write_signal,
                device=device,
            )
            trace.append(row)
            if row["accepted"] > 0.5:
                accepted_count += 1
            else:
                rejected_count += 1
            if row["rewire_second_trial_triggered"] > 0.5:
                rewire_attempted_count += 1
            if row["rewire_second_trial_selected"] > 0.5:
                rewire_selected_count += 1
            pbar.set_postfix(
                {
                    "U": f"{row['utility']:+.3g}",
                    "acc": accepted_count,
                    "rej": rejected_count,
                    "rw2": rewire_selected_count,
                    "new": f"{row['new_loss_after']:.3f}",
                    "old": f"{row['old_loss_after']:.3f}",
                }
            )
        epoch_old = evaluate_model(model, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
        epoch_new = evaluate_model(model, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
        epoch_drift = anchor_drift(model, anchors, old_inputs, batch_size=args.eval_batch_size, device=device, top_routes=args.top_routes)
        print(
            "epoch={:4d} accepted={} rejected={} rw2_attempt={} rw2_selected={} old_loss={:.6f} new_loss={:.6f} "
            "old_acc={:.4f} new_acc={:.4f} path_drift={:.4g} act_drift={:.4g}".format(
                epoch,
                accepted_count,
                rejected_count,
                rewire_attempted_count,
                rewire_selected_count,
                epoch_old["loss"],
                epoch_new["loss"],
                epoch_old["token_accuracy"],
                epoch_new["token_accuracy"],
                epoch_drift["anchor_pathway_drift_mean"],
                epoch_drift["anchor_activation_drift_mean"],
            )
        )

    final_old = evaluate_model(model, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    final_new = evaluate_model(model, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
    final_drift = anchor_drift(model, anchors, old_inputs, batch_size=args.eval_batch_size, device=device, top_routes=args.top_routes)
    accepted_total = sum(1 for row in trace if row["accepted"] > 0.5)
    rejected_total = len(trace) - accepted_total
    rewire_attempted_total = sum(1 for row in trace if row["rewire_second_trial_triggered"] > 0.5)
    rewire_selected_total = sum(1 for row in trace if row["rewire_second_trial_selected"] > 0.5)
    def trace_mean(key: str) -> float:
        values = [float(row[key]) for row in trace if key in row]
        if not values:
            raise RuntimeError(f"Trace is missing key {key!r}.")
        return sum(values) / float(len(values))

    route_summary = {
        "old_route_weight_delta_fraction_mean": trace_mean("old_route_weight_delta_fraction"),
        "reserve_route_weight_delta_fraction_mean": trace_mean("reserve_route_weight_delta_fraction"),
        "old_route_update_fraction_mean": trace_mean("old_route_update_fraction"),
        "reserve_route_update_fraction_mean": trace_mean("reserve_route_update_fraction"),
        "opened_route_fraction_mean": trace_mean("opened_route_fraction"),
        "opened_route_mass_mean": trace_mean("opened_route_mass"),
        "old_route_topology_delta_fraction_mean": trace_mean("old_route_topology_delta_fraction"),
        "reserve_route_topology_delta_fraction_mean": trace_mean("reserve_route_topology_delta_fraction"),
    }

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "base_checkpoint": str(args.base_checkpoint),
            "anchors_path": str(args.anchors_path),
            "model_state_dict": model.state_dict(),
            "native_gco_config": asdict(cfg),
            "model_config": checkpoint["model_config"],
            "source": {
                "new_chunk_index": args.new_chunk_index,
                "new_chunk_id": str(chunks[args.new_chunk_index]["chunk_id"]),
                "new_word_start": args.new_word_start,
                "new_word_count": args.new_word_count,
                "new_text": new_text,
                "new_token_ids": token_ids,
            },
            "old_final_metrics": final_old,
            "new_final_metrics": final_new,
            "final_anchor_drift": final_drift,
        },
        args.output_checkpoint,
    )

    result = {
        "question": "Can GCO learn new text from a stable tiny base while preserving old anchored geometry?",
        "base_checkpoint": str(args.base_checkpoint),
        "anchors_path": str(args.anchors_path),
        "output_checkpoint": str(args.output_checkpoint),
        "seed_summary": seed_summary,
        "old_initial": old_initial,
        "new_initial": new_initial,
        "initial_anchor_drift": initial_drift,
        "old_final": final_old,
        "new_final": final_new,
        "final_anchor_drift": final_drift,
        "accepted_total": accepted_total,
        "rejected_total": rejected_total,
        "rewire_second_trial_attempted_total": rewire_attempted_total,
        "rewire_second_trial_selected_total": rewire_selected_total,
        "route_summary": route_summary,
        "trace": trace,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nTINY CL REASONED WRITE SUMMARY")
    print("=" * 112)
    print(
        "old_loss {:.6f}->{:.6f} old_acc {:.4f}->{:.4f} old_margin {:.4f}->{:.4f}".format(
            old_initial["loss"],
            final_old["loss"],
            old_initial["token_accuracy"],
            final_old["token_accuracy"],
            old_initial["target_margin_mean"],
            final_old["target_margin_mean"],
        )
    )
    print(
        "new_loss {:.6f}->{:.6f} new_acc {:.4f}->{:.4f} accepted={} rejected={} rw2_attempted={} rw2_selected={}".format(
            new_initial["loss"],
            final_new["loss"],
            new_initial["token_accuracy"],
            final_new["token_accuracy"],
            accepted_total,
            rejected_total,
            rewire_attempted_total,
            rewire_selected_total,
        )
    )
    print(
        "anchor_drift path={:.4g}/{:.4g} act={:.4g}/{:.4g} top={:.4g}/{:.4g}".format(
            final_drift["anchor_pathway_drift_mean"],
            final_drift["anchor_pathway_drift_max"],
            final_drift["anchor_activation_drift_mean"],
            final_drift["anchor_activation_drift_max"],
            final_drift["anchor_top_route_abs_delta_mean"],
            final_drift["anchor_top_route_abs_delta_max"],
        )
    )
    print(
        "routes W_old/reserve={:.3f}/{:.3f} upd_old/reserve={:.4f}/{:.4f} opened={:.4f} A_old/reserve={:.3f}/{:.3f}".format(
            route_summary["old_route_weight_delta_fraction_mean"],
            route_summary["reserve_route_weight_delta_fraction_mean"],
            route_summary["old_route_update_fraction_mean"],
            route_summary["reserve_route_update_fraction_mean"],
            route_summary["opened_route_fraction_mean"],
            route_summary["old_route_topology_delta_fraction_mean"],
            route_summary["reserve_route_topology_delta_fraction_mean"],
        )
    )
    print(f"wrote_checkpoint={args.output_checkpoint}")
    print(f"wrote_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument("--anchors-path", type=Path, default=Path("model/analysis/gco-tiny-cl-base-100w-anchors-seed0.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--output-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-reasoned-write-seed0.pt"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-cl-reasoned-write-seed0.json"))
    parser.add_argument("--new-chunk-index", type=int, default=0)
    parser.add_argument("--new-word-start", type=int, default=100)
    parser.add_argument("--new-word-count", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--control-phase-gate", type=float, default=1.0)
    parser.add_argument("--gco-lr", type=float, default=0.05)
    parser.add_argument("--max-step-norm", type=float, default=1.0)
    parser.add_argument("--direct-write-ridge", type=float, default=1e-2)
    parser.add_argument("--direct-write-protect", type=float, default=1e-1)
    parser.add_argument("--protect-old-route-floor", type=float, default=0.5)
    parser.add_argument("--protect-collision-strength", type=float, default=1.0)
    parser.add_argument("--grow-lr", type=float, default=1e-2)
    parser.add_argument("--prune-lr", type=float, default=0.0)
    parser.add_argument("--forget-lr", type=float, default=0.0)
    parser.add_argument("--anchor-protect-scale", type=float, default=1.0)
    parser.add_argument("--normalize-anchor-protection", action="store_true")
    parser.add_argument("--old-loss-weight", type=float, default=5.0)
    parser.add_argument("--old-baseline-loss-weight", type=float, default=5.0)
    parser.add_argument("--old-margin-weight", type=float, default=0.1)
    parser.add_argument("--anchor-pathway-weight", type=float, default=0.1)
    parser.add_argument("--anchor-activation-weight", type=float, default=0.1)
    parser.add_argument("--commit-threshold", type=float, default=0.0)
    parser.add_argument("--failed-write-beta", type=float, default=0.95)
    parser.add_argument("--top-routes", type=int, default=16)
    parser.add_argument("--enable-rewire-second-trial", action="store_true")
    parser.add_argument("--rewire-second-trial-route-fraction", type=float, default=0.01)
    parser.add_argument("--rewire-second-trial-growth", type=float, default=0.25)
    parser.add_argument("--rewire-second-trial-isolation-route-fraction", type=float, default=0.0)
    parser.add_argument("--rewire-second-trial-isolation", type=float, default=0.0)
    parser.add_argument("--rewire-second-trial-min-new-loss-drop", type=float, default=0.0)
    parser.add_argument("--rewire-second-trial-old-loss-damage", type=float, default=1e-5)
    parser.add_argument("--rewire-second-trial-old-margin-damage", type=float, default=1e-3)
    parser.add_argument("--rewire-second-trial-anchor-pathway-drift", type=float, default=0.01)
    parser.add_argument("--rewire-second-trial-anchor-activation-drift", type=float, default=0.05)
    parser.add_argument("--rewire-second-trial-min-utility-gain", type=float, default=0.0)
    parser.add_argument("--rewire-second-trial-eps", type=float, default=1e-12)
    parser.add_argument("--route-active-threshold", type=float, default=0.5)
    parser.add_argument("--route-delta-eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
