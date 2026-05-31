#!/usr/bin/env python3
"""Evidence-driven GFO stream experiment.

This is the first integrated version of the full GFO plan on a small neural
network. It is intentionally still mathematical and synthetic:

    two-layer linear neural net
    activation/output anchor memory
    pending concept evidence
    create / merge / fuse decisions
    tolerance-aware safe writes
    lineage-aware forgetting metrics

No external retrieval system, no symbolic task router, no transformer yet.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from gfo_linear_activation_constraints import (
    Batch,
    ModelShape,
    affine_project,
    anchor_jacobian,
    forward,
    init_flat,
    layer_representation,
    loss_grad,
    mse,
    randn,
    resolve_device,
    resolve_dtype,
    set_seed,
)


@dataclass
class ConceptAnchor:
    concept_id: str
    lineage_id: str
    q: torch.Tensor
    target_y: torch.Tensor
    original_y: torch.Tensor
    z_hidden: torch.Tensor
    z_output: torch.Tensor
    importance: float
    tolerance: float
    count: float
    stability: float
    last_seen: int
    status: str = "active"
    transformed: bool = False


@dataclass
class PendingConcept:
    pending_id: str
    centroid_x: torch.Tensor
    target_y: torch.Tensor
    z_hidden: torch.Tensor
    z_output: torch.Tensor
    count: int
    variance: float
    pressure: float
    last_seen: int
    committed_anchor_id: Optional[str] = None
    committed_evidence_target_y: Optional[torch.Tensor] = None
    committed_write_pressure: float = 0.0
    last_action: str = "pending"


@dataclass(frozen=True)
class StreamEvent:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    expected_kind: str


@dataclass(frozen=True)
class ProjectionStats:
    candidate_count: int
    selected_count: int
    mean_violation: float
    max_violation: float
    candidate_ids: Tuple[str, ...]
    selected_ids: Tuple[str, ...]


@dataclass
class RunState:
    theta: torch.Tensor
    anchors: List[ConceptAnchor]
    pending: List[PendingConcept] = field(default_factory=list)
    action_counts: Dict[str, int] = field(default_factory=dict)
    writes: int = 0
    safe_steps: int = 0
    skipped: int = 0
    write_diagnostics: List[Dict[str, object]] = field(default_factory=list)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.dot(a, b) / ((torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)) + eps)


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.linalg.vector_norm(v) + eps)


def sigmoid_scalar(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def tensor_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().detach().cpu())


def make_world(
    shape: ModelShape,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Batch, Dict[str, torch.Tensor], List[str]]:
    q_base = torch.linalg.qr(randn((shape.input_dim, shape.input_dim), device=device, dtype=dtype))[0].T
    old_ids = ["old_red", "old_blue", "old_green"]
    x_map = {
        "old_red": q_base[0],
        "old_blue": q_base[1],
        "old_green": q_base[2],
        "new_safe": q_base[3],
        "noise": normalize(0.6 * q_base[4] + 0.4 * q_base[5]),
    }
    y_map = {
        "old_red": randn((shape.output_dim,), device=device, dtype=dtype),
        "old_blue": randn((shape.output_dim,), device=device, dtype=dtype),
        "old_green": randn((shape.output_dim,), device=device, dtype=dtype),
        "new_safe": randn((shape.output_dim,), device=device, dtype=dtype),
        "noise": randn((shape.output_dim,), device=device, dtype=dtype),
        "blue_conflict": randn((shape.output_dim,), device=device, dtype=dtype),
    }
    train = Batch(
        x=torch.stack([x_map[name] for name in old_ids], dim=0),
        y=torch.stack([y_map[name] for name in old_ids], dim=0),
    )
    values = {**{f"x_{k}": v for k, v in x_map.items()}, **{f"y_{k}": v for k, v in y_map.items()}}
    return train, values, old_ids


def make_stream(values: Dict[str, torch.Tensor]) -> List[StreamEvent]:
    events: List[StreamEvent] = []
    events.append(StreamEvent("one_shot_noise", values["x_noise"], values["y_noise"], "noise"))
    for idx in range(5):
        events.append(StreamEvent(f"new_safe_{idx}", values["x_new_safe"], values["y_new_safe"], "new"))
    for idx in range(3):
        events.append(StreamEvent(f"familiar_red_{idx}", values["x_old_red"], values["y_old_red"], "familiar"))
    for idx in range(7):
        events.append(StreamEvent(f"blue_conflict_{idx}", values["x_old_blue"], values["y_blue_conflict"], "conflict"))
    return events


def train_task1(
    theta: torch.Tensor,
    shape: ModelShape,
    batch: Batch,
    *,
    steps: int,
    lr: float,
) -> torch.Tensor:
    current = theta.detach().clone()
    for _ in range(steps):
        grad = loss_grad(current, shape, batch, freeze_output=False)
        current = current - lr * grad
    return current.detach()


def current_reps(theta: torch.Tensor, shape: ModelShape, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_b = x.unsqueeze(0)
    hidden = layer_representation(theta, shape, x_b, "hidden", 1e-8, "full", 1.0).reshape(-1).detach()
    output = layer_representation(theta, shape, x_b, "output", 1e-8, "full", 1.0).reshape(-1).detach()
    return hidden, output


def build_initial_anchors(
    theta: torch.Tensor,
    shape: ModelShape,
    train: Batch,
    old_ids: Sequence[str],
    *,
    tolerance: float,
) -> List[ConceptAnchor]:
    anchors = []
    for idx, concept_id in enumerate(old_ids):
        q = train.x[idx].detach().clone()
        target_y = train.y[idx].detach().clone()
        z_hidden, z_output = current_reps(theta, shape, q)
        anchors.append(
            ConceptAnchor(
                concept_id=concept_id,
                lineage_id=concept_id,
                q=q,
                target_y=target_y,
                original_y=target_y.clone(),
                z_hidden=z_hidden,
                z_output=z_output,
                importance=1.0,
                tolerance=tolerance,
                count=1.0,
                stability=1.0,
                last_seen=0,
            )
        )
    return anchors


def update_pending(
    pending: List[PendingConcept],
    theta: torch.Tensor,
    shape: ModelShape,
    event: StreamEvent,
    step: int,
    *,
    cluster_threshold: float,
    pressure_tau: float,
    variance_tau: float,
) -> PendingConcept:
    z_hidden, z_output = current_reps(theta, shape, event.x)
    best: Optional[PendingConcept] = None
    best_sim = -1.0
    for candidate in pending:
        sim = float(cosine(candidate.centroid_x, event.x).detach().cpu())
        if sim > best_sim:
            best = candidate
            best_sim = sim

    if best is None or best_sim < cluster_threshold:
        item = PendingConcept(
            pending_id=f"pending_{len(pending)}",
            centroid_x=event.x.detach().clone(),
            target_y=event.y.detach().clone(),
            z_hidden=z_hidden,
            z_output=z_output,
            count=1,
            variance=0.0,
            pressure=0.0,
            last_seen=step,
        )
        pending.append(item)
        best = item
    else:
        beta = 1.0 / float(best.count + 1)
        best.centroid_x = normalize((1.0 - beta) * best.centroid_x + beta * event.x.detach())
        best.target_y = (1.0 - beta) * best.target_y + beta * event.y.detach()
        best.z_hidden = (1.0 - beta) * best.z_hidden + beta * z_hidden
        best.z_output = (1.0 - beta) * best.z_output + beta * z_output
        drift = tensor_mse(z_output, best.z_output)
        best.variance = 0.9 * best.variance + 0.1 * drift
        best.count += 1
        best.last_seen = step

    pred, _ = forward(theta, shape, event.x.unsqueeze(0))
    err = tensor_mse(pred.reshape(-1), event.y)
    frequency = 1.0 - math.exp(-best.count / pressure_tau)
    consistency = math.exp(-best.variance / variance_tau)
    error_score = sigmoid_scalar((err - 0.02) / 0.05)
    best.pressure = frequency * consistency * error_score
    return best


def nearest_anchor(anchors: Sequence[ConceptAnchor], x: torch.Tensor) -> Tuple[Optional[ConceptAnchor], float]:
    best = None
    best_sim = -1.0
    for anchor in anchors:
        if anchor.status != "active":
            continue
        sim = float(cosine(anchor.q, x).detach().cpu())
        if sim > best_sim:
            best = anchor
            best_sim = sim
    return best, best_sim


def find_anchor_by_id(anchors: Sequence[ConceptAnchor], concept_id: str) -> Optional[ConceptAnchor]:
    for anchor in anchors:
        if anchor.concept_id == concept_id and anchor.status == "active":
            return anchor
    return None


def decide_action(
    pending: PendingConcept,
    anchors: Sequence[ConceptAnchor],
    *,
    pressure_threshold: float,
    merge_similarity: float,
    conflict_similarity: float,
    conflict_threshold: float,
    committed_target_change_threshold: float,
) -> Tuple[str, Optional[ConceptAnchor]]:
    if pending.committed_anchor_id is not None:
        committed = find_anchor_by_id(anchors, pending.committed_anchor_id)
        if committed is None:
            raise RuntimeError(f"Pending concept references missing anchor: {pending.committed_anchor_id}")
        if pending.committed_evidence_target_y is None:
            raise RuntimeError(f"Pending concept {pending.pending_id} has no committed evidence target.")
        target_change = tensor_mse(pending.target_y, pending.committed_evidence_target_y)
        if target_change <= committed_target_change_threshold:
            return "reinforce", committed
    if pending.pressure < pressure_threshold:
        return "ignore", None

    nearest, sim = nearest_anchor(anchors, pending.centroid_x)
    if nearest is None:
        return "create", None
    conflict = tensor_mse(nearest.target_y, pending.target_y)
    if sim >= merge_similarity and conflict < conflict_threshold:
        return "merge", nearest
    if sim >= conflict_similarity and conflict >= conflict_threshold:
        return "fuse", nearest
    return "create", None


def protected_projection(
    theta: torch.Tensor,
    shape: ModelShape,
    delta_raw: torch.Tensor,
    anchors: Sequence[ConceptAnchor],
    *,
    exclude_lineage: Optional[str],
    protect_top_k: int,
    damping: float,
    tolerance_tiny: float,
) -> Tuple[torch.Tensor, ProjectionStats]:
    candidates: List[Tuple[float, float, ConceptAnchor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for anchor in anchors:
        if anchor.status != "active":
            continue
        if exclude_lineage is not None and anchor.lineage_id == exclude_lineage:
            continue
        current = layer_representation(theta, shape, anchor.q.unsqueeze(0), "output", 1e-8, "full", 1.0).reshape(-1)
        residual = current - anchor.z_output
        jac = anchor_jacobian(theta, shape, anchor.q, "output", 1e-8, "full", 1.0)
        predicted = residual + jac @ delta_raw
        violation = max(0.0, float((torch.dot(predicted, predicted) - anchor.tolerance).detach().cpu()))
        if violation > 0.0:
            candidates.append((anchor.importance * violation, violation, anchor, jac, residual, predicted))

    candidates.sort(key=lambda row: row[0], reverse=True)
    candidate_count = len(candidates)
    candidate_ids = tuple(row[2].concept_id for row in candidates)
    if protect_top_k > 0:
        candidates = candidates[:protect_top_k]

    rows: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    selected_violations = [row[1] for row in candidates]
    stats = ProjectionStats(
        candidate_count=candidate_count,
        selected_count=len(candidates),
        mean_violation=float(sum(selected_violations) / len(selected_violations)) if selected_violations else 0.0,
        max_violation=float(max(selected_violations)) if selected_violations else 0.0,
        candidate_ids=candidate_ids,
        selected_ids=tuple(row[2].concept_id for row in candidates),
    )
    for _, _, anchor, jac, residual, predicted in candidates:
        predicted_norm = torch.sqrt(torch.dot(predicted, predicted))
        if predicted_norm < tolerance_tiny:
            boundary = torch.zeros_like(predicted)
        else:
            boundary = math.sqrt(anchor.tolerance) * predicted / (predicted_norm + tolerance_tiny)
        rows.append(jac)
        targets.append(boundary - residual)

    if not rows:
        return delta_raw, stats
    a = torch.cat(rows, dim=0)
    b = torch.cat(targets, dim=0)
    return affine_project(delta_raw, a, b, damping), stats


def output_drift_stats(
    theta: torch.Tensor,
    shape: ModelShape,
    anchors: Sequence[ConceptAnchor],
) -> Dict[str, float]:
    active = [anchor for anchor in anchors if anchor.status == "active"]
    if not active:
        return {"mean_anchor_output_drift": 0.0, "max_anchor_output_drift": 0.0}
    drifts = []
    for anchor in active:
        current = layer_representation(theta, shape, anchor.q.unsqueeze(0), "output", 1e-8, "full", 1.0).reshape(-1)
        drift = torch.dot(current - anchor.z_output, current - anchor.z_output)
        drifts.append(float(drift.detach().cpu()))
    return {
        "mean_anchor_output_drift": float(sum(drifts) / len(drifts)),
        "max_anchor_output_drift": float(max(drifts)),
    }


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_write(
    state: RunState,
    shape: ModelShape,
    event: StreamEvent,
    pending: PendingConcept,
    action: str,
    nearest: Optional[ConceptAnchor],
    step: int,
    *,
    original_old_ids: Sequence[str],
    write_lr: float,
    protect_top_k: int,
    damping: float,
    tolerance: float,
    tolerance_tiny: float,
    write_steps: int,
    target_loss_threshold: float,
    record_diagnostics: bool,
) -> None:
    state.action_counts[action] = state.action_counts.get(action, 0) + 1
    if action == "ignore":
        state.skipped += 1
        pending.last_action = "ignore"
        if record_diagnostics:
            event_batch = Batch(x=event.x.unsqueeze(0), y=event.y.unsqueeze(0))
            snapshot = evaluate_state(state.theta, shape, state.anchors, original_old_ids=original_old_ids)
            state.write_diagnostics.append(
                {
                    "step": step,
                    "event": event.name,
                    "expected_kind": event.expected_kind,
                    "pending_id": pending.pending_id,
                    "action": action,
                    "pressure": float(pending.pressure),
                    "write_pressure": 0.0,
                    "target_loss_before": float(mse(state.theta, shape, event_batch).detach().cpu()),
                    "target_loss_after": float(mse(state.theta, shape, event_batch).detach().cpu()),
                    "steps_applied": 0,
                    "candidate_protected_mean": 0.0,
                    "selected_protected_mean": 0.0,
                    "raw_update_norm_mean": 0.0,
                    "safe_update_norm_mean": 0.0,
                    "projection_ratio_mean": 0.0,
                    "mean_projection_violation": 0.0,
                    "max_projection_violation": 0.0,
                    "candidate_protected_ids": [],
                    "selected_protected_ids": [],
                    **snapshot,
                    **output_drift_stats(state.theta, shape, state.anchors),
                    "committed_anchor_id": pending.committed_anchor_id,
                    "last_action": pending.last_action,
                }
            )
        return

    exclude_lineage = None
    target_y = pending.target_y.detach().clone()
    target_x = pending.centroid_x.detach().clone()

    if action == "reinforce":
        if nearest is None:
            raise RuntimeError("reinforce action requires a committed anchor.")
        exclude_lineage = nearest.lineage_id
        target_y = nearest.target_y.detach().clone()
        target_x = nearest.q.detach().clone()
    elif action == "merge":
        if nearest is None:
            raise RuntimeError("merge action requires a nearest anchor.")
        exclude_lineage = nearest.lineage_id
        beta = pending.pressure / (nearest.count + pending.pressure + 1e-8)
        nearest.target_y = (1.0 - beta) * nearest.target_y + beta * pending.target_y
        target_y = nearest.target_y.detach().clone()
        target_x = nearest.q.detach().clone()
    elif action == "fuse":
        if nearest is None:
            raise RuntimeError("fuse action requires a nearest anchor.")
        exclude_lineage = nearest.lineage_id
        old_weight = nearest.importance * nearest.stability * max(1.0, nearest.count)
        new_weight = pending.pressure * max(1.0, float(pending.count))
        nearest.target_y = (old_weight * nearest.target_y + new_weight * pending.target_y) / (old_weight + new_weight)
        nearest.transformed = True
        target_y = nearest.target_y.detach().clone()
        target_x = nearest.q.detach().clone()
    elif action != "create":
        raise ValueError(f"Unknown write action: {action}")

    batch = Batch(x=target_x.unsqueeze(0), y=target_y.unsqueeze(0))
    write_pressure = pending.pressure
    if action == "reinforce":
        write_pressure = max(write_pressure, pending.committed_write_pressure)
    steps_applied = 0
    candidate_counts: List[float] = []
    selected_counts: List[float] = []
    raw_norms: List[float] = []
    safe_norms: List[float] = []
    projection_ratios: List[float] = []
    mean_violations: List[float] = []
    max_violations: List[float] = []
    candidate_ids: List[str] = []
    selected_ids: List[str] = []
    target_loss_before = float(mse(state.theta, shape, batch).detach().cpu())
    for _ in range(write_steps):
        current_loss = float(mse(state.theta, shape, batch).detach().cpu())
        if current_loss <= target_loss_threshold:
            break
        grad = loss_grad(state.theta, shape, batch, freeze_output=False)
        delta_raw = -write_lr * write_pressure * grad
        delta_safe, projection_stats = protected_projection(
            state.theta,
            shape,
            delta_raw,
            state.anchors,
            exclude_lineage=exclude_lineage,
            protect_top_k=protect_top_k,
            damping=damping,
            tolerance_tiny=tolerance_tiny,
        )
        raw_norm = float(torch.linalg.vector_norm(delta_raw).detach().cpu())
        safe_norm = float(torch.linalg.vector_norm(delta_safe).detach().cpu())
        raw_norms.append(raw_norm)
        safe_norms.append(safe_norm)
        projection_ratios.append(safe_norm / (raw_norm + tolerance_tiny))
        candidate_counts.append(float(projection_stats.candidate_count))
        selected_counts.append(float(projection_stats.selected_count))
        candidate_ids.extend(projection_stats.candidate_ids)
        selected_ids.extend(projection_stats.selected_ids)
        mean_violations.append(projection_stats.mean_violation)
        max_violations.append(projection_stats.max_violation)
        state.theta = (state.theta + delta_safe).detach()
        steps_applied += 1

    state.writes += 1
    state.safe_steps += steps_applied

    if action == "create":
        concept_id = f"concept_{len(state.anchors)}"
        z_hidden, z_output = current_reps(state.theta, shape, target_x)
        state.anchors.append(
            ConceptAnchor(
                concept_id=concept_id,
                lineage_id=concept_id,
                q=target_x.detach().clone(),
                target_y=target_y.detach().clone(),
                original_y=target_y.detach().clone(),
                z_hidden=z_hidden,
                z_output=z_output,
                importance=float(max(0.1, pending.pressure)),
                tolerance=tolerance,
                count=float(pending.count),
                stability=math.exp(-pending.variance),
                last_seen=step,
            )
        )
        pending.committed_anchor_id = concept_id
        pending.committed_evidence_target_y = pending.target_y.detach().clone()
        pending.committed_write_pressure = max(pending.committed_write_pressure, pending.pressure)
    else:
        if nearest is None:
            raise RuntimeError(f"{action} action requires an anchor update.")
        z_hidden, z_output = current_reps(state.theta, shape, nearest.q)
        nearest.z_hidden = z_hidden
        nearest.z_output = z_output
        nearest.count += pending.pressure
        nearest.stability = min(1.0, 0.9 * nearest.stability + 0.1 * math.exp(-pending.variance))
        nearest.last_seen = step
        pending.committed_anchor_id = nearest.concept_id
        if action in {"merge", "fuse"} or pending.committed_evidence_target_y is None:
            pending.committed_evidence_target_y = pending.target_y.detach().clone()
        pending.committed_write_pressure = max(pending.committed_write_pressure, pending.pressure)

    avg_protected = mean_or_zero(selected_counts)
    pending.last_action = f"{action}:steps={steps_applied}:protected={avg_protected:.2f}"
    if record_diagnostics:
        snapshot = evaluate_state(state.theta, shape, state.anchors, original_old_ids=original_old_ids)
        state.write_diagnostics.append(
            {
                "step": step,
                "event": event.name,
                "expected_kind": event.expected_kind,
                "pending_id": pending.pending_id,
                "action": action,
                "pressure": float(pending.pressure),
                "write_pressure": float(write_pressure),
                "target_loss_before": target_loss_before,
                "target_loss_after": float(mse(state.theta, shape, batch).detach().cpu()),
                "steps_applied": steps_applied,
                "candidate_protected_mean": mean_or_zero(candidate_counts),
                "selected_protected_mean": mean_or_zero(selected_counts),
                "raw_update_norm_mean": mean_or_zero(raw_norms),
                "safe_update_norm_mean": mean_or_zero(safe_norms),
                "projection_ratio_mean": mean_or_zero(projection_ratios),
                "mean_projection_violation": mean_or_zero(mean_violations),
                "max_projection_violation": max(max_violations) if max_violations else 0.0,
                "candidate_protected_ids": sorted(set(candidate_ids)),
                "selected_protected_ids": sorted(set(selected_ids)),
                **snapshot,
                **output_drift_stats(state.theta, shape, state.anchors),
                "committed_anchor_id": pending.committed_anchor_id,
                "last_action": pending.last_action,
            }
        )


def run_gfo_stream(
    theta_after_task1: torch.Tensor,
    shape: ModelShape,
    initial_anchors: Sequence[ConceptAnchor],
    stream: Sequence[StreamEvent],
    original_old_ids: Sequence[str],
    args: argparse.Namespace,
) -> RunState:
    state = RunState(
        theta=theta_after_task1.detach().clone(),
        anchors=[
            ConceptAnchor(
                concept_id=a.concept_id,
                lineage_id=a.lineage_id,
                q=a.q.clone(),
                target_y=a.target_y.clone(),
                original_y=a.original_y.clone(),
                z_hidden=a.z_hidden.clone(),
                z_output=a.z_output.clone(),
                importance=a.importance,
                tolerance=a.tolerance,
                count=a.count,
                stability=a.stability,
                last_seen=a.last_seen,
                status=a.status,
                transformed=a.transformed,
            )
            for a in initial_anchors
        ],
    )
    for step, event in enumerate(stream, start=1):
        pending = update_pending(
            state.pending,
            state.theta,
            shape,
            event,
            step,
            cluster_threshold=args.pending_cluster_threshold,
            pressure_tau=args.pressure_tau,
            variance_tau=args.variance_tau,
        )
        action, nearest = decide_action(
            pending,
            state.anchors,
            pressure_threshold=args.pressure_threshold,
            merge_similarity=args.merge_similarity,
            conflict_similarity=args.conflict_similarity,
            conflict_threshold=args.conflict_threshold,
            committed_target_change_threshold=args.committed_target_change_threshold,
        )
        safe_write(
            state,
            shape,
            event,
            pending,
            action,
            nearest,
            step,
            original_old_ids=original_old_ids,
            write_lr=args.write_lr,
            protect_top_k=args.protect_top_k,
            damping=args.damping,
            tolerance=args.anchor_tolerance,
            tolerance_tiny=args.tolerance_tiny,
            write_steps=args.write_steps,
            target_loss_threshold=args.target_loss_threshold,
            record_diagnostics=args.record_write_diagnostics,
        )
    return state


def run_blind_sgd(
    theta_after_task1: torch.Tensor,
    shape: ModelShape,
    stream: Sequence[StreamEvent],
    *,
    lr: float,
) -> torch.Tensor:
    theta = theta_after_task1.detach().clone()
    for event in stream:
        batch = Batch(x=event.x.unsqueeze(0), y=event.y.unsqueeze(0))
        grad = loss_grad(theta, shape, batch, freeze_output=False)
        theta = (theta - lr * grad).detach()
    return theta


def evaluate_state(
    theta: torch.Tensor,
    shape: ModelShape,
    anchors: Sequence[ConceptAnchor],
    *,
    original_old_ids: Sequence[str],
) -> Dict[str, float]:
    original_set = set(original_old_ids)
    untransformed_old = [a for a in anchors if a.lineage_id in original_set and not a.transformed]
    transformed = [a for a in anchors if a.transformed]
    created = [a for a in anchors if a.lineage_id not in original_set]

    def target_loss(items: Sequence[ConceptAnchor], original: bool = False) -> float:
        if not items:
            return 0.0
        losses = []
        for anchor in items:
            pred, _ = forward(theta, shape, anchor.q.unsqueeze(0))
            target = anchor.original_y if original else anchor.target_y
            losses.append(tensor_mse(pred.reshape(-1), target))
        return float(sum(losses) / len(losses))

    return {
        "destructive_forgetting": target_loss(untransformed_old),
        "consolidation_error": target_loss(transformed),
        "old_compatibility_loss": target_loss(transformed, original=True),
        "created_concept_loss": target_loss(created),
        "active_anchor_count": float(len([a for a in anchors if a.status == "active"])),
        "transformed_count": float(len(transformed)),
        "created_count": float(len(created)),
    }


def evaluate_blind_sgd(
    theta: torch.Tensor,
    shape: ModelShape,
    initial_anchors: Sequence[ConceptAnchor],
    *,
    transformed_lineages: Sequence[str],
) -> Dict[str, float]:
    transformed_set = set(transformed_lineages)
    untransformed = [a for a in initial_anchors if a.lineage_id not in transformed_set]
    transformed = [a for a in initial_anchors if a.lineage_id in transformed_set]

    def loss_to_original(items: Sequence[ConceptAnchor]) -> float:
        if not items:
            return 0.0
        losses = []
        for anchor in items:
            pred, _ = forward(theta, shape, anchor.q.unsqueeze(0))
            losses.append(tensor_mse(pred.reshape(-1), anchor.original_y))
        return float(sum(losses) / len(losses))

    return {
        "destructive_forgetting": loss_to_original(untransformed),
        "old_compatibility_loss": loss_to_original(transformed),
    }


def run_seed(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    set_seed(seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    shape = ModelShape(args.input_dim, args.hidden_dim, args.output_dim)
    train, values, old_ids = make_world(shape, device=device, dtype=dtype)
    theta0 = init_flat(shape, device=device, dtype=dtype)
    theta1 = train_task1(theta0, shape, train, steps=args.task1_steps, lr=args.task1_lr)
    initial_anchors = build_initial_anchors(theta1, shape, train, old_ids, tolerance=args.anchor_tolerance)
    stream = make_stream(values)

    gfo_state = run_gfo_stream(theta1, shape, initial_anchors, stream, old_ids, args)
    gfo_metrics = evaluate_state(gfo_state.theta, shape, gfo_state.anchors, original_old_ids=old_ids)
    gfo_metrics["writes"] = float(gfo_state.writes)
    gfo_metrics["safe_steps"] = float(gfo_state.safe_steps)
    gfo_metrics["skipped"] = float(gfo_state.skipped)

    transformed_lineages = [anchor.lineage_id for anchor in gfo_state.anchors if anchor.transformed]
    blind_theta = run_blind_sgd(theta1, shape, stream, lr=args.write_lr)
    blind_metrics = evaluate_blind_sgd(
        blind_theta,
        shape,
        initial_anchors,
        transformed_lineages=transformed_lineages,
    )

    return {
        "seed": seed,
        "gfo": gfo_metrics,
        "blind_sgd": blind_metrics,
        "action_counts": gfo_state.action_counts,
        "pending": [
            {
                "id": p.pending_id,
                "count": p.count,
                "pressure": p.pressure,
                "committed_write_pressure": p.committed_write_pressure,
                "last_action": p.last_action,
                "committed_anchor_id": p.committed_anchor_id,
            }
            for p in gfo_state.pending
        ],
        "write_diagnostics": gfo_state.write_diagnostics if args.record_write_diagnostics else [],
    }


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    tensor = torch.tensor(vals, dtype=torch.float64)
    return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False))}


def aggregate(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metric_groups = ["gfo", "blind_sgd"]
    summary: Dict[str, object] = {}
    for group in metric_groups:
        keys = sorted(reports[0][group].keys())  # type: ignore[index, union-attr]
        summary[group] = {
            key: mean_std(report[group][key] for report in reports)  # type: ignore[index]
            for key in keys
        }
    actions: Dict[str, List[float]] = {}
    for report in reports:
        for action, count in report["action_counts"].items():  # type: ignore[index, union-attr]
            actions.setdefault(action, []).append(float(count))
    summary["actions"] = {key: mean_std(vals) for key, vals in sorted(actions.items())}
    return summary


def fmt(stats: Dict[str, float]) -> str:
    return f"{stats['mean']:.4f} +/- {stats['std']:.4f}"


def print_summary(report: Dict[str, object]) -> None:
    summary = report["summary"]  # type: ignore[index]
    print("\nGFO EVIDENCE STREAM SUMMARY")
    print("=" * 120)
    print(
        f"seeds={report['seed_count']} input={report['config']['input_dim']} "
        f"hidden={report['config']['hidden_dim']} output={report['config']['output_dim']}"
    )
    print("-" * 120)
    print(f"{'metric':30s} {'gfo':>24s} {'blind_sgd':>24s}")
    print("-" * 120)
    metric_names = sorted(set(summary["gfo"].keys()) | set(summary["blind_sgd"].keys()))  # type: ignore[index, union-attr]
    for metric in metric_names:
        gfo_stats = summary["gfo"].get(metric)  # type: ignore[index, union-attr]
        blind_stats = summary["blind_sgd"].get(metric)  # type: ignore[index, union-attr]
        print(
            f"{metric:30s} "
            f"{fmt(gfo_stats) if gfo_stats else 'n/a':>24s} "
            f"{fmt(blind_stats) if blind_stats else 'n/a':>24s}"
        )
    print("-" * 120)
    print("actions")
    for action, stats in summary["actions"].items():  # type: ignore[index, union-attr]
        print(f"{action:30s} {fmt(stats)}")
    print("=" * 120)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--task1-steps", type=int, default=1200)
    parser.add_argument("--task1-lr", type=float, default=0.05)
    parser.add_argument("--write-lr", type=float, default=0.08)
    parser.add_argument("--write-steps", type=int, default=5)
    parser.add_argument("--target-loss-threshold", type=float, default=1e-3)
    parser.add_argument("--anchor-tolerance", type=float, default=0.2)
    parser.add_argument("--protect-top-k", type=int, default=2)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--tolerance-tiny", type=float, default=1e-12)

    parser.add_argument("--pending-cluster-threshold", type=float, default=0.98)
    parser.add_argument("--pressure-threshold", type=float, default=0.45)
    parser.add_argument("--pressure-tau", type=float, default=3.0)
    parser.add_argument("--variance-tau", type=float, default=0.05)
    parser.add_argument("--merge-similarity", type=float, default=0.98)
    parser.add_argument("--conflict-similarity", type=float, default=0.98)
    parser.add_argument("--conflict-threshold", type=float, default=0.05)
    parser.add_argument("--committed-target-change-threshold", type=float, default=1e-4)
    parser.add_argument("--record-write-diagnostics", action="store_true")

    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    if args.write_lr <= 0 or args.task1_lr <= 0:
        raise ValueError("--write-lr and --task1-lr must be positive.")
    if args.write_steps <= 0:
        raise ValueError("--write-steps must be positive.")
    if args.target_loss_threshold < 0:
        raise ValueError("--target-loss-threshold must be non-negative.")
    if args.anchor_tolerance <= 0:
        raise ValueError("--anchor-tolerance must be positive.")
    if args.protect_top_k < 0:
        raise ValueError("--protect-top-k must be non-negative.")


def config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "task1_steps": args.task1_steps,
        "task1_lr": args.task1_lr,
        "write_lr": args.write_lr,
        "write_steps": args.write_steps,
        "target_loss_threshold": args.target_loss_threshold,
        "anchor_tolerance": args.anchor_tolerance,
        "protect_top_k": args.protect_top_k,
        "pressure_threshold": args.pressure_threshold,
        "committed_target_change_threshold": args.committed_target_change_threshold,
        "record_write_diagnostics": args.record_write_diagnostics,
        "device": args.device,
        "dtype": args.dtype,
    }


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
        "experiment": "gfo_evidence_stream",
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
