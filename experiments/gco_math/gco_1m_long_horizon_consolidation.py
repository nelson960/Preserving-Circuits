"""Test unified constrained long-horizon CL in the approximately 1M-weight model.

The learner receives a sequence of book continuations, novel facts, factual
corrections, and isolated noise.  Four learner-side memories remain bounded:

* an executable committed trace bank;
* a pending candidate pool;
* a fixed guard coreset;
* hard protected margin floors and a fixed-rank streaming dependency sketch.

Evaluation history is retained only for offline measurement.  It is never used
for trace selection, gradient projection, restoration, or transaction checks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_1m_consequence_survival_cl import (
    absolute_weighted_language_loss,
    build_parser as build_scaled_parser,
    candidate_measurement,
    consequence_survival,
    correction_objective,
    model_width,
    permute_windows,
    restore_parameters,
    snapshot_parameters,
    split_windows,
    validate_args as validate_scaled_args,
)
from experiments.gco_math.gco_mini_cl_world_demo import (
    flat_autograd_gradient,
    project_gradient_away_from_constraints,
)
from experiments.gco_math.gco_unified_constraint_solver import (
    StreamingConstraintSketch,
    apply_flat_delta,
    constraint_rank_report,
    solve_unified_step,
    update_streaming_sketch,
)
from experiments.gco_math.gco_storage_capacity_sweep import (
    fact_sentences,
    first_words,
    load_book_text,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import trainable_weight_parameters
from experiments.gco_math.gco_tiny_text_dependency_cl import (
    ConstraintBasis,
    ExecutableTraceBank,
    FactQuery,
    PendingTraceCommit,
    TextWindows,
    balanced_block_windows,
    block_normalize,
    build_constraint_basis,
    collect_geometry,
    combine_windows,
    commit_trace_bank,
    encode_trace_evidence,
    energy_rank,
    evaluate_correction_queries,
    evaluate_windows,
    forward_with_states,
    geometry_report,
    initialize_trace_bank,
    instantiate_model,
    normalized_damage,
    protection_losses,
    propose_trace_update,
    random_noise_windows,
    word_span,
    windows_from_text,
)
from experiments.gco_math.gco_tiny_trace_invariant_tangent_integration import (
    bounded_restore,
    recurrence_probability,
    reconstruction_confidence,
)


@dataclass(frozen=True)
class CycleData:
    stream: TextWindows
    train_without_noise: TextWindows
    book_eval: TextWindows
    novel_eval: TextWindows
    corrected_eval: TextWindows
    obsolete_eval: TextWindows
    correction_queries: tuple[FactQuery, ...]
    archived_queries: tuple[FactQuery, ...]
    rare_eval: TextWindows
    misinformation_eval: TextWindows
    misinformation_truth_eval: TextWindows
    misinformation_queries: tuple[FactQuery, ...]


@dataclass
class PendingPool:
    windows: TextWindows
    masses: torch.Tensor
    ages: torch.Tensor


@dataclass(frozen=True)
class PendingGroupVerifier:
    queries: tuple[FactQuery, ...]
    mode: str


@dataclass
class SemanticAnchor:
    query: FactQuery
    group: str
    reference_margin: float
    fast_support: float
    slow_support: float
    support_sum: float
    observations: int
    last_update: int


@dataclass
class SemanticAnchorMemory:
    anchors: list[SemanticAnchor]


@dataclass
class SemanticCandidate:
    query: FactQuery
    group: str
    fast_support: float
    slow_support: float
    support_sum: float
    observations: int
    first_update: int
    last_update: int


@dataclass
class SemanticCandidateMemory:
    candidates: list[SemanticCandidate]


@dataclass
class TraceTimescaleRecord:
    fast_support: float
    slow_support: float
    support_sum: float
    observations: int
    last_update: int


@dataclass(frozen=True)
class FunctionalGeometryReference:
    """Bounded readout-sensitive geometry carried between accepted updates."""

    projector: torch.Tensor
    normalized_gram: torch.Tensor
    pooled_states: torch.Tensor


@dataclass
class TraceTimescaleMemory:
    records: dict[
        tuple[tuple[int, ...], tuple[int, ...]], TraceTimescaleRecord
    ]


def initialize_timescale_memory(bank: ExecutableTraceBank) -> TraceTimescaleMemory:
    records: dict[
        tuple[tuple[int, ...], tuple[int, ...]], TraceTimescaleRecord
    ] = {}
    for index in range(bank.inputs.shape[0]):
        key = exact_window_key(bank.inputs[index], bank.targets[index])
        if key in records:
            raise RuntimeError("Initial committed bank contains duplicate trace keys.")
        records[key] = TraceTimescaleRecord(
            fast_support=0.0,
            slow_support=0.0,
            support_sum=0.0,
            observations=0,
            last_update=0,
        )
    return TraceTimescaleMemory(records=records)


def update_timescale_record(
    record: TraceTimescaleRecord,
    support: float,
    *,
    update: int,
    fast_decay: float,
    slow_decay: float,
) -> TraceTimescaleRecord:
    if not math.isfinite(support) or not 0.0 <= support <= 1.0:
        raise ValueError("Trace support must be finite and lie in [0, 1].")
    fast = fast_decay * record.fast_support + (1.0 - fast_decay) * support
    slow = slow_decay * record.slow_support + (1.0 - slow_decay) * fast
    return TraceTimescaleRecord(
        fast_support=fast,
        slow_support=slow,
        support_sum=record.support_sum + support,
        observations=record.observations + 1,
        last_update=update,
    )


def durability_interval(
    record: TraceTimescaleRecord,
    *,
    confidence_delta: float,
) -> dict[str, float] | None:
    if record.observations <= 0:
        return None
    mean = record.support_sum / record.observations
    radius = math.sqrt(
        math.log(2.0 / confidence_delta) / (2.0 * record.observations)
    )
    lower = max(0.0, mean - radius)
    upper = min(1.0, mean + radius)
    return {
        "mean": mean,
        "radius": radius,
        "lower": lower,
        "upper": upper,
        "lower_score": record.slow_support * lower,
        "point_score": record.slow_support * mean,
        "upper_score": record.slow_support * upper,
    }


def update_committed_timescales(
    memory: TraceTimescaleMemory,
    bank: ExecutableTraceBank,
    consequence: dict[str, Any],
    *,
    update: int,
    fast_decay: float,
    slow_decay: float,
) -> None:
    survival = consequence["survival"]
    if len(survival) != bank.inputs.shape[0]:
        raise ValueError("Consequence survival does not match committed trace count.")
    for index, support in enumerate(survival):
        key = exact_window_key(bank.inputs[index], bank.targets[index])
        record = memory.records.get(
            key,
            TraceTimescaleRecord(0.0, 0.0, 0.0, 0, update),
        )
        memory.records[key] = update_timescale_record(
            record,
            float(support),
            update=update,
            fast_decay=fast_decay,
            slow_decay=slow_decay,
        )


def reinforce_pending_timescales(
    memory: TraceTimescaleMemory,
    pool: PendingPool | None,
    verified_weights: torch.Tensor,
    *,
    update: int,
    fast_decay: float,
    slow_decay: float,
) -> None:
    if pool is None:
        return
    if verified_weights.shape != pool.masses.shape:
        raise ValueError("Pending timescale weights do not match the pool.")
    for index, support_tensor in enumerate(verified_weights):
        support = float(support_tensor)
        key = exact_window_key(pool.windows.inputs[index], pool.windows.targets[index])
        record = memory.records.get(
            key,
            TraceTimescaleRecord(0.0, 0.0, 0.0, 0, update),
        )
        memory.records[key] = update_timescale_record(
            record,
            support,
            update=update,
            fast_decay=fast_decay,
            slow_decay=slow_decay,
        )


@torch.no_grad()
def query_margin_values(
    model: torch.nn.Module,
    queries: tuple[FactQuery, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not queries:
        raise ValueError("Pending semantic verifier requires at least one query.")
    margins: list[torch.Tensor] = []
    for query in queries:
        token_ids = list(query.input_ids[-model.max_seq_len :])
        if not token_ids:
            raise ValueError("Pending semantic verifier query has no input tokens.")
        inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1]
        margins.append(logits[query.new_target_id] - logits[query.old_target_id])
    return torch.stack(margins)


def semantic_query_margin_tensor(
    model: torch.nn.Module,
    query: FactQuery,
    *,
    device: torch.device,
) -> torch.Tensor:
    token_ids = list(query.input_ids[-model.max_seq_len :])
    if not token_ids:
        raise ValueError("Semantic anchor query has no input tokens.")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model(inputs)[0, -1]
    return logits[query.new_target_id] - logits[query.old_target_id]


def semantic_anchor_key(
    query: FactQuery,
) -> tuple[tuple[int, ...], int, int]:
    return query.input_ids, query.old_target_id, query.new_target_id


def semantic_anchor_floor(*, minimum_margin: float) -> float:
    """Protect semantic correctness, not the original confidence magnitude."""

    return minimum_margin


def semantic_anchor_constraints(
    model: torch.nn.Module,
    memory: SemanticAnchorMemory,
    *,
    parameters: list[torch.nn.Parameter],
    minimum_margin: float,
    activation_margin: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if minimum_margin < 0.0 or not math.isfinite(minimum_margin):
        raise ValueError("Semantic anchor minimum margin must be finite and non-negative.")
    if activation_margin < 0.0 or not math.isfinite(activation_margin):
        raise ValueError("Semantic anchor activation margin must be finite and non-negative.")
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if not memory.anchors:
        empty_rows = torch.empty(0, parameter_count, device=device)
        empty_bounds = torch.empty(0, device=device)
        return empty_rows, empty_bounds, empty_bounds.new_zeros(()), {
            "count": 0,
            "minimum_margin": None,
            "minimum_slack": None,
            "groups": {},
        }

    rows: list[torch.Tensor] = []
    bounds: list[torch.Tensor] = []
    margins: list[float] = []
    floors: list[float] = []
    restore_terms: list[torch.Tensor] = []
    groups: Counter[str] = Counter()
    for index, anchor in enumerate(memory.anchors):
        margin = semantic_query_margin_tensor(model, anchor.query, device=device)
        floor = semantic_anchor_floor(minimum_margin=minimum_margin)
        gradient = flat_autograd_gradient(
            margin,
            parameters,
            retain_graph=False,
            require_nonzero=True,
            label=f"semantic_anchor_{index}",
        )
        norm = torch.linalg.vector_norm(gradient)
        if not torch.isfinite(norm) or float(norm.detach().cpu()) <= 1e-12:
            raise FloatingPointError("Semantic anchor has an invalid margin gradient.")
        rows.append((gradient / norm).detach())
        bounds.append(((floor - margin.detach()) / norm).to(gradient))
        margins.append(float(margin.detach().cpu()))
        floors.append(floor)
        groups[anchor.group] += 1
        restore_margin = semantic_query_margin_tensor(
            model,
            anchor.query,
            device=device,
        )
        restore_terms.append(
            F.relu(restore_margin.new_tensor(floor + activation_margin) - restore_margin)
            .square()
        )
    matrix = torch.stack(rows)
    lower_bounds = torch.stack(bounds).to(matrix)
    restore_loss = torch.stack(restore_terms).mean()
    slacks = [margin - floor for margin, floor in zip(margins, floors, strict=True)]
    return matrix, lower_bounds, restore_loss, {
        "count": len(memory.anchors),
        "minimum_margin": min(margins),
        "minimum_slack": min(slacks),
        "groups": dict(groups),
        "margins": margins,
        "floors": floors,
    }


@torch.no_grad()
def semantic_anchor_measurement(
    model: torch.nn.Module,
    memory: SemanticAnchorMemory,
    *,
    minimum_margin: float,
    device: torch.device,
) -> dict[str, Any]:
    if not memory.anchors:
        return {
            "count": 0,
            "passed": True,
            "minimum_margin": None,
            "minimum_slack": None,
            "groups": {},
        }
    margins: list[float] = []
    floors: list[float] = []
    groups: Counter[str] = Counter()
    for anchor in memory.anchors:
        margin = float(
            semantic_query_margin_tensor(model, anchor.query, device=device)
            .detach()
            .cpu()
        )
        floor = semantic_anchor_floor(minimum_margin=minimum_margin)
        margins.append(margin)
        floors.append(floor)
        groups[anchor.group] += 1
    slacks = [margin - floor for margin, floor in zip(margins, floors, strict=True)]
    return {
        "count": len(memory.anchors),
        "passed": min(slacks) >= 0.0,
        "minimum_margin": min(margins),
        "minimum_slack": min(slacks),
        "groups": dict(groups),
        "margins": margins,
        "floors": floors,
    }


def semantic_anchor_restore_loss(
    model: torch.nn.Module,
    memory: SemanticAnchorMemory,
    *,
    minimum_margin: float,
    activation_margin: float,
    device: torch.device,
) -> torch.Tensor:
    if not memory.anchors:
        return torch.zeros((), device=device)
    terms: list[torch.Tensor] = []
    for anchor in memory.anchors:
        margin = semantic_query_margin_tensor(model, anchor.query, device=device)
        floor = semantic_anchor_floor(minimum_margin=minimum_margin)
        terms.append(
            F.relu(margin.new_tensor(floor + activation_margin) - margin).square()
        )
    return torch.stack(terms).mean()


@torch.no_grad()
def update_semantic_anchor_support(
    model: torch.nn.Module,
    memory: SemanticAnchorMemory,
    *,
    update: int,
    minimum_margin: float,
    fast_decay: float,
    slow_decay: float,
    device: torch.device,
) -> None:
    for anchor in memory.anchors:
        margin = float(
            semantic_query_margin_tensor(model, anchor.query, device=device)
            .detach()
            .cpu()
        )
        floor = semantic_anchor_floor(minimum_margin=minimum_margin)
        support = float(margin >= floor)
        anchor.fast_support = (
            fast_decay * anchor.fast_support + (1.0 - fast_decay) * support
        )
        anchor.slow_support = (
            slow_decay * anchor.slow_support
            + (1.0 - slow_decay) * anchor.fast_support
        )
        anchor.support_sum += support
        anchor.observations += 1
        anchor.last_update = update


def semantic_anchor_priority(
    anchor: SemanticAnchor,
    current_margin: float,
    *,
    minimum_margin: float,
    temperature: float,
    current_weight: float,
    reference_weight: float,
    history_weight: float,
) -> float:
    floor = semantic_anchor_floor(minimum_margin=minimum_margin)
    current = stable_sigmoid((current_margin - floor) / temperature)
    reference = stable_sigmoid((anchor.reference_margin - floor) / temperature)
    mean_support = anchor.support_sum / max(anchor.observations, 1)
    historical = anchor.slow_support * mean_support
    return (
        current_weight * current
        + reference_weight * reference
        + history_weight * historical
    )


def stable_sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Semantic priority input must be finite.")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def semantic_candidate_priority(
    candidate: SemanticCandidate,
    current_margin: float,
    *,
    minimum_margin: float,
    temperature: float,
    current_weight: float,
    history_weight: float,
) -> float:
    current = stable_sigmoid((current_margin - minimum_margin) / temperature)
    mean_support = candidate.support_sum / max(candidate.observations, 1)
    historical = candidate.slow_support * mean_support
    return current_weight * current + history_weight * historical


@torch.no_grad()
def compact_semantic_candidates(
    model: torch.nn.Module,
    memory: SemanticCandidateMemory,
    *,
    slots: int,
    minimum_margin: float,
    priority_temperature: float,
    current_priority_weight: float,
    history_priority_weight: float,
    device: torch.device,
) -> dict[str, int]:
    if slots <= 0:
        raise ValueError("Semantic candidate capacity must be positive.")
    scored: list[tuple[float, SemanticCandidate]] = []
    for candidate in memory.candidates:
        current_margin = float(
            semantic_query_margin_tensor(model, candidate.query, device=device)
            .detach()
            .cpu()
        )
        score = semantic_candidate_priority(
            candidate,
            current_margin,
            minimum_margin=minimum_margin,
            temperature=priority_temperature,
            current_weight=current_priority_weight,
            history_weight=history_priority_weight,
        )
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    evicted = Counter(candidate.group for _score, candidate in scored[slots:])
    memory.candidates = [candidate for _score, candidate in scored[:slots]]
    return dict(evicted)


@torch.no_grad()
def register_semantic_candidates(
    model: torch.nn.Module,
    candidate_memory: SemanticCandidateMemory,
    anchor_memory: SemanticAnchorMemory,
    *,
    group: str,
    queries: tuple[FactQuery, ...],
    update: int,
    slots: int,
    minimum_margin: float,
    priority_temperature: float,
    current_priority_weight: float,
    history_priority_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    anchor_keys = {semantic_anchor_key(anchor.query) for anchor in anchor_memory.anchors}
    existing = {
        semantic_anchor_key(candidate.query): candidate
        for candidate in candidate_memory.candidates
    }
    added = 0
    for query in queries:
        key = semantic_anchor_key(query)
        if key in anchor_keys or key in existing:
            continue
        existing[key] = SemanticCandidate(
            query=query,
            group=group,
            fast_support=0.0,
            slow_support=0.0,
            support_sum=0.0,
            observations=0,
            first_update=update,
            last_update=update,
        )
        added += 1
    candidate_memory.candidates = list(existing.values())
    evicted = compact_semantic_candidates(
        model,
        candidate_memory,
        slots=slots,
        minimum_margin=minimum_margin,
        priority_temperature=priority_temperature,
        current_priority_weight=current_priority_weight,
        history_priority_weight=history_priority_weight,
        device=device,
    )
    return {
        "added": added,
        "evicted": evicted,
        "size": len(candidate_memory.candidates),
        "groups": dict(Counter(candidate.group for candidate in candidate_memory.candidates)),
    }


@torch.no_grad()
def update_and_promote_semantic_candidates(
    model: torch.nn.Module,
    candidate_memory: SemanticCandidateMemory,
    anchor_memory: SemanticAnchorMemory,
    *,
    update: int,
    candidate_slots: int,
    anchor_slots: int,
    minimum_margin: float,
    minimum_age: int,
    minimum_verifications: int,
    minimum_mean_support: float,
    priority_temperature: float,
    current_priority_weight: float,
    reference_priority_weight: float,
    history_priority_weight: float,
    fast_decay: float,
    slow_decay: float,
    device: torch.device,
) -> dict[str, Any]:
    if candidate_slots <= 0 or anchor_slots <= 0:
        raise ValueError("Semantic candidate and anchor capacities must be positive.")
    eligible: list[tuple[SemanticCandidate, float]] = []
    candidate_records: list[dict[str, Any]] = []
    for candidate in candidate_memory.candidates:
        margin = float(
            semantic_query_margin_tensor(model, candidate.query, device=device)
            .detach()
            .cpu()
        )
        support = float(margin >= minimum_margin)
        candidate.fast_support = (
            fast_decay * candidate.fast_support + (1.0 - fast_decay) * support
        )
        candidate.slow_support = (
            slow_decay * candidate.slow_support
            + (1.0 - slow_decay) * candidate.fast_support
        )
        candidate.support_sum += support
        candidate.observations += 1
        candidate.last_update = update
        mean_support = candidate.support_sum / candidate.observations
        age = update - candidate.first_update
        passed = (
            age >= minimum_age
            and candidate.observations >= minimum_verifications
            and mean_support >= minimum_mean_support
            and support > 0.0
        )
        if passed:
            eligible.append((candidate, margin))
        candidate_records.append(
            {
                "group": candidate.group,
                "input_ids": list(candidate.query.input_ids),
                "old_target_id": candidate.query.old_target_id,
                "new_target_id": candidate.query.new_target_id,
                "margin": margin,
                "age": age,
                "observations": candidate.observations,
                "mean_support": mean_support,
                "eligible": passed,
            }
        )

    anchors_by_key = {
        semantic_anchor_key(anchor.query): anchor for anchor in anchor_memory.anchors
    }
    for candidate, margin in eligible:
        key = semantic_anchor_key(candidate.query)
        anchors_by_key[key] = SemanticAnchor(
            query=candidate.query,
            group=candidate.group,
            reference_margin=margin,
            fast_support=candidate.fast_support,
            slow_support=candidate.slow_support,
            support_sum=candidate.support_sum,
            observations=candidate.observations,
            last_update=update,
        )
    scored_anchors: list[tuple[float, SemanticAnchor]] = []
    for anchor in anchors_by_key.values():
        margin = float(
            semantic_query_margin_tensor(model, anchor.query, device=device)
            .detach()
            .cpu()
        )
        scored_anchors.append(
            (
                semantic_anchor_priority(
                    anchor,
                    margin,
                    minimum_margin=minimum_margin,
                    temperature=priority_temperature,
                    current_weight=current_priority_weight,
                    reference_weight=reference_priority_weight,
                    history_weight=history_priority_weight,
                ),
                anchor,
            )
        )
    scored_anchors.sort(key=lambda item: item[0], reverse=True)
    retained_anchors = [anchor for _score, anchor in scored_anchors[:anchor_slots]]
    retained_anchor_keys = {
        semantic_anchor_key(anchor.query) for anchor in retained_anchors
    }
    evicted_anchors = Counter(
        anchor.group for _score, anchor in scored_anchors[anchor_slots:]
    )
    anchor_memory.anchors = retained_anchors
    promoted_keys = {
        semantic_anchor_key(candidate.query)
        for candidate, _margin in eligible
        if semantic_anchor_key(candidate.query) in retained_anchor_keys
    }
    promoted_groups = Counter(
        candidate.group
        for candidate, _margin in eligible
        if semantic_anchor_key(candidate.query) in promoted_keys
    )
    candidate_memory.candidates = [
        candidate
        for candidate in candidate_memory.candidates
        if semantic_anchor_key(candidate.query) not in promoted_keys
    ]
    evicted_candidates = compact_semantic_candidates(
        model,
        candidate_memory,
        slots=candidate_slots,
        minimum_margin=minimum_margin,
        priority_temperature=priority_temperature,
        current_priority_weight=current_priority_weight,
        history_priority_weight=history_priority_weight,
        device=device,
    )
    return {
        "promoted": dict(promoted_groups),
        "anchor_evicted": dict(evicted_anchors),
        "candidate_evicted": evicted_candidates,
        "anchor_size": len(anchor_memory.anchors),
        "candidate_size": len(candidate_memory.candidates),
        "anchor_groups": dict(Counter(anchor.group for anchor in anchor_memory.anchors)),
        "candidate_groups": dict(
            Counter(candidate.group for candidate in candidate_memory.candidates)
        ),
        "candidates": candidate_records,
    }


@torch.no_grad()
def verify_pending_candidates(
    model: torch.nn.Module,
    pool: PendingPool | None,
    group_verifiers: dict[str, PendingGroupVerifier],
    *,
    minimum_behavior_confidence: float,
    semantic_margin: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not 0.0 <= minimum_behavior_confidence <= 1.0:
        raise ValueError("Pending behavior confidence threshold must be in [0, 1].")
    if semantic_margin < 0.0 or not math.isfinite(semantic_margin):
        raise ValueError("Pending semantic margin must be finite and non-negative.")
    if pool is None:
        return torch.empty(0), {"candidates": 0, "verified": 0, "groups": {}}

    logits, _states = forward_with_states(model, pool.windows.inputs.to(device))
    targets = pool.windows.targets.to(device)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    behavior_confidence = torch.exp(-token_losses.mean(dim=1)).clamp(0.0, 1.0)
    verified = torch.zeros_like(behavior_confidence)
    group_report: dict[str, Any] = {}
    for group in sorted(set(pool.windows.groups)):
        indices = torch.tensor(
            [index for index, value in enumerate(pool.windows.groups) if value == group],
            dtype=torch.long,
            device=device,
        )
        confidence = behavior_confidence[indices]
        behavior_passed = confidence >= minimum_behavior_confidence
        verifier = group_verifiers.get(group)
        semantic_passed = True
        semantic_values: list[float] = []
        mode = "behavior_only"
        if verifier is not None:
            if verifier.mode not in {"required_preference", "trusted_veto"}:
                raise ValueError(f"Unknown pending verifier mode {verifier.mode!r}.")
            margins = query_margin_values(model, verifier.queries, device=device)
            semantic_values = [float(value) for value in margins.detach().cpu()]
            mode = verifier.mode
            if verifier.mode == "required_preference":
                semantic_passed = bool((margins >= semantic_margin).all().item())
            else:
                # A trusted veto marks this candidate as ineligible for commitment.
                semantic_passed = False
        group_weights = torch.where(
            behavior_passed & semantic_passed,
            confidence,
            torch.zeros_like(confidence),
        )
        verified[indices] = group_weights
        group_report[group] = {
            "mode": mode,
            "candidates": int(indices.numel()),
            "behavior_confidence_mean": float(confidence.mean().detach().cpu()),
            "behavior_confidence_min": float(confidence.min().detach().cpu()),
            "behavior_passed": int(behavior_passed.sum().detach().cpu()),
            "semantic_margins": semantic_values,
            "semantic_passed": semantic_passed,
            "verified": int((group_weights > 0.0).sum().detach().cpu()),
        }
    return verified.detach().cpu(), {
        "candidates": int(verified.numel()),
        "verified": int((verified > 0.0).sum().detach().cpu()),
        "mean_support": float(verified.mean().detach().cpu()),
        "groups": group_report,
    }


def durable_bank_indices(
    memory: TraceTimescaleMemory,
    bank: ExecutableTraceBank,
    *,
    slots: int,
    minimum_slow_support: float,
    minimum_observations: int,
    confidence_delta: float,
) -> list[int]:
    candidates: list[tuple[float, int]] = []
    for index in range(bank.inputs.shape[0]):
        key = exact_window_key(bank.inputs[index], bank.targets[index])
        record = memory.records.get(key)
        if record is None:
            continue
        if (
            record.slow_support >= minimum_slow_support
            and record.observations >= minimum_observations
        ):
            interval = durability_interval(
                record,
                confidence_delta=confidence_delta,
            )
            if interval is None:
                raise RuntimeError("Eligible durable trace has no observations.")
            candidates.append((interval["lower_score"], index))
    candidates.sort(reverse=True)
    return [index for _score, index in candidates[:slots]]


def commit_with_durable_traces(
    model: torch.nn.Module,
    old_bank: ExecutableTraceBank,
    pending: PendingTraceCommit,
    incumbent_indices: list[int],
    memory: TraceTimescaleMemory,
    *,
    durable_slots: int,
    minimum_slow_support: float,
    minimum_observations: int,
    confidence_delta: float,
    device: torch.device,
) -> tuple[ExecutableTraceBank, dict[str, Any]]:
    proposed = commit_trace_bank(model, pending, device=device)
    capacity = old_bank.inputs.shape[0]
    if proposed.inputs.shape[0] != capacity:
        raise ValueError("Proposed and existing committed banks have different capacities.")
    old_by_key = {
        exact_window_key(old_bank.inputs[index], old_bank.targets[index]): index
        for index in range(old_bank.inputs.shape[0])
    }
    proposed_by_key = {
        exact_window_key(proposed.inputs[index], proposed.targets[index]): index
        for index in range(proposed.inputs.shape[0])
    }
    if len(old_by_key) != old_bank.inputs.shape[0]:
        raise RuntimeError("Existing bank contains duplicate trace keys.")
    if len(proposed_by_key) != proposed.inputs.shape[0]:
        raise RuntimeError("Proposed bank contains duplicate trace keys.")

    selected: dict[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[str, int, dict[str, float]]
    ] = {}
    for index in incumbent_indices:
        key = exact_window_key(old_bank.inputs[index], old_bank.targets[index])
        interval = durability_interval(
            memory.records[key],
            confidence_delta=confidence_delta,
        )
        if interval is None:
            raise RuntimeError("Durable incumbent has no statistical interval.")
        selected[key] = ("old", index, interval)

    challengers: list[
        tuple[
            float,
            tuple[tuple[int, ...], tuple[int, ...]],
            int,
            dict[str, float],
        ]
    ] = []
    for key, index in proposed_by_key.items():
        if key in old_by_key:
            continue
        record = memory.records.get(key)
        if (
            record is None
            or record.slow_support < minimum_slow_support
            or record.observations < minimum_observations
        ):
            continue
        interval = durability_interval(record, confidence_delta=confidence_delta)
        if interval is None:
            raise RuntimeError("Eligible durable challenger has no observations.")
        challengers.append((interval["lower_score"], key, index, interval))
    challengers.sort(reverse=True, key=lambda item: item[0])

    replacements: list[dict[str, Any]] = []
    for _score, key, index, interval in challengers:
        if len(selected) < durable_slots:
            selected[key] = ("proposed", index, interval)
            replacements.append(
                {"replaced": None, "challenger": proposed.groups[index]}
            )
            continue
        weakest_key, weakest = min(
            selected.items(),
            key=lambda item: item[1][2]["upper_score"],
        )
        if interval["lower_score"] <= weakest[2]["upper_score"]:
            continue
        weakest_source, weakest_index, _weakest_interval = weakest
        weakest_bank = old_bank if weakest_source == "old" else proposed
        selected.pop(weakest_key)
        selected[key] = ("proposed", index, interval)
        replacements.append(
            {
                "replaced": weakest_bank.groups[weakest_index],
                "challenger": proposed.groups[index],
            }
        )

    durable_entries = sorted(
        selected.values(),
        key=lambda entry: entry[2]["lower_score"],
        reverse=True,
    )
    durable_keys = set(selected)
    ranked_proposed = sorted(
        range(proposed.inputs.shape[0]),
        key=lambda index: float(proposed.masses[index]),
        reverse=True,
    )
    entries: list[tuple[str, int]] = [
        (source, index) for source, index, _interval in durable_entries
    ]
    for index in ranked_proposed:
        key = exact_window_key(proposed.inputs[index], proposed.targets[index])
        if key in durable_keys:
            continue
        entries.append(("proposed", index))
        if len(entries) == capacity:
            break
    if len(entries) != capacity:
        raise RuntimeError(
            "Durable commit could not fill the bounded bank without duplicate traces."
        )

    def stack_field(name: str) -> torch.Tensor:
        values = []
        for source, index in entries:
            source_bank = old_bank if source == "old" else proposed
            values.append(getattr(source_bank, name)[index])
        return torch.stack(values)

    groups = tuple(
        (old_bank if source == "old" else proposed).groups[index]
        for source, index in entries
    )
    committed = ExecutableTraceBank(
        inputs=stack_field("inputs"),
        targets=stack_field("targets"),
        groups=groups,
        centers=stack_field("centers"),
        masses=stack_field("masses"),
        reference_logits=stack_field("reference_logits"),
        reference_states=stack_field("reference_states"),
    )
    return committed, {
        "count": len(durable_entries),
        "groups": [
            (old_bank if source == "old" else proposed).groups[index]
            for source, index, _interval in durable_entries
        ],
        "lower_scores": [
            interval["lower_score"]
            for _source, _index, interval in durable_entries
        ],
        "replacements": replacements,
    }


def synchronize_timescale_memory(
    memory: TraceTimescaleMemory,
    bank: ExecutableTraceBank,
    pool: PendingPool | None,
    *,
    update: int,
) -> None:
    retained = {
        exact_window_key(bank.inputs[index], bank.targets[index])
        for index in range(bank.inputs.shape[0])
    }
    if pool is not None:
        retained.update(
            exact_window_key(pool.windows.inputs[index], pool.windows.targets[index])
            for index in range(pool.windows.inputs.shape[0])
        )
    for key in retained:
        if key not in memory.records:
            memory.records[key] = TraceTimescaleRecord(
                fast_support=0.0,
                slow_support=0.0,
                support_sum=0.0,
                observations=0,
                last_update=update,
            )
    memory.records = {
        key: value for key, value in memory.records.items() if key in retained
    }
    if set(memory.records) != retained:
        raise RuntimeError("Failed to synchronize bounded timescale metadata.")


def timescale_memory_report(
    memory: TraceTimescaleMemory,
    bank: ExecutableTraceBank,
    pool: PendingPool | None,
    *,
    confidence_delta: float,
) -> list[dict[str, Any]]:
    labels: dict[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[str, int, str]
    ] = {}
    for index in range(bank.inputs.shape[0]):
        key = exact_window_key(bank.inputs[index], bank.targets[index])
        labels[key] = ("committed", index, bank.groups[index])
    if pool is not None:
        for index in range(pool.windows.inputs.shape[0]):
            key = exact_window_key(
                pool.windows.inputs[index], pool.windows.targets[index]
            )
            if key in labels:
                raise RuntimeError("A trace exists in committed and pending memory.")
            labels[key] = ("pending", index, pool.windows.groups[index])
    if set(labels) != set(memory.records):
        raise RuntimeError("Timescale metadata does not match bounded trace memory.")
    report: list[dict[str, Any]] = []
    for key, (location, index, group) in labels.items():
        record = memory.records[key]
        report.append(
            {
                "location": location,
                "index": index,
                "group": group,
                "fast_support": record.fast_support,
                "slow_support": record.slow_support,
                "mean_support": (
                    record.support_sum / record.observations
                    if record.observations
                    else None
                ),
                "support_sum": record.support_sum,
                "observations": record.observations,
                "confidence": durability_interval(
                    record,
                    confidence_delta=confidence_delta,
                ),
                "last_update": record.last_update,
            }
        )
    return report


def hard_margin_constraints(
    model: torch.nn.Module,
    bank: ExecutableTraceBank,
    protection: torch.Tensor,
    *,
    parameters: list[torch.nn.Parameter],
    hard_slots: int,
    margin_tolerance: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build label-free hard floors from the most protected executable traces."""

    if protection.shape != (bank.inputs.shape[0],):
        raise ValueError("Hard-constraint protection does not match the trace bank.")
    if hard_slots <= 0 or hard_slots > bank.inputs.shape[0]:
        raise ValueError(
            f"hard_slots must be in [1, {bank.inputs.shape[0]}], got {hard_slots}."
        )
    if margin_tolerance < 0.0 or not math.isfinite(margin_tolerance):
        raise ValueError("Hard margin tolerance must be finite and non-negative.")
    selected = torch.topk(protection, k=hard_slots, largest=True).indices
    logits, _states = forward_with_states(model, bank.inputs.to(device))
    current_logits = logits[:, -1]
    reference_logits = bank.reference_logits.to(device)
    target_ids = bank.targets[:, -1].to(device)
    competitors = reference_logits.clone()
    competitors.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
    competitor_ids = competitors.argmax(dim=1)
    rows: list[torch.Tensor] = []
    lower_bounds: list[torch.Tensor] = []
    selected_groups: list[str] = []
    current_margins: list[float] = []
    reference_margins: list[float] = []
    row_norms: list[float] = []
    for position, trace_tensor in enumerate(selected):
        trace = int(trace_tensor.item())
        current_margin = (
            current_logits[trace, target_ids[trace]]
            - current_logits[trace, competitor_ids[trace]]
        )
        reference_margin = (
            reference_logits[trace, target_ids[trace]]
            - reference_logits[trace, competitor_ids[trace]]
        ).detach()
        row = flat_autograd_gradient(
            current_margin,
            parameters,
            retain_graph=position + 1 < selected.numel(),
            require_nonzero=True,
            label=f"hard_margin_trace_{trace}",
        )
        row_norm = torch.linalg.vector_norm(row)
        if not torch.isfinite(row_norm) or row_norm <= 1e-12:
            raise FloatingPointError(f"Hard trace {trace} has an invalid margin gradient.")
        normalized = row / row_norm
        floor = reference_margin - margin_tolerance
        floor_gap = (floor - current_margin.detach()) / row_norm
        # A hard barrier prevents additional damage. Existing floor violations
        # are repaired by the soft restore field instead of being forced into a
        # single potentially infeasible trust-region step.
        lower_bound = torch.minimum(floor_gap, torch.zeros_like(floor_gap))
        rows.append(normalized.detach())
        lower_bounds.append(lower_bound.to(dtype=normalized.dtype))
        selected_groups.append(bank.groups[trace])
        current_margins.append(float(current_margin.detach().cpu()))
        reference_margins.append(float(reference_margin.detach().cpu()))
        row_norms.append(float(row_norm.detach().cpu()))
    matrix = torch.stack(rows)
    bounds = torch.stack(lower_bounds).to(matrix)
    return matrix, bounds, {
        "selected_indices": [int(value) for value in selected.detach().cpu()],
        "selected_groups": selected_groups,
        "current_margins": current_margins,
        "reference_margins": reference_margins,
        "row_norms": row_norms,
        "existing_floor_violations": [
            max(
                0.0,
                (reference_margins[index] - margin_tolerance - current_margins[index])
                / row_norms[index],
            )
            for index in range(len(row_norms))
        ],
        "minimum_initial_slack": float(
            (matrix.new_zeros(matrix.shape[0]) - bounds).min().detach().cpu()
        ),
    }


def hard_guard_loss_constraint(
    model: torch.nn.Module,
    guard: TextWindows,
    *,
    parameters: list[torch.nn.Parameter],
    maximum_loss: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Linearize the immutable guard-loss ceiling as a hard inequality."""

    if maximum_loss <= 0.0 or not math.isfinite(maximum_loss):
        raise ValueError("Guard loss ceiling must be positive and finite.")
    logits, _states = forward_with_states(model, guard.inputs.to(device))
    targets = guard.targets.to(device)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    protected_measurement = -loss
    row = flat_autograd_gradient(
        protected_measurement,
        parameters,
        retain_graph=False,
        require_nonzero=True,
        label="hard_guard_loss",
    )
    norm = torch.linalg.vector_norm(row)
    if not torch.isfinite(norm) or float(norm.detach().cpu()) <= 1e-12:
        raise FloatingPointError("Guard loss has an invalid parameter gradient.")
    normalized = row / norm
    lower_bound = (loss.detach() - maximum_loss) / norm
    return normalized.unsqueeze(0), lower_bound.reshape(1).to(normalized), {
        "current_loss": float(loss.detach().cpu()),
        "maximum_loss": maximum_loss,
        "row_norm": float(norm.detach().cpu()),
        "initial_slack": float((-lower_bound).detach().cpu()),
    }


def guard_loss_tensor(
    model: torch.nn.Module,
    guard: TextWindows,
    *,
    device: torch.device,
) -> torch.Tensor:
    logits, _states = forward_with_states(model, guard.inputs.to(device))
    targets = guard.targets.to(device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def retract_to_guard_manifold(
    model: torch.nn.Module,
    guard: TextWindows,
    *,
    parameters: list[torch.nn.Parameter],
    maximum_loss: float,
    radius: float,
    maximum_steps: int,
    maximum_radius_fraction: float,
    safety_margin: float,
    damping: float,
    feasibility_tolerance: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Retract a trial point along successive nonlinear guard normals."""

    if maximum_loss <= 0.0 or not math.isfinite(maximum_loss):
        raise ValueError("Guard retraction requires a positive finite loss ceiling.")
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("Guard retraction requires a positive finite trust radius.")
    if maximum_steps <= 0:
        raise ValueError("Guard retraction maximum steps must be positive.")
    if maximum_radius_fraction <= 0.0 or not math.isfinite(maximum_radius_fraction):
        raise ValueError("Guard retraction radius fraction must be positive and finite.")
    if safety_margin < 0.0 or not math.isfinite(safety_margin):
        raise ValueError("Guard retraction safety margin must be finite and non-negative.")
    if safety_margin >= maximum_loss:
        raise ValueError("Guard retraction safety margin must be below the loss ceiling.")
    if damping <= 0.0 or not math.isfinite(damping):
        raise ValueError("Guard retraction damping must be positive and finite.")
    if feasibility_tolerance < 0.0 or not math.isfinite(feasibility_tolerance):
        raise ValueError(
            "Guard retraction feasibility tolerance must be finite and non-negative."
        )

    parameter_count = sum(parameter.numel() for parameter in parameters)
    correction = torch.zeros(parameter_count, device=device)
    correction_budget = maximum_radius_fraction * radius
    target_loss = maximum_loss - safety_margin
    trial_snapshot = snapshot_parameters(parameters)
    initial_loss = float(guard_loss_tensor(model, guard, device=device).detach().cpu())
    report: dict[str, Any] = {
        "attempted": initial_loss > maximum_loss,
        "success": initial_loss <= maximum_loss,
        "initial_loss": initial_loss,
        "final_loss": initial_loss,
        "maximum_loss": maximum_loss,
        "target_loss": target_loss,
        "feasibility_tolerance": feasibility_tolerance,
        "correction_budget": correction_budget,
        "correction_norm": 0.0,
        "steps": [],
        "failure_reason": None,
    }
    if initial_loss <= maximum_loss:
        return correction, report

    for iteration in range(1, maximum_steps + 1):
        loss = guard_loss_tensor(model, guard, device=device)
        loss_value = float(loss.detach().cpu())
        if loss_value <= target_loss + feasibility_tolerance:
            report["success"] = True
            report["final_loss"] = loss_value
            report["correction_norm"] = float(
                torch.linalg.vector_norm(correction).detach().cpu()
            )
            return correction.detach(), report
        gradient = flat_autograd_gradient(
            loss,
            parameters,
            retain_graph=False,
            require_nonzero=True,
            label=f"guard_retraction_{iteration}",
        )
        gradient_norm_squared = torch.dot(gradient, gradient)
        gradient_norm = torch.sqrt(gradient_norm_squared)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Guard retraction produced a non-finite normal.")
        gap = loss.detach() - target_loss
        normal_step = -(gap / (gradient_norm_squared + damping)) * gradient
        proposed_correction = correction + normal_step
        proposed_norm = float(
            torch.linalg.vector_norm(proposed_correction).detach().cpu()
        )
        step_report = {
            "iteration": iteration,
            "loss_before": loss_value,
            "gap": float(gap.cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "normal_step_norm": float(
                torch.linalg.vector_norm(normal_step).detach().cpu()
            ),
            "proposed_correction_norm": proposed_norm,
        }
        report["steps"].append(step_report)
        if proposed_norm > correction_budget:
            restore_parameters(parameters, trial_snapshot)
            report["failure_reason"] = "correction_budget_exceeded"
            report["final_loss"] = initial_loss
            return torch.zeros_like(correction), report
        apply_flat_delta(parameters, normal_step)
        correction = proposed_correction

    final_loss = float(guard_loss_tensor(model, guard, device=device).detach().cpu())
    if final_loss <= target_loss + feasibility_tolerance:
        report["success"] = True
        report["final_loss"] = final_loss
        report["correction_norm"] = float(
            torch.linalg.vector_norm(correction).detach().cpu()
        )
        return correction.detach(), report
    restore_parameters(parameters, trial_snapshot)
    report["failure_reason"] = "maximum_steps_exhausted"
    report["final_loss"] = initial_loss
    return torch.zeros_like(correction), report


def differentiable_linear_cka(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if current.ndim != 2 or reference.shape != current.shape:
        raise ValueError(
            "CKA tensors must be equally shaped matrices, got "
            f"current={tuple(current.shape)}, reference={tuple(reference.shape)}."
        )
    current_centered = current - current.mean(dim=0, keepdim=True)
    reference_centered = reference - reference.mean(dim=0, keepdim=True)
    cross = current_centered.T @ reference_centered
    current_gram = current_centered.T @ current_centered
    reference_gram = reference_centered.T @ reference_centered
    numerator = cross.square().sum()
    denominator = torch.sqrt(
        current_gram.square().sum() * reference_gram.square().sum()
    ).clamp_min(1e-12)
    return numerator / denominator


def _normalized_relational_gram(projected: torch.Tensor) -> torch.Tensor:
    if projected.ndim != 2 or projected.shape[0] < 2:
        raise ValueError("Projected states must be a matrix with at least two samples.")
    centered = projected - projected.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    norm = torch.linalg.vector_norm(gram)
    if not torch.isfinite(norm) or float(norm.detach().cpu()) <= 1e-12:
        raise FloatingPointError("Readout-sensitive relational Gram matrix is degenerate.")
    return gram / norm


def _functional_pooled_states(states: dict[str, torch.Tensor]) -> torch.Tensor:
    if "final" not in states:
        raise RuntimeError("Transformer state dictionary has no final representation.")
    final = states["final"]
    if final.ndim != 3:
        raise ValueError(f"Final states must be [batch, sequence, width], got {final.shape}.")
    return torch.cat([final.mean(dim=1), final[:, -1]], dim=0)


@torch.no_grad()
def capture_functional_geometry_reference(
    model: torch.nn.Module,
    guard: TextWindows,
    *,
    rank: int,
    device: torch.device,
) -> FunctionalGeometryReference:
    if not hasattr(model, "lm_head") or not hasattr(model.lm_head, "W"):
        raise TypeError("Functional geometry requires a model.lm_head.W readout matrix.")
    weight = model.lm_head.W.detach().cpu()
    if weight.dtype != torch.float32:
        raise TypeError(
            f"Functional geometry expects float32 model weights, got {weight.dtype}."
        )
    width = int(weight.shape[1])
    if not 0 < rank <= width:
        raise ValueError(f"Functional geometry rank must be in [1, {width}], got {rank}.")
    logits, states = forward_with_states(model, guard.inputs.to(device))
    targets = guard.targets.to(device)
    competitors = logits.detach().clone()
    competitors.scatter_(2, targets.unsqueeze(2), float("-inf"))
    competitor_ids = competitors.argmax(dim=2)
    protected_ids = torch.cat([targets.reshape(-1), competitor_ids.reshape(-1)])
    sensitivity_rows = weight[protected_ids.detach().cpu()]
    _left, singular, right = torch.linalg.svd(sensitivity_rows, full_matrices=False)
    numerical_rank = int(
        (singular > singular[0].clamp_min(1e-12) * 1e-6).sum().item()
    )
    if rank > numerical_rank:
        raise ValueError(
            f"Requested functional geometry rank={rank} exceeds protected readout "
            f"numerical rank={numerical_rank}."
        )
    projector = right[:rank].T.to(dtype=torch.float32)
    pooled = _functional_pooled_states(states).detach().cpu()
    if pooled.dtype != torch.float32:
        raise TypeError(
            f"Functional geometry expects float32 hidden states, got {pooled.dtype}."
        )
    gram = _normalized_relational_gram(pooled @ projector)
    return FunctionalGeometryReference(
        projector=projector,
        normalized_gram=gram,
        pooled_states=pooled,
    )


def functional_geometry_measurement(
    model: torch.nn.Module,
    guard: TextWindows,
    reference: FunctionalGeometryReference,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, states = forward_with_states(model, guard.inputs.to(device))
    pooled = _functional_pooled_states(states)
    if pooled.shape != reference.pooled_states.shape:
        raise RuntimeError(
            "Functional geometry sample shape changed: "
            f"current={tuple(pooled.shape)}, reference={tuple(reference.pooled_states.shape)}."
        )
    projector = reference.projector.to(pooled)
    current_gram = _normalized_relational_gram(pooled @ projector)
    reference_gram = reference.normalized_gram.to(current_gram)
    similarity = torch.sum(current_gram * reference_gram)
    distortion = 1.0 - similarity
    state_delta = pooled - reference.pooled_states.to(pooled)
    range_delta = state_delta @ projector @ projector.T
    null_delta = state_delta - range_delta
    reference_norm = torch.linalg.vector_norm(reference.pooled_states.to(pooled)).clamp_min(1e-12)
    report = {
        "guard_loss": float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                guard.targets.to(device).reshape(-1),
            )
            .detach()
            .cpu()
        ),
        "similarity": float(similarity.detach().cpu()),
        "distortion": float(max(0.0, distortion.detach().cpu().item())),
        "range_relative_drift": float(
            (torch.linalg.vector_norm(range_delta) / reference_norm).detach().cpu()
        ),
        "null_relative_drift": float(
            (torch.linalg.vector_norm(null_delta) / reference_norm).detach().cpu()
        ),
    }
    return distortion, report


def functional_geometry_constraints(
    model: torch.nn.Module,
    guard: TextWindows,
    reference: FunctionalGeometryReference,
    *,
    parameters: list[torch.nn.Parameter],
    maximum_distortion: float,
    activation_margin: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not 0.0 < maximum_distortion < 1.0:
        raise ValueError("Functional geometry distortion must be in (0, 1).")
    if not 0.0 < activation_margin < maximum_distortion:
        raise ValueError("Functional geometry activation margin must lie inside the distortion limit.")
    distortion, report = functional_geometry_measurement(
        model,
        guard,
        reference,
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in parameters)
    similarity = 1.0 - distortion
    similarity_floor = 1.0 - maximum_distortion
    if report["distortion"] < maximum_distortion - activation_margin:
        rows = torch.empty(0, parameter_count, device=device)
        bounds = torch.empty(0, device=device)
        active = False
    else:
        row = flat_autograd_gradient(
            similarity,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label="functional_geometry_similarity",
        )
        norm = torch.linalg.vector_norm(row)
        if not torch.isfinite(norm) or float(norm.detach().cpu()) <= 1e-12:
            raise FloatingPointError("Functional geometry constraint has an invalid gradient.")
        normalized = row / norm
        floor_gap = (similarity_floor - similarity.detach()) / norm
        rows = normalized.unsqueeze(0)
        bounds = torch.minimum(floor_gap, torch.zeros_like(floor_gap)).reshape(1).to(rows)
        active = True
    return rows, bounds, distortion, {**report, "active": active}


def build_functional_constraint_basis(
    *,
    model: torch.nn.Module,
    bank: ExecutableTraceBank,
    protection: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    args: argparse.Namespace,
    device: torch.device,
) -> ConstraintBasis:
    """Build behavior and readout-dependency rows without full-state distance rows."""

    _logits, states = forward_with_states(model, bank.inputs.to(device))
    final_states = states["final"][:, -1]
    final_logits = model.lm_head(final_states)
    target_ids = bank.targets[:, -1].to(device)
    reference_logits = bank.reference_logits.to(device)
    competitors = reference_logits.clone()
    competitors.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
    competitor_ids = competitors.argmax(dim=1)
    behavior_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []
    for probe in range(final_logits.shape[0]):
        probe_weight = torch.sqrt(protection[probe])
        for token_id in (target_ids[probe], competitor_ids[probe]):
            scalar = probe_weight * final_logits[probe, token_id]
            behavior_rows.append(
                flat_autograd_gradient(
                    scalar,
                    parameters,
                    retain_graph=True,
                    require_nonzero=False,
                    label=f"functional_behavior_{probe}_{int(token_id)}",
                )
            )
            state_gradient = torch.autograd.grad(
                scalar,
                final_states,
                retain_graph=True,
                allow_unused=False,
            )[0]
            feature_rows.append(state_gradient[probe].detach().to(dtype=torch.float32))
    behavior_matrix = block_normalize(
        behavior_rows,
        block_weight=args.behavior_block_weight,
        label="functional behavior",
    )
    feature_matrix = block_normalize(
        feature_rows,
        block_weight=args.feature_block_weight,
        label="functional readout features",
    )
    _left, feature_singular, feature_right = torch.linalg.svd(
        feature_matrix.detach().to(device="cpu", dtype=torch.float32),
        full_matrices=False,
    )
    feature_rank = energy_rank(feature_singular, args)
    feature_directions = feature_right[:feature_rank].to(final_states)
    feature_strengths = (
        feature_singular[:feature_rank] / feature_singular[0].clamp_min(1e-12)
    ).pow(args.dependency_power).to(final_states)
    normalized_protection = protection / protection.sum().clamp_min(1e-12)
    family_rows: list[torch.Tensor] = []
    for family in range(feature_rank):
        coordinate = (
            final_states @ feature_directions[family] * normalized_protection
        ).sum()
        family_rows.append(
            flat_autograd_gradient(
                feature_strengths[family] * coordinate,
                parameters,
                retain_graph=True,
                require_nonzero=False,
                label=f"functional_feature_family_{family}",
            )
        )
    family_matrix = block_normalize(
        family_rows,
        block_weight=args.feature_block_weight,
        label="functional feature family",
    )
    matrix = torch.cat([behavior_matrix, family_matrix], dim=0)
    gram = (matrix @ matrix.T).detach().to(device="cpu", dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    singular = torch.sqrt(eigenvalues[order].clamp_min(0.0))
    left = eigenvectors[:, order]
    rank = energy_rank(singular, args)
    right = (
        left[:, :rank].T.to(matrix) @ matrix
    ) / singular[:rank].to(matrix).unsqueeze(1).clamp_min(1e-12)
    strengths = (
        singular[:rank] / singular[0].clamp_min(1e-12)
    ).pow(args.dependency_power).to(matrix)
    compressed = strengths.unsqueeze(1) * right
    positive = singular[singular > 0.0]
    probabilities = positive / positive.sum().clamp_min(1e-12)
    effective_rank = float(
        torch.exp(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).item()
    )
    return ConstraintBasis(
        rows=[row for row in compressed],
        measurement_matrix=matrix.detach(),
        report={
            "kind": "readout_sensitive",
            "rows": int(matrix.shape[0]),
            "rank": rank,
            "effective_rank": effective_rank,
            "feature_rank": feature_rank,
        },
    )


def core_geometry_constraints(
    model: torch.nn.Module,
    guard: TextWindows,
    reference_geometry: dict[str, torch.Tensor],
    *,
    parameters: list[torch.nn.Parameter],
    cka_floor: float,
    activation_margin: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Linearize the immutable global CKA floor before it becomes active."""

    if not 0.0 < cka_floor <= 1.0:
        raise ValueError("Core CKA floor must be in (0, 1].")
    if activation_margin <= 0.0 or not math.isfinite(activation_margin):
        raise ValueError("Core CKA activation margin must be positive and finite.")
    _logits, states = forward_with_states(model, guard.inputs.to(device))
    if states.keys() != reference_geometry.keys():
        raise RuntimeError("Core geometry layer sets differ from the immutable reference.")
    rows: list[torch.Tensor] = []
    bounds: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    values: dict[str, float] = {}
    active_layers: list[str] = []
    for layer, state in states.items():
        current = state.reshape(-1, state.shape[-1])
        reference = reference_geometry[layer].to(current)
        cka = differentiable_linear_cka(current, reference)
        losses.append((current - reference).square().mean())
        value = float(cka.detach().cpu())
        values[layer] = value
        if value > cka_floor + activation_margin:
            continue
        row = flat_autograd_gradient(
            cka,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label=f"core_cka_{layer}",
        )
        norm = torch.linalg.vector_norm(row)
        if not torch.isfinite(norm) or norm <= 1e-12:
            raise FloatingPointError(f"Core CKA layer {layer!r} has an invalid gradient.")
        normalized = row / norm
        floor_gap = (cka_floor - cka.detach()) / norm
        lower_bound = torch.minimum(floor_gap, torch.zeros_like(floor_gap))
        rows.append(normalized.detach())
        bounds.append(lower_bound.to(normalized))
        active_layers.append(layer)
    core_loss = torch.stack(losses).mean()
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if rows:
        row_matrix = torch.stack(rows)
        lower_bounds = torch.stack(bounds).to(row_matrix)
    else:
        row_matrix = torch.empty(0, parameter_count, device=device)
        lower_bounds = torch.empty(0, device=device)
    return row_matrix, lower_bounds, core_loss, {
        "cka": values,
        "minimum_cka": min(values.values()),
        "active_layers": active_layers,
    }


def core_geometry_loss(
    model: torch.nn.Module,
    guard: TextWindows,
    reference_geometry: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    _logits, states = forward_with_states(model, guard.inputs.to(device))
    if states.keys() != reference_geometry.keys():
        raise RuntimeError("Core geometry layer sets differ from the immutable reference.")
    losses = [
        (
            state.reshape(-1, state.shape[-1])
            - reference_geometry[layer].to(state)
        ).square().mean()
        for layer, state in states.items()
    ]
    return torch.stack(losses).mean()


def candidate_objective_tensor(
    model: torch.nn.Module,
    current: TextWindows,
    write_weights: torch.Tensor,
    correction_queries: list[FactQuery],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, _states = forward_with_states(model, current.inputs.to(device))
    language = absolute_weighted_language_loss(
        logits,
        current.targets.to(device),
        write_weights.to(device),
        surprise_power=args.token_surprise_power,
    )
    correction, margin, nll = correction_objective(
        model,
        correction_queries,
        args=args,
        device=device,
    )
    objective = language + correction
    return objective, {
        "objective": float(objective.detach().cpu()),
        "language": float(language.detach().cpu()),
        "correction": float(correction.detach().cpu()),
        "margin": float(margin.detach().cpu()),
        "nll": float(nll.detach().cpu()),
    }


def train_unified_microstage(
    *,
    model: torch.nn.Module,
    bank: ExecutableTraceBank,
    current: TextWindows,
    guard: TextWindows,
    original_geometry: dict[str, torch.Tensor],
    functional_geometry: FunctionalGeometryReference | None,
    semantic_memory: SemanticAnchorMemory,
    control: Any,
    correction_queries: list[FactQuery],
    sketch: StreamingConstraintSketch,
    trust_radius: float,
    guard_loss_limit: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], StreamingConstraintSketch, float]:
    """Train one microstage with a single constrained normal-tangent solve."""

    parameters = trainable_weight_parameters(model)
    protection = control.protection_weights.to(device)
    write_weights = control.write_weights.to(device)
    if args.geometry_constraint_mode == "functional_transport":
        basis = build_functional_constraint_basis(
            model=model,
            bank=bank,
            protection=protection,
            parameters=parameters,
            args=args,
            device=device,
        )
    else:
        basis = build_constraint_basis(
            method="dependency_field",
            model=model,
            bank=bank,
            protection=protection,
            parameters=parameters,
            args=args,
            device=device,
        )
    if args.constraint_mode == "hard_soft":
        sketch = update_streaming_sketch(
            sketch,
            basis.measurement_matrix,
            rank=args.soft_sketch_rank,
            decay=args.soft_sketch_decay,
            rank_tolerance=args.dependency_rank_tolerance,
        )
        if sketch.rows is None:
            raise RuntimeError("Streaming constraint sketch did not produce rows.")
        soft_rows = sketch.rows.to(device)
    elif args.constraint_mode == "top_energy":
        soft_rows = torch.stack(basis.rows).to(device)
    else:
        raise ValueError(f"Unknown constraint mode {args.constraint_mode!r}.")

    trace: list[dict[str, Any]] = []
    margin_rows: torch.Tensor | None = None
    margin_bounds: torch.Tensor | None = None
    margin_report: dict[str, Any] = {}
    accepted_step_count = 0
    rejected_guard_step_count = 0
    retraction_attempt_count = 0
    successful_retraction_count = 0
    accepted_retraction_count = 0
    retraction_events: list[dict[str, Any]] = []
    for epoch in range(1, args.cl_epochs + 1):
        if margin_rows is None or (epoch - 1) % args.dependency_refresh == 0:
            if args.constraint_mode == "hard_soft":
                margin_rows, margin_bounds, margin_report = hard_margin_constraints(
                    model,
                    bank,
                    protection,
                    parameters=parameters,
                    hard_slots=args.hard_constraint_slots,
                    margin_tolerance=args.hard_margin_tolerance,
                    device=device,
                )
            else:
                parameter_count = sum(parameter.numel() for parameter in parameters)
                margin_rows = torch.empty(0, parameter_count, device=device)
                margin_bounds = torch.empty(0, device=device)
                margin_report = {"selected_indices": [], "selected_groups": []}

        guard_rows, guard_bounds, guard_report = hard_guard_loss_constraint(
            model,
            guard,
            parameters=parameters,
            maximum_loss=guard_loss_limit,
            device=device,
        )

        if args.geometry_constraint_mode == "fixed_cka":
            core_rows, core_bounds, core_loss, core_report = core_geometry_constraints(
                model,
                guard,
                original_geometry,
                parameters=parameters,
                cka_floor=args.guard_min_cka,
                activation_margin=args.core_geometry_activation_margin,
                device=device,
            )
        elif args.geometry_constraint_mode == "functional_transport":
            if functional_geometry is None:
                raise RuntimeError("Functional transport requires a geometry reference.")
            core_rows, core_bounds, core_loss, core_report = functional_geometry_constraints(
                model,
                guard,
                functional_geometry,
                parameters=parameters,
                maximum_distortion=args.functional_geometry_max_distortion,
                activation_margin=args.functional_geometry_activation_margin,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown geometry constraint mode {args.geometry_constraint_mode!r}."
            )
        semantic_rows, semantic_bounds, semantic_restore, semantic_report = (
            semantic_anchor_constraints(
                model,
                semantic_memory,
                parameters=parameters,
                minimum_margin=args.pending_commit_semantic_margin,
                activation_margin=args.semantic_anchor_activation_margin,
                device=device,
            )
        )
        if args.constraint_mode == "hard_soft":
            hard_rows = torch.cat(
                [margin_rows, guard_rows, core_rows, semantic_rows], dim=0
            )
            hard_bounds = torch.cat(
                [margin_bounds, guard_bounds, core_bounds, semantic_bounds], dim=0
            )
        else:
            hard_rows = torch.cat([guard_rows, semantic_rows], dim=0)
            hard_bounds = torch.cat([guard_bounds, semantic_bounds], dim=0)

        candidate_loss, before_candidate = candidate_objective_tensor(
            model,
            current,
            write_weights,
            correction_queries,
            args=args,
            device=device,
        )
        behavior_loss, state_loss, _logits, _states = protection_losses(
            model,
            bank,
            protection,
            args=args,
            device=device,
        )
        state_restore = (
            args.geometry_restore_weight * state_loss
            if args.geometry_constraint_mode == "fixed_cka"
            else state_loss.new_zeros(())
        )
        protected_loss = (
            behavior_loss
            + state_restore
            + args.core_geometry_restore_weight * core_loss
            + args.semantic_anchor_restore_weight * semantic_restore
        )
        new_gradient = flat_autograd_gradient(
            candidate_loss,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label="unified_candidate",
        )
        restore_gradient = flat_autograd_gradient(
            protected_loss,
            parameters,
            retain_graph=False,
            require_nonzero=False,
            label="unified_restore",
        )
        parameter_snapshot = snapshot_parameters(parameters)
        step = solve_unified_step(
            new_gradient=new_gradient,
            restore_gradient=restore_gradient,
            soft_rows=soft_rows,
            hard_rows=hard_rows,
            hard_lower_bounds=hard_bounds,
            learning_rate=args.cl_lr,
            soft_penalty=args.soft_constraint_penalty,
            soft_restore_fraction=args.soft_restore_fraction,
            trust_radius=trust_radius,
            damping=args.projection_damping,
            feasibility_tolerance=args.constraint_feasibility_tolerance,
            max_active_set_steps=args.max_active_set_steps,
            rank_tolerance=args.dependency_rank_tolerance,
        )
        apply_flat_delta(parameters, step.delta)
        trial_guard_loss = float(
            guard_loss_tensor(model, guard, device=device).detach().cpu()
        )
        retraction_correction = torch.zeros_like(step.delta)
        retraction_report: dict[str, Any] = {
            "enabled": args.guard_manifold_retraction,
            "attempted": False,
            "success": trial_guard_loss <= guard_loss_limit,
            "initial_loss": trial_guard_loss,
            "final_loss": trial_guard_loss,
            "maximum_loss": guard_loss_limit,
            "target_loss": guard_loss_limit,
            "feasibility_tolerance": args.guard_retraction_feasibility_tolerance,
            "correction_budget": 0.0,
            "correction_norm": 0.0,
            "steps": [],
            "failure_reason": None,
        }
        trial_candidate_gain: float | None = None
        if trial_guard_loss > guard_loss_limit and args.guard_manifold_retraction:
            trial_candidate_loss, _trial_candidate = candidate_objective_tensor(
                model,
                current,
                write_weights,
                correction_queries,
                args=args,
                device=device,
            )
            trial_candidate_gain = float(
                (candidate_loss.detach() - trial_candidate_loss.detach()).cpu()
            )
            retraction_attempt_count += 1
            retraction_correction, retraction_report = retract_to_guard_manifold(
                model,
                guard,
                parameters=parameters,
                maximum_loss=guard_loss_limit,
                radius=trust_radius,
                maximum_steps=args.guard_retraction_max_steps,
                maximum_radius_fraction=args.guard_retraction_max_radius_fraction,
                safety_margin=args.guard_retraction_safety_margin,
                damping=args.guard_retraction_damping,
                feasibility_tolerance=args.guard_retraction_feasibility_tolerance,
                device=device,
            )
            retraction_report["enabled"] = True
            if retraction_report["success"]:
                successful_retraction_count += 1
        after_loss, after_candidate = candidate_objective_tensor(
            model,
            current,
            write_weights,
            correction_queries,
            args=args,
            device=device,
        )
        after_behavior, after_state, _after_logits, _after_states = protection_losses(
            model,
            bank,
            protection,
            args=args,
            device=device,
        )
        if args.geometry_constraint_mode == "fixed_cka":
            after_core = core_geometry_loss(
                model,
                guard,
                original_geometry,
                device=device,
            )
            after_state_restore = args.geometry_restore_weight * after_state
            after_guard_loss = evaluate_windows(model, guard, device=device)["loss"]
            nonlinear_geometry_passed = True
            after_functional_distortion = None
        else:
            if functional_geometry is None:
                raise RuntimeError("Functional transport lost its geometry reference.")
            after_core, after_functional_report = functional_geometry_measurement(
                model,
                guard,
                functional_geometry,
                device=device,
            )
            after_state_restore = after_state.new_zeros(())
            after_guard_loss = after_functional_report["guard_loss"]
            after_functional_distortion = after_functional_report["distortion"]
            nonlinear_geometry_passed = (
                after_functional_distortion
                <= args.functional_geometry_max_distortion
            )
        after_semantic_restore = semantic_anchor_restore_loss(
            model,
            semantic_memory,
            minimum_margin=args.pending_commit_semantic_margin,
            activation_margin=args.semantic_anchor_activation_margin,
            device=device,
        )
        after_protected = (
            after_behavior
            + after_state_restore
            + args.core_geometry_restore_weight * after_core
            + args.semantic_anchor_restore_weight * after_semantic_restore
        )
        after_semantic = semantic_anchor_measurement(
            model,
            semantic_memory,
            minimum_margin=args.pending_commit_semantic_margin,
            device=device,
        )
        nonlinear_semantic_passed = after_semantic["passed"]
        actual_gain = float((candidate_loss.detach() - after_loss.detach()).cpu())
        protected_gain = float((protected_loss.detach() - after_protected.detach()).cpu())
        predicted_gain = step.report["predicted_gain"]
        gain_ratio = actual_gain / max(predicted_gain, 1e-12)
        required_retracted_gain = (
            None
            if trial_candidate_gain is None
            else args.guard_retraction_min_gain_retention
            * max(0.0, trial_candidate_gain)
        )
        retraction_gain_passed = (
            required_retracted_gain is None
            or actual_gain >= required_retracted_gain
        )
        gain_filter_passed = (
            actual_gain >= args.filter_min_candidate_gain
            or protected_gain >= args.filter_min_protected_gain
        )
        nonlinear_guard_passed = after_guard_loss <= guard_loss_limit
        final_delta = step.delta + retraction_correction
        final_hard_slack = hard_rows @ final_delta - hard_bounds
        minimum_final_hard_slack = (
            float(final_hard_slack.min().detach().cpu())
            if final_hard_slack.numel()
            else math.inf
        )
        hard_linearization_passed = (
            minimum_final_hard_slack
            >= -args.constraint_feasibility_tolerance
        )
        retraction_passed = (
            not retraction_report["attempted"] or retraction_report["success"]
        )
        filter_passed = (
            gain_filter_passed
            and nonlinear_guard_passed
            and nonlinear_geometry_passed
            and nonlinear_semantic_passed
            and hard_linearization_passed
            and retraction_passed
            and retraction_gain_passed
        )
        if not filter_passed:
            restore_parameters(parameters, parameter_snapshot)
            trust_radius *= args.trust_shrink
            if not nonlinear_guard_passed:
                rejected_guard_step_count += 1
        elif gain_ratio >= args.trust_expand_ratio:
            accepted_step_count += 1
            if retraction_report["attempted"]:
                accepted_retraction_count += 1
            trust_radius = min(args.trust_radius_max, trust_radius * args.trust_expand)
        elif gain_ratio < args.trust_shrink_ratio:
            accepted_step_count += 1
            if retraction_report["attempted"]:
                accepted_retraction_count += 1
            trust_radius *= args.trust_shrink
        else:
            accepted_step_count += 1
            if retraction_report["attempted"]:
                accepted_retraction_count += 1
        if retraction_report["attempted"]:
            retraction_events.append(
                {
                    **retraction_report,
                    "epoch": epoch,
                    "trial_candidate_gain": trial_candidate_gain,
                    "final_candidate_gain": actual_gain,
                    "required_retracted_gain": required_retracted_gain,
                    "gain_retention_passed": retraction_gain_passed,
                    "minimum_final_hard_slack": minimum_final_hard_slack,
                    "hard_linearization_passed": hard_linearization_passed,
                    "functional_distortion": after_functional_distortion,
                    "nonlinear_geometry_passed": nonlinear_geometry_passed,
                    "semantic_anchors_passed": nonlinear_semantic_passed,
                    "accepted_step": filter_passed,
                }
            )
        termination_reason = None
        if trust_radius < args.trust_radius_min:
            trust_radius = args.trust_radius_min
            if not filter_passed:
                termination_reason = "trust_radius_min_after_rejected_step"
        raw_damage = normalized_damage(basis.measurement_matrix, new_gradient)
        delta_damage = normalized_damage(basis.measurement_matrix, final_delta)
        row = {
            "epoch": epoch,
            "accepted_step": filter_passed,
            "gain_filter_passed": gain_filter_passed,
            "nonlinear_guard_passed": nonlinear_guard_passed,
            "nonlinear_geometry_passed": nonlinear_geometry_passed,
            "nonlinear_semantic_passed": nonlinear_semantic_passed,
            "semantic_anchor_measurement": after_semantic,
            "after_functional_distortion": after_functional_distortion,
            "minimum_final_hard_slack": minimum_final_hard_slack,
            "hard_linearization_passed": hard_linearization_passed,
            "after_guard_loss": after_guard_loss,
            "guard_loss_limit": guard_loss_limit,
            "accepted_step_count": accepted_step_count,
            "rejected_guard_step_count": rejected_guard_step_count,
            "retraction_attempt_count": retraction_attempt_count,
            "successful_retraction_count": successful_retraction_count,
            "accepted_retraction_count": accepted_retraction_count,
            "retraction_events": retraction_events,
            "termination_reason": termination_reason,
            "candidate_before": before_candidate,
            "candidate_after": after_candidate,
            "protected_loss_before": float(protected_loss.detach().cpu()),
            "protected_loss_after": float(after_protected.detach().cpu()),
            "actual_gain": actual_gain,
            "protected_gain": protected_gain,
            "gain_ratio": gain_ratio,
            "trial_candidate_gain": trial_candidate_gain,
            "required_retracted_gain": required_retracted_gain,
            "retraction_gain_passed": retraction_gain_passed,
            "guard_retraction": retraction_report,
            "trust_radius": trust_radius,
            "raw_damage": raw_damage,
            "safe_damage": delta_damage,
            "basis": basis.report,
            "hard": {
                "margins": margin_report,
                "guard_loss": guard_report,
                "core_geometry": core_report,
                "semantic_anchors": semantic_report,
            },
            "solver": step.report,
            "safe_grad_fraction": step.report["new_gradient_retained_fraction"],
            "projection_removed_fraction": float(
                max(0.0, 1.0 - step.report["new_gradient_retained_fraction"])
            ),
        }
        if (
            epoch in {1, args.cl_epochs}
            or epoch % args.print_every == 0
            or termination_reason is not None
        ):
            trace.append(row)
        if termination_reason is not None:
            break
    if not trace:
        raise RuntimeError("Unified microstage produced no diagnostics.")
    return trace, sketch, trust_radius


def train_comparison_microstage(
    *,
    operator: str,
    model: torch.nn.Module,
    bank: ExecutableTraceBank,
    current: TextWindows,
    guard: TextWindows,
    original_geometry: dict[str, torch.Tensor],
    control: Any,
    correction_queries: list[FactQuery],
    sketch: StreamingConstraintSketch,
    trust_radius: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], StreamingConstraintSketch, float]:
    if operator not in {"projection_only", "restore_only", "loss_mix", "replay"}:
        raise ValueError(f"Unknown comparison operator {operator!r}.")
    parameters = trainable_weight_parameters(model)
    protection = control.protection_weights.to(device)
    write_weights = control.write_weights.to(device)
    basis = build_constraint_basis(
        method="dependency_field",
        model=model,
        bank=bank,
        protection=protection,
        parameters=parameters,
        args=args,
        device=device,
    )
    constraint_rows = torch.stack(basis.rows).to(device)
    capacity = constraint_rank_report(
        constraint_rows,
        parameter_count=sum(parameter.numel() for parameter in parameters),
        rank_tolerance=args.dependency_rank_tolerance,
    )
    trace: list[dict[str, Any]] = []
    for epoch in range(1, args.cl_epochs + 1):
        candidate_loss, before_candidate = candidate_objective_tensor(
            model,
            current,
            write_weights,
            correction_queries,
            args=args,
            device=device,
        )
        behavior_loss, state_loss, _logits, _states = protection_losses(
            model,
            bank,
            protection,
            args=args,
            device=device,
        )
        core_loss = core_geometry_loss(
            model,
            guard,
            original_geometry,
            device=device,
        )
        protected_loss = (
            behavior_loss
            + args.geometry_restore_weight * state_loss
            + args.core_geometry_restore_weight * core_loss
        )
        raw_gradient = flat_autograd_gradient(
            candidate_loss,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label=f"{operator}_candidate",
        )
        if operator == "projection_only":
            final_gradient, projection = project_gradient_away_from_constraints(
                raw_gradient=raw_gradient,
                constraint_gradients=basis.rows,
                damping=args.projection_damping,
                solver="gram",
                rank_tolerance=args.dependency_rank_tolerance,
                plasticity_audit=False,
            )
        elif operator == "restore_only":
            restore_gradient = flat_autograd_gradient(
                protected_loss,
                parameters,
                retain_graph=False,
                require_nonzero=False,
                label="restore_only_protection",
            )
            final_gradient = raw_gradient + bounded_restore(
                restore_gradient,
                raw_gradient,
                strength=args.restore_strength,
                bound_fraction=args.restore_bound_fraction,
            )
            projection = {
                "safe_grad_fraction": float(
                    (
                        torch.linalg.vector_norm(final_gradient)
                        / torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
                    ).detach().cpu()
                ),
                "projection_removed_fraction": 0.0,
            }
        elif operator == "loss_mix":
            final_gradient = flat_autograd_gradient(
                candidate_loss + args.comparison_preservation_weight * protected_loss,
                parameters,
                retain_graph=False,
                require_nonzero=True,
                label="loss_mix_total",
            )
            projection = {
                "safe_grad_fraction": float(
                    (
                        torch.linalg.vector_norm(final_gradient)
                        / torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
                    ).detach().cpu()
                ),
                "projection_removed_fraction": 0.0,
            }
        else:
            replay_logits, _replay_states = forward_with_states(
                model, bank.inputs.to(device)
            )
            replay_loss = F.cross_entropy(
                replay_logits.reshape(-1, replay_logits.shape[-1]),
                bank.targets.to(device).reshape(-1),
            )
            final_gradient = flat_autograd_gradient(
                candidate_loss + args.comparison_replay_weight * replay_loss,
                parameters,
                retain_graph=False,
                require_nonzero=True,
                label="equal_bank_replay_total",
            )
            projection = {
                "safe_grad_fraction": float(
                    (
                        torch.linalg.vector_norm(final_gradient)
                        / torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
                    ).detach().cpu()
                ),
                "projection_removed_fraction": 0.0,
            }
        delta = -args.cl_lr * final_gradient
        delta_norm = torch.linalg.vector_norm(delta)
        clip_scale = torch.clamp(
            delta.new_tensor(trust_radius) / delta_norm.clamp_min(1e-12),
            max=1.0,
        )
        delta = delta * clip_scale
        apply_flat_delta(parameters, delta)
        after_loss, after_candidate = candidate_objective_tensor(
            model,
            current,
            write_weights,
            correction_queries,
            args=args,
            device=device,
        )
        after_behavior, after_state, _after_logits, _after_states = protection_losses(
            model,
            bank,
            protection,
            args=args,
            device=device,
        )
        after_core = core_geometry_loss(
            model,
            guard,
            original_geometry,
            device=device,
        )
        after_protected = (
            after_behavior
            + args.geometry_restore_weight * after_state
            + args.core_geometry_restore_weight * after_core
        )
        actual_gain = float((candidate_loss.detach() - after_loss.detach()).cpu())
        protected_gain = float((protected_loss.detach() - after_protected.detach()).cpu())
        solver_report = {
            "raw_delta_norm": float(
                torch.linalg.vector_norm(-args.cl_lr * raw_gradient).detach().cpu()
            ),
            "delta_norm": float(torch.linalg.vector_norm(delta).detach().cpu()),
            "trust_clip_scale": float(clip_scale.detach().cpu()),
            "predicted_gain": float((-torch.dot(raw_gradient, delta)).detach().cpu()),
            "active_hard_constraints": 0,
            "hard_constraint_count": 0,
            "soft_constraint_count": int(constraint_rows.shape[0]),
            "minimum_hard_slack": math.inf,
            "new_gradient_retained_fraction": projection["safe_grad_fraction"],
            "capacity": capacity,
        }
        row = {
            "epoch": epoch,
            "accepted_step": True,
            "candidate_before": before_candidate,
            "candidate_after": after_candidate,
            "protected_loss_before": float(protected_loss.detach().cpu()),
            "protected_loss_after": float(after_protected.detach().cpu()),
            "actual_gain": actual_gain,
            "protected_gain": protected_gain,
            "gain_ratio": actual_gain / max(solver_report["predicted_gain"], 1e-12),
            "trust_radius": trust_radius,
            "raw_damage": normalized_damage(basis.measurement_matrix, raw_gradient),
            "safe_damage": normalized_damage(basis.measurement_matrix, delta),
            "basis": basis.report,
            "hard": {"selected_indices": [], "selected_groups": []},
            "solver": solver_report,
            "safe_grad_fraction": projection["safe_grad_fraction"],
            "projection_removed_fraction": projection["projection_removed_fraction"],
        }
        if epoch in {1, args.cl_epochs} or epoch % args.print_every == 0:
            trace.append(row)
    if not trace:
        raise RuntimeError(f"{operator} microstage produced no diagnostics.")
    return trace, sketch, trust_radius


def transactional_update_with_backtracking(
    *,
    model: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
    bank: ExecutableTraceBank,
    current: TextWindows,
    guard: TextWindows,
    original_geometry: dict[str, torch.Tensor],
    moving_geometry: dict[str, torch.Tensor],
    functional_geometry: FunctionalGeometryReference | None,
    semantic_memory: SemanticAnchorMemory,
    control: Any,
    correction_queries: list[FactQuery],
    sketch: StreamingConstraintSketch,
    trust_radius: float,
    base_guard_loss: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    parameter_snapshot = snapshot_parameters(parameters)
    sketch_snapshot = StreamingConstraintSketch(
        rows=None if sketch.rows is None else sketch.rows.clone(),
        updates=sketch.updates,
        observed_rows=sketch.observed_rows,
    )
    candidate_before = candidate_measurement(
        model,
        current,
        control.write_weights,
        correction_queries,
        args=args,
        device=device,
    )
    stable_before = evaluate_windows(model, guard, device=device)["loss"]
    guard_limit = args.guard_loss_ratio * base_guard_loss + args.guard_loss_absolute
    training_function = (
        train_unified_microstage
        if args.update_operator == "unified"
        else train_comparison_microstage
    )
    attempt_radius = trust_radius
    attempts: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    for attempt in range(args.transaction_max_retries + 1):
        if attempt:
            restore_parameters(parameters, parameter_snapshot)
        attempt_sketch = StreamingConstraintSketch(
            rows=None if sketch_snapshot.rows is None else sketch_snapshot.rows.clone(),
            updates=sketch_snapshot.updates,
            observed_rows=sketch_snapshot.observed_rows,
        )
        training_kwargs = {
            "model": model,
            "bank": bank,
            "current": current,
            "guard": guard,
            "original_geometry": original_geometry,
            "control": control,
            "correction_queries": correction_queries,
            "sketch": attempt_sketch,
            "trust_radius": attempt_radius,
            "args": args,
            "device": device,
        }
        if args.update_operator == "unified":
            training_kwargs["functional_geometry"] = functional_geometry
            training_kwargs["semantic_memory"] = semantic_memory
            training_kwargs["guard_loss_limit"] = guard_limit
        else:
            training_kwargs["operator"] = args.update_operator
        training, trained_sketch, trained_radius = training_function(**training_kwargs)
        candidate_after = candidate_measurement(
            model,
            current,
            control.write_weights,
            correction_queries,
            args=args,
            device=device,
        )
        stable_after = evaluate_windows(model, guard, device=device)["loss"]
        candidate_geometry = collect_geometry(model, guard, device=device)
        moving_report = geometry_report(moving_geometry, candidate_geometry)
        original_report = geometry_report(original_geometry, candidate_geometry)
        moving_min_cka = min(layer["cka"] for layer in moving_report.values())
        original_min_cka = min(layer["cka"] for layer in original_report.values())
        if args.geometry_constraint_mode == "functional_transport":
            if functional_geometry is None:
                raise RuntimeError("Functional transport requires a transaction reference.")
            _functional_loss, functional_report = functional_geometry_measurement(
                model,
                guard,
                functional_geometry,
                device=device,
            )
        else:
            functional_report = None
        relative_gain = (
            candidate_before["objective"] - candidate_after["objective"]
        ) / max(abs(candidate_before["objective"]), torch.finfo(torch.float32).eps)
        gain_scale = min(1.0, attempt_radius / trust_radius)
        required_relative_gain = args.commit_min_relative_gain * gain_scale
        guard_passed = stable_after <= guard_limit
        if args.geometry_constraint_mode == "fixed_cka":
            moving_geometry_passed = moving_min_cka >= args.moving_guard_min_cka
            fixed_geometry_passed = original_min_cka >= args.guard_min_cka
            functional_geometry_passed = True
            geometry_passed = moving_geometry_passed and fixed_geometry_passed
        else:
            if functional_report is None:
                raise RuntimeError("Functional geometry measurement is missing.")
            functional_report["maximum_distortion"] = (
                args.functional_geometry_max_distortion
            )
            moving_geometry_passed = True
            fixed_geometry_passed = True
            functional_geometry_passed = (
                functional_report["distortion"]
                <= args.functional_geometry_max_distortion
            )
            geometry_passed = functional_geometry_passed
        correction_passed = (
            not correction_queries
            or candidate_after["margin"] > candidate_before["margin"]
        )
        semantic_anchor_report = semantic_anchor_measurement(
            model,
            semantic_memory,
            minimum_margin=args.pending_commit_semantic_margin,
            device=device,
        )
        semantic_anchors_passed = semantic_anchor_report["passed"]
        accepted = (
            relative_gain >= required_relative_gain
            and guard_passed
            and geometry_passed
            and correction_passed
            and semantic_anchors_passed
        )
        attempt_report = {
            "attempt": attempt + 1,
            "initial_trust_radius": attempt_radius,
            "trained_trust_radius": trained_radius,
            "relative_gain": relative_gain,
            "required_relative_gain": required_relative_gain,
            "stable_after": stable_after,
            "moving_min_cka": moving_min_cka,
            "original_min_cka": original_min_cka,
            "functional_geometry": functional_report,
            "functional_geometry_passed": functional_geometry_passed,
            "guard_passed": guard_passed,
            "geometry_passed": geometry_passed,
            "correction_passed": correction_passed,
            "semantic_anchors": semantic_anchor_report,
            "semantic_anchors_passed": semantic_anchors_passed,
            "accepted": accepted,
        }
        attempts.append(attempt_report)
        final_result = {
            "accepted": accepted,
            "training": training,
            "sketch": trained_sketch,
            "next_trust_radius": trust_radius,
            "accepted_trust_radius": attempt_radius if accepted else None,
            "candidate_before": candidate_before,
            "candidate_after": candidate_after,
            "stable_before": stable_before,
            "stable_after": stable_after,
            "attempted_stable_after": stable_after,
            "stable_guard_limit": guard_limit,
            "moving_report": moving_report,
            "original_report": original_report,
            "moving_min_cka": moving_min_cka,
            "original_min_cka": original_min_cka,
            "functional_geometry": functional_report,
            "relative_gain": relative_gain,
            "required_relative_gain": required_relative_gain,
            "guard_passed": guard_passed,
            "moving_geometry_passed": moving_geometry_passed,
            "fixed_geometry_passed": fixed_geometry_passed,
            "functional_geometry_passed": functional_geometry_passed,
            "geometry_passed": geometry_passed,
            "correction_passed": correction_passed,
            "semantic_anchors": semantic_anchor_report,
            "semantic_anchors_passed": semantic_anchors_passed,
            "candidate_geometry": candidate_geometry,
            "attempts": attempts,
        }
        if accepted:
            final_result["committed_stable_after"] = stable_after
            return final_result
        attempt_radius = max(
            args.trust_radius_min,
            attempt_radius * args.trust_shrink,
        )
    if final_result is None:
        raise RuntimeError("Transactional update produced no attempts.")
    restore_parameters(parameters, parameter_snapshot)
    final_result["sketch"] = sketch_snapshot
    final_result["next_trust_radius"] = trust_radius
    final_result["accepted_trust_radius"] = None
    final_result["stable_after"] = stable_before
    final_result["committed_stable_after"] = stable_before
    return final_result


def validate_long_args(args: argparse.Namespace) -> None:
    validate_scaled_args(args)
    for name in ("hard_constraint_slots", "soft_sketch_rank", "max_active_set_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.durable_slots <= 0 or args.durable_slots > args.trace_slots:
        raise ValueError("--durable-slots must be in [1, trace-slots].")
    if args.transaction_max_retries < 0:
        raise ValueError("--transaction-max-retries must be non-negative.")
    if args.guard_retraction_max_steps <= 0:
        raise ValueError("--guard-retraction-max-steps must be positive.")
    if (
        args.guard_retraction_max_radius_fraction <= 0.0
        or not math.isfinite(args.guard_retraction_max_radius_fraction)
    ):
        raise ValueError(
            "--guard-retraction-max-radius-fraction must be positive and finite."
        )
    if (
        args.guard_retraction_safety_margin < 0.0
        or not math.isfinite(args.guard_retraction_safety_margin)
    ):
        raise ValueError(
            "--guard-retraction-safety-margin must be finite and non-negative."
        )
    if (
        args.guard_retraction_damping <= 0.0
        or not math.isfinite(args.guard_retraction_damping)
    ):
        raise ValueError("--guard-retraction-damping must be positive and finite.")
    if (
        args.guard_retraction_feasibility_tolerance < 0.0
        or not math.isfinite(args.guard_retraction_feasibility_tolerance)
    ):
        raise ValueError(
            "--guard-retraction-feasibility-tolerance must be finite and non-negative."
        )
    if not 0.0 <= args.guard_retraction_min_gain_retention <= 1.0:
        raise ValueError("--guard-retraction-min-gain-retention must be in [0, 1].")
    if args.functional_geometry_rank <= 0:
        raise ValueError("--functional-geometry-rank must be positive.")
    if not 0.0 < args.functional_geometry_max_distortion < 1.0:
        raise ValueError("--functional-geometry-max-distortion must be in (0, 1).")
    if not 0.0 < args.functional_geometry_activation_margin < args.functional_geometry_max_distortion:
        raise ValueError(
            "--functional-geometry-activation-margin must lie inside the distortion limit."
        )
    if (
        args.geometry_constraint_mode == "functional_transport"
        and args.update_operator != "unified"
    ):
        raise ValueError("Functional geometry transport requires --update-operator unified.")
    for name in (
        "cycles",
        "cycle_book_words",
        "cycle_book_windows",
        "cycle_fact_windows",
        "cycle_correction_count",
        "cycle_novel_count",
        "pending_slots",
        "guard_windows",
        "cycle_eval_windows",
        "rare_confirmation_windows",
        "misinformation_windows",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "pending_decay",
        "pending_observation_mass",
        "pending_recurrence_weight",
        "pending_diversity_power",
        "moving_reference_rate",
    ):
        value = getattr(args, name)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if args.pending_decay > 1.0:
        raise ValueError("--pending-decay must not exceed 1.")
    for name in ("durable_fast_decay", "durable_slow_decay"):
        value = getattr(args, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1).")
    if args.durable_minimum_observations <= 0:
        raise ValueError("--durable-minimum-observations must be positive.")
    if not 0.0 < args.durable_minimum_slow_support <= 1.0:
        raise ValueError("--durable-minimum-slow-support must be in (0, 1].")
    if not 0.0 < args.durable_confidence_delta < 1.0:
        raise ValueError("--durable-confidence-delta must be in (0, 1).")
    if args.pending_commit_min_candidates < 2:
        raise ValueError(
            "--pending-commit-min-candidates must be at least two because "
            "trace recurrence is undefined for a single evidence item."
        )
    if args.pending_commit_min_candidates > args.pending_slots:
        raise ValueError(
            "--pending-commit-min-candidates must not exceed --pending-slots."
        )
    if args.pending_commit_min_age < 0:
        raise ValueError("--pending-commit-min-age must be non-negative.")
    if args.pending_commit_min_verifications <= 0:
        raise ValueError("--pending-commit-min-verifications must be positive.")
    for name in (
        "pending_commit_min_mean_support",
        "pending_commit_min_current_support",
        "pending_verification_min_behavior_confidence",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if (
        args.pending_commit_semantic_margin < 0.0
        or not math.isfinite(args.pending_commit_semantic_margin)
    ):
        raise ValueError(
            "--pending-commit-semantic-margin must be finite and non-negative."
        )
    if args.semantic_anchor_slots <= 0:
        raise ValueError("--semantic-anchor-slots must be positive.")
    if args.semantic_candidate_slots <= 0:
        raise ValueError("--semantic-candidate-slots must be positive.")
    for name in ("semantic_anchor_activation_margin",):
        value = getattr(args, name)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative."
            )
    for name in (
        "semantic_anchor_restore_weight",
        "semantic_anchor_priority_temperature",
    ):
        value = getattr(args, name)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    for name in (
        "semantic_anchor_current_priority_weight",
        "semantic_anchor_reference_priority_weight",
        "semantic_anchor_history_priority_weight",
    ):
        value = getattr(args, name)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative."
            )
    if (
        args.semantic_anchor_current_priority_weight
        + args.semantic_anchor_reference_priority_weight
        + args.semantic_anchor_history_priority_weight
        <= 0.0
    ):
        raise ValueError("At least one semantic anchor priority weight must be positive.")
    if args.moving_reference_rate > 1.0:
        raise ValueError("--moving-reference-rate must not exceed 1.")
    if args.hard_constraint_slots > args.trace_slots:
        raise ValueError("--hard-constraint-slots must not exceed --trace-slots.")
    if args.soft_sketch_rank > args.dependency_rank:
        raise ValueError("--soft-sketch-rank must not exceed --dependency-rank.")
    for name in (
        "soft_constraint_penalty",
        "core_geometry_restore_weight",
        "core_geometry_activation_margin",
        "comparison_preservation_weight",
        "comparison_replay_weight",
        "trust_radius",
        "trust_radius_min",
        "trust_radius_max",
        "constraint_feasibility_tolerance",
        "trust_shrink",
        "trust_expand",
        "trust_shrink_ratio",
        "trust_expand_ratio",
    ):
        value = getattr(args, name)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if not args.trust_radius_min <= args.trust_radius <= args.trust_radius_max:
        raise ValueError("Trust radius must lie between its configured minimum and maximum.")
    if not 0.0 < args.trust_shrink < 1.0:
        raise ValueError("--trust-shrink must be in (0, 1).")
    if args.trust_expand <= 1.0:
        raise ValueError("--trust-expand must exceed 1.")
    if not 0.0 < args.soft_sketch_decay <= 1.0:
        raise ValueError("--soft-sketch-decay must be in (0, 1].")
    if args.soft_restore_fraction < 0.0 or not math.isfinite(args.soft_restore_fraction):
        raise ValueError("--soft-restore-fraction must be finite and non-negative.")
    if args.hard_margin_tolerance < 0.0 or not math.isfinite(args.hard_margin_tolerance):
        raise ValueError("--hard-margin-tolerance must be finite and non-negative.")
    for name in ("filter_min_candidate_gain", "filter_min_protected_gain"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite.")
    if args.trust_expand_ratio <= args.trust_shrink_ratio:
        raise ValueError("--trust-expand-ratio must exceed --trust-shrink-ratio.")
    if (
        args.geometry_constraint_mode == "fixed_cka"
        and args.guard_min_cka + args.core_geometry_activation_margin >= 1.0
    ):
        raise ValueError(
            "--guard-min-cka + --core-geometry-activation-margin must be below 1. "
            "Linear CKA has a zero first derivative at perfect equality."
        )
    if not 0.0 < args.moving_guard_min_cka <= 1.0:
        raise ValueError("--moving-guard-min-cka must be in (0, 1].")
    if args.correction_source_start < 0 or args.correction_donor_offset <= 0:
        raise ValueError("Correction source start and donor offset are invalid.")
    if args.rare_fact_index < 0 or args.misinformation_source_index < 0:
        raise ValueError("Rare and misinformation fact indices must be non-negative.")
    if args.rare_confirmation_period <= 0 or args.misinformation_variants <= 1:
        raise ValueError(
            "Rare confirmation period must be positive and misinformation variants must exceed one."
        )
    if args.cycle_novel_start < 0:
        raise ValueError("--cycle-novel-start must be non-negative.")
    if args.cycle_fact_windows < max(
        args.cycle_correction_count, args.cycle_novel_count
    ):
        raise ValueError(
            "--cycle-fact-windows must represent every cycle fact at least once."
        )
    rows = len(fact_sentences()) // 4
    last_source = (
        args.correction_source_start + args.cycles * args.cycle_correction_count
    )
    last_donor = last_source + args.correction_donor_offset
    last_novel = args.cycle_novel_start + args.cycles * args.cycle_novel_count
    last_misinformation = (
        args.misinformation_source_index
        + args.misinformation_donor_offset
        + args.misinformation_variants
    )
    if max(
        last_source,
        last_donor,
        last_novel,
        args.rare_fact_index + 1,
        last_misinformation,
    ) > rows:
        raise ValueError(
            "Requested correction/novel ranges exceed the generated fact corpus: "
            f"source_end={last_source}, donor_end={last_donor}, "
            f"novel_end={last_novel}, blocks={rows}."
        )
    book_words = len(load_book_text(args.book_path).split())
    required_book_words = args.base_book_words + args.cycles * args.cycle_book_words
    if required_book_words > book_words:
        raise ValueError(
            f"Long-horizon stream needs {required_book_words} book words, source has {book_words}."
        )


def correction_block_range(
    *,
    source_start: int,
    donor_offset: int,
    count: int,
    tokenizer: Tokenizer,
) -> tuple[str, str, list[str], list[FactQuery], list[FactQuery]]:
    rows = fact_sentences()
    original_rows: list[str] = []
    corrected_rows: list[str] = []
    corrected_blocks: list[str] = []
    queries: list[FactQuery] = []
    archived_queries: list[FactQuery] = []
    for source_index in range(source_start, source_start + count):
        donor_index = source_index + donor_offset
        source = rows[source_index * 4 : source_index * 4 + 4]
        donor = rows[donor_index * 4 : donor_index * 4 + 4]
        if len(source) != 4 or len(donor) != 4:
            raise RuntimeError(
                f"Missing correction block source={source_index}, donor={donor_index}."
            )
        old_destination = source[3].split("Answer: ", 1)[1].rstrip(".")
        new_destination = donor[3].split("Answer: ", 1)[1].rstrip(".")
        if old_destination == new_destination:
            raise ValueError(
                f"Correction source={source_index} and donor={donor_index} have the same target."
            )
        corrected = [row.replace(old_destination, new_destination) for row in source]
        original_context = "Archived record. " + " ".join(source)
        corrected_context = "Current record. " + " ".join(corrected)
        original_rows.append(original_context)
        corrected_rows.append(corrected_context)
        corrected_blocks.append(corrected_context + " " + original_context)
        person = source[0].split()[0]
        prompt_ids = tokenizer.encode(
            f"Current record. Question: What can {person} open? Answer: the"
        ).ids
        archived_prompt_ids = tokenizer.encode(
            f"Archived record. Question: What can {person} open? Answer: the"
        ).ids
        old_ids = tokenizer.encode(f" {old_destination.removeprefix('the ')}").ids
        new_ids = tokenizer.encode(f" {new_destination.removeprefix('the ')}").ids
        if not prompt_ids or not archived_prompt_ids or not old_ids or not new_ids:
            raise RuntimeError(
                "Tokenizer returned an empty correction query component."
            )
        queries.append(
            distinguishing_fact_query(
                prompt_ids=prompt_ids,
                old_target_ids=old_ids,
                new_target_ids=new_ids,
            )
        )
        archived_queries.append(
            distinguishing_fact_query(
                prompt_ids=archived_prompt_ids,
                old_target_ids=new_ids,
                new_target_ids=old_ids,
            )
        )
    return (
        " ".join(original_rows),
        " ".join(corrected_rows),
        corrected_blocks,
        queries,
        archived_queries,
    )


def novel_block_range(start: int, count: int) -> tuple[str, list[str]]:
    rows = fact_sentences()
    selected = rows[start * 4 : (start + count) * 4]
    if len(selected) != count * 4:
        raise RuntimeError(f"Novel range [{start}, {start + count}) is unavailable.")
    blocks = [" ".join(selected[index * 4 : index * 4 + 4]) for index in range(count)]
    return " ".join(selected), blocks


def distinguishing_fact_query(
    *,
    prompt_ids: list[int],
    old_target_ids: list[int],
    new_target_ids: list[int],
) -> FactQuery:
    """Compare targets at their first differing token under a shared prefix."""

    if not prompt_ids or not old_target_ids or not new_target_ids:
        raise ValueError("A distinguishing query requires non-empty token sequences.")
    shared = 0
    limit = min(len(old_target_ids), len(new_target_ids))
    while shared < limit and old_target_ids[shared] == new_target_ids[shared]:
        shared += 1
    if shared == limit:
        raise ValueError(
            "Target token sequences have no position at which both contain different tokens: "
            f"old={old_target_ids}, new={new_target_ids}."
        )
    return FactQuery(
        input_ids=tuple(prompt_ids + old_target_ids[:shared]),
        old_target_id=int(old_target_ids[shared]),
        new_target_id=int(new_target_ids[shared]),
    )


def fact_block(index: int) -> list[str]:
    rows = fact_sentences()
    block = rows[index * 4 : index * 4 + 4]
    if len(block) != 4:
        raise RuntimeError(f"Fact block {index} is unavailable.")
    return block


def inconsistent_misinformation(
    *,
    cycle: int,
    source_index: int,
    donor_offset: int,
    variants: int,
    tokenizer: Tokenizer,
) -> tuple[str, str, FactQuery]:
    if variants <= 1:
        raise ValueError("Misinformation requires at least two conflicting variants.")
    source = fact_block(source_index)
    donor_index = source_index + donor_offset + cycle % variants
    donor = fact_block(donor_index)
    old_destination = source[3].split("Answer: ", 1)[1].rstrip(".")
    false_destination = donor[3].split("Answer: ", 1)[1].rstrip(".")
    if old_destination == false_destination:
        raise ValueError(
            f"Misinformation source={source_index} and donor={donor_index} have the same target."
        )
    misinformation = [
        row.replace(old_destination, false_destination) for row in source
    ]
    person = source[0].split()[0]
    prompt = tokenizer.encode(
        f"Question: What can {person} open? Answer: the"
    ).ids
    true_ids = tokenizer.encode(f" {old_destination.removeprefix('the ')}").ids
    false_ids = tokenizer.encode(f" {false_destination.removeprefix('the ')}").ids
    if not prompt or not true_ids or not false_ids:
        raise RuntimeError("Misinformation query contains an empty token sequence.")
    false_preference = distinguishing_fact_query(
        prompt_ids=prompt,
        old_target_ids=true_ids,
        new_target_ids=false_ids,
    )
    return " ".join(misinformation), " ".join(source), false_preference


def build_long_horizon_data(
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    vocab_size: int,
) -> tuple[TextWindows, TextWindows, list[CycleData]]:
    book = load_book_text(args.book_path)
    base_book = first_words(book, args.base_book_words)
    base_facts = " ".join(fact_sentences())
    base_candidates = combine_windows(
        [
            windows_from_text(
                base_book,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.base_candidate_windows // 2,
                group="base_book",
            ),
            windows_from_text(
                first_words(base_facts, args.base_fact_words),
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.base_candidate_windows
                - args.base_candidate_windows // 2,
                group="base_fact",
            ),
        ]
    )
    guard = windows_from_text(
        base_book,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        stride=args.stride,
        max_windows=args.guard_windows,
        group="stable_guard",
    )
    cycles: list[CycleData] = []
    for cycle in range(args.cycles):
        book_text = word_span(
            book,
            args.base_book_words + cycle * args.cycle_book_words,
            args.cycle_book_words,
        )
        novel_start = args.cycle_novel_start + cycle * args.cycle_novel_count
        novel_text, novel_blocks = novel_block_range(
            novel_start, args.cycle_novel_count
        )
        correction_start = (
            args.correction_source_start + cycle * args.cycle_correction_count
        )
        original_text, corrected_text, corrected_blocks, queries, archived_queries = (
            correction_block_range(
                source_start=correction_start,
                donor_offset=args.correction_donor_offset,
                count=args.cycle_correction_count,
                tokenizer=tokenizer,
            )
        )
        misinformation_text, misinformation_truth, misinformation_query = (
            inconsistent_misinformation(
                cycle=cycle,
                source_index=args.misinformation_source_index,
                donor_offset=args.misinformation_donor_offset,
                variants=args.misinformation_variants,
                tokenizer=tokenizer,
            )
        )
        rare_text = " ".join(fact_block(args.rare_fact_index))
        book_train = windows_from_text(
            book_text,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.cycle_book_windows,
            group=f"book_c{cycle + 1}",
        )
        novel_train = balanced_block_windows(
            novel_blocks,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.cycle_fact_windows,
            group=f"novel_c{cycle + 1}",
        )
        corrected_train = balanced_block_windows(
            corrected_blocks,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.cycle_fact_windows,
            group=f"corrected_c{cycle + 1}",
        )
        misinformation_train = windows_from_text(
            misinformation_text,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.misinformation_windows,
            group=f"inconsistent_report_c{cycle + 1}",
        )
        training_parts = [book_train, novel_train, corrected_train, misinformation_train]
        if cycle % args.rare_confirmation_period == 0:
            training_parts.append(
                windows_from_text(
                    rare_text,
                    tokenizer=tokenizer,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_windows=args.rare_confirmation_windows,
                    group=f"rare_confirmation_c{cycle + 1}",
                )
            )
        noise = random_noise_windows(
            count=args.noise_windows,
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            seed=args.seed + 100_003 * (cycle + 1),
        )
        train_without_noise = combine_windows(training_parts)
        stream = combine_windows([train_without_noise, noise])
        cycles.append(
            CycleData(
                stream=stream,
                train_without_noise=train_without_noise,
                book_eval=windows_from_text(
                    book_text,
                    tokenizer=tokenizer,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_windows=args.cycle_eval_windows,
                    group=f"book_c{cycle + 1}",
                ),
                novel_eval=windows_from_text(
                    novel_text,
                    tokenizer=tokenizer,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_windows=args.cycle_eval_windows,
                    group=f"novel_c{cycle + 1}",
                ),
                corrected_eval=windows_from_text(
                    corrected_text,
                    tokenizer=tokenizer,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_windows=args.cycle_eval_windows,
                    group=f"corrected_c{cycle + 1}",
                ),
                obsolete_eval=windows_from_text(
                    original_text,
                    tokenizer=tokenizer,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_windows=args.cycle_eval_windows,
                    group=f"obsolete_c{cycle + 1}",
                ),
                correction_queries=tuple(queries),
                archived_queries=tuple(archived_queries),
                rare_eval=windows_from_text(
                    rare_text,
                    tokenizer=tokenizer,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_windows=args.cycle_eval_windows,
                    group="rare_critical",
                ),
                misinformation_eval=windows_from_text(
                    misinformation_text,
                    tokenizer=tokenizer,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_windows=args.cycle_eval_windows,
                    group=f"inconsistent_report_c{cycle + 1}",
                ),
                misinformation_truth_eval=windows_from_text(
                    misinformation_truth,
                    tokenizer=tokenizer,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    max_windows=args.cycle_eval_windows,
                    group="misinformation_truth",
                ),
                misinformation_queries=(misinformation_query,),
            )
        )
    return base_candidates, guard, cycles


def pending_training_windows(
    pool: PendingPool | None, current: TextWindows
) -> TextWindows:
    if pool is None:
        return current
    return combine_windows([pool.windows, current])


@torch.no_grad()
def coherence_conditioned_write(
    model: torch.nn.Module,
    bank: ExecutableTraceBank,
    current: TextWindows,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Require either committed familiarity or coherent independent support."""

    if current.inputs.shape[0] == 0:
        raise ValueError("Coherence-conditioned writing requires candidates.")
    current_vectors = encode_trace_evidence(model, current, args=args, device=device)
    bank_vectors = encode_trace_evidence(
        model,
        TextWindows(bank.inputs, bank.targets, bank.groups),
        args=args,
        device=device,
    )
    familiarity, _error = reconstruction_confidence(
        current_vectors,
        bank_vectors,
        attention_scale=args.attention_scale,
    )
    width = model_width(model)
    states = current_vectors[:, :width]
    targets = F.normalize(current_vectors[:, width : 2 * width], dim=1)
    state_distance = (states[:, None, :] - states[None, :, :]).square().sum(dim=2)
    state_match = torch.exp(
        -state_distance / (2.0 * args.attention_scale**2)
    )
    state_match.fill_diagonal_(0.0)
    target_agreement = ((targets @ targets.T) + 1.0) * 0.5
    target_agreement.fill_diagonal_(0.0)
    support = (state_match * target_agreement).max(dim=1).values
    conflict = (state_match * (1.0 - target_agreement)).max(dim=1).values
    evidence_total = support + conflict
    coherence = torch.where(
        evidence_total > torch.finfo(evidence_total.dtype).eps,
        support / evidence_total.clamp_min(torch.finfo(evidence_total.dtype).eps),
        torch.zeros_like(evidence_total),
    )
    independent_support = support * coherence
    write = 1.0 - (1.0 - familiarity) * (1.0 - independent_support)
    write = write.clamp(0.0, 1.0)
    if not torch.isfinite(write).all():
        raise FloatingPointError("Coherence-conditioned write weights are non-finite.")
    group_report: dict[str, dict[str, float]] = {}
    for group in sorted(set(current.groups)):
        indices = torch.tensor(
            [index for index, value in enumerate(current.groups) if value == group],
            device=device,
        )
        group_report[group] = {
            "familiarity": float(familiarity[indices].mean().detach().cpu()),
            "support": float(support[indices].mean().detach().cpu()),
            "conflict": float(conflict[indices].mean().detach().cpu()),
            "coherence": float(coherence[indices].mean().detach().cpu()),
            "write": float(write[indices].mean().detach().cpu()),
        }
    return write.detach(), {"groups": group_report}


def mature_pending_candidates(
    pool: PendingPool | None,
    memory: TraceTimescaleMemory,
    current_verification: torch.Tensor,
    *,
    minimum_age: int,
    minimum_verifications: int,
    minimum_mean_support: float,
    minimum_current_support: float,
    minimum_candidates: int,
) -> tuple[tuple[TextWindows, torch.Tensor] | None, dict[str, Any]]:
    if minimum_age < 0:
        raise ValueError("Pending commit minimum age must be non-negative.")
    if minimum_verifications <= 0:
        raise ValueError("Pending commit minimum verifications must be positive.")
    if not 0.0 <= minimum_mean_support <= 1.0:
        raise ValueError("Pending commit mean support must be in [0, 1].")
    if not 0.0 <= minimum_current_support <= 1.0:
        raise ValueError("Pending commit current support must be in [0, 1].")
    if minimum_candidates <= 0:
        raise ValueError("Pending commit minimum candidate count must be positive.")
    if pool is None:
        if current_verification.numel() != 0:
            raise ValueError("Empty pending pool has non-empty verification weights.")
        return None, {"eligible": 0, "candidates": 0, "records": []}
    if current_verification.shape != pool.masses.shape:
        raise ValueError("Pending verification weights do not match the pool.")
    eligible: list[int] = []
    records: list[dict[str, Any]] = []
    for index, support_tensor in enumerate(current_verification):
        key = exact_window_key(pool.windows.inputs[index], pool.windows.targets[index])
        record = memory.records.get(key)
        if record is None:
            raise RuntimeError("Pending commitment candidate has no timescale record.")
        mean_support = (
            record.support_sum / record.observations
            if record.observations
            else 0.0
        )
        current_support = float(support_tensor)
        age = int(pool.ages[index])
        passed = (
            age >= minimum_age
            and record.observations >= minimum_verifications
            and mean_support >= minimum_mean_support
            and current_support >= minimum_current_support
        )
        if passed:
            eligible.append(index)
        records.append(
            {
                "index": index,
                "group": pool.windows.groups[index],
                "age": age,
                "observations": record.observations,
                "mean_support": mean_support,
                "current_support": current_support,
                "eligible": passed,
            }
        )
    selected = torch.tensor(eligible, dtype=torch.long)
    report = {
        "eligible": len(eligible),
        "candidates": int(pool.masses.numel()),
        "records": records,
    }
    if selected.numel() < minimum_candidates:
        return None, report
    indices = selected.tolist()
    windows = TextWindows(
        inputs=pool.windows.inputs[selected].clone(),
        targets=pool.windows.targets[selected].clone(),
        groups=tuple(pool.windows.groups[index] for index in indices),
    )
    weights = current_verification[selected].clamp(0.0, 1.0)
    return (windows, weights), report


def remove_committed_pending(
    pool: PendingPool | None,
    bank: ExecutableTraceBank,
) -> PendingPool | None:
    if pool is None:
        return None
    committed = {
        exact_window_key(bank.inputs[index], bank.targets[index])
        for index in range(bank.inputs.shape[0])
    }
    keep = [
        index
        for index in range(pool.windows.inputs.shape[0])
        if exact_window_key(pool.windows.inputs[index], pool.windows.targets[index])
        not in committed
    ]
    if not keep:
        return None
    indices = torch.tensor(keep, dtype=torch.long)
    return PendingPool(
        windows=TextWindows(
            inputs=pool.windows.inputs[indices].clone(),
            targets=pool.windows.targets[indices].clone(),
            groups=tuple(pool.windows.groups[index] for index in keep),
        ),
        masses=pool.masses[indices].clone(),
        ages=pool.ages[indices].clone(),
    )


def exact_window_key(
    inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(int(value) for value in inputs.tolist()), tuple(
        int(value) for value in targets.tolist()
    )


@torch.no_grad()
def update_pending_pool(
    pool: PendingPool | None,
    current: TextWindows,
    current_write_weights: torch.Tensor,
    bank: ExecutableTraceBank,
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PendingPool | None, dict[str, Any]]:
    if current_write_weights.shape != (current.inputs.shape[0],):
        raise ValueError("Current write weights do not match the incoming windows.")
    entries: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}
    if pool is not None:
        for index in range(pool.windows.inputs.shape[0]):
            key = exact_window_key(
                pool.windows.inputs[index], pool.windows.targets[index]
            )
            entries[key] = {
                "inputs": pool.windows.inputs[index].clone(),
                "targets": pool.windows.targets[index].clone(),
                "group": pool.windows.groups[index],
                "mass": float(pool.masses[index]) * args.pending_decay,
                "age": int(pool.ages[index]) + 1,
            }
    for index in range(current.inputs.shape[0]):
        key = exact_window_key(current.inputs[index], current.targets[index])
        observed_mass = args.pending_observation_mass * float(
            current_write_weights[index]
        )
        if key in entries:
            entries[key]["mass"] += observed_mass
            entries[key]["age"] = 0
        else:
            entries[key] = {
                "inputs": current.inputs[index].clone(),
                "targets": current.targets[index].clone(),
                "group": current.groups[index],
                "mass": observed_mass,
                "age": 0,
            }
    committed = {
        exact_window_key(bank.inputs[index], bank.targets[index])
        for index in range(bank.inputs.shape[0])
    }
    for key in committed:
        entries.pop(key, None)
    if not entries:
        return None, {
            "size": 0,
            "mass": 0.0,
            "groups": {},
            "selected_fraction": 0.0,
        }
    ordered = list(entries.values())
    windows = TextWindows(
        inputs=torch.stack([entry["inputs"] for entry in ordered]),
        targets=torch.stack([entry["targets"] for entry in ordered]),
        groups=tuple(entry["group"] for entry in ordered),
    )
    masses = torch.tensor(
        [entry["mass"] for entry in ordered], dtype=torch.float32, device=device
    )
    ages = torch.tensor([entry["age"] for entry in ordered], dtype=torch.long)
    vectors = encode_trace_evidence(model, windows, args=args, device=device)
    recurrence = recurrence_probability(vectors, attention_scale=args.attention_scale)
    bank_vectors = encode_trace_evidence(
        model,
        TextWindows(bank.inputs, bank.targets, bank.groups),
        args=args,
        device=device,
    )
    familiarity, _error = reconstruction_confidence(
        vectors,
        bank_vectors,
        attention_scale=args.attention_scale,
    )
    utility = masses * (
        1.0 + args.pending_recurrence_weight * (recurrence + familiarity)
    )
    capacity = min(args.pending_slots, len(ordered))
    normalized_vectors = F.normalize(vectors, dim=1)
    selected = [int(torch.argmax(utility).item())]
    while len(selected) < capacity:
        similarity = normalized_vectors @ normalized_vectors[selected].T
        distance = (1.0 - similarity.max(dim=1).values).clamp_min(0.0)
        priority = utility * distance.pow(args.pending_diversity_power)
        priority[torch.tensor(selected, device=device)] = -torch.inf
        next_index = int(torch.argmax(priority).item())
        if not torch.isfinite(priority[next_index]):
            raise FloatingPointError(
                "Pending diversity selection produced no finite candidate."
            )
        selected.append(next_index)
    indices = torch.tensor(selected, dtype=torch.long)
    selected_pool = PendingPool(
        windows=TextWindows(
            inputs=windows.inputs[indices].clone(),
            targets=windows.targets[indices].clone(),
            groups=tuple(windows.groups[index] for index in selected),
        ),
        masses=masses[indices.to(device)].detach().cpu(),
        ages=ages[indices].clone(),
    )
    return selected_pool, {
        "size": selected_pool.windows.inputs.shape[0],
        "mass": float(selected_pool.masses.sum()),
        "mean_age": float(selected_pool.ages.float().mean()),
        "groups": dict(Counter(selected_pool.windows.groups)),
        "selected_fraction": float(capacity / len(ordered)),
        "candidate_count": len(ordered),
        "mean_recurrence": float(recurrence.mean().detach().cpu()),
        "mean_familiarity": float(familiarity.mean().detach().cpu()),
    }


def blend_geometry(
    reference: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    *,
    rate: float,
) -> dict[str, torch.Tensor]:
    if reference.keys() != current.keys():
        raise RuntimeError("Moving geometry layer sets differ.")
    blended: dict[str, torch.Tensor] = {}
    for layer in reference:
        if reference[layer].shape != current[layer].shape:
            raise RuntimeError(f"Moving geometry shape changed for layer {layer!r}.")
        blended[layer] = (
            (1.0 - rate) * reference[layer] + rate * current[layer]
        ).detach()
    return blended


def bank_turnover(
    before: ExecutableTraceBank, after: ExecutableTraceBank
) -> dict[str, Any]:
    old = {
        exact_window_key(before.inputs[index], before.targets[index])
        for index in range(before.inputs.shape[0])
    }
    new = {
        exact_window_key(after.inputs[index], after.targets[index])
        for index in range(after.inputs.shape[0])
    }
    if len(old) != before.inputs.shape[0] or len(new) != after.inputs.shape[0]:
        raise RuntimeError("Committed bank contains duplicate executable windows.")
    retained = len(old.intersection(new))
    return {
        "retained": retained,
        "replaced": len(old) - retained,
        "turnover_fraction": float((len(old) - retained) / len(old)),
    }


def aggregate_history(
    cycles: list[CycleData], completed: int
) -> dict[str, TextWindows]:
    if completed <= 0 or completed > len(cycles):
        raise ValueError("Completed cycle count is outside the available history.")
    active = cycles[:completed]
    return {
        "book": combine_windows([cycle.book_eval for cycle in active]),
        "novel": combine_windows([cycle.novel_eval for cycle in active]),
        "corrected": combine_windows([cycle.corrected_eval for cycle in active]),
        "obsolete": combine_windows([cycle.obsolete_eval for cycle in active]),
        "misinformation": combine_windows(
            [cycle.misinformation_eval for cycle in active]
        ),
    }


def evaluate_history(
    model: torch.nn.Module,
    guard: TextWindows,
    cycles: list[CycleData],
    completed: int,
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    history = aggregate_history(cycles, completed)
    return {
        "stable_guard": evaluate_windows(model, guard, device=device),
        "rare_critical": evaluate_windows(
            model, cycles[0].rare_eval, device=device
        ),
        "misinformation_truth": evaluate_windows(
            model, cycles[0].misinformation_truth_eval, device=device
        ),
        **{
            name: evaluate_windows(model, windows, device=device)
            for name, windows in history.items()
        },
    }


def memory_budget_report(
    bank: ExecutableTraceBank,
    pool: PendingPool | None,
    guard: TextWindows,
    original_geometry: dict[str, torch.Tensor],
    moving_geometry: dict[str, torch.Tensor],
    functional_geometry: FunctionalGeometryReference | None,
    *,
    parameter_count: int,
    dependency_rank: int,
    dependency_rows: int,
    timescale_record_count: int,
    verifier_scalars: int,
    semantic_anchor_scalars: int,
    semantic_candidate_scalars: int,
) -> dict[str, int]:
    if dependency_rows <= 0:
        raise ValueError("Dependency row count must be positive.")
    if verifier_scalars < 0:
        raise ValueError("Pending verifier scalar count must be non-negative.")
    if semantic_anchor_scalars < 0:
        raise ValueError("Semantic anchor scalar count must be non-negative.")
    if semantic_candidate_scalars < 0:
        raise ValueError("Semantic candidate scalar count must be non-negative.")
    pending_tokens = 0 if pool is None else int(pool.windows.inputs.numel())
    pending_metadata = (
        0 if pool is None else int(pool.masses.numel() + pool.ages.numel())
    )
    committed_scalars = int(
        bank.inputs.numel()
        + bank.targets.numel()
        + bank.centers.numel()
        + bank.masses.numel()
        + bank.reference_logits.numel()
        + bank.reference_states.numel()
    )
    pending_scalars = pending_tokens * 2 + pending_metadata
    guard_scalars = int(guard.inputs.numel() + guard.targets.numel())
    original_geometry_scalars = sum(
        value.numel() for value in original_geometry.values()
    )
    moving_geometry_scalars = sum(value.numel() for value in moving_geometry.values())
    functional_geometry_scalars = (
        0
        if functional_geometry is None
        else int(
            functional_geometry.projector.numel()
            + functional_geometry.normalized_gram.numel()
            + functional_geometry.pooled_states.numel()
        )
    )
    timescale_metadata_scalars = 5 * timescale_record_count
    return {
        "committed_tokens": int(bank.inputs.numel()),
        "committed_total_scalars": committed_scalars,
        "pending_tokens": pending_tokens,
        "pending_metadata_scalars": pending_metadata,
        "pending_total_scalars": pending_scalars,
        "guard_tokens": int(guard.inputs.numel()),
        "guard_total_scalars": guard_scalars,
        "original_geometry_scalars": original_geometry_scalars,
        "moving_geometry_scalars": moving_geometry_scalars,
        "functional_geometry_scalars": functional_geometry_scalars,
        "timescale_metadata_scalars": timescale_metadata_scalars,
        "pending_verifier_scalars": verifier_scalars,
        "semantic_anchor_scalars": semantic_anchor_scalars,
        "semantic_candidate_scalars": semantic_candidate_scalars,
        "persistent_total_scalars": committed_scalars
        + pending_scalars
        + guard_scalars
        + original_geometry_scalars
        + moving_geometry_scalars
        + functional_geometry_scalars
        + timescale_metadata_scalars
        + verifier_scalars
        + semantic_anchor_scalars
        + semantic_candidate_scalars
        + parameter_count * dependency_rank,
        "compressed_dependency_scalars": int(parameter_count * dependency_rank),
        "dependency_measurement_scalars": int(parameter_count * dependency_rows),
        "maximum_dependency_working_scalars": int(
            parameter_count * (dependency_rank + dependency_rows)
        ),
    }


def pending_verifier_scalar_count(
    verifiers: dict[str, PendingGroupVerifier],
) -> int:
    total = 0
    for verifier in verifiers.values():
        if verifier.mode not in {"required_preference", "trusted_veto"}:
            raise ValueError(f"Unknown pending verifier mode {verifier.mode!r}.")
        total += 1
        total += sum(len(query.input_ids) + 2 for query in verifier.queries)
    return total


def semantic_anchor_scalar_count(memory: SemanticAnchorMemory) -> int:
    return sum(len(anchor.query.input_ids) + 8 for anchor in memory.anchors)


def semantic_candidate_scalar_count(memory: SemanticCandidateMemory) -> int:
    return sum(len(candidate.query.input_ids) + 8 for candidate in memory.candidates)


def plot_long_behavior(cycles: list[dict[str, Any]], output_path: Path) -> None:
    x = [cycle["cycle"] for cycle in cycles]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for name, color in (
        ("stable_guard", "#111827"),
        ("rare_critical", "#7c3aed"),
        ("misinformation_truth", "#be123c"),
        ("book", "#2563eb"),
        ("novel", "#16a34a"),
        ("corrected", "#ea580c"),
    ):
        axes[0].plot(
            x,
            [cycle["evaluation"][name]["loss"] for cycle in cycles],
            marker="o",
            label=name.replace("_", " "),
            color=color,
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Retained behavior as history grows")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].legend()
    axes[1].plot(
        x,
        [cycle["current_loss_before"] for cycle in cycles],
        marker="o",
        label="before cycle",
    )
    axes[1].plot(
        x,
        [cycle["current_loss_after"] for cycle in cycles],
        marker="o",
        label="after cycle",
    )
    axes[1].set_title("Plasticity on each incoming cycle")
    axes[1].set_ylabel("current-cycle loss")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("cycle")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_long_diagnostics(
    cycles: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    output_path: Path,
) -> None:
    cycle_x = [cycle["cycle"] for cycle in cycles]
    update_x = [update["update"] for update in updates]
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 11.0))
    functional_mode = updates[0]["functional_geometry"] is not None
    if functional_mode:
        functional = [update["functional_geometry"] for update in updates]
        if any(value is None for value in functional):
            raise RuntimeError("Functional geometry diagnostics are incomplete.")
        axes[0, 0].plot(
            update_x,
            [value["distortion"] for value in functional],
            label="relational distortion",
        )
        axes[0, 0].plot(
            update_x,
            [value["range_relative_drift"] for value in functional],
            label="readout-range drift",
        )
        axes[0, 0].plot(
            update_x,
            [value["null_relative_drift"] for value in functional],
            label="readout-null drift",
        )
        axes[0, 0].axhline(
            functional[0]["maximum_distortion"],
            color="#111827",
            linestyle="--",
            label="distortion limit",
        )
        axes[0, 0].set_title("Functional-manifold transport")
    else:
        axes[0, 0].plot(
            cycle_x,
            [cycle["original_min_cka"] for cycle in cycles],
            marker="o",
            label="original reference",
        )
        axes[0, 0].plot(
            cycle_x,
            [cycle["moving_min_cka"] for cycle in cycles],
            marker="o",
            label="moving reference",
        )
        axes[0, 0].set_ylim(0.0, 1.02)
        axes[0, 0].set_title("Guard geometry")
    axes[0, 0].legend()
    axes[0, 1].plot(
        update_x,
        [update["safe_grad_fraction"] for update in updates],
        label="safe gradient",
    )
    axes[0, 1].plot(
        update_x,
        [update["projection_removed_fraction"] for update in updates],
        label="removed",
    )
    axes[0, 1].set_title("Plasticity through the unified constrained solve")
    axes[0, 1].legend()
    axes[1, 0].plot(
        cycle_x,
        [cycle["accepted_fraction"] for cycle in cycles],
        marker="o",
        label="accepted",
    )
    axes[1, 0].plot(
        cycle_x,
        [cycle["mean_bank_turnover"] for cycle in cycles],
        marker="o",
        label="bank turnover",
    )
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].set_title("Commit and release")
    axes[1, 0].legend()
    axes[1, 1].plot(
        cycle_x,
        [cycle["pending_size"] for cycle in cycles],
        marker="o",
        label="pending",
    )
    axes[1, 1].plot(
        cycle_x,
        [cycle["committed_size"] for cycle in cycles],
        marker="o",
        label="committed",
    )
    axes[1, 1].plot(
        cycle_x,
        [cycle["durable_size"] for cycle in cycles],
        marker="o",
        label="durable",
    )
    axes[1, 1].set_title("Fixed learner memory")
    axes[1, 1].legend()
    axes[2, 0].plot(
        cycle_x,
        [cycle["correction_preference"]["new_minus_old_margin"] for cycle in cycles],
        marker="o",
        label="current correction",
    )
    axes[2, 0].plot(
        cycle_x,
        [cycle["archived_preference"]["new_minus_old_margin"] for cycle in cycles],
        marker="o",
        label="archived version",
    )
    axes[2, 0].plot(
        cycle_x,
        [
            cycle["misinformation_false_preference"]["new_minus_old_margin"]
            for cycle in cycles
        ],
        marker="o",
        label="false over true",
    )
    if any(cycle["semantic_anchors"]["count"] for cycle in cycles):
        axes[2, 0].plot(
            cycle_x,
            [
                (
                    cycle["semantic_anchors"]["minimum_margin"]
                    if cycle["semantic_anchors"]["count"]
                    else math.nan
                )
                for cycle in cycles
            ],
            marker="o",
            label="protected semantic minimum",
        )
        axes[2, 0].plot(
            cycle_x,
            [
                (
                    min(cycle["semantic_anchors"]["floors"])
                    if cycle["semantic_anchors"]["count"]
                    else math.nan
                )
                for cycle in cycles
            ],
            linestyle="--",
            label="semantic floor",
        )
    axes[2, 0].axhline(0.0, color="#111827", linewidth=1)
    axes[2, 0].set_title("Contextual correction and misinformation")
    axes[2, 0].set_ylabel("target margin")
    axes[2, 0].legend()
    axes[2, 1].plot(
        update_x,
        [update["solver"]["capacity"]["numerical_rank"] for update in updates],
        label="constraint rank",
    )
    axes[2, 1].plot(
        update_x,
        [update["solver"]["active_hard_constraints"] for update in updates],
        label="active hard rows",
    )
    axes[2, 1].set_title("Constraint occupancy")
    axes[2, 1].set_ylabel("directions")
    axes[2, 1].legend()
    update_axes = {id(axes[0, 1]), id(axes[2, 1])}
    if functional_mode:
        update_axes.add(id(axes[0, 0]))
    for axis in axes.flat:
        axis.set_xlabel("micro-update" if id(axis) in update_axes else "cycle")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_long_args(args)
    args.num_slots = args.trace_slots
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = instantiate_model(checkpoint, device)
    parameters = trainable_weight_parameters(model)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if (
        not args.min_trainable_parameters
        <= parameter_count
        <= args.max_trainable_parameters
    ):
        raise ValueError(
            f"Checkpoint has {parameter_count} trainable weights; expected "
            f"[{args.min_trainable_parameters}, {args.max_trainable_parameters}]."
        )
    vocab_size = int(checkpoint["model_config"]["vocab_size"])
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            f"Tokenizer/model vocabulary mismatch: {tokenizer.get_vocab_size()} vs {vocab_size}."
        )
    base_candidates, guard, cycle_data = build_long_horizon_data(
        args,
        tokenizer,
        vocab_size,
    )
    bank = initialize_trace_bank(model, base_candidates, args=args, device=device)
    timescale_memory = initialize_timescale_memory(bank)
    semantic_memory = SemanticAnchorMemory(anchors=[])
    semantic_candidate_memory = SemanticCandidateMemory(candidates=[])
    pool: PendingPool | None = None
    constraint_sketch = StreamingConstraintSketch(
        rows=None,
        updates=0,
        observed_rows=0,
    )
    persistent_dependency_rank = (
        args.soft_sketch_rank
        if args.update_operator == "unified" and args.constraint_mode == "hard_soft"
        else 0
    )
    current_trust_radius = args.trust_radius
    original_geometry = collect_geometry(model, guard, device=device)
    moving_geometry = {name: value.clone() for name, value in original_geometry.items()}
    functional_geometry = (
        capture_functional_geometry_reference(
            model,
            guard,
            rank=args.functional_geometry_rank,
            device=device,
        )
        if args.geometry_constraint_mode == "functional_transport"
        else None
    )
    base_guard_loss = evaluate_windows(model, guard, device=device)["loss"]
    all_queries = [query for cycle in cycle_data for query in cycle.correction_queries]
    all_archived_queries = [
        query for cycle in cycle_data for query in cycle.archived_queries
    ]
    all_misinformation_queries = [
        query for cycle in cycle_data for query in cycle.misinformation_queries
    ]
    initial_core = {
        "stable_guard": evaluate_windows(model, guard, device=device),
        "rare_critical": evaluate_windows(
            model, cycle_data[0].rare_eval, device=device
        ),
        "misinformation_truth": evaluate_windows(
            model, cycle_data[0].misinformation_truth_eval, device=device
        ),
        "misinformation_false_preference": evaluate_correction_queries(
            model,
            all_misinformation_queries,
            device=device,
        ),
    }
    cycle_reports: list[dict[str, Any]] = []
    update_reports: list[dict[str, Any]] = []
    pending_group_verifiers: dict[str, PendingGroupVerifier] = {}
    update_number = 0

    print("1M UNIFIED CONSTRAINED LONG-HORIZON CL")
    print("=" * 160)
    print(
        f"device={device} parameters={parameter_count} width={model_width(model)} "
        f"cycles={args.cycles} committed={args.trace_slots} pending={args.pending_slots} "
        f"guard={guard.inputs.shape[0]} dependency_rank={args.dependency_rank} "
        f"operator={args.update_operator} constraints={args.constraint_mode} "
        f"geometry={args.geometry_constraint_mode} cl_epochs={args.cl_epochs}"
    )
    for cycle_index, cycle in enumerate(cycle_data, start=1):
        semantic_registration_report = register_semantic_candidates(
            model,
            semantic_candidate_memory,
            semantic_memory,
            group=f"corrected_c{cycle_index}",
            queries=cycle.correction_queries,
            update=update_number + 1,
            slots=args.semantic_candidate_slots,
            minimum_margin=args.pending_commit_semantic_margin,
            priority_temperature=args.semantic_anchor_priority_temperature,
            current_priority_weight=args.semantic_anchor_current_priority_weight,
            history_priority_weight=args.semantic_anchor_history_priority_weight,
            device=device,
        )
        pending_group_verifiers[f"corrected_c{cycle_index}"] = PendingGroupVerifier(
            queries=cycle.correction_queries,
            mode="required_preference",
        )
        pending_group_verifiers[
            f"inconsistent_report_c{cycle_index}"
        ] = PendingGroupVerifier(
            queries=cycle.misinformation_queries,
            mode="trusted_veto",
        )
        current_before = evaluate_windows(
            model, cycle.train_without_noise, device=device
        )["loss"]
        shuffled = permute_windows(cycle.stream, seed=args.seed + 10_007 * cycle_index)
        batches = split_windows(shuffled, batch_size=args.micro_batch_windows)
        query_assignments: list[list[FactQuery]] = [list() for _ in batches]
        for query_index, query in enumerate(cycle.correction_queries):
            query_assignments[query_index % len(batches)].append(query)
        cycle_updates: list[dict[str, Any]] = []
        for current, active_queries in zip(batches, query_assignments, strict=True):
            update_number += 1
            started = time.perf_counter()
            incoming_write, incoming_coherence = coherence_conditioned_write(
                model,
                bank,
                current,
                args=args,
                device=device,
            )
            pool, pending_report = update_pending_pool(
                pool,
                current,
                incoming_write.detach().cpu(),
                bank,
                model,
                args=args,
                device=device,
            )
            synchronize_timescale_memory(
                timescale_memory,
                bank,
                pool,
                update=update_number,
            )
            training_windows = pool.windows if pool is not None else current
            survival, consequence = consequence_survival(
                model,
                bank,
                training_windows,
                active_queries,
                args=args,
                device=device,
            )
            update_committed_timescales(
                timescale_memory,
                bank,
                consequence,
                update=update_number,
                fast_decay=args.durable_fast_decay,
                slow_decay=args.durable_slow_decay,
            )
            durable_indices = durable_bank_indices(
                timescale_memory,
                bank,
                slots=args.durable_slots,
                minimum_slow_support=args.durable_minimum_slow_support,
                minimum_observations=args.durable_minimum_observations,
                confidence_delta=args.durable_confidence_delta,
            )
            durable_report: dict[str, Any] = {
                "count": len(durable_indices),
                "groups": [bank.groups[index] for index in durable_indices],
                "lower_scores": [
                    durability_interval(
                        timescale_memory.records[
                            exact_window_key(bank.inputs[index], bank.targets[index])
                        ],
                        confidence_delta=args.durable_confidence_delta,
                    )["lower_score"]
                    for index in durable_indices
                ],
                "replacements": [],
            }
            surviving_bank = bank.clone()
            surviving_bank.masses = surviving_bank.masses * survival.to(
                surviving_bank.masses
            )
            explicit_write, coherence_report = coherence_conditioned_write(
                model,
                surviving_bank,
                training_windows,
                args=args,
                device=device,
            )
            control = propose_trace_update(
                model,
                surviving_bank,
                training_windows,
                stage=update_number + 1,
                args=args,
                device=device,
                write_override=explicit_write,
            )
            control.protection_weights = control.protection_weights * survival.to(
                control.protection_weights
            )
            control.report["survival"] = consequence
            control.report["coherence"] = coherence_report
            control.report["incoming_coherence"] = incoming_coherence
            bank_before = bank.clone()
            transaction = transactional_update_with_backtracking(
                model=model,
                parameters=parameters,
                bank=surviving_bank,
                current=training_windows,
                guard=guard,
                original_geometry=original_geometry,
                moving_geometry=moving_geometry,
                functional_geometry=functional_geometry,
                semantic_memory=semantic_memory,
                control=control,
                correction_queries=active_queries,
                sketch=constraint_sketch,
                trust_radius=current_trust_radius,
                base_guard_loss=base_guard_loss,
                args=args,
                device=device,
            )
            accepted = transaction["accepted"]
            training = transaction["training"]
            constraint_sketch = transaction["sketch"]
            current_trust_radius = transaction["next_trust_radius"]
            candidate_before = transaction["candidate_before"]
            candidate_after = transaction["candidate_after"]
            stable_before = transaction["stable_before"]
            stable_after = transaction["stable_after"]
            attempted_stable_after = transaction["attempted_stable_after"]
            guard_limit = transaction["stable_guard_limit"]
            candidate_geometry = transaction["candidate_geometry"]
            moving_report = transaction["moving_report"]
            original_report = transaction["original_report"]
            moving_min_cka = transaction["moving_min_cka"]
            original_min_cka = transaction["original_min_cka"]
            relative_gain = transaction["relative_gain"]
            guard_passed = transaction["guard_passed"]
            moving_geometry_passed = transaction["moving_geometry_passed"]
            fixed_geometry_passed = transaction["fixed_geometry_passed"]
            functional_geometry_report = transaction["functional_geometry"]
            functional_geometry_passed = transaction["functional_geometry_passed"]
            geometry_passed = transaction["geometry_passed"]
            correction_passed = transaction["correction_passed"]
            semantic_anchor_report = transaction["semantic_anchors"]
            semantic_anchors_passed = transaction["semantic_anchors_passed"]
            verification_weights = torch.zeros_like(pool.masses) if pool is not None else torch.empty(0)
            verification_report: dict[str, Any] = {
                "candidates": 0 if pool is None else int(pool.masses.numel()),
                "verified": 0,
                "groups": {},
                "evaluated": False,
            }
            commitment_report: dict[str, Any] = {
                "eligible": 0,
                "candidates": 0 if pool is None else int(pool.masses.numel()),
                "records": [],
            }
            semantic_promotion_report: dict[str, Any] = {
                "promoted": {},
                "anchor_evicted": {},
                "candidate_evicted": {},
                "anchor_size": len(semantic_memory.anchors),
                "candidate_size": len(semantic_candidate_memory.candidates),
                "anchor_groups": dict(
                    Counter(anchor.group for anchor in semantic_memory.anchors)
                ),
                "candidate_groups": dict(
                    Counter(
                        candidate.group
                        for candidate in semantic_candidate_memory.candidates
                    )
                ),
                "candidates": [],
            }
            if accepted:
                update_semantic_anchor_support(
                    model,
                    semantic_memory,
                    update=update_number,
                    minimum_margin=args.pending_commit_semantic_margin,
                    fast_decay=args.durable_fast_decay,
                    slow_decay=args.durable_slow_decay,
                    device=device,
                )
                semantic_promotion_report = update_and_promote_semantic_candidates(
                    model,
                    semantic_candidate_memory,
                    semantic_memory,
                    update=update_number,
                    candidate_slots=args.semantic_candidate_slots,
                    anchor_slots=args.semantic_anchor_slots,
                    minimum_margin=args.pending_commit_semantic_margin,
                    minimum_age=args.pending_commit_min_age,
                    minimum_verifications=args.pending_commit_min_verifications,
                    minimum_mean_support=args.pending_commit_min_mean_support,
                    priority_temperature=args.semantic_anchor_priority_temperature,
                    current_priority_weight=(
                        args.semantic_anchor_current_priority_weight
                    ),
                    reference_priority_weight=(
                        args.semantic_anchor_reference_priority_weight
                    ),
                    history_priority_weight=(
                        args.semantic_anchor_history_priority_weight
                    ),
                    fast_decay=args.durable_fast_decay,
                    slow_decay=args.durable_slow_decay,
                    device=device,
                )
                verification_weights, verification_report = verify_pending_candidates(
                    model,
                    pool,
                    pending_group_verifiers,
                    minimum_behavior_confidence=(
                        args.pending_verification_min_behavior_confidence
                    ),
                    semantic_margin=args.pending_commit_semantic_margin,
                    device=device,
                )
                verification_report["evaluated"] = True
                reinforce_pending_timescales(
                    timescale_memory,
                    pool,
                    verification_weights,
                    update=update_number,
                    fast_decay=args.durable_fast_decay,
                    slow_decay=args.durable_slow_decay,
                )
                mature, commitment_report = mature_pending_candidates(
                    pool,
                    timescale_memory,
                    verification_weights,
                    minimum_age=args.pending_commit_min_age,
                    minimum_verifications=args.pending_commit_min_verifications,
                    minimum_mean_support=args.pending_commit_min_mean_support,
                    minimum_current_support=args.pending_commit_min_current_support,
                    minimum_candidates=args.pending_commit_min_candidates,
                )
                if mature is not None:
                    mature_windows, mature_weights = mature
                    commit_control = propose_trace_update(
                        model,
                        surviving_bank,
                        mature_windows,
                        stage=update_number + 1,
                        args=args,
                        device=device,
                        write_override=mature_weights.to(device),
                    )
                    bank, durable_report = commit_with_durable_traces(
                        model,
                        bank,
                        commit_control.pending,
                        durable_indices,
                        timescale_memory,
                        durable_slots=args.durable_slots,
                        minimum_slow_support=args.durable_minimum_slow_support,
                        minimum_observations=args.durable_minimum_observations,
                        confidence_delta=args.durable_confidence_delta,
                        device=device,
                    )
                    pool = remove_committed_pending(pool, bank)
                moving_geometry = blend_geometry(
                    moving_geometry,
                    candidate_geometry,
                    rate=args.moving_reference_rate,
                )
                if args.geometry_constraint_mode == "functional_transport":
                    functional_geometry = capture_functional_geometry_reference(
                        model,
                        guard,
                        rank=args.functional_geometry_rank,
                        device=device,
                    )
            synchronize_timescale_memory(
                timescale_memory,
                bank,
                pool,
                update=update_number,
            )
            pending_report["mature_candidates"] = commitment_report["eligible"]
            pending_report["verification"] = verification_report
            pending_report["commitment"] = commitment_report
            pending_report["semantic_promotion"] = semantic_promotion_report
            turnover = bank_turnover(bank_before, bank)
            final_training = training[-1]
            elapsed = time.perf_counter() - started
            report = {
                "cycle": cycle_index,
                "update": update_number,
                "accepted": accepted,
                "next_trust_radius": current_trust_radius,
                "seconds": elapsed,
                "candidate_before": candidate_before,
                "candidate_after": candidate_after,
                "relative_gain": relative_gain,
                "required_relative_gain": transaction["required_relative_gain"],
                "accepted_trust_radius": transaction["accepted_trust_radius"],
                "stable_before": stable_before,
                "stable_after": stable_after,
                "attempted_stable_after": attempted_stable_after,
                "stable_guard_limit": guard_limit,
                "guard_passed": guard_passed,
                "moving_min_cka": moving_min_cka,
                "original_min_cka": original_min_cka,
                "moving_geometry_passed": moving_geometry_passed,
                "fixed_geometry_passed": fixed_geometry_passed,
                "functional_geometry": functional_geometry_report,
                "functional_geometry_passed": functional_geometry_passed,
                "geometry_passed": geometry_passed,
                "correction_passed": correction_passed,
                "semantic_anchors": semantic_anchor_report,
                "semantic_anchors_passed": semantic_anchors_passed,
                "semantic_anchor_memory": semantic_promotion_report,
                "semantic_candidate_registration": semantic_registration_report,
                "transaction_attempts": transaction["attempts"],
                "active_correction_queries": len(active_queries),
                "safe_grad_fraction": final_training["safe_grad_fraction"],
                "projection_removed_fraction": final_training[
                    "projection_removed_fraction"
                ],
                "raw_damage": final_training["raw_damage"],
                "safe_damage": final_training["safe_damage"],
                "accepted_inner_steps": final_training["accepted_step_count"],
                "guard_rejected_inner_steps": final_training[
                    "rejected_guard_step_count"
                ],
                "retraction_attempts": final_training["retraction_attempt_count"],
                "successful_retractions": final_training[
                    "successful_retraction_count"
                ],
                "accepted_retractions": final_training[
                    "accepted_retraction_count"
                ],
                "retraction_events": final_training["retraction_events"],
                "inner_attempts": final_training["epoch"],
                "inner_termination_reason": final_training["termination_reason"],
                "basis": final_training["basis"],
                "solver": final_training["solver"],
                "constraint_sketch": {
                    "updates": constraint_sketch.updates,
                    "observed_rows": constraint_sketch.observed_rows,
                    "stored_rows": (
                        0 if constraint_sketch.rows is None else constraint_sketch.rows.shape[0]
                    ),
                },
                "survival": consequence,
                "trace_control": control.report,
                "pending": pending_report,
                "turnover": turnover,
                "bank_groups": dict(Counter(bank.groups)),
                "durable": {
                    **durable_report,
                    "records": len(timescale_memory.records),
                },
            }
            update_reports.append(report)
            cycle_updates.append(report)
            print(
                f"cycle={cycle_index:02d} update={update_number:03d} accepted={int(accepted)} "
                f"gain={relative_gain:.3f} pending={pending_report['size']:02d} "
                f"turnover={turnover['turnover_fraction']:.3f} "
                f"inner={final_training['accepted_step_count']}/"
                f"{final_training['epoch']} "
                f"retract={final_training['successful_retraction_count']}/"
                f"{final_training['retraction_attempt_count']} "
                f"stop={final_training['termination_reason'] or 'complete'} "
                f"free={final_training['safe_grad_fraction']:.3f} "
                f"hard={final_training['solver']['active_hard_constraints']}/"
                f"{final_training['solver']['hard_constraint_count']} "
                f"cka={moving_min_cka:.4f}/{original_min_cka:.4f} seconds={elapsed:.1f}"
            )
        retained_verifier_groups = (
            set() if pool is None else set(pool.windows.groups)
        )
        pending_group_verifiers = {
            group: verifier
            for group, verifier in pending_group_verifiers.items()
            if group in retained_verifier_groups
        }
        current_after = evaluate_windows(
            model, cycle.train_without_noise, device=device
        )["loss"]
        evaluation = evaluate_history(
            model, guard, cycle_data, cycle_index, device=device
        )
        current_geometry = collect_geometry(model, guard, device=device)
        original_end = geometry_report(original_geometry, current_geometry)
        moving_end = geometry_report(moving_geometry, current_geometry)
        memory = memory_budget_report(
            bank,
            pool,
            guard,
            original_geometry,
            moving_geometry,
            functional_geometry,
            parameter_count=parameter_count,
            dependency_rank=persistent_dependency_rank,
            dependency_rows=max(update["basis"]["rows"] for update in update_reports),
            timescale_record_count=len(timescale_memory.records),
            verifier_scalars=pending_verifier_scalar_count(
                pending_group_verifiers
            ),
            semantic_anchor_scalars=semantic_anchor_scalar_count(
                semantic_memory
            ),
            semantic_candidate_scalars=semantic_candidate_scalar_count(
                semantic_candidate_memory
            ),
        )
        cycle_report = {
            "cycle": cycle_index,
            "updates": len(cycle_updates),
            "accepted": sum(int(update["accepted"]) for update in cycle_updates),
            "accepted_fraction": sum(
                int(update["accepted"]) for update in cycle_updates
            )
            / len(cycle_updates),
            "current_loss_before": current_before,
            "current_loss_after": current_after,
            "evaluation": evaluation,
            "correction_preference": evaluate_correction_queries(
                model,
                all_queries[: cycle_index * args.cycle_correction_count],
                device=device,
            ),
            "archived_preference": evaluate_correction_queries(
                model,
                all_archived_queries[: cycle_index * args.cycle_correction_count],
                device=device,
            ),
            "misinformation_false_preference": evaluate_correction_queries(
                model,
                all_misinformation_queries[:cycle_index],
                device=device,
            ),
            "obsolete_loss": evaluation["obsolete"]["loss"],
            "rare_retention_loss_ratio": (
                evaluation["rare_critical"]["loss"]
                / max(initial_core["rare_critical"]["loss"], 1e-12)
            ),
            "original_geometry": original_end,
            "moving_geometry": moving_end,
            "functional_geometry": cycle_updates[-1]["functional_geometry"],
            "semantic_anchors": semantic_anchor_measurement(
                model,
                semantic_memory,
                minimum_margin=args.pending_commit_semantic_margin,
                device=device,
            ),
            "semantic_candidate_groups": dict(
                Counter(
                    candidate.group
                    for candidate in semantic_candidate_memory.candidates
                )
            ),
            "original_min_cka": min(value["cka"] for value in original_end.values()),
            "moving_min_cka": min(value["cka"] for value in moving_end.values()),
            "mean_safe_grad_fraction": sum(
                update["safe_grad_fraction"] for update in cycle_updates
            )
            / len(cycle_updates),
            "mean_bank_turnover": sum(
                update["turnover"]["turnover_fraction"] for update in cycle_updates
            )
            / len(cycle_updates),
            "committed_size": bank.inputs.shape[0],
            "durable_size": cycle_updates[-1]["durable"]["count"],
            "mean_transaction_attempts": sum(
                len(update["transaction_attempts"]) for update in cycle_updates
            )
            / len(cycle_updates),
            "pending_size": 0 if pool is None else pool.windows.inputs.shape[0],
            "bank_groups": dict(Counter(bank.groups)),
            "pending_groups": (
                {} if pool is None else dict(Counter(pool.windows.groups))
            ),
            "memory": memory,
        }
        cycle_reports.append(cycle_report)
        print(
            f"cycle_end={cycle_index:02d} current={current_before:.4f}->{current_after:.4f} "
            f"guard={evaluation['stable_guard']['loss']:.4f} "
            f"history={evaluation['book']['loss']:.4f}/{evaluation['novel']['loss']:.4f}/"
            f"{evaluation['corrected']['loss']:.4f} accepted={cycle_report['accepted']}/"
            f"{cycle_report['updates']} original_cka={cycle_report['original_min_cka']:.4f}"
        )

    if not cycle_reports or not update_reports:
        raise RuntimeError("Long-horizon run produced no cycle or update reports.")
    persistent_totals = [
        cycle["memory"]["persistent_total_scalars"] for cycle in cycle_reports
    ]
    maximum_verifier_scalars = args.pending_slots * (
        1 + args.cycle_correction_count * (model.max_seq_len + 2)
    )
    maximum_semantic_anchor_scalars = args.semantic_anchor_slots * (
        model.max_seq_len + 8
    )
    maximum_semantic_candidate_scalars = args.semantic_candidate_slots * (
        model.max_seq_len + 8
    )
    maximum_persistent = memory_budget_report(
        bank,
        PendingPool(
            windows=TextWindows(
                inputs=torch.zeros(args.pending_slots, args.seq_len, dtype=torch.long),
                targets=torch.zeros(args.pending_slots, args.seq_len, dtype=torch.long),
                groups=tuple("capacity" for _ in range(args.pending_slots)),
            ),
            masses=torch.zeros(args.pending_slots),
            ages=torch.zeros(args.pending_slots, dtype=torch.long),
        ),
        guard,
        original_geometry,
        moving_geometry,
        functional_geometry,
        parameter_count=parameter_count,
        dependency_rank=persistent_dependency_rank,
        dependency_rows=max(update["basis"]["rows"] for update in update_reports),
        timescale_record_count=args.trace_slots + args.pending_slots,
        verifier_scalars=maximum_verifier_scalars,
        semantic_anchor_scalars=maximum_semantic_anchor_scalars,
        semantic_candidate_scalars=maximum_semantic_candidate_scalars,
    )["persistent_total_scalars"]
    if any(total > maximum_persistent for total in persistent_totals):
        raise RuntimeError(
            "Learner persistent memory exceeded its configured capacity: "
            f"observed={persistent_totals}, maximum={maximum_persistent}."
        )
    final = cycle_reports[-1]
    final_semantic_records = []
    for anchor in semantic_memory.anchors:
        current_margin = float(
            semantic_query_margin_tensor(model, anchor.query, device=device)
            .detach()
            .cpu()
        )
        final_semantic_records.append(
            {
                "group": anchor.group,
                "input_ids": list(anchor.query.input_ids),
                "old_target_id": anchor.query.old_target_id,
                "new_target_id": anchor.query.new_target_id,
                "reference_margin": anchor.reference_margin,
                "floor": semantic_anchor_floor(
                    minimum_margin=args.pending_commit_semantic_margin
                ),
                "current_margin": current_margin,
                "fast_support": anchor.fast_support,
                "slow_support": anchor.slow_support,
                "support_sum": anchor.support_sum,
                "observations": anchor.observations,
                "last_update": anchor.last_update,
            }
        )
    final_semantic_candidate_records = []
    for candidate in semantic_candidate_memory.candidates:
        current_margin = float(
            semantic_query_margin_tensor(model, candidate.query, device=device)
            .detach()
            .cpu()
        )
        final_semantic_candidate_records.append(
            {
                "group": candidate.group,
                "input_ids": list(candidate.query.input_ids),
                "old_target_id": candidate.query.old_target_id,
                "new_target_id": candidate.query.new_target_id,
                "current_margin": current_margin,
                "fast_support": candidate.fast_support,
                "slow_support": candidate.slow_support,
                "support_sum": candidate.support_sum,
                "observations": candidate.observations,
                "first_update": candidate.first_update,
                "last_update": candidate.last_update,
            }
        )
    print("\nFINAL BOUNDED LONG-HORIZON STATE")
    print("-" * 160)
    print(
        f"accepted={sum(update['accepted'] for update in update_reports)}/{len(update_reports)} "
        f"guard_loss={base_guard_loss:.5f}->{final['evaluation']['stable_guard']['loss']:.5f} "
        f"history_book={final['evaluation']['book']['loss']:.5f} "
        f"history_novel={final['evaluation']['novel']['loss']:.5f} "
        f"history_corrected={final['evaluation']['corrected']['loss']:.5f} "
        f"rare_ratio={final['rare_retention_loss_ratio']:.3f} "
        f"original_min_cka={final['original_min_cka']:.4f} "
        f"persistent_scalars={persistent_totals[-1]}/{maximum_persistent}"
    )
    print(
        f"committed={final['committed_size']}/{args.trace_slots} "
        f"durable={final['durable_size']}/{args.durable_slots} "
        f"pending={final['pending_size']}/{args.pending_slots} "
        f"candidates={len(semantic_candidate_memory.candidates)}/{args.semantic_candidate_slots} "
        f"semantic={len(semantic_memory.anchors)}/{args.semantic_anchor_slots} "
        f"dependency_rank_cap={args.dependency_rank} "
        f"correction_margin={final['correction_preference']['new_minus_old_margin']:.4f} "
        f"archive_margin={final['archived_preference']['new_minus_old_margin']:.4f} "
        f"false_margin={final['misinformation_false_preference']['new_minus_old_margin']:.4f}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.output_dir / "long_horizon_behavior.png"
    diagnostics_path = args.output_dir / "long_horizon_diagnostics.png"
    output_json = args.output_dir / "long_horizon_consolidation.json"
    plot_long_behavior(cycle_reports, behavior_path)
    plot_long_diagnostics(cycle_reports, update_reports, diagnostics_path)
    output = {
        "question": (
            "Can the 1M learner continue through repeated unified constrained updates while committed, "
            "pending, guard, and dependency memories remain bounded?"
        ),
        "update_operator": args.update_operator,
        "scope": (
            "Single-seed controlled long-horizon experiment. Evaluation history grows only for offline "
            "measurement and is never visible to the learning operator."
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": {
            "trainable_parameters": parameter_count,
            "vocab_size": vocab_size,
            "layers": len(model.blocks),
            "width": model_width(model),
        },
        "fixed_budgets": {
            "committed_slots": args.trace_slots,
            "pending_slots": args.pending_slots,
            "guard_windows": args.guard_windows,
            "dependency_rank": args.dependency_rank,
            "soft_sketch_rank": persistent_dependency_rank,
            "hard_constraint_slots": args.hard_constraint_slots,
            "durable_slots": args.durable_slots,
            "functional_geometry_rank": (
                args.functional_geometry_rank
                if args.geometry_constraint_mode == "functional_transport"
                else 0
            ),
            "functional_geometry_scalars": final["memory"][
                "functional_geometry_scalars"
            ],
            "maximum_pending_verifier_scalars": maximum_verifier_scalars,
            "semantic_anchor_slots": args.semantic_anchor_slots,
            "maximum_semantic_anchor_scalars": maximum_semantic_anchor_scalars,
            "semantic_candidate_slots": args.semantic_candidate_slots,
            "maximum_semantic_candidate_scalars": maximum_semantic_candidate_scalars,
            "maximum_persistent_scalars": maximum_persistent,
        },
        "base_guard_loss": base_guard_loss,
        "initial_core": initial_core,
        "cycles": cycle_reports,
        "updates": update_reports,
        "final_constraint_sketch": {
            "updates": constraint_sketch.updates,
            "observed_rows": constraint_sketch.observed_rows,
            "stored_rows": (
                0 if constraint_sketch.rows is None else constraint_sketch.rows.shape[0]
            ),
        },
        "final_bank_groups": dict(Counter(bank.groups)),
        "final_pending_groups": (
            {} if pool is None else dict(Counter(pool.windows.groups))
        ),
        "final_timescale_records": timescale_memory_report(
            timescale_memory,
            bank,
            pool,
            confidence_delta=args.durable_confidence_delta,
        ),
        "final_semantic_anchors": final_semantic_records,
        "final_semantic_candidates": final_semantic_candidate_records,
        "plots": {
            "behavior": str(behavior_path),
            "diagnostics": str(diagnostics_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_plots={behavior_path},{diagnostics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = build_scaled_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=Path("model/analysis/gco-1m-long-horizon-consolidation-seed0"),
        cl_epochs=40,
        micro_batch_windows=8,
        noise_windows=2,
        guard_min_cka=0.95,
    )
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--cycle-book-words", type=int, default=500)
    parser.add_argument("--cycle-book-windows", type=int, default=8)
    parser.add_argument("--cycle-fact-windows", type=int, default=8)
    parser.add_argument("--cycle-correction-count", type=int, default=4)
    parser.add_argument("--cycle-novel-start", type=int, default=500)
    parser.add_argument("--cycle-novel-count", type=int, default=4)
    parser.add_argument("--correction-source-start", type=int, default=0)
    parser.add_argument("--correction-donor-offset", type=int, default=101)
    parser.add_argument("--pending-slots", type=int, default=16)
    parser.add_argument("--pending-decay", type=float, default=0.9)
    parser.add_argument("--pending-observation-mass", type=float, default=1.0)
    parser.add_argument("--pending-recurrence-weight", type=float, default=1.0)
    parser.add_argument("--pending-diversity-power", type=float, default=1.0)
    parser.add_argument("--pending-commit-min-candidates", type=int, default=2)
    parser.add_argument("--pending-commit-min-age", type=int, default=2)
    parser.add_argument("--pending-commit-min-verifications", type=int, default=3)
    parser.add_argument("--pending-commit-min-mean-support", type=float, default=0.1)
    parser.add_argument("--pending-commit-min-current-support", type=float, default=0.05)
    parser.add_argument(
        "--pending-verification-min-behavior-confidence",
        type=float,
        default=0.05,
    )
    parser.add_argument("--pending-commit-semantic-margin", type=float, default=1.0)
    parser.add_argument("--semantic-candidate-slots", type=int, default=16)
    parser.add_argument("--semantic-anchor-slots", type=int, default=16)
    parser.add_argument("--semantic-anchor-activation-margin", type=float, default=0.1)
    parser.add_argument("--semantic-anchor-restore-weight", type=float, default=0.25)
    parser.add_argument(
        "--semantic-anchor-priority-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--semantic-anchor-current-priority-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--semantic-anchor-reference-priority-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--semantic-anchor-history-priority-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--durable-slots", type=int, default=4)
    parser.add_argument("--durable-fast-decay", type=float, default=0.8)
    parser.add_argument("--durable-slow-decay", type=float, default=0.98)
    parser.add_argument("--durable-minimum-slow-support", type=float, default=0.1)
    parser.add_argument("--durable-minimum-observations", type=int, default=5)
    parser.add_argument("--durable-confidence-delta", type=float, default=0.05)
    parser.add_argument("--guard-windows", type=int, default=16)
    parser.add_argument("--cycle-eval-windows", type=int, default=8)
    parser.add_argument("--rare-fact-index", type=int, default=40)
    parser.add_argument("--rare-confirmation-period", type=int, default=8)
    parser.add_argument("--rare-confirmation-windows", type=int, default=1)
    parser.add_argument("--misinformation-source-index", type=int, default=60)
    parser.add_argument("--misinformation-donor-offset", type=int, default=211)
    parser.add_argument("--misinformation-variants", type=int, default=7)
    parser.add_argument("--misinformation-windows", type=int, default=1)
    parser.add_argument("--moving-reference-rate", type=float, default=0.25)
    parser.add_argument("--moving-guard-min-cka", type=float, default=0.98)
    parser.add_argument(
        "--geometry-constraint-mode",
        choices=("fixed_cka", "functional_transport"),
        default="fixed_cka",
    )
    parser.add_argument("--functional-geometry-rank", type=int, default=16)
    parser.add_argument("--functional-geometry-max-distortion", type=float, default=0.05)
    parser.add_argument("--functional-geometry-activation-margin", type=float, default=0.01)
    parser.add_argument(
        "--update-operator",
        choices=("unified", "projection_only", "restore_only", "loss_mix", "replay"),
        default="unified",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=("hard_soft", "top_energy"),
        default="hard_soft",
    )
    parser.add_argument("--hard-constraint-slots", type=int, default=4)
    parser.add_argument("--hard-margin-tolerance", type=float, default=0.5)
    parser.add_argument("--soft-sketch-rank", type=int, default=16)
    parser.add_argument("--soft-sketch-decay", type=float, default=0.95)
    parser.add_argument("--soft-constraint-penalty", type=float, default=10.0)
    parser.add_argument("--soft-restore-fraction", type=float, default=0.1)
    parser.add_argument("--core-geometry-restore-weight", type=float, default=0.25)
    parser.add_argument("--core-geometry-activation-margin", type=float, default=0.02)
    parser.add_argument("--trust-radius", type=float, default=0.01)
    parser.add_argument("--trust-radius-min", type=float, default=1e-5)
    parser.add_argument("--trust-radius-max", type=float, default=0.05)
    parser.add_argument("--trust-shrink", type=float, default=0.5)
    parser.add_argument("--trust-expand", type=float, default=1.25)
    parser.add_argument("--trust-shrink-ratio", type=float, default=0.1)
    parser.add_argument("--trust-expand-ratio", type=float, default=0.75)
    parser.add_argument("--constraint-feasibility-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-active-set-steps", type=int, default=32)
    parser.add_argument("--filter-min-candidate-gain", type=float, default=0.0)
    parser.add_argument("--filter-min-protected-gain", type=float, default=0.0)
    parser.add_argument(
        "--guard-manifold-retraction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--guard-retraction-max-steps", type=int, default=6)
    parser.add_argument(
        "--guard-retraction-max-radius-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument("--guard-retraction-safety-margin", type=float, default=1e-6)
    parser.add_argument("--guard-retraction-damping", type=float, default=1e-12)
    parser.add_argument(
        "--guard-retraction-feasibility-tolerance",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--guard-retraction-min-gain-retention",
        type=float,
        default=0.25,
    )
    parser.add_argument("--transaction-max-retries", type=int, default=2)
    parser.add_argument("--comparison-preservation-weight", type=float, default=1.0)
    parser.add_argument("--comparison-replay-weight", type=float, default=1.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
