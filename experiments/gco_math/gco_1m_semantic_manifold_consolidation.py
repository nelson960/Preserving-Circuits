"""Test bounded semantic-manifold consolidation in the 1M text model.

The experiment isolates the plasticity failure observed in the long-horizon
continual-learning loop.  Corrections arrive one query at a time.  Verified
corrections are represented by a fixed budget split between:

* exact semantic margin probes selected by measured constraint pressure and
  Jacobian diversity;
* a deterministic Frequent-Directions sketch of redundant semantic Jacobians.

The update uses hard semantic floors, a logarithmic interior barrier, the
compressed semantic field, a functional-geometry constraint, and nonlinear
joint retraction.  Complete query history is retained only for offline
evaluation and is never read by the learner after consolidation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_1m_consequence_survival_cl import (
    correction_objective,
    restore_parameters,
    snapshot_parameters,
)
from experiments.gco_math.gco_1m_long_horizon_consolidation import (
    FunctionalGeometryReference,
    build_long_horizon_data,
    build_parser as build_long_horizon_parser,
    capture_functional_geometry_reference,
    functional_geometry_constraints,
    functional_geometry_measurement,
    guard_loss_tensor,
    hard_guard_loss_constraint,
    semantic_query_margin_tensor,
    validate_long_args as validate_long_horizon_args,
)
from experiments.gco_math.gco_mini_cl_world_demo import flat_autograd_gradient
from experiments.gco_math.gco_unified_constraint_solver import (
    StreamingConstraintSketch,
    apply_flat_delta,
    constraint_rank_report,
    solve_unified_step,
    update_streaming_sketch,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import trainable_weight_parameters
from experiments.gco_math.gco_tiny_text_dependency_cl import (
    FactQuery,
    TextWindows,
    collect_geometry,
    evaluate_correction_queries,
    geometry_report,
    instantiate_model,
)


@dataclass
class SemanticProbe:
    query: FactQuery
    group: str
    source_update: int
    reference_margin: float
    support_sum: float
    observations: int
    dual_pressure: float


@dataclass
class ConsolidatedSemanticMemory:
    hard: list[SemanticProbe]
    sketch: StreamingConstraintSketch
    observed_queries: int
    compressed_queries: int
    dropped_queries: int


@dataclass
class OfflineQueryAudit:
    query: FactQuery
    group: str
    source_update: int
    committed: bool
    disposition: str
    compression_update: int | None
    compression_row: torch.Tensor | None
    first_incorrect_update: int | None
    first_floor_failure_update: int | None


def query_key(query: FactQuery) -> tuple[tuple[int, ...], int, int]:
    return query.input_ids, query.old_target_id, query.new_target_id


def validate_args(args: argparse.Namespace) -> None:
    validate_long_horizon_args(args)
    if args.semantic_memory_rows <= 1:
        raise ValueError("--semantic-memory-rows must be greater than one.")
    if not 0 < args.semantic_hard_rows <= args.semantic_memory_rows:
        raise ValueError(
            "--semantic-hard-rows must lie in [1, --semantic-memory-rows]."
        )
    if args.semantic_mode == "consolidated" and (
        args.semantic_hard_rows >= args.semantic_memory_rows
    ):
        raise ValueError(
            "Consolidated mode requires at least one semantic sketch row."
        )
    for name in (
        "semantic_floor",
        "semantic_interior_reserve",
        "semantic_barrier_strength",
        "semantic_retraction_radius_fraction",
        "semantic_retraction_damping",
        "semantic_retraction_candidate_retention",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if not 0.0 < args.semantic_dual_decay < 1.0:
        raise ValueError("--semantic-dual-decay must lie in (0, 1).")
    if not 0.0 < args.semantic_sketch_decay <= 1.0:
        raise ValueError("--semantic-sketch-decay must lie in (0, 1].")
    if not 0.0 < args.semantic_retraction_candidate_retention <= 1.0:
        raise ValueError(
            "--semantic-retraction-candidate-retention must lie in (0, 1]."
        )
    if args.semantic_retraction_steps <= 0:
        raise ValueError("--semantic-retraction-steps must be positive.")
    if args.semantic_retraction_tolerance < 0.0 or not math.isfinite(
        args.semantic_retraction_tolerance
    ):
        raise ValueError(
            "--semantic-retraction-tolerance must be finite and non-negative."
        )
    if not 0.0 < args.semantic_guard_safety < args.guard_loss_absolute:
        raise ValueError(
            "--semantic-guard-safety must lie inside --guard-loss-absolute."
        )
    if not (
        0.0
        < args.semantic_geometry_safety
        < args.functional_geometry_max_distortion
    ):
        raise ValueError(
            "--semantic-geometry-safety must lie inside the functional geometry limit."
        )


def semantic_margin_and_row(
    model: torch.nn.Module,
    query: FactQuery,
    parameters: list[torch.nn.Parameter],
    *,
    device: torch.device,
    label: str,
) -> tuple[float, torch.Tensor, float]:
    margin = semantic_query_margin_tensor(model, query, device=device)
    gradient = flat_autograd_gradient(
        margin,
        parameters,
        retain_graph=False,
        require_nonzero=True,
        label=label,
    )
    norm = torch.linalg.vector_norm(gradient)
    norm_value = float(norm.detach().cpu())
    if not math.isfinite(norm_value) or norm_value <= 1e-12:
        raise FloatingPointError(f"{label} produced an invalid semantic Jacobian.")
    return float(margin.detach().cpu()), (gradient / norm).detach(), norm_value


def hard_semantic_constraints(
    model: torch.nn.Module,
    memory: ConsolidatedSemanticMemory,
    parameters: list[torch.nn.Parameter],
    *,
    floor: float,
    reserve: float,
    barrier_strength: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if not memory.hard:
        return (
            torch.empty(0, parameter_count, device=device),
            torch.empty(0, device=device),
            torch.zeros((), device=device),
            {"count": 0, "margins": [], "slacks": [], "minimum_slack": None},
        )

    rows: list[torch.Tensor] = []
    bounds: list[torch.Tensor] = []
    margins: list[float] = []
    slacks: list[float] = []
    for index, probe in enumerate(memory.hard):
        margin = semantic_query_margin_tensor(model, probe.query, device=device)
        gradient = flat_autograd_gradient(
            margin,
            parameters,
            retain_graph=False,
            require_nonzero=True,
            label=f"semantic_hard_{index}",
        )
        norm = torch.linalg.vector_norm(gradient)
        norm_value = float(norm.detach().cpu())
        if not math.isfinite(norm_value) or norm_value <= 1e-12:
            raise FloatingPointError("Hard semantic probe has an invalid Jacobian.")
        rows.append((gradient / norm).detach())
        bounds.append(((floor - margin.detach()) / norm).to(gradient))
        margin_value = float(margin.detach().cpu())
        margins.append(margin_value)
        slacks.append(margin_value - floor)

    if min(slacks) <= 0.0:
        raise RuntimeError(
            "A hard semantic probe reached or crossed its floor before the update: "
            f"minimum_slack={min(slacks):.8g}."
        )
    barrier_terms = []
    for probe in memory.hard:
        margin = semantic_query_margin_tensor(model, probe.query, device=device)
        slack = margin - floor
        if float(slack.detach().cpu()) <= 0.0:
            raise RuntimeError("Semantic logarithmic barrier received non-positive slack.")
        barrier_terms.append(-torch.log(slack / reserve))
    barrier = barrier_strength * torch.stack(barrier_terms).mean()
    return (
        torch.stack(rows),
        torch.stack(bounds).to(rows[0]),
        barrier,
        {
            "count": len(memory.hard),
            "margins": margins,
            "slacks": slacks,
            "minimum_slack": min(slacks),
        },
    )


@torch.no_grad()
def semantic_memory_measurement(
    model: torch.nn.Module,
    memory: ConsolidatedSemanticMemory,
    *,
    floor: float,
    device: torch.device,
) -> dict[str, Any]:
    margins = [
        float(semantic_query_margin_tensor(model, probe.query, device=device).cpu())
        for probe in memory.hard
    ]
    slacks = [margin - floor for margin in margins]
    return {
        "count": len(margins),
        "passed": not slacks or min(slacks) >= 0.0,
        "margins": margins,
        "slacks": slacks,
        "minimum_margin": None if not margins else min(margins),
        "minimum_slack": None if not slacks else min(slacks),
    }


def greedy_diverse_hard_set(
    rows: torch.Tensor,
    priorities: torch.Tensor,
    *,
    slots: int,
    tolerance: float,
) -> list[int]:
    if rows.ndim != 2 or priorities.shape != (rows.shape[0],):
        raise ValueError("Semantic selection rows and priorities do not align.")
    if slots <= 0:
        raise ValueError("Semantic hard-set capacity must be positive.")
    if not torch.isfinite(rows).all() or not torch.isfinite(priorities).all():
        raise FloatingPointError("Semantic selection received non-finite values.")
    selected: list[int] = []
    basis: list[torch.Tensor] = []
    available = set(range(rows.shape[0]))
    while available and len(selected) < slots:
        best_index: int | None = None
        best_score = -math.inf
        best_residual: torch.Tensor | None = None
        for index in sorted(available):
            residual = rows[index].clone()
            for direction in basis:
                residual = residual - torch.dot(residual, direction) * direction
            residual_norm = float(torch.linalg.vector_norm(residual).detach().cpu())
            score = float(priorities[index].detach().cpu()) * residual_norm
            if score > best_score:
                best_score = score
                best_index = index
                best_residual = residual
        if best_index is None or best_residual is None:
            raise RuntimeError("Semantic diversity selection failed to choose a row.")
        residual_norm = torch.linalg.vector_norm(best_residual)
        if float(residual_norm.detach().cpu()) <= tolerance:
            break
        selected.append(best_index)
        available.remove(best_index)
        basis.append((best_residual / residual_norm).detach())
    return selected


def row_space_coverage(row: torch.Tensor, basis_rows: torch.Tensor | None) -> float:
    if row.ndim != 1 or not torch.isfinite(row).all():
        raise ValueError("Row-space coverage requires one finite row.")
    row_norm_squared = torch.dot(row, row)
    if float(row_norm_squared.detach().cpu()) <= 1e-12:
        raise FloatingPointError("Cannot measure coverage of a zero row.")
    if basis_rows is None:
        return 0.0
    if basis_rows.ndim != 2 or basis_rows.shape[1] != row.numel():
        raise ValueError("Coverage basis does not match the semantic row width.")
    basis = basis_rows.to(row)
    gram = (basis @ basis.T).detach().cpu().to(dtype=torch.float64)
    overlap = (basis @ row).detach().cpu().to(dtype=torch.float64)
    coefficients = torch.linalg.lstsq(gram, overlap).solution
    projected_norm_squared = torch.dot(overlap, coefficients).clamp_min(0.0)
    coverage = float(
        (projected_norm_squared / row_norm_squared.detach().cpu().to(torch.float64))
        .clamp(0.0, 1.0)
        .item()
    )
    if not math.isfinite(coverage):
        raise FloatingPointError("Semantic row-space coverage is non-finite.")
    return coverage


def consolidate_probe(
    model: torch.nn.Module,
    memory: ConsolidatedSemanticMemory,
    probe: SemanticProbe,
    parameters: list[torch.nn.Parameter],
    dependency_rows: torch.Tensor,
    *,
    mode: str,
    memory_rows: int,
    hard_rows: int,
    floor: float,
    reserve: float,
    sketch_decay: float,
    rank_tolerance: float,
    device: torch.device,
) -> dict[str, Any]:
    existing = {query_key(item.query): item for item in memory.hard}
    existing[query_key(probe.query)] = probe
    candidates = list(existing.values())
    rows: list[torch.Tensor] = []
    margins: list[float] = []
    hard_priorities: list[float] = []
    sketch_priorities: list[float] = []
    components: list[dict[str, float]] = []
    for index, candidate in enumerate(candidates):
        margin, row, _norm = semantic_margin_and_row(
            model,
            candidate.query,
            parameters,
            device=device,
            label=f"semantic_consolidation_{index}",
        )
        slack = margin - floor
        if slack <= 0.0:
            raise RuntimeError(
                "Cannot consolidate a semantic probe outside its feasible interior: "
                f"margin={margin:.6g}, floor={floor:.6g}."
            )
        mean_support = candidate.support_sum / max(candidate.observations, 1)
        boundary = reserve / (slack + reserve)
        dual = 1.0 + candidate.dual_pressure
        dependency = 1.0 + float(
            torch.linalg.vector_norm(dependency_rows @ row).detach().cpu()
        )
        sketch_coverage = row_space_coverage(row, memory.sketch.rows)
        novelty = 1.0 - sketch_coverage
        hard_priority = boundary * mean_support * dual * dependency
        sketch_priority = mean_support * dependency * novelty
        if not math.isfinite(hard_priority) or hard_priority <= 0.0:
            raise FloatingPointError("Semantic hard-selection priority is invalid.")
        if not math.isfinite(sketch_priority) or sketch_priority < 0.0:
            raise FloatingPointError("Semantic sketch-write priority is invalid.")
        rows.append(row)
        margins.append(margin)
        hard_priorities.append(hard_priority)
        sketch_priorities.append(sketch_priority)
        components.append(
            {
                "boundary": boundary,
                "support": mean_support,
                "dual": dual,
                "dependency": dependency,
                "sketch_coverage": sketch_coverage,
                "novelty": novelty,
                "hard_priority": hard_priority,
                "sketch_priority": sketch_priority,
            }
        )

    matrix = torch.stack(rows)
    priority_tensor = matrix.new_tensor(hard_priorities)
    selected_capacity = memory_rows if mode == "exact" else hard_rows
    selected = greedy_diverse_hard_set(
        matrix,
        priority_tensor,
        slots=selected_capacity,
        tolerance=rank_tolerance,
    )
    selected_set = set(selected)
    nonselected = [index for index in range(len(candidates)) if index not in selected_set]
    memory.hard = [candidates[index] for index in selected]
    compressed = 0
    dropped = 0
    compressed_records: list[tuple[int, torch.Tensor]] = []
    if nonselected:
        if mode == "consolidated":
            sketch_rank = memory_rows - hard_rows
            rows_to_write = [
                math.sqrt(sketch_priorities[index]) * rows[index]
                for index in nonselected
                if sketch_priorities[index] > rank_tolerance
            ]
            if rows_to_write:
                memory.sketch = update_streaming_sketch(
                    memory.sketch,
                    torch.stack(rows_to_write),
                    rank=sketch_rank,
                    decay=sketch_decay,
                    rank_tolerance=rank_tolerance,
                )
            compressed_records = [
                (candidates[index].source_update, rows[index].detach().cpu())
                for index in nonselected
            ]
            compressed = len(nonselected)
            memory.compressed_queries += compressed
        elif mode == "exact":
            dropped = len(nonselected)
            memory.dropped_queries += dropped
        else:
            raise ValueError(f"Unknown semantic memory mode {mode!r}.")
    memory.observed_queries += 1
    return {
        "candidate_count": len(candidates),
        "selected_indices": selected,
        "selected_updates": [candidates[index].source_update for index in selected],
        "selected_groups": [candidates[index].group for index in selected],
        "compressed_updates": [
            candidates[index].source_update for index in nonselected
        ] if mode == "consolidated" else [],
        "dropped_updates": [
            candidates[index].source_update for index in nonselected
        ] if mode == "exact" else [],
        "compressed": compressed,
        "dropped": dropped,
        "margins": margins,
        "components": components,
        "hard_size": len(memory.hard),
        "sketch_rows": 0 if memory.sketch.rows is None else memory.sketch.rows.shape[0],
        "_compressed_records": compressed_records,
    }


def update_probe_support_and_duals(
    model: torch.nn.Module,
    memory: ConsolidatedSemanticMemory,
    *,
    floor: float,
    dual_decay: float,
    active_indices: list[int],
    active_multipliers: list[float],
    semantic_offset: int,
    device: torch.device,
) -> None:
    if len(active_indices) != len(active_multipliers):
        raise ValueError("Active constraint indices and multipliers do not align.")
    observed_duals = [0.0 for _probe in memory.hard]
    for index, multiplier in zip(active_indices, active_multipliers, strict=True):
        semantic_index = index - semantic_offset
        if 0 <= semantic_index < len(memory.hard):
            observed_duals[semantic_index] = math.log1p(max(0.0, multiplier))
    for index, probe in enumerate(memory.hard):
        margin = float(
            semantic_query_margin_tensor(model, probe.query, device=device)
            .detach()
            .cpu()
        )
        support = float(margin >= floor)
        probe.support_sum += support
        probe.observations += 1
        probe.dual_pressure = (
            dual_decay * probe.dual_pressure
            + (1.0 - dual_decay) * observed_duals[index]
        )


def joint_manifold_retraction(
    model: torch.nn.Module,
    guard: TextWindows,
    geometry: FunctionalGeometryReference,
    memory: ConsolidatedSemanticMemory,
    candidate: FactQuery,
    parameters: list[torch.nn.Parameter],
    *,
    guard_limit: float,
    geometry_limit: float,
    semantic_floor: float,
    semantic_reserve: float,
    candidate_floor: float,
    trust_radius: float,
    radius_fraction: float,
    guard_safety: float,
    geometry_safety: float,
    damping: float,
    maximum_steps: int,
    tolerance: float,
    device: torch.device,
) -> dict[str, Any]:
    correction = torch.zeros(sum(parameter.numel() for parameter in parameters), device=device)
    budget = radius_fraction * trust_radius
    trial_snapshot = snapshot_parameters(parameters)
    steps: list[dict[str, Any]] = []

    for iteration in range(1, maximum_steps + 1):
        measurements: list[tuple[str, torch.Tensor, float]] = []
        guard_loss = guard_loss_tensor(model, guard, device=device)
        guard_target = guard_limit - guard_safety
        if float(guard_loss.detach().cpu()) > guard_target + tolerance:
            measurements.append(("guard", -guard_loss, -guard_target))

        distortion, _geometry_report = functional_geometry_measurement(
            model,
            guard,
            geometry,
            device=device,
        )
        geometry_target = geometry_limit - geometry_safety
        if float(distortion.detach().cpu()) > geometry_target + tolerance:
            measurements.append(("geometry", -distortion, -geometry_target))

        semantic_target = semantic_floor + semantic_reserve
        for index, probe in enumerate(memory.hard):
            margin = semantic_query_margin_tensor(model, probe.query, device=device)
            if float(margin.detach().cpu()) < semantic_target - tolerance:
                measurements.append((f"semantic_{index}", margin, semantic_target))

        candidate_margin = semantic_query_margin_tensor(model, candidate, device=device)
        if float(candidate_margin.detach().cpu()) < candidate_floor - tolerance:
            measurements.append(("candidate", candidate_margin, candidate_floor))

        if not measurements:
            return {
                "success": True,
                "iterations": iteration - 1,
                "correction_norm": float(torch.linalg.vector_norm(correction).cpu()),
                "budget": budget,
                "steps": steps,
            }

        rows: list[torch.Tensor] = []
        deficits: list[torch.Tensor] = []
        names: list[str] = []
        for position, (name, measurement, target) in enumerate(measurements):
            gradient = flat_autograd_gradient(
                measurement,
                parameters,
                retain_graph=position < len(measurements) - 1,
                require_nonzero=True,
                label=f"joint_retraction_{name}_{iteration}",
            )
            rows.append(gradient)
            deficits.append(measurement.new_tensor(target) - measurement.detach())
            names.append(name)
        matrix = torch.stack(rows)
        deficit = torch.stack(deficits).to(matrix)
        gram = (matrix @ matrix.T).detach().cpu().to(dtype=torch.float64)
        rhs = deficit.detach().cpu().to(dtype=torch.float64)
        identity = torch.eye(gram.shape[0], dtype=torch.float64)
        coefficients = torch.linalg.solve(gram + damping * identity, rhs).to(matrix)
        normal_step = matrix.T @ coefficients
        if not torch.isfinite(normal_step).all():
            raise FloatingPointError("Joint manifold retraction produced a non-finite step.")
        proposed = correction + normal_step
        proposed_norm = float(torch.linalg.vector_norm(proposed).detach().cpu())
        steps.append(
            {
                "iteration": iteration,
                "measurements": names,
                "maximum_deficit": float(deficit.max().detach().cpu()),
                "step_norm": float(torch.linalg.vector_norm(normal_step).detach().cpu()),
                "proposed_norm": proposed_norm,
            }
        )
        if proposed_norm > budget:
            restore_parameters(parameters, trial_snapshot)
            return {
                "success": False,
                "failure": "correction_budget_exceeded",
                "iterations": iteration,
                "correction_norm": 0.0,
                "budget": budget,
                "steps": steps,
            }
        apply_flat_delta(parameters, normal_step)
        correction = proposed

    restore_parameters(parameters, trial_snapshot)
    return {
        "success": False,
        "failure": "maximum_steps_exhausted",
        "iterations": maximum_steps,
        "correction_norm": 0.0,
        "budget": budget,
        "steps": steps,
    }


def train_query(
    model: torch.nn.Module,
    query: FactQuery,
    guard: TextWindows,
    geometry: FunctionalGeometryReference,
    memory: ConsolidatedSemanticMemory,
    *,
    guard_limit: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    parameters = trainable_weight_parameters(model)
    initial_objective, initial_margin, _initial_nll = correction_objective(
        model,
        [query],
        args=args,
        device=device,
    )
    initial_objective_value = float(initial_objective.detach().cpu())
    initial_margin_value = float(initial_margin.detach().cpu())
    trust_radius = args.trust_radius
    accepted_steps = 0
    rejected_steps = 0
    retraction_successes = 0
    traces: list[dict[str, Any]] = []

    for epoch in range(1, args.cl_epochs + 1):
        guard_rows, guard_bounds, guard_report = hard_guard_loss_constraint(
            model,
            guard,
            parameters=parameters,
            maximum_loss=guard_limit,
            device=device,
        )
        geometry_rows, geometry_bounds, geometry_loss, geometry_report = (
            functional_geometry_constraints(
                model,
                guard,
                geometry,
                parameters=parameters,
                maximum_distortion=args.functional_geometry_max_distortion,
                activation_margin=args.functional_geometry_activation_margin,
                device=device,
            )
        )
        semantic_rows, semantic_bounds, barrier, semantic_report = (
            hard_semantic_constraints(
                model,
                memory,
                parameters,
                floor=args.semantic_floor,
                reserve=args.semantic_interior_reserve,
                barrier_strength=args.semantic_barrier_strength,
                device=device,
            )
        )
        hard_rows = torch.cat([guard_rows, geometry_rows, semantic_rows], dim=0)
        hard_bounds = torch.cat([guard_bounds, geometry_bounds, semantic_bounds], dim=0)
        soft_blocks = [guard_rows]
        if geometry_rows.shape[0]:
            soft_blocks.append(geometry_rows)
        if memory.sketch.rows is not None:
            soft_blocks.append(memory.sketch.rows.to(device))
        soft_rows = torch.cat(soft_blocks, dim=0)

        candidate_loss, before_margin, _before_nll = correction_objective(
            model,
            [query],
            args=args,
            device=device,
        )
        restore_loss = (
            guard_loss_tensor(model, guard, device=device)
            + args.core_geometry_restore_weight * geometry_loss
            + barrier
        )
        new_gradient = flat_autograd_gradient(
            candidate_loss,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label="semantic_candidate",
        )
        restore_gradient = flat_autograd_gradient(
            restore_loss,
            parameters,
            retain_graph=False,
            require_nonzero=True,
            label="semantic_interior_restore",
        )
        snapshot = snapshot_parameters(parameters)
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
        trial_margin = float(
            semantic_query_margin_tensor(model, query, device=device).detach().cpu()
        )
        before_margin_value = float(before_margin.detach().cpu())
        retained_gain = args.semantic_retraction_candidate_retention * max(
            0.0, trial_margin - before_margin_value
        )
        candidate_floor = before_margin_value + retained_gain
        retraction = joint_manifold_retraction(
            model,
            guard,
            geometry,
            memory,
            query,
            parameters,
            guard_limit=guard_limit,
            geometry_limit=args.functional_geometry_max_distortion,
            semantic_floor=args.semantic_floor,
            semantic_reserve=args.semantic_interior_reserve,
            candidate_floor=candidate_floor,
            trust_radius=trust_radius,
            radius_fraction=args.semantic_retraction_radius_fraction,
            guard_safety=args.semantic_guard_safety,
            geometry_safety=args.semantic_geometry_safety,
            damping=args.semantic_retraction_damping,
            maximum_steps=args.semantic_retraction_steps,
            tolerance=args.semantic_retraction_tolerance,
            device=device,
        )
        after_loss, after_margin, _after_nll = correction_objective(
            model,
            [query],
            args=args,
            device=device,
        )
        after_measurement = semantic_memory_measurement(
            model,
            memory,
            floor=args.semantic_floor,
            device=device,
        )
        final_guard = float(guard_loss_tensor(model, guard, device=device).detach().cpu())
        final_distortion, final_geometry = functional_geometry_measurement(
            model,
            guard,
            geometry,
            device=device,
        )
        improved = float(after_loss.detach().cpu()) < float(candidate_loss.detach().cpu())
        feasible = (
            retraction["success"]
            and final_guard <= guard_limit + args.semantic_retraction_tolerance
            and float(final_distortion.detach().cpu())
            <= args.functional_geometry_max_distortion
            + args.semantic_retraction_tolerance
            and after_measurement["passed"]
            and float(after_margin.detach().cpu())
            >= candidate_floor - args.semantic_retraction_tolerance
        )
        accepted = improved and feasible
        if accepted:
            accepted_steps += 1
            if retraction["iterations"]:
                retraction_successes += 1
            trust_radius = min(args.trust_radius_max, trust_radius * args.trust_expand)
            semantic_offset = guard_rows.shape[0] + geometry_rows.shape[0]
            update_probe_support_and_duals(
                model,
                memory,
                floor=args.semantic_floor,
                dual_decay=args.semantic_dual_decay,
                active_indices=step.report["active_hard_indices"],
                active_multipliers=step.report["multipliers"],
                semantic_offset=semantic_offset,
                device=device,
            )
        else:
            restore_parameters(parameters, snapshot)
            rejected_steps += 1
            trust_radius *= args.trust_shrink
            if trust_radius < args.trust_radius_min:
                traces.append(
                    {
                        "epoch": epoch,
                        "accepted": False,
                        "termination": "trust_radius_below_minimum",
                        "solver": step.report,
                        "retraction": retraction,
                    }
                )
                break
        traces.append(
            {
                "epoch": epoch,
                "accepted": accepted,
                "candidate_loss": float(candidate_loss.detach().cpu()),
                "after_loss": float(after_loss.detach().cpu()),
                "before_margin": before_margin_value,
                "after_margin": float(after_margin.detach().cpu()),
                "guard": final_guard,
                "geometry": final_geometry,
                "hard_semantic": semantic_report,
                "solver": step.report,
                "retraction": retraction,
                "trust_radius": trust_radius,
            }
        )

    final_objective, final_margin, _final_nll = correction_objective(
        model,
        [query],
        args=args,
        device=device,
    )
    final_objective_value = float(final_objective.detach().cpu())
    final_margin_value = float(final_margin.detach().cpu())
    return {
        "initial_objective": initial_objective_value,
        "final_objective": final_objective_value,
        "relative_gain": (
            initial_objective_value - final_objective_value
        )
        / max(abs(initial_objective_value), 1e-12),
        "initial_margin": initial_margin_value,
        "final_margin": final_margin_value,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "retraction_successes": retraction_successes,
        "committable": final_margin_value
        >= args.semantic_floor + args.semantic_interior_reserve,
        "trace": traces,
    }


def semantic_sketch_report(
    memory: ConsolidatedSemanticMemory,
    *,
    parameter_count: int,
    tolerance: float,
) -> dict[str, Any]:
    if memory.sketch.rows is None:
        return {
            "rows": 0,
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "free_parameter_fraction": 1.0,
        }
    return constraint_rank_report(
        memory.sketch.rows,
        parameter_count=parameter_count,
        rank_tolerance=tolerance,
    )


@torch.no_grad()
def update_offline_failure_audit(
    model: torch.nn.Module,
    audits: list[OfflineQueryAudit],
    *,
    update: int,
    floor: float,
    device: torch.device,
) -> None:
    for audit in audits:
        if not audit.committed:
            continue
        margin = float(
            semantic_query_margin_tensor(model, audit.query, device=device).cpu()
        )
        if margin < 0.0 and audit.first_incorrect_update is None:
            audit.first_incorrect_update = update
        if margin < floor and audit.first_floor_failure_update is None:
            audit.first_floor_failure_update = update


def final_offline_audit(
    model: torch.nn.Module,
    audits: list[OfflineQueryAudit],
    memory: ConsolidatedSemanticMemory,
    parameters: list[torch.nn.Parameter],
    *,
    floor: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    hard_updates = {probe.source_update for probe in memory.hard}
    records: list[dict[str, Any]] = []
    for audit in audits:
        margin = float(
            semantic_query_margin_tensor(model, audit.query, device=device)
            .detach()
            .cpu()
        )
        coverage: float | None = None
        transport_cosine: float | None = None
        if audit.committed:
            _margin, final_row, _norm = semantic_margin_and_row(
                model,
                audit.query,
                parameters,
                device=device,
                label=f"offline_final_audit_{audit.source_update}",
            )
            coverage = row_space_coverage(final_row, memory.sketch.rows)
            if audit.compression_row is not None:
                compression_row = audit.compression_row.to(final_row)
                denominator = (
                    torch.linalg.vector_norm(compression_row)
                    * torch.linalg.vector_norm(final_row)
                )
                if float(denominator.detach().cpu()) <= 1e-12:
                    raise FloatingPointError(
                        "Offline transport audit encountered a zero Jacobian."
                    )
                transport_cosine = float(
                    (torch.dot(compression_row, final_row) / denominator)
                    .clamp(-1.0, 1.0)
                    .detach()
                    .cpu()
                )
        disposition = audit.disposition
        if audit.source_update in hard_updates:
            disposition = "hard"
        records.append(
            {
                "source_update": audit.source_update,
                "group": audit.group,
                "committed": audit.committed,
                "disposition": disposition,
                "compression_update": audit.compression_update,
                "final_margin": margin,
                "correct": margin >= 0.0,
                "above_floor": margin >= floor,
                "first_incorrect_update": audit.first_incorrect_update,
                "first_floor_failure_update": audit.first_floor_failure_update,
                "final_sketch_coverage": coverage,
                "jacobian_transport_cosine": transport_cosine,
            }
        )
    return records


def plot_results(stages: list[dict[str, Any]], output_path: Path) -> None:
    updates = [stage["update"] for stage in stages]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    axes[0].plot(
        updates,
        [stage["history"]["new_over_old_fraction"] for stage in stages],
        marker="o",
        color="#0f766e",
        label="verified corrections",
    )
    axes[0].plot(
        updates,
        [float(stage["committed"]) for stage in stages],
        marker="s",
        color="#c2410c",
        label="current correction committed",
    )
    axes[0].set(title="Semantic learning", xlabel="update", ylabel="fraction / decision")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=False)

    axes[1].plot(
        updates,
        [stage["mean_retained_gradient"] for stage in stages],
        marker="o",
        color="#2563eb",
        label="retained gradient",
    )
    axes[1].plot(
        updates,
        [stage["hard_size"] / stage["memory_rows"] for stage in stages],
        marker="s",
        color="#7c3aed",
        label="hard capacity",
    )
    axes[1].plot(
        updates,
        [stage["sketch"]["rows"] / stage["memory_rows"] for stage in stages],
        marker="^",
        color="#64748b",
        label="sketch capacity",
    )
    axes[1].set(title="Plasticity and bounded memory", xlabel="update", ylabel="fraction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(frameon=False)

    axes[2].plot(
        updates,
        [stage["guard_loss"] for stage in stages],
        marker="o",
        color="#111827",
        label="guard loss",
    )
    axes[2].plot(
        updates,
        [stage["geometry_distortion"] for stage in stages],
        marker="s",
        color="#16a34a",
        label="geometry distortion",
    )
    axes[2].plot(
        updates,
        [
            0.0
            if stage["hard_semantic"]["minimum_slack"] is None
            else stage["hard_semantic"]["minimum_slack"]
            for stage in stages
        ],
        marker="^",
        color="#dc2626",
        label="minimum semantic slack",
    )
    axes[2].set(title="Protected manifold", xlabel="update", ylabel="measurement")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = instantiate_model(checkpoint, device)
    parameters = trainable_weight_parameters(model)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    vocab_size = int(checkpoint["model_config"]["vocab_size"])
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.get_vocab_size()} vs {vocab_size}."
        )
    _base, guard, cycles = build_long_horizon_data(args, tokenizer, vocab_size)
    queries = [
        (f"corrected_c{cycle_index}", query)
        for cycle_index, cycle in enumerate(cycles, start=1)
        for query in cycle.correction_queries
    ]
    if not queries:
        raise RuntimeError("Semantic consolidation stream contains no correction queries.")
    geometry = capture_functional_geometry_reference(
        model,
        guard,
        rank=args.functional_geometry_rank,
        device=device,
    )
    original_geometry = collect_geometry(model, guard, device=device)
    base_guard_loss = float(guard_loss_tensor(model, guard, device=device).detach().cpu())
    guard_limit = base_guard_loss + args.guard_loss_absolute
    memory = ConsolidatedSemanticMemory(
        hard=[],
        sketch=StreamingConstraintSketch(rows=None, updates=0, observed_rows=0),
        observed_queries=0,
        compressed_queries=0,
        dropped_queries=0,
    )
    evaluation_history: list[FactQuery] = []
    offline_audits: list[OfflineQueryAudit] = []
    stage_reports: list[dict[str, Any]] = []

    print("1M SEMANTIC MANIFOLD CONSOLIDATION")
    print("=" * 160)
    print(
        f"device={device} parameters={parameter_count} mode={args.semantic_mode} "
        f"queries={len(queries)} memory_rows={args.semantic_memory_rows} "
        f"hard_rows={args.semantic_hard_rows} epochs={args.cl_epochs}"
    )
    for update, (group, query) in enumerate(queries, start=1):
        started = time.perf_counter()
        training = train_query(
            model,
            query,
            guard,
            geometry,
            memory,
            guard_limit=guard_limit,
            args=args,
            device=device,
        )
        evaluation_history.append(query)
        consolidation: dict[str, Any] | None = None
        committed = bool(training["committable"])
        offline_audits.append(
            OfflineQueryAudit(
                query=query,
                group=group,
                source_update=update,
                committed=committed,
                disposition="pending_hard" if committed else "uncommitted",
                compression_update=None,
                compression_row=None,
                first_incorrect_update=None,
                first_floor_failure_update=None,
            )
        )
        if committed:
            probe = SemanticProbe(
                query=query,
                group=group,
                source_update=update,
                reference_margin=training["final_margin"],
                support_sum=1.0,
                observations=1,
                dual_pressure=0.0,
            )
            guard_rows, _guard_bounds, _guard_report = hard_guard_loss_constraint(
                model,
                guard,
                parameters=parameters,
                maximum_loss=guard_limit,
                device=device,
            )
            geometry_rows, _geometry_bounds, _geometry_loss, _geometry_report = (
                functional_geometry_constraints(
                    model,
                    guard,
                    geometry,
                    parameters=parameters,
                    maximum_distortion=args.functional_geometry_max_distortion,
                    activation_margin=args.functional_geometry_activation_margin,
                    device=device,
                )
            )
            dependency_rows = torch.cat([guard_rows, geometry_rows], dim=0)
            consolidation = consolidate_probe(
                model,
                memory,
                probe,
                parameters,
                dependency_rows,
                mode=args.semantic_mode,
                memory_rows=args.semantic_memory_rows,
                hard_rows=args.semantic_hard_rows,
                floor=args.semantic_floor,
                reserve=args.semantic_interior_reserve,
                sketch_decay=args.semantic_sketch_decay,
                rank_tolerance=args.dependency_rank_tolerance,
                device=device,
            )
            compressed_records = consolidation.pop("_compressed_records")
            compressed_rows = {
                source_update: row for source_update, row in compressed_records
            }
            selected_updates = set(consolidation["selected_updates"])
            compressed_updates = set(consolidation["compressed_updates"])
            dropped_updates = set(consolidation["dropped_updates"])
            for audit in offline_audits:
                if audit.source_update in selected_updates:
                    audit.disposition = "hard"
                elif audit.source_update in compressed_updates:
                    audit.disposition = "compressed"
                    audit.compression_update = update
                    audit.compression_row = compressed_rows[audit.source_update]
                elif audit.source_update in dropped_updates:
                    audit.disposition = "dropped"

        update_offline_failure_audit(
            model,
            offline_audits,
            update=update,
            floor=args.semantic_floor,
            device=device,
        )

        history = evaluate_correction_queries(
            model,
            evaluation_history,
            device=device,
        )
        hard_measurement = semantic_memory_measurement(
            model,
            memory,
            floor=args.semantic_floor,
            device=device,
        )
        if not hard_measurement["passed"]:
            raise RuntimeError("An accepted update violated a retained hard semantic probe.")
        final_guard = float(guard_loss_tensor(model, guard, device=device).detach().cpu())
        final_distortion, final_geometry = functional_geometry_measurement(
            model,
            guard,
            geometry,
            device=device,
        )
        if final_guard > guard_limit + args.semantic_retraction_tolerance:
            raise RuntimeError("Accepted semantic stage violated the guard-loss ceiling.")
        if float(final_distortion.detach().cpu()) > (
            args.functional_geometry_max_distortion
            + args.semantic_retraction_tolerance
        ):
            raise RuntimeError("Accepted semantic stage violated functional geometry.")
        sketch = semantic_sketch_report(
            memory,
            parameter_count=parameter_count,
            tolerance=args.dependency_rank_tolerance,
        )
        retained = [
            record["solver"]["new_gradient_retained_fraction"]
            for record in training["trace"]
            if record.get("accepted") and "solver" in record
        ]
        stage_report = {
            "update": update,
            "group": group,
            "training": training,
            "committed": committed,
            "consolidation": consolidation,
            "history": history,
            "hard_semantic": hard_measurement,
            "hard_size": len(memory.hard),
            "memory_rows": args.semantic_memory_rows,
            "sketch": sketch,
            "mean_retained_gradient": 0.0 if not retained else sum(retained) / len(retained),
            "guard_loss": final_guard,
            "guard_limit": guard_limit,
            "geometry_distortion": float(final_distortion.detach().cpu()),
            "geometry": final_geometry,
            "seconds": time.perf_counter() - started,
        }
        stage_reports.append(stage_report)
        print(
            f"update={update:02d} committed={int(committed)} "
            f"margin={training['initial_margin']:.3f}->{training['final_margin']:.3f} "
            f"history={history['new_over_old_fraction']:.3f} "
            f"hard={len(memory.hard)}/{args.semantic_hard_rows if args.semantic_mode == 'consolidated' else args.semantic_memory_rows} "
            f"sketch={sketch['rows']} retained={stage_report['mean_retained_gradient']:.6f} "
            f"guard={final_guard:.5f} distortion={stage_report['geometry_distortion']:.5f} "
            f"seconds={stage_report['seconds']:.1f}"
        )

    final_geometry = collect_geometry(model, guard, device=device)
    residual_geometry = geometry_report(original_geometry, final_geometry)
    minimum_cka = min(value["cka"] for value in residual_geometry.values())
    final_audit_records = final_offline_audit(
        model,
        offline_audits,
        memory,
        parameters,
        floor=args.semantic_floor,
        device=device,
    )
    committed_audits = [record for record in final_audit_records if record["committed"]]
    compressed_audits = [
        record
        for record in committed_audits
        if record["disposition"] == "compressed"
    ]
    coverage_values = [
        record["final_sketch_coverage"]
        for record in compressed_audits
        if record["final_sketch_coverage"] is not None
    ]
    transport_values = [
        record["jacobian_transport_cosine"]
        for record in compressed_audits
        if record["jacobian_transport_cosine"] is not None
    ]
    audit_summary = {
        "committed_correct": sum(int(record["correct"]) for record in committed_audits),
        "committed_total": len(committed_audits),
        "compressed_correct": sum(
            int(record["correct"]) for record in compressed_audits
        ),
        "compressed_total": len(compressed_audits),
        "mean_compressed_sketch_coverage": (
            None if not coverage_values else sum(coverage_values) / len(coverage_values)
        ),
        "mean_compressed_transport_cosine": (
            None if not transport_values else sum(transport_values) / len(transport_values)
        ),
        "ever_incorrect": sum(
            int(record["first_incorrect_update"] is not None)
            for record in committed_audits
        ),
        "ever_below_floor": sum(
            int(record["first_floor_failure_update"] is not None)
            for record in committed_audits
        ),
    }
    hard_query_scalars = sum(len(probe.query.input_ids) + 7 for probe in memory.hard)
    sketch_scalars = 0 if memory.sketch.rows is None else int(memory.sketch.rows.numel())
    persistent_scalars = hard_query_scalars + sketch_scalars
    maximum_sketch_rows = (
        0
        if args.semantic_mode == "exact"
        else args.semantic_memory_rows - args.semantic_hard_rows
    )
    maximum_persistent_scalars = (
        args.semantic_memory_rows * (model.max_seq_len + 7)
        if args.semantic_mode == "exact"
        else args.semantic_hard_rows * (model.max_seq_len + 7)
        + maximum_sketch_rows * parameter_count
    )
    if persistent_scalars > maximum_persistent_scalars:
        raise RuntimeError(
            "Semantic memory exceeded its configured bound: "
            f"observed={persistent_scalars}, maximum={maximum_persistent_scalars}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "semantic_manifold_consolidation.json"
    plot_path = args.output_dir / "semantic_manifold_consolidation.png"
    plot_results(stage_reports, plot_path)
    output = {
        "question": (
            "Can exact critical semantic probes plus a bounded compressed Jacobian field "
            "retain corrections without exhausting useful update directions?"
        ),
        "scope": (
            "Single-seed isolated semantic-pressure experiment. Full query history is "
            "offline evaluation only and is not available to the learner after consolidation."
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": {
            "parameters": parameter_count,
            "vocab_size": vocab_size,
            "layers": len(model.blocks),
        },
        "memory": {
            "hard_probes": len(memory.hard),
            "sketch_rows": 0 if memory.sketch.rows is None else memory.sketch.rows.shape[0],
            "observed_queries": memory.observed_queries,
            "compressed_queries": memory.compressed_queries,
            "dropped_queries": memory.dropped_queries,
            "persistent_scalars": persistent_scalars,
            "maximum_persistent_scalars": maximum_persistent_scalars,
            "sketch": semantic_sketch_report(
                memory,
                parameter_count=parameter_count,
                tolerance=args.dependency_rank_tolerance,
            ),
        },
        "final": {
            "history": stage_reports[-1]["history"],
            "hard_semantic": stage_reports[-1]["hard_semantic"],
            "guard_loss": stage_reports[-1]["guard_loss"],
            "geometry_distortion": stage_reports[-1]["geometry_distortion"],
            "minimum_residual_cka": minimum_cka,
            "residual_geometry": residual_geometry,
            "audit": audit_summary,
        },
        "hard_probes": [
            {
                "group": probe.group,
                "source_update": probe.source_update,
                "input_ids": list(probe.query.input_ids),
                "old_target_id": probe.query.old_target_id,
                "new_target_id": probe.query.new_target_id,
                "reference_margin": probe.reference_margin,
                "support_sum": probe.support_sum,
                "observations": probe.observations,
                "dual_pressure": probe.dual_pressure,
            }
            for probe in memory.hard
        ],
        "offline_query_audit": final_audit_records,
        "stages": stage_reports,
        "plots": {"summary": str(plot_path)},
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nFINAL SEMANTIC MANIFOLD STATE")
    print("-" * 160)
    print(
        f"committed={sum(int(stage['committed']) for stage in stage_reports)}/{len(stage_reports)} "
        f"history_fraction={stage_reports[-1]['history']['new_over_old_fraction']:.4f} "
        f"history_margin={stage_reports[-1]['history']['new_minus_old_margin']:.4f} "
        f"hard={len(memory.hard)} sketch={0 if memory.sketch.rows is None else memory.sketch.rows.shape[0]} "
        f"compressed={memory.compressed_queries} dropped={memory.dropped_queries} "
        f"compressed_correct={audit_summary['compressed_correct']}/{audit_summary['compressed_total']} "
        f"min_cka={minimum_cka:.4f} memory={persistent_scalars}/{maximum_persistent_scalars}"
    )
    print(f"wrote_json={output_json}")
    print(f"wrote_plot={plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = build_long_horizon_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=Path("model/analysis/gco-1m-semantic-manifold-seed0"),
        geometry_constraint_mode="functional_transport",
        cycles=6,
    )
    parser.add_argument(
        "--semantic-mode",
        choices=("consolidated", "exact"),
        default="consolidated",
    )
    parser.add_argument("--semantic-memory-rows", type=int, default=16)
    parser.add_argument("--semantic-hard-rows", type=int, default=4)
    parser.add_argument("--semantic-floor", type=float, default=1.0)
    parser.add_argument("--semantic-interior-reserve", type=float, default=0.1)
    parser.add_argument("--semantic-barrier-strength", type=float, default=0.05)
    parser.add_argument("--semantic-dual-decay", type=float, default=0.9)
    parser.add_argument("--semantic-sketch-decay", type=float, default=1.0)
    parser.add_argument("--semantic-retraction-steps", type=int, default=8)
    parser.add_argument(
        "--semantic-retraction-radius-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--semantic-retraction-damping", type=float, default=1e-6)
    parser.add_argument("--semantic-retraction-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--semantic-retraction-candidate-retention",
        type=float,
        default=0.8,
    )
    parser.add_argument("--semantic-guard-safety", type=float, default=5e-4)
    parser.add_argument("--semantic-geometry-safety", type=float, default=5e-3)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
