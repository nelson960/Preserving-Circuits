"""Close the loop from functional dependency to autonomous trace survival.

The experiment extends the delayed autonomous trace mechanism with trace-indexed
dependency blocks. For every online event it estimates the consequence of
releasing each trace, validates that estimate against exact virtual updates, and
uses evidence contribution plus functional consequence in a bounded continuous
survival optimization. Hidden semantic groups are used only to construct and
evaluate the controlled stream; they never enter survival or model updates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_mini_cl_world_demo import (
    flat_autograd_gradient,
    project_gradient_away_from_constraints,
)
from experiments.gco_math.gco_tiny_autonomous_dependency_invariant_cl import (
    AutonomousTraceState,
    TraceTransition,
    absolute_weighted_mse,
    active_trace_slots,
    apply_trial_update,
    build_parser as build_autonomous_parser,
    commit_trace_transition,
    event_batches,
    geometry_loss,
    geometry_metrics,
    initialize_trace_state,
    plot_behavior,
    plot_capacity,
    plot_geometry,
    plot_trace_lifecycle,
    prepare_trace_transition,
    restore_parameters,
    snapshot_parameters,
    trace_group_report,
    trace_mass_report,
    train_event,
    validate_args as validate_autonomous_args,
)
from experiments.gco_math.gco_tiny_functional_dependency_field import (
    control_plane_svd,
    retained_rank,
)
from experiments.gco_math.gco_tiny_recurrent_trace_field import (
    TraceSummary,
    functional_attention,
)
from experiments.gco_math.gco_tiny_trace_invariant_tangent_integration import (
    FunctionExample,
    TinyFunctionModel,
    bounded_restore,
    build_function_stream,
    centered_kernel_alignment,
    evaluate_model,
    examples_to_tensors,
    hidden_pair_distances,
    train_base,
    weighted_mse,
)


@dataclass
class TraceDependencyBlock:
    rows: torch.Tensor
    output_rows: torch.Tensor
    row_basis: torch.Tensor


@dataclass
class ConsequenceReport:
    evidence: torch.Tensor
    verified: torch.Tensor
    conflict: torch.Tensor
    centrality: torch.Tensor
    direct_exact: torch.Tensor
    downstream_exact: torch.Tensor
    direct_predicted: torch.Tensor
    downstream_predicted: torch.Tensor
    full_direction: torch.Tensor
    release_directions: list[torch.Tensor]
    overlap: torch.Tensor


def validate_args(args: argparse.Namespace) -> None:
    validate_autonomous_args(args)
    if not 0.0 < args.survival_budget <= args.num_slots:
        raise ValueError("--survival-budget must be in (0, num_slots].")
    if args.survival_temperature <= 0.0 or not math.isfinite(args.survival_temperature):
        raise ValueError("--survival-temperature must be positive and finite.")
    if args.survival_bisection_steps <= 0:
        raise ValueError("--survival-bisection-steps must be positive.")
    if args.trace_dependency_rank <= 0:
        raise ValueError("--trace-dependency-rank must be positive.")
    if not 0.0 < args.trace_dependency_energy <= 1.0:
        raise ValueError("--trace-dependency-energy must be in (0, 1].")
    for name in (
        "utility_evidence_weight",
        "utility_verified_weight",
        "utility_direct_weight",
        "utility_downstream_weight",
        "utility_centrality_weight",
        "utility_conflict_weight",
    ):
        value = getattr(args, name)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")


def normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] <= 0:
        raise ValueError("Dependency row matrix must be non-empty [R, P].")
    norms = torch.linalg.vector_norm(matrix, dim=1)
    active = norms > torch.finfo(matrix.dtype).eps
    if not active.any():
        raise FloatingPointError("Dependency block has no nonzero rows.")
    return matrix[active] / norms[active].unsqueeze(1)


def compress_trace_rows(
    rows: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = normalize_rows(rows)
    singular, right = control_plane_svd(normalized)
    rank = retained_rank(
        singular,
        rank_budget=args.trace_dependency_rank,
        tolerance=args.dependency_rank_tolerance,
        energy_target=args.trace_dependency_energy,
    )
    strengths = (singular[:rank] / singular[0]).pow(args.dependency_power)
    compressed = strengths.unsqueeze(1) * right[:rank]
    basis = right[:rank]
    return compressed.to(rows), basis.to(rows)


def build_trace_dependencies(
    model: TinyFunctionModel,
    probe_inputs: torch.Tensor,
    *,
    parameters: list[torch.nn.Parameter],
    args: argparse.Namespace,
) -> tuple[list[TraceDependencyBlock], torch.Tensor, torch.Tensor]:
    outputs, hidden = model(probe_inputs)
    slots = probe_inputs.shape[0]
    per_trace_rows: list[list[torch.Tensor]] = [[] for _ in range(slots)]
    output_rows: list[list[torch.Tensor]] = [[] for _ in range(slots)]

    for trace in range(slots):
        for output_index in range(outputs.shape[1]):
            row = flat_autograd_gradient(
                outputs[trace, output_index],
                parameters,
                retain_graph=True,
                require_nonzero=False,
                label=f"utility_output_{trace}_{output_index}",
            )
            output_rows[trace].append(row)
            per_trace_rows[trace].append(row)
        for hidden_index in range(hidden.shape[1]):
            per_trace_rows[trace].append(
                flat_autograd_gradient(
                    hidden[trace, hidden_index],
                    parameters,
                    retain_graph=True,
                    require_nonzero=False,
                    label=f"utility_hidden_{trace}_{hidden_index}",
                )
            )

    for (left, right), distance in hidden_pair_distances(hidden).items():
        row = flat_autograd_gradient(
            distance,
            parameters,
            retain_graph=True,
            require_nonzero=False,
            label=f"utility_geometry_{left}_{right}",
        )
        per_trace_rows[left].append(row)
        per_trace_rows[right].append(row)

    blocks: list[TraceDependencyBlock] = []
    for trace in range(slots):
        compressed, basis = compress_trace_rows(torch.stack(per_trace_rows[trace]), args=args)
        blocks.append(
            TraceDependencyBlock(
                rows=compressed,
                output_rows=torch.stack(output_rows[trace]),
                row_basis=basis,
            )
        )
    return blocks, outputs, hidden


def dependency_overlap(blocks: list[TraceDependencyBlock]) -> torch.Tensor:
    slots = len(blocks)
    if slots <= 1:
        raise ValueError("Dependency overlap requires at least two traces.")
    device = blocks[0].rows.device
    overlap = torch.zeros(slots, slots, device=device, dtype=blocks[0].rows.dtype)
    for left in range(slots):
        for right in range(slots):
            cross = blocks[left].row_basis @ blocks[right].row_basis.T
            denominator = min(
                blocks[left].row_basis.shape[0],
                blocks[right].row_basis.shape[0],
            )
            overlap[left, right] = cross.square().sum() / denominator
    return overlap


def evidence_leave_one_out(
    state: AutonomousTraceState,
    *,
    attention_scale: float,
) -> torch.Tensor:
    means = state.summary.means
    weights = state.summary.weights
    full_attention = functional_attention(
        means,
        state.centers,
        attention_scale=attention_scale,
    )
    full_reconstruction = full_attention @ state.centers
    full_error = ((means - full_reconstruction).square().sum(dim=1) * weights).sum()
    values: list[torch.Tensor] = []
    for trace in range(state.centers.shape[0]):
        keep = torch.arange(state.centers.shape[0], device=means.device) != trace
        reduced_centers = state.centers[keep]
        reduced_attention = functional_attention(
            means,
            reduced_centers,
            attention_scale=attention_scale,
        )
        reconstruction = reduced_attention @ reduced_centers
        error = ((means - reconstruction).square().sum(dim=1) * weights).sum()
        values.append((error - full_error).clamp_min(0.0))
    return torch.stack(values)


def incoming_conflict(
    state: AutonomousTraceState,
    transition: TraceTransition,
    *,
    args: argparse.Namespace,
) -> torch.Tensor:
    prior = state.summary.means
    current = transition.current.means
    input_distance = (
        prior[:, None, : args.d_model] - current[None, :, : args.d_model]
    ).square().sum(dim=2)
    target_distance = (
        prior[:, None, args.d_model :] - current[None, :, args.d_model :]
    ).square().sum(dim=2)
    input_match = torch.exp(-input_distance / (2.0 * args.attention_scale**2))
    target_conflict = 1.0 - torch.exp(
        -target_distance / (2.0 * args.attention_scale**2)
    )
    return (input_match * target_conflict).max(dim=1).values


def weighted_constraint_rows(
    blocks: list[TraceDependencyBlock],
    weights: torch.Tensor,
    *,
    excluded: int | None,
) -> list[torch.Tensor]:
    if weights.shape != (len(blocks),):
        raise ValueError("Trace dependency weights do not match block count.")
    rows: list[torch.Tensor] = []
    for trace, block in enumerate(blocks):
        if trace == excluded:
            continue
        scale = torch.sqrt(weights[trace].clamp_min(0.0))
        rows.extend(scale * row for row in block.rows)
    if not rows:
        raise RuntimeError("Trace release produced an empty dependency basis.")
    return rows


def restore_direction(
    model: TinyFunctionModel,
    *,
    tangent: torch.Tensor,
    probe_inputs: torch.Tensor,
    probe_targets: torch.Tensor,
    weights: torch.Tensor,
    reference_distances: dict[tuple[int, int], torch.Tensor],
    parameters: list[torch.nn.Parameter],
    args: argparse.Namespace,
) -> torch.Tensor:
    outputs, hidden = model(probe_inputs)
    behavior = weighted_mse(outputs, probe_targets, weights)
    geometry = geometry_loss(hidden, reference_distances, weights)
    gradient = flat_autograd_gradient(
        args.loss_mix_strength * behavior + args.geometry_mix_strength * geometry,
        parameters,
        retain_graph=False,
        require_nonzero=False,
        label="utility_restore",
    )
    return bounded_restore(
        gradient,
        tangent,
        strength=args.restore_strength,
        bound_fraction=args.restore_bound_fraction,
    )


def candidate_direction(
    model: TinyFunctionModel,
    *,
    raw_gradient: torch.Tensor,
    blocks: list[TraceDependencyBlock],
    weights: torch.Tensor,
    excluded: int | None,
    probe_inputs: torch.Tensor,
    probe_targets: torch.Tensor,
    reference_distances: dict[tuple[int, int], torch.Tensor],
    parameters: list[torch.nn.Parameter],
    args: argparse.Namespace,
) -> torch.Tensor:
    effective_weights = weights.clone()
    if excluded is not None:
        effective_weights[excluded] = 0.0
    tangent, _stats = project_gradient_away_from_constraints(
        raw_gradient=raw_gradient,
        constraint_gradients=weighted_constraint_rows(
            blocks,
            effective_weights,
            excluded=excluded,
        ),
        damping=args.projection_damping,
        solver="gram",
        rank_tolerance=args.dependency_rank_tolerance,
        plasticity_audit=False,
    )
    restore = restore_direction(
        model,
        tangent=tangent,
        probe_inputs=probe_inputs,
        probe_targets=probe_targets,
        weights=effective_weights,
        reference_distances=reference_distances,
        parameters=parameters,
        args=args,
    )
    return tangent + restore


def clipped_direction(
    direction: torch.Tensor,
    *,
    grad_clip: float,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Virtual consequence direction is zero or non-finite.")
    scale = torch.clamp(direction.new_tensor(grad_clip) / norm, max=1.0)
    return direction * scale


@torch.no_grad()
def probe_losses(
    model: TinyFunctionModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    outputs, _hidden = model(inputs)
    return (outputs - targets).square().mean(dim=1)


def apply_virtual_direction(
    model: TinyFunctionModel,
    parameters: list[torch.nn.Parameter],
    snapshot: list[torch.Tensor],
    direction: torch.Tensor,
    *,
    probe_inputs: torch.Tensor,
    probe_targets: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    restore_parameters(parameters, snapshot)
    apply_trial_update(
        parameters,
        direction,
        learning_rate=args.cl_lr,
        grad_clip=args.grad_clip,
    )
    losses = probe_losses(model, probe_inputs, probe_targets)
    restore_parameters(parameters, snapshot)
    return losses


def predicted_probe_losses(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    blocks: list[TraceDependencyBlock],
    direction: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> torch.Tensor:
    effective = clipped_direction(direction, grad_clip=args.grad_clip)
    predicted: list[torch.Tensor] = []
    for trace, block in enumerate(blocks):
        change = block.output_rows @ effective
        value = outputs[trace] - args.cl_lr * change
        predicted.append((value - targets[trace]).square().mean())
    return torch.stack(predicted)


def consequence_from_losses(
    full: torch.Tensor,
    released: list[torch.Tensor],
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    direct: list[torch.Tensor] = []
    downstream: list[torch.Tensor] = []
    for trace, losses in enumerate(released):
        increase = (losses - full).clamp_min(0.0)
        direct.append(increase[trace])
        mask = torch.arange(weights.numel(), device=weights.device) != trace
        denominator = weights[mask].sum().clamp_min(torch.finfo(weights.dtype).eps)
        downstream.append((increase[mask] * weights[mask]).sum() / denominator)
    return torch.stack(direct), torch.stack(downstream)


def consequence_report(
    model: TinyFunctionModel,
    state: AutonomousTraceState,
    transition: TraceTransition,
    examples: list[FunctionExample],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> ConsequenceReport:
    parameters = list(model.parameters())
    probe_inputs = transition.prior_inputs.to(device)
    probe_targets = transition.prior_targets.to(device)
    weights = transition.prior_protection.to(device)
    blocks, outputs, hidden = build_trace_dependencies(
        model,
        probe_inputs,
        parameters=parameters,
        args=args,
    )
    overlap = dependency_overlap(blocks)
    centrality = (
        overlap * weights.unsqueeze(0)
    ).sum(dim=1) - overlap.diagonal() * weights

    inputs, targets = examples_to_tensors(examples, device=device)
    prediction, _current_hidden = model(inputs)
    new_loss = absolute_weighted_mse(
        prediction,
        targets,
        transition.write_weights.to(device),
    )
    raw_gradient = flat_autograd_gradient(
        new_loss,
        parameters,
        retain_graph=True,
        require_nonzero=False,
        label="utility_new_gradient",
    )
    numerical_zero = torch.finfo(raw_gradient.dtype).eps
    if torch.linalg.vector_norm(raw_gradient) <= numerical_zero:
        zero = torch.zeros(len(blocks), device=device, dtype=raw_gradient.dtype)
        return ConsequenceReport(
            evidence=evidence_leave_one_out(state, attention_scale=args.attention_scale),
            verified=state.learned_mass / state.summary.weights.clamp_min(numerical_zero),
            conflict=incoming_conflict(state, transition, args=args),
            centrality=centrality,
            direct_exact=zero,
            downstream_exact=zero,
            direct_predicted=zero,
            downstream_predicted=zero,
            full_direction=torch.zeros_like(raw_gradient),
            release_directions=[torch.zeros_like(raw_gradient) for _ in blocks],
            overlap=overlap,
        )

    reference_distances = {
        pair: value.detach()
        for pair, value in hidden_pair_distances(hidden).items()
    }
    full_direction = candidate_direction(
        model,
        raw_gradient=raw_gradient,
        blocks=blocks,
        weights=weights,
        excluded=None,
        probe_inputs=probe_inputs,
        probe_targets=probe_targets,
        reference_distances=reference_distances,
        parameters=parameters,
        args=args,
    )
    release_directions = [
        candidate_direction(
            model,
            raw_gradient=raw_gradient,
            blocks=blocks,
            weights=weights,
            excluded=trace,
            probe_inputs=probe_inputs,
            probe_targets=probe_targets,
            reference_distances=reference_distances,
            parameters=parameters,
            args=args,
        )
        for trace in range(len(blocks))
    ]

    snapshot = snapshot_parameters(parameters)
    exact_full = apply_virtual_direction(
        model,
        parameters,
        snapshot,
        full_direction,
        probe_inputs=probe_inputs,
        probe_targets=probe_targets,
        args=args,
    )
    exact_released = [
        apply_virtual_direction(
            model,
            parameters,
            snapshot,
            direction,
            probe_inputs=probe_inputs,
            probe_targets=probe_targets,
            args=args,
        )
        for direction in release_directions
    ]
    predicted_full = predicted_probe_losses(
        outputs,
        probe_targets,
        blocks,
        full_direction,
        args=args,
    )
    predicted_released = [
        predicted_probe_losses(outputs, probe_targets, blocks, direction, args=args)
        for direction in release_directions
    ]
    direct_exact, downstream_exact = consequence_from_losses(
        exact_full,
        exact_released,
        weights,
    )
    direct_predicted, downstream_predicted = consequence_from_losses(
        predicted_full,
        predicted_released,
        weights,
    )
    return ConsequenceReport(
        evidence=evidence_leave_one_out(state, attention_scale=args.attention_scale),
        verified=state.learned_mass / state.summary.weights.clamp_min(numerical_zero),
        conflict=incoming_conflict(state, transition, args=args),
        centrality=centrality,
        direct_exact=direct_exact,
        downstream_exact=downstream_exact,
        direct_predicted=direct_predicted,
        downstream_predicted=downstream_predicted,
        full_direction=full_direction,
        release_directions=release_directions,
        overlap=overlap,
    )


def rms_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("Utility component must be a non-empty vector.")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Utility component contains non-finite values.")
    nonnegative = values.clamp_min(0.0)
    scale = torch.sqrt(nonnegative.square().mean())
    if scale <= torch.finfo(values.dtype).eps:
        return torch.zeros_like(values)
    return nonnegative / scale


def survival_utility(
    report: ConsequenceReport,
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = {
        "evidence": rms_normalize(report.evidence),
        "verified": rms_normalize(report.verified),
        "direct": rms_normalize(report.direct_predicted),
        "downstream": rms_normalize(report.downstream_predicted),
        "centrality": rms_normalize(report.centrality),
        "conflict": rms_normalize(report.conflict),
    }
    utility = (
        args.utility_evidence_weight * components["evidence"]
        + args.utility_verified_weight * components["verified"]
        + args.utility_direct_weight * components["direct"]
        + args.utility_downstream_weight * components["downstream"]
        + args.utility_centrality_weight * components["centrality"]
        - args.utility_conflict_weight * components["conflict"]
    )
    return utility, components


def allocate_survival(
    utility: torch.Tensor,
    *,
    budget: float,
    temperature: float,
    bisection_steps: int,
) -> torch.Tensor:
    if not 0.0 < budget <= utility.numel():
        raise ValueError("Survival budget is outside the trace count.")
    span = utility.abs().max() + utility.new_tensor(32.0 * temperature)
    low = utility.min() - span
    high = utility.max() + span
    for _ in range(bisection_steps):
        midpoint = 0.5 * (low + high)
        mass = torch.sigmoid((utility - midpoint) / temperature).sum()
        if mass > budget:
            low = midpoint
        else:
            high = midpoint
    threshold = 0.5 * (low + high)
    survival = torch.sigmoid((utility - threshold) / temperature)
    if not torch.isfinite(survival).all():
        raise FloatingPointError("Survival allocation contains non-finite values.")
    return survival


def apply_survival(
    state: AutonomousTraceState,
    survival: torch.Tensor,
) -> AutonomousTraceState:
    if survival.shape != state.summary.weights.shape:
        raise ValueError("Survival vector does not match trace slots.")
    if torch.any(survival <= 0.0) or torch.any(survival > 1.0):
        raise ValueError("Survival values must be in (0, 1].")
    summary = replace(
        state.summary,
        weights=state.summary.weights * survival,
    )
    result = AutonomousTraceState(
        summary=summary,
        centers=state.centers,
        learned_mass=state.learned_mass * survival,
        event=state.event,
    )
    result.validate()
    return result


def tensor_values(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu()]


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Consequence correlation requires equal nontrivial vectors.")
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denominator <= torch.finfo(x.dtype).eps:
        raise FloatingPointError(
            "Consequence calibration is undefined because an exact or predicted vector "
            "has zero variance."
        )
    return float((torch.dot(x, y) / denominator).item())


def plot_survival(events: list[dict[str, Any]], *, output_path: Path) -> None:
    online = events[1:]
    survival = torch.tensor([event["survival"]["mass"] for event in online])
    utility = torch.tensor([event["survival"]["utility"] for event in online])
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.8), sharex=True)
    for axis, matrix, title in (
        (axes[0], survival, "Trace survival mass"),
        (axes[1], utility, "Normalized functional utility"),
    ):
        image = axis.imshow(matrix.T, aspect="auto", origin="lower", cmap="viridis")
        axis.set_ylabel("trace slot")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    axes[1].set_xlabel("online event")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_consequence(events: list[dict[str, Any]], *, output_path: Path) -> None:
    direct_exact: list[float] = []
    direct_predicted: list[float] = []
    downstream_exact: list[float] = []
    downstream_predicted: list[float] = []
    for event in events[1:]:
        direct_exact.extend(event["consequence"]["direct_exact"])
        direct_predicted.extend(event["consequence"]["direct_predicted"])
        downstream_exact.extend(event["consequence"]["downstream_exact"])
        downstream_predicted.extend(event["consequence"]["downstream_predicted"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for axis, exact, predicted, title in (
        (axes[0], direct_exact, direct_predicted, "Direct release consequence"),
        (axes[1], downstream_exact, downstream_predicted, "Downstream release consequence"),
    ):
        axis.scatter(exact, predicted, alpha=0.45, s=18)
        maximum = max(exact + predicted + [torch.finfo(torch.float32).eps])
        axis.plot([0.0, maximum], [0.0, maximum], color="#dc2626", linestyle="--")
        axis.set_xlabel("exact virtual update")
        axis.set_ylabel("first-order prediction")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overlap(overlap: list[list[float]], *, output_path: Path) -> None:
    matrix = torch.tensor(overlap)
    fig, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    axis.set_title("Final trace dependency overlap")
    axis.set_xlabel("trace slot")
    axis.set_ylabel("trace slot")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_with_stages(
    args: argparse.Namespace,
    *,
    stages: list[list[FunctionExample]],
    question: str,
    scope: str,
) -> dict[str, Any]:
    validate_args(args)
    if len(stages) < 2 or any(not stage for stage in stages):
        raise ValueError("A consequence-survival experiment requires at least two non-empty stages.")
    if not question.strip() or not scope.strip():
        raise ValueError("Experiment question and scope must be non-empty.")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    batches = event_batches(stages, batch_size=args.event_batch_size)
    all_examples = [example for stage in stages for example in stage]
    groups = sorted({example.hidden_group for example in all_examples})

    model = TinyFunctionModel(
        input_dim=args.d_model,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    ).to(device)
    base_trace = train_base(
        model,
        stages[0],
        epochs=args.base_epochs,
        learning_rate=args.base_lr,
        device=device,
    )
    base_inputs, _targets = examples_to_tensors(stages[0], device=device)
    with torch.no_grad():
        _outputs, base_hidden = model(base_inputs)
    state = initialize_trace_state(stages[0], args=args, device=device)
    initial_evaluation = evaluate_model(model, all_examples, device=device)
    initial_trace_groups = trace_group_report(
        state,
        all_examples,
        args=args,
        device=device,
    )
    events: list[dict[str, Any]] = [
        {
            "event": 0,
            "stage": 1,
            "groups_seen": sorted({example.hidden_group for example in stages[0]}),
            "evaluation": {
                "mse": initial_evaluation["mse"],
                "groups": initial_evaluation["groups"],
            },
            "geometry": geometry_metrics(model, base_inputs, base_hidden),
            "trace": {
                "write_mean": 1.0,
                "support_mean": 1.0,
                "protection_mean": 1.0,
                "release_mean": 0.0,
                "gain_mean": 1.0,
                **trace_mass_report(state),
                "effective_slots": float(args.num_slots),
                "active_slots": active_trace_slots(state),
                "persistent_scalars": state.persistent_scalars(),
            },
            "update": {
                "deferred": False,
                "dependency_rank": 0.0,
                "projection_removed_fraction": 0.0,
                "damage_ratio": 0.0,
            },
            "survival": {
                "mass": [1.0 for _ in range(args.num_slots)],
                "utility": [1.0 for _ in range(args.num_slots)],
            },
            "consequence": {
                "direct_exact": [0.0 for _ in range(args.num_slots)],
                "direct_predicted": [0.0 for _ in range(args.num_slots)],
                "downstream_exact": [0.0 for _ in range(args.num_slots)],
                "downstream_predicted": [0.0 for _ in range(args.num_slots)],
                "overlap": torch.eye(args.num_slots).tolist(),
            },
            "trace_groups": initial_trace_groups,
        }
    ]

    print("TINY FUNCTIONAL-CONSEQUENCE TRACE SURVIVAL")
    print("=" * 148)
    print(
        f"device={device} params={sum(parameter.numel() for parameter in model.parameters())} "
        f"slots={args.num_slots} survival_budget={args.survival_budget} events={len(batches)}"
    )
    for event_number, (stage_number, examples) in enumerate(batches, start=1):
        preliminary = prepare_trace_transition(
            state,
            examples,
            args=args,
            device=device,
        )
        consequence = consequence_report(
            model,
            state,
            preliminary,
            examples,
            args=args,
            device=device,
        )
        utility, components = survival_utility(consequence, args=args)
        survival = allocate_survival(
            utility,
            budget=args.survival_budget,
            temperature=args.survival_temperature,
            bisection_steps=args.survival_bisection_steps,
        ).detach()
        utility = utility.detach()
        components = {name: value.detach() for name, value in components.items()}
        surviving_state = apply_survival(state, survival)
        transition = prepare_trace_transition(
            surviving_state,
            examples,
            args=args,
            device=device,
        )
        training, gain = train_event(
            model,
            examples,
            transition,
            args=args,
            device=device,
        )
        state = commit_trace_transition(
            surviving_state,
            transition,
            gain,
            args=args,
        )
        evaluation = evaluate_model(model, all_examples, device=device)
        event_trace_groups = trace_group_report(
            state,
            all_examples,
            args=args,
            device=device,
        )
        last_epoch = training["epochs"][-1]
        raw_damage = last_epoch["raw_dependency_damage"]
        final_damage = last_epoch["final_dependency_damage"]
        event_report = {
            "event": event_number,
            "stage": stage_number,
            "groups_seen": sorted({example.hidden_group for example in examples}),
            "evaluation": {
                "mse": evaluation["mse"],
                "groups": evaluation["groups"],
            },
            "geometry": geometry_metrics(model, base_inputs, base_hidden),
            "trace": {
                "write_mean": float(transition.write_weights.mean().detach().cpu()),
                "support_mean": float(transition.delayed_support.mean().detach().cpu()),
                "protection_mean": float(transition.prior_protection.mean().detach().cpu()),
                "release_mean": float((1.0 - transition.old_confidence).mean().detach().cpu()),
                "gain_mean": float(gain.mean().detach().cpu()),
                **trace_mass_report(state),
                "effective_slots": transition.solution.effective_slots,
                "active_slots": active_trace_slots(state),
                "persistent_scalars": state.persistent_scalars(),
            },
            "update": {
                "deferred": training["deferred"],
                "dependency_rank": last_epoch["dependency_rank"],
                "projection_removed_fraction": last_epoch["projection_removed_fraction"],
                "raw_dependency_damage": raw_damage,
                "final_dependency_damage": final_damage,
                "damage_ratio": final_damage / max(raw_damage, torch.finfo(torch.float32).eps),
            },
            "survival": {
                "mass": tensor_values(survival),
                "utility": tensor_values(utility),
                "components": {name: tensor_values(value) for name, value in components.items()},
            },
            "consequence": {
                "direct_exact": tensor_values(consequence.direct_exact),
                "direct_predicted": tensor_values(consequence.direct_predicted),
                "downstream_exact": tensor_values(consequence.downstream_exact),
                "downstream_predicted": tensor_values(consequence.downstream_predicted),
                "overlap": consequence.overlap.detach().cpu().tolist(),
            },
            "trace_groups": event_trace_groups,
        }
        events.append(event_report)
        print(
            f"event={event_number:02d} stage={stage_number} "
            f"groups={','.join(event_report['groups_seen'])} "
            f"survival={min(survival).item():.3f}/{max(survival).item():.3f} "
            f"write={event_report['trace']['write_mean']:.3f} "
            f"gain={event_report['trace']['gain_mean']:.3f} "
            f"pending={event_report['trace']['pending_fraction']:.3f}"
        )

    final_evaluation = evaluate_model(model, all_examples, device=device)
    final_geometry = geometry_metrics(model, base_inputs, base_hidden)
    trace_groups = trace_group_report(state, all_examples, args=args, device=device)
    direct_exact = [value for event in events[1:] for value in event["consequence"]["direct_exact"]]
    direct_predicted = [
        value for event in events[1:] for value in event["consequence"]["direct_predicted"]
    ]
    downstream_exact = [
        value for event in events[1:] for value in event["consequence"]["downstream_exact"]
    ]
    downstream_predicted = [
        value for event in events[1:] for value in event["consequence"]["downstream_predicted"]
    ]
    calibration = {
        "direct_correlation": correlation(direct_exact, direct_predicted),
        "downstream_correlation": correlation(downstream_exact, downstream_predicted),
    }

    print("\nFINAL FUNCTIONAL-CONSEQUENCE CL STATE")
    print("-" * 148)
    print(f"{'group':>18} {'model_mse':>12} {'trace_error':>12} {'slot':>7} {'share':>9}")
    for group in groups:
        print(
            f"{group:>18} {final_evaluation['groups'][group]['mse']:12.5f} "
            f"{trace_groups[group]['trace_error']:12.5f} "
            f"{int(trace_groups[group]['dominant_slot']):7d} "
            f"{trace_groups[group]['dominant_share']:9.4f}"
        )
    print(
        f"calibration direct={calibration['direct_correlation']:.4f} "
        f"downstream={calibration['downstream_correlation']:.4f} "
        f"cka={final_geometry['cka']:.4f} pending={trace_mass_report(state)['pending_fraction']:.4f} "
        f"stored_scalars={state.persistent_scalars()}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.output_dir / "dependency_utility_behavior.png"
    lifecycle_path = args.output_dir / "dependency_utility_trace_lifecycle.png"
    geometry_path = args.output_dir / "dependency_utility_geometry.png"
    capacity_path = args.output_dir / "dependency_utility_capacity.png"
    survival_path = args.output_dir / "dependency_utility_survival.png"
    calibration_path = args.output_dir / "dependency_utility_calibration.png"
    overlap_path = args.output_dir / "dependency_utility_overlap.png"
    output_json = args.output_dir / "dependency_utility_survival.json"
    plot_behavior(events, groups, output_path=behavior_path)
    plot_trace_lifecycle(events, output_path=lifecycle_path)
    plot_geometry(events, output_path=geometry_path)
    plot_capacity(events, output_path=capacity_path)
    plot_survival(events, output_path=survival_path)
    plot_consequence(events, output_path=calibration_path)
    plot_overlap(events[-1]["consequence"]["overlap"], output_path=overlap_path)
    output = {
        "question": question,
        "scope": scope,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "base_training": {"initial_loss": base_trace[0], "final_loss": base_trace[-1]},
        "events": events,
        "calibration": calibration,
        "final": {
            "evaluation": {"mse": final_evaluation["mse"], "groups": final_evaluation["groups"]},
            "geometry": final_geometry,
            "trace_groups": trace_groups,
            **trace_mass_report(state),
            "persistent_scalars": state.persistent_scalars(),
        },
        "plots": {
            "behavior": str(behavior_path),
            "trace_lifecycle": str(lifecycle_path),
            "geometry": str(geometry_path),
            "capacity": str(capacity_path),
            "survival": str(survival_path),
            "calibration": str(calibration_path),
            "overlap": str(overlap_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(
        f"wrote_plots={behavior_path},{lifecycle_path},{geometry_path},{capacity_path},"
        f"{survival_path},{calibration_path},{overlap_path}"
    )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    return run_with_stages(
        args,
        stages=build_function_stream(args),
        question=(
            "Can functional dependency and predicted downstream consequence determine continuous trace "
            "survival and release under fixed capacity without semantic role labels?"
        ),
        scope=(
            "Single-seed tiny synthetic mechanism test. Exact virtual updates are a toy calibration oracle, "
            "not a scalable implementation."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_autonomous_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=Path("model/analysis/gco-tiny-dependency-utility-survival-seed0"),
        event_batch_size=1,
    )
    parser.add_argument("--survival-budget", type=float, default=4.0)
    parser.add_argument("--survival-temperature", type=float, default=0.35)
    parser.add_argument("--survival-bisection-steps", type=int, default=64)
    parser.add_argument("--trace-dependency-rank", type=int, default=8)
    parser.add_argument("--trace-dependency-energy", type=float, default=0.98)
    parser.add_argument("--utility-evidence-weight", type=float, default=1.0)
    parser.add_argument("--utility-verified-weight", type=float, default=1.0)
    parser.add_argument("--utility-direct-weight", type=float, default=1.0)
    parser.add_argument("--utility-downstream-weight", type=float, default=1.0)
    parser.add_argument("--utility-centrality-weight", type=float, default=0.5)
    parser.add_argument("--utility-conflict-weight", type=float, default=1.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
