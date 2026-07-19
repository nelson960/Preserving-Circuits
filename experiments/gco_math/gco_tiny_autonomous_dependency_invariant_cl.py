"""Integrate autonomous traces, dependency constraints, and Invariant-Tangent CL.

This is a single-method mechanism test. A fixed-size recurrent trace field sees
an unlabeled stream, maintains separate evidence and verified-learning mass,
and continuously determines write, protection, and release strength. Protected
trace probes are converted into a compressed functional dependency basis before
each model update. The new-learning gradient is projected away from that basis
and receives a bounded restore correction.

Hidden stream group names are used only for evaluation and plots. They never
enter trace fitting, write/protection weights, dependency construction, or model
updates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
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
from experiments.gco_math.gco_tiny_functional_dependency_field import (
    DependencyBasis,
    build_dependency_basis,
    build_parser as build_dependency_parser,
    normalized_damage,
    validate_args as validate_dependency_args,
)
from experiments.gco_math.gco_tiny_recurrent_trace_field import (
    EvidenceMoments,
    FunctionalTraceSolution,
    TraceSummary,
    concatenate_moments,
    encode_trace_summary,
    fit_functional_trace_field,
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
    recurrence_probability,
    reconstruction_confidence,
    relative_representation_drift,
    trace_evidence_moments,
    train_base,
    weighted_mse,
)


@dataclass
class AutonomousTraceState:
    summary: TraceSummary
    centers: torch.Tensor
    learned_mass: torch.Tensor
    event: int

    def validate(self) -> None:
        self.summary.validate()
        slots, dimension = self.summary.means.shape
        if self.centers.shape != (slots, dimension):
            raise ValueError("Trace centers and summary moments have incompatible shapes.")
        if self.learned_mass.shape != (slots,):
            raise ValueError("Learned trace mass must contain one value per slot.")
        if torch.any(self.learned_mass < 0.0):
            raise ValueError("Learned trace mass cannot be negative.")
        if torch.any(self.learned_mass > self.summary.weights + 1e-5):
            raise ValueError("Learned trace mass cannot exceed evidence mass.")
        for name, value in (
            ("centers", self.centers),
            ("learned_mass", self.learned_mass),
        ):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Trace state {name} contains non-finite values.")

    def persistent_scalars(self) -> int:
        self.validate()
        return self.summary.stored_scalars() + self.centers.numel() + self.learned_mass.numel()


@dataclass
class TraceTransition:
    current: EvidenceMoments
    write_weights: torch.Tensor
    delayed_support: torch.Tensor
    familiarity: torch.Tensor
    recurrence: torch.Tensor
    prior_inputs: torch.Tensor
    prior_targets: torch.Tensor
    prior_protection: torch.Tensor
    old_confidence: torch.Tensor
    old_attention: torch.Tensor
    current_attention: torch.Tensor
    training_evidence: EvidenceMoments
    solution: FunctionalTraceSolution


def compress_bounded_trace_moments(
    evidence: EvidenceMoments,
    solution: FunctionalTraceSolution,
) -> EvidenceMoments:
    """Compress evidence while representing genuinely inactive slots explicitly."""
    if evidence.means.ndim != 2 or evidence.means.shape[0] <= 0:
        raise ValueError("Bounded trace compression requires non-empty [N, D] evidence.")
    if solution.attention.shape != (evidence.means.shape[0], solution.centers.shape[0]):
        raise ValueError("Trace attention does not match evidence and slot counts.")
    responsibilities = evidence.weights.unsqueeze(1) * solution.attention
    masses = responsibilities.sum(dim=0)
    if not torch.isfinite(masses).all() or torch.any(masses < 0.0):
        raise FloatingPointError("Bounded trace compression produced invalid slot mass.")
    numerical_zero = torch.finfo(evidence.means.dtype).eps
    active = masses > numerical_zero
    stored_mass = masses.clamp_min(numerical_zero)
    weighted_sum = responsibilities.transpose(0, 1) @ evidence.means
    means = weighted_sum / stored_mass.unsqueeze(1)
    means = torch.where(active.unsqueeze(1), means, solution.centers)

    input_second = (
        evidence.covariances
        + evidence.means.unsqueeze(2) * evidence.means.unsqueeze(1)
    )
    second = torch.einsum("ns,ndk->sdk", responsibilities, input_second)
    second = second / stored_mass[:, None, None]
    covariances = second - means.unsqueeze(2) * means.unsqueeze(1)
    covariances = torch.where(
        active[:, None, None],
        covariances,
        torch.zeros_like(covariances),
    )
    covariances = 0.5 * (covariances + covariances.transpose(1, 2))
    diagonal = covariances.diagonal(dim1=1, dim2=2)
    if torch.any(diagonal < -1e-4):
        raise FloatingPointError(
            "Bounded trace covariance has a negative variance: "
            f"min={float(diagonal.min().item())}."
        )
    covariances = (
        covariances
        - torch.diag_embed(diagonal)
        + torch.diag_embed(diagonal.clamp_min(0.0))
    )
    return EvidenceMoments(
        means=means,
        covariances=covariances,
        weights=stored_mass,
    )


def validate_args(args: argparse.Namespace) -> None:
    validate_dependency_args(args)
    if args.event_batch_size <= 0:
        raise ValueError("--event-batch-size must be positive.")
    if not 0.0 < args.evidence_decay <= 1.0:
        raise ValueError("--evidence-decay must be in (0, 1].")
    if not math.isfinite(args.maturity_mass) or args.maturity_mass <= 0.0:
        raise ValueError("--maturity-mass must be positive and finite.")
    if args.line_search_steps <= 0:
        raise ValueError("--line-search-steps must be positive.")
    if not 0.0 < args.line_search_decay < 1.0:
        raise ValueError("--line-search-decay must be in (0, 1).")


def event_batches(
    stages: list[list[FunctionExample]],
    *,
    batch_size: int,
) -> list[tuple[int, list[FunctionExample]]]:
    """Construct an interleaved test stream; group names do not reach the learner."""
    batches: list[tuple[int, list[FunctionExample]]] = []
    for stage_number, stage in enumerate(stages[1:], start=2):
        names = sorted({example.hidden_group for example in stage})
        queues = {
            name: [example for example in stage if example.hidden_group == name]
            for name in names
        }
        cursor = 0
        while any(queues[name] for name in names):
            batch: list[FunctionExample] = []
            visited = 0
            while len(batch) < batch_size and visited < len(names):
                name = names[cursor % len(names)]
                cursor += 1
                visited += 1
                if queues[name]:
                    batch.append(queues[name].pop(0))
            if not batch:
                raise RuntimeError("Interleaved stream construction made no progress.")
            batches.append((stage_number, batch))
    if not batches:
        raise RuntimeError("The stream produced no online event batches.")
    return batches


def within_event_recurrence(
    means: torch.Tensor,
    *,
    attention_scale: float,
) -> torch.Tensor:
    if means.ndim != 2 or means.shape[0] <= 0:
        raise ValueError("Within-event recurrence expects non-empty [N, D] evidence.")
    if means.shape[0] == 1:
        return torch.zeros(1, device=means.device, dtype=means.dtype)
    return recurrence_probability(means, attention_scale=attention_scale)


def initialize_trace_state(
    base_examples: list[FunctionExample],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> AutonomousTraceState:
    evidence = trace_evidence_moments(base_examples, args=args, device=device)
    solution = fit_functional_trace_field(
        evidence,
        args=args,
        stage=1,
        previous_centers=None,
    )
    compressed = compress_bounded_trace_moments(evidence, solution)
    summary = encode_trace_summary(
        compressed,
        mode="trace",
        rank=1,
        power_iterations=1,
        seed=args.seed + 1,
    )
    state = AutonomousTraceState(
        summary=summary,
        centers=solution.centers.detach().clone(),
        learned_mass=summary.weights.detach().clone(),
        event=0,
    )
    state.validate()
    return state


def prepare_trace_transition(
    state: AutonomousTraceState,
    examples: list[FunctionExample],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> TraceTransition:
    state.validate()
    raw_current = trace_evidence_moments(examples, args=args, device=device)
    recurrence = within_event_recurrence(
        raw_current.means,
        attention_scale=args.attention_scale,
    )
    familiarity, _ = reconstruction_confidence(
        raw_current.means,
        state.centers,
        attention_scale=args.attention_scale,
    )
    prior = state.summary.to_moments()
    prior_attention = functional_attention(
        raw_current.means,
        state.centers,
        attention_scale=args.attention_scale,
    )
    slot_maturity = 1.0 - torch.exp(-state.summary.weights / args.maturity_mass)
    delayed_support = familiarity * (prior_attention @ slot_maturity)
    write_weights = 1.0 - (1.0 - delayed_support) * (1.0 - recurrence)
    current = raw_current

    decayed_prior = EvidenceMoments(
        means=prior.means,
        covariances=prior.covariances,
        weights=args.evidence_decay * prior.weights,
    )
    training_evidence = concatenate_moments([decayed_prior, current])
    solution = fit_functional_trace_field(
        training_evidence,
        args=args,
        stage=state.event + 2,
        previous_centers=state.centers,
    )
    old_confidence, _ = reconstruction_confidence(
        prior.means,
        solution.centers,
        attention_scale=args.attention_scale,
    )
    old_attention = functional_attention(
        prior.means,
        solution.centers,
        attention_scale=args.attention_scale,
    )
    current_attention = functional_attention(
        current.means,
        solution.centers,
        attention_scale=args.attention_scale,
    )

    learned_fraction = state.learned_mass / state.summary.weights.clamp_min(1e-12)
    statistical_maturity = 1.0 - torch.exp(-state.summary.weights / args.maturity_mass)
    prior_protection = (
        learned_fraction
        * statistical_maturity
        * old_confidence.pow(args.protection_power)
    )
    if prior_protection.sum() <= 0.0:
        raise RuntimeError("Autonomous trace state produced zero protected mass.")
    dimension = args.d_model
    transition = TraceTransition(
        current=current,
        write_weights=write_weights,
        delayed_support=delayed_support,
        familiarity=familiarity,
        recurrence=recurrence,
        prior_inputs=prior.means[:, :dimension],
        prior_targets=prior.means[:, dimension:] / args.trace_target_scale,
        prior_protection=prior_protection,
        old_confidence=old_confidence,
        old_attention=old_attention,
        current_attention=current_attention,
        training_evidence=training_evidence,
        solution=solution,
    )
    return transition


def geometry_loss(
    current_hidden: torch.Tensor,
    reference_distances: dict[tuple[int, int], torch.Tensor],
    probe_weights: torch.Tensor,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for (left, right), distance in hidden_pair_distances(current_hidden).items():
        weight = torch.sqrt(probe_weights[left] * probe_weights[right])
        terms.append(weight * (distance - reference_distances[(left, right)]).square())
    if not terms:
        raise RuntimeError("Protected probes produced no geometry terms.")
    return torch.stack(terms).mean()


def absolute_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")
    if weights.shape != (prediction.shape[0],):
        raise ValueError("Absolute write weights do not match the prediction batch.")
    if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
        raise ValueError("Absolute write weights must be finite and non-negative.")
    per_example = (prediction - target).square().mean(dim=1)
    return (per_example * weights).mean()


def snapshot_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def restore_parameters(
    parameters: list[torch.nn.Parameter],
    snapshot: list[torch.Tensor],
) -> None:
    if len(parameters) != len(snapshot):
        raise ValueError("Parameter snapshot length does not match the model.")
    with torch.no_grad():
        for parameter, reference in zip(parameters, snapshot, strict=True):
            if parameter.shape != reference.shape:
                raise ValueError("Parameter snapshot tensor shape does not match the model.")
            parameter.copy_(reference)


def apply_trial_update(
    parameters: list[torch.nn.Parameter],
    gradient: torch.Tensor,
    *,
    learning_rate: float,
    grad_clip: float,
) -> float:
    norm = torch.linalg.vector_norm(gradient)
    if not torch.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Candidate update has zero or non-finite gradient norm.")
    scale = torch.clamp(gradient.new_tensor(grad_clip) / norm, max=1.0)
    update = gradient * scale
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(update[offset : offset + count].reshape_as(parameter), alpha=-learning_rate)
            offset += count
    if offset != update.numel():
        raise RuntimeError("Candidate update did not consume the complete flat gradient.")
    return float(norm.detach().cpu())


@torch.no_grad()
def verification_objective(
    model: TinyFunctionModel,
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    write_weights: torch.Tensor,
    prior_inputs: torch.Tensor,
    prior_targets: torch.Tensor,
    prior_protection: torch.Tensor,
    reference_distances: dict[tuple[int, int], torch.Tensor],
    args: argparse.Namespace,
) -> tuple[float, torch.Tensor]:
    prediction, _ = model(inputs)
    per_example = (prediction - targets).square().mean(dim=1)
    new_loss = (per_example * write_weights).mean()
    old_output, old_hidden = model(prior_inputs)
    protection_loss = weighted_mse(old_output, prior_targets, prior_protection)
    state_loss = geometry_loss(old_hidden, reference_distances, prior_protection)
    objective = (
        new_loss
        + args.loss_mix_strength * protection_loss
        + args.geometry_mix_strength * state_loss
    )
    return float(objective.detach().cpu()), per_example.detach()


def train_event(
    model: TinyFunctionModel,
    examples: list[FunctionExample],
    transition: TraceTransition,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    parameters = list(model.parameters())
    inputs, targets = examples_to_tensors(examples, device=device)
    prior_inputs = transition.prior_inputs.to(device)
    prior_targets = transition.prior_targets.to(device)
    protection = transition.prior_protection.to(device)
    write_weights = transition.write_weights.to(device)
    with torch.no_grad():
        _reference_output, reference_hidden = model(prior_inputs)
        reference_distances = {
            pair: value.detach()
            for pair, value in hidden_pair_distances(reference_hidden).items()
        }
        _initial_objective, initial_errors = verification_objective(
            model,
            inputs=inputs,
            targets=targets,
            write_weights=write_weights,
            prior_inputs=prior_inputs,
            prior_targets=prior_targets,
            prior_protection=protection,
            reference_distances=reference_distances,
            args=args,
        )

    numerical_zero = torch.finfo(write_weights.dtype).eps
    if write_weights.max() <= numerical_zero:
        with torch.no_grad():
            old_output, old_hidden = model(prior_inputs)
            protection_loss = weighted_mse(old_output, prior_targets, protection)
            state_loss = geometry_loss(old_hidden, reference_distances, protection)
        return {
            "initial_error": float(initial_errors.mean().cpu()),
            "final_error": float(initial_errors.mean().cpu()),
            "verified_gain": 0.0,
            "deferred": True,
            "epochs": [
                {
                    "epoch": 0.0,
                    "new_loss": 0.0,
                    "protection_loss": float(protection_loss.detach().cpu()),
                    "geometry_loss": float(state_loss.detach().cpu()),
                    "gradient_norm": 0.0,
                    "accepted_learning_rate": 0.0,
                    "objective_before": _initial_objective,
                    "objective_after": _initial_objective,
                    "projection_removed_fraction": 0.0,
                    "safe_gradient_fraction": 0.0,
                    "raw_dependency_damage": 0.0,
                    "final_dependency_damage": 0.0,
                    "dependency_rank": 0.0,
                    "dependency_energy": 0.0,
                }
            ],
        }, torch.zeros_like(write_weights)

    basis: DependencyBasis | None = None
    epoch_trace: list[dict[str, float]] = []
    for epoch in range(1, args.cl_epochs + 1):
        if basis is None or (epoch - 1) % args.dependency_refresh == 0:
            basis = build_dependency_basis(
                model=model,
                probe_inputs=prior_inputs,
                probe_weights=protection,
                parameters=parameters,
                args=args,
            )

        prediction, _ = model(inputs)
        new_loss = absolute_weighted_mse(prediction, targets, write_weights)
        old_output, old_hidden = model(prior_inputs)
        protection_loss = weighted_mse(old_output, prior_targets, protection)
        state_loss = geometry_loss(old_hidden, reference_distances, protection)
        raw_gradient = flat_autograd_gradient(
            new_loss,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label=f"autonomous_event_{epoch}_new",
        )
        tangent, projection = project_gradient_away_from_constraints(
            raw_gradient=raw_gradient,
            constraint_gradients=basis.constraint_rows,
            damping=args.projection_damping,
            solver="gram",
            rank_tolerance=args.dependency_rank_tolerance,
            plasticity_audit=False,
        )
        restore_loss = (
            args.loss_mix_strength * protection_loss
            + args.geometry_mix_strength * state_loss
        )
        restore_gradient = flat_autograd_gradient(
            restore_loss,
            parameters,
            retain_graph=False,
            require_nonzero=False,
            label=f"autonomous_event_{epoch}_restore",
        )
        restore = bounded_restore(
            restore_gradient,
            tangent,
            strength=args.restore_strength,
            bound_fraction=args.restore_bound_fraction,
        )
        final_gradient = tangent + restore
        before, _ = verification_objective(
            model,
            inputs=inputs,
            targets=targets,
            write_weights=write_weights,
            prior_inputs=prior_inputs,
            prior_targets=prior_targets,
            prior_protection=protection,
            reference_distances=reference_distances,
            args=args,
        )
        snapshot = snapshot_parameters(parameters)
        accepted_rate: float | None = None
        gradient_norm = 0.0
        after = float("inf")
        for trial in range(args.line_search_steps):
            restore_parameters(parameters, snapshot)
            rate = args.cl_lr * args.line_search_decay**trial
            gradient_norm = apply_trial_update(
                parameters,
                final_gradient,
                learning_rate=rate,
                grad_clip=args.grad_clip,
            )
            after, _ = verification_objective(
                model,
                inputs=inputs,
                targets=targets,
                write_weights=write_weights,
                prior_inputs=prior_inputs,
                prior_targets=prior_targets,
                prior_protection=protection,
                reference_distances=reference_distances,
                args=args,
            )
            if math.isfinite(after) and after <= before:
                accepted_rate = rate
                break
        if accepted_rate is None:
            restore_parameters(parameters, snapshot)
            raise RuntimeError(
                f"No verified update found at event epoch {epoch}; "
                f"objective_before={before:.6f}, last_after={after:.6f}."
            )
        if epoch in {1, args.cl_epochs} or epoch % args.print_every == 0:
            epoch_trace.append(
                {
                    "epoch": float(epoch),
                    "new_loss": float(new_loss.detach().cpu()),
                    "protection_loss": float(protection_loss.detach().cpu()),
                    "geometry_loss": float(state_loss.detach().cpu()),
                    "gradient_norm": gradient_norm,
                    "accepted_learning_rate": accepted_rate,
                    "objective_before": before,
                    "objective_after": after,
                    "projection_removed_fraction": projection["projection_removed_fraction"],
                    "safe_gradient_fraction": projection["safe_grad_fraction"],
                    "raw_dependency_damage": normalized_damage(
                        basis.normalized_measurement_matrix,
                        raw_gradient,
                    ),
                    "final_dependency_damage": normalized_damage(
                        basis.normalized_measurement_matrix,
                        final_gradient,
                    ),
                    "dependency_rank": float(basis.parameter_retained_rank),
                    "dependency_energy": basis.parameter_retained_energy,
                }
            )

    with torch.no_grad():
        _final_objective, final_errors = verification_objective(
            model,
            inputs=inputs,
            targets=targets,
            write_weights=write_weights,
            prior_inputs=prior_inputs,
            prior_targets=prior_targets,
            prior_protection=protection,
            reference_distances=reference_distances,
            args=args,
        )
    gain = torch.clamp(
        (initial_errors - final_errors) / initial_errors.clamp_min(1e-12),
        min=0.0,
        max=1.0,
    )
    return {
        "initial_error": float(initial_errors.mean().cpu()),
        "final_error": float(final_errors.mean().cpu()),
        "verified_gain": float(gain.mean().cpu()),
        "deferred": False,
        "epochs": epoch_trace,
    }, gain


def commit_trace_transition(
    state: AutonomousTraceState,
    transition: TraceTransition,
    verified_gain: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> AutonomousTraceState:
    if verified_gain.shape != transition.write_weights.shape:
        raise ValueError("Verified gain does not match the current event batch.")
    compressed = compress_bounded_trace_moments(
        transition.training_evidence,
        transition.solution,
    )
    summary = encode_trace_summary(
        compressed,
        mode="trace",
        rank=1,
        power_iterations=1,
        seed=args.seed + state.event + 2,
    )
    old_verified = (
        args.evidence_decay
        * state.learned_mass
        * transition.old_confidence
    ).unsqueeze(1) * transition.old_attention
    current_verified = (
        transition.write_weights * verified_gain
    ).unsqueeze(1) * transition.current_attention
    learned_mass = old_verified.sum(dim=0) + current_verified.sum(dim=0)
    learned_mass = torch.minimum(learned_mass, summary.weights)
    new_state = AutonomousTraceState(
        summary=summary,
        centers=transition.solution.centers.detach().clone(),
        learned_mass=learned_mass.detach(),
        event=state.event + 1,
    )
    new_state.validate()
    return new_state


def evaluation_groups(examples: list[FunctionExample]) -> list[str]:
    return sorted({example.hidden_group for example in examples})


def active_trace_slots(state: AutonomousTraceState) -> int:
    numerical_zero = torch.finfo(state.summary.weights.dtype).eps
    return int((state.summary.weights > numerical_zero).sum().item())


def trace_mass_report(state: AutonomousTraceState) -> dict[str, Any]:
    pending = (state.summary.weights - state.learned_mass).clamp_min(0.0)
    total = state.summary.weights.sum().clamp_min(torch.finfo(state.summary.weights.dtype).eps)
    return {
        "evidence_mass": [float(value) for value in state.summary.weights.detach().cpu()],
        "learned_mass": [float(value) for value in state.learned_mass.detach().cpu()],
        "pending_mass": [float(value) for value in pending.detach().cpu()],
        "pending_fraction": float((pending.sum() / total).detach().cpu()),
    }


def group_means(
    examples: list[FunctionExample],
    values: torch.Tensor,
) -> dict[str, float]:
    if values.shape != (len(examples),):
        raise ValueError("Reported values do not match event examples.")
    result: dict[str, float] = {}
    for group in evaluation_groups(examples):
        indices = torch.tensor(
            [index for index, example in enumerate(examples) if example.hidden_group == group],
            device=values.device,
            dtype=torch.long,
        )
        result[group] = float(values[indices].mean().detach().cpu())
    return result


def trace_group_report(
    state: AutonomousTraceState,
    examples: list[FunctionExample],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    moments = trace_evidence_moments(examples, args=args, device=device)
    attention = functional_attention(
        moments.means,
        state.centers,
        attention_scale=args.attention_scale,
    )
    reconstruction = attention @ state.centers
    errors = (moments.means - reconstruction).square().mean(dim=1)
    report: dict[str, dict[str, float]] = {}
    for group in evaluation_groups(examples):
        indices = torch.tensor(
            [index for index, example in enumerate(examples) if example.hidden_group == group],
            device=device,
            dtype=torch.long,
        )
        group_attention = attention[indices].mean(dim=0)
        dominant = int(group_attention.argmax().item())
        report[group] = {
            "trace_error": float(errors[indices].mean().detach().cpu()),
            "dominant_slot": float(dominant),
            "dominant_share": float(group_attention[dominant].detach().cpu()),
        }
    return report


def plot_behavior(
    events: list[dict[str, Any]],
    groups: list[str],
    *,
    output_path: Path,
) -> None:
    x = [event["event"] for event in events]
    fig, axis = plt.subplots(figsize=(11.5, 5.2))
    for group in groups:
        axis.plot(
            x,
            [event["evaluation"]["groups"][group]["mse"] for event in events],
            marker="o",
            markersize=3,
            label=group.replace("_", " "),
        )
    axis.set_title("Behavior through the autonomous continual-learning stream")
    axis.set_xlabel("online event")
    axis.set_ylabel("mean squared error")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trace_lifecycle(events: list[dict[str, Any]], *, output_path: Path) -> None:
    x = [event["event"] for event in events[1:]]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.2), sharex=True)
    for key, label, color in (
        ("write_mean", "write", "#2563eb"),
        ("support_mean", "delayed support", "#0891b2"),
        ("protection_mean", "protect", "#0f9d58"),
        ("release_mean", "release", "#dc2626"),
        ("gain_mean", "verified gain", "#7c3aed"),
    ):
        axes[0].plot(x, [event["trace"][key] for event in events[1:]], label=label, color=color)
    axes[0].set_ylabel("continuous weight")
    axes[0].set_title("Autonomous trace lifecycle")
    axes[0].legend(ncol=4)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        [event["event"] for event in events],
        [sum(event["trace"]["evidence_mass"]) for event in events],
        label="evidence mass",
        color="#2563eb",
    )
    axes[1].plot(
        [event["event"] for event in events],
        [sum(event["trace"]["learned_mass"]) for event in events],
        label="verified learned mass",
        color="#0f9d58",
    )
    axes[1].plot(
        [event["event"] for event in events],
        [sum(event["trace"]["pending_mass"]) for event in events],
        label="pending mass",
        color="#f59e0b",
    )
    axes[1].set_ylabel("total trace mass")
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.25)

    slot_count = len(events[-1]["trace"]["evidence_mass"])
    for slot in range(slot_count):
        axes[2].plot(
            [event["event"] for event in events],
            [event["trace"]["learned_mass"][slot] for event in events],
            label=f"slot {slot}",
        )
    axes[2].set_xlabel("online event")
    axes[2].set_ylabel("per-slot learned mass")
    axes[2].grid(alpha=0.25)
    axes[2].legend(ncol=slot_count, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_geometry(events: list[dict[str, Any]], *, output_path: Path) -> None:
    x = [event["event"] for event in events]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(x, [event["geometry"]["cka"] for event in events], color="#2563eb")
    axes[0].set_title("Base-function representation similarity")
    axes[0].set_xlabel("online event")
    axes[0].set_ylabel("hidden CKA")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        x,
        [event["geometry"]["hidden_drift"] for event in events],
        label="hidden drift",
        color="#dc2626",
    )
    axes[1].plot(
        x,
        [event["geometry"]["pair_drift"] for event in events],
        label="pair geometry drift",
        color="#f59e0b",
    )
    axes[1].set_title("Geometry movement")
    axes[1].set_xlabel("online event")
    axes[1].set_ylabel("relative drift")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_capacity(events: list[dict[str, Any]], *, output_path: Path) -> None:
    online = events[1:]
    x = [event["event"] for event in online]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(
        x,
        [event["trace"]["effective_slots"] for event in online],
        label="effective slots",
        color="#2563eb",
    )
    axes[0].plot(
        x,
        [event["trace"]["active_slots"] for event in online],
        label="active slots",
        color="#0f9d58",
    )
    storage_axis = axes[0].twinx()
    storage_axis.plot(
        x,
        [event["trace"]["persistent_scalars"] for event in online],
        label="stored scalars",
        color="#f59e0b",
    )
    axes[0].set_title("Bounded trace capacity")
    axes[0].set_xlabel("online event")
    axes[0].set_ylabel("slot count")
    storage_axis.set_ylabel("persistent scalars")
    lines = axes[0].lines + storage_axis.lines
    axes[0].legend(lines, [line.get_label() for line in lines])
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, [event["update"]["dependency_rank"] for event in online], label="dependency rank")
    axes[1].plot(
        x,
        [event["update"]["projection_removed_fraction"] for event in online],
        label="gradient removed",
    )
    axes[1].plot(
        x,
        [event["update"]["damage_ratio"] for event in online],
        label="damage after / before",
    )
    axes[1].set_title("Executable protection constraints")
    axes[1].set_xlabel("online event")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def geometry_metrics(
    model: TinyFunctionModel,
    base_inputs: torch.Tensor,
    base_hidden: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        _output, hidden = model(base_inputs)
    hidden_drift, pair_drift = relative_representation_drift(base_hidden, hidden)
    return {
        "cka": centered_kernel_alignment(base_hidden, hidden),
        "hidden_drift": hidden_drift,
        "pair_drift": pair_drift,
    }


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    stages = build_function_stream(args)
    batches = event_batches(stages, batch_size=args.event_batch_size)
    all_examples = [example for stage in stages for example in stage]
    groups = evaluation_groups(all_examples)

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
    base_inputs, _base_targets = examples_to_tensors(stages[0], device=device)
    with torch.no_grad():
        _base_output, base_hidden = model(base_inputs)
    state = initialize_trace_state(stages[0], args=args, device=device)

    initial_evaluation = evaluate_model(model, all_examples, device=device)
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
                "effective_slots": state.summary.weights.numel(),
                "active_slots": active_trace_slots(state),
                "persistent_scalars": state.persistent_scalars(),
            },
            "update": {
                "deferred": False,
                "dependency_rank": 0.0,
                "projection_removed_fraction": 0.0,
                "damage_ratio": 0.0,
            },
        }
    ]

    print("TINY AUTONOMOUS DEPENDENCY + INVARIANT-TANGENT CL")
    print("=" * 144)
    print(
        f"device={device} params={sum(parameter.numel() for parameter in model.parameters())} "
        f"slots={args.num_slots} batches={len(batches)} stored_scalars={state.persistent_scalars()}"
    )
    for event_number, (stage_number, examples) in enumerate(batches, start=1):
        transition = prepare_trace_transition(
            state,
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
            state,
            transition,
            gain,
            args=args,
        )
        evaluation = evaluate_model(model, all_examples, device=device)
        last_epoch = training["epochs"][-1]
        damage_before = last_epoch["raw_dependency_damage"]
        damage_after = last_epoch["final_dependency_damage"]
        groups_seen = sorted({example.hidden_group for example in examples})
        event_report = {
            "event": event_number,
            "stage": stage_number,
            "groups_seen": groups_seen,
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
                "write_by_group": group_means(examples, transition.write_weights),
                "support_by_group": group_means(examples, transition.delayed_support),
                "gain_by_group": group_means(examples, gain),
                "familiarity_by_group": group_means(examples, transition.familiarity),
                "recurrence_by_group": group_means(examples, transition.recurrence),
                **trace_mass_report(state),
                "effective_slots": transition.solution.effective_slots,
                "active_slots": active_trace_slots(state),
                "persistent_scalars": state.persistent_scalars(),
            },
            "update": {
                "deferred": training["deferred"],
                "initial_error": training["initial_error"],
                "final_error": training["final_error"],
                "verified_gain": training["verified_gain"],
                "dependency_rank": last_epoch["dependency_rank"],
                "dependency_energy": last_epoch["dependency_energy"],
                "projection_removed_fraction": last_epoch["projection_removed_fraction"],
                "safe_gradient_fraction": last_epoch["safe_gradient_fraction"],
                "raw_dependency_damage": damage_before,
                "final_dependency_damage": damage_after,
                "damage_ratio": damage_after / max(damage_before, 1e-12),
                "epochs": training["epochs"],
            },
        }
        events.append(event_report)
        print(
            f"event={event_number:02d} stage={stage_number} groups={','.join(groups_seen)} "
            f"write={event_report['trace']['write_mean']:.3f} "
            f"support={event_report['trace']['support_mean']:.3f} "
            f"protect={event_report['trace']['protection_mean']:.3f} "
            f"release={event_report['trace']['release_mean']:.3f} "
            f"gain={event_report['trace']['gain_mean']:.3f} "
            f"deferred={int(event_report['update']['deferred'])} "
            f"damage={damage_before:.3f}->{damage_after:.3f}"
        )

    final_evaluation = evaluate_model(model, all_examples, device=device)
    trace_groups = trace_group_report(
        state,
        all_examples,
        args=args,
        device=device,
    )
    final_geometry = geometry_metrics(model, base_inputs, base_hidden)
    print("\nFINAL AUTONOMOUS CL STATE")
    print("-" * 144)
    print(f"{'group':>18} {'model_mse':>12} {'trace_error':>12} {'slot':>7} {'share':>9}")
    for group in groups:
        print(
            f"{group:>18} {final_evaluation['groups'][group]['mse']:12.5f} "
            f"{trace_groups[group]['trace_error']:12.5f} "
            f"{int(trace_groups[group]['dominant_slot']):7d} "
            f"{trace_groups[group]['dominant_share']:9.4f}"
        )
    print(
        f"geometry cka={final_geometry['cka']:.4f} "
        f"hidden_drift={final_geometry['hidden_drift']:.4f} "
        f"pair_drift={final_geometry['pair_drift']:.4f} "
        f"stored_scalars={state.persistent_scalars()}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.output_dir / "autonomous_behavior.png"
    lifecycle_path = args.output_dir / "autonomous_trace_lifecycle.png"
    geometry_path = args.output_dir / "autonomous_geometry.png"
    capacity_path = args.output_dir / "autonomous_capacity_dependency.png"
    output_json = args.output_dir / "autonomous_dependency_invariant_cl.json"
    plot_behavior(events, groups, output_path=behavior_path)
    plot_trace_lifecycle(events, output_path=lifecycle_path)
    plot_geometry(events, output_path=geometry_path)
    plot_capacity(events, output_path=capacity_path)
    output = {
        "question": (
            "Can a fixed-size autonomous trace field decide continuous write, protection, and release "
            "strength, translate mature traces into dependency constraints, and drive Invariant-Tangent "
            "updates without training-time role labels?"
        ),
        "scope": (
            "Single integrated tiny-model mechanism test. Hidden semantic groups are evaluation-only. "
            "This is not a real-language or scale claim."
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "base_training": {
            "initial_loss": base_trace[0],
            "final_loss": base_trace[-1],
        },
        "events": events,
        "final": {
            "evaluation": {
                "mse": final_evaluation["mse"],
                "groups": final_evaluation["groups"],
            },
            "geometry": final_geometry,
            "trace_groups": trace_groups,
            **trace_mass_report(state),
            "persistent_scalars": state.persistent_scalars(),
            "active_slots": active_trace_slots(state),
        },
        "plots": {
            "behavior": str(behavior_path),
            "trace_lifecycle": str(lifecycle_path),
            "geometry": str(geometry_path),
            "capacity_dependency": str(capacity_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_plots={behavior_path},{lifecycle_path},{geometry_path},{capacity_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = build_dependency_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=Path(
            "model/analysis/gco-tiny-autonomous-delayed-dependency-invariant-cl-seed0"
        ),
        trace_steps=80,
        restarts=2,
        cl_epochs=30,
        dependency_refresh=10,
        print_every=10,
        merge_points=12,
        stable_points=16,
        root_points=12,
        obsolete_points=4,
        branch_points=12,
        novel_points=16,
        noise_points=8,
    )
    parser.add_argument("--event-batch-size", type=int, default=1)
    parser.add_argument("--evidence-decay", type=float, default=0.98)
    parser.add_argument("--maturity-mass", type=float, default=2.0)
    parser.add_argument("--line-search-steps", type=int, default=8)
    parser.add_argument("--line-search-decay", type=float, default=0.5)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
