"""Run consequence-guided continual learning on an approximately 1M-weight transformer.

This is the first scale-transition experiment for the current architecture. It
uses a fitted 5,000-word checkpoint, a bounded executable trace bank, first-order
functional consequence sketches for trace survival, and the dependency-field
Invariant-Tangent update with bounded restore. Evaluation labels never enter
trace selection, survival utility, or gradient control.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
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
from experiments.gco_math.gco_mini_cl_world_demo import (
    flat_autograd_gradient,
    project_gradient_away_from_constraints,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import trainable_weight_parameters
from experiments.gco_math.gco_tiny_dependency_utility_survival import (
    allocate_survival,
    rms_normalize,
)
from experiments.gco_math.gco_tiny_text_dependency_cl import (
    ExecutableTraceBank,
    FactQuery,
    TextWindows,
    build_parser as build_text_parser,
    build_constraint_basis,
    build_staged_data,
    collect_geometry,
    commit_trace_bank,
    encode_trace_evidence,
    evaluate_correction_queries,
    evaluate_windows,
    forward_with_states,
    geometry_report,
    initialize_trace_bank,
    instantiate_model,
    normalized_damage,
    protection_losses,
    propose_trace_update,
    validate_args as validate_text_args,
)
from experiments.gco_math.gco_tiny_trace_invariant_tangent_integration import (
    apply_flat_update,
    bounded_restore,
    recurrence_probability,
    reconstruction_confidence,
)


def validate_args(args: argparse.Namespace) -> None:
    validate_text_args(args)
    if args.min_trainable_parameters <= 0:
        raise ValueError("--min-trainable-parameters must be positive.")
    if args.max_trainable_parameters < args.min_trainable_parameters:
        raise ValueError("--max-trainable-parameters must be >= the minimum.")
    if not 0.0 < args.survival_budget <= args.trace_slots:
        raise ValueError("--survival-budget must be in (0, trace-slots].")
    if args.survival_temperature <= 0.0 or not math.isfinite(args.survival_temperature):
        raise ValueError("--survival-temperature must be positive and finite.")
    if args.survival_bisection_steps <= 0:
        raise ValueError("--survival-bisection-steps must be positive.")
    if args.micro_batch_windows <= 0:
        raise ValueError("--micro-batch-windows must be positive.")
    if args.correction_margin < 0.0 or not math.isfinite(args.correction_margin):
        raise ValueError("--correction-margin must be finite and non-negative.")
    if not 0.0 < args.guard_min_cka <= 1.0:
        raise ValueError("--guard-min-cka must be in (0, 1].")
    for name in (
        "correction_query_weight",
        "correction_suppression_weight",
        "commit_min_relative_gain",
        "guard_loss_ratio",
        "guard_loss_absolute",
    ):
        value = getattr(args, name)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")
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


def model_width(model: torch.nn.Module) -> int:
    weight = model.token_embedding.W
    if weight.ndim != 2 or weight.shape[1] <= 0:
        raise RuntimeError("Token embedding weight does not expose a valid model width.")
    return int(weight.shape[1])


def leave_one_out_evidence(
    vectors: torch.Tensor,
    *,
    attention_scale: float,
) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("Leave-one-out evidence requires at least two trace vectors.")
    values: list[torch.Tensor] = []
    for trace in range(vectors.shape[0]):
        keep = torch.arange(vectors.shape[0], device=vectors.device) != trace
        _confidence, squared_error = reconstruction_confidence(
            vectors[trace : trace + 1],
            vectors[keep],
            attention_scale=attention_scale,
        )
        values.append(squared_error[0])
    return torch.stack(values)


def incoming_conflict(
    old_vectors: torch.Tensor,
    current_vectors: torch.Tensor,
    *,
    d_model: int,
    attention_scale: float,
) -> torch.Tensor:
    if old_vectors.shape[1] != 3 * d_model or current_vectors.shape[1] != 3 * d_model:
        raise ValueError("Text trace vectors do not match the expected three representation blocks.")
    old_state = old_vectors[:, :d_model]
    current_state = current_vectors[:, :d_model]
    state_distance = (
        old_state[:, None, :] - current_state[None, :, :]
    ).square().sum(dim=2)
    state_match = torch.exp(-state_distance / (2.0 * attention_scale**2))
    old_target = old_vectors[:, d_model : 2 * d_model]
    current_target = current_vectors[:, d_model : 2 * d_model]
    target_similarity = old_target @ current_target.T
    target_conflict = (1.0 - target_similarity).clamp(0.0, 2.0) * 0.5
    return (state_match * target_conflict).max(dim=1).values


def preliminary_write_weights(
    old_vectors: torch.Tensor,
    current_vectors: torch.Tensor,
    *,
    attention_scale: float,
) -> torch.Tensor:
    familiarity, _error = reconstruction_confidence(
        current_vectors,
        old_vectors,
        attention_scale=attention_scale,
    )
    recurrence = recurrence_probability(
        current_vectors,
        attention_scale=attention_scale,
    )
    return 1.0 - (1.0 - familiarity) * (1.0 - recurrence)


def trace_gradient_sketches(
    model: torch.nn.Module,
    bank: ExecutableTraceBank,
    *,
    parameters: list[torch.nn.Parameter],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, _states = forward_with_states(model, bank.inputs.to(device))
    final_logits = logits[:, -1]
    target_ids = bank.targets[:, -1].to(device)
    reference_logits = bank.reference_logits.to(device)
    competitors = reference_logits.clone()
    competitors.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
    competitor_ids = competitors.argmax(dim=1)
    rows: list[torch.Tensor] = []
    for trace in range(final_logits.shape[0]):
        margin = final_logits[trace, target_ids[trace]] - final_logits[
            trace, competitor_ids[trace]
        ]
        row = flat_autograd_gradient(
            margin,
            parameters,
            retain_graph=trace + 1 < final_logits.shape[0],
            require_nonzero=True,
            label=f"scaled_trace_margin_{trace}",
        ).detach()
        norm = torch.linalg.vector_norm(row)
        if not torch.isfinite(norm) or norm <= torch.finfo(row.dtype).eps:
            raise FloatingPointError(f"Trace {trace} produced an invalid gradient sketch.")
        rows.append(row / norm)
    target_probability = F.softmax(final_logits, dim=-1).gather(
        1,
        target_ids.unsqueeze(1),
    ).squeeze(1)
    return torch.stack(rows), target_probability.detach()


def consequence_survival(
    model: torch.nn.Module,
    bank: ExecutableTraceBank,
    current: TextWindows,
    correction_queries: list[FactQuery],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    parameters = trainable_weight_parameters(model)
    bank_windows = TextWindows(bank.inputs, bank.targets, bank.groups)
    old_vectors = encode_trace_evidence(model, bank_windows, args=args, device=device)
    current_vectors = encode_trace_evidence(model, current, args=args, device=device)
    write_weights = preliminary_write_weights(
        old_vectors,
        current_vectors,
        attention_scale=args.attention_scale,
    )

    inputs = current.inputs.to(device)
    targets = current.targets.to(device)
    logits, _states = forward_with_states(model, inputs)
    language_loss = absolute_weighted_language_loss(
        logits,
        targets,
        write_weights,
        surprise_power=args.token_surprise_power,
    )
    correction_loss, correction_margin, _correction_nll = correction_objective(
        model,
        correction_queries,
        args=args,
        device=device,
    )
    new_loss = language_loss + correction_loss
    raw_gradient = flat_autograd_gradient(
        new_loss,
        parameters,
        retain_graph=False,
        require_nonzero=True,
        label="scaled_consequence_new_gradient",
    ).detach()
    raw_norm = torch.linalg.vector_norm(raw_gradient)
    if not torch.isfinite(raw_norm) or raw_norm <= torch.finfo(raw_gradient.dtype).eps:
        raise FloatingPointError("Incoming text gradient is zero or non-finite.")
    raw_direction = raw_gradient / raw_norm

    sketches, knownness = trace_gradient_sketches(
        model,
        bank,
        parameters=parameters,
        device=device,
    )
    overlap = (sketches @ sketches.T).square()
    direct = (sketches @ raw_direction).abs()
    mass_weights = bank.masses.to(device)
    normalized_mass = mass_weights / mass_weights.sum().clamp_min(1e-12)
    centrality = (overlap * normalized_mass.unsqueeze(0)).sum(dim=1)
    centrality = centrality - overlap.diagonal() * normalized_mass
    downstream = direct * centrality
    evidence = leave_one_out_evidence(
        old_vectors,
        attention_scale=args.attention_scale,
    )
    conflict = incoming_conflict(
        old_vectors,
        current_vectors,
        d_model=model_width(model),
        attention_scale=args.attention_scale,
    )
    components = {
        "evidence": rms_normalize(evidence),
        "verified": rms_normalize(knownness),
        "direct": rms_normalize(direct),
        "downstream": rms_normalize(downstream),
        "centrality": rms_normalize(centrality),
        "conflict": rms_normalize(conflict),
    }
    utility = (
        args.utility_evidence_weight * components["evidence"]
        + args.utility_verified_weight * components["verified"]
        + args.utility_direct_weight * components["direct"]
        + args.utility_downstream_weight * components["downstream"]
        + args.utility_centrality_weight * components["centrality"]
        - args.utility_conflict_weight * components["conflict"]
    )
    survival = allocate_survival(
        utility,
        budget=args.survival_budget,
        temperature=args.survival_temperature,
        bisection_steps=args.survival_bisection_steps,
    ).detach()
    report = {
        "new_loss": float(new_loss.detach().cpu()),
        "language_loss": float(language_loss.detach().cpu()),
        "correction_loss": float(correction_loss.detach().cpu()),
        "correction_margin": float(correction_margin.detach().cpu()),
        "raw_gradient_norm": float(raw_norm.detach().cpu()),
        "survival": [float(value) for value in survival.detach().cpu()],
        "utility": [float(value) for value in utility.detach().cpu()],
        "components": {
            name: [float(value) for value in values.detach().cpu()]
            for name, values in components.items()
        },
        "raw": {
            "evidence": [float(value) for value in evidence.detach().cpu()],
            "knownness": [float(value) for value in knownness.detach().cpu()],
            "direct": [float(value) for value in direct.detach().cpu()],
            "downstream": [float(value) for value in downstream.detach().cpu()],
            "centrality": [float(value) for value in centrality.detach().cpu()],
            "conflict": [float(value) for value in conflict.detach().cpu()],
        },
        "dependency_overlap": overlap.detach().cpu().tolist(),
        "preliminary_write_mean": float(write_weights.mean().detach().cpu()),
    }
    return survival, report


def split_windows(windows: TextWindows, *, batch_size: int) -> list[TextWindows]:
    if batch_size <= 0:
        raise ValueError("Micro-batch size must be positive.")
    batches: list[TextWindows] = []
    for start in range(0, windows.inputs.shape[0], batch_size):
        end = min(start + batch_size, windows.inputs.shape[0])
        batches.append(
            TextWindows(
                inputs=windows.inputs[start:end].clone(),
                targets=windows.targets[start:end].clone(),
                groups=windows.groups[start:end],
            )
        )
    if not batches:
        raise RuntimeError("Text stage produced no micro-batches.")
    return batches


def permute_windows(windows: TextWindows, *, seed: int) -> TextWindows:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(windows.inputs.shape[0], generator=generator)
    indices = order.tolist()
    return TextWindows(
        inputs=windows.inputs[order].clone(),
        targets=windows.targets[order].clone(),
        groups=tuple(windows.groups[index] for index in indices),
    )


def absolute_weighted_language_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    write_weights: torch.Tensor,
    *,
    surprise_power: float,
) -> torch.Tensor:
    if write_weights.shape != (logits.shape[0],):
        raise ValueError("Write weights do not match the text micro-batch.")
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    with torch.no_grad():
        target_probability = torch.exp(-per_token).clamp(0.0, 1.0)
        token_weights = (1.0 - target_probability).pow(surprise_power)
        token_weights = token_weights / token_weights.mean(dim=1, keepdim=True).clamp_min(1e-12)
    per_window = (per_token * token_weights).mean(dim=1)
    return (per_window * write_weights).mean()


def correction_objective(
    model: torch.nn.Module,
    queries: list[FactQuery],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not queries:
        zero = next(model.parameters()).new_zeros(())
        return zero, zero, zero
    grouped: dict[int, list[FactQuery]] = {}
    for query in queries:
        grouped.setdefault(len(query.input_ids), []).append(query)
    likelihood_terms: list[torch.Tensor] = []
    suppression_terms: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for length, group in grouped.items():
        inputs = torch.tensor(
            [query.input_ids for query in group],
            dtype=torch.long,
            device=device,
        )
        if inputs.shape[1] != length:
            raise RuntimeError("Correction query length grouping failed.")
        logits = model(inputs)[:, -1]
        rows = torch.arange(len(group), device=device)
        new_ids = torch.tensor([query.new_target_id for query in group], device=device)
        old_ids = torch.tensor([query.old_target_id for query in group], device=device)
        log_probabilities = F.log_softmax(logits, dim=-1)
        likelihood_terms.append(-log_probabilities[rows, new_ids])
        margin = logits[rows, new_ids] - logits[rows, old_ids]
        margins.append(margin)
        suppression_terms.append(F.relu(args.correction_margin - margin))
    likelihood = torch.cat(likelihood_terms).mean()
    suppression = torch.cat(suppression_terms).mean()
    margin = torch.cat(margins).mean()
    objective = args.correction_query_weight * (
        likelihood + args.correction_suppression_weight * suppression
    )
    return objective, margin, likelihood


def snapshot_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def restore_parameters(
    parameters: list[torch.nn.Parameter],
    snapshot: list[torch.Tensor],
) -> None:
    if len(parameters) != len(snapshot):
        raise ValueError("Parameter snapshot length mismatch.")
    with torch.no_grad():
        for parameter, value in zip(parameters, snapshot, strict=True):
            if parameter.shape != value.shape:
                raise ValueError("Parameter snapshot shape mismatch.")
            parameter.copy_(value)


def train_microstage(
    *,
    model: torch.nn.Module,
    bank: ExecutableTraceBank,
    current: TextWindows,
    control: Any,
    correction_queries: list[FactQuery],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    parameters = trainable_weight_parameters(model)
    inputs = current.inputs.to(device)
    targets = current.targets.to(device)
    write_weights = control.write_weights.to(device)
    protection = control.protection_weights.to(device)
    basis = None
    trace: list[dict[str, Any]] = []
    for epoch in range(1, args.cl_epochs + 1):
        logits, _states = forward_with_states(model, inputs)
        language_loss = absolute_weighted_language_loss(
            logits,
            targets,
            write_weights,
            surprise_power=args.token_surprise_power,
        )
        correction_loss, correction_margin, correction_nll = correction_objective(
            model,
            correction_queries,
            args=args,
            device=device,
        )
        candidate_loss = language_loss + correction_loss
        behavior_loss, state_loss, _old_logits, _old_states = protection_losses(
            model,
            bank,
            protection,
            args=args,
            device=device,
        )
        if basis is None or (epoch - 1) % args.dependency_refresh == 0:
            basis = build_constraint_basis(
                method="dependency_field",
                model=model,
                bank=bank,
                protection=protection,
                parameters=parameters,
                args=args,
                device=device,
            )
        raw = flat_autograd_gradient(
            candidate_loss,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label="microstage_candidate",
        )
        tangent, stats = project_gradient_away_from_constraints(
            raw_gradient=raw,
            constraint_gradients=basis.rows,
            damping=args.projection_damping,
            solver="gram",
            rank_tolerance=args.dependency_rank_tolerance,
            plasticity_audit=False,
        )
        restore_gradient = flat_autograd_gradient(
            behavior_loss + args.geometry_restore_weight * state_loss,
            parameters,
            retain_graph=False,
            require_nonzero=False,
            label="microstage_restore",
        )
        restore = bounded_restore(
            restore_gradient,
            tangent,
            strength=args.restore_strength,
            bound_fraction=args.restore_bound_fraction,
        )
        final_gradient = tangent + restore
        raw_damage = normalized_damage(basis.measurement_matrix, raw)
        safe_damage = normalized_damage(basis.measurement_matrix, final_gradient)
        gradient_norm = apply_flat_update(
            parameters,
            final_gradient,
            learning_rate=args.cl_lr,
            grad_clip=args.grad_clip,
        )
        if epoch in {1, args.cl_epochs} or epoch % args.print_every == 0:
            trace.append(
                {
                    "epoch": epoch,
                    "candidate_loss": float(candidate_loss.detach().cpu()),
                    "language_loss": float(language_loss.detach().cpu()),
                    "correction_loss": float(correction_loss.detach().cpu()),
                    "correction_margin": float(correction_margin.detach().cpu()),
                    "correction_nll": float(correction_nll.detach().cpu()),
                    "behavior_loss": float(behavior_loss.detach().cpu()),
                    "state_loss": float(state_loss.detach().cpu()),
                    "gradient_norm": gradient_norm,
                    "projection_removed_fraction": stats["projection_removed_fraction"],
                    "safe_grad_fraction": stats["safe_grad_fraction"],
                    "raw_damage": raw_damage,
                    "safe_damage": safe_damage,
                    "basis": basis.report,
                }
            )
    return trace


@torch.no_grad()
def candidate_measurement(
    model: torch.nn.Module,
    current: TextWindows,
    write_weights: torch.Tensor,
    correction_queries: list[FactQuery],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
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
    return {
        "objective": float((language + correction).detach().cpu()),
        "language": float(language.detach().cpu()),
        "correction": float(correction.detach().cpu()),
        "margin": float(margin.detach().cpu()),
        "nll": float(nll.detach().cpu()),
    }


def evaluate_all(
    model: torch.nn.Module,
    evaluation: dict[str, TextWindows],
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    return {
        name: evaluate_windows(model, windows, device=device)
        for name, windows in evaluation.items()
    }


def plot_behavior(stages: list[dict[str, Any]], *, output_path: Path) -> None:
    categories = list(stages[0]["evaluation"])
    x = list(range(len(stages)))
    fig, axis = plt.subplots(figsize=(11.5, 5.3))
    for category in categories:
        axis.plot(
            x,
            [stage["evaluation"][category]["loss"] for stage in stages],
            marker="o",
            label=category.replace("_", " "),
        )
    axis.set_xticks(x, [stage["label"] for stage in stages])
    axis.set_yscale("log")
    axis.set_ylabel("language-model loss (log scale)")
    axis.set_title("Behavior across continual-learning stages")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_geometry(stages: list[dict[str, Any]], *, output_path: Path) -> None:
    updated = stages[1:]
    layers = list(updated[0]["geometry"])
    x = torch.arange(len(layers), dtype=torch.float32).numpy()
    width = 0.8 / len(updated)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for index, stage in enumerate(updated):
        offset = (index - (len(updated) - 1) / 2.0) * width
        axes[0].bar(
            x + offset,
            [stage["geometry"][layer]["cka"] for layer in layers],
            width=width,
            label=stage["label"],
        )
        axes[1].bar(
            x + offset,
            [stage["geometry"][layer]["relative_drift"] for layer in layers],
            width=width,
        )
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("linear CKA")
    axes[0].set_title("Residual geometry similarity")
    axes[0].legend()
    axes[1].set_ylabel("relative drift")
    axes[1].set_title("Residual geometry displacement")
    for axis in axes:
        axis.set_xticks(x, layers, rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_survival(stage_reports: list[dict[str, Any]], *, output_path: Path) -> None:
    survival = torch.tensor(
        [stage["consequence"]["survival"] for stage in stage_reports],
        dtype=torch.float32,
    )
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    image = axis.imshow(survival.T, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    axis.set_xticks(
        range(len(stage_reports)),
        [f"u{stage['update']} / s{stage['stage']}" for stage in stage_reports],
    )
    axis.set_yticks(range(survival.shape[1]))
    axis.set_xlabel("continual-learning update")
    axis.set_ylabel("trace slot")
    axis.set_title("Consequence-guided trace survival")
    fig.colorbar(image, ax=axis, label="survival mass")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_update_diagnostics(
    stage_history: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    x = list(range(1, len(updates) + 1))
    margins = [
        stage["correction_preference"]["new_minus_old_margin"]
        for stage in stage_history[1:]
    ]
    stable = [update["stable_after"] for update in updates]
    guard = [update["stable_guard_limit"] for update in updates]
    raw_damage = [update["training"][-1]["raw_damage"] for update in updates]
    safe_damage = [update["training"][-1]["safe_damage"] for update in updates]
    accepted = [update["accepted"] for update in updates]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
    axes[0].plot(x, margins, marker="o", color="#2563eb")
    axes[0].axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[0].set_title("Corrected minus obsolete margin")
    axes[0].set_ylabel("mean logit margin")
    axes[1].plot(x, stable, marker="o", label="stable loss")
    axes[1].plot(x, guard, linestyle="--", label="guard limit")
    axes[1].set_title("Transactional stable guard")
    axes[1].set_ylabel("language-model loss")
    axes[1].legend()
    axes[2].plot(x, raw_damage, marker="o", label="raw")
    axes[2].plot(x, safe_damage, marker="o", label="safe")
    for update, was_accepted in zip(x, accepted, strict=True):
        if not was_accepted:
            axes[2].axvline(update, color="#dc2626", alpha=0.35)
    axes[2].set_title("Dependency damage")
    axes[2].set_ylabel("normalized damage")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("micro-update")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    args.num_slots = args.trace_slots
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = instantiate_model(checkpoint, device)
    parameter_count = sum(parameter.numel() for parameter in trainable_weight_parameters(model))
    if not args.min_trainable_parameters <= parameter_count <= args.max_trainable_parameters:
        raise ValueError(
            f"Checkpoint has {parameter_count} trainable weights; expected "
            f"[{args.min_trainable_parameters}, {args.max_trainable_parameters}]."
        )
    vocab_size = int(checkpoint["model_config"]["vocab_size"])
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            f"Tokenizer/model vocabulary mismatch: {tokenizer.get_vocab_size()} vs {vocab_size}."
        )
    base_candidates, continual_stages, evaluation, correction_queries = build_staged_data(
        args,
        tokenizer,
        vocab_size,
    )
    micro_stages: list[tuple[int, TextWindows, list[FactQuery]]] = []
    correction_semantic_stage = 1 + len(continual_stages)
    for semantic_stage, stage in enumerate(continual_stages, start=2):
        shuffled_stage = permute_windows(
            stage,
            seed=args.seed + 1009 * semantic_stage,
        )
        stage_batches = split_windows(
            shuffled_stage,
            batch_size=args.micro_batch_windows,
        )
        assigned_queries: list[list[FactQuery]] = [list() for _ in stage_batches]
        if semantic_stage == correction_semantic_stage:
            for index, query in enumerate(correction_queries):
                assigned_queries[index % len(stage_batches)].append(query)
        for micro_batch, active_queries in zip(
            stage_batches,
            assigned_queries,
            strict=True,
        ):
            micro_stages.append((semantic_stage, micro_batch, active_queries))
    bank = initialize_trace_bank(model, base_candidates, args=args, device=device)
    initial_bank_groups = list(bank.groups)
    base_geometry = collect_geometry(model, evaluation["stable_book"], device=device)
    stage_history: list[dict[str, Any]] = [
        {
            "label": "base",
            "stage": 1,
            "evaluation": evaluate_all(model, evaluation, device=device),
            "geometry": {
                layer: {"cka": 1.0, "relative_drift": 0.0}
                for layer in base_geometry
            },
            "correction_preference": evaluate_correction_queries(
                model,
                correction_queries,
                device=device,
            ),
        }
    ]
    update_reports: list[dict[str, Any]] = []

    print("1M CONSEQUENCE-GUIDED CONTINUAL LEARNING")
    print("=" * 152)
    print(
        f"device={device} trainable_parameters={parameter_count} vocab={vocab_size} "
        f"layers={len(model.blocks)} width={model_width(model)} "
        f"slots={args.trace_slots} "
        f"stored_tokens={bank.inputs.numel()} stage_windows="
        f"{[stage.inputs.shape[0] for stage in continual_stages]} "
        f"micro_updates={len(micro_stages)} cl_epochs={args.cl_epochs}"
    )
    for update_number, (semantic_stage, current, active_queries) in enumerate(
        micro_stages,
        start=1,
    ):
        started = time.perf_counter()
        survival, consequence = consequence_survival(
            model,
            bank,
            current,
            active_queries,
            args=args,
            device=device,
        )
        surviving_bank = bank.clone()
        surviving_bank.masses = surviving_bank.masses * survival.to(surviving_bank.masses)
        control = propose_trace_update(
            model,
            surviving_bank,
            current,
            stage=update_number + 1,
            args=args,
            device=device,
        )
        control.protection_weights = control.protection_weights * survival.to(
            control.protection_weights
        )
        control.report["survival"] = consequence
        parameters = trainable_weight_parameters(model)
        parameter_snapshot = snapshot_parameters(parameters)
        candidate_before = candidate_measurement(
            model,
            current,
            control.write_weights,
            active_queries,
            args=args,
            device=device,
        )
        stable_before = evaluate_windows(
            model,
            evaluation["stable_book"],
            device=device,
        )["loss"]
        training = train_microstage(
            model=model,
            bank=surviving_bank,
            current=current,
            control=control,
            correction_queries=active_queries,
            args=args,
            device=device,
        )
        candidate_after = candidate_measurement(
            model,
            current,
            control.write_weights,
            active_queries,
            args=args,
            device=device,
        )
        stable_after = evaluate_windows(
            model,
            evaluation["stable_book"],
            device=device,
        )["loss"]
        candidate_geometry = geometry_report(
            base_geometry,
            collect_geometry(model, evaluation["stable_book"], device=device),
        )
        candidate_min_cka = min(
            layer["cka"] for layer in candidate_geometry.values()
        )
        relative_gain = (
            candidate_before["objective"] - candidate_after["objective"]
        ) / max(abs(candidate_before["objective"]), torch.finfo(torch.float32).eps)
        guard_limit = (
            args.guard_loss_ratio * stable_before + args.guard_loss_absolute
        )
        guard_passed = stable_after <= guard_limit
        geometry_passed = candidate_min_cka >= args.guard_min_cka
        correction_passed = (
            not active_queries
            or candidate_after["margin"] > candidate_before["margin"]
        )
        accepted = (
            relative_gain >= args.commit_min_relative_gain
            and guard_passed
            and geometry_passed
            and correction_passed
        )
        if accepted:
            bank = commit_trace_bank(model, control.pending, device=device)
        else:
            restore_parameters(parameters, parameter_snapshot)
            stable_after = evaluate_windows(
                model,
                evaluation["stable_book"],
                device=device,
            )["loss"]
        evaluation_report = evaluate_all(model, evaluation, device=device)
        geometry = geometry_report(
            base_geometry,
            collect_geometry(model, evaluation["stable_book"], device=device),
        )
        correction = evaluate_correction_queries(
            model,
            correction_queries,
            device=device,
        )
        elapsed = time.perf_counter() - started
        stage_history.append(
            {
                "label": f"update {update_number}",
                "stage": semantic_stage,
                "update": update_number,
                "accepted": accepted,
                "evaluation": evaluation_report,
                "geometry": geometry,
                "correction_preference": correction,
            }
        )
        update_reports.append(
            {
                "stage": semantic_stage,
                "update": update_number,
                "accepted": accepted,
                "seconds": elapsed,
                "consequence": consequence,
                "trace_control": control.report,
                "training": training,
                "active_correction_queries": len(active_queries),
                "candidate_before": candidate_before,
                "candidate_after": candidate_after,
                "relative_gain": relative_gain,
                "stable_before": stable_before,
                "stable_after": stable_after,
                "stable_guard_limit": guard_limit,
                "guard_passed": guard_passed,
                "candidate_min_cka": candidate_min_cka,
                "geometry_passed": geometry_passed,
                "correction_passed": correction_passed,
                "bank_groups": list(bank.groups),
                "stored_tokens": int(bank.inputs.numel()),
            }
        )
        final_training = training[-1]
        final_layer = geometry["final"]
        print(
            f"update={update_number:02d} semantic_stage={semantic_stage} "
            f"accepted={int(accepted)} seconds={elapsed:.1f} "
            f"survival={min(consequence['survival']):.3f}/{max(consequence['survival']):.3f} "
            f"candidate={candidate_before['objective']:.4f}->{candidate_after['objective']:.4f} "
            f"correction_margin={candidate_before['margin']:.3f}->{candidate_after['margin']:.3f} "
            f"damage={final_training['raw_damage']:.4f}->{final_training['safe_damage']:.4f} "
            f"final_cka={final_layer['cka']:.4f} drift={final_layer['relative_drift']:.4f}"
        )

    final = stage_history[-1]
    print("\nFINAL 1M CONTINUAL-LEARNING STATE")
    print("-" * 152)
    print(
        f"{'category':>18} {'base_loss':>12} {'final_loss':>12} "
        f"{'base_acc':>12} {'final_acc':>12}"
    )
    for name in final["evaluation"]:
        before = stage_history[0]["evaluation"][name]
        after = final["evaluation"][name]
        print(
            f"{name:>18} {before['loss']:12.5f} {after['loss']:12.5f} "
            f"{before['token_accuracy']:12.4f} {after['token_accuracy']:12.4f}"
        )
    print(
        "correction new>old={:.4f}->{:.4f} margin={:.4f}->{:.4f} accepted={}/{}".format(
            stage_history[0]["correction_preference"]["new_over_old_fraction"],
            final["correction_preference"]["new_over_old_fraction"],
            stage_history[0]["correction_preference"]["new_minus_old_margin"],
            final["correction_preference"]["new_minus_old_margin"],
            sum(update["accepted"] for update in update_reports),
            len(update_reports),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.output_dir / "consequence_1m_behavior.png"
    geometry_path = args.output_dir / "consequence_1m_geometry.png"
    survival_path = args.output_dir / "consequence_1m_survival.png"
    diagnostics_path = args.output_dir / "consequence_1m_update_diagnostics.png"
    output_json = args.output_dir / "consequence_1m_cl.json"
    plot_behavior(stage_history, output_path=behavior_path)
    plot_geometry(stage_history, output_path=geometry_path)
    plot_survival(update_reports, output_path=survival_path)
    plot_update_diagnostics(
        stage_history,
        update_reports,
        output_path=diagnostics_path,
    )
    output = {
        "question": (
            "Can bounded consequence-guided trace survival and Invariant-Tangent updates perform "
            "transactional micro-stage continual learning in an approximately 1M-weight transformer?"
        ),
        "scope": (
            "Single-seed scale-transition experiment on controlled real text, novel facts, token-level "
            "corrections, and random noise. This is not an open-world or production-scale claim."
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
            "max_seq_len": model.max_seq_len,
        },
        "initial_bank_groups": initial_bank_groups,
        "initial_stored_tokens": int(args.trace_slots * args.seq_len),
        "stage_history": stage_history,
        "updates": update_reports,
        "final_bank_groups": list(bank.groups),
        "final_stored_tokens": int(bank.inputs.numel()),
        "plots": {
            "behavior": str(behavior_path),
            "geometry": str(geometry_path),
            "survival": str(survival_path),
            "update_diagnostics": str(diagnostics_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(
        f"wrote_plots={behavior_path},{geometry_path},{survival_path},{diagnostics_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_text_parser()
    parser.description = __doc__
    parser.set_defaults(
        checkpoint=Path(
            "model/checkpoints/gco-storage-capacity-1m/one_m-mixed-5000w-seed0.pt"
        ),
        output_dir=Path("model/analysis/gco-1m-microcontinual-cl-seed0"),
        seq_len=64,
        stride=32,
        trace_slots=16,
        trace_steps=80,
        restarts=2,
        cl_epochs=80,
        cl_lr=2e-3,
        dependency_refresh=20,
        dependency_rank=32,
        geometry_pairs=24,
        minimum_trace_mass=1e-4,
    )
    parser.add_argument("--min-trainable-parameters", type=int, default=950_000)
    parser.add_argument("--max-trainable-parameters", type=int, default=1_200_000)
    parser.add_argument("--survival-budget", type=float, default=14.0)
    parser.add_argument("--survival-temperature", type=float, default=0.35)
    parser.add_argument("--survival-bisection-steps", type=int, default=64)
    parser.add_argument("--micro-batch-windows", type=int, default=16)
    parser.add_argument("--correction-query-weight", type=float, default=1.0)
    parser.add_argument("--correction-suppression-weight", type=float, default=1.0)
    parser.add_argument("--correction-margin", type=float, default=1.0)
    parser.add_argument("--commit-min-relative-gain", type=float, default=1e-4)
    parser.add_argument("--guard-loss-ratio", type=float, default=1.05)
    parser.add_argument("--guard-loss-absolute", type=float, default=0.005)
    parser.add_argument("--guard-min-cka", type=float, default=0.95)
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
