"""Integrate bounded autonomous traces with Invariant-Tangent neural updates.

A tiny nonlinear function model learns a staged representation stream. The
bounded trace field receives input-plus-target evidence and stores only five
mean/mass/total-variance summaries. Its reconstruction confidence produces:

* familiarity and recurrence probabilities for incoming supervised examples;
* protection weights for prior trace probes.

No preserve, drop, novel, obsolete, or noise labels enter optimization. Hidden
group names are used only for reporting. A recurring novel target deliberately
shares its input distribution with a rare old target, so capacity reallocation
must cause a real weight-level replacement. Retained trace probes define output
and hidden-distance Jacobian rows for the Invariant-Tangent update.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_mini_cl_world_demo import (
    flat_autograd_gradient,
    project_gradient_away_from_constraints,
)
from experiments.gco_math.gco_tiny_recurrent_trace_field import (
    EvidenceMoments,
    TraceSummary,
    compress_to_trace_moments,
    concatenate_moments,
    encode_trace_summary,
    fit_functional_trace_field,
    functional_attention,
    prepare_stream,
)


@dataclass(frozen=True)
class FunctionExample:
    x: tuple[float, ...]
    target: tuple[float, ...]
    hidden_group: str
    stage: int


@dataclass
class TraceStageControl:
    stage: int
    current_write_weights: torch.Tensor
    prior_probe_inputs: torch.Tensor
    prior_probe_targets: torch.Tensor
    prior_probe_weights: torch.Tensor
    trace_centers: torch.Tensor
    summary: TraceSummary
    report: dict[str, Any]


class TinyFunctionModel(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim, bias=True)
        self.output = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or values.shape[1] != self.input.in_features:
            raise ValueError(
                f"Model input must be [N, {self.input.in_features}], got {tuple(values.shape)}."
            )
        hidden = torch.tanh(self.input(values))
        return self.output(hidden), hidden


def validate_args(args: argparse.Namespace) -> None:
    if args.d_model < 6:
        raise ValueError(f"--d-model must be >= 6, got {args.d_model}.")
    if args.output_dim < 6:
        raise ValueError(f"--output-dim must be >= 6, got {args.output_dim}.")
    if args.hidden_dim <= 0:
        raise ValueError(f"--hidden-dim must be positive, got {args.hidden_dim}.")
    if not 2 <= args.num_slots <= 8:
        raise ValueError(f"--num-slots must be in [2, 8], got {args.num_slots}.")
    for name in ("base_epochs", "cl_epochs", "joint_epochs", "trace_steps", "restarts"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "base_lr",
        "cl_lr",
        "joint_lr",
        "trace_lr",
        "attention_scale",
        "observation_sigma",
        "encoding_span",
        "concept_radius",
        "trace_target_scale",
        "protection_power",
        "projection_damping",
        "restore_strength",
        "restore_bound_fraction",
        "loss_mix_strength",
        "geometry_mix_strength",
        "grad_clip",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite, got {value}.")
    if args.ambiguity_weight < 0.0:
        raise ValueError("--ambiguity-weight must be non-negative.")


def target_basis(output_dim: int) -> dict[str, torch.Tensor]:
    basis = torch.eye(output_dim, dtype=torch.float32)
    return {
        "merge_a": basis[0],
        "merge_b": basis[0],
        "stable": basis[1],
        "obsolete": basis[2],
        "branch_up": basis[3],
        "branch_down": basis[4],
        "branch_root": 0.5 * (basis[3] + basis[4]),
        "novel": basis[5],
    }


def build_function_stream(args: argparse.Namespace) -> list[list[FunctionExample]]:
    raw_stages = prepare_stream(args)
    targets = target_basis(args.output_dim)
    obsolete_values = [
        torch.tensor(point.vector, dtype=torch.float32)
        for point in raw_stages[0]
        if point.hidden_group == "obsolete"
    ]
    novel_values = [
        torch.tensor(point.vector, dtype=torch.float32)
        for point in raw_stages[2]
        if point.hidden_group == "novel"
    ]
    if not obsolete_values or not novel_values:
        raise RuntimeError("Integration stream requires both obsolete and novel evidence.")
    obsolete_mean = torch.stack(obsolete_values).mean(dim=0)
    novel_mean = torch.stack(novel_values).mean(dim=0)
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(args.seed + 9001)

    stages: list[list[FunctionExample]] = []
    for stage_index, raw_stage in enumerate(raw_stages, start=1):
        stage: list[FunctionExample] = []
        for point in raw_stage:
            value = torch.tensor(point.vector, dtype=torch.float32)
            if point.hidden_group == "novel":
                value = obsolete_mean + (value - novel_mean)
            if point.hidden_group == "noise":
                target = torch.randn(args.output_dim, generator=noise_generator)
                target = target / target.norm()
            else:
                target = targets[point.hidden_group]
            stage.append(
                FunctionExample(
                    x=tuple(float(component) for component in value),
                    target=tuple(float(component) for component in target),
                    hidden_group=point.hidden_group,
                    stage=stage_index,
                )
            )
        stages.append(stage)
    return stages


def examples_to_tensors(
    examples: list[FunctionExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot tensorize zero function examples.")
    return (
        torch.tensor([example.x for example in examples], dtype=torch.float32, device=device),
        torch.tensor([example.target for example in examples], dtype=torch.float32, device=device),
    )


def trace_evidence_moments(
    examples: list[FunctionExample],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> EvidenceMoments:
    inputs, targets = examples_to_tensors(examples, device=device)
    means = torch.cat([inputs, args.trace_target_scale * targets], dim=1)
    dimension = means.shape[1]
    return EvidenceMoments(
        means=means,
        covariances=torch.zeros(
            means.shape[0],
            dimension,
            dimension,
            dtype=means.dtype,
            device=device,
        ),
        weights=torch.ones(means.shape[0], dtype=means.dtype, device=device),
    )


def reconstruction_confidence(
    means: torch.Tensor,
    centers: torch.Tensor,
    *,
    attention_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention = functional_attention(means, centers, attention_scale=attention_scale)
    reconstruction = attention @ centers
    squared_error = ((means - reconstruction) ** 2).sum(dim=1)
    confidence = torch.exp(-squared_error / (2.0 * attention_scale**2))
    if not torch.isfinite(confidence).all():
        raise FloatingPointError("Trace reconstruction confidence contains non-finite values.")
    return confidence, squared_error


def recurrence_probability(
    means: torch.Tensor,
    *,
    attention_scale: float,
) -> torch.Tensor:
    """Measure whether each item has coherent support in the current evidence."""
    if means.ndim != 2 or means.shape[0] < 2:
        raise ValueError("Recurrence probability requires at least two evidence items.")
    squared_distance = ((means.unsqueeze(1) - means.unsqueeze(0)) ** 2).sum(dim=2)
    similarity = torch.exp(-squared_distance / (2.0 * attention_scale**2))
    similarity.fill_diagonal_(0.0)
    recurrence = similarity.max(dim=1).values
    if not torch.isfinite(recurrence).all():
        raise FloatingPointError("Trace recurrence probability contains non-finite values.")
    return recurrence


def report_trace_labels(targets: torch.Tensor, *, output_dim: int) -> list[str]:
    """Attach semantic names to trace probes for plots; labels never enter optimization."""
    basis = target_basis(output_dim)
    names = list(basis)
    references = torch.stack([basis[name] for name in names]).to(targets)
    squared_distance = ((targets.unsqueeze(1) - references.unsqueeze(0)) ** 2).sum(dim=2)
    return [names[index] for index in squared_distance.argmin(dim=1).detach().cpu().tolist()]


def build_trace_controls(
    stages: list[list[FunctionExample]],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> list[TraceStageControl]:
    summary: TraceSummary | None = None
    previous_centers: torch.Tensor | None = None
    controls: list[TraceStageControl] = []
    for stage_index, stage in enumerate(stages, start=1):
        current = trace_evidence_moments(stage, args=args, device=device)
        prior_moments = summary.to_moments() if summary is not None else None
        training_evidence = (
            current
            if prior_moments is None
            else concatenate_moments([prior_moments, current])
        )
        solution = fit_functional_trace_field(
            training_evidence,
            args=args,
            stage=stage_index,
            previous_centers=previous_centers,
        )
        centers = solution.centers
        current_recurrence = recurrence_probability(
            current.means,
            attention_scale=args.attention_scale,
        )
        if prior_moments is None:
            prior_inputs = torch.empty(0, args.d_model, device=device)
            prior_targets = torch.empty(0, args.output_dim, device=device)
            prior_confidence = torch.empty(0, device=device)
            prior_error = torch.empty(0, device=device)
            current_familiarity = torch.zeros_like(current_recurrence)
        else:
            if previous_centers is None:
                raise RuntimeError("Previous trace centers are missing for a non-initial stage.")
            current_familiarity, _current_prior_error = reconstruction_confidence(
                current.means,
                previous_centers,
                attention_scale=args.attention_scale,
            )
            prior_confidence, prior_error = reconstruction_confidence(
                prior_moments.means,
                centers,
                attention_scale=args.attention_scale,
            )
            prior_inputs = prior_moments.means[:, : args.d_model]
            prior_targets = prior_moments.means[:, args.d_model :] / args.trace_target_scale

        current_write = 1.0 - (1.0 - current_familiarity) * (1.0 - current_recurrence)
        effective_prior_protection = prior_confidence.pow(args.protection_power)
        if not torch.isfinite(current_write).all():
            raise FloatingPointError("Trace write probability contains non-finite values.")

        compressed = compress_to_trace_moments(training_evidence, solution)
        summary = encode_trace_summary(
            compressed,
            mode="trace",
            rank=1,
            power_iterations=1,
            seed=args.seed + stage_index,
        )
        previous_centers = centers.detach().clone()

        current_group_report: dict[str, dict[str, float]] = {}
        for group in sorted({example.hidden_group for example in stage}):
            indices = [index for index, example in enumerate(stage) if example.hidden_group == group]
            selected = torch.tensor(indices, device=device, dtype=torch.long)
            current_group_report[group] = {
                "write_weight": float(current_write[selected].mean().detach().cpu()),
                "familiarity": float(current_familiarity[selected].mean().detach().cpu()),
                "recurrence": float(current_recurrence[selected].mean().detach().cpu()),
            }
        controls.append(
            TraceStageControl(
                stage=stage_index,
                current_write_weights=current_write.detach(),
                prior_probe_inputs=prior_inputs.detach(),
                prior_probe_targets=prior_targets.detach(),
                prior_probe_weights=effective_prior_protection.detach(),
                trace_centers=centers.detach(),
                summary=summary,
                report={
                    "current_groups": current_group_report,
                    "prior_probe_confidence": [
                        float(value) for value in prior_confidence.detach().cpu()
                    ],
                    "prior_probe_weights": [
                        float(value) for value in effective_prior_protection.detach().cpu()
                    ],
                    "prior_probe_labels": report_trace_labels(
                        prior_targets,
                        output_dim=args.output_dim,
                    ),
                    "prior_probe_errors": [float(value) for value in prior_error.detach().cpu()],
                    "summary_scalars": summary.stored_scalars(),
                },
            )
        )
    return controls


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")
    if weights.shape != (prediction.shape[0],):
        raise ValueError("Example weights do not match prediction batch size.")
    denominator = weights.sum()
    if denominator <= 0.0:
        raise RuntimeError("Weighted MSE received zero total weight.")
    per_example = (prediction - target).square().mean(dim=1)
    return (per_example * weights).sum() / denominator


def apply_flat_update(
    parameters: list[nn.Parameter],
    gradient: torch.Tensor,
    *,
    learning_rate: float,
    grad_clip: float,
) -> float:
    norm = torch.linalg.vector_norm(gradient)
    scale = torch.clamp(gradient.new_tensor(grad_clip) / norm.clamp_min(1e-12), max=1.0)
    update = gradient * scale
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(update[offset : offset + count].reshape_as(parameter), alpha=-learning_rate)
            offset += count
    if offset != update.numel():
        raise RuntimeError(f"Flat update used {offset} entries from {update.numel()}.")
    return float(norm.detach().cpu())


def train_base(
    model: TinyFunctionModel,
    examples: list[FunctionExample],
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    inputs, targets = examples_to_tensors(examples, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trace: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction, _hidden = model(inputs)
        loss = (prediction - targets).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Base training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach().cpu()))
    return trace


def hidden_pair_distances(hidden: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
    result: dict[tuple[int, int], torch.Tensor] = {}
    for left in range(hidden.shape[0]):
        for right in range(left + 1, hidden.shape[0]):
            result[(left, right)] = (hidden[left] - hidden[right]).square().sum()
    return result


def bounded_restore(
    restore_gradient: torch.Tensor,
    tangent_gradient: torch.Tensor,
    *,
    strength: float,
    bound_fraction: float,
) -> torch.Tensor:
    restore_norm = torch.linalg.vector_norm(restore_gradient)
    tangent_norm = torch.linalg.vector_norm(tangent_gradient)
    if restore_norm <= 1e-12 or tangent_norm <= 1e-12:
        return torch.zeros_like(restore_gradient)
    scale = torch.clamp(
        tangent_norm * bound_fraction / (strength * restore_norm),
        max=1.0,
    )
    return strength * scale * restore_gradient


def train_sequential_method(
    *,
    method: str,
    base_model: TinyFunctionModel,
    stages: list[list[FunctionExample]],
    controls: list[TraceStageControl],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TinyFunctionModel, list[dict[str, Any]], dict[str, torch.Tensor]]:
    if method not in {"naive", "trace_loss_mix", "trace_invariant"}:
        raise ValueError(f"Unknown sequential method {method!r}.")
    model = copy.deepcopy(base_model)
    parameters = list(model.parameters())
    stage_reports: list[dict[str, Any]] = []
    stage2_snapshot: dict[str, torch.Tensor] = {}

    for stage_index in (1, 2):
        stage_number = stage_index + 1
        examples = stages[stage_index]
        control = controls[stage_index]
        inputs, targets = examples_to_tensors(examples, device=device)
        write_weights = (
            torch.ones(len(examples), device=device)
            if method == "naive"
            else control.current_write_weights.to(device)
        )
        prior_inputs = control.prior_probe_inputs.to(device)
        prior_weights = control.prior_probe_weights.to(device)
        if prior_inputs.shape[0] <= 0:
            raise RuntimeError(f"Stage {stage_number} has no prior trace probes.")
        with torch.no_grad():
            teacher_outputs, teacher_hidden = model(prior_inputs)
            reference_distances = {
                pair: value.detach()
                for pair, value in hidden_pair_distances(teacher_hidden).items()
            }

        epoch_trace: list[dict[str, float]] = []
        for epoch in range(1, args.cl_epochs + 1):
            prediction, _new_hidden = model(inputs)
            new_loss = weighted_mse(prediction, targets, write_weights)
            current_old_outputs, current_old_hidden = model(prior_inputs)
            protection_loss = weighted_mse(current_old_outputs, teacher_outputs, prior_weights)
            current_distances = hidden_pair_distances(current_old_hidden)
            geometry_terms: list[torch.Tensor] = []
            for (left, right), current_distance in current_distances.items():
                pair_weight = torch.sqrt(prior_weights[left] * prior_weights[right])
                geometry_terms.append(
                    pair_weight * (current_distance - reference_distances[(left, right)]).square()
                )
            if not geometry_terms:
                raise RuntimeError("No hidden geometry terms were created.")
            geometry_loss = torch.stack(geometry_terms).mean()

            if method == "naive":
                gradient = flat_autograd_gradient(
                    new_loss,
                    parameters,
                    retain_graph=False,
                    require_nonzero=True,
                    label=f"{method}:stage{stage_number}:new",
                )
                projection_stats = {"projection_removed_fraction": 0.0, "constraint_count": 0.0}
                restore_norm = 0.0
            elif method == "trace_loss_mix":
                combined = (
                    new_loss
                    + args.loss_mix_strength * protection_loss
                    + args.geometry_mix_strength * geometry_loss
                )
                gradient = flat_autograd_gradient(
                    combined,
                    parameters,
                    retain_graph=False,
                    require_nonzero=True,
                    label=f"{method}:stage{stage_number}:combined",
                )
                projection_stats = {"projection_removed_fraction": 0.0, "constraint_count": 0.0}
                restore_norm = 0.0
            else:
                raw_gradient = flat_autograd_gradient(
                    new_loss,
                    parameters,
                    retain_graph=True,
                    require_nonzero=True,
                    label=f"{method}:stage{stage_number}:new",
                )
                constraint_rows: list[torch.Tensor] = []
                for probe in range(current_old_outputs.shape[0]):
                    row_weight = torch.sqrt(prior_weights[probe])
                    for output_index in range(current_old_outputs.shape[1]):
                        constraint_rows.append(
                            flat_autograd_gradient(
                                row_weight * current_old_outputs[probe, output_index],
                                parameters,
                                retain_graph=True,
                                require_nonzero=False,
                                label=f"output_probe_{probe}_{output_index}",
                            )
                        )
                for (left, right), current_distance in current_distances.items():
                    pair_weight = torch.sqrt(prior_weights[left] * prior_weights[right])
                    constraint_rows.append(
                        flat_autograd_gradient(
                            pair_weight * current_distance,
                            parameters,
                            retain_graph=True,
                            require_nonzero=False,
                            label=f"geometry_pair_{left}_{right}",
                        )
                    )
                tangent, projection_stats = project_gradient_away_from_constraints(
                    raw_gradient=raw_gradient,
                    constraint_gradients=constraint_rows,
                    damping=args.projection_damping,
                    solver="gram",
                    rank_tolerance=1e-4,
                    plasticity_audit=False,
                )
                restore_loss = (
                    args.loss_mix_strength * protection_loss
                    + args.geometry_mix_strength * geometry_loss
                )
                restore_gradient = flat_autograd_gradient(
                    restore_loss,
                    parameters,
                    retain_graph=False,
                    require_nonzero=False,
                    label=f"{method}:stage{stage_number}:restore",
                )
                restore_update = bounded_restore(
                    restore_gradient,
                    tangent,
                    strength=args.restore_strength,
                    bound_fraction=args.restore_bound_fraction,
                )
                gradient = tangent + restore_update
                restore_norm = float(torch.linalg.vector_norm(restore_update).detach().cpu())

            gradient_norm = apply_flat_update(
                parameters,
                gradient,
                learning_rate=args.cl_lr,
                grad_clip=args.grad_clip,
            )
            if epoch in {1, args.cl_epochs} or epoch % args.print_every == 0:
                epoch_trace.append(
                    {
                        "epoch": float(epoch),
                        "new_loss": float(new_loss.detach().cpu()),
                        "protection_loss": float(protection_loss.detach().cpu()),
                        "geometry_loss": float(geometry_loss.detach().cpu()),
                        "gradient_norm": gradient_norm,
                        "projection_removed_fraction": projection_stats[
                            "projection_removed_fraction"
                        ],
                        "constraint_count": projection_stats["constraint_count"],
                        "restore_norm": restore_norm,
                    }
                )
        stage_reports.append(
            {
                "stage": stage_number,
                "trace": epoch_trace,
                "write_weight_mean": float(write_weights.mean().detach().cpu()),
                "protect_weight_mean": float(prior_weights.mean().detach().cpu()),
            }
        )
        if stage_number == 2:
            stage2_snapshot = {
                "parameters": torch.cat([parameter.detach().reshape(-1) for parameter in parameters]),
            }
    if not stage2_snapshot:
        raise RuntimeError("Sequential method did not capture a stage-2 snapshot.")
    return model, stage_reports, stage2_snapshot


def centered_kernel_alignment(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must contain the same number of rows.")
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    numerator = (left.transpose(0, 1) @ right).square().sum()
    denominator = torch.sqrt(
        (left.transpose(0, 1) @ left).square().sum()
        * (right.transpose(0, 1) @ right).square().sum()
    )
    if denominator <= 0.0:
        raise FloatingPointError("Cannot compute CKA with zero centered norm.")
    return float((numerator / denominator).detach().cpu())


def relative_representation_drift(
    reference: torch.Tensor,
    current: torch.Tensor,
) -> tuple[float, float]:
    if reference.shape != current.shape or reference.ndim != 2:
        raise ValueError("Representation drift expects equal [N, D] tensors.")
    hidden_denominator = torch.linalg.vector_norm(reference)
    reference_distances = torch.linalg.vector_norm(
        reference.unsqueeze(1) - reference.unsqueeze(0),
        dim=2,
    )
    current_distances = torch.linalg.vector_norm(
        current.unsqueeze(1) - current.unsqueeze(0),
        dim=2,
    )
    geometry_denominator = torch.linalg.vector_norm(reference_distances)
    if hidden_denominator <= 0.0 or geometry_denominator <= 0.0:
        raise FloatingPointError("Representation drift denominator is zero.")
    hidden_drift = torch.linalg.vector_norm(current - reference) / hidden_denominator
    geometry_drift = (
        torch.linalg.vector_norm(current_distances - reference_distances) / geometry_denominator
    )
    return float(hidden_drift.detach().cpu()), float(geometry_drift.detach().cpu())


def evaluate_model(
    model: TinyFunctionModel,
    examples: list[FunctionExample],
    *,
    device: torch.device,
) -> dict[str, Any]:
    inputs, targets = examples_to_tensors(examples, device=device)
    with torch.no_grad():
        prediction, hidden = model(inputs)
        errors = (prediction - targets).square().mean(dim=1)
    groups: dict[str, dict[str, float]] = {}
    for group in sorted({example.hidden_group for example in examples}):
        indices = torch.tensor(
            [index for index, example in enumerate(examples) if example.hidden_group == group],
            device=device,
            dtype=torch.long,
        )
        groups[group] = {
            "mse": float(errors[indices].mean().detach().cpu()),
            "count": float(indices.numel()),
        }
    return {
        "mse": float(errors.mean().detach().cpu()),
        "groups": groups,
        "outputs": prediction.detach(),
        "hidden": hidden.detach(),
    }


def plot_behavior(results: dict[str, dict[str, Any]], *, output_path: Path) -> None:
    methods = ["naive", "trace_loss_mix", "trace_invariant", "joint_reference"]
    groups = sorted(results[methods[0]]["evaluation"]["groups"])
    x = torch.arange(len(groups), dtype=torch.float32).numpy()
    width = 0.2
    fig, axis = plt.subplots(figsize=(13.0, 5.3))
    for index, method in enumerate(methods):
        values = [results[method]["evaluation"]["groups"][group]["mse"] for group in groups]
        axis.bar(x + (index - 1.5) * width, values, width=width, label=method.replace("_", " "))
    axis.set_xticks(x, groups, rotation=30, ha="right")
    axis.set_yscale("symlog", linthresh=1e-3)
    axis.set_ylabel("mean squared error")
    axis.set_title("Neural behavior after trace-controlled continual learning")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_geometry(results: dict[str, dict[str, Any]], *, output_path: Path) -> None:
    methods = ["naive", "trace_loss_mix", "trace_invariant"]
    x = torch.arange(len(methods), dtype=torch.float32).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        [results[method]["stage3_hidden_relative_drift"] for method in methods],
        width=width,
        label="hidden-state drift",
    )
    axes[0].bar(
        x + width / 2,
        [results[method]["stage3_pair_geometry_drift"] for method in methods],
        width=width,
        label="pair-distance drift",
    )
    axes[0].set_xticks(x, [method.replace("_", " ") for method in methods], rotation=20)
    axes[0].set_ylabel("relative drift (lower is better)")
    axes[0].set_title("Retained representation across final update")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(x, [results[method]["stage3_weight_drift"] for method in methods], color="#ff7f0e")
    axes[1].set_xticks(x, [method.replace("_", " ") for method in methods], rotation=20)
    axes[1].set_ylabel("relative parameter drift")
    axes[1].set_title("Final-stage weight movement")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trace_weights(controls: list[TraceStageControl], *, output_path: Path) -> None:
    stage3 = controls[2]
    groups = sorted(stage3.report["current_groups"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].bar(
        groups,
        [stage3.report["current_groups"][group]["write_weight"] for group in groups],
        color="#2ca02c",
    )
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("trace-derived write weight")
    axes[0].set_title("Incoming stage-3 evidence")
    axes[0].tick_params(axis="x", rotation=25)
    probe_positions = range(len(stage3.report["prior_probe_weights"]))
    axes[1].bar(
        probe_positions,
        stage3.report["prior_probe_weights"],
        color="#1f77b4",
    )
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xticks(
        list(probe_positions),
        stage3.report["prior_probe_labels"],
        rotation=25,
        ha="right",
    )
    axes[1].set_xlabel("nearest reporting label (not used by optimization)")
    axes[1].set_ylabel("trace-derived protection weight")
    axes[1].set_title("Prior evidence after reorganization")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    stages = build_function_stream(args)
    controls = build_trace_controls(stages, args=args, device=device)
    all_examples = [example for stage in stages for example in stage]

    print("TINY TRACE -> INVARIANT-TANGENT INTEGRATION")
    print("=" * 140)
    print(
        f"device={device} slots={args.num_slots} hidden={args.hidden_dim} "
        f"base_epochs={args.base_epochs} cl_epochs={args.cl_epochs}"
    )
    for control in controls:
        print(
            f"trace_stage={control.stage} summary_scalars={control.report['summary_scalars']} "
            f"write={control.report['current_groups']}"
        )

    base_model = TinyFunctionModel(
        input_dim=args.d_model,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    ).to(device)
    base_trace = train_base(
        base_model,
        stages[0],
        epochs=args.base_epochs,
        learning_rate=args.base_lr,
        device=device,
    )

    results: dict[str, dict[str, Any]] = {}
    retained_examples = [
        example
        for example in all_examples
        if example.hidden_group
        in {"merge_a", "merge_b", "stable", "branch_root", "branch_up", "branch_down"}
    ]
    retained_inputs, _retained_targets = examples_to_tensors(retained_examples, device=device)

    for method in ["naive", "trace_loss_mix", "trace_invariant"]:
        model, training_report, stage2_snapshot = train_sequential_method(
            method=method,
            base_model=base_model,
            stages=stages,
            controls=controls,
            args=args,
            device=device,
        )
        evaluation = evaluate_model(model, all_examples, device=device)
        with torch.no_grad():
            _outputs, final_hidden = model(retained_inputs)
        stage2_model = copy.deepcopy(model)
        offset = 0
        with torch.no_grad():
            for parameter in stage2_model.parameters():
                count = parameter.numel()
                parameter.copy_(stage2_snapshot["parameters"][offset : offset + count].reshape_as(parameter))
                offset += count
        if offset != stage2_snapshot["parameters"].numel():
            raise RuntimeError("Stage-2 snapshot had unused parameter entries.")
        with torch.no_grad():
            _stage2_outputs, stage2_hidden = stage2_model(retained_inputs)
        hidden_relative_drift, pair_geometry_drift = relative_representation_drift(
            stage2_hidden,
            final_hidden,
        )
        final_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        stage2_parameters = stage2_snapshot["parameters"].to(device)
        results[method] = {
            "evaluation": {
                "mse": evaluation["mse"],
                "groups": evaluation["groups"],
            },
            "training": training_report,
            "stage3_hidden_cka": centered_kernel_alignment(stage2_hidden, final_hidden),
            "stage3_hidden_relative_drift": hidden_relative_drift,
            "stage3_pair_geometry_drift": pair_geometry_drift,
            "stage3_weight_drift": float(
                (
                    torch.linalg.vector_norm(final_parameters - stage2_parameters)
                    / torch.linalg.vector_norm(stage2_parameters).clamp_min(1e-12)
                ).detach().cpu()
            ),
        }

    joint_examples = [
        example
        for example in all_examples
        if example.hidden_group not in {"obsolete", "noise"}
    ]
    joint_model = TinyFunctionModel(
        input_dim=args.d_model,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    ).to(device)
    joint_trace = train_base(
        joint_model,
        joint_examples,
        epochs=args.joint_epochs,
        learning_rate=args.joint_lr,
        device=device,
    )
    joint_evaluation = evaluate_model(joint_model, all_examples, device=device)
    results["joint_reference"] = {
        "evaluation": {
            "mse": joint_evaluation["mse"],
            "groups": joint_evaluation["groups"],
        }
    }

    print("\nFINAL NEURAL BEHAVIOR")
    print("-" * 140)
    print(
        f"{'method':>18} {'retained':>10} {'novel':>10} {'obsolete':>10} "
        f"{'noise':>10} {'hiddenCKA':>10} {'geomDrift':>10} {'weightDrift':>12}"
    )
    retained_groups = {"merge_a", "merge_b", "stable", "branch_root", "branch_up", "branch_down"}
    for method in ["naive", "trace_loss_mix", "trace_invariant", "joint_reference"]:
        groups = results[method]["evaluation"]["groups"]
        retained = sum(groups[group]["mse"] for group in retained_groups) / len(retained_groups)
        hidden_cka = results[method].get("stage3_hidden_cka")
        geometry_drift = results[method].get("stage3_pair_geometry_drift")
        drift = results[method].get("stage3_weight_drift")
        print(
            f"{method:>18} {retained:10.4g} {groups['novel']['mse']:10.4g} "
            f"{groups['obsolete']['mse']:10.4g} {groups['noise']['mse']:10.4g} "
            f"{('NA' if hidden_cka is None else f'{hidden_cka:.4f}'):>10} "
            f"{('NA' if geometry_drift is None else f'{geometry_drift:.4f}'):>10} "
            f"{('NA' if drift is None else f'{drift:.4f}'):>12}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.output_dir / "trace_invariant_behavior.png"
    geometry_path = args.output_dir / "trace_invariant_geometry.png"
    weights_path = args.output_dir / "trace_control_weights.png"
    output_json = args.output_dir / "trace_invariant_integration.json"
    plot_behavior(results, output_path=behavior_path)
    plot_geometry(results, output_path=geometry_path)
    plot_trace_weights(controls, output_path=weights_path)

    output = {
        "question": (
            "Can bounded autonomous traces drive real Invariant-Tangent model updates that retain old "
            "functions, learn recurring novelty, reject noise, and replace a superseded fact?"
        ),
        "scope": (
            "Tiny nonlinear regression model in synthetic representation space. Hidden group labels are "
            "reporting-only; trace optimization receives input and supervised target vectors."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stream": [[asdict(example) for example in stage] for stage in stages],
        "trace_controls": [control.report for control in controls],
        "base_training": {"initial": base_trace[0], "final": base_trace[-1]},
        "joint_training": {"initial": joint_trace[0], "final": joint_trace[-1]},
        "results": results,
        "plots": {
            "behavior": str(behavior_path),
            "geometry": str(geometry_path),
            "trace_weights": str(weights_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_plots={behavior_path},{geometry_path},{weights_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/analysis/gco-tiny-trace-invariant-integration-seed0"),
    )
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=6)
    parser.add_argument("--num-slots", type=int, default=5)
    parser.add_argument("--base-epochs", type=int, default=500)
    parser.add_argument("--cl-epochs", type=int, default=220)
    parser.add_argument("--joint-epochs", type=int, default=700)
    parser.add_argument("--base-lr", type=float, default=0.02)
    parser.add_argument("--cl-lr", type=float, default=0.03)
    parser.add_argument("--joint-lr", type=float, default=0.02)
    parser.add_argument("--trace-steps", type=int, default=700)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--trace-lr", type=float, default=0.03)
    parser.add_argument("--attention-scale", type=float, default=1.0)
    parser.add_argument("--ambiguity-weight", type=float, default=0.2)
    parser.add_argument("--observation-sigma", type=float, default=0.24)
    parser.add_argument("--encoding-span", type=float, default=5.0)
    parser.add_argument("--concept-radius", type=float, default=3.2)
    parser.add_argument("--trace-target-scale", type=float, default=3.2)
    parser.add_argument("--protection-power", type=float, default=8.0)
    parser.add_argument("--projection-damping", type=float, default=1e-3)
    parser.add_argument("--restore-strength", type=float, default=0.05)
    parser.add_argument("--restore-bound-fraction", type=float, default=0.5)
    parser.add_argument("--loss-mix-strength", type=float, default=1.0)
    parser.add_argument("--geometry-mix-strength", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--merge-points", type=int, default=24)
    parser.add_argument("--stable-points", type=int, default=30)
    parser.add_argument("--root-points", type=int, default=20)
    parser.add_argument("--obsolete-points", type=int, default=6)
    parser.add_argument("--branch-points", type=int, default=24)
    parser.add_argument("--novel-points", type=int, default=30)
    parser.add_argument("--noise-points", type=int, default=8)
    parser.add_argument("--stage3-old-points-per-group", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
