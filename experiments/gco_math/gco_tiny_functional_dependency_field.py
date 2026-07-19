"""Test a trace-conditioned functional dependency field for continual learning.

The bounded trace field decides which evidence remains supported. Protected
output and hidden-geometry measurements are differentiated with respect to the
current model, block-normalized, and compressed into a low-rank parameter
consequence basis. A separate SVD of measurement sensitivity with respect to
the hidden layer reports emergent feature-family directions.

Hidden group names are reporting-only. They do not enter trace optimization,
dependency construction, gradient projection, restore, or model training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
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
from experiments.gco_math.gco_tiny_trace_invariant_tangent_integration import (
    TinyFunctionModel,
    TraceStageControl,
    apply_flat_update,
    bounded_restore,
    build_function_stream,
    build_parser as build_integration_parser,
    build_trace_controls,
    centered_kernel_alignment,
    evaluate_model,
    examples_to_tensors,
    hidden_pair_distances,
    relative_representation_drift,
    train_base,
    train_sequential_method,
    validate_args as validate_integration_args,
    weighted_mse,
)


@dataclass
class DependencyBasis:
    constraint_rows: list[torch.Tensor]
    normalized_measurement_matrix: torch.Tensor
    parameter_singular_values: list[float]
    parameter_effective_rank: float
    parameter_retained_rank: int
    parameter_retained_energy: float
    feature_singular_values: list[float]
    feature_effective_rank: float
    feature_retained_rank: int
    feature_retained_energy: float


def validate_args(args: argparse.Namespace) -> None:
    validate_integration_args(args)
    for name in ("dependency_rank", "dependency_refresh"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "dependency_power",
        "behavior_block_weight",
        "geometry_block_weight",
        "feature_block_weight",
        "dependency_rank_tolerance",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if not 0.0 < args.dependency_energy <= 1.0:
        raise ValueError("--dependency-energy must be in (0, 1].")


def block_normalize(rows: list[torch.Tensor], *, block_weight: float, label: str) -> torch.Tensor:
    if not rows:
        raise RuntimeError(f"Cannot normalize empty dependency block {label!r}.")
    matrix = torch.stack(rows, dim=0)
    row_norm_sq = matrix.square().sum(dim=1)
    rms_norm = torch.sqrt(row_norm_sq.mean())
    if not torch.isfinite(rms_norm) or rms_norm <= 1e-12:
        raise FloatingPointError(f"Dependency block {label!r} has zero or non-finite RMS norm.")
    return block_weight * matrix / rms_norm


def effective_rank(singular_values: torch.Tensor) -> float:
    positive = singular_values[singular_values > 1e-12]
    if positive.numel() == 0:
        return 0.0
    probabilities = positive / positive.sum()
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).detach().cpu())


def retained_rank(
    singular_values: torch.Tensor,
    *,
    rank_budget: int,
    tolerance: float,
    energy_target: float,
) -> int:
    if singular_values.numel() == 0 or singular_values[0] <= 0.0:
        raise RuntimeError("Dependency decomposition produced no positive singular values.")
    numerical_rank = int((singular_values > singular_values[0] * tolerance).sum().item())
    if numerical_rank <= 0:
        raise RuntimeError("Dependency decomposition has zero numerical rank.")
    energy = singular_values[:numerical_rank].square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
    energy_rank = int(torch.searchsorted(cumulative, cumulative.new_tensor(energy_target)).item()) + 1
    return min(rank_budget, numerical_rank, energy_rank)


def parameter_measurement_rows(
    *,
    outputs: torch.Tensor,
    hidden: torch.Tensor,
    probe_weights: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    behavior_rows: list[torch.Tensor] = []
    for probe_index in range(outputs.shape[0]):
        row_weight = torch.sqrt(probe_weights[probe_index])
        for output_index in range(outputs.shape[1]):
            behavior_rows.append(
                flat_autograd_gradient(
                    row_weight * outputs[probe_index, output_index],
                    parameters,
                    retain_graph=True,
                    require_nonzero=False,
                    label=f"dependency_behavior_{probe_index}_{output_index}",
                )
            )

    geometry_rows: list[torch.Tensor] = []
    for (left, right), distance in hidden_pair_distances(hidden).items():
        pair_weight = torch.sqrt(probe_weights[left] * probe_weights[right])
        geometry_rows.append(
            flat_autograd_gradient(
                pair_weight * distance,
                parameters,
                retain_graph=True,
                require_nonzero=False,
                label=f"dependency_geometry_{left}_{right}",
            )
        )
    return behavior_rows, geometry_rows


def feature_measurement_rows(
    *,
    outputs: torch.Tensor,
    hidden: torch.Tensor,
    probe_weights: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    behavior_rows: list[torch.Tensor] = []
    for probe_index in range(outputs.shape[0]):
        row_weight = torch.sqrt(probe_weights[probe_index])
        for output_index in range(outputs.shape[1]):
            gradient = torch.autograd.grad(
                row_weight * outputs[probe_index, output_index],
                hidden,
                retain_graph=True,
                allow_unused=False,
            )[0]
            behavior_rows.append(gradient[probe_index].detach().to(dtype=torch.float32))

    geometry_rows: list[torch.Tensor] = []
    for (left, right), distance in hidden_pair_distances(hidden).items():
        pair_weight = torch.sqrt(probe_weights[left] * probe_weights[right])
        gradient = torch.autograd.grad(
            pair_weight * distance,
            hidden,
            retain_graph=True,
            allow_unused=False,
        )[0]
        geometry_rows.extend(
            [
                gradient[left].detach().to(dtype=torch.float32),
                gradient[right].detach().to(dtype=torch.float32),
            ]
        )
    return behavior_rows, geometry_rows


def control_plane_svd(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the small decomposition explicitly on the CPU control plane."""
    cpu_matrix = matrix.detach().to(device="cpu", dtype=torch.float32)
    _left, singular_values, right = torch.linalg.svd(cpu_matrix, full_matrices=False)
    if not torch.isfinite(singular_values).all() or not torch.isfinite(right).all():
        raise FloatingPointError("Dependency SVD produced non-finite values.")
    return singular_values, right


def feature_family_parameter_rows(
    *,
    hidden: torch.Tensor,
    probe_weights: torch.Tensor,
    feature_directions: torch.Tensor,
    feature_strengths: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    if feature_directions.shape[0] != feature_strengths.numel():
        raise ValueError("Feature directions and strengths have different ranks.")
    rows: list[torch.Tensor] = []
    for family_index in range(feature_directions.shape[0]):
        direction = feature_directions[family_index].to(hidden)
        strength = feature_strengths[family_index].to(hidden)
        for probe_index in range(hidden.shape[0]):
            coordinate = torch.dot(hidden[probe_index], direction)
            rows.append(
                flat_autograd_gradient(
                    torch.sqrt(probe_weights[probe_index]) * strength * coordinate,
                    parameters,
                    retain_graph=True,
                    require_nonzero=False,
                    label=f"feature_family_{family_index}_probe_{probe_index}",
                )
            )
    return rows


def build_dependency_basis(
    *,
    model: TinyFunctionModel,
    probe_inputs: torch.Tensor,
    probe_weights: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    args: argparse.Namespace,
) -> DependencyBasis:
    outputs, hidden = model(probe_inputs)
    feature_behavior, feature_geometry = feature_measurement_rows(
        outputs=outputs,
        hidden=hidden,
        probe_weights=probe_weights,
    )
    feature_matrix = torch.cat(
        [
            block_normalize(
                feature_behavior,
                block_weight=args.behavior_block_weight,
                label="feature behavior",
            ),
            block_normalize(
                feature_geometry,
                block_weight=args.geometry_block_weight,
                label="feature geometry",
            ),
        ],
        dim=0,
    )
    feature_singular, feature_right = control_plane_svd(feature_matrix)
    feature_rank = retained_rank(
        feature_singular,
        rank_budget=args.dependency_rank,
        tolerance=args.dependency_rank_tolerance,
        energy_target=args.dependency_energy,
    )
    feature_strengths = (feature_singular[:feature_rank] / feature_singular[0]).pow(
        args.dependency_power
    )
    feature_energy = feature_singular.square()
    feature_retained_energy = float(
        (
            feature_energy[:feature_rank].sum()
            / feature_energy.sum().clamp_min(1e-12)
        ).item()
    )

    parameter_behavior, parameter_geometry = parameter_measurement_rows(
        outputs=outputs,
        hidden=hidden,
        probe_weights=probe_weights,
        parameters=parameters,
    )
    family_parameter_rows = feature_family_parameter_rows(
        hidden=hidden,
        probe_weights=probe_weights,
        feature_directions=feature_right[:feature_rank],
        feature_strengths=feature_strengths,
        parameters=parameters,
    )
    parameter_matrix = torch.cat(
        [
            block_normalize(
                parameter_behavior,
                block_weight=args.behavior_block_weight,
                label="parameter behavior",
            ),
            block_normalize(
                parameter_geometry,
                block_weight=args.geometry_block_weight,
                label="parameter geometry",
            ),
            block_normalize(
                family_parameter_rows,
                block_weight=args.feature_block_weight,
                label="feature-family parameter",
            ),
        ],
        dim=0,
    )
    parameter_singular, parameter_right = control_plane_svd(parameter_matrix)
    parameter_rank = retained_rank(
        parameter_singular,
        rank_budget=args.dependency_rank,
        tolerance=args.dependency_rank_tolerance,
        energy_target=args.dependency_energy,
    )
    strengths = (parameter_singular[:parameter_rank] / parameter_singular[0]).pow(
        args.dependency_power
    )
    parameter_rows = (
        strengths.unsqueeze(1) * parameter_right[:parameter_rank]
    ).to(device=probe_inputs.device, dtype=torch.float32)
    parameter_energy = parameter_singular.square()
    retained_energy = float(
        (parameter_energy[:parameter_rank].sum() / parameter_energy.sum().clamp_min(1e-12)).item()
    )

    return DependencyBasis(
        constraint_rows=[row for row in parameter_rows],
        normalized_measurement_matrix=parameter_matrix.detach(),
        parameter_singular_values=[float(value) for value in parameter_singular],
        parameter_effective_rank=effective_rank(parameter_singular),
        parameter_retained_rank=parameter_rank,
        parameter_retained_energy=retained_energy,
        feature_singular_values=[float(value) for value in feature_singular],
        feature_effective_rank=effective_rank(feature_singular),
        feature_retained_rank=feature_rank,
        feature_retained_energy=feature_retained_energy,
    )


def normalized_damage(matrix: torch.Tensor, gradient: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(matrix @ gradient)
    denominator = torch.linalg.vector_norm(gradient)
    return float((numerator / denominator.clamp_min(1e-12)).detach().cpu())


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type != "cpu":
        raise ValueError(f"Unsupported timing device {device}.")


def train_dependency_field(
    *,
    base_model: TinyFunctionModel,
    stages: list[list[Any]],
    controls: list[TraceStageControl],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TinyFunctionModel, list[dict[str, Any]], torch.Tensor]:
    model = copy.deepcopy(base_model)
    parameters = list(model.parameters())
    stage_reports: list[dict[str, Any]] = []
    stage2_parameters: torch.Tensor | None = None

    for stage_index in (1, 2):
        stage_number = stage_index + 1
        inputs, targets = examples_to_tensors(stages[stage_index], device=device)
        control = controls[stage_index]
        write_weights = control.current_write_weights.to(device)
        prior_inputs = control.prior_probe_inputs.to(device)
        prior_weights = control.prior_probe_weights.to(device)
        if prior_inputs.shape[0] <= 1:
            raise RuntimeError("Dependency field requires at least two protected probes.")
        with torch.no_grad():
            teacher_outputs, teacher_hidden = model(prior_inputs)
            reference_distances = {
                pair: value.detach()
                for pair, value in hidden_pair_distances(teacher_hidden).items()
            }

        basis: DependencyBasis | None = None
        refresh_reports: list[dict[str, Any]] = []
        epoch_trace: list[dict[str, float]] = []
        for epoch in range(1, args.cl_epochs + 1):
            if basis is None or (epoch - 1) % args.dependency_refresh == 0:
                basis = build_dependency_basis(
                    model=model,
                    probe_inputs=prior_inputs,
                    probe_weights=prior_weights,
                    parameters=parameters,
                    args=args,
                )
                refresh_reports.append(
                    {
                        "epoch": epoch,
                        "parameter_singular_values": basis.parameter_singular_values,
                        "parameter_effective_rank": basis.parameter_effective_rank,
                        "parameter_retained_rank": basis.parameter_retained_rank,
                        "parameter_retained_energy": basis.parameter_retained_energy,
                        "feature_singular_values": basis.feature_singular_values,
                        "feature_effective_rank": basis.feature_effective_rank,
                        "feature_retained_rank": basis.feature_retained_rank,
                        "feature_retained_energy": basis.feature_retained_energy,
                    }
                )

            prediction, _new_hidden = model(inputs)
            new_loss = weighted_mse(prediction, targets, write_weights)
            current_old_outputs, current_old_hidden = model(prior_inputs)
            protection_loss = weighted_mse(
                current_old_outputs,
                teacher_outputs,
                prior_weights,
            )
            current_distances = hidden_pair_distances(current_old_hidden)
            geometry_terms = [
                torch.sqrt(prior_weights[left] * prior_weights[right])
                * (distance - reference_distances[(left, right)]).square()
                for (left, right), distance in current_distances.items()
            ]
            if not geometry_terms:
                raise RuntimeError("Dependency restore produced no geometry terms.")
            geometry_loss = torch.stack(geometry_terms).mean()

            raw_gradient = flat_autograd_gradient(
                new_loss,
                parameters,
                retain_graph=True,
                require_nonzero=True,
                label=f"dependency_field:stage{stage_number}:new",
            )
            tangent, projection_stats = project_gradient_away_from_constraints(
                raw_gradient=raw_gradient,
                constraint_gradients=basis.constraint_rows,
                damping=args.projection_damping,
                solver="gram",
                rank_tolerance=args.dependency_rank_tolerance,
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
                label=f"dependency_field:stage{stage_number}:restore",
            )
            restore_update = bounded_restore(
                restore_gradient,
                tangent,
                strength=args.restore_strength,
                bound_fraction=args.restore_bound_fraction,
            )
            final_gradient = tangent + restore_update
            raw_damage = normalized_damage(basis.normalized_measurement_matrix, raw_gradient)
            tangent_damage = normalized_damage(basis.normalized_measurement_matrix, tangent)
            final_damage = normalized_damage(basis.normalized_measurement_matrix, final_gradient)
            gradient_norm = apply_flat_update(
                parameters,
                final_gradient,
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
                        "safe_grad_fraction": projection_stats["safe_grad_fraction"],
                        "constraint_count": projection_stats["constraint_count"],
                        "raw_dependency_damage": raw_damage,
                        "tangent_dependency_damage": tangent_damage,
                        "final_dependency_damage": final_damage,
                        "restore_norm": float(
                            torch.linalg.vector_norm(restore_update).detach().cpu()
                        ),
                    }
                )

        stage_reports.append(
            {
                "stage": stage_number,
                "write_weight_mean": float(write_weights.mean().detach().cpu()),
                "protect_weight_mean": float(prior_weights.mean().detach().cpu()),
                "refreshes": refresh_reports,
                "trace": epoch_trace,
            }
        )
        if stage_number == 2:
            stage2_parameters = torch.cat(
                [parameter.detach().reshape(-1) for parameter in parameters]
            )

    if stage2_parameters is None:
        raise RuntimeError("Dependency field did not capture stage-2 parameters.")
    return model, stage_reports, stage2_parameters


def evaluate_sequential_method(
    *,
    model: TinyFunctionModel,
    stage2_parameters: torch.Tensor,
    retained_inputs: torch.Tensor,
    all_examples: list[Any],
    training_report: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    evaluation = evaluate_model(model, all_examples, device=device)
    with torch.no_grad():
        _outputs, final_hidden = model(retained_inputs)
    stage2_model = copy.deepcopy(model)
    offset = 0
    with torch.no_grad():
        for parameter in stage2_model.parameters():
            count = parameter.numel()
            parameter.copy_(stage2_parameters[offset : offset + count].reshape_as(parameter))
            offset += count
    if offset != stage2_parameters.numel():
        raise RuntimeError("Stage-2 parameter snapshot had unused entries.")
    with torch.no_grad():
        _outputs, stage2_hidden = stage2_model(retained_inputs)
    hidden_drift, geometry_drift = relative_representation_drift(
        stage2_hidden,
        final_hidden,
    )
    final_parameters = torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.parameters()]
    )
    return {
        "evaluation": {
            "mse": evaluation["mse"],
            "groups": evaluation["groups"],
        },
        "training": training_report,
        "stage3_hidden_cka": centered_kernel_alignment(stage2_hidden, final_hidden),
        "stage3_hidden_relative_drift": hidden_drift,
        "stage3_pair_geometry_drift": geometry_drift,
        "stage3_weight_drift": float(
            (
                torch.linalg.vector_norm(final_parameters - stage2_parameters)
                / torch.linalg.vector_norm(stage2_parameters).clamp_min(1e-12)
            ).detach().cpu()
        ),
    }


def plot_behavior(results: dict[str, dict[str, Any]], *, output_path: Path) -> None:
    methods = ["naive", "trace_loss_mix", "trace_invariant", "dependency_field", "joint_reference"]
    groups = sorted(results["naive"]["evaluation"]["groups"])
    x = torch.arange(len(groups), dtype=torch.float32).numpy()
    width = 0.16
    colors = ["#9aa0a6", "#f59e0b", "#2563eb", "#0f9d58", "#7c3aed"]
    fig, axis = plt.subplots(figsize=(13.5, 5.5))
    for index, (method, color) in enumerate(zip(methods, colors, strict=True)):
        values = [results[method]["evaluation"]["groups"][group]["mse"] for group in groups]
        axis.bar(
            x + (index - 2.0) * width,
            values,
            width=width,
            label=method.replace("_", " "),
            color=color,
        )
    axis.set_xticks(x, groups, rotation=30, ha="right")
    axis.set_yscale("symlog", linthresh=1e-3)
    axis.set_ylabel("mean squared error")
    axis.set_title("Behavior after trace-conditioned dependency updates")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_geometry(results: dict[str, dict[str, Any]], *, output_path: Path) -> None:
    methods = ["naive", "trace_loss_mix", "trace_invariant", "dependency_field"]
    labels = [method.replace("_", " ") for method in methods]
    x = torch.arange(len(methods), dtype=torch.float32).numpy()
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
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
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("relative drift (lower is better)")
    axes[0].set_title("Retained representation across final update")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x,
        [results[method]["stage3_weight_drift"] for method in methods],
        color=["#9aa0a6", "#f59e0b", "#2563eb", "#0f9d58"],
    )
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("relative parameter drift")
    axes[1].set_title("Final-stage weight movement")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dependency_spectrum(
    dependency_training: list[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for stage_report in dependency_training:
        final_refresh = stage_report["refreshes"][-1]
        parameter_values = final_refresh["parameter_singular_values"]
        feature_values = final_refresh["feature_singular_values"]
        axes[0].plot(
            range(1, len(parameter_values) + 1),
            parameter_values,
            marker="o",
            label=f"stage {stage_report['stage']}",
        )
        axes[1].plot(
            range(1, len(feature_values) + 1),
            feature_values,
            marker="o",
            label=f"stage {stage_report['stage']}",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("dependency direction")
    axes[0].set_ylabel("singular value")
    axes[0].set_title("Parameter consequence spectrum")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("feature-family direction")
    axes[1].set_ylabel("singular value")
    axes[1].set_title("Hidden feature-family spectrum")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
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
    retained_groups = {"merge_a", "merge_b", "stable", "branch_root", "branch_up", "branch_down"}
    retained_examples = [
        example for example in all_examples if example.hidden_group in retained_groups
    ]
    retained_inputs, _targets = examples_to_tensors(retained_examples, device=device)

    print("TINY TRACE-CONDITIONED FUNCTIONAL DEPENDENCY FIELD")
    print("=" * 144)
    print(
        f"device={device} slots={args.num_slots} dependency_rank={args.dependency_rank} "
        f"refresh={args.dependency_refresh} cl_epochs={args.cl_epochs}"
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
    for method in ("naive", "trace_loss_mix", "trace_invariant"):
        synchronize_device(device)
        started = time.perf_counter()
        model, training_report, snapshot = train_sequential_method(
            method=method,
            base_model=base_model,
            stages=stages,
            controls=controls,
            args=args,
            device=device,
        )
        synchronize_device(device)
        training_seconds = time.perf_counter() - started
        results[method] = evaluate_sequential_method(
            model=model,
            stage2_parameters=snapshot["parameters"].to(device),
            retained_inputs=retained_inputs,
            all_examples=all_examples,
            training_report=training_report,
            device=device,
        )
        results[method]["training_seconds"] = training_seconds

    synchronize_device(device)
    started = time.perf_counter()
    dependency_model, dependency_training, dependency_snapshot = train_dependency_field(
        base_model=base_model,
        stages=stages,
        controls=controls,
        args=args,
        device=device,
    )
    synchronize_device(device)
    dependency_seconds = time.perf_counter() - started
    results["dependency_field"] = evaluate_sequential_method(
        model=dependency_model,
        stage2_parameters=dependency_snapshot.to(device),
        retained_inputs=retained_inputs,
        all_examples=all_examples,
        training_report=dependency_training,
        device=device,
    )
    results["dependency_field"]["training_seconds"] = dependency_seconds

    joint_examples = [
        example for example in all_examples if example.hidden_group not in {"obsolete", "noise"}
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

    print("\nFINAL BEHAVIOR AND GEOMETRY")
    print("-" * 144)
    print(
        f"{'method':>18} {'retained':>10} {'novel':>10} {'obsolete':>10} {'noise':>10} "
        f"{'geomDrift':>10} {'weightDrift':>12} {'seconds':>10}"
    )
    for method in ("naive", "trace_loss_mix", "trace_invariant", "dependency_field", "joint_reference"):
        groups = results[method]["evaluation"]["groups"]
        retained = sum(groups[group]["mse"] for group in retained_groups) / len(retained_groups)
        geometry_drift = results[method].get("stage3_pair_geometry_drift")
        weight_drift = results[method].get("stage3_weight_drift")
        training_seconds = results[method].get("training_seconds")
        print(
            f"{method:>18} {retained:10.4g} {groups['novel']['mse']:10.4g} "
            f"{groups['obsolete']['mse']:10.4g} {groups['noise']['mse']:10.4g} "
            f"{('NA' if geometry_drift is None else f'{geometry_drift:.4f}'):>10} "
            f"{('NA' if weight_drift is None else f'{weight_drift:.4f}'):>12} "
            f"{('NA' if training_seconds is None else f'{training_seconds:.3f}'):>10}"
        )

    print("\nDEPENDENCY FIELD BY STAGE")
    print("-" * 144)
    print(
        f"{'stage':>7} {'paramEff':>10} {'keptRank':>10} {'keptEnergy':>12} "
        f"{'featureEff':>11} {'removed':>10} {'damageRaw':>11} {'damageSafe':>12}"
    )
    for stage_report in dependency_training:
        final_refresh = stage_report["refreshes"][-1]
        final_epoch = stage_report["trace"][-1]
        print(
            f"{stage_report['stage']:7d} {final_refresh['parameter_effective_rank']:10.3f} "
            f"{final_refresh['parameter_retained_rank']:10d} "
            f"{final_refresh['parameter_retained_energy']:12.4f} "
            f"{final_refresh['feature_effective_rank']:11.3f} "
            f"{final_epoch['projection_removed_fraction']:10.4f} "
            f"{final_epoch['raw_dependency_damage']:11.4f} "
            f"{final_epoch['final_dependency_damage']:12.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.output_dir / "functional_dependency_behavior.png"
    geometry_path = args.output_dir / "functional_dependency_geometry.png"
    spectrum_path = args.output_dir / "functional_dependency_spectrum.png"
    output_json = args.output_dir / "functional_dependency_field.json"
    plot_behavior(results, output_path=behavior_path)
    plot_geometry(results, output_path=geometry_path)
    plot_dependency_spectrum(dependency_training, output_path=spectrum_path)
    output = {
        "question": (
            "Can protected behavior and geometry Jacobians reveal a compact functional dependency "
            "basis that preserves load-bearing computation while retaining plastic directions?"
        ),
        "scope": (
            "Tiny nonlinear function model with bounded autonomous traces. Hidden group names are "
            "used only for evaluation and plot labels."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "trace_controls": [control.report for control in controls],
        "base_training": {"initial": base_trace[0], "final": base_trace[-1]},
        "joint_training": {"initial": joint_trace[0], "final": joint_trace[-1]},
        "results": results,
        "plots": {
            "behavior": str(behavior_path),
            "geometry": str(geometry_path),
            "spectrum": str(spectrum_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_plots={behavior_path},{geometry_path},{spectrum_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = build_integration_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=Path("model/analysis/gco-tiny-functional-dependency-field-seed0")
    )
    parser.add_argument("--dependency-rank", type=int, default=48)
    parser.add_argument("--dependency-energy", type=float, default=0.98)
    parser.add_argument("--dependency-refresh", type=int, default=20)
    parser.add_argument("--dependency-power", type=float, default=1.0)
    parser.add_argument("--dependency-rank-tolerance", type=float, default=1e-4)
    parser.add_argument("--behavior-block-weight", type=float, default=1.0)
    parser.add_argument("--geometry-block-weight", type=float, default=0.25)
    parser.add_argument("--feature-block-weight", type=float, default=0.5)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
