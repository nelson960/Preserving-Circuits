"""Test autonomous merge, branch, novelty, and forgetting in a bounded trace field.

The experiment isolates the evidence mechanism from language-model training and
Invariant-Tangent updates. A fixed number of differentiable trace slots must
compress an accumulated representation stream under one description-length
objective. No merge, branch, forget, novelty, event, or role label enters the
optimizer. Hidden labels are used only after fitting to measure what emerged.

The stream has three stages:

1. duplicate evidence sources, stable structure, one contextual root, and a
   rare trace that is still affordable;
2. the contextual root develops two genuine branches;
3. recurring novelty and isolated noise arrive while the slot budget is full.

The mechanism is supported only if the same objective merges the duplicates,
separates the branches, represents recurring novelty, rejects isolated noise,
retains stable structure, and releases the obsolete rare structure under
capacity pressure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device


@dataclass(frozen=True)
class EvidencePoint:
    vector: tuple[float, ...]
    hidden_group: str
    stage: int


@dataclass
class TraceSolution:
    centers: torch.Tensor
    attention: torch.Tensor
    usage: torch.Tensor
    reconstruction: torch.Tensor
    point_error: torch.Tensor
    objective: float
    residual_bits: float
    assignment_bits: float
    ambiguity_bits: float
    structure_bits: float
    effective_slots: float
    expected_active_slots: float


GROUP_COLORS = {
    "merge_a": "#1f77b4",
    "merge_b": "#6baed6",
    "stable": "#2ca02c",
    "branch_root": "#9467bd",
    "branch_up": "#d62728",
    "branch_down": "#ff7f0e",
    "obsolete": "#8c564b",
    "novel": "#17becf",
    "noise": "#7f7f7f",
}


def validate_args(args: argparse.Namespace) -> None:
    if args.num_slots < 2:
        raise ValueError(f"--num-slots must be >= 2, got {args.num_slots}.")
    if args.num_slots > 8:
        raise ValueError("--num-slots must be <= 8 because diagnostic slot alignment is exhaustive.")
    if args.d_model < 6:
        raise ValueError(f"--d-model must be >= 6, got {args.d_model}.")
    if args.trace_steps <= 0:
        raise ValueError(f"--trace-steps must be positive, got {args.trace_steps}.")
    if args.restarts <= 0:
        raise ValueError(f"--restarts must be positive, got {args.restarts}.")
    if args.trace_lr <= 0.0:
        raise ValueError(f"--trace-lr must be positive, got {args.trace_lr}.")
    if args.observation_sigma <= 0.0:
        raise ValueError(f"--observation-sigma must be positive, got {args.observation_sigma}.")
    if args.encoding_span <= args.observation_sigma:
        raise ValueError(
            "--encoding-span must exceed --observation-sigma so a trace center has positive coding cost."
        )
    if not 0.0 < args.concept_radius < args.encoding_span:
        raise ValueError(
            f"--concept-radius must be in (0, encoding_span), got {args.concept_radius}."
        )
    if args.attention_temperature <= 0.0:
        raise ValueError(
            f"--attention-temperature must be positive, got {args.attention_temperature}."
        )
    if args.ambiguity_weight < 0.0:
        raise ValueError(f"--ambiguity-weight must be non-negative, got {args.ambiguity_weight}.")
    if args.initial_logit_floor >= 0.0:
        raise ValueError(f"--initial-logit-floor must be negative, got {args.initial_logit_floor}.")
    for name in (
        "merge_points",
        "stable_points",
        "root_points",
        "obsolete_points",
        "branch_points",
        "novel_points",
        "noise_points",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}.")


def sample_group(
    *,
    center: torch.Tensor,
    count: int,
    sigma: float,
    hidden_group: str,
    stage: int,
    generator: torch.Generator,
) -> list[EvidencePoint]:
    if center.ndim != 1:
        raise ValueError(f"Group center must be one-dimensional, got {tuple(center.shape)}.")
    values = center.to(dtype=torch.float64).unsqueeze(0) + sigma * torch.randn(
        count,
        center.numel(),
        generator=generator,
        dtype=torch.float64,
    )
    return [
        EvidencePoint(
            vector=tuple(float(component) for component in value),
            hidden_group=hidden_group,
            stage=stage,
        )
        for value in values
    ]


def build_stream(args: argparse.Namespace) -> list[list[EvidencePoint]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    sigma = args.observation_sigma
    radius = args.concept_radius
    basis_source = torch.randn(args.d_model, args.d_model, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(basis_source)
    merge_center = radius * basis[:, 0]
    stable_center = radius * basis[:, 1]
    obsolete_center = radius * basis[:, 2]
    branch_up_center = radius * basis[:, 3]
    branch_down_center = radius * basis[:, 4]
    branch_root_center = 0.5 * (branch_up_center + branch_down_center)
    novel_center = radius * basis[:, 5]

    stage1: list[EvidencePoint] = []
    stage1.extend(
        sample_group(
            center=merge_center,
            count=args.merge_points,
            sigma=sigma,
            hidden_group="merge_a",
            stage=1,
            generator=generator,
        )
    )
    stage1.extend(
        sample_group(
            center=merge_center,
            count=args.merge_points,
            sigma=sigma,
            hidden_group="merge_b",
            stage=1,
            generator=generator,
        )
    )
    stage1.extend(
        sample_group(
            center=stable_center,
            count=args.stable_points,
            sigma=sigma,
            hidden_group="stable",
            stage=1,
            generator=generator,
        )
    )
    stage1.extend(
        sample_group(
            center=branch_root_center,
            count=args.root_points,
            sigma=sigma,
            hidden_group="branch_root",
            stage=1,
            generator=generator,
        )
    )
    stage1.extend(
        sample_group(
            center=obsolete_center,
            count=args.obsolete_points,
            sigma=sigma,
            hidden_group="obsolete",
            stage=1,
            generator=generator,
        )
    )

    stage2: list[EvidencePoint] = []
    stage2.extend(
        sample_group(
            center=merge_center,
            count=args.merge_points // 2,
            sigma=sigma,
            hidden_group="merge_a",
            stage=2,
            generator=generator,
        )
    )
    stage2.extend(
        sample_group(
            center=stable_center,
            count=args.stable_points // 2,
            sigma=sigma,
            hidden_group="stable",
            stage=2,
            generator=generator,
        )
    )
    stage2.extend(
        sample_group(
            center=branch_up_center,
            count=args.branch_points,
            sigma=sigma,
            hidden_group="branch_up",
            stage=2,
            generator=generator,
        )
    )
    stage2.extend(
        sample_group(
            center=branch_down_center,
            count=args.branch_points,
            sigma=sigma,
            hidden_group="branch_down",
            stage=2,
            generator=generator,
        )
    )

    stage3: list[EvidencePoint] = []
    stage3.extend(
        sample_group(
            center=merge_center,
            count=args.merge_points // 2,
            sigma=sigma,
            hidden_group="merge_b",
            stage=3,
            generator=generator,
        )
    )
    stage3.extend(
        sample_group(
            center=stable_center,
            count=args.stable_points // 2,
            sigma=sigma,
            hidden_group="stable",
            stage=3,
            generator=generator,
        )
    )
    stage3.extend(
        sample_group(
            center=branch_up_center,
            count=args.branch_points // 2,
            sigma=sigma,
            hidden_group="branch_up",
            stage=3,
            generator=generator,
        )
    )
    stage3.extend(
        sample_group(
            center=branch_down_center,
            count=args.branch_points // 2,
            sigma=sigma,
            hidden_group="branch_down",
            stage=3,
            generator=generator,
        )
    )
    stage3.extend(
        sample_group(
            center=novel_center,
            count=args.novel_points,
            sigma=sigma,
            hidden_group="novel",
            stage=3,
            generator=generator,
        )
    )

    noise = torch.randn(args.noise_points, args.d_model, generator=generator, dtype=torch.float64)
    noise = noise / noise.norm(dim=1, keepdim=True)
    noise_radius = radius * (0.75 + 0.5 * torch.rand(args.noise_points, 1, generator=generator))
    noise = noise * noise_radius
    stage3.extend(
        EvidencePoint(
            vector=tuple(float(component) for component in value),
            hidden_group="noise",
            stage=3,
        )
        for value in noise
    )
    return [stage1, stage2, stage3]


def points_to_tensor(points: list[EvidencePoint], *, device: torch.device) -> torch.Tensor:
    if not points:
        raise ValueError("Cannot build a trace field from zero evidence points.")
    return torch.tensor([point.vector for point in points], dtype=torch.float32, device=device)


def density_weighted_centers(
    points: torch.Tensor,
    *,
    num_slots: int,
    first_index: int,
    observation_sigma: float,
) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError(f"Expected points [N, D], got {tuple(points.shape)}.")
    if not 0 <= first_index < points.shape[0]:
        raise ValueError(f"first_index={first_index} is outside [0, {points.shape[0]}).")
    pairwise_distance = torch.cdist(points, points).square()
    density = torch.exp(-pairwise_distance / (2.0 * observation_sigma**2)).sum(dim=1)
    selected = [first_index]
    min_distance = ((points - points[first_index]) ** 2).sum(dim=1)
    for _ in range(1, num_slots):
        score = min_distance * density
        score[selected] = -torch.inf
        index = int(torch.argmax(score).item())
        selected.append(index)
        candidate_distance = ((points - points[index]) ** 2).sum(dim=1)
        min_distance = torch.minimum(min_distance, candidate_distance)
    return points[selected].clone()


def initial_centers(
    points: torch.Tensor,
    *,
    num_slots: int,
    restart: int,
    previous: torch.Tensor | None,
    observation_sigma: float,
    search_seed: int,
) -> torch.Tensor:
    if restart == 0 and previous is not None:
        if previous.shape != (num_slots, points.shape[1]):
            raise ValueError(
                f"Previous centers must be {(num_slots, points.shape[1])}, got {tuple(previous.shape)}."
            )
        return previous.detach().clone()
    search_rng = random.Random(search_seed + restart)
    first_index = search_rng.randrange(points.shape[0])
    return density_weighted_centers(
        points,
        num_slots=num_slots,
        first_index=first_index,
        observation_sigma=observation_sigma,
    )


def trace_objective(
    *,
    points: torch.Tensor,
    centers: torch.Tensor,
    assignment_logits: torch.Tensor,
    observation_sigma: float,
    encoding_span: float,
    attention_temperature: float,
    ambiguity_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if assignment_logits.shape != (points.shape[0], centers.shape[0]):
        raise ValueError(
            f"Assignment logits must be {(points.shape[0], centers.shape[0])}, "
            f"got {tuple(assignment_logits.shape)}."
        )
    eps = torch.finfo(points.dtype).eps
    attention = torch.softmax(assignment_logits / attention_temperature, dim=1)
    reconstruction = attention @ centers
    squared_error = ((points - reconstruction) ** 2).sum(dim=1)

    gaussian_bits = squared_error / (2.0 * observation_sigma**2 * math.log(2.0))
    outlier_bits = points.shape[1] * math.log2((2.0 * encoding_span) / observation_sigma)
    residual_bits = (outlier_bits * torch.log1p(gaussian_bits / outlier_bits)).sum()

    usage = attention.mean(dim=0)
    assignment_bits = -(attention * torch.log2(usage.clamp_min(eps)).unsqueeze(0)).sum()
    ambiguity_bits = -(attention * torch.log2(attention.clamp_min(eps))).sum()

    expected_active = (1.0 - (1.0 - usage.clamp(max=1.0 - eps)) ** points.shape[0]).sum()
    center_bits = points.shape[1] * math.log2((2.0 * encoding_span) / observation_sigma)
    structure_bits = center_bits * expected_active

    objective = (
        residual_bits
        + assignment_bits
        + ambiguity_weight * ambiguity_bits
        + structure_bits
    ) / points.shape[0]
    return objective, {
        "attention": attention,
        "reconstruction": reconstruction,
        "squared_error": squared_error,
        "usage": usage,
        "residual_bits": residual_bits,
        "assignment_bits": assignment_bits,
        "ambiguity_bits": ambiguity_bits,
        "structure_bits": structure_bits,
        "expected_active": expected_active,
    }


def fit_trace_field(
    points: torch.Tensor,
    *,
    args: argparse.Namespace,
    previous_centers: torch.Tensor | None,
    stage: int,
) -> TraceSolution:
    best: TraceSolution | None = None
    for restart in range(args.restarts):
        centers = initial_centers(
            points,
            num_slots=args.num_slots,
            restart=restart,
            previous=previous_centers,
            observation_sigma=args.observation_sigma,
            search_seed=args.seed + stage * args.restarts,
        ).requires_grad_(True)
        initial_distance = torch.cdist(points, centers).square()
        assignment_logits = (
            -args.attention_temperature
            * initial_distance
            / (2.0 * args.observation_sigma**2)
        ).clamp_min(args.attention_temperature * args.initial_logit_floor).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([centers, assignment_logits], lr=args.trace_lr)

        for step in range(1, args.trace_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            objective, _parts = trace_objective(
                points=points,
                centers=centers,
                assignment_logits=assignment_logits,
                observation_sigma=args.observation_sigma,
                encoding_span=args.encoding_span,
                attention_temperature=args.attention_temperature,
                ambiguity_weight=args.ambiguity_weight,
            )
            if not torch.isfinite(objective):
                diagnostics = {
                    name: (
                        float(value.detach().cpu().item())
                        if value.numel() == 1
                        else {
                            "finite": bool(torch.isfinite(value).all().detach().cpu().item()),
                            "min": float(value.detach().amin().cpu().item()),
                            "max": float(value.detach().amax().cpu().item()),
                        }
                    )
                    for name, value in _parts.items()
                }
                raise FloatingPointError(
                    f"Non-finite trace objective at stage={stage}, restart={restart}, step={step}: "
                    f"{diagnostics}"
                )
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                centers.clamp_(-args.encoding_span, args.encoding_span)

        with torch.no_grad():
            objective, parts = trace_objective(
                points=points,
                centers=centers,
                assignment_logits=assignment_logits,
                observation_sigma=args.observation_sigma,
                encoding_span=args.encoding_span,
                attention_temperature=args.attention_temperature,
                ambiguity_weight=args.ambiguity_weight,
            )
            usage = parts["usage"]
            entropy = -(usage * torch.log(usage.clamp_min(torch.finfo(usage.dtype).eps))).sum()
            solution = TraceSolution(
                centers=centers.detach().clone(),
                attention=parts["attention"].detach().clone(),
                usage=usage.detach().clone(),
                reconstruction=parts["reconstruction"].detach().clone(),
                point_error=parts["squared_error"].detach().clone(),
                objective=float(objective.item()),
                residual_bits=float(parts["residual_bits"].item()),
                assignment_bits=float(parts["assignment_bits"].item()),
                ambiguity_bits=float(parts["ambiguity_bits"].item()),
                structure_bits=float(parts["structure_bits"].item()),
                effective_slots=float(torch.exp(entropy).item()),
                expected_active_slots=float(parts["expected_active"].item()),
            )
        if best is None or solution.objective < best.objective:
            best = solution
    if best is None:
        raise RuntimeError(f"Trace optimization produced no solution for stage {stage}.")
    return best


def align_solution(previous: TraceSolution, current: TraceSolution) -> TraceSolution:
    num_slots = previous.centers.shape[0]
    if current.centers.shape[0] != num_slots:
        raise ValueError("Cannot align trace solutions with different slot counts.")
    cost = torch.cdist(previous.centers, current.centers).square().detach().cpu()
    best_permutation: tuple[int, ...] | None = None
    best_cost: float | None = None
    for permutation in itertools.permutations(range(num_slots)):
        value = sum(float(cost[index, permutation[index]]) for index in range(num_slots))
        if best_cost is None or value < best_cost:
            best_cost = value
            best_permutation = permutation
    if best_permutation is None:
        raise RuntimeError("Trace alignment did not produce a permutation.")
    order = torch.tensor(best_permutation, device=current.centers.device, dtype=torch.long)
    return TraceSolution(
        centers=current.centers[order],
        attention=current.attention[:, order],
        usage=current.usage[order],
        reconstruction=current.reconstruction,
        point_error=current.point_error,
        objective=current.objective,
        residual_bits=current.residual_bits,
        assignment_bits=current.assignment_bits,
        ambiguity_bits=current.ambiguity_bits,
        structure_bits=current.structure_bits,
        effective_slots=current.effective_slots,
        expected_active_slots=current.expected_active_slots,
    )


def group_metrics(points: list[EvidencePoint], solution: TraceSolution) -> dict[str, dict[str, Any]]:
    groups = sorted({point.hidden_group for point in points})
    output: dict[str, dict[str, Any]] = {}
    for group in groups:
        indices = [index for index, point in enumerate(points) if point.hidden_group == group]
        index_tensor = torch.tensor(indices, device=solution.attention.device, dtype=torch.long)
        group_attention = solution.attention[index_tensor]
        mean_attention = group_attention.mean(dim=0)
        attention_dispersion = (group_attention - mean_attention).abs().sum(dim=1).mean()
        output[group] = {
            "count": len(indices),
            "dominant_slot": int(mean_attention.argmax().item()),
            "dominant_share": float(mean_attention.max().item()),
            "mean_attention": [float(value) for value in mean_attention.detach().cpu()],
            "mean_attention_l1_dispersion": float(attention_dispersion.item()),
            "mean_squared_error": float(solution.point_error[index_tensor].mean().item()),
        }
    return output


def qualitative_checks(stage_reports: list[dict[str, Any]]) -> dict[str, Any]:
    stage1 = stage_reports[0]["groups"]
    stage3 = stage_reports[2]["groups"]
    noise_slot = stage3["noise"]["dominant_slot"]
    non_noise_groups = [group for group in stage3 if group != "noise"]
    non_noise_slots = {stage3[group]["dominant_slot"] for group in non_noise_groups}
    merge_a_code = torch.tensor(stage1["merge_a"]["mean_attention"])
    merge_b_code = torch.tensor(stage1["merge_b"]["mean_attention"])
    branch_up_code = torch.tensor(stage3["branch_up"]["mean_attention"])
    branch_down_code = torch.tensor(stage3["branch_down"]["mean_attention"])
    branch_root_code = torch.tensor(stage3["branch_root"]["mean_attention"])
    merge_code_distance = float((merge_a_code - merge_b_code).abs().sum().item())
    branch_code_distance = float((branch_up_code - branch_down_code).abs().sum().item())
    root_to_composition = float(
        (branch_root_code - 0.5 * (branch_up_code + branch_down_code)).abs().sum().item()
    )
    root_to_up = float((branch_root_code - branch_up_code).abs().sum().item())
    root_to_down = float((branch_root_code - branch_down_code).abs().sum().item())
    merge_noise_scale = (
        stage1["merge_a"]["mean_attention_l1_dispersion"]
        + stage1["merge_b"]["mean_attention_l1_dispersion"]
    )
    branch_noise_scale = (
        stage3["branch_up"]["mean_attention_l1_dispersion"]
        + stage3["branch_down"]["mean_attention_l1_dispersion"]
    )
    return {
        "duplicate_code_l1_distance": merge_code_distance,
        "duplicate_within_group_l1_scale": merge_noise_scale,
        "duplicate_sources_merge": merge_code_distance <= merge_noise_scale,
        "branch_code_l1_distance": branch_code_distance,
        "branch_within_group_l1_scale": branch_noise_scale,
        "context_branches_remain_distinct": branch_code_distance > branch_noise_scale,
        "root_to_branch_composition_l1": root_to_composition,
        "root_to_nearest_single_branch_l1": min(root_to_up, root_to_down),
        "shared_branch_root_is_compositional": root_to_composition < min(root_to_up, root_to_down),
        "novel_is_better_represented_than_released_obsolete": stage3["novel"]["mean_squared_error"]
        < stage3["obsolete"]["mean_squared_error"],
        "shared_branch_root_survives_as_composition": stage3["branch_root"]["mean_squared_error"]
        < stage3["obsolete"]["mean_squared_error"],
        "noise_has_no_exclusive_slot": noise_slot in non_noise_slots,
        "obsolete_error_increased_under_pressure": stage3["obsolete"]["mean_squared_error"]
        > stage1["obsolete"]["mean_squared_error"],
        "stable_error_remained_below_obsolete_error": stage3["stable"]["mean_squared_error"]
        < stage3["obsolete"]["mean_squared_error"],
    }


def plot_geometry(
    *,
    accumulated: list[list[EvidencePoint]],
    solutions: list[TraceSolution],
    output_path: Path,
) -> None:
    final_points = torch.tensor(
        [point.vector for point in accumulated[-1]],
        dtype=torch.float32,
    )
    projection_mean = final_points.mean(dim=0)
    _u, _s, vh = torch.linalg.svd(final_points - projection_mean, full_matrices=False)
    projection_basis = vh[:2].transpose(0, 1)

    def project(values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != projection_mean.numel():
            raise ValueError(
                f"Geometry projection expected [N, {projection_mean.numel()}], got {tuple(values.shape)}."
            )
        return (values - projection_mean) @ projection_basis

    fig, axes = plt.subplots(1, len(solutions), figsize=(5.2 * len(solutions), 4.8), sharex=True, sharey=True)
    if len(solutions) == 1:
        axes = [axes]
    for stage_index, (axis, points, solution) in enumerate(zip(axes, accumulated, solutions), start=1):
        for group in sorted({point.hidden_group for point in points}):
            values = [point.vector for point in points if point.hidden_group == group]
            projected = project(torch.tensor(values, dtype=torch.float32))
            axis.scatter(
                projected[:, 0].numpy(),
                projected[:, 1].numpy(),
                s=18,
                alpha=0.55,
                color=GROUP_COLORS[group],
                label=group,
            )
        centers = project(solution.centers.detach().cpu())
        usage = solution.usage.detach().cpu()
        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            s=180.0 + 900.0 * usage.numpy(),
            marker="X",
            color="#111111",
            edgecolor="white",
            linewidth=1.0,
            label="trace slot",
        )
        for slot, center in enumerate(centers):
            axis.text(float(center[0]) + 0.08, float(center[1]) + 0.08, str(slot), fontsize=9)
        axis.set_title(f"Stage {stage_index}: {solution.effective_slots:.2f} effective slots")
        axis.set_xlabel("representation PC1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("representation PC2")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Autonomous bounded trace organization", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_capacity(*, solutions: list[TraceSolution], output_path: Path) -> None:
    stages = list(range(1, len(solutions) + 1))
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for slot in range(solutions[0].usage.numel()):
        axis.plot(
            stages,
            [float(solution.usage[slot].item()) for solution in solutions],
            marker="o",
            linewidth=2,
            label=f"slot {slot}",
        )
    axis.set_xticks(stages)
    axis.set_xlabel("stream stage")
    axis.set_ylabel("fraction of evidence assigned")
    axis.set_title("Trace capacity redistribution")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_error(*, stage_reports: list[dict[str, Any]], output_path: Path) -> None:
    groups = sorted({group for report in stage_reports for group in report["groups"]})
    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    for group in groups:
        xs: list[int] = []
        ys: list[float] = []
        for stage, report in enumerate(stage_reports, start=1):
            if group in report["groups"]:
                xs.append(stage)
                ys.append(report["groups"][group]["mean_squared_error"])
        axis.plot(xs, ys, marker="o", linewidth=2, color=GROUP_COLORS[group], label=group)
    axis.set_xticks(range(1, len(stage_reports) + 1))
    axis.set_xlabel("stream stage")
    axis.set_ylabel("mean reconstruction error")
    axis.set_title("What the bounded trace field continues to represent")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    stages = build_stream(args)

    accumulated_points: list[EvidencePoint] = []
    accumulated_by_stage: list[list[EvidencePoint]] = []
    solutions: list[TraceSolution] = []
    stage_reports: list[dict[str, Any]] = []
    previous_centers: torch.Tensor | None = None

    print("TINY AUTONOMOUS TRACE FIELD")
    print("=" * 120)
    print(
        f"device={device} slots={args.num_slots} steps={args.trace_steps} restarts={args.restarts} "
        f"sigma={args.observation_sigma}"
    )

    for stage_index, stage_points in enumerate(stages, start=1):
        accumulated_points.extend(stage_points)
        accumulated_by_stage.append(list(accumulated_points))
        point_tensor = points_to_tensor(accumulated_points, device=device)
        solution = fit_trace_field(
            point_tensor,
            args=args,
            previous_centers=previous_centers,
            stage=stage_index,
        )
        if solutions:
            solution = align_solution(solutions[-1], solution)
        solutions.append(solution)
        previous_centers = solution.centers
        groups = group_metrics(accumulated_points, solution)
        report = {
            "stage": stage_index,
            "num_points": len(accumulated_points),
            "objective_bits_per_point": solution.objective,
            "residual_bits": solution.residual_bits,
            "assignment_bits": solution.assignment_bits,
            "ambiguity_bits": solution.ambiguity_bits,
            "structure_bits": solution.structure_bits,
            "effective_slots": solution.effective_slots,
            "expected_active_slots": solution.expected_active_slots,
            "usage": [float(value) for value in solution.usage.detach().cpu()],
            "centers": solution.centers.detach().cpu().tolist(),
            "groups": groups,
        }
        stage_reports.append(report)
        print(
            f"stage={stage_index} points={len(accumulated_points):3d} "
            f"cost={solution.objective:.4f} effective_slots={solution.effective_slots:.3f} "
            f"active={solution.expected_active_slots:.3f}"
        )
        for group, row in groups.items():
            print(
                f"  {group:>12} slot={row['dominant_slot']} share={row['dominant_share']:.3f} "
                f"error={row['mean_squared_error']:.4f} n={row['count']}"
            )

    checks = qualitative_checks(stage_reports)
    print("\nEMERGENT STRUCTURE CHECKS")
    print("-" * 120)
    for name, value in checks.items():
        print(f"{name:>48} = {value}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = args.output_dir / "trace_geometry.png"
    capacity_path = args.output_dir / "trace_capacity.png"
    group_error_path = args.output_dir / "trace_group_error.png"
    output_json = args.output_dir / "trace_field.json"
    plot_geometry(
        accumulated=accumulated_by_stage,
        solutions=solutions,
        output_path=geometry_path,
    )
    plot_capacity(solutions=solutions, output_path=capacity_path)
    plot_group_error(stage_reports=stage_reports, output_path=group_error_path)

    output = {
        "question": (
            "Can one fixed-capacity differentiable trace objective autonomously produce merge, branch, "
            "novel retention, noise rejection, and capacity-driven forgetting without action labels?"
        ),
        "scope": (
            "This isolates the trace objective in representation space. Hidden groups are evaluation-only. "
            "It does not test language-model weights, online evidence compression, or Invariant-Tangent CL."
        ),
        "mechanism": {
            "optimizer_inputs": ["representation vectors"],
            "optimizer_excludes": [
                "hidden event labels",
                "merge labels",
                "branch labels",
                "forget labels",
                "novelty labels",
                "role labels",
            ],
            "objective_terms": [
                "robust logarithmic residual code length",
                "trace assignment code length",
                "soft-assignment ambiguity",
                "expected active trace storage",
            ],
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stream": [[asdict(point) for point in stage] for stage in stages],
        "stages": stage_reports,
        "checks": checks,
        "plots": {
            "geometry": str(geometry_path),
            "capacity": str(capacity_path),
            "group_error": str(group_error_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_plots={geometry_path},{capacity_path},{group_error_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/analysis/gco-tiny-autonomous-trace-field-seed0"),
    )
    parser.add_argument("--num-slots", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--trace-steps", type=int, default=700)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--trace-lr", type=float, default=0.03)
    parser.add_argument("--attention-temperature", type=float, default=0.25)
    parser.add_argument("--initial-logit-floor", type=float, default=-12.0)
    parser.add_argument("--ambiguity-weight", type=float, default=0.2)
    parser.add_argument("--observation-sigma", type=float, default=0.24)
    parser.add_argument("--encoding-span", type=float, default=5.0)
    parser.add_argument("--concept-radius", type=float, default=3.2)
    parser.add_argument("--merge-points", type=int, default=24)
    parser.add_argument("--stable-points", type=int, default=30)
    parser.add_argument("--root-points", type=int, default=20)
    parser.add_argument("--obsolete-points", type=int, default=6)
    parser.add_argument("--branch-points", type=int, default=24)
    parser.add_argument("--novel-points", type=int, default=30)
    parser.add_argument("--noise-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
