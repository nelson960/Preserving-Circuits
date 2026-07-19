"""Test whether downstream consequence protects a rare shared function.

The stream gives one function very little direct recurrence while several
recurring functions reuse its input/output factors. Frequent local evidence and
new pressure consume the same fixed trace budget. Hidden group names construct
and evaluate the controlled stream; they never enter trace survival or weight
updates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.gco_math.gco_tiny_dependency_utility_survival import (
    build_parser as build_survival_parser,
    run_with_stages,
)
from experiments.gco_math.gco_tiny_trace_invariant_tangent_integration import (
    FunctionExample,
)


def validate_stream_args(args: argparse.Namespace) -> None:
    if args.d_model < 8:
        raise ValueError("The rare-functional stream requires --d-model >= 8.")
    if args.output_dim < 6:
        raise ValueError("The rare-functional stream requires --output-dim >= 6.")
    for name in (
        "rare_critical_points",
        "dependent_base_points",
        "dependent_replay_points",
        "dependent_final_points",
        "frequent_base_points",
        "frequent_replay_points",
        "rare_obsolete_points",
        "pressure_stage2_points",
        "pressure_stage3_points",
        "replacement_points",
        "rare_noise_points",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("rare_cluster_sigma", "rare_center_scale", "rare_branch_span"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")


def basis(dimension: int, index: int) -> torch.Tensor:
    if not 0 <= index < dimension:
        raise ValueError(f"Basis index {index} is outside dimension {dimension}.")
    result = torch.zeros(dimension, dtype=torch.float32)
    result[index] = 1.0
    return result


def clustered_examples(
    *,
    center: torch.Tensor,
    target: torch.Tensor,
    count: int,
    sigma: float,
    group: str,
    stage: int,
    generator: torch.Generator,
) -> list[FunctionExample]:
    if center.ndim != 1 or target.ndim != 1:
        raise ValueError("Cluster center and target must be vectors.")
    if count <= 0:
        raise ValueError("Cluster count must be positive.")
    noise = sigma * torch.randn(count, center.numel(), generator=generator)
    values = center.unsqueeze(0) + noise
    return [
        FunctionExample(
            x=tuple(float(value) for value in values[index]),
            target=tuple(float(value) for value in target),
            hidden_group=group,
            stage=stage,
        )
        for index in range(count)
    ]


def noise_examples(
    *,
    args: argparse.Namespace,
    stage: int,
    generator: torch.Generator,
) -> list[FunctionExample]:
    inputs = torch.randn(args.rare_noise_points, args.d_model, generator=generator)
    inputs = args.rare_center_scale * inputs / torch.linalg.vector_norm(
        inputs,
        dim=1,
        keepdim=True,
    )
    targets = torch.randn(args.rare_noise_points, args.output_dim, generator=generator)
    targets = targets / torch.linalg.vector_norm(targets, dim=1, keepdim=True)
    return [
        FunctionExample(
            x=tuple(float(value) for value in inputs[index]),
            target=tuple(float(value) for value in targets[index]),
            hidden_group="noise",
            stage=stage,
        )
        for index in range(args.rare_noise_points)
    ]


def build_rare_functional_stream(args: argparse.Namespace) -> list[list[FunctionExample]]:
    validate_stream_args(args)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 31001)

    x = [basis(args.d_model, index) for index in range(8)]
    y = [basis(args.output_dim, index) for index in range(6)]
    hub_center = args.rare_center_scale * x[0]
    left_center = hub_center + args.rare_branch_span * x[1]
    right_center = hub_center - args.rare_branch_span * x[1]
    frequent_center = args.rare_center_scale * x[2]
    obsolete_center = args.rare_center_scale * x[3]
    pressure_a_center = args.rare_center_scale * x[4]
    pressure_b_center = args.rare_center_scale * x[5]

    hub_target = y[0]
    left_target = y[0] + 0.35 * y[1]
    right_target = y[0] - 0.35 * y[1]
    frequent_target = y[2]
    obsolete_target = y[3]
    pressure_a_target = y[4]
    pressure_b_target = y[5]
    replacement_target = y[5]

    def cluster(
        center: torch.Tensor,
        target: torch.Tensor,
        count: int,
        group: str,
        stage: int,
    ) -> list[FunctionExample]:
        return clustered_examples(
            center=center,
            target=target,
            count=count,
            sigma=args.rare_cluster_sigma,
            group=group,
            stage=stage,
            generator=generator,
        )

    stage1 = (
        cluster(hub_center, hub_target, args.rare_critical_points, "rare_critical", 1)
        + cluster(
            left_center,
            left_target,
            args.dependent_base_points,
            "dependent_left",
            1,
        )
        + cluster(
            right_center,
            right_target,
            args.dependent_base_points,
            "dependent_right",
            1,
        )
        + cluster(
            frequent_center,
            frequent_target,
            args.frequent_base_points,
            "frequent_local",
            1,
        )
        + cluster(
            obsolete_center,
            obsolete_target,
            args.rare_obsolete_points,
            "obsolete",
            1,
        )
    )
    stage2 = (
        cluster(
            left_center,
            left_target,
            args.dependent_replay_points,
            "dependent_left",
            2,
        )
        + cluster(
            right_center,
            right_target,
            args.dependent_replay_points,
            "dependent_right",
            2,
        )
        + cluster(
            frequent_center,
            frequent_target,
            args.frequent_replay_points,
            "frequent_local",
            2,
        )
        + cluster(
            pressure_a_center,
            pressure_a_target,
            args.pressure_stage2_points,
            "pressure_a",
            2,
        )
        + cluster(
            pressure_b_center,
            pressure_b_target,
            args.pressure_stage2_points,
            "pressure_b",
            2,
        )
    )
    stage3 = (
        cluster(
            left_center,
            left_target,
            args.dependent_final_points,
            "dependent_left",
            3,
        )
        + cluster(
            right_center,
            right_target,
            args.dependent_final_points,
            "dependent_right",
            3,
        )
        + cluster(
            pressure_a_center,
            pressure_a_target,
            args.pressure_stage3_points,
            "pressure_a",
            3,
        )
        + cluster(
            pressure_b_center,
            pressure_b_target,
            args.pressure_stage3_points,
            "pressure_b",
            3,
        )
        + cluster(
            obsolete_center,
            replacement_target,
            args.replacement_points,
            "replacement",
            3,
        )
        + noise_examples(args=args, stage=3, generator=generator)
    )
    return [stage1, stage2, stage3]


def rare_gate(output: dict[str, Any]) -> dict[str, Any]:
    initial_groups = output["events"][0]["evaluation"]["groups"]
    final_groups = output["final"]["evaluation"]["groups"]
    trace_groups = output["final"]["trace_groups"]
    rare_slot = int(trace_groups["rare_critical"]["dominant_slot"])
    rare_history: list[dict[str, float | int]] = []
    for event in output["events"][1:]:
        event_slot = int(event["trace_groups"]["rare_critical"]["dominant_slot"])
        rare_history.append(
            {
                "event": int(event["event"]),
                "slot": event_slot,
                "survival": event["survival"]["mass"][event_slot],
                "direct": event["survival"]["components"]["direct"][event_slot],
                "downstream": event["survival"]["components"]["downstream"][event_slot],
                "trace_error": event["trace_groups"]["rare_critical"]["trace_error"],
            }
        )
    dependent_final = 0.5 * (
        final_groups["dependent_left"]["mse"]
        + final_groups["dependent_right"]["mse"]
    )
    return {
        "rare_initial_mse": initial_groups["rare_critical"]["mse"],
        "rare_final_mse": final_groups["rare_critical"]["mse"],
        "rare_trace_error": trace_groups["rare_critical"]["trace_error"],
        "rare_dominant_slot": rare_slot,
        "rare_slot_survival_mean": sum(item["survival"] for item in rare_history)
        / len(rare_history),
        "rare_slot_survival_min": min(item["survival"] for item in rare_history),
        "rare_slot_direct_consequence_mean": sum(item["direct"] for item in rare_history)
        / len(rare_history),
        "rare_slot_direct_consequence_max": max(item["direct"] for item in rare_history),
        "rare_slot_downstream_consequence_mean": sum(
            item["downstream"] for item in rare_history
        )
        / len(rare_history),
        "rare_slot_downstream_consequence_max": max(
            item["downstream"] for item in rare_history
        ),
        "rare_history": rare_history,
        "dependent_final_mse": dependent_final,
        "frequent_final_mse": final_groups["frequent_local"]["mse"],
        "replacement_final_mse": final_groups["replacement"]["mse"],
        "obsolete_final_mse": final_groups["obsolete"]["mse"],
        "noise_final_mse": final_groups["noise"]["mse"],
        "pending_fraction": output["final"]["pending_fraction"],
        "geometry": output["final"]["geometry"],
    }


def run(args: argparse.Namespace) -> None:
    stages = build_rare_functional_stream(args)
    output = run_with_stages(
        args,
        stages=stages,
        question=(
            "Can measured downstream consequence preserve a rarely observed shared function while "
            "frequent low-impact evidence competes for a fixed trace budget?"
        ),
        scope=(
            "Single-seed tiny synthetic pressure test. Semantic group names construct and evaluate "
            "the stream only; exact leave-one-out updates calibrate the first-order estimate."
        ),
    )
    gate = rare_gate(output)
    output["rare_functional_gate"] = gate
    output_json = args.output_dir / "dependency_utility_survival.json"
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nRARE FUNCTIONAL SURVIVAL GATE")
    print("-" * 148)
    for name, value in gate.items():
        if name == "rare_history":
            continue
        print(f"{name}={value}")
    print(f"rare_history_events={len(gate['rare_history'])}")
    print(f"updated_json={output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = build_survival_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=Path("model/analysis/gco-tiny-rare-functional-survival-seed0"),
    )
    parser.add_argument("--rare-critical-points", type=int, default=2)
    parser.add_argument("--dependent-base-points", type=int, default=12)
    parser.add_argument("--dependent-replay-points", type=int, default=6)
    parser.add_argument("--dependent-final-points", type=int, default=4)
    parser.add_argument("--frequent-base-points", type=int, default=18)
    parser.add_argument("--frequent-replay-points", type=int, default=12)
    parser.add_argument("--rare-obsolete-points", type=int, default=4)
    parser.add_argument("--pressure-stage2-points", type=int, default=6)
    parser.add_argument("--pressure-stage3-points", type=int, default=4)
    parser.add_argument("--replacement-points", type=int, default=12)
    parser.add_argument("--rare-noise-points", type=int, default=6)
    parser.add_argument("--rare-cluster-sigma", type=float, default=0.08)
    parser.add_argument("--rare-center-scale", type=float, default=1.4)
    parser.add_argument("--rare-branch-span", type=float, default=0.75)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
