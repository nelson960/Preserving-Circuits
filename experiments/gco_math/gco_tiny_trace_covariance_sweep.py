"""Sweep scalable covariance sketches for bounded recurrent trace memory.

The experiment compares full covariance, diagonal variance, and diagonal plus
low-rank factors at ranks requested on the command line. Every recurrent
condition uses the same functional attention, five trace slots, stream, and
description-length objective. A full-history condition is the fidelity
reference and a current-only condition verifies that the final no-rehearsal
stage genuinely requires memory.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_tiny_recurrent_trace_field import (
    build_parser as build_recurrent_parser,
    centered_kernel_alignment,
    evaluate_centers,
    prepare_stream,
    run_condition,
    structure_checks,
    validate_args,
)


RETAINED_GROUPS = ("merge_a", "merge_b", "stable", "branch_root", "branch_up", "branch_down")


def retained_error(report: dict[str, Any]) -> float:
    return sum(report["groups"][group]["mean_squared_error"] for group in RETAINED_GROUPS) / len(
        RETAINED_GROUPS
    )


def variant_namespace(args: argparse.Namespace, *, mode: str, rank: int) -> argparse.Namespace:
    variant = copy.deepcopy(args)
    variant.summary_covariance = mode
    variant.summary_rank = rank
    return variant


def validate_sweep_args(args: argparse.Namespace) -> None:
    validate_args(args)
    if not args.ranks:
        raise ValueError("--ranks must contain at least one rank.")
    if len(set(args.ranks)) != len(args.ranks):
        raise ValueError(f"--ranks contains duplicates: {args.ranks}.")
    for rank in args.ranks:
        if not 1 <= rank <= args.d_model:
            raise ValueError(f"Every sweep rank must be in [1, {args.d_model}], got {rank}.")


def plot_pareto(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 5.2))
    for index, row in enumerate(rows):
        axis.scatter(
            row["persistent_scalars"],
            row["old_error_ratio_vs_full"],
            s=90,
        )
        axis.annotate(
            row["label"],
            (row["persistent_scalars"], row["old_error_ratio_vs_full"]),
            xytext=(0, 8 + 13 * (index % 2)),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.axhline(1.0, color="#222222", linestyle="--", linewidth=1.2, label="full-history error")
    axis.set_xlabel("persistent summary scalars")
    axis.set_ylabel("retained-old error / full-history error")
    axis.set_title("Trace-memory fidelity versus storage")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fidelity(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    labels = [row["label"] for row in rows]
    x = torch.arange(len(rows), dtype=torch.float32).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].bar(x, [row["attention_cka_vs_full"] for row in rows], color="#1f77b4")
    axes[0].set_ylim(max(0.0, min(row["attention_cka_vs_full"] for row in rows) - 0.02), 1.001)
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("attention-code CKA")
    axes[0].set_title("Geometry fidelity")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [row["overall_mse"] for row in rows], color="#ff7f0e")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("mean reconstruction error")
    axes[1].set_title("Behavioral reconstruction")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_error_heatmap(
    *,
    reports: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    labels = list(reports)
    groups = sorted(reports[labels[0]]["groups"])
    values = torch.tensor(
        [
            [reports[label]["groups"][group]["mean_squared_error"] for group in groups]
            for label in labels
        ],
        dtype=torch.float32,
    )
    log_values = torch.log10(values.clamp_min(torch.finfo(values.dtype).eps))
    fig, axis = plt.subplots(figsize=(12.0, 0.7 * len(labels) + 2.8))
    image = axis.imshow(log_values.numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Final group reconstruction error (log10 scale)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{float(values[row, column]):.2g}",
                ha="center",
                va="center",
                color="white" if float(log_values[row, column]) < 0.7 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, label="log10 mean squared error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_sweep_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    stages = prepare_stream(args)
    accumulated = [point for stage in stages for point in stage]

    print("TINY TRACE COVARIANCE SWEEP")
    print("=" * 144)
    print(
        f"device={device} slots={args.num_slots} d_model={args.d_model} ranks={args.ranks} "
        f"steps={args.trace_steps} restarts={args.restarts}"
    )

    full_reports, full_centers, _ = run_condition(
        mode="full_history",
        stages=stages,
        args=args,
        device=device,
    )
    current_reports, _current_centers, _ = run_condition(
        mode="current_only",
        stages=stages,
        args=args,
        device=device,
    )
    full_final = full_reports[-1]
    current_final = current_reports[-1]
    full_error = retained_error(full_final)

    full_eval, _ = evaluate_centers(
        accumulated,
        full_centers[-1],
        args=args,
        device=device,
    )

    variants: list[tuple[str, str, int]] = [
        ("full covariance", "full", args.summary_rank),
        ("diagonal", "diagonal", args.summary_rank),
        ("variance trace", "trace", args.summary_rank),
    ]
    variants.extend((f"rank {rank}", "lowrank", rank) for rank in args.ranks)

    rows: list[dict[str, Any]] = []
    reports_for_plot: dict[str, dict[str, Any]] = {
        "full history": full_final,
    }
    reports_by_variant: dict[str, list[dict[str, Any]]] = {}
    checks_by_variant: dict[str, dict[str, Any]] = {}

    for label, mode, rank in variants:
        variant_args = variant_namespace(args, mode=mode, rank=rank)
        reports, centers, _summary = run_condition(
            mode="recurrent_summary",
            stages=stages,
            args=variant_args,
            device=device,
        )
        final = reports[-1]
        recurrent_eval, _ = evaluate_centers(
            accumulated,
            centers[-1],
            args=variant_args,
            device=device,
        )
        row = {
            "label": label,
            "covariance_mode": mode,
            "rank": rank if mode == "lowrank" else None,
            "persistent_scalars": final["persistent_scalars_after_stage"],
            "overall_mse": final["evaluation_mean_squared_error"],
            "retained_old_error": retained_error(final),
            "old_error_ratio_vs_full": retained_error(final) / full_error,
            "attention_cka_vs_full": centered_kernel_alignment(
                full_eval.attention,
                recurrent_eval.attention,
            ),
        }
        rows.append(row)
        reports_by_variant[label] = reports
        checks_by_variant[label] = structure_checks(final["groups"])
        reports_for_plot[label] = final

    reports_for_plot["current only"] = current_final

    print("\nCOVARIANCE COMPRESSION SUMMARY")
    print("-" * 144)
    print(
        f"{'summary':>20} {'scalars':>9} {'MSE':>10} {'oldErr':>10} "
        f"{'old/full':>10} {'CKA':>10} {'allChecks':>10}"
    )
    for row in rows:
        boolean_checks = [
            value for value in checks_by_variant[row["label"]].values() if isinstance(value, bool)
        ]
        all_checks = all(boolean_checks)
        print(
            f"{row['label']:>20} {row['persistent_scalars']:9d} {row['overall_mse']:10.4f} "
            f"{row['retained_old_error']:10.4f} {row['old_error_ratio_vs_full']:10.4f} "
            f"{row['attention_cka_vs_full']:10.4f} {str(all_checks):>10}"
        )
    print(
        f"\nfull_history scalars={full_final['persistent_scalars_after_stage']} "
        f"old_error={full_error:.4f} mse={full_final['evaluation_mean_squared_error']:.4f}"
    )
    print(
        f"current_only old_error={retained_error(current_final):.4f} "
        f"mse={current_final['evaluation_mean_squared_error']:.4f}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pareto_path = args.output_dir / "covariance_storage_fidelity.png"
    fidelity_path = args.output_dir / "covariance_geometry_fidelity.png"
    heatmap_path = args.output_dir / "covariance_group_errors.png"
    output_json = args.output_dir / "trace_covariance_sweep.json"
    plot_pareto(rows, output_path=pareto_path)
    plot_fidelity(rows, output_path=fidelity_path)
    plot_group_error_heatmap(reports=reports_for_plot, output_path=heatmap_path)

    output = {
        "question": (
            "How much covariance structure must each recurrent trace retain to preserve full-history "
            "organization with storage that scales linearly in model width?"
        ),
        "scope": (
            "Synthetic representation stream with no old-group rehearsal in the final stage. "
            "Low-rank factors are computed by explicit power iteration without eigendecomposition."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stream": [[asdict(point) for point in stage] for stage in stages],
        "full_history": full_reports,
        "current_only": current_reports,
        "variants": rows,
        "variant_reports": reports_by_variant,
        "variant_checks": checks_by_variant,
        "plots": {
            "storage_fidelity": str(pareto_path),
            "geometry_fidelity": str(fidelity_path),
            "group_errors": str(heatmap_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_plots={pareto_path},{fidelity_path},{heatmap_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = build_recurrent_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=Path("model/analysis/gco-tiny-trace-covariance-sweep-seed0"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
