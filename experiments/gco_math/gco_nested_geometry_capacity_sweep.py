#!/usr/bin/env python3
"""Capacity sweep for manual nested-geometry dynamics.

This script asks a narrower question than the main dynamics sandbox:

    How many bounded slots are needed before the nested geometry can keep
    stable memory, rare critical memory, replacement memory, and all important
    context branches while releasing obsolete and noisy traces?

It writes a CSV, JSON summary, and aggregate plots. It does not hide failed
settings: every failed validation check is emitted in the output table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gco_nested_geometry_dynamics import (  # noqa: E402
    Config,
    final_group_summary,
    run_dynamics,
    slot_table,
    summarize_validation,
    write_csv,
)


GROUPS = (
    "stable",
    "merge_a",
    "merge_b",
    "branch_root",
    "branch_up",
    "branch_down",
    "rare_critical",
    "novel",
    "obsolete_old",
    "replacement",
    "noise",
)


def parse_int_csv(name: str, value: str) -> tuple[int, ...]:
    pieces = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not pieces:
        raise ValueError(f"{name} must contain at least one integer.")
    numbers: list[int] = []
    for piece in pieces:
        try:
            number = int(piece)
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-integer value: {piece!r}.") from exc
        numbers.append(number)
    return tuple(numbers)


def bool_float(value: bool) -> float:
    return 1.0 if value else 0.0


def make_config(args: argparse.Namespace, *, slots: int, seed: int, output_dir: Path) -> Config:
    return Config(
        seed=seed,
        steps=args.steps,
        slots=slots,
        shell_count=args.shell_count,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        match_threshold=args.match_threshold,
        branch_threshold=args.branch_threshold,
        initial_strength=args.initial_strength,
        evidence_gain=args.evidence_gain,
        usefulness_gain=args.usefulness_gain,
        dependency_gain=args.dependency_gain,
        conflict_gain=args.conflict_gain,
        age_decay=args.age_decay,
        outer_learning_rate=args.outer_learning_rate,
        learning_timescale=args.learning_timescale,
        outward_decay=args.outward_decay,
        decay_timescale=args.decay_timescale,
        inward_rate=args.inward_rate,
        outward_rate=args.outward_rate,
        survival_temperature=args.survival_temperature,
        release_threshold=args.release_threshold,
        overwrite_margin=args.overwrite_margin,
        admission_threshold=args.admission_threshold,
        admission_temperature=args.admission_temperature,
        provisional_depth_cap=args.provisional_depth_cap,
        provisional_decay_multiplier=args.provisional_decay_multiplier,
        contradiction_release_threshold=args.contradiction_release_threshold,
        support_threshold=args.support_threshold,
        support_gain=args.support_gain,
        support_min_potential=args.support_min_potential,
        support_decay=args.support_decay,
        support_admission_gain=args.support_admission_gain,
        support_diversity_gain=args.support_diversity_gain,
        max_events_per_step=args.max_events_per_step,
        output_dir=output_dir,
    )


def row_group_lookup(group_rows: list[dict[str, float | str | int]]) -> dict[str, dict[str, float | str | int]]:
    lookup = {str(row["group"]): row for row in group_rows}
    missing = sorted(set(GROUPS) - set(lookup))
    if missing:
        raise RuntimeError(f"Missing group rows: {missing}")
    return lookup


def survives(group_row: dict[str, float | str | int], *, max_error: float = 0.18) -> bool:
    slot = int(group_row["slot"])
    if slot < 0:
        return False
    error = float(group_row["trace_error"])
    return math.isfinite(error) and error < max_error


def run_one(args: argparse.Namespace, *, slots: int, seed: int, output_dir: Path) -> dict[str, object]:
    config = make_config(args, slots=slots, seed=seed, output_dir=output_dir)
    slot_states, event_rows, metric_rows, prototypes = run_dynamics(config)
    group_rows = final_group_summary(slot_states, event_rows, prototypes, config)
    slot_rows = slot_table(slot_states, config)
    validation = summarize_validation(group_rows, metric_rows)
    group_lookup = row_group_lookup(group_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "group_summary.csv", group_rows)
    write_csv(output_dir / "slot_table.csv", slot_rows)
    write_csv(output_dir / "metrics_over_time.csv", metric_rows)

    important_groups = ("stable", "rare_critical", "replacement", "branch_root", "branch_up", "branch_down")
    important_survival = all(survives(group_lookup[group]) for group in important_groups)
    obsolete_released = int(group_lookup["obsolete_old"]["slot"]) < 0
    noise_row = group_lookup["noise"]
    noise_released_or_weak = (
        int(noise_row["slot"]) < 0
        or float(noise_row["group_share_in_slot"]) < 0.5
        or float(noise_row["strength"]) < 2.5
    )
    all_strict = (
        important_survival
        and obsolete_released
        and noise_released_or_weak
        and bool(validation["middle_layer_used"])
        and bool(validation["outer_geometry_still_available"])
    )

    final_metrics = metric_rows[-1]
    row: dict[str, object] = {
        "slots": slots,
        "seed": seed,
        "all_strict": all_strict,
        "important_survival": important_survival,
        "obsolete_released": obsolete_released,
        "noise_released_or_weak": noise_released_or_weak,
        "middle_layer_used": bool(validation["middle_layer_used"]),
        "outer_available": bool(validation["outer_geometry_still_available"]),
        "active_slots": final_metrics["active_slots"],
        "outer_mass": final_metrics["outer_mass"],
        "middle_mass": final_metrics["middle_mass"],
        "inner_mass": final_metrics["inner_mass"],
        "mean_depth": final_metrics["mean_depth"],
        "mean_trace_error": final_metrics["mean_trace_error"],
        "validation": validation,
        "groups": group_rows,
        "config": {**asdict(config), "output_dir": str(output_dir)},
    }
    for group in GROUPS:
        group_row = group_lookup[group]
        row[f"{group}_slot"] = int(group_row["slot"])
        row[f"{group}_survives"] = survives(group_row)
        row[f"{group}_depth"] = float(group_row["depth"])
        row[f"{group}_error"] = float(group_row["trace_error"]) if math.isfinite(float(group_row["trace_error"])) else math.nan
    return row


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty sweep CSV.")
    excluded = {"validation", "groups", "config"}
    fieldnames = [key for key in rows[0].keys() if key not in excluded]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def aggregate_by_slots(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    slots_values = sorted({int(row["slots"]) for row in rows})
    aggregate: list[dict[str, float]] = []
    for slots in slots_values:
        slot_rows = [row for row in rows if int(row["slots"]) == slots]
        if not slot_rows:
            raise RuntimeError(f"No rows for slot count {slots}.")
        aggregate.append(
            {
                "slots": float(slots),
                "all_strict_rate": float(np.mean([bool_float(bool(row["all_strict"])) for row in slot_rows])),
                "important_survival_rate": float(np.mean([bool_float(bool(row["important_survival"])) for row in slot_rows])),
                "obsolete_released_rate": float(np.mean([bool_float(bool(row["obsolete_released"])) for row in slot_rows])),
                "noise_released_or_weak_rate": float(np.mean([bool_float(bool(row["noise_released_or_weak"])) for row in slot_rows])),
                "middle_layer_used_rate": float(np.mean([bool_float(bool(row["middle_layer_used"])) for row in slot_rows])),
                "outer_available_rate": float(np.mean([bool_float(bool(row["outer_available"])) for row in slot_rows])),
                "mean_error": float(np.nanmean([float(row["mean_trace_error"]) for row in slot_rows])),
                "mean_depth": float(np.mean([float(row["mean_depth"]) for row in slot_rows])),
                "outer_mass": float(np.mean([float(row["outer_mass"]) for row in slot_rows])),
                "middle_mass": float(np.mean([float(row["middle_mass"]) for row in slot_rows])),
                "inner_mass": float(np.mean([float(row["inner_mass"]) for row in slot_rows])),
            }
        )
    return aggregate


def plot_capacity_rates(aggregate: list[dict[str, float]], output_path: Path) -> None:
    slots = np.array([row["slots"] for row in aggregate])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for key, label in (
        ("all_strict_rate", "all strict"),
        ("important_survival_rate", "important survives"),
        ("middle_layer_used_rate", "middle used"),
        ("obsolete_released_rate", "obsolete released"),
        ("noise_released_or_weak_rate", "noise released/weak"),
    ):
        ax.plot(slots, [row[key] for row in aggregate], marker="o", label=label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("available slots")
    ax.set_ylabel("pass rate across seeds")
    ax.set_title("Nested geometry capacity boundary")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_layer_usage(aggregate: list[dict[str, float]], output_path: Path) -> None:
    slots = np.array([row["slots"] for row in aggregate])
    outer = np.array([row["outer_mass"] for row in aggregate])
    middle = np.array([row["middle_mass"] for row in aggregate])
    inner = np.array([row["inner_mass"] for row in aggregate])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.bar(slots, outer, label="outer")
    ax.bar(slots, middle, bottom=outer, label="middle")
    ax.bar(slots, inner, bottom=outer + middle, label="inner")
    ax.set_xlabel("available slots")
    ax.set_ylabel("final slot fraction")
    ax.set_title("Final layer usage under capacity pressure")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_survival(rows: list[dict[str, object]], output_path: Path) -> None:
    slots_values = sorted({int(row["slots"]) for row in rows})
    matrix = np.zeros((len(GROUPS), len(slots_values)), dtype=np.float64)
    for col, slots in enumerate(slots_values):
        slot_rows = [row for row in rows if int(row["slots"]) == slots]
        for row_index, group in enumerate(GROUPS):
            matrix[row_index, col] = float(np.mean([bool_float(bool(row[f"{group}_survives"])) for row in slot_rows]))
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(slots_values)))
    ax.set_xticklabels([str(slot) for slot in slots_values])
    ax.set_yticks(np.arange(len(GROUPS)))
    ax.set_yticklabels(GROUPS)
    ax.set_xlabel("available slots")
    ax.set_title("Final group survival rate")
    fig.colorbar(im, ax=ax, label="survival rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_report(aggregate: list[dict[str, float]], rows: list[dict[str, object]], output_dir: Path) -> None:
    strict_slots = [int(row["slots"]) for row in aggregate if row["all_strict_rate"] >= 1.0]
    boundary = min(strict_slots) if strict_slots else None
    print("\nNESTED-GEOMETRY CAPACITY SWEEP")
    print("=" * 132)
    print(f"runs={len(rows)} minimum_strict_slots={boundary if boundary is not None else 'none'}")
    print("-" * 132)
    print(
        f"{'slots':>7} {'all':>7} {'important':>10} {'middle':>8} {'outer':>7} "
        f"{'obsolete':>9} {'noise':>7} {'err':>9} {'depth':>9} {'outer/mid/inner':>21}"
    )
    for row in aggregate:
        print(
            f"{int(row['slots']):7d} {row['all_strict_rate']:7.3f} "
            f"{row['important_survival_rate']:10.3f} {row['middle_layer_used_rate']:8.3f} "
            f"{row['outer_available_rate']:7.3f} {row['obsolete_released_rate']:9.3f} "
            f"{row['noise_released_or_weak_rate']:7.3f} {row['mean_error']:9.4f} "
            f"{row['mean_depth']:9.4f} "
            f"{row['outer_mass']:.2f}/{row['middle_mass']:.2f}/{row['inner_mass']:.2f}"
        )
    print("\nWROTE")
    print("-" * 132)
    for name in (
        "capacity_sweep_results.csv",
        "capacity_sweep_summary.json",
        "capacity_boundary.png",
        "capacity_layer_usage.png",
        "capacity_group_survival.png",
    ):
        print(output_dir / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots-list", type=str, default="5,6,7,8,9")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--inner-radius", type=float, default=0.85)
    parser.add_argument("--outer-radius", type=float, default=3.0)
    parser.add_argument("--match-threshold", type=float, default=0.10)
    parser.add_argument("--branch-threshold", type=float, default=0.22)
    parser.add_argument("--initial-strength", type=float, default=0.55)
    parser.add_argument("--evidence-gain", type=float, default=0.75)
    parser.add_argument("--usefulness-gain", type=float, default=1.20)
    parser.add_argument("--dependency-gain", type=float, default=1.25)
    parser.add_argument("--conflict-gain", type=float, default=1.85)
    parser.add_argument("--age-decay", type=float, default=0.012)
    parser.add_argument("--outer-learning-rate", type=float, default=0.42)
    parser.add_argument("--learning-timescale", type=float, default=0.62)
    parser.add_argument("--outward-decay", type=float, default=0.055)
    parser.add_argument("--decay-timescale", type=float, default=0.72)
    parser.add_argument("--inward-rate", type=float, default=0.07)
    parser.add_argument("--outward-rate", type=float, default=0.16)
    parser.add_argument("--survival-temperature", type=float, default=0.35)
    parser.add_argument("--release-threshold", type=float, default=-0.65)
    parser.add_argument("--overwrite-margin", type=float, default=0.18)
    parser.add_argument("--admission-threshold", type=float, default=1.85)
    parser.add_argument("--admission-temperature", type=float, default=0.35)
    parser.add_argument("--provisional-depth-cap", type=float, default=0.85)
    parser.add_argument("--provisional-decay-multiplier", type=float, default=2.50)
    parser.add_argument("--contradiction-release-threshold", type=float, default=1.15)
    parser.add_argument("--support-threshold", type=float, default=0.16)
    parser.add_argument("--support-gain", type=float, default=0.45)
    parser.add_argument("--support-min-potential", type=float, default=1.00)
    parser.add_argument("--support-decay", type=float, default=0.985)
    parser.add_argument("--support-admission-gain", type=float, default=1.20)
    parser.add_argument("--support-diversity-gain", type=float, default=0.70)
    parser.add_argument("--max-events-per-step", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/05_nested_geometry/results/gco-nested-geometry-capacity-sweep"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    slots_list = parse_int_csv("slots-list", args.slots_list)
    seeds = parse_int_csv("seeds", args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for slots in slots_list:
        for seed in seeds:
            run_dir = args.output_dir / f"slots{slots}-seed{seed}"
            rows.append(run_one(args, slots=slots, seed=seed, output_dir=run_dir))

    aggregate = aggregate_by_slots(rows)
    write_rows_csv(args.output_dir / "capacity_sweep_results.csv", rows)
    with (args.output_dir / "capacity_sweep_summary.json").open("w") as handle:
        json.dump({"runs": rows, "aggregate": aggregate}, handle, indent=2)
    plot_capacity_rates(aggregate, args.output_dir / "capacity_boundary.png")
    plot_layer_usage(aggregate, args.output_dir / "capacity_layer_usage.png")
    plot_group_survival(rows, args.output_dir / "capacity_group_survival.png")
    print_report(aggregate, rows, args.output_dir)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
