#!/usr/bin/env python3
"""Capacity and semanticity stress test for nested internal branches.

This driver runs the internal-branch nested optimizer with different numbers of
child branches per outer region. The goal is not only classification accuracy.
It measures whether the nested geometry has semantic structure:

* compatible variants should merge into the same child branch;
* contextual branches should remain separable;
* rare critical traces should survive despite low frequency;
* replacement evidence should beat obsolete evidence;
* noise should remain low-confidence and weakly consolidated;
* updates must not leak into unrelated outer regions or sibling child heads.
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

from gco_nested_branching_optimizer_tiny_nn import Config, run_sequence, write_outputs  # noqa: E402


SEMANTIC_GROUPS = {
    "compatible_pair": ("merge_a", "merge_b"),
    "context_branches": ("branch_root", "branch_up", "branch_down"),
    "protected": ("stable", "merge_a", "merge_b", "branch_root", "branch_up", "branch_down", "rare_critical"),
}


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list.") from exc
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return result


def exact_float(row: dict[str, float | str | int]) -> float:
    return float(row["correct"])


def slot_of(row: dict[str, float | str | int]) -> tuple[int, int]:
    return int(row["region"]), int(row["child"])


def group_row(rows: list[dict[str, float | str | int]], group: str) -> dict[str, float | str | int]:
    for row in rows:
        if row["group"] == group:
            return row
    raise RuntimeError(f"Missing evaluation row for group {group!r}.")


def semantic_metrics(
    eval_rows: list[dict[str, float | str | int]],
    child_rows: list[dict[str, float | str | int]],
    eval_summary: dict[str, float],
    children_per_region: int,
    noise_conf_threshold: float,
) -> dict[str, float | int | str]:
    merge_a = group_row(eval_rows, "merge_a")
    merge_b = group_row(eval_rows, "merge_b")
    branch_rows = [group_row(eval_rows, group) for group in SEMANTIC_GROUPS["context_branches"]]
    rare = group_row(eval_rows, "rare_critical")
    stable = group_row(eval_rows, "stable")
    replacement = group_row(eval_rows, "replacement")
    obsolete = group_row(eval_rows, "obsolete_old")
    noise = group_row(eval_rows, "noise")

    compatible_same_slot = float(slot_of(merge_a) == slot_of(merge_b))
    compatible_correct = float(exact_float(merge_a) == 1.0 and exact_float(merge_b) == 1.0)
    compatible_merge = float(compatible_same_slot == 1.0 and compatible_correct == 1.0)

    branch_slots = {slot_of(row) for row in branch_rows}
    branch_distinct_count = len(branch_slots)
    max_distinct_under_capacity = len(branch_rows)
    if max_distinct_under_capacity <= 0:
        raise RuntimeError("Invalid branch capacity denominator.")
    branch_capacity_separation = float(branch_distinct_count / max_distinct_under_capacity)
    branch_full_separation = float(branch_distinct_count == len(branch_rows))
    branch_correct = float(np.mean([exact_float(row) for row in branch_rows]))

    rare_survival = float(exact_float(rare) == 1.0)
    stable_survival = float(exact_float(stable) == 1.0)
    replacement_beats_obsolete = float(exact_float(replacement) == 1.0 and exact_float(obsolete) == 0.0)
    noise_rejected = float(float(noise["confidence"]) <= noise_conf_threshold)

    child_by_slot = {
        (int(row["region"]), int(row["child"])): row
        for row in child_rows
        if int(row["active"]) == 1
    }
    rare_child = child_by_slot.get(slot_of(rare))
    noise_child = child_by_slot.get(slot_of(noise))
    if rare_child is None:
        raise RuntimeError("Rare critical evaluation points to a missing child branch.")
    if noise_child is None:
        raise RuntimeError("Noise evaluation points to a missing child branch.")
    rare_survival_energy = float(rare_child["survival"])
    noise_survival_energy = float(noise_child["survival"])
    rare_beats_noise_energy = float(rare_survival_energy > noise_survival_energy)

    leakage_clean = float(eval_summary["max_cross_region_delta"] == 0.0 and eval_summary["max_cross_child_delta"] == 0.0)
    semantic_score_parts = [
        compatible_merge,
        branch_correct,
        branch_capacity_separation,
        rare_survival,
        stable_survival,
        replacement_beats_obsolete,
        noise_rejected,
        rare_beats_noise_energy,
    ]
    semantic_score = float(np.mean(semantic_score_parts))
    strict_semantic_score_parts = [
        compatible_merge,
        branch_correct,
        branch_full_separation,
        rare_survival,
        stable_survival,
        replacement_beats_obsolete,
        noise_rejected,
        rare_beats_noise_energy,
    ]
    strict_semantic_score = float(np.mean(strict_semantic_score_parts))
    return {
        "children_per_region": children_per_region,
        "protected_acc": float(eval_summary["protected_acc"]),
        "branch_acc": float(eval_summary["branch_acc"]),
        "compatible_same_slot": compatible_same_slot,
        "compatible_correct": compatible_correct,
        "compatible_merge": compatible_merge,
        "branch_distinct_count": branch_distinct_count,
        "branch_capacity_separation": branch_capacity_separation,
        "branch_full_separation": branch_full_separation,
        "branch_correct": branch_correct,
        "rare_survival": rare_survival,
        "rare_survival_energy": rare_survival_energy,
        "stable_survival": stable_survival,
        "replacement_beats_obsolete": replacement_beats_obsolete,
        "noise_rejected": noise_rejected,
        "noise_confidence": float(noise["confidence"]),
        "noise_survival_energy": noise_survival_energy,
        "rare_beats_noise_energy": rare_beats_noise_energy,
        "max_cross_region_delta": float(eval_summary["max_cross_region_delta"]),
        "max_cross_child_delta": float(eval_summary["max_cross_child_delta"]),
        "leakage_clean": leakage_clean,
        "semantic_score": semantic_score,
        "strict_semantic_score": strict_semantic_score,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_stress(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    xs = np.array([int(row["children_per_region"]) for row in rows], dtype=np.int64)
    semantic_score = np.array([float(row["semantic_score"]) for row in rows], dtype=np.float64)
    strict_semantic_score = np.array([float(row["strict_semantic_score"]) for row in rows], dtype=np.float64)
    protected = np.array([float(row["protected_acc"]) for row in rows], dtype=np.float64)
    branch = np.array([float(row["branch_acc"]) for row in rows], dtype=np.float64)
    merge = np.array([float(row["compatible_merge"]) for row in rows], dtype=np.float64)
    rare = np.array([float(row["rare_survival"]) for row in rows], dtype=np.float64)
    noise = np.array([float(row["noise_confidence"]) for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    axes[0].plot(xs, semantic_score, marker="o", label="semantic score")
    axes[0].plot(xs, strict_semantic_score, marker="o", label="strict semantic score")
    axes[0].plot(xs, protected, marker="o", label="protected accuracy")
    axes[0].plot(xs, branch, marker="o", label="branch accuracy")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("score")
    axes[0].legend()
    axes[1].plot(xs, merge, marker="o", label="compatible merge")
    axes[1].plot(xs, rare, marker="o", label="rare survival")
    axes[1].plot(xs, noise, marker="o", label="noise confidence")
    axes[1].set_xlabel("child branches per outer region")
    axes[1].set_ylabel("semantic checks")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_config(args: argparse.Namespace, children_per_region: int, seed: int, run_dir: Path) -> Config:
    return Config(
        seed=seed,
        scenario=args.scenario,
        mode=args.mode,
        steps=args.steps,
        regions=args.regions,
        children_per_region=children_per_region,
        d_input=args.d_input,
        hidden=args.hidden,
        classes=args.classes,
        inner_steps=args.inner_steps,
        memory_limit=args.memory_limit,
        shell_count=args.shell_count,
        base_lr=args.base_lr,
        trunk_lr_multiplier=args.trunk_lr_multiplier,
        child_lr_multiplier=args.child_lr_multiplier,
        depth_lr_decay=args.depth_lr_decay,
        strength_lr_decay=args.strength_lr_decay,
        center_lr=args.center_lr,
        child_center_lr=args.child_center_lr,
        match_threshold=args.match_threshold,
        branch_threshold=args.branch_threshold,
        child_match_threshold=args.child_match_threshold,
        child_branch_threshold=args.child_branch_threshold,
        evidence_gain=args.evidence_gain,
        usefulness_gain=args.usefulness_gain,
        dependency_gain=args.dependency_gain,
        conflict_gain=args.conflict_gain,
        age_decay=args.age_decay,
        support_gain=args.support_gain,
        support_decay=args.support_decay,
        support_threshold=args.support_threshold,
        min_consolidation_potential=args.min_consolidation_potential,
        admission_threshold=args.admission_threshold,
        admission_temperature=args.admission_temperature,
        inward_rate=args.inward_rate,
        outward_rate=args.outward_rate,
        survival_temperature=args.survival_temperature,
        provisional_depth_cap=args.provisional_depth_cap,
        low_potential_depth_cap=args.low_potential_depth_cap,
        low_potential_strength_cap=args.low_potential_strength_cap,
        release_threshold=args.release_threshold,
        overwrite_margin=args.overwrite_margin,
        tangent_damping=args.tangent_damping,
        restore_weight=args.restore_weight,
        restore_clip_ratio=args.restore_clip_ratio,
        protected_min_potential=args.protected_min_potential,
        protected_require_admitted=args.protected_require_admitted,
        protect_same_label=args.protect_same_label,
        sibling_protect=not args.no_sibling_protect,
        max_events_per_step=args.max_events_per_step,
        device=args.device,
        projection_device=args.projection_device,
        output_dir=run_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--children", type=str, default="1,2,3,4")
    parser.add_argument("--scenario", choices=("default", "long"), default="long")
    parser.add_argument("--mode", choices=("nested_sgd", "nested_tangent"), default="nested_tangent")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--regions", type=int, default=5)
    parser.add_argument("--d-input", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--memory-limit", type=int, default=32)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=0.12)
    parser.add_argument("--trunk-lr-multiplier", type=float, default=0.12)
    parser.add_argument("--child-lr-multiplier", type=float, default=1.00)
    parser.add_argument("--depth-lr-decay", type=float, default=0.65)
    parser.add_argument("--strength-lr-decay", type=float, default=0.05)
    parser.add_argument("--center-lr", type=float, default=0.12)
    parser.add_argument("--child-center-lr", type=float, default=0.18)
    parser.add_argument("--match-threshold", type=float, default=0.20)
    parser.add_argument("--branch-threshold", type=float, default=0.42)
    parser.add_argument("--child-match-threshold", type=float, default=0.08)
    parser.add_argument("--child-branch-threshold", type=float, default=0.16)
    parser.add_argument("--evidence-gain", type=float, default=0.75)
    parser.add_argument("--usefulness-gain", type=float, default=1.20)
    parser.add_argument("--dependency-gain", type=float, default=1.25)
    parser.add_argument("--conflict-gain", type=float, default=1.85)
    parser.add_argument("--age-decay", type=float, default=0.012)
    parser.add_argument("--support-gain", type=float, default=0.45)
    parser.add_argument("--support-decay", type=float, default=0.985)
    parser.add_argument("--support-threshold", type=float, default=0.10)
    parser.add_argument("--min-consolidation-potential", type=float, default=0.85)
    parser.add_argument("--admission-threshold", type=float, default=1.85)
    parser.add_argument("--admission-temperature", type=float, default=0.35)
    parser.add_argument("--inward-rate", type=float, default=0.025)
    parser.add_argument("--outward-rate", type=float, default=0.16)
    parser.add_argument("--survival-temperature", type=float, default=0.35)
    parser.add_argument("--provisional-depth-cap", type=float, default=0.85)
    parser.add_argument("--low-potential-depth-cap", type=float, default=0.35)
    parser.add_argument("--low-potential-strength-cap", type=float, default=1.25)
    parser.add_argument("--release-threshold", type=float, default=-0.65)
    parser.add_argument("--overwrite-margin", type=float, default=0.18)
    parser.add_argument("--tangent-damping", type=float, default=1e-3)
    parser.add_argument("--restore-weight", type=float, default=0.20)
    parser.add_argument("--restore-clip-ratio", type=float, default=0.35)
    parser.add_argument("--protected-min-potential", type=float, default=1.00)
    parser.add_argument("--protected-require-admitted", action="store_true", default=False)
    parser.add_argument("--protect-same-label", action="store_true", default=False)
    parser.add_argument("--no-sibling-protect", action="store_true", default=False)
    parser.add_argument("--max-events-per-step", type=int, default=4)
    parser.add_argument("--noise-conf-threshold", type=float, default=0.35)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--projection-device", choices=("cpu", "same"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/06_nested_tangent_optimizer/results/gco-nested-branch-semantic-capacity-stress-seed0"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    seeds = parse_int_list(args.seeds, "seeds")
    child_counts = parse_int_list(args.children, "children")
    if any(value < 1 for value in child_counts):
        raise ValueError("children values must all be >= 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for children_per_region in child_counts:
            run_dir = args.output_dir / f"seed{seed}-children{children_per_region}"
            config = build_config(args, children_per_region, seed, run_dir)
            model, event_rows, metric_rows, eval_rows, eval_summary = run_sequence(config)
            summary = write_outputs(model, event_rows, metric_rows, eval_rows, eval_summary, config)
            child_rows = summary["children"]
            if not isinstance(child_rows, list):
                raise RuntimeError("Malformed summary: children table is missing.")
            metrics = semantic_metrics(
                eval_rows=eval_rows,
                child_rows=child_rows,
                eval_summary=eval_summary,
                children_per_region=children_per_region,
                noise_conf_threshold=args.noise_conf_threshold,
            )
            rows.append({"seed": seed, **metrics, "run_dir": str(run_dir)})

    stress_csv = args.output_dir / "semantic_capacity_stress.csv"
    stress_json = args.output_dir / "semantic_capacity_stress.json"
    stress_plot = args.output_dir / "semantic_capacity_stress.png"
    write_csv(stress_csv, rows)
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with stress_json.open("w") as handle:
        json.dump({"config": serializable_config, "rows": rows}, handle, indent=2)
    plot_stress(rows, stress_plot)

    print("\nNESTED BRANCH SEMANTIC CAPACITY STRESS")
    print("=" * 152)
    print(
        f"{'seed':>5} {'children':>8} {'semantic':>9} {'strict':>8} {'protected':>9} "
        f"{'branch':>8} {'merge':>7} {'branchSep':>9} {'rare':>6} {'replace':>8} {'noise':>7} {'leak':>6}"
    )
    for row in rows:
        print(
            f"{int(row['seed']):5d} {int(row['children_per_region']):8d} "
            f"{float(row['semantic_score']):9.4f} {float(row['strict_semantic_score']):8.4f} "
            f"{float(row['protected_acc']):9.4f} {float(row['branch_acc']):8.4f} {float(row['compatible_merge']):7.4f} "
            f"{float(row['branch_capacity_separation']):9.4f} {float(row['rare_survival']):6.4f} "
            f"{float(row['replacement_beats_obsolete']):8.4f} {float(row['noise_confidence']):7.4f} "
            f"{float(row['leakage_clean']):6.4f}"
        )
    print("\nWROTE")
    print("-" * 152)
    print(stress_csv)
    print(stress_json)
    print(stress_plot)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
