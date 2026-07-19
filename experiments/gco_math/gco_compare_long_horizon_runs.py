"""Compare completed long-horizon CL runs produced by the shared harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_run(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Long-horizon result does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"config", "cycles", "updates", "fixed_budgets", "update_operator"}
    missing = required.difference(value)
    if missing:
        raise RuntimeError(f"Result {path} is missing fields: {sorted(missing)}")
    if not value["cycles"] or not value["updates"]:
        raise RuntimeError(f"Result {path} has no completed cycles or updates.")
    return value


def run_label(value: dict[str, Any]) -> str:
    operator = value["update_operator"]
    if operator == "unified":
        operator = f"unified:{value['config']['constraint_mode']}"
    return operator


def validate_comparable(runs: list[dict[str, Any]]) -> None:
    if len(runs) < 2:
        raise ValueError("Comparison requires at least two completed runs.")
    keys = (
        "checkpoint",
        "tokenizer_path",
        "book_path",
        "seed",
        "cycles",
        "cycle_book_words",
        "cycle_book_windows",
        "cycle_fact_windows",
        "cycle_correction_count",
        "cycle_novel_start",
        "cycle_novel_count",
        "rare_fact_index",
        "rare_confirmation_period",
        "misinformation_source_index",
        "misinformation_donor_offset",
        "misinformation_variants",
        "trace_slots",
        "pending_slots",
        "guard_windows",
        "cl_epochs",
        "cl_lr",
    )
    reference = runs[0]["config"]
    for value in runs[1:]:
        mismatches = {
            key: (reference[key], value["config"][key])
            for key in keys
            if reference[key] != value["config"][key]
        }
        if mismatches:
            raise ValueError(
                f"Runs {run_label(runs[0])!r} and {run_label(value)!r} are not comparable: "
                f"{mismatches}"
            )
    labels = [run_label(value) for value in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Comparison contains duplicate method labels: {labels}")


def summarize(value: dict[str, Any]) -> dict[str, Any]:
    final = value["cycles"][-1]
    updates = value["updates"]
    return {
        "method": run_label(value),
        "cycles": len(value["cycles"]),
        "updates": len(updates),
        "accepted_fraction": sum(int(update["accepted"]) for update in updates)
        / len(updates),
        "guard_loss": final["evaluation"]["stable_guard"]["loss"],
        "rare_loss_ratio": final["rare_retention_loss_ratio"],
        "misinformation_truth_loss": final["evaluation"]["misinformation_truth"]["loss"],
        "misinformation_false_margin": final["misinformation_false_preference"][
            "new_minus_old_margin"
        ],
        "current_correction_margin": final["correction_preference"][
            "new_minus_old_margin"
        ],
        "archived_correction_margin": final["archived_preference"][
            "new_minus_old_margin"
        ],
        "original_min_cka": final["original_min_cka"],
        "mean_executed_step_fraction": sum(
            update["safe_grad_fraction"] for update in updates
        )
        / len(updates),
        "mean_constraint_rank": sum(
            update["solver"]["capacity"]["numerical_rank"] for update in updates
        )
        / len(updates),
        "persistent_scalars": final["memory"]["persistent_total_scalars"],
        "seconds": sum(update["seconds"] for update in updates),
    }


def plot_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["method"] for row in rows]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes[0, 0].bar(x, [row["guard_loss"] for row in rows], color="#2563eb")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Final fixed-guard loss")
    axes[0, 1].bar(x, [row["rare_loss_ratio"] for row in rows], color="#7c3aed")
    axes[0, 1].axhline(1.0, color="#111827", linewidth=1)
    axes[0, 1].set_title("Rare-critical loss ratio")
    width = 0.25
    axes[1, 0].bar(
        [value - width for value in x],
        [row["current_correction_margin"] for row in rows],
        width=width,
        label="current",
    )
    axes[1, 0].bar(
        x,
        [row["archived_correction_margin"] for row in rows],
        width=width,
        label="archived",
    )
    axes[1, 0].bar(
        [value + width for value in x],
        [row["misinformation_false_margin"] for row in rows],
        width=width,
        label="false over true",
    )
    axes[1, 0].axhline(0.0, color="#111827", linewidth=1)
    axes[1, 0].set_title("Context and truth margins")
    axes[1, 0].legend()
    axes[1, 1].bar(
        x,
        [row["mean_executed_step_fraction"] for row in rows],
        color="#16a34a",
    )
    axes[1, 1].set_title("Mean executed/raw step fraction")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    runs = [load_run(path) for path in args.inputs]
    validate_comparable(runs)
    rows = [summarize(value) for value in runs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "long_horizon_comparison.json"
    output_plot = args.output_dir / "long_horizon_comparison.png"
    output_json.write_text(json.dumps({"methods": rows}, indent=2), encoding="utf-8")
    plot_summary(rows, output_plot)
    print("LONG-HORIZON CL COMPARISON")
    print("=" * 128)
    print(
        f"{'method':>28} {'guard':>10} {'rare':>10} {'current':>10} "
        f"{'archive':>10} {'false':>10} {'cka':>10} {'accept':>10}"
    )
    for row in rows:
        print(
            f"{row['method']:>28} {row['guard_loss']:10.4g} "
            f"{row['rare_loss_ratio']:10.4f} {row['current_correction_margin']:10.4f} "
            f"{row['archived_correction_margin']:10.4f} "
            f"{row['misinformation_false_margin']:10.4f} "
            f"{row['original_min_cka']:10.4f} {row['accepted_fraction']:10.4f}"
        )
    print(f"wrote_json={output_json}")
    print(f"wrote_plot={output_plot}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
