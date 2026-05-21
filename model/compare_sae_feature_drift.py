from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from discover_sae_concept_features import (
    auroc,
    load_sae,
    parse_csv,
    require_activation_matrix,
    require_input_mean,
    require_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare concept-selective SAE feature activations before and after "
            "a model update, using one fixed reference SAE."
        )
    )
    parser.add_argument("--sae-pt", required=True, type=Path)
    parser.add_argument("--before-activations-pt", required=True, type=Path)
    parser.add_argument("--after-activations-pt", required=True, type=Path)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--negative-labels", required=True)
    parser.add_argument("--feature-indices", required=True, help="Comma-separated SAE feature indices.")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    feature_indices = parse_feature_indices(args.feature_indices)
    negative_labels = parse_csv(args.negative_labels)
    if args.concept in negative_labels:
        raise ValueError("concept must not be included in negative-labels.")

    sae_payload = torch.load(args.sae_pt, map_location="cpu", weights_only=False)
    before_payload = torch.load(args.before_activations_pt, map_location="cpu", weights_only=False)
    after_payload = torch.load(args.after_activations_pt, map_location="cpu", weights_only=False)
    before_activations = require_activation_matrix(before_payload)
    after_activations = require_activation_matrix(after_payload)
    before_rows = require_rows(before_payload)
    after_rows = require_rows(after_payload)
    verify_same_rows(before_rows, after_rows)

    sae = load_sae(sae_payload)
    input_mean = require_input_mean(sae_payload)
    with torch.no_grad():
        _before_hat, before_z = sae(before_activations - input_mean)
        _after_hat, after_z = sae(after_activations - input_mean)
    if before_z.shape != after_z.shape:
        raise ValueError(f"before and after SAE activations differ: {before_z.shape} vs {after_z.shape}.")

    labels = [str(row["label"]) for row in before_rows]
    concept_indices = [idx for idx, label in enumerate(labels) if label == args.concept]
    negative_indices = [idx for idx, label in enumerate(labels) if label in negative_labels]
    if not concept_indices:
        raise RuntimeError(f"no rows found for concept {args.concept!r}.")
    if not negative_indices:
        raise RuntimeError(f"no rows found for negatives {negative_labels}.")

    feature_reports = []
    for feature_index in feature_indices:
        if feature_index < 0 or feature_index >= before_z.shape[1]:
            raise IndexError(f"feature index {feature_index} out of range for feature dim {before_z.shape[1]}.")
        before_metrics = feature_metrics(
            before_z[:, feature_index],
            concept_indices=concept_indices,
            negative_indices=negative_indices,
        )
        after_metrics = feature_metrics(
            after_z[:, feature_index],
            concept_indices=concept_indices,
            negative_indices=negative_indices,
        )
        feature_reports.append(
            {
                "feature_index": feature_index,
                "before": before_metrics,
                "after": after_metrics,
                "deltas": {
                    "concept_mean_delta": after_metrics["concept_mean"] - before_metrics["concept_mean"],
                    "negative_mean_delta": after_metrics["negative_mean"] - before_metrics["negative_mean"],
                    "selectivity_delta": after_metrics["selectivity"] - before_metrics["selectivity"],
                    "auroc_delta": after_metrics["auroc"] - before_metrics["auroc"],
                    "concept_firing_rate_delta": after_metrics["concept_firing_rate"]
                    - before_metrics["concept_firing_rate"],
                    "negative_firing_rate_delta": after_metrics["negative_firing_rate"]
                    - before_metrics["negative_firing_rate"],
                    "fading_ratio": safe_ratio(after_metrics["concept_mean"], before_metrics["concept_mean"]),
                },
            }
        )

    report = {
        "sae_path": str(args.sae_pt),
        "before_activations": str(args.before_activations_pt),
        "after_activations": str(args.after_activations_pt),
        "concept": args.concept,
        "negative_labels": negative_labels,
        "concept_count": len(concept_indices),
        "negative_count": len(negative_indices),
        "feature_indices": feature_indices,
        "features": feature_reports,
        "method": {
            "reference_sae": "same SAE used before and after; this measures feature activation drift without rematching features",
            "selectivity": "mean(z_j | concept) - mean(z_j | negatives)",
            "fading_ratio": "after concept mean activation divided by before concept mean activation",
            "note": "This measures decodable SAE-feature drift. Causal drift requires ablation or patching.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_json": str(args.output_json), "features": feature_reports}, indent=2, sort_keys=True))


def feature_metrics(
    values: torch.Tensor,
    *,
    concept_indices: list[int],
    negative_indices: list[int],
) -> dict[str, float]:
    labels = torch.zeros(values.shape[0], dtype=torch.bool)
    valid = torch.zeros(values.shape[0], dtype=torch.bool)
    labels[concept_indices] = True
    valid[concept_indices] = True
    valid[negative_indices] = True
    concept_values = values[concept_indices]
    negative_values = values[negative_indices]
    threshold = torch.quantile(values, 0.95)
    return {
        "concept_mean": float(concept_values.mean().item()),
        "negative_mean": float(negative_values.mean().item()),
        "selectivity": float(concept_values.mean().item() - negative_values.mean().item()),
        "concept_std": float(concept_values.std(unbiased=False).item()),
        "negative_std": float(negative_values.std(unbiased=False).item()),
        "concept_firing_rate": float((concept_values > threshold).float().mean().item()),
        "negative_firing_rate": float((negative_values > threshold).float().mean().item()),
        "auroc": auroc(values[valid], labels[valid]),
    }


def safe_ratio(after: float, before: float) -> float | str:
    if before == 0.0:
        return "undefined"
    ratio = after / before
    if not math.isfinite(ratio):
        return "undefined"
    return float(ratio)


def parse_feature_indices(value: str) -> list[int]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("feature-indices must contain at least one index.")
    indices: list[int] = []
    for item in raw:
        try:
            index = int(item)
        except ValueError as error:
            raise ValueError(f"feature index must be an int, got {item!r}.") from error
        if index < 0:
            raise ValueError(f"feature index must be non-negative, got {index}.")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(f"feature-indices contains duplicates: {indices}")
    return indices


def verify_same_rows(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> None:
    if len(before_rows) != len(after_rows):
        raise ValueError("before and after row counts differ.")
    for idx, (before, after) in enumerate(zip(before_rows, after_rows, strict=True)):
        before_key = row_key(before)
        after_key = row_key(after)
        if before_key != after_key:
            raise ValueError(f"row mismatch at index {idx}: {before_key} != {after_key}")


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("prompt_id")), str(row.get("label")), str(row.get("target")))


def validate_args(args: argparse.Namespace) -> None:
    if not args.sae_pt.exists():
        raise FileNotFoundError(f"sae-pt does not exist: {args.sae_pt}")
    if not args.before_activations_pt.exists():
        raise FileNotFoundError(f"before-activations-pt does not exist: {args.before_activations_pt}")
    if not args.after_activations_pt.exists():
        raise FileNotFoundError(f"after-activations-pt does not exist: {args.after_activations_pt}")
    if args.output_json.exists():
        raise FileExistsError(f"output-json already exists: {args.output_json}")


if __name__ == "__main__":
    main()
