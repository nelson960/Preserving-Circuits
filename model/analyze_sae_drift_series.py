from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from compare_sae_feature_drift import parse_feature_indices, safe_ratio
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
        description="Measure raw hidden-geometry and fixed-SAE feature drift across checkpoint activations."
    )
    parser.add_argument("--sae-pt", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", default=None, help="Format: name=/path/to/activations.pt")
    parser.add_argument("--capture-manifest-json", type=Path)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--negative-labels", required=True)
    parser.add_argument("--feature-indices", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    checkpoints = load_checkpoint_specs(args)
    negative_labels = parse_csv(args.negative_labels)
    if args.concept in negative_labels:
        raise ValueError("concept must not be included in negative-labels.")
    feature_indices = parse_feature_indices(args.feature_indices)

    sae_payload = torch.load(args.sae_pt, map_location="cpu", weights_only=False)
    sae = load_sae(sae_payload)
    input_mean = require_input_mean(sae_payload)

    loaded = []
    for name, path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        activations = require_activation_matrix(payload)
        rows = require_rows(payload)
        loaded.append({"name": name, "path": path, "activations": activations, "rows": rows})
    baseline = loaded[0]
    for item in loaded[1:]:
        verify_same_rows(baseline["rows"], item["rows"])

    labels = [str(row["label"]) for row in baseline["rows"]]
    concept_indices = [idx for idx, label in enumerate(labels) if label == args.concept]
    negative_indices = [idx for idx, label in enumerate(labels) if label in negative_labels]
    if not concept_indices:
        raise RuntimeError(f"no rows found for concept {args.concept!r}.")
    if not negative_indices:
        raise RuntimeError(f"no rows found for negative labels {negative_labels}.")

    baseline_raw = raw_concept_stats(baseline["activations"], concept_indices, negative_indices)
    encoded = []
    for item in loaded:
        centered = item["activations"] - input_mean
        with torch.no_grad():
            _x_hat, z = sae(centered)
        encoded.append({**item, "z": z})
    baseline_z = encoded[0]["z"]
    baseline_thresholds = {
        feature_index: torch.quantile(baseline_z[:, feature_index], 0.95)
        for feature_index in feature_indices
    }

    checkpoint_reports = []
    for item in encoded:
        raw = raw_concept_stats(item["activations"], concept_indices, negative_indices)
        raw_drift = raw_drift_against_baseline(raw, baseline_raw)
        feature_reports = []
        for feature_index in feature_indices:
            metrics = sae_feature_metrics(
                item["z"][:, feature_index],
                concept_indices=concept_indices,
                negative_indices=negative_indices,
                firing_threshold=baseline_thresholds[feature_index],
            )
            baseline_metrics = sae_feature_metrics(
                baseline_z[:, feature_index],
                concept_indices=concept_indices,
                negative_indices=negative_indices,
                firing_threshold=baseline_thresholds[feature_index],
            )
            feature_reports.append(
                {
                    "feature_index": feature_index,
                    **metrics,
                    "delta_from_baseline": {
                        "concept_mean_delta": metrics["concept_mean"] - baseline_metrics["concept_mean"],
                        "negative_mean_delta": metrics["negative_mean"] - baseline_metrics["negative_mean"],
                        "selectivity_delta": metrics["selectivity"] - baseline_metrics["selectivity"],
                        "auroc_delta": metrics["auroc"] - baseline_metrics["auroc"],
                        "concept_firing_rate_delta": metrics["concept_firing_rate"]
                        - baseline_metrics["concept_firing_rate"],
                        "negative_firing_rate_delta": metrics["negative_firing_rate"]
                        - baseline_metrics["negative_firing_rate"],
                        "fading_ratio": safe_ratio(metrics["concept_mean"], baseline_metrics["concept_mean"]),
                    },
                }
            )
        checkpoint_reports.append(
            {
                "name": item["name"],
                "activation_path": str(item["path"]),
                "raw_hidden_geometry": raw_to_json(raw),
                "raw_hidden_drift_from_baseline": raw_drift,
                "sae_features": feature_reports,
            }
        )

    report = {
        "sae_path": str(args.sae_pt),
        "concept": args.concept,
        "negative_labels": negative_labels,
        "feature_indices": feature_indices,
        "concept_count": len(concept_indices),
        "negative_count": len(negative_indices),
        "checkpoints": checkpoint_reports,
        "method": {
            "raw_direction": "mean(hidden | concept) - mean(hidden | negatives)",
            "raw_rotation_degrees": "angle between checkpoint raw direction and baseline raw direction",
            "raw_norm_ratio": "checkpoint raw direction norm divided by baseline norm",
            "sae_features": "fixed reference SAE; no feature rematching",
            "firing_threshold": "baseline 95th percentile per feature",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_json": str(args.output_json), "checkpoint_count": len(checkpoint_reports)}, sort_keys=True))


def raw_concept_stats(
    activations: torch.Tensor,
    concept_indices: list[int],
    negative_indices: list[int],
) -> dict[str, Any]:
    concept = activations[concept_indices]
    negative = activations[negative_indices]
    concept_mean = concept.mean(dim=0)
    negative_mean = negative.mean(dim=0)
    raw_direction = concept_mean - negative_mean
    norm = float(torch.linalg.vector_norm(raw_direction).item())
    if norm <= 0.0:
        raise RuntimeError("raw concept direction norm is zero.")
    direction = raw_direction / norm
    concept_scores = (concept - negative_mean) @ direction
    negative_scores = (negative - negative_mean) @ direction
    threshold = 0.5 * (float(concept_scores.mean().item()) + float(negative_scores.mean().item()))
    accuracy = float(
        ((concept_scores > threshold).sum().item() + (negative_scores <= threshold).sum().item())
        / (len(concept_indices) + len(negative_indices))
    )
    return {
        "concept_mean": concept_mean,
        "negative_mean": negative_mean,
        "direction": direction,
        "direction_norm": norm,
        "concept_score_mean": float(concept_scores.mean().item()),
        "negative_score_mean": float(negative_scores.mean().item()),
        "margin": float(concept_scores.mean().item() - negative_scores.mean().item()),
        "threshold": threshold,
        "linear_accuracy": accuracy,
    }


def raw_drift_against_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    cosine = float(torch.dot(current["direction"], baseline["direction"]).clamp(-1, 1).item())
    return {
        "direction_cosine": cosine,
        "rotation_degrees": float(math.degrees(math.acos(cosine))),
        "norm_ratio": float(current["direction_norm"] / baseline["direction_norm"]),
        "margin_delta": float(current["margin"] - baseline["margin"]),
        "linear_accuracy_delta": float(current["linear_accuracy"] - baseline["linear_accuracy"]),
        "concept_centroid_shift": float(
            torch.linalg.vector_norm(current["concept_mean"] - baseline["concept_mean"]).item()
        ),
        "negative_centroid_shift": float(
            torch.linalg.vector_norm(current["negative_mean"] - baseline["negative_mean"]).item()
        ),
    }


def raw_to_json(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "direction_norm": float(raw["direction_norm"]),
        "concept_score_mean": float(raw["concept_score_mean"]),
        "negative_score_mean": float(raw["negative_score_mean"]),
        "margin": float(raw["margin"]),
        "threshold": float(raw["threshold"]),
        "linear_accuracy": float(raw["linear_accuracy"]),
    }


def sae_feature_metrics(
    values: torch.Tensor,
    *,
    concept_indices: list[int],
    negative_indices: list[int],
    firing_threshold: torch.Tensor,
) -> dict[str, float]:
    concept_values = values[concept_indices]
    negative_values = values[negative_indices]
    labels = torch.zeros(values.shape[0], dtype=torch.bool)
    valid = torch.zeros(values.shape[0], dtype=torch.bool)
    labels[concept_indices] = True
    valid[concept_indices] = True
    valid[negative_indices] = True
    return {
        "concept_mean": float(concept_values.mean().item()),
        "negative_mean": float(negative_values.mean().item()),
        "selectivity": float(concept_values.mean().item() - negative_values.mean().item()),
        "concept_std": float(concept_values.std(unbiased=False).item()),
        "negative_std": float(negative_values.std(unbiased=False).item()),
        "concept_firing_rate": float((concept_values > firing_threshold).float().mean().item()),
        "negative_firing_rate": float((negative_values > firing_threshold).float().mean().item()),
        "auroc": auroc(values[valid], labels[valid]),
    }


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    checkpoints: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"checkpoint must be name=/path format, got {value!r}.")
        name, path_text = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"checkpoint name must be non-empty: {value!r}")
        if name in names:
            raise ValueError(f"duplicate checkpoint name: {name}")
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint activation path does not exist: {path}")
        checkpoints.append((name, path))
        names.add(name)
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoints are required.")
    return checkpoints


def load_checkpoint_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    has_direct = args.checkpoint is not None and len(args.checkpoint) > 0
    has_manifest = args.capture_manifest_json is not None
    if has_direct == has_manifest:
        raise ValueError("provide exactly one of --checkpoint or --capture-manifest-json.")
    if has_direct:
        return parse_checkpoints(args.checkpoint)
    if args.capture_manifest_json is None:
        raise RuntimeError("unreachable missing manifest.")
    manifest = json.loads(args.capture_manifest_json.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("capture manifest must be a JSON object.")
    captures = manifest.get("captures")
    if not isinstance(captures, list) or len(captures) < 2:
        raise ValueError("capture manifest must contain at least two captures.")
    specs: list[tuple[str, Path]] = []
    names: set[str] = set()
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise TypeError(f"captures[{index}] must be an object.")
        name = capture.get("name")
        path_text = capture.get("activation_path")
        if not isinstance(name, str) or not name:
            raise ValueError(f"captures[{index}].name must be a non-empty string.")
        if name in names:
            raise ValueError(f"duplicate capture name: {name}")
        if not isinstance(path_text, str) or not path_text:
            raise ValueError(f"captures[{index}].activation_path must be a non-empty string.")
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"capture activation path does not exist: {path}")
        specs.append((name, path))
        names.add(name)
    return specs


def verify_same_rows(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> None:
    if len(before_rows) != len(after_rows):
        raise ValueError("checkpoint row counts differ.")
    for index, (before, after) in enumerate(zip(before_rows, after_rows, strict=True)):
        if row_key(before) != row_key(after):
            raise ValueError(f"row mismatch at index {index}: {row_key(before)} != {row_key(after)}")


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("prompt_id")), str(row.get("label")), str(row.get("target")))


def validate_args(args: argparse.Namespace) -> None:
    if not args.sae_pt.exists():
        raise FileNotFoundError(f"sae-pt does not exist: {args.sae_pt}")
    if args.capture_manifest_json is not None and not args.capture_manifest_json.exists():
        raise FileNotFoundError(f"capture-manifest-json does not exist: {args.capture_manifest_json}")
    if args.output_json.exists():
        raise FileExistsError(f"output-json already exists: {args.output_json}")


if __name__ == "__main__":
    main()
