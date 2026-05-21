from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from compare_sae_feature_drift import parse_feature_indices
from discover_sae_concept_features import load_sae, require_input_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export an interactive 3D visualization that combines fixed-SAE feature-coordinate drift "
            "with causal ablation trajectories."
        )
    )
    parser.add_argument("--sae-pt", required=True, type=Path)
    parser.add_argument("--capture-manifest-json", required=True, type=Path)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--semantic-feature-index", required=True, type=int)
    parser.add_argument("--causal-feature-indices", required=True, help="Comma-separated SAE feature indices.")
    parser.add_argument("--semantic-drift-json", required=True, type=Path)
    parser.add_argument("--causal-drift-json", required=True, type=Path)
    parser.add_argument("--semantic-causal-json", required=True, type=Path)
    parser.add_argument("--causal-causal-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    causal_feature_indices = parse_feature_indices(args.causal_feature_indices)
    validate_feature_inputs(args.semantic_feature_index, causal_feature_indices)

    sae_payload = torch.load(args.sae_pt, map_location="cpu", weights_only=False)
    sae = load_sae(sae_payload)
    input_mean = require_input_mean(sae_payload)
    validate_sae_feature_index(args.semantic_feature_index, sae.encoder.out_features)
    for feature_index in causal_feature_indices:
        validate_sae_feature_index(feature_index, sae.encoder.out_features)

    manifest = load_json(args.capture_manifest_json)
    captures = require_captures(manifest)
    semantic_drift = load_json(args.semantic_drift_json)
    causal_drift = load_json(args.causal_drift_json)
    semantic_causal = load_json(args.semantic_causal_json)
    causal_causal = load_json(args.causal_causal_json)

    coordinate_rows = build_coordinate_rows(
        captures=captures,
        sae=sae,
        input_mean=input_mean,
        semantic_feature_index=args.semantic_feature_index,
        causal_feature_indices=causal_feature_indices,
    )
    metric_rows = build_metric_rows(
        concept=args.concept,
        semantic_feature_index=args.semantic_feature_index,
        causal_feature_indices=causal_feature_indices,
        semantic_drift=semantic_drift,
        causal_drift=causal_drift,
        semantic_causal=semantic_causal,
        causal_causal=causal_causal,
    )
    scene = {
        "name": f"SAE causal drift:{args.concept}",
        "concept": args.concept,
        "semantic_feature_index": args.semantic_feature_index,
        "causal_feature_indices": causal_feature_indices,
        "coordinate_rows": coordinate_rows,
        "metric_rows": metric_rows,
        "method": {
            "coordinate_scene": (
                "x = semantic SAE feature activation; y = sum of causal SAE feature activations; "
                "z = checkpoint step. Lines connect the same prompt target across checkpoints."
            ),
            "metric_scene": (
                "x = raw hidden rotation from baseline; y = fixed-SAE feature fading ratio; "
                "z = direct causal ablation delta on concept next-token logprob."
            ),
            "interpretation_warning": (
                "A feature can be decodable without being causal. The metric scene is included to prevent "
                "visual semantic purity from being mistaken for behavioral use."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n")
    write_html(scene=scene, output_html=args.output_html)
    print(json.dumps({"output_json": str(args.output_json), "output_html": str(args.output_html)}, sort_keys=True))


def build_coordinate_rows(
    *,
    captures: list[dict[str, Any]],
    sae: torch.nn.Module,
    input_mean: torch.Tensor,
    semantic_feature_index: int,
    causal_feature_indices: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_keys: list[str] | None = None
    for capture in captures:
        name = require_str(capture, "name")
        step = require_int(capture, "step")
        activation_path = Path(require_str(capture, "activation_path"))
        if not activation_path.exists():
            raise FileNotFoundError(f"activation path does not exist: {activation_path}")
        payload = torch.load(activation_path, map_location="cpu", weights_only=False)
        activations = require_activations(payload)
        capture_rows = require_activation_rows(payload)
        keys = [stable_row_key(row) for row in capture_rows]
        if baseline_keys is None:
            baseline_keys = keys
        elif keys != baseline_keys:
            raise ValueError(f"row alignment mismatch in capture {name!r}.")
        with torch.no_grad():
            _x_hat, z = sae(activations.float() - input_mean)
        for row_index, row in enumerate(capture_rows):
            stable_key = keys[row_index]
            semantic_activation = float(z[row_index, semantic_feature_index].item())
            causal_sum_activation = float(z[row_index, causal_feature_indices].sum().item())
            causal_mean_activation = float(z[row_index, causal_feature_indices].mean().item())
            rows.append(
                {
                    "checkpoint_name": name,
                    "step": step,
                    "stable_key": stable_key,
                    "prompt_id": str(row["prompt_id"]),
                    "label": str(row["label"]),
                    "target": str(row["target"]),
                    "text": str(row["text"]),
                    "semantic_activation": semantic_activation,
                    "causal_sum_activation": causal_sum_activation,
                    "causal_mean_activation": causal_mean_activation,
                    "is_concept": str(row["label"]) == str(row.get("concept", "")),
                }
            )
    if not rows:
        raise RuntimeError("no coordinate rows were built.")
    return rows


def build_metric_rows(
    *,
    concept: str,
    semantic_feature_index: int,
    causal_feature_indices: list[int],
    semantic_drift: dict[str, Any],
    causal_drift: dict[str, Any],
    semantic_causal: dict[str, Any],
    causal_causal: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        feature_metric_rows(
            feature_index=semantic_feature_index,
            feature_role="decodable_semantic",
            concept=concept,
            drift=semantic_drift,
            causal=semantic_causal,
        )
    )
    for feature_index in causal_feature_indices:
        rows.extend(
            feature_metric_rows(
                feature_index=feature_index,
                feature_role="causally_ranked",
                concept=concept,
                drift=causal_drift,
                causal=causal_causal,
            )
        )
    rows.extend(
        combined_metric_rows(
            feature_indices=causal_feature_indices,
            concept=concept,
            drift=causal_drift,
            causal=causal_causal,
        )
    )
    if not rows:
        raise RuntimeError("no metric rows were built.")
    return rows


def feature_metric_rows(
    *,
    feature_index: int,
    feature_role: str,
    concept: str,
    drift: dict[str, Any],
    causal: dict[str, Any],
) -> list[dict[str, Any]]:
    causal_by_checkpoint = causal_feature_by_checkpoint(causal, feature_set_name=f"feature_{feature_index}")
    rows = []
    for checkpoint in require_checkpoint_reports(drift):
        checkpoint_name = require_str(checkpoint, "name")
        feature_report = find_sae_feature_report(checkpoint, feature_index)
        causal_summary = causal_by_checkpoint[checkpoint_name]["summary_by_label"][concept]
        raw = checkpoint["raw_hidden_drift_from_baseline"]
        delta = feature_report["delta_from_baseline"]
        rows.append(
            {
                "feature_role": feature_role,
                "feature_name": f"feature_{feature_index}",
                "feature_index": feature_index,
                "checkpoint_name": checkpoint_name,
                "step": checkpoint_step_from_name_or_report(checkpoint),
                "raw_rotation_degrees": float(raw["rotation_degrees"]),
                "fading_ratio": nullable_float(delta["fading_ratio"]),
                "selectivity": float(feature_report["selectivity"]),
                "auroc": float(feature_report["auroc"]),
                "concept_mean": float(feature_report["concept_mean"]),
                "animal_ablate_delta": float(causal_summary["ablate_delta_mean"]),
                "animal_patch_from_reference_delta": float(causal_summary["patch_from_reference_delta_mean"]),
                "reverse_patch_delta": float(causal_summary["reverse_patch_delta_mean"]),
            }
        )
    return rows


def combined_metric_rows(
    *,
    feature_indices: list[int],
    concept: str,
    drift: dict[str, Any],
    causal: dict[str, Any],
) -> list[dict[str, Any]]:
    causal_by_checkpoint = causal_feature_by_checkpoint(causal, feature_set_name="combined_requested_features")
    rows = []
    for checkpoint in require_checkpoint_reports(drift):
        checkpoint_name = require_str(checkpoint, "name")
        raw = checkpoint["raw_hidden_drift_from_baseline"]
        feature_reports = [find_sae_feature_report(checkpoint, feature_index) for feature_index in feature_indices]
        fading_values = [
            nullable_float(feature_report["delta_from_baseline"]["fading_ratio"])
            for feature_report in feature_reports
        ]
        if any(value is None for value in fading_values):
            fading_ratio = None
        else:
            fading_ratio = sum(float(value) for value in fading_values) / len(fading_values)
        causal_summary = causal_by_checkpoint[checkpoint_name]["summary_by_label"][concept]
        rows.append(
            {
                "feature_role": "causal_combined_set",
                "feature_name": "combined_causal_top5",
                "feature_indices": feature_indices,
                "checkpoint_name": checkpoint_name,
                "step": checkpoint_step_from_name_or_report(checkpoint),
                "raw_rotation_degrees": float(raw["rotation_degrees"]),
                "fading_ratio": fading_ratio,
                "selectivity": sum(float(item["selectivity"]) for item in feature_reports) / len(feature_reports),
                "auroc": sum(float(item["auroc"]) for item in feature_reports) / len(feature_reports),
                "concept_mean": sum(float(item["concept_mean"]) for item in feature_reports),
                "animal_ablate_delta": float(causal_summary["ablate_delta_mean"]),
                "animal_patch_from_reference_delta": float(causal_summary["patch_from_reference_delta_mean"]),
                "reverse_patch_delta": float(causal_summary["reverse_patch_delta_mean"]),
            }
        )
    return rows


def write_html(*, scene: dict[str, Any], output_html: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_html.parent.mkdir(parents=True, exist_ok=True)
    coordinate_rows = scene["coordinate_rows"]
    metric_rows = scene["metric_rows"]
    labels = sorted({str(row["label"]) for row in coordinate_rows})
    label_colors = {
        "animal": "#dc2626",
        "vehicle": "#2563eb",
        "place": "#16a34a",
        "abstract": "#9333ea",
        "color": "#f59e0b",
    }
    fallback_colors = ["#64748b", "#0f766e", "#be123c", "#7c2d12", "#334155"]
    traces: list[Any] = []
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "SAE feature-coordinate drift",
            "Geometry vs causal effect",
        ),
    )

    for label_index, label in enumerate(labels):
        group = [row for row in coordinate_rows if str(row["label"]) == label]
        color = label_colors.get(label, fallback_colors[label_index % len(fallback_colors)])
        fig.add_trace(
            go.Scatter3d(
                x=[row["semantic_activation"] for row in group],
                y=[row["causal_sum_activation"] for row in group],
                z=[row["step"] for row in group],
                mode="markers",
                marker={"size": 4.5 if label != scene["concept"] else 6.5, "color": color, "opacity": 0.72},
                text=[coordinate_hover(row) for row in group],
                hoverinfo="text",
                name=f"{label} targets",
                legendgroup=f"label:{label}",
            ),
            row=1,
            col=1,
        )

    for stable_key in sorted({str(row["stable_key"]) for row in coordinate_rows}):
        path = sorted(
            [row for row in coordinate_rows if str(row["stable_key"]) == stable_key],
            key=lambda row: int(row["step"]),
        )
        label = str(path[0]["label"])
        color = label_colors.get(label, "#64748b")
        fig.add_trace(
            go.Scatter3d(
                x=[row["semantic_activation"] for row in path],
                y=[row["causal_sum_activation"] for row in path],
                z=[row["step"] for row in path],
                mode="lines",
                line={"color": color, "width": 1},
                opacity=0.22 if label != scene["concept"] else 0.42,
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"label:{label}",
            ),
            row=1,
            col=1,
        )

    metric_groups = sorted({str(row["feature_name"]) for row in metric_rows})
    metric_colors = {
        f"feature_{scene['semantic_feature_index']}": "#dc2626",
        "combined_causal_top5": "#111827",
    }
    for index, feature_name in enumerate(metric_groups):
        group = sorted([row for row in metric_rows if str(row["feature_name"]) == feature_name], key=lambda row: int(row["step"]))
        if not group:
            continue
        color = metric_colors.get(feature_name, fallback_colors[index % len(fallback_colors)])
        marker_size = 8 if feature_name == "combined_causal_top5" else 5.5
        fig.add_trace(
            go.Scatter3d(
                x=[row["raw_rotation_degrees"] for row in group],
                y=[row["fading_ratio"] if row["fading_ratio"] is not None else 0.0 for row in group],
                z=[row["animal_ablate_delta"] for row in group],
                mode="lines+markers",
                marker={"size": marker_size, "color": color, "opacity": 0.9},
                line={"color": color, "width": 5 if feature_name == "combined_causal_top5" else 3},
                text=[metric_hover(row) for row in group],
                hoverinfo="text",
                name=feature_name,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=str(scene["name"]),
        margin={"l": 0, "r": 0, "t": 58, "b": 0},
        legend={"orientation": "h", "y": -0.04},
    )
    fig.update_scenes(
        xaxis_title=f"semantic feature {scene['semantic_feature_index']} activation",
        yaxis_title="causal top-k activation sum",
        zaxis_title="fine-tune step",
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis_title="raw animal direction rotation (deg)",
        yaxis_title="feature fading ratio",
        zaxis_title="animal ablation delta",
        row=1,
        col=2,
    )
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)


def coordinate_hover(row: dict[str, Any]) -> str:
    return "<br>".join(
        [
            f"checkpoint={row['checkpoint_name']}",
            f"step={row['step']}",
            f"label={row['label']}",
            f"prompt={row['prompt_id']}",
            f"target={row['target']}",
            f"semantic_activation={float(row['semantic_activation']):.4f}",
            f"causal_sum_activation={float(row['causal_sum_activation']):.4f}",
            f"text={row['text']}",
        ]
    )


def metric_hover(row: dict[str, Any]) -> str:
    lines = [
        f"feature={row['feature_name']}",
        f"role={row['feature_role']}",
        f"checkpoint={row['checkpoint_name']}",
        f"step={row['step']}",
        f"rotation={float(row['raw_rotation_degrees']):.4f}",
        f"fading={row['fading_ratio']}",
        f"selectivity={float(row['selectivity']):.4f}",
        f"auroc={float(row['auroc']):.4f}",
        f"animal_ablate_delta={float(row['animal_ablate_delta']):.4f}",
        f"patch_delta={float(row['animal_patch_from_reference_delta']):.4f}",
        f"reverse_patch_delta={float(row['reverse_patch_delta']):.4f}",
    ]
    return "<br>".join(lines)


def causal_feature_by_checkpoint(causal: dict[str, Any], *, feature_set_name: str) -> dict[str, dict[str, Any]]:
    result = {}
    for checkpoint in require_checkpoint_reports(causal):
        name = require_str(checkpoint, "name")
        features = checkpoint.get("features")
        if not isinstance(features, list):
            raise TypeError(f"checkpoint {name!r} causal features must be a list.")
        matches = [
            item
            for item in features
            if causal_feature_name(item) == feature_set_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one causal feature set {feature_set_name!r} in checkpoint {name!r}, "
                f"got {len(matches)}."
            )
        result[name] = matches[0]
    return result


def causal_feature_name(feature_report: dict[str, Any]) -> str:
    explicit = feature_report.get("feature_set_name")
    if isinstance(explicit, str) and explicit:
        return explicit
    feature_index = feature_report.get("feature_index")
    if isinstance(feature_index, int):
        return f"feature_{feature_index}"
    raise ValueError("causal feature report must contain feature_set_name or integer feature_index.")


def find_sae_feature_report(checkpoint: dict[str, Any], feature_index: int) -> dict[str, Any]:
    features = checkpoint.get("sae_features")
    if not isinstance(features, list):
        raise TypeError("drift checkpoint must contain sae_features list.")
    matches = [item for item in features if int(item["feature_index"]) == feature_index]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one feature report for feature {feature_index} in checkpoint "
            f"{checkpoint.get('name')!r}, got {len(matches)}."
        )
    return matches[0]


def checkpoint_step_from_name_or_report(checkpoint: dict[str, Any]) -> int:
    if "step" in checkpoint:
        value = checkpoint["step"]
        if not isinstance(value, int):
            raise TypeError(f"checkpoint step must be int, got {type(value).__name__}.")
        return value
    name = require_str(checkpoint, "name")
    if name == "step0":
        return 0
    if not name.startswith("step"):
        raise ValueError(f"cannot infer step from checkpoint name {name!r}.")
    return int(name.removeprefix("step"))


def require_checkpoint_reports(payload: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("payload must contain a non-empty checkpoints list.")
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise TypeError(f"checkpoints[{index}] must be an object.")
    return checkpoints


def require_captures(payload: dict[str, Any]) -> list[dict[str, Any]]:
    captures = payload.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("manifest must contain a non-empty captures list.")
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise TypeError(f"captures[{index}] must be an object.")
        require_str(capture, "name")
        require_str(capture, "activation_path")
        require_int(capture, "step")
    return captures


def require_activations(payload: Any) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise TypeError("activation payload must be a dict.")
    activations = payload.get("activations")
    if not isinstance(activations, torch.Tensor):
        raise TypeError("activation payload must contain tensor activations.")
    if activations.ndim != 2:
        raise ValueError(f"activations must be rank-2, got shape {tuple(activations.shape)}.")
    return activations.float()


def require_activation_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise TypeError("activation payload must be a dict.")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("activation payload must contain a non-empty rows list.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"rows[{index}] must be an object.")
    return rows


def stable_row_key(row: dict[str, Any]) -> str:
    return "|".join([str(row["prompt_id"]), str(row["label"]), str(row["target"]), str(row["text"])])


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} must be a non-empty string.")
    return value


def require_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key!r} must be an int.")
    return value


def nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def validate_sae_feature_index(feature_index: int, feature_dim: int) -> None:
    if feature_index < 0 or feature_index >= feature_dim:
        raise IndexError(f"feature index {feature_index} out of range for feature_dim={feature_dim}.")


def validate_feature_inputs(semantic_feature_index: int, causal_feature_indices: list[int]) -> None:
    if semantic_feature_index in causal_feature_indices:
        raise ValueError("semantic-feature-index must not also appear in causal-feature-indices.")


def validate_args(args: argparse.Namespace) -> None:
    for path_name in (
        "sae_pt",
        "capture_manifest_json",
        "semantic_drift_json",
        "causal_drift_json",
        "semantic_causal_json",
        "causal_causal_json",
    ):
        path = getattr(args, path_name)
        if not path.exists():
            raise FileNotFoundError(f"{path_name.replace('_', '-')} does not exist: {path}")
    if args.output_json.exists():
        raise FileExistsError(f"output-json already exists: {args.output_json}")
    if args.output_html.exists():
        raise FileExistsError(f"output-html already exists: {args.output_html}")


if __name__ == "__main__":
    main()
