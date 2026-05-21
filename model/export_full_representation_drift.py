from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from discover_sae_concept_features import load_sae, require_input_mean
from export_native_moe_geometry import project_vectors_svd_3d


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize full target-token representation drift across checkpoints in both residual space "
            "and full SAE feature space."
        )
    )
    parser.add_argument("--sae-pt", required=True, type=Path)
    parser.add_argument("--capture-manifest-json", required=True, type=Path)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--top-hover-features", type=int, default=8)
    args = parser.parse_args()

    validate_args(args)
    if args.top_hover_features <= 0:
        raise ValueError(f"top-hover-features must be positive, got {args.top_hover_features}.")

    sae_payload = torch.load(args.sae_pt, map_location="cpu", weights_only=False)
    sae = load_sae(sae_payload)
    sae.eval()
    input_mean = require_input_mean(sae_payload)

    manifest = load_json(args.capture_manifest_json)
    captures = require_captures(manifest)
    samples = load_checkpoint_samples(
        captures=captures,
        sae=sae,
        input_mean=input_mean,
        top_hover_features=args.top_hover_features,
    )
    residual_vectors = torch.stack([sample["residual_vector"] for sample in samples], dim=0)
    sae_vectors = torch.stack([sample["sae_vector"] for sample in samples], dim=0)
    residual_coordinates = project_vectors_svd_3d(residual_vectors)
    sae_coordinates = project_vectors_svd_3d(sae_vectors)

    rows = []
    for index, sample in enumerate(samples):
        row = {
            key: value
            for key, value in sample.items()
            if key not in ("residual_vector", "sae_vector")
        }
        row["residual_position"] = tensor_row_to_float_list(residual_coordinates[index])
        row["sae_position"] = tensor_row_to_float_list(sae_coordinates[index])
        rows.append(row)

    scene = {
        "name": f"full representation drift:{args.concept}",
        "concept": args.concept,
        "sae_path": str(args.sae_pt),
        "capture_manifest_json": str(args.capture_manifest_json),
        "row_count": len(rows),
        "checkpoint_names": [require_str(capture, "name") for capture in captures],
        "spaces": {
            "residual": {
                "dimension": int(residual_vectors.shape[1]),
                "projection": "SVD/PCA over all target-token residual vectors from all checkpoints",
            },
            "sae": {
                "dimension": int(sae_vectors.shape[1]),
                "projection": "SVD/PCA over all target-token SAE activation vectors from all checkpoints",
            },
        },
        "rows": rows,
        "method": {
            "point": "one target-token example at one checkpoint",
            "line": "same prompt target connected across checkpoints",
            "left_plot": "full residual stream geometry, not selected features",
            "right_plot": "full SAE feature-vector geometry, not selected top features",
            "warning": "This is still a 3D projection of high-dimensional vectors; distances are diagnostic, not the full space.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n")
    write_html(scene=scene, output_html=args.output_html)
    print(json.dumps({"output_json": str(args.output_json), "output_html": str(args.output_html)}, sort_keys=True))


def load_checkpoint_samples(
    *,
    captures: list[dict[str, Any]],
    sae: torch.nn.Module,
    input_mean: torch.Tensor,
    top_hover_features: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    baseline_keys: list[str] | None = None
    for capture in captures:
        checkpoint_name = require_str(capture, "name")
        step = require_int(capture, "step")
        activation_path = Path(require_str(capture, "activation_path"))
        if not activation_path.exists():
            raise FileNotFoundError(f"activation path does not exist: {activation_path}")
        payload = torch.load(activation_path, map_location="cpu", weights_only=False)
        activations = require_activations(payload)
        rows = require_activation_rows(payload)
        if activations.shape[0] != len(rows):
            raise ValueError(
                f"activation row count {activations.shape[0]} != metadata row count {len(rows)} "
                f"for {activation_path}."
            )
        keys = [stable_row_key(row) for row in rows]
        if baseline_keys is None:
            baseline_keys = keys
        elif keys != baseline_keys:
            raise ValueError(f"row alignment mismatch in capture {checkpoint_name!r}.")
        with torch.no_grad():
            _x_hat, z = sae(activations.float() - input_mean)
        for row_index, row in enumerate(rows):
            sae_vector = z[row_index].detach().float().cpu()
            top_features = top_feature_list(sae_vector, top_hover_features)
            residual_vector = activations[row_index].detach().float().cpu()
            samples.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "step": step,
                    "stable_key": keys[row_index],
                    "row_index": int(row["row_index"]),
                    "prompt_id": str(row["prompt_id"]),
                    "label": str(row["label"]),
                    "target": str(row["target"]),
                    "text": str(row["text"]),
                    "target_token_text": str(row.get("target_token_text", "")),
                    "residual_norm": float(torch.linalg.vector_norm(residual_vector).item()),
                    "sae_l1": float(torch.sum(torch.abs(sae_vector)).item()),
                    "sae_l0": int((sae_vector > 0).sum().item()),
                    "top_sae_features": top_features,
                    "residual_vector": residual_vector,
                    "sae_vector": sae_vector,
                }
            )
    if not samples:
        raise RuntimeError("no samples were loaded.")
    return samples


def write_html(*, scene: dict[str, Any], output_html: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_html.parent.mkdir(parents=True, exist_ok=True)
    rows = scene["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("scene rows must be a non-empty list.")
    labels = sorted({str(row["label"]) for row in rows})
    label_colors = {
        "animal": "#dc2626",
        "vehicle": "#2563eb",
        "place": "#16a34a",
        "abstract": "#9333ea",
        "color": "#f59e0b",
    }
    fallback_colors = ["#64748b", "#0f766e", "#be123c", "#7c2d12", "#334155"]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Full residual stream geometry (512 dims -> 3D)",
            "Full SAE feature geometry (all features -> 3D)",
        ),
    )
    add_space_traces(
        fig=fig,
        rows=rows,
        labels=labels,
        label_colors=label_colors,
        fallback_colors=fallback_colors,
        position_key="residual_position",
        column=1,
        scene=scene,
    )
    add_space_traces(
        fig=fig,
        rows=rows,
        labels=labels,
        label_colors=label_colors,
        fallback_colors=fallback_colors,
        position_key="sae_position",
        column=2,
        scene=scene,
    )
    fig.update_layout(
        title=str(scene["name"]),
        autosize=True,
        height=980,
        margin={"l": 0, "r": 0, "t": 64, "b": 0},
        legend={"orientation": "h", "y": -0.05},
    )
    fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3", row=1, col=1)
    fig.update_scenes(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3", row=1, col=2)

    explanation = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html_escape(str(scene["name"]))}</title>
  <style>
    html, body {{ margin: 0; width: 100%; min-height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; }}
    .note {{ padding: 12px 18px 0; max-width: 1500px; line-height: 1.35; font-size: 14px; }}
    .note code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="note">
    <strong>How to read this:</strong>
    each point is one target-token representation at one checkpoint.
    Lines connect the same prompt target through steps {html_escape(str(scene["checkpoint_names"]))}.
    The left plot uses the full residual vector; the right plot uses the full SAE activation vector.
    Both are high-dimensional spaces projected to 3D with SVD/PCA, so this is a global geometry view,
    not a selected-feature-only plot.
  </div>
  {fig.to_html(include_plotlyjs=True, full_html=False, config={"responsive": True})}
</body>
</html>
"""
    output_html.write_text(explanation)


def add_space_traces(
    *,
    fig: Any,
    rows: list[dict[str, Any]],
    labels: list[str],
    label_colors: dict[str, str],
    fallback_colors: list[str],
    position_key: str,
    column: int,
    scene: dict[str, Any],
) -> None:
    concept = str(scene["concept"])
    for label_index, label in enumerate(labels):
        group = [row for row in rows if str(row["label"]) == label]
        color = label_colors.get(label, fallback_colors[label_index % len(fallback_colors)])
        fig.add_trace(
            marker_trace(
                rows=group,
                position_key=position_key,
                name=f"{label} targets",
                color=color,
                size=6.5 if label == concept else 4.5,
                opacity=0.78 if label == concept else 0.52,
            ),
            row=1,
            col=column,
        )
    for stable_key in sorted({str(row["stable_key"]) for row in rows}):
        path = sorted(
            [row for row in rows if str(row["stable_key"]) == stable_key],
            key=lambda row: int(row["step"]),
        )
        if len(path) < 2:
            raise ValueError(f"trajectory for {stable_key!r} has fewer than two points.")
        label = str(path[0]["label"])
        color = label_colors.get(label, "#64748b")
        fig.add_trace(
            line_trace(
                rows=path,
                position_key=position_key,
                color=color,
                width=2 if label == concept else 1,
                opacity=0.48 if label == concept else 0.16,
            ),
            row=1,
            col=column,
        )
    for label_index, label in enumerate(labels):
        color = label_colors.get(label, fallback_colors[label_index % len(fallback_colors)])
        centroids = label_centroids(rows=rows, label=label, position_key=position_key)
        fig.add_trace(
            centroid_trace(
                centroids=centroids,
                name=f"{label} centroid",
                color=color,
                size=8 if label == concept else 5,
                opacity=0.95 if label == concept else 0.62,
            ),
            row=1,
            col=column,
        )


def marker_trace(
    *,
    rows: list[dict[str, Any]],
    position_key: str,
    name: str,
    color: str,
    size: float,
    opacity: float,
) -> Any:
    import plotly.graph_objects as go

    return go.Scatter3d(
        x=[row[position_key][0] for row in rows],
        y=[row[position_key][1] for row in rows],
        z=[row[position_key][2] for row in rows],
        mode="markers",
        marker={"size": size, "color": color, "opacity": opacity},
        text=[hover_text(row) for row in rows],
        hoverinfo="text",
        name=name,
        legendgroup=name,
    )


def line_trace(
    *,
    rows: list[dict[str, Any]],
    position_key: str,
    color: str,
    width: int,
    opacity: float,
) -> Any:
    import plotly.graph_objects as go

    rgba = color_to_rgba(color, opacity)
    return go.Scatter3d(
        x=[row[position_key][0] for row in rows],
        y=[row[position_key][1] for row in rows],
        z=[row[position_key][2] for row in rows],
        mode="lines",
        line={"color": rgba, "width": width},
        hoverinfo="skip",
        showlegend=False,
    )


def centroid_trace(
    *,
    centroids: list[dict[str, Any]],
    name: str,
    color: str,
    size: float,
    opacity: float,
) -> Any:
    import plotly.graph_objects as go

    return go.Scatter3d(
        x=[item["position"][0] for item in centroids],
        y=[item["position"][1] for item in centroids],
        z=[item["position"][2] for item in centroids],
        mode="lines+markers",
        marker={"size": size, "color": color, "opacity": opacity, "symbol": "diamond"},
        line={"color": color_to_rgba(color, opacity), "width": 5},
        text=[centroid_hover(item) for item in centroids],
        hoverinfo="text",
        name=name,
    )


def label_centroids(
    *,
    rows: list[dict[str, Any]],
    label: str,
    position_key: str,
) -> list[dict[str, Any]]:
    steps = sorted({int(row["step"]) for row in rows})
    output: list[dict[str, Any]] = []
    for step in steps:
        group = [row for row in rows if str(row["label"]) == label and int(row["step"]) == step]
        if not group:
            raise RuntimeError(f"no rows for label={label!r} step={step}.")
        positions = torch.tensor([row[position_key] for row in group], dtype=torch.float32)
        output.append(
            {
                "label": label,
                "step": step,
                "count": len(group),
                "position": tensor_row_to_float_list(positions.mean(dim=0)),
            }
        )
    return output


def hover_text(row: dict[str, Any]) -> str:
    top_features = ", ".join(
        f"{item['feature_index']}:{float(item['activation']):.3f}"
        for item in row["top_sae_features"]
    )
    return "<br>".join(
        [
            f"checkpoint={row['checkpoint_name']}",
            f"step={row['step']}",
            f"label={row['label']}",
            f"prompt={row['prompt_id']}",
            f"target={row['target']}",
            f"token={row['target_token_text']}",
            f"residual_norm={float(row['residual_norm']):.4f}",
            f"sae_l0={row['sae_l0']}",
            f"sae_l1={float(row['sae_l1']):.4f}",
            f"top_sae={top_features}",
            f"text={row['text']}",
        ]
    )


def centroid_hover(item: dict[str, Any]) -> str:
    return "<br>".join(
        [
            f"label={item['label']}",
            f"step={item['step']}",
            f"count={item['count']}",
        ]
    )


def top_feature_list(vector: torch.Tensor, top_k: int) -> list[dict[str, float | int]]:
    k = min(top_k, vector.shape[0])
    values, indices = torch.topk(vector, k=k, largest=True)
    return [
        {"feature_index": int(index.item()), "activation": float(value.item())}
        for index, value in zip(indices, values, strict=True)
    ]


def color_to_rgba(hex_color: str, opacity: float) -> str:
    if not hex_color.startswith("#") or len(hex_color) != 7:
        raise ValueError(f"expected #RRGGBB color, got {hex_color!r}.")
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return f"rgba({red},{green},{blue},{opacity})"


def tensor_row_to_float_list(row: torch.Tensor) -> list[float]:
    if row.ndim != 1 or row.shape[0] != 3:
        raise ValueError(f"expected rank-1 tensor with 3 values, got shape {tuple(row.shape)}.")
    return [float(value) for value in row.tolist()]


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
    if not torch.is_floating_point(activations):
        raise TypeError(f"activations must be floating point, got {activations.dtype}.")
    if not torch.isfinite(activations).all():
        raise ValueError("activations contain non-finite values.")
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


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.sae_pt.exists():
        raise FileNotFoundError(f"sae-pt does not exist: {args.sae_pt}")
    if not args.capture_manifest_json.exists():
        raise FileNotFoundError(f"capture-manifest-json does not exist: {args.capture_manifest_json}")
    if args.output_json.exists():
        raise FileExistsError(f"output-json already exists: {args.output_json}")
    if args.output_html.exists():
        raise FileExistsError(f"output-html already exists: {args.output_html}")


if __name__ == "__main__":
    main()
