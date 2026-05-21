from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from export_native_moe_geometry import (
    collect_router_paths,
    capture_activation_geometry_inputs,
    device_from_name,
    dtype_from_name,
    set_hf_home,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Isolate a labeled concept as a contrastive hidden-state direction "
            "and export a feature-aligned 3D scene."
        )
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-kind", required=True, choices=("causal-lm", "seq2seq-lm"))
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--control-label", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output-scene-json", required=True, type=Path)
    parser.add_argument("--output-report-json", required=True, type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--top-neurons", type=int, default=16)
    parser.add_argument("--layer-gap", type=float, default=8.0)
    args = parser.parse_args()

    validate_args(args)
    set_hf_home(args.hf_home)

    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    prompt_set = load_labeled_prompt_set(args.prompts)
    labels = {example["label"] for example in prompt_set["examples"]}
    if args.concept not in labels:
        raise ValueError(f"concept label {args.concept!r} not present in prompt file labels {sorted(labels)}.")
    if args.control_label not in labels:
        raise ValueError(
            f"control label {args.control_label!r} not present in prompt file labels {sorted(labels)}."
        )
    if args.concept == args.control_label:
        raise ValueError("concept and control-label must be different.")

    device = device_from_name(args.device)
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model_cls = AutoModelForCausalLM if args.model_kind == "causal-lm" else AutoModelForSeq2SeqLM
    model = model_cls.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    router_paths = collect_router_paths(model, model_kind=args.model_kind)
    capture = capture_activation_geometry_inputs(
        model=model,
        tokenizer=tokenizer,
        prompt_set=prompt_set,
        device=device,
        max_length=args.max_length,
        model_kind=args.model_kind,
        router_paths=router_paths,
    )
    label_by_prompt_id = {
        str(example["id"]): str(example["label"]) for example in prompt_set["examples"]
    }
    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    selected_vectors, selected_rows = select_non_special_labeled_rows(
        vectors=capture["vectors"],
        metadata_rows=capture["metadata_rows"],
        label_by_prompt_id=label_by_prompt_id,
        special_ids=special_ids,
        concept=args.concept,
        control_label=args.control_label,
    )
    layer_results = compute_layer_feature_directions(
        vectors=selected_vectors,
        rows=selected_rows,
        concept=args.concept,
        control_label=args.control_label,
        top_neurons=args.top_neurons,
    )
    scene = build_feature_scene(
        vectors=selected_vectors,
        rows=selected_rows,
        layer_results=layer_results,
        concept=args.concept,
        control_label=args.control_label,
        scene_name=f"{args.model_dir.name}:{args.concept}_feature_direction",
        layer_gap=args.layer_gap,
    )
    report = build_report(
        model_dir=args.model_dir,
        prompt_file=args.prompts,
        prompt_name=str(prompt_set["name"]),
        concept=args.concept,
        control_label=args.control_label,
        selected_rows=selected_rows,
        layer_results=layer_results,
        scene_path=args.output_scene_json,
    )

    args.output_scene_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_scene_json.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n")
    print(f"wrote_scene_json={args.output_scene_json}")

    args.output_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote_report_json={args.output_report_json}")

    if args.output_html is not None:
        write_feature_html(scene=scene, report=report, output_html=args.output_html)
        print(f"wrote_html={args.output_html}")


def select_non_special_labeled_rows(
    *,
    vectors: torch.Tensor,
    metadata_rows: list[dict[str, Any]],
    label_by_prompt_id: dict[str, str],
    special_ids: set[int],
    concept: str,
    control_label: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    selected_vectors: list[torch.Tensor] = []
    selected_rows: list[dict[str, Any]] = []
    if vectors.shape[0] != len(metadata_rows):
        raise ValueError("vectors row count must match metadata row count.")
    for index, row in enumerate(metadata_rows):
        prompt_id = str(row["prompt_id"])
        if prompt_id not in label_by_prompt_id:
            raise KeyError(f"missing label for prompt_id {prompt_id!r}.")
        label = label_by_prompt_id[prompt_id]
        if label not in (concept, control_label):
            continue
        token_id = int(row["token_id"])
        if token_id in special_ids:
            continue
        selected = dict(row)
        selected["label"] = label
        selected_vectors.append(vectors[index].detach().float().cpu())
        selected_rows.append(selected)
    if not selected_vectors:
        raise RuntimeError("no non-special concept/control rows were selected.")
    concept_count = sum(1 for row in selected_rows if row["label"] == concept)
    control_count = sum(1 for row in selected_rows if row["label"] == control_label)
    if concept_count == 0:
        raise RuntimeError(f"no rows selected for concept {concept!r}.")
    if control_count == 0:
        raise RuntimeError(f"no rows selected for control label {control_label!r}.")
    return torch.stack(selected_vectors, dim=0), selected_rows


def compute_layer_feature_directions(
    *,
    vectors: torch.Tensor,
    rows: list[dict[str, Any]],
    concept: str,
    control_label: str,
    top_neurons: int,
) -> dict[int, dict[str, Any]]:
    if top_neurons <= 0:
        raise ValueError(f"top_neurons must be positive, got {top_neurons}.")
    layers = sorted({int(row["layer_index"]) for row in rows})
    results: dict[int, dict[str, Any]] = {}
    for layer in layers:
        indices = [idx for idx, row in enumerate(rows) if int(row["layer_index"]) == layer]
        concept_indices = [idx for idx in indices if rows[idx]["label"] == concept]
        control_indices = [idx for idx in indices if rows[idx]["label"] == control_label]
        if not concept_indices or not control_indices:
            raise RuntimeError(
                f"layer {layer} does not have both concept and control rows: "
                f"concept={len(concept_indices)} control={len(control_indices)}."
            )
        concept_vectors = vectors[concept_indices]
        control_vectors = vectors[control_indices]
        concept_mean = concept_vectors.mean(dim=0)
        control_mean = control_vectors.mean(dim=0)
        raw_direction = concept_mean - control_mean
        direction_norm = float(torch.linalg.vector_norm(raw_direction).item())
        if direction_norm <= 0.0:
            raise RuntimeError(f"layer {layer} concept direction has zero norm.")
        direction = raw_direction / direction_norm
        concept_scores = (concept_vectors - control_mean) @ direction
        control_scores = (control_vectors - control_mean) @ direction
        threshold = 0.5 * (float(concept_scores.mean().item()) + float(control_scores.mean().item()))
        concept_correct = int((concept_scores > threshold).sum().item())
        control_correct = int((control_scores <= threshold).sum().item())
        accuracy = float((concept_correct + control_correct) / (len(concept_indices) + len(control_indices)))
        pooled_std = pooled_score_std(concept_scores, control_scores)
        effect_size = direction_norm / pooled_std if pooled_std > 0.0 else math.inf
        top_positive, top_negative = top_direction_neurons(direction, top_neurons)
        results[layer] = {
            "layer_index": layer,
            "direction": direction,
            "concept_mean": concept_mean,
            "control_mean": control_mean,
            "direction_norm": direction_norm,
            "concept_score_mean": float(concept_scores.mean().item()),
            "control_score_mean": float(control_scores.mean().item()),
            "concept_score_std": float(concept_scores.std(unbiased=False).item()),
            "control_score_std": float(control_scores.std(unbiased=False).item()),
            "threshold": threshold,
            "linear_separation_accuracy": accuracy,
            "effect_size": effect_size,
            "concept_count": len(concept_indices),
            "control_count": len(control_indices),
            "top_positive_neurons": top_positive,
            "top_negative_neurons": top_negative,
        }
    return results


def pooled_score_std(concept_scores: torch.Tensor, control_scores: torch.Tensor) -> float:
    concept_var = float(concept_scores.var(unbiased=False).item())
    control_var = float(control_scores.var(unbiased=False).item())
    pooled = math.sqrt(max(0.0, 0.5 * (concept_var + control_var)))
    return pooled


def top_direction_neurons(direction: torch.Tensor, top_k: int) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    k = min(top_k, direction.shape[0])
    positive_values, positive_indices = torch.topk(direction, k=k, largest=True)
    negative_values, negative_indices = torch.topk(-direction, k=k, largest=True)
    positive = [
        {"neuron_index": int(index.item()), "coefficient": float(value.item())}
        for index, value in zip(positive_indices, positive_values, strict=True)
    ]
    negative = [
        {"neuron_index": int(index.item()), "coefficient": float(-value.item())}
        for index, value in zip(negative_indices, negative_values, strict=True)
    ]
    return positive, negative


def build_feature_scene(
    *,
    vectors: torch.Tensor,
    rows: list[dict[str, Any]],
    layer_results: dict[int, dict[str, Any]],
    concept: str,
    control_label: str,
    scene_name: str,
    layer_gap: float,
) -> dict[str, Any]:
    positions = compute_feature_aligned_positions(
        vectors=vectors,
        rows=rows,
        layer_results=layer_results,
        layer_gap=layer_gap,
    )
    nodes: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        layer = int(row["layer_index"])
        result = layer_results[layer]
        direction = result["direction"]
        control_mean = result["control_mean"]
        score = float(((vectors[index] - control_mean) @ direction).item())
        node_id = f"feature_sample:{row['prompt_id']}:{row['token_index']}:{layer}"
        nodes.append(
            {
                "id": node_id,
                "kind": "feature_sample",
                "position": [float(value) for value in positions[index].tolist()],
                "label": f"{row['label']} L{layer} {row['token_text']}",
                "layer": layer,
                "space": "concept_feature_axis",
                "size": 1.0 if row["label"] == concept else 0.75,
                "color_value": 1.0 if row["label"] == concept else 0.0,
                "metadata": sanitize_metadata(
                    {
                        **row,
                        "concept_score": score,
                        "feature_axis": f"{concept}_minus_{control_label}",
                    }
                ),
            }
        )

    edges: list[dict[str, Any]] = []
    for layer, result in sorted(layer_results.items()):
        y = layer * layer_gap
        control_id = f"feature_direction:{layer}:control"
        concept_id = f"feature_direction:{layer}:concept"
        nodes.append(
            {
                "id": control_id,
                "kind": "feature_direction_endpoint",
                "position": [0.0, float(y), 0.0],
                "label": f"L{layer} {control_label} centroid",
                "layer": layer,
                "space": "concept_feature_axis",
                "size": 2.4,
                "color_value": 0.0,
                "metadata": {
                    "layer_index": layer,
                    "label": control_label,
                    "endpoint": "control_centroid",
                    "feature_axis": f"{concept}_minus_{control_label}",
                },
            }
        )
        nodes.append(
            {
                "id": concept_id,
                "kind": "feature_direction_endpoint",
                "position": [float(result["direction_norm"]), float(y), 0.0],
                "label": f"L{layer} {concept} centroid",
                "layer": layer,
                "space": "concept_feature_axis",
                "size": 2.4,
                "color_value": 1.0,
                "metadata": {
                    "layer_index": layer,
                    "label": concept,
                    "endpoint": "concept_centroid",
                    "feature_axis": f"{concept}_minus_{control_label}",
                    "direction_norm": float(result["direction_norm"]),
                    "linear_separation_accuracy": float(result["linear_separation_accuracy"]),
                    "effect_size": finite_or_string(result["effect_size"]),
                },
            }
        )
        edges.append(
            {
                "source": control_id,
                "target": concept_id,
                "kind": "concept_direction",
                "weight": float(result["direction_norm"]),
                "label": f"{concept} direction L{layer}",
                "metadata": {
                    "layer_index": layer,
                    "feature_axis": f"{concept}_minus_{control_label}",
                    "direction_norm": float(result["direction_norm"]),
                    "linear_separation_accuracy": float(result["linear_separation_accuracy"]),
                    "effect_size": finite_or_string(result["effect_size"]),
                },
            }
        )
    return {
        "name": scene_name,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "scene_type": "concept_feature_geometry",
            "concept": concept,
            "control_label": control_label,
            "point_semantics": "non_special_token_hidden_state",
            "x_axis": "contrastive_concept_score",
            "y_axis": "layer_plus_residual_pc1",
            "z_axis": "residual_pc2",
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


def compute_feature_aligned_positions(
    *,
    vectors: torch.Tensor,
    rows: list[dict[str, Any]],
    layer_results: dict[int, dict[str, Any]],
    layer_gap: float,
) -> torch.Tensor:
    residuals: list[torch.Tensor] = []
    scores: list[float] = []
    layer_offsets: list[float] = []
    for index, row in enumerate(rows):
        layer = int(row["layer_index"])
        result = layer_results[layer]
        direction = result["direction"]
        control_mean = result["control_mean"]
        centered = vectors[index] - control_mean
        score = float((centered @ direction).item())
        residual = centered - score * direction
        scores.append(score)
        residuals.append(residual)
        layer_offsets.append(float(layer) * layer_gap)
    residual_matrix = torch.stack(residuals, dim=0)
    residual_coordinates = project_residuals_2d(residual_matrix)
    positions = torch.zeros((vectors.shape[0], 3), dtype=torch.float32)
    positions[:, 0] = torch.tensor(scores, dtype=torch.float32)
    positions[:, 1] = torch.tensor(layer_offsets, dtype=torch.float32) + residual_coordinates[:, 0]
    positions[:, 2] = residual_coordinates[:, 1]
    return positions


def project_residuals_2d(residuals: torch.Tensor) -> torch.Tensor:
    if residuals.ndim != 2:
        raise ValueError(f"residuals must be rank-2, got shape {tuple(residuals.shape)}.")
    if residuals.shape[0] <= 0 or residuals.shape[1] <= 0:
        raise ValueError(f"residuals must be non-empty, got shape {tuple(residuals.shape)}.")
    if not torch.isfinite(residuals).all():
        raise ValueError("residuals must be finite.")
    working = residuals.detach().float().cpu()
    working = working - working.mean(dim=0, keepdim=True)
    component_count = min(2, working.shape[0], working.shape[1])
    _u, _s, vh = torch.linalg.svd(working, full_matrices=False)
    coordinates = working @ vh[:component_count].transpose(0, 1)
    if component_count == 2:
        return coordinates
    padding = coordinates.new_zeros((coordinates.shape[0], 1))
    return torch.cat((coordinates, padding), dim=1)


def build_report(
    *,
    model_dir: Path,
    prompt_file: Path,
    prompt_name: str,
    concept: str,
    control_label: str,
    selected_rows: list[dict[str, Any]],
    layer_results: dict[int, dict[str, Any]],
    scene_path: Path,
) -> dict[str, Any]:
    layer_reports: list[dict[str, Any]] = []
    for layer, result in sorted(layer_results.items()):
        layer_reports.append(
            {
                "layer_index": layer,
                "direction_norm": float(result["direction_norm"]),
                "concept_score_mean": float(result["concept_score_mean"]),
                "control_score_mean": float(result["control_score_mean"]),
                "concept_score_std": float(result["concept_score_std"]),
                "control_score_std": float(result["control_score_std"]),
                "threshold": float(result["threshold"]),
                "linear_separation_accuracy": float(result["linear_separation_accuracy"]),
                "effect_size": finite_or_string(result["effect_size"]),
                "concept_count": int(result["concept_count"]),
                "control_count": int(result["control_count"]),
                "top_positive_neurons": result["top_positive_neurons"],
                "top_negative_neurons": result["top_negative_neurons"],
            }
        )
    best_layer = max(layer_reports, key=lambda item: (item["linear_separation_accuracy"], item["direction_norm"]))
    return {
        "model_dir": str(model_dir),
        "prompt_file": str(prompt_file),
        "prompt_name": prompt_name,
        "scene_path": str(scene_path),
        "concept": concept,
        "control_label": control_label,
        "selected_row_count": len(selected_rows),
        "selected_prompt_ids": sorted({str(row["prompt_id"]) for row in selected_rows}),
        "method": {
            "name": "contrastive_mean_direction",
            "formula": "direction_l = normalize(mean(hidden_l | concept) - mean(hidden_l | control))",
            "note": (
                "This estimates a linear concept direction. It does not prove that "
                "the feature is monosemantic or free of superposition."
            ),
        },
        "best_layer": best_layer,
        "layers": layer_reports,
    }


def write_feature_html(*, scene: dict[str, Any], report: dict[str, Any], output_html: Path) -> None:
    import plotly.graph_objects as go

    output_html.parent.mkdir(parents=True, exist_ok=True)
    nodes = scene["nodes"]
    edges = scene["edges"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("scene nodes must be a non-empty list.")
    if not isinstance(edges, list):
        raise TypeError("scene edges must be a list.")
    sample_nodes = [node for node in nodes if node["kind"] == "feature_sample"]
    endpoint_nodes = [node for node in nodes if node["kind"] == "feature_direction_endpoint"]
    if not sample_nodes:
        raise ValueError("scene must contain feature_sample nodes.")
    node_by_id = {str(node["id"]): node for node in nodes}

    traces: list[Any] = []
    for label_value, name, color in ((0.0, str(report["control_label"]), "#64748b"), (1.0, str(report["concept"]), "#dc2626")):
        group = [node for node in sample_nodes if float(node["color_value"]) == label_value]
        traces.append(
            go.Scatter3d(
                x=[node["position"][0] for node in group],
                y=[node["position"][1] for node in group],
                z=[node["position"][2] for node in group],
                mode="markers",
                marker={"size": [float(node["size"]) * 5 for node in group], "color": color, "opacity": 0.72},
                text=[feature_hover_text(node) for node in group],
                hoverinfo="text",
                name=name,
            )
        )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for edge in edges:
        if edge.get("kind") != "concept_direction":
            continue
        source = node_by_id[str(edge["source"])]
        target = node_by_id[str(edge["target"])]
        edge_x.extend([source["position"][0], target["position"][0], None])
        edge_y.extend([source["position"][1], target["position"][1], None])
        edge_z.extend([source["position"][2], target["position"][2], None])
    traces.append(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={"color": "#111827", "width": 5},
            hoverinfo="skip",
            name="concept direction",
        )
    )
    traces.append(
        go.Scatter3d(
            x=[node["position"][0] for node in endpoint_nodes],
            y=[node["position"][1] for node in endpoint_nodes],
            z=[node["position"][2] for node in endpoint_nodes],
            mode="markers",
            marker={
                "size": [float(node["size"]) * 5 for node in endpoint_nodes],
                "color": ["#64748b" if float(node["color_value"]) == 0.0 else "#dc2626" for node in endpoint_nodes],
                "symbol": "diamond",
                "opacity": 0.98,
            },
            text=[feature_hover_text(node) for node in endpoint_nodes],
            hoverinfo="text",
            name="centroids",
        )
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{report['concept']} feature direction vs {report['control_label']}",
        scene={
            "xaxis_title": f"{report['concept']} contrast score",
            "yaxis_title": "layer + residual PC1",
            "zaxis_title": "residual PC2",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)


def feature_hover_text(node: dict[str, Any]) -> str:
    metadata = node["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("node metadata must be a dict.")
    lines = [
        f"id={node['id']}",
        f"kind={node['kind']}",
        f"label={metadata.get('label', '')}",
        f"layer={metadata['layer_index']}",
    ]
    if "token_text" in metadata:
        lines.extend(
            [
                f"prompt={metadata['prompt_id']}",
                f"token={metadata['token_text']}",
                f"score={float(metadata['concept_score']):.4f}",
            ]
        )
    if "direction_norm" in metadata:
        lines.append(f"direction_norm={float(metadata['direction_norm']):.4f}")
    if "linear_separation_accuracy" in metadata:
        lines.append(f"linear_acc={float(metadata['linear_separation_accuracy']):.4f}")
    if "effect_size" in metadata:
        lines.append(f"effect_size={metadata['effect_size']}")
    return "<br>".join(lines)


def load_labeled_prompt_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("prompt file root must be a JSON object.")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("prompt file must contain a non-empty name.")
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("prompt file must contain a non-empty examples list.")
    seen_ids: set[str] = set()
    for idx, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"examples[{idx}] must be an object.")
        for key in ("id", "label", "text"):
            value = example.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"examples[{idx}].{key} must be a non-empty string.")
        prompt_id = str(example["id"])
        if prompt_id in seen_ids:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)
    return data


def sanitize_metadata(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in row.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings.")
        if isinstance(value, bool):
            sanitized[key] = value
        elif isinstance(value, int):
            sanitized[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"metadata value for {key!r} must be finite.")
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = value
        else:
            raise TypeError(
                f"metadata value for {key!r} must be str, int, float, or bool, "
                f"got {type(value).__name__}."
            )
    return sanitized


def finite_or_string(value: float) -> float | str:
    if math.isfinite(float(value)):
        return float(value)
    return str(value)


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {args.model_dir}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"model-dir is not a directory: {args.model_dir}")
    if not args.prompts.exists():
        raise FileNotFoundError(f"prompts file does not exist: {args.prompts}")
    if args.output_scene_json.exists():
        raise FileExistsError(f"output-scene-json already exists: {args.output_scene_json}")
    if args.output_report_json.exists():
        raise FileExistsError(f"output-report-json already exists: {args.output_report_json}")
    if args.output_html is not None and args.output_html.exists():
        raise FileExistsError(f"output-html already exists: {args.output_html}")
    if args.max_length <= 0:
        raise ValueError(f"max-length must be positive, got {args.max_length}.")
    if args.top_neurons <= 0:
        raise ValueError(f"top-neurons must be positive, got {args.top_neurons}.")
    if args.layer_gap <= 0:
        raise ValueError(f"layer-gap must be positive, got {args.layer_gap}.")


if __name__ == "__main__":
    main()
