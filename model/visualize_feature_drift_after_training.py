from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch

from export_native_moe_geometry import (
    capture_activation_geometry_inputs,
    collect_router_paths,
    device_from_name,
    dtype_from_name,
    set_hf_home,
)
from export_concept_feature_geometry import (
    load_labeled_prompt_set,
    project_residuals_2d,
    sanitize_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a Transformers-native model on a sequential task and visualize "
            "before/after drift for selected contrastive concept feature directions."
        )
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-kind", required=True, choices=("causal-lm", "seq2seq-lm"))
    parser.add_argument("--feature-prompts", required=True, type=Path)
    parser.add_argument("--train-prompts", required=True, type=Path)
    parser.add_argument("--concepts", required=True, help="Comma-separated labels, e.g. animal,vehicle")
    parser.add_argument(
        "--control-labels",
        required=True,
        help="Comma-separated labels used as hard negatives, e.g. place,vehicle,control",
    )
    parser.add_argument("--train-param-regex", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--layer-gap", type=float, default=9.0)
    parser.add_argument("--output-scene-json", required=True, type=Path)
    parser.add_argument("--output-report-json", required=True, type=Path)
    parser.add_argument("--output-html", type=Path)
    args = parser.parse_args()

    validate_args(args)
    set_hf_home(args.hf_home)

    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    concepts = parse_csv(args.concepts, "concepts")
    control_labels = parse_csv(args.control_labels, "control-labels")

    feature_prompt_set = load_labeled_prompt_set(args.feature_prompts)
    train_set = load_train_prompt_set(args.train_prompts)
    known_labels = {str(example["label"]) for example in feature_prompt_set["examples"]}
    missing = sorted(set(concepts + control_labels) - known_labels)
    if missing:
        raise ValueError(f"labels are not present in feature prompt file: {missing}")

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
    model.to(device)

    router_paths = collect_router_paths(model, model_kind=args.model_kind)
    before_capture = capture_eval_vectors(
        model=model,
        tokenizer=tokenizer,
        prompt_set=feature_prompt_set,
        concepts=concepts,
        control_labels=control_labels,
        device=device,
        max_length=args.max_length,
        model_kind=args.model_kind,
        router_paths=router_paths,
    )
    train_report = train_on_new_task(
        model=model,
        tokenizer=tokenizer,
        train_set=train_set,
        device=device,
        model_kind=args.model_kind,
        train_param_regex=args.train_param_regex,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    after_capture = capture_eval_vectors(
        model=model,
        tokenizer=tokenizer,
        prompt_set=feature_prompt_set,
        concepts=concepts,
        control_labels=control_labels,
        device=device,
        max_length=args.max_length,
        model_kind=args.model_kind,
        router_paths=router_paths,
    )

    drift = compute_feature_drift(
        before_vectors=before_capture["vectors"],
        before_rows=before_capture["rows"],
        after_vectors=after_capture["vectors"],
        after_rows=after_capture["rows"],
        concepts=concepts,
        control_labels=control_labels,
    )
    scene = build_drift_scene(
        before_vectors=before_capture["vectors"],
        before_rows=before_capture["rows"],
        after_vectors=after_capture["vectors"],
        after_rows=after_capture["rows"],
        drift=drift,
        concepts=concepts,
        control_labels=control_labels,
        scene_name=f"{args.model_dir.name}:feature_drift_after:{train_set['name']}",
        layer_gap=args.layer_gap,
    )
    report = {
        "model_dir": str(args.model_dir),
        "feature_prompt_file": str(args.feature_prompts),
        "train_prompt_file": str(args.train_prompts),
        "concepts": concepts,
        "control_labels": control_labels,
        "train_param_regex": args.train_param_regex,
        "train": train_report,
        "method": {
            "feature_direction": "mean(concept hidden) - mean(control hidden), computed per layer",
            "old_feature_survival": "after examples projected onto the before-training direction",
            "rotation": "angle between before and after contrastive directions",
            "fading": "after direction norm divided by before direction norm",
        },
        "drift": drift_to_json(drift),
        "scene_path": str(args.output_scene_json),
    }

    args.output_scene_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_scene_json.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n")
    print(f"wrote_scene_json={args.output_scene_json}")

    args.output_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote_report_json={args.output_report_json}")

    if args.output_html is not None:
        write_drift_html(scene=scene, report=report, output_html=args.output_html)
        print(f"wrote_html={args.output_html}")


def capture_eval_vectors(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_set: dict[str, Any],
    concepts: list[str],
    control_labels: list[str],
    device: torch.device,
    max_length: int,
    model_kind: str,
    router_paths: list[str],
) -> dict[str, Any]:
    model.eval()
    capture = capture_activation_geometry_inputs(
        model=model,
        tokenizer=tokenizer,
        prompt_set=prompt_set,
        device=device,
        max_length=max_length,
        model_kind=model_kind,
        router_paths=router_paths,
    )
    label_by_prompt_id = {
        str(example["id"]): str(example["label"]) for example in prompt_set["examples"]
    }
    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    keep_labels = set(concepts + control_labels)
    vectors: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(capture["metadata_rows"]):
        prompt_id = str(row["prompt_id"])
        if prompt_id not in label_by_prompt_id:
            raise KeyError(f"missing label for prompt_id {prompt_id!r}.")
        label = label_by_prompt_id[prompt_id]
        if label not in keep_labels:
            continue
        if int(row["token_id"]) in special_ids:
            continue
        selected = dict(row)
        selected["label"] = label
        vectors.append(capture["vectors"][index].detach().float().cpu())
        rows.append(selected)
    if not vectors:
        raise RuntimeError("feature evaluation capture selected no non-special rows.")
    return {"vectors": torch.stack(vectors, dim=0), "rows": rows}


def train_on_new_task(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    train_set: dict[str, Any],
    device: torch.device,
    model_kind: str,
    train_param_regex: str,
    steps: int,
    lr: float,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    pattern = re.compile(train_param_regex)
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, param in model.named_parameters():
        should_train = pattern.search(name) is not None
        param.requires_grad_(should_train)
        if should_train:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    if not trainable_names:
        raise RuntimeError(f"train-param-regex matched no parameters: {train_param_regex!r}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    examples = train_set["examples"]
    losses: list[float] = []
    model.train()
    for step in range(steps):
        batch_examples = [
            examples[(step * batch_size + offset) % len(examples)] for offset in range(batch_size)
        ]
        inputs = [str(example["input"]) for example in batch_examples]
        targets = [str(example["target"]) for example in batch_examples]
        loss = compute_training_loss(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            targets=targets,
            device=device,
            model_kind=model_kind,
            max_length=max_length,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"training loss became non-finite at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "name": train_set["name"],
        "steps": steps,
        "lr": lr,
        "batch_size": batch_size,
        "trainable_parameter_count": int(sum(param.numel() for param in trainable_params)),
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_count": int(
            sum(param.numel() for name, param in model.named_parameters() if name in frozen_names)
        ),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses": losses,
    }


def compute_training_loss(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    inputs: list[str],
    targets: list[str],
    device: torch.device,
    model_kind: str,
    max_length: int,
) -> torch.Tensor:
    if model_kind == "seq2seq-lm":
        encoded_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        encoded_targets = tokenizer(
            targets,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        labels = encoded_targets["input_ids"].clone()
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("seq2seq tokenizer must expose pad_token_id.")
        labels[labels == pad_id] = -100
        encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
        labels = labels.to(device)
        outputs = model(**encoded_inputs, labels=labels, return_dict=True)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError("seq2seq model did not return a loss.")
        return loss

    encoded = tokenizer(
        [f"{source} {target}" for source, target in zip(inputs, targets, strict=True)],
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    labels = encoded["input_ids"].clone()
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = labels.to(device)
    outputs = model(**encoded, labels=labels, return_dict=True)
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError("causal LM did not return a loss.")
    return loss


def compute_feature_drift(
    *,
    before_vectors: torch.Tensor,
    before_rows: list[dict[str, Any]],
    after_vectors: torch.Tensor,
    after_rows: list[dict[str, Any]],
    concepts: list[str],
    control_labels: list[str],
) -> dict[str, Any]:
    verify_row_alignment(before_rows, after_rows)
    layers = sorted({int(row["layer_index"]) for row in before_rows})
    drift: dict[str, Any] = {}
    for concept in concepts:
        concept_result: dict[int, dict[str, Any]] = {}
        for layer in layers:
            indices = [idx for idx, row in enumerate(before_rows) if int(row["layer_index"]) == layer]
            concept_indices = [idx for idx in indices if before_rows[idx]["label"] == concept]
            control_indices = [
                idx
                for idx in indices
                if before_rows[idx]["label"] in control_labels and before_rows[idx]["label"] != concept
            ]
            if not concept_indices or not control_indices:
                raise RuntimeError(
                    f"layer {layer} lacks concept/control rows for {concept}: "
                    f"concept={len(concept_indices)} control={len(control_indices)}"
                )
            before_stats = feature_stats_for_indices(before_vectors, concept_indices, control_indices)
            after_stats = feature_stats_for_indices(after_vectors, concept_indices, control_indices)
            cosine = float(torch.dot(before_stats["direction"], after_stats["direction"]).clamp(-1, 1).item())
            angle = float(math.degrees(math.acos(cosine)))
            norm_ratio = float(after_stats["direction_norm"] / before_stats["direction_norm"])
            before_old_axis = score_stats_on_reference_direction(
                vectors=before_vectors,
                concept_indices=concept_indices,
                control_indices=control_indices,
                reference_direction=before_stats["direction"],
                reference_control_mean=before_stats["control_mean"],
            )
            after_old_axis = score_stats_on_reference_direction(
                vectors=after_vectors,
                concept_indices=concept_indices,
                control_indices=control_indices,
                reference_direction=before_stats["direction"],
                reference_control_mean=before_stats["control_mean"],
            )
            concept_centroid_shift = float(
                torch.linalg.vector_norm(after_stats["concept_mean"] - before_stats["concept_mean"]).item()
            )
            control_centroid_shift = float(
                torch.linalg.vector_norm(after_stats["control_mean"] - before_stats["control_mean"]).item()
            )
            concept_result[layer] = {
                "layer_index": layer,
                "before_direction": before_stats["direction"],
                "after_direction": after_stats["direction"],
                "before_control_mean": before_stats["control_mean"],
                "before_concept_mean": before_stats["concept_mean"],
                "after_control_mean": after_stats["control_mean"],
                "after_concept_mean": after_stats["concept_mean"],
                "before_direction_norm": before_stats["direction_norm"],
                "after_direction_norm": after_stats["direction_norm"],
                "norm_ratio": norm_ratio,
                "direction_cosine": cosine,
                "rotation_degrees": angle,
                "before_old_axis": before_old_axis,
                "after_old_axis": after_old_axis,
                "old_axis_margin_delta": after_old_axis["margin"] - before_old_axis["margin"],
                "old_axis_accuracy_delta": after_old_axis["accuracy"] - before_old_axis["accuracy"],
                "concept_centroid_shift": concept_centroid_shift,
                "control_centroid_shift": control_centroid_shift,
            }
        drift[concept] = concept_result
    return drift


def feature_stats_for_indices(
    vectors: torch.Tensor,
    concept_indices: list[int],
    control_indices: list[int],
) -> dict[str, Any]:
    concept_vectors = vectors[concept_indices]
    control_vectors = vectors[control_indices]
    concept_mean = concept_vectors.mean(dim=0)
    control_mean = control_vectors.mean(dim=0)
    raw_direction = concept_mean - control_mean
    direction_norm = float(torch.linalg.vector_norm(raw_direction).item())
    if direction_norm <= 0.0:
        raise RuntimeError("feature direction has zero norm.")
    direction = raw_direction / direction_norm
    return {
        "concept_mean": concept_mean,
        "control_mean": control_mean,
        "direction": direction,
        "direction_norm": direction_norm,
    }


def score_stats_on_reference_direction(
    *,
    vectors: torch.Tensor,
    concept_indices: list[int],
    control_indices: list[int],
    reference_direction: torch.Tensor,
    reference_control_mean: torch.Tensor,
) -> dict[str, float]:
    concept_scores = (vectors[concept_indices] - reference_control_mean) @ reference_direction
    control_scores = (vectors[control_indices] - reference_control_mean) @ reference_direction
    threshold = 0.5 * (float(concept_scores.mean().item()) + float(control_scores.mean().item()))
    concept_correct = int((concept_scores > threshold).sum().item())
    control_correct = int((control_scores <= threshold).sum().item())
    accuracy = float((concept_correct + control_correct) / (len(concept_indices) + len(control_indices)))
    return {
        "concept_mean": float(concept_scores.mean().item()),
        "control_mean": float(control_scores.mean().item()),
        "margin": float(concept_scores.mean().item() - control_scores.mean().item()),
        "threshold": threshold,
        "accuracy": accuracy,
    }


def build_drift_scene(
    *,
    before_vectors: torch.Tensor,
    before_rows: list[dict[str, Any]],
    after_vectors: torch.Tensor,
    after_rows: list[dict[str, Any]],
    drift: dict[str, Any],
    concepts: list[str],
    control_labels: list[str],
    scene_name: str,
    layer_gap: float,
) -> dict[str, Any]:
    verify_row_alignment(before_rows, after_rows)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    concept_z_gap = 55.0
    for concept_index, concept in enumerate(concepts):
        z_offset = concept_index * concept_z_gap
        positions_before, positions_after = feature_positions_for_concept(
            before_vectors=before_vectors,
            before_rows=before_rows,
            after_vectors=after_vectors,
            drift=drift[concept],
            concept=concept,
            control_labels=control_labels,
            layer_gap=layer_gap,
            z_offset=z_offset,
        )
        for row_index, row in enumerate(before_rows):
            if row["label"] != concept and row["label"] not in control_labels:
                continue
            if row["label"] == concept and row["label"] in control_labels:
                pass
            row_key = stable_row_key(row)
            before_id = f"before:{concept}:{row_key}"
            after_id = f"after:{concept}:{row_key}"
            is_concept = row["label"] == concept
            nodes.append(
                build_sample_node(
                    node_id=before_id,
                    row=row,
                    position=positions_before[row_index],
                    phase="before",
                    concept=concept,
                    is_concept=is_concept,
                )
            )
            nodes.append(
                build_sample_node(
                    node_id=after_id,
                    row=after_rows[row_index],
                    position=positions_after[row_index],
                    phase="after",
                    concept=concept,
                    is_concept=is_concept,
                )
            )
            edges.append(
                {
                    "source": before_id,
                    "target": after_id,
                    "kind": "training_drift",
                    "weight": float(torch.linalg.vector_norm(positions_after[row_index] - positions_before[row_index]).item()),
                    "label": "before_to_after",
                    "metadata": {
                        "concept": concept,
                        "prompt_id": str(row["prompt_id"]),
                        "token_index": int(row["token_index"]),
                        "layer_index": int(row["layer_index"]),
                        "label": str(row["label"]),
                    },
                }
            )

        for layer, layer_result in sorted(drift[concept].items()):
            y = layer * layer_gap
            before_control_id = f"before_direction:{concept}:{layer}:control"
            before_concept_id = f"before_direction:{concept}:{layer}:concept"
            after_control_id = f"after_direction:{concept}:{layer}:control"
            after_concept_id = f"after_direction:{concept}:{layer}:concept"
            before_norm = float(layer_result["before_direction_norm"])
            after_margin = float(layer_result["after_old_axis"]["margin"])
            nodes.extend(
                [
                    direction_endpoint(before_control_id, concept, layer, "before", "control", (0.0, y, z_offset - 8.0)),
                    direction_endpoint(before_concept_id, concept, layer, "before", "concept", (before_norm, y, z_offset - 8.0)),
                    direction_endpoint(after_control_id, concept, layer, "after", "control", (0.0, y, z_offset + 8.0)),
                    direction_endpoint(after_concept_id, concept, layer, "after", "concept", (after_margin, y, z_offset + 8.0)),
                ]
            )
            edges.append(
                direction_edge(before_control_id, before_concept_id, concept, layer, "before", layer_result)
            )
            edges.append(
                direction_edge(after_control_id, after_concept_id, concept, layer, "after", layer_result)
            )
    return {
        "name": scene_name,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "scene_type": "feature_drift_after_training",
            "concepts": ",".join(concepts),
            "control_labels": ",".join(control_labels),
            "point_semantics": "same concept/control hidden states before and after training",
            "x_axis": "old feature score",
            "y_axis": "layer plus residual PC1",
            "z_axis": "concept offset plus residual PC2 / before-after offset",
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


def feature_positions_for_concept(
    *,
    before_vectors: torch.Tensor,
    before_rows: list[dict[str, Any]],
    after_vectors: torch.Tensor,
    drift: dict[int, dict[str, Any]],
    concept: str,
    control_labels: list[str],
    layer_gap: float,
    z_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    before_residuals: list[torch.Tensor] = []
    after_residuals: list[torch.Tensor] = []
    before_scores: list[float] = []
    after_scores: list[float] = []
    layer_offsets: list[float] = []
    keep_indices: list[int] = []
    for row_index, row in enumerate(before_rows):
        if str(row["label"]) != concept and str(row["label"]) not in control_labels:
            continue
        layer = int(row["layer_index"])
        layer_result = drift[layer]
        direction = layer_result["before_direction"]
        control_mean = layer_result["before_control_mean"]
        before_centered = before_vectors[row_index] - control_mean
        after_centered = after_vectors[row_index] - control_mean
        before_score = float((before_centered @ direction).item())
        after_score = float((after_centered @ direction).item())
        before_residuals.append(before_centered - before_score * direction)
        after_residuals.append(after_centered - after_score * direction)
        before_scores.append(before_score)
        after_scores.append(after_score)
        layer_offsets.append(float(layer) * layer_gap)
        keep_indices.append(row_index)
    residual_coordinates = project_residuals_2d(torch.stack(before_residuals + after_residuals, dim=0))
    before_coordinates = residual_coordinates[: len(before_residuals)]
    after_coordinates = residual_coordinates[len(before_residuals) :]
    before_positions = torch.zeros((before_vectors.shape[0], 3), dtype=torch.float32)
    after_positions = torch.zeros((after_vectors.shape[0], 3), dtype=torch.float32)
    for local_index, row_index in enumerate(keep_indices):
        before_positions[row_index, 0] = before_scores[local_index]
        before_positions[row_index, 1] = layer_offsets[local_index] + before_coordinates[local_index, 0]
        before_positions[row_index, 2] = z_offset + before_coordinates[local_index, 1] - 4.0
        after_positions[row_index, 0] = after_scores[local_index]
        after_positions[row_index, 1] = layer_offsets[local_index] + after_coordinates[local_index, 0]
        after_positions[row_index, 2] = z_offset + after_coordinates[local_index, 1] + 4.0
    return before_positions, after_positions


def build_sample_node(
    *,
    node_id: str,
    row: dict[str, Any],
    position: torch.Tensor,
    phase: str,
    concept: str,
    is_concept: bool,
) -> dict[str, Any]:
    metadata = sanitize_metadata({**row, "phase": phase, "tracked_concept": concept})
    return {
        "id": node_id,
        "kind": "feature_drift_sample",
        "position": [float(value) for value in position.tolist()],
        "label": f"{phase} {concept} L{row['layer_index']} {row['token_text']}",
        "layer": int(row["layer_index"]),
        "space": "old_feature_axis",
        "size": 1.1 if is_concept else 0.7,
        "color_value": drift_color_value(phase=phase, is_concept=is_concept),
        "metadata": metadata,
    }


def drift_color_value(*, phase: str, is_concept: bool) -> float:
    if phase == "before" and is_concept:
        return 3.0
    if phase == "after" and is_concept:
        return 2.0
    if phase == "before":
        return 1.0
    return 0.0


def direction_endpoint(
    node_id: str,
    concept: str,
    layer: int,
    phase: str,
    endpoint: str,
    position: tuple[float, float, float],
) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": "feature_drift_direction_endpoint",
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "label": f"{phase} {concept} L{layer} {endpoint}",
        "layer": layer,
        "space": "old_feature_axis",
        "size": 2.2,
        "color_value": 3.5 if endpoint == "concept" else -0.5,
        "metadata": {
            "concept": concept,
            "phase": phase,
            "endpoint": endpoint,
            "layer_index": layer,
        },
    }


def direction_edge(
    source: str,
    target: str,
    concept: str,
    layer: int,
    phase: str,
    layer_result: dict[str, Any],
) -> dict[str, Any]:
    if phase == "before":
        weight = float(layer_result["before_direction_norm"])
    elif phase == "after":
        weight = float(layer_result["after_old_axis"]["margin"])
    else:
        raise ValueError(f"unsupported phase: {phase}")
    return {
        "source": source,
        "target": target,
        "kind": "feature_direction_before_after",
        "weight": weight,
        "label": f"{phase} {concept} L{layer}",
        "metadata": {
            "concept": concept,
            "phase": phase,
            "layer_index": layer,
            "direction_cosine": float(layer_result["direction_cosine"]),
            "rotation_degrees": float(layer_result["rotation_degrees"]),
            "norm_ratio": float(layer_result["norm_ratio"]),
            "old_axis_margin_delta": float(layer_result["old_axis_margin_delta"]),
            "old_axis_accuracy_delta": float(layer_result["old_axis_accuracy_delta"]),
        },
    }


def write_drift_html(*, scene: dict[str, Any], report: dict[str, Any], output_html: Path) -> None:
    import plotly.graph_objects as go

    output_html.parent.mkdir(parents=True, exist_ok=True)
    nodes = scene["nodes"]
    edges = scene["edges"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("scene nodes must be a non-empty list.")
    if not isinstance(edges, list):
        raise TypeError("scene edges must be a list.")
    node_by_id = {str(node["id"]): node for node in nodes}
    sample_nodes = [node for node in nodes if node["kind"] == "feature_drift_sample"]
    direction_nodes = [node for node in nodes if node["kind"] == "feature_drift_direction_endpoint"]

    traces: list[Any] = []
    color_specs = {
        3.0: ("before concept", "#dc2626"),
        2.0: ("after concept", "#f97316"),
        1.0: ("before control", "#475569"),
        0.0: ("after control", "#94a3b8"),
    }
    for color_value, (name, color) in color_specs.items():
        group = [node for node in sample_nodes if float(node["color_value"]) == color_value]
        if not group:
            continue
        traces.append(
            go.Scatter3d(
                x=[node["position"][0] for node in group],
                y=[node["position"][1] for node in group],
                z=[node["position"][2] for node in group],
                mode="markers",
                marker={"size": [float(node["size"]) * 4.5 for node in group], "color": color, "opacity": 0.72},
                text=[drift_hover_text(node) for node in group],
                hoverinfo="text",
                name=name,
            )
        )

    drift_edge_x: list[float | None] = []
    drift_edge_y: list[float | None] = []
    drift_edge_z: list[float | None] = []
    direction_edge_x: list[float | None] = []
    direction_edge_y: list[float | None] = []
    direction_edge_z: list[float | None] = []
    for edge in edges:
        source = node_by_id[str(edge["source"])]
        target = node_by_id[str(edge["target"])]
        target_lists = (
            (drift_edge_x, drift_edge_y, drift_edge_z)
            if edge["kind"] == "training_drift"
            else (direction_edge_x, direction_edge_y, direction_edge_z)
        )
        target_lists[0].extend([source["position"][0], target["position"][0], None])
        target_lists[1].extend([source["position"][1], target["position"][1], None])
        target_lists[2].extend([source["position"][2], target["position"][2], None])
    traces.append(
        go.Scatter3d(
            x=drift_edge_x,
            y=drift_edge_y,
            z=drift_edge_z,
            mode="lines",
            line={"color": "rgba(30,41,59,0.22)", "width": 1},
            hoverinfo="skip",
            name="sample drift",
        )
    )
    traces.append(
        go.Scatter3d(
            x=direction_edge_x,
            y=direction_edge_y,
            z=direction_edge_z,
            mode="lines",
            line={"color": "#111827", "width": 5},
            hoverinfo="skip",
            name="feature direction",
        )
    )
    traces.append(
        go.Scatter3d(
            x=[node["position"][0] for node in direction_nodes],
            y=[node["position"][1] for node in direction_nodes],
            z=[node["position"][2] for node in direction_nodes],
            mode="markers",
            marker={"size": [float(node["size"]) * 4.5 for node in direction_nodes], "color": "#111827", "symbol": "diamond"},
            text=[drift_hover_text(node) for node in direction_nodes],
            hoverinfo="text",
            name="centroids",
        )
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=str(scene["name"]),
        scene={
            "xaxis_title": "old feature score",
            "yaxis_title": "layer + residual PC1",
            "zaxis_title": "concept / before-after offset",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)


def drift_hover_text(node: dict[str, Any]) -> str:
    metadata = node["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("node metadata must be a dict.")
    lines = [
        f"id={node['id']}",
        f"kind={node['kind']}",
    ]
    for key in ("tracked_concept", "concept", "phase", "label", "prompt_id", "token_text", "layer_index"):
        if key in metadata:
            lines.append(f"{key}={metadata[key]}")
    return "<br>".join(lines)


def verify_row_alignment(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> None:
    if len(before_rows) != len(after_rows):
        raise ValueError("before and after row counts differ.")
    for idx, (before, after) in enumerate(zip(before_rows, after_rows, strict=True)):
        if stable_row_key(before) != stable_row_key(after):
            raise ValueError(
                f"before/after row order mismatch at index {idx}: "
                f"{stable_row_key(before)} != {stable_row_key(after)}"
            )


def stable_row_key(row: dict[str, Any]) -> str:
    return f"{row['prompt_id']}:{row['token_index']}:{row['layer_index']}:{row['label']}"


def drift_to_json(drift: dict[str, Any]) -> dict[str, list[dict[str, float | int]]]:
    payload: dict[str, list[dict[str, float | int]]] = {}
    for concept, layers in drift.items():
        payload[concept] = []
        for layer, result in sorted(layers.items()):
            payload[concept].append(
                {
                    "layer_index": int(layer),
                    "before_direction_norm": float(result["before_direction_norm"]),
                    "after_direction_norm": float(result["after_direction_norm"]),
                    "norm_ratio": float(result["norm_ratio"]),
                    "direction_cosine": float(result["direction_cosine"]),
                    "rotation_degrees": float(result["rotation_degrees"]),
                    "before_old_axis_margin": float(result["before_old_axis"]["margin"]),
                    "after_old_axis_margin": float(result["after_old_axis"]["margin"]),
                    "old_axis_margin_delta": float(result["old_axis_margin_delta"]),
                    "before_old_axis_accuracy": float(result["before_old_axis"]["accuracy"]),
                    "after_old_axis_accuracy": float(result["after_old_axis"]["accuracy"]),
                    "old_axis_accuracy_delta": float(result["old_axis_accuracy_delta"]),
                    "concept_centroid_shift": float(result["concept_centroid_shift"]),
                    "control_centroid_shift": float(result["control_centroid_shift"]),
                }
            )
    return payload


def load_train_prompt_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("train prompt file root must be a JSON object.")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("train prompt file must contain a non-empty name.")
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("train prompt file must contain a non-empty examples list.")
    for idx, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"train examples[{idx}] must be an object.")
        for key in ("input", "target"):
            value = example.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"train examples[{idx}].{key} must be a non-empty string.")
    return data


def parse_csv(value: str, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one item.")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} contains duplicates: {items}")
    return items


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {args.model_dir}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"model-dir is not a directory: {args.model_dir}")
    if not args.feature_prompts.exists():
        raise FileNotFoundError(f"feature-prompts does not exist: {args.feature_prompts}")
    if not args.train_prompts.exists():
        raise FileNotFoundError(f"train-prompts does not exist: {args.train_prompts}")
    if args.steps <= 0:
        raise ValueError(f"steps must be positive, got {args.steps}.")
    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}.")
    if args.max_length <= 0:
        raise ValueError(f"max-length must be positive, got {args.max_length}.")
    if args.layer_gap <= 0:
        raise ValueError(f"layer-gap must be positive, got {args.layer_gap}.")
    if args.output_scene_json.exists():
        raise FileExistsError(f"output-scene-json already exists: {args.output_scene_json}")
    if args.output_report_json.exists():
        raise FileExistsError(f"output-report-json already exists: {args.output_report_json}")
    if args.output_html is not None and args.output_html.exists():
        raise FileExistsError(f"output-html already exists: {args.output_html}")


if __name__ == "__main__":
    main()
