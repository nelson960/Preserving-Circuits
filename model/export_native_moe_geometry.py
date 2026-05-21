from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture hidden activation vectors from a Transformers-native MoE model "
            "and export a 3D latent geometry scene."
        )
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-kind", required=True, choices=("causal-lm", "seq2seq-lm"))
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output-scene-json", required=True, type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--output-activations-pt", type=Path)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--color-by", choices=("layer", "norm", "expert", "prompt"), default="expert")
    parser.add_argument("--html-edges", choices=("none", "layer-flow"), default="layer-flow")
    args = parser.parse_args()

    validate_args(args)
    set_hf_home(args.hf_home)

    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    device = device_from_name(args.device)
    dtype = dtype_from_name(args.dtype)
    prompt_set = load_prompt_set(args.prompts)

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
    coordinates = project_vectors_svd_3d(capture["vectors"])
    scene = build_scene(
        coordinates=coordinates,
        metadata_rows=capture["metadata_rows"],
        color_by=args.color_by,
        scene_name=f"{args.model_dir.name}:{prompt_set['name']}:hidden_activations",
    )

    args.output_scene_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_scene_json.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n")
    print(f"wrote_scene_json={args.output_scene_json}")

    if args.output_activations_pt is not None:
        args.output_activations_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vectors": capture["vectors"],
                "coordinates": coordinates,
                "metadata_rows": capture["metadata_rows"],
                "router_paths": router_paths,
                "scene_json": str(args.output_scene_json),
            },
            args.output_activations_pt,
        )
        print(f"wrote_activations_pt={args.output_activations_pt}")

    if args.output_html is not None:
        write_plotly_html(
            scene=scene,
            output_html=args.output_html,
            color_by=args.color_by,
            edge_mode=args.html_edges,
        )
        print(f"wrote_html={args.output_html}")


def capture_activation_geometry_inputs(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_set: dict[str, Any],
    device: torch.device,
    max_length: int,
    model_kind: str,
    router_paths: list[str],
) -> dict[str, Any]:
    vectors: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    prompt_examples = prompt_set["examples"]
    for prompt_index, example in enumerate(prompt_examples):
        prompt_id = require_str(example, "id")
        text = require_str(example, "text")
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model_inputs = dict(encoded)
        if model_kind == "seq2seq-lm":
            start_id = getattr(model.config, "decoder_start_token_id", None)
            if start_id is None:
                start_id = getattr(model.config, "pad_token_id", None)
            if not isinstance(start_id, int):
                raise ValueError("seq2seq model config must expose decoder_start_token_id or pad_token_id.")
            model_inputs["decoder_input_ids"] = torch.tensor([[start_id]], device=device)

        with torch.no_grad():
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                output_router_logits=True,
                return_dict=True,
            )

        hidden_states = get_prompt_aligned_hidden_states(outputs, model_kind=model_kind)
        if hidden_states is None:
            raise RuntimeError("model returned no prompt-aligned hidden states.")

        input_ids = encoded["input_ids"][0].detach().cpu().tolist()
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        seq_len = len(input_ids)
        router_infos = build_router_infos(
            outputs=outputs,
            model_kind=model_kind,
            seq_len=seq_len,
            router_paths=router_paths,
        )

        for layer_index, hidden in enumerate(hidden_states):
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != seq_len:
                raise ValueError(
                    "hidden state must have shape [1, seq, hidden_dim], "
                    f"got layer={layer_index} shape={list(hidden.shape)}."
                )
            hidden_cpu = hidden[0].detach().float().cpu()
            router_for_layer = router_infos["by_hidden_layer"].get(layer_index)
            for token_index in range(seq_len):
                vector = hidden_cpu[token_index]
                norm = float(torch.linalg.vector_norm(vector).item())
                row: dict[str, Any] = {
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "text": text,
                    "token_index": token_index,
                    "token_id": int(input_ids[token_index]),
                    "token_text": token_texts[token_index],
                    "layer_index": layer_index,
                    "activation_norm": norm,
                    "has_router": False,
                }
                if router_for_layer is not None:
                    router = router_for_layer["router"]
                    probs = torch.softmax(router[token_index].float(), dim=-1)
                    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
                    top_prob, top_expert = torch.max(probs, dim=-1)
                    row.update(
                        {
                            "has_router": True,
                            "router_index": int(router_for_layer["router_index"]),
                            "router_path": str(router_for_layer["router_path"]),
                            "router_layer_index": int(router_for_layer["router_layer_index"]),
                            "top_expert": int(top_expert.item()),
                            "top_expert_prob": float(top_prob.item()),
                            "router_entropy": entropy,
                        }
                    )
                vectors.append(vector)
                metadata_rows.append(row)

    if not vectors:
        raise RuntimeError("no activation vectors were captured.")
    return {
        "vectors": torch.stack(vectors, dim=0),
        "metadata_rows": metadata_rows,
    }


def build_router_infos(
    *,
    outputs: Any,
    model_kind: str,
    seq_len: int,
    router_paths: list[str],
) -> dict[str, Any]:
    raw_router_logits = get_prompt_aligned_router_logits(outputs, model_kind=model_kind)
    if raw_router_logits is None:
        if router_paths:
            raise RuntimeError("model has router modules, but forward pass returned no router logits.")
        return {"by_hidden_layer": {}}
    router_tensors = normalize_router_logits(raw_router_logits, seq_len=seq_len)
    if len(router_tensors) != len(router_paths):
        raise RuntimeError(
            "router tensor count does not match discovered router path count: "
            f"tensors={len(router_tensors)} paths={len(router_paths)}."
        )

    by_hidden_layer: dict[int, dict[str, Any]] = {}
    for router_index, (router_path, router) in enumerate(zip(router_paths, router_tensors, strict=True)):
        router_layer = infer_layer_index_from_router_path(router_path)
        if router_layer is None:
            continue
        hidden_layer = router_layer + 1
        if hidden_layer in by_hidden_layer:
            raise RuntimeError(
                f"multiple routers mapped to hidden layer {hidden_layer}: "
                f"{by_hidden_layer[hidden_layer]['router_path']} and {router_path}."
            )
        by_hidden_layer[hidden_layer] = {
            "router_index": router_index,
            "router_path": router_path,
            "router_layer_index": router_layer,
            "router": router,
        }
    return {"by_hidden_layer": by_hidden_layer}


def build_scene(
    *,
    coordinates: torch.Tensor,
    metadata_rows: list[dict[str, Any]],
    color_by: str,
    scene_name: str,
) -> dict[str, Any]:
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"coordinates must have shape [N, 3], got {tuple(coordinates.shape)}.")
    if coordinates.shape[0] != len(metadata_rows):
        raise ValueError("coordinate row count must match metadata row count.")

    nodes: list[dict[str, Any]] = []
    for row_index, row in enumerate(metadata_rows):
        coordinate = coordinates[row_index].detach().cpu().tolist()
        color_value = color_value_for_row(row, color_by=color_by)
        node_id = (
            f"activation:{row['prompt_id']}:{row['token_index']}:"
            f"{row['layer_index']}"
        )
        label = f"{row['prompt_id']} L{row['layer_index']} {row['token_text']}"
        nodes.append(
            {
                "id": node_id,
                "kind": "activation_sample",
                "position": [float(coordinate[0]), float(coordinate[1]), float(coordinate[2])],
                "label": label,
                "layer": int(row["layer_index"]),
                "space": "encoder_hidden",
                "size": max(0.3, min(1.6, 0.35 + float(row["activation_norm"]) / 350.0)),
                "color_value": color_value,
                "metadata": sanitize_metadata(row),
            }
        )

    edges: list[dict[str, Any]] = []
    rows_by_path: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in metadata_rows:
        key = (str(row["prompt_id"]), int(row["token_index"]))
        rows_by_path.setdefault(key, []).append(row)
    for rows in rows_by_path.values():
        ordered = sorted(rows, key=lambda item: int(item["layer_index"]))
        for source, target in zip(ordered[:-1], ordered[1:], strict=True):
            source_id = f"activation:{source['prompt_id']}:{source['token_index']}:{source['layer_index']}"
            target_id = f"activation:{target['prompt_id']}:{target['token_index']}:{target['layer_index']}"
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "kind": "layer_flow",
                    "weight": 1.0,
                    "label": "same_token_next_layer",
                    "metadata": {
                        "prompt_id": str(source["prompt_id"]),
                        "token_index": int(source["token_index"]),
                        "source_layer": int(source["layer_index"]),
                        "target_layer": int(target["layer_index"]),
                    },
                }
            )

    return {
        "name": scene_name,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "scene_type": "representation_geometry",
            "point_semantics": "token_layer_activation_row",
            "projection": "svd_3d",
            "color_by": color_by,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


def write_plotly_html(
    *,
    scene: dict[str, Any],
    output_html: Path,
    color_by: str,
    edge_mode: str,
) -> None:
    import plotly.graph_objects as go

    output_html.parent.mkdir(parents=True, exist_ok=True)
    nodes = scene["nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("scene must contain a non-empty node list.")

    x = [node["position"][0] for node in nodes]
    y = [node["position"][1] for node in nodes]
    z = [node["position"][2] for node in nodes]
    color = [node["color_value"] for node in nodes]
    size = [float(node["size"]) * 4.5 for node in nodes]
    hover = [hover_text(node) for node in nodes]
    traces: list[Any] = []

    if edge_mode == "layer-flow":
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        node_by_id = {str(node["id"]): node for node in nodes}
        edges = scene["edges"]
        if not isinstance(edges, list):
            raise TypeError("scene edges must be a list.")
        for edge in edges:
            if edge.get("kind") != "layer_flow":
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
                line={"color": "rgba(120,120,120,0.22)", "width": 1},
                hoverinfo="skip",
                name="token layer flow",
            )
        )
    elif edge_mode != "none":
        raise ValueError(f"unsupported edge_mode: {edge_mode}")

    traces.append(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={
                "size": size,
                "color": color,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": color_by},
                "opacity": 0.84,
            },
            text=hover,
            hoverinfo="text",
            name="token-layer activations",
        )
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=str(scene["name"]),
        scene={
            "xaxis_title": "SVD-1",
            "yaxis_title": "SVD-2",
            "zaxis_title": "SVD-3",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)


def hover_text(node: dict[str, Any]) -> str:
    metadata = node["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("node metadata must be a dict.")
    lines = [
        f"id={node['id']}",
        f"prompt={metadata['prompt_id']}",
        f"token={metadata['token_index']} {metadata['token_text']}",
        f"layer={metadata['layer_index']}",
        f"norm={float(metadata['activation_norm']):.4f}",
    ]
    if metadata.get("has_router") is True:
        lines.extend(
            [
                f"router={metadata['router_path']}",
                f"top_expert={metadata['top_expert']}",
                f"top_prob={float(metadata['top_expert_prob']):.4f}",
                f"entropy={float(metadata['router_entropy']):.4f}",
            ]
        )
    return "<br>".join(lines)


def color_value_for_row(row: dict[str, Any], *, color_by: str) -> float:
    if color_by == "layer":
        return float(row["layer_index"])
    if color_by == "norm":
        return float(row["activation_norm"])
    if color_by == "prompt":
        return float(row["prompt_index"])
    if color_by == "expert":
        if row.get("has_router") is True:
            return float(row["top_expert"])
        return -1.0
    raise ValueError(f"unsupported color_by: {color_by}")


def project_vectors_svd_3d(vectors: torch.Tensor) -> torch.Tensor:
    if not isinstance(vectors, torch.Tensor):
        raise TypeError(f"vectors must be a torch.Tensor, got {type(vectors).__name__}.")
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be rank-2, got shape {tuple(vectors.shape)}.")
    if vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        raise ValueError(f"vectors must be non-empty, got shape {tuple(vectors.shape)}.")
    if not torch.is_floating_point(vectors):
        raise TypeError(f"vectors must be floating point, got {vectors.dtype}.")
    if not torch.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite values.")
    working = vectors.detach().float().cpu()
    working = working - working.mean(dim=0, keepdim=True)
    component_count = min(3, working.shape[0], working.shape[1])
    _u, _s, vh = torch.linalg.svd(working, full_matrices=False)
    coordinates = working @ vh[:component_count].transpose(0, 1)
    if component_count == 3:
        return coordinates
    padding = coordinates.new_zeros((coordinates.shape[0], 3 - component_count))
    return torch.cat((coordinates, padding), dim=1)


def normalize_router_logits(router_logits: Any, *, seq_len: int) -> list[torch.Tensor]:
    raw = extract_router_logit_tensors(router_logits, seq_len=seq_len)
    if not raw:
        raise RuntimeError(
            "router output was present, but no prompt token-level router-logit tensors matched "
            f"sequence length {seq_len}."
        )
    normalized: list[torch.Tensor] = []
    for router in raw:
        if router.ndim == 3 and router.shape[0] == 1 and router.shape[1] == seq_len:
            normalized.append(router[0].detach().float().cpu())
        elif router.ndim == 2 and router.shape[0] == seq_len:
            normalized.append(router.detach().float().cpu())
        else:
            raise ValueError(f"unsupported router logit shape: {list(router.shape)}.")
    return normalized


def extract_router_logit_tensors(router_logits: Any, *, seq_len: int) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for tensor in iter_tensors(router_logits):
        if tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] == seq_len:
            tensors.append(tensor)
        elif tensor.ndim == 2 and tensor.shape[0] == seq_len:
            tensors.append(tensor)
    return tensors


def iter_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(iter_tensors(item))
        return tensors
    return []


def collect_router_paths(model: torch.nn.Module, *, model_kind: str) -> list[str]:
    paths: list[str] = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if "router" not in class_name:
            continue
        if model_kind == "seq2seq-lm":
            if name.startswith("encoder."):
                paths.append(name)
        else:
            paths.append(name)
    return paths


def infer_layer_index_from_router_path(path: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:block|layers)\.(\d+)(?:\.|$)", path)
    if match is None:
        return None
    return int(match.group(1))


def get_prompt_aligned_hidden_states(outputs: Any, *, model_kind: str) -> Any:
    if model_kind == "seq2seq-lm":
        return getattr(outputs, "encoder_hidden_states", None)
    return getattr(outputs, "hidden_states", None)


def get_prompt_aligned_router_logits(outputs: Any, *, model_kind: str) -> Any:
    if model_kind == "seq2seq-lm":
        return getattr(outputs, "encoder_router_logits", None)
    return getattr(outputs, "router_logits", None)


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


def load_prompt_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("prompt file root must be a JSON object.")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("prompt file must contain a non-empty name.")
    examples = data.get("examples")
    if not isinstance(examples, list) or len(examples) == 0:
        raise ValueError("prompt file must contain a non-empty examples list.")
    for idx, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"examples[{idx}] must be an object.")
        require_str(example, "id")
        require_str(example, "text")
    return data


def require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} must be a non-empty string.")
    return value


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {args.model_dir}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"model-dir is not a directory: {args.model_dir}")
    if not args.prompts.exists():
        raise FileNotFoundError(f"prompts file does not exist: {args.prompts}")
    if args.output_scene_json.exists():
        raise FileExistsError(f"output-scene-json already exists: {args.output_scene_json}")
    if args.output_html is not None and args.output_html.exists():
        raise FileExistsError(f"output-html already exists: {args.output_html}")
    if args.output_activations_pt is not None and args.output_activations_pt.exists():
        raise FileExistsError(f"output-activations-pt already exists: {args.output_activations_pt}")
    if args.max_length <= 0:
        raise ValueError(f"max-length must be positive, got {args.max_length}.")


def set_hf_home(hf_home: Path | None) -> None:
    if hf_home is None:
        return
    if hf_home.exists() and not hf_home.is_dir():
        raise NotADirectoryError(f"hf-home is not a directory: {hf_home}")
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home.resolve())


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def device_from_name(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false.")
    return torch.device(name)


if __name__ == "__main__":
    main()
