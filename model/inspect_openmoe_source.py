from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive OpenMoE architecture paths from local config/source files."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {args.model_dir}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"model-dir is not a directory: {args.model_dir}")
    if args.output_json is not None and args.output_json.exists():
        raise FileExistsError(f"output-json already exists: {args.output_json}")

    config_path = args.model_dir / "config.json"
    source_path = args.model_dir / "modeling_openmoe.py"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"missing modeling_openmoe.py: {source_path}")

    config = json.loads(config_path.read_text())
    source_text = source_path.read_text()
    _require_source_marker(source_text, "class OpenMoeModel")
    _require_source_marker(source_text, "class OpenMoeDecoderLayer")
    _require_source_marker(source_text, "SparseMLP")

    inspection = build_static_inspection(config)
    print(json.dumps(inspection, indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n")
        print(f"wrote_json={args.output_json}")


def build_static_inspection(config: dict[str, Any]) -> dict[str, Any]:
    num_layers = _positive_int(config, "num_hidden_layers")
    interval = _positive_int(config, "moe_layer_interval")
    hidden_size = _positive_int(config, "hidden_size")
    intermediate_size = _positive_int(config, "intermediate_size")
    num_heads = _positive_int(config, "num_attention_heads")
    head_dim = _positive_int(config, "head_dim")
    num_experts = _positive_int(config, "num_experts")
    router_topk = _positive_int(config, "router_topk")

    layers = []
    activation_spaces = [
        {"name": "token_embedding", "path": "model.embed_tokens", "shape": ["batch", "seq", hidden_size]},
        {"name": "final_norm", "path": "model.norm", "shape": ["batch", "seq", hidden_size]},
        {"name": "lm_head", "path": "lm_head", "shape": ["batch", "seq", config["vocab_size"]]},
    ]
    attention_ops = []
    dense_mlp_ops = []
    moe_layers = []

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        is_moe = (layer_idx + 1) % interval == 0
        layers.append(
            {
                "layer": layer_idx,
                "path": prefix,
                "is_moe": is_moe,
                "attention_path": f"{prefix}.self_attn",
                "mlp_path": f"{prefix}.mlp",
            }
        )
        activation_spaces.extend(
            [
                {
                    "name": f"layer_{layer_idx}_input_norm",
                    "path": f"{prefix}.input_layernorm",
                    "shape": ["batch", "seq", hidden_size],
                },
                {
                    "name": f"layer_{layer_idx}_post_attention_norm",
                    "path": f"{prefix}.post_attention_layernorm",
                    "shape": ["batch", "seq", hidden_size],
                },
            ]
        )
        attention_ops.extend(
            [
                _linear_op(f"{prefix}.self_attn.q_proj", hidden_size, num_heads * head_dim),
                _linear_op(f"{prefix}.self_attn.k_proj", hidden_size, num_heads * head_dim),
                _linear_op(f"{prefix}.self_attn.v_proj", hidden_size, num_heads * head_dim),
                _linear_op(f"{prefix}.self_attn.o_proj", num_heads * head_dim, hidden_size),
            ]
        )
        if is_moe:
            moe_layers.append(
                {
                    "layer": layer_idx,
                    "sparse_mlp_path": f"{prefix}.mlp",
                    "num_experts": num_experts,
                    "router_topk": router_topk,
                    "internal_router_paths": "requires runtime ColossalAI module inspection",
                    "internal_expert_paths": "requires runtime ColossalAI module inspection",
                    "shared_extra_mlp_path": f"{prefix}.extra_mlp",
                }
            )
            dense_mlp_ops.extend(
                [
                    _linear_op(f"{prefix}.extra_mlp.gate_proj", hidden_size, intermediate_size * 2),
                    _linear_op(f"{prefix}.extra_mlp.up_proj", hidden_size, intermediate_size),
                    _linear_op(f"{prefix}.extra_mlp.down_proj", intermediate_size, hidden_size),
                ]
            )
        else:
            dense_mlp_ops.extend(
                [
                    _linear_op(f"{prefix}.mlp.gate_proj", hidden_size, intermediate_size * 2),
                    _linear_op(f"{prefix}.mlp.up_proj", hidden_size, intermediate_size),
                    _linear_op(f"{prefix}.mlp.down_proj", intermediate_size, hidden_size),
                ]
            )

    return {
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures"),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "num_experts": num_experts,
        "router_topk": router_topk,
        "moe_layer_interval": interval,
        "moe_layer_indices": [layer["layer"] for layer in moe_layers],
        "layers": layers,
        "activation_spaces": activation_spaces,
        "attention_linear_operators": attention_ops,
        "dense_mlp_linear_operators": dense_mlp_ops,
        "moe_layers": moe_layers,
        "runtime_blocker": {
            "reason": "AutoModelForCausalLM runtime loading requires colossalai and flash_attn.",
            "needed_for": [
                "exact SparseMLP router module paths",
                "exact expert module paths",
                "router logits",
                "top-k expert traces",
                "activation capture",
            ],
        },
    }


def _linear_op(path: str, in_features: int, out_features: int) -> dict[str, Any]:
    return {
        "path": path,
        "weight_shape": [out_features, in_features],
        "input_dim": in_features,
        "output_dim": out_features,
    }


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"config field {key!r} must be a positive int, got {value!r}.")
    return value


def _require_source_marker(source_text: str, marker: str) -> None:
    if marker not in source_text:
        raise ValueError(f"expected marker not found in modeling_openmoe.py: {marker}")


if __name__ == "__main__":
    main()
