from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an OpenMoE/Hugging Face causal LM module tree."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-name-width", type=int, default=96)
    args = parser.parse_args()

    if args.max_name_width <= 0:
        raise ValueError(f"max-name-width must be positive, got {args.max_name_width}.")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {args.model_dir}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"model-dir is not a directory: {args.model_dir}")
    if args.output_json is not None and args.output_json.exists():
        raise FileExistsError(f"output-json already exists: {args.output_json}")
    if args.hf_home is not None:
        if args.hf_home.exists() and not args.hf_home.is_dir():
            raise NotADirectoryError(f"hf-home is not a directory: {args.hf_home}")
        args.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
        os.environ["TRANSFORMERS_CACHE"] = str((args.hf_home / "transformers").resolve())

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = _device(args.device)
    dtype = _dtype(args.dtype)

    config = AutoConfig.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)

    rows = inspect_modules(model)
    summary = build_summary(model, tokenizer, config, rows)

    print_summary(summary)
    print_candidate_sections(rows, args.max_name_width)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"summary": summary, "modules": rows}, indent=2, sort_keys=True) + "\n"
        )
        print(f"\nwrote_json={args.output_json}")


def inspect_modules(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        direct_params = list(module.parameters(recurse=False))
        direct_param_count = sum(param.numel() for param in direct_params)
        recursive_param_count = sum(param.numel() for param in module.parameters(recurse=True))
        direct_shapes = [
            {
                "name": param_name,
                "shape": list(param.shape),
                "dtype": str(param.dtype).replace("torch.", ""),
                "requires_grad": bool(param.requires_grad),
            }
            for param_name, param in module.named_parameters(recurse=False)
        ]
        rows.append(
            {
                "path": name or "<root>",
                "class": module.__class__.__name__,
                "direct_param_count": int(direct_param_count),
                "recursive_param_count": int(recursive_param_count),
                "direct_parameters": direct_shapes,
                "tags": classify_module(name, module),
            }
        )
    return rows


def classify_module(name: str, module: torch.nn.Module) -> list[str]:
    text = f"{name}.{module.__class__.__name__}".lower()
    tags: list[str] = []
    if any(key in text for key in ("router", "gate", "gating")):
        tags.append("router_or_gate")
    if "expert" in text or "moe" in text:
        tags.append("moe_or_expert")
    if any(key in text for key in ("q_proj", "query", ".q", "wq")):
        tags.append("attention_q")
    if any(key in text for key in ("k_proj", "key", ".k", "wk")):
        tags.append("attention_k")
    if any(key in text for key in ("v_proj", "value", ".v", "wv")):
        tags.append("attention_v")
    if any(key in text for key in ("o_proj", "out_proj", "wo")):
        tags.append("attention_o")
    if any(key in text for key in ("attn", "attention")):
        tags.append("attention")
    if any(key in text for key in ("embed", "wte", "tok_embeddings")):
        tags.append("embedding")
    if any(key in text for key in ("lm_head", "unembed", "output")):
        tags.append("unembedding_or_output")
    if any(key in text for key in ("norm", "ln_")):
        tags.append("normalization")
    return sorted(set(tags))


def build_summary(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    param_count = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    class_counts = Counter(row["class"] for row in rows)
    tag_counts = Counter(tag for row in rows for tag in row["tags"])
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)

    config_fields = {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_experts": getattr(config, "num_experts", None),
        "num_local_experts": getattr(config, "num_local_experts", None),
        "num_experts_per_tok": getattr(config, "num_experts_per_tok", None),
        "moe_layer_interval": getattr(config, "moe_layer_interval", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }
    return {
        "parameter_count": int(param_count),
        "trainable_parameter_count": int(trainable_count),
        "tokenizer_vocab_size": int(vocab_size),
        "module_count": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "config": config_fields,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== MODEL SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


def print_candidate_sections(rows: list[dict[str, Any]], max_name_width: int) -> None:
    sections = (
        ("ROUTER / GATE CANDIDATES", "router_or_gate"),
        ("MOE / EXPERT CANDIDATES", "moe_or_expert"),
        ("ATTENTION CANDIDATES", "attention"),
        ("EMBEDDING CANDIDATES", "embedding"),
        ("UNEMBEDDING / OUTPUT CANDIDATES", "unembedding_or_output"),
    )
    for title, tag in sections:
        candidates = [row for row in rows if tag in row["tags"]]
        print(f"\n=== {title} ({len(candidates)}) ===")
        if not candidates:
            print("<none>")
            continue
        for row in candidates:
            path = row["path"]
            if len(path) > max_name_width:
                path = "..." + path[-(max_name_width - 3) :]
            print(
                f"{path:<{max_name_width}} "
                f"{row['class']:<32} "
                f"direct={row['direct_param_count']:<12} "
                f"recursive={row['recursive_param_count']:<12} "
                f"tags={','.join(row['tags'])}"
            )
            for param in row["direct_parameters"]:
                print(
                    f"  - {param['name']}: shape={param['shape']} "
                    f"dtype={param['dtype']} requires_grad={param['requires_grad']}"
                )


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false.")
    return torch.device(name)


if __name__ == "__main__":
    main()
