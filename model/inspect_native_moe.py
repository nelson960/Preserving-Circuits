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
        description="Inspect a Transformers-native MoE causal LM module tree."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=("causal-lm", "seq2seq-lm"),
    )
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--probe-text")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-name-width", type=int, default=96)
    args = parser.parse_args()

    _validate_common_args(args.model_dir, args.output_json, args.max_length, args.max_name_width)
    _set_hf_home(args.hf_home)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    device = _device(args.device)
    dtype = _dtype(args.dtype)

    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
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

    rows = inspect_modules(model)
    summary = build_summary(model, tokenizer, config, rows)
    result: dict[str, Any] = {"summary": summary, "modules": rows}

    print_summary(summary)
    print_candidate_sections(rows, args.max_name_width)

    if args.probe_text is not None:
        probe = run_probe(
            model=model,
            tokenizer=tokenizer,
            text=args.probe_text,
            device=device,
            max_length=args.max_length,
            model_kind=args.model_kind,
        )
        result["probe"] = probe
        print("\n=== PROBE ===")
        print(json.dumps(probe, indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote_json={args.output_json}")


def inspect_modules(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        rows.append(
            {
                "path": name or "<root>",
                "class": module.__class__.__name__,
                "direct_param_count": int(sum(p.numel() for p in module.parameters(recurse=False))),
                "recursive_param_count": int(sum(p.numel() for p in module.parameters(recurse=True))),
                "direct_parameters": [
                    {
                        "name": param_name,
                        "shape": list(param.shape),
                        "dtype": str(param.dtype).replace("torch.", ""),
                        "requires_grad": bool(param.requires_grad),
                    }
                    for param_name, param in module.named_parameters(recurse=False)
                ],
                "tags": classify_module(name, module),
            }
        )
    return rows


def classify_module(name: str, module: torch.nn.Module) -> list[str]:
    text = f"{name}.{module.__class__.__name__}".lower()
    tags: list[str] = []
    if any(key in text for key in ("router", "gate", "gating")):
        tags.append("router_or_gate")
    if any(key in text for key in ("moe", "expert", "sparse")):
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
    class_counts = Counter(row["class"] for row in rows)
    tag_counts = Counter(tag for row in rows for tag in row["tags"])
    config_fields = {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "num_local_experts": getattr(config, "num_local_experts", None),
        "num_experts_per_tok": getattr(config, "num_experts_per_tok", None),
        "shared_intermediate_size": getattr(config, "shared_intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }
    return {
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
        "trainable_parameter_count": int(
            sum(param.numel() for param in model.parameters() if param.requires_grad)
        ),
        "tokenizer_vocab_size": int(len(tokenizer)),
        "module_count": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "config": config_fields,
    }


def run_probe(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    device: torch.device,
    max_length: int,
    model_kind: str,
) -> dict[str, Any]:
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

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        encoder_hidden = getattr(outputs, "encoder_hidden_states", None)
        decoder_hidden = getattr(outputs, "decoder_hidden_states", None)
        hidden_states = encoder_hidden if encoder_hidden is not None else decoder_hidden
    router_logits = get_router_logits(outputs)
    if hidden_states is None:
        raise RuntimeError("probe requested hidden_states, but model returned hidden_states=None.")
    if router_logits is None:
        raise RuntimeError("probe requested router_logits, but model returned router_logits=None.")

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    token_texts = tokenizer.convert_ids_to_tokens(input_ids)
    router_logit_tensors = extract_router_logit_tensors(router_logits, seq_len=len(input_ids))
    if not router_logit_tensors:
        raise RuntimeError(
            "probe found router output, but no encoder token-level router-logit tensors matched "
            f"sequence length {len(input_ids)}."
        )
    return {
        "text": text,
        "token_count": len(input_ids),
        "tokens": [{"index": i, "id": int(tok), "text": token_texts[i]} for i, tok in enumerate(input_ids)],
        "hidden_state_count": len(hidden_states),
        "hidden_state_shapes": [list(state.shape) for state in hidden_states],
        "router_logits_count": len(router_logit_tensors),
        "router_logits_shapes": [list(tensor.shape) for tensor in router_logit_tensors],
        "router_raw_structure": summarize_router_structure(router_logits),
    }


def get_router_logits(outputs: Any) -> Any:
    router_logits = getattr(outputs, "router_logits", None)
    if router_logits is not None:
        return router_logits
    encoder_router = getattr(outputs, "encoder_router_logits", None)
    if encoder_router is not None:
        return encoder_router
    return getattr(outputs, "decoder_router_logits", None)


def extract_router_logit_tensors(router_logits: Any, *, seq_len: int) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for tensor in iter_tensors(router_logits):
        if tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] == seq_len:
            tensors.append(tensor.detach())
        elif tensor.ndim == 2 and tensor.shape[0] == seq_len:
            tensors.append(tensor.detach())
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


def summarize_router_structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [summarize_router_structure(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [summarize_router_structure(item) for item in value]}
    return {"type": type(value).__name__}


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== MODEL SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


def print_candidate_sections(rows: list[dict[str, Any]], max_name_width: int) -> None:
    for title, tag in (
        ("ROUTER / GATE CANDIDATES", "router_or_gate"),
        ("MOE / EXPERT CANDIDATES", "moe_or_expert"),
        ("ATTENTION CANDIDATES", "attention"),
        ("EMBEDDING CANDIDATES", "embedding"),
        ("UNEMBEDDING / OUTPUT CANDIDATES", "unembedding_or_output"),
    ):
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
                f"{row['class']:<36} "
                f"direct={row['direct_param_count']:<12} "
                f"recursive={row['recursive_param_count']:<12} "
                f"tags={','.join(row['tags'])}"
            )
            for param in row["direct_parameters"]:
                print(
                    f"  - {param['name']}: shape={param['shape']} "
                    f"dtype={param['dtype']} requires_grad={param['requires_grad']}"
                )


def _validate_common_args(
    model_dir: Path,
    output_json: Path | None,
    max_length: int,
    max_name_width: int,
) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model-dir is not a directory: {model_dir}")
    if output_json is not None and output_json.exists():
        raise FileExistsError(f"output-json already exists: {output_json}")
    if max_length <= 0:
        raise ValueError(f"max-length must be positive, got {max_length}.")
    if max_name_width <= 0:
        raise ValueError(f"max-name-width must be positive, got {max_name_width}.")


def _set_hf_home(hf_home: Path | None) -> None:
    if hf_home is None:
        return
    if hf_home.exists() and not hf_home.is_dir():
        raise NotADirectoryError(f"hf-home is not a directory: {hf_home}")
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home.resolve())


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
