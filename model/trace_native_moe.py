from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace hidden states and router choices for a Transformers-native MoE causal LM."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=("causal-lm", "seq2seq-lm"),
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=64)
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

    records: list[dict[str, Any]] = []
    expert_counts: dict[str, int] = {}
    router_shape_records: list[dict[str, Any]] = []

    for example in prompt_set["examples"]:
        prompt_id = require_str(example, "id")
        text = require_str(example, "text")
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model_inputs = dict(encoded)
        if args.model_kind == "seq2seq-lm":
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
        hidden_states = get_hidden_states(outputs)
        router_logits = get_router_logits(outputs)
        if hidden_states is None:
            raise RuntimeError("model returned hidden_states=None.")
        if router_logits is None:
            raise RuntimeError("model returned router_logits=None.")

        input_ids = encoded["input_ids"][0].detach().cpu().tolist()
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        seq_len = len(input_ids)

        for layer_index, hidden in enumerate(hidden_states):
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != seq_len:
                raise ValueError(
                    "hidden state must have shape [1, seq, hidden_dim], "
                    f"got layer={layer_index} shape={list(hidden.shape)}."
                )
            norms = torch.linalg.vector_norm(hidden[0].float(), dim=-1).detach().cpu().tolist()
            for token_index, norm in enumerate(norms):
                records.append(
                    {
                        "kind": "hidden_norm",
                        "prompt_id": prompt_id,
                        "text": text,
                        "token_index": token_index,
                        "token_id": int(input_ids[token_index]),
                        "token_text": token_texts[token_index],
                        "layer_index": layer_index,
                        "hidden_norm": float(norm),
                    }
                )

        normalized_router_logits = normalize_router_logits(router_logits, seq_len=seq_len)
        for router_index, router in enumerate(normalized_router_logits):
            router_shape_records.append(
                {
                    "prompt_id": prompt_id,
                    "router_index": router_index,
                    "shape": list(router.shape),
                }
            )
            probs = torch.softmax(router.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            top_probs, top_experts = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
            for token_index in range(seq_len):
                experts = [int(value) for value in top_experts[token_index].detach().cpu().tolist()]
                probabilities = [float(value) for value in top_probs[token_index].detach().cpu().tolist()]
                for expert in experts:
                    key = f"router_{router_index}.expert_{expert}"
                    expert_counts[key] = expert_counts.get(key, 0) + 1
                records.append(
                    {
                        "kind": "router",
                        "prompt_id": prompt_id,
                        "text": text,
                        "token_index": token_index,
                        "token_id": int(input_ids[token_index]),
                        "token_text": token_texts[token_index],
                        "router_index": router_index,
                        "top_experts": experts,
                        "top_probs": probabilities,
                        "entropy": float(entropy[token_index].detach().cpu().item()),
                    }
                )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "model_dir": str(args.model_dir),
        "prompt_file": str(args.prompts),
        "prompt_count": len(prompt_set["examples"]),
        "record_count": len(records),
        "router_shapes": router_shape_records,
        "expert_counts": dict(sorted(expert_counts.items())),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def normalize_router_logits(router_logits: Any, *, seq_len: int) -> list[torch.Tensor]:
    raw = extract_router_logit_tensors(router_logits, seq_len=seq_len)
    if not raw:
        raise RuntimeError(
            "router output was present, but no encoder token-level router-logit tensors matched "
            f"sequence length {seq_len}."
        )
    normalized: list[torch.Tensor] = []
    for idx, router in enumerate(raw):
        if router.ndim == 3 and router.shape[0] == 1 and router.shape[1] == seq_len:
            normalized.append(router[0].detach())
        elif router.ndim == 2 and router.shape[0] == seq_len:
            normalized.append(router.detach())
        else:
            raise ValueError(
                f"Unsupported router_logits item {idx} shape {list(router.shape)} for seq_len={seq_len}."
            )
    return normalized


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


def get_hidden_states(outputs: Any) -> Any:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states
    encoder_hidden = getattr(outputs, "encoder_hidden_states", None)
    if encoder_hidden is not None:
        return encoder_hidden
    decoder_hidden = getattr(outputs, "decoder_hidden_states", None)
    if decoder_hidden is not None:
        return decoder_hidden
    return None


def get_router_logits(outputs: Any) -> Any:
    router_logits = getattr(outputs, "router_logits", None)
    if router_logits is not None:
        return router_logits
    encoder_router = getattr(outputs, "encoder_router_logits", None)
    if encoder_router is not None:
        return encoder_router
    return getattr(outputs, "decoder_router_logits", None)


def load_prompt_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("prompt file root must be a JSON object.")
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
    if args.output_jsonl.exists():
        raise FileExistsError(f"output-jsonl already exists: {args.output_jsonl}")
    if args.summary_json.exists():
        raise FileExistsError(f"summary-json already exists: {args.summary_json}")
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
