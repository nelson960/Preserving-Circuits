from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture decoder-only LM hidden activations for SAE training or concept probing."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--layer-index", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=("all-tokens", "target-spans"))
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-pt", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    set_hf_home(args.hf_home)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device_from_name(args.device)
    dtype = dtype_from_name(args.dtype)
    prompt_set = load_prompt_set(args.prompts)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    rows: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    for prompt_index, example in enumerate(prompt_set["examples"]):
        prompt_id = require_str(example, "id")
        label = require_str(example, "label")
        text = require_str(example, "text")
        target = require_str(example, "target")
        target_span = find_unique_target_span(text, target)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
        )
        if "offset_mapping" not in encoded:
            raise RuntimeError("tokenizer did not return offset_mapping; a fast tokenizer is required.")
        offsets = encoded.pop("offset_mapping")[0].tolist()
        input_ids = encoded["input_ids"][0].tolist()
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("model returned hidden_states=None.")
        if args.layer_index < 0 or args.layer_index >= len(hidden_states):
            raise IndexError(
                f"layer-index {args.layer_index} out of range for {len(hidden_states)} hidden states."
            )
        hidden = hidden_states[args.layer_index]
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != len(input_ids):
            raise ValueError(
                "hidden state must have shape [1, seq, hidden_dim], "
                f"got {list(hidden.shape)} for seq len {len(input_ids)}."
            )
        hidden_cpu = hidden[0].detach().float().cpu()
        target_token_indices = token_indices_overlapping_span(offsets, target_span)
        if not target_token_indices:
            raise RuntimeError(f"no token overlapped target {target!r} in prompt {prompt_id!r}.")
        if args.mode == "target-spans":
            vector = hidden_cpu[target_token_indices].mean(dim=0)
            vectors.append(vector)
            rows.append(
                {
                    "row_index": len(rows),
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "label": label,
                    "text": text,
                    "target": target,
                    "layer_index": args.layer_index,
                    "mode": args.mode,
                    "target_token_indices": target_token_indices,
                    "target_token_text": "".join(token_texts[index] for index in target_token_indices),
                    "target_span_start": target_span[0],
                    "target_span_end": target_span[1],
                }
            )
        elif args.mode == "all-tokens":
            for token_index, token_id in enumerate(input_ids):
                if int(token_id) in special_ids:
                    continue
                start, end = offsets[token_index]
                if start == end:
                    continue
                vectors.append(hidden_cpu[token_index])
                rows.append(
                    {
                        "row_index": len(rows),
                        "prompt_index": prompt_index,
                        "prompt_id": prompt_id,
                        "label": label,
                        "text": text,
                        "target": target,
                        "token_index": token_index,
                        "token_id": int(token_id),
                        "token_text": token_texts[token_index],
                        "token_span_start": int(start),
                        "token_span_end": int(end),
                        "is_target_token": token_index in target_token_indices,
                        "layer_index": args.layer_index,
                        "mode": args.mode,
                    }
                )
        else:
            raise ValueError(f"unsupported mode: {args.mode}")

    if not vectors:
        raise RuntimeError("captured zero activation vectors.")
    activations = torch.stack(vectors, dim=0)
    if not torch.isfinite(activations).all():
        raise RuntimeError("captured activations contain non-finite values.")
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "activations": activations,
        "rows": rows,
        "metadata": {
            "model_dir": str(args.model_dir),
            "prompt_file": str(args.prompts),
            "prompt_name": prompt_set["name"],
            "layer_index": args.layer_index,
            "mode": args.mode,
            "hidden_dim": int(activations.shape[1]),
            "row_count": int(activations.shape[0]),
            "max_length": args.max_length,
        },
    }
    torch.save(payload, args.output_pt)
    print(
        json.dumps(
            {
                "output_pt": str(args.output_pt),
                "row_count": int(activations.shape[0]),
                "hidden_dim": int(activations.shape[1]),
                "mode": args.mode,
                "layer_index": args.layer_index,
            },
            indent=2,
            sort_keys=True,
        )
    )


def find_unique_target_span(text: str, target: str) -> tuple[int, int]:
    start = text.find(target)
    if start < 0:
        raise ValueError(f"target {target!r} was not found in text {text!r}.")
    second = text.find(target, start + len(target))
    if second >= 0:
        raise ValueError(f"target {target!r} appears multiple times in text {text!r}.")
    return start, start + len(target)


def token_indices_overlapping_span(
    offsets: list[list[int]],
    span: tuple[int, int],
) -> list[int]:
    start, end = span
    indices: list[int] = []
    for index, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ValueError(f"offset entry must contain two ints, got {pair}.")
        token_start, token_end = int(pair[0]), int(pair[1])
        if token_start < end and token_end > start:
            indices.append(index)
    return indices


def load_prompt_set(path: Path) -> dict[str, Any]:
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
        for key in ("id", "label", "text", "target"):
            require_str(example, key)
        prompt_id = str(example["id"])
        if prompt_id in seen_ids:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)
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
        raise FileNotFoundError(f"prompts does not exist: {args.prompts}")
    if args.layer_index < 0:
        raise ValueError(f"layer-index must be non-negative, got {args.layer_index}.")
    if args.max_length <= 0:
        raise ValueError(f"max-length must be positive, got {args.max_length}.")
    if args.output_pt.exists():
        raise FileExistsError(f"output-pt already exists: {args.output_pt}")


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
