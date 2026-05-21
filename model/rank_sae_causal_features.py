from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from capture_pythia_activations import find_unique_target_span, token_indices_overlapping_span
from discover_sae_concept_features import load_sae, require_input_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank SAE features by first-order causal attribution for explicit next-token behavior. "
            "For feature j, ablation delta is approximated as -z_j * <grad_logp, decoder_j>."
        )
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--sae-pt", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--negative-labels", required=True, help="Comma-separated labels.")
    parser.add_argument("--layer-index", required=True, type=int)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    set_hf_home(args.hf_home)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device_from_name(args.device)
    dtype = dtype_from_name(args.dtype)
    negative_labels = parse_csv(args.negative_labels)
    if args.concept in negative_labels:
        raise ValueError("concept must not be included in negative-labels.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    sae_payload = torch.load(args.sae_pt, map_location="cpu", weights_only=False)
    sae = load_sae(sae_payload).to(device)
    sae.eval()
    input_mean = require_input_mean(sae_payload).to(device)
    decoder = sae.decoder.weight.detach().to(device=device, dtype=torch.float32)
    prompt_set = load_prompt_set(args.prompts)
    examples = prepare_examples(prompt_set, tokenizer, args.max_length)
    block_index = block_index_from_layer_index(args.layer_index)

    feature_dim = int(sae.encoder.out_features)
    rows = []
    support_rows = []
    ablation_delta_rows = []
    activation_rows = []
    for example in examples:
        captured: list[torch.Tensor] = []

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = require_layer_output_tensor(output)
            hidden.retain_grad()
            captured.append(hidden)
            return output

        handle = model.gpt_neox.layers[block_index].register_forward_hook(hook)
        try:
            encoded = tokenizer(
                example["text"],
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
                padding=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model.zero_grad(set_to_none=True)
            outputs = model(**encoded, use_cache=False, return_dict=True)
            logits = outputs.logits[0, -1].float()
            logprob = torch.log_softmax(logits, dim=-1)[example["expected_token_id"]]
            logprob.backward()
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one captured hidden tensor, got {len(captured)}.")
        hidden = captured[0]
        grad = hidden.grad
        if grad is None:
            raise RuntimeError("captured hidden gradient is None.")
        target_hidden = hidden[0, example["target_token_indices"], :].detach()
        target_grad = grad[0, example["target_token_indices"], :].detach().float()
        with torch.no_grad():
            z = torch.relu(sae.encoder(target_hidden.float() - input_mean))
            grad_dot_decoder = target_grad @ decoder
            support = z * grad_dot_decoder
            ablation_delta = -support
            support_mean = support.mean(dim=0).detach().cpu()
            ablation_delta_mean = ablation_delta.mean(dim=0).detach().cpu()
            activation_mean = z.mean(dim=0).detach().cpu()
        support_rows.append(support_mean)
        ablation_delta_rows.append(ablation_delta_mean)
        activation_rows.append(activation_mean)
        rows.append(
            {
                "prompt_id": example["prompt_id"],
                "label": example["label"],
                "target": example["target"],
                "text": example["text"],
                "expected_next": example["expected_next"],
                "expected_logprob": float(logprob.detach().cpu().item()),
            }
        )

    support_matrix = torch.stack(support_rows)
    ablation_delta_matrix = torch.stack(ablation_delta_rows)
    activation_matrix = torch.stack(activation_rows)
    labels = [row["label"] for row in rows]
    concept_indices = [idx for idx, label in enumerate(labels) if label == args.concept]
    negative_indices = [idx for idx, label in enumerate(labels) if label in negative_labels]
    if not concept_indices:
        raise RuntimeError(f"no rows found for concept {args.concept!r}.")
    if not negative_indices:
        raise RuntimeError(f"no rows found for negative labels {negative_labels}.")

    concept_support = support_matrix[concept_indices].mean(dim=0)
    negative_support = support_matrix[negative_indices].mean(dim=0)
    concept_ablation_delta = ablation_delta_matrix[concept_indices].mean(dim=0)
    negative_ablation_delta = ablation_delta_matrix[negative_indices].mean(dim=0)
    concept_activation = activation_matrix[concept_indices].mean(dim=0)
    negative_activation = activation_matrix[negative_indices].mean(dim=0)

    feature_reports = []
    for feature_index in range(feature_dim):
        feature_reports.append(
            {
                "feature_index": feature_index,
                "concept_support": float(concept_support[feature_index].item()),
                "negative_support": float(negative_support[feature_index].item()),
                "support_selectivity": float(
                    (concept_support[feature_index] - negative_support[feature_index]).item()
                ),
                "concept_predicted_ablation_delta": float(concept_ablation_delta[feature_index].item()),
                "negative_predicted_ablation_delta": float(negative_ablation_delta[feature_index].item()),
                "concept_activation": float(concept_activation[feature_index].item()),
                "negative_activation": float(negative_activation[feature_index].item()),
            }
        )
    ranked_support = sorted(
        feature_reports,
        key=lambda item: (
            float(item["concept_support"]),
            float(item["support_selectivity"]),
            float(item["concept_activation"]),
        ),
        reverse=True,
    )
    ranked_damage = sorted(
        feature_reports,
        key=lambda item: (
            -float(item["concept_predicted_ablation_delta"]),
            float(item["support_selectivity"]),
            float(item["concept_activation"]),
        ),
        reverse=True,
    )
    report = {
        "model_dir": str(args.model_dir),
        "sae_path": str(args.sae_pt),
        "prompts": str(args.prompts),
        "concept": args.concept,
        "negative_labels": negative_labels,
        "layer_index": args.layer_index,
        "block_index": block_index,
        "top_k": args.top_k,
        "rows": rows,
        "top_support_features": ranked_support[: args.top_k],
        "top_predicted_damage_features": ranked_damage[: args.top_k],
        "method": {
            "support": "z_j * <grad expected_next logprob wrt residual, decoder_j>",
            "predicted_ablation_delta": "-support; negative means ablation should lower expected_next logprob",
            "note": "This is a first-order causal attribution used to choose features for direct ablation.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "top_support_feature": ranked_support[0],
                "top_predicted_damage_feature": ranked_damage[0],
            },
            indent=2,
            sort_keys=True,
        )
    )


def prepare_examples(prompt_set: dict[str, Any], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    examples = []
    for raw in prompt_set["examples"]:
        prompt_id = require_str(raw, "id")
        label = require_str(raw, "label")
        target = require_str(raw, "target")
        text = require_str(raw, "text")
        expected_next = require_str(raw, "expected_next")
        expected_ids = tokenizer.encode(expected_next, add_special_tokens=False)
        if len(expected_ids) != 1:
            raise ValueError(
                f"expected_next for prompt {prompt_id!r} must tokenize to one token, got ids={expected_ids}."
            )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"][0].tolist()
        target_span = find_unique_target_span(text, target)
        target_token_indices = token_indices_overlapping_span(offsets, target_span)
        if not target_token_indices:
            raise RuntimeError(f"no token overlapped target {target!r} in prompt {prompt_id!r}.")
        examples.append(
            {
                "prompt_id": prompt_id,
                "label": label,
                "target": target,
                "text": text,
                "expected_next": expected_next,
                "expected_token_id": int(expected_ids[0]),
                "target_token_indices": target_token_indices,
            }
        )
    return examples


def require_layer_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        if len(output) == 0:
            raise ValueError("layer output tuple is empty.")
        hidden = output[0]
    else:
        hidden = output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"layer hidden output must be a tensor, got {type(hidden).__name__}.")
    if hidden.ndim != 3:
        raise ValueError(f"layer hidden output must be rank-3, got shape {tuple(hidden.shape)}.")
    return hidden


def load_prompt_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("prompt file root must be an object.")
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("prompt file must contain a non-empty examples list.")
    seen_ids: set[str] = set()
    for idx, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"examples[{idx}] must be an object.")
        for key in ("id", "label", "target", "text", "expected_next"):
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


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("CSV argument must contain at least one label.")
    if len(set(items)) != len(items):
        raise ValueError(f"CSV argument contains duplicate labels: {items}")
    return items


def block_index_from_layer_index(layer_index: int) -> int:
    if layer_index <= 0:
        raise ValueError("causal interventions at embedding hidden_states[0] are not supported.")
    return layer_index - 1


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {args.model_dir}")
    if not args.sae_pt.exists():
        raise FileNotFoundError(f"sae-pt does not exist: {args.sae_pt}")
    if not args.prompts.exists():
        raise FileNotFoundError(f"prompts does not exist: {args.prompts}")
    if args.layer_index < 0:
        raise ValueError(f"layer-index must be non-negative, got {args.layer_index}.")
    if args.max_length <= 0:
        raise ValueError(f"max-length must be positive, got {args.max_length}.")
    if args.top_k <= 0:
        raise ValueError(f"top-k must be positive, got {args.top_k}.")
    if args.output_json.exists():
        raise FileExistsError(f"output-json already exists: {args.output_json}")


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
