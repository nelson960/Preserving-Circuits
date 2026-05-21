from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

from capture_pythia_activations import find_unique_target_span, token_indices_overlapping_span
from discover_sae_concept_features import load_sae, require_input_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Causally test SAE features by ablating or patching feature activations at a decoder-only "
            "LM residual stream site, then measuring next-token log-prob changes."
        )
    )
    parser.add_argument("--sae-pt", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest-json", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--feature-indices", required=True, help="Comma-separated SAE feature indices.")
    parser.add_argument(
        "--include-combined-feature-set",
        action="store_true",
        help="Also test one intervention that edits all requested feature indices together.",
    )
    parser.add_argument("--reference-checkpoint-name", required=True)
    parser.add_argument("--layer-index", required=True, type=int)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    set_hf_home(args.hf_home)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device_from_name(args.device)
    dtype = dtype_from_name(args.dtype)
    feature_indices = parse_feature_indices(args.feature_indices)
    feature_sets = build_feature_sets(feature_indices, include_combined=args.include_combined_feature_set)
    sae_payload = torch.load(args.sae_pt, map_location="cpu", weights_only=False)
    sae = load_sae(sae_payload).to(device)
    sae.eval()
    input_mean = require_input_mean(sae_payload).to(device)
    validate_feature_indices(feature_indices, sae.encoder.out_features)

    manifest = load_manifest(args.checkpoint_manifest_json)
    checkpoints = manifest["captures"]
    checkpoint_by_name = {str(item["name"]): item for item in checkpoints}
    if args.reference_checkpoint_name not in checkpoint_by_name:
        raise ValueError(
            f"reference checkpoint {args.reference_checkpoint_name!r} not found. "
            f"Available: {sorted(checkpoint_by_name)}"
        )
    reference_checkpoint = checkpoint_by_name[args.reference_checkpoint_name]
    reference_model_dir = Path(str(reference_checkpoint["checkpoint_path"]))
    prompts = load_prompt_set(args.prompts)

    tokenizer = AutoTokenizer.from_pretrained(reference_model_dir, local_files_only=True)
    examples = prepare_examples(prompts, tokenizer, args.max_length)
    block_index = block_index_from_layer_index(args.layer_index)

    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_dir,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    reference_model.eval()
    reference_model.to(device)

    reference_cache = build_feature_cache(
        model=reference_model,
        tokenizer=tokenizer,
        examples=examples,
        block_index=block_index,
        sae=sae,
        input_mean=input_mean,
        device=device,
    )

    checkpoint_reports: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_name = str(checkpoint["name"])
        checkpoint_model_dir = Path(str(checkpoint["checkpoint_path"]))
        model = reference_model
        if checkpoint_name != args.reference_checkpoint_name:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_model_dir,
                local_files_only=True,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.eval()
            model.to(device)

        checkpoint_cache = build_feature_cache(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            block_index=block_index,
            sae=sae,
            input_mean=input_mean,
            device=device,
        )

        feature_reports = []
        for feature_set in feature_sets:
            rows = []
            for example in examples:
                normal = score_example(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    device=device,
                    block_index=block_index,
                    intervention=None,
                )
                ablated = score_example(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    device=device,
                    block_index=block_index,
                    intervention=make_ablation_intervention(
                        sae=sae,
                        input_mean=input_mean,
                        feature_indices=feature_set["feature_indices"],
                        target_token_indices=example.target_token_indices,
                    ),
                )
                reference_z = reference_cache[example.prompt_id]
                patched_from_reference = score_example(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    device=device,
                    block_index=block_index,
                    intervention=make_patch_intervention(
                        sae=sae,
                        input_mean=input_mean,
                        feature_indices=feature_set["feature_indices"],
                        target_token_indices=example.target_token_indices,
                        source_z=reference_z,
                    ),
                )
                checkpoint_z = checkpoint_cache[example.prompt_id]
                reference_normal = score_example(
                    model=reference_model,
                    tokenizer=tokenizer,
                    example=example,
                    device=device,
                    block_index=block_index,
                    intervention=None,
                )
                reference_patched_from_checkpoint = score_example(
                    model=reference_model,
                    tokenizer=tokenizer,
                    example=example,
                    device=device,
                    block_index=block_index,
                    intervention=make_patch_intervention(
                        sae=sae,
                        input_mean=input_mean,
                        feature_indices=feature_set["feature_indices"],
                        target_token_indices=example.target_token_indices,
                        source_z=checkpoint_z,
                    ),
                )
                rows.append(
                    {
                        "prompt_id": example.prompt_id,
                        "label": example.label,
                        "target": example.target,
                        "text": example.text,
                        "expected_next": example.expected_next,
                        "expected_token_id": example.expected_token_id,
                        "normal_logprob": normal["expected_logprob"],
                        "ablate_logprob": ablated["expected_logprob"],
                        "patch_from_reference_logprob": patched_from_reference["expected_logprob"],
                        "reference_normal_logprob": reference_normal["expected_logprob"],
                        "reference_patch_from_checkpoint_logprob": reference_patched_from_checkpoint[
                            "expected_logprob"
                        ],
                        "ablate_delta": ablated["expected_logprob"] - normal["expected_logprob"],
                        "patch_from_reference_delta": patched_from_reference["expected_logprob"]
                        - normal["expected_logprob"],
                        "reverse_patch_delta": reference_patched_from_checkpoint["expected_logprob"]
                        - reference_normal["expected_logprob"],
                        "normal_top_token": normal["top_token"],
                        "ablate_top_token": ablated["top_token"],
                        "patch_from_reference_top_token": patched_from_reference["top_token"],
                    }
                )
            feature_reports.append(
                {
                    "feature_index": feature_set["feature_index"],
                    "feature_set_name": feature_set["name"],
                    "feature_indices": feature_set["feature_indices"],
                    "summary_by_label": summarize_rows(rows),
                    "rows": rows,
                }
            )

        checkpoint_reports.append(
            {
                "name": checkpoint_name,
                "step": checkpoint.get("step"),
                "checkpoint_path": str(checkpoint_model_dir),
                "features": feature_reports,
            }
        )
        if model is not reference_model:
            del model

    report = {
        "sae_path": str(args.sae_pt),
        "checkpoint_manifest_json": str(args.checkpoint_manifest_json),
        "prompts": str(args.prompts),
        "reference_checkpoint_name": args.reference_checkpoint_name,
        "layer_index": args.layer_index,
        "block_index": block_index,
        "feature_indices": feature_indices,
        "feature_sets": feature_sets,
        "method": {
            "site": "decoder residual stream after transformer block block_index; same site as hidden_states[layer_index]",
            "ablation": "h <- h - z_j * decoder[:, j] for target-token positions only",
            "patch": "h <- h + (z_source_j - z_target_j) * decoder[:, j] for target-token positions only",
            "metric": "next-token log-probability of each prompt's explicit expected_next token",
            "limits": "This tests causal influence on the selected next-token behavior, not full concept semantics.",
        },
        "checkpoints": checkpoint_reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_json": str(args.output_json), "checkpoint_count": len(checkpoint_reports)}))


class PreparedExample:
    def __init__(
        self,
        *,
        prompt_id: str,
        label: str,
        target: str,
        text: str,
        expected_next: str,
        expected_token_id: int,
        target_token_indices: list[int],
    ) -> None:
        self.prompt_id = prompt_id
        self.label = label
        self.target = target
        self.text = text
        self.expected_next = expected_next
        self.expected_token_id = expected_token_id
        self.target_token_indices = target_token_indices


def build_feature_cache(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[PreparedExample],
    block_index: int,
    sae: torch.nn.Module,
    input_mean: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for example in examples:
        hidden = capture_block_hidden(
            model=model,
            tokenizer=tokenizer,
            text=example.text,
            block_index=block_index,
            device=device,
        )
        target_hidden = hidden[0, example.target_token_indices, :]
        cache[example.prompt_id] = encode_sae(sae, input_mean, target_hidden).detach()
    return cache


def capture_block_hidden(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    block_index: int,
    device: torch.device,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = require_layer_output_tensor(output)
        captured.append(hidden.detach())
        return output

    handle = model.gpt_neox.layers[block_index].register_forward_hook(hook)
    try:
        encoded = tokenizer(text, return_tensors="pt", padding=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected exactly one captured hidden tensor, got {len(captured)}.")
    return captured[0]


def score_example(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    example: PreparedExample,
    device: torch.device,
    block_index: int,
    intervention: Callable[[Any], Any] | None,
) -> dict[str, Any]:
    handle = None
    if intervention is not None:
        handle = model.gpt_neox.layers[block_index].register_forward_hook(intervention)
    try:
        encoded = tokenizer(example.text, return_tensors="pt", padding=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, use_cache=False, return_dict=True)
    finally:
        if handle is not None:
            handle.remove()
    logits = outputs.logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    expected_logprob = float(log_probs[example.expected_token_id].detach().cpu().item())
    top_id = int(torch.argmax(logits).detach().cpu().item())
    return {
        "expected_logprob": expected_logprob,
        "top_token_id": top_id,
        "top_token": tokenizer.convert_ids_to_tokens([top_id])[0],
    }


def make_ablation_intervention(
    *,
    sae: torch.nn.Module,
    input_mean: torch.Tensor,
    feature_indices: list[int],
    target_token_indices: list[int],
) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], Any]:
    def intervention(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = require_layer_output_tensor(output)
        edited = hidden.clone()
        target_hidden = edited[:, target_token_indices, :]
        z = encode_sae(sae, input_mean, target_hidden[0])
        feature_tensor = torch.tensor(feature_indices, device=hidden.device, dtype=torch.long)
        decoder_cols = sae.decoder.weight[:, feature_tensor].to(device=hidden.device, dtype=torch.float32)
        delta = -z[:, feature_tensor] @ decoder_cols.T
        edited[:, target_token_indices, :] = target_hidden + delta.unsqueeze(0).to(dtype=hidden.dtype)
        return replace_layer_output_tensor(output, edited)

    return intervention


def make_patch_intervention(
    *,
    sae: torch.nn.Module,
    input_mean: torch.Tensor,
    feature_indices: list[int],
    target_token_indices: list[int],
    source_z: torch.Tensor,
) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], Any]:
    def intervention(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = require_layer_output_tensor(output)
        edited = hidden.clone()
        target_hidden = edited[:, target_token_indices, :]
        target_z = encode_sae(sae, input_mean, target_hidden[0])
        if source_z.shape != target_z.shape:
            raise ValueError(f"source_z shape {tuple(source_z.shape)} != target_z shape {tuple(target_z.shape)}.")
        source = source_z.to(device=hidden.device, dtype=torch.float32)
        feature_tensor = torch.tensor(feature_indices, device=hidden.device, dtype=torch.long)
        decoder_cols = sae.decoder.weight[:, feature_tensor].to(device=hidden.device, dtype=torch.float32)
        delta = (source[:, feature_tensor] - target_z[:, feature_tensor]) @ decoder_cols.T
        edited[:, target_token_indices, :] = target_hidden + delta.unsqueeze(0).to(dtype=hidden.dtype)
        return replace_layer_output_tensor(output, edited)

    return intervention


def encode_sae(sae: torch.nn.Module, input_mean: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 2:
        raise ValueError(f"hidden must have shape [tokens, hidden_dim], got {tuple(hidden.shape)}.")
    centered = hidden.float() - input_mean.to(device=hidden.device, dtype=torch.float32)
    with torch.no_grad():
        return torch.relu(sae.encoder(centered))


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


def replace_layer_output_tensor(output: Any, edited_hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        if len(output) == 0:
            raise ValueError("layer output tuple is empty.")
        return (edited_hidden,) + output[1:]
    return edited_hidden


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    labels = sorted({str(row["label"]) for row in rows})
    summary: dict[str, dict[str, float | int]] = {}
    for label in labels:
        group = [row for row in rows if row["label"] == label]
        summary[label] = {
            "count": len(group),
            "normal_logprob_mean": mean(row["normal_logprob"] for row in group),
            "ablate_delta_mean": mean(row["ablate_delta"] for row in group),
            "patch_from_reference_delta_mean": mean(row["patch_from_reference_delta"] for row in group),
            "reverse_patch_delta_mean": mean(row["reverse_patch_delta"] for row in group),
        }
    summary["all"] = {
        "count": len(rows),
        "normal_logprob_mean": mean(row["normal_logprob"] for row in rows),
        "ablate_delta_mean": mean(row["ablate_delta"] for row in rows),
        "patch_from_reference_delta_mean": mean(row["patch_from_reference_delta"] for row in rows),
        "reverse_patch_delta_mean": mean(row["reverse_patch_delta"] for row in rows),
    }
    return summary


def mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("cannot compute mean of empty sequence.")
    return sum(items) / len(items)


def prepare_examples(prompt_set: dict[str, Any], tokenizer: Any, max_length: int) -> list[PreparedExample]:
    examples: list[PreparedExample] = []
    for raw in prompt_set["examples"]:
        prompt_id = require_str(raw, "id")
        label = require_str(raw, "label")
        target = require_str(raw, "target")
        text = require_str(raw, "text")
        expected_next = require_str(raw, "expected_next")
        expected_ids = tokenizer.encode(expected_next, add_special_tokens=False)
        if len(expected_ids) != 1:
            raise ValueError(
                f"expected_next for prompt {prompt_id!r} must tokenize to one token, "
                f"got ids={expected_ids}."
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
            PreparedExample(
                prompt_id=prompt_id,
                label=label,
                target=target,
                text=text,
                expected_next=expected_next,
                expected_token_id=int(expected_ids[0]),
                target_token_indices=target_token_indices,
            )
        )
    return examples


def load_prompt_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("prompt file root must be an object.")
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
        for key in ("id", "label", "target", "text", "expected_next"):
            require_str(example, key)
        prompt_id = str(example["id"])
        if prompt_id in seen_ids:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)
    return data


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("manifest root must be an object.")
    captures = data.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("manifest must contain a non-empty captures list.")
    for idx, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise TypeError(f"captures[{idx}] must be an object.")
        for key in ("name", "checkpoint_path"):
            require_str(capture, key)
        checkpoint_path = Path(str(capture["checkpoint_path"]))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")
    return data


def require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} must be a non-empty string.")
    return value


def parse_feature_indices(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("feature-indices must contain at least one integer.")
    indices: list[int] = []
    for item in items:
        try:
            index = int(item)
        except ValueError as error:
            raise ValueError(f"feature index must be an int, got {item!r}.") from error
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(f"feature-indices contains duplicates: {indices}")
    return indices


def validate_feature_indices(feature_indices: list[int], feature_dim: int) -> None:
    for feature_index in feature_indices:
        if feature_index < 0 or feature_index >= feature_dim:
            raise IndexError(
                f"feature index {feature_index} out of range for SAE feature_dim={feature_dim}."
            )


def build_feature_sets(feature_indices: list[int], *, include_combined: bool) -> list[dict[str, Any]]:
    feature_sets: list[dict[str, Any]] = []
    for feature_index in feature_indices:
        feature_sets.append(
            {
                "name": f"feature_{feature_index}",
                "feature_index": feature_index,
                "feature_indices": [feature_index],
            }
        )
    if include_combined:
        feature_sets.append(
            {
                "name": "combined_requested_features",
                "feature_index": None,
                "feature_indices": feature_indices,
            }
        )
    return feature_sets


def block_index_from_layer_index(layer_index: int) -> int:
    if layer_index <= 0:
        raise ValueError("causal interventions at embedding hidden_states[0] are not supported.")
    return layer_index - 1


def validate_args(args: argparse.Namespace) -> None:
    if not args.sae_pt.exists():
        raise FileNotFoundError(f"sae-pt does not exist: {args.sae_pt}")
    if not args.checkpoint_manifest_json.exists():
        raise FileNotFoundError(f"checkpoint-manifest-json does not exist: {args.checkpoint_manifest_json}")
    if not args.prompts.exists():
        raise FileNotFoundError(f"prompts does not exist: {args.prompts}")
    if args.layer_index < 0:
        raise ValueError(f"layer-index must be non-negative, got {args.layer_index}.")
    if args.max_length <= 0:
        raise ValueError(f"max-length must be positive, got {args.max_length}.")
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
