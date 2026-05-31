#!/usr/bin/env python3
"""Real-book GFO benchmark with a dynamic living concept map.

This extends the activation-anchor NLP benchmark from a static anchor bank to a
stateful map:

    pending trace dynamics -> write gate -> create / reinforce / replacement-fuse
    -> lineage-aware drift metrics -> optional background repair

The write kernel is intentionally still the soft activation-anchor penalty from
the first real-book GFO benchmark. The living map now uses trace strength for
its commit gate: observations route into pending traces, repeated routed
activation strengthens the trace, unused traces decay, and only strong traces
commit into protected semantic circuits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm.auto import tqdm

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

from models import DecoderTransformer  # noqa: E402
from real_book_common import (  # noqa: E402
    format_qa_prompt,
    masked_cross_entropy,
    make_qa_supervision,
    require_token_id,
    resolve_device,
)
from gfo_real_book_activation_cl import (  # noqa: E402
    AnchorSpec,
    configure_trainable_parameters,
    encode_lm_tensors,
    evaluate_chunk_prompts,
    evaluate_prompt_group,
    instantiate_model,
    iter_batches,
    load_chunks,
    load_fact_probes,
    load_prompt_groups,
    prompt_list,
    qa_supervision_for_chunk,
    select_anchor_batch,
    set_seed,
)


METHODS = ("adamw", "gfo_living", "replay_living")
EPS = 1e-8


def require_finite_tensor(name: str, tensor: torch.Tensor, context: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite tensor detected for {name} ({context}).")


def require_finite_float(name: str, value: float, context: str) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite value detected for {name}={value!r} ({context}).")


def require_gradients_finite(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    context: str,
    component_values: dict[str, float],
) -> None:
    for index, (name, parameter) in enumerate(named_parameters):
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            grad = parameter.grad.detach()
            nan_count = int(torch.isnan(grad).sum().detach().cpu())
            posinf_count = int(torch.isposinf(grad).sum().detach().cpu())
            neginf_count = int(torch.isneginf(grad).sum().detach().cpu())
            finite = grad[torch.isfinite(grad)]
            finite_abs_max = float(finite.abs().max().detach().cpu()) if finite.numel() else float("nan")
            raise FloatingPointError(
                "Non-finite gradient detected "
                f"for trainable parameter index {index} name={name!r} shape={tuple(parameter.shape)} "
                f"nan_count={nan_count} posinf_count={posinf_count} neginf_count={neginf_count} "
                f"finite_abs_max={finite_abs_max:.6g} ({context}). "
                f"loss_components: {format_component_values(component_values)}"
            )


def format_component_values(component_values: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.6g}" for key, value in sorted(component_values.items()))


def clip_grad_norm_strict(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    max_norm: float,
    context: str,
    component_values: dict[str, float],
) -> float:
    if max_norm <= 0.0:
        raise ValueError("clip_grad_norm_strict requires max_norm > 0.")

    total_sq = 0.0
    max_abs = 0.0
    max_abs_name = ""
    grad_count = 0
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        grad_count += 1
        grad_cpu = parameter.grad.detach().cpu().to(torch.float64)
        if not torch.isfinite(grad_cpu).all():
            raise FloatingPointError(
                f"Non-finite gradient reached strict clipping for parameter {name!r} ({context}). "
                f"loss_components: {format_component_values(component_values)}"
            )
        param_sq = float((grad_cpu * grad_cpu).sum().item())
        if not math.isfinite(param_sq):
            raise FloatingPointError(
                f"Non-finite gradient squared norm for parameter {name!r} ({context}). "
                f"loss_components: {format_component_values(component_values)}"
            )
        total_sq += param_sq
        local_max = float(grad_cpu.abs().max().item()) if grad_cpu.numel() else 0.0
        if local_max > max_abs:
            max_abs = local_max
            max_abs_name = name

    if grad_count == 0:
        raise RuntimeError(f"No gradients were present before clipping ({context}).")
    if not math.isfinite(total_sq):
        raise FloatingPointError(
            f"Non-finite total gradient squared norm ({context}); "
            f"max_abs_grad={max_abs:.6g} max_abs_parameter={max_abs_name!r}. "
            f"loss_components: {format_component_values(component_values)}"
        )

    total_norm = math.sqrt(total_sq)
    if not math.isfinite(total_norm):
        raise FloatingPointError(
            f"Non-finite total gradient norm ({context}); "
            f"max_abs_grad={max_abs:.6g} max_abs_parameter={max_abs_name!r}. "
            f"loss_components: {format_component_values(component_values)}"
        )

    scale = max_norm / (total_norm + EPS)
    if scale < 1.0:
        for _, parameter in named_parameters:
            if parameter.grad is not None:
                parameter.grad.detach().mul_(scale)
    return total_norm


def select_supervision_rows(
    supervision: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    global_step: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, targets, mask = supervision
    if batch_size < 0:
        raise ValueError("Supervision batch size must be non-negative.")
    if batch_size == 0 or batch_size >= len(inputs):
        return inputs, targets, mask
    if len(inputs) <= 0:
        raise ValueError("Cannot batch an empty supervision tensor.")
    start = (global_step * batch_size) % len(inputs)
    indices = (torch.arange(batch_size, device=inputs.device) + start) % len(inputs)
    return inputs.index_select(0, indices), targets.index_select(0, indices), mask.index_select(0, indices)


def require_model_parameters_finite(model: DecoderTransformer, context: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite model parameter detected for {name!r} ({context}).")


def parameter_count_stats(model: DecoderTransformer) -> dict[str, float]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    if total <= 0:
        raise RuntimeError("Model has zero parameters.")
    return {
        "total_parameter_count": float(total),
        "trainable_parameter_count": float(trainable),
        "frozen_parameter_count": float(total - trainable),
        "trainable_parameter_fraction": float(trainable / total),
    }


def snapshot_trainable_parameters(model: DecoderTransformer) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        require_finite_tensor("snapshot_parameter", parameter, f"parameter={name}")
        snapshot[name] = parameter.detach().cpu().clone()
    if not snapshot:
        raise RuntimeError("No trainable parameters are available to snapshot.")
    return snapshot


def trainable_parameter_delta_stats(
    model: DecoderTransformer,
    snapshot: dict[str, torch.Tensor],
    *,
    metric_prefix: str,
) -> dict[str, float]:
    current_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    snapshot_names = set(snapshot)
    if current_names != snapshot_names:
        missing = sorted(snapshot_names - current_names)
        extra = sorted(current_names - snapshot_names)
        raise RuntimeError(
            "Trainable parameter set changed during the run: "
            f"missing_current={missing}, missing_snapshot={extra}"
        )

    delta_sq = 0.0
    reference_sq = 0.0
    max_abs = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current = parameter.detach().cpu()
        require_finite_tensor("current_parameter", current, f"parameter={name}")
        reference = snapshot[name]
        delta = current - reference
        delta_sq += float((delta.float() ** 2).sum().item())
        reference_sq += float((reference.float() ** 2).sum().item())
        if delta.numel() > 0:
            max_abs = max(max_abs, float(delta.float().abs().max().item()))

    delta_norm = math.sqrt(delta_sq)
    reference_norm = math.sqrt(reference_sq)
    relative = delta_norm / (reference_norm + EPS)
    return {
        f"{metric_prefix}_weight_delta_norm": float(delta_norm),
        f"{metric_prefix}_weight_delta_relative": float(relative),
        f"{metric_prefix}_weight_delta_max_abs": float(max_abs),
    }


@dataclass
class LayerActivationAnchor:
    anchor_id: str
    chunk_id: str
    source_type: str
    text: str
    inputs: torch.Tensor
    targets: torch.Tensor
    hidden_target: torch.Tensor
    mask: torch.Tensor | None
    importance: float
    layer_id: str
    native_slot_targets: dict[str, int] = field(default_factory=dict)


@dataclass
class SemanticMarginAnchor:
    anchor_id: str
    source_anchor_id: str
    chunk_id: str
    source_type: str
    question: str
    answer: str
    prompt_inputs: torch.Tensor
    correct_answer_ids: torch.Tensor
    negative_answer_ids: torch.Tensor
    answer_inputs: torch.Tensor
    answer_targets: torch.Tensor
    answer_mask: torch.Tensor
    importance: float
    cluster_source: str


@dataclass
class PendingConcept:
    pending_id: str
    source_type: str
    layer_id: str
    centroid: torch.Tensor
    count: int
    variance: float
    strength: float
    radius_sq: float
    routed_energy: float
    last_seen_step: int
    latest_anchor_id: str
    latest_pressure: float
    latest_evidence: dict[str, float] = field(default_factory=dict)
    anchors: list[LayerActivationAnchor] = field(default_factory=list)
    semantic_anchors: list[SemanticMarginAnchor] = field(default_factory=list)
    native_slot_targets: dict[str, int] = field(default_factory=dict)


@dataclass
class LivingConcept:
    concept_id: str
    lineage_id: str
    anchor: LayerActivationAnchor
    layer_id: str
    status: str
    created_step: int
    last_seen_step: int
    count: int
    importance: float
    tolerance: float
    stability: float
    pressure: float
    transformed_from: str | None = None
    transform_count: int = 0
    evidence: dict[str, float] = field(default_factory=dict)
    anchors: list[LayerActivationAnchor] = field(default_factory=list)
    semantic_anchors: list[SemanticMarginAnchor] = field(default_factory=list)
    native_slot_targets: dict[str, int] = field(default_factory=dict)


def concept_anchor_list(concept: LivingConcept) -> list[LayerActivationAnchor]:
    if concept.anchors:
        return concept.anchors
    return [concept.anchor]


def concept_semantic_anchor_list(concept: LivingConcept) -> list[SemanticMarginAnchor]:
    return concept.semantic_anchors


def concept_centroid(concept: LivingConcept, source_type: str | None = None) -> torch.Tensor | None:
    anchors = concept_anchor_list(concept)
    if source_type is not None:
        anchors = [anchor for anchor in anchors if anchor.source_type == source_type]
    if not anchors:
        return None
    vectors = [pooled_anchor_vector(anchor) for anchor in anchors]
    return torch.stack(vectors).mean(dim=0)


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Available methods: {METHODS}.")
    return methods


def parse_layer_ids(value: str) -> list[str]:
    layers = [item.strip() for item in value.split(",") if item.strip()]
    if not layers:
        raise ValueError("Layer list must contain at least one layer id.")
    if len(set(layers)) != len(layers):
        raise ValueError(f"Layer list contains duplicates: {layers}")
    return layers


def encode_answer_ids(tokenizer: Tokenizer, answer: str) -> list[int]:
    answer_ids = tokenizer.encode(answer.strip()).ids
    if not answer_ids:
        raise ValueError(f"Answer encoded to zero tokens: {answer!r}")
    return answer_ids


def collect_answer_vocabulary(
    chunks: Sequence[dict[str, object]],
    heldout_prompt_groups: dict[str, list[dict[str, str]]] | None,
    semantic_cluster_prompt_groups: dict[str, list[dict[str, str]]] | None,
) -> list[str]:
    answers: list[str] = []
    for chunk in chunks:
        for key in ("local_prompts", "retention_prompts", "composition_prompts"):
            for prompt in chunk[key]:  # type: ignore[index]
                if not isinstance(prompt, dict) or "answer" not in prompt:
                    raise ValueError(f"Invalid prompt in chunk {chunk.get('chunk_id', '<unknown>')!r} field {key!r}.")
                answers.append(str(prompt["answer"]).strip())
    if heldout_prompt_groups is not None:
        for group_prompts in heldout_prompt_groups.values():
            for prompt in group_prompts:
                answers.append(prompt["answer"].strip())
    if semantic_cluster_prompt_groups is not None:
        for group_prompts in semantic_cluster_prompt_groups.values():
            for prompt in group_prompts:
                answers.append(prompt["answer"].strip())
    unique = sorted({answer for answer in answers if answer})
    if len(unique) < 2:
        raise ValueError("Semantic margin anchors require at least two distinct known answers.")
    return unique


def answer_key(answer: str) -> str:
    return " ".join(answer.strip().lower().split())


def prompts_by_answer(prompt_groups: dict[str, list[dict[str, str]]] | None) -> dict[str, list[dict[str, str]]]:
    if prompt_groups is None:
        return {}
    grouped: dict[str, list[dict[str, str]]] = {}
    for group_name, prompts in prompt_groups.items():
        if not prompts:
            continue
        for index, prompt in enumerate(prompts):
            if "question" not in prompt or "answer" not in prompt:
                raise ValueError(f"Prompt group {group_name!r} item {index} must contain question and answer.")
            key = answer_key(prompt["answer"])
            grouped.setdefault(key, []).append({"question": prompt["question"], "answer": prompt["answer"]})
    return grouped


def prompts_by_answer_for_group(
    prompt_groups: dict[str, list[dict[str, str]]] | None,
    group_name: str,
) -> dict[str, list[dict[str, str]]]:
    if prompt_groups is None or group_name not in prompt_groups:
        return {}
    return prompts_by_answer({group_name: prompt_groups[group_name]})


def prompts_by_answer_by_source(
    prompt_groups: dict[str, list[dict[str, str]]] | None,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    return {
        "qa": prompts_by_answer_for_group(prompt_groups, "retention_clusters"),
        "composition_qa": prompts_by_answer_for_group(prompt_groups, "composition_clusters"),
    }


def cluster_prompts_for_source_answer(
    semantic_cluster_by_source: dict[str, dict[str, list[dict[str, str]]]],
    source_type: str,
    answer: str,
    max_prompts: int,
) -> list[dict[str, str]]:
    if source_type not in semantic_cluster_by_source:
        raise ValueError(f"Unknown semantic cluster source type: {source_type!r}.")
    prompts = semantic_cluster_by_source[source_type].get(answer_key(answer), [])
    if max_prompts < 0:
        raise ValueError("--semantic-cluster-max-prompts must be non-negative.")
    if max_prompts == 0:
        return list(prompts)
    return list(prompts[:max_prompts])


def living_anchor_specs_for_chunk(
    chunk: dict[str, object],
    fact_probes: dict[str, list[str]],
    *,
    include_local_prompts: bool,
    include_composition_prompts: bool,
    max_fact_probes: int,
) -> list[AnchorSpec]:
    chunk_id = str(chunk["chunk_id"])
    specs: list[AnchorSpec] = []
    if include_local_prompts:
        for prompt in prompt_list(chunk, "local_prompts"):
            text = f"{format_qa_prompt(prompt['question'])}{prompt['answer']}"
            specs.append(AnchorSpec("qa", text, prompt))
    if include_composition_prompts:
        for prompt in prompt_list(chunk, "composition_prompts"):
            text = f"{format_qa_prompt(prompt['question'])}{prompt['answer']}"
            specs.append(AnchorSpec("composition_qa", text, prompt))
    if max_fact_probes < 0:
        raise ValueError("--max-fact-probes-per-chunk must be non-negative.")
    for probe in fact_probes.get(chunk_id, [])[:max_fact_probes]:
        specs.append(AnchorSpec("fact_probe", probe))
    return specs


def negative_answer_ids_for(
    tokenizer: Tokenizer,
    answer_vocab: Sequence[str],
    correct_answer: str,
    max_negatives: int,
) -> torch.Tensor:
    if max_negatives <= 0:
        raise ValueError("--semantic-margin-negatives must be positive.")
    correct_ids = encode_answer_ids(tokenizer, correct_answer)
    correct_first_id = correct_ids[0]
    candidates = [answer for answer in answer_vocab if answer.strip().lower() != correct_answer.strip().lower()]
    if not candidates:
        raise ValueError(f"No negative answers available for correct answer {correct_answer!r}.")
    first_tokens: list[int] = []
    seen: set[int] = set()
    for answer in candidates:
        ids = encode_answer_ids(tokenizer, answer)
        first_id = ids[0]
        if first_id == correct_first_id:
            continue
        if first_id in seen:
            continue
        first_tokens.append(first_id)
        seen.add(first_id)
        if len(first_tokens) >= max_negatives:
            break
    if not first_tokens:
        raise ValueError(
            f"No distinct negative first-token ids available for correct answer {correct_answer!r}."
        )
    return torch.tensor(first_tokens, dtype=torch.long)


def valid_layer_ids(n_layers: int) -> set[str]:
    if n_layers <= 0:
        raise ValueError("--n-layers must be positive.")
    return {"embed", "final", *{f"block_{index}" for index in range(n_layers)}}


def validate_layer_ids(layer_ids: Sequence[str], n_layers: int, option_name: str) -> None:
    valid = valid_layer_ids(n_layers)
    unknown = sorted(set(layer_ids) - valid)
    if unknown:
        raise ValueError(f"{option_name} contains invalid layer id(s): {unknown}. Valid layers: {sorted(valid)}")


def layer_hidden(model: DecoderTransformer, tokens: torch.Tensor, layer_id: str) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be shape [batch, seq], got {tokens.shape}")
    batch, seq_len = tokens.shape
    if seq_len > model.max_seq_len:
        raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {model.max_seq_len}")

    positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)
    hidden = model.token_embedding(tokens) + model.position_embedding(positions)
    if layer_id == "embed":
        return hidden

    if layer_id.startswith("block_"):
        raw_index = layer_id.removeprefix("block_")
        if not raw_index.isdigit():
            raise ValueError(f"Invalid block layer id: {layer_id!r}")
        target_index = int(raw_index)
        if target_index < 0 or target_index >= len(model.blocks):
            raise ValueError(f"Block layer id {layer_id!r} is outside model.blocks length {len(model.blocks)}.")
        for block_index, block in enumerate(model.blocks):
            hidden = block(hidden)
            if block_index == target_index:
                return hidden
        raise RuntimeError(f"Failed to return hidden state for layer {layer_id!r}.")

    if layer_id == "final":
        for block in model.blocks:
            hidden = block(hidden)
        return model.ln_f(hidden)

    raise ValueError(f"Unknown layer id: {layer_id!r}")


def native_trace_adapters(model: DecoderTransformer) -> list[tuple[str, object]]:
    adapters: list[tuple[str, object]] = []
    for block_index, block in enumerate(model.blocks):
        adapter = getattr(block, "trace_adapter", None)
        if adapter is not None:
            adapters.append((f"block_{block_index}", adapter))
    return adapters


def native_trace_slot_distribution(
    slot_indices: torch.Tensor,
    slot_gates: torch.Tensor,
    *,
    slot_count: int,
    context: str,
) -> dict[str, object]:
    if slot_count <= 0:
        raise ValueError("slot_count must be positive for native trace diagnostics.")
    if slot_indices.shape != slot_gates.shape:
        raise ValueError(
            "Native trace slot indices and gates must have the same shape: "
            f"indices={tuple(slot_indices.shape)}, gates={tuple(slot_gates.shape)} ({context})."
        )
    if not torch.isfinite(slot_gates).all():
        raise FloatingPointError(f"Non-finite native trace gates detected ({context}).")
    flat_indices = slot_indices.reshape(-1).long()
    flat_gates = slot_gates.reshape(-1).float()
    if flat_indices.numel() <= 0:
        raise ValueError(f"Native trace diagnostics received zero slot assignments ({context}).")
    if int(flat_indices.min().item()) < 0 or int(flat_indices.max().item()) >= slot_count:
        raise ValueError(f"Native trace slot assignment out of range ({context}).")

    weights = torch.zeros(slot_count, dtype=torch.float32)
    counts = torch.zeros(slot_count, dtype=torch.float32)
    weights.scatter_add_(0, flat_indices, flat_gates)
    counts.scatter_add_(0, flat_indices, torch.ones_like(flat_gates))
    total_weight = float(weights.sum().item())
    if total_weight <= EPS:
        probabilities = torch.full((slot_count,), 1.0 / float(slot_count), dtype=torch.float32)
    else:
        probabilities = weights / total_weight
    active = probabilities > EPS
    active_count = int(active.sum().item())
    if active_count <= 1:
        entropy = 0.0
    else:
        active_probs = probabilities[active]
        entropy = float((-(active_probs * torch.log(active_probs)).sum() / math.log(float(slot_count))).item())
    max_share = float(probabilities.max().item())
    return {
        "slot_weights": [float(value) for value in weights.tolist()],
        "slot_counts": [float(value) for value in counts.tolist()],
        "slot_probabilities": [float(value) for value in probabilities.tolist()],
        "active_slot_count": float(active_count),
        "active_slot_fraction": float(active_count / slot_count),
        "slot_entropy": entropy,
        "max_slot_share": max_share,
    }


def native_trace_diagnostics_for_inputs(
    model: DecoderTransformer,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    context: str,
) -> dict[str, object]:
    adapters = native_trace_adapters(model)
    if not adapters:
        return {
            "native_trace_adapter_count": 0.0,
            "native_trace_slot_entropy_mean": 0.0,
            "native_trace_slot_active_fraction_mean": 0.0,
            "native_trace_slot_max_share_mean": 0.0,
            "native_trace_slot_records": [],
        }

    model.eval()
    with torch.no_grad():
        logits, _ = model(inputs.to(device))
        require_finite_tensor("native_trace_diagnostic_logits", logits, context)

    records: list[dict[str, object]] = []
    for layer_id, adapter in adapters:
        slot_indices = getattr(adapter, "last_top_indices", None)
        slot_gates = getattr(adapter, "last_top_gates", None)
        if slot_indices is None or slot_gates is None:
            raise RuntimeError(f"Native trace adapter {layer_id} did not record slot usage ({context}).")
        slot_count = int(getattr(adapter, "n_slots"))
        distribution = native_trace_slot_distribution(
            slot_indices,
            slot_gates,
            slot_count=slot_count,
            context=f"{context}, layer={layer_id}",
        )
        records.append({"layer_id": layer_id, **distribution})

    entropies = [float(record["slot_entropy"]) for record in records]
    active_fractions = [float(record["active_slot_fraction"]) for record in records]
    max_shares = [float(record["max_slot_share"]) for record in records]
    return {
        "native_trace_adapter_count": float(len(records)),
        "native_trace_slot_entropy_mean": float(sum(entropies) / len(entropies)),
        "native_trace_slot_active_fraction_mean": float(sum(active_fractions) / len(active_fractions)),
        "native_trace_slot_max_share_mean": float(sum(max_shares) / len(max_shares)),
        "native_trace_slot_records": records,
    }


def native_trace_slot_target_loss(
    model: DecoderTransformer,
    anchors: Sequence[LayerActivationAnchor],
    *,
    device: torch.device,
) -> torch.Tensor:
    targeted = [anchor for anchor in anchors if anchor.native_slot_targets]
    if not targeted:
        raise ValueError("native_trace_slot_target_loss requires at least one anchor with native slot targets.")

    losses: list[torch.Tensor] = []
    adapters_by_layer = dict(native_trace_adapters(model))
    for anchor in targeted:
        logits, _ = model(anchor.inputs.to(device))
        require_finite_tensor("native_trace_slot_target_logits", logits, f"anchor={anchor.anchor_id}")
        mask = anchor_weights(anchor).to(device)
        if mask.shape != anchor.inputs.shape:
            raise ValueError(
                f"Anchor {anchor.anchor_id!r} slot loss mask shape {tuple(mask.shape)} "
                f"does not match inputs shape {tuple(anchor.inputs.shape)}."
            )
        denom = mask.sum()
        if denom.item() <= 0.0:
            raise ValueError(f"Anchor {anchor.anchor_id!r} has no weighted positions for native slot loss.")
        for layer_id, target_slot in anchor.native_slot_targets.items():
            if layer_id not in adapters_by_layer:
                raise ValueError(f"Anchor {anchor.anchor_id!r} targets unknown native trace layer {layer_id!r}.")
            adapter = adapters_by_layer[layer_id]
            scores = getattr(adapter, "last_scores", None)
            if scores is None:
                raise RuntimeError(f"Native trace adapter {layer_id!r} did not record scores for slot loss.")
            if scores.ndim != 3:
                raise ValueError(f"Native trace adapter scores must be [batch, seq, slots], got {tuple(scores.shape)}.")
            slot_count = scores.shape[-1]
            if target_slot < 0 or target_slot >= slot_count:
                raise ValueError(
                    f"Native slot target {target_slot} for anchor {anchor.anchor_id!r} "
                    f"is outside slot count {slot_count}."
                )
            targets = torch.full(scores.shape[:2], int(target_slot), dtype=torch.long, device=device)
            token_losses = F.cross_entropy(
                scores.reshape(-1, slot_count),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            loss = (token_losses * mask).sum() / denom
            require_finite_tensor("native_trace_slot_target_loss", loss, f"anchor={anchor.anchor_id}, layer={layer_id}")
            losses.append(anchor.importance * loss)
    result = torch.stack(losses).mean()
    require_finite_tensor("native_trace_slot_target_loss", result, f"anchor_count={len(targeted)}")
    return result


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    require_finite_tensor("cosine_left", a, "cosine")
    require_finite_tensor("cosine_right", b, "cosine")
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = float(a_flat.norm().item() * b_flat.norm().item())
    if denom <= EPS:
        return 0.0
    return float(torch.dot(a_flat, b_flat).item() / denom)


def anchor_weights(anchor: LayerActivationAnchor) -> torch.Tensor:
    if anchor.mask is None:
        return torch.ones(anchor.hidden_target.shape[:2], dtype=torch.float32)
    return anchor.mask.float()


def pooled_anchor_vector(anchor: LayerActivationAnchor) -> torch.Tensor:
    hidden = anchor.hidden_target.float()
    weights = anchor_weights(anchor)
    denom = weights.sum()
    if denom.item() <= 0.0:
        raise ValueError(f"Anchor {anchor.anchor_id!r} has no weighted positions.")
    pooled = (hidden * weights.unsqueeze(-1)).sum(dim=(0, 1)) / denom
    return pooled.detach().cpu()


def anchor_breadth_depth(anchor: LayerActivationAnchor, args: argparse.Namespace) -> tuple[float, float, float]:
    hidden = anchor.hidden_target.float()
    require_finite_tensor("anchor_hidden_target", hidden, f"anchor={anchor.anchor_id}")
    weights = anchor_weights(anchor).bool()
    selected = hidden[weights]
    if selected.numel() == 0:
        raise ValueError(f"Anchor {anchor.anchor_id!r} has no selected hidden states.")
    abs_values = selected.abs()
    active = abs_values > args.breadth_threshold
    breadth = float(active.float().mean().item())
    if active.any():
        depth_raw = float(abs_values[active].mean().item())
    else:
        depth_raw = 0.0
    if args.depth_scale <= 0:
        raise ValueError("--depth-scale must be positive.")
    depth = float(torch.sigmoid(torch.tensor((depth_raw - args.depth_center) / args.depth_scale)).item())
    return breadth, depth, depth_raw


def trace_activation_energy(vector: torch.Tensor) -> float:
    require_finite_tensor("trace_vector", vector, "trace_activation_energy")
    if vector.numel() <= 0:
        raise ValueError("Cannot compute trace activation energy for an empty vector.")
    rms = torch.sqrt((vector.float() ** 2).mean())
    require_finite_tensor("trace_activation_rms", rms, "trace_activation_energy")
    return float(torch.tanh(rms).detach().cpu())


def trace_consistency(radius_sq: float) -> float:
    require_finite_float("trace_radius_sq", radius_sq, "trace_consistency")
    if radius_sq < 0:
        raise ValueError("Trace radius must be non-negative.")
    return float(math.exp(-radius_sq))


def routing_entropy(similarities: Sequence[float]) -> float:
    positive = [max(0.0, float(value)) for value in similarities]
    if not positive:
        return 0.0
    total = sum(positive)
    if total <= EPS:
        return 0.0
    probabilities = [value / total for value in positive if value > 0.0]
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return float(entropy / math.log(len(probabilities)))


def anchor_prediction_loss(
    model: DecoderTransformer,
    anchor: LayerActivationAnchor,
    *,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        inputs = anchor.inputs.to(device)
        targets = anchor.targets.to(device)
        logits, _ = model(inputs)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        if anchor.mask is None:
            value = float(token_losses.mean().detach().cpu())
            require_finite_float("anchor_prediction_loss", value, f"anchor={anchor.anchor_id}")
            return value
        mask = anchor.mask.to(device)
        denom = mask.sum()
        if denom.item() <= 0.0:
            raise ValueError(f"Anchor {anchor.anchor_id!r} has an empty loss mask.")
        value = float(((token_losses * mask).sum() / denom).detach().cpu())
        require_finite_float("anchor_prediction_loss", value, f"anchor={anchor.anchor_id}")
        return value


def normalized_error(loss: float, args: argparse.Namespace) -> float:
    require_finite_float("loss", loss, "normalized_error")
    if args.error_scale <= 0:
        raise ValueError("--error-scale must be positive.")
    return float(torch.sigmoid(torch.tensor((loss - args.error_center) / args.error_scale)).item())


def compatible_for_reinforce(old: LayerActivationAnchor, new: LayerActivationAnchor) -> bool:
    if old.layer_id != new.layer_id:
        return False
    if old.hidden_target.shape != new.hidden_target.shape:
        return False
    if old.inputs.shape != new.inputs.shape or old.targets.shape != new.targets.shape:
        return False
    if not torch.equal(old.inputs, new.inputs):
        return False
    if not torch.equal(old.targets, new.targets):
        return False
    if old.mask is None and new.mask is None:
        return True
    if old.mask is None or new.mask is None:
        return False
    return bool(torch.equal(old.mask, new.mask))


def reinforce_anchor(
    old: LayerActivationAnchor,
    new: LayerActivationAnchor,
    merge_rate: float,
    importance: float,
) -> LayerActivationAnchor:
    if not compatible_for_reinforce(old, new):
        raise ValueError(f"Cannot reinforce incompatible anchors {old.anchor_id!r} and {new.anchor_id!r}.")
    if not 0.0 <= merge_rate <= 1.0:
        raise ValueError("--merge-rate must be in [0, 1].")
    hidden_target = (1.0 - merge_rate) * old.hidden_target + merge_rate * new.hidden_target
    return LayerActivationAnchor(
        anchor_id=old.anchor_id,
        chunk_id=old.chunk_id,
        source_type=old.source_type,
        text=old.text,
        inputs=old.inputs,
        targets=old.targets,
        hidden_target=hidden_target.detach().cpu(),
        mask=old.mask,
        importance=importance,
        layer_id=old.layer_id,
        native_slot_targets=dict(old.native_slot_targets or new.native_slot_targets),
    )


def capture_layer_anchor(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    spec: object,
    *,
    anchor_id: str,
    chunk_id: str,
    layer_id: str,
    max_seq_len: int,
    pad_id: int,
    device: torch.device,
    importance: float,
    qa_anchor_mode: str,
) -> LayerActivationAnchor:
    source_type = str(getattr(spec, "source_type"))
    text = str(getattr(spec, "text"))
    prompt = getattr(spec, "prompt")
    mask = None
    if source_type in {"qa", "composition_qa"} and qa_anchor_mode == "answer_tokens":
        if prompt is None:
            raise RuntimeError("QA answer-token anchor requires the original prompt dictionary.")
        supervision = make_qa_supervision([prompt], tokenizer, max_seq_len, pad_id)
        if supervision is None:
            raise RuntimeError("QA anchor supervision unexpectedly returned None for one prompt.")
        inputs, targets, answer_mask = supervision
        mask = answer_mask
    else:
        inputs, targets = encode_lm_tensors(text, tokenizer, max_seq_len, pad_id)

    inputs = inputs.to(device)
    model.eval()
    with torch.no_grad():
        hidden = layer_hidden(model, inputs, layer_id)
        require_finite_tensor("captured_layer_hidden", hidden, f"anchor={anchor_id}, layer={layer_id}")
    return LayerActivationAnchor(
        anchor_id=anchor_id,
        chunk_id=chunk_id,
        source_type=source_type,
        text=text,
        inputs=inputs.detach().cpu(),
        targets=targets.detach().cpu(),
        hidden_target=hidden.detach().cpu(),
        mask=None if mask is None else mask.detach().cpu(),
        importance=importance,
        layer_id=layer_id,
    )


def capture_semantic_margin_anchor(
    tokenizer: Tokenizer,
    prompt: dict[str, str],
    *,
    anchor_id: str,
    source_anchor_id: str,
    chunk_id: str,
    source_type: str,
    answer_vocab: Sequence[str],
    max_negatives: int,
    max_seq_len: int,
    pad_id: int,
    importance: float,
    cluster_source: str,
) -> SemanticMarginAnchor:
    question = prompt["question"]
    answer = prompt["answer"].strip()
    prompt_ids = tokenizer.encode(format_qa_prompt(question)).ids
    if not prompt_ids:
        raise ValueError(f"Prompt encoded to zero tokens: {question!r}")
    if len(prompt_ids) > max_seq_len:
        raise ValueError(
            f"Semantic margin prompt exceeds max_seq_len={max_seq_len}: "
            f"question={question!r}, tokens={len(prompt_ids)}"
        )
    correct_answer_ids = torch.tensor(encode_answer_ids(tokenizer, answer), dtype=torch.long)
    negative_answer_ids = negative_answer_ids_for(tokenizer, answer_vocab, answer, max_negatives)
    answer_supervision = make_qa_supervision([{"question": question, "answer": answer}], tokenizer, max_seq_len, pad_id)
    if answer_supervision is None:
        raise RuntimeError("Semantic answer supervision unexpectedly returned None for one prompt.")
    answer_inputs, answer_targets, answer_mask = answer_supervision
    return SemanticMarginAnchor(
        anchor_id=anchor_id,
        source_anchor_id=source_anchor_id,
        chunk_id=chunk_id,
        source_type=source_type,
        question=question,
        answer=answer,
        prompt_inputs=torch.tensor([prompt_ids], dtype=torch.long),
        correct_answer_ids=correct_answer_ids,
        negative_answer_ids=negative_answer_ids,
        answer_inputs=answer_inputs,
        answer_targets=answer_targets,
        answer_mask=answer_mask,
        importance=importance,
        cluster_source=cluster_source,
    )


def semantic_margin_loss(
    model: DecoderTransformer,
    anchors: Sequence[SemanticMarginAnchor],
    *,
    device: torch.device,
    margin: float,
    loss_mode: str,
    answer_sequence_weight: float,
) -> torch.Tensor:
    if not anchors:
        raise ValueError("semantic_margin_loss requires at least one semantic anchor.")
    if margin < 0:
        raise ValueError("--semantic-margin must be non-negative.")
    if answer_sequence_weight < 0:
        raise ValueError("--semantic-answer-sequence-weight must be non-negative.")
    losses = []
    for anchor in anchors:
        prompt_inputs = anchor.prompt_inputs.to(device)
        correct_id = anchor.correct_answer_ids[0].to(device)
        negatives = anchor.negative_answer_ids.to(device)
        if negatives.numel() <= 0:
            raise ValueError(f"Semantic anchor {anchor.anchor_id!r} has no negative answers.")
        logits, _ = model(prompt_inputs)
        next_logits = logits[:, -1, :]
        correct_logit = next_logits[:, correct_id]
        negative_logits = next_logits.index_select(dim=-1, index=negatives)
        hardest_negative = negative_logits.max(dim=-1).values
        actual_margin = correct_logit - hardest_negative
        require_finite_tensor("semantic_actual_margin", actual_margin, f"anchor={anchor.anchor_id}")
        violation = F.relu(margin - actual_margin)
        if loss_mode == "squared_hinge":
            raw_loss = violation.pow(2).mean()
        elif loss_mode == "hinge":
            raw_loss = violation.mean()
        else:
            raise ValueError("--semantic-margin-loss must be hinge or squared_hinge.")
        if answer_sequence_weight > 0:
            sequence_loss = semantic_answer_sequence_loss(model, anchor, device=device)
            raw_loss = raw_loss + answer_sequence_weight * sequence_loss
        anchor_loss = anchor.importance * raw_loss
        require_finite_tensor("semantic_margin_loss", anchor_loss, f"anchor={anchor.anchor_id}")
        losses.append(anchor_loss)
    loss = torch.stack(losses).mean()
    require_finite_tensor("semantic_margin_loss", loss, f"anchor_count={len(anchors)}")
    return loss


def semantic_answer_sequence_loss(
    model: DecoderTransformer,
    anchor: SemanticMarginAnchor,
    *,
    device: torch.device,
) -> torch.Tensor:
    inputs = anchor.answer_inputs.to(device)
    targets = anchor.answer_targets.to(device)
    mask = anchor.answer_mask.to(device)
    logits, _ = model(inputs)
    loss = masked_cross_entropy(logits, targets, mask)
    require_finite_tensor("semantic_answer_sequence_loss", loss, f"anchor={anchor.anchor_id}")
    return loss


def semantic_answer_sequence_stats(
    model: DecoderTransformer,
    anchor: SemanticMarginAnchor,
    *,
    device: torch.device,
) -> dict[str, float]:
    inputs = anchor.answer_inputs.to(device)
    targets = anchor.answer_targets.to(device)
    mask = anchor.answer_mask.to(device)
    logits, _ = model(inputs)
    loss = masked_cross_entropy(logits, targets, mask)
    predictions = logits.argmax(dim=-1)
    token_correct = (predictions == targets).to(torch.float32) * mask
    denom = mask.sum()
    if denom.item() <= 0.0:
        raise ValueError(f"Semantic anchor {anchor.anchor_id!r} has an empty answer mask.")
    token_accuracy = token_correct.sum() / denom
    exact_match = (token_correct.sum() == denom).to(torch.float32)
    require_finite_tensor("semantic_answer_sequence_loss", loss, f"anchor={anchor.anchor_id}")
    require_finite_tensor("semantic_answer_token_accuracy", token_accuracy, f"anchor={anchor.anchor_id}")
    return {
        "answer_loss": float(loss.detach().cpu()),
        "answer_token_accuracy": float(token_accuracy.detach().cpu()),
        "answer_exact_match": float(exact_match.detach().cpu()),
    }


def semantic_anchor_margin_value(
    model: DecoderTransformer,
    anchor: SemanticMarginAnchor,
    *,
    device: torch.device,
) -> float:
    prompt_inputs = anchor.prompt_inputs.to(device)
    correct_id = anchor.correct_answer_ids[0].to(device)
    negatives = anchor.negative_answer_ids.to(device)
    logits, _ = model(prompt_inputs)
    next_logits = logits[:, -1, :]
    correct_logit = next_logits[:, correct_id]
    negative_logits = next_logits.index_select(dim=-1, index=negatives)
    hardest_negative = negative_logits.max(dim=-1).values
    actual_margin = correct_logit - hardest_negative
    require_finite_tensor("semantic_actual_margin", actual_margin, f"anchor={anchor.anchor_id}")
    return float(actual_margin.squeeze(0).detach().cpu())


def select_semantic_anchor_batch(
    model: DecoderTransformer,
    anchors: Sequence[SemanticMarginAnchor],
    global_step: int,
    batch_size: int,
    *,
    device: torch.device,
    selection: str,
) -> list[SemanticMarginAnchor]:
    if not anchors:
        raise ValueError("select_semantic_anchor_batch requires at least one anchor.")
    if batch_size <= 0:
        raise ValueError("--semantic-anchor-batch-size must be positive.")
    if batch_size >= len(anchors):
        return list(anchors)
    if selection == "round_robin":
        return list(select_anchor_batch(anchors, global_step, batch_size))
    if selection == "worst_margin":
        scored: list[tuple[float, int, SemanticMarginAnchor]] = []
        with torch.no_grad():
            for index, anchor in enumerate(anchors):
                margin_value = semantic_anchor_margin_value(model, anchor, device=device)
                scored.append((margin_value, index, anchor))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [anchor for _, _, anchor in scored[:batch_size]]
    if selection == "worst_answer_loss":
        scored = []
        with torch.no_grad():
            for index, anchor in enumerate(anchors):
                answer_stats = semantic_answer_sequence_stats(model, anchor, device=device)
                scored.append((answer_stats["answer_loss"], index, anchor))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [anchor for _, _, anchor in scored[:batch_size]]
    raise ValueError("--semantic-anchor-selection must be round_robin, worst_margin, or worst_answer_loss.")


def evaluate_semantic_margins(
    model: DecoderTransformer,
    anchors: Sequence[SemanticMarginAnchor],
    *,
    device: torch.device,
    margin: float,
) -> dict[str, float]:
    if not anchors:
        return {
            "semantic_margin_mean": 0.0,
            "semantic_margin_min": 0.0,
            "semantic_margin_violation_rate": 0.0,
            "semantic_answer_loss_mean": 0.0,
            "semantic_answer_loss_max": 0.0,
            "semantic_answer_token_accuracy_mean": 0.0,
            "semantic_answer_exact_match_rate": 0.0,
            "semantic_margin_source_count": 0.0,
            "semantic_margin_cluster_count": 0.0,
            "semantic_margin_source_violation_rate": 0.0,
            "semantic_margin_cluster_violation_rate": 0.0,
            "semantic_margin_records": [],
            "semantic_margin_worst": {},
        }
    records: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for anchor in anchors:
            prompt_inputs = anchor.prompt_inputs.to(device)
            negatives = anchor.negative_answer_ids.to(device)
            correct_id = anchor.correct_answer_ids[0].to(device)
            logits, _ = model(prompt_inputs)
            next_logits = logits[:, -1, :]
            correct_logit = next_logits[:, correct_id]
            negative_logits = next_logits.index_select(dim=-1, index=negatives)
            hardest_negative = negative_logits.max(dim=-1).values
            hardest_negative_index = int(negative_logits.argmax(dim=-1).item())
            hardest_negative_id = int(negatives[hardest_negative_index].detach().cpu())
            actual_margin = correct_logit - hardest_negative
            require_finite_tensor("semantic_actual_margin", actual_margin, f"anchor={anchor.anchor_id}")
            actual_margin_float = float(actual_margin.squeeze(0).detach().cpu())
            answer_stats = semantic_answer_sequence_stats(model, anchor, device=device)
            records.append(
                {
                    "anchor_id": anchor.anchor_id,
                    "source_anchor_id": anchor.source_anchor_id,
                    "chunk_id": anchor.chunk_id,
                    "source_type": anchor.source_type,
                    "cluster_source": anchor.cluster_source,
                    "question": anchor.question,
                    "answer": anchor.answer,
                    "correct_first_token_id": int(anchor.correct_answer_ids[0].detach().cpu()),
                    "hardest_negative_token_id": hardest_negative_id,
                    "margin": actual_margin_float,
                    "violated": actual_margin_float < margin,
                    **answer_stats,
                }
            )
    margins = [float(record["margin"]) for record in records]
    violations = [1.0 if value < margin else 0.0 for value in margins]
    answer_losses = [float(record["answer_loss"]) for record in records]
    answer_token_accuracies = [float(record["answer_token_accuracy"]) for record in records]
    answer_exact_matches = [float(record["answer_exact_match"]) for record in records]
    source_records = [record for record in records if record["cluster_source"] == "source"]
    cluster_records = [record for record in records if record["cluster_source"] == "cluster"]
    source_violations = [1.0 if float(record["margin"]) < margin else 0.0 for record in source_records]
    cluster_violations = [1.0 if float(record["margin"]) < margin else 0.0 for record in cluster_records]
    worst = min(records, key=lambda record: float(record["margin"]))
    return {
        "semantic_margin_mean": float(sum(margins) / len(margins)),
        "semantic_margin_min": float(min(margins)),
        "semantic_margin_violation_rate": float(sum(violations) / len(violations)),
        "semantic_answer_loss_mean": float(sum(answer_losses) / len(answer_losses)),
        "semantic_answer_loss_max": float(max(answer_losses)),
        "semantic_answer_token_accuracy_mean": float(sum(answer_token_accuracies) / len(answer_token_accuracies)),
        "semantic_answer_exact_match_rate": float(sum(answer_exact_matches) / len(answer_exact_matches)),
        "semantic_margin_source_count": float(len(source_records)),
        "semantic_margin_cluster_count": float(len(cluster_records)),
        "semantic_margin_source_violation_rate": (
            float(sum(source_violations) / len(source_violations)) if source_violations else 0.0
        ),
        "semantic_margin_cluster_violation_rate": (
            float(sum(cluster_violations) / len(cluster_violations)) if cluster_violations else 0.0
        ),
        "semantic_margin_records": records,
        "semantic_margin_worst": worst,
    }


def layer_anchor_drift_loss(
    model: DecoderTransformer,
    anchors: Sequence[LayerActivationAnchor],
    *,
    device: torch.device,
    normalization: str,
) -> torch.Tensor:
    if not anchors:
        raise ValueError("layer_anchor_drift_loss requires at least one anchor.")
    losses = []
    for anchor in anchors:
        drift = layer_anchor_drift_tensor(model, anchor, device=device, normalization=normalization)
        losses.append(anchor.importance * drift)
    return torch.stack(losses).mean()


def masked_token_mean(values: torch.Tensor, mask: torch.Tensor | None, anchor_id: str) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(values.device)
    denom = mask.sum()
    if denom.item() <= 0.0:
        raise ValueError(f"Anchor {anchor_id!r} has an empty mask.")
    return (values * mask).sum() / denom


def layer_anchor_drift_tensor(
    model: DecoderTransformer,
    anchor: LayerActivationAnchor,
    *,
    device: torch.device,
    normalization: str,
) -> torch.Tensor:
    inputs = anchor.inputs.to(device)
    target = anchor.hidden_target.to(device)
    mask = None if anchor.mask is None else anchor.mask.to(device)
    hidden = layer_hidden(model, inputs, anchor.layer_id)
    require_finite_tensor("layer_hidden", hidden, f"anchor={anchor.anchor_id}, layer={anchor.layer_id}")
    token_losses = ((hidden - target) ** 2).mean(dim=-1)
    drift = masked_token_mean(token_losses, mask, anchor.anchor_id)
    require_finite_tensor("layer_anchor_drift", drift, f"anchor={anchor.anchor_id}, layer={anchor.layer_id}")
    if normalization == "none":
        return drift
    if normalization == "target_energy":
        target_energy = masked_token_mean((target**2).mean(dim=-1), mask, anchor.anchor_id)
        require_finite_tensor("layer_anchor_target_energy", target_energy, f"anchor={anchor.anchor_id}")
        if target_energy.item() <= EPS:
            raise ValueError(
                f"Anchor {anchor.anchor_id!r} has target energy {target_energy.item():.6g}; "
                "cannot use --drift-normalization target_energy."
            )
        return drift / target_energy
    raise ValueError(f"Unknown drift normalization: {normalization!r}")


def layer_replay_anchor_loss(
    model: DecoderTransformer,
    anchors: Sequence[LayerActivationAnchor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not anchors:
        raise ValueError("layer_replay_anchor_loss requires at least one anchor.")
    losses = []
    for anchor in anchors:
        inputs = anchor.inputs.to(device)
        targets = anchor.targets.to(device)
        logits, _ = model(inputs)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        if anchor.mask is None:
            replay_loss = token_losses.mean()
        else:
            mask = anchor.mask.to(device)
            denom = mask.sum()
            if denom.item() <= 0.0:
                raise ValueError(f"Anchor {anchor.anchor_id!r} has an empty replay mask.")
            replay_loss = (token_losses * mask).sum() / denom
        losses.append(anchor.importance * replay_loss)
    return torch.stack(losses).mean()


def pending_trace_loss(
    model: DecoderTransformer,
    traces: Sequence[PendingConcept],
    *,
    global_step: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not traces:
        raise ValueError("pending_trace_loss requires at least one pending trace.")
    losses: list[torch.Tensor] = []
    native_slot_losses: list[torch.Tensor] = []
    for trace in traces:
        if trace.strength <= 0.0:
            continue
        if not trace.anchors and not trace.semantic_anchors:
            continue
        trace_terms: list[torch.Tensor] = []
        trace_native_slot_loss: torch.Tensor | None = None
        if trace.semantic_anchors and args.semantic_margin_weight > 0:
            selected_semantic = select_semantic_anchor_batch(
                model,
                trace.semantic_anchors,
                global_step,
                args.semantic_anchor_batch_size,
                device=device,
                selection=args.semantic_anchor_selection,
            )
            semantic_loss = semantic_margin_loss(
                model,
                selected_semantic,
                device=device,
                margin=args.semantic_margin,
                loss_mode=args.semantic_margin_loss,
                answer_sequence_weight=args.semantic_answer_sequence_weight,
            )
            trace_terms.append(args.semantic_margin_weight * semantic_loss)
        if trace.anchors and args.anchor_drift_weight > 0:
            selected_anchors = select_anchor_batch(trace.anchors, global_step, args.anchor_batch_size)
            anchor_loss = layer_anchor_drift_loss(
                model,
                selected_anchors,
                device=device,
                normalization=args.drift_normalization,
            )
            trace_terms.append(args.anchor_drift_weight * anchor_loss)
        if trace.anchors and args.native_trace_slot_loss_weight > 0:
            selected_slot_anchors = select_anchor_batch(trace.anchors, global_step, args.anchor_batch_size)
            slot_loss = native_trace_slot_target_loss(model, selected_slot_anchors, device=device)
            trace_terms.append(args.native_trace_slot_loss_weight * slot_loss)
            trace_native_slot_loss = slot_loss
        if trace_terms:
            trace_loss = torch.stack(trace_terms).sum()
            require_finite_tensor("pending_trace_inner_loss", trace_loss, f"pending={trace.pending_id}")
            losses.append(float(trace.strength) * trace_loss)
            if trace_native_slot_loss is None:
                native_slot_losses.append(trace_loss.new_zeros(()))
            else:
                native_slot_losses.append(float(trace.strength) * trace_native_slot_loss)
    if not losses:
        raise ValueError("No pending traces had trainable anchors for pending_trace_loss.")
    loss = torch.stack(losses).mean()
    require_finite_tensor("pending_trace_loss", loss, f"trace_count={len(traces)}")
    native_slot_loss = torch.stack(native_slot_losses).mean()
    require_finite_tensor("pending_trace_native_slot_loss", native_slot_loss, f"trace_count={len(traces)}")
    return loss, native_slot_loss


def evaluate_layer_anchor_drift(
    model: DecoderTransformer,
    anchors: Sequence[LayerActivationAnchor],
    *,
    device: torch.device,
    normalization: str,
) -> dict[str, float]:
    if not anchors:
        return {"anchor_drift_mean": 0.0, "anchor_drift_max": 0.0}
    values = []
    model.eval()
    with torch.no_grad():
        for anchor in anchors:
            drift = layer_anchor_drift_tensor(model, anchor, device=device, normalization=normalization)
            values.append(float(drift.detach().cpu()))
    return {
        "anchor_drift_mean": float(sum(values) / len(values)),
        "anchor_drift_max": float(max(values)),
    }


def layer_drift_breakdown(
    model: DecoderTransformer,
    anchors: Sequence[LayerActivationAnchor],
    *,
    device: torch.device,
    normalization: str,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    by_layer: dict[str, list[LayerActivationAnchor]] = {}
    for anchor in anchors:
        by_layer.setdefault(anchor.layer_id, []).append(anchor)
    for layer_id, layer_anchors in sorted(by_layer.items()):
        result[layer_id] = evaluate_layer_anchor_drift(
            model,
            layer_anchors,
            device=device,
            normalization=normalization,
        )
    return result


def layer_allowed(layer_id: str, selected_layers: set[str] | None) -> bool:
    return selected_layers is None or layer_id in selected_layers


class LivingMap:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.concepts: list[LivingConcept] = []
        self.pending: list[PendingConcept] = []
        self.next_pending_id = 0
        self.counters: dict[str, int] = {
            "create": 0,
            "reinforce": 0,
            "attach": 0,
            "replacement_fuse": 0,
            "ignore": 0,
            "defer": 0,
            "retire": 0,
            "maintenance_repairs": 0,
        }

    def native_slot_usage_counts(self, model: DecoderTransformer) -> dict[str, list[int]]:
        usage: dict[str, list[int]] = {}
        for layer_id, adapter in native_trace_adapters(model):
            slot_count = int(getattr(adapter, "n_slots"))
            usage[layer_id] = [0 for _ in range(slot_count)]
        for pending in self.active_pending_traces():
            for layer_id, slot_index in pending.native_slot_targets.items():
                if layer_id in usage:
                    usage[layer_id][slot_index] += 1
        for concept in self.active_concepts():
            for layer_id, slot_index in concept.native_slot_targets.items():
                if layer_id in usage:
                    usage[layer_id][slot_index] += 1
        return usage

    def ensure_pending_native_slot_targets(self, model: DecoderTransformer, pending: PendingConcept) -> None:
        if pending.native_slot_targets:
            return
        usage = self.native_slot_usage_counts(model)
        targets: dict[str, int] = {}
        for layer_id, counts in usage.items():
            if not counts:
                raise RuntimeError(f"Native trace layer {layer_id!r} has no slots.")
            min_count = min(counts)
            slot_index = next(index for index, count in enumerate(counts) if count == min_count)
            targets[layer_id] = slot_index
            counts[slot_index] += 1
        pending.native_slot_targets = targets

    def active_concepts(self) -> list[LivingConcept]:
        return [concept for concept in self.concepts if concept.status == "active"]

    def active_anchors(self) -> list[LayerActivationAnchor]:
        anchors: list[LayerActivationAnchor] = []
        for concept in self.active_concepts():
            anchors.extend(concept_anchor_list(concept))
        return anchors

    def active_semantic_anchors(self) -> list[SemanticMarginAnchor]:
        anchors: list[SemanticMarginAnchor] = []
        for concept in self.active_concepts():
            anchors.extend(concept_semantic_anchor_list(concept))
        return anchors

    def transformed_concepts(self) -> list[LivingConcept]:
        return [concept for concept in self.concepts if concept.status == "transformed"]

    def find_nearest_pending(
        self,
        vector: torch.Tensor,
        source_type: str,
        layer_id: str,
    ) -> tuple[PendingConcept | None, float]:
        best: PendingConcept | None = None
        best_sim = -1.0
        for pending in self.pending:
            if not self.args.allow_cross_layer_match and pending.layer_id != layer_id:
                continue
            if self.args.same_source_only and pending.source_type != source_type:
                continue
            sim = cosine(vector, pending.centroid)
            if sim > best_sim:
                best = pending
                best_sim = sim
        return best, best_sim

    def active_route_similarities(
        self,
        vector: torch.Tensor,
        source_type: str,
        layer_id: str,
    ) -> list[float]:
        similarities: list[float] = []
        for concept in self.active_concepts():
            if not self.args.allow_cross_layer_match and concept.layer_id != layer_id:
                continue
            centroid = concept_centroid(concept, source_type if self.args.same_source_only else None)
            if centroid is None:
                continue
            similarities.append(cosine(vector, centroid))
        return similarities

    def find_nearest_active(
        self,
        vector: torch.Tensor,
        source_type: str,
        layer_id: str,
    ) -> tuple[LivingConcept | None, float]:
        best: LivingConcept | None = None
        best_sim = -1.0
        for concept in self.active_concepts():
            if not self.args.allow_cross_layer_match and concept.layer_id != layer_id:
                continue
            centroid = concept_centroid(concept, source_type if self.args.same_source_only else None)
            if centroid is None:
                continue
            sim = cosine(vector, centroid)
            if sim > best_sim:
                best = concept
                best_sim = sim
        return best, best_sim

    def decay_pending_traces(self, step_index: int) -> None:
        if not 0.0 <= self.args.trace_decay <= 1.0:
            raise ValueError("--trace-decay must be in [0, 1].")
        if self.args.trace_prune_threshold < 0.0:
            raise ValueError("--trace-prune-threshold must be non-negative.")
        retained: list[PendingConcept] = []
        for pending in self.pending:
            age = step_index - pending.last_seen_step
            if age < 0:
                raise ValueError(
                    f"Pending trace {pending.pending_id!r} was last seen in the future: "
                    f"last_seen_step={pending.last_seen_step}, step_index={step_index}."
                )
            if age > 0:
                decay = self.args.trace_decay**age
                pending.strength *= decay
                pending.routed_energy *= decay
                pending.latest_pressure = pending.strength
            if pending.strength >= self.args.trace_prune_threshold:
                retained.append(pending)
        self.pending = retained

    def commit_pending_trace(self, pending: PendingConcept) -> None:
        self.pending = [item for item in self.pending if item.pending_id != pending.pending_id]

    def attach_anchor_to_pending_trace(
        self,
        pending: PendingConcept,
        anchor: LayerActivationAnchor,
        semantic_anchors: Sequence[SemanticMarginAnchor],
    ) -> None:
        anchor.native_slot_targets = dict(pending.native_slot_targets)
        replaced_anchor = False
        next_anchors: list[LayerActivationAnchor] = []
        for existing in pending.anchors:
            if existing.anchor_id == anchor.anchor_id:
                next_anchors.append(anchor)
                replaced_anchor = True
            else:
                next_anchors.append(existing)
        if not replaced_anchor:
            next_anchors.append(anchor)
        pending.anchors = next_anchors

        semantic_by_id = {existing.anchor_id: existing for existing in pending.semantic_anchors}
        for semantic_anchor in semantic_anchors:
            semantic_by_id[semantic_anchor.anchor_id] = semantic_anchor
        pending.semantic_anchors = list(semantic_by_id.values())

    def active_pending_traces(self) -> list[PendingConcept]:
        return [pending for pending in self.pending if pending.strength > 0.0]

    def pending_trace_payload(
        self,
        pending: PendingConcept,
        anchor: LayerActivationAnchor,
        semantic_anchors: Sequence[SemanticMarginAnchor],
    ) -> tuple[list[LayerActivationAnchor], list[SemanticMarginAnchor]]:
        self.attach_anchor_to_pending_trace(pending, anchor, semantic_anchors)
        if not pending.anchors:
            raise RuntimeError(f"Pending trace {pending.pending_id!r} has no anchors to commit.")
        anchor_payload = list(pending.anchors)
        semantic_payload = list(pending.semantic_anchors)
        return anchor_payload, semantic_payload

    def new_concept_from_trace(
        self,
        pending: PendingConcept,
        anchor: LayerActivationAnchor,
        semantic_anchors: Sequence[SemanticMarginAnchor],
        evidence: dict[str, float],
        pressure: float,
        step_index: int,
        *,
        lineage_id: str | None = None,
        transformed_from: str | None = None,
        transform_count: int = 0,
    ) -> LivingConcept:
        anchor_payload, semantic_payload = self.pending_trace_payload(pending, anchor, semantic_anchors)
        concept_id = f"concept_{len(self.concepts)}"
        importance = max(item.importance for item in anchor_payload)
        concept = LivingConcept(
            concept_id=concept_id,
            lineage_id=concept_id if lineage_id is None else lineage_id,
            anchor=anchor,
            layer_id=anchor.layer_id,
            status="active",
            created_step=step_index,
            last_seen_step=step_index,
            count=len(anchor_payload),
            importance=importance,
            tolerance=self.args.anchor_tolerance,
            stability=evidence["consistency"],
            pressure=pressure,
            transformed_from=transformed_from,
            transform_count=transform_count,
            evidence=evidence,
            anchors=anchor_payload,
            semantic_anchors=semantic_payload,
            native_slot_targets=dict(pending.native_slot_targets),
        )
        self.concepts.append(concept)
        return concept

    def append_pending_trace_to_concept(
        self,
        concept: LivingConcept,
        pending: PendingConcept,
        anchor: LayerActivationAnchor,
        semantic_anchors: Sequence[SemanticMarginAnchor],
        evidence: dict[str, float],
        pressure: float,
        step_index: int,
    ) -> None:
        anchor_payload, semantic_payload = self.pending_trace_payload(pending, anchor, semantic_anchors)
        anchor_by_id = {existing.anchor_id: existing for existing in concept_anchor_list(concept)}
        for payload_anchor in anchor_payload:
            anchor_by_id[payload_anchor.anchor_id] = payload_anchor
        concept.anchors = list(anchor_by_id.values())
        concept.anchor = anchor

        semantic_by_id = {existing.anchor_id: existing for existing in concept.semantic_anchors}
        for semantic_anchor in semantic_payload:
            semantic_by_id[semantic_anchor.anchor_id] = semantic_anchor
        concept.semantic_anchors = list(semantic_by_id.values())

        concept.count += len(anchor_payload)
        concept.importance = max(concept.importance, *(payload_anchor.importance for payload_anchor in anchor_payload))
        concept.stability = evidence["consistency"]
        concept.pressure = pressure
        concept.last_seen_step = step_index
        concept.evidence = evidence
        concept.native_slot_targets = dict(pending.native_slot_targets)

    def append_anchor_to_concept(
        self,
        concept: LivingConcept,
        anchor: LayerActivationAnchor,
        semantic_anchors: Sequence[SemanticMarginAnchor],
        evidence: dict[str, float],
        pressure: float,
        step_index: int,
    ) -> None:
        if any(existing.anchor_id == anchor.anchor_id for existing in concept_anchor_list(concept)):
            raise ValueError(f"Concept {concept.concept_id!r} already contains anchor {anchor.anchor_id!r}.")
        concept.anchors = [*concept_anchor_list(concept), anchor]
        existing_semantic_ids = {existing.anchor_id for existing in concept.semantic_anchors}
        for semantic_anchor in semantic_anchors:
            if semantic_anchor.anchor_id in existing_semantic_ids:
                raise ValueError(
                    f"Concept {concept.concept_id!r} already contains semantic anchor {semantic_anchor.anchor_id!r}."
                )
            existing_semantic_ids.add(semantic_anchor.anchor_id)
        concept.semantic_anchors = [*concept.semantic_anchors, *semantic_anchors]
        concept.count += 1
        concept.importance = max(concept.importance, anchor.importance)
        concept.stability = evidence["consistency"]
        concept.pressure = pressure
        concept.last_seen_step = step_index
        concept.evidence = evidence

    def replace_reinforced_anchor(
        self,
        concept: LivingConcept,
        old_anchor: LayerActivationAnchor,
        new_anchor: LayerActivationAnchor,
    ) -> None:
        anchors = concept_anchor_list(concept)
        replaced = False
        next_anchors: list[LayerActivationAnchor] = []
        for existing in anchors:
            if existing.anchor_id == old_anchor.anchor_id:
                next_anchors.append(new_anchor)
                replaced = True
            else:
                next_anchors.append(existing)
        if not replaced:
            raise ValueError(f"Concept {concept.concept_id!r} does not contain anchor {old_anchor.anchor_id!r}.")
        concept.anchors = next_anchors
        concept.anchor = new_anchor

    def update_pending(
        self,
        anchor: LayerActivationAnchor,
        vector: torch.Tensor,
        step_index: int,
    ) -> PendingConcept:
        if not 0.0 < self.args.trace_learning_rate <= 1.0:
            raise ValueError("--trace-learning-rate must be in (0, 1].")
        if not 0.0 < self.args.trace_centroid_rate <= 1.0:
            raise ValueError("--trace-centroid-rate must be in (0, 1].")
        nearest, nearest_sim = self.find_nearest_pending(vector, anchor.source_type, anchor.layer_id)
        if nearest is None or nearest_sim < self.args.pending_merge_similarity:
            pending_id = f"pending_{self.next_pending_id}"
            self.next_pending_id += 1
            pending = PendingConcept(
                pending_id=pending_id,
                source_type=anchor.source_type,
                layer_id=anchor.layer_id,
                centroid=vector.detach().clone(),
                count=0,
                variance=0.0,
                strength=0.0,
                radius_sq=0.0,
                routed_energy=0.0,
                last_seen_step=step_index,
                latest_anchor_id=anchor.anchor_id,
                latest_pressure=0.0,
            )
            self.pending.append(pending)
            route_affinity = 1.0
        else:
            pending = nearest
            route_affinity = max(0.0, nearest_sim)

        previous_centroid = pending.centroid.detach().clone()
        activation_energy = trace_activation_energy(vector)
        routed_signal = route_affinity * activation_energy
        pending.count += 1
        alpha = min(1.0, self.args.trace_centroid_rate * max(route_affinity, EPS))
        pending.centroid = (1.0 - alpha) * pending.centroid + alpha * vector.detach()
        distance = float(((vector - previous_centroid) ** 2).mean().item())
        require_finite_float("trace_distance", distance, f"pending={pending.pending_id}")
        pending.variance = (1.0 - alpha) * pending.variance + alpha * distance
        pending.radius_sq = pending.variance
        pending.routed_energy += routed_signal
        pending.strength = min(1.0, pending.strength + self.args.trace_learning_rate * routed_signal)
        pending.last_seen_step = step_index
        pending.latest_anchor_id = anchor.anchor_id
        pending.latest_pressure = pending.strength
        return pending

    def evidence_for_anchor(
        self,
        model: DecoderTransformer,
        anchor: LayerActivationAnchor,
        pending: PendingConcept,
        vector: torch.Tensor,
        *,
        device: torch.device,
    ) -> dict[str, float]:
        breadth, depth, depth_raw = anchor_breadth_depth(anchor, self.args)
        activation_energy = trace_activation_energy(vector)
        consistency = trace_consistency(pending.radius_sq)
        nearest_active, nearest_sim = self.find_nearest_active(vector, anchor.source_type, anchor.layer_id)
        positive_similarity = max(0.0, nearest_sim)
        novelty = 1.0 - positive_similarity if nearest_active is not None else 1.0
        loss = anchor_prediction_loss(model, anchor, device=device)
        error = normalized_error(loss, self.args)
        active_similarities = self.active_route_similarities(vector, anchor.source_type, anchor.layer_id)
        entropy = routing_entropy(active_similarities)
        route_count = float(sum(1 for value in active_similarities if value > 0.0))
        trace_strength = pending.strength
        base_pressure = trace_strength
        creation_pressure = trace_strength
        reinforce_pressure = trace_strength * positive_similarity
        evidence = {
            "breadth": breadth,
            "depth": depth,
            "depth_raw": depth_raw,
            "frequency": trace_strength,
            "consistency": consistency,
            "novelty": novelty,
            "nearest_active_present": 1.0 if nearest_active is not None else 0.0,
            "nearest_active_similarity": nearest_sim if nearest_active is not None else -1.0,
            "prediction_loss": loss,
            "error": error,
            "base_pressure": base_pressure,
            "creation_pressure": creation_pressure,
            "reinforce_pressure": reinforce_pressure,
            "pending_count": float(pending.count),
            "trace_strength": trace_strength,
            "trace_radius_sq": pending.radius_sq,
            "trace_activation_energy": activation_energy,
            "trace_routed_energy": pending.routed_energy,
            "trace_routing_entropy": entropy,
            "trace_active_route_count": route_count,
        }
        pending.latest_pressure = base_pressure
        pending.latest_evidence = evidence
        return evidence

    def apply_gate(
        self,
        anchor: LayerActivationAnchor,
        vector: torch.Tensor,
        pending: PendingConcept,
        evidence: dict[str, float],
        *,
        semantic_anchors: Sequence[SemanticMarginAnchor],
        step_index: int,
    ) -> str:
        nearest, nearest_sim = self.find_nearest_active(vector, anchor.source_type, anchor.layer_id)
        base_pressure = evidence["base_pressure"]
        creation_pressure = evidence["creation_pressure"]
        reinforce_pressure = evidence["reinforce_pressure"]
        trace_strength = evidence["trace_strength"]

        if trace_strength < self.args.pressure_threshold:
            self.counters["ignore"] += 1
            return "ignore"

        if nearest is not None and nearest_sim >= self.args.merge_similarity:
            compatible_anchor = next(
                (existing for existing in concept_anchor_list(nearest) if compatible_for_reinforce(existing, anchor)),
                None,
            )
            if compatible_anchor is None:
                if self.args.incompatible_merge_action == "defer":
                    self.counters["defer"] += 1
                    return "defer_incompatible_reinforce"
                if self.args.incompatible_merge_action == "create":
                    self.new_concept_from_trace(
                        pending,
                        anchor,
                        semantic_anchors,
                        evidence,
                        creation_pressure,
                        step_index,
                    )
                    self.counters["create"] += 1
                    self.commit_pending_trace(pending)
                    self.enforce_capacity(step_index)
                    return "create_incompatible_reinforce"
                if self.args.incompatible_merge_action == "attach":
                    self.append_pending_trace_to_concept(
                        nearest,
                        pending,
                        anchor,
                        semantic_anchors,
                        evidence,
                        reinforce_pressure,
                        step_index,
                    )
                    self.counters["attach"] += 1
                    self.commit_pending_trace(pending)
                    self.enforce_capacity(step_index)
                    return "attach_incompatible_reinforce"
                if self.args.incompatible_merge_action != "replacement_fuse":
                    raise ValueError(
                        "--incompatible-merge-action must be defer, create, attach, or replacement_fuse."
                    )
                nearest.status = "transformed"
                nearest.last_seen_step = step_index
                self.new_concept_from_trace(
                    pending,
                    anchor,
                    semantic_anchors,
                    evidence,
                    base_pressure,
                    step_index,
                    lineage_id=nearest.lineage_id,
                    transformed_from=nearest.concept_id,
                    transform_count=nearest.transform_count + 1,
                )
                self.counters["replacement_fuse"] += 1
                self.commit_pending_trace(pending)
                self.enforce_capacity(step_index)
                return "replacement_fuse_incompatible_reinforce"
            if nearest.native_slot_targets:
                pending.native_slot_targets = dict(nearest.native_slot_targets)
                anchor.native_slot_targets = dict(nearest.native_slot_targets)
            reinforced_anchor = reinforce_anchor(
                compatible_anchor,
                anchor,
                self.args.merge_rate,
                max(nearest.importance, anchor.importance),
            )
            self.replace_reinforced_anchor(nearest, compatible_anchor, reinforced_anchor)
            anchor_payload, semantic_payload = self.pending_trace_payload(pending, anchor, semantic_anchors)
            anchor_by_id = {existing.anchor_id: existing for existing in concept_anchor_list(nearest)}
            for payload_anchor in anchor_payload:
                if payload_anchor.anchor_id == anchor.anchor_id:
                    continue
                anchor_by_id[payload_anchor.anchor_id] = payload_anchor
            nearest.anchors = list(anchor_by_id.values())
            semantic_by_id = {existing.anchor_id: existing for existing in nearest.semantic_anchors}
            for semantic_anchor in semantic_payload:
                semantic_by_id[semantic_anchor.anchor_id] = semantic_anchor
            nearest.semantic_anchors = list(semantic_by_id.values())
            nearest.count += len(anchor_payload)
            nearest.importance = max(nearest.importance, anchor.importance)
            nearest.stability = evidence["consistency"]
            nearest.pressure = reinforce_pressure
            nearest.last_seen_step = step_index
            nearest.evidence = evidence
            nearest.anchor.importance = nearest.importance
            self.counters["reinforce"] += 1
            self.commit_pending_trace(pending)
            return "reinforce"

        if nearest is not None and nearest_sim >= self.args.fusion_similarity:
            nearest.status = "transformed"
            nearest.last_seen_step = step_index
            self.new_concept_from_trace(
                pending,
                anchor,
                semantic_anchors,
                evidence,
                base_pressure,
                step_index,
                lineage_id=nearest.lineage_id,
                transformed_from=nearest.concept_id,
                transform_count=nearest.transform_count + 1,
            )
            self.counters["replacement_fuse"] += 1
            self.commit_pending_trace(pending)
            self.enforce_capacity(step_index)
            return "replacement_fuse"

        self.new_concept_from_trace(
            pending,
            anchor,
            semantic_anchors,
            evidence,
            creation_pressure,
            step_index,
        )
        self.counters["create"] += 1
        self.commit_pending_trace(pending)
        self.enforce_capacity(step_index)
        return "create"

    def enforce_capacity(self, step_index: int) -> None:
        if self.args.max_active_concepts <= 0:
            return
        while len(self.active_concepts()) > self.args.max_active_concepts:
            active = self.active_concepts()
            if not active:
                return
            retiree = min(active, key=lambda concept: self.utility(concept, step_index))
            retiree.status = "retired"
            retiree.last_seen_step = step_index
            self.counters["retire"] += 1

    def utility(self, concept: LivingConcept, step_index: int) -> float:
        age = max(0, step_index - concept.last_seen_step)
        recency = math.exp(-float(age) / self.args.recency_tau) if self.args.recency_tau > 0 else 0.0
        return concept.importance * math.log1p(float(concept.count)) * concept.stability * recency

    def stats(self, model: DecoderTransformer, *, device: torch.device) -> dict[str, float]:
        active = self.active_concepts()
        transformed = self.transformed_concepts()
        active_anchors = self.active_anchors()
        pending_strengths = [pending.strength for pending in self.pending]
        pending_radii = [pending.radius_sq for pending in self.pending]
        active_strengths = [concept.pressure for concept in active]
        drift = evaluate_layer_anchor_drift(
            model,
            active_anchors,
            device=device,
            normalization=self.args.drift_normalization,
        )
        return {
            "active_concept_count": float(len(active)),
            "active_anchor_count": float(len(active_anchors)),
            "active_semantic_anchor_count": float(len(self.active_semantic_anchors())),
            "active_layer_count": float(len({anchor.layer_id for anchor in active_anchors})),
            "pending_concept_count": float(len(self.pending)),
            "pending_trace_strength_mean": (
                float(sum(pending_strengths) / len(pending_strengths)) if pending_strengths else 0.0
            ),
            "pending_trace_strength_max": float(max(pending_strengths)) if pending_strengths else 0.0,
            "pending_trace_radius_sq_mean": (
                float(sum(pending_radii) / len(pending_radii)) if pending_radii else 0.0
            ),
            "committed_trace_strength_mean": (
                float(sum(active_strengths) / len(active_strengths)) if active_strengths else 0.0
            ),
            "lineage_count": float(len({concept.lineage_id for concept in self.concepts})),
            "transformed_concept_count": float(len(transformed)),
            "created_count": float(self.counters["create"]),
            "reinforced_count": float(self.counters["reinforce"]),
            "attached_count": float(self.counters["attach"]),
            "replacement_fused_count": float(self.counters["replacement_fuse"]),
            "ignored_count": float(self.counters["ignore"]),
            "deferred_count": float(self.counters["defer"]),
            "retired_count": float(self.counters["retire"]),
            "maintenance_repairs": float(self.counters["maintenance_repairs"]),
            "destructive_drift_mean": drift["anchor_drift_mean"],
            "destructive_drift_max": drift["anchor_drift_max"],
        }


def train_chunk(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    chunk: dict[str, object],
    living_map: LivingMap,
    method: str,
    args: argparse.Namespace,
    pad_id: int,
    device: torch.device,
    answer_vocab: Sequence[str],
    semantic_cluster_by_source: dict[str, dict[str, list[dict[str, str]]]],
) -> dict[str, float]:
    from gfo_real_book_activation_cl import build_training_text  # Local import keeps dependency explicit.

    training_text = build_training_text(chunk, args.include_local_prompts_in_training)
    inputs, targets = encode_lm_tensors(training_text, tokenizer, args.max_seq_len, pad_id)
    qa_supervision = None
    if args.include_local_prompts_in_training:
        qa_supervision = qa_supervision_for_chunk(chunk, tokenizer, args.max_seq_len, pad_id, device)
    composition_supervision = None
    if args.include_composition_prompts_in_training:
        composition_supervision = make_qa_supervision(
            prompt_list(chunk, "composition_prompts"),
            tokenizer,
            args.max_seq_len,
            pad_id,
        )
        if composition_supervision is not None:
            composition_supervision = tuple(tensor.to(device) for tensor in composition_supervision)

    current_cluster_anchors: list[SemanticMarginAnchor] = []
    if args.current_semantic_cluster_weight > 0:
        if not semantic_cluster_by_source["qa"]:
            raise ValueError(
                "--current-semantic-cluster-weight requires --semantic-cluster-prompts-path "
                "with retention_clusters prompts."
            )
        chunk_id = str(chunk["chunk_id"])
        for prompt_index, prompt in enumerate(prompt_list(chunk, "local_prompts")):
            cluster_prompts = cluster_prompts_for_source_answer(
                semantic_cluster_by_source,
                "qa",
                prompt["answer"],
                args.semantic_cluster_max_prompts,
            )
            for cluster_index, cluster_prompt in enumerate(cluster_prompts):
                current_cluster_anchors.append(
                    capture_semantic_margin_anchor(
                        tokenizer,
                        cluster_prompt,
                        anchor_id=f"{chunk_id}:current_cluster:{prompt_index}:{cluster_index}",
                        source_anchor_id=f"{chunk_id}:current:{prompt_index}",
                        chunk_id=chunk_id,
                        source_type="qa_current_cluster",
                        answer_vocab=answer_vocab,
                        max_negatives=args.semantic_margin_negatives,
                        max_seq_len=args.max_seq_len,
                        pad_id=pad_id,
                        importance=args.anchor_importance,
                        cluster_source="cluster",
                    )
                )

    named_trainable_params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_trainable_params:
        raise RuntimeError("No trainable parameters are available during chunk training.")
    trainable_params = [parameter for _, parameter in named_trainable_params]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    global_step = 0
    epoch_losses: list[float] = []
    component_history: dict[str, list[float]] = {
        "train_lm_loss": [],
        "train_qa_loss": [],
        "train_qa_objective": [],
        "train_composition_qa_loss": [],
        "train_composition_qa_objective": [],
        "train_semantic_margin_loss": [],
        "train_semantic_margin_objective": [],
        "train_current_semantic_cluster_loss": [],
        "train_current_semantic_cluster_objective": [],
        "train_anchor_drift_loss": [],
        "train_anchor_drift_objective": [],
        "train_replay_loss": [],
        "train_replay_objective": [],
        "train_pending_trace_loss": [],
        "train_pending_trace_objective": [],
        "train_native_trace_slot_loss": [],
        "train_native_trace_slot_objective": [],
        "train_total_loss": [],
    }
    final_epoch_component_history: dict[str, list[float]] = {}
    for epoch in range(args.epochs_per_chunk):
        batch_losses: list[float] = []
        epoch_component_history = {key: [] for key in component_history}
        iterator = iter_batches(inputs, targets, args.batch_size)
        if not args.no_progress:
            iterator = tqdm(
                iterator,
                total=math.ceil(len(inputs) / args.batch_size),
                desc=f"{method}:{chunk['chunk_id']}:epoch{epoch + 1}",
                leave=False,
            )
        for batch_index, (batch_inputs, batch_targets) in enumerate(iterator):
            optimizer.zero_grad(set_to_none=True)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits, _ = model(batch_inputs)
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_targets.reshape(-1))
            loss = lm_loss
            qa_loss = logits.new_zeros(())
            composition_qa_loss = logits.new_zeros(())
            semantic_loss = logits.new_zeros(())
            current_cluster_loss = logits.new_zeros(())
            anchor_drift_loss = logits.new_zeros(())
            replay_loss = logits.new_zeros(())
            pending_loss = logits.new_zeros(())
            native_slot_loss = logits.new_zeros(())
            pending_native_slot_loss = logits.new_zeros(())
            if qa_supervision is not None:
                qa_inputs, qa_targets, qa_mask = qa_supervision
                qa_logits, _ = model(qa_inputs)
                qa_loss = masked_cross_entropy(qa_logits, qa_targets, qa_mask)
                loss = loss + args.qa_loss_weight * qa_loss
            if composition_supervision is not None:
                comp_inputs, comp_targets, comp_mask = select_supervision_rows(
                    composition_supervision,
                    global_step=global_step,
                    batch_size=args.composition_supervision_batch_size,
                )
                comp_logits, _ = model(comp_inputs)
                composition_qa_loss = masked_cross_entropy(comp_logits, comp_targets, comp_mask)
                loss = loss + args.composition_loss_weight * composition_qa_loss
            active_anchors = living_map.active_anchors()
            semantic_anchors = living_map.active_semantic_anchors()
            if current_cluster_anchors and args.current_semantic_cluster_weight > 0:
                selected_current_cluster = select_semantic_anchor_batch(
                    model,
                    current_cluster_anchors,
                    global_step,
                    args.semantic_anchor_batch_size,
                    device=device,
                    selection=args.semantic_anchor_selection,
                )
                current_cluster_loss = semantic_margin_loss(
                    model,
                    selected_current_cluster,
                    device=device,
                    margin=args.semantic_margin,
                    loss_mode=args.semantic_margin_loss,
                    answer_sequence_weight=args.semantic_answer_sequence_weight,
                )
                loss = loss + args.current_semantic_cluster_weight * current_cluster_loss
            if method == "gfo_living":
                if semantic_anchors and args.semantic_margin_weight > 0:
                    selected_semantic = select_semantic_anchor_batch(
                        model,
                        semantic_anchors,
                        global_step,
                        args.semantic_anchor_batch_size,
                        device=device,
                        selection=args.semantic_anchor_selection,
                    )
                    semantic_loss = semantic_margin_loss(
                        model,
                        selected_semantic,
                        device=device,
                        margin=args.semantic_margin,
                        loss_mode=args.semantic_margin_loss,
                        answer_sequence_weight=args.semantic_answer_sequence_weight,
                    )
                    loss = loss + args.semantic_margin_weight * semantic_loss
                if active_anchors:
                    selected = select_anchor_batch(active_anchors, global_step, args.anchor_batch_size)
                    if args.anchor_drift_weight > 0:
                        anchor_drift_loss = layer_anchor_drift_loss(
                            model,
                            selected,
                            device=device,
                            normalization=args.drift_normalization,
                        )
                        loss = loss + args.anchor_drift_weight * anchor_drift_loss
                    if args.native_trace_slot_loss_weight > 0:
                        slot_selected = [anchor for anchor in selected if anchor.native_slot_targets]
                        if slot_selected:
                            native_slot_loss = native_trace_slot_target_loss(model, slot_selected, device=device)
                            loss = loss + args.native_trace_slot_loss_weight * native_slot_loss
                pending_traces = living_map.active_pending_traces()
                if pending_traces and args.pending_trace_weight > 0:
                    pending_loss, pending_native_slot_loss = pending_trace_loss(
                        model,
                        pending_traces,
                        global_step=global_step,
                        args=args,
                        device=device,
                    )
                    loss = loss + args.pending_trace_weight * pending_loss
            elif method == "replay_living":
                if active_anchors:
                    selected = select_anchor_batch(active_anchors, global_step, args.anchor_batch_size)
                    replay_loss = layer_replay_anchor_loss(model, selected, device=device)
                    loss = loss + args.replay_loss_weight * replay_loss
            elif method != "adamw":
                raise ValueError(f"Unknown method: {method}")
            context = (
                f"method={method}, chunk={chunk['chunk_id']}, epoch={epoch + 1}, "
                f"batch={batch_index + 1}, global_step={global_step}"
            )
            require_finite_tensor("train_loss", loss, context)
            component_tensors = {
                "train_lm_loss": lm_loss,
                "train_qa_loss": qa_loss,
                "train_qa_objective": args.qa_loss_weight * qa_loss,
                "train_composition_qa_loss": composition_qa_loss,
                "train_composition_qa_objective": args.composition_loss_weight * composition_qa_loss,
                "train_semantic_margin_loss": semantic_loss,
                "train_semantic_margin_objective": args.semantic_margin_weight * semantic_loss,
                "train_current_semantic_cluster_loss": current_cluster_loss,
                "train_current_semantic_cluster_objective": args.current_semantic_cluster_weight
                * current_cluster_loss,
                "train_anchor_drift_loss": anchor_drift_loss,
                "train_anchor_drift_objective": args.anchor_drift_weight * anchor_drift_loss,
                "train_replay_loss": replay_loss,
                "train_replay_objective": args.replay_loss_weight * replay_loss,
                "train_pending_trace_loss": pending_loss,
                "train_pending_trace_objective": args.pending_trace_weight * pending_loss,
                "train_native_trace_slot_loss": native_slot_loss + pending_native_slot_loss,
                "train_native_trace_slot_objective": (
                    args.native_trace_slot_loss_weight * native_slot_loss
                    + args.pending_trace_weight * args.native_trace_slot_loss_weight * pending_native_slot_loss
                ),
                "train_total_loss": loss,
            }
            for key, tensor in component_tensors.items():
                require_finite_tensor(key, tensor, context)
                value = float(tensor.detach().cpu())
                component_history[key].append(value)
                epoch_component_history[key].append(value)
            component_values = {key: float(tensor.detach().cpu()) for key, tensor in component_tensors.items()}
            loss.backward()
            require_gradients_finite(named_trainable_params, context, component_values)
            if args.grad_clip > 0:
                clip_grad_norm_strict(named_trainable_params, args.grad_clip, context, component_values)
            optimizer.step()
            require_model_parameters_finite(model, context)
            batch_losses.append(float(loss.detach().cpu()))
            global_step += 1
        if not batch_losses:
            raise RuntimeError(f"No training batches were produced for chunk {chunk['chunk_id']!r}.")
        epoch_losses.append(float(sum(batch_losses) / len(batch_losses)))
        final_epoch_component_history = epoch_component_history

    result = {
        "train_loss": float(epoch_losses[-1]),
        "train_loss_mean": float(sum(epoch_losses) / len(epoch_losses)),
        "train_sequence_count": float(len(inputs)),
        "train_current_semantic_cluster_anchor_count": float(len(current_cluster_anchors)),
    }
    for key, values in component_history.items():
        if not values:
            raise RuntimeError(f"No component values recorded for {key}.")
        result[f"{key}_mean"] = float(sum(values) / len(values))
    for key, values in final_epoch_component_history.items():
        if not values:
            raise RuntimeError(f"No final-epoch component values recorded for {key}.")
        result[f"{key}_final_epoch"] = float(sum(values) / len(values))
    return result


def update_living_map_from_chunk(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    chunk: dict[str, object],
    fact_probes: dict[str, list[str]],
    living_map: LivingMap,
    args: argparse.Namespace,
    pad_id: int,
    device: torch.device,
    step_index: int,
    answer_vocab: Sequence[str],
    semantic_cluster_by_source: dict[str, dict[str, list[dict[str, str]]]],
    *,
    commit_traces: bool,
) -> dict[str, object]:
    if args.evidence_exposures <= 0:
        raise ValueError("--evidence-exposures must be positive.")
    living_map.decay_pending_traces(step_index)
    specs = living_anchor_specs_for_chunk(
        chunk,
        fact_probes,
        include_local_prompts=args.anchor_local_prompts,
        include_composition_prompts=args.anchor_composition_prompts,
        max_fact_probes=args.max_fact_probes_per_chunk,
    )
    if args.max_candidate_anchors_per_chunk > 0:
        specs = specs[: args.max_candidate_anchors_per_chunk]

    actions: list[dict[str, object]] = []
    native_entropy_values: list[float] = []
    native_active_fraction_values: list[float] = []
    native_max_share_values: list[float] = []
    chunk_id = str(chunk["chunk_id"])
    for source_index, spec in enumerate(specs):
        for layer_id in args.anchor_layers:
            anchor_id = f"{chunk_id}:{spec.source_type}:{source_index}:{layer_id}"
            anchor = capture_layer_anchor(
                model,
                tokenizer,
                spec,
                anchor_id=anchor_id,
                chunk_id=chunk_id,
                layer_id=layer_id,
                max_seq_len=args.max_seq_len,
                pad_id=pad_id,
                device=device,
                importance=args.anchor_importance,
                qa_anchor_mode=args.qa_anchor_mode,
            )
            native_trace_diagnostics = native_trace_diagnostics_for_inputs(
                model,
                anchor.inputs,
                device=device,
                context=f"anchor={anchor_id}",
            )
            native_entropy_values.append(float(native_trace_diagnostics["native_trace_slot_entropy_mean"]))
            native_active_fraction_values.append(
                float(native_trace_diagnostics["native_trace_slot_active_fraction_mean"])
            )
            native_max_share_values.append(float(native_trace_diagnostics["native_trace_slot_max_share_mean"]))
            semantic_anchors: list[SemanticMarginAnchor] = []
            prompt = getattr(spec, "prompt")
            if args.semantic_margin_weight > 0 and prompt is not None:
                spec_source_type = str(getattr(spec, "source_type"))
                prompt_candidates = [{"question": prompt["question"], "answer": prompt["answer"]}]
                cluster_prompts = []
                if spec_source_type in {"qa", "composition_qa"}:
                    cluster_prompts = cluster_prompts_for_source_answer(
                        semantic_cluster_by_source,
                        spec_source_type,
                        prompt["answer"],
                        args.semantic_cluster_max_prompts,
                    )
                seen_prompts = {(prompt["question"].strip(), prompt["answer"].strip())}
                for cluster_prompt in cluster_prompts:
                    identity = (cluster_prompt["question"].strip(), cluster_prompt["answer"].strip())
                    if identity in seen_prompts:
                        continue
                    prompt_candidates.append(cluster_prompt)
                    seen_prompts.add(identity)
                    if (
                        args.semantic_cluster_max_prompts > 0
                        and len(prompt_candidates) >= args.semantic_cluster_max_prompts + 1
                    ):
                        break
                for prompt_index, prompt_candidate in enumerate(prompt_candidates):
                    cluster_source = "source" if prompt_index == 0 else "cluster"
                    semantic_anchors.append(
                        capture_semantic_margin_anchor(
                            tokenizer,
                            prompt_candidate,
                            anchor_id=f"{anchor_id}:semantic_margin:{prompt_index}",
                            source_anchor_id=anchor_id,
                            chunk_id=chunk_id,
                            source_type=spec_source_type,
                            answer_vocab=answer_vocab,
                            max_negatives=args.semantic_margin_negatives,
                            max_seq_len=args.max_seq_len,
                            pad_id=pad_id,
                            importance=args.anchor_importance,
                            cluster_source=cluster_source,
                        )
                    )
            vector = pooled_anchor_vector(anchor)
            pending = None
            for _ in range(args.evidence_exposures):
                pending = living_map.update_pending(anchor, vector, step_index)
            if pending is None:
                raise RuntimeError("Pending concept update did not return a concept.")
            living_map.ensure_pending_native_slot_targets(model, pending)
            living_map.attach_anchor_to_pending_trace(pending, anchor, semantic_anchors)
            evidence = living_map.evidence_for_anchor(model, anchor, pending, vector, device=device)
            if commit_traces:
                action = living_map.apply_gate(
                    anchor,
                    vector,
                    pending,
                    evidence,
                    semantic_anchors=semantic_anchors,
                    step_index=step_index,
                )
            else:
                action = "observe"
            actions.append(
                {
                    "anchor_id": anchor.anchor_id,
                    "source_type": anchor.source_type,
                    "layer_id": anchor.layer_id,
                    "action": action,
                    **evidence,
                    **native_trace_diagnostics,
                }
            )
    if actions:
        native_trace_slot_entropy_mean = float(sum(native_entropy_values) / len(native_entropy_values))
        native_trace_slot_active_fraction_mean = float(
            sum(native_active_fraction_values) / len(native_active_fraction_values)
        )
        native_trace_slot_max_share_mean = float(sum(native_max_share_values) / len(native_max_share_values))
    else:
        native_trace_slot_entropy_mean = 0.0
        native_trace_slot_active_fraction_mean = 0.0
        native_trace_slot_max_share_mean = 0.0
    return {
        "candidate_count": len(specs) * len(args.anchor_layers),
        "native_trace_slot_entropy_mean": native_trace_slot_entropy_mean,
        "native_trace_slot_active_fraction_mean": native_trace_slot_active_fraction_mean,
        "native_trace_slot_max_share_mean": native_trace_slot_max_share_mean,
        "living_actions": actions,
    }


def background_maintenance(
    model: DecoderTransformer,
    living_map: LivingMap,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, float]:
    if args.maintenance_steps <= 0:
        return {"maintenance_violation_count": 0.0, "maintenance_loss": 0.0}
    active = living_map.active_concepts()
    if not active:
        return {"maintenance_violation_count": 0.0, "maintenance_loss": 0.0}

    selected_layers = None
    if args.maintenance_layers:
        selected_layers = set(args.maintenance_layers)
    violations: list[LayerActivationAnchor] = []
    model.eval()
    with torch.no_grad():
        for concept in active:
            for anchor in concept_anchor_list(concept):
                if not layer_allowed(anchor.layer_id, selected_layers):
                    continue
                stats = evaluate_layer_anchor_drift(
                    model,
                    [anchor],
                    device=device,
                    normalization=args.drift_normalization,
                )
                if stats["anchor_drift_mean"] > concept.tolerance:
                    violations.append(anchor)
    if not violations:
        return {"maintenance_violation_count": 0.0, "maintenance_loss": 0.0}

    named_trainable_params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_trainable_params:
        raise RuntimeError("No trainable parameters are available during maintenance.")
    trainable_params = [parameter for _, parameter in named_trainable_params]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.maintenance_lr, weight_decay=0.0)
    anchors = violations
    losses: list[float] = []
    model.train()
    for maintenance_step in range(args.maintenance_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = layer_anchor_drift_loss(
            model,
            anchors,
            device=device,
            normalization=args.drift_normalization,
        )
        context = f"maintenance_step={maintenance_step + 1}, violation_count={len(violations)}"
        require_finite_tensor("maintenance_loss", loss, context)
        loss.backward()
        component_values = {"maintenance_loss": float(loss.detach().cpu())}
        require_gradients_finite(named_trainable_params, context, component_values)
        if args.grad_clip > 0:
            clip_grad_norm_strict(named_trainable_params, args.grad_clip, context, component_values)
        optimizer.step()
        require_model_parameters_finite(model, context)
        losses.append(float(loss.detach().cpu()))
    living_map.counters["maintenance_repairs"] += len(violations)
    return {
        "maintenance_violation_count": float(len(violations)),
        "maintenance_loss": float(losses[-1]) if losses else 0.0,
    }


def run_method(
    method: str,
    tokenizer: Tokenizer,
    chunks: Sequence[dict[str, object]],
    fact_probes: dict[str, list[str]],
    heldout_prompt_groups: dict[str, list[dict[str, str]]] | None,
    semantic_cluster_prompt_groups: dict[str, list[dict[str, str]]] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model = instantiate_model(args, tokenizer.get_vocab_size(), device)
    configure_trainable_parameters(model, train_embeddings=args.train_embeddings)
    param_stats = parameter_count_stats(model)
    initial_parameter_snapshot = snapshot_trainable_parameters(model)
    previous_parameter_snapshot = snapshot_trainable_parameters(model)
    pad_id = require_token_id(tokenizer, "[PAD]")
    living_map = LivingMap(args)
    steps: list[dict[str, object]] = []

    selected_chunks = list(chunks[: args.max_chunks]) if args.max_chunks > 0 else list(chunks)
    if not selected_chunks:
        raise ValueError("No chunks selected for training.")
    answer_vocab = collect_answer_vocabulary(selected_chunks, heldout_prompt_groups, semantic_cluster_prompt_groups)
    semantic_cluster_by_source = prompts_by_answer_by_source(semantic_cluster_prompt_groups)

    for chunk_index, chunk in enumerate(selected_chunks):
        chunk_id = str(chunk["chunk_id"])
        print(f"[{method}] chunk {chunk_index + 1}/{len(selected_chunks)}: {chunk_id}")
        pre_map_update: dict[str, object] = {"pre_candidate_count": 0, "pre_living_actions": []}
        if method == "gfo_living" and args.observe_before_train:
            pre_observe = update_living_map_from_chunk(
                model,
                tokenizer,
                chunk,
                fact_probes,
                living_map,
                args,
                pad_id,
                device,
                chunk_index,
                answer_vocab,
                semantic_cluster_by_source,
                commit_traces=False,
            )
            pre_map_update = {
                "pre_candidate_count": pre_observe["candidate_count"],
                "pre_living_actions": pre_observe["living_actions"],
            }
        train_stats = train_chunk(
            model,
            tokenizer,
            chunk,
            living_map,
            method,
            args,
            pad_id,
            device,
            answer_vocab,
            semantic_cluster_by_source,
        )
        chunk_delta_stats = trainable_parameter_delta_stats(
            model,
            previous_parameter_snapshot,
            metric_prefix="chunk",
        )
        cumulative_delta_stats = trainable_parameter_delta_stats(
            model,
            initial_parameter_snapshot,
            metric_prefix="cumulative",
        )
        previous_parameter_snapshot = snapshot_trainable_parameters(model)
        eval_stats = evaluate_chunk_prompts(model, tokenizer, chunk, device)
        if heldout_prompt_groups is not None:
            if "retention_prompts" in heldout_prompt_groups:
                eval_stats.update(
                    evaluate_prompt_group(
                        model,
                        tokenizer,
                        heldout_prompt_groups["retention_prompts"],
                        device,
                        "heldout_retention",
                    )
                )
            if "composition_prompts" in heldout_prompt_groups:
                eval_stats.update(
                    evaluate_prompt_group(
                        model,
                        tokenizer,
                        heldout_prompt_groups["composition_prompts"],
                        device,
                        "heldout_composition",
                    )
                )
        map_update = update_living_map_from_chunk(
            model,
            tokenizer,
            chunk,
            fact_probes,
            living_map,
            args,
            pad_id,
            device,
            chunk_index,
            answer_vocab,
            semantic_cluster_by_source,
            commit_traces=True,
        )
        maintenance = background_maintenance(model, living_map, args, device=device)
        map_stats = living_map.stats(model, device=device)
        semantic_stats = evaluate_semantic_margins(
            model,
            living_map.active_semantic_anchors(),
            device=device,
            margin=args.semantic_margin,
        )
        layer_drift = layer_drift_breakdown(
            model,
            living_map.active_anchors(),
            device=device,
            normalization=args.drift_normalization,
        )
        steps.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                **param_stats,
                **chunk_delta_stats,
                **cumulative_delta_stats,
                **train_stats,
                **eval_stats,
                **pre_map_update,
                **map_update,
                **maintenance,
                **map_stats,
                **semantic_stats,
                "layer_drift": layer_drift,
            }
        )

    return {
        "method": method,
        "steps": steps,
        "summary": summarize_steps(steps),
    }


def summarize_steps(steps: Sequence[dict[str, object]]) -> dict[str, float]:
    if not steps:
        raise ValueError("Cannot summarize an empty step list.")
    numeric_keys = [
        "total_parameter_count",
        "trainable_parameter_count",
        "frozen_parameter_count",
        "trainable_parameter_fraction",
        "chunk_weight_delta_norm",
        "chunk_weight_delta_relative",
        "chunk_weight_delta_max_abs",
        "cumulative_weight_delta_norm",
        "cumulative_weight_delta_relative",
        "cumulative_weight_delta_max_abs",
        "train_loss",
        "train_loss_mean",
        "train_sequence_count",
        "train_current_semantic_cluster_anchor_count",
        "train_lm_loss_mean",
        "train_qa_loss_mean",
        "train_qa_objective_mean",
        "train_composition_qa_loss_mean",
        "train_composition_qa_objective_mean",
        "train_semantic_margin_loss_mean",
        "train_semantic_margin_objective_mean",
        "train_current_semantic_cluster_loss_mean",
        "train_current_semantic_cluster_objective_mean",
        "train_anchor_drift_loss_mean",
        "train_anchor_drift_objective_mean",
        "train_replay_loss_mean",
        "train_replay_objective_mean",
        "train_pending_trace_loss_mean",
        "train_pending_trace_objective_mean",
        "train_native_trace_slot_loss_mean",
        "train_native_trace_slot_objective_mean",
        "train_total_loss_mean",
        "train_lm_loss_final_epoch",
        "train_qa_loss_final_epoch",
        "train_qa_objective_final_epoch",
        "train_composition_qa_loss_final_epoch",
        "train_composition_qa_objective_final_epoch",
        "train_semantic_margin_loss_final_epoch",
        "train_semantic_margin_objective_final_epoch",
        "train_current_semantic_cluster_loss_final_epoch",
        "train_current_semantic_cluster_objective_final_epoch",
        "train_anchor_drift_loss_final_epoch",
        "train_anchor_drift_objective_final_epoch",
        "train_replay_loss_final_epoch",
        "train_replay_objective_final_epoch",
        "train_pending_trace_loss_final_epoch",
        "train_pending_trace_objective_final_epoch",
        "train_native_trace_slot_loss_final_epoch",
        "train_native_trace_slot_objective_final_epoch",
        "train_total_loss_final_epoch",
        "local_accuracy",
        "retention_accuracy",
        "composition_accuracy",
        "local_token_accuracy",
        "retention_token_accuracy",
        "composition_token_accuracy",
        "local_generation_match",
        "retention_generation_match",
        "composition_generation_match",
        "heldout_retention_accuracy",
        "heldout_retention_token_accuracy",
        "heldout_retention_generation_match",
        "heldout_composition_accuracy",
        "heldout_composition_token_accuracy",
        "heldout_composition_generation_match",
        "active_concept_count",
        "active_anchor_count",
        "active_semantic_anchor_count",
        "active_layer_count",
        "pending_concept_count",
        "pending_trace_strength_mean",
        "pending_trace_strength_max",
        "pending_trace_radius_sq_mean",
        "committed_trace_strength_mean",
        "lineage_count",
        "transformed_concept_count",
        "created_count",
        "reinforced_count",
        "attached_count",
        "replacement_fused_count",
        "ignored_count",
        "deferred_count",
        "retired_count",
        "maintenance_repairs",
        "maintenance_violation_count",
        "maintenance_loss",
        "native_trace_slot_entropy_mean",
        "native_trace_slot_active_fraction_mean",
        "native_trace_slot_max_share_mean",
        "destructive_drift_mean",
        "destructive_drift_max",
        "semantic_margin_mean",
        "semantic_margin_min",
        "semantic_margin_violation_rate",
        "semantic_answer_loss_mean",
        "semantic_answer_loss_max",
        "semantic_answer_token_accuracy_mean",
        "semantic_answer_exact_match_rate",
        "semantic_margin_source_count",
        "semantic_margin_cluster_count",
        "semantic_margin_source_violation_rate",
        "semantic_margin_cluster_violation_rate",
    ]
    summary: dict[str, float] = {}
    for key in numeric_keys:
        values: list[float] = []
        for step in steps:
            if key not in step:
                continue
            value = float(step[key])
            require_finite_float(key, value, f"summarize_steps chunk={step.get('chunk_id', '<unknown>')}")
            values.append(value)
        if not values:
            continue
        summary[f"{key}_mean"] = float(sum(values) / len(values))
        summary[f"{key}_final"] = float(steps[-1][key])
    return summary


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_model_path": str(args.base_model_path),
        "tokenizer_path": str(args.tokenizer_path),
        "chunks_path": str(args.chunks_path),
        "fact_probes_path": str(args.fact_probes_path),
        "heldout_prompts_path": None if args.heldout_prompts_path is None else str(args.heldout_prompts_path),
        "semantic_cluster_prompts_path": (
            None if args.semantic_cluster_prompts_path is None else str(args.semantic_cluster_prompts_path)
        ),
        "methods": args.methods,
        "seed": args.seed,
        "max_chunks": args.max_chunks,
        "epochs_per_chunk": args.epochs_per_chunk,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "include_local_prompts_in_training": args.include_local_prompts_in_training,
        "include_composition_prompts_in_training": args.include_composition_prompts_in_training,
        "qa_loss_weight": args.qa_loss_weight,
        "composition_loss_weight": args.composition_loss_weight,
        "composition_supervision_batch_size": args.composition_supervision_batch_size,
        "anchor_drift_weight": args.anchor_drift_weight,
        "semantic_margin_weight": args.semantic_margin_weight,
        "current_semantic_cluster_weight": args.current_semantic_cluster_weight,
        "semantic_margin": args.semantic_margin,
        "semantic_margin_loss": args.semantic_margin_loss,
        "semantic_answer_sequence_weight": args.semantic_answer_sequence_weight,
        "semantic_anchor_selection": args.semantic_anchor_selection,
        "pending_trace_weight": args.pending_trace_weight,
        "native_trace_slot_loss_weight": args.native_trace_slot_loss_weight,
        "semantic_margin_negatives": args.semantic_margin_negatives,
        "semantic_anchor_batch_size": args.semantic_anchor_batch_size,
        "semantic_cluster_max_prompts": args.semantic_cluster_max_prompts,
        "replay_loss_weight": args.replay_loss_weight,
        "anchor_batch_size": args.anchor_batch_size,
        "anchor_importance": args.anchor_importance,
        "anchor_tolerance": args.anchor_tolerance,
        "anchor_layers": args.anchor_layers,
        "anchor_local_prompts": args.anchor_local_prompts,
        "anchor_composition_prompts": args.anchor_composition_prompts,
        "qa_anchor_mode": args.qa_anchor_mode,
        "observe_before_train": args.observe_before_train,
        "native_trace_slots": args.native_trace_slots,
        "native_trace_rank": args.native_trace_rank,
        "native_trace_top_k": args.native_trace_top_k,
        "native_trace_init_scale": args.native_trace_init_scale,
        "train_native_traces_only": args.train_native_traces_only,
        "drift_normalization": args.drift_normalization,
        "allow_cross_layer_match": args.allow_cross_layer_match,
        "maintenance_layers": args.maintenance_layers,
        "max_candidate_anchors_per_chunk": args.max_candidate_anchors_per_chunk,
        "max_active_concepts": args.max_active_concepts,
        "pressure_threshold": args.pressure_threshold,
        "novelty_threshold": args.novelty_threshold,
        "pending_merge_similarity": args.pending_merge_similarity,
        "trace_learning_rate": args.trace_learning_rate,
        "trace_centroid_rate": args.trace_centroid_rate,
        "trace_decay": args.trace_decay,
        "trace_prune_threshold": args.trace_prune_threshold,
        "merge_similarity": args.merge_similarity,
        "fusion_similarity": args.fusion_similarity,
        "evidence_exposures": args.evidence_exposures,
        "same_source_only": args.same_source_only,
        "incompatible_merge_action": args.incompatible_merge_action,
        "maintenance_steps": args.maintenance_steps,
        "device": args.device,
    }


def print_summary(report: dict[str, object]) -> None:
    print("\nREAL-BOOK GFO LIVING-MAP SUMMARY")
    print("=" * 120)
    for method, method_report in report["methods"].items():  # type: ignore[union-attr]
        summary = method_report["summary"]
        print(f"method={method}")
        print("-" * 120)
        for key in (
            "trainable_parameter_count_final",
            "trainable_parameter_fraction_final",
            "cumulative_weight_delta_norm_final",
            "cumulative_weight_delta_relative_final",
            "train_total_loss_final_epoch_final",
            "train_lm_loss_final_epoch_final",
            "train_qa_objective_final_epoch_final",
            "train_composition_qa_objective_final_epoch_final",
            "train_current_semantic_cluster_objective_final_epoch_final",
            "train_semantic_margin_objective_final_epoch_final",
            "train_anchor_drift_objective_final_epoch_final",
            "train_replay_objective_final_epoch_final",
            "train_pending_trace_objective_final_epoch_final",
            "train_native_trace_slot_objective_final_epoch_final",
            "local_token_accuracy_mean",
            "retention_token_accuracy_mean",
            "composition_token_accuracy_mean",
            "heldout_retention_token_accuracy_final",
            "heldout_retention_generation_match_final",
            "heldout_composition_token_accuracy_final",
            "heldout_composition_generation_match_final",
            "active_concept_count_final",
            "active_anchor_count_final",
            "active_semantic_anchor_count_final",
            "active_layer_count_final",
            "pending_concept_count_final",
            "pending_trace_strength_mean_final",
            "pending_trace_strength_max_final",
            "pending_trace_radius_sq_mean_final",
            "committed_trace_strength_mean_final",
            "lineage_count_final",
            "created_count_final",
            "reinforced_count_final",
            "attached_count_final",
            "replacement_fused_count_final",
            "ignored_count_final",
            "deferred_count_final",
            "native_trace_slot_entropy_mean_final",
            "native_trace_slot_active_fraction_mean_final",
            "native_trace_slot_max_share_mean_final",
            "destructive_drift_mean_final",
            "destructive_drift_max_final",
            "semantic_margin_mean_final",
            "semantic_margin_min_final",
            "semantic_margin_violation_rate_final",
            "semantic_answer_loss_mean_final",
            "semantic_answer_token_accuracy_mean_final",
            "semantic_answer_exact_match_rate_final",
            "semantic_margin_source_violation_rate_final",
            "semantic_margin_cluster_violation_rate_final",
            "maintenance_repairs_final",
        ):
            value = summary.get(key, float("nan"))
            print(f"{key:46s} {value:.4f}")
        print("-" * 120)
    print("=" * 120)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--fact-probes-path", type=Path, default=Path("data/real_book/fact_probes.json"))
    parser.add_argument("--heldout-prompts-path", type=Path, default=None)
    parser.add_argument("--semantic-cluster-prompts-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gfo-real-book-living-map.json"))
    parser.add_argument("--methods", type=parse_methods, default=["adamw", "gfo_living", "replay_living"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--epochs-per-chunk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--qa-loss-weight", type=float, default=5.0)
    parser.add_argument("--include-local-prompts-in-training", action="store_true")
    parser.add_argument("--include-composition-prompts-in-training", action="store_true")
    parser.add_argument("--composition-loss-weight", type=float, default=5.0)
    parser.add_argument(
        "--composition-supervision-batch-size",
        type=int,
        default=0,
        help="Number of composition QA prompts to train per LM batch. Use 0 to train all composition prompts.",
    )
    parser.add_argument("--anchor-drift-weight", type=float, default=5.0)
    parser.add_argument("--semantic-margin-weight", type=float, default=0.0)
    parser.add_argument("--current-semantic-cluster-weight", type=float, default=0.0)
    parser.add_argument("--semantic-margin", type=float, default=1.0)
    parser.add_argument("--semantic-margin-loss", choices=["hinge", "squared_hinge"], default="hinge")
    parser.add_argument("--semantic-answer-sequence-weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic-anchor-selection",
        choices=["round_robin", "worst_margin", "worst_answer_loss"],
        default="round_robin",
    )
    parser.add_argument("--pending-trace-weight", type=float, default=1.0)
    parser.add_argument("--semantic-margin-negatives", type=int, default=16)
    parser.add_argument("--semantic-anchor-batch-size", type=int, default=4)
    parser.add_argument("--semantic-cluster-max-prompts", type=int, default=0)
    parser.add_argument("--replay-loss-weight", type=float, default=5.0)
    parser.add_argument("--anchor-batch-size", type=int, default=4)
    parser.add_argument("--anchor-importance", type=float, default=1.0)
    parser.add_argument("--anchor-tolerance", type=float, default=0.2)
    parser.add_argument("--anchor-layers", type=parse_layer_ids, default=["final"])
    parser.add_argument("--drift-normalization", choices=["none", "target_energy"], default="none")
    parser.add_argument("--allow-cross-layer-match", action="store_true")
    parser.add_argument("--maintenance-layers", type=parse_layer_ids, default=[])
    parser.add_argument("--max-candidate-anchors-per-chunk", type=int, default=8)
    parser.add_argument("--max-fact-probes-per-chunk", type=int, default=8)
    parser.add_argument("--max-active-concepts", type=int, default=0)
    parser.add_argument("--anchor-local-prompts", action="store_true")
    parser.add_argument("--anchor-composition-prompts", action="store_true")
    parser.add_argument("--qa-anchor-mode", choices=["full_sequence", "answer_tokens"], default="answer_tokens")
    parser.add_argument("--observe-before-train", action="store_true")
    parser.add_argument("--native-trace-slots", type=int, default=0)
    parser.add_argument("--native-trace-rank", type=int, default=8)
    parser.add_argument("--native-trace-top-k", type=int, default=2)
    parser.add_argument("--native-trace-init-scale", type=float, default=1e-3)
    parser.add_argument("--native-trace-slot-loss-weight", type=float, default=0.0)
    parser.add_argument("--train-native-traces-only", action="store_true")
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)

    parser.add_argument("--evidence-exposures", type=int, default=1)
    parser.add_argument("--pressure-threshold", type=float, default=0.2)
    parser.add_argument("--novelty-threshold", type=float, default=0.05)
    parser.add_argument("--pending-merge-similarity", type=float, default=0.8)
    parser.add_argument("--trace-learning-rate", type=float, default=0.15)
    parser.add_argument("--trace-centroid-rate", type=float, default=0.25)
    parser.add_argument("--trace-decay", type=float, default=0.95)
    parser.add_argument("--trace-prune-threshold", type=float, default=0.01)
    parser.add_argument("--merge-similarity", type=float, default=0.98)
    parser.add_argument("--fusion-similarity", type=float, default=0.75)
    parser.add_argument("--same-source-only", action="store_true")
    parser.add_argument("--merge-rate", type=float, default=0.2)
    parser.add_argument(
        "--incompatible-merge-action",
        choices=["defer", "create", "attach", "replacement_fuse"],
        default="attach",
    )
    parser.add_argument("--breadth-threshold", type=float, default=0.1)
    parser.add_argument("--depth-center", type=float, default=0.2)
    parser.add_argument("--depth-scale", type=float, default=0.2)
    parser.add_argument("--frequency-tau", type=float, default=3.0)
    parser.add_argument("--consistency-tau", type=float, default=0.1)
    parser.add_argument("--error-center", type=float, default=0.5)
    parser.add_argument("--error-scale", type=float, default=0.5)
    parser.add_argument("--breadth-exponent", type=float, default=0.25)
    parser.add_argument("--depth-exponent", type=float, default=0.25)
    parser.add_argument("--frequency-exponent", type=float, default=0.25)
    parser.add_argument("--consistency-exponent", type=float, default=0.25)
    parser.add_argument("--novelty-exponent", type=float, default=0.25)
    parser.add_argument("--familiarity-exponent", type=float, default=0.25)
    parser.add_argument("--error-exponent", type=float, default=0.0)
    parser.add_argument("--recency-tau", type=float, default=10.0)
    parser.add_argument("--maintenance-steps", type=int, default=0)
    parser.add_argument("--maintenance-lr", type=float, default=1e-4)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_chunks < 0:
        raise ValueError("--max-chunks must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.epochs_per_chunk <= 0:
        raise ValueError("--epochs-per-chunk must be positive.")
    validate_layer_ids(args.anchor_layers, args.n_layers, "--anchor-layers")
    validate_layer_ids(args.maintenance_layers, args.n_layers, "--maintenance-layers")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.qa_loss_weight < 0:
        raise ValueError("--qa-loss-weight must be non-negative.")
    if args.composition_loss_weight < 0:
        raise ValueError("--composition-loss-weight must be non-negative.")
    if args.include_composition_prompts_in_training and args.composition_loss_weight <= 0:
        raise ValueError("--include-composition-prompts-in-training requires --composition-loss-weight > 0.")
    if args.composition_supervision_batch_size < 0:
        raise ValueError("--composition-supervision-batch-size must be non-negative.")
    if args.anchor_batch_size <= 0:
        raise ValueError("--anchor-batch-size must be positive.")
    if args.anchor_drift_weight < 0:
        raise ValueError("--anchor-drift-weight must be non-negative.")
    if args.semantic_margin_weight < 0:
        raise ValueError("--semantic-margin-weight must be non-negative.")
    if args.current_semantic_cluster_weight < 0:
        raise ValueError("--current-semantic-cluster-weight must be non-negative.")
    if args.semantic_margin < 0:
        raise ValueError("--semantic-margin must be non-negative.")
    if args.semantic_answer_sequence_weight < 0:
        raise ValueError("--semantic-answer-sequence-weight must be non-negative.")
    if args.pending_trace_weight < 0:
        raise ValueError("--pending-trace-weight must be non-negative.")
    if (
        args.semantic_answer_sequence_weight > 0
        and args.semantic_margin_weight <= 0
        and args.current_semantic_cluster_weight <= 0
    ):
        raise ValueError(
            "--semantic-answer-sequence-weight has no effect unless --semantic-margin-weight > 0 "
            "or --current-semantic-cluster-weight > 0."
        )
    if args.semantic_margin_negatives <= 0:
        raise ValueError("--semantic-margin-negatives must be positive.")
    if args.semantic_anchor_batch_size <= 0:
        raise ValueError("--semantic-anchor-batch-size must be positive.")
    if args.semantic_cluster_max_prompts < 0:
        raise ValueError("--semantic-cluster-max-prompts must be non-negative.")
    if args.semantic_cluster_prompts_path is not None and args.semantic_margin_weight <= 0:
        if args.current_semantic_cluster_weight <= 0:
            raise ValueError(
                "--semantic-cluster-prompts-path requires --semantic-margin-weight > 0 "
                "or --current-semantic-cluster-weight > 0."
            )
    if args.current_semantic_cluster_weight > 0 and args.semantic_cluster_prompts_path is None:
        raise ValueError("--current-semantic-cluster-weight requires --semantic-cluster-prompts-path.")
    if (
        args.semantic_cluster_prompts_path is not None
        and args.heldout_prompts_path is not None
        and args.semantic_cluster_prompts_path.resolve() == args.heldout_prompts_path.resolve()
    ):
        raise ValueError("--semantic-cluster-prompts-path must not be the same file as --heldout-prompts-path.")
    if args.semantic_margin_weight > 0 and not (args.anchor_local_prompts or args.anchor_composition_prompts):
        raise ValueError(
            "--semantic-margin-weight requires --anchor-local-prompts or --anchor-composition-prompts "
            "so semantic anchors can be created."
        )
    if args.replay_loss_weight < 0:
        raise ValueError("--replay-loss-weight must be non-negative.")
    if args.anchor_tolerance < 0:
        raise ValueError("--anchor-tolerance must be non-negative.")
    if args.max_candidate_anchors_per_chunk < 0:
        raise ValueError("--max-candidate-anchors-per-chunk must be non-negative.")
    if args.max_fact_probes_per_chunk < 0:
        raise ValueError("--max-fact-probes-per-chunk must be non-negative.")
    if args.max_active_concepts < 0:
        raise ValueError("--max-active-concepts must be non-negative.")
    if args.native_trace_slots < 0:
        raise ValueError("--native-trace-slots must be non-negative.")
    if args.native_trace_rank <= 0:
        raise ValueError("--native-trace-rank must be positive.")
    if args.native_trace_top_k <= 0:
        raise ValueError("--native-trace-top-k must be positive.")
    if args.native_trace_slots > 0 and args.native_trace_top_k > args.native_trace_slots:
        raise ValueError("--native-trace-top-k must be <= --native-trace-slots.")
    if args.native_trace_init_scale <= 0:
        raise ValueError("--native-trace-init-scale must be positive.")
    if args.native_trace_slot_loss_weight < 0:
        raise ValueError("--native-trace-slot-loss-weight must be non-negative.")
    if args.native_trace_slot_loss_weight > 0 and args.native_trace_slots <= 0:
        raise ValueError("--native-trace-slot-loss-weight requires --native-trace-slots > 0.")
    if args.train_native_traces_only and args.native_trace_slots <= 0:
        raise ValueError("--train-native-traces-only requires --native-trace-slots > 0.")
    if args.evidence_exposures <= 0:
        raise ValueError("--evidence-exposures must be positive.")
    for name in (
        "pressure_threshold",
        "novelty_threshold",
        "pending_merge_similarity",
        "merge_similarity",
        "fusion_similarity",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if not 0.0 < args.trace_learning_rate <= 1.0:
        raise ValueError("--trace-learning-rate must be in (0, 1].")
    if not 0.0 < args.trace_centroid_rate <= 1.0:
        raise ValueError("--trace-centroid-rate must be in (0, 1].")
    if not 0.0 <= args.trace_decay <= 1.0:
        raise ValueError("--trace-decay must be in [0, 1].")
    if args.trace_prune_threshold < 0.0:
        raise ValueError("--trace-prune-threshold must be non-negative.")
    if args.fusion_similarity > args.merge_similarity:
        raise ValueError("--fusion-similarity must be <= --merge-similarity.")
    if args.maintenance_steps < 0:
        raise ValueError("--maintenance-steps must be non-negative.")
    if args.maintenance_lr <= 0:
        raise ValueError("--maintenance-lr must be positive.")
    if args.recency_tau <= 0:
        raise ValueError("--recency-tau must be positive.")
    for name in (
        "breadth_exponent",
        "depth_exponent",
        "frequency_exponent",
        "consistency_exponent",
        "novelty_exponent",
        "familiarity_exponent",
        "error_exponent",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    fact_probes = load_fact_probes(args.fact_probes_path)
    heldout_prompt_groups = None
    if args.heldout_prompts_path is not None:
        heldout_prompt_groups = load_prompt_groups(args.heldout_prompts_path)
    semantic_cluster_prompt_groups = None
    if args.semantic_cluster_prompts_path is not None:
        semantic_cluster_prompt_groups = load_prompt_groups(args.semantic_cluster_prompts_path)

    report: dict[str, object] = {
        "config": config_from_args(args),
        "methods": {},
    }
    for method in args.methods:
        set_seed(args.seed)
        report["methods"][method] = run_method(  # type: ignore[index]
            method,
            tokenizer,
            chunks,
            fact_probes,
            heldout_prompt_groups,
            semantic_cluster_prompt_groups,
            args,
            device,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print_summary(report)
    print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
