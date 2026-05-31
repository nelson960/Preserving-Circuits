#!/usr/bin/env python3
"""Real-book activation-anchor continual learning benchmark.

This is the NLP version of the GFO idea. It uses the existing real-book
benchmark data:

    public-domain book chunks
    BPE tokenizer
    tiny decoder transformer
    QA probes and fact probes

The experiment compares ordinary sequential AdamW with a soft GFO variant that
protects activation anchors captured from previous chunks. This is not a
symbolic fact extractor and does not use gold subject/relation/object events.
Raw text and QA strings are the input stream.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
    make_lm_sequences,
    make_qa_supervision,
    masked_cross_entropy,
    require_token_id,
    resolve_device,
)
from real_book_regular_adamw import evaluate_qa_loss_and_acc  # noqa: E402


METHODS = ("adamw", "gfo_soft", "replay")


@dataclass
class ActivationAnchor:
    anchor_id: str
    chunk_id: str
    source_type: str
    text: str
    inputs: torch.Tensor
    targets: torch.Tensor
    hidden_target: torch.Tensor
    mask: torch.Tensor | None
    importance: float


@dataclass(frozen=True)
class AnchorSpec:
    source_type: str
    text: str
    prompt: dict[str, str] | None = None


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Available methods: {METHODS}.")
    return methods


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(path: Path) -> list[dict[str, object]]:
    data = load_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Chunks JSON must be a non-empty list: {path}")
    for index, chunk in enumerate(data):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {index} must be an object.")
        for key in ("chunk_id", "text", "local_prompts", "retention_prompts", "composition_prompts"):
            if key not in chunk:
                raise ValueError(f"Chunk {index} missing required key {key!r}.")
    return data


def load_fact_probes(path: Path) -> dict[str, list[str]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Fact probes JSON must be an object: {path}")
    result: dict[str, list[str]] = {}
    for chunk_id, probes in data.items():
        if not isinstance(probes, list):
            raise ValueError(f"Fact probes for {chunk_id!r} must be a list.")
        result[str(chunk_id)] = [str(probe) for probe in probes]
    return result


def load_prompt_groups(path: Path) -> dict[str, list[dict[str, str]]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt groups JSON must be an object: {path}")
    groups: dict[str, list[dict[str, str]]] = {}
    for key, raw_prompts in data.items():
        if not isinstance(raw_prompts, list):
            raise ValueError(f"Prompt group {key!r} must be a list.")
        prompts: list[dict[str, str]] = []
        for index, item in enumerate(raw_prompts):
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                raise ValueError(f"Prompt group {key!r} item {index} must contain question and answer.")
            prompts.append({"question": str(item["question"]), "answer": str(item["answer"])})
        groups[str(key)] = prompts
    return groups


def instantiate_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> DecoderTransformer:
    native_trace_slots = int(getattr(args, "native_trace_slots", 0))
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        trace_slots=native_trace_slots,
        trace_rank=int(getattr(args, "native_trace_rank", 8)),
        trace_top_k=int(getattr(args, "native_trace_top_k", 2)),
        trace_init_scale=float(getattr(args, "native_trace_init_scale", 1e-3)),
        trace_state_update_rate=float(getattr(args, "native_trace_state_update_rate", 0.05)),
        trace_state_decay=float(getattr(args, "native_trace_state_decay", 0.99)),
        trace_initial_strength_logit=float(getattr(args, "native_trace_initial_strength_logit", -4.0)),
        trace_entropy_loss_weight=float(getattr(args, "native_slot_entropy_weight", 0.0)),
        trace_balance_loss_weight=float(getattr(args, "native_slot_balance_weight", 0.0)),
        trace_strength_loss_weight=float(getattr(args, "native_slot_strength_weight", 0.0)),
        trace_pressure_update_loss_weight=float(getattr(args, "native_pressure_update_weight", 0.0)),
        trace_state_delta_loss_weight=float(getattr(args, "native_state_delta_weight", 0.0)),
        trace_pressure_sparsity_loss_weight=float(getattr(args, "native_pressure_sparsity_weight", 0.0)),
        trace_capacity_pressure_loss_weight=float(getattr(args, "native_capacity_pressure_weight", 0.0)),
        trace_compression_pressure_loss_weight=float(getattr(args, "native_compression_pressure_weight", 0.0)),
        trace_forget_pressure_loss_weight=float(getattr(args, "native_forget_pressure_weight", 0.0)),
        native_trace_learning_only=bool(getattr(args, "train_native_traces_only", False)),
    ).to(device)
    model.train_native_traces_only = bool(getattr(args, "train_native_traces_only", False))
    if not args.base_model_path.exists():
        raise FileNotFoundError(f"Base model checkpoint does not exist: {args.base_model_path}")
    state = torch.load(args.base_model_path, map_location=device)
    if native_trace_slots > 0:
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected = list(unexpected)
        missing = list(missing)
        unexpected_trace_keys = [key for key in unexpected if "trace_adapter" not in key]
        missing_base_keys = [key for key in missing if "trace_adapter" not in key]
        if unexpected_trace_keys or missing_base_keys:
            raise RuntimeError(
                "Base checkpoint is incompatible with native trace model: "
                f"missing_base_keys={missing_base_keys}, unexpected_non_trace_keys={unexpected_trace_keys}"
            )
        if not missing:
            raise RuntimeError("Native trace slots were requested, but no trace parameters were missing from checkpoint.")
    else:
        model.load_state_dict(state)
    return model


def configure_trainable_parameters(model: DecoderTransformer, *, train_embeddings: bool) -> list[torch.nn.Parameter]:
    if hasattr(model, "configure_trainability"):
        return model.configure_trainability(train_embeddings=train_embeddings)
    train_native_traces_only = bool(getattr(model, "train_native_traces_only", False))
    for name, parameter in model.named_parameters():
        if train_native_traces_only:
            parameter.requires_grad_("trace_adapter" in name)
        elif not train_embeddings and (
            "token_embedding" in name or "position_embedding" in name or "lm_head" in name
        ):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters remain after applying freeze configuration.")
    return params


def build_training_text(chunk: dict[str, object], include_local_prompts: bool) -> str:
    text = str(chunk["text"])
    parts = [text]
    if include_local_prompts:
        for prompt in prompt_list(chunk, "local_prompts"):
            parts.append(f"{format_qa_prompt(prompt['question'])}{prompt['answer']}")
    return "\n\n".join(parts)


def prompt_list(chunk: dict[str, object], key: str) -> list[dict[str, str]]:
    raw = chunk[key]
    if not isinstance(raw, list):
        raise ValueError(f"Chunk {chunk.get('chunk_id', '<unknown>')!r} field {key!r} must be a list.")
    prompts: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise ValueError(
                f"Chunk {chunk.get('chunk_id', '<unknown>')!r} prompt {key}[{index}] "
                "must contain question and answer."
            )
        prompts.append({"question": str(item["question"]), "answer": str(item["answer"])})
    return prompts


def encode_lm_tensors(
    text: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer.encode(text).ids
    if len(token_ids) <= 1:
        raise ValueError(f"Text encoded to {len(token_ids)} token(s): {text[:80]!r}")
    input_seqs, target_seqs = make_lm_sequences(token_ids, max_seq_len, pad_id)
    return torch.tensor(input_seqs, dtype=torch.long), torch.tensor(target_seqs, dtype=torch.long)


def iter_batches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    if len(inputs) != len(targets):
        raise ValueError(f"Input/target size mismatch: {len(inputs)} != {len(targets)}.")
    permutation = torch.randperm(len(inputs))
    for start in range(0, len(inputs), batch_size):
        indices = permutation[start : start + batch_size]
        yield inputs[indices], targets[indices]


def qa_supervision_for_chunk(
    chunk: dict[str, object],
    tokenizer: Tokenizer,
    max_seq_len: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    supervision = make_qa_supervision(prompt_list(chunk, "local_prompts"), tokenizer, max_seq_len, pad_id)
    if supervision is None:
        return None
    inputs, targets, mask = supervision
    return inputs.to(device), targets.to(device), mask.to(device)


def anchor_specs_for_chunk(
    chunk: dict[str, object],
    fact_probes: dict[str, list[str]],
    *,
    include_local_prompts: bool,
    max_fact_probes: int,
) -> list[AnchorSpec]:
    chunk_id = str(chunk["chunk_id"])
    specs: list[AnchorSpec] = []
    if include_local_prompts:
        for prompt in prompt_list(chunk, "local_prompts"):
            text = f"{format_qa_prompt(prompt['question'])}{prompt['answer']}"
            specs.append(AnchorSpec("qa", text, prompt))
    probes = fact_probes.get(chunk_id, [])
    if max_fact_probes < 0:
        raise ValueError("--max-fact-probes-per-chunk must be non-negative.")
    for probe in probes[:max_fact_probes]:
        specs.append(AnchorSpec("fact_probe", probe))
    return specs


def capture_anchor(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    spec: AnchorSpec,
    *,
    anchor_id: str,
    chunk_id: str,
    max_seq_len: int,
    pad_id: int,
    device: torch.device,
    importance: float,
    qa_anchor_mode: str,
) -> ActivationAnchor:
    mask = None
    if spec.source_type == "qa" and qa_anchor_mode == "answer_tokens":
        if spec.prompt is None:
            raise RuntimeError("QA answer-token anchor requires the original prompt dictionary.")
        supervision = make_qa_supervision([spec.prompt], tokenizer, max_seq_len, pad_id)
        if supervision is None:
            raise RuntimeError("QA anchor supervision unexpectedly returned None for one prompt.")
        inputs, targets, answer_mask = supervision
        mask = answer_mask
    else:
        inputs, targets = encode_lm_tensors(spec.text, tokenizer, max_seq_len, pad_id)
    inputs = inputs.to(device)
    model.eval()
    with torch.no_grad():
        _, hidden = model(inputs)
    return ActivationAnchor(
        anchor_id=anchor_id,
        chunk_id=chunk_id,
        source_type=spec.source_type,
        text=spec.text,
        inputs=inputs.detach().cpu(),
        targets=targets.detach().cpu(),
        hidden_target=hidden.detach().cpu(),
        mask=None if mask is None else mask.detach().cpu(),
        importance=importance,
    )


def add_chunk_anchors(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    chunk: dict[str, object],
    fact_probes: dict[str, list[str]],
    anchors: list[ActivationAnchor],
    args: argparse.Namespace,
    pad_id: int,
    device: torch.device,
) -> None:
    chunk_id = str(chunk["chunk_id"])
    specs = anchor_specs_for_chunk(
        chunk,
        fact_probes,
        include_local_prompts=args.anchor_local_prompts,
        max_fact_probes=args.max_fact_probes_per_chunk,
    )
    if args.max_anchors_per_chunk > 0:
        specs = specs[: args.max_anchors_per_chunk]
    for source_index, spec in enumerate(specs):
        anchor_id = f"{chunk_id}:{spec.source_type}:{source_index}"
        anchors.append(
            capture_anchor(
                model,
                tokenizer,
                spec,
                anchor_id=anchor_id,
                chunk_id=chunk_id,
                max_seq_len=args.max_seq_len,
                pad_id=pad_id,
                device=device,
                importance=args.anchor_importance,
                qa_anchor_mode=args.qa_anchor_mode,
            )
        )


def select_anchor_batch(
    anchors: Sequence[ActivationAnchor],
    step_index: int,
    batch_size: int,
) -> list[ActivationAnchor]:
    if batch_size <= 0:
        raise ValueError("--anchor-batch-size must be positive.")
    if not anchors:
        return []
    selected = []
    for offset in range(min(batch_size, len(anchors))):
        selected.append(anchors[(step_index + offset) % len(anchors)])
    return selected


def anchor_drift_loss(
    model: DecoderTransformer,
    anchors: Sequence[ActivationAnchor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not anchors:
        raise ValueError("anchor_drift_loss requires at least one anchor.")
    losses = []
    for anchor in anchors:
        inputs = anchor.inputs.to(device)
        target = anchor.hidden_target.to(device)
        _, hidden = model(inputs)
        token_losses = ((hidden - target) ** 2).mean(dim=-1)
        if anchor.mask is None:
            drift = token_losses.mean()
        else:
            mask = anchor.mask.to(device)
            denom = mask.sum()
            if denom.item() <= 0.0:
                raise ValueError(f"Anchor {anchor.anchor_id!r} has an empty drift mask.")
            drift = (token_losses * mask).sum() / denom
        losses.append(anchor.importance * drift)
    return torch.stack(losses).mean()


def replay_anchor_loss(
    model: DecoderTransformer,
    anchors: Sequence[ActivationAnchor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not anchors:
        raise ValueError("replay_anchor_loss requires at least one anchor.")
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


def evaluate_anchor_drift(
    model: DecoderTransformer,
    anchors: Sequence[ActivationAnchor],
    *,
    device: torch.device,
) -> dict[str, float]:
    if not anchors:
        return {"anchor_drift_mean": 0.0, "anchor_drift_max": 0.0}
    values = []
    model.eval()
    with torch.no_grad():
        for anchor in anchors:
            inputs = anchor.inputs.to(device)
            target = anchor.hidden_target.to(device)
            _, hidden = model(inputs)
            token_losses = ((hidden - target) ** 2).mean(dim=-1)
            if anchor.mask is None:
                drift = token_losses.mean()
            else:
                mask = anchor.mask.to(device)
                denom = mask.sum()
                if denom.item() <= 0.0:
                    raise ValueError(f"Anchor {anchor.anchor_id!r} has an empty drift mask.")
                drift = (token_losses * mask).sum() / denom
            values.append(float(drift.detach().cpu()))
    return {
        "anchor_drift_mean": float(sum(values) / len(values)),
        "anchor_drift_max": float(max(values)),
    }


def mean_metric(rows: Sequence[dict[str, float | str]], key: str, empty_value: float = 1.0) -> float:
    if not rows:
        return empty_value
    return float(sum(float(row[key]) for row in rows) / len(rows))


def evaluate_chunk_prompts(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    chunk: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    local = [evaluate_qa_loss_and_acc(model, tokenizer, prompt, device) for prompt in prompt_list(chunk, "local_prompts")]
    retention = [
        evaluate_qa_loss_and_acc(model, tokenizer, prompt, device)
        for prompt in prompt_list(chunk, "retention_prompts")
    ]
    composition = [
        evaluate_qa_loss_and_acc(model, tokenizer, prompt, device)
        for prompt in prompt_list(chunk, "composition_prompts")
    ]
    return {
        "local_accuracy": mean_metric(local, "accuracy"),
        "retention_accuracy": mean_metric(retention, "accuracy"),
        "composition_accuracy": mean_metric(composition, "accuracy"),
        "local_token_accuracy": mean_metric(local, "token_accuracy"),
        "retention_token_accuracy": mean_metric(retention, "token_accuracy"),
        "composition_token_accuracy": mean_metric(composition, "token_accuracy"),
        "local_generation_match": mean_metric(local, "generation_match"),
        "retention_generation_match": mean_metric(retention, "generation_match"),
        "composition_generation_match": mean_metric(composition, "generation_match"),
        "local_evals": local,
        "retention_evals": retention,
        "composition_evals": composition,
    }


def evaluate_prompt_group(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    prompts: Sequence[dict[str, str]],
    device: torch.device,
    prefix: str,
) -> dict[str, object]:
    evals = [evaluate_qa_loss_and_acc(model, tokenizer, prompt, device) for prompt in prompts]
    return {
        f"{prefix}_accuracy": mean_metric(evals, "accuracy"),
        f"{prefix}_token_accuracy": mean_metric(evals, "token_accuracy"),
        f"{prefix}_generation_match": mean_metric(evals, "generation_match"),
        f"{prefix}_evals": evals,
    }


def train_chunk(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    chunk: dict[str, object],
    anchors: Sequence[ActivationAnchor],
    method: str,
    args: argparse.Namespace,
    pad_id: int,
    device: torch.device,
) -> dict[str, float]:
    training_text = build_training_text(chunk, args.include_local_prompts_in_training)
    inputs, targets = encode_lm_tensors(training_text, tokenizer, args.max_seq_len, pad_id)
    qa_supervision = None
    if args.include_local_prompts_in_training:
        qa_supervision = qa_supervision_for_chunk(chunk, tokenizer, args.max_seq_len, pad_id, device)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters are available during chunk training.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    global_step = 0
    epoch_losses: list[float] = []
    for epoch in range(args.epochs_per_chunk):
        batch_losses: list[float] = []
        iterator = iter_batches(inputs, targets, args.batch_size)
        if not args.no_progress:
            total_batches = math.ceil(len(inputs) / args.batch_size)
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc=f"{method}:{chunk['chunk_id']}:epoch{epoch + 1}",
                leave=False,
            )
        for batch_inputs, batch_targets in iterator:
            optimizer.zero_grad(set_to_none=True)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits, _ = model(batch_inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_targets.reshape(-1))
            if qa_supervision is not None:
                qa_inputs, qa_targets, qa_mask = qa_supervision
                qa_logits, _ = model(qa_inputs)
                loss = loss + args.qa_loss_weight * masked_cross_entropy(qa_logits, qa_targets, qa_mask)
            if method == "gfo_soft":
                if anchors:
                    selected_anchors = select_anchor_batch(anchors, global_step, args.anchor_batch_size)
                    drift = anchor_drift_loss(model, selected_anchors, device=device)
                    loss = loss + args.anchor_drift_weight * drift
            elif method == "replay":
                if anchors:
                    selected_anchors = select_anchor_batch(anchors, global_step, args.anchor_batch_size)
                    replay = replay_anchor_loss(model, selected_anchors, device=device)
                    loss = loss + args.replay_loss_weight * replay
            elif method != "adamw":
                raise ValueError(f"Unknown training method: {method}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            global_step += 1
        if not batch_losses:
            raise RuntimeError(f"No batches were produced for chunk {chunk['chunk_id']!r}.")
        epoch_losses.append(float(sum(batch_losses) / len(batch_losses)))

    return {
        "train_loss": float(epoch_losses[-1]),
        "train_loss_mean": float(sum(epoch_losses) / len(epoch_losses)),
        "train_sequence_count": float(len(inputs)),
    }


def run_method(
    method: str,
    tokenizer: Tokenizer,
    chunks: Sequence[dict[str, object]],
    fact_probes: dict[str, list[str]],
    heldout_prompt_groups: dict[str, list[dict[str, str]]] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model = instantiate_model(args, tokenizer.get_vocab_size(), device)
    configure_trainable_parameters(model, train_embeddings=args.train_embeddings)
    pad_id = require_token_id(tokenizer, "[PAD]")
    anchors: list[ActivationAnchor] = []
    steps: list[dict[str, object]] = []

    selected_chunks = list(chunks[: args.max_chunks]) if args.max_chunks > 0 else list(chunks)
    if not selected_chunks:
        raise ValueError("No chunks selected for training.")

    for chunk_index, chunk in enumerate(selected_chunks):
        chunk_id = str(chunk["chunk_id"])
        print(f"[{method}] chunk {chunk_index + 1}/{len(selected_chunks)}: {chunk_id}")
        train_stats = train_chunk(model, tokenizer, chunk, anchors, method, args, pad_id, device)
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
        drift_stats = evaluate_anchor_drift(model, anchors, device=device)
        anchor_count_before_add = len(anchors)
        add_chunk_anchors(model, tokenizer, chunk, fact_probes, anchors, args, pad_id, device)
        step = {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "anchor_count_before_add": anchor_count_before_add,
            "anchor_count": len(anchors),
            **train_stats,
            **eval_stats,
            **drift_stats,
        }
        steps.append(step)

    final_summary = summarize_steps(steps)
    return {
        "method": method,
        "steps": steps,
        "summary": final_summary,
    }


def summarize_steps(steps: Sequence[dict[str, object]]) -> dict[str, float]:
    if not steps:
        raise ValueError("Cannot summarize an empty step list.")
    numeric_keys = [
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
        "anchor_drift_mean",
        "anchor_drift_max",
    ]
    summary: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(step[key]) for step in steps if key in step and not math.isnan(float(step[key]))]
        summary[f"{key}_mean"] = float(sum(values) / len(values)) if values else float("nan")
        summary[f"{key}_final"] = float(steps[-1][key]) if key in steps[-1] else float("nan")
    return summary


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_model_path": str(args.base_model_path),
        "tokenizer_path": str(args.tokenizer_path),
        "chunks_path": str(args.chunks_path),
        "fact_probes_path": str(args.fact_probes_path),
        "heldout_prompts_path": None if args.heldout_prompts_path is None else str(args.heldout_prompts_path),
        "methods": args.methods,
        "seed": args.seed,
        "max_chunks": args.max_chunks,
        "epochs_per_chunk": args.epochs_per_chunk,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "anchor_drift_weight": args.anchor_drift_weight,
        "replay_loss_weight": args.replay_loss_weight,
        "anchor_batch_size": args.anchor_batch_size,
        "max_anchors_per_chunk": args.max_anchors_per_chunk,
        "max_fact_probes_per_chunk": args.max_fact_probes_per_chunk,
        "include_local_prompts_in_training": args.include_local_prompts_in_training,
        "anchor_local_prompts": args.anchor_local_prompts,
        "qa_anchor_mode": args.qa_anchor_mode,
        "train_embeddings": args.train_embeddings,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "max_seq_len": args.max_seq_len,
        "device": args.device,
    }


def print_summary(report: dict[str, object]) -> None:
    print("\nREAL-BOOK GFO ACTIVATION-ANCHOR SUMMARY")
    print("=" * 112)
    for method, method_report in report["methods"].items():  # type: ignore[union-attr]
        summary = method_report["summary"]
        print(f"method={method}")
        print("-" * 112)
        for key in (
            "local_token_accuracy_mean",
            "retention_token_accuracy_mean",
            "composition_token_accuracy_mean",
            "local_generation_match_mean",
            "retention_generation_match_mean",
            "composition_generation_match_mean",
            "heldout_retention_token_accuracy_final",
            "heldout_retention_generation_match_final",
            "heldout_composition_token_accuracy_final",
            "heldout_composition_generation_match_final",
            "anchor_drift_mean_final",
            "anchor_drift_max_final",
        ):
            value = summary.get(key, float("nan"))
            print(f"{key:42s} {value:.4f}")
        print("-" * 112)
    print("=" * 112)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--fact-probes-path", type=Path, default=Path("data/real_book/fact_probes.json"))
    parser.add_argument("--heldout-prompts-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gfo-real-book-activation-cl.json"))
    parser.add_argument("--methods", type=parse_methods, default=["adamw", "gfo_soft"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--epochs-per-chunk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--qa-loss-weight", type=float, default=5.0)
    parser.add_argument("--include-local-prompts-in-training", action="store_true")
    parser.add_argument("--anchor-drift-weight", type=float, default=5.0)
    parser.add_argument("--replay-loss-weight", type=float, default=5.0)
    parser.add_argument("--anchor-batch-size", type=int, default=4)
    parser.add_argument("--anchor-importance", type=float, default=1.0)
    parser.add_argument("--max-anchors-per-chunk", type=int, default=8)
    parser.add_argument("--max-fact-probes-per-chunk", type=int, default=8)
    parser.add_argument("--anchor-local-prompts", action="store_true")
    parser.add_argument("--qa-anchor-mode", choices=["full_sequence", "answer_tokens"], default="full_sequence")
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_chunks < 0:
        raise ValueError("--max-chunks must be non-negative.")
    if args.epochs_per_chunk <= 0:
        raise ValueError("--epochs-per-chunk must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.anchor_drift_weight < 0:
        raise ValueError("--anchor-drift-weight must be non-negative.")
    if args.replay_loss_weight < 0:
        raise ValueError("--replay-loss-weight must be non-negative.")
    if args.anchor_batch_size <= 0:
        raise ValueError("--anchor-batch-size must be positive.")
    if args.max_anchors_per_chunk < 0:
        raise ValueError("--max-anchors-per-chunk must be non-negative.")
    if args.max_fact_probes_per_chunk < 0:
        raise ValueError("--max-fact-probes-per-chunk must be non-negative.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    fact_probes = load_fact_probes(args.fact_probes_path)
    heldout_prompt_groups = None
    if args.heldout_prompts_path is not None:
        heldout_prompt_groups = load_prompt_groups(args.heldout_prompts_path)
    device = resolve_device(args.device)
    methods = args.methods
    report = {
        "experiment": "gfo_real_book_activation_cl",
        "config": config_from_args(args),
        "methods": {},
    }
    for method in methods:
        set_seed(args.seed)
        method_args = copy.copy(args)
        method_args.methods = [method]
        report["methods"][method] = run_method(
            method,
            tokenizer,
            chunks,
            fact_probes,
            heldout_prompt_groups,
            method_args,
            device,
        )
    print_summary(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
