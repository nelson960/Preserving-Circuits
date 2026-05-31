#!/usr/bin/env python3
"""Native fixed-space transformer continual learning stress test.

This intentionally removes the external living map, anchor bank, replay bank,
pending concept list, and semantic margin store. The only continual-learning
mechanism under test is inside the transformer: sparse native trace adapters
plus routing pressure from model-native losses computed inside DecoderTransformer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from tokenizers import Tokenizer
from tqdm.auto import tqdm

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

from models import DecoderTransformer  # noqa: E402
from gco_optimizer import GeometricContinualOptimizer  # noqa: E402
from real_book_common import (  # noqa: E402
    make_qa_supervision,
    require_token_id,
    resolve_device,
)
from gfo_real_book_activation_cl import (  # noqa: E402
    build_training_text,
    configure_trainable_parameters,
    encode_lm_tensors,
    evaluate_chunk_prompts,
    evaluate_prompt_group,
    instantiate_model,
    iter_batches,
    load_chunks,
    load_prompt_groups,
    prompt_list,
    qa_supervision_for_chunk,
    set_seed,
)

def require_finite_tensor(name: str, tensor: torch.Tensor, context: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite tensor detected for {name} ({context}).")


def require_finite_float(name: str, value: float, context: str) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite value detected for {name}={value!r} ({context}).")


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
        "trainable_parameter_fraction": float(trainable / total),
    }


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


def component_mean(values: Sequence[float], name: str) -> float:
    if not values:
        raise RuntimeError(f"No values recorded for component {name!r}.")
    return float(sum(values) / len(values))


def build_optimizer(
    trainable_params: Sequence[torch.nn.Parameter],
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "gco":
        return GeometricContinualOptimizer(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pressure_beta=args.gco_pressure_beta,
            pressure_gamma_base=args.gco_pressure_gamma_base,
            pressure_mu_base=args.gco_pressure_mu_base,
            pressure_warmup_steps=args.gco_pressure_warmup_steps,
            interference_threshold=args.gco_interference_threshold,
            projection_mode=args.gco_projection_mode,
        )
    raise ValueError(f"Unknown optimizer {args.optimizer!r}.")


def collect_gco_pathways(
    model: DecoderTransformer,
    optimizer: torch.optim.Optimizer,
    records: list[dict[str, object]],
) -> None:
    if not isinstance(optimizer, GeometricContinualOptimizer):
        return
    records.extend(model.gco_mlp_pathways())


def train_chunk(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    chunk: dict[str, object],
    args: argparse.Namespace,
    pad_id: int,
    device: torch.device,
) -> dict[str, float]:
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

    named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_trainable:
        raise RuntimeError("No trainable parameters are available during native training.")
    trainable_params = [parameter for _, parameter in named_trainable]
    optimizer = build_optimizer(trainable_params, args)

    history: dict[str, list[float]] = {
        "train_lm_loss": [],
        "train_qa_loss": [],
        "train_composition_qa_loss": [],
        "train_native_slot_entropy_loss": [],
        "train_native_slot_balance_loss": [],
        "train_native_slot_strength_loss": [],
        "train_native_pressure_update_loss": [],
        "train_native_state_delta_loss": [],
        "train_native_pressure_sparsity_loss": [],
        "train_native_capacity_pressure_loss": [],
        "train_native_compression_pressure_loss": [],
        "train_native_forget_pressure_loss": [],
        "train_native_slot_entropy": [],
        "train_native_slot_active_fraction": [],
        "train_native_slot_max_share": [],
        "train_native_slot_pressure": [],
        "train_native_slot_write_gate": [],
        "train_native_slot_residual_gate": [],
        "train_native_slot_consolidation_gate": [],
        "train_native_slot_frequency": [],
        "train_native_slot_state_norm": [],
        "train_native_slot_state_delta": [],
        "train_native_fast_update_energy": [],
        "train_native_fast_key_delta_norm": [],
        "train_native_fast_memory_norm": [],
        "train_native_fast_value_norm": [],
        "train_native_fast_memory_strength": [],
        "train_native_write_rate": [],
        "train_native_error_pressure": [],
        "train_native_error_write_rate": [],
        "train_native_source_mix": [],
        "train_native_reason_gate": [],
        "train_native_reason_gain": [],
        "train_native_reason_update_energy": [],
        "train_native_usage_imbalance": [],
        "train_native_capacity_pressure": [],
        "train_native_compression_gate": [],
        "train_native_forget_gate": [],
        "train_native_forget_rate": [],
        "train_native_fast_read_gain": [],
        "train_native_slot_usage_ema_max": [],
        "train_native_slot_usage_ema_min": [],
        "train_native_routing_homeostasis": [],
        "train_native_routing_homeostasis_gain": [],
        "train_gco_projected_parameter_count": [],
        "train_gco_pressure_mean": [],
        "train_gco_pressure_max": [],
        "train_gco_overlap_mean": [],
        "train_gco_safe_update_ratio": [],
        "train_gco_projection_delta_ratio": [],
        "train_total_loss": [],
    }

    model.train()
    global_step = 0
    for epoch in range(args.epochs_per_chunk):
        iterator: Iterable[tuple[torch.Tensor, torch.Tensor]] = iter_batches(inputs, targets, args.batch_size)
        if not args.no_progress:
            iterator = tqdm(
                iterator,
                total=math.ceil(len(inputs) / args.batch_size),
                desc=f"native:{chunk['chunk_id']}:epoch{epoch + 1}",
                leave=False,
            )
        for batch_index, (batch_inputs, batch_targets) in enumerate(iterator):
            optimizer.zero_grad(set_to_none=True)
            gco_pathway_records: list[dict[str, object]] = []
            context = (
                f"chunk={chunk['chunk_id']}, epoch={epoch + 1}, "
                f"batch={batch_index + 1}, global_step={global_step}"
            )
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            lm_output = model.native_cl_loss(batch_inputs, batch_targets)
            collect_gco_pathways(model, optimizer, gco_pathway_records)
            logits = lm_output["logits"]
            loss = lm_output["loss"]
            lm_loss = lm_output["task_loss"]
            native_terms = lm_output["native_terms"]
            if not isinstance(logits, torch.Tensor):
                raise TypeError("model.native_cl_loss returned non-tensor logits.")
            if not isinstance(loss, torch.Tensor):
                raise TypeError("model.native_cl_loss returned non-tensor loss.")
            if not isinstance(lm_loss, torch.Tensor):
                raise TypeError("model.native_cl_loss returned non-tensor task_loss.")
            if not isinstance(native_terms, dict):
                raise TypeError("model.native_cl_loss returned non-dict native_terms.")
            require_finite_tensor("native_logits", logits, context)
            require_finite_tensor("native_lm_loss", lm_loss, context)

            qa_loss = logits.new_zeros(())
            if qa_supervision is not None:
                qa_inputs, qa_targets, qa_mask = qa_supervision
                qa_output = model.native_cl_loss(
                    qa_inputs,
                    qa_targets,
                    mask=qa_mask,
                    task_weight=args.qa_loss_weight,
                )
                collect_gco_pathways(model, optimizer, gco_pathway_records)
                qa_logits = qa_output["logits"]
                qa_loss = qa_output["task_loss"]
                qa_total_loss = qa_output["loss"]
                if not isinstance(qa_logits, torch.Tensor):
                    raise TypeError("model.native_cl_loss returned non-tensor QA logits.")
                if not isinstance(qa_loss, torch.Tensor):
                    raise TypeError("model.native_cl_loss returned non-tensor QA task_loss.")
                if not isinstance(qa_total_loss, torch.Tensor):
                    raise TypeError("model.native_cl_loss returned non-tensor QA loss.")
                require_finite_tensor("native_qa_logits", qa_logits, context)
                require_finite_tensor("native_qa_loss", qa_loss, context)
                loss = loss + qa_total_loss

            composition_loss = logits.new_zeros(())
            if composition_supervision is not None:
                comp_inputs, comp_targets, comp_mask = select_supervision_rows(
                    composition_supervision,
                    global_step=global_step,
                    batch_size=args.composition_supervision_batch_size,
                )
                comp_output = model.native_cl_loss(
                    comp_inputs,
                    comp_targets,
                    mask=comp_mask,
                    task_weight=args.composition_loss_weight,
                )
                collect_gco_pathways(model, optimizer, gco_pathway_records)
                comp_logits = comp_output["logits"]
                composition_loss = comp_output["task_loss"]
                comp_total_loss = comp_output["loss"]
                if not isinstance(comp_logits, torch.Tensor):
                    raise TypeError("model.native_cl_loss returned non-tensor composition logits.")
                if not isinstance(composition_loss, torch.Tensor):
                    raise TypeError("model.native_cl_loss returned non-tensor composition task_loss.")
                if not isinstance(comp_total_loss, torch.Tensor):
                    raise TypeError("model.native_cl_loss returned non-tensor composition loss.")
                require_finite_tensor("native_composition_logits", comp_logits, context)
                require_finite_tensor("native_composition_loss", composition_loss, context)
                loss = loss + comp_total_loss

            require_finite_tensor("native_train_loss", loss, context)
            loss.backward()
            for name, parameter in named_trainable:
                if parameter.grad is not None:
                    require_finite_tensor("native_gradient", parameter.grad, f"{context}, parameter={name}")
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip, error_if_nonfinite=True)
            if isinstance(optimizer, GeometricContinualOptimizer):
                if not gco_pathway_records:
                    raise RuntimeError("GCO optimizer did not receive any MLP activation pathways.")
                optimizer.set_pathways(gco_pathway_records)
            optimizer.step()
            for name, parameter in model.named_parameters():
                require_finite_tensor("native_parameter", parameter, f"{context}, parameter={name}")

            history["train_lm_loss"].append(float(lm_loss.detach().cpu()))
            history["train_qa_loss"].append(float(qa_loss.detach().cpu()))
            history["train_composition_qa_loss"].append(float(composition_loss.detach().cpu()))
            history["train_native_slot_entropy_loss"].append(
                float(native_terms["native_slot_entropy_loss"].detach().cpu())
            )
            history["train_native_slot_balance_loss"].append(
                float(native_terms["native_slot_balance_loss"].detach().cpu())
            )
            history["train_native_slot_strength_loss"].append(
                float(native_terms["native_slot_strength_loss"].detach().cpu())
            )
            history["train_native_pressure_update_loss"].append(
                float(native_terms["native_pressure_update_loss"].detach().cpu())
            )
            history["train_native_state_delta_loss"].append(float(native_terms["native_state_delta_loss"].detach().cpu()))
            history["train_native_pressure_sparsity_loss"].append(
                float(native_terms["native_pressure_sparsity_loss"].detach().cpu())
            )
            history["train_native_capacity_pressure_loss"].append(
                float(native_terms["native_capacity_pressure_loss"].detach().cpu())
            )
            history["train_native_compression_pressure_loss"].append(
                float(native_terms["native_compression_pressure_loss"].detach().cpu())
            )
            history["train_native_forget_pressure_loss"].append(
                float(native_terms["native_forget_pressure_loss"].detach().cpu())
            )
            history["train_native_slot_entropy"].append(float(native_terms["native_slot_entropy"].detach().cpu()))
            history["train_native_slot_active_fraction"].append(
                float(native_terms["native_slot_active_fraction"].detach().cpu())
            )
            history["train_native_slot_max_share"].append(float(native_terms["native_slot_max_share"].detach().cpu()))
            history["train_native_slot_pressure"].append(float(native_terms["native_slot_pressure"].detach().cpu()))
            history["train_native_slot_write_gate"].append(float(native_terms["native_slot_write_gate"].detach().cpu()))
            history["train_native_slot_residual_gate"].append(
                float(native_terms["native_slot_residual_gate"].detach().cpu())
            )
            history["train_native_slot_consolidation_gate"].append(
                float(native_terms["native_slot_consolidation_gate"].detach().cpu())
            )
            history["train_native_slot_frequency"].append(float(native_terms["native_slot_frequency"].detach().cpu()))
            history["train_native_slot_state_norm"].append(float(native_terms["native_slot_state_norm"].detach().cpu()))
            history["train_native_slot_state_delta"].append(
                float(native_terms["native_slot_state_delta"].detach().cpu())
            )
            history["train_native_fast_update_energy"].append(
                float(native_terms["native_fast_update_energy"].detach().cpu())
            )
            history["train_native_fast_key_delta_norm"].append(
                float(native_terms["native_fast_key_delta_norm"].detach().cpu())
            )
            history["train_native_fast_memory_norm"].append(
                float(native_terms["native_fast_memory_norm"].detach().cpu())
            )
            history["train_native_fast_value_norm"].append(
                float(native_terms["native_fast_value_norm"].detach().cpu())
            )
            history["train_native_fast_memory_strength"].append(
                float(native_terms["native_fast_memory_strength"].detach().cpu())
            )
            history["train_native_write_rate"].append(float(native_terms["native_write_rate"].detach().cpu()))
            history["train_native_error_pressure"].append(
                float(native_terms["native_error_pressure"].detach().cpu())
            )
            history["train_native_error_write_rate"].append(
                float(native_terms["native_error_write_rate"].detach().cpu())
            )
            history["train_native_source_mix"].append(float(native_terms["native_source_mix"].detach().cpu()))
            history["train_native_reason_gate"].append(float(native_terms["native_reason_gate"].detach().cpu()))
            history["train_native_reason_gain"].append(float(native_terms["native_reason_gain"].detach().cpu()))
            history["train_native_reason_update_energy"].append(
                float(native_terms["native_reason_update_energy"].detach().cpu())
            )
            history["train_native_usage_imbalance"].append(
                float(native_terms["native_usage_imbalance"].detach().cpu())
            )
            history["train_native_capacity_pressure"].append(
                float(native_terms["native_capacity_pressure"].detach().cpu())
            )
            history["train_native_compression_gate"].append(
                float(native_terms["native_compression_gate"].detach().cpu())
            )
            history["train_native_forget_gate"].append(float(native_terms["native_forget_gate"].detach().cpu()))
            history["train_native_forget_rate"].append(float(native_terms["native_forget_rate"].detach().cpu()))
            history["train_native_fast_read_gain"].append(
                float(native_terms["native_fast_read_gain"].detach().cpu())
            )
            history["train_native_slot_usage_ema_max"].append(
                float(native_terms["native_slot_usage_ema_max"].detach().cpu())
            )
            history["train_native_slot_usage_ema_min"].append(
                float(native_terms["native_slot_usage_ema_min"].detach().cpu())
            )
            history["train_native_routing_homeostasis"].append(
                float(native_terms["native_routing_homeostasis"].detach().cpu())
            )
            history["train_native_routing_homeostasis_gain"].append(
                float(native_terms["native_routing_homeostasis_gain"].detach().cpu())
            )
            gco_metrics = getattr(optimizer, "last_metrics", {})
            history["train_gco_projected_parameter_count"].append(
                float(gco_metrics.get("gco_projected_parameter_count", 0.0))
            )
            history["train_gco_pressure_mean"].append(float(gco_metrics.get("gco_pressure_mean", 0.0)))
            history["train_gco_pressure_max"].append(float(gco_metrics.get("gco_pressure_max", 0.0)))
            history["train_gco_overlap_mean"].append(float(gco_metrics.get("gco_overlap_mean", 0.0)))
            history["train_gco_safe_update_ratio"].append(
                float(gco_metrics.get("gco_safe_update_ratio_mean", 1.0))
            )
            history["train_gco_projection_delta_ratio"].append(
                float(gco_metrics.get("gco_projection_delta_ratio_mean", 0.0))
            )
            history["train_total_loss"].append(float(loss.detach().cpu()))
            global_step += 1

    result = {
        "train_sequence_count": float(len(inputs)),
    }
    for key, values in history.items():
        result[f"{key}_mean"] = component_mean(values, key)
        result[f"{key}_final"] = float(values[-1])
    return result


def mean_metric(rows: Sequence[dict[str, object]], key: str) -> float:
    if not rows:
        raise ValueError(f"Cannot average empty metric list for {key!r}.")
    values = [float(row[key]) for row in rows]
    for value in values:
        require_finite_float(key, value, "mean_metric")
    return float(sum(values) / len(values))


def evaluate_seen_chunks(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    chunks: Sequence[dict[str, object]],
    device: torch.device,
    best_retention: dict[str, float],
    best_composition: dict[str, float],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    retention_forgetting: list[float] = []
    composition_forgetting: list[float] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        stats = evaluate_chunk_prompts(model, tokenizer, chunk, device)
        retention = float(stats["retention_token_accuracy"])
        composition = float(stats["composition_token_accuracy"])
        previous_best_retention = best_retention.get(chunk_id, retention)
        previous_best_composition = best_composition.get(chunk_id, composition)
        retention_forgetting.append(max(0.0, previous_best_retention - retention))
        composition_forgetting.append(max(0.0, previous_best_composition - composition))
        best_retention[chunk_id] = max(previous_best_retention, retention)
        best_composition[chunk_id] = max(previous_best_composition, composition)
        rows.append(
            {
                "chunk_id": chunk_id,
                "retention_token_accuracy": retention,
                "composition_token_accuracy": composition,
                "retention_generation_match": float(stats["retention_generation_match"]),
                "composition_generation_match": float(stats["composition_generation_match"]),
            }
        )

    return {
        "seen_chunk_count": float(len(rows)),
        "seen_retention_token_accuracy": mean_metric(rows, "retention_token_accuracy"),
        "seen_composition_token_accuracy": mean_metric(rows, "composition_token_accuracy"),
        "seen_retention_generation_match": mean_metric(rows, "retention_generation_match"),
        "seen_composition_generation_match": mean_metric(rows, "composition_generation_match"),
        "seen_retention_forgetting_mean": float(sum(retention_forgetting) / len(retention_forgetting)),
        "seen_retention_forgetting_max": float(max(retention_forgetting)),
        "seen_composition_forgetting_mean": float(sum(composition_forgetting) / len(composition_forgetting)),
        "seen_composition_forgetting_max": float(max(composition_forgetting)),
        "seen_chunk_records": rows,
    }


def run_native(
    tokenizer: Tokenizer,
    chunks: Sequence[dict[str, object]],
    heldout_prompt_groups: dict[str, list[dict[str, str]]] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model = instantiate_model(args, tokenizer.get_vocab_size(), device)
    configure_trainable_parameters(model, train_embeddings=args.train_embeddings)
    param_stats = parameter_count_stats(model)
    pad_id = require_token_id(tokenizer, "[PAD]")
    selected_chunks = list(chunks[: args.max_chunks]) if args.max_chunks > 0 else list(chunks)
    if not selected_chunks:
        raise ValueError("No chunks selected for native training.")

    best_retention: dict[str, float] = {}
    best_composition: dict[str, float] = {}
    steps: list[dict[str, object]] = []
    for chunk_index, chunk in enumerate(selected_chunks):
        chunk_id = str(chunk["chunk_id"])
        print(f"[native_gfo] chunk {chunk_index + 1}/{len(selected_chunks)}: {chunk_id}")
        train_stats = train_chunk(model, tokenizer, chunk, args, pad_id, device)
        current_eval = evaluate_chunk_prompts(model, tokenizer, chunk, device)
        seen_eval = evaluate_seen_chunks(
            model,
            tokenizer,
            selected_chunks[: chunk_index + 1],
            device,
            best_retention,
            best_composition,
        )
        heldout_eval: dict[str, object] = {}
        if heldout_prompt_groups is not None:
            if "retention_prompts" in heldout_prompt_groups:
                heldout_eval.update(
                    evaluate_prompt_group(
                        model,
                        tokenizer,
                        heldout_prompt_groups["retention_prompts"],
                        device,
                        "heldout_retention",
                    )
                )
            if "composition_prompts" in heldout_prompt_groups:
                heldout_eval.update(
                    evaluate_prompt_group(
                        model,
                        tokenizer,
                        heldout_prompt_groups["composition_prompts"],
                        device,
                        "heldout_composition",
                    )
                )
        steps.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": float(chunk_index),
                **param_stats,
                **train_stats,
                "current_retention_token_accuracy": float(current_eval["retention_token_accuracy"]),
                "current_composition_token_accuracy": float(current_eval["composition_token_accuracy"]),
                "current_retention_generation_match": float(current_eval["retention_generation_match"]),
                "current_composition_generation_match": float(current_eval["composition_generation_match"]),
                **seen_eval,
                **heldout_eval,
            }
        )

    return {
        "method": "native_gfo",
        "steps": steps,
        "summary": summarize_steps(steps),
    }


def summarize_steps(steps: Sequence[dict[str, object]]) -> dict[str, float]:
    if not steps:
        raise ValueError("Cannot summarize an empty native step list.")
    numeric_keys = [
        "total_parameter_count",
        "trainable_parameter_count",
        "trainable_parameter_fraction",
        "train_lm_loss_mean",
        "train_qa_loss_mean",
        "train_composition_qa_loss_mean",
        "train_native_slot_entropy_loss_mean",
        "train_native_slot_balance_loss_mean",
        "train_native_slot_strength_loss_mean",
        "train_native_pressure_update_loss_mean",
        "train_native_state_delta_loss_mean",
        "train_native_pressure_sparsity_loss_mean",
        "train_native_capacity_pressure_loss_mean",
        "train_native_compression_pressure_loss_mean",
        "train_native_forget_pressure_loss_mean",
        "train_native_slot_entropy_mean",
        "train_native_slot_active_fraction_mean",
        "train_native_slot_max_share_mean",
        "train_native_slot_pressure_mean",
        "train_native_slot_write_gate_mean",
        "train_native_slot_residual_gate_mean",
        "train_native_slot_consolidation_gate_mean",
        "train_native_slot_frequency_mean",
        "train_native_slot_state_norm_mean",
        "train_native_slot_state_delta_mean",
        "train_native_fast_update_energy_mean",
        "train_native_fast_key_delta_norm_mean",
        "train_native_fast_memory_norm_mean",
        "train_native_fast_value_norm_mean",
        "train_native_fast_memory_strength_mean",
        "train_native_write_rate_mean",
        "train_native_error_pressure_mean",
        "train_native_error_write_rate_mean",
        "train_native_source_mix_mean",
        "train_native_reason_gate_mean",
        "train_native_reason_gain_mean",
        "train_native_reason_update_energy_mean",
        "train_native_usage_imbalance_mean",
        "train_native_capacity_pressure_mean",
        "train_native_compression_gate_mean",
        "train_native_forget_gate_mean",
        "train_native_forget_rate_mean",
        "train_native_fast_read_gain_mean",
        "train_native_slot_usage_ema_max_mean",
        "train_native_slot_usage_ema_min_mean",
        "train_native_routing_homeostasis_mean",
        "train_native_routing_homeostasis_gain_mean",
        "train_gco_projected_parameter_count_mean",
        "train_gco_pressure_mean_mean",
        "train_gco_pressure_max_mean",
        "train_gco_overlap_mean_mean",
        "train_gco_safe_update_ratio_mean",
        "train_gco_projection_delta_ratio_mean",
        "train_total_loss_mean",
        "current_retention_token_accuracy",
        "current_composition_token_accuracy",
        "current_retention_generation_match",
        "current_composition_generation_match",
        "seen_retention_token_accuracy",
        "seen_composition_token_accuracy",
        "seen_retention_generation_match",
        "seen_composition_generation_match",
        "seen_retention_forgetting_mean",
        "seen_retention_forgetting_max",
        "seen_composition_forgetting_mean",
        "seen_composition_forgetting_max",
        "heldout_retention_token_accuracy",
        "heldout_retention_generation_match",
        "heldout_composition_token_accuracy",
        "heldout_composition_generation_match",
    ]
    summary: dict[str, float] = {}
    for key in numeric_keys:
        values: list[float] = []
        for step in steps:
            if key not in step:
                continue
            value = float(step[key])
            require_finite_float(key, value, f"summary chunk={step.get('chunk_id', '<unknown>')}")
            values.append(value)
        if values:
            summary[f"{key}_mean"] = float(sum(values) / len(values))
            summary[f"{key}_final"] = float(steps[-1][key])
    return summary


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_model_path": str(args.base_model_path),
        "tokenizer_path": str(args.tokenizer_path),
        "chunks_path": str(args.chunks_path),
        "heldout_prompts_path": None if args.heldout_prompts_path is None else str(args.heldout_prompts_path),
        "seed": args.seed,
        "max_chunks": args.max_chunks,
        "epochs_per_chunk": args.epochs_per_chunk,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gco_projection_mode": args.gco_projection_mode,
        "gco_pressure_beta": args.gco_pressure_beta,
        "gco_pressure_gamma_base": args.gco_pressure_gamma_base,
        "gco_pressure_mu_base": args.gco_pressure_mu_base,
        "gco_pressure_warmup_steps": args.gco_pressure_warmup_steps,
        "gco_interference_threshold": args.gco_interference_threshold,
        "qa_loss_weight": args.qa_loss_weight,
        "composition_loss_weight": args.composition_loss_weight,
        "composition_supervision_batch_size": args.composition_supervision_batch_size,
        "include_local_prompts_in_training": args.include_local_prompts_in_training,
        "include_composition_prompts_in_training": args.include_composition_prompts_in_training,
        "native_trace_slots": args.native_trace_slots,
        "native_trace_rank": args.native_trace_rank,
        "native_trace_top_k": args.native_trace_top_k,
        "native_trace_init_scale": args.native_trace_init_scale,
        "native_trace_state_update_rate": args.native_trace_state_update_rate,
        "native_trace_state_decay": args.native_trace_state_decay,
        "native_trace_initial_strength_logit": args.native_trace_initial_strength_logit,
        "native_slot_entropy_weight": args.native_slot_entropy_weight,
        "native_slot_balance_weight": args.native_slot_balance_weight,
        "native_slot_strength_weight": args.native_slot_strength_weight,
        "native_pressure_update_weight": args.native_pressure_update_weight,
        "native_state_delta_weight": args.native_state_delta_weight,
        "native_pressure_sparsity_weight": args.native_pressure_sparsity_weight,
        "native_capacity_pressure_weight": args.native_capacity_pressure_weight,
        "native_compression_pressure_weight": args.native_compression_pressure_weight,
        "native_forget_pressure_weight": args.native_forget_pressure_weight,
        "train_native_traces_only": args.train_native_traces_only,
        "train_embeddings": args.train_embeddings,
        "grad_clip": args.grad_clip,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "max_seq_len": args.max_seq_len,
        "device": args.device,
    }


def print_summary(report: dict[str, object]) -> None:
    summary = report["summary"]  # type: ignore[index]
    print("\nREAL-BOOK NATIVE GFO TRANSFORMER SUMMARY")
    print("=" * 112)
    for key in (
        "trainable_parameter_count_final",
        "trainable_parameter_fraction_final",
        "train_total_loss_mean_final",
        "train_lm_loss_mean_final",
        "train_native_slot_entropy_mean_final",
        "train_native_slot_active_fraction_mean_final",
        "train_native_slot_max_share_mean_final",
        "train_native_slot_pressure_mean_final",
        "train_native_slot_write_gate_mean_final",
        "train_native_slot_consolidation_gate_mean_final",
        "train_native_slot_frequency_mean_final",
        "train_native_slot_state_norm_mean_final",
        "train_native_slot_state_delta_mean_final",
        "train_native_fast_update_energy_mean_final",
        "train_native_fast_key_delta_norm_mean_final",
        "train_native_fast_memory_norm_mean_final",
        "train_native_fast_value_norm_mean_final",
        "train_native_fast_memory_strength_mean_final",
        "train_native_write_rate_mean_final",
        "train_native_error_pressure_mean_final",
        "train_native_error_write_rate_mean_final",
        "train_native_source_mix_mean_final",
        "train_native_reason_gate_mean_final",
        "train_native_reason_gain_mean_final",
        "train_native_reason_update_energy_mean_final",
        "train_native_usage_imbalance_mean_final",
        "train_native_capacity_pressure_mean_final",
        "train_native_compression_gate_mean_final",
        "train_native_forget_gate_mean_final",
        "train_native_forget_rate_mean_final",
        "train_native_fast_read_gain_mean_final",
        "train_native_slot_usage_ema_max_mean_final",
        "train_native_slot_usage_ema_min_mean_final",
        "train_native_routing_homeostasis_mean_final",
        "train_native_routing_homeostasis_gain_mean_final",
        "train_gco_projected_parameter_count_mean_final",
        "train_gco_pressure_mean_mean_final",
        "train_gco_pressure_max_mean_final",
        "train_gco_overlap_mean_mean_final",
        "train_gco_safe_update_ratio_mean_final",
        "train_gco_projection_delta_ratio_mean_final",
        "current_retention_token_accuracy_final",
        "current_composition_token_accuracy_final",
        "seen_retention_token_accuracy_final",
        "seen_composition_token_accuracy_final",
        "seen_retention_forgetting_mean_final",
        "seen_retention_forgetting_max_final",
        "seen_composition_forgetting_mean_final",
        "seen_composition_forgetting_max_final",
        "heldout_retention_token_accuracy_final",
        "heldout_retention_generation_match_final",
        "heldout_composition_token_accuracy_final",
        "heldout_composition_generation_match_final",
    ):
        value = float(summary.get(key, float("nan")))
        print(f"{key:54s} {value:.4f}")
    print("=" * 112)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--heldout-prompts-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gfo-real-book-native-cl.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--epochs-per-chunk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimizer", choices=["adamw", "gco"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gco-projection-mode", choices=["pre", "post"], default="pre")
    parser.add_argument("--gco-pressure-beta", type=float, default=0.99)
    parser.add_argument("--gco-pressure-gamma-base", type=float, default=20.0)
    parser.add_argument("--gco-pressure-mu-base", type=float, default=0.01)
    parser.add_argument("--gco-pressure-warmup-steps", type=int, default=20)
    parser.add_argument("--gco-interference-threshold", type=float, default=0.05)
    parser.add_argument("--qa-loss-weight", type=float, default=5.0)
    parser.add_argument("--composition-loss-weight", type=float, default=5.0)
    parser.add_argument("--composition-supervision-batch-size", type=int, default=0)
    parser.add_argument("--include-local-prompts-in-training", action="store_true")
    parser.add_argument("--include-composition-prompts-in-training", action="store_true")
    parser.add_argument("--native-trace-slots", type=int, default=8)
    parser.add_argument("--native-trace-rank", type=int, default=8)
    parser.add_argument("--native-trace-top-k", type=int, default=2)
    parser.add_argument("--native-trace-init-scale", type=float, default=1e-3)
    parser.add_argument("--native-trace-state-update-rate", type=float, default=0.05)
    parser.add_argument("--native-trace-state-decay", type=float, default=0.99)
    parser.add_argument("--native-trace-initial-strength-logit", type=float, default=-4.0)
    parser.add_argument("--native-slot-entropy-weight", type=float, default=0.01)
    parser.add_argument("--native-slot-balance-weight", type=float, default=0.1)
    parser.add_argument("--native-slot-strength-weight", type=float, default=0.0)
    parser.add_argument("--native-pressure-update-weight", type=float, default=0.0)
    parser.add_argument("--native-state-delta-weight", type=float, default=0.0)
    parser.add_argument("--native-pressure-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--native-capacity-pressure-weight", type=float, default=0.0)
    parser.add_argument("--native-compression-pressure-weight", type=float, default=0.0)
    parser.add_argument("--native-forget-pressure-weight", type=float, default=0.0)
    parser.add_argument("--train-native-traces-only", action="store_true")
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
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if not 0.0 <= args.gco_pressure_beta < 1.0:
        raise ValueError("--gco-pressure-beta must be in [0, 1).")
    if args.gco_pressure_gamma_base <= 0:
        raise ValueError("--gco-pressure-gamma-base must be positive.")
    if args.gco_pressure_mu_base < 0:
        raise ValueError("--gco-pressure-mu-base must be non-negative.")
    if args.gco_pressure_warmup_steps < 0:
        raise ValueError("--gco-pressure-warmup-steps must be non-negative.")
    if args.gco_interference_threshold < 0:
        raise ValueError("--gco-interference-threshold must be non-negative.")
    if args.qa_loss_weight < 0:
        raise ValueError("--qa-loss-weight must be non-negative.")
    if args.composition_loss_weight < 0:
        raise ValueError("--composition-loss-weight must be non-negative.")
    if args.composition_supervision_batch_size < 0:
        raise ValueError("--composition-supervision-batch-size must be non-negative.")
    if args.native_trace_slots <= 1:
        raise ValueError("--native-trace-slots must be greater than one for native CL.")
    if args.native_trace_rank <= 0:
        raise ValueError("--native-trace-rank must be positive.")
    if args.native_trace_top_k <= 0:
        raise ValueError("--native-trace-top-k must be positive.")
    if args.native_trace_top_k > args.native_trace_slots:
        raise ValueError("--native-trace-top-k must be <= --native-trace-slots.")
    if args.native_trace_init_scale <= 0:
        raise ValueError("--native-trace-init-scale must be positive.")
    if not 0.0 <= args.native_trace_state_update_rate <= 1.0:
        raise ValueError("--native-trace-state-update-rate must be in [0, 1].")
    if not 0.0 <= args.native_trace_state_decay <= 1.0:
        raise ValueError("--native-trace-state-decay must be in [0, 1].")
    if not math.isfinite(args.native_trace_initial_strength_logit):
        raise ValueError("--native-trace-initial-strength-logit must be finite.")
    if args.native_slot_entropy_weight < 0:
        raise ValueError("--native-slot-entropy-weight must be non-negative.")
    if args.native_slot_balance_weight < 0:
        raise ValueError("--native-slot-balance-weight must be non-negative.")
    if args.native_slot_strength_weight < 0:
        raise ValueError("--native-slot-strength-weight must be non-negative.")
    if args.native_pressure_update_weight < 0:
        raise ValueError("--native-pressure-update-weight must be non-negative.")
    if args.native_state_delta_weight < 0:
        raise ValueError("--native-state-delta-weight must be non-negative.")
    if args.native_pressure_sparsity_weight < 0:
        raise ValueError("--native-pressure-sparsity-weight must be non-negative.")
    if args.native_capacity_pressure_weight < 0:
        raise ValueError("--native-capacity-pressure-weight must be non-negative.")
    if args.native_compression_pressure_weight < 0:
        raise ValueError("--native-compression-pressure-weight must be non-negative.")
    if args.native_forget_pressure_weight < 0:
        raise ValueError("--native-forget-pressure-weight must be non-negative.")
    if args.grad_clip < 0:
        raise ValueError("--grad-clip must be non-negative.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    set_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    heldout_prompt_groups = None
    if args.heldout_prompts_path is not None:
        heldout_prompt_groups = load_prompt_groups(args.heldout_prompts_path)
    device = resolve_device(args.device)
    method_args = copy.copy(args)
    report = {
        "experiment": "gfo_real_book_native_cl",
        "config": config_from_args(args),
        **run_native(tokenizer, chunks, heldout_prompt_groups, method_args, device),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
