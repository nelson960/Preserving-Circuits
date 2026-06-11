"""Recursive controlled continual-learning architecture diagnostic.

This test combines the pieces that looked useful in isolation:

    stage bootstrap
    labeled behavior memory: preserve / guard / drop
    recursive learning stages
    optional plastic adapter followed by consolidation

The experiment is intentionally explicit about what must be preserved and what
is allowed to be forgotten. It is not a native autonomous policy yet. It tests
whether the architecture can obey those control signals over multiple stages.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    distillation_kl,
    load_checkpoint,
    set_only_native_weights_trainable,
    trainable_weight_parameters,
)
from experiments.gco_math.gco_tiny_cl_bridge_adapter_consolidation import (
    AdapterWrappedTransformer,
    FinalResidualAdapter,
    freeze_model,
    trainable_adapter_parameters,
)
from experiments.gco_math.gco_tiny_reusable_computation_cl import (
    EncodedExample,
    QAExample,
    affordance_examples,
    batch_examples,
    collect_example_logits,
    composition_chain_examples,
    composition_example,
    composition_rule_examples,
    distillation_loss_for_examples,
    encode_examples,
    evaluate_examples,
    make_model_from_config,
    make_optimizer_for_model,
    masked_ce_loss,
    possession_examples,
    relation_items,
)


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def parse_people(raw: str, *, name: str) -> set[str]:
    people = {item.strip() for item in raw.split(",") if item.strip()}
    if not people:
        raise ValueError(f"--{name.replace('_', '-')} must contain at least one person.")
    known = {item.person for item in relation_items()}
    unknown = people.difference(known)
    if unknown:
        raise ValueError(f"Unknown people in --{name.replace('_', '-')}: {sorted(unknown)}.")
    return people


def validate_policy(*, preserve_people: set[str], drop_people: set[str], composition_holdout_people: set[str]) -> None:
    overlap = preserve_people.intersection(drop_people)
    if overlap:
        raise ValueError(f"People cannot be both preserve and drop: {sorted(overlap)}.")
    known = {item.person for item in relation_items()}
    unknown_holdout = composition_holdout_people.difference(known)
    if unknown_holdout:
        raise ValueError(f"Unknown composition holdout people: {sorted(unknown_holdout)}.")


def build_raw_architecture_data(
    *,
    preserve_people: set[str],
    drop_people: set[str],
    composition_holdout_people: set[str],
    include_composition_rules: bool,
) -> dict[str, list[QAExample]]:
    items = relation_items()
    stage1_items = [item for item in items if item.stage == 1]
    stage2_items = [item for item in items if item.stage == 2]
    known_people = {item.person for item in items}
    composition_train_people = known_people.difference(composition_holdout_people)

    preserve = [
        example
        for item in stage1_items
        if item.person in preserve_people
        for example in possession_examples(item)
    ]
    drop = [
        example
        for item in stage1_items
        if item.person in drop_people
        for example in possession_examples(item)
    ]
    neutral = [
        example
        for item in stage1_items
        if item.person not in preserve_people and item.person not in drop_people
        for example in possession_examples(item)
    ]
    if not preserve:
        raise ValueError("Preserve policy produced no examples.")
    if not drop:
        raise ValueError("Drop policy produced no examples.")
    if not neutral:
        raise ValueError("Neutral guard policy produced no examples.")

    stage1 = [example for item in stage1_items for example in possession_examples(item)]
    stage2 = [example for item in stage2_items for example in possession_examples(item)]
    stage3 = [example for item in items for example in affordance_examples(item)]
    if include_composition_rules:
        stage3.extend(composition_rule_examples())
    stage3.extend(
        composition_example(item, trained=True)
        for item in items
        if item.person in composition_train_people
    )
    if include_composition_rules:
        for item in items:
            if item.person in composition_train_people:
                stage3.extend(composition_chain_examples(item))

    eval_all: list[QAExample] = []
    for item in items:
        eval_all.extend(possession_examples(item))
        eval_all.extend(affordance_examples(item))
        eval_all.append(composition_example(item, trained=item.person in composition_train_people))
    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "preserve": preserve,
        "drop": drop,
        "neutral": neutral,
        "eval_all": eval_all,
    }


def encode_groups(
    raw_groups: dict[str, list[QAExample]],
    tokenizer: Tokenizer,
    *,
    max_seq_len: int,
) -> dict[str, list[EncodedExample]]:
    return {name: encode_examples(examples, tokenizer, max_seq_len=max_seq_len) for name, examples in raw_groups.items()}


def cap_examples(examples: list[EncodedExample], *, budget: int) -> list[EncodedExample]:
    positive_int("budget", budget)
    if len(examples) <= budget:
        return list(examples)
    indices = torch.linspace(0, len(examples) - 1, steps=budget).round().to(dtype=torch.long).unique(sorted=True)
    if indices.numel() != budget:
        raise RuntimeError(f"Budget selector returned {indices.numel()} examples for budget={budget}.")
    return [examples[int(index)] for index in indices.tolist()]


def clone_core(model: nn.Module, *, checkpoint: dict[str, Any], device: torch.device) -> nn.Module:
    clone = make_model_from_config(checkpoint=checkpoint, device=device, seed=0)
    missing, unexpected = clone.load_state_dict(model.state_dict(), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Core clone mismatch: missing={missing}, unexpected={unexpected}.")
    return clone


def drop_suppression_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    target_probability: float,
) -> torch.Tensor:
    positive_float("target_probability", target_probability)
    if target_probability >= 1.0:
        raise ValueError(f"target_probability must be below 1.0, got {target_probability}.")
    log_probs = F.log_softmax(logits, dim=-1)
    old_answer_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    threshold = math.log(target_probability)
    penalty = F.relu(old_answer_log_probs - threshold).square()
    return (penalty * mask).sum() / mask.sum().clamp_min(1.0)


def controlled_loss(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    train_mask: torch.Tensor,
    preserve_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
    guard_examples: list[EncodedExample],
    guard_logits: list[torch.Tensor],
    drop_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(train_inputs)
    ce = masked_ce_loss(logits, train_targets, train_mask)
    batch_size = int(train_inputs.shape[0])

    preserve_indices = torch.randint(
        low=0,
        high=len(preserve_examples),
        size=(batch_size,),
        generator=generator,
        device=torch.device("cpu"),
    )
    preserve_inputs, _preserve_targets, _preserve_mask, preserve_selected = batch_examples(
        preserve_examples,
        indices=preserve_indices,
        pad_id=pad_id,
        device=device,
    )
    preserve_current = model(preserve_inputs)
    preserve_loss = distillation_loss_for_examples(
        preserve_current,
        preserve_selected,
        preserve_logits,
        preserve_indices,
        temperature=args.distill_temperature,
        device=device,
    )

    guard_indices = torch.randint(
        low=0,
        high=len(guard_examples),
        size=(batch_size,),
        generator=generator,
        device=torch.device("cpu"),
    )
    guard_inputs, _guard_targets, _guard_mask, guard_selected = batch_examples(
        guard_examples,
        indices=guard_indices,
        pad_id=pad_id,
        device=device,
    )
    guard_current = model(guard_inputs)
    guard_loss = distillation_loss_for_examples(
        guard_current,
        guard_selected,
        guard_logits,
        guard_indices,
        temperature=args.distill_temperature,
        device=device,
    )

    if args.drop_loss_weight > 0.0:
        drop_indices = torch.randint(
            low=0,
            high=len(drop_examples),
            size=(batch_size,),
            generator=generator,
            device=torch.device("cpu"),
        )
        drop_inputs, drop_targets, drop_mask, _drop_selected = batch_examples(
            drop_examples,
            indices=drop_indices,
            pad_id=pad_id,
            device=device,
        )
        drop_current = model(drop_inputs)
        drop_loss = drop_suppression_loss(
            logits=drop_current,
            targets=drop_targets,
            mask=drop_mask,
            target_probability=args.drop_target_probability,
        )
    else:
        drop_loss = ce.new_zeros(())

    loss = (
        ce
        + args.lambda_preserve * preserve_loss
        + args.lambda_guard * guard_loss
        + args.drop_loss_weight * drop_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "preserve": float(preserve_loss.detach().cpu()),
        "guard": float(guard_loss.detach().cpu()),
        "drop": float(drop_loss.detach().cpu()),
    }
    return loss, metrics


def train_direct_stage(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    stage_examples: list[EncodedExample],
    preserve_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
    guard_examples: list[EncodedExample],
    guard_logits: list[torch.Tensor],
    drop_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
    label: str,
) -> list[dict[str, float]]:
    set_only_native_weights_trainable(model)
    optimizer = make_optimizer_for_model(args, trainable_weight_parameters(model))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(stage_examples), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0, "preserve": 0.0, "guard": 0.0, "drop": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(stage_examples), args.batch_size), desc=f"{label} {epoch}/{epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, row = controlled_loss(
                args=args,
                model=model,
                train_inputs=inputs,
                train_targets=targets,
                train_mask=mask,
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
                guard_examples=guard_examples,
                guard_logits=guard_logits,
                drop_examples=drop_examples,
                pad_id=pad_id,
                device=device,
                generator=generator,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(model), args.grad_clip)
            optimizer.step()
            for key, value in row.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"ce": f"{row['ce']:.3g}", "p": f"{row['preserve']:.3g}", "g": f"{row['guard']:.3g}", "d": f"{row['drop']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"{label} epoch {epoch} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        trace.append(epoch_row)
        print(
            f"{label} epoch={epoch:4d} loss={epoch_row['loss']:.5f} ce={epoch_row['ce']:.5f} "
            f"preserve={epoch_row['preserve']:.5f} guard={epoch_row['guard']:.5f} drop={epoch_row['drop']:.5f}"
        )
    return trace


def train_bootstrap_stage(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    stage_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
    label: str,
) -> list[dict[str, float]]:
    set_only_native_weights_trainable(model)
    optimizer = make_optimizer_for_model(args, trainable_weight_parameters(model))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(stage_examples), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(stage_examples), args.batch_size), desc=f"{label} {epoch}/{epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            ce = masked_ce_loss(logits, targets, mask)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(model), args.grad_clip)
            optimizer.step()
            value = float(ce.detach().cpu())
            totals["loss"] += value
            totals["ce"] += value
            batches += 1
            pbar.set_postfix({"ce": f"{value:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"{label} epoch {epoch} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        trace.append(epoch_row)
        print(f"{label} epoch={epoch:4d} ce={epoch_row['ce']:.5f}")
    return trace


def train_adapter_controlled_stage(
    *,
    args: argparse.Namespace,
    core: nn.Module,
    checkpoint: dict[str, Any],
    stage_examples: list[EncodedExample],
    preserve_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
    guard_examples: list[EncodedExample],
    guard_logits: list[torch.Tensor],
    drop_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[nn.Module, dict[str, Any]]:
    freeze_model(core)
    adapter = FinalResidualAdapter(
        d_model=int(checkpoint["model_config"]["d_model"]),
        rank=args.adapter_rank,
        scale=args.adapter_scale,
    ).to(device)
    wrapped = AdapterWrappedTransformer(core, adapter).to(device)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    adapter_optimizer = make_optimizer_for_model(args, trainable_adapter_parameters(adapter))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    adapter_trace: list[dict[str, float]] = []
    for epoch in range(1, args.adapter_epochs + 1):
        wrapped.train()
        permutation = torch.randperm(len(stage_examples), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0, "preserve": 0.0, "guard": 0.0, "drop": 0.0, "adapter": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(stage_examples), args.batch_size), desc=f"{label} adapter {epoch}/{args.adapter_epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            adapter_optimizer.zero_grad(set_to_none=True)
            loss, row = controlled_loss(
                args=args,
                model=wrapped,
                train_inputs=inputs,
                train_targets=targets,
                train_mask=mask,
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
                guard_examples=guard_examples,
                guard_logits=guard_logits,
                drop_examples=drop_examples,
                pad_id=pad_id,
                device=device,
                generator=generator,
            )
            adapter_penalty = args.lambda_adapter * adapter.penalty()
            total_loss = loss + adapter_penalty
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_adapter_parameters(adapter), args.grad_clip)
            adapter_optimizer.step()
            row["loss"] = float(total_loss.detach().cpu())
            row["adapter"] = float(adapter_penalty.detach().cpu())
            for key, value in row.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"ce": f"{row['ce']:.3g}", "p": f"{row['preserve']:.3g}", "g": f"{row['guard']:.3g}", "d": f"{row['drop']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"{label} adapter epoch {epoch} saw zero batches.")
        adapter_trace.append({key: value / float(batches) for key, value in totals.items()} | {"epoch": float(epoch)})

    freeze_model(wrapped)
    wrapped.eval()
    consolidated = clone_core(core, checkpoint=checkpoint, device=device)
    set_only_native_weights_trainable(consolidated)
    optimizer = make_optimizer_for_model(args, trainable_weight_parameters(consolidated))
    generator.manual_seed(seed + 1)
    consolidation_trace: list[dict[str, float]] = []
    for epoch in range(1, args.consolidation_epochs + 1):
        consolidated.train()
        permutation = torch.randperm(len(stage_examples), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0, "preserve": 0.0, "guard": 0.0, "drop": 0.0, "adapter": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(stage_examples), args.batch_size), desc=f"{label} consolidate {epoch}/{args.consolidation_epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, row = controlled_loss(
                args=args,
                model=consolidated,
                train_inputs=inputs,
                train_targets=targets,
                train_mask=mask,
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
                guard_examples=guard_examples,
                guard_logits=guard_logits,
                drop_examples=drop_examples,
                pad_id=pad_id,
                device=device,
                generator=generator,
            )
            current_logits = consolidated(inputs)
            with torch.no_grad():
                adapter_logits = wrapped(inputs)
            adapter_loss = distillation_kl(current_logits, adapter_logits, temperature=args.distill_temperature)
            total_loss = loss + args.lambda_adapter_distill * adapter_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(consolidated), args.grad_clip)
            optimizer.step()
            row["loss"] = float(total_loss.detach().cpu())
            row["adapter"] = float(adapter_loss.detach().cpu())
            for key, value in row.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"ce": f"{row['ce']:.3g}", "p": f"{row['preserve']:.3g}", "g": f"{row['guard']:.3g}", "d": f"{row['drop']:.3g}", "a": f"{row['adapter']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"{label} consolidation epoch {epoch} saw zero batches.")
        consolidation_trace.append({key: value / float(batches) for key, value in totals.items()} | {"epoch": float(epoch)})
    return consolidated, {"adapter": adapter_trace, "consolidation": consolidation_trace}


def collect_labeled_logits(
    *,
    model: nn.Module,
    examples: list[EncodedExample],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collect logits for an empty labeled memory.")
    return collect_example_logits(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)


def evaluate_named_groups(
    *,
    model: nn.Module,
    groups: dict[str, list[EncodedExample]],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    selected = {
        "preserve": groups["preserve"],
        "drop": groups["drop"],
        "neutral": groups["neutral"],
        "stage2": groups["stage2"],
        "stage3": groups["stage3"],
        "eval_all": groups["eval_all"],
    }
    return {
        name: evaluate_examples(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)["overall"]
        for name, examples in selected.items()
    }


def run_architecture_method(
    *,
    args: argparse.Namespace,
    method: str,
    checkpoint: dict[str, Any],
    groups: dict[str, list[EncodedExample]],
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if method not in {"naive", "direct", "adapter", "joint"}:
        raise ValueError(f"Unknown method: {method!r}.")
    model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed + {"joint": 10, "naive": 20, "direct": 30, "adapter": 40}[method])
    traces: list[dict[str, Any]] = []
    if method == "joint":
        joint_examples = groups["stage1"] + groups["stage2"] + groups["stage3"]
        trace = train_bootstrap_stage(
            args=args,
            model=model,
            stage_examples=joint_examples,
            pad_id=pad_id,
            device=device,
            epochs=args.joint_epochs,
            seed=args.seed + 101,
            label=f"{method} joint",
        )
        traces.append({"stage": "joint", "trace": trace})
        return {"method": method, "metrics": evaluate_named_groups(model=model, groups=groups, pad_id=pad_id, batch_size=args.eval_batch_size, device=device), "traces": traces}

    bootstrap_trace = train_bootstrap_stage(
        args=args,
        model=model,
        stage_examples=groups["stage1"],
        pad_id=pad_id,
        device=device,
        epochs=args.stage1_epochs,
        seed=args.seed + 201,
        label=f"{method} stage1",
    )
    traces.append({"stage": 1, "mode": "bootstrap", "trace": bootstrap_trace})

    preserve_examples = cap_examples(groups["preserve"], budget=args.preserve_budget)
    drop_examples = cap_examples(groups["drop"], budget=args.drop_budget)
    guard_examples = cap_examples(groups["neutral"], budget=args.guard_budget)

    for stage_index, stage_name in enumerate(["stage2", "stage3"], start=2):
        stage_examples = groups[stage_name]
        preserve_logits = collect_labeled_logits(
            model=model,
            examples=preserve_examples,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        guard_logits = collect_labeled_logits(
            model=model,
            examples=guard_examples,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        if method == "naive":
            trace = train_bootstrap_stage(
                args=args,
                model=model,
                stage_examples=stage_examples,
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 300 + stage_index,
                label=f"{method} {stage_name}",
            )
            traces.append({"stage": stage_index, "mode": "naive", "trace": trace})
        elif method == "direct":
            trace = train_direct_stage(
                args=args,
                model=model,
                stage_examples=stage_examples,
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
                guard_examples=guard_examples,
                guard_logits=guard_logits,
                drop_examples=drop_examples,
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 400 + stage_index,
                label=f"{method} {stage_name}",
            )
            traces.append({"stage": stage_index, "mode": "direct", "trace": trace})
        else:
            model, trace_bundle = train_adapter_controlled_stage(
                args=args,
                core=model,
                checkpoint=checkpoint,
                stage_examples=stage_examples,
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
                guard_examples=guard_examples,
                guard_logits=guard_logits,
                drop_examples=drop_examples,
                pad_id=pad_id,
                device=device,
                seed=args.seed + 500 + stage_index,
                label=f"{method} {stage_name}",
            )
            traces.append({"stage": stage_index, "mode": "adapter_consolidate", "trace": trace_bundle})
        if args.add_learned_stage_to_guard:
            guard_examples = cap_examples(guard_examples + stage_examples, budget=args.guard_budget)

    metrics = evaluate_named_groups(model=model, groups=groups, pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
    return {"method": method, "metrics": metrics, "traces": traces}


def validate_args(args: argparse.Namespace) -> None:
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer does not exist: {args.tokenizer_path}")
    positive_int("stage1_epochs", args.stage1_epochs)
    positive_int("stage_epochs", args.stage_epochs)
    positive_int("joint_epochs", args.joint_epochs)
    positive_int("adapter_epochs", args.adapter_epochs)
    positive_int("consolidation_epochs", args.consolidation_epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("preserve_budget", args.preserve_budget)
    positive_int("guard_budget", args.guard_budget)
    positive_int("drop_budget", args.drop_budget)
    positive_int("adapter_rank", args.adapter_rank)
    positive_float("adapter_scale", args.adapter_scale)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    nonnegative_float("lambda_preserve", args.lambda_preserve)
    nonnegative_float("lambda_guard", args.lambda_guard)
    nonnegative_float("drop_loss_weight", args.drop_loss_weight)
    positive_float("drop_target_probability", args.drop_target_probability)
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")
    nonnegative_float("lambda_adapter", args.lambda_adapter)
    nonnegative_float("lambda_adapter_distill", args.lambda_adapter_distill)
    positive_float("distill_temperature", args.distill_temperature)
    positive_float("grad_clip", args.grad_clip)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer must define [PAD].")
    preserve_people = parse_people(args.preserve_people, name="preserve_people")
    drop_people = parse_people(args.drop_people, name="drop_people")
    composition_holdout_people = parse_people(args.composition_holdout_people, name="composition_holdout_people")
    validate_policy(
        preserve_people=preserve_people,
        drop_people=drop_people,
        composition_holdout_people=composition_holdout_people,
    )
    _config_model, checkpoint = load_checkpoint(args.config_checkpoint, device)
    max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    raw_groups = build_raw_architecture_data(
        preserve_people=preserve_people,
        drop_people=drop_people,
        composition_holdout_people=composition_holdout_people,
        include_composition_rules=args.include_composition_rules,
    )
    groups = encode_groups(raw_groups, tokenizer, max_seq_len=max_seq_len)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")

    print("TINY RECURSIVE CONTROLLED CL ARCHITECTURE")
    print("=" * 112)
    print(
        f"device={device} methods={methods} preserve={sorted(preserve_people)} drop={sorted(drop_people)} "
        f"composition_holdout={sorted(composition_holdout_people)}"
    )
    print(
        f"examples stage1={len(groups['stage1'])} stage2={len(groups['stage2'])} "
        f"stage3={len(groups['stage3'])} eval={len(groups['eval_all'])}"
    )

    results: list[dict[str, Any]] = []
    for method in methods:
        print("\n" + "-" * 112)
        print(f"method={method}")
        results.append(
            run_architecture_method(
                args=args,
                method=method,
                checkpoint=checkpoint,
                groups=groups,
                pad_id=pad_id,
                device=device,
            )
        )

    summary = {
        "question": "Can a recursive CL architecture learn new stages, preserve selected behavior, guard neutral behavior, and drop selected behavior?",
        "policy": {
            "preserve_people": sorted(preserve_people),
            "drop_people": sorted(drop_people),
            "composition_holdout_people": sorted(composition_holdout_people),
            "add_learned_stage_to_guard": args.add_learned_stage_to_guard,
        },
        "model_config": checkpoint["model_config"],
        "hyperparameters": {
            "stage1_epochs": args.stage1_epochs,
            "stage_epochs": args.stage_epochs,
            "joint_epochs": args.joint_epochs,
            "adapter_epochs": args.adapter_epochs,
            "consolidation_epochs": args.consolidation_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_preserve": args.lambda_preserve,
            "lambda_guard": args.lambda_guard,
            "drop_loss_weight": args.drop_loss_weight,
            "drop_target_probability": args.drop_target_probability,
            "adapter_rank": args.adapter_rank,
            "adapter_scale": args.adapter_scale,
            "lambda_adapter": args.lambda_adapter,
            "lambda_adapter_distill": args.lambda_adapter_distill,
        },
        "raw_groups": {
            name: [asdict(example) for example in examples]
            for name, examples in raw_groups.items()
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTINY RECURSIVE CONTROLLED CL SUMMARY")
    print("=" * 112)
    print(
        "{:>10} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "method",
            "preserve",
            "drop",
            "neutral",
            "stage2",
            "stage3",
            "eval_all",
        )
    )
    print(
        "{:>10} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "",
            "loss/exact",
            "loss/exact",
            "loss/exact",
            "loss/exact",
            "loss/exact",
            "loss/exact",
        )
    )

    def compact(row: dict[str, float]) -> str:
        return "{:.3g}/{:.3f}".format(row["loss"], row["exact_match"])

    for result in results:
        metrics = result["metrics"]
        print(
            "{:>10} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
                result["method"],
                compact(metrics["preserve"]),
                compact(metrics["drop"]),
                compact(metrics["neutral"]),
                compact(metrics["stage2"]),
                compact(metrics["stage3"]),
                compact(metrics["eval_all"]),
            )
        )
    print(f"wrote_json={args.output_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-recursive-controlled-architecture-seed0.json"))
    parser.add_argument("--methods", type=str, default="naive,direct,adapter")
    parser.add_argument("--preserve-people", type=str, default="Alice,Bruno")
    parser.add_argument("--drop-people", type=str, default="Clara,Darin")
    parser.add_argument("--composition-holdout-people", type=str, default="Kira,Luca")
    parser.add_argument("--include-composition-rules", action="store_true")
    parser.add_argument("--add-learned-stage-to-guard", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=300)
    parser.add_argument("--stage-epochs", type=int, default=300)
    parser.add_argument("--joint-epochs", type=int, default=900)
    parser.add_argument("--adapter-epochs", type=int, default=250)
    parser.add_argument("--consolidation-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--preserve-budget", type=int, default=4)
    parser.add_argument("--guard-budget", type=int, default=16)
    parser.add_argument("--drop-budget", type=int, default=4)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-guard", type=float, default=1.0)
    parser.add_argument("--drop-loss-weight", type=float, default=0.1)
    parser.add_argument("--drop-target-probability", type=float, default=0.05)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--lambda-adapter", type=float, default=1e-4)
    parser.add_argument("--lambda-adapter-distill", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
