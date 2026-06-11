"""Reusable-computation continual-learning benchmark.

This controlled benchmark asks whether a CL method preserves and reuses
computations, not just old token strings. The data is deliberately structured:

Stage 1 learns person -> object possession relations.
Stage 2 learns the same relation type for new people and objects.
Stage 3 learns object -> place affordance relations.

Evaluation checks:

    old direct facts
    old reverse facts
    new same-relation reuse
    object -> place facts
    person -> place composition

The model is trained on prompt/answer examples with answer-token loss only.
This is a controlled diagnostic; it is not intended to be a natural text
benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
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
from experiments.gco_math.gco_native_scratch_transformer import GCONativeTransformer, NativeGCOConfig
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    distillation_kl,
    load_checkpoint,
    old_margin_loss,
    set_only_native_weights_trainable,
    target_margins_from_logits,
    trainable_weight_parameters,
)
from experiments.gco_math.gco_tiny_cl_bridge_adapter_consolidation import (
    AdapterWrappedTransformer,
    FinalResidualAdapter,
    freeze_model,
    trainable_adapter_parameters,
)


@dataclass(frozen=True)
class RelationItem:
    person: str
    obj: str
    place: str
    stage: int


@dataclass(frozen=True)
class QAExample:
    stage: int
    category: str
    prompt: str
    answer: str

    @property
    def text(self) -> str:
        return self.prompt + self.answer


@dataclass(frozen=True)
class EncodedExample:
    stage: int
    category: str
    prompt: str
    answer: str
    input_ids: list[int]
    target_ids: list[int]
    loss_mask: list[float]


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


def relation_items() -> list[RelationItem]:
    return [
        RelationItem("Alice", "copper key", "tower", 1),
        RelationItem("Bruno", "red lantern", "tunnel", 1),
        RelationItem("Clara", "blue map", "river", 1),
        RelationItem("Darin", "silver coin", "vault", 1),
        RelationItem("Elena", "green rope", "bridge", 1),
        RelationItem("Farah", "black compass", "forest", 1),
        RelationItem("Galen", "white shell", "harbor", 2),
        RelationItem("Hana", "yellow ring", "garden", 2),
        RelationItem("Iris", "purple flute", "cavern", 2),
        RelationItem("Jules", "orange book", "library", 2),
        RelationItem("Kira", "bronze bell", "chapel", 2),
        RelationItem("Luca", "crystal lens", "observatory", 2),
    ]


def possession_examples(item: RelationItem) -> list[QAExample]:
    return [
        QAExample(
            stage=item.stage,
            category=f"stage{item.stage}_possession_direct",
            prompt=f"Question: What does {item.person} carry? Answer:",
            answer=f" {item.obj}.",
        ),
        QAExample(
            stage=item.stage,
            category=f"stage{item.stage}_possession_reverse",
            prompt=f"Question: Who carries {item.obj}? Answer:",
            answer=f" {item.person}.",
        ),
    ]


def affordance_examples(item: RelationItem) -> list[QAExample]:
    return [
        QAExample(
            stage=3,
            category="stage3_object_place_direct",
            prompt=f"Question: What does {item.obj} open? Answer:",
            answer=f" {item.place}.",
        ),
        QAExample(
            stage=3,
            category="stage3_object_place_reverse",
            prompt=f"Question: What opens the {item.place}? Answer:",
            answer=f" {item.obj}.",
        ),
    ]


def composition_example(item: RelationItem, *, trained: bool) -> QAExample:
    category = "composition_seen" if trained else "composition_heldout"
    return QAExample(
        stage=3,
        category=category,
        prompt=f"Question: What can {item.person} open? Answer:",
        answer=f" {item.place}.",
    )


def composition_rule_examples() -> list[QAExample]:
    return [
        QAExample(
            stage=3,
            category="composition_rule",
            prompt="Rule: person carries object. object opens place. person opens what? Answer:",
            answer=" the place.",
        ),
        QAExample(
            stage=3,
            category="composition_rule",
            prompt="Rule: carrier of opener can open what? Answer:",
            answer=" the place.",
        ),
    ]


def composition_chain_examples(item: RelationItem) -> list[QAExample]:
    return [
        QAExample(
            stage=3,
            category="composition_seen_chain",
            prompt=(
                f"{item.person} / {item.obj} / {item.place}. "
                f"{item.person} opens what? Answer:"
            ),
            answer=f" {item.place}.",
        ),
        QAExample(
            stage=3,
            category="composition_seen_chain",
            prompt=(
                f"{item.obj} / {item.place} / {item.person}. "
                f"{item.person} opens what? Answer:"
            ),
            answer=f" {item.place}.",
        ),
    ]


def build_examples(
    *,
    composition_holdout_people: set[str],
    include_composition_rules: bool,
) -> tuple[list[list[QAExample]], list[QAExample]]:
    if not composition_holdout_people:
        raise ValueError("composition_holdout_people must not be empty.")
    items = relation_items()
    known_people = {item.person for item in items}
    unknown = composition_holdout_people.difference(known_people)
    if unknown:
        raise ValueError(f"Unknown composition holdout people: {sorted(unknown)}.")
    stage1 = [example for item in items if item.stage == 1 for example in possession_examples(item)]
    stage2 = [example for item in items if item.stage == 2 for example in possession_examples(item)]
    composition_train_people = known_people.difference(composition_holdout_people)
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
    eval_examples: list[QAExample] = []
    for item in items:
        eval_examples.extend(possession_examples(item))
        eval_examples.extend(affordance_examples(item))
        eval_examples.append(composition_example(item, trained=item.person in composition_train_people))
    return [stage1, stage2, stage3], eval_examples


def encode_example(example: QAExample, tokenizer: Tokenizer, *, max_seq_len: int) -> EncodedExample:
    encoding = tokenizer.encode(example.text)
    ids = encoding.ids
    offsets = encoding.offsets
    if len(ids) < 2:
        raise ValueError(f"Encoded example too short: {example.text!r}")
    if len(ids) - 1 > max_seq_len:
        raise ValueError(
            f"Example exceeds max_seq_len={max_seq_len}: token_count={len(ids) - 1}, text={example.text!r}"
        )
    answer_start = len(example.prompt)
    input_ids = ids[:-1]
    target_ids = ids[1:]
    target_offsets = offsets[1:]
    loss_mask = [1.0 if start >= answer_start else 0.0 for start, _end in target_offsets]
    if sum(loss_mask) <= 0.0:
        raise RuntimeError(f"Example produced no answer-token labels: {example.text!r}")
    return EncodedExample(
        stage=example.stage,
        category=example.category,
        prompt=example.prompt,
        answer=example.answer,
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
    )


def encode_examples(examples: list[QAExample], tokenizer: Tokenizer, *, max_seq_len: int) -> list[EncodedExample]:
    return [encode_example(example, tokenizer, max_seq_len=max_seq_len) for example in examples]


def batch_examples(
    examples: list[EncodedExample],
    *,
    indices: torch.Tensor,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[EncodedExample]]:
    selected = [examples[int(index)] for index in indices.tolist()]
    max_len = max(len(example.input_ids) for example in selected)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    masks: list[list[float]] = []
    for example in selected:
        pad_count = max_len - len(example.input_ids)
        inputs.append(example.input_ids + [pad_id] * pad_count)
        targets.append(example.target_ids + [pad_id] * pad_count)
        masks.append(example.loss_mask + [0.0] * pad_count)
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.float32, device=device),
        selected,
    )


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError(f"Shape mismatch: logits={logits.shape} targets={targets.shape} mask={mask.shape}.")
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
    denom = mask.sum().clamp_min(1.0)
    return (losses * mask).sum() / denom


@torch.no_grad()
def collect_example_logits(
    model: nn.Module,
    examples: list[EncodedExample],
    *,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    positive_int("batch_size", batch_size)
    model.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(examples)), dtype=torch.long)
        inputs, _targets, _mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
        logits = model(inputs).detach().cpu()
        for row_index, example in enumerate(selected):
            rows.append(logits[row_index, : len(example.target_ids)].clone())
    if len(rows) != len(examples):
        raise RuntimeError(f"Collected {len(rows)} logits for {len(examples)} examples.")
    return rows


def distillation_loss_for_examples(
    logits: torch.Tensor,
    selected: list[EncodedExample],
    teacher_logits: list[torch.Tensor],
    global_indices: torch.Tensor,
    *,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for row_index, example in enumerate(selected):
        length = len(example.target_ids)
        current = logits[row_index, :length].unsqueeze(0)
        teacher = teacher_logits[int(global_indices[row_index].item())].to(device).unsqueeze(0)
        losses.append(distillation_kl(current, teacher, temperature=temperature))
    if not losses:
        raise RuntimeError("No distillation losses were built.")
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_examples(
    model: nn.Module,
    examples: list[EncodedExample],
    *,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    positive_int("batch_size", batch_size)
    model.eval()
    totals: dict[str, dict[str, float]] = {}
    for start in range(0, len(examples), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(examples)), dtype=torch.long)
        inputs, targets, mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
        logits = model(inputs)
        predictions = logits.argmax(dim=-1)
        token_correct = ((predictions == targets).to(torch.float32) * mask).detach().cpu()
        mask_cpu = mask.detach().cpu()
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
        losses_cpu = (losses.detach().cpu() * mask_cpu)
        for row_index, example in enumerate(selected):
            category = example.category
            if category not in totals:
                totals[category] = {
                    "loss_sum": 0.0,
                    "token_count": 0.0,
                    "token_correct": 0.0,
                    "example_count": 0.0,
                    "exact_count": 0.0,
                }
            answer_count = float(mask_cpu[row_index].sum().item())
            correct_count = float(token_correct[row_index].sum().item())
            exact = 1.0 if answer_count > 0.0 and correct_count == answer_count else 0.0
            totals[category]["loss_sum"] += float(losses_cpu[row_index].sum().item())
            totals[category]["token_count"] += answer_count
            totals[category]["token_correct"] += correct_count
            totals[category]["example_count"] += 1.0
            totals[category]["exact_count"] += exact
    result: dict[str, Any] = {}
    total_loss = 0.0
    total_tokens = 0.0
    total_correct = 0.0
    total_examples = 0.0
    total_exact = 0.0
    for category, row in sorted(totals.items()):
        if row["token_count"] <= 0.0:
            raise RuntimeError(f"Category {category} has zero answer tokens.")
        result[category] = {
            "loss": row["loss_sum"] / row["token_count"],
            "token_accuracy": row["token_correct"] / row["token_count"],
            "exact_match": row["exact_count"] / row["example_count"],
            "example_count": row["example_count"],
        }
        total_loss += row["loss_sum"]
        total_tokens += row["token_count"]
        total_correct += row["token_correct"]
        total_examples += row["example_count"]
        total_exact += row["exact_count"]
    if total_tokens <= 0.0 or total_examples <= 0.0:
        raise RuntimeError("Evaluation saw no answer tokens/examples.")
    result["overall"] = {
        "loss": total_loss / total_tokens,
        "token_accuracy": total_correct / total_tokens,
        "exact_match": total_exact / total_examples,
        "example_count": total_examples,
    }
    return result


def make_model_from_config(
    *,
    checkpoint: dict[str, Any],
    device: torch.device,
    seed: int,
) -> GCONativeTransformer:
    torch.manual_seed(seed)
    cfg = NativeGCOConfig(**checkpoint["native_gco_config"])
    model_config = checkpoint["model_config"]
    model = GCONativeTransformer(
        vocab_size=int(model_config["vocab_size"]),
        d_model=int(model_config["d_model"]),
        n_layers=int(model_config["n_layers"]),
        n_heads=int(model_config["n_heads"]),
        d_ff=int(model_config["d_ff"]),
        max_seq_len=int(model_config["max_seq_len"]),
        cfg=cfg,
    ).to(device)
    return model


def clone_model(model: GCONativeTransformer, *, checkpoint: dict[str, Any], device: torch.device) -> GCONativeTransformer:
    clone = make_model_from_config(checkpoint=checkpoint, device=device, seed=0)
    missing, unexpected = clone.load_state_dict(model.state_dict(), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Clone state mismatch: missing={missing}, unexpected={unexpected}.")
    return clone


def make_optimizer_for_model(args: argparse.Namespace, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer {args.optimizer!r}.")


def train_core(
    *,
    args: argparse.Namespace,
    model: GCONativeTransformer,
    train_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
    anchor_examples: list[EncodedExample] | None = None,
    anchor_logits: list[torch.Tensor] | None = None,
) -> list[dict[str, float]]:
    set_only_native_weights_trainable(model)
    optimizer = make_optimizer_for_model(args, trainable_weight_parameters(model))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    anchor_count = 0 if anchor_examples is None else len(anchor_examples)
    if (anchor_examples is None) != (anchor_logits is None):
        raise ValueError("anchor_examples and anchor_logits must both be set or both be None.")
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_examples), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0, "anchor": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(train_examples), args.batch_size), desc=f"core epoch {epoch}/{epochs}")
        for start in pbar:
            train_indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(train_examples, indices=train_indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            ce = masked_ce_loss(logits, targets, mask)
            if anchor_count > 0 and anchor_examples is not None and anchor_logits is not None:
                anchor_indices = torch.randint(
                    low=0,
                    high=anchor_count,
                    size=(int(train_indices.numel()),),
                    generator=generator,
                    device=torch.device("cpu"),
                )
                anchor_inputs, _anchor_targets, _anchor_mask, anchor_selected = batch_examples(
                    anchor_examples, indices=anchor_indices, pad_id=pad_id, device=device
                )
                anchor_current = model(anchor_inputs)
                anchor_loss = distillation_loss_for_examples(
                    anchor_current,
                    anchor_selected,
                    anchor_logits,
                    anchor_indices,
                    temperature=args.distill_temperature,
                    device=device,
                )
            else:
                anchor_loss = ce.new_zeros(())
            loss = ce + args.lambda_anchor * anchor_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(model), args.grad_clip)
            optimizer.step()
            values = {
                "loss": float(loss.detach().cpu()),
                "ce": float(ce.detach().cpu()),
                "anchor": float(anchor_loss.detach().cpu()),
            }
            for key, value in values.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"ce": f"{values['ce']:.3g}", "a": f"{values['anchor']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"Epoch {epoch} saw zero batches.")
        row = {key: value / float(batches) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        trace.append(row)
        print(
            "core epoch={:4d} loss={:.5f} ce={:.5f} anchor={:.5f}".format(
                epoch, row["loss"], row["ce"], row["anchor"]
            )
        )
    return trace


def select_anchor_examples(
    examples: list[EncodedExample],
    *,
    budget: int,
) -> list[EncodedExample]:
    positive_int("budget", budget)
    if budget > len(examples):
        raise ValueError(f"anchor budget={budget} exceeds example count={len(examples)}.")
    indices = torch.linspace(0, len(examples) - 1, steps=budget).round().to(dtype=torch.long).unique(sorted=True)
    if indices.numel() != budget:
        raise RuntimeError(f"Anchor selection returned {indices.numel()} examples for budget={budget}.")
    return [examples[int(index)] for index in indices.tolist()]


def refresh_anchors(
    *,
    model: nn.Module,
    old_anchor_examples: list[EncodedExample],
    new_stage_examples: list[EncodedExample],
    pad_id: int,
    budget: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[EncodedExample], list[torch.Tensor]]:
    combined = old_anchor_examples + new_stage_examples
    selected = select_anchor_examples(combined, budget=min(budget, len(combined)))
    logits = collect_example_logits(model, selected, pad_id=pad_id, batch_size=batch_size, device=device)
    return selected, logits


def train_rebase_stage(
    *,
    args: argparse.Namespace,
    core: GCONativeTransformer,
    checkpoint: dict[str, Any],
    stage_examples: list[EncodedExample],
    anchor_examples: list[EncodedExample],
    anchor_logits: list[torch.Tensor],
    pad_id: int,
    device: torch.device,
    seed: int,
) -> GCONativeTransformer:
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
    for epoch in range(1, args.adapter_epochs + 1):
        wrapped.train()
        permutation = torch.randperm(len(stage_examples), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0, "anchor": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(stage_examples), args.batch_size), desc=f"adapter epoch {epoch}/{args.adapter_epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            anchor_indices = torch.randint(
                low=0,
                high=len(anchor_examples),
                size=(int(indices.numel()),),
                generator=generator,
                device=torch.device("cpu"),
            )
            anchor_inputs, _anchor_targets, _anchor_mask, anchor_selected = batch_examples(
                anchor_examples, indices=anchor_indices, pad_id=pad_id, device=device
            )
            adapter_optimizer.zero_grad(set_to_none=True)
            logits = wrapped(inputs)
            ce = masked_ce_loss(logits, targets, mask)
            anchor_current = wrapped(anchor_inputs)
            anchor_loss = distillation_loss_for_examples(
                anchor_current,
                anchor_selected,
                anchor_logits,
                anchor_indices,
                temperature=args.distill_temperature,
                device=device,
            )
            loss = ce + args.lambda_anchor * anchor_loss + args.lambda_adapter * adapter.penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_adapter_parameters(adapter), args.grad_clip)
            adapter_optimizer.step()
            values = {"loss": float(loss.detach().cpu()), "ce": float(ce.detach().cpu()), "anchor": float(anchor_loss.detach().cpu())}
            for key, value in values.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"ce": f"{values['ce']:.3g}", "a": f"{values['anchor']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"Adapter epoch {epoch} saw zero batches.")
        print(
            "adapter epoch={:4d} loss={:.5f} ce={:.5f} anchor={:.5f}".format(
                epoch,
                totals["loss"] / batches,
                totals["ce"] / batches,
                totals["anchor"] / batches,
            )
        )

    consolidated = clone_model(core, checkpoint=checkpoint, device=device)
    set_only_native_weights_trainable(consolidated)
    optimizer = make_optimizer_for_model(args, trainable_weight_parameters(consolidated))
    freeze_model(wrapped)
    wrapped.eval()
    generator.manual_seed(seed + 1)
    for epoch in range(1, args.consolidation_epochs + 1):
        consolidated.train()
        permutation = torch.randperm(len(stage_examples), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0, "anchor": 0.0, "adapter": 0.0}
        batches = 0
        pbar = tqdm(
            range(0, len(stage_examples), args.batch_size),
            desc=f"consolidate epoch {epoch}/{args.consolidation_epochs}",
        )
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            anchor_indices = torch.randint(
                low=0,
                high=len(anchor_examples),
                size=(int(indices.numel()),),
                generator=generator,
                device=torch.device("cpu"),
            )
            anchor_inputs, _anchor_targets, _anchor_mask, anchor_selected = batch_examples(
                anchor_examples, indices=anchor_indices, pad_id=pad_id, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = consolidated(inputs)
            ce = masked_ce_loss(logits, targets, mask)
            with torch.no_grad():
                adapter_logits = wrapped(inputs)
            adapter_loss = distillation_kl(logits, adapter_logits, temperature=args.distill_temperature)
            anchor_current = consolidated(anchor_inputs)
            anchor_loss = distillation_loss_for_examples(
                anchor_current,
                anchor_selected,
                anchor_logits,
                anchor_indices,
                temperature=args.distill_temperature,
                device=device,
            )
            loss = ce + args.lambda_anchor * anchor_loss + args.lambda_adapter_distill * adapter_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(consolidated), args.grad_clip)
            optimizer.step()
            values = {
                "loss": float(loss.detach().cpu()),
                "ce": float(ce.detach().cpu()),
                "anchor": float(anchor_loss.detach().cpu()),
                "adapter": float(adapter_loss.detach().cpu()),
            }
            for key, value in values.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"ce": f"{values['ce']:.3g}", "a": f"{values['anchor']:.3g}", "d": f"{values['adapter']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"Consolidation epoch {epoch} saw zero batches.")
        print(
            "consolidate epoch={:4d} loss={:.5f} ce={:.5f} anchor={:.5f} adapter={:.5f}".format(
                epoch,
                totals["loss"] / batches,
                totals["ce"] / batches,
                totals["anchor"] / batches,
                totals["adapter"] / batches,
            )
        )
    return consolidated


def run_method(
    *,
    args: argparse.Namespace,
    method: str,
    checkpoint: dict[str, Any],
    stages: list[list[EncodedExample]],
    eval_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if method not in {"joint", "naive", "anchor", "rebase"}:
        raise ValueError(f"Unknown method: {method!r}.")
    method_seed_offsets = {"joint": 101, "naive": 202, "anchor": 303, "rebase": 404}
    model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed + method_seed_offsets[method])
    traces: list[dict[str, Any]] = []
    if method == "joint":
        train_examples = [example for stage in stages for example in stage]
        trace = train_core(
            args=args,
            model=model,
            train_examples=train_examples,
            pad_id=pad_id,
            device=device,
            epochs=args.joint_epochs,
            seed=args.seed + 11,
        )
        traces.append({"stage": "joint", "trace": trace})
    elif method == "naive":
        for stage_index, stage_examples in enumerate(stages, start=1):
            trace = train_core(
                args=args,
                model=model,
                train_examples=stage_examples,
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 100 * stage_index,
            )
            traces.append({"stage": stage_index, "trace": trace})
    elif method == "anchor":
        anchor_examples: list[EncodedExample] = []
        anchor_logits: list[torch.Tensor] = []
        for stage_index, stage_examples in enumerate(stages, start=1):
            trace = train_core(
                args=args,
                model=model,
                train_examples=stage_examples,
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 200 * stage_index,
                anchor_examples=anchor_examples if anchor_examples else None,
                anchor_logits=anchor_logits if anchor_logits else None,
            )
            anchor_examples, anchor_logits = refresh_anchors(
                model=model,
                old_anchor_examples=anchor_examples,
                new_stage_examples=stage_examples,
                pad_id=pad_id,
                budget=args.anchor_budget,
                batch_size=args.eval_batch_size,
                device=device,
            )
            traces.append({"stage": stage_index, "trace": trace, "anchor_count": len(anchor_examples)})
    else:
        anchor_examples = []
        anchor_logits = []
        for stage_index, stage_examples in enumerate(stages, start=1):
            if anchor_examples:
                model = train_rebase_stage(
                    args=args,
                    core=model,
                    checkpoint=checkpoint,
                    stage_examples=stage_examples,
                    anchor_examples=anchor_examples,
                    anchor_logits=anchor_logits,
                    pad_id=pad_id,
                    device=device,
                    seed=args.seed + 300 * stage_index,
                )
                traces.append({"stage": stage_index, "mode": "rebase", "anchor_count": len(anchor_examples)})
            else:
                trace = train_core(
                    args=args,
                    model=model,
                    train_examples=stage_examples,
                    pad_id=pad_id,
                    device=device,
                    epochs=args.stage_epochs,
                    seed=args.seed + 300 * stage_index,
                )
                traces.append({"stage": stage_index, "mode": "bootstrap", "trace": trace})
            anchor_examples, anchor_logits = refresh_anchors(
                model=model,
                old_anchor_examples=anchor_examples,
                new_stage_examples=stage_examples,
                pad_id=pad_id,
                budget=args.anchor_budget,
                batch_size=args.eval_batch_size,
                device=device,
            )
    metrics = evaluate_examples(model, eval_examples, pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
    return {"method": method, "metrics": metrics, "traces": traces}


def validate_args(args: argparse.Namespace) -> None:
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer does not exist: {args.tokenizer_path}")
    positive_int("stage_epochs", args.stage_epochs)
    positive_int("joint_epochs", args.joint_epochs)
    positive_int("adapter_epochs", args.adapter_epochs)
    positive_int("consolidation_epochs", args.consolidation_epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("anchor_budget", args.anchor_budget)
    positive_int("adapter_rank", args.adapter_rank)
    positive_float("adapter_scale", args.adapter_scale)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    nonnegative_float("lambda_anchor", args.lambda_anchor)
    nonnegative_float("lambda_adapter", args.lambda_adapter)
    nonnegative_float("lambda_adapter_distill", args.lambda_adapter_distill)
    positive_float("distill_temperature", args.distill_temperature)
    positive_float("grad_clip", args.grad_clip)
    holdouts = [item.strip() for item in args.composition_holdout_people.split(",") if item.strip()]
    if not holdouts:
        raise ValueError("--composition-holdout-people must contain at least one name.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer must define [PAD].")
    _config_model, checkpoint = load_checkpoint(args.config_checkpoint, device)
    max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    composition_holdouts = {item.strip() for item in args.composition_holdout_people.split(",") if item.strip()}
    raw_stages, raw_eval = build_examples(
        composition_holdout_people=composition_holdouts,
        include_composition_rules=args.include_composition_rules,
    )
    stages = [encode_examples(stage, tokenizer, max_seq_len=max_seq_len) for stage in raw_stages]
    eval_examples = encode_examples(raw_eval, tokenizer, max_seq_len=max_seq_len)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")

    print("TINY REUSABLE-COMPUTATION CL BENCHMARK")
    print("=" * 112)
    print(
        f"device={device} methods={methods} stages={[len(stage) for stage in stages]} "
        f"eval={len(eval_examples)} anchor_budget={args.anchor_budget}"
    )
    results: list[dict[str, Any]] = []
    for method in methods:
        print("\n" + "-" * 112)
        print(f"method={method}")
        result = run_method(
            args=args,
            method=method,
            checkpoint=checkpoint,
            stages=stages,
            eval_examples=eval_examples,
            pad_id=pad_id,
            device=device,
        )
        results.append(result)
        overall = result["metrics"]["overall"]
        print(
            "method={} overall loss={:.5f} token_acc={:.4f} exact={:.4f}".format(
                method,
                overall["loss"],
                overall["token_accuracy"],
                overall["exact_match"],
            )
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "question": "Does continual learning preserve and reuse computations rather than only preserving old outputs?",
        "methods": methods,
        "model_config": checkpoint["model_config"],
        "train_example_counts": [len(stage) for stage in stages],
        "eval_example_count": len(eval_examples),
        "composition_holdout_people": sorted(composition_holdouts),
        "include_composition_rules": args.include_composition_rules,
        "examples": {
            "train": [[asdict(example) for example in stage] for stage in raw_stages],
            "eval": [asdict(example) for example in raw_eval],
        },
        "hyperparameters": {
            "stage_epochs": args.stage_epochs,
            "joint_epochs": args.joint_epochs,
            "adapter_epochs": args.adapter_epochs,
            "consolidation_epochs": args.consolidation_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_anchor": args.lambda_anchor,
            "lambda_adapter": args.lambda_adapter,
            "lambda_adapter_distill": args.lambda_adapter_distill,
            "anchor_budget": args.anchor_budget,
        },
        "results": results,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    categories = sorted(
        key for result in results for key in result["metrics"].keys() if key != "overall"
    )
    categories = sorted(set(categories))
    print("\nTINY REUSABLE-COMPUTATION CL SUMMARY")
    print("=" * 112)
    print(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "method",
            "overall",
            "old",
            "reuse",
            "object",
            "comp_seen",
            "comp_hold",
        )
    )
    for result in results:
        metrics = result["metrics"]
        def category_average(prefix: str) -> float:
            scores = [row["exact_match"] for key, row in metrics.items() if key.startswith(prefix)]
            return sum(scores) / len(scores) if scores else 0.0

        print(
            "{:>10} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(
                result["method"],
                metrics["overall"]["exact_match"],
                category_average("stage1_possession"),
                category_average("stage2_possession"),
                category_average("stage3_object_place"),
                category_average("composition_seen"),
                category_average("composition_heldout"),
            )
        )
    print(f"wrote_json={args.output_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("model/analysis/gco-tiny-reusable-computation-cl-seed0.json"),
    )
    parser.add_argument("--methods", type=str, default="joint,naive,anchor,rebase")
    parser.add_argument("--composition-holdout-people", type=str, default="Clara,Darin,Elena,Farah,Iris,Jules,Kira,Luca")
    parser.add_argument("--include-composition-rules", action="store_true")
    parser.add_argument("--stage-epochs", type=int, default=250)
    parser.add_argument("--joint-epochs", type=int, default=750)
    parser.add_argument("--adapter-epochs", type=int, default=200)
    parser.add_argument("--consolidation-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--anchor-budget", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
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
