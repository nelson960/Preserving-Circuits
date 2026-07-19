#!/usr/bin/env python3
"""Semantic-closure continual update on a local pretrained Pythia model.

This experiment tests a narrower version of the current research idea:

    pretrained language model
    + small trainable hidden adapter
    + semantic-closure update packet
    + Invariant-Tangent constraints over sampled history/locality/rule behavior

The model is not expected to invent open-world truth.  The experiment asks a
smaller question: after a small foundation adapter learns a controlled semantic
world, does a correction update work better when the update packet includes the
direct correction plus its semantic consequences, history checks, locality
checks, and old-answer suppression?

The pretrained base is frozen.  Only the adapter is updated.  This keeps the
experiment laptop-safe and makes failure visible: if the adapter cannot learn
the semantic closure without damaging protected behavior, the run reports that
directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device


CITY_CANDIDATES = ("Paris", "Rome", "London", "Berlin", "Madrid")
ANSWER_CANDIDATES = CITY_CANDIDATES + ("blue", "green", "red", "yellow")


@dataclass(frozen=True)
class Entity:
    name: str
    old_city: str
    new_city: str
    color: str


@dataclass(frozen=True)
class QAItem:
    question: str
    answer: str
    group: str


@dataclass(frozen=True)
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    targets: torch.Tensor
    answer_mask: torch.Tensor

    def to(self, device: torch.device) -> "EncodedBatch":
        return EncodedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            targets=self.targets.to(device),
            answer_mask=self.answer_mask.to(device),
        )


@dataclass
class EvalRow:
    group: str
    total: int
    exact: float
    loss: float


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if value < 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")
    nonnegative_int("seed", int(args.seed))
    for name in (
        "adapter_rank",
        "foundation_epochs",
        "update_epochs",
        "constraint_limit",
        "max_seq_len",
    ):
        positive_int(name, int(getattr(args, name)))
    for name in (
        "foundation_lr",
        "update_lr",
        "projection_damping",
        "restore_strength",
        "restore_norm_ratio",
        "old_suppression_weight",
        "old_margin",
        "max_gradient_norm",
    ):
        positive_float(name, float(getattr(args, name)))
    nonnegative_float("adapter_scale", float(args.adapter_scale))
    if args.constraint_limit < 1:
        raise ValueError("--constraint-limit must be at least 1.")
    if args.pad_token_id < 0:
        raise ValueError("--pad-token-id must be non-negative.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_entities() -> tuple[Entity, ...]:
    return (
        Entity("Alice", "Paris", "Rome", "blue"),
        Entity("Bob", "London", "London", "green"),
        Entity("Clara", "Berlin", "Berlin", "red"),
        Entity("Diego", "Madrid", "Madrid", "yellow"),
    )


def qa(question: str, answer: str, group: str) -> QAItem:
    return QAItem(question=question, answer=answer, group=group)


def foundation_items(entities: Sequence[Entity]) -> tuple[QAItem, ...]:
    items: list[QAItem] = []
    for entity in entities:
        items.extend(
            [
                qa(f"Where does {entity.name} work now?", entity.old_city, "foundation_current"),
                qa(f"Which city is {entity.name} currently working in?", entity.old_city, "foundation_current"),
                qa(f"Where is {entity.name}'s office now?", entity.old_city, "foundation_ripple"),
                qa(f"Where did {entity.name} work before?", entity.old_city, "foundation_history"),
                qa(f"What color does {entity.name} like?", entity.color, "foundation_locality"),
            ]
        )
    for city in CITY_CANDIDATES:
        items.extend(
            [
                qa(f"If someone works in {city}, where is their office?", city, "foundation_rule"),
                qa(f"If a person currently works in {city}, which city is their office in?", city, "foundation_rule"),
            ]
        )
    return tuple(items)


def semantic_closure_items(edit: Entity, entities: Sequence[Entity]) -> tuple[QAItem, ...]:
    items = [
        qa(f"Where does {edit.name} work now?", edit.new_city, "direct"),
        qa(f"Which city is {edit.name} currently working in?", edit.new_city, "paraphrase"),
        qa(f"What is {edit.name}'s current work city?", edit.new_city, "paraphrase"),
        qa(f"Where is {edit.name}'s office now?", edit.new_city, "ripple"),
        qa(f"Where did {edit.name} work before?", edit.old_city, "history"),
    ]
    for entity in entities:
        if entity.name == edit.name:
            continue
        items.extend(
            [
                qa(f"Where does {entity.name} work now?", entity.old_city, "locality"),
                qa(f"Where is {entity.name}'s office now?", entity.old_city, "locality"),
                qa(f"What color does {entity.name} like?", entity.color, "locality"),
            ]
        )
    for city in CITY_CANDIDATES:
        items.append(qa(f"If someone works in {city}, where is their office?", city, "rule"))
    return tuple(items)


def direct_only_items(edit: Entity) -> tuple[QAItem, ...]:
    return (
        qa(f"Where does {edit.name} work now?", edit.new_city, "direct"),
        qa(f"Which city is {edit.name} currently working in?", edit.new_city, "paraphrase"),
    )


def protected_items(edit: Entity, entities: Sequence[Entity]) -> tuple[QAItem, ...]:
    protected: list[QAItem] = [
        qa(f"Where did {edit.name} work before?", edit.old_city, "history"),
    ]
    for entity in entities:
        if entity.name == edit.name:
            continue
        protected.extend(
            [
                qa(f"Where does {entity.name} work now?", entity.old_city, "locality"),
                qa(f"Where is {entity.name}'s office now?", entity.old_city, "locality"),
                qa(f"What color does {entity.name} like?", entity.color, "locality"),
            ]
        )
    for city in CITY_CANDIDATES:
        protected.append(qa(f"If someone works in {city}, where is their office?", city, "rule"))
    return tuple(protected)


def prompt_text(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def answer_text(answer: str) -> str:
    return " " + answer.strip()


def encode_items(
    tokenizer,
    items: Sequence[QAItem],
    *,
    pad_token_id: int,
    max_seq_len: int,
) -> EncodedBatch:
    if not items:
        raise ValueError("Cannot encode an empty item list.")
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[float]] = []

    for item in items:
        prompt_ids = tokenizer.encode(prompt_text(item.question), add_special_tokens=False)
        target_answer_ids = tokenizer.encode(answer_text(item.answer), add_special_tokens=False)
        if not prompt_ids:
            raise ValueError(f"Prompt encoded to zero tokens: {item.question!r}")
        if not target_answer_ids:
            raise ValueError(f"Answer encoded to zero tokens: {item.answer!r}")
        full_ids = prompt_ids + target_answer_ids
        if len(full_ids) > max_seq_len:
            raise ValueError(
                f"Example exceeds max_seq_len={max_seq_len}: question={item.question!r}, "
                f"answer={item.answer!r}, tokens={len(full_ids)}"
            )
        padded = full_ids + [pad_token_id] * (max_seq_len - len(full_ids))
        attention = [1] * len(full_ids) + [0] * (max_seq_len - len(full_ids))
        input_ids = padded[:-1]
        targets = padded[1:]
        mask = [0.0] * (max_seq_len - 1)
        answer_start = len(prompt_ids) - 1
        for offset in range(len(target_answer_ids)):
            mask[answer_start + offset] = 1.0
        input_rows.append(input_ids)
        attention_rows.append(attention[:-1])
        target_rows.append(targets)
        mask_rows.append(mask)

    return EncodedBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        attention_mask=torch.tensor(attention_rows, dtype=torch.long),
        targets=torch.tensor(target_rows, dtype=torch.long),
        answer_mask=torch.tensor(mask_rows, dtype=torch.float32),
    )


class HiddenAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, scale: float) -> None:
        super().__init__()
        positive_int("rank", rank)
        nonnegative_float("scale", scale)
        self.scale = scale
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(float(hidden_size)))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.scale * self.up(torch.tanh(self.down(hidden)))


class PythiaAdapterLM(nn.Module):
    def __init__(self, base_model: nn.Module, adapter: HiddenAdapter) -> None:
        super().__init__()
        if not hasattr(base_model, "gpt_neox"):
            raise TypeError("This experiment requires a GPT-NeoX/Pythia model with .gpt_neox.")
        if not hasattr(base_model, "embed_out"):
            raise TypeError("This experiment requires a GPT-NeoX/Pythia model with .embed_out.")
        self.base_model = base_model
        self.adapter = adapter
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.base_model.gpt_neox(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        adapted_hidden = self.adapter(hidden)
        logits = self.base_model.embed_out(adapted_hidden)
        return logits, adapted_hidden


def masked_cross_entropy(logits: torch.Tensor, batch: EncodedBatch) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.targets.reshape(-1),
        reduction="none",
    ).reshape_as(batch.targets)
    denom = batch.answer_mask.sum()
    if float(denom.detach().cpu()) <= 0.0:
        raise ValueError("Answer mask is empty.")
    return (losses * batch.answer_mask).sum() / denom


def flat_parameters(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    if not parameters:
        raise ValueError("No trainable parameters were provided.")
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def flat_gradient(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
    label: str,
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=False)
    flats: list[torch.Tensor] = []
    for grad in grads:
        if grad is None:
            raise RuntimeError(f"{label} produced an unused parameter gradient.")
        flats.append(grad.reshape(-1))
    flat = torch.cat(flats)
    if not torch.isfinite(flat).all():
        raise FloatingPointError(f"{label} produced non-finite gradients.")
    return flat


def assign_flat_gradient(parameters: Sequence[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].reshape_as(parameter).detach().clone()
        offset += count
    if offset != flat.numel():
        raise RuntimeError(f"Flat gradient size mismatch: consumed {offset}, total {flat.numel()}.")


def apply_sgd_step(parameters: Sequence[torch.nn.Parameter], flat_grad: torch.Tensor, lr: float) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(-lr * flat_grad[offset : offset + count].reshape_as(parameter))
            offset += count
    if offset != flat_grad.numel():
        raise RuntimeError(f"Flat step size mismatch: consumed {offset}, total {flat_grad.numel()}.")


def project_gradient(
    gradient: torch.Tensor,
    rows: torch.Tensor,
    *,
    damping: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if rows.ndim != 2:
        raise ValueError(f"rows must be rank-2, got shape={tuple(rows.shape)}.")
    if rows.shape[1] != gradient.numel():
        raise ValueError(f"Constraint row width {rows.shape[1]} does not match gradient size {gradient.numel()}.")
    if rows.shape[0] == 0:
        return gradient, {"rows": 0.0, "removed_fraction": 0.0, "safe_fraction": 1.0}

    device = gradient.device
    rows_cpu = rows.detach().to(device="cpu", dtype=torch.float32)
    gradient_cpu = gradient.detach().to(device="cpu", dtype=torch.float32)
    gram = rows_cpu @ rows_cpu.T
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    rhs = rows_cpu @ gradient_cpu
    coefficients = torch.linalg.solve(gram + damping * identity, rhs)
    damage_cpu = rows_cpu.T @ coefficients
    safe_cpu = gradient_cpu - damage_cpu
    raw_norm = torch.linalg.vector_norm(gradient_cpu).clamp_min(torch.finfo(torch.float32).eps)
    damage_norm = torch.linalg.vector_norm(damage_cpu)
    safe_norm = torch.linalg.vector_norm(safe_cpu)
    return safe_cpu.to(device=device, dtype=gradient.dtype), {
        "rows": float(rows.shape[0]),
        "removed_fraction": float((damage_norm / raw_norm).detach().cpu()),
        "safe_fraction": float((safe_norm / raw_norm).detach().cpu()),
    }


def clip_norm(flat: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, float]:
    norm = torch.linalg.vector_norm(flat)
    norm_value = float(norm.detach().cpu())
    if not math.isfinite(norm_value):
        raise FloatingPointError("Gradient norm is non-finite.")
    if norm_value <= max_norm:
        return flat, 1.0
    scale = max_norm / max(norm_value, torch.finfo(flat.dtype).eps)
    return flat * scale, float(scale)


def answer_logprob(
    model: PythiaAdapterLM,
    tokenizer,
    question: str,
    answer: str,
    *,
    pad_token_id: int,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    batch = encode_items(
        tokenizer,
        [QAItem(question=question, answer=answer, group="candidate")],
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
    ).to(device)
    logits, _hidden = model(batch.input_ids, batch.attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, batch.targets.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * batch.answer_mask).sum()


def old_suppression_loss(
    model: PythiaAdapterLM,
    tokenizer,
    edit: Entity,
    *,
    pad_token_id: int,
    max_seq_len: int,
    margin: float,
    device: torch.device,
) -> torch.Tensor:
    questions = (
        f"Where does {edit.name} work now?",
        f"Which city is {edit.name} currently working in?",
        f"Where is {edit.name}'s office now?",
    )
    losses: list[torch.Tensor] = []
    for question in questions:
        old_score = answer_logprob(
            model,
            tokenizer,
            question,
            edit.old_city,
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
            device=device,
        )
        new_score = answer_logprob(
            model,
            tokenizer,
            question,
            edit.new_city,
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
            device=device,
        )
        losses.append(F.relu(old_score - new_score + margin))
    return torch.stack(losses).mean()


def train_foundation(
    model: PythiaAdapterLM,
    tokenizer,
    items: Sequence[QAItem],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> list[dict[str, float]]:
    batch = encode_items(
        tokenizer,
        items,
        pad_token_id=args.pad_token_id,
        max_seq_len=args.max_seq_len,
    ).to(device)
    parameters = list(model.adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.foundation_lr)
    trace: list[dict[str, float]] = []
    iterator: Iterable[int] = range(1, args.foundation_epochs + 1)
    if args.progress:
        iterator = tqdm(iterator, desc="foundation", leave=False)
    for epoch in iterator:
        optimizer.zero_grad(set_to_none=True)
        logits, _hidden = model(batch.input_ids, batch.attention_mask)
        loss = masked_cross_entropy(logits, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Foundation loss became non-finite at epoch {epoch}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.max_gradient_norm)
        optimizer.step()
        if epoch == 1 or epoch == args.foundation_epochs or epoch % args.print_every == 0:
            trace.append({"epoch": float(epoch), "loss": float(loss.detach().cpu())})
    return trace


def constraint_rows_for_items(
    model: PythiaAdapterLM,
    tokenizer,
    items: Sequence[QAItem],
    raw_gradient: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float | str]]]:
    scored_rows: list[tuple[float, torch.Tensor, dict[str, float | str]]] = []
    for index, item in enumerate(items):
        batch = encode_items(
            tokenizer,
            [item],
            pad_token_id=args.pad_token_id,
            max_seq_len=args.max_seq_len,
        ).to(device)
        logits, _hidden = model(batch.input_ids, batch.attention_mask)
        loss = masked_cross_entropy(logits, batch)
        row = flat_gradient(loss, parameters, retain_graph=False, label=f"constraint_{index}")
        row_norm = torch.linalg.vector_norm(row)
        row_norm_value = float(row_norm.detach().cpu())
        if row_norm_value <= 1e-12 or not math.isfinite(row_norm_value):
            raise FloatingPointError(f"Constraint row {index} has invalid norm {row_norm_value}.")
        unit = row / row_norm
        damage = abs(float(torch.dot(unit.detach(), raw_gradient.detach()).cpu()))
        scored_rows.append(
            (
                damage,
                unit.detach(),
                {
                    "question": item.question,
                    "answer": item.answer,
                    "group": item.group,
                    "predicted_damage": damage,
                    "row_norm": row_norm_value,
                },
            )
        )
    scored_rows.sort(key=lambda row: row[0], reverse=True)
    selected = scored_rows[: args.constraint_limit]
    if not selected:
        raise RuntimeError("No constraint rows were selected.")
    rows = torch.stack([row for _damage, row, _meta in selected], dim=0)
    metadata = [meta for _damage, _row, meta in selected]
    return rows, metadata


def train_update_mode(
    model: PythiaAdapterLM,
    tokenizer,
    edit: Entity,
    update_items: Sequence[QAItem],
    protected: Sequence[QAItem],
    args: argparse.Namespace,
    *,
    device: torch.device,
    constrained: bool,
) -> tuple[list[dict[str, float]], list[dict[str, float | str]]]:
    update_batch = encode_items(
        tokenizer,
        update_items,
        pad_token_id=args.pad_token_id,
        max_seq_len=args.max_seq_len,
    ).to(device)
    protected_batch = None
    if constrained:
        protected_batch = encode_items(
            tokenizer,
            protected,
            pad_token_id=args.pad_token_id,
            max_seq_len=args.max_seq_len,
        ).to(device)
    parameters = list(model.adapter.parameters())
    trace: list[dict[str, float]] = []
    selected_metadata: list[dict[str, float | str]] = []

    iterator: Iterable[int] = range(1, args.update_epochs + 1)
    if args.progress:
        iterator = tqdm(iterator, desc="semantic_update", leave=False)
    for epoch in iterator:
        logits, _hidden = model(update_batch.input_ids, update_batch.attention_mask)
        packet_loss = masked_cross_entropy(logits, update_batch)
        suppress = old_suppression_loss(
            model,
            tokenizer,
            edit,
            pad_token_id=args.pad_token_id,
            max_seq_len=args.max_seq_len,
            margin=args.old_margin,
            device=device,
        )
        raw_loss = packet_loss + args.old_suppression_weight * suppress
        raw_gradient = flat_gradient(raw_loss, parameters, retain_graph=False, label=f"raw_update_{epoch}")

        restore_loss_value = float("nan")
        if constrained:
            rows, metadata = constraint_rows_for_items(
                model,
                tokenizer,
                protected,
                raw_gradient,
                parameters,
                args,
                device=device,
            )
            selected_metadata = metadata
            tangent, projection = project_gradient(
                raw_gradient,
                rows,
                damping=args.projection_damping,
            )
            if protected_batch is None:
                raise RuntimeError("constrained=True requires a protected batch.")
            protected_logits, _protected_hidden = model(protected_batch.input_ids, protected_batch.attention_mask)
            restore_loss = masked_cross_entropy(protected_logits, protected_batch)
            restore_loss_value = float(restore_loss.detach().cpu())
            restore_gradient = flat_gradient(restore_loss, parameters, retain_graph=False, label=f"restore_{epoch}")
            restore_norm = torch.linalg.vector_norm(restore_gradient)
            tangent_norm = torch.linalg.vector_norm(tangent)
            if float(restore_norm.detach().cpu()) > 0.0:
                limit = args.restore_norm_ratio * tangent_norm
                restore_scale = torch.minimum(
                    torch.ones((), device=device, dtype=restore_gradient.dtype),
                    limit / (args.restore_strength * restore_norm + torch.finfo(restore_gradient.dtype).eps),
                )
                restore_gradient = restore_gradient * restore_scale
            final_gradient = tangent + args.restore_strength * restore_gradient
        else:
            projection = {"safe_fraction": 1.0, "removed_fraction": 0.0}
            final_gradient = raw_gradient
        final_gradient, clip_scale = clip_norm(final_gradient, args.max_gradient_norm)
        apply_sgd_step(parameters, final_gradient, args.update_lr)

        if epoch == 1 or epoch == args.update_epochs or epoch % args.print_every == 0:
            trace.append(
                {
                    "epoch": float(epoch),
                    "packet_loss": float(packet_loss.detach().cpu()),
                    "suppression_loss": float(suppress.detach().cpu()),
                    "restore_loss": restore_loss_value,
                    "safe_fraction": projection["safe_fraction"],
                    "removed_fraction": projection["removed_fraction"],
                    "clip_scale": clip_scale,
                }
            )
    return trace, selected_metadata


@torch.no_grad()
def evaluate_items(
    model: PythiaAdapterLM,
    tokenizer,
    items: Sequence[QAItem],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[list[EvalRow], list[dict[str, str | float]]]:
    if not items:
        raise ValueError("Cannot evaluate an empty item list.")
    by_group: dict[str, list[QAItem]] = {}
    predictions: list[dict[str, str | float]] = []
    for item in items:
        by_group.setdefault(item.group, []).append(item)
        scores: dict[str, float] = {}
        for candidate in ANSWER_CANDIDATES:
            score = answer_logprob(
                model,
                tokenizer,
                item.question,
                candidate,
                pad_token_id=args.pad_token_id,
                max_seq_len=args.max_seq_len,
                device=device,
            )
            scores[candidate] = float(score.detach().cpu())
        predicted = max(scores, key=scores.get)
        predictions.append(
            {
                "group": item.group,
                "question": item.question,
                "answer": item.answer,
                "predicted": predicted,
                "correct": float(predicted == item.answer),
                "best_score": scores[predicted],
                "target_score": scores[item.answer],
            }
        )
    rows: list[EvalRow] = []
    for group, group_items in sorted(by_group.items()):
        batch = encode_items(
            tokenizer,
            group_items,
            pad_token_id=args.pad_token_id,
            max_seq_len=args.max_seq_len,
        ).to(device)
        logits, _hidden = model(batch.input_ids, batch.attention_mask)
        loss = masked_cross_entropy(logits, batch)
        exacts = [entry["correct"] for entry in predictions if entry["group"] == group]
        rows.append(
            EvalRow(
                group=group,
                total=len(group_items),
                exact=float(sum(float(value) for value in exacts) / len(exacts)),
                loss=float(loss.detach().cpu()),
            )
        )
    return rows, predictions


@torch.no_grad()
def correction_margin(
    model: PythiaAdapterLM,
    tokenizer,
    edit: Entity,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> float:
    question = f"Where does {edit.name} work now?"
    new_score = answer_logprob(
        model,
        tokenizer,
        question,
        edit.new_city,
        pad_token_id=args.pad_token_id,
        max_seq_len=args.max_seq_len,
        device=device,
    )
    old_score = answer_logprob(
        model,
        tokenizer,
        question,
        edit.old_city,
        pad_token_id=args.pad_token_id,
        max_seq_len=args.max_seq_len,
        device=device,
    )
    return float((new_score - old_score).detach().cpu())


@torch.no_grad()
def adapted_hidden_matrix(
    model: PythiaAdapterLM,
    tokenizer,
    items: Sequence[QAItem],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> torch.Tensor:
    batch = encode_items(
        tokenizer,
        items,
        pad_token_id=args.pad_token_id,
        max_seq_len=args.max_seq_len,
    ).to(device)
    _logits, hidden = model(batch.input_ids, batch.attention_mask)
    mask = batch.answer_mask.bool()
    vectors: list[torch.Tensor] = []
    for row in range(hidden.shape[0]):
        indices = torch.nonzero(mask[row], as_tuple=False).flatten()
        if indices.numel() == 0:
            raise RuntimeError("Answer mask row is empty during hidden capture.")
        vectors.append(hidden[row, indices].mean(dim=0))
    return torch.stack(vectors, dim=0).detach()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
    y_cpu = y.detach().to(device="cpu", dtype=torch.float32)
    x_centered = x_cpu - x_cpu.mean(dim=0, keepdim=True)
    y_centered = y_cpu - y_cpu.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x_centered.T @ y_centered) ** 2
    x_norm = torch.linalg.matrix_norm(x_centered.T @ x_centered)
    y_norm = torch.linalg.matrix_norm(y_centered.T @ y_centered)
    denom = (x_norm * y_norm).clamp_min(torch.finfo(torch.float32).eps)
    return float((cross / denom).detach().cpu())


def clone_adapter_model(model: PythiaAdapterLM) -> PythiaAdapterLM:
    adapter = copy.deepcopy(model.adapter)
    clone = PythiaAdapterLM(model.base_model, adapter)
    return clone


def rows_to_dict(rows: Sequence[EvalRow]) -> dict[str, dict[str, float]]:
    return {
        row.group: {
            "total": float(row.total),
            "exact": row.exact,
            "loss": row.loss,
        }
        for row in rows
    }


def plot_summary(report: dict, output_path: Path) -> None:
    modes = list(report["modes"].keys())
    metric_names = ("direct", "paraphrase", "ripple", "history", "locality", "rule")
    values = []
    for mode in modes:
        evals = report["modes"][mode]["eval"]
        values.append([evals.get(metric, {}).get("exact", 0.0) for metric in metric_names])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / max(1, len(modes))
    x_positions = list(range(len(metric_names)))
    for index, mode in enumerate(modes):
        offset = (index - (len(modes) - 1) / 2.0) * width
        ax.bar([x + offset for x in x_positions], values[index], width=width, label=mode)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("candidate exact")
    ax.set_title("Semantic closure update on frozen Pythia adapter")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if args.pad_token_id >= int(getattr(tokenizer, "vocab_size", 0)):
        raise ValueError(
            f"--pad-token-id={args.pad_token_id} is outside tokenizer vocab size {tokenizer.vocab_size}."
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
    ).to(device)
    hidden_size = int(base_model.config.hidden_size)
    adapter = HiddenAdapter(hidden_size=hidden_size, rank=args.adapter_rank, scale=args.adapter_scale).to(device)
    model = PythiaAdapterLM(base_model, adapter).to(device)

    entities = build_entities()
    edit = entities[0]
    foundation = foundation_items(entities)
    closure = semantic_closure_items(edit, entities)
    direct = direct_only_items(edit)
    protected = protected_items(edit, entities)
    eval_items = closure

    foundation_trace = train_foundation(model, tokenizer, foundation, args, device=device)
    reference_hidden = adapted_hidden_matrix(model, tokenizer, protected, args, device=device)
    foundation_eval, foundation_predictions = evaluate_items(model, tokenizer, eval_items, args, device=device)
    foundation_training_eval, foundation_training_predictions = evaluate_items(
        model,
        tokenizer,
        foundation,
        args,
        device=device,
    )

    modes: dict[str, dict] = {
        "foundation_before_update": {
            "eval": rows_to_dict(foundation_eval),
            "foundation_eval": rows_to_dict(foundation_training_eval),
            "margin": correction_margin(model, tokenizer, edit, args, device=device),
            "predictions": foundation_predictions,
            "foundation_predictions": foundation_training_predictions,
        }
    }

    update_modes = (
        ("direct_raw_packet", direct, False),
        ("semantic_closure_raw_packet", closure, False),
        ("direct_constrained_packet", direct, True),
        ("semantic_closure_constrained_packet", closure, True),
    )
    for mode_name, packet_items, constrained in update_modes:
        mode_model = clone_adapter_model(model).to(device)
        mode_trace, mode_constraints = train_update_mode(
            mode_model,
            tokenizer,
            edit,
            packet_items,
            protected,
            args,
            device=device,
            constrained=constrained,
        )
        mode_eval, mode_predictions = evaluate_items(mode_model, tokenizer, eval_items, args, device=device)
        mode_hidden = adapted_hidden_matrix(mode_model, tokenizer, protected, args, device=device)
        modes[mode_name] = {
            "constrained": constrained,
            "eval": rows_to_dict(mode_eval),
            "margin": correction_margin(mode_model, tokenizer, edit, args, device=device),
            "protected_cka": linear_cka(reference_hidden, mode_hidden),
            "trace": mode_trace,
            "selected_constraints": mode_constraints,
            "predictions": mode_predictions,
        }

    report = {
        "experiment": "pythia_semantic_closure_update",
        "model_dir": str(args.model_dir),
        "device": args.device,
        "seed": args.seed,
        "adapter_rank": args.adapter_rank,
        "adapter_scale": args.adapter_scale,
        "foundation_trace": foundation_trace,
        "edit": asdict(edit),
        "foundation_items": [asdict(item) for item in foundation],
        "closure_items": [asdict(item) for item in closure],
        "protected_items": [asdict(item) for item in protected],
        "modes": modes,
    }
    output_json = output_dir / "pythia_semantic_closure_update.json"
    output_plot = output_dir / "pythia_semantic_closure_update.png"
    with output_json.open("w") as handle:
        json.dump(report, handle, indent=2)
    plot_summary(report, output_plot)

    print("\nPYTHIA SEMANTIC-CLOSURE UPDATE")
    print("=" * 132)
    print(
        f"device={args.device} model={args.model_dir} adapter_rank={args.adapter_rank} "
        f"foundation_epochs={args.foundation_epochs} update_epochs={args.update_epochs}"
    )
    print("-" * 132)
    print(f"{'mode':>28} {'direct':>8} {'para':>8} {'ripple':>8} {'history':>8} {'local':>8} {'rule':>8} {'margin':>9} {'cka':>8}")
    for mode_name, mode in modes.items():
        evals = mode["eval"]
        cka = mode.get("protected_cka", float("nan"))
        print(
            f"{mode_name:>28} "
            f"{evals.get('direct', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('paraphrase', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('ripple', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('history', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('locality', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('rule', {}).get('exact', 0.0):8.4f} "
            f"{mode['margin']:9.4f} "
            f"{cka:8.4f}"
        )
    print(f"wrote_json={output_json}")
    print(f"wrote_plot={output_plot}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("model/checkpoints/pythia-70m"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis/gco-pythia-semantic-closure-update-seed0"))
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--foundation-epochs", type=int, default=160)
    parser.add_argument("--update-epochs", type=int, default=80)
    parser.add_argument("--foundation-lr", type=float, default=2e-3)
    parser.add_argument("--update-lr", type=float, default=2e-3)
    parser.add_argument("--projection-damping", type=float, default=1e-3)
    parser.add_argument("--restore-strength", type=float, default=0.25)
    parser.add_argument("--restore-norm-ratio", type=float, default=0.35)
    parser.add_argument("--old-suppression-weight", type=float, default=0.35)
    parser.add_argument("--old-margin", type=float, default=0.25)
    parser.add_argument("--constraint-limit", type=int, default=12)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
