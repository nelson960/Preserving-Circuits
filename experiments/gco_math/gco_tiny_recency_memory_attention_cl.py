#!/usr/bin/env python3
"""Recency-memory attention for tokenized continual learning.

This experiment tests a different CL idea from Invariant-Tangent weight edits:

    Attention only when needed, memory for the rest.

The model still uses a normal tokenized neural input path:

    BPE tokenizer -> token embedding -> positional embedding -> attention blocks.

But changing current facts are not forced into the backbone weights.  Instead,
the model is trained to answer from parametric knowledge when no memory is
present, and to let retrieved memory override current-state answers when memory
is present.  Repeated corrections are stored as memory entries.  A recency bias
then decides which memory entry should win when the same entity was corrected
multiple times.

This is still controlled text, not open-world reasoning.  It isolates one
architectural question: can a tokenized transformer use recent memory entries to
handle continual corrections without overwriting old/history/local behavior?
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import (  # noqa: E402
    make_qa_supervision,
    masked_cross_entropy,
    require_token_id,
    resolve_device,
)


CITY_TO_CURRENCY = {
    "Paris": "euro",
    "London": "pound",
    "Tokyo": "yen",
    "Berlin": "euro",
    "Rome": "euro",
    "Madrid": "euro",
}


@dataclass(frozen=True)
class EntityRecord:
    name: str
    old_city: str
    old_currency: str
    color: str


@dataclass(frozen=True)
class CorrectionStage:
    index: int
    entity: str
    target_city: str
    repeated_entity: bool

    @property
    def target_currency(self) -> str:
        return currency_for_city(self.target_city)


@dataclass(frozen=True)
class MemoryEntry:
    stage: int
    entity: str
    city: str


@dataclass(frozen=True)
class TensorBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    answer_mask: torch.Tensor
    memory: torch.Tensor
    memory_recency: torch.Tensor

    def to(self, device: torch.device) -> "TensorBatch":
        return TensorBatch(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            answer_mask=self.answer_mask.to(device),
            memory=self.memory.to(device),
            memory_recency=self.memory_recency.to(device),
        )


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    for name in (
        "stages",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "max_seq_len",
        "max_memory_len",
        "base_epochs",
        "cl_epochs",
        "batch_size",
        "memory_override_repeats",
    ):
        positive_int(name, getattr(args, name))
    if args.seed < 0:
        raise ValueError(f"seed must be non-negative, got {args.seed}.")
    for name in (
        "base_lr",
        "cl_lr",
        "gradient_clip",
        "recency_strength",
        "recency_tau",
        "memory_copy_strength",
    ):
        positive_float(name, getattr(args, name))
    probability("drop_memory_probability", args.drop_memory_probability)
    if args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def currency_for_city(city: str) -> str:
    currency = CITY_TO_CURRENCY.get(city)
    if currency is None:
        raise ValueError(f"Unknown city {city!r}.")
    return currency


def records() -> tuple[EntityRecord, ...]:
    return (
        EntityRecord("Alice", "Paris", "euro", "blue"),
        EntityRecord("Henry", "London", "pound", "green"),
        EntityRecord("Mary", "Rome", "euro", "red"),
        EntityRecord("John", "Madrid", "euro", "blue"),
        EntityRecord("Clara", "Berlin", "euro", "green"),
        EntityRecord("Darin", "Tokyo", "yen", "red"),
    )


def record_map(records_: Sequence[EntityRecord]) -> dict[str, EntityRecord]:
    mapping = {record.name: record for record in records_}
    if len(mapping) != len(records_):
        raise ValueError("Duplicate entity names are not allowed.")
    return mapping


def build_stage_plan(records_: Sequence[EntityRecord], stages: int) -> tuple[CorrectionStage, ...]:
    city_cycle = tuple(CITY_TO_CURRENCY.keys())
    current = {record.name: record.old_city for record in records_}
    counts = {record.name: 0 for record in records_}
    plan: list[CorrectionStage] = []
    for index in range(stages):
        record = records_[index % len(records_)]
        counts[record.name] += 1
        candidates = [
            city
            for city in city_cycle
            if city != current[record.name] and city != record.old_city
        ]
        if not candidates:
            raise RuntimeError(f"No correction target is available for {record.name}.")
        target = candidates[(index + counts[record.name]) % len(candidates)]
        current[record.name] = target
        plan.append(
            CorrectionStage(
                index=index + 1,
                entity=record.name,
                target_city=target,
                repeated_entity=counts[record.name] > 1,
            )
        )
    return tuple(plan)


def qa(question: str, answer: str) -> dict[str, str]:
    return {"question": question, "answer": answer}


def base_prompt_rows(records_: Sequence[EntityRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records_:
        rows.extend(
            (
                qa(f"Where does {record.name} live now?", record.old_city),
                qa(f"What city is {record.name} in?", record.old_city),
                qa(f"Which city is {record.name} currently in?", record.old_city),
                qa(f"What currency does {record.name} use now?", record.old_currency),
                qa(f"Where did {record.name} live in the original record?", record.old_city),
                qa(f"What color does {record.name} like?", record.color),
            )
        )
    return rows


def correction_prompt_rows(stage: CorrectionStage) -> list[dict[str, str]]:
    return [
        qa(f"Where does {stage.entity} live now?", stage.target_city),
        qa(f"What city is {stage.entity} in?", stage.target_city),
        qa(f"Which city is {stage.entity} currently in?", stage.target_city),
        qa(f"What currency does {stage.entity} use now?", stage.target_currency),
    ]


def current_eval_rows(
    corrected_names: Sequence[str],
    current_city: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    return {
        "direct": [
            qa(f"Where does name live now?".replace("name", name), current_city[name])
            for name in corrected_names
        ],
        "paraphrase": [
            qa(f"What city is {name} in?", current_city[name])
            for name in corrected_names
        ],
        "heldout": [
            qa(f"Which city is {name} currently in?", current_city[name])
            for name in corrected_names
        ],
        "ripple": [
            qa(f"What currency does {name} use now?", currency_for_city(current_city[name]))
            for name in corrected_names
        ],
    }


def retention_eval_rows(
    records_: Sequence[EntityRecord],
    corrected_names: Sequence[str],
    current_city: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    corrected = set(corrected_names)
    return {
        "history": [
            qa(f"Where did {record.name} live in the original record?", record.old_city)
            for record in records_
        ],
        "locality": [
            qa(f"What color does {record.name} like?", record.color)
            for record in records_
        ]
        + [
            qa(f"Where does {record.name} live now?", current_city[record.name])
            for record in records_
            if record.name not in corrected
        ],
    }


def memory_entry_text(entry: MemoryEntry) -> str:
    return f"stage {entry.stage}: {entry.entity} lives now in {entry.city}."


def encode_memory(
    entries: Sequence[MemoryEntry],
    *,
    tokenizer: Tokenizer,
    max_memory_len: int,
    pad_id: int,
    latest_stage: int,
    recency_tau: float,
) -> tuple[list[int], list[float]]:
    if max_memory_len <= 1:
        raise ValueError("max_memory_len must leave room for memory tokens.")
    if not entries:
        return [pad_id] * max_memory_len, [0.0] * max_memory_len
    ids: list[int] = []
    recency: list[float] = []
    for entry in reversed(entries):
        token_ids = tokenizer.encode(memory_entry_text(entry)).ids
        if not token_ids:
            raise ValueError(f"Memory entry encoded to no tokens: {entry}")
        age = max(0, latest_stage - entry.stage)
        weight = math.exp(-float(age) / recency_tau)
        if not math.isfinite(weight):
            raise FloatingPointError(f"Non-finite recency weight for entry {entry}.")
        for token_id in token_ids:
            ids.append(int(token_id))
            recency.append(float(weight))
    if len(ids) > max_memory_len:
        ids = ids[:max_memory_len]
        recency = recency[:max_memory_len]
    padding = max_memory_len - len(ids)
    ids.extend([pad_id] * padding)
    recency.extend([0.0] * padding)
    return ids, recency


def empty_memory(max_memory_len: int, pad_id: int) -> tuple[list[int], list[float]]:
    return [pad_id] * max_memory_len, [0.0] * max_memory_len


def make_memory_batch(
    prompts: Sequence[dict[str, str]],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_memory_len: int,
    pad_id: int,
    memory_entries: Sequence[MemoryEntry],
    latest_stage: int,
    recency_tau: float,
    use_memory: bool,
    use_recency: bool,
) -> TensorBatch:
    supervision = make_qa_supervision(list(prompts), tokenizer, max_seq_len, pad_id)
    if supervision is None:
        raise ValueError("Cannot create a batch without prompts.")
    inputs, targets, answer_mask = supervision
    if use_memory:
        memory_ids, memory_recency = encode_memory(
            memory_entries,
            tokenizer=tokenizer,
            max_memory_len=max_memory_len,
            pad_id=pad_id,
            latest_stage=latest_stage,
            recency_tau=recency_tau,
        )
        if not use_recency:
            memory_recency = [1.0 if token_id != pad_id else 0.0 for token_id in memory_ids]
    else:
        memory_ids, memory_recency = empty_memory(max_memory_len, pad_id)
    memory = torch.tensor([memory_ids for _ in range(inputs.shape[0])], dtype=torch.long)
    recency = torch.tensor([memory_recency for _ in range(inputs.shape[0])], dtype=torch.float32)
    return TensorBatch(inputs=inputs, targets=targets, answer_mask=answer_mask, memory=memory, memory_recency=recency)


def make_parametric_batch(
    prompts: Sequence[dict[str, str]],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_memory_len: int,
    pad_id: int,
) -> TensorBatch:
    return make_memory_batch(
        prompts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_memory_len=max_memory_len,
        pad_id=pad_id,
        memory_entries=(),
        latest_stage=0,
        recency_tau=1.0,
        use_memory=False,
        use_recency=False,
    )


def concatenate_batches(batches: Sequence[TensorBatch]) -> TensorBatch:
    if not batches:
        raise ValueError("Cannot concatenate zero batches.")
    return TensorBatch(
        inputs=torch.cat([batch.inputs for batch in batches], dim=0),
        targets=torch.cat([batch.targets for batch in batches], dim=0),
        answer_mask=torch.cat([batch.answer_mask for batch in batches], dim=0),
        memory=torch.cat([batch.memory for batch in batches], dim=0),
        memory_recency=torch.cat([batch.memory_recency for batch in batches], dim=0),
    )


def minibatches(batch: TensorBatch, batch_size: int) -> Iterable[TensorBatch]:
    count = batch.inputs.shape[0]
    if count <= 0:
        raise ValueError("Cannot iterate an empty batch.")
    permutation = torch.randperm(count)
    for start in range(0, count, batch_size):
        indices = permutation[start : start + batch_size]
        yield TensorBatch(
            inputs=batch.inputs[indices],
            targets=batch.targets[indices],
            answer_mask=batch.answer_mask[indices],
            memory=batch.memory[indices],
            memory_recency=batch.memory_recency[indices],
        )


class WindowSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int | None) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(float(self.d_head))
        positions = torch.arange(seq_len, device=x.device)
        causal = positions.view(1, -1) <= positions.view(-1, 1)
        if self.window is not None:
            local = positions.view(-1, 1) - positions.view(1, -1) < self.window
            allowed = causal & local
        else:
            allowed = causal
        scores = scores.masked_fill(~allowed.view(1, 1, seq_len, seq_len), -torch.inf)
        attn = F.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out(y)


class MemoryCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        memory_hidden: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_recency: torch.Tensor,
        recency_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, channels = x.shape
        mem_len = memory_hidden.shape[1]
        q = self.q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(memory_hidden).view(batch, mem_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(memory_hidden).view(batch, mem_len, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(float(self.d_head))
        scores = scores + recency_strength * memory_recency.view(batch, 1, 1, mem_len)
        scores = scores.masked_fill(~memory_mask.view(batch, 1, 1, mem_len), -torch.inf)
        no_memory = ~memory_mask.any(dim=1)
        if bool(no_memory.any()):
            scores = scores.masked_fill(no_memory.view(batch, 1, 1, 1), 0.0)
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(~memory_mask.view(batch, 1, 1, mem_len), 0.0)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        gate = torch.sigmoid(self.gate(x))
        gate = gate.masked_fill(no_memory.view(batch, 1, 1), 0.0)
        return gate * self.out(y), attn.mean(dim=1)


class MemoryBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, window: int | None) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = WindowSelfAttention(d_model, n_heads, window)
        self.ln_mem = nn.LayerNorm(d_model)
        self.cross = MemoryCrossAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(
        self,
        x: torch.Tensor,
        memory_hidden: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_recency: torch.Tensor,
        recency_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.self_attn(self.ln1(x))
        memory_update, attention = self.cross(
            self.ln_mem(x),
            memory_hidden,
            memory_mask,
            memory_recency,
            recency_strength,
        )
        x = x + memory_update
        x = x + self.ff(self.ln2(x))
        return x, attention


class RecencyMemoryLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        max_memory_len: int,
        window: int | None,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.max_memory_len = max_memory_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max(max_seq_len, max_memory_len), d_model)
        self.blocks = nn.ModuleList(
            [MemoryBlock(d_model, n_heads, d_ff, window) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def embed_with_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [batch, seq], got {tuple(tokens.shape)}.")
        seq_len = tokens.shape[1]
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError("Sequence exceeds available position embeddings.")
        positions = torch.arange(seq_len, device=tokens.device).view(1, seq_len).expand_as(tokens)
        return self.token_embedding(tokens) + self.position_embedding(positions)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_recency: torch.Tensor,
        *,
        recency_strength: float,
        memory_copy_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(f"Question sequence exceeds max_seq_len={self.max_seq_len}.")
        if memory.shape[1] > self.max_memory_len:
            raise ValueError(f"Memory sequence exceeds max_memory_len={self.max_memory_len}.")
        if memory_recency.shape != memory.shape:
            raise ValueError("memory_recency must match memory token shape.")
        h = self.embed_with_positions(tokens)
        memory_hidden = self.embed_with_positions(memory)
        memory_mask = memory != self.pad_id
        memory_recency = memory_recency.to(device=memory.device, dtype=memory_hidden.dtype)
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            h, attention = block(h, memory_hidden, memory_mask, memory_recency, recency_strength)
            attentions.append(attention.detach())
        h = self.ln_f(h)
        logits = self.lm_head(h)
        if attentions:
            attention_mean = torch.stack(attentions).mean(dim=0)
        else:
            attention_mean = torch.zeros(
                tokens.shape[0],
                tokens.shape[1],
                memory.shape[1],
                device=tokens.device,
                dtype=h.dtype,
            )
        if memory_copy_strength > 0.0:
            copy_distribution = torch.zeros(
                logits.shape,
                device=logits.device,
                dtype=logits.dtype,
            )
            memory_indices = memory.unsqueeze(1).expand(
                memory.shape[0],
                tokens.shape[1],
                memory.shape[1],
            )
            copy_distribution.scatter_add_(2, memory_indices, attention_mean.to(dtype=logits.dtype))
            copy_distribution[:, :, self.pad_id] = 0.0
            logits = logits + memory_copy_strength * copy_distribution
        return logits, h, attention_mean


def answer_loss(
    model: RecencyMemoryLM,
    batch: TensorBatch,
    *,
    recency_strength: float,
    memory_copy_strength: float,
) -> torch.Tensor:
    logits, _hidden, _attention = model(
        batch.inputs,
        batch.memory,
        batch.memory_recency,
        recency_strength=recency_strength,
        memory_copy_strength=memory_copy_strength,
    )
    return masked_cross_entropy(logits, batch.targets, batch.answer_mask)


def train_model(
    model: RecencyMemoryLM,
    batch: TensorBatch,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    gradient_clip: float,
    recency_strength: float,
    memory_copy_strength: float,
    progress: bool,
    label: str,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"No trainable parameters for {label}.")
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    trace: list[dict[str, float]] = []
    iterator = range(1, epochs + 1)
    if progress:
        iterator = tqdm(iterator, desc=label, leave=False, dynamic_ncols=True)
    for epoch in iterator:
        losses: list[float] = []
        model.train()
        for mini in minibatches(batch, batch_size):
            mini = mini.to(next(model.parameters()).device)
            optimizer.zero_grad(set_to_none=True)
            loss = answer_loss(
                model,
                mini,
                recency_strength=recency_strength,
                memory_copy_strength=memory_copy_strength,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {label} at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        trace.append({"epoch": float(epoch), "loss": float(sum(losses) / len(losses))})
    return trace


@torch.no_grad()
def evaluate_prompts(
    model: RecencyMemoryLM,
    prompts: Sequence[dict[str, str]],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_memory_len: int,
    pad_id: int,
    memory_entries: Sequence[MemoryEntry],
    latest_stage: int,
    recency_tau: float,
    use_memory: bool,
    use_recency: bool,
    recency_strength: float,
    memory_copy_strength: float,
    device: torch.device,
) -> dict[str, object]:
    batch = make_memory_batch(
        prompts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_memory_len=max_memory_len,
        pad_id=pad_id,
        memory_entries=memory_entries,
        latest_stage=latest_stage,
        recency_tau=recency_tau,
        use_memory=use_memory,
        use_recency=use_recency,
    ).to(device)
    model.eval()
    logits, hidden, memory_attention = model(
        batch.inputs,
        batch.memory,
        batch.memory_recency,
        recency_strength=recency_strength,
        memory_copy_strength=memory_copy_strength,
    )
    loss = masked_cross_entropy(logits, batch.targets, batch.answer_mask)
    predictions = logits.argmax(dim=-1)
    mask = batch.answer_mask.to(dtype=torch.bool)
    token_accuracy = float((predictions[mask] == batch.targets[mask]).to(torch.float32).mean().cpu())
    rows: list[dict[str, object]] = []
    exact_values: list[float] = []
    memory_mass_values: list[float] = []
    for row_index, prompt in enumerate(prompts):
        row_mask = mask[row_index]
        expected_ids = batch.targets[row_index][row_mask].detach().cpu().tolist()
        predicted_ids = predictions[row_index][row_mask].detach().cpu().tolist()
        exact = float(torch.all(predictions[row_index][row_mask] == batch.targets[row_index][row_mask]).cpu())
        exact_values.append(exact)
        memory_mass = float(memory_attention[row_index][row_mask].sum(dim=-1).mean().detach().cpu())
        memory_mass_values.append(memory_mass)
        rows.append(
            {
                "question": prompt["question"],
                "expected": prompt["answer"],
                "predicted": tokenizer.decode(predicted_ids),
                "expected_token_ids": expected_ids,
                "predicted_token_ids": predicted_ids,
                "exact": exact,
                "memory_attention_mass": memory_mass,
            }
        )
    return {
        "loss": float(loss.detach().cpu()),
        "token_accuracy": token_accuracy,
        "exact": float(sum(exact_values) / len(exact_values)),
        "memory_attention_mass": float(sum(memory_mass_values) / len(memory_mass_values)),
        "rows": rows,
        "hidden": hidden.detach().cpu(),
        "answer_mask": batch.answer_mask.detach().cpu(),
    }


def hidden_answer_rows(eval_result: dict[str, object]) -> torch.Tensor:
    hidden = eval_result["hidden"]
    mask = eval_result["answer_mask"]
    if not isinstance(hidden, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Evaluation result does not contain tensor hidden/mask fields.")
    rows = hidden[mask.to(dtype=torch.bool)]
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError("Need at least two answer hidden rows for geometry.")
    return rows.to(dtype=torch.float32)


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"CKA requires matching shapes, got {left.shape} and {right.shape}.")
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    numerator = torch.linalg.matrix_norm(left.T @ right) ** 2
    denominator = torch.linalg.matrix_norm(left.T @ left) * torch.linalg.matrix_norm(right.T @ right)
    return float((numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0).cpu())


def relative_drift(reference: torch.Tensor, current: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(current - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).cpu()
    )


def build_override_training_batches(
    records_: Sequence[EntityRecord],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_memory_len: int,
    pad_id: int,
    repeats: int,
    recency_tau: float,
) -> list[TensorBatch]:
    city_cycle = tuple(CITY_TO_CURRENCY.keys())
    batches: list[TensorBatch] = []
    stage = 0
    per_entity_memory: dict[str, list[MemoryEntry]] = {record.name: [] for record in records_}
    for repeat in range(repeats):
        for index, record in enumerate(records_):
            candidates = [city for city in city_cycle if city != record.old_city]
            city = candidates[(index + repeat) % len(candidates)]
            stage += 1
            entry = MemoryEntry(stage=stage, entity=record.name, city=city)
            per_entity_memory[record.name].append(entry)
            prompts = [
                qa(f"Where does {record.name} live now?", city),
                qa(f"What city is {record.name} in?", city),
                qa(f"Which city is {record.name} currently in?", city),
                qa(f"What currency does {record.name} use now?", currency_for_city(city)),
                qa(f"Where did {record.name} live in the original record?", record.old_city),
                qa(f"What color does {record.name} like?", record.color),
            ]
            batches.append(
                make_memory_batch(
                    prompts,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    max_memory_len=max_memory_len,
                    pad_id=pad_id,
                    memory_entries=tuple(per_entity_memory[record.name]),
                    latest_stage=stage,
                    recency_tau=recency_tau,
                    use_memory=True,
                    use_recency=True,
                )
            )
    return batches


def build_base_training_batch(
    records_: Sequence[EntityRecord],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_memory_len: int,
    pad_id: int,
    override_repeats: int,
    recency_tau: float,
) -> TensorBatch:
    parametric = make_parametric_batch(
        base_prompt_rows(records_),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_memory_len=max_memory_len,
        pad_id=pad_id,
    )
    memory_old = make_memory_batch(
        base_prompt_rows(records_),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_memory_len=max_memory_len,
        pad_id=pad_id,
        memory_entries=tuple(
            MemoryEntry(stage=index + 1, entity=record.name, city=record.old_city)
            for index, record in enumerate(records_)
        ),
        latest_stage=len(records_),
        recency_tau=recency_tau,
        use_memory=True,
        use_recency=True,
    )
    override = build_override_training_batches(
        records_,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_memory_len=max_memory_len,
        pad_id=pad_id,
        repeats=override_repeats,
        recency_tau=recency_tau,
    )
    return concatenate_batches([parametric, memory_old, *override])


def instantiate(args: argparse.Namespace, *, vocab_size: int, pad_id: int, device: torch.device) -> RecencyMemoryLM:
    window = None if args.attention_window <= 0 else args.attention_window
    return RecencyMemoryLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        max_memory_len=args.max_memory_len,
        window=window,
        pad_id=pad_id,
    ).to(device)


def evaluation_groups(
    records_: Sequence[EntityRecord],
    corrected_names: Sequence[str],
    current_city: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    groups = current_eval_rows(corrected_names, current_city)
    groups.update(retention_eval_rows(records_, corrected_names, current_city))
    return groups


def evaluate_method(
    model: RecencyMemoryLM,
    base_model: RecencyMemoryLM,
    records_: Sequence[EntityRecord],
    corrected_names: Sequence[str],
    current_city: dict[str, str],
    memory_entries: Sequence[MemoryEntry],
    *,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    pad_id: int,
    device: torch.device,
    use_memory: bool,
    use_recency: bool,
) -> dict[str, object]:
    groups = evaluation_groups(records_, corrected_names, current_city)
    behavior: dict[str, object] = {}
    hidden_rows: list[torch.Tensor] = []
    base_rows: list[torch.Tensor] = []
    for name, prompts in groups.items():
        result = evaluate_prompts(
            model,
            prompts,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_memory_len=args.max_memory_len,
            pad_id=pad_id,
            memory_entries=memory_entries,
            latest_stage=args.stages,
            recency_tau=args.recency_tau,
            use_memory=use_memory,
            use_recency=use_recency,
            recency_strength=args.recency_strength if use_recency else 0.0,
            memory_copy_strength=args.memory_copy_strength if use_memory else 0.0,
            device=device,
        )
        base_result = evaluate_prompts(
            base_model,
            prompts,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_memory_len=args.max_memory_len,
            pad_id=pad_id,
            memory_entries=(),
            latest_stage=0,
            recency_tau=args.recency_tau,
            use_memory=False,
            use_recency=False,
            recency_strength=0.0,
            memory_copy_strength=0.0,
            device=device,
        )
        hidden_rows.append(hidden_answer_rows(result))
        base_rows.append(hidden_answer_rows(base_result))
        result = dict(result)
        result.pop("hidden")
        result.pop("answer_mask")
        behavior[name] = result
    current_hidden = torch.cat(hidden_rows, dim=0)
    reference_hidden = torch.cat(base_rows, dim=0)
    return {
        "behavior": behavior,
        "geometry": {
            "answer_hidden_cka_vs_base": linear_cka(reference_hidden, current_hidden),
            "answer_hidden_relative_drift": relative_drift(reference_hidden, current_hidden),
        },
    }


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = require_token_id(tokenizer, "[PAD]")
    records_ = records()
    plan = build_stage_plan(records_, args.stages)

    base_model = instantiate(args, vocab_size=tokenizer.get_vocab_size(), pad_id=pad_id, device=device)
    base_batch = build_base_training_batch(
        records_,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_memory_len=args.max_memory_len,
        pad_id=pad_id,
        override_repeats=args.memory_override_repeats,
        recency_tau=args.recency_tau,
    ).to(device)
    base_trace = train_model(
        base_model,
        base_batch,
        epochs=args.base_epochs,
        lr=args.base_lr,
        batch_size=args.batch_size,
        gradient_clip=args.gradient_clip,
        recency_strength=args.recency_strength,
        memory_copy_strength=args.memory_copy_strength,
        progress=not args.no_progress,
        label="base memory-attention training",
    )
    base_model.eval()

    methods = {
        "parametric_naive": copy.deepcopy(base_model),
        "memory_no_recency": copy.deepcopy(base_model),
        "memory_recency": copy.deepcopy(base_model),
    }
    for name in ("memory_no_recency", "memory_recency"):
        for parameter in methods[name].parameters():
            parameter.requires_grad_(False)

    current_city = {method: {record.name: record.old_city for record in records_} for method in methods}
    corrected_names = {method: [] for method in methods}
    memory_entries = {method: [] for method in methods}
    traces: dict[str, list[dict[str, object]]] = {method: [] for method in methods}

    for stage in plan:
        prompts = correction_prompt_rows(stage)
        naive_batch = make_parametric_batch(
            prompts,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_memory_len=args.max_memory_len,
            pad_id=pad_id,
        ).to(device)
        trace = train_model(
            methods["parametric_naive"],
            naive_batch,
            epochs=args.cl_epochs,
            lr=args.cl_lr,
            batch_size=args.batch_size,
            gradient_clip=args.gradient_clip,
            recency_strength=0.0,
            memory_copy_strength=0.0,
            progress=not args.no_progress,
            label=f"parametric naive stage {stage.index}",
        )
        current_city["parametric_naive"][stage.entity] = stage.target_city
        if stage.entity not in corrected_names["parametric_naive"]:
            corrected_names["parametric_naive"].append(stage.entity)
        traces["parametric_naive"].append({"stage": asdict(stage), "trace": trace})

        for method in ("memory_no_recency", "memory_recency"):
            memory_entries[method].append(
                MemoryEntry(stage=stage.index, entity=stage.entity, city=stage.target_city)
            )
            current_city[method][stage.entity] = stage.target_city
            if stage.entity not in corrected_names[method]:
                corrected_names[method].append(stage.entity)
            # No weight update: changing facts remain in retrieved memory.
            eval_loss = evaluate_prompts(
                methods[method],
                prompts,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                max_memory_len=args.max_memory_len,
                pad_id=pad_id,
                memory_entries=memory_entries[method],
                latest_stage=stage.index,
                recency_tau=args.recency_tau,
                use_memory=True,
                use_recency=method == "memory_recency",
                recency_strength=args.recency_strength if method == "memory_recency" else 0.0,
                memory_copy_strength=args.memory_copy_strength,
                device=device,
            )
            traces[method].append(
                {
                    "stage": asdict(stage),
                    "memory_entries": [asdict(entry) for entry in memory_entries[method]],
                    "candidate_exact": eval_loss["exact"],
                    "candidate_loss": eval_loss["loss"],
                    "candidate_memory_attention": eval_loss["memory_attention_mass"],
                }
            )

    final: dict[str, object] = {}
    final["parametric_naive"] = evaluate_method(
        methods["parametric_naive"],
        base_model,
        records_,
        corrected_names["parametric_naive"],
        current_city["parametric_naive"],
        (),
        tokenizer=tokenizer,
        args=args,
        pad_id=pad_id,
        device=device,
        use_memory=False,
        use_recency=False,
    )
    for method in ("memory_no_recency", "memory_recency"):
        final[method] = evaluate_method(
            methods[method],
            base_model,
            records_,
            corrected_names[method],
            current_city[method],
            memory_entries[method],
            tokenizer=tokenizer,
            args=args,
            pad_id=pad_id,
            device=device,
            use_memory=True,
            use_recency=method == "memory_recency",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {**vars(args), "tokenizer_path": str(args.tokenizer_path), "output_dir": str(args.output_dir)},
        "records": [asdict(record) for record in records_],
        "stage_plan": [asdict(stage) for stage in plan],
        "base_trace": base_trace,
        "traces": traces,
        "final": final,
    }
    json_path = args.output_dir / "recency_memory_attention_cl.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("\nRECENCY MEMORY ATTENTION CL SUMMARY")
    print("=" * 144)
    print(
        f"device={device.type} stages={args.stages} d_model={args.d_model} "
        f"layers={args.n_layers} window={args.attention_window} memory_len={args.max_memory_len}"
    )
    print("-" * 144)
    print(
        f"{'method':>24} {'direct':>9} {'para':>9} {'heldout':>9} {'ripple':>9} "
        f"{'history':>9} {'locality':>9} {'memAttn':>9} {'cka':>8} {'drift':>8}"
    )
    for method, result in final.items():
        behavior = result["behavior"]  # type: ignore[index]
        geometry = result["geometry"]  # type: ignore[index]

        def exact(name: str) -> float:
            return float(behavior[name]["exact"])  # type: ignore[index]

        mem_attention = sum(
            float(behavior[name]["memory_attention_mass"])  # type: ignore[index]
            for name in ("direct", "paraphrase", "heldout", "ripple")
        ) / 4.0
        print(
            f"{method:>24} "
            f"{exact('direct'):9.4f} {exact('paraphrase'):9.4f} {exact('heldout'):9.4f} "
            f"{exact('ripple'):9.4f} {exact('history'):9.4f} {exact('locality'):9.4f} "
            f"{mem_attention:9.4f} "
            f"{float(geometry['answer_hidden_cka_vs_base']):8.4f} "
            f"{float(geometry['answer_hidden_relative_drift']):8.4f}"
        )
    print(f"wrote_json={json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis/gco-tiny-recency-memory-attention-cl-seed0"))
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--max-memory-len", type=int, default=96)
    parser.add_argument("--attention-window", type=int, default=12)
    parser.add_argument("--base-epochs", type=int, default=450)
    parser.add_argument("--cl-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--base-lr", type=float, default=2e-3)
    parser.add_argument("--cl-lr", type=float, default=8e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--recency-strength", type=float, default=3.0)
    parser.add_argument("--recency-tau", type=float, default=2.0)
    parser.add_argument("--memory-copy-strength", type=float, default=5.0)
    parser.add_argument("--memory-override-repeats", type=int, default=3)
    parser.add_argument("--drop-memory-probability", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
