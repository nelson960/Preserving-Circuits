#!/usr/bin/env python3
"""Tokenized semantic continual-learning bridge.

This experiment is the first step away from graph-ID toy inputs.  Facts,
questions, corrections, and consequence questions are ordinary text strings.
They are encoded by a BPE tokenizer, passed through token embeddings and
positional embeddings in a small decoder transformer, and trained with masked
answer-token losses.

The experiment deliberately keeps the semantic world small so the mechanism is
inspectable:

* train a base tokenized language model on current facts, historical facts, and
  unrelated local facts;
* apply several corrections sequentially;
* compare naive answer-loss finetuning against an Invariant-Tangent update that
  learns direct/paraphrase/ripple answers while protecting historical/local
  behavior and answer-token geometry.

This is still controlled text, not open-world reasoning.  Its purpose is to
test whether the CL mechanism survives the real neural input path:

    tokenizer ids -> token embedding + positional embedding -> transformer.
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
from typing import Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.gco_math.gco_mini_cl_world_demo import (  # noqa: E402
    assign_flat_gradient,
    flat_autograd_gradient,
)
from experiments.models import DecoderTransformer  # noqa: E402
from experiments.real_book_common import (  # noqa: E402
    make_qa_supervision,
    masked_cross_entropy,
    require_token_id,
    resolve_device,
)


@dataclass(frozen=True)
class EntityRecord:
    name: str
    old_city: str
    new_city: str
    old_currency: str
    new_currency: str
    color: str


@dataclass(frozen=True)
class CorrectionStage:
    index: int
    entity: str
    target_city: str
    target_currency: str
    repeated_entity: bool


@dataclass(frozen=True)
class PromptGroup:
    name: str
    prompts: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class TensorBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device) -> "TensorBatch":
        return TensorBatch(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            mask=self.mask.to(device),
        )


CITY_TO_CURRENCY = {
    "Paris": "euro",
    "London": "pound",
    "Tokyo": "yen",
    "Berlin": "euro",
    "Rome": "euro",
    "Madrid": "euro",
}


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    for name in (
        "base_epochs",
        "cl_epochs",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "max_seq_len",
        "batch_size",
        "constraint_limit",
        "stages",
    ):
        positive_int(name, getattr(args, name))
    for name in (
        "base_lr",
        "cl_lr",
        "projection_damping",
        "restore_strength",
        "restore_norm_ratio",
        "geometry_restore_weight",
        "max_gradient_norm",
    ):
        positive_float(name, getattr(args, name))
    if args.n_heads <= 0 or args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if args.constraint_limit < 2:
        raise ValueError("constraint_limit must leave at least two protected rows.")
    if not args.stress_mode and args.stages > len(build_records()):
        raise ValueError(
            f"--stages={args.stages} exceeds the {len(build_records())} base records. "
            "Use --stress-mode to generate repeated corrections."
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_records() -> tuple[EntityRecord, ...]:
    return (
        EntityRecord("Alice", "Paris", "Tokyo", "euro", "yen", "blue"),
        EntityRecord("Henry", "London", "Berlin", "pound", "euro", "green"),
        EntityRecord("Mary", "Rome", "Tokyo", "euro", "yen", "red"),
        EntityRecord("John", "Madrid", "London", "euro", "pound", "blue"),
        EntityRecord("Clara", "Berlin", "Paris", "euro", "euro", "green"),
        EntityRecord("Darin", "Tokyo", "Rome", "yen", "euro", "red"),
    )


def require_currency_consistency(records: Sequence[EntityRecord]) -> None:
    for record in records:
        for city, currency in ((record.old_city, record.old_currency), (record.new_city, record.new_currency)):
            expected = CITY_TO_CURRENCY.get(city)
            if expected is None:
                raise ValueError(f"City {city!r} has no known currency mapping.")
            if expected != currency:
                raise ValueError(
                    f"Record {record.name!r} has currency {currency!r} for {city!r}, expected {expected!r}."
                )


def currency_for_city(city: str) -> str:
    currency = CITY_TO_CURRENCY.get(city)
    if currency is None:
        raise ValueError(f"City {city!r} has no currency mapping.")
    return currency


def qa(question: str, answer: str) -> dict[str, str]:
    return {"question": question, "answer": answer}


def base_prompts(records: Sequence[EntityRecord]) -> tuple[dict[str, str], ...]:
    prompts: list[dict[str, str]] = []
    for record in records:
        prompts.extend(
            (
                qa(f"Where does {record.name} live now?", record.old_city),
                qa(f"What city is {record.name} in?", record.old_city),
                qa(f"Which city is {record.name} currently in?", record.old_city),
                qa(f"What currency does {record.name} use now?", record.old_currency),
                qa(f"Where did {record.name} live in the original record?", record.old_city),
                qa(f"What color does {record.name} like?", record.color),
            )
        )
    return tuple(prompts)


def build_stage_plan(
    records: Sequence[EntityRecord],
    *,
    stages: int,
    stress_mode: bool,
) -> tuple[CorrectionStage, ...]:
    if not stress_mode:
        return tuple(
            CorrectionStage(
                index=index + 1,
                entity=record.name,
                target_city=record.new_city,
                target_currency=record.new_currency,
                repeated_entity=False,
            )
            for index, record in enumerate(records[:stages])
        )

    city_cycle = tuple(CITY_TO_CURRENCY.keys())
    current_city = {record.name: record.old_city for record in records}
    stage_counts = {record.name: 0 for record in records}
    plan: list[CorrectionStage] = []
    for index in range(stages):
        record = records[index % len(records)]
        stage_counts[record.name] += 1
        if stage_counts[record.name] == 1:
            target_city = record.new_city
        else:
            candidates = [
                city
                for city in city_cycle
                if city != current_city[record.name] and city != record.old_city
            ]
            if not candidates:
                raise RuntimeError(f"No stress target city is available for {record.name}.")
            target_city = candidates[(index + stage_counts[record.name]) % len(candidates)]
        current_city[record.name] = target_city
        plan.append(
            CorrectionStage(
                index=index + 1,
                entity=record.name,
                target_city=target_city,
                target_currency=currency_for_city(target_city),
                repeated_entity=stage_counts[record.name] > 1,
            )
        )
    return tuple(plan)


def record_by_name(records: Sequence[EntityRecord]) -> dict[str, EntityRecord]:
    mapping = {record.name: record for record in records}
    if len(mapping) != len(records):
        raise ValueError("Entity records contain duplicate names.")
    return mapping


def candidate_prompts(stage: CorrectionStage) -> tuple[dict[str, str], ...]:
    return (
        qa(f"Where does {stage.entity} live now?", stage.target_city),
        qa(f"What city is {stage.entity} in?", stage.target_city),
        qa(f"What currency does {stage.entity} use now?", stage.target_currency),
    )


def current_prompts(entity: str, city: str, currency: str) -> tuple[dict[str, str], ...]:
    return (
        qa(f"Where does {entity} live now?", city),
        qa(f"What city is {entity} in?", city),
        qa(f"What currency does {entity} use now?", currency),
    )


def protected_prompts(
    records: Sequence[EntityRecord],
    current_city: dict[str, str],
    current_currency: dict[str, str],
    released_current: Sequence[str] = (),
) -> tuple[PromptGroup, ...]:
    released_names = set(released_current)
    record_names = {record.name for record in records}
    if set(current_city) != record_names or set(current_currency) != record_names:
        raise ValueError("Current state dictionaries must contain every entity exactly once.")
    history = [
        qa(f"Where did {record.name} live in the original record?", record.old_city)
        for record in records
    ]
    locality = [
        qa(f"What color does {record.name} like?", record.color)
        for record in records
    ]
    stable_current = [
        prompt
        for record in records
        if record.name not in released_names
        for prompt in current_prompts(
            record.name,
            current_city[record.name],
            current_currency[record.name],
        )
    ]
    groups = [
        PromptGroup("history", tuple(history)),
        PromptGroup("locality", tuple(locality)),
    ]
    if stable_current:
        groups.append(PromptGroup("current_state", tuple(stable_current)))
    return tuple(groups)


def make_batch(
    prompts: Sequence[dict[str, str]],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    pad_id: int,
) -> TensorBatch:
    supervision = make_qa_supervision(list(prompts), tokenizer, max_seq_len, pad_id)
    if supervision is None:
        raise ValueError("Cannot create a tensor batch from no prompts.")
    inputs, targets, mask = supervision
    return TensorBatch(inputs=inputs, targets=targets, mask=mask)


def concatenate_batches(batches: Sequence[TensorBatch]) -> TensorBatch:
    if not batches:
        raise ValueError("No batches to concatenate.")
    return TensorBatch(
        inputs=torch.cat([batch.inputs for batch in batches], dim=0),
        targets=torch.cat([batch.targets for batch in batches], dim=0),
        mask=torch.cat([batch.mask for batch in batches], dim=0),
    )


def select_answer_rows(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor [batch, seq, channels], got {tuple(tensor.shape)}.")
    if mask.shape != tensor.shape[:2]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match tensor prefix {tuple(tensor.shape[:2])}.")
    selected = tensor[mask.to(dtype=torch.bool)]
    if selected.ndim != 2 or selected.shape[0] <= 0:
        raise ValueError("Answer mask selected no rows.")
    return selected


def answer_loss(model: DecoderTransformer, batch: TensorBatch) -> torch.Tensor:
    logits, _hidden = model(batch.inputs)
    return masked_cross_entropy(logits, batch.targets, batch.mask)


def iter_prompt_batches(batch: TensorBatch, batch_size: int) -> list[TensorBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    count = batch.inputs.shape[0]
    if count <= 0:
        raise ValueError("Cannot iterate an empty prompt batch.")
    permutation = torch.randperm(count, device=batch.inputs.device)
    batches: list[TensorBatch] = []
    for start in range(0, count, batch_size):
        indices = permutation[start : start + batch_size]
        batches.append(
            TensorBatch(
                inputs=batch.inputs[indices],
                targets=batch.targets[indices],
                mask=batch.mask[indices],
            )
        )
    return batches


def train_answer_loss(
    model: DecoderTransformer,
    batch: TensorBatch,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    max_gradient_norm: float,
    label: str,
    progress: bool,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters for answer-loss training.")
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    trace: list[dict[str, float]] = []
    iterator = range(1, epochs + 1)
    if progress:
        iterator = tqdm(iterator, desc=label, leave=False, dynamic_ncols=True)
    for epoch in iterator:
        losses: list[float] = []
        model.train()
        for mini_batch in iter_prompt_batches(batch, batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = answer_loss(model, mini_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{label} loss became non-finite at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / len(losses)
        trace.append({"epoch": float(epoch), "loss": float(mean_loss)})
    return trace


@torch.no_grad()
def capture_answer_reference(model: DecoderTransformer, batch: TensorBatch) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, hidden = model(batch.inputs)
    answer_logits = select_answer_rows(logits, batch.mask).detach()
    answer_hidden = select_answer_rows(hidden, batch.mask).detach()
    return answer_logits, answer_hidden


def block_normalized_restore_loss(
    current_logits: torch.Tensor,
    current_hidden: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_hidden: torch.Tensor,
    *,
    geometry_weight: float,
) -> torch.Tensor:
    eps = torch.finfo(current_logits.dtype).eps
    reference_probabilities = torch.softmax(reference_logits, dim=-1)
    behavior = (
        reference_probabilities
        * (
            reference_probabilities.clamp_min(eps).log()
            - torch.log_softmax(current_logits, dim=-1)
        )
    ).sum(dim=-1).mean()
    geometry = F.mse_loss(current_hidden, reference_hidden)
    return behavior + geometry_weight * geometry


def bounded_restore_gradient(
    restore_gradient: torch.Tensor,
    tangent_gradient: torch.Tensor,
    *,
    strength: float,
    norm_ratio: float,
) -> tuple[torch.Tensor, float]:
    restore_norm = torch.linalg.vector_norm(restore_gradient)
    tangent_norm = torch.linalg.vector_norm(tangent_gradient)
    if float(restore_norm.detach().cpu()) <= 1e-12:
        return torch.zeros_like(restore_gradient), 0.0
    maximum = norm_ratio * tangent_norm
    coefficient = torch.minimum(
        restore_norm.new_tensor(strength),
        maximum / restore_norm.clamp_min(1e-12),
    )
    return coefficient * restore_gradient, float(coefficient.detach().cpu())


def project_gradient_control_plane(
    *,
    raw_gradient: torch.Tensor,
    constraint_gradients: Sequence[torch.Tensor],
    damping: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project a gradient through a small dense constraint solve.

    The model gradients stay on the training device, but the Gram solve is a
    tiny control-plane linear algebra problem.  Solving it on CPU avoids MPS
    backend failures without changing the mathematical projection.
    """

    positive_float("projection_damping", damping)
    active_rows: list[torch.Tensor] = []
    for constraint in constraint_gradients:
        if constraint.shape != raw_gradient.shape:
            raise RuntimeError(
                f"Constraint gradient shape mismatch: constraint={constraint.shape}, raw={raw_gradient.shape}."
            )
        row = constraint.to(device=raw_gradient.device, dtype=raw_gradient.dtype)
        norm = torch.linalg.vector_norm(row)
        if float(norm.detach().cpu()) > 1e-12:
            active_rows.append(row)

    raw_norm = torch.linalg.vector_norm(raw_gradient)
    if not active_rows:
        return raw_gradient.clone(), {
            "constraint_count": 0.0,
            "raw_grad_norm": float(raw_norm.detach().cpu()),
            "projected_grad_norm": float(raw_norm.detach().cpu()),
            "projection_removed_fraction": 0.0,
            "safe_grad_fraction": 1.0,
        }

    matrix = torch.stack(active_rows, dim=0)
    matrix_cpu = matrix.detach().to(device="cpu")
    matrix_cpu = matrix_cpu.to(dtype=torch.float64)
    raw_cpu = raw_gradient.detach().to(device="cpu")
    raw_cpu = raw_cpu.to(dtype=torch.float64)
    gram = matrix_cpu @ matrix_cpu.T
    rhs = matrix_cpu @ raw_cpu
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(gram + float(damping) * identity, rhs)
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError("Constraint solve produced non-finite coefficients.")
    coefficients = coefficients.to(device=raw_gradient.device, dtype=raw_gradient.dtype)
    projected = raw_gradient - matrix.T @ coefficients
    projected_norm = torch.linalg.vector_norm(projected)
    removed_norm = torch.linalg.vector_norm(raw_gradient - projected)
    return projected, {
        "constraint_count": float(len(active_rows)),
        "raw_grad_norm": float(raw_norm.detach().cpu()),
        "projected_grad_norm": float(projected_norm.detach().cpu()),
        "projection_removed_fraction": float((removed_norm / raw_norm.clamp_min(1e-12)).detach().cpu()),
        "safe_grad_fraction": float((projected_norm / raw_norm.clamp_min(1e-12)).detach().cpu()),
    }


def protected_constraint_losses(
    model: DecoderTransformer,
    groups: Sequence[tuple[str, TensorBatch, torch.Tensor, torch.Tensor]],
    *,
    geometry_weight: float,
    constraint_limit: int,
) -> list[torch.Tensor]:
    losses: list[torch.Tensor] = []
    for group_name, batch, reference_logits, reference_hidden in groups:
        logits, hidden = model(batch.inputs)
        current_logits = select_answer_rows(logits, batch.mask)
        current_hidden = select_answer_rows(hidden, batch.mask)
        row_count = current_logits.shape[0]
        if row_count <= 0:
            raise RuntimeError(f"Protected group {group_name!r} has no answer rows.")
        stride = max(1, math.ceil(row_count / constraint_limit))
        selected = list(range(0, row_count, stride))[:constraint_limit]
        for row in selected:
            losses.append(
                block_normalized_restore_loss(
                    current_logits[row : row + 1],
                    current_hidden[row : row + 1],
                    reference_logits[row : row + 1],
                    reference_hidden[row : row + 1],
                    geometry_weight=geometry_weight,
                )
            )
    if not losses:
        raise RuntimeError("No protected constraint losses were built.")
    return losses


def invariant_tangent_update(
    model: DecoderTransformer,
    candidate_batch: TensorBatch,
    protected_groups: Sequence[tuple[str, TensorBatch, torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    lr: float,
    projection_damping: float,
    restore_strength: float,
    restore_norm_ratio: float,
    geometry_restore_weight: float,
    max_gradient_norm: float,
    constraint_limit: int,
    label: str,
    progress: bool,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters for Invariant-Tangent update.")
    trace: list[dict[str, float]] = []
    iterator = range(1, epochs + 1)
    if progress:
        iterator = tqdm(iterator, desc=label, leave=False, dynamic_ncols=True)
    for epoch in iterator:
        model.train()
        candidate_loss = answer_loss(model, candidate_batch)
        if not torch.isfinite(candidate_loss):
            raise FloatingPointError(f"Candidate loss became non-finite at epoch {epoch}.")
        restore_terms: list[torch.Tensor] = []
        for _group_name, batch, reference_logits, reference_hidden in protected_groups:
            logits, hidden = model(batch.inputs)
            current_logits = select_answer_rows(logits, batch.mask)
            current_hidden = select_answer_rows(hidden, batch.mask)
            restore_terms.append(
                block_normalized_restore_loss(
                    current_logits,
                    current_hidden,
                    reference_logits,
                    reference_hidden,
                    geometry_weight=geometry_restore_weight,
                )
            )
        restore_loss = torch.stack(restore_terms).mean()
        constraint_losses = protected_constraint_losses(
            model,
            protected_groups,
            geometry_weight=geometry_restore_weight,
            constraint_limit=constraint_limit,
        )
        raw_gradient = flat_autograd_gradient(
            candidate_loss,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label="tokenized candidate",
        )
        constraint_gradients = [
            flat_autograd_gradient(
                loss,
                parameters,
                retain_graph=True,
                require_nonzero=False,
                label=f"protected token constraint {index}",
            )
            for index, loss in enumerate(constraint_losses)
        ]
        tangent_gradient, projection = project_gradient_control_plane(
            raw_gradient=raw_gradient,
            constraint_gradients=constraint_gradients,
            damping=projection_damping,
        )
        restore_gradient = flat_autograd_gradient(
            restore_loss,
            parameters,
            retain_graph=False,
            require_nonzero=False,
            label="tokenized bounded restore",
        )
        bounded_restore, restore_coefficient = bounded_restore_gradient(
            restore_gradient,
            tangent_gradient,
            strength=restore_strength,
            norm_ratio=restore_norm_ratio,
        )
        final_gradient = tangent_gradient + bounded_restore
        final_norm = torch.linalg.vector_norm(final_gradient)
        clip_scale = torch.minimum(
            final_norm.new_ones(()),
            final_norm.new_tensor(max_gradient_norm) / final_norm.clamp_min(1e-12),
        )
        final_gradient = final_gradient * clip_scale
        if not torch.isfinite(final_gradient).all():
            raise FloatingPointError(f"Final tokenized CL gradient is non-finite at epoch {epoch}.")
        assign_flat_gradient(parameters, final_gradient)
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is None:
                    raise RuntimeError("A trainable parameter did not receive a flat gradient.")
                parameter.add_(parameter.grad, alpha=-lr)
                parameter.grad = None
        trace.append(
            {
                "epoch": float(epoch),
                "candidate_loss": float(candidate_loss.detach().cpu()),
                "restore_loss": float(restore_loss.detach().cpu()),
                "safe_fraction": projection["safe_grad_fraction"],
                "removed_fraction": projection["projection_removed_fraction"],
                "constraint_rows": projection["constraint_count"],
                "restore_coefficient": restore_coefficient,
                "final_gradient_norm": float(torch.linalg.vector_norm(final_gradient).detach().cpu()),
            }
        )
    return trace


@torch.no_grad()
def evaluate_prompts(
    model: DecoderTransformer,
    prompts: Sequence[dict[str, str]],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    pad_id: int,
    device: torch.device,
) -> dict[str, float]:
    batch = make_batch(prompts, tokenizer=tokenizer, max_seq_len=max_seq_len, pad_id=pad_id).to(device)
    model.eval()
    logits, _hidden = model(batch.inputs)
    loss = masked_cross_entropy(logits, batch.targets, batch.mask)
    predictions = logits.argmax(dim=-1)
    answer_mask = batch.mask.to(dtype=torch.bool)
    token_accuracy = float((predictions[answer_mask] == batch.targets[answer_mask]).to(torch.float32).mean().cpu())
    exact_rows: list[float] = []
    rows: list[dict[str, object]] = []
    for row in range(batch.inputs.shape[0]):
        mask = answer_mask[row]
        if int(mask.sum().item()) <= 0:
            raise RuntimeError("Evaluation prompt has no answer-token mask.")
        target_ids = batch.targets[row][mask].detach().cpu().tolist()
        prediction_ids = predictions[row][mask].detach().cpu().tolist()
        exact = float(torch.all(predictions[row][mask] == batch.targets[row][mask]).cpu())
        exact_rows.append(exact)
        prompt = prompts[row]
        rows.append(
            {
                "question": prompt["question"],
                "expected": prompt["answer"],
                "expected_token_ids": target_ids,
                "predicted": tokenizer.decode(prediction_ids),
                "predicted_token_ids": prediction_ids,
                "exact": exact,
            }
        )
    return {
        "loss": float(loss.detach().cpu()),
        "token_accuracy": token_accuracy,
        "exact": float(sum(exact_rows) / len(exact_rows)),
        "count": float(len(exact_rows)),
        "rows": rows,
    }


@torch.no_grad()
def answer_hidden_matrix(
    model: DecoderTransformer,
    prompts: Sequence[dict[str, str]],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    batch = make_batch(prompts, tokenizer=tokenizer, max_seq_len=max_seq_len, pad_id=pad_id).to(device)
    _logits, hidden = model(batch.inputs)
    return select_answer_rows(hidden, batch.mask).detach().to(device="cpu", dtype=torch.float32)


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"CKA matrices must have the same shape, got {tuple(left.shape)} vs {tuple(right.shape)}.")
    if left.ndim != 2 or left.shape[0] < 2:
        raise ValueError("CKA requires a matrix with at least two rows.")
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(left.T @ right) ** 2
    left_norm = torch.linalg.matrix_norm(left.T @ left)
    right_norm = torch.linalg.matrix_norm(right.T @ right)
    value = cross / (left_norm * right_norm).clamp_min(1e-12)
    return float(value.clamp(0.0, 1.0).cpu())


def relative_hidden_drift(reference: torch.Tensor, current: torch.Tensor) -> float:
    if reference.shape != current.shape:
        raise ValueError("Hidden drift matrices must have matching shapes.")
    return float(
        (
            torch.linalg.vector_norm(current - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).cpu()
    )


def build_eval_groups(
    records: Sequence[EntityRecord],
    corrected_names: Sequence[str],
    current_city: dict[str, str],
    current_currency: dict[str, str],
) -> dict[str, tuple[dict[str, str], ...]]:
    corrected_name_set = set(corrected_names)
    direct = [
        qa(f"Where does {name} live now?", current_city[name])
        for name in corrected_names
    ]
    paraphrase = [
        qa(f"What city is {name} in?", current_city[name])
        for name in corrected_names
    ]
    heldout_paraphrase = [
        qa(f"Which city is {name} currently in?", current_city[name])
        for name in corrected_names
    ]
    ripple = [
        qa(f"What currency does {name} use now?", current_currency[name])
        for name in corrected_names
    ]
    history = [qa(f"Where did {record.name} live in the original record?", record.old_city) for record in records]
    locality = [qa(f"What color does {record.name} like?", record.color) for record in records]
    stable_current = [
        qa(f"Where does {record.name} live now?", record.old_city)
        for record in records
        if record.name not in corrected_name_set
    ]
    if stable_current:
        locality.extend(stable_current)
    return {
        "direct": tuple(direct),
        "paraphrase": tuple(paraphrase),
        "heldout_para": tuple(heldout_paraphrase),
        "ripple": tuple(ripple),
        "history": tuple(history),
        "locality": tuple(locality),
    }


def evaluate_method(
    model: DecoderTransformer,
    base_model: DecoderTransformer,
    records: Sequence[EntityRecord],
    corrected_names: Sequence[str],
    current_city: dict[str, str],
    current_currency: dict[str, str],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    pad_id: int,
    device: torch.device,
) -> dict[str, object]:
    groups = build_eval_groups(records, corrected_names, current_city, current_currency)
    metrics = {
        name: evaluate_prompts(
            model,
            prompts,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
            device=device,
        )
        for name, prompts in groups.items()
        if prompts
    }
    geometry_prompts = tuple(prompt for prompts in groups.values() for prompt in prompts)
    reference_hidden = answer_hidden_matrix(
        base_model,
        geometry_prompts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        device=device,
    )
    current_hidden = answer_hidden_matrix(
        model,
        geometry_prompts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pad_id=pad_id,
        device=device,
    )
    return {
        "behavior": metrics,
        "geometry": {
            "answer_hidden_cka_vs_base": linear_cka(reference_hidden, current_hidden),
            "answer_hidden_relative_drift": relative_hidden_drift(reference_hidden, current_hidden),
        },
    }


def instantiate_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> DecoderTransformer:
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    ).to(device)
    model.configure_trainability(train_embeddings=args.train_embeddings)
    return model


def build_protected_references(
    model: DecoderTransformer,
    groups: Sequence[PromptGroup],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
    pad_id: int,
    device: torch.device,
) -> list[tuple[str, TensorBatch, torch.Tensor, torch.Tensor]]:
    references: list[tuple[str, TensorBatch, torch.Tensor, torch.Tensor]] = []
    for group in groups:
        batch = make_batch(group.prompts, tokenizer=tokenizer, max_seq_len=max_seq_len, pad_id=pad_id).to(device)
        reference_logits, reference_hidden = capture_answer_reference(model, batch)
        references.append((group.name, batch, reference_logits, reference_hidden))
    if not references:
        raise RuntimeError("No protected references were captured.")
    return references


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = require_token_id(tokenizer, "[PAD]")
    records = build_records()
    require_currency_consistency(records)
    records_by_name = record_by_name(records)
    stage_plan = build_stage_plan(
        records,
        stages=args.stages,
        stress_mode=args.stress_mode,
    )
    if not stage_plan:
        raise RuntimeError("No correction stages selected.")

    base_prompt_batch = make_batch(
        base_prompts(records),
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        pad_id=pad_id,
    ).to(device)
    base_model = instantiate_model(args, tokenizer.get_vocab_size(), device)
    train_answer_loss(
        base_model,
        base_prompt_batch,
        epochs=args.base_epochs,
        lr=args.base_lr,
        batch_size=args.batch_size,
        max_gradient_norm=args.max_gradient_norm,
        label="base tokenized semantic pretrain",
        progress=not args.no_progress,
    )
    base_model.eval()

    methods: dict[str, DecoderTransformer] = {
        "naive": copy.deepcopy(base_model),
        "semantic_invariant_tangent": copy.deepcopy(base_model),
    }
    traces: dict[str, list[dict[str, object]]] = {method: [] for method in methods}
    current_city_by_method: dict[str, dict[str, str]] = {
        method: {record.name: record.old_city for record in records}
        for method in methods
    }
    current_currency_by_method: dict[str, dict[str, str]] = {
        method: {record.name: record.old_currency for record in records}
        for method in methods
    }
    corrected_names_by_method: dict[str, list[str]] = {method: [] for method in methods}

    for stage in stage_plan:
        if stage.entity not in records_by_name:
            raise RuntimeError(f"Stage references unknown entity {stage.entity!r}.")
        candidate_batch = make_batch(
            candidate_prompts(stage),
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            pad_id=pad_id,
        ).to(device)
        naive_trace = train_answer_loss(
            methods["naive"],
            candidate_batch,
            epochs=args.cl_epochs,
            lr=args.cl_lr,
            batch_size=args.batch_size,
            max_gradient_norm=args.max_gradient_norm,
            label=f"naive stage {stage.index}",
            progress=not args.no_progress,
        )
        current_city_by_method["naive"][stage.entity] = stage.target_city
        current_currency_by_method["naive"][stage.entity] = stage.target_currency
        if stage.entity not in corrected_names_by_method["naive"]:
            corrected_names_by_method["naive"].append(stage.entity)
        traces["naive"].append(
            {
                "stage": stage.index,
                "entity": stage.entity,
                "target_city": stage.target_city,
                "target_currency": stage.target_currency,
                "repeated_entity": stage.repeated_entity,
                "trace": naive_trace,
            }
        )

        protected = build_protected_references(
            methods["semantic_invariant_tangent"],
            protected_prompts(
                records,
                current_city_by_method["semantic_invariant_tangent"],
                current_currency_by_method["semantic_invariant_tangent"],
                released_current=(stage.entity,),
            ),
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            pad_id=pad_id,
            device=device,
        )
        invariant_trace = invariant_tangent_update(
            methods["semantic_invariant_tangent"],
            candidate_batch,
            protected,
            epochs=args.cl_epochs,
            lr=args.cl_lr,
            projection_damping=args.projection_damping,
            restore_strength=args.restore_strength,
            restore_norm_ratio=args.restore_norm_ratio,
            geometry_restore_weight=args.geometry_restore_weight,
            max_gradient_norm=args.max_gradient_norm,
            constraint_limit=args.constraint_limit,
            label=f"invariant stage {stage.index}",
            progress=not args.no_progress,
        )
        current_city_by_method["semantic_invariant_tangent"][stage.entity] = stage.target_city
        current_currency_by_method["semantic_invariant_tangent"][stage.entity] = stage.target_currency
        if stage.entity not in corrected_names_by_method["semantic_invariant_tangent"]:
            corrected_names_by_method["semantic_invariant_tangent"].append(stage.entity)
        traces["semantic_invariant_tangent"].append(
            {
                "stage": stage.index,
                "entity": stage.entity,
                "target_city": stage.target_city,
                "target_currency": stage.target_currency,
                "repeated_entity": stage.repeated_entity,
                "trace": invariant_trace,
            }
        )

    final: dict[str, object] = {}
    for method, model in methods.items():
        final[method] = evaluate_method(
            model,
            base_model,
            records,
            corrected_names_by_method[method],
            current_city_by_method[method],
            current_currency_by_method[method],
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            pad_id=pad_id,
            device=device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            **vars(args),
            "tokenizer_path": str(args.tokenizer_path),
            "output_dir": str(args.output_dir),
        },
        "records": [asdict(record) for record in records],
        "stages": [asdict(stage) for stage in stage_plan],
        "final_current_state": {
            method: {
                name: {
                    "city": current_city_by_method[method][name],
                    "currency": current_currency_by_method[method][name],
                }
                for name in sorted(current_city_by_method[method])
            }
            for method in methods
        },
        "traces": traces,
        "final": final,
    }
    json_path = args.output_dir / "tokenized_semantic_cl.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("\nTOKENIZED SEMANTIC CL SUMMARY")
    print("=" * 128)
    print(
        "pipeline=tokenizer_ids -> token_embedding + position_embedding -> transformer -> answer logits"
    )
    print(
        f"device={device.type} stages={len(stage_plan)} stress={args.stress_mode} "
        f"vocab={tokenizer.get_vocab_size()} d_model={args.d_model} layers={args.n_layers}"
    )
    print("-" * 128)
    header = (
        f"{'method':>28} {'direct':>10} {'para':>10} {'heldout':>10} {'ripple':>10} "
        f"{'history':>10} {'locality':>10} {'cka':>8} {'drift':>8}"
    )
    print(header)
    for method, method_report in final.items():
        behavior = method_report["behavior"]  # type: ignore[index]
        geometry = method_report["geometry"]  # type: ignore[index]
        def exact(name: str) -> float:
            return float(behavior[name]["exact"])  # type: ignore[index]
        print(
            f"{method:>28} "
            f"{exact('direct'):10.4f} "
            f"{exact('paraphrase'):10.4f} "
            f"{exact('heldout_para'):10.4f} "
            f"{exact('ripple'):10.4f} "
            f"{exact('history'):10.4f} "
            f"{exact('locality'):10.4f} "
            f"{float(geometry['answer_hidden_cka_vs_base']):8.4f} "
            f"{float(geometry['answer_hidden_relative_drift']):8.4f}"
        )
    invariant_stage_rows = traces["semantic_invariant_tangent"]
    if invariant_stage_rows:
        print("\nINVARIANT-TANGENT STAGE PLASTICITY")
        print("-" * 128)
        print(f"{'stage':>6} {'entity':>10} {'repeat':>8} {'target':>10} {'safe_min':>10} {'safe_last':>10} {'loss_last':>12}")
        for row in invariant_stage_rows:
            trace = row["trace"]  # type: ignore[index]
            safe_values = [float(item["safe_fraction"]) for item in trace]  # type: ignore[index]
            print(
                f"{int(row['stage']):6d} "
                f"{str(row['entity']):>10} "
                f"{str(bool(row['repeated_entity'])):>8} "
                f"{str(row['target_city']):>10} "
                f"{min(safe_values):10.4f} "
                f"{safe_values[-1]:10.4f} "
                f"{float(trace[-1]['candidate_loss']):12.5f}"  # type: ignore[index]
            )
    print(f"wrote_json={json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis/gco-tiny-tokenized-semantic-cl-seed0"))
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--stress-mode", action="store_true")
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=96)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--base-epochs", type=int, default=250)
    parser.add_argument("--cl-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=2e-3)
    parser.add_argument("--cl-lr", type=float, default=2e-2)
    parser.add_argument("--projection-damping", type=float, default=1e-3)
    parser.add_argument("--restore-strength", type=float, default=0.3)
    parser.add_argument("--restore-norm-ratio", type=float, default=1.0)
    parser.add_argument("--geometry-restore-weight", type=float, default=2.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--constraint-limit", type=int, default=8)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
