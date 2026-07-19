"""Run bounded trace-conditioned continual learning on a real tiny transformer.

The experiment loads the fitted 5,000-word mixed-language checkpoint, then
processes two stages containing natural book continuation, repeated novel
facts, corrections to old facts, and isolated random noise. A fixed-size bank
of executable token-window medoids is reorganized from co-moving residual and
target representations. No evaluation category enters trace assignment or the
weight-update operator.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_mini_cl_world_demo import (
    flat_autograd_gradient,
    project_gradient_away_from_constraints,
)
from experiments.gco_math.gco_native_scratch_transformer import (
    GCONativeTransformer,
    NativeGCOConfig,
)
from experiments.gco_math.gco_prepare_tiny_cl_base import build_lm_windows
from experiments.gco_math.gco_storage_capacity_sweep import (
    build_text,
    fact_sentences,
    first_words,
    load_book_text,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    set_only_native_weights_trainable,
    trainable_weight_parameters,
)
from experiments.gco_math.gco_tiny_functional_dependency_field import (
    block_normalize,
    effective_rank,
    synchronize_device,
)
from experiments.gco_math.gco_tiny_recurrent_trace_field import (
    EvidenceMoments,
    fit_functional_trace_field,
    functional_attention,
)
from experiments.gco_math.gco_tiny_trace_invariant_tangent_integration import (
    apply_flat_update,
    bounded_restore,
    recurrence_probability,
    reconstruction_confidence,
)


@dataclass(frozen=True)
class TextWindows:
    inputs: torch.Tensor
    targets: torch.Tensor
    groups: tuple[str, ...]


@dataclass(frozen=True)
class FactQuery:
    input_ids: tuple[int, ...]
    old_target_id: int
    new_target_id: int


@dataclass
class ExecutableTraceBank:
    inputs: torch.Tensor
    targets: torch.Tensor
    groups: tuple[str, ...]
    centers: torch.Tensor
    masses: torch.Tensor
    reference_logits: torch.Tensor
    reference_states: torch.Tensor

    def clone(self) -> "ExecutableTraceBank":
        return ExecutableTraceBank(
            inputs=self.inputs.clone(),
            targets=self.targets.clone(),
            groups=tuple(self.groups),
            centers=self.centers.clone(),
            masses=self.masses.clone(),
            reference_logits=self.reference_logits.clone(),
            reference_states=self.reference_states.clone(),
        )


@dataclass
class PendingTraceCommit:
    inputs: torch.Tensor
    targets: torch.Tensor
    groups: tuple[str, ...]
    centers: torch.Tensor
    masses: torch.Tensor


@dataclass
class TraceControl:
    write_weights: torch.Tensor
    protection_weights: torch.Tensor
    pending: PendingTraceCommit
    report: dict[str, Any]


@dataclass
class ConstraintBasis:
    rows: list[torch.Tensor]
    measurement_matrix: torch.Tensor
    report: dict[str, Any]


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "seq_len",
        "stride",
        "base_candidate_windows",
        "stage_book_windows",
        "stage_fact_windows",
        "noise_windows",
        "geometry_pairs",
        "trace_slots",
        "trace_steps",
        "restarts",
        "cl_epochs",
        "dependency_refresh",
        "dependency_rank",
        "correction_count",
        "novel_fact_count",
        "print_every",
    ):
        positive_int(name, getattr(args, name))
    for name in (
        "cl_lr",
        "grad_clip",
        "attention_scale",
        "observation_sigma",
        "encoding_span",
        "concept_radius",
        "trace_lr",
        "ambiguity_weight",
        "protection_power",
        "projection_damping",
        "restore_strength",
        "restore_bound_fraction",
        "behavior_block_weight",
        "geometry_block_weight",
        "feature_block_weight",
        "dependency_rank_tolerance",
        "temperature",
        "trace_surprise_temperature",
        "token_surprise_power",
        "minimum_trace_mass",
    ):
        positive_float(name, getattr(args, name))
    if not 0.0 < args.dependency_energy <= 1.0:
        raise ValueError("dependency_energy must be in (0, 1].")
    if args.novel_fact_start < 0:
        raise ValueError("novel_fact_start must be non-negative.")
    if args.stage_fact_windows < max(args.correction_count, args.novel_fact_count):
        raise ValueError(
            "stage_fact_windows must cover every correction and novel fact at least once."
        )
    for path_name in ("checkpoint", "tokenizer_path", "book_path"):
        path = getattr(args, path_name)
        if not path.exists():
            raise FileNotFoundError(f"{path_name} does not exist: {path}")


def instantiate_model(checkpoint: dict[str, Any], device: torch.device) -> GCONativeTransformer:
    required = {"model_state_dict", "native_gco_config", "model_config"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint missing fields: {sorted(missing)}.")
    model_config = checkpoint["model_config"]
    if {"layers", "heads"}.issubset(model_config):
        n_layers = int(model_config["layers"])
        n_heads = int(model_config["heads"])
    elif {"n_layers", "n_heads"}.issubset(model_config):
        n_layers = int(model_config["n_layers"])
        n_heads = int(model_config["n_heads"])
    else:
        raise RuntimeError("Checkpoint model_config has no recognized layer/head schema.")
    cfg = NativeGCOConfig(**checkpoint["native_gco_config"])
    model = GCONativeTransformer(
        vocab_size=int(model_config["vocab_size"]),
        d_model=int(model_config["d_model"]),
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=int(model_config["d_ff"]),
        max_seq_len=int(model_config["max_seq_len"]),
        cfg=cfg,
    ).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch: missing={missing_keys}, unexpected={unexpected_keys}."
        )
    set_only_native_weights_trainable(model)
    return model


def forward_with_states(
    model: GCONativeTransformer,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [batch, seq], got {tokens.shape}.")
    batch, seq_len = tokens.shape
    if seq_len > model.max_seq_len:
        raise ValueError(f"seq_len={seq_len} exceeds model maximum {model.max_seq_len}.")
    positions = torch.arange(seq_len, device=tokens.device).reshape(1, seq_len).expand(batch, seq_len)
    hidden = model.token_embedding(tokens) + model.position_embedding(positions)
    states: dict[str, torch.Tensor] = {"embed": hidden}
    for index, block in enumerate(model.blocks):
        hidden = block(hidden)
        states[f"block_{index}"] = hidden
    final = model.ln_f(hidden)
    states["final"] = final
    return model.lm_head(final), states


def windows_from_text(
    text: str,
    *,
    tokenizer: Tokenizer,
    seq_len: int,
    stride: int,
    max_windows: int,
    group: str,
) -> TextWindows:
    token_ids = tokenizer.encode(text).ids
    inputs, targets = build_lm_windows(
        token_ids,
        seq_len=seq_len,
        stride=stride,
        max_windows=max_windows,
    )
    return TextWindows(inputs=inputs, targets=targets, groups=tuple(group for _ in range(inputs.shape[0])))


def combine_windows(parts: list[TextWindows]) -> TextWindows:
    if not parts:
        raise ValueError("Cannot combine zero window sets.")
    seq_len = parts[0].inputs.shape[1]
    for part in parts:
        if part.inputs.shape != part.targets.shape or part.inputs.shape[1] != seq_len:
            raise ValueError("Window parts have incompatible shapes.")
        if part.inputs.shape[0] != len(part.groups):
            raise ValueError("Window group count does not match tensor rows.")
    return TextWindows(
        inputs=torch.cat([part.inputs for part in parts], dim=0),
        targets=torch.cat([part.targets for part in parts], dim=0),
        groups=tuple(group for part in parts for group in part.groups),
    )


def word_span(text: str, start: int, count: int) -> str:
    words = text.split()
    end = start + count
    if start < 0 or end > len(words):
        raise ValueError(f"Requested word span [{start}, {end}) from {len(words)} words.")
    return " ".join(words[start:end])


def corrected_fact_text(
    count: int,
    tokenizer: Tokenizer,
) -> tuple[str, str, list[FactQuery], list[str], list[str]]:
    rows = fact_sentences()
    original: list[str] = []
    corrected: list[str] = []
    queries: list[FactQuery] = []
    original_blocks: list[str] = []
    corrected_blocks: list[str] = []
    for index in range(count):
        source = rows[index * 4 : index * 4 + 4]
        donor = rows[(index + count) * 4 : (index + count) * 4 + 4]
        if len(source) != 4 or len(donor) != 4:
            raise RuntimeError("Fact corpus does not contain the requested correction range.")
        old_destination = source[3].split("Answer: ", 1)[1].rstrip(".")
        new_destination = donor[3].split("Answer: ", 1)[1].rstrip(".")
        original.extend(source)
        corrected_source = [row.replace(old_destination, new_destination) for row in source]
        corrected.extend(corrected_source)
        original_blocks.append(" ".join(source))
        corrected_blocks.append(" ".join(corrected_source))
        person = source[0].split()[0]
        prompt_ids = tokenizer.encode(
            f"Question: What can {person} open? Answer: the"
        ).ids
        old_core = old_destination.removeprefix("the ")
        new_core = new_destination.removeprefix("the ")
        old_ids = tokenizer.encode(f" {old_core}").ids
        new_ids = tokenizer.encode(f" {new_core}").ids
        if not prompt_ids or not old_ids or not new_ids:
            raise RuntimeError("Tokenizer returned an empty correction query component.")
        queries.append(
            FactQuery(
                input_ids=tuple(prompt_ids),
                old_target_id=int(old_ids[0]),
                new_target_id=int(new_ids[0]),
            )
        )
    return " ".join(original), " ".join(corrected), queries, original_blocks, corrected_blocks


def novel_fact_text(start: int, count: int) -> tuple[str, list[str]]:
    rows = fact_sentences()
    selected = rows[start * 4 : (start + count) * 4]
    if len(selected) != count * 4:
        raise RuntimeError("Fact corpus does not contain the requested novel range.")
    blocks = [" ".join(selected[index * 4 : index * 4 + 4]) for index in range(count)]
    return " ".join(selected), blocks


def balanced_block_windows(
    blocks: list[str],
    *,
    tokenizer: Tokenizer,
    seq_len: int,
    stride: int,
    max_windows: int,
    group: str,
) -> TextWindows:
    if max_windows < len(blocks):
        raise ValueError("max_windows must provide at least one window per fact block.")
    per_block_limit = math.ceil(max_windows / len(blocks))
    encoded: list[TextWindows] = []
    for block in blocks:
        token_count = len(tokenizer.encode(block).ids)
        available = len(range(0, token_count - seq_len, stride))
        if available <= 0:
            raise ValueError("Fact block is too short to produce a language-model window.")
        all_windows = windows_from_text(
            block,
            tokenizer=tokenizer,
            seq_len=seq_len,
            stride=stride,
            max_windows=available,
            group=group,
        )
        selected_count = min(per_block_limit, all_windows.inputs.shape[0])
        selected = torch.linspace(
            0,
            all_windows.inputs.shape[0] - 1,
            selected_count,
        ).round().to(dtype=torch.long)
        if selected.unique().numel() != selected_count:
            raise RuntimeError("Balanced fact-window selection produced duplicate indices.")
        encoded.append(
            TextWindows(
                inputs=all_windows.inputs[selected],
                targets=all_windows.targets[selected],
                groups=tuple(group for _ in range(selected_count)),
            )
        )
    input_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    groups: list[str] = []
    for round_index in range(per_block_limit):
        for part in encoded:
            if round_index < part.inputs.shape[0]:
                input_rows.append(part.inputs[round_index])
                target_rows.append(part.targets[round_index])
                groups.append(group)
                if len(input_rows) >= max_windows:
                    break
        if len(input_rows) >= max_windows:
            break
    if len(input_rows) < len(blocks):
        raise RuntimeError("Balanced fact windows failed to represent every fact block.")
    return TextWindows(
        inputs=torch.stack(input_rows),
        targets=torch.stack(target_rows),
        groups=tuple(groups),
    )


def random_noise_windows(
    *,
    count: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> TextWindows:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.randint(vocab_size, (count, seq_len), generator=generator)
    targets = torch.randint(vocab_size, (count, seq_len), generator=generator)
    return TextWindows(inputs=inputs, targets=targets, groups=tuple("noise" for _ in range(count)))


def build_staged_data(
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    vocab_size: int,
) -> tuple[TextWindows, list[TextWindows], dict[str, TextWindows], list[FactQuery]]:
    book = load_book_text(args.book_path)
    base_book = first_words(book, args.base_book_words)
    base_facts = " ".join(fact_sentences())
    base_candidates = combine_windows(
        [
            windows_from_text(
                base_book,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.base_candidate_windows // 2,
                group="base_book",
            ),
            windows_from_text(
                first_words(base_facts, args.base_fact_words),
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.base_candidate_windows - args.base_candidate_windows // 2,
                group="base_fact",
            ),
        ]
    )
    (
        original_facts,
        corrected_facts,
        correction_queries,
        _original_blocks,
        corrected_blocks,
    ) = corrected_fact_text(
        args.correction_count,
        tokenizer,
    )
    novel_facts, novel_blocks = novel_fact_text(
        args.novel_fact_start,
        args.novel_fact_count,
    )
    stage2 = combine_windows(
        [
            windows_from_text(
                word_span(book, args.base_book_words, args.stage_book_words),
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.stage_book_windows,
                group="new_book_1",
            ),
            balanced_block_windows(
                novel_blocks,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.stage_fact_windows,
                group="novel_fact",
            ),
        ]
    )
    stage3 = combine_windows(
        [
            windows_from_text(
                word_span(book, args.base_book_words + args.stage_book_words, args.stage_book_words),
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.stage_book_windows,
                group="new_book_2",
            ),
            balanced_block_windows(
                corrected_blocks,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                max_windows=args.stage_fact_windows,
                group="corrected_fact",
            ),
            random_noise_windows(
                count=args.noise_windows,
                seq_len=args.seq_len,
                vocab_size=vocab_size,
                seed=args.seed + 7001,
            ),
        ]
    )
    evaluation = {
        "stable_book": windows_from_text(
            base_book,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.eval_windows,
            group="stable_book",
        ),
        "obsolete_fact": windows_from_text(
            original_facts,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.eval_windows,
            group="obsolete_fact",
        ),
        "corrected_fact": windows_from_text(
            corrected_facts,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.eval_windows,
            group="corrected_fact",
        ),
        "novel_fact": windows_from_text(
            novel_facts,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.eval_windows,
            group="novel_fact",
        ),
        "new_book": windows_from_text(
            word_span(book, args.base_book_words, args.stage_book_words * 2),
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.eval_windows,
            group="new_book",
        ),
        "noise": random_noise_windows(
            count=args.noise_windows,
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            seed=args.seed + 7001,
        ),
    }
    return base_candidates, [stage2, stage3], evaluation, correction_queries


@torch.no_grad()
def encode_trace_evidence(
    model: GCONativeTransformer,
    windows: TextWindows,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    inputs = windows.inputs.to(device)
    targets = windows.targets.to(device)
    logits, states = forward_with_states(model, inputs)
    target_probabilities = F.softmax(logits, dim=-1).gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    surprise = -torch.log(target_probabilities.clamp_min(1e-12))
    token_weights = torch.softmax(
        surprise / args.trace_surprise_temperature,
        dim=1,
    )
    target_tokens = model.token_embedding(targets)
    predicted_tokens = model.token_embedding(logits.argmax(dim=-1))
    residual = F.normalize(
        (states["final"] * token_weights.unsqueeze(-1)).sum(dim=1),
        dim=1,
    )
    target_embedding = F.normalize(
        (target_tokens * token_weights.unsqueeze(-1)).sum(dim=1),
        dim=1,
    )
    prediction_error = F.normalize(
        ((target_tokens - predicted_tokens) * token_weights.unsqueeze(-1)).sum(dim=1),
        dim=1,
    )
    evidence = torch.cat(
        [residual, target_embedding, prediction_error],
        dim=1,
    ).to(dtype=torch.float32)
    if not torch.isfinite(evidence).all():
        raise FloatingPointError("Text trace evidence contains non-finite values.")
    return evidence


def evidence_moments(vectors: torch.Tensor, weights: torch.Tensor) -> EvidenceMoments:
    dimension = vectors.shape[1]
    if weights.shape != (vectors.shape[0],):
        raise ValueError("Evidence weights do not match vectors.")
    return EvidenceMoments(
        means=vectors,
        covariances=torch.zeros(
            vectors.shape[0], dimension, dimension, device=vectors.device, dtype=vectors.dtype
        ),
        weights=weights,
    )


@torch.no_grad()
def capture_references(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, states = forward_with_states(model, inputs.to(device))
    return logits[:, -1].detach(), states["final"][:, -1].detach()


def choose_medoids(
    centers: torch.Tensor,
    candidate_vectors: torch.Tensor,
    *,
    support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    costs = ((centers.unsqueeze(1) - candidate_vectors.unsqueeze(0)) ** 2).sum(dim=2)
    if support is None:
        assignment_costs = costs
    else:
        if support.shape != (candidate_vectors.shape[0],):
            raise ValueError("Medoid support does not match candidate count.")
        assignment_costs = costs / support.unsqueeze(0).clamp_min(1e-6)
    rows, columns = linear_sum_assignment(assignment_costs.detach().cpu().numpy())
    if len(rows) != centers.shape[0] or sorted(rows.tolist()) != list(range(centers.shape[0])):
        raise RuntimeError("Medoid assignment did not cover every trace slot.")
    return (
        torch.tensor(columns, device=centers.device, dtype=torch.long),
        costs[torch.tensor(rows, device=centers.device), torch.tensor(columns, device=centers.device)],
    )


def initialize_trace_bank(
    model: GCONativeTransformer,
    candidates: TextWindows,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> ExecutableTraceBank:
    vectors = encode_trace_evidence(model, candidates, args=args, device=device)
    solution = fit_functional_trace_field(
        evidence_moments(vectors, torch.ones(vectors.shape[0], device=device)),
        args=args,
        stage=1,
        previous_centers=None,
    )
    medoid_indices, _costs = choose_medoids(solution.centers, vectors)
    attention = functional_attention(vectors, solution.centers, attention_scale=args.attention_scale)
    masses = attention.sum(dim=0)
    inputs = candidates.inputs[medoid_indices.cpu()].clone()
    targets = candidates.targets[medoid_indices.cpu()].clone()
    groups = tuple(candidates.groups[index] for index in medoid_indices.cpu().tolist())
    reference_logits, reference_states = capture_references(model, inputs, device=device)
    return ExecutableTraceBank(
        inputs=inputs,
        targets=targets,
        groups=groups,
        centers=solution.centers.detach(),
        masses=masses.detach(),
        reference_logits=reference_logits,
        reference_states=reference_states,
    )


def propose_trace_update(
    model: GCONativeTransformer,
    bank: ExecutableTraceBank,
    current: TextWindows,
    *,
    stage: int,
    args: argparse.Namespace,
    device: torch.device,
    write_override: torch.Tensor | None = None,
) -> TraceControl:
    bank_windows = TextWindows(bank.inputs, bank.targets, bank.groups)
    old_vectors = encode_trace_evidence(model, bank_windows, args=args, device=device)
    current_vectors = encode_trace_evidence(model, current, args=args, device=device)
    familiarity, _ = reconstruction_confidence(
        current_vectors, old_vectors, attention_scale=args.attention_scale
    )
    recurrence = recurrence_probability(current_vectors, attention_scale=args.attention_scale)
    inferred_write = 1.0 - (1.0 - familiarity) * (1.0 - recurrence)
    if write_override is None:
        write_weights = inferred_write
    else:
        if write_override.shape != inferred_write.shape:
            raise ValueError(
                "Explicit trace write weights do not match the incoming evidence."
            )
        if not torch.isfinite(write_override).all() or (write_override < 0.0).any() or (
            write_override > 1.0
        ).any():
            raise ValueError("Explicit trace write weights must be finite and in [0, 1].")
        write_weights = write_override.to(inferred_write)
    vectors = torch.cat([old_vectors, current_vectors], dim=0)
    weights = torch.cat(
        [
            bank.masses.to(device).clamp_min(args.minimum_trace_mass),
            write_weights.clamp_min(1e-6),
        ]
    )
    solution = fit_functional_trace_field(
        evidence_moments(vectors, weights),
        args=args,
        stage=stage,
        previous_centers=old_vectors,
    )
    old_confidence, old_error = reconstruction_confidence(
        old_vectors, solution.centers, attention_scale=args.attention_scale
    )
    protection = old_confidence.pow(args.protection_power)
    medoid_support = torch.cat([protection, write_weights], dim=0)
    medoid_indices, medoid_costs = choose_medoids(
        solution.centers,
        vectors,
        support=medoid_support,
    )
    candidate_inputs = torch.cat([bank.inputs, current.inputs], dim=0)
    candidate_targets = torch.cat([bank.targets, current.targets], dim=0)
    candidate_groups = tuple(bank.groups) + tuple(current.groups)
    attention = functional_attention(vectors, solution.centers, attention_scale=args.attention_scale)
    masses = (attention * weights.unsqueeze(1)).sum(dim=0)
    pending = PendingTraceCommit(
        inputs=candidate_inputs[medoid_indices.cpu()].clone(),
        targets=candidate_targets[medoid_indices.cpu()].clone(),
        groups=tuple(candidate_groups[index] for index in medoid_indices.cpu().tolist()),
        centers=solution.centers.detach(),
        masses=masses.detach(),
    )
    group_report: dict[str, dict[str, float]] = {}
    for group in sorted(set(current.groups)):
        indices = torch.tensor(
            [index for index, value in enumerate(current.groups) if value == group],
            device=device,
        )
        group_report[group] = {
            "write": float(write_weights[indices].mean().detach().cpu()),
            "familiarity": float(familiarity[indices].mean().detach().cpu()),
            "recurrence": float(recurrence[indices].mean().detach().cpu()),
        }
    return TraceControl(
        write_weights=write_weights.detach(),
        protection_weights=protection.detach(),
        pending=pending,
        report={
            "stage": stage,
            "current_groups": group_report,
            "old_groups": list(bank.groups),
            "old_confidence": [float(value) for value in old_confidence.detach().cpu()],
            "old_protection": [float(value) for value in protection.detach().cpu()],
            "old_error": [float(value) for value in old_error.detach().cpu()],
            "committed_groups": list(pending.groups),
            "medoid_cost_mean": float(medoid_costs.mean().detach().cpu()),
            "stored_tokens": int(pending.inputs.numel()),
            "write_override": write_override is not None,
        },
    )


def commit_trace_bank(
    model: GCONativeTransformer,
    pending: PendingTraceCommit,
    *,
    device: torch.device,
) -> ExecutableTraceBank:
    logits, states = capture_references(model, pending.inputs, device=device)
    return ExecutableTraceBank(
        inputs=pending.inputs,
        targets=pending.targets,
        groups=pending.groups,
        centers=pending.centers,
        masses=pending.masses,
        reference_logits=logits,
        reference_states=states,
    )


def weighted_language_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    surprise_power: float,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    with torch.no_grad():
        target_probability = torch.exp(-per_token).clamp(0.0, 1.0)
        token_weights = (1.0 - target_probability).pow(surprise_power)
        token_weights = token_weights / token_weights.mean(dim=1, keepdim=True).clamp_min(1e-12)
    per_window = (per_token * token_weights).mean(dim=1)
    return (per_window * weights).sum() / weights.sum().clamp_min(1e-12)


def pair_distances(states: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
    return {
        (left, right): (states[left] - states[right]).square().sum()
        for left in range(states.shape[0])
        for right in range(left + 1, states.shape[0])
    }


def protection_losses(
    model: GCONativeTransformer,
    bank: ExecutableTraceBank,
    protection: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, states = forward_with_states(model, bank.inputs.to(device))
    current_logits = logits[:, -1]
    current_states = states["final"][:, -1]
    teacher_probs = F.softmax(bank.reference_logits.to(device) / args.temperature, dim=-1)
    current_log_probs = F.log_softmax(current_logits / args.temperature, dim=-1)
    per_probe_kl = F.kl_div(current_log_probs, teacher_probs, reduction="none").sum(dim=1)
    behavior_loss = (
        per_probe_kl * protection
    ).sum() / protection.sum().clamp_min(1e-12) * (args.temperature**2)
    state_error = (current_states - bank.reference_states.to(device)).square().mean(dim=1)
    state_loss = (state_error * protection).sum() / protection.sum().clamp_min(1e-12)
    return behavior_loss, state_loss, current_logits, current_states


def direct_constraint_blocks(
    *,
    model: GCONativeTransformer,
    bank: ExecutableTraceBank,
    protection: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _logits, states = forward_with_states(model, bank.inputs.to(device))
    final_states = states["final"][:, -1]
    final_logits = model.lm_head(final_states)
    target_ids = bank.targets[:, -1].to(device)
    reference_logits = bank.reference_logits.to(device)
    competitor_logits = reference_logits.clone()
    competitor_logits.scatter_(1, target_ids.unsqueeze(1), float("-inf"))
    competitor_ids = competitor_logits.argmax(dim=1)
    behavior_rows: list[torch.Tensor] = []
    feature_behavior_rows: list[torch.Tensor] = []
    for probe in range(final_logits.shape[0]):
        row_weight = torch.sqrt(protection[probe])
        for token_id in (target_ids[probe], competitor_ids[probe]):
            scalar = row_weight * final_logits[probe, token_id]
            behavior_rows.append(
                flat_autograd_gradient(
                    scalar,
                    parameters,
                    retain_graph=True,
                    require_nonzero=False,
                    label=f"text_behavior_{probe}_{int(token_id)}",
                )
            )
            state_gradient = torch.autograd.grad(
                scalar, final_states, retain_graph=True, allow_unused=False
            )[0]
            feature_behavior_rows.append(state_gradient[probe].detach().to(dtype=torch.float32))
    geometry_rows: list[torch.Tensor] = []
    feature_geometry_rows: list[torch.Tensor] = []
    all_distances = pair_distances(final_states)
    ranked_pairs = sorted(
        all_distances,
        key=lambda pair: float((protection[pair[0]] * protection[pair[1]]).detach().cpu()),
        reverse=True,
    )
    selected_pairs = ranked_pairs[: min(args.geometry_pairs, len(ranked_pairs))]
    if not selected_pairs:
        raise RuntimeError("No protected geometry pairs were selected.")
    for left, right in selected_pairs:
        distance = all_distances[(left, right)]
        pair_weight = torch.sqrt(protection[left] * protection[right])
        scalar = pair_weight * distance
        geometry_rows.append(
            flat_autograd_gradient(
                scalar,
                parameters,
                retain_graph=True,
                require_nonzero=False,
                label=f"text_geometry_{left}_{right}",
            )
        )
        state_gradient = torch.autograd.grad(
            scalar, final_states, retain_graph=True, allow_unused=False
        )[0]
        feature_geometry_rows.extend(
            [state_gradient[left].detach().to(dtype=torch.float32), state_gradient[right].detach().to(dtype=torch.float32)]
        )
    behavior_matrix = block_normalize(
        behavior_rows, block_weight=args.behavior_block_weight, label="text behavior"
    )
    geometry_matrix = block_normalize(
        geometry_rows, block_weight=args.geometry_block_weight, label="text geometry"
    )
    feature_matrix = torch.cat(
        [
            block_normalize(
                feature_behavior_rows,
                block_weight=args.behavior_block_weight,
                label="text feature behavior",
            ),
            block_normalize(
                feature_geometry_rows,
                block_weight=args.geometry_block_weight,
                label="text feature geometry",
            ),
        ],
        dim=0,
    )
    return behavior_matrix, geometry_matrix, feature_matrix, final_states


def energy_rank(singular_values: torch.Tensor, args: argparse.Namespace) -> int:
    positive = singular_values > singular_values[0] * args.dependency_rank_tolerance
    numerical_rank = int(positive.sum().item())
    if numerical_rank <= 0:
        raise RuntimeError("Constraint matrix has zero numerical rank.")
    energy = singular_values[:numerical_rank].square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
    required = int(
        torch.searchsorted(cumulative, cumulative.new_tensor(args.dependency_energy)).item()
    ) + 1
    return min(required, numerical_rank, args.dependency_rank)


def build_constraint_basis(
    *,
    method: str,
    model: GCONativeTransformer,
    bank: ExecutableTraceBank,
    protection: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    args: argparse.Namespace,
    device: torch.device,
) -> ConstraintBasis:
    behavior, geometry, feature_matrix, final_states = direct_constraint_blocks(
        model=model,
        bank=bank,
        protection=protection,
        parameters=parameters,
        args=args,
        device=device,
    )
    direct_matrix = torch.cat([behavior, geometry], dim=0)
    if method == "trace_invariant":
        return ConstraintBasis(
            rows=[row for row in direct_matrix],
            measurement_matrix=direct_matrix.detach(),
            report={"rows": int(direct_matrix.shape[0]), "rank": int(direct_matrix.shape[0])},
        )
    if method != "dependency_field":
        raise ValueError(f"Cannot build constraints for method {method!r}.")
    _u, feature_singular, feature_right = torch.linalg.svd(
        feature_matrix.detach().to(device="cpu", dtype=torch.float32), full_matrices=False
    )
    feature_rank = energy_rank(feature_singular, args)
    feature_directions = feature_right[:feature_rank].to(final_states)
    feature_strengths = (
        feature_singular[:feature_rank] / feature_singular[0]
    ).pow(args.dependency_power).to(final_states)
    family_rows: list[torch.Tensor] = []
    normalized_protection = protection / protection.sum().clamp_min(1e-12)
    for family in range(feature_rank):
        coordinate = (final_states @ feature_directions[family] * normalized_protection).sum()
        family_rows.append(
            flat_autograd_gradient(
                feature_strengths[family] * coordinate,
                parameters,
                retain_graph=True,
                require_nonzero=False,
                label=f"text_feature_family_{family}",
            )
        )
    family_matrix = block_normalize(
        family_rows, block_weight=args.feature_block_weight, label="text feature family"
    )
    matrix = torch.cat([direct_matrix, family_matrix], dim=0)
    gram = (matrix @ matrix.T).detach().to(device="cpu", dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    singular = torch.sqrt(eigenvalues[order].clamp_min(0.0))
    left = eigenvectors[:, order]
    rank = energy_rank(singular, args)
    right = (left[:, :rank].T.to(matrix) @ matrix) / singular[:rank].to(matrix).unsqueeze(1).clamp_min(1e-12)
    strengths = (singular[:rank] / singular[0]).pow(args.dependency_power).to(matrix)
    compressed = strengths.unsqueeze(1) * right
    retained_energy = float(
        (singular[:rank].square().sum() / singular.square().sum().clamp_min(1e-12)).item()
    )
    return ConstraintBasis(
        rows=[row for row in compressed],
        measurement_matrix=matrix.detach(),
        report={
            "rows": int(matrix.shape[0]),
            "rank": rank,
            "retained_energy": retained_energy,
            "effective_rank": effective_rank(singular),
            "feature_rank": feature_rank,
            "feature_effective_rank": effective_rank(feature_singular),
            "singular_values": [float(value) for value in singular],
            "feature_singular_values": [float(value) for value in feature_singular],
        },
    )


def normalized_damage(matrix: torch.Tensor, gradient: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(matrix @ gradient)
            / torch.linalg.vector_norm(gradient).clamp_min(1e-12)
        ).detach().cpu()
    )


def train_stage(
    *,
    method: str,
    model: GCONativeTransformer,
    bank: ExecutableTraceBank | None,
    current: TextWindows,
    control: TraceControl | None,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    parameters = trainable_weight_parameters(model)
    inputs = current.inputs.to(device)
    targets = current.targets.to(device)
    if method == "naive":
        write_weights = torch.ones(inputs.shape[0], device=device)
        protection = None
    else:
        if bank is None or control is None:
            raise RuntimeError(f"Method {method} requires a trace bank and control.")
        write_weights = control.write_weights.to(device)
        protection = control.protection_weights.to(device)
    basis: ConstraintBasis | None = None
    trace: list[dict[str, Any]] = []
    for epoch in range(1, args.cl_epochs + 1):
        logits, _states = forward_with_states(model, inputs)
        new_loss = weighted_language_loss(
            logits,
            targets,
            write_weights,
            surprise_power=args.token_surprise_power,
        )
        if method == "naive":
            final_loss = new_loss
            raw = flat_autograd_gradient(
                final_loss,
                parameters,
                retain_graph=False,
                require_nonzero=True,
                label="text_naive",
            )
            final_gradient = raw
            stats = {"projection_removed_fraction": 0.0, "safe_grad_fraction": 1.0}
            behavior_value = 0.0
            state_value = 0.0
            raw_damage = 0.0
            safe_damage = 0.0
        else:
            behavior_loss, state_loss, _old_logits, _old_states = protection_losses(
                model, bank, protection, args=args, device=device
            )
            if method == "trace_loss_mix":
                final_loss = new_loss + args.restore_strength * (
                    behavior_loss + args.geometry_restore_weight * state_loss
                )
                final_gradient = flat_autograd_gradient(
                    final_loss,
                    parameters,
                    retain_graph=False,
                    require_nonzero=True,
                    label="text_trace_loss_mix",
                )
                stats = {"projection_removed_fraction": 0.0, "safe_grad_fraction": 1.0}
                raw_damage = 0.0
                safe_damage = 0.0
            else:
                if basis is None or (epoch - 1) % args.dependency_refresh == 0:
                    basis = build_constraint_basis(
                        method=method,
                        model=model,
                        bank=bank,
                        protection=protection,
                        parameters=parameters,
                        args=args,
                        device=device,
                    )
                raw = flat_autograd_gradient(
                    new_loss,
                    parameters,
                    retain_graph=True,
                    require_nonzero=True,
                    label=f"text_{method}_new",
                )
                tangent, stats = project_gradient_away_from_constraints(
                    raw_gradient=raw,
                    constraint_gradients=basis.rows,
                    damping=args.projection_damping,
                    solver="gram",
                    rank_tolerance=args.dependency_rank_tolerance,
                    plasticity_audit=False,
                )
                restore_gradient = flat_autograd_gradient(
                    behavior_loss + args.geometry_restore_weight * state_loss,
                    parameters,
                    retain_graph=False,
                    require_nonzero=False,
                    label=f"text_{method}_restore",
                )
                restore = bounded_restore(
                    restore_gradient,
                    tangent,
                    strength=args.restore_strength,
                    bound_fraction=args.restore_bound_fraction,
                )
                final_gradient = tangent + restore
                raw_damage = normalized_damage(basis.measurement_matrix, raw)
                safe_damage = normalized_damage(basis.measurement_matrix, final_gradient)
            behavior_value = float(behavior_loss.detach().cpu())
            state_value = float(state_loss.detach().cpu())
        gradient_norm = apply_flat_update(
            parameters,
            final_gradient,
            learning_rate=args.cl_lr,
            grad_clip=args.grad_clip,
        )
        if epoch in {1, args.cl_epochs} or epoch % args.print_every == 0:
            trace.append(
                {
                    "epoch": epoch,
                    "new_loss": float(new_loss.detach().cpu()),
                    "behavior_loss": behavior_value,
                    "state_loss": state_value,
                    "gradient_norm": gradient_norm,
                    "projection_removed_fraction": stats["projection_removed_fraction"],
                    "safe_grad_fraction": stats["safe_grad_fraction"],
                    "raw_damage": raw_damage,
                    "safe_damage": safe_damage,
                    "basis": None if basis is None else basis.report,
                }
            )
    return trace


@torch.no_grad()
def evaluate_windows(
    model: GCONativeTransformer,
    windows: TextWindows,
    *,
    device: torch.device,
) -> dict[str, float]:
    logits, _states = forward_with_states(model, windows.inputs.to(device))
    targets = windows.targets.to(device)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    predictions = logits.argmax(dim=-1)
    return {
        "loss": float(loss.detach().cpu()),
        "token_accuracy": float((predictions == targets).float().mean().detach().cpu()),
        "last_token_accuracy": float(
            (predictions[:, -1] == targets[:, -1]).float().mean().detach().cpu()
        ),
        "windows": float(windows.inputs.shape[0]),
    }


@torch.no_grad()
def evaluate_correction_queries(
    model: GCONativeTransformer,
    queries: list[FactQuery],
    *,
    device: torch.device,
) -> dict[str, float]:
    if not queries:
        raise ValueError("Correction query set is empty.")
    margins: list[torch.Tensor] = []
    for query in queries:
        token_ids = list(query.input_ids[-model.max_seq_len :])
        inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1]
        margins.append(logits[query.new_target_id] - logits[query.old_target_id])
    values = torch.stack(margins)
    return {
        "new_over_old_fraction": float((values > 0.0).float().mean().detach().cpu()),
        "new_minus_old_margin": float(values.mean().detach().cpu()),
        "queries": float(len(queries)),
    }


@torch.no_grad()
def collect_geometry(
    model: GCONativeTransformer,
    windows: TextWindows,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    _logits, states = forward_with_states(model, windows.inputs.to(device))
    return {
        name: value.reshape(-1, value.shape[-1]).detach().to(device="cpu", dtype=torch.float32)
        for name, value in states.items()
    }


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    numerator = (left.T @ right).square().sum()
    denominator = torch.sqrt((left.T @ left).square().sum() * (right.T @ right).square().sum())
    return float((numerator / denominator.clamp_min(1e-12)).item())


def geometry_report(
    reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    if reference.keys() != current.keys():
        raise RuntimeError("Geometry layer sets differ.")
    report: dict[str, dict[str, float]] = {}
    for layer in reference:
        ref = reference[layer]
        value = current[layer]
        report[layer] = {
            "cka": linear_cka(ref, value),
            "relative_drift": float(
                (torch.linalg.vector_norm(value - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-12)).item()
            ),
        }
    return report


def run_method(
    *,
    method: str,
    base_model: GCONativeTransformer,
    initial_bank: ExecutableTraceBank,
    stages: list[TextWindows],
    evaluation: dict[str, TextWindows],
    correction_queries: list[FactQuery],
    base_geometry: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = copy.deepcopy(base_model)
    bank = initial_bank.clone()
    stage_reports: list[dict[str, Any]] = []
    synchronize_device(device)
    started = time.perf_counter()
    for stage_index, current in enumerate(stages, start=2):
        control = None
        if method != "naive":
            control = propose_trace_update(
                model, bank, current, stage=stage_index, args=args, device=device
            )
        trace = train_stage(
            method=method,
            model=model,
            bank=None if method == "naive" else bank,
            current=current,
            control=control,
            args=args,
            device=device,
        )
        if control is not None:
            bank = commit_trace_bank(model, control.pending, device=device)
        stage_reports.append(
            {
                "stage": stage_index,
                "trace_control": None if control is None else control.report,
                "training": trace,
                "evaluation": {
                    name: evaluate_windows(model, windows, device=device)
                    for name, windows in evaluation.items()
                },
            }
        )
    synchronize_device(device)
    seconds = time.perf_counter() - started
    final_eval = {
        name: evaluate_windows(model, windows, device=device)
        for name, windows in evaluation.items()
    }
    correction_preference = evaluate_correction_queries(
        model,
        correction_queries,
        device=device,
    )
    geometry = geometry_report(
        base_geometry,
        collect_geometry(model, evaluation["stable_book"], device=device),
    )
    return {
        "seconds": seconds,
        "stages": stage_reports,
        "final_evaluation": final_eval,
        "correction_preference": correction_preference,
        "geometry": geometry,
        "final_bank_groups": list(bank.groups) if method != "naive" else [],
        "stored_tokens": int(bank.inputs.numel()) if method != "naive" else 0,
    }


def plot_behavior(results: dict[str, Any], *, output_path: Path) -> None:
    methods = list(results)
    categories = list(results[methods[0]]["final_evaluation"])
    x = torch.arange(len(categories), dtype=torch.float32).numpy()
    width = 0.2
    colors = ["#9aa0a6", "#f59e0b", "#2563eb", "#0f9d58"]
    fig, axis = plt.subplots(figsize=(12.5, 5.2))
    for index, method in enumerate(methods):
        values = [results[method]["final_evaluation"][name]["loss"] for name in categories]
        axis.bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            label=method.replace("_", " "),
            color=colors[index],
        )
    axis.set_xticks(x, categories, rotation=25, ha="right")
    axis.set_ylabel("language-model loss")
    axis.set_title("Real-text behavior after staged continual learning")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_geometry(results: dict[str, Any], *, output_path: Path) -> None:
    methods = list(results)
    layers = list(results[methods[0]]["geometry"])
    x = torch.arange(len(layers), dtype=torch.float32).numpy()
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    for index, method in enumerate(methods):
        axes[0].bar(
            x + (index - 1.5) * width,
            [results[method]["geometry"][layer]["cka"] for layer in layers],
            width=width,
            label=method.replace("_", " "),
        )
        axes[1].bar(
            x + (index - 1.5) * width,
            [results[method]["geometry"][layer]["relative_drift"] for layer in layers],
            width=width,
        )
    axes[0].set_xticks(x, layers, rotation=20)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("linear CKA")
    axes[0].set_title("Stable-text geometry similarity")
    axes[0].legend()
    axes[1].set_xticks(x, layers, rotation=20)
    axes[1].set_ylabel("relative activation drift")
    axes[1].set_title("Stable-text geometry displacement")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trace_control(results: dict[str, Any], *, output_path: Path) -> None:
    method = "dependency_field"
    stages = results[method]["stages"]
    fig, axes = plt.subplots(1, len(stages), figsize=(6.0 * len(stages), 4.3))
    if len(stages) == 1:
        axes = [axes]
    for axis, stage in zip(axes, stages, strict=True):
        control = stage["trace_control"]
        groups = sorted(control["current_groups"])
        axis.bar(groups, [control["current_groups"][group]["write"] for group in groups])
        axis.set_ylim(0.0, 1.02)
        axis.set_ylabel("write probability")
        axis.set_title(f"Stage {stage['stage']} autonomous trace decision")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    args.num_slots = args.trace_slots
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model = instantiate_model(checkpoint, device)
    vocab_size = int(checkpoint["model_config"]["vocab_size"])
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            f"Tokenizer/model vocabulary mismatch: {tokenizer.get_vocab_size()} vs {vocab_size}."
        )
    base_candidates, stages, evaluation, correction_queries = build_staged_data(
        args,
        tokenizer,
        vocab_size,
    )
    if args.trace_slots > base_candidates.inputs.shape[0]:
        raise ValueError("trace_slots exceeds available base candidate windows.")
    initial_bank = initialize_trace_bank(
        base_model, base_candidates, args=args, device=device
    )
    base_geometry = collect_geometry(
        base_model, evaluation["stable_book"], device=device
    )

    print("TINY REAL-TEXT TRACE + FUNCTIONAL DEPENDENCY CL")
    print("=" * 152)
    print(
        f"device={device} params={sum(parameter.numel() for parameter in trainable_weight_parameters(base_model))} "
        f"slots={args.trace_slots} stored_tokens={initial_bank.inputs.numel()} "
        f"stage_windows={[stage.inputs.shape[0] for stage in stages]}"
    )
    results: dict[str, Any] = {}
    for method in ("naive", "trace_loss_mix", "trace_invariant", "dependency_field"):
        print(f"running_method={method}")
        results[method] = run_method(
            method=method,
            base_model=base_model,
            initial_bank=initial_bank,
            stages=stages,
            evaluation=evaluation,
            correction_queries=correction_queries,
            base_geometry=base_geometry,
            args=args,
            device=device,
        )

    print("\nFINAL REAL-TEXT BEHAVIOR")
    print("-" * 152)
    print(
        f"{'method':>18} {'stable':>9} {'newBook':>9} {'novel':>9} {'correct':>9} "
        f"{'obsolete':>9} {'new>old':>9} {'margin':>9} {'finalCKA':>10} {'seconds':>10}"
    )
    for method, result in results.items():
        values = result["final_evaluation"]
        final_cka = result["geometry"]["final"]["cka"]
        preference = result["correction_preference"]
        print(
            f"{method:>18} {values['stable_book']['loss']:9.4f} {values['new_book']['loss']:9.4f} "
            f"{values['novel_fact']['loss']:9.4f} {values['corrected_fact']['loss']:9.4f} "
            f"{values['obsolete_fact']['loss']:9.4f} "
            f"{preference['new_over_old_fraction']:9.4f} "
            f"{preference['new_minus_old_margin']:9.4f} "
            f"{final_cka:10.4f} {result['seconds']:10.3f}"
        )
    dependency_last = results["dependency_field"]["stages"][-1]["training"][-1]
    print("\nDEPENDENCY FIELD FINAL UPDATE")
    print("-" * 152)
    print(
        f"removed={dependency_last['projection_removed_fraction']:.4f} "
        f"safe_fraction={dependency_last['safe_grad_fraction']:.4f} "
        f"damage={dependency_last['raw_damage']:.4f}->{dependency_last['safe_damage']:.4f} "
        f"basis={dependency_last['basis']}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.output_dir / "text_dependency_behavior.png"
    geometry_path = args.output_dir / "text_dependency_geometry.png"
    trace_path = args.output_dir / "text_dependency_trace_control.png"
    output_json = args.output_dir / "text_dependency_cl.json"
    plot_behavior(results, output_path=behavior_path)
    plot_geometry(results, output_path=geometry_path)
    plot_trace_control(results, output_path=trace_path)
    output = {
        "question": (
            "Can a bounded executable trace bank and functional dependency metric preserve real-text "
            "behavior and geometry while a fitted tiny transformer learns staged language data?"
        ),
        "scope": (
            "Two staged updates on a 782k-parameter transformer fitted to 5,000 mixed words. "
            "Evaluation labels never enter trace assignment or gradient control."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "initial_bank_groups": list(initial_bank.groups),
        "initial_stored_tokens": int(initial_bank.inputs.numel()),
        "results": results,
        "plots": {
            "behavior": str(behavior_path),
            "geometry": str(geometry_path),
            "trace_control": str(trace_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_plots={behavior_path},{geometry_path},{trace_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model/checkpoints/gco-storage-capacity-frontier/tiny-mixed-5000w-seed16000.pt"),
    )
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--book-path", type=Path, default=Path("data/real_book/book.txt"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/analysis/gco-tiny-text-dependency-cl-seed0"),
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--base-book-words", type=int, default=2500)
    parser.add_argument("--base-fact-words", type=int, default=2500)
    parser.add_argument("--base-candidate-windows", type=int, default=64)
    parser.add_argument("--stage-book-words", type=int, default=800)
    parser.add_argument("--stage-book-windows", type=int, default=24)
    parser.add_argument("--stage-fact-windows", type=int, default=32)
    parser.add_argument("--eval-windows", type=int, default=24)
    parser.add_argument("--noise-windows", type=int, default=8)
    parser.add_argument("--correction-count", type=int, default=16)
    parser.add_argument("--novel-fact-start", type=int, default=500)
    parser.add_argument("--novel-fact-count", type=int, default=16)
    parser.add_argument("--trace-slots", type=int, default=32)
    parser.add_argument("--trace-steps", type=int, default=120)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--trace-lr", type=float, default=0.03)
    parser.add_argument("--attention-scale", type=float, default=0.8)
    parser.add_argument("--observation-sigma", type=float, default=0.24)
    parser.add_argument("--encoding-span", type=float, default=5.0)
    parser.add_argument("--concept-radius", type=float, default=3.2)
    parser.add_argument("--ambiguity-weight", type=float, default=0.2)
    parser.add_argument("--protection-power", type=float, default=4.0)
    parser.add_argument("--trace-surprise-temperature", type=float, default=1.0)
    parser.add_argument("--cl-epochs", type=int, default=80)
    parser.add_argument("--cl-lr", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--token-surprise-power", type=float, default=1.0)
    parser.add_argument("--minimum-trace-mass", type=float, default=1.0)
    parser.add_argument("--dependency-refresh", type=int, default=20)
    parser.add_argument("--dependency-rank", type=int, default=48)
    parser.add_argument("--dependency-energy", type=float, default=0.98)
    parser.add_argument("--dependency-power", type=float, default=1.0)
    parser.add_argument("--dependency-rank-tolerance", type=float, default=1e-4)
    parser.add_argument("--projection-damping", type=float, default=1e-3)
    parser.add_argument("--restore-strength", type=float, default=0.05)
    parser.add_argument("--restore-bound-fraction", type=float, default=0.5)
    parser.add_argument("--geometry-restore-weight", type=float, default=0.1)
    parser.add_argument("--behavior-block-weight", type=float, default=1.0)
    parser.add_argument("--geometry-block-weight", type=float, default=0.25)
    parser.add_argument("--geometry-pairs", type=int, default=24)
    parser.add_argument("--feature-block-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
