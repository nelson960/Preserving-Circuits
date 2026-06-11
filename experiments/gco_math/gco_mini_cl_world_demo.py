"""Miniature controlled-continual-learning world demo.

This script builds a small but complete CL protocol:

    1. Generate a 3k-5k word base world and staged conversations.
    2. Train one small transformer on the base world.
    3. Clone the base into candidate learners.
    4. Compare naive fine-tuning, controlled consolidation, and joint training.
    5. Evaluate beyond context: no conversation text is included in eval prompts.
    6. Measure residual-stream geometry shift between checkpoints.

This is a benchmark harness, not a final optimizer. It exists to make the CL
claim measurable before we optimize the update mechanism.
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
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_native_scratch_transformer import GCONativeTransformer
from experiments.gco_math.gco_prepare_tiny_cl_base import native_config
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    distillation_kl,
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
    batch_examples,
    collect_example_logits,
    encode_examples,
    evaluate_examples,
    masked_ce_loss,
)
from experiments.gco_math.gco_visualize_tiny_geometry_drift import (
    collect_states,
    effective_rank,
    linear_cka,
    procrustes_align,
)


@dataclass(frozen=True)
class PersonFact:
    person: str
    item: str
    place: str
    role: str
    stage: str
    old_item: str | None = None
    old_place: str | None = None


@dataclass(frozen=True)
class MiniCLProtocol:
    base_text: str
    conversation_text: str
    conversation_stage_texts: list[str]
    base_train: list[QAExample]
    conversation_stage_train: list[list[QAExample]]
    final_train: list[QAExample]
    preserve_eval: list[QAExample]
    guard_eval: list[QAExample]
    changed_eval: list[QAExample]
    new_eval: list[QAExample]
    composition_eval: list[QAExample]
    obsolete_eval: list[QAExample]


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def bounded_float(name: str, value: float, low: float, high: float) -> None:
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}.")


def words(text: str) -> int:
    return len(text.split())


def make_text_example(category: str, sentence: str) -> QAExample:
    clean = sentence.strip()
    if not clean:
        raise ValueError("Cannot build a text example from an empty sentence.")
    return QAExample(stage=0, category=category, prompt="", answer=clean + "\n")


def carry_question(fact: PersonFact, *, category: str) -> QAExample:
    return QAExample(
        stage=0,
        category=category,
        prompt=f"Question: What does {fact.person} carry? Answer:",
        answer=f" {fact.item}.",
    )


def place_question(fact: PersonFact, *, category: str) -> QAExample:
    return QAExample(
        stage=0,
        category=category,
        prompt=f"Question: What does {fact.item} open? Answer:",
        answer=f" {fact.place}.",
    )


def composition_question(fact: PersonFact, *, category: str) -> QAExample:
    return QAExample(
        stage=0,
        category=category,
        prompt=f"Question: What can {fact.person} open? Answer:",
        answer=f" {fact.place}.",
    )


def obsolete_question(fact: PersonFact) -> QAExample:
    if fact.old_item is None:
        raise ValueError(f"obsolete_question requires old_item for {fact.person}.")
    return QAExample(
        stage=0,
        category="obsolete_old_answer",
        prompt=f"Question: What does {fact.person} carry? Answer:",
        answer=f" {fact.old_item}.",
    )


def base_people() -> list[PersonFact]:
    return [
        PersonFact("Alice", "copper key", "tower", "preserve", "base"),
        PersonFact("Bruno", "red lantern", "tunnel", "preserve", "base"),
        PersonFact("Clara", "blue map", "river", "change", "base"),
        PersonFact("Darin", "silver coin", "vault", "change", "base"),
        PersonFact("Elena", "green rope", "bridge", "guard", "base"),
        PersonFact("Farah", "black compass", "forest", "guard", "base"),
        PersonFact("Mira", "glass feather", "aviary", "guard", "base"),
        PersonFact("Noel", "iron whistle", "station", "preserve", "base"),
    ]


def changed_people() -> list[PersonFact]:
    return [
        PersonFact("Clara", "amber pass", "market", "change", "conversation", old_item="blue map", old_place="river"),
        PersonFact("Darin", "ivory token", "gallery", "change", "conversation", old_item="silver coin", old_place="vault"),
    ]


def new_people() -> list[PersonFact]:
    return [
        PersonFact("Galen", "white shell", "harbor", "new", "conversation"),
        PersonFact("Hana", "yellow ring", "garden", "new", "conversation"),
        PersonFact("Iris", "purple flute", "cavern", "new", "conversation"),
        PersonFact("Jules", "orange book", "library", "new", "conversation"),
        PersonFact("Kira", "bronze bell", "chapel", "new", "conversation"),
        PersonFact("Luca", "crystal lens", "observatory", "new", "conversation"),
        PersonFact("Mona", "ruby ticket", "theater", "new", "conversation"),
        PersonFact("Omar", "silver brush", "studio", "new", "conversation"),
        PersonFact("Priya", "golden seed", "orchard", "new", "conversation"),
        PersonFact("Quinn", "navy flag", "citadel", "new", "conversation"),
    ]


def fact_sentences(fact: PersonFact) -> list[str]:
    return [
        f"{fact.person} carries the {fact.item} during the archive rounds.",
        f"The {fact.item} opens the {fact.place} when the city is quiet.",
        f"Because {fact.person} carries the {fact.item}, {fact.person} can open the {fact.place}.",
        f"Archivists remember that {fact.person} is linked with the {fact.item} and the {fact.place}.",
    ]


def conversation_sentences(fact: PersonFact) -> list[str]:
    if fact.role == "change":
        if fact.old_item is None or fact.old_place is None:
            raise ValueError(f"Changed fact for {fact.person} must include old item/place.")
        return [
            f"Conversation update: {fact.person} no longer carries the {fact.old_item}.",
            f"Conversation update: {fact.person} now carries the {fact.item}.",
            f"The {fact.item} opens the {fact.place}; the old {fact.old_item} route is obsolete.",
            f"Future answers about {fact.person} should use the {fact.item}, not the {fact.old_item}.",
        ]
    return [
        f"Conversation update: {fact.person} joined the archive with the {fact.item}.",
        f"The {fact.item} opens the {fact.place} after the night bell.",
        f"When asked what {fact.person} can open, the answer is the {fact.place}.",
    ]


def repeat_to_word_target(sentences: list[str], *, target_words: int) -> str:
    positive_int("target_words", target_words)
    if not sentences:
        raise ValueError("repeat_to_word_target requires at least one sentence.")
    rows: list[str] = []
    index = 0
    while words(" ".join(rows)) < target_words:
        rows.append(sentences[index % len(sentences)])
        index += 1
    return " ".join(rows)


def split_staged_facts(facts: list[PersonFact], *, stage_count: int) -> list[list[PersonFact]]:
    positive_int("stage_count", stage_count)
    if len(facts) < stage_count:
        raise ValueError(f"Cannot split {len(facts)} conversation facts into {stage_count} non-empty stages.")
    stages: list[list[PersonFact]] = [[] for _ in range(stage_count)]
    for index, fact in enumerate(facts):
        stages[index % stage_count].append(fact)
    empty = [index + 1 for index, stage in enumerate(stages) if not stage]
    if empty:
        raise RuntimeError(f"Conversation stage split produced empty stages: {empty}.")
    return stages


def build_protocol(args: argparse.Namespace) -> MiniCLProtocol:
    base_facts = base_people()
    conversation_facts = changed_people() + new_people()
    conversation_fact_stages = split_staged_facts(conversation_facts, stage_count=args.conversation_stages)
    final_base_by_person = {fact.person: fact for fact in base_facts}
    for fact in changed_people():
        final_base_by_person[fact.person] = fact
    final_facts = list(final_base_by_person.values()) + new_people()

    base_sentences = [sentence for fact in base_facts for sentence in fact_sentences(fact)]
    base_text = repeat_to_word_target(base_sentences, target_words=args.base_word_target)

    stage_word_target = max(1, math.ceil(args.conversation_word_target / args.conversation_stages))
    conversation_stage_texts: list[str] = []
    conversation_stage_examples: list[list[QAExample]] = []
    for stage_index, facts in enumerate(conversation_fact_stages, start=1):
        stage_sentences = [sentence for fact in facts for sentence in conversation_sentences(fact)]
        stage_text = repeat_to_word_target(stage_sentences, target_words=stage_word_target)
        conversation_stage_texts.append(stage_text)
        stage_qa = [
            example
            for fact in facts
            for example in (
                carry_question(fact, category=f"{fact.role}_carry"),
                place_question(fact, category=f"{fact.role}_place"),
                composition_question(fact, category=f"{fact.role}_composition"),
            )
        ]
        conversation_stage_examples.append(stage_qa)
    conversation_text = "\n".join(conversation_stage_texts)

    base_qa = [
        example
        for fact in base_facts
        for example in (
            carry_question(fact, category=f"{fact.role}_carry"),
            place_question(fact, category=f"{fact.role}_place"),
            composition_question(fact, category=f"{fact.role}_composition"),
        )
    ]
    final_qa = [
        example
        for fact in final_facts
        for example in (
            carry_question(fact, category=f"{fact.role}_carry"),
            place_question(fact, category=f"{fact.role}_place"),
            composition_question(fact, category=f"{fact.role}_composition"),
        )
    ]

    preserve_eval = [
        example
        for fact in final_facts
        if fact.role == "preserve"
        for example in (
            carry_question(fact, category="preserve"),
            place_question(fact, category="preserve"),
            composition_question(fact, category="preserve"),
        )
    ]
    guard_eval = [
        example
        for fact in final_facts
        if fact.role == "guard"
        for example in (
            carry_question(fact, category="guard"),
            place_question(fact, category="guard"),
            composition_question(fact, category="guard"),
        )
    ]
    changed_eval = [
        example
        for fact in changed_people()
        for example in (
            carry_question(fact, category="changed"),
            place_question(fact, category="changed"),
            composition_question(fact, category="changed"),
        )
    ]
    new_eval = [
        example
        for fact in new_people()
        for example in (
            carry_question(fact, category="new"),
            place_question(fact, category="new"),
            composition_question(fact, category="new"),
        )
    ]
    composition_eval = [
        composition_question(fact, category="composition")
        for fact in final_facts
    ]
    obsolete_eval = [obsolete_question(fact) for fact in changed_people()]

    return MiniCLProtocol(
        base_text=base_text,
        conversation_text=conversation_text,
        conversation_stage_texts=conversation_stage_texts,
        base_train=base_qa,
        conversation_stage_train=conversation_stage_examples,
        final_train=final_qa,
        preserve_eval=preserve_eval,
        guard_eval=guard_eval,
        changed_eval=changed_eval,
        new_eval=new_eval,
        composition_eval=composition_eval,
        obsolete_eval=obsolete_eval,
    )


def lm_window_examples(
    text: str,
    tokenizer: Tokenizer,
    *,
    max_seq_len: int,
    stride: int,
    max_windows: int,
    category: str,
) -> list[EncodedExample]:
    positive_int("max_seq_len", max_seq_len)
    positive_int("stride", stride)
    positive_int("max_windows", max_windows)
    if not text.strip():
        raise ValueError(f"{category} text is empty.")
    token_ids = tokenizer.encode(text).ids
    if len(token_ids) < max_seq_len + 1:
        raise ValueError(
            f"{category} needs at least {max_seq_len + 1} tokens for LM windows, got {len(token_ids)}."
        )
    examples: list[EncodedExample] = []
    for start in range(0, len(token_ids) - max_seq_len, stride):
        window = token_ids[start : start + max_seq_len + 1]
        if len(window) != max_seq_len + 1:
            raise RuntimeError(f"{category} LM window length mismatch at start={start}: {len(window)}.")
        examples.append(
            EncodedExample(
                stage=0,
                category=category,
                prompt=f"<{category}:{start}>",
                answer="",
                input_ids=window[:-1],
                target_ids=window[1:],
                loss_mask=[1.0] * max_seq_len,
            )
        )
        if len(examples) >= max_windows:
            break
    if not examples:
        raise RuntimeError(f"{category} produced no LM windows.")
    return examples


def encode_protocol(
    protocol: MiniCLProtocol,
    tokenizer: Tokenizer,
    *,
    max_seq_len: int,
    lm_stride: int,
    base_lm_max_windows: int,
    conversation_lm_max_windows: int,
) -> dict[str, list[EncodedExample]]:
    base_lm = lm_window_examples(
        protocol.base_text,
        tokenizer,
        max_seq_len=max_seq_len,
        stride=lm_stride,
        max_windows=base_lm_max_windows,
        category="base_lm",
    )
    conversation_lm_stages = [
        lm_window_examples(
            stage_text,
            tokenizer,
            max_seq_len=max_seq_len,
            stride=lm_stride,
            max_windows=conversation_lm_max_windows,
            category=f"conversation_lm_stage_{stage_index}",
        )
        for stage_index, stage_text in enumerate(protocol.conversation_stage_texts, start=1)
    ]
    conversation_qa_stages = [
        encode_examples(stage_examples, tokenizer, max_seq_len=max_seq_len)
        for stage_examples in protocol.conversation_stage_train
    ]
    if len(conversation_lm_stages) != len(conversation_qa_stages):
        raise RuntimeError(
            f"Conversation LM/QA stage count mismatch: lm={len(conversation_lm_stages)} qa={len(conversation_qa_stages)}."
        )
    conversation_stage_train = [
        lm_stage + qa_stage
        for lm_stage, qa_stage in zip(conversation_lm_stages, conversation_qa_stages, strict=True)
    ]
    final_lm = base_lm + [example for stage in conversation_lm_stages for example in stage]
    return {
        "base_lm": base_lm,
        "conversation_lm_stages": [list(stage) for stage in conversation_lm_stages],
        "conversation_qa_stages": [list(stage) for stage in conversation_qa_stages],
        "base_train": base_lm + encode_examples(protocol.base_train, tokenizer, max_seq_len=max_seq_len),
        "conversation_stage_train": conversation_stage_train,
        "update_train": [example for stage in conversation_stage_train for example in stage],
        "final_train": final_lm + encode_examples(protocol.final_train, tokenizer, max_seq_len=max_seq_len),
        "preserve_eval": encode_examples(protocol.preserve_eval, tokenizer, max_seq_len=max_seq_len),
        "guard_eval": encode_examples(protocol.guard_eval, tokenizer, max_seq_len=max_seq_len),
        "changed_eval": encode_examples(protocol.changed_eval, tokenizer, max_seq_len=max_seq_len),
        "new_eval": encode_examples(protocol.new_eval, tokenizer, max_seq_len=max_seq_len),
        "composition_eval": encode_examples(protocol.composition_eval, tokenizer, max_seq_len=max_seq_len),
        "obsolete_eval": encode_examples(protocol.obsolete_eval, tokenizer, max_seq_len=max_seq_len),
    }


def make_model(args: argparse.Namespace, *, vocab_size: int, device: torch.device, seed: int) -> GCONativeTransformer:
    torch.manual_seed(seed)
    cfg = native_config(args)
    model = GCONativeTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        cfg=cfg,
    ).to(device)
    return model


def clone_model(model: GCONativeTransformer, args: argparse.Namespace, *, vocab_size: int, device: torch.device, seed: int) -> GCONativeTransformer:
    clone = make_model(args, vocab_size=vocab_size, device=device, seed=seed)
    missing, unexpected = clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Clone state mismatch: missing={missing}, unexpected={unexpected}.")
    return clone


def make_optimizer(args: argparse.Namespace, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer {args.optimizer!r}.")


def flatten_gradient_list(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    if len(gradients) != len(parameters):
        raise RuntimeError(f"Gradient/parameter length mismatch: {len(gradients)} vs {len(parameters)}.")
    chunks: list[torch.Tensor] = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        if gradient is None:
            chunks.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            if gradient.shape != parameter.shape:
                raise RuntimeError(f"Gradient shape mismatch: gradient={gradient.shape}, parameter={parameter.shape}.")
            chunks.append(gradient.detach().to(dtype=torch.float32).reshape(-1))
    if not chunks:
        raise ValueError("Cannot flatten an empty gradient list.")
    return torch.cat(chunks, dim=0)


def assign_flat_gradient(parameters: list[torch.nn.Parameter], flat_gradient: torch.Tensor) -> None:
    if not parameters:
        raise ValueError("Cannot assign gradients to an empty parameter list.")
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        chunk = flat_gradient[offset : offset + count]
        if chunk.numel() != count:
            raise RuntimeError(
                f"Flat gradient ended early for parameter with {count} entries at offset {offset}."
            )
        parameter.grad = chunk.reshape_as(parameter).to(device=parameter.device, dtype=parameter.dtype).clone()
        offset += count
    if offset != flat_gradient.numel():
        raise RuntimeError(f"Flat gradient had unused entries: used={offset}, total={flat_gradient.numel()}.")


def flat_autograd_gradient(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
    require_nonzero: bool,
    label: str,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flat = flatten_gradient_list(gradients, parameters)
    norm = float(torch.linalg.vector_norm(flat).detach().cpu())
    if require_nonzero and norm <= 1e-12:
        raise RuntimeError(f"{label} gradient is zero; cannot build a controlled update.")
    return flat


def project_gradient_away_from_constraints(
    *,
    raw_gradient: torch.Tensor,
    constraint_gradients: list[torch.Tensor],
    damping: float,
    solver: str,
    rank_tolerance: float,
    plasticity_audit: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_float("projected_update_damping", damping)
    positive_float("plasticity_audit_rank_tolerance", rank_tolerance)
    if solver not in {"sequential", "gram"}:
        raise ValueError(f"Unknown projected solver {solver!r}.")
    active_rows: list[torch.Tensor] = []
    for constraint in constraint_gradients:
        if constraint.shape != raw_gradient.shape:
            raise RuntimeError(
                f"Constraint gradient shape mismatch: constraint={constraint.shape}, raw={raw_gradient.shape}."
            )
        constraint = constraint.to(device=raw_gradient.device, dtype=raw_gradient.dtype)
        norm_sq = torch.dot(constraint, constraint)
        norm = float(torch.sqrt(norm_sq.clamp_min(0.0)).detach().cpu())
        if norm <= 1e-12:
            continue
        active_rows.append(constraint)

    projected = raw_gradient.clone()
    if solver == "sequential":
        for constraint in active_rows:
            norm_sq = torch.dot(constraint, constraint)
            coefficient = torch.dot(projected, constraint) / (norm_sq + damping)
            projected = projected - coefficient * constraint
    elif active_rows:
        matrix = torch.stack(active_rows, dim=0)
        gram = matrix @ matrix.T
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs = matrix @ raw_gradient
        coefficients = torch.linalg.solve(gram + damping * identity, rhs)
        projected = raw_gradient - matrix.T @ coefficients

    raw_norm = torch.linalg.vector_norm(raw_gradient)
    projected_norm = torch.linalg.vector_norm(projected)
    removed_norm = torch.linalg.vector_norm(raw_gradient - projected)
    stats = {
        "constraint_count": float(len(active_rows)),
        "constraint_norm_mean": 0.0,
        "raw_grad_norm": float(raw_norm.detach().cpu()),
        "projected_grad_norm": float(projected_norm.detach().cpu()),
        "projection_removed_fraction": float((removed_norm / raw_norm.clamp_min(1e-12)).detach().cpu()),
        "safe_grad_fraction": float((projected_norm / raw_norm.clamp_min(1e-12)).detach().cpu()),
    }
    if not active_rows:
        stats.update(
            {
                "constraint_effective_rank": 0.0,
                "constraint_numerical_rank": 0.0,
                "constraint_redundancy": 0.0,
                "constraint_condition": 0.0,
                "raw_constraint_cosine_mean": 0.0,
                "raw_constraint_cosine_max": 0.0,
                "projected_constraint_cosine_mean": 0.0,
                "projected_constraint_cosine_max": 0.0,
            }
        )
        return projected, stats

    matrix = torch.stack(active_rows, dim=0)
    row_norms = torch.linalg.vector_norm(matrix, dim=1).clamp_min(1e-12)
    stats["constraint_norm_mean"] = float(row_norms.mean().detach().cpu())
    if plasticity_audit:
        gram = matrix @ matrix.T
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        singular_values = torch.sqrt(eigenvalues)
        positive = singular_values[singular_values > 1e-12]
        if positive.numel() > 0:
            weights = positive / positive.sum().clamp_min(1e-12)
            entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()
            effective_rank_value = torch.exp(entropy)
            max_singular = positive.max()
            numerical_rank = torch.sum(positive > max_singular * rank_tolerance)
            min_kept = positive[positive > max_singular * rank_tolerance].min()
            condition = max_singular / min_kept.clamp_min(1e-12)
        else:
            effective_rank_value = singular_values.new_tensor(0.0)
            numerical_rank = singular_values.new_tensor(0)
            condition = singular_values.new_tensor(0.0)
        raw_cosines = torch.abs((matrix @ raw_gradient) / (row_norms * raw_norm.clamp_min(1e-12)))
        projected_cosines = torch.abs((matrix @ projected) / (row_norms * projected_norm.clamp_min(1e-12)))
        count = float(len(active_rows))
        stats.update(
            {
                "constraint_effective_rank": float(effective_rank_value.detach().cpu()),
                "constraint_numerical_rank": float(numerical_rank.detach().cpu()),
                "constraint_redundancy": float(
                    (1.0 - effective_rank_value / singular_values.new_tensor(count)).detach().cpu()
                ),
                "constraint_condition": float(condition.detach().cpu()),
                "raw_constraint_cosine_mean": float(raw_cosines.mean().detach().cpu()),
                "raw_constraint_cosine_max": float(raw_cosines.max().detach().cpu()),
                "projected_constraint_cosine_mean": float(projected_cosines.mean().detach().cpu()),
                "projected_constraint_cosine_max": float(projected_cosines.max().detach().cpu()),
            }
        )
    else:
        stats.update(
            {
                "constraint_effective_rank": 0.0,
                "constraint_numerical_rank": 0.0,
                "constraint_redundancy": 0.0,
                "constraint_condition": 0.0,
                "raw_constraint_cosine_mean": 0.0,
                "raw_constraint_cosine_max": 0.0,
                "projected_constraint_cosine_mean": 0.0,
                "projected_constraint_cosine_max": 0.0,
            }
        )
    return projected, stats


def distillation_loss_for_batch(
    current_logits: torch.Tensor,
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
        current = current_logits[row_index, :length].unsqueeze(0)
        teacher = teacher_logits[int(global_indices[row_index].item())].to(device).unsqueeze(0)
        losses.append(distillation_kl(current, teacher, temperature=temperature))
    if not losses:
        raise RuntimeError("No distillation losses were built.")
    return torch.stack(losses).mean()


def aggregate_losses_by_category(
    losses: list[tuple[str, torch.Tensor]],
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[torch.Tensor]] = {}
    for category, loss in losses:
        grouped.setdefault(category, []).append(loss)
    if not grouped:
        raise RuntimeError(f"No losses were available for {prefix}.")
    return {
        f"{prefix}:{category}": torch.stack(rows).mean()
        for category, rows in sorted(grouped.items())
    }


def constraint_mode_parts(mode: str) -> tuple[bool, bool, bool]:
    if mode == "scalar":
        return False, False, False
    if mode == "category":
        return True, False, False
    if mode == "category_centroid":
        return True, True, False
    if mode == "category_centroid_separation":
        return True, True, True
    raise ValueError(f"Unknown projected constraint mode {mode!r}.")


def pairwise_squared_distance(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"pairwise_squared_distance expects [n, d], got {values.shape}.")
    diff = values.unsqueeze(1) - values.unsqueeze(0)
    return torch.sum(diff * diff, dim=-1)


def distillation_loss_and_constraint_rows_for_batch(
    current_logits: torch.Tensor,
    selected: list[EncodedExample],
    teacher_logits: list[torch.Tensor],
    global_indices: torch.Tensor,
    *,
    temperature: float,
    device: torch.device,
    constraint_mode: str,
    prefix: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    category_rows, _centroid_rows, _separation_rows = constraint_mode_parts(constraint_mode)
    losses: list[tuple[str, torch.Tensor]] = []
    for row_index, example in enumerate(selected):
        length = len(example.target_ids)
        current = current_logits[row_index, :length].unsqueeze(0)
        teacher = teacher_logits[int(global_indices[row_index].item())].to(device).unsqueeze(0)
        losses.append((example.category, distillation_kl(current, teacher, temperature=temperature)))
    if not losses:
        raise RuntimeError(f"No distillation losses were built for {prefix}.")
    overall = torch.stack([loss for _category, loss in losses]).mean()
    if not category_rows:
        return overall, {f"{prefix}:all": overall}
    return overall, aggregate_losses_by_category(losses, prefix=prefix)


def final_residual(model: GCONativeTransformer, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [batch, seq], got {tokens.shape}.")
    batch, seq_len = tokens.shape
    if seq_len > model.max_seq_len:
        raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={model.max_seq_len}.")
    positions = torch.arange(seq_len, device=tokens.device, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
    h = model.token_embedding(tokens) + model.position_embedding(positions)
    for block in model.blocks:
        h = block(h)
    return model.ln_f(h)


@torch.no_grad()
def collect_final_answer_states(
    model: GCONativeTransformer,
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
        inputs, _targets, masks, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
        states = final_residual(model, inputs).detach().cpu()
        masks_cpu = masks.detach().cpu().to(dtype=torch.bool)
        for row_index, example in enumerate(selected):
            selected_states = states[row_index, : len(example.input_ids)][masks_cpu[row_index, : len(example.input_ids)]]
            if selected_states.numel() <= 0:
                raise RuntimeError(f"No final answer states collected for {example.prompt!r}{example.answer!r}")
            rows.append(selected_states.clone())
    if len(rows) != len(examples):
        raise RuntimeError(f"Collected {len(rows)} final-state anchors for {len(examples)} examples.")
    return rows


def geometry_anchor_loss_for_batch(
    *,
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    selected: list[EncodedExample],
    teacher_states: list[torch.Tensor],
    global_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    states = final_residual(model, inputs)
    losses: list[torch.Tensor] = []
    for row_index, example in enumerate(selected):
        length = len(example.input_ids)
        mask = masks[row_index, :length].to(dtype=torch.bool)
        current = states[row_index, :length][mask]
        teacher = teacher_states[int(global_indices[row_index].item())].to(device)
        if current.shape != teacher.shape:
            raise RuntimeError(
                f"Geometry anchor shape mismatch for {example.prompt!r}: current={current.shape}, teacher={teacher.shape}."
            )
        losses.append(F.mse_loss(current, teacher))
    if not losses:
        raise RuntimeError("No geometry anchor losses were built.")
    return torch.stack(losses).mean()


def geometry_anchor_loss_and_constraint_rows_for_batch(
    *,
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    selected: list[EncodedExample],
    teacher_states: list[torch.Tensor],
    global_indices: torch.Tensor,
    device: torch.device,
    constraint_mode: str,
    prefix: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    category_rows, centroid_rows, separation_rows = constraint_mode_parts(constraint_mode)
    states = final_residual(model, inputs)
    exact_losses: list[tuple[str, torch.Tensor]] = []
    current_by_category: dict[str, list[torch.Tensor]] = {}
    teacher_by_category: dict[str, list[torch.Tensor]] = {}
    for row_index, example in enumerate(selected):
        length = len(example.input_ids)
        mask = masks[row_index, :length].to(dtype=torch.bool)
        current = states[row_index, :length][mask]
        teacher = teacher_states[int(global_indices[row_index].item())].to(device)
        if current.shape != teacher.shape:
            raise RuntimeError(
                f"Geometry anchor shape mismatch for {example.prompt!r}: current={current.shape}, teacher={teacher.shape}."
            )
        exact_losses.append((example.category, F.mse_loss(current, teacher)))
        current_by_category.setdefault(example.category, []).append(current)
        teacher_by_category.setdefault(example.category, []).append(teacher)
    if not exact_losses:
        raise RuntimeError(f"No geometry anchor losses were built for {prefix}.")
    overall = torch.stack([loss for _category, loss in exact_losses]).mean()
    if not category_rows:
        return overall, {f"{prefix}:all": overall}

    rows = aggregate_losses_by_category(exact_losses, prefix=f"{prefix}:exact")
    if centroid_rows:
        current_centroids: dict[str, torch.Tensor] = {}
        teacher_centroids: dict[str, torch.Tensor] = {}
        for category in sorted(current_by_category):
            current_values = torch.cat(current_by_category[category], dim=0)
            teacher_values = torch.cat(teacher_by_category[category], dim=0)
            if current_values.shape != teacher_values.shape:
                raise RuntimeError(
                    f"Centroid geometry shape mismatch at {prefix}/{category}: "
                    f"current={current_values.shape}, teacher={teacher_values.shape}."
                )
            current_centroids[category] = current_values.mean(dim=0)
            teacher_centroids[category] = teacher_values.mean(dim=0)
            rows[f"{prefix}:centroid:{category}"] = F.mse_loss(
                current_centroids[category],
                teacher_centroids[category],
            )
        if separation_rows and len(current_centroids) >= 2:
            categories = sorted(current_centroids)
            current_stack = torch.stack([current_centroids[category] for category in categories], dim=0)
            teacher_stack = torch.stack([teacher_centroids[category] for category in categories], dim=0)
            current_distance = pairwise_squared_distance(current_stack)
            teacher_distance = pairwise_squared_distance(teacher_stack)
            rows[f"{prefix}:separation"] = F.mse_loss(current_distance, teacher_distance)
    if not rows:
        raise RuntimeError(f"No geometry constraint rows were built for {prefix}.")
    return overall, rows


@torch.no_grad()
def behavior_anchor_report(
    *,
    model: GCONativeTransformer,
    examples: list[EncodedExample],
    teacher_logits: list[torch.Tensor],
    pad_id: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    positive_int("batch_size", batch_size)
    model.eval()
    losses: list[float] = []
    for start in range(0, len(examples), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(examples)), dtype=torch.long)
        inputs, _targets, _mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
        logits = model(inputs)
        loss = distillation_loss_for_batch(
            logits,
            selected,
            teacher_logits,
            indices,
            temperature=temperature,
            device=device,
        )
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("Behavior anchor report saw no batches.")
    return {
        "mean_kl": sum(losses) / float(len(losses)),
        "max_batch_kl": max(losses),
        "batch_count": float(len(losses)),
    }


def train_plain(
    *,
    args: argparse.Namespace,
    model: GCONativeTransformer,
    examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
    label: str,
) -> list[dict[str, float]]:
    positive_int("epochs", epochs)
    if not examples:
        raise ValueError(f"{label} received no examples.")
    set_only_native_weights_trainable(model)
    trainable_params = trainable_weight_parameters(model)
    optimizer = make_optimizer(args, trainable_params)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(examples), generator=generator)
        total = 0.0
        grad_total = 0.0
        grad_max = 0.0
        batches = 0
        pbar = tqdm(range(0, len(examples), args.batch_size), desc=f"{label} {epoch}/{epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = masked_ce_loss(logits, targets, mask)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip).detach().cpu())
            optimizer.step()
            value = float(loss.detach().cpu())
            total += value
            grad_total += grad_norm
            grad_max = max(grad_max, grad_norm)
            batches += 1
            pbar.set_postfix({"ce": f"{value:.3g}", "grad": f"{grad_norm:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"{label} epoch {epoch} saw zero batches.")
        row = {
            "epoch": float(epoch),
            "loss": total / float(batches),
            "grad_norm_mean": grad_total / float(batches),
            "grad_norm_max": grad_max,
            "parameter_l2_norm": parameter_l2_norm(trainable_params),
        }
        trace.append(row)
        if epoch == 1 or epoch == epochs or epoch % args.print_every == 0:
            print(
                f"{label} epoch={epoch:4d} loss={row['loss']:.5f} "
                f"grad={row['grad_norm_mean']:.4g}/{row['grad_norm_max']:.4g} "
                f"param={row['parameter_l2_norm']:.4g}"
            )
    return trace


def train_controlled(
    *,
    args: argparse.Namespace,
    model: GCONativeTransformer,
    update_examples: list[EncodedExample],
    preserve_examples: list[EncodedExample],
    guard_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
    guard_logits: list[torch.Tensor],
    preserve_states: list[torch.Tensor],
    guard_states: list[torch.Tensor],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
) -> list[dict[str, float]]:
    positive_int("epochs", epochs)
    if not update_examples:
        raise ValueError("Controlled training received no update examples.")
    if not preserve_examples or not guard_examples:
        raise ValueError("Controlled training requires non-empty preserve and guard examples.")
    set_only_native_weights_trainable(model)
    trainable_params = trainable_weight_parameters(model)
    optimizer = make_optimizer(args, trainable_params)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(update_examples), generator=generator)
        totals = {
            "loss": 0.0,
            "new": 0.0,
            "preserve": 0.0,
            "guard": 0.0,
            "geometry": 0.0,
            "constraint_count": 0.0,
            "constraint_norm_mean": 0.0,
            "constraint_effective_rank": 0.0,
            "constraint_numerical_rank": 0.0,
            "constraint_redundancy": 0.0,
            "constraint_condition": 0.0,
            "raw_grad_norm": 0.0,
            "projected_grad_norm": 0.0,
            "final_update_grad_norm": 0.0,
            "restore_grad_norm": 0.0,
            "projection_removed_fraction": 0.0,
            "safe_grad_fraction": 0.0,
            "final_update_fraction": 0.0,
            "restore_fraction": 0.0,
            "raw_constraint_cosine_mean": 0.0,
            "raw_constraint_cosine_max": 0.0,
            "projected_constraint_cosine_mean": 0.0,
            "projected_constraint_cosine_max": 0.0,
            "final_constraint_cosine_mean": 0.0,
            "final_constraint_cosine_max": 0.0,
        }
        grad_total = 0.0
        grad_max = 0.0
        batches = 0
        pbar = tqdm(range(0, len(update_examples), args.batch_size), desc=f"controlled {epoch}/{epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(update_examples, indices=indices, pad_id=pad_id, device=device)

            preserve_indices = torch.randint(
                low=0,
                high=len(preserve_examples),
                size=(int(indices.numel()),),
                generator=generator,
                device=torch.device("cpu"),
            )
            preserve_inputs, _preserve_targets, preserve_mask, preserve_selected = batch_examples(
                preserve_examples,
                indices=preserve_indices,
                pad_id=pad_id,
                device=device,
            )
            guard_indices = torch.randint(
                low=0,
                high=len(guard_examples),
                size=(int(indices.numel()),),
                generator=generator,
                device=torch.device("cpu"),
            )
            guard_inputs, _guard_targets, guard_mask, guard_selected = batch_examples(
                guard_examples,
                indices=guard_indices,
                pad_id=pad_id,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            update_logits = model(inputs)
            new_loss = masked_ce_loss(update_logits, targets, mask)
            preserve_current = model(preserve_inputs)
            guard_current = model(guard_inputs)
            preserve_loss, preserve_constraint_rows = distillation_loss_and_constraint_rows_for_batch(
                preserve_current,
                preserve_selected,
                preserve_logits,
                preserve_indices,
                temperature=args.distill_temperature,
                device=device,
                constraint_mode=args.projected_constraint_mode,
                prefix="preserve_behavior",
            )
            guard_loss, guard_constraint_rows = distillation_loss_and_constraint_rows_for_batch(
                guard_current,
                guard_selected,
                guard_logits,
                guard_indices,
                temperature=args.distill_temperature,
                device=device,
                constraint_mode=args.projected_constraint_mode,
                prefix="guard_behavior",
            )
            geometry_constraint_rows: dict[str, torch.Tensor] = {}
            if args.lambda_geometry_anchor > 0.0:
                preserve_geometry, preserve_geometry_rows = geometry_anchor_loss_and_constraint_rows_for_batch(
                    model=model,
                    inputs=preserve_inputs,
                    masks=preserve_mask,
                    selected=preserve_selected,
                    teacher_states=preserve_states,
                    global_indices=preserve_indices,
                    device=device,
                    constraint_mode=args.projected_constraint_mode,
                    prefix="preserve_geometry",
                )
                guard_geometry, guard_geometry_rows = geometry_anchor_loss_and_constraint_rows_for_batch(
                    model=model,
                    inputs=guard_inputs,
                    masks=guard_mask,
                    selected=guard_selected,
                    teacher_states=guard_states,
                    global_indices=guard_indices,
                    device=device,
                    constraint_mode=args.projected_constraint_mode,
                    prefix="guard_geometry",
                )
                geometry_loss = 0.5 * (preserve_geometry + guard_geometry)
                geometry_constraint_rows = {
                    **preserve_geometry_rows,
                    **guard_geometry_rows,
                }
            else:
                geometry_loss = new_loss.new_zeros(())
            constraint_row_losses = {
                **preserve_constraint_rows,
                **guard_constraint_rows,
                **geometry_constraint_rows,
            }
            constraint_loss = (
                args.lambda_preserve * preserve_loss
                + args.lambda_guard * guard_loss
                + args.lambda_geometry_anchor * geometry_loss
            )
            loss = new_loss + constraint_loss
            projection_stats = {
                "constraint_count": 0.0,
                "constraint_norm_mean": 0.0,
                "constraint_effective_rank": 0.0,
                "constraint_numerical_rank": 0.0,
                "constraint_redundancy": 0.0,
                "constraint_condition": 0.0,
                "raw_grad_norm": 0.0,
                "projected_grad_norm": 0.0,
                "final_update_grad_norm": 0.0,
                "restore_grad_norm": 0.0,
                "projection_removed_fraction": 0.0,
                "safe_grad_fraction": 1.0,
                "final_update_fraction": 1.0,
                "restore_fraction": 0.0,
                "raw_constraint_cosine_mean": 0.0,
                "raw_constraint_cosine_max": 0.0,
                "projected_constraint_cosine_mean": 0.0,
                "projected_constraint_cosine_max": 0.0,
                "final_constraint_cosine_mean": 0.0,
                "final_constraint_cosine_max": 0.0,
            }
            if args.controlled_update_mode == "loss":
                loss.backward()
            elif args.controlled_update_mode == "projected_invariant_tangent":
                raw_gradient = flat_autograd_gradient(
                    new_loss,
                    trainable_params,
                    retain_graph=True,
                    require_nonzero=True,
                    label="new_loss",
                )
                constraint_gradients: list[torch.Tensor] = []
                for constraint_name, constraint_row_loss in sorted(constraint_row_losses.items()):
                    constraint_gradients.append(
                        flat_autograd_gradient(
                            constraint_row_loss,
                            trainable_params,
                            retain_graph=True,
                            require_nonzero=False,
                            label=constraint_name,
                        )
                    )
                safe_gradient, projection_stats = project_gradient_away_from_constraints(
                    raw_gradient=raw_gradient,
                    constraint_gradients=constraint_gradients,
                    damping=args.projected_update_damping,
                    solver=args.projected_solver,
                    rank_tolerance=args.plasticity_audit_rank_tolerance,
                    plasticity_audit=args.plasticity_audit,
                )
                restore_gradient = torch.zeros_like(safe_gradient)
                if args.projected_restore_strength > 0.0:
                    restore_gradient = flat_autograd_gradient(
                        constraint_loss,
                        trainable_params,
                        retain_graph=True,
                        require_nonzero=False,
                        label="constraint_loss",
                    )
                    safe_gradient = safe_gradient + args.projected_restore_strength * restore_gradient
                raw_norm = torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
                final_norm = torch.linalg.vector_norm(safe_gradient)
                restore_norm = torch.linalg.vector_norm(args.projected_restore_strength * restore_gradient)
                projection_stats["final_update_grad_norm"] = float(final_norm.detach().cpu())
                projection_stats["restore_grad_norm"] = float(restore_norm.detach().cpu())
                projection_stats["final_update_fraction"] = float((final_norm / raw_norm).detach().cpu())
                projection_stats["restore_fraction"] = float((restore_norm / raw_norm).detach().cpu())
                if args.plasticity_audit and constraint_gradients:
                    active_rows = [
                        constraint.to(device=raw_gradient.device, dtype=raw_gradient.dtype)
                        for constraint in constraint_gradients
                        if constraint.shape == raw_gradient.shape
                        and float(torch.linalg.vector_norm(constraint).detach().cpu()) > 1e-12
                    ]
                    if active_rows:
                        matrix = torch.stack(active_rows, dim=0)
                        row_norms = torch.linalg.vector_norm(matrix, dim=1).clamp_min(1e-12)
                        final_cosines = torch.abs((matrix @ safe_gradient) / (row_norms * final_norm.clamp_min(1e-12)))
                        projection_stats["final_constraint_cosine_mean"] = float(final_cosines.mean().detach().cpu())
                        projection_stats["final_constraint_cosine_max"] = float(final_cosines.max().detach().cpu())
                assign_flat_gradient(trainable_params, safe_gradient)
            else:
                raise ValueError(f"Unknown controlled update mode {args.controlled_update_mode!r}.")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip).detach().cpu())
            optimizer.step()
            row = {
                "loss": float(loss.detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "preserve": float(preserve_loss.detach().cpu()),
                "guard": float(guard_loss.detach().cpu()),
                "geometry": float(geometry_loss.detach().cpu()),
                **projection_stats,
            }
            for key, value in row.items():
                totals[key] += value
            grad_total += grad_norm
            grad_max = max(grad_max, grad_norm)
            batches += 1
            postfix = {
                "new": f"{row['new']:.3g}",
                "p": f"{row['preserve']:.3g}",
                "g": f"{row['guard']:.3g}",
                "geom": f"{row['geometry']:.3g}",
                "grad": f"{grad_norm:.3g}",
            }
            if args.controlled_update_mode == "projected_invariant_tangent":
                postfix["safe"] = f"{row['safe_grad_fraction']:.2f}"
                postfix["rem"] = f"{row['projection_removed_fraction']:.2f}"
            pbar.set_postfix(postfix)
        if batches <= 0:
            raise RuntimeError(f"Controlled epoch {epoch} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        epoch_row["grad_norm_mean"] = grad_total / float(batches)
        epoch_row["grad_norm_max"] = grad_max
        epoch_row["parameter_l2_norm"] = parameter_l2_norm(trainable_params)
        trace.append(epoch_row)
        if epoch == 1 or epoch == epochs or epoch % args.print_every == 0:
            message = (
                "controlled epoch={:4d} loss={:.5f} new={:.5f} preserve={:.5f} guard={:.5f} "
                "geometry={:.5f} grad={:.4g}/{:.4g} param={:.4g}"
            ).format(
                epoch,
                epoch_row["loss"],
                epoch_row["new"],
                epoch_row["preserve"],
                epoch_row["guard"],
                epoch_row["geometry"],
                epoch_row["grad_norm_mean"],
                epoch_row["grad_norm_max"],
                epoch_row["parameter_l2_norm"],
            )
            if args.controlled_update_mode == "projected_invariant_tangent":
                message += " safe={:.3f} removed={:.3f} constraints={:.1f}".format(
                    epoch_row["safe_grad_fraction"],
                    epoch_row["projection_removed_fraction"],
                    epoch_row["constraint_count"],
                )
                if args.plasticity_audit:
                    message += " rank={:.2f} red={:.2f} rawCos={:.3f} finalCos={:.3f}".format(
                        epoch_row["constraint_effective_rank"],
                        epoch_row["constraint_redundancy"],
                        epoch_row["raw_constraint_cosine_mean"],
                        epoch_row["final_constraint_cosine_mean"],
                    )
            print(message)
    return trace


def make_optimizer_with_lr(args: argparse.Namespace, params: list[torch.nn.Parameter], *, lr: float) -> torch.optim.Optimizer:
    positive_float("lr", lr)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer {args.optimizer!r}.")


def train_plastic_adapter_stage(
    *,
    args: argparse.Namespace,
    wrapped: AdapterWrappedTransformer,
    adapter: FinalResidualAdapter,
    update_examples: list[EncodedExample],
    preserve_examples: list[EncodedExample],
    guard_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
    guard_logits: list[torch.Tensor],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
    stage_index: int,
) -> list[dict[str, float]]:
    positive_int("epochs", epochs)
    freeze_model(wrapped.base)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    adapter_params = trainable_adapter_parameters(adapter)
    optimizer = make_optimizer_with_lr(args, adapter_params, lr=args.adapter_lr)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        wrapped.train()
        permutation = torch.randperm(len(update_examples), generator=generator)
        totals = {"loss": 0.0, "new": 0.0, "preserve": 0.0, "guard": 0.0, "adapter": 0.0}
        grad_total = 0.0
        grad_max = 0.0
        batches = 0
        pbar = tqdm(
            range(0, len(update_examples), args.batch_size),
            desc=f"adapter_stage_{stage_index} plastic {epoch}/{epochs}",
        )
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(update_examples, indices=indices, pad_id=pad_id, device=device)
            preserve_indices = torch.randint(
                low=0,
                high=len(preserve_examples),
                size=(int(indices.numel()),),
                generator=generator,
                device=torch.device("cpu"),
            )
            preserve_inputs, _preserve_targets, _preserve_mask, preserve_selected = batch_examples(
                preserve_examples,
                indices=preserve_indices,
                pad_id=pad_id,
                device=device,
            )
            guard_indices = torch.randint(
                low=0,
                high=len(guard_examples),
                size=(int(indices.numel()),),
                generator=generator,
                device=torch.device("cpu"),
            )
            guard_inputs, _guard_targets, _guard_mask, guard_selected = batch_examples(
                guard_examples,
                indices=guard_indices,
                pad_id=pad_id,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            update_logits = wrapped(inputs)
            new_loss = masked_ce_loss(update_logits, targets, mask)
            preserve_current = wrapped(preserve_inputs)
            guard_current = wrapped(guard_inputs)
            preserve_loss = distillation_loss_for_batch(
                preserve_current,
                preserve_selected,
                preserve_logits,
                preserve_indices,
                temperature=args.distill_temperature,
                device=device,
            )
            guard_loss = distillation_loss_for_batch(
                guard_current,
                guard_selected,
                guard_logits,
                guard_indices,
                temperature=args.distill_temperature,
                device=device,
            )
            adapter_penalty = adapter.penalty()
            loss = (
                new_loss
                + args.lambda_preserve * preserve_loss
                + args.lambda_guard * guard_loss
                + args.lambda_adapter * adapter_penalty
            )
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter_params, args.grad_clip).detach().cpu())
            optimizer.step()
            row = {
                "loss": float(loss.detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "preserve": float(preserve_loss.detach().cpu()),
                "guard": float(guard_loss.detach().cpu()),
                "adapter": float(adapter_penalty.detach().cpu()),
            }
            for key, value in row.items():
                totals[key] += value
            grad_total += grad_norm
            grad_max = max(grad_max, grad_norm)
            batches += 1
            pbar.set_postfix({"new": f"{row['new']:.3g}", "p": f"{row['preserve']:.3g}", "g": f"{row['guard']:.3g}", "grad": f"{grad_norm:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"Adapter plastic stage {stage_index} epoch {epoch} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        epoch_row["grad_norm_mean"] = grad_total / float(batches)
        epoch_row["grad_norm_max"] = grad_max
        epoch_row["parameter_l2_norm"] = parameter_l2_norm(adapter_params)
        trace.append(epoch_row)
        if epoch == 1 or epoch == epochs or epoch % args.print_every == 0:
            print(
                "adapter_stage={} plastic epoch={:4d} loss={:.5f} new={:.5f} preserve={:.5f} guard={:.5f} adapter={:.5f} grad={:.4g}/{:.4g} param={:.4g}".format(
                    stage_index,
                    epoch,
                    epoch_row["loss"],
                    epoch_row["new"],
                    epoch_row["preserve"],
                    epoch_row["guard"],
                    epoch_row["adapter"],
                    epoch_row["grad_norm_mean"],
                    epoch_row["grad_norm_max"],
                    epoch_row["parameter_l2_norm"],
                )
            )
    return trace


def train_adapter_consolidation_stage(
    *,
    args: argparse.Namespace,
    core: GCONativeTransformer,
    adapter_teacher: AdapterWrappedTransformer,
    update_examples: list[EncodedExample],
    preserve_examples: list[EncodedExample],
    guard_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
    guard_logits: list[torch.Tensor],
    preserve_states: list[torch.Tensor],
    guard_states: list[torch.Tensor],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
    stage_index: int,
) -> list[dict[str, float]]:
    positive_int("epochs", epochs)
    set_only_native_weights_trainable(core)
    freeze_model(adapter_teacher)
    adapter_teacher.eval()
    trainable_params = trainable_weight_parameters(core)
    optimizer = make_optimizer_with_lr(args, trainable_params, lr=args.consolidation_lr)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        core.train()
        permutation = torch.randperm(len(update_examples), generator=generator)
        totals = {
            "loss": 0.0,
            "new": 0.0,
            "preserve": 0.0,
            "guard": 0.0,
            "adapter_distill": 0.0,
            "geometry": 0.0,
        }
        grad_total = 0.0
        grad_max = 0.0
        batches = 0
        pbar = tqdm(
            range(0, len(update_examples), args.batch_size),
            desc=f"adapter_stage_{stage_index} consolidate {epoch}/{epochs}",
        )
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, selected_update = batch_examples(update_examples, indices=indices, pad_id=pad_id, device=device)
            preserve_indices = torch.randint(
                low=0,
                high=len(preserve_examples),
                size=(int(indices.numel()),),
                generator=generator,
                device=torch.device("cpu"),
            )
            preserve_inputs, _preserve_targets, preserve_mask, preserve_selected = batch_examples(
                preserve_examples,
                indices=preserve_indices,
                pad_id=pad_id,
                device=device,
            )
            guard_indices = torch.randint(
                low=0,
                high=len(guard_examples),
                size=(int(indices.numel()),),
                generator=generator,
                device=torch.device("cpu"),
            )
            guard_inputs, _guard_targets, guard_mask, guard_selected = batch_examples(
                guard_examples,
                indices=guard_indices,
                pad_id=pad_id,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            update_logits = core(inputs)
            new_loss = masked_ce_loss(update_logits, targets, mask)
            with torch.no_grad():
                adapter_logits = adapter_teacher(inputs)
            adapter_distill = distillation_kl(update_logits, adapter_logits, temperature=args.distill_temperature)

            preserve_current = core(preserve_inputs)
            guard_current = core(guard_inputs)
            preserve_loss = distillation_loss_for_batch(
                preserve_current,
                preserve_selected,
                preserve_logits,
                preserve_indices,
                temperature=args.distill_temperature,
                device=device,
            )
            guard_loss = distillation_loss_for_batch(
                guard_current,
                guard_selected,
                guard_logits,
                guard_indices,
                temperature=args.distill_temperature,
                device=device,
            )
            if args.lambda_geometry_anchor > 0.0:
                preserve_geometry = geometry_anchor_loss_for_batch(
                    model=core,
                    inputs=preserve_inputs,
                    masks=preserve_mask,
                    selected=preserve_selected,
                    teacher_states=preserve_states,
                    global_indices=preserve_indices,
                    device=device,
                )
                guard_geometry = geometry_anchor_loss_for_batch(
                    model=core,
                    inputs=guard_inputs,
                    masks=guard_mask,
                    selected=guard_selected,
                    teacher_states=guard_states,
                    global_indices=guard_indices,
                    device=device,
                )
                geometry_loss = 0.5 * (preserve_geometry + guard_geometry)
            else:
                geometry_loss = new_loss.new_zeros(())
            loss = (
                new_loss
                + args.lambda_preserve * preserve_loss
                + args.lambda_guard * guard_loss
                + args.lambda_adapter_distill * adapter_distill
                + args.lambda_geometry_anchor * geometry_loss
            )
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip).detach().cpu())
            optimizer.step()
            row = {
                "loss": float(loss.detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "preserve": float(preserve_loss.detach().cpu()),
                "guard": float(guard_loss.detach().cpu()),
                "adapter_distill": float(adapter_distill.detach().cpu()),
                "geometry": float(geometry_loss.detach().cpu()),
            }
            for key, value in row.items():
                totals[key] += value
            grad_total += grad_norm
            grad_max = max(grad_max, grad_norm)
            batches += 1
            pbar.set_postfix(
                {
                    "new": f"{row['new']:.3g}",
                    "ad": f"{row['adapter_distill']:.3g}",
                    "p": f"{row['preserve']:.3g}",
                    "g": f"{row['guard']:.3g}",
                    "grad": f"{grad_norm:.3g}",
                }
            )
        if batches <= 0:
            raise RuntimeError(f"Adapter consolidation stage {stage_index} epoch {epoch} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        epoch_row["grad_norm_mean"] = grad_total / float(batches)
        epoch_row["grad_norm_max"] = grad_max
        epoch_row["parameter_l2_norm"] = parameter_l2_norm(trainable_params)
        trace.append(epoch_row)
        if epoch == 1 or epoch == epochs or epoch % args.print_every == 0:
            print(
                "adapter_stage={} consolidate epoch={:4d} loss={:.5f} new={:.5f} adapter_distill={:.5f} preserve={:.5f} guard={:.5f} geometry={:.5f} grad={:.4g}/{:.4g} param={:.4g}".format(
                    stage_index,
                    epoch,
                    epoch_row["loss"],
                    epoch_row["new"],
                    epoch_row["adapter_distill"],
                    epoch_row["preserve"],
                    epoch_row["guard"],
                    epoch_row["geometry"],
                    epoch_row["grad_norm_mean"],
                    epoch_row["grad_norm_max"],
                    epoch_row["parameter_l2_norm"],
                )
            )
    return trace


def collect_geometry_examples(encoded: dict[str, list[EncodedExample]]) -> list[EncodedExample]:
    examples = (
        encoded["preserve_eval"]
        + encoded["guard_eval"]
        + encoded["changed_eval"]
        + encoded["new_eval"]
        + encoded["composition_eval"]
    )
    if not examples:
        raise RuntimeError("Geometry example set is empty.")
    return examples


@torch.no_grad()
def collect_flat_residual_states(
    model: GCONativeTransformer,
    examples: list[EncodedExample],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    rows: dict[str, list[torch.Tensor]] = {}
    for example in examples:
        tokens = torch.tensor([example.input_ids], dtype=torch.long)
        states = collect_states(model, tokens, device)
        for layer, value in states.items():
            length = len(example.input_ids)
            rows.setdefault(layer, []).append(value[:, :length].reshape(-1, value.shape[-1]).to(dtype=torch.float32))
    if not rows:
        raise RuntimeError("No residual states collected.")
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in rows.items()}


@torch.no_grad()
def collect_answer_residual_states(
    model: GCONativeTransformer,
    examples: list[EncodedExample],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    rows: dict[str, list[torch.Tensor]] = {}
    for example in examples:
        tokens = torch.tensor([example.input_ids], dtype=torch.long)
        mask = torch.tensor(example.loss_mask, dtype=torch.bool)
        if mask.numel() != len(example.input_ids):
            raise RuntimeError(
                f"Loss mask/input length mismatch for {example.prompt!r}: "
                f"mask={mask.numel()} inputs={len(example.input_ids)}."
            )
        if not bool(mask.any().item()):
            raise RuntimeError(f"Example has no answer-state positions: {example.prompt!r}{example.answer!r}")
        states = collect_states(model, tokens, device)
        for layer, value in states.items():
            selected = value[0, mask.to(value.device)].detach().cpu().to(dtype=torch.float32)
            rows.setdefault(layer, []).append(selected)
    if not rows:
        raise RuntimeError("No answer residual states collected.")
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in rows.items()}


@torch.no_grad()
def geometry_report(
    *,
    reference: GCONativeTransformer,
    candidate: GCONativeTransformer,
    examples: list[EncodedExample],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    reference_states = collect_flat_residual_states(reference, examples, device=device)
    candidate_states = collect_flat_residual_states(candidate, examples, device=device)
    report: dict[str, dict[str, float]] = {}
    for layer in sorted(reference_states):
        ref = reference_states[layer]
        cur = candidate_states[layer]
        if ref.shape != cur.shape:
            raise RuntimeError(f"Geometry state shape mismatch at {layer}: ref={ref.shape}, cur={cur.shape}.")
        _aligned, drift = procrustes_align(cur, ref)
        _u_ref, s_ref, _vh_ref = torch.linalg.svd(ref - ref.mean(dim=0, keepdim=True), full_matrices=False)
        _u_cur, s_cur, _vh_cur = torch.linalg.svd(cur - cur.mean(dim=0, keepdim=True), full_matrices=False)
        report[layer] = {
            "reference_rank": effective_rank(s_ref),
            "candidate_rank": effective_rank(s_cur),
            "rank_delta": effective_rank(s_cur) - effective_rank(s_ref),
            "drift_relative": drift["aligned_drift_relative"],
            "drift_mean": drift["aligned_drift_mean"],
            "drift_max": drift["aligned_drift_max"],
            "cka": linear_cka(ref, cur),
        }
    return report


def role_groups(encoded: dict[str, list[EncodedExample]]) -> dict[str, list[EncodedExample]]:
    return {
        "preserve": encoded["preserve_eval"],
        "guard": encoded["guard_eval"],
        "changed": encoded["changed_eval"],
        "new": encoded["new_eval"],
        "composition": encoded["composition_eval"],
        "obsolete_old_answer": encoded["obsolete_eval"],
    }


def feature_groups(encoded: dict[str, list[EncodedExample]]) -> dict[str, list[EncodedExample]]:
    all_examples = (
        encoded["preserve_eval"]
        + encoded["guard_eval"]
        + encoded["changed_eval"]
        + encoded["new_eval"]
        + encoded["composition_eval"]
        + encoded["obsolete_eval"]
    )
    carry = [example for example in all_examples if example.prompt.startswith("Question: What does ")]
    place = [example for example in all_examples if " open? Answer:" in example.prompt and not example.prompt.startswith("Question: What can ")]
    composition = [example for example in all_examples if example.prompt.startswith("Question: What can ")]
    obsolete = list(encoded["obsolete_eval"])
    groups = {
        "carry_feature": carry,
        "place_feature": place,
        "composition_feature": composition,
        "obsolete_feature": obsolete,
    }
    empty = [name for name, examples in groups.items() if not examples]
    if empty:
        raise RuntimeError(f"Feature groups unexpectedly empty: {empty}.")
    return groups


def centroid_distance_matrix(centroids: dict[str, torch.Tensor]) -> torch.Tensor:
    names = sorted(centroids)
    if not names:
        raise RuntimeError("Cannot build centroid matrix from empty centroids.")
    rows = torch.stack([centroids[name] for name in names], dim=0)
    return torch.cdist(rows, rows)


def vector_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Cosine vectors must have the same shape: {a.shape} vs {b.shape}.")
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 1e-12:
        return 1.0 if float(torch.linalg.vector_norm(a - b).item()) <= 1e-12 else 0.0
    return float((torch.dot(a.reshape(-1), b.reshape(-1)) / denom.clamp_min(1e-12)).item())


@torch.no_grad()
def grouped_geometry_report(
    *,
    reference: GCONativeTransformer,
    candidate: GCONativeTransformer,
    groups: dict[str, list[EncodedExample]],
    device: torch.device,
) -> dict[str, Any]:
    group_metrics: dict[str, dict[str, dict[str, float]]] = {}
    ref_centroids_by_layer: dict[str, dict[str, torch.Tensor]] = {}
    cur_centroids_by_layer: dict[str, dict[str, torch.Tensor]] = {}
    for group_name, examples in sorted(groups.items()):
        if not examples:
            raise ValueError(f"Group {group_name!r} is empty.")
        ref_states = collect_answer_residual_states(reference, examples, device=device)
        cur_states = collect_answer_residual_states(candidate, examples, device=device)
        group_metrics[group_name] = {}
        for layer in sorted(ref_states):
            ref = ref_states[layer]
            cur = cur_states[layer]
            if ref.shape != cur.shape:
                raise RuntimeError(
                    f"Grouped geometry shape mismatch at {group_name}/{layer}: ref={ref.shape}, cur={cur.shape}."
                )
            _aligned, drift = procrustes_align(cur, ref)
            _u_ref, s_ref, _vh_ref = torch.linalg.svd(ref - ref.mean(dim=0, keepdim=True), full_matrices=False)
            _u_cur, s_cur, _vh_cur = torch.linalg.svd(cur - cur.mean(dim=0, keepdim=True), full_matrices=False)
            ref_centroid = ref.mean(dim=0)
            cur_centroid = cur.mean(dim=0)
            ref_centroids_by_layer.setdefault(layer, {})[group_name] = ref_centroid
            cur_centroids_by_layer.setdefault(layer, {})[group_name] = cur_centroid
            group_metrics[group_name][layer] = {
                "sample_count": float(ref.shape[0]),
                "centroid_drift": float(torch.linalg.vector_norm(cur_centroid - ref_centroid).item()),
                "centroid_cosine": vector_cosine(cur_centroid, ref_centroid),
                "drift_relative": drift["aligned_drift_relative"],
                "drift_mean": drift["aligned_drift_mean"],
                "cka": linear_cka(ref, cur),
                "reference_rank": effective_rank(s_ref),
                "candidate_rank": effective_rank(s_cur),
                "rank_delta": effective_rank(s_cur) - effective_rank(s_ref),
            }

    separation: dict[str, dict[str, float]] = {}
    for layer in sorted(ref_centroids_by_layer):
        ref_matrix = centroid_distance_matrix(ref_centroids_by_layer[layer])
        cur_matrix = centroid_distance_matrix(cur_centroids_by_layer[layer])
        if ref_matrix.shape != cur_matrix.shape:
            raise RuntimeError(f"Centroid distance matrix mismatch at {layer}: {ref_matrix.shape} vs {cur_matrix.shape}.")
        delta = cur_matrix - ref_matrix
        separation[layer] = {
            "separation_drift": float(torch.linalg.vector_norm(delta).item()),
            "separation_drift_relative": float(
                (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref_matrix).clamp_min(1e-12)).item()
            ),
            "separation_cosine": vector_cosine(cur_matrix.reshape(-1), ref_matrix.reshape(-1)),
            "reference_mean_distance": float(ref_matrix.mean().item()),
            "candidate_mean_distance": float(cur_matrix.mean().item()),
        }
    return {"groups": group_metrics, "separation": separation}


def evaluate_named_groups(
    *,
    model: GCONativeTransformer,
    encoded: dict[str, list[EncodedExample]],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    selected = {
        "preserve": encoded["preserve_eval"],
        "guard": encoded["guard_eval"],
        "changed": encoded["changed_eval"],
        "new": encoded["new_eval"],
        "composition": encoded["composition_eval"],
        "obsolete_old_answer": encoded["obsolete_eval"],
    }
    return {
        name: evaluate_examples(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)["overall"]
        for name, examples in selected.items()
    }


@torch.no_grad()
def evaluate_individual_examples(
    *,
    model: torch.nn.Module,
    examples: list[EncodedExample],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    positive_int("batch_size", batch_size)
    if not examples:
        raise ValueError("evaluate_individual_examples received no examples.")
    model.eval()
    rows: list[dict[str, float | int | str]] = []
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
            answer_count = float(mask_cpu[row_index].sum().item())
            if answer_count <= 0.0:
                raise RuntimeError(f"Individual evaluation saw zero answer tokens for {example.prompt!r}{example.answer!r}")
            correct_count = float(token_correct[row_index].sum().item())
            rows.append(
                {
                    "index": int(start + row_index),
                    "category": example.category,
                    "prompt": example.prompt,
                    "answer": example.answer,
                    "loss": float(losses_cpu[row_index].sum().item() / answer_count),
                    "token_accuracy": correct_count / answer_count,
                    "exact_match": 1.0 if correct_count == answer_count else 0.0,
                    "answer_token_count": answer_count,
                }
            )
    if len(rows) != len(examples):
        raise RuntimeError(f"Individual evaluation returned {len(rows)} rows for {len(examples)} examples.")
    return rows


def encoded_example_key(example: EncodedExample) -> str:
    return "\n".join([example.category, example.prompt, example.answer])


def select_commit_examples(
    *,
    model: torch.nn.Module,
    examples: list[EncodedExample],
    existing_keys: set[str],
    pad_id: int,
    batch_size: int,
    device: torch.device,
    min_exact: float,
    min_token_accuracy: float,
    max_loss: float,
) -> tuple[list[EncodedExample], dict[str, Any]]:
    rows = evaluate_individual_examples(
        model=model,
        examples=examples,
        pad_id=pad_id,
        batch_size=batch_size,
        device=device,
    )
    selected: list[EncodedExample] = []
    selected_rows: list[dict[str, float | int | str]] = []
    rejected_rows: list[dict[str, float | int | str]] = []
    duplicate_count = 0
    for row, example in zip(rows, examples, strict=True):
        key = encoded_example_key(example)
        if key in existing_keys:
            duplicate_count += 1
            rejected_rows.append({**row, "reject_reason": "duplicate"})
            continue
        passes = (
            float(row["exact_match"]) >= min_exact
            and float(row["token_accuracy"]) >= min_token_accuracy
            and float(row["loss"]) <= max_loss
        )
        if passes:
            selected.append(example)
            selected_rows.append({**row, "reject_reason": ""})
        else:
            rejected_rows.append({**row, "reject_reason": "threshold"})
    category_counts: dict[str, int] = {}
    for example in selected:
        category_counts[example.category] = category_counts.get(example.category, 0) + 1
    report = {
        "candidate_count": len(examples),
        "selected_count": len(selected),
        "rejected_count": len(rejected_rows),
        "duplicate_count": duplicate_count,
        "selected_category_counts": category_counts,
        "criteria": {
            "min_exact": min_exact,
            "min_token_accuracy": min_token_accuracy,
            "max_loss": max_loss,
        },
        "selected": selected_rows,
        "rejected": rejected_rows,
    }
    return selected, report


def build_committed_anchor_bank(
    *,
    model: GCONativeTransformer,
    selected: list[EncodedExample],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if not selected:
        raise ValueError("Cannot build committed anchors from an empty selection.")
    return {
        "examples": selected,
        "logits": collect_example_logits(
            model,
            selected,
            pad_id=pad_id,
            batch_size=batch_size,
            device=device,
        ),
        "states": collect_final_answer_states(
            model,
            selected,
            pad_id=pad_id,
            batch_size=batch_size,
            device=device,
        ),
    }


def committed_bank_examples(committed_banks: list[dict[str, Any]]) -> list[EncodedExample]:
    examples: list[EncodedExample] = []
    for bank in committed_banks:
        bank_examples = bank["examples"]
        bank_logits = bank["logits"]
        bank_states = bank["states"]
        if not (len(bank_examples) == len(bank_logits) == len(bank_states)):
            raise RuntimeError(
                "Committed bank length mismatch before budget selection: "
                f"examples={len(bank_examples)} logits={len(bank_logits)} states={len(bank_states)}."
            )
        examples.extend(bank_examples)
    return examples


def deduplicate_examples(examples: list[EncodedExample]) -> list[EncodedExample]:
    seen: set[str] = set()
    rows: list[EncodedExample] = []
    for example in examples:
        key = encoded_example_key(example)
        if key in seen:
            continue
        seen.add(key)
        rows.append(example)
    return rows


def category_counts(examples: list[EncodedExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.category] = counts.get(example.category, 0) + 1
    return counts


def select_budgeted_commit_examples(
    *,
    model: torch.nn.Module,
    examples: list[EncodedExample],
    budget: int,
    selection: str,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[EncodedExample], dict[str, Any]]:
    nonnegative_int("budget", budget)
    if selection != "diverse_by_category_then_loss":
        raise ValueError(f"Unknown commit selection mode {selection!r}.")
    unique_examples = deduplicate_examples(examples)
    if budget == 0 or len(unique_examples) <= budget:
        return unique_examples, {
            "selection": selection,
            "budget": budget,
            "before_count": len(unique_examples),
            "after_count": len(unique_examples),
            "dropped_count": 0,
            "selected_category_counts": category_counts(unique_examples),
            "dropped_category_counts": {},
        }

    rows = evaluate_individual_examples(
        model=model,
        examples=unique_examples,
        pad_id=pad_id,
        batch_size=batch_size,
        device=device,
    )
    by_category: dict[str, list[tuple[EncodedExample, dict[str, float | int | str]]]] = {}
    for example, row in zip(unique_examples, rows, strict=True):
        by_category.setdefault(example.category, []).append((example, row))
    for category, items in by_category.items():
        if not items:
            raise RuntimeError(f"Budget category {category!r} unexpectedly empty.")
        items.sort(
            key=lambda item: (
                -float(item[1]["exact_match"]),
                -float(item[1]["token_accuracy"]),
                float(item[1]["loss"]),
                item[0].prompt,
                item[0].answer,
            )
        )

    selected: list[EncodedExample] = []
    selected_keys: set[str] = set()
    categories = sorted(by_category)
    while len(selected) < budget:
        added_this_round = False
        for category in categories:
            if len(selected) >= budget:
                break
            bucket = by_category[category]
            while bucket:
                example, _row = bucket.pop(0)
                key = encoded_example_key(example)
                if key in selected_keys:
                    continue
                selected.append(example)
                selected_keys.add(key)
                added_this_round = True
                break
        if not added_this_round:
            break
    if len(selected) != budget:
        raise RuntimeError(f"Budget selection chose {len(selected)} examples for budget={budget}.")

    selected_key_set = {encoded_example_key(example) for example in selected}
    dropped = [example for example in unique_examples if encoded_example_key(example) not in selected_key_set]
    return selected, {
        "selection": selection,
        "budget": budget,
        "before_count": len(unique_examples),
        "after_count": len(selected),
        "dropped_count": len(dropped),
        "selected_category_counts": category_counts(selected),
        "dropped_category_counts": category_counts(dropped),
    }


def apply_committed_memory_budget(
    *,
    model: GCONativeTransformer,
    committed_banks: list[dict[str, Any]],
    budget: int,
    selection: str,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    all_examples = committed_bank_examples(committed_banks)
    selected, report = select_budgeted_commit_examples(
        model=model,
        examples=all_examples,
        budget=budget,
        selection=selection,
        pad_id=pad_id,
        batch_size=batch_size,
        device=device,
    )
    if selected:
        banks = [
            build_committed_anchor_bank(
                model=model,
                selected=selected,
                pad_id=pad_id,
                batch_size=batch_size,
                device=device,
            )
        ]
    else:
        banks = []
    keys = {encoded_example_key(example) for example in selected}
    if len(keys) != len(selected):
        raise RuntimeError("Budgeted committed memory contains duplicate keys after selection.")
    return banks, keys, report


def combine_anchor_banks(
    *,
    base_examples: list[EncodedExample],
    base_logits: list[torch.Tensor],
    base_states: list[torch.Tensor],
    committed_banks: list[dict[str, Any]],
) -> tuple[list[EncodedExample], list[torch.Tensor], list[torch.Tensor]]:
    examples = list(base_examples)
    logits = list(base_logits)
    states = list(base_states)
    for bank in committed_banks:
        bank_examples = bank["examples"]
        bank_logits = bank["logits"]
        bank_states = bank["states"]
        if not (len(bank_examples) == len(bank_logits) == len(bank_states)):
            raise RuntimeError(
                "Committed anchor bank length mismatch: "
                f"examples={len(bank_examples)} logits={len(bank_logits)} states={len(bank_states)}."
            )
        examples.extend(bank_examples)
        logits.extend(bank_logits)
        states.extend(bank_states)
    if not (len(examples) == len(logits) == len(states)):
        raise RuntimeError(
            f"Combined anchor length mismatch: examples={len(examples)} logits={len(logits)} states={len(states)}."
        )
    return examples, logits, states


def metric_pair(row: dict[str, float]) -> str:
    return "{:.4g}/{:.3f}".format(row["loss"], row["exact_match"])


def print_metric_delta(
    *,
    label: str,
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> None:
    groups = ["preserve", "guard", "changed", "new", "composition", "obsolete_old_answer"]
    print(f"\n{label} diagnostics")
    print("-" * 112)
    print("{:>20} {:>18} {:>18} {:>14} {:>14}".format("group", "before", "after", "loss_delta", "exact_delta"))
    for group in groups:
        if group not in before or group not in after:
            raise RuntimeError(f"Missing diagnostic group {group!r} for {label}.")
        loss_delta = after[group]["loss"] - before[group]["loss"]
        exact_delta = after[group]["exact_match"] - before[group]["exact_match"]
        print(
            "{:>20} {:>18} {:>18} {:14.5g} {:14.5g}".format(
                group,
                metric_pair(before[group]),
                metric_pair(after[group]),
                loss_delta,
                exact_delta,
            )
        )


def summarize_eval_delta(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for group, before_row in before.items():
        if group not in after:
            raise RuntimeError(f"After-metrics missing group {group!r}.")
        after_row = after[group]
        summary[group] = {
            "before_loss": float(before_row["loss"]),
            "after_loss": float(after_row["loss"]),
            "loss_delta": float(after_row["loss"] - before_row["loss"]),
            "before_exact": float(before_row["exact_match"]),
            "after_exact": float(after_row["exact_match"]),
            "exact_delta": float(after_row["exact_match"] - before_row["exact_match"]),
            "before_token_accuracy": float(before_row["token_accuracy"]),
            "after_token_accuracy": float(after_row["token_accuracy"]),
            "token_accuracy_delta": float(after_row["token_accuracy"] - before_row["token_accuracy"]),
        }
    return summary


def summarize_geometry_means(report: dict[str, dict[str, float]]) -> dict[str, float]:
    if not report:
        raise ValueError("summarize_geometry_means received an empty report.")
    keys = ["drift_relative", "cka", "rank_delta"]
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in report.values()]
        if not values:
            raise RuntimeError(f"No geometry values for {key}.")
        summary[key] = sum(values) / float(len(values))
    return summary


def print_geometry_delta(*, label: str, report: dict[str, dict[str, float]]) -> None:
    summary = summarize_geometry_means(report)
    print(
        "{} geometry mean drift_rel={:.4f} cka={:.4f} rank_delta={:.4f}".format(
            label,
            summary["drift_relative"],
            summary["cka"],
            summary["rank_delta"],
        )
    )


def mean_role_geometry_metric(
    report: dict[str, Any],
    *,
    role: str,
    metric: str,
) -> float:
    groups = report["groups"]
    if role not in groups:
        raise RuntimeError(f"Role geometry report missing role {role!r}.")
    rows = [float(row[metric]) for row in groups[role].values()]
    if not rows:
        raise RuntimeError(f"No role geometry values for role={role} metric={metric}.")
    return sum(rows) / float(len(rows))


def plot_behavior_outcome(
    *,
    metrics: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    methods = ["naive", "controlled", "joint"]
    groups = ["preserve", "guard", "changed", "new", "composition", "obsolete_old_answer"]
    labels = ["preserve", "guard", "changed", "new", "compose", "old-wrong"]
    width = 0.24
    x_values = list(range(len(groups)))
    fig, ax = plt.subplots(figsize=(11, 5))
    for method_index, method in enumerate(methods):
        offsets = [x + (method_index - 1) * width for x in x_values]
        values = [float(metrics[method][group]["exact_match"]) for group in groups]
        ax.bar(offsets, values, width=width, label=method)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("exact match")
    ax.set_title("Behavior after continual learning")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stage_trajectory(
    *,
    controlled_trace: list[dict[str, Any]],
    output_path: Path,
    memory_budget: int,
) -> None:
    import matplotlib.pyplot as plt

    if not controlled_trace:
        raise ValueError("Cannot plot stage trajectory without controlled trace.")
    stages = [int(row["stage"]) for row in controlled_trace]
    groups = ["preserve", "guard", "changed", "new", "composition", "obsolete_old_answer"]
    labels = {
        "preserve": "preserve",
        "guard": "guard",
        "changed": "changed",
        "new": "new",
        "composition": "compose",
        "obsolete_old_answer": "old-wrong",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5))
    for group in groups:
        values: list[float] = []
        for row in controlled_trace:
            diagnostics = row["diagnostics"]
            if "after_metrics" not in diagnostics:
                raise RuntimeError("--diagnostics is required for stage trajectory plotting.")
            values.append(float(diagnostics["after_metrics"][group]["exact_match"]))
        ax.plot(stages, values, marker="o", linewidth=2, label=labels[group])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("CL stage")
    ax.set_ylabel("exact match")
    ax.set_title("Controlled CL stage trajectory")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    memory_sizes = [int(row["committed_anchor_count_after_stage"]) for row in controlled_trace]
    ax2.plot(stages, memory_sizes, color="black", marker="s", linestyle="--", linewidth=2, label="K size")
    if memory_budget > 0:
        ax2.axhline(memory_budget, color="black", linestyle=":", linewidth=1.5, label="K budget")
    ax2.set_ylabel("committed memory size")
    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, line_labels + line_labels2, ncol=4, fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_protected_geometry_cka(
    *,
    role_geometry: dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    comparisons = ["naive_vs_base", "controlled_vs_base", "joint_vs_base"]
    roles = ["preserve", "guard", "changed", "new"]
    width = 0.24
    x_values = list(range(len(roles)))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for comparison_index, comparison in enumerate(comparisons):
        values = [
            mean_role_geometry_metric(role_geometry[comparison], role=role, metric="cka")
            for role in roles
        ]
        offsets = [x + (comparison_index - 1) * width for x in x_values]
        ax.bar(offsets, values, width=width, label=comparison.replace("_vs_base", ""))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("CKA to base geometry")
    ax.set_title("Protected role geometry health")
    ax.set_xticks(x_values)
    ax.set_xticklabels(roles)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_role_geometry_drift_heatmap(
    *,
    role_geometry: dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    roles = ["preserve", "guard", "changed", "new", "composition", "obsolete_old_answer"]
    comparisons = ["naive_vs_base", "controlled_vs_base", "joint_vs_base"]
    matrix = [
        [
            mean_role_geometry_metric(role_geometry[comparison], role=role, metric="centroid_drift")
            for comparison in comparisons
        ]
        for role in roles
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("Role centroid drift")
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels([value.replace("_vs_base", "") for value in comparisons], rotation=20, ha="right")
    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(["old-wrong" if role == "obsolete_old_answer" else role for role in roles])
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="centroid drift")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def plot_final_residual_pca_roles(
    *,
    models: dict[str, GCONativeTransformer],
    groups: dict[str, list[EncodedExample]],
    device: torch.device,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    panel_methods = ["naive", "controlled"]
    roles = ["preserve", "guard", "changed", "new", "obsolete_old_answer"]
    colors = {
        "preserve": "#2ca02c",
        "guard": "#1f77b4",
        "changed": "#ff7f0e",
        "new": "#9467bd",
        "obsolete_old_answer": "#d62728",
    }
    raw: dict[str, dict[str, torch.Tensor]] = {}
    all_states: list[torch.Tensor] = []
    for method in panel_methods:
        raw[method] = {}
        for role in roles:
            states = collect_answer_residual_states(models[method], groups[role], device=device)
            if "final" not in states:
                raise RuntimeError("Final residual states are required for PCA plotting.")
            final_states = states["final"].to(dtype=torch.float32).cpu()
            raw[method][role] = final_states
            all_states.append(final_states)
    matrix = torch.cat(all_states, dim=0)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < 2:
        raise RuntimeError(f"Need at least two PCA components, got {vh.shape}.")
    basis = vh[:2].T
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for axis, method in zip(axes, panel_methods, strict=True):
        for role in roles:
            projected = (raw[method][role] - matrix.mean(dim=0, keepdim=True)) @ basis
            label = "old-wrong" if role == "obsolete_old_answer" else role
            axis.scatter(
                projected[:, 0].numpy(),
                projected[:, 1].numpy(),
                s=18,
                alpha=0.75,
                label=label,
                color=colors[role],
            )
        axis.set_title(method)
        axis.set_xlabel("PC1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("PC2")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Final residual role geometry after CL")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_committed_memory_budget(
    *,
    controlled_trace: list[dict[str, Any]],
    output_path: Path,
    memory_budget: int,
) -> None:
    import matplotlib.pyplot as plt

    if not controlled_trace:
        raise ValueError("Cannot plot committed memory without controlled trace.")
    stages = [int(row["stage"]) for row in controlled_trace]
    reports = []
    for row in controlled_trace:
        report = row.get("committed_anchor_report")
        if report is None:
            raise RuntimeError("Committed memory plot requires --dynamic-committed-anchors.")
        reports.append(report["memory_budget"])
    categories = sorted(
        {
            category
            for report in reports
            for category in report["selected_category_counts"]
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = [0 for _ in stages]
    for category in categories:
        values = [int(report["selected_category_counts"].get(category, 0)) for report in reports]
        ax.bar(stages, values, bottom=bottoms, label=category)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    if memory_budget > 0:
        ax.axhline(memory_budget, color="black", linestyle="--", linewidth=1.5, label="budget")
    ax.set_xlabel("CL stage")
    ax.set_ylabel("committed anchors in K")
    ax.set_title("Bounded dynamic committed memory")
    ax.set_xticks(stages)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_plasticity_audit(
    *,
    controlled_trace: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not controlled_trace:
        raise ValueError("Cannot plot plasticity audit without controlled trace.")

    def last_metric(stage_row: dict[str, Any], metric: str) -> float:
        if not stage_row["trace"]:
            raise RuntimeError(f"Stage {stage_row['stage']} has empty trace.")
        last = stage_row["trace"][-1]
        if metric not in last:
            raise RuntimeError(f"Plasticity audit metric {metric!r} missing from stage {stage_row['stage']}.")
        return float(last[metric])

    stages = [int(row["stage"]) for row in controlled_trace]
    safe = [last_metric(row, "safe_grad_fraction") for row in controlled_trace]
    final = [last_metric(row, "final_update_fraction") for row in controlled_trace]
    removed = [last_metric(row, "projection_removed_fraction") for row in controlled_trace]
    count = [last_metric(row, "constraint_count") for row in controlled_trace]
    rank = [last_metric(row, "constraint_effective_rank") for row in controlled_trace]
    redundancy = [last_metric(row, "constraint_redundancy") for row in controlled_trace]
    raw_cos = [last_metric(row, "raw_constraint_cosine_mean") for row in controlled_trace]
    final_cos = [last_metric(row, "final_constraint_cosine_mean") for row in controlled_trace]
    new_loss = [last_metric(row, "new") for row in controlled_trace]
    geometry_loss = [last_metric(row, "geometry") for row in controlled_trace]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(stages, safe, marker="o", linewidth=2, label="projected/raw")
    ax.plot(stages, final, marker="o", linewidth=2, label="final/raw")
    ax.plot(stages, removed, marker="o", linewidth=2, label="removed/raw")
    ax.set_title("Plasticity retained by tangent update")
    ax.set_xlabel("CL stage")
    ax.set_ylabel("gradient fraction")
    ax.set_ylim(0.0, max(1.05, max(final + safe + removed) * 1.1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(stages, count, marker="o", linewidth=2, label="constraint rows")
    ax.plot(stages, rank, marker="o", linewidth=2, label="effective rank")
    ax2 = ax.twinx()
    ax2.plot(stages, redundancy, color="tab:red", marker="s", linestyle="--", linewidth=2, label="redundancy")
    ax.set_title("Constraint basis size")
    ax.set_xlabel("CL stage")
    ax.set_ylabel("rows / effective rank")
    ax2.set_ylabel("redundancy")
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = axes[1, 0]
    ax.plot(stages, raw_cos, marker="o", linewidth=2, label="raw gradient")
    ax.plot(stages, final_cos, marker="o", linewidth=2, label="final update")
    ax.set_title("Alignment with protected constraint normals")
    ax.set_xlabel("CL stage")
    ax.set_ylabel("mean absolute cosine")
    ax.set_ylim(0.0, max(0.05, max(raw_cos + final_cos) * 1.15))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(stages, new_loss, marker="o", linewidth=2, label="new loss")
    ax.plot(stages, geometry_loss, marker="o", linewidth=2, label="geometry loss")
    ax.set_title("Learning pressure vs geometry pressure")
    ax.set_xlabel("CL stage")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Plasticity audit across the continual-learning loop")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_plots(
    *,
    plot_dir: Path,
    metrics: dict[str, dict[str, dict[str, float]]],
    controlled_trace: list[dict[str, Any]],
    role_feature_geometry: dict[str, Any],
    models: dict[str, GCONativeTransformer],
    encoded: dict[str, list[EncodedExample]],
    device: torch.device,
    memory_budget: int,
    plasticity_audit: bool,
) -> dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "behavior_outcome": plot_dir / "01_behavior_outcome.png",
        "stage_trajectory": plot_dir / "02_stage_trajectory.png",
        "protected_geometry_cka": plot_dir / "03_protected_geometry_cka.png",
        "role_geometry_drift": plot_dir / "04_role_geometry_drift.png",
        "final_residual_pca_roles": plot_dir / "05_final_residual_pca_roles.png",
        "committed_memory_budget": plot_dir / "06_committed_memory_budget.png",
    }
    if plasticity_audit:
        paths["plasticity_audit"] = plot_dir / "07_plasticity_audit.png"
    plot_behavior_outcome(metrics=metrics, output_path=paths["behavior_outcome"])
    plot_stage_trajectory(
        controlled_trace=controlled_trace,
        output_path=paths["stage_trajectory"],
        memory_budget=memory_budget,
    )
    plot_protected_geometry_cka(
        role_geometry=role_feature_geometry["roles"],
        output_path=paths["protected_geometry_cka"],
    )
    plot_role_geometry_drift_heatmap(
        role_geometry=role_feature_geometry["roles"],
        output_path=paths["role_geometry_drift"],
    )
    plot_final_residual_pca_roles(
        models=models,
        groups=role_groups(encoded),
        device=device,
        output_path=paths["final_residual_pca_roles"],
    )
    plot_committed_memory_budget(
        controlled_trace=controlled_trace,
        output_path=paths["committed_memory_budget"],
        memory_budget=memory_budget,
    )
    if plasticity_audit:
        plot_plasticity_audit(
            controlled_trace=controlled_trace,
            output_path=paths["plasticity_audit"],
        )
    return {key: str(path) for key, path in paths.items()}


def compact(row: dict[str, float]) -> str:
    return "{:.3g}/{:.3f}".format(row["loss"], row["exact_match"])


def count_trainable_parameters(model: GCONativeTransformer) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_parameters(model: GCONativeTransformer) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def parameter_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    if not parameters:
        raise ValueError("parameter_l2_norm received an empty parameter list.")
    total = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        total = total + torch.sum(parameter.detach().to(dtype=torch.float32) ** 2)
    return float(torch.sqrt(total).detach().cpu())


def save_checkpoint(
    *,
    path: Path,
    model: GCONativeTransformer,
    args: argparse.Namespace,
    vocab_size: int,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "label": label,
            "model_state_dict": model.state_dict(),
            "native_gco_config": asdict(native_config(args)),
            "model_config": {
                "vocab_size": vocab_size,
                "d_model": args.d_model,
                "n_layers": args.layers,
                "n_heads": args.heads,
                "d_ff": args.d_ff,
                "max_seq_len": args.max_seq_len,
            },
        },
        path,
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    positive_int("base_word_target", args.base_word_target)
    positive_int("conversation_word_target", args.conversation_word_target)
    positive_int("conversation_stages", args.conversation_stages)
    positive_int("max_seq_len", args.max_seq_len)
    positive_int("lm_stride", args.lm_stride)
    positive_int("base_lm_max_windows", args.base_lm_max_windows)
    positive_int("conversation_lm_max_windows", args.conversation_lm_max_windows)
    positive_int("d_model", args.d_model)
    positive_int("layers", args.layers)
    positive_int("heads", args.heads)
    positive_int("d_ff", args.d_ff)
    positive_int("base_epochs", args.base_epochs)
    positive_int("cl_epochs", args.cl_epochs)
    positive_int("joint_epochs", args.joint_epochs)
    positive_int("adapter_epochs", args.adapter_epochs)
    positive_int("adapter_consolidation_epochs", args.adapter_consolidation_epochs)
    positive_int("adapter_rank", args.adapter_rank)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("print_every", args.print_every)
    positive_float("lr", args.lr)
    positive_float("adapter_lr", args.adapter_lr)
    positive_float("consolidation_lr", args.consolidation_lr)
    positive_float("adapter_scale", args.adapter_scale)
    nonnegative_float("weight_decay", args.weight_decay)
    positive_float("distill_temperature", args.distill_temperature)
    nonnegative_float("lambda_preserve", args.lambda_preserve)
    nonnegative_float("lambda_guard", args.lambda_guard)
    nonnegative_float("lambda_geometry_anchor", args.lambda_geometry_anchor)
    constraint_mode_parts(args.projected_constraint_mode)
    positive_float("projected_update_damping", args.projected_update_damping)
    nonnegative_float("projected_restore_strength", args.projected_restore_strength)
    bounded_float("plasticity_audit_rank_tolerance", args.plasticity_audit_rank_tolerance, 0.0, 1.0)
    if args.plasticity_audit_rank_tolerance <= 0.0:
        raise ValueError(
            f"plasticity_audit_rank_tolerance must be positive, got {args.plasticity_audit_rank_tolerance}."
        )
    nonnegative_float("lambda_adapter", args.lambda_adapter)
    nonnegative_float("lambda_adapter_distill", args.lambda_adapter_distill)
    bounded_float("commit_min_exact", args.commit_min_exact, 0.0, 1.0)
    bounded_float("commit_min_token_accuracy", args.commit_min_token_accuracy, 0.0, 1.0)
    nonnegative_float("commit_max_loss", args.commit_max_loss)
    nonnegative_int("commit_memory_budget", args.commit_memory_budget)
    positive_float("grad_clip", args.grad_clip)
    bounded_float("init_topology", args.init_topology, 0.0, 1.0)
    bounded_float("formation_weight_mix", args.formation_weight_mix, 0.0, 1.0)
    bounded_float("formation_row_mix", args.formation_row_mix, 0.0, 1.0)
    bounded_float("formation_col_mix", args.formation_col_mix, 0.0, 1.0)
    bounded_float("formation_module_mix", args.formation_module_mix, 0.0, 1.0)
    if args.commit_memory_budget > 0 and not args.dynamic_committed_anchors:
        raise ValueError("--commit-memory-budget requires --dynamic-committed-anchors.")
    if args.plot_dir is not None and not args.diagnostics:
        raise ValueError("--plot-dir requires --diagnostics so stage trajectories can be plotted.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer must define [PAD].")
    vocab_size = tokenizer.get_vocab_size()
    protocol = build_protocol(args)
    encoded = encode_protocol(
        protocol,
        tokenizer,
        max_seq_len=args.max_seq_len,
        lm_stride=args.lm_stride,
        base_lm_max_windows=args.base_lm_max_windows,
        conversation_lm_max_windows=args.conversation_lm_max_windows,
    )

    print("MINI CONTROLLED CL WORLD DEMO")
    print("=" * 112)
    print(
        f"device={device} vocab={vocab_size} d_model={args.d_model} layers={args.layers} "
        f"heads={args.heads} d_ff={args.d_ff} ctx={args.max_seq_len}"
    )
    print(
        f"base_words={words(protocol.base_text)} conversation_words={words(protocol.conversation_text)} "
        f"base_train={len(encoded['base_train'])} update_train={len(encoded['update_train'])} "
        f"final_train={len(encoded['final_train'])}"
    )
    print(
        f"base_lm_windows={len(encoded['base_lm'])} conversation_stages={len(encoded['conversation_stage_train'])} "
        f"stage_lm_windows={[len(stage) for stage in encoded['conversation_lm_stages']]} "
        f"stage_train_examples={[len(stage) for stage in encoded['conversation_stage_train']]}"
    )
    print(
        f"controlled_update_mode={args.controlled_update_mode} "
        f"constraint_mode={args.projected_constraint_mode} "
        f"projected_damping={args.projected_update_damping} restore={args.projected_restore_strength}"
    )
    if args.diagnostics:
        print(
            "eval_examples preserve={} guard={} changed={} new={} composition={} obsolete={}".format(
                len(encoded["preserve_eval"]),
                len(encoded["guard_eval"]),
                len(encoded["changed_eval"]),
                len(encoded["new_eval"]),
                len(encoded["composition_eval"]),
                len(encoded["obsolete_eval"]),
            )
        )
    diagnostic_geometry_examples = collect_geometry_examples(encoded) if args.diagnostic_stage_geometry else []

    base_model = make_model(args, vocab_size=vocab_size, device=device, seed=args.seed + 10)
    set_only_native_weights_trainable(base_model)
    print(f"parameters total={count_parameters(base_model)} trainable_native={count_trainable_parameters(base_model)}")
    base_trace = train_plain(
        args=args,
        model=base_model,
        examples=encoded["base_train"],
        pad_id=pad_id,
        device=device,
        epochs=args.base_epochs,
        seed=args.seed + 20,
        label="base",
    )
    base_metrics = evaluate_named_groups(
        model=base_model,
        encoded=encoded,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )

    naive = clone_model(base_model, args, vocab_size=vocab_size, device=device, seed=args.seed + 30)
    naive_trace: list[dict[str, Any]] = []
    for stage_index, stage_examples in enumerate(encoded["conversation_stage_train"], start=1):
        stage_before_metrics = (
            evaluate_named_groups(
                model=naive,
                encoded=encoded,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            if args.diagnostics
            else None
        )
        pre_stage_for_geometry = (
            clone_model(naive, args, vocab_size=vocab_size, device=device, seed=args.seed + 300 + stage_index)
            if args.diagnostic_stage_geometry
            else None
        )
        trace = train_plain(
            args=args,
            model=naive,
            examples=stage_examples,
            pad_id=pad_id,
            device=device,
            epochs=args.cl_epochs,
            seed=args.seed + 40 + stage_index,
            label=f"naive_stage_{stage_index}",
        )
        stage_diagnostics: dict[str, Any] = {}
        if args.diagnostics:
            stage_after_metrics = evaluate_named_groups(
                model=naive,
                encoded=encoded,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            if stage_before_metrics is None:
                raise RuntimeError("Naive stage diagnostics missing before metrics.")
            print_metric_delta(
                label=f"naive stage {stage_index}",
                before=stage_before_metrics,
                after=stage_after_metrics,
            )
            stage_diagnostics["before_metrics"] = stage_before_metrics
            stage_diagnostics["after_metrics"] = stage_after_metrics
            stage_diagnostics["eval_delta"] = summarize_eval_delta(stage_before_metrics, stage_after_metrics)
        if args.diagnostic_stage_geometry:
            if pre_stage_for_geometry is None:
                raise RuntimeError("Naive stage geometry requested without pre-stage model.")
            stage_geometry = geometry_report(
                reference=pre_stage_for_geometry,
                candidate=naive,
                examples=diagnostic_geometry_examples,
                device=device,
            )
            print_geometry_delta(label=f"naive stage {stage_index}", report=stage_geometry)
            stage_diagnostics["residual_geometry_vs_pre_stage"] = stage_geometry
            stage_diagnostics["residual_geometry_mean_vs_pre_stage"] = summarize_geometry_means(stage_geometry)
        naive_trace.append(
            {
                "stage": stage_index,
                "trace": trace,
                "train_examples": len(stage_examples),
                "diagnostics": stage_diagnostics,
            }
        )
    naive_metrics = evaluate_named_groups(
        model=naive,
        encoded=encoded,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )

    controlled = clone_model(base_model, args, vocab_size=vocab_size, device=device, seed=args.seed + 50)
    controlled_trace: list[dict[str, Any]] = []
    controlled_committed_banks: list[dict[str, Any]] = []
    controlled_committed_keys: set[str] = set()
    for stage_index, stage_examples in enumerate(encoded["conversation_stage_train"], start=1):
        pre_stage = clone_model(controlled, args, vocab_size=vocab_size, device=device, seed=args.seed + 500 + stage_index)
        stage_before_metrics = (
            evaluate_named_groups(
                model=controlled,
                encoded=encoded,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            if args.diagnostics
            else None
        )
        preserve_teacher = collect_example_logits(
            pre_stage,
            encoded["preserve_eval"],
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        guard_teacher = collect_example_logits(
            pre_stage,
            encoded["guard_eval"],
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        preserve_states = collect_final_answer_states(
            pre_stage,
            encoded["preserve_eval"],
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        guard_states = collect_final_answer_states(
            pre_stage,
            encoded["guard_eval"],
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        preserve_anchor_examples, preserve_anchor_logits, preserve_anchor_states = combine_anchor_banks(
            base_examples=encoded["preserve_eval"],
            base_logits=preserve_teacher,
            base_states=preserve_states,
            committed_banks=controlled_committed_banks if args.dynamic_committed_anchors else [],
        )
        trace = train_controlled(
            args=args,
            model=controlled,
            update_examples=stage_examples,
            preserve_examples=preserve_anchor_examples,
            guard_examples=encoded["guard_eval"],
            preserve_logits=preserve_anchor_logits,
            guard_logits=guard_teacher,
            preserve_states=preserve_anchor_states,
            guard_states=guard_states,
            pad_id=pad_id,
            device=device,
            epochs=args.cl_epochs,
            seed=args.seed + 60 + stage_index,
        )
        stage_behavior_anchors = {
            "preserve": behavior_anchor_report(
                model=controlled,
                examples=encoded["preserve_eval"],
                teacher_logits=preserve_teacher,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                temperature=args.distill_temperature,
                device=device,
            ),
            "guard": behavior_anchor_report(
                model=controlled,
                examples=encoded["guard_eval"],
                teacher_logits=guard_teacher,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                temperature=args.distill_temperature,
                device=device,
            ),
        }
        committed_anchor_report: dict[str, Any] | None = None
        if args.dynamic_committed_anchors:
            selected, committed_anchor_report = select_commit_examples(
                model=controlled,
                examples=encoded["conversation_qa_stages"][stage_index - 1],
                existing_keys=controlled_committed_keys,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
                min_exact=args.commit_min_exact,
                min_token_accuracy=args.commit_min_token_accuracy,
                max_loss=args.commit_max_loss,
            )
            if selected:
                bank = build_committed_anchor_bank(
                    model=controlled,
                    selected=selected,
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                controlled_committed_banks.append(bank)
                controlled_committed_keys.update(encoded_example_key(example) for example in selected)
            if args.commit_memory_budget > 0:
                controlled_committed_banks, controlled_committed_keys, budget_report = apply_committed_memory_budget(
                    model=controlled,
                    committed_banks=controlled_committed_banks,
                    budget=args.commit_memory_budget,
                    selection=args.commit_selection,
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
            else:
                budget_report = {
                    "selection": args.commit_selection,
                    "budget": 0,
                    "before_count": len(controlled_committed_keys),
                    "after_count": len(controlled_committed_keys),
                    "dropped_count": 0,
                    "selected_category_counts": category_counts(committed_bank_examples(controlled_committed_banks)),
                    "dropped_category_counts": {},
                }
            committed_anchor_report["memory_budget"] = budget_report
            print(
                "controlled stage={} committed selected={} rejected={} total_committed={} budget_dropped={}".format(
                    stage_index,
                    committed_anchor_report["selected_count"],
                    committed_anchor_report["rejected_count"],
                    len(controlled_committed_keys),
                    budget_report["dropped_count"],
                )
            )
        stage_geometry_anchors = {
            "roles_vs_pre_stage": grouped_geometry_report(
                reference=pre_stage,
                candidate=controlled,
                groups=role_groups(encoded),
                device=device,
            ),
                "features_vs_pre_stage": grouped_geometry_report(
                    reference=pre_stage,
                    candidate=controlled,
                    groups=feature_groups(encoded),
                    device=device,
                ),
            }
        stage_diagnostics: dict[str, Any] = {}
        if args.diagnostics:
            stage_after_metrics = evaluate_named_groups(
                model=controlled,
                encoded=encoded,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            if stage_before_metrics is None:
                raise RuntimeError("Controlled stage diagnostics missing before metrics.")
            print_metric_delta(
                label=f"controlled stage {stage_index}",
                before=stage_before_metrics,
                after=stage_after_metrics,
            )
            stage_diagnostics["before_metrics"] = stage_before_metrics
            stage_diagnostics["after_metrics"] = stage_after_metrics
            stage_diagnostics["eval_delta"] = summarize_eval_delta(stage_before_metrics, stage_after_metrics)
        if args.diagnostic_stage_geometry:
            stage_geometry = geometry_report(
                reference=pre_stage,
                candidate=controlled,
                examples=diagnostic_geometry_examples,
                device=device,
            )
            print_geometry_delta(label=f"controlled stage {stage_index}", report=stage_geometry)
            stage_diagnostics["residual_geometry_vs_pre_stage"] = stage_geometry
            stage_diagnostics["residual_geometry_mean_vs_pre_stage"] = summarize_geometry_means(stage_geometry)
        controlled_trace.append(
            {
                "stage": stage_index,
                "trace": trace,
                "train_examples": len(stage_examples),
                "behavior_anchor_report": stage_behavior_anchors,
                "geometry_anchor_report": stage_geometry_anchors,
                "committed_anchor_report": committed_anchor_report,
                "committed_anchor_count_after_stage": len(controlled_committed_keys),
                "diagnostics": stage_diagnostics,
            }
        )
    controlled_metrics = evaluate_named_groups(
        model=controlled,
        encoded=encoded,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )

    adapter_controlled: GCONativeTransformer | None = None
    adapter_trace: list[dict[str, Any]] = []
    adapter_metrics: dict[str, dict[str, float]] | None = None
    if args.include_adapter:
        adapter_controlled = clone_model(base_model, args, vocab_size=vocab_size, device=device, seed=args.seed + 90)
        adapter_committed_banks: list[dict[str, Any]] = []
        adapter_committed_keys: set[str] = set()
        for stage_index, stage_examples in enumerate(encoded["conversation_stage_train"], start=1):
            pre_stage = clone_model(adapter_controlled, args, vocab_size=vocab_size, device=device, seed=args.seed + 900 + stage_index)
            stage_before_metrics = (
                evaluate_named_groups(
                    model=adapter_controlled,
                    encoded=encoded,
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                if args.diagnostics
                else None
            )
            preserve_teacher = collect_example_logits(
                pre_stage,
                encoded["preserve_eval"],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            guard_teacher = collect_example_logits(
                pre_stage,
                encoded["guard_eval"],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            preserve_states = collect_final_answer_states(
                pre_stage,
                encoded["preserve_eval"],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            guard_states = collect_final_answer_states(
                pre_stage,
                encoded["guard_eval"],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            preserve_anchor_examples, preserve_anchor_logits, preserve_anchor_states = combine_anchor_banks(
                base_examples=encoded["preserve_eval"],
                base_logits=preserve_teacher,
                base_states=preserve_states,
                committed_banks=adapter_committed_banks if args.dynamic_committed_anchors else [],
            )
            adapter = FinalResidualAdapter(d_model=args.d_model, rank=args.adapter_rank, scale=args.adapter_scale).to(device)
            wrapped = AdapterWrappedTransformer(pre_stage, adapter).to(device)
            plastic_trace = train_plastic_adapter_stage(
                args=args,
                wrapped=wrapped,
                adapter=adapter,
                update_examples=stage_examples,
                preserve_examples=preserve_anchor_examples,
                guard_examples=encoded["guard_eval"],
                preserve_logits=preserve_anchor_logits,
                guard_logits=guard_teacher,
                pad_id=pad_id,
                device=device,
                epochs=args.adapter_epochs,
                seed=args.seed + 1000 + stage_index,
                stage_index=stage_index,
            )
            adapter_behavior_metrics = evaluate_named_groups(
                model=wrapped,
                encoded=encoded,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            stage_diagnostics: dict[str, Any] = {}
            if args.diagnostics:
                if stage_before_metrics is None:
                    raise RuntimeError("Adapter stage diagnostics missing before metrics.")
                print_metric_delta(
                    label=f"adapter stage {stage_index} plastic side path",
                    before=stage_before_metrics,
                    after=adapter_behavior_metrics,
                )
                stage_diagnostics["before_metrics"] = stage_before_metrics
                stage_diagnostics["plastic_metrics"] = adapter_behavior_metrics
                stage_diagnostics["plastic_eval_delta"] = summarize_eval_delta(
                    stage_before_metrics,
                    adapter_behavior_metrics,
                )
            consolidated = clone_model(pre_stage, args, vocab_size=vocab_size, device=device, seed=args.seed + 1100 + stage_index)
            consolidation_trace = train_adapter_consolidation_stage(
                args=args,
                core=consolidated,
                adapter_teacher=wrapped,
                update_examples=stage_examples,
                preserve_examples=preserve_anchor_examples,
                guard_examples=encoded["guard_eval"],
                preserve_logits=preserve_anchor_logits,
                guard_logits=guard_teacher,
                preserve_states=preserve_anchor_states,
                guard_states=guard_states,
                pad_id=pad_id,
                device=device,
                epochs=args.adapter_consolidation_epochs,
                seed=args.seed + 1200 + stage_index,
                stage_index=stage_index,
            )
            adapter_controlled = consolidated
            stage_behavior_anchors = {
                "preserve": behavior_anchor_report(
                    model=adapter_controlled,
                    examples=encoded["preserve_eval"],
                    teacher_logits=preserve_teacher,
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    temperature=args.distill_temperature,
                    device=device,
                ),
                "guard": behavior_anchor_report(
                    model=adapter_controlled,
                    examples=encoded["guard_eval"],
                    teacher_logits=guard_teacher,
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    temperature=args.distill_temperature,
                    device=device,
                ),
            }
            stage_geometry_anchors = {
                "roles_vs_pre_stage": grouped_geometry_report(
                    reference=pre_stage,
                    candidate=adapter_controlled,
                    groups=role_groups(encoded),
                    device=device,
                ),
                "features_vs_pre_stage": grouped_geometry_report(
                    reference=pre_stage,
                    candidate=adapter_controlled,
                    groups=feature_groups(encoded),
                device=device,
            ),
        }
            committed_anchor_report: dict[str, Any] | None = None
            if args.dynamic_committed_anchors:
                selected, committed_anchor_report = select_commit_examples(
                    model=adapter_controlled,
                    examples=encoded["conversation_qa_stages"][stage_index - 1],
                    existing_keys=adapter_committed_keys,
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    device=device,
                    min_exact=args.commit_min_exact,
                    min_token_accuracy=args.commit_min_token_accuracy,
                    max_loss=args.commit_max_loss,
                )
                if selected:
                    bank = build_committed_anchor_bank(
                        model=adapter_controlled,
                        selected=selected,
                        pad_id=pad_id,
                        batch_size=args.eval_batch_size,
                        device=device,
                    )
                    adapter_committed_banks.append(bank)
                    adapter_committed_keys.update(encoded_example_key(example) for example in selected)
                if args.commit_memory_budget > 0:
                    adapter_committed_banks, adapter_committed_keys, budget_report = apply_committed_memory_budget(
                        model=adapter_controlled,
                        committed_banks=adapter_committed_banks,
                        budget=args.commit_memory_budget,
                        selection=args.commit_selection,
                        pad_id=pad_id,
                        batch_size=args.eval_batch_size,
                        device=device,
                    )
                else:
                    budget_report = {
                        "selection": args.commit_selection,
                        "budget": 0,
                        "before_count": len(adapter_committed_keys),
                        "after_count": len(adapter_committed_keys),
                        "dropped_count": 0,
                        "selected_category_counts": category_counts(committed_bank_examples(adapter_committed_banks)),
                        "dropped_category_counts": {},
                    }
                committed_anchor_report["memory_budget"] = budget_report
                print(
                    "adapter stage={} committed selected={} rejected={} total_committed={} budget_dropped={}".format(
                        stage_index,
                        committed_anchor_report["selected_count"],
                        committed_anchor_report["rejected_count"],
                        len(adapter_committed_keys),
                        budget_report["dropped_count"],
                    )
                )
            if args.diagnostics:
                consolidated_metrics = evaluate_named_groups(
                    model=adapter_controlled,
                    encoded=encoded,
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                print_metric_delta(
                    label=f"adapter stage {stage_index} consolidated core",
                    before=stage_before_metrics,
                    after=consolidated_metrics,
                )
                stage_diagnostics["consolidated_metrics"] = consolidated_metrics
                stage_diagnostics["consolidated_eval_delta"] = summarize_eval_delta(
                    stage_before_metrics,
                    consolidated_metrics,
                )
            if args.diagnostic_stage_geometry:
                stage_geometry = geometry_report(
                    reference=pre_stage,
                    candidate=adapter_controlled,
                    examples=diagnostic_geometry_examples,
                    device=device,
                )
                print_geometry_delta(label=f"adapter stage {stage_index} consolidated core", report=stage_geometry)
                stage_diagnostics["residual_geometry_vs_pre_stage"] = stage_geometry
                stage_diagnostics["residual_geometry_mean_vs_pre_stage"] = summarize_geometry_means(stage_geometry)
            adapter_trace.append(
                {
                    "stage": stage_index,
                    "train_examples": len(stage_examples),
                    "plastic_trace": plastic_trace,
                    "consolidation_trace": consolidation_trace,
                    "adapter_behavior_metrics_before_consolidation": adapter_behavior_metrics,
                    "behavior_anchor_report": stage_behavior_anchors,
                    "geometry_anchor_report": stage_geometry_anchors,
                    "committed_anchor_report": committed_anchor_report,
                    "committed_anchor_count_after_stage": len(adapter_committed_keys),
                    "diagnostics": stage_diagnostics,
                }
            )
        adapter_metrics = evaluate_named_groups(
            model=adapter_controlled,
            encoded=encoded,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )

    joint = make_model(args, vocab_size=vocab_size, device=device, seed=args.seed + 70)
    joint_trace = train_plain(
        args=args,
        model=joint,
        examples=encoded["final_train"],
        pad_id=pad_id,
        device=device,
        epochs=args.joint_epochs,
        seed=args.seed + 80,
        label="joint",
    )
    joint_metrics = evaluate_named_groups(
        model=joint,
        encoded=encoded,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )

    geometry_examples = collect_geometry_examples(encoded)
    geometry = {
        "naive_vs_base": geometry_report(reference=base_model, candidate=naive, examples=geometry_examples, device=device),
        "controlled_vs_base": geometry_report(reference=base_model, candidate=controlled, examples=geometry_examples, device=device),
        "joint_vs_base": geometry_report(reference=base_model, candidate=joint, examples=geometry_examples, device=device),
        "controlled_vs_joint": geometry_report(reference=joint, candidate=controlled, examples=geometry_examples, device=device),
    }
    if adapter_controlled is not None:
        geometry["adapter_vs_base"] = geometry_report(
            reference=base_model,
            candidate=adapter_controlled,
            examples=geometry_examples,
            device=device,
        )
        geometry["adapter_vs_joint"] = geometry_report(
            reference=joint,
            candidate=adapter_controlled,
            examples=geometry_examples,
            device=device,
        )
    role_feature_geometry = {
        "roles": {
            "naive_vs_base": grouped_geometry_report(reference=base_model, candidate=naive, groups=role_groups(encoded), device=device),
            "controlled_vs_base": grouped_geometry_report(reference=base_model, candidate=controlled, groups=role_groups(encoded), device=device),
            "joint_vs_base": grouped_geometry_report(reference=base_model, candidate=joint, groups=role_groups(encoded), device=device),
            "controlled_vs_joint": grouped_geometry_report(reference=joint, candidate=controlled, groups=role_groups(encoded), device=device),
        },
        "features": {
            "naive_vs_base": grouped_geometry_report(reference=base_model, candidate=naive, groups=feature_groups(encoded), device=device),
            "controlled_vs_base": grouped_geometry_report(reference=base_model, candidate=controlled, groups=feature_groups(encoded), device=device),
            "joint_vs_base": grouped_geometry_report(reference=base_model, candidate=joint, groups=feature_groups(encoded), device=device),
            "controlled_vs_joint": grouped_geometry_report(reference=joint, candidate=controlled, groups=feature_groups(encoded), device=device),
        },
    }
    if adapter_controlled is not None:
        role_feature_geometry["roles"]["adapter_vs_base"] = grouped_geometry_report(
            reference=base_model,
            candidate=adapter_controlled,
            groups=role_groups(encoded),
            device=device,
        )
        role_feature_geometry["roles"]["adapter_vs_joint"] = grouped_geometry_report(
            reference=joint,
            candidate=adapter_controlled,
            groups=role_groups(encoded),
            device=device,
        )
        role_feature_geometry["features"]["adapter_vs_base"] = grouped_geometry_report(
            reference=base_model,
            candidate=adapter_controlled,
            groups=feature_groups(encoded),
            device=device,
        )
        role_feature_geometry["features"]["adapter_vs_joint"] = grouped_geometry_report(
            reference=joint,
            candidate=adapter_controlled,
            groups=feature_groups(encoded),
            device=device,
        )

    checkpoint_dir = args.checkpoint_dir
    save_checkpoint(path=checkpoint_dir / "mini-cl-base.pt", model=base_model, args=args, vocab_size=vocab_size, label="base")
    save_checkpoint(path=checkpoint_dir / "mini-cl-naive.pt", model=naive, args=args, vocab_size=vocab_size, label="naive")
    save_checkpoint(path=checkpoint_dir / "mini-cl-controlled.pt", model=controlled, args=args, vocab_size=vocab_size, label="controlled")
    if adapter_controlled is not None:
        save_checkpoint(path=checkpoint_dir / "mini-cl-adapter-controlled.pt", model=adapter_controlled, args=args, vocab_size=vocab_size, label="adapter_controlled")
    save_checkpoint(path=checkpoint_dir / "mini-cl-joint.pt", model=joint, args=args, vocab_size=vocab_size, label="joint")

    summary = {
        "question": "Can a single small transformer learn beyond context while preserving, changing, and guarding selected behavior?",
        "model_config": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "parameter_count": count_parameters(base_model),
            "trainable_native_parameter_count": count_trainable_parameters(base_model),
        },
        "protocol": {
            "base_word_count": words(protocol.base_text),
            "conversation_word_count": words(protocol.conversation_text),
            "base_text": protocol.base_text,
            "conversation_text": protocol.conversation_text,
            "conversation_stage_texts": protocol.conversation_stage_texts,
            "base_train": [asdict(example) for example in protocol.base_train],
            "conversation_stage_train": [
                [asdict(example) for example in stage]
                for stage in protocol.conversation_stage_train
            ],
            "final_train": [asdict(example) for example in protocol.final_train],
            "preserve_eval": [asdict(example) for example in protocol.preserve_eval],
            "guard_eval": [asdict(example) for example in protocol.guard_eval],
            "changed_eval": [asdict(example) for example in protocol.changed_eval],
            "new_eval": [asdict(example) for example in protocol.new_eval],
            "composition_eval": [asdict(example) for example in protocol.composition_eval],
            "obsolete_eval": [asdict(example) for example in protocol.obsolete_eval],
        },
        "hyperparameters": {
            "base_epochs": args.base_epochs,
            "cl_epochs": args.cl_epochs,
            "joint_epochs": args.joint_epochs,
            "batch_size": args.batch_size,
            "lm_stride": args.lm_stride,
            "base_lm_max_windows": args.base_lm_max_windows,
            "conversation_lm_max_windows": args.conversation_lm_max_windows,
            "conversation_stages": args.conversation_stages,
            "lr": args.lr,
            "adapter_lr": args.adapter_lr,
            "consolidation_lr": args.consolidation_lr,
            "optimizer": args.optimizer,
            "controlled_update_mode": args.controlled_update_mode,
            "projected_constraint_mode": args.projected_constraint_mode,
            "projected_solver": args.projected_solver,
            "projected_update_damping": args.projected_update_damping,
            "projected_restore_strength": args.projected_restore_strength,
            "plasticity_audit": args.plasticity_audit,
            "plasticity_audit_rank_tolerance": args.plasticity_audit_rank_tolerance,
            "lambda_preserve": args.lambda_preserve,
            "lambda_guard": args.lambda_guard,
            "lambda_geometry_anchor": args.lambda_geometry_anchor,
            "include_adapter": args.include_adapter,
            "adapter_epochs": args.adapter_epochs,
            "adapter_consolidation_epochs": args.adapter_consolidation_epochs,
            "adapter_rank": args.adapter_rank,
            "adapter_scale": args.adapter_scale,
            "lambda_adapter": args.lambda_adapter,
            "lambda_adapter_distill": args.lambda_adapter_distill,
            "distill_temperature": args.distill_temperature,
            "dynamic_committed_anchors": args.dynamic_committed_anchors,
            "commit_min_exact": args.commit_min_exact,
            "commit_min_token_accuracy": args.commit_min_token_accuracy,
            "commit_max_loss": args.commit_max_loss,
            "commit_memory_budget": args.commit_memory_budget,
            "commit_selection": args.commit_selection,
        },
        "traces": {
            "base": base_trace,
            "naive": naive_trace,
            "controlled": controlled_trace,
            "adapter_controlled": adapter_trace,
            "joint": joint_trace,
        },
        "metrics": {
            "base": base_metrics,
            "naive": naive_metrics,
            "controlled": controlled_metrics,
            "adapter_controlled": adapter_metrics,
            "joint": joint_metrics,
        },
        "geometry": geometry,
        "role_feature_geometry": role_feature_geometry,
        "checkpoints": {
            "base": str(checkpoint_dir / "mini-cl-base.pt"),
            "naive": str(checkpoint_dir / "mini-cl-naive.pt"),
            "controlled": str(checkpoint_dir / "mini-cl-controlled.pt"),
            "adapter_controlled": str(checkpoint_dir / "mini-cl-adapter-controlled.pt") if adapter_controlled is not None else None,
            "joint": str(checkpoint_dir / "mini-cl-joint.pt"),
        },
    }
    if args.plot_dir is not None:
        summary["plots"] = write_summary_plots(
            plot_dir=args.plot_dir,
            metrics={
                "base": base_metrics,
                "naive": naive_metrics,
                "controlled": controlled_metrics,
                "joint": joint_metrics,
            },
            controlled_trace=controlled_trace,
            role_feature_geometry=role_feature_geometry,
            models={
                "naive": naive,
                "controlled": controlled,
            },
            encoded=encoded,
            device=device,
            memory_budget=args.commit_memory_budget,
            plasticity_audit=args.plasticity_audit,
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nMINI CONTROLLED CL WORLD SUMMARY")
    print("=" * 112)
    print(
        "{:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "method",
            "preserve",
            "guard",
            "changed",
            "new",
            "compose",
            "oldWrong",
        )
    )
    print(
        "{:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "",
            "loss/exact",
            "loss/exact",
            "loss/exact",
            "loss/exact",
            "loss/exact",
            "loss/exact",
        )
    )
    method_metric_rows: list[tuple[str, dict[str, dict[str, float]]]] = [
        ("base", base_metrics),
        ("naive", naive_metrics),
        ("controlled", controlled_metrics),
    ]
    if adapter_metrics is not None:
        method_metric_rows.append(("adapter", adapter_metrics))
    method_metric_rows.append(("joint", joint_metrics))
    for method, metrics in method_metric_rows:
        print(
            "{:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
                method,
                compact(metrics["preserve"]),
                compact(metrics["guard"]),
                compact(metrics["changed"]),
                compact(metrics["new"]),
                compact(metrics["composition"]),
                compact(metrics["obsolete_old_answer"]),
            )
        )

    def mean_geometry(label: str, key: str) -> float:
        rows = geometry[label].values()
        values = [float(row[key]) for row in rows]
        if not values:
            raise RuntimeError(f"No geometry values for {label}/{key}.")
        return sum(values) / float(len(values))

    print("\nRESIDUAL GEOMETRY MEAN")
    print("-" * 112)
    print("{:>24} {:>12} {:>12} {:>12}".format("comparison", "drift_rel", "cka", "rank_delta"))
    geometry_labels = ["naive_vs_base", "controlled_vs_base"]
    if adapter_controlled is not None:
        geometry_labels.append("adapter_vs_base")
    geometry_labels.extend(["joint_vs_base", "controlled_vs_joint"])
    if adapter_controlled is not None:
        geometry_labels.append("adapter_vs_joint")
    for label in geometry_labels:
        print(
            "{:>24} {:12.4f} {:12.4f} {:12.4f}".format(
                label,
                mean_geometry(label, "drift_relative"),
                mean_geometry(label, "cka"),
                mean_geometry(label, "rank_delta"),
            )
        )

    def mean_group_metric(section: str, comparison: str, metric: str) -> float:
        rows: list[float] = []
        groups = role_feature_geometry[section][comparison]["groups"]
        for layer_metrics in groups.values():
            for row in layer_metrics.values():
                rows.append(float(row[metric]))
        if not rows:
            raise RuntimeError(f"No grouped geometry values for {section}/{comparison}/{metric}.")
        return sum(rows) / float(len(rows))

    def mean_separation_metric(section: str, comparison: str, metric: str) -> float:
        rows = [
            float(row[metric])
            for row in role_feature_geometry[section][comparison]["separation"].values()
        ]
        if not rows:
            raise RuntimeError(f"No separation values for {section}/{comparison}/{metric}.")
        return sum(rows) / float(len(rows))

    print("\nROLE / FEATURE GEOMETRY MEAN")
    print("-" * 112)
    print(
        "{:>12} {:>24} {:>12} {:>12} {:>12} {:>12}".format(
            "section",
            "comparison",
            "cent_drift",
            "cent_cos",
            "group_cka",
            "sep_rel",
        )
    )
    for section in ["roles", "features"]:
        for comparison in geometry_labels:
            print(
                "{:>12} {:>24} {:12.4f} {:12.4f} {:12.4f} {:12.4f}".format(
                    section,
                    comparison,
                    mean_group_metric(section, comparison, "centroid_drift"),
                    mean_group_metric(section, comparison, "centroid_cosine"),
                    mean_group_metric(section, comparison, "cka"),
                    mean_separation_metric(section, comparison, "separation_drift_relative"),
                )
            )

    def stage_anchor_mean(stage_row: dict[str, Any], section: str, metric: str) -> float:
        report = stage_row["geometry_anchor_report"][section]
        values: list[float] = []
        for layer_rows in report["groups"].values():
            for row in layer_rows.values():
                values.append(float(row[metric]))
        if not values:
            raise RuntimeError(f"No stage anchor metric values for stage={stage_row['stage']} section={section} metric={metric}.")
        return sum(values) / float(len(values))

    print("\nCONTROLLED REBASING ANCHORS BY STAGE")
    print("-" * 112)
    print(
        "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "stage",
            "presKL",
            "guardKL",
            "roleDrift",
            "featDrift",
            "geomLoss",
        )
    )
    for stage_row in controlled_trace:
        last_trace = stage_row["trace"][-1]
        print(
            "{:8d} {:12.5g} {:12.5g} {:12.4f} {:12.4f} {:12.5g}".format(
                int(stage_row["stage"]),
                float(stage_row["behavior_anchor_report"]["preserve"]["mean_kl"]),
                float(stage_row["behavior_anchor_report"]["guard"]["mean_kl"]),
                stage_anchor_mean(stage_row, "roles_vs_pre_stage", "centroid_drift"),
                stage_anchor_mean(stage_row, "features_vs_pre_stage", "centroid_drift"),
                float(last_trace["geometry"]),
            )
        )

    if args.plasticity_audit:
        print("\nCONTROLLED PLASTICITY AUDIT BY STAGE")
        print("-" * 112)
        print(
            "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
                "stage",
                "safe/raw",
                "final/raw",
                "removed",
                "rank",
                "rows",
                "redund",
                "rawCos",
            )
        )
        for stage_row in controlled_trace:
            last_trace = stage_row["trace"][-1]
            print(
                "{:8d} {:10.4f} {:10.4f} {:10.4f} {:10.3f} {:10.1f} {:10.4f} {:10.4f}".format(
                    int(stage_row["stage"]),
                    float(last_trace["safe_grad_fraction"]),
                    float(last_trace["final_update_fraction"]),
                    float(last_trace["projection_removed_fraction"]),
                    float(last_trace["constraint_effective_rank"]),
                    float(last_trace["constraint_count"]),
                    float(last_trace["constraint_redundancy"]),
                    float(last_trace["raw_constraint_cosine_mean"]),
                )
            )

    def print_commit_summary(label: str, rows: list[dict[str, Any]]) -> None:
        if not args.dynamic_committed_anchors:
            return
        print(f"\n{label.upper()} COMMITTED ANCHORS BY STAGE")
        print("-" * 112)
        print(
            "{:>8} {:>12} {:>12} {:>12} {:>24}".format(
                "stage",
                "selected",
                "rejected",
                "total",
                "selected_categories",
            )
        )
        for stage_row in rows:
            report = stage_row.get("committed_anchor_report")
            if report is None:
                raise RuntimeError(f"Missing committed-anchor report for {label} stage {stage_row['stage']}.")
            category_counts = report["selected_category_counts"]
            if category_counts:
                category_text = ",".join(f"{key}:{value}" for key, value in sorted(category_counts.items()))
            else:
                category_text = "-"
            print(
                "{:8d} {:12d} {:12d} {:12d} {:>24}".format(
                    int(stage_row["stage"]),
                    int(report["selected_count"]),
                    int(report["rejected_count"]),
                    int(stage_row["committed_anchor_count_after_stage"]),
                    category_text,
                )
            )

    print_commit_summary("controlled", controlled_trace)

    if adapter_trace:
        print("\nADAPTER REBASING ANCHORS BY STAGE")
        print("-" * 112)
        print(
            "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
                "stage",
                "presKL",
                "guardKL",
                "roleDrift",
                "featDrift",
                "plastNew",
                "consAD",
            )
        )
        for stage_row in adapter_trace:
            plastic_last = stage_row["plastic_trace"][-1]
            consolidation_last = stage_row["consolidation_trace"][-1]
            print(
                "{:8d} {:12.5g} {:12.5g} {:12.4f} {:12.4f} {:12.5g} {:12.5g}".format(
                    int(stage_row["stage"]),
                    float(stage_row["behavior_anchor_report"]["preserve"]["mean_kl"]),
                    float(stage_row["behavior_anchor_report"]["guard"]["mean_kl"]),
                    stage_anchor_mean(stage_row, "roles_vs_pre_stage", "centroid_drift"),
                    stage_anchor_mean(stage_row, "features_vs_pre_stage", "centroid_drift"),
                    float(plastic_last["new"]),
                    float(consolidation_last["adapter_distill"]),
                )
            )
        print_commit_summary("adapter", adapter_trace)
    print(f"wrote_json={args.output_json}")
    print(f"wrote_checkpoints={checkpoint_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/mini-cl-world-demo-seed0.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints/mini-cl-world-demo-seed0"))
    parser.add_argument("--base-word-target", type=int, default=5000)
    parser.add_argument("--conversation-word-target", type=int, default=1800)
    parser.add_argument("--conversation-stages", type=int, default=3)
    parser.add_argument("--lm-stride", type=int, default=16)
    parser.add_argument("--base-lm-max-windows", type=int, default=384)
    parser.add_argument("--conversation-lm-max-windows", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--base-epochs", type=int, default=250)
    parser.add_argument("--cl-epochs", type=int, default=180)
    parser.add_argument("--joint-epochs", type=int, default=300)
    parser.add_argument("--include-adapter", action="store_true")
    parser.add_argument("--adapter-epochs", type=int, default=120)
    parser.add_argument("--adapter-consolidation-epochs", type=int, default=120)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--controlled-update-mode", choices=["loss", "projected_invariant_tangent"], default="loss")
    parser.add_argument(
        "--projected-constraint-mode",
        choices=["scalar", "category", "category_centroid", "category_centroid_separation"],
        default="scalar",
    )
    parser.add_argument("--projected-solver", choices=["sequential", "gram"], default="sequential")
    parser.add_argument("--projected-update-damping", type=float, default=1e-6)
    parser.add_argument("--projected-restore-strength", type=float, default=0.0)
    parser.add_argument("--plasticity-audit", action="store_true")
    parser.add_argument("--plasticity-audit-rank-tolerance", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--consolidation-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-guard", type=float, default=1.0)
    parser.add_argument("--lambda-geometry-anchor", type=float, default=0.05)
    parser.add_argument("--lambda-adapter", type=float, default=1e-4)
    parser.add_argument("--lambda-adapter-distill", type=float, default=1.0)
    parser.add_argument("--dynamic-committed-anchors", action="store_true")
    parser.add_argument("--commit-min-exact", type=float, default=1.0)
    parser.add_argument("--commit-min-token-accuracy", type=float, default=1.0)
    parser.add_argument("--commit-max-loss", type=float, default=0.5)
    parser.add_argument("--commit-memory-budget", type=int, default=0)
    parser.add_argument("--commit-selection", choices=["diverse_by_category_then_loss"], default="diverse_by_category_then_loss")
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--init-topology", type=float, default=1.0)
    parser.add_argument("--formation-weight-mix", type=float, default=1.0)
    parser.add_argument("--formation-row-mix", type=float, default=1.0)
    parser.add_argument("--formation-col-mix", type=float, default=1.0)
    parser.add_argument("--formation-module-mix", type=float, default=1.0)
    parser.add_argument("--formation-multiscale-pooling", choices=["none", "mean"], default="mean")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostic-stage-geometry", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
