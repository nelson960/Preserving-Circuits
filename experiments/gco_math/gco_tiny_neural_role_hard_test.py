"""Hard role-separation test for neural-controller continual learning.

This experiment makes the behavior roles mechanically different:

    preserve: strong behavior distillation during every new-learning update.
    guard: no normal training; monitor drift and restore only if drift is high.
    drop: actively suppress the old answer.

The previous neural role-controller test showed that a small controller can
often infer useful roles. This test asks whether those roles matter when wrong
decisions have visible consequences.
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
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_tiny_auto_role_controller import (
    build_raw_stream,
    encode_raw_groups,
    evaluate_category_breakdown,
    evaluate_role_summary,
    examples_by_stage1_person,
    oracle_roles,
    random_roles_matching_counts,
    role_match_report,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    load_checkpoint,
    set_only_native_weights_trainable,
    trainable_weight_parameters,
)
from experiments.gco_math.gco_tiny_neural_role_controller import (
    FEATURE_NAMES,
    NeuralRoleController,
    observed_evidence_counts,
    neural_role_prediction,
    parse_people,
    train_neural_controller,
)
from experiments.gco_math.gco_mini_cl_world_demo import (
    geometry_report,
    grouped_geometry_report,
    summarize_geometry_means,
)
from experiments.gco_math.gco_tiny_recursive_controlled_architecture import (
    cap_examples,
    drop_suppression_loss,
    train_bootstrap_stage,
)
from experiments.gco_math.gco_tiny_reusable_computation_cl import (
    EncodedExample,
    QAExample,
    batch_examples,
    collect_example_logits,
    distillation_loss_for_examples,
    evaluate_examples,
    make_model_from_config,
    make_optimizer_for_model,
    masked_ce_loss,
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


def probability(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    allowed = {"naive", "oracle", "neural", "random", "consequence"}
    unknown = sorted(set(methods).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; allowed={sorted(allowed)}.")
    return methods


def cap_example_logit_pairs(
    examples: list[EncodedExample],
    logits: list[torch.Tensor],
    *,
    budget: int,
) -> tuple[list[EncodedExample], list[torch.Tensor]]:
    positive_int("budget", budget)
    if len(examples) != len(logits):
        raise ValueError(f"Example/logit length mismatch: {len(examples)} vs {len(logits)}.")
    if len(examples) <= budget:
        return list(examples), [row.clone() for row in logits]
    indices = torch.linspace(0, len(examples) - 1, steps=budget).round().to(dtype=torch.long).unique(sorted=True)
    if indices.numel() != budget:
        raise RuntimeError(f"Budget selector returned {indices.numel()} examples for budget={budget}.")
    selected = [int(index) for index in indices.tolist()]
    return [examples[index] for index in selected], [logits[index].clone() for index in selected]


def build_role_groups_allow_empty(
    *,
    role_by_person: dict[str, str],
    encoded_people: dict[str, list[EncodedExample]],
) -> dict[str, list[EncodedExample]]:
    groups: dict[str, list[EncodedExample]] = {"preserve": [], "drop": [], "neutral": []}
    for person, role in sorted(role_by_person.items()):
        if person not in encoded_people:
            raise KeyError(f"Missing encoded examples for person {person!r}.")
        if role == "preserve":
            groups["preserve"].extend(encoded_people[person])
        elif role == "drop":
            groups["drop"].extend(encoded_people[person])
        elif role == "guard":
            groups["neutral"].extend(encoded_people[person])
        else:
            raise ValueError(f"Unknown role {role!r} for person {person}.")
    return groups


def mean_distillation_kl(
    *,
    model: torch.nn.Module,
    examples: list[EncodedExample],
    teacher_logits: list[torch.Tensor],
    pad_id: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> float:
    if not examples:
        raise ValueError("Cannot compute KL for an empty example set.")
    if len(examples) != len(teacher_logits):
        raise ValueError(f"Example/logit length mismatch: {len(examples)} vs {len(teacher_logits)}.")
    positive_int("batch_size", batch_size)
    positive_float("temperature", temperature)
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(examples)), dtype=torch.long)
            inputs, _targets, _mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
            logits = model(inputs)
            losses: list[torch.Tensor] = []
            for row_index, example in enumerate(selected):
                length = len(example.target_ids)
                current = logits[row_index, :length].unsqueeze(0)
                teacher = teacher_logits[int(indices[row_index].item())].to(device).unsqueeze(0)
                current_log_probs = F.log_softmax(current / temperature, dim=-1)
                teacher_probs = F.softmax(teacher / temperature, dim=-1)
                losses.append(
                    F.kl_div(current_log_probs, teacher_probs, reduction="batchmean")
                    * (temperature * temperature)
                )
            if not losses:
                raise RuntimeError("No KL losses were built.")
            total += float(torch.stack(losses).sum().detach().cpu())
            count += len(losses)
    if count <= 0:
        raise RuntimeError("KL evaluation saw zero examples.")
    return total / float(count)


def sampled_distillation_loss(
    *,
    model: torch.nn.Module,
    examples: list[EncodedExample],
    teacher_logits: list[torch.Tensor],
    pad_id: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if not examples:
        raise ValueError("Cannot sample a distillation loss from an empty example set.")
    if len(examples) != len(teacher_logits):
        raise ValueError(f"Example/logit length mismatch: {len(examples)} vs {len(teacher_logits)}.")
    positive_int("batch_size", batch_size)
    indices = torch.randint(
        low=0,
        high=len(examples),
        size=(batch_size,),
        generator=generator,
        device=torch.device("cpu"),
    )
    inputs, _targets, _mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
    current = model(inputs)
    return distillation_loss_for_examples(
        current,
        selected,
        teacher_logits,
        indices,
        temperature=temperature,
        device=device,
    )


def clone_current_model(
    model: torch.nn.Module,
    *,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    clone = make_model_from_config(checkpoint=checkpoint, device=device, seed=0)
    missing, unexpected = clone.load_state_dict(model.state_dict(), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Clone state mismatch: missing={missing}, unexpected={unexpected}.")
    clone.eval()
    return clone


def preview_drop_update(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    person_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    seed: int,
) -> torch.nn.Module:
    if not person_examples:
        raise ValueError("Cannot preview a drop update with an empty person example set.")
    clone = clone_current_model(model, checkpoint=checkpoint, device=device)
    set_only_native_weights_trainable(clone)
    optimizer = make_optimizer_for_model(args, trainable_weight_parameters(clone))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for _epoch in range(args.consequence_preview_epochs):
        permutation = torch.randperm(len(person_examples), generator=generator)
        for start in range(0, len(person_examples), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(person_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = clone(inputs)
            loss = drop_suppression_loss(
                logits=logits,
                targets=targets,
                mask=mask,
                target_probability=args.drop_target_probability,
            )
            loss = args.drop_loss_weight * loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(clone), args.grad_clip)
            optimizer.step()
    clone.eval()
    return clone


def mean_people_kl(
    *,
    model: torch.nn.Module,
    encoded_people: dict[str, list[EncodedExample]],
    reference_logits_by_person: dict[str, list[torch.Tensor]],
    people: list[str],
    pad_id: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> float:
    if not people:
        return 0.0
    values = [
        mean_distillation_kl(
            model=model,
            examples=encoded_people[person],
            teacher_logits=reference_logits_by_person[person],
            pad_id=pad_id,
            batch_size=batch_size,
            temperature=temperature,
            device=device,
        )
        for person in people
    ]
    return float(sum(values) / float(len(values)))


def role_action_score(
    *,
    action: str,
    useful_score: float,
    obsolete_score: float,
    exact_after: float,
    person_kl_after: float,
    global_damage_kl: float,
    capacity_pressure: float,
    args: argparse.Namespace,
) -> float:
    conflict = min(useful_score, obsolete_score)
    keep_value = useful_score * exact_after
    drop_value = obsolete_score * (1.0 - exact_after)
    damage_penalty = args.consequence_global_damage_weight * global_damage_kl
    if action == "preserve":
        capacity_cost = args.consequence_preserve_capacity_cost * capacity_pressure * (1.0 - useful_score)
        return keep_value - damage_penalty - capacity_cost
    if action == "guard":
        capacity_cost = args.consequence_guard_capacity_cost * capacity_pressure
        return (
            keep_value
            + args.consequence_guard_conflict_bonus * conflict
            - args.consequence_person_kl_weight * person_kl_after
            - damage_penalty
            - capacity_cost
        )
    if action == "drop":
        mistaken_drop_penalty = args.consequence_drop_risk_weight * useful_score * (1.0 - exact_after)
        return drop_value - mistaken_drop_penalty - damage_penalty
    raise ValueError(f"Unknown consequence action {action!r}.")


def consequence_role_prediction(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    raw_groups: dict[str, list[QAExample]],
    encoded_people: dict[str, list[EncodedExample]],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
    pad_id: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    reference_logits_by_person = {
        person: collect_example_logits(model, examples, pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
        for person, examples in sorted(encoded_people.items())
    }
    max_use_count = max(
        1,
        max(sum(1 for example in raw_groups["stage2"] + raw_groups["stage3"] if person in example.prompt or person in example.answer) for person in encoded_people),
    )
    rng = random.Random(seed)
    role_by_person: dict[str, str] = {}
    evidence_by_person: dict[str, dict[str, Any]] = {}
    all_people = sorted(encoded_people)
    for person in all_people:
        useful_count, obsolete_count, raw_use_count = observed_evidence_counts(
            args=args,
            person=person,
            raw_groups=raw_groups,
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
            rng=rng,
        )
        useful_score = min(1.0, float(useful_count) / float(max_use_count))
        obsolete_score = min(1.0, float(obsolete_count) / float(args.obsolete_threshold_count))
        no_update_metrics = evaluate_examples(
            model,
            encoded_people[person],
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )["overall"]
        no_update_kl = mean_distillation_kl(
            model=model,
            examples=encoded_people[person],
            teacher_logits=reference_logits_by_person[person],
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            temperature=args.distill_temperature,
            device=device,
        )
        other_people = [candidate for candidate in all_people if candidate != person]
        no_update_global_kl = mean_people_kl(
            model=model,
            encoded_people=encoded_people,
            reference_logits_by_person=reference_logits_by_person,
            people=other_people,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            temperature=args.distill_temperature,
            device=device,
        )
        drop_preview = preview_drop_update(
            args=args,
            checkpoint=checkpoint,
            model=model,
            person_examples=encoded_people[person],
            pad_id=pad_id,
            device=device,
            seed=seed + 1000 + all_people.index(person),
        )
        drop_metrics = evaluate_examples(
            drop_preview,
            encoded_people[person],
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )["overall"]
        drop_person_kl = mean_distillation_kl(
            model=drop_preview,
            examples=encoded_people[person],
            teacher_logits=reference_logits_by_person[person],
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            temperature=args.distill_temperature,
            device=device,
        )
        drop_global_kl = mean_people_kl(
            model=drop_preview,
            encoded_people=encoded_people,
            reference_logits_by_person=reference_logits_by_person,
            people=other_people,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            temperature=args.distill_temperature,
            device=device,
        )
        action_rows = {
            "preserve": {
                "exact_after": float(no_update_metrics["exact_match"]),
                "token_accuracy_after": float(no_update_metrics["token_accuracy"]),
                "loss_after": float(no_update_metrics["loss"]),
                "person_kl_after": no_update_kl,
                "global_damage_kl": no_update_global_kl,
            },
            "guard": {
                "exact_after": float(no_update_metrics["exact_match"]),
                "token_accuracy_after": float(no_update_metrics["token_accuracy"]),
                "loss_after": float(no_update_metrics["loss"]),
                "person_kl_after": no_update_kl,
                "global_damage_kl": no_update_global_kl,
            },
            "drop": {
                "exact_after": float(drop_metrics["exact_match"]),
                "token_accuracy_after": float(drop_metrics["token_accuracy"]),
                "loss_after": float(drop_metrics["loss"]),
                "person_kl_after": drop_person_kl,
                "global_damage_kl": drop_global_kl,
            },
        }
        for action, row in action_rows.items():
            row["score"] = role_action_score(
                action=action,
                useful_score=useful_score,
                obsolete_score=obsolete_score,
                exact_after=float(row["exact_after"]),
                person_kl_after=float(row["person_kl_after"]),
                global_damage_kl=float(row["global_damage_kl"]),
                capacity_pressure=args.capacity_pressure,
                args=args,
            )
        best_action = max(action_rows, key=lambda name: float(action_rows[name]["score"]))
        role_by_person[person] = best_action
        evidence_by_person[person] = {
            "raw_use_count": float(raw_use_count),
            "observed_useful_count": float(useful_count),
            "observed_obsolete_count": float(obsolete_count),
            "useful_score": useful_score,
            "obsolete_score": obsolete_score,
            "action_scores": action_rows,
            "selected_role": best_action,
        }
    return role_by_person, evidence_by_person


def hard_stage_loss(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    train_mask: torch.Tensor,
    preserve_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
    drop_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(train_inputs)
    ce = masked_ce_loss(logits, train_targets, train_mask)
    batch_size = int(train_inputs.shape[0])

    if preserve_examples:
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
    else:
        preserve_loss = ce.new_zeros(())

    if args.drop_loss_weight > 0.0 and drop_examples:
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

    loss = ce + args.lambda_preserve * preserve_loss + args.drop_loss_weight * drop_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "preserve": float(preserve_loss.detach().cpu()),
        "drop": float(drop_loss.detach().cpu()),
    }


def train_hard_stage(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    stage_examples: list[EncodedExample],
    preserve_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
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
        totals = {"loss": 0.0, "ce": 0.0, "preserve": 0.0, "drop": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(stage_examples), args.batch_size), desc=f"{label} {epoch}/{epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, row = hard_stage_loss(
                args=args,
                model=model,
                train_inputs=inputs,
                train_targets=targets,
                train_mask=mask,
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
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
            pbar.set_postfix({"ce": f"{row['ce']:.3g}", "p": f"{row['preserve']:.3g}", "d": f"{row['drop']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"{label} epoch {epoch} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        trace.append(epoch_row)
        print(
            f"{label} epoch={epoch:4d} loss={epoch_row['loss']:.5f} ce={epoch_row['ce']:.5f} "
            f"preserve={epoch_row['preserve']:.5f} drop={epoch_row['drop']:.5f}"
        )
    return trace


def restore_guard_if_needed(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    guard_examples: list[EncodedExample],
    guard_logits: list[torch.Tensor],
    candidate_examples: list[EncodedExample],
    candidate_logits: list[torch.Tensor],
    pad_id: int,
    device: torch.device,
    seed: int,
    label: str,
) -> dict[str, Any]:
    if not guard_examples:
        return {
            "before_kl": None,
            "after_kl": None,
            "candidate_before_kl": None,
            "candidate_after_kl": None,
            "threshold": args.guard_restore_threshold,
            "restored": False,
            "restore_epochs": 0,
            "skipped_reason": "empty_guard_set",
            "candidate_aware_restore": args.candidate_aware_restore,
            "lambda_candidate_restore": args.lambda_candidate_restore,
            "trace": [],
        }
    before_kl = mean_distillation_kl(
        model=model,
        examples=guard_examples,
        teacher_logits=guard_logits,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        temperature=args.distill_temperature,
        device=device,
    )
    candidate_before_kl = (
        mean_distillation_kl(
            model=model,
            examples=candidate_examples,
            teacher_logits=candidate_logits,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            temperature=args.distill_temperature,
            device=device,
        )
        if args.candidate_aware_restore and candidate_examples
        else None
    )
    should_restore = before_kl > args.guard_restore_threshold and args.guard_restore_epochs > 0
    trace: list[dict[str, float]] = []
    if should_restore:
        set_only_native_weights_trainable(model)
        optimizer = make_optimizer_for_model(args, trainable_weight_parameters(model))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for epoch in range(1, args.guard_restore_epochs + 1):
            model.train()
            permutation = torch.randperm(len(guard_examples), generator=generator)
            guard_total = 0.0
            candidate_total = 0.0
            batches = 0
            pbar = tqdm(range(0, len(guard_examples), args.batch_size), desc=f"{label} guard-restore {epoch}/{args.guard_restore_epochs}")
            for start in pbar:
                indices = permutation[start : start + args.batch_size]
                inputs, _targets, _mask, selected = batch_examples(guard_examples, indices=indices, pad_id=pad_id, device=device)
                optimizer.zero_grad(set_to_none=True)
                current = model(inputs)
                guard_restore_loss = distillation_loss_for_examples(
                    current,
                    selected,
                    guard_logits,
                    indices,
                    temperature=args.distill_temperature,
                    device=device,
                )
                if args.candidate_aware_restore and candidate_examples:
                    candidate_restore_loss = sampled_distillation_loss(
                        model=model,
                        examples=candidate_examples,
                        teacher_logits=candidate_logits,
                        pad_id=pad_id,
                        batch_size=min(args.batch_size, len(candidate_examples)),
                        temperature=args.distill_temperature,
                        device=device,
                        generator=generator,
                    )
                else:
                    candidate_restore_loss = guard_restore_loss.new_zeros(())
                loss = (
                    args.lambda_guard_restore * guard_restore_loss
                    + args.lambda_candidate_restore * candidate_restore_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(model), args.grad_clip)
                optimizer.step()
                value = float(guard_restore_loss.detach().cpu())
                candidate_value = float(candidate_restore_loss.detach().cpu())
                guard_total += value
                candidate_total += candidate_value
                batches += 1
                pbar.set_postfix({"guard": f"{value:.3g}", "cand": f"{candidate_value:.3g}"})
            if batches <= 0:
                raise RuntimeError(f"{label} guard restore epoch {epoch} saw zero batches.")
            trace.append(
                {
                    "epoch": float(epoch),
                    "guard_restore_kl": guard_total / float(batches),
                    "candidate_restore_kl": candidate_total / float(batches),
                }
            )
    after_kl = mean_distillation_kl(
        model=model,
        examples=guard_examples,
        teacher_logits=guard_logits,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        temperature=args.distill_temperature,
        device=device,
    )
    candidate_after_kl = (
        mean_distillation_kl(
            model=model,
            examples=candidate_examples,
            teacher_logits=candidate_logits,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            temperature=args.distill_temperature,
            device=device,
        )
        if args.candidate_aware_restore and candidate_examples
        else None
    )
    return {
        "before_kl": before_kl,
        "after_kl": after_kl,
        "candidate_before_kl": candidate_before_kl,
        "candidate_after_kl": candidate_after_kl,
        "threshold": args.guard_restore_threshold,
        "restored": should_restore,
        "restore_epochs": args.guard_restore_epochs if should_restore else 0,
        "candidate_aware_restore": args.candidate_aware_restore,
        "lambda_candidate_restore": args.lambda_candidate_restore,
        "trace": trace,
    }


def evaluate_people(
    *,
    model: torch.nn.Module,
    encoded_people: dict[str, list[EncodedExample]],
    reference_logits: dict[str, list[torch.Tensor]],
    true_roles: dict[str, str],
    predicted_roles: dict[str, str],
    pad_id: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for person, examples in sorted(encoded_people.items()):
        metrics = evaluate_examples(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)["overall"]
        drift = mean_distillation_kl(
            model=model,
            examples=examples,
            teacher_logits=reference_logits[person],
            pad_id=pad_id,
            batch_size=batch_size,
            temperature=temperature,
            device=device,
        )
        rows[person] = {
            "true_role": true_roles[person],
            "predicted_role": predicted_roles[person],
            "loss": float(metrics["loss"]),
            "token_accuracy": float(metrics["token_accuracy"]),
            "exact_match": float(metrics["exact_match"]),
            "reference_kl": drift,
        }
    return rows


def choose_roles(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    method: str,
    controller: NeuralRoleController,
    model: torch.nn.Module,
    raw_groups: dict[str, list[QAExample]],
    encoded_people: dict[str, list[EncodedExample]],
    true_roles: dict[str, str],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
    pad_id: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    if method == "oracle":
        return dict(true_roles), {"source": "oracle"}
    if method == "random":
        return random_roles_matching_counts(true_roles=true_roles, seed=seed), {"source": "random_count_matched"}
    if method == "neural":
        roles, evidence = neural_role_prediction(
            args=args,
            controller=controller,
            model=model,
            raw_groups=raw_groups,
            encoded_people=encoded_people,
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
            pad_id=pad_id,
            device=device,
            seed=seed,
        )
        return roles, {"source": "neural_controller", "per_person": evidence}
    if method == "consequence":
        roles, evidence = consequence_role_prediction(
            args=args,
            checkpoint=checkpoint,
            model=model,
            raw_groups=raw_groups,
            encoded_people=encoded_people,
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
            pad_id=pad_id,
            device=device,
            seed=seed,
        )
        return roles, {"source": "consequence_preview", "per_person": evidence}
    raise ValueError(f"Role selection is not defined for method {method!r}.")


def run_method(
    *,
    args: argparse.Namespace,
    method: str,
    controller: NeuralRoleController,
    checkpoint: dict[str, Any],
    raw_groups: dict[str, list[QAExample]],
    encoded_base_groups: dict[str, list[EncodedExample]],
    encoded_people: dict[str, list[EncodedExample]],
    true_roles: dict[str, str],
    true_role_groups: dict[str, list[EncodedExample]],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_offsets = {"naive": 101, "oracle": 202, "neural": 303, "random": 404, "consequence": 505}
    if method not in seed_offsets:
        raise ValueError(f"Unknown method {method!r}.")
    model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed + seed_offsets[method])
    traces: list[dict[str, Any]] = []
    stage1_trace = train_bootstrap_stage(
        args=args,
        model=model,
        stage_examples=encoded_base_groups["stage1"],
        pad_id=pad_id,
        device=device,
        epochs=args.stage1_epochs,
        seed=args.seed + 1100 + seed_offsets[method],
        label=f"{method} stage1",
    )
    traces.append({"stage": 1, "mode": "bootstrap", "trace": stage1_trace})
    stage1_reference = clone_current_model(model, checkpoint=checkpoint, device=device)

    reference_logits_by_person = {
        person: collect_example_logits(model, examples, pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
        for person, examples in sorted(encoded_people.items())
    }

    if method == "naive":
        role_by_person = dict(true_roles)
        role_evidence: dict[str, Any] = {"source": "not_used_by_naive"}
        training_role_groups = build_role_groups_allow_empty(role_by_person=role_by_person, encoded_people=encoded_people)
        train_groups = {**encoded_base_groups, "preserve": training_role_groups["preserve"], "drop": training_role_groups["drop"], "neutral": training_role_groups["neutral"]}
        for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
            trace = train_bootstrap_stage(
                args=args,
                model=model,
                stage_examples=train_groups[stage_name],
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 1200 + seed_offsets[method] + stage_number,
                label=f"{method} {stage_name}",
            )
            traces.append({"stage": stage_number, "mode": "naive", "trace": trace})
    else:
        role_by_person, role_evidence = choose_roles(
            args=args,
            checkpoint=checkpoint,
            method=method,
            controller=controller,
            model=model,
            raw_groups=raw_groups,
            encoded_people=encoded_people,
            true_roles=true_roles,
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
            pad_id=pad_id,
            device=device,
            seed=args.seed + 1300 + seed_offsets[method],
        )
        training_role_groups = build_role_groups_allow_empty(role_by_person=role_by_person, encoded_people=encoded_people)
        train_groups = {**encoded_base_groups, "preserve": training_role_groups["preserve"], "drop": training_role_groups["drop"], "neutral": training_role_groups["neutral"]}
        preserve_examples = cap_examples(train_groups["preserve"], budget=args.preserve_budget)
        drop_examples = cap_examples(train_groups["drop"], budget=args.drop_budget)
        guard_examples = cap_examples(train_groups["neutral"], budget=args.guard_budget)
        for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
            preserve_logits = (
                collect_example_logits(model, preserve_examples, pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
                if preserve_examples
                else []
            )
            guard_logits = (
                collect_example_logits(model, guard_examples, pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
                if guard_examples
                else []
            )
            stage_metrics_before_training = evaluate_examples(
                model,
                train_groups[stage_name],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )["overall"]
            trace = train_hard_stage(
                args=args,
                model=model,
                stage_examples=train_groups[stage_name],
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
                drop_examples=drop_examples,
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 1400 + seed_offsets[method] + stage_number,
                label=f"{method} {stage_name}",
            )
            candidate_examples = cap_examples(train_groups[stage_name], budget=args.candidate_restore_budget)
            candidate_logits = collect_example_logits(
                model,
                candidate_examples,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            stage_metrics_before_restore = evaluate_examples(
                model,
                train_groups[stage_name],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )["overall"]
            guard_report = restore_guard_if_needed(
                args=args,
                model=model,
                guard_examples=guard_examples,
                guard_logits=guard_logits,
                candidate_examples=candidate_examples,
                candidate_logits=candidate_logits,
                pad_id=pad_id,
                device=device,
                seed=args.seed + 1500 + seed_offsets[method] + stage_number,
                label=f"{method} stage{stage_number}",
            )
            stage_metrics_after_restore = evaluate_examples(
                model,
                train_groups[stage_name],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )["overall"]
            traces.append(
                {
                    "stage": stage_number,
                    "mode": "hard_role_update",
                    "trace": trace,
                    "guard_report": guard_report,
                    "stage_metrics_before_training": stage_metrics_before_training,
                    "stage_metrics_before_restore": stage_metrics_before_restore,
                    "stage_metrics_after_restore": stage_metrics_after_restore,
                }
            )
            if args.commit_learned_stage_to_guard:
                new_logits = collect_example_logits(model, train_groups[stage_name], pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
                guard_examples, guard_logits = cap_example_logit_pairs(
                    guard_examples + train_groups[stage_name],
                    guard_logits + new_logits,
                    budget=args.guard_budget,
                )

    eval_groups = {
        **encoded_base_groups,
        "preserve": true_role_groups["preserve"],
        "drop": true_role_groups["drop"],
        "neutral": true_role_groups["neutral"],
    }
    metrics = evaluate_role_summary(model=model, groups=eval_groups, pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
    category_breakdown = evaluate_category_breakdown(model=model, examples=eval_groups["eval_all"], pad_id=pad_id, batch_size=args.eval_batch_size, device=device)
    role_report = role_match_report(predicted_roles=role_by_person, true_roles=true_roles)
    people_report = evaluate_people(
        model=model,
        encoded_people=encoded_people,
        reference_logits=reference_logits_by_person,
        true_roles=true_roles,
        predicted_roles=role_by_person,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        temperature=args.distill_temperature,
        device=device,
    )
    if args.diagnostic_geometry:
        geometry_examples = (
            eval_groups["preserve"]
            + eval_groups["drop"]
            + eval_groups["neutral"]
            + eval_groups["stage2"]
            + eval_groups["stage3"]
        )
        residual_geometry = geometry_report(
            reference=stage1_reference,
            candidate=model,
            examples=geometry_examples,
            device=device,
        )
        grouped_geometry = grouped_geometry_report(
            reference=stage1_reference,
            candidate=model,
            groups={
                "preserve": eval_groups["preserve"],
                "drop": eval_groups["drop"],
                "guard": eval_groups["neutral"],
                "stage2": eval_groups["stage2"],
                "stage3": eval_groups["stage3"],
            },
            device=device,
        )
    else:
        residual_geometry = None
        grouped_geometry = None
    return {
        "method": method,
        "role_by_person": role_by_person,
        "true_role_by_person": true_roles,
        "role_match_report": role_report,
        "role_evidence": role_evidence,
        "metrics": metrics,
        "category_breakdown": category_breakdown,
        "people_report": people_report,
        "residual_geometry_vs_stage1": residual_geometry,
        "residual_geometry_mean_vs_stage1": summarize_geometry_means(residual_geometry) if residual_geometry is not None else None,
        "grouped_geometry_vs_stage1": grouped_geometry,
        "traces": traces,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer does not exist: {args.tokenizer_path}")
    parse_methods(args.methods)
    for name in [
        "stage1_epochs",
        "stage_epochs",
        "batch_size",
        "eval_batch_size",
        "preserve_budget",
        "guard_budget",
        "drop_budget",
        "guard_restore_epochs",
        "candidate_restore_budget",
        "consequence_preview_epochs",
    ]:
        positive_int(name, getattr(args, name))
    for name in ["obsolete_evidence_count", "false_useful_evidence_count", "false_obsolete_evidence_count"]:
        nonnegative_int(name, getattr(args, name))
    for name in [
        "lr",
        "distill_temperature",
        "grad_clip",
        "drop_target_probability",
        "guard_restore_threshold",
        "lambda_guard_restore",
        "lambda_candidate_restore",
    ]:
        positive_float(name, getattr(args, name))
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")
    for name in [
        "weight_decay",
        "momentum",
        "lambda_preserve",
        "drop_loss_weight",
        "consequence_global_damage_weight",
        "consequence_drop_risk_weight",
        "consequence_guard_conflict_bonus",
        "consequence_person_kl_weight",
        "consequence_preserve_capacity_cost",
        "consequence_guard_capacity_cost",
    ]:
        nonnegative_float(name, getattr(args, name))
    for name in [
        "capacity_pressure",
        "useful_evidence_keep_probability",
        "obsolete_evidence_keep_probability",
        "false_useful_evidence_probability",
        "false_obsolete_evidence_probability",
    ]:
        probability(name, getattr(args, name))


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer must define [PAD].")

    oracle_preserve_people = parse_people(args.oracle_preserve_people, name="oracle_preserve_people")
    oracle_drop_people = parse_people(args.oracle_drop_people, name="oracle_drop_people")
    useful_evidence_people = parse_people(args.useful_evidence_people, name="useful_evidence_people")
    obsolete_evidence_people = parse_people(args.obsolete_evidence_people, name="obsolete_evidence_people")
    composition_holdout_people = parse_people(args.composition_holdout_people, name="composition_holdout_people", allow_empty=True)
    overlap = useful_evidence_people.intersection(obsolete_evidence_people)
    if overlap:
        raise ValueError(f"People cannot have both useful and obsolete evidence: {sorted(overlap)}.")
    true_roles = oracle_roles(preserve_people=oracle_preserve_people, drop_people=oracle_drop_people)
    _config_model, checkpoint = load_checkpoint(args.config_checkpoint, device)
    max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    raw_groups = build_raw_stream(
        useful_evidence_people=useful_evidence_people,
        composition_holdout_people=composition_holdout_people,
        include_composition_rules=args.include_composition_rules,
    )
    encoded_base_groups = encode_raw_groups(raw_groups, tokenizer, max_seq_len=max_seq_len)
    encoded_people = encode_raw_groups(examples_by_stage1_person(), tokenizer, max_seq_len=max_seq_len)
    true_role_groups = build_role_groups_allow_empty(role_by_person=true_roles, encoded_people=encoded_people)
    methods = parse_methods(args.methods)

    print("TINY NEURAL HARD ROLE-SEPARATION CL")
    print("=" * 112)
    print(f"device={device} methods={methods} true_roles={true_roles}")
    print(
        "role mechanics: preserve=continuous distill, guard=monitor/restore, drop=suppress; "
        f"guard_threshold={args.guard_restore_threshold:.4g} "
        f"candidate_restore={args.candidate_aware_restore}"
    )
    print(
        f"examples stage1={len(encoded_base_groups['stage1'])} stage2={len(encoded_base_groups['stage2'])} "
        f"stage3={len(encoded_base_groups['stage3'])} eval={len(encoded_base_groups['eval_all'])}"
    )

    controller, controller_report = train_neural_controller(args=args, device=device)
    print(
        "role-controller final train_acc={:.4f} eval_acc={:.4f}".format(
            controller_report["train_accuracy"],
            controller_report["eval_accuracy"],
        )
    )
    results = [
        run_method(
            args=args,
            method=method,
            controller=controller,
            checkpoint=checkpoint,
            raw_groups=raw_groups,
            encoded_base_groups=encoded_base_groups,
            encoded_people=encoded_people,
            true_roles=true_roles,
            true_role_groups=true_role_groups,
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
            pad_id=pad_id,
            device=device,
        )
        for method in methods
    ]

    summary = {
        "question": "Can a neural role controller drive mechanically distinct preserve/guard/drop CL decisions?",
        "controller": {
            "type": "tiny_mlp",
            "feature_names": FEATURE_NAMES,
            "report": controller_report,
            "hidden_dim": args.controller_hidden_dim,
        },
        "role_mechanics": {
            "preserve": "continuous behavior distillation during new learning",
            "guard": "monitor-only, then restore if KL drift exceeds threshold",
            "drop": "active old-answer suppression",
            "guard_restore_threshold": args.guard_restore_threshold,
            "guard_restore_epochs": args.guard_restore_epochs,
            "candidate_aware_restore": args.candidate_aware_restore,
            "candidate_restore_budget": args.candidate_restore_budget,
            "lambda_candidate_restore": args.lambda_candidate_restore,
            "commit_learned_stage_to_guard": args.commit_learned_stage_to_guard,
        },
        "evidence_config": {
            "true_role_by_person": true_roles,
            "useful_evidence_people": sorted(useful_evidence_people),
            "obsolete_evidence_people": sorted(obsolete_evidence_people),
            "composition_holdout_people": sorted(composition_holdout_people),
            "capacity_pressure": args.capacity_pressure,
            "useful_evidence_keep_probability": args.useful_evidence_keep_probability,
            "obsolete_evidence_keep_probability": args.obsolete_evidence_keep_probability,
        },
        "model_config": checkpoint["model_config"],
        "hyperparameters": {
            "stage1_epochs": args.stage1_epochs,
            "stage_epochs": args.stage_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "preserve_budget": args.preserve_budget,
            "guard_budget": args.guard_budget,
            "drop_budget": args.drop_budget,
            "candidate_restore_budget": args.candidate_restore_budget,
            "lambda_preserve": args.lambda_preserve,
            "lambda_guard_restore": args.lambda_guard_restore,
            "lambda_candidate_restore": args.lambda_candidate_restore,
            "drop_loss_weight": args.drop_loss_weight,
            "drop_target_probability": args.drop_target_probability,
            "consequence_preview_epochs": args.consequence_preview_epochs,
            "consequence_global_damage_weight": args.consequence_global_damage_weight,
            "consequence_drop_risk_weight": args.consequence_drop_risk_weight,
            "consequence_guard_conflict_bonus": args.consequence_guard_conflict_bonus,
            "consequence_person_kl_weight": args.consequence_person_kl_weight,
        },
        "raw_groups": {name: [asdict(example) for example in examples] for name, examples in raw_groups.items()},
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTINY NEURAL HARD ROLE-SEPARATION SUMMARY")
    print("=" * 112)
    for result in results:
        print(
            f"method={result['method']} role_accuracy={result['role_match_report']['accuracy']:.3f} "
            f"roles={result['role_by_person']}"
        )
    print("-" * 112)
    print(
        "{:>10} {:>8} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "method",
            "roleAcc",
            "preserve",
            "drop",
            "guard",
            "stage2",
            "stage3",
            "eval_all",
        )
    )
    print(
        "{:>10} {:>8} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "",
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
            "{:>10} {:>8.3f} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
                result["method"],
                result["role_match_report"]["accuracy"],
                compact(metrics["preserve"]),
                compact(metrics["drop"]),
                compact(metrics["neutral"]),
                compact(metrics["stage2"]),
                compact(metrics["stage3"]),
                compact(metrics["eval_all"]),
            )
        )
    print("\nPER-PERSON FINAL ROLE OUTCOME")
    print("-" * 112)
    print("{:>10} {:>10} {:>10} {:>12} {:>12} {:>12}".format("method", "person", "roles", "exact", "tok_acc", "ref_kl"))
    for result in results:
        for person, row in sorted(result["people_report"].items()):
            roles = f"{row['true_role']}->{row['predicted_role']}"
            print(
                "{:>10} {:>10} {:>10} {:12.3f} {:12.3f} {:12.4g}".format(
                    result["method"],
                    person,
                    roles,
                    float(row["exact_match"]),
                    float(row["token_accuracy"]),
                    float(row["reference_kl"]),
                )
            )
    print("\nSTAGE RESTORE EFFECT")
    print("-" * 112)
    print(
        "{:>10} {:>6} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "method",
            "stage",
            "pre_train",
            "pre_restore",
            "post_restore",
            "guard_kl",
            "candidate_kl",
        )
    )
    print(
        "{:>10} {:>6} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "",
            "",
            "loss/exact",
            "loss/exact",
            "loss/exact",
            "before->after",
            "before->after",
        )
    )
    for result in results:
        for trace_row in result["traces"]:
            if "stage_metrics_before_training" not in trace_row:
                continue
            guard_report = trace_row["guard_report"]
            guard_kl = (
                "n/a"
                if guard_report["before_kl"] is None
                else "{:.3g}->{:.3g}".format(guard_report["before_kl"], guard_report["after_kl"])
            )
            candidate_kl = (
                "n/a"
                if guard_report["candidate_before_kl"] is None
                else "{:.3g}->{:.3g}".format(
                    guard_report["candidate_before_kl"],
                    guard_report["candidate_after_kl"],
                )
            )
            print(
                "{:>10} {:>6} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
                    result["method"],
                    int(trace_row["stage"]),
                    compact(trace_row["stage_metrics_before_training"]),
                    compact(trace_row["stage_metrics_before_restore"]),
                    compact(trace_row["stage_metrics_after_restore"]),
                    guard_kl,
                    candidate_kl,
                )
            )
    if args.diagnostic_geometry:
        print("\nRESIDUAL GEOMETRY VS POST-STAGE1")
        print("-" * 112)
        print("{:>10} {:>12} {:>12} {:>12}".format("method", "drift_rel", "cka", "rank_delta"))
        for result in results:
            summary = result["residual_geometry_mean_vs_stage1"]
            if summary is None:
                raise RuntimeError(f"Missing residual geometry summary for method {result['method']}.")
            print(
                "{:>10} {:12.4f} {:12.4f} {:12.4f}".format(
                    result["method"],
                    summary["drift_relative"],
                    summary["cka"],
                    summary["rank_delta"],
                )
            )
    print(f"wrote_json={args.output_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-neural-role-hard-test-seed0.json"))
    parser.add_argument("--methods", type=str, default="naive,oracle,neural,consequence,random")
    parser.add_argument("--oracle-preserve-people", type=str, default="Alice,Bruno")
    parser.add_argument("--oracle-drop-people", type=str, default="Clara,Darin")
    parser.add_argument("--useful-evidence-people", type=str, default="Alice,Bruno")
    parser.add_argument("--obsolete-evidence-people", type=str, default="Clara,Darin")
    parser.add_argument("--composition-holdout-people", type=str, default="Kira,Luca")
    parser.add_argument("--include-composition-rules", action="store_true")
    parser.add_argument("--commit-learned-stage-to-guard", action="store_true")
    parser.add_argument("--diagnostic-geometry", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=300)
    parser.add_argument("--stage-epochs", type=int, default=300)
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
    parser.add_argument("--lambda-guard-restore", type=float, default=1.0)
    parser.add_argument("--lambda-candidate-restore", type=float, default=1.0)
    parser.add_argument("--guard-restore-threshold", type=float, default=0.01)
    parser.add_argument("--guard-restore-epochs", type=int, default=40)
    parser.add_argument("--candidate-aware-restore", action="store_true")
    parser.add_argument("--candidate-restore-budget", type=int, default=32)
    parser.add_argument("--drop-loss-weight", type=float, default=0.1)
    parser.add_argument("--drop-target-probability", type=float, default=0.05)
    parser.add_argument("--consequence-preview-epochs", type=int, default=40)
    parser.add_argument("--consequence-global-damage-weight", type=float, default=0.1)
    parser.add_argument("--consequence-drop-risk-weight", type=float, default=2.0)
    parser.add_argument("--consequence-guard-conflict-bonus", type=float, default=1.0)
    parser.add_argument("--consequence-person-kl-weight", type=float, default=0.05)
    parser.add_argument("--consequence-preserve-capacity-cost", type=float, default=0.05)
    parser.add_argument("--consequence-guard-capacity-cost", type=float, default=0.01)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--obsolete-evidence-count", type=int, default=4)
    parser.add_argument("--obsolete-threshold-count", type=int, default=3)
    parser.add_argument("--capacity-pressure", type=float, default=1.0)
    parser.add_argument("--useful-evidence-keep-probability", type=float, default=0.5)
    parser.add_argument("--obsolete-evidence-keep-probability", type=float, default=0.5)
    parser.add_argument("--false-useful-evidence-count", type=int, default=1)
    parser.add_argument("--false-useful-evidence-probability", type=float, default=0.15)
    parser.add_argument("--false-obsolete-evidence-count", type=int, default=1)
    parser.add_argument("--false-obsolete-evidence-probability", type=float, default=0.15)
    parser.add_argument("--controller-hidden-dim", type=int, default=32)
    parser.add_argument("--controller-train-examples", type=int, default=4096)
    parser.add_argument("--controller-eval-examples", type=int, default=1024)
    parser.add_argument("--controller-epochs", type=int, default=40)
    parser.add_argument("--controller-batch-size", type=int, default=128)
    parser.add_argument("--controller-lr", type=float, default=1e-3)
    parser.add_argument("--controller-weight-decay", type=float, default=1e-4)
    parser.add_argument("--controller-grad-clip", type=float, default=1.0)
    parser.add_argument("--controller-print-every", type=int, default=10)
    parser.add_argument("--controller-max-use-count", type=int, default=8)
    parser.add_argument("--controller-loss-clip", type=float, default=8.0)
    parser.add_argument("--controller-learned-high-probability", type=float, default=0.9)
    parser.add_argument("--controller-guard-learned-probability", type=float, default=0.75)
    parser.add_argument("--controller-learned-exact-min", type=float, default=0.8)
    parser.add_argument("--controller-learned-token-min", type=float, default=0.85)
    parser.add_argument("--controller-learned-loss-max", type=float, default=0.2)
    parser.add_argument("--controller-guard-exact-max", type=float, default=0.65)
    parser.add_argument("--controller-guard-token-max", type=float, default=0.75)
    parser.add_argument("--controller-guard-loss-min", type=float, default=0.6)
    parser.add_argument("--controller-useful-recall", type=float, default=0.8)
    parser.add_argument("--controller-useful-false-positive", type=float, default=0.1)
    parser.add_argument("--controller-obsolete-recall", type=float, default=0.85)
    parser.add_argument("--controller-obsolete-false-positive", type=float, default=0.1)
    parser.add_argument("--controller-guard-useful-signal-rate", type=float, default=0.05)
    parser.add_argument("--controller-guard-obsolete-signal-rate", type=float, default=0.05)
    parser.add_argument("--controller-capacity-min", type=float, default=0.3)
    parser.add_argument("--controller-capacity-max", type=float, default=1.0)
    parser.add_argument("--controller-drop-capacity-min", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
