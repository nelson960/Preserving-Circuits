"""Learn learn/guard/drop decisions from measured update consequences.

This experiment attacks the missing selector problem separately from the
Invariant-Tangent update operator.  For each trace and candidate action it runs
a cheap virtual update, measures the downstream consequence, and trains a small
neural scorer to predict that consequence from model/trace evidence.

The scorer is not trained from preserve/drop/guard labels.  The target is the
measured consequence of an action:

    lower protected loss
  + lower guard loss
  + lower new-stage loss
  + lower obsolete-answer probability

The result tells us whether the evidence currently exposed to the controller is
enough for a learned network to choose actions by consequence rather than by a
hand-written role rule.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
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
from experiments.gco_math.gco_tiny_auto_role_controller import (
    build_raw_stream,
    encode_raw_groups,
    oracle_roles,
    role_match_report,
    stage1_people,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import load_checkpoint
from experiments.gco_math.gco_tiny_end_to_end_cl_controller import (
    ActionLogitAdapterModel,
    freeze_module,
    person_for_example,
    teacher_logits_by_person,
)
from experiments.gco_math.gco_tiny_recurrent_trace_plasticity import (
    TRACE_FEATURE_NAMES,
    make_adapter,
    make_encoded_trace_people,
    make_initial_trace_state,
    stage_evidence,
)
from experiments.gco_math.gco_tiny_recursive_controlled_architecture import (
    drop_suppression_loss,
    train_bootstrap_stage,
)
from experiments.gco_math.gco_tiny_reusable_computation_cl import (
    EncodedExample,
    QAExample,
    batch_examples,
    distillation_loss_for_examples,
    evaluate_examples,
    make_model_from_config,
    masked_ce_loss,
)


ACTION_NAMES = ("learn", "preserve", "guard", "drop")


@dataclass(frozen=True)
class DecisionSample:
    episode: int
    stage: str
    person: str
    action: str
    score: float
    components: dict[str, float]
    features: list[float]


class ConsequenceActionScorer(nn.Module):
    def __init__(self, *, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim + len(ACTION_NAMES)),
            nn.Linear(feature_dim + len(ACTION_NAMES), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(TRACE_FEATURE_NAMES):
            raise ValueError(
                f"features must have shape [n, {len(TRACE_FEATURE_NAMES)}], got {tuple(features.shape)}."
            )
        if action_indices.ndim != 1 or action_indices.shape[0] != features.shape[0]:
            raise ValueError(
                f"action_indices must have shape [{features.shape[0]}], got {tuple(action_indices.shape)}."
            )
        action_one_hot = F.one_hot(action_indices, num_classes=len(ACTION_NAMES)).to(dtype=features.dtype)
        return self.net(torch.cat([features, action_one_hot], dim=1)).squeeze(1)


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"Config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    for name in [
        "stage1_epochs",
        "train_episodes",
        "test_episodes",
        "virtual_steps",
        "scorer_epochs",
        "batch_size",
        "eval_batch_size",
        "adapter_rank",
        "hidden_dim",
        "preserve_count",
        "drop_count",
        "guard_count",
        "print_every",
    ]:
        positive_int(name, getattr(args, name))
    for name in [
        "lr",
        "virtual_lr",
        "scorer_lr",
        "adapter_scale",
        "adapter_init_std",
        "drop_target_probability",
        "distill_temperature",
        "loss_clip",
        "trace_budget",
        "grad_clip",
        "lambda_preserve",
        "lambda_guard",
        "lambda_new",
        "lambda_drop",
    ]:
        positive_float(name, getattr(args, name))
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")
    nonnegative_float("weight_decay", args.weight_decay)
    parse_composition_holdout_people(args.composition_holdout_people)
    role_total = args.preserve_count + args.drop_count + args.guard_count
    stage1_total = len(stage1_people())
    if role_total != stage1_total:
        raise ValueError(
            "--preserve-count + --drop-count + --guard-count must equal the number of stage-1 people; "
            f"got {role_total}, expected {stage1_total}."
        )


def parse_composition_holdout_people(raw: str) -> set[str]:
    people = {item.strip() for item in raw.split(",") if item.strip()}
    if not people:
        raise ValueError("--composition-holdout-people must contain at least one person.")
    return people


def action_gates(action: str) -> dict[str, float]:
    if action not in ACTION_NAMES:
        raise ValueError(f"Unknown action {action!r}; expected one of {ACTION_NAMES}.")
    return {name: 1.0 if name == action else 0.0 for name in ACTION_NAMES}


def people_for_stage_examples(examples: list[EncodedExample], people: list[str]) -> dict[str, list[EncodedExample]]:
    grouped = {person: [] for person in people}
    for example in examples:
        person = person_for_example(example, people)
        if person is not None:
            grouped[person].append(example)
    return grouped


def examples_for_people(
    *,
    people: set[str],
    encoded_people: dict[str, list[EncodedExample]],
) -> list[EncodedExample]:
    selected = [example for person in sorted(people) for example in encoded_people[person]]
    if not selected:
        raise RuntimeError(f"No examples found for people={sorted(people)}.")
    return selected


def evaluate_group(
    *,
    model: ActionLogitAdapterModel,
    examples: list[EncodedExample],
    action: str,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.set_action_gates(action_gates(action))
    metrics = evaluate_examples(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)["overall"]
    return {
        "loss": float(metrics["loss"]),
        "exact_match": float(metrics["exact_match"]),
        "token_accuracy": float(metrics["token_accuracy"]),
    }


def average_drop_suppression(
    *,
    model: ActionLogitAdapterModel,
    examples: list[EncodedExample],
    action: str,
    pad_id: int,
    batch_size: int,
    device: torch.device,
    target_probability: float,
) -> float:
    model.set_action_gates(action_gates(action))
    losses: list[float] = []
    indices = torch.arange(len(examples), dtype=torch.long)
    for start in range(0, len(examples), batch_size):
        batch_indices = indices[start : start + batch_size]
        inputs, targets, mask, _selected = batch_examples(
            examples,
            indices=batch_indices,
            pad_id=pad_id,
            device=device,
        )
        with torch.no_grad():
            logits = model(inputs)
            loss = drop_suppression_loss(
                logits=logits,
                targets=targets,
                mask=mask,
                target_probability=target_probability,
            )
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("average_drop_suppression saw no batches.")
    return sum(losses) / float(len(losses))


def distill_group_loss(
    *,
    model: ActionLogitAdapterModel,
    examples: list[EncodedExample],
    reference_logits: dict[str, list[torch.Tensor]],
    people: list[str],
    action: str,
    pad_id: int,
    batch_size: int,
    device: torch.device,
    temperature: float,
) -> float:
    model.set_action_gates(action_gates(action))
    losses: list[float] = []
    grouped = people_for_stage_examples(examples, people)
    for person, person_examples in sorted(grouped.items()):
        if not person_examples:
            continue
        all_person_examples = person_examples
        indices = torch.arange(len(all_person_examples), dtype=torch.long)
        for start in range(0, len(all_person_examples), batch_size):
            batch_indices = indices[start : start + batch_size]
            inputs, _targets, _mask, selected = batch_examples(
                all_person_examples,
                indices=batch_indices,
                pad_id=pad_id,
                device=device,
            )
            logits = model(inputs)
            loss = distillation_loss_for_examples(
                logits,
                selected,
                reference_logits[person],
                batch_indices,
                temperature=temperature,
                device=device,
            )
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("distill_group_loss saw no examples linked to known people.")
    return sum(losses) / float(len(losses))


def train_virtual_action(
    *,
    args: argparse.Namespace,
    adapter: ActionLogitAdapterModel,
    action: str,
    examples: list[EncodedExample],
    reference_logits: dict[str, list[torch.Tensor]],
    person: str,
    pad_id: int,
    device: torch.device,
    seed: int,
) -> ActionLogitAdapterModel:
    if not examples:
        raise RuntimeError(f"Cannot virtual-train action {action!r} with no examples.")
    candidate = copy.deepcopy(adapter).to(device)
    candidate.set_action_gates(action_gates("learn"))
    trainable = [parameter for parameter in candidate.adapters["learn"].parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Shared virtual learn adapter has no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable, lr=args.virtual_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for _step in range(args.virtual_steps):
        sample_count = min(args.batch_size, len(examples))
        indices = torch.randint(0, len(examples), size=(sample_count,), generator=generator)
        inputs, targets, mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = candidate(inputs)
        if action == "drop":
            loss = drop_suppression_loss(
                logits=logits,
                targets=targets,
                mask=mask,
                target_probability=args.drop_target_probability,
            )
        elif action == "learn":
            loss = masked_ce_loss(logits, targets, mask)
        elif action in {"preserve", "guard"}:
            loss = distillation_loss_for_examples(
                logits,
                selected,
                reference_logits[person],
                indices,
                temperature=args.distill_temperature,
                device=device,
            )
        else:
            raise ValueError(f"Unhandled action {action!r}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
    return candidate


def build_feature_rows(
    *,
    args: argparse.Namespace,
    adapter: ActionLogitAdapterModel,
    people: list[str],
    encoded_people: dict[str, list[EncodedExample]],
    trace_state: dict[str, torch.Tensor],
    evidence: dict[str, torch.Tensor],
    pad_id: int,
    device: torch.device,
) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    capacity_pressure = float((trace_state["strength"].sum() / args.trace_budget - 1.0).clamp(min=0.0, max=2.0).cpu())
    for person_index, person in enumerate(people):
        metrics = evaluate_group(
            model=adapter,
            examples=encoded_people[person],
            action="learn",
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        rows[person] = [
            min(metrics["loss"], args.loss_clip) / args.loss_clip,
            metrics["exact_match"],
            metrics["token_accuracy"],
            float(evidence["recurrence"][person_index]),
            float(evidence["conflict"][person_index]),
            float(evidence["stream_reference"][person_index]),
            capacity_pressure,
            float(trace_state["strength"][person_index].detach().cpu()),
            float(trace_state["protected"][person_index].detach().cpu()),
            float(trace_state["known"][person_index].detach().cpu()),
            float(trace_state["stability"][person_index].detach().cpu()),
            float((trace_state["age"][person_index].clamp(max=10.0) / 10.0).detach().cpu()),
            float(trace_state["usefulness"][person_index].detach().cpu()),
            float(trace_state["last_gain"][person_index].detach().cpu()),
        ]
    return rows


def choose_episode_roles(*, args: argparse.Namespace, seed: int) -> tuple[set[str], set[str], set[str]]:
    people = sorted(stage1_people())
    generator = random.Random(seed)
    generator.shuffle(people)
    preserve_end = args.preserve_count
    drop_end = preserve_end + args.drop_count
    preserve = set(people[:preserve_end])
    drop = set(people[preserve_end:drop_end])
    guard = set(people[drop_end : drop_end + args.guard_count])
    if len(preserve) != args.preserve_count or len(drop) != args.drop_count or len(guard) != args.guard_count:
        raise RuntimeError(
            "Episode role split did not produce the requested group sizes: "
            f"preserve={len(preserve)}/{args.preserve_count} "
            f"drop={len(drop)}/{args.drop_count} guard={len(guard)}/{args.guard_count}."
        )
    return preserve, drop, guard


def score_candidate(
    *,
    args: argparse.Namespace,
    candidate: ActionLogitAdapterModel,
    preserve_examples: list[EncodedExample],
    guard_examples: list[EncodedExample],
    drop_examples: list[EncodedExample],
    new_examples: list[EncodedExample],
    reference_logits: dict[str, list[torch.Tensor]],
    people: list[str],
    pad_id: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    preserve = distill_group_loss(
        model=candidate,
        examples=preserve_examples,
        reference_logits=reference_logits,
        people=people,
        action="learn",
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
        temperature=args.distill_temperature,
    )
    guard = distill_group_loss(
        model=candidate,
        examples=guard_examples,
        reference_logits=reference_logits,
        people=people,
        action="learn",
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
        temperature=args.distill_temperature,
    )
    new = evaluate_group(
        model=candidate,
        examples=new_examples,
        action="learn",
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )["loss"]
    drop = average_drop_suppression(
        model=candidate,
        examples=drop_examples,
        action="learn",
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
        target_probability=args.drop_target_probability,
    )
    components = {
        "preserve": preserve,
        "guard": guard,
        "new": new,
        "drop": drop,
    }
    score = (
        args.lambda_preserve * preserve
        + args.lambda_guard * guard
        + args.lambda_new * new
        + args.lambda_drop * drop
    )
    return float(score), components


def collect_episode_samples(
    *,
    args: argparse.Namespace,
    episode_index: int,
    base_model: nn.Module,
    checkpoint: dict[str, Any],
    tokenizer: Tokenizer,
    reference_logits: dict[str, list[torch.Tensor]],
    encoded_people: dict[str, list[EncodedExample]],
    pad_id: int,
    device: torch.device,
) -> list[DecisionSample]:
    preserve_people, drop_people, guard_people = choose_episode_roles(args=args, seed=args.seed + 10_000 + episode_index)
    true_roles = oracle_roles(preserve_people=preserve_people, drop_people=drop_people)
    raw_groups = build_raw_stream(
        useful_evidence_people=preserve_people,
        composition_holdout_people=parse_composition_holdout_people(args.composition_holdout_people),
        include_composition_rules=args.include_composition_rules,
    )
    max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    encoded_groups = encode_raw_groups(raw_groups, tokenizer, max_seq_len=max_seq_len)
    people = sorted(encoded_people)
    adapter = make_adapter(args=args, base_model=base_model, checkpoint=checkpoint, device=device)
    trace_state = make_initial_trace_state(
        people=people,
        old_people=set(true_roles),
        hidden_dim=args.hidden_dim,
        device=device,
    )
    preserve_examples = examples_for_people(people=preserve_people, encoded_people=encoded_people)
    drop_examples = examples_for_people(people=drop_people, encoded_people=encoded_people)
    guard_examples = examples_for_people(people=guard_people, encoded_people=encoded_people)
    samples: list[DecisionSample] = []
    for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
        evidence = stage_evidence(
            people=people,
            raw_examples=raw_groups[stage_name],
            useful_evidence_people=preserve_people,
            obsolete_evidence_people=drop_people,
        )
        feature_rows = build_feature_rows(
            args=args,
            adapter=adapter,
            people=people,
            encoded_people=encoded_people,
            trace_state=trace_state,
            evidence=evidence,
            pad_id=pad_id,
            device=device,
        )
        stage_people_examples = people_for_stage_examples(encoded_groups[stage_name], people)
        for person in people:
            person_stage_examples = stage_people_examples[person]
            for action_index, action in enumerate(ACTION_NAMES):
                if action == "learn":
                    update_examples = person_stage_examples if person_stage_examples else encoded_people[person]
                elif action in {"preserve", "guard", "drop"}:
                    update_examples = encoded_people[person]
                else:
                    raise ValueError(f"Unhandled action {action!r}.")
                candidate = train_virtual_action(
                    args=args,
                    adapter=adapter,
                    action=action,
                    examples=update_examples,
                    reference_logits=reference_logits,
                    person=person,
                    pad_id=pad_id,
                    device=device,
                    seed=args.seed + 100_000 + episode_index * 100 + stage_number * 10 + action_index,
                )
                score, components = score_candidate(
                    args=args,
                    candidate=candidate,
                    preserve_examples=preserve_examples,
                    guard_examples=guard_examples,
                    drop_examples=drop_examples,
                    new_examples=encoded_groups[stage_name],
                    reference_logits=reference_logits,
                    people=people,
                    pad_id=pad_id,
                    device=device,
                )
                samples.append(
                    DecisionSample(
                        episode=episode_index,
                        stage=stage_name,
                        person=person,
                        action=action,
                        score=score,
                        components=components,
                        features=feature_rows[person],
                    )
                )
        trace_state["age"] = trace_state["age"] + 1.0
    return samples


def tensors_from_samples(samples: list[DecisionSample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not samples:
        raise RuntimeError("Cannot build tensors from an empty sample list.")
    features = torch.tensor([sample.features for sample in samples], dtype=torch.float32, device=device)
    action_indices = torch.tensor([ACTION_NAMES.index(sample.action) for sample in samples], dtype=torch.long, device=device)
    scores = torch.tensor([math.log1p(sample.score) for sample in samples], dtype=torch.float32, device=device)
    return features, action_indices, scores


def train_scorer(
    *,
    args: argparse.Namespace,
    samples: list[DecisionSample],
    device: torch.device,
) -> tuple[ConsequenceActionScorer, list[dict[str, float]]]:
    scorer = ConsequenceActionScorer(feature_dim=len(TRACE_FEATURE_NAMES), hidden_dim=args.hidden_dim).to(device)
    features, action_indices, scores = tensors_from_samples(samples, device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.scorer_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 200_000)
    trace: list[dict[str, float]] = []
    for epoch in range(1, args.scorer_epochs + 1):
        permutation = torch.randperm(len(samples), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, len(samples), args.batch_size):
            batch_indices = permutation[start : start + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = scorer(features[batch_indices], action_indices[batch_indices])
            loss = F.mse_loss(predicted, scores[batch_indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        if batches <= 0:
            raise RuntimeError(f"Scorer epoch {epoch} saw zero batches.")
        row = {"epoch": float(epoch), "mse": total / float(batches)}
        trace.append(row)
        if epoch == 1 or epoch == args.scorer_epochs or epoch % args.print_every == 0:
            print(f"scorer epoch={epoch:4d} mse={row['mse']:.6f}")
    return scorer, trace


def evaluate_scorer(
    *,
    scorer: ConsequenceActionScorer,
    samples: list[DecisionSample],
    device: torch.device,
) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], list[DecisionSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.episode, sample.stage, sample.person), []).append(sample)
    correct = 0
    total = 0
    regret_sum = 0.0
    best_counts: Counter[str] = Counter()
    chosen_counts: Counter[str] = Counter()
    per_action_correct: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    scorer.eval()
    with torch.no_grad():
        for key, decision_samples in sorted(grouped.items()):
            if len(decision_samples) != len(ACTION_NAMES):
                raise RuntimeError(f"Decision {key} has {len(decision_samples)} samples, expected {len(ACTION_NAMES)}.")
            actual_by_action = {sample.action: sample.score for sample in decision_samples}
            best_action = min(actual_by_action, key=actual_by_action.get)
            features = torch.tensor(
                [decision_samples[0].features for _name in ACTION_NAMES],
                dtype=torch.float32,
                device=device,
            )
            action_indices = torch.arange(len(ACTION_NAMES), dtype=torch.long, device=device)
            predicted_scores = scorer(features, action_indices).detach().cpu().tolist()
            predicted_by_action = dict(zip(ACTION_NAMES, predicted_scores, strict=True))
            chosen_action = min(predicted_by_action, key=predicted_by_action.get)
            matched = chosen_action == best_action
            best_counts[best_action] += 1
            chosen_counts[chosen_action] += 1
            per_action_correct[best_action] += int(matched)
            correct += int(matched)
            total += 1
            regret = actual_by_action[chosen_action] - actual_by_action[best_action]
            regret_sum += regret
            rows.append(
                {
                    "episode": key[0],
                    "stage": key[1],
                    "person": key[2],
                    "best_action": best_action,
                    "chosen_action": chosen_action,
                    "match": matched,
                    "regret": regret,
                    "actual_scores": actual_by_action,
                    "predicted_scores_log1p": predicted_by_action,
                }
            )
    if total <= 0:
        raise RuntimeError("Scorer evaluation saw no grouped decisions.")
    supported_actions = [action for action in ACTION_NAMES if best_counts[action] > 0]
    if not supported_actions:
        raise RuntimeError("Scorer evaluation has no supported best actions.")
    macro_recall = sum(
        per_action_correct[action] / float(best_counts[action])
        for action in supported_actions
    ) / float(len(supported_actions))
    majority_accuracy = max(best_counts.values()) / float(total)
    return {
        "action_accuracy": correct / float(total),
        "majority_accuracy": majority_accuracy,
        "macro_recall": macro_recall,
        "mean_regret": regret_sum / float(total),
        "correct": correct,
        "total": total,
        "best_action_counts": {action: best_counts[action] for action in ACTION_NAMES},
        "chosen_action_counts": {action: chosen_counts[action] for action in ACTION_NAMES},
        "decisions": rows,
    }


def summarize_best_actions(samples: list[DecisionSample]) -> dict[str, int]:
    grouped: dict[tuple[int, str, str], list[DecisionSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.episode, sample.stage, sample.person), []).append(sample)
    counts = {action: 0 for action in ACTION_NAMES}
    for decision_samples in grouped.values():
        best = min(decision_samples, key=lambda sample: sample.score)
        counts[best.action] += 1
    return counts


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError(f"Tokenizer {args.tokenizer_path} has no [PAD] token.")
    _loaded_model, checkpoint = load_checkpoint(args.config_checkpoint, device)
    max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    warmup_raw = build_raw_stream(
        useful_evidence_people=set(),
        composition_holdout_people=parse_composition_holdout_people(args.composition_holdout_people),
        include_composition_rules=args.include_composition_rules,
    )
    warmup_encoded = encode_raw_groups(warmup_raw, tokenizer, max_seq_len=max_seq_len)
    encoded_people = make_encoded_trace_people(tokenizer=tokenizer, max_seq_len=max_seq_len)
    base_model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed)
    print("TINY CONSEQUENCE ACTION LEARNER")
    print("=" * 120)
    print(
        f"device={device} train_episodes={args.train_episodes} test_episodes={args.test_episodes} "
        f"virtual_steps={args.virtual_steps}"
    )
    train_bootstrap_stage(
        args=args,
        model=base_model,
        stage_examples=warmup_encoded["stage1"],
        pad_id=pad_id,
        device=device,
        epochs=args.stage1_epochs,
        seed=args.seed + 1000,
        label="base stage1",
    )
    freeze_module(base_model)
    reference_logits = teacher_logits_by_person(
        model=base_model,
        encoded_people=encoded_people,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    train_samples: list[DecisionSample] = []
    for episode_index in tqdm(range(args.train_episodes), desc="collect train consequences"):
        train_samples.extend(
            collect_episode_samples(
                args=args,
                episode_index=episode_index,
                base_model=base_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                reference_logits=reference_logits,
                encoded_people=encoded_people,
                pad_id=pad_id,
                device=device,
            )
        )
    test_samples: list[DecisionSample] = []
    for episode_index in tqdm(range(args.train_episodes, args.train_episodes + args.test_episodes), desc="collect test consequences"):
        test_samples.extend(
            collect_episode_samples(
                args=args,
                episode_index=episode_index,
                base_model=base_model,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                reference_logits=reference_logits,
                encoded_people=encoded_people,
                pad_id=pad_id,
                device=device,
            )
        )
    scorer, scorer_trace = train_scorer(args=args, samples=train_samples, device=device)
    train_eval = evaluate_scorer(scorer=scorer, samples=train_samples, device=device)
    test_eval = evaluate_scorer(scorer=scorer, samples=test_samples, device=device)
    output = {
        "question": (
            "Can a small neural scorer learn learn/preserve/guard/drop choices from measured virtual-update "
            "consequences rather than explicit role labels?"
        ),
        "config": {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "trace_feature_names": list(TRACE_FEATURE_NAMES),
            "action_names": list(ACTION_NAMES),
        },
        "summary": {
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "train_best_actions": summarize_best_actions(train_samples),
            "test_best_actions": summarize_best_actions(test_samples),
            "train_action_accuracy": train_eval["action_accuracy"],
            "test_action_accuracy": test_eval["action_accuracy"],
            "train_majority_accuracy": train_eval["majority_accuracy"],
            "test_majority_accuracy": test_eval["majority_accuracy"],
            "train_macro_recall": train_eval["macro_recall"],
            "test_macro_recall": test_eval["macro_recall"],
            "train_mean_regret": train_eval["mean_regret"],
            "test_mean_regret": test_eval["mean_regret"],
        },
        "scorer_trace": scorer_trace,
        "train_eval": train_eval,
        "test_eval": test_eval,
        "train_samples": [sample.__dict__ for sample in train_samples],
        "test_samples": [sample.__dict__ for sample in test_samples],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nTINY CONSEQUENCE ACTION LEARNER SUMMARY")
    print("=" * 120)
    print(f"train_samples={len(train_samples)} test_samples={len(test_samples)}")
    print(f"train_best_actions={output['summary']['train_best_actions']}")
    print(f"test_best_actions={output['summary']['test_best_actions']}")
    print(
        f"train action_acc={train_eval['action_accuracy']:.3f} regret={train_eval['mean_regret']:.4f} "
        f"test action_acc={test_eval['action_accuracy']:.3f} regret={test_eval['mean_regret']:.4f}"
    )
    print(
        f"train majority={train_eval['majority_accuracy']:.3f} macro_recall={train_eval['macro_recall']:.3f} "
        f"chosen={train_eval['chosen_action_counts']}"
    )
    print(
        f"test  majority={test_eval['majority_accuracy']:.3f} macro_recall={test_eval['macro_recall']:.3f} "
        f"chosen={test_eval['chosen_action_counts']}"
    )
    print(f"wrote_json={args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-consequence-action-learner-seed0.json"))
    parser.add_argument("--composition-holdout-people", type=str, default="Kira,Luca")
    parser.add_argument("--include-composition-rules", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=80)
    parser.add_argument("--train-episodes", type=int, default=6)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--virtual-steps", type=int, default=12)
    parser.add_argument("--scorer-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--virtual-lr", type=float, default=2e-3)
    parser.add_argument("--scorer-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--adapter-scale", type=float, default=4.0)
    parser.add_argument("--adapter-init-std", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--preserve-count", type=int, default=2)
    parser.add_argument("--drop-count", type=int, default=2)
    parser.add_argument("--guard-count", type=int, default=2)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-guard", type=float, default=1.0)
    parser.add_argument("--lambda-new", type=float, default=1.0)
    parser.add_argument("--lambda-drop", type=float, default=1.0)
    parser.add_argument("--drop-target-probability", type=float, default=0.05)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--loss-clip", type=float, default=8.0)
    parser.add_argument("--trace-budget", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
