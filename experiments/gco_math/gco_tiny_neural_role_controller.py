"""Neural role-controller diagnostic for toy continual learning.

This experiment removes the threshold rule used by
gco_tiny_auto_role_controller.py. A small neural controller receives evidence
features for each learned behavior and predicts one of:

    preserve
    drop
    guard

The predicted roles then drive the same controlled continual-learning loop used
by the recursive architecture experiments. This is still a toy world, but the
role choice is a learned neural mapping rather than an if/else policy.
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
from experiments.gco_math.gco_tiny_auto_role_controller import (
    build_raw_stream,
    build_role_groups,
    count_person_references,
    encode_raw_groups,
    evaluate_category_breakdown,
    evaluate_role_summary,
    examples_by_stage1_person,
    oracle_roles,
    random_roles_matching_counts,
    role_match_report,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import load_checkpoint
from experiments.gco_math.gco_tiny_recursive_controlled_architecture import (
    cap_examples,
    train_bootstrap_stage,
    train_direct_stage,
)
from experiments.gco_math.gco_tiny_reusable_computation_cl import (
    EncodedExample,
    QAExample,
    collect_example_logits,
    evaluate_examples,
    make_model_from_config,
    relation_items,
)


ROLE_TO_INDEX = {"preserve": 0, "drop": 1, "guard": 2}
INDEX_TO_ROLE = {index: role for role, index in ROLE_TO_INDEX.items()}
FEATURE_NAMES = [
    "learned_exact",
    "learned_token_accuracy",
    "loss_score",
    "usefulness_score",
    "obsolete_score",
    "capacity_pressure",
    "evidence_conflict",
]


class NeuralRoleController(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(ROLE_TO_INDEX)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


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


def parse_people(raw: str, *, name: str, allow_empty: bool = False) -> set[str]:
    people = {item.strip() for item in raw.split(",") if item.strip()}
    if not people and not allow_empty:
        raise ValueError(f"--{name.replace('_', '-')} must contain at least one person.")
    known = {item.person for item in relation_items()}
    unknown = people.difference(known)
    if unknown:
        raise ValueError(f"Unknown people in --{name.replace('_', '-')}: {sorted(unknown)}.")
    return people


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    allowed = {"naive", "oracle", "neural", "random"}
    unknown = sorted(set(methods).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; allowed={sorted(allowed)}.")
    return methods


def bernoulli_count(*, count: int, keep_probability: float, rng: random.Random) -> int:
    nonnegative_int("count", count)
    probability("keep_probability", keep_probability)
    return sum(1 for _ in range(count) if rng.random() < keep_probability)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def feature_vector(
    *,
    learned_exact: float,
    learned_token_accuracy: float,
    learned_loss: float,
    useful_count: int,
    max_use_count: int,
    obsolete_count: int,
    obsolete_threshold_count: int,
    capacity_pressure: float,
    loss_clip: float,
) -> list[float]:
    probability("learned_exact", learned_exact)
    probability("learned_token_accuracy", learned_token_accuracy)
    nonnegative_float("learned_loss", learned_loss)
    positive_int("max_use_count", max_use_count)
    positive_int("obsolete_threshold_count", obsolete_threshold_count)
    probability("capacity_pressure", capacity_pressure)
    positive_float("loss_clip", loss_clip)
    usefulness_score = clamp01(float(useful_count) / float(max_use_count))
    obsolete_score = clamp01(float(obsolete_count) / float(obsolete_threshold_count))
    loss_score = math.exp(-min(learned_loss, loss_clip))
    return [
        learned_exact,
        learned_token_accuracy,
        loss_score,
        usefulness_score,
        obsolete_score,
        capacity_pressure,
        min(usefulness_score, obsolete_score),
    ]


def sampled_feature_for_role(*, args: argparse.Namespace, role: str, rng: random.Random) -> list[float]:
    if role not in ROLE_TO_INDEX:
        raise ValueError(f"Unknown sampled role {role!r}.")
    if role == "guard":
        learned_high = rng.random() < args.controller_guard_learned_probability
    else:
        learned_high = rng.random() < args.controller_learned_high_probability
    if learned_high:
        learned_exact = rng.uniform(args.controller_learned_exact_min, 1.0)
        learned_token_accuracy = rng.uniform(args.controller_learned_token_min, 1.0)
        learned_loss = rng.uniform(0.0, args.controller_learned_loss_max)
    else:
        learned_exact = rng.uniform(0.0, args.controller_guard_exact_max)
        learned_token_accuracy = rng.uniform(0.0, args.controller_guard_token_max)
        learned_loss = rng.uniform(args.controller_guard_loss_min, args.controller_loss_clip)

    max_use_count = args.controller_max_use_count
    if role == "preserve":
        useful_count = bernoulli_count(count=max_use_count, keep_probability=args.controller_useful_recall, rng=rng)
        obsolete_count = bernoulli_count(
            count=args.obsolete_threshold_count,
            keep_probability=args.controller_obsolete_false_positive,
            rng=rng,
        )
        capacity_pressure = rng.uniform(args.controller_capacity_min, args.controller_capacity_max)
    elif role == "drop":
        useful_count = bernoulli_count(count=max_use_count, keep_probability=args.controller_useful_false_positive, rng=rng)
        obsolete_count = bernoulli_count(
            count=args.obsolete_threshold_count,
            keep_probability=args.controller_obsolete_recall,
            rng=rng,
        )
        capacity_pressure = rng.uniform(args.controller_drop_capacity_min, args.controller_capacity_max)
    else:
        useful_count = bernoulli_count(count=max_use_count, keep_probability=args.controller_guard_useful_signal_rate, rng=rng)
        obsolete_count = bernoulli_count(
            count=args.obsolete_threshold_count,
            keep_probability=args.controller_guard_obsolete_signal_rate,
            rng=rng,
        )
        capacity_pressure = rng.uniform(args.controller_capacity_min, args.controller_capacity_max)

    return feature_vector(
        learned_exact=learned_exact,
        learned_token_accuracy=learned_token_accuracy,
        learned_loss=learned_loss,
        useful_count=useful_count,
        max_use_count=max_use_count,
        obsolete_count=obsolete_count,
        obsolete_threshold_count=args.obsolete_threshold_count,
        capacity_pressure=capacity_pressure,
        loss_clip=args.controller_loss_clip,
    )


def make_controller_dataset(
    *,
    args: argparse.Namespace,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_int("count", count)
    rng = random.Random(seed)
    features: list[list[float]] = []
    labels: list[int] = []
    roles = list(ROLE_TO_INDEX)
    for _ in range(count):
        role = roles[rng.randrange(len(roles))]
        features.append(sampled_feature_for_role(args=args, role=role, rng=rng))
        labels.append(ROLE_TO_INDEX[role])
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def controller_accuracy(controller: NeuralRoleController, features: torch.Tensor, labels: torch.Tensor) -> float:
    controller.eval()
    with torch.no_grad():
        predictions = controller(features).argmax(dim=-1)
    return float((predictions == labels).to(torch.float32).mean().item())


def train_neural_controller(
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[NeuralRoleController, dict[str, float]]:
    train_features, train_labels = make_controller_dataset(
        args=args,
        count=args.controller_train_examples,
        seed=args.seed + 7000,
    )
    eval_features, eval_labels = make_controller_dataset(
        args=args,
        count=args.controller_eval_examples,
        seed=args.seed + 8000,
    )
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    eval_features = eval_features.to(device)
    eval_labels = eval_labels.to(device)
    controller = NeuralRoleController(input_dim=len(FEATURE_NAMES), hidden_dim=args.controller_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.controller_lr, weight_decay=args.controller_weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 9000)
    for epoch in range(1, args.controller_epochs + 1):
        controller.train()
        permutation = torch.randperm(train_features.shape[0], generator=generator)
        total_loss = 0.0
        batches = 0
        pbar = tqdm(range(0, train_features.shape[0], args.controller_batch_size), desc=f"role-controller {epoch}/{args.controller_epochs}")
        for start in pbar:
            indices = permutation[start : start + args.controller_batch_size].to(device)
            batch_features = train_features.index_select(0, indices)
            batch_labels = train_labels.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            logits = controller(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), args.controller_grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            pbar.set_postfix({"loss": f"{float(loss.detach().cpu()):.3g}"})
        if batches <= 0:
            raise RuntimeError("Role-controller training saw zero batches.")
        if epoch == 1 or epoch == args.controller_epochs or epoch % args.controller_print_every == 0:
            train_acc = controller_accuracy(controller, train_features, train_labels)
            eval_acc = controller_accuracy(controller, eval_features, eval_labels)
            print(
                "role-controller epoch={:4d} loss={:.5f} train_acc={:.4f} eval_acc={:.4f}".format(
                    epoch,
                    total_loss / float(batches),
                    train_acc,
                    eval_acc,
                )
            )
    report = {
        "train_accuracy": controller_accuracy(controller, train_features, train_labels),
        "eval_accuracy": controller_accuracy(controller, eval_features, eval_labels),
        "train_examples": float(args.controller_train_examples),
        "eval_examples": float(args.controller_eval_examples),
    }
    return controller, report


def person_behavior_metrics(
    *,
    model: torch.nn.Module,
    encoded_people: dict[str, list[EncodedExample]],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    return {
        person: evaluate_examples(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)["overall"]
        for person, examples in sorted(encoded_people.items())
    }


def observed_evidence_counts(
    *,
    args: argparse.Namespace,
    person: str,
    raw_groups: dict[str, list[QAExample]],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
    rng: random.Random,
) -> tuple[int, int, int]:
    raw_use_count = count_person_references(raw_groups["stage2"] + raw_groups["stage3"], person)
    if person in useful_evidence_people:
        useful_count = bernoulli_count(
            count=raw_use_count,
            keep_probability=args.useful_evidence_keep_probability,
            rng=rng,
        )
    else:
        useful_count = bernoulli_count(
            count=args.false_useful_evidence_count,
            keep_probability=args.false_useful_evidence_probability,
            rng=rng,
        )

    if person in obsolete_evidence_people:
        obsolete_count = bernoulli_count(
            count=args.obsolete_evidence_count,
            keep_probability=args.obsolete_evidence_keep_probability,
            rng=rng,
        )
    else:
        obsolete_count = bernoulli_count(
            count=args.false_obsolete_evidence_count,
            keep_probability=args.false_obsolete_evidence_probability,
            rng=rng,
        )
    return useful_count, obsolete_count, raw_use_count


def neural_role_prediction(
    *,
    args: argparse.Namespace,
    controller: NeuralRoleController,
    model: torch.nn.Module,
    raw_groups: dict[str, list[QAExample]],
    encoded_people: dict[str, list[EncodedExample]],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
    pad_id: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    behavior = person_behavior_metrics(
        model=model,
        encoded_people=encoded_people,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    max_use_count = max(
        1,
        max(count_person_references(raw_groups["stage2"] + raw_groups["stage3"], person) for person in encoded_people),
    )
    rng = random.Random(seed)
    role_by_person: dict[str, str] = {}
    evidence_by_person: dict[str, dict[str, Any]] = {}
    controller.eval()
    for person in sorted(encoded_people):
        useful_count, obsolete_count, raw_use_count = observed_evidence_counts(
            args=args,
            person=person,
            raw_groups=raw_groups,
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
            rng=rng,
        )
        row = behavior[person]
        features = feature_vector(
            learned_exact=float(row["exact_match"]),
            learned_token_accuracy=float(row["token_accuracy"]),
            learned_loss=float(row["loss"]),
            useful_count=useful_count,
            max_use_count=max_use_count,
            obsolete_count=obsolete_count,
            obsolete_threshold_count=args.obsolete_threshold_count,
            capacity_pressure=args.capacity_pressure,
            loss_clip=args.controller_loss_clip,
        )
        feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = controller(feature_tensor).squeeze(0)
            probabilities = F.softmax(logits, dim=-1)
            role_index = int(logits.argmax(dim=-1).item())
        role = INDEX_TO_ROLE[role_index]
        role_by_person[person] = role
        evidence_by_person[person] = {
            "features": dict(zip(FEATURE_NAMES, features, strict=True)),
            "raw_use_count": float(raw_use_count),
            "observed_useful_count": float(useful_count),
            "observed_obsolete_count": float(obsolete_count),
            "behavior": row,
            "logits": {INDEX_TO_ROLE[index]: float(logits[index].detach().cpu()) for index in INDEX_TO_ROLE},
            "probabilities": {
                INDEX_TO_ROLE[index]: float(probabilities[index].detach().cpu())
                for index in INDEX_TO_ROLE
            },
        }
    return role_by_person, evidence_by_person


def choose_roles(
    *,
    args: argparse.Namespace,
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
        roles = random_roles_matching_counts(true_roles=true_roles, seed=seed)
        return roles, {"source": "random_count_matched"}
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
    if method not in {"naive", "oracle", "neural", "random"}:
        raise ValueError(f"Unknown method: {method!r}.")
    seed_offsets = {"naive": 101, "oracle": 202, "neural": 303, "random": 404}
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

    if method == "naive":
        role_by_person = dict(true_roles)
        role_evidence: dict[str, Any] = {"source": "not_used_by_naive"}
    else:
        role_by_person, role_evidence = choose_roles(
            args=args,
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
            seed=args.seed + 1200 + seed_offsets[method],
        )

    training_role_groups = build_role_groups(role_by_person=role_by_person, encoded_people=encoded_people)
    train_groups = {
        **encoded_base_groups,
        "preserve": training_role_groups["preserve"],
        "drop": training_role_groups["drop"],
        "neutral": training_role_groups["neutral"],
    }
    eval_groups = {
        **encoded_base_groups,
        "preserve": true_role_groups["preserve"],
        "drop": true_role_groups["drop"],
        "neutral": true_role_groups["neutral"],
    }

    if method == "naive":
        for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
            trace = train_bootstrap_stage(
                args=args,
                model=model,
                stage_examples=train_groups[stage_name],
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 1300 + seed_offsets[method] + stage_number,
                label=f"{method} {stage_name}",
            )
            traces.append({"stage": stage_number, "mode": "naive", "trace": trace})
    else:
        preserve_examples = cap_examples(train_groups["preserve"], budget=args.preserve_budget)
        drop_examples = cap_examples(train_groups["drop"], budget=args.drop_budget)
        guard_examples = cap_examples(train_groups["neutral"], budget=args.guard_budget)
        for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
            preserve_logits = collect_example_logits(
                model,
                preserve_examples,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            guard_logits = collect_example_logits(
                model,
                guard_examples,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            trace = train_direct_stage(
                args=args,
                model=model,
                stage_examples=train_groups[stage_name],
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
                guard_examples=guard_examples,
                guard_logits=guard_logits,
                drop_examples=drop_examples,
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 1400 + seed_offsets[method] + stage_number,
                label=f"{method} {stage_name}",
            )
            traces.append({"stage": stage_number, "mode": method, "trace": trace})
            if args.add_learned_stage_to_guard:
                guard_examples = cap_examples(guard_examples + train_groups[stage_name], budget=args.guard_budget)

    role_report = role_match_report(predicted_roles=role_by_person, true_roles=true_roles)
    metrics = evaluate_role_summary(
        model=model,
        groups=eval_groups,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    category_breakdown = evaluate_category_breakdown(
        model=model,
        examples=eval_groups["eval_all"],
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    return {
        "method": method,
        "role_by_person": role_by_person,
        "true_role_by_person": true_roles,
        "role_match_report": role_report,
        "role_evidence": role_evidence,
        "metrics": metrics,
        "category_breakdown": category_breakdown,
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
        "obsolete_threshold_count",
        "controller_train_examples",
        "controller_eval_examples",
        "controller_epochs",
        "controller_batch_size",
        "controller_hidden_dim",
        "controller_max_use_count",
        "controller_print_every",
    ]:
        positive_int(name, getattr(args, name))
    for name in [
        "false_useful_evidence_count",
        "false_obsolete_evidence_count",
        "obsolete_evidence_count",
    ]:
        nonnegative_int(name, getattr(args, name))
    for name in [
        "lr",
        "controller_lr",
        "distill_temperature",
        "grad_clip",
        "controller_grad_clip",
        "drop_target_probability",
        "controller_loss_clip",
        "controller_guard_loss_min",
        "controller_learned_loss_max",
    ]:
        positive_float(name, getattr(args, name))
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")
    if args.controller_guard_loss_min > args.controller_loss_clip:
        raise ValueError("--controller-guard-loss-min cannot exceed --controller-loss-clip.")
    for name in [
        "weight_decay",
        "lambda_preserve",
        "lambda_guard",
        "drop_loss_weight",
        "momentum",
        "controller_weight_decay",
    ]:
        nonnegative_float(name, getattr(args, name))
    for name in [
        "capacity_pressure",
        "useful_evidence_keep_probability",
        "obsolete_evidence_keep_probability",
        "false_useful_evidence_probability",
        "false_obsolete_evidence_probability",
        "controller_learned_high_probability",
        "controller_guard_learned_probability",
        "controller_learned_exact_min",
        "controller_learned_token_min",
        "controller_guard_exact_max",
        "controller_guard_token_max",
        "controller_useful_recall",
        "controller_useful_false_positive",
        "controller_obsolete_recall",
        "controller_obsolete_false_positive",
        "controller_guard_useful_signal_rate",
        "controller_guard_obsolete_signal_rate",
        "controller_capacity_min",
        "controller_capacity_max",
        "controller_drop_capacity_min",
    ]:
        probability(name, getattr(args, name))
    if args.controller_capacity_min > args.controller_capacity_max:
        raise ValueError("--controller-capacity-min cannot exceed --controller-capacity-max.")
    if args.controller_drop_capacity_min > args.controller_capacity_max:
        raise ValueError("--controller-drop-capacity-min cannot exceed --controller-capacity-max.")


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
    true_role_groups = build_role_groups(role_by_person=true_roles, encoded_people=encoded_people)
    methods = parse_methods(args.methods)

    print("TINY NEURAL ROLE-CONTROLLER CL")
    print("=" * 112)
    print(
        f"device={device} methods={methods} true_roles={true_roles} "
        f"useful_evidence={sorted(useful_evidence_people)} obsolete_evidence={sorted(obsolete_evidence_people)}"
    )
    print(
        f"evidence_keep useful={args.useful_evidence_keep_probability:.3f} "
        f"obsolete={args.obsolete_evidence_keep_probability:.3f} capacity={args.capacity_pressure:.3f}"
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
        "question": "Can a learned neural controller choose preserve/drop/guard roles and drive controlled CL under scarce evidence?",
        "controller": {
            "type": "tiny_mlp",
            "feature_names": FEATURE_NAMES,
            "report": controller_report,
            "hidden_dim": args.controller_hidden_dim,
        },
        "evidence_config": {
            "oracle_preserve_people": sorted(oracle_preserve_people),
            "oracle_drop_people": sorted(oracle_drop_people),
            "true_role_by_person": true_roles,
            "useful_evidence_people": sorted(useful_evidence_people),
            "obsolete_evidence_people": sorted(obsolete_evidence_people),
            "composition_holdout_people": sorted(composition_holdout_people),
            "capacity_pressure": args.capacity_pressure,
            "useful_evidence_keep_probability": args.useful_evidence_keep_probability,
            "obsolete_evidence_keep_probability": args.obsolete_evidence_keep_probability,
            "false_useful_evidence_count": args.false_useful_evidence_count,
            "false_obsolete_evidence_count": args.false_obsolete_evidence_count,
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
            "lambda_preserve": args.lambda_preserve,
            "lambda_guard": args.lambda_guard,
            "drop_loss_weight": args.drop_loss_weight,
            "drop_target_probability": args.drop_target_probability,
            "distill_temperature": args.distill_temperature,
        },
        "raw_groups": {name: [asdict(example) for example in examples] for name, examples in raw_groups.items()},
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTINY NEURAL ROLE-CONTROLLER SUMMARY")
    print("=" * 112)
    for result in results:
        role_report = result["role_match_report"]
        print(
            f"method={result['method']} role_accuracy={role_report['accuracy']:.3f} "
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
    print("\nEVAL CATEGORY BREAKDOWN")
    print("-" * 112)
    print(
        "{:>10} {:>34} {:>12} {:>12} {:>12} {:>8}".format(
            "method",
            "category",
            "loss",
            "tok_acc",
            "exact",
            "n",
        )
    )
    for result in results:
        for category, row in sorted(result["category_breakdown"].items()):
            print(
                "{:>10} {:>34} {:12.5f} {:12.4f} {:12.4f} {:8.0f}".format(
                    result["method"],
                    category,
                    row["loss"],
                    row["token_accuracy"],
                    row["exact_match"],
                    row["example_count"],
                )
            )
    print(f"wrote_json={args.output_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-neural-role-controller-seed0.json"))
    parser.add_argument("--methods", type=str, default="naive,oracle,neural,random")
    parser.add_argument("--oracle-preserve-people", type=str, default="Alice,Bruno")
    parser.add_argument("--oracle-drop-people", type=str, default="Clara,Darin")
    parser.add_argument("--useful-evidence-people", type=str, default="Alice,Bruno")
    parser.add_argument("--obsolete-evidence-people", type=str, default="Clara,Darin")
    parser.add_argument("--composition-holdout-people", type=str, default="Kira,Luca")
    parser.add_argument("--include-composition-rules", action="store_true")
    parser.add_argument("--add-learned-stage-to-guard", action="store_true")
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
    parser.add_argument("--lambda-guard", type=float, default=1.0)
    parser.add_argument("--drop-loss-weight", type=float, default=0.1)
    parser.add_argument("--drop-target-probability", type=float, default=0.05)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--obsolete-evidence-count", type=int, default=4)
    parser.add_argument("--obsolete-threshold-count", type=int, default=3)
    parser.add_argument("--capacity-pressure", type=float, default=1.0)
    parser.add_argument("--useful-evidence-keep-probability", type=float, default=0.6)
    parser.add_argument("--obsolete-evidence-keep-probability", type=float, default=0.6)
    parser.add_argument("--false-useful-evidence-count", type=int, default=1)
    parser.add_argument("--false-useful-evidence-probability", type=float, default=0.1)
    parser.add_argument("--false-obsolete-evidence-count", type=int, default=1)
    parser.add_argument("--false-obsolete-evidence-probability", type=float, default=0.1)
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
