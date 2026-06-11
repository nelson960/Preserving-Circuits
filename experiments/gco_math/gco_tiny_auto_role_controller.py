"""Automatic role-controller diagnostic for toy continual learning.

The previous controlled architecture used explicit preserve/drop/neutral
labels. This script removes those direct role labels. Instead, it gives the
controller evidence:

    learned behavior quality from the model
    current usefulness from the incoming task stream
    obsolete evidence from an external event stream
    capacity pressure

The controller maps evidence to roles:

    preserve: learned and currently useful
    drop: learned, obsolete, not useful, and capacity pressure is high
    guard: everything uncertain or unproven

The model still does not get final authority over deletion. The hard rule is:
if evidence is uncertain, guard rather than drop.
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
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_tiny_cl_behavior_path import load_checkpoint
from experiments.gco_math.gco_tiny_recursive_controlled_architecture import (
    cap_examples,
    train_bootstrap_stage,
    train_direct_stage,
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
    encode_examples,
    evaluate_examples,
    make_model_from_config,
    possession_examples,
    relation_items,
)


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


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


def build_raw_stream(
    *,
    useful_evidence_people: set[str],
    composition_holdout_people: set[str],
    include_composition_rules: bool,
) -> dict[str, list[QAExample]]:
    items = relation_items()
    known_people = {item.person for item in items}
    unknown_useful = useful_evidence_people.difference(known_people)
    unknown_holdout = composition_holdout_people.difference(known_people)
    if unknown_useful:
        raise ValueError(f"Unknown useful evidence people: {sorted(unknown_useful)}.")
    if unknown_holdout:
        raise ValueError(f"Unknown composition holdout people: {sorted(unknown_holdout)}.")
    stage1_items = [item for item in items if item.stage == 1]
    stage2_items = [item for item in items if item.stage == 2]
    stage1 = [example for item in stage1_items for example in possession_examples(item)]
    stage2 = [example for item in stage2_items for example in possession_examples(item)]

    stage3 = [example for item in items for example in affordance_examples(item)]
    if include_composition_rules:
        stage3.extend(composition_rule_examples())
    for item in items:
        if item.person in useful_evidence_people and item.person not in composition_holdout_people:
            stage3.append(composition_example(item, trained=True))
            if include_composition_rules:
                stage3.extend(composition_chain_examples(item))

    eval_all: list[QAExample] = []
    for item in items:
        eval_all.extend(possession_examples(item))
        eval_all.extend(affordance_examples(item))
        eval_all.append(composition_example(item, trained=item.person in useful_evidence_people))
    return {"stage1": stage1, "stage2": stage2, "stage3": stage3, "eval_all": eval_all}


def examples_by_stage1_person() -> dict[str, list[QAExample]]:
    result: dict[str, list[QAExample]] = {}
    for item in relation_items():
        if item.stage == 1:
            result[item.person] = possession_examples(item)
    return result


def encode_raw_groups(
    raw_groups: dict[str, list[QAExample]],
    tokenizer: Tokenizer,
    *,
    max_seq_len: int,
) -> dict[str, list[EncodedExample]]:
    return {name: encode_examples(examples, tokenizer, max_seq_len=max_seq_len) for name, examples in raw_groups.items()}


def exact_for_examples(
    model: torch.nn.Module,
    examples: list[EncodedExample],
    *,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    metrics = evaluate_examples(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)["overall"]
    return {
        "loss": float(metrics["loss"]),
        "token_accuracy": float(metrics["token_accuracy"]),
        "exact_match": float(metrics["exact_match"]),
    }


def count_person_references(examples: list[QAExample], person: str) -> int:
    needle = person.lower()
    return sum(1 for example in examples if needle in example.text.lower())


def assign_roles(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    raw_groups: dict[str, list[QAExample]],
    encoded_people: dict[str, list[EncodedExample]],
    obsolete_evidence_people: set[str],
    pad_id: int,
    device: torch.device,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    role_by_person: dict[str, str] = {}
    evidence_by_person: dict[str, dict[str, float]] = {}
    max_use_count = max(
        1,
        max(count_person_references(raw_groups["stage2"] + raw_groups["stage3"], person) for person in encoded_people),
    )
    for person, examples in sorted(encoded_people.items()):
        behavior = exact_for_examples(
            model,
            examples,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        learned_score = behavior["exact_match"]
        use_count = count_person_references(raw_groups["stage2"] + raw_groups["stage3"], person)
        usefulness_score = min(1.0, float(use_count) / float(max_use_count))
        obsolete_count = args.obsolete_evidence_count if person in obsolete_evidence_people else 0
        obsolete_score = min(1.0, float(obsolete_count) / float(args.obsolete_threshold_count))
        capacity_score = args.capacity_pressure
        preserve_score = learned_score * usefulness_score
        drop_score = learned_score * obsolete_score * capacity_score * (1.0 - usefulness_score)
        if preserve_score >= args.preserve_threshold:
            role = "preserve"
        elif drop_score >= args.drop_threshold:
            role = "drop"
        else:
            role = "guard"
        role_by_person[person] = role
        evidence_by_person[person] = {
            "loss": behavior["loss"],
            "token_accuracy": behavior["token_accuracy"],
            "exact_match": behavior["exact_match"],
            "use_count": float(use_count),
            "usefulness_score": usefulness_score,
            "obsolete_count": float(obsolete_count),
            "obsolete_score": obsolete_score,
            "capacity_score": capacity_score,
            "preserve_score": preserve_score,
            "drop_score": drop_score,
        }
    roles = set(role_by_person.values())
    required = {"preserve", "drop", "guard"}
    missing = required.difference(roles)
    if missing:
        raise RuntimeError(f"Role controller did not produce required roles {sorted(missing)}; roles={role_by_person}.")
    return role_by_person, evidence_by_person


def build_role_groups(
    *,
    role_by_person: dict[str, str],
    encoded_people: dict[str, list[EncodedExample]],
) -> dict[str, list[EncodedExample]]:
    groups = {"preserve": [], "drop": [], "neutral": []}
    for person, role in sorted(role_by_person.items()):
        if role == "preserve":
            groups["preserve"].extend(encoded_people[person])
        elif role == "drop":
            groups["drop"].extend(encoded_people[person])
        elif role == "guard":
            groups["neutral"].extend(encoded_people[person])
        else:
            raise ValueError(f"Unknown role {role!r} for person {person}.")
    for name, examples in groups.items():
        if not examples:
            raise RuntimeError(f"Role group {name} is empty.")
    return groups


def evaluate_role_summary(
    *,
    model: torch.nn.Module,
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


def evaluate_category_breakdown(
    *,
    model: torch.nn.Module,
    examples: list[EncodedExample],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    metrics = evaluate_examples(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)
    return {
        category: row
        for category, row in metrics.items()
        if category != "overall"
    }


def run_method(
    *,
    args: argparse.Namespace,
    method: str,
    checkpoint: dict[str, Any],
    raw_groups: dict[str, list[QAExample]],
    encoded_base_groups: dict[str, list[EncodedExample]],
    encoded_people: dict[str, list[EncodedExample]],
    obsolete_evidence_people: set[str],
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if method not in {"naive", "auto_direct"}:
        raise ValueError(f"Unknown method: {method!r}.")
    model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed + {"naive": 11, "auto_direct": 22}[method])
    traces: list[dict[str, Any]] = []
    trace1 = train_bootstrap_stage(
        args=args,
        model=model,
        stage_examples=encoded_base_groups["stage1"],
        pad_id=pad_id,
        device=device,
        epochs=args.stage1_epochs,
        seed=args.seed + 100,
        label=f"{method} stage1",
    )
    traces.append({"stage": 1, "mode": "bootstrap", "trace": trace1})

    role_by_person, role_evidence = assign_roles(
        args=args,
        model=model,
        raw_groups=raw_groups,
        encoded_people=encoded_people,
        obsolete_evidence_people=obsolete_evidence_people,
        pad_id=pad_id,
        device=device,
    )
    role_groups = build_role_groups(role_by_person=role_by_person, encoded_people=encoded_people)
    groups = {
        **encoded_base_groups,
        "preserve": role_groups["preserve"],
        "drop": role_groups["drop"],
        "neutral": role_groups["neutral"],
    }

    if method == "naive":
        for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
            trace = train_bootstrap_stage(
                args=args,
                model=model,
                stage_examples=groups[stage_name],
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 100 * stage_number,
                label=f"{method} {stage_name}",
            )
            traces.append({"stage": stage_number, "mode": "naive", "trace": trace})
    else:
        preserve_examples = cap_examples(groups["preserve"], budget=args.preserve_budget)
        drop_examples = cap_examples(groups["drop"], budget=args.drop_budget)
        guard_examples = cap_examples(groups["neutral"], budget=args.guard_budget)
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
                stage_examples=groups[stage_name],
                preserve_examples=preserve_examples,
                preserve_logits=preserve_logits,
                guard_examples=guard_examples,
                guard_logits=guard_logits,
                drop_examples=drop_examples,
                pad_id=pad_id,
                device=device,
                epochs=args.stage_epochs,
                seed=args.seed + 200 * stage_number,
                label=f"{method} {stage_name}",
            )
            traces.append({"stage": stage_number, "mode": "auto_direct", "trace": trace})
            if args.add_learned_stage_to_guard:
                guard_examples = cap_examples(guard_examples + groups[stage_name], budget=args.guard_budget)

    metrics = evaluate_role_summary(
        model=model,
        groups=groups,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    category_breakdown = evaluate_category_breakdown(
        model=model,
        examples=groups["eval_all"],
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    return {
        "method": method,
        "role_by_person": role_by_person,
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
    positive_int("stage1_epochs", args.stage1_epochs)
    positive_int("stage_epochs", args.stage_epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("preserve_budget", args.preserve_budget)
    positive_int("guard_budget", args.guard_budget)
    positive_int("drop_budget", args.drop_budget)
    positive_int("obsolete_evidence_count", args.obsolete_evidence_count)
    positive_int("obsolete_threshold_count", args.obsolete_threshold_count)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    nonnegative_float("lambda_preserve", args.lambda_preserve)
    nonnegative_float("lambda_guard", args.lambda_guard)
    nonnegative_float("drop_loss_weight", args.drop_loss_weight)
    positive_float("drop_target_probability", args.drop_target_probability)
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")
    positive_float("distill_temperature", args.distill_temperature)
    positive_float("grad_clip", args.grad_clip)
    probability("capacity_pressure", args.capacity_pressure)
    probability("preserve_threshold", args.preserve_threshold)
    probability("drop_threshold", args.drop_threshold)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer must define [PAD].")
    useful_evidence_people = parse_people(args.useful_evidence_people, name="useful_evidence_people")
    obsolete_evidence_people = parse_people(args.obsolete_evidence_people, name="obsolete_evidence_people")
    composition_holdout_people = parse_people(args.composition_holdout_people, name="composition_holdout_people", allow_empty=True)
    overlap = useful_evidence_people.intersection(obsolete_evidence_people)
    if overlap:
        raise ValueError(f"People cannot have both useful and obsolete evidence: {sorted(overlap)}.")

    _config_model, checkpoint = load_checkpoint(args.config_checkpoint, device)
    max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    raw_groups = build_raw_stream(
        useful_evidence_people=useful_evidence_people,
        composition_holdout_people=composition_holdout_people,
        include_composition_rules=args.include_composition_rules,
    )
    raw_people = examples_by_stage1_person()
    encoded_base_groups = encode_raw_groups(raw_groups, tokenizer, max_seq_len=max_seq_len)
    encoded_people = encode_raw_groups(raw_people, tokenizer, max_seq_len=max_seq_len)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")

    print("TINY AUTO ROLE-CONTROLLER CL")
    print("=" * 112)
    print(
        f"device={device} methods={methods} useful_evidence={sorted(useful_evidence_people)} "
        f"obsolete_evidence={sorted(obsolete_evidence_people)} capacity={args.capacity_pressure:.3f}"
    )
    print(
        f"examples stage1={len(encoded_base_groups['stage1'])} stage2={len(encoded_base_groups['stage2'])} "
        f"stage3={len(encoded_base_groups['stage3'])} eval={len(encoded_base_groups['eval_all'])}"
    )

    results = [
        run_method(
            args=args,
            method=method,
            checkpoint=checkpoint,
            raw_groups=raw_groups,
            encoded_base_groups=encoded_base_groups,
            encoded_people=encoded_people,
            obsolete_evidence_people=obsolete_evidence_people,
            pad_id=pad_id,
            device=device,
        )
        for method in methods
    ]

    summary = {
        "question": "Can a bounded controller assign preserve/drop/guard roles from evidence and drive recursive CL?",
        "evidence_config": {
            "useful_evidence_people": sorted(useful_evidence_people),
            "obsolete_evidence_people": sorted(obsolete_evidence_people),
            "composition_holdout_people": sorted(composition_holdout_people),
            "capacity_pressure": args.capacity_pressure,
            "obsolete_evidence_count": args.obsolete_evidence_count,
            "obsolete_threshold_count": args.obsolete_threshold_count,
            "preserve_threshold": args.preserve_threshold,
            "drop_threshold": args.drop_threshold,
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

    print("\nTINY AUTO ROLE-CONTROLLER SUMMARY")
    print("=" * 112)
    for result in results:
        print(f"method={result['method']} roles={result['role_by_person']}")
    print("-" * 112)
    print(
        "{:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "method",
            "preserve",
            "drop",
            "guard",
            "stage2",
            "stage3",
            "eval_all",
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

    def compact(row: dict[str, float]) -> str:
        return "{:.3g}/{:.3f}".format(row["loss"], row["exact_match"])

    for result in results:
        metrics = result["metrics"]
        print(
            "{:>12} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
                result["method"],
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
        "{:>12} {:>34} {:>12} {:>12} {:>12} {:>8}".format(
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
                "{:>12} {:>34} {:12.5f} {:12.4f} {:12.4f} {:8.0f}".format(
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
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-auto-role-controller-seed0.json"))
    parser.add_argument("--methods", type=str, default="naive,auto_direct")
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
    parser.add_argument("--obsolete-evidence-count", type=int, default=4)
    parser.add_argument("--obsolete-threshold-count", type=int, default=3)
    parser.add_argument("--capacity-pressure", type=float, default=1.0)
    parser.add_argument("--preserve-threshold", type=float, default=0.5)
    parser.add_argument("--drop-threshold", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
