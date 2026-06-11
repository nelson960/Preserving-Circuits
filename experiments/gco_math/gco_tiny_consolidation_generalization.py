"""Tiny consolidation-generalization diagnostic.

This experiment isolates the consolidation question from the write/protect
question. The model first learns reusable direct facts:

    person -> object
    object -> place

Then a consolidation stage receives different kinds of bridge/schema evidence.
Evaluation asks whether the model can answer held-out composition queries:

    person -> object -> place

The important distinction is:

    memory composition: "What can Alice open?"
    chain-context composition: facts are present in the prompt

If chain-context succeeds but memory composition fails, the model learned an
in-context rule but did not consolidate the rule into parametric memory. If both
fail, it did not learn the reusable computation. If held-out memory composition
succeeds, consolidation created a reusable internal structure rather than only
memorizing seen bridges.
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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    load_checkpoint,
    set_only_native_weights_trainable,
    trainable_weight_parameters,
)
from experiments.gco_math.gco_tiny_reusable_computation_cl import (
    EncodedExample,
    QAExample,
    affordance_examples,
    batch_examples,
    encode_examples,
    evaluate_examples,
    make_model_from_config,
    make_optimizer_for_model,
    masked_ce_loss,
    possession_examples,
    relation_items,
)


CONDITIONS = {
    "facts_only",
    "seen_composition",
    "chain_schema",
    "memory_schema",
    "oracle_all_composition",
}


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def parse_people(raw: str, *, name: str) -> set[str]:
    people = {item.strip() for item in raw.split(",") if item.strip()}
    if not people:
        raise ValueError(f"--{name.replace('_', '-')} must contain at least one person.")
    known = {item.person for item in relation_items()}
    unknown = people.difference(known)
    if unknown:
        raise ValueError(f"Unknown people in --{name.replace('_', '-')}: {sorted(unknown)}.")
    return people


def parse_conditions(raw: str) -> list[str]:
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    if not conditions:
        raise ValueError("--conditions must contain at least one condition.")
    unknown = sorted(set(conditions).difference(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Valid conditions: {sorted(CONDITIONS)}.")
    return conditions


def direct_fact_examples() -> list[QAExample]:
    return [
        example
        for item in relation_items()
        for example in (possession_examples(item) + affordance_examples(item))
    ]


def composition_memory_example(item: Any, *, trained: bool) -> QAExample:
    category = "composition_seen_memory" if trained else "composition_heldout_memory"
    return QAExample(
        stage=3,
        category=category,
        prompt=f"Question: What can {item.person} open? Answer:",
        answer=f" {item.place}.",
    )


def composition_chain_examples(item: Any, *, trained: bool) -> list[QAExample]:
    category = "composition_seen_chain" if trained else "composition_heldout_chain"
    return [
        QAExample(
            stage=3,
            category=category,
            prompt=f"{item.person}/{item.obj}/{item.place}. {item.person} opens? Answer:",
            answer=f" {item.place}.",
        ),
        QAExample(
            stage=3,
            category=category,
            prompt=f"{item.person}->{item.obj}->{item.place}. {item.person}->? Answer:",
            answer=f" {item.place}.",
        ),
    ]


def schema_rule_examples() -> list[QAExample]:
    return [
        QAExample(
            stage=3,
            category="schema_rule",
            prompt="Rule: person has object. object opens place. person opens? Answer:",
            answer=" the place.",
        ),
        QAExample(
            stage=3,
            category="schema_rule",
            prompt="Rule: person carries object. object opens place. person opens what? Answer:",
            answer=" the place.",
        ),
    ]


def build_raw_groups(*, holdout_people: set[str]) -> dict[str, list[QAExample]]:
    items = relation_items()
    known_people = {item.person for item in items}
    unknown = holdout_people.difference(known_people)
    if unknown:
        raise ValueError(f"Unknown holdout people: {sorted(unknown)}.")
    train_people = known_people.difference(holdout_people)
    direct = direct_fact_examples()

    seen_memory = [
        composition_memory_example(item, trained=True)
        for item in items
        if item.person in train_people
    ]
    heldout_memory = [
        composition_memory_example(item, trained=False)
        for item in items
        if item.person in holdout_people
    ]
    seen_chain = [
        example
        for item in items
        if item.person in train_people
        for example in composition_chain_examples(item, trained=True)
    ]
    heldout_chain = [
        example
        for item in items
        if item.person in holdout_people
        for example in composition_chain_examples(item, trained=False)
    ]
    return {
        "direct": direct,
        "seen_memory": seen_memory,
        "heldout_memory": heldout_memory,
        "seen_chain": seen_chain,
        "heldout_chain": heldout_chain,
        "schema_rule": schema_rule_examples(),
        "eval_all": direct + seen_memory + heldout_memory + seen_chain + heldout_chain,
    }


def condition_consolidation_examples(
    *,
    condition: str,
    raw_groups: dict[str, list[QAExample]],
) -> list[QAExample]:
    if condition == "facts_only":
        return []
    if condition == "seen_composition":
        return raw_groups["seen_memory"]
    if condition == "chain_schema":
        return raw_groups["seen_chain"] + raw_groups["schema_rule"]
    if condition == "memory_schema":
        return raw_groups["seen_memory"] + raw_groups["seen_chain"] + raw_groups["schema_rule"]
    if condition == "oracle_all_composition":
        return raw_groups["seen_memory"] + raw_groups["heldout_memory"] + raw_groups["seen_chain"] + raw_groups["heldout_chain"]
    raise ValueError(f"Unknown condition: {condition!r}.")


def encode_groups(
    raw_groups: dict[str, list[QAExample]],
    tokenizer: Tokenizer,
    *,
    max_seq_len: int,
) -> dict[str, list[EncodedExample]]:
    return {
        name: encode_examples(examples, tokenizer, max_seq_len=max_seq_len)
        for name, examples in raw_groups.items()
    }


def train_examples(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
    label: str,
) -> list[dict[str, float]]:
    positive_int("epochs", epochs)
    if not examples:
        raise ValueError(f"{label} received no training examples.")
    set_only_native_weights_trainable(model)
    optimizer = make_optimizer_for_model(args, trainable_weight_parameters(model))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(examples), generator=generator)
        loss_sum = 0.0
        batches = 0
        pbar = tqdm(range(0, len(examples), args.batch_size), desc=f"{label} {epoch}/{epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = masked_ce_loss(logits, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(model), args.grad_clip)
            optimizer.step()
            value = float(loss.detach().cpu())
            loss_sum += value
            batches += 1
            pbar.set_postfix({"ce": f"{value:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"{label} epoch {epoch} saw zero batches.")
        row = {"epoch": float(epoch), "loss": loss_sum / float(batches)}
        trace.append(row)
        if epoch == 1 or epoch == epochs or epoch % args.print_every == 0:
            print(f"{label} epoch={epoch:4d} loss={row['loss']:.5f}")
    return trace


def compact_metric(metrics: dict[str, Any], category: str) -> tuple[float, float, float]:
    row = metrics.get(category)
    if row is None:
        return 0.0, 0.0, 0.0
    return float(row["loss"]), float(row["token_accuracy"]), float(row["exact_match"])


def run_condition(
    *,
    args: argparse.Namespace,
    condition: str,
    checkpoint: dict[str, Any],
    encoded_groups: dict[str, list[EncodedExample]],
    raw_groups: dict[str, list[QAExample]],
    pad_id: int,
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    model = make_model_from_config(
        checkpoint=checkpoint,
        device=device,
        seed=args.seed + 1000 + condition_index,
    )
    traces: list[dict[str, Any]] = []
    fact_trace = train_examples(
        args=args,
        model=model,
        examples=encoded_groups["direct"],
        pad_id=pad_id,
        device=device,
        epochs=args.fact_epochs,
        seed=args.seed + 2000 + condition_index,
        label=f"{condition} facts",
    )
    traces.append({"stage": "direct_facts", "trace": fact_trace})

    consolidation_raw = condition_consolidation_examples(condition=condition, raw_groups=raw_groups)
    if consolidation_raw:
        tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
        max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
        consolidation = encode_examples(consolidation_raw, tokenizer, max_seq_len=max_seq_len)
        if args.replay_direct_during_consolidation:
            train_set = encoded_groups["direct"] + consolidation
        else:
            train_set = consolidation
        consolidation_trace = train_examples(
            args=args,
            model=model,
            examples=train_set,
            pad_id=pad_id,
            device=device,
            epochs=args.consolidation_epochs,
            seed=args.seed + 3000 + condition_index,
            label=f"{condition} consolidate",
        )
        traces.append(
            {
                "stage": "consolidation",
                "raw_example_count": len(consolidation_raw),
                "train_example_count": len(train_set),
                "trace": consolidation_trace,
            }
        )
    metrics = evaluate_examples(
        model,
        encoded_groups["eval_all"],
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    return {
        "condition": condition,
        "oracle_leakage": condition == "oracle_all_composition",
        "metrics": metrics,
        "traces": traces,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer does not exist: {args.tokenizer_path}")
    positive_int("fact_epochs", args.fact_epochs)
    positive_int("consolidation_epochs", args.consolidation_epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    positive_float("grad_clip", args.grad_clip)
    positive_int("print_every", args.print_every)
    parse_conditions(args.conditions)
    parse_people(args.composition_holdout_people, name="composition_holdout_people")


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
    holdout_people = parse_people(args.composition_holdout_people, name="composition_holdout_people")
    raw_groups = build_raw_groups(holdout_people=holdout_people)
    encoded_groups = encode_groups(raw_groups, tokenizer, max_seq_len=max_seq_len)
    conditions = parse_conditions(args.conditions)

    print("TINY CONSOLIDATION GENERALIZATION")
    print("=" * 112)
    print(
        f"device={device} conditions={conditions} holdout={sorted(holdout_people)} "
        f"direct={len(encoded_groups['direct'])} eval={len(encoded_groups['eval_all'])}"
    )
    print(
        "Question: after direct facts are known, does consolidation create reusable composition "
        "or only memorize seen bridges?"
    )

    results: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        print("\n" + "-" * 112)
        print(f"condition={condition}")
        results.append(
            run_condition(
                args=args,
                condition=condition,
                checkpoint=checkpoint,
                encoded_groups=encoded_groups,
                raw_groups=raw_groups,
                pad_id=pad_id,
                device=device,
                condition_index=condition_index,
            )
        )

    summary = {
        "question": "Can consolidation turn learned direct facts into held-out compositional generalization?",
        "holdout_people": sorted(holdout_people),
        "conditions": conditions,
        "model_config": checkpoint["model_config"],
        "hyperparameters": {
            "fact_epochs": args.fact_epochs,
            "consolidation_epochs": args.consolidation_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "replay_direct_during_consolidation": args.replay_direct_during_consolidation,
        },
        "raw_groups": {name: [asdict(example) for example in examples] for name, examples in raw_groups.items()},
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTINY CONSOLIDATION GENERALIZATION SUMMARY")
    print("=" * 112)
    print(
        "{:>24} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
            "condition",
            "direct",
            "seenMem",
            "holdMem",
            "seenCtx",
            "holdCtx",
            "overall",
        )
    )
    print(
        "{:>24} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
            "",
            "exact",
            "exact",
            "exact",
            "exact",
            "exact",
            "exact",
        )
    )
    for result in results:
        metrics = result["metrics"]
        direct_scores = [
            row["exact_match"]
            for key, row in metrics.items()
            if key.startswith("stage") and ("possession" in key or "object_place" in key)
        ]
        direct_exact = sum(direct_scores) / float(len(direct_scores)) if direct_scores else 0.0
        _loss, _tok, seen_memory = compact_metric(metrics, "composition_seen_memory")
        _loss, _tok, heldout_memory = compact_metric(metrics, "composition_heldout_memory")
        _loss, _tok, seen_chain = compact_metric(metrics, "composition_seen_chain")
        _loss, _tok, heldout_chain = compact_metric(metrics, "composition_heldout_chain")
        print(
            "{:>24} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f}".format(
                result["condition"],
                direct_exact,
                seen_memory,
                heldout_memory,
                seen_chain,
                heldout_chain,
                metrics["overall"]["exact_match"],
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
        default=Path("model/analysis/gco-tiny-consolidation-generalization-seed0.json"),
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="facts_only,seen_composition,chain_schema,memory_schema,oracle_all_composition",
    )
    parser.add_argument("--composition-holdout-people", type=str, default="Kira,Luca")
    parser.add_argument("--fact-epochs", type=int, default=300)
    parser.add_argument("--consolidation-epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--replay-direct-during-consolidation", action="store_true")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
