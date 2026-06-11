"""Controlled preserve/drop continual-learning diagnostic.

The experiment explicitly labels old behaviors as:

    preserve: must remain stable during the next learning step
    drop: not protected; optionally actively suppressed
    new: must be learned

This separates "old behavior" into behavior we care about and behavior we are
allowed to forget. The first run should usually use no active drop suppression
so we can see whether non-preserved behavior drifts naturally. A second run can
enable drop suppression to test deliberate forgetting.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
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
from experiments.gco_math.gco_tiny_cl_behavior_path import load_checkpoint, set_only_native_weights_trainable, trainable_weight_parameters
from experiments.gco_math.gco_tiny_reusable_computation_cl import (
    EncodedExample,
    batch_examples,
    collect_example_logits,
    distillation_loss_for_examples,
    encode_examples,
    evaluate_examples,
    make_model_from_config,
    make_optimizer_for_model,
    masked_ce_loss,
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


def parse_names(raw: str, *, field_name: str) -> set[str]:
    names = {item.strip() for item in raw.split(",") if item.strip()}
    if not names:
        raise ValueError(f"--{field_name.replace('_', '-')} must contain at least one name.")
    return names


def build_stage_groups(
    *,
    preserve_people: set[str],
    drop_people: set[str],
) -> dict[str, list[Any]]:
    items = relation_items()
    known_people = {item.person for item in items}
    unknown = preserve_people.union(drop_people).difference(known_people)
    if unknown:
        raise ValueError(f"Unknown people in preserve/drop sets: {sorted(unknown)}.")
    overlap = preserve_people.intersection(drop_people)
    if overlap:
        raise ValueError(f"People cannot be both preserve and drop: {sorted(overlap)}.")
    stage1_items = [item for item in items if item.stage == 1]
    stage2_items = [item for item in items if item.stage == 2]
    preserve_examples = [
        example
        for item in stage1_items
        if item.person in preserve_people
        for example in possession_examples(item)
    ]
    drop_examples = [
        example
        for item in stage1_items
        if item.person in drop_people
        for example in possession_examples(item)
    ]
    neutral_examples = [
        example
        for item in stage1_items
        if item.person not in preserve_people and item.person not in drop_people
        for example in possession_examples(item)
    ]
    if not preserve_examples:
        raise ValueError("Preserve set produced no examples.")
    if not drop_examples:
        raise ValueError("Drop set produced no examples.")
    stage1_examples = [example for item in stage1_items for example in possession_examples(item)]
    stage2_examples = [example for item in stage2_items for example in possession_examples(item)]
    return {
        "stage1": stage1_examples,
        "stage2": stage2_examples,
        "preserve": preserve_examples,
        "drop": drop_examples,
        "neutral": neutral_examples,
    }


def train_core_on_examples(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    train_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    epochs: int,
    seed: int,
    preserve_examples: list[EncodedExample] | None = None,
    preserve_logits: list[torch.Tensor] | None = None,
    neutral_guard_examples: list[EncodedExample] | None = None,
    neutral_guard_logits: list[torch.Tensor] | None = None,
    drop_examples: list[EncodedExample] | None = None,
    drop_weight: float = 0.0,
) -> list[dict[str, float]]:
    set_only_native_weights_trainable(model)
    optimizer = make_optimizer_for_model(args, trainable_weight_parameters(model))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    preserve_count = 0 if preserve_examples is None else len(preserve_examples)
    neutral_guard_count = 0 if neutral_guard_examples is None else len(neutral_guard_examples)
    drop_count = 0 if drop_examples is None else len(drop_examples)
    if (preserve_examples is None) != (preserve_logits is None):
        raise ValueError("preserve_examples and preserve_logits must both be set or both be None.")
    if (neutral_guard_examples is None) != (neutral_guard_logits is None):
        raise ValueError("neutral_guard_examples and neutral_guard_logits must both be set or both be None.")
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_examples), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0, "preserve": 0.0, "neutral_guard": 0.0, "drop": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(train_examples), args.batch_size), desc=f"train epoch {epoch}/{epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(train_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            ce = masked_ce_loss(logits, targets, mask)
            if preserve_count > 0 and preserve_examples is not None and preserve_logits is not None:
                preserve_indices = torch.randint(
                    low=0,
                    high=preserve_count,
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
            if neutral_guard_count > 0 and neutral_guard_examples is not None and neutral_guard_logits is not None:
                neutral_guard_indices = torch.randint(
                    low=0,
                    high=neutral_guard_count,
                    size=(int(indices.numel()),),
                    generator=generator,
                    device=torch.device("cpu"),
                )
                neutral_guard_inputs, _neutral_guard_targets, _neutral_guard_mask, neutral_guard_selected = batch_examples(
                    neutral_guard_examples,
                    indices=neutral_guard_indices,
                    pad_id=pad_id,
                    device=device,
                )
                neutral_guard_current = model(neutral_guard_inputs)
                neutral_guard_loss = distillation_loss_for_examples(
                    neutral_guard_current,
                    neutral_guard_selected,
                    neutral_guard_logits,
                    neutral_guard_indices,
                    temperature=args.distill_temperature,
                    device=device,
                )
            else:
                neutral_guard_loss = ce.new_zeros(())
            if drop_weight > 0.0:
                if drop_count <= 0 or drop_examples is None:
                    raise RuntimeError("drop_weight > 0 requires drop_examples.")
                drop_indices = torch.randint(
                    low=0,
                    high=drop_count,
                    size=(int(indices.numel()),),
                    generator=generator,
                    device=torch.device("cpu"),
                )
                drop_inputs, drop_targets, drop_mask, _drop_selected = batch_examples(
                    drop_examples,
                    indices=drop_indices,
                    pad_id=pad_id,
                    device=device,
                )
                drop_logits = model(drop_inputs)
                log_probs = F.log_softmax(drop_logits, dim=-1)
                old_answer_log_probs = log_probs.gather(-1, drop_targets.unsqueeze(-1)).squeeze(-1)
                threshold = math.log(args.drop_target_probability)
                drop_loss = F.relu(old_answer_log_probs - threshold).square()
                drop_loss = (drop_loss * drop_mask).sum() / drop_mask.sum().clamp_min(1.0)
            else:
                drop_loss = ce.new_zeros(())
            loss = (
                ce
                + args.lambda_preserve * preserve_loss
                + args.lambda_neutral_guard * neutral_guard_loss
                + drop_weight * drop_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(model), args.grad_clip)
            optimizer.step()
            values = {
                "loss": float(loss.detach().cpu()),
                "ce": float(ce.detach().cpu()),
                "preserve": float(preserve_loss.detach().cpu()),
                "neutral_guard": float(neutral_guard_loss.detach().cpu()),
                "drop": float(drop_loss.detach().cpu()),
            }
            for key, value in values.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix(
                {
                    "ce": f"{values['ce']:.3g}",
                    "p": f"{values['preserve']:.3g}",
                    "n": f"{values['neutral_guard']:.3g}",
                    "d": f"{values['drop']:.3g}",
                }
            )
        if batches <= 0:
            raise RuntimeError(f"Epoch {epoch} saw zero batches.")
        row = {key: value / float(batches) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        trace.append(row)
        print(
            "epoch={:4d} loss={:.5f} ce={:.5f} preserve={:.5f} neutral_guard={:.5f} drop={:.5f}".format(
                epoch,
                row["loss"],
                row["ce"],
                row["preserve"],
                row["neutral_guard"],
                row["drop"],
            )
        )
    return trace


def evaluate_groups(
    *,
    model: nn.Module,
    groups: dict[str, list[EncodedExample]],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, examples in groups.items():
        if examples:
            metrics = evaluate_examples(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)
            result[name] = metrics["overall"]
    return result


def run_method(
    *,
    args: argparse.Namespace,
    method: str,
    stage1_state: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    encoded_groups: dict[str, list[EncodedExample]],
    preserve_logits: list[torch.Tensor],
    neutral_guard_logits: list[torch.Tensor],
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if method not in {"naive", "preserve", "preserve_suppress", "preserve_guard", "preserve_guard_suppress"}:
        raise ValueError(f"Unknown method: {method!r}.")
    model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed + 900)
    missing, unexpected = model.load_state_dict(stage1_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Stage1 state load mismatch: missing={missing}, unexpected={unexpected}.")
    uses_preserve = method in {"preserve", "preserve_suppress", "preserve_guard", "preserve_guard_suppress"}
    uses_neutral_guard = method in {"preserve_guard", "preserve_guard_suppress"}
    preserve_examples = encoded_groups["preserve"] if uses_preserve else None
    teacher_logits = preserve_logits if uses_preserve else None
    neutral_guard_examples = encoded_groups["neutral"] if uses_neutral_guard else None
    neutral_teacher_logits = neutral_guard_logits if uses_neutral_guard else None
    drop_weight = args.drop_loss_weight if method in {"preserve_suppress", "preserve_guard_suppress"} else 0.0
    trace = train_core_on_examples(
        args=args,
        model=model,
        train_examples=encoded_groups["stage2"],
        pad_id=pad_id,
        device=device,
        epochs=args.stage2_epochs,
        seed=args.seed + 1000,
        preserve_examples=preserve_examples,
        preserve_logits=teacher_logits,
        neutral_guard_examples=neutral_guard_examples,
        neutral_guard_logits=neutral_teacher_logits,
        drop_examples=encoded_groups["drop"],
        drop_weight=drop_weight,
    )
    metrics = evaluate_groups(
        model=model,
        groups=encoded_groups,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    return {"method": method, "trace": trace, "metrics": metrics}


def validate_args(args: argparse.Namespace) -> None:
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer does not exist: {args.tokenizer_path}")
    positive_int("stage1_epochs", args.stage1_epochs)
    positive_int("stage2_epochs", args.stage2_epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    nonnegative_float("lambda_preserve", args.lambda_preserve)
    nonnegative_float("drop_loss_weight", args.drop_loss_weight)
    nonnegative_float("lambda_neutral_guard", args.lambda_neutral_guard)
    positive_float("drop_target_probability", args.drop_target_probability)
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")
    positive_float("distill_temperature", args.distill_temperature)
    positive_float("grad_clip", args.grad_clip)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer must define [PAD].")
    preserve_people = parse_names(args.preserve_people, field_name="preserve_people")
    drop_people = parse_names(args.drop_people, field_name="drop_people")
    raw_groups = build_stage_groups(preserve_people=preserve_people, drop_people=drop_people)
    _config_model, checkpoint = load_checkpoint(args.config_checkpoint, device)
    max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    encoded_groups = {
        name: encode_examples(examples, tokenizer, max_seq_len=max_seq_len)
        for name, examples in raw_groups.items()
    }
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")

    print("TINY CONTROLLED PRESERVE/DROP CL")
    print("=" * 112)
    print(
        f"device={device} preserve={sorted(preserve_people)} drop={sorted(drop_people)} "
        f"methods={methods} stage1={len(encoded_groups['stage1'])} stage2={len(encoded_groups['stage2'])}"
    )

    stage1_model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed + 77)
    stage1_trace = train_core_on_examples(
        args=args,
        model=stage1_model,
        train_examples=encoded_groups["stage1"],
        pad_id=pad_id,
        device=device,
        epochs=args.stage1_epochs,
        seed=args.seed + 88,
    )
    stage1_metrics = evaluate_groups(
        model=stage1_model,
        groups=encoded_groups,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    preserve_logits = collect_example_logits(
        stage1_model,
        encoded_groups["preserve"],
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    neutral_guard_logits = collect_example_logits(
        stage1_model,
        encoded_groups["neutral"],
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    stage1_state = {key: value.detach().cpu().clone() for key, value in stage1_model.state_dict().items()}

    results: list[dict[str, Any]] = []
    for method in methods:
        print("\n" + "-" * 112)
        print(f"method={method}")
        results.append(
            run_method(
                args=args,
                method=method,
                stage1_state=stage1_state,
                checkpoint=checkpoint,
                encoded_groups=encoded_groups,
                preserve_logits=preserve_logits,
                neutral_guard_logits=neutral_guard_logits,
                pad_id=pad_id,
                device=device,
            )
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "question": "Can CL explicitly preserve selected old behavior while allowing selected old behavior to drop?",
        "preserve_people": sorted(preserve_people),
        "drop_people": sorted(drop_people),
        "methods": methods,
        "model_config": checkpoint["model_config"],
        "hyperparameters": {
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_preserve": args.lambda_preserve,
            "lambda_neutral_guard": args.lambda_neutral_guard,
            "drop_loss_weight": args.drop_loss_weight,
            "drop_target_probability": args.drop_target_probability,
            "distill_temperature": args.distill_temperature,
        },
        "raw_groups": {
            name: [example.text for example in examples]
            for name, examples in raw_groups.items()
        },
        "stage1_trace": stage1_trace,
        "stage1_metrics": stage1_metrics,
        "results": results,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTINY CONTROLLED PRESERVE/DROP SUMMARY")
    print("=" * 112)
    print(
        "{:>18} {:>16} {:>16} {:>16} {:>16}".format(
            "method",
            "preserve",
            "drop",
            "new",
            "neutral",
        )
    )
    print(
        "{:>18} {:>16} {:>16} {:>16} {:>16}".format(
            "",
            "loss/tok/exact",
            "loss/tok/exact",
            "loss/tok/exact",
            "loss/tok/exact",
        )
    )
    for result in results:
        metrics = result["metrics"]

        def compact(group: str) -> str:
            if group not in metrics:
                return "n/a"
            row = metrics[group]
            return "{:.3g}/{:.3f}/{:.3f}".format(
                row["loss"],
                row["token_accuracy"],
                row["exact_match"],
            )

        print(
            "{:>18} {:>16} {:>16} {:>16} {:>16}".format(
                result["method"],
                compact("preserve"),
                compact("drop"),
                compact("stage2"),
                compact("neutral"),
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
        default=Path("model/analysis/gco-tiny-controlled-preserve-drop-seed0.json"),
    )
    parser.add_argument("--methods", type=str, default="naive,preserve,preserve_suppress,preserve_guard_suppress")
    parser.add_argument("--preserve-people", type=str, default="Alice,Bruno")
    parser.add_argument("--drop-people", type=str, default="Clara,Darin")
    parser.add_argument("--stage1-epochs", type=int, default=300)
    parser.add_argument("--stage2-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-neutral-guard", type=float, default=1.0)
    parser.add_argument("--drop-loss-weight", type=float, default=0.1)
    parser.add_argument("--drop-target-probability", type=float, default=0.05)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
