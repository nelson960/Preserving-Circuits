"""Sweep how much old output behavior is needed to find a CL path.

This diagnostic starts every run from the same 100-word fitted checkpoint,
trains on the next 100 words, and preserves old behavior using only a chosen
budget of old output probes. It then evaluates on the full old span, full new
span, and representation geometry against the 200-word-from-scratch reference.

The test is intentionally a diagnostic, not the final optimizer. It answers:

    How small can the old behavior signal be while still learning the new span,
    preserving old output behavior, and moving toward base200-like geometry?
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
from experiments.gco_math.gco_native_scratch_transformer import NativeGCOConfig
from experiments.gco_math.gco_prepare_tiny_cl_base import build_lm_windows, evaluate_model, load_chunks
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    collect_logits,
    distillation_kl,
    geometry_report,
    load_checkpoint,
    make_optimizer,
    mean_layer_metric,
    old_margin_loss,
    set_only_native_weights_trainable,
    target_margins_from_logits,
    trainable_weight_parameters,
)
from experiments.gco_math.gco_visualize_tiny_geometry_drift import build_windows, word_span


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


def parse_probe_counts(raw: str, total_old_windows: int) -> list[int]:
    positive_int("total_old_windows", total_old_windows)
    counts: list[int] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            raise ValueError(f"Empty item in --old-probe-counts={raw!r}.")
        if token == "all":
            count = total_old_windows
        else:
            try:
                count = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid probe count {item!r}; use integers or 'all'.") from exc
            nonnegative_int("old_probe_count", count)
            if count > total_old_windows:
                raise ValueError(
                    f"old_probe_count={count} exceeds available old windows={total_old_windows}. "
                    "Use 'all' for the full budget."
                )
        if count not in counts:
            counts.append(count)
    if not counts:
        raise ValueError("No old probe counts requested.")
    return counts


@torch.no_grad()
def per_window_teacher_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: logits={logits.shape}, targets={targets.shape}.")
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return token_losses.mean(dim=1)


def select_old_probe_indices(
    *,
    count: int,
    mode: str,
    teacher_logits: torch.Tensor,
    teacher_margins: torch.Tensor,
    old_targets: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    total = old_targets.shape[0]
    if teacher_logits.shape[0] != total or teacher_margins.shape[0] != total:
        raise ValueError(
            "Teacher tensors must have one row per old window: "
            f"logits={teacher_logits.shape[0]} margins={teacher_margins.shape[0]} targets={total}."
        )
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    if count == total:
        return torch.arange(total, dtype=torch.long)
    if count > total:
        raise ValueError(f"count={count} exceeds total old windows={total}.")
    if mode == "uniform":
        return torch.linspace(0, total - 1, steps=count).round().to(dtype=torch.long).unique(sorted=True)
    if mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randperm(total, generator=generator)[:count].sort().values
    if mode == "low-margin":
        scores = teacher_margins.mean(dim=1)
        return torch.topk(-scores, k=count).indices.sort().values
    if mode == "high-loss":
        scores = per_window_teacher_loss(teacher_logits, old_targets)
        return torch.topk(scores, k=count).indices.sort().values
    raise ValueError(f"Unknown --old-probe-selection={mode!r}.")


@torch.no_grad()
def output_kl_and_agreement(
    *,
    current_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    kl = distillation_kl(current_logits, teacher_logits, temperature=temperature)
    current_argmax = current_logits.argmax(dim=-1)
    teacher_argmax = teacher_logits.argmax(dim=-1)
    agreement = (current_argmax == teacher_argmax).to(torch.float32).mean()
    return {
        "kl": float(kl.detach().cpu()),
        "argmax_agreement": float(agreement.detach().cpu()),
    }


def validate_args(args: argparse.Namespace) -> None:
    for name in ["base_checkpoint", "target_checkpoint", "tokenizer_path", "chunks_path"]:
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    nonnegative_int("chunk_index", args.chunk_index)
    nonnegative_int("old_word_start", args.old_word_start)
    positive_int("old_word_count", args.old_word_count)
    nonnegative_int("new_word_start", args.new_word_start)
    positive_int("new_word_count", args.new_word_count)
    positive_int("stride", args.stride)
    positive_int("max_windows", args.max_windows)
    positive_int("geometry_max_windows", args.geometry_max_windows)
    positive_int("epochs", args.epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    nonnegative_float("lambda_kl", args.lambda_kl)
    nonnegative_float("lambda_margin", args.lambda_margin)
    positive_float("distill_temperature", args.distill_temperature)
    nonnegative_float("margin_slack", args.margin_slack)
    positive_float("grad_clip", args.grad_clip)
    positive_int("geometry_every", args.geometry_every)
    if args.old_word_start + args.old_word_count != args.new_word_start:
        raise ValueError(
            "This diagnostic expects adjacent old and new spans so the target reference is old+new: "
            f"old=[{args.old_word_start},{args.old_word_start + args.old_word_count}) "
            f"new_start={args.new_word_start}."
        )


def build_data(args: argparse.Namespace, seq_len: int) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.chunk_index >= len(chunks):
        raise ValueError(f"chunk_index={args.chunk_index} outside chunk count {len(chunks)}.")
    text = str(chunks[args.chunk_index]["text"])
    old_text = word_span(text, args.old_word_start, args.old_word_count)
    new_text = word_span(text, args.new_word_start, args.new_word_count)
    full_text = word_span(text, args.old_word_start, args.old_word_count + args.new_word_count)
    old_token_ids = tokenizer.encode(old_text).ids
    new_token_ids = tokenizer.encode(new_text).ids
    full_token_ids = tokenizer.encode(full_text).ids
    old_inputs, old_targets = build_lm_windows(
        old_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows
    )
    new_inputs, new_targets = build_lm_windows(
        new_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows
    )
    full_inputs, full_targets = build_lm_windows(
        full_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows
    )
    full_geometry_windows, _positions = build_windows(
        full_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.geometry_max_windows
    )
    return {
        "chunks": chunks,
        "old_text": old_text,
        "new_text": new_text,
        "old_inputs": old_inputs,
        "old_targets": old_targets,
        "new_inputs": new_inputs,
        "new_targets": new_targets,
        "full_inputs": full_inputs,
        "full_targets": full_targets,
        "full_geometry_windows": full_geometry_windows,
        "seq_len": seq_len,
    }


def train_for_probe_budget(
    *,
    args: argparse.Namespace,
    probe_count: int,
    probe_indices: torch.Tensor,
    base_checkpoint: dict[str, Any],
    data: dict[str, Any],
    teacher_old_logits: torch.Tensor,
    teacher_old_margins: torch.Tensor,
    base100_logits: dict[str, torch.Tensor],
    base200_logits: dict[str, torch.Tensor],
    base100_model: torch.nn.Module,
    base200_model: torch.nn.Module,
    initial_geometry: dict[str, dict[str, float]],
    device: torch.device,
) -> dict[str, Any]:
    current, _checkpoint = load_checkpoint(args.base_checkpoint, device)
    set_only_native_weights_trainable(current)
    optimizer = make_optimizer(args, trainable_weight_parameters(current))

    old_inputs = data["old_inputs"]
    old_targets = data["old_targets"]
    new_inputs = data["new_inputs"]
    new_targets = data["new_targets"]
    full_inputs = data["full_inputs"]
    full_targets = data["full_targets"]
    full_geometry_windows = data["full_geometry_windows"]

    selected_old_inputs = old_inputs[probe_indices] if probe_count > 0 else old_inputs[:0]
    selected_old_targets = old_targets[probe_indices] if probe_count > 0 else old_targets[:0]
    selected_teacher_logits = teacher_old_logits[probe_indices] if probe_count > 0 else teacher_old_logits[:0]
    selected_teacher_margins = teacher_old_margins[probe_indices] if probe_count > 0 else teacher_old_margins[:0]

    old_initial = evaluate_model(current, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    new_initial = evaluate_model(current, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)

    print("\n" + "-" * 112)
    print(f"budget={probe_count}/{old_inputs.shape[0]} selection={args.old_probe_selection}")
    print(
        "initial old_loss={:.6f} old_acc={:.4f} new_loss={:.6f} new_acc={:.4f}".format(
            old_initial["loss"],
            old_initial["token_accuracy"],
            new_initial["loss"],
            new_initial["token_accuracy"],
        )
    )

    trace: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + probe_count * 1009)
    for epoch in range(1, args.epochs + 1):
        current.train()
        permutation = torch.randperm(new_inputs.shape[0], generator=generator)
        epoch_total = 0.0
        epoch_new = 0.0
        epoch_kl = 0.0
        epoch_margin = 0.0
        batches = 0
        pbar = tqdm(
            range(0, new_inputs.shape[0], args.batch_size),
            desc=f"budget {probe_count} epoch {epoch}/{args.epochs}",
        )
        for start in pbar:
            new_indices = permutation[start : start + args.batch_size]
            if new_indices.numel() <= 0:
                raise RuntimeError(f"Empty new batch at start={start}.")
            batch_new_inputs = new_inputs[new_indices].to(device)
            batch_new_targets = new_targets[new_indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            new_logits = current(batch_new_inputs)
            new_ce = F.cross_entropy(new_logits.reshape(-1, new_logits.shape[-1]), batch_new_targets.reshape(-1))
            if probe_count > 0:
                old_batch_indices = torch.randint(
                    low=0,
                    high=probe_count,
                    size=(int(new_indices.numel()),),
                    generator=generator,
                    device=torch.device("cpu"),
                )
                batch_old_inputs = selected_old_inputs[old_batch_indices].to(device)
                batch_old_targets = selected_old_targets[old_batch_indices].to(device)
                batch_teacher_logits = selected_teacher_logits[old_batch_indices].to(device)
                batch_teacher_margins = selected_teacher_margins[old_batch_indices].to(device)
                old_logits = current(batch_old_inputs)
                kl = distillation_kl(old_logits, batch_teacher_logits, temperature=args.distill_temperature)
                margin = old_margin_loss(
                    old_logits,
                    batch_old_targets,
                    batch_teacher_margins,
                    margin_slack=args.margin_slack,
                )
            else:
                kl = new_ce.new_zeros(())
                margin = new_ce.new_zeros(())
            loss = new_ce + args.lambda_kl * kl + args.lambda_margin * margin
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(current), args.grad_clip)
            optimizer.step()

            total_value = float(loss.detach().cpu())
            new_value = float(new_ce.detach().cpu())
            kl_value = float(kl.detach().cpu())
            margin_value = float(margin.detach().cpu())
            epoch_total += total_value
            epoch_new += new_value
            epoch_kl += kl_value
            epoch_margin += margin_value
            batches += 1
            pbar.set_postfix(
                {
                    "new": f"{new_value:.3f}",
                    "kl": f"{kl_value:.3g}",
                    "m": f"{margin_value:.3g}",
                    "tot": f"{total_value:.3f}",
                }
            )
        if batches <= 0:
            raise RuntimeError(f"Budget {probe_count} epoch {epoch} saw zero batches.")

        old_eval = evaluate_model(current, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
        new_eval = evaluate_model(current, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "mean_total_loss": epoch_total / float(batches),
            "mean_new_ce": epoch_new / float(batches),
            "mean_old_kl": epoch_kl / float(batches),
            "mean_old_margin_loss": epoch_margin / float(batches),
            "old_loss": old_eval["loss"],
            "old_token_accuracy": old_eval["token_accuracy"],
            "old_target_margin_mean": old_eval["target_margin_mean"],
            "new_loss": new_eval["loss"],
            "new_token_accuracy": new_eval["token_accuracy"],
            "new_target_margin_mean": new_eval["target_margin_mean"],
        }
        if epoch == 1 or epoch == args.epochs or epoch % args.geometry_every == 0:
            epoch_geometry = geometry_report(
                current=current,
                base100=base100_model,
                base200=base200_model,
                windows=full_geometry_windows,
                device=device,
            )
            row["geometry_mean"] = {
                "current_to_base100_drift_rel": mean_layer_metric(epoch_geometry, "current_to_base100_drift_rel"),
                "current_to_base100_cka": mean_layer_metric(epoch_geometry, "current_to_base100_cka"),
                "current_to_base200_drift_rel": mean_layer_metric(epoch_geometry, "current_to_base200_drift_rel"),
                "current_to_base200_cka": mean_layer_metric(epoch_geometry, "current_to_base200_cka"),
                "base200_closeness_gain": mean_layer_metric(epoch_geometry, "base200_closeness_gain"),
                "base200_cka_gain": mean_layer_metric(epoch_geometry, "base200_cka_gain"),
            }
        trace.append(row)
        geometry_text = ""
        if "geometry_mean" in row:
            geometry_text = (
                " geom_to200_rel={:.4f} cka={:.4f} gain={:+.4f}".format(
                    row["geometry_mean"]["current_to_base200_drift_rel"],
                    row["geometry_mean"]["current_to_base200_cka"],
                    row["geometry_mean"]["base200_closeness_gain"],
                )
            )
        print(
            "budget={:4d} epoch={:4d} total={:.5f} new={:.5f} kl={:.5f} margin={:.5f} "
            "old_loss={:.6f} old_acc={:.4f} new_loss={:.6f} new_acc={:.4f}{}".format(
                probe_count,
                epoch,
                row["mean_total_loss"],
                row["mean_new_ce"],
                row["mean_old_kl"],
                row["mean_old_margin_loss"],
                old_eval["loss"],
                old_eval["token_accuracy"],
                new_eval["loss"],
                new_eval["token_accuracy"],
                geometry_text,
            )
        )

    final_old = evaluate_model(current, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    final_new = evaluate_model(current, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
    final_full = evaluate_model(current, full_inputs, full_targets, batch_size=args.eval_batch_size, device=device)
    final_geometry = geometry_report(
        current=current,
        base100=base100_model,
        base200=base200_model,
        windows=full_geometry_windows,
        device=device,
    )
    final_geometry_mean = {
        "current_to_base100_drift_rel": mean_layer_metric(final_geometry, "current_to_base100_drift_rel"),
        "current_to_base100_cka": mean_layer_metric(final_geometry, "current_to_base100_cka"),
        "current_to_base200_drift_rel": mean_layer_metric(final_geometry, "current_to_base200_drift_rel"),
        "current_to_base200_cka": mean_layer_metric(final_geometry, "current_to_base200_cka"),
        "base100_to_base200_drift_rel": mean_layer_metric(final_geometry, "base100_to_base200_drift_rel"),
        "base100_to_base200_cka": mean_layer_metric(final_geometry, "base100_to_base200_cka"),
        "base200_closeness_gain": mean_layer_metric(final_geometry, "base200_closeness_gain"),
        "base200_cka_gain": mean_layer_metric(final_geometry, "base200_cka_gain"),
    }
    current_old_logits = collect_logits(current, old_inputs, batch_size=args.eval_batch_size, device=device)
    current_new_logits = collect_logits(current, new_inputs, batch_size=args.eval_batch_size, device=device)
    current_full_logits = collect_logits(current, full_inputs, batch_size=args.eval_batch_size, device=device)
    behavior_match = {
        "old_to_base100": output_kl_and_agreement(
            current_logits=current_old_logits,
            teacher_logits=base100_logits["old"],
            temperature=args.distill_temperature,
        ),
        "old_to_base200": output_kl_and_agreement(
            current_logits=current_old_logits,
            teacher_logits=base200_logits["old"],
            temperature=args.distill_temperature,
        ),
        "new_to_base200": output_kl_and_agreement(
            current_logits=current_new_logits,
            teacher_logits=base200_logits["new"],
            temperature=args.distill_temperature,
        ),
        "full_to_base200": output_kl_and_agreement(
            current_logits=current_full_logits,
            teacher_logits=base200_logits["full"],
            temperature=args.distill_temperature,
        ),
    }

    checkpoint_path = args.checkpoint_dir / f"behavior-budget-{probe_count}-seed{args.seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "base_checkpoint": str(args.base_checkpoint),
            "target_checkpoint": str(args.target_checkpoint),
            "old_probe_count": probe_count,
            "old_probe_indices": probe_indices.tolist(),
            "model_state_dict": current.state_dict(),
            "native_gco_config": asdict(NativeGCOConfig(**base_checkpoint["native_gco_config"])),
            "model_config": base_checkpoint["model_config"],
            "old_final_metrics": final_old,
            "new_final_metrics": final_new,
            "full_final_metrics": final_full,
            "final_geometry_mean": final_geometry_mean,
            "behavior_match": behavior_match,
        },
        checkpoint_path,
    )
    result = {
        "old_probe_count": probe_count,
        "old_probe_fraction": float(probe_count) / float(old_inputs.shape[0]),
        "old_probe_selection": args.old_probe_selection,
        "old_probe_indices": probe_indices.tolist(),
        "checkpoint_path": str(checkpoint_path),
        "old_initial": old_initial,
        "new_initial": new_initial,
        "old_final": final_old,
        "new_final": final_new,
        "full_final": final_full,
        "behavior_match": behavior_match,
        "initial_geometry_mean": {
            "current_to_base200_drift_rel": mean_layer_metric(initial_geometry, "current_to_base200_drift_rel"),
            "current_to_base200_cka": mean_layer_metric(initial_geometry, "current_to_base200_cka"),
            "base200_closeness_gain": mean_layer_metric(initial_geometry, "base200_closeness_gain"),
            "base200_cka_gain": mean_layer_metric(initial_geometry, "base200_cka_gain"),
        },
        "final_geometry": final_geometry,
        "final_geometry_mean": final_geometry_mean,
        "trace": trace,
    }
    print(
        "budget={:4d} final old_loss {:.6f}->{:.6f} old_acc {:.4f}->{:.4f} "
        "new_loss {:.6f}->{:.6f} new_acc {:.4f}->{:.4f} "
        "to200_rel {:.4f}->{:.4f} cka {:.4f}->{:.4f} "
        "oldKL={:.5f} new200KL={:.5f} wrote={}".format(
            probe_count,
            old_initial["loss"],
            final_old["loss"],
            old_initial["token_accuracy"],
            final_old["token_accuracy"],
            new_initial["loss"],
            final_new["loss"],
            new_initial["token_accuracy"],
            final_new["token_accuracy"],
            mean_layer_metric(initial_geometry, "current_to_base200_drift_rel"),
            final_geometry_mean["current_to_base200_drift_rel"],
            mean_layer_metric(initial_geometry, "current_to_base200_cka"),
            final_geometry_mean["current_to_base200_cka"],
            behavior_match["old_to_base100"]["kl"],
            behavior_match["new_to_base200"]["kl"],
            checkpoint_path,
        )
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    base100_model, base_checkpoint = load_checkpoint(args.base_checkpoint, device)
    base200_model, target_checkpoint = load_checkpoint(args.target_checkpoint, device)
    if base_checkpoint["model_config"] != target_checkpoint["model_config"]:
        raise ValueError(
            "Base and target checkpoint specs differ. "
            f"base={base_checkpoint['model_config']} target={target_checkpoint['model_config']}"
        )
    for model in [base100_model, base200_model]:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    seq_len = int(base_checkpoint["model_config"]["max_seq_len"])
    data = build_data(args, seq_len)
    old_inputs = data["old_inputs"]
    old_targets = data["old_targets"]
    new_inputs = data["new_inputs"]
    new_targets = data["new_targets"]
    full_inputs = data["full_inputs"]
    full_targets = data["full_targets"]
    full_geometry_windows = data["full_geometry_windows"]

    teacher_old_logits = collect_logits(base100_model, old_inputs, batch_size=args.eval_batch_size, device=device)
    teacher_old_margins = target_margins_from_logits(teacher_old_logits, old_targets)
    base100_logits = {
        "old": teacher_old_logits,
    }
    base200_logits = {
        "old": collect_logits(base200_model, old_inputs, batch_size=args.eval_batch_size, device=device),
        "new": collect_logits(base200_model, new_inputs, batch_size=args.eval_batch_size, device=device),
        "full": collect_logits(base200_model, full_inputs, batch_size=args.eval_batch_size, device=device),
    }
    target_old = evaluate_model(base200_model, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    target_new = evaluate_model(base200_model, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
    target_full = evaluate_model(base200_model, full_inputs, full_targets, batch_size=args.eval_batch_size, device=device)

    current_for_geometry, _checkpoint = load_checkpoint(args.base_checkpoint, device)
    current_for_geometry.eval()
    initial_geometry = geometry_report(
        current=current_for_geometry,
        base100=base100_model,
        base200=base200_model,
        windows=full_geometry_windows,
        device=device,
    )
    del current_for_geometry

    counts = parse_probe_counts(args.old_probe_counts, int(old_inputs.shape[0]))
    print("TINY BEHAVIOR SIGNAL BUDGET SWEEP")
    print("=" * 112)
    print("Each run starts from base100. Old signal is output KL + margin preservation on the selected probe budget.")
    print(
        f"device={device} optimizer={args.optimizer} lr={args.lr:g} epochs={args.epochs} "
        f"counts={counts} selection={args.old_probe_selection}"
    )
    print(
        "windows old={} new={} full={} geometry={} seq_len={} target200 old_loss={:.6f} new_loss={:.6f} full_loss={:.6f}".format(
            old_inputs.shape[0],
            new_inputs.shape[0],
            full_inputs.shape[0],
            full_geometry_windows.shape[0],
            seq_len,
            target_old["loss"],
            target_new["loss"],
            target_full["loss"],
        )
    )
    print(
        "initial geometry to base200 rel={:.4f} cka={:.4f}".format(
            mean_layer_metric(initial_geometry, "current_to_base200_drift_rel"),
            mean_layer_metric(initial_geometry, "current_to_base200_cka"),
        )
    )

    results: list[dict[str, Any]] = []
    for count in counts:
        probe_indices = select_old_probe_indices(
            count=count,
            mode=args.old_probe_selection,
            teacher_logits=teacher_old_logits,
            teacher_margins=teacher_old_margins,
            old_targets=old_targets,
            seed=args.seed + count * 17,
        )
        if probe_indices.numel() != count:
            raise RuntimeError(
                f"Probe selector returned {probe_indices.numel()} indices for requested count={count}."
            )
        result = train_for_probe_budget(
            args=args,
            probe_count=count,
            probe_indices=probe_indices,
            base_checkpoint=base_checkpoint,
            data=data,
            teacher_old_logits=teacher_old_logits,
            teacher_old_margins=teacher_old_margins,
            base100_logits=base100_logits,
            base200_logits=base200_logits,
            base100_model=base100_model,
            base200_model=base200_model,
            initial_geometry=initial_geometry,
            device=device,
        )
        results.append(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "question": "How little old output behavior signal is needed to find a base100->base200-like CL path?",
        "constraint": "No hidden-state anchor loss. Each budget preserves old behavior through output KL and margin loss only.",
        "base_checkpoint": str(args.base_checkpoint),
        "target_checkpoint": str(args.target_checkpoint),
        "model_config": base_checkpoint["model_config"],
        "source": {
            "chunks_path": str(args.chunks_path),
            "chunk_index": args.chunk_index,
            "chunk_id": str(data["chunks"][args.chunk_index]["chunk_id"]),
            "old_word_start": args.old_word_start,
            "old_word_count": args.old_word_count,
            "new_word_start": args.new_word_start,
            "new_word_count": args.new_word_count,
            "old_window_count": int(old_inputs.shape[0]),
            "new_window_count": int(new_inputs.shape[0]),
            "full_window_count": int(full_inputs.shape[0]),
            "geometry_window_count": int(full_geometry_windows.shape[0]),
            "seq_len": seq_len,
            "stride": args.stride,
        },
        "optimizer": {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "grad_clip": args.grad_clip,
        },
        "loss_weights": {
            "lambda_kl": args.lambda_kl,
            "lambda_margin": args.lambda_margin,
            "distill_temperature": args.distill_temperature,
            "margin_slack": args.margin_slack,
        },
        "target200_old": target_old,
        "target200_new": target_new,
        "target200_full": target_full,
        "initial_geometry": initial_geometry,
        "results": results,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTINY BEHAVIOR SIGNAL BUDGET SWEEP SUMMARY")
    print("=" * 112)
    print(
        "{:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
            "oldN",
            "old_loss",
            "old_acc",
            "new_loss",
            "new_acc",
            "to200",
            "cka200",
            "oldKL",
            "newKL",
            "fullKL",
        )
    )
    for result in results:
        print(
            "{:8d} {:9.5f} {:9.4f} {:9.5f} {:9.4f} {:9.4f} {:9.4f} {:9.5f} {:9.5f} {:9.5f}".format(
                result["old_probe_count"],
                result["old_final"]["loss"],
                result["old_final"]["token_accuracy"],
                result["new_final"]["loss"],
                result["new_final"]["token_accuracy"],
                result["final_geometry_mean"]["current_to_base200_drift_rel"],
                result["final_geometry_mean"]["current_to_base200_cka"],
                result["behavior_match"]["old_to_base100"]["kl"],
                result["behavior_match"]["new_to_base200"]["kl"],
                result["behavior_match"]["full_to_base200"]["kl"],
            )
        )
    print(f"wrote_json={args.output_json}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        default=Path("model/checkpoints/gco-tiny-cl-base-200w-samespec-seed0.pt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("model/analysis/gco-tiny-cl-behavior-budget-sweep-100to200-seed0.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("model/checkpoints/gco-tiny-cl-behavior-budget-sweep-100to200-seed0"),
    )
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--old-word-start", type=int, default=0)
    parser.add_argument("--old-word-count", type=int, default=100)
    parser.add_argument("--new-word-start", type=int, default=100)
    parser.add_argument("--new-word-count", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--geometry-max-windows", type=int, default=128)
    parser.add_argument("--old-probe-counts", type=str, default="0,1,2,4,8,16,32,64,all")
    parser.add_argument(
        "--old-probe-selection",
        choices=["uniform", "random", "low-margin", "high-loss"],
        default="uniform",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-margin", type=float, default=0.1)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--margin-slack", type=float, default=0.5)
    parser.add_argument("--geometry-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
