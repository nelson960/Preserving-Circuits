"""Test whether a behavior-preserving path exists from base100 toward base200.

This diagnostic intentionally does not anchor hidden states. It starts from a
100-word fitted tiny transformer, trains on the next 100 words, and preserves
old behavior only through output-level teacher constraints:

    L = L_new_ce + lambda_kl * KL(base100_old || current_old)
        + lambda_margin * old_margin_preservation

The experiment then measures whether the updated model moves closer to the
from-scratch 200-word model in representation geometry while old behavior
remains intact.
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
from experiments.gco_math.gco_native_scratch_transformer import GCONativeTransformer, NativeGCOConfig
from experiments.gco_math.gco_prepare_tiny_cl_base import build_lm_windows, evaluate_model, load_chunks
from experiments.gco_math.gco_visualize_tiny_geometry_drift import (
    build_windows,
    collect_states,
    linear_cka,
    procrustes_align,
    word_span,
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


def bounded_float(name: str, value: float, low: float, high: float) -> None:
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}.")


def load_checkpoint(path: Path, device: torch.device) -> tuple[GCONativeTransformer, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state_dict", "native_gco_config", "model_config"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint {path} missing fields: {sorted(missing)}.")
    cfg = NativeGCOConfig(**checkpoint["native_gco_config"])
    model_config = checkpoint["model_config"]
    model = GCONativeTransformer(
        vocab_size=int(model_config["vocab_size"]),
        d_model=int(model_config["d_model"]),
        n_layers=int(model_config["n_layers"]),
        n_heads=int(model_config["n_heads"]),
        d_ff=int(model_config["d_ff"]),
        max_seq_len=int(model_config["max_seq_len"]),
        cfg=cfg,
    ).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch for {path}: missing={missing_keys}, unexpected={unexpected_keys}."
        )
    return model, checkpoint


def trainable_weight_parameters(model: GCONativeTransformer) -> list[torch.nn.Parameter]:
    params = [module.W for module in model.gco_modules()]
    if not params:
        raise RuntimeError("Model exposes no trainable native module weights.")
    return params


def set_only_native_weights_trainable(model: GCONativeTransformer) -> None:
    weight_ids = {id(parameter) for parameter in trainable_weight_parameters(model)}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in weight_ids)


@torch.no_grad()
def collect_logits(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    positive_int("batch_size", batch_size)
    model.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        rows.append(model(batch_inputs).detach().cpu())
    if not rows:
        raise RuntimeError("No logits collected.")
    return torch.cat(rows, dim=0)


def old_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    teacher_margins: torch.Tensor,
    *,
    margin_slack: float,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: logits={logits.shape}, targets={targets.shape}.")
    if teacher_margins.shape != targets.shape:
        raise ValueError(f"teacher margin shape {teacher_margins.shape} != targets shape {targets.shape}.")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_teacher_margins = teacher_margins.reshape(-1)
    if flat_logits.shape[-1] < 2:
        raise ValueError("Margin preservation requires vocab size >= 2.")
    target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    top_values, top_indices = torch.topk(flat_logits, k=2, dim=-1)
    competitor_logits = torch.where(top_indices[:, 0] == flat_targets, top_values[:, 1], top_values[:, 0])
    current_margins = target_logits - competitor_logits
    required_margin = flat_teacher_margins - margin_slack
    return F.relu(required_margin - current_margins).square().mean()


@torch.no_grad()
def target_margins_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: logits={logits.shape}, targets={targets.shape}.")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    top_values, top_indices = torch.topk(flat_logits, k=2, dim=-1)
    competitor_logits = torch.where(top_indices[:, 0] == flat_targets, top_values[:, 1], top_values[:, 0])
    return (target_logits - competitor_logits).reshape_as(targets).detach().cpu()


def distillation_kl(
    current_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    positive_float("temperature", temperature)
    if current_logits.shape != teacher_logits.shape:
        raise ValueError(f"current/teacher logit shape mismatch: {current_logits.shape} vs {teacher_logits.shape}.")
    current_log_probs = F.log_softmax(current_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    token_count = current_logits[..., 0].numel()
    if token_count <= 0:
        raise RuntimeError("Distillation KL received zero tokens.")
    return F.kl_div(current_log_probs, teacher_probs, reduction="sum") * (temperature * temperature) / float(token_count)


def geometry_distance_to_target(
    source_states: torch.Tensor,
    target_states: torch.Tensor,
) -> dict[str, float]:
    aligned, drift = procrustes_align(source_states, target_states)
    cka_raw = linear_cka(target_states, source_states)
    cka_aligned = linear_cka(target_states, aligned)
    return {
        "aligned_drift_mean": drift["aligned_drift_mean"],
        "aligned_drift_relative": drift["aligned_drift_relative"],
        "aligned_drift_max": drift["aligned_drift_max"],
        "centroid_shift": drift["centroid_shift"],
        "cka_raw": cka_raw,
        "cka_aligned": cka_aligned,
    }


@torch.no_grad()
def collect_flat_states(
    model: GCONativeTransformer,
    windows: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        layer: value.reshape(-1, value.shape[-1]).to(dtype=torch.float32)
        for layer, value in collect_states(model, windows, device).items()
    }


@torch.no_grad()
def geometry_report(
    *,
    current: GCONativeTransformer,
    base100: GCONativeTransformer,
    base200: GCONativeTransformer,
    windows: torch.Tensor,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    current_states = collect_flat_states(current, windows, device=device)
    base100_states = collect_flat_states(base100, windows, device=device)
    base200_states = collect_flat_states(base200, windows, device=device)
    report: dict[str, dict[str, float]] = {}
    for layer in current_states:
        to_base100 = geometry_distance_to_target(current_states[layer], base100_states[layer])
        to_base200 = geometry_distance_to_target(current_states[layer], base200_states[layer])
        base100_to_base200 = geometry_distance_to_target(base100_states[layer], base200_states[layer])
        report[layer] = {
            "current_to_base100_drift_rel": to_base100["aligned_drift_relative"],
            "current_to_base100_cka": to_base100["cka_aligned"],
            "current_to_base200_drift_rel": to_base200["aligned_drift_relative"],
            "current_to_base200_cka": to_base200["cka_aligned"],
            "base100_to_base200_drift_rel": base100_to_base200["aligned_drift_relative"],
            "base100_to_base200_cka": base100_to_base200["cka_aligned"],
            "base200_closeness_gain": base100_to_base200["aligned_drift_relative"]
            - to_base200["aligned_drift_relative"],
            "base200_cka_gain": to_base200["cka_aligned"] - base100_to_base200["cka_aligned"],
            "current_to_base200_drift_mean": to_base200["aligned_drift_mean"],
            "current_to_base100_drift_mean": to_base100["aligned_drift_mean"],
        }
    return report


def mean_layer_metric(report: dict[str, dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in report.values()]
    if not values:
        raise RuntimeError(f"Geometry report has no values for {key!r}.")
    return sum(values) / float(len(values))


def make_optimizer(args: argparse.Namespace, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optimizer!r}.")


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
    positive_int("epochs", args.epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("geometry_max_windows", args.geometry_max_windows)
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
            "This diagnostic expects adjacent old and new spans so the geometry target is exactly old+new: "
            f"old=[{args.old_word_start},{args.old_word_start + args.old_word_count}) "
            f"new_start={args.new_word_start}."
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    current, base_checkpoint = load_checkpoint(args.base_checkpoint, device)
    base100, base100_checkpoint = load_checkpoint(args.base_checkpoint, device)
    base200, base200_checkpoint = load_checkpoint(args.target_checkpoint, device)
    if base_checkpoint["model_config"] != base200_checkpoint["model_config"]:
        raise ValueError(
            "Base and target checkpoint specs differ. "
            f"base={base_checkpoint['model_config']} target={base200_checkpoint['model_config']}"
        )
    if base_checkpoint["model_config"] != base100_checkpoint["model_config"]:
        raise RuntimeError("Loading the same base checkpoint produced mismatched model_config.")
    set_only_native_weights_trainable(current)
    base100.eval()
    base200.eval()
    for parameter in base100.parameters():
        parameter.requires_grad_(False)
    for parameter in base200.parameters():
        parameter.requires_grad_(False)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.chunk_index < 0 or args.chunk_index >= len(chunks):
        raise ValueError(f"chunk_index={args.chunk_index} outside chunk count {len(chunks)}.")
    text = str(chunks[args.chunk_index]["text"])
    old_text = word_span(text, args.old_word_start, args.old_word_count)
    new_text = word_span(text, args.new_word_start, args.new_word_count)
    full_text = word_span(text, args.old_word_start, args.old_word_count + args.new_word_count)
    old_token_ids = tokenizer.encode(old_text).ids
    new_token_ids = tokenizer.encode(new_text).ids
    full_token_ids = tokenizer.encode(full_text).ids
    seq_len = int(base_checkpoint["model_config"]["max_seq_len"])
    old_inputs, old_targets = build_lm_windows(
        old_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows
    )
    new_inputs, new_targets = build_lm_windows(
        new_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows
    )
    full_windows, _full_positions = build_windows(
        full_token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.geometry_max_windows
    )

    teacher_old_logits = collect_logits(base100, old_inputs, batch_size=args.eval_batch_size, device=device)
    teacher_old_margins = target_margins_from_logits(teacher_old_logits, old_targets)

    optimizer = make_optimizer(args, trainable_weight_parameters(current))
    old_initial = evaluate_model(current, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    new_initial = evaluate_model(current, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
    target_old = evaluate_model(base200, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    target_new = evaluate_model(base200, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
    initial_geometry = geometry_report(
        current=current,
        base100=base100,
        base200=base200,
        windows=full_windows,
        device=device,
    )

    print("TINY BEHAVIOR-PRESERVING PATH TEST")
    print("=" * 112)
    print("No hidden-state anchor loss is used. Old preservation is output KL + target-margin preservation only.")
    print(
        f"device={device} optimizer={args.optimizer} lr={args.lr:g} "
        f"lambda_kl={args.lambda_kl:g} lambda_margin={args.lambda_margin:g}"
    )
    print(
        "initial old_loss={:.6f} old_acc={:.4f} new_loss={:.6f} new_acc={:.4f}".format(
            old_initial["loss"],
            old_initial["token_accuracy"],
            new_initial["loss"],
            new_initial["token_accuracy"],
        )
    )
    print(
        "target200 old_loss={:.6f} old_acc={:.4f} new_loss={:.6f} new_acc={:.4f}".format(
            target_old["loss"],
            target_old["token_accuracy"],
            target_new["loss"],
            target_new["token_accuracy"],
        )
    )
    print(
        "initial geometry mean current->base200 rel={:.4f} cka={:.4f}".format(
            mean_layer_metric(initial_geometry, "current_to_base200_drift_rel"),
            mean_layer_metric(initial_geometry, "current_to_base200_cka"),
        )
    )

    trace: list[dict[str, Any]] = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        current.train()
        permutation = torch.randperm(new_inputs.shape[0])
        old_permutation = torch.randperm(old_inputs.shape[0])
        old_cursor = 0
        epoch_new_loss = 0.0
        epoch_kl = 0.0
        epoch_margin = 0.0
        epoch_total = 0.0
        batches = 0
        pbar = tqdm(range(0, new_inputs.shape[0], args.batch_size), desc=f"path epoch {epoch}/{args.epochs}")
        for start in pbar:
            step += 1
            new_indices = permutation[start : start + args.batch_size]
            if new_indices.numel() <= 0:
                raise RuntimeError(f"Empty new batch at start={start}.")
            old_batch_size = int(new_indices.numel())
            if old_cursor + old_batch_size > old_inputs.shape[0]:
                old_permutation = torch.randperm(old_inputs.shape[0])
                old_cursor = 0
            old_indices = old_permutation[old_cursor : old_cursor + old_batch_size]
            old_cursor += old_batch_size
            if old_indices.numel() != new_indices.numel():
                raise RuntimeError(f"Old/new batch size mismatch: old={old_indices.numel()} new={new_indices.numel()}.")

            batch_new_inputs = new_inputs[new_indices].to(device)
            batch_new_targets = new_targets[new_indices].to(device)
            batch_old_inputs = old_inputs[old_indices].to(device)
            batch_old_targets = old_targets[old_indices].to(device)
            batch_teacher_logits = teacher_old_logits[old_indices].to(device)
            batch_teacher_margins = teacher_old_margins[old_indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            new_logits = current(batch_new_inputs)
            old_logits = current(batch_old_inputs)
            new_ce = F.cross_entropy(new_logits.reshape(-1, new_logits.shape[-1]), batch_new_targets.reshape(-1))
            kl = distillation_kl(old_logits, batch_teacher_logits, temperature=args.distill_temperature)
            margin = old_margin_loss(
                old_logits,
                batch_old_targets,
                batch_teacher_margins,
                margin_slack=args.margin_slack,
            )
            loss = new_ce + args.lambda_kl * kl + args.lambda_margin * margin
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(current), args.grad_clip)
            optimizer.step()

            new_loss_value = float(new_ce.detach().cpu())
            kl_value = float(kl.detach().cpu())
            margin_value = float(margin.detach().cpu())
            total_value = float(loss.detach().cpu())
            epoch_new_loss += new_loss_value
            epoch_kl += kl_value
            epoch_margin += margin_value
            epoch_total += total_value
            batches += 1
            pbar.set_postfix(
                {
                    "new": f"{new_loss_value:.3f}",
                    "kl": f"{kl_value:.3g}",
                    "m": f"{margin_value:.3g}",
                    "tot": f"{total_value:.3f}",
                }
            )

        if batches <= 0:
            raise RuntimeError(f"Epoch {epoch} saw zero batches.")
        old_eval = evaluate_model(current, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
        new_eval = evaluate_model(current, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "mean_total_loss": epoch_total / float(batches),
            "mean_new_ce": epoch_new_loss / float(batches),
            "mean_old_kl": epoch_kl / float(batches),
            "mean_old_margin_loss": epoch_margin / float(batches),
            "old_loss": old_eval["loss"],
            "old_token_accuracy": old_eval["token_accuracy"],
            "old_target_margin_mean": old_eval["target_margin_mean"],
            "old_target_margin_min": old_eval["target_margin_min"],
            "new_loss": new_eval["loss"],
            "new_token_accuracy": new_eval["token_accuracy"],
            "new_target_margin_mean": new_eval["target_margin_mean"],
            "new_target_margin_min": new_eval["target_margin_min"],
        }
        if epoch == 1 or epoch == args.epochs or epoch % args.geometry_every == 0:
            epoch_geometry = geometry_report(
                current=current,
                base100=base100,
                base200=base200,
                windows=full_windows,
                device=device,
            )
            epoch_row["geometry"] = epoch_geometry
            epoch_row["geometry_mean"] = {
                "current_to_base100_drift_rel": mean_layer_metric(epoch_geometry, "current_to_base100_drift_rel"),
                "current_to_base100_cka": mean_layer_metric(epoch_geometry, "current_to_base100_cka"),
                "current_to_base200_drift_rel": mean_layer_metric(epoch_geometry, "current_to_base200_drift_rel"),
                "current_to_base200_cka": mean_layer_metric(epoch_geometry, "current_to_base200_cka"),
                "base200_closeness_gain": mean_layer_metric(epoch_geometry, "base200_closeness_gain"),
                "base200_cka_gain": mean_layer_metric(epoch_geometry, "base200_cka_gain"),
            }
        trace.append(epoch_row)
        geometry_text = ""
        if "geometry_mean" in epoch_row:
            geometry_mean = epoch_row["geometry_mean"]
            geometry_text = (
                " geom_to200_rel={:.4f} cka={:.4f} gain={:+.4f}".format(
                    geometry_mean["current_to_base200_drift_rel"],
                    geometry_mean["current_to_base200_cka"],
                    geometry_mean["base200_closeness_gain"],
                )
            )
        print(
            "epoch={:4d} total={:.5f} new={:.5f} kl={:.5f} margin={:.5f} "
            "old_loss={:.6f} old_acc={:.4f} new_loss={:.6f} new_acc={:.4f}{}".format(
                epoch,
                epoch_row["mean_total_loss"],
                epoch_row["mean_new_ce"],
                epoch_row["mean_old_kl"],
                epoch_row["mean_old_margin_loss"],
                old_eval["loss"],
                old_eval["token_accuracy"],
                new_eval["loss"],
                new_eval["token_accuracy"],
                geometry_text,
            )
        )

    final_old = evaluate_model(current, old_inputs, old_targets, batch_size=args.eval_batch_size, device=device)
    final_new = evaluate_model(current, new_inputs, new_targets, batch_size=args.eval_batch_size, device=device)
    final_geometry = geometry_report(
        current=current,
        base100=base100,
        base200=base200,
        windows=full_windows,
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

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "base_checkpoint": str(args.base_checkpoint),
            "target_checkpoint": str(args.target_checkpoint),
            "model_state_dict": current.state_dict(),
            "native_gco_config": asdict(NativeGCOConfig(**base_checkpoint["native_gco_config"])),
            "model_config": base_checkpoint["model_config"],
            "source": {
                "chunks_path": str(args.chunks_path),
                "chunk_index": args.chunk_index,
                "chunk_id": str(chunks[args.chunk_index]["chunk_id"]),
                "old_word_start": args.old_word_start,
                "old_word_count": args.old_word_count,
                "new_word_start": args.new_word_start,
                "new_word_count": args.new_word_count,
                "old_text": old_text,
                "new_text": new_text,
            },
            "old_final_metrics": final_old,
            "new_final_metrics": final_new,
            "final_geometry_mean": final_geometry_mean,
        },
        args.output_checkpoint,
    )
    result = {
        "question": "Does an output-behavior-preserving path from base100 toward base200-like geometry exist?",
        "constraint": "No hidden-state anchor loss; old preservation is output KL plus target-margin preservation.",
        "base_checkpoint": str(args.base_checkpoint),
        "target_checkpoint": str(args.target_checkpoint),
        "output_checkpoint": str(args.output_checkpoint),
        "model_config": base_checkpoint["model_config"],
        "source": {
            "chunks_path": str(args.chunks_path),
            "chunk_index": args.chunk_index,
            "chunk_id": str(chunks[args.chunk_index]["chunk_id"]),
            "old_word_start": args.old_word_start,
            "old_word_count": args.old_word_count,
            "new_word_start": args.new_word_start,
            "new_word_count": args.new_word_count,
            "old_window_count": int(old_inputs.shape[0]),
            "new_window_count": int(new_inputs.shape[0]),
            "geometry_window_count": int(full_windows.shape[0]),
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
        "old_initial": old_initial,
        "new_initial": new_initial,
        "target200_old": target_old,
        "target200_new": target_new,
        "old_final": final_old,
        "new_final": final_new,
        "initial_geometry": initial_geometry,
        "final_geometry": final_geometry,
        "final_geometry_mean": final_geometry_mean,
        "trace": trace,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nTINY BEHAVIOR-PRESERVING PATH SUMMARY")
    print("=" * 112)
    print(
        "old_loss {:.6f}->{:.6f} old_acc {:.4f}->{:.4f} old_margin {:.4f}->{:.4f}".format(
            old_initial["loss"],
            final_old["loss"],
            old_initial["token_accuracy"],
            final_old["token_accuracy"],
            old_initial["target_margin_mean"],
            final_old["target_margin_mean"],
        )
    )
    print(
        "new_loss {:.6f}->{:.6f} new_acc {:.4f}->{:.4f} target200_new_loss={:.6f}".format(
            new_initial["loss"],
            final_new["loss"],
            new_initial["token_accuracy"],
            final_new["token_accuracy"],
            target_new["loss"],
        )
    )
    print(
        "geometry mean to_base200 rel {:.4f}->{:.4f} cka {:.4f}->{:.4f} closeness_gain={:+.4f} cka_gain={:+.4f}".format(
            mean_layer_metric(initial_geometry, "current_to_base200_drift_rel"),
            final_geometry_mean["current_to_base200_drift_rel"],
            mean_layer_metric(initial_geometry, "current_to_base200_cka"),
            final_geometry_mean["current_to_base200_cka"],
            final_geometry_mean["base200_closeness_gain"],
            final_geometry_mean["base200_cka_gain"],
        )
    )
    print(f"wrote_checkpoint={args.output_checkpoint}")
    print(f"wrote_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        default=Path("model/checkpoints/gco-tiny-cl-base-200w-samespec-seed0.pt"),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("model/checkpoints/gco-tiny-cl-behavior-path-100to200-seed0.pt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("model/analysis/gco-tiny-cl-behavior-path-100to200-seed0.json"),
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
    parser.add_argument("--geometry-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
