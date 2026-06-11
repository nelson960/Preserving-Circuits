"""Recursive residual rebasing over multiple tiny continual-learning stages.

This is the first complete fixed-budget version of the architecture:

    core transformer
    + temporary plastic adapter during learning
    + fixed-size behavior anchor memory
    + bridge windows around old->new boundaries
    + consolidation back into the core
    + anchor refresh under a strict budget

Every stage starts from the current consolidated core. The adapter is discarded
after consolidation, so inference remains a single core model.
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
from experiments.gco_math.gco_native_scratch_transformer import GCONativeTransformer, NativeGCOConfig
from experiments.gco_math.gco_prepare_tiny_cl_base import build_lm_windows, evaluate_model, load_chunks
from experiments.gco_math.gco_tiny_cl_behavior_budget_sweep import output_kl_and_agreement
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
from experiments.gco_math.gco_tiny_cl_bridge_adapter_consolidation import (
    AdapterWrappedTransformer,
    FinalResidualAdapter,
    build_bridge_lm_windows,
    freeze_model,
    trainable_adapter_parameters,
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


def parse_target_checkpoints(raw: str, stages: int) -> list[Path]:
    positive_int("stages", stages)
    paths = [Path(item.strip()) for item in raw.split(",") if item.strip()]
    if len(paths) != stages:
        raise ValueError(f"--stage-target-checkpoints must contain {stages} paths, got {len(paths)}.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Stage target checkpoint does not exist: {path}")
    return paths


def clone_core_from_state(
    *,
    state_dict: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> GCONativeTransformer:
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
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Clone load mismatch: missing={missing}, unexpected={unexpected}.")
    return model


def detached_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_stage_data(
    *,
    tokenizer: Tokenizer,
    text: str,
    seq_len: int,
    stride: int,
    max_windows: int,
    max_bridge_windows: int,
    bridge_radius_tokens: int,
    geometry_max_windows: int,
    old_word_start: int,
    previous_word_count: int,
    new_word_start: int,
    new_word_count: int,
) -> dict[str, Any]:
    previous_text = word_span(text, old_word_start, previous_word_count)
    new_text = word_span(text, new_word_start, new_word_count)
    full_text = word_span(text, old_word_start, previous_word_count + new_word_count)
    previous_token_ids = tokenizer.encode(previous_text).ids
    new_token_ids = tokenizer.encode(new_text).ids
    full_token_ids = tokenizer.encode(full_text).ids
    boundary_token_index = len(previous_token_ids)

    previous_inputs, previous_targets = build_lm_windows(
        previous_token_ids, seq_len=seq_len, stride=stride, max_windows=max_windows
    )
    new_inputs, new_targets = build_lm_windows(new_token_ids, seq_len=seq_len, stride=stride, max_windows=max_windows)
    full_inputs, full_targets = build_lm_windows(full_token_ids, seq_len=seq_len, stride=stride, max_windows=max_windows)
    bridge_inputs, bridge_targets, bridge_starts = build_bridge_lm_windows(
        full_token_ids,
        boundary_token_index=boundary_token_index,
        seq_len=seq_len,
        stride=stride,
        max_windows=max_bridge_windows,
        radius_tokens=bridge_radius_tokens,
    )
    geometry_windows, _positions = build_windows(
        full_token_ids, seq_len=seq_len, stride=stride, max_windows=geometry_max_windows
    )
    return {
        "previous_text": previous_text,
        "new_text": new_text,
        "full_text": full_text,
        "previous_inputs": previous_inputs,
        "previous_targets": previous_targets,
        "new_inputs": new_inputs,
        "new_targets": new_targets,
        "full_inputs": full_inputs,
        "full_targets": full_targets,
        "bridge_inputs": bridge_inputs,
        "bridge_targets": bridge_targets,
        "bridge_starts": bridge_starts,
        "geometry_windows": geometry_windows,
        "previous_token_count": len(previous_token_ids),
        "new_token_count": len(new_token_ids),
        "full_token_count": len(full_token_ids),
        "boundary_token_index": boundary_token_index,
    }


def initial_anchor_memory(
    *,
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    budget: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    positive_int("budget", budget)
    if budget > inputs.shape[0]:
        raise ValueError(f"anchor_budget={budget} exceeds available initial windows={inputs.shape[0]}.")
    indices = torch.linspace(0, inputs.shape[0] - 1, steps=budget).round().to(dtype=torch.long).unique(sorted=True)
    if indices.numel() != budget:
        raise RuntimeError(f"Initial anchor selection returned {indices.numel()} unique indices for budget={budget}.")
    anchor_inputs = inputs[indices].clone()
    anchor_targets = targets[indices].clone()
    anchor_logits = collect_logits(model, anchor_inputs, batch_size=batch_size, device=device)
    anchor_margins = target_margins_from_logits(anchor_logits, anchor_targets)
    return {
        "inputs": anchor_inputs,
        "targets": anchor_targets,
        "logits": anchor_logits,
        "margins": anchor_margins,
        "source": ["initial"] * int(anchor_inputs.shape[0]),
    }


def sample_indices(total: int, count: int, *, generator: torch.Generator) -> torch.Tensor:
    positive_int("total", total)
    positive_int("count", count)
    return torch.randint(low=0, high=total, size=(count,), generator=generator, device=torch.device("cpu"))


def select_anchor_indices_from_logits(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    count: int,
    selection: str,
) -> torch.Tensor:
    nonnegative_int("count", count)
    total = int(targets.shape[0])
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    if count > total:
        raise ValueError(f"Requested {count} anchors from a pool of {total}.")
    if logits.shape[:-1] != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: logits={logits.shape}, targets={targets.shape}.")
    if count == total:
        return torch.arange(total, dtype=torch.long)
    if selection == "uniform":
        indices = torch.linspace(0, total - 1, steps=count).round().to(dtype=torch.long).unique(sorted=True)
        if indices.numel() != count:
            raise RuntimeError(f"Uniform selector returned {indices.numel()} unique indices for count={count}.")
        return indices
    if selection != "mixed":
        raise ValueError(f"Unknown per-pool anchor selection: {selection!r}.")
    margins = target_margins_from_logits(logits, targets)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape).mean(dim=1)
    uniform_count = max(1, count // 2)
    low_margin_count = max(1, count // 4) if count >= 2 else 0
    high_loss_count = count - uniform_count - low_margin_count
    selected: list[int] = []

    def add(indices: torch.Tensor) -> None:
        for value in indices.tolist():
            item = int(value)
            if item not in selected:
                selected.append(item)
            if len(selected) >= count:
                break

    add(torch.linspace(0, total - 1, steps=uniform_count).round().to(dtype=torch.long))
    if low_margin_count > 0:
        add(torch.topk(-margins.mean(dim=1), k=low_margin_count).indices)
    if high_loss_count > 0:
        add(torch.topk(token_losses, k=high_loss_count).indices)
    if len(selected) < count:
        add(torch.arange(total, dtype=torch.long))
    if len(selected) != count:
        raise RuntimeError(f"Mixed selector returned {len(selected)} indices for count={count}.")
    return torch.tensor(selected, dtype=torch.long)


def evaluate_splits(
    *,
    model: nn.Module,
    stage_data: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    return {
        "previous": evaluate_model(
            model, stage_data["previous_inputs"], stage_data["previous_targets"], batch_size=batch_size, device=device
        ),
        "new": evaluate_model(model, stage_data["new_inputs"], stage_data["new_targets"], batch_size=batch_size, device=device),
        "bridge": evaluate_model(
            model, stage_data["bridge_inputs"], stage_data["bridge_targets"], batch_size=batch_size, device=device
        ),
        "full": evaluate_model(
            model, stage_data["full_inputs"], stage_data["full_targets"], batch_size=batch_size, device=device
        ),
    }


def train_adapter_stage(
    *,
    args: argparse.Namespace,
    wrapped: AdapterWrappedTransformer,
    adapter: FinalResidualAdapter,
    stage_data: dict[str, Any],
    anchors: dict[str, Any],
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    freeze_model(wrapped.base)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    optimizer = make_optimizer(args, trainable_adapter_parameters(adapter))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    anchor_count = int(anchors["inputs"].shape[0])
    trace: list[dict[str, float]] = []
    for epoch in range(1, args.plastic_epochs + 1):
        wrapped.train()
        permutation = torch.randperm(stage_data["new_inputs"].shape[0], generator=generator)
        totals = {"loss": 0.0, "new": 0.0, "bridge": 0.0, "anchor_kl": 0.0, "margin": 0.0, "adapter": 0.0}
        batches = 0
        pbar = tqdm(
            range(0, stage_data["new_inputs"].shape[0], args.batch_size),
            desc=f"stage adapter {epoch}/{args.plastic_epochs}",
        )
        for start in pbar:
            new_indices = permutation[start : start + args.batch_size]
            if new_indices.numel() <= 0:
                raise RuntimeError(f"Empty new batch at start={start}.")
            batch_size = int(new_indices.numel())
            bridge_indices = sample_indices(stage_data["bridge_inputs"].shape[0], batch_size, generator=generator)
            anchor_indices = sample_indices(anchor_count, batch_size, generator=generator)

            optimizer.zero_grad(set_to_none=True)
            new_logits = wrapped(stage_data["new_inputs"][new_indices].to(device))
            bridge_logits = wrapped(stage_data["bridge_inputs"][bridge_indices].to(device))
            anchor_logits = wrapped(anchors["inputs"][anchor_indices].to(device))
            new_loss = F.cross_entropy(
                new_logits.reshape(-1, new_logits.shape[-1]),
                stage_data["new_targets"][new_indices].to(device).reshape(-1),
            )
            bridge_loss = F.cross_entropy(
                bridge_logits.reshape(-1, bridge_logits.shape[-1]),
                stage_data["bridge_targets"][bridge_indices].to(device).reshape(-1),
            )
            anchor_kl = distillation_kl(
                anchor_logits,
                anchors["logits"][anchor_indices].to(device),
                temperature=args.distill_temperature,
            )
            margin = old_margin_loss(
                anchor_logits,
                anchors["targets"][anchor_indices].to(device),
                anchors["margins"][anchor_indices].to(device),
                margin_slack=args.margin_slack,
            )
            adapter_penalty = adapter.penalty()
            loss = (
                new_loss
                + args.lambda_bridge * bridge_loss
                + args.lambda_anchor * anchor_kl
                + args.lambda_margin * margin
                + args.lambda_adapter * adapter_penalty
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_adapter_parameters(adapter), args.grad_clip)
            optimizer.step()
            values = {
                "loss": float(loss.detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "bridge": float(bridge_loss.detach().cpu()),
                "anchor_kl": float(anchor_kl.detach().cpu()),
                "margin": float(margin.detach().cpu()),
                "adapter": float(adapter_penalty.detach().cpu()),
            }
            for key, value in values.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"new": f"{values['new']:.3g}", "bridge": f"{values['bridge']:.3g}", "kl": f"{values['anchor_kl']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"Adapter epoch {epoch} saw zero batches.")
        row = {key: value / float(batches) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        trace.append(row)
        print(
            "adapter epoch={:4d} loss={:.5f} new={:.5f} bridge={:.5f} anchor_kl={:.5f}".format(
                epoch, row["loss"], row["new"], row["bridge"], row["anchor_kl"]
            )
        )
    return trace


def train_consolidation_stage(
    *,
    args: argparse.Namespace,
    core: GCONativeTransformer,
    adapter_teacher: AdapterWrappedTransformer,
    stage_data: dict[str, Any],
    anchors: dict[str, Any],
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    set_only_native_weights_trainable(core)
    freeze_model(adapter_teacher)
    adapter_teacher.eval()
    optimizer = make_optimizer(args, trainable_weight_parameters(core))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    anchor_count = int(anchors["inputs"].shape[0])

    trace: list[dict[str, float]] = []
    for epoch in range(1, args.consolidation_epochs + 1):
        core.train()
        permutation = torch.randperm(stage_data["new_inputs"].shape[0], generator=generator)
        totals = {"loss": 0.0, "new": 0.0, "bridge": 0.0, "anchor_kl": 0.0, "margin": 0.0, "adapter_kl": 0.0}
        batches = 0
        pbar = tqdm(
            range(0, stage_data["new_inputs"].shape[0], args.batch_size),
            desc=f"stage consolidate {epoch}/{args.consolidation_epochs}",
        )
        for start in pbar:
            new_indices = permutation[start : start + args.batch_size]
            if new_indices.numel() <= 0:
                raise RuntimeError(f"Empty new batch at start={start}.")
            batch_size = int(new_indices.numel())
            bridge_indices = sample_indices(stage_data["bridge_inputs"].shape[0], batch_size, generator=generator)
            full_indices = sample_indices(stage_data["full_inputs"].shape[0], batch_size, generator=generator)
            anchor_indices = sample_indices(anchor_count, batch_size, generator=generator)

            new_inputs = stage_data["new_inputs"][new_indices].to(device)
            new_targets = stage_data["new_targets"][new_indices].to(device)
            bridge_inputs = stage_data["bridge_inputs"][bridge_indices].to(device)
            bridge_targets = stage_data["bridge_targets"][bridge_indices].to(device)
            full_inputs = stage_data["full_inputs"][full_indices].to(device)
            anchor_inputs = anchors["inputs"][anchor_indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            new_logits = core(new_inputs)
            bridge_logits = core(bridge_inputs)
            full_logits = core(full_inputs)
            anchor_logits = core(anchor_inputs)
            new_loss = F.cross_entropy(new_logits.reshape(-1, new_logits.shape[-1]), new_targets.reshape(-1))
            bridge_loss = F.cross_entropy(
                bridge_logits.reshape(-1, bridge_logits.shape[-1]), bridge_targets.reshape(-1)
            )
            with torch.no_grad():
                adapter_new_logits = adapter_teacher(new_inputs)
                adapter_bridge_logits = adapter_teacher(bridge_inputs)
                adapter_full_logits = adapter_teacher(full_inputs)
            adapter_kl = (
                distillation_kl(new_logits, adapter_new_logits, temperature=args.distill_temperature)
                + distillation_kl(bridge_logits, adapter_bridge_logits, temperature=args.distill_temperature)
                + distillation_kl(full_logits, adapter_full_logits, temperature=args.distill_temperature)
            ) / 3.0
            anchor_kl = distillation_kl(
                anchor_logits,
                anchors["logits"][anchor_indices].to(device),
                temperature=args.distill_temperature,
            )
            margin = old_margin_loss(
                anchor_logits,
                anchors["targets"][anchor_indices].to(device),
                anchors["margins"][anchor_indices].to(device),
                margin_slack=args.margin_slack,
            )
            loss = (
                new_loss
                + args.lambda_bridge * bridge_loss
                + args.lambda_anchor * anchor_kl
                + args.lambda_margin * margin
                + args.lambda_adapter_distill * adapter_kl
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_weight_parameters(core), args.grad_clip)
            optimizer.step()
            values = {
                "loss": float(loss.detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "bridge": float(bridge_loss.detach().cpu()),
                "anchor_kl": float(anchor_kl.detach().cpu()),
                "margin": float(margin.detach().cpu()),
                "adapter_kl": float(adapter_kl.detach().cpu()),
            }
            for key, value in values.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"new": f"{values['new']:.3g}", "bridge": f"{values['bridge']:.3g}", "aKL": f"{values['anchor_kl']:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"Consolidation epoch {epoch} saw zero batches.")
        row = {key: value / float(batches) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        trace.append(row)
        print(
            "consolidate epoch={:4d} loss={:.5f} new={:.5f} bridge={:.5f} anchor_kl={:.5f} adapter_kl={:.5f}".format(
                epoch, row["loss"], row["new"], row["bridge"], row["anchor_kl"], row["adapter_kl"]
            )
        )
    return trace


@torch.no_grad()
def refresh_anchor_memory(
    *,
    model: nn.Module,
    old_anchors: dict[str, Any],
    stage_data: dict[str, Any],
    budget: int,
    selection: str,
    long_term_count: int,
    recent_count: int,
    bridge_count: int,
    per_pool_selection: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    positive_int("budget", budget)
    nonnegative_int("long_term_count", long_term_count)
    nonnegative_int("recent_count", recent_count)
    nonnegative_int("bridge_count", bridge_count)
    if long_term_count + recent_count + bridge_count != budget:
        raise ValueError(
            "Stratified anchor counts must sum to anchor budget: "
            f"{long_term_count}+{recent_count}+{bridge_count}!={budget}."
        )
    if selection == "stratified":
        if old_anchors["inputs"].shape[0] < long_term_count:
            raise ValueError(
                f"Cannot keep {long_term_count} long-term anchors from old memory of size "
                f"{old_anchors['inputs'].shape[0]}."
            )
        if stage_data["new_inputs"].shape[0] < recent_count:
            raise ValueError(
                f"Cannot select {recent_count} recent anchors from {stage_data['new_inputs'].shape[0]} new windows."
            )
        if stage_data["bridge_inputs"].shape[0] < bridge_count:
            raise ValueError(
                f"Cannot select {bridge_count} bridge anchors from {stage_data['bridge_inputs'].shape[0]} bridge windows."
            )

        long_inputs = old_anchors["inputs"][:long_term_count].clone()
        long_targets = old_anchors["targets"][:long_term_count].clone()
        long_logits = old_anchors["logits"][:long_term_count].clone()
        long_margins = old_anchors["margins"][:long_term_count].clone()
        long_source = list(old_anchors["source"][:long_term_count])

        recent_logits_all = collect_logits(model, stage_data["new_inputs"], batch_size=batch_size, device=device)
        recent_indices = select_anchor_indices_from_logits(
            logits=recent_logits_all,
            targets=stage_data["new_targets"],
            count=recent_count,
            selection=per_pool_selection,
        )
        recent_inputs = stage_data["new_inputs"][recent_indices].clone()
        recent_targets = stage_data["new_targets"][recent_indices].clone()
        recent_logits = recent_logits_all[recent_indices].clone()
        recent_margins = target_margins_from_logits(recent_logits, recent_targets)
        recent_source = ["recent"] * recent_count

        bridge_logits_all = collect_logits(model, stage_data["bridge_inputs"], batch_size=batch_size, device=device)
        bridge_indices = select_anchor_indices_from_logits(
            logits=bridge_logits_all,
            targets=stage_data["bridge_targets"],
            count=bridge_count,
            selection=per_pool_selection,
        )
        bridge_inputs = stage_data["bridge_inputs"][bridge_indices].clone()
        bridge_targets = stage_data["bridge_targets"][bridge_indices].clone()
        bridge_logits = bridge_logits_all[bridge_indices].clone()
        bridge_margins = target_margins_from_logits(bridge_logits, bridge_targets)
        bridge_source = ["bridge"] * bridge_count

        return {
            "inputs": torch.cat([long_inputs, recent_inputs, bridge_inputs], dim=0),
            "targets": torch.cat([long_targets, recent_targets, bridge_targets], dim=0),
            "logits": torch.cat([long_logits, recent_logits, bridge_logits], dim=0),
            "margins": torch.cat([long_margins, recent_margins, bridge_margins], dim=0),
            "source": long_source + recent_source + bridge_source,
        }

    candidate_inputs = torch.cat(
        [old_anchors["inputs"], stage_data["new_inputs"], stage_data["bridge_inputs"]],
        dim=0,
    )
    candidate_targets = torch.cat(
        [old_anchors["targets"], stage_data["new_targets"], stage_data["bridge_targets"]],
        dim=0,
    )
    candidate_source = (
        list(old_anchors["source"])
        + ["new"] * int(stage_data["new_inputs"].shape[0])
        + ["bridge"] * int(stage_data["bridge_inputs"].shape[0])
    )
    total = int(candidate_inputs.shape[0])
    if budget > total:
        raise ValueError(f"anchor budget={budget} exceeds candidate count={total}.")
    logits = collect_logits(model, candidate_inputs, batch_size=batch_size, device=device)
    margins = target_margins_from_logits(logits, candidate_targets)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        candidate_targets.reshape(-1),
        reduction="none",
    ).reshape(candidate_targets.shape).mean(dim=1)

    selected: list[int] = []

    def add_indices(indices: torch.Tensor) -> None:
        for value in indices.tolist():
            item = int(value)
            if item not in selected:
                selected.append(item)
            if len(selected) >= budget:
                break

    if selection == "uniform":
        add_indices(torch.linspace(0, total - 1, steps=budget).round().to(dtype=torch.long))
    elif selection == "mixed":
        uniform_count = max(1, budget // 2)
        low_margin_count = max(1, budget // 4)
        high_loss_count = budget - uniform_count - low_margin_count
        add_indices(torch.linspace(0, total - 1, steps=uniform_count).round().to(dtype=torch.long))
        add_indices(torch.topk(-margins.mean(dim=1), k=low_margin_count).indices)
        if high_loss_count > 0:
            add_indices(torch.topk(token_losses, k=high_loss_count).indices)
    else:
        raise ValueError(f"Unknown anchor refresh selection: {selection!r}.")
    if len(selected) < budget:
        add_indices(torch.arange(total, dtype=torch.long))
    if len(selected) != budget:
        raise RuntimeError(f"Anchor refresh selected {len(selected)} anchors for budget={budget}.")
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    return {
        "inputs": candidate_inputs[selected_tensor].clone(),
        "targets": candidate_targets[selected_tensor].clone(),
        "logits": logits[selected_tensor].clone(),
        "margins": margins[selected_tensor].clone(),
        "source": [candidate_source[index] for index in selected],
    }


def anchor_source_counts(anchors: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in anchors["source"]:
        counts[str(item)] = counts.get(str(item), 0) + 1
    return counts


def validate_args(args: argparse.Namespace) -> None:
    for name in ["base_checkpoint", "tokenizer_path", "chunks_path"]:
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    nonnegative_int("chunk_index", args.chunk_index)
    nonnegative_int("old_word_start", args.old_word_start)
    positive_int("initial_word_count", args.initial_word_count)
    positive_int("new_word_count", args.new_word_count)
    positive_int("stages", args.stages)
    positive_int("anchor_budget", args.anchor_budget)
    nonnegative_int("anchor_long_term_count", args.anchor_long_term_count)
    nonnegative_int("anchor_recent_count", args.anchor_recent_count)
    nonnegative_int("anchor_bridge_count", args.anchor_bridge_count)
    if args.anchor_refresh_selection == "stratified":
        total_slots = args.anchor_long_term_count + args.anchor_recent_count + args.anchor_bridge_count
        if total_slots != args.anchor_budget:
            raise ValueError(
                "Stratified anchor counts must sum to --anchor-budget: "
                f"{args.anchor_long_term_count}+{args.anchor_recent_count}+{args.anchor_bridge_count}"
                f"!={args.anchor_budget}."
            )
    positive_int("stride", args.stride)
    positive_int("max_windows", args.max_windows)
    positive_int("max_bridge_windows", args.max_bridge_windows)
    nonnegative_int("bridge_radius_tokens", args.bridge_radius_tokens)
    positive_int("geometry_max_windows", args.geometry_max_windows)
    positive_int("plastic_epochs", args.plastic_epochs)
    positive_int("consolidation_epochs", args.consolidation_epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("adapter_rank", args.adapter_rank)
    positive_float("adapter_scale", args.adapter_scale)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    nonnegative_float("lambda_anchor", args.lambda_anchor)
    nonnegative_float("lambda_margin", args.lambda_margin)
    nonnegative_float("lambda_bridge", args.lambda_bridge)
    nonnegative_float("lambda_adapter", args.lambda_adapter)
    nonnegative_float("lambda_adapter_distill", args.lambda_adapter_distill)
    positive_float("distill_temperature", args.distill_temperature)
    nonnegative_float("margin_slack", args.margin_slack)
    positive_float("grad_clip", args.grad_clip)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    target_paths = parse_target_checkpoints(args.stage_target_checkpoints, args.stages)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    base_core, base_checkpoint = load_checkpoint(args.base_checkpoint, device)
    freeze_model(base_core)
    base_core.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.chunk_index >= len(chunks):
        raise ValueError(f"chunk_index={args.chunk_index} outside chunk count {len(chunks)}.")
    text = str(chunks[args.chunk_index]["text"])
    seq_len = int(base_checkpoint["model_config"]["max_seq_len"])
    d_model = int(base_checkpoint["model_config"]["d_model"])

    initial_data = build_stage_data(
        tokenizer=tokenizer,
        text=text,
        seq_len=seq_len,
        stride=args.stride,
        max_windows=args.max_windows,
        max_bridge_windows=args.max_bridge_windows,
        bridge_radius_tokens=args.bridge_radius_tokens,
        geometry_max_windows=args.geometry_max_windows,
        old_word_start=args.old_word_start,
        previous_word_count=args.initial_word_count,
        new_word_start=args.old_word_start + args.initial_word_count,
        new_word_count=args.new_word_count,
    )
    anchors = initial_anchor_memory(
        model=base_core,
        inputs=initial_data["previous_inputs"],
        targets=initial_data["previous_targets"],
        budget=args.anchor_budget,
        batch_size=args.eval_batch_size,
        device=device,
    )

    current_state = detached_state_dict(base_core)
    stage_results: list[dict[str, Any]] = []

    print("TINY RECURSIVE RESIDUAL REBASING")
    print("=" * 112)
    print(
        f"device={device} stages={args.stages} anchor_budget={args.anchor_budget} "
        f"refresh={args.anchor_refresh_selection} adapter_rank={args.adapter_rank}"
    )

    for stage_index in range(args.stages):
        previous_word_count = args.initial_word_count + stage_index * args.new_word_count
        new_word_start = args.old_word_start + previous_word_count
        stage_data = build_stage_data(
            tokenizer=tokenizer,
            text=text,
            seq_len=seq_len,
            stride=args.stride,
            max_windows=args.max_windows,
            max_bridge_windows=args.max_bridge_windows,
            bridge_radius_tokens=args.bridge_radius_tokens,
            geometry_max_windows=args.geometry_max_windows,
            old_word_start=args.old_word_start,
            previous_word_count=previous_word_count,
            new_word_start=new_word_start,
            new_word_count=args.new_word_count,
        )
        target_model, target_checkpoint = load_checkpoint(target_paths[stage_index], device)
        if base_checkpoint["model_config"] != target_checkpoint["model_config"]:
            raise ValueError(
                f"Stage {stage_index + 1} target spec differs from base spec: "
                f"{target_checkpoint['model_config']} vs {base_checkpoint['model_config']}"
            )
        freeze_model(target_model)
        target_model.eval()
        stage_core_teacher = clone_core_from_state(state_dict=current_state, checkpoint=base_checkpoint, device=device)
        freeze_model(stage_core_teacher)
        stage_core_teacher.eval()

        adapter_base = clone_core_from_state(state_dict=current_state, checkpoint=base_checkpoint, device=device)
        freeze_model(adapter_base)
        adapter = FinalResidualAdapter(d_model=d_model, rank=args.adapter_rank, scale=args.adapter_scale).to(device)
        wrapped = AdapterWrappedTransformer(adapter_base, adapter).to(device)

        initial_metrics = evaluate_splits(
            model=stage_core_teacher,
            stage_data=stage_data,
            batch_size=args.eval_batch_size,
            device=device,
        )
        target_metrics = evaluate_splits(
            model=target_model,
            stage_data=stage_data,
            batch_size=args.eval_batch_size,
            device=device,
        )
        print("\n" + "-" * 112)
        print(
            f"stage={stage_index + 1}/{args.stages} previous_words={previous_word_count} "
            f"new_words=[{new_word_start},{new_word_start + args.new_word_count}) "
            f"anchors={anchors['inputs'].shape[0]} sources={anchor_source_counts(anchors)} "
            f"bridge_windows={stage_data['bridge_inputs'].shape[0]}"
        )
        print(
            "stage initial previous={:.5f}/{:.4f} new={:.5f}/{:.4f} bridge={:.5f}/{:.4f} full={:.5f}/{:.4f}".format(
                initial_metrics["previous"]["loss"],
                initial_metrics["previous"]["token_accuracy"],
                initial_metrics["new"]["loss"],
                initial_metrics["new"]["token_accuracy"],
                initial_metrics["bridge"]["loss"],
                initial_metrics["bridge"]["token_accuracy"],
                initial_metrics["full"]["loss"],
                initial_metrics["full"]["token_accuracy"],
            )
        )
        print(
            "stage target  previous={:.5f}/{:.4f} new={:.5f}/{:.4f} bridge={:.5f}/{:.4f} full={:.5f}/{:.4f}".format(
                target_metrics["previous"]["loss"],
                target_metrics["previous"]["token_accuracy"],
                target_metrics["new"]["loss"],
                target_metrics["new"]["token_accuracy"],
                target_metrics["bridge"]["loss"],
                target_metrics["bridge"]["token_accuracy"],
                target_metrics["full"]["loss"],
                target_metrics["full"]["token_accuracy"],
            )
        )

        adapter_trace = train_adapter_stage(
            args=args,
            wrapped=wrapped,
            adapter=adapter,
            stage_data=stage_data,
            anchors=anchors,
            device=device,
            seed=args.seed + stage_index * 10000 + 101,
        )
        plastic_metrics = evaluate_splits(model=wrapped, stage_data=stage_data, batch_size=args.eval_batch_size, device=device)

        consolidated_core = clone_core_from_state(state_dict=current_state, checkpoint=base_checkpoint, device=device)
        consolidation_trace = train_consolidation_stage(
            args=args,
            core=consolidated_core,
            adapter_teacher=wrapped,
            stage_data=stage_data,
            anchors=anchors,
            device=device,
            seed=args.seed + stage_index * 10000 + 202,
        )
        consolidated_metrics = evaluate_splits(
            model=consolidated_core,
            stage_data=stage_data,
            batch_size=args.eval_batch_size,
            device=device,
        )
        geometry = geometry_report(
            current=consolidated_core,
            base100=stage_core_teacher,
            base200=target_model,
            windows=stage_data["geometry_windows"],
            device=device,
        )
        geometry_mean = {
            "current_to_stage_start_drift_rel": mean_layer_metric(geometry, "current_to_base100_drift_rel"),
            "current_to_stage_start_cka": mean_layer_metric(geometry, "current_to_base100_cka"),
            "current_to_target_drift_rel": mean_layer_metric(geometry, "current_to_base200_drift_rel"),
            "current_to_target_cka": mean_layer_metric(geometry, "current_to_base200_cka"),
            "stage_start_to_target_drift_rel": mean_layer_metric(geometry, "base100_to_base200_drift_rel"),
            "stage_start_to_target_cka": mean_layer_metric(geometry, "base100_to_base200_cka"),
            "target_closeness_gain": mean_layer_metric(geometry, "base200_closeness_gain"),
            "target_cka_gain": mean_layer_metric(geometry, "base200_cka_gain"),
        }
        target_logits = {
            "previous": collect_logits(target_model, stage_data["previous_inputs"], batch_size=args.eval_batch_size, device=device),
            "new": collect_logits(target_model, stage_data["new_inputs"], batch_size=args.eval_batch_size, device=device),
            "bridge": collect_logits(target_model, stage_data["bridge_inputs"], batch_size=args.eval_batch_size, device=device),
            "full": collect_logits(target_model, stage_data["full_inputs"], batch_size=args.eval_batch_size, device=device),
        }
        current_logits = {
            "previous": collect_logits(consolidated_core, stage_data["previous_inputs"], batch_size=args.eval_batch_size, device=device),
            "new": collect_logits(consolidated_core, stage_data["new_inputs"], batch_size=args.eval_batch_size, device=device),
            "bridge": collect_logits(consolidated_core, stage_data["bridge_inputs"], batch_size=args.eval_batch_size, device=device),
            "full": collect_logits(consolidated_core, stage_data["full_inputs"], batch_size=args.eval_batch_size, device=device),
        }
        behavior_match = {
            key: output_kl_and_agreement(
                current_logits=current_logits[key],
                teacher_logits=target_logits[key],
                temperature=args.distill_temperature,
            )
            for key in ["previous", "new", "bridge", "full"]
        }

        anchors = refresh_anchor_memory(
            model=consolidated_core,
            old_anchors=anchors,
            stage_data=stage_data,
            budget=args.anchor_budget,
            selection=args.anchor_refresh_selection,
            long_term_count=args.anchor_long_term_count,
            recent_count=args.anchor_recent_count,
            bridge_count=args.anchor_bridge_count,
            per_pool_selection=args.anchor_stratified_pool_selection,
            batch_size=args.eval_batch_size,
            device=device,
        )
        current_state = detached_state_dict(consolidated_core)

        stage_result = {
            "stage": stage_index + 1,
            "previous_word_count": previous_word_count,
            "new_word_start": new_word_start,
            "new_word_count": args.new_word_count,
            "target_checkpoint": str(target_paths[stage_index]),
            "initial_metrics": initial_metrics,
            "target_metrics": target_metrics,
            "plastic_metrics": plastic_metrics,
            "consolidated_metrics": consolidated_metrics,
            "geometry_mean": geometry_mean,
            "behavior_match_to_target": behavior_match,
            "anchor_source_counts_after_refresh": anchor_source_counts(anchors),
            "adapter_trace": adapter_trace,
            "consolidation_trace": consolidation_trace,
        }
        stage_results.append(stage_result)
        print(
            "stage consolidated previous={:.5f}/{:.4f} new={:.5f}/{:.4f} bridge={:.5f}/{:.4f} full={:.5f}/{:.4f}".format(
                consolidated_metrics["previous"]["loss"],
                consolidated_metrics["previous"]["token_accuracy"],
                consolidated_metrics["new"]["loss"],
                consolidated_metrics["new"]["token_accuracy"],
                consolidated_metrics["bridge"]["loss"],
                consolidated_metrics["bridge"]["token_accuracy"],
                consolidated_metrics["full"]["loss"],
                consolidated_metrics["full"]["token_accuracy"],
            )
        )
        print(
            "stage geometry target_rel={:.4f} target_cka={:.4f} closeness_gain={:+.4f} fullKL={:.5f} anchors={}".format(
                geometry_mean["current_to_target_drift_rel"],
                geometry_mean["current_to_target_cka"],
                geometry_mean["target_closeness_gain"],
                behavior_match["full"]["kl"],
                anchor_source_counts(anchors),
            )
        )

    final_model = clone_core_from_state(state_dict=current_state, checkpoint=base_checkpoint, device=device)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "base_checkpoint": str(args.base_checkpoint),
            "stage_target_checkpoints": [str(path) for path in target_paths],
            "model_state_dict": final_model.state_dict(),
            "native_gco_config": asdict(NativeGCOConfig(**base_checkpoint["native_gco_config"])),
            "model_config": base_checkpoint["model_config"],
            "anchor_memory": {
                "inputs": anchors["inputs"],
                "targets": anchors["targets"],
                "logits": anchors["logits"],
                "margins": anchors["margins"],
                "source": anchors["source"],
            },
        },
        args.output_checkpoint,
    )
    result = {
        "question": "Can recursive fixed-budget residual rebasing learn multiple chunks without growing inference-time capacity?",
        "base_checkpoint": str(args.base_checkpoint),
        "stage_target_checkpoints": [str(path) for path in target_paths],
        "output_checkpoint": str(args.output_checkpoint),
        "model_config": base_checkpoint["model_config"],
        "source": {
            "chunks_path": str(args.chunks_path),
            "chunk_index": args.chunk_index,
            "chunk_id": str(chunks[args.chunk_index]["chunk_id"]),
            "old_word_start": args.old_word_start,
            "initial_word_count": args.initial_word_count,
            "new_word_count": args.new_word_count,
            "stages": args.stages,
            "seq_len": seq_len,
            "stride": args.stride,
        },
        "anchor_budget": args.anchor_budget,
        "anchor_refresh_selection": args.anchor_refresh_selection,
        "anchor_stratified_counts": {
            "long_term": args.anchor_long_term_count,
            "recent": args.anchor_recent_count,
            "bridge": args.anchor_bridge_count,
            "per_pool_selection": args.anchor_stratified_pool_selection,
        },
        "final_anchor_source_counts": anchor_source_counts(anchors),
        "optimizer": {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "grad_clip": args.grad_clip,
        },
        "loss_weights": {
            "lambda_anchor": args.lambda_anchor,
            "lambda_margin": args.lambda_margin,
            "lambda_bridge": args.lambda_bridge,
            "lambda_adapter": args.lambda_adapter,
            "lambda_adapter_distill": args.lambda_adapter_distill,
            "distill_temperature": args.distill_temperature,
            "margin_slack": args.margin_slack,
        },
        "adapter": {
            "rank": args.adapter_rank,
            "scale": args.adapter_scale,
            "plastic_epochs": args.plastic_epochs,
            "consolidation_epochs": args.consolidation_epochs,
        },
        "stages": stage_results,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nTINY RECURSIVE REBASING SUMMARY")
    print("=" * 112)
    print(
        "{:>5} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
            "stage", "prev_acc", "new_acc", "bridge", "full_acc", "targetR", "targetC", "fullKL"
        )
    )
    for row in stage_results:
        print(
            "{:5d} {:10.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.5f}".format(
                int(row["stage"]),
                row["consolidated_metrics"]["previous"]["token_accuracy"],
                row["consolidated_metrics"]["new"]["token_accuracy"],
                row["consolidated_metrics"]["bridge"]["token_accuracy"],
                row["consolidated_metrics"]["full"]["token_accuracy"],
                row["geometry_mean"]["current_to_target_drift_rel"],
                row["geometry_mean"]["current_to_target_cka"],
                row["behavior_match_to_target"]["full"]["kl"],
            )
        )
    print(f"final_anchor_sources={anchor_source_counts(anchors)}")
    print(f"wrote_checkpoint={args.output_checkpoint}")
    print(f"wrote_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument(
        "--stage-target-checkpoints",
        type=str,
        default=(
            "model/checkpoints/gco-tiny-cl-base-200w-samespec-seed0.pt,"
            "model/checkpoints/gco-tiny-cl-base-300w-samespec-seed0.pt"
        ),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("model/checkpoints/gco-tiny-recursive-rebasing-100to300-seed0.pt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("model/analysis/gco-tiny-recursive-rebasing-100to300-seed0.json"),
    )
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--old-word-start", type=int, default=0)
    parser.add_argument("--initial-word-count", type=int, default=100)
    parser.add_argument("--new-word-count", type=int, default=100)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--anchor-budget", type=int, default=64)
    parser.add_argument("--anchor-refresh-selection", choices=["uniform", "mixed", "stratified"], default="mixed")
    parser.add_argument("--anchor-long-term-count", type=int, default=24)
    parser.add_argument("--anchor-recent-count", type=int, default=24)
    parser.add_argument("--anchor-bridge-count", type=int, default=16)
    parser.add_argument("--anchor-stratified-pool-selection", choices=["uniform", "mixed"], default="mixed")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--max-bridge-windows", type=int, default=128)
    parser.add_argument("--bridge-radius-tokens", type=int, default=0)
    parser.add_argument("--geometry-max-windows", type=int, default=128)
    parser.add_argument("--plastic-epochs", type=int, default=300)
    parser.add_argument("--consolidation-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--lambda-margin", type=float, default=0.1)
    parser.add_argument("--lambda-bridge", type=float, default=1.0)
    parser.add_argument("--lambda-adapter", type=float, default=1e-4)
    parser.add_argument("--lambda-adapter-distill", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--margin-slack", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
