"""Prepare a tiny stable base model and protected geometry anchors for GCO CL.

This script is stage 1 + stage 2 of the CL-only experiment:

1. Fit a very small native GCO transformer on a short real-text slice.
2. Save the fitted checkpoint.
3. Capture old-knowledge probes and module geometry anchors for later CL runs.

The bootstrap optimizer here is only for creating a stable old model. The CL
optimizer experiment should load this checkpoint and perform native GCO writes.
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
from experiments.gco_math.gco_native_scratch_transformer import (
    GCOEmbedding,
    GCOLinear,
    GCONativeTransformer,
    NativeGCOConfig,
    require_finite_tensor,
)


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Chunks file must contain a non-empty list: {path}")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or "chunk_id" not in chunk or "text" not in chunk:
            raise ValueError(f"Chunk at index {index} must contain chunk_id and text.")
    return chunks


def first_words(text: str, word_count: int) -> str:
    positive_int("word_count", word_count)
    words = text.split()
    if len(words) < word_count:
        raise ValueError(f"Requested {word_count} words but source text has only {len(words)} words.")
    return " ".join(words[:word_count])


def build_lm_windows(token_ids: list[int], *, seq_len: int, stride: int, max_windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    positive_int("seq_len", seq_len)
    positive_int("stride", stride)
    positive_int("max_windows", max_windows)
    if len(token_ids) < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} tokens, got {len(token_ids)}.")
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, len(token_ids) - seq_len, stride):
        window = token_ids[start : start + seq_len + 1]
        if len(window) != seq_len + 1:
            raise RuntimeError(f"Window length mismatch at start={start}: {len(window)}.")
        inputs.append(window[:-1])
        targets.append(window[1:])
        if len(inputs) >= max_windows:
            break
    if not inputs:
        raise RuntimeError("No LM windows were built.")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def native_config(args: argparse.Namespace) -> NativeGCOConfig:
    return NativeGCOConfig(
        reasoner_policy="fixed_geometric",
        write_mode="direct",
        direct_write_error_scale="mean",
        min_active_topology=0.05,
        lr=0.1,
        direct_write_ridge=1e-2,
        direct_write_protect=1e-1,
        beta_pressure=0.98,
        beta_formation=0.98,
        beta_protection=0.98,
        beta_decay=0.98,
        beta_usage=0.98,
        beta_capacity=0.98,
        beta_recurrent=0.98,
        reasoner_lr=0.0,
        reasoner_weight_decay=1e-4,
        reasoner_update_clip=1.0,
        internal_gate_lr_scale=1.0,
        internal_write_gate_lr_scale=1.0,
        internal_protect_gate_lr_scale=1.0,
        internal_rewire_gate_lr_scale=1.0,
        internal_forget_gate_lr_scale=1.0,
        internal_compress_gate_lr_scale=1.0,
        state_space_reasoner_dim=0,
        state_space_reasoner_lr=0.0,
        state_space_value_lr=0.0,
        state_space_weight_decay=1e-4,
        state_space_update_clip=1.0,
        state_space_init_std=0.02,
        state_space_credit_beta=0.95,
        state_space_write_blend=0.0,
        state_space_protect_blend=0.0,
        state_space_rewire_blend=0.0,
        state_space_forget_blend=0.0,
        state_space_compress_blend=0.0,
        state_space_priority_blend=0.0,
        outcome_credit_lr=0.0,
        outcome_formation_lr=0.0,
        outcome_baseline_beta=0.95,
        outcome_failure_scale=1e-3,
        route_credit_scale=1e-5,
        route_credit_logit_clip=2.0,
        route_credit_warmup_steps=100,
        route_formation_scale=1e-5,
        route_formation_logit_clip=2.0,
        formation_weight_mix=args.formation_weight_mix,
        formation_row_mix=args.formation_row_mix,
        formation_col_mix=args.formation_col_mix,
        formation_module_mix=args.formation_module_mix,
        formation_multiscale_pooling=args.formation_multiscale_pooling,
        positive_utility_write_floor=0.5,
        protect_old_route_floor=0.25,
        protect_collision_strength=1.0,
        hardening_exit_threshold=0.2,
        hardening_protection_strength=0.0,
        structural_protect_strength=0.0,
        outcome_edit_cost=0.0,
        outcome_capacity_cost=0.0,
        outcome_rewire_cost=0.0,
        outcome_forget_cost=0.0,
        failed_write_beta=0.95,
        gamma=12.0,
        mu=0.5,
        warmup_steps=200,
        recency_tau=200.0,
        grow_lr=0.0,
        prune_lr=0.0,
        forget_lr=0.0,
        max_step_norm=1.0,
        init_topology=args.init_topology,
        hardening_threshold=0.35,
        crystalline_threshold=0.75,
        pathway_percentile=0.99,
        eps=1e-8,
    )


def trainable_base_parameters(model: GCONativeTransformer) -> list[torch.nn.Parameter]:
    params = [module.W for module in model.gco_modules()]
    if not params:
        raise RuntimeError("No trainable GCO module weights found.")
    return params


def make_optimizer(args: argparse.Namespace, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if args.bootstrap_optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    if args.bootstrap_optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    raise ValueError(f"Unknown bootstrap optimizer: {args.bootstrap_optimizer!r}.")


@torch.no_grad()
def initialize_reserved_topology(model: GCONativeTransformer, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    if args.topology_mode == "uniform":
        route_count = 0
        active_count = 0
        reserve_count = 0
        topology_sum = 0.0
        for module in model.gco_modules():
            route_count += module.A.numel()
            active_count += int((module.A >= args.topology_active_threshold).to(dtype=torch.long).sum().detach().cpu())
            reserve_count += int((module.A < args.topology_active_threshold).to(dtype=torch.long).sum().detach().cpu())
            topology_sum += float(module.A.sum().detach().cpu())
        if route_count <= 0:
            raise RuntimeError("No topology routes found in model.")
        return {
            "topology_route_count": float(route_count),
            "topology_active_fraction": float(active_count) / float(route_count),
            "topology_reserve_fraction": float(reserve_count) / float(route_count),
            "topology_mean": topology_sum / float(route_count),
        }
    if args.topology_mode != "bernoulli_reserve":
        raise ValueError(f"Unknown topology_mode: {args.topology_mode!r}.")
    generator = torch.Generator()
    generator.manual_seed(args.seed + args.topology_seed_offset)
    route_count = 0
    active_count = 0
    reserve_count = 0
    topology_sum = 0.0
    for module in model.gco_modules():
        random_values = torch.rand(module.A.shape, dtype=module.A.dtype, generator=generator)
        active = (random_values < args.topology_active_fraction).to(device=device)
        if active.ndim != 2:
            raise RuntimeError(f"{module.name} topology must be a matrix, got shape {active.shape}.")
        for row_index in range(active.shape[0]):
            if not bool(active[row_index].any().detach().cpu()):
                col_index = int(torch.randint(active.shape[1], (1,), generator=generator).item())
                active[row_index, col_index] = True
        for col_index in range(active.shape[1]):
            if not bool(active[:, col_index].any().detach().cpu()):
                row_index = int(torch.randint(active.shape[0], (1,), generator=generator).item())
                active[row_index, col_index] = True
        topology = torch.where(
            active,
            torch.as_tensor(args.topology_active_value, device=device, dtype=module.A.dtype),
            torch.as_tensor(args.topology_reserve_value, device=device, dtype=module.A.dtype),
        )
        module.A.copy_(topology)
        route_count += module.A.numel()
        active_count += int(active.to(dtype=torch.long).sum().detach().cpu())
        reserve_count += int((~active).to(dtype=torch.long).sum().detach().cpu())
        topology_sum += float(module.A.sum().detach().cpu())
    if route_count <= 0:
        raise RuntimeError("No topology routes found in model.")
    return {
        "topology_route_count": float(route_count),
        "topology_active_fraction": float(active_count) / float(route_count),
        "topology_reserve_fraction": float(reserve_count) / float(route_count),
        "topology_mean": topology_sum / float(route_count),
    }


@torch.no_grad()
def evaluate_model(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    positive_int("batch_size", batch_size)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0.0
    total_margin = 0.0
    min_margin: float | None = None
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        batch_targets = targets[start : start + batch_size].to(device)
        logits = model(batch_inputs)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = batch_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
        predictions = flat_logits.argmax(dim=-1)
        top_values, top_indices = torch.topk(flat_logits, k=2, dim=-1)
        competitors = torch.where(top_indices[:, 0] == flat_targets, top_values[:, 1], top_values[:, 0])
        target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        margins = target_logits - competitors
        total_loss += float(loss.detach().cpu())
        total_tokens += int(flat_targets.numel())
        total_correct += float((predictions == flat_targets).to(dtype=torch.float32).sum().detach().cpu())
        total_margin += float(margins.sum().detach().cpu())
        batch_min_margin = float(margins.min().detach().cpu())
        min_margin = batch_min_margin if min_margin is None else min(min_margin, batch_min_margin)
    if total_tokens <= 0:
        raise RuntimeError("Evaluation produced zero target tokens.")
    if min_margin is None:
        raise RuntimeError("Evaluation did not produce margins.")
    return {
        "loss": total_loss / float(total_tokens),
        "token_accuracy": total_correct / float(total_tokens),
        "target_margin_mean": total_margin / float(total_tokens),
        "target_margin_min": min_margin,
    }


@torch.no_grad()
def collect_logits(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    positive_int("batch_size", batch_size)
    logits: list[torch.Tensor] = []
    model.eval()
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        logits.append(model(batch_inputs).detach().cpu())
    if not logits:
        raise RuntimeError("No logits collected.")
    return torch.cat(logits, dim=0)


@torch.no_grad()
def capture_geometry_anchors(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    top_routes: int,
    store_activation_snapshots: bool,
) -> dict[str, Any]:
    positive_int("batch_size", batch_size)
    positive_int("top_routes", top_routes)
    model.eval()
    module_accumulators: dict[str, dict[str, Any]] = {}
    total_batches = 0
    total_windows = 0
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        _ = model(batch_inputs)
        total_batches += 1
        total_windows += int(batch_inputs.shape[0])
        for module in model.gco_modules():
            if isinstance(module, GCOLinear):
                activation = module._y
                if activation is None:
                    raise RuntimeError(f"{module.name} missing linear activation during anchor capture.")
            elif isinstance(module, GCOEmbedding):
                activation = module._y_for_grad
                if activation is None:
                    raise RuntimeError(f"{module.name} missing embedding activation during anchor capture.")
                activation = activation.detach()
            else:
                raise TypeError(f"Unsupported GCO module type: {type(module).__name__}.")
            pathway = module.pathway().detach().cpu()
            activation_cpu = activation.detach().cpu()
            flattened = activation_cpu.reshape(-1, activation_cpu.shape[-1])
            accumulator = module_accumulators.setdefault(
                module.name,
                {
                    "pathway_sum": torch.zeros_like(pathway),
                    "activation_sum": torch.zeros(flattened.shape[-1], dtype=torch.float32),
                    "activation_sq_sum": torch.zeros(flattened.shape[-1], dtype=torch.float32),
                    "activation_count": 0,
                    "snapshots": [],
                    "weight_effective": (module.W.detach().cpu() * module.A.detach().cpu()).clone(),
                    "topology": module.A.detach().cpu().clone(),
                },
            )
            accumulator["pathway_sum"].add_(pathway)
            accumulator["activation_sum"].add_(flattened.sum(dim=0).to(dtype=torch.float32))
            accumulator["activation_sq_sum"].add_((flattened * flattened).sum(dim=0).to(dtype=torch.float32))
            accumulator["activation_count"] += int(flattened.shape[0])
            if store_activation_snapshots:
                accumulator["snapshots"].append(activation_cpu)
    if total_batches <= 0:
        raise RuntimeError("Anchor capture saw zero batches.")

    modules: dict[str, dict[str, Any]] = {}
    for name, accumulator in module_accumulators.items():
        activation_count = int(accumulator["activation_count"])
        if activation_count <= 0:
            raise RuntimeError(f"{name} activation count is zero.")
        pathway_mean = accumulator["pathway_sum"] / float(total_batches)
        require_finite_tensor(f"{name}_anchor_pathway_mean", pathway_mean)
        flat_pathway = pathway_mean.reshape(-1)
        k = min(top_routes, int(flat_pathway.numel()))
        values, indices = torch.topk(flat_pathway, k=k)
        rows = torch.div(indices, pathway_mean.shape[1], rounding_mode="floor")
        cols = indices.remainder(pathway_mean.shape[1])
        activation_mean = accumulator["activation_sum"] / float(activation_count)
        activation_rms = torch.sqrt(accumulator["activation_sq_sum"] / float(activation_count))
        module_entry: dict[str, Any] = {
            "pathway_mean": pathway_mean,
            "activation_mean": activation_mean,
            "activation_rms": activation_rms,
            "activation_count": activation_count,
            "weight_effective": accumulator["weight_effective"],
            "topology": accumulator["topology"],
            "top_routes": {
                "rows": rows,
                "cols": cols,
                "values": values,
            },
        }
        if store_activation_snapshots:
            module_entry["activation_snapshots"] = torch.cat(accumulator["snapshots"], dim=0)
        modules[name] = module_entry
    return {
        "total_batches": total_batches,
        "total_windows": total_windows,
        "modules": modules,
    }


def top_routes_for_json(anchor_bank: dict[str, Any]) -> dict[str, list[dict[str, float | int]]]:
    result: dict[str, list[dict[str, float | int]]] = {}
    for name, module_entry in anchor_bank["modules"].items():
        top_routes = module_entry["top_routes"]
        rows = top_routes["rows"].tolist()
        cols = top_routes["cols"].tolist()
        values = top_routes["values"].tolist()
        result[name] = [
            {"row": int(row), "col": int(col), "value": float(value)}
            for row, col, value in zip(rows, cols, values, strict=True)
        ]
    return result


def validate_args(args: argparse.Namespace) -> None:
    positive_int("seed", args.seed + 1)
    positive_int("word_count", args.word_count)
    positive_int("seq_len", args.seq_len)
    positive_int("stride", args.stride)
    positive_int("max_windows", args.max_windows)
    positive_int("epochs", args.epochs)
    positive_int("batch_size", args.batch_size)
    positive_float("lr", args.lr)
    nonnegative_float("momentum", args.momentum)
    positive_float("grad_clip", args.grad_clip)
    if args.momentum >= 1.0:
        raise ValueError(f"momentum must be less than 1, got {args.momentum}.")
    positive_float("target_loss", args.target_loss)
    positive_int("d_model", args.d_model)
    positive_int("n_layers", args.n_layers)
    positive_int("n_heads", args.n_heads)
    positive_int("d_ff", args.d_ff)
    if args.d_model % args.n_heads != 0:
        raise ValueError(f"d_model={args.d_model} must be divisible by n_heads={args.n_heads}.")
    if not (0.0 < args.init_topology <= 1.0):
        raise ValueError(f"init_topology must be in (0,1], got {args.init_topology}.")
    if args.topology_mode not in {"uniform", "bernoulli_reserve"}:
        raise ValueError(f"Unknown topology_mode: {args.topology_mode!r}.")
    if not (0.0 < args.topology_active_fraction < 1.0):
        raise ValueError(f"topology_active_fraction must be in (0,1), got {args.topology_active_fraction}.")
    if not (0.0 <= args.topology_reserve_value < args.topology_active_value <= 1.0):
        raise ValueError(
            "Expected 0 <= topology_reserve_value < topology_active_value <= 1, got "
            f"reserve={args.topology_reserve_value}, active={args.topology_active_value}."
        )
    if not (0.0 <= args.topology_active_threshold <= 1.0):
        raise ValueError(f"topology_active_threshold must be in [0,1], got {args.topology_active_threshold}.")
    if args.topology_seed_offset < 0:
        raise ValueError(f"topology_seed_offset must be non-negative, got {args.topology_seed_offset}.")
    positive_int("top_routes", args.top_routes)
    if args.bootstrap_optimizer not in {"sgd", "adam"}:
        raise ValueError(f"Unknown bootstrap optimizer: {args.bootstrap_optimizer!r}.")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file does not exist: {args.tokenizer_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.chunk_index < 0 or args.chunk_index >= len(chunks):
        raise ValueError(f"chunk_index={args.chunk_index} outside chunk count {len(chunks)}.")
    base_text = first_words(str(chunks[args.chunk_index]["text"]), args.word_count)
    token_ids = tokenizer.encode(base_text).ids
    inputs, targets = build_lm_windows(token_ids, seq_len=args.seq_len, stride=args.stride, max_windows=args.max_windows)
    vocab_size = tokenizer.get_vocab_size()
    cfg = native_config(args)
    model = GCONativeTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        cfg=cfg,
    ).to(device)
    topology_summary = initialize_reserved_topology(model, args, device)
    optimizer = make_optimizer(args, trainable_base_parameters(model))

    print("GCO TINY CL BASE PREPARATION")
    print("=" * 112)
    print(
        f"device={device} optimizer={args.bootstrap_optimizer} lr={args.lr:g} "
        f"words={args.word_count} tokens={len(token_ids)} windows={inputs.shape[0]} seq_len={args.seq_len}"
    )
    print(
        f"model d={args.d_model} layers={args.n_layers} heads={args.n_heads} d_ff={args.d_ff} "
        f"topology={args.init_topology:g} params={sum(parameter.numel() for parameter in trainable_base_parameters(model))}"
    )
    print(
        "topology_mode={} active={:.4f} reserve={:.4f} mean={:.4f}".format(
            args.topology_mode,
            topology_summary["topology_active_fraction"],
            topology_summary["topology_reserve_fraction"],
            topology_summary["topology_mean"],
        )
    )

    train_trace: list[dict[str, float | int]] = []
    best_loss = float("inf")
    final_metrics = evaluate_model(model, inputs, targets, batch_size=args.batch_size, device=device)
    first_loss = final_metrics["loss"]
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(inputs.shape[0])
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        pbar = tqdm(range(0, inputs.shape[0], args.batch_size), desc=f"epoch {epoch}/{args.epochs}")
        for start in pbar:
            step += 1
            indices = permutation[start : start + args.batch_size]
            batch_inputs = inputs[indices].to(device)
            batch_targets = targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_base_parameters(model), args.grad_clip)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            token_count = int(batch_targets.numel())
            epoch_loss_sum += loss_value * float(token_count)
            epoch_tokens += token_count
            pbar.set_postfix({"loss": f"{loss_value:.5f}"})
        if epoch_tokens <= 0:
            raise RuntimeError(f"Epoch {epoch} saw zero tokens.")
        final_metrics = evaluate_model(model, inputs, targets, batch_size=args.batch_size, device=device)
        best_loss = min(best_loss, final_metrics["loss"])
        row = {
            "epoch": epoch,
            "mean_train_loss": epoch_loss_sum / float(epoch_tokens),
            "eval_loss": final_metrics["loss"],
            "token_accuracy": final_metrics["token_accuracy"],
            "target_margin_mean": final_metrics["target_margin_mean"],
            "target_margin_min": final_metrics["target_margin_min"],
        }
        train_trace.append(row)
        print(
            "epoch={epoch:4d} train_loss={mean_train_loss:.6f} eval_loss={eval_loss:.6f} "
            "acc={token_accuracy:.4f} margin={target_margin_mean:.4f}/{target_margin_min:.4f}".format(**row)
        )
        if final_metrics["loss"] <= args.target_loss:
            print(f"early_stop target_loss reached: {final_metrics['loss']:.6f} <= {args.target_loss:.6f}")
            break

    final_logits = collect_logits(model, inputs, batch_size=args.batch_size, device=device)
    anchor_bank = capture_geometry_anchors(
        model,
        inputs,
        batch_size=args.batch_size,
        device=device,
        top_routes=args.top_routes,
        store_activation_snapshots=args.store_activation_snapshots,
    )

    checkpoint = {
        "schema_version": 1,
        "model_state_dict": model.state_dict(),
        "native_gco_config": asdict(cfg),
        "model_config": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "max_seq_len": args.seq_len,
        },
        "source": {
            "chunks_path": str(args.chunks_path),
            "chunk_index": args.chunk_index,
            "chunk_id": str(chunks[args.chunk_index]["chunk_id"]),
            "word_count": args.word_count,
            "text": base_text,
            "token_ids": token_ids,
        },
        "probe_inputs": inputs,
        "probe_targets": targets,
        "final_metrics": final_metrics,
        "topology_summary": topology_summary,
    }
    anchors = {
        "schema_version": 1,
        "checkpoint_path": str(args.checkpoint_path),
        "tokenizer_path": str(args.tokenizer_path),
        "source": checkpoint["source"],
        "probe_inputs": inputs,
        "probe_targets": targets,
        "probe_logits": final_logits,
        "final_metrics": final_metrics,
        "anchor_bank": anchor_bank,
    }

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    args.anchors_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint_path)
    torch.save(anchors, args.anchors_path)

    result = {
        "question": "Can we create a stable old tiny transformer and protected geometry anchors for CL?",
        "checkpoint_path": str(args.checkpoint_path),
        "anchors_path": str(args.anchors_path),
        "tokenizer_path": str(args.tokenizer_path),
        "source": checkpoint["source"],
        "bootstrap_optimizer": args.bootstrap_optimizer,
        "topology_summary": topology_summary,
        "first_loss": first_loss,
        "best_loss": best_loss,
        "final_metrics": final_metrics,
        "train_trace": train_trace,
        "anchor_summary": {
            "total_batches": anchor_bank["total_batches"],
            "total_windows": anchor_bank["total_windows"],
            "module_count": len(anchor_bank["modules"]),
            "top_routes": top_routes_for_json(anchor_bank),
        },
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nTINY CL BASE SUMMARY")
    print("=" * 112)
    print(
        "loss {:.6f} -> {:.6f} best={:.6f} acc={:.4f} margin={:.4f}/{:.4f}".format(
            first_loss,
            final_metrics["loss"],
            best_loss,
            final_metrics["token_accuracy"],
            final_metrics["target_margin_mean"],
            final_metrics["target_margin_min"],
        )
    )
    print(f"wrote_checkpoint={args.checkpoint_path}")
    print(f"wrote_anchors={args.anchors_path}")
    print(f"wrote_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-seed0.pt"))
    parser.add_argument("--anchors-path", type=Path, default=Path("model/analysis/gco-tiny-cl-base-anchors-seed0.pt"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-cl-base-seed0.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--word-count", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-loss", type=float, default=0.01)
    parser.add_argument("--bootstrap-optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--init-topology", type=float, default=1.0)
    parser.add_argument("--topology-mode", choices=["uniform", "bernoulli_reserve"], default="uniform")
    parser.add_argument("--topology-active-fraction", type=float, default=0.7)
    parser.add_argument("--topology-active-value", type=float, default=1.0)
    parser.add_argument("--topology-reserve-value", type=float, default=0.0)
    parser.add_argument("--topology-active-threshold", type=float, default=0.5)
    parser.add_argument("--topology-seed-offset", type=int, default=1009)
    parser.add_argument("--formation-weight-mix", type=float, default=1.0)
    parser.add_argument("--formation-row-mix", type=float, default=0.0)
    parser.add_argument("--formation-col-mix", type=float, default=0.0)
    parser.add_argument("--formation-module-mix", type=float, default=0.0)
    parser.add_argument("--formation-multiscale-pooling", choices=["mean", "max"], default="max")
    parser.add_argument("--top-routes", type=int, default=8)
    parser.add_argument("--store-activation-snapshots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
