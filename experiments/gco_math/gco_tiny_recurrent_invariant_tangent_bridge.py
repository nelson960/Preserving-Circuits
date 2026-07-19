"""Bridge recurrent trace decisions into an invariant-tangent update.

This experiment separates two questions:

    1. Can recurrent trace plasticity choose preserve / guard / drop roles?
    2. If those roles are used as constraints, can a projected tangent update
       learn new data better than the selector's weak adapter-write path?

The recurrent selector is trained on the same staged toy stream as
gco_tiny_recurrent_trace_plasticity.py. The preferred bridge mode does not
collapse its output into one symbolic role. Instead, it uses the continuous
trace gates directly:

    raw gradient = write-weighted new CE + drop-weighted suppression
    protected constraints = protect/guard/commit-weighted distillation rows
    update = projected raw gradient + bounded restore gradient

The older role-collapsed path is still available for comparison with
--bridge-update-mode roles.

This is still a toy adapter-space bridge, not the final full-model optimizer.
It tests whether the recurrent selector can feed the invariant-tangent math.
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
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_mini_cl_world_demo import (
    assign_flat_gradient,
    distillation_loss_and_constraint_rows_for_batch,
    flat_autograd_gradient,
    project_gradient_away_from_constraints,
)
from experiments.gco_math.gco_tiny_auto_role_controller import (
    build_raw_stream,
    encode_raw_groups,
    oracle_roles,
    role_match_report,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import distillation_kl, load_checkpoint
from experiments.gco_math.gco_tiny_end_to_end_cl_controller import person_for_example, teacher_logits_by_person
from experiments.gco_math.gco_tiny_recurrent_trace_plasticity import (
    RecurrentTracePlasticityNet,
    TRACE_FEATURE_NAMES,
    TRACE_GATE_NAMES,
    TRACE_PREDICTION_NAMES,
    compact,
    make_adapter,
    make_encoded_trace_people,
    make_initial_trace_state,
    parse_people,
    person_metrics_with_gates,
    predicted_roles_from_trace_gates,
    serialize_trace_state,
    stage_evidence,
    trace_features,
    train_naive_stage,
    train_recurrent_stage,
    update_trace_state,
)
from experiments.gco_math.gco_tiny_recursive_controlled_architecture import (
    drop_suppression_loss,
    train_bootstrap_stage,
)
from experiments.gco_math.gco_tiny_reusable_computation_cl import (
    EncodedExample,
    QAExample,
    batch_examples,
    collect_example_logits,
    encode_examples,
    evaluate_examples,
    make_model_from_config,
    masked_ce_loss,
)


BRIDGE_METHODS = {"naive", "recurrent_bridge"}


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def parse_bridge_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    unknown = sorted(set(methods).difference(BRIDGE_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; allowed={sorted(BRIDGE_METHODS)}.")
    return methods


def validate_args(args: argparse.Namespace) -> None:
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"Config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    parse_bridge_methods(args.methods)
    for name in [
        "stage1_epochs",
        "stage_epochs",
        "bridge_epochs",
        "batch_size",
        "eval_batch_size",
        "adapter_rank",
        "plasticity_hidden_dim",
        "print_every",
    ]:
        positive_int(name, getattr(args, name))
    for name in [
        "lr",
        "plasticity_lr",
        "bridge_lr",
        "distill_temperature",
        "loss_clip",
        "drop_target_probability",
        "adapter_init_std",
        "trace_budget",
        "trace_ema",
        "trace_commit_rate",
        "trace_candidate_rate",
        "trace_decay_rate",
        "trace_compress_rate",
        "candidate_write_gain",
        "novelty_grace_stages",
        "grad_clip",
        "projected_update_damping",
        "plasticity_audit_rank_tolerance",
        "restore_bound_fraction",
        "min_bridge_weight_sum",
    ]:
        positive_float(name, getattr(args, name))
    for name in [
        "adapter_scale",
        "write_gate_scale",
        "protect_gate_scale",
        "guard_gate_scale",
        "commit_gate_scale",
        "drop_gate_scale",
        "unassigned_write_weight",
        "lambda_new",
        "lambda_protect",
        "lambda_guard",
        "lambda_drop",
        "lambda_capacity",
        "lambda_gate_balance",
        "lambda_write_target",
        "lambda_protect_target",
        "lambda_commit_target",
        "lambda_consequence_prediction",
        "lambda_adapter_norm",
        "lambda_plasticity_norm",
        "projected_restore_strength",
        "weight_decay",
    ]:
        nonnegative_float(name, getattr(args, name))
    if args.trace_ema >= 1.0:
        raise ValueError("--trace-ema must be below 1.0.")
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")


def learn_action_gates() -> dict[str, float]:
    return {"learn": 1.0, "preserve": 0.0, "drop": 0.0, "guard": 0.0}


def trainable_adapter_parameters(adapter: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("Bridge adapter has no trainable parameters.")
    return params


def examples_for_roles(
    *,
    roles: dict[str, str],
    encoded_people: dict[str, list[EncodedExample]],
    role: str,
) -> list[EncodedExample]:
    selected = [
        example
        for person, person_role in sorted(roles.items())
        if person_role == role
        for example in encoded_people[person]
    ]
    if not selected:
        raise RuntimeError(f"Predicted role group {role!r} is empty; cannot build bridge constraints.")
    return selected


def evaluate_learn_adapter(
    *,
    adapter: torch.nn.Module,
    groups: dict[str, list[EncodedExample]],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    adapter.set_action_gates(learn_action_gates())
    return {
        name: evaluate_examples(
            adapter,
            examples,
            pad_id=pad_id,
            batch_size=batch_size,
            device=device,
        )["overall"]
        for name, examples in sorted(groups.items())
    }


def sample_encoded_batch(
    *,
    examples: list[EncodedExample],
    count: int,
    generator: torch.Generator,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[EncodedExample], torch.Tensor]:
    indices = torch.randint(
        low=0,
        high=len(examples),
        size=(count,),
        generator=generator,
        device=torch.device("cpu"),
    )
    inputs, targets, mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
    return inputs, targets, mask, selected, indices


def per_example_gate_weights(
    *,
    selected: list[EncodedExample],
    people: list[str],
    gates: torch.Tensor,
    gate_scales: dict[str, float],
    unassigned_weight: float | None,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if gates.shape != (len(people), len(TRACE_GATE_NAMES)):
        raise ValueError(f"{name} gate shape mismatch: expected {(len(people), len(TRACE_GATE_NAMES))}, got {tuple(gates.shape)}.")
    if not selected:
        raise RuntimeError(f"{name} selected batch is empty.")
    weights: list[torch.Tensor] = []
    gate_indices = {gate_name: TRACE_GATE_NAMES.index(gate_name) for gate_name in gate_scales}
    for example in selected:
        person = person_for_example(example, people)
        if person is None:
            if unassigned_weight is None:
                raise RuntimeError(f"{name} cannot assign gate weight to unpersoned example: {example.prompt!r}{example.answer!r}")
            weights.append(gates.new_tensor(float(unassigned_weight)))
            continue
        person_index = people.index(person)
        weight = gates.new_zeros(())
        for gate_name, scale in gate_scales.items():
            weight = weight + float(scale) * gates[person_index, gate_indices[gate_name]].detach()
        weights.append(weight.clamp_min(0.0))
    return torch.stack(weights).to(device=device)


def weighted_masked_ce_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    example_weights: torch.Tensor,
    min_weight_sum: float,
    name: str,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError(f"{name} shape mismatch: logits={logits.shape} targets={targets.shape} mask={mask.shape}.")
    if example_weights.shape != (targets.shape[0],):
        raise ValueError(f"{name} expected example weights shape {(targets.shape[0],)}, got {tuple(example_weights.shape)}.")
    token_weights = mask * example_weights.reshape(-1, 1)
    denom = token_weights.sum()
    if float(denom.detach().cpu()) <= min_weight_sum:
        raise RuntimeError(f"{name} has insufficient positive token weight: {float(denom.detach().cpu()):.6g}.")
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
    return (losses * token_weights).sum() / denom


def weighted_drop_suppression_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    example_weights: torch.Tensor,
    target_probability: float,
    min_weight_sum: float,
    name: str,
) -> torch.Tensor:
    positive_float("target_probability", target_probability)
    if target_probability >= 1.0:
        raise ValueError(f"target_probability must be below 1.0, got {target_probability}.")
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError(f"{name} shape mismatch: logits={logits.shape} targets={targets.shape} mask={mask.shape}.")
    if example_weights.shape != (targets.shape[0],):
        raise ValueError(f"{name} expected example weights shape {(targets.shape[0],)}, got {tuple(example_weights.shape)}.")
    token_weights = mask * example_weights.reshape(-1, 1)
    denom = token_weights.sum()
    if float(denom.detach().cpu()) <= min_weight_sum:
        raise RuntimeError(f"{name} has insufficient positive token weight: {float(denom.detach().cpu()):.6g}.")
    log_probs = F.log_softmax(logits, dim=-1)
    old_answer_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    threshold = math.log(target_probability)
    penalty = F.relu(old_answer_log_probs - threshold).square()
    return (penalty * token_weights).sum() / denom


def aggregate_weighted_losses_by_category(
    losses: list[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    prefix: str,
    min_weight_sum: float,
) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for category, loss, weight in losses:
        grouped.setdefault(category, []).append((loss, weight))
    rows: dict[str, torch.Tensor] = {}
    for category, category_losses in sorted(grouped.items()):
        loss_values = torch.stack([loss for loss, _weight in category_losses])
        weights = torch.stack([weight for _loss, weight in category_losses])
        denom = weights.sum()
        if float(denom.detach().cpu()) <= min_weight_sum:
            continue
        rows[f"{prefix}:{category}"] = (loss_values * weights).sum() / denom
    if not rows:
        raise RuntimeError(f"No weighted constraint rows remained for {prefix}.")
    return rows


def weighted_distillation_loss_and_constraint_rows_for_batch(
    current_logits: torch.Tensor,
    selected: list[EncodedExample],
    teacher_logits: list[torch.Tensor],
    global_indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float,
    device: torch.device,
    constraint_mode: str,
    prefix: str,
    min_weight_sum: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weights.shape != (len(selected),):
        raise ValueError(f"{prefix} weights shape mismatch: expected {(len(selected),)}, got {tuple(weights.shape)}.")
    losses: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for row_index, example in enumerate(selected):
        length = len(example.target_ids)
        current = current_logits[row_index, :length].unsqueeze(0)
        teacher = teacher_logits[int(global_indices[row_index].item())].to(device).unsqueeze(0)
        losses.append((example.category, distillation_kl(current, teacher, temperature=temperature), weights[row_index]))
    if not losses:
        raise RuntimeError(f"No distillation losses were built for {prefix}.")
    weight_values = torch.stack([weight for _category, _loss, weight in losses])
    denom = weight_values.sum()
    if float(denom.detach().cpu()) <= min_weight_sum:
        raise RuntimeError(f"{prefix} has insufficient positive constraint weight: {float(denom.detach().cpu()):.6g}.")
    loss_values = torch.stack([loss for _category, loss, _weight in losses])
    overall = (loss_values * weight_values).sum() / denom
    if constraint_mode == "scalar":
        return overall, {f"{prefix}:all": overall}
    if constraint_mode != "category":
        raise ValueError(f"Gate-weighted bridge currently supports scalar/category constraints, got {constraint_mode!r}.")
    return overall, aggregate_weighted_losses_by_category(losses, prefix=prefix, min_weight_sum=min_weight_sum)


def bounded_restore_gradient(
    *,
    restore_gradient: torch.Tensor,
    safe_gradient: torch.Tensor,
    restore_strength: float,
    bound_fraction: float,
) -> torch.Tensor:
    if restore_strength <= 0.0:
        return torch.zeros_like(safe_gradient)
    scaled = restore_strength * restore_gradient
    scaled_norm = torch.linalg.vector_norm(scaled)
    safe_norm = torch.linalg.vector_norm(safe_gradient)
    limit = bound_fraction * safe_norm
    scale = torch.minimum(scaled.new_tensor(1.0), limit / scaled_norm.clamp_min(1e-12))
    return scaled * scale


def train_projected_bridge_stage(
    *,
    args: argparse.Namespace,
    adapter: torch.nn.Module,
    update_examples: list[EncodedExample],
    preserve_examples: list[EncodedExample],
    guard_examples: list[EncodedExample],
    drop_examples: list[EncodedExample],
    preserve_logits: list[torch.Tensor],
    guard_logits: list[torch.Tensor],
    pad_id: int,
    device: torch.device,
    seed: int,
    label: str,
) -> list[dict[str, float]]:
    if not update_examples:
        raise ValueError(f"{label} received no update examples.")
    positive_int("bridge_epochs", args.bridge_epochs)
    params = trainable_adapter_parameters(adapter)
    optimizer = torch.optim.AdamW(params, lr=args.bridge_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, args.bridge_epochs + 1):
        permutation = torch.randperm(len(update_examples), generator=generator)
        totals = {
            "loss": 0.0,
            "new": 0.0,
            "drop": 0.0,
            "preserve": 0.0,
            "guard": 0.0,
            "raw_grad_norm": 0.0,
            "projected_grad_norm": 0.0,
            "restore_grad_norm": 0.0,
            "projection_removed_fraction": 0.0,
            "safe_grad_fraction": 0.0,
            "constraint_count": 0.0,
        }
        batches = 0
        pbar = tqdm(range(0, len(update_examples), args.batch_size), desc=f"{label} {epoch}/{args.bridge_epochs}")
        for start in pbar:
            update_indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(
                update_examples,
                indices=update_indices,
                pad_id=pad_id,
                device=device,
            )
            batch_count = int(update_indices.numel())
            preserve_inputs, _preserve_targets, _preserve_mask, preserve_selected, preserve_indices = sample_encoded_batch(
                examples=preserve_examples,
                count=batch_count,
                generator=generator,
                pad_id=pad_id,
                device=device,
            )
            guard_inputs, _guard_targets, _guard_mask, guard_selected, guard_indices = sample_encoded_batch(
                examples=guard_examples,
                count=batch_count,
                generator=generator,
                pad_id=pad_id,
                device=device,
            )
            drop_inputs, drop_targets, drop_mask, _drop_selected, _drop_indices = sample_encoded_batch(
                examples=drop_examples,
                count=batch_count,
                generator=generator,
                pad_id=pad_id,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            adapter.set_action_gates(learn_action_gates())
            update_logits = adapter(inputs)
            new_loss = masked_ce_loss(update_logits, targets, mask)
            drop_logits = adapter(drop_inputs)
            drop_loss = drop_suppression_loss(
                logits=drop_logits,
                targets=drop_targets,
                mask=drop_mask,
                target_probability=args.drop_target_probability,
            )
            raw_loss = args.lambda_new * new_loss + args.lambda_drop * drop_loss
            preserve_logits_current = adapter(preserve_inputs)
            guard_logits_current = adapter(guard_inputs)
            preserve_loss, preserve_rows = distillation_loss_and_constraint_rows_for_batch(
                preserve_logits_current,
                preserve_selected,
                preserve_logits,
                preserve_indices,
                temperature=args.distill_temperature,
                device=device,
                constraint_mode=args.projected_constraint_mode,
                prefix="preserve_behavior",
            )
            guard_loss, guard_rows = distillation_loss_and_constraint_rows_for_batch(
                guard_logits_current,
                guard_selected,
                guard_logits,
                guard_indices,
                temperature=args.distill_temperature,
                device=device,
                constraint_mode=args.projected_constraint_mode,
                prefix="guard_behavior",
            )
            constraint_rows = {**preserve_rows, **guard_rows}
            constraint_loss = args.lambda_protect * preserve_loss + args.lambda_guard * guard_loss
            raw_gradient = flat_autograd_gradient(
                raw_loss,
                params,
                retain_graph=True,
                require_nonzero=True,
                label=f"{label}:raw_loss",
            )
            constraint_gradients = [
                flat_autograd_gradient(
                    loss,
                    params,
                    retain_graph=True,
                    require_nonzero=False,
                    label=f"{label}:{name}",
                )
                for name, loss in sorted(constraint_rows.items())
            ]
            safe_gradient, projection_stats = project_gradient_away_from_constraints(
                raw_gradient=raw_gradient,
                constraint_gradients=constraint_gradients,
                damping=args.projected_update_damping,
                solver=args.projected_solver,
                rank_tolerance=args.plasticity_audit_rank_tolerance,
                plasticity_audit=args.projected_plasticity_audit,
            )
            restore_gradient = flat_autograd_gradient(
                constraint_loss,
                params,
                retain_graph=False,
                require_nonzero=False,
                label=f"{label}:restore",
            )
            restore_update = bounded_restore_gradient(
                restore_gradient=restore_gradient,
                safe_gradient=safe_gradient,
                restore_strength=args.projected_restore_strength,
                bound_fraction=args.restore_bound_fraction,
            )
            final_gradient = safe_gradient + restore_update
            optimizer.zero_grad(set_to_none=True)
            assign_flat_gradient(params, final_gradient)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip).detach().cpu())
            optimizer.step()

            row = {
                "loss": float((raw_loss + constraint_loss).detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "drop": float(drop_loss.detach().cpu()),
                "preserve": float(preserve_loss.detach().cpu()),
                "guard": float(guard_loss.detach().cpu()),
                "raw_grad_norm": projection_stats["raw_grad_norm"],
                "projected_grad_norm": projection_stats["projected_grad_norm"],
                "restore_grad_norm": float(torch.linalg.vector_norm(restore_update).detach().cpu()),
                "projection_removed_fraction": projection_stats["projection_removed_fraction"],
                "safe_grad_fraction": projection_stats["safe_grad_fraction"],
                "constraint_count": projection_stats["constraint_count"],
                "grad_norm": grad_norm,
            }
            for key in totals:
                totals[key] += row[key]
            batches += 1
            pbar.set_postfix(
                {
                    "new": f"{row['new']:.3g}",
                    "drop": f"{row['drop']:.3g}",
                    "rem": f"{row['projection_removed_fraction']:.2f}",
                }
            )
        if batches <= 0:
            raise RuntimeError(f"{label} epoch {epoch} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        trace.append(epoch_row)
        if epoch == 1 or epoch == args.bridge_epochs or epoch % args.print_every == 0:
            print(
                f"{label} epoch={epoch:4d} new={epoch_row['new']:.5f} "
                f"drop={epoch_row['drop']:.5f} preserve={epoch_row['preserve']:.5f} "
                f"guard={epoch_row['guard']:.5f} removed={epoch_row['projection_removed_fraction']:.3f} "
                f"restore={epoch_row['restore_grad_norm']:.4g}"
            )
    return trace


def train_gate_weighted_bridge_stage(
    *,
    args: argparse.Namespace,
    adapter: torch.nn.Module,
    update_examples: list[EncodedExample],
    memory_examples: list[EncodedExample],
    memory_logits: list[torch.Tensor],
    gates: torch.Tensor,
    people: list[str],
    pad_id: int,
    device: torch.device,
    seed: int,
    label: str,
) -> list[dict[str, float]]:
    if not update_examples:
        raise ValueError(f"{label} received no update examples.")
    if not memory_examples:
        raise ValueError(f"{label} received no memory examples.")
    if len(memory_examples) != len(memory_logits):
        raise ValueError(f"{label} memory/logit count mismatch: {len(memory_examples)} vs {len(memory_logits)}.")
    params = trainable_adapter_parameters(adapter)
    optimizer = torch.optim.AdamW(params, lr=args.bridge_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    protect_scales = {
        "protect": args.protect_gate_scale,
        "guard": args.guard_gate_scale,
        "commit": args.commit_gate_scale,
    }
    write_scales = {"write": args.write_gate_scale}
    drop_scales = {"drop": args.drop_gate_scale}
    trace: list[dict[str, float]] = []
    for epoch in range(1, args.bridge_epochs + 1):
        permutation = torch.randperm(len(update_examples), generator=generator)
        totals = {
            "loss": 0.0,
            "new": 0.0,
            "drop": 0.0,
            "protected": 0.0,
            "raw_grad_norm": 0.0,
            "projected_grad_norm": 0.0,
            "restore_grad_norm": 0.0,
            "projection_removed_fraction": 0.0,
            "safe_grad_fraction": 0.0,
            "constraint_count": 0.0,
            "write_weight_mean": 0.0,
            "drop_weight_mean": 0.0,
            "protect_weight_mean": 0.0,
        }
        batches = 0
        pbar = tqdm(range(0, len(update_examples), args.batch_size), desc=f"{label} {epoch}/{args.bridge_epochs}")
        for start in pbar:
            update_indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, selected = batch_examples(
                update_examples,
                indices=update_indices,
                pad_id=pad_id,
                device=device,
            )
            batch_count = int(update_indices.numel())
            memory_inputs, memory_targets, memory_mask, memory_selected, memory_indices = sample_encoded_batch(
                examples=memory_examples,
                count=batch_count,
                generator=generator,
                pad_id=pad_id,
                device=device,
            )
            write_weights = per_example_gate_weights(
                selected=selected,
                people=people,
                gates=gates,
                gate_scales=write_scales,
                unassigned_weight=args.unassigned_write_weight,
                device=device,
                name=f"{label}:write",
            )
            protect_weights = per_example_gate_weights(
                selected=memory_selected,
                people=people,
                gates=gates,
                gate_scales=protect_scales,
                unassigned_weight=None,
                device=device,
                name=f"{label}:protect",
            )
            drop_weights = per_example_gate_weights(
                selected=memory_selected,
                people=people,
                gates=gates,
                gate_scales=drop_scales,
                unassigned_weight=None,
                device=device,
                name=f"{label}:drop",
            )

            optimizer.zero_grad(set_to_none=True)
            adapter.set_action_gates(learn_action_gates())
            update_logits = adapter(inputs)
            new_loss = weighted_masked_ce_loss(
                logits=update_logits,
                targets=targets,
                mask=mask,
                example_weights=write_weights,
                min_weight_sum=args.min_bridge_weight_sum,
                name=f"{label}:new",
            )
            memory_logits_current = adapter(memory_inputs)
            drop_loss = weighted_drop_suppression_loss(
                logits=memory_logits_current,
                targets=memory_targets,
                mask=memory_mask,
                example_weights=drop_weights,
                target_probability=args.drop_target_probability,
                min_weight_sum=args.min_bridge_weight_sum,
                name=f"{label}:drop",
            )
            protected_loss, protected_rows = weighted_distillation_loss_and_constraint_rows_for_batch(
                memory_logits_current,
                memory_selected,
                memory_logits,
                memory_indices,
                protect_weights,
                temperature=args.distill_temperature,
                device=device,
                constraint_mode=args.projected_constraint_mode,
                prefix="protected_behavior",
                min_weight_sum=args.min_bridge_weight_sum,
            )
            raw_loss = args.lambda_new * new_loss + args.lambda_drop * drop_loss
            constraint_loss = (args.lambda_protect + args.lambda_guard) * 0.5 * protected_loss
            raw_gradient = flat_autograd_gradient(
                raw_loss,
                params,
                retain_graph=True,
                require_nonzero=True,
                label=f"{label}:raw_loss",
            )
            constraint_gradients = [
                flat_autograd_gradient(
                    loss,
                    params,
                    retain_graph=True,
                    require_nonzero=False,
                    label=f"{label}:{name}",
                )
                for name, loss in sorted(protected_rows.items())
            ]
            safe_gradient, projection_stats = project_gradient_away_from_constraints(
                raw_gradient=raw_gradient,
                constraint_gradients=constraint_gradients,
                damping=args.projected_update_damping,
                solver=args.projected_solver,
                rank_tolerance=args.plasticity_audit_rank_tolerance,
                plasticity_audit=args.projected_plasticity_audit,
            )
            restore_gradient = flat_autograd_gradient(
                constraint_loss,
                params,
                retain_graph=False,
                require_nonzero=False,
                label=f"{label}:restore",
            )
            restore_update = bounded_restore_gradient(
                restore_gradient=restore_gradient,
                safe_gradient=safe_gradient,
                restore_strength=args.projected_restore_strength,
                bound_fraction=args.restore_bound_fraction,
            )
            final_gradient = safe_gradient + restore_update
            optimizer.zero_grad(set_to_none=True)
            assign_flat_gradient(params, final_gradient)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip).detach().cpu())
            optimizer.step()

            row = {
                "loss": float((raw_loss + constraint_loss).detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "drop": float(drop_loss.detach().cpu()),
                "protected": float(protected_loss.detach().cpu()),
                "raw_grad_norm": projection_stats["raw_grad_norm"],
                "projected_grad_norm": projection_stats["projected_grad_norm"],
                "restore_grad_norm": float(torch.linalg.vector_norm(restore_update).detach().cpu()),
                "projection_removed_fraction": projection_stats["projection_removed_fraction"],
                "safe_grad_fraction": projection_stats["safe_grad_fraction"],
                "constraint_count": projection_stats["constraint_count"],
                "write_weight_mean": float(write_weights.mean().detach().cpu()),
                "drop_weight_mean": float(drop_weights.mean().detach().cpu()),
                "protect_weight_mean": float(protect_weights.mean().detach().cpu()),
                "grad_norm": grad_norm,
            }
            for key in totals:
                totals[key] += row[key]
            batches += 1
            pbar.set_postfix(
                {
                    "new": f"{row['new']:.3g}",
                    "drop": f"{row['drop']:.3g}",
                    "prot": f"{row['protected']:.3g}",
                    "rem": f"{row['projection_removed_fraction']:.2f}",
                }
            )
        if batches <= 0:
            raise RuntimeError(f"{label} epoch {epoch} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        trace.append(epoch_row)
        if epoch == 1 or epoch == args.bridge_epochs or epoch % args.print_every == 0:
            print(
                f"{label} epoch={epoch:4d} new={epoch_row['new']:.5f} "
                f"drop={epoch_row['drop']:.5f} protected={epoch_row['protected']:.5f} "
                f"removed={epoch_row['projection_removed_fraction']:.3f} "
                f"writeW={epoch_row['write_weight_mean']:.3f} protectW={epoch_row['protect_weight_mean']:.3f} "
                f"dropW={epoch_row['drop_weight_mean']:.3f}"
            )
    return trace


def role_groups_from_predicted(
    *,
    predicted_roles: dict[str, str],
    encoded_people: dict[str, list[EncodedExample]],
) -> dict[str, list[EncodedExample]]:
    try:
        return {
            role: examples_for_roles(roles=predicted_roles, encoded_people=encoded_people, role=role)
            for role in ["preserve", "guard", "drop"]
        }
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} predicted_roles={predicted_roles}") from exc


def run_recurrent_bridge(
    *,
    args: argparse.Namespace,
    base_model: torch.nn.Module,
    checkpoint: dict[str, Any],
    people: list[str],
    encoded_groups: dict[str, list[EncodedExample]],
    encoded_people: dict[str, list[EncodedExample]],
    raw_groups: dict[str, list[QAExample]],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
    reference_logits_by_person: dict[str, list[torch.Tensor]],
    true_roles: dict[str, str],
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    selector_adapter = make_adapter(args=args, base_model=base_model, checkpoint=checkpoint, device=device)
    bridge_adapter = make_adapter(args=args, base_model=base_model, checkpoint=checkpoint, device=device)
    plasticity = RecurrentTracePlasticityNet(
        input_dim=len(TRACE_FEATURE_NAMES),
        hidden_dim=args.plasticity_hidden_dim,
        candidate_write_gain=args.candidate_write_gain,
    ).to(device)
    trace_state = make_initial_trace_state(
        people=people,
        old_people=set(true_roles),
        hidden_dim=args.plasticity_hidden_dim,
        device=device,
    )
    memory_examples = [example for person in people for example in encoded_people[person]]
    if not memory_examples:
        raise RuntimeError("No memory examples were available for gate-weighted bridge update.")
    memory_logits = collect_example_logits(
        base_model,
        memory_examples,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    final_gates = torch.zeros((len(people), len(TRACE_GATE_NAMES)), dtype=torch.float32, device=device)
    final_predictions = torch.zeros((len(people), len(TRACE_PREDICTION_NAMES)), dtype=torch.float32, device=device)
    stage_reports: list[dict[str, Any]] = []
    for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
        evidence = stage_evidence(
            people=people,
            raw_examples=raw_groups[stage_name],
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
        )
        before_features, before_metrics = trace_features(
            model=selector_adapter,
            plasticity=plasticity,
            people=people,
            encoded_people=encoded_people,
            trace_state=trace_state,
            evidence=evidence,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
            loss_clip=args.loss_clip,
            trace_budget=args.trace_budget,
        )
        del before_features
        selector_trace, final_gates, final_predictions, _during_metrics = train_recurrent_stage(
            args=args,
            adapter=selector_adapter,
            plasticity=plasticity,
            people=people,
            encoded_people=encoded_people,
            stage_examples=encoded_groups[stage_name],
            reference_logits=reference_logits_by_person,
            trace_state=trace_state,
            evidence=evidence,
            pad_id=pad_id,
            device=device,
            seed=args.seed + 4000 + stage_number,
            label=f"selector {stage_name}",
        )
        after_metrics = person_metrics_with_gates(
            model=selector_adapter,
            people=people,
            encoded_people=encoded_people,
            gates=final_gates,
            pad_id=pad_id,
            batch_size=args.eval_batch_size,
            device=device,
        )
        trace_state = update_trace_state(
            args=args,
            trace_state=trace_state,
            evidence=evidence,
            gates=final_gates,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            people=people,
            device=device,
        )
        predicted_roles = predicted_roles_from_trace_gates(people=people, gates=final_gates)
        stage_report = {
            "stage": stage_number,
            "stage_name": stage_name,
            "predicted_roles": predicted_roles,
            "role_report": role_match_report(
                predicted_roles={person: predicted_roles[person] for person in sorted(true_roles)},
                true_roles=true_roles,
            ),
            "selector_trace": selector_trace,
            "bridge_trace": None,
            "bridge_role_timing": args.bridge_role_timing,
            "bridge_update_mode": args.bridge_update_mode,
            "trace_state": serialize_trace_state(people=people, trace_state=trace_state),
        }
        if args.bridge_role_timing == "online":
            if args.bridge_update_mode == "roles":
                role_groups = role_groups_from_predicted(predicted_roles=predicted_roles, encoded_people=encoded_people)
                preserve_logits = collect_example_logits(
                    base_model,
                    role_groups["preserve"],
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                guard_logits = collect_example_logits(
                    base_model,
                    role_groups["guard"],
                    pad_id=pad_id,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                stage_report["bridge_trace"] = train_projected_bridge_stage(
                    args=args,
                    adapter=bridge_adapter,
                    update_examples=encoded_groups[stage_name],
                    preserve_examples=role_groups["preserve"],
                    guard_examples=role_groups["guard"],
                    drop_examples=role_groups["drop"],
                    preserve_logits=preserve_logits,
                    guard_logits=guard_logits,
                    pad_id=pad_id,
                    device=device,
                    seed=args.seed + 5000 + stage_number,
                    label=f"bridge {stage_name}",
                )
            elif args.bridge_update_mode == "gates":
                stage_report["bridge_trace"] = train_gate_weighted_bridge_stage(
                    args=args,
                    adapter=bridge_adapter,
                    update_examples=encoded_groups[stage_name],
                    memory_examples=memory_examples,
                    memory_logits=memory_logits,
                    gates=final_gates,
                    people=people,
                    pad_id=pad_id,
                    device=device,
                    seed=args.seed + 5000 + stage_number,
                    label=f"bridge {stage_name}",
                )
            else:
                raise ValueError(f"Unknown bridge_update_mode={args.bridge_update_mode!r}.")
        stage_reports.append(stage_report)
    predicted_roles = predicted_roles_from_trace_gates(people=people, gates=final_gates)
    if args.bridge_role_timing == "warmup":
        if args.bridge_update_mode == "roles":
            role_groups = role_groups_from_predicted(predicted_roles=predicted_roles, encoded_people=encoded_people)
            preserve_logits = collect_example_logits(
                base_model,
                role_groups["preserve"],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
            guard_logits = collect_example_logits(
                base_model,
                role_groups["guard"],
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            )
        for stage_report in stage_reports:
            stage_number = int(stage_report["stage"])
            stage_name = str(stage_report["stage_name"])
            if args.bridge_update_mode == "roles":
                stage_report["bridge_trace"] = train_projected_bridge_stage(
                    args=args,
                    adapter=bridge_adapter,
                    update_examples=encoded_groups[stage_name],
                    preserve_examples=role_groups["preserve"],
                    guard_examples=role_groups["guard"],
                    drop_examples=role_groups["drop"],
                    preserve_logits=preserve_logits,
                    guard_logits=guard_logits,
                    pad_id=pad_id,
                    device=device,
                    seed=args.seed + 5000 + stage_number,
                    label=f"bridge {stage_name}",
                )
            elif args.bridge_update_mode == "gates":
                stage_report["bridge_trace"] = train_gate_weighted_bridge_stage(
                    args=args,
                    adapter=bridge_adapter,
                    update_examples=encoded_groups[stage_name],
                    memory_examples=memory_examples,
                    memory_logits=memory_logits,
                    gates=final_gates,
                    people=people,
                    pad_id=pad_id,
                    device=device,
                    seed=args.seed + 5000 + stage_number,
                    label=f"bridge {stage_name}",
                )
            else:
                raise ValueError(f"Unknown bridge_update_mode={args.bridge_update_mode!r}.")
    metrics = evaluate_learn_adapter(
        adapter=bridge_adapter,
        groups=encoded_groups,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    return {
        "method": "recurrent_bridge",
        "metrics": metrics,
        "predicted_roles": predicted_roles,
        "role_report": role_match_report(
            predicted_roles={person: predicted_roles[person] for person in sorted(true_roles)},
            true_roles=true_roles,
        ),
        "final_gates": {
            person: {
                name: float(final_gates[index, gate_index].detach().cpu())
                for gate_index, name in enumerate(TRACE_GATE_NAMES)
            }
            for index, person in enumerate(people)
        },
        "final_predictions": {
            person: {
                name: float(final_predictions[index, prediction_index].detach().cpu())
                for prediction_index, name in enumerate(TRACE_PREDICTION_NAMES)
            }
            for index, person in enumerate(people)
        },
        "trace_state": serialize_trace_state(people=people, trace_state=trace_state),
        "stages": stage_reports,
    }


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
    useful_evidence_people = parse_people(args.useful_evidence_people, name="useful_evidence_people")
    obsolete_evidence_people = parse_people(args.obsolete_evidence_people, name="obsolete_evidence_people")
    preserve_people = parse_people(args.reporting_preserve_people, name="reporting_preserve_people")
    drop_people = parse_people(args.reporting_drop_people, name="reporting_drop_people")
    true_roles = oracle_roles(preserve_people=preserve_people, drop_people=drop_people)
    composition_holdout_people = {item.strip() for item in args.composition_holdout_people.split(",") if item.strip()}
    if not composition_holdout_people:
        raise ValueError("--composition-holdout-people must not be empty.")
    raw_groups = build_raw_stream(
        useful_evidence_people=useful_evidence_people,
        composition_holdout_people=composition_holdout_people,
        include_composition_rules=args.include_composition_rules,
    )
    encoded_base_groups = encode_raw_groups(raw_groups, tokenizer, max_seq_len=max_seq_len)
    encoded_people = make_encoded_trace_people(tokenizer=tokenizer, max_seq_len=max_seq_len)
    encoded_groups = {
        **encoded_base_groups,
        "preserve": [
            example
            for person, examples in encoded_people.items()
            if person in true_roles and true_roles[person] == "preserve"
            for example in examples
        ],
        "drop": [
            example
            for person, examples in encoded_people.items()
            if person in true_roles and true_roles[person] == "drop"
            for example in examples
        ],
        "neutral": [
            example
            for person, examples in encoded_people.items()
            if person in true_roles and true_roles[person] == "guard"
            for example in examples
        ],
    }
    for group_name in ["stage1", "stage2", "stage3", "preserve", "drop", "neutral", "eval_all"]:
        if group_name not in encoded_groups or not encoded_groups[group_name]:
            raise RuntimeError(f"Encoded group {group_name!r} is empty.")

    base_model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed)
    print("TINY RECURRENT -> INVARIANT-TANGENT BRIDGE")
    print("=" * 120)
    print(f"device={device} methods={parse_bridge_methods(args.methods)}")
    print(
        f"stage1={len(encoded_groups['stage1'])} stage2={len(encoded_groups['stage2'])} "
        f"stage3={len(encoded_groups['stage3'])} eval={len(encoded_groups['eval_all'])}"
    )
    train_bootstrap_stage(
        args=args,
        model=base_model,
        stage_examples=encoded_groups["stage1"],
        pad_id=pad_id,
        device=device,
        epochs=args.stage1_epochs,
        seed=args.seed + 1000,
        label="base stage1",
    )
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    people = sorted(encoded_people)
    reference_logits_by_person = teacher_logits_by_person(
        model=base_model,
        encoded_people=encoded_people,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )

    results: dict[str, Any] = {}
    if "naive" in parse_bridge_methods(args.methods):
        naive_adapter = make_adapter(args=args, base_model=base_model, checkpoint=checkpoint, device=device)
        for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
            train_naive_stage(
                args=args,
                adapter=naive_adapter,
                stage_examples=encoded_groups[stage_name],
                pad_id=pad_id,
                device=device,
                seed=args.seed + 2000 + stage_number,
                label=f"naive {stage_name}",
            )
        naive_adapter.set_action_gates(learn_action_gates())
        results["naive"] = {
            "method": "naive",
            "metrics": evaluate_learn_adapter(
                adapter=naive_adapter,
                groups=encoded_groups,
                pad_id=pad_id,
                batch_size=args.eval_batch_size,
                device=device,
            ),
            "role_report": {"accuracy": None},
        }
    if "recurrent_bridge" in parse_bridge_methods(args.methods):
        results["recurrent_bridge"] = run_recurrent_bridge(
            args=args,
            base_model=base_model,
            checkpoint=checkpoint,
            people=people,
            encoded_groups=encoded_groups,
            encoded_people=encoded_people,
            raw_groups=raw_groups,
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
            reference_logits_by_person=reference_logits_by_person,
            true_roles=true_roles,
            pad_id=pad_id,
            device=device,
        )

    print("\nTINY RECURRENT -> INVARIANT-TANGENT BRIDGE SUMMARY")
    print("=" * 120)
    print(
        f"{'method':>18} {'roleAcc':>8} {'preserve':>14} {'drop':>14} {'guard':>14} "
        f"{'stage2':>14} {'stage3':>14} {'eval_all':>14}"
    )
    print(
        f"{'':>18} {'':>8} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14} "
        f"{'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14}"
    )
    for method in parse_bridge_methods(args.methods):
        row = results[method]
        metrics = row["metrics"]
        role_acc = row["role_report"].get("accuracy")
        role_text = "NA" if role_acc is None else f"{role_acc:.3f}"
        print(
            f"{method:>18} {role_text:>8} {compact(metrics['preserve']):>14} "
            f"{compact(metrics['drop']):>14} {compact(metrics['neutral']):>14} "
            f"{compact(metrics['stage2']):>14} {compact(metrics['stage3']):>14} "
            f"{compact(metrics['eval_all']):>14}"
        )

    if "recurrent_bridge" in results:
        print("\nRECURRENT BRIDGE TRACE STATE")
        print("-" * 120)
        bridge = results["recurrent_bridge"]
        for person, state in bridge["trace_state"].items():
            gates = bridge["final_gates"][person]
            predictions = bridge["final_predictions"][person]
            print(
                f"{person:>6} pred={bridge['predicted_roles'][person]:<8} "
                f"strength={state['strength']:.3f} known={state['known']:.3f} "
                f"stable={state['stability']:.3f} write={gates['write']:.3f} "
                f"protect={gates['protect']:.3f} drop={gates['drop']:.3f} "
                f"commit={gates['commit']:.3f} predKnown={predictions['known']:.3f} "
                f"gain={predictions['gain']:.3f}"
            )

    output = {
        "question": "Can recurrent trace decisions drive an invariant-tangent style projected update?",
        "config": {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "true_roles_for_reporting": true_roles,
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-recurrent-invariant-bridge-seed0.json"))
    parser.add_argument("--methods", type=str, default="naive,recurrent_bridge")
    parser.add_argument("--bridge-role-timing", choices=["warmup", "online"], default="warmup")
    parser.add_argument("--bridge-update-mode", choices=["gates", "roles"], default="gates")
    parser.add_argument("--useful-evidence-people", type=str, default="Alice,Bruno")
    parser.add_argument("--obsolete-evidence-people", type=str, default="Clara,Darin")
    parser.add_argument("--reporting-preserve-people", type=str, default="Alice,Bruno")
    parser.add_argument("--reporting-drop-people", type=str, default="Clara,Darin")
    parser.add_argument("--composition-holdout-people", type=str, default="Kira,Luca")
    parser.add_argument("--include-composition-rules", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=120)
    parser.add_argument("--stage-epochs", type=int, default=120)
    parser.add_argument("--bridge-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plasticity-lr", type=float, default=1e-3)
    parser.add_argument("--bridge-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--adapter-scale", type=float, default=4.0)
    parser.add_argument("--adapter-init-std", type=float, default=0.02)
    parser.add_argument("--plasticity-hidden-dim", type=int, default=64)
    parser.add_argument("--write-gate-scale", type=float, default=1.0)
    parser.add_argument("--protect-gate-scale", type=float, default=1.0)
    parser.add_argument("--guard-gate-scale", type=float, default=1.0)
    parser.add_argument("--commit-gate-scale", type=float, default=1.0)
    parser.add_argument("--drop-gate-scale", type=float, default=1.0)
    parser.add_argument("--unassigned-write-weight", type=float, default=1.0)
    parser.add_argument("--min-bridge-weight-sum", type=float, default=1e-6)
    parser.add_argument("--lambda-new", type=float, default=4.0)
    parser.add_argument("--lambda-protect", type=float, default=1.0)
    parser.add_argument("--lambda-guard", type=float, default=1.0)
    parser.add_argument("--lambda-drop", type=float, default=0.4)
    parser.add_argument("--lambda-capacity", type=float, default=0.02)
    parser.add_argument("--lambda-gate-balance", type=float, default=0.05)
    parser.add_argument("--lambda-write-target", type=float, default=0.2)
    parser.add_argument("--lambda-protect-target", type=float, default=0.2)
    parser.add_argument("--lambda-commit-target", type=float, default=0.2)
    parser.add_argument("--lambda-consequence-prediction", type=float, default=1.0)
    parser.add_argument("--lambda-adapter-norm", type=float, default=1e-4)
    parser.add_argument("--lambda-plasticity-norm", type=float, default=1e-5)
    parser.add_argument("--drop-target-probability", type=float, default=0.05)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--loss-clip", type=float, default=8.0)
    parser.add_argument("--trace-budget", type=float, default=2.0)
    parser.add_argument("--trace-ema", type=float, default=0.85)
    parser.add_argument("--trace-commit-rate", type=float, default=0.12)
    parser.add_argument("--trace-candidate-rate", type=float, default=0.08)
    parser.add_argument("--trace-decay-rate", type=float, default=0.10)
    parser.add_argument("--trace-compress-rate", type=float, default=0.08)
    parser.add_argument("--candidate-write-gain", type=float, default=3.0)
    parser.add_argument("--novelty-grace-stages", type=float, default=2.0)
    parser.add_argument("--projected-constraint-mode", choices=["scalar", "category"], default="category")
    parser.add_argument("--projected-solver", choices=["sequential", "gram"], default="gram")
    parser.add_argument("--projected-plasticity-audit", action="store_true")
    parser.add_argument("--projected-update-damping", type=float, default=1e-6)
    parser.add_argument("--projected-restore-strength", type=float, default=0.05)
    parser.add_argument("--restore-bound-fraction", type=float, default=0.5)
    parser.add_argument("--plasticity-audit-rank-tolerance", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
