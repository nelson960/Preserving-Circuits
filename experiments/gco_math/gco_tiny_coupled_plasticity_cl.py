"""Coupled trace-plasticity continual-learning experiment.

This is a research diagnostic for a changing plasticity mechanism, not a fixed
role controller. The model has two trainable weight systems:

    theta: low-rank action adapters that alter the frozen tiny transformer
    psi:   a trace-plasticity network that decides how theta should change

The plasticity network is not trained from explicit preserve/drop/guard labels.
It sees evolving trace state: recurrence, conflict evidence, current behavior
loss, capacity pressure, age, and previous trace strength. It emits an
independent write gate, a competing softmax over
protect/guard/drop/decay/compress state actions, and a separate commit gate.
Those gates select action adapters and weight consequence losses. Trace state
is updated after each CL stage from the consequences of the stage.
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
from experiments.gco_math.gco_tiny_auto_role_controller import (
    build_raw_stream,
    count_person_references,
    encode_raw_groups,
    examples_by_stage1_person,
    oracle_roles,
    role_match_report,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import load_checkpoint
from experiments.gco_math.gco_tiny_end_to_end_cl_controller import (
    ActionLogitAdapterModel,
    freeze_module,
    person_for_example,
    teacher_logits_by_person,
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
    encode_examples,
    evaluate_examples,
    make_model_from_config,
    masked_ce_loss,
    possession_examples,
    relation_items,
)


TRACE_FEATURE_NAMES = (
    "loss_norm",
    "exact",
    "token_accuracy",
    "recurrence",
    "conflict",
    "stream_reference",
    "capacity_pressure",
    "strength",
    "protected",
    "age_norm",
    "usefulness",
    "last_gain",
)
TRACE_STATE_GATE_NAMES = (
    "protect",
    "guard",
    "drop",
    "decay",
    "compress",
)
TRACE_GATE_NAMES = ("write",) + TRACE_STATE_GATE_NAMES + ("commit",)
TRACE_GATE_TO_ACTION = {
    "write": "learn",
    "protect": "preserve",
    "guard": "guard",
    "drop": "drop",
}
ACTION_NAME_SET = set(TRACE_GATE_TO_ACTION.values())


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


def parse_people(raw: str, *, name: str) -> set[str]:
    people = {item.strip() for item in raw.split(",") if item.strip()}
    if not people:
        raise ValueError(f"--{name.replace('_', '-')} must contain at least one person.")
    known = set(examples_by_stage1_person())
    unknown = people.difference(known)
    if unknown:
        raise ValueError(f"Unknown stage-1 people in --{name.replace('_', '-')}: {sorted(unknown)}.")
    return people


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    allowed = {"naive", "coupled"}
    unknown = sorted(set(methods).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; allowed={sorted(allowed)}.")
    return methods


def validate_args(args: argparse.Namespace) -> None:
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"Config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    parse_methods(args.methods)
    for name in [
        "stage1_epochs",
        "stage_epochs",
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
        "distill_temperature",
        "loss_clip",
        "drop_target_probability",
        "adapter_init_std",
        "trace_budget",
        "trace_ema",
        "trace_commit_rate",
        "trace_decay_rate",
        "trace_compress_rate",
        "novelty_grace_stages",
        "grad_clip",
    ]:
        positive_float(name, getattr(args, name))
    if args.trace_ema >= 1.0:
        raise ValueError("--trace-ema must be below 1.0.")
    probability("drop_target_probability", args.drop_target_probability)
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")
    for name in [
        "adapter_scale",
        "lambda_new",
        "lambda_protect",
        "lambda_guard",
        "lambda_drop",
        "lambda_capacity",
        "lambda_gate_balance",
        "lambda_write_target",
        "lambda_protect_target",
        "lambda_commit_target",
        "lambda_adapter_norm",
        "lambda_plasticity_norm",
    ]:
        nonnegative_float(name, getattr(args, name))
    if args.adapter_scale == 0.0:
        raise ValueError("--adapter-scale must be non-zero.")


class TracePlasticityNet(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        positive_int("input_dim", input_dim)
        positive_int("hidden_dim", hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(TRACE_STATE_GATE_NAMES) + 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(TRACE_FEATURE_NAMES):
            raise ValueError(
                f"Trace features must have shape [n, {len(TRACE_FEATURE_NAMES)}], got {tuple(features.shape)}."
            )
        raw = self.net(features)
        write_logit = raw[:, :1]
        state_logits = raw[:, 1 : 1 + len(TRACE_STATE_GATE_NAMES)]
        commit_logit = raw[:, 1 + len(TRACE_STATE_GATE_NAMES) :]
        write_gate = torch.sigmoid(write_logit)
        state_gates = F.softmax(state_logits, dim=-1)
        commit_gate = torch.sigmoid(commit_logit)
        gates = torch.cat([write_gate, state_gates, commit_gate], dim=-1)
        if gates.shape != (features.shape[0], len(TRACE_GATE_NAMES)):
            raise RuntimeError(f"Invalid plasticity gate shape {tuple(gates.shape)}.")
        return gates


def make_encoded_trace_people(
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> dict[str, list[EncodedExample]]:
    return {
        item.person: encode_examples(possession_examples(item), tokenizer, max_seq_len=max_seq_len)
        for item in relation_items()
    }


def make_initial_trace_state(
    *,
    people: list[str],
    old_people: set[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count = len(people)
    if count <= 0:
        raise ValueError("Cannot initialize trace state with no people.")
    old_mask = torch.tensor([1.0 if person in old_people else 0.0 for person in people], dtype=torch.float32, device=device)
    return {
        "strength": 0.03 + 0.22 * old_mask,
        "protected": 0.01 + 0.09 * old_mask,
        "usefulness": torch.zeros((count,), dtype=torch.float32, device=device),
        "last_gain": torch.zeros((count,), dtype=torch.float32, device=device),
        "age": torch.zeros((count,), dtype=torch.float32, device=device),
    }


def normalize_counts(values: list[float]) -> list[float]:
    maximum = max(values)
    if maximum <= 0.0:
        return [0.0 for _value in values]
    return [value / maximum for value in values]


def stage_evidence(
    *,
    people: list[str],
    raw_examples: list[QAExample],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
) -> dict[str, torch.Tensor]:
    raw_reference = [float(count_person_references(raw_examples, person)) for person in people]
    recurrence = normalize_counts(raw_reference)
    useful_reference = [
        value if person in useful_evidence_people else 0.0
        for person, value in zip(people, recurrence, strict=True)
    ]
    conflict = [1.0 if person in obsolete_evidence_people else 0.0 for person in people]
    return {
        "recurrence": torch.tensor(recurrence, dtype=torch.float32),
        "stream_reference": torch.tensor(useful_reference, dtype=torch.float32),
        "conflict": torch.tensor(conflict, dtype=torch.float32),
    }


def person_metrics_with_gates(
    *,
    model: ActionLogitAdapterModel,
    people: list[str],
    encoded_people: dict[str, list[EncodedExample]],
    gates: torch.Tensor,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    if gates.shape != (len(people), len(TRACE_GATE_NAMES)):
        raise ValueError(f"Gate shape mismatch: expected {(len(people), len(TRACE_GATE_NAMES))}, got {tuple(gates.shape)}.")
    result: dict[str, dict[str, float]] = {}
    for row_index, person in enumerate(people):
        model.set_action_gates(action_gates_from_trace_row(gates[row_index]))
        result[person] = evaluate_examples(
            model,
            encoded_people[person],
            pad_id=pad_id,
            batch_size=batch_size,
            device=device,
        )["overall"]
    return result


def trace_features(
    *,
    model: ActionLogitAdapterModel,
    plasticity: TracePlasticityNet,
    people: list[str],
    encoded_people: dict[str, list[EncodedExample]],
    trace_state: dict[str, torch.Tensor],
    evidence: dict[str, torch.Tensor],
    pad_id: int,
    batch_size: int,
    device: torch.device,
    loss_clip: float,
    trace_budget: float,
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    positive_float("loss_clip", loss_clip)
    positive_float("trace_budget", trace_budget)
    with torch.no_grad():
        provisional = plasticity(
            torch.stack(
                [
                    torch.zeros(len(people), device=device),
                    torch.ones(len(people), device=device),
                    torch.ones(len(people), device=device),
                    evidence["recurrence"].to(device),
                    evidence["conflict"].to(device),
                    evidence["stream_reference"].to(device),
                    torch.zeros(len(people), device=device),
                    trace_state["strength"],
                    trace_state["protected"],
                    trace_state["age"].clamp(max=10.0) / 10.0,
                    trace_state["usefulness"],
                    trace_state["last_gain"],
                ],
                dim=1,
            )
        )
        metrics = person_metrics_with_gates(
            model=model,
            people=people,
            encoded_people=encoded_people,
            gates=provisional,
            pad_id=pad_id,
            batch_size=batch_size,
            device=device,
        )
    capacity_pressure = (trace_state["strength"].sum() / trace_budget - 1.0).clamp(min=0.0, max=2.0)
    rows: list[list[float]] = []
    for person in people:
        row = metrics[person]
        rows.append(
            [
                min(float(row["loss"]), loss_clip) / loss_clip,
                float(row["exact_match"]),
                float(row["token_accuracy"]),
            ]
        )
    metric_tensor = torch.tensor(rows, dtype=torch.float32, device=device)
    features = torch.cat(
        [
            metric_tensor,
            evidence["recurrence"].to(device).unsqueeze(1),
            evidence["conflict"].to(device).unsqueeze(1),
            evidence["stream_reference"].to(device).unsqueeze(1),
            capacity_pressure.expand(len(people)).unsqueeze(1),
            trace_state["strength"].unsqueeze(1),
            trace_state["protected"].unsqueeze(1),
            (trace_state["age"].clamp(max=10.0) / 10.0).unsqueeze(1),
            trace_state["usefulness"].unsqueeze(1),
            trace_state["last_gain"].unsqueeze(1),
        ],
        dim=1,
    )
    if features.shape != (len(people), len(TRACE_FEATURE_NAMES)):
        raise RuntimeError(f"Built invalid trace feature shape {tuple(features.shape)}.")
    return features, metrics


def action_gates_from_trace_row(row: torch.Tensor) -> dict[str, torch.Tensor]:
    if row.shape != (len(TRACE_GATE_NAMES),):
        raise ValueError(f"Trace gate row shape mismatch: {tuple(row.shape)}.")
    gate_by_name = {name: row[index] for index, name in enumerate(TRACE_GATE_NAMES)}
    return {
        action: gate_by_name[trace_gate]
        for trace_gate, action in TRACE_GATE_TO_ACTION.items()
    }


def single_action_gate(*, action: str, gate: torch.Tensor) -> dict[str, torch.Tensor]:
    if action not in ACTION_NAME_SET:
        raise ValueError(f"Unknown action {action!r}; expected one of {sorted(ACTION_NAME_SET)}.")
    zero = gate.new_zeros(())
    return {
        "learn": gate if action == "learn" else zero,
        "preserve": gate if action == "preserve" else zero,
        "drop": gate if action == "drop" else zero,
        "guard": gate if action == "guard" else zero,
    }


def average_action_gates_for_examples(
    *,
    selected: list[EncodedExample],
    people: list[str],
    gates: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if gates.shape != (len(people), len(TRACE_GATE_NAMES)):
        raise ValueError(f"Gate shape mismatch: expected {(len(people), len(TRACE_GATE_NAMES))}, got {tuple(gates.shape)}.")
    indices: list[int] = []
    for example in selected:
        person = person_for_example(example, people)
        if person is not None:
            indices.append(people.index(person))
    if indices:
        row = gates[torch.tensor(indices, dtype=torch.long, device=gates.device)].mean(dim=0)
    else:
        row = gates.mean(dim=0)
    return action_gates_from_trace_row(row)


def average_trace_gate_for_examples(
    *,
    selected: list[EncodedExample],
    people: list[str],
    gates: torch.Tensor,
    gate_name: str,
) -> torch.Tensor:
    if gate_name not in TRACE_GATE_NAMES:
        raise ValueError(f"Unknown trace gate {gate_name!r}; expected one of {TRACE_GATE_NAMES}.")
    if gates.shape != (len(people), len(TRACE_GATE_NAMES)):
        raise ValueError(f"Gate shape mismatch: expected {(len(people), len(TRACE_GATE_NAMES))}, got {tuple(gates.shape)}.")
    indices: list[int] = []
    for example in selected:
        person = person_for_example(example, people)
        if person is not None:
            indices.append(people.index(person))
    gate_index = TRACE_GATE_NAMES.index(gate_name)
    if indices:
        return gates[torch.tensor(indices, dtype=torch.long, device=gates.device), gate_index].mean()
    return gates[:, gate_index].mean()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError(f"{name} values/weights shape mismatch: {tuple(values.shape)} vs {tuple(weights.shape)}.")
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return values.new_zeros(())
    return (values * weights).sum() / denom


def per_person_consequence_losses(
    *,
    model: ActionLogitAdapterModel,
    people: list[str],
    encoded_people: dict[str, list[EncodedExample]],
    reference_logits: dict[str, list[torch.Tensor]],
    gates: torch.Tensor,
    pad_id: int,
    device: torch.device,
    distill_temperature: float,
    drop_target_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    protect_losses: list[torch.Tensor] = []
    guard_losses: list[torch.Tensor] = []
    drop_losses: list[torch.Tensor] = []
    for row_index, person in enumerate(people):
        examples = encoded_people[person]
        indices = torch.arange(len(examples), dtype=torch.long)
        inputs, targets, mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)

        model.set_action_gates(
            single_action_gate(
                action="preserve",
                gate=gates[row_index, TRACE_GATE_NAMES.index("protect")],
            )
        )
        logits = model(inputs)
        protect_losses.append(
            distillation_loss_for_examples(
                logits,
                selected,
                reference_logits[person],
                indices,
                temperature=distill_temperature,
                device=device,
            )
        )

        model.set_action_gates(
            single_action_gate(
                action="guard",
                gate=gates[row_index, TRACE_GATE_NAMES.index("guard")],
            )
        )
        logits = model(inputs)
        guard_losses.append(
            distillation_loss_for_examples(
                logits,
                selected,
                reference_logits[person],
                indices,
                temperature=distill_temperature,
                device=device,
            )
        )

        model.set_action_gates(
            single_action_gate(
                action="drop",
                gate=gates[row_index, TRACE_GATE_NAMES.index("drop")],
            )
        )
        logits = model(inputs)
        drop_losses.append(
            drop_suppression_loss(
                logits=logits,
                targets=targets,
                mask=mask,
                target_probability=drop_target_probability,
            )
        )
    return torch.stack(protect_losses), torch.stack(guard_losses), torch.stack(drop_losses)


def trainable_parameters(
    *,
    adapter: ActionLogitAdapterModel,
    plasticity: TracePlasticityNet | None,
) -> list[nn.Parameter]:
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if plasticity is not None:
        parameters.extend(parameter for parameter in plasticity.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("No trainable parameters were selected.")
    return parameters


def train_naive_stage(
    *,
    args: argparse.Namespace,
    adapter: ActionLogitAdapterModel,
    stage_examples: list[EncodedExample],
    pad_id: int,
    device: torch.device,
    seed: int,
    label: str,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, args.stage_epochs + 1):
        permutation = torch.randperm(len(stage_examples), generator=generator)
        totals = {"loss": 0.0}
        batches = 0
        pbar = tqdm(range(0, len(stage_examples), args.batch_size), desc=f"{label} {epoch}/{args.stage_epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            adapter.set_action_gates({"learn": 1.0, "preserve": 0.0, "drop": 0.0, "guard": 0.0})
            logits = adapter(inputs)
            loss = masked_ce_loss(logits, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(adapter=adapter, plasticity=None), args.grad_clip)
            optimizer.step()
            value = float(loss.detach().cpu())
            totals["loss"] += value
            batches += 1
            pbar.set_postfix({"loss": f"{value:.3g}"})
        if batches <= 0:
            raise RuntimeError(f"{label} saw zero batches.")
        row = {key: value / float(batches) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        trace.append(row)
    return trace


def train_coupled_stage(
    *,
    args: argparse.Namespace,
    adapter: ActionLogitAdapterModel,
    plasticity: TracePlasticityNet,
    people: list[str],
    encoded_people: dict[str, list[EncodedExample]],
    stage_examples: list[EncodedExample],
    reference_logits: dict[str, list[torch.Tensor]],
    trace_state: dict[str, torch.Tensor],
    evidence: dict[str, torch.Tensor],
    pad_id: int,
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[list[dict[str, float]], torch.Tensor, dict[str, dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        trainable_parameters(adapter=adapter, plasticity=plasticity),
        lr=args.plasticity_lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    latest_gates: torch.Tensor | None = None
    latest_metrics: dict[str, dict[str, float]] | None = None
    for epoch in range(1, args.stage_epochs + 1):
        permutation = torch.randperm(len(stage_examples), generator=generator)
        totals = {
            "loss": 0.0,
            "new": 0.0,
            "protect": 0.0,
            "guard": 0.0,
            "drop": 0.0,
            "capacity": 0.0,
            "write_target": 0.0,
            "commit_target": 0.0,
            "maturity": 0.0,
            "novelty": 0.0,
            "write": 0.0,
            "commit": 0.0,
            "decay": 0.0,
            "compress": 0.0,
        }
        batches = 0
        pbar = tqdm(range(0, len(stage_examples), args.batch_size), desc=f"{label} {epoch}/{args.stage_epochs}")
        for start in pbar:
            features, person_metrics = trace_features(
                model=adapter,
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
            gates = plasticity(features)
            latest_gates = gates.detach()
            latest_metrics = person_metrics

            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, selected = batch_examples(stage_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            write_gate = average_trace_gate_for_examples(
                selected=selected,
                people=people,
                gates=gates,
                gate_name="write",
            )
            adapter.set_action_gates(single_action_gate(action="learn", gate=write_gate))
            logits = adapter(inputs)
            new_loss = masked_ce_loss(logits, targets, mask)

            protect_losses, guard_losses, drop_losses = per_person_consequence_losses(
                model=adapter,
                people=people,
                encoded_people=encoded_people,
                reference_logits=reference_logits,
                gates=gates,
                pad_id=pad_id,
                device=device,
                distill_temperature=args.distill_temperature,
                drop_target_probability=args.drop_target_probability,
            )
            recurrence = evidence["recurrence"].to(device)
            conflict = evidence["conflict"].to(device)
            stream_reference = evidence["stream_reference"].to(device)
            strength = trace_state["strength"].detach()
            protected = trace_state["protected"].detach()
            protect_pressure = (strength * (1.0 - conflict) + stream_reference + protected).clamp(min=0.0)
            guard_pressure = (strength * (1.0 - recurrence) * (1.0 - conflict)).clamp(min=0.0)
            drop_pressure = conflict.clamp(min=0.0)
            protect_loss = weighted_mean(protect_losses, protect_pressure, name="protect")
            guard_loss = weighted_mean(guard_losses, guard_pressure, name="guard")
            drop_loss = weighted_mean(drop_losses, drop_pressure, name="drop")

            gate_by_name = {name: gates[:, index] for index, name in enumerate(TRACE_GATE_NAMES)}
            capacity_pressure = (trace_state["strength"].sum().detach() / args.trace_budget - 1.0).clamp(min=0.0, max=2.0)
            stale_pressure = ((1.0 - recurrence) * (1.0 - conflict) * (1.0 - stream_reference)).clamp(min=0.0)
            novelty_grace = (trace_state["age"].detach() / args.novelty_grace_stages).clamp(min=0.0, max=1.0)
            expected_decay = capacity_pressure * stale_pressure * novelty_grace
            decay_alignment = F.mse_loss(gate_by_name["decay"], expected_decay)
            loss_norm = features[:, TRACE_FEATURE_NAMES.index("loss_norm")].detach()
            token_accuracy = features[:, TRACE_FEATURE_NAMES.index("token_accuracy")].detach()
            useful = (recurrence + stream_reference + trace_state["usefulness"].detach()).clamp(max=1.0)
            non_conflict = (1.0 - conflict).clamp(min=0.0, max=1.0)
            learned_stable = ((1.0 - loss_norm).clamp(min=0.0, max=1.0) * token_accuracy).clamp(min=0.0, max=1.0)
            maturity = (
                trace_state["strength"].detach()
                * trace_state["protected"].detach()
                * learned_stable
            ).clamp(min=0.0, max=1.0).pow(1.0 / 3.0)
            novelty = (1.0 - trace_state["strength"].detach()).clamp(min=0.0, max=1.0)
            write_target = (
                (
                    loss_norm
                    * useful
                    * non_conflict
                    * (1.0 - maturity).clamp(min=0.0, max=1.0)
                )
                + (novelty * recurrence * loss_norm * non_conflict)
            ).clamp(max=1.0).detach()
            write_alignment = F.binary_cross_entropy(
                gate_by_name["write"].clamp(min=1e-6, max=1.0 - 1e-6),
                write_target,
            )
            protect_target = (
                learned_stable
                * useful
                * non_conflict
                * (maturity + recurrence).clamp(max=1.0)
            ).detach()
            guard_target = (
                non_conflict
                * (learned_stable + trace_state["protected"].detach()).clamp(max=1.0)
                * (1.0 - protect_target).clamp(min=0.0, max=1.0)
            ).detach()
            drop_target = conflict.detach()
            decay_target = (
                capacity_pressure
                * stale_pressure
                * (1.0 - useful).clamp(min=0.0, max=1.0)
                * (1.0 - write_target).clamp(min=0.0, max=1.0)
                * novelty_grace
            ).detach()
            compress_target = (
                capacity_pressure
                * learned_stable
                * (1.0 - useful).clamp(min=0.0, max=1.0)
                * non_conflict
                * novelty_grace
            ).detach()
            state_target_raw = torch.stack(
                [protect_target, guard_target, drop_target, decay_target, compress_target],
                dim=1,
            ).clamp_min(1e-6)
            state_target = state_target_raw / state_target_raw.sum(dim=1, keepdim=True)
            state_gates = torch.stack(
                [
                    gate_by_name["protect"],
                    gate_by_name["guard"],
                    gate_by_name["drop"],
                    gate_by_name["decay"],
                    gate_by_name["compress"],
                ],
                dim=1,
            )
            state_alignment = -(state_target * state_gates.clamp_min(1e-6).log()).sum(dim=1).mean()
            commit_target = (learned_stable * useful * non_conflict).detach()
            commit_alignment = F.binary_cross_entropy(
                gate_by_name["commit"].clamp(min=1e-6, max=1.0 - 1e-6),
                commit_target,
            )
            gate_cost = capacity_pressure * (
                gate_by_name["protect"].mean()
                + gate_by_name["guard"].mean()
                + 0.25 * gate_by_name["commit"].mean()
                + gate_by_name["compress"].mean()
            )
            adapter_norm = adapter.adapter_norm()
            plasticity_norm = torch.stack([parameter.square().mean() for parameter in plasticity.parameters()]).sum()
            loss = (
                args.lambda_new * new_loss
                + args.lambda_protect * protect_loss
                + args.lambda_guard * guard_loss
                + args.lambda_drop * drop_loss
                + args.lambda_capacity * gate_cost
                + args.lambda_gate_balance * decay_alignment
                + args.lambda_write_target * write_alignment
                + args.lambda_protect_target * state_alignment
                + args.lambda_commit_target * commit_alignment
                + args.lambda_adapter_norm * adapter_norm
                + args.lambda_plasticity_norm * plasticity_norm
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(adapter=adapter, plasticity=plasticity), args.grad_clip)
            optimizer.step()

            row = {
                "loss": float(loss.detach().cpu()),
                "new": float(new_loss.detach().cpu()),
                "protect": float(protect_loss.detach().cpu()),
                "guard": float(guard_loss.detach().cpu()),
                "drop": float(drop_loss.detach().cpu()),
                "capacity": float(gate_cost.detach().cpu()),
                "write_target": float(write_alignment.detach().cpu()),
                "commit_target": float(commit_alignment.detach().cpu()),
                "maturity": float(maturity.mean().detach().cpu()),
                "novelty": float(novelty.mean().detach().cpu()),
                "write": float(gate_by_name["write"].mean().detach().cpu()),
                "commit": float(gate_by_name["commit"].mean().detach().cpu()),
                "decay": float(gate_by_name["decay"].mean().detach().cpu()),
                "compress": float(gate_by_name["compress"].mean().detach().cpu()),
            }
            for key, value in row.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"loss": f"{row['loss']:.3g}", "new": f"{row['new']:.3g}", "w": f"{row['write']:.2f}"})
        if batches <= 0:
            raise RuntimeError(f"{label} saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        trace.append(epoch_row)
        if epoch == 1 or epoch == args.stage_epochs or epoch % args.print_every == 0:
            print(
                f"{label} epoch={epoch:4d} loss={epoch_row['loss']:.5f} "
                f"new={epoch_row['new']:.5f} protect={epoch_row['protect']:.5f} "
                f"drop={epoch_row['drop']:.5f} write={epoch_row['write']:.3f} "
                f"commit={epoch_row['commit']:.3f} decay={epoch_row['decay']:.3f}"
            )
    if latest_gates is None or latest_metrics is None:
        raise RuntimeError(f"{label} produced no gates.")
    return trace, latest_gates, latest_metrics


def update_trace_state(
    *,
    args: argparse.Namespace,
    trace_state: dict[str, torch.Tensor],
    evidence: dict[str, torch.Tensor],
    gates: torch.Tensor,
    before_metrics: dict[str, dict[str, float]],
    after_metrics: dict[str, dict[str, float]],
    people: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    gate_by_name = {name: gates[:, index].detach() for index, name in enumerate(TRACE_GATE_NAMES)}
    recurrence = evidence["recurrence"].to(device)
    conflict = evidence["conflict"].to(device)
    stream_reference = evidence["stream_reference"].to(device)
    before_loss = torch.tensor([float(before_metrics[p]["loss"]) for p in people], dtype=torch.float32, device=device)
    after_loss = torch.tensor([float(after_metrics[p]["loss"]) for p in people], dtype=torch.float32, device=device)
    gain = (before_loss - after_loss).clamp(min=-args.loss_clip, max=args.loss_clip) / args.loss_clip
    useful = (
        args.trace_ema * trace_state["usefulness"]
        + (1.0 - args.trace_ema) * (stream_reference + recurrence + gain.clamp_min(0.0)).clamp(max=1.0)
    ).clamp(0.0, 1.0)
    strength_delta = (
        args.trace_commit_rate * gate_by_name["commit"] * (stream_reference + recurrence).clamp(max=1.0)
        - args.trace_decay_rate * gate_by_name["decay"] * (1.0 - stream_reference).clamp_min(0.0)
        - args.trace_compress_rate * gate_by_name["compress"] * (1.0 - useful).clamp_min(0.0)
        - args.trace_decay_rate * gate_by_name["drop"] * conflict
    )
    strength = (trace_state["strength"] + strength_delta).clamp(0.0, 1.0)
    protected = (
        trace_state["protected"]
        + args.trace_commit_rate * (gate_by_name["protect"] + gate_by_name["guard"] + gate_by_name["commit"]) / 3.0
        - args.trace_decay_rate * (gate_by_name["decay"] + gate_by_name["drop"]) / 2.0
    ).clamp(0.0, 1.0)
    return {
        "strength": strength.detach(),
        "protected": protected.detach(),
        "usefulness": useful.detach(),
        "last_gain": gain.detach(),
        "age": (trace_state["age"] + 1.0).detach(),
    }


def evaluate_with_trace_gates(
    *,
    model: ActionLogitAdapterModel,
    examples: list[EncodedExample],
    people: list[str],
    gates: torch.Tensor,
    pad_id: int,
    device: torch.device,
) -> dict[str, float]:
    totals = {"loss_sum": 0.0, "token_count": 0.0, "token_correct": 0.0, "example_count": 0.0, "exact_count": 0.0}
    for example_index, example in enumerate(examples):
        model.set_action_gates(average_action_gates_for_examples(selected=[example], people=people, gates=gates))
        inputs, targets, mask, _selected = batch_examples(
            examples,
            indices=torch.tensor([example_index], dtype=torch.long),
            pad_id=pad_id,
            device=device,
        )
        logits = model(inputs)
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
        predictions = logits.argmax(dim=-1)
        token_correct = ((predictions == targets).to(torch.float32) * mask).sum()
        token_count = mask.sum()
        if float(token_count.detach().cpu()) <= 0.0:
            raise RuntimeError(f"Example has no answer tokens: {example.prompt!r}{example.answer!r}")
        exact = float(token_correct.detach().cpu()) == float(token_count.detach().cpu())
        totals["loss_sum"] += float((losses * mask).sum().detach().cpu())
        totals["token_count"] += float(token_count.detach().cpu())
        totals["token_correct"] += float(token_correct.detach().cpu())
        totals["example_count"] += 1.0
        totals["exact_count"] += 1.0 if exact else 0.0
    return {
        "loss": totals["loss_sum"] / totals["token_count"],
        "token_accuracy": totals["token_correct"] / totals["token_count"],
        "exact_match": totals["exact_count"] / totals["example_count"],
        "example_count": totals["example_count"],
    }


def evaluate_group_map(
    *,
    model: ActionLogitAdapterModel,
    groups: dict[str, list[EncodedExample]],
    people: list[str],
    gates: torch.Tensor,
    pad_id: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    return {
        name: evaluate_with_trace_gates(
            model=model,
            examples=examples,
            people=people,
            gates=gates,
            pad_id=pad_id,
            device=device,
        )
        for name, examples in sorted(groups.items())
    }


def predicted_roles_from_trace_gates(*, people: list[str], gates: torch.Tensor) -> dict[str, str]:
    role_indices = {
        "preserve": TRACE_GATE_NAMES.index("protect"),
        "drop": TRACE_GATE_NAMES.index("drop"),
        "guard": TRACE_GATE_NAMES.index("guard"),
    }
    roles = list(role_indices)
    result: dict[str, str] = {}
    for row_index, person in enumerate(people):
        scores = torch.tensor([float(gates[row_index, role_indices[role]].detach().cpu()) for role in roles])
        result[person] = roles[int(scores.argmax().item())]
    return result


def compact(metrics: dict[str, float]) -> str:
    return f"{metrics['loss']:.3g}/{metrics['exact_match']:.3f}"


def make_adapter(
    *,
    args: argparse.Namespace,
    base_model: nn.Module,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> ActionLogitAdapterModel:
    adapter = ActionLogitAdapterModel(
        base_model=base_model,
        d_model=int(checkpoint["model_config"]["d_model"]),
        vocab_size=int(checkpoint["model_config"]["vocab_size"]),
        rank=args.adapter_rank,
        adapter_scale=args.adapter_scale,
        init_std=args.adapter_init_std,
    ).to(device)
    freeze_module(adapter.base_model)
    return adapter


def run_method(
    *,
    args: argparse.Namespace,
    method: str,
    base_model: nn.Module,
    checkpoint: dict[str, Any],
    people: list[str],
    encoded_groups: dict[str, list[EncodedExample]],
    encoded_people: dict[str, list[EncodedExample]],
    raw_groups: dict[str, list[QAExample]],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
    reference_logits: dict[str, list[torch.Tensor]],
    true_roles: dict[str, str],
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    adapter = make_adapter(args=args, base_model=base_model, checkpoint=checkpoint, device=device)
    trace_state = make_initial_trace_state(people=people, old_people=set(true_roles), device=device)
    final_gates = torch.zeros((len(people), len(TRACE_GATE_NAMES)), dtype=torch.float32, device=device)
    stage_reports: list[dict[str, Any]] = []
    if method == "naive":
        final_gates[:, TRACE_GATE_NAMES.index("write")] = 1.0
        for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
            trace = train_naive_stage(
                args=args,
                adapter=adapter,
                stage_examples=encoded_groups[stage_name],
                pad_id=pad_id,
                device=device,
                seed=args.seed + 2000 + stage_number,
                label=f"naive {stage_name}",
            )
            stage_reports.append({"stage": stage_number, "mode": "naive", "trace": trace})
    elif method == "coupled":
        plasticity = TracePlasticityNet(input_dim=len(TRACE_FEATURE_NAMES), hidden_dim=args.plasticity_hidden_dim).to(device)
        for stage_number, stage_name in enumerate(["stage2", "stage3"], start=2):
            evidence = stage_evidence(
                people=people,
                raw_examples=raw_groups[stage_name],
                useful_evidence_people=useful_evidence_people,
                obsolete_evidence_people=obsolete_evidence_people,
            )
            before_features, before_metrics = trace_features(
                model=adapter,
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
            trace, final_gates, _during_metrics = train_coupled_stage(
                args=args,
                adapter=adapter,
                plasticity=plasticity,
                people=people,
                encoded_people=encoded_people,
                stage_examples=encoded_groups[stage_name],
                reference_logits=reference_logits,
                trace_state=trace_state,
                evidence=evidence,
                pad_id=pad_id,
                device=device,
                seed=args.seed + 3000 + stage_number,
                label=f"coupled {stage_name}",
            )
            after_metrics = person_metrics_with_gates(
                model=adapter,
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
            stage_reports.append(
                {
                    "stage": stage_number,
                    "mode": "coupled",
                    "trace": trace,
                    "gates": {
                        person: {
                            name: float(final_gates[index, gate_index].detach().cpu())
                            for gate_index, name in enumerate(TRACE_GATE_NAMES)
                        }
                        for index, person in enumerate(people)
                    },
                    "trace_state": {
                        person: {
                            name: float(value[index].detach().cpu())
                            for name, value in trace_state.items()
                        }
                        for index, person in enumerate(people)
                    },
                }
            )
    else:
        raise ValueError(f"Unknown method {method!r}.")

    metrics = evaluate_group_map(
        model=adapter,
        groups=encoded_groups,
        people=people,
        gates=final_gates,
        pad_id=pad_id,
        device=device,
    )
    if method == "naive":
        role_report = {"accuracy": None, "correct": None, "total": len(people), "confusion": None, "per_person": {}}
        predicted_roles = {person: "none" for person in people}
    else:
        predicted_roles = predicted_roles_from_trace_gates(people=people, gates=final_gates)
        role_report = role_match_report(
            predicted_roles={person: predicted_roles[person] for person in sorted(true_roles)},
            true_roles=true_roles,
        )
    return {
        "method": method,
        "metrics": metrics,
        "role_report": role_report,
        "predicted_roles": predicted_roles,
        "trace_state": {
            person: {
                name: float(value[index].detach().cpu())
                for name, value in trace_state.items()
            }
            for index, person in enumerate(people)
        },
        "final_gates": {
            person: {
                name: float(final_gates[index, gate_index].detach().cpu())
                for gate_index, name in enumerate(TRACE_GATE_NAMES)
            }
            for index, person in enumerate(people)
        },
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
        "preserve": [example for person, examples in encoded_people.items() if person in true_roles and true_roles[person] == "preserve" for example in examples],
        "drop": [example for person, examples in encoded_people.items() if person in true_roles and true_roles[person] == "drop" for example in examples],
        "neutral": [example for person, examples in encoded_people.items() if person in true_roles and true_roles[person] == "guard" for example in examples],
    }
    for group_name in ["stage1", "stage2", "stage3", "preserve", "drop", "neutral", "eval_all"]:
        if group_name not in encoded_groups or not encoded_groups[group_name]:
            raise RuntimeError(f"Encoded group {group_name!r} is empty.")

    base_model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed)
    print("TINY COUPLED TRACE-PLASTICITY CL")
    print("=" * 120)
    print(f"device={device} methods={parse_methods(args.methods)}")
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
    freeze_module(base_model)
    people = sorted(encoded_people)
    reference_logits = teacher_logits_by_person(
        model=base_model,
        encoded_people=encoded_people,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )

    results: dict[str, Any] = {}
    for method in parse_methods(args.methods):
        results[method] = run_method(
            args=args,
            method=method,
            base_model=base_model,
            checkpoint=checkpoint,
            people=people,
            encoded_groups=encoded_groups,
            encoded_people=encoded_people,
            raw_groups=raw_groups,
            useful_evidence_people=useful_evidence_people,
            obsolete_evidence_people=obsolete_evidence_people,
            reference_logits=reference_logits,
            true_roles=true_roles,
            pad_id=pad_id,
            device=device,
        )

    print("\nTINY COUPLED TRACE-PLASTICITY CL SUMMARY")
    print("=" * 120)
    print(
        f"{'method':>10} {'roleAcc':>8} {'preserve':>14} {'drop':>14} {'guard':>14} "
        f"{'stage2':>14} {'stage3':>14} {'eval_all':>14}"
    )
    print(f"{'':>10} {'':>8} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14}")
    for method in parse_methods(args.methods):
        row = results[method]
        metrics = row["metrics"]
        role_acc = row["role_report"]["accuracy"]
        role_text = "NA" if role_acc is None else f"{role_acc:.3f}"
        print(
            f"{method:>10} {role_text:>8} {compact(metrics['preserve']):>14} "
            f"{compact(metrics['drop']):>14} {compact(metrics['neutral']):>14} "
            f"{compact(metrics['stage2']):>14} {compact(metrics['stage3']):>14} "
            f"{compact(metrics['eval_all']):>14}"
        )

    if "coupled" in results:
        print("\nCOUPLED TRACE STATE")
        print("-" * 120)
        for person, state in results["coupled"]["trace_state"].items():
            gates = results["coupled"]["final_gates"][person]
            print(
                f"{person:>6} pred={results['coupled']['predicted_roles'][person]:<8} "
                f"strength={state['strength']:.3f} protected={state['protected']:.3f} "
                f"use={state['usefulness']:.3f} write={gates['write']:.3f} "
                f"protect={gates['protect']:.3f} drop={gates['drop']:.3f} "
                f"decay={gates['decay']:.3f} commit={gates['commit']:.3f}"
            )

    output = {
        "question": "Can changing plasticity weights learn trace hardening, decay, suppression, and protection from evidence over time?",
        "config": {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "trace_feature_names": list(TRACE_FEATURE_NAMES),
            "trace_gate_names": list(TRACE_GATE_NAMES),
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
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-coupled-plasticity-cl-seed0.json"))
    parser.add_argument("--methods", type=str, default="naive,coupled")
    parser.add_argument("--useful-evidence-people", type=str, default="Alice,Bruno")
    parser.add_argument("--obsolete-evidence-people", type=str, default="Clara,Darin")
    parser.add_argument("--reporting-preserve-people", type=str, default="Alice,Bruno")
    parser.add_argument("--reporting-drop-people", type=str, default="Clara,Darin")
    parser.add_argument("--composition-holdout-people", type=str, default="Kira,Luca")
    parser.add_argument("--include-composition-rules", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=120)
    parser.add_argument("--stage-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plasticity-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--adapter-scale", type=float, default=4.0)
    parser.add_argument("--adapter-init-std", type=float, default=0.02)
    parser.add_argument("--plasticity-hidden-dim", type=int, default=64)
    parser.add_argument("--lambda-new", type=float, default=2.0)
    parser.add_argument("--lambda-protect", type=float, default=1.0)
    parser.add_argument("--lambda-guard", type=float, default=1.0)
    parser.add_argument("--lambda-drop", type=float, default=0.4)
    parser.add_argument("--lambda-capacity", type=float, default=0.02)
    parser.add_argument("--lambda-gate-balance", type=float, default=0.05)
    parser.add_argument("--lambda-write-target", type=float, default=0.2)
    parser.add_argument("--lambda-protect-target", type=float, default=0.2)
    parser.add_argument("--lambda-commit-target", type=float, default=0.2)
    parser.add_argument("--lambda-adapter-norm", type=float, default=1e-4)
    parser.add_argument("--lambda-plasticity-norm", type=float, default=1e-5)
    parser.add_argument("--drop-target-probability", type=float, default=0.05)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--loss-clip", type=float, default=8.0)
    parser.add_argument("--trace-budget", type=float, default=2.0)
    parser.add_argument("--trace-ema", type=float, default=0.85)
    parser.add_argument("--trace-commit-rate", type=float, default=0.12)
    parser.add_argument("--trace-decay-rate", type=float, default=0.10)
    parser.add_argument("--trace-compress-rate", type=float, default=0.08)
    parser.add_argument("--novelty-grace-stages", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
