"""Trace-conditioned attention operator for continual task-weight updates.

The task transformer owns all factual content. The plasticity mechanism stores
only trace metadata keyed by task activations. Each incoming observation first
attends to its temporal metadata trace, then the retrieved statistical context
conditions global attention across every task parameter tensor.

No trace stores target logits or answers. No event or action labels are used in
training. The operator is optimized across randomized worlds through future
prediction loss and remains fixed during held-out continual-learning episodes;
only task weights and recurrent metadata state change online.

This tests the missing hierarchy:

    observation -> trace metadata attention -> parameter attention -> weight update

Invariant-Tangent projection and bounded restore are intentionally deferred
until this operator demonstrates conditional plasticity beyond ordinary SGD.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_tiny_plasticity_attention_operator import (
    PARAMETER_FEATURE_NAMES,
    OperatorState,
    apply_probabilistic_update,
    apply_sgd_update,
    build_parameter_features,
    functional_task_forward,
    parameter_delta_norm,
    task_parameters,
    update_operator_state,
)
from experiments.gco_math.gco_tiny_probabilistic_plasticity_attention import (
    ACTION_INDEX,
    ACTION_NAMES,
    StreamEvent,
    TinyQueryTransformer,
    centered_kernel_alignment,
    generate_episode,
)


OBSERVATION_FEATURE_NAMES = (
    "loss_log",
    "observed_probability",
    "prediction_entropy",
    "prediction_margin",
    "gradient_rms_mean_log",
    "gradient_recurrence_mean",
    "gradient_conflict_mean",
)


@dataclass
class TraceMetadataState:
    keys: torch.Tensor
    hidden: torch.Tensor
    mass: torch.Tensor
    age: torch.Tensor


class TraceMetadataAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int,
        evidence_dim: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        recurrent_input_dim = evidence_dim + 3
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(recurrent_input_dim),
            nn.Linear(recurrent_input_dim, hidden_dim),
            nn.GELU(),
        )
        self.recurrent = nn.GRUCell(hidden_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)

    def initial_state(self, *, keys: torch.Tensor) -> TraceMetadataState:
        if keys.ndim != 2 or keys.shape[1] != self.d_model:
            raise ValueError(f"Trace keys must be [traces, {self.d_model}], got {tuple(keys.shape)}.")
        return TraceMetadataState(
            keys=keys,
            hidden=torch.zeros(keys.shape[0], self.hidden_dim, device=keys.device, dtype=keys.dtype),
            mass=torch.zeros(keys.shape[0], device=keys.device, dtype=keys.dtype),
            age=torch.zeros(keys.shape[0], device=keys.device, dtype=keys.dtype),
        )

    def attend(self, query: torch.Tensor, state: TraceMetadataState) -> torch.Tensor:
        if query.shape != (1, self.d_model):
            raise ValueError(f"Trace query must be [1, {self.d_model}], got {tuple(query.shape)}.")
        eps = torch.finfo(query.dtype).eps
        projected_query = F.normalize(self.query_projection(query), dim=1, eps=eps)
        projected_keys = F.normalize(self.key_projection(state.keys), dim=1, eps=eps)
        temperature = F.softplus(self.log_temperature) + eps
        scores = projected_query @ projected_keys.transpose(0, 1) / temperature
        return torch.softmax(scores, dim=1)

    def update(
        self,
        *,
        state: TraceMetadataState,
        attention: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[TraceMetadataState, torch.Tensor]:
        if attention.shape != (1, state.keys.shape[0]):
            raise ValueError(
                f"Trace attention must be {(1, state.keys.shape[0])}, got {tuple(attention.shape)}."
            )
        if evidence.ndim != 1:
            raise ValueError(f"Trace evidence must be one-dimensional, got {tuple(evidence.shape)}.")
        attention_column = attention.transpose(0, 1)
        mass_scale = state.mass.sum().clamp_min(torch.finfo(state.mass.dtype).eps)
        mass_fraction = (state.mass / mass_scale).unsqueeze(1)
        age_scale = state.age.max().clamp_min(1.0)
        age_fraction = (state.age / age_scale).unsqueeze(1)
        evidence_rows = evidence.unsqueeze(0).expand(state.keys.shape[0], -1)
        recurrent_input = torch.cat(
            [attention_column * evidence_rows, attention_column, mass_fraction, age_fraction],
            dim=1,
        )
        encoded = self.evidence_projection(recurrent_input)
        next_hidden = self.recurrent(encoded, state.hidden)
        next_hidden = self.context_norm(next_hidden)
        next_mass = state.mass + attention.squeeze(0)
        next_age = (state.age + 1.0) * (1.0 - attention.squeeze(0))
        context = attention @ next_hidden
        return (
            TraceMetadataState(
                keys=state.keys,
                hidden=next_hidden,
                mass=next_mass,
                age=next_age,
            ),
            context.squeeze(0),
        )


class TraceConditionedPlasticityOperator(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        n_heads: int,
        trace_context_dim: int,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}.")
        self.hidden_dim = hidden_dim
        feature_dim = len(PARAMETER_FEATURE_NAMES)
        self.parameter_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.trace_projection = nn.Sequential(
            nn.LayerNorm(trace_context_dim),
            nn.Linear(trace_context_dim, hidden_dim),
            nn.GELU(),
        )
        self.recurrent = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(ACTION_NAMES)),
        )

    def initial_state(
        self,
        *,
        named_parameters: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> OperatorState:
        if not named_parameters:
            raise ValueError("Cannot initialize operator state without parameters.")
        return OperatorState(
            hidden=torch.zeros(len(named_parameters), self.hidden_dim, device=device, dtype=dtype),
            previous_gradients={name: torch.zeros_like(parameter) for name, parameter in named_parameters.items()},
            previous_update_rms=torch.zeros(len(named_parameters), device=device, dtype=dtype),
        )

    def forward(
        self,
        *,
        parameter_features: torch.Tensor,
        trace_context: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if parameter_features.ndim != 2 or parameter_features.shape[1] != len(PARAMETER_FEATURE_NAMES):
            raise ValueError(
                f"parameter_features must be [groups, {len(PARAMETER_FEATURE_NAMES)}], "
                f"got {tuple(parameter_features.shape)}."
            )
        if trace_context.ndim != 1:
            raise ValueError(f"trace_context must be one-dimensional, got {tuple(trace_context.shape)}.")
        local = self.parameter_projection(parameter_features)
        trace = self.trace_projection(trace_context).unsqueeze(0).expand(local.shape[0], -1)
        next_local = self.recurrent(torch.cat([local, trace], dim=1), hidden)
        attended, weights = self.global_attention(
            next_local.unsqueeze(0),
            next_local.unsqueeze(0),
            next_local.unsqueeze(0),
            need_weights=True,
            average_attn_weights=True,
        )
        next_hidden = self.context_norm(next_local + attended.squeeze(0))
        probabilities = torch.softmax(self.action_head(next_hidden), dim=1)
        return probabilities, next_hidden, weights.squeeze(0)


class TraceConditionedSystem(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.task_model = TinyQueryTransformer(
            num_entities=args.num_entities,
            num_values=args.num_values,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
        )
        evidence_dim = len(OBSERVATION_FEATURE_NAMES) + len(PARAMETER_FEATURE_NAMES)
        self.trace_attention = TraceMetadataAttention(
            d_model=args.d_model,
            hidden_dim=args.trace_hidden_dim,
            evidence_dim=evidence_dim,
        )
        self.operator = TraceConditionedPlasticityOperator(
            hidden_dim=args.operator_hidden_dim,
            n_heads=args.operator_heads,
            trace_context_dim=args.trace_hidden_dim,
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


def validate_args(args: argparse.Namespace) -> None:
    for name in [
        "num_entities",
        "num_values",
        "initial_entities",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "trace_hidden_dim",
        "operator_hidden_dim",
        "operator_heads",
        "stable_events",
        "train_episodes",
        "test_episodes",
        "print_every",
    ]:
        positive_int(name, getattr(args, name))
    for name in [
        "novel_events",
        "correction_events",
        "noise_events",
        "novel_confirmations",
        "correction_confirmations",
    ]:
        nonnegative_int(name, getattr(args, name))
    for name in ["outer_lr", "inner_lr", "grad_clip"]:
        positive_float(name, getattr(args, name))
    nonnegative_float("weight_decay", args.weight_decay)
    if args.d_model % args.n_heads != 0:
        raise ValueError(f"--d-model={args.d_model} must be divisible by --n-heads={args.n_heads}.")
    if args.operator_hidden_dim % args.operator_heads != 0:
        raise ValueError(
            f"--operator-hidden-dim={args.operator_hidden_dim} must be divisible by "
            f"--operator-heads={args.operator_heads}."
        )
    if args.initial_entities + args.novel_events > args.num_entities:
        raise ValueError(
            "--initial-entities + --novel-events must not exceed --num-entities; "
            f"got {args.initial_entities} + {args.novel_events} > {args.num_entities}."
        )
    if args.correction_events > args.initial_entities:
        raise ValueError(
            "--correction-events must not exceed --initial-entities; "
            f"got {args.correction_events} > {args.initial_entities}."
        )
    if args.num_values < 2:
        raise ValueError("--num-values must be at least 2.")


def observation_features(
    *,
    logits: torch.Tensor,
    observed_target: torch.Tensor,
    parameter_features: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"Observation logits must be [1, values], got {tuple(logits.shape)}.")
    probabilities = torch.softmax(logits.detach(), dim=1)
    eps = torch.finfo(probabilities.dtype).eps
    target_probability = probabilities[0, observed_target.item()]
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1).squeeze(0)
    sorted_probabilities = probabilities.sort(dim=1, descending=True).values
    margin = sorted_probabilities[0, 0] - sorted_probabilities[0, 1]
    loss = F.cross_entropy(logits.detach(), observed_target)
    recurrence_column = PARAMETER_FEATURE_NAMES.index("gradient_recurrence")
    recurrence = parameter_features[:, recurrence_column]
    conflict = (-recurrence).clamp_min(0.0)
    gradient_rms_column = PARAMETER_FEATURE_NAMES.index("gradient_rms_log")
    return torch.stack(
        [
            torch.log1p(loss),
            target_probability,
            entropy,
            margin,
            parameter_features[:, gradient_rms_column].mean(),
            recurrence.mean(),
            conflict.mean(),
        ]
    )


def trace_evidence(
    *,
    observation: torch.Tensor,
    parameter_features: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([observation, parameter_features.mean(dim=0)], dim=0)


def trace_attention_metrics(attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eps = torch.finfo(attention.dtype).eps
    entropy = -(attention * attention.clamp_min(eps).log()).sum(dim=1).mean()
    top_share = attention.max(dim=1).values.mean()
    return entropy, top_share


def action_summary(probabilities: torch.Tensor) -> dict[str, float]:
    mean = probabilities.mean(dim=0)
    return {name: float(mean[index].detach().cpu()) for index, name in enumerate(ACTION_NAMES)}


def run_episode(
    *,
    system: TraceConditionedSystem,
    events: list[StreamEvent],
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
    training: bool,
    collect_details: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode not in {"learned", "uniform", "sgd", "no_update"}:
        raise ValueError(f"Unknown mode {mode!r}.")
    initial_parameters = task_parameters(system.task_model)
    if training:
        parameters = dict(initial_parameters)
    else:
        parameters = {
            name: parameter.detach().clone().requires_grad_(True)
            for name, parameter in initial_parameters.items()
        }
    names = list(parameters)
    dtype = next(iter(parameters.values())).dtype
    operator_state = system.operator.initial_state(
        named_parameters=parameters,
        device=device,
        dtype=dtype,
    )
    all_entities = torch.arange(args.num_entities, dtype=torch.long, device=device)
    initial_geometry, _initial_logits = functional_task_forward(system.task_model, parameters, all_entities)
    trace_state = system.trace_attention.initial_state(keys=initial_geometry.detach())

    future_losses: list[torch.Tensor] = []
    event_rows: list[dict[str, Any]] = []
    actions_by_kind: dict[str, list[torch.Tensor]] = defaultdict(list)
    noise_adoptions: list[float] = []
    action_entropies: list[torch.Tensor] = []
    tensor_diversities: list[torch.Tensor] = []
    trace_entropies: list[torch.Tensor] = []
    trace_top_shares: list[torch.Tensor] = []
    global_attention_entropies: list[torch.Tensor] = []

    for event_index, event in enumerate(events):
        entity = torch.tensor([event.entity], dtype=torch.long, device=device)
        observed_target = torch.tensor([event.observed_value], dtype=torch.long, device=device)
        representation, logits = functional_task_forward(system.task_model, parameters, entity)
        observed_loss = F.cross_entropy(logits, observed_target)
        gradients_tuple = torch.autograd.grad(
            observed_loss,
            tuple(parameters.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        gradients = {
            name: gradient.detach()
            for name, gradient in zip(names, gradients_tuple, strict=True)
        }
        parameter_features = build_parameter_features(
            parameters=parameters,
            gradients=gradients,
            state=operator_state,
            observed_loss=observed_loss,
        ).detach()
        observation = observation_features(
            logits=logits,
            observed_target=observed_target,
            parameter_features=parameter_features,
        )
        trace_attention = system.trace_attention.attend(representation.detach(), trace_state)
        trace_state, trace_context = system.trace_attention.update(
            state=trace_state,
            attention=trace_attention,
            evidence=trace_evidence(observation=observation, parameter_features=parameter_features),
        )
        learned_probabilities, next_hidden, global_attention = system.operator(
            parameter_features=parameter_features,
            trace_context=trace_context,
            hidden=operator_state.hidden,
        )
        if mode == "learned":
            probabilities = learned_probabilities
            parameters, update_rms = apply_probabilistic_update(
                parameters=parameters,
                gradients=gradients,
                probabilities=probabilities,
                inner_lr=args.inner_lr,
            )
        elif mode == "uniform":
            probabilities = torch.full_like(learned_probabilities, 1.0 / float(len(ACTION_NAMES)))
            parameters, update_rms = apply_probabilistic_update(
                parameters=parameters,
                gradients=gradients,
                probabilities=probabilities,
                inner_lr=args.inner_lr,
            )
        elif mode == "sgd":
            probabilities = torch.zeros_like(learned_probabilities)
            probabilities[:, ACTION_INDEX["write"]] = 1.0
            parameters, update_rms = apply_sgd_update(
                parameters=parameters,
                gradients=gradients,
                inner_lr=args.inner_lr,
            )
        elif mode == "no_update":
            probabilities = torch.zeros_like(learned_probabilities)
            probabilities[:, ACTION_INDEX["defer"]] = 1.0
            update_rms = torch.zeros(len(parameters), device=device, dtype=dtype)
        else:
            raise ValueError(f"Unhandled mode {mode!r}.")
        operator_state = update_operator_state(
            state=operator_state,
            hidden=next_hidden,
            gradients=gradients,
            update_rms=update_rms,
        )
        actions_by_kind[event.kind].append(probabilities.mean(dim=0))
        eps = torch.finfo(probabilities.dtype).eps
        action_entropies.append(
            -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1).mean()
        )
        tensor_diversities.append(probabilities.var(dim=0, unbiased=False).mean())
        trace_entropy, trace_top_share = trace_attention_metrics(trace_attention)
        trace_entropies.append(trace_entropy)
        trace_top_shares.append(trace_top_share)
        global_attention_entropies.append(
            -(global_attention * global_attention.clamp_min(eps).log()).sum(dim=1).mean()
        )

        active_entities = torch.tensor(event.active_entities, dtype=torch.long, device=device)
        active_values = torch.tensor(event.active_values, dtype=torch.long, device=device)
        _future_representation, future_logits = functional_task_forward(
            system.task_model,
            parameters,
            active_entities,
        )
        future_loss = F.cross_entropy(future_logits, active_values)
        future_losses.append(future_loss)
        predictions = future_logits.argmax(dim=1)
        accuracy = float((predictions == active_values).to(torch.float32).mean().detach().cpu())
        if event.kind == "noise":
            position = event.active_entities.index(event.entity)
            noise_adoptions.append(float(predictions[position].item() == event.observed_value))
        if collect_details:
            event_rows.append(
                {
                    "index": event_index,
                    "kind": event.kind,
                    "entity": event.entity,
                    "observed_value": event.observed_value,
                    "true_value": event.true_value,
                    "active_accuracy": accuracy,
                    "observed_probability": float(observation[1].detach().cpu()),
                    "trace_attention_entropy": float(trace_entropy.detach().cpu()),
                    "trace_top_share": float(trace_top_share.detach().cpu()),
                    "actions": action_summary(probabilities),
                }
            )
        if not training:
            parameters = {
                name: parameter.detach().requires_grad_(True)
                for name, parameter in parameters.items()
            }
            operator_state.hidden = operator_state.hidden.detach()
            trace_state.hidden = trace_state.hidden.detach()

    if not future_losses:
        raise RuntimeError("Episode produced no future losses.")
    final_entities = torch.tensor(events[-1].active_entities, dtype=torch.long, device=device)
    final_values = torch.tensor(events[-1].active_values, dtype=torch.long, device=device)
    _final_representation, final_logits = functional_task_forward(system.task_model, parameters, final_entities)
    final_accuracy = float((final_logits.argmax(dim=1) == final_values).to(torch.float32).mean().detach().cpu())
    final_geometry, _final_geometry_logits = functional_task_forward(system.task_model, parameters, all_entities)
    representation_drift = float(
        (
            torch.linalg.vector_norm(final_geometry - initial_geometry)
            / torch.linalg.vector_norm(initial_geometry).clamp_min(torch.finfo(dtype).eps)
        ).detach().cpu()
    )
    representation_cka = centered_kernel_alignment(initial_geometry, final_geometry)
    mean_loss = torch.stack(future_losses).mean()
    action_means_by_event = {
        kind: {
            name: float(torch.stack(rows).mean(dim=0)[index].detach().cpu())
            for index, name in enumerate(ACTION_NAMES)
        }
        for kind, rows in sorted(actions_by_kind.items())
    }
    initial_norm = torch.stack(
        [torch.linalg.vector_norm(parameter) for parameter in initial_parameters.values()]
    ).square().sum().sqrt()
    final_delta = torch.stack(
        [torch.linalg.vector_norm(parameters[name] - initial_parameters[name]) for name in initial_parameters]
    ).square().sum().sqrt()
    report = {
        "mean_future_loss": float(mean_loss.detach().cpu()),
        "final_accuracy": final_accuracy,
        "noise_adoption_rate": sum(noise_adoptions) / float(len(noise_adoptions)) if noise_adoptions else None,
        "action_entropy": float(torch.stack(action_entropies).mean().detach().cpu()),
        "tensor_action_diversity": float(torch.stack(tensor_diversities).mean().detach().cpu()),
        "trace_attention_entropy": float(torch.stack(trace_entropies).mean().detach().cpu()),
        "trace_top_share": float(torch.stack(trace_top_shares).mean().detach().cpu()),
        "global_attention_entropy": float(torch.stack(global_attention_entropies).mean().detach().cpu()),
        "parameter_delta_relative": float(
            (final_delta / initial_norm.clamp_min(torch.finfo(dtype).eps)).detach().cpu()
        ),
        "representation_drift": representation_drift,
        "representation_cka": representation_cka,
        "action_means_by_event": action_means_by_event,
        "events": event_rows,
    }
    return mean_loss, report


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise RuntimeError("Cannot aggregate empty reports.")
    scalar_names = [
        "mean_future_loss",
        "final_accuracy",
        "action_entropy",
        "tensor_action_diversity",
        "trace_attention_entropy",
        "trace_top_share",
        "global_attention_entropy",
        "parameter_delta_relative",
        "representation_drift",
        "representation_cka",
    ]
    aggregate = {
        name: sum(float(report[name]) for report in reports) / float(len(reports))
        for name in scalar_names
    }
    noise = [float(report["noise_adoption_rate"]) for report in reports if report["noise_adoption_rate"] is not None]
    aggregate["noise_adoption_rate"] = sum(noise) / float(len(noise)) if noise else None
    action_rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for report in reports:
        for kind, action_map in report["action_means_by_event"].items():
            for action, value in action_map.items():
                action_rows[kind][action].append(float(value))
    aggregate["action_means_by_event"] = {
        kind: {
            action: sum(values) / float(len(values))
            for action, values in sorted(action_map.items())
        }
        for kind, action_map in sorted(action_rows.items())
    }
    return aggregate


def train_system(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[TraceConditionedSystem, list[dict[str, float]]]:
    system = TraceConditionedSystem(args).to(device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=args.outer_lr, weight_decay=args.weight_decay)
    trace: list[dict[str, float]] = []
    for episode_index in tqdm(range(args.train_episodes), desc="train trace-conditioned operator"):
        events = generate_episode(args, seed=args.seed + episode_index)
        optimizer.zero_grad(set_to_none=True)
        loss, report = run_episode(
            system=system,
            events=events,
            args=args,
            device=device,
            mode="learned",
            training=True,
            collect_details=False,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite outer loss at episode {episode_index}: {float(loss.detach().cpu())}.")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(system.parameters(), args.grad_clip).detach().cpu())
        optimizer.step()
        row = {
            "episode": float(episode_index + 1),
            "loss": float(loss.detach().cpu()),
            "final_accuracy": float(report["final_accuracy"]),
            "action_entropy": float(report["action_entropy"]),
            "tensor_action_diversity": float(report["tensor_action_diversity"]),
            "trace_top_share": float(report["trace_top_share"]),
            "grad_norm": grad_norm,
        }
        trace.append(row)
        if episode_index == 0 or episode_index + 1 == args.train_episodes or (episode_index + 1) % args.print_every == 0:
            print(
                f"episode={episode_index + 1:4d} loss={row['loss']:.5f} "
                f"final_acc={row['final_accuracy']:.3f} actionH={row['action_entropy']:.3f} "
                f"tensorDiv={row['tensor_action_diversity']:.5f} traceTop={row['trace_top_share']:.3f} "
                f"grad={row['grad_norm']:.3f}"
            )
    return system, trace


def evaluate_system(
    args: argparse.Namespace,
    *,
    system: TraceConditionedSystem,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system.eval()
    modes = ["learned", "uniform", "sgd", "no_update"]
    reports: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    details: dict[str, Any] = {}
    for test_index in range(args.test_episodes):
        events = generate_episode(args, seed=args.seed + args.train_episodes + test_index)
        for mode in modes:
            _loss, report = run_episode(
                system=system,
                events=events,
                args=args,
                device=device,
                mode=mode,
                training=False,
                collect_details=test_index == 0,
            )
            reports[mode].append(report)
            if test_index == 0:
                details[mode] = {
                    "stream": [asdict(event) for event in events],
                    "report": report,
                }
    return {mode: aggregate_reports(rows) for mode, rows in reports.items()}, details


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("TINY TRACE-CONDITIONED PLASTICITY OPERATOR")
    print("=" * 140)
    print(
        f"device={device} train_episodes={args.train_episodes} test_episodes={args.test_episodes} "
        f"actions={ACTION_NAMES}"
    )
    system, training_trace = train_system(args, device=device)
    aggregate, details = evaluate_system(args, system=system, device=device)
    print("\nTRACE-CONDITIONED PLASTICITY SUMMARY")
    print("=" * 150)
    print(
        f"{'mode':>10} {'loss':>9} {'finalAcc':>9} {'noise':>9} {'dTheta':>9} {'drift':>9} "
        f"{'cka':>9} {'actionH':>9} {'tensorDiv':>10} {'traceTop':>9}"
    )
    for mode in ["learned", "uniform", "sgd", "no_update"]:
        row = aggregate[mode]
        noise = "NA" if row["noise_adoption_rate"] is None else f"{row['noise_adoption_rate']:.4f}"
        print(
            f"{mode:>10} {row['mean_future_loss']:9.4f} {row['final_accuracy']:9.4f} {noise:>9} "
            f"{row['parameter_delta_relative']:9.4f} {row['representation_drift']:9.4f} "
            f"{row['representation_cka']:9.4f} {row['action_entropy']:9.4f} "
            f"{row['tensor_action_diversity']:10.6f} {row['trace_top_share']:9.4f}"
        )
    print("\nLEARNED UPDATE PROBABILITIES BY HIDDEN EVENT TYPE (EVALUATION ONLY)")
    print("-" * 150)
    for kind, action_map in aggregate["learned"]["action_means_by_event"].items():
        print(f"{kind:>26} " + " ".join(f"{name}={action_map[name]:.3f}" for name in ACTION_NAMES))

    output = {
        "question": (
            "Can trace-level temporal metadata make a fixed global plasticity-attention operator conditionally "
            "allocate task-weight updates beyond ordinary SGD?"
        ),
        "scope": (
            "The task transformer stores content and changes online. Trace states store only statistics keyed by "
            "activations. Invariant-Tangent projection is not yet applied."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
        "parameter_feature_names": list(PARAMETER_FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "training_trace": training_trace,
        "aggregate": aggregate,
        "first_test_episode": details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("model/analysis/gco-tiny-trace-conditioned-plasticity-operator-seed0.json"),
    )
    parser.add_argument("--num-entities", type=int, default=12)
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--initial-entities", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--trace-hidden-dim", type=int, default=64)
    parser.add_argument("--operator-hidden-dim", type=int, default=64)
    parser.add_argument("--operator-heads", type=int, default=4)
    parser.add_argument("--stable-events", type=int, default=12)
    parser.add_argument("--novel-events", type=int, default=4)
    parser.add_argument("--correction-events", type=int, default=3)
    parser.add_argument("--noise-events", type=int, default=6)
    parser.add_argument("--novel-confirmations", type=int, default=2)
    parser.add_argument("--correction-confirmations", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--outer-lr", type=float, default=3e-4)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
