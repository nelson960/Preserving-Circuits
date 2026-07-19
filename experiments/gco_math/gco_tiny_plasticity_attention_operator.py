"""Attention-based plasticity operator for continual task-weight updates.

The task transformer owns all learned content. A separate plasticity operator
observes statistics from every trainable parameter tensor, attends globally
across those tensors, and emits soft probabilities for update operations:

    write, reuse, stabilize, revise, defer, decay

The operator never stores target logits or factual content. During held-out
continual-learning episodes its parameters are fixed; only its recurrent
statistical state evolves while the task-transformer weights are updated.

The operator is trained across randomized temporal worlds using future
prediction loss. Hidden event types are used only for evaluation. Online task
updates use a first-order differentiable approximation: task gradients are
detached before applying the operator, while future loss remains differentiable
with respect to the operator's probabilities and the initial task model.

This experiment tests autonomous update allocation. It does not yet apply the
repository's Invariant-Tangent projection or bounded restore.
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
from torch.func import functional_call
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_tiny_probabilistic_plasticity_attention import (
    ACTION_INDEX,
    ACTION_NAMES,
    StreamEvent,
    TinyQueryTransformer,
    centered_kernel_alignment,
    generate_episode,
)


PARAMETER_FEATURE_NAMES = (
    "loss_log",
    "gradient_rms_log",
    "parameter_rms_log",
    "gradient_recurrence",
    "gradient_parameter_alignment",
    "previous_update_rms_log",
    "tensor_size_fraction",
)


@dataclass
class OperatorState:
    hidden: torch.Tensor
    previous_gradients: dict[str, torch.Tensor]
    previous_update_rms: torch.Tensor


class PlasticityAttentionOperator(nn.Module):
    def __init__(self, *, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}.")
        feature_dim = len(PARAMETER_FEATURE_NAMES)
        self.hidden_dim = hidden_dim
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.recurrent = nn.GRUCell(hidden_dim, hidden_dim)
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
            raise ValueError("Cannot initialize operator state without task parameters.")
        return OperatorState(
            hidden=torch.zeros(len(named_parameters), self.hidden_dim, device=device, dtype=dtype),
            previous_gradients={name: torch.zeros_like(parameter) for name, parameter in named_parameters.items()},
            previous_update_rms=torch.zeros(len(named_parameters), device=device, dtype=dtype),
        )

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != len(PARAMETER_FEATURE_NAMES):
            raise ValueError(
                f"features must be [groups, {len(PARAMETER_FEATURE_NAMES)}], got {tuple(features.shape)}."
            )
        if hidden.shape != (features.shape[0], self.hidden_dim):
            raise ValueError(
                f"hidden must be {(features.shape[0], self.hidden_dim)}, got {tuple(hidden.shape)}."
            )
        encoded = self.feature_projection(self.feature_norm(features))
        local = self.recurrent(encoded, hidden)
        attended, attention_weights = self.global_attention(
            local.unsqueeze(0),
            local.unsqueeze(0),
            local.unsqueeze(0),
            need_weights=True,
            average_attn_weights=True,
        )
        next_hidden = self.context_norm(local + attended.squeeze(0))
        probabilities = torch.softmax(self.action_head(next_hidden), dim=1)
        return probabilities, next_hidden, attention_weights.squeeze(0)


class PlasticityTaskSystem(nn.Module):
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
        self.operator = PlasticityAttentionOperator(
            hidden_dim=args.operator_hidden_dim,
            n_heads=args.operator_heads,
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


def task_parameters(model: TinyQueryTransformer) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    if not parameters:
        raise RuntimeError("Task model exposes no trainable parameters.")
    return parameters


def functional_task_forward(
    model: TinyQueryTransformer,
    parameters: dict[str, torch.Tensor],
    entities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    representation, logits = functional_call(model, parameters, (entities,))
    return representation, logits


def rms(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean().add(torch.finfo(value.dtype).eps).sqrt()


def cosine_similarity_flat(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denominator.detach().cpu()) <= torch.finfo(left.dtype).eps:
        return left.new_zeros(())
    return torch.dot(left_flat, right_flat) / denominator


def build_parameter_features(
    *,
    parameters: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    state: OperatorState,
    observed_loss: torch.Tensor,
) -> torch.Tensor:
    if list(parameters) != list(gradients):
        raise ValueError("Parameter and gradient order must match exactly.")
    if list(parameters) != list(state.previous_gradients):
        raise ValueError("Parameter and previous-gradient order must match exactly.")
    total_parameters = sum(parameter.numel() for parameter in parameters.values())
    if total_parameters <= 0:
        raise RuntimeError("Task model has zero scalar parameters.")
    rows: list[torch.Tensor] = []
    for index, (name, parameter) in enumerate(parameters.items()):
        gradient = gradients[name]
        previous_gradient = state.previous_gradients[name]
        gradient_rms = rms(gradient)
        parameter_rms = rms(parameter)
        recurrence = cosine_similarity_flat(gradient, previous_gradient)
        alignment = cosine_similarity_flat(gradient, parameter)
        size_fraction = parameter.new_tensor(float(parameter.numel()) / float(total_parameters))
        rows.append(
            torch.stack(
                [
                    torch.log1p(observed_loss.detach()),
                    torch.log1p(gradient_rms),
                    torch.log1p(parameter_rms),
                    recurrence,
                    alignment,
                    torch.log1p(state.previous_update_rms[index]),
                    size_fraction,
                ]
            )
        )
    return torch.stack(rows)


def apply_probabilistic_update(
    *,
    parameters: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    probabilities: torch.Tensor,
    inner_lr: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if probabilities.shape != (len(parameters), len(ACTION_NAMES)):
        raise ValueError(
            f"probabilities must be {(len(parameters), len(ACTION_NAMES))}, got {tuple(probabilities.shape)}."
        )
    updated: dict[str, torch.Tensor] = {}
    update_rms_values: list[torch.Tensor] = []
    for index, (name, parameter) in enumerate(parameters.items()):
        gradient = gradients[name]
        action = probabilities[index]
        write_probability = action[ACTION_INDEX["write"]]
        revise_probability = action[ACTION_INDEX["revise"]]
        decay_probability = action[ACTION_INDEX["decay"]]
        learn_probability = write_probability + revise_probability
        erase_probability = revise_probability + decay_probability
        parameter_direction = parameter / rms(parameter)
        forgetting_direction = parameter_direction * rms(gradient)
        update = -inner_lr * (
            learn_probability * gradient
            + erase_probability * forgetting_direction
        )
        updated[name] = parameter + update
        update_rms_values.append(rms(update))
    return updated, torch.stack(update_rms_values)


def apply_sgd_update(
    *,
    parameters: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    inner_lr: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    updated: dict[str, torch.Tensor] = {}
    update_rms_values: list[torch.Tensor] = []
    for name, parameter in parameters.items():
        update = -inner_lr * gradients[name]
        updated[name] = parameter + update
        update_rms_values.append(rms(update))
    return updated, torch.stack(update_rms_values)


def update_operator_state(
    *,
    state: OperatorState,
    hidden: torch.Tensor,
    gradients: dict[str, torch.Tensor],
    update_rms: torch.Tensor,
) -> OperatorState:
    return OperatorState(
        hidden=hidden,
        previous_gradients={name: gradient.detach() for name, gradient in gradients.items()},
        previous_update_rms=update_rms.detach(),
    )


def action_summary(probabilities: torch.Tensor) -> dict[str, float]:
    mean = probabilities.mean(dim=0)
    return {name: float(mean[index].detach().cpu()) for index, name in enumerate(ACTION_NAMES)}


def parameter_delta_norm(
    initial_parameters: dict[str, torch.Tensor],
    final_parameters: dict[str, torch.Tensor],
) -> float:
    numerator = torch.stack(
        [torch.linalg.vector_norm(final_parameters[name] - initial_parameters[name]) for name in initial_parameters]
    ).square().sum().sqrt()
    denominator = torch.stack(
        [torch.linalg.vector_norm(parameter) for parameter in initial_parameters.values()]
    ).square().sum().sqrt()
    return float((numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)).detach().cpu())


def run_cl_episode(
    *,
    system: PlasticityTaskSystem,
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
    state = system.operator.initial_state(
        named_parameters=parameters,
        device=device,
        dtype=dtype,
    )
    geometry_entities = torch.arange(args.num_entities, dtype=torch.long, device=device)
    initial_representation, _initial_logits = functional_task_forward(
        system.task_model,
        parameters,
        geometry_entities,
    )
    future_losses: list[torch.Tensor] = []
    action_by_kind: dict[str, list[torch.Tensor]] = defaultdict(list)
    event_rows: list[dict[str, Any]] = []
    noise_adoptions: list[float] = []
    action_entropies: list[torch.Tensor] = []
    tensor_diversities: list[torch.Tensor] = []
    global_attention_entropies: list[torch.Tensor] = []

    for event_index, event in enumerate(events):
        entity = torch.tensor([event.entity], dtype=torch.long, device=device)
        target = torch.tensor([event.observed_value], dtype=torch.long, device=device)
        _representation, logits = functional_task_forward(system.task_model, parameters, entity)
        observed_loss = F.cross_entropy(logits, target)
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
        features = build_parameter_features(
            parameters=parameters,
            gradients=gradients,
            state=state,
            observed_loss=observed_loss,
        )
        learned_probabilities, next_hidden, attention_weights = system.operator(features, state.hidden)
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
        state = update_operator_state(
            state=state,
            hidden=next_hidden,
            gradients=gradients,
            update_rms=update_rms,
        )
        action_by_kind[event.kind].append(probabilities.mean(dim=0))
        probability_eps = torch.finfo(probabilities.dtype).eps
        action_entropies.append(
            -(probabilities * probabilities.clamp_min(probability_eps).log()).sum(dim=1).mean()
        )
        tensor_diversities.append(probabilities.var(dim=0, unbiased=False).mean())
        global_attention_entropies.append(
            -(attention_weights * attention_weights.clamp_min(probability_eps).log()).sum(dim=1).mean()
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
                    "actions": action_summary(probabilities),
                }
            )
        if not training:
            parameters = {
                name: parameter.detach().requires_grad_(True)
                for name, parameter in parameters.items()
            }
            state.hidden = state.hidden.detach()

    if not future_losses:
        raise RuntimeError("CL episode produced no future losses.")
    final_entities = torch.tensor(events[-1].active_entities, dtype=torch.long, device=device)
    final_values = torch.tensor(events[-1].active_values, dtype=torch.long, device=device)
    final_representation, final_logits = functional_task_forward(system.task_model, parameters, final_entities)
    final_accuracy = float((final_logits.argmax(dim=1) == final_values).to(torch.float32).mean().detach().cpu())
    final_geometry, _final_geometry_logits = functional_task_forward(
        system.task_model,
        parameters,
        geometry_entities,
    )
    representation_drift = float(
        (
            torch.linalg.vector_norm(final_geometry - initial_representation)
            / torch.linalg.vector_norm(initial_representation).clamp_min(torch.finfo(dtype).eps)
        ).detach().cpu()
    )
    representation_cka = centered_kernel_alignment(initial_representation, final_geometry)
    mean_loss = torch.stack(future_losses).mean()
    action_means_by_event = {
        kind: {
            name: float(torch.stack(rows).mean(dim=0)[index].detach().cpu())
            for index, name in enumerate(ACTION_NAMES)
        }
        for kind, rows in sorted(action_by_kind.items())
    }
    report = {
        "mean_future_loss": float(mean_loss.detach().cpu()),
        "final_accuracy": final_accuracy,
        "noise_adoption_rate": sum(noise_adoptions) / float(len(noise_adoptions)) if noise_adoptions else None,
        "action_entropy": float(torch.stack(action_entropies).mean().detach().cpu()),
        "tensor_action_diversity": float(torch.stack(tensor_diversities).mean().detach().cpu()),
        "global_attention_entropy": float(torch.stack(global_attention_entropies).mean().detach().cpu()),
        "parameter_delta_relative": parameter_delta_norm(initial_parameters, parameters),
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
) -> tuple[PlasticityTaskSystem, list[dict[str, float]]]:
    system = PlasticityTaskSystem(args).to(device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=args.outer_lr, weight_decay=args.weight_decay)
    trace: list[dict[str, float]] = []
    for episode_index in tqdm(range(args.train_episodes), desc="train plasticity operator"):
        events = generate_episode(args, seed=args.seed + episode_index)
        optimizer.zero_grad(set_to_none=True)
        loss, report = run_cl_episode(
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
            "grad_norm": grad_norm,
        }
        trace.append(row)
        if episode_index == 0 or episode_index + 1 == args.train_episodes or (episode_index + 1) % args.print_every == 0:
            print(
                f"episode={episode_index + 1:4d} loss={row['loss']:.5f} "
                f"final_acc={row['final_accuracy']:.3f} actionH={row['action_entropy']:.3f} "
                f"tensorDiv={row['tensor_action_diversity']:.5f} grad={row['grad_norm']:.3f}"
            )
    return system, trace


def evaluate_system(
    args: argparse.Namespace,
    *,
    system: PlasticityTaskSystem,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system.eval()
    modes = ["learned", "uniform", "sgd", "no_update"]
    reports_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    details: dict[str, Any] = {}
    for test_index in range(args.test_episodes):
        events = generate_episode(args, seed=args.seed + args.train_episodes + test_index)
        for mode in modes:
            _loss, report = run_cl_episode(
                system=system,
                events=events,
                args=args,
                device=device,
                mode=mode,
                training=False,
                collect_details=test_index == 0,
            )
            reports_by_mode[mode].append(report)
            if test_index == 0:
                details[mode] = {
                    "stream": [asdict(event) for event in events],
                    "report": report,
                }
    aggregate = {mode: aggregate_reports(reports) for mode, reports in reports_by_mode.items()}
    return aggregate, details


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("TINY PLASTICITY ATTENTION OPERATOR")
    print("=" * 120)
    print(
        f"device={device} train_episodes={args.train_episodes} test_episodes={args.test_episodes} "
        f"actions={ACTION_NAMES}"
    )
    system, training_trace = train_system(args, device=device)
    aggregate, details = evaluate_system(args, system=system, device=device)
    print("\nPLASTICITY ATTENTION OPERATOR SUMMARY")
    print("=" * 140)
    print(
        f"{'mode':>10} {'loss':>9} {'finalAcc':>9} {'noise':>9} {'dTheta':>9} "
        f"{'drift':>9} {'cka':>9} {'actionH':>9} {'tensorDiv':>10}"
    )
    for mode in ["learned", "uniform", "sgd", "no_update"]:
        row = aggregate[mode]
        noise = "NA" if row["noise_adoption_rate"] is None else f"{row['noise_adoption_rate']:.4f}"
        print(
            f"{mode:>10} {row['mean_future_loss']:9.4f} {row['final_accuracy']:9.4f} {noise:>9} "
            f"{row['parameter_delta_relative']:9.4f} {row['representation_drift']:9.4f} "
            f"{row['representation_cka']:9.4f} {row['action_entropy']:9.4f} "
            f"{row['tensor_action_diversity']:10.6f}"
        )
    print("\nLEARNED UPDATE PROBABILITIES BY HIDDEN EVENT TYPE (EVALUATION ONLY)")
    print("-" * 140)
    for kind, action_map in aggregate["learned"]["action_means_by_event"].items():
        print(f"{kind:>26} " + " ".join(f"{name}={action_map[name]:.3f}" for name in ACTION_NAMES))

    output = {
        "question": (
            "Can a fixed attention-based plasticity mechanism allocate continual task-weight updates from global "
            "parameter statistics without storing content or receiving action labels?"
        ),
        "scope": (
            "The task model owns content and changes online. The operator is fixed during held-out CL runs; only "
            "its statistical state evolves. Invariant-Tangent projection is not yet applied."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
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
        default=Path("model/analysis/gco-tiny-plasticity-attention-operator-seed0.json"),
    )
    parser.add_argument("--num-entities", type=int, default=12)
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--initial-entities", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--operator-hidden-dim", type=int, default=64)
    parser.add_argument("--operator-heads", type=int, default=4)
    parser.add_argument("--stable-events", type=int, default=12)
    parser.add_argument("--novel-events", type=int, default=4)
    parser.add_argument("--correction-events", type=int, default=3)
    parser.add_argument("--noise-events", type=int, default=6)
    parser.add_argument("--novel-confirmations", type=int, default=2)
    parser.add_argument("--correction-confirmations", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=200)
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
