"""Probabilistic plasticity attention on randomized temporal worlds.

This experiment tests the missing autonomous-plasticity mechanism in isolation.
A tiny transformer produces query features. A bank of differentiable fast-weight
feature slots receives observations through attention and maintains a posterior
over continuous plasticity operations:

    write, reuse, stabilize, revise, defer, decay

No event type or plasticity action is supplied to the model. Hidden temporal
types are retained only for evaluation. The entire mechanism is trained through
future prediction loss after online observations.

This is the feature-plasticity stage. It does not yet apply the learned
probabilities to the repository's Invariant-Tangent weight executor.
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


ACTION_NAMES = ("write", "reuse", "stabilize", "revise", "defer", "decay")
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}


@dataclass(frozen=True)
class PlannedObservation:
    kind: str
    entity: int | None
    value: int | None


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    entity: int
    observed_value: int
    true_value: int
    active_entities: tuple[int, ...]
    active_values: tuple[int, ...]


@dataclass
class PlasticityState:
    keys: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor
    alpha: torch.Tensor
    mass: torch.Tensor


class TinyQueryTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_entities: int,
        num_values: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.num_entities = num_entities
        self.query_token = num_entities
        self.token_embedding = nn.Embedding(num_entities + 1, d_model)
        self.position_embedding = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, num_values)

    def forward(self, entities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if entities.ndim != 1:
            raise ValueError(f"entities must be one-dimensional, got {tuple(entities.shape)}.")
        if entities.numel() <= 0:
            raise ValueError("entities must not be empty.")
        if int(entities.min()) < 0 or int(entities.max()) >= self.num_entities:
            raise ValueError(f"entity ids must be in [0, {self.num_entities}), got {entities.tolist()}.")
        query = torch.full_like(entities, self.query_token)
        tokens = torch.stack([query, entities], dim=1)
        positions = torch.arange(2, device=entities.device).unsqueeze(0).expand(entities.shape[0], 2)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.encoder(hidden)
        representation = self.final_norm(hidden[:, -1])
        return representation, self.output(representation)


class PlasticFeatureBank(nn.Module):
    def __init__(
        self,
        *,
        num_slots: int,
        d_model: int,
        hidden_dim: int,
        num_values: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_values = num_values
        self.initial_keys = nn.Parameter(torch.empty(num_slots, d_model))
        self.initial_values = nn.Parameter(torch.zeros(num_slots, num_values))
        self.initial_alpha_logits = nn.Parameter(torch.zeros(len(ACTION_NAMES)))
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_embedding = nn.Embedding(num_values, d_model)
        transition_input_dim = d_model + 3 + hidden_dim
        self.transition = nn.GRUCell(transition_input_dim, hidden_dim)
        evidence_input_dim = hidden_dim + d_model + len(ACTION_NAMES) + 3
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(evidence_input_dim),
            nn.Linear(evidence_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(ACTION_NAMES)),
        )
        self.retention_head = nn.Sequential(
            nn.LayerNorm(evidence_input_dim),
            nn.Linear(evidence_input_dim, 1),
        )
        proposal_input_dim = hidden_dim + d_model * 2 + num_values
        self.value_proposal = nn.Sequential(
            nn.LayerNorm(proposal_input_dim),
            nn.Linear(proposal_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_values),
        )
        self.key_proposal = nn.Sequential(
            nn.LayerNorm(hidden_dim + d_model),
            nn.Linear(hidden_dim + d_model, d_model),
            nn.Tanh(),
        )
        self.memory_scale = nn.Parameter(torch.tensor(1.0))
        self.mass_attraction = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.initial_keys, mean=0.0, std=1.0 / math.sqrt(float(d_model)))

    def initial_state(self, *, device: torch.device, dtype: torch.dtype) -> PlasticityState:
        eps = torch.finfo(dtype).eps
        alpha_row = F.softplus(self.initial_alpha_logits).to(device=device, dtype=dtype) + eps
        return PlasticityState(
            keys=self.initial_keys.to(device=device, dtype=dtype),
            values=self.initial_values.to(device=device, dtype=dtype),
            hidden=torch.zeros(self.num_slots, self.hidden_dim, device=device, dtype=dtype),
            alpha=alpha_row.unsqueeze(0).expand(self.num_slots, -1),
            mass=torch.zeros(self.num_slots, device=device, dtype=dtype),
        )

    def posterior(self, state: PlasticityState) -> torch.Tensor:
        denominator = state.alpha.sum(dim=1, keepdim=True)
        if torch.any(denominator <= 0):
            raise RuntimeError("Plasticity concentrations must remain positive.")
        return state.alpha / denominator

    def attend(self, queries: torch.Tensor, state: PlasticityState) -> torch.Tensor:
        if queries.ndim != 2 or queries.shape[1] != self.d_model:
            raise ValueError(f"queries must be [batch, {self.d_model}], got {tuple(queries.shape)}.")
        projected = self.query_projection(queries)
        scores = projected @ state.keys.transpose(0, 1) / math.sqrt(float(self.d_model))
        scores = scores + self.mass_attraction * torch.log1p(state.mass).unsqueeze(0)
        return torch.softmax(scores, dim=1)

    def read(
        self,
        queries: torch.Tensor,
        base_logits: torch.Tensor,
        state: PlasticityState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = self.attend(queries, state)
        memory_logits = attention @ state.values
        key_context = attention @ state.keys
        logits = base_logits + self.memory_scale * memory_logits
        representation = queries + key_context
        return logits, representation, attention

    def observe(
        self,
        *,
        query: torch.Tensor,
        logits: torch.Tensor,
        observed_value: torch.Tensor,
        attention: torch.Tensor,
        state: PlasticityState,
        posterior_mode: str,
    ) -> tuple[PlasticityState, torch.Tensor]:
        if query.shape != (1, self.d_model):
            raise ValueError(f"observe query must be [1, {self.d_model}], got {tuple(query.shape)}.")
        if logits.shape != (1, self.num_values):
            raise ValueError(f"observe logits must be [1, {self.num_values}], got {tuple(logits.shape)}.")
        if observed_value.shape != (1,):
            raise ValueError(f"observed_value must be [1], got {tuple(observed_value.shape)}.")
        if attention.shape != (1, self.num_slots):
            raise ValueError(f"attention must be [1, {self.num_slots}], got {tuple(attention.shape)}.")
        if posterior_mode not in {"learned", "uniform"}:
            raise ValueError(f"Unknown posterior_mode={posterior_mode!r}.")

        probabilities = torch.softmax(logits, dim=-1)
        target = F.one_hot(observed_value, num_classes=self.num_values).to(dtype=logits.dtype)
        error = target - probabilities
        error_norm = torch.linalg.vector_norm(error, dim=1).expand(self.num_slots, 1)
        query_rows = query.expand(self.num_slots, -1)
        attention_rows = attention.transpose(0, 1)
        mass_scale = state.mass.sum().clamp_min(torch.finfo(state.mass.dtype).eps)
        mass_rows = (state.mass / mass_scale).unsqueeze(1)
        transition_input = torch.cat(
            [query_rows, attention_rows, error_norm, mass_rows, state.hidden],
            dim=1,
        )
        next_hidden = self.transition(transition_input, state.hidden)
        prior_posterior = self.posterior(state)
        evidence_input = torch.cat(
            [next_hidden, query_rows, prior_posterior, attention_rows, error_norm, mass_rows],
            dim=1,
        )
        evidence = F.softplus(self.evidence_head(evidence_input))
        retention = torch.sigmoid(self.retention_head(evidence_input))
        next_alpha = retention * state.alpha + attention_rows * evidence
        learned_posterior = next_alpha / next_alpha.sum(dim=1, keepdim=True)
        if posterior_mode == "learned":
            action_posterior = learned_posterior
        elif posterior_mode == "uniform":
            action_posterior = torch.full_like(learned_posterior, 1.0 / float(len(ACTION_NAMES)))
        else:
            raise ValueError(f"Unhandled posterior_mode={posterior_mode!r}.")

        value_context = self.value_embedding(observed_value).expand(self.num_slots, -1)
        error_rows = error.expand(self.num_slots, -1)
        proposal_input = torch.cat([next_hidden, query_rows, value_context, error_rows], dim=1)
        proposed_values = self.value_proposal(proposal_input)
        proposed_keys = self.key_proposal(torch.cat([next_hidden, query_rows], dim=1))

        write = attention_rows * action_posterior[:, ACTION_INDEX["write"]].unsqueeze(1)
        revise = attention_rows * action_posterior[:, ACTION_INDEX["revise"]].unsqueeze(1)
        stabilize = action_posterior[:, ACTION_INDEX["stabilize"]].unsqueeze(1)
        decay = attention_rows * action_posterior[:, ACTION_INDEX["decay"]].unsqueeze(1)
        plastic = 1.0 - stabilize
        next_values = (1.0 - decay) * (
            (1.0 - revise * plastic) * state.values
            + revise * plastic * proposed_values
            + write * plastic * proposed_values
        )
        key_write = (write + revise) * plastic
        next_keys = (1.0 - decay) * ((1.0 - key_write) * state.keys + key_write * proposed_keys)
        next_mass = (1.0 - decay.squeeze(1)) * state.mass + attention.squeeze(0)
        next_state = PlasticityState(
            keys=next_keys,
            values=next_values,
            hidden=next_hidden,
            alpha=next_alpha,
            mass=next_mass,
        )
        attended_posterior = attention @ action_posterior
        return next_state, attended_posterior.squeeze(0)


class ProbabilisticPlasticityModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.backbone = TinyQueryTransformer(
            num_entities=args.num_entities,
            num_values=args.num_values,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
        )
        self.bank = PlasticFeatureBank(
            num_slots=args.num_slots,
            d_model=args.d_model,
            hidden_dim=args.plasticity_hidden_dim,
            num_values=args.num_values,
        )

    def initial_state(self, *, device: torch.device) -> PlasticityState:
        dtype = next(self.parameters()).dtype
        return self.bank.initial_state(device=device, dtype=dtype)

    def predict(
        self,
        entities: torch.Tensor,
        state: PlasticityState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query, base_logits = self.backbone(entities)
        logits, representation, attention = self.bank.read(query, base_logits, state)
        return logits, representation, attention, query


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
        "num_slots",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "plasticity_hidden_dim",
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
    for name in ["lr", "grad_clip"]:
        positive_float(name, getattr(args, name))
    nonnegative_float("weight_decay", args.weight_decay)
    if args.d_model % args.n_heads != 0:
        raise ValueError(f"--d-model={args.d_model} must be divisible by --n-heads={args.n_heads}.")
    if args.initial_entities + args.novel_events > args.num_entities:
        raise ValueError(
            "--initial-entities + --novel-events must not exceed --num-entities; "
            f"got {args.initial_entities} + {args.novel_events} > {args.num_entities}."
        )
    if args.correction_events > args.initial_entities:
        raise ValueError(
            "--correction-events must not exceed --initial-entities because correction targets are unique; "
            f"got {args.correction_events} > {args.initial_entities}."
        )
    if args.num_values < 2:
        raise ValueError("--num-values must be at least 2 to generate corrections and noise.")


def different_value(*, current: int, num_values: int, generator: random.Random) -> int:
    candidates = [value for value in range(num_values) if value != current]
    if not candidates:
        raise RuntimeError("No alternative value exists.")
    return generator.choice(candidates)


def interleave_chains(
    chains: list[list[PlannedObservation]],
    *,
    generator: random.Random,
) -> list[PlannedObservation]:
    active = [list(chain) for chain in chains if chain]
    result: list[PlannedObservation] = []
    while active:
        chain_index = generator.randrange(len(active))
        chain = active[chain_index]
        result.append(chain.pop(0))
        if not chain:
            active.pop(chain_index)
    return result


def generate_episode(args: argparse.Namespace, *, seed: int) -> list[StreamEvent]:
    generator = random.Random(seed)
    entity_order = list(range(args.num_entities))
    generator.shuffle(entity_order)
    initial_entities = entity_order[: args.initial_entities]
    novel_entities = entity_order[args.initial_entities : args.initial_entities + args.novel_events]
    initial_values = {entity: generator.randrange(args.num_values) for entity in initial_entities}
    novel_values = {entity: generator.randrange(args.num_values) for entity in novel_entities}
    initial_plan = [
        PlannedObservation(kind="initial", entity=entity, value=initial_values[entity])
        for entity in initial_entities
    ]
    generator.shuffle(initial_plan)

    correction_entities = generator.sample(initial_entities, args.correction_events)
    chains: list[list[PlannedObservation]] = []
    for entity in correction_entities:
        corrected = different_value(current=initial_values[entity], num_values=args.num_values, generator=generator)
        chains.append(
            [PlannedObservation(kind="correction", entity=entity, value=corrected)]
            + [
                PlannedObservation(kind="correction_confirmation", entity=entity, value=corrected)
                for _ in range(args.correction_confirmations)
            ]
        )
    for entity in novel_entities:
        value = novel_values[entity]
        chains.append(
            [PlannedObservation(kind="novel", entity=entity, value=value)]
            + [
                PlannedObservation(kind="novel_confirmation", entity=entity, value=value)
                for _ in range(args.novel_confirmations)
            ]
        )
    chains.extend(
        [PlannedObservation(kind="stable", entity=None, value=None)]
        for _ in range(args.stable_events)
    )
    chains.extend(
        [PlannedObservation(kind="noise", entity=None, value=None)]
        for _ in range(args.noise_events)
    )
    plan = initial_plan + interleave_chains(chains, generator=generator)

    truth: dict[int, int] = {}
    events: list[StreamEvent] = []
    for observation in plan:
        if observation.kind == "initial":
            if observation.entity is None or observation.value is None:
                raise RuntimeError("Initial observation must specify entity and value.")
            truth[observation.entity] = observation.value
            entity = observation.entity
            observed = observation.value
        elif observation.kind == "novel":
            if observation.entity is None or observation.value is None:
                raise RuntimeError("Novel observation must specify entity and value.")
            if observation.entity in truth:
                raise RuntimeError(f"Novel entity {observation.entity} is already active.")
            truth[observation.entity] = observation.value
            entity = observation.entity
            observed = observation.value
        elif observation.kind == "correction":
            if observation.entity is None or observation.value is None:
                raise RuntimeError("Correction observation must specify entity and value.")
            if observation.entity not in truth:
                raise RuntimeError(f"Correction entity {observation.entity} is not active.")
            truth[observation.entity] = observation.value
            entity = observation.entity
            observed = observation.value
        elif observation.kind in {"correction_confirmation", "novel_confirmation"}:
            if observation.entity is None or observation.value is None:
                raise RuntimeError("Confirmation observation must specify entity and value.")
            if truth.get(observation.entity) != observation.value:
                raise RuntimeError(
                    f"Confirmation mismatch for entity {observation.entity}: "
                    f"truth={truth.get(observation.entity)} expected={observation.value}."
                )
            entity = observation.entity
            observed = observation.value
        elif observation.kind == "stable":
            if not truth:
                raise RuntimeError("Stable observation requires at least one active entity.")
            entity = generator.choice(sorted(truth))
            observed = truth[entity]
        elif observation.kind == "noise":
            if not truth:
                raise RuntimeError("Noise observation requires at least one active entity.")
            entity = generator.choice(sorted(truth))
            observed = different_value(current=truth[entity], num_values=args.num_values, generator=generator)
        else:
            raise ValueError(f"Unhandled observation kind {observation.kind!r}.")
        active_entities = tuple(sorted(truth))
        active_values = tuple(truth[item] for item in active_entities)
        events.append(
            StreamEvent(
                kind=observation.kind,
                entity=entity,
                observed_value=observed,
                true_value=truth[entity],
                active_entities=active_entities,
                active_values=active_values,
            )
        )
    return events


def centered_kernel_alignment(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"CKA inputs must share a two-dimensional shape, got {tuple(x.shape)} and {tuple(y.shape)}.")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x.transpose(0, 1) @ y).square()
    denominator = torch.linalg.matrix_norm(x.transpose(0, 1) @ x) * torch.linalg.matrix_norm(y.transpose(0, 1) @ y)
    if float(denominator.detach().cpu()) <= 0.0:
        raise RuntimeError("Cannot compute CKA with zero centered representation norm.")
    return float((cross / denominator).detach().cpu())


def run_episode(
    *,
    model: ProbabilisticPlasticityModel,
    events: list[StreamEvent],
    device: torch.device,
    mode: str,
    collect_details: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode not in {"learned", "uniform", "no_update"}:
        raise ValueError(f"Unknown episode mode {mode!r}.")
    state = model.initial_state(device=device)
    geometry_entities = torch.arange(model.backbone.num_entities, dtype=torch.long, device=device)
    _geometry_logits, initial_representation, _geometry_attention, _geometry_query = model.predict(
        geometry_entities,
        state,
    )
    losses: list[torch.Tensor] = []
    event_rows: list[dict[str, Any]] = []
    action_by_kind: dict[str, list[torch.Tensor]] = defaultdict(list)
    action_posteriors: list[torch.Tensor] = []
    attention_rows: list[torch.Tensor] = []
    noise_adoptions: list[float] = []
    final_representation: torch.Tensor | None = None
    final_entities: torch.Tensor | None = None
    final_values: torch.Tensor | None = None

    for event_index, event in enumerate(events):
        event_entity = torch.tensor([event.entity], dtype=torch.long, device=device)
        observed = torch.tensor([event.observed_value], dtype=torch.long, device=device)
        event_logits, _event_representation, event_attention, event_query = model.predict(event_entity, state)
        if mode == "no_update":
            attended_posterior = event_attention @ model.bank.posterior(state)
            attended_posterior = attended_posterior.squeeze(0)
        else:
            state, attended_posterior = model.bank.observe(
                query=event_query,
                logits=event_logits,
                observed_value=observed,
                attention=event_attention,
                state=state,
                posterior_mode=mode,
            )
        action_by_kind[event.kind].append(attended_posterior)
        action_posteriors.append(attended_posterior)
        attention_rows.append(event_attention.squeeze(0))

        active_entities = torch.tensor(event.active_entities, dtype=torch.long, device=device)
        active_values = torch.tensor(event.active_values, dtype=torch.long, device=device)
        logits, representation, _attention, _query = model.predict(active_entities, state)
        losses.append(F.cross_entropy(logits, active_values))
        predictions = logits.argmax(dim=1)
        exact = float((predictions == active_values).to(torch.float32).mean().detach().cpu())
        if event.kind == "noise":
            event_position = event.active_entities.index(event.entity)
            noise_adoptions.append(float(predictions[event_position].item() == event.observed_value))
        final_representation = representation
        final_entities = active_entities
        final_values = active_values
        if collect_details:
            event_rows.append(
                {
                    "index": event_index,
                    "kind": event.kind,
                    "entity": event.entity,
                    "observed_value": event.observed_value,
                    "true_value": event.true_value,
                    "active_accuracy": exact,
                    "posterior": {
                        name: float(attended_posterior[index].detach().cpu())
                        for index, name in enumerate(ACTION_NAMES)
                    },
                }
            )
    if not losses or final_representation is None or final_entities is None or final_values is None:
        raise RuntimeError("Episode produced no evaluable events.")
    final_logits, final_representation, _final_attention, _final_query = model.predict(final_entities, state)
    final_predictions = final_logits.argmax(dim=1)
    final_accuracy = float((final_predictions == final_values).to(torch.float32).mean().detach().cpu())
    _geometry_logits, final_geometry, _geometry_attention, _geometry_query = model.predict(
        geometry_entities,
        state,
    )
    drift = float(
        (
            torch.linalg.vector_norm(final_geometry - initial_representation)
            / torch.linalg.vector_norm(initial_representation).clamp_min(torch.finfo(initial_representation.dtype).eps)
        ).detach().cpu()
    )
    cka = centered_kernel_alignment(initial_representation, final_geometry)
    action_means = {
        kind: {
            name: float(torch.stack(rows).mean(dim=0)[index].detach().cpu())
            for index, name in enumerate(ACTION_NAMES)
        }
        for kind, rows in sorted(action_by_kind.items())
    }
    state_posterior = model.bank.posterior(state)
    state_posterior_entropy = float(
        (-(state_posterior * state_posterior.clamp_min(torch.finfo(state_posterior.dtype).eps).log()).sum(dim=1).mean()).detach().cpu()
    )
    actual_posteriors = torch.stack(action_posteriors)
    action_posterior_entropy = float(
        (
            -(actual_posteriors * actual_posteriors.clamp_min(torch.finfo(actual_posteriors.dtype).eps).log())
            .sum(dim=1)
            .mean()
        ).detach().cpu()
    )
    attentions = torch.stack(attention_rows)
    attention_entropy = float(
        (
            -(attentions * attentions.clamp_min(torch.finfo(attentions.dtype).eps).log())
            .sum(dim=1)
            .mean()
        ).detach().cpu()
    )
    mass_distribution = state.mass / state.mass.sum().clamp_min(torch.finfo(state.mass.dtype).eps)
    mass_entropy = -(
        mass_distribution * mass_distribution.clamp_min(torch.finfo(mass_distribution.dtype).eps).log()
    ).sum()
    effective_slots = float(torch.exp(mass_entropy).detach().cpu())
    top_slot_share = float(mass_distribution.max().detach().cpu())
    report = {
        "mean_future_loss": float(torch.stack(losses).mean().detach().cpu()),
        "final_accuracy": final_accuracy,
        "noise_adoption_rate": sum(noise_adoptions) / float(len(noise_adoptions)) if noise_adoptions else None,
        "representation_drift": drift,
        "representation_cka": cka,
        "action_posterior_entropy": action_posterior_entropy,
        "state_posterior_entropy": state_posterior_entropy,
        "attention_entropy": attention_entropy,
        "effective_slots": effective_slots,
        "top_slot_share": top_slot_share,
        "slot_mass": [float(value) for value in state.mass.detach().cpu()],
        "action_means_by_event": action_means,
        "events": event_rows,
    }
    return torch.stack(losses).mean(), report


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise RuntimeError("Cannot aggregate an empty report list.")
    scalar_names = [
        "mean_future_loss",
        "final_accuracy",
        "representation_drift",
        "representation_cka",
        "action_posterior_entropy",
        "state_posterior_entropy",
        "attention_entropy",
        "effective_slots",
        "top_slot_share",
    ]
    aggregate = {
        name: sum(float(report[name]) for report in reports) / float(len(reports))
        for name in scalar_names
    }
    noise_values = [float(report["noise_adoption_rate"]) for report in reports if report["noise_adoption_rate"] is not None]
    aggregate["noise_adoption_rate"] = (
        sum(noise_values) / float(len(noise_values)) if noise_values else None
    )
    action_rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for report in reports:
        for kind, values in report["action_means_by_event"].items():
            for action, value in values.items():
                action_rows[kind][action].append(float(value))
    aggregate["action_means_by_event"] = {
        kind: {
            action: sum(values) / float(len(values))
            for action, values in sorted(action_map.items())
        }
        for kind, action_map in sorted(action_rows.items())
    }
    return aggregate


def train_model(args: argparse.Namespace, *, device: torch.device) -> tuple[ProbabilisticPlasticityModel, list[dict[str, float]]]:
    model = ProbabilisticPlasticityModel(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trace: list[dict[str, float]] = []
    for episode_index in tqdm(range(args.train_episodes), desc="train probabilistic plasticity"):
        events = generate_episode(args, seed=args.seed + episode_index)
        optimizer.zero_grad(set_to_none=True)
        loss, report = run_episode(
            model=model,
            events=events,
            device=device,
            mode="learned",
            collect_details=False,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at episode {episode_index}: {float(loss.detach().cpu())}.")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().cpu())
        optimizer.step()
        row = {
            "episode": float(episode_index + 1),
            "loss": float(loss.detach().cpu()),
            "final_accuracy": float(report["final_accuracy"]),
            "action_posterior_entropy": float(report["action_posterior_entropy"]),
            "effective_slots": float(report["effective_slots"]),
            "grad_norm": grad_norm,
        }
        trace.append(row)
        if episode_index == 0 or episode_index + 1 == args.train_episodes or (episode_index + 1) % args.print_every == 0:
            print(
                f"episode={episode_index + 1:4d} loss={row['loss']:.5f} "
                f"final_acc={row['final_accuracy']:.3f} entropy={row['action_posterior_entropy']:.3f} "
                f"slots={row['effective_slots']:.2f} "
                f"grad={row['grad_norm']:.3f}"
            )
    return model, trace


@torch.no_grad()
def evaluate_modes(
    args: argparse.Namespace,
    *,
    model: ProbabilisticPlasticityModel,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    reports_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in ["learned", "uniform", "no_update"]}
    detailed: dict[str, Any] = {}
    for test_index in range(args.test_episodes):
        events = generate_episode(args, seed=args.seed + args.train_episodes + test_index)
        for mode in reports_by_mode:
            _loss, report = run_episode(
                model=model,
                events=events,
                device=device,
                mode=mode,
                collect_details=test_index == 0,
            )
            reports_by_mode[mode].append(report)
            if test_index == 0:
                detailed[mode] = {
                    "stream": [asdict(event) for event in events],
                    "report": report,
                }
    aggregate = {mode: aggregate_reports(reports) for mode, reports in reports_by_mode.items()}
    return aggregate, detailed


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("TINY PROBABILISTIC PLASTICITY ATTENTION")
    print("=" * 120)
    print(
        f"device={device} train_episodes={args.train_episodes} test_episodes={args.test_episodes} "
        f"slots={args.num_slots} actions={ACTION_NAMES}"
    )
    model, training_trace = train_model(args, device=device)
    aggregate, detailed = evaluate_modes(args, model=model, device=device)

    print("\nPROBABILISTIC PLASTICITY SUMMARY")
    print("=" * 120)
    print(
        f"{'mode':>12} {'loss':>10} {'finalAcc':>10} {'noise':>10} {'drift':>10} "
        f"{'cka':>10} {'actH':>10} {'slots':>10} {'topSlot':>10}"
    )
    for mode in ["learned", "uniform", "no_update"]:
        row = aggregate[mode]
        noise = "NA" if row["noise_adoption_rate"] is None else f"{row['noise_adoption_rate']:.4f}"
        print(
            f"{mode:>12} {row['mean_future_loss']:10.4f} {row['final_accuracy']:10.4f} "
            f"{noise:>10} {row['representation_drift']:10.4f} {row['representation_cka']:10.4f} "
            f"{row['action_posterior_entropy']:10.4f} {row['effective_slots']:10.4f} "
            f"{row['top_slot_share']:10.4f}"
        )
    print("\nLEARNED ACTION POSTERIOR BY HIDDEN EVENT TYPE (EVALUATION ONLY)")
    print("-" * 120)
    for kind, values in aggregate["learned"]["action_means_by_event"].items():
        formatted = " ".join(f"{name}={values[name]:.3f}" for name in ACTION_NAMES)
        print(f"{kind:>26} {formatted}")

    output = {
        "question": (
            "Can continuous feature-level plasticity probabilities emerge from temporal data and future prediction "
            "loss without role or action labels?"
        ),
        "scope": (
            "This tests probabilistic feature plasticity. Learned probabilities are not yet connected to the "
            "Invariant-Tangent parameter-space executor."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "action_names": list(ACTION_NAMES),
        "training_trace": training_trace,
        "aggregate": aggregate,
        "first_test_episode": detailed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("model/analysis/gco-tiny-probabilistic-plasticity-attention-seed0.json"),
    )
    parser.add_argument("--num-entities", type=int, default=12)
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--initial-entities", type=int, default=6)
    parser.add_argument("--num-slots", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--plasticity-hidden-dim", type=int, default=64)
    parser.add_argument("--stable-events", type=int, default=12)
    parser.add_argument("--novel-events", type=int, default=4)
    parser.add_argument("--correction-events", type=int, default=3)
    parser.add_argument("--noise-events", type=int, default=6)
    parser.add_argument("--novel-confirmations", type=int, default=2)
    parser.add_argument("--correction-confirmations", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
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
