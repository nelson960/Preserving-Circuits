"""Single-model semantic latent-geometry write-control experiment.

This is a deliberately small benchmark for the next research question:

    Can a training-time write controller decide whether a semantic event should
    be discarded, reused, composed from existing knowledge, allocated into
    unused internal capacity, or used to rewrite an old belief?

The model is one PyTorch module. It contains token embeddings, relation
embeddings, and internal attention-addressed memory slots. Inference is a normal
forward pass:

    subject + relation chain -> memory attention -> decoded object

The latent-geometry reasoner is used only during learning. It creates candidate
weight updates on a shadow copy, measures whether each candidate improves the
current semantic event while preserving the active belief set, and commits only
the selected candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


ACTIONS = ("discard", "reuse", "compose", "allocate", "rewrite", "update")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
ROLE_NAMES = ("subject", "relation", "object", "time", "source", "evidence")
ROLE_TO_INDEX = {role: index for index, role in enumerate(ROLE_NAMES)}
SOURCE_NAMES = ("trusted", "untrusted")
SOURCE_TO_INDEX = {source: index for index, source in enumerate(SOURCE_NAMES)}
MAX_EVIDENCE_INDEX = 5
FEATURE_ABLATIONS = (
    "role",
    "position",
    "policy_identity",
    "policy_position",
    "policy_time",
    "policy_source",
    "policy_evidence",
)
CONSOLIDATION_ADMISSIONS = ("current", "same_relation", "composition_preserving")
CONSOLIDATION_GROUP_ORDERS = ("current", "same_relation_first", "geometry")


@dataclass(frozen=True)
class World:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]
    relation_to_id: dict[str, int]
    id_to_relation: tuple[str, ...]


@dataclass(frozen=True)
class SemanticEvent:
    name: str
    subject: str
    relations: tuple[str, ...]
    target: str
    reliable: bool
    expected_action: str
    commit_truth: bool
    evidence: int
    timestamp: int = 0
    source: str = "trusted"
    subject_pos: int = 0
    relation_pos: int = 1
    object_pos: int = 2

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return (self.subject, self.relations)

    @property
    def is_one_hop(self) -> bool:
        return len(self.relations) == 1


@dataclass(frozen=True)
class CandidateResult:
    label: str
    action: str
    model: "SemanticMemoryModel"
    new_acc: float
    active_acc: float
    protected_acc: float
    closure: float
    slot_index: int | None
    slot_usage: float
    key_cosine: float | None
    value_cosine: float | None
    target_attention: float | None
    attention_margin: float | None
    changed: bool
    score: float


@dataclass
class ConsolidationStats:
    attempts: int = 0
    commits: int = 0
    rejected: int = 0
    no_capacity: int = 0
    freed_slots: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, int]:
        return {
            "attempts": self.attempts,
            "commits": self.commits,
            "rejected": self.rejected,
            "no_capacity": self.no_capacity,
            "freed_slots": self.freed_slots,
        }


class SemanticMemoryModel(nn.Module):
    """A small internal-memory model with always-on memory attention."""

    def __init__(
        self,
        vocab_size: int,
        relation_count: int,
        d_model: int,
        num_slots: int,
        temperature: float,
        max_parents: int = 0,
        parent_confidence_weight: float = 0.5,
        use_role_embeddings: bool = True,
        use_position_encoding: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if max_parents < 0:
            raise ValueError(f"max_parents must be non-negative, got {max_parents}.")
        if parent_confidence_weight < 0.0:
            raise ValueError(f"parent_confidence_weight must be non-negative, got {parent_confidence_weight}.")

        self.d_model = d_model
        self.num_slots = num_slots
        self.temperature = temperature
        self.max_parents = max_parents
        self.parent_confidence_weight = parent_confidence_weight
        self.use_role_embeddings = use_role_embeddings
        self.use_position_encoding = use_position_encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.relation_embedding = nn.Embedding(relation_count, d_model)
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), d_model)
        self.source_embedding = nn.Embedding(len(SOURCE_NAMES), d_model)
        self.evidence_embedding = nn.Embedding(MAX_EVIDENCE_INDEX + 1, d_model)
        self.memory_keys = nn.Parameter(torch.empty(num_slots, d_model))
        self.memory_values = nn.Parameter(torch.empty(num_slots, d_model))
        self.parent_keys = nn.Parameter(torch.empty(max_parents, d_model))
        self.parent_scale = nn.Parameter(torch.empty(max_parents, d_model))
        self.parent_bias = nn.Parameter(torch.empty(max_parents, d_model))
        self.register_buffer("parent_active", torch.zeros(max_parents, dtype=torch.bool))
        self.state_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.relation_embedding.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.source_embedding.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.evidence_embedding.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.memory_keys, mean=0.0, std=0.2)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.2)
        if self.max_parents > 0:
            nn.init.normal_(self.parent_keys, mean=0.0, std=0.2)
            nn.init.ones_(self.parent_scale)
            nn.init.zeros_(self.parent_bias)
            self.parent_active.zero_()

    def scalar_sincos(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 1:
            raise ValueError(f"value must be rank-1 [batch], got {tuple(value.shape)}.")
        half_dim = self.d_model // 2
        if half_dim <= 0:
            raise ValueError(f"d_model={self.d_model} is too small for sin/cos encoding.")
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32, device=value.device)
            / max(1.0, float(half_dim - 1))
        )
        angles = value.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if encoding.shape[-1] < self.d_model:
            encoding = F.pad(encoding, (0, self.d_model - encoding.shape[-1]))
        return encoding[:, : self.d_model]

    def role_vector(self, role: str, batch: int, device: torch.device) -> torch.Tensor:
        if role not in ROLE_TO_INDEX:
            raise ValueError(f"Unknown role {role!r}.")
        if not self.use_role_embeddings:
            return torch.zeros(batch, self.d_model, device=device)
        role_ids = torch.full((batch,), ROLE_TO_INDEX[role], dtype=torch.long, device=device)
        return self.role_embedding(role_ids)

    def position_vector(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 1:
            raise ValueError(f"positions must be rank-1 [batch], got {tuple(positions.shape)}.")
        if not self.use_position_encoding:
            return torch.zeros(positions.shape[0], self.d_model, device=positions.device)
        return self.scalar_sincos(positions)

    def positioned_token(self, token_ids: torch.Tensor, role: str, pos: int) -> torch.Tensor:
        position = torch.full((token_ids.shape[0],), pos, dtype=torch.float32, device=token_ids.device)
        return self.positioned_tokens(token_ids, role, position)

    def positioned_tokens(self, token_ids: torch.Tensor, role: str, positions: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 1:
            raise ValueError(f"token_ids must be rank-1 [batch], got {tuple(token_ids.shape)}.")
        if positions.ndim != 1:
            raise ValueError(f"positions must be rank-1 [batch], got {tuple(positions.shape)}.")
        if token_ids.shape[0] != positions.shape[0]:
            raise ValueError("token_ids and positions batch size mismatch.")
        return (
            self.token_embedding(token_ids)
            + self.role_vector(role, token_ids.shape[0], token_ids.device)
            + self.position_vector(positions)
        )

    def relation_query(self, state: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        relation_pos = torch.full((relation_ids.shape[0],), 1, dtype=torch.float32, device=relation_ids.device)
        relation = (
            self.relation_embedding(relation_ids)
            + self.role_vector("relation", relation_ids.shape[0], relation_ids.device)
            + self.position_vector(relation_pos)
        )
        query = self.query_norm(state + relation)
        return F.normalize(query, dim=-1)

    def normalized_keys(self) -> torch.Tensor:
        return F.normalize(self.memory_keys, dim=-1)

    def active_parent_count(self) -> int:
        return int(self.parent_active.sum().item())

    def active_parent_mask(self) -> torch.Tensor:
        if self.parent_active.ndim != 1:
            raise RuntimeError("parent_active buffer must be rank-1.")
        return self.parent_active

    def forward(
        self,
        subject_ids: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if subject_ids.ndim != 1:
            raise ValueError(f"subject_ids must be rank-1 [batch], got {tuple(subject_ids.shape)}.")
        if relation_ids.ndim != 2:
            raise ValueError(f"relation_ids must be rank-2 [batch, steps], got {tuple(relation_ids.shape)}.")
        if subject_ids.shape[0] != relation_ids.shape[0]:
            raise ValueError("subject_ids and relation_ids batch size mismatch.")

        state = self.positioned_token(subject_ids, "subject", pos=0)
        attentions: list[torch.Tensor] = []
        queries: list[torch.Tensor] = []
        keys = self.normalized_keys()
        for step in range(relation_ids.shape[1]):
            relation_pos = torch.full(
                (relation_ids.shape[0],), step + 1, dtype=torch.float32, device=relation_ids.device
            )
            relation = (
                self.relation_embedding(relation_ids[:, step])
                + self.role_vector("relation", relation_ids.shape[0], relation_ids.device)
                + self.position_vector(relation_pos)
            )
            query = F.normalize(self.query_norm(state + relation), dim=-1)
            slot_scores = query @ keys.T / self.temperature
            slot_values = self.memory_values.unsqueeze(0).expand(query.shape[0], -1, -1)
            if self.max_parents > 0 and bool(self.parent_active.any().item()):
                active_parent_mask = self.active_parent_mask()
                parent_keys = F.normalize(self.parent_keys[active_parent_mask], dim=-1)
                parent_key_scores = query @ parent_keys.T
                parent_values = (
                    query.unsqueeze(1) * self.parent_scale[active_parent_mask].unsqueeze(0)
                    + self.parent_bias[active_parent_mask].unsqueeze(0)
                )
                token_basis = F.normalize(self.token_embedding.weight, dim=-1)
                parent_value_confidence = torch.einsum(
                    "bpd,vd->bpv",
                    F.normalize(parent_values, dim=-1),
                    token_basis,
                ).max(dim=-1).values
                parent_scores = (
                    parent_key_scores + self.parent_confidence_weight * parent_value_confidence
                ) / self.temperature
                scores = torch.cat([slot_scores, parent_scores], dim=1)
                values = torch.cat([slot_values, parent_values], dim=1)
            else:
                scores = slot_scores
                values = slot_values
            attention = F.softmax(scores, dim=-1)
            state = self.state_norm(torch.bmm(attention.unsqueeze(1), values).squeeze(1))
            queries.append(query)
            attentions.append(attention)

        logits = state @ self.token_embedding.weight.T
        diagnostics = {
            "state": state,
            "attentions": torch.stack(attentions, dim=1),
            "queries": torch.stack(queries, dim=1),
        }
        return logits, diagnostics

    def event_context(
        self,
        world: World,
        event: SemanticEvent,
        device: torch.device,
    ) -> torch.Tensor:
        if event.source not in SOURCE_TO_INDEX:
            raise ValueError(f"Unknown event source {event.source!r}.")
        subject = torch.tensor([world.token_to_id[event.subject]], dtype=torch.long, device=device)
        target = torch.tensor([world.token_to_id[event.target]], dtype=torch.long, device=device)
        relation_ids = torch.tensor(
            [world.relation_to_id[relation] for relation in event.relations],
            dtype=torch.long,
            device=device,
        )
        relation_positions = torch.arange(
            event.relation_pos,
            event.relation_pos + len(event.relations),
            dtype=torch.float32,
            device=device,
        )
        relation_vectors = (
            self.relation_embedding(relation_ids)
            + self.role_vector("relation", len(event.relations), device)
            + self.position_vector(relation_positions)
        )
        source_id = torch.tensor([SOURCE_TO_INDEX[event.source]], dtype=torch.long, device=device)
        evidence_id = torch.tensor([min(MAX_EVIDENCE_INDEX, event.evidence)], dtype=torch.long, device=device)
        time_value = torch.tensor([event.timestamp], dtype=torch.float32, device=device)
        parts = torch.stack(
            [
                self.positioned_token(subject, "subject", event.subject_pos)[0],
                relation_vectors.mean(dim=0),
                self.positioned_token(target, "object", event.object_pos)[0],
                self.position_vector(time_value)[0] + self.role_vector("time", 1, device)[0],
                self.source_embedding(source_id)[0] + self.role_vector("source", 1, device)[0],
                self.evidence_embedding(evidence_id)[0] + self.role_vector("evidence", 1, device)[0],
            ],
            dim=0,
        )
        return self.context_norm(parts.mean(dim=0))


class CandidateWritePolicy(nn.Module):
    """Scores candidate write futures from context and geometry features."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must be rank-2 [candidates, dim], got {tuple(features.shape)}.")
        return self.net(features).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown device {name!r}. Expected cpu, cuda, or mps.")


def build_world() -> World:
    entities = (
        "alice",
        "bob",
        "carol",
        "dave",
        "erin",
        "paris",
        "rome",
        "london",
        "delhi",
        "oslo",
        "france",
        "italy",
        "uk",
        "india",
        "norway",
        "cat",
        "dog",
        "bird",
        "red",
        "blue",
        "green",
    )
    relations = ("lives_in", "country_of", "pet", "color", "parent")
    return World(
        token_to_id={token: index for index, token in enumerate(entities)},
        id_to_token=entities,
        relation_to_id={relation: index for index, relation in enumerate(relations)},
        id_to_relation=relations,
    )


def make_model(
    world: World,
    args: argparse.Namespace,
    device: torch.device,
    base_state: dict[str, torch.Tensor] | None = None,
) -> SemanticMemoryModel:
    model = SemanticMemoryModel(
        vocab_size=len(world.id_to_token),
        relation_count=len(world.id_to_relation),
        d_model=args.d_model,
        num_slots=args.num_slots,
        temperature=args.temperature,
        max_parents=args.max_parents,
        parent_confidence_weight=args.parent_confidence_weight,
        use_role_embeddings=not args.disable_role_embeddings,
        use_position_encoding=not args.disable_position_encoding,
    ).to(device)
    if base_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in base_state.items()}, strict=True)
    return model


def rotated(items: tuple[str, ...], amount: int) -> tuple[str, ...]:
    if not items:
        raise ValueError("Cannot rotate an empty tuple.")
    amount = amount % len(items)
    return items[amount:] + items[:amount]


def build_stream(seed: int) -> list[SemanticEvent]:
    names = rotated(("alice", "bob", "carol", "dave", "erin"), seed)
    cities = rotated(("paris", "rome", "london", "delhi", "oslo"), seed)
    countries = {
        "paris": "france",
        "rome": "italy",
        "london": "uk",
        "delhi": "india",
        "oslo": "norway",
    }
    pets = rotated(("cat", "dog", "bird"), seed)
    colors = rotated(("red", "blue", "green"), seed)

    subject = names[0]
    stable_subject = names[1]
    parent_subject = names[2]
    first_city = cities[0]
    conflict_city = cities[1]
    first_country = countries[first_city]
    conflict_country = countries[conflict_city]

    events = [
        SemanticEvent(
            name="country_base_1",
            subject=first_city,
            relations=("country_of",),
            target=first_country,
            reliable=True,
            expected_action="allocate",
            commit_truth=True,
            evidence=1,
        ),
        SemanticEvent(
            name="country_base_2",
            subject=conflict_city,
            relations=("country_of",),
            target=conflict_country,
            reliable=True,
            expected_action="allocate",
            commit_truth=True,
            evidence=1,
        ),
        SemanticEvent(
            name="new_lives_fact",
            subject=subject,
            relations=("lives_in",),
            target=first_city,
            reliable=True,
            expected_action="allocate",
            commit_truth=True,
            evidence=1,
        ),
        SemanticEvent(
            name="position_swapped_noise",
            subject=first_city,
            relations=("lives_in",),
            target=subject,
            reliable=False,
            expected_action="discard",
            commit_truth=False,
            evidence=1,
            subject_pos=0,
            relation_pos=1,
            object_pos=2,
        ),
        SemanticEvent(
            name="repeat_lives_fact",
            subject=subject,
            relations=("lives_in",),
            target=first_city,
            reliable=True,
            expected_action="reuse",
            commit_truth=False,
            evidence=2,
        ),
        SemanticEvent(
            name="composition_country",
            subject=subject,
            relations=("lives_in", "country_of"),
            target=first_country,
            reliable=True,
            expected_action="compose",
            commit_truth=False,
            evidence=1,
        ),
        SemanticEvent(
            name="unreliable_conflict",
            subject=subject,
            relations=("lives_in",),
            target=conflict_city,
            reliable=False,
            expected_action="discard",
            commit_truth=False,
            evidence=1,
        ),
        SemanticEvent(
            name="repeated_reliable_conflict",
            subject=subject,
            relations=("lives_in",),
            target=conflict_city,
            reliable=True,
            expected_action="rewrite",
            commit_truth=True,
            evidence=3,
        ),
        SemanticEvent(
            name="composition_after_rewrite",
            subject=subject,
            relations=("lives_in", "country_of"),
            target=conflict_country,
            reliable=True,
            expected_action="compose",
            commit_truth=False,
            evidence=1,
        ),
        SemanticEvent(
            name="stable_pet",
            subject=stable_subject,
            relations=("pet",),
            target=pets[0],
            reliable=True,
            expected_action="allocate",
            commit_truth=True,
            evidence=1,
        ),
        SemanticEvent(
            name="stable_color",
            subject=pets[0],
            relations=("color",),
            target=colors[0],
            reliable=True,
            expected_action="allocate",
            commit_truth=True,
            evidence=1,
        ),
        SemanticEvent(
            name="pet_color_composition",
            subject=stable_subject,
            relations=("pet", "color"),
            target=colors[0],
            reliable=True,
            expected_action="compose",
            commit_truth=False,
            evidence=1,
        ),
        SemanticEvent(
            name="parent_fact",
            subject=parent_subject,
            relations=("parent",),
            target=stable_subject,
            reliable=True,
            expected_action="allocate",
            commit_truth=True,
            evidence=1,
        ),
        SemanticEvent(
            name="parent_pet_composition",
            subject=parent_subject,
            relations=("parent", "pet"),
            target=pets[0],
            reliable=True,
            expected_action="compose",
            commit_truth=False,
            evidence=1,
        ),
    ]
    return [
        replace(
            event,
            timestamp=index,
            source="trusted" if event.reliable else "untrusted",
        )
        for index, event in enumerate(events)
    ]


def event_to_tensors(
    world: World,
    event: SemanticEvent,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    subject_id = world.token_to_id[event.subject]
    relation_ids = [world.relation_to_id[relation] for relation in event.relations]
    target_id = world.token_to_id[event.target]
    return (
        torch.tensor([subject_id], dtype=torch.long, device=device),
        torch.tensor([relation_ids], dtype=torch.long, device=device),
        torch.tensor([target_id], dtype=torch.long, device=device),
    )


def batch_events(
    world: World,
    events: list[SemanticEvent],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not events:
        raise ValueError("Cannot batch an empty event list.")
    steps = len(events[0].relations)
    if any(len(event.relations) != steps for event in events):
        raise ValueError("All batched events must have the same number of relations.")
    subjects = [world.token_to_id[event.subject] for event in events]
    relations = [[world.relation_to_id[relation] for relation in event.relations] for event in events]
    targets = [world.token_to_id[event.target] for event in events]
    return (
        torch.tensor(subjects, dtype=torch.long, device=device),
        torch.tensor(relations, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def compute_loss(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    device: torch.device,
    lambda_closure: float,
) -> torch.Tensor:
    subject_ids, relation_ids, target_ids = event_to_tensors(world, event, device)
    logits, diagnostics = model(subject_ids, relation_ids)
    ce = F.cross_entropy(logits, target_ids)
    target_code = model.token_embedding(target_ids)
    closure = F.mse_loss(diagnostics["state"], target_code)
    return ce + lambda_closure * closure


@torch.no_grad()
def evaluate_events(
    model: SemanticMemoryModel,
    world: World,
    events: list[SemanticEvent],
    device: torch.device,
) -> float:
    if not events:
        return 1.0
    groups: dict[int, list[SemanticEvent]] = {}
    for event in events:
        groups.setdefault(len(event.relations), []).append(event)
    correct = 0
    total = 0
    for grouped_events in groups.values():
        subject_ids, relation_ids, target_ids = batch_events(world, grouped_events, device)
        logits, _ = model(subject_ids, relation_ids)
        predictions = logits.argmax(dim=-1)
        correct += int((predictions == target_ids).sum().item())
        total += len(grouped_events)
    if total <= 0:
        raise ValueError("No events were evaluated.")
    return correct / float(total)


@torch.no_grad()
def evaluate_closure(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    device: torch.device,
) -> float:
    subject_ids, relation_ids, target_ids = event_to_tensors(world, event, device)
    _, diagnostics = model(subject_ids, relation_ids)
    target_code = model.token_embedding(target_ids)
    return float(F.mse_loss(diagnostics["state"], target_code).item())


@torch.no_grad()
def top_attention_slot(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    device: torch.device,
) -> int:
    subject_ids, relation_ids, _ = event_to_tensors(world, event, device)
    _, diagnostics = model(subject_ids, relation_ids)
    attention = diagnostics["attentions"][0, -1, : model.num_slots]
    return int(attention.argmax().item())


def least_used_slot(slot_use: torch.Tensor) -> int:
    if slot_use.ndim != 1:
        raise ValueError(f"slot_use must be rank-1, got {tuple(slot_use.shape)}.")
    return int(slot_use.argmin().item())


def one_hop_query(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    device: torch.device,
) -> torch.Tensor:
    if not event.is_one_hop:
        raise ValueError(f"one_hop_query requires a one-hop event, got {event.name}.")
    subject_id = torch.tensor([world.token_to_id[event.subject]], dtype=torch.long, device=device)
    relation_id = torch.tensor([world.relation_to_id[event.relations[0]]], dtype=torch.long, device=device)
    state = model.positioned_token(subject_id, "subject", pos=event.subject_pos)
    relation_pos = torch.tensor([event.relation_pos], dtype=torch.float32, device=device)
    relation = (
        model.relation_embedding(relation_id)
        + model.role_vector("relation", 1, device)
        + model.position_vector(relation_pos)
    )
    return F.normalize(model.query_norm(state + relation), dim=-1)


def reliable_one_hop_events_from_stream(
    seed_offset: int,
    seed_count: int,
) -> list[SemanticEvent]:
    if seed_offset < 0:
        raise ValueError(f"seed_offset must be non-negative, got {seed_offset}.")
    if seed_count <= 0:
        raise ValueError(f"seed_count must be positive, got {seed_count}.")
    events: list[SemanticEvent] = []
    for seed in range(seed_offset, seed_offset + seed_count):
        events.extend(
            event
            for event in build_stream(seed)
            if event.is_one_hop and event.reliable and event.commit_truth
        )
    if not events:
        raise RuntimeError("No reliable one-hop events were available for latent geometry warmup.")
    return events


def warmup_query_batch(
    model: SemanticMemoryModel,
    world: World,
    events: list[SemanticEvent],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not events:
        raise ValueError("Cannot build a warmup batch from an empty event list.")
    if any(not event.is_one_hop for event in events):
        raise ValueError("Warmup query batch only supports one-hop events.")
    subject_ids = torch.tensor([world.token_to_id[event.subject] for event in events], dtype=torch.long, device=device)
    subject_positions = torch.tensor([event.subject_pos for event in events], dtype=torch.float32, device=device)
    relation_ids = torch.tensor(
        [world.relation_to_id[event.relations[0]] for event in events],
        dtype=torch.long,
        device=device,
    )
    relation_positions = torch.tensor([event.relation_pos for event in events], dtype=torch.float32, device=device)
    state = model.positioned_tokens(subject_ids, "subject", subject_positions)
    relation = (
        model.relation_embedding(relation_ids)
        + model.role_vector("relation", len(events), device)
        + model.position_vector(relation_positions)
    )
    queries = F.normalize(model.query_norm(state + relation), dim=-1)
    target_ids = torch.tensor([world.token_to_id[event.target] for event in events], dtype=torch.long, device=device)
    return queries, relation_ids, target_ids


def embedding_separation_loss(model: SemanticMemoryModel, max_cosine: float) -> torch.Tensor:
    if max_cosine < -1.0 or max_cosine > 1.0:
        raise ValueError(f"max_cosine must be in [-1, 1], got {max_cosine}.")
    codes = F.normalize(model.token_embedding.weight, dim=-1)
    similarity = codes @ codes.T
    off_diag = similarity - torch.eye(similarity.shape[0], device=similarity.device)
    return F.relu(off_diag - max_cosine).pow(2).mean()


def code_norm_loss(model: SemanticMemoryModel) -> torch.Tensor:
    target_norm = math.sqrt(float(model.d_model))
    return (model.token_embedding.weight.norm(dim=-1) - target_norm).pow(2).mean()


def train_latent_geometry_base(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, float | int]]:
    if not args.geometry_warmup:
        return None, {
            "enabled": 0,
            "events": 0,
            "final_loss": 0.0,
            "final_reconstruction": 0.0,
            "final_token_acc": 0.0,
        }
    world = build_world()
    set_seed(args.geometry_seed)
    model = make_model(world, args, device)
    events = reliable_one_hop_events_from_stream(args.geometry_train_seed_offset, args.geometry_train_seed_count)
    relation_scale = nn.Parameter(torch.ones(len(world.id_to_relation), args.d_model, device=device))
    relation_bias = nn.Parameter(torch.zeros(len(world.id_to_relation), args.d_model, device=device))
    optimizer = torch.optim.AdamW(
        [
            model.token_embedding.weight,
            model.relation_embedding.weight,
            model.role_embedding.weight,
            *model.query_norm.parameters(),
            relation_scale,
            relation_bias,
        ],
        lr=args.geometry_warmup_lr,
        weight_decay=0.0,
    )
    progress = tqdm(
        range(args.geometry_warmup_epochs),
        desc="latent geometry warmup",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    final_loss = 0.0
    final_reconstruction = 0.0
    final_token_acc = 0.0
    for _ in progress:
        queries, relation_ids, target_ids = warmup_query_batch(model, world, events, device)
        targets = model.token_embedding(target_ids)
        predictions = queries * relation_scale[relation_ids] + relation_bias[relation_ids]
        reconstruction = F.mse_loss(predictions, targets)
        cosine = (1.0 - F.cosine_similarity(predictions, targets, dim=-1)).mean()
        logits = predictions @ model.token_embedding.weight.T
        token_loss = F.cross_entropy(logits, target_ids)
        separation = embedding_separation_loss(model, args.geometry_max_code_cosine)
        norm = code_norm_loss(model)
        loss = (
            reconstruction
            + cosine
            + args.geometry_token_ce_weight * token_loss
            + args.geometry_separation_weight * separation
            + args.geometry_code_norm_weight * norm
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite latent geometry warmup loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            final_loss = float(loss.item())
            final_reconstruction = float(reconstruction.item())
            final_token_acc = float((logits.argmax(dim=-1) == target_ids).float().mean().item())
        progress.set_postfix(loss=f"{final_loss:.4f}", acc=f"{final_token_acc:.3f}")

    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return base_state, {
        "enabled": 1,
        "events": len(events),
        "final_loss": final_loss,
        "final_reconstruction": final_reconstruction,
        "final_token_acc": final_token_acc,
    }


def slot_write_losses(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    device: torch.device,
    slot_index: int,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if slot_index < 0 or slot_index >= model.num_slots:
        raise ValueError(f"slot_index={slot_index} outside [0, {model.num_slots}).")
    if margin <= 0.0:
        raise ValueError(f"attention margin must be positive, got {margin}.")

    _, _, target_ids = event_to_tensors(world, event, device)
    query = one_hop_query(model, world, event, device)
    keys = model.normalized_keys()
    scores = query @ keys.T
    target_score = scores[:, slot_index]
    if model.num_slots <= 1:
        raise ValueError("Direct slot write losses require at least two memory slots.")
    wrong_scores = torch.cat([scores[:, :slot_index], scores[:, slot_index + 1 :]], dim=1)
    max_wrong_score = wrong_scores.max(dim=1).values
    attention = F.softmax(scores / model.temperature, dim=-1)
    target_attention = attention[:, slot_index]

    target_code = model.token_embedding(target_ids)
    value = model.memory_values[slot_index].unsqueeze(0)
    value_cosine = F.cosine_similarity(value, target_code, dim=-1)

    key_loss = (1.0 - target_score).mean()
    value_loss = F.mse_loss(value, target_code)
    margin_loss = F.relu(margin + max_wrong_score - target_score).mean()
    diagnostics = {
        "key_cosine": target_score.detach().mean(),
        "value_cosine": value_cosine.detach().mean(),
        "target_attention": target_attention.detach().mean(),
        "attention_margin": (target_score - max_wrong_score).detach().mean(),
        "key_loss": key_loss.detach(),
        "value_loss": value_loss.detach(),
        "margin_loss": margin_loss.detach(),
    }
    return key_loss + value_loss + margin_loss, diagnostics


@torch.no_grad()
def slot_write_diagnostics(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    device: torch.device,
    slot_index: int | None,
) -> dict[str, float | None]:
    if slot_index is None or not event.is_one_hop:
        return {
            "key_cosine": None,
            "value_cosine": None,
            "target_attention": None,
            "attention_margin": None,
        }
    _, diagnostics = slot_write_losses(
        model,
        world,
        event,
        device,
        slot_index=slot_index,
        margin=1.0,
    )
    return {
        "key_cosine": float(diagnostics["key_cosine"].item()),
        "value_cosine": float(diagnostics["value_cosine"].item()),
        "target_attention": float(diagnostics["target_attention"].item()),
        "attention_margin": float(diagnostics["attention_margin"].item()),
    }


def mask_gradients_for_slot(model: SemanticMemoryModel, slot_index: int) -> None:
    if slot_index < 0 or slot_index >= model.num_slots:
        raise ValueError(f"slot_index={slot_index} outside [0, {model.num_slots}).")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.zero_()
    if model.memory_keys.grad is None or model.memory_values.grad is None:
        raise RuntimeError("Expected memory key/value gradients before masking.")


def apply_slot_gradient_mask(model: SemanticMemoryModel, slot_index: int) -> None:
    if model.memory_keys.grad is None or model.memory_values.grad is None:
        raise RuntimeError("Expected memory key/value gradients before applying slot mask.")
    key_mask = torch.zeros_like(model.memory_keys.grad)
    value_mask = torch.zeros_like(model.memory_values.grad)
    key_mask[slot_index] = 1.0
    value_mask[slot_index] = 1.0
    model.memory_keys.grad.mul_(key_mask)
    model.memory_values.grad.mul_(value_mask)
    for name, parameter in model.named_parameters():
        if name not in {"memory_keys", "memory_values"} and parameter.grad is not None:
            parameter.grad.zero_()


def next_inactive_parent(model: SemanticMemoryModel) -> int | None:
    if model.max_parents <= 0:
        return None
    inactive = (~model.parent_active).nonzero(as_tuple=False).flatten()
    if inactive.numel() == 0:
        return None
    return int(inactive[0].item())


def parent_output(model: SemanticMemoryModel, parent_index: int, queries: torch.Tensor) -> torch.Tensor:
    if parent_index < 0 or parent_index >= model.max_parents:
        raise ValueError(f"parent_index={parent_index} outside [0, {model.max_parents}).")
    if queries.ndim != 2:
        raise ValueError(f"queries must be rank-2 [batch, d_model], got {tuple(queries.shape)}.")
    return queries * model.parent_scale[parent_index].unsqueeze(0) + model.parent_bias[parent_index].unsqueeze(0)


def reset_memory_slot(model: SemanticMemoryModel, slot_index: int) -> None:
    if slot_index < 0 or slot_index >= model.num_slots:
        raise ValueError(f"slot_index={slot_index} outside [0, {model.num_slots}).")
    with torch.no_grad():
        model.memory_keys[slot_index].zero_()
        model.memory_values[slot_index].zero_()


def update_slot_bookkeeping(
    fact_slots: dict[tuple[str, tuple[str, ...]], int],
    slot_owners: list[tuple[str, tuple[str, ...]] | None],
    slot_use: torch.Tensor,
    event: SemanticEvent,
    chosen: CandidateResult,
    commit_acc_threshold: float,
) -> None:
    if chosen.slot_index is None or not chosen.changed or not event.is_one_hop:
        return
    if chosen.action not in {"allocate", "rewrite", "update"}:
        return
    if chosen.new_acc < commit_acc_threshold:
        return
    slot_index = chosen.slot_index
    if slot_index < 0 or slot_index >= len(slot_owners):
        raise ValueError(f"slot_index={slot_index} outside slot owner table.")
    previous_owner = slot_owners[slot_index]
    if previous_owner is not None and previous_owner != event.key:
        fact_slots.pop(previous_owner, None)
    previous_slot = fact_slots.get(event.key)
    if previous_slot is not None and previous_slot != slot_index:
        slot_owners[previous_slot] = None
    slot_owners[slot_index] = event.key
    fact_slots[event.key] = slot_index
    slot_use[slot_index] += 1.0


def should_try_consolidation(
    event: SemanticEvent,
    belief_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    slot_owners: list[tuple[str, tuple[str, ...]] | None],
) -> bool:
    if not event.is_one_hop or not event.reliable or not event.commit_truth:
        return False
    if event.key in belief_facts:
        return False
    return all(owner is not None for owner in slot_owners)


def parent_training_batch(
    model: SemanticMemoryModel,
    world: World,
    group: list[SemanticEvent],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(group) < 2:
        raise ValueError("Consolidation group must contain at least two events.")
    if any(not event.is_one_hop for event in group):
        raise ValueError("Consolidation only supports one-hop facts.")
    queries = torch.cat([one_hop_query(model, world, event, device) for event in group], dim=0)
    target_ids = torch.tensor([world.token_to_id[event.target] for event in group], dtype=torch.long, device=device)
    targets = model.token_embedding(target_ids)
    return queries.detach(), targets.detach()


def event_target_code(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    device: torch.device,
) -> torch.Tensor:
    target_id = torch.tensor([world.token_to_id[event.target]], dtype=torch.long, device=device)
    return model.token_embedding(target_id)


def memory_step(
    model: SemanticMemoryModel,
    state: torch.Tensor,
    relation_ids: torch.Tensor,
    relation_position: int,
    disabled_slots: set[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.ndim != 2:
        raise ValueError(f"state must be rank-2 [batch, d_model], got {tuple(state.shape)}.")
    if relation_ids.ndim != 1:
        raise ValueError(f"relation_ids must be rank-1 [batch], got {tuple(relation_ids.shape)}.")
    if state.shape[0] != relation_ids.shape[0]:
        raise ValueError("state and relation_ids batch size mismatch.")
    if any(slot < 0 or slot >= model.num_slots for slot in disabled_slots):
        raise ValueError(f"disabled_slots outside valid slot range: {sorted(disabled_slots)}.")

    relation_pos = torch.full(
        (relation_ids.shape[0],),
        relation_position,
        dtype=torch.float32,
        device=relation_ids.device,
    )
    relation = (
        model.relation_embedding(relation_ids)
        + model.role_vector("relation", relation_ids.shape[0], relation_ids.device)
        + model.position_vector(relation_pos)
    )
    query = F.normalize(model.query_norm(state + relation), dim=-1)
    slot_scores = query @ model.normalized_keys().T / model.temperature
    if disabled_slots:
        mask = torch.zeros(model.num_slots, dtype=torch.bool, device=slot_scores.device)
        mask[list(sorted(disabled_slots))] = True
        slot_scores = slot_scores.masked_fill(mask.unsqueeze(0), torch.finfo(slot_scores.dtype).min)
    slot_values = model.memory_values.unsqueeze(0).expand(query.shape[0], -1, -1)

    if model.max_parents > 0 and bool(model.parent_active.any().item()):
        active_parent_mask = model.active_parent_mask()
        parent_keys = F.normalize(model.parent_keys[active_parent_mask], dim=-1)
        parent_key_scores = query @ parent_keys.T
        parent_values = (
            query.unsqueeze(1) * model.parent_scale[active_parent_mask].unsqueeze(0)
            + model.parent_bias[active_parent_mask].unsqueeze(0)
        )
        token_basis = F.normalize(model.token_embedding.weight, dim=-1)
        parent_value_confidence = torch.einsum(
            "bpd,vd->bpv",
            F.normalize(parent_values, dim=-1),
            token_basis,
        ).max(dim=-1).values
        parent_scores = (
            parent_key_scores + model.parent_confidence_weight * parent_value_confidence
        ) / model.temperature
        scores = torch.cat([slot_scores, parent_scores], dim=1)
        values = torch.cat([slot_values, parent_values], dim=1)
    else:
        scores = slot_scores
        values = slot_values

    attention = F.softmax(scores, dim=-1)
    next_state = model.state_norm(torch.bmm(attention.unsqueeze(1), values).squeeze(1))
    logits = next_state @ model.token_embedding.weight.T
    return logits, next_state, query


def parent_offset_loss(
    model: SemanticMemoryModel,
    parent_index: int,
    queries: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    outputs = parent_output(model, parent_index, queries)
    parent_offsets = outputs - queries
    target_offsets = targets - queries
    return (1.0 - F.cosine_similarity(parent_offsets, target_offsets, dim=-1)).mean()


def parent_first_hop_loss(
    model: SemanticMemoryModel,
    parent_index: int,
    queries: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    outputs = model.state_norm(parent_output(model, parent_index, queries))
    return F.mse_loss(outputs, targets)


def parent_anti_interference_loss(
    model: SemanticMemoryModel,
    world: World,
    group: list[SemanticEvent],
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    fact_slots: dict[tuple[str, tuple[str, ...]], int],
    parent_index: int,
    device: torch.device,
    margin: float,
) -> torch.Tensor:
    if margin <= 0.0:
        raise ValueError(f"margin must be positive, got {margin}.")
    group_keys = {event.key for event in group}
    losses: list[torch.Tensor] = []
    parent_key = F.normalize(model.parent_keys[parent_index], dim=0)
    token_basis = F.normalize(model.token_embedding.weight, dim=-1)
    keys = model.normalized_keys()
    for key, event in active_facts.items():
        if key in group_keys or key not in fact_slots or not event.is_one_hop:
            continue
        slot_index = fact_slots[key]
        if slot_index < 0 or slot_index >= model.num_slots:
            raise ValueError(f"slot_index={slot_index} outside [0, {model.num_slots}).")
        query = one_hop_query(model, world, event, device)
        parent_value = parent_output(model, parent_index, query)
        parent_confidence = (F.normalize(parent_value, dim=-1) @ token_basis.T).max(dim=1).values
        parent_score = query @ parent_key + model.parent_confidence_weight * parent_confidence
        correct_score = query @ keys[slot_index : slot_index + 1].T
        losses.append(F.relu(margin + parent_score.squeeze(0) - correct_score.squeeze(0)).mean())
    if not losses:
        return torch.zeros((), dtype=model.memory_keys.dtype, device=device)
    return torch.stack(losses).mean()


def parent_dependent_composition_loss(
    model: SemanticMemoryModel,
    world: World,
    group: list[SemanticEvent],
    slots_to_free: list[int],
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    parent_index: int,
    device: torch.device,
    lambda_closure: float,
) -> torch.Tensor:
    if lambda_closure < 0.0:
        raise ValueError(f"lambda_closure must be non-negative, got {lambda_closure}.")
    group_keys = {event.key for event in group}
    disabled_slots = set(slots_to_free)
    losses: list[torch.Tensor] = []
    for composition_event, first, _second in dependent_composition_events(active_facts, group):
        second_relation = composition_event.relations[1]
        second_relation_id = torch.tensor(
            [world.relation_to_id[second_relation]],
            dtype=torch.long,
            device=device,
        )
        target_id = torch.tensor([world.token_to_id[composition_event.target]], dtype=torch.long, device=device)
        if first.key in group_keys:
            first_query = one_hop_query(model, world, first, device)
            first_state = model.state_norm(parent_output(model, parent_index, first_query))
        else:
            first_subject_id = torch.tensor([world.token_to_id[first.subject]], dtype=torch.long, device=device)
            first_relation_id = torch.tensor(
                [[world.relation_to_id[first.relations[0]]]],
                dtype=torch.long,
                device=device,
            )
            _, first_diagnostics = model(first_subject_id, first_relation_id)
            first_state = first_diagnostics["state"]

        logits, second_state, _ = memory_step(
            model,
            first_state,
            second_relation_id,
            relation_position=2,
            disabled_slots=disabled_slots,
        )
        target_code = model.token_embedding(target_id)
        losses.append(F.cross_entropy(logits, target_id) + lambda_closure * F.mse_loss(second_state, target_code))
    if not losses:
        return torch.zeros((), dtype=model.memory_keys.dtype, device=device)
    return torch.stack(losses).mean()


@torch.no_grad()
def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def composition_event_dependencies(
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    composition_event: SemanticEvent,
) -> tuple[SemanticEvent, SemanticEvent] | None:
    if len(composition_event.relations) != 2:
        return None
    first_key = (composition_event.subject, (composition_event.relations[0],))
    first = active_facts.get(first_key)
    if first is None:
        return None
    second_key = (first.target, (composition_event.relations[1],))
    second = active_facts.get(second_key)
    if second is None:
        return None
    return first, second


def dependent_composition_events(
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    group: list[SemanticEvent],
) -> list[tuple[SemanticEvent, SemanticEvent, SemanticEvent]]:
    group_keys = {event.key for event in group}
    dependent: list[tuple[SemanticEvent, SemanticEvent, SemanticEvent]] = []
    for composition_event in active_composition_events(active_facts):
        dependencies = composition_event_dependencies(active_facts, composition_event)
        if dependencies is None:
            continue
        first, second = dependencies
        if first.key in group_keys or second.key in group_keys:
            dependent.append((composition_event, first, second))
    return dependent


@torch.no_grad()
def consolidation_attempt_diagnostics(
    before_model: SemanticMemoryModel,
    after_model: SemanticMemoryModel,
    world: World,
    group: list[SemanticEvent],
    slots_to_free: list[int],
    parent_index: int,
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    device: torch.device,
    accepted: bool,
    candidate_active_acc: float,
    candidate_composition_acc: float,
    candidate_group_acc: float,
    before_active_acc: float,
    before_composition_acc: float,
) -> dict[str, Any]:
    if len(group) < 2:
        raise ValueError("Consolidation diagnostics require at least two grouped events.")
    queries, targets = parent_training_batch(after_model, world, group, device)
    outputs = parent_output(after_model, parent_index, queries)
    parent_key = F.normalize(after_model.parent_keys[parent_index], dim=0)
    target_offsets = targets - queries
    parent_offsets = outputs - queries

    own_target_cosines = F.cosine_similarity(outputs, targets, dim=-1).detach().cpu().tolist()
    offset_cosines = F.cosine_similarity(parent_offsets, target_offsets, dim=-1).detach().cpu().tolist()
    offset_norm_ratios = (
        parent_offsets.norm(dim=-1) / (target_offsets.norm(dim=-1) + 1e-8)
    ).detach().cpu().tolist()
    key_cosines = (queries @ parent_key).detach().cpu().tolist()
    query_pair_cosine = None
    if len(group) == 2:
        query_pair_cosine = float(F.cosine_similarity(queries[0:1], queries[1:2], dim=-1).item())

    target_basis = F.normalize(targets, dim=-1)
    output_basis = F.normalize(outputs, dim=-1)
    output_to_target = output_basis @ target_basis.T
    own_vs_other_margins: list[float] = []
    own_slot_cosines: list[float] = []
    other_slot_cosines: list[float] = []
    for index, slot_index in enumerate(slots_to_free):
        own = float(output_to_target[index, index].item())
        others = [
            float(output_to_target[index, other_index].item())
            for other_index in range(len(group))
            if other_index != index
        ]
        if others:
            own_vs_other_margins.append(own - max(others))
        before_slot_value = before_model.memory_values[slot_index].unsqueeze(0)
        own_slot_cosines.append(
            float(F.cosine_similarity(outputs[index : index + 1], before_slot_value, dim=-1).item())
        )
        other_slot_values = [
            before_model.memory_values[other_slot].unsqueeze(0)
            for other_slot_index, other_slot in enumerate(slots_to_free)
            if other_slot_index != index
        ]
        if other_slot_values:
            other_cosines = [
                float(F.cosine_similarity(outputs[index : index + 1], other_slot_value, dim=-1).item())
                for other_slot_value in other_slot_values
            ]
            other_slot_cosines.append(max(other_cosines))

    before_direct_closures = [evaluate_closure(before_model, world, event, device) for event in group]
    after_direct_closures = [evaluate_closure(after_model, world, event, device) for event in group]
    dependent = dependent_composition_events(active_facts, group)
    before_comp_accs: list[float] = []
    after_comp_accs: list[float] = []
    before_comp_closures: list[float] = []
    after_comp_closures: list[float] = []
    before_first_hop_closures: list[float] = []
    after_first_hop_closures: list[float] = []
    for composition_event, first, _second in dependent:
        before_comp_accs.append(evaluate_events(before_model, world, [composition_event], device))
        after_comp_accs.append(evaluate_events(after_model, world, [composition_event], device))
        before_comp_closures.append(evaluate_closure(before_model, world, composition_event, device))
        after_comp_closures.append(evaluate_closure(after_model, world, composition_event, device))
        before_first_hop_closures.append(evaluate_closure(before_model, world, first, device))
        after_first_hop_closures.append(evaluate_closure(after_model, world, first, device))

    relation_types = [event.relations[0] for event in group]
    return {
        "accepted": accepted,
        "group_events": [event.name for event in group],
        "group_keys": [str(event.key) for event in group],
        "relation_types": relation_types,
        "same_relation": len(set(relation_types)) == 1,
        "slots_to_free": [int(slot) for slot in slots_to_free],
        "parent_index": int(parent_index),
        "candidate_active_acc": candidate_active_acc,
        "candidate_composition_acc": candidate_composition_acc,
        "candidate_group_acc": candidate_group_acc,
        "before_active_acc": before_active_acc,
        "before_composition_acc": before_composition_acc,
        "query_pair_cosine": query_pair_cosine,
        "parent_key_cosine_mean": safe_mean([float(value) for value in key_cosines]),
        "parent_key_cosine_min": None if not key_cosines else float(min(key_cosines)),
        "parent_key_cosine_max": None if not key_cosines else float(max(key_cosines)),
        "parent_output_target_cosine_mean": safe_mean([float(value) for value in own_target_cosines]),
        "parent_output_target_cosine_min": None if not own_target_cosines else float(min(own_target_cosines)),
        "parent_output_own_vs_other_target_margin_mean": safe_mean(own_vs_other_margins),
        "parent_output_own_slot_cosine_mean": safe_mean(own_slot_cosines),
        "parent_output_other_slot_cosine_mean": safe_mean(other_slot_cosines),
        "offset_cosine_mean": safe_mean([float(value) for value in offset_cosines]),
        "offset_cosine_min": None if not offset_cosines else float(min(offset_cosines)),
        "offset_norm_ratio_mean": safe_mean([float(value) for value in offset_norm_ratios]),
        "before_direct_closure_mean": safe_mean(before_direct_closures),
        "after_direct_closure_mean": safe_mean(after_direct_closures),
        "direct_closure_delta_mean": safe_mean(
            [
                after_value - before_value
                for before_value, after_value in zip(before_direct_closures, after_direct_closures)
            ]
        ),
        "dependent_composition_count": len(dependent),
        "before_dependent_composition_acc_mean": safe_mean(before_comp_accs),
        "after_dependent_composition_acc_mean": safe_mean(after_comp_accs),
        "before_dependent_composition_closure_mean": safe_mean(before_comp_closures),
        "after_dependent_composition_closure_mean": safe_mean(after_comp_closures),
        "dependent_composition_closure_delta_mean": safe_mean(
            [
                after_value - before_value
                for before_value, after_value in zip(before_comp_closures, after_comp_closures)
            ]
        ),
        "before_first_hop_closure_mean": safe_mean(before_first_hop_closures),
        "after_first_hop_closure_mean": safe_mean(after_first_hop_closures),
        "first_hop_closure_delta_mean": safe_mean(
            [
                after_value - before_value
                for before_value, after_value in zip(before_first_hop_closures, after_first_hop_closures)
            ]
        ),
    }


def none_or_leq(value: float | None, limit: float) -> bool:
    return value is None or value <= limit


def none_or_geq(value: float | None, limit: float) -> bool:
    return value is None or value >= limit


def consolidation_admission_safe(
    args: argparse.Namespace,
    group: list[SemanticEvent],
    active_acc: float,
    composition_acc: float,
    group_acc: float,
    before_active: float,
    before_composition: float,
    diagnostic: dict[str, Any] | None,
) -> bool:
    current_safe = (
        active_acc + 1e-9 >= before_active
        and composition_acc + 1e-9 >= before_composition
        and group_acc >= args.commit_acc_threshold
    )
    if args.consolidation_admission == "current":
        return current_safe

    relation_types = [event.relations[0] for event in group]
    same_relation = len(set(relation_types)) == 1
    if args.consolidation_admission == "same_relation":
        return current_safe and same_relation

    if args.consolidation_admission != "composition_preserving":
        raise ValueError(f"Unknown consolidation admission mode {args.consolidation_admission!r}.")
    if diagnostic is None:
        raise RuntimeError("composition_preserving admission requires consolidation diagnostics.")

    return (
        current_safe
        and none_or_geq(diagnostic["offset_cosine_mean"], args.consolidation_min_offset_cosine)
        and none_or_leq(
            diagnostic["direct_closure_delta_mean"],
            args.consolidation_max_direct_closure_delta,
        )
        and none_or_leq(
            diagnostic["first_hop_closure_delta_mean"],
            args.consolidation_max_first_hop_closure_delta,
        )
        and none_or_leq(
            diagnostic["dependent_composition_closure_delta_mean"],
            args.consolidation_max_dependent_composition_closure_delta,
        )
    )


def train_parent_candidate(
    model: SemanticMemoryModel,
    world: World,
    group: list[SemanticEvent],
    slots_to_free: list[int],
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    fact_slots: dict[tuple[str, tuple[str, ...]], int],
    parent_index: int,
    device: torch.device,
    args: argparse.Namespace,
) -> SemanticMemoryModel:
    if args.consolidation_epochs <= 0:
        raise ValueError(f"consolidation_epochs must be positive, got {args.consolidation_epochs}.")
    if args.consolidation_lr <= 0.0:
        raise ValueError(f"consolidation_lr must be positive, got {args.consolidation_lr}.")
    if args.consolidation_margin <= 0.0:
        raise ValueError(f"consolidation_margin must be positive, got {args.consolidation_margin}.")

    shadow = copy.deepcopy(model).to(device)
    queries, targets = parent_training_batch(shadow, world, group, device)
    with torch.no_grad():
        mean_query = F.normalize(queries.mean(dim=0), dim=0)
        shadow.parent_keys[parent_index].copy_(mean_query)
        shadow.parent_scale[parent_index].fill_(1.0)
        shadow.parent_bias[parent_index].zero_()
        shadow.parent_active[parent_index] = True

    optimizer = torch.optim.AdamW(
        [shadow.parent_keys, shadow.parent_scale, shadow.parent_bias],
        lr=args.consolidation_lr,
        weight_decay=0.0,
    )
    for _ in range(args.consolidation_epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = parent_output(shadow, parent_index, queries)
        reconstruction_loss = F.mse_loss(outputs, targets)
        cosine_loss = (1.0 - F.cosine_similarity(outputs, targets, dim=-1)).mean()
        parent_key = F.normalize(shadow.parent_keys[parent_index], dim=0)
        parent_key_scores = queries @ parent_key
        token_basis = F.normalize(shadow.token_embedding.weight, dim=-1)
        parent_value_confidence = (F.normalize(outputs, dim=-1) @ token_basis.T).max(dim=1).values
        parent_scores = parent_key_scores + shadow.parent_confidence_weight * parent_value_confidence
        slot_scores = queries @ shadow.normalized_keys().T
        wrong_scores = [slot_scores]
        active_parents = shadow.parent_active.clone()
        active_parents[parent_index] = False
        if bool(active_parents.any().item()):
            wrong_scores.append(queries @ F.normalize(shadow.parent_keys[active_parents], dim=-1).T)
        max_wrong = torch.cat(wrong_scores, dim=1).max(dim=1).values
        margin_loss = F.relu(args.consolidation_margin + max_wrong - parent_scores).mean()
        loss = reconstruction_loss + cosine_loss + margin_loss
        if args.parent_offset_weight > 0.0:
            loss = loss + args.parent_offset_weight * parent_offset_loss(
                shadow,
                parent_index,
                queries,
                targets,
            )
        if args.parent_first_hop_weight > 0.0:
            loss = loss + args.parent_first_hop_weight * parent_first_hop_loss(
                shadow,
                parent_index,
                queries,
                targets,
            )
        if args.parent_composition_weight > 0.0:
            loss = loss + args.parent_composition_weight * parent_dependent_composition_loss(
                shadow,
                world,
                group,
                slots_to_free,
                active_facts,
                parent_index,
                device,
                args.lambda_closure,
            )
        if args.parent_anti_interference_weight > 0.0:
            loss = loss + args.parent_anti_interference_weight * parent_anti_interference_loss(
                shadow,
                world,
                group,
                active_facts,
                fact_slots,
                parent_index,
                device,
                args.consolidation_margin,
            )
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite consolidation loss.")
        loss.backward()
        optimizer.step()

    for slot_index in slots_to_free:
        reset_memory_slot(shadow, slot_index)
    return shadow


def candidate_group_pairs(
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    fact_slots: dict[tuple[str, tuple[str, ...]], int],
) -> list[tuple[list[SemanticEvent], list[int]]]:
    slotted_facts = [
        (key, fact, fact_slots[key])
        for key, fact in active_facts.items()
        if key in fact_slots and fact.is_one_hop
    ]
    groups: list[tuple[list[SemanticEvent], list[int]]] = []
    for left_index in range(len(slotted_facts)):
        _, left_fact, left_slot = slotted_facts[left_index]
        for right_index in range(left_index + 1, len(slotted_facts)):
            _, right_fact, right_slot = slotted_facts[right_index]
            if left_slot == right_slot:
                continue
            groups.append(([left_fact, right_fact], [left_slot, right_slot]))
    return groups


def same_relation_group(group: list[SemanticEvent]) -> bool:
    if not group:
        raise ValueError("Cannot check relation homogeneity for an empty group.")
    relation_types = [event.relations[0] for event in group]
    return len(set(relation_types)) == 1


@torch.no_grad()
def group_geometry_score(
    model: SemanticMemoryModel,
    world: World,
    group: list[SemanticEvent],
    device: torch.device,
) -> tuple[float, float]:
    queries, targets = parent_training_batch(model, world, group, device)
    if queries.shape[0] < 2:
        raise ValueError("Group geometry score requires at least two events.")
    normalized_queries = F.normalize(queries, dim=-1)
    normalized_offsets = F.normalize(targets - queries, dim=-1)
    query_similarities: list[float] = []
    offset_similarities: list[float] = []
    for left_index in range(queries.shape[0]):
        for right_index in range(left_index + 1, queries.shape[0]):
            query_similarities.append(
                float(
                    F.cosine_similarity(
                        normalized_queries[left_index : left_index + 1],
                        normalized_queries[right_index : right_index + 1],
                        dim=-1,
                    ).item()
                )
            )
            offset_similarities.append(
                float(
                    F.cosine_similarity(
                        normalized_offsets[left_index : left_index + 1],
                        normalized_offsets[right_index : right_index + 1],
                        dim=-1,
                    ).item()
                )
            )
    if not query_similarities or not offset_similarities:
        raise RuntimeError("Group geometry scoring produced no pairwise similarities.")
    return (
        float(np.mean(np.asarray(offset_similarities, dtype=np.float64))),
        float(np.mean(np.asarray(query_similarities, dtype=np.float64))),
    )


def order_consolidation_groups(
    groups: list[tuple[list[SemanticEvent], list[int]]],
    model: SemanticMemoryModel,
    world: World,
    device: torch.device,
    order: str,
) -> list[tuple[list[SemanticEvent], list[int]]]:
    if order not in CONSOLIDATION_GROUP_ORDERS:
        raise ValueError(f"Unknown consolidation group order {order!r}.")
    if order == "current":
        return groups
    indexed_groups = list(enumerate(groups))
    if order == "same_relation_first":
        return [
            group
            for _index, group in sorted(
                indexed_groups,
                key=lambda item: (0 if same_relation_group(item[1][0]) else 1, item[0]),
            )
        ]
    if order == "geometry":
        scored: list[tuple[int, tuple[list[SemanticEvent], list[int]], bool, float, float]] = []
        for index, group_pair in indexed_groups:
            offset_similarity, query_similarity = group_geometry_score(model, world, group_pair[0], device)
            scored.append(
                (
                    index,
                    group_pair,
                    same_relation_group(group_pair[0]),
                    offset_similarity,
                    query_similarity,
                )
            )
        return [
            group_pair
            for index, group_pair, is_same_relation, offset_similarity, query_similarity in sorted(
                scored,
                key=lambda item: (
                    0 if item[2] else 1,
                    -item[3],
                    -item[4],
                    item[0],
                ),
            )
        ]
    raise ValueError(f"Unhandled consolidation group order {order!r}.")


def try_dynamic_consolidation(
    model: SemanticMemoryModel,
    world: World,
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    fact_slots: dict[tuple[str, tuple[str, ...]], int],
    slot_owners: list[tuple[str, tuple[str, ...]] | None],
    slot_use: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    stats: ConsolidationStats,
    record_diagnostics: bool,
) -> SemanticMemoryModel:
    if not args.enable_consolidation:
        return model
    parent_index = next_inactive_parent(model)
    if parent_index is None:
        stats.no_capacity += 1
        return model

    groups = candidate_group_pairs(active_facts, fact_slots)
    if not groups:
        stats.rejected += 1
        return model
    groups = order_consolidation_groups(groups, model, world, device, args.consolidation_group_order)
    groups = groups[: args.consolidation_max_candidates]

    before_active = evaluate_events(model, world, list(active_facts.values()), device)
    before_composition = evaluate_events(model, world, active_composition_events(active_facts), device)
    best_model: SemanticMemoryModel | None = None
    best_slots: list[int] = []
    best_diagnostic_index: int | None = None
    best_score = -float("inf")

    for group, slots_to_free in groups:
        stats.attempts += 1
        candidate = train_parent_candidate(
            model,
            world,
            group,
            slots_to_free,
            active_facts,
            fact_slots,
            parent_index,
            device,
            args,
        )
        active_acc = evaluate_events(candidate, world, list(active_facts.values()), device)
        composition_acc = evaluate_events(candidate, world, active_composition_events(active_facts), device)
        group_acc = evaluate_events(candidate, world, group, device)
        score = active_acc + composition_acc + group_acc + 0.01 * len(set(slots_to_free))
        needs_diagnostic = record_diagnostics or args.consolidation_admission == "composition_preserving"
        diagnostic: dict[str, Any] | None = None
        diagnostic_index: int | None = None
        if needs_diagnostic:
            diagnostic = consolidation_attempt_diagnostics(
                before_model=model,
                after_model=candidate,
                world=world,
                group=group,
                slots_to_free=slots_to_free,
                parent_index=parent_index,
                active_facts=active_facts,
                device=device,
                accepted=False,
                candidate_active_acc=active_acc,
                candidate_composition_acc=composition_acc,
                candidate_group_acc=group_acc,
                before_active_acc=before_active,
                before_composition_acc=before_composition,
            )
        safe = consolidation_admission_safe(
            args=args,
            group=group,
            active_acc=active_acc,
            composition_acc=composition_acc,
            group_acc=group_acc,
            before_active=before_active,
            before_composition=before_composition,
            diagnostic=diagnostic,
        )
        if record_diagnostics:
            if diagnostic is None:
                raise RuntimeError("Expected consolidation diagnostic when record_diagnostics is enabled.")
            diagnostic["safe"] = safe
            diagnostic["score"] = score
            diagnostic["admission"] = args.consolidation_admission
            diagnostic_index = len(stats.diagnostics)
            stats.diagnostics.append(diagnostic)
        if safe and score > best_score:
            best_score = score
            best_model = candidate
            best_slots = sorted(set(slots_to_free))
            best_diagnostic_index = diagnostic_index

    if best_model is None:
        stats.rejected += 1
        return model

    if best_diagnostic_index is not None:
        stats.diagnostics[best_diagnostic_index]["accepted"] = True
    for slot_index in best_slots:
        owner = slot_owners[slot_index]
        if owner is not None:
            fact_slots.pop(owner, None)
        slot_owners[slot_index] = None
        slot_use[slot_index] = 0.0
    stats.commits += 1
    stats.freed_slots += len(best_slots)
    return best_model


def train_candidate(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    device: torch.device,
    args: argparse.Namespace,
    mode: str,
    slot_index: int | None,
) -> SemanticMemoryModel:
    if args.update_epochs <= 0:
        raise ValueError(f"update_epochs must be positive, got {args.update_epochs}.")
    shadow = copy.deepcopy(model).to(device)
    optimizer = torch.optim.AdamW(shadow.parameters(), lr=args.lr)

    projected_facts = dict(active_facts)
    if event.commit_truth and event.is_one_hop:
        projected_facts[event.key] = event
    composition_events = active_composition_events(projected_facts)

    for _ in range(args.update_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(shadow, world, event, device, args.lambda_closure)
        if mode == "slot":
            if slot_index is None:
                raise ValueError("slot_index is required for slot candidate training.")
            direct_loss, _ = slot_write_losses(
                shadow,
                world,
                event,
                device,
                slot_index=slot_index,
                margin=args.attention_margin,
            )
            loss = loss + args.direct_write_weight * direct_loss
            if args.composition_write_weight > 0.0 and composition_events:
                comp_losses = [
                    compute_loss(shadow, world, comp_event, device, args.lambda_closure)
                    for comp_event in composition_events
                ]
                loss = loss + args.composition_write_weight * torch.stack(comp_losses).mean()
        elif mode != "full":
            raise ValueError(f"Unknown candidate training mode {mode!r}.")
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite candidate loss for event {event.name}.")
        loss.backward()
        if mode == "slot":
            apply_slot_gradient_mask(shadow, slot_index)
        optimizer.step()
    return shadow


def projected_active_events(
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    event: SemanticEvent,
) -> list[SemanticEvent]:
    projected = dict(active_facts)
    if event.commit_truth and event.is_one_hop:
        projected[event.key] = event
    return list(projected.values())


def protected_events(
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    event: SemanticEvent,
) -> list[SemanticEvent]:
    return [fact for key, fact in active_facts.items() if key != event.key]


def active_composition_events(
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
) -> list[SemanticEvent]:
    events: list[SemanticEvent] = []
    facts = list(active_facts.values())
    for first in facts:
        if len(first.relations) != 1:
            continue
        for second in facts:
            if len(second.relations) != 1:
                continue
            if second.subject != first.target:
                continue
            events.append(
                SemanticEvent(
                    name=f"active_compose_{first.subject}_{first.relations[0]}_{second.relations[0]}",
                    subject=first.subject,
                    relations=(first.relations[0], second.relations[0]),
                    target=second.target,
                    reliable=True,
                    expected_action="compose",
                    commit_truth=False,
                    evidence=1,
                )
            )
    return events


def score_candidate(
    candidate: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    device: torch.device,
    action: str,
    closure_penalty: float,
    full_update_penalty: float,
) -> tuple[float, float, float, float, float]:
    new_acc = evaluate_events(candidate, world, [event], device)
    projected = projected_active_events(active_facts, event)
    active_acc = evaluate_events(candidate, world, projected, device)
    protected_acc = evaluate_events(candidate, world, protected_events(active_facts, event), device)
    closure = evaluate_closure(candidate, world, event, device)

    if event.reliable:
        new_weight = 2.0
    else:
        new_weight = -1.0

    score = (
        new_weight * new_acc
        + 2.0 * active_acc
        + 1.0 * protected_acc
        - closure_penalty * closure
    )
    if action == "update":
        score -= full_update_penalty
    return score, new_acc, active_acc, protected_acc, closure


def build_candidates(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    slot_use: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> list[CandidateResult]:
    no_update_actions = ("compose", "discard") if len(event.relations) > 1 else ("reuse", "discard")
    candidates: list[tuple[str, str, SemanticMemoryModel, int | None, bool]] = [
        (f"no_update_{action}", action, copy.deepcopy(model).to(device), None, False)
        for action in no_update_actions
    ]

    if event.is_one_hop:
        allocate_slot = least_used_slot(slot_use)
        allocate_model = train_candidate(
            model,
            world,
            event,
            active_facts,
            device,
            args,
            mode="slot",
            slot_index=allocate_slot,
        )
        candidates.append(("slot_allocate", "allocate", allocate_model, allocate_slot, True))

        rewrite_slot = top_attention_slot(model, world, event, device)
        rewrite_model = train_candidate(
            model,
            world,
            event,
            active_facts,
            device,
            args,
            mode="slot",
            slot_index=rewrite_slot,
        )
        candidates.append(("slot_rewrite", "rewrite", rewrite_model, rewrite_slot, True))

        full_model = train_candidate(
            model,
            world,
            event,
            active_facts,
            device,
            args,
            mode="full",
            slot_index=None,
        )
        candidates.append(("full_update", "update", full_model, None, True))
    elif event.reliable:
        full_model = train_candidate(
            model,
            world,
            event,
            active_facts,
            device,
            args,
            mode="full",
            slot_index=None,
        )
        candidates.append(("full_update", "update", full_model, None, True))

    results: list[CandidateResult] = []
    for label, action, candidate_model, slot_index, changed in candidates:
        score, new_acc, active_acc, protected_acc, closure = score_candidate(
            candidate_model,
            world,
            event,
            active_facts,
            device,
            action=action,
            closure_penalty=args.closure_penalty,
            full_update_penalty=args.full_update_penalty,
        )
        if not np.isfinite(score):
            raise FloatingPointError(f"Non-finite candidate score for {event.name}/{label}.")
        write_diag = slot_write_diagnostics(candidate_model, world, event, device, slot_index)
        results.append(
            CandidateResult(
                label=label,
                action=action,
                model=candidate_model,
                new_acc=new_acc,
                active_acc=active_acc,
                protected_acc=protected_acc,
                closure=closure,
                slot_index=slot_index,
                slot_usage=-1.0 if slot_index is None else float(slot_use[slot_index].item()),
                key_cosine=write_diag["key_cosine"],
                value_cosine=write_diag["value_cosine"],
                target_attention=write_diag["target_attention"],
                attention_margin=write_diag["attention_margin"],
                changed=changed,
                score=score,
            )
        )
    return results


def scalar_sincos_list(value: float, dim: int = 8) -> list[float]:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")
    half_dim = max(1, dim // 2)
    frequencies = np.exp(-np.log(10000.0) * np.arange(half_dim, dtype=np.float64) / max(1.0, half_dim - 1.0))
    angles = float(value) * frequencies
    encoded = np.concatenate([np.sin(angles), np.cos(angles)], axis=0)
    if encoded.shape[0] < dim:
        encoded = np.pad(encoded, (0, dim - encoded.shape[0]))
    return [float(x) for x in encoded[:dim]]


def append_scalar_feature(context: list[float], value: float, enabled: bool, dim: int = 8) -> None:
    if enabled:
        context.extend(scalar_sincos_list(value, dim=dim))
    else:
        context.extend([0.0] * dim)


def event_numeric_context(world: World, event: SemanticEvent, args: argparse.Namespace) -> list[float]:
    if event.source not in SOURCE_TO_INDEX:
        raise ValueError(f"Unknown event source {event.source!r}.")
    relation_ids = [world.relation_to_id[relation] for relation in event.relations]
    if not relation_ids:
        raise ValueError(f"Event {event.name} has no relation ids.")
    context: list[float] = []
    append_scalar_feature(
        context,
        float(world.token_to_id[event.subject]),
        enabled=not args.disable_policy_identity_features,
    )
    append_scalar_feature(
        context,
        float(world.token_to_id[event.target]),
        enabled=not args.disable_policy_identity_features,
    )
    append_scalar_feature(
        context,
        float(np.mean(relation_ids)),
        enabled=not args.disable_policy_identity_features,
    )
    append_scalar_feature(
        context,
        float(len(relation_ids)),
        enabled=not args.disable_policy_identity_features,
    )
    append_scalar_feature(
        context,
        float(event.subject_pos),
        enabled=not args.disable_policy_position_features,
    )
    append_scalar_feature(
        context,
        float(event.relation_pos),
        enabled=not args.disable_policy_position_features,
    )
    append_scalar_feature(
        context,
        float(event.object_pos),
        enabled=not args.disable_policy_position_features,
    )
    append_scalar_feature(context, float(event.timestamp), enabled=not args.disable_policy_time_features)
    append_scalar_feature(
        context,
        float(SOURCE_TO_INDEX[event.source]),
        enabled=not args.disable_policy_source_features,
    )
    append_scalar_feature(
        context,
        float(min(MAX_EVIDENCE_INDEX, event.evidence)),
        enabled=not args.disable_policy_evidence_features,
    )
    return context


def candidate_feature_vector(
    candidate: CandidateResult,
    event: SemanticEvent,
    world: World,
    num_slots: int,
    args: argparse.Namespace,
) -> list[float]:
    if candidate.action not in ACTION_TO_INDEX:
        raise ValueError(f"Unknown candidate action {candidate.action!r}.")
    slot_value = -1.0 if candidate.slot_index is None else candidate.slot_index / max(1.0, float(num_slots - 1))
    slot_usage = min(candidate.slot_usage, 5.0) / 5.0
    key_cosine = -1.0 if candidate.key_cosine is None else candidate.key_cosine
    value_cosine = -1.0 if candidate.value_cosine is None else candidate.value_cosine
    target_attention = -1.0 if candidate.target_attention is None else candidate.target_attention
    attention_margin = -1.0 if candidate.attention_margin is None else candidate.attention_margin
    event_context = event_numeric_context(world, event, args)
    action_features = [0.0] * len(ACTIONS)
    action_features[ACTION_TO_INDEX[candidate.action]] = 1.0
    return action_features + [
        1.0 if candidate.changed else 0.0,
        slot_value,
        slot_usage,
        float(candidate.new_acc),
        float(candidate.active_acc),
        float(candidate.protected_acc),
        float(candidate.closure),
        float(key_cosine),
        float(value_cosine),
        float(target_attention),
        float(attention_margin),
    ] + [float(value) for value in event_context]


def candidate_feature_tensor(
    candidates: list[CandidateResult],
    event: SemanticEvent,
    world: World,
    num_slots: int,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    if not candidates:
        raise ValueError("Cannot featurize an empty candidate list.")
    rows = [candidate_feature_vector(candidate, event, world, num_slots, args) for candidate in candidates]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def teacher_candidate_index(event: SemanticEvent, candidates: list[CandidateResult]) -> int:
    matches = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate.action == event.expected_action
    ]
    if not matches:
        raise RuntimeError(f"No training candidate matches expected action {event.expected_action!r}.")
    return max(matches, key=lambda item: item[1].score)[0]


def best_action_candidate(candidates: list[CandidateResult], action: str) -> CandidateResult:
    action_candidates = [candidate for candidate in candidates if candidate.action == action]
    if not action_candidates:
        raise RuntimeError(f"No candidate exists for action {action!r}.")
    return max(action_candidates, key=lambda candidate: candidate.score)


def choose_candidate(
    event: SemanticEvent,
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    candidates: list[CandidateResult],
    commit_acc_threshold: float,
) -> CandidateResult:
    if not candidates:
        raise ValueError(f"No candidates available for event {event.name}.")

    if not event.reliable:
        return best_action_candidate(candidates, "discard")

    if len(event.relations) > 1:
        return best_action_candidate(candidates, "compose")

    current_fact = active_facts.get(event.key)
    no_update = best_action_candidate(candidates, "reuse")
    if current_fact is not None and current_fact.target == event.target and no_update.new_acc >= commit_acc_threshold:
        return no_update

    if current_fact is not None and current_fact.target != event.target:
        if event.evidence >= 2:
            rewrite = best_action_candidate(candidates, "rewrite")
            if rewrite.new_acc >= commit_acc_threshold:
                return rewrite
        return best_action_candidate(candidates, "discard")

    allocate = best_action_candidate(candidates, "allocate")
    if allocate.new_acc >= commit_acc_threshold:
        return allocate

    return max(candidates, key=lambda candidate: candidate.score)


def commit_belief_from_choice(
    belief_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    event: SemanticEvent,
    chosen: CandidateResult,
    commit_acc_threshold: float,
) -> None:
    if not event.is_one_hop:
        return
    if chosen.action not in {"allocate", "rewrite", "update"}:
        return
    if chosen.new_acc < commit_acc_threshold:
        return
    belief_facts[event.key] = event


def commit_truth(
    truth_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent],
    event: SemanticEvent,
) -> None:
    if event.commit_truth and event.is_one_hop:
        truth_facts[event.key] = event


def collect_policy_examples(
    args: argparse.Namespace,
    device: torch.device,
    base_state: dict[str, torch.Tensor] | None,
) -> list[tuple[torch.Tensor, int]]:
    examples: list[tuple[torch.Tensor, int]] = []
    world = build_world()
    seed_iter = range(args.policy_train_seed_offset, args.policy_train_seed_offset + args.policy_train_seed_count)
    for seed in tqdm(seed_iter, desc="policy examples", dynamic_ncols=True, disable=args.no_progress):
        set_seed(seed)
        model = make_model(world, args, device, base_state)
        slot_use = torch.zeros(args.num_slots, dtype=torch.float32, device=device)
        slot_owners: list[tuple[str, tuple[str, ...]] | None] = [None] * args.num_slots
        fact_slots: dict[tuple[str, tuple[str, ...]], int] = {}
        belief_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
        consolidation_stats = ConsolidationStats()

        for event in build_stream(seed):
            if should_try_consolidation(event, belief_facts, slot_owners):
                model = try_dynamic_consolidation(
                    model,
                    world,
                    belief_facts,
                    fact_slots,
                    slot_owners,
                    slot_use,
                    device,
                    args,
                    consolidation_stats,
                    record_diagnostics=False,
                )
            candidates = build_candidates(model, world, event, belief_facts, slot_use, device, args)
            features = candidate_feature_tensor(candidates, event, world, args.num_slots, device, args)
            target_index = teacher_candidate_index(event, candidates)
            examples.append((features.detach().cpu(), target_index))
            teacher = candidates[target_index]
            model = teacher.model
            commit_belief_from_choice(belief_facts, event, teacher, args.commit_acc_threshold)
            update_slot_bookkeeping(fact_slots, slot_owners, slot_use, event, teacher, args.commit_acc_threshold)

    if not examples:
        raise RuntimeError("Policy training produced no examples.")
    return examples


def train_write_policy(
    args: argparse.Namespace,
    device: torch.device,
    base_state: dict[str, torch.Tensor] | None,
) -> CandidateWritePolicy:
    examples = collect_policy_examples(args, device, base_state)
    input_dim = examples[0][0].shape[-1]
    if any(example[0].shape[-1] != input_dim for example in examples):
        raise RuntimeError("Policy examples have inconsistent feature dimensions.")

    policy = CandidateWritePolicy(input_dim=input_dim, hidden_dim=args.policy_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.policy_lr)
    order = np.arange(len(examples))
    for _ in tqdm(range(args.policy_epochs), desc="policy train", dynamic_ncols=True, disable=args.no_progress):
        np.random.shuffle(order)
        for index in order:
            features_cpu, target_index = examples[int(index)]
            features = features_cpu.to(device)
            target = torch.tensor([target_index], dtype=torch.long, device=device)
            logits = policy(features).unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite write-policy loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return policy


@torch.no_grad()
def choose_with_policy(
    policy: CandidateWritePolicy,
    candidates: list[CandidateResult],
    event: SemanticEvent,
    model: SemanticMemoryModel,
    world: World,
    num_slots: int,
    device: torch.device,
    args: argparse.Namespace,
) -> CandidateResult:
    features = candidate_feature_tensor(candidates, event, world, num_slots, device, args)
    logits = policy(features)
    chosen_index = int(logits.argmax().item())
    return candidates[chosen_index]


def run_geometry_reasoner(
    seed: int,
    args: argparse.Namespace,
    base_state: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(args.device)
    world = build_world()
    model = make_model(world, args, device, base_state)
    slot_use = torch.zeros(args.num_slots, dtype=torch.float32, device=device)
    slot_owners: list[tuple[str, tuple[str, ...]] | None] = [None] * args.num_slots
    fact_slots: dict[tuple[str, tuple[str, ...]], int] = {}
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
    consolidation_stats = ConsolidationStats()
    rows: list[dict[str, Any]] = []

    for event in build_stream(seed):
        if should_try_consolidation(event, active_facts, slot_owners):
            model = try_dynamic_consolidation(
                model,
                world,
                active_facts,
                fact_slots,
                slot_owners,
                slot_use,
                device,
                args,
                consolidation_stats,
                record_diagnostics=args.record_consolidation_diagnostics,
            )
        candidates = build_candidates(model, world, event, active_facts, slot_use, device, args)
        chosen = choose_candidate(event, active_facts, candidates, args.commit_acc_threshold)
        action_ok = chosen.action == event.expected_action

        model = chosen.model
        if event.commit_truth and event.is_one_hop and chosen.new_acc >= args.commit_acc_threshold:
            active_facts[event.key] = event
        update_slot_bookkeeping(fact_slots, slot_owners, slot_use, event, chosen, args.commit_acc_threshold)

        rows.append(
            {
                "event": event.name,
                "expected_action": event.expected_action,
                "chosen_action": chosen.action,
                "candidate": chosen.label,
                "action_ok": action_ok,
                "new_acc": chosen.new_acc,
                "active_acc": chosen.active_acc,
                "protected_acc": chosen.protected_acc,
                "closure": chosen.closure,
                "slot": chosen.slot_index,
                "key_cosine": chosen.key_cosine,
                "value_cosine": chosen.value_cosine,
                "target_attention": chosen.target_attention,
                "attention_margin": chosen.attention_margin,
                "parent_count": model.active_parent_count(),
                "score": chosen.score,
            }
        )

    final_active = evaluate_events(model, world, list(active_facts.values()), device)
    composition_events = active_composition_events(active_facts)
    final_composition = evaluate_events(model, world, composition_events, device)
    action_acc = sum(1 for row in rows if row["action_ok"]) / float(len(rows))
    return {
        "seed": seed,
        "method": "geometry_reasoner",
        "action_acc": action_acc,
        "final_active_acc": final_active,
        "final_composition_acc": final_composition,
        "final_slot_use": [float(value) for value in slot_use.detach().cpu().tolist()],
        "final_parent_count": model.active_parent_count(),
        "consolidation": consolidation_stats.to_dict(),
        "consolidation_diagnostics": consolidation_stats.diagnostics,
        "rows": rows,
    }


def run_learned_policy_reasoner(
    seed: int,
    args: argparse.Namespace,
    policy: CandidateWritePolicy,
    base_state: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(args.device)
    world = build_world()
    model = make_model(world, args, device, base_state)
    slot_use = torch.zeros(args.num_slots, dtype=torch.float32, device=device)
    slot_owners: list[tuple[str, tuple[str, ...]] | None] = [None] * args.num_slots
    fact_slots: dict[tuple[str, tuple[str, ...]], int] = {}
    belief_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
    truth_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
    consolidation_stats = ConsolidationStats()
    rows: list[dict[str, Any]] = []

    for event in build_stream(seed):
        if should_try_consolidation(event, belief_facts, slot_owners):
            model = try_dynamic_consolidation(
                model,
                world,
                belief_facts,
                fact_slots,
                slot_owners,
                slot_use,
                device,
                args,
                consolidation_stats,
                record_diagnostics=args.record_consolidation_diagnostics,
            )
        candidates = build_candidates(model, world, event, belief_facts, slot_use, device, args)
        chosen = choose_with_policy(policy, candidates, event, model, world, args.num_slots, device, args)
        action_ok = chosen.action == event.expected_action

        model = chosen.model
        commit_belief_from_choice(belief_facts, event, chosen, args.commit_acc_threshold)
        commit_truth(truth_facts, event)
        update_slot_bookkeeping(fact_slots, slot_owners, slot_use, event, chosen, args.commit_acc_threshold)

        rows.append(
            {
                "event": event.name,
                "expected_action": event.expected_action,
                "chosen_action": chosen.action,
                "candidate": chosen.label,
                "action_ok": action_ok,
                "new_acc": chosen.new_acc,
                "truth_acc": evaluate_events(model, world, list(truth_facts.values()), device),
                "belief_acc": evaluate_events(model, world, list(belief_facts.values()), device),
                "active_acc": chosen.active_acc,
                "protected_acc": chosen.protected_acc,
                "closure": chosen.closure,
                "slot": chosen.slot_index,
                "key_cosine": chosen.key_cosine,
                "value_cosine": chosen.value_cosine,
                "target_attention": chosen.target_attention,
                "attention_margin": chosen.attention_margin,
                "parent_count": model.active_parent_count(),
                "score": chosen.score,
            }
        )

    final_active = evaluate_events(model, world, list(truth_facts.values()), device)
    composition_events = active_composition_events(truth_facts)
    final_composition = evaluate_events(model, world, composition_events, device)
    action_acc = sum(1 for row in rows if row["action_ok"]) / float(len(rows))
    return {
        "seed": seed,
        "method": "learned_policy",
        "action_acc": action_acc,
        "final_active_acc": final_active,
        "final_composition_acc": final_composition,
        "final_slot_use": [float(value) for value in slot_use.detach().cpu().tolist()],
        "final_parent_count": model.active_parent_count(),
        "consolidation": consolidation_stats.to_dict(),
        "consolidation_diagnostics": consolidation_stats.diagnostics,
        "rows": rows,
    }


def train_blind_event(
    model: SemanticMemoryModel,
    world: World,
    event: SemanticEvent,
    device: torch.device,
    args: argparse.Namespace,
) -> SemanticMemoryModel:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for _ in range(args.update_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, world, event, device, args.lambda_closure)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite blind AdamW loss for event {event.name}.")
        loss.backward()
        optimizer.step()
    return model


def run_blind_adamw(
    seed: int,
    args: argparse.Namespace,
    base_state: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(args.device)
    world = build_world()
    model = make_model(world, args, device, base_state)
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
    rows: list[dict[str, Any]] = []

    for event in build_stream(seed):
        # Blind AdamW treats every labeled observation as a training target,
        # including unreliable conflict/noise.
        model = train_blind_event(model, world, event, device, args)
        if event.commit_truth and event.is_one_hop:
            active_facts[event.key] = event
        active_acc = evaluate_events(model, world, list(active_facts.values()), device)
        new_acc = evaluate_events(model, world, [event], device)
        rows.append(
            {
                "event": event.name,
                "new_acc": new_acc,
                "active_acc": active_acc,
                "expected_action": event.expected_action,
            }
        )

    final_active = evaluate_events(model, world, list(active_facts.values()), device)
    composition_events = active_composition_events(active_facts)
    final_composition = evaluate_events(model, world, composition_events, device)
    return {
        "seed": seed,
        "method": "blind_adamw",
        "final_active_acc": final_active,
        "final_composition_acc": final_composition,
        "rows": rows,
    }


def mean_std(values: list[float]) -> str:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    array = np.asarray(values, dtype=np.float64)
    return f"{array.mean():.4f} +/- {array.std():.4f}"


def optional_float_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def consolidation_diagnostic_summary(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        diagnostics = report.get("consolidation_diagnostics", [])
        if diagnostics:
            by_method.setdefault(str(report["method"]), []).extend(diagnostics)

    summary: dict[str, dict[str, Any]] = {}
    for method, diagnostics in by_method.items():
        same_relation = [row for row in diagnostics if row["same_relation"]]
        mixed_relation = [row for row in diagnostics if not row["same_relation"]]
        accepted = [row for row in diagnostics if row["accepted"]]
        summary[method] = {
            "attempts": len(diagnostics),
            "accepted": len(accepted),
            "same_relation_attempts": len(same_relation),
            "same_relation_accepted": sum(1 for row in same_relation if row["accepted"]),
            "mixed_relation_attempts": len(mixed_relation),
            "mixed_relation_accepted": sum(1 for row in mixed_relation if row["accepted"]),
            "offset_cosine_mean": optional_float_mean(diagnostics, "offset_cosine_mean"),
            "accepted_offset_cosine_mean": optional_float_mean(accepted, "offset_cosine_mean"),
            "direct_closure_delta_mean": optional_float_mean(diagnostics, "direct_closure_delta_mean"),
            "dependent_composition_closure_delta_mean": optional_float_mean(
                diagnostics,
                "dependent_composition_closure_delta_mean",
            ),
            "first_hop_closure_delta_mean": optional_float_mean(diagnostics, "first_hop_closure_delta_mean"),
            "after_dependent_composition_acc_mean": optional_float_mean(
                diagnostics,
                "after_dependent_composition_acc_mean",
            ),
        }
    return summary


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("No reports to summarize.")
    by_method: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        by_method.setdefault(report["method"], []).append(report)

    summary: dict[str, Any] = {}
    for method, method_reports in by_method.items():
        summary[method] = {
            "final_active_acc": mean_std([float(report["final_active_acc"]) for report in method_reports]),
            "final_composition_acc": mean_std(
                [float(report["final_composition_acc"]) for report in method_reports]
            ),
        }
        if method in {"geometry_reasoner", "learned_policy"}:
            summary[method]["action_acc"] = mean_std([float(report["action_acc"]) for report in method_reports])
            summary[method]["final_parent_count"] = mean_std(
                [float(report.get("final_parent_count", 0.0)) for report in method_reports]
            )
            summary[method]["consolidation_commits"] = mean_std(
                [float(report.get("consolidation", {}).get("commits", 0.0)) for report in method_reports]
            )
            summary[method]["consolidation_freed_slots"] = mean_std(
                [float(report.get("consolidation", {}).get("freed_slots", 0.0)) for report in method_reports]
            )

    geometry_rows: dict[str, list[dict[str, Any]]] = {}
    preferred_event_method = "learned_policy" if "learned_policy" in by_method else "geometry_reasoner"
    for report in reports:
        if report["method"] != preferred_event_method:
            continue
        for row in report["rows"]:
            geometry_rows.setdefault(str(row["event"]), []).append(row)

    event_summary = {}
    for event_name, rows in geometry_rows.items():
        actions: dict[str, int] = {}
        for row in rows:
            action = str(row["chosen_action"])
            actions[action] = actions.get(action, 0) + 1
        key_cosines = [float(row["key_cosine"]) for row in rows if row.get("key_cosine") is not None]
        value_cosines = [float(row["value_cosine"]) for row in rows if row.get("value_cosine") is not None]
        target_attentions = [
            float(row["target_attention"]) for row in rows if row.get("target_attention") is not None
        ]
        attention_margins = [
            float(row["attention_margin"]) for row in rows if row.get("attention_margin") is not None
        ]
        event_summary[event_name] = {
            "expected_action": rows[0]["expected_action"],
            "chosen_actions": actions,
            "action_acc": float(sum(1 for row in rows if row["action_ok"]) / len(rows)),
            "new_acc": float(np.mean([float(row["new_acc"]) for row in rows])),
            "active_acc": float(np.mean([float(row["active_acc"]) for row in rows])),
            "closure": float(np.mean([float(row["closure"]) for row in rows])),
            "key_cosine": None if not key_cosines else float(np.mean(key_cosines)),
            "value_cosine": None if not value_cosines else float(np.mean(value_cosines)),
            "target_attention": None if not target_attentions else float(np.mean(target_attentions)),
            "attention_margin": None if not attention_margins else float(np.mean(attention_margins)),
        }

    return {
        "methods": summary,
        "geometry_events": event_summary,
        "consolidation_diagnostics": consolidation_diagnostic_summary(reports),
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\nSEMANTIC GEOMETRY WRITE-CONTROL SUMMARY")
    print("=" * 124)
    print(f"model=single_internal_memory_model seeds={report['seed_count']}")
    print("-" * 124)
    print(
        f"{'method':<28} {'active_acc':<24} {'composition_acc':<24} "
        f"{'action_acc':<24} {'parents':<18} {'freed_slots':<18}"
    )
    print("-" * 124)
    for method, metrics in summary["methods"].items():
        action_acc = metrics.get("action_acc", "n/a")
        parent_count = metrics.get("final_parent_count", "n/a")
        freed_slots = metrics.get("consolidation_freed_slots", "n/a")
        print(
            f"{method:<28} "
            f"{metrics['final_active_acc']:<24} "
            f"{metrics['final_composition_acc']:<24} "
            f"{action_acc:<24} "
            f"{parent_count:<18} "
            f"{freed_slots:<18}"
        )
    print("-" * 124)
    print(f"{'event':<30} {'expected':<12} {'chosen':<30} {'act_acc':<8} {'new':<8} {'active':<8} {'closure':<8}")
    print("-" * 124)
    for event_name, metrics in summary["geometry_events"].items():
        print(
            f"{event_name:<30} "
            f"{metrics['expected_action']:<12} "
            f"{str(metrics['chosen_actions']):<30} "
            f"{metrics['action_acc']:<8.3f} "
            f"{metrics['new_acc']:<8.3f} "
            f"{metrics['active_acc']:<8.3f} "
            f"{metrics['closure']:<8.4f}"
        )
    if summary["consolidation_diagnostics"]:
        print("-" * 124)
        print("CONSOLIDATION DIAGNOSTICS")
        print("-" * 124)
        print(
            f"{'method':<28} {'attempts':<9} {'accepted':<9} "
            f"{'same_acc':<10} {'mixed_acc':<10} {'offset_cos':<12} "
            f"{'dep_comp_acc':<13}"
        )
        for method, metrics in summary["consolidation_diagnostics"].items():
            same_attempts = int(metrics["same_relation_attempts"])
            mixed_attempts = int(metrics["mixed_relation_attempts"])
            same_accept_rate = (
                "n/a" if same_attempts == 0 else f"{metrics['same_relation_accepted'] / same_attempts:.3f}"
            )
            mixed_accept_rate = (
                "n/a" if mixed_attempts == 0 else f"{metrics['mixed_relation_accepted'] / mixed_attempts:.3f}"
            )
            offset = metrics["accepted_offset_cosine_mean"]
            dep_acc = metrics["after_dependent_composition_acc_mean"]
            print(
                f"{method:<28} "
                f"{metrics['attempts']:<9} "
                f"{metrics['accepted']:<9} "
                f"{same_accept_rate:<10} "
                f"{mixed_accept_rate:<10} "
                f"{'n/a' if offset is None else f'{offset:.3f}':<12} "
                f"{'n/a' if dep_acc is None else f'{dep_acc:.3f}':<13}"
            )
    print("=" * 124)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--update-epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--lambda-closure", type=float, default=1.0)
    parser.add_argument("--geometry-warmup", action="store_true")
    parser.add_argument("--geometry-seed", type=int, default=12345)
    parser.add_argument("--geometry-train-seed-count", type=int, default=40)
    parser.add_argument("--geometry-train-seed-offset", type=int, default=2000)
    parser.add_argument("--geometry-warmup-epochs", type=int, default=300)
    parser.add_argument("--geometry-warmup-lr", type=float, default=3e-3)
    parser.add_argument("--geometry-token-ce-weight", type=float, default=0.25)
    parser.add_argument("--geometry-separation-weight", type=float, default=0.05)
    parser.add_argument("--geometry-code-norm-weight", type=float, default=0.01)
    parser.add_argument("--geometry-max-code-cosine", type=float, default=0.3)
    parser.add_argument("--direct-write-weight", type=float, default=1.0)
    parser.add_argument("--composition-write-weight", type=float, default=0.25)
    parser.add_argument("--attention-margin", type=float, default=0.25)
    parser.add_argument("--enable-consolidation", action="store_true")
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--parent-confidence-weight", type=float, default=0.5)
    parser.add_argument("--consolidation-epochs", type=int, default=80)
    parser.add_argument("--consolidation-lr", type=float, default=1e-2)
    parser.add_argument("--consolidation-margin", type=float, default=0.25)
    parser.add_argument("--consolidation-max-candidates", type=int, default=6)
    parser.add_argument("--consolidation-group-order", choices=CONSOLIDATION_GROUP_ORDERS, default="current")
    parser.add_argument("--consolidation-admission", choices=CONSOLIDATION_ADMISSIONS, default="current")
    parser.add_argument("--consolidation-min-offset-cosine", type=float, default=0.85)
    parser.add_argument("--consolidation-max-direct-closure-delta", type=float, default=0.0)
    parser.add_argument("--consolidation-max-first-hop-closure-delta", type=float, default=0.0)
    parser.add_argument("--consolidation-max-dependent-composition-closure-delta", type=float, default=0.0)
    parser.add_argument("--parent-offset-weight", type=float, default=0.0)
    parser.add_argument("--parent-first-hop-weight", type=float, default=0.0)
    parser.add_argument("--parent-composition-weight", type=float, default=0.0)
    parser.add_argument("--parent-anti-interference-weight", type=float, default=0.0)
    parser.add_argument("--record-consolidation-diagnostics", action="store_true")
    parser.add_argument("--closure-penalty", type=float, default=0.1)
    parser.add_argument("--full-update-penalty", type=float, default=0.15)
    parser.add_argument("--commit-acc-threshold", type=float, default=0.999)
    parser.add_argument("--policy-train-seed-count", type=int, default=20)
    parser.add_argument("--policy-train-seed-offset", type=int, default=1000)
    parser.add_argument("--policy-epochs", type=int, default=50)
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--include-rule-oracle", action="store_true")
    parser.add_argument("--disable-role-embeddings", action="store_true")
    parser.add_argument("--disable-position-encoding", action="store_true")
    parser.add_argument("--disable-policy-identity-features", action="store_true")
    parser.add_argument("--disable-policy-position-features", action="store_true")
    parser.add_argument("--disable-policy-time-features", action="store_true")
    parser.add_argument("--disable-policy-source-features", action="store_true")
    parser.add_argument("--disable-policy-evidence-features", action="store_true")
    parser.add_argument("--feature-ablation", action="store_true")
    parser.add_argument("--feature-ablation-list", type=str, default=",".join(FEATURE_ABLATIONS))
    parser.add_argument("--feature-ablation-include-blind", action="store_true")
    parser.add_argument("--feature-ablation-include-diagnostics", action="store_true")
    parser.add_argument("--skip-blind-adamw", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/semantic-geometry-write-control.json"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def disabled_feature_names(args: argparse.Namespace) -> list[str]:
    features: list[str] = []
    if args.disable_role_embeddings:
        features.append("role")
    if args.disable_position_encoding:
        features.append("position")
    if args.disable_policy_identity_features:
        features.append("policy_identity")
    if args.disable_policy_position_features:
        features.append("policy_position")
    if args.disable_policy_time_features:
        features.append("policy_time")
    if args.disable_policy_source_features:
        features.append("policy_source")
    if args.disable_policy_evidence_features:
        features.append("policy_evidence")
    return features


def parse_feature_ablation_list(raw: str) -> list[str]:
    features = [feature.strip() for feature in raw.split(",") if feature.strip()]
    if not features:
        raise ValueError("--feature-ablation-list must contain at least one feature name.")
    unknown = [feature for feature in features if feature not in FEATURE_ABLATIONS]
    if unknown:
        raise ValueError(
            f"Unknown feature ablation(s): {unknown}. Available: {list(FEATURE_ABLATIONS)}."
        )
    if len(set(features)) != len(features):
        raise ValueError(f"--feature-ablation-list contains duplicates: {features}.")
    return features


def args_with_feature_disabled(args: argparse.Namespace, feature: str) -> argparse.Namespace:
    if feature not in FEATURE_ABLATIONS:
        raise ValueError(f"Unknown feature ablation {feature!r}.")
    variant = copy.copy(args)
    if feature == "role":
        variant.disable_role_embeddings = True
    elif feature == "position":
        variant.disable_position_encoding = True
    elif feature == "policy_identity":
        variant.disable_policy_identity_features = True
    elif feature == "policy_position":
        variant.disable_policy_position_features = True
    elif feature == "policy_time":
        variant.disable_policy_time_features = True
    elif feature == "policy_source":
        variant.disable_policy_source_features = True
    elif feature == "policy_evidence":
        variant.disable_policy_evidence_features = True
    else:
        raise ValueError(f"Unhandled feature ablation {feature!r}.")
    return variant


def validate_args(args: argparse.Namespace) -> None:
    if args.seed_count <= 0:
        raise ValueError(f"--seed-count must be positive, got {args.seed_count}.")
    if args.update_epochs <= 0:
        raise ValueError(f"--update-epochs must be positive, got {args.update_epochs}.")
    if args.d_model <= 0:
        raise ValueError(f"--d-model must be positive, got {args.d_model}.")
    if args.num_slots <= 0:
        raise ValueError(f"--num-slots must be positive, got {args.num_slots}.")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if args.lambda_closure < 0.0:
        raise ValueError(f"--lambda-closure must be non-negative, got {args.lambda_closure}.")
    if args.geometry_seed < 0:
        raise ValueError(f"--geometry-seed must be non-negative, got {args.geometry_seed}.")
    if args.geometry_train_seed_count <= 0:
        raise ValueError(
            f"--geometry-train-seed-count must be positive, got {args.geometry_train_seed_count}."
        )
    if args.geometry_train_seed_offset < 0:
        raise ValueError(
            f"--geometry-train-seed-offset must be non-negative, got {args.geometry_train_seed_offset}."
        )
    if args.geometry_warmup_epochs <= 0:
        raise ValueError(f"--geometry-warmup-epochs must be positive, got {args.geometry_warmup_epochs}.")
    if args.geometry_warmup_lr <= 0.0:
        raise ValueError(f"--geometry-warmup-lr must be positive, got {args.geometry_warmup_lr}.")
    if args.geometry_token_ce_weight < 0.0:
        raise ValueError(
            f"--geometry-token-ce-weight must be non-negative, got {args.geometry_token_ce_weight}."
        )
    if args.geometry_separation_weight < 0.0:
        raise ValueError(
            f"--geometry-separation-weight must be non-negative, got {args.geometry_separation_weight}."
        )
    if args.geometry_code_norm_weight < 0.0:
        raise ValueError(
            f"--geometry-code-norm-weight must be non-negative, got {args.geometry_code_norm_weight}."
        )
    if args.geometry_max_code_cosine < -1.0 or args.geometry_max_code_cosine > 1.0:
        raise ValueError(
            f"--geometry-max-code-cosine must be in [-1, 1], got {args.geometry_max_code_cosine}."
        )
    if args.direct_write_weight < 0.0:
        raise ValueError(f"--direct-write-weight must be non-negative, got {args.direct_write_weight}.")
    if args.composition_write_weight < 0.0:
        raise ValueError(f"--composition-write-weight must be non-negative, got {args.composition_write_weight}.")
    if args.attention_margin <= 0.0:
        raise ValueError(f"--attention-margin must be positive, got {args.attention_margin}.")
    if args.max_parents < 0:
        raise ValueError(f"--max-parents must be non-negative, got {args.max_parents}.")
    if args.enable_consolidation and args.max_parents <= 0:
        raise ValueError("--enable-consolidation requires --max-parents > 0.")
    if args.parent_confidence_weight < 0.0:
        raise ValueError(
            f"--parent-confidence-weight must be non-negative, got {args.parent_confidence_weight}."
        )
    if args.consolidation_epochs <= 0:
        raise ValueError(f"--consolidation-epochs must be positive, got {args.consolidation_epochs}.")
    if args.consolidation_lr <= 0.0:
        raise ValueError(f"--consolidation-lr must be positive, got {args.consolidation_lr}.")
    if args.consolidation_margin <= 0.0:
        raise ValueError(f"--consolidation-margin must be positive, got {args.consolidation_margin}.")
    if args.consolidation_max_candidates <= 0:
        raise ValueError(
            f"--consolidation-max-candidates must be positive, got {args.consolidation_max_candidates}."
        )
    if args.consolidation_min_offset_cosine < -1.0 or args.consolidation_min_offset_cosine > 1.0:
        raise ValueError(
            f"--consolidation-min-offset-cosine must be in [-1, 1], got {args.consolidation_min_offset_cosine}."
        )
    if args.parent_offset_weight < 0.0:
        raise ValueError(f"--parent-offset-weight must be non-negative, got {args.parent_offset_weight}.")
    if args.parent_first_hop_weight < 0.0:
        raise ValueError(f"--parent-first-hop-weight must be non-negative, got {args.parent_first_hop_weight}.")
    if args.parent_composition_weight < 0.0:
        raise ValueError(f"--parent-composition-weight must be non-negative, got {args.parent_composition_weight}.")
    if args.parent_anti_interference_weight < 0.0:
        raise ValueError(
            f"--parent-anti-interference-weight must be non-negative, got {args.parent_anti_interference_weight}."
        )
    if args.policy_train_seed_count <= 0:
        raise ValueError(f"--policy-train-seed-count must be positive, got {args.policy_train_seed_count}.")
    if args.policy_train_seed_offset < 0:
        raise ValueError(f"--policy-train-seed-offset must be non-negative, got {args.policy_train_seed_offset}.")
    if args.policy_epochs <= 0:
        raise ValueError(f"--policy-epochs must be positive, got {args.policy_epochs}.")
    if args.policy_lr <= 0.0:
        raise ValueError(f"--policy-lr must be positive, got {args.policy_lr}.")
    if args.policy_hidden_dim <= 0:
        raise ValueError(f"--policy-hidden-dim must be positive, got {args.policy_hidden_dim}.")
    parse_feature_ablation_list(args.feature_ablation_list)
    if args.feature_ablation and disabled_feature_names(args):
        raise ValueError("--feature-ablation must be run without explicit --disable-* feature flags.")


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed_count": args.seed_count,
        "d_model": args.d_model,
        "num_slots": args.num_slots,
        "temperature": args.temperature,
        "update_epochs": args.update_epochs,
        "lr": args.lr,
        "lambda_closure": args.lambda_closure,
        "geometry_warmup": args.geometry_warmup,
        "geometry_seed": args.geometry_seed,
        "geometry_train_seed_count": args.geometry_train_seed_count,
        "geometry_train_seed_offset": args.geometry_train_seed_offset,
        "geometry_warmup_epochs": args.geometry_warmup_epochs,
        "geometry_warmup_lr": args.geometry_warmup_lr,
        "geometry_token_ce_weight": args.geometry_token_ce_weight,
        "geometry_separation_weight": args.geometry_separation_weight,
        "geometry_code_norm_weight": args.geometry_code_norm_weight,
        "geometry_max_code_cosine": args.geometry_max_code_cosine,
        "direct_write_weight": args.direct_write_weight,
        "composition_write_weight": args.composition_write_weight,
        "attention_margin": args.attention_margin,
        "enable_consolidation": args.enable_consolidation,
        "max_parents": args.max_parents,
        "parent_confidence_weight": args.parent_confidence_weight,
        "consolidation_epochs": args.consolidation_epochs,
        "consolidation_lr": args.consolidation_lr,
        "consolidation_margin": args.consolidation_margin,
        "consolidation_max_candidates": args.consolidation_max_candidates,
        "consolidation_group_order": args.consolidation_group_order,
        "consolidation_admission": args.consolidation_admission,
        "consolidation_min_offset_cosine": args.consolidation_min_offset_cosine,
        "consolidation_max_direct_closure_delta": args.consolidation_max_direct_closure_delta,
        "consolidation_max_first_hop_closure_delta": args.consolidation_max_first_hop_closure_delta,
        "consolidation_max_dependent_composition_closure_delta": (
            args.consolidation_max_dependent_composition_closure_delta
        ),
        "parent_offset_weight": args.parent_offset_weight,
        "parent_first_hop_weight": args.parent_first_hop_weight,
        "parent_composition_weight": args.parent_composition_weight,
        "parent_anti_interference_weight": args.parent_anti_interference_weight,
        "record_consolidation_diagnostics": args.record_consolidation_diagnostics,
        "closure_penalty": args.closure_penalty,
        "full_update_penalty": args.full_update_penalty,
        "commit_acc_threshold": args.commit_acc_threshold,
        "policy_train_seed_count": args.policy_train_seed_count,
        "policy_train_seed_offset": args.policy_train_seed_offset,
        "policy_epochs": args.policy_epochs,
        "policy_lr": args.policy_lr,
        "policy_hidden_dim": args.policy_hidden_dim,
        "include_rule_oracle": args.include_rule_oracle,
        "disable_role_embeddings": args.disable_role_embeddings,
        "disable_position_encoding": args.disable_position_encoding,
        "disable_policy_identity_features": args.disable_policy_identity_features,
        "disable_policy_position_features": args.disable_policy_position_features,
        "disable_policy_time_features": args.disable_policy_time_features,
        "disable_policy_source_features": args.disable_policy_source_features,
        "disable_policy_evidence_features": args.disable_policy_evidence_features,
        "disabled_features": disabled_feature_names(args),
        "feature_ablation": args.feature_ablation,
        "feature_ablation_list": parse_feature_ablation_list(args.feature_ablation_list),
        "feature_ablation_include_blind": args.feature_ablation_include_blind,
        "feature_ablation_include_diagnostics": args.feature_ablation_include_diagnostics,
        "skip_blind_adamw": args.skip_blind_adamw,
        "device": args.device,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    base_state, geometry_warmup_metrics = train_latent_geometry_base(args, device)
    policy = train_write_policy(args, device, base_state)

    reports: list[dict[str, Any]] = []
    seed_iter = range(args.seed_count)
    for seed in tqdm(seed_iter, desc="evaluation seeds", dynamic_ncols=True, disable=args.no_progress):
        reports.append(run_learned_policy_reasoner(seed, args, policy, base_state))
        if args.include_rule_oracle:
            reports.append(run_geometry_reasoner(seed, args, base_state))
        if not args.skip_blind_adamw:
            reports.append(run_blind_adamw(seed, args, base_state))

    return {
        "config": experiment_config(args),
        "geometry_warmup": geometry_warmup_metrics,
        "seed_count": args.seed_count,
        "reports": reports,
        "summary": summarize(reports),
    }


def print_feature_ablation_summary(output: dict[str, Any]) -> None:
    print("\nSEMANTIC GEOMETRY FEATURE ABLATION SUMMARY")
    print("=" * 124)
    print(
        f"{'variant':<24} {'disabled':<26} {'active_acc':<24} "
        f"{'composition_acc':<24} {'action_acc':<24}"
    )
    print("-" * 124)
    for variant in output["variants"]:
        methods = variant["summary"]["methods"]
        if "learned_policy" not in methods:
            raise RuntimeError(f"Variant {variant['name']} has no learned_policy summary.")
        metrics = methods["learned_policy"]
        print(
            f"{variant['name']:<24} "
            f"{','.join(variant['disabled_features']) or 'none':<26} "
            f"{metrics['final_active_acc']:<24} "
            f"{metrics['final_composition_acc']:<24} "
            f"{metrics['action_acc']:<24}"
        )
    print("=" * 124)


def run_feature_ablation(args: argparse.Namespace) -> dict[str, Any]:
    features = parse_feature_ablation_list(args.feature_ablation_list)
    variants: list[dict[str, Any]] = []
    variant_specs = [("baseline", args)] + [
        (f"no_{feature}", args_with_feature_disabled(args, feature)) for feature in features
    ]
    for name, variant_args in tqdm(
        variant_specs,
        desc="feature ablations",
        dynamic_ncols=True,
        disable=args.no_progress,
    ):
        current_args = copy.copy(variant_args)
        current_args.feature_ablation = False
        current_args.skip_blind_adamw = not args.feature_ablation_include_blind
        if not args.feature_ablation_include_diagnostics:
            current_args.record_consolidation_diagnostics = False
        result = run_experiment(current_args)
        variants.append(
            {
                "name": name,
                "disabled_features": disabled_feature_names(current_args),
                "config": result["config"],
                "geometry_warmup": result["geometry_warmup"],
                "summary": result["summary"],
                "reports": result["reports"],
            }
        )
    return {
        "feature_ablation": 1,
        "config": experiment_config(args),
        "variants": variants,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    output = run_feature_ablation(args) if args.feature_ablation else run_experiment(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.feature_ablation:
        print_feature_ablation_summary(output)
    else:
        print_summary(output)
    print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
