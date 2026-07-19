"""Test semantic context, consequence reasoning, and bounded fast-weight modes.

This experiment isolates the missing semantic-plasticity mechanism before it is
integrated into the 1M weight-native continual learner.  A frozen foundation
model knows a base relational world.  During each continual episode, sources
report observations about that world.  Some sources are reliable and some are
not; their reliability is randomized every episode and is never supplied to the
learner.  Reliable sources introduce current-time corrections.  The model must:

* infer source reliability from recurrent evidence;
* bind a correction to the matching entity/relation/current-time context;
* propagate the correction to learned derived relations;
* preserve historical and unrelated queries;
* use a fixed-size bank of context-conditioned fast-weight modes.

The plasticity mechanism receives no event-kind, truth, preserve/drop, or
correct/incorrect labels.  It is meta-trained only through future query loss.
Hidden truth and source reliability are used after an episode for evaluation.

This is not the complete Invariant-Tangent architecture.  The frozen foundation
makes representation geometry constant so this experiment can answer one
question cleanly: can a neural, context-addressed, consequence-aware mechanism
perform semantic replacement without a hardcoded role controller?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device


@dataclass(frozen=True)
class ReportEvent:
    entity: int
    observed_value: int
    source: int
    source_is_reliable: bool
    phase: str


@dataclass(frozen=True)
class Episode:
    events: tuple[ReportEvent, ...]
    corrected_entities: tuple[int, ...]
    corrected_values: tuple[int, ...]
    source_reliability: tuple[bool, ...]


@dataclass
class FastModeState:
    keys: torch.Tensor
    fast: torch.Tensor
    mass: torch.Tensor
    source_hidden: torch.Tensor
    global_hidden: torch.Tensor
    source_agreement: torch.Tensor
    source_count: torch.Tensor


@dataclass(frozen=True)
class ExperimentThresholds:
    foundation_accuracy: float
    direct_accuracy: float
    ripple_accuracy: float
    paraphrase_accuracy: float
    historical_accuracy: float
    locality_accuracy: float
    consequence_accuracy: float
    correction_margin: float


class SemanticBackbone(nn.Module):
    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        num_contexts: int,
        num_variants: int,
        num_values: int,
        d_model: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_contexts = num_contexts
        self.num_variants = num_variants
        self.num_values = num_values
        self.d_model = d_model
        self.entity = nn.Embedding(num_entities, d_model)
        self.relation = nn.Embedding(num_relations, d_model)
        self.context = nn.Embedding(num_contexts, d_model)
        self.variant = nn.Embedding(num_variants, d_model)
        self.encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, num_values)

    def forward(
        self,
        entities: torch.Tensor,
        relations: torch.Tensor,
        contexts: torch.Tensor,
        variants: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shapes = {tuple(value.shape) for value in (entities, relations, contexts, variants)}
        if len(shapes) != 1:
            raise ValueError(f"Semantic query tensors must share a shape, got {shapes}.")
        if entities.ndim != 1 or entities.numel() <= 0:
            raise ValueError("Semantic query tensors must be non-empty vectors.")
        hidden = (
            self.entity(entities)
            + self.relation(relations)
            + self.context(contexts)
            + self.variant(variants)
        )
        hidden = self.final_norm(hidden + self.encoder(hidden))
        return hidden, self.output(hidden)


class SemanticPlasticity(nn.Module):
    def __init__(
        self,
        *,
        num_slots: int,
        num_sources: int,
        num_relations: int,
        num_values: int,
        d_model: int,
        source_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_sources = num_sources
        self.num_relations = num_relations
        self.num_values = num_values
        self.d_model = d_model
        self.source_hidden_dim = source_hidden_dim

        self.initial_keys = nn.Parameter(torch.empty(num_slots, d_model))
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.source_embedding = nn.Embedding(num_sources, source_hidden_dim)
        self.value_embedding = nn.Embedding(num_values, source_hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, source_hidden_dim)

        event_input_dim = d_model + num_values + source_hidden_dim * 2 + 6
        self.event_encoder = nn.Sequential(
            nn.LayerNorm(event_input_dim),
            nn.Linear(event_input_dim, source_hidden_dim),
            nn.GELU(),
        )
        self.global_recurrence = nn.GRUCell(source_hidden_dim, source_hidden_dim)
        self.source_recurrence = nn.GRUCell(source_hidden_dim * 2, source_hidden_dim)
        decision_dim = source_hidden_dim * 3
        self.write_head = nn.Sequential(
            nn.LayerNorm(decision_dim),
            nn.Linear(decision_dim, source_hidden_dim),
            nn.GELU(),
            nn.Linear(source_hidden_dim, 1),
        )
        consequence_input_dim = source_hidden_dim * 5
        self.consequence_head = nn.Sequential(
            nn.LayerNorm(consequence_input_dim),
            nn.Linear(consequence_input_dim, source_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(source_hidden_dim * 2, num_values),
        )
        self.consequence_confidence = nn.Sequential(
            nn.LayerNorm(consequence_input_dim),
            nn.Linear(consequence_input_dim, 1),
        )

        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.mass_attraction = nn.Parameter(torch.tensor(0.0))
        self.fast_scale_log = nn.Parameter(torch.tensor(0.54132485))
        self.write_rate_log = nn.Parameter(torch.tensor(0.54132485))
        self.key_rate_log = nn.Parameter(torch.tensor(-2.0))
        self.source_evidence_decay_logit = nn.Parameter(torch.tensor(2.944439))
        nn.init.normal_(self.initial_keys, mean=0.0, std=1.0 / math.sqrt(float(d_model)))

    def initial_state(self, *, device: torch.device, dtype: torch.dtype) -> FastModeState:
        return FastModeState(
            keys=F.normalize(self.initial_keys.to(device=device, dtype=dtype), dim=1),
            fast=torch.zeros(
                self.num_slots,
                self.num_values,
                self.d_model,
                device=device,
                dtype=dtype,
            ),
            mass=torch.zeros(self.num_slots, device=device, dtype=dtype),
            source_hidden=torch.zeros(
                self.num_sources,
                self.source_hidden_dim,
                device=device,
                dtype=dtype,
            ),
            global_hidden=torch.zeros(
                self.source_hidden_dim,
                device=device,
                dtype=dtype,
            ),
            source_agreement=torch.zeros(
                self.num_sources,
                device=device,
                dtype=dtype,
            ),
            source_count=torch.zeros(
                self.num_sources,
                device=device,
                dtype=dtype,
            ),
        )

    def route(self, hidden: torch.Tensor, state: FastModeState) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[1] != self.d_model:
            raise ValueError(
                f"Routing hidden states must be [batch, {self.d_model}], got {tuple(hidden.shape)}."
            )
        eps = torch.finfo(hidden.dtype).eps
        queries = F.normalize(self.query_projection(hidden), dim=1, eps=eps)
        keys = F.normalize(state.keys, dim=1, eps=eps)
        temperature = F.softplus(self.log_temperature) + eps
        mass_scale = state.mass.mean().clamp_min(eps)
        normalized_mass = state.mass / mass_scale
        scores = queries @ keys.transpose(0, 1) / temperature
        scores = scores + self.mass_attraction * torch.log1p(normalized_mass).unsqueeze(0)
        return torch.softmax(scores, dim=1)

    def read(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        state: FastModeState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self.route(hidden, state)
        mode_logits = torch.einsum("kvd,bd->bkv", state.fast, hidden)
        fast_logits = torch.einsum("bk,bkv->bv", attention, mode_logits)
        scale = F.softplus(self.fast_scale_log)
        return base_logits + scale * fast_logits, attention

    def evidence_step(
        self,
        *,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        observed_value: int,
        source: int,
        state: FastModeState,
    ) -> tuple[
        FastModeState,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if hidden.shape != (1, self.d_model) or logits.shape != (1, self.num_values):
            raise ValueError("Evidence step expects one semantic query and one logit vector.")
        if not 0 <= observed_value < self.num_values:
            raise ValueError(f"Observed value {observed_value} is outside the value vocabulary.")
        if not 0 <= source < self.num_sources:
            raise ValueError(f"Source {source} is outside the source vocabulary.")

        probabilities = torch.softmax(logits, dim=1)
        eps = torch.finfo(probabilities.dtype).eps
        observed_probability = probabilities[0, observed_value]
        entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
        sorted_probabilities = probabilities.sort(dim=1, descending=True).values
        margin = sorted_probabilities[0, 0] - sorted_probabilities[0, 1]
        disagreement = 1.0 - observed_probability
        scalar_evidence = torch.stack(
            [observed_probability, entropy, margin, disagreement]
        )
        source_index = torch.tensor(source, device=hidden.device, dtype=torch.long)
        value_index = torch.tensor(observed_value, device=hidden.device, dtype=torch.long)
        source_embedding = self.source_embedding(source_index)
        value_embedding = self.value_embedding(value_index)
        prior_count = state.source_count[source]
        prior_agreement = torch.where(
            prior_count > 0.0,
            state.source_agreement[source],
            state.source_agreement.new_tensor(0.5),
        )
        prior_confidence = prior_count / (prior_count + 1.0)
        source_evidence = torch.stack(
            [prior_agreement, torch.log1p(prior_count)]
        )
        event_input = torch.cat(
            [
                hidden.squeeze(0),
                probabilities.squeeze(0),
                source_embedding,
                value_embedding,
                scalar_evidence,
                source_evidence,
            ],
            dim=0,
        )
        encoded = self.event_encoder(event_input)
        next_global = self.global_recurrence(encoded, state.global_hidden)
        previous_source = state.source_hidden[source]
        next_source = self.source_recurrence(
            torch.cat([encoded, next_global], dim=0),
            previous_source,
        )
        decision_context = torch.cat([encoded, next_source, next_global], dim=0)
        write_gate = torch.sigmoid(self.write_head(decision_context)).squeeze(0)

        source_mask = F.one_hot(
            source_index,
            num_classes=self.num_sources,
        ).to(dtype=state.source_hidden.dtype).unsqueeze(1)
        next_source_hidden = (
            state.source_hidden * (1.0 - source_mask)
            + next_source.unsqueeze(0) * source_mask
        )
        source_mask_flat = source_mask.squeeze(1)
        evidence_decay = torch.sigmoid(self.source_evidence_decay_logit)
        updated_agreement = torch.where(
            prior_count > 0.0,
            evidence_decay * prior_agreement
            + (1.0 - evidence_decay) * observed_probability,
            observed_probability,
        )
        next_source_agreement = (
            state.source_agreement * (1.0 - source_mask_flat)
            + updated_agreement * source_mask_flat
        )
        next_source_count = state.source_count + source_mask_flat
        next_state = FastModeState(
            keys=state.keys,
            fast=state.fast,
            mass=state.mass,
            source_hidden=next_source_hidden,
            global_hidden=next_global,
            source_agreement=next_source_agreement,
            source_count=next_source_count,
        )
        return (
            next_state,
            write_gate,
            next_source,
            value_embedding,
            prior_agreement,
            prior_confidence,
        )

    def predict_consequences(
        self,
        *,
        event_hidden: torch.Tensor,
        source_hidden: torch.Tensor,
        value_embedding: torch.Tensor,
        global_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation_ids = torch.arange(
            self.num_relations,
            device=event_hidden.device,
            dtype=torch.long,
        )
        relation_embeddings = self.relation_embedding(relation_ids)
        rows = torch.cat(
            [
                event_hidden.expand(self.num_relations, -1),
                source_hidden.unsqueeze(0).expand(self.num_relations, -1),
                value_embedding.unsqueeze(0).expand(self.num_relations, -1),
                global_hidden.unsqueeze(0).expand(self.num_relations, -1),
                relation_embeddings,
            ],
            dim=1,
        )
        logits = self.consequence_head(rows)
        confidence = torch.sigmoid(self.consequence_confidence(rows)).squeeze(1)
        return torch.softmax(logits, dim=1), confidence

    def write(
        self,
        *,
        query_hidden: torch.Tensor,
        current_logits: torch.Tensor,
        target_probabilities: torch.Tensor,
        relation_confidence: torch.Tensor,
        write_gate: torch.Tensor,
        covariance_inverse: torch.Tensor,
        state: FastModeState,
        fast_capacity: float,
    ) -> FastModeState:
        if query_hidden.shape != (self.num_relations, self.d_model):
            raise ValueError("Fast write requires one current-context query per relation.")
        if target_probabilities.shape != (self.num_relations, self.num_values):
            raise ValueError("Consequence targets have an invalid shape.")
        if current_logits.shape != target_probabilities.shape:
            raise ValueError("Current logits and consequence targets must share a shape.")
        if covariance_inverse.shape != (self.d_model, self.d_model):
            raise ValueError("Covariance inverse has an invalid shape.")
        if fast_capacity <= 0.0 or not math.isfinite(fast_capacity):
            raise ValueError("Fast capacity must be positive and finite.")

        eps = torch.finfo(query_hidden.dtype).eps
        attention = self.route(query_hidden, state)
        routing_energy = attention.square().sum(dim=1, keepdim=True)
        if torch.any(routing_energy <= eps):
            raise RuntimeError("Context routing has zero squared mass.")
        minimum_norm_coefficients = attention / routing_energy
        current_probabilities = torch.softmax(current_logits, dim=1)
        errors = target_probabilities - current_probabilities
        directions = query_hidden @ covariance_inverse.transpose(0, 1)
        denominators = (directions * query_hidden).sum(dim=1, keepdim=True)
        if torch.any(denominators <= eps):
            raise RuntimeError("A semantic key has non-positive covariance-normalized energy.")
        directions = directions / denominators
        strengths = write_gate * relation_confidence
        mode_updates = torch.einsum(
            "rk,rv,rd,r->kvd",
            minimum_norm_coefficients,
            errors,
            directions,
            strengths,
        )
        write_rate = F.softplus(self.write_rate_log)
        next_fast = state.fast + write_rate * mode_updates

        projected_queries = F.normalize(
            self.query_projection(query_hidden),
            dim=1,
            eps=eps,
        )
        key_weights = attention * strengths.unsqueeze(1)
        key_targets = key_weights.transpose(0, 1) @ projected_queries
        key_denominator = key_weights.sum(dim=0, keepdim=False).unsqueeze(1)
        key_targets = key_targets / key_denominator.clamp_min(eps)
        key_rate = torch.sigmoid(self.key_rate_log)
        key_fraction = key_rate * key_denominator / (1.0 + state.mass.unsqueeze(1))
        blended_keys = state.keys + key_fraction * key_targets
        next_keys = F.normalize(blended_keys, dim=1, eps=eps)
        next_mass = state.mass + key_weights.sum(dim=0)

        fast_norm = torch.linalg.vector_norm(next_fast)
        capacity_tensor = next_fast.new_tensor(fast_capacity)
        capacity_scale = torch.minimum(
            next_fast.new_ones(()),
            capacity_tensor / fast_norm.clamp_min(eps),
        )
        next_fast = next_fast * capacity_scale
        final_norm = torch.linalg.vector_norm(next_fast)
        if not torch.isfinite(final_norm) or final_norm > capacity_tensor + 10.0 * eps:
            raise RuntimeError(
                f"Fast-mode capacity invariant failed: norm={float(final_norm.detach().cpu()):.8f}."
            )
        return FastModeState(
            keys=next_keys,
            fast=next_fast,
            mass=next_mass,
            source_hidden=state.source_hidden,
            global_hidden=state.global_hidden,
            source_agreement=state.source_agreement,
            source_count=state.source_count,
        )


class SemanticContextSystem(nn.Module):
    def __init__(self, backbone: SemanticBackbone, plasticity: SemanticPlasticity) -> None:
        super().__init__()
        self.backbone = backbone
        self.plasticity = plasticity

    def predict(
        self,
        *,
        entities: torch.Tensor,
        relations: torch.Tensor,
        contexts: torch.Tensor,
        variants: torch.Tensor,
        state: FastModeState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, base_logits = self.backbone(entities, relations, contexts, variants)
        logits, attention = self.plasticity.read(hidden, base_logits, state)
        return hidden, logits, attention


def validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and in [0, 1].")


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "num_entities",
        "meta_train_entities",
        "num_values",
        "num_relations",
        "num_variants",
        "num_sources",
        "reliable_sources",
        "d_model",
        "hidden_dim",
        "source_hidden_dim",
        "num_slots",
        "foundation_epochs",
        "consequence_pretrain_epochs",
        "meta_episodes",
        "test_episodes",
        "calibration_events_per_source",
        "correction_entities",
        "correction_rounds",
        "eval_stable_queries",
        "print_every",
    )
    for name in positive_ints:
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}.")
    positive_floats = (
        "foundation_lr",
        "consequence_pretrain_lr",
        "meta_lr",
        "gradient_clip",
        "covariance_ridge",
        "fast_capacity",
    )
    for name in positive_floats:
        value = getattr(args, name)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    for name in (
        "semantic_consistency_weight",
        "context_separation_weight",
        "proposal_weight",
        "source_evidence_weight",
        "write_cost",
    ):
        value = getattr(args, name)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative and finite.")
    if args.meta_train_entities >= args.num_entities:
        raise ValueError("--meta-train-entities must leave at least one held-out entity.")
    if args.reliable_sources >= args.num_sources:
        raise ValueError("--reliable-sources must leave at least one unreliable source.")
    if args.num_relations < 2:
        raise ValueError("--num-relations must be at least two to test consequence propagation.")
    if args.num_variants < 3:
        raise ValueError("--num-variants must be at least three to test held-out paraphrases.")
    if args.correction_entities > args.meta_train_entities:
        raise ValueError("--correction-entities exceeds the meta-training entity pool.")
    for name in (
        "foundation_accuracy_threshold",
        "direct_accuracy_threshold",
        "ripple_accuracy_threshold",
        "paraphrase_accuracy_threshold",
        "historical_accuracy_threshold",
        "locality_accuracy_threshold",
        "consequence_accuracy_threshold",
    ):
        validate_probability(name, getattr(args, name))
    if not math.isfinite(args.correction_margin_threshold):
        raise ValueError("--correction-margin-threshold must be finite.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def alternative_value(current: int, *, num_values: int, generator: random.Random) -> int:
    if num_values < 2:
        raise ValueError("Alternative values require at least two values.")
    offset = generator.randrange(1, num_values)
    return (current + offset) % num_values


def build_world(
    *,
    num_entities: int,
    num_values: int,
    num_relations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = random.Random(seed)
    base_values = torch.tensor(
        [generator.randrange(num_values) for _ in range(num_entities)],
        dtype=torch.long,
    )
    relation_maps = [torch.arange(num_values, dtype=torch.long)]
    for _relation in range(1, num_relations):
        permutation = list(range(num_values))
        generator.shuffle(permutation)
        relation_maps.append(torch.tensor(permutation, dtype=torch.long))
    return base_values, torch.stack(relation_maps)


def relational_targets(
    values: torch.Tensor,
    relations: torch.Tensor,
    relation_maps: torch.Tensor,
) -> torch.Tensor:
    if values.shape != relations.shape:
        raise ValueError("Values and relations must share a shape.")
    return relation_maps.to(values.device)[relations, values]


def all_foundation_queries(
    *,
    num_entities: int,
    num_relations: int,
    num_variants: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [
        (entity, relation, context, variant)
        for entity in range(num_entities)
        for relation in range(num_relations)
        for context in range(2)
        for variant in range(num_variants)
    ]
    columns = list(zip(*rows, strict=True))
    return tuple(
        torch.tensor(column, device=device, dtype=torch.long)
        for column in columns
    )  # type: ignore[return-value]


def train_foundation(
    backbone: SemanticBackbone,
    *,
    base_values: torch.Tensor,
    relation_maps: torch.Tensor,
    epochs: int,
    learning_rate: float,
    semantic_weight: float,
    context_separation_weight: float,
    device: torch.device,
    print_every: int,
) -> list[dict[str, float]]:
    if semantic_weight < 0.0 or not math.isfinite(semantic_weight):
        raise ValueError("Semantic consistency weight must be non-negative and finite.")
    if context_separation_weight < 0.0 or not math.isfinite(context_separation_weight):
        raise ValueError("Context separation weight must be non-negative and finite.")
    entities, relations, contexts, variants = all_foundation_queries(
        num_entities=backbone.num_entities,
        num_relations=backbone.num_relations,
        num_variants=backbone.num_variants,
        device=device,
    )
    values = base_values.to(device)[entities]
    targets = relational_targets(values, relations, relation_maps)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=learning_rate)
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        hidden, logits = backbone(entities, relations, contexts, variants)
        prediction_loss = F.cross_entropy(logits, targets)
        shaped = hidden.reshape(
            backbone.num_entities,
            backbone.num_relations,
            2,
            backbone.num_variants,
            backbone.d_model,
        )
        variant_mean = shaped.mean(dim=3, keepdim=True)
        semantic_loss = (shaped - variant_mean).square().mean()
        context_means = shaped.mean(dim=3)
        context_cosine = F.cosine_similarity(
            context_means[:, :, 0],
            context_means[:, :, 1],
            dim=2,
        )
        context_loss = torch.relu(context_cosine).square().mean()
        loss = (
            prediction_loss
            + semantic_weight * semantic_loss
            + context_separation_weight * context_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Foundation objective became non-finite.")
        loss.backward()
        optimizer.step()
        accuracy = float((logits.argmax(dim=1) == targets).to(torch.float32).mean().detach().cpu())
        row = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "prediction_loss": float(prediction_loss.detach().cpu()),
            "semantic_loss": float(semantic_loss.detach().cpu()),
            "context_loss": float(context_loss.detach().cpu()),
            "accuracy": accuracy,
        }
        trace.append(row)
        if epoch == 1 or epoch == epochs or epoch % print_every == 0:
            print(
                f"foundation epoch={epoch:4d} loss={row['loss']:.5f} "
                f"accuracy={accuracy:.4f} semantic={row['semantic_loss']:.5f} "
                f"context={row['context_loss']:.5f}"
            )
    return trace


def pretrain_consequence_reasoner(
    system: SemanticContextSystem,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    print_every: int,
) -> list[dict[str, float]]:
    """Distill relational consequences already represented by the foundation.

    The teacher targets are the frozen model's own predictions.  No source
    reliability, correction identity, event kind, or hidden world truth is used.
    """

    if epochs <= 0:
        raise ValueError("Consequence pretraining epochs must be positive.")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("Consequence pretraining learning rate must be positive and finite.")
    optimizer = torch.optim.AdamW(system.plasticity.parameters(), lr=learning_rate)
    entities = torch.arange(
        system.backbone.num_entities,
        device=device,
        dtype=torch.long,
    )
    direct_relations = torch.zeros_like(entities)
    current_contexts = torch.zeros_like(entities)
    canonical_variants = torch.zeros_like(entities)
    with torch.no_grad():
        direct_hidden, direct_logits = system.backbone(
            entities,
            direct_relations,
            current_contexts,
            canonical_variants,
        )
        direct_values = direct_logits.argmax(dim=1)
        relation_ids = torch.arange(
            system.backbone.num_relations,
            device=device,
            dtype=torch.long,
        )
        teacher_rows: list[torch.Tensor] = []
        for entity in range(system.backbone.num_entities):
            entity_ids = torch.full_like(relation_ids, entity)
            _hidden, relation_logits = system.backbone(
                entity_ids,
                relation_ids,
                torch.zeros_like(relation_ids),
                torch.zeros_like(relation_ids),
            )
            teacher_rows.append(torch.softmax(relation_logits, dim=1))
        teacher = torch.stack(teacher_rows)
    zero_source = torch.zeros(
        system.plasticity.source_hidden_dim,
        device=device,
        dtype=direct_hidden.dtype,
    )
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted_rows: list[torch.Tensor] = []
        confidence_rows: list[torch.Tensor] = []
        for entity in range(system.backbone.num_entities):
            value_embedding = system.plasticity.value_embedding(direct_values[entity])
            predicted, confidence = system.plasticity.predict_consequences(
                event_hidden=direct_hidden[entity : entity + 1],
                source_hidden=zero_source,
                value_embedding=value_embedding,
                global_hidden=zero_source,
            )
            predicted_rows.append(predicted)
            confidence_rows.append(confidence)
        predicted = torch.stack(predicted_rows)
        confidence = torch.stack(confidence_rows)
        eps = torch.finfo(predicted.dtype).eps
        distillation = -(
            teacher * predicted.clamp_min(eps).log()
        ).sum(dim=2).mean()
        confidence_loss = F.binary_cross_entropy(
            confidence,
            torch.ones_like(confidence),
        )
        loss = distillation + 0.05 * confidence_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Consequence pretraining objective became non-finite.")
        loss.backward()
        optimizer.step()
        accuracy = float(
            (predicted.argmax(dim=2) == teacher.argmax(dim=2))
            .to(torch.float32)
            .mean()
            .detach()
            .cpu()
        )
        row = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "distillation": float(distillation.detach().cpu()),
            "confidence_loss": float(confidence_loss.detach().cpu()),
            "accuracy": accuracy,
        }
        trace.append(row)
        if epoch == 1 or epoch == epochs or epoch % print_every == 0:
            print(
                f"consequence epoch={epoch:4d} loss={row['loss']:.5f} "
                f"accuracy={accuracy:.4f}"
            )
    return trace


@torch.no_grad()
def foundation_covariance_inverse(
    backbone: SemanticBackbone,
    *,
    ridge: float,
    device: torch.device,
) -> torch.Tensor:
    entities, relations, contexts, variants = all_foundation_queries(
        num_entities=backbone.num_entities,
        num_relations=backbone.num_relations,
        num_variants=backbone.num_variants,
        device=device,
    )
    hidden, _logits = backbone(entities, relations, contexts, variants)
    hidden_cpu = hidden.detach().to(device="cpu")
    hidden_cpu = hidden_cpu.to(dtype=torch.float64)
    covariance = hidden_cpu.transpose(0, 1) @ hidden_cpu / float(hidden_cpu.shape[0])
    covariance = covariance + ridge * torch.eye(backbone.d_model, dtype=torch.float64)
    inverse = torch.linalg.inv(covariance)
    if not torch.isfinite(inverse).all():
        raise FloatingPointError("Foundation covariance inverse is non-finite.")
    inverse = inverse.to(dtype=hidden.dtype)
    return inverse.to(device=device)


def build_episode(
    *,
    entity_pool: tuple[int, ...],
    base_values: torch.Tensor,
    num_values: int,
    num_sources: int,
    reliable_sources: int,
    calibration_events_per_source: int,
    correction_entities: int,
    correction_rounds: int,
    seed: int,
) -> Episode:
    generator = random.Random(seed)
    if correction_entities > len(entity_pool):
        raise ValueError("Correction count exceeds the available entity pool.")
    reliability_ids = set(generator.sample(range(num_sources), reliable_sources))
    reliability = tuple(source in reliability_ids for source in range(num_sources))
    corrected = tuple(generator.sample(list(entity_pool), correction_entities))
    corrected_values = tuple(
        alternative_value(
            int(base_values[entity]),
            num_values=num_values,
            generator=generator,
        )
        for entity in corrected
    )

    calibration: list[ReportEvent] = []
    calibration_pool = [entity for entity in entity_pool if entity not in corrected]
    if not calibration_pool:
        raise RuntimeError("Calibration requires at least one non-corrected entity.")
    for source in range(num_sources):
        for _index in range(calibration_events_per_source):
            entity = generator.choice(calibration_pool)
            truth = int(base_values[entity])
            observed = (
                truth
                if reliability[source]
                else alternative_value(truth, num_values=num_values, generator=generator)
            )
            calibration.append(
                ReportEvent(
                    entity=entity,
                    observed_value=observed,
                    source=source,
                    source_is_reliable=reliability[source],
                    phase="calibration",
                )
            )
    generator.shuffle(calibration)

    correction_reports: list[ReportEvent] = []
    for _round in range(correction_rounds):
        round_rows: list[ReportEvent] = []
        for entity, corrected_value in zip(corrected, corrected_values, strict=True):
            old_value = int(base_values[entity])
            for source in range(num_sources):
                observed = corrected_value if reliability[source] else old_value
                round_rows.append(
                    ReportEvent(
                        entity=entity,
                        observed_value=observed,
                        source=source,
                        source_is_reliable=reliability[source],
                        phase="correction",
                    )
                )
        generator.shuffle(round_rows)
        correction_reports.extend(round_rows)
    return Episode(
        events=tuple(calibration + correction_reports),
        corrected_entities=corrected,
        corrected_values=corrected_values,
        source_reliability=reliability,
    )


def current_truth_values(
    *,
    base_values: torch.Tensor,
    episode: Episode,
    corrections_active: bool,
) -> torch.Tensor:
    truth = base_values.clone()
    if corrections_active:
        for entity, value in zip(
            episode.corrected_entities,
            episode.corrected_values,
            strict=True,
        ):
            truth[entity] = value
    return truth


def make_query_batch(
    rows: list[tuple[int, int, int, int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("Query batch must not be empty.")
    columns = list(zip(*rows, strict=True))
    return tuple(
        torch.tensor(column, device=device, dtype=torch.long)
        for column in columns
    )  # type: ignore[return-value]


def sampled_future_loss(
    system: SemanticContextSystem,
    *,
    state: FastModeState,
    episode: Episode,
    base_values: torch.Tensor,
    relation_maps: torch.Tensor,
    corrections_active: bool,
    stable_queries: int,
    train_variants: tuple[int, ...],
    entity_pool: tuple[int, ...],
    generator: random.Random,
    device: torch.device,
) -> torch.Tensor:
    truth = current_truth_values(
        base_values=base_values,
        episode=episode,
        corrections_active=corrections_active,
    )
    corrected = set(episode.corrected_entities)
    stable_pool = [entity for entity in entity_pool if entity not in corrected]
    if not stable_pool:
        raise RuntimeError("Future loss requires at least one stable entity.")
    rows: list[tuple[int, int, int, int]] = []
    for entity in episode.corrected_entities:
        for relation in range(system.backbone.num_relations):
            variant = generator.choice(train_variants)
            rows.append((entity, relation, 0, variant))
            rows.append((entity, relation, 1, variant))
    for _index in range(stable_queries):
        entity = generator.choice(stable_pool)
        relation = generator.randrange(system.backbone.num_relations)
        context = generator.randrange(2)
        variant = generator.choice(train_variants)
        rows.append((entity, relation, context, variant))
    entities, relations, contexts, variants = make_query_batch(rows, device=device)
    values = torch.where(
        contexts.cpu() == 0,
        truth[entities.cpu()],
        base_values[entities.cpu()],
    ).to(device)
    targets = relational_targets(values, relations, relation_maps)
    _hidden, logits, _attention = system.predict(
        entities=entities,
        relations=relations,
        contexts=contexts,
        variants=variants,
        state=state,
    )
    return F.cross_entropy(logits, targets)


def run_episode(
    system: SemanticContextSystem,
    episode: Episode,
    *,
    base_values: torch.Tensor,
    relation_maps: torch.Tensor,
    covariance_inverse: torch.Tensor,
    entity_pool: tuple[int, ...],
    train_variants: tuple[int, ...],
    stable_queries: int,
    fast_capacity: float,
    proposal_weight: float,
    source_evidence_weight: float,
    write_cost: float,
    seed: int,
    device: torch.device,
    collect: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if proposal_weight < 0.0 or not math.isfinite(proposal_weight):
        raise ValueError("Proposal weight must be non-negative and finite.")
    if source_evidence_weight < 0.0 or not math.isfinite(source_evidence_weight):
        raise ValueError("Source evidence weight must be non-negative and finite.")
    if write_cost < 0.0 or not math.isfinite(write_cost):
        raise ValueError("Write cost must be non-negative and finite.")
    dtype = next(system.plasticity.parameters()).dtype
    state = system.plasticity.initial_state(device=device, dtype=dtype)
    generator = random.Random(seed)
    losses: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    gate_reliable: list[torch.Tensor] = []
    gate_unreliable: list[torch.Tensor] = []
    consequence_correct: list[torch.Tensor] = []
    reliable_consequence_correct: list[torch.Tensor] = []

    for event_index, event in enumerate(episode.events):
        variant = generator.choice(train_variants)
        entity_tensor = torch.tensor([event.entity], device=device, dtype=torch.long)
        direct_relation = torch.zeros(1, device=device, dtype=torch.long)
        current_context = torch.zeros(1, device=device, dtype=torch.long)
        variant_tensor = torch.tensor([variant], device=device, dtype=torch.long)
        event_hidden, event_logits, _event_attention = system.predict(
            entities=entity_tensor,
            relations=direct_relation,
            contexts=current_context,
            variants=variant_tensor,
            state=state,
        )
        (
            state,
            write_gate,
            source_hidden,
            value_embedding,
            prior_source_agreement,
            prior_source_confidence,
        ) = system.plasticity.evidence_step(
            hidden=event_hidden,
            logits=event_logits,
            observed_value=event.observed_value,
            source=event.source,
            state=state,
        )
        target_probabilities, relation_confidence = system.plasticity.predict_consequences(
            event_hidden=event_hidden,
            source_hidden=source_hidden,
            value_embedding=value_embedding,
            global_hidden=state.global_hidden,
        )
        observed_one_hot = F.one_hot(
            torch.tensor(event.observed_value, device=device, dtype=torch.long),
            num_classes=system.backbone.num_values,
        ).to(dtype=target_probabilities.dtype)
        target_probabilities = torch.cat(
            [observed_one_hot.unsqueeze(0), target_probabilities[1:]],
            dim=0,
        )
        relation_ids = torch.arange(
            system.backbone.num_relations,
            device=device,
            dtype=torch.long,
        )
        entity_ids = torch.full_like(relation_ids, event.entity)
        context_ids = torch.zeros_like(relation_ids)
        canonical_variants = torch.zeros_like(relation_ids)
        relation_hidden, relation_logits, _relation_attention = system.predict(
            entities=entity_ids,
            relations=relation_ids,
            contexts=context_ids,
            variants=canonical_variants,
            state=state,
        )
        teacher_probabilities = torch.softmax(relation_logits.detach(), dim=1)
        observed_probability = torch.softmax(event_logits.detach(), dim=1)[
            0, event.observed_value
        ]
        eps = torch.finfo(target_probabilities.dtype).eps
        derived_proposal_loss = -(
            teacher_probabilities[1:]
            * target_probabilities[1:].clamp_min(eps).log()
        ).sum(dim=1).mean()
        proposal_loss = observed_probability * derived_proposal_loss
        source_evidence_loss = prior_source_confidence * F.binary_cross_entropy(
            write_gate,
            prior_source_agreement,
        )
        state = system.plasticity.write(
            query_hidden=relation_hidden,
            current_logits=relation_logits,
            target_probabilities=target_probabilities,
            relation_confidence=relation_confidence,
            write_gate=write_gate,
            covariance_inverse=covariance_inverse,
            state=state,
            fast_capacity=fast_capacity,
        )

        if event.source_is_reliable:
            gate_reliable.append(write_gate)
        else:
            gate_unreliable.append(write_gate)
        true_direct = (
            episode.corrected_values[episode.corrected_entities.index(event.entity)]
            if event.phase == "correction" and event.entity in episode.corrected_entities
            else int(base_values[event.entity])
        )
        true_values = torch.full(
            (system.backbone.num_relations,),
            true_direct,
            device=device,
            dtype=torch.long,
        )
        true_consequences = relational_targets(true_values, relation_ids, relation_maps)
        consequence_correct.append(
            (target_probabilities.argmax(dim=1) == true_consequences).to(torch.float32).mean()
        )
        if event.source_is_reliable:
            reliable_consequence_correct.append(consequence_correct[-1])

        corrections_active = event.phase == "correction"
        future_loss = sampled_future_loss(
            system,
            state=state,
            episode=episode,
            base_values=base_values,
            relation_maps=relation_maps,
            corrections_active=corrections_active,
            stable_queries=stable_queries,
            train_variants=train_variants,
            entity_pool=entity_pool,
            generator=generator,
            device=device,
        )
        capacity_penalty = torch.linalg.vector_norm(state.fast) / fast_capacity
        losses.append(
            future_loss
            + proposal_weight * proposal_loss
            + source_evidence_weight * source_evidence_loss
            + write_cost * write_gate
            + 1e-3 * capacity_penalty.square()
        )
        if collect:
            rows.append(
                {
                    "index": event_index,
                    "phase": event.phase,
                    "entity": event.entity,
                    "source": event.source,
                    "source_is_reliable": event.source_is_reliable,
                    "observed_value": event.observed_value,
                    "write_gate": float(write_gate.detach().cpu()),
                    "mean_relation_confidence": float(relation_confidence.mean().detach().cpu()),
                    "consequence_accuracy": float(consequence_correct[-1].detach().cpu()),
                    "future_loss": float(future_loss.detach().cpu()),
                    "proposal_loss": float(proposal_loss.detach().cpu()),
                    "source_evidence_loss": float(source_evidence_loss.detach().cpu()),
                    "fast_norm": float(torch.linalg.vector_norm(state.fast).detach().cpu()),
                }
            )
    if (
        not losses
        or not gate_reliable
        or not gate_unreliable
        or not reliable_consequence_correct
    ):
        raise RuntimeError("Episode did not produce complete training evidence.")
    objective = torch.stack(losses).mean()
    report = {
        "objective": float(objective.detach().cpu()),
        "reliable_write_gate": float(torch.stack(gate_reliable).mean().detach().cpu()),
        "unreliable_write_gate": float(torch.stack(gate_unreliable).mean().detach().cpu()),
        "consequence_accuracy": float(torch.stack(consequence_correct).mean().detach().cpu()),
        "reliable_consequence_accuracy": float(
            torch.stack(reliable_consequence_correct).mean().detach().cpu()
        ),
        "fast_norm": float(torch.linalg.vector_norm(state.fast).detach().cpu()),
        "effective_modes": float(
            torch.exp(
                -(
                    state.mass.div(state.mass.sum().clamp_min(torch.finfo(dtype).eps))
                    * state.mass.div(state.mass.sum().clamp_min(torch.finfo(dtype).eps))
                    .clamp_min(torch.finfo(dtype).eps)
                    .log()
                ).sum()
            )
            .detach()
            .cpu()
        ),
        "events": rows,
        "state": state,
    }
    return objective, report


@torch.no_grad()
def evaluate_episode(
    system: SemanticContextSystem,
    episode: Episode,
    *,
    state: FastModeState,
    base_values: torch.Tensor,
    relation_maps: torch.Tensor,
    entity_pool: tuple[int, ...],
    heldout_variants: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    corrected = set(episode.corrected_entities)
    stable = [entity for entity in entity_pool if entity not in corrected]
    if not stable:
        raise RuntimeError("Evaluation requires stable entities.")
    truth = current_truth_values(
        base_values=base_values,
        episode=episode,
        corrections_active=True,
    )

    def evaluate_rows(rows: list[tuple[int, int, int, int]], values: torch.Tensor) -> tuple[float, torch.Tensor]:
        entities, relations, contexts, variants = make_query_batch(rows, device=device)
        targets = relational_targets(values.to(device), relations, relation_maps)
        _hidden, logits, _attention = system.predict(
            entities=entities,
            relations=relations,
            contexts=contexts,
            variants=variants,
            state=state,
        )
        accuracy = float((logits.argmax(dim=1) == targets).to(torch.float32).mean().cpu())
        return accuracy, logits

    direct_rows = [
        (entity, 0, 0, heldout_variants[0])
        for entity in episode.corrected_entities
    ]
    direct_values = truth[torch.tensor(episode.corrected_entities)]
    direct_accuracy, direct_logits = evaluate_rows(direct_rows, direct_values)
    old_values = base_values[torch.tensor(episode.corrected_entities)].to(device)
    old_targets = relational_targets(
        old_values,
        torch.zeros_like(old_values),
        relation_maps,
    )
    new_targets = relational_targets(
        direct_values.to(device),
        torch.zeros_like(direct_values, device=device),
        relation_maps,
    )
    row_ids = torch.arange(direct_logits.shape[0], device=device)
    correction_margin = float(
        (direct_logits[row_ids, new_targets] - direct_logits[row_ids, old_targets]).mean().cpu()
    )

    ripple_rows = [
        (entity, relation, 0, heldout_variants[0])
        for entity in episode.corrected_entities
        for relation in range(1, system.backbone.num_relations)
    ]
    ripple_base = torch.tensor(
        [int(truth[entity]) for entity in episode.corrected_entities for _ in range(1, system.backbone.num_relations)],
        dtype=torch.long,
    )
    ripple_accuracy, _ripple_logits = evaluate_rows(ripple_rows, ripple_base)

    paraphrase_rows = [
        (entity, relation, 0, variant)
        for entity in episode.corrected_entities
        for relation in range(system.backbone.num_relations)
        for variant in heldout_variants
    ]
    paraphrase_values = torch.tensor(
        [
            int(truth[entity])
            for entity in episode.corrected_entities
            for _relation in range(system.backbone.num_relations)
            for _variant in heldout_variants
        ],
        dtype=torch.long,
    )
    paraphrase_accuracy, _paraphrase_logits = evaluate_rows(paraphrase_rows, paraphrase_values)

    historical_rows = [
        (entity, relation, 1, heldout_variants[0])
        for entity in episode.corrected_entities
        for relation in range(system.backbone.num_relations)
    ]
    historical_values = torch.tensor(
        [
            int(base_values[entity])
            for entity in episode.corrected_entities
            for _relation in range(system.backbone.num_relations)
        ],
        dtype=torch.long,
    )
    historical_accuracy, _historical_logits = evaluate_rows(historical_rows, historical_values)

    locality_rows = [
        (entity, relation, context, heldout_variants[0])
        for entity in stable
        for relation in range(system.backbone.num_relations)
        for context in range(2)
    ]
    locality_values = torch.tensor(
        [int(base_values[entity]) for entity in stable for _relation in range(system.backbone.num_relations) for _context in range(2)],
        dtype=torch.long,
    )
    locality_accuracy, _locality_logits = evaluate_rows(locality_rows, locality_values)

    same_entity = episode.corrected_entities[0]
    paraphrase_route_rows = [
        (same_entity, 0, 0, variant)
        for variant in heldout_variants
    ]
    entities, relations, contexts, variants = make_query_batch(paraphrase_route_rows, device=device)
    _h, _l, paraphrase_attention = system.predict(
        entities=entities,
        relations=relations,
        contexts=contexts,
        variants=variants,
        state=state,
    )
    paraphrase_route_similarity = float(
        F.cosine_similarity(
            paraphrase_attention[0].unsqueeze(0),
            paraphrase_attention[1:],
            dim=1,
        ).mean().cpu()
    )
    unrelated_entity = stable[0]
    comparison_rows = [
        (same_entity, 0, 0, heldout_variants[0]),
        (unrelated_entity, 0, 0, heldout_variants[0]),
    ]
    entities, relations, contexts, variants = make_query_batch(comparison_rows, device=device)
    _h, _l, comparison_attention = system.predict(
        entities=entities,
        relations=relations,
        contexts=contexts,
        variants=variants,
        state=state,
    )
    unrelated_route_similarity = float(
        F.cosine_similarity(
            comparison_attention[0].unsqueeze(0),
            comparison_attention[1].unsqueeze(0),
            dim=1,
        ).item()
    )
    return {
        "direct_accuracy": direct_accuracy,
        "ripple_accuracy": ripple_accuracy,
        "paraphrase_accuracy": paraphrase_accuracy,
        "historical_accuracy": historical_accuracy,
        "locality_accuracy": locality_accuracy,
        "correction_margin": correction_margin,
        "paraphrase_route_similarity": paraphrase_route_similarity,
        "unrelated_route_similarity": unrelated_route_similarity,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate empty metric rows.")
    keys = rows[0].keys()
    if any(row.keys() != keys for row in rows[1:]):
        raise RuntimeError("Metric rows have inconsistent schemas.")
    return {key: sum(row[key] for row in rows) / float(len(rows)) for key in keys}


def plot_results(
    foundation_trace: list[dict[str, float]],
    meta_trace: list[dict[str, float]],
    final: dict[str, float],
    output_path: Path,
) -> None:
    if not foundation_trace or not meta_trace:
        raise ValueError("Plotting requires foundation and meta-training traces.")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes[0, 0].plot(
        [row["epoch"] for row in foundation_trace],
        [row["accuracy"] for row in foundation_trace],
        color="#2563eb",
    )
    axes[0, 0].set_title("Foundation accuracy")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylim(0.0, 1.02)

    axes[0, 1].plot(
        [row["episode"] for row in meta_trace],
        [row["objective"] for row in meta_trace],
        color="#7c3aed",
    )
    axes[0, 1].set_title("Future-loss meta objective")
    axes[0, 1].set_xlabel("episode")

    behavior_names = [
        "direct_accuracy",
        "ripple_accuracy",
        "paraphrase_accuracy",
        "historical_accuracy",
        "locality_accuracy",
    ]
    axes[1, 0].bar(
        [name.replace("_accuracy", "") for name in behavior_names],
        [final[name] for name in behavior_names],
        color=["#dc2626", "#ea580c", "#ca8a04", "#16a34a", "#0891b2"],
    )
    axes[1, 0].set_title("Held-out semantic behavior")
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].tick_params(axis="x", rotation=25)

    axes[1, 1].bar(
        ["reliable write", "unreliable write", "same-context route", "unrelated route"],
        [
            final["reliable_write_gate"],
            final["unreliable_write_gate"],
            final["paraphrase_route_similarity"],
            final["unrelated_route_similarity"],
        ],
        color=["#16a34a", "#dc2626", "#2563eb", "#9ca3af"],
    )
    axes[1, 1].set_title("Learned evidence and routing")
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = ExperimentThresholds(
        foundation_accuracy=args.foundation_accuracy_threshold,
        direct_accuracy=args.direct_accuracy_threshold,
        ripple_accuracy=args.ripple_accuracy_threshold,
        paraphrase_accuracy=args.paraphrase_accuracy_threshold,
        historical_accuracy=args.historical_accuracy_threshold,
        locality_accuracy=args.locality_accuracy_threshold,
        consequence_accuracy=args.consequence_accuracy_threshold,
        correction_margin=args.correction_margin_threshold,
    )
    base_values, relation_maps = build_world(
        num_entities=args.num_entities,
        num_values=args.num_values,
        num_relations=args.num_relations,
        seed=args.seed,
    )
    backbone = SemanticBackbone(
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        num_contexts=2,
        num_variants=args.num_variants,
        num_values=args.num_values,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
    ).to(device)
    foundation_trace = train_foundation(
        backbone,
        base_values=base_values,
        relation_maps=relation_maps,
        epochs=args.foundation_epochs,
        learning_rate=args.foundation_lr,
        semantic_weight=args.semantic_consistency_weight,
        context_separation_weight=args.context_separation_weight,
        device=device,
        print_every=args.print_every,
    )
    foundation_accuracy = foundation_trace[-1]["accuracy"]
    if foundation_accuracy < thresholds.foundation_accuracy:
        raise RuntimeError(
            f"Foundation accuracy {foundation_accuracy:.4f} is below the required "
            f"{thresholds.foundation_accuracy:.4f}; semantic plasticity was not tested."
        )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    covariance_inverse = foundation_covariance_inverse(
        backbone,
        ridge=args.covariance_ridge,
        device=device,
    )
    plasticity = SemanticPlasticity(
        num_slots=args.num_slots,
        num_sources=args.num_sources,
        num_relations=args.num_relations,
        num_values=args.num_values,
        d_model=args.d_model,
        source_hidden_dim=args.source_hidden_dim,
    ).to(device)
    system = SemanticContextSystem(backbone, plasticity).to(device)
    print("\nFOUNDATION CONSEQUENCE DISTILLATION")
    print("=" * 128)
    consequence_trace = pretrain_consequence_reasoner(
        system,
        epochs=args.consequence_pretrain_epochs,
        learning_rate=args.consequence_pretrain_lr,
        device=device,
        print_every=args.print_every,
    )
    optimizer = torch.optim.AdamW(plasticity.parameters(), lr=args.meta_lr)
    train_pool = tuple(range(args.meta_train_entities))
    test_pool = tuple(range(args.meta_train_entities, args.num_entities))
    train_variants = tuple(range(args.num_variants - 2))
    heldout_variants = tuple(range(args.num_variants - 2, args.num_variants))
    if not train_variants or len(heldout_variants) != 2:
        raise RuntimeError("Variant split failed to produce training and two held-out variants.")

    print("\nSEMANTIC CONTEXT PLASTICITY META-TRAINING")
    print("=" * 128)
    print(
        f"device={device} slots={args.num_slots} train_entities={len(train_pool)} "
        f"heldout_entities={len(test_pool)} episodes={args.meta_episodes}"
    )
    meta_trace: list[dict[str, float]] = []
    progress = tqdm(range(1, args.meta_episodes + 1), desc="semantic plasticity")
    for episode_index in progress:
        episode = build_episode(
            entity_pool=train_pool,
            base_values=base_values,
            num_values=args.num_values,
            num_sources=args.num_sources,
            reliable_sources=args.reliable_sources,
            calibration_events_per_source=args.calibration_events_per_source,
            correction_entities=args.correction_entities,
            correction_rounds=args.correction_rounds,
            seed=args.seed * 100_003 + episode_index,
        )
        optimizer.zero_grad(set_to_none=True)
        objective, report = run_episode(
            system,
            episode,
            base_values=base_values,
            relation_maps=relation_maps,
            covariance_inverse=covariance_inverse,
            entity_pool=train_pool,
            train_variants=train_variants,
            stable_queries=args.eval_stable_queries,
            fast_capacity=args.fast_capacity,
            proposal_weight=args.proposal_weight,
            source_evidence_weight=args.source_evidence_weight,
            write_cost=args.write_cost,
            seed=args.seed * 1_000_003 + episode_index,
            device=device,
            collect=False,
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Meta objective became non-finite at episode {episode_index}.")
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(plasticity.parameters(), args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Meta gradient became non-finite at episode {episode_index}.")
        optimizer.step()
        row = {
            "episode": float(episode_index),
            "objective": report["objective"],
            "reliable_write_gate": report["reliable_write_gate"],
            "unreliable_write_gate": report["unreliable_write_gate"],
            "consequence_accuracy": report["consequence_accuracy"],
            "reliable_consequence_accuracy": report["reliable_consequence_accuracy"],
            "fast_norm": report["fast_norm"],
            "effective_modes": report["effective_modes"],
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        meta_trace.append(row)
        progress.set_postfix(loss=f"{row['objective']:.4f}")
        if episode_index == 1 or episode_index % args.print_every == 0:
            print(
                f"episode={episode_index:4d} objective={row['objective']:.5f} "
                f"gate={row['reliable_write_gate']:.3f}/{row['unreliable_write_gate']:.3f} "
                f"consequence={row['reliable_consequence_accuracy']:.3f} "
                f"modes={row['effective_modes']:.2f}"
            )

    test_rows: list[dict[str, float]] = []
    detailed_episode: dict[str, Any] | None = None
    system.eval()
    for test_index in range(args.test_episodes):
        episode = build_episode(
            entity_pool=test_pool,
            base_values=base_values,
            num_values=args.num_values,
            num_sources=args.num_sources,
            reliable_sources=args.reliable_sources,
            calibration_events_per_source=args.calibration_events_per_source,
            correction_entities=min(args.correction_entities, len(test_pool) - 1),
            correction_rounds=args.correction_rounds,
            seed=args.seed * 200_003 + test_index,
        )
        with torch.no_grad():
            _objective, episode_report = run_episode(
                system,
                episode,
                base_values=base_values,
                relation_maps=relation_maps,
                covariance_inverse=covariance_inverse,
                entity_pool=test_pool,
                train_variants=train_variants,
                stable_queries=args.eval_stable_queries,
                fast_capacity=args.fast_capacity,
                proposal_weight=args.proposal_weight,
                source_evidence_weight=args.source_evidence_weight,
                write_cost=args.write_cost,
                seed=args.seed * 2_000_003 + test_index,
                device=device,
                collect=test_index == 0,
            )
            state = episode_report.pop("state")
            semantic = evaluate_episode(
                system,
                episode,
                state=state,
                base_values=base_values,
                relation_maps=relation_maps,
                entity_pool=test_pool,
                heldout_variants=heldout_variants,
                device=device,
            )
        metrics = {
            "objective": episode_report["objective"],
            "reliable_write_gate": episode_report["reliable_write_gate"],
            "unreliable_write_gate": episode_report["unreliable_write_gate"],
            "consequence_accuracy": episode_report["consequence_accuracy"],
            "reliable_consequence_accuracy": episode_report[
                "reliable_consequence_accuracy"
            ],
            "fast_norm": episode_report["fast_norm"],
            "effective_modes": episode_report["effective_modes"],
            **semantic,
        }
        test_rows.append(metrics)
        if test_index == 0:
            detailed_episode = {
                "episode": asdict(episode),
                "events": episode_report["events"],
                "metrics": metrics,
            }
    final = aggregate(test_rows)
    validation = {
        "foundation_ready": foundation_accuracy >= thresholds.foundation_accuracy,
        "direct_replacement": final["direct_accuracy"] >= thresholds.direct_accuracy,
        "ripple_propagation": final["ripple_accuracy"] >= thresholds.ripple_accuracy,
        "paraphrase_generalization": final["paraphrase_accuracy"] >= thresholds.paraphrase_accuracy,
        "historical_retention": final["historical_accuracy"] >= thresholds.historical_accuracy,
        "unrelated_locality": final["locality_accuracy"] >= thresholds.locality_accuracy,
        "consequence_reasoning": final["reliable_consequence_accuracy"]
        >= thresholds.consequence_accuracy,
        "positive_correction_margin": final["correction_margin"] >= thresholds.correction_margin,
        "source_discrimination": final["reliable_write_gate"] > final["unreliable_write_gate"],
        "contextual_routing": final["paraphrase_route_similarity"] > final["unrelated_route_similarity"],
        "capacity_respected": final["fast_norm"] <= args.fast_capacity + 1e-5,
    }

    print("\nFINAL SEMANTIC CONTEXT PLASTICITY STATE")
    print("-" * 128)
    print(
        f"direct={final['direct_accuracy']:.4f} ripple={final['ripple_accuracy']:.4f} "
        f"paraphrase={final['paraphrase_accuracy']:.4f} history={final['historical_accuracy']:.4f} "
        f"locality={final['locality_accuracy']:.4f} margin={final['correction_margin']:.4f}"
    )
    print(
        f"write_gate reliable/unreliable={final['reliable_write_gate']:.4f}/"
        f"{final['unreliable_write_gate']:.4f} "
        f"reliable_consequence={final['reliable_consequence_accuracy']:.4f} "
        f"routes same/unrelated={final['paraphrase_route_similarity']:.4f}/"
        f"{final['unrelated_route_similarity']:.4f} modes={final['effective_modes']:.3f}"
    )
    print(f"validation={validation}")

    plot_path = args.output_dir / "semantic_context_plasticity.png"
    json_path = args.output_dir / "semantic_context_plasticity.json"
    checkpoint_path = args.output_dir / "semantic_context_plasticity.pt"
    plot_results(foundation_trace, meta_trace, final, plot_path)
    output = {
        "question": (
            "Can a fixed-capacity neural fast-weight bank use semantic context, recurrent source evidence, "
            "and learned consequence prediction to perform local replacement without role labels?"
        ),
        "scope": (
            "Isolated semantic-plasticity experiment. The foundation is frozen, so geometry preservation "
            "is not tested here. Hidden truth, source reliability, and event phase are evaluation-only."
        ),
        "mechanism": {
            "semantic_address": "frozen distributed entity/relation/time/paraphrase representation",
            "routing": "learned context-to-fixed-mode attention",
            "fast_write": "covariance-minimal rank-one contextual update",
            "evidence": "per-source and global recurrent hidden state",
            "reasoning": "future-loss-trained relation consequence predictor",
            "decision": "continuous learned write and consequence confidence",
            "capacity": "fixed mode count and explicit global fast-weight norm",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "thresholds": asdict(thresholds),
        "foundation": {
            "final": foundation_trace[-1],
            "trace": foundation_trace,
        },
        "consequence_pretraining": {
            "final": consequence_trace[-1],
            "trace": consequence_trace,
        },
        "meta_trace": meta_trace,
        "test_episodes": test_rows,
        "detailed_episode": detailed_episode,
        "final": final,
        "validation": validation,
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "format": "semantic_context_plasticity_v1",
            "backbone": backbone.state_dict(),
            "plasticity": plasticity.state_dict(),
            "base_values": base_values,
            "relation_maps": relation_maps,
            "config": output["config"],
        },
        checkpoint_path,
    )
    print(f"wrote_json={json_path}")
    print(f"wrote_plot={plot_path}")
    print(f"wrote_checkpoint={checkpoint_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/analysis/gco-tiny-semantic-context-plasticity-seed0"),
    )
    parser.add_argument("--num-entities", type=int, default=32)
    parser.add_argument("--meta-train-entities", type=int, default=24)
    parser.add_argument("--num-values", type=int, default=8)
    parser.add_argument("--num-relations", type=int, default=3)
    parser.add_argument("--num-variants", type=int, default=4)
    parser.add_argument("--num-sources", type=int, default=4)
    parser.add_argument("--reliable-sources", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--source-hidden-dim", type=int, default=48)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--foundation-epochs", type=int, default=400)
    parser.add_argument("--foundation-lr", type=float, default=3e-3)
    parser.add_argument("--semantic-consistency-weight", type=float, default=1.0)
    parser.add_argument("--context-separation-weight", type=float, default=0.5)
    parser.add_argument("--consequence-pretrain-epochs", type=int, default=200)
    parser.add_argument("--consequence-pretrain-lr", type=float, default=2e-3)
    parser.add_argument("--meta-episodes", type=int, default=600)
    parser.add_argument("--test-episodes", type=int, default=24)
    parser.add_argument("--meta-lr", type=float, default=2e-3)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--covariance-ridge", type=float, default=1e-2)
    parser.add_argument("--fast-capacity", type=float, default=32.0)
    parser.add_argument("--proposal-weight", type=float, default=0.1)
    parser.add_argument("--source-evidence-weight", type=float, default=0.5)
    parser.add_argument("--write-cost", type=float, default=0.01)
    parser.add_argument("--calibration-events-per-source", type=int, default=3)
    parser.add_argument("--correction-entities", type=int, default=2)
    parser.add_argument("--correction-rounds", type=int, default=2)
    parser.add_argument("--eval-stable-queries", type=int, default=6)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--foundation-accuracy-threshold", type=float, default=0.99)
    parser.add_argument("--direct-accuracy-threshold", type=float, default=0.75)
    parser.add_argument("--ripple-accuracy-threshold", type=float, default=0.70)
    parser.add_argument("--paraphrase-accuracy-threshold", type=float, default=0.70)
    parser.add_argument("--historical-accuracy-threshold", type=float, default=0.90)
    parser.add_argument("--locality-accuracy-threshold", type=float, default=0.95)
    parser.add_argument("--consequence-accuracy-threshold", type=float, default=0.90)
    parser.add_argument("--correction-margin-threshold", type=float, default=0.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
