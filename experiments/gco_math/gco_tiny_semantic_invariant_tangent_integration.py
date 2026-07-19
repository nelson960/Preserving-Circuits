"""Consolidate semantic fast-weight candidates through Invariant-Tangent updates.

The semantic context plasticity model first observes an unlabeled stream and
constructs a temporary candidate behavior using recurrent source evidence,
learned consequence prediction, and context-conditioned fast modes.  This
experiment then removes the fast state and teaches the actual semantic
backbone to reproduce that candidate while protecting historical behavior and
historical representation geometry.

Hidden source reliability, correction identity, and world truth are used only
for evaluation.  The consolidation objective sees candidate probabilities
produced by the semantic plasticity mechanism, not those hidden labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.gco_math.gco_mini_cl_world_demo import (
    assign_flat_gradient,
    flat_autograd_gradient,
    project_gradient_away_from_constraints,
)
from experiments.gco_math.gco_tiny_semantic_context_plasticity import (
    Episode,
    FastModeState,
    SemanticBackbone,
    SemanticContextSystem,
    SemanticPlasticity,
    build_episode,
    current_truth_values,
    foundation_covariance_inverse,
    make_query_batch,
    relational_targets,
    seed_everything,
)
from experiments.real_book_common import resolve_device


REQUIRED_CONFIG_KEYS = (
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
    "covariance_ridge",
    "fast_capacity",
    "calibration_events_per_source",
    "correction_entities",
    "correction_rounds",
)


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.semantic_checkpoint.is_file():
        raise FileNotFoundError(f"Semantic checkpoint does not exist: {args.semantic_checkpoint}")
    for name in ("episodes", "consolidation_epochs", "constraint_rank", "print_every"):
        positive_int(name, getattr(args, name))
    for name in (
        "learning_rate",
        "projection_damping",
        "restore_strength",
        "restore_norm_ratio",
        "geometry_restore_weight",
        "max_gradient_norm",
    ):
        positive_float(name, getattr(args, name))
    if args.candidate_focus_strength < 0.0 or not math.isfinite(args.candidate_focus_strength):
        raise ValueError("candidate_focus_strength must be non-negative and finite.")
    for name in (
        "direct_accuracy_threshold",
        "ripple_accuracy_threshold",
        "paraphrase_accuracy_threshold",
        "historical_accuracy_threshold",
        "locality_accuracy_threshold",
        "protected_cka_threshold",
    ):
        probability(name, getattr(args, name))
    if not math.isfinite(args.correction_margin_threshold):
        raise ValueError("correction_margin_threshold must be finite.")


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Semantic checkpoint has no configuration dictionary.")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise RuntimeError(f"Semantic checkpoint is missing config keys: {missing}")
    return config


def construct_system(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[SemanticContextSystem, torch.Tensor, torch.Tensor, dict[str, Any]]:
    config = checkpoint_config(checkpoint)
    backbone = SemanticBackbone(
        num_entities=int(config["num_entities"]),
        num_relations=int(config["num_relations"]),
        num_contexts=2,
        num_variants=int(config["num_variants"]),
        num_values=int(config["num_values"]),
        d_model=int(config["d_model"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    plasticity = SemanticPlasticity(
        num_slots=int(config["num_slots"]),
        num_sources=int(config["num_sources"]),
        num_relations=int(config["num_relations"]),
        num_values=int(config["num_values"]),
        d_model=int(config["d_model"]),
        source_hidden_dim=int(config["source_hidden_dim"]),
    ).to(device)
    backbone_state = checkpoint.get("backbone")
    plasticity_state = checkpoint.get("plasticity")
    if not isinstance(backbone_state, dict) or not isinstance(plasticity_state, dict):
        raise RuntimeError("Semantic checkpoint is missing model state dictionaries.")
    backbone.load_state_dict(backbone_state, strict=True)
    plasticity.load_state_dict(plasticity_state, strict=True)
    base_values = checkpoint.get("base_values")
    relation_maps = checkpoint.get("relation_maps")
    if not isinstance(base_values, torch.Tensor) or not isinstance(relation_maps, torch.Tensor):
        raise RuntimeError("Semantic checkpoint is missing world tensors.")
    if base_values.shape != (backbone.num_entities,):
        raise RuntimeError(f"Unexpected base value shape: {tuple(base_values.shape)}.")
    if relation_maps.shape != (backbone.num_relations, backbone.num_values):
        raise RuntimeError(f"Unexpected relation map shape: {tuple(relation_maps.shape)}.")
    plasticity.eval()
    for parameter in plasticity.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    return (
        SemanticContextSystem(backbone, plasticity).to(device),
        base_values.detach().cpu().to(dtype=torch.long),
        relation_maps.detach().cpu().to(dtype=torch.long),
        config,
    )


@torch.no_grad()
def infer_candidate_state(
    system: SemanticContextSystem,
    episode: Episode,
    *,
    covariance_inverse: torch.Tensor,
    train_variants: tuple[int, ...],
    fast_capacity: float,
    seed: int,
    device: torch.device,
) -> tuple[FastModeState, dict[str, Any]]:
    if not train_variants:
        raise ValueError("Candidate inference requires at least one training variant.")
    dtype = next(system.plasticity.parameters()).dtype
    state = system.plasticity.initial_state(device=device, dtype=dtype)
    generator = random.Random(seed)
    reliable_gates: list[float] = []
    unreliable_gates: list[float] = []
    event_rows: list[dict[str, Any]] = []

    for index, event in enumerate(episode.events):
        variant = generator.choice(train_variants)
        entities = torch.tensor([event.entity], device=device, dtype=torch.long)
        relations = torch.zeros(1, device=device, dtype=torch.long)
        contexts = torch.zeros(1, device=device, dtype=torch.long)
        variants = torch.tensor([variant], device=device, dtype=torch.long)
        event_hidden, event_logits, _attention = system.predict(
            entities=entities,
            relations=relations,
            contexts=contexts,
            variants=variants,
            state=state,
        )
        (
            state,
            write_gate,
            source_hidden,
            value_embedding,
            _prior_agreement,
            _prior_confidence,
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
        direct_target = F.one_hot(
            torch.tensor(event.observed_value, device=device, dtype=torch.long),
            num_classes=system.backbone.num_values,
        ).to(dtype=target_probabilities.dtype)
        target_probabilities = torch.cat(
            [direct_target.unsqueeze(0), target_probabilities[1:]],
            dim=0,
        )
        relation_ids = torch.arange(
            system.backbone.num_relations,
            device=device,
            dtype=torch.long,
        )
        entity_ids = torch.full_like(relation_ids, event.entity)
        relation_hidden, relation_logits, _relation_attention = system.predict(
            entities=entity_ids,
            relations=relation_ids,
            contexts=torch.zeros_like(relation_ids),
            variants=torch.zeros_like(relation_ids),
            state=state,
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
        gate_value = float(write_gate.cpu())
        if event.source_is_reliable:
            reliable_gates.append(gate_value)
        else:
            unreliable_gates.append(gate_value)
        event_rows.append(
            {
                "index": index,
                "source": event.source,
                "phase": event.phase,
                "write_gate": gate_value,
                "mean_consequence_confidence": float(relation_confidence.mean().cpu()),
            }
        )
    if not reliable_gates or not unreliable_gates:
        raise RuntimeError("Candidate stream did not contain both reliable and unreliable evidence.")
    return state, {
        "reliable_write_gate": sum(reliable_gates) / float(len(reliable_gates)),
        "unreliable_write_gate": sum(unreliable_gates) / float(len(unreliable_gates)),
        "fast_norm": float(torch.linalg.vector_norm(state.fast).cpu()),
        "events": event_rows,
    }


def query_grid(
    *,
    entity_pool: tuple[int, ...],
    num_relations: int,
    num_variants: int,
    context: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if context not in (0, 1):
        raise ValueError(f"Context must be current (0) or historical (1), got {context}.")
    rows = [
        (entity, relation, context, variant)
        for entity in entity_pool
        for relation in range(num_relations)
        for variant in range(num_variants)
    ]
    return make_query_batch(rows, device=device)


@torch.no_grad()
def candidate_targets(
    system: SemanticContextSystem,
    *,
    state: FastModeState,
    entity_pool: tuple[int, ...],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, float]:
    queries = query_grid(
        entity_pool=entity_pool,
        num_relations=system.backbone.num_relations,
        num_variants=system.backbone.num_variants,
        context=0,
        device=device,
    )
    hidden, base_logits = system.backbone(*queries)
    candidate_logits, _attention = system.plasticity.read(hidden, base_logits, state)
    candidate = torch.softmax(candidate_logits, dim=1)
    base = torch.softmax(base_logits, dim=1)
    eps = torch.finfo(candidate.dtype).eps
    divergence = (
        candidate * (candidate.clamp_min(eps).log() - base.clamp_min(eps).log())
    ).sum(dim=1)
    return queries, candidate.detach(), float(divergence.mean().cpu())


def principal_basis(values: torch.Tensor, rank: int) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"PCA values must be a matrix with at least two rows, got {values.shape}.")
    positive_int("constraint_rank", rank)
    centered = values.detach().to(device="cpu")
    centered = centered.to(dtype=torch.float64)
    centered = centered - centered.mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    if not torch.isfinite(singular_values).all():
        raise FloatingPointError("Constraint PCA produced non-finite singular values.")
    available = int((singular_values > torch.finfo(singular_values.dtype).eps).sum().item())
    if available <= 0:
        raise RuntimeError("Constraint PCA has zero numerical rank.")
    kept = min(rank, available)
    return vh[:kept].transpose(0, 1).to(dtype=values.dtype, device=values.device)


def invariant_measurements(
    probabilities: torch.Tensor,
    hidden: torch.Tensor,
    relations: torch.Tensor,
    *,
    probability_basis: torch.Tensor,
    hidden_basis: torch.Tensor,
    num_relations: int,
) -> list[torch.Tensor]:
    if probabilities.ndim != 2 or hidden.ndim != 2:
        raise ValueError("Invariant measurements require probability and hidden matrices.")
    rows: list[torch.Tensor] = []
    for relation in range(num_relations):
        mask = relations == relation
        if int(mask.sum().item()) < 2:
            raise RuntimeError(f"Relation {relation} has too few protected measurements.")
        relation_probabilities = probabilities[mask]
        relation_hidden = hidden[mask]
        probability_centroid = relation_probabilities.mean(dim=0)
        hidden_centroid = relation_hidden.mean(dim=0)
        centered_hidden = relation_hidden - hidden_centroid.unsqueeze(0)
        projected_hidden = centered_hidden @ hidden_basis
        rows.extend((probability_centroid @ probability_basis).unbind())
        rows.extend((hidden_centroid @ hidden_basis).unbind())
        rows.extend(projected_hidden.square().mean(dim=0).unbind())
    if not rows:
        raise RuntimeError("No invariant measurement rows were constructed.")
    return rows


def bounded_restore_gradient(
    restore_gradient: torch.Tensor,
    tangent_gradient: torch.Tensor,
    *,
    strength: float,
    norm_ratio: float,
) -> tuple[torch.Tensor, float]:
    restore_norm = torch.linalg.vector_norm(restore_gradient)
    tangent_norm = torch.linalg.vector_norm(tangent_gradient)
    if float(restore_norm.detach().cpu()) <= 1e-12:
        return torch.zeros_like(restore_gradient), 0.0
    maximum = norm_ratio * tangent_norm
    coefficient = torch.minimum(
        restore_norm.new_tensor(strength),
        maximum / restore_norm.clamp_min(1e-12),
    )
    return coefficient * restore_gradient, float(coefficient.detach().cpu())


def consolidate_candidate(
    backbone: SemanticBackbone,
    *,
    candidate_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    candidate_probabilities: torch.Tensor,
    protected_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    protected_hidden_reference: torch.Tensor | None,
    protected_logits_reference: torch.Tensor | None,
    protected_group_ids: torch.Tensor | None,
    epochs: int,
    learning_rate: float,
    projection_damping: float,
    constraint_rank: int,
    restore_strength: float,
    restore_norm_ratio: float,
    geometry_restore_weight: float,
    candidate_focus_strength: float,
    max_gradient_norm: float,
    print_every: int,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Backbone consolidation has no trainable parameters.")
    if candidate_focus_strength < 0.0 or not math.isfinite(candidate_focus_strength):
        raise ValueError("Candidate focus strength must be non-negative and finite.")
    with torch.no_grad():
        if protected_hidden_reference is None and protected_logits_reference is None:
            protected_hidden_reference, protected_logits_reference = backbone(*protected_queries)
        elif protected_hidden_reference is None or protected_logits_reference is None:
            raise ValueError("Protected hidden and logit references must be supplied together.")
        protected_hidden_reference = protected_hidden_reference.detach().to(
            device=candidate_probabilities.device,
        )
        protected_logits_reference = protected_logits_reference.detach().to(
            device=candidate_probabilities.device,
        )
        reference_hidden_check, reference_logits_check = backbone(*protected_queries)
        if protected_hidden_reference.shape != reference_hidden_check.shape:
            raise ValueError(
                "Protected hidden reference shape does not match protected queries: "
                f"{protected_hidden_reference.shape} vs {reference_hidden_check.shape}."
            )
        if protected_logits_reference.shape != reference_logits_check.shape:
            raise ValueError(
                "Protected logit reference shape does not match protected queries: "
                f"{protected_logits_reference.shape} vs {reference_logits_check.shape}."
            )
        if protected_group_ids is None:
            protected_group_ids = torch.zeros(
                protected_logits_reference.shape[0],
                device=protected_logits_reference.device,
                dtype=torch.long,
            )
        else:
            protected_group_ids = protected_group_ids.to(
                device=protected_logits_reference.device,
                dtype=torch.long,
            )
        if protected_group_ids.shape != (protected_logits_reference.shape[0],):
            raise ValueError(
                "Protected group IDs must provide one group per protected query row."
            )
        unique_protected_groups = torch.unique(protected_group_ids, sorted=True)
        if unique_protected_groups.numel() <= 0:
            raise RuntimeError("Protected restore has no normalization blocks.")
        protected_probabilities_reference = torch.softmax(protected_logits_reference, dim=1)
        probability_basis = principal_basis(protected_probabilities_reference, constraint_rank)
        hidden_basis = principal_basis(protected_hidden_reference, constraint_rank)
        _candidate_hidden_reference, candidate_logits_reference = backbone(*candidate_queries)
        candidate_probabilities_reference = torch.softmax(candidate_logits_reference, dim=1)
        candidate_divergence = (
            candidate_probabilities
            * (
                candidate_probabilities.clamp_min(torch.finfo(candidate_probabilities.dtype).eps).log()
                - candidate_probabilities_reference.clamp_min(
                    torch.finfo(candidate_probabilities.dtype).eps
                ).log()
            )
        ).sum(dim=1)
        mean_divergence = candidate_divergence.mean()
        if float(mean_divergence.cpu()) <= 1e-12:
            raise RuntimeError("Semantic candidate does not differ from the current backbone.")
        candidate_transport = candidate_divergence / (
            candidate_divergence + mean_divergence
        )
        consolidation_probabilities = (
            candidate_transport.unsqueeze(1) * candidate_probabilities
            + (1.0 - candidate_transport).unsqueeze(1) * candidate_probabilities_reference
        )
        consolidation_probabilities = consolidation_probabilities / (
            consolidation_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        candidate_weights = 1.0 + candidate_focus_strength * (
            candidate_divergence / mean_divergence
        )
        candidate_weights = candidate_weights / candidate_weights.mean()
        normalized_weights = candidate_weights / candidate_weights.sum()
        candidate_effective_rows = torch.exp(
            -(normalized_weights * normalized_weights.clamp_min(1e-12).log()).sum()
        )
    trace: list[dict[str, float]] = []
    eps = torch.finfo(candidate_probabilities.dtype).eps
    protected_relations = protected_queries[1]

    for epoch in range(1, epochs + 1):
        _candidate_hidden, current_candidate_logits = backbone(*candidate_queries)
        current_protected_hidden, current_protected_logits = backbone(*protected_queries)
        current_candidate_log_probabilities = torch.log_softmax(current_candidate_logits, dim=1)
        candidate_cross_entropy_rows = -(
            consolidation_probabilities * current_candidate_log_probabilities
        ).sum(dim=1)
        candidate_entropy_rows = -(
            consolidation_probabilities
            * consolidation_probabilities.clamp_min(eps).log()
        ).sum(dim=1)
        candidate_loss = (
            candidate_weights * (candidate_cross_entropy_rows - candidate_entropy_rows)
        ).mean()
        current_protected_probabilities = torch.softmax(current_protected_logits, dim=1)
        protected_row_kl = (
            protected_probabilities_reference
            * (
                protected_probabilities_reference.clamp_min(eps).log()
                - torch.log_softmax(current_protected_logits, dim=1)
            )
        ).sum(dim=1)
        behavior_blocks: list[torch.Tensor] = []
        geometry_blocks: list[torch.Tensor] = []
        for group_id in unique_protected_groups:
            group_mask = protected_group_ids == group_id
            if not bool(group_mask.any()):
                raise RuntimeError(f"Protected restore group {int(group_id)} is empty.")
            behavior_blocks.append(protected_row_kl[group_mask].mean())
            geometry_blocks.append(
                F.mse_loss(
                    current_protected_hidden[group_mask],
                    protected_hidden_reference[group_mask],
                )
            )
        behavior_restore = torch.stack(behavior_blocks).mean()
        geometry_restore = torch.stack(geometry_blocks).mean()
        restore_loss = behavior_restore + geometry_restore_weight * geometry_restore
        raw_gradient = flat_autograd_gradient(
            candidate_loss,
            parameters,
            retain_graph=True,
            require_nonzero=True,
            label="semantic candidate",
        )
        measurement_losses = invariant_measurements(
            current_protected_probabilities,
            current_protected_hidden,
            protected_relations,
            probability_basis=probability_basis,
            hidden_basis=hidden_basis,
            num_relations=backbone.num_relations,
        )
        constraint_gradients = [
            flat_autograd_gradient(
                measurement,
                parameters,
                retain_graph=True,
                require_nonzero=False,
                label=f"protected invariant {index}",
            )
            for index, measurement in enumerate(measurement_losses)
        ]
        tangent_gradient, projection = project_gradient_away_from_constraints(
            raw_gradient=raw_gradient,
            constraint_gradients=constraint_gradients,
            damping=projection_damping,
            solver="gram",
            rank_tolerance=1e-5,
            plasticity_audit=False,
        )
        restore_gradient = flat_autograd_gradient(
            restore_loss,
            parameters,
            retain_graph=False,
            require_nonzero=False,
            label="bounded restore",
        )
        bounded_restore, restore_coefficient = bounded_restore_gradient(
            restore_gradient,
            tangent_gradient,
            strength=restore_strength,
            norm_ratio=restore_norm_ratio,
        )
        final_gradient = tangent_gradient + bounded_restore
        final_norm = torch.linalg.vector_norm(final_gradient)
        clip_scale = torch.minimum(
            final_norm.new_ones(()),
            final_norm.new_tensor(max_gradient_norm) / final_norm.clamp_min(1e-12),
        )
        final_gradient = final_gradient * clip_scale
        if not torch.isfinite(final_gradient).all():
            raise FloatingPointError(f"Consolidation gradient is non-finite at epoch {epoch}.")
        assign_flat_gradient(parameters, final_gradient)
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is None:
                    raise RuntimeError("A backbone parameter did not receive a consolidation gradient.")
                parameter.add_(parameter.grad, alpha=-learning_rate)
                parameter.grad = None
        row = {
            "epoch": float(epoch),
            "candidate_loss": float(candidate_loss.detach().cpu()),
            "behavior_restore": float(behavior_restore.detach().cpu()),
            "geometry_restore": float(geometry_restore.detach().cpu()),
            "raw_gradient_norm": projection["raw_grad_norm"],
            "tangent_gradient_norm": projection["projected_grad_norm"],
            "removed_fraction": projection["projection_removed_fraction"],
            "safe_gradient_fraction": projection["safe_grad_fraction"],
            "constraint_rows": projection["constraint_count"],
            "candidate_effective_rows": float(candidate_effective_rows.cpu()),
            "candidate_transport_mean": float(candidate_transport.mean().cpu()),
            "candidate_transport_max": float(candidate_transport.max().cpu()),
            "restore_coefficient": restore_coefficient,
            "final_gradient_norm": float(torch.linalg.vector_norm(final_gradient).detach().cpu()),
        }
        trace.append(row)
        if epoch == 1 or epoch == epochs or epoch % print_every == 0:
            print(
                f"consolidate epoch={epoch:4d} candidate={row['candidate_loss']:.5f} "
                f"restore={row['behavior_restore']:.5f}/{row['geometry_restore']:.5f} "
                f"removed={row['removed_fraction']:.3f} rows={row['constraint_rows']:.0f}"
            )
    return trace


@torch.no_grad()
def evaluate_backbone(
    backbone: SemanticBackbone,
    episode: Episode,
    *,
    base_values: torch.Tensor,
    relation_maps: torch.Tensor,
    entity_pool: tuple[int, ...],
    heldout_variants: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    if len(heldout_variants) < 2:
        raise ValueError("Evaluation requires at least two held-out paraphrase variants.")
    corrected = set(episode.corrected_entities)
    stable = [entity for entity in entity_pool if entity not in corrected]
    if not stable:
        raise RuntimeError("Backbone evaluation requires at least one stable entity.")
    truth = current_truth_values(
        base_values=base_values,
        episode=episode,
        corrections_active=True,
    )

    def accuracy(
        rows: list[tuple[int, int, int, int]],
        values: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        queries = make_query_batch(rows, device=device)
        targets = relational_targets(values.to(device), queries[1], relation_maps)
        _hidden, logits = backbone(*queries)
        score = float((logits.argmax(dim=1) == targets).to(torch.float32).mean().cpu())
        return score, logits

    direct_rows = [(entity, 0, 0, heldout_variants[0]) for entity in episode.corrected_entities]
    direct_values = truth[torch.tensor(episode.corrected_entities)]
    direct_accuracy, direct_logits = accuracy(direct_rows, direct_values)
    old_values = base_values[torch.tensor(episode.corrected_entities)].to(device)
    new_values = direct_values.to(device)
    row_ids = torch.arange(direct_logits.shape[0], device=device)
    correction_margin = float(
        (direct_logits[row_ids, new_values] - direct_logits[row_ids, old_values]).mean().cpu()
    )

    ripple_rows = [
        (entity, relation, 0, heldout_variants[0])
        for entity in episode.corrected_entities
        for relation in range(1, backbone.num_relations)
    ]
    ripple_values = torch.tensor(
        [int(truth[entity]) for entity in episode.corrected_entities for _ in range(1, backbone.num_relations)],
        dtype=torch.long,
    )
    ripple_accuracy, _ = accuracy(ripple_rows, ripple_values)

    paraphrase_rows = [
        (entity, relation, 0, variant)
        for entity in episode.corrected_entities
        for relation in range(backbone.num_relations)
        for variant in heldout_variants
    ]
    paraphrase_values = torch.tensor(
        [
            int(truth[entity])
            for entity in episode.corrected_entities
            for _relation in range(backbone.num_relations)
            for _variant in heldout_variants
        ],
        dtype=torch.long,
    )
    paraphrase_accuracy, _ = accuracy(paraphrase_rows, paraphrase_values)

    historical_rows = [
        (entity, relation, 1, heldout_variants[0])
        for entity in episode.corrected_entities
        for relation in range(backbone.num_relations)
    ]
    historical_values = torch.tensor(
        [
            int(base_values[entity])
            for entity in episode.corrected_entities
            for _relation in range(backbone.num_relations)
        ],
        dtype=torch.long,
    )
    historical_accuracy, _ = accuracy(historical_rows, historical_values)

    locality_rows = [
        (entity, relation, context, heldout_variants[0])
        for entity in stable
        for relation in range(backbone.num_relations)
        for context in range(2)
    ]
    locality_values = torch.tensor(
        [
            int(base_values[entity])
            for entity in stable
            for _relation in range(backbone.num_relations)
            for _context in range(2)
        ],
        dtype=torch.long,
    )
    locality_accuracy, _ = accuracy(locality_rows, locality_values)
    return {
        "direct_accuracy": direct_accuracy,
        "ripple_accuracy": ripple_accuracy,
        "paraphrase_accuracy": paraphrase_accuracy,
        "historical_accuracy": historical_accuracy,
        "locality_accuracy": locality_accuracy,
        "correction_margin": correction_margin,
    }


def linear_cka(reference: torch.Tensor, current: torch.Tensor) -> float:
    if reference.shape != current.shape or reference.ndim != 2:
        raise ValueError(f"CKA matrices must share a two-dimensional shape, got {reference.shape}/{current.shape}.")
    x = reference.detach().to(device="cpu").to(dtype=torch.float64)
    y = current.detach().to(device="cpu").to(dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = torch.linalg.vector_norm(x.transpose(0, 1) @ y).square()
    denominator = (
        torch.linalg.vector_norm(x.transpose(0, 1) @ x)
        * torch.linalg.vector_norm(y.transpose(0, 1) @ y)
    )
    if float(denominator) <= 1e-20:
        raise RuntimeError("CKA denominator is zero.")
    return float(cross / denominator)


def representation_report(reference: torch.Tensor, current: torch.Tensor) -> dict[str, float]:
    if reference.shape != current.shape:
        raise ValueError("Representation report tensors must share a shape.")
    x = reference.detach().to(device="cpu").to(dtype=torch.float64)
    y = current.detach().to(device="cpu").to(dtype=torch.float64)
    hidden_drift = torch.linalg.vector_norm(y - x) / torch.linalg.vector_norm(x).clamp_min(1e-12)
    x_differences = x.unsqueeze(1) - x.unsqueeze(0)
    y_differences = y.unsqueeze(1) - y.unsqueeze(0)
    x_distances = torch.sqrt(x_differences.square().sum(dim=2).clamp_min(0.0))
    y_distances = torch.sqrt(y_differences.square().sum(dim=2).clamp_min(0.0))
    pair_drift = (
        torch.linalg.vector_norm(y_distances - x_distances)
        / torch.linalg.vector_norm(x_distances).clamp_min(1e-12)
    )
    return {
        "cka": linear_cka(x, y),
        "hidden_drift": float(hidden_drift),
        "pair_drift": float(pair_drift),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list.")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows[1:]):
        raise RuntimeError("Metric rows do not share a schema.")
    return {key: sum(row[key] for row in rows) / float(len(rows)) for key in keys}


def minimum(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot find minima for an empty metric list.")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows[1:]):
        raise RuntimeError("Metric rows do not share a schema.")
    return {key: min(row[key] for row in rows) for key in keys}


def plot_results(
    *,
    before: dict[str, float],
    after: dict[str, float],
    controller: dict[str, float],
    geometry: dict[str, float],
    trace: list[dict[str, float]],
    output_path: Path,
) -> None:
    if not trace:
        raise ValueError("Plotting requires a consolidation trace.")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    behavior = [
        "direct_accuracy",
        "ripple_accuracy",
        "paraphrase_accuracy",
        "historical_accuracy",
        "locality_accuracy",
    ]
    x = list(range(len(behavior)))
    axes[0, 0].bar([value - 0.18 for value in x], [before[key] for key in behavior], width=0.36, label="before")
    axes[0, 0].bar([value + 0.18 for value in x], [after[key] for key in behavior], width=0.36, label="after")
    axes[0, 0].set_xticks(x, [key.replace("_accuracy", "") for key in behavior], rotation=22)
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].set_title("Durable backbone behavior")
    axes[0, 0].legend()

    axes[0, 1].bar(
        ["reliable", "unreliable"],
        [controller["reliable_write_gate"], controller["unreliable_write_gate"]],
        color=["#16a34a", "#dc2626"],
    )
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].set_title("Semantic evidence write gate")

    axes[1, 0].bar(
        ["protected CKA", "hidden drift", "pair drift"],
        [geometry["cka"], geometry["hidden_drift"], geometry["pair_drift"]],
        color=["#2563eb", "#ea580c", "#ca8a04"],
    )
    axes[1, 0].set_title("Protected representation geometry")

    epochs = [row["epoch"] for row in trace]
    axes[1, 1].plot(epochs, [row["candidate_loss"] for row in trace], label="candidate")
    axes[1, 1].plot(epochs, [row["geometry_restore"] for row in trace], label="geometry restore")
    axes[1, 1].plot(epochs, [row["removed_fraction"] for row in trace], label="projected fraction")
    axes[1, 1].set_title("Invariant-Tangent consolidation")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.semantic_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Semantic checkpoint root must be a dictionary.")
    template_system, base_values, relation_maps, config = construct_system(checkpoint, device=device)
    foundation_state = copy.deepcopy(template_system.backbone.state_dict())
    covariance_inverse = foundation_covariance_inverse(
        template_system.backbone,
        ridge=float(config["covariance_ridge"]),
        device=device,
    )
    test_pool = tuple(range(int(config["meta_train_entities"]), int(config["num_entities"])))
    train_variants = tuple(range(int(config["num_variants"]) - 2))
    heldout_variants = tuple(range(int(config["num_variants"]) - 2, int(config["num_variants"])))
    if len(test_pool) < 2 or not train_variants or len(heldout_variants) != 2:
        raise RuntimeError("Checkpoint split cannot support integration evaluation.")
    correction_entities = min(int(config["correction_entities"]), len(test_pool) - 1)

    print("TINY SEMANTIC -> INVARIANT-TANGENT BACKBONE INTEGRATION")
    print("=" * 144)
    print(
        f"device={device} episodes={args.episodes} consolidation_epochs={args.consolidation_epochs} "
        f"constraint_rank={args.constraint_rank} checkpoint={args.semantic_checkpoint}"
    )
    episode_records: list[dict[str, Any]] = []
    before_rows: list[dict[str, float]] = []
    after_rows: list[dict[str, float]] = []
    controller_rows: list[dict[str, float]] = []
    geometry_rows: list[dict[str, float]] = []
    first_backbone_state: dict[str, torch.Tensor] | None = None

    for episode_index in range(args.episodes):
        system, _base, _maps, _config = construct_system(checkpoint, device=device)
        system.backbone.load_state_dict(foundation_state, strict=True)
        for parameter in system.backbone.parameters():
            parameter.requires_grad_(True)
        episode = build_episode(
            entity_pool=test_pool,
            base_values=base_values,
            num_values=int(config["num_values"]),
            num_sources=int(config["num_sources"]),
            reliable_sources=int(config["reliable_sources"]),
            calibration_events_per_source=int(config["calibration_events_per_source"]),
            correction_entities=correction_entities,
            correction_rounds=int(config["correction_rounds"]),
            seed=args.seed * 200_003 + episode_index,
        )
        before = evaluate_backbone(
            system.backbone,
            episode,
            base_values=base_values,
            relation_maps=relation_maps,
            entity_pool=test_pool,
            heldout_variants=heldout_variants,
            device=device,
        )
        state, controller = infer_candidate_state(
            system,
            episode,
            covariance_inverse=covariance_inverse,
            train_variants=train_variants,
            fast_capacity=float(config["fast_capacity"]),
            seed=args.seed * 2_000_003 + episode_index,
            device=device,
        )
        queries, targets, candidate_divergence = candidate_targets(
            system,
            state=state,
            entity_pool=test_pool,
            device=device,
        )
        protected_queries = query_grid(
            entity_pool=test_pool,
            num_relations=system.backbone.num_relations,
            num_variants=system.backbone.num_variants,
            context=1,
            device=device,
        )
        with torch.no_grad():
            protected_reference, _reference_logits = system.backbone(*protected_queries)
        trace = consolidate_candidate(
            system.backbone,
            candidate_queries=queries,
            candidate_probabilities=targets,
            protected_queries=protected_queries,
            protected_hidden_reference=None,
            protected_logits_reference=None,
            protected_group_ids=None,
            epochs=args.consolidation_epochs,
            learning_rate=args.learning_rate,
            projection_damping=args.projection_damping,
            constraint_rank=args.constraint_rank,
            restore_strength=args.restore_strength,
            restore_norm_ratio=args.restore_norm_ratio,
            geometry_restore_weight=args.geometry_restore_weight,
            candidate_focus_strength=args.candidate_focus_strength,
            max_gradient_norm=args.max_gradient_norm,
            print_every=args.print_every,
        )
        after = evaluate_backbone(
            system.backbone,
            episode,
            base_values=base_values,
            relation_maps=relation_maps,
            entity_pool=test_pool,
            heldout_variants=heldout_variants,
            device=device,
        )
        with torch.no_grad():
            protected_current, _current_logits = system.backbone(*protected_queries)
        geometry = representation_report(protected_reference, protected_current)
        controller_metrics = {
            "reliable_write_gate": float(controller["reliable_write_gate"]),
            "unreliable_write_gate": float(controller["unreliable_write_gate"]),
            "fast_norm": float(controller["fast_norm"]),
            "candidate_divergence": candidate_divergence,
        }
        before_rows.append(before)
        after_rows.append(after)
        controller_rows.append(controller_metrics)
        geometry_rows.append(geometry)
        episode_records.append(
            {
                "index": episode_index,
                "episode": asdict(episode),
                "before": before,
                "after": after,
                "controller": controller,
                "candidate_divergence": candidate_divergence,
                "geometry": geometry,
                "consolidation": trace,
            }
        )
        if episode_index == 0:
            first_backbone_state = {
                key: value.detach().cpu().clone()
                for key, value in system.backbone.state_dict().items()
            }
        print(
            f"episode={episode_index + 1:02d} direct={before['direct_accuracy']:.3f}->{after['direct_accuracy']:.3f} "
            f"ripple={after['ripple_accuracy']:.3f} history={after['historical_accuracy']:.3f} "
            f"locality={after['locality_accuracy']:.3f} cka={geometry['cka']:.4f} "
            f"gate={controller_metrics['reliable_write_gate']:.3f}/{controller_metrics['unreliable_write_gate']:.3f}"
        )

    if first_backbone_state is None:
        raise RuntimeError("No integrated backbone checkpoint was produced.")
    before_final = aggregate(before_rows)
    after_final = aggregate(after_rows)
    controller_final = aggregate(controller_rows)
    geometry_final = aggregate(geometry_rows)
    after_minimum = minimum(after_rows)
    geometry_minimum = minimum(geometry_rows)
    validation = {
        "direct_replacement": after_final["direct_accuracy"] >= args.direct_accuracy_threshold,
        "ripple_propagation": after_final["ripple_accuracy"] >= args.ripple_accuracy_threshold,
        "paraphrase_generalization": after_final["paraphrase_accuracy"] >= args.paraphrase_accuracy_threshold,
        "historical_retention": after_final["historical_accuracy"] >= args.historical_accuracy_threshold,
        "unrelated_locality": after_final["locality_accuracy"] >= args.locality_accuracy_threshold,
        "positive_correction_margin": after_final["correction_margin"] >= args.correction_margin_threshold,
        "source_discrimination": controller_final["reliable_write_gate"] > controller_final["unreliable_write_gate"],
        "protected_geometry": geometry_final["cka"] >= args.protected_cka_threshold,
        "durable_without_fast_state": after_final["direct_accuracy"] > before_final["direct_accuracy"],
    }

    print("\nFINAL SEMANTIC INVARIANT-TANGENT STATE")
    print("-" * 144)
    print(
        f"direct={before_final['direct_accuracy']:.4f}->{after_final['direct_accuracy']:.4f} "
        f"ripple={after_final['ripple_accuracy']:.4f} paraphrase={after_final['paraphrase_accuracy']:.4f} "
        f"history={after_final['historical_accuracy']:.4f} locality={after_final['locality_accuracy']:.4f} "
        f"margin={before_final['correction_margin']:.4f}->{after_final['correction_margin']:.4f}"
    )
    print(
        f"gate reliable/unreliable={controller_final['reliable_write_gate']:.4f}/"
        f"{controller_final['unreliable_write_gate']:.4f} candidate_kl={controller_final['candidate_divergence']:.5f} "
        f"protected_cka={geometry_final['cka']:.4f} hidden_drift={geometry_final['hidden_drift']:.4f} "
        f"pair_drift={geometry_final['pair_drift']:.4f}"
    )
    print(
        f"worst_episode direct={after_minimum['direct_accuracy']:.4f} "
        f"ripple={after_minimum['ripple_accuracy']:.4f} "
        f"history={after_minimum['historical_accuracy']:.4f} "
        f"locality={after_minimum['locality_accuracy']:.4f} "
        f"protected_cka={geometry_minimum['cka']:.4f}"
    )
    print(f"validation={validation}")

    json_path = args.output_dir / "semantic_invariant_tangent.json"
    plot_path = args.output_dir / "semantic_invariant_tangent.png"
    checkpoint_path = args.output_dir / "semantic_invariant_tangent.pt"
    output = {
        "question": (
            "Can unlabeled semantic evidence and consequence reasoning drive a durable backbone update "
            "through Invariant-Tangent constraints, with the temporary fast state removed at evaluation?"
        ),
        "scope": (
            "Tiny synthetic semantic world. Source reliability and correction identity are evaluation-only; "
            "the backbone learns exclusively from the semantic mechanism's candidate distribution."
        ),
        "mechanism": {
            "candidate": "recurrent evidence + consequence prediction + context-conditioned fast modes",
            "durable_update": "candidate distillation into actual backbone parameters",
            "tangent": "projection away from historical behavior and geometry Jacobian rows",
            "restore": "norm-bounded correction toward historical behavior and hidden states",
            "evaluation": "backbone only, with temporary fast state removed",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "semantic_checkpoint_config": config,
        "before": before_final,
        "after": after_final,
        "controller": controller_final,
        "geometry": geometry_final,
        "worst_episode": {
            "behavior": after_minimum,
            "geometry": geometry_minimum,
        },
        "validation": validation,
        "episodes": episode_records,
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    plot_results(
        before=before_final,
        after=after_final,
        controller=controller_final,
        geometry=geometry_final,
        trace=episode_records[0]["consolidation"],
        output_path=plot_path,
    )
    torch.save(
        {
            "format": "semantic_invariant_tangent_v1",
            "backbone": first_backbone_state,
            "source_checkpoint": str(args.semantic_checkpoint),
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
        "--semantic-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/analysis/gco-tiny-semantic-invariant-tangent-seed0"),
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--consolidation-epochs", type=int, default=140)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--projection-damping", type=float, default=1e-4)
    parser.add_argument("--constraint-rank", type=int, default=8)
    parser.add_argument("--restore-strength", type=float, default=0.3)
    parser.add_argument("--restore-norm-ratio", type=float, default=0.75)
    parser.add_argument("--geometry-restore-weight", type=float, default=10.0)
    parser.add_argument("--candidate-focus-strength", type=float, default=4.0)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--direct-accuracy-threshold", type=float, default=0.75)
    parser.add_argument("--ripple-accuracy-threshold", type=float, default=0.70)
    parser.add_argument("--paraphrase-accuracy-threshold", type=float, default=0.70)
    parser.add_argument("--historical-accuracy-threshold", type=float, default=0.90)
    parser.add_argument("--locality-accuracy-threshold", type=float, default=0.95)
    parser.add_argument("--protected-cka-threshold", type=float, default=0.95)
    parser.add_argument("--correction-margin-threshold", type=float, default=0.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
