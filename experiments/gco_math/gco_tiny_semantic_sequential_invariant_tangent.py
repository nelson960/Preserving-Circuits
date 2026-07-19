"""Run cumulative semantic Invariant-Tangent updates on one changing backbone.

Unlike the independent integration test, this experiment never restores the
foundation weights between stages.  Each unlabeled evidence stream produces a
temporary semantic candidate; that candidate is consolidated into the same
backbone while historical behavior and previously committed current behavior
define a moving protected manifold.  The Jacobian basis has fixed rank even as
the number of committed corrections grows.

Source reliability, correction identity, and world truth are used only to
generate the controlled stream and evaluate the result.  They are not inputs
to the semantic controller or the backbone consolidation loss.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.gco_math.gco_tiny_semantic_context_plasticity import (
    Episode,
    ReportEvent,
    alternative_value,
    foundation_covariance_inverse,
    make_query_batch,
    relational_targets,
    seed_everything,
)
from experiments.gco_math.gco_tiny_semantic_invariant_tangent_integration import (
    aggregate,
    candidate_targets,
    checkpoint_config,
    consolidate_candidate,
    construct_system,
    infer_candidate_state,
    minimum,
    probability,
    query_grid,
    representation_report,
)
from experiments.real_book_common import resolve_device


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def validate_args(args: argparse.Namespace) -> None:
    if not args.semantic_checkpoint.is_file():
        raise FileNotFoundError(f"Semantic checkpoint does not exist: {args.semantic_checkpoint}")
    for name in ("stages", "consolidation_epochs", "constraint_rank", "print_every"):
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


def build_stage_episode(
    *,
    entity_pool: tuple[int, ...],
    corrected_entity: int,
    current_truth: torch.Tensor,
    num_values: int,
    num_sources: int,
    reliable_sources: int,
    calibration_events_per_source: int,
    correction_rounds: int,
    seed: int,
) -> Episode:
    if corrected_entity not in entity_pool:
        raise ValueError(f"Corrected entity {corrected_entity} is outside the stage pool.")
    if current_truth.ndim != 1:
        raise ValueError("Current truth must be a vector.")
    generator = random.Random(seed)
    reliable_ids = set(generator.sample(range(num_sources), reliable_sources))
    reliability = tuple(source in reliable_ids for source in range(num_sources))
    old_value = int(current_truth[corrected_entity])
    new_value = alternative_value(old_value, num_values=num_values, generator=generator)
    calibration_pool = [entity for entity in entity_pool if entity != corrected_entity]
    if not calibration_pool:
        raise RuntimeError("Sequential calibration requires a non-corrected entity.")
    calibration: list[ReportEvent] = []
    for source in range(num_sources):
        for _index in range(calibration_events_per_source):
            entity = generator.choice(calibration_pool)
            truth = int(current_truth[entity])
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
    correction: list[ReportEvent] = []
    for _round in range(correction_rounds):
        round_rows = [
            ReportEvent(
                entity=corrected_entity,
                observed_value=new_value if reliability[source] else old_value,
                source=source,
                source_is_reliable=reliability[source],
                phase="correction",
            )
            for source in range(num_sources)
        ]
        generator.shuffle(round_rows)
        correction.extend(round_rows)
    return Episode(
        events=tuple(calibration + correction),
        corrected_entities=(corrected_entity,),
        corrected_values=(new_value,),
        source_reliability=reliability,
    )


def concatenate_queries(
    query_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not query_groups:
        raise ValueError("At least one protected query group is required.")
    return tuple(
        torch.cat([group[column] for group in query_groups], dim=0)
        for column in range(4)
    )  # type: ignore[return-value]


def protected_query_set(
    *,
    all_entities: tuple[int, ...],
    committed_entities: tuple[int, ...],
    num_relations: int,
    num_variants: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = [
        query_grid(
            entity_pool=all_entities,
            num_relations=num_relations,
            num_variants=num_variants,
            context=1,
            device=device,
        )
    ]
    if committed_entities:
        groups.append(
            query_grid(
                entity_pool=committed_entities,
                num_relations=num_relations,
                num_variants=num_variants,
                context=0,
                device=device,
            )
        )
    return concatenate_queries(groups)


@torch.no_grad()
def query_accuracy(
    backbone: torch.nn.Module,
    *,
    rows: list[tuple[int, int, int, int]],
    values: torch.Tensor,
    relation_maps: torch.Tensor,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    queries = make_query_batch(rows, device=device)
    targets = relational_targets(values.to(device), queries[1], relation_maps)
    hidden, logits = backbone(*queries)
    accuracy = float((logits.argmax(dim=1) == targets).to(torch.float32).mean().cpu())
    return accuracy, logits, hidden


@torch.no_grad()
def evaluate_cumulative(
    backbone: torch.nn.Module,
    *,
    original_truth: torch.Tensor,
    current_truth: torch.Tensor,
    relation_maps: torch.Tensor,
    all_entities: tuple[int, ...],
    committed_entities: tuple[int, ...],
    newest_entity: int,
    num_relations: int,
    heldout_variants: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    if not committed_entities or newest_entity not in committed_entities:
        raise ValueError("Cumulative evaluation requires the newest committed entity.")
    uncommitted = tuple(entity for entity in all_entities if entity not in set(committed_entities))
    if not uncommitted:
        raise RuntimeError("Cumulative evaluation requires at least one locality entity.")
    canonical_variant = heldout_variants[0]

    direct_rows = [(entity, 0, 0, canonical_variant) for entity in committed_entities]
    direct_values = current_truth[torch.tensor(committed_entities)]
    direct_accuracy, direct_logits, _ = query_accuracy(
        backbone,
        rows=direct_rows,
        values=direct_values,
        relation_maps=relation_maps,
        device=device,
    )
    direct_relations = torch.zeros(len(committed_entities), device=device, dtype=torch.long)
    new_targets = relational_targets(direct_values.to(device), direct_relations, relation_maps)
    old_values = original_truth[torch.tensor(committed_entities)].to(device)
    old_targets = relational_targets(old_values, direct_relations, relation_maps)
    indices = torch.arange(len(committed_entities), device=device)
    correction_margin = float(
        (direct_logits[indices, new_targets] - direct_logits[indices, old_targets]).mean().cpu()
    )

    newest_rows = [(newest_entity, 0, 0, canonical_variant)]
    newest_values = current_truth[torch.tensor([newest_entity])]
    newest_accuracy, _newest_logits, _ = query_accuracy(
        backbone,
        rows=newest_rows,
        values=newest_values,
        relation_maps=relation_maps,
        device=device,
    )

    ripple_rows = [
        (entity, relation, 0, canonical_variant)
        for entity in committed_entities
        for relation in range(1, num_relations)
    ]
    ripple_values = torch.tensor(
        [int(current_truth[entity]) for entity in committed_entities for _ in range(1, num_relations)],
        dtype=torch.long,
    )
    ripple_accuracy, _ripple_logits, _ = query_accuracy(
        backbone,
        rows=ripple_rows,
        values=ripple_values,
        relation_maps=relation_maps,
        device=device,
    )

    paraphrase_rows = [
        (entity, relation, 0, variant)
        for entity in committed_entities
        for relation in range(num_relations)
        for variant in heldout_variants
    ]
    paraphrase_values = torch.tensor(
        [
            int(current_truth[entity])
            for entity in committed_entities
            for _relation in range(num_relations)
            for _variant in heldout_variants
        ],
        dtype=torch.long,
    )
    paraphrase_accuracy, _paraphrase_logits, _ = query_accuracy(
        backbone,
        rows=paraphrase_rows,
        values=paraphrase_values,
        relation_maps=relation_maps,
        device=device,
    )

    historical_rows = [
        (entity, relation, 1, canonical_variant)
        for entity in committed_entities
        for relation in range(num_relations)
    ]
    historical_values = torch.tensor(
        [int(original_truth[entity]) for entity in committed_entities for _ in range(num_relations)],
        dtype=torch.long,
    )
    historical_accuracy, _historical_logits, _ = query_accuracy(
        backbone,
        rows=historical_rows,
        values=historical_values,
        relation_maps=relation_maps,
        device=device,
    )

    locality_rows = [
        (entity, relation, context, canonical_variant)
        for entity in uncommitted
        for relation in range(num_relations)
        for context in range(2)
    ]
    locality_values = torch.tensor(
        [
            int(original_truth[entity])
            for entity in uncommitted
            for _relation in range(num_relations)
            for _context in range(2)
        ],
        dtype=torch.long,
    )
    locality_accuracy, _locality_logits, _ = query_accuracy(
        backbone,
        rows=locality_rows,
        values=locality_values,
        relation_maps=relation_maps,
        device=device,
    )
    return {
        "newest_direct_accuracy": newest_accuracy,
        "committed_direct_accuracy": direct_accuracy,
        "ripple_accuracy": ripple_accuracy,
        "paraphrase_accuracy": paraphrase_accuracy,
        "historical_accuracy": historical_accuracy,
        "locality_accuracy": locality_accuracy,
        "correction_margin": correction_margin,
    }


def plot_results(stages: list[dict[str, Any]], output_path: Path) -> None:
    if not stages:
        raise ValueError("Sequential plotting requires stage records.")
    x = [row["stage"] for row in stages]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for key, label in (
        ("committed_direct_accuracy", "committed direct"),
        ("ripple_accuracy", "ripple"),
        ("historical_accuracy", "history"),
        ("locality_accuracy", "locality"),
    ):
        axes[0, 0].plot(x, [row["behavior"][key] for row in stages], marker="o", label=label)
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].set_title("Accumulated backbone behavior")
    axes[0, 0].set_xlabel("continual stage")
    axes[0, 0].legend()

    axes[0, 1].plot(x, [row["geometry"]["cka"] for row in stages], marker="o", label="protected CKA")
    axes[0, 1].plot(x, [row["geometry"]["hidden_drift"] for row in stages], marker="o", label="hidden drift")
    axes[0, 1].plot(x, [row["geometry"]["pair_drift"] for row in stages], marker="o", label="pair drift")
    axes[0, 1].set_title("Moving protected manifold")
    axes[0, 1].set_xlabel("continual stage")
    axes[0, 1].legend()

    axes[1, 0].plot(x, [row["controller"]["reliable_write_gate"] for row in stages], marker="o", label="reliable")
    axes[1, 0].plot(x, [row["controller"]["unreliable_write_gate"] for row in stages], marker="o", label="unreliable")
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].set_title("Semantic evidence selection")
    axes[1, 0].set_xlabel("continual stage")
    axes[1, 0].legend()

    axes[1, 1].plot(x, [row["plasticity"]["safe_gradient_fraction"] for row in stages], marker="o", label="safe gradient")
    axes[1, 1].plot(x, [row["plasticity"]["removed_fraction"] for row in stages], marker="o", label="removed")
    axes[1, 1].plot(x, [row["plasticity"]["constraint_rows"] / stages[0]["plasticity"]["constraint_rows"] for row in stages], marker="o", label="row ratio")
    axes[1, 1].set_title("Fixed-rank update plasticity")
    axes[1, 1].set_xlabel("continual stage")
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
    config = checkpoint_config(checkpoint)
    system, original_truth, relation_maps, _config = construct_system(checkpoint, device=device)
    for parameter in system.backbone.parameters():
        parameter.requires_grad_(True)
    all_entities = tuple(range(int(config["num_entities"])))
    test_entities = list(range(int(config["meta_train_entities"]), int(config["num_entities"])))
    if args.stages > len(test_entities):
        raise ValueError(
            f"Requested {args.stages} unique held-out stages, but checkpoint provides {len(test_entities)}."
        )
    sequence_generator = random.Random(args.seed + 91_337)
    sequence_generator.shuffle(test_entities)
    stage_entities = tuple(test_entities[: args.stages])
    train_variants = tuple(range(int(config["num_variants"]) - 2))
    heldout_variants = tuple(range(int(config["num_variants"]) - 2, int(config["num_variants"])))
    if not train_variants or len(heldout_variants) != 2:
        raise RuntimeError("Checkpoint variant split is invalid.")
    current_truth = original_truth.clone()
    committed: list[int] = []

    foundation_history_queries = query_grid(
        entity_pool=all_entities,
        num_relations=int(config["num_relations"]),
        num_variants=int(config["num_variants"]),
        context=1,
        device=device,
    )
    with torch.no_grad():
        foundation_history_hidden, foundation_history_logits = system.backbone(*foundation_history_queries)
        foundation_history_hidden = foundation_history_hidden.detach()
        foundation_history_logits = foundation_history_logits.detach()

    print("TINY SEQUENTIAL SEMANTIC INVARIANT-TANGENT CL")
    print("=" * 152)
    print(
        f"device={device} stages={args.stages} epochs_per_stage={args.consolidation_epochs} "
        f"constraint_rank={args.constraint_rank} unique_entities={stage_entities}"
    )
    stage_records: list[dict[str, Any]] = []
    constraint_counts: list[int] = []
    committed_hidden_references: dict[int, torch.Tensor] = {}
    committed_logit_references: dict[int, torch.Tensor] = {}

    for stage_index, entity in enumerate(stage_entities, start=1):
        episode = build_stage_episode(
            entity_pool=all_entities,
            corrected_entity=entity,
            current_truth=current_truth,
            num_values=int(config["num_values"]),
            num_sources=int(config["num_sources"]),
            reliable_sources=int(config["reliable_sources"]),
            calibration_events_per_source=int(config["calibration_events_per_source"]),
            correction_rounds=int(config["correction_rounds"]),
            seed=args.seed * 300_007 + stage_index,
        )
        covariance_inverse = foundation_covariance_inverse(
            system.backbone,
            ridge=float(config["covariance_ridge"]),
            device=device,
        )
        candidate_state, controller_report = infer_candidate_state(
            system,
            episode,
            covariance_inverse=covariance_inverse,
            train_variants=train_variants,
            fast_capacity=float(config["fast_capacity"]),
            seed=args.seed * 3_000_017 + stage_index,
            device=device,
        )
        candidate_queries, targets, candidate_divergence = candidate_targets(
            system,
            state=candidate_state,
            entity_pool=all_entities,
            device=device,
        )
        protected_queries = protected_query_set(
            all_entities=all_entities,
            committed_entities=tuple(committed),
            num_relations=int(config["num_relations"]),
            num_variants=int(config["num_variants"]),
            device=device,
        )
        protected_hidden_reference = torch.cat(
            [foundation_history_hidden]
            + [committed_hidden_references[committed_entity] for committed_entity in committed],
            dim=0,
        )
        protected_logits_reference = torch.cat(
            [foundation_history_logits]
            + [committed_logit_references[committed_entity] for committed_entity in committed],
            dim=0,
        )
        protected_group_ids = torch.cat(
            [
                torch.zeros(
                    foundation_history_hidden.shape[0],
                    device=device,
                    dtype=torch.long,
                )
            ]
            + (
                [
                    torch.ones(
                        sum(
                            committed_hidden_references[committed_entity].shape[0]
                            for committed_entity in committed
                        ),
                        device=device,
                        dtype=torch.long,
                    )
                ]
                if committed
                else []
            ),
            dim=0,
        )
        trace = consolidate_candidate(
            system.backbone,
            candidate_queries=candidate_queries,
            candidate_probabilities=targets,
            protected_queries=protected_queries,
            protected_hidden_reference=protected_hidden_reference,
            protected_logits_reference=protected_logits_reference,
            protected_group_ids=protected_group_ids,
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
        current_truth[entity] = episode.corrected_values[0]
        committed.append(entity)
        committed_queries = query_grid(
            entity_pool=(entity,),
            num_relations=int(config["num_relations"]),
            num_variants=int(config["num_variants"]),
            context=0,
            device=device,
        )
        with torch.no_grad():
            committed_hidden, committed_logits = system.backbone(*committed_queries)
        committed_hidden_references[entity] = committed_hidden.detach()
        committed_logit_references[entity] = committed_logits.detach()
        behavior = evaluate_cumulative(
            system.backbone,
            original_truth=original_truth,
            current_truth=current_truth,
            relation_maps=relation_maps,
            all_entities=all_entities,
            committed_entities=tuple(committed),
            newest_entity=entity,
            num_relations=int(config["num_relations"]),
            heldout_variants=heldout_variants,
            device=device,
        )
        with torch.no_grad():
            protected_after, _protected_logits = system.backbone(*protected_queries)
        geometry = representation_report(protected_hidden_reference, protected_after)
        final_update = trace[-1]
        constraint_count = int(final_update["constraint_rows"])
        constraint_counts.append(constraint_count)
        controller = {
            "reliable_write_gate": float(controller_report["reliable_write_gate"]),
            "unreliable_write_gate": float(controller_report["unreliable_write_gate"]),
            "fast_norm": float(controller_report["fast_norm"]),
            "candidate_divergence": candidate_divergence,
        }
        plasticity = {
            "safe_gradient_fraction": final_update["safe_gradient_fraction"],
            "removed_fraction": final_update["removed_fraction"],
            "constraint_rows": final_update["constraint_rows"],
            "candidate_loss": final_update["candidate_loss"],
        }
        stage_records.append(
            {
                "stage": stage_index,
                "entity": entity,
                "episode": asdict(episode),
                "committed_entities": list(committed),
                "behavior": behavior,
                "controller": controller,
                "geometry": geometry,
                "plasticity": plasticity,
                "consolidation": trace,
            }
        )
        print(
            f"stage={stage_index:02d} entity={entity:02d} new={behavior['newest_direct_accuracy']:.3f} "
            f"committed={behavior['committed_direct_accuracy']:.3f} ripple={behavior['ripple_accuracy']:.3f} "
            f"history={behavior['historical_accuracy']:.3f} locality={behavior['locality_accuracy']:.3f} "
            f"cka={geometry['cka']:.4f} safe={plasticity['safe_gradient_fraction']:.3f} "
            f"gate={controller['reliable_write_gate']:.3f}/{controller['unreliable_write_gate']:.3f}"
        )

    behavior_rows = [row["behavior"] for row in stage_records]
    geometry_rows = [row["geometry"] for row in stage_records]
    controller_rows = [row["controller"] for row in stage_records]
    final_behavior = behavior_rows[-1]
    worst_behavior = minimum(behavior_rows)
    worst_geometry = minimum(geometry_rows)
    controller_mean = aggregate(controller_rows)
    with torch.no_grad():
        final_history_hidden, _final_history_logits = system.backbone(*foundation_history_queries)
    original_history_geometry = representation_report(
        foundation_history_hidden,
        final_history_hidden,
    )
    fixed_constraint_rows = len(set(constraint_counts)) == 1
    validation = {
        "every_new_correction_learned": worst_behavior["newest_direct_accuracy"] >= args.direct_accuracy_threshold,
        "committed_corrections_retained": final_behavior["committed_direct_accuracy"] >= args.direct_accuracy_threshold,
        "ripple_propagation": final_behavior["ripple_accuracy"] >= args.ripple_accuracy_threshold,
        "paraphrase_generalization": final_behavior["paraphrase_accuracy"] >= args.paraphrase_accuracy_threshold,
        "historical_retention": final_behavior["historical_accuracy"] >= args.historical_accuracy_threshold,
        "unrelated_locality": final_behavior["locality_accuracy"] >= args.locality_accuracy_threshold,
        "positive_correction_margin": final_behavior["correction_margin"] >= args.correction_margin_threshold,
        "source_discrimination": all(
            row["controller"]["reliable_write_gate"] > row["controller"]["unreliable_write_gate"]
            for row in stage_records
        ),
        "moving_protected_geometry": worst_geometry["cka"] >= args.protected_cka_threshold,
        "fixed_constraint_row_count": fixed_constraint_rows,
    }

    print("\nFINAL SEQUENTIAL SEMANTIC INVARIANT-TANGENT STATE")
    print("-" * 152)
    print(
        f"stages={args.stages} committed={final_behavior['committed_direct_accuracy']:.4f} "
        f"ripple={final_behavior['ripple_accuracy']:.4f} paraphrase={final_behavior['paraphrase_accuracy']:.4f} "
        f"history={final_behavior['historical_accuracy']:.4f} locality={final_behavior['locality_accuracy']:.4f} "
        f"margin={final_behavior['correction_margin']:.4f}"
    )
    print(
        f"worst_new={worst_behavior['newest_direct_accuracy']:.4f} "
        f"worst_committed={worst_behavior['committed_direct_accuracy']:.4f} "
        f"worst_stage_cka={worst_geometry['cka']:.4f} "
        f"original_history_cka={original_history_geometry['cka']:.4f} "
        f"constraint_rows={constraint_counts[0]} fixed={fixed_constraint_rows} "
        f"gate={controller_mean['reliable_write_gate']:.4f}/{controller_mean['unreliable_write_gate']:.4f}"
    )
    print(f"validation={validation}")

    json_path = args.output_dir / "sequential_semantic_invariant_tangent.json"
    plot_path = args.output_dir / "sequential_semantic_invariant_tangent.png"
    checkpoint_path = args.output_dir / "sequential_semantic_invariant_tangent.pt"
    output = {
        "question": (
            "Can semantic evidence selection and Invariant-Tangent consolidation accumulate multiple "
            "durable corrections in one backbone without growing the Jacobian row count?"
        ),
        "scope": (
            "Eight or fewer unique held-out entities in a tiny synthetic semantic world. The same backbone "
            "changes at every stage; temporary fast state is discarded after each candidate is consolidated."
        ),
        "mechanism": {
            "candidate": "semantic recurrent evidence, consequence prediction, and contextual fast modes",
            "commit": "candidate distribution distilled into the same backbone at every stage",
            "dynamic_manifold": "original historical context plus all previously committed current entities",
            "bounded_basis": "fixed-rank behavior/centroid/variance Jacobian measurements",
            "restore": "norm-bounded return toward the moving protected manifold",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "semantic_checkpoint_config": config,
        "stage_entities": stage_entities,
        "final_behavior": final_behavior,
        "worst_behavior": worst_behavior,
        "worst_stage_geometry": worst_geometry,
        "original_history_geometry": original_history_geometry,
        "controller_mean": controller_mean,
        "constraint_rows": constraint_counts,
        "validation": validation,
        "stages": stage_records,
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    plot_results(stage_records, plot_path)
    torch.save(
        {
            "format": "sequential_semantic_invariant_tangent_v1",
            "backbone": {
                key: value.detach().cpu().clone()
                for key, value in system.backbone.state_dict().items()
            },
            "current_truth": current_truth,
            "committed_entities": tuple(committed),
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
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/analysis/gco-tiny-sequential-semantic-invariant-tangent-seed0"),
    )
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--consolidation-epochs", type=int, default=140)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--projection-damping", type=float, default=1e-4)
    parser.add_argument("--constraint-rank", type=int, default=8)
    parser.add_argument("--restore-strength", type=float, default=0.8)
    parser.add_argument("--restore-norm-ratio", type=float, default=1.0)
    parser.add_argument("--geometry-restore-weight", type=float, default=20.0)
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
