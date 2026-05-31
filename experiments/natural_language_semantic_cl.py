"""Natural-language sentence-level semantic continual-learning benchmark.

This benchmark keeps the write-control mechanism from
semantic_geometry_write_reasoner.py, but replaces the hand-coded stream with a
JSON sentence stream. The first version uses gold extraction: every sentence is
paired with an explicit subject/relation/object annotation. That isolates the
continual-learning/write-control question from parser errors.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from semantic_geometry_write_reasoner import (
    ACTIONS,
    SOURCE_NAMES,
    SOURCE_TO_INDEX,
    CandidateWritePolicy,
    ConsolidationStats,
    SemanticEvent,
    World,
    active_composition_events,
    build_candidates,
    candidate_feature_tensor,
    code_norm_loss,
    commit_belief_from_choice,
    commit_truth,
    embedding_separation_loss,
    evaluate_events,
    make_model,
    run_blind_adamw,
    set_seed,
    should_try_consolidation,
    summarize,
    teacher_candidate_index,
    train_blind_event,
    try_dynamic_consolidation,
    update_slot_bookkeeping,
    warmup_query_batch,
)


def canonical_token(value: str) -> str:
    token = value.strip().lower().replace(" ", "_")
    if not token:
        raise ValueError("Cannot canonicalize an empty token.")
    return token


def rotated(items: list[str], amount: int) -> list[str]:
    if not items:
        raise ValueError("Cannot rotate an empty list.")
    amount = amount % len(items)
    return items[amount:] + items[:amount]


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset JSON does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "extraction_mode", "variables", "bindings", "events"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Dataset JSON missing required keys: {missing}")
    if data["extraction_mode"] != "gold":
        raise ValueError(f"Only extraction_mode='gold' is supported, got {data['extraction_mode']!r}.")
    if not isinstance(data["events"], list) or not data["events"]:
        raise ValueError("Dataset JSON must contain a non-empty events list.")
    return data


def format_value(template: str, bindings: dict[str, str]) -> str:
    try:
        return template.format(**bindings)
    except KeyError as exc:
        raise KeyError(f"Unknown placeholder {exc} in template {template!r}.") from exc


def resolve_bindings(data: dict[str, Any], seed: int) -> dict[str, str]:
    variables = data["variables"]
    bindings_spec = data["bindings"]
    if not isinstance(bindings_spec, dict) or not bindings_spec:
        raise ValueError("Dataset bindings must be a non-empty object.")

    rotated_lists: dict[str, list[str]] = {}
    for name, value in variables.items():
        if isinstance(value, list):
            rotated_lists[name] = rotated([str(item) for item in value], seed)

    bindings: dict[str, str] = {}
    unresolved = dict(bindings_spec)
    while unresolved:
        progressed = False
        for name, spec in list(unresolved.items()):
            if not isinstance(spec, dict):
                raise ValueError(f"Binding {name!r} must be an object.")
            if "from" in spec:
                source_name = str(spec["from"])
                if source_name not in rotated_lists:
                    raise KeyError(f"Binding {name!r} references unknown list variable {source_name!r}.")
                index = int(spec["index"])
                source = rotated_lists[source_name]
                if index < 0 or index >= len(source):
                    raise IndexError(f"Binding {name!r} index {index} outside variable {source_name!r}.")
                bindings[name] = source[index]
                unresolved.pop(name)
                progressed = True
            elif "lookup" in spec:
                lookup_name = str(spec["lookup"])
                key_name = str(spec["key"])
                if key_name not in bindings:
                    continue
                lookup = variables.get(lookup_name)
                if not isinstance(lookup, dict):
                    raise KeyError(f"Binding {name!r} references unknown lookup {lookup_name!r}.")
                key_value = bindings[key_name]
                if key_value not in lookup:
                    raise KeyError(f"Lookup {lookup_name!r} has no key {key_value!r}.")
                bindings[name] = str(lookup[key_value])
                unresolved.pop(name)
                progressed = True
            else:
                raise ValueError(f"Binding {name!r} must use either 'from' or 'lookup'.")
        if not progressed:
            raise RuntimeError(f"Could not resolve dataset bindings: {sorted(unresolved)}")
    return bindings


def sentence_templates(record: dict[str, Any]) -> list[str]:
    if "sentences" in record:
        sentences = record["sentences"]
        if not isinstance(sentences, list) or not sentences:
            raise ValueError(f"Event {record.get('name', '<unknown>')!r} has invalid non-empty sentences list.")
        return [str(sentence) for sentence in sentences]
    if "sentence" in record:
        return [str(record["sentence"])]
    raise ValueError(f"Event {record.get('name', '<unknown>')!r} must contain sentence or sentences.")


def select_sentence_template(record: dict[str, Any], seed: int, timestamp: int, mode: str) -> str:
    templates = sentence_templates(record)
    if mode == "first":
        return templates[0]
    if mode == "seed":
        return templates[(seed + timestamp) % len(templates)]
    raise ValueError(f"Unknown sentence variant mode {mode!r}.")


def selected_event_sentence(
    data: dict[str, Any],
    bindings: dict[str, str],
    event_name: str,
    seed: int,
    timestamp: int,
    mode: str,
) -> str:
    matches = [record for record in data["events"] if record["name"] == event_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one event record named {event_name!r}, found {len(matches)}.")
    return format_value(select_sentence_template(matches[0], seed, timestamp, mode), bindings)


def event_from_record(
    record: dict[str, Any],
    bindings: dict[str, str],
    timestamp: int,
    seed: int,
    sentence_variant_mode: str,
) -> SemanticEvent:
    required = {
        "name",
        "subject",
        "relations",
        "target",
        "reliable",
        "expected_action",
        "commit_truth",
        "evidence",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Event record missing required keys: {missing}")

    expected_action = str(record["expected_action"])
    if expected_action not in ACTIONS:
        raise ValueError(f"Unknown expected_action {expected_action!r} in event {record['name']!r}.")

    relations = tuple(canonical_token(str(relation)) for relation in record["relations"])
    if not relations:
        raise ValueError(f"Event {record['name']!r} must contain at least one relation.")

    reliable = bool(record["reliable"])
    source = str(record.get("source", "trusted" if reliable else "untrusted"))
    if source not in SOURCE_TO_INDEX:
        raise ValueError(f"Unknown source {source!r}; available sources: {SOURCE_NAMES}.")

    sentence = format_value(select_sentence_template(record, seed, timestamp, sentence_variant_mode), bindings)
    subject = canonical_token(format_value(str(record["subject"]), bindings))
    target = canonical_token(format_value(str(record["target"]), bindings))
    if subject not in canonical_token(sentence):
        raise ValueError(f"Sentence for event {record['name']!r} does not contain subject {subject!r}.")
    if target not in canonical_token(sentence):
        raise ValueError(f"Sentence for event {record['name']!r} does not contain target {target!r}.")

    return SemanticEvent(
        name=str(record["name"]),
        subject=subject,
        relations=relations,
        target=target,
        reliable=reliable,
        expected_action=expected_action,
        commit_truth=bool(record["commit_truth"]),
        evidence=int(record["evidence"]),
        timestamp=timestamp,
        source=source,
    )


def events_for_seed(
    data: dict[str, Any],
    seed: int,
    sentence_variant_mode: str,
) -> tuple[list[SemanticEvent], dict[str, str]]:
    bindings = resolve_bindings(data, seed)
    events = [
        event_from_record(record, bindings, index, seed, sentence_variant_mode)
        for index, record in enumerate(data["events"])
    ]
    return events, bindings


def build_world_from_dataset(data: dict[str, Any]) -> World:
    token_values: set[str] = set()
    relation_values: set[str] = set()
    variables = data["variables"]

    for value in variables.values():
        if isinstance(value, list):
            token_values.update(canonical_token(str(item)) for item in value)
        elif isinstance(value, dict):
            token_values.update(canonical_token(str(key)) for key in value.keys())
            token_values.update(canonical_token(str(item)) for item in value.values())

    for record in data["events"]:
        for relation in record["relations"]:
            relation_values.add(canonical_token(str(relation)))

    if not token_values:
        raise ValueError("No token values were found in dataset variables.")
    if not relation_values:
        raise ValueError("No relation values were found in dataset events.")

    id_to_token = tuple(sorted(token_values))
    id_to_relation = tuple(sorted(relation_values))
    return World(
        token_to_id={token: index for index, token in enumerate(id_to_token)},
        id_to_token=id_to_token,
        relation_to_id={relation: index for index, relation in enumerate(id_to_relation)},
        id_to_relation=id_to_relation,
    )


def reliable_one_hop_events_from_dataset(
    data: dict[str, Any],
    seed_offset: int,
    seed_count: int,
) -> list[SemanticEvent]:
    if seed_count <= 0:
        raise ValueError(f"seed_count must be positive, got {seed_count}.")
    events: list[SemanticEvent] = []
    for seed in range(seed_offset, seed_offset + seed_count):
        stream, _ = events_for_seed(data, seed, "first")
        events.extend(
            event
            for event in stream
            if event.is_one_hop and event.reliable and event.commit_truth
        )
    if not events:
        raise RuntimeError("No reliable one-hop events were available for geometry warmup.")
    return events


def train_latent_geometry_base_nl(
    args: argparse.Namespace,
    data: dict[str, Any],
    world: World,
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

    set_seed(args.geometry_seed)
    model = make_model(world, args, device)
    events = reliable_one_hop_events_from_dataset(
        data,
        args.geometry_train_seed_offset,
        args.geometry_train_seed_count,
    )
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
        desc="nl latent geometry warmup",
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
            raise FloatingPointError("Non-finite natural-language geometry warmup loss.")
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


def collect_policy_examples_nl(
    args: argparse.Namespace,
    data: dict[str, Any],
    world: World,
    device: torch.device,
    base_state: dict[str, torch.Tensor] | None,
) -> list[tuple[torch.Tensor, int]]:
    examples: list[tuple[torch.Tensor, int]] = []
    seed_iter = range(args.policy_train_seed_offset, args.policy_train_seed_offset + args.policy_train_seed_count)
    for seed in tqdm(seed_iter, desc="nl policy examples", dynamic_ncols=True, disable=args.no_progress):
        set_seed(seed)
        model = make_model(world, args, device, base_state)
        slot_use = torch.zeros(args.num_slots, dtype=torch.float32, device=device)
        slot_owners: list[tuple[str, tuple[str, ...]] | None] = [None] * args.num_slots
        fact_slots: dict[tuple[str, tuple[str, ...]], int] = {}
        belief_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
        consolidation_stats = ConsolidationStats()
        stream, _bindings = events_for_seed(data, seed, args.sentence_variant_mode)

        for event in stream:
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
        raise RuntimeError("Natural-language policy training produced no examples.")
    return examples


def train_write_policy_nl(
    args: argparse.Namespace,
    data: dict[str, Any],
    world: World,
    device: torch.device,
    base_state: dict[str, torch.Tensor] | None,
) -> CandidateWritePolicy:
    examples = collect_policy_examples_nl(args, data, world, device, base_state)
    input_dim = examples[0][0].shape[-1]
    if any(example[0].shape[-1] != input_dim for example in examples):
        raise RuntimeError("Policy examples have inconsistent feature dimensions.")

    policy = CandidateWritePolicy(input_dim=input_dim, hidden_dim=args.policy_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.policy_lr)
    order = np.arange(len(examples))
    for _ in tqdm(range(args.policy_epochs), desc="nl policy train", dynamic_ncols=True, disable=args.no_progress):
        np.random.shuffle(order)
        for index in order:
            features_cpu, target_index = examples[int(index)]
            features = features_cpu.to(device)
            target = torch.tensor([target_index], dtype=torch.long, device=device)
            logits = policy(features).unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite natural-language write-policy loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return policy


@torch.no_grad()
def choose_with_policy_nl(
    policy: CandidateWritePolicy,
    candidates: list[Any],
    event: SemanticEvent,
    world: World,
    args: argparse.Namespace,
    device: torch.device,
) -> Any:
    features = candidate_feature_tensor(candidates, event, world, args.num_slots, device, args)
    logits = policy(features)
    chosen_index = int(logits.argmax().item())
    return candidates[chosen_index]


def run_learned_policy_nl(
    seed: int,
    args: argparse.Namespace,
    data: dict[str, Any],
    world: World,
    policy: CandidateWritePolicy,
    base_state: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    model = make_model(world, args, device, base_state)
    slot_use = torch.zeros(args.num_slots, dtype=torch.float32, device=device)
    slot_owners: list[tuple[str, tuple[str, ...]] | None] = [None] * args.num_slots
    fact_slots: dict[tuple[str, tuple[str, ...]], int] = {}
    belief_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
    truth_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
    consolidation_stats = ConsolidationStats()
    rows: list[dict[str, Any]] = []
    stream, bindings = events_for_seed(data, seed, args.sentence_variant_mode)

    for event in stream:
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
        chosen = choose_with_policy_nl(policy, candidates, event, world, args, device)
        action_ok = chosen.action == event.expected_action

        model = chosen.model
        commit_belief_from_choice(belief_facts, event, chosen, args.commit_acc_threshold)
        commit_truth(truth_facts, event)
        update_slot_bookkeeping(fact_slots, slot_owners, slot_use, event, chosen, args.commit_acc_threshold)

        rows.append(
            {
                "event": event.name,
                "sentence": selected_event_sentence(
                    data,
                    bindings,
                    event.name,
                    seed,
                    event.timestamp,
                    args.sentence_variant_mode,
                ),
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


def run_blind_adamw_nl(
    seed: int,
    args: argparse.Namespace,
    data: dict[str, Any],
    world: World,
    base_state: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    model = make_model(world, args, device, base_state)
    active_facts: dict[tuple[str, tuple[str, ...]], SemanticEvent] = {}
    rows: list[dict[str, Any]] = []
    stream, _bindings = events_for_seed(data, seed, args.sentence_variant_mode)

    for event in stream:
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


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\nNATURAL-LANGUAGE SEMANTIC CL SUMMARY")
    print("=" * 124)
    print(
        f"dataset={report['dataset']['name']} extraction={report['dataset']['extraction_mode']} "
        f"seeds={report['seed_count']} vocab={report['world']['vocab_size']} relations={report['world']['relation_count']}"
    )
    print("-" * 124)
    print(
        f"{'method':<28} {'active_acc':<24} {'composition_acc':<24} "
        f"{'action_acc':<24} {'parents':<18} {'freed_slots':<18}"
    )
    print("-" * 124)
    for method, metrics in summary["methods"].items():
        print(
            f"{method:<28} "
            f"{metrics['final_active_acc']:<24} "
            f"{metrics['final_composition_acc']:<24} "
            f"{metrics.get('action_acc', 'n/a'):<24} "
            f"{metrics.get('final_parent_count', 'n/a'):<18} "
            f"{metrics.get('consolidation_freed_slots', 'n/a'):<18}"
        )
    print("=" * 124)


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
    raise ValueError(f"Unknown device {name!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("data/natural_language_semantic_cl/fact_stream_spec.json"),
    )
    parser.add_argument("--extraction-mode", choices=("gold",), default="gold")
    parser.add_argument("--sentence-variant-mode", choices=("first", "seed"), default="seed")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-slots", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--update-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--lambda-closure", type=float, default=1.0)
    parser.add_argument("--geometry-warmup", action="store_true")
    parser.add_argument("--geometry-seed", type=int, default=12345)
    parser.add_argument("--geometry-train-seed-count", type=int, default=80)
    parser.add_argument("--geometry-train-seed-offset", type=int, default=2000)
    parser.add_argument("--geometry-warmup-epochs", type=int, default=600)
    parser.add_argument("--geometry-warmup-lr", type=float, default=3e-3)
    parser.add_argument("--geometry-token-ce-weight", type=float, default=1.0)
    parser.add_argument("--geometry-separation-weight", type=float, default=0.05)
    parser.add_argument("--geometry-code-norm-weight", type=float, default=0.01)
    parser.add_argument("--geometry-max-code-cosine", type=float, default=0.3)
    parser.add_argument("--direct-write-weight", type=float, default=1.0)
    parser.add_argument("--composition-write-weight", type=float, default=0.25)
    parser.add_argument("--attention-margin", type=float, default=0.25)
    parser.add_argument("--enable-consolidation", action="store_true")
    parser.add_argument("--max-parents", type=int, default=4)
    parser.add_argument("--parent-confidence-weight", type=float, default=1.0)
    parser.add_argument("--consolidation-epochs", type=int, default=160)
    parser.add_argument("--consolidation-lr", type=float, default=1e-2)
    parser.add_argument("--consolidation-margin", type=float, default=0.25)
    parser.add_argument("--consolidation-max-candidates", type=int, default=6)
    parser.add_argument(
        "--consolidation-group-order",
        choices=("current", "same_relation_first", "geometry"),
        default="geometry",
    )
    parser.add_argument(
        "--consolidation-admission",
        choices=("current", "same_relation", "composition_preserving"),
        default="composition_preserving",
    )
    parser.add_argument("--consolidation-min-offset-cosine", type=float, default=0.78)
    parser.add_argument("--consolidation-max-direct-closure-delta", type=float, default=0.15)
    parser.add_argument("--consolidation-max-first-hop-closure-delta", type=float, default=0.05)
    parser.add_argument("--consolidation-max-dependent-composition-closure-delta", type=float, default=0.45)
    parser.add_argument("--parent-offset-weight", type=float, default=1.0)
    parser.add_argument("--parent-first-hop-weight", type=float, default=1.0)
    parser.add_argument("--parent-composition-weight", type=float, default=0.25)
    parser.add_argument("--parent-anti-interference-weight", type=float, default=0.25)
    parser.add_argument("--record-consolidation-diagnostics", action="store_true")
    parser.add_argument("--closure-penalty", type=float, default=0.1)
    parser.add_argument("--full-update-penalty", type=float, default=0.15)
    parser.add_argument("--commit-acc-threshold", type=float, default=0.999)
    parser.add_argument("--policy-train-seed-count", type=int, default=40)
    parser.add_argument("--policy-train-seed-offset", type=int, default=1000)
    parser.add_argument("--policy-epochs", type=int, default=100)
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--disable-role-embeddings", action="store_true")
    parser.add_argument("--disable-position-encoding", action="store_true")
    parser.add_argument("--disable-policy-identity-features", action="store_true")
    parser.add_argument("--disable-policy-position-features", action="store_true")
    parser.add_argument("--disable-policy-time-features", action="store_true")
    parser.add_argument("--disable-policy-source-features", action="store_true")
    parser.add_argument("--disable-policy-evidence-features", action="store_true")
    parser.add_argument("--skip-blind-adamw", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("model/analysis/natural-language-semantic-cl.json"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.seed_count <= 0:
        raise ValueError(f"--seed-count must be positive, got {args.seed_count}.")
    if args.d_model <= 0:
        raise ValueError(f"--d-model must be positive, got {args.d_model}.")
    if args.num_slots <= 0:
        raise ValueError(f"--num-slots must be positive, got {args.num_slots}.")
    if args.update_epochs <= 0:
        raise ValueError(f"--update-epochs must be positive, got {args.update_epochs}.")
    if args.policy_train_seed_count <= 0:
        raise ValueError(f"--policy-train-seed-count must be positive, got {args.policy_train_seed_count}.")
    if args.policy_epochs <= 0:
        raise ValueError(f"--policy-epochs must be positive, got {args.policy_epochs}.")
    if args.enable_consolidation and args.max_parents <= 0:
        raise ValueError("--enable-consolidation requires --max-parents > 0.")


def config_dict(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    config["dataset_json"] = str(args.dataset_json)
    config["output_json"] = str(args.output_json)
    return config


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    data = load_dataset(args.dataset_json)
    if data["extraction_mode"] != args.extraction_mode:
        raise ValueError(
            f"Dataset extraction_mode={data['extraction_mode']!r} does not match "
            f"--extraction-mode={args.extraction_mode!r}."
        )
    device = resolve_device(args.device)
    world = build_world_from_dataset(data)
    base_state, geometry_warmup = train_latent_geometry_base_nl(args, data, world, device)
    policy = train_write_policy_nl(args, data, world, device, base_state)

    reports: list[dict[str, Any]] = []
    for seed in tqdm(range(args.seed_count), desc="nl evaluation seeds", dynamic_ncols=True, disable=args.no_progress):
        reports.append(run_learned_policy_nl(seed, args, data, world, policy, base_state))
        if not args.skip_blind_adamw:
            reports.append(run_blind_adamw_nl(seed, args, data, world, base_state))

    return {
        "dataset": {
            "path": str(args.dataset_json),
            "name": str(data["name"]),
            "extraction_mode": str(data["extraction_mode"]),
            "sentence_variant_mode": args.sentence_variant_mode,
        },
        "world": {
            "vocab_size": len(world.id_to_token),
            "relation_count": len(world.id_to_relation),
            "tokens": list(world.id_to_token),
            "relations": list(world.id_to_relation),
        },
        "config": config_dict(args),
        "geometry_warmup": geometry_warmup,
        "seed_count": args.seed_count,
        "reports": reports,
        "summary": summarize(reports),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    output = run_experiment(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print_summary(output)
    print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
