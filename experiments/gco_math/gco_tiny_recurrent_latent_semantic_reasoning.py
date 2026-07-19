"""Test recurrent internal semantic reasoning without reasoning tokens.

Each synthetic world is a directed semantic graph.  Every relation and context
has a hidden categorical transition rule.  The learner sees old node values,
graph structure, and a trusted revision to the root value, but never receives
the transition tables.  It must infer the rules from training experience and
propagate the revision through the graph.

Two matched neural models are trained:

* one-step: one latent message-passing update;
* recurrent: a weight-tied latent update with a learned halting distribution.

The held-out evaluation uses deeper graphs than training, context-dependent
rules, disconnected locality nodes, and repeated revisions.  Hidden rules and
truth are training targets and evaluation metadata, not model inputs.
"""

from __future__ import annotations

import argparse
import copy
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device


@dataclass(frozen=True)
class HiddenRules:
    transition: torch.Tensor


@dataclass
class SemanticGraphBatch:
    old_values: torch.Tensor
    revised_root: torch.Tensor
    targets: torch.Tensor
    parents: torch.Tensor
    relations: torch.Tensor
    contexts: torch.Tensor
    depth: torch.Tensor
    affected: torch.Tensor

    def to(self, device: torch.device) -> SemanticGraphBatch:
        return SemanticGraphBatch(
            old_values=self.old_values.to(device),
            revised_root=self.revised_root.to(device),
            targets=self.targets.to(device),
            parents=self.parents.to(device),
            relations=self.relations.to(device),
            contexts=self.contexts.to(device),
            depth=self.depth.to(device),
            affected=self.affected.to(device),
        )


@dataclass
class ReasoningRollout:
    logits: list[torch.Tensor]
    halt_weights: torch.Tensor
    expected_steps: torch.Tensor


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def positive_float(name: str, value: float) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def probability(name: str, value: float) -> None:
    if not 0.0 < value < 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and strictly between zero and one.")


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_nodes",
        "num_relations",
        "num_values",
        "d_model",
        "hidden_dim",
        "batch_size",
        "train_updates",
        "train_max_depth",
        "test_max_depth",
        "train_recurrent_steps",
        "test_recurrent_steps",
        "eval_batches",
        "revisions",
        "print_every",
    ):
        positive_int(name, getattr(args, name))
    for name in ("learning_rate", "gradient_clip", "ponder_cost"):
        positive_float(name, getattr(args, name))
    probability("disconnected_fraction", args.disconnected_fraction)
    for name in (
        "ood_accuracy_threshold",
        "deep_hop_threshold",
        "locality_threshold",
        "context_threshold",
        "revision_threshold",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and in [0, 1].")
    if args.test_max_depth <= args.train_max_depth:
        raise ValueError("test_max_depth must exceed train_max_depth to test extrapolation.")
    if args.train_recurrent_steps <= args.train_max_depth:
        raise ValueError("train_recurrent_steps must exceed train_max_depth.")
    if args.test_recurrent_steps <= args.test_max_depth:
        raise ValueError("test_recurrent_steps must exceed test_max_depth.")
    if args.num_nodes <= args.test_max_depth + 1:
        raise ValueError("num_nodes must leave room for the deepest chain and locality nodes.")
    connected = int(round(args.num_nodes * (1.0 - args.disconnected_fraction)))
    if connected <= args.test_max_depth:
        raise ValueError("disconnected_fraction leaves too few nodes for the requested depth.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def alternative_value(value: int, *, num_values: int, generator: random.Random) -> int:
    return (value + generator.randrange(1, num_values)) % num_values


def build_hidden_rules(*, num_relations: int, num_values: int, seed: int) -> HiddenRules:
    generator = random.Random(seed)
    rows: list[list[list[int]]] = []
    for context in range(2):
        context_rows: list[list[int]] = []
        for relation in range(num_relations):
            permutation = list(range(num_values))
            generator.shuffle(permutation)
            if context == 1 and permutation == rows[0][relation]:
                permutation = permutation[1:] + permutation[:1]
            context_rows.append(permutation)
        rows.append(context_rows)
    transition = torch.tensor(rows, dtype=torch.long)
    if transition.shape != (2, num_relations, num_values):
        raise RuntimeError("Hidden transition table has an invalid shape.")
    return HiddenRules(transition=transition)


def apply_rule(
    rules: HiddenRules,
    *,
    context: int,
    relation: int,
    parent_value: int,
) -> int:
    return int(rules.transition[context, relation, parent_value])


def generate_graph_batch(
    rules: HiddenRules,
    *,
    batch_size: int,
    num_nodes: int,
    num_relations: int,
    num_values: int,
    max_depth: int,
    disconnected_fraction: float,
    seed: int,
) -> SemanticGraphBatch:
    generator = random.Random(seed)
    connected_count = int(round(num_nodes * (1.0 - disconnected_fraction)))
    if connected_count <= max_depth:
        raise ValueError("Graph batch cannot realize the requested depth.")
    old_rows: list[list[int]] = []
    revised_roots: list[int] = []
    target_rows: list[list[int]] = []
    parent_rows: list[list[int]] = []
    relation_rows: list[list[int]] = []
    context_rows: list[list[int]] = []
    depth_rows: list[list[int]] = []
    affected_rows: list[list[bool]] = []

    for _sample in range(batch_size):
        parents = [-1] * num_nodes
        relations = [0] * num_nodes
        contexts = [0] * num_nodes
        depths = [-1] * num_nodes
        depths[0] = 0
        for node in range(1, max_depth + 1):
            parents[node] = node - 1
            relations[node] = generator.randrange(num_relations)
            contexts[node] = generator.randrange(2)
            depths[node] = node
        for node in range(max_depth + 1, connected_count):
            candidates = [index for index in range(node) if 0 <= depths[index] < max_depth]
            if not candidates:
                raise RuntimeError("Graph generator found no valid parent candidate.")
            parent = generator.choice(candidates)
            parents[node] = parent
            relations[node] = generator.randrange(num_relations)
            contexts[node] = generator.randrange(2)
            depths[node] = depths[parent] + 1

        old_values = [generator.randrange(num_values) for _node in range(num_nodes)]
        for node in range(1, connected_count):
            old_values[node] = apply_rule(
                rules,
                context=contexts[node],
                relation=relations[node],
                parent_value=old_values[parents[node]],
            )
        revised_root = alternative_value(
            old_values[0],
            num_values=num_values,
            generator=generator,
        )
        targets = list(old_values)
        targets[0] = revised_root
        for node in range(1, connected_count):
            targets[node] = apply_rule(
                rules,
                context=contexts[node],
                relation=relations[node],
                parent_value=targets[parents[node]],
            )
        affected = [node < connected_count for node in range(num_nodes)]
        old_rows.append(old_values)
        revised_roots.append(revised_root)
        target_rows.append(targets)
        parent_rows.append(parents)
        relation_rows.append(relations)
        context_rows.append(contexts)
        depth_rows.append(depths)
        affected_rows.append(affected)

    return SemanticGraphBatch(
        old_values=torch.tensor(old_rows, dtype=torch.long),
        revised_root=torch.tensor(revised_roots, dtype=torch.long),
        targets=torch.tensor(target_rows, dtype=torch.long),
        parents=torch.tensor(parent_rows, dtype=torch.long),
        relations=torch.tensor(relation_rows, dtype=torch.long),
        contexts=torch.tensor(context_rows, dtype=torch.long),
        depth=torch.tensor(depth_rows, dtype=torch.long),
        affected=torch.tensor(affected_rows, dtype=torch.bool),
    )


def revise_existing_batch(
    batch: SemanticGraphBatch,
    rules: HiddenRules,
    *,
    num_values: int,
    seed: int,
) -> SemanticGraphBatch:
    generator = random.Random(seed)
    old_values = batch.targets.detach().cpu().clone()
    parents = batch.parents.detach().cpu()
    relations = batch.relations.detach().cpu()
    contexts = batch.contexts.detach().cpu()
    affected = batch.affected.detach().cpu()
    targets = old_values.clone()
    revised_roots: list[int] = []
    for sample in range(old_values.shape[0]):
        revised_root = alternative_value(
            int(old_values[sample, 0]),
            num_values=num_values,
            generator=generator,
        )
        revised_roots.append(revised_root)
        targets[sample, 0] = revised_root
        for node in range(1, old_values.shape[1]):
            if not bool(affected[sample, node]):
                targets[sample, node] = old_values[sample, node]
                continue
            parent = int(parents[sample, node])
            if parent < 0:
                raise RuntimeError("Affected non-root node has no parent.")
            targets[sample, node] = apply_rule(
                rules,
                context=int(contexts[sample, node]),
                relation=int(relations[sample, node]),
                parent_value=int(targets[sample, parent]),
            )
    return SemanticGraphBatch(
        old_values=old_values,
        revised_root=torch.tensor(revised_roots, dtype=torch.long),
        targets=targets,
        parents=parents.clone(),
        relations=relations.clone(),
        contexts=contexts.clone(),
        depth=batch.depth.detach().cpu().clone(),
        affected=affected.clone(),
    )


class LatentSemanticReasoner(nn.Module):
    def __init__(
        self,
        *,
        num_values: int,
        num_relations: int,
        d_model: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_values = num_values
        self.num_relations = num_relations
        self.d_model = d_model
        self.value_embedding = nn.Embedding(num_values, d_model)
        self.relation_embedding = nn.Embedding(num_relations, d_model)
        self.context_embedding = nn.Embedding(2, d_model)
        self.root_encoder = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.message = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.recurrence = nn.GRUCell(d_model, d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_values),
        )
        self.halt = nn.Sequential(
            nn.LayerNorm(d_model + 2),
            nn.Linear(d_model + 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def initial_state(self, batch: SemanticGraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        old = self.value_embedding(batch.old_values)
        revised = self.value_embedding(batch.revised_root)
        root = self.root_encoder(torch.cat([old[:, 0], revised], dim=1))
        state = old.clone()
        state[:, 0] = root
        return state, root

    def reasoning_step(
        self,
        state: torch.Tensor,
        root_state: torch.Tensor,
        batch: SemanticGraphBatch,
    ) -> torch.Tensor:
        batch_size, num_nodes, width = state.shape
        if width != self.d_model:
            raise ValueError("Latent state width does not match the reasoner.")
        parent_indices = batch.parents.clamp_min(0)
        parent_state = state.gather(
            1,
            parent_indices.unsqueeze(2).expand(batch_size, num_nodes, width),
        )
        relation = self.relation_embedding(batch.relations)
        context = self.context_embedding(batch.contexts)
        message = self.message(torch.cat([parent_state, state, relation, context], dim=2))
        updated = self.recurrence(
            message.reshape(batch_size * num_nodes, width),
            state.reshape(batch_size * num_nodes, width),
        ).reshape(batch_size, num_nodes, width)
        has_parent = batch.parents >= 0
        next_state = torch.where(has_parent.unsqueeze(2), updated, state)
        next_state[:, 0] = root_state
        return next_state

    def rollout(self, batch: SemanticGraphBatch, *, max_steps: int) -> ReasoningRollout:
        positive_int("max_steps", max_steps)
        state, root_state = self.initial_state(batch)
        logits: list[torch.Tensor] = []
        halt_probabilities: list[torch.Tensor] = []
        previous_state = state
        for _step in range(max_steps):
            state = self.reasoning_step(state, root_state, batch)
            step_logits = self.output(state)
            probabilities = torch.softmax(step_logits, dim=2)
            eps = torch.finfo(probabilities.dtype).eps
            entropy = -(
                probabilities * probabilities.clamp_min(eps).log()
            ).sum(dim=2).mean(dim=1)
            residual = torch.linalg.vector_norm(state - previous_state, dim=2).mean(dim=1)
            pooled = state.mean(dim=1)
            halt_input = torch.cat([pooled, entropy.unsqueeze(1), residual.unsqueeze(1)], dim=1)
            halt_probabilities.append(torch.sigmoid(self.halt(halt_input)).squeeze(1))
            logits.append(step_logits)
            previous_state = state
        remaining = torch.ones_like(halt_probabilities[0])
        weights: list[torch.Tensor] = []
        for step, halt_probability in enumerate(halt_probabilities):
            if step == max_steps - 1:
                weight = remaining
            else:
                weight = remaining * halt_probability
                remaining = remaining * (1.0 - halt_probability)
            weights.append(weight)
        halt_weights = torch.stack(weights, dim=1)
        if not torch.allclose(
            halt_weights.sum(dim=1),
            torch.ones_like(remaining),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("Halting distribution does not sum to one.")
        step_numbers = torch.arange(
            1,
            max_steps + 1,
            device=halt_weights.device,
            dtype=halt_weights.dtype,
        )
        expected_steps = (halt_weights * step_numbers.unsqueeze(0)).sum(dim=1)
        return ReasoningRollout(
            logits=logits,
            halt_weights=halt_weights,
            expected_steps=expected_steps,
        )


def per_sample_prediction_loss(logits: torch.Tensor, batch: SemanticGraphBatch) -> torch.Tensor:
    batch_size, num_nodes, num_values = logits.shape
    row_loss = F.cross_entropy(
        logits.reshape(batch_size * num_nodes, num_values),
        batch.targets.reshape(batch_size * num_nodes),
        reduction="none",
    ).reshape(batch_size, num_nodes)
    affected = batch.affected.to(dtype=row_loss.dtype)
    locality = (~batch.affected).to(dtype=row_loss.dtype)
    affected_loss = (row_loss * affected).sum(dim=1) / affected.sum(dim=1).clamp_min(1.0)
    locality_loss = (row_loss * locality).sum(dim=1) / locality.sum(dim=1).clamp_min(1.0)
    return 0.5 * (affected_loss + locality_loss)


def training_objective(
    rollout: ReasoningRollout,
    batch: SemanticGraphBatch,
    *,
    ponder_cost: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    step_losses = torch.stack(
        [per_sample_prediction_loss(logits, batch) for logits in rollout.logits],
        dim=1,
    )
    expected_prediction = (rollout.halt_weights * step_losses).sum(dim=1).mean()
    ponder = ponder_cost * rollout.expected_steps.mean()
    objective = expected_prediction + ponder
    if not torch.isfinite(objective):
        raise FloatingPointError("Latent reasoning objective became non-finite.")
    return objective, {
        "objective": float(objective.detach().cpu()),
        "prediction": float(expected_prediction.detach().cpu()),
        "ponder": float(ponder.detach().cpu()),
        "expected_steps": float(rollout.expected_steps.mean().detach().cpu()),
    }


@torch.no_grad()
def rollout_probabilities(
    rollout: ReasoningRollout,
    *,
    mode: str,
) -> torch.Tensor:
    probabilities = torch.stack([torch.softmax(logits, dim=2) for logits in rollout.logits], dim=1)
    if mode == "adaptive":
        return (rollout.halt_weights.unsqueeze(2).unsqueeze(3) * probabilities).sum(dim=1)
    if mode == "final":
        return probabilities[:, -1]
    raise ValueError(f"Unknown rollout evaluation mode {mode!r}.")


@torch.no_grad()
def evaluate_at_depth(
    model: LatentSemanticReasoner,
    rules: HiddenRules,
    *,
    depth: int,
    max_steps: int,
    mode: str,
    args: argparse.Namespace,
    seed_offset: int,
    device: torch.device,
) -> dict[str, float]:
    correct = 0
    total = 0
    affected_correct = 0
    affected_total = 0
    locality_correct = 0
    locality_total = 0
    context_correct = 0
    context_total = 0
    hop_correct = {hop: 0 for hop in range(depth + 1)}
    hop_total = {hop: 0 for hop in range(depth + 1)}
    expected_steps: list[float] = []
    model.eval()
    for batch_index in range(args.eval_batches):
        batch = generate_graph_batch(
            rules,
            batch_size=args.batch_size,
            num_nodes=args.num_nodes,
            num_relations=args.num_relations,
            num_values=args.num_values,
            max_depth=depth,
            disconnected_fraction=args.disconnected_fraction,
            seed=args.seed * 1_000_003 + seed_offset + batch_index,
        ).to(device)
        rollout = model.rollout(batch, max_steps=max_steps)
        predictions = rollout_probabilities(rollout, mode=mode).argmax(dim=2)
        matches = predictions == batch.targets
        correct += int(matches.sum().cpu())
        total += matches.numel()
        affected_correct += int(matches[batch.affected].sum().cpu())
        affected_total += int(batch.affected.sum().cpu())
        locality_correct += int(matches[~batch.affected].sum().cpu())
        locality_total += int((~batch.affected).sum().cpu())
        exception_mask = batch.affected & (batch.parents >= 0) & (batch.contexts == 1)
        context_correct += int(matches[exception_mask].sum().cpu())
        context_total += int(exception_mask.sum().cpu())
        for hop in range(depth + 1):
            mask = batch.depth == hop
            hop_correct[hop] += int(matches[mask].sum().cpu())
            hop_total[hop] += int(mask.sum().cpu())
        expected_steps.append(float(rollout.expected_steps.mean().cpu()))
    if min(total, affected_total, locality_total, context_total) <= 0:
        raise RuntimeError("Evaluation produced an empty metric group.")
    result = {
        "depth": float(depth),
        "accuracy": correct / float(total),
        "affected_accuracy": affected_correct / float(affected_total),
        "locality_accuracy": locality_correct / float(locality_total),
        "context_accuracy": context_correct / float(context_total),
        "expected_steps": sum(expected_steps) / float(len(expected_steps)),
    }
    result.update(
        {
            f"hop_{hop}_accuracy": hop_correct[hop] / float(hop_total[hop])
            for hop in range(depth + 1)
            if hop_total[hop] > 0
        }
    )
    return result


@torch.no_grad()
def evaluate_revisions(
    model: LatentSemanticReasoner,
    rules: HiddenRules,
    *,
    max_steps: int,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    batch = generate_graph_batch(
        rules,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        num_relations=args.num_relations,
        num_values=args.num_values,
        max_depth=args.test_max_depth,
        disconnected_fraction=args.disconnected_fraction,
        seed=args.seed * 5_000_011 + 17,
    )
    rows: list[dict[str, float]] = []
    for revision in range(1, args.revisions + 1):
        if revision > 1:
            batch = revise_existing_batch(
                batch,
                rules,
                num_values=args.num_values,
                seed=args.seed * 7_000_027 + revision,
            )
        device_batch = batch.to(device)
        rollout = model.rollout(device_batch, max_steps=max_steps)
        predictions = rollout_probabilities(rollout, mode=mode).argmax(dim=2)
        matches = predictions == device_batch.targets
        rows.append(
            {
                "revision": float(revision),
                "affected_accuracy": float(matches[device_batch.affected].to(torch.float32).mean().cpu()),
                "locality_accuracy": float(matches[~device_batch.affected].to(torch.float32).mean().cpu()),
                "expected_steps": float(rollout.expected_steps.mean().cpu()),
            }
        )
    return {
        "mean_affected_accuracy": sum(row["affected_accuracy"] for row in rows) / len(rows),
        "minimum_affected_accuracy": min(row["affected_accuracy"] for row in rows),
        "mean_locality_accuracy": sum(row["locality_accuracy"] for row in rows) / len(rows),
        "revisions": rows,
    }


def train_models(
    one_step: LatentSemanticReasoner,
    recurrent: LatentSemanticReasoner,
    rules: HiddenRules,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    optimizers = {
        "one_step": torch.optim.AdamW(one_step.parameters(), lr=args.learning_rate),
        "recurrent": torch.optim.AdamW(recurrent.parameters(), lr=args.learning_rate),
    }
    models = {"one_step": one_step, "recurrent": recurrent}
    traces: dict[str, list[dict[str, float]]] = {name: [] for name in models}
    depth_generator = random.Random(args.seed + 33_191)
    print("LATENT SEMANTIC REASONING TRAINING")
    print("=" * 136)
    for update in range(1, args.train_updates + 1):
        depth = depth_generator.randint(1, args.train_max_depth)
        batch = generate_graph_batch(
            rules,
            batch_size=args.batch_size,
            num_nodes=args.num_nodes,
            num_relations=args.num_relations,
            num_values=args.num_values,
            max_depth=depth,
            disconnected_fraction=args.disconnected_fraction,
            seed=args.seed * 100_003 + update,
        ).to(device)
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            steps = 1 if name == "one_step" else args.train_recurrent_steps
            rollout = model.rollout(batch, max_steps=steps)
            objective, report = training_objective(
                rollout,
                batch,
                ponder_cost=args.ponder_cost if name == "recurrent" else 1e-12,
            )
            objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"{name} gradient became non-finite at update {update}.")
            optimizer.step()
            report.update(
                {
                    "update": float(update),
                    "depth": float(depth),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                }
            )
            traces[name].append(report)
        if update == 1 or update == args.train_updates or update % args.print_every == 0:
            one = traces["one_step"][-1]
            rec = traces["recurrent"][-1]
            print(
                f"update={update:4d} depth={depth} "
                f"one={one['objective']:.4f} recurrent={rec['objective']:.4f} "
                f"steps={rec['expected_steps']:.2f}"
            )
    return traces


def plot_results(
    traces: dict[str, list[dict[str, float]]],
    evaluations: dict[str, list[dict[str, float]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for name, color in (("one_step", "#dc2626"), ("recurrent", "#2563eb")):
        trace = traces[name]
        axes[0, 0].plot(
            [row["update"] for row in trace],
            [row["prediction"] for row in trace],
            color=color,
            alpha=0.8,
            label=name.replace("_", " "),
        )
    axes[0, 0].set_title("Future-consequence training loss")
    axes[0, 0].set_xlabel("update")
    axes[0, 0].legend()

    for name, color in (("one_step", "#dc2626"), ("recurrent", "#2563eb")):
        rows = evaluations[name]
        axes[0, 1].plot(
            [row["depth"] for row in rows],
            [row["affected_accuracy"] for row in rows],
            marker="o",
            color=color,
            label=name.replace("_", " "),
        )
    axes[0, 1].set_title("Affected accuracy by graph depth")
    axes[0, 1].set_xlabel("graph depth")
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].legend()

    deepest = int(evaluations["recurrent"][-1]["depth"])
    hops = list(range(deepest + 1))
    width = 0.35
    axes[1, 0].bar(
        [hop - width / 2 for hop in hops],
        [evaluations["one_step"][-1][f"hop_{hop}_accuracy"] for hop in hops],
        width=width,
        color="#dc2626",
        label="one step",
    )
    axes[1, 0].bar(
        [hop + width / 2 for hop in hops],
        [evaluations["recurrent"][-1][f"hop_{hop}_accuracy"] for hop in hops],
        width=width,
        color="#2563eb",
        label="recurrent",
    )
    axes[1, 0].set_title("Deepest held-out graph by reasoning hop")
    axes[1, 0].set_xlabel("distance from revised root")
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].legend()

    axes[1, 1].plot(
        [row["depth"] for row in evaluations["recurrent"]],
        [row["expected_steps"] for row in evaluations["recurrent"]],
        marker="o",
        color="#7c3aed",
    )
    axes[1, 1].set_title("Learned latent computation")
    axes[1, 1].set_xlabel("graph depth")
    axes[1, 1].set_ylabel("expected recurrent steps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = build_hidden_rules(
        num_relations=args.num_relations,
        num_values=args.num_values,
        seed=args.seed + 17_711,
    )
    recurrent = LatentSemanticReasoner(
        num_values=args.num_values,
        num_relations=args.num_relations,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
    ).to(device)
    one_step = copy.deepcopy(recurrent).to(device)
    traces = train_models(one_step, recurrent, rules, args=args, device=device)

    evaluations: dict[str, list[dict[str, float]]] = {"one_step": [], "recurrent": []}
    for depth in range(1, args.test_max_depth + 1):
        evaluations["one_step"].append(
            evaluate_at_depth(
                one_step,
                rules,
                depth=depth,
                max_steps=1,
                mode="final",
                args=args,
                seed_offset=depth * 10_007,
                device=device,
            )
        )
        evaluations["recurrent"].append(
            evaluate_at_depth(
                recurrent,
                rules,
                depth=depth,
                max_steps=args.test_recurrent_steps,
                mode="adaptive",
                args=args,
                seed_offset=depth * 10_007,
                device=device,
            )
        )
    revisions = {
        "one_step": evaluate_revisions(
            one_step,
            rules,
            max_steps=1,
            mode="final",
            args=args,
            device=device,
        ),
        "recurrent": evaluate_revisions(
            recurrent,
            rules,
            max_steps=args.test_recurrent_steps,
            mode="adaptive",
            args=args,
            device=device,
        ),
    }
    one_deep = evaluations["one_step"][-1]
    recurrent_deep = evaluations["recurrent"][-1]
    deepest_hop_key = f"hop_{args.test_max_depth}_accuracy"
    validation = {
        "recurrent_ood_reasoning": recurrent_deep["affected_accuracy"] >= args.ood_accuracy_threshold,
        "deep_hop_reasoning": recurrent_deep[deepest_hop_key] >= args.deep_hop_threshold,
        "recurrent_beats_one_step": recurrent_deep["affected_accuracy"] > one_deep["affected_accuracy"],
        "context_exceptions": recurrent_deep["context_accuracy"] >= args.context_threshold,
        "unaffected_locality": recurrent_deep["locality_accuracy"] >= args.locality_threshold,
        "repeated_revisions": revisions["recurrent"]["minimum_affected_accuracy"] >= args.revision_threshold,
        "adaptive_computation": evaluations["recurrent"][-1]["expected_steps"]
        > evaluations["recurrent"][0]["expected_steps"],
    }

    print("\nFINAL RECURRENT LATENT SEMANTIC REASONING")
    print("-" * 136)
    print(
        f"deep={args.test_max_depth} one_step={one_deep['affected_accuracy']:.4f} "
        f"recurrent={recurrent_deep['affected_accuracy']:.4f} "
        f"deep_hop={recurrent_deep[deepest_hop_key]:.4f} "
        f"context={recurrent_deep['context_accuracy']:.4f} "
        f"locality={recurrent_deep['locality_accuracy']:.4f}"
    )
    print(
        f"expected_steps shallow/deep={evaluations['recurrent'][0]['expected_steps']:.3f}/"
        f"{recurrent_deep['expected_steps']:.3f} "
        f"revisions={revisions['recurrent']['minimum_affected_accuracy']:.4f} "
        f"one_step_revisions={revisions['one_step']['minimum_affected_accuracy']:.4f}"
    )
    print(f"validation={validation}")

    json_path = args.output_dir / "recurrent_latent_semantic_reasoning.json"
    plot_path = args.output_dir / "recurrent_latent_semantic_reasoning.png"
    checkpoint_path = args.output_dir / "recurrent_latent_semantic_reasoning.pt"
    output: dict[str, Any] = {
        "question": (
            "Can weight-tied recurrent latent computation infer hidden semantic transition rules and "
            "propagate revisions farther than a matched one-step neural reasoner without reasoning tokens?"
        ),
        "scope": (
            "Isolated consequence-reasoning gate. The root revision is trusted; source selection and "
            "Invariant-Tangent weight consolidation are intentionally not part of this experiment."
        ),
        "mechanism": {
            "state": "distributed node-value embeddings",
            "transition": "relation- and context-conditioned tied GRU message passing",
            "internal_compute": "recurrent latent states with no intermediate token decoding",
            "halting": "learned per-example PonderNet-style probability distribution",
            "training": "future semantic query loss plus expected-compute cost",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "hidden_rules": asdict(rules),
        "training": traces,
        "evaluation": evaluations,
        "revisions": revisions,
        "validation": validation,
    }
    output["hidden_rules"]["transition"] = rules.transition.tolist()
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    plot_results(traces, evaluations, plot_path)
    torch.save(
        {
            "format": "recurrent_latent_semantic_reasoning_v1",
            "one_step": one_step.state_dict(),
            "recurrent": recurrent.state_dict(),
            "hidden_rules": rules.transition,
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
        default=Path("model/analysis/gco-tiny-recurrent-latent-semantic-reasoning-seed0"),
    )
    parser.add_argument("--num-nodes", type=int, default=16)
    parser.add_argument("--num-relations", type=int, default=4)
    parser.add_argument("--num-values", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-updates", type=int, default=1200)
    parser.add_argument("--train-max-depth", type=int, default=4)
    parser.add_argument("--test-max-depth", type=int, default=5)
    parser.add_argument("--train-recurrent-steps", type=int, default=6)
    parser.add_argument("--test-recurrent-steps", type=int, default=7)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--revisions", type=int, default=4)
    parser.add_argument("--disconnected-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--ponder-cost", type=float, default=0.01)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--ood-accuracy-threshold", type=float, default=0.80)
    parser.add_argument("--deep-hop-threshold", type=float, default=0.75)
    parser.add_argument("--locality-threshold", type=float, default=0.95)
    parser.add_argument("--context-threshold", type=float, default=0.80)
    parser.add_argument("--revision-threshold", type=float, default=0.80)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
