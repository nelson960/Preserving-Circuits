#!/usr/bin/env python3
"""Real-text multi-layer nested geometry continual-learning prototype.

This experiment moves the nested-geometry prototype one step past synthetic
group tokens. Inputs are short English fact/update/noise statements, encoded by
a small trainable word-token model, then routed through a deeper nested tree:

    tokens -> shared encoder -> L0 -> L1 -> L2 -> L3 -> L4 answer head

Each selected path has independent parameters. Updates touch only the shared
encoder slightly plus the selected nested path. Weak or noisy events receive a
small write gate, while useful/dependent events can consolidate inward.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gco_hierarchical_nested_semantic_stress import (  # noqa: E402
    NodeState,
    adjust_node_depth,
    node_potential,
    node_share,
    node_survival,
    update_node_admission,
)
from gco_nested_tangent_optimizer_tiny_nn import (  # noqa: E402
    clip_relative,
    cosine_distance,
    grad_or_raise,
    project_gradient,
    resolve_device,
    unflatten_like,
)


ANSWER_LABELS = {
    "blue cabinet": 0,
    "green vault": 1,
    "north route": 2,
    "east route": 3,
    "west route": 4,
    "ember seven": 5,
    "silver key": 6,
    "gold key": 7,
    "kyoto lab": 8,
    "unknown": 9,
}

GROUP_COLORS = {
    "stable": "#377eb8",
    "merge_a": "#4daf4a",
    "merge_b": "#4daf4a",
    "branch_root": "#984ea3",
    "branch_up": "#ff7f00",
    "branch_down": "#a65628",
    "rare_critical": "#e41a1c",
    "obsolete_old": "#999999",
    "replacement": "#dede00",
    "novel": "#f781bf",
    "noise": "#555555",
}


@dataclass(frozen=True)
class FactSpec:
    group: str
    answer: str
    texts: tuple[str, ...]
    route_family: str
    route_branch: str
    route_leaf: str
    usefulness: float
    dependency: float
    start: int
    end: int
    period: int
    contradiction_target: str | None = None


@dataclass(frozen=True)
class TextEvent:
    tokens: torch.Tensor
    text: str
    y: int
    group: str
    route_family: str
    route_branch: str
    route_leaf: str
    usefulness: float
    dependency: float
    contradiction_target: str | None
    step: int


@dataclass(frozen=True)
class MemoryItem:
    tokens: torch.Tensor
    y: int
    group: str
    usefulness: float
    dependency: float


@dataclass(frozen=True)
class Config:
    seed: int
    steps: int
    slots_per_layer: str
    seq_len: int
    d_model: int
    hidden: int
    foundation_epochs: int
    foundation_lr: float
    foundation_geometry_weight: float
    inner_steps: int
    memory_limit: int
    base_lr: float
    shared_lr_multiplier: float
    path_lr_multiplier: float
    head_lr_multiplier: float
    depth_lr_decay: float
    strength_lr_decay: float
    center_lr: float
    family_match_threshold: float
    family_branch_threshold: float
    family_routing_bias: float
    match_threshold: float
    branch_threshold: float
    family_prefix_length: int
    leaf_prefix_length: int
    evidence_gain: float
    usefulness_gain: float
    dependency_gain: float
    conflict_gain: float
    age_decay: float
    support_gain: float
    support_decay: float
    support_threshold: float
    min_consolidation_potential: float
    write_gate_threshold: float
    write_gate_temperature: float
    commit_gate_threshold: float
    admission_threshold: float
    admission_temperature: float
    inward_rate: float
    outward_rate: float
    survival_temperature: float
    provisional_depth_cap: float
    low_potential_depth_cap: float
    low_potential_strength_cap: float
    release_threshold: float
    tangent_damping: float
    restore_weight: float
    restore_clip_ratio: float
    protected_min_potential: float
    read_reject_distance: float
    read_reject_confidence: float
    noise_conf_threshold: float
    device: str
    projection_device: str
    output_dir: Path

    @property
    def shell_count(self) -> int:
        return len(parse_slots(self.slots_per_layer)) + 1


def parse_slots(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("slots_per_layer must contain at least one integer.")
    if any(value < 1 for value in values):
        raise ValueError("Every slots_per_layer value must be positive.")
    return values


def validate_config(config: Config) -> None:
    slots = parse_slots(config.slots_per_layer)
    if len(slots) < 3:
        raise ValueError("The real-text nested experiment needs at least three nested layers.")
    if not 1 <= config.family_prefix_length <= len(slots):
        raise ValueError("family_prefix_length must fit inside slots_per_layer.")
    if not 1 <= config.leaf_prefix_length <= len(slots):
        raise ValueError("leaf_prefix_length must fit inside slots_per_layer.")
    if config.family_prefix_length > config.leaf_prefix_length:
        raise ValueError("family_prefix_length cannot exceed leaf_prefix_length.")
    if config.seq_len < 6:
        raise ValueError("seq_len must be at least 6.")
    if config.hidden < 8 or config.d_model < 8:
        raise ValueError("hidden and d_model must be at least 8.")
    if config.foundation_epochs < 0:
        raise ValueError("foundation_epochs cannot be negative.")
    if config.foundation_lr <= 0.0:
        raise ValueError("foundation_lr must be positive.")
    if config.foundation_geometry_weight < 0.0:
        raise ValueError("foundation_geometry_weight must be non-negative.")
    if config.write_gate_temperature <= 0.0:
        raise ValueError("write_gate_temperature must be positive.")
    if not 0.0 <= config.commit_gate_threshold <= 1.0:
        raise ValueError("commit_gate_threshold must be between 0 and 1.")
    if config.projection_device not in {"cpu", "same"}:
        raise ValueError("projection_device must be cpu or same.")
    if config.family_match_threshold < config.match_threshold:
        raise ValueError("family_match_threshold must be at least match_threshold.")
    if config.family_branch_threshold < config.branch_threshold:
        raise ValueError("family_branch_threshold must be at least branch_threshold.")
    if config.family_routing_bias < 0.0:
        raise ValueError("family_routing_bias must be non-negative.")
    if config.read_reject_distance < 0.0:
        raise ValueError("read_reject_distance must be non-negative.")
    if not 0.0 <= config.read_reject_confidence <= 1.0:
        raise ValueError("read_reject_confidence must be between 0 and 1.")


def fact_specs(steps: int) -> list[FactSpec]:
    return [
        FactSpec(
            "stable",
            "blue cabinet",
            (
                "Mira keeps the archive key in the blue cabinet.",
                "The archive key for Mira remains inside the blue cabinet.",
                "If asked about Mira's archive key, answer blue cabinet.",
            ),
            "archive",
            "stable",
            "blue cabinet",
            0.96,
            0.92,
            0,
            steps,
            18,
        ),
        FactSpec(
            "merge_a",
            "green vault",
            (
                "Nolan stores the climate ledger in the green vault.",
                "The climate ledger location for Nolan is the green vault.",
            ),
            "ledger",
            "same",
            "green vault",
            0.70,
            0.66,
            0,
            steps,
            22,
        ),
        FactSpec(
            "merge_b",
            "green vault",
            (
                "A second note says Nolan's climate ledger is kept in the green vault.",
                "Nolan ledger storage is also described as the green vault.",
            ),
            "ledger",
            "same",
            "green vault",
            0.70,
            0.66,
            0,
            steps,
            23,
        ),
        FactSpec(
            "branch_root",
            "north route",
            (
                "For normal weather, the Orion convoy uses the north route.",
                "The default Orion convoy path is the north route.",
            ),
            "orion",
            "default",
            "north route",
            0.58,
            0.56,
            0,
            steps,
            31,
        ),
        FactSpec(
            "branch_up",
            "east route",
            (
                "When the river rises, the Orion convoy must use the east route.",
                "Under flood context, Orion switches to the east route.",
            ),
            "orion",
            "flood",
            "east route",
            0.74,
            0.72,
            steps // 3,
            steps,
            19,
        ),
        FactSpec(
            "branch_down",
            "west route",
            (
                "When the bridge closes, the Orion convoy must use the west route.",
                "Under bridge closure context, Orion switches to the west route.",
            ),
            "orion",
            "bridge",
            "west route",
            0.74,
            0.72,
            steps // 3,
            steps,
            20,
        ),
        FactSpec(
            "rare_critical",
            "ember seven",
            (
                "The emergency shutdown phrase is ember seven.",
                "For emergency shutdown, the required phrase is ember seven.",
            ),
            "safety",
            "rare",
            "ember seven",
            1.30,
            1.50,
            0,
            steps,
            53,
        ),
        FactSpec(
            "obsolete_old",
            "silver key",
            (
                "The old lab door code was the silver key.",
                "Earlier records listed the lab door code as the silver key.",
            ),
            "lab_code",
            "old",
            "silver key",
            0.55,
            0.50,
            0,
            (2 * steps) // 3,
            17,
        ),
        FactSpec(
            "replacement",
            "gold key",
            (
                "The corrected lab door code is the gold key.",
                "Updated records replace the lab door code with the gold key.",
            ),
            "lab_code",
            "current",
            "gold key",
            1.00,
            0.86,
            steps // 2,
            steps,
            17,
            contradiction_target="obsolete_old",
        ),
        FactSpec(
            "novel",
            "kyoto lab",
            (
                "Leona's new robotics workspace is the Kyoto lab.",
                "The latest note says Leona works from the Kyoto lab.",
            ),
            "workspace",
            "new",
            "kyoto lab",
            0.72,
            0.70,
            (2 * steps) // 3,
            steps,
            14,
        ),
        FactSpec(
            "noise",
            "unknown",
            (
                "An unverified scrap claims the moon archive smells like purple clocks.",
                "A corrupted note says the mirror cabinet sings about triangle rain.",
                "A low trust message lists random symbols instead of a usable fact.",
            ),
            "noise",
            "untrusted",
            "unknown",
            0.06,
            0.04,
            0,
            steps,
            11,
        ),
    ]


TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize_text(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        raise ValueError(f"Text produced no tokens: {text!r}")
    return tokens


def build_vocab(specs: list[FactSpec]) -> dict[str, int]:
    words = {"<pad>", "<unk>"}
    for spec in specs:
        for text in spec.texts:
            words.update(tokenize_text(text))
        words.update(tokenize_text(f"What is the answer for {spec.group} {spec.route_family} {spec.route_branch}?"))
    words.update(tokenize_text("What is the current answer?"))
    return {word: index for index, word in enumerate(sorted(words))}


def encode_text(text: str, vocab: dict[str, int], config: Config, device: torch.device) -> torch.Tensor:
    words = tokenize_text(text)
    ids = [vocab.get(word, vocab["<unk>"]) for word in words[: config.seq_len]]
    ids.extend([vocab["<pad>"]] * (config.seq_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long, device=device)


def build_events(config: Config, device: torch.device) -> tuple[list[TextEvent], dict[str, TextEvent], dict[str, int]]:
    specs = fact_specs(config.steps)
    vocab = build_vocab(specs)
    events: list[TextEvent] = []
    for step in range(config.steps):
        step_events: list[tuple[int, FactSpec, str]] = []
        for spec in specs:
            if not (spec.start <= step < spec.end):
                continue
            if step % spec.period == 0:
                text_index = (step // spec.period) % len(spec.texts)
                priority = 1 if spec.group == "noise" else 0
                step_events.append((priority, spec, spec.texts[text_index]))
        step_events.sort(key=lambda row: (row[0], -row[1].dependency, -row[1].usefulness, row[1].group))
        for _, spec, text in step_events:
            events.append(
                TextEvent(
                    tokens=encode_text(text, vocab, config, device),
                    text=text,
                    y=ANSWER_LABELS[spec.answer],
                    group=spec.group,
                    route_family=spec.route_family,
                    route_branch=spec.route_branch,
                    route_leaf=spec.route_leaf,
                    usefulness=spec.usefulness,
                    dependency=spec.dependency,
                    contradiction_target=spec.contradiction_target,
                    step=step,
                )
            )
    if not events:
        raise RuntimeError("Event stream is empty.")
    probes: dict[str, TextEvent] = {}
    for spec in specs:
        probe_text = spec.texts[-1]
        probes[spec.group] = TextEvent(
            tokens=encode_text(probe_text, vocab, config, device),
            text=probe_text,
            y=ANSWER_LABELS[spec.answer],
            group=spec.group,
            route_family=spec.route_family,
            route_branch=spec.route_branch,
            route_leaf=spec.route_leaf,
            usefulness=spec.usefulness,
            dependency=spec.dependency,
            contradiction_target=spec.contradiction_target,
            step=config.steps,
        )
    return events, probes, vocab


def group_label_map() -> dict[str, int]:
    return {spec.group: ANSWER_LABELS[spec.answer] for spec in fact_specs(2)}


def randn_on_device(*shape: int, generator: torch.Generator, device: torch.device, scale: float) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float32, device="cpu").to(device) * scale


def event_write_gate(event: TextEvent, config: Config) -> float:
    potential = event.usefulness + event.dependency
    scaled = (potential - config.write_gate_threshold) / config.write_gate_temperature
    if scaled >= 0.0:
        return float(1.0 / (1.0 + math.exp(-scaled)))
    exp_value = math.exp(scaled)
    return float(exp_value / (1.0 + exp_value))


def initialize_node(node: NodeState, feature: torch.Tensor, event: TextEvent, config: Config) -> None:
    node.active = True
    node.center = F.normalize(feature.detach().clone(), dim=0)
    node.depth = 0.0
    node.strength = 0.55
    node.evidence = 1.0
    node.usefulness = event.usefulness
    node.dependency = event.dependency
    node.conflict = 0.0
    node.downstream_support = 0.0
    node.admitted = False
    node.age = 0
    node.updates = 1
    node.dominant_group = event.group
    node.group_mass = {event.group: 1.0}
    node.support_mass = {event.route_family: 1.0}
    update_node_admission(node, config)
    if node_potential(node) < config.min_consolidation_potential:
        node.strength = min(node.strength, config.low_potential_strength_cap)
    adjust_node_depth(node, config, multiplier=1.0)


def update_node(node: NodeState, feature: torch.Tensor, event: TextEvent, config: Config) -> None:
    node.evidence = 0.97 * node.evidence + 1.0
    node.usefulness = 0.92 * node.usefulness + 0.08 * event.usefulness
    node.dependency = max(0.96 * node.dependency, event.dependency)
    node.conflict *= 0.90
    node.strength = min(10.0, 0.985 * node.strength + 0.20 + 0.12 * event.usefulness + 0.10 * event.dependency)
    node.age = 0
    node.updates += 1
    node.group_mass[event.group] = node.group_mass.get(event.group, 0.0) + 1.0
    node.support_mass[event.route_family] = node.support_mass.get(event.route_family, 0.0) + 1.0
    node.dominant_group = max(node.group_mass.items(), key=lambda row: (row[1], row[0]))[0]
    update_node_admission(node, config)
    if node_potential(node) < config.min_consolidation_potential:
        node.strength = min(node.strength, config.low_potential_strength_cap)
    adjust_node_depth(node, config, multiplier=1.0)
    if node.center is None:
        raise RuntimeError("Active node is missing center during update.")
    rate = config.center_lr * math.exp(-0.35 * node.depth)
    node.center = F.normalize((1.0 - rate) * node.center + rate * feature.detach(), dim=0)


@dataclass
class TreeNode:
    layer: int
    slot: int
    path: tuple[int, ...]
    W: torch.Tensor
    b: torch.Tensor
    state: NodeState
    children: list["TreeNode"]
    head_W: torch.Tensor | None = None
    head_b: torch.Tensor | None = None
    memory: list[MemoryItem] | None = None


class MultiLayerNestedTextNet:
    def __init__(self, config: Config, vocab_size: int, classes: int, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.classes = classes
        self.slots = parse_slots(config.slots_per_layer)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 911)
        self.token_embedding = randn_on_device(vocab_size, config.d_model, generator=generator, device=device, scale=0.12)
        self.position_embedding = randn_on_device(config.seq_len, config.d_model, generator=generator, device=device, scale=0.03)
        self.encoder_w1 = randn_on_device(config.d_model, config.hidden, generator=generator, device=device, scale=0.14)
        self.encoder_b1 = torch.zeros(config.hidden, device=device)
        self.encoder_w2 = randn_on_device(config.hidden, config.hidden, generator=generator, device=device, scale=0.14)
        self.encoder_b2 = torch.zeros(config.hidden, device=device)
        self.root = self._make_children(parent_path=(), layer=0, generator=generator)

    def _new_transform(self, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        W = randn_on_device(self.config.hidden, self.config.hidden, generator=generator, device=self.device, scale=0.14)
        b = torch.zeros(self.config.hidden, device=self.device)
        return W, b

    def _make_children(self, parent_path: tuple[int, ...], layer: int, generator: torch.Generator) -> list[TreeNode]:
        nodes: list[TreeNode] = []
        for slot in range(self.slots[layer]):
            W, b = self._new_transform(generator)
            path = (*parent_path, slot)
            node = TreeNode(
                layer=layer,
                slot=slot,
                path=path,
                W=W,
                b=b,
                state=NodeState(index=slot),
                children=[],
            )
            if layer == len(self.slots) - 1:
                node.head_W = randn_on_device(self.config.hidden, self.classes, generator=generator, device=self.device, scale=0.12)
                node.head_b = torch.zeros(self.classes, device=self.device)
                node.memory = []
            else:
                node.children = self._make_children(path, layer + 1, generator)
            nodes.append(node)
        return nodes

    def shared_params(self) -> list[torch.Tensor]:
        return [
            self.token_embedding,
            self.position_embedding,
            self.encoder_w1,
            self.encoder_b1,
            self.encoder_w2,
            self.encoder_b2,
        ]

    def encode(self, tokens: torch.Tensor, shared_params: list[torch.Tensor] | None = None) -> torch.Tensor:
        if shared_params is None:
            tok, pos, w1, b1, w2, b2 = self.shared_params()
        else:
            tok, pos, w1, b1, w2, b2 = shared_params
        emb = F.embedding(tokens, tok) + pos[: tokens.shape[0]]
        pooled = emb.mean(dim=0)
        hidden = torch.tanh(pooled @ w1 + b1)
        return torch.tanh(hidden @ w2 + b2)

    def choose_child(
        self,
        children: list[TreeNode],
        feature: torch.Tensor,
        *,
        match_threshold: float,
        branch_threshold: float,
        route_family: str,
        family_routing_bias: float,
    ) -> tuple[TreeNode, str, float, float]:
        active = [child for child in children if child.state.active]
        if not active:
            return children[0], "new", math.inf, -math.inf
        ranked = []
        for child in active:
            if child.state.center is None:
                raise RuntimeError(f"Active node {child.path} has no center.")
            distance = cosine_distance(feature, child.state.center)
            support_total = sum(child.state.support_mass.values())
            family_share = child.state.support_mass.get(route_family, 0.0) / support_total if support_total > 0.0 else 0.0
            score = distance - family_routing_bias * family_share
            ranked.append((score, distance, -node_survival(child.state, self.config), child.slot, child))
        ranked.sort(key=lambda row: (row[0], row[2], row[3]))
        best_score, best_distance, negative_survival, _, best = ranked[0]
        free = next((child for child in children if not child.state.active), None)
        if best_score <= match_threshold:
            return best, "matched", best_distance, -negative_survival
        if best_score >= branch_threshold and free is not None:
            return free, "new_branch", best_distance, -negative_survival
        if free is not None:
            return free, "new", best_distance, -negative_survival
        weakest = min(
            active,
            key=lambda child: (
                child.state.admitted,
                node_survival(child.state, self.config),
                child.state.strength,
                -child.state.age,
                child.slot,
            ),
        )
        return weakest, "overwrite_low_survival", best_distance, node_survival(weakest.state, self.config)

    def route(self, event: TextEvent) -> tuple[list[TreeNode], list[str], list[float], list[float]]:
        feature = self.encode(event.tokens)
        children = self.root
        path: list[TreeNode] = []
        actions: list[str] = []
        distances: list[float] = []
        survivals: list[float] = []
        for layer in range(len(self.slots)):
            match_threshold = self.config.family_match_threshold if layer < self.config.family_prefix_length else self.config.match_threshold
            branch_threshold = self.config.family_branch_threshold if layer < self.config.family_prefix_length else self.config.branch_threshold
            family_routing_bias = self.config.family_routing_bias if layer < self.config.family_prefix_length else 0.0
            selected, action, distance, survival = self.choose_child(
                children,
                feature,
                match_threshold=match_threshold,
                branch_threshold=branch_threshold,
                route_family=event.route_family,
                family_routing_bias=family_routing_bias,
            )
            if action == "matched" and layer >= self.config.family_prefix_length and event.contradiction_target is None:
                labels_by_group = group_label_map()
                known_labels = {labels_by_group[group] for group in selected.state.group_mass if group in labels_by_group}
                free = next((child for child in children if not child.state.active), None)
                if known_labels and event.y not in known_labels and free is not None:
                    selected = free
                    action = "answer_conflict_branch"
            if not selected.state.active or action in {"new", "new_branch", "overwrite_low_survival"}:
                initialize_node(selected.state, feature, event, self.config)
                self.release_descendants(selected)
            elif action == "answer_conflict_branch":
                initialize_node(selected.state, feature, event, self.config)
                self.release_descendants(selected)
            path.append(selected)
            actions.append(action)
            distances.append(distance)
            survivals.append(survival)
            feature = torch.tanh(feature @ selected.W + selected.b)
            children = selected.children
        return path, actions, distances, survivals

    def release_descendants(self, node: TreeNode) -> None:
        for child in node.children:
            self.release_subtree(child)

    def release_subtree(self, node: TreeNode) -> None:
        node.state.active = False
        node.state.center = None
        node.state.depth = 0.0
        node.state.strength = 0.0
        node.state.evidence = 0.0
        node.state.usefulness = 0.0
        node.state.dependency = 0.0
        node.state.conflict = 0.0
        node.state.downstream_support = 0.0
        node.state.admitted = False
        node.state.age = 0
        node.state.updates = 0
        node.state.dominant_group = "empty"
        node.state.group_mass = {}
        node.state.support_mass = {}
        node.state.released_count += 1
        if node.memory is not None:
            node.memory = []
        for child in node.children:
            self.release_subtree(child)

    def path_params(self, path: list[TreeNode]) -> list[torch.Tensor]:
        leaf = path[-1]
        if leaf.head_W is None or leaf.head_b is None:
            raise RuntimeError("Selected path does not end in a leaf head.")
        params = self.shared_params()
        for node in path:
            params.extend([node.W, node.b])
        params.extend([leaf.head_W, leaf.head_b])
        return params

    def forward_path(
        self,
        tokens: torch.Tensor,
        path: list[TreeNode],
        params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if params is None:
            feature = self.encode(tokens)
            for node in path:
                feature = torch.tanh(feature @ node.W + node.b)
            leaf = path[-1]
            if leaf.head_W is None or leaf.head_b is None:
                raise RuntimeError("Selected path does not end in a leaf head.")
            return feature @ leaf.head_W + leaf.head_b
        shared = params[:6]
        feature = self.encode(tokens, shared_params=shared)
        offset = 6
        for _ in path:
            W = params[offset]
            b = params[offset + 1]
            feature = torch.tanh(feature @ W + b)
            offset += 2
        return feature @ params[offset] + params[offset + 1]

    def loss_for_items(
        self,
        path: list[TreeNode],
        items: list[TextEvent | MemoryItem],
        params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not items:
            raise ValueError("Cannot compute loss over an empty item list.")
        losses = []
        for item in items:
            logits = self.forward_path(item.tokens, path, params=params)
            target = torch.tensor([item.y], dtype=torch.long, device=self.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        return torch.stack(losses).mean()

    def protected_items(self, leaf: TreeNode, event: TextEvent) -> list[MemoryItem]:
        if leaf.memory is None:
            raise RuntimeError("Leaf memory is missing.")
        protected = [
            item
            for item in leaf.memory
            if item.group != event.group
            and item.group != event.contradiction_target
            and item.y != event.y
            and item.usefulness + item.dependency >= self.config.protected_min_potential
        ]
        return protected[-self.config.memory_limit :]

    def train_path(self, path: list[TreeNode], event: TextEvent) -> dict[str, float]:
        params = self.path_params(path)
        leaf = path[-1]
        protected = self.protected_items(leaf, event)
        write_gate = event_write_gate(event, self.config)
        diagnostics = {
            "loss": math.nan,
            "write_gate": write_gate,
            "protected_rows": 0.0,
            "removed_fraction": 0.0,
            "safe_fraction": 1.0,
            "restore_ratio": 0.0,
        }
        for _ in range(self.config.inner_steps):
            grad_params = [param.detach().clone().requires_grad_(True) for param in params]
            loss_new = self.loss_for_items(path, [event], params=grad_params)
            raw_gradient = grad_or_raise(loss_new, grad_params)
            safe_gradient = raw_gradient
            restore_gradient = torch.zeros_like(raw_gradient)
            projection_stats = {"rows": 0.0, "removed_fraction": 0.0, "safe_fraction": 1.0}
            if protected:
                rows = []
                for item in protected:
                    protected_params = [param.detach().clone().requires_grad_(True) for param in params]
                    protected_loss = self.loss_for_items(path, [item], params=protected_params)
                    rows.append(grad_or_raise(protected_loss, protected_params).detach())
                safe_gradient, projection_stats = project_gradient(
                    raw_gradient,
                    rows,
                    self.config.tangent_damping,
                    self.config.projection_device,
                )
                restore_params = [param.detach().clone().requires_grad_(True) for param in params]
                restore_loss = self.loss_for_items(path, protected, params=restore_params)
                restore_gradient = grad_or_raise(restore_loss, restore_params).detach()
                restore_gradient = clip_relative(restore_gradient, safe_gradient, self.config.restore_clip_ratio)
            final_gradient = write_gate * (safe_gradient + self.config.restore_weight * restore_gradient)
            depth = max(node.state.depth for node in path)
            strength = max(node.state.strength for node in path)
            base_lr = self.config.base_lr * math.exp(-self.config.depth_lr_decay * depth) * math.exp(
                -self.config.strength_lr_decay * strength
            )
            pieces = unflatten_like(final_gradient, params)
            with torch.no_grad():
                for index in range(6):
                    params[index] -= base_lr * self.config.shared_lr_multiplier * pieces[index]
                offset = 6
                for _ in path:
                    params[offset] -= base_lr * self.config.path_lr_multiplier * pieces[offset]
                    params[offset + 1] -= base_lr * self.config.path_lr_multiplier * pieces[offset + 1]
                    offset += 2
                params[offset] -= base_lr * self.config.head_lr_multiplier * pieces[offset]
                params[offset + 1] -= base_lr * self.config.head_lr_multiplier * pieces[offset + 1]
            raw_norm = torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
            diagnostics["loss"] = float(loss_new.detach().cpu())
            diagnostics["protected_rows"] = projection_stats["rows"]
            diagnostics["removed_fraction"] = projection_stats["removed_fraction"]
            diagnostics["safe_fraction"] = projection_stats["safe_fraction"]
            diagnostics["restore_ratio"] = float((torch.linalg.vector_norm(self.config.restore_weight * restore_gradient) / raw_norm).detach().cpu())
        return diagnostics

    def commit(self, path: list[TreeNode], event: TextEvent, write_gate: float) -> bool:
        if write_gate < self.config.commit_gate_threshold:
            return False
        feature = self.encode(event.tokens)
        for node in path:
            update_node(node.state, feature, event, self.config)
            feature = torch.tanh(feature @ node.W + node.b)
        leaf = path[-1]
        if leaf.memory is None:
            raise RuntimeError("Selected leaf has no memory list.")
        leaf.memory.append(
            MemoryItem(
                tokens=event.tokens.detach().clone(),
                y=event.y,
                group=event.group,
                usefulness=event.usefulness,
                dependency=event.dependency,
            )
        )
        if len(leaf.memory) > self.config.memory_limit:
            leaf.memory = leaf.memory[-self.config.memory_limit :]
        return True

    def apply_contradiction(self, event: TextEvent) -> tuple[int, float]:
        if event.contradiction_target is None:
            return 0, 0.0
        touched = 0
        pressure_sum = 0.0
        to_release: list[TreeNode] = []
        for leaf in self.leaves():
            state = leaf.state
            if not state.active:
                continue
            share = node_share(state, event.contradiction_target)
            if share <= 0.0:
                continue
            pressure = share * share * (1.0 + event.usefulness + event.dependency)
            state.conflict = 0.82 * state.conflict + pressure
            old_mass = state.group_mass.get(event.contradiction_target, 0.0)
            reduced = old_mass * max(0.0, 1.0 - 0.45 * share)
            if reduced < 0.15:
                del state.group_mass[event.contradiction_target]
            else:
                state.group_mass[event.contradiction_target] = reduced
            state.dominant_group = max(state.group_mass.items(), key=lambda row: (row[1], row[0]))[0] if state.group_mass else "empty"
            touched += 1
            pressure_sum += pressure
            if share >= 0.50 and event.usefulness + event.dependency > state.usefulness + state.dependency:
                to_release.append(leaf)
        for leaf in to_release:
            self.release_subtree(leaf)
        return touched, pressure_sum

    def decay_idle(self, selected_path: tuple[int, ...]) -> int:
        released = 0
        for node in self.all_nodes():
            if not node.state.active or node.path == selected_path:
                continue
            node.state.age += 1
            decay = 0.035 * math.exp(-0.72 * node.state.depth)
            if not node.state.admitted:
                decay *= 2.10
            node.state.strength *= max(0.0, 1.0 - decay)
            node.state.evidence *= max(0.0, 1.0 - 0.30 * decay)
            node.state.conflict *= 0.985
            node.state.downstream_support *= self.config.support_decay
            update_node_admission(node.state, self.config)
            adjust_node_depth(node.state, self.config, multiplier=0.16)
            if (not node.state.admitted) and node_survival(node.state, self.config) < self.config.release_threshold and node.state.strength < 0.40:
                self.release_subtree(node)
                released += 1
        return released

    def all_nodes(self) -> list[TreeNode]:
        nodes: list[TreeNode] = []

        def walk(children: list[TreeNode]) -> None:
            for node in children:
                nodes.append(node)
                walk(node.children)

        walk(self.root)
        return nodes

    def leaves(self) -> list[TreeNode]:
        return [node for node in self.all_nodes() if not node.children]

    def predict(self, tokens: torch.Tensor) -> tuple[int, tuple[int, ...], float, float, bool]:
        candidates: list[tuple[float, float, list[TreeNode]]] = []

        def walk(children: list[TreeNode], feature: torch.Tensor, path: list[TreeNode], distance_sum: float) -> None:
            for node in children:
                if not node.state.active:
                    continue
                if node.state.center is None:
                    raise RuntimeError(f"Active node {node.path} has no center.")
                distance = cosine_distance(feature, node.state.center)
                next_feature = torch.tanh(feature @ node.W + node.b)
                next_path = [*path, node]
                if not node.children:
                    candidates.append((distance_sum + distance, -node_survival(node.state, self.config), next_path))
                else:
                    walk(node.children, next_feature, next_path, distance_sum + distance)

        z = self.encode(tokens)
        walk(self.root, z, [], 0.0)
        if not candidates:
            return ANSWER_LABELS["unknown"], (), 0.0, math.inf, True
        candidates.sort(key=lambda row: (row[0], row[1], [node.slot for node in row[2]]))
        distance_sum, _, path = candidates[0]
        logits = self.forward_path(tokens, path)
        probs = torch.softmax(logits, dim=0)
        pred = int(torch.argmax(probs).detach().cpu())
        confidence = float(torch.max(probs).detach().cpu())
        route_distance = float(distance_sum / max(1, len(path)))
        rejected = route_distance > self.config.read_reject_distance or confidence < self.config.read_reject_confidence
        if rejected:
            pred = ANSWER_LABELS["unknown"]
        return pred, tuple(node.slot for node in path), confidence, route_distance, rejected


def squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, dim=0)
    b_norm = F.normalize(b, dim=0)
    return 1.0 - torch.dot(a_norm, b_norm)


def pretrain_foundation(model: MultiLayerNestedTextNet, probes: dict[str, TextEvent], config: Config) -> None:
    train_items = [event for event in probes.values() if event.group != "noise"]
    if not train_items:
        raise RuntimeError("Foundation pretraining has no non-noise examples.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1207)
    head_w = randn_on_device(config.hidden, len(ANSWER_LABELS), generator=generator, device=model.device, scale=0.10)
    head_b = torch.zeros(len(ANSWER_LABELS), device=model.device)
    params = model.shared_params() + [head_w, head_b]
    for _ in range(config.foundation_epochs):
        trainable = [param.detach().clone().requires_grad_(True) for param in params]
        encoded: dict[str, torch.Tensor] = {}
        losses = []
        for event in train_items:
            z = model.encode(event.tokens, shared_params=trainable[:6])
            encoded[event.group] = z
            logits = z @ trainable[6] + trainable[7]
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([event.y], dtype=torch.long, device=model.device)))
        geometry_terms = []
        geometry_terms.append(squared_distance(encoded["merge_a"], encoded["merge_b"]).pow(2))
        for left_index, left_event in enumerate(train_items):
            for right_event in train_items[left_index + 1 :]:
                distance = squared_distance(encoded[left_event.group], encoded[right_event.group])
                if left_event.route_family == right_event.route_family:
                    geometry_terms.append(F.relu(distance - 0.10).pow(2))
                else:
                    geometry_terms.append(F.relu(torch.tensor(0.42, device=model.device) - distance).pow(2))
        geometry_terms.append(F.relu(torch.tensor(0.08, device=model.device) - squared_distance(encoded["obsolete_old"], encoded["replacement"])).pow(2))
        noise_z = model.encode(probes["noise"].tokens, shared_params=trainable[:6])
        for event in train_items:
            geometry_terms.append(F.relu(torch.tensor(0.55, device=model.device) - squared_distance(noise_z, encoded[event.group])).pow(2))
        prediction_loss = torch.stack(losses).mean()
        geometry_loss = torch.stack(geometry_terms).mean()
        loss = prediction_loss + config.foundation_geometry_weight * geometry_loss
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError("Foundation loss became non-finite.")
        gradient = grad_or_raise(loss, trainable)
        pieces = unflatten_like(gradient, params)
        with torch.no_grad():
            for param, piece in zip(params, pieces, strict=True):
                param -= config.foundation_lr * piece


def evaluate(model: MultiLayerNestedTextNet, probes: dict[str, TextEvent], config: Config) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    for group, event in probes.items():
        pred, path, confidence, route_distance, rejected = model.predict(event.tokens)
        correct = float(pred == event.y)
        rows.append(
            {
                "group": group,
                "text": event.text,
                "label": event.y,
                "pred": pred,
                "path": ".".join(str(part) for part in path),
                "correct": correct,
                "confidence": confidence,
                "route_distance": route_distance,
                "rejected": int(rejected),
            }
        )
    by_group = {str(row["group"]): row for row in rows}
    protected_groups = ("stable", "merge_a", "merge_b", "branch_root", "branch_up", "branch_down", "rare_critical")
    protected_acc = float(np.mean([float(by_group[group]["correct"]) for group in protected_groups]))
    branch_acc = float(np.mean([float(by_group[group]["correct"]) for group in ("branch_root", "branch_up", "branch_down")]))
    merge_same = float(by_group["merge_a"]["path"] == by_group["merge_b"]["path"])
    branch_paths = [str(by_group[group]["path"]).split(".") for group in ("branch_root", "branch_up", "branch_down")]
    family_prefix = {".".join(path[: config.family_prefix_length]) for path in branch_paths}
    leaf_prefix = {".".join(path[: config.leaf_prefix_length]) for path in branch_paths}
    family_cohesion = float(len(family_prefix) == 1)
    leaf_separation = float(len(leaf_prefix) / 3.0)
    full_leaf_separation = float(len(leaf_prefix) == 3)
    replacement_beats_obsolete = float(float(by_group["replacement"]["correct"]) == 1.0 and float(by_group["obsolete_old"]["correct"]) == 0.0)
    noise_rejected = float(float(by_group["noise"]["confidence"]) <= config.noise_conf_threshold)
    rare_survives = float(float(by_group["rare_critical"]["correct"]) == 1.0)
    stable_survives = float(float(by_group["stable"]["correct"]) == 1.0)
    semantic_score = float(
        np.mean(
            [
                protected_acc,
                merge_same,
                family_cohesion,
                leaf_separation,
                replacement_beats_obsolete,
                noise_rejected,
                rare_survives,
                stable_survives,
            ]
        )
    )
    strict_semantic_score = float(
        np.mean(
            [
                protected_acc,
                merge_same,
                family_cohesion,
                full_leaf_separation,
                replacement_beats_obsolete,
                noise_rejected,
                rare_survives,
                stable_survives,
            ]
        )
    )
    return rows, {
        "protected_acc": protected_acc,
        "branch_acc": branch_acc,
        "compatible_merge": merge_same,
        "branch_family_cohesion": family_cohesion,
        "branch_leaf_separation": leaf_separation,
        "replacement_beats_obsolete": replacement_beats_obsolete,
        "noise_confidence": float(by_group["noise"]["confidence"]),
        "noise_rejected": noise_rejected,
        "rare_survival": rare_survives,
        "stable_survival": stable_survives,
        "semantic_score": semantic_score,
        "strict_semantic_score": strict_semantic_score,
    }


def node_rows(model: MultiLayerNestedTextNet) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node in model.all_nodes():
        rows.append(
            {
                "path": ".".join(str(part) for part in node.path),
                "layer": node.layer,
                "slot": node.slot,
                "active": int(node.state.active),
                "dominant_group": node.state.dominant_group,
                "dominant_share": node.state.dominant_share(),
                "depth": node.state.depth,
                "strength": node.state.strength,
                "evidence": node.state.evidence,
                "usefulness": node.state.usefulness,
                "dependency": node.state.dependency,
                "conflict": node.state.conflict,
                "admitted": int(node.state.admitted),
                "age": node.state.age,
                "survival": node_survival(node.state, model.config),
                "memory": len(node.memory) if node.memory is not None else 0,
            }
        )
    return rows


def run_sequence(config: Config) -> tuple[MultiLayerNestedTextNet, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, float], dict[str, int]]:
    validate_config(config)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    events, probes, vocab = build_events(config, device)
    model = MultiLayerNestedTextNet(config, len(vocab), len(ANSWER_LABELS), device)
    if config.foundation_epochs > 0:
        pretrain_foundation(model, probes, config)
    event_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for index, event in enumerate(events):
        contradiction_nodes, contradiction_pressure = model.apply_contradiction(event)
        path, actions, distances, survivals = model.route(event)
        diagnostics = model.train_path(path, event)
        committed = model.commit(path, event, diagnostics["write_gate"])
        released = model.decay_idle(tuple(node.slot for node in path))
        active_nodes = [node for node in model.all_nodes() if node.state.active]
        depths = np.array([node.state.depth for node in active_nodes], dtype=np.float64)
        metric_rows.append(
            {
                "event_index": index,
                "step": event.step,
                "active_nodes": len(active_nodes),
                "active_leaves": sum(1 for leaf in model.leaves() if leaf.state.active),
                "mean_depth": float(np.mean(depths)) if len(depths) else 0.0,
                "mean_write_gate": diagnostics["write_gate"],
            }
        )
        event_rows.append(
            {
                "event_index": index,
                "step": event.step,
                "group": event.group,
                "text": event.text,
                "path": ".".join(str(node.slot) for node in path),
                "actions": ".".join(actions),
                "loss": diagnostics["loss"],
                "write_gate": diagnostics["write_gate"],
                "committed": int(committed),
                "protected_rows": diagnostics["protected_rows"],
                "removed_fraction": diagnostics["removed_fraction"],
                "safe_fraction": diagnostics["safe_fraction"],
                "restore_ratio": diagnostics["restore_ratio"],
                "contradiction_nodes": contradiction_nodes,
                "contradiction_pressure": contradiction_pressure,
                "released": released,
                "distances": ".".join("inf" if math.isinf(value) else f"{value:.4f}" for value in distances),
                "survivals": ".".join("-inf" if math.isinf(value) and value < 0 else f"{value:.4f}" for value in survivals),
            }
        )
    eval_rows, summary = evaluate(model, probes, config)
    return model, event_rows, metric_rows, eval_rows, summary, {"vocab_size": len(vocab), "events": len(events)}


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary(node_table: list[dict[str, object]], eval_rows: list[dict[str, object]], output_path: Path) -> None:
    active = [row for row in node_table if int(row["active"]) == 1]
    if not active:
        raise ValueError("Cannot plot because no active nodes exist.")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    layers = sorted({int(row["layer"]) for row in active})
    mean_depth = [np.mean([float(row["depth"]) for row in active if int(row["layer"]) == layer]) for layer in layers]
    active_counts = [sum(1 for row in active if int(row["layer"]) == layer) for layer in layers]
    axes[0].bar([str(layer) for layer in layers], mean_depth, color="#4c78a8")
    axes[0].set_title("Nested depth by layer")
    axes[0].set_ylabel("mean depth")
    axes[0].grid(axis="y", alpha=0.25)
    labels = [str(row["group"]) for row in eval_rows]
    correct = [float(row["correct"]) for row in eval_rows]
    colors = [GROUP_COLORS.get(str(row["group"]), "#333333") for row in eval_rows]
    axes[1].bar(labels, correct, color=colors)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Final probe correctness")
    axes[1].set_ylabel("correct")
    axes[1].tick_params(axis="x", rotation=35)
    for index, count in enumerate(active_counts):
        axes[0].text(index, mean_depth[index] + 0.03, f"n={count}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(
    model: MultiLayerNestedTextNet,
    event_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    summary: dict[str, float],
    run_info: dict[str, int],
    config: Config,
) -> dict[str, str]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    nodes = node_rows(model)
    artifacts = {
        "event_log": config.output_dir / "event_log.csv",
        "metrics": config.output_dir / "metrics_over_time.csv",
        "nodes": config.output_dir / "node_table.csv",
        "eval": config.output_dir / "eval_by_group.csv",
        "summary": config.output_dir / "real_text_multilayer_nested_summary.json",
        "plot": config.output_dir / "real_text_multilayer_nested_summary.png",
    }
    write_csv(artifacts["event_log"], event_rows)
    write_csv(artifacts["metrics"], metric_rows)
    write_csv(artifacts["nodes"], nodes)
    write_csv(artifacts["eval"], eval_rows)
    plot_summary(nodes, eval_rows, artifacts["plot"])
    payload = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "run_info": run_info,
        "summary": summary,
        "eval_rows": eval_rows,
        "node_rows": nodes,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    with artifacts["summary"].open("w") as handle:
        json.dump(payload, handle, indent=2)
    return {key: str(value) for key, value in artifacts.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--slots-per-layer", type=str, default="8,3,3,2,3")
    parser.add_argument("--seq-len", type=int, default=18)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--foundation-epochs", type=int, default=420)
    parser.add_argument("--foundation-lr", type=float, default=0.10)
    parser.add_argument("--foundation-geometry-weight", type=float, default=0.80)
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--memory-limit", type=int, default=32)
    parser.add_argument("--base-lr", type=float, default=0.10)
    parser.add_argument("--shared-lr-multiplier", type=float, default=0.035)
    parser.add_argument("--path-lr-multiplier", type=float, default=0.22)
    parser.add_argument("--head-lr-multiplier", type=float, default=1.00)
    parser.add_argument("--depth-lr-decay", type=float, default=0.58)
    parser.add_argument("--strength-lr-decay", type=float, default=0.05)
    parser.add_argument("--center-lr", type=float, default=0.16)
    parser.add_argument("--family-match-threshold", type=float, default=0.34)
    parser.add_argument("--family-branch-threshold", type=float, default=0.72)
    parser.add_argument("--family-routing-bias", type=float, default=0.00)
    parser.add_argument("--match-threshold", type=float, default=0.18)
    parser.add_argument("--branch-threshold", type=float, default=0.36)
    parser.add_argument("--family-prefix-length", type=int, default=2)
    parser.add_argument("--leaf-prefix-length", type=int, default=5)
    parser.add_argument("--evidence-gain", type=float, default=0.75)
    parser.add_argument("--usefulness-gain", type=float, default=1.20)
    parser.add_argument("--dependency-gain", type=float, default=1.25)
    parser.add_argument("--conflict-gain", type=float, default=1.85)
    parser.add_argument("--age-decay", type=float, default=0.012)
    parser.add_argument("--support-gain", type=float, default=0.45)
    parser.add_argument("--support-decay", type=float, default=0.985)
    parser.add_argument("--support-threshold", type=float, default=0.10)
    parser.add_argument("--min-consolidation-potential", type=float, default=0.85)
    parser.add_argument("--write-gate-threshold", type=float, default=0.65)
    parser.add_argument("--write-gate-temperature", type=float, default=0.35)
    parser.add_argument("--commit-gate-threshold", type=float, default=0.35)
    parser.add_argument("--admission-threshold", type=float, default=1.85)
    parser.add_argument("--admission-temperature", type=float, default=0.35)
    parser.add_argument("--inward-rate", type=float, default=0.025)
    parser.add_argument("--outward-rate", type=float, default=0.16)
    parser.add_argument("--survival-temperature", type=float, default=0.35)
    parser.add_argument("--provisional-depth-cap", type=float, default=0.85)
    parser.add_argument("--low-potential-depth-cap", type=float, default=0.35)
    parser.add_argument("--low-potential-strength-cap", type=float, default=1.25)
    parser.add_argument("--release-threshold", type=float, default=-0.65)
    parser.add_argument("--tangent-damping", type=float, default=1e-3)
    parser.add_argument("--restore-weight", type=float, default=0.20)
    parser.add_argument("--restore-clip-ratio", type=float, default=0.35)
    parser.add_argument("--protected-min-potential", type=float, default=1.00)
    parser.add_argument("--read-reject-distance", type=float, default=math.inf)
    parser.add_argument("--read-reject-confidence", type=float, default=0.00)
    parser.add_argument("--noise-conf-threshold", type=float, default=0.35)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--projection-device", choices=("cpu", "same"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/09_real_text_nested_geometry/results/gco-real-text-multilayer-nested-cl-seed0"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = Config(**vars(args))
    model, event_rows, metric_rows, eval_rows, summary, run_info = run_sequence(config)
    artifacts = write_outputs(model, event_rows, metric_rows, eval_rows, summary, run_info, config)
    print("\nREAL-TEXT MULTI-LAYER NESTED CL")
    print("=" * 152)
    print(
        f"events={run_info['events']} vocab={run_info['vocab_size']} slots={config.slots_per_layer} "
        f"protected={summary['protected_acc']:.4f} branch={summary['branch_acc']:.4f} "
        f"semantic={summary['semantic_score']:.4f} strict={summary['strict_semantic_score']:.4f} "
        f"merge={summary['compatible_merge']:.4f} family={summary['branch_family_cohesion']:.4f} "
        f"leafSep={summary['branch_leaf_separation']:.4f} replace={summary['replacement_beats_obsolete']:.4f} "
        f"rare={summary['rare_survival']:.4f} noise={summary['noise_confidence']:.4f}"
    )
    print("WROTE")
    print("-" * 152)
    for path in artifacts.values():
        print(path)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
