#!/usr/bin/env python3
"""Tokenized hierarchical nested continual-learning experiment.

This is the first bridge from fixed synthetic vectors to a small trainable
neural input stack:

    token ids -> token/position embeddings -> learned encoder
              -> outer region -> middle branch -> inner leaf

Routing centers live in learned hidden space. Updates can change the selected
leaf, its middle branch, its outer region, and the shared encoder slowly. The
experiment reports both behavior and semantic organization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    semantic_metrics,
    update_node_admission,
)
from gco_nested_geometry_dynamics import build_events  # noqa: E402
from gco_nested_tangent_optimizer_tiny_nn import (  # noqa: E402
    GROUP_COLORS,
    LABELS,
    RegionMemory,
    clip_relative,
    cosine_distance,
    flatten_tensors,
    grad_or_raise,
    manual_config,
    project_gradient,
    resolve_device,
    scenario_specs,
    unflatten_like,
)


GROUP_TEMPLATES = {
    "stable": ("entity", "stable", "keeps", "core"),
    "merge_a": ("entity", "merge", "variant", "alpha"),
    "merge_b": ("entity", "merge", "variant", "beta"),
    "branch_root": ("branch", "family", "root", "base"),
    "branch_up": ("branch", "family", "direction", "up"),
    "branch_down": ("branch", "family", "direction", "down"),
    "rare_critical": ("rare", "critical", "safety", "anchor"),
    "obsolete_old": ("change", "target", "old", "answer"),
    "replacement": ("change", "target", "new", "answer"),
    "novel": ("novel", "fresh", "new", "trace"),
    "noise": ("noise", "random", "untrusted", "trace"),
}


@dataclass(frozen=True)
class Config:
    seed: int
    scenario: str
    mode: str
    steps: int
    outer_regions: int
    middle_per_outer: int
    inner_per_middle: int
    vocab_extra_noise: int
    seq_len: int
    d_model: int
    hidden: int
    classes: int
    foundation_epochs: int
    foundation_lr: float
    foundation_geometry_weight: float
    routing_geometry_weight: float
    compatible_distance_target: float
    outer_family_distance_max: float
    branch_distance_min: float
    branch_distance_max: float
    noise_distance_min: float
    inner_steps: int
    memory_limit: int
    shell_count: int
    base_lr: float
    shared_lr_multiplier: float
    outer_lr_multiplier: float
    middle_lr_multiplier: float
    inner_lr_multiplier: float
    depth_lr_decay: float
    strength_lr_decay: float
    outer_center_lr: float
    middle_center_lr: float
    inner_center_lr: float
    outer_match_threshold: float
    outer_branch_threshold: float
    middle_match_threshold: float
    middle_branch_threshold: float
    inner_match_threshold: float
    inner_branch_threshold: float
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
    admission_threshold: float
    admission_temperature: float
    inward_rate: float
    outward_rate: float
    survival_temperature: float
    provisional_depth_cap: float
    low_potential_depth_cap: float
    low_potential_strength_cap: float
    release_threshold: float
    overwrite_margin: float
    tangent_damping: float
    restore_weight: float
    restore_clip_ratio: float
    protected_min_potential: float
    protect_same_label: bool
    semantic_label_branch: bool
    max_events_per_step: int
    noise_conf_threshold: float
    device: str
    projection_device: str
    output_dir: Path

    @property
    def regions(self) -> int:
        return self.outer_regions


@dataclass(frozen=True)
class TokenExample:
    tokens: torch.Tensor
    y: int
    group: str
    usefulness: float
    dependency: float
    contradiction_target: str | None
    step: int


def validate_config(config: Config) -> None:
    if config.mode not in {"nested_sgd", "nested_tangent"}:
        raise ValueError("mode must be nested_sgd or nested_tangent.")
    if config.scenario not in {"default", "long"}:
        raise ValueError("scenario must be default or long.")
    if config.outer_regions < 2:
        raise ValueError("outer_regions must be at least 2.")
    if config.middle_per_outer < 1:
        raise ValueError("middle_per_outer must be at least 1.")
    if config.inner_per_middle < 1:
        raise ValueError("inner_per_middle must be at least 1.")
    if config.seq_len < 4:
        raise ValueError("seq_len must be at least 4.")
    if config.d_model < 8:
        raise ValueError("d_model must be at least 8.")
    if config.hidden < 8:
        raise ValueError("hidden must be at least 8.")
    if config.classes < max(LABELS.values()) + 1:
        raise ValueError("classes must cover all synthetic labels.")
    if config.inner_steps < 1:
        raise ValueError("inner_steps must be at least 1.")
    if config.foundation_epochs < 0:
        raise ValueError("foundation_epochs must be non-negative.")
    if config.foundation_lr <= 0.0:
        raise ValueError("foundation_lr must be positive.")
    if config.foundation_geometry_weight < 0.0:
        raise ValueError("foundation_geometry_weight must be non-negative.")
    if config.routing_geometry_weight < 0.0:
        raise ValueError("routing_geometry_weight must be non-negative.")
    if config.outer_family_distance_max < 0.0:
        raise ValueError("outer_family_distance_max must be non-negative.")
    if config.branch_distance_min > config.branch_distance_max:
        raise ValueError("branch_distance_min cannot exceed branch_distance_max.")
    if config.memory_limit < 1:
        raise ValueError("memory_limit must be at least 1.")
    if config.max_events_per_step < 1:
        raise ValueError("max_events_per_step must be at least 1.")
    if config.write_gate_temperature <= 0.0:
        raise ValueError("write_gate_temperature must be positive.")
    if config.write_gate_threshold < 0.0:
        raise ValueError("write_gate_threshold must be non-negative.")
    if config.projection_device not in {"cpu", "same"}:
        raise ValueError("projection_device must be cpu or same.")


def build_vocab(config: Config) -> dict[str, int]:
    tokens = {"<pad>"}
    for template in GROUP_TEMPLATES.values():
        tokens.update(template)
    for index in range(config.vocab_extra_noise):
        tokens.add(f"noise_{index}")
    return {token: index for index, token in enumerate(sorted(tokens))}


def encode_template(group: str, step: int, vocab: dict[str, int], config: Config) -> torch.Tensor:
    template = list(GROUP_TEMPLATES[group])
    if group == "noise":
        template = [
            "noise",
            f"noise_{step % config.vocab_extra_noise}",
            f"noise_{(step * 7 + 3) % config.vocab_extra_noise}",
            "untrusted",
        ]
    if len(template) > config.seq_len:
        raise ValueError(f"Template for {group!r} is longer than seq_len.")
    ids = [vocab[token] for token in template]
    ids.extend([vocab["<pad>"]] * (config.seq_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def build_token_examples(config: Config, device: torch.device) -> tuple[list[TokenExample], dict[str, torch.Tensor], dict[str, int]]:
    specs = scenario_specs(config)
    events = build_events(manual_config(config), specs)
    vocab = build_vocab(config)
    examples: list[TokenExample] = []
    for event in events:
        if event.group == "noise":
            label = (event.step + int(np.argmax(event.vector))) % config.classes
        else:
            label = LABELS[event.group]
        examples.append(
            TokenExample(
                tokens=encode_template(event.group, event.step, vocab, config).to(device),
                y=int(label),
                group=event.group,
                usefulness=event.usefulness,
                dependency=event.dependency,
                contradiction_target=event.contradiction_target,
                step=event.step,
            )
        )
    prototypes = {
        group: encode_template(group, config.steps + index, vocab, config).to(device)
        for index, group in enumerate(GROUP_TEMPLATES)
    }
    return examples, prototypes, vocab


def initialize_node_from_feature(node: NodeState, feature: torch.Tensor, example: TokenExample, config: Config) -> None:
    node.active = True
    node.center = F.normalize(feature.detach().clone(), dim=0)
    node.depth = 0.0
    node.strength = 0.55
    node.evidence = 1.0
    node.usefulness = example.usefulness
    node.dependency = example.dependency
    node.conflict = 0.0
    node.downstream_support = 0.0
    node.admitted = False
    node.age = 0
    node.updates = 1
    node.dominant_group = example.group
    node.group_mass = {example.group: 1.0}
    node.support_mass = {}
    update_node_admission(node, config)
    if node_potential(node) < config.min_consolidation_potential:
        node.strength = min(node.strength, config.low_potential_strength_cap)
    adjust_node_depth(node, config, multiplier=1.0)


def update_node_metadata(node: NodeState, example: TokenExample, config: Config) -> None:
    node.evidence = 0.97 * node.evidence + 1.0
    node.usefulness = 0.92 * node.usefulness + 0.08 * example.usefulness
    node.dependency = max(0.96 * node.dependency, example.dependency)
    node.conflict *= 0.90
    node.strength = min(10.0, 0.985 * node.strength + 0.20 + 0.12 * example.usefulness + 0.10 * example.dependency)
    node.age = 0
    node.updates += 1
    node.group_mass[example.group] = node.group_mass.get(example.group, 0.0) + 1.0
    node.dominant_group = max(node.group_mass.items(), key=lambda row: (row[1], row[0]))[0]
    update_node_admission(node, config)
    if node_potential(node) < config.min_consolidation_potential:
        node.strength = min(node.strength, config.low_potential_strength_cap)
    adjust_node_depth(node, config, multiplier=1.0)


def event_write_gate(example: TokenExample, config: Config) -> float:
    potential = example.usefulness + example.dependency
    scaled = (potential - config.write_gate_threshold) / config.write_gate_temperature
    if scaled >= 0.0:
        inv = math.exp(-scaled)
        return float(1.0 / (1.0 + inv))
    exp_value = math.exp(scaled)
    return float(exp_value / (1.0 + exp_value))


class TokenHierarchicalNestedNet:
    def __init__(self, config: Config, vocab_size: int, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 1777)
        self.token_embedding = torch.randn(vocab_size, config.d_model, generator=generator, dtype=torch.float32, device=device) * 0.12
        self.position_embedding = torch.randn(config.seq_len, config.d_model, generator=generator, dtype=torch.float32, device=device) * 0.03
        self.encoder_w1 = torch.randn(config.d_model, config.hidden, generator=generator, dtype=torch.float32, device=device) * 0.16
        self.encoder_b1 = torch.zeros(config.hidden, dtype=torch.float32, device=device)
        self.encoder_w2 = torch.randn(config.hidden, config.hidden, generator=generator, dtype=torch.float32, device=device) * 0.16
        self.encoder_b2 = torch.zeros(config.hidden, dtype=torch.float32, device=device)
        self.w_outer = [
            torch.randn(config.hidden, config.hidden, generator=generator, dtype=torch.float32, device=device) * 0.16
            for _ in range(config.outer_regions)
        ]
        self.b_outer = [torch.zeros(config.hidden, dtype=torch.float32, device=device) for _ in range(config.outer_regions)]
        self.w_middle = [
            [
                torch.randn(config.hidden, config.hidden, generator=generator, dtype=torch.float32, device=device) * 0.16
                for _ in range(config.middle_per_outer)
            ]
            for _ in range(config.outer_regions)
        ]
        self.b_middle = [
            [torch.zeros(config.hidden, dtype=torch.float32, device=device) for _ in range(config.middle_per_outer)]
            for _ in range(config.outer_regions)
        ]
        self.w_inner = [
            [
                [
                    torch.randn(config.hidden, config.classes, generator=generator, dtype=torch.float32, device=device) * 0.16
                    for _ in range(config.inner_per_middle)
                ]
                for _ in range(config.middle_per_outer)
            ]
            for _ in range(config.outer_regions)
        ]
        self.b_inner = [
            [
                [torch.zeros(config.classes, dtype=torch.float32, device=device) for _ in range(config.inner_per_middle)]
                for _ in range(config.middle_per_outer)
            ]
            for _ in range(config.outer_regions)
        ]
        self.outer = [NodeState(index=index) for index in range(config.outer_regions)]
        self.middle = [[NodeState(index=index) for index in range(config.middle_per_outer)] for _ in range(config.outer_regions)]
        self.inner = [
            [[NodeState(index=index) for index in range(config.inner_per_middle)] for _ in range(config.middle_per_outer)]
            for _ in range(config.outer_regions)
        ]
        self.memory: list[list[list[list[RegionMemory]]]] = [
            [[[] for _ in range(config.inner_per_middle)] for _ in range(config.middle_per_outer)]
            for _ in range(config.outer_regions)
        ]

    def selected_params(self, outer: int, middle: int, inner: int) -> list[torch.Tensor]:
        return [
            self.token_embedding,
            self.position_embedding,
            self.encoder_w1,
            self.encoder_b1,
            self.encoder_w2,
            self.encoder_b2,
            self.w_outer[outer],
            self.b_outer[outer],
            self.w_middle[outer][middle],
            self.b_middle[outer][middle],
            self.w_inner[outer][middle][inner],
            self.b_inner[outer][middle][inner],
        ]

    def param_stream(self) -> list[tuple[str, int | None, int | None, int | None, torch.Tensor]]:
        stream: list[tuple[str, int | None, int | None, int | None, torch.Tensor]] = [
            ("shared", None, None, None, self.token_embedding),
            ("shared", None, None, None, self.position_embedding),
            ("shared", None, None, None, self.encoder_w1),
            ("shared", None, None, None, self.encoder_b1),
            ("shared", None, None, None, self.encoder_w2),
            ("shared", None, None, None, self.encoder_b2),
        ]
        for outer in range(self.config.outer_regions):
            stream.append(("outer", outer, None, None, self.w_outer[outer]))
            stream.append(("outer", outer, None, None, self.b_outer[outer]))
            for middle in range(self.config.middle_per_outer):
                stream.append(("middle", outer, middle, None, self.w_middle[outer][middle]))
                stream.append(("middle", outer, middle, None, self.b_middle[outer][middle]))
                for inner in range(self.config.inner_per_middle):
                    stream.append(("inner", outer, middle, inner, self.w_inner[outer][middle][inner]))
                    stream.append(("inner", outer, middle, inner, self.b_inner[outer][middle][inner]))
        return stream

    def snapshot(self) -> list[torch.Tensor]:
        return [tensor.detach().clone() for *_, tensor in self.param_stream()]

    def leakage(self, before: list[torch.Tensor], selected_outer: int, selected_middle: int, selected_inner: int) -> tuple[float, float, float, float]:
        shared: list[float] = []
        outer_leak: list[float] = []
        middle_leak: list[float] = []
        inner_leak: list[float] = []
        for before_tensor, (kind, outer, middle, inner, after_tensor) in zip(before, self.param_stream(), strict=True):
            delta = float(torch.max(torch.abs(after_tensor.detach() - before_tensor)).cpu())
            if kind == "shared":
                shared.append(delta)
            elif kind == "outer" and outer != selected_outer:
                outer_leak.append(delta)
            elif kind == "middle" and (outer != selected_outer or middle != selected_middle):
                middle_leak.append(delta)
            elif kind == "inner" and (outer != selected_outer or middle != selected_middle or inner != selected_inner):
                inner_leak.append(delta)
        return (
            max(shared) if shared else 0.0,
            max(outer_leak) if outer_leak else 0.0,
            max(middle_leak) if middle_leak else 0.0,
            max(inner_leak) if inner_leak else 0.0,
        )

    def encode(self, tokens: torch.Tensor, params: list[torch.Tensor] | None = None) -> torch.Tensor:
        if params is None:
            tok, pos, w1, b1, w2, b2 = (
                self.token_embedding,
                self.position_embedding,
                self.encoder_w1,
                self.encoder_b1,
                self.encoder_w2,
                self.encoder_b2,
            )
        else:
            tok, pos, w1, b1, w2, b2 = params[:6]
        emb = F.embedding(tokens, tok) + pos[: tokens.shape[0]]
        pooled = emb.mean(dim=0)
        hidden = torch.tanh(pooled @ w1 + b1)
        return torch.tanh(hidden @ w2 + b2)

    def shared_params(self) -> list[torch.Tensor]:
        return [
            self.token_embedding,
            self.position_embedding,
            self.encoder_w1,
            self.encoder_b1,
            self.encoder_w2,
            self.encoder_b2,
        ]

    def routing_params(self) -> list[torch.Tensor]:
        params: list[torch.Tensor] = []
        for outer in range(self.config.outer_regions):
            params.extend([self.w_outer[outer], self.b_outer[outer]])
            for middle in range(self.config.middle_per_outer):
                params.extend([self.w_middle[outer][middle], self.b_middle[outer][middle]])
        return params

    def features(
        self,
        outer: int,
        middle: int,
        tokens: torch.Tensor,
        params: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if params is None:
            z = self.encode(tokens)
            h_outer = torch.tanh(z @ self.w_outer[outer] + self.b_outer[outer])
            h_middle = torch.tanh(h_outer @ self.w_middle[outer][middle] + self.b_middle[outer][middle])
            return z, h_outer, h_middle
        z = self.encode(tokens, params=params)
        h_outer = torch.tanh(z @ params[6] + params[7])
        h_middle = torch.tanh(h_outer @ params[8] + params[9])
        return z, h_outer, h_middle

    def forward_leaf(
        self,
        outer: int,
        middle: int,
        inner: int,
        tokens: torch.Tensor,
        params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if params is None:
            _, _, h_middle = self.features(outer, middle, tokens)
            return h_middle @ self.w_inner[outer][middle][inner] + self.b_inner[outer][middle][inner]
        _, _, h_middle = self.features(outer, middle, tokens, params=params)
        return h_middle @ params[10] + params[11]

    def active_outer(self) -> list[int]:
        return [node.index for node in self.outer if node.active]

    def active_middle(self, outer: int) -> list[int]:
        return [node.index for node in self.middle[outer] if node.active]

    def active_inner(self, outer: int, middle: int) -> list[int]:
        return [node.index for node in self.inner[outer][middle] if node.active]

    def choose_node(self, nodes: list[NodeState], feature: torch.Tensor, threshold: float, branch_threshold: float) -> tuple[int, str, float, float]:
        active = [node.index for node in nodes if node.active]
        if not active:
            return 0, "new", math.inf, -math.inf
        ranked = []
        for index in active:
            node = nodes[index]
            if node.center is None:
                raise RuntimeError(f"Active node {index} has no center.")
            ranked.append((cosine_distance(feature, node.center), -node_survival(node, self.config), index))
        ranked.sort(key=lambda row: (row[0], row[1], row[2]))
        best_distance, negative_survival, best_index = ranked[0]
        best_survival = -negative_survival
        free = next((node.index for node in nodes if not node.active), None)
        if best_distance <= threshold:
            return best_index, "matched", best_distance, best_survival
        if best_distance >= branch_threshold and free is not None:
            return free, "new_branch", best_distance, best_survival
        if free is not None:
            return free, "new", best_distance, best_survival
        weakest = min(active, key=lambda idx: (nodes[idx].admitted, node_survival(nodes[idx], self.config), nodes[idx].strength, -nodes[idx].age, idx))
        weakest_survival = node_survival(nodes[weakest], self.config)
        return weakest, "overwrite_low_survival", best_distance, weakest_survival

    def route(self, example: TokenExample) -> tuple[int, str, int, str, int, str, dict[str, float]]:
        z = self.encode(example.tokens)
        outer, outer_action, outer_distance, outer_survival = self.choose_node(
            self.outer,
            z,
            self.config.outer_match_threshold,
            self.config.outer_branch_threshold,
        )
        if not self.outer[outer].active or outer_action in {"new", "new_branch", "overwrite_low_survival"}:
            initialize_node_from_feature(self.outer[outer], z, example, self.config)
            for middle in range(self.config.middle_per_outer):
                self.release_middle(outer, middle)
        h_outer = torch.tanh(z @ self.w_outer[outer] + self.b_outer[outer])
        middle, middle_action, middle_distance, middle_survival = self.choose_node(
            self.middle[outer],
            h_outer,
            self.config.middle_match_threshold,
            self.config.middle_branch_threshold,
        )
        if not self.middle[outer][middle].active or middle_action in {"new", "new_branch", "overwrite_low_survival"}:
            initialize_node_from_feature(self.middle[outer][middle], h_outer, example, self.config)
            for inner in range(self.config.inner_per_middle):
                self.release_inner(outer, middle, inner)
        h_middle = torch.tanh(h_outer @ self.w_middle[outer][middle] + self.b_middle[outer][middle])
        inner, inner_action, inner_distance, inner_survival = self.choose_node(
            self.inner[outer][middle],
            h_middle,
            self.config.inner_match_threshold,
            self.config.inner_branch_threshold,
        )
        if self.config.semantic_label_branch and inner_action == "matched":
            matched_node = self.inner[outer][middle][inner]
            known_labels = {LABELS[group] for group in matched_node.group_mass if group in LABELS}
            free_inner = next((node.index for node in self.inner[outer][middle] if not node.active), None)
            if known_labels and example.y not in known_labels and free_inner is not None:
                inner = free_inner
                inner_action = "semantic_label_branch"
        if not self.inner[outer][middle][inner].active or inner_action in {"new", "new_branch", "overwrite_low_survival", "semantic_label_branch"}:
            initialize_node_from_feature(self.inner[outer][middle][inner], h_middle, example, self.config)
            self.memory[outer][middle][inner] = []
        return outer, outer_action, middle, middle_action, inner, inner_action, {
            "outer_distance": outer_distance,
            "middle_distance": middle_distance,
            "inner_distance": inner_distance,
            "outer_survival": outer_survival,
            "middle_survival": middle_survival,
            "inner_survival": inner_survival,
        }

    def token_loss(
        self,
        outer: int,
        middle: int,
        inner: int,
        examples: list[TokenExample | RegionMemory],
        params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not examples:
            raise ValueError("Cannot compute loss for empty examples.")
        losses = []
        for item in examples:
            tokens = item.tokens if isinstance(item, TokenExample) else item.x.long()
            logits = self.forward_leaf(outer, middle, inner, tokens, params=params)
            target = torch.tensor([item.y], device=self.device, dtype=torch.long)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        return torch.stack(losses).mean()

    def protected_examples(self, outer: int, middle: int, inner: int, example: TokenExample) -> list[RegionMemory]:
        protected = [
            item
            for item in self.memory[outer][middle][inner]
            if item.group != example.group
            and (self.config.protect_same_label or item.y != example.y)
            and item.usefulness + item.dependency >= self.config.protected_min_potential
        ]
        return protected[-self.config.memory_limit :]

    def train_leaf(self, outer: int, middle: int, inner: int, example: TokenExample) -> dict[str, float]:
        params = self.selected_params(outer, middle, inner)
        before = self.snapshot()
        protected = self.protected_examples(outer, middle, inner, example)
        diagnostics = {
            "loss": math.nan,
            "protected_rows": 0.0,
            "removed_fraction": 0.0,
            "safe_fraction": 1.0,
            "restore_ratio": 0.0,
            "write_gate": event_write_gate(example, self.config),
            "shared_delta": 0.0,
            "outer_leak": 0.0,
            "middle_leak": 0.0,
            "inner_leak": 0.0,
        }
        for _ in range(self.config.inner_steps):
            params_for_grad = [param.detach().clone().requires_grad_(True) for param in params]
            loss_new = self.token_loss(outer, middle, inner, [example], params=params_for_grad)
            raw_gradient = grad_or_raise(loss_new, params_for_grad)
            safe_gradient = raw_gradient
            projection_stats = {"rows": 0.0, "removed_fraction": 0.0, "safe_fraction": 1.0}
            restore_gradient = torch.zeros_like(raw_gradient)
            if self.config.mode == "nested_tangent" and protected:
                rows = []
                for item in protected:
                    protected_params = [param.detach().clone().requires_grad_(True) for param in params]
                    protected_loss = self.token_loss(outer, middle, inner, [item], params=protected_params)
                    rows.append(grad_or_raise(protected_loss, protected_params).detach())
                safe_gradient, projection_stats = project_gradient(raw_gradient, rows, self.config.tangent_damping, self.config.projection_device)
                restore_params = [param.detach().clone().requires_grad_(True) for param in params]
                restore_loss = self.token_loss(outer, middle, inner, protected, params=restore_params)
                restore_gradient = grad_or_raise(restore_loss, restore_params).detach()
                restore_gradient = clip_relative(restore_gradient, safe_gradient, self.config.restore_clip_ratio)
            final_gradient = diagnostics["write_gate"] * (safe_gradient + self.config.restore_weight * restore_gradient)
            depth = max(self.outer[outer].depth, self.middle[outer][middle].depth, self.inner[outer][middle][inner].depth)
            strength = max(self.outer[outer].strength, self.middle[outer][middle].strength, self.inner[outer][middle][inner].strength)
            base_lr = self.config.base_lr * math.exp(-self.config.depth_lr_decay * depth) * math.exp(-self.config.strength_lr_decay * strength)
            pieces = unflatten_like(final_gradient, params)
            with torch.no_grad():
                for index in range(6):
                    params[index] -= base_lr * self.config.shared_lr_multiplier * pieces[index]
                params[6] -= base_lr * self.config.outer_lr_multiplier * pieces[6]
                params[7] -= base_lr * self.config.outer_lr_multiplier * pieces[7]
                params[8] -= base_lr * self.config.middle_lr_multiplier * pieces[8]
                params[9] -= base_lr * self.config.middle_lr_multiplier * pieces[9]
                params[10] -= base_lr * self.config.inner_lr_multiplier * pieces[10]
                params[11] -= base_lr * self.config.inner_lr_multiplier * pieces[11]
            raw_norm = torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
            diagnostics["loss"] = float(loss_new.detach().cpu())
            diagnostics["protected_rows"] = projection_stats["rows"]
            diagnostics["removed_fraction"] = projection_stats["removed_fraction"]
            diagnostics["safe_fraction"] = projection_stats["safe_fraction"]
            diagnostics["restore_ratio"] = float((torch.linalg.vector_norm(self.config.restore_weight * restore_gradient) / raw_norm).detach().cpu())
        shared_delta, outer_leak, middle_leak, inner_leak = self.leakage(before, outer, middle, inner)
        diagnostics["shared_delta"] = shared_delta
        diagnostics["outer_leak"] = outer_leak
        diagnostics["middle_leak"] = middle_leak
        diagnostics["inner_leak"] = inner_leak
        if outer_leak > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected outer branches: {outer_leak:.6g}")
        if middle_leak > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected middle branches: {middle_leak:.6g}")
        if inner_leak > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected inner heads: {inner_leak:.6g}")
        return diagnostics

    def commit(self, outer: int, middle: int, inner: int, example: TokenExample) -> None:
        z, h_outer, h_middle = self.features(outer, middle, example.tokens)
        nodes_and_features = [
            (self.outer[outer], z, self.config.outer_center_lr),
            (self.middle[outer][middle], h_outer, self.config.middle_center_lr),
            (self.inner[outer][middle][inner], h_middle, self.config.inner_center_lr),
        ]
        for node, feature, center_lr in nodes_and_features:
            update_node_metadata(node, example, self.config)
            if node.center is None:
                raise RuntimeError("Cannot update missing center.")
            rate = center_lr * math.exp(-0.4 * node.depth)
            node.center = F.normalize((1.0 - rate) * node.center + rate * feature.detach(), dim=0)
        self.memory[outer][middle][inner].append(
            RegionMemory(
                x=example.tokens.detach().clone(),
                y=example.y,
                group=example.group,
                usefulness=example.usefulness,
                dependency=example.dependency,
                admitted=self.inner[outer][middle][inner].admitted,
            )
        )
        if len(self.memory[outer][middle][inner]) > self.config.memory_limit:
            self.memory[outer][middle][inner] = self.memory[outer][middle][inner][-self.config.memory_limit :]

    def apply_contradiction(self, example: TokenExample) -> tuple[int, float]:
        if example.contradiction_target is None:
            return 0, 0.0
        touched = 0
        pressure_total = 0.0
        for outer in range(self.config.outer_regions):
            for middle in range(self.config.middle_per_outer):
                for inner in range(self.config.inner_per_middle):
                    node = self.inner[outer][middle][inner]
                    if not node.active:
                        continue
                    share = node_share(node, example.contradiction_target)
                    if share <= 0.0:
                        continue
                    pressure = share * share * (1.0 + example.usefulness + example.dependency)
                    node.conflict = 0.82 * node.conflict + pressure
                    target_mass = node.group_mass.get(example.contradiction_target, 0.0)
                    reduced = target_mass * max(0.0, 1.0 - 0.45 * share)
                    if reduced < 0.15:
                        del node.group_mass[example.contradiction_target]
                    else:
                        node.group_mass[example.contradiction_target] = reduced
                    node.dominant_group = max(node.group_mass.items(), key=lambda row: (row[1], row[0]))[0] if node.group_mass else "empty"
                    touched += 1
                    pressure_total += pressure
        return touched, pressure_total

    def decay_idle(self, selected: tuple[int, int, int]) -> int:
        releases = 0
        for outer in range(self.config.outer_regions):
            for middle in range(self.config.middle_per_outer):
                for inner in range(self.config.inner_per_middle):
                    node = self.inner[outer][middle][inner]
                    if not node.active or (outer, middle, inner) == selected:
                        continue
                    node.age += 1
                    decay = 0.045 * math.exp(-0.72 * node.depth)
                    if not node.admitted:
                        decay *= 2.50
                    node.strength *= max(0.0, 1.0 - decay)
                    node.evidence *= max(0.0, 1.0 - 0.35 * decay)
                    node.conflict *= 0.985
                    node.downstream_support *= self.config.support_decay
                    update_node_admission(node, self.config)
                    adjust_node_depth(node, self.config, multiplier=0.18)
                    energy = node_survival(node, self.config)
                    if (not node.admitted) and energy < self.config.release_threshold and node.strength < 0.45:
                        self.release_inner(outer, middle, inner)
                        releases += 1
        return releases

    def release_inner(self, outer: int, middle: int, inner: int) -> None:
        node = self.inner[outer][middle][inner]
        node.active = False
        node.center = None
        node.depth = 0.0
        node.strength = 0.0
        node.evidence = 0.0
        node.usefulness = 0.0
        node.dependency = 0.0
        node.conflict = 0.0
        node.downstream_support = 0.0
        node.admitted = False
        node.age = 0
        node.updates = 0
        node.dominant_group = "empty"
        node.group_mass = {}
        node.support_mass = {}
        node.released_count += 1
        self.memory[outer][middle][inner] = []

    def release_middle(self, outer: int, middle: int) -> None:
        node = self.middle[outer][middle]
        node.active = False
        node.center = None
        node.depth = 0.0
        node.strength = 0.0
        node.evidence = 0.0
        node.usefulness = 0.0
        node.dependency = 0.0
        node.conflict = 0.0
        node.downstream_support = 0.0
        node.admitted = False
        node.age = 0
        node.updates = 0
        node.dominant_group = "empty"
        node.group_mass = {}
        node.support_mass = {}
        node.released_count += 1
        for inner in range(self.config.inner_per_middle):
            self.release_inner(outer, middle, inner)

    def predict(self, tokens: torch.Tensor) -> tuple[int, int, int, int, float]:
        candidates = []
        z = self.encode(tokens)
        for outer in self.active_outer():
            if self.outer[outer].center is None:
                raise RuntimeError("Active outer node missing center.")
            outer_distance = cosine_distance(z, self.outer[outer].center)
            h_outer = torch.tanh(z @ self.w_outer[outer] + self.b_outer[outer])
            for middle in self.active_middle(outer):
                if self.middle[outer][middle].center is None:
                    raise RuntimeError("Active middle node missing center.")
                middle_distance = cosine_distance(h_outer, self.middle[outer][middle].center)
                h_middle = torch.tanh(h_outer @ self.w_middle[outer][middle] + self.b_middle[outer][middle])
                for inner in self.active_inner(outer, middle):
                    node = self.inner[outer][middle][inner]
                    if node.center is None:
                        raise RuntimeError("Active inner node missing center.")
                    inner_distance = cosine_distance(h_middle, node.center)
                    score = outer_distance + middle_distance + inner_distance
                    candidates.append((score, -node_survival(node, self.config), outer, middle, inner))
        if not candidates:
            return -1, -1, -1, -1, 0.0
        candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
        _, _, outer, middle, inner = candidates[0]
        logits = self.forward_leaf(outer, middle, inner, tokens)
        probs = torch.softmax(logits, dim=0)
        pred = int(torch.argmax(probs).detach().cpu())
        confidence = float(torch.max(probs).detach().cpu())
        return pred, outer, middle, inner, confidence

    def active_middle(self, outer: int) -> list[int]:
        return [node.index for node in self.middle[outer] if node.active]

    def active_inner(self, outer: int, middle: int) -> list[int]:
        return [node.index for node in self.inner[outer][middle] if node.active]


def evaluate(model: TokenHierarchicalNestedNet, prototypes: dict[str, torch.Tensor]) -> tuple[list[dict[str, float | str | int]], dict[str, float]]:
    rows: list[dict[str, float | str | int]] = []
    protected: list[float] = []
    branches: list[float] = []
    noise_conf: list[float] = []
    losses: list[float] = []
    for group, tokens in prototypes.items():
        label = LABELS[group]
        pred, outer, middle, inner, confidence = model.predict(tokens)
        loss = math.nan
        if outer >= 0:
            with torch.no_grad():
                logits = model.forward_leaf(outer, middle, inner, tokens)
                loss = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], device=model.device)).detach().cpu())
        correct = float(pred == label)
        if group in {"stable", "merge_a", "merge_b", "branch_root", "branch_up", "branch_down", "rare_critical"}:
            protected.append(correct)
        if group in {"branch_root", "branch_up", "branch_down"}:
            branches.append(correct)
        if group == "noise":
            noise_conf.append(confidence)
        losses.append(loss)
        rows.append(
            {
                "group": group,
                "label": label,
                "pred": pred,
                "outer": outer,
                "middle": middle,
                "inner": inner,
                "correct": correct,
                "loss": loss,
                "confidence": confidence,
            }
        )
    replacement = next(row for row in rows if row["group"] == "replacement")
    obsolete = next(row for row in rows if row["group"] == "obsolete_old")
    return rows, {
        "eval_loss": float(np.nanmean(losses)),
        "protected_acc": float(np.mean(protected)) if protected else math.nan,
        "branch_acc": float(np.mean(branches)) if branches else math.nan,
        "replacement_correct": float(replacement["correct"]),
        "obsolete_old_correct": float(obsolete["correct"]),
        "noise_confidence": float(np.mean(noise_conf)) if noise_conf else math.nan,
    }


def leaf_table(model: TokenHierarchicalNestedNet) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for outer in range(model.config.outer_regions):
        for middle in range(model.config.middle_per_outer):
            for inner in range(model.config.inner_per_middle):
                node = model.inner[outer][middle][inner]
                rows.append(
                    {
                        "outer": outer,
                        "middle": middle,
                        "inner": inner,
                        "active": int(node.active),
                        "dominant_group": node.dominant_group,
                        "dominant_share": node.dominant_share(),
                        "depth": node.depth,
                        "strength": node.strength,
                        "evidence": node.evidence,
                        "usefulness": node.usefulness,
                        "dependency": node.dependency,
                        "conflict": node.conflict,
                        "admitted": int(node.admitted),
                        "age": node.age,
                        "updates": node.updates,
                        "survival": node_survival(node, model.config),
                        "memory": len(model.memory[outer][middle][inner]),
                    }
                )
    return rows


def run_sequence(config: Config) -> tuple[TokenHierarchicalNestedNet, list[dict[str, float | str | int]], list[dict[str, float]], list[dict[str, float | str | int]], dict[str, float]]:
    validate_config(config)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    examples, prototypes, vocab = build_token_examples(config, device)
    model = TokenHierarchicalNestedNet(config, len(vocab), device)
    if config.foundation_epochs > 0:
        pretrain_encoder(model, prototypes, config)
    event_rows: list[dict[str, float | str | int]] = []
    metric_rows: list[dict[str, float]] = []
    max_shared_delta = 0.0
    max_outer_leak = 0.0
    max_middle_leak = 0.0
    max_inner_leak = 0.0
    for event_index, example in enumerate(examples):
        contradiction_nodes, contradiction_pressure = model.apply_contradiction(example)
        outer, outer_action, middle, middle_action, inner, inner_action, routing = model.route(example)
        diagnostics = model.train_leaf(outer, middle, inner, example)
        model.commit(outer, middle, inner, example)
        releases = model.decay_idle((outer, middle, inner))
        max_shared_delta = max(max_shared_delta, diagnostics["shared_delta"])
        max_outer_leak = max(max_outer_leak, diagnostics["outer_leak"])
        max_middle_leak = max(max_middle_leak, diagnostics["middle_leak"])
        max_inner_leak = max(max_inner_leak, diagnostics["inner_leak"])
        active_leaves = [node for outer_nodes in model.inner for middle_nodes in outer_nodes for node in middle_nodes if node.active]
        depths = np.array([node.depth for node in active_leaves], dtype=np.float64) if active_leaves else np.array([], dtype=np.float64)
        metric_rows.append(
            {
                "event_index": float(event_index),
                "step": float(example.step),
                "active_outer": float(len(model.active_outer())),
                "active_middle": float(sum(len(model.active_middle(outer)) for outer in range(config.outer_regions))),
                "active_leaves": float(len(active_leaves)),
                "mean_leaf_depth": float(np.mean(depths)) if len(depths) else 0.0,
                "max_shared_delta": max_shared_delta,
                "max_outer_leak": max_outer_leak,
                "max_middle_leak": max_middle_leak,
                "max_inner_leak": max_inner_leak,
            }
        )
        event_rows.append(
            {
                "event_index": event_index,
                "step": example.step,
                "group": example.group,
                "outer": outer,
                "middle": middle,
                "inner": inner,
                "outer_action": outer_action,
                "middle_action": middle_action,
                "inner_action": inner_action,
                "loss": diagnostics["loss"],
                "protected_rows": diagnostics["protected_rows"],
                "removed_fraction": diagnostics["removed_fraction"],
                "safe_fraction": diagnostics["safe_fraction"],
                "restore_ratio": diagnostics["restore_ratio"],
                "write_gate": diagnostics["write_gate"],
                "shared_delta": diagnostics["shared_delta"],
                "outer_leak": diagnostics["outer_leak"],
                "middle_leak": diagnostics["middle_leak"],
                "inner_leak": diagnostics["inner_leak"],
                "contradiction_nodes": contradiction_nodes,
                "contradiction_pressure": contradiction_pressure,
                "released": releases,
                **routing,
            }
        )
    eval_rows, eval_summary = evaluate(model, prototypes)
    eval_summary["max_shared_delta"] = max_shared_delta
    eval_summary["max_outer_leak"] = max_outer_leak
    eval_summary["max_middle_leak"] = max_middle_leak
    eval_summary["max_inner_leak"] = max_inner_leak
    return model, event_rows, metric_rows, eval_rows, eval_summary


def pretrain_encoder(model: TokenHierarchicalNestedNet, prototypes: dict[str, torch.Tensor], config: Config) -> None:
    train_items = [(group, tokens) for group, tokens in prototypes.items() if group != "noise"]
    if not train_items:
        raise RuntimeError("Foundation pretraining has no non-noise items.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 2881)
    head_w = torch.randn(config.hidden, config.classes, generator=generator, dtype=torch.float32, device=model.device) * 0.10
    head_b = torch.zeros(config.classes, dtype=torch.float32, device=model.device)
    route_params = model.routing_params()
    params = model.shared_params() + route_params + [head_w, head_b]
    head_offset = 6 + len(route_params)
    for _ in range(config.foundation_epochs):
        trainable = [param.detach().clone().requires_grad_(True) for param in params]
        losses = []
        encoded: dict[str, torch.Tensor] = {}
        for group, tokens in train_items:
            z = model.encode(tokens, params=trainable[:6])
            encoded[group] = z
            logits = z @ trainable[head_offset] + trainable[head_offset + 1]
            target = torch.tensor([LABELS[group]], dtype=torch.long, device=model.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        prediction_loss = torch.stack(losses).mean()
        geometry_loss = foundation_geometry_loss(
            encoded=encoded,
            prototypes=prototypes,
            model=model,
            shared_params=trainable[:6],
            routing_params=trainable[6:head_offset],
            config=config,
        )
        loss = prediction_loss + geometry_loss
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError("Foundation encoder loss became non-finite.")
        gradient = grad_or_raise(loss, trainable)
        pieces = unflatten_like(gradient, params)
        with torch.no_grad():
            for param, grad_piece in zip(params, pieces, strict=True):
                param -= config.foundation_lr * grad_piece


def squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, dim=0)
    b_norm = F.normalize(b, dim=0)
    return 1.0 - torch.dot(a_norm, b_norm)


def band_loss(distance: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return F.relu(torch.tensor(low, device=distance.device, dtype=distance.dtype) - distance).pow(2) + F.relu(
        distance - torch.tensor(high, device=distance.device, dtype=distance.dtype)
    ).pow(2)


def semantic_geometry_terms(
    features: dict[str, torch.Tensor],
    config: Config,
    *,
    branch_low: float,
    branch_high: float,
    replacement_low: float,
    replacement_high: float,
) -> list[torch.Tensor]:
    losses: list[torch.Tensor] = []
    merge_distance = squared_distance(features["merge_a"], features["merge_b"])
    losses.append((merge_distance - config.compatible_distance_target).pow(2))
    branch_groups = ["branch_root", "branch_up", "branch_down"]
    for left_index, left in enumerate(branch_groups):
        for right in branch_groups[left_index + 1 :]:
            distance = squared_distance(features[left], features[right])
            losses.append(band_loss(distance, branch_low, branch_high))
    replacement_distance = squared_distance(features["obsolete_old"], features["replacement"])
    losses.append(band_loss(replacement_distance, replacement_low, replacement_high))
    for group in ("stable", "merge_a", "branch_root", "rare_critical"):
        distance = squared_distance(features["noise"], features[group])
        losses.append(F.relu(torch.tensor(config.noise_distance_min, device=distance.device, dtype=distance.dtype) - distance).pow(2))
    return losses


def route_param_index(config: Config, outer: int, middle: int | None = None) -> int:
    per_outer = 2 + 2 * config.middle_per_outer
    base = outer * per_outer
    if middle is None:
        return base
    return base + 2 + 2 * middle


def routed_feature_maps(
    encoded: dict[str, torch.Tensor],
    routing_params: list[torch.Tensor],
    config: Config,
    outer: int,
    middle: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    outer_base = route_param_index(config, outer)
    w_outer = routing_params[outer_base]
    b_outer = routing_params[outer_base + 1]
    middle_base = route_param_index(config, outer, middle)
    w_middle = routing_params[middle_base]
    b_middle = routing_params[middle_base + 1]
    outer_features = {group: torch.tanh(feature @ w_outer + b_outer) for group, feature in encoded.items()}
    middle_features = {group: torch.tanh(feature @ w_middle + b_middle) for group, feature in outer_features.items()}
    return outer_features, middle_features


def foundation_geometry_loss(
    encoded: dict[str, torch.Tensor],
    prototypes: dict[str, torch.Tensor],
    model: TokenHierarchicalNestedNet,
    shared_params: list[torch.Tensor],
    routing_params: list[torch.Tensor],
    config: Config,
) -> torch.Tensor:
    noise_tokens = prototypes["noise"]
    all_encoded = {**encoded, "noise": model.encode(noise_tokens, params=shared_params)}
    encoder_terms = semantic_geometry_terms(
        all_encoded,
        config,
        branch_low=0.0,
        branch_high=config.outer_family_distance_max,
        replacement_low=config.branch_distance_min,
        replacement_high=config.branch_distance_max,
    )
    route_terms: list[torch.Tensor] = []
    inner_branch_low = max(config.inner_branch_threshold + 0.04, config.branch_distance_min)
    inner_branch_high = max(inner_branch_low + 0.05, config.branch_distance_max)
    for outer in range(config.outer_regions):
        for middle in range(config.middle_per_outer):
            outer_features, middle_features = routed_feature_maps(all_encoded, routing_params, config, outer, middle)
            route_terms.extend(
                semantic_geometry_terms(
                    outer_features,
                    config,
                    branch_low=0.0,
                    branch_high=config.outer_family_distance_max,
                    replacement_low=config.branch_distance_min,
                    replacement_high=config.branch_distance_max,
                )
            )
            route_terms.extend(
                semantic_geometry_terms(
                    middle_features,
                    config,
                    branch_low=inner_branch_low,
                    branch_high=inner_branch_high,
                    replacement_low=inner_branch_low,
                    replacement_high=inner_branch_high,
                )
            )
    encoder_loss = torch.stack(encoder_terms).mean()
    route_loss = torch.stack(route_terms).mean()
    return config.foundation_geometry_weight * encoder_loss + config.routing_geometry_weight * route_loss




def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_leaves(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    active = [row for row in rows if int(row["active"]) == 1]
    if not active:
        raise ValueError("Cannot plot empty leaf table.")
    labels = [f"{row['outer']}:{row['middle']}:{row['inner']}\n{row['dominant_group']}" for row in active]
    depth = np.array([float(row["depth"]) for row in active])
    strength = np.array([float(row["strength"]) for row in active])
    colors = [GROUP_COLORS.get(str(row["dominant_group"]), "#333333") for row in active]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    xs = np.arange(len(active))
    axes[0].bar(xs, depth, color=colors)
    axes[0].set_ylabel("leaf depth")
    axes[0].set_title("Tokenized learned-geometry nested leaves")
    axes[1].bar(xs, strength, color=colors)
    axes[1].set_ylabel("leaf strength")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(
    model: TokenHierarchicalNestedNet,
    event_rows: list[dict[str, float | str | int]],
    metric_rows: list[dict[str, float]],
    eval_rows: list[dict[str, float | str | int]],
    eval_summary: dict[str, float],
    semantic: dict[str, float],
    config: Config,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    leaves = leaf_table(model)
    artifacts = {
        "event_log": config.output_dir / "event_log.csv",
        "metrics": config.output_dir / "metrics_over_time.csv",
        "leaves": config.output_dir / "leaf_table.csv",
        "eval": config.output_dir / "eval_by_group.csv",
        "summary": config.output_dir / "tokenized_hierarchical_nested_summary.json",
        "leaf_plot": config.output_dir / "tokenized_hierarchical_nested_leaves.png",
    }
    write_csv(artifacts["event_log"], event_rows)
    write_csv(artifacts["metrics"], metric_rows)
    write_csv(artifacts["leaves"], leaves)
    write_csv(artifacts["eval"], eval_rows)
    plot_leaves(leaves, artifacts["leaf_plot"])
    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "eval_summary": eval_summary,
        "semantic": semantic,
        "eval_rows": eval_rows,
        "leaves": leaves,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    with artifacts["summary"].open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=("default", "long"), default="long")
    parser.add_argument("--mode", choices=("nested_sgd", "nested_tangent"), default="nested_tangent")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--outer-regions", type=int, default=4)
    parser.add_argument("--middle-per-outer", type=int, default=2)
    parser.add_argument("--inner-per-middle", type=int, default=3)
    parser.add_argument("--vocab-extra-noise", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--foundation-epochs", type=int, default=450)
    parser.add_argument("--foundation-lr", type=float, default=0.12)
    parser.add_argument("--foundation-geometry-weight", type=float, default=1.00)
    parser.add_argument("--routing-geometry-weight", type=float, default=1.00)
    parser.add_argument("--compatible-distance-target", type=float, default=0.015)
    parser.add_argument("--outer-family-distance-max", type=float, default=0.12)
    parser.add_argument("--branch-distance-min", type=float, default=0.10)
    parser.add_argument("--branch-distance-max", type=float, default=0.35)
    parser.add_argument("--noise-distance-min", type=float, default=0.45)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--memory-limit", type=int, default=32)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=0.11)
    parser.add_argument("--shared-lr-multiplier", type=float, default=0.04)
    parser.add_argument("--outer-lr-multiplier", type=float, default=0.08)
    parser.add_argument("--middle-lr-multiplier", type=float, default=0.25)
    parser.add_argument("--inner-lr-multiplier", type=float, default=1.00)
    parser.add_argument("--depth-lr-decay", type=float, default=0.65)
    parser.add_argument("--strength-lr-decay", type=float, default=0.05)
    parser.add_argument("--outer-center-lr", type=float, default=0.10)
    parser.add_argument("--middle-center-lr", type=float, default=0.14)
    parser.add_argument("--inner-center-lr", type=float, default=0.18)
    parser.add_argument("--outer-match-threshold", type=float, default=0.30)
    parser.add_argument("--outer-branch-threshold", type=float, default=0.62)
    parser.add_argument("--middle-match-threshold", type=float, default=0.22)
    parser.add_argument("--middle-branch-threshold", type=float, default=0.40)
    parser.add_argument("--inner-match-threshold", type=float, default=0.12)
    parser.add_argument("--inner-branch-threshold", type=float, default=0.24)
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
    parser.add_argument("--admission-threshold", type=float, default=1.85)
    parser.add_argument("--admission-temperature", type=float, default=0.35)
    parser.add_argument("--inward-rate", type=float, default=0.025)
    parser.add_argument("--outward-rate", type=float, default=0.16)
    parser.add_argument("--survival-temperature", type=float, default=0.35)
    parser.add_argument("--provisional-depth-cap", type=float, default=0.85)
    parser.add_argument("--low-potential-depth-cap", type=float, default=0.35)
    parser.add_argument("--low-potential-strength-cap", type=float, default=1.25)
    parser.add_argument("--release-threshold", type=float, default=-0.65)
    parser.add_argument("--overwrite-margin", type=float, default=0.18)
    parser.add_argument("--tangent-damping", type=float, default=1e-3)
    parser.add_argument("--restore-weight", type=float, default=0.20)
    parser.add_argument("--restore-clip-ratio", type=float, default=0.35)
    parser.add_argument("--protected-min-potential", type=float, default=1.00)
    parser.add_argument("--protect-same-label", action="store_true", default=False)
    parser.add_argument("--semantic-label-branch", action="store_true", default=False)
    parser.add_argument("--max-events-per-step", type=int, default=4)
    parser.add_argument("--noise-conf-threshold", type=float, default=0.35)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--projection-device", choices=("cpu", "same"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/08_tokenized_nested_geometry/results/gco-tokenized-hierarchical-nested-cl-seed0"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = Config(**vars(args))
    model, event_rows, metric_rows, eval_rows, eval_summary = run_sequence(config)
    semantic = semantic_metrics(eval_rows, eval_summary, config.noise_conf_threshold)
    summary = write_outputs(model, event_rows, metric_rows, eval_rows, eval_summary, semantic, config)
    print("\nTOKENIZED HIERARCHICAL NESTED CL")
    print("=" * 152)
    print(
        f"protected={eval_summary['protected_acc']:.4f} branch={eval_summary['branch_acc']:.4f} "
        f"semantic={semantic['semantic_score']:.4f} strict={semantic['strict_semantic_score']:.4f} "
        f"merge={semantic['compatible_merge']:.4f} family={semantic['branch_family_cohesion']:.4f} "
        f"leafSep={semantic['branch_leaf_separation']:.4f} replace={semantic['replacement_beats_obsolete']:.4f} "
        f"noise={eval_summary['noise_confidence']:.4f} shared_delta={eval_summary['max_shared_delta']:.3g} "
        f"leak={semantic['leakage_clean']:.1f}"
    )
    print("WROTE")
    print("-" * 152)
    for path in summary["artifacts"].values():
        print(path)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
