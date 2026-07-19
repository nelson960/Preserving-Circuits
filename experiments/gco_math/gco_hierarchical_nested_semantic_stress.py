#!/usr/bin/env python3
"""Three-level nested-geometry stress test.

This experiment extends the internal-branch model from:

    outer region -> child branch

to:

    outer region -> middle branch -> inner leaf

The test asks whether extra semantic pressure becomes organized
hierarchically. A good run should show:

* compatible variants merge into the same inner leaf;
* contextual variants share a broad outer family but occupy different leaves;
* rare critical traces survive;
* replacement beats obsolete memory;
* noise remains weak;
* updates remain parameter-local.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
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

from gco_nested_tangent_optimizer_tiny_nn import (  # noqa: E402
    GROUP_COLORS,
    LABELS,
    Example,
    RegionMemory,
    build_examples,
    clip_relative,
    consolidation_potential,
    cosine_distance,
    flatten_tensors,
    grad_or_raise,
    project_gradient,
    resolve_device,
    sigmoid,
    survival_energy,
    unflatten_like,
)


@dataclass(frozen=True)
class Config:
    seed: int
    scenario: str
    mode: str
    steps: int
    outer_regions: int
    middle_per_outer: int
    inner_per_middle: int
    d_input: int
    hidden: int
    classes: int
    inner_steps: int
    memory_limit: int
    shell_count: int
    base_lr: float
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
    max_events_per_step: int
    device: str
    projection_device: str
    output_dir: Path

    @property
    def regions(self) -> int:
        return self.outer_regions


@dataclass
class NodeState:
    index: int
    active: bool = False
    center: torch.Tensor | None = None
    depth: float = 0.0
    strength: float = 0.0
    evidence: float = 0.0
    usefulness: float = 0.0
    dependency: float = 0.0
    conflict: float = 0.0
    downstream_support: float = 0.0
    admitted: bool = False
    age: int = 0
    updates: int = 0
    dominant_group: str = "empty"
    group_mass: dict[str, float] = field(default_factory=dict)
    support_mass: dict[str, float] = field(default_factory=dict)
    released_count: int = 0

    def dominant_share(self) -> float:
        total = sum(self.group_mass.values())
        if total <= 0.0:
            return 0.0
        return max(self.group_mass.values()) / total


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
    if config.d_input < 3:
        raise ValueError("d_input must be at least 3.")
    if config.hidden < 2:
        raise ValueError("hidden must be at least 2.")
    if config.classes < max(LABELS.values()) + 1:
        raise ValueError("classes must cover all synthetic labels.")
    if config.inner_steps < 1:
        raise ValueError("inner_steps must be at least 1.")
    if config.memory_limit < 1:
        raise ValueError("memory_limit must be at least 1.")
    if config.shell_count < 3:
        raise ValueError("shell_count must be at least 3.")
    if config.projection_device not in {"cpu", "same"}:
        raise ValueError("projection_device must be cpu or same.")
    if config.max_events_per_step < 1:
        raise ValueError("max_events_per_step must be at least 1.")
    for name in (
        "base_lr",
        "outer_lr_multiplier",
        "middle_lr_multiplier",
        "inner_lr_multiplier",
        "depth_lr_decay",
        "strength_lr_decay",
        "outer_center_lr",
        "middle_center_lr",
        "inner_center_lr",
        "outer_match_threshold",
        "outer_branch_threshold",
        "middle_match_threshold",
        "middle_branch_threshold",
        "inner_match_threshold",
        "inner_branch_threshold",
        "evidence_gain",
        "usefulness_gain",
        "dependency_gain",
        "conflict_gain",
        "age_decay",
        "support_gain",
        "support_decay",
        "support_threshold",
        "min_consolidation_potential",
        "admission_threshold",
        "admission_temperature",
        "inward_rate",
        "outward_rate",
        "survival_temperature",
        "provisional_depth_cap",
        "low_potential_depth_cap",
        "low_potential_strength_cap",
        "overwrite_margin",
        "tangent_damping",
        "restore_weight",
        "restore_clip_ratio",
        "protected_min_potential",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative.")


def node_survival(node: NodeState, config: Config) -> float:
    if not node.active:
        return -math.inf
    evidence = math.log1p(max(0.0, node.evidence))
    support = math.log1p(max(0.0, node.downstream_support))
    diversity = math.log1p(float(len(node.support_mass)))
    return float(
        config.evidence_gain * evidence
        + config.usefulness_gain * node.usefulness
        + config.dependency_gain * node.dependency
        + 0.60 * support
        + 0.35 * diversity
        - config.conflict_gain * node.conflict
        - config.age_decay * float(node.age)
    )


def node_potential(node: NodeState) -> float:
    support = math.log1p(max(0.0, node.downstream_support))
    diversity = math.log1p(float(len(node.support_mass)))
    return float(node.usefulness + node.dependency + 0.25 * support + 0.15 * diversity)


def update_node_admission(node: NodeState, config: Config) -> None:
    if not node.active:
        node.admitted = False
        return
    if node_potential(node) < config.min_consolidation_potential:
        node.admitted = False
        return
    probability = sigmoid((node_survival(node, config) - config.admission_threshold) / max(1e-6, config.admission_temperature))
    if probability >= 0.5:
        node.admitted = True
    if node.conflict > 1.15:
        node.admitted = False


def adjust_node_depth(node: NodeState, config: Config, multiplier: float) -> None:
    if not node.active:
        return
    max_depth = float(config.shell_count - 1)
    energy = node_survival(node, config)
    desired = max_depth * sigmoid((energy - 1.35) / max(1e-6, config.survival_temperature))
    if node_potential(node) < config.min_consolidation_potential:
        desired = min(desired, config.low_potential_depth_cap)
    if not node.admitted:
        desired = min(desired, config.provisional_depth_cap)
    if node.conflict > 0.65:
        desired *= 1.0 - sigmoid((node.conflict - 0.65) / max(1e-6, config.survival_temperature))
    rate = config.inward_rate if desired >= node.depth else config.outward_rate
    node.depth = min(max(node.depth + multiplier * rate * (desired - node.depth), 0.0), max_depth)


def initialize_node(node: NodeState, example: Example, config: Config) -> None:
    node.active = True
    node.center = F.normalize(example.x.detach().clone(), dim=0)
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


def update_node(node: NodeState, example: Example, config: Config) -> None:
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


def node_share(node: NodeState, group: str) -> float:
    total = sum(node.group_mass.values())
    if total <= 0.0:
        return 0.0
    return node.group_mass.get(group, 0.0) / total


class HierarchicalNestedNet:
    def __init__(self, config: Config, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed + 1231)
        self.w_outer = [
            torch.randn(config.d_input, config.hidden, generator=self.generator, dtype=torch.float32, device=device) * 0.16
            for _ in range(config.outer_regions)
        ]
        self.b_outer = [torch.zeros(config.hidden, dtype=torch.float32, device=device) for _ in range(config.outer_regions)]
        self.w_middle = [
            [
                torch.randn(config.hidden, config.hidden, generator=self.generator, dtype=torch.float32, device=device) * 0.16
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
                    torch.randn(config.hidden, config.classes, generator=self.generator, dtype=torch.float32, device=device) * 0.16
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
            self.w_outer[outer],
            self.b_outer[outer],
            self.w_middle[outer][middle],
            self.b_middle[outer][middle],
            self.w_inner[outer][middle][inner],
            self.b_inner[outer][middle][inner],
        ]

    def reset_middle(self, outer: int, middle: int) -> None:
        self.w_middle[outer][middle] = (
            torch.randn(self.config.hidden, self.config.hidden, generator=self.generator, dtype=torch.float32, device=self.device) * 0.16
        )
        self.b_middle[outer][middle] = torch.zeros(self.config.hidden, dtype=torch.float32, device=self.device)
        for inner in range(self.config.inner_per_middle):
            self.release_inner(outer, middle, inner, reset_head=True)

    def reset_inner_head(self, outer: int, middle: int, inner: int) -> None:
        self.w_inner[outer][middle][inner] = (
            torch.randn(self.config.hidden, self.config.classes, generator=self.generator, dtype=torch.float32, device=self.device) * 0.16
        )
        self.b_inner[outer][middle][inner] = torch.zeros(self.config.classes, dtype=torch.float32, device=self.device)

    def param_stream(self) -> list[tuple[int, int | None, int | None, str, torch.Tensor]]:
        stream: list[tuple[int, int | None, int | None, str, torch.Tensor]] = []
        for outer in range(self.config.outer_regions):
            stream.append((outer, None, None, "w_outer", self.w_outer[outer]))
            stream.append((outer, None, None, "b_outer", self.b_outer[outer]))
            for middle in range(self.config.middle_per_outer):
                stream.append((outer, middle, None, "w_middle", self.w_middle[outer][middle]))
                stream.append((outer, middle, None, "b_middle", self.b_middle[outer][middle]))
                for inner in range(self.config.inner_per_middle):
                    stream.append((outer, middle, inner, "w_inner", self.w_inner[outer][middle][inner]))
                    stream.append((outer, middle, inner, "b_inner", self.b_inner[outer][middle][inner]))
        return stream

    def param_snapshot(self) -> list[torch.Tensor]:
        return [tensor.detach().clone() for *_, tensor in self.param_stream()]

    def leakage(self, before: list[torch.Tensor], selected_outer: int, selected_middle: int, selected_inner: int) -> tuple[float, float, float]:
        outer_deltas: list[float] = []
        middle_deltas: list[float] = []
        inner_deltas: list[float] = []
        for before_tensor, (outer, middle, inner, _, after_tensor) in zip(before, self.param_stream(), strict=True):
            delta = float(torch.max(torch.abs(after_tensor.detach() - before_tensor)).cpu())
            if outer != selected_outer:
                outer_deltas.append(delta)
            elif middle is not None and middle != selected_middle:
                middle_deltas.append(delta)
            elif middle == selected_middle and inner is not None and inner != selected_inner:
                inner_deltas.append(delta)
        return (
            max(outer_deltas) if outer_deltas else 0.0,
            max(middle_deltas) if middle_deltas else 0.0,
            max(inner_deltas) if inner_deltas else 0.0,
        )

    def forward_leaf(
        self,
        outer: int,
        middle: int,
        inner: int,
        x: torch.Tensor,
        params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if params is None:
            w_o, b_o, w_m, b_m, w_i, b_i = self.selected_params(outer, middle, inner)
        else:
            w_o, b_o, w_m, b_m, w_i, b_i = params
        h_outer = torch.tanh(x @ w_o + b_o)
        h_middle = torch.tanh(h_outer @ w_m + b_m)
        return h_middle @ w_i + b_i

    def active_outer(self) -> list[int]:
        return [node.index for node in self.outer if node.active]

    def active_middle(self, outer: int) -> list[int]:
        return [node.index for node in self.middle[outer] if node.active]

    def active_inner(self, outer: int, middle: int) -> list[int]:
        return [node.index for node in self.inner[outer][middle] if node.active]

    def choose_node(
        self,
        nodes: list[NodeState],
        example: Example,
        match_threshold: float,
        branch_threshold: float,
        contradiction_target: str | None,
    ) -> tuple[int, str, float, float]:
        active = [node.index for node in nodes if node.active]
        if not active:
            return 0, "new", math.inf, -math.inf
        ranked = []
        for index in active:
            node = nodes[index]
            if node.center is None:
                raise RuntimeError(f"Active node {index} has no center.")
            ranked.append((cosine_distance(example.x, node.center), -node_survival(node, self.config), index))
        ranked.sort(key=lambda row: (row[0], row[1], row[2]))
        best_distance, negative_survival, best_index = ranked[0]
        best_survival = -negative_survival
        free = next((node.index for node in nodes if not node.active), None)
        if contradiction_target is not None:
            targets = [idx for idx in active if contradiction_target in nodes[idx].group_mass]
            if targets and free is not None:
                return free, "branch_for_contradiction", best_distance, best_survival
        if best_distance <= match_threshold:
            return best_index, "matched", best_distance, best_survival
        if best_distance >= branch_threshold and free is not None:
            return free, "new_branch", best_distance, best_survival
        if free is not None:
            return free, "new", best_distance, best_survival
        weakest = min(
            active,
            key=lambda idx: (
                nodes[idx].admitted,
                node_survival(nodes[idx], self.config),
                nodes[idx].strength,
                -nodes[idx].age,
                idx,
            ),
        )
        weakest_survival = node_survival(nodes[weakest], self.config)
        if example.usefulness + example.dependency + self.config.overwrite_margin > weakest_survival:
            return weakest, "overwrite_low_survival", best_distance, weakest_survival
        return best_index, "forced_match_capacity_full", best_distance, best_survival

    def initialize_outer(self, outer: int, example: Example) -> None:
        initialize_node(self.outer[outer], example, self.config)
        for middle in range(self.config.middle_per_outer):
            self.release_middle(outer, middle, reset_adapter=False)

    def initialize_middle(self, outer: int, middle: int, example: Example, reset_adapter: bool) -> None:
        if reset_adapter:
            self.reset_middle(outer, middle)
        initialize_node(self.middle[outer][middle], example, self.config)
        for inner in range(self.config.inner_per_middle):
            self.release_inner(outer, middle, inner, reset_head=False)

    def initialize_inner(self, outer: int, middle: int, inner: int, example: Example, reset_head: bool) -> None:
        if reset_head:
            self.reset_inner_head(outer, middle, inner)
        initialize_node(self.inner[outer][middle][inner], example, self.config)
        self.memory[outer][middle][inner] = []

    def update_centers(self, outer: int, middle: int, inner: int, x: torch.Tensor) -> None:
        nodes = [
            (self.outer[outer], self.config.outer_center_lr),
            (self.middle[outer][middle], self.config.middle_center_lr),
            (self.inner[outer][middle][inner], self.config.inner_center_lr),
        ]
        for node, base_rate in nodes:
            if node.center is None:
                raise RuntimeError("Cannot update missing node center.")
            rate = base_rate * math.exp(-0.4 * node.depth)
            node.center = F.normalize((1.0 - rate) * node.center + rate * x.detach(), dim=0)

    def leaf_loss(
        self,
        outer: int,
        middle: int,
        inner: int,
        examples: list[RegionMemory | Example],
        params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not examples:
            raise ValueError("Cannot compute leaf loss for empty examples.")
        losses = []
        for item in examples:
            logits = self.forward_leaf(outer, middle, inner, item.x, params=params)
            target = torch.tensor([item.y], device=self.device, dtype=torch.long)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        return torch.stack(losses).mean()

    def protected_examples(self, outer: int, middle: int, inner: int, example: Example) -> list[RegionMemory]:
        protected = [
            item
            for item in self.memory[outer][middle][inner]
            if item.group != example.group
            and (self.config.protect_same_label or item.y != example.y)
            and item.usefulness + item.dependency >= self.config.protected_min_potential
        ]
        return protected[-self.config.memory_limit :]

    def train_leaf(self, outer: int, middle: int, inner: int, example: Example) -> dict[str, float]:
        if not self.outer[outer].active or not self.middle[outer][middle].active or not self.inner[outer][middle][inner].active:
            raise RuntimeError(f"Cannot train inactive leaf {outer}:{middle}:{inner}.")
        params = self.selected_params(outer, middle, inner)
        before = self.param_snapshot()
        protected = self.protected_examples(outer, middle, inner, example)
        diagnostics = {
            "loss": math.nan,
            "protected_rows": 0.0,
            "removed_fraction": 0.0,
            "safe_fraction": 1.0,
            "restore_ratio": 0.0,
            "outer_leak": 0.0,
            "middle_leak": 0.0,
            "inner_leak": 0.0,
        }
        for _ in range(self.config.inner_steps):
            params_for_grad = [param.detach().clone().requires_grad_(True) for param in params]
            loss_new = self.leaf_loss(outer, middle, inner, [example], params=params_for_grad)
            raw_gradient = grad_or_raise(loss_new, params_for_grad)
            safe_gradient = raw_gradient
            projection_stats = {"rows": 0.0, "removed_fraction": 0.0, "safe_fraction": 1.0}
            restore_gradient = torch.zeros_like(raw_gradient)
            if self.config.mode == "nested_tangent" and protected:
                rows = []
                for item in protected:
                    protected_params = [param.detach().clone().requires_grad_(True) for param in params]
                    protected_loss = self.leaf_loss(outer, middle, inner, [item], params=protected_params)
                    rows.append(grad_or_raise(protected_loss, protected_params).detach())
                safe_gradient, projection_stats = project_gradient(
                    raw_gradient,
                    rows,
                    self.config.tangent_damping,
                    self.config.projection_device,
                )
                restore_params = [param.detach().clone().requires_grad_(True) for param in params]
                restore_loss = self.leaf_loss(outer, middle, inner, protected, params=restore_params)
                restore_gradient = grad_or_raise(restore_loss, restore_params).detach()
                restore_gradient = clip_relative(restore_gradient, safe_gradient, self.config.restore_clip_ratio)
            final_gradient = safe_gradient + self.config.restore_weight * restore_gradient
            depth = max(self.outer[outer].depth, self.middle[outer][middle].depth, self.inner[outer][middle][inner].depth)
            strength = max(self.outer[outer].strength, self.middle[outer][middle].strength, self.inner[outer][middle][inner].strength)
            base_lr = self.config.base_lr * math.exp(-self.config.depth_lr_decay * depth) * math.exp(-self.config.strength_lr_decay * strength)
            pieces = unflatten_like(final_gradient, params)
            with torch.no_grad():
                params[0] -= base_lr * self.config.outer_lr_multiplier * pieces[0]
                params[1] -= base_lr * self.config.outer_lr_multiplier * pieces[1]
                params[2] -= base_lr * self.config.middle_lr_multiplier * pieces[2]
                params[3] -= base_lr * self.config.middle_lr_multiplier * pieces[3]
                params[4] -= base_lr * self.config.inner_lr_multiplier * pieces[4]
                params[5] -= base_lr * self.config.inner_lr_multiplier * pieces[5]
            raw_norm = torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
            diagnostics["loss"] = float(loss_new.detach().cpu())
            diagnostics["protected_rows"] = projection_stats["rows"]
            diagnostics["removed_fraction"] = projection_stats["removed_fraction"]
            diagnostics["safe_fraction"] = projection_stats["safe_fraction"]
            diagnostics["restore_ratio"] = float((torch.linalg.vector_norm(self.config.restore_weight * restore_gradient) / raw_norm).detach().cpu())
        outer_leak, middle_leak, inner_leak = self.leakage(before, outer, middle, inner)
        diagnostics["outer_leak"] = outer_leak
        diagnostics["middle_leak"] = middle_leak
        diagnostics["inner_leak"] = inner_leak
        if outer_leak > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected outer regions: {outer_leak:.6g}")
        if middle_leak > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected middle branches: {middle_leak:.6g}")
        if inner_leak > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected inner heads: {inner_leak:.6g}")
        return diagnostics

    def apply_contradiction(self, example: Example) -> tuple[int, float]:
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
                    node.usefulness *= 1.0 - 0.10 * share
                    node.strength *= 1.0 - 0.08 * share
                    update_node_admission(node, self.config)
                    touched += 1
                    pressure_total += pressure
        return touched, pressure_total

    def commit_example(self, outer: int, middle: int, inner: int, example: Example) -> None:
        for node in (self.outer[outer], self.middle[outer][middle], self.inner[outer][middle][inner]):
            update_node(node, example, self.config)
        self.update_centers(outer, middle, inner, example.x)
        admitted = self.inner[outer][middle][inner].admitted
        self.memory[outer][middle][inner].append(
            RegionMemory(
                x=example.x.detach().clone(),
                y=example.y,
                group=example.group,
                usefulness=example.usefulness,
                dependency=example.dependency,
                admitted=admitted,
            )
        )
        if len(self.memory[outer][middle][inner]) > self.config.memory_limit:
            self.memory[outer][middle][inner] = self.memory[outer][middle][inner][-self.config.memory_limit :]

    def decay_idle(self, selected: tuple[int, int, int]) -> int:
        releases = 0
        selected_outer, selected_middle, selected_inner = selected
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
                    node.downstream_support *= 0.985
                    update_node_admission(node, self.config)
                    adjust_node_depth(node, self.config, multiplier=0.18)
                    energy = node_survival(node, self.config)
                    should_release = (not node.admitted) and energy < self.config.release_threshold and node.strength < 0.45
                    should_release = should_release or (node.conflict > 1.15 and energy < self.config.release_threshold + 1.25)
                    if should_release:
                        self.release_inner(outer, middle, inner, reset_head=False)
                        releases += 1
            for middle in range(self.config.middle_per_outer):
                middle_node = self.middle[outer][middle]
                if not middle_node.active:
                    continue
                if outer == selected_outer and middle == selected_middle:
                    continue
                if not self.active_inner(outer, middle):
                    self.release_middle(outer, middle, reset_adapter=False)
                    releases += 1
            outer_node = self.outer[outer]
            if outer_node.active and outer != selected_outer and not self.active_middle(outer):
                self.release_outer(outer)
                releases += 1
        return releases

    def release_inner(self, outer: int, middle: int, inner: int, reset_head: bool) -> None:
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
        if reset_head:
            self.reset_inner_head(outer, middle, inner)

    def release_middle(self, outer: int, middle: int, reset_adapter: bool) -> None:
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
        if reset_adapter:
            self.reset_middle(outer, middle)
        else:
            for inner in range(self.config.inner_per_middle):
                self.release_inner(outer, middle, inner, reset_head=False)

    def release_outer(self, outer: int) -> None:
        node = self.outer[outer]
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
        for middle in range(self.config.middle_per_outer):
            self.release_middle(outer, middle, reset_adapter=False)

    def predict(self, x: torch.Tensor) -> tuple[int, int, int, int, float]:
        candidates = []
        for outer in self.active_outer():
            for middle in self.active_middle(outer):
                for inner in self.active_inner(outer, middle):
                    node = self.inner[outer][middle][inner]
                    if node.center is None:
                        raise RuntimeError(f"Active leaf {outer}:{middle}:{inner} has no center.")
                    candidates.append((cosine_distance(x, node.center), -node_survival(node, self.config), outer, middle, inner))
        if not candidates:
            return -1, -1, -1, -1, 0.0
        candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
        _, _, outer, middle, inner = candidates[0]
        logits = self.forward_leaf(outer, middle, inner, x)
        probs = torch.softmax(logits, dim=0)
        pred = int(torch.argmax(probs).detach().cpu())
        confidence = float(torch.max(probs).detach().cpu())
        return pred, outer, middle, inner, confidence


def evaluate(model: HierarchicalNestedNet, prototypes: dict[str, torch.Tensor]) -> tuple[list[dict[str, float | str | int]], dict[str, float]]:
    rows: list[dict[str, float | str | int]] = []
    protected_correct: list[float] = []
    branch_correct: list[float] = []
    noise_conf: list[float] = []
    losses: list[float] = []
    for group, x in prototypes.items():
        label = LABELS[group]
        pred, outer, middle, inner, confidence = model.predict(x)
        loss = math.nan
        if outer >= 0:
            with torch.no_grad():
                logits = model.forward_leaf(outer, middle, inner, x)
                loss = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], device=model.device)).detach().cpu())
        correct = float(pred == label)
        if group in {"stable", "merge_a", "merge_b", "branch_root", "branch_up", "branch_down", "rare_critical"}:
            protected_correct.append(correct)
        if group in {"branch_root", "branch_up", "branch_down"}:
            branch_correct.append(correct)
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
        "protected_acc": float(np.mean(protected_correct)) if protected_correct else math.nan,
        "branch_acc": float(np.mean(branch_correct)) if branch_correct else math.nan,
        "replacement_correct": float(replacement["correct"]),
        "obsolete_old_correct": float(obsolete["correct"]),
        "noise_confidence": float(np.mean(noise_conf)) if noise_conf else math.nan,
    }


def leaf_table(model: HierarchicalNestedNet) -> list[dict[str, float | str | int]]:
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


def run_sequence(
    config: Config,
) -> tuple[HierarchicalNestedNet, list[dict[str, float | str | int]], list[dict[str, float]], list[dict[str, float | str | int]], dict[str, float]]:
    validate_config(config)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    examples, prototypes = build_examples(config, device)
    model = HierarchicalNestedNet(config, device)
    event_rows: list[dict[str, float | str | int]] = []
    metric_rows: list[dict[str, float]] = []
    max_outer_leak = 0.0
    max_middle_leak = 0.0
    max_inner_leak = 0.0
    for event_index, example in enumerate(examples):
        contradiction_nodes, contradiction_pressure = model.apply_contradiction(example)
        outer, outer_action, outer_distance, outer_survival = model.choose_node(
            model.outer,
            example,
            config.outer_match_threshold,
            config.outer_branch_threshold,
            example.contradiction_target,
        )
        if not model.outer[outer].active or outer_action in {"new", "new_branch", "branch_for_contradiction", "overwrite_low_survival"}:
            model.initialize_outer(outer, example)
        middle, middle_action, middle_distance, middle_survival = model.choose_node(
            model.middle[outer],
            example,
            config.middle_match_threshold,
            config.middle_branch_threshold,
            example.contradiction_target,
        )
        if not model.middle[outer][middle].active or middle_action in {"new", "new_branch", "branch_for_contradiction", "overwrite_low_survival"}:
            model.initialize_middle(outer, middle, example, reset_adapter=True)
        inner, inner_action, inner_distance, inner_survival = model.choose_node(
            model.inner[outer][middle],
            example,
            config.inner_match_threshold,
            config.inner_branch_threshold,
            example.contradiction_target,
        )
        if not model.inner[outer][middle][inner].active or inner_action in {"new", "new_branch", "branch_for_contradiction", "overwrite_low_survival"}:
            model.initialize_inner(outer, middle, inner, example, reset_head=True)
        diagnostics = model.train_leaf(outer, middle, inner, example)
        model.commit_example(outer, middle, inner, example)
        releases = model.decay_idle((outer, middle, inner))
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
                "outer_distance": outer_distance,
                "middle_distance": middle_distance,
                "inner_distance": inner_distance,
                "outer_survival": outer_survival,
                "middle_survival": middle_survival,
                "inner_survival": inner_survival,
                "contradiction_nodes": contradiction_nodes,
                "contradiction_pressure": contradiction_pressure,
                "released": releases,
                "outer_leak": diagnostics["outer_leak"],
                "middle_leak": diagnostics["middle_leak"],
                "inner_leak": diagnostics["inner_leak"],
            }
        )
    eval_rows, eval_summary = evaluate(model, prototypes)
    eval_summary["max_outer_leak"] = max_outer_leak
    eval_summary["max_middle_leak"] = max_middle_leak
    eval_summary["max_inner_leak"] = max_inner_leak
    return model, event_rows, metric_rows, eval_rows, eval_summary


def location(row: dict[str, float | str | int]) -> tuple[int, int, int]:
    return int(row["outer"]), int(row["middle"]), int(row["inner"])


def group_row(rows: list[dict[str, float | str | int]], group: str) -> dict[str, float | str | int]:
    for row in rows:
        if row["group"] == group:
            return row
    raise RuntimeError(f"Missing group {group!r}.")


def semantic_metrics(eval_rows: list[dict[str, float | str | int]], eval_summary: dict[str, float], noise_threshold: float) -> dict[str, float]:
    merge_a = group_row(eval_rows, "merge_a")
    merge_b = group_row(eval_rows, "merge_b")
    branch_rows = [group_row(eval_rows, group) for group in ("branch_root", "branch_up", "branch_down")]
    rare = group_row(eval_rows, "rare_critical")
    stable = group_row(eval_rows, "stable")
    replacement = group_row(eval_rows, "replacement")
    obsolete = group_row(eval_rows, "obsolete_old")
    noise = group_row(eval_rows, "noise")
    branch_outer_count = len({int(row["outer"]) for row in branch_rows})
    branch_leaf_count = len({location(row) for row in branch_rows})
    compatible_merge = float(location(merge_a) == location(merge_b) and float(merge_a["correct"]) == 1.0 and float(merge_b["correct"]) == 1.0)
    branch_family_cohesion = float(branch_outer_count == 1)
    branch_leaf_separation = float(branch_leaf_count / 3.0)
    branch_full_leaf_separation = float(branch_leaf_count == 3)
    branch_correct = float(np.mean([float(row["correct"]) for row in branch_rows]))
    rare_survival = float(float(rare["correct"]) == 1.0)
    stable_survival = float(float(stable["correct"]) == 1.0)
    replacement_beats_obsolete = float(float(replacement["correct"]) == 1.0 and float(obsolete["correct"]) == 0.0)
    noise_rejected = float(float(noise["confidence"]) <= noise_threshold)
    leakage_clean = float(
        eval_summary["max_outer_leak"] == 0.0 and eval_summary["max_middle_leak"] == 0.0 and eval_summary["max_inner_leak"] == 0.0
    )
    semantic_score = float(
        np.mean(
            [
                compatible_merge,
                branch_family_cohesion,
                branch_leaf_separation,
                branch_correct,
                rare_survival,
                stable_survival,
                replacement_beats_obsolete,
                noise_rejected,
                leakage_clean,
            ]
        )
    )
    strict_semantic_score = float(
        np.mean(
            [
                compatible_merge,
                branch_family_cohesion,
                branch_full_leaf_separation,
                branch_correct,
                rare_survival,
                stable_survival,
                replacement_beats_obsolete,
                noise_rejected,
                leakage_clean,
            ]
        )
    )
    return {
        "semantic_score": semantic_score,
        "strict_semantic_score": strict_semantic_score,
        "compatible_merge": compatible_merge,
        "branch_family_cohesion": branch_family_cohesion,
        "branch_leaf_separation": branch_leaf_separation,
        "branch_correct": branch_correct,
        "rare_survival": rare_survival,
        "stable_survival": stable_survival,
        "replacement_beats_obsolete": replacement_beats_obsolete,
        "noise_rejected": noise_rejected,
        "noise_confidence": float(noise["confidence"]),
        "leakage_clean": leakage_clean,
    }


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


def plot_leaf_table(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    active = [row for row in rows if int(row["active"]) == 1]
    if not active:
        raise ValueError("Cannot plot empty leaf table.")
    labels = [f"{row['outer']}:{row['middle']}:{row['inner']}\n{row['dominant_group']}" for row in active]
    depth = np.array([float(row["depth"]) for row in active], dtype=np.float64)
    strength = np.array([float(row["strength"]) for row in active], dtype=np.float64)
    colors = [GROUP_COLORS.get(str(row["dominant_group"]), "#333333") for row in active]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    xs = np.arange(len(active))
    axes[0].bar(xs, depth, color=colors)
    axes[0].set_ylabel("leaf depth")
    axes[0].set_title("Three-level nested semantic leaves")
    axes[1].bar(xs, strength, color=colors)
    axes[1].set_ylabel("leaf strength")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics(rows: list[dict[str, float]], output_path: Path) -> None:
    xs = np.array([row["event_index"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    axes[0].plot(xs, [row["active_outer"] for row in rows], label="outer")
    axes[0].plot(xs, [row["active_middle"] for row in rows], label="middle")
    axes[0].plot(xs, [row["active_leaves"] for row in rows], label="leaves")
    axes[0].set_ylabel("active count")
    axes[0].legend()
    axes[1].plot(xs, [row["mean_leaf_depth"] for row in rows], color="#984ea3")
    axes[1].set_ylabel("mean leaf depth")
    axes[2].plot(xs, [row["max_outer_leak"] for row in rows], label="outer leak")
    axes[2].plot(xs, [row["max_middle_leak"] for row in rows], label="middle leak")
    axes[2].plot(xs, [row["max_inner_leak"] for row in rows], label="inner leak")
    axes[2].set_ylabel("max leak")
    axes[2].set_xlabel("event")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(
    model: HierarchicalNestedNet,
    event_rows: list[dict[str, float | str | int]],
    metric_rows: list[dict[str, float]],
    eval_rows: list[dict[str, float | str | int]],
    eval_summary: dict[str, float],
    config: Config,
    semantic: dict[str, float],
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    leaves = leaf_table(model)
    artifacts = {
        "event_log": config.output_dir / "event_log.csv",
        "metrics": config.output_dir / "metrics_over_time.csv",
        "leaves": config.output_dir / "leaf_table.csv",
        "eval": config.output_dir / "eval_by_group.csv",
        "summary": config.output_dir / "hierarchical_nested_semantic_summary.json",
        "leaf_plot": config.output_dir / "hierarchical_nested_leaves.png",
        "metric_plot": config.output_dir / "hierarchical_nested_metrics.png",
    }
    write_csv(artifacts["event_log"], event_rows)
    write_csv(artifacts["metrics"], metric_rows)
    write_csv(artifacts["leaves"], leaves)
    write_csv(artifacts["eval"], eval_rows)
    plot_leaf_table(leaves, artifacts["leaf_plot"])
    plot_metrics(metric_rows, artifacts["metric_plot"])
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


def parse_pair_list(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        if "x" not in text:
            raise ValueError("capacities must be comma-separated middlexinner pairs, e.g. 1x2,2x2.")
        left, right = text.split("x", maxsplit=1)
        pairs.append((int(left), int(right)))
    if not pairs:
        raise ValueError("capacities cannot be empty.")
    if any(middle < 1 or inner < 1 for middle, inner in pairs):
        raise ValueError("capacity values must be positive.")
    return pairs


def build_config(args: argparse.Namespace, seed: int, middle: int, inner: int, output_dir: Path) -> Config:
    return Config(
        seed=seed,
        scenario=args.scenario,
        mode=args.mode,
        steps=args.steps,
        outer_regions=args.outer_regions,
        middle_per_outer=middle,
        inner_per_middle=inner,
        d_input=args.d_input,
        hidden=args.hidden,
        classes=args.classes,
        inner_steps=args.inner_steps,
        memory_limit=args.memory_limit,
        shell_count=args.shell_count,
        base_lr=args.base_lr,
        outer_lr_multiplier=args.outer_lr_multiplier,
        middle_lr_multiplier=args.middle_lr_multiplier,
        inner_lr_multiplier=args.inner_lr_multiplier,
        depth_lr_decay=args.depth_lr_decay,
        strength_lr_decay=args.strength_lr_decay,
        outer_center_lr=args.outer_center_lr,
        middle_center_lr=args.middle_center_lr,
        inner_center_lr=args.inner_center_lr,
        outer_match_threshold=args.outer_match_threshold,
        outer_branch_threshold=args.outer_branch_threshold,
        middle_match_threshold=args.middle_match_threshold,
        middle_branch_threshold=args.middle_branch_threshold,
        inner_match_threshold=args.inner_match_threshold,
        inner_branch_threshold=args.inner_branch_threshold,
        evidence_gain=args.evidence_gain,
        usefulness_gain=args.usefulness_gain,
        dependency_gain=args.dependency_gain,
        conflict_gain=args.conflict_gain,
        age_decay=args.age_decay,
        support_gain=args.support_gain,
        support_decay=args.support_decay,
        support_threshold=args.support_threshold,
        min_consolidation_potential=args.min_consolidation_potential,
        admission_threshold=args.admission_threshold,
        admission_temperature=args.admission_temperature,
        inward_rate=args.inward_rate,
        outward_rate=args.outward_rate,
        survival_temperature=args.survival_temperature,
        provisional_depth_cap=args.provisional_depth_cap,
        low_potential_depth_cap=args.low_potential_depth_cap,
        low_potential_strength_cap=args.low_potential_strength_cap,
        release_threshold=args.release_threshold,
        overwrite_margin=args.overwrite_margin,
        tangent_damping=args.tangent_damping,
        restore_weight=args.restore_weight,
        restore_clip_ratio=args.restore_clip_ratio,
        protected_min_potential=args.protected_min_potential,
        protect_same_label=args.protect_same_label,
        max_events_per_step=args.max_events_per_step,
        device=args.device,
        projection_device=args.projection_device,
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--capacities", type=str, default="1x1,1x2,2x2,2x3")
    parser.add_argument("--scenario", choices=("default", "long"), default="long")
    parser.add_argument("--mode", choices=("nested_sgd", "nested_tangent"), default="nested_tangent")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--outer-regions", type=int, default=4)
    parser.add_argument("--d-input", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--memory-limit", type=int, default=32)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=0.12)
    parser.add_argument("--outer-lr-multiplier", type=float, default=0.08)
    parser.add_argument("--middle-lr-multiplier", type=float, default=0.25)
    parser.add_argument("--inner-lr-multiplier", type=float, default=1.00)
    parser.add_argument("--depth-lr-decay", type=float, default=0.65)
    parser.add_argument("--strength-lr-decay", type=float, default=0.05)
    parser.add_argument("--outer-center-lr", type=float, default=0.10)
    parser.add_argument("--middle-center-lr", type=float, default=0.14)
    parser.add_argument("--inner-center-lr", type=float, default=0.18)
    parser.add_argument("--outer-match-threshold", type=float, default=0.25)
    parser.add_argument("--outer-branch-threshold", type=float, default=0.55)
    parser.add_argument("--middle-match-threshold", type=float, default=0.16)
    parser.add_argument("--middle-branch-threshold", type=float, default=0.30)
    parser.add_argument("--inner-match-threshold", type=float, default=0.08)
    parser.add_argument("--inner-branch-threshold", type=float, default=0.16)
    parser.add_argument("--evidence-gain", type=float, default=0.75)
    parser.add_argument("--usefulness-gain", type=float, default=1.20)
    parser.add_argument("--dependency-gain", type=float, default=1.25)
    parser.add_argument("--conflict-gain", type=float, default=1.85)
    parser.add_argument("--age-decay", type=float, default=0.012)
    parser.add_argument("--support-gain", type=float, default=0.45)
    parser.add_argument("--support-decay", type=float, default=0.985)
    parser.add_argument("--support-threshold", type=float, default=0.10)
    parser.add_argument("--min-consolidation-potential", type=float, default=0.85)
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
    parser.add_argument("--max-events-per-step", type=int, default=4)
    parser.add_argument("--noise-conf-threshold", type=float, default=0.35)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--projection-device", choices=("cpu", "same"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/07_hierarchical_nested_geometry/results/gco-hierarchical-nested-semantic-stress-seed0"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("seeds cannot be empty.")
    capacities = parse_pair_list(args.capacities)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stress_rows: list[dict[str, object]] = []
    for seed in seeds:
        for middle_count, inner_count in capacities:
            run_dir = args.output_dir / f"seed{seed}-middle{middle_count}-inner{inner_count}"
            config = build_config(args, seed, middle_count, inner_count, run_dir)
            model, event_rows, metric_rows, eval_rows, eval_summary = run_sequence(config)
            semantic = semantic_metrics(eval_rows, eval_summary, args.noise_conf_threshold)
            write_outputs(model, event_rows, metric_rows, eval_rows, eval_summary, config, semantic)
            stress_rows.append(
                {
                    "seed": seed,
                    "middle_per_outer": middle_count,
                    "inner_per_middle": inner_count,
                    "leaf_capacity_per_outer": middle_count * inner_count,
                    "protected_acc": eval_summary["protected_acc"],
                    "branch_acc": eval_summary["branch_acc"],
                    "replacement_correct": eval_summary["replacement_correct"],
                    "obsolete_old_correct": eval_summary["obsolete_old_correct"],
                    "max_outer_leak": eval_summary["max_outer_leak"],
                    "max_middle_leak": eval_summary["max_middle_leak"],
                    "max_inner_leak": eval_summary["max_inner_leak"],
                    **semantic,
                    "run_dir": str(run_dir),
                }
            )
    stress_csv = args.output_dir / "hierarchical_semantic_stress.csv"
    stress_json = args.output_dir / "hierarchical_semantic_stress.json"
    write_csv(stress_csv, stress_rows)
    serializable_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with stress_json.open("w") as handle:
        json.dump({"config": serializable_config, "rows": stress_rows}, handle, indent=2)
    print("\nHIERARCHICAL NESTED SEMANTIC STRESS")
    print("=" * 160)
    print(
        f"{'seed':>5} {'middle':>6} {'inner':>5} {'semantic':>9} {'strict':>8} {'protected':>9} "
        f"{'branch':>8} {'family':>7} {'leafSep':>7} {'merge':>7} {'replace':>8} {'noise':>7} {'leak':>5}"
    )
    for row in stress_rows:
        print(
            f"{int(row['seed']):5d} {int(row['middle_per_outer']):6d} {int(row['inner_per_middle']):5d} "
            f"{float(row['semantic_score']):9.4f} {float(row['strict_semantic_score']):8.4f} "
            f"{float(row['protected_acc']):9.4f} {float(row['branch_acc']):8.4f} "
            f"{float(row['branch_family_cohesion']):7.4f} {float(row['branch_leaf_separation']):7.4f} "
            f"{float(row['compatible_merge']):7.4f} {float(row['replacement_beats_obsolete']):8.4f} "
            f"{float(row['noise_confidence']):7.4f} {float(row['leakage_clean']):5.1f}"
        )
    print("\nWROTE")
    print("-" * 160)
    print(stress_csv)
    print(stress_json)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
