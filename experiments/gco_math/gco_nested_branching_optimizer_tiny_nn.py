#!/usr/bin/env python3
"""Tiny nested neural network with internal branches inside each region.

The previous nested-tangent experiment separated the model into outer parameter
regions. That proved updates can be kept local, but a forced-sharing run showed
that branch_root, branch_up, and branch_down can collide when they are routed
into the same outer region.

This experiment adds one more level:

* an outer region owns a shared trunk;
* each outer region owns several child branches, each with its own output head;
* an update chooses one outer region and one child branch;
* non-selected outer regions must remain bitwise unchanged;
* non-selected child heads inside the selected region must remain unchanged;
* the shared trunk moves slowly and can be constrained by sibling branches.

This is not meant to be a finished architecture. It is a pressure test for the
"internal branching inside a region" idea: when a region is forced to hold
several related meanings, can child branches prevent destructive overwriting?
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
    RegionState,
    adjust_depth,
    build_examples,
    clip_relative,
    consolidation_potential,
    cosine_distance,
    flatten_tensors,
    grad_or_raise,
    project_gradient,
    resolve_device,
    survival_energy,
    unflatten_like,
    update_admission,
    validate_config as validate_base_config,
)


@dataclass(frozen=True)
class Config:
    seed: int
    scenario: str
    mode: str
    steps: int
    regions: int
    children_per_region: int
    d_input: int
    hidden: int
    classes: int
    inner_steps: int
    memory_limit: int
    shell_count: int
    base_lr: float
    trunk_lr_multiplier: float
    child_lr_multiplier: float
    depth_lr_decay: float
    strength_lr_decay: float
    center_lr: float
    child_center_lr: float
    match_threshold: float
    branch_threshold: float
    child_match_threshold: float
    child_branch_threshold: float
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
    protected_require_admitted: bool
    protect_same_label: bool
    sibling_protect: bool
    max_events_per_step: int
    device: str
    projection_device: str
    output_dir: Path


@dataclass
class ChildState:
    child: int
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
    validate_base_config(config)
    if config.children_per_region < 1:
        raise ValueError("children_per_region must be at least 1.")
    if config.trunk_lr_multiplier < 0.0:
        raise ValueError("trunk_lr_multiplier must be non-negative.")
    if config.child_lr_multiplier <= 0.0:
        raise ValueError("child_lr_multiplier must be positive.")
    if config.child_center_lr < 0.0:
        raise ValueError("child_center_lr must be non-negative.")
    if config.child_match_threshold > 2.0 or config.child_branch_threshold > 2.0:
        raise ValueError("child distance thresholds are cosine distances and must be <= 2.")


def state_groups_share(state: RegionState | ChildState, group: str) -> float:
    mass = sum(state.group_mass.values())
    if mass <= 0.0:
        return 0.0
    return state.group_mass.get(group, 0.0) / mass


class BranchingNestedNet:
    def __init__(self, config: Config, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed + 917)
        self.w1 = [
            torch.randn(config.d_input, config.hidden, generator=self.generator, dtype=torch.float32, device=device) * 0.18
            for _ in range(config.regions)
        ]
        self.b1 = [torch.zeros(config.hidden, dtype=torch.float32, device=device) for _ in range(config.regions)]
        self.w2 = [
            [
                torch.randn(config.hidden, config.classes, generator=self.generator, dtype=torch.float32, device=device) * 0.18
                for _ in range(config.children_per_region)
            ]
            for _ in range(config.regions)
        ]
        self.b2 = [
            [torch.zeros(config.classes, dtype=torch.float32, device=device) for _ in range(config.children_per_region)]
            for _ in range(config.regions)
        ]
        self.regions = [RegionState(region=index) for index in range(config.regions)]
        self.children = [[ChildState(child=index) for index in range(config.children_per_region)] for _ in range(config.regions)]
        self.memory: list[list[list[RegionMemory]]] = [
            [[] for _ in range(config.children_per_region)] for _ in range(config.regions)
        ]

    def selected_params(self, region: int, child: int) -> list[torch.Tensor]:
        return [self.w1[region], self.b1[region], self.w2[region][child], self.b2[region][child]]

    def reset_child_head(self, region: int, child: int) -> None:
        self.w2[region][child] = (
            torch.randn(
                self.config.hidden,
                self.config.classes,
                generator=self.generator,
                dtype=torch.float32,
                device=self.device,
            )
            * 0.18
        )
        self.b2[region][child] = torch.zeros(self.config.classes, dtype=torch.float32, device=self.device)

    def param_snapshot(self) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        for region in range(self.config.regions):
            tensors.extend([self.w1[region], self.b1[region]])
            for child in range(self.config.children_per_region):
                tensors.extend([self.w2[region][child], self.b2[region][child]])
        return [tensor.detach().clone() for tensor in tensors]

    def _param_stream(self) -> list[tuple[int, int | None, str, torch.Tensor]]:
        tensors: list[tuple[int, int | None, str, torch.Tensor]] = []
        for region in range(self.config.regions):
            tensors.append((region, None, "w1", self.w1[region]))
            tensors.append((region, None, "b1", self.b1[region]))
            for child in range(self.config.children_per_region):
                tensors.append((region, child, "w2", self.w2[region][child]))
                tensors.append((region, child, "b2", self.b2[region][child]))
        return tensors

    def cross_region_delta(self, before: list[torch.Tensor], selected_region: int) -> float:
        deltas: list[float] = []
        for before_tensor, (region, _, _, after_tensor) in zip(before, self._param_stream(), strict=True):
            if region != selected_region:
                deltas.append(float(torch.max(torch.abs(after_tensor.detach() - before_tensor)).cpu()))
        return max(deltas) if deltas else 0.0

    def cross_child_delta(self, before: list[torch.Tensor], selected_region: int, selected_child: int) -> float:
        deltas: list[float] = []
        for before_tensor, (region, child, _, after_tensor) in zip(before, self._param_stream(), strict=True):
            if region == selected_region and child is not None and child != selected_child:
                deltas.append(float(torch.max(torch.abs(after_tensor.detach() - before_tensor)).cpu()))
        return max(deltas) if deltas else 0.0

    def forward_child(
        self,
        region: int,
        child: int,
        x: torch.Tensor,
        params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if params is None:
            w1, b1, w2, b2 = self.selected_params(region, child)
        else:
            w1, b1, w2, b2 = params
        hidden = torch.tanh(x @ w1 + b1)
        return hidden @ w2 + b2

    def forward_with_trunk(
        self,
        region: int,
        child: int,
        x: torch.Tensor,
        trunk_params: list[torch.Tensor],
    ) -> torch.Tensor:
        w1, b1 = trunk_params
        hidden = torch.tanh(x @ w1 + b1)
        return hidden @ self.w2[region][child].detach() + self.b2[region][child].detach()

    def active_regions(self) -> list[int]:
        return [state.region for state in self.regions if state.active]

    def active_children(self, region: int) -> list[int]:
        return [state.child for state in self.children[region] if state.active]

    def choose_region(self, example: Example) -> tuple[int, str, float, float]:
        active = self.active_regions()
        if not active:
            return 0, "new_region", math.inf, -math.inf
        ranked = []
        for index in active:
            state = self.regions[index]
            if state.center is None:
                raise RuntimeError(f"Active region {index} has no center.")
            ranked.append((cosine_distance(example.x, state.center), -survival_energy(state, self.config), index))
        ranked.sort(key=lambda row: (row[0], row[1], row[2]))
        best_distance, negative_survival, best_index = ranked[0]
        best_survival = -negative_survival
        free = next((state.region for state in self.regions if not state.active), None)
        if best_distance <= self.config.match_threshold:
            return best_index, "matched_region", best_distance, best_survival
        if best_distance >= self.config.branch_threshold and free is not None:
            return free, "new_branch_region", best_distance, best_survival
        if free is not None:
            return free, "new_region", best_distance, best_survival
        return best_index, "forced_match_region_capacity_full", best_distance, best_survival

    def choose_child(self, region: int, example: Example) -> tuple[int, str, float, float]:
        active = self.active_children(region)
        if not active:
            return 0, "new_child", math.inf, -math.inf
        ranked = []
        for index in active:
            state = self.children[region][index]
            if state.center is None:
                raise RuntimeError(f"Active child {region}:{index} has no center.")
            ranked.append((cosine_distance(example.x, state.center), -survival_energy(state, self.config), index))
        ranked.sort(key=lambda row: (row[0], row[1], row[2]))
        best_distance, negative_survival, best_index = ranked[0]
        best_survival = -negative_survival
        free = next((state.child for state in self.children[region] if not state.active), None)
        low_potential = example.usefulness + example.dependency < self.config.min_consolidation_potential
        if low_potential and self.children[region][best_index].admitted:
            if free is not None:
                return free, "new_provisional_child", best_distance, best_survival
            provisional = [index for index in active if not self.children[region][index].admitted]
            if provisional:
                chosen = min(
                    provisional,
                    key=lambda idx: (
                        survival_energy(self.children[region][idx], self.config),
                        self.children[region][idx].strength,
                        -self.children[region][idx].age,
                        idx,
                    ),
                )
                return chosen, "provisional_low_potential_child", best_distance, survival_energy(self.children[region][chosen], self.config)
            return -1, "reject_low_potential_child", best_distance, best_survival
        if example.contradiction_target is not None:
            target_children = [idx for idx in active if example.contradiction_target in self.children[region][idx].group_mass]
            if target_children and free is not None:
                return free, "branch_for_contradiction_child", best_distance, best_survival
        if best_distance <= self.config.child_match_threshold:
            return best_index, "matched_child", best_distance, best_survival
        if best_distance >= self.config.child_branch_threshold and free is not None:
            return free, "new_branch_child", best_distance, best_survival
        if free is not None:
            return free, "new_child", best_distance, best_survival
        weakest = min(
            active,
            key=lambda idx: (
                self.children[region][idx].admitted,
                survival_energy(self.children[region][idx], self.config),
                self.children[region][idx].strength,
                -self.children[region][idx].age,
                idx,
            ),
        )
        weakest_survival = survival_energy(self.children[region][weakest], self.config)
        if example.usefulness + example.dependency + self.config.overwrite_margin > weakest_survival:
            return weakest, "overwrite_low_survival_child", best_distance, weakest_survival
        return best_index, "forced_match_child_capacity_full", best_distance, best_survival

    def initialize_region(self, region: int, example: Example) -> None:
        state = self.regions[region]
        state.active = True
        state.center = F.normalize(example.x.detach().clone(), dim=0)
        state.depth = 0.0
        state.strength = 0.55
        state.evidence = 1.0
        state.usefulness = example.usefulness
        state.dependency = example.dependency
        state.conflict = 0.0
        state.downstream_support = 0.0
        state.admitted = False
        state.age = 0
        state.updates = 1
        state.dominant_group = example.group
        state.group_mass = {example.group: 1.0}
        state.support_mass = {}
        update_admission(state, self.config)
        adjust_depth(state, self.config, multiplier=0.6)
        for child in range(self.config.children_per_region):
            self.release_child(region, child, reset_head=True)

    def initialize_child(self, region: int, child: int, example: Example, reset_head: bool) -> None:
        if reset_head:
            self.reset_child_head(region, child)
        state = self.children[region][child]
        state.active = True
        state.center = F.normalize(example.x.detach().clone(), dim=0)
        state.depth = 0.0
        state.strength = 0.55
        state.evidence = 1.0
        state.usefulness = example.usefulness
        state.dependency = example.dependency
        state.conflict = 0.0
        state.downstream_support = 0.0
        state.admitted = False
        state.age = 0
        state.updates = 1
        state.dominant_group = example.group
        state.group_mass = {example.group: 1.0}
        state.support_mass = {}
        update_admission(state, self.config)
        if consolidation_potential(state) < self.config.min_consolidation_potential:
            state.strength = min(state.strength, self.config.low_potential_strength_cap)
        adjust_depth(state, self.config, multiplier=1.0)
        self.memory[region][child] = []

    def update_centers(self, region: int, child: int, x: torch.Tensor) -> None:
        region_state = self.regions[region]
        child_state = self.children[region][child]
        if region_state.center is None or child_state.center is None:
            raise RuntimeError(f"Cannot update missing center for region {region}, child {child}.")
        region_rate = self.config.center_lr * math.exp(-0.4 * region_state.depth)
        child_rate = self.config.child_center_lr * math.exp(-0.4 * child_state.depth)
        region_state.center = F.normalize((1.0 - region_rate) * region_state.center + region_rate * x.detach(), dim=0)
        child_state.center = F.normalize((1.0 - child_rate) * child_state.center + child_rate * x.detach(), dim=0)

    def child_loss(
        self,
        region: int,
        child: int,
        examples: list[RegionMemory | Example],
        params: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not examples:
            raise ValueError("Cannot compute child loss for empty examples.")
        losses = []
        for item in examples:
            logits = self.forward_child(region, child, item.x, params=params)
            target = torch.tensor([item.y], device=self.device, dtype=torch.long)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        return torch.stack(losses).mean()

    def protected_same_child(self, region: int, child: int, current_group: str, current_y: int) -> list[RegionMemory]:
        protected = [
            item
            for item in self.memory[region][child]
            if item.group != current_group
            and (self.config.protect_same_label or item.y != current_y)
            and item.usefulness + item.dependency >= self.config.protected_min_potential
            and (item.admitted or not self.config.protected_require_admitted)
        ]
        return protected[-self.config.memory_limit :]

    def protected_siblings(self, region: int, child: int, current_group: str, current_y: int) -> list[tuple[int, RegionMemory]]:
        if not self.config.sibling_protect:
            return []
        rows: list[tuple[int, RegionMemory]] = []
        for sibling in self.active_children(region):
            if sibling == child:
                continue
            for item in self.memory[region][sibling]:
                if (
                    item.group != current_group
                    and (self.config.protect_same_label or item.y != current_y)
                    and item.usefulness + item.dependency >= self.config.protected_min_potential
                    and (item.admitted or not self.config.protected_require_admitted)
                ):
                    rows.append((sibling, item))
        return rows[-self.config.memory_limit :]

    def sibling_constraint_row(
        self,
        region: int,
        selected_child: int,
        sibling_child: int,
        item: RegionMemory,
        selected_params: list[torch.Tensor],
    ) -> torch.Tensor:
        trunk_params = [param.detach().clone().requires_grad_(True) for param in selected_params[:2]]
        logits = self.forward_with_trunk(region, sibling_child, item.x, trunk_params=trunk_params)
        target = torch.tensor([item.y], device=self.device, dtype=torch.long)
        loss = F.cross_entropy(logits.unsqueeze(0), target)
        trunk_gradient = grad_or_raise(loss, trunk_params)
        head_zeros = flatten_tensors([torch.zeros_like(selected_params[2]), torch.zeros_like(selected_params[3])])
        return torch.cat([trunk_gradient.detach(), head_zeros.to(trunk_gradient.device)], dim=0)

    def train_child(self, region: int, child: int, example: Example) -> dict[str, float]:
        region_state = self.regions[region]
        child_state = self.children[region][child]
        if not region_state.active or not child_state.active:
            raise RuntimeError(f"Cannot train inactive branch {region}:{child}.")
        params = self.selected_params(region, child)
        before = self.param_snapshot()
        diagnostics = {
            "protected_rows": 0.0,
            "sibling_rows": 0.0,
            "removed_fraction": 0.0,
            "safe_fraction": 1.0,
            "restore_ratio": 0.0,
            "loss": math.nan,
            "cross_region_delta": 0.0,
            "cross_child_delta": 0.0,
        }
        protected = self.protected_same_child(region, child, example.group, example.y)
        sibling_protected = self.protected_siblings(region, child, example.group, example.y)
        for _ in range(self.config.inner_steps):
            params_for_grad = [param.detach().clone().requires_grad_(True) for param in params]
            loss_new = self.child_loss(region, child, [example], params=params_for_grad)
            raw_gradient = grad_or_raise(loss_new, params_for_grad)
            safe_gradient = raw_gradient
            projection_stats = {"rows": 0.0, "removed_fraction": 0.0, "safe_fraction": 1.0}
            restore_gradient = torch.zeros_like(raw_gradient)
            if self.config.mode == "nested_tangent" and (protected or sibling_protected):
                rows: list[torch.Tensor] = []
                for item in protected:
                    protected_params = [param.detach().clone().requires_grad_(True) for param in params]
                    protected_loss = self.child_loss(region, child, [item], params=protected_params)
                    rows.append(grad_or_raise(protected_loss, protected_params).detach())
                for sibling_child, item in sibling_protected:
                    rows.append(self.sibling_constraint_row(region, child, sibling_child, item, params))
                safe_gradient, projection_stats = project_gradient(
                    raw_gradient,
                    rows,
                    self.config.tangent_damping,
                    self.config.projection_device,
                )
                if protected:
                    restore_params = [param.detach().clone().requires_grad_(True) for param in params]
                    restore_loss = self.child_loss(region, child, protected, params=restore_params)
                    restore_gradient = grad_or_raise(restore_loss, restore_params).detach()
                    restore_gradient = clip_relative(restore_gradient, safe_gradient, self.config.restore_clip_ratio)
            final_gradient = safe_gradient + self.config.restore_weight * restore_gradient
            base_lr = (
                self.config.base_lr
                * math.exp(-self.config.depth_lr_decay * max(region_state.depth, child_state.depth))
                * math.exp(-self.config.strength_lr_decay * max(region_state.strength, child_state.strength))
            )
            pieces = unflatten_like(final_gradient, params)
            with torch.no_grad():
                params[0] -= base_lr * self.config.trunk_lr_multiplier * pieces[0]
                params[1] -= base_lr * self.config.trunk_lr_multiplier * pieces[1]
                params[2] -= base_lr * self.config.child_lr_multiplier * pieces[2]
                params[3] -= base_lr * self.config.child_lr_multiplier * pieces[3]
            diagnostics["loss"] = float(loss_new.detach().cpu())
            diagnostics["protected_rows"] = projection_stats["rows"]
            diagnostics["sibling_rows"] = float(len(sibling_protected))
            diagnostics["removed_fraction"] = projection_stats["removed_fraction"]
            diagnostics["safe_fraction"] = projection_stats["safe_fraction"]
            raw_norm = torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
            diagnostics["restore_ratio"] = float((torch.linalg.vector_norm(self.config.restore_weight * restore_gradient) / raw_norm).detach().cpu())
        diagnostics["cross_region_delta"] = self.cross_region_delta(before, region)
        diagnostics["cross_child_delta"] = self.cross_child_delta(before, region, child)
        if diagnostics["cross_region_delta"] > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected outer regions: {diagnostics['cross_region_delta']:.6g}")
        if diagnostics["cross_child_delta"] > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected child heads: {diagnostics['cross_child_delta']:.6g}")
        return diagnostics

    def apply_contradiction(self, example: Example) -> tuple[int, float]:
        if example.contradiction_target is None:
            return 0, 0.0
        touched = 0
        pressure_total = 0.0
        for region_state in self.regions:
            if not region_state.active:
                continue
            region_share = state_groups_share(region_state, example.contradiction_target)
            if region_share > 0.0:
                pressure = region_share * region_share * (1.0 + example.usefulness + example.dependency)
                region_state.conflict = 0.82 * region_state.conflict + pressure
                region_state.usefulness *= 1.0 - 0.08 * region_share
                region_state.strength *= 1.0 - 0.06 * region_share
                pressure_total += pressure
            for child_state in self.children[region_state.region]:
                if not child_state.active:
                    continue
                share = state_groups_share(child_state, example.contradiction_target)
                if share <= 0.0:
                    continue
                pressure = share * share * (1.0 + example.usefulness + example.dependency)
                child_state.conflict = 0.82 * child_state.conflict + pressure
                target_mass = child_state.group_mass.get(example.contradiction_target, 0.0)
                reduced = target_mass * max(0.0, 1.0 - 0.45 * share)
                if reduced < 0.15:
                    del child_state.group_mass[example.contradiction_target]
                else:
                    child_state.group_mass[example.contradiction_target] = reduced
                child_state.dominant_group = max(child_state.group_mass.items(), key=lambda row: (row[1], row[0]))[0] if child_state.group_mass else "empty"
                child_state.usefulness *= 1.0 - 0.10 * share
                child_state.strength *= 1.0 - 0.08 * share
                update_admission(child_state, self.config)
                touched += 1
                pressure_total += pressure
        return touched, pressure_total

    def apply_support(self, example: Example, selected_region: int, selected_child: int) -> tuple[int, float]:
        potential = example.usefulness + example.dependency
        if potential < self.config.min_consolidation_potential:
            return 0, 0.0
        touched = 0
        support_total = 0.0
        for child_state in self.children[selected_region]:
            if not child_state.active or child_state.child == selected_child or child_state.center is None:
                continue
            if example.group in child_state.group_mass:
                continue
            distance = cosine_distance(example.x, child_state.center)
            if distance > self.config.support_threshold:
                continue
            closeness = 1.0 - distance / max(1e-12, self.config.support_threshold)
            support = self.config.support_gain * closeness * potential
            child_state.evidence += support
            child_state.downstream_support = min(25.0, child_state.downstream_support + support)
            child_state.support_mass[example.group] = child_state.support_mass.get(example.group, 0.0) + support
            child_state.usefulness = max(child_state.usefulness, 0.94 * child_state.usefulness + 0.06 * example.usefulness)
            child_state.dependency = max(child_state.dependency, 0.94 * child_state.dependency + 0.06 * example.dependency)
            child_state.strength = min(10.0, child_state.strength + 0.12 * support)
            child_state.age = max(0, child_state.age - 1)
            update_admission(child_state, self.config)
            adjust_depth(child_state, self.config, multiplier=0.25)
            touched += 1
            support_total += support
        return touched, support_total

    def update_region_state(self, region: int, example: Example, action: str) -> None:
        state = self.regions[region]
        if action in {"new_region", "new_branch_region"} or not state.active:
            self.initialize_region(region, example)
            return
        state.evidence = 0.97 * state.evidence + 1.0
        state.usefulness = 0.92 * state.usefulness + 0.08 * example.usefulness
        state.dependency = max(0.96 * state.dependency, example.dependency)
        state.conflict *= 0.90
        state.strength = min(10.0, 0.985 * state.strength + 0.20 + 0.12 * example.usefulness + 0.10 * example.dependency)
        state.age = 0
        state.updates += 1
        state.group_mass[example.group] = state.group_mass.get(example.group, 0.0) + 1.0
        state.dominant_group = max(state.group_mass.items(), key=lambda row: (row[1], row[0]))[0]
        update_admission(state, self.config)
        adjust_depth(state, self.config, multiplier=0.45)

    def update_child_state(self, region: int, child: int, example: Example, action: str) -> None:
        state = self.children[region][child]
        if action in {"new_child", "new_branch_child", "new_provisional_child", "branch_for_contradiction_child"} or not state.active:
            self.initialize_child(region, child, example, reset_head=True)
            state = self.children[region][child]
        elif action == "overwrite_low_survival_child":
            state.released_count += 1
            self.initialize_child(region, child, example, reset_head=True)
            state = self.children[region][child]
        else:
            state.evidence = 0.97 * state.evidence + 1.0
            state.usefulness = 0.92 * state.usefulness + 0.08 * example.usefulness
            state.dependency = max(0.96 * state.dependency, example.dependency)
            state.conflict *= 0.90
            state.strength = min(10.0, 0.985 * state.strength + 0.20 + 0.12 * example.usefulness + 0.10 * example.dependency)
            state.age = 0
            state.updates += 1
            state.group_mass[example.group] = state.group_mass.get(example.group, 0.0) + 1.0
            state.dominant_group = max(state.group_mass.items(), key=lambda row: (row[1], row[0]))[0]
            update_admission(state, self.config)
            if consolidation_potential(state) < self.config.min_consolidation_potential:
                state.strength = min(state.strength, self.config.low_potential_strength_cap)
            adjust_depth(state, self.config, multiplier=1.0)
        self.update_centers(region, child, example.x)
        self.memory[region][child].append(
            RegionMemory(
                x=example.x.detach().clone(),
                y=example.y,
                group=example.group,
                usefulness=example.usefulness,
                dependency=example.dependency,
                admitted=state.admitted,
            )
        )
        if len(self.memory[region][child]) > self.config.memory_limit:
            self.memory[region][child] = self.memory[region][child][-self.config.memory_limit :]

    def decay_idle(self, selected_region: int, selected_child: int) -> int:
        releases = 0
        for region_state in self.regions:
            if not region_state.active:
                continue
            if region_state.region != selected_region:
                region_state.age += 1
                region_state.strength *= max(0.0, 1.0 - 0.030 * math.exp(-0.72 * region_state.depth))
                region_state.downstream_support *= self.config.support_decay
                update_admission(region_state, self.config)
                adjust_depth(region_state, self.config, multiplier=0.12)
            active_children = 0
            for child_state in self.children[region_state.region]:
                if not child_state.active:
                    continue
                active_children += 1
                if region_state.region == selected_region and child_state.child == selected_child:
                    continue
                child_state.age += 1
                decay = 0.045 * math.exp(-0.72 * child_state.depth)
                if not child_state.admitted:
                    decay *= 2.50
                child_state.strength *= max(0.0, 1.0 - decay)
                child_state.evidence *= max(0.0, 1.0 - 0.35 * decay)
                child_state.conflict *= 0.985
                child_state.downstream_support *= self.config.support_decay
                if child_state.support_mass:
                    child_state.support_mass = {
                        group: mass * self.config.support_decay
                        for group, mass in child_state.support_mass.items()
                        if mass * self.config.support_decay > 1e-6
                    }
                update_admission(child_state, self.config)
                adjust_depth(child_state, self.config, multiplier=0.18)
                energy = survival_energy(child_state, self.config)
                should_release = (not child_state.admitted) and energy < self.config.release_threshold and child_state.strength < 0.45
                should_release = should_release or (child_state.conflict > 1.15 and energy < self.config.release_threshold + 1.25)
                if should_release:
                    self.release_child(region_state.region, child_state.child, reset_head=False)
                    releases += 1
                    active_children -= 1
            if active_children <= 0 and region_state.region != selected_region:
                self.release_region(region_state.region)
                releases += 1
        return releases

    def release_child(self, region: int, child: int, reset_head: bool) -> None:
        state = self.children[region][child]
        state.active = False
        state.center = None
        state.depth = 0.0
        state.strength = 0.0
        state.evidence = 0.0
        state.usefulness = 0.0
        state.dependency = 0.0
        state.conflict = 0.0
        state.downstream_support = 0.0
        state.admitted = False
        state.age = 0
        state.updates = 0
        state.dominant_group = "empty"
        state.group_mass = {}
        state.support_mass = {}
        state.released_count += 1
        self.memory[region][child] = []
        if reset_head:
            self.reset_child_head(region, child)

    def release_region(self, region: int) -> None:
        state = self.regions[region]
        state.active = False
        state.center = None
        state.depth = 0.0
        state.strength = 0.0
        state.evidence = 0.0
        state.usefulness = 0.0
        state.dependency = 0.0
        state.conflict = 0.0
        state.downstream_support = 0.0
        state.admitted = False
        state.age = 0
        state.updates = 0
        state.dominant_group = "empty"
        state.group_mass = {}
        state.support_mass = {}
        state.released_count += 1
        for child in range(self.config.children_per_region):
            self.release_child(region, child, reset_head=False)

    def predict(self, x: torch.Tensor) -> tuple[int, int, int, float]:
        candidates = []
        for region in self.active_regions():
            for child in self.active_children(region):
                state = self.children[region][child]
                if state.center is None:
                    raise RuntimeError(f"Active child {region}:{child} has no center.")
                candidates.append((cosine_distance(x, state.center), -survival_energy(state, self.config), region, child))
        if not candidates:
            return -1, -1, -1, 0.0
        candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        _, _, region, child = candidates[0]
        logits = self.forward_child(region, child, x)
        probs = torch.softmax(logits, dim=0)
        pred = int(torch.argmax(probs).detach().cpu())
        confidence = float(torch.max(probs).detach().cpu())
        return pred, region, child, confidence


def evaluate(model: BranchingNestedNet, prototypes: dict[str, torch.Tensor], config: Config) -> tuple[list[dict[str, float | str | int]], dict[str, float]]:
    rows: list[dict[str, float | str | int]] = []
    losses: list[float] = []
    protected_correct: list[float] = []
    branch_correct: list[float] = []
    noise_conf: list[float] = []
    for group, x in prototypes.items():
        label = LABELS[group]
        pred, region, child, confidence = model.predict(x)
        loss = math.nan
        if region >= 0 and child >= 0:
            with torch.no_grad():
                logits = model.forward_child(region, child, x)
                loss = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], device=model.device)).detach().cpu())
        correct = float(pred == label)
        if group in {"stable", "merge_a", "merge_b", "branch_root", "branch_up", "branch_down", "rare_critical"}:
            protected_correct.append(correct)
        if group in {"branch_root", "branch_up", "branch_down"}:
            branch_correct.append(correct)
        if group == "noise":
            noise_conf.append(confidence)
        losses.append(loss)
        child_state = model.children[region][child] if region >= 0 and child >= 0 else None
        rows.append(
            {
                "group": group,
                "label": label,
                "pred": pred,
                "region": region,
                "child": child,
                "correct": correct,
                "loss": loss,
                "confidence": confidence,
                "child_depth": child_state.depth if child_state is not None else 0.0,
                "child_strength": child_state.strength if child_state is not None else 0.0,
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


def region_table(model: BranchingNestedNet, config: Config) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for state in model.regions:
        rows.append(
            {
                "region": state.region,
                "active": int(state.active),
                "dominant_group": state.dominant_group,
                "dominant_share": state.dominant_share(),
                "depth": state.depth,
                "strength": state.strength,
                "evidence": state.evidence,
                "usefulness": state.usefulness,
                "dependency": state.dependency,
                "conflict": state.conflict,
                "downstream_support": state.downstream_support,
                "admitted": int(state.admitted),
                "age": state.age,
                "updates": state.updates,
                "survival": survival_energy(state, config),
                "active_children": len(model.active_children(state.region)),
            }
        )
    return rows


def child_table(model: BranchingNestedNet, config: Config) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for region in range(config.regions):
        for state in model.children[region]:
            rows.append(
                {
                    "region": region,
                    "child": state.child,
                    "active": int(state.active),
                    "dominant_group": state.dominant_group,
                    "dominant_share": state.dominant_share(),
                    "depth": state.depth,
                    "strength": state.strength,
                    "evidence": state.evidence,
                    "usefulness": state.usefulness,
                    "dependency": state.dependency,
                    "conflict": state.conflict,
                    "downstream_support": state.downstream_support,
                    "admitted": int(state.admitted),
                    "age": state.age,
                    "updates": state.updates,
                    "survival": survival_energy(state, config),
                    "memory": len(model.memory[region][state.child]),
                }
            )
    return rows


def run_sequence(
    config: Config,
) -> tuple[BranchingNestedNet, list[dict[str, float | str | int]], list[dict[str, float]], list[dict[str, float | str | int]], dict[str, float]]:
    validate_config(config)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    examples, prototypes = build_examples(config, device)
    model = BranchingNestedNet(config, device)
    event_rows: list[dict[str, float | str | int]] = []
    metric_rows: list[dict[str, float]] = []
    max_cross_region_delta = 0.0
    max_cross_child_delta = 0.0
    for index, example in enumerate(examples):
        contradiction_slots, contradiction_pressure = model.apply_contradiction(example)
        region, region_action, best_region_distance, best_region_survival = model.choose_region(example)
        if region < 0:
            raise RuntimeError("Outer region routing returned rejection; this experiment expects child-level rejection only.")
        if not model.regions[region].active or region_action in {"new_region", "new_branch_region"}:
            model.initialize_region(region, example)
        child, child_action, best_child_distance, best_child_survival = model.choose_child(region, example)
        if child < 0:
            release_count = model.decay_idle(region, -1)
            event_rows.append(
                {
                    "event_index": index,
                    "step": example.step,
                    "group": example.group,
                    "region_action": region_action,
                    "child_action": child_action,
                    "region": region,
                    "child": -1,
                    "loss": math.nan,
                    "best_region_distance": best_region_distance,
                    "best_child_distance": best_child_distance,
                    "best_region_survival": best_region_survival,
                    "best_child_survival": best_child_survival,
                    "protected_rows": 0.0,
                    "sibling_rows": 0.0,
                    "removed_fraction": 0.0,
                    "safe_fraction": 1.0,
                    "restore_ratio": 0.0,
                    "cross_region_delta": 0.0,
                    "cross_child_delta": 0.0,
                    "support_children": 0,
                    "support_amount": 0.0,
                    "contradiction_slots": contradiction_slots,
                    "contradiction_pressure": contradiction_pressure,
                    "released": release_count,
                }
            )
            continue
        if not model.children[region][child].active or child_action in {
            "new_child",
            "new_branch_child",
            "new_provisional_child",
            "branch_for_contradiction_child",
            "overwrite_low_survival_child",
        }:
            model.initialize_child(region, child, example, reset_head=True)
        diagnostics = model.train_child(region, child, example)
        model.update_region_state(region, example, region_action)
        model.update_child_state(region, child, example, child_action)
        support_children, support_amount = model.apply_support(example, region, child)
        release_count = model.decay_idle(region, child)
        max_cross_region_delta = max(max_cross_region_delta, diagnostics["cross_region_delta"])
        max_cross_child_delta = max(max_cross_child_delta, diagnostics["cross_child_delta"])
        active_children = [state for region_children in model.children for state in region_children if state.active]
        depths = np.array([state.depth for state in active_children], dtype=np.float64) if active_children else np.array([], dtype=np.float64)
        metric_rows.append(
            {
                "event_index": float(index),
                "step": float(example.step),
                "active_regions": float(len(model.active_regions())),
                "active_children": float(len(active_children)),
                "outer_mass": float(np.mean(depths <= 0.32 * (config.shell_count - 1))) if len(depths) else 0.0,
                "middle_mass": float(np.mean((depths > 0.32 * (config.shell_count - 1)) & (depths <= 0.72 * (config.shell_count - 1)))) if len(depths) else 0.0,
                "inner_mass": float(np.mean(depths > 0.72 * (config.shell_count - 1))) if len(depths) else 0.0,
                "mean_child_depth": float(np.mean(depths)) if len(depths) else 0.0,
                "max_cross_region_delta": max_cross_region_delta,
                "max_cross_child_delta": max_cross_child_delta,
            }
        )
        event_rows.append(
            {
                "event_index": index,
                "step": example.step,
                "group": example.group,
                "region_action": region_action,
                "child_action": child_action,
                "region": region,
                "child": child,
                "loss": diagnostics["loss"],
                "best_region_distance": best_region_distance,
                "best_child_distance": best_child_distance,
                "best_region_survival": best_region_survival,
                "best_child_survival": best_child_survival,
                "protected_rows": diagnostics["protected_rows"],
                "sibling_rows": diagnostics["sibling_rows"],
                "removed_fraction": diagnostics["removed_fraction"],
                "safe_fraction": diagnostics["safe_fraction"],
                "restore_ratio": diagnostics["restore_ratio"],
                "cross_region_delta": diagnostics["cross_region_delta"],
                "cross_child_delta": diagnostics["cross_child_delta"],
                "support_children": support_children,
                "support_amount": support_amount,
                "contradiction_slots": contradiction_slots,
                "contradiction_pressure": contradiction_pressure,
                "released": release_count,
            }
        )
    eval_rows, eval_summary = evaluate(model, prototypes, config)
    eval_summary["max_cross_region_delta"] = max_cross_region_delta
    eval_summary["max_cross_child_delta"] = max_cross_child_delta
    return model, event_rows, metric_rows, eval_rows, eval_summary


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


def plot_children(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    active = [row for row in rows if int(row["active"]) == 1]
    if not active:
        raise ValueError("Cannot plot children; no active child branches.")
    labels = [f"{int(row['region'])}:{int(row['child'])}\n{row['dominant_group']}" for row in active]
    depth = np.array([float(row["depth"]) for row in active], dtype=np.float64)
    strength = np.array([float(row["strength"]) for row in active], dtype=np.float64)
    colors = [GROUP_COLORS.get(str(row["dominant_group"]), "#333333") for row in active]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(np.arange(len(active)), depth, color=colors)
    axes[0].set_ylabel("child depth")
    axes[0].set_title("Internal child branches inside outer regions")
    axes[1].bar(np.arange(len(active)), strength, color=colors)
    axes[1].set_ylabel("child strength")
    axes[1].set_xticks(np.arange(len(active)))
    axes[1].set_xticklabels(labels, rotation=40, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics(metric_rows: list[dict[str, float]], output_path: Path) -> None:
    xs = np.array([row["event_index"] for row in metric_rows])
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(xs, [row["outer_mass"] for row in metric_rows], label="outer")
    axes[0].plot(xs, [row["middle_mass"] for row in metric_rows], label="middle")
    axes[0].plot(xs, [row["inner_mass"] for row in metric_rows], label="inner")
    axes[0].set_ylabel("child fraction")
    axes[0].legend()
    axes[1].plot(xs, [row["active_regions"] for row in metric_rows], label="regions")
    axes[1].plot(xs, [row["active_children"] for row in metric_rows], label="children")
    axes[1].set_ylabel("active count")
    axes[1].legend()
    axes[2].plot(xs, [row["mean_child_depth"] for row in metric_rows], color="#984ea3")
    axes[2].set_ylabel("mean child depth")
    axes[3].plot(xs, [row["max_cross_region_delta"] for row in metric_rows], label="outer leak")
    axes[3].plot(xs, [row["max_cross_child_delta"] for row in metric_rows], label="child head leak")
    axes[3].set_ylabel("max leaked delta")
    axes[3].set_xlabel("event index")
    axes[3].legend()
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(
    model: BranchingNestedNet,
    event_rows: list[dict[str, float | str | int]],
    metric_rows: list[dict[str, float]],
    eval_rows: list[dict[str, float | str | int]],
    eval_summary: dict[str, float],
    config: Config,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    region_rows = region_table(model, config)
    children = child_table(model, config)
    artifacts = {
        "event_log": config.output_dir / "event_log.csv",
        "metrics": config.output_dir / "metrics_over_time.csv",
        "regions": config.output_dir / "region_table.csv",
        "children": config.output_dir / "child_branch_table.csv",
        "eval": config.output_dir / "eval_by_group.csv",
        "summary": config.output_dir / "nested_branching_optimizer_summary.json",
        "child_plot": config.output_dir / "nested_branching_children.png",
        "metric_plot": config.output_dir / "nested_branching_metrics.png",
    }
    write_csv(artifacts["event_log"], event_rows)
    write_csv(artifacts["metrics"], metric_rows)
    write_csv(artifacts["regions"], region_rows)
    write_csv(artifacts["children"], children)
    write_csv(artifacts["eval"], eval_rows)
    plot_children(children, artifacts["child_plot"])
    plot_metrics(metric_rows, artifacts["metric_plot"])
    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "eval_summary": eval_summary,
        "regions": region_rows,
        "children": children,
        "eval_rows": eval_rows,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    with artifacts["summary"].open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def print_report(summary: dict[str, object]) -> None:
    eval_summary = summary["eval_summary"]
    children = summary["children"]
    artifacts = summary["artifacts"]
    if not isinstance(eval_summary, dict) or not isinstance(children, list) or not isinstance(artifacts, dict):
        raise RuntimeError("Malformed nested-branching summary.")
    print("\nTINY NESTED-BRANCHING OPTIMIZER")
    print("=" * 152)
    print(
        f"protected_acc={float(eval_summary['protected_acc']):.4f} "
        f"branch_acc={float(eval_summary['branch_acc']):.4f} "
        f"replacement={float(eval_summary['replacement_correct']):.4f} "
        f"obsolete_old={float(eval_summary['obsolete_old_correct']):.4f} "
        f"noise_conf={float(eval_summary['noise_confidence']):.4f} "
        f"max_cross_region_delta={float(eval_summary['max_cross_region_delta']):.3g} "
        f"max_cross_child_delta={float(eval_summary['max_cross_child_delta']):.3g}"
    )
    print("-" * 152)
    print(f"{'region':>6} {'child':>5} {'active':>6} {'group':>18} {'share':>8} {'depth':>8} {'strength':>10} {'survival':>10} {'memory':>7}")
    for row in children:
        if int(row["active"]) != 1:
            continue
        print(
            f"{int(row['region']):6d} {int(row['child']):5d} {int(row['active']):6d} {str(row['dominant_group']):>18} "
            f"{float(row['dominant_share']):8.4f} {float(row['depth']):8.4f} "
            f"{float(row['strength']):10.4f} {float(row['survival']):10.4f} {int(row['memory']):7d}"
        )
    print("\nWROTE")
    print("-" * 152)
    for _, path in artifacts.items():
        print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=("default", "long"), default="long")
    parser.add_argument("--mode", choices=("nested_sgd", "nested_tangent"), default="nested_tangent")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--regions", type=int, default=5)
    parser.add_argument("--children-per-region", type=int, default=3)
    parser.add_argument("--d-input", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--memory-limit", type=int, default=32)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=0.12)
    parser.add_argument("--trunk-lr-multiplier", type=float, default=0.12)
    parser.add_argument("--child-lr-multiplier", type=float, default=1.00)
    parser.add_argument("--depth-lr-decay", type=float, default=0.65)
    parser.add_argument("--strength-lr-decay", type=float, default=0.05)
    parser.add_argument("--center-lr", type=float, default=0.12)
    parser.add_argument("--child-center-lr", type=float, default=0.18)
    parser.add_argument("--match-threshold", type=float, default=0.20)
    parser.add_argument("--branch-threshold", type=float, default=0.42)
    parser.add_argument("--child-match-threshold", type=float, default=0.08)
    parser.add_argument("--child-branch-threshold", type=float, default=0.16)
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
    parser.add_argument("--protected-require-admitted", action="store_true", default=False)
    parser.add_argument("--protect-same-label", action="store_true", default=False)
    parser.add_argument("--no-sibling-protect", action="store_true", default=False)
    parser.add_argument("--max-events-per-step", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--projection-device", choices=("cpu", "same"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/06_nested_tangent_optimizer/results/gco-nested-branching-optimizer-seed0"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = Config(
        seed=args.seed,
        scenario=args.scenario,
        mode=args.mode,
        steps=args.steps,
        regions=args.regions,
        children_per_region=args.children_per_region,
        d_input=args.d_input,
        hidden=args.hidden,
        classes=args.classes,
        inner_steps=args.inner_steps,
        memory_limit=args.memory_limit,
        shell_count=args.shell_count,
        base_lr=args.base_lr,
        trunk_lr_multiplier=args.trunk_lr_multiplier,
        child_lr_multiplier=args.child_lr_multiplier,
        depth_lr_decay=args.depth_lr_decay,
        strength_lr_decay=args.strength_lr_decay,
        center_lr=args.center_lr,
        child_center_lr=args.child_center_lr,
        match_threshold=args.match_threshold,
        branch_threshold=args.branch_threshold,
        child_match_threshold=args.child_match_threshold,
        child_branch_threshold=args.child_branch_threshold,
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
        protected_require_admitted=args.protected_require_admitted,
        protect_same_label=args.protect_same_label,
        sibling_protect=not args.no_sibling_protect,
        max_events_per_step=args.max_events_per_step,
        device=args.device,
        projection_device=args.projection_device,
        output_dir=args.output_dir,
    )
    model, event_rows, metric_rows, eval_rows, eval_summary = run_sequence(config)
    summary = write_outputs(model, event_rows, metric_rows, eval_rows, eval_summary, config)
    print_report(summary)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
