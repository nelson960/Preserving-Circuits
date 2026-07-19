#!/usr/bin/env python3
"""Tiny nested-weight neural network with a nested-tangent optimizer.

This experiment tests the architectural idea that continual updates should not
push through one dense parameter block. The network has separated parameter
regions. Each region has its own tiny MLP, lifecycle state, and nested depth.
An update touches one selected region only; the script checks that all other
regions remain unchanged.

The optimizer combines:

* nested depth: outer regions learn fast, inner regions learn slowly;
* survival energy: recurrence alone is not enough to consolidate;
* invariant tangent: remove gradient components that damage protected examples
  inside the selected region;
* bounded restore: add a clipped correction toward protected behavior.

This is still a controlled synthetic experiment. It is meant to fail loudly
when the optimizer rule is wrong, so we can tune the math before scaling.
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

from gco_nested_geometry_dynamics import Config as ManualConfig, GroupSpec, build_events, default_group_specs, unit  # noqa: E402


GROUP_COLORS = {
    "stable": "#1b9e77",
    "merge_a": "#377eb8",
    "merge_b": "#4daf4a",
    "branch_root": "#984ea3",
    "branch_up": "#ff7f00",
    "branch_down": "#a65628",
    "rare_critical": "#e41a1c",
    "novel": "#f781bf",
    "obsolete_old": "#999999",
    "replacement": "#dede00",
    "noise": "#555555",
}


LABELS = {
    "stable": 0,
    "merge_a": 1,
    "merge_b": 1,
    "branch_root": 2,
    "branch_up": 3,
    "branch_down": 4,
    "rare_critical": 5,
    "obsolete_old": 6,
    "replacement": 7,
    "novel": 8,
    "noise": 9,
}


@dataclass(frozen=True)
class Config:
    seed: int
    scenario: str
    mode: str
    steps: int
    regions: int
    d_input: int
    hidden: int
    classes: int
    inner_steps: int
    memory_limit: int
    shell_count: int
    base_lr: float
    depth_lr_decay: float
    strength_lr_decay: float
    center_lr: float
    match_threshold: float
    branch_threshold: float
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
    max_events_per_step: int
    device: str
    projection_device: str
    output_dir: Path


@dataclass
class RegionState:
    region: int
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


@dataclass(frozen=True)
class Example:
    x: torch.Tensor
    y: int
    group: str
    usefulness: float
    dependency: float
    contradiction_target: str | None
    kind: str
    step: int


@dataclass(frozen=True)
class RegionMemory:
    x: torch.Tensor
    y: int
    group: str
    usefulness: float
    dependency: float
    admitted: bool


def finite_float(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")


def positive_float(name: str, value: float) -> None:
    finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    finite_float(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def validate_config(config: Config) -> None:
    if config.mode not in {"nested_sgd", "nested_tangent"}:
        raise ValueError("mode must be nested_sgd or nested_tangent.")
    if config.scenario not in {"default", "long"}:
        raise ValueError("scenario must be default or long.")
    if config.regions < 3:
        raise ValueError("regions must be at least 3.")
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
    for name in (
        "base_lr",
        "depth_lr_decay",
        "strength_lr_decay",
        "center_lr",
        "match_threshold",
        "branch_threshold",
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
        nonnegative_float(name, float(getattr(config, name)))
    if config.support_decay > 1.0:
        raise ValueError("support_decay must be in [0, 1].")
    if config.match_threshold > 2.0 or config.branch_threshold > 2.0 or config.support_threshold > 2.0:
        raise ValueError("distance thresholds are cosine distances and must be <= 2.")
    if config.low_potential_depth_cap > config.provisional_depth_cap:
        raise ValueError("low_potential_depth_cap cannot exceed provisional_depth_cap.")
    if config.projection_device not in {"cpu", "same"}:
        raise ValueError("projection_device must be cpu or same.")
    if config.max_events_per_step < 1:
        raise ValueError("max_events_per_step must be at least 1.")


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    raise ValueError("device must be one of: cpu, cuda, mps.")


def scenario_specs(config: Config) -> list[GroupSpec]:
    if config.scenario == "default":
        return default_group_specs(config.steps)
    steps = config.steps
    if steps < 96:
        raise ValueError("long scenario requires at least 96 steps.")
    return [
        GroupSpec("stable", (0.00, 1.00, 0.24), 5, 0.92, 0.96, 0, steps, 0.026),
        GroupSpec("merge_a", (0.90, 0.10, 0.32), 4, 0.66, 0.70, 0, steps, 0.032),
        GroupSpec("merge_b", (0.86, 0.18, 0.28), 4, 0.66, 0.70, 0, steps, 0.032),
        GroupSpec("branch_root", (-0.62, 0.54, 0.42), 4, 0.56, 0.58, 0, steps // 2, 0.035),
        GroupSpec("branch_up", (-0.32, 0.56, 0.76), 4, 0.72, 0.74, steps // 3, steps, 0.032),
        GroupSpec("branch_down", (-0.88, 0.20, -0.42), 4, 0.72, 0.74, steps // 3, steps, 0.032),
        GroupSpec("rare_critical", (0.20, -0.83, 0.52), 24, 1.50, 1.30, 0, steps, 0.022),
        GroupSpec("obsolete_old", (-0.12, -0.92, -0.35), 5, 0.58, 0.50, 0, steps // 2, 0.035),
        GroupSpec(
            "replacement",
            (0.13, -0.84, -0.50),
            6,
            0.86,
            1.00,
            steps // 2,
            steps,
            0.030,
            contradiction_target="obsolete_old",
        ),
        GroupSpec("novel", (0.62, -0.44, 0.70), 20, 0.70, 0.72, (2 * steps) // 3, steps, 0.032),
        GroupSpec("noise", (-0.58, -0.15, 0.79), 37, 0.05, 0.08, 0, steps, 0.23),
    ]


def manual_config(config: Config) -> ManualConfig:
    return ManualConfig(
        seed=config.seed,
        steps=config.steps,
        slots=config.regions,
        shell_count=config.shell_count,
        inner_radius=0.85,
        outer_radius=3.0,
        match_threshold=0.10,
        branch_threshold=0.22,
        initial_strength=0.55,
        evidence_gain=config.evidence_gain,
        usefulness_gain=config.usefulness_gain,
        dependency_gain=config.dependency_gain,
        conflict_gain=config.conflict_gain,
        age_decay=config.age_decay,
        outer_learning_rate=0.42,
        learning_timescale=config.depth_lr_decay,
        outward_decay=0.055,
        decay_timescale=0.72,
        inward_rate=config.inward_rate,
        outward_rate=config.outward_rate,
        survival_temperature=config.survival_temperature,
        release_threshold=config.release_threshold,
        overwrite_margin=config.overwrite_margin,
        admission_threshold=config.admission_threshold,
        admission_temperature=config.admission_temperature,
        provisional_depth_cap=config.provisional_depth_cap,
        provisional_decay_multiplier=2.50,
        contradiction_release_threshold=1.15,
        support_threshold=config.support_threshold,
        support_gain=config.support_gain,
        support_min_potential=config.min_consolidation_potential,
        support_decay=config.support_decay,
        support_admission_gain=1.20,
        support_diversity_gain=0.70,
        max_events_per_step=config.max_events_per_step,
        output_dir=config.output_dir,
    )


def orthonormal_embedding(seed: int, d_input: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 33)
    matrix = rng.normal(size=(3, d_input)).astype(np.float32)
    matrix[:3, :3] += np.eye(3, dtype=np.float32) * 2.0
    q, _ = np.linalg.qr(matrix.T)
    basis = q[:, :3].T.astype(np.float32)
    if basis.shape != (3, d_input):
        raise RuntimeError(f"Unexpected embedding basis shape {basis.shape}.")
    return basis


def normalize_np(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Cannot normalize zero/non-finite vector.")
    return vector / norm


def embed(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return normalize_np(vector.astype(np.float32) @ basis).astype(np.float32)


def build_examples(config: Config, device: torch.device) -> tuple[list[Example], dict[str, torch.Tensor]]:
    specs = scenario_specs(config)
    manual = manual_config(config)
    basis = orthonormal_embedding(config.seed, config.d_input)
    prototypes = {spec.name: torch.tensor(embed(unit(np.array(spec.center, dtype=np.float64)), basis), device=device) for spec in specs}
    examples: list[Example] = []
    for event in build_events(manual, specs):
        x = torch.tensor(embed(event.vector, basis), device=device)
        if event.group == "noise":
            label = (event.step + event.vector.argmax().item()) % config.classes
        else:
            label = LABELS[event.group]
        examples.append(
            Example(
                x=x,
                y=int(label),
                group=event.group,
                usefulness=event.usefulness,
                dependency=event.dependency,
                contradiction_target=event.contradiction_target,
                kind=event.kind,
                step=event.step,
            )
        )
    if not examples:
        raise RuntimeError("No examples generated.")
    return examples, prototypes


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = F.normalize(a.detach().float(), dim=0)
    b_norm = F.normalize(b.detach().float(), dim=0)
    value = float((1.0 - torch.dot(a_norm, b_norm)).detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite cosine distance: {value}.")
    return value


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def survival_energy(region: RegionState, config: Config) -> float:
    if not region.active:
        return -math.inf
    evidence = math.log1p(max(0.0, region.evidence))
    support = math.log1p(max(0.0, region.downstream_support))
    diversity = math.log1p(float(len(region.support_mass)))
    return float(
        config.evidence_gain * evidence
        + config.usefulness_gain * region.usefulness
        + config.dependency_gain * region.dependency
        + 0.60 * support
        + 0.35 * diversity
        - config.conflict_gain * region.conflict
        - config.age_decay * float(region.age)
    )


def consolidation_potential(region: RegionState) -> float:
    support = math.log1p(max(0.0, region.downstream_support))
    diversity = math.log1p(float(len(region.support_mass)))
    return float(region.usefulness + region.dependency + 0.25 * support + 0.15 * diversity)


def update_admission(region: RegionState, config: Config) -> None:
    if not region.active:
        region.admitted = False
        return
    if consolidation_potential(region) < config.min_consolidation_potential:
        region.admitted = False
        return
    probability = sigmoid((survival_energy(region, config) - config.admission_threshold) / max(1e-6, config.admission_temperature))
    if probability >= 0.5:
        region.admitted = True
    if region.conflict > 1.15:
        region.admitted = False


def adjust_depth(region: RegionState, config: Config, multiplier: float) -> None:
    if not region.active:
        return
    max_depth = float(config.shell_count - 1)
    energy = survival_energy(region, config)
    desired = max_depth * sigmoid((energy - 1.35) / max(1e-6, config.survival_temperature))
    if consolidation_potential(region) < config.min_consolidation_potential:
        desired = min(desired, config.low_potential_depth_cap)
    if not region.admitted:
        desired = min(desired, config.provisional_depth_cap)
    if region.conflict > 0.65:
        desired *= 1.0 - sigmoid((region.conflict - 0.65) / max(1e-6, config.survival_temperature))
    rate = config.inward_rate if desired >= region.depth else config.outward_rate
    region.depth = min(max(region.depth + multiplier * rate * (desired - region.depth), 0.0), max_depth)


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    flats = [tensor.reshape(-1) for tensor in tensors]
    if not flats:
        raise ValueError("Cannot flatten empty tensor list.")
    return torch.cat(flats)


def unflatten_like(flat: torch.Tensor, params: list[torch.Tensor]) -> list[torch.Tensor]:
    pieces: list[torch.Tensor] = []
    cursor = 0
    for param in params:
        count = param.numel()
        pieces.append(flat[cursor : cursor + count].reshape_as(param))
        cursor += count
    if cursor != flat.numel():
        raise RuntimeError("Flat gradient size does not match parameter sizes.")
    return pieces


def grad_or_raise(loss: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)
    return flatten_tensors(grads)


def project_gradient(
    gradient: torch.Tensor,
    constraint_rows: list[torch.Tensor],
    damping: float,
    projection_device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not constraint_rows:
        return gradient, {"rows": 0.0, "removed_fraction": 0.0, "safe_fraction": 1.0}
    target_device = torch.device("cpu") if projection_device == "cpu" else gradient.device
    g = gradient.detach().to(target_device, dtype=torch.float32)
    rows = torch.stack([row.detach().to(target_device, dtype=torch.float32) for row in constraint_rows], dim=0)
    row_norms = torch.linalg.vector_norm(rows, dim=1).clamp_min(1e-8)
    rows = rows / row_norms[:, None]
    gram = rows @ rows.T
    identity = torch.eye(gram.shape[0], device=target_device, dtype=torch.float32)
    rhs = rows @ g
    coeff = torch.linalg.solve(gram + damping * identity, rhs)
    removed = rows.T @ coeff
    safe = g - removed
    g_norm = torch.linalg.vector_norm(g).clamp_min(1e-12)
    removed_fraction = float((torch.linalg.vector_norm(removed) / g_norm).detach().cpu())
    safe_fraction = float((torch.linalg.vector_norm(safe) / g_norm).detach().cpu())
    return safe.to(gradient.device, dtype=gradient.dtype), {
        "rows": float(rows.shape[0]),
        "removed_fraction": removed_fraction,
        "safe_fraction": safe_fraction,
    }


def clip_relative(gradient: torch.Tensor, reference: torch.Tensor, ratio: float) -> torch.Tensor:
    grad_norm = torch.linalg.vector_norm(gradient)
    ref_norm = torch.linalg.vector_norm(reference)
    max_norm = ratio * ref_norm
    if float(grad_norm.detach().cpu()) <= float(max_norm.detach().cpu()) or float(grad_norm.detach().cpu()) <= 1e-12:
        return gradient
    return gradient * (max_norm / grad_norm)


class NestedRegionNet:
    def __init__(self, config: Config, device: torch.device) -> None:
        self.config = config
        self.device = device
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 71)
        self.w1 = [torch.randn(config.d_input, config.hidden, generator=generator, dtype=torch.float32, device=device) * 0.18 for _ in range(config.regions)]
        self.b1 = [torch.zeros(config.hidden, dtype=torch.float32, device=device) for _ in range(config.regions)]
        self.w2 = [torch.randn(config.hidden, config.classes, generator=generator, dtype=torch.float32, device=device) * 0.18 for _ in range(config.regions)]
        self.b2 = [torch.zeros(config.classes, dtype=torch.float32, device=device) for _ in range(config.regions)]
        self.regions = [RegionState(region=index) for index in range(config.regions)]
        self.memory: list[list[RegionMemory]] = [[] for _ in range(config.regions)]

    def region_params(self, region: int) -> list[torch.Tensor]:
        return [self.w1[region], self.b1[region], self.w2[region], self.b2[region]]

    def param_snapshot(self) -> list[torch.Tensor]:
        return [tensor.detach().clone() for params in zip(self.w1, self.b1, self.w2, self.b2) for tensor in params]

    def cross_region_delta(self, before: list[torch.Tensor], selected: int) -> float:
        after = [tensor for params in zip(self.w1, self.b1, self.w2, self.b2) for tensor in params]
        deltas: list[float] = []
        cursor = 0
        for region in range(self.config.regions):
            for _ in range(4):
                if region != selected:
                    deltas.append(float(torch.max(torch.abs(after[cursor].detach() - before[cursor])).cpu()))
                cursor += 1
        return max(deltas) if deltas else 0.0

    def forward_region(self, region: int, x: torch.Tensor, params: list[torch.Tensor] | None = None) -> torch.Tensor:
        if params is None:
            w1, b1, w2, b2 = self.region_params(region)
        else:
            w1, b1, w2, b2 = params
        hidden = torch.tanh(x @ w1 + b1)
        return hidden @ w2 + b2

    def active_indices(self) -> list[int]:
        return [region.region for region in self.regions if region.active]

    def choose_region(self, example: Example) -> tuple[int, str, float, float]:
        active = self.active_indices()
        if not active:
            return 0, "new_region", math.inf, -math.inf
        distances = []
        for index in active:
            state = self.regions[index]
            if state.center is None:
                raise RuntimeError(f"Active region {index} has no center.")
            distances.append((cosine_distance(example.x, state.center), -survival_energy(state, self.config), index))
        distances.sort(key=lambda row: (row[0], row[1], row[2]))
        best_distance, negative_survival, best_index = distances[0]
        best_survival = -negative_survival
        free = next((region.region for region in self.regions if not region.active), None)
        low_potential = example.usefulness + example.dependency < self.config.min_consolidation_potential
        if low_potential and self.regions[best_index].admitted:
            if free is not None:
                return free, "new_provisional_region", best_distance, best_survival
            provisional = [index for index in active if not self.regions[index].admitted]
            if provisional:
                chosen = min(provisional, key=lambda idx: (survival_energy(self.regions[idx], self.config), self.regions[idx].strength, -self.regions[idx].age, idx))
                return chosen, "provisional_low_potential", best_distance, survival_energy(self.regions[chosen], self.config)
            return -1, "reject_low_potential", best_distance, best_survival
        if example.contradiction_target is not None:
            target_regions = [index for index in active if example.contradiction_target in self.regions[index].group_mass]
            if target_regions and free is not None:
                return free, "branch_for_contradiction", best_distance, best_survival
        if best_distance <= self.config.match_threshold:
            return best_index, "matched_region", best_distance, best_survival
        if best_distance >= self.config.branch_threshold and free is not None:
            return free, "new_branch_region", best_distance, best_survival
        if free is not None:
            return free, "new_region", best_distance, best_survival
        weakest = min(active, key=lambda idx: (self.regions[idx].admitted, survival_energy(self.regions[idx], self.config), self.regions[idx].strength, -self.regions[idx].age, idx))
        weakest_survival = survival_energy(self.regions[weakest], self.config)
        if example.usefulness + example.dependency + self.config.overwrite_margin > weakest_survival:
            return weakest, "overwrite_low_survival", best_distance, weakest_survival
        return best_index, "forced_match_capacity_full", best_distance, best_survival

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
        if consolidation_potential(state) < self.config.min_consolidation_potential:
            state.strength = min(state.strength, self.config.low_potential_strength_cap)
        adjust_depth(state, self.config, multiplier=1.0)
        self.memory[region] = []

    def update_center(self, region: int, x: torch.Tensor) -> None:
        state = self.regions[region]
        if state.center is None:
            raise RuntimeError(f"Region {region} has no center.")
        rate = self.config.center_lr * math.exp(-0.4 * state.depth)
        state.center = F.normalize((1.0 - rate) * state.center + rate * x.detach(), dim=0)

    def region_loss(self, region: int, examples: list[RegionMemory | Example], params: list[torch.Tensor] | None = None) -> torch.Tensor:
        if not examples:
            raise ValueError("Cannot compute region loss for empty example list.")
        losses = []
        for item in examples:
            logits = self.forward_region(region, item.x, params=params)
            target = torch.tensor([item.y], device=self.device, dtype=torch.long)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        return torch.stack(losses).mean()

    def protected_examples(self, region: int, current_group: str) -> list[RegionMemory]:
        protected = [
            item
            for item in self.memory[region]
            if item.group != current_group
            and item.usefulness + item.dependency >= self.config.protected_min_potential
            and (item.admitted or not self.config.protected_require_admitted)
        ]
        return protected[-self.config.memory_limit :]

    def train_region(self, region: int, example: Example) -> dict[str, float]:
        state = self.regions[region]
        if not state.active:
            raise RuntimeError(f"Cannot train inactive region {region}.")
        params = self.region_params(region)
        before = self.param_snapshot()
        diagnostics = {
            "protected_rows": 0.0,
            "removed_fraction": 0.0,
            "safe_fraction": 1.0,
            "restore_ratio": 0.0,
            "loss": math.nan,
            "cross_region_delta": 0.0,
        }
        protected = self.protected_examples(region, example.group)
        for _ in range(self.config.inner_steps):
            params_for_grad = [param.detach().clone().requires_grad_(True) for param in params]
            loss_new = self.region_loss(region, [example], params=params_for_grad)
            raw_gradient = grad_or_raise(loss_new, params_for_grad)
            safe_gradient = raw_gradient
            projection_stats = {"rows": 0.0, "removed_fraction": 0.0, "safe_fraction": 1.0}
            restore_gradient = torch.zeros_like(raw_gradient)
            if self.config.mode == "nested_tangent" and protected:
                rows: list[torch.Tensor] = []
                for item in protected:
                    protected_params = [param.detach().clone().requires_grad_(True) for param in params]
                    protected_loss = self.region_loss(region, [item], params=protected_params)
                    rows.append(grad_or_raise(protected_loss, protected_params).detach())
                safe_gradient, projection_stats = project_gradient(
                    raw_gradient,
                    rows,
                    self.config.tangent_damping,
                    self.config.projection_device,
                )
                restore_params = [param.detach().clone().requires_grad_(True) for param in params]
                restore_loss = self.region_loss(region, protected, params=restore_params)
                restore_gradient = grad_or_raise(restore_loss, restore_params).detach()
                restore_gradient = clip_relative(restore_gradient, safe_gradient, self.config.restore_clip_ratio)
            final_gradient = safe_gradient + self.config.restore_weight * restore_gradient
            region_lr = self.config.base_lr * math.exp(-self.config.depth_lr_decay * state.depth) * math.exp(-self.config.strength_lr_decay * state.strength)
            pieces = unflatten_like(final_gradient, params)
            with torch.no_grad():
                for param, grad_piece in zip(params, pieces):
                    param -= region_lr * grad_piece
            diagnostics["loss"] = float(loss_new.detach().cpu())
            diagnostics["protected_rows"] = projection_stats["rows"]
            diagnostics["removed_fraction"] = projection_stats["removed_fraction"]
            diagnostics["safe_fraction"] = projection_stats["safe_fraction"]
            raw_norm = torch.linalg.vector_norm(raw_gradient).clamp_min(1e-12)
            diagnostics["restore_ratio"] = float((torch.linalg.vector_norm(self.config.restore_weight * restore_gradient) / raw_norm).detach().cpu())
        diagnostics["cross_region_delta"] = self.cross_region_delta(before, region)
        if diagnostics["cross_region_delta"] > 1e-7:
            raise RuntimeError(f"Update leaked into non-selected regions: {diagnostics['cross_region_delta']:.6g}")
        return diagnostics

    def apply_contradiction(self, example: Example) -> tuple[int, float]:
        if example.contradiction_target is None:
            return 0, 0.0
        touched = 0
        pressure_total = 0.0
        for state in self.regions:
            if not state.active:
                continue
            target_mass = state.group_mass.get(example.contradiction_target, 0.0)
            if target_mass <= 0.0:
                continue
            mass = max(1e-12, sum(state.group_mass.values()))
            share = target_mass / mass
            pressure = share * share * (1.0 + example.usefulness + example.dependency)
            state.conflict = 0.82 * state.conflict + pressure
            reduced = target_mass * max(0.0, 1.0 - 0.45 * share)
            if reduced < 0.15:
                del state.group_mass[example.contradiction_target]
            else:
                state.group_mass[example.contradiction_target] = reduced
            if state.group_mass:
                state.dominant_group = max(state.group_mass.items(), key=lambda row: (row[1], row[0]))[0]
            else:
                state.dominant_group = "empty"
            state.usefulness *= 1.0 - 0.10 * share
            state.strength *= 1.0 - 0.08 * share
            update_admission(state, self.config)
            touched += 1
            pressure_total += pressure
        return touched, pressure_total

    def apply_support(self, example: Example, selected: int) -> tuple[int, float]:
        potential = example.usefulness + example.dependency
        if potential < self.config.min_consolidation_potential:
            return 0, 0.0
        touched = 0
        support_total = 0.0
        for state in self.regions:
            if not state.active or state.region == selected or state.center is None:
                continue
            if example.group in state.group_mass:
                continue
            distance = cosine_distance(example.x, state.center)
            if distance > self.config.support_threshold:
                continue
            closeness = 1.0 - distance / max(1e-12, self.config.support_threshold)
            support = self.config.support_gain * closeness * potential
            state.evidence += support
            state.downstream_support = min(25.0, state.downstream_support + support)
            state.support_mass[example.group] = state.support_mass.get(example.group, 0.0) + support
            state.usefulness = max(state.usefulness, 0.94 * state.usefulness + 0.06 * example.usefulness)
            state.dependency = max(state.dependency, 0.94 * state.dependency + 0.06 * example.dependency)
            state.strength = min(10.0, state.strength + 0.12 * support)
            state.age = max(0, state.age - 1)
            update_admission(state, self.config)
            adjust_depth(state, self.config, multiplier=0.25)
            touched += 1
            support_total += support
        return touched, support_total

    def update_region_state(self, region: int, example: Example, action: str) -> None:
        state = self.regions[region]
        if action in {"new_region", "new_branch_region", "new_provisional_region", "branch_for_contradiction"} or not state.active:
            self.initialize_region(region, example)
            state = self.regions[region]
        elif action == "overwrite_low_survival":
            state.released_count += 1
            self.initialize_region(region, example)
            state = self.regions[region]
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
        self.update_center(region, example.x)
        self.memory[region].append(
            RegionMemory(
                x=example.x.detach().clone(),
                y=example.y,
                group=example.group,
                usefulness=example.usefulness,
                dependency=example.dependency,
                admitted=state.admitted,
            )
        )
        if len(self.memory[region]) > self.config.memory_limit:
            self.memory[region] = self.memory[region][-self.config.memory_limit :]

    def decay_idle_regions(self, selected: int) -> int:
        releases = 0
        for state in self.regions:
            if not state.active or state.region == selected:
                continue
            state.age += 1
            decay = 0.045 * math.exp(-0.72 * state.depth)
            if not state.admitted:
                decay *= 2.50
            state.strength *= max(0.0, 1.0 - decay)
            state.evidence *= max(0.0, 1.0 - 0.35 * decay)
            state.conflict *= 0.985
            state.downstream_support *= self.config.support_decay
            if state.support_mass:
                state.support_mass = {
                    group: mass * self.config.support_decay
                    for group, mass in state.support_mass.items()
                    if mass * self.config.support_decay > 1e-6
                }
            update_admission(state, self.config)
            adjust_depth(state, self.config, multiplier=0.18)
            energy = survival_energy(state, self.config)
            should_release = (not state.admitted) and energy < self.config.release_threshold and state.strength < 0.45
            should_release = should_release or (state.conflict > 1.15 and energy < self.config.release_threshold + 1.25)
            if should_release:
                self.release_region(state.region)
                releases += 1
        return releases

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
        self.memory[region] = []

    def predict(self, x: torch.Tensor) -> tuple[int, int, float]:
        active = self.active_indices()
        if not active:
            return -1, -1, 0.0
        ranked = []
        for index in active:
            center = self.regions[index].center
            if center is None:
                raise RuntimeError(f"Active region {index} has no center.")
            ranked.append((cosine_distance(x, center), index))
        ranked.sort(key=lambda row: (row[0], row[1]))
        region = ranked[0][1]
        logits = self.forward_region(region, x)
        probs = torch.softmax(logits, dim=0)
        pred = int(torch.argmax(probs).detach().cpu())
        confidence = float(torch.max(probs).detach().cpu())
        return pred, region, confidence


def evaluate(model: NestedRegionNet, prototypes: dict[str, torch.Tensor], config: Config) -> tuple[list[dict[str, float | str | int]], dict[str, float]]:
    rows: list[dict[str, float | str | int]] = []
    losses = []
    protected_correct = []
    branch_correct = []
    noise_conf = []
    for group, x in prototypes.items():
        label = LABELS[group]
        pred, region, confidence = model.predict(x)
        loss = math.nan
        if region >= 0:
            with torch.no_grad():
                logits = model.forward_region(region, x)
                loss = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], device=model.device)).detach().cpu())
        correct = float(pred == label)
        if group in {"stable", "merge_a", "merge_b", "branch_root", "branch_up", "branch_down", "rare_critical"}:
            protected_correct.append(correct)
        if group in {"branch_root", "branch_up", "branch_down"}:
            branch_correct.append(correct)
        if group == "noise":
            noise_conf.append(confidence)
        losses.append(loss)
        state = model.regions[region] if region >= 0 else None
        rows.append(
            {
                "group": group,
                "label": label,
                "pred": pred,
                "region": region,
                "correct": correct,
                "loss": loss,
                "confidence": confidence,
                "depth": state.depth if state is not None else 0.0,
                "strength": state.strength if state is not None else 0.0,
            }
        )
    replacement = next(row for row in rows if row["group"] == "replacement")
    obsolete = next(row for row in rows if row["group"] == "obsolete_old")
    summary = {
        "eval_loss": float(np.nanmean(losses)),
        "protected_acc": float(np.mean(protected_correct)) if protected_correct else math.nan,
        "branch_acc": float(np.mean(branch_correct)) if branch_correct else math.nan,
        "replacement_correct": float(replacement["correct"]),
        "obsolete_old_correct": float(obsolete["correct"]),
        "noise_confidence": float(np.mean(noise_conf)) if noise_conf else math.nan,
    }
    return rows, summary


def region_table(model: NestedRegionNet, config: Config) -> list[dict[str, float | str | int]]:
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
                "support_diversity": len(state.support_mass),
                "admitted": int(state.admitted),
                "age": state.age,
                "updates": state.updates,
                "survival": survival_energy(state, config),
                "memory": len(model.memory[state.region]),
            }
        )
    return rows


def run_sequence(config: Config) -> tuple[NestedRegionNet, list[dict[str, float | str | int]], list[dict[str, float]], list[dict[str, float | str | int]], dict[str, float]]:
    validate_config(config)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    examples, prototypes = build_examples(config, device)
    model = NestedRegionNet(config, device)
    event_rows: list[dict[str, float | str | int]] = []
    metric_rows: list[dict[str, float]] = []
    max_cross_region_delta = 0.0
    for index, example in enumerate(examples):
        contradiction_slots, contradiction_pressure = model.apply_contradiction(example)
        region, action, best_distance, best_survival = model.choose_region(example)
        if region < 0:
            release_count = model.decay_idle_regions(region)
            event_rows.append(
                {
                    "event_index": index,
                    "step": example.step,
                    "group": example.group,
                    "action": action,
                    "region": -1,
                    "loss": math.nan,
                    "best_distance": best_distance,
                    "best_survival": best_survival,
                    "protected_rows": 0.0,
                    "removed_fraction": 0.0,
                    "safe_fraction": 1.0,
                    "restore_ratio": 0.0,
                    "cross_region_delta": 0.0,
                    "support_slots": 0,
                    "support_amount": 0.0,
                    "contradiction_slots": contradiction_slots,
                    "contradiction_pressure": contradiction_pressure,
                    "released": release_count,
                }
            )
            continue
        if not model.regions[region].active or action in {"new_region", "new_branch_region", "new_provisional_region", "branch_for_contradiction", "overwrite_low_survival"}:
            model.initialize_region(region, example)
        diagnostics = model.train_region(region, example)
        max_cross_region_delta = max(max_cross_region_delta, diagnostics["cross_region_delta"])
        model.update_region_state(region, example, action)
        support_slots, support_amount = model.apply_support(example, region)
        release_count = model.decay_idle_regions(region)
        active = [state for state in model.regions if state.active]
        depths = np.array([state.depth for state in active], dtype=np.float64) if active else np.array([], dtype=np.float64)
        metric_rows.append(
            {
                "event_index": float(index),
                "step": float(example.step),
                "active_regions": float(len(active)),
                "outer_mass": float(np.mean(depths <= 0.32 * (config.shell_count - 1))) if len(depths) else 0.0,
                "middle_mass": float(np.mean((depths > 0.32 * (config.shell_count - 1)) & (depths <= 0.72 * (config.shell_count - 1)))) if len(depths) else 0.0,
                "inner_mass": float(np.mean(depths > 0.72 * (config.shell_count - 1))) if len(depths) else 0.0,
                "mean_depth": float(np.mean(depths)) if len(depths) else 0.0,
                "max_cross_region_delta": max_cross_region_delta,
            }
        )
        event_rows.append(
            {
                "event_index": index,
                "step": example.step,
                "group": example.group,
                "action": action,
                "region": region,
                "loss": diagnostics["loss"],
                "best_distance": best_distance,
                "best_survival": best_survival,
                "protected_rows": diagnostics["protected_rows"],
                "removed_fraction": diagnostics["removed_fraction"],
                "safe_fraction": diagnostics["safe_fraction"],
                "restore_ratio": diagnostics["restore_ratio"],
                "cross_region_delta": diagnostics["cross_region_delta"],
                "support_slots": support_slots,
                "support_amount": support_amount,
                "contradiction_slots": contradiction_slots,
                "contradiction_pressure": contradiction_pressure,
                "released": release_count,
            }
        )
    eval_rows, eval_summary = evaluate(model, prototypes, config)
    eval_summary["max_cross_region_delta"] = max_cross_region_delta
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


def plot_regions(region_rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    regions = [int(row["region"]) for row in region_rows]
    depth = np.array([float(row["depth"]) for row in region_rows])
    strength = np.array([float(row["strength"]) for row in region_rows])
    labels = [str(row["dominant_group"]) for row in region_rows]
    colors = [GROUP_COLORS.get(label, "#333333") for label in labels]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].bar(regions, depth, color=colors)
    axes[0].set_ylabel("depth")
    axes[0].set_title("Nested parameter regions")
    axes[1].bar(regions, strength, color=colors)
    axes[1].set_ylabel("strength")
    axes[1].set_xlabel("region")
    axes[1].set_xticks(regions)
    axes[1].set_xticklabels([f"{idx}:{label}" for idx, label in zip(regions, labels)], rotation=30, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics(metric_rows: list[dict[str, float]], output_path: Path) -> None:
    xs = np.array([row["event_index"] for row in metric_rows])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    axes[0].plot(xs, [row["outer_mass"] for row in metric_rows], label="outer")
    axes[0].plot(xs, [row["middle_mass"] for row in metric_rows], label="middle")
    axes[0].plot(xs, [row["inner_mass"] for row in metric_rows], label="inner")
    axes[0].legend()
    axes[0].set_ylabel("region fraction")
    axes[1].plot(xs, [row["mean_depth"] for row in metric_rows], color="#984ea3")
    axes[1].set_ylabel("mean depth")
    axes[2].plot(xs, [row["max_cross_region_delta"] for row in metric_rows], color="#e41a1c")
    axes[2].set_ylabel("max leaked delta")
    axes[2].set_xlabel("event index")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(
    model: NestedRegionNet,
    event_rows: list[dict[str, float | str | int]],
    metric_rows: list[dict[str, float]],
    eval_rows: list[dict[str, float | str | int]],
    eval_summary: dict[str, float],
    config: Config,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    region_rows = region_table(model, config)
    artifacts = {
        "event_log": config.output_dir / "event_log.csv",
        "metrics": config.output_dir / "metrics_over_time.csv",
        "regions": config.output_dir / "region_table.csv",
        "eval": config.output_dir / "eval_by_group.csv",
        "summary": config.output_dir / "nested_tangent_optimizer_summary.json",
        "region_plot": config.output_dir / "nested_tangent_regions.png",
        "metric_plot": config.output_dir / "nested_tangent_metrics.png",
    }
    write_csv(artifacts["event_log"], event_rows)
    write_csv(artifacts["metrics"], metric_rows)
    write_csv(artifacts["regions"], region_rows)
    write_csv(artifacts["eval"], eval_rows)
    plot_regions(region_rows, artifacts["region_plot"])
    plot_metrics(metric_rows, artifacts["metric_plot"])
    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "eval_summary": eval_summary,
        "regions": region_rows,
        "eval_rows": eval_rows,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    with artifacts["summary"].open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def print_report(summary: dict[str, object]) -> None:
    eval_summary = summary["eval_summary"]
    regions = summary["regions"]
    artifacts = summary["artifacts"]
    if not isinstance(eval_summary, dict) or not isinstance(regions, list) or not isinstance(artifacts, dict):
        raise RuntimeError("Malformed nested-tangent summary.")
    print("\nTINY NESTED-TANGENT OPTIMIZER")
    print("=" * 144)
    print(
        f"protected_acc={float(eval_summary['protected_acc']):.4f} "
        f"branch_acc={float(eval_summary['branch_acc']):.4f} "
        f"replacement={float(eval_summary['replacement_correct']):.4f} "
        f"obsolete_old={float(eval_summary['obsolete_old_correct']):.4f} "
        f"noise_conf={float(eval_summary['noise_confidence']):.4f} "
        f"max_cross_region_delta={float(eval_summary['max_cross_region_delta']):.3g}"
    )
    print("-" * 144)
    print(f"{'region':>6} {'active':>6} {'group':>18} {'share':>8} {'depth':>8} {'strength':>10} {'survival':>10} {'memory':>7}")
    for row in regions:
        print(
            f"{int(row['region']):6d} {int(row['active']):6d} {str(row['dominant_group']):>18} "
            f"{float(row['dominant_share']):8.4f} {float(row['depth']):8.4f} "
            f"{float(row['strength']):10.4f} {float(row['survival']):10.4f} {int(row['memory']):7d}"
        )
    print("\nWROTE")
    print("-" * 144)
    for _, path in artifacts.items():
        print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=("default", "long"), default="default")
    parser.add_argument("--mode", choices=("nested_sgd", "nested_tangent"), default="nested_tangent")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--regions", type=int, default=8)
    parser.add_argument("--d-input", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--memory-limit", type=int, default=32)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=0.12)
    parser.add_argument("--depth-lr-decay", type=float, default=0.65)
    parser.add_argument("--strength-lr-decay", type=float, default=0.05)
    parser.add_argument("--center-lr", type=float, default=0.18)
    parser.add_argument("--match-threshold", type=float, default=0.08)
    parser.add_argument("--branch-threshold", type=float, default=0.16)
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
    parser.add_argument("--max-events-per-step", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--projection-device", choices=("cpu", "same"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/06_nested_tangent_optimizer/results/gco-nested-tangent-optimizer-seed0"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = Config(
        seed=args.seed,
        scenario=args.scenario,
        mode=args.mode,
        steps=args.steps,
        regions=args.regions,
        d_input=args.d_input,
        hidden=args.hidden,
        classes=args.classes,
        inner_steps=args.inner_steps,
        memory_limit=args.memory_limit,
        shell_count=args.shell_count,
        base_lr=args.base_lr,
        depth_lr_decay=args.depth_lr_decay,
        strength_lr_decay=args.strength_lr_decay,
        center_lr=args.center_lr,
        match_threshold=args.match_threshold,
        branch_threshold=args.branch_threshold,
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
