#!/usr/bin/env python3
"""Neural nested-geometry slot sandbox.

This is the first neural version of the manual nested-geometry dynamics. Each
slot is no longer a single hand-updated vector. It is a learnable affine
subspace with a center and a low-rank basis:

    z_k(h) = c_k + U_k a_k

The experiment keeps the transparent lifecycle state from the manual sandbox:
depth, strength, evidence, downstream support, admission, release, and
capacity. The difference is that the slot geometry itself is learned by
gradient descent on reconstruction error. This tests whether nested geometry
can be implemented as trainable neural memory regions before connecting it to
a full continual-learning model.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gco_nested_geometry_dynamics import (  # noqa: E402
    GROUP_COLORS,
    Config as ManualConfig,
    GroupSpec,
    build_events,
    default_group_specs,
    unit,
)


@dataclass(frozen=True)
class Config:
    seed: int
    scenario: str
    steps: int
    slots: int
    d_model: int
    subspace_rank: int
    inner_updates: int
    shell_count: int
    inner_radius: float
    outer_radius: float
    device: str
    lr_center: float
    lr_basis: float
    lr_timescale: float
    branch_error_threshold: float
    support_error_threshold: float
    evidence_gain: float
    usefulness_gain: float
    dependency_gain: float
    support_gain: float
    support_decay: float
    support_admission_gain: float
    support_diversity_gain: float
    conflict_gain: float
    age_decay: float
    inward_rate: float
    outward_rate: float
    survival_temperature: float
    admission_threshold: float
    admission_temperature: float
    min_consolidation_potential: float
    low_potential_depth_cap: float
    low_potential_strength_cap: float
    provisional_depth_cap: float
    release_threshold: float
    overwrite_margin: float
    orthogonal_loss_weight: float
    shell_loss_weight: float
    max_events_per_step: int
    output_dir: Path


@dataclass
class SlotState:
    slot: int
    active: bool = False
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
class NeuralEvent:
    index: int
    step: int
    group: str
    vector: np.ndarray
    embedding: np.ndarray
    dependency: float
    usefulness: float
    contradiction_target: str | None
    kind: str


def finite_float(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    finite_float(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def positive_float(name: str, value: float) -> None:
    finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def validate_config(config: Config) -> None:
    if config.steps < 8:
        raise ValueError("steps must be at least 8.")
    if config.slots < 4:
        raise ValueError("slots must be at least 4.")
    if config.d_model < 3:
        raise ValueError("d_model must be at least 3.")
    if config.subspace_rank < 1 or config.subspace_rank >= config.d_model:
        raise ValueError("subspace_rank must be in [1, d_model).")
    if config.inner_updates < 1:
        raise ValueError("inner_updates must be at least 1.")
    if config.shell_count < 3:
        raise ValueError("shell_count must be at least 3.")
    positive_float("inner_radius", config.inner_radius)
    positive_float("outer_radius", config.outer_radius)
    if config.inner_radius >= config.outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius.")
    for name in (
        "lr_center",
        "lr_basis",
        "lr_timescale",
        "branch_error_threshold",
        "support_error_threshold",
        "evidence_gain",
        "usefulness_gain",
        "dependency_gain",
        "support_gain",
        "support_decay",
        "support_admission_gain",
        "support_diversity_gain",
        "conflict_gain",
        "age_decay",
        "inward_rate",
        "outward_rate",
        "survival_temperature",
        "admission_threshold",
        "admission_temperature",
        "min_consolidation_potential",
        "low_potential_depth_cap",
        "low_potential_strength_cap",
        "provisional_depth_cap",
        "overwrite_margin",
        "orthogonal_loss_weight",
        "shell_loss_weight",
    ):
        nonnegative_float(name, float(getattr(config, name)))
    if config.support_decay > 1.0:
        raise ValueError("support_decay must be in [0, 1].")
    if config.provisional_depth_cap > float(config.shell_count - 1):
        raise ValueError("provisional_depth_cap cannot exceed maximum depth.")
    if config.low_potential_depth_cap > config.provisional_depth_cap:
        raise ValueError("low_potential_depth_cap cannot exceed provisional_depth_cap.")
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


def manual_config(config: Config) -> ManualConfig:
    return ManualConfig(
        seed=config.seed,
        steps=config.steps,
        slots=config.slots,
        shell_count=config.shell_count,
        inner_radius=config.inner_radius,
        outer_radius=config.outer_radius,
        match_threshold=0.10,
        branch_threshold=0.22,
        initial_strength=0.55,
        evidence_gain=config.evidence_gain,
        usefulness_gain=config.usefulness_gain,
        dependency_gain=config.dependency_gain,
        conflict_gain=config.conflict_gain,
        age_decay=config.age_decay,
        outer_learning_rate=0.42,
        learning_timescale=config.lr_timescale,
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
        support_threshold=0.16,
        support_gain=config.support_gain,
        support_min_potential=1.00,
        support_decay=config.support_decay,
        support_admission_gain=config.support_admission_gain,
        support_diversity_gain=config.support_diversity_gain,
        max_events_per_step=config.max_events_per_step,
        output_dir=config.output_dir,
    )


def scenario_group_specs(config: Config) -> list[GroupSpec]:
    if config.scenario == "default":
        return default_group_specs(config.steps)
    if config.scenario != "long_consolidation":
        raise ValueError(f"Unknown scenario: {config.scenario!r}.")
    steps = config.steps
    if steps < 96:
        raise ValueError("long_consolidation requires at least 96 steps.")
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


def shell_radius(depth: float, config: Config) -> float:
    max_depth = float(config.shell_count - 1)
    alpha = min(max(depth / max_depth, 0.0), 1.0)
    return config.outer_radius * (1.0 - alpha) + config.inner_radius * alpha


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def np_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Cannot normalize a zero or non-finite vector.")
    return vector / norm


def embedding_matrix(config: Config) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 17)
    matrix = rng.normal(size=(3, config.d_model)).astype(np.float32)
    matrix[:3, :3] += np.eye(3, dtype=np.float32) * 2.0
    q, _ = np.linalg.qr(matrix.T)
    basis = q[:, :3].T.astype(np.float32)
    if basis.shape != (3, config.d_model):
        raise RuntimeError(f"Unexpected embedding basis shape: {basis.shape}")
    return basis


def embed_vector(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    embedded = vector.astype(np.float32) @ matrix
    return np_normalize(embedded).astype(np.float32)


def build_neural_events(config: Config) -> tuple[list[NeuralEvent], dict[str, np.ndarray], dict[str, np.ndarray]]:
    manual = manual_config(config)
    specs = scenario_group_specs(config)
    prototypes_3d = {spec.name: unit(np.array(spec.center, dtype=np.float64)) for spec in specs}
    matrix = embedding_matrix(config)
    prototypes = {name: embed_vector(vector, matrix) for name, vector in prototypes_3d.items()}
    events = []
    for index, event in enumerate(build_events(manual, specs)):
        events.append(
            NeuralEvent(
                index=index,
                step=event.step,
                group=event.group,
                vector=event.vector.astype(np.float32),
                embedding=embed_vector(event.vector, matrix),
                dependency=event.dependency,
                usefulness=event.usefulness,
                contradiction_target=event.contradiction_target,
                kind=event.kind,
            )
        )
    if not events:
        raise RuntimeError("No neural events were generated.")
    return events, prototypes, prototypes_3d


def slot_survival(slot: SlotState, config: Config) -> float:
    if not slot.active:
        return -math.inf
    evidence_term = math.log1p(max(0.0, slot.evidence))
    support_term = math.log1p(max(0.0, slot.downstream_support))
    support_diversity = math.log1p(float(len(slot.support_mass)))
    return float(
        config.evidence_gain * evidence_term
        + config.usefulness_gain * slot.usefulness
        + config.dependency_gain * slot.dependency
        + 0.55 * config.support_admission_gain * support_term
        + 0.40 * config.support_diversity_gain * support_diversity
        - config.conflict_gain * slot.conflict
        - config.age_decay * float(slot.age)
    )


def admission_score(slot: SlotState, config: Config) -> float:
    if not slot.active:
        return -math.inf
    evidence_term = math.log1p(max(0.0, slot.evidence))
    support_term = math.log1p(max(0.0, slot.downstream_support))
    support_diversity = math.log1p(float(len(slot.support_mass)))
    return float(
        0.80 * evidence_term
        + 1.10 * slot.usefulness
        + 1.25 * slot.dependency
        + config.support_admission_gain * support_term
        + config.support_diversity_gain * support_diversity
        - 1.40 * slot.conflict
    )


def consolidation_potential(slot: SlotState) -> float:
    support_term = math.log1p(max(0.0, slot.downstream_support))
    support_diversity = math.log1p(float(len(slot.support_mass)))
    return float(slot.usefulness + slot.dependency + 0.25 * support_term + 0.15 * support_diversity)


def update_admission(slot: SlotState, config: Config) -> None:
    if not slot.active:
        slot.admitted = False
        return
    if consolidation_potential(slot) < config.min_consolidation_potential:
        slot.admitted = False
        return
    probability = sigmoid((admission_score(slot, config) - config.admission_threshold) / max(1e-6, config.admission_temperature))
    if probability >= 0.5:
        slot.admitted = True
    if slot.conflict > 1.15:
        slot.admitted = False


def adjust_depth(slot: SlotState, config: Config, multiplier: float) -> None:
    if not slot.active:
        return
    max_depth = float(config.shell_count - 1)
    survival = slot_survival(slot, config)
    desired = max_depth * sigmoid((survival - 1.35) / max(1e-6, config.survival_temperature))
    if consolidation_potential(slot) < config.min_consolidation_potential:
        desired = min(desired, config.low_potential_depth_cap)
    if not slot.admitted:
        desired = min(desired, config.provisional_depth_cap)
    if slot.conflict > 0.65:
        desired *= 1.0 - sigmoid((slot.conflict - 0.65) / max(1e-6, config.survival_temperature))
    rate = config.inward_rate if desired >= slot.depth else config.outward_rate
    slot.depth += multiplier * rate * (desired - slot.depth)
    slot.depth = min(max(slot.depth, 0.0), max_depth)


class NeuralNestedSlots:
    def __init__(self, config: Config, device: torch.device) -> None:
        self.config = config
        self.device = device
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 101)
        self.centers = torch.randn(config.slots, config.d_model, generator=generator, dtype=torch.float32) * 0.05
        self.bases = torch.randn(config.slots, config.d_model, config.subspace_rank, generator=generator, dtype=torch.float32) * 0.02
        self.centers = self.centers.to(device)
        self.bases = self.bases.to(device)
        self.slots = [SlotState(slot=index) for index in range(config.slots)]

    def active_indices(self) -> list[int]:
        return [slot.slot for slot in self.slots if slot.active]

    def projection_for_slot(self, slot_index: int, h: torch.Tensor) -> torch.Tensor:
        center = self.centers[slot_index]
        basis = self.bases[slot_index]
        delta = h - center
        coeff = torch.matmul(delta, basis)
        return center + torch.matmul(basis, coeff)

    def reconstruction_error(self, slot_index: int, h: torch.Tensor) -> torch.Tensor:
        projected = self.projection_for_slot(slot_index, h)
        return torch.mean((projected - h) ** 2)

    def reconstruction_errors(self, h: torch.Tensor) -> dict[int, float]:
        errors: dict[int, float] = {}
        with torch.no_grad():
            for slot_index in self.active_indices():
                error = self.reconstruction_error(slot_index, h)
                value = float(error.detach().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite reconstruction error for slot {slot_index}: {value}")
                errors[slot_index] = value
        return errors

    def shell_loss(self, slot_index: int) -> torch.Tensor:
        target_radius = shell_radius(self.slots[slot_index].depth, self.config)
        center_norm = torch.linalg.vector_norm(self.centers[slot_index])
        return (center_norm - target_radius) ** 2

    def orthogonal_loss(self, slot_index: int) -> torch.Tensor:
        basis = self.bases[slot_index]
        gram = torch.matmul(basis.T, basis)
        identity = torch.eye(self.config.subspace_rank, device=self.device, dtype=torch.float32)
        return torch.mean((gram - identity) ** 2)

    def train_slot(self, slot_index: int, h: torch.Tensor) -> float:
        slot = self.slots[slot_index]
        scale = math.exp(-self.config.lr_timescale * slot.depth)
        last_loss = math.nan
        for _ in range(self.config.inner_updates):
            centers = self.centers.detach().clone().requires_grad_(True)
            bases = self.bases.detach().clone().requires_grad_(True)
            center = centers[slot_index]
            basis = bases[slot_index]
            delta = h - center
            coeff = torch.matmul(delta, basis)
            recon = center + torch.matmul(basis, coeff)
            recon_loss = torch.mean((recon - h) ** 2)
            gram = torch.matmul(basis.T, basis)
            identity = torch.eye(self.config.subspace_rank, device=self.device, dtype=torch.float32)
            ortho_loss = torch.mean((gram - identity) ** 2)
            radius = torch.linalg.vector_norm(center)
            target_radius = torch.tensor(shell_radius(slot.depth, self.config), device=self.device, dtype=torch.float32)
            shell_loss = (radius - target_radius) ** 2
            loss = recon_loss + self.config.orthogonal_loss_weight * ortho_loss + self.config.shell_loss_weight * shell_loss
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite slot training loss for slot {slot_index}.")
            loss.backward()
            with torch.no_grad():
                center_grad = centers.grad[slot_index]
                basis_grad = bases.grad[slot_index]
                if center_grad is None or basis_grad is None:
                    raise RuntimeError("Missing gradients for neural nested slot update.")
                self.centers[slot_index] -= self.config.lr_center * scale * center_grad
                self.bases[slot_index] -= self.config.lr_basis * scale * basis_grad
            last_loss = float(loss.detach().cpu())
        return last_loss

    def initialize_slot(self, slot_index: int, event: NeuralEvent) -> None:
        h = torch.tensor(event.embedding, device=self.device, dtype=torch.float32)
        radius = shell_radius(0.0, self.config)
        with torch.no_grad():
            self.centers[slot_index] = h / torch.linalg.vector_norm(h) * radius
            noise = torch.randn_like(self.bases[slot_index]) * 0.025
            first = h / torch.linalg.vector_norm(h)
            basis = noise
            basis[:, 0] = first
            self.bases[slot_index] = basis
        slot = self.slots[slot_index]
        slot.active = True
        slot.depth = 0.0
        slot.strength = 0.55
        slot.evidence = 1.0
        slot.usefulness = event.usefulness
        slot.dependency = event.dependency
        slot.conflict = 0.0
        slot.downstream_support = 0.0
        slot.admitted = False
        slot.age = 0
        slot.updates = 1
        slot.dominant_group = event.group
        slot.group_mass = {event.group: 1.0}
        slot.support_mass = {}
        update_admission(slot, self.config)
        if consolidation_potential(slot) < self.config.min_consolidation_potential:
            slot.strength = min(slot.strength, self.config.low_potential_strength_cap)
        adjust_depth(slot, self.config, multiplier=1.0)

    def choose_slot(self, event: NeuralEvent, h: torch.Tensor) -> tuple[int, str, float, float]:
        active = self.active_indices()
        if not active:
            return 0, "new_free_slot", math.inf, -math.inf
        errors = self.reconstruction_errors(h)
        ranked = sorted(
            ((errors[index], -slot_survival(self.slots[index], self.config), index) for index in active),
            key=lambda row: (row[0], row[1], row[2]),
        )
        best_error, negative_survival, best_index = ranked[0]
        best_survival = -negative_survival
        free = next((slot.slot for slot in self.slots if not slot.active), None)
        low_potential = event.usefulness + event.dependency < self.config.min_consolidation_potential
        if low_potential and self.slots[best_index].admitted:
            if free is not None:
                return free, "new_provisional_slot", best_error, best_survival
            provisional = [index for index in active if not self.slots[index].admitted]
            if provisional:
                chosen = min(provisional, key=lambda index: (slot_survival(self.slots[index], self.config), self.slots[index].strength, -self.slots[index].age, index))
                return chosen, "provisional_low_potential", best_error, slot_survival(self.slots[chosen], self.config)
            return -1, "reject_low_potential", best_error, best_survival
        if event.contradiction_target is not None and self.slots[best_index].dominant_group == event.contradiction_target:
            if free is not None:
                return free, "branch_for_contradiction", best_error, best_survival
        if event.contradiction_target is not None:
            target_indices = [index for index in active if event.contradiction_target in self.slots[index].group_mass]
            if target_indices and free is not None:
                return free, "branch_for_contradiction", best_error, best_survival
            non_target = [index for index in active if event.contradiction_target not in self.slots[index].group_mass]
            if non_target:
                non_target_ranked = sorted(
                    ((errors[index], -slot_survival(self.slots[index], self.config), index) for index in non_target),
                    key=lambda row: (row[0], row[1], row[2]),
                )
                candidate_error, candidate_negative_survival, candidate_index = non_target_ranked[0]
                if candidate_error <= self.config.branch_error_threshold:
                    return candidate_index, "matched_existing_nonconflict", candidate_error, -candidate_negative_survival
        if best_error <= self.config.branch_error_threshold:
            return best_index, "matched_existing", best_error, best_survival
        if free is not None:
            return free, "new_branch_slot", best_error, best_survival
        weakest_index = min(active, key=lambda index: (self.slots[index].admitted, slot_survival(self.slots[index], self.config), self.slots[index].strength, -self.slots[index].age, index))
        weakest_survival = slot_survival(self.slots[weakest_index], self.config)
        potential = event.usefulness + event.dependency
        if potential + self.config.overwrite_margin > weakest_survival:
            return weakest_index, "overwrite_low_survival", best_error, weakest_survival
        return best_index, "forced_match_capacity_full", best_error, best_survival

    def apply_support(self, event: NeuralEvent, h: torch.Tensor, touched_index: int) -> tuple[int, float]:
        potential = event.usefulness + event.dependency
        if potential < self.config.min_consolidation_potential:
            return 0, 0.0
        touched = 0
        total = 0.0
        errors = self.reconstruction_errors(h)
        for index, error in errors.items():
            if index == touched_index:
                continue
            slot = self.slots[index]
            if event.group in slot.group_mass:
                continue
            if error > self.config.support_error_threshold:
                continue
            closeness = 1.0 - error / max(1e-12, self.config.support_error_threshold)
            support = self.config.support_gain * closeness * potential
            slot.evidence += support
            slot.downstream_support = min(25.0, slot.downstream_support + support)
            slot.support_mass[event.group] = slot.support_mass.get(event.group, 0.0) + support
            slot.usefulness = max(slot.usefulness, 0.94 * slot.usefulness + 0.06 * event.usefulness)
            slot.dependency = max(slot.dependency, 0.94 * slot.dependency + 0.06 * event.dependency)
            slot.strength = min(10.0, slot.strength + 0.12 * support)
            slot.age = max(0, slot.age - 1)
            update_admission(slot, self.config)
            adjust_depth(slot, self.config, multiplier=0.25)
            touched += 1
            total += support
        return touched, total

    def apply_contradiction(self, event: NeuralEvent) -> tuple[int, float]:
        if event.contradiction_target is None:
            return 0, 0.0
        touched = 0
        total = 0.0
        for slot in self.slots:
            if not slot.active:
                continue
            target_mass = slot.group_mass.get(event.contradiction_target, 0.0)
            if target_mass <= 0.0:
                continue
            mass = max(1e-12, sum(slot.group_mass.values()))
            share = target_mass / mass
            pressure = share * share * (1.0 + event.usefulness + event.dependency)
            slot.conflict = 0.82 * slot.conflict + pressure
            reduced_target_mass = target_mass * max(0.0, 1.0 - 0.45 * share)
            if reduced_target_mass < 0.15:
                del slot.group_mass[event.contradiction_target]
            else:
                slot.group_mass[event.contradiction_target] = reduced_target_mass
            if slot.group_mass:
                slot.dominant_group = max(slot.group_mass.items(), key=lambda row: (row[1], row[0]))[0]
            else:
                slot.dominant_group = "empty"
            slot.usefulness *= 1.0 - 0.10 * share
            slot.strength *= 1.0 - 0.08 * share
            update_admission(slot, self.config)
            touched += 1
            total += pressure
        return touched, total

    def update_slot_state(self, slot_index: int, event: NeuralEvent, action: str, train_loss: float) -> dict[str, float | str | int]:
        slot = self.slots[slot_index]
        previous_group = slot.dominant_group
        previous_depth = slot.depth
        previous_survival = slot_survival(slot, self.config) if slot.active else -math.inf
        if action == "overwrite_low_survival":
            slot.released_count += 1
            self.initialize_slot(slot_index, event)
            slot = self.slots[slot_index]
        elif not slot.active or action in {"new_free_slot", "new_branch_slot", "branch_for_contradiction"}:
            self.initialize_slot(slot_index, event)
            slot = self.slots[slot_index]
        else:
            slot.evidence = 0.97 * slot.evidence + 1.0
            slot.usefulness = 0.92 * slot.usefulness + 0.08 * event.usefulness
            slot.dependency = max(0.96 * slot.dependency, event.dependency)
            slot.conflict *= 0.90
            slot.strength = min(10.0, 0.985 * slot.strength + 0.20 + 0.12 * event.usefulness + 0.10 * event.dependency)
            slot.age = 0
            slot.updates += 1
            slot.group_mass[event.group] = slot.group_mass.get(event.group, 0.0) + 1.0
            slot.dominant_group = max(slot.group_mass.items(), key=lambda row: (row[1], row[0]))[0]
            update_admission(slot, self.config)
            if consolidation_potential(slot) < self.config.min_consolidation_potential:
                slot.strength = min(slot.strength, self.config.low_potential_strength_cap)
            adjust_depth(slot, self.config, multiplier=1.0)
        survival = slot_survival(slot, self.config)
        return {
            "slot": slot_index,
            "action": action,
            "previous_group": previous_group,
            "dominant_group": slot.dominant_group,
            "previous_survival": previous_survival,
            "survival": survival,
            "previous_depth": previous_depth,
            "depth": slot.depth,
            "depth_delta": slot.depth - previous_depth,
            "strength": slot.strength,
            "evidence": slot.evidence,
            "usefulness": slot.usefulness,
            "dependency": slot.dependency,
            "conflict": slot.conflict,
            "downstream_support": slot.downstream_support,
            "support_diversity": len(slot.support_mass),
            "admitted": int(slot.admitted),
            "dominant_share": slot.dominant_share(),
            "train_loss": train_loss,
        }

    def decay_idle_slots(self, touched_index: int) -> int:
        releases = 0
        for slot in self.slots:
            if not slot.active or slot.slot == touched_index:
                continue
            slot.age += 1
            depth_decay = 0.045 * math.exp(-0.72 * slot.depth)
            if not slot.admitted:
                depth_decay *= 2.50
            slot.strength *= max(0.0, 1.0 - depth_decay)
            slot.evidence *= max(0.0, 1.0 - 0.35 * depth_decay)
            slot.conflict *= 0.985
            slot.downstream_support *= self.config.support_decay
            if slot.support_mass:
                slot.support_mass = {
                    group: mass * self.config.support_decay
                    for group, mass in slot.support_mass.items()
                    if mass * self.config.support_decay > 1e-6
                }
            update_admission(slot, self.config)
            adjust_depth(slot, self.config, multiplier=0.18)
            survival = slot_survival(slot, self.config)
            should_release = (not slot.admitted) and survival < self.config.release_threshold and slot.strength < 0.45
            should_release = should_release or (slot.conflict > 1.15 and survival < self.config.release_threshold + 1.25)
            if should_release:
                self.release_slot(slot.slot)
                releases += 1
        return releases

    def release_slot(self, slot_index: int) -> None:
        slot = self.slots[slot_index]
        slot.active = False
        slot.depth = 0.0
        slot.strength = 0.0
        slot.evidence = 0.0
        slot.usefulness = 0.0
        slot.dependency = 0.0
        slot.conflict = 0.0
        slot.downstream_support = 0.0
        slot.admitted = False
        slot.age = 0
        slot.updates = 0
        slot.dominant_group = "empty"
        slot.group_mass = {}
        slot.support_mass = {}
        slot.released_count += 1


def run_sequence(config: Config) -> tuple[NeuralNestedSlots, list[dict[str, float | str | int]], list[dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    validate_config(config)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    events, prototypes, prototypes_3d = build_neural_events(config)
    model = NeuralNestedSlots(config, device)
    event_rows: list[dict[str, float | str | int]] = []
    metric_rows: list[dict[str, float]] = []

    for event in events:
        h = torch.tensor(event.embedding, device=device, dtype=torch.float32)
        contradiction_slots, contradiction_pressure = model.apply_contradiction(event)
        slot_index, action, best_error, best_survival = model.choose_slot(event, h)
        if slot_index < 0:
            release_count = model.decay_idle_slots(slot_index)
            metrics = compute_metrics(model, prototypes, config)
            metrics.update({"event_index": float(event.index), "step": float(event.step), "release_count": float(release_count)})
            metric_rows.append(metrics)
            event_rows.append(
                {
                    "event_index": event.index,
                    "step": event.step,
                    "group": event.group,
                    "kind": event.kind,
                    "contradiction_target": event.contradiction_target or "",
                    "best_error": best_error,
                    "best_survival": best_survival,
                    "slot": -1,
                    "action": action,
                    "previous_group": "none",
                    "dominant_group": "none",
                    "previous_survival": -math.inf,
                    "survival": -math.inf,
                    "previous_depth": 0.0,
                    "depth": 0.0,
                    "depth_delta": 0.0,
                    "strength": 0.0,
                    "evidence": 0.0,
                    "usefulness": event.usefulness,
                    "dependency": event.dependency,
                    "conflict": 0.0,
                    "downstream_support": 0.0,
                    "support_diversity": 0,
                    "admitted": 0,
                    "dominant_share": 0.0,
                    "train_loss": math.nan,
                    "contradiction_slots": contradiction_slots,
                    "contradiction_pressure": contradiction_pressure,
                    "support_slots": 0,
                    "support_amount": 0.0,
                    "released_slots_after_event": release_count,
                }
            )
            continue
        if action == "overwrite_low_survival" or not model.slots[slot_index].active:
            model.initialize_slot(slot_index, event)
        train_loss = model.train_slot(slot_index, h)
        update = model.update_slot_state(slot_index, event, action, train_loss)
        support_slots, support_amount = model.apply_support(event, h, slot_index)
        release_count = model.decay_idle_slots(slot_index)
        metrics = compute_metrics(model, prototypes, config)
        metrics.update({"event_index": float(event.index), "step": float(event.step), "release_count": float(release_count)})
        metric_rows.append(metrics)
        event_rows.append(
            {
                "event_index": event.index,
                "step": event.step,
                "group": event.group,
                "kind": event.kind,
                "contradiction_target": event.contradiction_target or "",
                "best_error": best_error,
                "best_survival": best_survival,
                **update,
                "contradiction_slots": contradiction_slots,
                "contradiction_pressure": contradiction_pressure,
                "support_slots": support_slots,
                "support_amount": support_amount,
                "released_slots_after_event": release_count,
            }
        )
    return model, event_rows, metric_rows, prototypes, prototypes_3d


def compute_slot_error(model: NeuralNestedSlots, slot_index: int, vector: np.ndarray) -> float:
    h = torch.tensor(vector, device=model.device, dtype=torch.float32)
    with torch.no_grad():
        value = float(model.reconstruction_error(slot_index, h).detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite slot error for slot {slot_index}.")
    return value


def compute_metrics(model: NeuralNestedSlots, prototypes: dict[str, np.ndarray], config: Config) -> dict[str, float]:
    active = [slot for slot in model.slots if slot.active]
    if not active:
        return {
            "active_slots": 0.0,
            "free_slots": float(config.slots),
            "mean_depth": 0.0,
            "outer_mass": 0.0,
            "middle_mass": 0.0,
            "inner_mass": 0.0,
            "mean_strength": 0.0,
            "mean_trace_error": math.nan,
        }
    depths = np.array([slot.depth for slot in active], dtype=np.float64)
    strengths = np.array([slot.strength for slot in active], dtype=np.float64)
    errors = []
    for slot in active:
        if slot.dominant_group in prototypes:
            errors.append(compute_slot_error(model, slot.slot, prototypes[slot.dominant_group]))
    max_depth = float(config.shell_count - 1)
    return {
        "active_slots": float(len(active)),
        "free_slots": float(config.slots - len(active)),
        "mean_depth": float(np.mean(depths)),
        "outer_mass": float(np.mean(depths <= 0.32 * max_depth)),
        "middle_mass": float(np.mean((depths > 0.32 * max_depth) & (depths <= 0.72 * max_depth))),
        "inner_mass": float(np.mean(depths > 0.72 * max_depth)),
        "mean_strength": float(np.mean(strengths)),
        "mean_trace_error": float(np.mean(errors)) if errors else math.nan,
    }


def final_group_summary(model: NeuralNestedSlots, event_rows: list[dict[str, float | str | int]], prototypes: dict[str, np.ndarray], config: Config) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for group in sorted(prototypes):
        candidates = [slot for slot in model.slots if slot.active and group in slot.group_mass]
        if candidates:
            best = min(candidates, key=lambda slot: compute_slot_error(model, slot.slot, prototypes[group]))
            trace_error = compute_slot_error(model, best.slot, prototypes[group])
            assigned_slot = best.slot
            depth = best.depth
            strength = best.strength
            survival = slot_survival(best, config)
            share = best.group_mass[group] / max(1e-12, sum(best.group_mass.values()))
        else:
            assigned_slot = -1
            trace_error = math.nan
            depth = 0.0
            strength = 0.0
            survival = -math.inf
            share = 0.0
        rows.append(
            {
                "group": group,
                "events": sum(1 for row in event_rows if row["group"] == group),
                "slot": assigned_slot,
                "depth": depth,
                "shell_radius": shell_radius(depth, config),
                "strength": strength,
                "survival": survival,
                "trace_error": trace_error,
                "group_share_in_slot": share,
            }
        )
    return rows


def slot_table(model: NeuralNestedSlots, config: Config) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    centers = model.centers.detach().cpu().numpy()
    bases = model.bases.detach().cpu().numpy()
    for slot in model.slots:
        basis_norm = float(np.linalg.norm(bases[slot.slot]))
        rows.append(
            {
                "slot": slot.slot,
                "active": int(slot.active),
                "dominant_group": slot.dominant_group,
                "dominant_share": slot.dominant_share(),
                "depth": slot.depth,
                "shell_radius": shell_radius(slot.depth, config),
                "strength": slot.strength,
                "evidence": slot.evidence,
                "usefulness": slot.usefulness,
                "dependency": slot.dependency,
                "conflict": slot.conflict,
                "downstream_support": slot.downstream_support,
                "support_diversity": len(slot.support_mass),
                "admitted": int(slot.admitted),
                "age": slot.age,
                "updates": slot.updates,
                "released_count": slot.released_count,
                "center_norm": float(np.linalg.norm(centers[slot.slot])),
                "basis_norm": basis_norm,
            }
        )
    return rows


def summarize_validation(group_rows: list[dict[str, float | str | int]], metric_rows: list[dict[str, float]]) -> dict[str, bool]:
    by_group = {str(row["group"]): row for row in group_rows}

    def group_error(name: str) -> float:
        value = float(by_group[name]["trace_error"])
        return value if math.isfinite(value) else math.inf

    def survived(name: str, max_error: float = 0.018, min_depth: float = 0.0) -> bool:
        slot = int(by_group[name]["slot"])
        if slot < 0:
            return False
        return group_error(name) < max_error and float(by_group[name]["depth"]) >= min_depth

    merge_slots = {int(by_group["merge_a"]["slot"]), int(by_group["merge_b"]["slot"])}
    branch_slots = {int(by_group["branch_root"]["slot"]), int(by_group["branch_up"]["slot"]), int(by_group["branch_down"]["slot"])}
    branch_slots = {slot for slot in branch_slots if slot >= 0}
    stable_depth = float(by_group["stable"]["depth"])
    rare_depth = float(by_group["rare_critical"]["depth"])
    replacement_error = group_error("replacement")
    obsolete_error = group_error("obsolete_old")
    noise_slot = int(by_group["noise"]["slot"])
    noise_share = float(by_group["noise"]["group_share_in_slot"])
    noise_strength = float(by_group["noise"]["strength"])
    max_middle = max(row["middle_mass"] for row in metric_rows)
    final_middle = metric_rows[-1]["middle_mass"]
    final_outer = metric_rows[-1]["outer_mass"]
    return {
        "stable_consolidated_inward": stable_depth > 1.25,
        "rare_critical_survived": survived("rare_critical", min_depth=0.65) and rare_depth > 0.65,
        "replacement_beats_obsolete": replacement_error < obsolete_error,
        "noise_remains_weak": noise_slot < 0 or noise_share < 0.5 or noise_strength < 2.5,
        "duplicate_sources_merge": len(merge_slots) == 1,
        "branch_root_final_survives": survived("branch_root"),
        "branch_up_final_survives": survived("branch_up"),
        "branch_down_final_survives": survived("branch_down"),
        "important_branches_survive_under_capacity": all(survived(group) for group in ("branch_root", "branch_up", "branch_down")),
        "branches_can_separate": len(branch_slots) > 1,
        "middle_layer_used": max_middle > 0.0 and final_middle > 0.0,
        "outer_geometry_still_available": final_outer > 0.0,
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


def plot_slot_state(model: NeuralNestedSlots, prototypes_3d: dict[str, np.ndarray], group_rows: list[dict[str, float | str | int]], config: Config, path: Path) -> None:
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * math.pi, 34)
    v = np.linspace(0, math.pi, 18)
    for shell_index in range(config.shell_count):
        radius = shell_radius(float(shell_index), config)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, alpha=0.12, linewidth=0.45)

    for group, proto in prototypes_3d.items():
        point = config.outer_radius * proto
        ax.scatter([point[0]], [point[1]], [point[2]], marker=".", s=16, color=GROUP_COLORS.get(group, "#333333"), alpha=0.35)

    rows_by_slot = {int(row["slot"]): str(row["group"]) for row in group_rows if int(row["slot"]) >= 0}
    centers = model.centers.detach().cpu().numpy()
    for slot in model.slots:
        if not slot.active:
            continue
        center = centers[slot.slot]
        norm = max(1e-12, float(np.linalg.norm(center)))
        display = center[:3] / norm * shell_radius(slot.depth, config)
        group = rows_by_slot.get(slot.slot, slot.dominant_group)
        ax.scatter([display[0]], [display[1]], [display[2]], color=GROUP_COLORS.get(group, "#333333"), s=42 + 28 * slot.strength, edgecolor="black", linewidth=0.5)
        ax.text(display[0], display[1], display[2], f"{slot.slot}:{slot.dominant_group}", fontsize=7)
    limit = config.outer_radius * 1.18
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Neural nested geometry: learned slot subspaces projected into 3D")
    ax.set_xlabel("PC-like x")
    ax.set_ylabel("PC-like y")
    ax.set_zlabel("PC-like z")
    ax.view_init(elev=22, azim=42)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lifecycle(group_rows: list[dict[str, float | str | int]], path: Path) -> None:
    groups = [str(row["group"]) for row in group_rows]
    depth = np.array([float(row["depth"]) for row in group_rows])
    strength = np.array([float(row["strength"]) for row in group_rows])
    error = np.array([float(row["trace_error"]) if math.isfinite(float(row["trace_error"])) else np.nan for row in group_rows])
    survival = np.array([float(row["survival"]) if math.isfinite(float(row["survival"])) else np.nan for row in group_rows])
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    xs = np.arange(len(groups))
    colors = [GROUP_COLORS.get(group, "#333333") for group in groups]
    axes[0].bar(xs, depth, color=colors)
    axes[0].set_ylabel("final depth")
    axes[1].bar(xs, strength, color=colors)
    axes[1].set_ylabel("strength")
    axes[2].bar(xs, error, color=colors)
    axes[2].set_ylabel("subspace error")
    axes[3].bar(xs, survival, color=colors)
    axes[3].set_ylabel("survival utility")
    axes[3].set_xticks(xs)
    axes[3].set_xticklabels(groups, rotation=35, ha="right")
    axes[0].set_title("Neural nested slot lifecycle")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metrics(metric_rows: list[dict[str, float]], path: Path) -> None:
    xs = np.array([row["event_index"] for row in metric_rows])
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(xs, [row["active_slots"] for row in metric_rows], label="active")
    axes[0].plot(xs, [row["free_slots"] for row in metric_rows], label="free")
    axes[0].legend()
    axes[0].set_ylabel("slots")
    axes[1].plot(xs, [row["outer_mass"] for row in metric_rows], label="outer")
    axes[1].plot(xs, [row["middle_mass"] for row in metric_rows], label="middle")
    axes[1].plot(xs, [row["inner_mass"] for row in metric_rows], label="inner")
    axes[1].legend()
    axes[1].set_ylabel("layer usage")
    axes[2].plot(xs, [row["mean_depth"] for row in metric_rows], color="#984ea3")
    axes[2].set_ylabel("mean depth")
    axes[3].plot(xs, [row["mean_trace_error"] for row in metric_rows], color="#e41a1c")
    axes[3].set_ylabel("mean subspace error")
    axes[3].set_xlabel("event index")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_slot_heatmap(model: NeuralNestedSlots, path: Path) -> None:
    groups = sorted({group for slot in model.slots for group in slot.group_mass})
    if not groups:
        raise RuntimeError("Cannot plot slot heatmap without group assignments.")
    matrix = np.zeros((len(model.slots), len(groups)), dtype=np.float64)
    for row, slot in enumerate(model.slots):
        total = sum(slot.group_mass.values())
        if total <= 0.0:
            continue
        for col, group in enumerate(groups):
            matrix[row, col] = slot.group_mass.get(group, 0.0) / total
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(model.slots)))
    ax.set_yticklabels([f"slot {slot.slot}" for slot in model.slots])
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_title("Neural slot composition")
    fig.colorbar(im, ax=ax, label="group share in slot")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(
    model: NeuralNestedSlots,
    event_rows: list[dict[str, float | str | int]],
    metric_rows: list[dict[str, float]],
    prototypes: dict[str, np.ndarray],
    prototypes_3d: dict[str, np.ndarray],
    config: Config,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    group_rows = final_group_summary(model, event_rows, prototypes, config)
    slot_rows = slot_table(model, config)
    validation = summarize_validation(group_rows, metric_rows)
    artifacts = {
        "event_log": config.output_dir / "event_log.csv",
        "metrics": config.output_dir / "metrics_over_time.csv",
        "slots": config.output_dir / "slot_table.csv",
        "groups": config.output_dir / "group_summary.csv",
        "summary": config.output_dir / "neural_nested_geometry_summary.json",
        "state_plot": config.output_dir / "neural_nested_geometry_final_state.png",
        "lifecycle_plot": config.output_dir / "neural_group_lifecycle_summary.png",
        "metrics_plot": config.output_dir / "neural_nested_metrics_over_time.png",
        "heatmap": config.output_dir / "neural_slot_group_heatmap.png",
    }
    write_csv(artifacts["event_log"], event_rows)
    write_csv(artifacts["metrics"], metric_rows)
    write_csv(artifacts["slots"], slot_rows)
    write_csv(artifacts["groups"], group_rows)
    plot_slot_state(model, prototypes_3d, group_rows, config, artifacts["state_plot"])
    plot_lifecycle(group_rows, artifacts["lifecycle_plot"])
    plot_metrics(metric_rows, artifacts["metrics_plot"])
    plot_slot_heatmap(model, artifacts["heatmap"])
    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "final_metrics": metric_rows[-1],
        "groups": group_rows,
        "slots": slot_rows,
        "validation": validation,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    with artifacts["summary"].open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def print_report(summary: dict[str, object]) -> None:
    metrics = summary["final_metrics"]
    groups = summary["groups"]
    validation = summary["validation"]
    artifacts = summary["artifacts"]
    if not isinstance(metrics, dict) or not isinstance(groups, list) or not isinstance(validation, dict) or not isinstance(artifacts, dict):
        raise RuntimeError("Malformed neural nested geometry summary.")
    print("\nNEURAL NESTED-GEOMETRY SLOTS")
    print("=" * 132)
    print(
        f"active_slots={int(float(metrics['active_slots']))} free_slots={int(float(metrics['free_slots']))} "
        f"mean_depth={float(metrics['mean_depth']):.4f} outer={float(metrics['outer_mass']):.3f} "
        f"middle={float(metrics['middle_mass']):.3f} inner={float(metrics['inner_mass']):.3f} "
        f"mean_error={float(metrics['mean_trace_error']):.5f}"
    )
    print("-" * 132)
    print(f"{'group':>18} {'events':>7} {'slot':>5} {'depth':>8} {'radius':>8} {'strength':>10} {'survival':>10} {'error':>10} {'share':>8}")
    for row in groups:
        print(
            f"{row['group']:>18} {int(row['events']):7d} {int(row['slot']):5d} "
            f"{float(row['depth']):8.4f} {float(row['shell_radius']):8.4f} "
            f"{float(row['strength']):10.4f} {float(row['survival']):10.4f} "
            f"{float(row['trace_error']):10.5f} {float(row['group_share_in_slot']):8.4f}"
        )
    print("\nCHECKS")
    print("-" * 132)
    for name, passed in validation.items():
        print(f"{name:>42} = {passed}")
    print("\nWROTE")
    print("-" * 132)
    for _, path in artifacts.items():
        print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=("default", "long_consolidation"), default="default")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=12)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--inner-updates", type=int, default=5)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--inner-radius", type=float, default=0.85)
    parser.add_argument("--outer-radius", type=float, default=3.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--lr-center", type=float, default=0.22)
    parser.add_argument("--lr-basis", type=float, default=0.16)
    parser.add_argument("--lr-timescale", type=float, default=0.62)
    parser.add_argument("--branch-error-threshold", type=float, default=0.012)
    parser.add_argument("--support-error-threshold", type=float, default=0.010)
    parser.add_argument("--evidence-gain", type=float, default=0.75)
    parser.add_argument("--usefulness-gain", type=float, default=1.20)
    parser.add_argument("--dependency-gain", type=float, default=1.25)
    parser.add_argument("--support-gain", type=float, default=0.45)
    parser.add_argument("--support-decay", type=float, default=0.985)
    parser.add_argument("--support-admission-gain", type=float, default=1.20)
    parser.add_argument("--support-diversity-gain", type=float, default=0.70)
    parser.add_argument("--conflict-gain", type=float, default=1.85)
    parser.add_argument("--age-decay", type=float, default=0.012)
    parser.add_argument("--inward-rate", type=float, default=0.025)
    parser.add_argument("--outward-rate", type=float, default=0.16)
    parser.add_argument("--survival-temperature", type=float, default=0.35)
    parser.add_argument("--admission-threshold", type=float, default=1.85)
    parser.add_argument("--admission-temperature", type=float, default=0.35)
    parser.add_argument("--min-consolidation-potential", type=float, default=0.85)
    parser.add_argument("--low-potential-depth-cap", type=float, default=0.35)
    parser.add_argument("--low-potential-strength-cap", type=float, default=1.25)
    parser.add_argument("--provisional-depth-cap", type=float, default=0.85)
    parser.add_argument("--release-threshold", type=float, default=-0.65)
    parser.add_argument("--overwrite-margin", type=float, default=0.18)
    parser.add_argument("--orthogonal-loss-weight", type=float, default=0.006)
    parser.add_argument("--shell-loss-weight", type=float, default=0.002)
    parser.add_argument("--max-events-per-step", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/05_nested_geometry/results/gco-neural-nested-geometry-slots-seed0"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = Config(
        seed=args.seed,
        scenario=args.scenario,
        steps=args.steps,
        slots=args.slots,
        d_model=args.d_model,
        subspace_rank=args.subspace_rank,
        inner_updates=args.inner_updates,
        shell_count=args.shell_count,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        device=args.device,
        lr_center=args.lr_center,
        lr_basis=args.lr_basis,
        lr_timescale=args.lr_timescale,
        branch_error_threshold=args.branch_error_threshold,
        support_error_threshold=args.support_error_threshold,
        evidence_gain=args.evidence_gain,
        usefulness_gain=args.usefulness_gain,
        dependency_gain=args.dependency_gain,
        support_gain=args.support_gain,
        support_decay=args.support_decay,
        support_admission_gain=args.support_admission_gain,
        support_diversity_gain=args.support_diversity_gain,
        conflict_gain=args.conflict_gain,
        age_decay=args.age_decay,
        inward_rate=args.inward_rate,
        outward_rate=args.outward_rate,
        survival_temperature=args.survival_temperature,
        admission_threshold=args.admission_threshold,
        admission_temperature=args.admission_temperature,
        min_consolidation_potential=args.min_consolidation_potential,
        low_potential_depth_cap=args.low_potential_depth_cap,
        low_potential_strength_cap=args.low_potential_strength_cap,
        provisional_depth_cap=args.provisional_depth_cap,
        release_threshold=args.release_threshold,
        overwrite_margin=args.overwrite_margin,
        orthogonal_loss_weight=args.orthogonal_loss_weight,
        shell_loss_weight=args.shell_loss_weight,
        max_events_per_step=args.max_events_per_step,
        output_dir=args.output_dir,
    )
    model, event_rows, metric_rows, prototypes, prototypes_3d = run_sequence(config)
    summary = write_outputs(model, event_rows, metric_rows, prototypes, prototypes_3d, config)
    print_report(summary)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
