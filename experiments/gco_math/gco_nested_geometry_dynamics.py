#!/usr/bin/env python3
"""Manual nested-geometry dynamics sandbox.

This experiment has no neural network and no optimizer. It tests whether a
nested geometry with bounded slots can express the basic continual-learning
memory dynamics we want before building a neural implementation:

* recent traces enter the outer geometry;
* repeated/useful/dependent traces consolidate inward;
* weak noise decays or is overwritten;
* contradicted obsolete traces are released;
* rare but functionally important traces can survive without high frequency;
* inner geometry moves more slowly than outer geometry.

The script writes visualizations, CSV logs, and a JSON summary. It is designed
to make every decision inspectable rather than hiding behavior in a loss curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


@dataclass(frozen=True)
class GroupSpec:
    name: str
    center: tuple[float, float, float]
    recurrence: int
    dependency: float
    usefulness: float
    start: int
    stop: int
    noise: float
    contradiction_target: str | None = None


@dataclass(frozen=True)
class Config:
    seed: int
    steps: int
    slots: int
    shell_count: int
    inner_radius: float
    outer_radius: float
    match_threshold: float
    branch_threshold: float
    initial_strength: float
    evidence_gain: float
    usefulness_gain: float
    dependency_gain: float
    conflict_gain: float
    age_decay: float
    outer_learning_rate: float
    learning_timescale: float
    outward_decay: float
    decay_timescale: float
    inward_rate: float
    outward_rate: float
    survival_temperature: float
    release_threshold: float
    overwrite_margin: float
    admission_threshold: float
    admission_temperature: float
    provisional_depth_cap: float
    provisional_decay_multiplier: float
    contradiction_release_threshold: float
    support_threshold: float
    support_gain: float
    support_min_potential: float
    support_decay: float
    support_admission_gain: float
    support_diversity_gain: float
    max_events_per_step: int
    output_dir: Path


@dataclass
class SlotState:
    slot: int
    active: bool = False
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    depth: float = 0.0
    strength: float = 0.0
    evidence: float = 0.0
    usefulness: float = 0.0
    dependency: float = 0.0
    conflict: float = 0.0
    contradiction_evidence: float = 0.0
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
class TraceEvent:
    step: int
    group: str
    vector: np.ndarray
    dependency: float
    usefulness: float
    contradiction_target: str | None
    kind: str


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
    if config.steps < 8:
        raise ValueError("steps must be at least 8.")
    if config.slots < 4:
        raise ValueError("slots must be at least 4.")
    if config.shell_count < 3:
        raise ValueError("shell_count must be at least 3.")
    positive_float("inner_radius", config.inner_radius)
    positive_float("outer_radius", config.outer_radius)
    if config.inner_radius >= config.outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius.")
    for name in (
        "match_threshold",
        "branch_threshold",
        "initial_strength",
        "evidence_gain",
        "usefulness_gain",
        "dependency_gain",
        "conflict_gain",
        "age_decay",
        "outer_learning_rate",
        "learning_timescale",
        "outward_decay",
        "decay_timescale",
        "inward_rate",
        "outward_rate",
        "survival_temperature",
        "overwrite_margin",
        "admission_threshold",
        "admission_temperature",
        "provisional_depth_cap",
        "provisional_decay_multiplier",
        "contradiction_release_threshold",
        "support_threshold",
        "support_gain",
        "support_min_potential",
        "support_decay",
        "support_admission_gain",
        "support_diversity_gain",
    ):
        nonnegative_float(name, float(getattr(config, name)))
    if config.match_threshold > 2.0:
        raise ValueError("match_threshold is a cosine distance and should be <= 2.")
    if config.branch_threshold > 2.0:
        raise ValueError("branch_threshold is a cosine distance and should be <= 2.")
    if config.support_threshold > 2.0:
        raise ValueError("support_threshold is a cosine distance and should be <= 2.")
    if config.support_decay > 1.0:
        raise ValueError("support_decay must be in [0, 1].")
    if config.release_threshold < -10.0 or config.release_threshold > 10.0:
        raise ValueError("release_threshold outside expected diagnostic range.")
    if config.provisional_depth_cap > float(config.shell_count - 1):
        raise ValueError("provisional_depth_cap cannot exceed the maximum shell depth.")
    if config.max_events_per_step < 1:
        raise ValueError("max_events_per_step must be at least 1.")


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Cannot normalize a zero or non-finite vector.")
    return vector / norm


def shell_radius(depth: float, config: Config) -> float:
    max_depth = float(config.shell_count - 1)
    clipped = min(max(depth, 0.0), max_depth)
    alpha = clipped / max_depth
    return config.outer_radius * (1.0 - alpha) + config.inner_radius * alpha


def project_to_shell(direction: np.ndarray, depth: float, config: Config) -> np.ndarray:
    return shell_radius(depth, config) * unit(direction)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    ua = unit(a)
    ub = unit(b)
    return float(1.0 - np.clip(np.dot(ua, ub), -1.0, 1.0))


def tangent_step(center: np.ndarray, target: np.ndarray, learning_rate: float, depth: float, config: Config) -> np.ndarray:
    normal = unit(center)
    raw_delta = target - center
    normal_part = float(np.dot(raw_delta, normal)) * normal
    tangent_part = raw_delta - normal_part
    moved = center + learning_rate * tangent_part
    return project_to_shell(moved, depth, config)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def slot_survival(slot: SlotState, config: Config) -> float:
    if not slot.active:
        return -math.inf
    evidence_term = math.log1p(max(0.0, slot.evidence))
    support_term = math.log1p(max(0.0, slot.downstream_support))
    support_diversity = math.log1p(float(len(slot.support_mass)))
    utility = (
        config.evidence_gain * evidence_term
        + config.usefulness_gain * slot.usefulness
        + config.dependency_gain * slot.dependency
        + 0.55 * config.support_admission_gain * support_term
        + 0.40 * config.support_diversity_gain * support_diversity
        - config.conflict_gain * slot.conflict
        - config.age_decay * float(slot.age)
    )
    return float(utility)


def admission_score(slot: SlotState, config: Config) -> float:
    if not slot.active:
        return -math.inf
    evidence_term = math.log1p(max(0.0, slot.evidence))
    support_term = math.log1p(max(0.0, slot.downstream_support))
    support_diversity = math.log1p(float(len(slot.support_mass)))
    score = (
        0.80 * evidence_term
        + 1.10 * slot.usefulness
        + 1.25 * slot.dependency
        + config.support_admission_gain * support_term
        + config.support_diversity_gain * support_diversity
        - 1.40 * slot.conflict
        - 1.15 * slot.contradiction_evidence
    )
    return float(score)


def update_admission(slot: SlotState, config: Config) -> None:
    if not slot.active:
        slot.admitted = False
        return
    score = admission_score(slot, config)
    probability = sigmoid((score - config.admission_threshold) / max(1e-6, config.admission_temperature))
    if probability >= 0.5:
        slot.admitted = True
    if slot.contradiction_evidence >= config.contradiction_release_threshold:
        slot.admitted = False


def slot_learning_rate(slot: SlotState, config: Config) -> float:
    return config.outer_learning_rate * math.exp(-config.learning_timescale * slot.depth)


def slot_decay(slot: SlotState, config: Config) -> float:
    return config.outward_decay * math.exp(-config.decay_timescale * slot.depth)


def default_group_specs(steps: int) -> list[GroupSpec]:
    if steps < 8:
        raise ValueError("steps must be at least 8 for the default scenario.")
    return [
        GroupSpec("stable", (0.00, 1.00, 0.24), 6, 0.85, 0.90, 0, steps, 0.035),
        GroupSpec("merge_a", (0.90, 0.10, 0.32), 4, 0.60, 0.62, 0, steps, 0.04),
        GroupSpec("merge_b", (0.86, 0.18, 0.28), 4, 0.60, 0.62, 0, steps, 0.04),
        GroupSpec("branch_root", (-0.62, 0.54, 0.42), 3, 0.50, 0.55, 0, steps // 2, 0.04),
        GroupSpec("branch_up", (-0.32, 0.56, 0.76), 3, 0.64, 0.70, steps // 3, steps, 0.04),
        GroupSpec("branch_down", (-0.88, 0.20, -0.42), 3, 0.64, 0.70, steps // 3, steps, 0.04),
        GroupSpec("rare_critical", (0.20, -0.83, 0.52), 1, 1.45, 1.25, 0, steps, 0.025),
        GroupSpec("obsolete_old", (-0.12, -0.92, -0.35), 4, 0.58, 0.50, 0, steps // 2, 0.04),
        GroupSpec(
            "replacement",
            (0.13, -0.84, -0.50),
            4,
            0.80,
            0.95,
            steps // 2,
            steps,
            0.035,
            contradiction_target="obsolete_old",
        ),
        GroupSpec("novel", (0.70, -0.30, 0.64), 2, 0.68, 0.76, steps // 2, steps, 0.04),
        GroupSpec("noise", (-0.35, -0.15, 0.92), 1, 0.05, 0.02, 0, steps, 0.55),
    ]


def build_events(config: Config, specs: list[GroupSpec]) -> list[TraceEvent]:
    rng = np.random.default_rng(config.seed)
    events: list[TraceEvent] = []
    for step in range(config.steps):
        step_events: list[TraceEvent] = []
        for spec in specs:
            if not (spec.start <= step < spec.stop):
                continue
            period = max(1, int(round(config.steps / max(1, spec.recurrence * 3))))
            offset = sum((index + 1) * ord(char) for index, char in enumerate(spec.name)) % period
            recurrent_hit = (step + offset) % period == 0
            sparse_hit = spec.name in {"rare_critical", "noise"} and (step + offset) % max(2, period * 2) == 0
            if recurrent_hit or sparse_hit:
                base = unit(np.array(spec.center, dtype=np.float64))
                noise = rng.normal(0.0, spec.noise, size=3)
                vector = unit(base + noise)
                kind = "contradiction" if spec.contradiction_target else "trace"
                if spec.name == "noise":
                    kind = "noise"
                if spec.name == "rare_critical":
                    kind = "rare_critical"
                step_events.append(
                    TraceEvent(
                        step=step,
                        group=spec.name,
                        vector=vector,
                        dependency=spec.dependency,
                        usefulness=spec.usefulness,
                        contradiction_target=spec.contradiction_target,
                        kind=kind,
                    )
                )
        if len(step_events) > config.max_events_per_step:
            ranked = sorted(
                step_events,
                key=lambda event: (event.group == "noise", -event.dependency, -event.usefulness, event.group),
            )
            step_events = ranked[: config.max_events_per_step]
        events.extend(step_events)
    if not events:
        raise RuntimeError("No events were generated. Check scenario configuration.")
    return events


def choose_slot(event: TraceEvent, slots: list[SlotState], config: Config) -> tuple[SlotState, str, float, float]:
    active = [slot for slot in slots if slot.active]
    if not active:
        free = next((slot for slot in slots if not slot.active), None)
        if free is None:
            raise RuntimeError("No active slots and no free slots.")
        return free, "new_free_slot", math.inf, -math.inf

    distances = [(cosine_distance(slot.center, event.vector), slot_survival(slot, config), slot) for slot in active]
    distances.sort(key=lambda row: (row[0], -row[1], row[2].slot))
    best_distance, best_survival, best_slot = distances[0]
    event_admission_potential = event.usefulness + event.dependency
    low_potential_event = event_admission_potential < config.admission_threshold

    if event.contradiction_target is not None and best_slot.dominant_group == event.contradiction_target:
        free = next((slot for slot in slots if not slot.active), None)
        if free is not None:
            return free, "branch_for_contradiction", best_distance, best_survival

    if low_potential_event and best_slot.admitted and best_distance > config.match_threshold * 0.35:
        provisional_slots = [slot for slot in active if not slot.admitted]
        if provisional_slots:
            provisional_slots.sort(key=lambda slot: (slot_survival(slot, config), slot.strength, -slot.age, slot.slot))
            chosen = provisional_slots[0]
            return chosen, "provisional_low_potential", best_distance, slot_survival(chosen, config)
        free = next((slot for slot in slots if not slot.active), None)
        if free is not None:
            return free, "new_provisional_slot", best_distance, best_survival

    if best_distance <= config.match_threshold:
        return best_slot, "matched_existing", best_distance, best_survival

    if best_distance >= config.branch_threshold:
        free = next((slot for slot in slots if not slot.active), None)
        if free is not None:
            return free, "new_branch_slot", best_distance, best_survival

    free = next((slot for slot in slots if not slot.active), None)
    if free is not None:
        return free, "new_free_slot", best_distance, best_survival

    weakest = min(active, key=lambda slot: (slot.admitted, slot_survival(slot, config), slot.strength, -slot.age, slot.slot))
    weakest_survival = slot_survival(weakest, config)
    if weakest_survival + config.overwrite_margin < best_survival:
        return weakest, "overwrite_low_survival", best_distance, weakest_survival
    return best_slot, "forced_match_capacity_full", best_distance, best_survival


def adjust_depth_from_survival(slot: SlotState, config: Config, multiplier: float) -> float:
    if not slot.active:
        return 0.0
    previous_depth = slot.depth
    survival = slot_survival(slot, config)
    max_depth = float(config.shell_count - 1)
    desired_depth = max_depth * sigmoid((survival - 1.35) / max(1e-6, config.survival_temperature))
    if not slot.admitted:
        desired_depth = min(desired_depth, config.provisional_depth_cap)
    if slot.conflict > 0.65 or slot.contradiction_evidence > 0.0:
        release_signal = max(slot.conflict - 0.65, slot.contradiction_evidence - config.contradiction_release_threshold)
        conflict_release = sigmoid(release_signal / max(1e-6, config.survival_temperature))
        desired_depth *= 1.0 - conflict_release
    if desired_depth >= slot.depth:
        slot.depth += multiplier * config.inward_rate * (desired_depth - slot.depth)
    else:
        slot.depth += multiplier * config.outward_rate * (desired_depth - slot.depth)
    slot.depth = min(max(slot.depth, 0.0), max_depth)
    slot.center = project_to_shell(slot.center, slot.depth, config)
    return float(slot.depth - previous_depth)


def apply_contradiction_pressure(event: TraceEvent, slots: list[SlotState], config: Config) -> tuple[int, float]:
    if event.contradiction_target is None:
        return 0, 0.0
    touched = 0
    total_pressure = 0.0
    for slot in slots:
        if not slot.active:
            continue
        target_mass = slot.group_mass.get(event.contradiction_target, 0.0)
        if target_mass <= 0.0:
            continue
        total_mass = max(1e-12, sum(slot.group_mass.values()))
        share = target_mass / total_mass
        pressure = share * (1.0 + cosine_distance(slot.center, event.vector))
        slot.conflict = 0.82 * slot.conflict + pressure
        slot.contradiction_evidence = 0.88 * slot.contradiction_evidence + pressure
        slot.usefulness *= 1.0 - 0.10 * share
        slot.strength *= 1.0 - 0.08 * share
        update_admission(slot, config)
        touched += 1
        total_pressure += pressure
    return touched, total_pressure


def apply_neighborhood_support(event: TraceEvent, slots: list[SlotState], config: Config) -> tuple[int, float]:
    potential = event.usefulness + event.dependency
    if potential < config.support_min_potential:
        return 0, 0.0
    touched = 0
    total_support = 0.0
    for slot in slots:
        if not slot.active:
            continue
        if event.group in slot.group_mass:
            continue
        distance = cosine_distance(slot.center, project_to_shell(event.vector, slot.depth, config))
        if distance > config.support_threshold:
            continue
        closeness = 1.0 - distance / max(1e-12, config.support_threshold)
        support = config.support_gain * closeness * potential
        slot.evidence += support
        slot.usefulness = max(slot.usefulness, 0.94 * slot.usefulness + 0.06 * event.usefulness)
        slot.dependency = max(slot.dependency, 0.94 * slot.dependency + 0.06 * event.dependency)
        slot.downstream_support = min(25.0, slot.downstream_support + support)
        slot.support_mass[event.group] = slot.support_mass.get(event.group, 0.0) + support
        slot.strength = min(10.0, slot.strength + 0.18 * support)
        slot.age = max(0, slot.age - 2)
        update_admission(slot, config)
        adjust_depth_from_survival(slot, config, multiplier=0.25)
        touched += 1
        total_support += support
    return touched, total_support


def reset_slot(slot: SlotState, event: TraceEvent, config: Config) -> None:
    slot.active = True
    slot.center = project_to_shell(event.vector, 0.0, config)
    slot.depth = 0.0
    slot.strength = config.initial_strength
    slot.evidence = 1.0
    slot.usefulness = event.usefulness
    slot.dependency = event.dependency
    slot.conflict = 0.0
    slot.contradiction_evidence = 0.0
    slot.downstream_support = 0.0
    slot.admitted = False
    slot.age = 0
    slot.updates = 1
    slot.dominant_group = event.group
    slot.group_mass = {event.group: 1.0}
    slot.support_mass = {}


def update_slot(slot: SlotState, event: TraceEvent, action: str, config: Config) -> dict[str, float | str | int]:
    previous_depth = slot.depth
    previous_center = slot.center.copy()
    previous_survival = slot_survival(slot, config) if slot.active else -math.inf
    previous_group = slot.dominant_group

    if action in {"new_free_slot", "new_branch_slot", "new_provisional_slot", "branch_for_contradiction"} or not slot.active:
        reset_slot(slot, event, config)
    elif action == "overwrite_low_survival":
        slot.released_count += 1
        reset_slot(slot, event, config)
    else:
        distance = cosine_distance(slot.center, event.vector)
        contradiction = 0.0
        if event.contradiction_target is not None and slot.dominant_group == event.contradiction_target:
            contradiction = 1.0 + distance
        elif distance > config.branch_threshold:
            contradiction = max(0.0, distance - config.branch_threshold)

        lr = slot_learning_rate(slot, config)
        target = project_to_shell(event.vector, slot.depth, config)
        slot.center = tangent_step(slot.center, target, lr, slot.depth, config)
        slot.evidence = 0.97 * slot.evidence + 1.0
        slot.usefulness = 0.92 * slot.usefulness + 0.08 * event.usefulness
        slot.dependency = max(0.96 * slot.dependency, event.dependency)
        slot.conflict = 0.88 * slot.conflict + contradiction
        slot.contradiction_evidence *= 0.90
        slot.strength = min(10.0, 0.985 * slot.strength + 0.20 + 0.12 * event.usefulness + 0.10 * event.dependency)
        slot.age = 0
        slot.updates += 1
        slot.group_mass[event.group] = slot.group_mass.get(event.group, 0.0) + 1.0
        slot.dominant_group = max(slot.group_mass.items(), key=lambda row: (row[1], row[0]))[0]

    update_admission(slot, config)
    adjust_depth_from_survival(slot, config, multiplier=1.0)
    survival = slot_survival(slot, config)

    movement = float(np.linalg.norm(slot.center - previous_center)) if np.all(np.isfinite(previous_center)) else 0.0
    shell_changed = abs(slot.depth - previous_depth)
    return {
        "slot": slot.slot,
        "action": action,
        "previous_group": previous_group,
        "dominant_group": slot.dominant_group,
        "previous_survival": previous_survival,
        "survival": survival,
        "previous_depth": previous_depth,
        "depth": slot.depth,
        "depth_delta": shell_changed,
        "movement": movement,
        "strength": slot.strength,
        "evidence": slot.evidence,
        "usefulness": slot.usefulness,
        "dependency": slot.dependency,
        "conflict": slot.conflict,
        "contradiction_evidence": slot.contradiction_evidence,
        "downstream_support": slot.downstream_support,
        "support_diversity": len(slot.support_mass),
        "admitted": int(slot.admitted),
        "dominant_share": slot.dominant_share(),
    }


def decay_idle_slots(slots: list[SlotState], touched_slot: int, config: Config) -> list[dict[str, float | str | int]]:
    releases: list[dict[str, float | str | int]] = []
    for slot in slots:
        if not slot.active or slot.slot == touched_slot:
            continue
        slot.age += 1
        decay = slot_decay(slot, config)
        slot.strength *= max(0.0, 1.0 - decay)
        if not slot.admitted:
            slot.strength *= max(0.0, 1.0 - decay * config.provisional_decay_multiplier)
        slot.evidence *= max(0.0, 1.0 - 0.35 * decay)
        slot.downstream_support *= config.support_decay
        if slot.support_mass:
            slot.support_mass = {
                group: mass * config.support_decay
                for group, mass in slot.support_mass.items()
                if mass * config.support_decay > 1e-6
            }
        slot.conflict *= 0.985
        slot.contradiction_evidence *= 0.995
        update_admission(slot, config)
        adjust_depth_from_survival(slot, config, multiplier=0.18)
        survival = slot_survival(slot, config)
        provisional_release = (not slot.admitted) and survival < config.release_threshold and slot.strength < 0.45
        contradiction_release = (
            slot.contradiction_evidence >= config.contradiction_release_threshold
            and survival < config.release_threshold + 1.25
        )
        if provisional_release or contradiction_release:
            releases.append(
                {
                    "slot": slot.slot,
                    "released_group": slot.dominant_group,
                    "survival": survival,
                    "strength": slot.strength,
                    "age": slot.age,
                }
            )
            slot.active = False
            slot.released_count += 1
            slot.center = np.zeros(3, dtype=np.float64)
            slot.depth = 0.0
            slot.strength = 0.0
            slot.evidence = 0.0
            slot.usefulness = 0.0
            slot.dependency = 0.0
            slot.conflict = 0.0
            slot.contradiction_evidence = 0.0
            slot.downstream_support = 0.0
            slot.admitted = False
            slot.age = 0
            slot.updates = 0
            slot.dominant_group = "empty"
            slot.group_mass = {}
            slot.support_mass = {}
    return releases


def compute_global_metrics(slots: list[SlotState], prototypes: dict[str, np.ndarray], config: Config) -> dict[str, float]:
    active = [slot for slot in slots if slot.active]
    if not active:
        return {
            "active_slots": 0.0,
            "mean_depth": 0.0,
            "inner_mass": 0.0,
            "middle_mass": 0.0,
            "outer_mass": 0.0,
            "mean_strength": 0.0,
            "mean_trace_error": math.nan,
            "free_slots": float(config.slots),
        }
    depths = np.array([slot.depth for slot in active], dtype=np.float64)
    strengths = np.array([slot.strength for slot in active], dtype=np.float64)
    errors = []
    for slot in active:
        proto = prototypes.get(slot.dominant_group)
        if proto is not None:
            errors.append(cosine_distance(slot.center, project_to_shell(proto, slot.depth, config)))
    error_mean = float(np.mean(errors)) if errors else math.nan
    max_depth = float(config.shell_count - 1)
    return {
        "active_slots": float(len(active)),
        "free_slots": float(config.slots - len(active)),
        "mean_depth": float(np.mean(depths)),
        "inner_mass": float(np.mean(depths > 0.72 * max_depth)),
        "middle_mass": float(np.mean((depths > 0.32 * max_depth) & (depths <= 0.72 * max_depth))),
        "outer_mass": float(np.mean(depths <= 0.32 * max_depth)),
        "mean_strength": float(np.mean(strengths)),
        "mean_trace_error": error_mean,
    }


def run_dynamics(config: Config) -> tuple[list[SlotState], list[dict[str, float | str | int]], list[dict[str, float]], dict[str, np.ndarray]]:
    validate_config(config)
    specs = default_group_specs(config.steps)
    prototypes = {spec.name: unit(np.array(spec.center, dtype=np.float64)) for spec in specs}
    events = build_events(config, specs)
    slots = [SlotState(slot=index) for index in range(config.slots)]

    event_rows: list[dict[str, float | str | int]] = []
    metric_rows: list[dict[str, float]] = []

    for event_index, event in enumerate(events):
        contradiction_slots, contradiction_pressure = apply_contradiction_pressure(event, slots, config)
        support_slots, support_amount = apply_neighborhood_support(event, slots, config)
        slot, action, best_distance, best_survival = choose_slot(event, slots, config)
        update = update_slot(slot, event, action, config)
        releases = decay_idle_slots(slots, slot.slot, config)
        metrics = compute_global_metrics(slots, prototypes, config)
        metrics.update({"event_index": float(event_index), "step": float(event.step), "release_count": float(len(releases))})
        metric_rows.append(metrics)

        row: dict[str, float | str | int] = {
            "event_index": event_index,
            "step": event.step,
            "group": event.group,
            "kind": event.kind,
            "contradiction_target": event.contradiction_target or "",
            "best_distance": best_distance,
            "best_survival": best_survival,
            **update,
            "contradiction_slots": contradiction_slots,
            "contradiction_pressure": contradiction_pressure,
            "support_slots": support_slots,
            "support_amount": support_amount,
            "released_slots_after_event": len(releases),
        }
        event_rows.append(row)
    return slots, event_rows, metric_rows, prototypes


def final_group_summary(
    slots: list[SlotState],
    event_rows: list[dict[str, float | str | int]],
    prototypes: dict[str, np.ndarray],
    config: Config,
) -> list[dict[str, float | str | int]]:
    groups = sorted(prototypes)
    rows: list[dict[str, float | str | int]] = []
    for group in groups:
        candidate_slots = [slot for slot in slots if slot.active and group in slot.group_mass]
        if candidate_slots:
            best = max(candidate_slots, key=lambda slot: (slot.group_mass[group], slot.strength, slot.depth))
            trace_error = cosine_distance(best.center, project_to_shell(prototypes[group], best.depth, config))
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
        count = sum(1 for row in event_rows if row["group"] == group)
        rows.append(
            {
                "group": group,
                "events": count,
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


def slot_table(slots: list[SlotState], config: Config) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for slot in slots:
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
                "contradiction_evidence": slot.contradiction_evidence,
                "downstream_support": slot.downstream_support,
                "support_diversity": len(slot.support_mass),
                "admitted": int(slot.admitted),
                "age": slot.age,
                "updates": slot.updates,
                "released_count": slot.released_count,
                "x": float(slot.center[0]),
                "y": float(slot.center[1]),
                "z": float(slot.center[2]),
            }
        )
    return rows


def plot_shell_state(slots: list[SlotState], prototypes: dict[str, np.ndarray], config: Config, output_path: Path) -> None:
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

    for group, proto in prototypes.items():
        point = config.outer_radius * proto
        ax.scatter([point[0]], [point[1]], [point[2]], marker=".", s=16, color=GROUP_COLORS.get(group, "#333333"), alpha=0.35)

    for slot in slots:
        if not slot.active:
            continue
        color = GROUP_COLORS.get(slot.dominant_group, "#333333")
        size = 42.0 + 28.0 * slot.strength
        ax.scatter([slot.center[0]], [slot.center[1]], [slot.center[2]], color=color, s=size, edgecolor="black", linewidth=0.5)
        ax.text(slot.center[0], slot.center[1], slot.center[2], f"{slot.slot}:{slot.dominant_group}", fontsize=7)

    limit = config.outer_radius * 1.18
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Final nested geometry state: slot depth, group identity, and strength")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=42)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lifecycle(group_rows: list[dict[str, float | str | int]], output_path: Path) -> None:
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
    axes[0].set_title("Per-group lifecycle summary")
    axes[1].bar(xs, strength, color=colors)
    axes[1].set_ylabel("strength")
    axes[2].bar(xs, error, color=colors)
    axes[2].set_ylabel("trace error")
    axes[3].bar(xs, survival, color=colors)
    axes[3].set_ylabel("survival utility")
    axes[3].set_xticks(xs)
    axes[3].set_xticklabels(groups, rotation=35, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metrics(metric_rows: list[dict[str, float]], output_path: Path) -> None:
    steps = np.array([row["event_index"] for row in metric_rows])
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(steps, [row["active_slots"] for row in metric_rows], label="active slots")
    axes[0].plot(steps, [row["free_slots"] for row in metric_rows], label="free slots")
    axes[0].set_ylabel("slots")
    axes[0].legend()
    axes[1].plot(steps, [row["outer_mass"] for row in metric_rows], label="outer")
    axes[1].plot(steps, [row["middle_mass"] for row in metric_rows], label="middle")
    axes[1].plot(steps, [row["inner_mass"] for row in metric_rows], label="inner")
    axes[1].set_ylabel("slot fraction")
    axes[1].legend()
    axes[2].plot(steps, [row["mean_depth"] for row in metric_rows], color="#984ea3")
    axes[2].set_ylabel("mean depth")
    axes[3].plot(steps, [row["mean_trace_error"] for row in metric_rows], color="#e41a1c")
    axes[3].set_ylabel("mean trace error")
    axes[3].set_xlabel("event index")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_event_timeline(event_rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.8))
    action_to_y = {
        "new_free_slot": 0,
        "new_branch_slot": 1,
        "new_provisional_slot": 2,
        "branch_for_contradiction": 3,
        "provisional_low_potential": 4,
        "matched_existing": 5,
        "forced_match_capacity_full": 6,
        "overwrite_low_survival": 7,
    }
    for row in event_rows:
        group = str(row["group"])
        action = str(row["action"])
        y = action_to_y[action]
        ax.scatter(
            float(row["event_index"]),
            y,
            s=34 + 20 * float(row["depth"]),
            color=GROUP_COLORS.get(group, "#333333"),
            alpha=0.84,
            edgecolor="black",
            linewidth=0.25,
        )
    ax.set_yticks(list(action_to_y.values()))
    ax.set_yticklabels(list(action_to_y.keys()))
    ax.set_xlabel("event index")
    ax.set_title("Every update decision: slot action over time")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_slot_group_heatmap(slots: list[SlotState], output_path: Path) -> None:
    groups = sorted({group for slot in slots for group in slot.group_mass})
    if not groups:
        raise RuntimeError("Cannot plot heatmap without group assignments.")
    matrix = np.zeros((len(slots), len(groups)), dtype=np.float64)
    for row, slot in enumerate(slots):
        total = sum(slot.group_mass.values())
        if total <= 0.0:
            continue
        for col, group in enumerate(groups):
            matrix[row, col] = slot.group_mass.get(group, 0.0) / total
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(slots)))
    ax.set_yticklabels([f"slot {slot.slot}" for slot in slots])
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_title("Slot composition: which traces share or compete for the same slot")
    fig.colorbar(im, ax=ax, label="group share in slot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_validation(group_rows: list[dict[str, float | str | int]], metric_rows: list[dict[str, float]]) -> dict[str, bool]:
    by_group = {str(row["group"]): row for row in group_rows}

    def finite_metric(group: str, key: str) -> float:
        value = float(by_group[group][key])
        if not math.isfinite(value):
            return math.inf
        return value

    def survived(group: str, *, max_error: float = 0.18, min_depth: float = 0.0) -> bool:
        row = by_group[group]
        slot = int(row["slot"])
        if slot < 0:
            return False
        error = float(row["trace_error"])
        depth = float(row["depth"])
        return math.isfinite(error) and error < max_error and depth >= min_depth

    stable_depth = finite_metric("stable", "depth")
    noise_strength = finite_metric("noise", "strength")
    noise_slot = int(by_group["noise"]["slot"])
    noise_share = float(by_group["noise"]["group_share_in_slot"])
    obsolete_error = finite_metric("obsolete_old", "trace_error")
    replacement_error = finite_metric("replacement", "trace_error")
    rare_error = finite_metric("rare_critical", "trace_error")
    rare_depth = finite_metric("rare_critical", "depth")
    merge_slots = {int(by_group["merge_a"]["slot"]), int(by_group["merge_b"]["slot"])}
    branch_slots = {int(by_group["branch_root"]["slot"]), int(by_group["branch_up"]["slot"]), int(by_group["branch_down"]["slot"])}
    branch_slots_without_released = {slot for slot in branch_slots if slot >= 0}
    max_inner_mass = max(row["inner_mass"] for row in metric_rows)
    max_middle_mass = max(row["middle_mass"] for row in metric_rows)
    final_outer_mass = metric_rows[-1]["outer_mass"]
    final_middle_mass = metric_rows[-1]["middle_mass"]

    return {
        "stable_consolidated_inward": stable_depth > 1.25,
        "rare_critical_survived": rare_error < 0.18 and rare_depth > 0.75,
        "replacement_beats_obsolete": replacement_error < obsolete_error,
        "noise_remains_weak": noise_slot < 0 or noise_share < 0.5 or noise_strength < 2.5,
        "duplicate_sources_merge": len(merge_slots) == 1,
        "branch_root_final_survives": survived("branch_root"),
        "branch_up_final_survives": survived("branch_up"),
        "branch_down_final_survives": survived("branch_down"),
        "important_branches_survive_under_capacity": all(
            survived(group) for group in ("branch_root", "branch_up", "branch_down")
        ),
        "branches_can_separate": len(branch_slots_without_released) > 1,
        "some_inner_consolidation_occurs": max_inner_mass > 0.0,
        "middle_layer_used": max_middle_mass > 0.0 and final_middle_mass > 0.0,
        "outer_geometry_still_available": final_outer_mass > 0.0,
    }


def write_outputs(
    slots: list[SlotState],
    event_rows: list[dict[str, float | str | int]],
    metric_rows: list[dict[str, float]],
    prototypes: dict[str, np.ndarray],
    config: Config,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    group_rows = final_group_summary(slots, event_rows, prototypes, config)
    slot_rows = slot_table(slots, config)
    validation = summarize_validation(group_rows, metric_rows)

    write_csv(config.output_dir / "event_log.csv", event_rows)
    write_csv(config.output_dir / "metrics_over_time.csv", metric_rows)
    write_csv(config.output_dir / "slot_table.csv", slot_rows)
    write_csv(config.output_dir / "group_summary.csv", group_rows)

    plot_shell_state(slots, prototypes, config, config.output_dir / "nested_geometry_final_state.png")
    plot_lifecycle(group_rows, config.output_dir / "group_lifecycle_summary.png")
    plot_metrics(metric_rows, config.output_dir / "nested_metrics_over_time.png")
    plot_event_timeline(event_rows, config.output_dir / "event_decision_timeline.png")
    plot_slot_group_heatmap(slots, config.output_dir / "slot_group_heatmap.png")

    summary = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "validation": validation,
        "final_metrics": metric_rows[-1],
        "groups": group_rows,
        "slots": slot_rows,
        "artifacts": {
            "event_log": str(config.output_dir / "event_log.csv"),
            "metrics_over_time": str(config.output_dir / "metrics_over_time.csv"),
            "slot_table": str(config.output_dir / "slot_table.csv"),
            "group_summary": str(config.output_dir / "group_summary.csv"),
            "final_state_plot": str(config.output_dir / "nested_geometry_final_state.png"),
            "lifecycle_plot": str(config.output_dir / "group_lifecycle_summary.png"),
            "metrics_plot": str(config.output_dir / "nested_metrics_over_time.png"),
            "timeline_plot": str(config.output_dir / "event_decision_timeline.png"),
            "heatmap_plot": str(config.output_dir / "slot_group_heatmap.png"),
        },
    }
    with (config.output_dir / "nested_geometry_dynamics_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def print_report(summary: dict[str, object]) -> None:
    groups = summary["groups"]
    validation = summary["validation"]
    final_metrics = summary["final_metrics"]
    artifacts = summary["artifacts"]
    if not isinstance(groups, list) or not isinstance(validation, dict) or not isinstance(final_metrics, dict) or not isinstance(artifacts, dict):
        raise TypeError("Invalid summary structure.")

    print("\nMANUAL NESTED-GEOMETRY DYNAMICS")
    print("=" * 132)
    print(
        f"active_slots={final_metrics['active_slots']:.0f} "
        f"free_slots={final_metrics['free_slots']:.0f} "
        f"mean_depth={final_metrics['mean_depth']:.4f} "
        f"outer={final_metrics['outer_mass']:.3f} "
        f"middle={final_metrics['middle_mass']:.3f} "
        f"inner={final_metrics['inner_mass']:.3f} "
        f"mean_error={final_metrics['mean_trace_error']:.4f}"
    )
    print("-" * 132)
    print(f"{'group':>18} {'events':>7} {'slot':>5} {'depth':>8} {'radius':>8} {'strength':>10} {'survival':>10} {'error':>10} {'share':>8}")
    for row in groups:
        print(
            f"{row['group']:>18} {int(row['events']):7d} {int(row['slot']):5d} "
            f"{float(row['depth']):8.4f} {float(row['shell_radius']):8.4f} "
            f"{float(row['strength']):10.4f} {float(row['survival']):10.4f} "
            f"{float(row['trace_error']):10.4f} {float(row['group_share_in_slot']):8.4f}"
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
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--slots", type=int, default=9)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--inner-radius", type=float, default=0.85)
    parser.add_argument("--outer-radius", type=float, default=3.0)
    parser.add_argument("--match-threshold", type=float, default=0.10)
    parser.add_argument("--branch-threshold", type=float, default=0.22)
    parser.add_argument("--initial-strength", type=float, default=0.55)
    parser.add_argument("--evidence-gain", type=float, default=0.75)
    parser.add_argument("--usefulness-gain", type=float, default=1.20)
    parser.add_argument("--dependency-gain", type=float, default=1.25)
    parser.add_argument("--conflict-gain", type=float, default=1.85)
    parser.add_argument("--age-decay", type=float, default=0.012)
    parser.add_argument("--outer-learning-rate", type=float, default=0.42)
    parser.add_argument("--learning-timescale", type=float, default=0.62)
    parser.add_argument("--outward-decay", type=float, default=0.055)
    parser.add_argument("--decay-timescale", type=float, default=0.72)
    parser.add_argument("--inward-rate", type=float, default=0.16)
    parser.add_argument("--outward-rate", type=float, default=0.22)
    parser.add_argument("--survival-temperature", type=float, default=0.35)
    parser.add_argument("--release-threshold", type=float, default=-0.65)
    parser.add_argument("--overwrite-margin", type=float, default=0.18)
    parser.add_argument("--admission-threshold", type=float, default=1.85)
    parser.add_argument("--admission-temperature", type=float, default=0.35)
    parser.add_argument("--provisional-depth-cap", type=float, default=0.85)
    parser.add_argument("--provisional-decay-multiplier", type=float, default=2.50)
    parser.add_argument("--contradiction-release-threshold", type=float, default=1.15)
    parser.add_argument("--support-threshold", type=float, default=0.16)
    parser.add_argument("--support-gain", type=float, default=0.45)
    parser.add_argument("--support-min-potential", type=float, default=1.00)
    parser.add_argument("--support-decay", type=float, default=0.985)
    parser.add_argument("--support-admission-gain", type=float, default=1.20)
    parser.add_argument("--support-diversity-gain", type=float, default=0.70)
    parser.add_argument("--max-events-per-step", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/05_nested_geometry/results/gco-nested-geometry-dynamics-seed0"),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = Config(
        seed=args.seed,
        steps=args.steps,
        slots=args.slots,
        shell_count=args.shell_count,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        match_threshold=args.match_threshold,
        branch_threshold=args.branch_threshold,
        initial_strength=args.initial_strength,
        evidence_gain=args.evidence_gain,
        usefulness_gain=args.usefulness_gain,
        dependency_gain=args.dependency_gain,
        conflict_gain=args.conflict_gain,
        age_decay=args.age_decay,
        outer_learning_rate=args.outer_learning_rate,
        learning_timescale=args.learning_timescale,
        outward_decay=args.outward_decay,
        decay_timescale=args.decay_timescale,
        inward_rate=args.inward_rate,
        outward_rate=args.outward_rate,
        survival_temperature=args.survival_temperature,
        release_threshold=args.release_threshold,
        overwrite_margin=args.overwrite_margin,
        admission_threshold=args.admission_threshold,
        admission_temperature=args.admission_temperature,
        provisional_depth_cap=args.provisional_depth_cap,
        provisional_decay_multiplier=args.provisional_decay_multiplier,
        contradiction_release_threshold=args.contradiction_release_threshold,
        support_threshold=args.support_threshold,
        support_gain=args.support_gain,
        support_min_potential=args.support_min_potential,
        support_decay=args.support_decay,
        support_admission_gain=args.support_admission_gain,
        support_diversity_gain=args.support_diversity_gain,
        max_events_per_step=args.max_events_per_step,
        output_dir=args.output_dir,
    )
    slots, event_rows, metric_rows, prototypes = run_dynamics(config)
    summary = write_outputs(slots, event_rows, metric_rows, prototypes, config)
    print_report(summary)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
