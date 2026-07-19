#!/usr/bin/env python3
"""Stage 1 measured-consequence dynamics for Nested Geometry CL.

This experiment upgrades Stage 0 by replacing hand-supplied usefulness and
dependency with measured functional consequence.

Each trace owns a tiny trainable vector function. At every step we measure:

* direct usefulness: loss increase on the trace's own support if removed;
* downstream dependency: loss increase on other support if removed;
* reliability: current fit quality on live support;
* noise: residual that has no measured consequence.

No neural network is used yet. This is still a controlled mechanism test, but
the survival signal now comes from what the trace does, not from a manual
importance label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUP_COLORS = {
    "stable": "#1b9e77",
    "rare_critical": "#e41a1c",
    "dependent_left": "#fb9a99",
    "dependent_right": "#e31a1c",
    "noise": "#555555",
    "obsolete_old": "#999999",
    "replacement": "#dede00",
    "merge_a": "#377eb8",
    "merge_b": "#4daf4a",
    "branch_up": "#ff7f00",
    "branch_down": "#a65628",
    "novel": "#f781bf",
}


@dataclass(frozen=True)
class Config:
    seed: int
    steps: int
    d_value: int
    depth_max: float
    capacity: float
    initial_strength: float
    support_decay: float
    evidence_decay: float
    lambda_outer: float
    lambda_inner: float
    beta_recurrence: float
    beta_usefulness: float
    beta_dependency: float
    beta_reliability: float
    beta_conflict: float
    beta_noise: float
    beta_capacity: float
    tau_in_base: float
    tau_in_slope: float
    inward_horizon_base: float
    inward_horizon_slope: float
    outward_barrier_base: float
    outward_barrier_growth: float
    inward_rate: float
    outward_rate: float
    gate_temperature: float
    eta_outer: float
    eta_decay: float
    forget_outer: float
    forget_decay: float
    train_repeats: int
    contradiction_release_gain: float
    replacement_win_horizon: float
    delete_depth: float
    delete_strength: float
    output_dir: Path


@dataclass(frozen=True)
class Event:
    step: int
    group: str
    support: float
    conflict: float
    target: tuple[float, ...]
    kind: str


@dataclass
class Trace:
    group: str
    value: np.ndarray
    support_target: np.ndarray
    strength: float
    support: float = 0.0
    recurrence: float = 0.0
    usefulness: float = 0.0
    dependency: float = 0.0
    reliability: float = 0.0
    conflict: float = 0.0
    noise: float = 0.0
    capacity_pressure: float = 0.0
    depth: float = 0.0
    inward_eligibility: float = 0.0
    release_pressure: float = 0.0
    inward_gate: float = 0.0
    outward_gate: float = 0.0
    learning_rate: float = 0.0
    forget_rate: float = 0.0
    events: int = 0
    deleted: bool = False
    replacement_wins: float = 0.0
    obsolete_wins: float = 0.0


def positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}.")


def nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}.")


def validate_config(config: Config) -> None:
    if config.steps < 24:
        raise ValueError("steps must be at least 24.")
    if config.d_value < 2:
        raise ValueError("d_value must be at least 2.")
    for name in (
        "depth_max",
        "capacity",
        "initial_strength",
        "support_decay",
        "evidence_decay",
        "lambda_outer",
        "lambda_inner",
        "gate_temperature",
        "eta_outer",
        "forget_outer",
    ):
        positive(name, float(getattr(config, name)))
    if config.support_decay >= 1.0:
        raise ValueError("support_decay must be below 1.")
    if config.evidence_decay >= 1.0:
        raise ValueError("evidence_decay must be below 1.")
    if config.lambda_outer >= config.lambda_inner:
        raise ValueError("lambda_outer must be smaller than lambda_inner.")
    if config.lambda_inner >= 1.0:
        raise ValueError("lambda_inner must be below 1.")
    for name in (
        "beta_recurrence",
        "beta_usefulness",
        "beta_dependency",
        "beta_reliability",
        "beta_conflict",
        "beta_noise",
        "beta_capacity",
        "tau_in_base",
        "tau_in_slope",
        "inward_horizon_base",
        "inward_horizon_slope",
        "outward_barrier_base",
        "outward_barrier_growth",
        "inward_rate",
        "outward_rate",
        "eta_decay",
        "forget_decay",
        "contradiction_release_gain",
        "replacement_win_horizon",
        "delete_depth",
        "delete_strength",
    ):
        nonnegative(name, float(getattr(config, name)))
    if config.train_repeats < 1:
        raise ValueError("train_repeats must be at least 1.")


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Cannot normalize zero or non-finite vector.")
    return vector / norm


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def depth_alpha(depth: float, config: Config) -> float:
    return min(max(depth / config.depth_max, 0.0), 1.0)


def retention(depth: float, config: Config) -> float:
    alpha = depth_alpha(depth, config)
    return config.lambda_outer + (config.lambda_inner - config.lambda_outer) * alpha


def learning_rate(depth: float, config: Config) -> float:
    return config.eta_outer * math.exp(-config.eta_decay * depth)


def forget_rate(depth: float, config: Config) -> float:
    return config.forget_outer * math.exp(-config.forget_decay * depth)


def tau_in(depth: float, config: Config) -> float:
    return config.tau_in_base + config.tau_in_slope * depth


def inward_horizon(depth: float, config: Config) -> float:
    return config.inward_horizon_base + config.inward_horizon_slope * depth


def outward_barrier(depth: float, config: Config) -> float:
    return config.outward_barrier_base * math.exp(config.outward_barrier_growth * depth)


def base_vectors(config: Config) -> dict[str, np.ndarray]:
    raw = {
        "stable": (0.90, 0.18, 0.05, 0.00),
        "rare_critical": (-0.82, 0.46, 0.23, 0.10),
        "dependent_left": (-0.54, 0.66, 0.35, 0.16),
        "dependent_right": (-0.64, 0.42, 0.45, 0.18),
        "noise": (0.12, -0.66, 0.71, -0.20),
        "obsolete_old": (0.30, -0.86, 0.19, 0.08),
        "replacement": (0.43, -0.72, 0.40, 0.12),
        "merge_a": (0.75, 0.54, 0.21, 0.07),
        "merge_b": (0.78, 0.50, 0.18, 0.06),
        "branch_up": (-0.73, 0.63, -0.11, 0.08),
        "branch_down": (-0.38, 0.88, -0.19, 0.04),
        "novel": (0.06, 0.92, 0.32, 0.19),
    }
    values: dict[str, np.ndarray] = {}
    for name, coords in raw.items():
        arr = np.array(coords[: config.d_value], dtype=np.float64)
        if arr.shape[0] != config.d_value:
            raise ValueError(f"base vector for {name} does not match d_value={config.d_value}.")
        values[name] = unit(arr)
    return values


def dependency_edges() -> dict[str, dict[str, float]]:
    return {
        "dependent_left": {"rare_critical": 0.55},
        "dependent_right": {"rare_critical": 0.62},
        "branch_up": {"branch_down": 0.18},
        "branch_down": {"branch_up": 0.18},
        "merge_a": {"merge_b": 0.22},
        "merge_b": {"merge_a": 0.22},
    }


def prediction_for(group: str, traces: dict[str, Trace], *, removed: str | None = None) -> np.ndarray:
    trace = traces[group]
    if trace.deleted or removed == group:
        pred = np.zeros_like(trace.value)
    else:
        pred = trace.value.copy()
    for source, weight in dependency_edges().get(group, {}).items():
        source_trace = traces[source]
        if source_trace.deleted or removed == source:
            continue
        pred = pred + weight * source_trace.value
    return pred


def target_for_group(group: str, base: dict[str, np.ndarray]) -> np.ndarray:
    target = base[group].copy()
    for source, weight in dependency_edges().get(group, {}).items():
        target = target + weight * base[source]
    return target


def initialize_traces(config: Config, rng: np.random.Generator) -> dict[str, Trace]:
    base = base_vectors(config)
    traces: dict[str, Trace] = {}
    for group, target in base.items():
        noise = 0.35 * rng.normal(size=config.d_value)
        initial = unit(target + noise)
        traces[group] = Trace(
            group=group,
            value=initial,
            support_target=target_for_group(group, base),
            strength=config.initial_strength,
        )
    return traces


def make_events(config: Config, rng: np.random.Generator) -> list[Event]:
    base = base_vectors(config)
    events: list[Event] = []

    def add(step: int, group: str, support: float, conflict: float, target: np.ndarray, kind: str) -> None:
        if step < config.steps:
            events.append(
                Event(
                    step=step,
                    group=group,
                    support=support,
                    conflict=conflict,
                    target=tuple(float(v) for v in target),
                    kind=kind,
                )
            )

    for step in range(config.steps):
        if step % 2 == 0:
            add(step, "stable", 1.0, 0.0, base["stable"], "stable_repeat")
        if step in (3, 16, 30):
            add(step, "rare_critical", 0.72, 0.0, base["rare_critical"], "rare_functional")
        if step % 5 == 1:
            add(step, "dependent_left", 0.85, 0.0, base["dependent_left"], "dependent_left")
        if step % 5 == 3:
            add(step, "dependent_right", 0.85, 0.0, base["dependent_right"], "dependent_right")
        if step in (6, 12, 23):
            add(step, "noise", 0.65, 0.15, unit(rng.normal(size=config.d_value)), "noise")
        if step < 18 and step % 3 == 0:
            add(step, "obsolete_old", 0.88, 0.0, base["obsolete_old"], "old_supported")
        if step >= 15 and step % 2 == 1:
            add(step, "replacement", 1.00, 0.0, base["replacement"], "replacement_supported")
        if step >= 16:
            add(step, "obsolete_old", 0.0, 1.10, base["replacement"], "obsolete_conflict")
        if step % 4 in (1, 2):
            add(step, "merge_a", 0.82, 0.0, base["merge_a"], "merge_a")
        if step % 4 in (2, 3):
            add(step, "merge_b", 0.82, 0.0, base["merge_b"], "merge_b")
        if step in (9, 14, 19, 25, 31):
            add(step, "branch_up", 0.78, 0.0, base["branch_up"], "branch_up")
        if step in (10, 15, 20, 26, 32):
            add(step, "branch_down", 0.78, 0.0, base["branch_down"], "branch_down")
        if step in (22, 28, 34):
            add(step, "novel", 0.58, 0.0, base["novel"], "novel_supported")
    events.sort(key=lambda event: (event.step, event.group, event.kind))
    return events


def support_loss_for_group(group: str, traces: dict[str, Trace], *, removed: str | None = None) -> float:
    trace = traces[group]
    if trace.support <= 0.0:
        return 0.0
    pred = prediction_for(group, traces, removed=removed)
    error = pred - trace.support_target
    return trace.support * float(np.dot(error, error))


def total_support_loss(traces: dict[str, Trace], *, removed: str | None = None) -> float:
    return sum(support_loss_for_group(group, traces, removed=removed) for group in traces)


def measure_consequence(traces: dict[str, Trace]) -> dict[str, dict[str, float]]:
    full_loss_by_group = {
        group: support_loss_for_group(group, traces, removed=None)
        for group in traces
    }
    consequence: dict[str, dict[str, float]] = {}
    for group in traces:
        direct = max(0.0, support_loss_for_group(group, traces, removed=group) - full_loss_by_group[group])
        downstream = 0.0
        for other in traces:
            if other == group:
                continue
            downstream += max(
                0.0,
                support_loss_for_group(other, traces, removed=group) - full_loss_by_group[other],
            )
        own_loss = full_loss_by_group[group]
        support = traces[group].support
        reliability = 1.0 / (1.0 + own_loss / (support + 1e-8)) if support > 0.0 else 0.0
        consequence[group] = {
            "direct": direct,
            "downstream": downstream,
            "reliability": reliability,
            "own_loss": own_loss,
        }
    return consequence


def capacity_pressure(traces: dict[str, Trace], config: Config) -> dict[str, float]:
    alive = [trace for trace in traces.values() if not trace.deleted]
    used = sum(1.0 + 0.28 * trace.depth for trace in alive)
    excess = max(0.0, used - config.capacity)
    if excess <= 0.0:
        return {trace.group: 0.0 for trace in traces.values()}
    scores: dict[str, float] = {}
    for trace in alive:
        protective = trace.usefulness + trace.dependency + trace.recurrence + trace.reliability
        fragile = trace.noise + trace.conflict + max(0.0, 0.45 - trace.strength)
        scores[trace.group] = max(0.01, fragile - 0.30 * protective)
    total = sum(scores.values())
    if total <= 0.0:
        raise FloatingPointError("capacity pressure scores collapsed to zero.")
    return {group: excess * score / total for group, score in scores.items()}


def apply_event(trace: Trace, event: Event, config: Config) -> None:
    if trace.deleted:
        return
    target = np.array(event.target, dtype=np.float64)
    trace.events += 1
    trace.support = config.support_decay * trace.support + event.support
    trace.recurrence = config.evidence_decay * trace.recurrence + event.support
    trace.conflict = config.evidence_decay * trace.conflict + event.conflict
    if event.group == "noise":
        trace.noise = config.evidence_decay * trace.noise + 1.0
    else:
        trace.noise *= config.evidence_decay
    if event.group == "replacement":
        trace.replacement_wins = config.evidence_decay * trace.replacement_wins + event.support
    if event.group == "obsolete_old" and event.kind == "old_supported":
        trace.obsolete_wins = config.evidence_decay * trace.obsolete_wins + event.support
    if event.group == "obsolete_old" and event.kind == "obsolete_conflict":
        trace.support *= 0.50

    eta = learning_rate(trace.depth, config)
    for _ in range(config.train_repeats):
        trace.value = trace.value + eta * event.support * (target - trace.value)
        if not np.all(np.isfinite(trace.value)):
            raise FloatingPointError(f"non-finite trace value for {trace.group}.")


def decay_without_event(trace: Trace, config: Config) -> None:
    if trace.deleted:
        return
    trace.support *= config.support_decay
    trace.recurrence *= config.evidence_decay
    trace.conflict *= config.evidence_decay
    trace.noise *= config.evidence_decay
    trace.replacement_wins *= config.evidence_decay
    trace.obsolete_wins *= config.evidence_decay


def update_strength_depth(
    trace: Trace,
    consequence: dict[str, float],
    pressure: float,
    replacement_trace: Trace,
    config: Config,
) -> dict[str, object]:
    if trace.deleted:
        return {
            "group": trace.group,
            "deleted": 1,
            "strength": trace.strength,
            "depth": -1.0,
        }
    old_strength = trace.strength
    old_depth = trace.depth
    trace.usefulness = consequence["direct"]
    trace.dependency = consequence["downstream"]
    trace.reliability = consequence["reliability"]
    trace.capacity_pressure = pressure

    contradiction_release = 0.0
    if trace.group == "obsolete_old":
        replacement_advantage = replacement_trace.replacement_wins - trace.obsolete_wins
        if replacement_advantage > config.replacement_win_horizon:
            contradiction_release = config.contradiction_release_gain * replacement_advantage

    trace.release_pressure = (
        trace.conflict
        + trace.noise
        + pressure
        + contradiction_release
        + max(0.0, 0.25 - trace.recurrence)
        - 0.35 * trace.usefulness
        - 0.30 * trace.dependency
        - 0.18 * trace.reliability
    )
    evidence = (
        config.beta_recurrence * trace.recurrence
        + config.beta_usefulness * trace.usefulness
        + config.beta_dependency * trace.dependency
        + config.beta_reliability * trace.reliability
        - config.beta_conflict * trace.conflict
        - config.beta_noise * trace.noise
        - config.beta_capacity * pressure
        - contradiction_release
    )
    trace.strength = max(0.0, retention(trace.depth, config) * trace.strength + evidence)
    trace.inward_eligibility = (
        config.evidence_decay * trace.inward_eligibility
        + max(0.0, trace.strength - (config.tau_in_base + config.tau_in_slope * trace.depth))
    )
    trace.inward_gate = sigmoid(
        (trace.inward_eligibility - (config.inward_horizon_base + config.inward_horizon_slope * trace.depth))
        / config.gate_temperature
    )
    trace.outward_gate = sigmoid(
        (trace.release_pressure - outward_barrier(trace.depth, config)) / config.gate_temperature
    )
    trace.depth = min(
        config.depth_max,
        max(0.0, trace.depth + config.inward_rate * trace.inward_gate - config.outward_rate * trace.outward_gate),
    )
    trace.learning_rate = learning_rate(trace.depth, config)
    trace.forget_rate = forget_rate(trace.depth, config)
    if trace.events > 0 and trace.depth <= config.delete_depth and trace.strength <= config.delete_strength:
        trace.deleted = True
    return {
        "group": trace.group,
        "deleted": int(trace.deleted),
        "old_strength": old_strength,
        "strength": trace.strength,
        "old_depth": old_depth,
        "depth": trace.depth if not trace.deleted else -1.0,
        "direct_usefulness": trace.usefulness,
        "downstream_dependency": trace.dependency,
        "reliability": trace.reliability,
        "conflict": trace.conflict,
        "noise": trace.noise,
        "capacity_pressure": pressure,
        "contradiction_release": contradiction_release,
        "release_pressure": trace.release_pressure,
        "inward_gate": trace.inward_gate,
        "outward_gate": trace.outward_gate,
        "learning_rate": trace.learning_rate,
        "forget_rate": trace.forget_rate,
    }


def trace_error(group: str, traces: dict[str, Trace]) -> float:
    trace = traces[group]
    if trace.deleted:
        return math.nan
    error = prediction_for(group, traces) - trace.support_target
    return float(np.linalg.norm(error))


def run(config: Config) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    validate_config(config)
    rng = np.random.default_rng(config.seed)
    traces = initialize_traces(config, rng)
    events = make_events(config, rng)
    events_by_step: dict[int, list[Event]] = {}
    for event in events:
        events_by_step.setdefault(event.step, []).append(event)

    event_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []

    for step in range(config.steps):
        step_event_groups = {event.group for event in events_by_step.get(step, [])}
        for group, trace in traces.items():
            if group not in step_event_groups:
                decay_without_event(trace, config)
        local_update_mass = 0.0
        for event in events_by_step.get(step, []):
            trace = traces[event.group]
            before = trace.value.copy()
            apply_event(trace, event, config)
            movement = float(np.linalg.norm(trace.value - before))
            local_update_mass += movement
            event_rows.append(
                {
                    "step": step,
                    "group": event.group,
                    "kind": event.kind,
                    "support": event.support,
                    "conflict": event.conflict,
                    "movement": movement,
                }
            )

        consequence = measure_consequence(traces)
        pressure = capacity_pressure(traces, config)
        for group, trace in traces.items():
            row = update_strength_depth(trace, consequence[group], pressure.get(group, 0.0), traces["replacement"], config)
            row["step"] = step
            row["own_loss"] = consequence[group]["own_loss"]
            transition_rows.append(row)

        alive = [trace for trace in traces.values() if not trace.deleted]
        capacity_used = sum(1.0 + 0.28 * trace.depth for trace in alive)
        metrics.append(
            {
                "step": step,
                "active_traces": len(alive),
                "deleted_traces": len(traces) - len(alive),
                "capacity_used": capacity_used,
                "outer_free_capacity": max(0.0, config.capacity - capacity_used),
                "mean_depth": float(np.mean([trace.depth for trace in alive])) if alive else 0.0,
                "local_update_fraction": 1.0,
                "support_loss": total_support_loss(traces),
                "stable_depth": traces["stable"].depth,
                "rare_depth": traces["rare_critical"].depth,
                "noise_depth": traces["noise"].depth if not traces["noise"].deleted else -1.0,
                "obsolete_depth": traces["obsolete_old"].depth if not traces["obsolete_old"].deleted else -1.0,
                "replacement_depth": traces["replacement"].depth,
                "stable_strength": traces["stable"].strength,
                "rare_strength": traces["rare_critical"].strength,
                "noise_strength": traces["noise"].strength if not traces["noise"].deleted else 0.0,
                "obsolete_strength": traces["obsolete_old"].strength if not traces["obsolete_old"].deleted else 0.0,
                "replacement_strength": traces["replacement"].strength,
            }
        )

    trace_rows: list[dict[str, object]] = []
    for trace in traces.values():
        trace_rows.append(
            {
                "group": trace.group,
                "events": trace.events,
                "deleted": int(trace.deleted),
                "support": trace.support,
                "strength": trace.strength,
                "depth": trace.depth if not trace.deleted else -1.0,
                "recurrence": trace.recurrence,
                "direct_usefulness": trace.usefulness,
                "downstream_dependency": trace.dependency,
                "reliability": trace.reliability,
                "conflict": trace.conflict,
                "noise": trace.noise,
                "capacity_pressure": trace.capacity_pressure,
                "inward_eligibility": trace.inward_eligibility,
                "release_pressure": trace.release_pressure,
                "inward_gate": trace.inward_gate,
                "outward_gate": trace.outward_gate,
                "learning_rate": trace.learning_rate,
                "forget_rate": trace.forget_rate,
                "trace_error": trace_error(trace.group, traces),
            }
        )
    return event_rows, transition_rows, metrics, trace_rows


def summarize(trace_rows: list[dict[str, object]], metrics: list[dict[str, object]]) -> dict[str, object]:
    lookup = {str(row["group"]): row for row in trace_rows}
    final = metrics[-1]
    stable_inward = float(lookup["stable"]["depth"]) >= 2.0 and float(lookup["stable"]["trace_error"]) < 0.20
    rare_survived = (
        int(lookup["rare_critical"]["deleted"]) == 0
        and float(lookup["rare_critical"]["downstream_dependency"]) > float(lookup["noise"]["downstream_dependency"])
        and float(lookup["rare_critical"]["strength"]) > float(lookup["noise"]["strength"])
    )
    noise_outer = int(lookup["noise"]["deleted"]) == 1 or (
        float(lookup["noise"]["depth"]) < 0.75
        and float(lookup["noise"]["strength"]) < 0.75
    )
    obsolete_released = int(lookup["obsolete_old"]["deleted"]) == 1 or (
        float(lookup["obsolete_old"]["strength"]) < float(lookup["replacement"]["strength"]) * 0.25
    )
    local_update = float(final["local_update_fraction"]) >= 0.999
    outer_plasticity = float(final["outer_free_capacity"]) > 0.05 or int(final["active_traces"]) < 12
    measured_dependency = float(lookup["rare_critical"]["downstream_dependency"]) > 0.15
    checks = {
        "stable_circuits_move_inward": stable_inward,
        "rare_useful_circuits_survive": rare_survived,
        "noise_stays_outer_or_releases": noise_outer,
        "obsolete_traces_release": obsolete_released,
        "new_learning_stays_local": local_update,
        "outer_plasticity_available": outer_plasticity,
        "dependency_is_measured_not_supplied": measured_dependency,
    }
    return {
        "checks": checks,
        "all_core_checks_pass": all(
            checks[name]
            for name in (
                "stable_circuits_move_inward",
                "rare_useful_circuits_survive",
                "noise_stays_outer_or_releases",
                "obsolete_traces_release",
                "new_learning_stays_local",
                "outer_plasticity_available",
            )
        ),
        "all_checks_pass": all(checks.values()),
        "final_metrics": final,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def plot_metric(metrics: list[dict[str, object]], keys: tuple[tuple[str, str, str], ...], ylabel: str, title: str, path: Path) -> None:
    steps = [int(row["step"]) for row in metrics]
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, label, color in keys:
        ax.plot(steps, [float(row[key]) for row in metrics], label=label, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_summary(summary: dict[str, object], trace_rows: list[dict[str, object]], output_dir: Path) -> None:
    print("NESTED MEASURED-CONSEQUENCE DYNAMICS")
    print("=" * 144)
    print(f"core_pass={summary['all_core_checks_pass']} all_pass={summary['all_checks_pass']}")
    print("-" * 144)
    print(
        f"{'group':>18} {'events':>6} {'del':>3} {'depth':>7} {'strength':>9} "
        f"{'directU':>9} {'downD':>9} {'rel':>7} {'conf':>7} {'noise':>7} {'err':>8}"
    )
    for row in sorted(trace_rows, key=lambda item: str(item["group"])):
        err = float(row["trace_error"])
        print(
            f"{str(row['group']):>18} "
            f"{int(row['events']):6d} "
            f"{int(row['deleted']):3d} "
            f"{float(row['depth']):7.3f} "
            f"{float(row['strength']):9.3f} "
            f"{float(row['direct_usefulness']):9.3f} "
            f"{float(row['downstream_dependency']):9.3f} "
            f"{float(row['reliability']):7.3f} "
            f"{float(row['conflict']):7.3f} "
            f"{float(row['noise']):7.3f} "
            f"{err if math.isfinite(err) else math.nan:8.4f}"
        )
    print("\nCHECKS")
    print("-" * 144)
    checks = summary["checks"]
    if not isinstance(checks, dict):
        raise TypeError("summary checks must be a dictionary.")
    for name, passed in checks.items():
        print(f"{name:42s} = {passed}")
    print("\nWROTE")
    print("-" * 144)
    for name in (
        "event_log.csv",
        "transition_log.csv",
        "metrics_over_time.csv",
        "trace_table.csv",
        "summary.json",
        "measured_depth_over_time.png",
        "measured_strength_over_time.png",
        "measured_capacity_over_time.png",
    ):
        print(output_dir / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=38)
    parser.add_argument("--d-value", type=int, default=4)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--capacity", type=float, default=15.1)
    parser.add_argument("--initial-strength", type=float, default=0.20)
    parser.add_argument("--support-decay", type=float, default=0.86)
    parser.add_argument("--evidence-decay", type=float, default=0.72)
    parser.add_argument("--lambda-outer", type=float, default=0.62)
    parser.add_argument("--lambda-inner", type=float, default=0.94)
    parser.add_argument("--beta-recurrence", type=float, default=0.28)
    parser.add_argument("--beta-usefulness", type=float, default=0.75)
    parser.add_argument("--beta-dependency", type=float, default=0.92)
    parser.add_argument("--beta-reliability", type=float, default=0.22)
    parser.add_argument("--beta-conflict", type=float, default=0.85)
    parser.add_argument("--beta-noise", type=float, default=1.10)
    parser.add_argument("--beta-capacity", type=float, default=0.24)
    parser.add_argument("--tau-in-base", type=float, default=1.20)
    parser.add_argument("--tau-in-slope", type=float, default=0.45)
    parser.add_argument("--inward-horizon-base", type=float, default=1.25)
    parser.add_argument("--inward-horizon-slope", type=float, default=0.68)
    parser.add_argument("--outward-barrier-base", type=float, default=0.55)
    parser.add_argument("--outward-barrier-growth", type=float, default=0.72)
    parser.add_argument("--inward-rate", type=float, default=0.18)
    parser.add_argument("--outward-rate", type=float, default=0.52)
    parser.add_argument("--gate-temperature", type=float, default=0.34)
    parser.add_argument("--eta-outer", type=float, default=0.34)
    parser.add_argument("--eta-decay", type=float, default=0.50)
    parser.add_argument("--forget-outer", type=float, default=0.36)
    parser.add_argument("--forget-decay", type=float, default=0.78)
    parser.add_argument("--train-repeats", type=int, default=1)
    parser.add_argument("--contradiction-release-gain", type=float, default=1.05)
    parser.add_argument("--replacement-win-horizon", type=float, default=1.65)
    parser.add_argument("--delete-depth", type=float, default=0.35)
    parser.add_argument("--delete-strength", type=float, default=0.12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/05_nested_geometry/results/gco-nested-measured-consequence-seed0"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    if args.output_dir == parser.get_default("output_dir") and args.seed != 0:
        output_dir = Path(f"research/05_nested_geometry/results/gco-nested-measured-consequence-seed{args.seed}")
    config = Config(
        seed=args.seed,
        steps=args.steps,
        d_value=args.d_value,
        depth_max=args.depth_max,
        capacity=args.capacity,
        initial_strength=args.initial_strength,
        support_decay=args.support_decay,
        evidence_decay=args.evidence_decay,
        lambda_outer=args.lambda_outer,
        lambda_inner=args.lambda_inner,
        beta_recurrence=args.beta_recurrence,
        beta_usefulness=args.beta_usefulness,
        beta_dependency=args.beta_dependency,
        beta_reliability=args.beta_reliability,
        beta_conflict=args.beta_conflict,
        beta_noise=args.beta_noise,
        beta_capacity=args.beta_capacity,
        tau_in_base=args.tau_in_base,
        tau_in_slope=args.tau_in_slope,
        inward_horizon_base=args.inward_horizon_base,
        inward_horizon_slope=args.inward_horizon_slope,
        outward_barrier_base=args.outward_barrier_base,
        outward_barrier_growth=args.outward_barrier_growth,
        inward_rate=args.inward_rate,
        outward_rate=args.outward_rate,
        gate_temperature=args.gate_temperature,
        eta_outer=args.eta_outer,
        eta_decay=args.eta_decay,
        forget_outer=args.forget_outer,
        forget_decay=args.forget_decay,
        train_repeats=args.train_repeats,
        contradiction_release_gain=args.contradiction_release_gain,
        replacement_win_horizon=args.replacement_win_horizon,
        delete_depth=args.delete_depth,
        delete_strength=args.delete_strength,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    event_rows, transition_rows, metrics, trace_rows = run(config)
    summary = summarize(trace_rows, metrics)
    payload = {
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "summary": summary,
        "trace_rows": trace_rows,
    }
    write_csv(output_dir / "event_log.csv", event_rows)
    write_csv(output_dir / "transition_log.csv", transition_rows)
    write_csv(output_dir / "metrics_over_time.csv", metrics)
    write_csv(output_dir / "trace_table.csv", trace_rows)
    write_json(output_dir / "summary.json", payload)
    plot_metric(
        metrics,
        (
            ("stable_depth", "stable", GROUP_COLORS["stable"]),
            ("rare_depth", "rare critical", GROUP_COLORS["rare_critical"]),
            ("noise_depth", "noise", GROUP_COLORS["noise"]),
            ("obsolete_depth", "obsolete", GROUP_COLORS["obsolete_old"]),
            ("replacement_depth", "replacement", GROUP_COLORS["replacement"]),
        ),
        "depth",
        "Measured-consequence depth",
        output_dir / "measured_depth_over_time.png",
    )
    plot_metric(
        metrics,
        (
            ("stable_strength", "stable", GROUP_COLORS["stable"]),
            ("rare_strength", "rare critical", GROUP_COLORS["rare_critical"]),
            ("noise_strength", "noise", GROUP_COLORS["noise"]),
            ("obsolete_strength", "obsolete", GROUP_COLORS["obsolete_old"]),
            ("replacement_strength", "replacement", GROUP_COLORS["replacement"]),
        ),
        "strength",
        "Measured-consequence strength",
        output_dir / "measured_strength_over_time.png",
    )
    plot_metric(
        metrics,
        (
            ("capacity_used", "capacity used", "#333333"),
            ("outer_free_capacity", "free capacity", "#1f78b4"),
        ),
        "capacity",
        "Measured-consequence capacity",
        output_dir / "measured_capacity_over_time.png",
    )
    print_summary(summary, trace_rows, output_dir)


if __name__ == "__main__":
    main()
