#!/usr/bin/env python3
"""Stage 0 manual dynamics for Nested Geometry CL.

This experiment has no neural network. It tests only the proposed trace
strength, inward consolidation, outward release, and capacity equations.

The goal is to make the architecture falsifiable before building a trainable
model. Every event, score, gate, and pass/fail condition is written to disk.
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
    "noise": "#555555",
    "obsolete_old": "#999999",
    "replacement": "#dede00",
    "branch_root": "#984ea3",
    "branch_up": "#ff7f00",
    "branch_down": "#a65628",
    "merge_a": "#377eb8",
    "merge_b": "#4daf4a",
    "novel": "#f781bf",
}


@dataclass(frozen=True)
class Config:
    seed: int
    steps: int
    depth_max: float
    capacity: float
    initial_strength: float
    lambda_outer: float
    lambda_inner: float
    evidence_decay: float
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
    contradiction_release_gain: float
    replacement_win_horizon: float
    delete_depth: float
    delete_strength: float
    output_dir: Path


@dataclass(frozen=True)
class Event:
    step: int
    group: str
    recurrence: float
    usefulness: float
    dependency: float
    reliability: float
    conflict: float
    noise: float
    target: tuple[float, float]
    kind: str


@dataclass
class TraceState:
    group: str
    target: np.ndarray
    strength: float
    depth: float = 0.0
    recurrence: float = 0.0
    usefulness: float = 0.0
    dependency: float = 0.0
    reliability: float = 0.0
    conflict: float = 0.0
    noise: float = 0.0
    inward_eligibility: float = 0.0
    release_pressure: float = 0.0
    inward_gate: float = 0.0
    outward_gate: float = 0.0
    learning_rate: float = 0.0
    forget_rate: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
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
    if config.steps < 16:
        raise ValueError("steps must be at least 16.")
    for name in (
        "depth_max",
        "capacity",
        "initial_strength",
        "lambda_outer",
        "lambda_inner",
        "evidence_decay",
        "gate_temperature",
        "eta_outer",
        "forget_outer",
    ):
        positive(name, float(getattr(config, name)))
    if config.lambda_outer >= config.lambda_inner:
        raise ValueError("lambda_outer must be smaller than lambda_inner.")
    if config.lambda_inner >= 1.0:
        raise ValueError("lambda_inner must be below 1.")
    if config.evidence_decay >= 1.0:
        raise ValueError("evidence_decay must be below 1.")
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


def group_targets() -> dict[str, np.ndarray]:
    raw = {
        "stable": (0.95, 0.20),
        "rare_critical": (-0.85, 0.48),
        "noise": (-0.20, -0.95),
        "obsolete_old": (0.26, -0.87),
        "replacement": (0.40, -0.79),
        "branch_root": (-0.55, 0.80),
        "branch_up": (-0.72, 0.68),
        "branch_down": (-0.38, 0.91),
        "merge_a": (0.77, 0.58),
        "merge_b": (0.80, 0.54),
        "novel": (0.07, 0.99),
    }
    return {name: unit(np.array(value, dtype=np.float64)) for name, value in raw.items()}


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("Cannot normalize zero or non-finite vector.")
    return vector / norm


def make_events(config: Config) -> list[Event]:
    targets = group_targets()
    events: list[Event] = []

    def add(
        step: int,
        group: str,
        recurrence: float,
        usefulness: float,
        dependency: float,
        reliability: float,
        conflict: float,
        noise: float,
        kind: str,
    ) -> None:
        if step >= config.steps:
            return
        events.append(
            Event(
                step=step,
                group=group,
                recurrence=recurrence,
                usefulness=usefulness,
                dependency=dependency,
                reliability=reliability,
                conflict=conflict,
                noise=noise,
                target=tuple(float(v) for v in targets[group]),
                kind=kind,
            )
        )

    for step in range(config.steps):
        if step % 2 == 0:
            add(step, "stable", 1.00, 0.62, 0.72, 0.95, 0.00, 0.00, "stable_repeat")
        if step in (2, 14, 26):
            add(step, "rare_critical", 0.28, 1.35, 1.65, 0.96, 0.00, 0.00, "rare_functional")
        if step in (5, 11, 24):
            add(step, "noise", 0.08, 0.02, 0.00, 0.12, 0.10, 1.35, "noise")
        if step < min(config.steps, 18) and step % 3 == 0:
            add(step, "obsolete_old", 0.80, 0.44, 0.28, 0.78, 0.00, 0.00, "old_supported")
        if step >= 14 and step % 2 == 1:
            add(step, "replacement", 1.00, 1.05, 0.82, 0.95, 0.00, 0.00, "replacement_supported")
        if step >= 15:
            add(step, "obsolete_old", 0.05, 0.02, 0.02, 0.20, 1.25, 0.10, "obsolete_conflict")
        if step in (0, 4, 8, 12):
            add(step, "branch_root", 0.58, 0.50, 0.55, 0.86, 0.00, 0.00, "branch_root")
        if step in (10, 14, 18, 22, 27):
            add(step, "branch_up", 0.72, 0.66, 0.63, 0.90, 0.00, 0.00, "branch_up")
        if step in (11, 15, 19, 23, 28):
            add(step, "branch_down", 0.72, 0.66, 0.63, 0.90, 0.00, 0.00, "branch_down")
        if step % 4 in (1, 2):
            add(step, "merge_a", 0.78, 0.55, 0.52, 0.90, 0.00, 0.00, "merge_family")
        if step % 4 in (2, 3):
            add(step, "merge_b", 0.78, 0.55, 0.52, 0.90, 0.00, 0.00, "merge_family")
        if step in (20, 25, 30):
            add(step, "novel", 0.36, 0.72, 0.28, 0.82, 0.00, 0.02, "novel_supported")

    events.sort(key=lambda event: (event.step, event.group, event.kind))
    return events


def initialize_traces(config: Config) -> dict[str, TraceState]:
    traces: dict[str, TraceState] = {}
    for name, target in group_targets().items():
        traces[name] = TraceState(
            group=name,
            target=target,
            strength=config.initial_strength,
            position=target.copy(),
        )
    return traces


def capacity_pressure(traces: dict[str, TraceState], config: Config) -> dict[str, float]:
    alive = [trace for trace in traces.values() if not trace.deleted]
    used = sum(1.0 + 0.30 * trace.depth for trace in alive)
    excess = max(0.0, used - config.capacity)
    if excess <= 0.0:
        return {trace.group: 0.0 for trace in traces.values()}
    release_scores: dict[str, float] = {}
    for trace in alive:
        protective = trace.usefulness + trace.dependency + trace.recurrence + trace.reliability
        fragile = trace.noise + trace.conflict + max(0.0, 0.4 - trace.strength)
        release_scores[trace.group] = max(0.02, fragile - 0.35 * protective)
    total = sum(release_scores.values())
    if total <= 0.0:
        raise FloatingPointError("capacity release scores collapsed to zero.")
    return {name: excess * score / total for name, score in release_scores.items()}


def apply_decay(trace: TraceState, config: Config) -> None:
    trace.recurrence *= config.evidence_decay
    trace.usefulness *= config.evidence_decay
    trace.dependency *= config.evidence_decay
    trace.reliability *= config.evidence_decay
    trace.conflict *= config.evidence_decay
    trace.noise *= config.evidence_decay
    trace.replacement_wins *= config.evidence_decay
    trace.obsolete_wins *= config.evidence_decay


def apply_event(trace: TraceState, event: Event, config: Config) -> None:
    trace.recurrence += event.recurrence
    trace.usefulness += event.usefulness
    trace.dependency += event.dependency
    trace.reliability += event.reliability
    trace.conflict += event.conflict
    trace.noise += event.noise
    trace.events += 1
    if event.group == "replacement":
        trace.replacement_wins += event.recurrence + event.usefulness
    if event.group == "obsolete_old" and event.kind == "old_supported":
        trace.obsolete_wins += event.recurrence + event.usefulness

    target = np.array(event.target, dtype=np.float64)
    eta = learning_rate(trace.depth, config)
    tangent_delta = target - trace.position
    normal = unit(trace.position)
    tangent_delta = tangent_delta - float(np.dot(tangent_delta, normal)) * normal
    trace.position = unit(trace.position + eta * tangent_delta)


def update_trace_state(
    trace: TraceState,
    pressure: float,
    replacement_trace: TraceState,
    config: Config,
) -> dict[str, float | str | int]:
    old_strength = trace.strength
    old_depth = trace.depth

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
        - 0.45 * trace.usefulness
        - 0.35 * trace.dependency
        - 0.20 * trace.reliability
    )

    retained = retention(trace.depth, config) * trace.strength
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
    trace.strength = max(0.0, retained + evidence)

    trace.inward_eligibility = (
        config.evidence_decay * trace.inward_eligibility
        + max(0.0, trace.strength - tau_in(trace.depth, config))
    )
    trace.inward_gate = sigmoid(
        (trace.inward_eligibility - inward_horizon(trace.depth, config)) / config.gate_temperature
    )
    trace.outward_gate = sigmoid(
        (trace.release_pressure - outward_barrier(trace.depth, config)) / config.gate_temperature
    )

    trace.depth = min(
        config.depth_max,
        max(
            0.0,
            trace.depth
            + config.inward_rate * trace.inward_gate
            - config.outward_rate * trace.outward_gate,
        ),
    )
    trace.learning_rate = learning_rate(trace.depth, config)
    trace.forget_rate = forget_rate(trace.depth, config)

    if trace.events > 0 and trace.depth <= config.delete_depth and trace.strength <= config.delete_strength:
        trace.deleted = True

    return {
        "group": trace.group,
        "old_strength": old_strength,
        "strength": trace.strength,
        "old_depth": old_depth,
        "depth": trace.depth,
        "pressure": pressure,
        "contradiction_release": contradiction_release,
        "release_pressure": trace.release_pressure,
        "inward_gate": trace.inward_gate,
        "outward_gate": trace.outward_gate,
        "learning_rate": trace.learning_rate,
        "forget_rate": trace.forget_rate,
        "deleted": int(trace.deleted),
    }


def trace_error(trace: TraceState) -> float:
    if trace.deleted:
        return math.nan
    return float(np.linalg.norm(unit(trace.position) - trace.target))


def run_dynamics(config: Config) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    validate_config(config)
    events = make_events(config)
    traces = initialize_traces(config)

    event_log: list[dict[str, object]] = []
    transition_log: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []

    events_by_step: dict[int, list[Event]] = {}
    for event in events:
        events_by_step.setdefault(event.step, []).append(event)

    for step in range(config.steps):
        for trace in traces.values():
            if not trace.deleted:
                apply_decay(trace, config)

        step_events = events_by_step.get(step, [])
        local_update_mass = 0.0
        total_update_mass = 0.0
        for event in step_events:
            trace = traces[event.group]
            if trace.deleted:
                continue
            before_depth = trace.depth
            before_position = trace.position.copy()
            apply_event(trace, event, config)
            movement = float(np.linalg.norm(trace.position - before_position))
            local_update_mass += movement
            total_update_mass += movement
            event_log.append(
                {
                    "step": step,
                    "group": event.group,
                    "kind": event.kind,
                    "recurrence": event.recurrence,
                    "usefulness": event.usefulness,
                    "dependency": event.dependency,
                    "reliability": event.reliability,
                    "conflict": event.conflict,
                    "noise": event.noise,
                    "depth_before_event": before_depth,
                    "position_movement": movement,
                }
            )

        pressure = capacity_pressure(traces, config)
        replacement_trace = traces["replacement"]
        for trace in traces.values():
            if trace.deleted:
                continue
            row = update_trace_state(trace, pressure.get(trace.group, 0.0), replacement_trace, config)
            row["step"] = step
            transition_log.append(row)

        alive = [trace for trace in traces.values() if not trace.deleted]
        outer_alive = [trace for trace in alive if trace.depth < 0.75]
        mean_depth = float(np.mean([trace.depth for trace in alive])) if alive else 0.0
        max_capacity_used = sum(1.0 + 0.30 * trace.depth for trace in alive)
        local_update_fraction = 1.0 if total_update_mass <= 0.0 else local_update_mass / total_update_mass
        metrics.append(
            {
                "step": step,
                "active_traces": len(alive),
                "deleted_traces": len(traces) - len(alive),
                "mean_depth": mean_depth,
                "outer_traces": len(outer_alive),
                "capacity_used": max_capacity_used,
                "outer_free_capacity": max(0.0, config.capacity - max_capacity_used),
                "local_update_fraction": local_update_fraction,
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

    trace_rows = []
    for trace in traces.values():
        trace_rows.append(
            {
                "group": trace.group,
                "events": trace.events,
                "deleted": int(trace.deleted),
                "strength": trace.strength,
                "depth": trace.depth if not trace.deleted else -1.0,
                "recurrence": trace.recurrence,
                "usefulness": trace.usefulness,
                "dependency": trace.dependency,
                "reliability": trace.reliability,
                "conflict": trace.conflict,
                "noise": trace.noise,
                "inward_eligibility": trace.inward_eligibility,
                "release_pressure": trace.release_pressure,
                "inward_gate": trace.inward_gate,
                "outward_gate": trace.outward_gate,
                "learning_rate": trace.learning_rate,
                "forget_rate": trace.forget_rate,
                "trace_error": trace_error(trace),
                "x": float(trace.position[0]) if not trace.deleted else math.nan,
                "y": float(trace.position[1]) if not trace.deleted else math.nan,
            }
        )

    return event_log, transition_log, metrics, trace_rows


def summarize(trace_rows: list[dict[str, object]], metrics: list[dict[str, object]]) -> dict[str, object]:
    lookup = {str(row["group"]): row for row in trace_rows}
    required = {
        "stable",
        "rare_critical",
        "noise",
        "obsolete_old",
        "replacement",
        "branch_up",
        "branch_down",
        "merge_a",
        "merge_b",
    }
    missing = sorted(required - set(lookup))
    if missing:
        raise RuntimeError(f"Missing trace rows: {missing}")

    final_metrics = metrics[-1]
    stable_inward = float(lookup["stable"]["depth"]) >= 2.0 and float(lookup["stable"]["trace_error"]) < 0.05
    rare_survived = (
        int(lookup["rare_critical"]["deleted"]) == 0
        and float(lookup["rare_critical"]["strength"]) > float(lookup["noise"]["strength"])
        and float(lookup["rare_critical"]["trace_error"]) < 0.08
    )
    noise_outer = int(lookup["noise"]["deleted"]) == 1 or (
        float(lookup["noise"]["depth"]) < 0.75 and float(lookup["noise"]["strength"]) < 0.8
    )
    obsolete_released = int(lookup["obsolete_old"]["deleted"]) == 1 or (
        float(lookup["obsolete_old"]["strength"]) < 0.45
        and float(lookup["replacement"]["strength"]) > float(lookup["obsolete_old"]["strength"])
    )
    local_updates = float(final_metrics["local_update_fraction"]) > 0.999
    outer_plasticity = float(final_metrics["outer_free_capacity"]) > 0.05 or int(final_metrics["outer_traces"]) > 0
    merge_family_survived = (
        int(lookup["merge_a"]["deleted"]) == 0
        and int(lookup["merge_b"]["deleted"]) == 0
        and abs(float(lookup["merge_a"]["depth"]) - float(lookup["merge_b"]["depth"])) < 0.8
    )
    branches_separated = (
        int(lookup["branch_up"]["deleted"]) == 0
        and int(lookup["branch_down"]["deleted"]) == 0
        and abs(float(lookup["branch_up"]["depth"]) - float(lookup["branch_down"]["depth"])) < 1.0
    )

    checks = {
        "stable_circuits_move_inward": stable_inward,
        "rare_useful_circuits_survive": rare_survived,
        "noise_stays_outer_or_releases": noise_outer,
        "obsolete_traces_release": obsolete_released,
        "new_learning_stays_local": local_updates,
        "outer_plasticity_available": outer_plasticity,
        "merge_family_remains_coherent": merge_family_survived,
        "context_branches_survive": branches_separated,
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
        "final_metrics": final_metrics,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV to {path}.")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
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


def plot_depth(metrics: list[dict[str, object]], output_dir: Path) -> None:
    steps = [int(row["step"]) for row in metrics]
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, label, color in (
        ("stable_depth", "stable", GROUP_COLORS["stable"]),
        ("rare_depth", "rare critical", GROUP_COLORS["rare_critical"]),
        ("noise_depth", "noise", GROUP_COLORS["noise"]),
        ("obsolete_depth", "obsolete", GROUP_COLORS["obsolete_old"]),
        ("replacement_depth", "replacement", GROUP_COLORS["replacement"]),
    ):
        ax.plot(steps, [float(row[key]) for row in metrics], label=label, color=color, linewidth=2)
    ax.set_title("Nested depth over time")
    ax.set_xlabel("step")
    ax.set_ylabel("depth")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "depth_over_time.png", dpi=180)
    plt.close(fig)


def plot_strength(metrics: list[dict[str, object]], output_dir: Path) -> None:
    steps = [int(row["step"]) for row in metrics]
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, label, color in (
        ("stable_strength", "stable", GROUP_COLORS["stable"]),
        ("rare_strength", "rare critical", GROUP_COLORS["rare_critical"]),
        ("noise_strength", "noise", GROUP_COLORS["noise"]),
        ("obsolete_strength", "obsolete", GROUP_COLORS["obsolete_old"]),
        ("replacement_strength", "replacement", GROUP_COLORS["replacement"]),
    ):
        ax.plot(steps, [float(row[key]) for row in metrics], label=label, color=color, linewidth=2)
    ax.set_title("Trace strength over time")
    ax.set_xlabel("step")
    ax.set_ylabel("strength")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "strength_over_time.png", dpi=180)
    plt.close(fig)


def plot_capacity(metrics: list[dict[str, object]], output_dir: Path) -> None:
    steps = [int(row["step"]) for row in metrics]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, [float(row["capacity_used"]) for row in metrics], label="used", linewidth=2)
    ax.plot(steps, [float(row["outer_free_capacity"]) for row in metrics], label="free", linewidth=2)
    ax.set_title("Capacity and outer plasticity")
    ax.set_xlabel("step")
    ax.set_ylabel("capacity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "capacity_over_time.png", dpi=180)
    plt.close(fig)


def plot_final_geometry(trace_rows: list[dict[str, object]], output_dir: Path, config: Config) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for depth in np.linspace(0.0, config.depth_max, 4):
        radius = 1.0 - 0.22 * depth
        circle = plt.Circle((0.0, 0.0), radius, fill=False, color="#999999", alpha=0.35)
        ax.add_patch(circle)
    for row in trace_rows:
        group = str(row["group"])
        if int(row["deleted"]):
            continue
        depth = float(row["depth"])
        radius = 1.0 - 0.22 * depth
        direction = unit(np.array([float(row["x"]), float(row["y"])], dtype=np.float64))
        point = radius * direction
        ax.scatter(point[0], point[1], s=90 + 20 * float(row["strength"]), color=GROUP_COLORS.get(group, "#333333"), label=group)
        ax.text(point[0] * 1.06, point[1] * 1.06, group, fontsize=8)
    ax.set_title("Final nested geometry state")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "final_nested_geometry.png", dpi=180)
    plt.close(fig)


def print_summary(summary: dict[str, object], trace_rows: list[dict[str, object]], output_dir: Path) -> None:
    print("NESTED STRENGTH DYNAMICS")
    print("=" * 128)
    checks = summary["checks"]
    if not isinstance(checks, dict):
        raise TypeError("summary checks must be a dictionary.")
    print(f"core_pass={summary['all_core_checks_pass']} all_pass={summary['all_checks_pass']}")
    print("-" * 128)
    print(f"{'group':>18} {'events':>6} {'deleted':>7} {'depth':>8} {'strength':>10} {'R':>8} {'U':>8} {'D':>8} {'C':>8} {'N':>8} {'err':>8}")
    for row in sorted(trace_rows, key=lambda item: str(item["group"])):
        print(
            f"{str(row['group']):>18} "
            f"{int(row['events']):6d} "
            f"{int(row['deleted']):7d} "
            f"{float(row['depth']):8.3f} "
            f"{float(row['strength']):10.3f} "
            f"{float(row['recurrence']):8.3f} "
            f"{float(row['usefulness']):8.3f} "
            f"{float(row['dependency']):8.3f} "
            f"{float(row['conflict']):8.3f} "
            f"{float(row['noise']):8.3f} "
            f"{float(row['trace_error']) if math.isfinite(float(row['trace_error'])) else math.nan:8.4f}"
        )
    print("\nCHECKS")
    print("-" * 128)
    for name, passed in checks.items():
        print(f"{name:42s} = {passed}")
    print("\nWROTE")
    print("-" * 128)
    for name in (
        "event_log.csv",
        "transition_log.csv",
        "metrics_over_time.csv",
        "trace_table.csv",
        "summary.json",
        "depth_over_time.png",
        "strength_over_time.png",
        "capacity_over_time.png",
        "final_nested_geometry.png",
    ):
        print(output_dir / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--capacity", type=float, default=14.6)
    parser.add_argument("--initial-strength", type=float, default=0.18)
    parser.add_argument("--lambda-outer", type=float, default=0.62)
    parser.add_argument("--lambda-inner", type=float, default=0.94)
    parser.add_argument("--evidence-decay", type=float, default=0.72)
    parser.add_argument("--beta-recurrence", type=float, default=0.36)
    parser.add_argument("--beta-usefulness", type=float, default=0.52)
    parser.add_argument("--beta-dependency", type=float, default=0.46)
    parser.add_argument("--beta-reliability", type=float, default=0.20)
    parser.add_argument("--beta-conflict", type=float, default=0.80)
    parser.add_argument("--beta-noise", type=float, default=1.10)
    parser.add_argument("--beta-capacity", type=float, default=0.26)
    parser.add_argument("--tau-in-base", type=float, default=1.35)
    parser.add_argument("--tau-in-slope", type=float, default=0.46)
    parser.add_argument("--inward-horizon-base", type=float, default=1.50)
    parser.add_argument("--inward-horizon-slope", type=float, default=0.72)
    parser.add_argument("--outward-barrier-base", type=float, default=0.55)
    parser.add_argument("--outward-barrier-growth", type=float, default=0.75)
    parser.add_argument("--inward-rate", type=float, default=0.18)
    parser.add_argument("--outward-rate", type=float, default=0.55)
    parser.add_argument("--gate-temperature", type=float, default=0.35)
    parser.add_argument("--eta-outer", type=float, default=0.80)
    parser.add_argument("--eta-decay", type=float, default=0.55)
    parser.add_argument("--forget-outer", type=float, default=0.36)
    parser.add_argument("--forget-decay", type=float, default=0.80)
    parser.add_argument("--contradiction-release-gain", type=float, default=1.15)
    parser.add_argument("--replacement-win-horizon", type=float, default=2.2)
    parser.add_argument("--delete-depth", type=float, default=0.35)
    parser.add_argument("--delete-strength", type=float, default=0.12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/05_nested_geometry/results/gco-nested-strength-dynamics-seed0"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    if args.output_dir == build_parser().get_default("output_dir") and args.seed != 0:
        output_dir = Path(f"research/05_nested_geometry/results/gco-nested-strength-dynamics-seed{args.seed}")
    config = Config(
        seed=args.seed,
        steps=args.steps,
        depth_max=args.depth_max,
        capacity=args.capacity,
        initial_strength=args.initial_strength,
        lambda_outer=args.lambda_outer,
        lambda_inner=args.lambda_inner,
        evidence_decay=args.evidence_decay,
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
        contradiction_release_gain=args.contradiction_release_gain,
        replacement_win_horizon=args.replacement_win_horizon,
        delete_depth=args.delete_depth,
        delete_strength=args.delete_strength,
        output_dir=output_dir,
    )
    np.random.seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_log, transition_log, metrics, trace_rows = run_dynamics(config)
    summary = summarize(trace_rows, metrics)
    payload = {
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "summary": summary,
        "trace_rows": trace_rows,
    }

    write_csv(output_dir / "event_log.csv", event_log)
    write_csv(output_dir / "transition_log.csv", transition_log)
    write_csv(output_dir / "metrics_over_time.csv", metrics)
    write_csv(output_dir / "trace_table.csv", trace_rows)
    write_json(output_dir / "summary.json", payload)
    plot_depth(metrics, output_dir)
    plot_strength(metrics, output_dir)
    plot_capacity(metrics, output_dir)
    plot_final_geometry(trace_rows, output_dir, config)
    print_summary(summary, trace_rows, output_dir)


if __name__ == "__main__":
    main()
