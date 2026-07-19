#!/usr/bin/env python3
"""Pure nested-geometry visualization sandbox.

This file intentionally contains no neural network, no optimizer, and no
training loop.  It only visualizes the mathematical object we are trying to
define before building any model:

    fast trace shell -> middle schema shell -> slow core shell

The diagrams show:

* a nested representation geometry;
* why a flat update moves everything at once;
* tangent/normal decomposition of an update on a shell;
* trace consolidation and decay before any weight update exists.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    core_rx: float
    core_ry: float
    schema_rx: float
    schema_ry: float
    trace_rx: float
    trace_ry: float
    update_x: float
    update_y: float
    shell_angle_deg: float
    trace_scale: float
    schema_scale: float
    core_scale: float
    stage_count: int


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def validate_config(config: GeometryConfig) -> None:
    for name, value in asdict(config).items():
        if name == "shell_angle_deg":
            if not math.isfinite(value):
                raise ValueError("shell_angle_deg must be finite.")
            continue
        if name == "stage_count":
            if int(value) < 2:
                raise ValueError("stage_count must be at least 2.")
            continue
        if name.endswith("_scale"):
            nonnegative_float(name, float(value))
        else:
            positive_float(name, float(value))
    if not (config.core_rx < config.schema_rx < config.trace_rx):
        raise ValueError("Expected core_rx < schema_rx < trace_rx.")
    if not (config.core_ry < config.schema_ry < config.trace_ry):
        raise ValueError("Expected core_ry < schema_ry < trace_ry.")


def ellipse_points(rx: float, ry: float, count: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if count < 16:
        raise ValueError("ellipse point count must be at least 16.")
    theta = np.linspace(0.0, 2.0 * math.pi, count)
    return rx * np.cos(theta), ry * np.sin(theta)


def ellipse_point(rx: float, ry: float, angle_rad: float) -> np.ndarray:
    return np.array([rx * math.cos(angle_rad), ry * math.sin(angle_rad)], dtype=np.float64)


def ellipse_tangent(rx: float, ry: float, angle_rad: float) -> np.ndarray:
    tangent = np.array([-rx * math.sin(angle_rad), ry * math.cos(angle_rad)], dtype=np.float64)
    norm = np.linalg.norm(tangent)
    if norm <= 0.0 or not math.isfinite(float(norm)):
        raise FloatingPointError("Invalid tangent norm.")
    return tangent / norm


def projection_parts(vector: np.ndarray, unit_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent_component = float(np.dot(vector, unit_direction)) * unit_direction
    normal_component = vector - tangent_component
    return tangent_component, normal_component


def setup_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.axhline(0.0, color="#c7c7c7", linewidth=0.8)
    ax.axvline(0.0, color="#c7c7c7", linewidth=0.8)
    ax.set_xlabel("representation coordinate 1")
    ax.set_ylabel("representation coordinate 2")


def draw_shells(ax: plt.Axes, config: GeometryConfig) -> None:
    shells = (
        ("slow core", config.core_rx, config.core_ry, "#2a6f97", 2.3),
        ("schema shell", config.schema_rx, config.schema_ry, "#f77f00", 2.1),
        ("fast trace shell", config.trace_rx, config.trace_ry, "#6a4c93", 2.1),
    )
    for label, rx, ry, color, width in shells:
        xs, ys = ellipse_points(rx, ry)
        ax.plot(xs, ys, color=color, linewidth=width, label=label)
    ax.legend(loc="upper right", frameon=True)


def arrow(ax: plt.Axes, start: np.ndarray, delta: np.ndarray, color: str, label: str, width: float = 0.012) -> None:
    ax.arrow(
        float(start[0]),
        float(start[1]),
        float(delta[0]),
        float(delta[1]),
        head_width=0.08,
        head_length=0.12,
        length_includes_head=True,
        color=color,
        linewidth=2.0,
        width=width,
        label=label,
    )


def plot_nested_shells(config: GeometryConfig, output_path: Path) -> dict[str, float]:
    angle = math.radians(config.shell_angle_deg)
    raw_update = np.array([config.update_x, config.update_y], dtype=np.float64)
    schema_point = ellipse_point(config.schema_rx, config.schema_ry, angle)
    tangent = ellipse_tangent(config.schema_rx, config.schema_ry, angle)
    tangent_part, normal_part = projection_parts(raw_update, tangent)

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_shells(ax, config)
    setup_axis(ax, "Nested geometry: shells with different update permissions")
    ax.scatter([schema_point[0]], [schema_point[1]], s=70, color="#111111", zorder=5)
    arrow(ax, schema_point, raw_update, "#d62828", "raw update")
    arrow(ax, schema_point, tangent_part, "#008000", "allowed tangent part")
    arrow(ax, schema_point + tangent_part, normal_part, "#7f7f7f", "normal damage part", width=0.008)
    ax.text(-config.trace_rx, config.trace_ry + 0.25, "outer trace can move fast", color="#6a4c93")
    ax.text(-config.schema_rx, config.schema_ry + 0.18, "schema bends carefully", color="#f77f00")
    ax.text(-config.core_rx, config.core_ry + 0.12, "core moves rarely", color="#2a6f97")
    ax.set_xlim(-config.trace_rx - 0.8, config.trace_rx + 1.2)
    ax.set_ylim(-config.trace_ry - 0.8, config.trace_ry + 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    raw_norm = float(np.linalg.norm(raw_update))
    tangent_norm = float(np.linalg.norm(tangent_part))
    normal_norm = float(np.linalg.norm(normal_part))
    return {
        "raw_update_norm": raw_norm,
        "tangent_fraction": tangent_norm / raw_norm,
        "normal_fraction": normal_norm / raw_norm,
    }


def ring_points(rx: float, ry: float, angles_deg: list[float]) -> np.ndarray:
    return np.stack([ellipse_point(rx, ry, math.radians(angle)) for angle in angles_deg], axis=0)


def plot_flat_vs_nested(config: GeometryConfig, output_path: Path) -> dict[str, float]:
    raw_update = np.array([config.update_x, config.update_y], dtype=np.float64)
    angles = [25.0, 80.0, 145.0, 215.0, 300.0]
    core_points = ring_points(config.core_rx, config.core_ry, angles)
    schema_points = ring_points(config.schema_rx, config.schema_ry, angles)
    trace_points = ring_points(config.trace_rx, config.trace_ry, angles)

    flat_core_after = core_points + raw_update
    flat_schema_after = schema_points + raw_update
    flat_trace_after = trace_points + raw_update

    nested_core_after = core_points + config.core_scale * raw_update
    nested_schema_after = []
    for angle_deg, point in zip(angles, schema_points):
        tangent = ellipse_tangent(config.schema_rx, config.schema_ry, math.radians(angle_deg))
        tangent_part, _normal_part = projection_parts(raw_update, tangent)
        nested_schema_after.append(point + config.schema_scale * tangent_part)
    nested_schema_after = np.stack(nested_schema_after, axis=0)
    nested_trace_after = trace_points + config.trace_scale * raw_update

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
    panels = (
        ("Flat update: one movement hits every level", flat_core_after, flat_schema_after, flat_trace_after),
        ("Nested update: movement is shell-local", nested_core_after, nested_schema_after, nested_trace_after),
    )
    for ax, (title, core_after, schema_after, trace_after) in zip(axes, panels):
        draw_shells(ax, config)
        setup_axis(ax, title)
        for before, after, color in (
            (core_points, core_after, "#2a6f97"),
            (schema_points, schema_after, "#f77f00"),
            (trace_points, trace_after, "#6a4c93"),
        ):
            ax.scatter(before[:, 0], before[:, 1], color=color, s=35)
            ax.scatter(after[:, 0], after[:, 1], color=color, marker="x", s=45)
            for start, end in zip(before, after):
                delta = end - start
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1.2, alpha=0.75)
        ax.set_xlim(-config.trace_rx - 0.8, config.trace_rx + 1.2)
        ax.set_ylim(-config.trace_ry - 0.8, config.trace_ry + 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    flat_core_drift = float(np.mean(np.linalg.norm(flat_core_after - core_points, axis=1)))
    nested_core_drift = float(np.mean(np.linalg.norm(nested_core_after - core_points, axis=1)))
    flat_schema_drift = float(np.mean(np.linalg.norm(flat_schema_after - schema_points, axis=1)))
    nested_schema_drift = float(np.mean(np.linalg.norm(nested_schema_after - schema_points, axis=1)))
    return {
        "flat_core_drift": flat_core_drift,
        "nested_core_drift": nested_core_drift,
        "flat_schema_drift": flat_schema_drift,
        "nested_schema_drift": nested_schema_drift,
    }


def plot_tangent_projection(config: GeometryConfig, output_path: Path) -> dict[str, float]:
    angle = math.radians(config.shell_angle_deg)
    point = ellipse_point(config.schema_rx, config.schema_ry, angle)
    tangent = ellipse_tangent(config.schema_rx, config.schema_ry, angle)
    raw_update = np.array([config.update_x, config.update_y], dtype=np.float64)
    tangent_part, normal_part = projection_parts(raw_update, tangent)

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_shells(ax, config)
    setup_axis(ax, "Local shell math: raw update = tangent part + normal damage")
    tangent_line = np.stack([point - 1.3 * tangent, point + 1.3 * tangent], axis=0)
    ax.plot(tangent_line[:, 0], tangent_line[:, 1], color="#008000", linestyle="--", linewidth=2.0)
    ax.scatter([point[0]], [point[1]], color="#111111", s=70, zorder=5)
    arrow(ax, point, raw_update, "#d62828", "g_raw")
    arrow(ax, point, tangent_part, "#008000", "P_T g_raw")
    arrow(ax, point + tangent_part, normal_part, "#7f7f7f", "P_N g_raw", width=0.008)
    ax.text(point[0] + 0.15, point[1] - 0.25, "current representation", color="#111111")
    ax.set_xlim(-config.trace_rx - 0.8, config.trace_rx + 1.2)
    ax.set_ylim(-config.trace_ry - 0.8, config.trace_ry + 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    raw_norm = np.linalg.norm(raw_update)
    damage_angle = math.degrees(math.acos(float(np.dot(raw_update, tangent) / (raw_norm + 1e-12))))
    return {"raw_to_tangent_angle_deg": damage_angle}


def plot_consolidation_lifecycle(config: GeometryConfig, output_path: Path) -> dict[str, float]:
    stage_count = int(config.stage_count)
    theta = np.linspace(0.0, 2.0 * math.pi, stage_count, endpoint=False)
    recurring = np.stack(
        [
            config.trace_rx * np.cos(theta[:stage_count]) * 0.95,
            config.trace_ry * np.sin(theta[:stage_count]) * 0.95,
        ],
        axis=1,
    )
    noise = np.stack(
        [
            config.trace_rx * np.cos(theta[:stage_count] + 0.55) * 1.06,
            config.trace_ry * np.sin(theta[:stage_count] + 0.55) * 1.06,
        ],
        axis=1,
    )
    consolidated = recurring.copy()
    consolidated[:, 0] *= config.schema_rx / config.trace_rx
    consolidated[:, 1] *= config.schema_ry / config.trace_ry
    decayed = noise * 1.18

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    titles = ("1. Fast traces appear", "2. Repetition pulls traces inward", "3. Weak traces decay")
    for ax, title in zip(axes, titles):
        draw_shells(ax, config)
        setup_axis(ax, title)
        ax.set_xlim(-config.trace_rx - 1.0, config.trace_rx + 1.0)
        ax.set_ylim(-config.trace_ry - 1.0, config.trace_ry + 1.0)

    axes[0].scatter(recurring[:, 0], recurring[:, 1], color="#6a4c93", s=45, label="recurring trace")
    axes[0].scatter(noise[:, 0], noise[:, 1], color="#d62828", marker="x", s=50, label="weak/noisy trace")
    axes[0].legend(fontsize=8)

    axes[1].scatter(recurring[:, 0], recurring[:, 1], color="#6a4c93", s=35)
    axes[1].scatter(consolidated[:, 0], consolidated[:, 1], color="#f77f00", s=45)
    for before, after in zip(recurring, consolidated):
        axes[1].plot([before[0], after[0]], [before[1], after[1]], color="#f77f00", linewidth=1.2)

    axes[2].scatter(consolidated[:, 0], consolidated[:, 1], color="#f77f00", s=45, label="surviving schema")
    axes[2].scatter(decayed[:, 0], decayed[:, 1], color="#d62828", marker="x", s=50, label="released trace")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    recurring_shift = float(np.mean(np.linalg.norm(consolidated - recurring, axis=1)))
    decay_shift = float(np.mean(np.linalg.norm(decayed - noise, axis=1)))
    return {"recurring_inward_shift": recurring_shift, "weak_trace_release_shift": decay_shift}


def write_summary(output_dir: Path, config: GeometryConfig, metrics: dict[str, dict[str, float]]) -> Path:
    output_path = output_dir / "nested_geometry_summary.json"
    report = {
        "experiment": "nested_geometry_sandbox",
        "description": "Pure geometry visualization with no neural model.",
        "config": asdict(config),
        "metrics": metrics,
        "core_equations": {
            "nested_state": "z = (z_trace, z_schema, z_core)",
            "timescale_constraint": "||dz_core|| << ||dz_schema|| << ||dz_trace||",
            "projection": "g = P_T(g) + P_N(g)",
            "allowed_update": "dz_shell = eta_shell * P_T(g)",
            "damage_term": "D_normal = ||P_N(g)||",
        },
    }
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis/gco-nested-geometry-sandbox"))
    parser.add_argument("--core-rx", type=float, default=1.15)
    parser.add_argument("--core-ry", type=float, default=0.72)
    parser.add_argument("--schema-rx", type=float, default=2.25)
    parser.add_argument("--schema-ry", type=float, default=1.42)
    parser.add_argument("--trace-rx", type=float, default=3.45)
    parser.add_argument("--trace-ry", type=float, default=2.25)
    parser.add_argument("--update-x", type=float, default=0.86)
    parser.add_argument("--update-y", type=float, default=0.48)
    parser.add_argument("--shell-angle-deg", type=float, default=34.0)
    parser.add_argument("--trace-scale", type=float, default=0.95)
    parser.add_argument("--schema-scale", type=float, default=0.34)
    parser.add_argument("--core-scale", type=float, default=0.04)
    parser.add_argument("--stage-count", type=int, default=7)
    return parser


def run(args: argparse.Namespace) -> None:
    config = GeometryConfig(
        core_rx=args.core_rx,
        core_ry=args.core_ry,
        schema_rx=args.schema_rx,
        schema_ry=args.schema_ry,
        trace_rx=args.trace_rx,
        trace_ry=args.trace_ry,
        update_x=args.update_x,
        update_y=args.update_y,
        shell_angle_deg=args.shell_angle_deg,
        trace_scale=args.trace_scale,
        schema_scale=args.schema_scale,
        core_scale=args.core_scale,
        stage_count=args.stage_count,
    )
    validate_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "nested_shells": plot_nested_shells(config, args.output_dir / "nested_geometry_shells.png"),
        "flat_vs_nested": plot_flat_vs_nested(config, args.output_dir / "flat_vs_nested_update.png"),
        "tangent_projection": plot_tangent_projection(config, args.output_dir / "shell_tangent_projection.png"),
        "consolidation": plot_consolidation_lifecycle(config, args.output_dir / "trace_consolidation_lifecycle.png"),
    }
    summary_path = write_summary(args.output_dir, config, metrics)

    print("NESTED GEOMETRY SANDBOX")
    print("=" * 96)
    print("pure_geometry=yes neural_model=no training_loop=no")
    for section, values in metrics.items():
        value_text = ", ".join(f"{name}={value:.4f}" for name, value in values.items())
        print(f"{section}: {value_text}")
    print(f"wrote_dir={args.output_dir}")
    print(f"wrote_json={summary_path}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
