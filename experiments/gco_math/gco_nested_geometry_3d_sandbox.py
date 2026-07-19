#!/usr/bin/env python3
"""Pure 3D nested-geometry sandbox.

This is still not a neural model.  It visualizes nested representation shells as
transparent dense globes.  The number of shells is configurable: three shells
are only the default starting point, not an architectural assumption.
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
from matplotlib.colors import to_hex


@dataclass(frozen=True)
class Geometry3DConfig:
    shell_names: tuple[str, ...]
    shell_radii: tuple[float, ...]
    points_per_shell: int
    surface_resolution: int
    target_shell_index: int
    theta_deg: float
    phi_deg: float
    update_x: float
    update_y: float
    update_z: float
    outer_scale: float
    middle_scale: float
    inner_scale: float
    view_elev: float
    view_azim: float


def parse_csv_floats(name: str, value: str) -> tuple[float, ...]:
    pieces = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not pieces:
        raise ValueError(f"{name} must contain at least one comma-separated value.")
    numbers: list[float] = []
    for piece in pieces:
        try:
            number = float(piece)
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-float value: {piece!r}.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} contains a non-finite value: {piece!r}.")
        numbers.append(number)
    return tuple(numbers)


def parse_csv_strings(name: str, value: str) -> tuple[str, ...]:
    pieces = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not pieces:
        raise ValueError(f"{name} must contain at least one comma-separated value.")
    return pieces


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def validate_config(config: Geometry3DConfig) -> None:
    if len(config.shell_names) != len(config.shell_radii):
        raise ValueError(
            f"shell_names length {len(config.shell_names)} does not match "
            f"shell_radii length {len(config.shell_radii)}."
        )
    if len(config.shell_radii) < 2:
        raise ValueError("At least two nested shells are required.")
    previous = 0.0
    for index, radius in enumerate(config.shell_radii):
        positive_float(f"shell_radii[{index}]", radius)
        if radius <= previous:
            raise ValueError("shell_radii must be strictly increasing from inner to outer shell.")
        previous = radius
    if not (0 <= config.target_shell_index < len(config.shell_radii)):
        raise ValueError("target_shell_index is outside the shell list.")
    if config.points_per_shell < 32:
        raise ValueError("points_per_shell must be at least 32 for a dense globe.")
    if config.surface_resolution < 12:
        raise ValueError("surface_resolution must be at least 12.")
    for name in ("theta_deg", "phi_deg", "view_elev", "view_azim"):
        value = float(getattr(config, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    for name in ("outer_scale", "middle_scale", "inner_scale"):
        nonnegative_float(name, float(getattr(config, name)))
    update = np.array([config.update_x, config.update_y, config.update_z], dtype=np.float64)
    if not np.all(np.isfinite(update)):
        raise ValueError("update vector must be finite.")
    if float(np.linalg.norm(update)) <= 0.0:
        raise ValueError("update vector must be non-zero.")


def shell_colors(count: int) -> tuple[str, ...]:
    if count < 2:
        raise ValueError("Need at least two colors.")
    cmap = plt.get_cmap("viridis")
    return tuple(to_hex(cmap(index / max(1, count - 1))) for index in range(count))


def sphere_mesh(radius: float, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * math.pi, resolution)
    v = np.linspace(0.0, math.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def fibonacci_sphere(radius: float, count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * indices
    x = np.cos(theta) * radial
    y = np.sin(theta) * radial
    return radius * np.stack([x, y, z], axis=1)


def shell_point(radius: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return radius * np.array(
        [
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ],
        dtype=np.float64,
    )


def tangent_normal_parts(point: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = point / np.linalg.norm(point)
    normal_part = float(np.dot(vector, normal)) * normal
    tangent_part = vector - normal_part
    return tangent_part, normal_part, normal


def set_3d_axes(ax, config: Geometry3DConfig, title: str) -> None:
    outer = config.shell_radii[-1]
    limit = outer * 1.22
    ax.set_title(title)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=config.view_elev, azim=config.view_azim)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_shells(ax, config: Geometry3DConfig, *, dense_points: bool) -> None:
    colors = shell_colors(len(config.shell_radii))
    for index, (name, radius, color) in enumerate(zip(config.shell_names, config.shell_radii, colors)):
        x, y, z = sphere_mesh(radius, config.surface_resolution)
        alpha = 0.08 + 0.04 * index / max(1, len(config.shell_radii) - 1)
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.0, shade=False)
        ax.plot_wireframe(x, y, z, color=color, alpha=0.15, linewidth=0.35)
        if dense_points:
            points = fibonacci_sphere(radius, config.points_per_shell)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                s=3.5,
                alpha=0.24,
                label=name,
            )
        else:
            ax.plot([], [], [], color=color, label=name)
    ax.legend(loc="upper left", fontsize=8)


def quiver(ax, start: np.ndarray, vector: np.ndarray, color: str, label: str) -> None:
    ax.quiver(
        start[0],
        start[1],
        start[2],
        vector[0],
        vector[1],
        vector[2],
        color=color,
        linewidth=2.2,
        arrow_length_ratio=0.12,
        label=label,
    )


def tangent_plane_patch(point: np.ndarray, normal: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, normal))) > 0.92:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_a = np.cross(normal, axis)
    basis_a = basis_a / np.linalg.norm(basis_a)
    basis_b = np.cross(normal, basis_a)
    grid = np.linspace(-radius, radius, 2)
    a, b = np.meshgrid(grid, grid)
    patch = point[:, None, None] + basis_a[:, None, None] * a + basis_b[:, None, None] * b
    return patch[0], patch[1], patch[2]


def plot_nested_globes(config: Geometry3DConfig, output_path: Path) -> dict[str, float]:
    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    draw_shells(ax, config, dense_points=True)
    set_3d_axes(ax, config, "Dense nested globes: inner core to outer trace")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "shell_count": float(len(config.shell_radii)),
        "total_dense_points": float(len(config.shell_radii) * config.points_per_shell),
        "outer_radius": float(config.shell_radii[-1]),
    }


def plot_tangent_projection(config: Geometry3DConfig, output_path: Path) -> dict[str, float]:
    target_radius = config.shell_radii[config.target_shell_index]
    point = shell_point(target_radius, config.theta_deg, config.phi_deg)
    raw_update = np.array([config.update_x, config.update_y, config.update_z], dtype=np.float64)
    tangent_part, normal_part, normal = tangent_normal_parts(point, raw_update)

    fig = plt.figure(figsize=(8.6, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    draw_shells(ax, config, dense_points=False)
    plane = tangent_plane_patch(point, normal, radius=target_radius * 0.38)
    ax.plot_surface(*plane, color="#2ca02c", alpha=0.16, linewidth=0.0)
    ax.scatter([point[0]], [point[1]], [point[2]], color="#111111", s=45)
    quiver(ax, point, raw_update, "#d62728", "raw update")
    quiver(ax, point, tangent_part, "#2ca02c", "tangent component")
    quiver(ax, point + tangent_part, normal_part, "#7f7f7f", "normal component")
    set_3d_axes(ax, config, "3D shell-local update: tangent movement vs normal damage")
    ax.legend(loc="upper left", fontsize=8)
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
        "target_shell_radius": float(target_radius),
    }


def plot_dense_many_layers(config: Geometry3DConfig, output_path: Path) -> dict[str, float]:
    fig = plt.figure(figsize=(9.2, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    colors = shell_colors(len(config.shell_radii))
    for index, (name, radius, color) in enumerate(zip(config.shell_names, config.shell_radii, colors)):
        points = fibonacci_sphere(radius, config.points_per_shell)
        cut = points[:, 0] >= -0.18 * config.shell_radii[-1]
        visible = points[cut]
        ax.scatter(
            visible[:, 0],
            visible[:, 1],
            visible[:, 2],
            color=color,
            s=4.5,
            alpha=0.34,
            label=f"{index}: {name}",
        )
    set_3d_axes(ax, config, "Dense cutaway: many nested layers can fit")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"visible_cutaway_fraction": 0.5, "shell_count": float(len(config.shell_radii))}


def write_summary(output_dir: Path, config: Geometry3DConfig, metrics: dict[str, dict[str, float]]) -> Path:
    output_path = output_dir / "nested_geometry_3d_summary.json"
    report = {
        "experiment": "nested_geometry_3d_sandbox",
        "description": "Pure 3D nested geometry with configurable dense shell count.",
        "config": asdict(config),
        "metrics": metrics,
        "core_equations": {
            "nested_state": "z = (z_1, z_2, ..., z_L)",
            "ordered_shells": "r_1 < r_2 < ... < r_L",
            "local_projection": "g = P_T_shell(g) + P_N_shell(g)",
            "shell_timescales": "eta_1 << eta_2 << ... << eta_L",
            "default_three_shells": "core -> schema -> trace",
        },
    }
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis/gco-nested-geometry-3d-sandbox"))
    parser.add_argument("--shell-names", type=str, default="core,schema,trace")
    parser.add_argument("--shell-radii", type=str, default="1.0,1.8,2.7")
    parser.add_argument("--points-per-shell", type=int, default=1200)
    parser.add_argument("--surface-resolution", type=int, default=38)
    parser.add_argument("--target-shell-index", type=int, default=1)
    parser.add_argument("--theta-deg", type=float, default=38.0)
    parser.add_argument("--phi-deg", type=float, default=58.0)
    parser.add_argument("--update-x", type=float, default=0.55)
    parser.add_argument("--update-y", type=float, default=0.38)
    parser.add_argument("--update-z", type=float, default=0.42)
    parser.add_argument("--outer-scale", type=float, default=0.9)
    parser.add_argument("--middle-scale", type=float, default=0.28)
    parser.add_argument("--inner-scale", type=float, default=0.035)
    parser.add_argument("--view-elev", type=float, default=22.0)
    parser.add_argument("--view-azim", type=float, default=38.0)
    return parser


def run(args: argparse.Namespace) -> None:
    config = Geometry3DConfig(
        shell_names=parse_csv_strings("--shell-names", args.shell_names),
        shell_radii=parse_csv_floats("--shell-radii", args.shell_radii),
        points_per_shell=args.points_per_shell,
        surface_resolution=args.surface_resolution,
        target_shell_index=args.target_shell_index,
        theta_deg=args.theta_deg,
        phi_deg=args.phi_deg,
        update_x=args.update_x,
        update_y=args.update_y,
        update_z=args.update_z,
        outer_scale=args.outer_scale,
        middle_scale=args.middle_scale,
        inner_scale=args.inner_scale,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )
    validate_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "nested_globes": plot_nested_globes(config, args.output_dir / "nested_globes_3d.png"),
        "tangent_projection": plot_tangent_projection(config, args.output_dir / "tangent_projection_3d.png"),
        "dense_many_layers": plot_dense_many_layers(config, args.output_dir / "dense_many_layers_3d.png"),
    }
    summary_path = write_summary(args.output_dir, config, metrics)

    print("NESTED GEOMETRY 3D SANDBOX")
    print("=" * 104)
    print(
        f"pure_geometry=yes neural_model=no shells={len(config.shell_radii)} "
        f"points_per_shell={config.points_per_shell}"
    )
    for section, values in metrics.items():
        value_text = ", ".join(f"{name}={value:.4f}" for name, value in values.items())
        print(f"{section}: {value_text}")
    print(f"wrote_dir={args.output_dir}")
    print(f"wrote_json={summary_path}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
