#!/usr/bin/env python3
"""Conceptual GPT-style geometry visualization.

This is not a trained model inspection.  It is a geometry diagram for the GPT
architecture:

* tokens live as vectors in the residual stream;
* every transformer block moves those vectors;
* attention mixes token vectors across positions;
* MLPs bend vectors along learned feature directions;
* the unembedding/readout turns final vectors into token logits.

The real geometry is high-dimensional.  These figures are low-dimensional
projections meant to build intuition before connecting nested geometry to a
transformer.
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
class TransformerGeometryConfig:
    output_dir: Path
    token_count: int
    layer_count: int
    d_projection: int
    seed: int


def validate_config(config: TransformerGeometryConfig) -> None:
    if config.token_count < 3:
        raise ValueError("token_count must be at least 3.")
    if config.layer_count < 2:
        raise ValueError("layer_count must be at least 2.")
    if config.d_projection != 3:
        raise ValueError("This visualizer currently draws a 3D projection, so d_projection must be 3.")
    if config.seed < 0:
        raise ValueError("seed must be non-negative.")


def deterministic_token_trajectory(config: TransformerGeometryConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    token_angles = np.linspace(0.0, 2.0 * math.pi, config.token_count, endpoint=False)
    positions = np.zeros((config.layer_count + 1, config.token_count, 3), dtype=np.float64)
    positions[0, :, 0] = 1.2 * np.cos(token_angles)
    positions[0, :, 1] = 0.7 * np.sin(token_angles)
    positions[0, :, 2] = np.linspace(-0.35, 0.35, config.token_count)

    semantic_axis = np.array([0.65, 0.22, 0.72], dtype=np.float64)
    semantic_axis = semantic_axis / np.linalg.norm(semantic_axis)
    syntax_axis = np.array([-0.2, 0.95, 0.25], dtype=np.float64)
    syntax_axis = syntax_axis / np.linalg.norm(syntax_axis)
    memory_axis = np.array([0.74, -0.44, 0.29], dtype=np.float64)
    memory_axis = memory_axis / np.linalg.norm(memory_axis)

    for layer in range(1, config.layer_count + 1):
        previous = positions[layer - 1]
        attention_center = previous.mean(axis=0)
        shifted = np.roll(previous, shift=1, axis=0)
        attention_mix = 0.18 * (attention_center - previous) + 0.12 * (shifted - previous)
        feature_gate = np.tanh(previous @ semantic_axis + 0.35 * layer)
        syntax_gate = np.sin(previous @ syntax_axis + 0.45 * layer)
        mlp_bend = (
            0.16 * feature_gate[:, None] * semantic_axis[None, :]
            + 0.09 * syntax_gate[:, None] * syntax_axis[None, :]
        )
        recency_push = (0.035 * layer) * np.linspace(0.0, 1.0, config.token_count)[:, None] * memory_axis[None, :]
        small_rotation = rng.normal(0.0, 0.012, size=previous.shape)
        positions[layer] = previous + attention_mix + mlp_bend + recency_push + small_rotation
    return positions


def setup_3d_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("residual projection 1")
    ax.set_ylabel("residual projection 2")
    ax.set_zlabel("residual projection 3")
    ax.view_init(elev=24.0, azim=42.0)
    ax.set_box_aspect((1.0, 1.0, 0.82))
    ax.grid(alpha=0.25)


def plot_residual_stream_trajectory(positions: np.ndarray, output_path: Path) -> dict[str, float]:
    layer_count = positions.shape[0] - 1
    token_count = positions.shape[1]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, token_count))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for token_index in range(token_count):
        path = positions[:, token_index, :]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colors[token_index], linewidth=2.0)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=colors[token_index], marker="o", s=45)
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=colors[token_index], marker="x", s=65)
        ax.text(path[-1, 0], path[-1, 1], path[-1, 2], f"tok{token_index}", fontsize=8)
    for layer in range(layer_count + 1):
        points = positions[layer]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#888888", alpha=0.25, linewidth=1.0)
    setup_3d_axis(ax, "GPT-like residual stream geometry: token vectors move through layers")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    total_drift = np.linalg.norm(positions[-1] - positions[0], axis=1)
    return {
        "mean_token_drift": float(np.mean(total_drift)),
        "max_token_drift": float(np.max(total_drift)),
    }


def plot_attention_mlp_geometry(positions: np.ndarray, output_path: Path) -> dict[str, float]:
    layer = min(2, positions.shape[0] - 1)
    before = positions[layer - 1]
    after = positions[layer]
    center = before.mean(axis=0)
    token_count = before.shape[0]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, token_count))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(before[:, 0], before[:, 1], before[:, 2], color=colors, s=55, label="before block")
    ax.scatter(after[:, 0], after[:, 1], after[:, 2], color=colors, marker="x", s=65, label="after block")
    ax.scatter([center[0]], [center[1]], [center[2]], color="#111111", s=80, label="attention context")

    for token_index in range(token_count):
        start = before[token_index]
        end = after[token_index]
        delta = end - start
        ax.quiver(
            start[0],
            start[1],
            start[2],
            delta[0],
            delta[1],
            delta[2],
            color=colors[token_index],
            linewidth=1.8,
            arrow_length_ratio=0.14,
        )
        ax.plot(
            [start[0], center[0]],
            [start[1], center[1]],
            [start[2], center[2]],
            color="#666666",
            alpha=0.22,
            linewidth=1.0,
        )
    semantic_axis = np.array([0.65, 0.22, 0.72], dtype=np.float64)
    semantic_axis = semantic_axis / np.linalg.norm(semantic_axis)
    origin = center - 0.45 * semantic_axis
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        0.9 * semantic_axis[0],
        0.9 * semantic_axis[1],
        0.9 * semantic_axis[2],
        color="#d62728",
        linewidth=3.0,
        arrow_length_ratio=0.16,
        label="MLP feature direction",
    )
    setup_3d_axis(ax, "One transformer block: attention mixes, MLP bends feature directions")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    block_move = np.linalg.norm(after - before, axis=1)
    return {"block_mean_move": float(np.mean(block_move)), "block_max_move": float(np.max(block_move))}


def plot_readout_geometry(positions: np.ndarray, output_path: Path) -> dict[str, float]:
    final = positions[-1]
    token_count = final.shape[0]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, token_count))
    readout_axes = np.array(
        [
            [0.88, 0.05, 0.38],
            [-0.3, 0.91, 0.26],
            [0.22, -0.35, 0.91],
        ],
        dtype=np.float64,
    )
    readout_axes = readout_axes / np.linalg.norm(readout_axes, axis=1, keepdims=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(final[:, 0], final[:, 1], final[:, 2], color=colors, s=70, label="final token vectors")
    origin = final.mean(axis=0)
    labels = ("answer A readout", "answer B readout", "answer C readout")
    for axis, label, color in zip(readout_axes, labels, ("#1f77b4", "#2ca02c", "#d62728")):
        start = origin - 0.55 * axis
        ax.quiver(
            start[0],
            start[1],
            start[2],
            1.1 * axis[0],
            1.1 * axis[1],
            1.1 * axis[2],
            color=color,
            linewidth=3.0,
            arrow_length_ratio=0.13,
            label=label,
        )
    for token_index, point in enumerate(final):
        logits = readout_axes @ point
        winner = int(np.argmax(logits))
        ax.text(point[0], point[1], point[2], f"tok{token_index}->A{winner}", fontsize=8)
    setup_3d_axis(ax, "Readout geometry: final residual vectors projected onto output directions")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    logits = final @ readout_axes.T
    margins = np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]
    return {"mean_readout_margin": float(np.mean(margins)), "min_readout_margin": float(np.min(margins))}


def plot_nested_transformer_view(positions: np.ndarray, output_path: Path) -> dict[str, float]:
    final = positions[-1]
    radii = np.linalg.norm(final, axis=1)
    inner = float(np.percentile(radii, 35))
    middle = float(np.percentile(radii, 70))
    outer = max(float(np.max(radii)) + 0.25, middle + 0.25)
    theta = np.linspace(0.0, 2.0 * math.pi, 64)
    phi = np.linspace(0.0, math.pi, 32)
    colors = ("#2a6f97", "#f77f00", "#6a4c93")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for radius, color, label in zip((inner, middle, outer), colors, ("core-like stable region", "schema region", "trace/readout-active region")):
        x = radius * np.outer(np.cos(theta), np.sin(phi))
        y = radius * np.outer(np.sin(theta), np.sin(phi))
        z = radius * np.outer(np.ones_like(theta), np.cos(phi))
        ax.plot_wireframe(x, y, z, color=color, alpha=0.16, linewidth=0.45, label=label)
    ax.scatter(final[:, 0], final[:, 1], final[:, 2], color="#111111", s=55, label="final token vectors")
    for token_index, point in enumerate(final):
        ax.text(point[0], point[1], point[2], f"tok{token_index}", fontsize=8)
    setup_3d_axis(ax, "Nested view over a transformer residual stream projection")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"inner_radius": inner, "middle_radius": middle, "outer_radius": outer}


def write_summary(config: TransformerGeometryConfig, metrics: dict[str, dict[str, float]]) -> Path:
    output_path = config.output_dir / "transformer_geometry_summary.json"
    config_dict = asdict(config)
    config_dict["output_dir"] = str(config.output_dir)
    report = {
        "experiment": "transformer_geometry_visual",
        "description": "Conceptual GPT-style residual stream geometry projection, not a trained model inspection.",
        "config": config_dict,
        "metrics": metrics,
        "geometry_terms": {
            "residual_stream": "the vector space where token representations live through the model",
            "attention": "token-to-token mixing inside the residual stream",
            "mlp": "feature-direction bending inside each block",
            "readout": "projection from final residual vectors to output-token logits",
            "nested_view": "possible shell structure over residual-stream geometry",
        },
    }
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis/gco-transformer-geometry-visual"))
    parser.add_argument("--token-count", type=int, default=7)
    parser.add_argument("--layer-count", type=int, default=6)
    parser.add_argument("--d-projection", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> None:
    config = TransformerGeometryConfig(
        output_dir=args.output_dir,
        token_count=args.token_count,
        layer_count=args.layer_count,
        d_projection=args.d_projection,
        seed=args.seed,
    )
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    positions = deterministic_token_trajectory(config)

    metrics = {
        "residual_stream": plot_residual_stream_trajectory(
            positions,
            config.output_dir / "gpt_residual_stream_geometry.png",
        ),
        "attention_mlp": plot_attention_mlp_geometry(
            positions,
            config.output_dir / "gpt_attention_mlp_geometry.png",
        ),
        "readout": plot_readout_geometry(
            positions,
            config.output_dir / "gpt_readout_geometry.png",
        ),
        "nested_transformer_view": plot_nested_transformer_view(
            positions,
            config.output_dir / "gpt_nested_residual_geometry.png",
        ),
    }
    summary_path = write_summary(config, metrics)
    print("TRANSFORMER GEOMETRY VISUAL")
    print("=" * 96)
    print("conceptual=yes trained_model=no projection=3d")
    for section, values in metrics.items():
        value_text = ", ".join(f"{key}={value:.4f}" for key, value in values.items())
        print(f"{section}: {value_text}")
    print(f"wrote_dir={config.output_dir}")
    print(f"wrote_json={summary_path}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
