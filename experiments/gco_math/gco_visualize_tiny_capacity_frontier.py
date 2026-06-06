"""Visualize a tiny transformer's representation geometry across data scale.

This script does not train or fine-tune. It loads multiple same-spec
checkpoints, runs the same real-book probe span through every model, projects
residual states into a shared PCA space per layer, and writes one grid image per
layer plus rank/drift metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_native_scratch_transformer import GCONativeTransformer, NativeGCOConfig
from experiments.gco_math.gco_prepare_tiny_cl_base import load_chunks
from experiments.gco_math.gco_visualize_tiny_geometry import (
    build_windows,
    centered_svd_basis,
    collect_states,
    effective_rank,
    pca_2d,
    word_span,
)
from experiments.gco_math.gco_visualize_tiny_geometry_drift import linear_cka, procrustes_align


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def bounded_float(name: str, value: float, low: float, high: float) -> None:
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}.")


def load_checkpoint(path: Path, device: torch.device) -> tuple[GCONativeTransformer, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state_dict", "native_gco_config", "model_config"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint {path} missing fields: {sorted(missing)}.")
    cfg = NativeGCOConfig(**checkpoint["native_gco_config"])
    model_config = checkpoint["model_config"]
    model = GCONativeTransformer(
        vocab_size=int(model_config["vocab_size"]),
        d_model=int(model_config["d_model"]),
        n_layers=int(model_config["n_layers"]),
        n_heads=int(model_config["n_heads"]),
        d_ff=int(model_config["d_ff"]),
        max_seq_len=int(model_config["max_seq_len"]),
        cfg=cfg,
    ).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch for {path}: missing={missing_keys}, unexpected={unexpected_keys}."
        )
    model.eval()
    return model, checkpoint


def geometry_metrics(states: torch.Tensor, segment_mask: torch.Tensor, variance_threshold: float) -> dict[str, float]:
    if states.ndim != 2:
        raise ValueError(f"states must be [samples, dim], got {states.shape}.")
    if segment_mask.shape != (states.shape[0],):
        raise ValueError(f"segment mask shape {segment_mask.shape} does not match sample count {states.shape[0]}.")
    old = states[segment_mask]
    new = states[~segment_mask]
    if old.shape[0] < 2 or new.shape[0] < 2:
        raise ValueError(f"Need at least two first-span and second-span samples, got {old.shape[0]} and {new.shape[0]}.")
    old_basis, old_s = centered_svd_basis(old, variance_threshold)
    new_basis, new_s = centered_svd_basis(new, variance_threshold)
    all_centered = states - states.mean(dim=0, keepdim=True)
    _u, all_s, _vh = torch.linalg.svd(all_centered, full_matrices=False)
    old_center = old.mean(dim=0, keepdim=True)
    new_center = new.mean(dim=0, keepdim=True)
    new_centered_to_old = new - old_center
    projection = (new_centered_to_old @ old_basis) @ old_basis.T
    novelty = torch.linalg.vector_norm(new_centered_to_old - projection) / torch.linalg.vector_norm(
        new_centered_to_old
    ).clamp_min(1e-12)
    cross = old_basis.T @ new_basis
    angle_s = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(angle_s))
    return {
        "effective_rank_all": effective_rank(all_s),
        "effective_rank_first_span": effective_rank(old_s),
        "effective_rank_second_span": effective_rank(new_s),
        "rank_95_first_span": float(old_basis.shape[1]),
        "rank_95_second_span": float(new_basis.shape[1]),
        "novelty_outside_first_span": float(novelty.item()),
        "centroid_distance": float(torch.linalg.vector_norm(old_center - new_center).item()),
        "principal_angle_mean_deg": float(angles.mean().item()),
        "principal_angle_min_deg": float(angles.min().item()),
    }


def add_reference_drift_metrics(
    *,
    reference_states: torch.Tensor,
    states: torch.Tensor,
    item: dict[str, float],
) -> None:
    aligned, drift = procrustes_align(states, reference_states)
    item.update(
        {
            "drift_to_reference_mean": drift["aligned_drift_mean"],
            "drift_to_reference_max": drift["aligned_drift_max"],
            "drift_to_reference_relative": drift["aligned_drift_relative"],
            "centroid_shift_to_reference": drift["centroid_shift"],
            "cka_to_reference_raw": linear_cka(reference_states, states),
            "cka_to_reference_aligned": linear_cka(reference_states, aligned),
        }
    )


def plot_layer_grid(
    *,
    layer: str,
    points_by_model: dict[str, torch.Tensor],
    segment_mask: torch.Tensor,
    output_path: Path,
    plot_columns: int,
) -> None:
    positive_int("plot_columns", plot_columns)
    labels = list(points_by_model)
    combined = torch.cat([points_by_model[label] for label in labels], dim=0)
    projected = pca_2d(combined)
    rows = math.ceil(len(labels) / plot_columns)
    fig, axes = plt.subplots(rows, plot_columns, figsize=(6.0 * plot_columns, 5.0 * rows), squeeze=False)
    offset = 0
    for index, label in enumerate(labels):
        axis = axes[index // plot_columns][index % plot_columns]
        points = points_by_model[label]
        model_projected = projected[offset : offset + points.shape[0]]
        offset += points.shape[0]
        if segment_mask.shape[0] != model_projected.shape[0]:
            raise ValueError(
                f"segment mask length {segment_mask.shape[0]} does not match projected points {model_projected.shape[0]}."
            )
        first_span = model_projected[segment_mask]
        second_span = model_projected[~segment_mask]
        axis.scatter(first_span[:, 0], first_span[:, 1], s=8, alpha=0.65, label="first probe span")
        axis.scatter(second_span[:, 0], second_span[:, 1], s=8, alpha=0.65, label="second probe span")
        axis.set_title(f"{label}: {layer}")
        axis.set_xlabel("shared PC1")
        axis.set_ylabel("shared PC2")
        axis.legend(loc="best", fontsize=8)
    for index in range(len(labels), rows * plot_columns):
        axes[index // plot_columns][index % plot_columns].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("At least one --checkpoint LABEL PATH pair is required.")
    labels: set[str] = set()
    for label, raw_path in args.checkpoint:
        if not label:
            raise ValueError("Checkpoint label cannot be empty.")
        if label in labels:
            raise ValueError(f"Duplicate checkpoint label: {label!r}.")
        labels.add(label)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist for {label}: {path}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer_path does not exist: {args.tokenizer_path}")
    if not args.chunks_path.exists():
        raise FileNotFoundError(f"chunks_path does not exist: {args.chunks_path}")
    nonnegative_int("word_start", args.word_start)
    positive_int("word_count", args.word_count)
    positive_int("split_word_count", args.split_word_count)
    if args.split_word_count >= args.word_count:
        raise ValueError(f"split_word_count must be smaller than word_count, got {args.split_word_count}.")
    positive_int("stride", args.stride)
    positive_int("max_windows", args.max_windows)
    positive_int("plot_columns", args.plot_columns)
    bounded_float("variance_threshold", args.variance_threshold, 0.0, 1.0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.chunk_index < 0 or args.chunk_index >= len(chunks):
        raise ValueError(f"chunk_index={args.chunk_index} outside chunk count {len(chunks)}.")
    full_text = word_span(str(chunks[args.chunk_index]["text"]), args.word_start, args.word_count)
    split_text = word_span(str(chunks[args.chunk_index]["text"]), args.word_start, args.split_word_count)
    token_ids = tokenizer.encode(full_text).ids
    split_token_count = len(tokenizer.encode(split_text).ids)
    if split_token_count <= 0 or split_token_count >= len(token_ids):
        raise ValueError(f"Invalid split token count {split_token_count} for total token count {len(token_ids)}.")
    checkpoint_specs = [(label, Path(raw_path)) for label, raw_path in args.checkpoint]
    first_model, first_checkpoint = load_checkpoint(checkpoint_specs[0][1], device)
    models: dict[str, GCONativeTransformer] = {checkpoint_specs[0][0]: first_model}
    checkpoint_paths: dict[str, str] = {checkpoint_specs[0][0]: str(checkpoint_specs[0][1])}
    for label, path in checkpoint_specs[1:]:
        model, checkpoint = load_checkpoint(path, device)
        if checkpoint["model_config"] != first_checkpoint["model_config"]:
            raise ValueError(
                "Checkpoint model specs differ. Capacity frontier comparison requires identical model_config. "
                f"First={first_checkpoint['model_config']} {label}={checkpoint['model_config']}"
            )
        models[label] = model
        checkpoint_paths[label] = str(path)
    seq_len = int(first_checkpoint["model_config"]["max_seq_len"])
    windows, source_positions = build_windows(token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows)
    segment_mask = source_positions.reshape(-1) < split_token_count
    if int(segment_mask.to(dtype=torch.long).sum().item()) <= 1 or int((~segment_mask).to(dtype=torch.long).sum().item()) <= 1:
        raise RuntimeError("The selected probe did not produce enough first-span and second-span samples.")
    states_by_model = {
        label: {
            layer: value.reshape(-1, value.shape[-1]).to(dtype=torch.float32)
            for layer, value in collect_states(model, windows, device).items()
        }
        for label, model in models.items()
    }
    layers = list(next(iter(states_by_model.values())).keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    plots: dict[str, str] = {}
    reference_label = checkpoint_specs[0][0]
    for layer in layers:
        points_by_model = {label: states[layer] for label, states in states_by_model.items()}
        metrics[layer] = {}
        reference_states = points_by_model[reference_label]
        for label, points in points_by_model.items():
            item = geometry_metrics(points, segment_mask, args.variance_threshold)
            add_reference_drift_metrics(reference_states=reference_states, states=points, item=item)
            metrics[layer][label] = item
        plot_path = args.output_dir / f"capacity_frontier_{layer}.png"
        plot_layer_grid(
            layer=layer,
            points_by_model=points_by_model,
            segment_mask=segment_mask,
            output_path=plot_path,
            plot_columns=args.plot_columns,
        )
        plots[layer] = str(plot_path)
    result = {
        "question": "How does a same-spec tiny transformer re-represent a fixed probe as training word count increases?",
        "checkpoints": checkpoint_paths,
        "reference_label": reference_label,
        "model_config": first_checkpoint["model_config"],
        "source": {
            "chunks_path": str(args.chunks_path),
            "chunk_index": args.chunk_index,
            "word_start": args.word_start,
            "word_count": args.word_count,
            "split_word_count": args.split_word_count,
            "token_count": len(token_ids),
            "split_token_count": split_token_count,
            "window_count": int(windows.shape[0]),
            "seq_len": seq_len,
            "stride": args.stride,
        },
        "metrics": metrics,
        "plots": plots,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("TINY CAPACITY FRONTIER GEOMETRY")
    print("=" * 112)
    print(
        f"models={list(models)} reference={reference_label} windows={windows.shape[0]} "
        f"seq_len={seq_len} tokens={len(token_ids)} split_tokens={split_token_count}"
    )
    for layer in layers:
        print(f"layer={layer} plot={plots[layer]}")
        print("  label rank novelty angle drift_rel cka centroid")
        for label in models:
            item = metrics[layer][label]
            print(
                "  {label} {rank:.2f} {novelty:.4f} {angle:.2f} {drift:.4f} {cka:.4f} {centroid:.4f}".format(
                    label=label,
                    rank=item["effective_rank_all"],
                    novelty=item["novelty_outside_first_span"],
                    angle=item["principal_angle_mean_deg"],
                    drift=item["drift_to_reference_relative"],
                    cka=item["cka_to_reference_aligned"],
                    centroid=item["centroid_distance"],
                )
            )
    print(f"wrote_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs=2, action="append", metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--word-start", type=int, default=0)
    parser.add_argument("--word-count", type=int, required=True)
    parser.add_argument("--split-word-count", type=int, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--plot-columns", type=int, default=2)
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
