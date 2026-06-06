"""Visualize residual geometry for tiny native GCO transformer checkpoints.

This script does not train or fine-tune. It loads one or two checkpoints,
runs the same real-book text through them, captures residual states, projects
the states with PCA, writes PNG plots, and stores layer-wise geometry metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
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


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def bounded_float(name: str, value: float, low: float, high: float) -> None:
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}.")


def word_span(text: str, start: int, count: int) -> str:
    nonnegative_int("word_start", start)
    positive_int("word_count", count)
    words = text.split()
    end = start + count
    if end > len(words):
        raise ValueError(f"Requested word span [{start}, {end}) but text has only {len(words)} words.")
    return " ".join(words[start:end])


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


def build_windows(token_ids: list[int], *, seq_len: int, stride: int, max_windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    positive_int("seq_len", seq_len)
    positive_int("stride", stride)
    positive_int("max_windows", max_windows)
    if len(token_ids) < seq_len:
        raise ValueError(f"Need at least {seq_len} tokens, got {len(token_ids)}.")
    windows: list[list[int]] = []
    positions: list[list[int]] = []
    for start in range(0, len(token_ids) - seq_len + 1, stride):
        windows.append(token_ids[start : start + seq_len])
        positions.append(list(range(start, start + seq_len)))
        if len(windows) >= max_windows:
            break
    if not windows:
        raise RuntimeError("No geometry windows were built.")
    return torch.tensor(windows, dtype=torch.long), torch.tensor(positions, dtype=torch.long)


@torch.no_grad()
def collect_states(model: GCONativeTransformer, tokens: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    tokens = tokens.to(device)
    batch, seq_len = tokens.shape
    positions = torch.arange(seq_len, device=device, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
    h = model.token_embedding(tokens) + model.position_embedding(positions)
    states: dict[str, torch.Tensor] = {"embed": h.detach().cpu()}
    for index, block in enumerate(model.blocks):
        h = block(h)
        states[f"block_{index}"] = h.detach().cpu()
    final = model.ln_f(h)
    states["final"] = final.detach().cpu()
    return states


def centered_svd_basis(x: torch.Tensor, variance_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"Expected matrix for SVD basis, got {x.shape}.")
    if x.shape[0] < 2:
        raise ValueError(f"Need at least two samples for SVD basis, got {x.shape[0]}.")
    centered = x - x.mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    if singular_values.numel() <= 0:
        raise RuntimeError("SVD produced no singular values.")
    energy = singular_values.square()
    total = energy.sum().clamp_min(1e-12)
    cumulative = torch.cumsum(energy, dim=0) / total
    rank = int((cumulative < variance_threshold).to(dtype=torch.long).sum().item()) + 1
    rank = min(rank, vh.shape[0])
    return vh[:rank].T.contiguous(), singular_values


def effective_rank(singular_values: torch.Tensor) -> float:
    values = singular_values.abs()
    probs = values / values.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy).item())


def geometry_metrics(states: torch.Tensor, segment_mask: torch.Tensor, variance_threshold: float) -> dict[str, float]:
    if states.ndim != 2:
        raise ValueError(f"states must be [samples, dim], got {states.shape}.")
    if segment_mask.shape != (states.shape[0],):
        raise ValueError(f"segment mask shape {segment_mask.shape} does not match sample count {states.shape[0]}.")
    old = states[segment_mask]
    new = states[~segment_mask]
    if old.shape[0] < 2 or new.shape[0] < 2:
        raise ValueError(f"Need at least two old and new samples, got old={old.shape[0]}, new={new.shape[0]}.")
    old_basis, old_s = centered_svd_basis(old, variance_threshold)
    new_basis, new_s = centered_svd_basis(new, variance_threshold)
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
    all_centered = states - states.mean(dim=0, keepdim=True)
    _u, all_s, _vh = torch.linalg.svd(all_centered, full_matrices=False)
    return {
        "sample_count": float(states.shape[0]),
        "old_sample_count": float(old.shape[0]),
        "new_sample_count": float(new.shape[0]),
        "old_rank_95": float(old_basis.shape[1]),
        "new_rank_95": float(new_basis.shape[1]),
        "effective_rank_all": effective_rank(all_s),
        "effective_rank_old": effective_rank(old_s),
        "effective_rank_new": effective_rank(new_s),
        "new_novelty_outside_old_span": float(novelty.item()),
        "old_new_centroid_distance": float(torch.linalg.vector_norm(old_center - new_center).item()),
        "principal_angle_min_deg": float(angles.min().item()),
        "principal_angle_mean_deg": float(angles.mean().item()),
        "principal_cos_max": float(angle_s.max().item()),
        "principal_cos_mean": float(angle_s.mean().item()),
    }


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"PCA input must be [samples, dim], got {x.shape}.")
    centered = x - x.mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < 2:
        raise ValueError(f"Need at least two PCA components, got {vh.shape[0]}.")
    return centered @ vh[:2].T


def plot_layer(
    *,
    layer: str,
    points_by_model: dict[str, torch.Tensor],
    segment_mask: torch.Tensor,
    output_path: Path,
) -> None:
    combined = torch.cat(list(points_by_model.values()), dim=0)
    projected = pca_2d(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(points_by_model), figsize=(6.0 * len(points_by_model), 5.0), squeeze=False)
    offset = 0
    for axis, (label, points) in zip(axes[0], points_by_model.items(), strict=True):
        model_projected = projected[offset : offset + points.shape[0]]
        offset += points.shape[0]
        if segment_mask.shape[0] != model_projected.shape[0]:
            raise ValueError(
                f"segment mask length {segment_mask.shape[0]} does not match projected points {model_projected.shape[0]}."
            )
        old_points = model_projected[segment_mask]
        new_points = model_projected[~segment_mask]
        axis.scatter(old_points[:, 0], old_points[:, 1], s=8, alpha=0.65, label="first span")
        axis.scatter(new_points[:, 0], new_points[:, 1], s=8, alpha=0.65, label="second span")
        axis.set_title(f"{label}: {layer}")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint_a.exists():
        raise FileNotFoundError(f"checkpoint_a does not exist: {args.checkpoint_a}")
    if args.checkpoint_b is not None and not args.checkpoint_b.exists():
        raise FileNotFoundError(f"checkpoint_b does not exist: {args.checkpoint_b}")
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
        raise ValueError(
            f"Invalid split token count {split_token_count} for total token count {len(token_ids)}."
        )
    model_a, checkpoint_a = load_checkpoint(args.checkpoint_a, device)
    models: dict[str, GCONativeTransformer] = {args.label_a: model_a}
    checkpoints: dict[str, dict[str, Any]] = {args.label_a: checkpoint_a}
    if args.checkpoint_b is not None:
        model_b, checkpoint_b = load_checkpoint(args.checkpoint_b, device)
        if checkpoint_b["model_config"] != checkpoint_a["model_config"]:
            raise ValueError(
                "Checkpoint model specs differ. Geometry comparison requires identical model_config. "
                f"A={checkpoint_a['model_config']} B={checkpoint_b['model_config']}"
            )
        models[args.label_b] = model_b
        checkpoints[args.label_b] = checkpoint_b
    seq_len = int(checkpoint_a["model_config"]["max_seq_len"])
    windows, source_positions = build_windows(token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows)
    flat_source_positions = source_positions.reshape(-1)
    segment_mask = flat_source_positions < split_token_count
    if int(segment_mask.to(dtype=torch.long).sum().item()) <= 1 or int((~segment_mask).to(dtype=torch.long).sum().item()) <= 1:
        raise RuntimeError("The selected windows did not produce enough first-span and second-span samples.")
    states_by_model = {
        label: {layer: value.reshape(-1, value.shape[-1]).to(dtype=torch.float32) for layer, value in collect_states(model, windows, device).items()}
        for label, model in models.items()
    }
    layers = list(next(iter(states_by_model.values())).keys())
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    plot_paths: dict[str, str] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        points_by_model = {label: states[layer] for label, states in states_by_model.items()}
        plot_path = args.output_dir / f"geometry_{layer}.png"
        plot_layer(layer=layer, points_by_model=points_by_model, segment_mask=segment_mask, output_path=plot_path)
        plot_paths[layer] = str(plot_path)
        metrics[layer] = {
            label: geometry_metrics(points, segment_mask, args.variance_threshold)
            for label, points in points_by_model.items()
        }
    result = {
        "question": "What does the tiny transformer residual geometry look like before any CL fine-tuning?",
        "checkpoints": {label: str(path) for label, path in [(args.label_a, args.checkpoint_a), (args.label_b, args.checkpoint_b)] if path is not None},
        "model_config": checkpoint_a["model_config"],
        "native_gco_config_a": checkpoint_a.get("native_gco_config"),
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
        "plots": plot_paths,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("TINY GEOMETRY VISUALIZATION")
    print("=" * 112)
    print(
        f"models={list(models)} windows={windows.shape[0]} seq_len={seq_len} "
        f"tokens={len(token_ids)} split_tokens={split_token_count}"
    )
    for layer in layers:
        print(f"layer={layer} plot={plot_paths[layer]}")
        for label in models:
            item = metrics[layer][label]
            print(
                "  {label}: rank={rank:.2f} novelty={novelty:.4f} angle={angle:.2f}deg "
                "centroid={centroid:.4f}".format(
                    label=label,
                    rank=item["effective_rank_all"],
                    novelty=item["new_novelty_outside_old_span"],
                    angle=item["principal_angle_mean_deg"],
                    centroid=item["old_new_centroid_distance"],
                )
            )
    print(f"wrote_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--label-a", type=str, default="model_a")
    parser.add_argument("--label-b", type=str, default="model_b")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--word-start", type=int, default=0)
    parser.add_argument("--word-count", type=int, default=200)
    parser.add_argument("--split-word-count", type=int, default=100)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis/tiny-geometry"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/tiny-geometry/geometry.json"))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
