"""Visualize matched residual-geometry drift between two tiny GCO checkpoints.

This script does not train or fine-tune. It loads two same-spec checkpoints,
runs the exact same real-book token windows through both, aligns checkpoint B
to checkpoint A with an orthogonal Procrustes map per layer, and writes drift
plots plus layer-wise metrics.
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


def centered_svd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"Expected matrix, got {x.shape}.")
    if x.shape[0] < 2:
        raise ValueError(f"Need at least two samples, got {x.shape[0]}.")
    centered = x - x.mean(dim=0, keepdim=True)
    return torch.linalg.svd(centered, full_matrices=False)


def effective_rank(singular_values: torch.Tensor) -> float:
    values = singular_values.abs()
    probs = values / values.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy).item())


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"PCA input must be [samples, dim], got {x.shape}.")
    centered = x - x.mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < 2:
        raise ValueError(f"Need at least two PCA components, got {vh.shape[0]}.")
    return centered @ vh[:2].T


def procrustes_align(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if source.shape != target.shape:
        raise ValueError(f"source and target shapes differ: source={source.shape}, target={target.shape}.")
    source_center = source.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)
    source_centered = source - source_center
    target_centered = target - target_center
    u, singular_values, vh = torch.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
    rotation = u @ vh
    aligned = source_centered @ rotation + target_center
    raw_error = torch.linalg.vector_norm(source - target, dim=1)
    aligned_error = torch.linalg.vector_norm(aligned - target, dim=1)
    target_norm = torch.linalg.vector_norm(target_centered, dim=1).mean().clamp_min(1e-12)
    return aligned, {
        "raw_drift_mean": float(raw_error.mean().item()),
        "raw_drift_max": float(raw_error.max().item()),
        "aligned_drift_mean": float(aligned_error.mean().item()),
        "aligned_drift_max": float(aligned_error.max().item()),
        "aligned_drift_relative": float((aligned_error.mean() / target_norm).item()),
        "centroid_shift": float(torch.linalg.vector_norm(source_center - target_center).item()),
        "procrustes_singular_sum": float(singular_values.sum().item()),
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA sample counts differ: x={x.shape[0]}, y={y.shape[0]}.")
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x_centered.T @ y_centered).square()
    x_norm = torch.linalg.matrix_norm(x_centered.T @ x_centered)
    y_norm = torch.linalg.matrix_norm(y_centered.T @ y_centered)
    return float((cross / (x_norm * y_norm).clamp_min(1e-12)).item())


def geometry_metrics(target: torch.Tensor, source: torch.Tensor, aligned_source: torch.Tensor) -> dict[str, float]:
    _u_a, s_a, vh_a = centered_svd(target)
    _u_b, s_b, vh_b = centered_svd(source)
    cross = vh_a.T @ vh_b
    angle_s = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(angle_s))
    drift = torch.linalg.vector_norm(aligned_source - target, dim=1)
    target_norm = torch.linalg.vector_norm(target - target.mean(dim=0, keepdim=True), dim=1).mean().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        aligned_source - aligned_source.mean(dim=0, keepdim=True),
        target - target.mean(dim=0, keepdim=True),
        dim=1,
    )
    return {
        "effective_rank_a": effective_rank(s_a),
        "effective_rank_b": effective_rank(s_b),
        "rank_delta_b_minus_a": effective_rank(s_b) - effective_rank(s_a),
        "principal_angle_mean_deg": float(angles.mean().item()),
        "principal_angle_min_deg": float(angles.min().item()),
        "principal_cos_mean": float(angle_s.mean().item()),
        "linear_cka_raw": linear_cka(target, source),
        "linear_cka_aligned": linear_cka(target, aligned_source),
        "aligned_cosine_mean": float(cosine.mean().item()),
        "aligned_cosine_min": float(cosine.min().item()),
        "matched_drift_mean": float(drift.mean().item()),
        "matched_drift_p95": float(torch.quantile(drift, 0.95).item()),
        "matched_drift_max": float(drift.max().item()),
        "matched_drift_relative": float((drift.mean() / target_norm).item()),
    }


def plot_drift_layer(
    *,
    layer: str,
    label_a: str,
    label_b: str,
    states_a: torch.Tensor,
    aligned_b: torch.Tensor,
    source_positions: torch.Tensor,
    arrow_count: int,
    output_path: Path,
) -> None:
    positive_int("arrow_count", arrow_count)
    if states_a.shape != aligned_b.shape:
        raise ValueError(f"states_a and aligned_b shapes differ: {states_a.shape} vs {aligned_b.shape}.")
    projected = pca_2d(torch.cat([states_a, aligned_b], dim=0))
    a_2d = projected[: states_a.shape[0]]
    b_2d = projected[states_a.shape[0] :]
    positions = source_positions.to(dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 6.0))
    scatter_a = axis.scatter(
        a_2d[:, 0],
        a_2d[:, 1],
        c=positions,
        s=8,
        alpha=0.72,
        cmap="viridis",
        label=label_a,
    )
    axis.scatter(
        b_2d[:, 0],
        b_2d[:, 1],
        c=positions,
        s=8,
        alpha=0.42,
        cmap="magma",
        marker="x",
        label=f"{label_b} aligned",
    )
    sample_count = states_a.shape[0]
    count = min(arrow_count, sample_count)
    arrow_indices = torch.linspace(0, sample_count - 1, count, dtype=torch.long)
    dx = b_2d[arrow_indices, 0] - a_2d[arrow_indices, 0]
    dy = b_2d[arrow_indices, 1] - a_2d[arrow_indices, 1]
    axis.quiver(
        a_2d[arrow_indices, 0],
        a_2d[arrow_indices, 1],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
        alpha=0.38,
        color="black",
    )
    axis.set_title(f"Matched residual drift: {layer}")
    axis.set_xlabel("shared PCA 1")
    axis.set_ylabel("shared PCA 2")
    axis.legend(loc="best", fontsize=8)
    colorbar = fig.colorbar(scatter_a, ax=axis)
    colorbar.set_label("source token position")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint_a.exists():
        raise FileNotFoundError(f"checkpoint_a does not exist: {args.checkpoint_a}")
    if not args.checkpoint_b.exists():
        raise FileNotFoundError(f"checkpoint_b does not exist: {args.checkpoint_b}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer_path does not exist: {args.tokenizer_path}")
    if not args.chunks_path.exists():
        raise FileNotFoundError(f"chunks_path does not exist: {args.chunks_path}")
    nonnegative_int("word_start", args.word_start)
    positive_int("word_count", args.word_count)
    positive_int("stride", args.stride)
    positive_int("max_windows", args.max_windows)
    positive_int("arrow_count", args.arrow_count)
    bounded_float("variance_threshold", args.variance_threshold, 0.0, 1.0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.chunk_index < 0 or args.chunk_index >= len(chunks):
        raise ValueError(f"chunk_index={args.chunk_index} outside chunk count {len(chunks)}.")
    full_text = word_span(str(chunks[args.chunk_index]["text"]), args.word_start, args.word_count)
    token_ids = tokenizer.encode(full_text).ids
    model_a, checkpoint_a = load_checkpoint(args.checkpoint_a, device)
    model_b, checkpoint_b = load_checkpoint(args.checkpoint_b, device)
    if checkpoint_a["model_config"] != checkpoint_b["model_config"]:
        raise ValueError(
            "Checkpoint model specs differ. Geometry drift comparison requires identical model_config. "
            f"A={checkpoint_a['model_config']} B={checkpoint_b['model_config']}"
        )
    seq_len = int(checkpoint_a["model_config"]["max_seq_len"])
    windows, source_positions = build_windows(token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows)
    flat_positions = source_positions.reshape(-1)
    states_a = {
        layer: value.reshape(-1, value.shape[-1]).to(dtype=torch.float32)
        for layer, value in collect_states(model_a, windows, device).items()
    }
    states_b = {
        layer: value.reshape(-1, value.shape[-1]).to(dtype=torch.float32)
        for layer, value in collect_states(model_b, windows, device).items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}
    plots: dict[str, str] = {}
    for layer in states_a:
        aligned_b, align_metrics = procrustes_align(states_b[layer], states_a[layer])
        layer_metrics = geometry_metrics(states_a[layer], states_b[layer], aligned_b)
        layer_metrics.update(align_metrics)
        metrics[layer] = layer_metrics
        plot_path = args.output_dir / f"drift_{layer}.png"
        plot_drift_layer(
            layer=layer,
            label_a=args.label_a,
            label_b=args.label_b,
            states_a=states_a[layer],
            aligned_b=aligned_b,
            source_positions=flat_positions,
            arrow_count=args.arrow_count,
            output_path=plot_path,
        )
        plots[layer] = str(plot_path)
    result = {
        "question": "How does successful extra-data training rebase the exact same first-span residual geometry?",
        "checkpoints": {
            args.label_a: str(args.checkpoint_a),
            args.label_b: str(args.checkpoint_b),
        },
        "model_config": checkpoint_a["model_config"],
        "source": {
            "chunks_path": str(args.chunks_path),
            "chunk_index": args.chunk_index,
            "word_start": args.word_start,
            "word_count": args.word_count,
            "token_count": len(token_ids),
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
    print("TINY GEOMETRY DRIFT VISUALIZATION")
    print("=" * 112)
    print(
        f"models={args.label_a}->{args.label_b} windows={windows.shape[0]} "
        f"seq_len={seq_len} tokens={len(token_ids)}"
    )
    for layer, item in metrics.items():
        print(f"layer={layer} plot={plots[layer]}")
        print(
            "  rank {ra:.2f}->{rb:.2f} delta={rd:+.2f} "
            "drift={dm:.4f}/{dp:.4f}/{dx:.4f} rel={rel:.4f} "
            "cka={cka:.4f} cos={cos:.4f} angle={angle:.2f} centroid={centroid:.4f}".format(
                ra=item["effective_rank_a"],
                rb=item["effective_rank_b"],
                rd=item["rank_delta_b_minus_a"],
                dm=item["matched_drift_mean"],
                dp=item["matched_drift_p95"],
                dx=item["matched_drift_max"],
                rel=item["matched_drift_relative"],
                cka=item["linear_cka_aligned"],
                cos=item["aligned_cosine_mean"],
                angle=item["principal_angle_mean_deg"],
                centroid=item["centroid_shift"],
            )
        )
    print(f"wrote_json={args.output_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, required=True)
    parser.add_argument("--label-b", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--word-start", type=int, default=0)
    parser.add_argument("--word-count", type=int, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--arrow-count", type=int, default=220)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
