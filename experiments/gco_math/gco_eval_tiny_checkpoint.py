"""Evaluate tiny GCO transformer checkpoints on a fixed real-book word span.

This script does not train or fine-tune. It loads one or more same-family tiny
checkpoints, builds next-token LM windows from a selected word span, and reports
loss, token accuracy, and target-margin metrics for behavior comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_native_scratch_transformer import GCONativeTransformer, NativeGCOConfig
from experiments.gco_math.gco_prepare_tiny_cl_base import build_lm_windows, evaluate_model, load_chunks


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def finite_float(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")


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
            raise FileNotFoundError(f"Checkpoint does not exist for label {label!r}: {path}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer_path does not exist: {args.tokenizer_path}")
    if not args.chunks_path.exists():
        raise FileNotFoundError(f"chunks_path does not exist: {args.chunks_path}")
    nonnegative_int("word_start", args.word_start)
    positive_int("word_count", args.word_count)
    positive_int("stride", args.stride)
    positive_int("max_windows", args.max_windows)
    positive_int("batch_size", args.batch_size)
    finite_float("loss_warning_threshold", args.loss_warning_threshold)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    if args.chunk_index < 0 or args.chunk_index >= len(chunks):
        raise ValueError(f"chunk_index={args.chunk_index} outside chunk count {len(chunks)}.")
    text = word_span(str(chunks[args.chunk_index]["text"]), args.word_start, args.word_count)
    token_ids = tokenizer.encode(text).ids
    checkpoints = [(label, Path(raw_path)) for label, raw_path in args.checkpoint]
    first_model, first_checkpoint = load_checkpoint(checkpoints[0][1], device)
    seq_len = int(first_checkpoint["model_config"]["max_seq_len"])
    inputs, targets = build_lm_windows(token_ids, seq_len=seq_len, stride=args.stride, max_windows=args.max_windows)
    metrics: dict[str, dict[str, float]] = {}
    checkpoint_paths: dict[str, str] = {}
    for index, (label, path) in enumerate(checkpoints):
        if index == 0:
            model = first_model
            checkpoint = first_checkpoint
        else:
            model, checkpoint = load_checkpoint(path, device)
        if checkpoint["model_config"] != first_checkpoint["model_config"]:
            raise ValueError(
                "Checkpoint model specs differ. Behavior comparison requires identical model_config. "
                f"First={first_checkpoint['model_config']} {label}={checkpoint['model_config']}"
            )
        item = evaluate_model(model, inputs, targets, batch_size=args.batch_size, device=device)
        metrics[label] = item
        checkpoint_paths[label] = str(path)
    result = {
        "question": "How well do fixed tiny checkpoints preserve behavior on a selected real-book span?",
        "checkpoints": checkpoint_paths,
        "model_config": first_checkpoint["model_config"],
        "source": {
            "chunks_path": str(args.chunks_path),
            "chunk_index": args.chunk_index,
            "word_start": args.word_start,
            "word_count": args.word_count,
            "token_count": len(token_ids),
            "window_count": int(inputs.shape[0]),
            "seq_len": seq_len,
            "stride": args.stride,
        },
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("TINY CHECKPOINT BEHAVIOR EVAL")
    print("=" * 112)
    print(
        f"span=words[{args.word_start}:{args.word_start + args.word_count}] "
        f"tokens={len(token_ids)} windows={inputs.shape[0]} seq_len={seq_len}"
    )
    print("label loss acc margin_mean margin_min")
    print("-" * 112)
    for label, item in metrics.items():
        warning = " !" if item["loss"] > args.loss_warning_threshold else ""
        print(
            "{label} {loss:.6f} {acc:.4f} {margin_mean:+.4f} {margin_min:+.4f}{warning}".format(
                label=label,
                loss=item["loss"],
                acc=item["token_accuracy"],
                margin_mean=item["target_margin_mean"],
                margin_min=item["target_margin_min"],
                warning=warning,
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
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loss-warning-threshold", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
