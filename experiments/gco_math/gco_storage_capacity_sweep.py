"""Sweep model/data capacity for miniature transformer storage.

This experiment estimates a storage frontier before adding continual-learning
control. It trains the same transformer family across model specs, corpus
types, and word budgets, then records:

    - words, tokens, token/word ratio
    - parameter count and token pressure per parameter
    - training fit loss and token accuracy
    - gradient/update speed
    - residual effective rank
    - weight effective rank

The goal is to build empirical data for mathematical capacity equations rather
than assume a universal limit.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_native_scratch_transformer import GCONativeTransformer
from experiments.gco_math.gco_prepare_tiny_cl_base import native_config
from experiments.gco_math.gco_tiny_cl_behavior_path import (
    set_only_native_weights_trainable,
    trainable_weight_parameters,
)
from experiments.gco_math.gco_visualize_tiny_geometry_drift import collect_states, effective_rank, linear_cka


@dataclass(frozen=True)
class ModelSpec:
    label: str
    d_model: int
    layers: int
    heads: int
    d_ff: int


@dataclass(frozen=True)
class CorpusSample:
    kind: str
    text: str
    requested_words: int
    actual_words: int
    chars: int
    token_count: int
    unique_token_count: int
    unique_token_ratio: float
    token_per_word: float
    char_per_token: float
    char_per_word: float


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def bounded_float(name: str, value: float, low: float, high: float) -> None:
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}.")


def parse_int_list(text: str, *, name: str) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        positive_int(name, value)
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one integer.")
    return values


def parse_str_list(text: str, *, name: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def parse_specs(text: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 5:
            raise ValueError(
                "Each spec must be label:d_model:layers:heads:d_ff, "
                f"got {item!r}."
            )
        label = parts[0]
        if not label:
            raise ValueError(f"Spec label must not be empty: {item!r}.")
        d_model = int(parts[1])
        layers = int(parts[2])
        heads = int(parts[3])
        d_ff = int(parts[4])
        positive_int("d_model", d_model)
        positive_int("layers", layers)
        positive_int("heads", heads)
        positive_int("d_ff", d_ff)
        if d_model % heads != 0:
            raise ValueError(f"Spec {label!r} has d_model={d_model} not divisible by heads={heads}.")
        specs.append(ModelSpec(label=label, d_model=d_model, layers=layers, heads=heads, d_ff=d_ff))
    if not specs:
        raise ValueError("At least one model spec is required.")
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Spec labels must be unique, got {labels}.")
    return specs


def word_count(text: str) -> int:
    return len(text.split())


def first_words(text: str, count: int) -> str:
    positive_int("count", count)
    words = text.split()
    if len(words) < count:
        raise ValueError(f"Requested {count} words but source has only {len(words)} words.")
    return " ".join(words[:count])


def load_book_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Book path does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Book path is empty: {path}")
    return text


def fact_sentences() -> list[str]:
    colors = [
        "amber",
        "blue",
        "copper",
        "green",
        "ivory",
        "navy",
        "orange",
        "purple",
        "ruby",
        "silver",
    ]
    objects = [
        "key",
        "lantern",
        "map",
        "coin",
        "rope",
        "compass",
        "feather",
        "whistle",
        "shell",
        "ring",
    ]
    places = [
        "tower",
        "tunnel",
        "river",
        "vault",
        "bridge",
        "forest",
        "aviary",
        "station",
        "harbor",
        "garden",
    ]
    sentences: list[str] = []
    for index in range(1000):
        color = colors[index % len(colors)]
        obj = objects[(index * 3) % len(objects)]
        place = places[(index * 7) % len(places)]
        person = f"Person{index:04d}"
        item = f"{color} {obj} {index:04d}"
        sentences.append(f"{person} carries the {item}.")
        sentences.append(f"The {item} opens the {place} gate {index:04d}.")
        sentences.append(f"Because {person} carries the {item}, {person} can open the {place} gate {index:04d}.")
        sentences.append(f"Question: What can {person} open? Answer: the {place} gate {index:04d}.")
    return sentences


def repeat_to_words(sentences: list[str], target_words: int) -> str:
    positive_int("target_words", target_words)
    if not sentences:
        raise ValueError("repeat_to_words requires at least one sentence.")
    rows: list[str] = []
    index = 0
    while word_count(" ".join(rows)) < target_words:
        rows.append(sentences[index % len(sentences)])
        index += 1
    return first_words(" ".join(rows), target_words)


def build_text(kind: str, *, requested_words: int, book_text: str) -> str:
    if kind == "book":
        return first_words(book_text, requested_words)
    if kind == "facts":
        return repeat_to_words(fact_sentences(), requested_words)
    if kind == "mixed":
        book_words = requested_words // 2
        fact_words = requested_words - book_words
        book_part = first_words(book_text, book_words)
        fact_part = repeat_to_words(fact_sentences(), fact_words)
        return f"{book_part} {fact_part}"
    raise ValueError(f"Unknown corpus kind {kind!r}; expected book, facts, or mixed.")


def make_corpus_sample(kind: str, *, requested_words: int, book_text: str, tokenizer: Tokenizer) -> CorpusSample:
    text = build_text(kind, requested_words=requested_words, book_text=book_text)
    tokens = tokenizer.encode(text).ids
    actual_words = word_count(text)
    if actual_words != requested_words:
        raise RuntimeError(f"Corpus builder returned {actual_words} words for requested {requested_words}.")
    if not tokens:
        raise RuntimeError(f"Tokenizer returned no tokens for kind={kind} words={requested_words}.")
    chars = len(text)
    return CorpusSample(
        kind=kind,
        text=text,
        requested_words=requested_words,
        actual_words=actual_words,
        chars=chars,
        token_count=len(tokens),
        unique_token_count=len(set(tokens)),
        unique_token_ratio=float(len(set(tokens))) / float(len(tokens)),
        token_per_word=float(len(tokens)) / float(actual_words),
        char_per_token=float(chars) / float(len(tokens)),
        char_per_word=float(chars) / float(actual_words),
    )


def build_lm_windows(
    token_ids: list[int],
    *,
    seq_len: int,
    stride: int,
    max_windows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_int("seq_len", seq_len)
    positive_int("stride", stride)
    positive_int("max_windows", max_windows)
    if len(token_ids) < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} tokens for LM windows, got {len(token_ids)}.")
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, len(token_ids) - seq_len, stride):
        window = token_ids[start : start + seq_len + 1]
        if len(window) != seq_len + 1:
            raise RuntimeError(f"Window length mismatch at start={start}: {len(window)}.")
        inputs.append(window[:-1])
        targets.append(window[1:])
        if len(inputs) >= max_windows:
            break
    if not inputs:
        raise RuntimeError("No LM windows were built.")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def make_model(
    args: argparse.Namespace,
    spec: ModelSpec,
    *,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> GCONativeTransformer:
    torch.manual_seed(seed)
    cfg = native_config(args)
    return GCONativeTransformer(
        vocab_size=vocab_size,
        d_model=spec.d_model,
        n_layers=spec.layers,
        n_heads=spec.heads,
        d_ff=spec.d_ff,
        max_seq_len=args.max_seq_len,
        cfg=cfg,
    ).to(device)


def make_optimizer(args: argparse.Namespace, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer {args.optimizer!r}.")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def threshold_passed(*, mode: str, loss: float, token_accuracy: float, target_loss: float, target_accuracy: float) -> bool:
    loss_pass = loss <= target_loss
    accuracy_pass = token_accuracy >= target_accuracy
    if mode == "either":
        return loss_pass or accuracy_pass
    if mode == "both":
        return loss_pass and accuracy_pass
    if mode == "loss":
        return loss_pass
    if mode == "accuracy":
        return accuracy_pass
    raise ValueError(f"Unknown threshold mode {mode!r}.")


@torch.no_grad()
def flatten_params(params: list[torch.nn.Parameter]) -> torch.Tensor:
    if not params:
        raise ValueError("flatten_params received no parameters.")
    return torch.cat([parameter.detach().cpu().reshape(-1).to(dtype=torch.float32) for parameter in params])


@torch.no_grad()
def evaluate_lm(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    positive_int("batch_size", batch_size)
    if inputs.shape != targets.shape:
        raise ValueError(f"input/target shape mismatch: {inputs.shape} vs {targets.shape}.")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_windows = 0
    exact_windows = 0
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        batch_targets = targets[start : start + batch_size].to(device)
        logits = model(batch_inputs)
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_targets.reshape(-1), reduction="none")
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(batch_targets)
        total_loss += float(losses.detach().sum().cpu())
        total_tokens += int(batch_targets.numel())
        total_correct += int(correct.detach().sum().cpu())
        total_windows += int(batch_targets.shape[0])
        exact_windows += int(correct.all(dim=1).detach().sum().cpu())
    if total_tokens <= 0 or total_windows <= 0:
        raise RuntimeError("Evaluation saw no tokens/windows.")
    loss = total_loss / float(total_tokens)
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "token_accuracy": float(total_correct) / float(total_tokens),
        "window_exact": float(exact_windows) / float(total_windows),
        "token_count": float(total_tokens),
        "window_count": float(total_windows),
    }


def train_lm(
    *,
    args: argparse.Namespace,
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    seed: int,
    label: str,
) -> list[dict[str, float]]:
    if inputs.shape != targets.shape:
        raise ValueError(f"input/target shape mismatch: {inputs.shape} vs {targets.shape}.")
    set_only_native_weights_trainable(model)
    params = trainable_weight_parameters(model)
    optimizer = make_optimizer(args, params)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(inputs.shape[0], generator=generator)
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        grad_total = 0.0
        grad_max = 0.0
        batches = 0
        pbar = tqdm(range(0, inputs.shape[0], args.batch_size), desc=f"{label} {epoch}/{args.epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            batch_inputs = inputs[indices].to(device)
            batch_targets = targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_targets.reshape(-1))
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip).detach().cpu())
            optimizer.step()
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = int(predictions.eq(batch_targets).detach().sum().cpu())
                token_count = int(batch_targets.numel())
            total_loss += float(loss.detach().cpu()) * float(token_count)
            total_tokens += token_count
            total_correct += correct
            grad_total += grad_norm
            grad_max = max(grad_max, grad_norm)
            batches += 1
            pbar.set_postfix(
                {
                    "loss": f"{float(loss.detach().cpu()):.3g}",
                    "acc": f"{float(correct) / float(token_count):.3f}",
                    "grad": f"{grad_norm:.3g}",
                }
            )
        if batches <= 0 or total_tokens <= 0:
            raise RuntimeError(f"{label} epoch {epoch} produced no batches/tokens.")
        row = {
            "epoch": float(epoch),
            "loss": total_loss / float(total_tokens),
            "token_accuracy": float(total_correct) / float(total_tokens),
            "grad_norm_mean": grad_total / float(batches),
            "grad_norm_max": grad_max,
        }
        trace.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(
                "{} epoch={:4d} loss={:.5f} acc={:.4f} grad={:.4g}/{:.4g}".format(
                    label,
                    epoch,
                    row["loss"],
                    row["token_accuracy"],
                    row["grad_norm_mean"],
                    row["grad_norm_max"],
                )
            )
        if args.early_stop and epoch >= args.min_epochs:
            if threshold_passed(
                mode=args.early_stop_mode,
                loss=row["loss"],
                token_accuracy=row["token_accuracy"],
                target_loss=args.target_loss,
                target_accuracy=args.target_token_accuracy,
            ):
                break
    return trace


def centered_singular_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a matrix, got {matrix.shape}.")
    centered = matrix.to(dtype=torch.float32) - matrix.to(dtype=torch.float32).mean(dim=0, keepdim=True)
    return torch.linalg.svdvals(centered.cpu())


@torch.no_grad()
def weight_rank_report(model: GCONativeTransformer) -> dict[str, Any]:
    rows: list[dict[str, float | str]] = []
    for module in model.gco_modules():
        weight = module.W.detach().cpu().to(dtype=torch.float32)
        if weight.ndim != 2:
            raise RuntimeError(f"Module {module.name} weight is not 2D: {weight.shape}.")
        singular_values = centered_singular_values(weight)
        min_dim = min(weight.shape)
        rank = effective_rank(singular_values)
        rows.append(
            {
                "name": module.name,
                "rows": float(weight.shape[0]),
                "cols": float(weight.shape[1]),
                "effective_rank": rank,
                "rank_fraction": rank / float(min_dim),
                "fro_norm": float(torch.linalg.matrix_norm(weight).item()),
            }
        )
    if not rows:
        raise RuntimeError("No GCO modules found for weight rank report.")
    return {
        "modules": rows,
        "effective_rank_mean": sum(float(row["effective_rank"]) for row in rows) / float(len(rows)),
        "rank_fraction_mean": sum(float(row["rank_fraction"]) for row in rows) / float(len(rows)),
        "rank_fraction_min": min(float(row["rank_fraction"]) for row in rows),
        "rank_fraction_max": max(float(row["rank_fraction"]) for row in rows),
    }


@torch.no_grad()
def residual_rank_report(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    *,
    max_windows: int,
    device: torch.device,
) -> dict[str, Any]:
    positive_int("max_windows", max_windows)
    if inputs.shape[0] < 1:
        raise ValueError("No inputs for residual rank report.")
    probe = inputs[: min(max_windows, inputs.shape[0])].to(device)
    states = collect_states(model, probe, device)
    layers: dict[str, dict[str, float]] = {}
    for layer, value in states.items():
        flat = value.reshape(-1, value.shape[-1]).to(dtype=torch.float32)
        singular_values = centered_singular_values(flat)
        rank = effective_rank(singular_values)
        layers[layer] = {
            "effective_rank": rank,
            "rank_fraction": rank / float(min(flat.shape)),
            "state_norm_mean": float(torch.linalg.vector_norm(flat, dim=1).mean().item()),
            "state_norm_max": float(torch.linalg.vector_norm(flat, dim=1).max().item()),
            "sample_count": float(flat.shape[0]),
        }
    if not layers:
        raise RuntimeError("No residual states found.")
    return {
        "layers": layers,
        "effective_rank_mean": sum(row["effective_rank"] for row in layers.values()) / float(len(layers)),
        "rank_fraction_mean": sum(row["rank_fraction"] for row in layers.values()) / float(len(layers)),
    }


@torch.no_grad()
def collect_flat_probe_states(
    model: GCONativeTransformer,
    inputs: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if inputs.ndim != 2:
        raise ValueError(f"probe inputs must be [windows, seq], got {inputs.shape}.")
    if inputs.shape[0] <= 0:
        raise ValueError("probe inputs are empty.")
    states = collect_states(model, inputs.to(device), device)
    return {
        layer: value.reshape(-1, value.shape[-1]).to(dtype=torch.float32).cpu()
        for layer, value in states.items()
    }


def probe_cka_report(
    *,
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    for layer, ref in sorted(reference.items()):
        if layer not in candidate:
            raise RuntimeError(f"Candidate probe states missing layer {layer!r}.")
        cur = candidate[layer]
        if ref.shape[0] != cur.shape[0]:
            raise RuntimeError(f"Probe sample mismatch for {layer}: ref={ref.shape}, cur={cur.shape}.")
        rows[layer] = {
            "cka_to_reference": linear_cka(ref, cur),
            "reference_effective_rank": effective_rank(centered_singular_values(ref)),
            "candidate_effective_rank": effective_rank(centered_singular_values(cur)),
        }
        rows[layer]["rank_delta_to_reference"] = (
            rows[layer]["candidate_effective_rank"] - rows[layer]["reference_effective_rank"]
        )
    values = [row["cka_to_reference"] for row in rows.values()]
    rank_deltas = [row["rank_delta_to_reference"] for row in rows.values()]
    if not values:
        raise RuntimeError("No probe CKA values were computed.")
    return {
        "layers": rows,
        "cka_to_reference_mean": sum(values) / float(len(values)),
        "rank_delta_to_reference_mean": sum(rank_deltas) / float(len(rank_deltas)),
    }


def model_config_dict(spec: ModelSpec, *, vocab_size: int, max_seq_len: int) -> dict[str, int | str]:
    return {
        "label": spec.label,
        "vocab_size": vocab_size,
        "d_model": spec.d_model,
        "layers": spec.layers,
        "heads": spec.heads,
        "d_ff": spec.d_ff,
        "max_seq_len": max_seq_len,
    }


def save_checkpoint(
    *,
    path: Path,
    model: GCONativeTransformer,
    args: argparse.Namespace,
    spec: ModelSpec,
    vocab_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "native_gco_config": asdict(native_config(args)),
            "model_config": model_config_dict(spec, vocab_size=vocab_size, max_seq_len=args.max_seq_len),
        },
        path,
    )


def run_case(
    *,
    args: argparse.Namespace,
    spec: ModelSpec,
    corpus: CorpusSample,
    tokenizer: Tokenizer,
    device: torch.device,
    case_seed: int,
    probe_inputs: torch.Tensor | None,
) -> dict[str, Any]:
    token_ids = tokenizer.encode(corpus.text).ids
    inputs, targets = build_lm_windows(
        token_ids,
        seq_len=args.max_seq_len,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    if probe_inputs is None:
        probe_inputs = inputs[: min(args.geometry_windows, inputs.shape[0])].clone()
    if probe_inputs.ndim != 2 or probe_inputs.shape[1] != args.max_seq_len:
        raise ValueError(f"probe_inputs must be [windows, {args.max_seq_len}], got {probe_inputs.shape}.")
    model = make_model(args, spec, vocab_size=tokenizer.get_vocab_size(), device=device, seed=case_seed)
    set_only_native_weights_trainable(model)
    params = trainable_weight_parameters(model)
    total_params = parameter_count(model)
    trainable_params = trainable_parameter_count(model)
    initial_params = flatten_params(params)
    label = f"{spec.label}/{corpus.kind}/{corpus.actual_words}w"
    trace = train_lm(
        args=args,
        model=model,
        inputs=inputs,
        targets=targets,
        device=device,
        seed=case_seed + 1,
        label=label,
    )
    final_metrics = evaluate_lm(
        model,
        inputs,
        targets,
        batch_size=args.eval_batch_size,
        device=device,
    )
    final_params = flatten_params(params)
    delta = final_params - initial_params
    initial_norm = torch.linalg.vector_norm(initial_params).clamp_min(1e-12)
    delta_norm = torch.linalg.vector_norm(delta)
    relative_delta = float((delta_norm / initial_norm).item())
    weight_ranks = weight_rank_report(model)
    residual_ranks = residual_rank_report(model, inputs, max_windows=args.geometry_windows, device=device)
    probe_states = collect_flat_probe_states(model, probe_inputs, device=device)
    fit_success_loss = final_metrics["loss"] <= args.success_loss
    fit_success_token_accuracy = final_metrics["token_accuracy"] >= args.success_token_accuracy
    fit_success_strict = fit_success_loss and fit_success_token_accuracy
    fit_success_loose = fit_success_loss or fit_success_token_accuracy
    fit_success = threshold_passed(
        mode=args.success_mode,
        loss=final_metrics["loss"],
        token_accuracy=final_metrics["token_accuracy"],
        target_loss=args.success_loss,
        target_accuracy=args.success_token_accuracy,
    )
    checkpoint_path: str | None = None
    if args.save_checkpoints:
        path = (
            args.checkpoint_dir
            / f"{spec.label}-{corpus.kind}-{corpus.actual_words}w-seed{case_seed}.pt"
        )
        save_checkpoint(path=path, model=model, args=args, spec=spec, vocab_size=tokenizer.get_vocab_size())
        checkpoint_path = str(path)
    if device.type == "mps":
        torch.mps.empty_cache()
    return {
        "spec": asdict(spec),
        "corpus": {
            key: value
            for key, value in asdict(corpus).items()
            if key != "text"
        },
        "window_count": int(inputs.shape[0]),
        "trained_token_positions": int(targets.numel()),
        "total_parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "tokens_per_total_parameter": corpus.token_count / float(total_params),
        "tokens_per_trainable_parameter": corpus.token_count / float(trainable_params),
        "trained_positions_per_trainable_parameter": float(targets.numel()) / float(trainable_params),
        "trace": trace,
        "final": final_metrics,
        "fit_success": bool(fit_success),
        "fit_success_loss": bool(fit_success_loss),
        "fit_success_token_accuracy": bool(fit_success_token_accuracy),
        "fit_success_strict": bool(fit_success_strict),
        "fit_success_loose": bool(fit_success_loose),
        "success_mode": args.success_mode,
        "epochs_ran": len(trace),
        "weight_delta_norm": float(delta_norm.item()),
        "weight_delta_relative": relative_delta,
        "weight_rank": weight_ranks,
        "residual_rank": residual_ranks,
        "_probe_inputs": probe_inputs.cpu(),
        "_probe_states": probe_states,
        "checkpoint": checkpoint_path,
    }


def strip_private_tensors(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def validate_args(args: argparse.Namespace) -> None:
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {args.tokenizer_path}")
    if not args.book_path.exists():
        raise FileNotFoundError(f"Book path does not exist: {args.book_path}")
    parse_specs(args.specs)
    parse_int_list(args.word_counts, name="word_counts")
    for kind in parse_str_list(args.corpus_kinds, name="corpus_kinds"):
        if kind not in {"book", "facts", "mixed"}:
            raise ValueError(f"Unknown corpus kind {kind!r}; expected book, facts, or mixed.")
    positive_int("max_seq_len", args.max_seq_len)
    positive_int("stride", args.stride)
    positive_int("max_windows", args.max_windows)
    positive_int("geometry_windows", args.geometry_windows)
    positive_int("epochs", args.epochs)
    positive_int("min_epochs", args.min_epochs)
    if args.min_epochs > args.epochs:
        raise ValueError(f"min_epochs={args.min_epochs} exceeds epochs={args.epochs}.")
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("print_every", args.print_every)
    positive_float("lr", args.lr)
    nonnegative_float("weight_decay", args.weight_decay)
    nonnegative_float("momentum", args.momentum)
    positive_float("grad_clip", args.grad_clip)
    positive_float("target_loss", args.target_loss)
    bounded_float("target_token_accuracy", args.target_token_accuracy, 0.0, 1.0)
    positive_float("success_loss", args.success_loss)
    bounded_float("success_token_accuracy", args.success_token_accuracy, 0.0, 1.0)
    bounded_float("init_topology", args.init_topology, 0.0, 1.0)
    bounded_float("formation_weight_mix", args.formation_weight_mix, 0.0, 1.0)
    bounded_float("formation_row_mix", args.formation_row_mix, 0.0, 1.0)
    bounded_float("formation_col_mix", args.formation_col_mix, 0.0, 1.0)
    bounded_float("formation_module_mix", args.formation_module_mix, 0.0, 1.0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    book_text = load_book_text(args.book_path)
    specs = parse_specs(args.specs)
    word_counts = parse_int_list(args.word_counts, name="word_counts")
    corpus_kinds = parse_str_list(args.corpus_kinds, name="corpus_kinds")
    print("GCO STORAGE CAPACITY SWEEP")
    print("=" * 120)
    print(f"device={device} vocab={tokenizer.get_vocab_size()} specs={[asdict(spec) for spec in specs]}")
    print(f"word_counts={word_counts} corpus_kinds={corpus_kinds}")
    results: list[dict[str, Any]] = []
    reference_probe_inputs: dict[tuple[str, str], torch.Tensor] = {}
    reference_probe_states: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    reference_word_counts: dict[tuple[str, str], int] = {}
    case_index = 0
    for spec in specs:
        for kind in corpus_kinds:
            for words in word_counts:
                corpus = make_corpus_sample(kind, requested_words=words, book_text=book_text, tokenizer=tokenizer)
                case_seed = args.seed + 1000 * case_index
                print(
                    "\nCASE spec={} kind={} words={} tokens={} tok/word={:.3f}".format(
                        spec.label,
                        kind,
                        corpus.actual_words,
                        corpus.token_count,
                        corpus.token_per_word,
                    )
                )
                group_key = (spec.label, kind)
                row = run_case(
                    args=args,
                    spec=spec,
                    corpus=corpus,
                    tokenizer=tokenizer,
                    device=device,
                    case_seed=case_seed,
                    probe_inputs=reference_probe_inputs.get(group_key),
                )
                if group_key not in reference_probe_inputs:
                    reference_probe_inputs[group_key] = row["_probe_inputs"]
                    reference_probe_states[group_key] = row["_probe_states"]
                    reference_word_counts[group_key] = int(row["corpus"]["actual_words"])
                row["reference_word_count"] = reference_word_counts[group_key]
                row["probe_cka"] = probe_cka_report(
                    reference=reference_probe_states[group_key],
                    candidate=row["_probe_states"],
                )
                results.append(row)
                case_index += 1
    json_results = [strip_private_tensors(row) for row in results]
    summary = {
        "question": "How much token/word data can fixed transformer specs fit before capacity measurements degrade?",
        "hyperparameters": {
            "specs": [asdict(spec) for spec in specs],
            "word_counts": word_counts,
            "corpus_kinds": corpus_kinds,
            "max_seq_len": args.max_seq_len,
            "stride": args.stride,
            "max_windows": args.max_windows,
            "geometry_windows": args.geometry_windows,
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "early_stop": args.early_stop,
            "early_stop_mode": args.early_stop_mode,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "success_mode": args.success_mode,
            "success_loss": args.success_loss,
            "success_token_accuracy": args.success_token_accuracy,
        },
        "results": json_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nSTORAGE FRONTIER SUMMARY")
    print("=" * 120)
    columns = [
        ("spec", 8),
        ("kind", 8),
        ("words", 8),
        ("tokens", 8),
        ("tok/w", 8),
        ("uniq/tok", 8),
        ("tok/param", 10),
        ("pos/param", 10),
        ("loss", 10),
        ("ppl", 10),
        ("acc", 8),
        ("winEx", 10),
        ("dWrel", 10),
        ("resRank", 8),
        ("ckaRef", 8),
        ("strict", 8),
        ("loose", 8),
        ("fit", 8),
    ]
    print(" ".join(f"{name:>{width}}" for name, width in columns))
    for row in json_results:
        values = [
            str(row["spec"]["label"]),
            str(row["corpus"]["kind"]),
            f"{int(row['corpus']['actual_words'])}",
            f"{int(row['corpus']['token_count'])}",
            f"{float(row['corpus']['token_per_word']):.3f}",
            f"{float(row['corpus']['unique_token_ratio']):.3f}",
            f"{float(row['tokens_per_trainable_parameter']):.5f}",
            f"{float(row['trained_positions_per_trainable_parameter']):.5f}",
            f"{float(row['final']['loss']):.4f}",
            f"{float(row['final']['perplexity']):.3f}",
            f"{float(row['final']['token_accuracy']):.4f}",
            f"{float(row['final']['window_exact']):.4f}",
            f"{float(row['weight_delta_relative']):.4f}",
            f"{float(row['residual_rank']['effective_rank_mean']):.3f}",
            f"{float(row['probe_cka']['cka_to_reference_mean']):.4f}",
            "yes" if row["fit_success_strict"] else "no",
            "yes" if row["fit_success_loose"] else "no",
            "yes" if row["fit_success"] else "no",
        ]
        print(" ".join(f"{value:>{width}}" for value, (_, width) in zip(values, columns, strict=True)))
    print(f"wrote_json={args.output_json}")
    if args.save_checkpoints:
        print(f"wrote_checkpoints={args.checkpoint_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--book-path", type=Path, default=Path("data/real_book/book.txt"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-storage-capacity-sweep.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints/gco-storage-capacity-sweep"))
    parser.add_argument("--specs", type=str, default="tiny:128:2:4:256")
    parser.add_argument("--word-counts", type=str, default="1000,2000,3000,5000")
    parser.add_argument("--corpus-kinds", type=str, default="book,facts,mixed")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--geometry-windows", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--target-loss", type=float, default=0.03)
    parser.add_argument("--target-token-accuracy", type=float, default=0.99)
    parser.add_argument("--early-stop-mode", choices=["either", "both", "loss", "accuracy"], default="either")
    parser.add_argument("--success-loss", type=float, default=0.05)
    parser.add_argument("--success-token-accuracy", type=float, default=0.98)
    parser.add_argument("--success-mode", choices=["either", "both", "loss", "accuracy"], default="both")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--init-topology", type=float, default=1.0)
    parser.add_argument("--formation-weight-mix", type=float, default=1.0)
    parser.add_argument("--formation-row-mix", type=float, default=1.0)
    parser.add_argument("--formation-col-mix", type=float, default=1.0)
    parser.add_argument("--formation-module-mix", type=float, default=1.0)
    parser.add_argument("--formation-multiscale-pooling", choices=["none", "mean"], default="mean")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
