#!/usr/bin/env python3
"""Residual-stream route-growth test for GCO topology learning.

This moves the learned route-constructor test out of synthetic key space.
Keys are real residual-stream vectors captured from the real-book transformer.
Values are local residual write targets estimated from the negative gradient of
the next-token loss with respect to that same residual stream.

Question:
    Can a learned route gate grow around protected old residual states while
    still permitting a useful new residual write?

The route is still deliberately simple:
    phi(h) = relu(r_hat^T h - b_hat)

The difference from the previous route-constructor experiment is that h comes
from actual transformer residual dynamics, not sampled orthogonal vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

from models import DecoderTransformer  # noqa: E402

from gco_learned_route_constructor import (  # noqa: E402
    RouteConstructor,
    apply_budget,
    apply_budget_tensor,
    make_features,
    mse_columns,
    oracle_route,
    parse_int_list,
    resolve_device,
    route_activations,
    scalar,
    set_seed,
    soft_route_activations,
)


@dataclass(frozen=True)
class ResidualExample:
    key: torch.Tensor
    value: torch.Tensor
    chunk_id: str
    sequence_index: int
    position: int
    token_loss: float
    grad_norm: float
    score: float


@dataclass(frozen=True)
class ResidualRouteCase:
    old_keys: torch.Tensor
    old_values: torch.Tensor
    k_new: torch.Tensor
    v_new: torch.Tensor
    free_room_ratio: float
    protected_overlap_ratio: float
    old_count: int
    new_chunk_id: str
    new_position: int
    new_token_loss: float
    new_grad_norm: float


@dataclass(frozen=True)
class ResidualRouteEval:
    method: str
    old_count: int
    overlap_bucket: str
    protected_overlap_ratio: float
    free_room_ratio: float
    new_gain_fraction: float
    old_damage: float
    update_norm_unclipped: float
    update_norm_after_budget: float
    budget_scale: float
    new_activation: float
    old_activation_rms: float
    old_activation_max: float
    direction_cosine_to_new: float
    threshold: float


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite tensor detected: {name}.")


def require_finite_float(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite float detected: {name}={value!r}.")


def normalize_vector(vector: torch.Tensor, eps: float, *, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if scalar(norm) <= eps:
        raise RuntimeError(f"Cannot normalize near-zero vector for {name}.")
    result = vector / norm
    require_finite_tensor(name, result)
    return result


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        if device.type == "mps":
            raise RuntimeError("MPS does not support float64 for this experiment.")
        return torch.float64
    raise ValueError(f"Unknown dtype {name!r}.")


def load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_chunks(path: Path) -> list[dict[str, object]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Chunks file must contain a list: {path}")
    chunks: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk item {index} is not an object.")
        if "chunk_id" not in item or "text" not in item:
            raise ValueError(f"Chunk item {index} must contain chunk_id and text.")
        chunks.append(item)
    return chunks


def instantiate_base_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> DecoderTransformer:
    if not args.base_model_path.exists():
        raise FileNotFoundError(f"Base model checkpoint does not exist: {args.base_model_path}")
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    ).to(device)
    state = torch.load(args.base_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def resolve_layer_index(layer_index: int, layer_count: int) -> int:
    if layer_count <= 0:
        raise ValueError("Model has no transformer blocks.")
    resolved = layer_index
    if resolved < 0:
        resolved = layer_count + resolved
    if not (0 <= resolved < layer_count):
        raise ValueError(f"Layer index {layer_index} resolves to {resolved}, outside [0, {layer_count}).")
    return resolved


def token_windows(token_ids: Sequence[int], *, max_seq_len: int, stride: int, max_windows: int) -> list[list[int]]:
    if max_seq_len < 2:
        raise ValueError("--max-seq-len must be at least 2.")
    if stride <= 0:
        raise ValueError("--window-stride must be positive.")
    if max_windows <= 0:
        raise ValueError("--sequences-per-chunk must be positive.")
    if len(token_ids) < 2:
        raise ValueError("Cannot build residual examples from fewer than two tokens.")
    windows: list[list[int]] = []
    start = 0
    while start < len(token_ids) - 1 and len(windows) < max_windows:
        window = list(token_ids[start : start + max_seq_len])
        if len(window) >= 2:
            windows.append(window)
        start += stride
    if not windows:
        raise RuntimeError("No token windows were built.")
    return windows


def capture_residual_examples_for_chunk(
    *,
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    chunk: dict[str, object],
    layer_index: int,
    max_seq_len: int,
    window_stride: int,
    sequences_per_chunk: int,
    max_examples: int,
    min_grad_norm: float,
    device: torch.device,
    eps: float,
) -> list[ResidualExample]:
    if max_examples <= 0:
        raise ValueError("--examples-per-chunk must be positive.")
    if min_grad_norm < 0:
        raise ValueError("--min-grad-norm must be non-negative.")
    chunk_id = str(chunk["chunk_id"])
    text = str(chunk["text"])
    encoded = tokenizer.encode(text).ids
    windows = token_windows(
        encoded,
        max_seq_len=max_seq_len,
        stride=window_stride,
        max_windows=sequences_per_chunk,
    )
    examples: list[ResidualExample] = []
    block = model.blocks[layer_index]
    for sequence_index, window in enumerate(windows):
        capture: dict[str, torch.Tensor] = {}

        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError("Residual hook received non-tensor output.")
            output.retain_grad()
            capture["residual"] = output

        handle = block.register_forward_hook(hook)
        try:
            model.zero_grad(set_to_none=True)
            tokens = torch.tensor([window], device=device, dtype=torch.long)
            logits, _ = model(tokens)
            if "residual" not in capture:
                raise RuntimeError("Residual hook did not capture a tensor.")
            residual = capture["residual"]
            targets = tokens[:, 1:]
            pred_logits = logits[:, :-1, :]
            token_losses = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            loss = token_losses.mean()
            loss.backward()
            if residual.grad is None:
                raise RuntimeError("Residual hook tensor has no gradient.")
            residual_values = residual.detach()[0, :-1, :]
            residual_grads = residual.grad.detach()[0, :-1, :]
            grad_norms = torch.linalg.vector_norm(residual_grads, dim=-1)
            scores = token_losses.detach()[0] * grad_norms
            for position in range(residual_values.shape[0]):
                grad_norm = scalar(grad_norms[position])
                if grad_norm < min_grad_norm:
                    continue
                key = normalize_vector(residual_values[position], eps, name="residual_key")
                value = normalize_vector(-residual_grads[position], eps, name="residual_value")
                token_loss = scalar(token_losses.detach()[0, position])
                score = scalar(scores[position])
                examples.append(
                    ResidualExample(
                        key=key.detach(),
                        value=value.detach(),
                        chunk_id=chunk_id,
                        sequence_index=sequence_index,
                        position=position,
                        token_loss=token_loss,
                        grad_norm=grad_norm,
                        score=score,
                    )
                )
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    examples.sort(key=lambda item: item.score, reverse=True)
    selected = examples[:max_examples]
    if len(selected) < max_examples:
        raise RuntimeError(
            f"Chunk {chunk_id} produced only {len(selected)} residual examples, "
            f"but {max_examples} were requested."
        )
    return selected


def select_chunks(chunks: list[dict[str, object]], *, start: int, count: int, name: str) -> list[dict[str, object]]:
    if start < 0:
        raise ValueError(f"{name} start must be non-negative.")
    if count <= 0:
        raise ValueError(f"{name} count must be positive.")
    end = start + count
    if end > len(chunks):
        raise ValueError(f"{name} selection [{start}, {end}) exceeds chunk count {len(chunks)}.")
    return chunks[start:end]


def ridge_base_weight(old_keys: torch.Tensor, old_values: torch.Tensor, ridge: float) -> torch.Tensor:
    if ridge <= 0:
        raise ValueError("--base-ridge must be positive.")
    if old_keys.ndim != 2 or old_values.ndim != 2:
        raise ValueError("old_keys and old_values must be matrices.")
    if old_keys.shape != old_values.shape:
        raise ValueError(f"old_keys and old_values must share shape, got {old_keys.shape} and {old_values.shape}.")
    key_dim = old_keys.shape[0]
    identity = torch.eye(key_dim, device=old_keys.device, dtype=old_keys.dtype)
    left = old_values @ old_keys.T
    system = old_keys @ old_keys.T + ridge * identity
    weight = torch.linalg.solve(system.T, left.T).T
    require_finite_tensor("ridge_base_weight", weight)
    return weight


def protected_weight_delta(
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    lambda_protect: float,
    lambda_ridge: float,
) -> torch.Tensor:
    if lambda_protect < 0:
        raise ValueError("--lambda-protect must be non-negative.")
    if lambda_ridge <= 0:
        raise ValueError("--lambda-ridge must be positive.")
    key_dim = k_new.shape[0]
    identity = torch.eye(key_dim, device=k_new.device, dtype=k_new.dtype)
    system = k_new @ k_new.T + lambda_protect * (old_keys @ old_keys.T) + lambda_ridge * identity
    right = torch.linalg.solve(system, k_new)
    error = v_new - w_base @ k_new
    delta = error @ right.T
    require_finite_tensor("protected_weight_delta", delta)
    return delta


def route_delta(
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    *,
    lambda_old: float,
    lambda_ridge: float,
) -> torch.Tensor:
    if lambda_old < 0:
        raise ValueError("--route-lambda-old must be non-negative.")
    if lambda_ridge <= 0:
        raise ValueError("--route-lambda-ridge must be positive.")
    route_dim = z_new.shape[0]
    identity = torch.eye(route_dim, device=z_new.device, dtype=z_new.dtype)
    system = z_new @ z_new.T + lambda_old * (z_old @ z_old.T) + lambda_ridge * identity
    coeff = torch.linalg.solve(system, z_new)
    error = v_new - w_base @ k_new
    delta = error @ coeff.T
    require_finite_tensor("route_delta", delta)
    return delta


def build_cases(
    *,
    old_examples: Sequence[ResidualExample],
    new_examples: Sequence[ResidualExample],
    old_counts: Sequence[int],
    max_cases_per_old_count: int,
    base_ridge: float,
    eps: float,
) -> list[ResidualRouteCase]:
    if not old_examples:
        raise ValueError("old_examples cannot be empty.")
    if not new_examples:
        raise ValueError("new_examples cannot be empty.")
    if max_cases_per_old_count <= 0:
        raise ValueError("--cases-per-old-count must be positive.")
    old_key_matrix = torch.stack([item.key for item in old_examples], dim=1)
    cases: list[ResidualRouteCase] = []
    selected_new = list(new_examples[:max_cases_per_old_count])
    for old_count in old_counts:
        if old_count <= 0:
            raise ValueError("old_count values must be positive.")
        if old_count > len(old_examples):
            raise ValueError(f"old_count={old_count} exceeds available old examples {len(old_examples)}.")
        for new_example in selected_new:
            k_new_vec = new_example.key
            sims = k_new_vec.unsqueeze(0) @ old_key_matrix
            top = torch.topk(sims.squeeze(0), k=old_count, largest=True).indices
            old_keys = old_key_matrix.index_select(1, top)
            old_values = torch.stack([old_examples[int(index)].value for index in top.detach().cpu()], dim=1)
            max_overlap = scalar(torch.clamp((k_new_vec.unsqueeze(0) @ old_keys).max(), min=0.0, max=1.0))
            protected = float(max_overlap * max_overlap)
            free = float(max(0.0, 1.0 - protected))
            # Validate the local old map is numerically sane before storing the case.
            _ = ridge_base_weight(old_keys, old_values, base_ridge)
            cases.append(
                ResidualRouteCase(
                    old_keys=old_keys,
                    old_values=old_values,
                    k_new=k_new_vec.reshape(-1, 1),
                    v_new=new_example.value.reshape(-1, 1),
                    free_room_ratio=free,
                    protected_overlap_ratio=protected,
                    old_count=int(old_count),
                    new_chunk_id=new_example.chunk_id,
                    new_position=int(new_example.position),
                    new_token_loss=float(new_example.token_loss),
                    new_grad_norm=float(new_example.grad_norm),
                )
            )
    if not cases:
        raise RuntimeError("No residual route cases were built.")
    return cases


def overlap_bucket(value: float) -> str:
    require_finite_float("protected_overlap_ratio", value)
    if value < 0.25:
        return "0.00-0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    if value < 0.90:
        return "0.75-0.90"
    if value < 0.97:
        return "0.90-0.97"
    return "0.97-1.00"


def evaluate_protected(
    *,
    case: ResidualRouteCase,
    max_update_norm: float,
    lambda_protect: float,
    lambda_ridge: float,
    base_ridge: float,
) -> ResidualRouteEval:
    w_base = ridge_base_weight(case.old_keys, case.old_values, base_ridge)
    delta = protected_weight_delta(
        w_base,
        case.old_keys,
        case.k_new,
        case.v_new,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )
    budgeted, raw_norm, budgeted_norm, scale = apply_budget(delta, max_update_norm)
    old_before = scalar(mse_columns(w_base @ case.old_keys, case.old_values).mean())
    new_before = scalar(mse_columns(w_base @ case.k_new, case.v_new).mean())
    w_after = w_base + budgeted
    old_after = scalar(mse_columns(w_after @ case.old_keys, case.old_values).mean())
    new_after = scalar(mse_columns(w_after @ case.k_new, case.v_new).mean())
    return ResidualRouteEval(
        method="protected_budget",
        old_count=case.old_count,
        overlap_bucket=overlap_bucket(case.protected_overlap_ratio),
        protected_overlap_ratio=case.protected_overlap_ratio,
        free_room_ratio=case.free_room_ratio,
        new_gain_fraction=(new_before - new_after) / (new_before + 1e-12),
        old_damage=old_after - old_before,
        update_norm_unclipped=raw_norm,
        update_norm_after_budget=budgeted_norm,
        budget_scale=scale,
        new_activation=0.0,
        old_activation_rms=0.0,
        old_activation_max=0.0,
        direction_cosine_to_new=0.0,
        threshold=0.0,
    )


def evaluate_route(
    *,
    method: str,
    case: ResidualRouteCase,
    direction: torch.Tensor,
    threshold: torch.Tensor,
    max_update_norm: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
    base_ridge: float,
) -> ResidualRouteEval:
    w_base = ridge_base_weight(case.old_keys, case.old_values, base_ridge)
    z_old, z_new = route_activations(direction, threshold, case.old_keys, case.k_new)
    delta = route_delta(
        w_base,
        case.old_keys,
        case.k_new,
        case.v_new,
        z_old,
        z_new,
        lambda_old=route_lambda_old,
        lambda_ridge=route_lambda_ridge,
    )
    budgeted, raw_norm, budgeted_norm, scale = apply_budget(delta, max_update_norm)
    old_before = scalar(mse_columns(w_base @ case.old_keys, case.old_values).mean())
    new_before = scalar(mse_columns(w_base @ case.k_new, case.v_new).mean())
    old_after = scalar(mse_columns(w_base @ case.old_keys + budgeted @ z_old, case.old_values).mean())
    new_after = scalar(mse_columns(w_base @ case.k_new + budgeted @ z_new, case.v_new).mean())
    k_dir = F.normalize(case.k_new.T, dim=-1)
    return ResidualRouteEval(
        method=method,
        old_count=case.old_count,
        overlap_bucket=overlap_bucket(case.protected_overlap_ratio),
        protected_overlap_ratio=case.protected_overlap_ratio,
        free_room_ratio=case.free_room_ratio,
        new_gain_fraction=(new_before - new_after) / (new_before + 1e-12),
        old_damage=old_after - old_before,
        update_norm_unclipped=raw_norm,
        update_norm_after_budget=budgeted_norm,
        budget_scale=scale,
        new_activation=scalar(z_new.squeeze()),
        old_activation_rms=scalar(torch.sqrt((z_old**2).mean())),
        old_activation_max=scalar(z_old.max()),
        direction_cosine_to_new=scalar((direction * k_dir).sum(dim=1).mean()),
        threshold=scalar(threshold.mean()),
    )


def utility_loss_for_case(
    *,
    model: RouteConstructor,
    case: ResidualRouteCase,
    max_update_norm: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
    base_ridge: float,
    old_weight: float,
    budget_weight: float,
    old_gate_weight: float,
    train_temperature: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w_base = ridge_base_weight(case.old_keys, case.old_values, base_ridge)
    features = make_features(case.old_keys, case.k_new, case.free_room_ratio, case.protected_overlap_ratio)
    direction, threshold = model(features)
    z_old, z_new = soft_route_activations(
        direction,
        threshold,
        case.old_keys,
        case.k_new,
        train_temperature,
    )
    delta = route_delta(
        w_base,
        case.old_keys,
        case.k_new,
        case.v_new,
        z_old,
        z_new,
        lambda_old=route_lambda_old,
        lambda_ridge=route_lambda_ridge,
    )
    budgeted, raw_norm, budget_scale = apply_budget_tensor(delta, max_update_norm, eps)
    new_before = mse_columns(w_base @ case.k_new, case.v_new).mean().detach()
    new_after = mse_columns(w_base @ case.k_new + budgeted @ z_new, case.v_new).mean()
    old_after = mse_columns(w_base @ case.old_keys + budgeted @ z_old, case.old_values).mean()
    budget_penalty = torch.relu(raw_norm / max_update_norm - 1.0).pow(2)
    old_gate_penalty = (z_old**2).mean()
    normalized_new_after = new_after / (new_before + eps)
    loss = (
        normalized_new_after
        + old_weight * old_after
        + budget_weight * budget_penalty
        + old_gate_weight * old_gate_penalty
    )
    require_finite_tensor("utility_loss", loss)
    return loss, {
        "new_after_fraction": normalized_new_after.detach(),
        "old_damage": old_after.detach(),
        "budget_penalty": budget_penalty.detach(),
        "old_gate": old_gate_penalty.detach(),
        "budget_scale": budget_scale.detach(),
        "new_activation": z_new.detach().mean(),
        "old_activation": z_old.detach().mean(),
    }


def train_route_constructor(
    *,
    model: RouteConstructor,
    cases: Sequence[ResidualRouteCase],
    steps: int,
    batch_size: int,
    lr: float,
    max_update_norm: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
    base_ridge: float,
    old_weight: float,
    budget_weight: float,
    old_gate_weight: float,
    train_temperature: float,
    eps: float,
) -> list[dict[str, float]]:
    if steps <= 0:
        raise ValueError("--train-steps must be positive.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not cases:
        raise ValueError("Cannot train on empty residual route case list.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, steps + 1):
        metric_sums: dict[str, torch.Tensor] = {}
        losses: list[torch.Tensor] = []
        for _ in range(batch_size):
            index = int(torch.randint(0, len(cases), ()).item())
            loss, metrics = utility_loss_for_case(
                model=model,
                case=cases[index],
                max_update_norm=max_update_norm,
                route_lambda_old=route_lambda_old,
                route_lambda_ridge=route_lambda_ridge,
                base_ridge=base_ridge,
                old_weight=old_weight,
                budget_weight=budget_weight,
                old_gate_weight=old_gate_weight,
                train_temperature=train_temperature,
                eps=eps,
            )
            losses.append(loss)
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value
        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            item = {"step": float(step), "loss": scalar(batch_loss)}
            for name, total in metric_sums.items():
                item[name] = scalar(total / float(batch_size))
            history.append(item)
    return history


def evaluate_cases(
    *,
    model: RouteConstructor,
    cases: Sequence[ResidualRouteCase],
    max_update_norm: float,
    lambda_protect: float,
    lambda_ridge: float,
    route_lambda_old: float,
    route_lambda_ridge: float,
    base_ridge: float,
) -> list[ResidualRouteEval]:
    if not cases:
        raise ValueError("Cannot evaluate empty residual route case list.")
    rows: list[ResidualRouteEval] = []
    model.eval()
    with torch.no_grad():
        for case in cases:
            rows.append(
                evaluate_protected(
                    case=case,
                    max_update_norm=max_update_norm,
                    lambda_protect=lambda_protect,
                    lambda_ridge=lambda_ridge,
                    base_ridge=base_ridge,
                )
            )
            oracle_direction, oracle_threshold = oracle_route(case.old_keys, case.k_new)
            rows.append(
                evaluate_route(
                    method="constructive_relu",
                    case=case,
                    direction=oracle_direction,
                    threshold=oracle_threshold,
                    max_update_norm=max_update_norm,
                    route_lambda_old=route_lambda_old,
                    route_lambda_ridge=route_lambda_ridge,
                    base_ridge=base_ridge,
                )
            )
            features = make_features(case.old_keys, case.k_new, case.free_room_ratio, case.protected_overlap_ratio)
            learned_direction, learned_threshold = model(features)
            rows.append(
                evaluate_route(
                    method="learned_relu",
                    case=case,
                    direction=learned_direction,
                    threshold=learned_threshold,
                    max_update_norm=max_update_norm,
                    route_lambda_old=route_lambda_old,
                    route_lambda_ridge=route_lambda_ridge,
                    base_ridge=base_ridge,
                )
            )
    return rows


def aggregate(rows: Sequence[ResidualRouteEval]) -> list[dict[str, float | int | str]]:
    if not rows:
        raise ValueError("Cannot aggregate empty result rows.")
    buckets: dict[tuple[int, str, str], list[ResidualRouteEval]] = {}
    for row in rows:
        buckets.setdefault((row.old_count, row.overlap_bucket, row.method), []).append(row)
    aggregated: list[dict[str, float | int | str]] = []
    for (old_count, bucket, method), items in sorted(buckets.items()):
        def avg(name: str) -> float:
            values = [float(getattr(item, name)) for item in items]
            for value in values:
                require_finite_float(name, value)
            return float(sum(values) / len(values))

        aggregated.append(
            {
                "old_count": old_count,
                "overlap_bucket": bucket,
                "method": method,
                "case_count": len(items),
                "protected_overlap_ratio": avg("protected_overlap_ratio"),
                "free_room_ratio": avg("free_room_ratio"),
                "new_gain_fraction": avg("new_gain_fraction"),
                "old_damage": avg("old_damage"),
                "update_norm_after_budget": avg("update_norm_after_budget"),
                "budget_scale": avg("budget_scale"),
                "new_activation": avg("new_activation"),
                "old_activation_rms": avg("old_activation_rms"),
                "old_activation_max": avg("old_activation_max"),
                "direction_cosine_to_new": avg("direction_cosine_to_new"),
                "threshold": avg("threshold"),
            }
        )
    return aggregated


def print_training_trace(history: Sequence[dict[str, float]]) -> None:
    print("\nTraining trace")
    print("-" * 112)
    for row in history:
        print(
            f"step={int(row['step']):6d} "
            f"loss={row['loss']:.5f} "
            f"new_after_fraction={row['new_after_fraction']:.5f} "
            f"old_damage={row['old_damage']:.5f} "
            f"budget_penalty={row['budget_penalty']:.5f} "
            f"old_gate={row['old_gate']:.5f} "
            f"budget_scale={row['budget_scale']:.5f} "
            f"new_activation={row['new_activation']:.5f} "
            f"old_activation={row['old_activation']:.5f}"
        )


def print_aggregate(rows: Sequence[dict[str, float | int | str]]) -> None:
    print("\nReadable aggregate summary")
    print("-" * 156)
    print(
        "old overlap       n  method              gain_frac old_damage  upd_norm scale  "
        "new_act old_rms old_max cos_new threshold"
    )
    print("-" * 156)
    for row in rows:
        print(
            f"{int(row['old_count']):3d} "
            f"{str(row['overlap_bucket']):>11} "
            f"{int(row['case_count']):3d} "
            f"{str(row['method']):<19} "
            f"{float(row['new_gain_fraction']):9.3f} "
            f"{float(row['old_damage']):10.3g} "
            f"{float(row['update_norm_after_budget']):8.3g} "
            f"{float(row['budget_scale']):5.3f} "
            f"{float(row['new_activation']):7.3f} "
            f"{float(row['old_activation_rms']):7.3f} "
            f"{float(row['old_activation_max']):7.3f} "
            f"{float(row['direction_cosine_to_new']):7.3f} "
            f"{float(row['threshold']):9.3f}"
        )
    print("-" * 156)
    print("\nWhat to look for:")
    print("  learned_relu high gain_frac means the learned gate found a useful residual route.")
    print("  old_damage near 0 means the residual route stayed isolated from old protected states.")
    print("  old_rms/old_max near 0 means old residual states did not activate the new route.")
    print("  cos_new near 1 means the learned direction follows the new residual state.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-real-book-residual-route-growth-seed0.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--old-start", type=int, default=0)
    parser.add_argument("--old-chunk-count", type=int, default=2)
    parser.add_argument("--new-start", type=int, default=2)
    parser.add_argument("--new-chunk-count", type=int, default=1)
    parser.add_argument("--sequences-per-chunk", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=64)
    parser.add_argument("--examples-per-chunk", type=int, default=96)
    parser.add_argument("--min-grad-norm", type=float, default=1e-8)
    parser.add_argument("--old-counts", type=str, default="8,24,48")
    parser.add_argument("--cases-per-old-count", type=int, default=64)
    parser.add_argument("--eval-case-fraction", type=float, default=0.5)
    parser.add_argument("--constructor-hidden-dim", type=int, default=128)
    parser.add_argument("--direction-anchor", choices=["none", "new_key"], default="new_key")
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--threshold-bias", type=float, default=0.0)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-update-norm", type=float, default=2.0)
    parser.add_argument("--base-ridge", type=float, default=1e-3)
    parser.add_argument("--lambda-protect", type=float, default=10.0)
    parser.add_argument("--lambda-ridge", type=float, default=1e-3)
    parser.add_argument("--route-lambda-old", type=float, default=20.0)
    parser.add_argument("--route-lambda-ridge", type=float, default=1e-3)
    parser.add_argument("--old-weight", type=float, default=10.0)
    parser.add_argument("--budget-weight", type=float, default=0.25)
    parser.add_argument("--old-gate-weight", type=float, default=10.0)
    parser.add_argument("--train-temperature", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")
    if not (0.0 < args.eval_case_fraction < 1.0):
        raise ValueError("--eval-case-fraction must be in (0, 1).")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    chunks = load_chunks(args.chunks_path)
    old_chunks = select_chunks(chunks, start=args.old_start, count=args.old_chunk_count, name="old chunks")
    new_chunks = select_chunks(chunks, start=args.new_start, count=args.new_chunk_count, name="new chunks")
    old_chunk_ids = {str(chunk["chunk_id"]) for chunk in old_chunks}
    new_chunk_ids = {str(chunk["chunk_id"]) for chunk in new_chunks}
    if old_chunk_ids & new_chunk_ids:
        raise ValueError(f"Old and new chunk selections overlap: {sorted(old_chunk_ids & new_chunk_ids)}")
    model = instantiate_base_model(args, tokenizer.get_vocab_size(), device)
    layer_index = resolve_layer_index(args.layer_index, len(model.blocks))

    print("GCO REAL-BOOK RESIDUAL ROUTE-GROWTH TEST")
    print("=" * 112)
    print("Question:")
    print("  Can a learned gate grow around protected old residual states while")
    print("  still permitting a useful write from real transformer residual gradients?")
    print("\nRoute:")
    print("  phi(h) = relu(r_hat^T h - b_hat)")
    print("\nResidual source:")
    print(f"  layer_index={layer_index}, old_chunks={sorted(old_chunk_ids)}, new_chunks={sorted(new_chunk_ids)}")
    print("=" * 112)

    old_examples: list[ResidualExample] = []
    for chunk in old_chunks:
        old_examples.extend(
            capture_residual_examples_for_chunk(
                model=model,
                tokenizer=tokenizer,
                chunk=chunk,
                layer_index=layer_index,
                max_seq_len=args.max_seq_len,
                window_stride=args.window_stride,
                sequences_per_chunk=args.sequences_per_chunk,
                max_examples=args.examples_per_chunk,
                min_grad_norm=args.min_grad_norm,
                device=device,
                eps=args.eps,
            )
        )
    new_examples: list[ResidualExample] = []
    for chunk in new_chunks:
        new_examples.extend(
            capture_residual_examples_for_chunk(
                model=model,
                tokenizer=tokenizer,
                chunk=chunk,
                layer_index=layer_index,
                max_seq_len=args.max_seq_len,
                window_stride=args.window_stride,
                sequences_per_chunk=args.sequences_per_chunk,
                max_examples=args.examples_per_chunk,
                min_grad_norm=args.min_grad_norm,
                device=device,
                eps=args.eps,
            )
        )
    old_examples.sort(key=lambda item: item.score, reverse=True)
    new_examples.sort(key=lambda item: item.score, reverse=True)
    old_counts = parse_int_list(args.old_counts)
    cases = build_cases(
        old_examples=old_examples,
        new_examples=new_examples,
        old_counts=old_counts,
        max_cases_per_old_count=args.cases_per_old_count,
        base_ridge=args.base_ridge,
        eps=args.eps,
    )
    permutation = torch.randperm(len(cases)).tolist()
    cases = [cases[index] for index in permutation]
    split_index = int(round(len(cases) * (1.0 - args.eval_case_fraction)))
    if split_index <= 0 or split_index >= len(cases):
        raise RuntimeError(
            f"Invalid train/eval split: split_index={split_index}, case_count={len(cases)}. "
            "Adjust --eval-case-fraction or --cases-per-old-count."
        )
    train_cases = cases[:split_index]
    eval_cases = cases[split_index:]
    route_constructor = RouteConstructor(
        args.d_model,
        args.constructor_hidden_dim,
        direction_anchor=args.direction_anchor,
        residual_scale=args.residual_scale,
        threshold_bias=args.threshold_bias,
    ).to(device=device, dtype=dtype)
    history = train_route_constructor(
        model=route_constructor,
        cases=train_cases,
        steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_update_norm=args.max_update_norm,
        route_lambda_old=args.route_lambda_old,
        route_lambda_ridge=args.route_lambda_ridge,
        base_ridge=args.base_ridge,
        old_weight=args.old_weight,
        budget_weight=args.budget_weight,
        old_gate_weight=args.old_gate_weight,
        train_temperature=args.train_temperature,
        eps=args.eps,
    )
    eval_rows = evaluate_cases(
        model=route_constructor,
        cases=eval_cases,
        max_update_norm=args.max_update_norm,
        lambda_protect=args.lambda_protect,
        lambda_ridge=args.lambda_ridge,
        route_lambda_old=args.route_lambda_old,
        route_lambda_ridge=args.route_lambda_ridge,
        base_ridge=args.base_ridge,
    )
    aggregate_rows = aggregate(eval_rows)
    print_training_trace(history)
    print_aggregate(aggregate_rows)
    result = {
        "config": {
            "base_model_path": str(args.base_model_path),
            "tokenizer_path": str(args.tokenizer_path),
            "chunks_path": str(args.chunks_path),
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "max_seq_len": args.max_seq_len,
            "layer_index": layer_index,
            "old_chunk_ids": sorted(old_chunk_ids),
            "new_chunk_ids": sorted(new_chunk_ids),
            "old_counts": old_counts,
            "cases_per_old_count": args.cases_per_old_count,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_update_norm": args.max_update_norm,
            "base_ridge": args.base_ridge,
            "lambda_protect": args.lambda_protect,
            "lambda_ridge": args.lambda_ridge,
            "route_lambda_old": args.route_lambda_old,
            "route_lambda_ridge": args.route_lambda_ridge,
        },
        "question": "Can learned topology growth separate new residual states from protected old residual states?",
        "old_example_count": len(old_examples),
        "new_example_count": len(new_examples),
        "case_count": len(cases),
        "train_case_count": len(train_cases),
        "eval_case_count": len(eval_cases),
        "training_trace": history,
        "aggregate": aggregate_rows,
        "eval_records": [asdict(row) for row in eval_rows],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nwrote_json={args.output_json}")
    return result


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
