"""Latent-geometry-guided optimizer benchmark on closed latent algebra.

This script tests a narrow first claim:

    Can latent geometry decide when to reuse, when to update, and when to
    allocate a closed latent operator?

It intentionally lives outside usage_score_ops.py so the optimizer experiment
can evolve without changing the older benchmark file.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TaskFn = Callable[[np.ndarray, np.ndarray | None, int], np.ndarray]
Program = tuple[Any, ...]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    arity: int
    fn: TaskFn


@dataclass(frozen=True)
class Config:
    num_values: int
    code_dim: int
    hidden_dim: int
    ae_epochs: int
    operator_epochs: int
    update_epochs: int
    ae_lr: float
    operator_lr: float
    update_lr: float
    lambda_closure: float
    separation_margin: float
    separation_weight: float
    grad_memory_beta: float
    reuse_acc_threshold: float
    reuse_closure_threshold: float
    update_min_acc: float
    update_closure_low: float
    update_closure_high: float
    repair_update_rule: str
    projection_norm_floor: float
    min_responsibility: float
    neuron_gate_power: float
    structural_risk_signal: str
    structural_need_signal: str
    structural_risk_mode: str
    structural_risk_power: float
    structural_need_power: float
    search_depth: int
    max_programs: int
    device: torch.device


@dataclass
class DecisionEvent:
    seed: int
    policy: str
    task: str
    decision: str
    best_program: str | None
    best_accuracy: float
    best_loss: float
    closure_norm: float
    manifold_norm: float
    gate_mean: float | None
    operator_count: int
    candidate_outputs: list[list[float]]
    candidate_targets: list[int]


class LatentBoundary(nn.Module):
    def __init__(self, num_values: int, code_dim: int) -> None:
        super().__init__()
        self.codebook = nn.Parameter(torch.empty(num_values, code_dim))
        self.decoder = nn.Linear(code_dim, num_values)
        nn.init.normal_(self.codebook, mean=0.0, std=1.0)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=1.0 / np.sqrt(code_dim))
        nn.init.zeros_(self.decoder.bias)

    def encode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook[indices]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class ClosedOperator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, code_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        first = self.net[0]
        second = self.net[2]
        assert isinstance(first, nn.Linear)
        assert isinstance(second, nn.Linear)
        nn.init.normal_(first.weight, mean=0.0, std=1.0 / np.sqrt(input_dim))
        nn.init.constant_(first.bias, 0.01)
        nn.init.normal_(second.weight, mean=0.0, std=1.0 / np.sqrt(hidden_dim))
        nn.init.zeros_(second.bias)

    def forward(self, codes: list[torch.Tensor]) -> torch.Tensor:
        return self.net(torch.cat(codes, dim=-1))


@dataclass
class OperatorRecord:
    name: str
    arity: int
    module: ClosedOperator
    origin_task: str
    grad_memory: dict[str, torch.Tensor]
    parameter_count: int
    update_count: int = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def task_add(a: np.ndarray, b: np.ndarray | None, num_values: int) -> np.ndarray:
    if b is None:
        raise ValueError("ADD requires two operands.")
    return (a + b) % num_values


def task_max(a: np.ndarray, b: np.ndarray | None, num_values: int) -> np.ndarray:
    if b is None:
        raise ValueError("MAX requires two operands.")
    return np.maximum(a, b)


def task_copy(a: np.ndarray, b: np.ndarray | None, num_values: int) -> np.ndarray:
    del b, num_values
    return a.copy()


def task_min(a: np.ndarray, b: np.ndarray | None, num_values: int) -> np.ndarray:
    if b is None:
        raise ValueError("MIN requires two operands.")
    return np.minimum(a, b)


def task_sub(a: np.ndarray, b: np.ndarray | None, num_values: int) -> np.ndarray:
    if b is None:
        raise ValueError("SUB requires two operands.")
    return (a - b) % num_values


TASKS: dict[str, TaskSpec] = {
    "ADD": TaskSpec("ADD", 2, task_add),
    "MAX": TaskSpec("MAX", 2, task_max),
    "COPY": TaskSpec("COPY", 1, task_copy),
    "MIN": TaskSpec("MIN", 2, task_min),
    "SUB": TaskSpec("SUB", 2, task_sub),
}


COMPOSITIONS: dict[str, tuple[Program, TaskFn]] = {
    "max_of_sum": (
        ("op", "MAX", (("op", "ADD", (("var", 0), ("var", 1))), ("var", 2))),
        lambda a, b, n: np.maximum((a[:, 0] + a[:, 1]) % n, a[:, 2]),
    ),
    "sum_of_max": (
        ("op", "ADD", (("op", "MAX", (("var", 0), ("var", 1))), ("var", 2))),
        lambda a, b, n: (np.maximum(a[:, 0], a[:, 1]) + a[:, 2]) % n,
    ),
    "sub_of_sum": (
        ("op", "SUB", (("op", "ADD", (("var", 0), ("var", 1))), ("var", 2))),
        lambda a, b, n: ((a[:, 0] + a[:, 1]) % n - a[:, 2]) % n,
    ),
    "max_of_min": (
        ("op", "MAX", (("op", "MIN", (("var", 0), ("var", 1))), ("var", 2))),
        lambda a, b, n: np.maximum(np.minimum(a[:, 0], a[:, 1]), a[:, 2]),
    ),
    "sum_of_copy": (
        ("op", "ADD", (("op", "COPY", (("var", 2),)), ("var", 0))),
        lambda a, b, n: (a[:, 2] + a[:, 0]) % n,
    ),
}


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but torch.backends.mps is not available.")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    raise ValueError(f"Unknown device {name!r}. Expected cpu, mps, or cuda.")


def parse_stream(value: str) -> list[str]:
    tasks = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not tasks:
        raise ValueError("--stream must contain at least one task.")
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s) in --stream: {unknown}. Available: {sorted(TASKS)}")
    return tasks


def parse_csv_items(value: str, option_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option_name} must contain at least one item.")
    return items


def make_task_data(task: TaskSpec, num_values: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if task.arity == 1:
        inputs = np.arange(num_values, dtype=np.int64).reshape(-1, 1)
        targets = task.fn(inputs[:, 0], None, num_values)
    elif task.arity == 2:
        a, b = np.meshgrid(np.arange(num_values), np.arange(num_values), indexing="ij")
        inputs = np.stack([a.reshape(-1), b.reshape(-1)], axis=1).astype(np.int64)
        targets = task.fn(inputs[:, 0], inputs[:, 1], num_values)
    else:
        raise ValueError(f"Unsupported task arity {task.arity} for task {task.name}.")
    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )


def make_composition_data(num_values: int, device: torch.device) -> torch.Tensor:
    d0, d1, d2 = np.meshgrid(
        np.arange(num_values),
        np.arange(num_values),
        np.arange(num_values),
        indexing="ij",
    )
    inputs = np.stack([d0.reshape(-1), d1.reshape(-1), d2.reshape(-1)], axis=1).astype(np.int64)
    return torch.as_tensor(inputs, dtype=torch.long, device=device)


def code_distance_scale(boundary: LatentBoundary) -> float:
    with torch.no_grad():
        codes = boundary.codebook.detach()
        dists = torch.cdist(codes, codes, p=2.0).pow(2)
        mask = ~torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
        nearest = dists.masked_select(mask).view(dists.shape[0], dists.shape[0] - 1).min(dim=1).values
        scale = nearest.mean().item()
    if scale <= 0.0:
        raise RuntimeError("Code distance scale is non-positive; latent codebook collapsed.")
    return scale


def train_boundary(boundary: LatentBoundary, cfg: Config) -> None:
    values = torch.arange(cfg.num_values, dtype=torch.long, device=cfg.device)
    optimizer = torch.optim.Adam(boundary.parameters(), lr=cfg.ae_lr)
    for _ in range(cfg.ae_epochs):
        optimizer.zero_grad()
        codes = boundary.encode(values)
        logits = boundary.decode(codes)
        ce = F.cross_entropy(logits, values)
        pair_dists = torch.cdist(codes, codes, p=2.0).pow(2)
        mask = ~torch.eye(cfg.num_values, dtype=torch.bool, device=cfg.device)
        sep = F.relu(cfg.separation_margin - pair_dists.masked_select(mask)).mean()
        loss = ce + cfg.separation_weight * sep
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = (boundary.decode(boundary.encode(values)).argmax(dim=-1) == values).float().mean().item()
    if acc < 1.0:
        raise RuntimeError(f"Boundary autoencoder did not reach perfect reconstruction: acc={acc:.4f}")
    boundary.freeze()


def operator_parameter_count(arity: int, code_dim: int, hidden_dim: int) -> int:
    input_dim = arity * code_dim
    return input_dim * hidden_dim + hidden_dim + hidden_dim * code_dim + code_dim


def new_operator(arity: int, origin_task: str, cfg: Config, name: str) -> OperatorRecord:
    module = ClosedOperator(arity * cfg.code_dim, cfg.hidden_dim, cfg.code_dim).to(cfg.device)
    return OperatorRecord(
        name=name,
        arity=arity,
        module=module,
        origin_task=origin_task,
        grad_memory={},
        parameter_count=operator_parameter_count(arity, cfg.code_dim, cfg.hidden_dim),
    )


def gather_codes(boundary: LatentBoundary, digit_indices: torch.Tensor, arity: int) -> list[torch.Tensor]:
    return [boundary.encode(digit_indices[:, idx]) for idx in range(arity)]


def closure_loss(out_code: torch.Tensor, target_code: torch.Tensor) -> torch.Tensor:
    return (out_code - target_code).pow(2).sum(dim=-1).mean()


def manifold_error(out_code: torch.Tensor, boundary: LatentBoundary) -> torch.Tensor:
    dists = torch.cdist(out_code, boundary.codebook.detach(), p=2.0).pow(2)
    return dists.min(dim=1).values.mean()


def forward_operator_on_task(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
) -> torch.Tensor:
    codes = gather_codes(boundary, digit_indices, record.arity)
    return record.module(codes)


def update_grad_memory(record: OperatorRecord, beta: float) -> None:
    for name, parameter in record.module.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for operator {record.name} parameter {name}.")
        grad = parameter.grad.detach().clone()
        if name not in record.grad_memory:
            record.grad_memory[name] = grad
        else:
            record.grad_memory[name].mul_(beta).add_(grad, alpha=1.0 - beta)


def train_new_operator(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    require_perfect: bool = True,
) -> None:
    optimizer = torch.optim.Adam(record.module.parameters(), lr=cfg.operator_lr)
    target_code = boundary.encode(targets).detach()
    for _ in range(cfg.operator_epochs):
        optimizer.zero_grad()
        out_code = forward_operator_on_task(boundary, record, digit_indices)
        logits = boundary.decode(out_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * closure_loss(out_code, target_code)
        loss.backward()
        update_grad_memory(record, cfg.grad_memory_beta)
        optimizer.step()
    metrics = evaluate_direct_operator(boundary, record, digit_indices, targets, cfg)
    if require_perfect and metrics["accuracy"] < 1.0:
        raise RuntimeError(
            f"New operator {record.name} failed to fit origin task {record.origin_task}: "
            f"accuracy={metrics['accuracy']:.4f}"
        )


def decompose_gradient(
    grad: torch.Tensor,
    slow: torch.Tensor | None,
    norm_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if norm_floor <= 0.0:
        raise ValueError(f"norm_floor must be positive, got {norm_floor}.")
    if slow is None:
        return torch.zeros_like(grad), grad
    slow_norm_sq = torch.sum(slow * slow)
    if slow_norm_sq.item() <= norm_floor:
        return torch.zeros_like(grad), grad
    coefficient = torch.sum(grad * slow) / slow_norm_sq
    reinforce = coefficient * slow
    novel = grad - reinforce
    return reinforce, novel


def medium_band_gate(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError(f"update_closure_high must be greater than update_closure_low, got {high} <= {low}.")
    if value <= low:
        return 0.0
    if value >= high:
        return 0.0
    midpoint = 0.5 * (low + high)
    if value <= midpoint:
        return (value - low) / (midpoint - low)
    return (high - value) / (high - midpoint)


def latent_guided_update(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    target_code = boundary.encode(targets).detach()
    gates: list[float] = []
    reinforce_norms: list[float] = []
    novel_norms: list[float] = []
    update_norms: list[float] = []
    for _ in range(cfg.update_epochs):
        record.module.zero_grad(set_to_none=True)
        out_code = forward_operator_on_task(boundary, record, digit_indices)
        logits = boundary.decode(out_code)
        ce = F.cross_entropy(logits, targets)
        close = closure_loss(out_code, target_code)
        loss = ce + cfg.lambda_closure * close
        loss.backward()
        closure_norm = close.detach().item() / distance_scale
        gate = medium_band_gate(closure_norm, cfg.update_closure_low, cfg.update_closure_high)
        gates.append(gate)
        with torch.no_grad():
            step_reinforce_norm = 0.0
            step_novel_norm = 0.0
            step_update_norm = 0.0
            for name, parameter in record.module.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"Missing gradient for operator {record.name} parameter {name}.")
                grad = parameter.grad.detach()
                slow = record.grad_memory.get(name)
                reinforce, novel = decompose_gradient(grad, slow, cfg.projection_norm_floor)
                update = reinforce + gate * novel
                parameter.add_(update, alpha=-cfg.update_lr)
                step_reinforce_norm += reinforce.pow(2).sum().item()
                step_novel_norm += novel.pow(2).sum().item()
                step_update_norm += update.pow(2).sum().item()
            reinforce_norms.append(float(np.sqrt(step_reinforce_norm)))
            novel_norms.append(float(np.sqrt(step_novel_norm)))
            update_norms.append(float(np.sqrt(step_update_norm)))
        update_grad_memory(record, cfg.grad_memory_beta)
    record.update_count += 1
    return {
        "gate_mean": float(np.mean(gates)),
        "reinforce_norm_mean": float(np.mean(reinforce_norms)),
        "novel_norm_mean": float(np.mean(novel_norms)),
        "update_norm_mean": float(np.mean(update_norms)),
    }


def adam_repair_update(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> None:
    optimizer = torch.optim.Adam(record.module.parameters(), lr=cfg.update_lr)
    target_code = boundary.encode(targets).detach()
    for _ in range(cfg.update_epochs):
        optimizer.zero_grad()
        out_code = forward_operator_on_task(boundary, record, digit_indices)
        logits = boundary.decode(out_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * closure_loss(out_code, target_code)
        loss.backward()
        update_grad_memory(record, cfg.grad_memory_beta)
        optimizer.step()
    record.update_count += 1


def operator_layers(record: OperatorRecord) -> tuple[nn.Linear, nn.Linear]:
    first = record.module.net[0]
    second = record.module.net[2]
    if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
        raise TypeError("ClosedOperator must be Linear -> ReLU -> Linear for neuron-gated updates.")
    return first, second


def forward_operator_with_hidden(
    record: OperatorRecord,
    codes: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = operator_layers(record)
    hidden = F.relu(first(torch.cat(codes, dim=-1)))
    out_code = second(hidden)
    return out_code, hidden


def tensor_is_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains non-finite values.")


def neuron_gate_vector(
    record: OperatorRecord,
    hidden: torch.Tensor,
    closure_gate: float,
    cfg: Config,
) -> torch.Tensor:
    first, second = operator_layers(record)
    if first.weight.grad is None or first.bias.grad is None or second.weight.grad is None:
        raise RuntimeError(f"Missing gradients for neuron gate calculation in {record.name}.")
    activation = hidden.detach().abs().mean(dim=0)
    incoming_grad = first.weight.grad.detach().norm(dim=1) + first.bias.grad.detach().abs()
    outgoing_grad = second.weight.grad.detach().norm(dim=0)
    responsibility = activation * (incoming_grad + outgoing_grad)
    tensor_is_finite("neuron responsibility", responsibility)
    max_responsibility = responsibility.max().item()
    if max_responsibility <= cfg.min_responsibility:
        return torch.zeros_like(responsibility)
    normalized = responsibility / max_responsibility
    return closure_gate * normalized.pow(cfg.neuron_gate_power)


def normalize_gate_signal(name: str, values: torch.Tensor, floor: float) -> torch.Tensor:
    tensor_is_finite(name, values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D per-neuron vector, got shape {tuple(values.shape)}.")
    if floor <= 0.0:
        raise ValueError(f"floor must be positive, got {floor}.")
    max_value = values.max().item()
    if max_value <= floor:
        return torch.zeros_like(values)
    return values / max_value


def structural_gate_vector(
    record: OperatorRecord,
    hidden: torch.Tensor,
    closure_gate: float,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    first, second = operator_layers(record)
    if first.weight.grad is None or first.bias.grad is None or second.weight.grad is None:
        raise RuntimeError(f"Missing gradients for structural gate calculation in {record.name}.")
    activation = hidden.detach().abs().mean(dim=0)
    incoming_weight = first.weight.detach().norm(dim=1) + first.bias.detach().abs()
    downstream_weight = second.weight.detach().norm(dim=0)
    incoming_grad = first.weight.grad.detach().norm(dim=1) + first.bias.grad.detach().abs()
    downstream_grad = second.weight.grad.detach().norm(dim=0)
    gradient_need = incoming_grad + downstream_grad
    responsibility_need = activation * gradient_need

    if cfg.structural_risk_signal == "activation_downstream":
        risk_raw = activation * downstream_weight
    elif cfg.structural_risk_signal == "activation_weight_product":
        risk_raw = activation * incoming_weight * downstream_weight
    else:
        raise ValueError(
            f"Unknown structural_risk_signal={cfg.structural_risk_signal!r}. "
            "Expected activation_downstream or activation_weight_product."
        )

    if cfg.structural_need_signal == "gradient":
        need_raw = gradient_need
    elif cfg.structural_need_signal == "responsibility":
        need_raw = responsibility_need
    else:
        raise ValueError(
            f"Unknown structural_need_signal={cfg.structural_need_signal!r}. "
            "Expected gradient or responsibility."
        )

    risk = normalize_gate_signal("structural risk", risk_raw, cfg.min_responsibility)
    need = normalize_gate_signal("structural need", need_raw, cfg.min_responsibility)
    if cfg.structural_risk_mode == "repair":
        risk_factor = risk.pow(cfg.structural_risk_power)
    elif cfg.structural_risk_mode == "protect":
        risk_factor = (1.0 - risk).clamp(min=0.0, max=1.0).pow(cfg.structural_risk_power)
    else:
        raise ValueError(f"Unknown structural_risk_mode={cfg.structural_risk_mode!r}. Expected repair or protect.")
    need_factor = need.pow(cfg.structural_need_power)
    gates = closure_gate * need_factor * risk_factor
    tensor_is_finite("structural gates", gates)
    return gates, {
        "risk_mean": float(risk.mean().item()),
        "risk_max": float(risk.max().item()),
        "need_mean": float(need.mean().item()),
        "need_max": float(need.max().item()),
    }


def update_with_slice(
    parameter_slice: torch.Tensor,
    grad_slice: torch.Tensor,
    slow_slice: torch.Tensor | None,
    gate: float,
    closure_gate: float,
    cfg: Config,
) -> tuple[float, float, float]:
    reinforce, novel = decompose_gradient(grad_slice, slow_slice, cfg.projection_norm_floor)
    update = closure_gate * reinforce + gate * novel
    tensor_is_finite("neuron-gated update", update)
    parameter_slice.add_(update, alpha=-cfg.update_lr)
    return (
        float(reinforce.pow(2).sum().item()),
        float(novel.pow(2).sum().item()),
        float(update.pow(2).sum().item()),
    )


def slow_slice(
    record: OperatorRecord,
    name: str,
    index: int | None = None,
    column: int | None = None,
) -> torch.Tensor | None:
    value = record.grad_memory.get(name)
    if value is None:
        return None
    if index is not None:
        return value[index]
    if column is not None:
        return value[:, column]
    return value


def neuron_gated_repair_update(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    first, second = operator_layers(record)
    target_code = boundary.encode(targets).detach()
    gate_means: list[float] = []
    gate_maxes: list[float] = []
    active_neuron_counts: list[float] = []
    reinforce_norms: list[float] = []
    novel_norms: list[float] = []
    update_norms: list[float] = []
    for _ in range(cfg.update_epochs):
        record.module.zero_grad(set_to_none=True)
        codes = gather_codes(boundary, digit_indices, record.arity)
        out_code, hidden = forward_operator_with_hidden(record, codes)
        logits = boundary.decode(out_code)
        close = closure_loss(out_code, target_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * close
        tensor_is_finite("neuron-gated loss", loss)
        loss.backward()
        closure_norm = close.detach().item() / distance_scale
        closure_gate = medium_band_gate(closure_norm, cfg.update_closure_low, cfg.update_closure_high)
        gates = neuron_gate_vector(record, hidden, closure_gate, cfg)
        tensor_is_finite("neuron gates", gates)
        with torch.no_grad():
            step_reinforce_norm = 0.0
            step_novel_norm = 0.0
            step_update_norm = 0.0
            for neuron_index, gate_value_tensor in enumerate(gates):
                gate_value = float(gate_value_tensor.item())
                r_norm, n_norm, u_norm = update_with_slice(
                    first.weight[neuron_index],
                    first.weight.grad[neuron_index],
                    slow_slice(record, "net.0.weight", index=neuron_index),
                    gate_value,
                    closure_gate,
                    cfg,
                )
                step_reinforce_norm += r_norm
                step_novel_norm += n_norm
                step_update_norm += u_norm
                r_norm, n_norm, u_norm = update_with_slice(
                    first.bias[neuron_index],
                    first.bias.grad[neuron_index],
                    slow_slice(record, "net.0.bias", index=neuron_index),
                    gate_value,
                    closure_gate,
                    cfg,
                )
                step_reinforce_norm += r_norm
                step_novel_norm += n_norm
                step_update_norm += u_norm
                r_norm, n_norm, u_norm = update_with_slice(
                    second.weight[:, neuron_index],
                    second.weight.grad[:, neuron_index],
                    slow_slice(record, "net.2.weight", column=neuron_index),
                    gate_value,
                    closure_gate,
                    cfg,
                )
                step_reinforce_norm += r_norm
                step_novel_norm += n_norm
                step_update_norm += u_norm
            if second.bias.grad is None:
                raise RuntimeError(f"Missing second-layer bias gradient for {record.name}.")
            mean_gate = float(gates.mean().item())
            r_norm, n_norm, u_norm = update_with_slice(
                second.bias,
                second.bias.grad,
                slow_slice(record, "net.2.bias"),
                mean_gate,
                closure_gate,
                cfg,
            )
            step_reinforce_norm += r_norm
            step_novel_norm += n_norm
            step_update_norm += u_norm
            gate_means.append(mean_gate)
            gate_maxes.append(float(gates.max().item()))
            active_neuron_counts.append(float((gates > 0.0).sum().item()))
            reinforce_norms.append(float(np.sqrt(step_reinforce_norm)))
            novel_norms.append(float(np.sqrt(step_novel_norm)))
            update_norms.append(float(np.sqrt(step_update_norm)))
        update_grad_memory(record, cfg.grad_memory_beta)
    record.update_count += 1
    return {
        "gate_mean": float(np.mean(gate_means)),
        "gate_max": float(np.mean(gate_maxes)),
        "active_neurons": float(np.mean(active_neuron_counts)),
        "reinforce_norm_mean": float(np.mean(reinforce_norms)),
        "novel_norm_mean": float(np.mean(novel_norms)),
        "update_norm_mean": float(np.mean(update_norms)),
    }


def structural_gated_repair_update(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    first, second = operator_layers(record)
    target_code = boundary.encode(targets).detach()
    gate_means: list[float] = []
    gate_maxes: list[float] = []
    active_neuron_counts: list[float] = []
    risk_means: list[float] = []
    need_means: list[float] = []
    reinforce_norms: list[float] = []
    novel_norms: list[float] = []
    update_norms: list[float] = []
    for _ in range(cfg.update_epochs):
        record.module.zero_grad(set_to_none=True)
        codes = gather_codes(boundary, digit_indices, record.arity)
        out_code, hidden = forward_operator_with_hidden(record, codes)
        logits = boundary.decode(out_code)
        close = closure_loss(out_code, target_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * close
        tensor_is_finite("structural-gated loss", loss)
        loss.backward()
        closure_norm = close.detach().item() / distance_scale
        closure_gate = medium_band_gate(closure_norm, cfg.update_closure_low, cfg.update_closure_high)
        gates, gate_stats = structural_gate_vector(record, hidden, closure_gate, cfg)
        with torch.no_grad():
            step_reinforce_norm = 0.0
            step_novel_norm = 0.0
            step_update_norm = 0.0
            for neuron_index, gate_value_tensor in enumerate(gates):
                gate_value = float(gate_value_tensor.item())
                r_norm, n_norm, u_norm = update_with_slice(
                    first.weight[neuron_index],
                    first.weight.grad[neuron_index],
                    slow_slice(record, "net.0.weight", index=neuron_index),
                    gate_value,
                    closure_gate,
                    cfg,
                )
                step_reinforce_norm += r_norm
                step_novel_norm += n_norm
                step_update_norm += u_norm
                r_norm, n_norm, u_norm = update_with_slice(
                    first.bias[neuron_index],
                    first.bias.grad[neuron_index],
                    slow_slice(record, "net.0.bias", index=neuron_index),
                    gate_value,
                    closure_gate,
                    cfg,
                )
                step_reinforce_norm += r_norm
                step_novel_norm += n_norm
                step_update_norm += u_norm
                r_norm, n_norm, u_norm = update_with_slice(
                    second.weight[:, neuron_index],
                    second.weight.grad[:, neuron_index],
                    slow_slice(record, "net.2.weight", column=neuron_index),
                    gate_value,
                    closure_gate,
                    cfg,
                )
                step_reinforce_norm += r_norm
                step_novel_norm += n_norm
                step_update_norm += u_norm
            if second.bias.grad is None:
                raise RuntimeError(f"Missing second-layer bias gradient for {record.name}.")
            mean_gate = float(gates.mean().item())
            r_norm, n_norm, u_norm = update_with_slice(
                second.bias,
                second.bias.grad,
                slow_slice(record, "net.2.bias"),
                mean_gate,
                closure_gate,
                cfg,
            )
            step_reinforce_norm += r_norm
            step_novel_norm += n_norm
            step_update_norm += u_norm
            gate_means.append(mean_gate)
            gate_maxes.append(float(gates.max().item()))
            active_neuron_counts.append(float((gates > 0.0).sum().item()))
            risk_means.append(gate_stats["risk_mean"])
            need_means.append(gate_stats["need_mean"])
            reinforce_norms.append(float(np.sqrt(step_reinforce_norm)))
            novel_norms.append(float(np.sqrt(step_novel_norm)))
            update_norms.append(float(np.sqrt(step_update_norm)))
        update_grad_memory(record, cfg.grad_memory_beta)
    record.update_count += 1
    return {
        "gate_mean": float(np.mean(gate_means)),
        "gate_max": float(np.mean(gate_maxes)),
        "active_neurons": float(np.mean(active_neuron_counts)),
        "risk_mean": float(np.mean(risk_means)),
        "need_mean": float(np.mean(need_means)),
        "reinforce_norm_mean": float(np.mean(reinforce_norms)),
        "novel_norm_mean": float(np.mean(novel_norms)),
        "update_norm_mean": float(np.mean(update_norms)),
    }


def evaluate_direct_operator(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> dict[str, float]:
    del cfg
    with torch.no_grad():
        out_code = forward_operator_on_task(boundary, record, digit_indices)
        logits = boundary.decode(out_code)
        preds = logits.argmax(dim=-1)
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "loss": float(F.cross_entropy(logits, targets).item()),
            "closure": float(closure_loss(out_code, boundary.encode(targets)).item()),
            "manifold": float(manifold_error(out_code, boundary).item()),
        }


def generate_programs(
    library: dict[str, OperatorRecord],
    sequence_arity: int,
    max_depth: int,
    max_programs: int,
) -> list[Program]:
    vars_list: list[Program] = [("var", idx) for idx in range(sequence_arity)]
    by_depth: list[list[Program]] = [vars_list]
    programs: list[Program] = []
    seen = {program_to_str(program) for program in vars_list}
    for depth in range(1, max_depth + 1):
        depth_programs: list[Program] = []
        child_pool = [program for depth_list in by_depth for program in depth_list]
        for op_name, record in library.items():
            if record.arity == 1:
                arg_sets = [(arg,) for arg in child_pool]
            elif record.arity == 2:
                arg_sets = [(left, right) for left in child_pool for right in child_pool]
            elif record.arity == 3:
                arg_sets = [
                    (left, middle, right)
                    for left in child_pool
                    for middle in child_pool
                    for right in child_pool
                ]
            else:
                raise ValueError(f"Unsupported operator arity {record.arity} for {op_name}.")
            for args in arg_sets:
                if not any(program_depth(arg) == depth - 1 for arg in args):
                    continue
                program: Program = ("op", op_name, args)
                key = program_to_str(program)
                if key in seen:
                    continue
                seen.add(key)
                depth_programs.append(program)
                programs.append(program)
                if len(programs) > max_programs:
                    raise RuntimeError(
                        f"Program search exceeded --max-programs={max_programs}. "
                        "Increase the limit or reduce --search-depth."
                    )
        by_depth.append(depth_programs)
    return programs


def program_depth(program: Program) -> int:
    if program[0] == "var":
        return 0
    if program[0] == "op":
        return 1 + max(program_depth(arg) for arg in program[2])
    raise ValueError(f"Unknown program node type: {program[0]!r}")


def program_to_str(program: Program) -> str:
    if program[0] == "var":
        return f"x{program[1]}"
    if program[0] == "op":
        args = ", ".join(program_to_str(arg) for arg in program[2])
        return f"{program[1]}({args})"
    raise ValueError(f"Unknown program node type: {program[0]!r}")


def eval_program(
    program: Program,
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    digit_indices: torch.Tensor,
) -> torch.Tensor:
    node_type = program[0]
    if node_type == "var":
        return boundary.encode(digit_indices[:, int(program[1])])
    if node_type == "op":
        op_name = str(program[1])
        if op_name not in library:
            raise KeyError(f"Program references missing operator {op_name!r}.")
        record = library[op_name]
        args = [eval_program(arg, boundary, library, digit_indices) for arg in program[2]]
        if len(args) != record.arity:
            raise ValueError(f"Operator {op_name} expects arity {record.arity}, got {len(args)}.")
        return record.module(args)
    raise ValueError(f"Unknown program node type: {node_type!r}")


def search_best_program(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> tuple[Program | None, dict[str, float], torch.Tensor | None]:
    if not library:
        return None, {
            "accuracy": 0.0,
            "loss": float("inf"),
            "closure": float("inf"),
            "manifold": float("inf"),
            "closure_norm": float("inf"),
            "manifold_norm": float("inf"),
        }, None
    programs = generate_programs(library, digit_indices.shape[1], cfg.search_depth, cfg.max_programs)
    if not programs:
        raise RuntimeError("Program generation returned no candidates despite a non-empty library.")
    best_program: Program | None = None
    best_metrics: dict[str, float] | None = None
    best_out: torch.Tensor | None = None
    best_depth: int | None = None
    target_code = boundary.encode(targets)
    with torch.no_grad():
        for program in programs:
            out_code = eval_program(program, boundary, library, digit_indices)
            logits = boundary.decode(out_code)
            preds = logits.argmax(dim=-1)
            metrics = {
                "accuracy": float((preds == targets).float().mean().item()),
                "loss": float(F.cross_entropy(logits, targets).item()),
                "closure": float(closure_loss(out_code, target_code).item()),
                "manifold": float(manifold_error(out_code, boundary).item()),
            }
            metrics["closure_norm"] = metrics["closure"] / distance_scale
            metrics["manifold_norm"] = metrics["manifold"] / distance_scale
            if best_metrics is None:
                best_program = program
                best_metrics = metrics
                best_out = out_code.detach().clone()
                best_depth = program_depth(program)
            elif metrics["accuracy"] > best_metrics["accuracy"]:
                best_program = program
                best_metrics = metrics
                best_out = out_code.detach().clone()
                best_depth = program_depth(program)
            elif metrics["accuracy"] == best_metrics["accuracy"]:
                current_depth = program_depth(program)
                if best_depth is None or current_depth < best_depth:
                    best_program = program
                    best_metrics = metrics
                    best_out = out_code.detach().clone()
                    best_depth = current_depth
                elif current_depth == best_depth and metrics["loss"] < best_metrics["loss"]:
                    best_program = program
                    best_metrics = metrics
                    best_out = out_code.detach().clone()
                    best_depth = current_depth
    if best_metrics is None:
        raise RuntimeError("Best-program search failed to select a candidate.")
    return best_program, best_metrics, best_out


def compile_task_template(template: Program, task_to_program: dict[str, Program]) -> Program:
    if template[0] == "var":
        return template
    if template[0] != "op":
        raise ValueError(f"Unknown template node type {template[0]!r}.")
    task_name = str(template[1])
    if task_name not in task_to_program:
        raise KeyError(f"Missing task program for composition task {task_name}.")
    compiled_args = tuple(compile_task_template(arg, task_to_program) for arg in template[2])
    task_program = task_to_program[task_name]

    def bind(node: Program, bindings: tuple[Program, ...]) -> Program:
        if node[0] == "var":
            return bindings[int(node[1])]
        if node[0] == "op":
            return ("op", node[1], tuple(bind(arg, bindings) for arg in node[2]))
        raise ValueError(f"Unknown program node type {node[0]!r}.")

    return bind(task_program, compiled_args)


def evaluate_task_programs(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    stream: list[str],
    cfg: Config,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task_name in stream:
        if task_name not in task_to_program:
            raise KeyError(f"Missing program for learned task {task_name}.")
        spec = TASKS[task_name]
        digit_indices, targets = make_task_data(spec, cfg.num_values, cfg.device)
        with torch.no_grad():
            out_code = eval_program(task_to_program[task_name], boundary, library, digit_indices)
            logits = boundary.decode(out_code)
            preds = logits.argmax(dim=-1)
            metrics[f"{task_name}_acc"] = float((preds == targets).float().mean().item())
    return metrics


def evaluate_compositions(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    cfg: Config,
) -> dict[str, float]:
    comp_inputs = make_composition_data(cfg.num_values, cfg.device)
    input_np = comp_inputs.detach().cpu().numpy()
    metrics: dict[str, float] = {}
    for name, (template, target_fn) in COMPOSITIONS.items():
        try:
            program = compile_task_template(template, task_to_program)
        except KeyError:
            continue
        targets_np = target_fn(input_np, None, cfg.num_values).astype(np.int64)
        targets = torch.as_tensor(targets_np, dtype=torch.long, device=cfg.device)
        with torch.no_grad():
            out_code = eval_program(program, boundary, library, comp_inputs)
            logits = boundary.decode(out_code)
            preds = logits.argmax(dim=-1)
            metrics[f"{name}_acc"] = float((preds == targets).float().mean().item())
            metrics[f"{name}_drift"] = float(closure_loss(out_code, boundary.encode(targets)).item())
    if metrics:
        accs = [value for key, value in metrics.items() if key.endswith("_acc")]
        drifts = [value for key, value in metrics.items() if key.endswith("_drift")]
        metrics["avg_comp_acc"] = float(np.mean(accs))
        metrics["avg_comp_drift"] = float(np.mean(drifts))
    else:
        metrics["avg_comp_acc"] = float("nan")
        metrics["avg_comp_drift"] = float("nan")
    return metrics


def collect_library_drift(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    cfg: Config,
) -> dict[str, float]:
    if not library:
        return {"closure_mse": 0.0, "manifold_drift": 0.0}
    closures: list[float] = []
    manifolds: list[float] = []
    for record in library.values():
        spec = TASKS[record.origin_task]
        digit_indices, targets = make_task_data(spec, cfg.num_values, cfg.device)
        metrics = evaluate_direct_operator(boundary, record, digit_indices, targets, cfg)
        closures.append(metrics["closure"])
        manifolds.append(metrics["manifold"])
    return {"closure_mse": float(np.mean(closures)), "manifold_drift": float(np.mean(manifolds))}


def single_operator_program(record: OperatorRecord, task_arity: int) -> Program:
    if task_arity < record.arity and not (task_arity == 1 and record.arity == 2):
        raise ValueError(
            f"Cannot build direct program for operator {record.name}: "
            f"operator arity {record.arity} exceeds task arity {task_arity}."
        )
    if record.arity == 1:
        return ("op", record.name, (("var", 0),))
    if record.arity == 2:
        if task_arity == 1:
            return ("op", record.name, (("var", 0), ("var", 0)))
        return ("op", record.name, (("var", 0), ("var", 1)))
    if record.arity == 3:
        return ("op", record.name, (("var", 0), ("var", 1), ("var", 2)))
    raise ValueError(f"Unsupported record arity {record.arity}.")


def best_update_candidate(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> tuple[OperatorRecord | None, Program | None, dict[str, float], torch.Tensor | None]:
    best_record: OperatorRecord | None = None
    best_program: Program | None = None
    best_metrics: dict[str, float] | None = None
    best_out: torch.Tensor | None = None
    for record in library.values():
        program = single_operator_program(record, digit_indices.shape[1])
        with torch.no_grad():
            out_code = eval_program(program, boundary, library, digit_indices)
            logits = boundary.decode(out_code)
            preds = logits.argmax(dim=-1)
            target_code = boundary.encode(targets)
            metrics = {
                "accuracy": float((preds == targets).float().mean().item()),
                "loss": float(F.cross_entropy(logits, targets).item()),
                "closure": float(closure_loss(out_code, target_code).item()),
                "manifold": float(manifold_error(out_code, boundary).item()),
            }
            metrics["closure_norm"] = metrics["closure"] / distance_scale
            metrics["manifold_norm"] = metrics["manifold"] / distance_scale
        if best_metrics is None:
            best_record = record
            best_program = program
            best_metrics = metrics
            best_out = out_code.detach().clone()
        elif metrics["accuracy"] > best_metrics["accuracy"]:
            best_record = record
            best_program = program
            best_metrics = metrics
            best_out = out_code.detach().clone()
        elif metrics["accuracy"] == best_metrics["accuracy"] and metrics["closure"] < best_metrics["closure"]:
            best_record = record
            best_program = program
            best_metrics = metrics
            best_out = out_code.detach().clone()
    if best_metrics is None:
        return None, None, {
            "accuracy": 0.0,
            "loss": float("inf"),
            "closure": float("inf"),
            "manifold": float("inf"),
            "closure_norm": float("inf"),
            "manifold_norm": float("inf"),
        }, None
    return best_record, best_program, best_metrics, best_out


def add_decision_event(
    events: list[DecisionEvent],
    seed: int,
    policy: str,
    task: str,
    decision: str,
    best_program: Program | None,
    best_metrics: dict[str, float],
    gate_mean: float | None,
    library: dict[str, OperatorRecord],
    candidate_outputs: torch.Tensor | None,
    targets: torch.Tensor,
) -> None:
    output_list: list[list[float]] = []
    if candidate_outputs is not None:
        output_list = candidate_outputs.detach().cpu().tolist()
    events.append(
        DecisionEvent(
            seed=seed,
            policy=policy,
            task=task,
            decision=decision,
            best_program=program_to_str(best_program) if best_program is not None else None,
            best_accuracy=float(best_metrics["accuracy"]),
            best_loss=float(best_metrics["loss"]),
            closure_norm=float(best_metrics["closure_norm"]),
            manifold_norm=float(best_metrics["manifold_norm"]),
            gate_mean=gate_mean,
            operator_count=len(library),
            candidate_outputs=output_list,
            candidate_targets=targets.detach().cpu().tolist(),
        )
    )


def run_policy(
    policy: str,
    seed: int,
    stream: list[str],
    cfg: Config,
) -> tuple[dict[str, float], list[DecisionEvent], list[list[float]]]:
    set_seed(seed)
    boundary = LatentBoundary(cfg.num_values, cfg.code_dim).to(cfg.device)
    train_boundary(boundary, cfg)
    distance_scale = code_distance_scale(boundary)
    library: dict[str, OperatorRecord] = {}
    task_to_program: dict[str, Program] = {}
    events: list[DecisionEvent] = []
    false_reuse = 0
    reuse_count = 0
    allocate_count = 0
    update_count = 0
    gate_values: list[float] = []

    for task_index, task_name in enumerate(stream):
        spec = TASKS[task_name]
        digit_indices, targets = make_task_data(spec, cfg.num_values, cfg.device)
        best_prog, best_metrics, best_outputs = search_best_program(
            boundary, library, digit_indices, targets, cfg, distance_scale
        )

        if policy == "always_new_operator":
            decision = "allocate"
        elif policy == "always_update_existing_operator":
            decision = "update" if library else "allocate"
        elif policy == "frozen_admission_gated_reuse":
            if (
                best_prog is not None
                and best_metrics["accuracy"] >= cfg.reuse_acc_threshold
                and best_metrics["closure_norm"] <= cfg.reuse_closure_threshold
            ):
                decision = "reuse"
            else:
                decision = "allocate"
        elif policy == "latent_geometry_guided_optimizer":
            if (
                best_prog is not None
                and best_metrics["accuracy"] >= cfg.reuse_acc_threshold
                and best_metrics["closure_norm"] <= cfg.reuse_closure_threshold
            ):
                decision = "reuse"
            else:
                candidate, candidate_program, candidate_metrics, candidate_outputs = best_update_candidate(
                    boundary, library, digit_indices, targets, cfg, distance_scale
                )
                if (
                    candidate is not None
                    and candidate_metrics["accuracy"] >= cfg.update_min_acc
                    and cfg.update_closure_low < candidate_metrics["closure_norm"] < cfg.update_closure_high
                ):
                    decision = "update"
                    best_prog = candidate_program
                    best_metrics = candidate_metrics
                    best_outputs = candidate_outputs
                else:
                    decision = "allocate"
        else:
            raise ValueError(f"Unknown policy {policy!r}.")

        gate_mean: float | None = None
        if decision == "reuse":
            if best_prog is None:
                raise RuntimeError("Reuse decision made without a candidate program.")
            task_to_program[task_name] = best_prog
            reuse_count += 1
            if best_metrics["accuracy"] < cfg.reuse_acc_threshold:
                false_reuse += 1
        elif decision == "allocate":
            op_name = f"OP_{task_name}_{task_index}"
            record = new_operator(spec.arity, task_name, cfg, op_name)
            train_new_operator(boundary, record, digit_indices, targets, cfg)
            library[op_name] = record
            default_vars = tuple(("var", idx) for idx in range(spec.arity))
            task_to_program[task_name] = ("op", op_name, default_vars)
            allocate_count += 1
        elif decision == "update":
            if policy == "always_update_existing_operator":
                candidate, candidate_program, candidate_metrics, candidate_outputs = best_update_candidate(
                    boundary, library, digit_indices, targets, cfg, distance_scale
                )
                if candidate is None or candidate_program is None:
                    raise RuntimeError("Update decision made without an update candidate.")
                optimizer = torch.optim.Adam(candidate.module.parameters(), lr=cfg.operator_lr)
                target_code = boundary.encode(targets).detach()
                for _ in range(cfg.operator_epochs):
                    optimizer.zero_grad()
                    out_code = eval_program(candidate_program, boundary, library, digit_indices)
                    logits = boundary.decode(out_code)
                    loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * closure_loss(out_code, target_code)
                    loss.backward()
                    update_grad_memory(candidate, cfg.grad_memory_beta)
                    optimizer.step()
                task_to_program[task_name] = candidate_program
                best_prog = candidate_program
                best_metrics = candidate_metrics
                best_outputs = candidate_outputs
            else:
                candidate, candidate_program, candidate_metrics, candidate_outputs = best_update_candidate(
                    boundary, library, digit_indices, targets, cfg, distance_scale
                )
                if candidate is None or candidate_program is None:
                    raise RuntimeError("Latent-guided update decision made without an update candidate.")
                update_stats = latent_guided_update(boundary, candidate, digit_indices, targets, cfg, distance_scale)
                gate_mean = update_stats["gate_mean"]
                gate_values.append(gate_mean)
                task_to_program[task_name] = candidate_program
                best_prog = candidate_program
                best_metrics = candidate_metrics
                best_outputs = candidate_outputs
            update_count += 1
        else:
            raise RuntimeError(f"Unhandled decision {decision!r}.")

        add_decision_event(
            events,
            seed,
            policy,
            task_name,
            decision,
            best_prog,
            best_metrics,
            gate_mean,
            library,
            best_outputs,
            targets,
        )

    task_metrics = evaluate_task_programs(boundary, library, task_to_program, stream, cfg)
    comp_metrics = evaluate_compositions(boundary, library, task_to_program, cfg)
    drift_metrics = collect_library_drift(boundary, library, cfg)
    metrics: dict[str, float] = {}
    metrics.update(task_metrics)
    metrics.update(comp_metrics)
    metrics.update(drift_metrics)
    metrics["operator_count"] = float(len(library))
    metrics["new_parameters_added"] = float(sum(record.parameter_count for record in library.values()))
    metrics["false_reuse_rate"] = float(false_reuse / len(stream))
    metrics["reuse_count"] = float(reuse_count)
    metrics["allocate_count"] = float(allocate_count)
    metrics["update_count"] = float(update_count)
    metrics["mean_gate"] = float(np.mean(gate_values)) if gate_values else 0.0
    return metrics, events, boundary.codebook.detach().cpu().tolist()


def summarize(values: list[float]) -> str:
    return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"


def config_from_args(args: argparse.Namespace) -> Config:
    device = resolve_device(args.device)
    return Config(
        num_values=args.num_values,
        code_dim=args.code_dim,
        hidden_dim=args.hidden_dim,
        ae_epochs=args.ae_epochs,
        operator_epochs=args.operator_epochs,
        update_epochs=args.update_epochs,
        ae_lr=args.ae_lr,
        operator_lr=args.operator_lr,
        update_lr=args.update_lr,
        lambda_closure=args.lambda_closure,
        separation_margin=args.separation_margin,
        separation_weight=args.separation_weight,
        grad_memory_beta=args.grad_memory_beta,
        reuse_acc_threshold=args.reuse_acc_threshold,
        reuse_closure_threshold=args.reuse_closure_threshold,
        update_min_acc=args.update_min_acc,
        update_closure_low=args.update_closure_low,
        update_closure_high=args.update_closure_high,
        repair_update_rule=args.repair_update_rule,
        projection_norm_floor=args.projection_norm_floor,
        min_responsibility=args.min_responsibility,
        neuron_gate_power=args.neuron_gate_power,
        structural_risk_signal=args.structural_risk_signal,
        structural_need_signal=args.structural_need_signal,
        structural_risk_mode=args.structural_risk_mode,
        structural_risk_power=args.structural_risk_power,
        structural_need_power=args.structural_need_power,
        search_depth=args.search_depth,
        max_programs=args.max_programs,
        device=device,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    stream = parse_stream(args.stream)
    cfg = config_from_args(args)
    policies = [
        "always_new_operator",
        "always_update_existing_operator",
        "frozen_admission_gated_reuse",
        "latent_geometry_guided_optimizer",
    ]
    results: dict[str, dict[str, list[float]]] = {policy: {} for policy in policies}
    all_events: list[DecisionEvent] = []
    visual_codebook: list[list[float]] | None = None
    for seed in range(args.seed_count):
        print(f"running_seed={seed}")
        for policy in policies:
            metrics, events, codebook = run_policy(policy, seed, stream, cfg)
            if visual_codebook is None and policy == "latent_geometry_guided_optimizer":
                visual_codebook = codebook
            all_events.extend(events)
            for key, value in metrics.items():
                results[policy].setdefault(key, []).append(float(value))
    report = {
        "config": {key: value for key, value in vars(args).items() if key not in {"output_json", "output_html"}},
        "stream": stream,
        "policies": policies,
        "summary": {
            policy: {metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))} for metric, vals in metrics.items()}
            for policy, metrics in results.items()
        },
        "events": [event.__dict__ for event in all_events],
        "visual_codebook": visual_codebook,
    }
    print_summary(report, stream)
    return report


def print_summary(report: dict[str, Any], stream: list[str]) -> None:
    policies = report["policies"]
    summary = report["summary"]
    metrics = [
        "operator_count",
        "new_parameters_added",
        "reuse_count",
        "update_count",
        "allocate_count",
        "false_reuse_rate",
        "closure_mse",
        "manifold_drift",
        "avg_comp_acc",
        "avg_comp_drift",
    ]
    metrics.extend(f"{task}_acc" for task in stream)
    print("\nLATENT-GEOMETRY-GUIDED OPTIMIZER SUMMARY")
    print("=" * 92)
    print(f"{'metric':<26}" + "".join(f"{policy:<32}" for policy in policies))
    print("-" * 92)
    for metric in metrics:
        row = f"{metric:<26}"
        for policy in policies:
            if metric not in summary[policy]:
                row += f"{'n/a':<32}"
            else:
                item = summary[policy][metric]
                row += f"{item['mean']:.4f} +/- {item['std']:.4f}".ljust(32)
        print(row)
    print("=" * 92)


def average_ranks(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"average_ranks expects a 1D array, got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("average_ranks received non-finite values.")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_with_status(signal: np.ndarray, target: np.ndarray) -> tuple[float | None, str]:
    if signal.shape != target.shape:
        raise ValueError(f"Spearman inputs must have the same shape, got {signal.shape} and {target.shape}.")
    if signal.ndim != 1:
        raise ValueError(f"Spearman inputs must be 1D, got shape {signal.shape}.")
    if signal.shape[0] < 2:
        raise ValueError("Spearman correlation requires at least two observations.")
    if not np.isfinite(signal).all() or not np.isfinite(target).all():
        raise ValueError("Spearman inputs contain non-finite values.")
    if np.all(signal == signal[0]):
        return None, "undefined_constant_signal"
    if np.all(target == target[0]):
        return None, "undefined_constant_target"
    signal_ranks = average_ranks(signal)
    target_ranks = average_ranks(target)
    signal_centered = signal_ranks - signal_ranks.mean()
    target_centered = target_ranks - target_ranks.mean()
    denom = np.sqrt(np.sum(signal_centered * signal_centered) * np.sum(target_centered * target_centered))
    if denom <= 0.0:
        return None, "undefined_zero_rank_variance"
    value = float(np.sum(signal_centered * target_centered) / denom)
    if not np.isfinite(value):
        raise FloatingPointError("Spearman correlation produced a non-finite value.")
    return value, "ok"


def top_k_overlap(signal: np.ndarray, target: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError(f"top-k must be positive, got {k}.")
    if signal.shape != target.shape:
        raise ValueError(f"Top-k inputs must have the same shape, got {signal.shape} and {target.shape}.")
    if signal.ndim != 1:
        raise ValueError(f"Top-k inputs must be 1D, got shape {signal.shape}.")
    if k > signal.shape[0]:
        raise ValueError(f"top-k={k} exceeds vector length {signal.shape[0]}.")
    signal_top = set(np.argsort(-signal, kind="mergesort")[:k].tolist())
    target_top = set(np.argsort(-target, kind="mergesort")[:k].tolist())
    return float(len(signal_top.intersection(target_top)) / k)


def per_neuron_gradient_scores(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    loss_kind: str,
    cfg: Config,
) -> dict[str, np.ndarray]:
    del cfg
    if loss_kind not in {"ce", "closure", "combined"}:
        raise ValueError(f"Unknown loss_kind {loss_kind!r}. Expected ce, closure, or combined.")
    record.module.zero_grad(set_to_none=True)
    codes = gather_codes(boundary, digit_indices, record.arity)
    out_code, hidden = forward_operator_with_hidden(record, codes)
    logits = boundary.decode(out_code)
    target_code = boundary.encode(targets).detach()
    ce = F.cross_entropy(logits, targets)
    close = closure_loss(out_code, target_code)
    if loss_kind == "ce":
        loss = ce
    elif loss_kind == "closure":
        loss = close
    else:
        loss = ce + close
    loss.backward()
    first, second = operator_layers(record)
    if first.weight.grad is None or first.bias.grad is None or second.weight.grad is None:
        raise RuntimeError(f"Missing gradients for per-neuron diagnostic on {record.name}.")
    activation = hidden.detach().abs().mean(dim=0)
    incoming_grad = first.weight.grad.detach().norm(dim=1) + first.bias.grad.detach().abs()
    outgoing_grad = second.weight.grad.detach().norm(dim=0)
    total_grad = incoming_grad + outgoing_grad
    responsibility = activation * total_grad
    tensors = {
        f"{loss_kind}_gradient": total_grad,
        f"{loss_kind}_responsibility": responsibility,
    }
    result: dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        tensor_is_finite(name, tensor)
        result[name] = tensor.detach().cpu().numpy().astype(np.float64)
    record.module.zero_grad(set_to_none=True)
    return result


def evaluate_with_hidden_ablation(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    neuron_index: int,
) -> dict[str, float]:
    with torch.no_grad():
        codes = gather_codes(boundary, digit_indices, record.arity)
        first, second = operator_layers(record)
        hidden = F.relu(first(torch.cat(codes, dim=-1)))
        if neuron_index < 0 or neuron_index >= hidden.shape[1]:
            raise IndexError(f"neuron_index={neuron_index} outside hidden width {hidden.shape[1]}.")
        hidden[:, neuron_index] = 0.0
        out_code = second(hidden)
        logits = boundary.decode(out_code)
        preds = logits.argmax(dim=-1)
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "loss": float(F.cross_entropy(logits, targets).item()),
            "closure": float(closure_loss(out_code, boundary.encode(targets)).item()),
            "manifold": float(manifold_error(out_code, boundary).item()),
        }


def geometry_signals_for_operator(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    base_metrics = evaluate_direct_operator(boundary, record, digit_indices, targets, cfg)
    codes = gather_codes(boundary, digit_indices, record.arity)
    with torch.no_grad():
        _, hidden = forward_operator_with_hidden(record, codes)
        activation = hidden.abs().mean(dim=0)
        first, second = operator_layers(record)
        incoming_norm = first.weight.detach().norm(dim=1) + first.bias.detach().abs()
        downstream_norm = second.weight.detach().norm(dim=0)
        weight_product = incoming_norm * downstream_norm
        activation_downstream = activation * downstream_norm
        activation_weight_product = activation * weight_product
    signals: dict[str, np.ndarray] = {
        "activation": activation.detach().cpu().numpy().astype(np.float64),
        "incoming_weight_norm": incoming_norm.detach().cpu().numpy().astype(np.float64),
        "downstream_weight_norm": downstream_norm.detach().cpu().numpy().astype(np.float64),
        "weight_product": weight_product.detach().cpu().numpy().astype(np.float64),
        "activation_downstream": activation_downstream.detach().cpu().numpy().astype(np.float64),
        "activation_weight_product": activation_weight_product.detach().cpu().numpy().astype(np.float64),
    }
    for loss_kind in ("ce", "closure", "combined"):
        signals.update(per_neuron_gradient_scores(boundary, record, digit_indices, targets, loss_kind, cfg))

    ground_truth: dict[str, list[float]] = {
        "loss_damage": [],
        "closure_damage": [],
        "accuracy_damage": [],
        "manifold_damage": [],
    }
    for neuron_index in range(cfg.hidden_dim):
        ablated = evaluate_with_hidden_ablation(boundary, record, digit_indices, targets, neuron_index)
        ground_truth["loss_damage"].append(max(0.0, ablated["loss"] - base_metrics["loss"]))
        ground_truth["closure_damage"].append(max(0.0, ablated["closure"] - base_metrics["closure"]))
        ground_truth["accuracy_damage"].append(max(0.0, base_metrics["accuracy"] - ablated["accuracy"]))
        ground_truth["manifold_damage"].append(max(0.0, ablated["manifold"] - base_metrics["manifold"]))
    ground_truth_arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in ground_truth.items()
    }
    return signals, ground_truth_arrays, base_metrics


def run_geometry_signal_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    task_name = args.diagnostic_task.upper()
    if task_name not in TASKS:
        raise ValueError(f"Unknown --diagnostic-task {task_name!r}. Available: {sorted(TASKS)}")
    cfg = config_from_args(args)
    task = TASKS[task_name]
    per_seed: list[dict[str, Any]] = []
    for seed in range(args.seed_count):
        print(f"running_seed={seed}")
        set_seed(seed)
        boundary = LatentBoundary(cfg.num_values, cfg.code_dim).to(cfg.device)
        train_boundary(boundary, cfg)
        record = new_operator(task.arity, task.name, cfg, f"OP_{task.name}_diagnostic")
        digit_indices, targets = make_task_data(task, cfg.num_values, cfg.device)
        train_new_operator(boundary, record, digit_indices, targets, cfg)
        if args.diagnostic_noise_std > 0.0:
            perturb_operator(record, args.diagnostic_noise_std, seed=seed + 50_000)
        signals, ground_truths, base_metrics = geometry_signals_for_operator(
            boundary, record, digit_indices, targets, cfg
        )
        comparisons: dict[str, dict[str, dict[str, float | str | None]]] = {}
        for target_name, target_values in ground_truths.items():
            comparisons[target_name] = {}
            for signal_name, signal_values in signals.items():
                rho, status = spearman_with_status(signal_values, target_values)
                comparisons[target_name][signal_name] = {
                    "spearman": rho,
                    "status": status,
                    "top_k_overlap": top_k_overlap(signal_values, target_values, args.diagnostic_top_k),
                }
        per_seed.append(
            {
                "seed": seed,
                "task": task_name,
                "base_metrics": base_metrics,
                "signals": {name: values.tolist() for name, values in signals.items()},
                "ground_truths": {name: values.tolist() for name, values in ground_truths.items()},
                "comparisons": comparisons,
            }
        )
    summary = summarize_geometry_diagnostics(per_seed)
    report = {
        "mode": "geometry_signal_diagnostics",
        "config": {key: value for key, value in vars(args).items() if key not in {"output_json", "output_html"}},
        "summary": summary,
        "seeds": per_seed,
    }
    print_geometry_diagnostic_summary(report)
    return report


def summarize_geometry_diagnostics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_seed:
        raise ValueError("No per-seed diagnostic rows were produced.")
    target_names = sorted(per_seed[0]["comparisons"].keys())
    signal_names = sorted(next(iter(per_seed[0]["comparisons"].values())).keys())
    summary: dict[str, Any] = {}
    for target_name in target_names:
        summary[target_name] = {}
        for signal_name in signal_names:
            spearman_values: list[float] = []
            top_k_values: list[float] = []
            statuses: dict[str, int] = {}
            for row in per_seed:
                item = row["comparisons"][target_name][signal_name]
                statuses[item["status"]] = statuses.get(item["status"], 0) + 1
                if item["spearman"] is not None:
                    spearman_values.append(float(item["spearman"]))
                top_k_values.append(float(item["top_k_overlap"]))
            summary[target_name][signal_name] = {
                "spearman_mean": float(np.mean(spearman_values)) if spearman_values else None,
                "spearman_std": float(np.std(spearman_values)) if spearman_values else None,
                "spearman_defined_count": len(spearman_values),
                "top_k_overlap_mean": float(np.mean(top_k_values)),
                "top_k_overlap_std": float(np.std(top_k_values)),
                "statuses": statuses,
            }
    return summary


def print_geometry_diagnostic_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\nGEOMETRY SIGNAL DIAGNOSTICS")
    print("=" * 116)
    for target_name, signal_items in summary.items():
        print(f"\nGround truth target: {target_name}")
        print(f"{'signal':<34} {'spearman':<24} {'defined':<9} {'top_k_overlap':<18} statuses")
        print("-" * 116)
        for signal_name, item in signal_items.items():
            if item["spearman_mean"] is None:
                spearman_text = "n/a"
            else:
                spearman_text = f"{item['spearman_mean']:.4f} +/- {item['spearman_std']:.4f}"
            topk_text = f"{item['top_k_overlap_mean']:.4f} +/- {item['top_k_overlap_std']:.4f}"
            print(
                f"{signal_name:<34} {spearman_text:<24} "
                f"{item['spearman_defined_count']:<9} {topk_text:<18} {item['statuses']}"
            )
    print("=" * 116)


def write_json_report(report: dict[str, Any], output_json: Path) -> None:
    if output_json.exists():
        raise FileExistsError(f"output-json already exists: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_html_report(report: dict[str, Any], output_html: Path) -> None:
    if output_html.exists():
        raise FileExistsError(f"output-html already exists: {output_html}")
    import plotly.graph_objects as go

    output_html.parent.mkdir(parents=True, exist_ok=True)
    codebook = np.asarray(report["visual_codebook"], dtype=np.float64)
    if codebook.ndim != 2:
        raise ValueError("visual_codebook must be a 2D array.")
    vectors = [codebook]
    labels: list[str] = [f"code_{idx}" for idx in range(codebook.shape[0])]
    kinds: list[str] = ["code"] * codebook.shape[0]
    event_policy = "latent_geometry_guided_optimizer"
    for event in report["events"]:
        if event["policy"] != event_policy or event["seed"] != 0:
            continue
        outputs = np.asarray(event["candidate_outputs"], dtype=np.float64)
        if outputs.size == 0:
            continue
        vectors.append(outputs)
        labels.extend([f"{event['task']} {event['decision']}" for _ in range(outputs.shape[0])])
        kinds.extend([event["decision"] for _ in range(outputs.shape[0])])
    matrix = np.vstack(vectors)
    coords = project_svd_3d(matrix)
    color_by_kind = {
        "code": "#111827",
        "reuse": "#1d4ed8",
        "allocate": "#dc2626",
        "update": "#ca8a04",
    }
    fig = go.Figure()
    for kind in sorted(set(kinds)):
        indices = [idx for idx, item in enumerate(kinds) if item == kind]
        fig.add_trace(
            go.Scatter3d(
                x=coords[indices, 0],
                y=coords[indices, 1],
                z=coords[indices, 2],
                mode="markers",
                name=kind,
                text=[labels[idx] for idx in indices],
                marker={"size": 5 if kind != "code" else 8, "color": color_by_kind.get(kind, "#6b7280")},
            )
        )
    fig.update_layout(
        title="Latent Geometry Guided Optimizer Decisions (seed 0)",
        scene={"xaxis_title": "SVD-1", "yaxis_title": "SVD-2", "zaxis_title": "SVD-3"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        height=900,
    )
    fig.write_html(output_html, include_plotlyjs=True, full_html=True)


def project_svd_3d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    dims = min(3, vh.shape[0])
    coords = centered @ vh[:dims].T
    if dims < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - dims)))
    return coords


COUNTERFACTUAL_STREAM = [
    "ADD",
    "MAX",
    "COPY",
    "NOISY_ADD_REPAIR",
    "DOUBLE_ADD",
    "ADD_PLUS_ONE",
    "MIN",
    "SUB",
]

EXPECTED_COUNTERFACTUAL_ACTION = {
    "ADD": "allocate",
    "MAX": "allocate",
    "COPY": "reuse",
    "NOISY_ADD_REPAIR": "update",
    "DOUBLE_ADD": "compose",
    "ADD_PLUS_ONE": "allocate",
    "MIN": "allocate",
    "SUB": "allocate",
}


def make_counterfactual_task_data(
    task_name: str,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if task_name in TASKS:
        spec = TASKS[task_name]
        digit_indices, targets = make_task_data(spec, cfg.num_values, cfg.device)
        return digit_indices, targets, spec.arity
    if task_name == "NOISY_ADD_REPAIR":
        spec = TASKS["ADD"]
        digit_indices, targets = make_task_data(spec, cfg.num_values, cfg.device)
        return digit_indices, targets, spec.arity
    if task_name == "ADD_PLUS_ONE":
        a, b = np.meshgrid(np.arange(cfg.num_values), np.arange(cfg.num_values), indexing="ij")
        inputs = np.stack([a.reshape(-1), b.reshape(-1)], axis=1).astype(np.int64)
        targets = (inputs[:, 0] + inputs[:, 1] + 1) % cfg.num_values
        return (
            torch.as_tensor(inputs, dtype=torch.long, device=cfg.device),
            torch.as_tensor(targets, dtype=torch.long, device=cfg.device),
            2,
        )
    if task_name == "DOUBLE_ADD":
        d0, d1, d2 = np.meshgrid(
            np.arange(cfg.num_values),
            np.arange(cfg.num_values),
            np.arange(cfg.num_values),
            indexing="ij",
        )
        inputs = np.stack([d0.reshape(-1), d1.reshape(-1), d2.reshape(-1)], axis=1).astype(np.int64)
        targets = (inputs[:, 0] + inputs[:, 1] + inputs[:, 2]) % cfg.num_values
        return (
            torch.as_tensor(inputs, dtype=torch.long, device=cfg.device),
            torch.as_tensor(targets, dtype=torch.long, device=cfg.device),
            3,
        )
    raise ValueError(f"Unknown counterfactual task {task_name!r}.")


def clone_library(library: dict[str, OperatorRecord]) -> dict[str, OperatorRecord]:
    cloned: dict[str, OperatorRecord] = {}
    for name, record in library.items():
        cloned[name] = OperatorRecord(
            name=record.name,
            arity=record.arity,
            module=copy.deepcopy(record.module),
            origin_task=record.origin_task,
            grad_memory={key: value.detach().clone() for key, value in record.grad_memory.items()},
            parameter_count=record.parameter_count,
            update_count=record.update_count,
        )
    return cloned


def perturb_operator(record: OperatorRecord, noise_std: float, seed: int) -> None:
    if noise_std <= 0.0:
        raise ValueError(f"repair noise must be positive, got {noise_std}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in record.module.parameters():
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            parameter.add_(noise.to(device=parameter.device, dtype=parameter.dtype), alpha=noise_std)


def direct_root_operator(program: Program) -> str:
    if program[0] != "op":
        raise ValueError(f"Expected direct operator program, got {program_to_str(program)}.")
    for arg in program[2]:
        if arg[0] != "var":
            raise ValueError(f"Expected direct operator program, got {program_to_str(program)}.")
    return str(program[1])


def evaluate_program_metrics(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    program: Program,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    distance_scale: float,
) -> dict[str, float]:
    with torch.no_grad():
        out_code = eval_program(program, boundary, library, digit_indices)
        logits = boundary.decode(out_code)
        preds = logits.argmax(dim=-1)
        close = closure_loss(out_code, boundary.encode(targets)).item()
        manifold = manifold_error(out_code, boundary).item()
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "loss": float(F.cross_entropy(logits, targets).item()),
            "closure": float(close),
            "manifold": float(manifold),
            "closure_norm": float(close / distance_scale),
            "manifold_norm": float(manifold / distance_scale),
        }


def evaluate_counterfactual_tasks(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    task_names: list[str],
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task_name in task_names:
        if task_name not in task_to_program:
            raise KeyError(f"Missing task program for {task_name}.")
        digit_indices, targets, _ = make_counterfactual_task_data(task_name, cfg)
        task_metrics = evaluate_program_metrics(
            boundary,
            library,
            task_to_program[task_name],
            digit_indices,
            targets,
            distance_scale,
        )
        metrics[f"{task_name}_acc"] = task_metrics["accuracy"]
        metrics[f"{task_name}_closure_norm"] = task_metrics["closure_norm"]
    return metrics


def safe_mean(values: list[float], default: float) -> float:
    if not values:
        return default
    return float(np.mean(values))


def safe_min(values: list[float], default: float) -> float:
    if not values:
        return default
    return float(np.min(values))


def evaluate_counterfactual_state(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    learned_tasks: list[str],
    current_task: str,
    current_program: Program,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    digit_indices, targets, _ = make_counterfactual_task_data(current_task, cfg)
    current_metrics = evaluate_program_metrics(
        boundary,
        library,
        current_program,
        digit_indices,
        targets,
        distance_scale,
    )
    old_metrics = evaluate_counterfactual_tasks(
        boundary,
        library,
        task_to_program,
        learned_tasks,
        cfg,
        distance_scale,
    )
    old_accs = [value for key, value in old_metrics.items() if key.endswith("_acc")]
    old_closures = [value for key, value in old_metrics.items() if key.endswith("_closure_norm")]
    comp_metrics = evaluate_compositions(boundary, library, task_to_program, cfg)
    comp_acc = comp_metrics["avg_comp_acc"]
    comp_drift = comp_metrics["avg_comp_drift"]
    if np.isnan(comp_acc):
        comp_acc = 1.0
    if np.isnan(comp_drift):
        comp_drift = 0.0
    return {
        "new_acc": current_metrics["accuracy"],
        "new_loss": current_metrics["loss"],
        "new_closure_norm": current_metrics["closure_norm"],
        "new_manifold_norm": current_metrics["manifold_norm"],
        "old_min_acc": safe_min(old_accs, 1.0),
        "old_mean_acc": safe_mean(old_accs, 1.0),
        "old_mean_closure_norm": safe_mean(old_closures, 0.0),
        "composition_acc": float(comp_acc),
        "composition_drift": float(comp_drift),
    }


def counterfactual_score(
    state_metrics: dict[str, float],
    action: str,
    new_parameters: int,
    cfg: Config,
) -> tuple[float, bool]:
    closure_limit = cfg.reuse_closure_threshold if action in {"reuse", "compose"} else cfg.update_closure_high
    safe = (
        state_metrics["new_acc"] >= cfg.reuse_acc_threshold
        and state_metrics["old_min_acc"] >= cfg.reuse_acc_threshold
        and state_metrics["composition_acc"] >= cfg.reuse_acc_threshold
        and state_metrics["old_mean_closure_norm"] <= cfg.reuse_closure_threshold
        and state_metrics["new_closure_norm"] <= closure_limit
    )
    action_penalty = {
        "reuse": 0.00,
        "compose": 0.00,
        "update": 0.01,
        "allocate": 0.03,
    }[action]
    parameter_penalty = new_parameters / 100_000.0
    score = (
        3.0 * state_metrics["new_acc"]
        + 2.0 * state_metrics["old_min_acc"]
        + state_metrics["composition_acc"]
        - 0.10 * state_metrics["new_closure_norm"]
        - 0.05 * state_metrics["old_mean_closure_norm"]
        - action_penalty
        - parameter_penalty
    )
    if not safe:
        score -= 10.0
    return float(score), bool(safe)


def candidate_row(
    action: str,
    program: Program,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    metrics: dict[str, float],
    score: float,
    safe: bool,
    new_parameters: int,
    gate_mean: float | None = None,
    update_stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "program": program_to_str(program),
        "library": library,
        "task_to_program": task_to_program,
        "metrics": metrics,
        "score": score,
        "safe": safe,
        "new_parameters": new_parameters,
        "gate_mean": gate_mean,
        "update_stats": update_stats or {},
    }


def build_counterfactual_candidates(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    learned_tasks: list[str],
    task_name: str,
    cfg: Config,
    distance_scale: float,
) -> list[dict[str, Any]]:
    digit_indices, targets, arity = make_counterfactual_task_data(task_name, cfg)
    candidates: list[dict[str, Any]] = []

    best_prog, _, _ = search_best_program(boundary, library, digit_indices, targets, cfg, distance_scale)
    if best_prog is not None:
        action = "compose" if program_depth(best_prog) > 1 else "reuse"
        shadow_task_to_program = dict(task_to_program)
        shadow_task_to_program[task_name] = best_prog
        metrics = evaluate_counterfactual_state(
            boundary,
            library,
            shadow_task_to_program,
            learned_tasks,
            task_name,
            best_prog,
            cfg,
            distance_scale,
        )
        score, safe = counterfactual_score(metrics, action, 0, cfg)
        candidates.append(candidate_row(action, best_prog, library, shadow_task_to_program, metrics, score, safe, 0))

    for record in library.values():
        if record.arity != arity:
            continue
        shadow_library = clone_library(library)
        shadow_record = shadow_library[record.name]
        if cfg.repair_update_rule == "adam":
            adam_repair_update(
                boundary,
                shadow_record,
                digit_indices,
                targets,
                cfg,
            )
            update_stats = {"gate_mean": None}
        elif cfg.repair_update_rule == "neuron_gated":
            update_stats = neuron_gated_repair_update(
                boundary,
                shadow_record,
                digit_indices,
                targets,
                cfg,
                distance_scale,
            )
        elif cfg.repair_update_rule == "structural_gated":
            update_stats = structural_gated_repair_update(
                boundary,
                shadow_record,
                digit_indices,
                targets,
                cfg,
                distance_scale,
            )
        else:
            raise ValueError(f"Unknown repair_update_rule={cfg.repair_update_rule!r}.")
        program = single_operator_program(shadow_record, arity)
        shadow_task_to_program = dict(task_to_program)
        shadow_task_to_program[task_name] = program
        metrics = evaluate_counterfactual_state(
            boundary,
            shadow_library,
            shadow_task_to_program,
            learned_tasks,
            task_name,
            program,
            cfg,
            distance_scale,
        )
        score, safe = counterfactual_score(metrics, "update", 0, cfg)
        candidates.append(
            candidate_row(
                "update",
                program,
                shadow_library,
                shadow_task_to_program,
                metrics,
                score,
                safe,
                0,
                gate_mean=update_stats["gate_mean"],
                update_stats=update_stats,
            )
        )

    shadow_library = clone_library(library)
    op_name = f"OP_{task_name}_{len(library)}"
    record = new_operator(arity, task_name, cfg, op_name)
    train_new_operator(boundary, record, digit_indices, targets, cfg, require_perfect=False)
    shadow_library[op_name] = record
    program = ("op", op_name, tuple(("var", idx) for idx in range(arity)))
    shadow_task_to_program = dict(task_to_program)
    shadow_task_to_program[task_name] = program
    metrics = evaluate_counterfactual_state(
        boundary,
        shadow_library,
        shadow_task_to_program,
        learned_tasks,
        task_name,
        program,
        cfg,
        distance_scale,
    )
    score, safe = counterfactual_score(metrics, "allocate", record.parameter_count, cfg)
    candidates.append(
        candidate_row(
            "allocate",
            program,
            shadow_library,
            shadow_task_to_program,
            metrics,
            score,
            safe,
            record.parameter_count,
        )
    )

    return candidates


def choose_counterfactual_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No counterfactual candidates were generated.")
    return max(candidates, key=lambda item: item["score"])


FORCED_CONFLICT_RULES = {
    "adam",
    "neuron_gated",
    "structural_repair",
    "structural_protect",
}


def config_for_forced_rule(cfg: Config, rule: str) -> Config:
    if rule == "adam":
        return replace(cfg, repair_update_rule="adam")
    if rule == "neuron_gated":
        return replace(cfg, repair_update_rule="neuron_gated")
    if rule == "structural_repair":
        return replace(cfg, repair_update_rule="structural_gated", structural_risk_mode="repair")
    if rule == "structural_protect":
        return replace(cfg, repair_update_rule="structural_gated", structural_risk_mode="protect")
    raise ValueError(f"Unknown forced conflict rule {rule!r}. Available: {sorted(FORCED_CONFLICT_RULES)}")


def apply_repair_update_rule(
    boundary: LatentBoundary,
    record: OperatorRecord,
    digit_indices: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float | None]:
    if cfg.repair_update_rule == "adam":
        adam_repair_update(
            boundary,
            record,
            digit_indices,
            targets,
            cfg,
        )
        return {"gate_mean": None}
    if cfg.repair_update_rule == "neuron_gated":
        return neuron_gated_repair_update(
            boundary,
            record,
            digit_indices,
            targets,
            cfg,
            distance_scale,
        )
    if cfg.repair_update_rule == "structural_gated":
        return structural_gated_repair_update(
            boundary,
            record,
            digit_indices,
            targets,
            cfg,
            distance_scale,
        )
    raise ValueError(f"Unknown repair_update_rule={cfg.repair_update_rule!r}.")


def evaluate_double_add_with_record(
    boundary: LatentBoundary,
    record: OperatorRecord,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    if record.arity != 2:
        raise ValueError(f"DOUBLE_ADD evaluation requires a binary operator, got arity={record.arity}.")
    digit_indices, targets, _ = make_counterfactual_task_data("DOUBLE_ADD", cfg)
    program: Program = (
        "op",
        record.name,
        (("op", record.name, (("var", 0), ("var", 1))), ("var", 2)),
    )
    return evaluate_program_metrics(boundary, {record.name: record}, program, digit_indices, targets, distance_scale)


def clone_operator_record(record: OperatorRecord) -> OperatorRecord:
    return clone_library({record.name: record})[record.name]


def run_forced_conflict_update(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_from_args(args)
    rules = parse_csv_items(args.forced_conflict_rules, "--forced-conflict-rules")
    unknown_rules = [rule for rule in rules if rule not in FORCED_CONFLICT_RULES]
    if unknown_rules:
        raise ValueError(f"Unknown forced conflict rule(s): {unknown_rules}. Available: {sorted(FORCED_CONFLICT_RULES)}")
    conflict_task = args.conflict_task.upper()
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_count):
        print(f"running_seed={seed}")
        set_seed(seed)
        boundary = LatentBoundary(cfg.num_values, cfg.code_dim).to(cfg.device)
        train_boundary(boundary, cfg)
        distance_scale = code_distance_scale(boundary)
        add_inputs, add_targets = make_task_data(TASKS["ADD"], cfg.num_values, cfg.device)
        conflict_inputs, conflict_targets, conflict_arity = make_counterfactual_task_data(conflict_task, cfg)
        if conflict_arity != TASKS["ADD"].arity:
            raise ValueError(
                f"--forced-conflict-update currently requires a binary conflict task; "
                f"{conflict_task} has arity={conflict_arity}."
            )
        base_record = new_operator(TASKS["ADD"].arity, "ADD", cfg, "OP_ADD_0")
        train_new_operator(boundary, base_record, add_inputs, add_targets, cfg)
        base_old = evaluate_direct_operator(boundary, base_record, add_inputs, add_targets, cfg)
        base_conflict = evaluate_direct_operator(boundary, base_record, conflict_inputs, conflict_targets, cfg)
        base_double = evaluate_double_add_with_record(boundary, base_record, cfg, distance_scale)

        for rule in rules:
            rule_cfg = config_for_forced_rule(cfg, rule)
            record = clone_operator_record(base_record)
            update_stats = apply_repair_update_rule(
                boundary,
                record,
                conflict_inputs,
                conflict_targets,
                rule_cfg,
                distance_scale,
            )
            old_after = evaluate_direct_operator(boundary, record, add_inputs, add_targets, cfg)
            conflict_after = evaluate_direct_operator(boundary, record, conflict_inputs, conflict_targets, cfg)
            double_after = evaluate_double_add_with_record(boundary, record, cfg, distance_scale)
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "conflict_task": conflict_task,
                    "base_old_acc": base_old["accuracy"],
                    "base_old_closure_norm": base_old["closure"] / distance_scale,
                    "base_conflict_acc": base_conflict["accuracy"],
                    "base_conflict_closure_norm": base_conflict["closure"] / distance_scale,
                    "base_double_add_acc": base_double["accuracy"],
                    "base_double_add_closure_norm": base_double["closure_norm"],
                    "old_acc": old_after["accuracy"],
                    "old_loss": old_after["loss"],
                    "old_closure_norm": old_after["closure"] / distance_scale,
                    "new_acc": conflict_after["accuracy"],
                    "new_loss": conflict_after["loss"],
                    "new_closure_norm": conflict_after["closure"] / distance_scale,
                    "double_add_acc": double_after["accuracy"],
                    "double_add_closure_norm": double_after["closure_norm"],
                    "forgetting": max(0.0, base_old["accuracy"] - old_after["accuracy"]),
                    "new_gain": max(0.0, conflict_after["accuracy"] - base_conflict["accuracy"]),
                    "update_stats": update_stats,
                }
            )
    report = {
        "mode": "forced_conflict_update",
        "config": {key: value for key, value in vars(args).items() if key not in {"output_json", "output_html"}},
        "summary": summarize_forced_conflict(rows),
        "rows": rows,
    }
    print_forced_conflict_summary(report)
    return report


def summarize_forced_conflict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No forced-conflict rows were produced.")
    rules = sorted({row["rule"] for row in rows})
    metrics = [
        "old_acc",
        "new_acc",
        "double_add_acc",
        "forgetting",
        "new_gain",
        "old_closure_norm",
        "new_closure_norm",
        "double_add_closure_norm",
    ]
    summary: dict[str, Any] = {}
    for rule in rules:
        rule_rows = [row for row in rows if row["rule"] == rule]
        summary[rule] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in rule_rows]
            summary[rule][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return summary


def print_forced_conflict_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    metrics = [
        "old_acc",
        "new_acc",
        "double_add_acc",
        "forgetting",
        "new_gain",
        "old_closure_norm",
        "new_closure_norm",
        "double_add_closure_norm",
    ]
    print("\nFORCED CONFLICT UPDATE SUMMARY")
    print("=" * 120)
    print(f"{'metric':<28}" + "".join(f"{rule:<24}" for rule in sorted(summary)))
    print("-" * 120)
    for metric in metrics:
        row = f"{metric:<28}"
        for rule in sorted(summary):
            item = summary[rule][metric]
            row += f"{item['mean']:.4f} +/- {item['std']:.4f}".ljust(24)
        print(row)
    print("=" * 120)


def run_counterfactual_action_selection(args: argparse.Namespace) -> dict[str, Any]:
    if args.search_depth < 2:
        raise ValueError("--counterfactual-action-selection requires --search-depth 2 or higher.")
    cfg = config_from_args(args)
    ledgers: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    visual_codebook: list[list[float]] | None = None

    for seed in range(args.seed_count):
        print(f"running_seed={seed}")
        set_seed(seed)
        boundary = LatentBoundary(cfg.num_values, cfg.code_dim).to(cfg.device)
        train_boundary(boundary, cfg)
        if visual_codebook is None:
            visual_codebook = boundary.codebook.detach().cpu().tolist()
        distance_scale = code_distance_scale(boundary)
        library: dict[str, OperatorRecord] = {}
        task_to_program: dict[str, Program] = {}
        learned_tasks: list[str] = []

        for task_name in COUNTERFACTUAL_STREAM:
            perturb_info: dict[str, float] = {}
            if task_name == "NOISY_ADD_REPAIR":
                if "ADD" not in task_to_program:
                    raise RuntimeError("NOISY_ADD_REPAIR requires ADD to have been learned first.")
                add_op_name = direct_root_operator(task_to_program["ADD"])
                if add_op_name not in library:
                    raise RuntimeError(f"ADD program references missing operator {add_op_name}.")
                digit_indices, targets, _ = make_counterfactual_task_data("ADD", cfg)
                before = evaluate_program_metrics(
                    boundary,
                    library,
                    task_to_program["ADD"],
                    digit_indices,
                    targets,
                    distance_scale,
                )
                perturb_operator(library[add_op_name], args.repair_noise_std, seed=seed + 10_000)
                after = evaluate_program_metrics(
                    boundary,
                    library,
                    task_to_program["ADD"],
                    digit_indices,
                    targets,
                    distance_scale,
                )
                perturb_info = {
                    "add_acc_before_perturb": before["accuracy"],
                    "add_acc_after_perturb": after["accuracy"],
                    "add_closure_norm_after_perturb": after["closure_norm"],
                }
                if after["accuracy"] >= cfg.reuse_acc_threshold and after["closure_norm"] <= cfg.reuse_closure_threshold:
                    raise RuntimeError(
                        "Repair perturbation did not create a repair case. "
                        f"ADD acc={after['accuracy']:.4f}, closure_norm={after['closure_norm']:.4f}. "
                        "Increase --repair-noise-std."
                    )

            candidates = build_counterfactual_candidates(
                boundary,
                library,
                task_to_program,
                learned_tasks,
                task_name,
                cfg,
                distance_scale,
            )
            chosen = choose_counterfactual_candidate(candidates)
            library = chosen["library"]
            task_to_program = chosen["task_to_program"]
            learned_tasks.append(task_name)
            expected = EXPECTED_COUNTERFACTUAL_ACTION[task_name]
            correct = chosen["action"] == expected
            ledger_row = {
                "seed": seed,
                "task": task_name,
                "expected": expected,
                "chosen": chosen["action"],
                "correct": correct,
                "program": chosen["program"],
                "safe": chosen["safe"],
                "score": chosen["score"],
                "new_parameters": chosen["new_parameters"],
                "operator_count": len(library),
                **chosen["metrics"],
                **perturb_info,
                "candidate_actions": [
                    {
                        "action": candidate["action"],
                        "program": candidate["program"],
                        "safe": candidate["safe"],
                        "score": candidate["score"],
                        "new_acc": candidate["metrics"]["new_acc"],
                        "old_min_acc": candidate["metrics"]["old_min_acc"],
                        "new_closure_norm": candidate["metrics"]["new_closure_norm"],
                        "new_parameters": candidate["new_parameters"],
                        "gate_mean": candidate["gate_mean"],
                        "update_stats": candidate["update_stats"],
                    }
                    for candidate in candidates
                ],
            }
            ledgers.append(ledger_row)

            digit_indices, targets, _ = make_counterfactual_task_data(task_name, cfg)
            with torch.no_grad():
                out_code = eval_program(task_to_program[task_name], boundary, library, digit_indices)
            events.append(
                {
                    "seed": seed,
                    "policy": "latent_geometry_guided_optimizer",
                    "task": task_name,
                    "decision": chosen["action"],
                    "best_program": chosen["program"],
                    "best_accuracy": chosen["metrics"]["new_acc"],
                    "best_loss": chosen["metrics"]["new_loss"],
                    "closure_norm": chosen["metrics"]["new_closure_norm"],
                    "manifold_norm": chosen["metrics"]["new_manifold_norm"],
                    "gate_mean": chosen["gate_mean"],
                    "operator_count": len(library),
                    "candidate_outputs": out_code.detach().cpu().tolist(),
                    "candidate_targets": targets.detach().cpu().tolist(),
                }
            )

    print_counterfactual_summary(ledgers)
    decision_accuracy = float(np.mean([row["correct"] for row in ledgers]))
    destructive_updates = [
        row for row in ledgers
        if row["chosen"] == "update" and row["old_min_acc"] < args.reuse_acc_threshold
    ]
    unnecessary_allocations = [
        row for row in ledgers
        if row["chosen"] == "allocate" and row["expected"] != "allocate"
    ]
    return {
        "mode": "counterfactual_action_selection",
        "config": {key: value for key, value in vars(args).items() if key not in {"output_json", "output_html"}},
        "stream": COUNTERFACTUAL_STREAM,
        "expected_actions": EXPECTED_COUNTERFACTUAL_ACTION,
        "decision_accuracy": decision_accuracy,
        "destructive_update_count": len(destructive_updates),
        "unnecessary_allocation_count": len(unnecessary_allocations),
        "ledger": ledgers,
        "events": events,
        "visual_codebook": visual_codebook,
        "policies": ["latent_geometry_guided_optimizer"],
    }


def print_counterfactual_summary(ledgers: list[dict[str, Any]]) -> None:
    print("\nCOUNTERFACTUAL ACTION-SELECTION LEDGER")
    print("=" * 150)
    print(
        f"{'seed':<5} {'task':<20} {'expected':<10} {'chosen':<10} {'ok':<4} "
        f"{'safe':<5} {'new_acc':<8} {'old_min':<8} {'comp':<8} {'closure':<9} {'ops':<4} program"
    )
    print("-" * 150)
    for row in ledgers:
        print(
            f"{row['seed']:<5} {row['task']:<20} {row['expected']:<10} {row['chosen']:<10} "
            f"{str(row['correct']):<4} {str(row['safe']):<5} "
            f"{row['new_acc']:<8.3f} {row['old_min_acc']:<8.3f} "
            f"{row['composition_acc']:<8.3f} {row['new_closure_norm']:<9.4f} "
            f"{row['operator_count']:<4} {row['program']}"
        )
    decision_accuracy = float(np.mean([row["correct"] for row in ledgers]))
    destructive_updates = [
        row for row in ledgers
        if row["chosen"] == "update" and not row["safe"]
    ]
    unnecessary_allocations = [
        row for row in ledgers
        if row["chosen"] == "allocate" and row["expected"] != "allocate"
    ]
    print("-" * 150)
    print(f"decision_accuracy={decision_accuracy:.4f}")
    print(f"destructive_update_count={len(destructive_updates)}")
    print(f"unnecessary_allocation_count={len(unnecessary_allocations)}")
    print("=" * 150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-action-selection", action="store_true")
    parser.add_argument("--geometry-signal-diagnostics", action="store_true")
    parser.add_argument("--forced-conflict-update", action="store_true")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--stream", default="ADD,MAX,COPY,MIN,SUB")
    parser.add_argument("--num-values", type=int, default=5)
    parser.add_argument("--code-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--ae-epochs", type=int, default=600)
    parser.add_argument("--operator-epochs", type=int, default=1500)
    parser.add_argument("--update-epochs", type=int, default=500)
    parser.add_argument("--ae-lr", type=float, default=0.01)
    parser.add_argument("--operator-lr", type=float, default=0.01)
    parser.add_argument("--update-lr", type=float, default=0.005)
    parser.add_argument("--lambda-closure", type=float, default=10.0)
    parser.add_argument("--separation-margin", type=float, default=2.0)
    parser.add_argument("--separation-weight", type=float, default=0.5)
    parser.add_argument("--grad-memory-beta", type=float, default=0.98)
    parser.add_argument("--reuse-acc-threshold", type=float, default=0.98)
    parser.add_argument("--reuse-closure-threshold", type=float, default=0.02)
    parser.add_argument("--update-min-acc", type=float, default=0.55)
    parser.add_argument("--update-closure-low", type=float, default=0.02)
    parser.add_argument("--update-closure-high", type=float, default=0.65)
    parser.add_argument("--repair-update-rule", choices=("adam", "neuron_gated", "structural_gated"), default="neuron_gated")
    parser.add_argument("--projection-norm-floor", type=float, default=1e-8)
    parser.add_argument("--min-responsibility", type=float, default=1e-12)
    parser.add_argument("--neuron-gate-power", type=float, default=1.0)
    parser.add_argument(
        "--structural-risk-signal",
        choices=("activation_downstream", "activation_weight_product"),
        default="activation_weight_product",
    )
    parser.add_argument("--structural-need-signal", choices=("gradient", "responsibility"), default="responsibility")
    parser.add_argument("--structural-risk-mode", choices=("repair", "protect"), default="repair")
    parser.add_argument("--structural-risk-power", type=float, default=1.0)
    parser.add_argument("--structural-need-power", type=float, default=1.0)
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--max-programs", type=int, default=10000)
    parser.add_argument("--repair-noise-std", type=float, default=0.10)
    parser.add_argument("--diagnostic-task", default="ADD")
    parser.add_argument("--diagnostic-noise-std", type=float, default=0.0)
    parser.add_argument("--diagnostic-top-k", type=int, default=5)
    parser.add_argument("--conflict-task", default="ADD_PLUS_ONE")
    parser.add_argument(
        "--forced-conflict-rules",
        default="adam,neuron_gated,structural_repair,structural_protect",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-html", type=Path)
    args = parser.parse_args()
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    if args.num_values <= 1:
        raise ValueError("--num-values must be greater than 1.")
    if args.code_dim <= 0 or args.hidden_dim <= 0:
        raise ValueError("--code-dim and --hidden-dim must be positive.")
    if args.repair_noise_std <= 0.0:
        raise ValueError("--repair-noise-std must be positive.")
    if args.projection_norm_floor <= 0.0:
        raise ValueError("--projection-norm-floor must be positive.")
    if args.min_responsibility <= 0.0:
        raise ValueError("--min-responsibility must be positive.")
    if args.neuron_gate_power <= 0.0:
        raise ValueError("--neuron-gate-power must be positive.")
    if args.structural_risk_power <= 0.0:
        raise ValueError("--structural-risk-power must be positive.")
    if args.structural_need_power <= 0.0:
        raise ValueError("--structural-need-power must be positive.")
    if args.diagnostic_noise_std < 0.0:
        raise ValueError("--diagnostic-noise-std must be non-negative.")
    if args.diagnostic_top_k <= 0:
        raise ValueError("--diagnostic-top-k must be positive.")
    if args.diagnostic_top_k > args.hidden_dim:
        raise ValueError("--diagnostic-top-k cannot exceed --hidden-dim.")
    if args.diagnostic_task.upper() not in TASKS:
        raise ValueError(f"Unknown --diagnostic-task {args.diagnostic_task!r}. Available: {sorted(TASKS)}")
    selected_modes = (
        int(args.counterfactual_action_selection)
        + int(args.geometry_signal_diagnostics)
        + int(args.forced_conflict_update)
    )
    if selected_modes > 1:
        raise ValueError(
            "Choose only one mode: benchmark, --counterfactual-action-selection, "
            "--geometry-signal-diagnostics, or --forced-conflict-update."
        )
    if args.geometry_signal_diagnostics and args.output_html is not None:
        raise ValueError("--output-html is not supported for --geometry-signal-diagnostics; use --output-json.")
    if args.forced_conflict_update and args.output_html is not None:
        raise ValueError("--output-html is not supported for --forced-conflict-update; use --output-json.")
    if args.forced_conflict_update:
        forced_rules = parse_csv_items(args.forced_conflict_rules, "--forced-conflict-rules")
        unknown_rules = [rule for rule in forced_rules if rule not in FORCED_CONFLICT_RULES]
        if unknown_rules:
            raise ValueError(
                f"Unknown forced conflict rule(s): {unknown_rules}. "
                f"Available: {sorted(FORCED_CONFLICT_RULES)}"
            )
    return args


def main() -> None:
    args = parse_args()
    if args.forced_conflict_update:
        report = run_forced_conflict_update(args)
    elif args.geometry_signal_diagnostics:
        report = run_geometry_signal_diagnostics(args)
    elif args.counterfactual_action_selection:
        report = run_counterfactual_action_selection(args)
    else:
        report = run_benchmark(args)
    if args.output_json is not None:
        write_json_report(report, args.output_json)
        print(f"wrote_json={args.output_json}")
    if args.output_html is not None:
        write_html_report(report, args.output_html)
        print(f"wrote_html={args.output_html}")


if __name__ == "__main__":
    main()
