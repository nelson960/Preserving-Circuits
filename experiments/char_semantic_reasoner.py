"""Character-level latent-geometry continual learner.

This experiment is deliberately small. It tests whether a learner can use a
stable character code space, closed unary operators, program search, and a
counterfactual optimizer loop to decide:

    reuse / compose / update / allocate

The dataset is intentionally inspectable: a small lowercase/uppercase alphabet
with semantic operators such as SHIFT, CAPS, LOWER, RESET, and REVERSE_SHIFT.
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


Program = tuple[Any, ...]
TokenFn = Callable[[np.ndarray, "CharacterSpace"], np.ndarray]


@dataclass(frozen=True)
class CharacterSpace:
    lower: tuple[str, ...]
    upper: tuple[str, ...]
    include_separator: bool

    @property
    def tokens(self) -> tuple[str, ...]:
        suffix = ("_",) if self.include_separator else ()
        return self.lower + self.upper + suffix

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def alphabet_size(self) -> int:
        return len(self.lower)

    @property
    def separator_index(self) -> int | None:
        if not self.include_separator:
            return None
        return len(self.lower) + len(self.upper)

    def token_index(self, token: str) -> int:
        try:
            return self.tokens.index(token)
        except ValueError as error:
            raise ValueError(f"Unknown character token {token!r}.") from error

    def is_lower(self, values: np.ndarray) -> np.ndarray:
        return (values >= 0) & (values < self.alphabet_size)

    def is_upper(self, values: np.ndarray) -> np.ndarray:
        return (values >= self.alphabet_size) & (values < 2 * self.alphabet_size)

    def is_letter(self, values: np.ndarray) -> np.ndarray:
        return self.is_lower(values) | self.is_upper(values)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fn: TokenFn


@dataclass(frozen=True)
class Config:
    alphabet_size: int
    include_separator: bool
    code_dim: int
    hidden_dim: int
    boundary_epochs: int
    operator_epochs: int
    update_epochs: int
    abstraction_epochs: int
    boundary_lr: float
    operator_lr: float
    update_lr: float
    abstraction_lr: float
    lambda_closure: float
    separation_margin: float
    separation_weight: float
    grad_memory_beta: float
    reuse_acc_threshold: float
    reuse_closure_threshold: float
    update_closure_low: float
    update_closure_high: float
    repair_update_rule: str
    projection_norm_floor: float
    min_responsibility: float
    structural_risk_signal: str
    structural_need_signal: str
    structural_risk_mode: str
    structural_risk_power: float
    structural_need_power: float
    control_dim: int
    identity_weight: float
    composition_target_weight: float
    composition_agreement_weight: float
    search_depth: int
    max_programs: int
    device: torch.device


@dataclass
class OperatorRecord:
    name: str
    origin_task: str
    module: "ClosedUnaryOperator"
    grad_memory: dict[str, torch.Tensor]
    parameter_count: int
    update_count: int = 0


@dataclass(frozen=True)
class StreamEvent:
    label: str
    task_name: str
    repair: bool


class LatentBoundary(nn.Module):
    def __init__(self, num_tokens: int, code_dim: int) -> None:
        super().__init__()
        self.codebook = nn.Parameter(torch.empty(num_tokens, code_dim))
        self.decoder = nn.Linear(code_dim, num_tokens)
        nn.init.normal_(self.codebook, mean=0.0, std=1.0)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=1.0 / np.sqrt(code_dim))
        nn.init.zeros_(self.decoder.bias)

    def encode(self, token_indices: torch.Tensor) -> torch.Tensor:
        return self.codebook[token_indices]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class ClosedUnaryOperator(nn.Module):
    def __init__(self, code_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        first = self.net[0]
        second = self.net[2]
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise TypeError("ClosedUnaryOperator must be Linear -> ReLU -> Linear.")
        nn.init.normal_(first.weight, mean=0.0, std=1.0 / np.sqrt(code_dim))
        nn.init.constant_(first.bias, 0.01)
        nn.init.normal_(second.weight, mean=0.0, std=1.0 / np.sqrt(hidden_dim))
        nn.init.zeros_(second.bias)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        return self.net(code)


class ClosedControlledUnaryOperator(nn.Module):
    def __init__(self, code_dim: int, control_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        first = self.net[0]
        second = self.net[2]
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise TypeError("ClosedControlledUnaryOperator must be Linear -> ReLU -> Linear.")
        nn.init.normal_(first.weight, mean=0.0, std=1.0 / np.sqrt(code_dim + control_dim))
        nn.init.constant_(first.bias, 0.01)
        nn.init.normal_(second.weight, mean=0.0, std=1.0 / np.sqrt(hidden_dim))
        nn.init.zeros_(second.bias)

    def forward(self, code: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if code.shape[0] != control.shape[0]:
            raise ValueError(f"code/control batch mismatch: {code.shape[0]} != {control.shape[0]}.")
        return self.net(torch.cat([code, control], dim=-1))


class LearnedActionPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    raise ValueError(f"Unknown device {name!r}. Expected cpu, mps, or cuda.")


def parse_csv_items(value: str, option_name: str) -> list[str]:
    items = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option_name} must contain at least one item.")
    return items


def parse_csv_rule_items(value: str, option_name: str) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option_name} must contain at least one item.")
    return items


def parse_csv_int_items(value: str, option_name: str) -> list[int]:
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_items:
        raise ValueError(f"{option_name} must contain at least one integer.")
    items: list[int] = []
    for item in raw_items:
        try:
            items.append(int(item))
        except ValueError as error:
            raise ValueError(f"{option_name} contains non-integer item {item!r}.") from error
    return items


def parse_stream_events(value: str, option_name: str) -> list[StreamEvent]:
    items = parse_csv_items(value, option_name)
    events: list[StreamEvent] = []
    for item in items:
        if item.startswith("REPAIR_"):
            task_name = item.removeprefix("REPAIR_")
            if task_name not in TASKS:
                raise ValueError(f"{option_name} contains unknown repair task {task_name!r}. Available: {sorted(TASKS)}")
            events.append(StreamEvent(label=item, task_name=task_name, repair=True))
            continue
        if item not in TASKS:
            raise ValueError(f"{option_name} contains unknown task {item!r}. Available: {sorted(TASKS)}")
        events.append(StreamEvent(label=item, task_name=item, repair=False))
    return events


def parse_stream_specs(value: str, option_name: str) -> list[list[StreamEvent]]:
    raw_specs = [spec.strip() for spec in value.split(";") if spec.strip()]
    if not raw_specs:
        raise ValueError(f"{option_name} must contain at least one comma-separated stream.")
    return [parse_stream_events(spec, option_name) for spec in raw_specs]


def make_character_space(alphabet_size: int, include_separator: bool) -> CharacterSpace:
    if alphabet_size <= 1:
        raise ValueError("--alphabet-size must be greater than 1.")
    if alphabet_size > 26:
        raise ValueError("--alphabet-size cannot exceed 26.")
    lower = tuple(chr(ord("a") + idx) for idx in range(alphabet_size))
    upper = tuple(chr(ord("A") + idx) for idx in range(alphabet_size))
    return CharacterSpace(lower=lower, upper=upper, include_separator=include_separator)


def op_copy(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    del space
    return values.copy()


def shift_values(values: np.ndarray, space: CharacterSpace, amount: int) -> np.ndarray:
    shifted = values.copy()
    lower_mask = space.is_lower(values)
    upper_mask = space.is_upper(values)
    shifted[lower_mask] = (values[lower_mask] + amount) % space.alphabet_size
    shifted[upper_mask] = space.alphabet_size + (
        (values[upper_mask] - space.alphabet_size + amount) % space.alphabet_size
    )
    return shifted


def op_shift(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    return shift_values(values, space, 1)


def op_double_shift(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    return shift_values(values, space, 2)


def op_shift3(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    return shift_values(values, space, 3)


def op_reverse_shift(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    return shift_values(values, space, -1)


def op_caps(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    result = values.copy()
    lower_mask = space.is_lower(values)
    result[lower_mask] = values[lower_mask] + space.alphabet_size
    return result


def op_lower(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    result = values.copy()
    upper_mask = space.is_upper(values)
    result[upper_mask] = values[upper_mask] - space.alphabet_size
    return result


def op_case_flip(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    result = values.copy()
    lower_mask = space.is_lower(values)
    upper_mask = space.is_upper(values)
    result[lower_mask] = values[lower_mask] + space.alphabet_size
    result[upper_mask] = values[upper_mask] - space.alphabet_size
    return result


def op_reset(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
    result = values.copy()
    lower_mask = space.is_lower(values)
    upper_mask = space.is_upper(values)
    result[lower_mask] = 0
    result[upper_mask] = space.alphabet_size
    return result


def compose_unary(first: TokenFn, second: TokenFn) -> TokenFn:
    def fn(values: np.ndarray, space: CharacterSpace) -> np.ndarray:
        return second(first(values, space), space)

    return fn


TASKS: dict[str, TaskSpec] = {
    "COPY": TaskSpec("COPY", op_copy),
    "SHIFT": TaskSpec("SHIFT", op_shift),
    "DOUBLE_SHIFT": TaskSpec("DOUBLE_SHIFT", op_double_shift),
    "SHIFT3": TaskSpec("SHIFT3", op_shift3),
    "REVERSE_SHIFT": TaskSpec("REVERSE_SHIFT", op_reverse_shift),
    "CAPS": TaskSpec("CAPS", op_caps),
    "LOWER": TaskSpec("LOWER", op_lower),
    "CASE_FLIP": TaskSpec("CASE_FLIP", op_case_flip),
    "RESET": TaskSpec("RESET", op_reset),
    "SHIFT_THEN_CAPS": TaskSpec("SHIFT_THEN_CAPS", compose_unary(op_shift, op_caps)),
    "CAPS_THEN_SHIFT": TaskSpec("CAPS_THEN_SHIFT", compose_unary(op_caps, op_shift)),
    "LOWER_THEN_SHIFT": TaskSpec("LOWER_THEN_SHIFT", compose_unary(op_lower, op_shift)),
    "CAPS_THEN_LOWER": TaskSpec("CAPS_THEN_LOWER", compose_unary(op_caps, op_lower)),
}


DEFAULT_STREAM = "COPY,SHIFT,DOUBLE_SHIFT,CAPS,SHIFT_THEN_CAPS,LOWER,CAPS_THEN_LOWER,REVERSE_SHIFT,RESET,SHIFT3"
DEFAULT_POLICY_TRAIN_STREAMS = (
    "COPY,SHIFT,REPAIR_SHIFT,DOUBLE_SHIFT,CAPS,SHIFT_THEN_CAPS;"
    "COPY,CAPS,LOWER,CAPS_THEN_LOWER,SHIFT,SHIFT3,REVERSE_SHIFT;"
    "SHIFT,REPAIR_SHIFT,COPY,DOUBLE_SHIFT,REVERSE_SHIFT,RESET,CAPS_THEN_SHIFT"
)
DEFAULT_POLICY_TEST_STREAMS = (
    "COPY,SHIFT,REPAIR_SHIFT,DOUBLE_SHIFT,CAPS,SHIFT_THEN_CAPS,LOWER,CAPS_THEN_LOWER,REVERSE_SHIFT,RESET,SHIFT3;"
    "COPY,CAPS,SHIFT_THEN_CAPS,REVERSE_SHIFT,RESET,DOUBLE_SHIFT"
)
ACTION_NAMES = ("reuse", "compose", "update", "allocate")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}


def make_task_data(
    task_name: str,
    space: CharacterSpace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name!r}. Available: {sorted(TASKS)}")
    inputs = np.arange(space.num_tokens, dtype=np.int64)
    targets = TASKS[task_name].fn(inputs, space).astype(np.int64)
    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )


def closure_loss(out_code: torch.Tensor, target_code: torch.Tensor) -> torch.Tensor:
    return (out_code - target_code).pow(2).sum(dim=-1).mean()


def manifold_error(out_code: torch.Tensor, boundary: LatentBoundary) -> torch.Tensor:
    distances = torch.cdist(out_code, boundary.codebook.detach(), p=2.0).pow(2)
    return distances.min(dim=1).values.mean()


def code_distance_scale(boundary: LatentBoundary) -> float:
    with torch.no_grad():
        codes = boundary.codebook.detach()
        distances = torch.cdist(codes, codes, p=2.0).pow(2)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
        nearest = distances.masked_select(mask).view(distances.shape[0], distances.shape[0] - 1).min(dim=1).values
        scale = nearest.mean().item()
    if scale <= 0.0:
        raise RuntimeError("Code distance scale is non-positive; latent codebook collapsed.")
    return scale


def train_boundary(boundary: LatentBoundary, cfg: Config) -> None:
    values = torch.arange(boundary.codebook.shape[0], dtype=torch.long, device=cfg.device)
    optimizer = torch.optim.Adam(boundary.parameters(), lr=cfg.boundary_lr)
    for _ in range(cfg.boundary_epochs):
        optimizer.zero_grad()
        codes = boundary.encode(values)
        logits = boundary.decode(codes)
        ce = F.cross_entropy(logits, values)
        distances = torch.cdist(codes, codes, p=2.0).pow(2)
        mask = ~torch.eye(values.shape[0], dtype=torch.bool, device=cfg.device)
        separation = F.relu(cfg.separation_margin - distances.masked_select(mask)).mean()
        loss = ce + cfg.separation_weight * separation
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = boundary.decode(boundary.encode(values)).argmax(dim=-1)
        accuracy = (preds == values).float().mean().item()
    if accuracy < 1.0:
        raise RuntimeError(f"Boundary failed to reconstruct the character codebook: accuracy={accuracy:.4f}")
    boundary.freeze()


def operator_parameter_count(cfg: Config) -> int:
    return cfg.code_dim * cfg.hidden_dim + cfg.hidden_dim + cfg.hidden_dim * cfg.code_dim + cfg.code_dim


def controlled_operator_parameter_count(cfg: Config) -> int:
    input_dim = cfg.code_dim + cfg.control_dim
    return input_dim * cfg.hidden_dim + cfg.hidden_dim + cfg.hidden_dim * cfg.code_dim + cfg.code_dim


def new_operator(origin_task: str, cfg: Config, name: str) -> OperatorRecord:
    module = ClosedUnaryOperator(cfg.code_dim, cfg.hidden_dim).to(cfg.device)
    return OperatorRecord(
        name=name,
        origin_task=origin_task,
        module=module,
        grad_memory={},
        parameter_count=operator_parameter_count(cfg),
    )


def operator_layers(record: OperatorRecord) -> tuple[nn.Linear, nn.Linear]:
    first = record.module.net[0]
    second = record.module.net[2]
    if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
        raise TypeError("Operator must be Linear -> ReLU -> Linear.")
    return first, second


def forward_with_hidden(record: OperatorRecord, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = operator_layers(record)
    hidden = F.relu(first(code))
    out_code = second(hidden)
    return out_code, hidden


def tensor_is_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains non-finite values.")


def update_grad_memory(record: OperatorRecord, beta: float) -> None:
    for name, parameter in record.module.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for {record.name} parameter {name}.")
        grad = parameter.grad.detach().clone()
        if name not in record.grad_memory:
            record.grad_memory[name] = grad
        else:
            record.grad_memory[name].mul_(beta).add_(grad, alpha=1.0 - beta)


def train_new_operator(
    boundary: LatentBoundary,
    record: OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    require_perfect: bool,
) -> None:
    optimizer = torch.optim.Adam(record.module.parameters(), lr=cfg.operator_lr)
    input_code = boundary.encode(inputs).detach()
    target_code = boundary.encode(targets).detach()
    for _ in range(cfg.operator_epochs):
        optimizer.zero_grad()
        out_code = record.module(input_code)
        logits = boundary.decode(out_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * closure_loss(out_code, target_code)
        tensor_is_finite("new-operator loss", loss)
        loss.backward()
        update_grad_memory(record, cfg.grad_memory_beta)
        optimizer.step()
    metrics = evaluate_direct_operator(boundary, record, inputs, targets, cfg)
    if require_perfect and metrics["accuracy"] < 1.0:
        raise RuntimeError(
            f"New operator {record.name} failed to fit {record.origin_task}: "
            f"accuracy={metrics['accuracy']:.4f}"
        )


def control_vectors_for_shifts(
    shifts: torch.Tensor,
    alphabet_size: int,
    control_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if control_dim < 2:
        raise ValueError(f"control_dim must be at least 2 for circular shift controls, got {control_dim}.")
    shifts_float = shifts.to(device=device, dtype=torch.float32)
    angle = 2.0 * np.pi * shifts_float / float(alphabet_size)
    pieces = [torch.cos(angle).unsqueeze(-1), torch.sin(angle).unsqueeze(-1)]
    harmonic = 2
    while len(pieces) < control_dim:
        pieces.append(torch.cos(float(harmonic) * angle).unsqueeze(-1))
        if len(pieces) < control_dim:
            pieces.append(torch.sin(float(harmonic) * angle).unsqueeze(-1))
        harmonic += 1
    return torch.cat(pieces[:control_dim], dim=-1)


def make_shift_k_dataset(
    shifts: list[int],
    space: CharacterSpace,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not shifts:
        raise ValueError("make_shift_k_dataset requires at least one shift.")
    input_blocks: list[np.ndarray] = []
    shift_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    base_inputs = np.arange(space.num_tokens, dtype=np.int64)
    for shift in shifts:
        input_blocks.append(base_inputs)
        shift_blocks.append(np.full(space.num_tokens, shift, dtype=np.int64))
        target_blocks.append(shift_values(base_inputs, space, shift).astype(np.int64))
    return (
        torch.as_tensor(np.concatenate(input_blocks), dtype=torch.long, device=cfg.device),
        torch.as_tensor(np.concatenate(shift_blocks), dtype=torch.long, device=cfg.device),
        torch.as_tensor(np.concatenate(target_blocks), dtype=torch.long, device=cfg.device),
    )


def canonical_shift(shift: int, alphabet_size: int) -> int:
    if alphabet_size <= 0:
        raise ValueError(f"alphabet_size must be positive, got {alphabet_size}.")
    value = shift % alphabet_size
    half = alphabet_size // 2
    if value > half:
        value -= alphabet_size
    return value


def pairwise_shift_sums(shifts: list[int], alphabet_size: int) -> list[int]:
    if not shifts:
        raise ValueError("pairwise_shift_sums requires at least one shift.")
    sums = {
        canonical_shift(left + right, alphabet_size)
        for left in shifts
        for right in shifts
    }
    return sorted(sums)


def train_shift_k_parent(
    boundary: LatentBoundary,
    model: ClosedControlledUnaryOperator,
    shifts: list[int],
    space: CharacterSpace,
    cfg: Config,
) -> None:
    inputs, shift_values_tensor, targets = make_shift_k_dataset(shifts, space, cfg)
    input_code = boundary.encode(inputs).detach()
    target_code = boundary.encode(targets).detach()
    controls = control_vectors_for_shifts(shift_values_tensor, space.alphabet_size, cfg.control_dim, cfg.device)
    identity_inputs, identity_shift_values, identity_targets = make_shift_k_dataset([0], space, cfg)
    identity_input_code = boundary.encode(identity_inputs).detach()
    identity_target_code = boundary.encode(identity_targets).detach()
    identity_controls = control_vectors_for_shifts(
        identity_shift_values,
        space.alphabet_size,
        cfg.control_dim,
        cfg.device,
    )
    composition_shift_values = pairwise_shift_sums(shifts, space.alphabet_size)
    composition_inputs, composition_shifts, composition_targets = make_shift_k_dataset(composition_shift_values, space, cfg)
    composition_input_code = boundary.encode(composition_inputs).detach()
    composition_target_code = boundary.encode(composition_targets).detach()
    composition_controls = control_vectors_for_shifts(
        composition_shifts,
        space.alphabet_size,
        cfg.control_dim,
        cfg.device,
    )
    pair_left: list[int] = []
    pair_right: list[int] = []
    pair_sum: list[int] = []
    for left in shifts:
        for right in shifts:
            pair_left.extend([left] * space.num_tokens)
            pair_right.extend([right] * space.num_tokens)
            pair_sum.extend([canonical_shift(left + right, space.alphabet_size)] * space.num_tokens)
    base_inputs = np.tile(np.arange(space.num_tokens, dtype=np.int64), len(shifts) * len(shifts))
    pair_inputs = torch.as_tensor(base_inputs, dtype=torch.long, device=cfg.device)
    pair_input_code = boundary.encode(pair_inputs).detach()
    pair_left_tensor = torch.as_tensor(pair_left, dtype=torch.long, device=cfg.device)
    pair_right_tensor = torch.as_tensor(pair_right, dtype=torch.long, device=cfg.device)
    pair_sum_tensor = torch.as_tensor(pair_sum, dtype=torch.long, device=cfg.device)
    pair_left_controls = control_vectors_for_shifts(pair_left_tensor, space.alphabet_size, cfg.control_dim, cfg.device)
    pair_right_controls = control_vectors_for_shifts(pair_right_tensor, space.alphabet_size, cfg.control_dim, cfg.device)
    pair_sum_controls = control_vectors_for_shifts(pair_sum_tensor, space.alphabet_size, cfg.control_dim, cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.abstraction_lr)
    for _ in range(cfg.abstraction_epochs):
        optimizer.zero_grad()
        out_code = model(input_code, controls)
        logits = boundary.decode(out_code)
        supervised_loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * closure_loss(out_code, target_code)
        identity_out = model(identity_input_code, identity_controls)
        identity_logits = boundary.decode(identity_out)
        identity_loss = F.cross_entropy(identity_logits, identity_targets) + cfg.lambda_closure * closure_loss(
            identity_out,
            identity_target_code,
        )
        composition_out = model(composition_input_code, composition_controls)
        composition_logits = boundary.decode(composition_out)
        composition_target_loss = F.cross_entropy(composition_logits, composition_targets) + cfg.lambda_closure * closure_loss(
            composition_out,
            composition_target_code,
        )
        first_step = model(pair_input_code, pair_left_controls)
        second_step = model(first_step, pair_right_controls)
        direct_sum = model(pair_input_code, pair_sum_controls)
        composition_agreement = closure_loss(second_step, direct_sum.detach())
        loss = (
            supervised_loss
            + cfg.identity_weight * identity_loss
            + cfg.composition_target_weight * composition_target_loss
            + cfg.composition_agreement_weight * composition_agreement
        )
        tensor_is_finite("shift-k parent loss", loss)
        loss.backward()
        optimizer.step()


def evaluate_shift_k_parent(
    boundary: LatentBoundary,
    model: ClosedControlledUnaryOperator,
    shift: int,
    space: CharacterSpace,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    inputs, shifts, targets = make_shift_k_dataset([shift], space, cfg)
    with torch.no_grad():
        input_code = boundary.encode(inputs)
        controls = control_vectors_for_shifts(shifts, space.alphabet_size, cfg.control_dim, cfg.device)
        out_code = model(input_code, controls)
        logits = boundary.decode(out_code)
        preds = logits.argmax(dim=-1)
        close = closure_loss(out_code, boundary.encode(targets)).item()
        manifold = manifold_error(out_code, boundary).item()
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "loss": float(F.cross_entropy(logits, targets).item()),
            "closure_norm": float(close / distance_scale),
            "manifold_norm": float(manifold / distance_scale),
        }


def evaluate_direct_operator(
    boundary: LatentBoundary,
    record: OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> dict[str, float]:
    del cfg
    with torch.no_grad():
        out_code = record.module(boundary.encode(inputs))
        logits = boundary.decode(out_code)
        preds = logits.argmax(dim=-1)
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "loss": float(F.cross_entropy(logits, targets).item()),
            "closure": float(closure_loss(out_code, boundary.encode(targets)).item()),
            "manifold": float(manifold_error(out_code, boundary).item()),
        }


def program_depth(program: Program) -> int:
    if program[0] == "var":
        return 0
    if program[0] == "op":
        return 1 + program_depth(program[2][0])
    raise ValueError(f"Unknown program node type {program[0]!r}.")


def program_to_str(program: Program) -> str:
    if program[0] == "var":
        return "x"
    if program[0] == "op":
        return f"{program[1]}({program_to_str(program[2][0])})"
    raise ValueError(f"Unknown program node type {program[0]!r}.")


def eval_program(
    program: Program,
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    inputs: torch.Tensor,
) -> torch.Tensor:
    if program[0] == "var":
        return boundary.encode(inputs)
    if program[0] == "op":
        op_name = str(program[1])
        if op_name not in library:
            raise KeyError(f"Program references missing operator {op_name!r}.")
        child = eval_program(program[2][0], boundary, library, inputs)
        return library[op_name].module(child)
    raise ValueError(f"Unknown program node type {program[0]!r}.")


def generate_programs(
    library: dict[str, OperatorRecord],
    max_depth: int,
    max_programs: int,
) -> list[Program]:
    programs: list[Program] = [("var", 0)]
    by_depth: list[list[Program]] = [[("var", 0)]]
    seen = {program_to_str(("var", 0))}
    for depth in range(1, max_depth + 1):
        depth_programs: list[Program] = []
        child_pool = [program for depth_list in by_depth for program in depth_list]
        for op_name in library:
            for child in child_pool:
                if program_depth(child) != depth - 1:
                    continue
                program: Program = ("op", op_name, (child,))
                key = program_to_str(program)
                if key in seen:
                    continue
                seen.add(key)
                depth_programs.append(program)
                programs.append(program)
                if len(programs) > max_programs:
                    raise RuntimeError(
                        f"Program search exceeded --max-programs={max_programs}; "
                        "increase the limit or reduce --search-depth."
                    )
        by_depth.append(depth_programs)
    return programs


def evaluate_program(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    program: Program,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    distance_scale: float,
) -> dict[str, float]:
    with torch.no_grad():
        out_code = eval_program(program, boundary, library, inputs)
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


def search_best_program(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> tuple[Program, dict[str, float]]:
    programs = generate_programs(library, cfg.search_depth, cfg.max_programs)
    best_program: Program | None = None
    best_metrics: dict[str, float] | None = None
    best_depth: int | None = None
    for program in programs:
        metrics = evaluate_program(boundary, library, program, inputs, targets, distance_scale)
        current_depth = program_depth(program)
        if best_metrics is None:
            best_program = program
            best_metrics = metrics
            best_depth = current_depth
        elif metrics["accuracy"] > best_metrics["accuracy"]:
            best_program = program
            best_metrics = metrics
            best_depth = current_depth
        elif metrics["accuracy"] == best_metrics["accuracy"]:
            if best_depth is None or current_depth < best_depth:
                best_program = program
                best_metrics = metrics
                best_depth = current_depth
            elif current_depth == best_depth and metrics["closure_norm"] < best_metrics["closure_norm"]:
                best_program = program
                best_metrics = metrics
                best_depth = current_depth
    if best_program is None or best_metrics is None:
        raise RuntimeError("Program search failed to select a candidate.")
    return best_program, best_metrics


def direct_operator_program(record: OperatorRecord) -> Program:
    return ("op", record.name, (("var", 0),))


def direct_program_operator_name(program: Program) -> str:
    if program[0] != "op":
        raise ValueError(f"Expected a direct operator program, got {program_to_str(program)}.")
    child = program[2][0]
    if child[0] != "var":
        raise ValueError(f"Expected a direct operator program, got {program_to_str(program)}.")
    return str(program[1])


def perturb_operator(record: OperatorRecord, noise_std: float) -> None:
    if noise_std <= 0.0:
        raise ValueError(f"repair noise standard deviation must be positive, got {noise_std}.")
    with torch.no_grad():
        for name, parameter in record.module.named_parameters():
            noise = torch.randn_like(parameter) * noise_std
            tensor_is_finite(f"repair noise for {record.name}.{name}", noise)
            parameter.add_(noise)


def inject_repair_noise(
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    task_name: str,
    noise_std: float,
) -> str:
    if task_name not in task_to_program:
        raise KeyError(f"Repair event requested for {task_name}, but the task has no learned program.")
    op_name = direct_program_operator_name(task_to_program[task_name])
    if op_name not in library:
        raise KeyError(f"Repair event references missing operator {op_name!r}.")
    perturb_operator(library[op_name], noise_std)
    return op_name


def clone_library(library: dict[str, OperatorRecord]) -> dict[str, OperatorRecord]:
    cloned: dict[str, OperatorRecord] = {}
    for name, record in library.items():
        module = copy.deepcopy(record.module)
        grad_memory = {key: value.detach().clone() for key, value in record.grad_memory.items()}
        cloned[name] = OperatorRecord(
            name=record.name,
            origin_task=record.origin_task,
            module=module,
            grad_memory=grad_memory,
            parameter_count=record.parameter_count,
            update_count=record.update_count,
        )
    return cloned


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
    if value <= low or value >= high:
        return 0.0
    midpoint = 0.5 * (low + high)
    if value <= midpoint:
        return (value - low) / (midpoint - low)
    return (high - value) / (high - midpoint)


def slow_slice(record: OperatorRecord, name: str, index: int | None = None, column: int | None = None) -> torch.Tensor | None:
    value = record.grad_memory.get(name)
    if value is None:
        return None
    if index is not None:
        return value[index]
    if column is not None:
        return value[:, column]
    return value


def update_slice(
    parameter_slice: torch.Tensor,
    grad_slice: torch.Tensor,
    slow_value: torch.Tensor | None,
    neuron_gate: float,
    closure_gate: float,
    cfg: Config,
) -> tuple[float, float, float]:
    reinforce, novel = decompose_gradient(grad_slice, slow_value, cfg.projection_norm_floor)
    update = closure_gate * reinforce + neuron_gate * novel
    tensor_is_finite("custom update", update)
    parameter_slice.add_(update, alpha=-cfg.update_lr)
    return (
        float(reinforce.pow(2).sum().item()),
        float(novel.pow(2).sum().item()),
        float(update.pow(2).sum().item()),
    )


def normalize_vector(name: str, values: torch.Tensor, floor: float) -> torch.Tensor:
    tensor_is_finite(name, values)
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
    risk = normalize_vector("structural risk", risk_raw, cfg.min_responsibility)
    need = normalize_vector("structural need", need_raw, cfg.min_responsibility)
    if cfg.structural_risk_mode == "repair":
        risk_factor = risk.pow(cfg.structural_risk_power)
    elif cfg.structural_risk_mode == "protect":
        risk_factor = (1.0 - risk).clamp(min=0.0, max=1.0).pow(cfg.structural_risk_power)
    else:
        raise ValueError(f"Unknown structural_risk_mode={cfg.structural_risk_mode!r}.")
    gates = closure_gate * need.pow(cfg.structural_need_power) * risk_factor
    tensor_is_finite("structural gates", gates)
    return gates, {
        "risk_mean": float(risk.mean().item()),
        "need_mean": float(need.mean().item()),
        "gate_mean": float(gates.mean().item()),
        "gate_max": float(gates.max().item()),
    }


def adam_update(
    boundary: LatentBoundary,
    record: OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> dict[str, float | None]:
    optimizer = torch.optim.Adam(record.module.parameters(), lr=cfg.update_lr)
    input_code = boundary.encode(inputs).detach()
    target_code = boundary.encode(targets).detach()
    for _ in range(cfg.update_epochs):
        optimizer.zero_grad()
        out_code = record.module(input_code)
        logits = boundary.decode(out_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * closure_loss(out_code, target_code)
        tensor_is_finite("adam update loss", loss)
        loss.backward()
        update_grad_memory(record, cfg.grad_memory_beta)
        optimizer.step()
    record.update_count += 1
    return {"gate_mean": None}


def structural_update(
    boundary: LatentBoundary,
    record: OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    first, second = operator_layers(record)
    input_code = boundary.encode(inputs).detach()
    target_code = boundary.encode(targets).detach()
    gate_means: list[float] = []
    gate_maxes: list[float] = []
    risk_means: list[float] = []
    need_means: list[float] = []
    active_counts: list[float] = []
    update_norms: list[float] = []
    for _ in range(cfg.update_epochs):
        record.module.zero_grad(set_to_none=True)
        out_code, hidden = forward_with_hidden(record, input_code)
        logits = boundary.decode(out_code)
        close = closure_loss(out_code, target_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * close
        tensor_is_finite("structural update loss", loss)
        loss.backward()
        closure_norm = close.detach().item() / distance_scale
        closure_gate = medium_band_gate(closure_norm, cfg.update_closure_low, cfg.update_closure_high)
        gates, gate_stats = structural_gate_vector(record, hidden, closure_gate, cfg)
        with torch.no_grad():
            step_update_norm = 0.0
            for neuron_index, gate_tensor in enumerate(gates):
                gate = float(gate_tensor.item())
                _, _, update_norm = update_slice(
                    first.weight[neuron_index],
                    first.weight.grad[neuron_index],
                    slow_slice(record, "net.0.weight", index=neuron_index),
                    gate,
                    closure_gate,
                    cfg,
                )
                step_update_norm += update_norm
                _, _, update_norm = update_slice(
                    first.bias[neuron_index],
                    first.bias.grad[neuron_index],
                    slow_slice(record, "net.0.bias", index=neuron_index),
                    gate,
                    closure_gate,
                    cfg,
                )
                step_update_norm += update_norm
                _, _, update_norm = update_slice(
                    second.weight[:, neuron_index],
                    second.weight.grad[:, neuron_index],
                    slow_slice(record, "net.2.weight", column=neuron_index),
                    gate,
                    closure_gate,
                    cfg,
                )
                step_update_norm += update_norm
            if second.bias.grad is None:
                raise RuntimeError(f"Missing output bias gradient for {record.name}.")
            mean_gate = float(gates.mean().item())
            _, _, bias_update_norm = update_slice(
                second.bias,
                second.bias.grad,
                slow_slice(record, "net.2.bias"),
                mean_gate,
                closure_gate,
                cfg,
            )
            step_update_norm += bias_update_norm
        update_grad_memory(record, cfg.grad_memory_beta)
        gate_means.append(gate_stats["gate_mean"])
        gate_maxes.append(gate_stats["gate_max"])
        risk_means.append(gate_stats["risk_mean"])
        need_means.append(gate_stats["need_mean"])
        active_counts.append(float((gates > 0.0).sum().item()))
        update_norms.append(float(np.sqrt(step_update_norm)))
    record.update_count += 1
    return {
        "gate_mean": float(np.mean(gate_means)),
        "gate_max": float(np.mean(gate_maxes)),
        "risk_mean": float(np.mean(risk_means)),
        "need_mean": float(np.mean(need_means)),
        "active_neurons": float(np.mean(active_counts)),
        "update_norm_mean": float(np.mean(update_norms)),
    }


def apply_update_rule(
    boundary: LatentBoundary,
    record: OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float | None]:
    if cfg.repair_update_rule == "adam":
        return adam_update(boundary, record, inputs, targets, cfg)
    if cfg.repair_update_rule == "structural_gated":
        return structural_update(boundary, record, inputs, targets, cfg, distance_scale)
    raise ValueError(f"Unknown repair_update_rule={cfg.repair_update_rule!r}.")


def action_name_for_program(program: Program) -> str:
    if program[0] == "var":
        return "reuse"
    if program_depth(program) > 1:
        return "compose"
    return "reuse"


def safe_mean(values: list[float], default: float) -> float:
    return float(np.mean(values)) if values else default


def safe_min(values: list[float], default: float) -> float:
    return float(np.min(values)) if values else default


def evaluate_state(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    learned_tasks: list[str],
    current_task: str,
    current_program: Program,
    space: CharacterSpace,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    current_inputs, current_targets = make_task_data(current_task, space, cfg.device)
    current_metrics = evaluate_program(boundary, library, current_program, current_inputs, current_targets, distance_scale)
    old_accs: list[float] = []
    old_closures: list[float] = []
    for task_name in learned_tasks:
        if task_name not in task_to_program:
            raise KeyError(f"Missing learned program for {task_name}.")
        inputs, targets = make_task_data(task_name, space, cfg.device)
        metrics = evaluate_program(boundary, library, task_to_program[task_name], inputs, targets, distance_scale)
        old_accs.append(metrics["accuracy"])
        old_closures.append(metrics["closure_norm"])
    return {
        "new_acc": current_metrics["accuracy"],
        "new_loss": current_metrics["loss"],
        "new_closure_norm": current_metrics["closure_norm"],
        "new_manifold_norm": current_metrics["manifold_norm"],
        "old_min_acc": safe_min(old_accs, 1.0),
        "old_mean_acc": safe_mean(old_accs, 1.0),
        "old_mean_closure_norm": safe_mean(old_closures, 0.0),
    }


def counterfactual_score(metrics: dict[str, float], action: str, new_parameters: int, cfg: Config) -> tuple[float, bool]:
    closure_limit = cfg.reuse_closure_threshold if action in {"reuse", "compose"} else cfg.update_closure_high
    safe = (
        metrics["new_acc"] >= cfg.reuse_acc_threshold
        and metrics["old_min_acc"] >= cfg.reuse_acc_threshold
        and metrics["old_mean_closure_norm"] <= cfg.reuse_closure_threshold
        and metrics["new_closure_norm"] <= closure_limit
    )
    action_penalty = {
        "reuse": 0.00,
        "compose": 0.00,
        "update": 0.01,
        "allocate": 0.03,
    }
    if action not in action_penalty:
        raise ValueError(f"Unknown action {action!r}.")
    score = (
        10.0 * metrics["new_acc"]
        + 4.0 * metrics["old_min_acc"]
        - 1.0 * metrics["new_closure_norm"]
        - 1.0 * metrics["old_mean_closure_norm"]
        - action_penalty[action]
        - 1e-5 * new_parameters
    )
    if not safe:
        score -= 100.0
    return float(score), safe


def candidate_row(
    action: str,
    program: Program,
    metrics: dict[str, float],
    score: float,
    safe: bool,
    new_parameters: int,
    update_stats: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "program": program_to_str(program),
        "metrics": metrics,
        "score": score,
        "safe": safe,
        "new_parameters": new_parameters,
        "update_stats": update_stats or {},
    }


def build_candidates(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    learned_tasks: list[str],
    task_name: str,
    space: CharacterSpace,
    cfg: Config,
    distance_scale: float,
) -> list[dict[str, Any]]:
    inputs, targets = make_task_data(task_name, space, cfg.device)
    candidates: list[dict[str, Any]] = []

    best_program, _ = search_best_program(boundary, library, inputs, targets, cfg, distance_scale)
    action = action_name_for_program(best_program)
    reuse_task_to_program = dict(task_to_program)
    reuse_task_to_program[task_name] = best_program
    metrics = evaluate_state(
        boundary,
        library,
        reuse_task_to_program,
        learned_tasks,
        task_name,
        best_program,
        space,
        cfg,
        distance_scale,
    )
    score, safe = counterfactual_score(metrics, action, 0, cfg)
    candidates.append(candidate_row(action, best_program, metrics, score, safe, 0))

    for record in library.values():
        shadow_library = clone_library(library)
        shadow_record = shadow_library[record.name]
        update_cfg = replace(cfg, structural_risk_mode="repair")
        update_stats = apply_update_rule(boundary, shadow_record, inputs, targets, update_cfg, distance_scale)
        program = direct_operator_program(shadow_record)
        update_task_to_program = dict(task_to_program)
        update_task_to_program[task_name] = program
        metrics = evaluate_state(
            boundary,
            shadow_library,
            update_task_to_program,
            learned_tasks,
            task_name,
            program,
            space,
            cfg,
            distance_scale,
        )
        score, safe = counterfactual_score(metrics, "update", 0, cfg)
        candidates.append(candidate_row("update", program, metrics, score, safe, 0, update_stats))

    shadow_library = clone_library(library)
    op_name = f"OP_{task_name}_{len(library)}"
    record = new_operator(task_name, cfg, op_name)
    train_new_operator(boundary, record, inputs, targets, cfg, require_perfect=False)
    shadow_library[op_name] = record
    program = direct_operator_program(record)
    allocate_task_to_program = dict(task_to_program)
    allocate_task_to_program[task_name] = program
    metrics = evaluate_state(
        boundary,
        shadow_library,
        allocate_task_to_program,
        learned_tasks,
        task_name,
        program,
        space,
        cfg,
        distance_scale,
    )
    score, safe = counterfactual_score(metrics, "allocate", record.parameter_count, cfg)
    candidates.append(candidate_row("allocate", program, metrics, score, safe, record.parameter_count))
    return candidates


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No candidates were generated.")
    return max(candidates, key=lambda row: row["score"])


def best_candidates_by_action(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not candidates:
        raise RuntimeError("No candidates were generated.")
    grouped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        action = candidate["action"]
        if action not in ACTION_TO_INDEX:
            raise ValueError(f"Unknown candidate action {action!r}.")
        if action not in grouped or candidate["score"] > grouped[action]["score"]:
            grouped[action] = candidate
    return grouped


def metric_value(candidate: dict[str, Any], name: str) -> float:
    metrics = candidate["metrics"]
    if name not in metrics:
        raise KeyError(f"Candidate metrics missing required key {name!r}.")
    return float(metrics[name])


def policy_feature_vector(candidates: list[dict[str, Any]], parameter_scale: float) -> list[float]:
    if parameter_scale <= 0.0:
        raise ValueError(f"parameter_scale must be positive, got {parameter_scale}.")
    grouped = best_candidates_by_action(candidates)
    features: list[float] = []
    for action in ACTION_NAMES:
        candidate = grouped.get(action)
        if candidate is None:
            features.extend([0.0] * 9)
            continue
        features.extend(
            [
                1.0,
                metric_value(candidate, "new_acc"),
                metric_value(candidate, "old_min_acc"),
                metric_value(candidate, "old_mean_acc"),
                float(np.log1p(metric_value(candidate, "new_loss"))),
                float(np.log1p(metric_value(candidate, "new_closure_norm"))),
                float(np.log1p(metric_value(candidate, "old_mean_closure_norm"))),
                float(np.log1p(metric_value(candidate, "new_manifold_norm"))),
                float(candidate["new_parameters"]) / parameter_scale,
            ]
        )
    features.extend(
        [
            float(len(candidates)),
            float(len(grouped)),
        ]
    )
    return features


def available_action_mask(candidates: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    grouped = best_candidates_by_action(candidates)
    return torch.tensor([action in grouped for action in ACTION_NAMES], dtype=torch.bool, device=device)


def train_action_policy(
    features: list[list[float]],
    labels: list[int],
    args: argparse.Namespace,
    cfg: Config,
) -> LearnedActionPolicy:
    if not features:
        raise ValueError("No policy training examples were produced.")
    if len(features) != len(labels):
        raise ValueError(f"Policy feature/label length mismatch: {len(features)} != {len(labels)}.")
    feature_dim = len(features[0])
    if feature_dim <= 0:
        raise ValueError("Policy feature vectors must be non-empty.")
    for row_index, row in enumerate(features):
        if len(row) != feature_dim:
            raise ValueError(f"Policy feature vector {row_index} has length {len(row)}, expected {feature_dim}.")
    set_seed(args.policy_seed)
    model = LearnedActionPolicy(feature_dim, args.policy_hidden_dim, len(ACTION_NAMES)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.policy_lr)
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=cfg.device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=cfg.device)
    for _ in range(args.policy_epochs):
        optimizer.zero_grad()
        logits = model(feature_tensor)
        loss = F.cross_entropy(logits, label_tensor)
        tensor_is_finite("policy loss", loss)
        loss.backward()
        optimizer.step()
    return model


def select_policy_candidate(
    policy: LearnedActionPolicy,
    candidates: list[dict[str, Any]],
    parameter_scale: float,
    device: torch.device,
) -> tuple[dict[str, Any], str, str, bool]:
    grouped = best_candidates_by_action(candidates)
    features = torch.tensor([policy_feature_vector(candidates, parameter_scale)], dtype=torch.float32, device=device)
    mask = available_action_mask(candidates, device)
    with torch.no_grad():
        logits = policy(features).squeeze(0)
    raw_index = int(torch.argmax(logits).item())
    raw_action = ACTION_NAMES[raw_index]
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    if not torch.isfinite(masked_logits).any().item():
        raise RuntimeError("Policy action mask removed every candidate action.")
    chosen_index = int(torch.argmax(masked_logits).item())
    chosen_action = ACTION_NAMES[chosen_index]
    if chosen_action not in grouped:
        raise RuntimeError(f"Masked policy selected unavailable action {chosen_action!r}.")
    return grouped[chosen_action], raw_action, chosen_action, raw_action != chosen_action


def materialize_choice(
    boundary: LatentBoundary,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    task_name: str,
    chosen: dict[str, Any],
    space: CharacterSpace,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float | None]:
    inputs, targets = make_task_data(task_name, space, cfg.device)
    action = chosen["action"]
    if action in {"reuse", "compose"}:
        task_to_program[task_name] = parse_materialized_program(chosen["program"], library)
        return {}
    if action == "allocate":
        op_name = f"OP_{task_name}_{len(library)}"
        record = new_operator(task_name, cfg, op_name)
        train_new_operator(boundary, record, inputs, targets, cfg, require_perfect=True)
        library[op_name] = record
        task_to_program[task_name] = direct_operator_program(record)
        return {}
    if action == "update":
        program = parse_materialized_program(chosen["program"], library)
        op_name = direct_program_operator_name(program)
        if op_name not in library:
            raise KeyError(f"Update action selected missing operator {op_name!r}.")
        best_record = library[op_name]
        update_cfg = replace(cfg, structural_risk_mode="repair")
        update_stats = apply_update_rule(boundary, best_record, inputs, targets, update_cfg, distance_scale)
        task_to_program[task_name] = direct_operator_program(best_record)
        return update_stats
    raise ValueError(f"Unknown chosen action {action!r}.")


def parse_materialized_program(program_text: str, library: dict[str, OperatorRecord]) -> Program:
    if program_text == "x":
        return ("var", 0)

    def parse_expr(text: str) -> Program:
        text = text.strip()
        if text == "x":
            return ("var", 0)
        if not text.endswith(")"):
            raise ValueError(f"Cannot parse program expression {text!r}.")
        open_index = text.find("(")
        if open_index <= 0:
            raise ValueError(f"Cannot parse program expression {text!r}.")
        op_name = text[:open_index]
        if op_name not in library:
            raise KeyError(f"Program references missing operator {op_name!r}.")
        inner = text[open_index + 1:-1]
        return ("op", op_name, (parse_expr(inner),))

    return parse_expr(program_text)


def run_late_abstraction(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_from_args(args)
    space = make_character_space(cfg.alphabet_size, cfg.include_separator)
    train_shifts = parse_csv_int_items(args.abstraction_train_shifts, "--abstraction-train-shifts")
    eval_shifts = parse_csv_int_items(args.abstraction_eval_shifts, "--abstraction-eval-shifts")
    if len(set(train_shifts)) != len(train_shifts):
        raise ValueError("--abstraction-train-shifts contains duplicate shifts.")
    if len(set(eval_shifts)) != len(eval_shifts):
        raise ValueError("--abstraction-eval-shifts contains duplicate shifts.")
    rows: list[dict[str, Any]] = []
    compression_rows: list[dict[str, Any]] = []
    for seed in range(args.seed_count):
        print(f"running_seed={seed}")
        set_seed(seed)
        boundary = LatentBoundary(space.num_tokens, cfg.code_dim).to(cfg.device)
        train_boundary(boundary, cfg)
        distance_scale = code_distance_scale(boundary)

        concrete_records: dict[int, OperatorRecord] = {}
        for shift in train_shifts:
            inputs, _, targets = make_shift_k_dataset([shift], space, cfg)
            record = new_operator(f"SHIFT_{shift}", cfg, f"OP_SHIFT_{shift}_{len(concrete_records)}")
            train_new_operator(boundary, record, inputs, targets, cfg, require_perfect=True)
            concrete_records[shift] = record

        parent = ClosedControlledUnaryOperator(cfg.code_dim, cfg.control_dim, cfg.hidden_dim).to(cfg.device)
        train_shift_k_parent(boundary, parent, train_shifts, space, cfg)

        concrete_parameters = len(concrete_records) * operator_parameter_count(cfg)
        parent_parameters = controlled_operator_parameter_count(cfg)
        compression_rows.append(
            {
                "seed": seed,
                "concrete_operator_count": len(concrete_records),
                "concrete_parameters": concrete_parameters,
                "parent_operator_count": 1,
                "parent_parameters": parent_parameters,
                "parameter_reduction": 1.0 - (parent_parameters / concrete_parameters),
            }
        )

        for shift in eval_shifts:
            parent_metrics = evaluate_shift_k_parent(boundary, parent, shift, space, cfg, distance_scale)
            concrete_metrics: dict[str, float] | None = None
            if shift in concrete_records:
                inputs, _, targets = make_shift_k_dataset([shift], space, cfg)
                concrete_metrics = evaluate_direct_operator(boundary, concrete_records[shift], inputs, targets, cfg)
                concrete_metrics = {
                    "accuracy": concrete_metrics["accuracy"],
                    "closure_norm": concrete_metrics["closure"] / distance_scale,
                    "manifold_norm": concrete_metrics["manifold"] / distance_scale,
                }
            rows.append(
                {
                    "seed": seed,
                    "shift": shift,
                    "seen_in_parent_training": shift in train_shifts,
                    "has_concrete_operator": shift in concrete_records,
                    "parent_acc": parent_metrics["accuracy"],
                    "parent_closure_norm": parent_metrics["closure_norm"],
                    "parent_manifold_norm": parent_metrics["manifold_norm"],
                    "concrete_acc": None if concrete_metrics is None else concrete_metrics["accuracy"],
                    "concrete_closure_norm": None if concrete_metrics is None else concrete_metrics["closure_norm"],
                    "concrete_manifold_norm": None if concrete_metrics is None else concrete_metrics["manifold_norm"],
                }
            )

    report = {
        "mode": "char_late_abstraction",
        "tokens": list(space.tokens),
        "train_shifts": train_shifts,
        "eval_shifts": eval_shifts,
        "config": serializable_config(args),
        "summary": summarize_late_abstraction(rows, compression_rows),
        "rows": rows,
        "compression_rows": compression_rows,
    }
    print_late_abstraction_summary(report)
    return report


def run_reasoner(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_from_args(args)
    space = make_character_space(cfg.alphabet_size, cfg.include_separator)
    stream = parse_csv_items(args.stream, "--stream")
    unknown_tasks = [task for task in stream if task not in TASKS]
    if unknown_tasks:
        raise ValueError(f"Unknown stream task(s): {unknown_tasks}. Available: {sorted(TASKS)}")
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_count):
        print(f"running_seed={seed}")
        set_seed(seed)
        boundary = LatentBoundary(space.num_tokens, cfg.code_dim).to(cfg.device)
        train_boundary(boundary, cfg)
        distance_scale = code_distance_scale(boundary)
        library: dict[str, OperatorRecord] = {}
        task_to_program: dict[str, Program] = {}
        learned_tasks: list[str] = []
        for task_name in stream:
            candidates = build_candidates(boundary, library, task_to_program, learned_tasks, task_name, space, cfg, distance_scale)
            chosen = choose_candidate(candidates)
            update_stats = materialize_choice(boundary, library, task_to_program, task_name, chosen, space, cfg, distance_scale)
            if task_name not in task_to_program:
                raise RuntimeError(f"Task {task_name} was not materialized into a program.")
            final_metrics = evaluate_state(
                boundary,
                library,
                task_to_program,
                learned_tasks,
                task_name,
                task_to_program[task_name],
                space,
                cfg,
                distance_scale,
            )
            learned_tasks.append(task_name)
            rows.append(
                {
                    "seed": seed,
                    "task": task_name,
                    "chosen_action": chosen["action"],
                    "program": program_to_str(task_to_program[task_name]),
                    "safe": chosen["safe"],
                    "new_acc": final_metrics["new_acc"],
                    "old_min_acc": final_metrics["old_min_acc"],
                    "new_closure_norm": final_metrics["new_closure_norm"],
                    "old_mean_closure_norm": final_metrics["old_mean_closure_norm"],
                    "operator_count": len(library),
                    "new_parameters_added": sum(record.parameter_count for record in library.values()),
                    "candidate_count": len(candidates),
                    "update_stats": update_stats,
                    "candidates": candidates,
                }
            )
    report = {
        "mode": "char_semantic_reasoner",
        "tokens": list(space.tokens),
        "config": serializable_config(args),
        "summary": summarize_reasoner(rows),
        "rows": rows,
    }
    print_reasoner_summary(report)
    return report


def run_forced_conflict(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_from_args(args)
    space = make_character_space(cfg.alphabet_size, cfg.include_separator)
    base_task = args.base_task.upper()
    conflict_task = args.conflict_task.upper()
    if base_task not in TASKS:
        raise ValueError(f"Unknown --base-task {base_task!r}. Available: {sorted(TASKS)}")
    if conflict_task not in TASKS:
        raise ValueError(f"Unknown --conflict-task {conflict_task!r}. Available: {sorted(TASKS)}")
    rules = parse_csv_rule_items(args.forced_conflict_rules, "--forced-conflict-rules")
    valid_rules = {"adam", "structural_repair", "structural_protect"}
    unknown_rules = [rule for rule in rules if rule not in valid_rules]
    if unknown_rules:
        raise ValueError(f"Unknown forced conflict rule(s): {unknown_rules}. Available: {sorted(valid_rules)}")
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_count):
        print(f"running_seed={seed}")
        set_seed(seed)
        boundary = LatentBoundary(space.num_tokens, cfg.code_dim).to(cfg.device)
        train_boundary(boundary, cfg)
        distance_scale = code_distance_scale(boundary)
        base_inputs, base_targets = make_task_data(base_task, space, cfg.device)
        conflict_inputs, conflict_targets = make_task_data(conflict_task, space, cfg.device)
        base_record = new_operator(base_task, cfg, f"OP_{base_task}_0")
        train_new_operator(boundary, base_record, base_inputs, base_targets, cfg, require_perfect=True)
        base_old = evaluate_direct_operator(boundary, base_record, base_inputs, base_targets, cfg)
        base_new = evaluate_direct_operator(boundary, base_record, conflict_inputs, conflict_targets, cfg)
        for rule in rules:
            record = clone_library({base_record.name: base_record})[base_record.name]
            if rule == "adam":
                rule_cfg = replace(cfg, repair_update_rule="adam")
            elif rule == "structural_repair":
                rule_cfg = replace(cfg, repair_update_rule="structural_gated", structural_risk_mode="repair")
            elif rule == "structural_protect":
                rule_cfg = replace(cfg, repair_update_rule="structural_gated", structural_risk_mode="protect")
            else:
                raise ValueError(f"Unknown forced conflict rule {rule!r}.")
            update_stats = apply_update_rule(boundary, record, conflict_inputs, conflict_targets, rule_cfg, distance_scale)
            old_after = evaluate_direct_operator(boundary, record, base_inputs, base_targets, cfg)
            new_after = evaluate_direct_operator(boundary, record, conflict_inputs, conflict_targets, cfg)
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "base_task": base_task,
                    "conflict_task": conflict_task,
                    "base_old_acc": base_old["accuracy"],
                    "base_new_acc": base_new["accuracy"],
                    "old_acc": old_after["accuracy"],
                    "new_acc": new_after["accuracy"],
                    "forgetting": max(0.0, base_old["accuracy"] - old_after["accuracy"]),
                    "new_gain": max(0.0, new_after["accuracy"] - base_new["accuracy"]),
                    "old_closure_norm": old_after["closure"] / distance_scale,
                    "new_closure_norm": new_after["closure"] / distance_scale,
                    "update_stats": update_stats,
                }
            )
    report = {
        "mode": "char_forced_conflict",
        "tokens": list(space.tokens),
        "config": serializable_config(args),
        "summary": summarize_forced(rows),
        "rows": rows,
    }
    print_forced_summary(report)
    return report


def collect_policy_examples(
    args: argparse.Namespace,
    cfg: Config,
    space: CharacterSpace,
    streams: list[list[StreamEvent]],
    seed_offset: int,
) -> tuple[list[list[float]], list[int], list[dict[str, Any]]]:
    features: list[list[float]] = []
    labels: list[int] = []
    rows: list[dict[str, Any]] = []
    parameter_scale = float(operator_parameter_count(cfg))
    for seed in range(args.seed_count):
        for stream_index, events in enumerate(streams):
            actual_seed = seed_offset + seed
            print(f"collect_policy_seed={actual_seed} stream={stream_index}")
            set_seed(actual_seed)
            boundary = LatentBoundary(space.num_tokens, cfg.code_dim).to(cfg.device)
            train_boundary(boundary, cfg)
            distance_scale = code_distance_scale(boundary)
            library: dict[str, OperatorRecord] = {}
            task_to_program: dict[str, Program] = {}
            learned_tasks: list[str] = []
            for step_index, event in enumerate(events):
                repaired_operator: str | None = None
                if event.repair:
                    repaired_operator = inject_repair_noise(
                        library,
                        task_to_program,
                        event.task_name,
                        args.policy_repair_noise_std,
                    )
                candidates = build_candidates(
                    boundary,
                    library,
                    task_to_program,
                    learned_tasks,
                    event.task_name,
                    space,
                    cfg,
                    distance_scale,
                )
                teacher = choose_candidate(candidates)
                features.append(policy_feature_vector(candidates, parameter_scale))
                labels.append(ACTION_TO_INDEX[teacher["action"]])
                update_stats = materialize_choice(
                    boundary,
                    library,
                    task_to_program,
                    event.task_name,
                    teacher,
                    space,
                    cfg,
                    distance_scale,
                )
                if event.task_name not in task_to_program:
                    raise RuntimeError(f"Policy teacher failed to materialize {event.task_name}.")
                final_metrics = evaluate_state(
                    boundary,
                    library,
                    task_to_program,
                    learned_tasks,
                    event.task_name,
                    task_to_program[event.task_name],
                    space,
                    cfg,
                    distance_scale,
                )
                if not event.repair and event.task_name not in learned_tasks:
                    learned_tasks.append(event.task_name)
                rows.append(
                    {
                        "phase": "train_teacher",
                        "seed": actual_seed,
                        "stream_index": stream_index,
                        "step_index": step_index,
                        "event": event.label,
                        "task": event.task_name,
                        "repair": event.repair,
                        "repaired_operator": repaired_operator,
                        "teacher_action": teacher["action"],
                        "chosen_action": teacher["action"],
                        "action_correct": True,
                        "raw_action": teacher["action"],
                        "masked_preference": False,
                        "safe": teacher["safe"],
                        "program": program_to_str(task_to_program[event.task_name]),
                        "new_acc": final_metrics["new_acc"],
                        "old_min_acc": final_metrics["old_min_acc"],
                        "new_closure_norm": final_metrics["new_closure_norm"],
                        "old_mean_closure_norm": final_metrics["old_mean_closure_norm"],
                        "operator_count": len(library),
                        "new_parameters_added": sum(record.parameter_count for record in library.values()),
                        "candidate_count": len(candidates),
                        "update_stats": update_stats,
                    }
                )
    return features, labels, rows


def evaluate_learned_policy(
    args: argparse.Namespace,
    cfg: Config,
    space: CharacterSpace,
    streams: list[list[StreamEvent]],
    policy: LearnedActionPolicy,
    seed_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parameter_scale = float(operator_parameter_count(cfg))
    for seed in range(args.seed_count):
        for stream_index, events in enumerate(streams):
            actual_seed = seed_offset + seed
            print(f"eval_policy_seed={actual_seed} stream={stream_index}")
            set_seed(actual_seed)
            boundary = LatentBoundary(space.num_tokens, cfg.code_dim).to(cfg.device)
            train_boundary(boundary, cfg)
            distance_scale = code_distance_scale(boundary)
            library: dict[str, OperatorRecord] = {}
            task_to_program: dict[str, Program] = {}
            learned_tasks: list[str] = []
            for step_index, event in enumerate(events):
                repaired_operator: str | None = None
                if event.repair:
                    repaired_operator = inject_repair_noise(
                        library,
                        task_to_program,
                        event.task_name,
                        args.policy_repair_noise_std,
                    )
                candidates = build_candidates(
                    boundary,
                    library,
                    task_to_program,
                    learned_tasks,
                    event.task_name,
                    space,
                    cfg,
                    distance_scale,
                )
                teacher = choose_candidate(candidates)
                chosen, raw_action, chosen_action, masked_preference = select_policy_candidate(
                    policy,
                    candidates,
                    parameter_scale,
                    cfg.device,
                )
                update_stats = materialize_choice(
                    boundary,
                    library,
                    task_to_program,
                    event.task_name,
                    chosen,
                    space,
                    cfg,
                    distance_scale,
                )
                if event.task_name not in task_to_program:
                    raise RuntimeError(f"Learned policy failed to materialize {event.task_name}.")
                final_metrics = evaluate_state(
                    boundary,
                    library,
                    task_to_program,
                    learned_tasks,
                    event.task_name,
                    task_to_program[event.task_name],
                    space,
                    cfg,
                    distance_scale,
                )
                if not event.repair and event.task_name not in learned_tasks:
                    learned_tasks.append(event.task_name)
                rows.append(
                    {
                        "phase": "test_learned_policy",
                        "seed": actual_seed,
                        "stream_index": stream_index,
                        "step_index": step_index,
                        "event": event.label,
                        "task": event.task_name,
                        "repair": event.repair,
                        "repaired_operator": repaired_operator,
                        "teacher_action": teacher["action"],
                        "chosen_action": chosen_action,
                        "raw_action": raw_action,
                        "masked_preference": masked_preference,
                        "action_correct": chosen_action == teacher["action"],
                        "safe": chosen["safe"],
                        "program": program_to_str(task_to_program[event.task_name]),
                        "new_acc": final_metrics["new_acc"],
                        "old_min_acc": final_metrics["old_min_acc"],
                        "new_closure_norm": final_metrics["new_closure_norm"],
                        "old_mean_closure_norm": final_metrics["old_mean_closure_norm"],
                        "operator_count": len(library),
                        "new_parameters_added": sum(record.parameter_count for record in library.values()),
                        "candidate_count": len(candidates),
                        "update_stats": update_stats,
                    }
                )
    return rows


def run_learned_policy(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_from_args(args)
    space = make_character_space(cfg.alphabet_size, cfg.include_separator)
    train_streams = parse_stream_specs(args.policy_train_streams, "--policy-train-streams")
    test_streams = parse_stream_specs(args.policy_test_streams, "--policy-test-streams")
    train_features, train_labels, train_rows = collect_policy_examples(args, cfg, space, train_streams, seed_offset=0)
    policy = train_action_policy(train_features, train_labels, args, cfg)
    test_rows = evaluate_learned_policy(args, cfg, space, test_streams, policy, seed_offset=args.policy_test_seed_offset)
    rows = train_rows + test_rows
    report = {
        "mode": "char_learned_policy_reasoner",
        "tokens": list(space.tokens),
        "actions": list(ACTION_NAMES),
        "policy_feature_dim": len(train_features[0]),
        "policy_train_action_counts": count_values([ACTION_NAMES[label] for label in train_labels]),
        "config": serializable_config(args),
        "summary": summarize_learned_policy(test_rows),
        "rows": rows,
    }
    print_learned_policy_summary(report)
    return report


def summarize_reasoner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No reasoner rows were produced.")
    tasks = sorted({row["task"] for row in rows})
    summary: dict[str, Any] = {}
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        summary[task] = {
            "action_counts": count_values([row["chosen_action"] for row in task_rows]),
            "operator_count_mean": float(np.mean([row["operator_count"] for row in task_rows])),
            "new_acc_mean": float(np.mean([row["new_acc"] for row in task_rows])),
            "old_min_acc_mean": float(np.mean([row["old_min_acc"] for row in task_rows])),
            "new_closure_norm_mean": float(np.mean([row["new_closure_norm"] for row in task_rows])),
        }
    final_rows = [row for row in rows if row["task"] == rows[-1]["task"]]
    summary["_final"] = {
        "operator_count_mean": float(np.mean([row["operator_count"] for row in final_rows])),
        "new_parameters_added_mean": float(np.mean([row["new_parameters_added"] for row in final_rows])),
    }
    return summary


def summarize_forced(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No forced-conflict rows were produced.")
    rules = sorted({row["rule"] for row in rows})
    metrics = ["old_acc", "new_acc", "forgetting", "new_gain", "old_closure_norm", "new_closure_norm"]
    summary: dict[str, Any] = {}
    for rule in rules:
        rule_rows = [row for row in rows if row["rule"] == rule]
        summary[rule] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in rule_rows]
            summary[rule][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def summarize_late_abstraction(
    rows: list[dict[str, Any]],
    compression_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No late-abstraction rows were produced.")
    if not compression_rows:
        raise ValueError("No compression rows were produced.")
    shifts = sorted({int(row["shift"]) for row in rows})
    summary: dict[str, Any] = {"shifts": {}, "compression": {}}
    for shift in shifts:
        shift_rows = [row for row in rows if int(row["shift"]) == shift]
        summary["shifts"][str(shift)] = {
            "seen_in_parent_training": bool(shift_rows[0]["seen_in_parent_training"]),
            "has_concrete_operator": bool(shift_rows[0]["has_concrete_operator"]),
            "parent_acc_mean": float(np.mean([row["parent_acc"] for row in shift_rows])),
            "parent_acc_std": float(np.std([row["parent_acc"] for row in shift_rows])),
            "parent_closure_norm_mean": float(np.mean([row["parent_closure_norm"] for row in shift_rows])),
            "parent_closure_norm_std": float(np.std([row["parent_closure_norm"] for row in shift_rows])),
            "parent_manifold_norm_mean": float(np.mean([row["parent_manifold_norm"] for row in shift_rows])),
            "parent_manifold_norm_std": float(np.std([row["parent_manifold_norm"] for row in shift_rows])),
        }
        concrete_values = [row["concrete_acc"] for row in shift_rows if row["concrete_acc"] is not None]
        if concrete_values:
            summary["shifts"][str(shift)]["concrete_acc_mean"] = float(np.mean(concrete_values))
            summary["shifts"][str(shift)]["concrete_acc_std"] = float(np.std(concrete_values))
        else:
            summary["shifts"][str(shift)]["concrete_acc_mean"] = None
            summary["shifts"][str(shift)]["concrete_acc_std"] = None
    for metric in ["concrete_operator_count", "concrete_parameters", "parent_parameters", "parameter_reduction"]:
        values = [float(row[metric]) for row in compression_rows]
        summary["compression"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    seen_rows = [row for row in rows if row["seen_in_parent_training"]]
    heldout_rows = [row for row in rows if not row["seen_in_parent_training"]]
    summary["aggregate"] = {
        "seen_parent_acc_mean": float(np.mean([row["parent_acc"] for row in seen_rows])) if seen_rows else None,
        "heldout_parent_acc_mean": float(np.mean([row["parent_acc"] for row in heldout_rows])) if heldout_rows else None,
        "seen_parent_closure_norm_mean": float(np.mean([row["parent_closure_norm"] for row in seen_rows])) if seen_rows else None,
        "heldout_parent_closure_norm_mean": float(np.mean([row["parent_closure_norm"] for row in heldout_rows])) if heldout_rows else None,
    }
    return summary


def summarize_learned_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No learned-policy rows were produced.")
    final_rows: list[dict[str, Any]] = []
    stream_keys = sorted({(row["seed"], row["stream_index"]) for row in rows})
    for seed, stream_index in stream_keys:
        stream_rows = [row for row in rows if row["seed"] == seed and row["stream_index"] == stream_index]
        if not stream_rows:
            raise RuntimeError(f"Missing learned-policy rows for seed={seed}, stream={stream_index}.")
        final_rows.append(max(stream_rows, key=lambda row: row["step_index"]))
    summary: dict[str, Any] = {
        "overall": {
            "action_accuracy_mean": float(np.mean([float(row["action_correct"]) for row in rows])),
            "raw_masked_preference_rate": float(np.mean([float(row["masked_preference"]) for row in rows])),
            "unsafe_choice_rate": float(np.mean([float(not row["safe"]) for row in rows])),
            "new_acc_mean": float(np.mean([row["new_acc"] for row in rows])),
            "old_min_acc_mean": float(np.mean([row["old_min_acc"] for row in rows])),
            "new_closure_norm_mean": float(np.mean([row["new_closure_norm"] for row in rows])),
            "final_operator_count_mean": float(np.mean([row["operator_count"] for row in final_rows])),
            "final_new_parameters_mean": float(np.mean([row["new_parameters_added"] for row in final_rows])),
        },
        "chosen_action_counts": count_values([row["chosen_action"] for row in rows]),
        "teacher_action_counts": count_values([row["teacher_action"] for row in rows]),
        "events": {},
    }
    for event in sorted({row["event"] for row in rows}):
        event_rows = [row for row in rows if row["event"] == event]
        summary["events"][event] = {
            "chosen_action_counts": count_values([row["chosen_action"] for row in event_rows]),
            "teacher_action_counts": count_values([row["teacher_action"] for row in event_rows]),
            "action_accuracy_mean": float(np.mean([float(row["action_correct"]) for row in event_rows])),
            "new_acc_mean": float(np.mean([row["new_acc"] for row in event_rows])),
            "old_min_acc_mean": float(np.mean([row["old_min_acc"] for row in event_rows])),
            "new_closure_norm_mean": float(np.mean([row["new_closure_norm"] for row in event_rows])),
            "operator_count_mean": float(np.mean([row["operator_count"] for row in event_rows])),
        }
    return summary


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def print_reasoner_summary(report: dict[str, Any]) -> None:
    print("\nCHAR SEMANTIC REASONER SUMMARY")
    print("=" * 118)
    print(f"{'task':<22} {'actions':<36} {'new_acc':<10} {'old_min':<10} {'closure':<10} {'ops':<10}")
    print("-" * 118)
    for task, item in report["summary"].items():
        if task == "_final":
            continue
        print(
            f"{task:<22} {str(item['action_counts']):<36} "
            f"{item['new_acc_mean']:<10.3f} {item['old_min_acc_mean']:<10.3f} "
            f"{item['new_closure_norm_mean']:<10.4f} {item['operator_count_mean']:<10.2f}"
        )
    print("-" * 118)
    print(f"final_operator_count={report['summary']['_final']['operator_count_mean']:.2f}")
    print(f"final_new_parameters={report['summary']['_final']['new_parameters_added_mean']:.2f}")
    print("=" * 118)


def print_forced_summary(report: dict[str, Any]) -> None:
    print("\nCHAR FORCED CONFLICT SUMMARY")
    print("=" * 118)
    print(f"{'metric':<24}" + "".join(f"{rule:<26}" for rule in sorted(report["summary"])))
    print("-" * 118)
    metrics = ["old_acc", "new_acc", "forgetting", "new_gain", "old_closure_norm", "new_closure_norm"]
    for metric in metrics:
        row = f"{metric:<24}"
        for rule in sorted(report["summary"]):
            item = report["summary"][rule][metric]
            row += f"{item['mean']:.4f} +/- {item['std']:.4f}".ljust(26)
        print(row)
    print("=" * 118)


def print_late_abstraction_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\nCHAR LATE ABSTRACTION SUMMARY")
    print("=" * 118)
    print(f"train_shifts={report['train_shifts']} eval_shifts={report['eval_shifts']}")
    compression = summary["compression"]
    print(
        "compression: "
        f"concrete_ops={compression['concrete_operator_count']['mean']:.2f}, "
        f"concrete_params={compression['concrete_parameters']['mean']:.2f}, "
        f"parent_params={compression['parent_parameters']['mean']:.2f}, "
        f"parameter_reduction={compression['parameter_reduction']['mean']:.4f}"
    )
    aggregate = summary["aggregate"]
    print(
        "aggregate: "
        f"seen_acc={format_optional_float(aggregate['seen_parent_acc_mean'])}, "
        f"heldout_acc={format_optional_float(aggregate['heldout_parent_acc_mean'])}, "
        f"seen_closure={format_optional_float(aggregate['seen_parent_closure_norm_mean'])}, "
        f"heldout_closure={format_optional_float(aggregate['heldout_parent_closure_norm_mean'])}"
    )
    print("-" * 118)
    print(f"{'shift':<8} {'seen':<6} {'parent_acc':<20} {'parent_closure':<20} {'concrete_acc':<16}")
    print("-" * 118)
    for shift, item in summary["shifts"].items():
        concrete_text = (
            "n/a"
            if item["concrete_acc_mean"] is None
            else f"{item['concrete_acc_mean']:.4f} +/- {item['concrete_acc_std']:.4f}"
        )
        print(
            f"{shift:<8} {str(item['seen_in_parent_training']):<6} "
            f"{item['parent_acc_mean']:.4f} +/- {item['parent_acc_std']:.4f}".ljust(20)
            + " "
            + f"{item['parent_closure_norm_mean']:.4f} +/- {item['parent_closure_norm_std']:.4f}".ljust(20)
            + " "
            + f"{concrete_text:<16}"
        )
    print("=" * 118)


def print_learned_policy_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    overall = summary["overall"]
    print("\nCHAR LEARNED LATENT-GEOMETRY POLICY SUMMARY")
    print("=" * 132)
    print(f"train_action_counts={report['policy_train_action_counts']}")
    print(
        "overall: "
        f"action_acc={overall['action_accuracy_mean']:.4f}, "
        f"masked_pref={overall['raw_masked_preference_rate']:.4f}, "
        f"unsafe={overall['unsafe_choice_rate']:.4f}, "
        f"new_acc={overall['new_acc_mean']:.4f}, "
        f"old_min={overall['old_min_acc_mean']:.4f}, "
        f"closure={overall['new_closure_norm_mean']:.4f}, "
        f"final_ops={overall['final_operator_count_mean']:.2f}"
    )
    print(f"teacher_actions={summary['teacher_action_counts']}")
    print(f"chosen_actions={summary['chosen_action_counts']}")
    print("-" * 132)
    print(f"{'event':<24} {'teacher':<28} {'chosen':<28} {'act_acc':<10} {'new_acc':<10} {'old_min':<10} {'closure':<10} {'ops':<8}")
    print("-" * 132)
    for event, item in summary["events"].items():
        print(
            f"{event:<24} {str(item['teacher_action_counts']):<28} {str(item['chosen_action_counts']):<28} "
            f"{item['action_accuracy_mean']:<10.3f} {item['new_acc_mean']:<10.3f} "
            f"{item['old_min_acc_mean']:<10.3f} {item['new_closure_norm_mean']:<10.4f} "
            f"{item['operator_count_mean']:<8.2f}"
        )
    print("=" * 132)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def write_json_report(report: dict[str, Any], output_json: Path) -> None:
    if output_json.exists():
        raise FileExistsError(f"output-json already exists: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "output_json"
    }


def config_from_args(args: argparse.Namespace) -> Config:
    device = resolve_device(args.device)
    return Config(
        alphabet_size=args.alphabet_size,
        include_separator=args.include_separator,
        code_dim=args.code_dim,
        hidden_dim=args.hidden_dim,
        boundary_epochs=args.boundary_epochs,
        operator_epochs=args.operator_epochs,
        update_epochs=args.update_epochs,
        abstraction_epochs=args.abstraction_epochs,
        boundary_lr=args.boundary_lr,
        operator_lr=args.operator_lr,
        update_lr=args.update_lr,
        abstraction_lr=args.abstraction_lr,
        lambda_closure=args.lambda_closure,
        separation_margin=args.separation_margin,
        separation_weight=args.separation_weight,
        grad_memory_beta=args.grad_memory_beta,
        reuse_acc_threshold=args.reuse_acc_threshold,
        reuse_closure_threshold=args.reuse_closure_threshold,
        update_closure_low=args.update_closure_low,
        update_closure_high=args.update_closure_high,
        repair_update_rule=args.repair_update_rule,
        projection_norm_floor=args.projection_norm_floor,
        min_responsibility=args.min_responsibility,
        structural_risk_signal=args.structural_risk_signal,
        structural_need_signal=args.structural_need_signal,
        structural_risk_mode=args.structural_risk_mode,
        structural_risk_power=args.structural_risk_power,
        structural_need_power=args.structural_need_power,
        control_dim=args.control_dim,
        identity_weight=args.identity_weight,
        composition_target_weight=args.composition_target_weight,
        composition_agreement_weight=args.composition_agreement_weight,
        search_depth=args.search_depth,
        max_programs=args.max_programs,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forced-conflict", action="store_true")
    parser.add_argument("--late-abstraction", action="store_true")
    parser.add_argument("--learned-policy", action="store_true")
    parser.add_argument("--stream", default=DEFAULT_STREAM)
    parser.add_argument("--base-task", default="SHIFT")
    parser.add_argument("--conflict-task", default="REVERSE_SHIFT")
    parser.add_argument("--forced-conflict-rules", default="adam,structural_repair,structural_protect")
    parser.add_argument("--abstraction-train-shifts", default="1,-1,2")
    parser.add_argument("--abstraction-eval-shifts", default="1,-1,2,3,-2,0")
    parser.add_argument("--policy-train-streams", default=DEFAULT_POLICY_TRAIN_STREAMS)
    parser.add_argument("--policy-test-streams", default=DEFAULT_POLICY_TEST_STREAMS)
    parser.add_argument("--policy-epochs", type=int, default=800)
    parser.add_argument("--policy-lr", type=float, default=0.01)
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--policy-seed", type=int, default=12345)
    parser.add_argument("--policy-test-seed-offset", type=int, default=10000)
    parser.add_argument("--policy-repair-noise-std", type=float, default=0.03)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--alphabet-size", type=int, default=5)
    parser.add_argument("--include-separator", action="store_true")
    parser.add_argument("--code-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--boundary-epochs", type=int, default=700)
    parser.add_argument("--operator-epochs", type=int, default=1200)
    parser.add_argument("--update-epochs", type=int, default=1000)
    parser.add_argument("--abstraction-epochs", type=int, default=2000)
    parser.add_argument("--boundary-lr", type=float, default=0.01)
    parser.add_argument("--operator-lr", type=float, default=0.01)
    parser.add_argument("--update-lr", type=float, default=0.005)
    parser.add_argument("--abstraction-lr", type=float, default=0.01)
    parser.add_argument("--lambda-closure", type=float, default=10.0)
    parser.add_argument("--separation-margin", type=float, default=2.0)
    parser.add_argument("--separation-weight", type=float, default=0.5)
    parser.add_argument("--grad-memory-beta", type=float, default=0.98)
    parser.add_argument("--reuse-acc-threshold", type=float, default=0.98)
    parser.add_argument("--reuse-closure-threshold", type=float, default=0.02)
    parser.add_argument("--update-closure-low", type=float, default=0.001)
    parser.add_argument("--update-closure-high", type=float, default=0.65)
    parser.add_argument("--repair-update-rule", choices=("adam", "structural_gated"), default="structural_gated")
    parser.add_argument("--projection-norm-floor", type=float, default=1e-8)
    parser.add_argument("--min-responsibility", type=float, default=1e-12)
    parser.add_argument(
        "--structural-risk-signal",
        choices=("activation_downstream", "activation_weight_product"),
        default="activation_weight_product",
    )
    parser.add_argument("--structural-need-signal", choices=("gradient", "responsibility"), default="responsibility")
    parser.add_argument("--structural-risk-mode", choices=("repair", "protect"), default="repair")
    parser.add_argument("--structural-risk-power", type=float, default=1.0)
    parser.add_argument("--structural-need-power", type=float, default=1.0)
    parser.add_argument("--control-dim", type=int, default=4)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--composition-target-weight", type=float, default=1.0)
    parser.add_argument("--composition-agreement-weight", type=float, default=0.25)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--max-programs", type=int, default=100000)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    if args.code_dim <= 0 or args.hidden_dim <= 0:
        raise ValueError("--code-dim and --hidden-dim must be positive.")
    if args.boundary_epochs <= 0 or args.operator_epochs <= 0 or args.update_epochs <= 0 or args.abstraction_epochs <= 0:
        raise ValueError("epoch counts must be positive.")
    if args.control_dim < 2:
        raise ValueError("--control-dim must be at least 2.")
    if args.identity_weight < 0.0:
        raise ValueError("--identity-weight must be non-negative.")
    if args.composition_target_weight < 0.0:
        raise ValueError("--composition-target-weight must be non-negative.")
    if args.composition_agreement_weight < 0.0:
        raise ValueError("--composition-agreement-weight must be non-negative.")
    if args.policy_epochs <= 0:
        raise ValueError("--policy-epochs must be positive.")
    if args.policy_lr <= 0.0:
        raise ValueError("--policy-lr must be positive.")
    if args.policy_hidden_dim <= 0:
        raise ValueError("--policy-hidden-dim must be positive.")
    if args.policy_repair_noise_std <= 0.0:
        raise ValueError("--policy-repair-noise-std must be positive.")
    if args.projection_norm_floor <= 0.0:
        raise ValueError("--projection-norm-floor must be positive.")
    if args.min_responsibility <= 0.0:
        raise ValueError("--min-responsibility must be positive.")
    if args.structural_risk_power <= 0.0 or args.structural_need_power <= 0.0:
        raise ValueError("structural powers must be positive.")
    selected_modes = int(args.forced_conflict) + int(args.late_abstraction) + int(args.learned_policy)
    if selected_modes > 1:
        raise ValueError("Choose only one mode: reasoner, --forced-conflict, --late-abstraction, or --learned-policy.")
    return args


def main() -> None:
    args = parse_args()
    if args.learned_policy:
        report = run_learned_policy(args)
    elif args.late_abstraction:
        report = run_late_abstraction(args)
    elif args.forced_conflict:
        report = run_forced_conflict(args)
    else:
        report = run_reasoner(args)
    if args.output_json is not None:
        write_json_report(report, args.output_json)
        print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
