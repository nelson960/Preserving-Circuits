from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OpsConfig:
    num_digits: int = 5
    sequence_digits: int = 3
    hidden_dim: int = 64
    seed: int = 7
    use_position_flags: bool = False

    @property
    def op_names(self) -> tuple[str, ...]:
        return ("COPY0", "COPY1", "COPY2", "ADD01", "MAX", "ADD12")

    @property
    def base_ops(self) -> tuple[str, ...]:
        return ("COPY0", "COPY1", "ADD01", "MAX")

    @property
    def new_op(self) -> str:
        return "ADD12"

    @property
    def input_dim(self) -> int:
        return (
            len(self.op_names)
            + self.sequence_digits * self.num_digits
            + self.position_flag_dim
        )

    @property
    def position_flag_dim(self) -> int:
        if not self.use_position_flags:
            return 0
        return len(self.add_position_pairs)

    @property
    def add_position_pairs(self) -> tuple[tuple[int, int], ...]:
        return ((0, 1), (1, 2))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    log_every: int
    protection_strength: float = 0.0
    protected_fraction: float = 0.0
    needed_fraction: float = 0.0
    quiet: bool = False


@dataclass(frozen=True)
class OpsDataset:
    x: np.ndarray
    y: np.ndarray
    op_ids: np.ndarray
    op_names: tuple[str, ...]


@dataclass
class TinyMLP:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class FactorizedAddMLP:
    W_router_ADD01: np.ndarray
    b_router_ADD01: np.ndarray
    W_router_ADD12: np.ndarray
    b_router_ADD12: np.ndarray
    W_op: np.ndarray
    b_op: np.ndarray
    W_out: np.ndarray
    b_out: np.ndarray


@dataclass(frozen=True)
class MultiRouteOpSpec:
    name: str
    kind: str
    operands: tuple[int, ...]


@dataclass(frozen=True)
class MultiRouteDataset:
    digit_x: np.ndarray
    y: np.ndarray
    op_spec: MultiRouteOpSpec


@dataclass
class MultiRouteFactorizedMLP:
    W_router: dict[str, np.ndarray]
    b_router: dict[str, np.ndarray]
    W_op: np.ndarray
    b_op: np.ndarray
    W_out: np.ndarray
    b_out: np.ndarray


@dataclass
class MultiRouteGrads:
    W_router: dict[str, np.ndarray]
    b_router: dict[str, np.ndarray]
    W_op: np.ndarray
    b_op: np.ndarray
    W_out: np.ndarray
    b_out: np.ndarray


@dataclass(frozen=True)
class SubspaceBasis:
    basis: np.ndarray
    rank: int
    retained_energy: float


SCORE_NAMES = ("A", "D", "E", "AE", "ADE")
DAMAGE_NAMES = ("loss_attribution", "fading", "readout_damage", "total_drift")
SUBSPACE_ENERGY = 0.95
BLEND_LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ONLINE_BLEND_LAMBDAS = (2.0, 4.0, 8.0)
ONLINE_BLEND_RECOMPUTE_EVERY = 100
FAMILY_FRACTION = 0.25
FAMILY_BLEND_LAMBDAS = (2.0, 4.0, 8.0)
MEANING_TRANSFORM_ALPHAS = (0.1, 0.3)
MEANING_TRANSFORM_RANKS = (1, 2, 4)
MEANING_ACTIVATION_TOP_FRACTION = 0.25
FUNCTIONAL_TRANSFORM_EPOCHS = 400
FUNCTIONAL_TRANSFORM_LR = 0.05
FUNCTIONAL_TRANSFORM_NEW_WEIGHT = 1.0
FACTORIZED_ADD_OPS = ("ADD01", "ADD12")
FACTORIZED_PARAM_KEYS = (
    "W_router_ADD01",
    "b_router_ADD01",
    "W_router_ADD12",
    "b_router_ADD12",
    "W_op",
    "b_op",
    "W_out",
    "b_out",
)
FACTORIZED_ADD01_SHARED_TRAIN_KEYS = (
    "W_router_ADD01",
    "b_router_ADD01",
    "W_op",
    "b_op",
    "W_out",
    "b_out",
)
FACTORIZED_ADD12_ROUTER_KEYS = ("W_router_ADD12", "b_router_ADD12")
FACTORIZED_SHARED_KEYS = ("W_op", "b_op", "W_out", "b_out")
FACTORIZED_ALIGNMENT_TARGETS = ("route", "op")
FACTORIZED_ALIGNMENT_WEIGHTS = (0.1, 1.0, 10.0)
FACTORIZED_CONSOLIDATION_OBJECTIVES = ("new_ce", "balanced_ce", "balanced_ce_align")
FACTORIZED_CONSOLIDATION_ALIGNMENT_WEIGHT = 10.0
FACTORIZED_CONSOLIDATION_STEPS = 400
FACTORIZED_CONSOLIDATION_LR = 0.005
FACTORIZED_CONSOLIDATION_LOW_LR = 0.0005
FACTORIZED_CONSOLIDATION_MAX_ALIGNMENT_DROP = 0.02
FACTORIZED_TANGENT_EPSILON = 1e-4
DEFAULT_STRESS_HIDDEN_DIMS = (64, 16)
DEFAULT_STRESS_LEARNING_RATES = (0.02, 0.05, 0.1)
DEFAULT_STRESS_STEPS = (400, 1000)
DEFAULT_MULTI_ROUTE_ADDITION_SPECS = (
    MultiRouteOpSpec("ADD01", "add", (0, 1)),
    MultiRouteOpSpec("ADD12", "add", (1, 2)),
    MultiRouteOpSpec("ADD02", "add", (0, 2)),
)
DEFAULT_MULTI_ROUTE_BASE_NAME = "ADD01"
DEFAULT_MULTI_ROUTE_ALIGNMENT_WEIGHT = 10.0
SCORE_EPSILON = 1e-12


def make_model(config: OpsConfig) -> TinyMLP:
    rng = np.random.default_rng(config.seed)
    return TinyMLP(
        W1=rng.normal(0.0, 1.0 / np.sqrt(config.input_dim), (config.input_dim, config.hidden_dim)),
        b1=np.zeros(config.hidden_dim),
        W2=rng.normal(0.0, 1.0 / np.sqrt(config.hidden_dim), (config.hidden_dim, config.num_digits)),
        b2=np.zeros(config.num_digits),
    )


def clone_model(model: TinyMLP) -> TinyMLP:
    return TinyMLP(
        W1=model.W1.copy(),
        b1=model.b1.copy(),
        W2=model.W2.copy(),
        b2=model.b2.copy(),
    )


def make_factorized_add_model(config: OpsConfig) -> FactorizedAddMLP:
    rng = np.random.default_rng(config.seed)
    digit_input_dim = config.sequence_digits * config.num_digits
    hidden_dim = config.hidden_dim
    return FactorizedAddMLP(
        W_router_ADD01=rng.normal(
            0.0,
            1.0 / np.sqrt(digit_input_dim),
            (digit_input_dim, hidden_dim),
        ),
        b_router_ADD01=np.zeros(hidden_dim),
        W_router_ADD12=rng.normal(
            0.0,
            1.0 / np.sqrt(digit_input_dim),
            (digit_input_dim, hidden_dim),
        ),
        b_router_ADD12=np.zeros(hidden_dim),
        W_op=rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, hidden_dim)),
        b_op=np.zeros(hidden_dim),
        W_out=rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, config.num_digits)),
        b_out=np.zeros(config.num_digits),
    )


def clone_factorized_add_model(model: FactorizedAddMLP) -> FactorizedAddMLP:
    return FactorizedAddMLP(
        W_router_ADD01=model.W_router_ADD01.copy(),
        b_router_ADD01=model.b_router_ADD01.copy(),
        W_router_ADD12=model.W_router_ADD12.copy(),
        b_router_ADD12=model.b_router_ADD12.copy(),
        W_op=model.W_op.copy(),
        b_op=model.b_op.copy(),
        W_out=model.W_out.copy(),
        b_out=model.b_out.copy(),
    )


def validate_multi_route_specs(specs: tuple[MultiRouteOpSpec, ...], sequence_digits: int) -> None:
    if not specs:
        raise ValueError("multi-route specs must not be empty.")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"multi-route spec names must be unique, got {names}.")
    for spec in specs:
        if not spec.name:
            raise ValueError("multi-route spec name must not be empty.")
        if spec.kind not in ("add", "copy", "max"):
            raise ValueError(f"unsupported multi-route op kind {spec.kind!r} for {spec.name}.")
        if not spec.operands:
            raise ValueError(f"multi-route op {spec.name} has no operands.")
        invalid_positions = [
            position for position in spec.operands if position < 0 or position >= sequence_digits
        ]
        if invalid_positions:
            raise ValueError(
                f"multi-route op {spec.name} has invalid operands {invalid_positions}; "
                f"sequence_digits={sequence_digits}."
            )
        if spec.kind == "copy" and len(spec.operands) != 1:
            raise ValueError(f"copy op {spec.name} must have exactly one operand.")
        if spec.kind in ("add", "max") and len(spec.operands) < 2:
            raise ValueError(f"{spec.kind} op {spec.name} must have at least two operands.")


def multi_route_digit_features(digits: tuple[int, ...], num_digits: int) -> np.ndarray:
    rows = np.zeros(len(digits) * num_digits)
    for position, digit in enumerate(digits):
        if digit < 0 or digit >= num_digits:
            raise ValueError(f"digit {digit} is outside num_digits={num_digits}.")
        rows[position * num_digits + digit] = 1.0
    return rows


def multi_route_target(
    spec: MultiRouteOpSpec,
    digits: tuple[int, ...],
    num_digits: int,
) -> int:
    values = tuple(digits[position] for position in spec.operands)
    if spec.kind == "add":
        return int(sum(values) % num_digits)
    if spec.kind == "copy":
        return int(values[0])
    if spec.kind == "max":
        return int(max(values))
    raise ValueError(f"unsupported multi-route op kind {spec.kind!r}.")


def make_multi_route_dataset(
    spec: MultiRouteOpSpec,
    num_digits: int,
    sequence_digits: int,
) -> MultiRouteDataset:
    validate_multi_route_specs((spec,), sequence_digits)
    digit_rows: list[np.ndarray] = []
    targets: list[int] = []
    for digits_raw in np.ndindex(*(num_digits for _ in range(sequence_digits))):
        digits = tuple(int(digit) for digit in digits_raw)
        digit_rows.append(multi_route_digit_features(digits, num_digits))
        targets.append(multi_route_target(spec, digits, num_digits))
    return MultiRouteDataset(
        digit_x=np.array(digit_rows),
        y=np.array(targets, dtype=int),
        op_spec=spec,
    )


def decode_multi_route_digits(
    digit_x: np.ndarray,
    num_digits: int,
    sequence_digits: int,
) -> tuple[int, ...]:
    if digit_x.shape != (num_digits * sequence_digits,):
        raise ValueError(
            f"digit_x shape {digit_x.shape} does not match "
            f"{sequence_digits} positions x {num_digits} digits."
        )
    digits: list[int] = []
    for position in range(sequence_digits):
        start = position * num_digits
        end = start + num_digits
        slice_values = digit_x[start:end]
        active = np.flatnonzero(slice_values == 1.0)
        if len(active) != 1:
            raise ValueError(
                f"position {position} is not one-hot; active indices={active.tolist()}."
            )
        if not np.all((slice_values == 0.0) | (slice_values == 1.0)):
            raise ValueError(f"position {position} contains non-binary values.")
        digits.append(int(active[0]))
    return tuple(digits)


def analogous_multi_route_dataset(
    source_spec: MultiRouteOpSpec,
    target_dataset: MultiRouteDataset,
    num_digits: int,
    sequence_digits: int,
) -> MultiRouteDataset:
    target_spec = target_dataset.op_spec
    if source_spec.kind != target_spec.kind:
        raise ValueError(
            f"cannot build analog dataset across kinds: {source_spec.kind!r} vs {target_spec.kind!r}."
        )
    if len(source_spec.operands) != len(target_spec.operands):
        raise ValueError(
            f"analog specs must have same operand count, got "
            f"{source_spec.operands} vs {target_spec.operands}."
        )

    rows: list[np.ndarray] = []
    targets: list[int] = []
    source_operand_positions = set(source_spec.operands)
    for digit_row, expected_target in zip(target_dataset.digit_x, target_dataset.y, strict=True):
        target_digits = decode_multi_route_digits(digit_row, num_digits, sequence_digits)
        analog_digits = list(target_digits)
        for source_position, target_position in zip(
            source_spec.operands,
            target_spec.operands,
            strict=True,
        ):
            analog_digits[source_position] = target_digits[target_position]
        for position in range(sequence_digits):
            if position not in source_operand_positions:
                analog_digits[position] = target_digits[position]
        analog_tuple = tuple(analog_digits)
        target = multi_route_target(source_spec, analog_tuple, num_digits)
        if target != int(expected_target):
            raise ValueError(
                f"analog target {target} does not match target op value {int(expected_target)} "
                f"for source={source_spec.name}, target={target_spec.name}."
            )
        rows.append(multi_route_digit_features(analog_tuple, num_digits))
        targets.append(target)
    return MultiRouteDataset(
        digit_x=np.array(rows),
        y=np.array(targets, dtype=int),
        op_spec=source_spec,
    )


def make_multi_route_factorized_model(
    specs: tuple[MultiRouteOpSpec, ...],
    num_digits: int,
    sequence_digits: int,
    hidden_dim: int,
    seed: int,
) -> MultiRouteFactorizedMLP:
    validate_multi_route_specs(specs, sequence_digits)
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
    rng = np.random.default_rng(seed)
    digit_input_dim = num_digits * sequence_digits
    return MultiRouteFactorizedMLP(
        W_router={
            spec.name: rng.normal(
                0.0,
                1.0 / np.sqrt(digit_input_dim),
                (digit_input_dim, hidden_dim),
            )
            for spec in specs
        },
        b_router={spec.name: np.zeros(hidden_dim) for spec in specs},
        W_op=rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, hidden_dim)),
        b_op=np.zeros(hidden_dim),
        W_out=rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, num_digits)),
        b_out=np.zeros(num_digits),
    )


def clone_multi_route_factorized_model(
    model: MultiRouteFactorizedMLP,
) -> MultiRouteFactorizedMLP:
    return MultiRouteFactorizedMLP(
        W_router={name: value.copy() for name, value in model.W_router.items()},
        b_router={name: value.copy() for name, value in model.b_router.items()},
        W_op=model.W_op.copy(),
        b_op=model.b_op.copy(),
        W_out=model.W_out.copy(),
        b_out=model.b_out.copy(),
    )


def reinitialize_hidden_neurons(model: TinyMLP, neuron_mask: np.ndarray, seed: int) -> None:
    input_dim, hidden_dim = model.W1.shape
    if neuron_mask.shape != (hidden_dim,):
        raise ValueError(
            f"neuron_mask shape {neuron_mask.shape} does not match hidden shape {(hidden_dim,)}."
        )
    if not np.any(neuron_mask):
        raise ValueError("neuron_mask contains no neurons to reinitialize.")

    output_dim = model.W2.shape[1]
    neuron_count = int(np.sum(neuron_mask))
    rng = np.random.default_rng(seed)
    model.W1[:, neuron_mask] = rng.normal(
        0.0,
        1.0 / np.sqrt(input_dim),
        (input_dim, neuron_count),
    )
    model.b1[neuron_mask] = 0.0
    model.W2[neuron_mask, :] = rng.normal(
        0.0,
        1.0 / np.sqrt(hidden_dim),
        (neuron_count, output_dim),
    )


def target_for_op(op_name: str, digits: tuple[int, int, int], config: OpsConfig) -> int:
    d0, d1, d2 = digits
    if op_name == "COPY0":
        return d0
    if op_name == "COPY1":
        return d1
    if op_name == "COPY2":
        return d2
    if op_name == "ADD01":
        return (d0 + d1) % config.num_digits
    if op_name == "ADD12":
        return (d1 + d2) % config.num_digits
    if op_name == "MAX":
        return max(d0, d1, d2)
    raise ValueError(f"unknown op_name={op_name!r}")


def encode_example(op_name: str, digits: tuple[int, int, int], config: OpsConfig) -> np.ndarray:
    x = np.zeros(config.input_dim)
    op_index = config.op_names.index(op_name)
    x[op_index] = 1.0
    offset = len(config.op_names)
    for position, digit in enumerate(digits):
        x[offset + position * config.num_digits + digit] = 1.0
    if config.use_position_flags and op_name in ("ADD01", "ADD12"):
        flag_offset = len(config.op_names) + config.sequence_digits * config.num_digits
        pair = (0, 1) if op_name == "ADD01" else (1, 2)
        pair_index = config.add_position_pairs.index(pair)
        x[flag_offset + pair_index] = 1.0
    return x


def make_dataset(config: OpsConfig, ops: Iterable[str]) -> OpsDataset:
    rows: list[np.ndarray] = []
    targets: list[int] = []
    op_ids: list[int] = []
    op_names = tuple(ops)
    for op_name in op_names:
        for digits_raw in np.ndindex(config.num_digits, config.num_digits, config.num_digits):
            digits = tuple(int(digit) for digit in digits_raw)
            rows.append(encode_example(op_name, digits, config))
            targets.append(target_for_op(op_name, digits, config))
            op_ids.append(config.op_names.index(op_name))
    return OpsDataset(
        x=np.array(rows),
        y=np.array(targets, dtype=int),
        op_ids=np.array(op_ids, dtype=int),
        op_names=op_names,
    )


def digit_features_from_dataset(dataset: OpsDataset, config: OpsConfig) -> np.ndarray:
    offset = len(config.op_names)
    end = offset + config.sequence_digits * config.num_digits
    if dataset.x.ndim != 2 or dataset.x.shape[1] != config.input_dim:
        raise ValueError(
            f"dataset x shape {dataset.x.shape} does not match input_dim={config.input_dim}."
        )
    digit_features = dataset.x[:, offset:end]
    expected_active = float(config.sequence_digits)
    active_counts = np.sum(digit_features, axis=1)
    if not np.all(active_counts == expected_active):
        raise ValueError(
            "digit feature rows are not valid one-hot position encodings; "
            f"active count range=({float(np.min(active_counts))}, {float(np.max(active_counts))})."
        )
    return digit_features


def decode_digits_from_example(x: np.ndarray, config: OpsConfig) -> tuple[int, int, int]:
    if x.shape != (config.input_dim,):
        raise ValueError(f"example shape {x.shape} does not match input_dim={config.input_dim}.")
    offset = len(config.op_names)
    digits: list[int] = []
    for position in range(config.sequence_digits):
        start = offset + position * config.num_digits
        end = start + config.num_digits
        slice_values = x[start:end]
        active = np.flatnonzero(slice_values == 1.0)
        if len(active) != 1:
            raise ValueError(
                f"position {position} is not one-hot; active indices={active.tolist()}."
            )
        if not np.all((slice_values == 0.0) | (slice_values == 1.0)):
            raise ValueError(f"position {position} contains non-binary values.")
        digits.append(int(active[0]))
    return (digits[0], digits[1], digits[2])


def analogous_add01_dataset_for_add12(config: OpsConfig, add12_dataset: OpsDataset) -> OpsDataset:
    if add12_dataset.op_names != ("ADD12",):
        raise ValueError(f"expected ADD12-only dataset, got {add12_dataset.op_names}.")

    rows: list[np.ndarray] = []
    targets: list[int] = []
    op_id = config.op_names.index("ADD01")
    for x, expected_target in zip(add12_dataset.x, add12_dataset.y, strict=True):
        d0, d1, d2 = decode_digits_from_example(x, config)
        analog_digits = (d1, d2, d0)
        target = target_for_op("ADD01", analog_digits, config)
        if target != int(expected_target):
            raise ValueError(
                f"analog target {target} does not match ADD12 target {int(expected_target)}."
            )
        rows.append(encode_example("ADD01", analog_digits, config))
        targets.append(target)

    return OpsDataset(
        x=np.array(rows),
        y=np.array(targets, dtype=int),
        op_ids=np.full(len(rows), op_id, dtype=int),
        op_names=("ADD01",),
    )


def forward(model: TinyMLP, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = x @ model.W1 + model.b1
    h = np.maximum(z1, 0.0)
    logits = h @ model.W2 + model.b2
    return logits, {"x": x, "z1": z1, "h": h, "logits": logits}


def factorized_router_params(
    model: FactorizedAddMLP,
    op_name: str,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    if op_name == "ADD01":
        return (
            "W_router_ADD01",
            "b_router_ADD01",
            model.W_router_ADD01,
            model.b_router_ADD01,
        )
    if op_name == "ADD12":
        return (
            "W_router_ADD12",
            "b_router_ADD12",
            model.W_router_ADD12,
            model.b_router_ADD12,
        )
    raise ValueError(f"factorized add model only supports {FACTORIZED_ADD_OPS}, got {op_name!r}.")


def single_op_name(dataset: OpsDataset, allowed_ops: tuple[str, ...]) -> str:
    if len(dataset.op_names) != 1:
        raise ValueError(f"expected single-op dataset, got {dataset.op_names}.")
    op_name = dataset.op_names[0]
    if op_name not in allowed_ops:
        raise ValueError(f"expected op in {allowed_ops}, got {op_name!r}.")
    return op_name


def factorized_forward(
    model: FactorizedAddMLP,
    digit_x: np.ndarray,
    op_name: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _, _, W_router, b_router = factorized_router_params(model, op_name)
    route_z = digit_x @ W_router + b_router
    route_h = np.maximum(route_z, 0.0)
    op_z = route_h @ model.W_op + model.b_op
    op_h = np.maximum(op_z, 0.0)
    logits = op_h @ model.W_out + model.b_out
    return logits, {
        "digit_x": digit_x,
        "route_z": route_z,
        "route_h": route_h,
        "op_z": op_z,
        "op_h": op_h,
        "logits": logits,
        "op_name": np.array(op_name),
    }


def loss_and_grad_logits(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    dlogits = probs
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= len(y)
    return float(loss), dlogits


def backward(model: TinyMLP, cache: dict[str, np.ndarray], dlogits: np.ndarray) -> dict[str, np.ndarray]:
    h = cache["h"]
    z1 = cache["z1"]
    x = cache["x"]
    dW2 = h.T @ dlogits
    db2 = np.sum(dlogits, axis=0)
    dh = dlogits @ model.W2.T
    dz1 = dh * (z1 > 0.0)
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def factorized_backward(
    model: FactorizedAddMLP,
    cache: dict[str, np.ndarray],
    dlogits: np.ndarray,
) -> dict[str, np.ndarray]:
    op_name = str(cache["op_name"])
    W_router_name, b_router_name, _, _ = factorized_router_params(model, op_name)
    op_h = cache["op_h"]
    op_z = cache["op_z"]
    route_h = cache["route_h"]
    route_z = cache["route_z"]
    digit_x = cache["digit_x"]

    grads = {
        "W_router_ADD01": np.zeros_like(model.W_router_ADD01),
        "b_router_ADD01": np.zeros_like(model.b_router_ADD01),
        "W_router_ADD12": np.zeros_like(model.W_router_ADD12),
        "b_router_ADD12": np.zeros_like(model.b_router_ADD12),
        "W_op": np.zeros_like(model.W_op),
        "b_op": np.zeros_like(model.b_op),
        "W_out": np.zeros_like(model.W_out),
        "b_out": np.zeros_like(model.b_out),
    }

    grads["W_out"] = op_h.T @ dlogits
    grads["b_out"] = np.sum(dlogits, axis=0)
    dop_h = dlogits @ model.W_out.T
    dop_z = dop_h * (op_z > 0.0)
    grads["W_op"] = route_h.T @ dop_z
    grads["b_op"] = np.sum(dop_z, axis=0)
    droute_h = dop_z @ model.W_op.T
    droute_z = droute_h * (route_z > 0.0)
    grads[W_router_name] = digit_x.T @ droute_z
    grads[b_router_name] = np.sum(droute_z, axis=0)
    return grads


def apply_update(
    model: TinyMLP,
    grads: dict[str, np.ndarray],
    learning_rate: float,
    usage: np.ndarray | None = None,
    protection_strength: float = 0.0,
    protected_mask: np.ndarray | None = None,
    allowed_mask: np.ndarray | None = None,
) -> None:
    if allowed_mask is not None:
        if allowed_mask.shape != model.b1.shape:
            raise ValueError(
                f"allowed_mask shape {allowed_mask.shape} does not match hidden shape {model.b1.shape}."
            )
        scale = allowed_mask.astype(float)
        if protected_mask is not None:
            if protected_mask.shape != model.b1.shape:
                raise ValueError(
                    f"protected_mask shape {protected_mask.shape} does not match hidden shape {model.b1.shape}."
                )
            scale[protected_mask] = 0.0
        model.W1 -= learning_rate * (grads["W1"] * scale[None, :])
        model.b1 -= learning_rate * (grads["b1"] * scale)
        model.W2 -= learning_rate * (grads["W2"] * scale[:, None])
        # b2 is not owned by a hidden neuron, so the surgical branch leaves it fixed.
        return

    if protected_mask is not None:
        if protected_mask.shape != model.b1.shape:
            raise ValueError(
                f"protected_mask shape {protected_mask.shape} does not match hidden shape {model.b1.shape}."
            )
        scale = np.ones(model.b1.shape[0])
        scale[protected_mask] = 0.0
        model.W1 -= learning_rate * (grads["W1"] * scale[None, :])
        model.b1 -= learning_rate * (grads["b1"] * scale)
        model.W2 -= learning_rate * (grads["W2"] * scale[:, None])
        model.b2 -= learning_rate * grads["b2"]
        return

    if usage is None or protection_strength == 0.0:
        model.W1 -= learning_rate * grads["W1"]
        model.b1 -= learning_rate * grads["b1"]
        model.W2 -= learning_rate * grads["W2"]
        model.b2 -= learning_rate * grads["b2"]
        return

    max_usage = float(np.max(usage))
    if max_usage <= 0.0:
        scale = np.ones_like(usage)
    else:
        normalized = usage / max_usage
        scale = 1.0 / (1.0 + protection_strength * normalized)

    model.W1 -= learning_rate * (grads["W1"] * scale[None, :])
    model.b1 -= learning_rate * (grads["b1"] * scale)
    model.W2 -= learning_rate * (grads["W2"] * scale[:, None])
    model.b2 -= learning_rate * grads["b2"]


def apply_split_hidden_update(
    model: TinyMLP,
    grads: dict[str, np.ndarray],
    learning_rate: float,
    incoming_mask: np.ndarray,
    readout_mask: np.ndarray,
) -> None:
    hidden_shape = model.b1.shape
    if incoming_mask.shape != hidden_shape:
        raise ValueError(
            f"incoming_mask shape {incoming_mask.shape} does not match hidden shape {hidden_shape}."
        )
    if readout_mask.shape != hidden_shape:
        raise ValueError(
            f"readout_mask shape {readout_mask.shape} does not match hidden shape {hidden_shape}."
        )
    incoming_scale = incoming_mask.astype(float)
    readout_scale = readout_mask.astype(float)
    model.W1 -= learning_rate * (grads["W1"] * incoming_scale[None, :])
    model.b1 -= learning_rate * (grads["b1"] * incoming_scale)
    model.W2 -= learning_rate * (grads["W2"] * readout_scale[:, None])


def normalize_nonnegative_score(score: np.ndarray, name: str) -> np.ndarray:
    if score.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {score.shape}.")
    if not np.all(np.isfinite(score)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(score < 0.0):
        raise ValueError(f"{name} contains negative values.")
    max_score = float(np.max(score))
    if max_score <= 0.0:
        raise ValueError(f"{name} has no positive signal.")
    return score / max_score


def blend_scale(old_importance: np.ndarray, new_need: np.ndarray, lambda_old: float) -> np.ndarray:
    if lambda_old <= 0.0:
        raise ValueError(f"lambda_old must be positive, got {lambda_old}.")
    if old_importance.shape != new_need.shape:
        raise ValueError(
            f"old_importance shape {old_importance.shape} does not match new_need shape {new_need.shape}."
        )
    old_norm = normalize_nonnegative_score(old_importance, "old_importance")
    new_norm = normalize_nonnegative_score(new_need, "new_need")
    denom = new_norm + lambda_old * old_norm
    scale = np.zeros_like(new_norm)
    active = denom > SCORE_EPSILON
    scale[active] = new_norm[active] / denom[active]
    return scale


def apply_blended_hidden_update(
    model: TinyMLP,
    grads: dict[str, np.ndarray],
    learning_rate: float,
    scale: np.ndarray,
) -> None:
    if scale.shape != model.b1.shape:
        raise ValueError(f"scale shape {scale.shape} does not match hidden shape {model.b1.shape}.")
    if np.any(scale < 0.0) or np.any(scale > 1.0):
        raise ValueError("scale must stay within [0, 1].")
    model.W1 -= learning_rate * (grads["W1"] * scale[None, :])
    model.b1 -= learning_rate * (grads["b1"] * scale)
    model.W2 -= learning_rate * (grads["W2"] * scale[:, None])


def factorized_apply_update(
    model: FactorizedAddMLP,
    grads: dict[str, np.ndarray],
    learning_rate: float,
    trainable_keys: tuple[str, ...],
) -> None:
    unknown = set(trainable_keys) - set(FACTORIZED_PARAM_KEYS)
    if unknown:
        raise ValueError(f"unknown trainable factorized parameter keys: {sorted(unknown)}.")
    for key in trainable_keys:
        value = getattr(model, key)
        grad = grads[key]
        if value.shape != grad.shape:
            raise ValueError(f"gradient shape for {key} {grad.shape} does not match {value.shape}.")
        setattr(model, key, value - learning_rate * grad)


def evaluate(model: TinyMLP, dataset: OpsDataset) -> tuple[float, float]:
    logits, _ = forward(model, dataset.x)
    loss, _ = loss_and_grad_logits(logits, dataset.y)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == dataset.y))
    return loss, accuracy


def factorized_evaluate(
    model: FactorizedAddMLP,
    dataset: OpsDataset,
    config: OpsConfig,
) -> tuple[float, float]:
    op_name = single_op_name(dataset, FACTORIZED_ADD_OPS)
    digit_x = digit_features_from_dataset(dataset, config)
    logits, _ = factorized_forward(model, digit_x, op_name)
    loss, _ = loss_and_grad_logits(logits, dataset.y)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == dataset.y))
    return loss, accuracy


def evaluate_by_op(model: TinyMLP, dataset: OpsDataset, config: OpsConfig) -> dict[str, float]:
    logits, _ = forward(model, dataset.x)
    predictions = np.argmax(logits, axis=1)
    result: dict[str, float] = {}
    for op_name in dataset.op_names:
        op_id = config.op_names.index(op_name)
        mask = dataset.op_ids == op_id
        result[op_name] = float(np.mean(predictions[mask] == dataset.y[mask]))
    return result


def hidden_activations(model: TinyMLP, dataset: OpsDataset) -> np.ndarray:
    _, cache = forward(model, dataset.x)
    return cache["h"]


def factorized_op_activations(
    model: FactorizedAddMLP,
    dataset: OpsDataset,
    config: OpsConfig,
) -> np.ndarray:
    op_name = single_op_name(dataset, FACTORIZED_ADD_OPS)
    digit_x = digit_features_from_dataset(dataset, config)
    _, cache = factorized_forward(model, digit_x, op_name)
    return cache["op_h"]


def centered_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"CKA needs the same number of rows, got {left.shape[0]} and {right.shape[0]}."
        )
    left_centered = left - np.mean(left, axis=0, keepdims=True)
    right_centered = right - np.mean(right, axis=0, keepdims=True)
    numerator = float(np.sum((left_centered.T @ right_centered) ** 2))
    left_norm = float(np.sum((left_centered.T @ left_centered) ** 2))
    right_norm = float(np.sum((right_centered.T @ right_centered) ** 2))
    denom = np.sqrt(left_norm * right_norm)
    if denom <= 0.0:
        raise ValueError("CKA denominator is zero; one representation has no variance.")
    return numerator / denom


def mean_row_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"row cosine arrays must have matching shapes, got {left.shape} and {right.shape}.")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = left_norm * right_norm
    valid = denom > SCORE_EPSILON
    if not np.any(valid):
        raise ValueError("all paired rows have zero norm; row cosine is undefined.")
    return float(np.mean(np.sum(left[valid] * right[valid], axis=1) / denom[valid]))


def class_center_cosine(left: np.ndarray, right: np.ndarray, labels: np.ndarray, class_count: int) -> float:
    if left.shape != right.shape:
        raise ValueError(
            f"class-center cosine arrays must have matching shapes, got {left.shape} and {right.shape}."
        )
    if labels.shape != (left.shape[0],):
        raise ValueError(f"labels shape {labels.shape} does not match row count {left.shape[0]}.")

    cosines: list[float] = []
    for class_id in range(class_count):
        mask = labels == class_id
        if not np.any(mask):
            raise ValueError(f"class {class_id} has no examples.")
        left_center = np.mean(left[mask], axis=0)
        right_center = np.mean(right[mask], axis=0)
        cosines.append(cosine(left_center, right_center))
    return float(np.mean(cosines))


def factorized_pair_alignment_metrics(
    model: FactorizedAddMLP,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
) -> dict[str, float]:
    if add12_dataset.op_names != ("ADD12",):
        raise ValueError(f"expected ADD12 dataset, got {add12_dataset.op_names}.")
    if analog_add01_dataset.op_names != ("ADD01",):
        raise ValueError(f"expected analog ADD01 dataset, got {analog_add01_dataset.op_names}.")
    if not np.array_equal(add12_dataset.y, analog_add01_dataset.y):
        raise ValueError("ADD12 and analogous ADD01 targets must match row-by-row.")

    add12_digits = digit_features_from_dataset(add12_dataset, config)
    analog_digits = digit_features_from_dataset(analog_add01_dataset, config)
    _, add12_cache = factorized_forward(model, add12_digits, "ADD12")
    _, analog_cache = factorized_forward(model, analog_digits, "ADD01")
    route_left = add12_cache["route_h"]
    route_right = analog_cache["route_h"]
    op_left = add12_cache["op_h"]
    op_right = analog_cache["op_h"]
    return {
        "route_pair_mse": float(np.mean((route_left - route_right) ** 2)),
        "op_pair_mse": float(np.mean((op_left - op_right) ** 2)),
        "route_pair_cosine": mean_row_cosine(route_left, route_right),
        "op_pair_cosine": mean_row_cosine(op_left, op_right),
        "route_class_cosine": class_center_cosine(
            route_left,
            route_right,
            add12_dataset.y,
            config.num_digits,
        ),
        "op_class_cosine": class_center_cosine(
            op_left,
            op_right,
            add12_dataset.y,
            config.num_digits,
        ),
        "paired_op_cka": centered_linear_cka(op_left, op_right),
    }


def activation_subspace_basis(
    model: TinyMLP,
    dataset: OpsDataset,
    retained_energy: float,
) -> SubspaceBasis:
    if not 0.0 < retained_energy <= 1.0:
        raise ValueError(f"retained_energy must be in (0, 1], got {retained_energy}.")
    h = hidden_activations(model, dataset)
    centered_h = h - np.mean(h, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered_h, full_matrices=False)
    squared = singular_values**2
    total_energy = float(np.sum(squared))
    if total_energy <= 0.0:
        raise ValueError("activation matrix has zero variance; subspace is undefined.")

    cumulative_energy = np.cumsum(squared) / total_energy
    rank = int(np.searchsorted(cumulative_energy, retained_energy, side="left") + 1)
    basis = vh[:rank].T
    return SubspaceBasis(
        basis=basis,
        rank=rank,
        retained_energy=float(cumulative_energy[rank - 1]),
    )


def principal_angle_summary(left: SubspaceBasis, right: SubspaceBasis) -> dict[str, float]:
    singular_values = np.linalg.svd(left.basis.T @ right.basis, compute_uv=False)
    if np.any(singular_values > 1.0 + 1e-8):
        raise ValueError(f"principal-angle singular values exceed 1: {singular_values}.")
    singular_values = np.clip(singular_values, -1.0, 1.0)
    angles = np.arccos(singular_values)
    return {
        "left_rank": float(left.rank),
        "right_rank": float(right.rank),
        "left_energy": left.retained_energy,
        "right_energy": right.retained_energy,
        "max_cosine": float(np.max(singular_values)),
        "mean_cosine": float(np.mean(singular_values)),
        "min_angle_deg": float(np.degrees(np.min(angles))),
        "mean_angle_deg": float(np.degrees(np.mean(angles))),
    }


def activation_subspace_metrics(
    model: TinyMLP,
    config: OpsConfig,
    retained_energy: float,
) -> dict[str, dict[str, float]]:
    bases = {
        op_name: activation_subspace_basis(
            model,
            make_dataset(config, (op_name,)),
            retained_energy,
        )
        for op_name in config.base_ops + (config.new_op,)
    }
    return {
        f"{op_name}_vs_{config.new_op}": principal_angle_summary(
            bases[op_name],
            bases[config.new_op],
        )
        for op_name in config.base_ops
    }


def print_activation_subspace_table(metrics: dict[str, dict[str, float]]) -> None:
    print(f"\nActivation subspace overlap before new-task training, energy={SUBSPACE_ENERGY:.2f}")
    print("pair             rank_old  rank_new  max_cos  mean_cos  min_angle  mean_angle")
    for pair_name, values in metrics.items():
        print(
            f"{pair_name:<16} "
            f"{int(values['left_rank']):>8} "
            f"{int(values['right_rank']):>9} "
            f"{values['max_cosine']:>8.3f} "
            f"{values['mean_cosine']:>9.3f} "
            f"{values['min_angle_deg']:>9.2f} "
            f"{values['mean_angle_deg']:>11.2f}"
        )


def train(
    model: TinyMLP,
    dataset: OpsDataset,
    train_config: TrainConfig,
    label: str,
    usage: np.ndarray | None = None,
    protected_mask: np.ndarray | None = None,
    allowed_mask: np.ndarray | None = None,
) -> None:
    for epoch in range(1, train_config.epochs + 1):
        logits, cache = forward(model, dataset.x)
        loss, dlogits = loss_and_grad_logits(logits, dataset.y)
        grads = backward(model, cache, dlogits)
        apply_update(
            model,
            grads,
            train_config.learning_rate,
            usage=usage,
            protection_strength=train_config.protection_strength,
            protected_mask=protected_mask,
            allowed_mask=allowed_mask,
        )
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = evaluate(model, dataset)
            print(f"{label} epoch={epoch:04d} loss={loss:.4f} accuracy={accuracy:.3f}")


def train_split_hidden_update(
    model: TinyMLP,
    dataset: OpsDataset,
    train_config: TrainConfig,
    label: str,
    incoming_mask: np.ndarray,
    readout_mask: np.ndarray,
) -> None:
    if not np.any(incoming_mask) and not np.any(readout_mask):
        raise ValueError("incoming_mask and readout_mask are both empty.")
    for epoch in range(1, train_config.epochs + 1):
        logits, cache = forward(model, dataset.x)
        loss, dlogits = loss_and_grad_logits(logits, dataset.y)
        grads = backward(model, cache, dlogits)
        apply_split_hidden_update(
            model,
            grads,
            train_config.learning_rate,
            incoming_mask,
            readout_mask,
        )
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = evaluate(model, dataset)
            print(f"{label} epoch={epoch:04d} loss={loss:.4f} accuracy={accuracy:.3f}")


def train_blended_hidden_update(
    model: TinyMLP,
    dataset: OpsDataset,
    train_config: TrainConfig,
    label: str,
    scale: np.ndarray,
) -> None:
    if not np.any(scale > 0.0):
        raise ValueError("blend scale is all zero.")
    for epoch in range(1, train_config.epochs + 1):
        logits, cache = forward(model, dataset.x)
        loss, dlogits = loss_and_grad_logits(logits, dataset.y)
        grads = backward(model, cache, dlogits)
        apply_blended_hidden_update(model, grads, train_config.learning_rate, scale)
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = evaluate(model, dataset)
            print(f"{label} epoch={epoch:04d} loss={loss:.4f} accuracy={accuracy:.3f}")


def train_factorized_add(
    model: FactorizedAddMLP,
    dataset: OpsDataset,
    config: OpsConfig,
    train_config: TrainConfig,
    label: str,
    trainable_keys: tuple[str, ...],
) -> None:
    if not trainable_keys:
        raise ValueError("trainable_keys is empty.")
    op_name = single_op_name(dataset, FACTORIZED_ADD_OPS)
    digit_x = digit_features_from_dataset(dataset, config)
    for epoch in range(1, train_config.epochs + 1):
        logits, cache = factorized_forward(model, digit_x, op_name)
        loss, dlogits = loss_and_grad_logits(logits, dataset.y)
        grads = factorized_backward(model, cache, dlogits)
        factorized_apply_update(model, grads, train_config.learning_rate, trainable_keys)
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = factorized_evaluate(model, dataset, config)
            print(f"{label} epoch={epoch:04d} loss={loss:.4f} accuracy={accuracy:.3f}")


def factorized_alignment_loss_and_grads(
    model: FactorizedAddMLP,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
    alignment_target: str,
    alignment_weight: float,
) -> tuple[float, float, float, dict[str, np.ndarray]]:
    if alignment_target not in FACTORIZED_ALIGNMENT_TARGETS:
        raise ValueError(
            f"alignment_target must be one of {FACTORIZED_ALIGNMENT_TARGETS}, got {alignment_target!r}."
        )
    if alignment_weight <= 0.0:
        raise ValueError(f"alignment_weight must be positive, got {alignment_weight}.")
    if add12_dataset.op_names != ("ADD12",):
        raise ValueError(f"expected ADD12 dataset, got {add12_dataset.op_names}.")
    if analog_add01_dataset.op_names != ("ADD01",):
        raise ValueError(f"expected analog ADD01 dataset, got {analog_add01_dataset.op_names}.")
    if not np.array_equal(add12_dataset.y, analog_add01_dataset.y):
        raise ValueError("ADD12 and analogous ADD01 targets must match row-by-row.")

    add12_digits = digit_features_from_dataset(add12_dataset, config)
    analog_digits = digit_features_from_dataset(analog_add01_dataset, config)
    logits, add12_cache = factorized_forward(model, add12_digits, "ADD12")
    _, analog_cache = factorized_forward(model, analog_digits, "ADD01")
    ce_loss, dlogits = loss_and_grad_logits(logits, add12_dataset.y)
    grads = factorized_backward(model, add12_cache, dlogits)

    if alignment_target == "route":
        source = add12_cache["route_h"]
        target = analog_cache["route_h"]
        alignment_loss = float(np.mean((source - target) ** 2))
        droute_h = alignment_weight * 2.0 * (source - target) / source.size
    elif alignment_target == "op":
        source = add12_cache["op_h"]
        target = analog_cache["op_h"]
        alignment_loss = float(np.mean((source - target) ** 2))
        dop_h = alignment_weight * 2.0 * (source - target) / source.size
        dop_z = dop_h * (add12_cache["op_z"] > 0.0)
        droute_h = dop_z @ model.W_op.T
    else:
        raise ValueError(f"unknown alignment_target={alignment_target!r}.")

    droute_z = droute_h * (add12_cache["route_z"] > 0.0)
    grads["W_router_ADD12"] += add12_digits.T @ droute_z
    grads["b_router_ADD12"] += np.sum(droute_z, axis=0)
    total_loss = ce_loss + alignment_weight * alignment_loss
    return total_loss, ce_loss, alignment_loss, grads


def train_factorized_add_alignment(
    model: FactorizedAddMLP,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
    train_config: TrainConfig,
    label: str,
    alignment_target: str,
    alignment_weight: float,
) -> None:
    for epoch in range(1, train_config.epochs + 1):
        loss, ce_loss, alignment_loss, grads = factorized_alignment_loss_and_grads(
            model,
            add12_dataset,
            analog_add01_dataset,
            config,
            alignment_target,
            alignment_weight,
        )
        factorized_apply_update(
            model,
            grads,
            train_config.learning_rate,
            ("W_router_ADD12", "b_router_ADD12"),
        )
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = factorized_evaluate(model, add12_dataset, config)
            print(
                f"{label} epoch={epoch:04d} loss={loss:.4f} "
                f"ce={ce_loss:.4f} align={alignment_loss:.4f} accuracy={accuracy:.3f}"
            )


def factorized_op_alignment_shared_loss_and_grads(
    model: FactorizedAddMLP,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
) -> tuple[float, dict[str, np.ndarray]]:
    if add12_dataset.op_names != ("ADD12",):
        raise ValueError(f"expected ADD12 dataset, got {add12_dataset.op_names}.")
    if analog_add01_dataset.op_names != ("ADD01",):
        raise ValueError(f"expected analog ADD01 dataset, got {analog_add01_dataset.op_names}.")
    if not np.array_equal(add12_dataset.y, analog_add01_dataset.y):
        raise ValueError("ADD12 and analogous ADD01 targets must match row-by-row.")

    add12_digits = digit_features_from_dataset(add12_dataset, config)
    analog_digits = digit_features_from_dataset(analog_add01_dataset, config)
    _, add12_cache = factorized_forward(model, add12_digits, "ADD12")
    _, analog_cache = factorized_forward(model, analog_digits, "ADD01")
    diff = add12_cache["op_h"] - analog_cache["op_h"]
    alignment_loss = float(np.mean(diff**2))
    dop_h_add12 = 2.0 * diff / diff.size
    dop_h_analog = -2.0 * diff / diff.size
    dop_z_add12 = dop_h_add12 * (add12_cache["op_z"] > 0.0)
    dop_z_analog = dop_h_analog * (analog_cache["op_z"] > 0.0)

    grads = empty_factorized_grads(model)
    grads["W_op"] = (
        add12_cache["route_h"].T @ dop_z_add12
        + analog_cache["route_h"].T @ dop_z_analog
    )
    grads["b_op"] = np.sum(dop_z_add12, axis=0) + np.sum(dop_z_analog, axis=0)
    return alignment_loss, grads


def factorized_consolidation_objective(
    model: FactorizedAddMLP,
    add01_dataset: OpsDataset,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
    objective_name: str,
) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
    if objective_name not in FACTORIZED_CONSOLIDATION_OBJECTIVES:
        raise ValueError(
            f"objective_name must be one of {FACTORIZED_CONSOLIDATION_OBJECTIVES}, got {objective_name!r}."
        )

    old_loss, old_grads = factorized_loss_and_grads(model, add01_dataset, config)
    new_loss, new_grads = factorized_loss_and_grads(model, add12_dataset, config)
    grads = empty_factorized_grads(model)
    align_loss = 0.0

    if objective_name == "new_ce":
        add_scaled_factorized_grads(grads, new_grads, 1.0)
        objective_loss = new_loss
    elif objective_name == "balanced_ce":
        add_scaled_factorized_grads(grads, old_grads, 0.5)
        add_scaled_factorized_grads(grads, new_grads, 0.5)
        objective_loss = 0.5 * (old_loss + new_loss)
    elif objective_name == "balanced_ce_align":
        align_loss, align_grads = factorized_op_alignment_shared_loss_and_grads(
            model,
            add12_dataset,
            analog_add01_dataset,
            config,
        )
        add_scaled_factorized_grads(grads, old_grads, 0.5)
        add_scaled_factorized_grads(grads, new_grads, 0.5)
        add_scaled_factorized_grads(
            grads,
            align_grads,
            FACTORIZED_CONSOLIDATION_ALIGNMENT_WEIGHT,
        )
        objective_loss = (
            0.5 * (old_loss + new_loss)
            + FACTORIZED_CONSOLIDATION_ALIGNMENT_WEIGHT * align_loss
        )
    else:
        raise ValueError(f"unknown objective_name={objective_name!r}.")

    return objective_loss, {
        "old_loss_component": old_loss,
        "new_loss_component": new_loss,
        "alignment_loss_component": align_loss,
    }, grads


def current_blend_scale(
    model: TinyMLP,
    old_dataset: OpsDataset,
    new_dataset: OpsDataset,
    old_score_name: str,
    lambda_old: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if old_score_name not in ("E", "AE"):
        raise ValueError(f"old_score_name must be 'E' or 'AE', got {old_score_name!r}.")

    old_scores = candidate_scores(model, old_dataset)
    new_gradient = gradient_salience(model, new_dataset)
    gradient_total = float(np.sum(new_gradient))
    if gradient_total <= 0.0:
        raise ValueError("new task gradient is zero; online blending cannot be measured.")

    scale = blend_scale(old_scores[old_score_name], new_gradient, lambda_old)
    return scale, {
        "mean_scale": float(np.mean(scale)),
        "max_scale": float(np.max(scale)),
        "effective_new_gradient_fraction": float(np.sum(new_gradient * scale) / gradient_total),
    }


def train_online_blended_hidden_update(
    model: TinyMLP,
    old_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
    label: str,
    old_score_name: str,
    lambda_old: float,
    recompute_every: int,
) -> list[dict[str, float]]:
    if recompute_every <= 0:
        raise ValueError(f"recompute_every must be positive, got {recompute_every}.")

    scale: np.ndarray | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, train_config.epochs + 1):
        if scale is None or (epoch - 1) % recompute_every == 0:
            scale, stats = current_blend_scale(
                model,
                old_dataset,
                new_dataset,
                old_score_name,
                lambda_old,
            )
            stats["epoch"] = float(epoch)
            history.append(stats)

        logits, cache = forward(model, new_dataset.x)
        loss, dlogits = loss_and_grad_logits(logits, new_dataset.y)
        grads = backward(model, cache, dlogits)
        apply_blended_hidden_update(model, grads, train_config.learning_rate, scale)
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = evaluate(model, new_dataset)
            print(f"{label} epoch={epoch:04d} loss={loss:.4f} accuracy={accuracy:.3f}")

    if not history:
        raise ValueError("online blending did not record any scale history.")
    return history


def causal_ablation_effects(model: TinyMLP, dataset: OpsDataset) -> np.ndarray:
    logits, cache = forward(model, dataset.x)
    base_loss, _ = loss_and_grad_logits(logits, dataset.y)
    effects = np.zeros(model.b1.shape[0])
    h = cache["h"]
    for neuron in range(model.b1.shape[0]):
        h_ablated = h.copy()
        h_ablated[:, neuron] = 0.0
        ablated_logits = h_ablated @ model.W2 + model.b2
        ablated_loss, _ = loss_and_grad_logits(ablated_logits, dataset.y)
        effects[neuron] = max(0.0, ablated_loss - base_loss)
    return effects


def usage_scores(model: TinyMLP, dataset: OpsDataset) -> dict[str, np.ndarray]:
    _, cache = forward(model, dataset.x)
    activation_strength = np.mean(np.abs(cache["h"]), axis=0)
    downstream_influence = np.linalg.norm(model.W2, axis=1)
    causal_effect = causal_ablation_effects(model, dataset)
    usage = activation_strength * downstream_influence * causal_effect
    return {
        "activation_strength": activation_strength,
        "downstream_influence": downstream_influence,
        "causal_effect": causal_effect,
        "usage": usage,
    }


def candidate_scores(model: TinyMLP, dataset: OpsDataset) -> dict[str, np.ndarray]:
    raw_scores = usage_scores(model, dataset)
    activation = raw_scores["activation_strength"]
    downstream = raw_scores["downstream_influence"]
    causal = raw_scores["causal_effect"]
    return {
        "A": activation,
        "D": downstream,
        "E": causal,
        "AE": activation * causal,
        "ADE": activation * downstream * causal,
    }


def gradient_salience(model: TinyMLP, dataset: OpsDataset) -> np.ndarray:
    logits, cache = forward(model, dataset.x)
    _, dlogits = loss_and_grad_logits(logits, dataset.y)
    grads = backward(model, cache, dlogits)
    incoming = np.linalg.norm(grads["W1"], axis=0)
    bias = np.abs(grads["b1"])
    outgoing = np.linalg.norm(grads["W2"], axis=1)
    return incoming + bias + outgoing


def factorized_gradients(
    model: FactorizedAddMLP,
    dataset: OpsDataset,
    config: OpsConfig,
) -> dict[str, np.ndarray]:
    _, grads = factorized_loss_and_grads(model, dataset, config)
    return grads


def factorized_loss_and_grads(
    model: FactorizedAddMLP,
    dataset: OpsDataset,
    config: OpsConfig,
) -> tuple[float, dict[str, np.ndarray]]:
    op_name = single_op_name(dataset, FACTORIZED_ADD_OPS)
    digit_x = digit_features_from_dataset(dataset, config)
    logits, cache = factorized_forward(model, digit_x, op_name)
    loss, dlogits = loss_and_grad_logits(logits, dataset.y)
    return loss, factorized_backward(model, cache, dlogits)


def empty_factorized_grads(model: FactorizedAddMLP) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(getattr(model, key)) for key in FACTORIZED_PARAM_KEYS}


def add_scaled_factorized_grads(
    target: dict[str, np.ndarray],
    source: dict[str, np.ndarray],
    scale: float,
) -> None:
    if set(target) != set(FACTORIZED_PARAM_KEYS):
        raise ValueError("target gradient dict does not contain exactly the factorized parameter keys.")
    if set(source) != set(FACTORIZED_PARAM_KEYS):
        raise ValueError("source gradient dict does not contain exactly the factorized parameter keys.")
    for key in FACTORIZED_PARAM_KEYS:
        if target[key].shape != source[key].shape:
            raise ValueError(
                f"gradient shape mismatch for {key}: {target[key].shape} vs {source[key].shape}."
            )
        target[key] += scale * source[key]


def gradient_squared_norm(grads: dict[str, np.ndarray], keys: tuple[str, ...]) -> float:
    total = 0.0
    for key in keys:
        if key not in grads:
            raise ValueError(f"missing gradient key {key!r}.")
        total += float(np.sum(grads[key] ** 2))
    return total


def factorized_gradient_pressure(
    model: FactorizedAddMLP,
    dataset: OpsDataset,
    config: OpsConfig,
) -> dict[str, float]:
    grads = factorized_gradients(model, dataset, config)
    total_squared = gradient_squared_norm(grads, FACTORIZED_PARAM_KEYS)
    shared_squared = gradient_squared_norm(grads, FACTORIZED_SHARED_KEYS)
    if total_squared <= 0.0:
        raise ValueError("factorized gradient has zero total squared norm.")
    return {
        "total_gradient_norm": float(np.sqrt(total_squared)),
        "shared_gradient_norm": float(np.sqrt(shared_squared)),
        "shared_gradient_fraction": shared_squared / total_squared,
    }


def factorized_shared_gradient_vector(grads: dict[str, np.ndarray]) -> np.ndarray:
    for key in FACTORIZED_SHARED_KEYS:
        if key not in grads:
            raise ValueError(f"missing shared gradient key {key!r}.")
    return np.concatenate([grads[key].reshape(-1) for key in FACTORIZED_SHARED_KEYS])


def shared_gradient_direction_grads(
    grads: dict[str, np.ndarray],
    normalize: bool,
) -> dict[str, np.ndarray]:
    direction = empty_factorized_grads_from_shapes(grads)
    shared_vector = factorized_shared_gradient_vector(grads)
    norm = float(np.linalg.norm(shared_vector))
    if norm <= 0.0:
        raise ValueError("shared gradient direction has zero norm.")
    scale = 1.0 / norm if normalize else 1.0
    for key in FACTORIZED_SHARED_KEYS:
        direction[key] = grads[key] * scale
    return direction


def empty_factorized_grads_from_shapes(grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = set(FACTORIZED_PARAM_KEYS) - set(grads)
    if missing:
        raise ValueError(f"gradient dict is missing keys: {sorted(missing)}.")
    return {key: np.zeros_like(grads[key]) for key in FACTORIZED_PARAM_KEYS}


def factorized_old_tangent_damage(
    model: FactorizedAddMLP,
    old_dataset: OpsDataset,
    config: OpsConfig,
    direction_grads: dict[str, np.ndarray],
    epsilon: float,
) -> float:
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    op_name = single_op_name(old_dataset, FACTORIZED_ADD_OPS)
    digit_x = digit_features_from_dataset(old_dataset, config)
    base_logits, _ = factorized_forward(model, digit_x, op_name)
    stepped = clone_factorized_add_model(model)
    factorized_apply_update(
        stepped,
        direction_grads,
        epsilon,
        FACTORIZED_SHARED_KEYS,
    )
    stepped_logits, _ = factorized_forward(stepped, digit_x, op_name)
    tangent = (stepped_logits - base_logits) / epsilon
    return float(np.mean(np.sum(tangent**2, axis=1)))


def shared_gradient_compatibility_metrics(
    model: FactorizedAddMLP,
    add01_dataset: OpsDataset,
    add12_dataset: OpsDataset,
    config: OpsConfig,
) -> dict[str, float]:
    old_grads = factorized_gradients(model, add01_dataset, config)
    new_grads = factorized_gradients(model, add12_dataset, config)
    old_vector = factorized_shared_gradient_vector(old_grads)
    new_vector = factorized_shared_gradient_vector(new_grads)
    old_norm = float(np.linalg.norm(old_vector))
    new_norm = float(np.linalg.norm(new_vector))
    if old_norm <= 0.0:
        raise ValueError("old shared gradient norm is zero.")
    if new_norm <= 0.0:
        raise ValueError("new shared gradient norm is zero.")

    raw_new_direction = shared_gradient_direction_grads(new_grads, normalize=False)
    unit_new_direction = shared_gradient_direction_grads(new_grads, normalize=True)
    return {
        "shared_grad_cosine": float(np.dot(old_vector, new_vector) / (old_norm * new_norm)),
        "old_shared_gradient_norm": old_norm,
        "new_shared_gradient_norm": new_norm,
        "old_tangent_damage_raw": factorized_old_tangent_damage(
            model,
            add01_dataset,
            config,
            raw_new_direction,
            FACTORIZED_TANGENT_EPSILON,
        ),
        "old_tangent_damage_unit": factorized_old_tangent_damage(
            model,
            add01_dataset,
            config,
            unit_new_direction,
            FACTORIZED_TANGENT_EPSILON,
        ),
    }


def factorized_gradient_partition(
    model: FactorizedAddMLP,
    dataset: OpsDataset,
    config: OpsConfig,
) -> dict[str, float]:
    grads = factorized_gradients(model, dataset, config)
    total = gradient_squared_norm(grads, FACTORIZED_PARAM_KEYS)
    if total <= 0.0:
        raise ValueError("factorized gradient has zero total squared norm.")
    return {
        "old_router_gradient_fraction": gradient_squared_norm(
            grads,
            ("W_router_ADD01", "b_router_ADD01"),
        )
        / total,
        "new_router_gradient_fraction": gradient_squared_norm(
            grads,
            FACTORIZED_ADD12_ROUTER_KEYS,
        )
        / total,
        "shared_gradient_fraction": gradient_squared_norm(grads, FACTORIZED_SHARED_KEYS)
        / total,
    }


def gradient_components(model: TinyMLP, dataset: OpsDataset) -> dict[str, np.ndarray]:
    logits, cache = forward(model, dataset.x)
    _, dlogits = loss_and_grad_logits(logits, dataset.y)
    grads = backward(model, cache, dlogits)
    return {
        "incoming": np.linalg.norm(grads["W1"], axis=0),
        "bias": np.abs(grads["b1"]),
        "outgoing": np.linalg.norm(grads["W2"], axis=1),
    }


def eval_loss_from_hidden(model: TinyMLP, h: np.ndarray, y: np.ndarray) -> float:
    logits = h @ model.W2 + model.b2
    loss, _ = loss_and_grad_logits(logits, y)
    return loss


def damage_measures(
    before: TinyMLP, after: TinyMLP, old_dataset: OpsDataset
) -> dict[str, np.ndarray]:
    _, before_cache = forward(before, old_dataset.x)
    _, after_cache = forward(after, old_dataset.x)
    after_loss, _ = evaluate(after, old_dataset)
    hidden_dim = before.b1.shape[0]

    loss_attribution = np.zeros(hidden_dim)
    for neuron in range(hidden_dim):
        patched_h = after_cache["h"].copy()
        patched_h[:, neuron] = before_cache["h"][:, neuron]
        patched_loss = eval_loss_from_hidden(after, patched_h, old_dataset.y)
        loss_attribution[neuron] = max(0.0, after_loss - patched_loss)

    fading = np.maximum(
        0.0,
        np.mean(np.abs(before_cache["h"]), axis=0)
        - np.mean(np.abs(after_cache["h"]), axis=0),
    )
    readout_damage = np.abs(
        np.linalg.norm(before.W2, axis=1) - np.linalg.norm(after.W2, axis=1)
    )
    total_drift = hidden_drift(before, after)
    return {
        "loss_attribution": loss_attribution,
        "fading": fading,
        "readout_damage": readout_damage,
        "total_drift": total_drift,
    }


def hidden_drift(before: TinyMLP, after: TinyMLP) -> np.ndarray:
    incoming = np.linalg.norm(after.W1 - before.W1, axis=0)
    bias = np.abs(after.b1 - before.b1)
    outgoing = np.linalg.norm(after.W2 - before.W2, axis=1)
    return incoming + bias + outgoing


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 0.0:
        raise ValueError("cosine is undefined for a zero-norm vector.")
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def print_correlation_table(
    scores: dict[str, np.ndarray], damage: dict[str, np.ndarray]
) -> None:
    print("\nSpearman rank correlation: candidate score vs damage")
    print("score        " + "  ".join(f"{name:>17}" for name in DAMAGE_NAMES))
    for score_name, score_values in scores.items():
        row = [f"{score_name:<10}"]
        for damage_name in DAMAGE_NAMES:
            row.append(f"{spearman(score_values, damage[damage_name]):>17.6f}")
        print("  ".join(row))


def top_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}.")
    if score.ndim != 1:
        raise ValueError(f"score must be one-dimensional, got shape {score.shape}.")
    if not np.all(np.isfinite(score)):
        raise ValueError("score contains non-finite values.")
    if float(np.max(score)) == float(np.min(score)):
        raise ValueError("score has no rank information; all values are identical.")
    count = max(1, int(round(len(score) * fraction)))
    order = np.argsort(-score)
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:count]] = True
    return mask


def bottom_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}.")
    if score.ndim != 1:
        raise ValueError(f"score must be one-dimensional, got shape {score.shape}.")
    if not np.all(np.isfinite(score)):
        raise ValueError("score contains non-finite values.")
    if float(np.max(score)) == float(np.min(score)):
        raise ValueError("score has no rank information; all values are identical.")
    count = max(1, int(round(len(score) * fraction)))
    order = np.argsort(score)
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:count]] = True
    return mask


def top_fraction_mask_within_pool(
    score: np.ndarray,
    pool_mask: np.ndarray,
    fraction: float,
) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}.")
    if score.shape != pool_mask.shape:
        raise ValueError(f"score shape {score.shape} does not match pool shape {pool_mask.shape}.")
    if score.ndim != 1:
        raise ValueError(f"score must be one-dimensional, got shape {score.shape}.")
    if not np.any(pool_mask):
        raise ValueError("pool_mask contains no available neurons.")
    if not np.all(np.isfinite(score)):
        raise ValueError("score contains non-finite values.")
    count = min(int(np.sum(pool_mask)), max(1, int(round(len(score) * fraction))))
    pool_indices = np.where(pool_mask)[0]
    order = pool_indices[np.argsort(-score[pool_indices])]
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:count]] = True
    return mask


def loss_with_hidden_ablation(model: TinyMLP, dataset: OpsDataset, neuron_mask: np.ndarray) -> float:
    if neuron_mask.shape != model.b1.shape:
        raise ValueError(
            f"neuron_mask shape {neuron_mask.shape} does not match hidden shape {model.b1.shape}."
        )
    if not np.any(neuron_mask):
        raise ValueError("neuron_mask contains no neurons.")
    _, cache = forward(model, dataset.x)
    h = cache["h"].copy()
    h[:, neuron_mask] = 0.0
    return eval_loss_from_hidden(model, h, dataset.y)


def pairwise_synergy_matrix(
    model: TinyMLP,
    dataset: OpsDataset,
    individual_effects: np.ndarray,
) -> np.ndarray:
    base_loss, _ = evaluate(model, dataset)
    hidden_dim = model.b1.shape[0]
    if individual_effects.shape != (hidden_dim,):
        raise ValueError(
            f"individual_effects shape {individual_effects.shape} does not match hidden shape {(hidden_dim,)}."
        )
    synergy = np.zeros((hidden_dim, hidden_dim))
    for left in range(hidden_dim):
        for right in range(left + 1, hidden_dim):
            mask = np.zeros(hidden_dim, dtype=bool)
            mask[[left, right]] = True
            joint_loss = loss_with_hidden_ablation(model, dataset, mask)
            joint_effect = max(0.0, joint_loss - base_loss)
            value = joint_effect - individual_effects[left] - individual_effects[right]
            synergy[left, right] = value
            synergy[right, left] = value
    return synergy


def family_from_synergy(synergy: np.ndarray, family_fraction: float) -> np.ndarray:
    if synergy.ndim != 2 or synergy.shape[0] != synergy.shape[1]:
        raise ValueError(f"synergy must be square, got shape {synergy.shape}.")
    if not 0.0 < family_fraction < 1.0:
        raise ValueError(f"family_fraction must be between 0 and 1, got {family_fraction}.")
    hidden_dim = synergy.shape[0]
    target_count = max(2, int(round(hidden_dim * family_fraction)))
    edge_indices = np.triu_indices(hidden_dim, k=1)
    edge_values = synergy[edge_indices]
    if not np.any(edge_values > 0.0):
        raise ValueError("no positive pairwise synergy found.")
    order = np.argsort(-edge_values)
    family = np.zeros(hidden_dim, dtype=bool)
    for edge_order_index in order:
        if edge_values[edge_order_index] <= 0.0:
            break
        left = edge_indices[0][edge_order_index]
        right = edge_indices[1][edge_order_index]
        family[left] = True
        family[right] = True
        if int(np.sum(family)) >= target_count:
            break
    if int(np.sum(family)) < target_count:
        raise ValueError(
            f"positive synergy produced only {int(np.sum(family))} family neurons; needed {target_count}."
        )
    return family


def family_blend_scale(
    old_importance: np.ndarray,
    new_need: np.ndarray,
    family_mask: np.ndarray,
    lambda_old: float,
) -> np.ndarray:
    if old_importance.shape != family_mask.shape or new_need.shape != family_mask.shape:
        raise ValueError(
            "old_importance, new_need, and family_mask must have the same shape."
        )
    if not np.any(family_mask):
        raise ValueError("family_mask contains no neurons.")
    old_norm = normalize_nonnegative_score(old_importance, "old_importance")
    new_norm = normalize_nonnegative_score(new_need, "new_need")
    family_old = float(np.max(old_norm[family_mask]))
    if family_old <= 0.0:
        raise ValueError("family old importance has no positive signal.")
    scale = np.zeros_like(new_norm)
    denom = new_norm[family_mask] + lambda_old * family_old
    active = denom > SCORE_EPSILON
    family_indices = np.where(family_mask)[0]
    scale[family_indices[active]] = new_norm[family_mask][active] / denom[active]
    return scale


def top_activating_inputs(
    model: TinyMLP,
    dataset: OpsDataset,
    neuron: int,
    fraction: float,
) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}.")
    hidden_dim = model.b1.shape[0]
    if not 0 <= neuron < hidden_dim:
        raise ValueError(f"neuron index {neuron} is outside hidden_dim={hidden_dim}.")
    h = hidden_activations(model, dataset)
    count = max(1, int(round(h.shape[0] * fraction)))
    order = np.argsort(-h[:, neuron])
    return dataset.x[order[:count]]


def project_vector_onto_input_subspace(vector: np.ndarray, inputs: np.ndarray, rank: int) -> np.ndarray:
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be two-dimensional, got shape {inputs.shape}.")
    if vector.ndim != 1:
        raise ValueError(f"vector must be one-dimensional, got shape {vector.shape}.")
    if inputs.shape[1] != vector.shape[0]:
        raise ValueError(
            f"input dimension {inputs.shape[1]} does not match vector dimension {vector.shape[0]}."
        )
    max_rank = min(inputs.shape)
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}.")
    _, _, vh = np.linalg.svd(inputs, full_matrices=False)
    basis = vh[:rank].T
    return basis @ (basis.T @ vector)


def meaning_conflict_mask(
    model: TinyMLP,
    old_related_dataset: OpsDataset,
    new_dataset: OpsDataset,
    old_score_name: str,
    protect_fraction: float,
    needed_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if old_score_name not in ("E", "AE"):
        raise ValueError(f"old_score_name must be 'E' or 'AE', got {old_score_name!r}.")
    old_scores = candidate_scores(model, old_related_dataset)[old_score_name]
    new_gradient = gradient_salience(model, new_dataset)
    old_mask = top_fraction_mask(old_scores, protect_fraction)
    needed_mask = top_fraction_mask(new_gradient, needed_fraction)
    conflict_mask = old_mask & needed_mask
    if not np.any(conflict_mask):
        raise ValueError("meaning transform found no conflict neurons.")
    return old_mask, needed_mask, conflict_mask


def transform_conflict_neuron_meaning(
    model: TinyMLP,
    old_related_dataset: OpsDataset,
    new_dataset: OpsDataset,
    conflict_mask: np.ndarray,
    rank: int,
    alpha: float,
    activation_fraction: float,
) -> dict[str, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if conflict_mask.shape != model.b1.shape:
        raise ValueError(
            f"conflict_mask shape {conflict_mask.shape} does not match hidden shape {model.b1.shape}."
        )

    alignments: list[float] = []
    projection_ratios: list[float] = []
    for neuron in np.where(conflict_mask)[0]:
        old_inputs = top_activating_inputs(
            model,
            old_related_dataset,
            int(neuron),
            activation_fraction,
        )
        new_inputs = top_activating_inputs(
            model,
            new_dataset,
            int(neuron),
            activation_fraction,
        )
        alignments.append(cosine(np.mean(old_inputs, axis=0), np.mean(new_inputs, axis=0)))
        combined_inputs = np.vstack([old_inputs, new_inputs])
        current_weight = model.W1[:, neuron].copy()
        shared_weight = project_vector_onto_input_subspace(current_weight, combined_inputs, rank)
        projection_ratios.append(float(np.linalg.norm(shared_weight) / np.linalg.norm(current_weight)))
        model.W1[:, neuron] = (1.0 - alpha) * current_weight + alpha * shared_weight

    return {
        "conflict_count": float(np.sum(conflict_mask)),
        "mean_alignment": float(np.mean(alignments)),
        "min_alignment": float(np.min(alignments)),
        "mean_projection_ratio": float(np.mean(projection_ratios)),
    }


def activation_mse(
    model: TinyMLP,
    dataset: OpsDataset,
    target_hidden: np.ndarray,
    neuron_mask: np.ndarray,
) -> float:
    h = hidden_activations(model, dataset)
    if h.shape != target_hidden.shape:
        raise ValueError(f"hidden shape {h.shape} does not match target shape {target_hidden.shape}.")
    if neuron_mask.shape != model.b1.shape:
        raise ValueError(
            f"neuron_mask shape {neuron_mask.shape} does not match hidden shape {model.b1.shape}."
        )
    if not np.any(neuron_mask):
        raise ValueError("neuron_mask contains no neurons.")
    diff = h[:, neuron_mask] - target_hidden[:, neuron_mask]
    return float(np.mean(diff**2))


def functional_transform_conflict_neurons(
    model: TinyMLP,
    old_related_dataset: OpsDataset,
    new_dataset: OpsDataset,
    analog_new_dataset: OpsDataset,
    conflict_mask: np.ndarray,
    epochs: int,
    learning_rate: float,
    new_weight: float,
) -> dict[str, float]:
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
    if new_weight <= 0.0:
        raise ValueError(f"new_weight must be positive, got {new_weight}.")
    if conflict_mask.shape != model.b1.shape:
        raise ValueError(
            f"conflict_mask shape {conflict_mask.shape} does not match hidden shape {model.b1.shape}."
        )
    if not np.any(conflict_mask):
        raise ValueError("conflict_mask contains no neurons.")

    target_old = hidden_activations(model, old_related_dataset)
    target_new = hidden_activations(model, analog_new_dataset)
    if target_new.shape[0] != new_dataset.x.shape[0]:
        raise ValueError(
            f"analog target rows {target_new.shape[0]} do not match new rows {new_dataset.x.shape[0]}."
        )

    selected = np.where(conflict_mask)[0]
    for _ in range(epochs):
        _, old_cache = forward(model, old_related_dataset.x)
        old_h = old_cache["h"][:, selected]
        old_z = old_cache["z1"][:, selected]
        old_diff = old_h - target_old[:, selected]
        old_dz = (2.0 / old_diff.size) * old_diff * (old_z > 0.0)

        _, new_cache = forward(model, new_dataset.x)
        new_h = new_cache["h"][:, selected]
        new_z = new_cache["z1"][:, selected]
        new_diff = new_h - target_new[:, selected]
        new_dz = new_weight * (2.0 / new_diff.size) * new_diff * (new_z > 0.0)

        dW_selected = old_related_dataset.x.T @ old_dz + new_dataset.x.T @ new_dz
        db_selected = np.sum(old_dz, axis=0) + np.sum(new_dz, axis=0)
        model.W1[:, selected] -= learning_rate * dW_selected
        model.b1[selected] -= learning_rate * db_selected

    old_mse = activation_mse(model, old_related_dataset, target_old, conflict_mask)
    new_mse = activation_mse(model, new_dataset, target_new, conflict_mask)
    return {
        "conflict_count": float(np.sum(conflict_mask)),
        "old_activation_mse": old_mse,
        "new_activation_mse": new_mse,
    }


def run_protection_table(
    base_checkpoint: TinyMLP,
    scores: dict[str, np.ndarray],
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    config: OpsConfig,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nThreshold protection test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print("score       protected  old_loss  old_acc  new_loss  new_acc  forgetting")
    results: dict[str, dict[str, float]] = {}
    for score_name, score_values in scores.items():
        model = clone_model(base_checkpoint)
        mask = top_fraction_mask(score_values, train_config.protected_fraction)
        train(model, new_dataset, train_config, label=f"protect_{score_name}", protected_mask=mask)
        old_loss, old_accuracy = evaluate(model, base_dataset)
        new_loss, new_accuracy = evaluate(model, new_dataset)
        results[score_name] = {
            "old_loss": old_loss,
            "old_accuracy": old_accuracy,
            "new_loss": new_loss,
            "new_accuracy": new_accuracy,
            "forgetting": old_loss - base_loss,
        }
        print(
            f"{score_name:<10} {int(np.sum(mask)):>9} "
            f"{old_loss:>8.4f} {old_accuracy:>7.3f} "
            f"{new_loss:>8.4f} {new_accuracy:>7.3f} "
            f"{old_loss - base_loss:>10.4f}"
        )
    return results


def protection_metrics(
    base_checkpoint: TinyMLP,
    scores: dict[str, np.ndarray],
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    results: dict[str, dict[str, float]] = {}
    for score_name, score_values in scores.items():
        model = clone_model(base_checkpoint)
        mask = top_fraction_mask(score_values, train_config.protected_fraction)
        train(model, new_dataset, train_config, label=f"protect_{score_name}", protected_mask=mask)
        old_loss, old_accuracy = evaluate(model, base_dataset)
        new_loss, new_accuracy = evaluate(model, new_dataset)
        results[score_name] = {
            "old_loss": old_loss,
            "old_accuracy": old_accuracy,
            "new_loss": new_loss,
            "new_accuracy": new_accuracy,
            "forgetting": old_loss - base_loss,
        }
    return results


def surgical_variants(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    old_scores = candidate_scores(base_checkpoint, base_dataset)
    new_scores = candidate_scores(base_checkpoint, new_dataset)
    new_gradient = gradient_salience(base_checkpoint, new_dataset)
    return {
        "Eold_Enew": (old_scores["E"], new_scores["E"]),
        "Eold_Gnew": (old_scores["E"], new_gradient),
        "AEold_Gnew": (old_scores["AE"], new_gradient),
    }


def surgical_masks(
    protect_score: np.ndarray,
    needed_score: np.ndarray,
    protect_fraction: float,
    needed_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    protect_mask = top_fraction_mask(protect_score, protect_fraction)
    needed_mask = top_fraction_mask(needed_score, needed_fraction)
    allowed_mask = needed_mask & ~protect_mask
    if not np.any(allowed_mask):
        raise ValueError(
            "surgical mask has zero allowed neurons; lower protected_fraction or raise needed_fraction."
        )
    return protect_mask, needed_mask, allowed_mask


def surgical_metrics(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    new_gradient = gradient_salience(base_checkpoint, new_dataset)
    results: dict[str, dict[str, float]] = {}
    for variant_name, (protect_score, needed_score) in surgical_variants(
        base_checkpoint, base_dataset, new_dataset
    ).items():
        model = clone_model(base_checkpoint)
        protect_mask, needed_mask, allowed_mask = surgical_masks(
            protect_score,
            needed_score,
            train_config.protected_fraction,
            train_config.needed_fraction,
        )
        conflict_mask = protect_mask & needed_mask
        blocked_new_gradient = float(np.sum(new_gradient[conflict_mask]))
        allowed_new_gradient = float(np.sum(new_gradient[allowed_mask]))
        gradient_denom = blocked_new_gradient + allowed_new_gradient
        blocked_new_gradient_fraction = (
            blocked_new_gradient / gradient_denom if gradient_denom > 0.0 else 0.0
        )
        train(
            model,
            new_dataset,
            train_config,
            label=f"surgical_{variant_name}",
            protected_mask=protect_mask,
            allowed_mask=allowed_mask,
        )
        old_loss, old_accuracy = evaluate(model, base_dataset)
        new_loss, new_accuracy = evaluate(model, new_dataset)
        results[variant_name] = {
            "protected_count": float(np.sum(protect_mask)),
            "needed_count": float(np.sum(needed_mask)),
            "allowed_count": float(np.sum(allowed_mask)),
            "conflict_count": float(np.sum(conflict_mask)),
            "blocked_new_gradient_fraction": blocked_new_gradient_fraction,
            "old_loss": old_loss,
            "old_accuracy": old_accuracy,
            "new_loss": new_loss,
            "new_accuracy": new_accuracy,
            "forgetting": old_loss - base_loss,
        }
    return results


def run_surgical_table(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nSurgical mask test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print(
        "variant       protect  needed  allowed  conflict  blocked_g  "
        "old_loss  old_acc  new_loss  new_acc  forgetting"
    )
    metrics = surgical_metrics(base_checkpoint, base_dataset, new_dataset, train_config)
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<12} "
            f"{int(values['protected_count']):>7} "
            f"{int(values['needed_count']):>7} "
            f"{int(values['allowed_count']):>8} "
            f"{int(values['conflict_count']):>9} "
            f"{values['blocked_new_gradient_fraction']:>9.3f} "
            f"{values['old_loss']:>8.4f} {values['old_accuracy']:>7.3f} "
            f"{values['new_loss']:>8.4f} {values['new_accuracy']:>7.3f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def allocation_masks(
    old_score: np.ndarray,
    new_gradient: np.ndarray,
    protect_fraction: float,
    allocation_fraction: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    protect_mask = top_fraction_mask(old_score, protect_fraction)
    safe_mask = ~protect_mask
    low_old_mask = bottom_fraction_mask(old_score, allocation_fraction) & safe_mask
    if not np.any(low_old_mask):
        raise ValueError("low-old allocation mask has zero allowed neurons.")
    safe_top_gradient_mask = top_fraction_mask_within_pool(
        new_gradient,
        safe_mask,
        allocation_fraction,
    )
    return {
        "safe_all": (protect_mask, safe_mask),
        "safe_topG": (protect_mask, safe_top_gradient_mask),
        "low_old": (protect_mask, low_old_mask),
    }


def allocation_metrics(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    old_scores = candidate_scores(base_checkpoint, base_dataset)
    new_gradient = gradient_salience(base_checkpoint, new_dataset)
    gradient_total = float(np.sum(new_gradient))
    if gradient_total <= 0.0:
        raise ValueError("new task gradient is zero; allocation cannot be measured.")

    results: dict[str, dict[str, float]] = {}
    for score_name in ("E", "AE"):
        variants = allocation_masks(
            old_scores[score_name],
            new_gradient,
            train_config.protected_fraction,
            train_config.needed_fraction,
        )
        for variant_name, (protect_mask, allowed_mask) in variants.items():
            model = clone_model(base_checkpoint)
            train(
                model,
                new_dataset,
                train_config,
                label=f"allocate_{score_name}_{variant_name}",
                protected_mask=protect_mask,
                allowed_mask=allowed_mask,
            )
            old_loss, old_accuracy = evaluate(model, base_dataset)
            new_loss, new_accuracy = evaluate(model, new_dataset)
            allowed_gradient_fraction = float(np.sum(new_gradient[allowed_mask]) / gradient_total)
            key = f"{score_name}_{variant_name}"
            results[key] = {
                "protected_count": float(np.sum(protect_mask)),
                "allowed_count": float(np.sum(allowed_mask)),
                "allowed_new_gradient_fraction": allowed_gradient_fraction,
                "old_loss": old_loss,
                "old_accuracy": old_accuracy,
                "new_loss": new_loss,
                "new_accuracy": new_accuracy,
                "forgetting": old_loss - base_loss,
            }
    return results


def run_allocation_table(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nAlternative path allocation test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print("variant       protect  allowed  allowed_g  old_loss  old_acc  new_loss  new_acc  forgetting")
    metrics = allocation_metrics(base_checkpoint, base_dataset, new_dataset, train_config)
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<12} "
            f"{int(values['protected_count']):>7} "
            f"{int(values['allowed_count']):>8} "
            f"{values['allowed_new_gradient_fraction']:>9.3f} "
            f"{values['old_loss']:>8.4f} {values['old_accuracy']:>7.3f} "
            f"{values['new_loss']:>8.4f} {values['new_accuracy']:>7.3f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def reclamation_metrics(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
    seed: int,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    old_scores = candidate_scores(base_checkpoint, base_dataset)
    new_gradient = gradient_salience(base_checkpoint, new_dataset)
    gradient_total = float(np.sum(new_gradient))
    if gradient_total <= 0.0:
        raise ValueError("new task gradient is zero; reclamation cannot be measured.")

    results: dict[str, dict[str, float]] = {}
    for score_name in ("E", "AE"):
        protect_mask = top_fraction_mask(old_scores[score_name], train_config.protected_fraction)
        allowed_mask = bottom_fraction_mask(old_scores[score_name], train_config.needed_fraction)
        allowed_mask &= ~protect_mask
        if not np.any(allowed_mask):
            raise ValueError(f"{score_name} reclamation mask has zero allowed neurons.")

        model = clone_model(base_checkpoint)
        reinitialize_hidden_neurons(model, allowed_mask, seed)
        reset_old_loss, reset_old_accuracy = evaluate(model, base_dataset)
        train(
            model,
            new_dataset,
            train_config,
            label=f"reclaim_{score_name}_low_old",
            protected_mask=protect_mask,
            allowed_mask=allowed_mask,
        )
        old_loss, old_accuracy = evaluate(model, base_dataset)
        new_loss, new_accuracy = evaluate(model, new_dataset)
        allowed_gradient_fraction = float(np.sum(new_gradient[allowed_mask]) / gradient_total)
        key = f"{score_name}_low_old_reset"
        results[key] = {
            "protected_count": float(np.sum(protect_mask)),
            "allowed_count": float(np.sum(allowed_mask)),
            "allowed_new_gradient_fraction": allowed_gradient_fraction,
            "reset_old_loss": reset_old_loss,
            "reset_old_accuracy": reset_old_accuracy,
            "reset_forgetting": reset_old_loss - base_loss,
            "old_loss": old_loss,
            "old_accuracy": old_accuracy,
            "new_loss": new_loss,
            "new_accuracy": new_accuracy,
            "forgetting": old_loss - base_loss,
        }
    return results


def run_reclamation_table(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
    seed: int,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nCapacity reclamation test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print(
        "variant            allowed  allowed_g  reset_acc  old_acc  new_acc  "
        "reset_forget  forgetting"
    )
    metrics = reclamation_metrics(base_checkpoint, base_dataset, new_dataset, train_config, seed)
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<18} "
            f"{int(values['allowed_count']):>7} "
            f"{values['allowed_new_gradient_fraction']:>9.3f} "
            f"{values['reset_old_accuracy']:>9.3f} "
            f"{values['old_accuracy']:>7.3f} "
            f"{values['new_accuracy']:>7.3f} "
            f"{values['reset_forgetting']:>12.4f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def readout_decomposition_masks(
    old_score: np.ndarray,
    readout_gradient: np.ndarray,
    protect_fraction: float,
    needed_fraction: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    protect_mask = top_fraction_mask(old_score, protect_fraction)
    useful_mask = top_fraction_mask(readout_gradient, needed_fraction)
    low_old_mask = bottom_fraction_mask(old_score, needed_fraction) & ~protect_mask
    safe_mask = ~protect_mask
    safe_top_readout_mask = top_fraction_mask_within_pool(
        readout_gradient,
        safe_mask,
        needed_fraction,
    )
    conflict_readout_mask = protect_mask & useful_mask
    hybrid_readout_mask = low_old_mask | conflict_readout_mask
    if not np.any(conflict_readout_mask):
        raise ValueError("readout decomposition found no useful protected neurons.")
    if not np.any(low_old_mask):
        raise ValueError("readout decomposition found no low-old neurons.")
    return {
        "readout_all": (np.zeros_like(old_score, dtype=bool), np.ones_like(old_score, dtype=bool)),
        "safe_readout": (np.zeros_like(old_score, dtype=bool), safe_mask),
        "safe_top_readout": (np.zeros_like(old_score, dtype=bool), safe_top_readout_mask),
        "conflict_readout": (
            np.zeros_like(old_score, dtype=bool),
            conflict_readout_mask,
        ),
        "hybrid": (low_old_mask, hybrid_readout_mask),
    }


def readout_decomposition_metrics(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    old_scores = candidate_scores(base_checkpoint, base_dataset)
    readout_gradient = gradient_components(base_checkpoint, new_dataset)["outgoing"]
    readout_gradient_total = float(np.sum(readout_gradient))
    if readout_gradient_total <= 0.0:
        raise ValueError("new task readout gradient is zero; decomposition cannot be measured.")

    results: dict[str, dict[str, float]] = {}
    for score_name in ("E", "AE"):
        variants = readout_decomposition_masks(
            old_scores[score_name],
            readout_gradient,
            train_config.protected_fraction,
            train_config.needed_fraction,
        )
        for variant_name, (incoming_mask, readout_mask) in variants.items():
            model = clone_model(base_checkpoint)
            train_split_hidden_update(
                model,
                new_dataset,
                train_config,
                label=f"readout_{score_name}_{variant_name}",
                incoming_mask=incoming_mask,
                readout_mask=readout_mask,
            )
            old_loss, old_accuracy = evaluate(model, base_dataset)
            new_loss, new_accuracy = evaluate(model, new_dataset)
            key = f"{score_name}_{variant_name}"
            results[key] = {
                "incoming_count": float(np.sum(incoming_mask)),
                "readout_count": float(np.sum(readout_mask)),
                "readout_gradient_fraction": float(
                    np.sum(readout_gradient[readout_mask]) / readout_gradient_total
                ),
                "old_loss": old_loss,
                "old_accuracy": old_accuracy,
                "new_loss": new_loss,
                "new_accuracy": new_accuracy,
                "forgetting": old_loss - base_loss,
            }
    return results


def run_readout_decomposition_table(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nReadout decomposition test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print("variant              incoming  readout  readout_g  old_acc  new_acc  forgetting")
    metrics = readout_decomposition_metrics(
        base_checkpoint,
        base_dataset,
        new_dataset,
        train_config,
    )
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<20} "
            f"{int(values['incoming_count']):>8} "
            f"{int(values['readout_count']):>8} "
            f"{values['readout_gradient_fraction']:>10.3f} "
            f"{values['old_accuracy']:>7.3f} "
            f"{values['new_accuracy']:>7.3f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def blending_score_variants(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    old_scores = candidate_scores(base_checkpoint, base_dataset)
    new_scores = candidate_scores(base_checkpoint, new_dataset)
    new_gradient = gradient_salience(base_checkpoint, new_dataset)
    return {
        "Eold_Enew": (old_scores["E"], new_scores["E"]),
        "Eold_Gnew": (old_scores["E"], new_gradient),
        "AEold_Gnew": (old_scores["AE"], new_gradient),
    }


def blending_metrics(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    new_gradient = gradient_salience(base_checkpoint, new_dataset)
    gradient_total = float(np.sum(new_gradient))
    if gradient_total <= 0.0:
        raise ValueError("new task gradient is zero; blending cannot be measured.")

    results: dict[str, dict[str, float]] = {}
    for variant_name, (old_importance, new_need) in blending_score_variants(
        base_checkpoint,
        base_dataset,
        new_dataset,
    ).items():
        for lambda_old in BLEND_LAMBDAS:
            scale = blend_scale(old_importance, new_need, lambda_old)
            model = clone_model(base_checkpoint)
            train_blended_hidden_update(
                model,
                new_dataset,
                train_config,
                label=f"blend_{variant_name}_lambda_{lambda_old:g}",
                scale=scale,
            )
            old_loss, old_accuracy = evaluate(model, base_dataset)
            new_loss, new_accuracy = evaluate(model, new_dataset)
            key = f"{variant_name}_lambda_{lambda_old:g}"
            results[key] = {
                "lambda": lambda_old,
                "mean_scale": float(np.mean(scale)),
                "max_scale": float(np.max(scale)),
                "effective_new_gradient_fraction": float(np.sum(new_gradient * scale) / gradient_total),
                "old_loss": old_loss,
                "old_accuracy": old_accuracy,
                "new_loss": new_loss,
                "new_accuracy": new_accuracy,
                "forgetting": old_loss - base_loss,
            }
    return results


def run_blending_table(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nSoft gradient blending test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print("variant                    mean_s  eff_g  old_acc  new_acc  forgetting")
    metrics = blending_metrics(base_checkpoint, base_dataset, new_dataset, train_config)
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<26} "
            f"{values['mean_scale']:>6.3f} "
            f"{values['effective_new_gradient_fraction']:>6.3f} "
            f"{values['old_accuracy']:>7.3f} "
            f"{values['new_accuracy']:>7.3f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def summarize_scale_history(history: list[dict[str, float]]) -> dict[str, float]:
    mean_scales = [entry["mean_scale"] for entry in history]
    effective_gradients = [entry["effective_new_gradient_fraction"] for entry in history]
    final = history[-1]
    return {
        "recompute_count": float(len(history)),
        "avg_mean_scale": float(np.mean(mean_scales)),
        "final_mean_scale": final["mean_scale"],
        "avg_effective_new_gradient_fraction": float(np.mean(effective_gradients)),
        "final_effective_new_gradient_fraction": final["effective_new_gradient_fraction"],
    }


def online_blending_metrics(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    results: dict[str, dict[str, float]] = {}
    for old_score_name in ("E", "AE"):
        for lambda_old in ONLINE_BLEND_LAMBDAS:
            model = clone_model(base_checkpoint)
            history = train_online_blended_hidden_update(
                model,
                base_dataset,
                new_dataset,
                train_config,
                label=f"online_blend_{old_score_name}old_Gnew_lambda_{lambda_old:g}",
                old_score_name=old_score_name,
                lambda_old=lambda_old,
                recompute_every=ONLINE_BLEND_RECOMPUTE_EVERY,
            )
            old_loss, old_accuracy = evaluate(model, base_dataset)
            new_loss, new_accuracy = evaluate(model, new_dataset)
            key = (
                f"{old_score_name}old_Gnew_lambda_{lambda_old:g}_"
                f"every_{ONLINE_BLEND_RECOMPUTE_EVERY}"
            )
            results[key] = {
                "lambda": lambda_old,
                "recompute_every": float(ONLINE_BLEND_RECOMPUTE_EVERY),
                **summarize_scale_history(history),
                "old_loss": old_loss,
                "old_accuracy": old_accuracy,
                "new_loss": new_loss,
                "new_accuracy": new_accuracy,
                "forgetting": old_loss - base_loss,
            }
    return results


def run_online_blending_table(
    base_checkpoint: TinyMLP,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nOnline soft gradient blending test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print(
        "variant                           avg_s  final_s  avg_eff_g  "
        "final_eff_g  old_acc  new_acc  forgetting"
    )
    metrics = online_blending_metrics(base_checkpoint, base_dataset, new_dataset, train_config)
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<33} "
            f"{values['avg_mean_scale']:>5.3f} "
            f"{values['final_mean_scale']:>8.3f} "
            f"{values['avg_effective_new_gradient_fraction']:>9.3f} "
            f"{values['final_effective_new_gradient_fraction']:>11.3f} "
            f"{values['old_accuracy']:>7.3f} "
            f"{values['new_accuracy']:>7.3f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def family_blending_metrics(
    base_checkpoint: TinyMLP,
    config: OpsConfig,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    old_related_dataset = make_dataset(config, ("ADD01",))
    old_scores = candidate_scores(base_checkpoint, old_related_dataset)
    addition_effects = causal_ablation_effects(base_checkpoint, old_related_dataset)
    synergy = pairwise_synergy_matrix(
        base_checkpoint,
        old_related_dataset,
        addition_effects,
    )
    family_mask = family_from_synergy(synergy, FAMILY_FRACTION)
    positive_edges = synergy[np.triu_indices_from(synergy, k=1)]
    positive_edges = positive_edges[positive_edges > 0.0]
    if len(positive_edges) == 0:
        raise ValueError("positive synergy vanished after family selection.")

    new_gradient = gradient_salience(base_checkpoint, new_dataset)
    gradient_total = float(np.sum(new_gradient))
    if gradient_total <= 0.0:
        raise ValueError("new task gradient is zero; family blending cannot be measured.")

    results: dict[str, dict[str, float]] = {}
    for score_name in ("E", "AE"):
        for lambda_old in FAMILY_BLEND_LAMBDAS:
            scale = family_blend_scale(
                old_scores[score_name],
                new_gradient,
                family_mask,
                lambda_old,
            )
            model = clone_model(base_checkpoint)
            train_blended_hidden_update(
                model,
                new_dataset,
                train_config,
                label=f"family_blend_{score_name}_lambda_{lambda_old:g}",
                scale=scale,
            )
            old_loss, old_accuracy = evaluate(model, base_dataset)
            new_loss, new_accuracy = evaluate(model, new_dataset)
            key = f"{score_name}_lambda_{lambda_old:g}"
            results[key] = {
                "family_count": float(np.sum(family_mask)),
                "mean_positive_synergy": float(np.mean(positive_edges)),
                "max_positive_synergy": float(np.max(positive_edges)),
                "mean_scale": float(np.mean(scale[family_mask])),
                "effective_new_gradient_fraction": float(np.sum(new_gradient * scale) / gradient_total),
                "old_loss": old_loss,
                "old_accuracy": old_accuracy,
                "new_loss": new_loss,
                "new_accuracy": new_accuracy,
                "forgetting": old_loss - base_loss,
            }
    return results


def run_family_blending_table(
    base_checkpoint: TinyMLP,
    config: OpsConfig,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nFamily-level soft blending test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print("variant       family  mean_syn  mean_s  eff_g  old_acc  new_acc  forgetting")
    metrics = family_blending_metrics(
        base_checkpoint,
        config,
        base_dataset,
        new_dataset,
        train_config,
    )
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<12} "
            f"{int(values['family_count']):>6} "
            f"{values['mean_positive_synergy']:>9.5f} "
            f"{values['mean_scale']:>7.3f} "
            f"{values['effective_new_gradient_fraction']:>6.3f} "
            f"{values['old_accuracy']:>7.3f} "
            f"{values['new_accuracy']:>7.3f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def conflict_stats(
    old_importance: np.ndarray,
    new_need: np.ndarray,
    protect_fraction: float,
    needed_fraction: float,
) -> dict[str, float]:
    protect_mask = top_fraction_mask(old_importance, protect_fraction)
    needed_mask = top_fraction_mask(new_need, needed_fraction)
    conflict_mask = protect_mask & needed_mask
    needed_mass = float(np.sum(new_need[needed_mask]))
    if needed_mass <= 0.0:
        raise ValueError("needed new signal mass is zero.")
    return {
        "protected_count": float(np.sum(protect_mask)),
        "needed_count": float(np.sum(needed_mask)),
        "conflict_count": float(np.sum(conflict_mask)),
        "blocked_new_signal_fraction": float(np.sum(new_need[conflict_mask]) / needed_mass),
    }


def position_factorization_metrics(seed: int, verbose: bool) -> dict[str, dict[str, float]]:
    config = OpsConfig(seed=seed, use_position_flags=True)
    base_dataset = make_dataset(config, config.base_ops)
    new_dataset = make_dataset(config, (config.new_op,))
    base_train, new_train, _, _, surgical_train = build_train_configs(verbose)

    model = make_model(config)
    train(model, base_dataset, base_train, label="position_flag_base")
    base_loss, base_accuracy = evaluate(model, base_dataset)
    new_before_loss, new_before_accuracy = evaluate(model, new_dataset)
    old_scores = candidate_scores(model, base_dataset)
    new_gradient = gradient_salience(model, new_dataset)

    results: dict[str, dict[str, float]] = {}
    for score_name, lambda_old in (("E", 4.0), ("AE", 8.0)):
        stats = conflict_stats(
            old_scores[score_name],
            new_gradient,
            surgical_train.protected_fraction,
            surgical_train.needed_fraction,
        )

        naive_model = clone_model(model)
        train(naive_model, new_dataset, new_train, label=f"position_flag_naive_{score_name}")
        naive_old_loss, naive_old_accuracy = evaluate(naive_model, base_dataset)
        naive_new_loss, naive_new_accuracy = evaluate(naive_model, new_dataset)

        blend_model = clone_model(model)
        scale = blend_scale(old_scores[score_name], new_gradient, lambda_old)
        train_blended_hidden_update(
            blend_model,
            new_dataset,
            surgical_train,
            label=f"position_flag_blend_{score_name}",
            scale=scale,
        )
        blend_old_loss, blend_old_accuracy = evaluate(blend_model, base_dataset)
        blend_new_loss, blend_new_accuracy = evaluate(blend_model, new_dataset)

        results[score_name] = {
            **stats,
            "base_loss": base_loss,
            "base_accuracy": base_accuracy,
            "new_before_loss": new_before_loss,
            "new_before_accuracy": new_before_accuracy,
            "naive_old_loss": naive_old_loss,
            "naive_old_accuracy": naive_old_accuracy,
            "naive_new_loss": naive_new_loss,
            "naive_new_accuracy": naive_new_accuracy,
            "naive_forgetting": naive_old_loss - base_loss,
            "blend_lambda": lambda_old,
            "blend_mean_scale": float(np.mean(scale)),
            "blend_old_loss": blend_old_loss,
            "blend_old_accuracy": blend_old_accuracy,
            "blend_new_loss": blend_new_loss,
            "blend_new_accuracy": blend_new_accuracy,
            "blend_forgetting": blend_old_loss - base_loss,
        }
    return results


def run_position_factorization_table(seed: int, verbose: bool) -> dict[str, dict[str, float]]:
    print("\nPosition-factorized input test")
    print("score  conflict  blocked_g  naive_old  naive_new  blend_old  blend_new")
    metrics = position_factorization_metrics(seed, verbose)
    for score_name, values in metrics.items():
        print(
            f"{score_name:<5} "
            f"{int(values['conflict_count']):>8} "
            f"{values['blocked_new_signal_fraction']:>9.3f} "
            f"{values['naive_old_accuracy']:>10.3f} "
            f"{values['naive_new_accuracy']:>9.3f} "
            f"{values['blend_old_accuracy']:>10.3f} "
            f"{values['blend_new_accuracy']:>9.3f}"
        )
    return metrics


def factorized_architecture_metrics(seed: int, verbose: bool) -> dict[str, float]:
    config = OpsConfig(seed=seed)
    add01_dataset = make_dataset(config, ("ADD01",))
    add12_dataset = make_dataset(config, ("ADD12",))
    base_train, new_train, _, _, surgical_train = build_train_configs(verbose)

    entangled_model = make_model(config)
    train(entangled_model, add01_dataset, base_train, label="factorized_entangled_base")
    entangled_base_loss, entangled_base_accuracy = evaluate(entangled_model, add01_dataset)
    entangled_old_scores = candidate_scores(entangled_model, add01_dataset)
    entangled_new_gradient = gradient_salience(entangled_model, add12_dataset)
    entangled_conflict = conflict_stats(
        entangled_old_scores["AE"],
        entangled_new_gradient,
        surgical_train.protected_fraction,
        surgical_train.needed_fraction,
    )
    entangled_after = clone_model(entangled_model)
    train(entangled_after, add12_dataset, new_train, label="factorized_entangled_new")
    entangled_old_loss, entangled_old_accuracy = evaluate(entangled_after, add01_dataset)
    entangled_new_loss, entangled_new_accuracy = evaluate(entangled_after, add12_dataset)

    factorized_model = make_factorized_add_model(config)
    shared_add01_keys = (
        "W_router_ADD01",
        "b_router_ADD01",
        "W_op",
        "b_op",
        "W_out",
        "b_out",
    )
    train_factorized_add(
        factorized_model,
        add01_dataset,
        config,
        base_train,
        label="factorized_base",
        trainable_keys=shared_add01_keys,
    )
    factorized_base_loss, factorized_base_accuracy = factorized_evaluate(
        factorized_model,
        add01_dataset,
        config,
    )
    factorized_new_before_loss, factorized_new_before_accuracy = factorized_evaluate(
        factorized_model,
        add12_dataset,
        config,
    )
    before_partition = factorized_gradient_partition(
        factorized_model,
        add12_dataset,
        config,
    )

    factorized_after = clone_factorized_add_model(factorized_model)
    train_factorized_add(
        factorized_after,
        add12_dataset,
        config,
        new_train,
        label="factorized_new_router_only",
        trainable_keys=("W_router_ADD12", "b_router_ADD12"),
    )
    factorized_old_loss, factorized_old_accuracy = factorized_evaluate(
        factorized_after,
        add01_dataset,
        config,
    )
    factorized_new_loss, factorized_new_accuracy = factorized_evaluate(
        factorized_after,
        add12_dataset,
        config,
    )
    after_partition = factorized_gradient_partition(
        factorized_after,
        add12_dataset,
        config,
    )
    add01_op_h = factorized_op_activations(factorized_after, add01_dataset, config)
    add12_op_h = factorized_op_activations(factorized_after, add12_dataset, config)
    op_cka = centered_linear_cka(add01_op_h, add12_op_h)

    return {
        "entangled_base_loss": entangled_base_loss,
        "entangled_base_accuracy": entangled_base_accuracy,
        "entangled_conflict_count": entangled_conflict["conflict_count"],
        "entangled_blocked_new_signal_fraction": entangled_conflict[
            "blocked_new_signal_fraction"
        ],
        "entangled_old_loss": entangled_old_loss,
        "entangled_old_accuracy": entangled_old_accuracy,
        "entangled_new_loss": entangled_new_loss,
        "entangled_new_accuracy": entangled_new_accuracy,
        "entangled_forgetting": entangled_old_loss - entangled_base_loss,
        "factorized_base_loss": factorized_base_loss,
        "factorized_base_accuracy": factorized_base_accuracy,
        "factorized_new_before_loss": factorized_new_before_loss,
        "factorized_new_before_accuracy": factorized_new_before_accuracy,
        "before_old_router_gradient_fraction": before_partition[
            "old_router_gradient_fraction"
        ],
        "before_new_router_gradient_fraction": before_partition[
            "new_router_gradient_fraction"
        ],
        "before_shared_gradient_fraction": before_partition["shared_gradient_fraction"],
        "factorized_old_loss": factorized_old_loss,
        "factorized_old_accuracy": factorized_old_accuracy,
        "factorized_new_loss": factorized_new_loss,
        "factorized_new_accuracy": factorized_new_accuracy,
        "factorized_forgetting": factorized_old_loss - factorized_base_loss,
        "after_old_router_gradient_fraction": after_partition[
            "old_router_gradient_fraction"
        ],
        "after_new_router_gradient_fraction": after_partition[
            "new_router_gradient_fraction"
        ],
        "after_shared_gradient_fraction": after_partition["shared_gradient_fraction"],
        "op_representation_cka": op_cka,
    }


def run_factorized_architecture_table(seed: int, verbose: bool) -> dict[str, float]:
    print("\nFactorized architecture test")
    print(
        "model       blocked_g  old_router_g  shared_g_before  shared_g_after  "
        "old_acc  new_acc  op_cka"
    )
    metrics = factorized_architecture_metrics(seed, verbose)
    print(
        f"{'entangled':<11} "
        f"{metrics['entangled_blocked_new_signal_fraction']:>9.3f} "
        f"{'n/a':>12} "
        f"{'n/a':>15} "
        f"{'n/a':>14} "
        f"{metrics['entangled_old_accuracy']:>7.3f} "
        f"{metrics['entangled_new_accuracy']:>7.3f} "
        f"{'n/a':>7}"
    )
    print(
        f"{'factorized':<11} "
        f"{'n/a':>9} "
        f"{metrics['before_old_router_gradient_fraction']:>12.3f} "
        f"{metrics['before_shared_gradient_fraction']:>15.3f} "
        f"{metrics['after_shared_gradient_fraction']:>14.3f} "
        f"{metrics['factorized_old_accuracy']:>7.3f} "
        f"{metrics['factorized_new_accuracy']:>7.3f} "
        f"{metrics['op_representation_cka']:>7.3f}"
    )
    return metrics


def factorized_alignment_metrics(seed: int, verbose: bool) -> dict[str, dict[str, float]]:
    config = OpsConfig(seed=seed)
    add01_dataset = make_dataset(config, ("ADD01",))
    add12_dataset = make_dataset(config, ("ADD12",))
    analog_add01_dataset = analogous_add01_dataset_for_add12(config, add12_dataset)
    base_train, new_train, _, _, _ = build_train_configs(verbose)

    base_model = make_factorized_add_model(config)
    train_factorized_add(
        base_model,
        add01_dataset,
        config,
        base_train,
        label="factorized_align_base",
        trainable_keys=(
            "W_router_ADD01",
            "b_router_ADD01",
            "W_op",
            "b_op",
            "W_out",
            "b_out",
        ),
    )
    base_loss, base_accuracy = factorized_evaluate(base_model, add01_dataset, config)
    results: dict[str, dict[str, float]] = {}

    router_only = clone_factorized_add_model(base_model)
    train_factorized_add(
        router_only,
        add12_dataset,
        config,
        new_train,
        label="factorized_align_router_only",
        trainable_keys=("W_router_ADD12", "b_router_ADD12"),
    )
    router_old_loss, router_old_accuracy = factorized_evaluate(
        router_only,
        add01_dataset,
        config,
    )
    router_new_loss, router_new_accuracy = factorized_evaluate(
        router_only,
        add12_dataset,
        config,
    )
    router_partition = factorized_gradient_partition(router_only, add12_dataset, config)
    results["router_only"] = {
        "base_loss": base_loss,
        "base_accuracy": base_accuracy,
        "alignment_weight": 0.0,
        "old_loss": router_old_loss,
        "old_accuracy": router_old_accuracy,
        "new_loss": router_new_loss,
        "new_accuracy": router_new_accuracy,
        "forgetting": router_old_loss - base_loss,
        "after_shared_gradient_fraction": router_partition["shared_gradient_fraction"],
        **factorized_pair_alignment_metrics(
            router_only,
            add12_dataset,
            analog_add01_dataset,
            config,
        ),
        **shared_gradient_compatibility_metrics(
            router_only,
            add01_dataset,
            add12_dataset,
            config,
        ),
    }

    for alignment_target in FACTORIZED_ALIGNMENT_TARGETS:
        for alignment_weight in FACTORIZED_ALIGNMENT_WEIGHTS:
            model = clone_factorized_add_model(base_model)
            train_factorized_add_alignment(
                model,
                add12_dataset,
                analog_add01_dataset,
                config,
                new_train,
                label=f"factorized_align_{alignment_target}_{alignment_weight:g}",
                alignment_target=alignment_target,
                alignment_weight=alignment_weight,
            )
            old_loss, old_accuracy = factorized_evaluate(model, add01_dataset, config)
            new_loss, new_accuracy = factorized_evaluate(model, add12_dataset, config)
            partition = factorized_gradient_partition(model, add12_dataset, config)
            key = f"{alignment_target}_w_{alignment_weight:g}"
            results[key] = {
                "base_loss": base_loss,
                "base_accuracy": base_accuracy,
                "alignment_weight": alignment_weight,
                "old_loss": old_loss,
                "old_accuracy": old_accuracy,
                "new_loss": new_loss,
                "new_accuracy": new_accuracy,
                "forgetting": old_loss - base_loss,
                "after_shared_gradient_fraction": partition["shared_gradient_fraction"],
                **factorized_pair_alignment_metrics(
                    model,
                    add12_dataset,
                    analog_add01_dataset,
                    config,
                ),
                **shared_gradient_compatibility_metrics(
                    model,
                    add01_dataset,
                    add12_dataset,
                    config,
                ),
            }
    return results


def run_factorized_alignment_table(seed: int, verbose: bool) -> dict[str, dict[str, float]]:
    print("\nFactorized representation-alignment test")
    print("variant        old_acc  new_acc  shared_g  op_mse  op_pair_cos  op_class_cos  paired_cka")
    metrics = factorized_alignment_metrics(seed, verbose)
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<14} "
            f"{values['old_accuracy']:>7.3f} "
            f"{values['new_accuracy']:>7.3f} "
            f"{values['after_shared_gradient_fraction']:>8.3f} "
            f"{values['op_pair_mse']:>7.4f} "
            f"{values['op_pair_cosine']:>11.3f} "
            f"{values['op_class_cosine']:>12.3f} "
            f"{values['paired_op_cka']:>10.3f}"
        )
    print("\nFactorized shared-gradient compatibility diagnostic")
    print("variant        grad_cos  old_tan_raw  old_tan_unit  new_g_norm")
    for variant_name in ("router_only", "route_w_10", "op_w_10"):
        values = metrics[variant_name]
        print(
            f"{variant_name:<14} "
            f"{values['shared_grad_cosine']:>8.3f} "
            f"{values['old_tangent_damage_raw']:>12.6f} "
            f"{values['old_tangent_damage_unit']:>13.6f} "
            f"{values['new_shared_gradient_norm']:>10.6f}"
        )
    return metrics


def passes_consolidation_gate(
    candidate: FactorizedAddMLP,
    add01_dataset: OpsDataset,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
    min_op_pair_cosine: float,
    min_paired_op_cka: float,
    max_shared_gradient_norm: float,
) -> tuple[bool, dict[str, float]]:
    old_loss, old_accuracy = factorized_evaluate(candidate, add01_dataset, config)
    new_loss, new_accuracy = factorized_evaluate(candidate, add12_dataset, config)
    alignment = factorized_pair_alignment_metrics(
        candidate,
        add12_dataset,
        analog_add01_dataset,
        config,
    )
    pressure = factorized_gradient_pressure(candidate, add12_dataset, config)
    metrics = {
        "old_loss": old_loss,
        "old_accuracy": old_accuracy,
        "new_loss": new_loss,
        "new_accuracy": new_accuracy,
        **alignment,
        **pressure,
    }
    passed = (
        old_accuracy == 1.0
        and new_accuracy == 1.0
        and alignment["op_pair_cosine"] >= min_op_pair_cosine
        and alignment["paired_op_cka"] >= min_paired_op_cka
        and pressure["shared_gradient_norm"] <= max_shared_gradient_norm + SCORE_EPSILON
    )
    return passed, metrics


def consolidation_candidate_deltas(
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> dict[str, float]:
    required_keys = (
        "old_loss",
        "old_accuracy",
        "new_loss",
        "new_accuracy",
        "op_pair_cosine",
        "paired_op_cka",
        "shared_gradient_norm",
    )
    for key in required_keys:
        if key not in before_metrics or key not in after_metrics:
            raise ValueError(f"missing metric {key!r} for candidate delta computation.")
    return {
        "old_loss_delta": after_metrics["old_loss"] - before_metrics["old_loss"],
        "old_accuracy_delta": after_metrics["old_accuracy"] - before_metrics["old_accuracy"],
        "new_loss_delta": after_metrics["new_loss"] - before_metrics["new_loss"],
        "new_accuracy_delta": after_metrics["new_accuracy"] - before_metrics["new_accuracy"],
        "op_pair_cosine_delta": after_metrics["op_pair_cosine"]
        - before_metrics["op_pair_cosine"],
        "paired_op_cka_delta": after_metrics["paired_op_cka"] - before_metrics["paired_op_cka"],
        "shared_gradient_norm_delta": after_metrics["shared_gradient_norm"]
        - before_metrics["shared_gradient_norm"],
    }


def mean_of_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    values = [row[key] for row in rows]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values found while averaging metric {key!r}.")
    return float(np.mean(values))


def run_factorized_consolidation_gate(
    start_model: FactorizedAddMLP,
    add01_dataset: OpsDataset,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
    objective_name: str,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")

    working = clone_factorized_add_model(start_model)
    start_gate_passed, start_metrics = passes_consolidation_gate(
        working,
        add01_dataset,
        add12_dataset,
        analog_add01_dataset,
        config,
        min_op_pair_cosine=-1.0,
        min_paired_op_cka=-1.0,
        max_shared_gradient_norm=np.inf,
    )
    if not start_gate_passed:
        raise ValueError("initial consolidation model failed basic finite metric evaluation.")

    min_op_pair_cosine = (
        start_metrics["op_pair_cosine"] - FACTORIZED_CONSOLIDATION_MAX_ALIGNMENT_DROP
    )
    min_paired_op_cka = (
        start_metrics["paired_op_cka"] - FACTORIZED_CONSOLIDATION_MAX_ALIGNMENT_DROP
    )
    committed_steps = 0
    attempted_steps = 0
    rejected_steps = 0
    last_objective_loss = np.nan
    last_old_loss_component = np.nan
    last_new_loss_component = np.nan
    last_alignment_loss_component = np.nan
    current_shared_gradient_norm = start_metrics["shared_gradient_norm"]
    current_metrics = start_metrics
    accepted_deltas: list[dict[str, float]] = []
    rejected_delta: dict[str, float] | None = None

    for _ in range(steps):
        objective_loss, components, grads = factorized_consolidation_objective(
            working,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            objective_name,
        )
        candidate = clone_factorized_add_model(working)
        factorized_apply_update(
            candidate,
            grads,
            learning_rate,
            FACTORIZED_SHARED_KEYS,
        )
        attempted_steps += 1
        passed, candidate_metrics = passes_consolidation_gate(
            candidate,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            min_op_pair_cosine,
            min_paired_op_cka,
            current_shared_gradient_norm,
        )
        delta = consolidation_candidate_deltas(current_metrics, candidate_metrics)
        if not passed:
            rejected_steps += 1
            rejected_delta = delta
            break

        working = candidate
        committed_steps += 1
        current_shared_gradient_norm = candidate_metrics["shared_gradient_norm"]
        current_metrics = candidate_metrics
        accepted_deltas.append(delta)
        last_objective_loss = objective_loss
        last_old_loss_component = components["old_loss_component"]
        last_new_loss_component = components["new_loss_component"]
        last_alignment_loss_component = components["alignment_loss_component"]

    _, final_metrics = passes_consolidation_gate(
        working,
        add01_dataset,
        add12_dataset,
        analog_add01_dataset,
        config,
        min_op_pair_cosine=-1.0,
        min_paired_op_cka=-1.0,
        max_shared_gradient_norm=np.inf,
    )
    shared_gradient_drop = (
        start_metrics["shared_gradient_norm"] - final_metrics["shared_gradient_norm"]
    )
    return {
        "attempted_steps": float(attempted_steps),
        "committed_steps": float(committed_steps),
        "rejected_steps": float(rejected_steps),
        "stopped_steps": 0.0,
        "learning_rate": learning_rate,
        "start_old_accuracy": start_metrics["old_accuracy"],
        "start_new_accuracy": start_metrics["new_accuracy"],
        "start_shared_gradient_norm": start_metrics["shared_gradient_norm"],
        "start_shared_gradient_fraction": start_metrics["shared_gradient_fraction"],
        "start_op_pair_cosine": start_metrics["op_pair_cosine"],
        "start_paired_op_cka": start_metrics["paired_op_cka"],
        "final_old_loss": final_metrics["old_loss"],
        "final_old_accuracy": final_metrics["old_accuracy"],
        "final_new_loss": final_metrics["new_loss"],
        "final_new_accuracy": final_metrics["new_accuracy"],
        "final_shared_gradient_norm": final_metrics["shared_gradient_norm"],
        "final_shared_gradient_fraction": final_metrics["shared_gradient_fraction"],
        "shared_gradient_norm_drop": shared_gradient_drop,
        "final_op_pair_cosine": final_metrics["op_pair_cosine"],
        "final_paired_op_cka": final_metrics["paired_op_cka"],
        "final_op_pair_mse": final_metrics["op_pair_mse"],
        "last_objective_loss": float(last_objective_loss),
        "last_old_loss_component": float(last_old_loss_component),
        "last_new_loss_component": float(last_new_loss_component),
        "last_alignment_loss_component": float(last_alignment_loss_component),
        "accepted_mean_old_loss_delta": mean_of_metric(accepted_deltas, "old_loss_delta"),
        "accepted_mean_new_loss_delta": mean_of_metric(accepted_deltas, "new_loss_delta"),
        "accepted_mean_op_pair_cosine_delta": mean_of_metric(
            accepted_deltas,
            "op_pair_cosine_delta",
        ),
        "accepted_mean_paired_op_cka_delta": mean_of_metric(
            accepted_deltas,
            "paired_op_cka_delta",
        ),
        "accepted_mean_shared_gradient_norm_delta": mean_of_metric(
            accepted_deltas,
            "shared_gradient_norm_delta",
        ),
        "rejected_old_loss_delta": np.nan
        if rejected_delta is None
        else rejected_delta["old_loss_delta"],
        "rejected_new_loss_delta": np.nan
        if rejected_delta is None
        else rejected_delta["new_loss_delta"],
        "rejected_op_pair_cosine_delta": np.nan
        if rejected_delta is None
        else rejected_delta["op_pair_cosine_delta"],
        "rejected_paired_op_cka_delta": np.nan
        if rejected_delta is None
        else rejected_delta["paired_op_cka_delta"],
        "rejected_shared_gradient_norm_delta": np.nan
        if rejected_delta is None
        else rejected_delta["shared_gradient_norm_delta"],
        "failed_old_loss_delta": np.nan
        if rejected_delta is None
        else rejected_delta["old_loss_delta"],
        "failed_new_loss_delta": np.nan
        if rejected_delta is None
        else rejected_delta["new_loss_delta"],
        "failed_op_pair_cosine_delta": np.nan
        if rejected_delta is None
        else rejected_delta["op_pair_cosine_delta"],
        "failed_paired_op_cka_delta": np.nan
        if rejected_delta is None
        else rejected_delta["paired_op_cka_delta"],
        "failed_shared_gradient_norm_delta": np.nan
        if rejected_delta is None
        else rejected_delta["shared_gradient_norm_delta"],
    }


def build_aligned_factorized_consolidation_start(
    seed: int,
    verbose: bool,
    label_prefix: str,
    hidden_dim: int = OpsConfig.hidden_dim,
) -> tuple[OpsConfig, OpsDataset, OpsDataset, OpsDataset, FactorizedAddMLP]:
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
    config = OpsConfig(seed=seed, hidden_dim=hidden_dim)
    add01_dataset = make_dataset(config, ("ADD01",))
    add12_dataset = make_dataset(config, ("ADD12",))
    analog_add01_dataset = analogous_add01_dataset_for_add12(config, add12_dataset)
    base_train, new_train, _, _, _ = build_train_configs(verbose)

    model = make_factorized_add_model(config)
    train_factorized_add(
        model,
        add01_dataset,
        config,
        base_train,
        label=f"{label_prefix}_base",
        trainable_keys=FACTORIZED_ADD01_SHARED_TRAIN_KEYS,
    )
    train_factorized_add_alignment(
        model,
        add12_dataset,
        analog_add01_dataset,
        config,
        new_train,
        label=f"{label_prefix}_align_op_10",
        alignment_target="op",
        alignment_weight=FACTORIZED_CONSOLIDATION_ALIGNMENT_WEIGHT,
    )
    return config, add01_dataset, add12_dataset, analog_add01_dataset, model


def factorized_consolidation_metrics(seed: int, verbose: bool) -> dict[str, dict[str, float]]:
    (
        config,
        add01_dataset,
        add12_dataset,
        analog_add01_dataset,
        model,
    ) = build_aligned_factorized_consolidation_start(
        seed,
        verbose,
        "factorized_consolidation",
    )

    results: dict[str, dict[str, float]] = {}
    for objective_name in FACTORIZED_CONSOLIDATION_OBJECTIVES:
        results[objective_name] = run_factorized_consolidation_gate(
            model,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            objective_name,
            FACTORIZED_CONSOLIDATION_STEPS,
            FACTORIZED_CONSOLIDATION_LR,
        )
    return results


def run_factorized_consolidation_table(seed: int, verbose: bool) -> dict[str, dict[str, float]]:
    print("\nFactorized shared-consolidation gate test")
    print("objective          commit  reject  old_acc  new_acc  shared_norm_drop  op_cos  paired_cka")
    metrics = factorized_consolidation_metrics(seed, verbose)
    for objective_name, values in metrics.items():
        print(
            f"{objective_name:<18} "
            f"{int(values['committed_steps']):>6} "
            f"{int(values['rejected_steps']):>7} "
            f"{values['final_old_accuracy']:>7.3f} "
            f"{values['final_new_accuracy']:>7.3f} "
            f"{values['shared_gradient_norm_drop']:>16.6f} "
            f"{values['final_op_pair_cosine']:>7.3f} "
            f"{values['final_paired_op_cka']:>10.3f}"
        )
    return metrics


def run_factorized_shared_update_baseline(
    start_model: FactorizedAddMLP,
    add01_dataset: OpsDataset,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
    objective_name: str,
    steps: int,
    learning_rate: float,
    stop_after_violation: bool,
) -> dict[str, float]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")

    working = clone_factorized_add_model(start_model)
    _, start_metrics = passes_consolidation_gate(
        working,
        add01_dataset,
        add12_dataset,
        analog_add01_dataset,
        config,
        min_op_pair_cosine=-1.0,
        min_paired_op_cka=-1.0,
        max_shared_gradient_norm=np.inf,
    )
    min_op_pair_cosine = (
        start_metrics["op_pair_cosine"] - FACTORIZED_CONSOLIDATION_MAX_ALIGNMENT_DROP
    )
    min_paired_op_cka = (
        start_metrics["paired_op_cka"] - FACTORIZED_CONSOLIDATION_MAX_ALIGNMENT_DROP
    )
    current_metrics = start_metrics
    accepted_deltas: list[dict[str, float]] = []
    failed_delta: dict[str, float] | None = None
    committed_steps = 0
    stopped_steps = 0

    for _ in range(steps):
        _, _, grads = factorized_consolidation_objective(
            working,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            objective_name,
        )
        candidate = clone_factorized_add_model(working)
        factorized_apply_update(
            candidate,
            grads,
            learning_rate,
            FACTORIZED_SHARED_KEYS,
        )
        _, candidate_metrics = passes_consolidation_gate(
            candidate,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            min_op_pair_cosine=-1.0,
            min_paired_op_cka=-1.0,
            max_shared_gradient_norm=np.inf,
        )
        delta = consolidation_candidate_deltas(current_metrics, candidate_metrics)
        working = candidate
        current_metrics = candidate_metrics
        accepted_deltas.append(delta)
        committed_steps += 1

        violated = (
            candidate_metrics["old_accuracy"] != 1.0
            or candidate_metrics["new_accuracy"] != 1.0
            or candidate_metrics["op_pair_cosine"] < min_op_pair_cosine
            or candidate_metrics["paired_op_cka"] < min_paired_op_cka
        )
        if stop_after_violation and violated:
            stopped_steps = 1
            failed_delta = delta
            break

    _, final_metrics = passes_consolidation_gate(
        working,
        add01_dataset,
        add12_dataset,
        analog_add01_dataset,
        config,
        min_op_pair_cosine=-1.0,
        min_paired_op_cka=-1.0,
        max_shared_gradient_norm=np.inf,
    )
    shared_gradient_drop = (
        start_metrics["shared_gradient_norm"] - final_metrics["shared_gradient_norm"]
    )
    return {
        "attempted_steps": float(committed_steps),
        "committed_steps": float(committed_steps),
        "rejected_steps": 0.0,
        "stopped_steps": float(stopped_steps),
        "learning_rate": learning_rate,
        "start_old_accuracy": start_metrics["old_accuracy"],
        "start_new_accuracy": start_metrics["new_accuracy"],
        "start_shared_gradient_norm": start_metrics["shared_gradient_norm"],
        "start_shared_gradient_fraction": start_metrics["shared_gradient_fraction"],
        "start_op_pair_cosine": start_metrics["op_pair_cosine"],
        "start_paired_op_cka": start_metrics["paired_op_cka"],
        "final_old_loss": final_metrics["old_loss"],
        "final_old_accuracy": final_metrics["old_accuracy"],
        "final_new_loss": final_metrics["new_loss"],
        "final_new_accuracy": final_metrics["new_accuracy"],
        "final_shared_gradient_norm": final_metrics["shared_gradient_norm"],
        "final_shared_gradient_fraction": final_metrics["shared_gradient_fraction"],
        "shared_gradient_norm_drop": shared_gradient_drop,
        "final_op_pair_cosine": final_metrics["op_pair_cosine"],
        "final_paired_op_cka": final_metrics["paired_op_cka"],
        "final_op_pair_mse": final_metrics["op_pair_mse"],
        "accepted_mean_old_loss_delta": mean_of_metric(accepted_deltas, "old_loss_delta"),
        "accepted_mean_new_loss_delta": mean_of_metric(accepted_deltas, "new_loss_delta"),
        "accepted_mean_op_pair_cosine_delta": mean_of_metric(
            accepted_deltas,
            "op_pair_cosine_delta",
        ),
        "accepted_mean_paired_op_cka_delta": mean_of_metric(
            accepted_deltas,
            "paired_op_cka_delta",
        ),
        "accepted_mean_shared_gradient_norm_delta": mean_of_metric(
            accepted_deltas,
            "shared_gradient_norm_delta",
        ),
        "failed_old_loss_delta": np.nan
        if failed_delta is None
        else failed_delta["old_loss_delta"],
        "failed_new_loss_delta": np.nan
        if failed_delta is None
        else failed_delta["new_loss_delta"],
        "failed_op_pair_cosine_delta": np.nan
        if failed_delta is None
        else failed_delta["op_pair_cosine_delta"],
        "failed_paired_op_cka_delta": np.nan
        if failed_delta is None
        else failed_delta["paired_op_cka_delta"],
        "failed_shared_gradient_norm_delta": np.nan
        if failed_delta is None
        else failed_delta["shared_gradient_norm_delta"],
    }


def factorized_consolidation_ablation_metrics(seed: int, verbose: bool) -> dict[str, dict[str, float]]:
    (
        config,
        add01_dataset,
        add12_dataset,
        analog_add01_dataset,
        model,
    ) = build_aligned_factorized_consolidation_start(
        seed,
        verbose,
        "factorized_consolidation_ablation",
    )

    return {
        "naive_new_ce": run_factorized_shared_update_baseline(
            model,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            "new_ce",
            FACTORIZED_CONSOLIDATION_STEPS,
            FACTORIZED_CONSOLIDATION_LR,
            stop_after_violation=False,
        ),
        "low_lr_new_ce": run_factorized_shared_update_baseline(
            model,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            "new_ce",
            FACTORIZED_CONSOLIDATION_STEPS,
            FACTORIZED_CONSOLIDATION_LOW_LR,
            stop_after_violation=False,
        ),
        "early_stop_new_ce": run_factorized_shared_update_baseline(
            model,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            "new_ce",
            FACTORIZED_CONSOLIDATION_STEPS,
            FACTORIZED_CONSOLIDATION_LR,
            stop_after_violation=True,
        ),
        "alignment_only": run_factorized_shared_update_baseline(
            model,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            "balanced_ce_align",
            FACTORIZED_CONSOLIDATION_STEPS,
            FACTORIZED_CONSOLIDATION_LR,
            stop_after_violation=False,
        ),
        "gate_only": run_factorized_consolidation_gate(
            model,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            "new_ce",
            FACTORIZED_CONSOLIDATION_STEPS,
            FACTORIZED_CONSOLIDATION_LR,
        ),
        "alignment_gate": run_factorized_consolidation_gate(
            model,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            "balanced_ce_align",
            FACTORIZED_CONSOLIDATION_STEPS,
            FACTORIZED_CONSOLIDATION_LR,
        ),
    }


def run_factorized_consolidation_ablation_table(
    seed: int,
    verbose: bool,
) -> dict[str, dict[str, float]]:
    print("\nFactorized consolidation ablation test")
    print(
        "variant            commit  stop  old_acc  new_acc  shared_drop  "
        "op_cos  paired_cka  acc_g_delta  stop_g_delta"
    )
    metrics = factorized_consolidation_ablation_metrics(seed, verbose)
    for variant_name, values in metrics.items():
        stop_count = values["rejected_steps"] + values["stopped_steps"]
        print(
            f"{variant_name:<18} "
            f"{int(values['committed_steps']):>6} "
            f"{int(stop_count):>5} "
            f"{values['final_old_accuracy']:>7.3f} "
            f"{values['final_new_accuracy']:>7.3f} "
            f"{values['shared_gradient_norm_drop']:>11.6f} "
            f"{values['final_op_pair_cosine']:>7.3f} "
            f"{values['final_paired_op_cka']:>10.3f} "
            f"{values['accepted_mean_shared_gradient_norm_delta']:>11.6f} "
            f"{values['failed_shared_gradient_norm_delta']:>12.6f}"
        )
    return metrics


def stress_event_step(values: dict[str, float]) -> float:
    event_step = values.get("event_step", np.nan)
    if np.isfinite(event_step):
        return float(event_step)
    return float("nan")


def behavior_alignment_violation_flags(
    metrics: dict[str, float],
    min_op_pair_cosine: float,
    min_paired_op_cka: float,
) -> tuple[str, ...]:
    flags: list[str] = []
    if metrics["old_accuracy"] != 1.0:
        flags.append("old_acc")
    if metrics["new_accuracy"] != 1.0:
        flags.append("new_acc")
    if metrics["op_pair_cosine"] < min_op_pair_cosine:
        flags.append("op_cos")
    if metrics["paired_op_cka"] < min_paired_op_cka:
        flags.append("paired_cka")
    return tuple(flags)


def assert_finite_metrics(metrics: dict[str, float], label: str) -> None:
    non_finite = {
        key: value for key, value in metrics.items() if not np.isfinite(value)
    }
    if non_finite:
        raise ValueError(f"{label} produced non-finite metrics: {non_finite}.")


def run_factorized_stress_naive(
    start_model: FactorizedAddMLP,
    add01_dataset: OpsDataset,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")

    working = clone_factorized_add_model(start_model)
    _, start_metrics = passes_consolidation_gate(
        working,
        add01_dataset,
        add12_dataset,
        analog_add01_dataset,
        config,
        min_op_pair_cosine=-1.0,
        min_paired_op_cka=-1.0,
        max_shared_gradient_norm=np.inf,
    )
    assert_finite_metrics(start_metrics, "stress naive start")
    min_op_pair_cosine = (
        start_metrics["op_pair_cosine"] - FACTORIZED_CONSOLIDATION_MAX_ALIGNMENT_DROP
    )
    min_paired_op_cka = (
        start_metrics["paired_op_cka"] - FACTORIZED_CONSOLIDATION_MAX_ALIGNMENT_DROP
    )

    current_metrics = start_metrics
    first_violation_step = np.nan
    first_violation_flag_count = 0.0

    for step in range(1, steps + 1):
        _, _, grads = factorized_consolidation_objective(
            working,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            "new_ce",
        )
        factorized_apply_update(
            working,
            grads,
            learning_rate,
            FACTORIZED_SHARED_KEYS,
        )
        _, current_metrics = passes_consolidation_gate(
            working,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            min_op_pair_cosine=-1.0,
            min_paired_op_cka=-1.0,
            max_shared_gradient_norm=np.inf,
        )
        assert_finite_metrics(current_metrics, f"stress naive step {step}")
        if not np.isfinite(first_violation_step):
            flags = behavior_alignment_violation_flags(
                current_metrics,
                min_op_pair_cosine,
                min_paired_op_cka,
            )
            if flags:
                first_violation_step = float(step)
                first_violation_flag_count = float(len(flags))

    return {
        "method_id": 0.0,
        "attempted_steps": float(steps),
        "committed_steps": float(steps),
        "rejected_steps": 0.0,
        "event_step": float(first_violation_step),
        "event_flag_count": first_violation_flag_count,
        "learning_rate": learning_rate,
        "hidden_dim": float(config.hidden_dim),
        "start_old_accuracy": start_metrics["old_accuracy"],
        "start_new_accuracy": start_metrics["new_accuracy"],
        "start_shared_gradient_norm": start_metrics["shared_gradient_norm"],
        "start_op_pair_cosine": start_metrics["op_pair_cosine"],
        "start_paired_op_cka": start_metrics["paired_op_cka"],
        "final_old_loss": current_metrics["old_loss"],
        "final_old_accuracy": current_metrics["old_accuracy"],
        "final_new_loss": current_metrics["new_loss"],
        "final_new_accuracy": current_metrics["new_accuracy"],
        "final_shared_gradient_norm": current_metrics["shared_gradient_norm"],
        "final_shared_gradient_fraction": current_metrics["shared_gradient_fraction"],
        "shared_gradient_norm_drop": (
            start_metrics["shared_gradient_norm"] - current_metrics["shared_gradient_norm"]
        ),
        "final_op_pair_cosine": current_metrics["op_pair_cosine"],
        "final_paired_op_cka": current_metrics["paired_op_cka"],
        "final_op_pair_mse": current_metrics["op_pair_mse"],
    }


def run_factorized_stress_gate(
    start_model: FactorizedAddMLP,
    add01_dataset: OpsDataset,
    add12_dataset: OpsDataset,
    analog_add01_dataset: OpsDataset,
    config: OpsConfig,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    values = run_factorized_consolidation_gate(
        start_model,
        add01_dataset,
        add12_dataset,
        analog_add01_dataset,
        config,
        "new_ce",
        steps,
        learning_rate,
    )
    event_step = np.nan
    if values["rejected_steps"] > 0.0:
        event_step = values["committed_steps"] + 1.0
    values["method_id"] = 1.0
    values["event_step"] = float(event_step)
    values["event_flag_count"] = values["rejected_steps"]
    values["hidden_dim"] = float(config.hidden_dim)
    return values


def invalid_stress_result(
    method_id: float,
    config: OpsConfig,
    learning_rate: float,
    steps: int,
    start_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        "method_id": method_id,
        "attempted_steps": 0.0,
        "committed_steps": 0.0,
        "rejected_steps": 0.0,
        "event_step": 0.0,
        "event_flag_count": 1.0,
        "learning_rate": learning_rate,
        "hidden_dim": float(config.hidden_dim),
        "start_old_accuracy": start_metrics["old_accuracy"],
        "start_new_accuracy": start_metrics["new_accuracy"],
        "start_shared_gradient_norm": start_metrics["shared_gradient_norm"],
        "start_op_pair_cosine": start_metrics["op_pair_cosine"],
        "start_paired_op_cka": start_metrics["paired_op_cka"],
        "final_old_loss": start_metrics["old_loss"],
        "final_old_accuracy": start_metrics["old_accuracy"],
        "final_new_loss": start_metrics["new_loss"],
        "final_new_accuracy": start_metrics["new_accuracy"],
        "final_shared_gradient_norm": start_metrics["shared_gradient_norm"],
        "final_shared_gradient_fraction": start_metrics["shared_gradient_fraction"],
        "shared_gradient_norm_drop": 0.0,
        "final_op_pair_cosine": start_metrics["op_pair_cosine"],
        "final_paired_op_cka": start_metrics["paired_op_cka"],
        "final_op_pair_mse": start_metrics["op_pair_mse"],
    }


def factorized_consolidation_stress_metrics(
    seed: int,
    hidden_dims: tuple[int, ...],
    learning_rates: tuple[float, ...],
    step_counts: tuple[int, ...],
    verbose: bool,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for hidden_dim in hidden_dims:
        (
            config,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            model,
        ) = build_aligned_factorized_consolidation_start(
            seed,
            verbose,
            f"stress_h{hidden_dim}",
            hidden_dim=hidden_dim,
        )
        start_passed, start_metrics = passes_consolidation_gate(
            model,
            add01_dataset,
            add12_dataset,
            analog_add01_dataset,
            config,
            min_op_pair_cosine=-1.0,
            min_paired_op_cka=-1.0,
            max_shared_gradient_norm=np.inf,
        )
        assert_finite_metrics(start_metrics, f"stress h{hidden_dim} start")
        for steps in step_counts:
            for learning_rate in learning_rates:
                setting = f"h{hidden_dim}_lr{learning_rate:g}_steps{steps}"
                if not start_passed:
                    results[f"{setting}_naive"] = invalid_stress_result(
                        0.0,
                        config,
                        learning_rate,
                        steps,
                        start_metrics,
                    )
                    results[f"{setting}_gate"] = invalid_stress_result(
                        1.0,
                        config,
                        learning_rate,
                        steps,
                        start_metrics,
                    )
                    continue
                results[f"{setting}_naive"] = run_factorized_stress_naive(
                    model,
                    add01_dataset,
                    add12_dataset,
                    analog_add01_dataset,
                    config,
                    steps,
                    learning_rate,
                )
                results[f"{setting}_gate"] = run_factorized_stress_gate(
                    model,
                    add01_dataset,
                    add12_dataset,
                    analog_add01_dataset,
                    config,
                    steps,
                    learning_rate,
                )
    return results


def run_factorized_consolidation_stress_table(
    seed: int,
    hidden_dims: tuple[int, ...],
    learning_rates: tuple[float, ...],
    step_counts: tuple[int, ...],
    verbose: bool,
) -> dict[str, dict[str, float]]:
    print("\nFactorized consolidation stress test")
    print(
        "setting                    method  commit  event  old_acc  new_acc  "
        "shared_drop  final_norm  op_cos  paired_cka"
    )
    metrics = factorized_consolidation_stress_metrics(
        seed,
        hidden_dims,
        learning_rates,
        step_counts,
        verbose,
    )
    for setting_name, values in metrics.items():
        method = "gate" if values["method_id"] == 1.0 else "naive"
        event = values["event_step"]
        event_text = "none" if not np.isfinite(event) else f"{int(event)}"
        print(
            f"{setting_name:<26} "
            f"{method:<6} "
            f"{int(values['committed_steps']):>6} "
            f"{event_text:>6} "
            f"{values['final_old_accuracy']:>7.3f} "
            f"{values['final_new_accuracy']:>7.3f} "
            f"{values['shared_gradient_norm_drop']:>11.6f} "
            f"{values['final_shared_gradient_norm']:>10.6f} "
            f"{values['final_op_pair_cosine']:>7.3f} "
            f"{values['final_paired_op_cka']:>10.3f}"
        )
    return metrics


def meaning_transform_metrics(
    base_checkpoint: TinyMLP,
    config: OpsConfig,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    old_related_dataset = make_dataset(config, ("ADD01",))
    results: dict[str, dict[str, float]] = {}
    for old_score_name in ("E", "AE"):
        _, _, conflict_mask = meaning_conflict_mask(
            base_checkpoint,
            old_related_dataset,
            new_dataset,
            old_score_name,
            train_config.protected_fraction,
            train_config.needed_fraction,
        )
        for rank in MEANING_TRANSFORM_RANKS:
            for alpha in MEANING_TRANSFORM_ALPHAS:
                model = clone_model(base_checkpoint)
                transform_stats = transform_conflict_neuron_meaning(
                    model,
                    old_related_dataset,
                    new_dataset,
                    conflict_mask,
                    rank,
                    alpha,
                    MEANING_ACTIVATION_TOP_FRACTION,
                )
                transformed_old_loss, transformed_old_accuracy = evaluate(model, base_dataset)
                transformed_new_loss, transformed_new_accuracy = evaluate(model, new_dataset)
                train(
                    model,
                    new_dataset,
                    train_config,
                    label=f"meaning_{old_score_name}_rank_{rank}_alpha_{alpha:g}",
                    allowed_mask=conflict_mask,
                )
                old_loss, old_accuracy = evaluate(model, base_dataset)
                new_loss, new_accuracy = evaluate(model, new_dataset)
                key = f"{old_score_name}_rank_{rank}_alpha_{alpha:g}"
                results[key] = {
                    **transform_stats,
                    "rank": float(rank),
                    "alpha": alpha,
                    "transform_old_loss": transformed_old_loss,
                    "transform_old_accuracy": transformed_old_accuracy,
                    "transform_new_loss": transformed_new_loss,
                    "transform_new_accuracy": transformed_new_accuracy,
                    "transform_forgetting": transformed_old_loss - base_loss,
                    "old_loss": old_loss,
                    "old_accuracy": old_accuracy,
                    "new_loss": new_loss,
                    "new_accuracy": new_accuracy,
                    "forgetting": old_loss - base_loss,
                }
    return results


def run_meaning_transform_table(
    base_checkpoint: TinyMLP,
    config: OpsConfig,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nMeaning transformation test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print(
        "variant              conflict  align  proj_r  trans_old  "
        "old_acc  new_acc  forgetting"
    )
    metrics = meaning_transform_metrics(
        base_checkpoint,
        config,
        base_dataset,
        new_dataset,
        train_config,
    )
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<20} "
            f"{int(values['conflict_count']):>8} "
            f"{values['mean_alignment']:>6.3f} "
            f"{values['mean_projection_ratio']:>7.3f} "
            f"{values['transform_old_accuracy']:>9.3f} "
            f"{values['old_accuracy']:>7.3f} "
            f"{values['new_accuracy']:>7.3f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def functional_transform_metrics(
    base_checkpoint: TinyMLP,
    config: OpsConfig,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, _ = evaluate(base_checkpoint, base_dataset)
    old_related_dataset = make_dataset(config, ("ADD01",))
    analog_new_dataset = analogous_add01_dataset_for_add12(config, new_dataset)
    results: dict[str, dict[str, float]] = {}
    for old_score_name in ("E", "AE"):
        _, _, conflict_mask = meaning_conflict_mask(
            base_checkpoint,
            old_related_dataset,
            new_dataset,
            old_score_name,
            train_config.protected_fraction,
            train_config.needed_fraction,
        )
        transformed = clone_model(base_checkpoint)
        transform_stats = functional_transform_conflict_neurons(
            transformed,
            old_related_dataset,
            new_dataset,
            analog_new_dataset,
            conflict_mask,
            FUNCTIONAL_TRANSFORM_EPOCHS,
            FUNCTIONAL_TRANSFORM_LR,
            FUNCTIONAL_TRANSFORM_NEW_WEIGHT,
        )
        transformed_old_loss, transformed_old_accuracy = evaluate(transformed, base_dataset)
        transformed_new_loss, transformed_new_accuracy = evaluate(transformed, new_dataset)

        for update_mode in ("readout", "full"):
            model = clone_model(transformed)
            if update_mode == "readout":
                train_split_hidden_update(
                    model,
                    new_dataset,
                    train_config,
                    label=f"functional_{old_score_name}_{update_mode}",
                    incoming_mask=np.zeros_like(conflict_mask, dtype=bool),
                    readout_mask=conflict_mask,
                )
            elif update_mode == "full":
                train(
                    model,
                    new_dataset,
                    train_config,
                    label=f"functional_{old_score_name}_{update_mode}",
                    allowed_mask=conflict_mask,
                )
            else:
                raise ValueError(f"unknown functional update_mode={update_mode!r}.")

            old_loss, old_accuracy = evaluate(model, base_dataset)
            new_loss, new_accuracy = evaluate(model, new_dataset)
            key = f"{old_score_name}_{update_mode}"
            results[key] = {
                **transform_stats,
                "transform_old_loss": transformed_old_loss,
                "transform_old_accuracy": transformed_old_accuracy,
                "transform_new_loss": transformed_new_loss,
                "transform_new_accuracy": transformed_new_accuracy,
                "transform_forgetting": transformed_old_loss - base_loss,
                "old_loss": old_loss,
                "old_accuracy": old_accuracy,
                "new_loss": new_loss,
                "new_accuracy": new_accuracy,
                "forgetting": old_loss - base_loss,
            }
    return results


def run_functional_transform_table(
    base_checkpoint: TinyMLP,
    config: OpsConfig,
    base_dataset: OpsDataset,
    new_dataset: OpsDataset,
    train_config: TrainConfig,
) -> dict[str, dict[str, float]]:
    base_loss, base_accuracy = evaluate(base_checkpoint, base_dataset)
    print("\nFunctional transformation test")
    print(f"base_old_loss={base_loss:.4f} base_old_accuracy={base_accuracy:.3f}")
    print(
        "variant        conflict  old_mse  new_mse  trans_old  "
        "old_acc  new_acc  forgetting"
    )
    metrics = functional_transform_metrics(
        base_checkpoint,
        config,
        base_dataset,
        new_dataset,
        train_config,
    )
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<14} "
            f"{int(values['conflict_count']):>8} "
            f"{values['old_activation_mse']:>8.5f} "
            f"{values['new_activation_mse']:>8.5f} "
            f"{values['transform_old_accuracy']:>9.3f} "
            f"{values['old_accuracy']:>7.3f} "
            f"{values['new_accuracy']:>7.3f} "
            f"{values['forgetting']:>10.4f}"
        )
    return metrics


def print_top_usage(scores: dict[str, np.ndarray], top_k: int = 8) -> None:
    usage = scores["usage"]
    order = np.argsort(-usage)[:top_k]
    print("top_usage_neurons")
    for rank, neuron in enumerate(order, start=1):
        print(
            f"  rank={rank:02d} neuron={int(neuron):02d} "
            f"usage={usage[neuron]:.6f} "
            f"activation={scores['activation_strength'][neuron]:.6f} "
            f"downstream={scores['downstream_influence'][neuron]:.6f} "
            f"causal={scores['causal_effect'][neuron]:.6f}"
        )


def build_train_configs(
    verbose: bool,
) -> tuple[TrainConfig, TrainConfig, TrainConfig, TrainConfig, TrainConfig]:
    if verbose:
        return (
            TrainConfig(epochs=2500, learning_rate=0.08, log_every=500),
            TrainConfig(epochs=800, learning_rate=0.05, log_every=200),
            TrainConfig(
                epochs=800,
                learning_rate=0.05,
                log_every=200,
                protection_strength=15.0,
            ),
            TrainConfig(
                epochs=800,
                learning_rate=0.05,
                log_every=800,
                protected_fraction=0.2,
            ),
            TrainConfig(
                epochs=800,
                learning_rate=0.05,
                log_every=800,
                protected_fraction=0.2,
                needed_fraction=0.5,
            ),
        )
    return (
        TrainConfig(epochs=2500, learning_rate=0.08, log_every=10_000, quiet=True),
        TrainConfig(epochs=800, learning_rate=0.05, log_every=10_000, quiet=True),
        TrainConfig(
            epochs=800,
            learning_rate=0.05,
            log_every=10_000,
            protection_strength=15.0,
            quiet=True,
        ),
        TrainConfig(
            epochs=800,
            learning_rate=0.05,
            log_every=10_000,
            protected_fraction=0.2,
            quiet=True,
        ),
        TrainConfig(
            epochs=800,
            learning_rate=0.05,
            log_every=10_000,
            protected_fraction=0.2,
            needed_fraction=0.5,
            quiet=True,
        ),
    )


def run_seed(seed: int, verbose: bool) -> dict[str, object]:
    config = OpsConfig(seed=seed)
    base_dataset = make_dataset(config, config.base_ops)
    new_dataset = make_dataset(config, (config.new_op,))
    all_old_and_new = make_dataset(config, config.base_ops + (config.new_op,))
    (
        base_train,
        new_train,
        protected_new_train,
        threshold_protection_train,
        surgical_train,
    ) = build_train_configs(verbose)

    model = make_model(config)
    if verbose:
        print("Training base operations")
    train(model, base_dataset, base_train, label="base")
    if verbose:
        print(f"base_by_op={evaluate_by_op(model, base_dataset, config)}")
        print(f"new_op_before={evaluate_by_op(model, new_dataset, config)}")

    scores = usage_scores(model, base_dataset)
    if verbose:
        print_top_usage(scores)
    candidates = candidate_scores(model, base_dataset)
    subspace = activation_subspace_metrics(model, config, SUBSPACE_ENERGY)
    if verbose:
        print_activation_subspace_table(subspace)

    base_checkpoint = clone_model(model)
    normal_model = clone_model(base_checkpoint)
    protected_model = clone_model(base_checkpoint)

    if verbose:
        print("\nTraining new op with normal updates")
    train(normal_model, new_dataset, new_train, label="new_normal")
    if verbose:
        print(f"normal_old_by_op={evaluate_by_op(normal_model, base_dataset, config)}")
        print(f"normal_new_by_op={evaluate_by_op(normal_model, new_dataset, config)}")
    normal_drift = hidden_drift(base_checkpoint, normal_model)
    if verbose:
        print(f"normal_usage_drift_corr={pearson(scores['usage'], normal_drift):.6f}")
        print(f"normal_activation_drift_corr={pearson(scores['activation_strength'], normal_drift):.6f}")
    damage = damage_measures(base_checkpoint, normal_model, base_dataset)
    if verbose:
        print_correlation_table(candidates, damage)

    if verbose:
        print("\nTraining new op with usage-protected hidden updates")
    train(
        protected_model,
        new_dataset,
        protected_new_train,
        label="new_usage_protected",
        usage=scores["usage"],
    )
    if verbose:
        print(f"protected_old_by_op={evaluate_by_op(protected_model, base_dataset, config)}")
        print(f"protected_new_by_op={evaluate_by_op(protected_model, new_dataset, config)}")
    protected_drift = hidden_drift(base_checkpoint, protected_model)
    if verbose:
        print(f"protected_usage_drift_corr={pearson(scores['usage'], protected_drift):.6f}")
        print(f"protected_activation_drift_corr={pearson(scores['activation_strength'], protected_drift):.6f}")

    normal_loss, normal_accuracy = evaluate(normal_model, all_old_and_new)
    protected_loss, protected_accuracy = evaluate(protected_model, all_old_and_new)
    if verbose:
        print("\nCombined old+new evaluation")
        print(f"normal_loss={normal_loss:.4f} normal_accuracy={normal_accuracy:.3f}")
        print(f"protected_loss={protected_loss:.4f} protected_accuracy={protected_accuracy:.3f}")
        protections = run_protection_table(
            base_checkpoint,
            candidates,
            base_dataset,
            new_dataset,
            config,
            threshold_protection_train,
        )
        surgical = run_surgical_table(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        allocation = run_allocation_table(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        reclamation = run_reclamation_table(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
            seed,
        )
        readout_decomposition = run_readout_decomposition_table(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        blending = run_blending_table(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        online_blending = run_online_blending_table(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        family_blending = run_family_blending_table(
            base_checkpoint,
            config,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        position_factorization = run_position_factorization_table(seed, verbose)
        factorized_architecture = run_factorized_architecture_table(seed, verbose)
        factorized_alignment = run_factorized_alignment_table(seed, verbose)
        factorized_consolidation = run_factorized_consolidation_table(seed, verbose)
        factorized_consolidation_ablation = run_factorized_consolidation_ablation_table(
            seed,
            verbose,
        )
        meaning_transform = run_meaning_transform_table(
            base_checkpoint,
            config,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        functional_transform = run_functional_transform_table(
            base_checkpoint,
            config,
            base_dataset,
            new_dataset,
            surgical_train,
        )
    else:
        protections = protection_metrics(
            base_checkpoint,
            candidates,
            base_dataset,
            new_dataset,
            threshold_protection_train,
        )
        surgical = surgical_metrics(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        allocation = allocation_metrics(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        reclamation = reclamation_metrics(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
            seed,
        )
        readout_decomposition = readout_decomposition_metrics(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        blending = blending_metrics(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        online_blending = online_blending_metrics(
            base_checkpoint,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        family_blending = family_blending_metrics(
            base_checkpoint,
            config,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        position_factorization = position_factorization_metrics(seed, verbose)
        factorized_architecture = factorized_architecture_metrics(seed, verbose)
        factorized_alignment = factorized_alignment_metrics(seed, verbose)
        factorized_consolidation = factorized_consolidation_metrics(seed, verbose)
        factorized_consolidation_ablation = factorized_consolidation_ablation_metrics(
            seed,
            verbose,
        )
        meaning_transform = meaning_transform_metrics(
            base_checkpoint,
            config,
            base_dataset,
            new_dataset,
            surgical_train,
        )
        functional_transform = functional_transform_metrics(
            base_checkpoint,
            config,
            base_dataset,
            new_dataset,
            surgical_train,
        )

    correlations = {
        score_name: {
            damage_name: spearman(candidates[score_name], damage[damage_name])
            for damage_name in DAMAGE_NAMES
        }
        for score_name in SCORE_NAMES
    }
    return {
        "seed": seed,
        "normal_loss": normal_loss,
        "normal_accuracy": normal_accuracy,
        "protected_loss": protected_loss,
        "protected_accuracy": protected_accuracy,
        "correlations": correlations,
        "subspace": subspace,
        "protections": protections,
        "surgical": surgical,
        "allocation": allocation,
        "reclamation": reclamation,
        "readout_decomposition": readout_decomposition,
        "blending": blending,
        "online_blending": online_blending,
        "family_blending": family_blending,
        "position_factorization": position_factorization,
        "factorized_architecture": factorized_architecture,
        "factorized_alignment": factorized_alignment,
        "factorized_consolidation": factorized_consolidation,
        "factorized_consolidation_ablation": factorized_consolidation_ablation,
        "meaning_transform": meaning_transform,
        "functional_transform": functional_transform,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.array(values, dtype=float)
    return float(np.mean(array)), float(np.std(array))


def mean_std_finite(values: list[float]) -> tuple[float, float, int]:
    array = np.array(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(finite)), float(np.std(finite)), int(len(finite))


def print_multi_seed_summary(results: list[dict[str, object]]) -> None:
    print("\nMulti-seed summary")
    print(f"seeds={[result['seed'] for result in results]}")

    print("\nCorrelation summary: mean +/- std")
    for damage_name in DAMAGE_NAMES:
        print(f"\nDamage target: {damage_name}")
        for score_name in SCORE_NAMES:
            values = [
                result["correlations"][score_name][damage_name]  # type: ignore[index]
                for result in results
            ]
            mean, std = mean_std(values)
            print(f"  {score_name:<4} rho={mean: .4f} +/- {std:.4f}")

    print(f"\nActivation subspace overlap summary, energy={SUBSPACE_ENERGY:.2f}")
    print("pair             max_cos       mean_cos      min_angle     mean_angle")
    pair_names = tuple(results[0]["subspace"].keys())  # type: ignore[union-attr]
    for pair_name in pair_names:
        max_cos_values = [
            result["subspace"][pair_name]["max_cosine"]  # type: ignore[index]
            for result in results
        ]
        mean_cos_values = [
            result["subspace"][pair_name]["mean_cosine"]  # type: ignore[index]
            for result in results
        ]
        min_angle_values = [
            result["subspace"][pair_name]["min_angle_deg"]  # type: ignore[index]
            for result in results
        ]
        mean_angle_values = [
            result["subspace"][pair_name]["mean_angle_deg"]  # type: ignore[index]
            for result in results
        ]
        max_cos_mean, max_cos_std = mean_std(max_cos_values)
        mean_cos_mean, mean_cos_std = mean_std(mean_cos_values)
        min_angle_mean, min_angle_std = mean_std(min_angle_values)
        mean_angle_mean, mean_angle_std = mean_std(mean_angle_values)
        print(
            f"{pair_name:<16} "
            f"{max_cos_mean:.3f}+/-{max_cos_std:.3f}  "
            f"{mean_cos_mean:.3f}+/-{mean_cos_std:.3f}  "
            f"{min_angle_mean:.2f}+/-{min_angle_std:.2f}  "
            f"{mean_angle_mean:.2f}+/-{mean_angle_std:.2f}"
        )

    print("\nProtection summary: mean +/- std")
    print("score     old_acc        new_acc        forgetting")
    for score_name in SCORE_NAMES:
        old_acc_values = [
            result["protections"][score_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["protections"][score_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["protections"][score_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{score_name:<8} "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nSurgical summary: mean +/- std")
    print("variant       allowed       conflict      blocked_g     old_acc        new_acc        forgetting")
    for variant_name in ("Eold_Enew", "Eold_Gnew", "AEold_Gnew"):
        allowed_values = [
            result["surgical"][variant_name]["allowed_count"]  # type: ignore[index]
            for result in results
        ]
        conflict_values = [
            result["surgical"][variant_name]["conflict_count"]  # type: ignore[index]
            for result in results
        ]
        blocked_values = [
            result["surgical"][variant_name]["blocked_new_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["surgical"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["surgical"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["surgical"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        allowed_mean, allowed_std = mean_std(allowed_values)
        conflict_mean, conflict_std = mean_std(conflict_values)
        blocked_mean, blocked_std = mean_std(blocked_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<12} "
            f"{allowed_mean:.1f}+/-{allowed_std:.1f}  "
            f"{conflict_mean:.1f}+/-{conflict_std:.1f}  "
            f"{blocked_mean:.3f}+/-{blocked_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nAlternative path allocation summary: mean +/- std")
    print("variant       allowed       allowed_g     old_acc        new_acc        forgetting")
    for variant_name in (
        "E_safe_all",
        "E_safe_topG",
        "E_low_old",
        "AE_safe_all",
        "AE_safe_topG",
        "AE_low_old",
    ):
        allowed_values = [
            result["allocation"][variant_name]["allowed_count"]  # type: ignore[index]
            for result in results
        ]
        allowed_gradient_values = [
            result["allocation"][variant_name]["allowed_new_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["allocation"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["allocation"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["allocation"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        allowed_mean, allowed_std = mean_std(allowed_values)
        gradient_mean, gradient_std = mean_std(allowed_gradient_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<12} "
            f"{allowed_mean:.1f}+/-{allowed_std:.1f}  "
            f"{gradient_mean:.3f}+/-{gradient_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nCapacity reclamation summary: mean +/- std")
    print(
        "variant            allowed       allowed_g     reset_acc     old_acc       "
        "new_acc       reset_forget   forgetting"
    )
    for variant_name in ("E_low_old_reset", "AE_low_old_reset"):
        allowed_values = [
            result["reclamation"][variant_name]["allowed_count"]  # type: ignore[index]
            for result in results
        ]
        allowed_gradient_values = [
            result["reclamation"][variant_name]["allowed_new_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        reset_acc_values = [
            result["reclamation"][variant_name]["reset_old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["reclamation"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["reclamation"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        reset_forgetting_values = [
            result["reclamation"][variant_name]["reset_forgetting"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["reclamation"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        allowed_mean, allowed_std = mean_std(allowed_values)
        gradient_mean, gradient_std = mean_std(allowed_gradient_values)
        reset_acc_mean, reset_acc_std = mean_std(reset_acc_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        reset_forget_mean, reset_forget_std = mean_std(reset_forgetting_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<18} "
            f"{allowed_mean:.1f}+/-{allowed_std:.1f}  "
            f"{gradient_mean:.3f}+/-{gradient_std:.3f}  "
            f"{reset_acc_mean:.3f}+/-{reset_acc_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{reset_forget_mean:.4f}+/-{reset_forget_std:.4f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nReadout decomposition summary: mean +/- std")
    print("variant              incoming     readout      readout_g    old_acc      new_acc      forgetting")
    for variant_name in (
        "E_readout_all",
        "E_safe_readout",
        "E_safe_top_readout",
        "E_conflict_readout",
        "E_hybrid",
        "AE_readout_all",
        "AE_safe_readout",
        "AE_safe_top_readout",
        "AE_conflict_readout",
        "AE_hybrid",
    ):
        incoming_values = [
            result["readout_decomposition"][variant_name]["incoming_count"]  # type: ignore[index]
            for result in results
        ]
        readout_values = [
            result["readout_decomposition"][variant_name]["readout_count"]  # type: ignore[index]
            for result in results
        ]
        readout_gradient_values = [
            result["readout_decomposition"][variant_name]["readout_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["readout_decomposition"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["readout_decomposition"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["readout_decomposition"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        incoming_mean, incoming_std = mean_std(incoming_values)
        readout_mean, readout_std = mean_std(readout_values)
        readout_gradient_mean, readout_gradient_std = mean_std(readout_gradient_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<20} "
            f"{incoming_mean:.1f}+/-{incoming_std:.1f}  "
            f"{readout_mean:.1f}+/-{readout_std:.1f}  "
            f"{readout_gradient_mean:.3f}+/-{readout_gradient_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nSoft gradient blending summary: mean +/- std")
    print("variant                    mean_s      eff_g       old_acc      new_acc      forgetting")
    blending_names = tuple(results[0]["blending"].keys())  # type: ignore[union-attr]
    for variant_name in blending_names:
        mean_scale_values = [
            result["blending"][variant_name]["mean_scale"]  # type: ignore[index]
            for result in results
        ]
        effective_gradient_values = [
            result["blending"][variant_name]["effective_new_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["blending"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["blending"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["blending"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        scale_mean, scale_std = mean_std(mean_scale_values)
        gradient_mean, gradient_std = mean_std(effective_gradient_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<26} "
            f"{scale_mean:.3f}+/-{scale_std:.3f}  "
            f"{gradient_mean:.3f}+/-{gradient_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nOnline soft gradient blending summary: mean +/- std")
    print(
        "variant                           avg_s       final_s     avg_eff_g   "
        "final_eff_g old_acc      new_acc      forgetting"
    )
    online_names = tuple(results[0]["online_blending"].keys())  # type: ignore[union-attr]
    for variant_name in online_names:
        avg_scale_values = [
            result["online_blending"][variant_name]["avg_mean_scale"]  # type: ignore[index]
            for result in results
        ]
        final_scale_values = [
            result["online_blending"][variant_name]["final_mean_scale"]  # type: ignore[index]
            for result in results
        ]
        avg_gradient_values = [
            result["online_blending"][variant_name]["avg_effective_new_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        final_gradient_values = [
            result["online_blending"][variant_name]["final_effective_new_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["online_blending"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["online_blending"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["online_blending"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        avg_scale_mean, avg_scale_std = mean_std(avg_scale_values)
        final_scale_mean, final_scale_std = mean_std(final_scale_values)
        avg_gradient_mean, avg_gradient_std = mean_std(avg_gradient_values)
        final_gradient_mean, final_gradient_std = mean_std(final_gradient_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<33} "
            f"{avg_scale_mean:.3f}+/-{avg_scale_std:.3f}  "
            f"{final_scale_mean:.3f}+/-{final_scale_std:.3f}  "
            f"{avg_gradient_mean:.3f}+/-{avg_gradient_std:.3f}  "
            f"{final_gradient_mean:.3f}+/-{final_gradient_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nFamily-level soft blending summary: mean +/- std")
    print("variant       family      mean_syn     mean_s      eff_g       old_acc      new_acc      forgetting")
    family_names = tuple(results[0]["family_blending"].keys())  # type: ignore[union-attr]
    for variant_name in family_names:
        family_count_values = [
            result["family_blending"][variant_name]["family_count"]  # type: ignore[index]
            for result in results
        ]
        synergy_values = [
            result["family_blending"][variant_name]["mean_positive_synergy"]  # type: ignore[index]
            for result in results
        ]
        mean_scale_values = [
            result["family_blending"][variant_name]["mean_scale"]  # type: ignore[index]
            for result in results
        ]
        effective_gradient_values = [
            result["family_blending"][variant_name]["effective_new_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["family_blending"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["family_blending"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["family_blending"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        family_mean, family_std = mean_std(family_count_values)
        synergy_mean, synergy_std = mean_std(synergy_values)
        scale_mean, scale_std = mean_std(mean_scale_values)
        gradient_mean, gradient_std = mean_std(effective_gradient_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<12} "
            f"{family_mean:.1f}+/-{family_std:.1f}  "
            f"{synergy_mean:.5f}+/-{synergy_std:.5f}  "
            f"{scale_mean:.3f}+/-{scale_std:.3f}  "
            f"{gradient_mean:.3f}+/-{gradient_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nPosition-factorized input summary: mean +/- std")
    print("score  conflict     blocked_g    naive_old    naive_new    blend_old    blend_new")
    for score_name in ("E", "AE"):
        conflict_values = [
            result["position_factorization"][score_name]["conflict_count"]  # type: ignore[index]
            for result in results
        ]
        blocked_values = [
            result["position_factorization"][score_name]["blocked_new_signal_fraction"]  # type: ignore[index]
            for result in results
        ]
        naive_old_values = [
            result["position_factorization"][score_name]["naive_old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        naive_new_values = [
            result["position_factorization"][score_name]["naive_new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        blend_old_values = [
            result["position_factorization"][score_name]["blend_old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        blend_new_values = [
            result["position_factorization"][score_name]["blend_new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        conflict_mean, conflict_std = mean_std(conflict_values)
        blocked_mean, blocked_std = mean_std(blocked_values)
        naive_old_mean, naive_old_std = mean_std(naive_old_values)
        naive_new_mean, naive_new_std = mean_std(naive_new_values)
        blend_old_mean, blend_old_std = mean_std(blend_old_values)
        blend_new_mean, blend_new_std = mean_std(blend_new_values)
        print(
            f"{score_name:<5} "
            f"{conflict_mean:.1f}+/-{conflict_std:.1f}  "
            f"{blocked_mean:.3f}+/-{blocked_std:.3f}  "
            f"{naive_old_mean:.3f}+/-{naive_old_std:.3f}  "
            f"{naive_new_mean:.3f}+/-{naive_new_std:.3f}  "
            f"{blend_old_mean:.3f}+/-{blend_old_std:.3f}  "
            f"{blend_new_mean:.3f}+/-{blend_new_std:.3f}"
        )

    print("\nFactorized architecture summary: mean +/- std")
    print(
        "metric                    entangled        factorized"
    )
    factorized_pairs = (
        (
            "old_acc",
            "entangled_old_accuracy",
            "factorized_old_accuracy",
        ),
        (
            "new_acc",
            "entangled_new_accuracy",
            "factorized_new_accuracy",
        ),
        (
            "forgetting",
            "entangled_forgetting",
            "factorized_forgetting",
        ),
    )
    for label, entangled_key, factorized_key in factorized_pairs:
        entangled_values = [
            result["factorized_architecture"][entangled_key]  # type: ignore[index]
            for result in results
        ]
        factorized_values = [
            result["factorized_architecture"][factorized_key]  # type: ignore[index]
            for result in results
        ]
        entangled_mean, entangled_std = mean_std(entangled_values)
        factorized_mean, factorized_std = mean_std(factorized_values)
        print(
            f"{label:<25} "
            f"{entangled_mean:.3f}+/-{entangled_std:.3f}  "
            f"{factorized_mean:.3f}+/-{factorized_std:.3f}"
        )

    blocked_values = [
        result["factorized_architecture"]["entangled_blocked_new_signal_fraction"]  # type: ignore[index]
        for result in results
    ]
    old_router_values = [
        result["factorized_architecture"]["before_old_router_gradient_fraction"]  # type: ignore[index]
        for result in results
    ]
    shared_before_values = [
        result["factorized_architecture"]["before_shared_gradient_fraction"]  # type: ignore[index]
        for result in results
    ]
    shared_after_values = [
        result["factorized_architecture"]["after_shared_gradient_fraction"]  # type: ignore[index]
        for result in results
    ]
    cka_values = [
        result["factorized_architecture"]["op_representation_cka"]  # type: ignore[index]
        for result in results
    ]
    blocked_mean, blocked_std = mean_std(blocked_values)
    old_router_mean, old_router_std = mean_std(old_router_values)
    shared_before_mean, shared_before_std = mean_std(shared_before_values)
    shared_after_mean, shared_after_std = mean_std(shared_after_values)
    cka_mean, cka_std = mean_std(cka_values)
    print(
        f"{'entangled_blocked_g':<25} "
        f"{blocked_mean:.3f}+/-{blocked_std:.3f}  {'n/a':>13}"
    )
    print(
        f"{'factorized_old_router_g':<25} "
        f"{'n/a':>13}  {old_router_mean:.3f}+/-{old_router_std:.3f}"
    )
    print(
        f"{'factorized_shared_g_pre':<25} "
        f"{'n/a':>13}  {shared_before_mean:.3f}+/-{shared_before_std:.3f}"
    )
    print(
        f"{'factorized_shared_g_post':<25} "
        f"{'n/a':>13}  {shared_after_mean:.3f}+/-{shared_after_std:.3f}"
    )
    print(
        f"{'factorized_op_cka':<25} "
        f"{'n/a':>13}  {cka_mean:.3f}+/-{cka_std:.3f}"
    )

    print("\nFactorized representation-alignment summary: mean +/- std")
    print("variant        old_acc      new_acc      shared_g     op_mse      op_pair_cos op_class_cos paired_cka")
    alignment_names = tuple(results[0]["factorized_alignment"].keys())  # type: ignore[union-attr]
    for variant_name in alignment_names:
        old_acc_values = [
            result["factorized_alignment"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["factorized_alignment"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        shared_gradient_values = [
            result["factorized_alignment"][variant_name]["after_shared_gradient_fraction"]  # type: ignore[index]
            for result in results
        ]
        op_mse_values = [
            result["factorized_alignment"][variant_name]["op_pair_mse"]  # type: ignore[index]
            for result in results
        ]
        op_pair_cos_values = [
            result["factorized_alignment"][variant_name]["op_pair_cosine"]  # type: ignore[index]
            for result in results
        ]
        op_class_cos_values = [
            result["factorized_alignment"][variant_name]["op_class_cosine"]  # type: ignore[index]
            for result in results
        ]
        paired_cka_values = [
            result["factorized_alignment"][variant_name]["paired_op_cka"]  # type: ignore[index]
            for result in results
        ]
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        shared_mean, shared_std = mean_std(shared_gradient_values)
        mse_mean, mse_std = mean_std(op_mse_values)
        pair_mean, pair_std = mean_std(op_pair_cos_values)
        class_mean, class_std = mean_std(op_class_cos_values)
        cka_mean, cka_std = mean_std(paired_cka_values)
        print(
            f"{variant_name:<14} "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{shared_mean:.3f}+/-{shared_std:.3f}  "
            f"{mse_mean:.4f}+/-{mse_std:.4f}  "
            f"{pair_mean:.3f}+/-{pair_std:.3f}  "
            f"{class_mean:.3f}+/-{class_std:.3f}  "
            f"{cka_mean:.3f}+/-{cka_std:.3f}"
        )

    print("\nFactorized shared-gradient compatibility summary: mean +/- std")
    print("variant        grad_cos     old_tan_raw      old_tan_unit     new_g_norm")
    for variant_name in ("router_only", "route_w_10", "op_w_10"):
        grad_cos_values = [
            result["factorized_alignment"][variant_name]["shared_grad_cosine"]  # type: ignore[index]
            for result in results
        ]
        raw_tangent_values = [
            result["factorized_alignment"][variant_name]["old_tangent_damage_raw"]  # type: ignore[index]
            for result in results
        ]
        unit_tangent_values = [
            result["factorized_alignment"][variant_name]["old_tangent_damage_unit"]  # type: ignore[index]
            for result in results
        ]
        new_norm_values = [
            result["factorized_alignment"][variant_name]["new_shared_gradient_norm"]  # type: ignore[index]
            for result in results
        ]
        grad_cos_mean, grad_cos_std = mean_std(grad_cos_values)
        raw_mean, raw_std = mean_std(raw_tangent_values)
        unit_mean, unit_std = mean_std(unit_tangent_values)
        norm_mean, norm_std = mean_std(new_norm_values)
        print(
            f"{variant_name:<14} "
            f"{grad_cos_mean:.3f}+/-{grad_cos_std:.3f}  "
            f"{raw_mean:.6f}+/-{raw_std:.6f}  "
            f"{unit_mean:.6f}+/-{unit_std:.6f}  "
            f"{norm_mean:.6f}+/-{norm_std:.6f}"
        )

    print("\nFactorized shared-consolidation gate summary: mean +/- std")
    print(
        "objective          commit      reject      old_acc      new_acc      "
        "shared_drop     final_norm     op_cos      paired_cka"
    )
    consolidation_names = tuple(results[0]["factorized_consolidation"].keys())  # type: ignore[union-attr]
    for objective_name in consolidation_names:
        committed_values = [
            result["factorized_consolidation"][objective_name]["committed_steps"]  # type: ignore[index]
            for result in results
        ]
        rejected_values = [
            result["factorized_consolidation"][objective_name]["rejected_steps"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["factorized_consolidation"][objective_name]["final_old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["factorized_consolidation"][objective_name]["final_new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        shared_drop_values = [
            result["factorized_consolidation"][objective_name]["shared_gradient_norm_drop"]  # type: ignore[index]
            for result in results
        ]
        final_norm_values = [
            result["factorized_consolidation"][objective_name]["final_shared_gradient_norm"]  # type: ignore[index]
            for result in results
        ]
        op_cos_values = [
            result["factorized_consolidation"][objective_name]["final_op_pair_cosine"]  # type: ignore[index]
            for result in results
        ]
        paired_cka_values = [
            result["factorized_consolidation"][objective_name]["final_paired_op_cka"]  # type: ignore[index]
            for result in results
        ]
        committed_mean, committed_std = mean_std(committed_values)
        rejected_mean, rejected_std = mean_std(rejected_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        drop_mean, drop_std = mean_std(shared_drop_values)
        norm_mean, norm_std = mean_std(final_norm_values)
        cos_mean, cos_std = mean_std(op_cos_values)
        cka_mean, cka_std = mean_std(paired_cka_values)
        print(
            f"{objective_name:<18} "
            f"{committed_mean:.1f}+/-{committed_std:.1f}  "
            f"{rejected_mean:.1f}+/-{rejected_std:.1f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{drop_mean:.6f}+/-{drop_std:.6f}  "
            f"{norm_mean:.6f}+/-{norm_std:.6f}  "
            f"{cos_mean:.3f}+/-{cos_std:.3f}  "
            f"{cka_mean:.3f}+/-{cka_std:.3f}"
        )

    print("\nFactorized consolidation ablation summary: mean +/- std")
    print(
        "variant            commit      stop        old_acc      new_acc      "
        "shared_drop    op_cos      paired_cka   acc_g_delta  stop_g_delta"
    )
    ablation_names = tuple(results[0]["factorized_consolidation_ablation"].keys())  # type: ignore[union-attr]
    for variant_name in ablation_names:
        committed_values = [
            result["factorized_consolidation_ablation"][variant_name]["committed_steps"]  # type: ignore[index]
            for result in results
        ]
        stop_values = [
            result["factorized_consolidation_ablation"][variant_name]["rejected_steps"]  # type: ignore[index]
            + result["factorized_consolidation_ablation"][variant_name]["stopped_steps"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["factorized_consolidation_ablation"][variant_name]["final_old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["factorized_consolidation_ablation"][variant_name]["final_new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        shared_drop_values = [
            result["factorized_consolidation_ablation"][variant_name]["shared_gradient_norm_drop"]  # type: ignore[index]
            for result in results
        ]
        op_cos_values = [
            result["factorized_consolidation_ablation"][variant_name]["final_op_pair_cosine"]  # type: ignore[index]
            for result in results
        ]
        paired_cka_values = [
            result["factorized_consolidation_ablation"][variant_name]["final_paired_op_cka"]  # type: ignore[index]
            for result in results
        ]
        accepted_delta_values = [
            result["factorized_consolidation_ablation"][variant_name][
                "accepted_mean_shared_gradient_norm_delta"
            ]  # type: ignore[index]
            for result in results
        ]
        failed_delta_values = [
            result["factorized_consolidation_ablation"][variant_name][
                "failed_shared_gradient_norm_delta"
            ]  # type: ignore[index]
            for result in results
        ]
        committed_mean, committed_std = mean_std(committed_values)
        stop_mean, stop_std = mean_std(stop_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        drop_mean, drop_std = mean_std(shared_drop_values)
        cos_mean, cos_std = mean_std(op_cos_values)
        cka_mean, cka_std = mean_std(paired_cka_values)
        accepted_delta_mean, accepted_delta_std, _ = mean_std_finite(accepted_delta_values)
        failed_delta_mean, failed_delta_std, failed_delta_count = mean_std_finite(
            failed_delta_values
        )
        failed_delta_text = (
            f"{failed_delta_mean:.6f}+/-{failed_delta_std:.6f}"
            if failed_delta_count > 0
            else "none"
        )
        print(
            f"{variant_name:<18} "
            f"{committed_mean:.1f}+/-{committed_std:.1f}  "
            f"{stop_mean:.1f}+/-{stop_std:.1f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{drop_mean:.6f}+/-{drop_std:.6f}  "
            f"{cos_mean:.3f}+/-{cos_std:.3f}  "
            f"{cka_mean:.3f}+/-{cka_std:.3f}  "
            f"{accepted_delta_mean:.6f}+/-{accepted_delta_std:.6f}  "
            f"{failed_delta_text}"
        )

    print("\nMeaning transformation summary: mean +/- std")
    print(
        "variant              conflict     align       proj_r      trans_old   "
        "old_acc      new_acc      forgetting"
    )
    meaning_names = tuple(results[0]["meaning_transform"].keys())  # type: ignore[union-attr]
    for variant_name in meaning_names:
        conflict_values = [
            result["meaning_transform"][variant_name]["conflict_count"]  # type: ignore[index]
            for result in results
        ]
        alignment_values = [
            result["meaning_transform"][variant_name]["mean_alignment"]  # type: ignore[index]
            for result in results
        ]
        projection_values = [
            result["meaning_transform"][variant_name]["mean_projection_ratio"]  # type: ignore[index]
            for result in results
        ]
        transform_old_values = [
            result["meaning_transform"][variant_name]["transform_old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["meaning_transform"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["meaning_transform"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["meaning_transform"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        conflict_mean, conflict_std = mean_std(conflict_values)
        alignment_mean, alignment_std = mean_std(alignment_values)
        projection_mean, projection_std = mean_std(projection_values)
        transform_old_mean, transform_old_std = mean_std(transform_old_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<20} "
            f"{conflict_mean:.1f}+/-{conflict_std:.1f}  "
            f"{alignment_mean:.3f}+/-{alignment_std:.3f}  "
            f"{projection_mean:.3f}+/-{projection_std:.3f}  "
            f"{transform_old_mean:.3f}+/-{transform_old_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )

    print("\nFunctional transformation summary: mean +/- std")
    print(
        "variant        conflict     old_mse       new_mse       trans_old   "
        "old_acc      new_acc      forgetting"
    )
    functional_names = tuple(results[0]["functional_transform"].keys())  # type: ignore[union-attr]
    for variant_name in functional_names:
        conflict_values = [
            result["functional_transform"][variant_name]["conflict_count"]  # type: ignore[index]
            for result in results
        ]
        old_mse_values = [
            result["functional_transform"][variant_name]["old_activation_mse"]  # type: ignore[index]
            for result in results
        ]
        new_mse_values = [
            result["functional_transform"][variant_name]["new_activation_mse"]  # type: ignore[index]
            for result in results
        ]
        transform_old_values = [
            result["functional_transform"][variant_name]["transform_old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        old_acc_values = [
            result["functional_transform"][variant_name]["old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_acc_values = [
            result["functional_transform"][variant_name]["new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        forgetting_values = [
            result["functional_transform"][variant_name]["forgetting"]  # type: ignore[index]
            for result in results
        ]
        conflict_mean, conflict_std = mean_std(conflict_values)
        old_mse_mean, old_mse_std = mean_std(old_mse_values)
        new_mse_mean, new_mse_std = mean_std(new_mse_values)
        transform_old_mean, transform_old_std = mean_std(transform_old_values)
        old_mean, old_std = mean_std(old_acc_values)
        new_mean, new_std = mean_std(new_acc_values)
        forget_mean, forget_std = mean_std(forgetting_values)
        print(
            f"{variant_name:<14} "
            f"{conflict_mean:.1f}+/-{conflict_std:.1f}  "
            f"{old_mse_mean:.5f}+/-{old_mse_std:.5f}  "
            f"{new_mse_mean:.5f}+/-{new_mse_std:.5f}  "
            f"{transform_old_mean:.3f}+/-{transform_old_std:.3f}  "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{forget_mean:.4f}+/-{forget_std:.4f}"
        )


def run_usage_probe() -> None:
    _ = run_seed(seed=7, verbose=True)


def run_multi_seed(seed_count: int) -> None:
    results = []
    for seed in range(seed_count):
        print(f"running_seed={seed}")
        results.append(run_seed(seed=seed, verbose=False))
    print_multi_seed_summary(results)


def print_factorized_stress_summary(results: list[dict[str, object]]) -> None:
    print("\nFactorized consolidation stress summary: mean +/- std")
    print(f"seeds={[result['seed'] for result in results]}")
    print(
        "setting                    method  commit      event_rate  event_step  "
        "old_acc      new_acc      shared_drop    final_norm    op_cos      paired_cka"
    )
    stress_names = tuple(results[0]["stress"].keys())  # type: ignore[union-attr]
    for setting_name in stress_names:
        method_id = results[0]["stress"][setting_name]["method_id"]  # type: ignore[index]
        method = "gate" if method_id == 1.0 else "naive"
        committed_values = [
            result["stress"][setting_name]["committed_steps"]  # type: ignore[index]
            for result in results
        ]
        event_values = [
            result["stress"][setting_name]["event_step"]  # type: ignore[index]
            for result in results
        ]
        old_values = [
            result["stress"][setting_name]["final_old_accuracy"]  # type: ignore[index]
            for result in results
        ]
        new_values = [
            result["stress"][setting_name]["final_new_accuracy"]  # type: ignore[index]
            for result in results
        ]
        drop_values = [
            result["stress"][setting_name]["shared_gradient_norm_drop"]  # type: ignore[index]
            for result in results
        ]
        norm_values = [
            result["stress"][setting_name]["final_shared_gradient_norm"]  # type: ignore[index]
            for result in results
        ]
        cos_values = [
            result["stress"][setting_name]["final_op_pair_cosine"]  # type: ignore[index]
            for result in results
        ]
        cka_values = [
            result["stress"][setting_name]["final_paired_op_cka"]  # type: ignore[index]
            for result in results
        ]
        committed_mean, committed_std = mean_std(committed_values)
        event_mean, event_std, event_count = mean_std_finite(event_values)
        event_rate = event_count / len(results)
        event_text = f"{event_mean:.1f}+/-{event_std:.1f}" if event_count > 0 else "none"
        old_mean, old_std = mean_std(old_values)
        new_mean, new_std = mean_std(new_values)
        drop_mean, drop_std = mean_std(drop_values)
        norm_mean, norm_std = mean_std(norm_values)
        cos_mean, cos_std = mean_std(cos_values)
        cka_mean, cka_std = mean_std(cka_values)
        print(
            f"{setting_name:<26} "
            f"{method:<6} "
            f"{committed_mean:.1f}+/-{committed_std:.1f}  "
            f"{event_rate:.2f}        "
            f"{event_text:<13} "
            f"{old_mean:.3f}+/-{old_std:.3f}  "
            f"{new_mean:.3f}+/-{new_std:.3f}  "
            f"{drop_mean:.6f}+/-{drop_std:.6f}  "
            f"{norm_mean:.6f}+/-{norm_std:.6f}  "
            f"{cos_mean:.3f}+/-{cos_std:.3f}  "
            f"{cka_mean:.3f}+/-{cka_std:.3f}"
        )


def run_factorized_stress_multi_seed(
    seed_count: int,
    hidden_dims: tuple[int, ...],
    learning_rates: tuple[float, ...],
    step_counts: tuple[int, ...],
) -> None:
    results: list[dict[str, object]] = []
    for seed in range(seed_count):
        print(f"running_seed={seed}")
        stress = factorized_consolidation_stress_metrics(
            seed,
            hidden_dims,
            learning_rates,
            step_counts,
            verbose=False,
        )
        results.append({"seed": seed, "stress": stress})
    print_factorized_stress_summary(results)


def parse_int_tuple(raw: str, name: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            raise ValueError(f"{name} contains an empty item: {raw!r}.")
        value = int(text)
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}.")
        values.append(value)
    return tuple(values)


def parse_float_tuple(raw: str, name: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            raise ValueError(f"{name} contains an empty item: {raw!r}.")
        value = float(text)
        if value <= 0.0:
            raise ValueError(f"{name} values must be positive, got {value}.")
        values.append(value)
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--factorized-stress", action="store_true")
    parser.add_argument(
        "--stress-hidden-dims",
        default=",".join(str(value) for value in DEFAULT_STRESS_HIDDEN_DIMS),
    )
    parser.add_argument(
        "--stress-learning-rates",
        default=",".join(str(value) for value in DEFAULT_STRESS_LEARNING_RATES),
    )
    parser.add_argument(
        "--stress-steps",
        default=",".join(str(value) for value in DEFAULT_STRESS_STEPS),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.factorized_stress:
        stress_hidden_dims = parse_int_tuple(args.stress_hidden_dims, "stress-hidden-dims")
        stress_learning_rates = parse_float_tuple(
            args.stress_learning_rates,
            "stress-learning-rates",
        )
        stress_steps = parse_int_tuple(args.stress_steps, "stress-steps")
        if args.multi_seed:
            run_factorized_stress_multi_seed(
                args.seed_count,
                stress_hidden_dims,
                stress_learning_rates,
                stress_steps,
            )
        else:
            run_factorized_consolidation_stress_table(
                seed=7,
                hidden_dims=stress_hidden_dims,
                learning_rates=stress_learning_rates,
                step_counts=stress_steps,
                verbose=True,
            )
    elif args.multi_seed:
        run_multi_seed(args.seed_count)
    else:
        run_usage_probe()
