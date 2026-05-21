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
DEFAULT_MULTI_ROUTE_NON_ANALOG_SPECS = (
    MultiRouteOpSpec("MAX12", "max", (1, 2)),
    MultiRouteOpSpec("COPY2", "copy", (2,)),
)
DEFAULT_MULTI_ROUTE_COMPOSITION_SPECS = (
    MultiRouteOpSpec("SUM012", "add", (0, 1, 2)),
    MultiRouteOpSpec("MAX_ADD01_2", "max_add", (0, 1, 2)),
    MultiRouteOpSpec("ADD_MAX01_2", "add_max", (0, 1, 2)),
)
DEFAULT_MULTI_ROUTE_BASE_NAME = "ADD01"
DEFAULT_MULTI_ROUTE_ALIGNMENT_WEIGHT = 10.0
COMPOSITION_BENCHMARK_POLICIES = ("admission", "force_class_align")
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
        if spec.kind not in ("add", "copy", "max", "max_add", "add_max"):
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
        if spec.kind in ("max_add", "add_max") and len(spec.operands) != 3:
            raise ValueError(f"{spec.kind} op {spec.name} must have exactly three operands.")


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
    if spec.kind == "max_add":
        first, second, third = values
        return int(max((first + second) % num_digits, third))
    if spec.kind == "add_max":
        first, second, third = values
        return int((max(first, second) + third) % num_digits)
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


def validate_multi_route_model(model: MultiRouteFactorizedMLP) -> None:
    if set(model.W_router) != set(model.b_router):
        raise ValueError(
            f"router weight names {sorted(model.W_router)} do not match "
            f"router bias names {sorted(model.b_router)}."
        )
    if not model.W_router:
        raise ValueError("multi-route model has no routers.")
    hidden_dim = model.W_op.shape[0]
    if model.W_op.shape != (hidden_dim, hidden_dim):
        raise ValueError(f"W_op must be square, got {model.W_op.shape}.")
    if model.b_op.shape != (hidden_dim,):
        raise ValueError(f"b_op shape {model.b_op.shape} does not match hidden_dim={hidden_dim}.")
    if model.W_out.shape[0] != hidden_dim:
        raise ValueError(
            f"W_out input dimension {model.W_out.shape[0]} does not match hidden_dim={hidden_dim}."
        )
    for name, W_router in model.W_router.items():
        b_router = model.b_router[name]
        if W_router.ndim != 2:
            raise ValueError(f"router {name} W must be 2D, got {W_router.shape}.")
        if W_router.shape[1] != hidden_dim:
            raise ValueError(
                f"router {name} hidden dimension {W_router.shape[1]} "
                f"does not match shared hidden_dim={hidden_dim}."
            )
        if b_router.shape != (hidden_dim,):
            raise ValueError(
                f"router {name} bias shape {b_router.shape} does not match hidden_dim={hidden_dim}."
            )


def multi_route_forward(
    model: MultiRouteFactorizedMLP,
    digit_x: np.ndarray,
    route_name: str,
) -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
    validate_multi_route_model(model)
    if route_name not in model.W_router:
        raise ValueError(f"unknown route {route_name!r}; available={sorted(model.W_router)}.")
    W_router = model.W_router[route_name]
    b_router = model.b_router[route_name]
    if digit_x.ndim != 2 or digit_x.shape[1] != W_router.shape[0]:
        raise ValueError(
            f"digit_x shape {digit_x.shape} does not match router input dimension "
            f"{W_router.shape[0]} for route {route_name}."
        )
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
        "route_name": route_name,
    }


def empty_multi_route_grads(model: MultiRouteFactorizedMLP) -> MultiRouteGrads:
    validate_multi_route_model(model)
    return MultiRouteGrads(
        W_router={name: np.zeros_like(value) for name, value in model.W_router.items()},
        b_router={name: np.zeros_like(value) for name, value in model.b_router.items()},
        W_op=np.zeros_like(model.W_op),
        b_op=np.zeros_like(model.b_op),
        W_out=np.zeros_like(model.W_out),
        b_out=np.zeros_like(model.b_out),
    )


def multi_route_backward(
    model: MultiRouteFactorizedMLP,
    cache: dict[str, np.ndarray | str],
    dlogits: np.ndarray,
) -> MultiRouteGrads:
    route_name_raw = cache["route_name"]
    if not isinstance(route_name_raw, str):
        raise ValueError("multi-route cache route_name is not a string.")
    route_name = route_name_raw
    if route_name not in model.W_router:
        raise ValueError(f"unknown route {route_name!r}; available={sorted(model.W_router)}.")

    op_h = cache["op_h"]
    op_z = cache["op_z"]
    route_h = cache["route_h"]
    route_z = cache["route_z"]
    digit_x = cache["digit_x"]
    if not all(isinstance(value, np.ndarray) for value in (op_h, op_z, route_h, route_z, digit_x)):
        raise ValueError("multi-route cache contains non-array activation values.")
    op_h = np.asarray(op_h)
    op_z = np.asarray(op_z)
    route_h = np.asarray(route_h)
    route_z = np.asarray(route_z)
    digit_x = np.asarray(digit_x)

    grads = empty_multi_route_grads(model)
    grads.W_out = op_h.T @ dlogits
    grads.b_out = np.sum(dlogits, axis=0)
    dop_h = dlogits @ model.W_out.T
    dop_z = dop_h * (op_z > 0.0)
    grads.W_op = route_h.T @ dop_z
    grads.b_op = np.sum(dop_z, axis=0)
    droute_h = dop_z @ model.W_op.T
    droute_z = droute_h * (route_z > 0.0)
    grads.W_router[route_name] = digit_x.T @ droute_z
    grads.b_router[route_name] = np.sum(droute_z, axis=0)
    return grads


def multi_route_add_scaled_grads(
    target: MultiRouteGrads,
    source: MultiRouteGrads,
    scale: float,
) -> None:
    if set(target.W_router) != set(source.W_router) or set(target.b_router) != set(source.b_router):
        raise ValueError("multi-route gradient router keys do not match.")
    for name in target.W_router:
        if target.W_router[name].shape != source.W_router[name].shape:
            raise ValueError(f"W_router gradient shape mismatch for route {name}.")
        target.W_router[name] += scale * source.W_router[name]
        target.b_router[name] += scale * source.b_router[name]
    if target.W_op.shape != source.W_op.shape:
        raise ValueError("W_op gradient shapes do not match.")
    target.W_op += scale * source.W_op
    target.b_op += scale * source.b_op
    target.W_out += scale * source.W_out
    target.b_out += scale * source.b_out


def multi_route_apply_update(
    model: MultiRouteFactorizedMLP,
    grads: MultiRouteGrads,
    learning_rate: float,
    trainable_routes: tuple[str, ...],
    train_shared: bool,
) -> None:
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
    unknown_routes = set(trainable_routes) - set(model.W_router)
    if unknown_routes:
        raise ValueError(f"unknown trainable routes: {sorted(unknown_routes)}.")
    if set(grads.W_router) != set(model.W_router) or set(grads.b_router) != set(model.b_router):
        raise ValueError("multi-route gradient keys do not match model routes.")
    for route_name in trainable_routes:
        model.W_router[route_name] -= learning_rate * grads.W_router[route_name]
        model.b_router[route_name] -= learning_rate * grads.b_router[route_name]
    if train_shared:
        model.W_op -= learning_rate * grads.W_op
        model.b_op -= learning_rate * grads.b_op
        model.W_out -= learning_rate * grads.W_out
        model.b_out -= learning_rate * grads.b_out


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


def multi_route_evaluate(
    model: MultiRouteFactorizedMLP,
    dataset: MultiRouteDataset,
) -> tuple[float, float]:
    logits, _ = multi_route_forward(model, dataset.digit_x, dataset.op_spec.name)
    loss, _ = loss_and_grad_logits(logits, dataset.y)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == dataset.y))
    return loss, accuracy


def multi_route_loss_and_grads(
    model: MultiRouteFactorizedMLP,
    dataset: MultiRouteDataset,
) -> tuple[float, MultiRouteGrads]:
    logits, cache = multi_route_forward(model, dataset.digit_x, dataset.op_spec.name)
    loss, dlogits = loss_and_grad_logits(logits, dataset.y)
    return loss, multi_route_backward(model, cache, dlogits)


def train_multi_route(
    model: MultiRouteFactorizedMLP,
    dataset: MultiRouteDataset,
    train_config: TrainConfig,
    label: str,
    trainable_routes: tuple[str, ...],
    train_shared: bool,
) -> None:
    if not trainable_routes and not train_shared:
        raise ValueError("train_multi_route received no trainable parameters.")
    for epoch in range(1, train_config.epochs + 1):
        loss, grads = multi_route_loss_and_grads(model, dataset)
        multi_route_apply_update(
            model,
            grads,
            train_config.learning_rate,
            trainable_routes,
            train_shared,
        )
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = multi_route_evaluate(model, dataset)
            print(f"{label} epoch={epoch:04d} loss={loss:.4f} accuracy={accuracy:.3f}")


def multi_route_alignment_loss_and_grads(
    model: MultiRouteFactorizedMLP,
    target_dataset: MultiRouteDataset,
    analog_source_dataset: MultiRouteDataset,
    alignment_weight: float,
) -> tuple[float, float, float, MultiRouteGrads]:
    if alignment_weight <= 0.0:
        raise ValueError(f"alignment_weight must be positive, got {alignment_weight}.")
    target_name = target_dataset.op_spec.name
    source_name = analog_source_dataset.op_spec.name
    if target_name == source_name:
        raise ValueError("target and source routes must be different for alignment training.")
    if not np.array_equal(target_dataset.y, analog_source_dataset.y):
        raise ValueError("target and analog source targets must match row-by-row.")

    logits, target_cache = multi_route_forward(model, target_dataset.digit_x, target_name)
    _, source_cache = multi_route_forward(model, analog_source_dataset.digit_x, source_name)
    ce_loss, dlogits = loss_and_grad_logits(logits, target_dataset.y)
    grads = multi_route_backward(model, target_cache, dlogits)

    target_op = target_cache["op_h"]
    source_op = source_cache["op_h"]
    target_op_z = target_cache["op_z"]
    target_route_z = target_cache["route_z"]
    target_digit_x = target_cache["digit_x"]
    if not all(
        isinstance(value, np.ndarray)
        for value in (target_op, source_op, target_op_z, target_route_z, target_digit_x)
    ):
        raise ValueError("alignment cache contains non-array values.")
    target_op = np.asarray(target_op)
    source_op = np.asarray(source_op)
    target_op_z = np.asarray(target_op_z)
    target_route_z = np.asarray(target_route_z)
    target_digit_x = np.asarray(target_digit_x)

    diff = target_op - source_op
    alignment_loss = float(np.mean(diff**2))
    dop_h = alignment_weight * 2.0 * diff / diff.size
    dop_z = dop_h * (target_op_z > 0.0)
    droute_h = dop_z @ model.W_op.T
    droute_z = droute_h * (target_route_z > 0.0)
    grads.W_router[target_name] += target_digit_x.T @ droute_z
    grads.b_router[target_name] += np.sum(droute_z, axis=0)
    total_loss = ce_loss + alignment_weight * alignment_loss
    return total_loss, ce_loss, alignment_loss, grads


def train_multi_route_alignment(
    model: MultiRouteFactorizedMLP,
    target_dataset: MultiRouteDataset,
    analog_source_dataset: MultiRouteDataset,
    train_config: TrainConfig,
    label: str,
    alignment_weight: float,
) -> None:
    target_name = target_dataset.op_spec.name
    for epoch in range(1, train_config.epochs + 1):
        loss, ce_loss, alignment_loss, grads = multi_route_alignment_loss_and_grads(
            model,
            target_dataset,
            analog_source_dataset,
            alignment_weight,
        )
        multi_route_apply_update(
            model,
            grads,
            train_config.learning_rate,
            (target_name,),
            train_shared=False,
        )
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = multi_route_evaluate(model, target_dataset)
            print(
                f"{label} epoch={epoch:04d} loss={loss:.4f} "
                f"ce={ce_loss:.4f} align={alignment_loss:.4f} accuracy={accuracy:.3f}"
            )


def multi_route_pair_alignment_metrics(
    model: MultiRouteFactorizedMLP,
    target_dataset: MultiRouteDataset,
    analog_source_dataset: MultiRouteDataset,
    num_digits: int,
) -> dict[str, float]:
    if not np.array_equal(target_dataset.y, analog_source_dataset.y):
        raise ValueError("target and analog source targets must match row-by-row.")
    _, target_cache = multi_route_forward(
        model,
        target_dataset.digit_x,
        target_dataset.op_spec.name,
    )
    _, source_cache = multi_route_forward(
        model,
        analog_source_dataset.digit_x,
        analog_source_dataset.op_spec.name,
    )
    target_route = target_cache["route_h"]
    source_route = source_cache["route_h"]
    target_op = target_cache["op_h"]
    source_op = source_cache["op_h"]
    if not all(isinstance(value, np.ndarray) for value in (target_route, source_route, target_op, source_op)):
        raise ValueError("pair alignment cache contains non-array values.")
    target_route = np.asarray(target_route)
    source_route = np.asarray(source_route)
    target_op = np.asarray(target_op)
    source_op = np.asarray(source_op)
    return {
        "route_pair_mse": float(np.mean((target_route - source_route) ** 2)),
        "op_pair_mse": float(np.mean((target_op - source_op) ** 2)),
        "route_pair_cosine": mean_row_cosine(target_route, source_route),
        "op_pair_cosine": mean_row_cosine(target_op, source_op),
        "route_class_cosine": class_center_cosine(
            target_route,
            source_route,
            target_dataset.y,
            num_digits,
        ),
        "op_class_cosine": class_center_cosine(
            target_op,
            source_op,
            target_dataset.y,
            num_digits,
        ),
        "paired_op_cka": centered_linear_cka(target_op, source_op),
    }


def class_centers(
    activations: np.ndarray,
    labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got {activations.shape}.")
    if labels.shape != (activations.shape[0],):
        raise ValueError(
            f"labels shape {labels.shape} does not match row count {activations.shape[0]}."
        )
    if class_count <= 0:
        raise ValueError(f"class_count must be positive, got {class_count}.")
    centers = np.zeros((class_count, activations.shape[1]))
    for class_id in range(class_count):
        mask = labels == class_id
        if not np.any(mask):
            raise ValueError(f"class {class_id} has no examples.")
        centers[class_id] = np.mean(activations[mask], axis=0)
    return centers


def mean_center_cosine(left_centers: np.ndarray, right_centers: np.ndarray) -> float:
    if left_centers.shape != right_centers.shape:
        raise ValueError(
            f"center shapes must match, got {left_centers.shape} and {right_centers.shape}."
        )
    return float(
        np.mean(
            [
                cosine(left_centers[class_id], right_centers[class_id])
                for class_id in range(left_centers.shape[0])
            ]
        )
    )


def multi_route_op_cache(
    model: MultiRouteFactorizedMLP,
    dataset: MultiRouteDataset,
) -> dict[str, np.ndarray | str]:
    _, cache = multi_route_forward(model, dataset.digit_x, dataset.op_spec.name)
    return cache


def multi_route_op_h(
    model: MultiRouteFactorizedMLP,
    dataset: MultiRouteDataset,
) -> np.ndarray:
    cache = multi_route_op_cache(model, dataset)
    op_h = cache["op_h"]
    if not isinstance(op_h, np.ndarray):
        raise ValueError("multi-route op_h cache value is not an array.")
    return op_h


def multi_route_class_center_alignment_metrics(
    model: MultiRouteFactorizedMLP,
    target_dataset: MultiRouteDataset,
    source_dataset: MultiRouteDataset,
    class_count: int,
) -> dict[str, float]:
    target_op = multi_route_op_h(model, target_dataset)
    source_op = multi_route_op_h(model, source_dataset)
    target_centers = class_centers(target_op, target_dataset.y, class_count)
    source_centers = class_centers(source_op, source_dataset.y, class_count)
    return {
        "center_op_cosine": mean_center_cosine(target_centers, source_centers),
        "center_op_mse": float(np.mean((target_centers - source_centers) ** 2)),
        "center_op_cka": centered_linear_cka(target_centers, source_centers),
    }


def logits_from_multi_route_op_h(
    model: MultiRouteFactorizedMLP,
    op_h: np.ndarray,
) -> np.ndarray:
    if op_h.ndim != 2:
        raise ValueError(f"op_h must be 2D, got shape {op_h.shape}.")
    if op_h.shape[1] != model.W_out.shape[0]:
        raise ValueError(
            f"op_h dimension {op_h.shape[1]} does not match W_out input "
            f"dimension {model.W_out.shape[0]}."
        )
    return op_h @ model.W_out + model.b_out


def evaluate_logits(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits.shape}.")
    if targets.shape != (logits.shape[0],):
        raise ValueError(
            f"target shape {targets.shape} does not match logits row count {logits.shape[0]}."
        )
    loss, _ = loss_and_grad_logits(logits, targets)
    accuracy = float(np.mean(np.argmax(logits, axis=1) == targets))
    return loss, accuracy


def row_span_basis(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}.")
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > SCORE_EPSILON))
    if rank <= 0:
        raise ValueError("matrix row span has zero rank.")
    return vh[:rank].T


def multi_route_output_code_causal_metrics(
    model: MultiRouteFactorizedMLP,
    target_dataset: MultiRouteDataset,
    source_dataset: MultiRouteDataset,
    class_count: int,
) -> dict[str, float]:
    target_op = multi_route_op_h(model, target_dataset)
    source_op = multi_route_op_h(model, source_dataset)
    source_centers = class_centers(source_op, source_dataset.y, class_count)
    center_by_label = source_centers[target_dataset.y]
    basis = row_span_basis(source_centers)
    target_projection = target_op @ basis @ basis.T

    normal_loss, normal_accuracy = evaluate_logits(
        logits_from_multi_route_op_h(model, target_op),
        target_dataset.y,
    )
    center_patch_loss, center_patch_accuracy = evaluate_logits(
        logits_from_multi_route_op_h(model, center_by_label),
        target_dataset.y,
    )
    residual_only_loss, residual_only_accuracy = evaluate_logits(
        logits_from_multi_route_op_h(model, target_op - center_by_label),
        target_dataset.y,
    )
    subspace_only_loss, subspace_only_accuracy = evaluate_logits(
        logits_from_multi_route_op_h(model, target_projection),
        target_dataset.y,
    )
    subspace_removed_loss, subspace_removed_accuracy = evaluate_logits(
        logits_from_multi_route_op_h(model, target_op - target_projection),
        target_dataset.y,
    )

    return {
        "normal_loss": normal_loss,
        "normal_accuracy": normal_accuracy,
        "center_patch_loss": center_patch_loss,
        "center_patch_accuracy": center_patch_accuracy,
        "residual_only_loss": residual_only_loss,
        "residual_only_accuracy": residual_only_accuracy,
        "subspace_only_loss": subspace_only_loss,
        "subspace_only_accuracy": subspace_only_accuracy,
        "subspace_removed_loss": subspace_removed_loss,
        "subspace_removed_accuracy": subspace_removed_accuracy,
        "center_patch_accuracy_delta": center_patch_accuracy - normal_accuracy,
        "subspace_only_accuracy_delta": subspace_only_accuracy - normal_accuracy,
        "subspace_removed_accuracy_drop": normal_accuracy - subspace_removed_accuracy,
        "residual_only_accuracy_drop": normal_accuracy - residual_only_accuracy,
        "center_patch_loss_delta": center_patch_loss - normal_loss,
        "subspace_removed_loss_delta": subspace_removed_loss - normal_loss,
    }


def logits_from_multi_route_route_h(
    model: MultiRouteFactorizedMLP,
    route_h: np.ndarray,
) -> np.ndarray:
    if route_h.ndim != 2:
        raise ValueError(f"route_h must be 2D, got shape {route_h.shape}.")
    if route_h.shape[1] != model.W_op.shape[0]:
        raise ValueError(
            f"route_h dimension {route_h.shape[1]} does not match W_op input "
            f"dimension {model.W_op.shape[0]}."
        )
    op_h = np.maximum(route_h @ model.W_op + model.b_op, 0.0)
    return logits_from_multi_route_op_h(model, op_h)


def digit_one_hot(values: np.ndarray, class_count: int) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"digit values must be 1D, got shape {values.shape}.")
    if class_count <= 0:
        raise ValueError(f"class_count must be positive, got {class_count}.")
    if np.any(values < 0) or np.any(values >= class_count):
        raise ValueError(f"digit values are outside [0, {class_count}).")
    one_hot = np.zeros((values.shape[0], class_count))
    one_hot[np.arange(values.shape[0]), values.astype(int)] = 1.0
    return one_hot


def ridge_linear_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    ridge: float,
) -> np.ndarray:
    if train_x.ndim != 2 or eval_x.ndim != 2:
        raise ValueError(
            f"ridge inputs must be 2D, got train_x={train_x.shape}, eval_x={eval_x.shape}."
        )
    if train_y.ndim != 2:
        raise ValueError(f"ridge target must be 2D, got train_y={train_y.shape}.")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError(
            f"train_x row count {train_x.shape[0]} does not match train_y "
            f"row count {train_y.shape[0]}."
        )
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError(
            f"train/eval feature dimensions differ: {train_x.shape[1]} vs {eval_x.shape[1]}."
        )
    if ridge <= 0.0:
        raise ValueError(f"ridge must be positive, got {ridge}.")
    train_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    eval_aug = np.concatenate([eval_x, np.ones((eval_x.shape[0], 1))], axis=1)
    penalty = ridge * np.eye(train_aug.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train_aug.T @ train_aug + penalty, train_aug.T @ train_y)
    return eval_aug @ weights


def split_indices(count: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count <= 1:
        raise ValueError(f"count must be greater than one, got {count}.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    train_count = int(round(count * train_fraction))
    if train_count <= 0 or train_count >= count:
        raise ValueError(
            f"train split produced train_count={train_count} for count={count}."
        )
    return order[:train_count], order[train_count:]


def multi_route_analog_causal_metrics(
    model: MultiRouteFactorizedMLP,
    target_dataset: MultiRouteDataset,
    analog_source_dataset: MultiRouteDataset,
) -> dict[str, float]:
    if not np.array_equal(target_dataset.y, analog_source_dataset.y):
        raise ValueError("target and analog source targets must match row-by-row.")
    target_op = multi_route_op_h(model, target_dataset)
    analog_op = multi_route_op_h(model, analog_source_dataset)
    normal_loss, normal_accuracy = evaluate_logits(
        logits_from_multi_route_op_h(model, target_op),
        target_dataset.y,
    )
    analog_patch_loss, analog_patch_accuracy = evaluate_logits(
        logits_from_multi_route_op_h(model, analog_op),
        target_dataset.y,
    )
    return {
        "normal_loss": normal_loss,
        "normal_accuracy": normal_accuracy,
        "analog_patch_loss": analog_patch_loss,
        "analog_patch_accuracy": analog_patch_accuracy,
        "analog_patch_accuracy_delta": analog_patch_accuracy - normal_accuracy,
        "analog_patch_loss_delta": analog_patch_loss - normal_loss,
    }


def multi_route_class_center_alignment_loss_and_grads(
    model: MultiRouteFactorizedMLP,
    target_dataset: MultiRouteDataset,
    source_dataset: MultiRouteDataset,
    class_count: int,
    alignment_weight: float,
) -> tuple[float, float, float, MultiRouteGrads]:
    if target_dataset.op_spec.name == source_dataset.op_spec.name:
        raise ValueError("target and source routes must be different for class-center alignment.")
    if alignment_weight <= 0.0:
        raise ValueError(f"alignment_weight must be positive, got {alignment_weight}.")
    source_op = multi_route_op_h(model, source_dataset)
    source_centers = class_centers(source_op, source_dataset.y, class_count)

    target_name = target_dataset.op_spec.name
    logits, target_cache = multi_route_forward(model, target_dataset.digit_x, target_name)
    ce_loss, dlogits = loss_and_grad_logits(logits, target_dataset.y)
    grads = multi_route_backward(model, target_cache, dlogits)

    target_op = target_cache["op_h"]
    target_op_z = target_cache["op_z"]
    target_route_z = target_cache["route_z"]
    target_digit_x = target_cache["digit_x"]
    if not all(
        isinstance(value, np.ndarray)
        for value in (target_op, target_op_z, target_route_z, target_digit_x)
    ):
        raise ValueError("class-center alignment cache contains non-array values.")
    target_op = np.asarray(target_op)
    target_op_z = np.asarray(target_op_z)
    target_route_z = np.asarray(target_route_z)
    target_digit_x = np.asarray(target_digit_x)

    target_centers = source_centers[target_dataset.y]
    diff = target_op - target_centers
    alignment_loss = float(np.mean(diff**2))
    dop_h = alignment_weight * 2.0 * diff / diff.size
    dop_z = dop_h * (target_op_z > 0.0)
    droute_h = dop_z @ model.W_op.T
    droute_z = droute_h * (target_route_z > 0.0)
    grads.W_router[target_name] += target_digit_x.T @ droute_z
    grads.b_router[target_name] += np.sum(droute_z, axis=0)
    total_loss = ce_loss + alignment_weight * alignment_loss
    return total_loss, ce_loss, alignment_loss, grads


def train_multi_route_class_center_alignment(
    model: MultiRouteFactorizedMLP,
    target_dataset: MultiRouteDataset,
    source_dataset: MultiRouteDataset,
    train_config: TrainConfig,
    label: str,
    class_count: int,
    alignment_weight: float,
) -> None:
    target_name = target_dataset.op_spec.name
    for epoch in range(1, train_config.epochs + 1):
        loss, ce_loss, alignment_loss, grads = (
            multi_route_class_center_alignment_loss_and_grads(
                model,
                target_dataset,
                source_dataset,
                class_count,
                alignment_weight,
            )
        )
        multi_route_apply_update(
            model,
            grads,
            train_config.learning_rate,
            (target_name,),
            train_shared=False,
        )
        if not train_config.quiet and (
            epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs
        ):
            _, accuracy = multi_route_evaluate(model, target_dataset)
            print(
                f"{label} epoch={epoch:04d} loss={loss:.4f} "
                f"ce={ce_loss:.4f} align={alignment_loss:.4f} accuracy={accuracy:.3f}"
            )


def multi_route_shared_gradient_norm(
    model: MultiRouteFactorizedMLP,
    dataset: MultiRouteDataset,
) -> float:
    _, grads = multi_route_loss_and_grads(model, dataset)
    total = (
        float(np.sum(grads.W_op**2))
        + float(np.sum(grads.b_op**2))
        + float(np.sum(grads.W_out**2))
        + float(np.sum(grads.b_out**2))
    )
    return float(np.sqrt(total))


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


def multi_route_addition_metrics(
    seed: int,
    hidden_dim: int,
    alignment_weight: float,
    verbose: bool,
) -> dict[str, dict[str, float]]:
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
    if alignment_weight <= 0.0:
        raise ValueError(f"alignment_weight must be positive, got {alignment_weight}.")

    config = OpsConfig(seed=seed, hidden_dim=hidden_dim)
    specs = DEFAULT_MULTI_ROUTE_ADDITION_SPECS
    validate_multi_route_specs(specs, config.sequence_digits)
    spec_by_name = {spec.name: spec for spec in specs}
    if DEFAULT_MULTI_ROUTE_BASE_NAME not in spec_by_name:
        raise ValueError(
            f"base route {DEFAULT_MULTI_ROUTE_BASE_NAME!r} missing from specs "
            f"{sorted(spec_by_name)}."
        )
    base_spec = spec_by_name[DEFAULT_MULTI_ROUTE_BASE_NAME]
    datasets = {
        spec.name: make_multi_route_dataset(
            spec,
            config.num_digits,
            config.sequence_digits,
        )
        for spec in specs
    }
    analogs = {
        spec.name: analogous_multi_route_dataset(
            base_spec,
            datasets[spec.name],
            config.num_digits,
            config.sequence_digits,
        )
        for spec in specs
        if spec.name != base_spec.name
    }
    base_train, new_train, _, _, _ = build_train_configs(verbose)
    model = make_multi_route_factorized_model(
        specs,
        config.num_digits,
        config.sequence_digits,
        hidden_dim,
        seed,
    )
    train_multi_route(
        model,
        datasets[base_spec.name],
        base_train,
        label="multi_route_base",
        trainable_routes=(base_spec.name,),
        train_shared=True,
    )

    for spec in specs:
        if spec.name == base_spec.name:
            continue
        train_multi_route_alignment(
            model,
            datasets[spec.name],
            analogs[spec.name],
            new_train,
            label=f"multi_route_align_{spec.name}",
            alignment_weight=alignment_weight,
        )

    route_metrics: dict[str, float] = {}
    for spec in specs:
        loss, accuracy = multi_route_evaluate(model, datasets[spec.name])
        route_metrics[f"{spec.name}_loss"] = loss
        route_metrics[f"{spec.name}_accuracy"] = accuracy
        route_metrics[f"{spec.name}_shared_gradient_norm"] = multi_route_shared_gradient_norm(
            model,
            datasets[spec.name],
        )

    alignment_metrics: dict[str, float] = {}
    for spec in specs:
        if spec.name == base_spec.name:
            continue
        pair_metrics = multi_route_pair_alignment_metrics(
            model,
            datasets[spec.name],
            analogs[spec.name],
            config.num_digits,
        )
        for metric_name, value in pair_metrics.items():
            alignment_metrics[f"{spec.name}_{metric_name}"] = value
        causal_metrics = multi_route_analog_causal_metrics(
            model,
            datasets[spec.name],
            analogs[spec.name],
        )
        for metric_name, value in causal_metrics.items():
            alignment_metrics[f"{spec.name}_{metric_name}"] = value

    return {
        "route": route_metrics,
        "alignment": alignment_metrics,
    }


def run_multi_route_addition_table(
    seed: int,
    hidden_dim: int,
    alignment_weight: float,
    verbose: bool,
) -> dict[str, dict[str, float]]:
    print("\nMulti-route aligned addition test")
    metrics = multi_route_addition_metrics(seed, hidden_dim, alignment_weight, verbose)
    specs = DEFAULT_MULTI_ROUTE_ADDITION_SPECS
    print("route  accuracy  loss      shared_g")
    for spec in specs:
        print(
            f"{spec.name:<6} "
            f"{metrics['route'][f'{spec.name}_accuracy']:>8.3f} "
            f"{metrics['route'][f'{spec.name}_loss']:>8.4f} "
            f"{metrics['route'][f'{spec.name}_shared_gradient_norm']:>9.6f}"
        )
    print("target  op_cos   op_class  paired_cka  op_mse  analog_patch_acc")
    for spec in specs:
        if spec.name == DEFAULT_MULTI_ROUTE_BASE_NAME:
            continue
        print(
            f"{spec.name:<7} "
            f"{metrics['alignment'][f'{spec.name}_op_pair_cosine']:>7.3f} "
            f"{metrics['alignment'][f'{spec.name}_op_class_cosine']:>8.3f} "
            f"{metrics['alignment'][f'{spec.name}_paired_op_cka']:>10.3f} "
            f"{metrics['alignment'][f'{spec.name}_op_pair_mse']:>7.4f} "
            f"{metrics['alignment'][f'{spec.name}_analog_patch_accuracy']:>16.3f}"
        )
    return metrics


def print_multi_route_addition_summary(results: list[dict[str, object]]) -> None:
    print("\nMulti-route aligned addition summary: mean +/- std")
    print(f"seeds={[result['seed'] for result in results]}")
    specs = DEFAULT_MULTI_ROUTE_ADDITION_SPECS
    print("\nRoute behavior")
    print("route  accuracy       loss           shared_g")
    for spec in specs:
        accuracy_values = [
            result["multi_route"]["route"][f"{spec.name}_accuracy"]  # type: ignore[index]
            for result in results
        ]
        loss_values = [
            result["multi_route"]["route"][f"{spec.name}_loss"]  # type: ignore[index]
            for result in results
        ]
        gradient_values = [
            result["multi_route"]["route"][f"{spec.name}_shared_gradient_norm"]  # type: ignore[index]
            for result in results
        ]
        acc_mean, acc_std = mean_std(accuracy_values)
        loss_mean, loss_std = mean_std(loss_values)
        grad_mean, grad_std = mean_std(gradient_values)
        print(
            f"{spec.name:<6} "
            f"{acc_mean:.3f}+/-{acc_std:.3f}  "
            f"{loss_mean:.5f}+/-{loss_std:.5f}  "
            f"{grad_mean:.6f}+/-{grad_std:.6f}"
        )

    print("\nAlignment to base route")
    print("target  op_cos        op_class      paired_cka    op_mse        analog_patch")
    for spec in specs:
        if spec.name == DEFAULT_MULTI_ROUTE_BASE_NAME:
            continue
        op_cos_values = [
            result["multi_route"]["alignment"][f"{spec.name}_op_pair_cosine"]  # type: ignore[index]
            for result in results
        ]
        class_cos_values = [
            result["multi_route"]["alignment"][f"{spec.name}_op_class_cosine"]  # type: ignore[index]
            for result in results
        ]
        cka_values = [
            result["multi_route"]["alignment"][f"{spec.name}_paired_op_cka"]  # type: ignore[index]
            for result in results
        ]
        mse_values = [
            result["multi_route"]["alignment"][f"{spec.name}_op_pair_mse"]  # type: ignore[index]
            for result in results
        ]
        analog_patch_values = [
            result["multi_route"]["alignment"][f"{spec.name}_analog_patch_accuracy"]  # type: ignore[index]
            for result in results
        ]
        op_mean, op_std = mean_std(op_cos_values)
        class_mean, class_std = mean_std(class_cos_values)
        cka_mean, cka_std = mean_std(cka_values)
        mse_mean, mse_std = mean_std(mse_values)
        patch_mean, patch_std = mean_std(analog_patch_values)
        print(
            f"{spec.name:<7} "
            f"{op_mean:.3f}+/-{op_std:.3f}  "
            f"{class_mean:.3f}+/-{class_std:.3f}  "
            f"{cka_mean:.3f}+/-{cka_std:.3f}  "
            f"{mse_mean:.5f}+/-{mse_std:.5f}  "
            f"{patch_mean:.3f}+/-{patch_std:.3f}"
        )


def run_multi_route_addition_multi_seed(
    seed_count: int,
    hidden_dim: int,
    alignment_weight: float,
) -> None:
    results: list[dict[str, object]] = []
    for seed in range(seed_count):
        print(f"running_seed={seed}")
        metrics = multi_route_addition_metrics(
            seed,
            hidden_dim,
            alignment_weight,
            verbose=False,
        )
        results.append({"seed": seed, "multi_route": metrics})
    print_multi_route_addition_summary(results)


def build_multi_route_add_family_model(
    seed: int,
    hidden_dim: int,
    alignment_weight: float,
    extra_specs: tuple[MultiRouteOpSpec, ...],
    verbose: bool,
) -> tuple[
    OpsConfig,
    MultiRouteFactorizedMLP,
    tuple[MultiRouteOpSpec, ...],
    dict[str, MultiRouteDataset],
]:
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
    if alignment_weight <= 0.0:
        raise ValueError(f"alignment_weight must be positive, got {alignment_weight}.")
    config = OpsConfig(seed=seed, hidden_dim=hidden_dim)
    addition_specs = DEFAULT_MULTI_ROUTE_ADDITION_SPECS
    all_specs = addition_specs + extra_specs
    validate_multi_route_specs(all_specs, config.sequence_digits)
    spec_by_name = {spec.name: spec for spec in all_specs}
    if DEFAULT_MULTI_ROUTE_BASE_NAME not in spec_by_name:
        raise ValueError(f"missing base route {DEFAULT_MULTI_ROUTE_BASE_NAME!r}.")
    base_spec = spec_by_name[DEFAULT_MULTI_ROUTE_BASE_NAME]
    datasets = {
        spec.name: make_multi_route_dataset(
            spec,
            config.num_digits,
            config.sequence_digits,
        )
        for spec in all_specs
    }
    model = make_multi_route_factorized_model(
        all_specs,
        config.num_digits,
        config.sequence_digits,
        hidden_dim,
        seed,
    )
    base_train, new_train, _, _, _ = build_train_configs(verbose)
    train_multi_route(
        model,
        datasets[base_spec.name],
        base_train,
        label="multi_route_non_analog_base",
        trainable_routes=(base_spec.name,),
        train_shared=True,
    )
    for spec in addition_specs:
        if spec.name == base_spec.name:
            continue
        analog = analogous_multi_route_dataset(
            base_spec,
            datasets[spec.name],
            config.num_digits,
            config.sequence_digits,
        )
        train_multi_route_alignment(
            model,
            datasets[spec.name],
            analog,
            new_train,
            label=f"multi_route_non_analog_align_{spec.name}",
            alignment_weight=alignment_weight,
        )
    return config, model, all_specs, datasets


def evaluate_multi_route_add_family(
    model: MultiRouteFactorizedMLP,
    datasets: dict[str, MultiRouteDataset],
) -> dict[str, float]:
    add_values: dict[str, float] = {}
    accuracies: list[float] = []
    for spec in DEFAULT_MULTI_ROUTE_ADDITION_SPECS:
        loss, accuracy = multi_route_evaluate(model, datasets[spec.name])
        add_values[f"{spec.name}_loss"] = loss
        add_values[f"{spec.name}_accuracy"] = accuracy
        accuracies.append(accuracy)
    add_values["add_family_min_accuracy"] = float(np.min(accuracies))
    add_values["add_family_mean_accuracy"] = float(np.mean(accuracies))
    return add_values


def multi_route_non_analog_metrics(
    seed: int,
    hidden_dim: int,
    alignment_weight: float,
    verbose: bool,
) -> dict[str, dict[str, float]]:
    (
        config,
        add_family_model,
        _,
        datasets,
    ) = build_multi_route_add_family_model(
        seed,
        hidden_dim,
        alignment_weight,
        DEFAULT_MULTI_ROUTE_NON_ANALOG_SPECS,
        verbose,
    )
    _, new_train, _, _, _ = build_train_configs(verbose)
    source_dataset = datasets[DEFAULT_MULTI_ROUTE_BASE_NAME]
    results: dict[str, dict[str, float]] = {}
    for target_spec in DEFAULT_MULTI_ROUTE_NON_ANALOG_SPECS:
        target_dataset = datasets[target_spec.name]

        ce_model = clone_multi_route_factorized_model(add_family_model)
        train_multi_route(
            ce_model,
            target_dataset,
            new_train,
            label=f"multi_route_non_analog_ce_{target_spec.name}",
            trainable_routes=(target_spec.name,),
            train_shared=False,
        )
        ce_loss, ce_accuracy = multi_route_evaluate(ce_model, target_dataset)
        ce_center = multi_route_class_center_alignment_metrics(
            ce_model,
            target_dataset,
            source_dataset,
            config.num_digits,
        )
        ce_causal = multi_route_output_code_causal_metrics(
            ce_model,
            target_dataset,
            source_dataset,
            config.num_digits,
        )
        results[f"{target_spec.name}_ce_only"] = {
            **evaluate_multi_route_add_family(ce_model, datasets),
            "target_loss": ce_loss,
            "target_accuracy": ce_accuracy,
            "target_shared_gradient_norm": multi_route_shared_gradient_norm(
                ce_model,
                target_dataset,
            ),
            **ce_center,
            **ce_causal,
        }

        aligned_model = clone_multi_route_factorized_model(add_family_model)
        train_multi_route_class_center_alignment(
            aligned_model,
            target_dataset,
            source_dataset,
            new_train,
            label=f"multi_route_non_analog_class_align_{target_spec.name}",
            class_count=config.num_digits,
            alignment_weight=alignment_weight,
        )
        aligned_loss, aligned_accuracy = multi_route_evaluate(aligned_model, target_dataset)
        aligned_center = multi_route_class_center_alignment_metrics(
            aligned_model,
            target_dataset,
            source_dataset,
            config.num_digits,
        )
        aligned_causal = multi_route_output_code_causal_metrics(
            aligned_model,
            target_dataset,
            source_dataset,
            config.num_digits,
        )
        results[f"{target_spec.name}_class_align"] = {
            **evaluate_multi_route_add_family(aligned_model, datasets),
            "target_loss": aligned_loss,
            "target_accuracy": aligned_accuracy,
            "target_shared_gradient_norm": multi_route_shared_gradient_norm(
                aligned_model,
                target_dataset,
            ),
            **aligned_center,
            **aligned_causal,
        }
    return results


def route_family_admission_metrics(
    non_analog_metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    admission: dict[str, dict[str, float]] = {}
    for target_spec in DEFAULT_MULTI_ROUTE_NON_ANALOG_SPECS:
        ce_key = f"{target_spec.name}_ce_only"
        align_key = f"{target_spec.name}_class_align"
        if ce_key not in non_analog_metrics or align_key not in non_analog_metrics:
            raise ValueError(
                f"missing CE/alignment metrics for route {target_spec.name}: "
                f"needed {ce_key!r} and {align_key!r}."
            )
        ce_values = non_analog_metrics[ce_key]
        align_values = non_analog_metrics[align_key]
        ce_shared_gradient = ce_values["target_shared_gradient_norm"]
        align_shared_gradient = align_values["target_shared_gradient_norm"]
        if ce_shared_gradient <= 0.0:
            raise ValueError(
                f"CE-only shared-gradient norm for {target_spec.name} is non-positive."
            )

        target_accuracy_delta = (
            align_values["target_accuracy"] - ce_values["target_accuracy"]
        )
        add_family_min_accuracy_delta = (
            align_values["add_family_min_accuracy"]
            - ce_values["add_family_min_accuracy"]
        )
        target_loss_delta = align_values["target_loss"] - ce_values["target_loss"]
        shared_gradient_norm_delta = align_shared_gradient - ce_shared_gradient
        shared_gradient_norm_ratio = align_shared_gradient / ce_shared_gradient
        center_cosine_delta = (
            align_values["center_op_cosine"] - ce_values["center_op_cosine"]
        )
        center_cka_delta = align_values["center_op_cka"] - ce_values["center_op_cka"]
        center_mse_delta = align_values["center_op_mse"] - ce_values["center_op_mse"]
        center_patch_accuracy_delta = (
            align_values["center_patch_accuracy"] - ce_values["center_patch_accuracy"]
        )
        subspace_removed_drop_delta = (
            align_values["subspace_removed_accuracy_drop"]
            - ce_values["subspace_removed_accuracy_drop"]
        )
        residual_only_drop_delta = (
            align_values["residual_only_accuracy_drop"]
            - ce_values["residual_only_accuracy_drop"]
        )

        behavior_pressure_margin = target_accuracy_delta - np.log(
            shared_gradient_norm_ratio
        )
        geometry_gain = center_cosine_delta + center_cka_delta - center_mse_delta
        false_alignment_gap = geometry_gain - behavior_pressure_margin

        admission[target_spec.name] = {
            "target_accuracy_delta": target_accuracy_delta,
            "add_family_min_accuracy_delta": add_family_min_accuracy_delta,
            "target_loss_delta": target_loss_delta,
            "shared_gradient_norm_delta": shared_gradient_norm_delta,
            "shared_gradient_norm_ratio": shared_gradient_norm_ratio,
            "center_cosine_delta": center_cosine_delta,
            "center_cka_delta": center_cka_delta,
            "center_mse_delta": center_mse_delta,
            "center_patch_accuracy_delta": center_patch_accuracy_delta,
            "subspace_removed_drop_delta": subspace_removed_drop_delta,
            "residual_only_drop_delta": residual_only_drop_delta,
            "behavior_pressure_margin": behavior_pressure_margin,
            "geometry_gain": geometry_gain,
            "false_alignment_gap": false_alignment_gap,
        }
    return admission


def run_multi_route_non_analog_table(
    seed: int,
    hidden_dim: int,
    alignment_weight: float,
    verbose: bool,
) -> dict[str, dict[str, float]]:
    print("\nMulti-route non-analog routing test")
    metrics = multi_route_non_analog_metrics(seed, hidden_dim, alignment_weight, verbose)
    print(
        "variant                target_acc  add_min  target_loss  shared_g   "
        "center_cos  center_cka  center_mse"
    )
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<22} "
            f"{values['target_accuracy']:>10.3f} "
            f"{values['add_family_min_accuracy']:>8.3f} "
            f"{values['target_loss']:>11.4f} "
            f"{values['target_shared_gradient_norm']:>9.6f} "
            f"{values['center_op_cosine']:>10.3f} "
            f"{values['center_op_cka']:>10.3f} "
            f"{values['center_op_mse']:>10.4f}"
        )
    print(
        "\nCausal output-code probes "
        "(center/subspace accuracies use ADD01 class-code interventions)"
    )
    print(
        "variant                center_patch  subspace_only  subspace_removed  "
        "residual_only  removed_drop"
    )
    for variant_name, values in metrics.items():
        print(
            f"{variant_name:<22} "
            f"{values['center_patch_accuracy']:>12.3f} "
            f"{values['subspace_only_accuracy']:>13.3f} "
            f"{values['subspace_removed_accuracy']:>16.3f} "
            f"{values['residual_only_accuracy']:>13.3f} "
            f"{values['subspace_removed_accuracy_drop']:>12.3f}"
        )
    admission = route_family_admission_metrics(metrics)
    print(
        "\nRoute-family admission comparison "
        "(class-align minus CE-only; pressure ratio > 1 means more pressure)"
    )
    print(
        "target  acc_delta  add_delta  loss_delta  shared_g_ratio  "
        "cos_delta  cka_delta  mse_delta  patch_delta  rm_drop_delta  "
        "bp_margin  false_gap"
    )
    for target_name, values in admission.items():
        print(
            f"{target_name:<7} "
            f"{values['target_accuracy_delta']:>9.3f} "
            f"{values['add_family_min_accuracy_delta']:>9.3f} "
            f"{values['target_loss_delta']:>10.5f} "
            f"{values['shared_gradient_norm_ratio']:>15.3f} "
            f"{values['center_cosine_delta']:>10.3f} "
            f"{values['center_cka_delta']:>10.3f} "
            f"{values['center_mse_delta']:>9.5f} "
            f"{values['center_patch_accuracy_delta']:>11.3f} "
            f"{values['subspace_removed_drop_delta']:>13.3f} "
            f"{values['behavior_pressure_margin']:>9.3f} "
            f"{values['false_alignment_gap']:>9.3f}"
        )
    return metrics


def print_multi_route_non_analog_summary(results: list[dict[str, object]]) -> None:
    print("\nMulti-route non-analog routing summary: mean +/- std")
    print(f"seeds={[result['seed'] for result in results]}")
    print(
        "variant                target_acc     add_min        target_loss    shared_g      "
        "center_cos    center_cka    center_mse"
    )
    variant_names = tuple(results[0]["non_analog"].keys())  # type: ignore[union-attr]
    for variant_name in variant_names:
        target_acc_values = [
            result["non_analog"][variant_name]["target_accuracy"]  # type: ignore[index]
            for result in results
        ]
        add_min_values = [
            result["non_analog"][variant_name]["add_family_min_accuracy"]  # type: ignore[index]
            for result in results
        ]
        loss_values = [
            result["non_analog"][variant_name]["target_loss"]  # type: ignore[index]
            for result in results
        ]
        gradient_values = [
            result["non_analog"][variant_name]["target_shared_gradient_norm"]  # type: ignore[index]
            for result in results
        ]
        center_cos_values = [
            result["non_analog"][variant_name]["center_op_cosine"]  # type: ignore[index]
            for result in results
        ]
        center_cka_values = [
            result["non_analog"][variant_name]["center_op_cka"]  # type: ignore[index]
            for result in results
        ]
        center_mse_values = [
            result["non_analog"][variant_name]["center_op_mse"]  # type: ignore[index]
            for result in results
        ]
        target_mean, target_std = mean_std(target_acc_values)
        add_mean, add_std = mean_std(add_min_values)
        loss_mean, loss_std = mean_std(loss_values)
        grad_mean, grad_std = mean_std(gradient_values)
        cos_mean, cos_std = mean_std(center_cos_values)
        cka_mean, cka_std = mean_std(center_cka_values)
        mse_mean, mse_std = mean_std(center_mse_values)
        print(
            f"{variant_name:<22} "
            f"{target_mean:.3f}+/-{target_std:.3f}  "
            f"{add_mean:.3f}+/-{add_std:.3f}  "
            f"{loss_mean:.5f}+/-{loss_std:.5f}  "
            f"{grad_mean:.6f}+/-{grad_std:.6f}  "
            f"{cos_mean:.3f}+/-{cos_std:.3f}  "
            f"{cka_mean:.3f}+/-{cka_std:.3f}  "
            f"{mse_mean:.5f}+/-{mse_std:.5f}"
        )

    print("\nCausal output-code probes: mean +/- std")
    print(
        "variant                center_patch  subspace_only  subspace_removed  "
        "residual_only  removed_drop"
    )
    for variant_name in variant_names:
        center_patch_values = [
            result["non_analog"][variant_name]["center_patch_accuracy"]  # type: ignore[index]
            for result in results
        ]
        subspace_only_values = [
            result["non_analog"][variant_name]["subspace_only_accuracy"]  # type: ignore[index]
            for result in results
        ]
        subspace_removed_values = [
            result["non_analog"][variant_name]["subspace_removed_accuracy"]  # type: ignore[index]
            for result in results
        ]
        residual_only_values = [
            result["non_analog"][variant_name]["residual_only_accuracy"]  # type: ignore[index]
            for result in results
        ]
        removed_drop_values = [
            result["non_analog"][variant_name]["subspace_removed_accuracy_drop"]  # type: ignore[index]
            for result in results
        ]
        center_mean, center_std = mean_std(center_patch_values)
        subspace_mean, subspace_std = mean_std(subspace_only_values)
        removed_mean, removed_std = mean_std(subspace_removed_values)
        residual_mean, residual_std = mean_std(residual_only_values)
        drop_mean, drop_std = mean_std(removed_drop_values)
        print(
            f"{variant_name:<22} "
            f"{center_mean:.3f}+/-{center_std:.3f}  "
            f"{subspace_mean:.3f}+/-{subspace_std:.3f}  "
            f"{removed_mean:.3f}+/-{removed_std:.3f}  "
            f"{residual_mean:.3f}+/-{residual_std:.3f}  "
            f"{drop_mean:.3f}+/-{drop_std:.3f}"
        )

    print(
        "\nRoute-family admission comparison: class-align minus CE-only "
        "(mean +/- std)"
    )
    print(
        "target  acc_delta     add_delta     loss_delta    shared_g_ratio  "
        "cos_delta    cka_delta    mse_delta    patch_delta  rm_drop_delta  "
        "bp_margin    false_gap"
    )
    for target_spec in DEFAULT_MULTI_ROUTE_NON_ANALOG_SPECS:
        per_seed_admission = [
            route_family_admission_metrics(result["non_analog"])[target_spec.name]  # type: ignore[arg-type,index]
            for result in results
        ]
        target_delta_values = [
            values["target_accuracy_delta"] for values in per_seed_admission
        ]
        add_delta_values = [
            values["add_family_min_accuracy_delta"] for values in per_seed_admission
        ]
        loss_delta_values = [
            values["target_loss_delta"] for values in per_seed_admission
        ]
        pressure_ratio_values = [
            values["shared_gradient_norm_ratio"] for values in per_seed_admission
        ]
        cosine_delta_values = [
            values["center_cosine_delta"] for values in per_seed_admission
        ]
        cka_delta_values = [
            values["center_cka_delta"] for values in per_seed_admission
        ]
        mse_delta_values = [
            values["center_mse_delta"] for values in per_seed_admission
        ]
        patch_delta_values = [
            values["center_patch_accuracy_delta"] for values in per_seed_admission
        ]
        removed_drop_delta_values = [
            values["subspace_removed_drop_delta"] for values in per_seed_admission
        ]
        behavior_pressure_values = [
            values["behavior_pressure_margin"] for values in per_seed_admission
        ]
        false_gap_values = [
            values["false_alignment_gap"] for values in per_seed_admission
        ]
        target_mean, target_std = mean_std(target_delta_values)
        add_mean, add_std = mean_std(add_delta_values)
        loss_mean, loss_std = mean_std(loss_delta_values)
        pressure_mean, pressure_std = mean_std(pressure_ratio_values)
        cosine_mean, cosine_std = mean_std(cosine_delta_values)
        cka_mean, cka_std = mean_std(cka_delta_values)
        mse_mean, mse_std = mean_std(mse_delta_values)
        patch_mean, patch_std = mean_std(patch_delta_values)
        removed_drop_mean, removed_drop_std = mean_std(removed_drop_delta_values)
        bp_mean, bp_std = mean_std(behavior_pressure_values)
        gap_mean, gap_std = mean_std(false_gap_values)
        print(
            f"{target_spec.name:<7} "
            f"{target_mean:.3f}+/-{target_std:.3f}  "
            f"{add_mean:.3f}+/-{add_std:.3f}  "
            f"{loss_mean:.5f}+/-{loss_std:.5f}  "
            f"{pressure_mean:.3f}+/-{pressure_std:.3f}  "
            f"{cosine_mean:.3f}+/-{cosine_std:.3f}  "
            f"{cka_mean:.3f}+/-{cka_std:.3f}  "
            f"{mse_mean:.5f}+/-{mse_std:.5f}  "
            f"{patch_mean:.3f}+/-{patch_std:.3f}  "
            f"{removed_drop_mean:.3f}+/-{removed_drop_std:.3f}  "
            f"{bp_mean:.3f}+/-{bp_std:.3f}  "
            f"{gap_mean:.3f}+/-{gap_std:.3f}"
        )


def run_multi_route_non_analog_multi_seed(
    seed_count: int,
    hidden_dim: int,
    alignment_weight: float,
) -> None:
    results: list[dict[str, object]] = []
    for seed in range(seed_count):
        print(f"running_seed={seed}")
        metrics = multi_route_non_analog_metrics(
            seed,
            hidden_dim,
            alignment_weight,
            verbose=False,
        )
        results.append({"seed": seed, "non_analog": metrics})
    print_multi_route_non_analog_summary(results)


def composition_benchmark_specs() -> tuple[MultiRouteOpSpec, ...]:
    return (
        DEFAULT_MULTI_ROUTE_ADDITION_SPECS
        + DEFAULT_MULTI_ROUTE_NON_ANALOG_SPECS
        + DEFAULT_MULTI_ROUTE_COMPOSITION_SPECS
    )


def train_composition_benchmark_route(
    model: MultiRouteFactorizedMLP,
    dataset: MultiRouteDataset,
    source_dataset: MultiRouteDataset,
    config: OpsConfig,
    train_config: TrainConfig,
    policy: str,
    label: str,
    alignment_weight: float,
) -> None:
    if policy not in COMPOSITION_BENCHMARK_POLICIES:
        raise ValueError(
            f"composition benchmark policy must be one of "
            f"{COMPOSITION_BENCHMARK_POLICIES}, got {policy!r}."
        )
    if policy == "admission":
        train_multi_route(
            model,
            dataset,
            train_config,
            label=label,
            trainable_routes=(dataset.op_spec.name,),
            train_shared=False,
        )
        return
    if policy == "force_class_align":
        train_multi_route_class_center_alignment(
            model,
            dataset,
            source_dataset,
            train_config,
            label=label,
            class_count=config.num_digits,
            alignment_weight=alignment_weight,
        )
        return
    raise ValueError(f"unsupported composition benchmark policy {policy!r}.")


def multi_route_parameter_metrics(
    model: MultiRouteFactorizedMLP,
    active_route_count: int,
) -> dict[str, float]:
    validate_multi_route_model(model)
    if active_route_count <= 0:
        raise ValueError(f"active_route_count must be positive, got {active_route_count}.")
    route_names = tuple(model.W_router)
    first_route = route_names[0]
    route_param_count = (
        model.W_router[first_route].size
        + model.b_router[first_route].size
    )
    for route_name in route_names[1:]:
        current_count = (
            model.W_router[route_name].size
            + model.b_router[route_name].size
        )
        if current_count != route_param_count:
            raise ValueError(
                f"router {route_name} has {current_count} parameters; "
                f"expected {route_param_count}."
            )
    shared_params = (
        model.W_op.size
        + model.b_op.size
        + model.W_out.size
        + model.b_out.size
    )
    active_router_params = route_param_count * active_route_count
    total_active_params = active_router_params + shared_params
    if shared_params <= 0:
        raise ValueError("shared parameter count must be positive.")
    return {
        "route_param_count": float(route_param_count),
        "active_route_count": float(active_route_count),
        "active_router_params": float(active_router_params),
        "shared_params": float(shared_params),
        "total_active_params": float(total_active_params),
        "router_to_shared_ratio": float(active_router_params / shared_params),
    }


def composition_benchmark_metrics(
    seed: int,
    hidden_dim: int,
    alignment_weight: float,
    policy: str,
    verbose: bool,
) -> dict[str, dict[str, float]]:
    if policy not in COMPOSITION_BENCHMARK_POLICIES:
        raise ValueError(
            f"composition benchmark policy must be one of "
            f"{COMPOSITION_BENCHMARK_POLICIES}, got {policy!r}."
        )
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
    if alignment_weight <= 0.0:
        raise ValueError(f"alignment_weight must be positive, got {alignment_weight}.")

    config = OpsConfig(seed=seed, hidden_dim=hidden_dim)
    specs = composition_benchmark_specs()
    validate_multi_route_specs(specs, config.sequence_digits)
    spec_by_name = {spec.name: spec for spec in specs}
    if DEFAULT_MULTI_ROUTE_BASE_NAME not in spec_by_name:
        raise ValueError(f"missing base route {DEFAULT_MULTI_ROUTE_BASE_NAME!r}.")
    base_spec = spec_by_name[DEFAULT_MULTI_ROUTE_BASE_NAME]
    datasets = {
        spec.name: make_multi_route_dataset(
            spec,
            config.num_digits,
            config.sequence_digits,
        )
        for spec in specs
    }
    base_train, new_train, _, _, _ = build_train_configs(verbose)
    model = make_multi_route_factorized_model(
        specs,
        config.num_digits,
        config.sequence_digits,
        hidden_dim,
        seed,
    )
    train_multi_route(
        model,
        datasets[base_spec.name],
        base_train,
        label=f"composition_{policy}_base",
        trainable_routes=(base_spec.name,),
        train_shared=True,
    )
    for spec in DEFAULT_MULTI_ROUTE_ADDITION_SPECS:
        if spec.name == base_spec.name:
            continue
        analog = analogous_multi_route_dataset(
            base_spec,
            datasets[spec.name],
            config.num_digits,
            config.sequence_digits,
        )
        train_multi_route_alignment(
            model,
            datasets[spec.name],
            analog,
            new_train,
            label=f"composition_{policy}_analog_{spec.name}",
            alignment_weight=alignment_weight,
        )

    source_dataset = datasets[base_spec.name]
    for spec in DEFAULT_MULTI_ROUTE_NON_ANALOG_SPECS + DEFAULT_MULTI_ROUTE_COMPOSITION_SPECS:
        train_composition_benchmark_route(
            model,
            datasets[spec.name],
            source_dataset,
            config,
            new_train,
            policy,
            label=f"composition_{policy}_{spec.name}",
            alignment_weight=alignment_weight,
        )

    route_metrics: dict[str, float] = {}
    route_accuracies: list[float] = []
    route_losses: list[float] = []
    shared_gradients: list[float] = []
    for spec in specs:
        loss, accuracy = multi_route_evaluate(model, datasets[spec.name])
        shared_gradient = multi_route_shared_gradient_norm(model, datasets[spec.name])
        route_metrics[f"{spec.name}_loss"] = loss
        route_metrics[f"{spec.name}_accuracy"] = accuracy
        route_metrics[f"{spec.name}_shared_gradient_norm"] = shared_gradient
        route_accuracies.append(accuracy)
        route_losses.append(loss)
        shared_gradients.append(shared_gradient)

    add_accuracies = [
        route_metrics[f"{spec.name}_accuracy"]
        for spec in DEFAULT_MULTI_ROUTE_ADDITION_SPECS
    ]
    non_analog_accuracies = [
        route_metrics[f"{spec.name}_accuracy"]
        for spec in DEFAULT_MULTI_ROUTE_NON_ANALOG_SPECS
    ]
    composition_accuracies = [
        route_metrics[f"{spec.name}_accuracy"]
        for spec in DEFAULT_MULTI_ROUTE_COMPOSITION_SPECS
    ]
    aggregate_metrics = {
        "base_accuracy": route_metrics[f"{base_spec.name}_accuracy"],
        "mean_accuracy": float(np.mean(route_accuracies)),
        "worst_accuracy": float(np.min(route_accuracies)),
        "mean_loss": float(np.mean(route_losses)),
        "mean_shared_gradient_norm": float(np.mean(shared_gradients)),
        "max_shared_gradient_norm": float(np.max(shared_gradients)),
        "add_family_min_accuracy": float(np.min(add_accuracies)),
        "add_family_mean_accuracy": float(np.mean(add_accuracies)),
        "non_analog_min_accuracy": float(np.min(non_analog_accuracies)),
        "non_analog_mean_accuracy": float(np.mean(non_analog_accuracies)),
        "composition_min_accuracy": float(np.min(composition_accuracies)),
        "composition_mean_accuracy": float(np.mean(composition_accuracies)),
        "examples_per_route": float(len(next(iter(datasets.values())).y)),
        "new_route_train_steps": float((len(specs) - 1) * new_train.epochs),
        **multi_route_parameter_metrics(model, len(specs)),
    }

    causal_metrics: dict[str, float] = {}
    for spec in specs:
        if spec.name == base_spec.name:
            continue
        code_metrics = multi_route_output_code_causal_metrics(
            model,
            datasets[spec.name],
            source_dataset,
            config.num_digits,
        )
        for metric_name, value in code_metrics.items():
            causal_metrics[f"{spec.name}_{metric_name}"] = value
        if spec.kind == base_spec.kind and len(spec.operands) == len(base_spec.operands):
            analog = analogous_multi_route_dataset(
                base_spec,
                datasets[spec.name],
                config.num_digits,
                config.sequence_digits,
            )
            analog_metrics = multi_route_analog_causal_metrics(
                model,
                datasets[spec.name],
                analog,
            )
            for metric_name, value in analog_metrics.items():
                causal_metrics[f"{spec.name}_{metric_name}"] = value

    closure_candidates = [
        spec
        for spec in DEFAULT_MULTI_ROUTE_COMPOSITION_SPECS
        if spec.kind == "add"
        and len(spec.operands) == len(base_spec.operands) + 1
        and tuple(spec.operands[: len(base_spec.operands)]) == base_spec.operands
    ]
    if len(closure_candidates) != 1:
        raise ValueError(
            f"expected exactly one iterative-add closure target, got {closure_candidates}."
        )
    sum_spec = closure_candidates[0]
    closure_metrics = iterative_add_closure_metrics(
        model,
        datasets[sum_spec.name],
        base_spec,
        config.num_digits,
        config.sequence_digits,
        seed,
    )

    return {
        "route": route_metrics,
        "aggregate": aggregate_metrics,
        "causal": causal_metrics,
        "closure": closure_metrics,
    }


def composition_benchmark_policy_comparison(
    policy_metrics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    if set(policy_metrics) != set(COMPOSITION_BENCHMARK_POLICIES):
        raise ValueError(
            f"expected metrics for policies {COMPOSITION_BENCHMARK_POLICIES}, "
            f"got {sorted(policy_metrics)}."
        )
    admission = policy_metrics["admission"]["aggregate"]
    forced = policy_metrics["force_class_align"]["aggregate"]
    return {
        "forced_minus_admission_mean_accuracy": (
            forced["mean_accuracy"] - admission["mean_accuracy"]
        ),
        "forced_minus_admission_worst_accuracy": (
            forced["worst_accuracy"] - admission["worst_accuracy"]
        ),
        "forced_minus_admission_composition_mean_accuracy": (
            forced["composition_mean_accuracy"] - admission["composition_mean_accuracy"]
        ),
        "forced_over_admission_mean_shared_gradient_norm": (
            forced["mean_shared_gradient_norm"] / admission["mean_shared_gradient_norm"]
        ),
        "forced_minus_admission_mean_loss": (
            forced["mean_loss"] - admission["mean_loss"]
        ),
    }


def two_step_add_digit_x(
    source_dataset: MultiRouteDataset,
    base_spec: MultiRouteOpSpec,
    intermediate_digits: np.ndarray,
    final_operand_position: int,
    num_digits: int,
    sequence_digits: int,
) -> np.ndarray:
    if base_spec.kind != "add" or len(base_spec.operands) != 2:
        raise ValueError(
            f"base spec must be a binary add route, got {base_spec}."
        )
    if intermediate_digits.shape != (source_dataset.digit_x.shape[0],):
        raise ValueError(
            f"intermediate digit shape {intermediate_digits.shape} does not match "
            f"dataset row count {source_dataset.digit_x.shape[0]}."
        )
    if final_operand_position < 0 or final_operand_position >= sequence_digits:
        raise ValueError(
            f"final_operand_position={final_operand_position} outside sequence length "
            f"{sequence_digits}."
        )
    rows: list[np.ndarray] = []
    first_operand, second_operand = base_spec.operands
    for digit_row, intermediate_digit in zip(
        source_dataset.digit_x,
        intermediate_digits,
        strict=True,
    ):
        digits = list(decode_multi_route_digits(digit_row, num_digits, sequence_digits))
        digits[first_operand] = int(intermediate_digit)
        digits[second_operand] = digits[final_operand_position]
        rows.append(multi_route_digit_features(tuple(digits), num_digits))
    return np.array(rows)


def iterative_add_closure_metrics(
    model: MultiRouteFactorizedMLP,
    sum_dataset: MultiRouteDataset,
    base_spec: MultiRouteOpSpec,
    num_digits: int,
    sequence_digits: int,
    seed: int,
    train_fraction: float = 0.7,
    ridge: float = 1e-4,
) -> dict[str, float]:
    if sum_dataset.op_spec.kind != "add":
        raise ValueError(f"closure target must be an add task, got {sum_dataset.op_spec}.")
    if base_spec.kind != "add" or len(base_spec.operands) != 2:
        raise ValueError(f"base spec must be binary add, got {base_spec}.")
    if len(sum_dataset.op_spec.operands) != len(base_spec.operands) + 1:
        raise ValueError(
            f"closure target must add exactly one operand beyond base route; "
            f"got base={base_spec.operands}, target={sum_dataset.op_spec.operands}."
        )
    if tuple(sum_dataset.op_spec.operands[: len(base_spec.operands)]) != base_spec.operands:
        raise ValueError(
            f"closure target operands must start with base operands; "
            f"got base={base_spec.operands}, target={sum_dataset.op_spec.operands}."
        )
    final_operand_position = sum_dataset.op_spec.operands[-1]

    direct_loss, direct_accuracy = multi_route_evaluate(model, sum_dataset)
    first_logits, first_cache = multi_route_forward(
        model,
        sum_dataset.digit_x,
        base_spec.name,
    )
    first_targets = np.array(
        [
            multi_route_target(
                base_spec,
                decode_multi_route_digits(row, num_digits, sequence_digits),
                num_digits,
            )
            for row in sum_dataset.digit_x
        ],
        dtype=int,
    )
    first_predictions = np.argmax(first_logits, axis=1)
    first_step_accuracy = float(np.mean(first_predictions == first_targets))

    symbolic_digit_x = two_step_add_digit_x(
        sum_dataset,
        base_spec,
        first_targets,
        final_operand_position,
        num_digits,
        sequence_digits,
    )
    decoded_digit_x = two_step_add_digit_x(
        sum_dataset,
        base_spec,
        first_predictions,
        final_operand_position,
        num_digits,
        sequence_digits,
    )
    symbolic_logits, symbolic_cache = multi_route_forward(
        model,
        symbolic_digit_x,
        base_spec.name,
    )
    decoded_logits, _ = multi_route_forward(
        model,
        decoded_digit_x,
        base_spec.name,
    )
    symbolic_loss, symbolic_accuracy = evaluate_logits(symbolic_logits, sum_dataset.y)
    decoded_loss, decoded_accuracy = evaluate_logits(decoded_logits, sum_dataset.y)

    first_op_h = first_cache["op_h"]
    symbolic_route_h = symbolic_cache["route_h"]
    if not isinstance(first_op_h, np.ndarray) or not isinstance(symbolic_route_h, np.ndarray):
        raise ValueError("closure cache contains non-array values.")

    final_operand_digits = np.array(
        [
            decode_multi_route_digits(row, num_digits, sequence_digits)[final_operand_position]
            for row in sum_dataset.digit_x
        ],
        dtype=int,
    )
    bridge_x = np.concatenate(
        [first_op_h, digit_one_hot(final_operand_digits, num_digits)],
        axis=1,
    )
    train_idx, test_idx = split_indices(len(sum_dataset.y), train_fraction, seed)
    train_pred_route_h = ridge_linear_predict(
        bridge_x[train_idx],
        symbolic_route_h[train_idx],
        bridge_x[train_idx],
        ridge,
    )
    test_pred_route_h = ridge_linear_predict(
        bridge_x[train_idx],
        symbolic_route_h[train_idx],
        bridge_x[test_idx],
        ridge,
    )
    train_bridge_loss, train_bridge_accuracy = evaluate_logits(
        logits_from_multi_route_route_h(model, train_pred_route_h),
        sum_dataset.y[train_idx],
    )
    test_bridge_loss, test_bridge_accuracy = evaluate_logits(
        logits_from_multi_route_route_h(model, test_pred_route_h),
        sum_dataset.y[test_idx],
    )
    test_route_h_mse = float(np.mean((test_pred_route_h - symbolic_route_h[test_idx]) ** 2))
    test_route_h_cosine = mean_row_cosine(test_pred_route_h, symbolic_route_h[test_idx])

    symbolic_test_loss, symbolic_test_accuracy = evaluate_logits(
        symbolic_logits[test_idx],
        sum_dataset.y[test_idx],
    )
    decoded_test_loss, decoded_test_accuracy = evaluate_logits(
        decoded_logits[test_idx],
        sum_dataset.y[test_idx],
    )

    return {
        "direct_loss": direct_loss,
        "direct_accuracy": direct_accuracy,
        "first_step_accuracy": first_step_accuracy,
        "symbolic_two_step_loss": symbolic_loss,
        "symbolic_two_step_accuracy": symbolic_accuracy,
        "decoded_two_step_loss": decoded_loss,
        "decoded_two_step_accuracy": decoded_accuracy,
        "symbolic_two_step_test_loss": symbolic_test_loss,
        "symbolic_two_step_test_accuracy": symbolic_test_accuracy,
        "decoded_two_step_test_loss": decoded_test_loss,
        "decoded_two_step_test_accuracy": decoded_test_accuracy,
        "latent_bridge_train_loss": train_bridge_loss,
        "latent_bridge_train_accuracy": train_bridge_accuracy,
        "latent_bridge_test_loss": test_bridge_loss,
        "latent_bridge_test_accuracy": test_bridge_accuracy,
        "latent_bridge_route_h_mse": test_route_h_mse,
        "latent_bridge_route_h_cosine": test_route_h_cosine,
        "symbolic_minus_direct_accuracy": symbolic_accuracy - direct_accuracy,
        "decoded_minus_direct_accuracy": decoded_accuracy - direct_accuracy,
        "latent_bridge_test_minus_direct_accuracy": test_bridge_accuracy - direct_accuracy,
    }


def run_composition_benchmark_metrics(
    seed: int,
    hidden_dim: int,
    alignment_weight: float,
    verbose: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        policy: composition_benchmark_metrics(
            seed,
            hidden_dim,
            alignment_weight,
            policy,
            verbose,
        )
        for policy in COMPOSITION_BENCHMARK_POLICIES
    }


def print_composition_benchmark_table(
    policy_metrics: dict[str, dict[str, dict[str, float]]],
) -> None:
    print("\nComposition continual-learning benchmark")
    print(
        "policy             mean_acc  worst_acc  add_min  nonanalog_min  "
        "composition_min  mean_shared_g  routes  router/shared"
    )
    for policy in COMPOSITION_BENCHMARK_POLICIES:
        values = policy_metrics[policy]["aggregate"]
        print(
            f"{policy:<18} "
            f"{values['mean_accuracy']:>8.3f} "
            f"{values['worst_accuracy']:>10.3f} "
            f"{values['add_family_min_accuracy']:>8.3f} "
            f"{values['non_analog_min_accuracy']:>14.3f} "
            f"{values['composition_min_accuracy']:>15.3f} "
            f"{values['mean_shared_gradient_norm']:>14.6f} "
            f"{int(values['active_route_count']):>7} "
            f"{values['router_to_shared_ratio']:>13.3f}"
        )

    print("\nRoute accuracies")
    print("route         " + "  ".join(f"{policy:>17}" for policy in COMPOSITION_BENCHMARK_POLICIES))
    for spec in composition_benchmark_specs():
        row = [spec.name.ljust(12)]
        for policy in COMPOSITION_BENCHMARK_POLICIES:
            row.append(f"{policy_metrics[policy]['route'][f'{spec.name}_accuracy']:>17.3f}")
        print("  ".join(row))

    print("\nCausal-code summary")
    print("policy             analog_patch  center_patch  removed_drop")
    for policy in COMPOSITION_BENCHMARK_POLICIES:
        causal = policy_metrics[policy]["causal"]
        analog_values = [
            causal[f"{spec.name}_analog_patch_accuracy"]
            for spec in DEFAULT_MULTI_ROUTE_ADDITION_SPECS
            if spec.name != DEFAULT_MULTI_ROUTE_BASE_NAME
        ]
        center_values = [
            causal[f"{spec.name}_center_patch_accuracy"]
            for spec in composition_benchmark_specs()
            if spec.name != DEFAULT_MULTI_ROUTE_BASE_NAME
        ]
        removed_drop_values = [
            causal[f"{spec.name}_subspace_removed_accuracy_drop"]
            for spec in composition_benchmark_specs()
            if spec.name != DEFAULT_MULTI_ROUTE_BASE_NAME
        ]
        analog_mean, _ = mean_std(analog_values)
        center_mean, _ = mean_std(center_values)
        removed_mean, _ = mean_std(removed_drop_values)
        print(
            f"{policy:<18} "
            f"{analog_mean:>12.3f} "
            f"{center_mean:>13.3f} "
            f"{removed_mean:>13.3f}"
        )

    print("\nIterative ADD closure on composition target")
    print(
        "policy             direct  symbolic_2step  decoded_2step  "
        "latent_bridge_test  bridge_route_cos"
    )
    for policy in COMPOSITION_BENCHMARK_POLICIES:
        closure = policy_metrics[policy]["closure"]
        print(
            f"{policy:<18} "
            f"{closure['direct_accuracy']:>6.3f} "
            f"{closure['symbolic_two_step_accuracy']:>15.3f} "
            f"{closure['decoded_two_step_accuracy']:>14.3f} "
            f"{closure['latent_bridge_test_accuracy']:>18.3f} "
            f"{closure['latent_bridge_route_h_cosine']:>16.3f}"
        )

    comparison = composition_benchmark_policy_comparison(policy_metrics)
    print("\nForced class-align minus admission policy")
    for metric_name, value in comparison.items():
        print(f"{metric_name}={value:.6f}")


def run_composition_benchmark_table(
    seed: int,
    hidden_dim: int,
    alignment_weight: float,
    verbose: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    metrics = run_composition_benchmark_metrics(seed, hidden_dim, alignment_weight, verbose)
    print_composition_benchmark_table(metrics)
    return metrics


def print_composition_benchmark_summary(results: list[dict[str, object]]) -> None:
    print("\nComposition continual-learning benchmark summary: mean +/- std")
    print(f"seeds={[result['seed'] for result in results]}")
    print(
        "policy             mean_acc      worst_acc     add_min       nonanalog_min  "
        "composition_min  mean_shared_g  router/shared"
    )
    for policy in COMPOSITION_BENCHMARK_POLICIES:
        aggregates = [
            result["composition"][policy]["aggregate"]  # type: ignore[index]
            for result in results
        ]
        mean_acc = [values["mean_accuracy"] for values in aggregates]
        worst_acc = [values["worst_accuracy"] for values in aggregates]
        add_min = [values["add_family_min_accuracy"] for values in aggregates]
        nonanalog_min = [values["non_analog_min_accuracy"] for values in aggregates]
        composition_min = [values["composition_min_accuracy"] for values in aggregates]
        mean_shared_g = [values["mean_shared_gradient_norm"] for values in aggregates]
        router_ratios = [values["router_to_shared_ratio"] for values in aggregates]
        mean_acc_mean, mean_acc_std = mean_std(mean_acc)
        worst_mean, worst_std = mean_std(worst_acc)
        add_mean, add_std = mean_std(add_min)
        nonanalog_mean, nonanalog_std = mean_std(nonanalog_min)
        composition_mean, composition_std = mean_std(composition_min)
        shared_mean, shared_std = mean_std(mean_shared_g)
        ratio_mean, ratio_std = mean_std(router_ratios)
        print(
            f"{policy:<18} "
            f"{mean_acc_mean:.3f}+/-{mean_acc_std:.3f}  "
            f"{worst_mean:.3f}+/-{worst_std:.3f}  "
            f"{add_mean:.3f}+/-{add_std:.3f}  "
            f"{nonanalog_mean:.3f}+/-{nonanalog_std:.3f}  "
            f"{composition_mean:.3f}+/-{composition_std:.3f}  "
            f"{shared_mean:.6f}+/-{shared_std:.6f}  "
            f"{ratio_mean:.3f}+/-{ratio_std:.3f}"
        )

    print("\nComposition route accuracies: mean +/- std")
    print("route         " + "  ".join(f"{policy:>25}" for policy in COMPOSITION_BENCHMARK_POLICIES))
    for spec in composition_benchmark_specs():
        row = [spec.name.ljust(12)]
        for policy in COMPOSITION_BENCHMARK_POLICIES:
            values = [
                result["composition"][policy]["route"][f"{spec.name}_accuracy"]  # type: ignore[index]
                for result in results
            ]
            value_mean, value_std = mean_std(values)
            row.append(f"{value_mean:.3f}+/-{value_std:.3f}".rjust(25))
        print("  ".join(row))

    print("\nCausal-code summary: mean +/- std")
    print("policy             analog_patch  center_patch  removed_drop")
    for policy in COMPOSITION_BENCHMARK_POLICIES:
        analog_values = []
        center_values = []
        removed_drop_values = []
        for result in results:
            causal = result["composition"][policy]["causal"]  # type: ignore[index]
            for spec in DEFAULT_MULTI_ROUTE_ADDITION_SPECS:
                if spec.name != DEFAULT_MULTI_ROUTE_BASE_NAME:
                    analog_values.append(causal[f"{spec.name}_analog_patch_accuracy"])
            for spec in composition_benchmark_specs():
                if spec.name == DEFAULT_MULTI_ROUTE_BASE_NAME:
                    continue
                center_values.append(causal[f"{spec.name}_center_patch_accuracy"])
                removed_drop_values.append(
                    causal[f"{spec.name}_subspace_removed_accuracy_drop"]
                )
        analog_mean, analog_std = mean_std(analog_values)
        center_mean, center_std = mean_std(center_values)
        removed_mean, removed_std = mean_std(removed_drop_values)
        print(
            f"{policy:<18} "
            f"{analog_mean:.3f}+/-{analog_std:.3f}  "
            f"{center_mean:.3f}+/-{center_std:.3f}  "
            f"{removed_mean:.3f}+/-{removed_std:.3f}"
        )

    print("\nIterative ADD closure on composition target: mean +/- std")
    print(
        "policy             direct        symbolic_2step  decoded_2step   "
        "latent_bridge   bridge_route_cos"
    )
    for policy in COMPOSITION_BENCHMARK_POLICIES:
        closures = [
            result["composition"][policy]["closure"]  # type: ignore[index]
            for result in results
        ]
        direct_values = [values["direct_accuracy"] for values in closures]
        symbolic_values = [values["symbolic_two_step_accuracy"] for values in closures]
        decoded_values = [values["decoded_two_step_accuracy"] for values in closures]
        bridge_values = [values["latent_bridge_test_accuracy"] for values in closures]
        bridge_cos_values = [values["latent_bridge_route_h_cosine"] for values in closures]
        direct_mean, direct_std = mean_std(direct_values)
        symbolic_mean, symbolic_std = mean_std(symbolic_values)
        decoded_mean, decoded_std = mean_std(decoded_values)
        bridge_mean, bridge_std = mean_std(bridge_values)
        bridge_cos_mean, bridge_cos_std = mean_std(bridge_cos_values)
        print(
            f"{policy:<18} "
            f"{direct_mean:.3f}+/-{direct_std:.3f}  "
            f"{symbolic_mean:.3f}+/-{symbolic_std:.3f}  "
            f"{decoded_mean:.3f}+/-{decoded_std:.3f}  "
            f"{bridge_mean:.3f}+/-{bridge_std:.3f}  "
            f"{bridge_cos_mean:.3f}+/-{bridge_cos_std:.3f}"
        )

    print("\nForced class-align minus admission policy: mean +/- std")
    comparison_names = tuple(
        composition_benchmark_policy_comparison(
            results[0]["composition"]  # type: ignore[arg-type]
        ).keys()
    )
    for comparison_name in comparison_names:
        values = [
            composition_benchmark_policy_comparison(
                result["composition"]  # type: ignore[arg-type]
            )[comparison_name]
            for result in results
        ]
        value_mean, value_std = mean_std(values)
        print(f"{comparison_name}={value_mean:.6f}+/-{value_std:.6f}")


def run_composition_benchmark_multi_seed(
    seed_count: int,
    hidden_dim: int,
    alignment_weight: float,
) -> None:
    results: list[dict[str, object]] = []
    for seed in range(seed_count):
        print(f"running_seed={seed}")
        metrics = run_composition_benchmark_metrics(
            seed,
            hidden_dim,
            alignment_weight,
            verbose=False,
        )
        results.append({"seed": seed, "composition": metrics})
    print_composition_benchmark_summary(results)


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


# =====================================================================
# CLOSED LATENT ADD MODEL AND EXPERIMENTS (PHASES 1-4)
# =====================================================================

class AdamOptimizer:
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])
            g = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class ClosedLatentModel:
    E: np.ndarray        # (num_digits, code_dim)
    W_D: np.ndarray      # (code_dim, num_digits)
    b_D: np.ndarray      # (num_digits,)
    W_op1: np.ndarray    # (2 * code_dim, hidden_dim)
    b_op1: np.ndarray    # (hidden_dim,)
    W_op2: np.ndarray    # (hidden_dim, code_dim)
    b_op2: np.ndarray    # (code_dim,)


@dataclass
class ClosedLatentRouter:
    W_R: np.ndarray      # (input_dim, 2 * code_dim)
    b_R: np.ndarray      # (2 * code_dim,)


def make_closed_latent_model(
    num_digits: int,
    code_dim: int,
    hidden_dim: int,
    seed: int,
) -> ClosedLatentModel:
    rng = np.random.default_rng(seed)
    return ClosedLatentModel(
        E=rng.normal(0.0, 1.0, (num_digits, code_dim)),
        W_D=rng.normal(0.0, 1.0 / np.sqrt(code_dim), (code_dim, num_digits)),
        b_D=np.zeros(num_digits),
        W_op1=rng.normal(0.0, 1.0 / np.sqrt(2 * code_dim), (2 * code_dim, hidden_dim)),
        b_op1=np.full(hidden_dim, 0.01),  # tiny positive bias to prevent dead ReLUs
        W_op2=rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, code_dim)),
        b_op2=np.zeros(code_dim),
    )


def train_autoencoder(
    model,
    epochs: int,
    lr: float,
    margin: float = 2.0,
    sep_weight: float = 0.5,
    quiet: bool = True,
) -> None:
    num_digits = model.E.shape[0]
    x = np.eye(num_digits)
    y = np.arange(num_digits)
    params = {
        "E": model.E,
        "W_D": model.W_D,
        "b_D": model.b_D,
    }
    opt = AdamOptimizer(lr=lr)
    for epoch in range(1, epochs + 1):
        c = x @ model.E
        logits = c @ model.W_D + model.b_D
        ce_loss, dlogits = loss_and_grad_logits(logits, y)
        
        # Separation loss and grads
        sep_loss = 0.0
        dE_sep = np.zeros_like(model.E)
        for i in range(num_digits):
            for j in range(num_digits):
                if i == j:
                    continue
                diff = model.E[i] - model.E[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < margin:
                    sep_loss += (margin - dist_sq)
                    dE_sep[i] += -2.0 * diff
        
        dW_D = c.T @ dlogits
        db_D = np.sum(dlogits, axis=0)
        dc = dlogits @ model.W_D.T
        dE_ce = x.T @ dc
        
        grads = {
            "E": dE_ce + sep_weight * dE_sep,
            "W_D": dW_D,
            "b_D": db_D,
        }
        opt.update(params, grads)
        
        if not quiet and (epoch == 1 or epoch % 100 == 0 or epoch == epochs):
            preds = np.argmax(logits, axis=1)
            accuracy = np.mean(preds == y)
            print(f"  [AE] epoch={epoch:04d} loss={ce_loss:.4f} + sep={sep_loss:.4f} accuracy={accuracy:.3f}")


def train_closed_op(
    model: ClosedLatentModel,
    epochs: int,
    lr: float,
    lambda_closure: float,
    unfreeze_AE: bool = False,
    quiet: bool = True,
) -> None:
    num_digits = model.E.shape[0]
    code_dim = model.E.shape[1]
    a_list, b_list, y_list = [], [], []
    for a in range(num_digits):
        for b in range(num_digits):
            a_list.append(a)
            b_list.append(b)
            y_list.append((a + b) % num_digits)
    a_batch = np.array(a_list)
    b_batch = np.array(b_list)
    y_batch = np.array(y_list)
    
    params = {
        "W_op1": model.W_op1,
        "b_op1": model.b_op1,
        "W_op2": model.W_op2,
        "b_op2": model.b_op2,
    }
    if unfreeze_AE:
        params["E"] = model.E
        params["W_D"] = model.W_D
        params["b_D"] = model.b_D
        
    opt = AdamOptimizer(lr=lr)
    
    for epoch in range(1, epochs + 1):
        c_a = model.E[a_batch]
        c_b = model.E[b_batch]
        concatenated = np.concatenate([c_a, c_b], axis=1)
        z_op1 = concatenated @ model.W_op1 + model.b_op1
        h_op1 = np.maximum(z_op1, 0.0)
        c_sum = h_op1 @ model.W_op2 + model.b_op2
        logits = c_sum @ model.W_D + model.b_D
        
        ce_loss, dlogits = loss_and_grad_logits(logits, y_batch)
        
        target_code = model.E[y_batch]
        diff_closure = c_sum - target_code
        closure_loss = np.mean(np.sum(diff_closure**2, axis=1))
        
        # Gradients
        dc_sum_ce = dlogits @ model.W_D.T
        dc_sum_closure = 2.0 * diff_closure / len(y_batch)
        dc_sum = dc_sum_ce + lambda_closure * dc_sum_closure
        
        dW_op2 = h_op1.T @ dc_sum
        db_op2 = np.sum(dc_sum, axis=0)
        dh_op1 = dc_sum @ model.W_op2.T
        dz_op1 = dh_op1 * (z_op1 > 0.0)
        dW_op1 = concatenated.T @ dz_op1
        db_op1 = np.sum(dz_op1, axis=0)
        
        grads = {
            "W_op1": dW_op1,
            "b_op1": db_op1,
            "W_op2": dW_op2,
            "b_op2": db_op2,
        }
        if unfreeze_AE:
            dW_D = c_sum.T @ dlogits
            db_D = np.sum(dlogits, axis=0)
            grads["W_D"] = dW_D
            grads["b_D"] = db_D
            
            dconcatenated = dz_op1 @ model.W_op1.T
            dc_a = dconcatenated[:, :code_dim]
            dc_b = dconcatenated[:, code_dim:]
            
            dE = np.zeros_like(model.E)
            np.add.at(dE, a_batch, dc_a)
            np.add.at(dE, b_batch, dc_b)
            dc_target_closure = -2.0 * lambda_closure * diff_closure / len(y_batch)
            np.add.at(dE, y_batch, dc_target_closure)
            grads["E"] = dE
            
        opt.update(params, grads)
        
        if not quiet and (epoch == 1 or epoch % 200 == 0 or epoch == epochs):
            preds = np.argmax(logits, axis=1)
            accuracy = np.mean(preds == y_batch)
            print(f"  [Op] epoch={epoch:04d} ce={ce_loss:.4f} closure={closure_loss:.4f} accuracy={accuracy:.3f}")


def evaluate_closed_add(model: ClosedLatentModel) -> float:
    num_digits = model.E.shape[0]
    a_list, b_list, y_list = [], [], []
    for a in range(num_digits):
        for b in range(num_digits):
            a_list.append(a)
            b_list.append(b)
            y_list.append((a + b) % num_digits)
    a_batch = np.array(a_list)
    b_batch = np.array(b_list)
    y_batch = np.array(y_list)
    
    c_a = model.E[a_batch]
    c_b = model.E[b_batch]
    concatenated = np.concatenate([c_a, c_b], axis=1)
    z_op1 = concatenated @ model.W_op1 + model.b_op1
    h_op1 = np.maximum(z_op1, 0.0)
    c_sum = h_op1 @ model.W_op2 + model.b_op2
    logits = c_sum @ model.W_D + model.b_D
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_batch))


def evaluate_closed_add_metrics(model: ClosedLatentModel) -> tuple[float, float, float]:
    num_digits = model.E.shape[0]
    a_list, b_list, y_list = [], [], []
    for a in range(num_digits):
        for b in range(num_digits):
            a_list.append(a)
            b_list.append(b)
            y_list.append((a + b) % num_digits)
    a_batch = np.array(a_list)
    b_batch = np.array(b_list)
    y_batch = np.array(y_list)
    
    c_a = model.E[a_batch]
    c_b = model.E[b_batch]
    concatenated = np.concatenate([c_a, c_b], axis=1)
    z_op1 = concatenated @ model.W_op1 + model.b_op1
    h_op1 = np.maximum(z_op1, 0.0)
    c_sum = h_op1 @ model.W_op2 + model.b_op2
    
    # 2-operand acc (via decoder)
    logits = c_sum @ model.W_D + model.b_D
    preds = np.argmax(logits, axis=1)
    two_op_acc = float(np.mean(preds == y_batch))
    
    # code MSE
    target_code = model.E[y_batch]
    code_mse = float(np.mean(np.sum((c_sum - target_code)**2, axis=1)))
    
    # nearest-code acc
    dists = np.sum((c_sum[:, np.newaxis, :] - model.E[np.newaxis, :, :])**2, axis=2)
    nearest_preds = np.argmin(dists, axis=1)
    nearest_acc = float(np.mean(nearest_preds == y_batch))
    
    return two_op_acc, code_mse, nearest_acc


def evaluate_closed_sum012(model: ClosedLatentModel) -> float:
    num_digits = model.E.shape[0]
    d0_list, d1_list, d2_list, y_list = [], [], [], []
    for d0 in range(num_digits):
        for d1 in range(num_digits):
            for d2 in range(num_digits):
                d0_list.append(d0)
                d1_list.append(d1)
                d2_list.append(d2)
                y_list.append((d0 + d1 + d2) % num_digits)
    d0_batch = np.array(d0_list)
    d1_batch = np.array(d1_list)
    d2_batch = np.array(d2_list)
    y_batch = np.array(y_list)
    
    c0 = model.E[d0_batch]
    c1 = model.E[d1_batch]
    c2 = model.E[d2_batch]
    
    # Step 1: c01 = F(E(d0), E(d1))
    concat1 = np.concatenate([c0, c1], axis=1)
    z1 = concat1 @ model.W_op1 + model.b_op1
    h1 = np.maximum(z1, 0.0)
    c01 = h1 @ model.W_op2 + model.b_op2
    
    # Step 2: c012 = F(c01, E(d2))
    concat2 = np.concatenate([c01, c2], axis=1)
    z2 = concat2 @ model.W_op1 + model.b_op1
    h2 = np.maximum(z2, 0.0)
    c012 = h2 @ model.W_op2 + model.b_op2
    
    # Decode
    logits = c012 @ model.W_D + model.b_D
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_batch))


def train_closed_latent_router(
    model: ClosedLatentModel,
    router: ClosedLatentRouter,
    digit_x: np.ndarray,
    y: np.ndarray,
    operand_positions: tuple[int, int],
    epochs: int,
    lr: float,
    alignment_weight: float,
    unfreeze_all: bool = False,
    quiet: bool = True,
) -> None:
    B = digit_x.shape[0]
    code_dim = model.E.shape[1]
    
    # Extract inputs and decode them for alignment targets
    d0 = np.argmax(digit_x[:, 0:5], axis=1)
    d1 = np.argmax(digit_x[:, 5:10], axis=1)
    d2 = np.argmax(digit_x[:, 10:15], axis=1)
    digits_by_pos = [d0, d1, d2]
    
    pos_a, pos_b = operand_positions
    target_a_digits = digits_by_pos[pos_a]
    target_b_digits = digits_by_pos[pos_b]
    
    target_a = model.E[target_a_digits]
    target_b = model.E[target_b_digits]
    
    params = {
        "W_R": router.W_R,
        "b_R": router.b_R,
    }
    if unfreeze_all:
        params["W_op1"] = model.W_op1
        params["b_op1"] = model.b_op1
        params["W_op2"] = model.W_op2
        params["b_op2"] = model.b_op2
        params["E"] = model.E
        params["W_D"] = model.W_D
        params["b_D"] = model.b_D
        
    opt = AdamOptimizer(lr=lr)
    
    for epoch in range(1, epochs + 1):
        if unfreeze_all:
            target_a = model.E[target_a_digits]
            target_b = model.E[target_b_digits]
            
        route_out = digit_x @ router.W_R + router.b_R
        c_a = route_out[:, :code_dim]
        c_b = route_out[:, code_dim:]
        
        concatenated = np.concatenate([c_a, c_b], axis=1)
        z_op1 = concatenated @ model.W_op1 + model.b_op1
        h_op1 = np.maximum(z_op1, 0.0)
        c_sum = h_op1 @ model.W_op2 + model.b_op2
        logits = c_sum @ model.W_D + model.b_D
        
        ce_loss, dlogits = loss_and_grad_logits(logits, y)
        
        align_loss_a = np.mean(np.sum((c_a - target_a)**2, axis=1))
        align_loss_b = np.mean(np.sum((c_b - target_b)**2, axis=1))
        align_loss = align_loss_a + align_loss_b
        
        # Backprop through F_add (frozen or unfrozen)
        dc_sum = dlogits @ model.W_D.T
        dh_op1 = dc_sum @ model.W_op2.T
        dz_op1 = dh_op1 * (z_op1 > 0.0)
        dconcatenated = dz_op1 @ model.W_op1.T
        
        dc_a_ce = dconcatenated[:, :code_dim]
        dc_b_ce = dconcatenated[:, code_dim:]
        
        dc_a_align = 2.0 * (c_a - target_a) / B
        dc_b_align = 2.0 * (c_b - target_b) / B
        
        dc_a = dc_a_ce + alignment_weight * dc_a_align
        dc_b = dc_b_ce + alignment_weight * dc_b_align
        droute_out = np.concatenate([dc_a, dc_b], axis=1)
        
        dW_R = digit_x.T @ droute_out
        db_R = np.sum(droute_out, axis=0)
        
        grads = {
            "W_R": dW_R,
            "b_R": db_R,
        }
        
        if unfreeze_all:
            dW_op2 = h_op1.T @ dc_sum
            db_op2 = np.sum(dc_sum, axis=0)
            dW_op1 = concatenated.T @ dz_op1
            db_op1 = np.sum(dz_op1, axis=0)
            dW_D = c_sum.T @ dlogits
            db_D = np.sum(dlogits, axis=0)
            
            grads["W_op2"] = dW_op2
            grads["b_op2"] = db_op2
            grads["W_op1"] = dW_op1
            grads["b_op1"] = db_op1
            grads["W_D"] = dW_D
            grads["b_D"] = db_D
            
            dE = np.zeros_like(model.E)
            np.add.at(dE, target_a_digits, -2.0 * alignment_weight * (c_a - target_a) / B)
            np.add.at(dE, target_b_digits, -2.0 * alignment_weight * (c_b - target_b) / B)
            grads["E"] = dE
            
        opt.update(params, grads)
        
        if not quiet and (epoch == 1 or epoch % 200 == 0 or epoch == epochs):
            preds = np.argmax(logits, axis=1)
            accuracy = np.mean(preds == y)
            print(f"    [Router] epoch={epoch:04d} ce={ce_loss:.4f} align={align_loss:.4f} accuracy={accuracy:.3f}")


def make_combinations(k: int, num_digits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    grids = np.meshgrid(*[np.arange(num_digits) for _ in range(k)], indexing="ij")
    digits = np.stack(grids, axis=-1).reshape(-1, k)
    sums = np.sum(digits, axis=1) % num_digits
    return digits, sums


def evaluate_long_chain(
    model: ClosedLatentModel,
    digits: np.ndarray,
    sums: np.ndarray,
) -> tuple[float, float, float]:
    c = model.E[digits[:, 0]]
    for i in range(1, digits.shape[1]):
        c_next = model.E[digits[:, i]]
        concat = np.concatenate([c, c_next], axis=1)
        z = concat @ model.W_op1 + model.b_op1
        h = np.maximum(z, 0.0)
        c = h @ model.W_op2 + model.b_op2
        
    logits = c @ model.W_D + model.b_D
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == sums))
    
    target_code = model.E[sums]
    gt_dist = float(np.mean(np.sum((c - target_code)**2, axis=1)))
    
    dists = np.sum((c[:, np.newaxis, :] - model.E[np.newaxis, :, :])**2, axis=2)
    nearest_dist = float(np.mean(np.min(dists, axis=1)))
    
    return acc, gt_dist, nearest_dist


def run_closure_ablation(
    seed_count: int,
    alignment_weight: float,
) -> None:
    lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]
    hidden_dim = 16
    code_dim = 4
    
    print("\n=================================================================")
    print("RUNNING CLOSURE LOSS ABLATION (LAMBDA SWEEP)")
    print(f"Seeds: {seed_count}, Hidden Dim: {hidden_dim}, Code Dim: {code_dim}")
    print("=================================================================")
    
    results = {}
    for lam in lambdas:
        two_op_accs = []
        sum012_accs = []
        code_mses = []
        nearest_accs = []
        
        for seed in range(seed_count):
            model = make_closed_latent_model(
                num_digits=5,
                code_dim=code_dim,
                hidden_dim=hidden_dim,
                seed=seed,
            )
            # Phase 1: AE
            train_autoencoder(model, epochs=600, lr=0.01, margin=2.0, sep_weight=0.5, quiet=True)
            # Phase 2: Operator
            train_closed_op(model, epochs=1500, lr=0.01, lambda_closure=lam, quiet=True)
            
            two_op_acc, code_mse, nearest_acc = evaluate_closed_add_metrics(model)
            sum012_acc = evaluate_closed_sum012(model)
            
            two_op_accs.append(two_op_acc)
            sum012_accs.append(sum012_acc)
            code_mses.append(code_mse)
            nearest_accs.append(nearest_acc)
            
        results[lam] = {
            "two_op": mean_std(two_op_accs),
            "sum012": mean_std(sum012_accs),
            "mse": mean_std(code_mses),
            "nearest": mean_std(nearest_accs),
        }
        
    print("\n=================================================================")
    print("CLOSURE LOSS ABLATION RESULTS (SWEEP OVER LAMBDA)")
    print("=================================================================")
    print("lambda   2-operand acc    iterative SUM012  code MSE to E(tgt) nearest-code acc")
    for lam in lambdas:
        r = results[lam]
        t_m, t_s = r["two_op"]
        s_m, s_s = r["sum012"]
        m_m, m_s = r["mse"]
        n_m, n_s = r["nearest"]
        print(
            f"{lam:<8} "
            f"{t_m:.3f}+/-{t_s:.3f}    "
            f"{s_m:.3f}+/-{s_s:.3f}    "
            f"{m_m:.3f}+/-{m_s:.3f}    "
            f"{n_m:.3f}+/-{n_s:.3f}"
        )
    print("=================================================================\n")


def run_freeze_ablation(
    seed_count: int,
    alignment_weight: float,
) -> None:
    hidden_dim = 16
    code_dim = 4
    
    print("\n=================================================================")
    print("RUNNING FREEZE VS UNFREEZE ABLATION")
    print(f"Seeds: {seed_count}, Hidden Dim: {hidden_dim}, Code Dim: {code_dim}")
    print("=================================================================")
    
    regimens = ["A (Fully Frozen)", "B (Unfreeze AE in Phase 2)", "C (Unfreeze all in Phase 3)"]
    results = {reg: {
        "two_op": [], "sum012": [], "routed_sum012": [],
        "router01": [], "router12": [], "router02": []
    } for reg in regimens}
    
    for seed in range(seed_count):
        model_a = make_closed_latent_model(5, code_dim, hidden_dim, seed)
        model_b = make_closed_latent_model(5, code_dim, hidden_dim, seed)
        model_c = make_closed_latent_model(5, code_dim, hidden_dim, seed)
        
        train_autoencoder(model_a, epochs=600, lr=0.01, margin=2.0, sep_weight=0.5, quiet=True)
        model_b.E = model_a.E.copy()
        model_b.W_D = model_a.W_D.copy()
        model_b.b_D = model_a.b_D.copy()
        model_c.E = model_a.E.copy()
        model_c.W_D = model_a.W_D.copy()
        model_c.b_D = model_a.b_D.copy()
        
        # Train operator
        train_closed_op(model_a, epochs=1500, lr=0.01, lambda_closure=10.0, unfreeze_AE=False, quiet=True)
        two_op_a = evaluate_closed_add(model_a)
        sum012_a = evaluate_closed_sum012(model_a)
        
        train_closed_op(model_b, epochs=1500, lr=0.01, lambda_closure=10.0, unfreeze_AE=True, quiet=True)
        two_op_b = evaluate_closed_add(model_b)
        sum012_b = evaluate_closed_sum012(model_b)
        
        train_closed_op(model_c, epochs=1500, lr=0.01, lambda_closure=10.0, unfreeze_AE=False, quiet=True)
        two_op_c = evaluate_closed_add(model_c)
        sum012_c = evaluate_closed_sum012(model_c)
        
        specs = (
            MultiRouteOpSpec("ADD01", "add", (0, 1)),
            MultiRouteOpSpec("ADD12", "add", (1, 2)),
            MultiRouteOpSpec("ADD02", "add", (0, 2)),
        )
        datasets = {
            spec.name: make_multi_route_dataset(spec, num_digits=5, sequence_digits=3)
            for spec in specs
        }
        
        routers_a = {
            spec.name: ClosedLatentRouter(
                W_R=np.random.default_rng(seed).normal(0.0, 1.0 / np.sqrt(15), (15, 2 * code_dim)),
                b_R=np.zeros(2 * code_dim),
            ) for spec in specs
        }
        routers_b = {
            spec.name: ClosedLatentRouter(
                W_R=np.random.default_rng(seed).normal(0.0, 1.0 / np.sqrt(15), (15, 2 * code_dim)),
                b_R=np.zeros(2 * code_dim),
            ) for spec in specs
        }
        routers_c = {
            spec.name: ClosedLatentRouter(
                W_R=np.random.default_rng(seed).normal(0.0, 1.0 / np.sqrt(15), (15, 2 * code_dim)),
                b_R=np.zeros(2 * code_dim),
            ) for spec in specs
        }
        
        train_closed_latent_router(model_a, routers_a["ADD01"], datasets["ADD01"].digit_x, datasets["ADD01"].y, (0, 1), 800, 0.01, alignment_weight, False, True)
        train_closed_latent_router(model_a, routers_a["ADD12"], datasets["ADD12"].digit_x, datasets["ADD12"].y, (1, 2), 800, 0.01, alignment_weight, False, True)
        train_closed_latent_router(model_a, routers_a["ADD02"], datasets["ADD02"].digit_x, datasets["ADD02"].y, (0, 2), 800, 0.01, alignment_weight, False, True)
        
        train_closed_latent_router(model_b, routers_b["ADD01"], datasets["ADD01"].digit_x, datasets["ADD01"].y, (0, 1), 800, 0.01, alignment_weight, False, True)
        train_closed_latent_router(model_b, routers_b["ADD12"], datasets["ADD12"].digit_x, datasets["ADD12"].y, (1, 2), 800, 0.01, alignment_weight, False, True)
        train_closed_latent_router(model_b, routers_b["ADD02"], datasets["ADD02"].digit_x, datasets["ADD02"].y, (0, 2), 800, 0.01, alignment_weight, False, True)
        
        train_closed_latent_router(model_c, routers_c["ADD01"], datasets["ADD01"].digit_x, datasets["ADD01"].y, (0, 1), 800, 0.01, alignment_weight, True, True)
        train_closed_latent_router(model_c, routers_c["ADD12"], datasets["ADD12"].digit_x, datasets["ADD12"].y, (1, 2), 800, 0.01, alignment_weight, True, True)
        train_closed_latent_router(model_c, routers_c["ADD02"], datasets["ADD02"].digit_x, datasets["ADD02"].y, (0, 2), 800, 0.01, alignment_weight, True, True)
        
        def eval_regimen(model_reg, routers_reg):
            accs = {}
            for name, spec in zip(["ADD01", "ADD12", "ADD02"], specs):
                ds = datasets[name]
                route_out = ds.digit_x @ routers_reg[name].W_R + routers_reg[name].b_R
                c_a = route_out[:, :code_dim]
                c_b = route_out[:, code_dim:]
                concat = np.concatenate([c_a, c_b], axis=1)
                z = concat @ model_reg.W_op1 + model_reg.b_op1
                h = np.maximum(z, 0.0)
                c_sum = h @ model_reg.W_op2 + model_reg.b_op2
                logits = c_sum @ model_reg.W_D + model_reg.b_D
                preds = np.argmax(logits, axis=1)
                accs[name] = float(np.mean(preds == ds.y))
                
            ds_sum = make_multi_route_dataset(MultiRouteOpSpec("SUM012", "add", (0, 1, 2)), num_digits=5, sequence_digits=3)
            route_out01 = ds_sum.digit_x @ routers_reg["ADD01"].W_R + routers_reg["ADD01"].b_R
            c0 = route_out01[:, :code_dim]
            c1 = route_out01[:, code_dim:]
            route_out12 = ds_sum.digit_x @ routers_reg["ADD12"].W_R + routers_reg["ADD12"].b_R
            c2 = route_out12[:, code_dim:]
            
            concat1 = np.concatenate([c0, c1], axis=1)
            z1 = concat1 @ model_reg.W_op1 + model_reg.b_op1
            h1 = np.maximum(z1, 0.0)
            c01 = h1 @ model_reg.W_op2 + model_reg.b_op2
            
            concat2 = np.concatenate([c01, c2], axis=1)
            z2 = concat2 @ model_reg.W_op1 + model_reg.b_op1
            h2 = np.maximum(z2, 0.0)
            c012 = h2 @ model_reg.W_op2 + model_reg.b_op2
            
            logits = c012 @ model_reg.W_D + model_reg.b_D
            preds = np.argmax(logits, axis=1)
            accs["SUM012"] = float(np.mean(preds == ds_sum.y))
            return accs
            
        accs_a = eval_regimen(model_a, routers_a)
        accs_b = eval_regimen(model_b, routers_b)
        accs_c = eval_regimen(model_c, routers_c)
        
        results["A (Fully Frozen)"]["two_op"].append(two_op_a)
        results["A (Fully Frozen)"]["sum012"].append(sum012_a)
        results["A (Fully Frozen)"]["routed_sum012"].append(accs_a["SUM012"])
        results["A (Fully Frozen)"]["router01"].append(accs_a["ADD01"])
        results["A (Fully Frozen)"]["router12"].append(accs_a["ADD12"])
        results["A (Fully Frozen)"]["router02"].append(accs_a["ADD02"])
        
        results["B (Unfreeze AE in Phase 2)"]["two_op"].append(two_op_b)
        results["B (Unfreeze AE in Phase 2)"]["sum012"].append(sum012_b)
        results["B (Unfreeze AE in Phase 2)"]["routed_sum012"].append(accs_b["SUM012"])
        results["B (Unfreeze AE in Phase 2)"]["router01"].append(accs_b["ADD01"])
        results["B (Unfreeze AE in Phase 2)"]["router12"].append(accs_b["ADD12"])
        results["B (Unfreeze AE in Phase 2)"]["router02"].append(accs_b["ADD02"])
        
        results["C (Unfreeze all in Phase 3)"]["two_op"].append(two_op_c)
        results["C (Unfreeze all in Phase 3)"]["sum012"].append(sum012_c)
        results["C (Unfreeze all in Phase 3)"]["routed_sum012"].append(accs_c["SUM012"])
        results["C (Unfreeze all in Phase 3)"]["router01"].append(accs_c["ADD01"])
        results["C (Unfreeze all in Phase 3)"]["router12"].append(accs_c["ADD12"])
        results["C (Unfreeze all in Phase 3)"]["router02"].append(accs_c["ADD02"])
        
    print("\n=================================================================")
    print("FREEZE VS UNFREEZE ABLATION SUMMARY")
    print("=================================================================")
    for reg in regimens:
        print(f"\nRegimen: {reg}")
        print("-" * 50)
        two_m, two_s = mean_std(results[reg]["two_op"])
        sum_m, sum_s = mean_std(results[reg]["sum012"])
        rsum_m, rsum_s = mean_std(results[reg]["routed_sum012"])
        r01_m, r01_s = mean_std(results[reg]["router01"])
        r12_m, r12_s = mean_std(results[reg]["router12"])
        r02_m, r02_s = mean_std(results[reg]["router02"])
        print(f"  2-operand ADD Acc           : {two_m:.3f}+/-{two_s:.3f}")
        print(f"  Iterative SUM012 Acc (Unrt) : {sum_m:.3f}+/-{sum_s:.3f}")
        print(f"  ADD01 Router Acc            : {r01_m:.3f}+/-{r01_s:.3f}")
        print(f"  ADD12 Router Acc            : {r12_m:.3f}+/-{r12_s:.3f}")
        print(f"  ADD02 Router Acc            : {r02_m:.3f}+/-{r02_s:.3f}")
        print(f"  Routed SUM012 Acc           : {rsum_m:.3f}+/-{rsum_s:.3f}")
    print("=================================================================\n")


def run_long_composition_test(
    seed_count: int,
    alignment_weight: float,
) -> None:
    hidden_dims = (64, 16, 8)
    
    print("\n=================================================================")
    print("RUNNING LONG COMPOSITION & MANIFOLD DRIFT TEST")
    print(f"Seeds: {seed_count}, Alignment Weight: {alignment_weight}")
    print("=================================================================")
    
    results = {}
    for hidden_dim in hidden_dims:
        code_dim = max(4, hidden_dim // 4)
        results[hidden_dim] = {k: {"acc": [], "gt_dist": [], "manifold_dist": []} for k in (2, 3, 4, 5)}
        
        for seed in range(seed_count):
            model = make_closed_latent_model(
                num_digits=5,
                code_dim=code_dim,
                hidden_dim=hidden_dim,
                seed=seed,
            )
            train_autoencoder(model, epochs=600, lr=0.01, margin=2.0, sep_weight=0.5, quiet=True)
            train_closed_op(model, epochs=1500, lr=0.01, lambda_closure=10.0, quiet=True)
            
            for k in (2, 3, 4, 5):
                digits, sums = make_combinations(k, num_digits=5)
                acc, gt_dist, manifold_dist = evaluate_long_chain(model, digits, sums)
                results[hidden_dim][k]["acc"].append(acc)
                results[hidden_dim][k]["gt_dist"].append(gt_dist)
                results[hidden_dim][k]["manifold_dist"].append(manifold_dist)
                
    print("\n=================================================================")
    print("LONG COMPOSITION & DRIFT RESULTS SUMMARY (MEAN +/- STD)")
    print("=================================================================")
    for hidden_dim in hidden_dims:
        print(f"\nHidden Dim: {hidden_dim} (Code Dim: {max(4, hidden_dim // 4)})")
        print("-" * 75)
        print("Operands  Calls  Classification Acc  Distance to Target  Distance to Manifold")
        for k in (2, 3, 4, 5):
            r = results[hidden_dim][k]
            a_m, a_s = mean_std(r["acc"])
            g_m, g_s = mean_std(r["gt_dist"])
            m_m, m_s = mean_std(r["manifold_dist"])
            print(
                f"{k:<9} "
                f"{k-1:<6} "
                f"{a_m:.3f}+/-{a_s:.3f}         "
                f"{g_m:.3f}+/-{g_s:.3f}          "
                f"{m_m:.3f}+/-{m_s:.3f}"
            )
    print("=================================================================\n")


def run_closed_latent_benchmark(
    seed_count: int,
    alignment_weight: float,
    verbose: bool,
) -> None:
    hidden_dims = (64, 16, 8)
    
    print("\n=================================================================")
    print("RUNNING CLOSED LATENT ADD EXPERIMENT LADDER")
    print(f"Seeds: {seed_count}, Alignment Weight: {alignment_weight}")
    print("=================================================================")
    
    final_rows = []
    
    for hidden_dim in hidden_dims:
        code_dim = max(4, hidden_dim // 4)
        
        direct_accs = []
        bridge_accs = []
        closed_accs = []
        closed_add_accs = []
        
        router_add01_accs = []
        router_add12_accs = []
        router_add02_accs = []
        router_sum012_accs = []
        
        for seed in range(seed_count):
            if verbose:
                print(f"\n--- Running hidden_dim={hidden_dim}, code_dim={code_dim}, seed={seed} ---")
                
            baseline_results = composition_benchmark_metrics(
                seed=seed,
                hidden_dim=hidden_dim,
                alignment_weight=alignment_weight,
                policy="admission",
                verbose=False,
            )
            direct_acc = baseline_results["closure"]["direct_accuracy"]
            bridge_acc = baseline_results["closure"]["latent_bridge_test_accuracy"]
            direct_accs.append(direct_acc)
            bridge_accs.append(bridge_acc)
            
            model = make_closed_latent_model(
                num_digits=5,
                code_dim=code_dim,
                hidden_dim=hidden_dim,
                seed=seed,
            )
            
            # Phase 1: Pretrain Autoencoder
            train_autoencoder(model, epochs=600, lr=0.01, margin=2.0, sep_weight=0.5, quiet=not verbose)
            
            # Phase 2: Train addition MLP (Closed Latent ADD)
            train_closed_op(model, epochs=1500, lr=0.01, lambda_closure=10.0, quiet=not verbose)
            
            # Evaluate Two-Operand (Experiment 1)
            two_op_acc = evaluate_closed_add(model)
            closed_add_accs.append(two_op_acc)
            
            # Evaluate Iterative SUM012 (Experiment 2)
            closed_sum012_acc = evaluate_closed_sum012(model)
            closed_accs.append(closed_sum012_acc)
            
            # Phase 4: Sequential route learning
            specs = (
                MultiRouteOpSpec("ADD01", "add", (0, 1)),
                MultiRouteOpSpec("ADD12", "add", (1, 2)),
                MultiRouteOpSpec("ADD02", "add", (0, 2)),
            )
            datasets = {
                spec.name: make_multi_route_dataset(spec, num_digits=5, sequence_digits=3)
                for spec in specs
            }
            
            routers = {
                spec.name: ClosedLatentRouter(
                    W_R=np.random.default_rng(seed).normal(0.0, 1.0 / np.sqrt(15), (15, 2 * code_dim)),
                    b_R=np.zeros(2 * code_dim),
                )
                for spec in specs
            }
            
            # Train routers sequentially
            train_closed_latent_router(
                model, routers["ADD01"], datasets["ADD01"].digit_x, datasets["ADD01"].y,
                operand_positions=(0, 1), epochs=800, lr=0.01, alignment_weight=alignment_weight, quiet=not verbose
            )
            train_closed_latent_router(
                model, routers["ADD12"], datasets["ADD12"].digit_x, datasets["ADD12"].y,
                operand_positions=(1, 2), epochs=800, lr=0.01, alignment_weight=alignment_weight, quiet=not verbose
            )
            train_closed_latent_router(
                model, routers["ADD02"], datasets["ADD02"].digit_x, datasets["ADD02"].y,
                operand_positions=(0, 2), epochs=800, lr=0.01, alignment_weight=alignment_weight, quiet=not verbose
            )
            
            # Evaluate routers
            def eval_router(r_name, pos_a, pos_b):
                ds = datasets[r_name]
                route_out = ds.digit_x @ routers[r_name].W_R + routers[r_name].b_R
                c_a = route_out[:, :code_dim]
                c_b = route_out[:, code_dim:]
                concat = np.concatenate([c_a, c_b], axis=1)
                z = concat @ model.W_op1 + model.b_op1
                h = np.maximum(z, 0.0)
                c_sum = h @ model.W_op2 + model.b_op2
                logits = c_sum @ model.W_D + model.b_D
                preds = np.argmax(logits, axis=1)
                return float(np.mean(preds == ds.y))
                
            acc01 = eval_router("ADD01", 0, 1)
            acc12 = eval_router("ADD12", 1, 2)
            acc02 = eval_router("ADD02", 0, 2)
            router_add01_accs.append(acc01)
            router_add12_accs.append(acc12)
            router_add02_accs.append(acc02)
            
            # Evaluate iterative SUM012 with sequential routers
            ds_sum = make_multi_route_dataset(MultiRouteOpSpec("SUM012", "add", (0, 1, 2)), num_digits=5, sequence_digits=3)
            route_out01 = ds_sum.digit_x @ routers["ADD01"].W_R + routers["ADD01"].b_R
            c0 = route_out01[:, :code_dim]
            c1 = route_out01[:, code_dim:]
            
            route_out12 = ds_sum.digit_x @ routers["ADD12"].W_R + routers["ADD12"].b_R
            c2 = route_out12[:, code_dim:]
            
            concat1 = np.concatenate([c0, c1], axis=1)
            z1 = concat1 @ model.W_op1 + model.b_op1
            h1 = np.maximum(z1, 0.0)
            c01 = h1 @ model.W_op2 + model.b_op2
            
            concat2 = np.concatenate([c01, c2], axis=1)
            z2 = concat2 @ model.W_op1 + model.b_op1
            h2 = np.maximum(z2, 0.0)
            c012 = h2 @ model.W_op2 + model.b_op2
            
            logits = c012 @ model.W_D + model.b_D
            preds = np.argmax(logits, axis=1)
            routed_sum012_acc = float(np.mean(preds == ds_sum.y))
            router_sum012_accs.append(routed_sum012_acc)
            
            if verbose:
                print(f"  [Seed {seed}] 2-Op Acc={two_op_acc:.3f}, Closed SUM012={closed_sum012_acc:.3f}")
                print(f"  [Seed {seed}] Seq Router Accs: ADD01={acc01:.3f}, ADD12={acc12:.3f}, ADD02={acc02:.3f}")
                print(f"  [Seed {seed}] Routed SUM012 (Composition with sequential routers)={routed_sum012_acc:.3f}")
                
        d_mean, d_std = mean_std(direct_accs)
        b_mean, b_std = mean_std(bridge_accs)
        c_mean, c_std = mean_std(closed_accs)
        ca_mean, ca_std = mean_std(closed_add_accs)
        
        r01_mean, r01_std = mean_std(router_add01_accs)
        r12_mean, r12_std = mean_std(router_add12_accs)
        r02_mean, r02_std = mean_std(router_add02_accs)
        rsum_mean, rsum_std = mean_std(router_sum012_accs)
        
        final_rows.append({
            "hidden_dim": hidden_dim,
            "code_dim": code_dim,
            "direct": (d_mean, d_std),
            "bridge": (b_mean, b_std),
            "closed": (c_mean, c_std),
            "closed_add": (ca_mean, ca_std),
            "router_add01": (r01_mean, r01_std),
            "router_add12": (r12_mean, r12_std),
            "router_add02": (r02_mean, r02_std),
            "router_sum012": (rsum_mean, rsum_std),
        })

    print("\n=================================================================")
    print("CLOSED LATENT ADD COMPOSITION COMPARISON SUMMARY (MEAN +/- STD)")
    print("=================================================================")
    print(f"Setting: seeds={seed_count}")
    print("-----------------------------------------------------------------")
    print(
        "hidden_dim  code_dim  direct_acc     latent_bridge  closed_latent  "
        "closed_2operand"
    )
    for row in final_rows:
        d_mean, d_std = row["direct"]
        b_mean, b_std = row["bridge"]
        c_mean, c_std = row["closed"]
        ca_mean, ca_std = row["closed_add"]
        print(
            f"{row['hidden_dim']:<11} "
            f"{row['code_dim']:<9} "
            f"{d_mean:.3f}+/-{d_std:.3f}  "
            f"{b_mean:.3f}+/-{b_std:.3f}  "
            f"{c_mean:.3f}+/-{c_std:.3f}  "
            f"{ca_mean:.3f}+/-{ca_std:.3f}"
        )

    print("\n=================================================================")
    print("SEQUENTIAL ROUTING (CONTINUAL LEARNING) SUMMARY (MEAN +/- STD)")
    print("=================================================================")
    print("hidden_dim  ADD01_acc      ADD12_acc      ADD02_acc      Routed SUM012_acc")
    for row in final_rows:
        r01_m, r01_s = row["router_add01"]
        r12_m, r12_s = row["router_add12"]
        r02_m, r02_s = row["router_add02"]
        rsum_m, rsum_s = row["router_sum012"]
        print(
            f"{row['hidden_dim']:<11} "
            f"{r01_m:.3f}+/-{r01_s:.3f}  "
            f"{r12_m:.3f}+/-{r12_s:.3f}  "
            f"{r02_m:.3f}+/-{r02_s:.3f}  "
            f"{rsum_m:.3f}+/-{rsum_s:.3f}"
        )
    print("=================================================================\n")


class MultiOperatorLatentModel:
    def __init__(self, num_digits: int, code_dim: int, hidden_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        # Shared boundary
        self.E = rng.normal(0.0, 1.0, (num_digits, code_dim))
        self.W_D = rng.normal(0.0, 1.0 / np.sqrt(code_dim), (code_dim, num_digits))
        self.b_D = np.zeros(num_digits)
        
        # Addition operator MLP
        self.W_add1 = rng.normal(0.0, 1.0 / np.sqrt(2 * code_dim), (2 * code_dim, hidden_dim))
        self.b_add1 = np.full(hidden_dim, 0.01)  # tiny positive bias to prevent dead ReLUs
        self.W_add2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, code_dim))
        self.b_add2 = np.zeros(code_dim)
        
        # Maximum operator MLP
        self.W_max1 = rng.normal(0.0, 1.0 / np.sqrt(2 * code_dim), (2 * code_dim, hidden_dim))
        self.b_max1 = np.full(hidden_dim, 0.01)
        self.W_max2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, code_dim))
        self.b_max2 = np.zeros(code_dim)
        
        # Copy operator MLP
        self.W_copy1 = rng.normal(0.0, 1.0 / np.sqrt(code_dim), (code_dim, hidden_dim))
        self.b_copy1 = np.full(hidden_dim, 0.01)
        self.W_copy2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, code_dim))
        self.b_copy2 = np.zeros(code_dim)


def train_multi_closed_ops(
    model: MultiOperatorLatentModel,
    ops_to_train: list,
    epochs: int,
    lr: float,
    lambda_closure: float,
    quiet: bool = True,
) -> None:
    num_digits = model.E.shape[0]
    
    a_list, b_list = [], []
    for a in range(num_digits):
        for b in range(num_digits):
            a_list.append(a)
            b_list.append(b)
    ab_a = np.array(a_list)
    ab_b = np.array(b_list)
    copy_batch = np.arange(num_digits)
    
    params = {}
    if "add" in ops_to_train:
        params["W_add1"] = model.W_add1
        params["b_add1"] = model.b_add1
        params["W_add2"] = model.W_add2
        params["b_add2"] = model.b_add2
    if "max" in ops_to_train:
        params["W_max1"] = model.W_max1
        params["b_max1"] = model.b_max1
        params["W_max2"] = model.W_max2
        params["b_max2"] = model.b_max2
    if "copy" in ops_to_train:
        params["W_copy1"] = model.W_copy1
        params["b_copy1"] = model.b_copy1
        params["W_copy2"] = model.W_copy2
        params["b_copy2"] = model.b_copy2
        
    opt = AdamOptimizer(lr=lr)
    
    for epoch in range(1, epochs + 1):
        grads = {}
        
        if "add" in ops_to_train:
            c_a = model.E[ab_a]
            c_b = model.E[ab_b]
            concatenated = np.concatenate([c_a, c_b], axis=1)
            z1 = concatenated @ model.W_add1 + model.b_add1
            h1 = np.maximum(z1, 0.0)
            c_out = h1 @ model.W_add2 + model.b_add2
            logits = c_out @ model.W_D + model.b_D
            
            y_add = (ab_a + ab_b) % num_digits
            _, dlogits = loss_and_grad_logits(logits, y_add)
            
            target_code = model.E[y_add]
            diff_closure = c_out - target_code
            
            dc_out_ce = dlogits @ model.W_D.T
            dc_out_closure = 2.0 * diff_closure / len(y_add)
            dc_out = dc_out_ce + lambda_closure * dc_out_closure
            
            grads["W_add2"] = h1.T @ dc_out
            grads["b_add2"] = np.sum(dc_out, axis=0)
            dh1 = dc_out @ model.W_add2.T
            dz1 = dh1 * (z1 > 0.0)
            grads["W_add1"] = concatenated.T @ dz1
            grads["b_add1"] = np.sum(dz1, axis=0)
            
        if "max" in ops_to_train:
            c_a = model.E[ab_a]
            c_b = model.E[ab_b]
            concatenated = np.concatenate([c_a, c_b], axis=1)
            z1 = concatenated @ model.W_max1 + model.b_max1
            h1 = np.maximum(z1, 0.0)
            c_out = h1 @ model.W_max2 + model.b_max2
            logits = c_out @ model.W_D + model.b_D
            
            y_max = np.maximum(ab_a, ab_b)
            _, dlogits = loss_and_grad_logits(logits, y_max)
            
            target_code = model.E[y_max]
            diff_closure = c_out - target_code
            
            dc_out_ce = dlogits @ model.W_D.T
            dc_out_closure = 2.0 * diff_closure / len(y_max)
            dc_out = dc_out_ce + lambda_closure * dc_out_closure
            
            grads["W_max2"] = h1.T @ dc_out
            grads["b_max2"] = np.sum(dc_out, axis=0)
            dh1 = dc_out @ model.W_max2.T
            dz1 = dh1 * (z1 > 0.0)
            grads["W_max1"] = concatenated.T @ dz1
            grads["b_max1"] = np.sum(dz1, axis=0)
            
        if "copy" in ops_to_train:
            c_a = model.E[copy_batch]
            z1 = c_a @ model.W_copy1 + model.b_copy1
            h1 = np.maximum(z1, 0.0)
            c_out = h1 @ model.W_copy2 + model.b_copy2
            logits = c_out @ model.W_D + model.b_D
            
            _, dlogits = loss_and_grad_logits(logits, copy_batch)
            
            target_code = model.E[copy_batch]
            diff_closure = c_out - target_code
            
            dc_out_ce = dlogits @ model.W_D.T
            dc_out_closure = 2.0 * diff_closure / len(copy_batch)
            dc_out = dc_out_ce + lambda_closure * dc_out_closure
            
            grads["W_copy2"] = h1.T @ dc_out
            grads["b_copy2"] = np.sum(dc_out, axis=0)
            dh1 = dc_out @ model.W_copy2.T
            dz1 = dh1 * (z1 > 0.0)
            grads["W_copy1"] = c_a.T @ dz1
            grads["b_copy1"] = np.sum(dz1, axis=0)
            
        opt.update(params, grads)


def evaluate_multi_ops(model: MultiOperatorLatentModel) -> dict:
    num_digits = model.E.shape[0]
    
    a_list, b_list = [], []
    for a in range(num_digits):
        for b in range(num_digits):
            a_list.append(a)
            b_list.append(b)
    ab_a = np.array(a_list)
    ab_b = np.array(b_list)
    
    c_a = model.E[ab_a]
    c_b = model.E[ab_b]
    
    # ADD
    concat_add = np.concatenate([c_a, c_b], axis=1)
    h_add = np.maximum(concat_add @ model.W_add1 + model.b_add1, 0.0)
    c_add = h_add @ model.W_add2 + model.b_add2
    logits_add = c_add @ model.W_D + model.b_D
    y_add = (ab_a + ab_b) % num_digits
    add_acc = float(np.mean(np.argmax(logits_add, axis=1) == y_add))
    
    # MAX
    concat_max = np.concatenate([c_a, c_b], axis=1)
    h_max = np.maximum(concat_max @ model.W_max1 + model.b_max1, 0.0)
    c_max = h_max @ model.W_max2 + model.b_max2
    logits_max = c_max @ model.W_D + model.b_D
    y_max = np.maximum(ab_a, ab_b)
    max_acc = float(np.mean(np.argmax(logits_max, axis=1) == y_max))
    
    # COPY
    copy_batch = np.arange(num_digits)
    c_copy_in = model.E[copy_batch]
    h_copy = np.maximum(c_copy_in @ model.W_copy1 + model.b_copy1, 0.0)
    c_copy = h_copy @ model.W_copy2 + model.b_copy2
    logits_copy = c_copy @ model.W_D + model.b_D
    copy_acc = float(np.mean(np.argmax(logits_copy, axis=1) == copy_batch))
    
    return {"add": add_acc, "max": max_acc, "copy": copy_acc}


def evaluate_mixed_compositions(model: MultiOperatorLatentModel) -> dict:
    num_digits = model.E.shape[0]
    d0_list, d1_list, d2_list = [], [], []
    for d0 in range(num_digits):
        for d1 in range(num_digits):
            for d2 in range(num_digits):
                d0_list.append(d0)
                d1_list.append(d1)
                d2_list.append(d2)
    d0 = np.array(d0_list)
    d1 = np.array(d1_list)
    d2 = np.array(d2_list)
    
    c0 = model.E[d0]
    c1 = model.E[d1]
    c2 = model.E[d2]
    
    # Task 1: max(add(d0, d1), d2)
    concat1 = np.concatenate([c0, c1], axis=1)
    h_add1 = np.maximum(concat1 @ model.W_add1 + model.b_add1, 0.0)
    c_add1 = h_add1 @ model.W_add2 + model.b_add2
    concat2 = np.concatenate([c_add1, c2], axis=1)
    h_max2 = np.maximum(concat2 @ model.W_max1 + model.b_max1, 0.0)
    c_out1 = h_max2 @ model.W_max2 + model.b_max2
    logits1 = c_out1 @ model.W_D + model.b_D
    y_tgt1 = np.maximum((d0 + d1) % num_digits, d2)
    acc1 = float(np.mean(np.argmax(logits1, axis=1) == y_tgt1))
    
    # Task 2: add(max(d0, d1), d2)
    concat1 = np.concatenate([c0, c1], axis=1)
    h_max1 = np.maximum(concat1 @ model.W_max1 + model.b_max1, 0.0)
    c_max1 = h_max1 @ model.W_max2 + model.b_max2
    concat2 = np.concatenate([c_max1, c2], axis=1)
    h_add2 = np.maximum(concat2 @ model.W_add1 + model.b_add1, 0.0)
    c_out2 = h_add2 @ model.W_add2 + model.b_add2
    logits2 = c_out2 @ model.W_D + model.b_D
    y_tgt2 = (np.maximum(d0, d1) + d2) % num_digits
    acc2 = float(np.mean(np.argmax(logits2, axis=1) == y_tgt2))
    
    # Task 3: add(copy(d2), d0)
    h_copy1 = np.maximum(c2 @ model.W_copy1 + model.b_copy1, 0.0)
    c_copy1 = h_copy1 @ model.W_copy2 + model.b_copy2
    concat2 = np.concatenate([c_copy1, c0], axis=1)
    h_add2 = np.maximum(concat2 @ model.W_add1 + model.b_add1, 0.0)
    c_out3 = h_add2 @ model.W_add2 + model.b_add2
    logits3 = c_out3 @ model.W_D + model.b_D
    y_tgt3 = (d2 + d0) % num_digits
    acc3 = float(np.mean(np.argmax(logits3, axis=1) == y_tgt3))
    
    return {"max_of_sum": acc1, "sum_of_max": acc2, "sum_of_copy": acc3}


def run_mixed_operator_benchmark(
    seed_count: int,
    alignment_weight: float = 10.0,
) -> None:
    hidden_dim = 16
    code_dim = 4
    
    print("\n=================================================================")
    print("RUNNING CLOSED LATENT ALGEBRA (MIXED OPERATORS)")
    print(f"Seeds: {seed_count}, Hidden Dim: {hidden_dim}, Code Dim: {code_dim}")
    print("=================================================================")
    
    timelines = ["A (Simultaneous)", "B (Sequential)"]
    metrics = ["add_acc", "max_acc", "copy_acc", "max_of_sum", "sum_of_max", "sum_of_copy"]
    
    results = {time: {met: [] for met in metrics} for time in timelines}
    
    for seed in range(seed_count):
        # Timeline A (Simultaneous)
        model_a = MultiOperatorLatentModel(5, code_dim, hidden_dim, seed)
        train_autoencoder(model_a, epochs=600, lr=0.01, margin=2.0, sep_weight=0.5, quiet=True)
        train_multi_closed_ops(model_a, ["add", "max", "copy"], epochs=1500, lr=0.01, lambda_closure=alignment_weight, quiet=True)
        
        ops_a = evaluate_multi_ops(model_a)
        comp_a = evaluate_mixed_compositions(model_a)
        results["A (Simultaneous)"]["add_acc"].append(ops_a["add"])
        results["A (Simultaneous)"]["max_acc"].append(ops_a["max"])
        results["A (Simultaneous)"]["copy_acc"].append(ops_a["copy"])
        results["A (Simultaneous)"]["max_of_sum"].append(comp_a["max_of_sum"])
        results["A (Simultaneous)"]["sum_of_max"].append(comp_a["sum_of_max"])
        results["A (Simultaneous)"]["sum_of_copy"].append(comp_a["sum_of_copy"])
        
        # Timeline B (Sequential Operator learning)
        model_b = MultiOperatorLatentModel(5, code_dim, hidden_dim, seed)
        train_autoencoder(model_b, epochs=600, lr=0.01, margin=2.0, sep_weight=0.5, quiet=True)
        
        # Stage 1: Train ADD only
        train_multi_closed_ops(model_b, ["add"], epochs=1500, lr=0.01, lambda_closure=alignment_weight, quiet=True)
        # Stage 2: Freeze E/D and ADD, train MAX and COPY
        train_multi_closed_ops(model_b, ["max", "copy"], epochs=1500, lr=0.01, lambda_closure=alignment_weight, quiet=True)
        
        ops_b = evaluate_multi_ops(model_b)
        comp_b = evaluate_mixed_compositions(model_b)
        results["B (Sequential)"]["add_acc"].append(ops_b["add"])
        results["B (Sequential)"]["max_acc"].append(ops_b["max"])
        results["B (Sequential)"]["copy_acc"].append(ops_b["copy"])
        results["B (Sequential)"]["max_of_sum"].append(comp_b["max_of_sum"])
        results["B (Sequential)"]["sum_of_max"].append(comp_b["sum_of_max"])
        results["B (Sequential)"]["sum_of_copy"].append(comp_b["sum_of_copy"])
        
    print("\n=================================================================")
    print("CLOSED LATENT ALGEBRA (MIXED OPERATORS) SUMMARY (MEAN +/- STD)")
    print("=================================================================")
    for time in timelines:
        print(f"\nTimeline: {time}")
        print("-" * 50)
        for met in metrics:
            m, s = mean_std(results[time][met])
            print(f"  {met:<15}: {m:.3f}+/-{s:.3f}")
    print("=================================================================\n")


# =====================================================================
# CONTINUAL OPERATOR LEARNING EXPERIMENT (PHASE 5)
# =====================================================================

def generate_programs(library_ops: dict[str, int], sequence_digits: int, max_depth: int = 2) -> list[tuple]:
    vars_list = [("var", i) for i in range(sequence_digits)]
    programs = []
    
    # Depth 1: op(args...)
    for op_name, op_arity in library_ops.items():
        if op_arity == 1:
            for v in vars_list:
                programs.append(("op", op_name, (v,)))
        elif op_arity == 2:
            for v1 in vars_list:
                for v2 in vars_list:
                    programs.append(("op", op_name, (v1, v2)))
                    
    if max_depth < 2:
        return programs
        
    depth1_programs = list(programs)
    
    # Depth 2: op(args...) where at least one arg is a depth 1 program.
    for op_name, op_arity in library_ops.items():
        if op_arity == 1:
            for p1 in depth1_programs:
                programs.append(("op", op_name, (p1,)))
        elif op_arity == 2:
            for p1 in depth1_programs:
                for v in vars_list:
                    programs.append(("op", op_name, (p1, v)))
            for v in vars_list:
                for p2 in depth1_programs:
                    programs.append(("op", op_name, (v, p2)))
            for p1 in depth1_programs:
                for p2 in depth1_programs:
                    programs.append(("op", op_name, (p1, p2)))
                    
    return programs


def eval_program(program, E, W_D, b_D, operator_library, digit_indices):
    def rec(node):
        if node[0] == "var":
            idx = node[1]
            return E[digit_indices[:, idx]]
        elif node[0] == "op":
            op_name = node[1]
            args = node[2]
            arg_codes = []
            for arg in args:
                arg_codes.append(rec(arg))
            if len(arg_codes) == 1:
                inp = arg_codes[0]
            else:
                inp = np.concatenate(arg_codes, axis=1)
            op_params = operator_library[op_name]
            z1 = inp @ op_params["W1"] + op_params["b1"]
            h1 = np.maximum(z1, 0.0)
            return h1 @ op_params["W2"] + op_params["b2"]
            
    out_code = rec(program)
    logits = out_code @ W_D + b_D
    return out_code, logits


def search_best_program(library_ops, sequence_digits, E, W_D, b_D, operator_library, digit_indices, targets):
    if not library_ops:
        return None, 0.0, float('inf')
        
    candidate_programs = generate_programs(library_ops, sequence_digits, max_depth=2)
    best_program = None
    best_acc = -1.0
    best_loss = float('inf')
    
    for prog in candidate_programs:
        _, logits = eval_program(prog, E, W_D, b_D, operator_library, digit_indices)
        preds = np.argmax(logits, axis=1)
        acc = float(np.mean(preds == targets))
        loss, _ = loss_and_grad_logits(logits, targets)
        
        if acc > best_acc or (abs(acc - best_acc) < 1e-9 and loss < best_loss):
            best_acc = acc
            best_loss = loss
            best_program = prog
            
    return best_program, best_acc, best_loss


def train_dynamic_operator(
    E: np.ndarray,
    W_D: np.ndarray,
    b_D: np.ndarray,
    arity: int,
    code_dim: int,
    hidden_dim: int,
    digit_indices: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    lr: float,
    lambda_closure: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    input_dim = arity * code_dim
    
    op_params = {
        "W1": rng.normal(0.0, 1.0 / np.sqrt(input_dim), (input_dim, hidden_dim)),
        "b1": np.full(hidden_dim, 0.01),
        "W2": rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), (hidden_dim, code_dim)),
        "b2": np.zeros(code_dim),
    }
    
    opt = AdamOptimizer(lr=lr)
    
    for epoch in range(1, epochs + 1):
        if arity == 1:
            inp = E[digit_indices[:, 0]]
          # binary or higher
        else:
            inp = np.concatenate([E[digit_indices[:, i]] for i in range(arity)], axis=1)
            
        z1 = inp @ op_params["W1"] + op_params["b1"]
        h1 = np.maximum(z1, 0.0)
        out_code = h1 @ op_params["W2"] + op_params["b2"]
        logits = out_code @ W_D + b_D
        
        ce_loss, dlogits = loss_and_grad_logits(logits, targets)
        
        target_code = E[targets]
        diff_closure = out_code - target_code
        
        dc_out_ce = dlogits @ W_D.T
        dc_out_closure = 2.0 * diff_closure / len(targets)
        dc_out = dc_out_ce + lambda_closure * dc_out_closure
        
        dW2 = h1.T @ dc_out
        db2 = np.sum(dc_out, axis=0)
        dh1 = dc_out @ op_params["W2"].T
        dz1 = dh1 * (z1 > 0.0)
        dW1 = inp.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        opt.update(op_params, grads)
        
    return op_params


def get_task_data(task_name):
    if task_name == "ADD":
        a_list, b_list = [], []
        for a in range(5):
            for b in range(5):
                 a_list.append(a)
                 b_list.append(b)
        digit_indices = np.stack([a_list, b_list], axis=1)
        targets = (digit_indices[:, 0] + digit_indices[:, 1]) % 5
        arity = 2
    elif task_name == "MAX":
        a_list, b_list = [], []
        for a in range(5):
            for b in range(5):
                 a_list.append(a)
                 b_list.append(b)
        digit_indices = np.stack([a_list, b_list], axis=1)
        targets = np.maximum(digit_indices[:, 0], digit_indices[:, 1])
        arity = 2
    elif task_name == "COPY":
        digit_indices = np.arange(5).reshape(-1, 1)
        targets = digit_indices[:, 0]
        arity = 1
    elif task_name == "MIN":
        a_list, b_list = [], []
        for a in range(5):
            for b in range(5):
                 a_list.append(a)
                 b_list.append(b)
        digit_indices = np.stack([a_list, b_list], axis=1)
        targets = np.minimum(digit_indices[:, 0], digit_indices[:, 1])
        arity = 2
    elif task_name == "SUB":
        a_list, b_list = [], []
        for a in range(5):
            for b in range(5):
                 a_list.append(a)
                 b_list.append(b)
        digit_indices = np.stack([a_list, b_list], axis=1)
        targets = (digit_indices[:, 0] - digit_indices[:, 1]) % 5
        arity = 2
    else:
        raise ValueError(f"unknown task_name={task_name}")
    return digit_indices, targets, arity


def compile_composition(template, task_to_program):
    if template[0] == "var":
        return template
    elif template[0] == "op":
        task_name = template[1]
        args = template[2]
        sub_args = [compile_composition(arg, task_to_program) for arg in args]
        
        task_prog = task_to_program[task_name]
        
        def bind_vars(node, bindings):
            if node[0] == "var":
                var_idx = node[1]
                return bindings[var_idx]
            elif node[0] == "op":
                op_nm = node[1]
                op_args = node[2]
                return ("op", op_nm, tuple(bind_vars(a, bindings) for a in op_args))
                
        return bind_vars(task_prog, sub_args)


def run_continual_operator_learning(seed_count: int, alignment_weight: float = 10.0) -> None:
    hidden_dim = 16
    code_dim = 4
    
    print("\n=================================================================")
    print("RUNNING CONTINUAL OPERATOR LEARNING BENCHMARK")
    print(f"Seeds: {seed_count}, Hidden Dim: {hidden_dim}, Code Dim: {code_dim}")
    print("=================================================================")
    
    policies = ["always_new_operator", "always_try_reuse", "admission_gated_reuse"]
    stages = ["ADD", "MAX", "COPY", "MIN", "SUB"]
    
    policy_results = {
        pol: {
            "ADD_acc": [], "MAX_acc": [], "COPY_acc": [], "MIN_acc": [], "SUB_acc": [],
            "max_of_sum_acc": [], "sum_of_max_acc": [], "sub_of_sum_acc": [],
            "max_of_min_acc": [], "sum_of_copy_acc": [],
            "comp_acc": [], "op_count": [], "params_added": [],
            "closure_mse": [], "manifold_drift": [], "false_reuse_rate": []
        }
        for pol in policies
    }
    
    d0_list, d1_list, d2_list = [], [], []
    for d0 in range(5):
        for d1 in range(5):
            for d2 in range(5):
                d0_list.append(d0)
                d1_list.append(d1)
                d2_list.append(d2)
    comp_indices = np.stack([d0_list, d1_list, d2_list], axis=1)
    
    comp_targets = {
        "max_of_sum": np.maximum((comp_indices[:, 0] + comp_indices[:, 1]) % 5, comp_indices[:, 2]),
        "sum_of_max": (np.maximum(comp_indices[:, 0], comp_indices[:, 1]) + comp_indices[:, 2]) % 5,
        "sub_of_sum": ((comp_indices[:, 0] + comp_indices[:, 1]) % 5 - comp_indices[:, 2]) % 5,
        "max_of_min": np.maximum(np.minimum(comp_indices[:, 0], comp_indices[:, 1]), comp_indices[:, 2]),
        "sum_of_copy": (comp_indices[:, 2] + comp_indices[:, 0]) % 5
    }
    
    comp_templates = {
        "max_of_sum": ("op", "MAX", (("op", "ADD", (("var", 0), ("var", 1))), ("var", 2))),
        "sum_of_max": ("op", "ADD", (("op", "MAX", (("var", 0), ("var", 1))), ("var", 2))),
        "sub_of_sum": ("op", "SUB", (("op", "ADD", (("var", 0), ("var", 1))), ("var", 2))),
        "max_of_min": ("op", "MAX", (("op", "MIN", (("var", 0), ("var", 1))), ("var", 2))),
        "sum_of_copy": ("op", "ADD", (("op", "COPY", (("var", 2),)), ("var", 0)))
    }
    
    for seed in range(seed_count):
        print(f"\n--- Seed {seed} ---")
        model = make_closed_latent_model(num_digits=5, code_dim=code_dim, hidden_dim=hidden_dim, seed=seed)
        train_autoencoder(model, epochs=600, lr=0.01, margin=2.0, sep_weight=0.5, quiet=True)
        E = model.E
        W_D = model.W_D
        b_D = model.b_D
        
        for pol in policies:
            operator_library = {}
            library_ops = {}
            task_to_program = {}
            operator_origin_task = {}
            false_reuse_count = 0
            
            for stage in stages:
                digit_indices, targets, arity = get_task_data(stage)
                
                reused = False
                failed_reuse = False
                chosen_prog = None
                
                if pol == "always_new_operator":
                    pass
                else:
                    best_prog, best_acc, best_loss = search_best_program(
                        library_ops, arity, E, W_D, b_D, operator_library, digit_indices, targets
                    )
                    if best_prog is not None:
                        if pol == "always_try_reuse":
                            chosen_prog = best_prog
                            reused = True
                            failed_reuse = (best_acc < 0.95)
                        elif pol == "admission_gated_reuse":
                            if best_acc >= 0.98:
                                chosen_prog = best_prog
                                reused = True
                                failed_reuse = False
                                
                if reused:
                    task_to_program[stage] = chosen_prog
                    if failed_reuse:
                        false_reuse_count += 1
                else:
                    op_name = f"OP_{stage}"
                    op_params = train_dynamic_operator(
                        E, W_D, b_D, arity, code_dim, hidden_dim, digit_indices, targets,
                        epochs=1500, lr=0.01, lambda_closure=alignment_weight, seed=seed
                    )
                    operator_library[op_name] = op_params
                    library_ops[op_name] = arity
                    operator_origin_task[op_name] = stage
                    default_vars = tuple(("var", i) for i in range(arity))
                    task_to_program[stage] = ("op", op_name, default_vars)
                    
            # Evaluate final task accuracies
            stage_accs = {}
            for stage in stages:
                digit_indices, targets, arity = get_task_data(stage)
                prog = task_to_program[stage]
                _, logits = eval_program(prog, E, W_D, b_D, operator_library, digit_indices)
                preds = np.argmax(logits, axis=1)
                acc = float(np.mean(preds == targets))
                stage_accs[stage] = acc
                policy_results[pol][f"{stage}_acc"].append(acc)
                
            # Evaluate final compositions
            comp_accs_list = []
            for comp_name, template in comp_templates.items():
                compiled = compile_composition(template, task_to_program)
                _, logits = eval_program(compiled, E, W_D, b_D, operator_library, comp_indices)
                preds = np.argmax(logits, axis=1)
                acc = float(np.mean(preds == comp_targets[comp_name]))
                policy_results[pol][f"{comp_name}_acc"].append(acc)
                comp_accs_list.append(acc)
            avg_comp_acc = float(np.mean(comp_accs_list))
            policy_results[pol]["comp_acc"].append(avg_comp_acc)
            
            op_cnt = len(operator_library)
            policy_results[pol]["op_count"].append(op_cnt)
            
            params_sum = 0
            for op_name, op_params in operator_library.items():
                arity = library_ops[op_name]
                input_dim = arity * code_dim
                params_sum += input_dim * hidden_dim + hidden_dim + hidden_dim * code_dim + code_dim
            policy_results[pol]["params_added"].append(params_sum)
            
            mse_vals = []
            drift_vals = []
            for op_name, op_params in operator_library.items():
                orig_task = operator_origin_task[op_name]
                digit_indices, targets, arity = get_task_data(orig_task)
                if arity == 1:
                    inp = E[digit_indices[:, 0]]
                else:
                    inp = np.concatenate([E[digit_indices[:, i]] for i in range(arity)], axis=1)
                z1 = inp @ op_params["W1"] + op_params["b1"]
                h1 = np.maximum(z1, 0.0)
                out_code = h1 @ op_params["W2"] + op_params["b2"]
                
                target_code = E[targets]
                mse = np.mean(np.sum((out_code - target_code)**2, axis=1))
                mse_vals.append(mse)
                
                dists = np.sum((out_code[:, np.newaxis, :] - E[np.newaxis, :, :])**2, axis=2)
                drift = np.mean(np.min(dists, axis=1))
                drift_vals.append(drift)
                
            avg_mse = float(np.mean(mse_vals)) if mse_vals else 0.0
            avg_drift = float(np.mean(drift_vals)) if drift_vals else 0.0
            policy_results[pol]["closure_mse"].append(avg_mse)
            policy_results[pol]["manifold_drift"].append(avg_drift)
            
            policy_results[pol]["false_reuse_rate"].append(false_reuse_count / len(stages))
            
            print(f"  [{pol}] ops={op_cnt} params={params_sum} ADD_acc={stage_accs['ADD']:.3f} MAX_acc={stage_accs['MAX']:.3f} COPY_acc={stage_accs['COPY']:.3f} avg_comp={avg_comp_acc:.3f}")
            
    print("\n=================================================================")
    print("CONTINUAL OPERATOR LEARNING FINAL COMPARATIVE SUMMARY")
    print("=================================================================")
    print(f"Metric / Policy            always_new_operator    always_try_reuse      admission_gated_reuse")
    print("-" * 95)
    
    summary_metrics = [
        ("operator_count", "op_count"),
        ("new_parameters_added", "params_added"),
        ("ADD accuracy", "ADD_acc"),
        ("MAX accuracy", "MAX_acc"),
        ("COPY accuracy", "COPY_acc"),
        ("MIN accuracy", "MIN_acc"),
        ("SUB accuracy", "SUB_acc"),
        ("max_of_sum accuracy", "max_of_sum_acc"),
        ("sum_of_max accuracy", "sum_of_max_acc"),
        ("sub_of_sum accuracy", "sub_of_sum_acc"),
        ("max_of_min accuracy", "max_of_min_acc"),
        ("sum_of_copy accuracy", "sum_of_copy_acc"),
        ("Average Composition Acc", "comp_acc"),
        ("closure_mse", "closure_mse"),
        ("manifold_drift", "manifold_drift"),
        ("false_reuse_rate", "false_reuse_rate")
    ]
    
    for label, key in summary_metrics:
        row_strs = []
        for pol in policies:
            vals = policy_results[pol][key]
            m, s = mean_std(vals)
            if key in ("op_count", "params_added"):
                row_strs.append(f"{m:.1f} +/- {s:.1f}")
            else:
                row_strs.append(f"{m:.4f} +/- {s:.4f}")
        print(f"{label:<25}  {row_strs[0]:<22}  {row_strs[1]:<22}  {row_strs[2]:<22}")
    print("=================================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--factorized-stress", action="store_true")
    parser.add_argument("--multi-route-addition", action="store_true")
    parser.add_argument("--multi-route-non-analog", action="store_true")
    parser.add_argument("--composition-benchmark", action="store_true")
    parser.add_argument("--closed-latent-benchmark", action="store_true")
    parser.add_argument("--closure-ablation", action="store_true")
    parser.add_argument("--freeze-ablation", action="store_true")
    parser.add_argument("--long-composition", action="store_true")
    parser.add_argument("--mixed-operators", action="store_true")
    parser.add_argument("--continual-operator-learning", action="store_true")
    parser.add_argument("--multi-route-hidden-dim", type=int, default=OpsConfig.hidden_dim)
    parser.add_argument(
        "--multi-route-alignment-weight",
        type=float,
        default=DEFAULT_MULTI_ROUTE_ALIGNMENT_WEIGHT,
    )
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
    if args.continual_operator_learning:
        run_continual_operator_learning(
            seed_count=args.seed_count if args.multi_seed else 1,
            alignment_weight=args.multi_route_alignment_weight,
        )
    elif args.mixed_operators:
        run_mixed_operator_benchmark(
            seed_count=args.seed_count if args.multi_seed else 1,
            alignment_weight=args.multi_route_alignment_weight,
        )
    elif args.closure_ablation:
        run_closure_ablation(
            seed_count=args.seed_count if args.multi_seed else 1,
            alignment_weight=args.multi_route_alignment_weight,
        )
    elif args.freeze_ablation:
        run_freeze_ablation(
            seed_count=args.seed_count if args.multi_seed else 1,
            alignment_weight=args.multi_route_alignment_weight,
        )
    elif args.long_composition:
        run_long_composition_test(
            seed_count=args.seed_count if args.multi_seed else 1,
            alignment_weight=args.multi_route_alignment_weight,
        )
    elif args.closed_latent_benchmark:
        if args.multi_seed:
            run_closed_latent_benchmark(
                seed_count=args.seed_count,
                alignment_weight=args.multi_route_alignment_weight,
                verbose=False,
            )
        else:
            run_closed_latent_benchmark(
                seed_count=1,
                alignment_weight=args.multi_route_alignment_weight,
                verbose=True,
            )
    elif args.composition_benchmark:
        if args.multi_seed:
            run_composition_benchmark_multi_seed(
                args.seed_count,
                args.multi_route_hidden_dim,
                args.multi_route_alignment_weight,
            )
        else:
            run_composition_benchmark_table(
                seed=7,
                hidden_dim=args.multi_route_hidden_dim,
                alignment_weight=args.multi_route_alignment_weight,
                verbose=True,
            )
    elif args.multi_route_non_analog:
        if args.multi_seed:
            run_multi_route_non_analog_multi_seed(
                args.seed_count,
                args.multi_route_hidden_dim,
                args.multi_route_alignment_weight,
            )
        else:
            run_multi_route_non_analog_table(
                seed=7,
                hidden_dim=args.multi_route_hidden_dim,
                alignment_weight=args.multi_route_alignment_weight,
                verbose=True,
            )
    elif args.multi_route_addition:
        if args.multi_seed:
            run_multi_route_addition_multi_seed(
                args.seed_count,
                args.multi_route_hidden_dim,
                args.multi_route_alignment_weight,
            )
        else:
            run_multi_route_addition_table(
                seed=7,
                hidden_dim=args.multi_route_hidden_dim,
                alignment_weight=args.multi_route_alignment_weight,
                verbose=True,
            )
    elif args.factorized_stress:
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
