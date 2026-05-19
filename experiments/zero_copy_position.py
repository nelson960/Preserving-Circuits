from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from zero_transformer import ZeroTransformer


@dataclass(frozen=True)
class CopyPositionConfig:
    num_digits: int = 10
    sequence_digits: int = 3
    query_token_name: str = "QUERY"
    source_position: int = 0
    seed: int = 0
    d_model: int = 32
    num_heads: int = 2

    @property
    def query_token(self) -> int:
        return self.num_digits

    @property
    def vocab_size(self) -> int:
        return self.num_digits + 1

    @property
    def sequence_length(self) -> int:
        return self.sequence_digits + 1


@dataclass(frozen=True)
class CopyPositionExample:
    tokens: tuple[int, ...]
    target: int
    source_position: int


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 0.03
    grad_clip_norm: float = 5.0
    log_every: int = 20


def make_copy_position_examples(config: CopyPositionConfig) -> tuple[CopyPositionExample, ...]:
    if config.num_digits <= 1:
        raise ValueError("num_digits must be greater than 1.")
    if config.sequence_digits <= 0:
        raise ValueError("sequence_digits must be positive.")
    if not 0 <= config.source_position < config.sequence_digits:
        raise ValueError(
            "source_position must be inside the digit sequence, got "
            f"{config.source_position} for length {config.sequence_digits}."
        )

    examples: list[CopyPositionExample] = []
    for digit_tuple in np.ndindex(*(config.num_digits,) * config.sequence_digits):
        tokens = tuple(int(token) for token in digit_tuple) + (config.query_token,)
        examples.append(
            CopyPositionExample(
                tokens=tokens,
                target=int(digit_tuple[config.source_position]),
                source_position=config.source_position,
            )
        )
    return tuple(examples)


def make_probe_example(config: CopyPositionConfig) -> CopyPositionExample:
    if config.num_digits < config.sequence_digits:
        raise ValueError(
            "num_digits must be at least sequence_digits for a distinct-digit probe, "
            f"got num_digits={config.num_digits}, sequence_digits={config.sequence_digits}."
        )
    digit_tuple = tuple(range(config.sequence_digits))
    return CopyPositionExample(
        tokens=digit_tuple + (config.query_token,),
        target=int(digit_tuple[config.source_position]),
        source_position=config.source_position,
    )


def token_label(token: int, config: CopyPositionConfig) -> str:
    if token == config.query_token:
        return config.query_token_name
    if 0 <= token < config.num_digits:
        return f"D{token}"
    raise ValueError(f"token {token} is outside vocab size {config.vocab_size}.")


def describe_example(example: CopyPositionExample, config: CopyPositionConfig) -> str:
    token_labels = [token_label(token, config) for token in example.tokens]
    target_label = token_label(example.target, config)
    return (
        f"tokens={token_labels}, source_position={example.source_position}, "
        f"target={target_label}"
    )


def forward_full(
    model: ZeroTransformer, tokens: tuple[int, ...] | np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    token_array = np.array(tokens, dtype=int)
    T = len(token_array)
    X = model.W_E[token_array] + model.W_P[:T]

    Q = np.zeros((model.num_heads, T, model.d_head))
    K = np.zeros((model.num_heads, T, model.d_head))
    V = np.zeros((model.num_heads, T, model.d_head))
    for head in range(model.num_heads):
        Q[head] = X @ model.W_Q[head]
        K[head] = X @ model.W_K[head]
        V[head] = X @ model.W_V[head]

    S = np.einsum("hti,hki->htk", Q, K) / np.sqrt(model.d_head)
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)
    S[:, mask] = -1e9
    A = model.softmax(S, axis=-1)
    head_outputs = np.einsum("htk,hki->hti", A, V)

    projected_outputs = np.zeros((model.num_heads, T, model.d_model))
    for head in range(model.num_heads):
        projected_outputs[head] = head_outputs[head] @ model.W_O[head]

    attention_out = np.sum(projected_outputs, axis=0)
    X_final = X + attention_out
    logits = X_final @ model.W_U
    cache = {
        "tokens": token_array,
        "X": X,
        "Q": Q,
        "K": K,
        "V": V,
        "S": S,
        "A": A,
        "head_outputs": head_outputs,
        "projected_outputs": projected_outputs,
        "attention_out": attention_out,
        "X_final": X_final,
        "logits": logits,
    }
    return logits, cache


def cross_entropy_final_token(
    logits: np.ndarray, target: int
) -> tuple[float, np.ndarray]:
    final_logits = logits[-1]
    shifted = final_logits - np.max(final_logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    loss = -np.log(probs[target] + 1e-12)
    dlogits = np.zeros_like(logits)
    dlogits[-1] = probs
    dlogits[-1, target] -= 1.0
    return float(loss), dlogits


def empty_grads(model: ZeroTransformer) -> dict[str, np.ndarray]:
    return {
        "W_E": np.zeros_like(model.W_E),
        "W_P": np.zeros_like(model.W_P),
        "W_Q": np.zeros_like(model.W_Q),
        "W_K": np.zeros_like(model.W_K),
        "W_V": np.zeros_like(model.W_V),
        "W_O": np.zeros_like(model.W_O),
        "W_U": np.zeros_like(model.W_U),
    }


def add_grads(target: dict[str, np.ndarray], source: dict[str, np.ndarray]) -> None:
    for name, grad in source.items():
        target[name] += grad


def scale_grads(grads: dict[str, np.ndarray], scale: float) -> None:
    for name in grads:
        grads[name] *= scale


def grad_norm(grads: dict[str, np.ndarray]) -> float:
    squared = sum(float(np.sum(grad * grad)) for grad in grads.values())
    return float(np.sqrt(squared))


def clip_grads(grads: dict[str, np.ndarray], max_norm: float) -> float:
    norm = grad_norm(grads)
    if max_norm > 0.0 and norm > max_norm:
        scale_grads(grads, max_norm / (norm + 1e-12))
    return norm


def backward_full(
    model: ZeroTransformer, cache: dict[str, np.ndarray], dlogits: np.ndarray
) -> dict[str, np.ndarray]:
    grads = empty_grads(model)
    tokens = cache["tokens"]
    X = cache["X"]
    Q = cache["Q"]
    K = cache["K"]
    V = cache["V"]
    A = cache["A"]
    head_outputs = cache["head_outputs"]
    X_final = cache["X_final"]
    T = len(tokens)
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)

    grads["W_U"] += X_final.T @ dlogits
    dX_final = dlogits @ model.W_U.T

    dX = dX_final.copy()
    dattention_out = dX_final

    for head in range(model.num_heads):
        dprojected = dattention_out
        grads["W_O"][head] += head_outputs[head].T @ dprojected
        dhead_outputs = dprojected @ model.W_O[head].T

        dA = dhead_outputs @ V[head].T
        dV = A[head].T @ dhead_outputs

        row_dot = np.sum(dA * A[head], axis=-1, keepdims=True)
        dS = A[head] * (dA - row_dot)
        dS[mask] = 0.0

        scale = 1.0 / np.sqrt(model.d_head)
        dQ = dS @ K[head] * scale
        dK = dS.T @ Q[head] * scale

        grads["W_Q"][head] += X.T @ dQ
        grads["W_K"][head] += X.T @ dK
        grads["W_V"][head] += X.T @ dV

        dX += dQ @ model.W_Q[head].T
        dX += dK @ model.W_K[head].T
        dX += dV @ model.W_V[head].T

    np.add.at(grads["W_E"], tokens, dX)
    grads["W_P"][:T] += dX
    return grads


def apply_sgd_update(
    model: ZeroTransformer,
    grads: dict[str, np.ndarray],
    learning_rate: float,
    frozen: frozenset[str] = frozenset(),
) -> None:
    if "W_E" not in frozen:
        model.W_E -= learning_rate * grads["W_E"]
    if "W_P" not in frozen:
        model.W_P -= learning_rate * grads["W_P"]
    if "W_Q" not in frozen:
        model.W_Q -= learning_rate * grads["W_Q"]
    if "W_K" not in frozen:
        model.W_K -= learning_rate * grads["W_K"]
    if "W_V" not in frozen:
        model.W_V -= learning_rate * grads["W_V"]
    if "W_O" not in frozen:
        model.W_O -= learning_rate * grads["W_O"]
    if "W_U" not in frozen:
        model.W_U -= learning_rate * grads["W_U"]


def evaluate(
    model: ZeroTransformer, examples: tuple[CopyPositionExample, ...]
) -> tuple[float, float]:
    losses: list[float] = []
    correct = 0
    for example in examples:
        logits, _ = forward_full(model, example.tokens)
        loss, _ = cross_entropy_final_token(logits, example.target)
        losses.append(loss)
        if int(np.argmax(logits[-1])) == example.target:
            correct += 1
    return float(np.mean(losses)), correct / len(examples)


def train_task(
    model: ZeroTransformer,
    examples: tuple[CopyPositionExample, ...],
    train_config: TrainConfig,
    seed: int,
    label: str,
    frozen: Iterable[str] = (),
) -> None:
    rng = np.random.default_rng(seed)
    frozen_set = frozenset(frozen)
    for epoch in range(1, train_config.epochs + 1):
        order = rng.permutation(len(examples))
        for batch_start in range(0, len(examples), train_config.batch_size):
            batch_indices = order[batch_start : batch_start + train_config.batch_size]
            batch_grads = empty_grads(model)
            batch_loss = 0.0
            for index in batch_indices:
                example = examples[int(index)]
                logits, cache = forward_full(model, example.tokens)
                loss, dlogits = cross_entropy_final_token(logits, example.target)
                batch_loss += loss
                add_grads(batch_grads, backward_full(model, cache, dlogits))
            scale_grads(batch_grads, 1.0 / len(batch_indices))
            clip_grads(batch_grads, train_config.grad_clip_norm)
            apply_sgd_update(model, batch_grads, train_config.learning_rate, frozen=frozen_set)

        if epoch == 1 or epoch % train_config.log_every == 0 or epoch == train_config.epochs:
            loss, accuracy = evaluate(model, examples)
            print(f"{label} epoch={epoch:03d} loss={loss:.4f} accuracy={accuracy:.3f}")


def inspect_random_forward(config: CopyPositionConfig) -> None:
    np.random.seed(config.seed)
    _ = make_copy_position_examples(config)
    example = make_probe_example(config)
    model = ZeroTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
    )

    token_array = np.array(example.tokens, dtype=int)
    logits, cache = model.forward(token_array)
    final_logits = logits[-1, : config.num_digits]
    prediction = int(np.argmax(final_logits))

    print("Zero Transformer copy-position probe")
    print("------------------------------------")
    print(describe_example(example, config))
    print(f"prediction={token_label(prediction, config)}")
    print(f"X_initial_shape={cache['X_initial'].shape}")
    print(f"X_final_shape={cache['X_final'].shape}")

    final_query_index = config.sequence_length - 1
    for head in range(config.num_heads):
        attention = cache["A"][head, final_query_index]
        attention_by_position = [
            (position, token_label(example.tokens[position], config), float(weight))
            for position, weight in enumerate(attention)
        ]
        print(f"head_{head}_final_query_attention={attention_by_position}")
        print(f"head_{head}_C_QK_fro_norm={np.linalg.norm(model.get_C_QK(head)):.6f}")
        print(f"head_{head}_C_OV_fro_norm={np.linalg.norm(model.get_C_OV(head)):.6f}")


def mean_final_query_attention(
    model: ZeroTransformer, examples: tuple[CopyPositionExample, ...], config: CopyPositionConfig
) -> np.ndarray:
    final_query_index = config.sequence_length - 1
    attention = np.zeros((model.num_heads, config.sequence_length))
    for example in examples:
        _, cache = forward_full(model, example.tokens)
        attention += cache["A"][:, final_query_index, :]
    return attention / len(examples)


def clone_zero_transformer(model: ZeroTransformer) -> ZeroTransformer:
    clone = ZeroTransformer(
        vocab_size=model.vocab_size,
        d_model=model.d_model,
        num_heads=model.num_heads,
    )
    clone.W_E = model.W_E.copy()
    clone.W_P = model.W_P.copy()
    clone.W_Q = model.W_Q.copy()
    clone.W_K = model.W_K.copy()
    clone.W_V = model.W_V.copy()
    clone.W_O = model.W_O.copy()
    clone.W_U = model.W_U.copy()
    return clone


def circuit_drift_report(
    model: ZeroTransformer,
    c_qk_reference: np.ndarray,
    c_ov_reference: np.ndarray,
    w_u_reference: np.ndarray,
) -> tuple[float, float, float]:
    c_qk_current = np.array([model.get_C_QK(head).copy() for head in range(model.num_heads)])
    c_ov_current = np.array([model.get_C_OV(head).copy() for head in range(model.num_heads)])
    return (
        float(np.linalg.norm(c_qk_current - c_qk_reference)),
        float(np.linalg.norm(c_ov_current - c_ov_reference)),
        float(np.linalg.norm(model.W_U - w_u_reference)),
    )


def run_task_b_ablation(
    base_after_a: ZeroTransformer,
    task_a_examples: tuple[CopyPositionExample, ...],
    task_b_examples: tuple[CopyPositionExample, ...],
    model_config_a: CopyPositionConfig,
    train_config: TrainConfig,
    frozen: tuple[str, ...],
    label: str,
    c_qk_after_a: np.ndarray,
    c_ov_after_a: np.ndarray,
    w_u_after_a: np.ndarray,
) -> None:
    model = clone_zero_transformer(base_after_a)
    train_task(
        model,
        task_b_examples,
        train_config,
        seed=30 + len(frozen),
        label=label,
        frozen=frozen,
    )
    task_a_loss, task_a_accuracy = evaluate(model, task_a_examples)
    task_b_loss, task_b_accuracy = evaluate(model, task_b_examples)
    qk_drift, ov_drift, w_u_drift = circuit_drift_report(
        model, c_qk_after_a, c_ov_after_a, w_u_after_a
    )
    print(f"\nAblation {label}")
    print(f"frozen={frozen}")
    print(f"task_a_loss={task_a_loss:.4f} task_a_accuracy={task_a_accuracy:.3f}")
    print(f"task_b_loss={task_b_loss:.4f} task_b_accuracy={task_b_accuracy:.3f}")
    print(f"mean_final_query_attention_task_a={mean_final_query_attention(model, task_a_examples, model_config_a)}")
    print(f"C_QK_drift={qk_drift:.6f} C_OV_drift={ov_drift:.6f} W_U_drift={w_u_drift:.6f}")
    print(
        "route_input_drift="
        f"W_E {np.linalg.norm(model.W_E - base_after_a.W_E):.6f}, "
        f"W_P {np.linalg.norm(model.W_P - base_after_a.W_P):.6f}"
    )


def run_task_a_then_b_demo() -> None:
    model_config_a = CopyPositionConfig(
        num_digits=5,
        sequence_digits=3,
        source_position=0,
        seed=1,
        d_model=24,
        num_heads=2,
    )
    model_config_b = CopyPositionConfig(
        num_digits=model_config_a.num_digits,
        sequence_digits=model_config_a.sequence_digits,
        source_position=1,
        seed=model_config_a.seed,
        d_model=model_config_a.d_model,
        num_heads=model_config_a.num_heads,
    )
    train_config = TrainConfig(epochs=120, batch_size=25, learning_rate=0.04, log_every=30)

    np.random.seed(model_config_a.seed)
    model = ZeroTransformer(
        vocab_size=model_config_a.vocab_size,
        d_model=model_config_a.d_model,
        num_heads=model_config_a.num_heads,
    )
    task_a_examples = make_copy_position_examples(model_config_a)
    task_b_examples = make_copy_position_examples(model_config_b)

    print("\nTraining Task A: final query should copy position 0")
    train_task(model, task_a_examples, train_config, seed=10, label="task_a")
    model_after_a = clone_zero_transformer(model)
    task_a_loss, task_a_accuracy = evaluate(model, task_a_examples)
    task_b_loss_before, task_b_accuracy_before = evaluate(model, task_b_examples)
    c_qk_after_a = np.array([model.get_C_QK(head).copy() for head in range(model.num_heads)])
    c_ov_after_a = np.array([model.get_C_OV(head).copy() for head in range(model.num_heads)])
    w_u_after_a = model.W_U.copy()

    print("\nAfter Task A")
    print(f"task_a_loss={task_a_loss:.4f} task_a_accuracy={task_a_accuracy:.3f}")
    print(f"task_b_loss={task_b_loss_before:.4f} task_b_accuracy={task_b_accuracy_before:.3f}")
    print(f"mean_final_query_attention_task_a={mean_final_query_attention(model, task_a_examples, model_config_a)}")

    print("\nTraining Task B: final query should copy position 1")
    train_task(model, task_b_examples, train_config, seed=20, label="task_b")
    task_a_loss_after, task_a_accuracy_after = evaluate(model, task_a_examples)
    task_b_loss_after, task_b_accuracy_after = evaluate(model, task_b_examples)
    c_qk_after_b = np.array([model.get_C_QK(head).copy() for head in range(model.num_heads)])
    c_ov_after_b = np.array([model.get_C_OV(head).copy() for head in range(model.num_heads)])

    print("\nAfter Task B")
    print(f"task_a_loss={task_a_loss_after:.4f} task_a_accuracy={task_a_accuracy_after:.3f}")
    print(f"task_b_loss={task_b_loss_after:.4f} task_b_accuracy={task_b_accuracy_after:.3f}")
    print(f"mean_final_query_attention_task_a={mean_final_query_attention(model, task_a_examples, model_config_a)}")
    print(f"mean_final_query_attention_task_b={mean_final_query_attention(model, task_b_examples, model_config_b)}")
    print(f"C_QK_drift_after_task_b={np.linalg.norm(c_qk_after_b - c_qk_after_a):.6f}")
    print(f"C_OV_drift_after_task_b={np.linalg.norm(c_ov_after_b - c_ov_after_a):.6f}")
    print(f"W_U_drift_after_task_b={np.linalg.norm(model.W_U - w_u_after_a):.6f}")

    print("\nTask B ablations from the same Task A checkpoint")
    run_task_b_ablation(
        model_after_a,
        task_a_examples,
        task_b_examples,
        model_config_a,
        train_config,
        frozen=("W_Q", "W_K"),
        label="task_b_freeze_qk",
        c_qk_after_a=c_qk_after_a,
        c_ov_after_a=c_ov_after_a,
        w_u_after_a=w_u_after_a,
    )
    run_task_b_ablation(
        model_after_a,
        task_a_examples,
        task_b_examples,
        model_config_a,
        train_config,
        frozen=("W_E", "W_P", "W_Q", "W_K"),
        label="task_b_freeze_route_inputs_and_qk",
        c_qk_after_a=c_qk_after_a,
        c_ov_after_a=c_ov_after_a,
        w_u_after_a=w_u_after_a,
    )
    run_task_b_ablation(
        model_after_a,
        task_a_examples,
        task_b_examples,
        model_config_a,
        train_config,
        frozen=("W_V", "W_O"),
        label="task_b_freeze_ov",
        c_qk_after_a=c_qk_after_a,
        c_ov_after_a=c_ov_after_a,
        w_u_after_a=w_u_after_a,
    )
    run_task_b_ablation(
        model_after_a,
        task_a_examples,
        task_b_examples,
        model_config_a,
        train_config,
        frozen=("W_U",),
        label="task_b_freeze_readout",
        c_qk_after_a=c_qk_after_a,
        c_ov_after_a=c_ov_after_a,
        w_u_after_a=w_u_after_a,
    )


if __name__ == "__main__":
    inspect_random_forward(CopyPositionConfig())
    run_task_a_then_b_demo()
