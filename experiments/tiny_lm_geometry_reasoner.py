"""Tiny transformer latent-geometry continual learning experiment.

This is the first scale step after the character semantic reasoner. It uses a
small decoder-only transformer as a stable token manifold, then learns closed
semantic operators over that manifold with a counterfactual optimizer loop.

The optimizer loop is intentionally explicit:

    generate candidate futures -> measure latent geometry -> choose action

The learned-policy mode trains a small PyTorch policy to choose among
reuse / compose / update / allocate from the counterfactual geometry features.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


Program = tuple[Any, ...]
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")
ACTION_NAMES = ("reuse", "compose", "update", "allocate")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}


@dataclass(frozen=True)
class SemanticSpec:
    input_domain: tuple[str, ...]
    relations: dict[str, dict[str, str]]
    train_streams: tuple[tuple[str, ...], ...]
    test_streams: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    def encode(self, token: str) -> int:
        if token not in self.token_to_id:
            raise KeyError(f"Unknown token {token!r}.")
        return self.token_to_id[token]

    @property
    def size(self) -> int:
        return len(self.id_to_token)


@dataclass(frozen=True)
class Config:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    operator_hidden_dim: int
    base_epochs: int
    operator_epochs: int
    update_epochs: int
    shadow_operator_epochs: int
    shadow_update_epochs: int
    base_lr: float
    operator_lr: float
    update_lr: float
    lambda_closure: float
    separation_margin: float
    separation_weight: float
    lm_weight: float
    ae_weight: float
    reuse_acc_threshold: float
    reuse_closure_threshold: float
    update_closure_low: float
    update_closure_high: float
    search_depth: int
    max_programs: int
    repair_noise_std: float
    structural_risk_power: float
    structural_need_power: float
    progress: bool
    device: torch.device


@dataclass
class OperatorRecord:
    name: str
    origin_task: str
    module: "ClosedOperator"
    parameter_count: int
    update_count: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(float(self.d_head))
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.view(1, 1, seq_len, seq_len), -torch.inf)
        attn = F.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be rank-2 [batch, seq], got shape={tuple(tokens.shape)}.")
        batch, seq_len = tokens.shape
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.position_embedding.num_embeddings}."
            )
        positions = torch.arange(seq_len, device=tokens.device).view(1, seq_len).expand(batch, seq_len)
        h = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.lm_head(h), h

    def code(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(token_ids)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes @ self.token_embedding.weight.T

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class ClosedOperator(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        first = self.net[0]
        second = self.net[2]
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise TypeError("ClosedOperator must be Linear -> ReLU -> Linear.")
        nn.init.normal_(first.weight, mean=0.0, std=1.0 / math.sqrt(float(d_model)))
        nn.init.constant_(first.bias, 0.01)
        nn.init.normal_(second.weight, mean=0.0, std=1.0 / math.sqrt(float(hidden_dim)))
        nn.init.zeros_(second.bias)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        return self.net(code)


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown device {name!r}. Expected cpu, cuda, or mps.")


def tensor_is_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains non-finite values.")


def progress_range(total: int, desc: str, cfg: Config) -> Any:
    if total <= 0:
        raise ValueError(f"progress total must be positive, got {total}.")
    return tqdm(
        range(total),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not cfg.progress,
    )


def load_spec(path: Path) -> SemanticSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = ["input_domain", "relations", "policy_train_streams", "policy_test_streams"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Lexicon spec missing required key(s): {missing}.")
    input_domain = tuple(str(item) for item in raw["input_domain"])
    if len(set(input_domain)) != len(input_domain):
        raise ValueError("input_domain contains duplicate tokens.")
    relations: dict[str, dict[str, str]] = {}
    for task_name, mapping in raw["relations"].items():
        task = str(task_name).upper()
        if task in relations:
            raise ValueError(f"Duplicate relation after uppercasing: {task}.")
        relation = {str(source): str(target) for source, target in mapping.items()}
        missing_sources = [token for token in input_domain if token not in relation]
        if missing_sources:
            raise ValueError(f"Relation {task} is missing input-domain tokens: {missing_sources}.")
        relations[task] = relation
    train_streams = tuple(tuple(str(item).upper() for item in stream) for stream in raw["policy_train_streams"])
    test_streams = tuple(tuple(str(item).upper() for item in stream) for stream in raw["policy_test_streams"])
    validate_streams(train_streams, relations, "policy_train_streams")
    validate_streams(test_streams, relations, "policy_test_streams")
    return SemanticSpec(input_domain, relations, train_streams, test_streams)


def validate_streams(
    streams: tuple[tuple[str, ...], ...],
    relations: dict[str, dict[str, str]],
    name: str,
) -> None:
    if not streams:
        raise ValueError(f"{name} must contain at least one stream.")
    for stream_index, stream in enumerate(streams):
        if not stream:
            raise ValueError(f"{name}[{stream_index}] is empty.")
        for item in stream:
            task_name = item.removeprefix("REPAIR_")
            if task_name not in relations:
                raise ValueError(f"{name}[{stream_index}] references unknown task {item!r}.")


def make_vocab(spec: SemanticSpec) -> Vocab:
    tokens: list[str] = list(SPECIAL_TOKENS)
    for token in spec.input_domain:
        if token not in tokens:
            tokens.append(token)
    for task_name, relation in spec.relations.items():
        task_token = f"<task:{task_name.lower()}>"
        if task_token not in tokens:
            tokens.append(task_token)
        for target in relation.values():
            if target not in tokens:
                tokens.append(target)
    token_to_id = {token: index for index, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=tuple(tokens))


def operator_parameter_count(cfg: Config) -> int:
    return cfg.d_model * cfg.operator_hidden_dim + cfg.operator_hidden_dim + cfg.operator_hidden_dim * cfg.d_model + cfg.d_model


def model_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def make_lm_corpus(spec: SemanticSpec, vocab: Vocab, cfg: Config) -> torch.Tensor:
    bos = vocab.encode("<bos>")
    eos = vocab.encode("<eos>")
    sequences: list[list[int]] = []
    for task_name, relation in spec.relations.items():
        task_token = vocab.encode(f"<task:{task_name.lower()}>")
        for source in spec.input_domain:
            target = relation[source]
            sequences.append([bos, vocab.encode(source), task_token, vocab.encode(target), eos])
    max_len = max(len(sequence) for sequence in sequences)
    if max_len > cfg.max_seq_len:
        raise ValueError(f"LM corpus sequence length {max_len} exceeds max_seq_len={cfg.max_seq_len}.")
    pad = vocab.encode("<pad>")
    padded = [sequence + [pad] * (max_len - len(sequence)) for sequence in sequences]
    return torch.tensor(padded, dtype=torch.long, device=cfg.device)


def code_distance_scale(model: TinyCausalLM, token_ids: torch.Tensor) -> float:
    with torch.no_grad():
        codes = model.code(token_ids)
        distances = torch.cdist(codes, codes, p=2.0).pow(2)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
        nearest = distances.masked_select(mask).view(distances.shape[0], distances.shape[0] - 1).min(dim=1).values
        scale = nearest.mean().item()
    if scale <= 0.0:
        raise RuntimeError("Code distance scale is non-positive; token manifold collapsed.")
    return scale


def closure_loss(out_code: torch.Tensor, target_code: torch.Tensor) -> torch.Tensor:
    return (out_code - target_code).pow(2).sum(dim=-1).mean()


def manifold_error(out_code: torch.Tensor, model: TinyCausalLM, code_token_ids: torch.Tensor) -> torch.Tensor:
    codebook = model.code(code_token_ids).detach()
    distances = torch.cdist(out_code, codebook, p=2.0).pow(2)
    return distances.min(dim=1).values.mean()


def train_base_lm(model: TinyCausalLM, corpus: torch.Tensor, code_token_ids: torch.Tensor, cfg: Config) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
    for _ in progress_range(cfg.base_epochs, "base manifold", cfg):
        optimizer.zero_grad()
        logits, _ = model(corpus[:, :-1])
        targets = corpus[:, 1:]
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        codes = model.code(code_token_ids)
        ae_logits = model.decode(codes)
        ae_loss = F.cross_entropy(ae_logits, code_token_ids)
        distances = torch.cdist(codes, codes, p=2.0).pow(2)
        mask = ~torch.eye(codes.shape[0], dtype=torch.bool, device=cfg.device)
        separation = F.relu(cfg.separation_margin - distances.masked_select(mask)).mean()
        loss = cfg.lm_weight * lm_loss + cfg.ae_weight * ae_loss + cfg.separation_weight * separation
        tensor_is_finite("base lm loss", loss)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model.decode(model.code(code_token_ids))
        reconstruction = (logits.argmax(dim=-1) == code_token_ids).float().mean().item()
    if reconstruction < 1.0:
        raise RuntimeError(f"Base token manifold failed reconstruction: accuracy={reconstruction:.4f}.")
    model.freeze()


def make_task_data(
    spec: SemanticSpec,
    vocab: Vocab,
    task_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if task_name not in spec.relations:
        raise KeyError(f"Unknown task {task_name!r}. Available: {sorted(spec.relations)}")
    relation = spec.relations[task_name]
    inputs = [vocab.encode(source) for source in spec.input_domain]
    targets = [vocab.encode(relation[source]) for source in spec.input_domain]
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def new_operator(origin_task: str, cfg: Config, name: str) -> OperatorRecord:
    module = ClosedOperator(cfg.d_model, cfg.operator_hidden_dim).to(cfg.device)
    return OperatorRecord(name=name, origin_task=origin_task, module=module, parameter_count=operator_parameter_count(cfg))


def direct_operator_program(record: OperatorRecord) -> Program:
    return ("op", record.name, (("var", 0),))


def program_depth(program: Program) -> int:
    if program[0] == "var":
        return 0
    if program[0] == "op":
        return 1 + program_depth(program[2][0])
    raise ValueError(f"Unknown program node {program[0]!r}.")


def program_to_str(program: Program) -> str:
    if program[0] == "var":
        return "x"
    if program[0] == "op":
        return f"{program[1]}({program_to_str(program[2][0])})"
    raise ValueError(f"Unknown program node {program[0]!r}.")


def parse_materialized_program(program_text: str, library: dict[str, OperatorRecord]) -> Program:
    text = program_text.strip()
    if text == "x":
        return ("var", 0)
    open_index = text.find("(")
    if open_index <= 0 or not text.endswith(")"):
        raise ValueError(f"Cannot parse program {program_text!r}.")
    op_name = text[:open_index]
    if op_name not in library:
        raise KeyError(f"Program references missing operator {op_name!r}.")
    inner = text[open_index + 1 : -1]
    return ("op", op_name, (parse_materialized_program(inner, library),))


def direct_program_operator_name(program: Program) -> str:
    if program[0] != "op" or program[2][0][0] != "var":
        raise ValueError(f"Expected direct operator program, got {program_to_str(program)}.")
    return str(program[1])


def eval_program(
    program: Program,
    model: TinyCausalLM,
    library: dict[str, OperatorRecord],
    inputs: torch.Tensor,
) -> torch.Tensor:
    if program[0] == "var":
        return model.code(inputs)
    if program[0] == "op":
        op_name = str(program[1])
        if op_name not in library:
            raise KeyError(f"Program references missing operator {op_name!r}.")
        child = eval_program(program[2][0], model, library, inputs)
        return library[op_name].module(child)
    raise ValueError(f"Unknown program node {program[0]!r}.")


def generate_programs(library: dict[str, OperatorRecord], max_depth: int, max_programs: int) -> list[Program]:
    programs: list[Program] = [("var", 0)]
    by_depth: list[list[Program]] = [[("var", 0)]]
    seen = {program_to_str(("var", 0))}
    for depth in range(1, max_depth + 1):
        depth_programs: list[Program] = []
        child_pool = [program for level in by_depth for program in level]
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
                        f"Program search exceeded max_programs={max_programs}; reduce search_depth or increase max_programs."
                    )
        by_depth.append(depth_programs)
    return programs


def evaluate_program(
    model: TinyCausalLM,
    library: dict[str, OperatorRecord],
    program: Program,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    code_token_ids: torch.Tensor,
    distance_scale: float,
) -> dict[str, float]:
    with torch.no_grad():
        out_code = eval_program(program, model, library, inputs)
        logits = model.decode(out_code)
        preds = logits.argmax(dim=-1)
        close = closure_loss(out_code, model.code(targets)).item()
        manifold = manifold_error(out_code, model, code_token_ids).item()
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "loss": float(F.cross_entropy(logits, targets).item()),
            "closure_norm": float(close / distance_scale),
            "manifold_norm": float(manifold / distance_scale),
        }


def evaluate_direct_operator(
    model: TinyCausalLM,
    record: OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    code_token_ids: torch.Tensor,
    distance_scale: float,
) -> dict[str, float]:
    return evaluate_program(model, {record.name: record}, direct_operator_program(record), inputs, targets, code_token_ids, distance_scale)


def search_best_program(
    model: TinyCausalLM,
    library: dict[str, OperatorRecord],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    code_token_ids: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> Program:
    programs = generate_programs(library, cfg.search_depth, cfg.max_programs)
    best_program: Program | None = None
    best_metrics: dict[str, float] | None = None
    best_depth: int | None = None
    for program in programs:
        metrics = evaluate_program(model, library, program, inputs, targets, code_token_ids, distance_scale)
        depth = program_depth(program)
        if best_metrics is None:
            best_program = program
            best_metrics = metrics
            best_depth = depth
            continue
        if metrics["accuracy"] > best_metrics["accuracy"]:
            best_program = program
            best_metrics = metrics
            best_depth = depth
        elif metrics["accuracy"] == best_metrics["accuracy"]:
            if best_depth is None or depth < best_depth:
                best_program = program
                best_metrics = metrics
                best_depth = depth
            elif depth == best_depth and metrics["closure_norm"] < best_metrics["closure_norm"]:
                best_program = program
                best_metrics = metrics
                best_depth = depth
    if best_program is None:
        raise RuntimeError("Program search did not produce a best program.")
    return best_program


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
    spec: SemanticSpec,
    vocab: Vocab,
    model: TinyCausalLM,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    learned_tasks: list[str],
    current_task: str,
    current_program: Program,
    code_token_ids: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    current_inputs, current_targets = make_task_data(spec, vocab, current_task, cfg.device)
    current = evaluate_program(model, library, current_program, current_inputs, current_targets, code_token_ids, distance_scale)
    old_accs: list[float] = []
    old_closures: list[float] = []
    for task_name in learned_tasks:
        if task_name not in task_to_program:
            raise KeyError(f"Missing learned program for old task {task_name}.")
        inputs, targets = make_task_data(spec, vocab, task_name, cfg.device)
        metrics = evaluate_program(model, library, task_to_program[task_name], inputs, targets, code_token_ids, distance_scale)
        old_accs.append(metrics["accuracy"])
        old_closures.append(metrics["closure_norm"])
    return {
        "new_acc": current["accuracy"],
        "new_loss": current["loss"],
        "new_closure_norm": current["closure_norm"],
        "new_manifold_norm": current["manifold_norm"],
        "old_min_acc": safe_min(old_accs, 1.0),
        "old_mean_acc": safe_mean(old_accs, 1.0),
        "old_mean_closure_norm": safe_mean(old_closures, 0.0),
    }


def train_new_operator(
    model: TinyCausalLM,
    record: OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    require_perfect: bool,
    code_token_ids: torch.Tensor,
    distance_scale: float,
) -> None:
    optimizer = torch.optim.Adam(record.module.parameters(), lr=cfg.operator_lr)
    input_code = model.code(inputs).detach()
    target_code = model.code(targets).detach()
    for _ in progress_range(cfg.operator_epochs, f"train {record.name}", cfg):
        optimizer.zero_grad()
        out_code = record.module(input_code)
        logits = model.decode(out_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * closure_loss(out_code, target_code)
        tensor_is_finite("new operator loss", loss)
        loss.backward()
        optimizer.step()
    metrics = evaluate_direct_operator(model, record, inputs, targets, code_token_ids, distance_scale)
    if require_perfect and metrics["accuracy"] < 1.0:
        raise RuntimeError(f"Operator {record.name} failed to fit {record.origin_task}: accuracy={metrics['accuracy']:.4f}.")


def operator_layers(record: OperatorRecord) -> tuple[nn.Linear, nn.Linear]:
    first = record.module.net[0]
    second = record.module.net[2]
    if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
        raise TypeError("ClosedOperator must be Linear -> ReLU -> Linear.")
    return first, second


def forward_with_hidden(record: OperatorRecord, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = operator_layers(record)
    hidden = F.relu(first(code))
    return second(hidden), hidden


def medium_band_gate(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError(f"update_closure_high must be greater than update_closure_low, got {high} <= {low}.")
    if value <= low or value >= high:
        return 0.0
    midpoint = 0.5 * (low + high)
    if value <= midpoint:
        return (value - low) / (midpoint - low)
    return (high - value) / (high - midpoint)


def normalize(values: torch.Tensor) -> torch.Tensor:
    tensor_is_finite("normalization input", values)
    max_value = values.max().item()
    if max_value <= 0.0:
        return torch.zeros_like(values)
    return values / max_value


def structural_update(
    model: TinyCausalLM,
    record: OperatorRecord,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    first, second = operator_layers(record)
    input_code = model.code(inputs).detach()
    target_code = model.code(targets).detach()
    gate_means: list[float] = []
    active_counts: list[float] = []
    for _ in progress_range(cfg.update_epochs, f"update {record.name}", cfg):
        record.module.zero_grad(set_to_none=True)
        out_code, hidden = forward_with_hidden(record, input_code)
        logits = model.decode(out_code)
        close = closure_loss(out_code, target_code)
        loss = F.cross_entropy(logits, targets) + cfg.lambda_closure * close
        tensor_is_finite("structural update loss", loss)
        loss.backward()
        closure_gate = medium_band_gate(close.detach().item() / distance_scale, cfg.update_closure_low, cfg.update_closure_high)
        if first.weight.grad is None or first.bias.grad is None or second.weight.grad is None or second.bias.grad is None:
            raise RuntimeError(f"Missing gradients during structural update for {record.name}.")
        activation = hidden.detach().abs().mean(dim=0)
        downstream_weight = second.weight.detach().norm(dim=0)
        incoming_grad = first.weight.grad.detach().norm(dim=1) + first.bias.grad.detach().abs()
        downstream_grad = second.weight.grad.detach().norm(dim=0)
        need = normalize(activation * (incoming_grad + downstream_grad)).pow(cfg.structural_need_power)
        risk = normalize(activation * downstream_weight).pow(cfg.structural_risk_power)
        gates = closure_gate * need * risk
        tensor_is_finite("structural gates", gates)
        with torch.no_grad():
            for neuron_index, gate_tensor in enumerate(gates):
                gate = float(gate_tensor.item())
                first.weight[neuron_index].add_(first.weight.grad[neuron_index], alpha=-cfg.update_lr * gate)
                first.bias[neuron_index].add_(first.bias.grad[neuron_index], alpha=-cfg.update_lr * gate)
                second.weight[:, neuron_index].add_(second.weight.grad[:, neuron_index], alpha=-cfg.update_lr * gate)
            second.bias.add_(second.bias.grad, alpha=-cfg.update_lr * float(gates.mean().item()))
        gate_means.append(float(gates.mean().item()))
        active_counts.append(float((gates > 0).sum().item()))
    record.update_count += 1
    return {
        "gate_mean": float(np.mean(gate_means)),
        "active_neurons": float(np.mean(active_counts)),
    }


def perturb_operator(record: OperatorRecord, noise_std: float) -> None:
    if noise_std <= 0.0:
        raise ValueError(f"repair_noise_std must be positive, got {noise_std}.")
    with torch.no_grad():
        for name, parameter in record.module.named_parameters():
            noise = torch.randn_like(parameter) * noise_std
            tensor_is_finite(f"repair noise {record.name}.{name}", noise)
            parameter.add_(noise)


def inject_repair_noise(
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    task_name: str,
    noise_std: float,
) -> str:
    if task_name not in task_to_program:
        raise KeyError(f"Repair requested for {task_name}, but no learned program exists.")
    op_name = direct_program_operator_name(task_to_program[task_name])
    if op_name not in library:
        raise KeyError(f"Repair program references missing operator {op_name!r}.")
    perturb_operator(library[op_name], noise_std)
    return op_name


def clone_library(library: dict[str, OperatorRecord]) -> dict[str, OperatorRecord]:
    cloned: dict[str, OperatorRecord] = {}
    for name, record in library.items():
        cloned[name] = OperatorRecord(
            name=record.name,
            origin_task=record.origin_task,
            module=copy.deepcopy(record.module),
            parameter_count=record.parameter_count,
            update_count=record.update_count,
        )
    return cloned


def counterfactual_score(metrics: dict[str, float], action: str, new_parameters: int, cfg: Config) -> tuple[float, bool]:
    closure_limit = cfg.reuse_closure_threshold if action in {"reuse", "compose"} else cfg.update_closure_high
    safe = (
        metrics["new_acc"] >= cfg.reuse_acc_threshold
        and metrics["old_min_acc"] >= cfg.reuse_acc_threshold
        and metrics["old_mean_closure_norm"] <= cfg.reuse_closure_threshold
        and metrics["new_closure_norm"] <= closure_limit
    )
    penalties = {"reuse": 0.00, "compose": 0.00, "update": 0.01, "allocate": 0.03}
    if action not in penalties:
        raise ValueError(f"Unknown action {action!r}.")
    score = (
        10.0 * metrics["new_acc"]
        + 4.0 * metrics["old_min_acc"]
        - metrics["new_closure_norm"]
        - metrics["old_mean_closure_norm"]
        - penalties[action]
        - 1e-6 * new_parameters
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
    update_stats: dict[str, float] | None = None,
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
    spec: SemanticSpec,
    vocab: Vocab,
    model: TinyCausalLM,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    learned_tasks: list[str],
    task_name: str,
    code_token_ids: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> list[dict[str, Any]]:
    inputs, targets = make_task_data(spec, vocab, task_name, cfg.device)
    candidates: list[dict[str, Any]] = []

    best_program = search_best_program(model, library, inputs, targets, code_token_ids, cfg, distance_scale)
    action = action_name_for_program(best_program)
    candidate_task_to_program = dict(task_to_program)
    candidate_task_to_program[task_name] = best_program
    metrics = evaluate_state(
        spec,
        vocab,
        model,
        library,
        candidate_task_to_program,
        learned_tasks,
        task_name,
        best_program,
        code_token_ids,
        cfg,
        distance_scale,
    )
    score, safe = counterfactual_score(metrics, action, 0, cfg)
    candidates.append(candidate_row(action, best_program, metrics, score, safe, 0))

    for record in library.values():
        shadow_library = clone_library(library)
        shadow_record = shadow_library[record.name]
        shadow_update_cfg = replace(cfg, update_epochs=cfg.shadow_update_epochs)
        update_stats = structural_update(model, shadow_record, inputs, targets, shadow_update_cfg, distance_scale)
        program = direct_operator_program(shadow_record)
        candidate_task_to_program = dict(task_to_program)
        candidate_task_to_program[task_name] = program
        metrics = evaluate_state(
            spec,
            vocab,
            model,
            shadow_library,
            candidate_task_to_program,
            learned_tasks,
            task_name,
            program,
            code_token_ids,
            cfg,
            distance_scale,
        )
        score, safe = counterfactual_score(metrics, "update", 0, cfg)
        candidates.append(candidate_row("update", program, metrics, score, safe, 0, update_stats))

    shadow_library = clone_library(library)
    op_name = f"OP_{task_name}_{len(library)}"
    record = new_operator(task_name, cfg, op_name)
    shadow_operator_cfg = replace(cfg, operator_epochs=cfg.shadow_operator_epochs)
    train_new_operator(model, record, inputs, targets, shadow_operator_cfg, False, code_token_ids, distance_scale)
    shadow_library[op_name] = record
    program = direct_operator_program(record)
    candidate_task_to_program = dict(task_to_program)
    candidate_task_to_program[task_name] = program
    metrics = evaluate_state(
        spec,
        vocab,
        model,
        shadow_library,
        candidate_task_to_program,
        learned_tasks,
        task_name,
        program,
        code_token_ids,
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


def materialize_choice(
    spec: SemanticSpec,
    vocab: Vocab,
    model: TinyCausalLM,
    library: dict[str, OperatorRecord],
    task_to_program: dict[str, Program],
    task_name: str,
    chosen: dict[str, Any],
    code_token_ids: torch.Tensor,
    cfg: Config,
    distance_scale: float,
) -> dict[str, float]:
    inputs, targets = make_task_data(spec, vocab, task_name, cfg.device)
    action = chosen["action"]
    if action in {"reuse", "compose"}:
        task_to_program[task_name] = parse_materialized_program(chosen["program"], library)
        return {}
    if action == "allocate":
        op_name = f"OP_{task_name}_{len(library)}"
        record = new_operator(task_name, cfg, op_name)
        train_new_operator(model, record, inputs, targets, cfg, True, code_token_ids, distance_scale)
        library[op_name] = record
        task_to_program[task_name] = direct_operator_program(record)
        return {}
    if action == "update":
        program = parse_materialized_program(chosen["program"], library)
        op_name = direct_program_operator_name(program)
        if op_name not in library:
            raise KeyError(f"Update selected missing operator {op_name!r}.")
        stats = structural_update(model, library[op_name], inputs, targets, cfg, distance_scale)
        task_to_program[task_name] = direct_operator_program(library[op_name])
        return stats
    raise ValueError(f"Unknown action {action!r}.")


def best_candidates_by_action(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        action = candidate["action"]
        if action not in ACTION_TO_INDEX:
            raise ValueError(f"Unknown candidate action {action!r}.")
        if action not in grouped or candidate["score"] > grouped[action]["score"]:
            grouped[action] = candidate
    if not grouped:
        raise RuntimeError("No candidates available for action grouping.")
    return grouped


def metric_value(candidate: dict[str, Any], key: str) -> float:
    metrics = candidate["metrics"]
    if key not in metrics:
        raise KeyError(f"Candidate metrics missing {key!r}.")
    return float(metrics[key])


def policy_features(candidates: list[dict[str, Any]], parameter_scale: float) -> list[float]:
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
    features.extend([float(len(candidates)), float(len(grouped))])
    return features


def select_policy_candidate(
    policy: LearnedActionPolicy,
    candidates: list[dict[str, Any]],
    cfg: Config,
    parameter_scale: float,
) -> tuple[dict[str, Any], str, str, bool]:
    grouped = best_candidates_by_action(candidates)
    features = torch.tensor([policy_features(candidates, parameter_scale)], dtype=torch.float32, device=cfg.device)
    with torch.no_grad():
        logits = policy(features).squeeze(0)
    raw_index = int(torch.argmax(logits).item())
    raw_action = ACTION_NAMES[raw_index]
    mask = torch.tensor([action in grouped for action in ACTION_NAMES], dtype=torch.bool, device=cfg.device)
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    if not torch.isfinite(masked_logits).any().item():
        raise RuntimeError("Policy mask removed every action.")
    chosen_index = int(torch.argmax(masked_logits).item())
    chosen_action = ACTION_NAMES[chosen_index]
    if chosen_action not in grouped:
        raise RuntimeError(f"Masked policy selected unavailable action {chosen_action!r}.")
    return grouped[chosen_action], raw_action, chosen_action, raw_action != chosen_action


def make_run_state(spec: SemanticSpec, vocab: Vocab, cfg: Config, seed: int) -> tuple[TinyCausalLM, torch.Tensor, float]:
    set_seed(seed)
    model = TinyCausalLM(vocab.size, cfg).to(cfg.device)
    corpus = make_lm_corpus(spec, vocab, cfg)
    code_tokens = sorted({vocab.encode(token) for token in spec.input_domain} | {
        vocab.encode(target) for relation in spec.relations.values() for target in relation.values()
    })
    code_token_ids = torch.tensor(code_tokens, dtype=torch.long, device=cfg.device)
    train_base_lm(model, corpus, code_token_ids, cfg)
    distance_scale = code_distance_scale(model, code_token_ids)
    return model, code_token_ids, distance_scale


def run_stream(
    spec: SemanticSpec,
    vocab: Vocab,
    model: TinyCausalLM,
    code_token_ids: torch.Tensor,
    distance_scale: float,
    stream: tuple[str, ...],
    cfg: Config,
    policy: LearnedActionPolicy | None,
    phase: str,
    seed: int,
    stream_index: int,
    collect_features: list[list[float]] | None = None,
    collect_labels: list[int] | None = None,
) -> list[dict[str, Any]]:
    library: dict[str, OperatorRecord] = {}
    task_to_program: dict[str, Program] = {}
    learned_tasks: list[str] = []
    rows: list[dict[str, Any]] = []
    parameter_scale = float(operator_parameter_count(cfg))
    for step_index, event in enumerate(stream):
        repair = event.startswith("REPAIR_")
        task_name = event.removeprefix("REPAIR_")
        repaired_operator: str | None = None
        if repair:
            repaired_operator = inject_repair_noise(library, task_to_program, task_name, cfg.repair_noise_std)
        candidates = build_candidates(
            spec,
            vocab,
            model,
            library,
            task_to_program,
            learned_tasks,
            task_name,
            code_token_ids,
            cfg,
            distance_scale,
        )
        teacher = choose_candidate(candidates)
        if collect_features is not None and collect_labels is not None:
            collect_features.append(policy_features(candidates, parameter_scale))
            collect_labels.append(ACTION_TO_INDEX[teacher["action"]])
        if policy is None:
            chosen = teacher
            raw_action = teacher["action"]
            chosen_action = teacher["action"]
            masked_preference = False
        else:
            chosen, raw_action, chosen_action, masked_preference = select_policy_candidate(policy, candidates, cfg, parameter_scale)
        update_stats = materialize_choice(
            spec,
            vocab,
            model,
            library,
            task_to_program,
            task_name,
            chosen,
            code_token_ids,
            cfg,
            distance_scale,
        )
        if task_name not in task_to_program:
            raise RuntimeError(f"Task {task_name} was not materialized.")
        final_metrics = evaluate_state(
            spec,
            vocab,
            model,
            library,
            task_to_program,
            learned_tasks,
            task_name,
            task_to_program[task_name],
            code_token_ids,
            cfg,
            distance_scale,
        )
        if not repair and task_name not in learned_tasks:
            learned_tasks.append(task_name)
        rows.append(
            {
                "phase": phase,
                "seed": seed,
                "stream_index": stream_index,
                "step_index": step_index,
                "event": event,
                "task": task_name,
                "repair": repair,
                "repaired_operator": repaired_operator,
                "teacher_action": teacher["action"],
                "chosen_action": chosen_action,
                "raw_action": raw_action,
                "masked_preference": masked_preference,
                "action_correct": chosen_action == teacher["action"],
                "safe": chosen["safe"],
                "program": program_to_str(task_to_program[task_name]),
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


def train_policy(
    features: list[list[float]],
    labels: list[int],
    cfg: Config,
    args: argparse.Namespace,
) -> LearnedActionPolicy:
    if not features:
        raise ValueError("No policy examples were collected.")
    if len(features) != len(labels):
        raise ValueError(f"Feature/label length mismatch: {len(features)} != {len(labels)}.")
    feature_dim = len(features[0])
    for index, row in enumerate(features):
        if len(row) != feature_dim:
            raise ValueError(f"Feature row {index} has length {len(row)}, expected {feature_dim}.")
    set_seed(args.policy_seed)
    policy = LearnedActionPolicy(feature_dim, args.policy_hidden_dim, len(ACTION_NAMES)).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    x = torch.tensor(features, dtype=torch.float32, device=cfg.device)
    y = torch.tensor(labels, dtype=torch.long, device=cfg.device)
    for _ in progress_range(args.policy_epochs, "train action policy", cfg):
        optimizer.zero_grad()
        loss = F.cross_entropy(policy(x), y)
        tensor_is_finite("policy loss", loss)
        loss.backward()
        optimizer.step()
    return policy


def summarize(rows: list[dict[str, Any]], base_parameter_count: int, train_action_counts: dict[str, int] | None) -> dict[str, Any]:
    if not rows:
        raise ValueError("No rows to summarize.")
    final_rows: list[dict[str, Any]] = []
    for key in sorted({(row["seed"], row["stream_index"], row["phase"]) for row in rows}):
        phase_rows = [row for row in rows if (row["seed"], row["stream_index"], row["phase"]) == key]
        final_rows.append(max(phase_rows, key=lambda row: row["step_index"]))
    summary: dict[str, Any] = {
        "base_parameter_count": base_parameter_count,
        "train_action_counts": train_action_counts,
        "overall": {
            "action_accuracy": float(np.mean([float(row["action_correct"]) for row in rows])),
            "unsafe_choice_rate": float(np.mean([float(not row["safe"]) for row in rows])),
            "masked_preference_rate": float(np.mean([float(row["masked_preference"]) for row in rows])),
            "new_acc": float(np.mean([row["new_acc"] for row in rows])),
            "old_min_acc": float(np.mean([row["old_min_acc"] for row in rows])),
            "closure": float(np.mean([row["new_closure_norm"] for row in rows])),
            "final_operator_count": float(np.mean([row["operator_count"] for row in final_rows])),
            "final_new_parameters": float(np.mean([row["new_parameters_added"] for row in final_rows])),
        },
        "events": {},
    }
    for event in sorted({row["event"] for row in rows}):
        event_rows = [row for row in rows if row["event"] == event]
        summary["events"][event] = {
            "teacher": count_values([row["teacher_action"] for row in event_rows]),
            "chosen": count_values([row["chosen_action"] for row in event_rows]),
            "action_accuracy": float(np.mean([float(row["action_correct"]) for row in event_rows])),
            "new_acc": float(np.mean([row["new_acc"] for row in event_rows])),
            "old_min_acc": float(np.mean([row["old_min_acc"] for row in event_rows])),
            "closure": float(np.mean([row["new_closure_norm"] for row in event_rows])),
            "operator_count": float(np.mean([row["operator_count"] for row in event_rows])),
        }
    return summary


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config_from_args(args)
    spec = load_spec(args.lexicon_json)
    vocab = make_vocab(spec)
    train_seed_count = args.policy_train_seed_count if args.policy_train_seed_count is not None else args.seed_count
    eval_seed_count = args.eval_seed_count if args.eval_seed_count is not None else args.seed_count
    train_features: list[list[float]] = []
    train_labels: list[int] = []
    train_rows: list[dict[str, Any]] = []
    base_parameter_count: int | None = None
    for seed in tqdm(
        range(train_seed_count),
        desc="collect policy examples",
        dynamic_ncols=True,
        disable=not cfg.progress,
    ):
        print(f"collect_seed={seed}")
        model, code_token_ids, distance_scale = make_run_state(spec, vocab, cfg, seed)
        if base_parameter_count is None:
            base_parameter_count = model_parameter_count(model)
        for stream_index, stream in enumerate(spec.train_streams):
            train_rows.extend(
                run_stream(
                    spec,
                    vocab,
                    model,
                    code_token_ids,
                    distance_scale,
                    stream,
                    cfg,
                    policy=None,
                    phase="teacher_train",
                    seed=seed,
                    stream_index=stream_index,
                    collect_features=train_features,
                    collect_labels=train_labels,
                )
            )
    if base_parameter_count is None:
        raise RuntimeError("No base model was created.")
    policy = train_policy(train_features, train_labels, cfg, args)
    test_rows: list[dict[str, Any]] = []
    for seed in tqdm(
        range(eval_seed_count),
        desc="evaluate learned policy",
        dynamic_ncols=True,
        disable=not cfg.progress,
    ):
        eval_seed = args.test_seed_offset + seed
        print(f"eval_seed={eval_seed}")
        model, code_token_ids, distance_scale = make_run_state(spec, vocab, cfg, eval_seed)
        for stream_index, stream in enumerate(spec.test_streams):
            test_rows.extend(
                run_stream(
                    spec,
                    vocab,
                    model,
                    code_token_ids,
                    distance_scale,
                    stream,
                    cfg,
                    policy=policy,
                    phase="learned_policy_test",
                    seed=eval_seed,
                    stream_index=stream_index,
                )
            )
    report = {
        "mode": "tiny_lm_learned_geometry_policy",
        "lexicon_json": str(args.lexicon_json),
        "vocab_size": vocab.size,
        "actions": list(ACTION_NAMES),
        "config": serializable_config(args),
        "summary": summarize(test_rows, base_parameter_count, count_values([ACTION_NAMES[label] for label in train_labels])),
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
    print_summary(report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    overall = summary["overall"]
    print("\nTINY LM LEARNED LATENT-GEOMETRY POLICY SUMMARY")
    print("=" * 132)
    print(f"base_parameters={summary['base_parameter_count']}")
    print(f"train_action_counts={summary['train_action_counts']}")
    print(
        "overall: "
        f"action_acc={overall['action_accuracy']:.4f}, "
        f"unsafe={overall['unsafe_choice_rate']:.4f}, "
        f"masked_pref={overall['masked_preference_rate']:.4f}, "
        f"new_acc={overall['new_acc']:.4f}, "
        f"old_min={overall['old_min_acc']:.4f}, "
        f"closure={overall['closure']:.4f}, "
        f"final_ops={overall['final_operator_count']:.2f}, "
        f"new_params={overall['final_new_parameters']:.2f}"
    )
    print("-" * 132)
    print(f"{'event':<18} {'teacher':<28} {'chosen':<28} {'act_acc':<10} {'new_acc':<10} {'old_min':<10} {'closure':<10} {'ops':<8}")
    print("-" * 132)
    for event, item in summary["events"].items():
        print(
            f"{event:<18} {str(item['teacher']):<28} {str(item['chosen']):<28} "
            f"{item['action_accuracy']:<10.3f} {item['new_acc']:<10.3f} "
            f"{item['old_min_acc']:<10.3f} {item['closure']:<10.4f} {item['operator_count']:<8.2f}"
        )
    print("=" * 132)


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "output_json"}


def write_json_report(report: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output-json already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def config_from_args(args: argparse.Namespace) -> Config:
    device = resolve_device(args.device)
    return Config(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        operator_hidden_dim=args.operator_hidden_dim,
        base_epochs=args.base_epochs,
        operator_epochs=args.operator_epochs,
        update_epochs=args.update_epochs,
        shadow_operator_epochs=args.shadow_operator_epochs
        if args.shadow_operator_epochs is not None
        else args.operator_epochs,
        shadow_update_epochs=args.shadow_update_epochs
        if args.shadow_update_epochs is not None
        else args.update_epochs,
        base_lr=args.base_lr,
        operator_lr=args.operator_lr,
        update_lr=args.update_lr,
        lambda_closure=args.lambda_closure,
        separation_margin=args.separation_margin,
        separation_weight=args.separation_weight,
        lm_weight=args.lm_weight,
        ae_weight=args.ae_weight,
        reuse_acc_threshold=args.reuse_acc_threshold,
        reuse_closure_threshold=args.reuse_closure_threshold,
        update_closure_low=args.update_closure_low,
        update_closure_high=args.update_closure_high,
        search_depth=args.search_depth,
        max_programs=args.max_programs,
        repair_noise_std=args.repair_noise_std,
        structural_risk_power=args.structural_risk_power,
        structural_need_power=args.structural_need_power,
        progress=not args.no_progress,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon-json", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--policy-train-seed-count", type=int)
    parser.add_argument("--eval-seed-count", type=int)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--operator-hidden-dim", type=int, default=128)
    parser.add_argument("--base-epochs", type=int, default=1200)
    parser.add_argument("--operator-epochs", type=int, default=1200)
    parser.add_argument("--update-epochs", type=int, default=1000)
    parser.add_argument("--shadow-operator-epochs", type=int)
    parser.add_argument("--shadow-update-epochs", type=int)
    parser.add_argument("--base-lr", type=float, default=0.003)
    parser.add_argument("--operator-lr", type=float, default=0.01)
    parser.add_argument("--update-lr", type=float, default=0.005)
    parser.add_argument("--lambda-closure", type=float, default=10.0)
    parser.add_argument("--separation-margin", type=float, default=2.0)
    parser.add_argument("--separation-weight", type=float, default=0.5)
    parser.add_argument("--lm-weight", type=float, default=1.0)
    parser.add_argument("--ae-weight", type=float, default=0.5)
    parser.add_argument("--reuse-acc-threshold", type=float, default=0.98)
    parser.add_argument("--reuse-closure-threshold", type=float, default=0.03)
    parser.add_argument("--update-closure-low", type=float, default=0.001)
    parser.add_argument("--update-closure-high", type=float, default=0.75)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--max-programs", type=int, default=100000)
    parser.add_argument("--repair-noise-std", type=float, default=0.03)
    parser.add_argument("--structural-risk-power", type=float, default=1.0)
    parser.add_argument("--structural-need-power", type=float, default=1.0)
    parser.add_argument("--policy-epochs", type=int, default=800)
    parser.add_argument("--policy-lr", type=float, default=0.01)
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--policy-seed", type=int, default=12345)
    parser.add_argument("--test-seed-offset", type=int, default=10000)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.lexicon_json.exists():
        raise FileNotFoundError(f"--lexicon-json does not exist: {args.lexicon_json}")
    positive_ints = [
        "seed_count",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "max_seq_len",
        "operator_hidden_dim",
        "base_epochs",
        "operator_epochs",
        "update_epochs",
        "search_depth",
        "max_programs",
        "policy_epochs",
        "policy_hidden_dim",
    ]
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    optional_positive_ints = [
        "policy_train_seed_count",
        "eval_seed_count",
        "shadow_operator_epochs",
        "shadow_update_epochs",
    ]
    for name in optional_positive_ints:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided.")
    if args.d_model % args.n_heads != 0:
        raise ValueError("--d-model must be divisible by --n-heads.")
    positive_floats = [
        "base_lr",
        "operator_lr",
        "update_lr",
        "lambda_closure",
        "separation_margin",
        "separation_weight",
        "lm_weight",
        "ae_weight",
        "reuse_acc_threshold",
        "reuse_closure_threshold",
        "update_closure_low",
        "update_closure_high",
        "repair_noise_std",
        "structural_risk_power",
        "structural_need_power",
        "policy_lr",
    ]
    for name in positive_floats:
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.update_closure_high <= args.update_closure_low:
        raise ValueError("--update-closure-high must be greater than --update-closure-low.")


def main() -> None:
    args = parse_args()
    report = run_experiment(args)
    if args.output_json is not None:
        write_json_report(report, args.output_json)
        print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
