"""End-to-end neural controller for toy continual learning.

This experiment removes explicit role labels from the update decision. A small
controller sees behavior evidence and outputs soft preserve/drop/guard gates.
Those gates select differentiable action modules on top of a frozen base
transformer.

The controller is not supervised with role labels. It is trained through the
final continual-learning objective:

    learn incoming examples
    preserve useful old behavior
    suppress obsolete old behavior
    keep guarded behavior stable
    avoid large adapter edits

The true roles are used only for reporting whether the learned gates match the
intended toy-world split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_mini_cl_world_demo import final_residual
from experiments.gco_math.gco_tiny_auto_role_controller import (
    build_raw_stream,
    count_person_references,
    encode_raw_groups,
    examples_by_stage1_person,
    oracle_roles,
    random_roles_matching_counts,
    role_match_report,
)
from experiments.gco_math.gco_tiny_cl_behavior_path import load_checkpoint
from experiments.gco_math.gco_tiny_recursive_controlled_architecture import (
    drop_suppression_loss,
    train_bootstrap_stage,
)
from experiments.gco_math.gco_tiny_reusable_computation_cl import (
    EncodedExample,
    QAExample,
    batch_examples,
    collect_example_logits,
    distillation_loss_for_examples,
    encode_examples,
    evaluate_examples,
    make_model_from_config,
    masked_ce_loss,
)


ROLE_NAMES = ("preserve", "drop", "guard")
ROLE_TO_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}
ACTION_NAMES = ("learn", "preserve", "drop", "guard")
FEATURE_NAMES = (
    "base_loss_norm",
    "base_exact",
    "base_token_accuracy",
    "useful_signal",
    "obsolete_signal",
    "stream_reference_signal",
    "capacity_pressure",
)


def positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def nonnegative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}.")


def positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def probability(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    allowed = {"naive", "oracle", "neural", "random"}
    unknown = sorted(set(methods).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; allowed={sorted(allowed)}.")
    return methods


def parse_people(raw: str, *, name: str) -> set[str]:
    people = {item.strip() for item in raw.split(",") if item.strip()}
    if not people:
        raise ValueError(f"--{name.replace('_', '-')} must contain at least one person.")
    known = set(examples_by_stage1_person())
    unknown = people.difference(known)
    if unknown:
        raise ValueError(f"Unknown stage-1 people in --{name.replace('_', '-')}: {sorted(unknown)}.")
    return people


def validate_args(args: argparse.Namespace) -> None:
    positive_int("stage1_epochs", args.stage1_epochs)
    positive_int("controller_epochs", args.controller_epochs)
    positive_int("batch_size", args.batch_size)
    positive_int("eval_batch_size", args.eval_batch_size)
    positive_int("adapter_rank", args.adapter_rank)
    positive_int("controller_hidden_dim", args.controller_hidden_dim)
    positive_float("lr", args.lr)
    positive_float("controller_lr", args.controller_lr)
    positive_float("distill_temperature", args.distill_temperature)
    positive_float("drop_target_probability", args.drop_target_probability)
    probability("drop_target_probability", args.drop_target_probability)
    positive_float("loss_clip", args.loss_clip)
    positive_float("role_temperature", args.role_temperature)
    nonnegative_float("lambda_protect", args.lambda_protect)
    nonnegative_float("lambda_forget", args.lambda_forget)
    nonnegative_float("lambda_guard", args.lambda_guard)
    nonnegative_float("lambda_adapter_norm", args.lambda_adapter_norm)
    nonnegative_float("lambda_gate_entropy", args.lambda_gate_entropy)
    nonnegative_float("adapter_scale", args.adapter_scale)
    positive_float("adapter_init_std", args.adapter_init_std)
    nonnegative_float("capacity_pressure", args.capacity_pressure)
    positive_int("obsolete_evidence_count", args.obsolete_evidence_count)
    if args.drop_target_probability >= 1.0:
        raise ValueError("--drop-target-probability must be below 1.0.")
    if not args.config_checkpoint.exists():
        raise FileNotFoundError(f"Config checkpoint does not exist: {args.config_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer does not exist: {args.tokenizer_path}")


class LowRankLogitAdapter(nn.Module):
    def __init__(self, *, d_model: int, vocab_size: int, rank: int, init_std: float) -> None:
        super().__init__()
        positive_int("rank", rank)
        positive_int("vocab_size", vocab_size)
        positive_float("init_std", init_std)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.up.weight, mean=0.0, std=init_std)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(torch.tanh(self.down(h)))

    def adapter_norm(self) -> torch.Tensor:
        return self.down.weight.square().mean() + self.up.weight.square().mean()


class ActionLogitAdapterModel(nn.Module):
    def __init__(
        self,
        *,
        base_model: nn.Module,
        d_model: int,
        vocab_size: int,
        rank: int,
        adapter_scale: float,
        init_std: float,
    ) -> None:
        super().__init__()
        positive_int("rank", rank)
        positive_float("init_std", init_std)
        self.base_model = base_model
        self.adapter_scale = float(adapter_scale)
        self.adapters = nn.ModuleDict(
            {
                name: LowRankLogitAdapter(d_model=d_model, vocab_size=vocab_size, rank=rank, init_std=init_std)
                for name in ACTION_NAMES
            }
        )
        self._action_gates: dict[str, torch.Tensor] | None = None

    def set_action_gates(self, gates: dict[str, torch.Tensor | float]) -> None:
        missing = sorted(set(ACTION_NAMES).difference(gates))
        extra = sorted(set(gates).difference(ACTION_NAMES))
        if missing or extra:
            raise ValueError(f"Action gates must exactly cover {ACTION_NAMES}; missing={missing} extra={extra}.")
        self._action_gates = {
            name: value if isinstance(value, torch.Tensor) else torch.tensor(float(value), dtype=torch.float32)
            for name, value in gates.items()
        }

    def adapter_norm(self) -> torch.Tensor:
        return torch.stack([adapter.adapter_norm() for adapter in self.adapters.values()]).sum()

    def base_logits_and_residual(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = final_residual(self.base_model, tokens)
        return self.base_model.lm_head(h), h

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._action_gates is None:
            raise RuntimeError("action gates have not been set before forward.")
        base_logits, h = self.base_logits_and_residual(tokens)
        delta = torch.zeros_like(base_logits)
        for name in ACTION_NAMES:
            gate = self._action_gates[name].to(device=h.device, dtype=h.dtype)
            while gate.ndim < h.ndim:
                gate = gate.unsqueeze(0)
            delta = delta + gate * self.adapters[name](h)
        return base_logits + self.adapter_scale * delta


class EndToEndCLController(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        positive_int("input_dim", input_dim)
        positive_int("hidden_dim", hidden_dim)
        self.role_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(ROLE_NAMES)),
        )
        self.learn_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def role_probs(self, features: torch.Tensor, *, temperature: float) -> torch.Tensor:
        positive_float("temperature", temperature)
        return F.softmax(self.role_net(features) / temperature, dim=-1)

    def learn_gate(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.learn_net(features.mean(dim=0, keepdim=True))).squeeze()


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def make_encoded_people(
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> dict[str, list[EncodedExample]]:
    return {
        person: encode_examples(examples, tokenizer, max_seq_len=max_seq_len)
        for person, examples in sorted(examples_by_stage1_person().items())
    }


def feature_tensor_for_people(
    *,
    model: nn.Module,
    encoded_people: dict[str, list[EncodedExample]],
    raw_groups: dict[str, list[QAExample]],
    useful_evidence_people: set[str],
    obsolete_evidence_people: set[str],
    pad_id: int,
    batch_size: int,
    device: torch.device,
    loss_clip: float,
    obsolete_evidence_count: int,
    capacity_pressure: float,
) -> tuple[list[str], torch.Tensor, dict[str, dict[str, float]]]:
    positive_float("loss_clip", loss_clip)
    positive_int("batch_size", batch_size)
    people = sorted(encoded_people)
    max_reference_count = max(count_person_references(raw_groups["stage2"] + raw_groups["stage3"], person) for person in people)
    if max_reference_count <= 0:
        raise RuntimeError("No stage-1 person is referenced in the incoming stream; useful evidence is undefined.")
    rows: list[list[float]] = []
    report: dict[str, dict[str, float]] = {}
    for person in people:
        metrics = evaluate_examples(
            model,
            encoded_people[person],
            pad_id=pad_id,
            batch_size=batch_size,
            device=device,
        )["overall"]
        raw_reference_count = count_person_references(raw_groups["stage2"] + raw_groups["stage3"], person)
        useful_signal = raw_reference_count / float(max_reference_count)
        if person not in useful_evidence_people:
            useful_signal = 0.0
        obsolete_signal = float(obsolete_evidence_count if person in obsolete_evidence_people else 0) / float(
            obsolete_evidence_count
        )
        row = [
            min(float(metrics["loss"]), loss_clip) / loss_clip,
            float(metrics["exact_match"]),
            float(metrics["token_accuracy"]),
            useful_signal,
            obsolete_signal,
            raw_reference_count / float(max_reference_count),
            capacity_pressure,
        ]
        rows.append(row)
        report[person] = dict(zip(FEATURE_NAMES, row, strict=True))
    return people, torch.tensor(rows, dtype=torch.float32, device=device), report


def teacher_logits_by_person(
    *,
    model: nn.Module,
    encoded_people: dict[str, list[EncodedExample]],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    return {
        person: collect_example_logits(model, examples, pad_id=pad_id, batch_size=batch_size, device=device)
        for person, examples in sorted(encoded_people.items())
    }


def per_person_losses(
    *,
    model: ActionLogitAdapterModel,
    people: list[str],
    encoded_people: dict[str, list[EncodedExample]],
    reference_logits: dict[str, list[torch.Tensor]],
    role_probs: torch.Tensor,
    learn_gate: torch.Tensor,
    pad_id: int,
    device: torch.device,
    distill_temperature: float,
    drop_target_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    protect_losses: list[torch.Tensor] = []
    drop_losses: list[torch.Tensor] = []
    if role_probs.shape != (len(people), len(ROLE_NAMES)):
        raise ValueError(f"role_probs shape must be {(len(people), len(ROLE_NAMES))}, got {tuple(role_probs.shape)}.")
    for person_index, person in enumerate(people):
        model.set_action_gates(
            {
                "learn": learn_gate,
                "preserve": role_probs[person_index, ROLE_TO_INDEX["preserve"]],
                "drop": role_probs[person_index, ROLE_TO_INDEX["drop"]],
                "guard": role_probs[person_index, ROLE_TO_INDEX["guard"]],
            }
        )
        examples = encoded_people[person]
        indices = torch.arange(len(examples), dtype=torch.long)
        inputs, targets, mask, selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
        logits = model(inputs)
        protect_losses.append(
            distillation_loss_for_examples(
                logits,
                selected,
                reference_logits[person],
                indices,
                temperature=distill_temperature,
                device=device,
            )
        )
        drop_losses.append(
            drop_suppression_loss(
                logits=logits,
                targets=targets,
                mask=mask,
                target_probability=drop_target_probability,
            )
        )
    return torch.stack(protect_losses), torch.stack(drop_losses)


def trainable_parameters(
    *,
    adapter: ActionLogitAdapterModel,
    controller: EndToEndCLController | None,
) -> list[nn.Parameter]:
    params = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if controller is not None:
        params.extend(parameter for parameter in controller.parameters() if parameter.requires_grad)
    if not params:
        raise RuntimeError("No trainable parameters were selected.")
    return params


def fixed_role_probs(
    *,
    method: str,
    true_roles: dict[str, str],
    people: list[str],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if method == "oracle":
        role_map = true_roles
    elif method == "random":
        role_map = random_roles_matching_counts(true_roles=true_roles, seed=seed)
    else:
        raise ValueError(f"Fixed role probabilities are not defined for method {method!r}.")
    rows: list[list[float]] = []
    for person in people:
        row = [0.0, 0.0, 0.0]
        row[ROLE_TO_INDEX[role_map[person]]] = 1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def role_regularization(
    *,
    role_probs: torch.Tensor,
    lambda_entropy: float,
) -> torch.Tensor:
    if lambda_entropy <= 0.0:
        return role_probs.new_zeros(())
    entropy = -(role_probs * role_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
    return lambda_entropy * entropy


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError(f"{name} values/weights shape mismatch: {tuple(values.shape)} vs {tuple(weights.shape)}.")
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return values.new_zeros(())
    return (values * weights).sum() / denom


def learn_only_gates(learn_gate: torch.Tensor | float) -> dict[str, torch.Tensor | float]:
    return {"learn": learn_gate, "preserve": 0.0, "drop": 0.0, "guard": 0.0}


def person_for_example(example: EncodedExample, people: list[str]) -> str | None:
    text = (example.prompt + example.answer).lower()
    matches = [person for person in people if person.lower() in text]
    if len(matches) > 1:
        raise RuntimeError(f"Example mentions multiple known people {matches}: {example.prompt!r}{example.answer!r}")
    return matches[0] if matches else None


@torch.no_grad()
def evaluate_with_action_gates(
    *,
    model: ActionLogitAdapterModel,
    examples: list[EncodedExample],
    people: list[str],
    role_probs: torch.Tensor,
    learn_gate: torch.Tensor,
    pad_id: int,
    device: torch.device,
) -> dict[str, float]:
    if role_probs.shape != (len(people), len(ROLE_NAMES)):
        raise ValueError(f"role_probs shape must be {(len(people), len(ROLE_NAMES))}, got {tuple(role_probs.shape)}.")
    totals = {
        "loss_sum": 0.0,
        "token_count": 0.0,
        "token_correct": 0.0,
        "example_count": 0.0,
        "exact_count": 0.0,
    }
    person_to_index = {person: index for index, person in enumerate(people)}
    for example_index, example in enumerate(examples):
        person = person_for_example(example, people)
        if person is None:
            model.set_action_gates(learn_only_gates(learn_gate))
        else:
            row_index = person_to_index[person]
            model.set_action_gates(
                {
                    "learn": learn_gate,
                    "preserve": role_probs[row_index, ROLE_TO_INDEX["preserve"]],
                    "drop": role_probs[row_index, ROLE_TO_INDEX["drop"]],
                    "guard": role_probs[row_index, ROLE_TO_INDEX["guard"]],
                }
            )
        indices = torch.tensor([example_index], dtype=torch.long)
        inputs, targets, mask, _selected = batch_examples(examples, indices=indices, pad_id=pad_id, device=device)
        logits = model(inputs)
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
        predictions = logits.argmax(dim=-1)
        token_correct = ((predictions == targets).to(torch.float32) * mask).sum()
        token_count = mask.sum()
        if float(token_count.detach().cpu()) <= 0.0:
            raise RuntimeError(f"Example has no answer-token labels: {example.prompt!r}{example.answer!r}")
        exact = 1.0 if float(token_correct.detach().cpu()) == float(token_count.detach().cpu()) else 0.0
        totals["loss_sum"] += float((losses * mask).sum().detach().cpu())
        totals["token_count"] += float(token_count.detach().cpu())
        totals["token_correct"] += float(token_correct.detach().cpu())
        totals["example_count"] += 1.0
        totals["exact_count"] += exact
    if totals["token_count"] <= 0.0 or totals["example_count"] <= 0.0:
        raise RuntimeError("Action-gated evaluation saw no examples/tokens.")
    return {
        "loss": totals["loss_sum"] / totals["token_count"],
        "token_accuracy": totals["token_correct"] / totals["token_count"],
        "exact_match": totals["exact_count"] / totals["example_count"],
        "example_count": totals["example_count"],
    }


def evaluate_groups_with_action_gates(
    *,
    model: ActionLogitAdapterModel,
    groups: dict[str, list[EncodedExample]],
    people: list[str],
    role_probs: torch.Tensor,
    learn_gate: torch.Tensor,
    pad_id: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    return {
        name: evaluate_with_action_gates(
            model=model,
            examples=examples,
            people=people,
            role_probs=role_probs,
            learn_gate=learn_gate,
            pad_id=pad_id,
            device=device,
        )
        for name, examples in sorted(groups.items())
    }


def train_method(
    *,
    args: argparse.Namespace,
    method: str,
    base_model: nn.Module,
    checkpoint: dict[str, Any],
    people: list[str],
    person_features: torch.Tensor,
    encoded_groups: dict[str, list[EncodedExample]],
    encoded_people: dict[str, list[EncodedExample]],
    reference_logits: dict[str, list[torch.Tensor]],
    true_roles: dict[str, str],
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    d_model = int(checkpoint["model_config"]["d_model"])
    adapter = ActionLogitAdapterModel(
        base_model=base_model,
        d_model=d_model,
        vocab_size=int(checkpoint["model_config"]["vocab_size"]),
        rank=args.adapter_rank,
        adapter_scale=args.adapter_scale,
        init_std=args.adapter_init_std,
    ).to(device)
    freeze_module(adapter.base_model)
    controller = (
        EndToEndCLController(input_dim=len(FEATURE_NAMES), hidden_dim=args.controller_hidden_dim).to(device)
        if method == "neural"
        else None
    )
    if method == "naive":
        fixed_probs = None
    elif method in {"oracle", "random"}:
        fixed_probs = fixed_role_probs(
            method=method,
            true_roles=true_roles,
            people=people,
            seed=args.seed + 5000,
            device=device,
        )
    elif method == "neural":
        fixed_probs = None
    else:
        raise ValueError(f"Unknown method {method!r}.")

    optimizer = torch.optim.AdamW(
        trainable_parameters(adapter=adapter, controller=controller),
        lr=args.controller_lr,
        weight_decay=args.controller_weight_decay,
    )
    train_examples = encoded_groups["stage2"] + encoded_groups["stage3"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 6000)
    trace: list[dict[str, float]] = []
    for epoch in range(1, args.controller_epochs + 1):
        adapter.train()
        if controller is not None:
            controller.train()
        permutation = torch.randperm(len(train_examples), generator=generator)
        totals = {
            "loss": 0.0,
            "new_ce": 0.0,
            "protect": 0.0,
            "guard": 0.0,
            "forget": 0.0,
            "adapter_norm": 0.0,
            "learn_gate": 0.0,
        }
        batches = 0
        pbar = tqdm(range(0, len(train_examples), args.batch_size), desc=f"{method} controller {epoch}/{args.controller_epochs}")
        for start in pbar:
            indices = permutation[start : start + args.batch_size]
            inputs, targets, mask, _selected = batch_examples(train_examples, indices=indices, pad_id=pad_id, device=device)
            optimizer.zero_grad(set_to_none=True)
            if controller is None:
                learn_gate = inputs.new_tensor(1.0, dtype=torch.float32)
                role_probs = fixed_probs
            else:
                learn_gate = controller.learn_gate(person_features)
                role_probs = controller.role_probs(person_features, temperature=args.role_temperature)
            adapter.set_action_gates(learn_only_gates(learn_gate))
            logits = adapter(inputs)
            new_ce = masked_ce_loss(logits, targets, mask)

            if method == "naive":
                protect_loss = new_ce.new_zeros(())
                guard_loss = new_ce.new_zeros(())
                forget_loss = new_ce.new_zeros(())
                gate_loss = new_ce.new_zeros(())
            else:
                if role_probs is None:
                    raise RuntimeError(f"{method} has no role probabilities.")
                protect_per_person, drop_per_person = per_person_losses(
                    model=adapter,
                    people=people,
                    encoded_people=encoded_people,
                    reference_logits=reference_logits,
                    role_probs=role_probs,
                    learn_gate=learn_gate,
                    pad_id=pad_id,
                    device=device,
                    distill_temperature=args.distill_temperature,
                    drop_target_probability=args.drop_target_probability,
                )
                useful_signal = person_features[:, FEATURE_NAMES.index("useful_signal")]
                obsolete_signal = person_features[:, FEATURE_NAMES.index("obsolete_signal")]
                guard_signal = ((1.0 - useful_signal).clamp_min(0.0) * (1.0 - obsolete_signal).clamp_min(0.0))
                protect_loss = weighted_mean(protect_per_person, useful_signal, name="protect")
                guard_loss = weighted_mean(protect_per_person, guard_signal, name="guard")
                forget_loss = weighted_mean(drop_per_person, obsolete_signal, name="forget")
                gate_loss = role_regularization(role_probs=role_probs, lambda_entropy=args.lambda_gate_entropy)

            norm_loss = adapter.adapter_norm()
            loss = (
                new_ce
                + args.lambda_protect * protect_loss
                + args.lambda_guard * guard_loss
                + args.lambda_forget * forget_loss
                + args.lambda_adapter_norm * norm_loss
                + gate_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(adapter=adapter, controller=controller), args.grad_clip)
            optimizer.step()

            row = {
                "loss": float(loss.detach().cpu()),
                "new_ce": float(new_ce.detach().cpu()),
                "protect": float(protect_loss.detach().cpu()),
                "guard": float(guard_loss.detach().cpu()),
                "forget": float(forget_loss.detach().cpu()),
                "adapter_norm": float(norm_loss.detach().cpu()),
                "learn_gate": float(learn_gate.detach().cpu()),
            }
            for key, value in row.items():
                totals[key] += value
            batches += 1
            pbar.set_postfix({"loss": f"{row['loss']:.3g}", "new": f"{row['new_ce']:.3g}", "g": f"{row['learn_gate']:.2f}"})
        if batches <= 0:
            raise RuntimeError(f"{method} controller training saw zero batches.")
        epoch_row = {key: value / float(batches) for key, value in totals.items()}
        epoch_row["epoch"] = float(epoch)
        trace.append(epoch_row)
        if epoch == 1 or epoch == args.controller_epochs or epoch % args.print_every == 0:
            print(
                f"{method} epoch={epoch:4d} loss={epoch_row['loss']:.5f} "
                f"new={epoch_row['new_ce']:.5f} protect={epoch_row['protect']:.5f} "
                f"guard={epoch_row['guard']:.5f} forget={epoch_row['forget']:.5f} "
                f"learn_gate={epoch_row['learn_gate']:.3f}"
            )

    adapter.eval()
    if controller is not None:
        controller.eval()
        with torch.no_grad():
            final_role_probs = controller.role_probs(person_features, temperature=args.role_temperature)
            final_learn_gate = controller.learn_gate(person_features)
    elif fixed_probs is not None:
        final_role_probs = fixed_probs
        final_learn_gate = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        final_role_probs = torch.zeros((len(people), len(ROLE_NAMES)), dtype=torch.float32, device=device)
        final_learn_gate = torch.tensor(1.0, dtype=torch.float32, device=device)
    metrics = evaluate_groups_with_action_gates(
        model=adapter,
        groups=encoded_groups,
        people=people,
        role_probs=final_role_probs,
        learn_gate=final_learn_gate,
        pad_id=pad_id,
        device=device,
    )
    category_breakdown = {"note": "category breakdown is disabled for action-gated per-example evaluation"}
    predicted_roles = {
        person: ROLE_NAMES[int(final_role_probs[index].argmax().item())]
        for index, person in enumerate(people)
    }
    if method == "naive":
        predicted_roles = {person: "none" for person in people}
        role_report = {"accuracy": None, "correct": None, "total": len(people), "confusion": None, "per_person": {}}
    else:
        role_report = role_match_report(predicted_roles=predicted_roles, true_roles=true_roles)
    gate_report = {
        person: {
            "true_role": true_roles[person],
            "predicted_role": predicted_roles[person],
            "preserve": float(final_role_probs[index, ROLE_TO_INDEX["preserve"]].detach().cpu()),
            "drop": float(final_role_probs[index, ROLE_TO_INDEX["drop"]].detach().cpu()),
            "guard": float(final_role_probs[index, ROLE_TO_INDEX["guard"]].detach().cpu()),
        }
        for index, person in enumerate(people)
    }
    return {
        "method": method,
        "trace": trace,
        "metrics": metrics,
        "category_breakdown": category_breakdown,
        "role_report": role_report,
        "gate_report": gate_report,
        "learn_gate": float(final_learn_gate.detach().cpu()),
        "adapter_norm": float(adapter.adapter_norm().detach().cpu()),
    }


def compact(metrics: dict[str, float]) -> str:
    return f"{metrics['loss']:.3g}/{metrics['exact_match']:.3f}"


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError(f"Tokenizer {args.tokenizer_path} has no [PAD] token.")

    _loaded_model, checkpoint = load_checkpoint(args.config_checkpoint, device)
    max_seq_len = int(checkpoint["model_config"]["max_seq_len"])
    preserve_people = parse_people(args.oracle_preserve_people, name="oracle_preserve_people")
    drop_people = parse_people(args.oracle_drop_people, name="oracle_drop_people")
    true_roles = oracle_roles(preserve_people=preserve_people, drop_people=drop_people)
    useful_evidence_people = parse_people(args.useful_evidence_people, name="useful_evidence_people")
    obsolete_evidence_people = parse_people(args.obsolete_evidence_people, name="obsolete_evidence_people")

    raw_groups = build_raw_stream(
        useful_evidence_people=useful_evidence_people,
        composition_holdout_people={item.strip() for item in args.composition_holdout_people.split(",") if item.strip()},
        include_composition_rules=args.include_composition_rules,
    )
    encoded_base_groups = encode_raw_groups(raw_groups, tokenizer, max_seq_len=max_seq_len)
    encoded_people = make_encoded_people(tokenizer=tokenizer, max_seq_len=max_seq_len)
    encoded_groups = {
        **encoded_base_groups,
        "preserve": [example for person, examples in encoded_people.items() if true_roles[person] == "preserve" for example in examples],
        "drop": [example for person, examples in encoded_people.items() if true_roles[person] == "drop" for example in examples],
        "neutral": [example for person, examples in encoded_people.items() if true_roles[person] == "guard" for example in examples],
    }
    for name in ["stage1", "stage2", "stage3", "preserve", "drop", "neutral", "eval_all"]:
        if name not in encoded_groups or not encoded_groups[name]:
            raise RuntimeError(f"Encoded group {name!r} is empty.")

    base_model = make_model_from_config(checkpoint=checkpoint, device=device, seed=args.seed)
    print("TINY END-TO-END NEURAL CL CONTROLLER")
    print("=" * 120)
    print(f"device={device} methods={parse_methods(args.methods)}")
    print(
        f"stage1={len(encoded_groups['stage1'])} stage2={len(encoded_groups['stage2'])} "
        f"stage3={len(encoded_groups['stage3'])} eval={len(encoded_groups['eval_all'])}"
    )
    train_bootstrap_stage(
        args=args,
        model=base_model,
        stage_examples=encoded_groups["stage1"],
        pad_id=pad_id,
        device=device,
        epochs=args.stage1_epochs,
        seed=args.seed + 1000,
        label="base stage1",
    )
    freeze_module(base_model)
    reference_logits = teacher_logits_by_person(
        model=base_model,
        encoded_people=encoded_people,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
    )
    people, person_features, feature_report = feature_tensor_for_people(
        model=base_model,
        encoded_people=encoded_people,
        raw_groups=raw_groups,
        useful_evidence_people=useful_evidence_people,
        obsolete_evidence_people=obsolete_evidence_people,
        pad_id=pad_id,
        batch_size=args.eval_batch_size,
        device=device,
        loss_clip=args.loss_clip,
        obsolete_evidence_count=args.obsolete_evidence_count,
        capacity_pressure=args.capacity_pressure,
    )
    methods = parse_methods(args.methods)
    results: dict[str, Any] = {}
    for method in methods:
        results[method] = train_method(
            args=args,
            method=method,
            base_model=base_model,
            checkpoint=checkpoint,
            people=people,
            person_features=person_features,
            encoded_groups=encoded_groups,
            encoded_people=encoded_people,
            reference_logits=reference_logits,
            true_roles=true_roles,
            pad_id=pad_id,
            device=device,
        )

    print("\nTINY END-TO-END NEURAL CL CONTROLLER SUMMARY")
    print("=" * 120)
    print(
        f"{'method':>10} {'roleAcc':>8} {'preserve':>14} {'drop':>14} {'guard':>14} "
        f"{'stage2':>14} {'stage3':>14} {'eval_all':>14}"
    )
    print(f"{'':>10} {'':>8} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14} {'loss/exact':>14}")
    for method in methods:
        row = results[method]
        metrics = row["metrics"]
        role_acc = row["role_report"]["accuracy"]
        role_text = "NA" if role_acc is None else f"{role_acc:.3f}"
        print(
            f"{method:>10} {role_text:>8} {compact(metrics['preserve']):>14} "
            f"{compact(metrics['drop']):>14} {compact(metrics['neutral']):>14} "
            f"{compact(metrics['stage2']):>14} {compact(metrics['stage3']):>14} "
            f"{compact(metrics['eval_all']):>14}"
        )

    print("\nNEURAL GATE REPORT")
    print("-" * 120)
    for method in methods:
        if method == "naive":
            continue
        print(f"method={method}")
        for person, row in results[method]["gate_report"].items():
            print(
                f"  {person:>6} true={row['true_role']:<8} pred={row['predicted_role']:<8} "
                f"p={row['preserve']:.3f} d={row['drop']:.3f} g={row['guard']:.3f}"
            )

    output = {
        "question": "Can a neural controller learn soft CL actions from outcome loss rather than explicit role labels?",
        "config": {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "feature_names": list(FEATURE_NAMES),
            "role_names": list(ROLE_NAMES),
            "true_roles": true_roles,
        },
        "feature_report": feature_report,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-checkpoint", type=Path, default=Path("model/checkpoints/gco-tiny-cl-base-100w-seed0.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json"))
    parser.add_argument("--output-json", type=Path, default=Path("model/analysis/gco-tiny-end-to-end-cl-controller-seed0.json"))
    parser.add_argument("--methods", type=str, default="naive,oracle,neural,random")
    parser.add_argument("--oracle-preserve-people", type=str, default="Alice,Bruno")
    parser.add_argument("--oracle-drop-people", type=str, default="Clara,Darin")
    parser.add_argument("--useful-evidence-people", type=str, default="Alice,Bruno")
    parser.add_argument("--obsolete-evidence-people", type=str, default="Clara,Darin")
    parser.add_argument("--composition-holdout-people", type=str, default="Kira,Luca")
    parser.add_argument("--include-composition-rules", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=300)
    parser.add_argument("--controller-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--controller-lr", type=float, default=1e-3)
    parser.add_argument("--controller-weight-decay", type=float, default=1e-4)
    parser.add_argument("--controller-hidden-dim", type=int, default=48)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--adapter-scale", type=float, default=4.0)
    parser.add_argument("--adapter-init-std", type=float, default=0.02)
    parser.add_argument("--role-temperature", type=float, default=1.0)
    parser.add_argument("--lambda-protect", type=float, default=1.0)
    parser.add_argument("--lambda-forget", type=float, default=0.4)
    parser.add_argument("--lambda-guard", type=float, default=1.0)
    parser.add_argument("--lambda-adapter-norm", type=float, default=1e-4)
    parser.add_argument("--lambda-gate-entropy", type=float, default=0.0)
    parser.add_argument("--drop-target-probability", type=float, default=0.05)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--loss-clip", type=float, default=8.0)
    parser.add_argument("--capacity-pressure", type=float, default=1.0)
    parser.add_argument("--obsolete-evidence-count", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
