#!/usr/bin/env python3
"""LoRA semantic-closure continual update on local pretrained Pythia.

This is the follow-up to ``gco_pythia_semantic_closure_update.py``.  The first
version used a final-hidden adapter and showed a clean failure: the foundation
world was not learned well enough and the constrained update had almost no
usable tangent room.

This version changes the editable subspace:

* freeze the pretrained Pythia base;
* install LoRA modules in the final transformer layers;
* train the LoRA state on a small foundation semantic world;
* stop before correction if the foundation does not pass a measured gate;
* protect only foundation measurements that the current model already answers
  correctly;
* compare raw direct, raw semantic closure, constrained direct, and
  constrained semantic closure updates;
* use explicit backtracking so unstable updates are reported, not hidden.

The point is not to prove open-world reasoning.  The point is to test whether a
pretrained semantic base plus a better editable subspace can learn a correction
packet that includes the direct edit and its reasoning consequences.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.gco_math.gco_pythia_semantic_closure_update import (
    QAItem,
    adapted_hidden_matrix,
    build_entities,
    clip_norm,
    correction_margin,
    direct_only_items,
    encode_items,
    evaluate_items,
    flat_gradient,
    foundation_items,
    linear_cka,
    old_suppression_loss,
    positive_float,
    positive_int,
    project_gradient,
    protected_items,
    rows_to_dict,
    seed_everything,
    semantic_closure_items,
)
from experiments.real_book_common import resolve_device


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        positive_int("rank", rank)
        positive_float("alpha", alpha)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base).__name__}.")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_a.weight, std=1.0 / math.sqrt(float(base.in_features)))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(x))


class PythiaLoRALM(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        if not hasattr(base_model, "gpt_neox"):
            raise TypeError("Expected a GPT-NeoX/Pythia model with .gpt_neox.")
        self.base_model = base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Pythia forward did not return hidden states.")
        return outputs.logits, outputs.hidden_states[-1]


def parse_target_modules(value: str) -> tuple[str, ...]:
    modules = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modules:
        raise ValueError("--target-modules must contain at least one module name.")
    return modules


def install_lora(
    base_model: nn.Module,
    *,
    last_layers: int,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
) -> list[str]:
    if not hasattr(base_model, "gpt_neox") or not hasattr(base_model.gpt_neox, "layers"):
        raise TypeError("Expected base_model.gpt_neox.layers for Pythia LoRA installation.")
    layers = base_model.gpt_neox.layers
    layer_count = len(layers)
    if last_layers <= 0 or last_layers > layer_count:
        raise ValueError(f"last_layers must be in [1, {layer_count}], got {last_layers}.")
    installed: list[str] = []
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    start = layer_count - last_layers
    for layer_index in range(start, layer_count):
        layer = layers[layer_index]
        for module_path in target_modules:
            parent = layer
            pieces = module_path.split(".")
            for piece in pieces[:-1]:
                if not hasattr(parent, piece):
                    raise AttributeError(f"Layer {layer_index} has no module path {module_path!r}.")
                parent = getattr(parent, piece)
            leaf = pieces[-1]
            if not hasattr(parent, leaf):
                raise AttributeError(f"Layer {layer_index} has no module path {module_path!r}.")
            original = getattr(parent, leaf)
            if not isinstance(original, nn.Linear):
                raise TypeError(
                    f"Layer {layer_index}.{module_path} is {type(original).__name__}, expected nn.Linear."
                )
            setattr(parent, leaf, LoRALinear(original, rank=rank, alpha=alpha))
            installed.append(f"gpt_neox.layers.{layer_index}.{module_path}")
    if not installed:
        raise RuntimeError("No LoRA modules were installed.")
    return installed


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (".lora_a." in name or ".lora_b." in name)
    ]
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters found.")
    return parameters


def snapshot_parameters(parameters: Sequence[nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def restore_parameters(parameters: Sequence[nn.Parameter], snapshot: Sequence[torch.Tensor]) -> None:
    if len(parameters) != len(snapshot):
        raise RuntimeError("Parameter snapshot length mismatch.")
    with torch.no_grad():
        for parameter, saved in zip(parameters, snapshot):
            parameter.copy_(saved)


def apply_flat_delta(parameters: Sequence[nn.Parameter], flat_delta: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(flat_delta[offset : offset + count].reshape_as(parameter))
            offset += count
    if offset != flat_delta.numel():
        raise RuntimeError(f"Flat delta size mismatch: consumed {offset}, total {flat_delta.numel()}.")


def flat_parameter_vector(parameters: Sequence[nn.Parameter]) -> torch.Tensor:
    if not parameters:
        raise RuntimeError("Cannot flatten an empty parameter list.")
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def grouped_batches(
    tokenizer,
    items: Sequence[QAItem],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> list[tuple[str, object]]:
    by_group: dict[str, list[QAItem]] = {}
    for item in items:
        by_group.setdefault(item.group, []).append(item)
    batches = []
    for group, group_items in sorted(by_group.items()):
        batches.append(
            (
                group,
                encode_items(
                    tokenizer,
                    group_items,
                    pad_token_id=args.pad_token_id,
                    max_seq_len=args.max_seq_len,
                ).to(device),
            )
        )
    return batches


def group_weight(args: argparse.Namespace, group: str) -> float:
    weights = {
        "direct": args.weight_direct,
        "paraphrase": args.weight_paraphrase,
        "ripple": args.weight_ripple,
        "history": args.weight_history,
        "locality": args.weight_locality,
        "rule": args.weight_rule,
    }
    return weights.get(group, 1.0)


def plot_lora_summary(report: dict, output_path: Path) -> None:
    modes = [mode for mode in report["modes"] if mode != "foundation_after_training"]
    metric_names = ("direct", "paraphrase", "ripple", "history", "locality", "rule")
    if not modes:
        raise RuntimeError("No update modes available for plotting.")
    values = []
    for mode in modes:
        evals = report["modes"][mode]["eval"]
        values.append([evals.get(metric, {}).get("exact", 0.0) for metric in metric_names])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / max(1, len(modes))
    x_positions = list(range(len(metric_names)))
    for index, mode in enumerate(modes):
        offset = (index - (len(modes) - 1) / 2.0) * width
        ax.bar([x + offset for x in x_positions], values[index], width=width, label=mode)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("exact match")
    ax.set_title("Pythia LoRA semantic-closure correction")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def weighted_packet_loss(
    model: PythiaLoRALM,
    group_batches: Sequence[tuple[str, object]],
    args: argparse.Namespace,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for group, batch in group_batches:
        logits, _hidden = model(batch.input_ids, batch.attention_mask)
        loss = masked_cross_entropy(logits, batch)
        weight = group_weight(args, group)
        if weight <= 0.0 or not math.isfinite(weight):
            raise ValueError(f"Invalid group weight for {group!r}: {weight}")
        losses.append(weight * loss)
        weights.append(weight)
    if not losses:
        raise RuntimeError("No packet losses were produced.")
    return torch.stack(losses).sum() / sum(weights)


def masked_cross_entropy(logits: torch.Tensor, batch) -> torch.Tensor:
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.targets.reshape(-1),
        reduction="none",
    ).reshape_as(batch.targets)
    denom = batch.answer_mask.sum()
    if float(denom.detach().cpu()) <= 0.0:
        raise ValueError("Answer mask is empty.")
    return (losses * batch.answer_mask).sum() / denom


@torch.no_grad()
def correct_items(
    model: PythiaLoRALM,
    tokenizer,
    items: Sequence[QAItem],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[QAItem, ...]:
    _rows, predictions = evaluate_items(model, tokenizer, items, args, device=device)
    correct_keys = {
        (str(prediction["question"]), str(prediction["answer"]))
        for prediction in predictions
        if float(prediction["correct"]) == 1.0
    }
    return tuple(item for item in items if (item.question, item.answer) in correct_keys)


def constraint_rows_for_items(
    model: PythiaLoRALM,
    tokenizer,
    items: Sequence[QAItem],
    raw_gradient: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float | str]]]:
    scored_rows: list[tuple[float, torch.Tensor, dict[str, float | str]]] = []
    for index, item in enumerate(items):
        batch = encode_items(
            tokenizer,
            [item],
            pad_token_id=args.pad_token_id,
            max_seq_len=args.max_seq_len,
        ).to(device)
        logits, _hidden = model(batch.input_ids, batch.attention_mask)
        loss = masked_cross_entropy(logits, batch)
        row = flat_gradient(loss, parameters, retain_graph=False, label=f"constraint_{index}")
        row_norm = torch.linalg.vector_norm(row)
        row_norm_value = float(row_norm.detach().cpu())
        if row_norm_value <= 1e-12 or not math.isfinite(row_norm_value):
            raise FloatingPointError(f"Constraint row {index} has invalid norm {row_norm_value}.")
        unit = row / row_norm
        damage = abs(float(torch.dot(unit.detach(), raw_gradient.detach()).cpu()))
        scored_rows.append(
            (
                damage,
                unit.detach(),
                {
                    "question": item.question,
                    "answer": item.answer,
                    "group": item.group,
                    "predicted_damage": damage,
                    "row_norm": row_norm_value,
                },
            )
        )
    scored_rows.sort(key=lambda row: row[0], reverse=True)
    selected = scored_rows[: args.constraint_limit]
    if not selected:
        raise RuntimeError("No constraint rows were selected.")
    return torch.stack([row for _damage, row, _meta in selected], dim=0), [
        meta for _damage, _row, meta in selected
    ]


def try_backtracked_step(
    model: PythiaLoRALM,
    parameters: Sequence[nn.Parameter],
    flat_gradient_step: torch.Tensor,
    group_batches_for_loss: Sequence[tuple[str, object]],
    args: argparse.Namespace,
    *,
    step_lr: float,
    max_allowed_loss: float,
) -> tuple[bool, float, float]:
    if step_lr <= 0.0 or not math.isfinite(step_lr):
        raise ValueError(f"step_lr must be positive and finite, got {step_lr}.")
    if max_allowed_loss <= 0.0 or not math.isfinite(max_allowed_loss):
        raise ValueError(f"max_allowed_loss must be positive and finite, got {max_allowed_loss}.")
    before = snapshot_parameters(parameters)
    for backtrack_index in range(args.backtrack_steps + 1):
        factor = args.backtrack_decay ** backtrack_index
        restore_parameters(parameters, before)
        apply_flat_delta(parameters, -step_lr * factor * flat_gradient_step)
        with torch.no_grad():
            loss = weighted_packet_loss(model, group_batches_for_loss, args)
        loss_value = float(loss.detach().cpu())
        if math.isfinite(loss_value) and loss_value <= max_allowed_loss:
            return True, factor, loss_value
    restore_parameters(parameters, before)
    return False, 0.0, float("nan")


def train_foundation_lora(
    model: PythiaLoRALM,
    tokenizer,
    items: Sequence[QAItem],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> list[dict[str, float]]:
    batches = grouped_batches(tokenizer, items, args, device=device)
    parameters = lora_parameters(model)
    trace: list[dict[str, float]] = []
    accepted_steps = 0
    iterator: Iterable[int] = range(1, args.foundation_epochs + 1)
    if args.progress:
        iterator = tqdm(iterator, desc="foundation_lora", leave=False)
    for epoch in iterator:
        loss = weighted_packet_loss(model, batches, args)
        loss_value = float(loss.detach().cpu())
        if not torch.isfinite(loss) or not math.isfinite(loss_value):
            raise FloatingPointError(f"Foundation loss became non-finite before step at epoch {epoch}.")
        gradient = flat_gradient(loss, parameters, retain_graph=False, label=f"foundation_{epoch}")
        if args.weight_decay > 0.0:
            gradient = gradient + args.weight_decay * flat_parameter_vector(parameters).to(
                device=gradient.device,
                dtype=gradient.dtype,
            )
        gradient, clip_scale = clip_norm(gradient, args.max_gradient_norm)
        max_allowed_loss = min(args.loss_ceiling, loss_value * args.foundation_loss_growth + 1e-6)
        accepted, backtrack_factor, post_loss = try_backtracked_step(
            model,
            parameters,
            gradient,
            batches,
            args,
            step_lr=args.foundation_lr,
            max_allowed_loss=max_allowed_loss,
        )
        if not accepted:
            raise RuntimeError(
                "Foundation backtracking could not find a finite bounded step "
                f"at epoch {epoch}: loss={loss_value:.6g}, max_allowed_loss={max_allowed_loss:.6g}."
            )
        accepted_steps += 1
        if epoch == 1 or epoch == args.foundation_epochs or epoch % args.print_every == 0:
            trace.append(
                {
                    "epoch": float(epoch),
                    "loss": loss_value,
                    "post_step_loss": post_loss,
                    "clip_scale": clip_scale,
                    "backtrack_factor": backtrack_factor,
                    "accepted_steps": float(accepted_steps),
                }
            )
    return trace


def train_update_lora(
    model: PythiaLoRALM,
    tokenizer,
    edit,
    update_items: Sequence[QAItem],
    protected: Sequence[QAItem],
    args: argparse.Namespace,
    *,
    device: torch.device,
    constrained: bool,
) -> tuple[list[dict[str, float]], list[dict[str, float | str]]]:
    update_batches = grouped_batches(tokenizer, update_items, args, device=device)
    protected_batches = grouped_batches(tokenizer, protected, args, device=device)
    parameters = lora_parameters(model)
    trace: list[dict[str, float]] = []
    selected_metadata: list[dict[str, float | str]] = []
    cached_rows: torch.Tensor | None = None
    accepted_steps = 0
    rejected_steps = 0
    iterator: Iterable[int] = range(1, args.update_epochs + 1)
    if args.progress:
        iterator = tqdm(iterator, desc="lora_update", leave=False)
    for epoch in iterator:
        packet_loss = weighted_packet_loss(model, update_batches, args)
        suppress = old_suppression_loss(
            model,
            tokenizer,
            edit,
            pad_token_id=args.pad_token_id,
            max_seq_len=args.max_seq_len,
            margin=args.old_margin,
            device=device,
        )
        raw_loss = packet_loss + args.old_suppression_weight * suppress
        raw_gradient = flat_gradient(raw_loss, parameters, retain_graph=False, label=f"raw_update_{epoch}")

        restore_loss_value = float("nan")
        if constrained:
            if cached_rows is None or epoch == 1 or (epoch - 1) % args.constraint_refresh == 0:
                cached_rows, selected_metadata = constraint_rows_for_items(
                    model,
                    tokenizer,
                    protected,
                    raw_gradient,
                    parameters,
                    args,
                    device=device,
                )
            tangent, projection = project_gradient(
                raw_gradient,
                cached_rows,
                damping=args.projection_damping,
            )
            restore_loss = weighted_packet_loss(model, protected_batches, args)
            restore_loss_value = float(restore_loss.detach().cpu())
            restore_gradient = flat_gradient(restore_loss, parameters, retain_graph=False, label=f"restore_{epoch}")
            restore_norm = torch.linalg.vector_norm(restore_gradient)
            tangent_norm = torch.linalg.vector_norm(tangent)
            if float(restore_norm.detach().cpu()) > 0.0:
                limit = args.restore_norm_ratio * tangent_norm
                restore_scale = torch.minimum(
                    torch.ones((), device=device, dtype=restore_gradient.dtype),
                    limit / (args.restore_strength * restore_norm + torch.finfo(restore_gradient.dtype).eps),
                )
                restore_gradient = restore_gradient * restore_scale
            final_gradient = tangent + args.restore_strength * restore_gradient
        else:
            projection = {"safe_fraction": 1.0, "removed_fraction": 0.0}
            final_gradient = raw_gradient

        final_gradient, clip_scale = clip_norm(final_gradient, args.max_gradient_norm)
        accepted, backtrack_factor, post_loss = try_backtracked_step(
            model,
            parameters,
            final_gradient,
            update_batches,
            args,
            step_lr=args.update_lr,
            max_allowed_loss=args.loss_ceiling,
        )
        if accepted:
            accepted_steps += 1
        else:
            rejected_steps += 1

        if epoch == 1 or epoch == args.update_epochs or epoch % args.print_every == 0:
            trace.append(
                {
                    "epoch": float(epoch),
                    "packet_loss": float(packet_loss.detach().cpu()),
                    "suppression_loss": float(suppress.detach().cpu()),
                    "restore_loss": restore_loss_value,
                    "safe_fraction": projection["safe_fraction"],
                    "removed_fraction": projection["removed_fraction"],
                    "clip_scale": clip_scale,
                    "accepted": float(accepted),
                    "accepted_steps": float(accepted_steps),
                    "rejected_steps": float(rejected_steps),
                    "backtrack_factor": backtrack_factor,
                    "post_step_packet_loss": post_loss,
                }
            )
    return trace, selected_metadata


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")
    for name in (
        "seed",
        "lora_rank",
        "lora_last_layers",
        "foundation_epochs",
        "update_epochs",
        "constraint_limit",
        "constraint_refresh",
        "max_seq_len",
        "backtrack_steps",
    ):
        value = int(getattr(args, name))
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    for name in (
        "lora_alpha",
        "foundation_lr",
        "foundation_loss_growth",
        "update_lr",
        "projection_damping",
        "restore_strength",
        "restore_norm_ratio",
        "old_suppression_weight",
        "old_margin",
        "max_gradient_norm",
        "foundation_gate",
        "weight_direct",
        "weight_paraphrase",
        "weight_ripple",
        "weight_history",
        "weight_locality",
        "weight_rule",
        "loss_ceiling",
        "backtrack_decay",
    ):
        positive_float(name, float(getattr(args, name)))
    if not 0.0 < args.backtrack_decay < 1.0:
        raise ValueError("--backtrack-decay must lie in (0, 1).")
    if args.foundation_loss_growth < 1.0:
        raise ValueError("--foundation-loss-growth must be at least 1.0.")
    if args.weight_decay < 0.0 or not math.isfinite(args.weight_decay):
        raise ValueError("--weight-decay must be non-negative and finite.")
    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive.")
    if args.lora_last_layers <= 0:
        raise ValueError("--lora-last-layers must be positive.")
    if args.foundation_epochs <= 0 or args.update_epochs <= 0:
        raise ValueError("--foundation-epochs and --update-epochs must be positive.")
    if args.constraint_limit <= 0:
        raise ValueError("--constraint-limit must be positive.")
    if args.constraint_refresh <= 0:
        raise ValueError("--constraint-refresh must be positive.")
    if not 0.0 < args.foundation_gate <= 1.0:
        raise ValueError("--foundation-gate must lie in (0, 1].")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_modules = parse_target_modules(args.target_modules)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if args.pad_token_id >= int(getattr(tokenizer, "vocab_size", 0)):
        raise ValueError(
            f"--pad-token-id={args.pad_token_id} is outside tokenizer vocab size {tokenizer.vocab_size}."
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
    )
    installed = install_lora(
        base_model,
        last_layers=args.lora_last_layers,
        target_modules=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    model = PythiaLoRALM(base_model).to(device)
    model.eval()

    entities = build_entities()
    edit = entities[0]
    foundation = foundation_items(entities)
    closure = semantic_closure_items(edit, entities)
    direct = direct_only_items(edit)
    candidate_protected = protected_items(edit, entities)
    eval_items = closure

    foundation_trace = train_foundation_lora(model, tokenizer, foundation, args, device=device)
    foundation_eval, foundation_predictions = evaluate_items(model, tokenizer, foundation, args, device=device)
    foundation_exact = sum(row.exact * row.total for row in foundation_eval) / sum(row.total for row in foundation_eval)
    modes: dict[str, dict] = {
        "foundation_after_training": {
            "eval": rows_to_dict(foundation_eval),
            "foundation_exact": foundation_exact,
            "predictions": foundation_predictions,
            "margin": correction_margin(model, tokenizer, edit, args, device=device),
        }
    }
    if foundation_exact < args.foundation_gate:
        report = {
            "experiment": "pythia_lora_semantic_closure_update",
            "status": "foundation_gate_failed",
            "foundation_exact": foundation_exact,
            "foundation_gate": args.foundation_gate,
            "model_dir": str(args.model_dir),
            "device": args.device,
            "seed": args.seed,
            "installed_lora_modules": installed,
            "foundation_trace": foundation_trace,
            "modes": modes,
        }
        output_json = args.output_dir / "pythia_lora_semantic_closure_update.json"
        with output_json.open("w") as handle:
            json.dump(report, handle, indent=2)
        print("\nPYTHIA LORA SEMANTIC-CLOSURE UPDATE")
        print("=" * 132)
        print(
            f"foundation_gate_failed exact={foundation_exact:.4f} gate={args.foundation_gate:.4f} "
            f"wrote_json={output_json}"
        )
        return

    protected = correct_items(model, tokenizer, candidate_protected, args, device=device)
    if not protected:
        raise RuntimeError("Foundation passed, but no candidate protected items were answered correctly.")
    reference_hidden = adapted_hidden_matrix(model, tokenizer, protected, args, device=device)
    foundation_snapshot = snapshot_parameters(lora_parameters(model))

    update_modes = (
        ("direct_raw_packet", direct, False),
        ("semantic_closure_raw_packet", closure, False),
        ("direct_constrained_packet", direct, True),
        ("semantic_closure_constrained_packet", closure, True),
    )
    parameters = lora_parameters(model)
    for mode_name, packet_items, constrained in update_modes:
        restore_parameters(parameters, foundation_snapshot)
        mode_trace, mode_constraints = train_update_lora(
            model,
            tokenizer,
            edit,
            packet_items,
            protected,
            args,
            device=device,
            constrained=constrained,
        )
        mode_eval, mode_predictions = evaluate_items(model, tokenizer, eval_items, args, device=device)
        mode_hidden = adapted_hidden_matrix(model, tokenizer, protected, args, device=device)
        modes[mode_name] = {
            "constrained": constrained,
            "eval": rows_to_dict(mode_eval),
            "margin": correction_margin(model, tokenizer, edit, args, device=device),
            "protected_cka": linear_cka(reference_hidden, mode_hidden),
            "trace": mode_trace,
            "selected_constraints": mode_constraints,
            "predictions": mode_predictions,
        }

    report = {
        "experiment": "pythia_lora_semantic_closure_update",
        "status": "completed",
        "model_dir": str(args.model_dir),
        "device": args.device,
        "seed": args.seed,
        "installed_lora_modules": installed,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "foundation_gate": args.foundation_gate,
        "foundation_exact": foundation_exact,
        "protected_count": len(protected),
        "foundation_trace": foundation_trace,
        "edit": asdict(edit),
        "foundation_items": [asdict(item) for item in foundation],
        "closure_items": [asdict(item) for item in closure],
        "protected_items": [asdict(item) for item in protected],
        "modes": modes,
    }
    output_json = args.output_dir / "pythia_lora_semantic_closure_update.json"
    output_plot = args.output_dir / "pythia_lora_semantic_closure_update.png"
    with output_json.open("w") as handle:
        json.dump(report, handle, indent=2)
    plot_lora_summary(report, output_plot)

    print("\nPYTHIA LORA SEMANTIC-CLOSURE UPDATE")
    print("=" * 148)
    print(
        f"device={args.device} model={args.model_dir} lora_rank={args.lora_rank} "
        f"layers={args.lora_last_layers} foundation={args.foundation_epochs} update={args.update_epochs} "
        f"foundation_exact={foundation_exact:.4f} protected={len(protected)}"
    )
    foundation_group_text = ", ".join(
        f"{group}:{metrics['exact']:.3f}"
        for group, metrics in sorted(modes["foundation_after_training"]["eval"].items())
    )
    print(f"foundation_groups={foundation_group_text}")
    print("-" * 148)
    print(
        f"{'mode':>38} {'direct':>8} {'para':>8} {'ripple':>8} {'history':>8} "
        f"{'local':>8} {'rule':>8} {'margin':>9} {'cka':>8}"
    )
    for mode_name, mode in modes.items():
        if mode_name == "foundation_after_training":
            continue
        evals = mode["eval"]
        cka = mode.get("protected_cka", float("nan"))
        print(
            f"{mode_name:>38} "
            f"{evals.get('direct', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('paraphrase', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('ripple', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('history', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('locality', {}).get('exact', 0.0):8.4f} "
            f"{evals.get('rule', {}).get('exact', 0.0):8.4f} "
            f"{mode['margin']:9.4f} "
            f"{cka:8.4f}"
        )
    print(f"wrote_json={output_json}")
    print(f"wrote_plot={output_plot}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("model/checkpoints/pythia-70m"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis/gco-pythia-lora-semantic-closure-seed0"))
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-last-layers", type=int, default=2)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="attention.query_key_value,attention.dense,mlp.dense_h_to_4h,mlp.dense_4h_to_h",
    )
    parser.add_argument("--foundation-epochs", type=int, default=800)
    parser.add_argument("--update-epochs", type=int, default=160)
    parser.add_argument("--foundation-lr", type=float, default=8e-4)
    parser.add_argument("--foundation-loss-growth", type=float, default=1.05)
    parser.add_argument("--update-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--projection-damping", type=float, default=1e-2)
    parser.add_argument("--restore-strength", type=float, default=0.15)
    parser.add_argument("--restore-norm-ratio", type=float, default=0.25)
    parser.add_argument("--old-suppression-weight", type=float, default=1.0)
    parser.add_argument("--old-margin", type=float, default=0.5)
    parser.add_argument("--constraint-limit", type=int, default=8)
    parser.add_argument("--constraint-refresh", type=int, default=20)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--loss-ceiling", type=float, default=25.0)
    parser.add_argument("--backtrack-steps", type=int, default=6)
    parser.add_argument("--backtrack-decay", type=float, default=0.5)
    parser.add_argument("--foundation-gate", type=float, default=0.85)
    parser.add_argument("--weight-direct", type=float, default=5.0)
    parser.add_argument("--weight-paraphrase", type=float, default=3.0)
    parser.add_argument("--weight-ripple", type=float, default=5.0)
    parser.add_argument("--weight-history", type=float, default=1.5)
    parser.add_argument("--weight-locality", type=float, default=0.7)
    parser.add_argument("--weight-rule", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=40)
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
