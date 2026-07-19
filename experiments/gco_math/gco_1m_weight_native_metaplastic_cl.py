"""Test fixed-state, weight-native continual learning on the 1M transformer.

The learner has no replay bank, protected-example list, semantic anchor table,
or learned role controller.  Every native model matrix is parameterized as

    W_effective = W_slow + W_fast.

The matrix also owns fixed-size gradient, sensitivity, and importance traces.
Incoming language-model gradients update these tensors, define an intrinsic
row/column tangent step, consolidate recurrent fast changes into slow weights,
and release conflicting protection under a fixed fast-weight capacity budget.

Category labels and historical examples are used only after updates for offline
evaluation.  They are never passed to the weight update operator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.nn.utils import parametrize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_1m_consequence_survival_cl import (
    permute_windows,
    split_windows,
)
from experiments.gco_math.gco_1m_long_horizon_consolidation import (
    CycleData,
    build_long_horizon_data,
)
from experiments.gco_math.gco_tiny_text_dependency_cl import (
    TextWindows,
    collect_geometry,
    combine_windows,
    evaluate_correction_queries,
    evaluate_windows,
    geometry_report,
    instantiate_model,
)


@dataclass(frozen=True)
class UpdateConfig:
    learning_rate: float
    gradient_mean_decay: float
    gradient_square_decay: float
    importance_decay: float
    protection_strength: float
    tangent_strength: float
    tangent_sweeps: int
    gradient_mode_decay: float
    gradient_mode_power_steps: int
    consolidation_rate: float
    release_rate: float
    fast_decay: float
    fast_capacity_ratio: float
    capacity_decay: float
    step_relative_radius: float
    epsilon: float


class IntrinsicFastSlowWeight(nn.Module):
    """Dense fast weights with factorized row/column metaplastic state."""

    def __init__(
        self,
        initial_weight: torch.Tensor,
        *,
        gradient_mode_rank: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        if initial_weight.ndim != 2:
            raise ValueError(
                f"Weight-native plasticity requires matrices, got {initial_weight.shape}."
            )
        if epsilon <= 0.0 or not math.isfinite(epsilon):
            raise ValueError("Metaplastic epsilon must be positive and finite.")
        if gradient_mode_rank <= 0 or gradient_mode_rank > min(initial_weight.shape):
            raise ValueError(
                f"gradient_mode_rank must be in [1, {min(initial_weight.shape)}]."
            )
        self.fast = nn.Parameter(torch.zeros_like(initial_weight))
        squared = initial_weight.detach().square()
        row_energy = squared.mean(dim=1, keepdim=True)
        column_energy = squared.mean(dim=0, keepdim=True)
        self.register_buffer("row_gradient_mean", torch.zeros_like(row_energy))
        self.register_buffer("column_gradient_mean", torch.zeros_like(column_energy))
        self.register_buffer("row_gradient_square", torch.zeros_like(row_energy))
        self.register_buffer("column_gradient_square", torch.zeros_like(column_energy))
        self.register_buffer("row_conflict", torch.zeros_like(row_energy))
        self.register_buffer("column_conflict", torch.zeros_like(column_energy))
        row_scale = row_energy.mean().clamp_min(epsilon)
        column_scale = column_energy.mean().clamp_min(epsilon)
        self.register_buffer("row_importance", row_energy / (row_energy + row_scale))
        self.register_buffer(
            "column_importance", column_energy / (column_energy + column_scale)
        )
        self.register_buffer(
            "gradient_mode_left",
            torch.zeros(
                initial_weight.shape[0],
                gradient_mode_rank,
                device=initial_weight.device,
                dtype=initial_weight.dtype,
            ),
        )
        self.register_buffer(
            "gradient_mode_right",
            torch.zeros(
                initial_weight.shape[1],
                gradient_mode_rank,
                device=initial_weight.device,
                dtype=initial_weight.dtype,
            ),
        )
        self.register_buffer(
            "gradient_mode_strength",
            torch.zeros(
                gradient_mode_rank,
                device=initial_weight.device,
                dtype=initial_weight.dtype,
            ),
        )
        self.register_buffer(
            "last_mode_recurrence",
            torch.zeros((), device=initial_weight.device, dtype=initial_weight.dtype),
        )
        self.register_buffer(
            "last_mode_opposition",
            torch.zeros((), device=initial_weight.device, dtype=initial_weight.dtype),
        )
        self.register_buffer(
            "observation_count", torch.zeros((), dtype=torch.long, device=initial_weight.device)
        )

    def forward(self, slow: torch.Tensor) -> torch.Tensor:
        if slow.shape != self.fast.shape:
            raise RuntimeError("Slow and fast weight shapes diverged.")
        return slow + self.fast

    def persistent_scalars(self) -> int:
        tensors = (
            self.fast,
            self.row_gradient_mean,
            self.column_gradient_mean,
            self.row_gradient_square,
            self.column_gradient_square,
            self.row_conflict,
            self.column_conflict,
            self.row_importance,
            self.column_importance,
            self.gradient_mode_left,
            self.gradient_mode_right,
            self.gradient_mode_strength,
            self.last_mode_recurrence,
            self.last_mode_opposition,
        )
        return sum(value.numel() for value in tensors) + self.observation_count.numel()

    def importance_field(self) -> torch.Tensor:
        return 1.0 - (1.0 - self.row_importance) * (1.0 - self.column_importance)

    def conflict_field(self) -> torch.Tensor:
        return 1.0 - (1.0 - self.row_conflict) * (1.0 - self.column_conflict)


@dataclass(frozen=True)
class NativeWeightState:
    name: str
    parametrization: IntrinsicFastSlowWeight
    slow: nn.Parameter


def validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and in [0, 1).")


def validate_args(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "tokenizer_path", "book_path"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    for name in (
        "seq_len",
        "stride",
        "base_book_words",
        "base_fact_words",
        "base_candidate_windows",
        "guard_windows",
        "cycles",
        "cycle_book_words",
        "cycle_book_windows",
        "cycle_fact_windows",
        "cycle_correction_count",
        "cycle_novel_count",
        "cycle_eval_windows",
        "rare_confirmation_period",
        "rare_confirmation_windows",
        "misinformation_variants",
        "misinformation_windows",
        "noise_windows",
        "micro_batch_windows",
        "inner_steps",
        "tangent_sweeps",
        "gradient_mode_rank",
        "gradient_mode_power_steps",
        "calibration_windows",
        "calibration_passes",
        "print_every",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    for name in (
        "cycle_novel_start",
        "correction_source_start",
        "rare_fact_index",
        "misinformation_source_index",
        "confirmation_windows",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative.")
    for name in (
        "gradient_mean_decay",
        "gradient_square_decay",
        "importance_decay",
        "gradient_mode_decay",
    ):
        validate_probability(name, getattr(args, name))
    for name in (
        "learning_rate",
        "protection_strength",
        "consolidation_rate",
        "release_rate",
        "fast_decay",
        "fast_capacity_ratio",
        "capacity_decay",
        "step_relative_radius",
        "epsilon",
        "maximum_guard_loss_ratio",
    ):
        value = getattr(args, name)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be positive and finite.")
    if not 0.0 <= args.tangent_strength <= 1.0:
        raise ValueError("tangent_strength must be in [0, 1].")
    if not 0.0 < args.minimum_acceptable_cka <= 1.0:
        raise ValueError("minimum_acceptable_cka must be in (0, 1].")
    if args.minimum_cycle_gain < 0.0 or not math.isfinite(args.minimum_cycle_gain):
        raise ValueError("minimum_cycle_gain must be finite and non-negative.")
    if args.confirmation_windows > 0:
        if not 1 <= args.confirmation_source_cycle <= args.cycles:
            raise ValueError("confirmation_source_cycle must identify an existing cycle.")
        if not 1 <= args.confirmation_start_cycle <= args.cycles:
            raise ValueError("confirmation_start_cycle must identify an existing cycle.")
        if args.confirmation_start_cycle <= args.confirmation_source_cycle:
            raise ValueError("Confirmation must start after its source cycle.")


def attach_weight_native_state(
    model: nn.Module,
    *,
    gradient_mode_rank: int,
    epsilon: float,
) -> list[NativeWeightState]:
    states: list[NativeWeightState] = []
    native_modules = {id(module) for module in model.gco_modules()}
    for name, module in list(model.named_modules()):
        if id(module) not in native_modules:
            continue
        if not isinstance(module.W, nn.Parameter):
            raise RuntimeError(f"Native module {name} does not expose an unparameterized W.")
        plasticity = IntrinsicFastSlowWeight(
            module.W.detach(),
            gradient_mode_rank=gradient_mode_rank,
            epsilon=epsilon,
        ).to(
            device=module.W.device, dtype=module.W.dtype
        )
        parametrize.register_parametrization(module, "W", plasticity)
        slow = module.parametrizations.W.original
        slow.requires_grad_(False)
        plasticity.fast.requires_grad_(True)
        states.append(NativeWeightState(name=name, parametrization=plasticity, slow=slow))
    if not states:
        raise RuntimeError("No native matrices were parameterized.")
    allowed = {id(state.parametrization.fast) for state in states}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in allowed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable) != len(states):
        raise RuntimeError(
            f"Expected one trainable fast tensor per matrix, got {len(trainable)} for {len(states)}."
        )
    return states


def project_intrinsic_tangent(
    direction: torch.Tensor,
    effective_weight: torch.Tensor,
    importance: torch.Tensor,
    *,
    strength: float,
    sweeps: int,
    epsilon: float,
) -> tuple[torch.Tensor, float, float]:
    """Preserve importance-weighted row and column energies to first order."""

    if direction.shape != effective_weight.shape or direction.shape != importance.shape:
        raise ValueError("Intrinsic tangent tensors have incompatible shapes.")
    normal = importance * effective_weight
    projected = direction
    if strength > 0.0:
        for _sweep in range(sweeps):
            row_denominator = normal.square().sum(dim=1, keepdim=True)
            row_coefficient = (projected * normal).sum(dim=1, keepdim=True) / (
                row_denominator + epsilon
            )
            projected = projected - strength * row_coefficient * normal
            column_denominator = normal.square().sum(dim=0, keepdim=True)
            column_coefficient = (projected * normal).sum(dim=0, keepdim=True) / (
                column_denominator + epsilon
            )
            projected = projected - strength * column_coefficient * normal
    original_norm = torch.linalg.vector_norm(direction)
    removed = torch.linalg.vector_norm(direction - projected) / original_norm.clamp_min(epsilon)
    row_residual = (projected * normal).sum(dim=1).abs() / (
        torch.linalg.vector_norm(projected, dim=1)
        * torch.linalg.vector_norm(normal, dim=1)
    ).clamp_min(epsilon)
    column_residual = (projected * normal).sum(dim=0).abs() / (
        torch.linalg.vector_norm(projected, dim=0)
        * torch.linalg.vector_norm(normal, dim=0)
    ).clamp_min(epsilon)
    residual = torch.maximum(row_residual.max(), column_residual.max())
    return projected, float(removed.detach().cpu()), float(residual.detach().cpu())


def dominant_gradient_mode(
    gradient: torch.Tensor,
    *,
    power_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gradient.ndim != 2:
        raise ValueError("Gradient-mode extraction requires a matrix.")
    numerical_floor = torch.finfo(gradient.dtype).tiny
    column_energy = gradient.square().sum(dim=0)
    column_norm = torch.linalg.vector_norm(column_energy)
    if not torch.isfinite(column_norm) or column_norm <= numerical_floor:
        raise FloatingPointError("Gradient matrix has no finite dominant-mode seed.")
    right = column_energy / column_norm
    left = gradient @ right
    for _step in range(power_steps):
        left_norm = torch.linalg.vector_norm(left)
        if not torch.isfinite(left_norm) or left_norm <= numerical_floor:
            raise FloatingPointError("Gradient left mode collapsed during power iteration.")
        left = left / left_norm
        right = gradient.T @ left
        right_norm = torch.linalg.vector_norm(right)
        if not torch.isfinite(right_norm) or right_norm <= numerical_floor:
            raise FloatingPointError("Gradient right mode collapsed during power iteration.")
        right = right / right_norm
        left = gradient @ right
    left_norm = torch.linalg.vector_norm(left)
    if not torch.isfinite(left_norm) or left_norm <= numerical_floor:
        raise FloatingPointError("Final gradient left mode is invalid.")
    left = left / left_norm
    amplitude = left @ (gradient @ right)
    if amplitude < 0.0:
        left = -left
        amplitude = -amplitude
    gradient_norm = torch.linalg.vector_norm(gradient).clamp_min(numerical_floor)
    strength = (amplitude / gradient_norm).clamp(0.0, 1.0)
    return left, right, strength


@torch.no_grad()
def observe_gradient_modes(
    plasticity: IntrinsicFastSlowWeight,
    gradient: torch.Tensor,
    *,
    decay: float,
    power_steps: int,
    epsilon: float,
) -> dict[str, float]:
    left, right, current_strength = dominant_gradient_mode(
        gradient,
        power_steps=power_steps,
    )
    left_alignment = plasticity.gradient_mode_left.T @ left
    right_alignment = plasticity.gradient_mode_right.T @ right
    signed_similarity = left_alignment * right_alignment
    absolute_similarity = signed_similarity.abs().clamp(0.0, 1.0)
    stored_strength = plasticity.gradient_mode_strength.clamp(0.0, 1.0)
    positive_match = torch.relu(signed_similarity) * stored_strength
    negative_match = torch.relu(-signed_similarity) * stored_strength
    recurrence = positive_match.max()
    opposition = negative_match.max()
    maximum_similarity = absolute_similarity.max()
    routing_score = (
        maximum_similarity * absolute_similarity
        + (1.0 - maximum_similarity) * (1.0 - stored_strength)
    )
    selected = int(routing_score.argmax().item())

    plasticity.gradient_mode_strength.mul_(decay)
    selected_similarity = signed_similarity[selected]
    if stored_strength[selected] <= epsilon or selected_similarity >= 0.0:
        selected_left = left
        selected_right = right
        if stored_strength[selected] > epsilon:
            left_sign = torch.where(
                left_alignment[selected] >= 0.0,
                left.new_ones(()),
                -left.new_ones(()),
            )
            selected_left = left_sign * left
            selected_right = left_sign * right
        blended_left = (
            decay * plasticity.gradient_mode_left[:, selected]
            + (1.0 - decay) * selected_left
        )
        blended_right = (
            decay * plasticity.gradient_mode_right[:, selected]
            + (1.0 - decay) * selected_right
        )
        left_norm = torch.linalg.vector_norm(blended_left)
        right_norm = torch.linalg.vector_norm(blended_right)
        if left_norm <= epsilon or right_norm <= epsilon:
            raise FloatingPointError("Gradient mode update produced a zero vector.")
        plasticity.gradient_mode_left[:, selected].copy_(blended_left / left_norm)
        plasticity.gradient_mode_right[:, selected].copy_(blended_right / right_norm)
        if stored_strength[selected] <= epsilon:
            plasticity.gradient_mode_strength[selected].copy_(current_strength)
        else:
            plasticity.gradient_mode_strength[selected].add_(
                (1.0 - decay) * current_strength
            )
    plasticity.gradient_mode_strength.clamp_(0.0, 1.0)
    plasticity.last_mode_recurrence.copy_(recurrence)
    plasticity.last_mode_opposition.copy_(opposition)
    return {
        "mode_recurrence": float(recurrence.detach().cpu()),
        "mode_opposition": float(opposition.detach().cpu()),
        "mode_strength": float(plasticity.gradient_mode_strength.mean().detach().cpu()),
        "mode_selected": float(selected),
    }


@torch.no_grad()
def update_native_weight(
    state: NativeWeightState,
    *,
    config: UpdateConfig,
    observe: bool,
    consolidate: bool,
    learn: bool = True,
) -> dict[str, float]:
    plasticity = state.parametrization
    gradient = plasticity.fast.grad
    if gradient is None:
        raise RuntimeError(f"Fast weight {state.name} has no gradient.")
    if not torch.isfinite(gradient).all():
        raise FloatingPointError(f"Fast weight {state.name} has a non-finite gradient.")
    if observe:
        mode_report = observe_gradient_modes(
            plasticity,
            gradient,
            decay=config.gradient_mode_decay,
            power_steps=config.gradient_mode_power_steps,
            epsilon=config.epsilon,
        )
    else:
        mode_report = {
            "mode_recurrence": float(plasticity.last_mode_recurrence.detach().cpu()),
            "mode_opposition": float(plasticity.last_mode_opposition.detach().cpu()),
            "mode_strength": float(
                plasticity.gradient_mode_strength.mean().detach().cpu()
            ),
            "mode_selected": -1.0,
        }
    mode_recurrence = plasticity.last_mode_recurrence.clamp(0.0, 1.0)
    mode_opposition = plasticity.last_mode_opposition.clamp(0.0, 1.0)

    effective = state.slow + plasticity.fast
    row_weight_scale = effective.square().mean(dim=1, keepdim=True).sqrt().clamp_min(
        config.epsilon
    )
    column_weight_scale = effective.square().mean(dim=0, keepdim=True).sqrt().clamp_min(
        config.epsilon
    )
    row_signal = (gradient * effective).mean(dim=1, keepdim=True) / row_weight_scale
    column_signal = (gradient * effective).mean(dim=0, keepdim=True) / column_weight_scale
    row_gradient_energy = gradient.square().mean(dim=1, keepdim=True)
    column_gradient_energy = gradient.square().mean(dim=0, keepdim=True)

    previous_count = int(plasticity.observation_count.item())
    if previous_count > 0:
        previous_mean_correction = 1.0 - config.gradient_mean_decay**previous_count
        previous_square_correction = 1.0 - config.gradient_square_decay**previous_count
        previous_row_mean = plasticity.row_gradient_mean / previous_mean_correction
        previous_column_mean = plasticity.column_gradient_mean / previous_mean_correction
        previous_row_square = plasticity.row_gradient_square / previous_square_correction
        previous_column_square = (
            plasticity.column_gradient_square / previous_square_correction
        )
    else:
        previous_row_mean = torch.zeros_like(row_signal)
        previous_column_mean = torch.zeros_like(column_signal)
        previous_row_square = torch.zeros_like(row_signal)
        previous_column_square = torch.zeros_like(column_signal)

    if observe:
        plasticity.observation_count.add_(1)
        plasticity.row_gradient_mean.mul_(config.gradient_mean_decay).add_(
            row_signal, alpha=1.0 - config.gradient_mean_decay
        )
        plasticity.column_gradient_mean.mul_(config.gradient_mean_decay).add_(
            column_signal, alpha=1.0 - config.gradient_mean_decay
        )
        plasticity.row_gradient_square.mul_(config.gradient_square_decay).addcmul_(
            row_signal,
            row_signal,
            value=1.0 - config.gradient_square_decay,
        )
        plasticity.column_gradient_square.mul_(
            config.gradient_square_decay
        ).addcmul_(
            column_signal,
            column_signal,
            value=1.0 - config.gradient_square_decay,
        )

    observation_count = int(plasticity.observation_count.item())
    if observation_count <= 0:
        raise RuntimeError("A native update was attempted before any observation.")
    mean_correction = 1.0 - config.gradient_mean_decay**observation_count
    square_correction = 1.0 - config.gradient_square_decay**observation_count
    row_mean = plasticity.row_gradient_mean / mean_correction
    column_mean = plasticity.column_gradient_mean / mean_correction
    row_square = plasticity.row_gradient_square / square_correction
    column_square = plasticity.column_gradient_square / square_correction
    row_coherence = (
        row_mean.abs() / row_square.sqrt().clamp_min(config.epsilon)
    ).clamp(0.0, 1.0)
    column_coherence = (
        column_mean.abs() / column_square.sqrt().clamp_min(config.epsilon)
    ).clamp(0.0, 1.0)
    row_sensitivity = row_gradient_energy / (
        row_gradient_energy + row_gradient_energy.mean().clamp_min(config.epsilon)
    )
    column_sensitivity = column_gradient_energy / (
        column_gradient_energy
        + column_gradient_energy.mean().clamp_min(config.epsilon)
    )
    row_prior_product = row_signal * previous_row_mean
    column_prior_product = column_signal * previous_column_mean
    row_instant_conflict = torch.relu(-row_prior_product) / row_prior_product.abs().clamp_min(
        config.epsilon
    )
    column_instant_conflict = torch.relu(
        -column_prior_product
    ) / column_prior_product.abs().clamp_min(config.epsilon)
    row_evidence = row_signal.square() / (
        row_signal.square() + previous_row_square + config.epsilon
    )
    column_evidence = column_signal.square() / (
        column_signal.square() + previous_column_square + config.epsilon
    )

    if observe:
        row_recurrence = 1.0 - (1.0 - row_coherence) * (1.0 - mode_recurrence)
        column_recurrence = 1.0 - (1.0 - column_coherence) * (
            1.0 - mode_recurrence
        )
        plasticity.row_importance.add_(
            (1.0 - plasticity.row_importance)
            * (1.0 - config.importance_decay)
            * row_recurrence
            * row_sensitivity
        )
        plasticity.column_importance.add_(
            (1.0 - plasticity.column_importance)
            * (1.0 - config.importance_decay)
            * column_recurrence
            * column_sensitivity
        )
        plasticity.row_conflict.mul_(config.importance_decay).add_(
            row_instant_conflict * row_evidence,
            alpha=1.0 - config.importance_decay,
        )
        plasticity.column_conflict.mul_(config.importance_decay).add_(
            column_instant_conflict * column_evidence,
            alpha=1.0 - config.importance_decay,
        )
        plasticity.row_importance.mul_(
            torch.exp(-config.release_rate * plasticity.row_conflict * row_evidence)
        )
        plasticity.column_importance.mul_(
            torch.exp(
                -config.release_rate
                * plasticity.column_conflict
                * column_evidence
            )
        )
        plasticity.row_importance.clamp_(0.0, 1.0)
        plasticity.column_importance.clamp_(0.0, 1.0)

    importance = plasticity.importance_field()
    family_conflict = plasticity.conflict_field()
    conflict = 1.0 - (1.0 - family_conflict) * (1.0 - mode_opposition)
    family_coherence = torch.sqrt(row_coherence * column_coherence)
    coherence = 1.0 - (1.0 - family_coherence) * (1.0 - mode_recurrence)
    sensitivity = torch.sqrt(row_sensitivity * column_sensitivity)
    if not learn:
        slow_norm = torch.linalg.vector_norm(state.slow).clamp_min(config.epsilon)
        return {
            "gradient_norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
            "delta_relative_norm": 0.0,
            "radius_scale": 1.0,
            "tangent_removed": 0.0,
            "tangent_residual": 0.0,
            "coherence": float(coherence.mean().detach().cpu()),
            "sensitivity": float(sensitivity.mean().detach().cpu()),
            "importance": float(importance.mean().detach().cpu()),
            "instant_conflict": float(
                (
                    1.0
                    - (1.0 - row_instant_conflict)
                    * (1.0 - column_instant_conflict)
                )
                .mean()
                .detach()
                .cpu()
            ),
            "conflict": float(conflict.mean().detach().cpu()),
            "mode_recurrence": mode_report["mode_recurrence"],
            "mode_opposition": mode_report["mode_opposition"],
            "mode_strength": mode_report["mode_strength"],
            "mode_selected": mode_report["mode_selected"],
            "consequence_gate": 0.0,
            "consolidation": 0.0,
            "fast_retention": 0.0,
            "fast_capacity_ratio": float(
                (torch.linalg.vector_norm(plasticity.fast) / slow_norm).detach().cpu()
            ),
            "global_fast_capacity_ratio": 0.0,
            "capacity_pressure": 0.0,
            "capacity_scale": 1.0,
        }
    preconditioner = (row_gradient_energy * column_gradient_energy).clamp_min(
        config.epsilon**4
    ).pow(0.25)
    raw_direction = -gradient / preconditioner.clamp_min(config.epsilon)
    metric_direction = raw_direction / (1.0 + config.protection_strength * importance)
    tangent_direction, removed, tangent_residual = project_intrinsic_tangent(
        metric_direction,
        effective,
        importance,
        strength=config.tangent_strength,
        sweeps=config.tangent_sweeps,
        epsilon=config.epsilon,
    )

    proposed = config.learning_rate * tangent_direction
    predicted_gain = torch.relu(-gradient * proposed).sum()
    invariant_change = importance * effective * proposed
    protected_damage = 0.5 * (
        invariant_change.sum(dim=1).abs().sum()
        + invariant_change.sum(dim=0).abs().sum()
    )
    consequence_gate = predicted_gain / (
        predicted_gain
        + config.protection_strength * protected_damage
        + config.epsilon
    )
    delta = consequence_gate * proposed
    weight_norm = torch.linalg.vector_norm(effective).clamp_min(config.epsilon)
    delta_norm = torch.linalg.vector_norm(delta)
    maximum_delta_norm = config.step_relative_radius * weight_norm
    radius_scale = torch.minimum(
        torch.ones((), device=delta.device, dtype=delta.dtype),
        maximum_delta_norm / delta_norm.clamp_min(config.epsilon),
    )
    delta.mul_(radius_scale)
    plasticity.fast.add_(delta)

    consolidation_fraction = torch.zeros_like(plasticity.fast)
    if consolidate:
        consolidation_fraction = (
            config.consolidation_rate
            * coherence.square()
            * sensitivity
            * (1.0 - conflict)
        ).clamp(0.0, 1.0)
        transfer = consolidation_fraction * plasticity.fast
        state.slow.add_(transfer)
        plasticity.fast.sub_(transfer)

    retention = (coherence * sensitivity * (1.0 - conflict)).clamp(0.0, 1.0)
    if consolidate:
        plasticity.fast.mul_(torch.exp(-config.fast_decay * (1.0 - retention)))
    slow_norm = torch.linalg.vector_norm(state.slow).clamp_min(config.epsilon)
    if not all(
        torch.isfinite(value).all()
        for value in (
            state.slow,
            plasticity.fast,
            plasticity.row_importance,
            plasticity.column_importance,
            plasticity.row_gradient_mean,
            plasticity.column_gradient_mean,
            plasticity.row_gradient_square,
            plasticity.column_gradient_square,
            plasticity.row_conflict,
            plasticity.column_conflict,
            plasticity.gradient_mode_left,
            plasticity.gradient_mode_right,
            plasticity.gradient_mode_strength,
            plasticity.last_mode_recurrence,
            plasticity.last_mode_opposition,
        )
    ):
        raise FloatingPointError(f"Native state became non-finite for {state.name}.")

    return {
        "gradient_norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
        "delta_relative_norm": float((torch.linalg.vector_norm(delta) / weight_norm).detach().cpu()),
        "radius_scale": float(radius_scale.detach().cpu()),
        "tangent_removed": removed,
        "tangent_residual": tangent_residual,
        "coherence": float(coherence.mean().detach().cpu()),
        "sensitivity": float(sensitivity.mean().detach().cpu()),
        "importance": float(importance.mean().detach().cpu()),
        "instant_conflict": float(
            (1.0 - (1.0 - row_instant_conflict) * (1.0 - column_instant_conflict))
            .mean()
            .detach()
            .cpu()
        ),
        "conflict": float(conflict.mean().detach().cpu()),
        "mode_recurrence": mode_report["mode_recurrence"],
        "mode_opposition": mode_report["mode_opposition"],
        "mode_strength": mode_report["mode_strength"],
        "mode_selected": mode_report["mode_selected"],
        "consequence_gate": float(consequence_gate.detach().cpu()),
        "consolidation": float(consolidation_fraction.mean().detach().cpu()),
        "fast_retention": float(retention.mean().detach().cpu()),
        "fast_capacity_ratio": float(
            (torch.linalg.vector_norm(plasticity.fast) / slow_norm).detach().cpu()
        ),
        "global_fast_capacity_ratio": 0.0,
        "capacity_pressure": 0.0,
        "capacity_scale": 1.0,
    }


@torch.no_grad()
def enforce_global_fast_capacity(
    states: list[NativeWeightState],
    module_reports: list[dict[str, float]],
    *,
    config: UpdateConfig,
) -> dict[str, float]:
    if len(states) != len(module_reports) or not states:
        raise ValueError("Global capacity requires one report per native matrix.")
    slow_square = sum(state.slow.detach().square().sum() for state in states)
    fast_square = sum(
        state.parametrization.fast.detach().square().sum() for state in states
    )
    slow_norm = torch.sqrt(slow_square).clamp_min(config.epsilon)
    fast_norm = torch.sqrt(fast_square)
    budget = config.fast_capacity_ratio * slow_norm
    pressure = torch.relu(fast_norm / budget - 1.0)
    if float(pressure.detach().cpu()) > 0.0:
        maximum_utility = max(report["fast_retention"] for report in module_reports)
        for state, report in zip(states, module_reports, strict=True):
            utility_gap = maximum_utility - report["fast_retention"]
            state.parametrization.fast.mul_(
                math.exp(
                    -config.capacity_decay
                    * float(pressure.detach().cpu())
                    * utility_gap
                )
            )
    fast_square_after_utility = sum(
        state.parametrization.fast.detach().square().sum() for state in states
    )
    fast_norm_after_utility = torch.sqrt(fast_square_after_utility)
    capacity_scale = torch.minimum(
        torch.ones(
            (),
            device=fast_norm_after_utility.device,
            dtype=fast_norm_after_utility.dtype,
        ),
        budget / fast_norm_after_utility.clamp_min(config.epsilon),
    )
    for state in states:
        state.parametrization.fast.mul_(capacity_scale)
    final_fast_square = sum(
        state.parametrization.fast.detach().square().sum() for state in states
    )
    final_ratio = torch.sqrt(final_fast_square) / slow_norm
    if not torch.isfinite(final_ratio) or final_ratio > (
        config.fast_capacity_ratio + 10.0 * config.epsilon
    ):
        raise RuntimeError(
            f"Global fast capacity invariant failed: ratio={float(final_ratio):.8f}."
        )
    return {
        "global_fast_capacity_ratio": float(final_ratio.detach().cpu()),
        "capacity_pressure": float(pressure.detach().cpu()),
        "capacity_scale": float(capacity_scale.detach().cpu()),
    }


def aggregate_update_reports(reports: list[dict[str, float]]) -> dict[str, float]:
    if not reports:
        raise ValueError("Cannot aggregate zero native-weight reports.")
    keys = reports[0].keys()
    if any(report.keys() != keys for report in reports[1:]):
        raise RuntimeError("Native-weight report schemas differ.")
    return {key: sum(report[key] for report in reports) / len(reports) for key in keys}


def train_stream_batch(
    model: nn.Module,
    states: list[NativeWeightState],
    batch: TextWindows,
    *,
    config: UpdateConfig,
    inner_steps: int,
    device: torch.device,
) -> list[dict[str, float]]:
    inputs = batch.inputs.to(device)
    targets = batch.targets.to(device)
    reports: list[dict[str, float]] = []
    for inner_step in range(inner_steps):
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("Current-stream language loss is non-finite.")
        loss.backward()
        module_reports = [
            update_native_weight(
                state,
                config=config,
                observe=inner_step == 0,
                consolidate=inner_step + 1 == inner_steps,
                learn=True,
            )
            for state in states
        ]
        capacity = enforce_global_fast_capacity(
            states,
            module_reports,
            config=config,
        )
        for state, module_report in zip(states, module_reports, strict=True):
            slow_norm = torch.linalg.vector_norm(state.slow).clamp_min(config.epsilon)
            module_report["fast_capacity_ratio"] = float(
                (
                    torch.linalg.vector_norm(state.parametrization.fast) / slow_norm
                )
                .detach()
                .cpu()
            )
            module_report.update(capacity)
        report = aggregate_update_reports(module_reports)
        report["loss"] = float(loss.detach().cpu())
        report["inner_step"] = float(inner_step + 1)
        reports.append(report)
    model.zero_grad(set_to_none=True)
    return reports


def calibrate_intrinsic_state(
    model: nn.Module,
    states: list[NativeWeightState],
    windows: TextWindows,
    *,
    config: UpdateConfig,
    batch_size: int,
    passes: int,
    device: torch.device,
) -> list[dict[str, float]]:
    slow_before = [state.slow.detach().clone() for state in states]
    reports: list[dict[str, float]] = []
    for calibration_pass in range(passes):
        for batch in split_windows(windows, batch_size=batch_size):
            model.zero_grad(set_to_none=True)
            inputs = batch.inputs.to(device)
            targets = batch.targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Intrinsic calibration loss is non-finite.")
            loss.backward()
            module_reports = [
                update_native_weight(
                    state,
                    config=config,
                    observe=True,
                    consolidate=False,
                    learn=False,
                )
                for state in states
            ]
            report = aggregate_update_reports(module_reports)
            report["loss"] = float(loss.detach().cpu())
            report["pass"] = float(calibration_pass + 1)
            reports.append(report)
    model.zero_grad(set_to_none=True)
    for state, reference in zip(states, slow_before, strict=True):
        if not torch.equal(state.slow.detach(), reference):
            raise RuntimeError(f"Calibration changed slow weight {state.name}.")
        if torch.count_nonzero(state.parametrization.fast).item() != 0:
            raise RuntimeError(f"Calibration changed fast weight {state.name}.")
    return reports


def combine_history(cycles: list[CycleData], attribute: str) -> TextWindows:
    if not cycles:
        raise ValueError("History requires at least one completed cycle.")
    values = [getattr(cycle, attribute) for cycle in cycles]
    if not all(isinstance(value, TextWindows) for value in values):
        raise TypeError(f"Cycle attribute {attribute} is not TextWindows.")
    return combine_windows(values)


def prefix_windows(windows: TextWindows, count: int) -> TextWindows:
    if count <= 0:
        raise ValueError("Window prefix count must be positive.")
    selected = min(count, windows.inputs.shape[0])
    if selected <= 0:
        raise RuntimeError("Cannot select confirmations from an empty window set.")
    return TextWindows(
        inputs=windows.inputs[:selected].clone(),
        targets=windows.targets[:selected].clone(),
        groups=windows.groups[:selected],
    )


def evaluate_history(
    model: nn.Module,
    completed: list[CycleData],
    guard: TextWindows,
    base_geometry: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    current_geometry = collect_geometry(model, guard, device=device)
    geometry = geometry_report(base_geometry, current_geometry)
    correction_queries = [query for cycle in completed for query in cycle.correction_queries]
    archived_queries = [query for cycle in completed for query in cycle.archived_queries]
    misinformation_queries = [
        query for cycle in completed for query in cycle.misinformation_queries
    ]
    return {
        "stable_guard": evaluate_windows(model, guard, device=device),
        "book": evaluate_windows(
            model, combine_history(completed, "book_eval"), device=device
        ),
        "novel": evaluate_windows(
            model, combine_history(completed, "novel_eval"), device=device
        ),
        "corrected": evaluate_windows(
            model, combine_history(completed, "corrected_eval"), device=device
        ),
        "obsolete": evaluate_windows(
            model, combine_history(completed, "obsolete_eval"), device=device
        ),
        "rare": evaluate_windows(model, completed[-1].rare_eval, device=device),
        "misinformation": evaluate_windows(
            model, combine_history(completed, "misinformation_eval"), device=device
        ),
        "misinformation_truth": evaluate_windows(
            model,
            combine_history(completed, "misinformation_truth_eval"),
            device=device,
        ),
        "correction_preference": evaluate_correction_queries(
            model, correction_queries, device=device
        ),
        "archived_preference": evaluate_correction_queries(
            model, archived_queries, device=device
        ),
        "misinformation_false_preference": evaluate_correction_queries(
            model, misinformation_queries, device=device
        ),
        "geometry": geometry,
        "minimum_cka": min(layer["cka"] for layer in geometry.values()),
        "mean_relative_drift": sum(
            layer["relative_drift"] for layer in geometry.values()
        )
        / len(geometry),
    }


def intrinsic_state_report(states: list[NativeWeightState]) -> dict[str, Any]:
    modules: dict[str, dict[str, float]] = {}
    total_persistent = 0
    total_weights = 0
    total_slow_square = 0.0
    total_fast_square = 0.0
    for state in states:
        plasticity = state.parametrization
        slow_norm = torch.linalg.vector_norm(state.slow).clamp_min(1e-12)
        importance = plasticity.importance_field()
        conflict = plasticity.conflict_field()
        modules[state.name] = {
            "weights": float(state.slow.numel()),
            "fast_relative_norm": float(
                (torch.linalg.vector_norm(plasticity.fast) / slow_norm).detach().cpu()
            ),
            "importance_mean": float(importance.mean().detach().cpu()),
            "importance_max": float(importance.max().detach().cpu()),
            "conflict_mean": float(conflict.mean().detach().cpu()),
            "row_state_scalars": float(plasticity.row_importance.numel()),
            "column_state_scalars": float(plasticity.column_importance.numel()),
            "gradient_mode_rank": float(plasticity.gradient_mode_strength.numel()),
            "gradient_mode_strength_mean": float(
                plasticity.gradient_mode_strength.mean().detach().cpu()
            ),
            "observation_count": float(plasticity.observation_count.detach().cpu()),
        }
        total_weights += state.slow.numel()
        total_persistent += plasticity.persistent_scalars()
        total_slow_square += float(state.slow.detach().square().sum().cpu())
        total_fast_square += float(plasticity.fast.detach().square().sum().cpu())
    return {
        "modules": modules,
        "native_weight_scalars": total_weights,
        "metaplastic_state_scalars": total_persistent,
        "state_to_weight_ratio": total_persistent / total_weights,
        "global_fast_capacity_ratio": math.sqrt(total_fast_square)
        / max(math.sqrt(total_slow_square), 1e-12),
    }


def plot_results(
    cycle_reports: list[dict[str, Any]],
    update_reports: list[dict[str, float]],
    output_path: Path,
) -> None:
    if not cycle_reports or not update_reports:
        raise ValueError("Plotting requires cycle and update reports.")
    cycles = [report["cycle"] for report in cycle_reports]
    updates = list(range(1, len(update_reports) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    for key, color in (
        ("stable_guard", "#111827"),
        ("book", "#2563eb"),
        ("novel", "#16a34a"),
        ("corrected", "#ea580c"),
        ("obsolete", "#be123c"),
    ):
        axes[0, 0].plot(
            cycles,
            [report["evaluation"][key]["loss"] for report in cycle_reports],
            marker="o",
            color=color,
            label=key.replace("_", " "),
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Behavior through the incoming stream")
    axes[0, 0].set_ylabel("loss (log scale)")
    axes[0, 0].legend()
    axes[0, 1].plot(
        cycles,
        [report["evaluation"]["minimum_cka"] for report in cycle_reports],
        marker="o",
        color="#7c3aed",
        label="minimum CKA",
    )
    axes[0, 1].plot(
        cycles,
        [report["evaluation"]["mean_relative_drift"] for report in cycle_reports],
        marker="o",
        color="#64748b",
        label="mean relative drift",
    )
    axes[0, 1].set_title("Offline representational geometry")
    axes[0, 1].legend()
    axes[1, 0].plot(
        updates,
        [report["coherence"] for report in update_reports],
        label="recurrence coherence",
    )
    axes[1, 0].plot(
        updates,
        [report["importance"] for report in update_reports],
        label="dependency importance",
    )
    axes[1, 0].plot(
        updates,
        [report["conflict"] for report in update_reports],
        label="conflict release",
    )
    axes[1, 0].plot(
        updates,
        [report["mode_recurrence"] for report in update_reports],
        label="gradient-mode recurrence",
    )
    axes[1, 0].set_title("Intrinsic metaplastic state")
    axes[1, 0].legend()
    axes[1, 1].plot(
        updates,
        [report["global_fast_capacity_ratio"] for report in update_reports],
        label="global fast / slow norm",
    )
    axes[1, 1].plot(
        updates,
        [report["tangent_removed"] for report in update_reports],
        label="tangent removal",
    )
    axes[1, 1].plot(
        updates,
        [report["consolidation"] for report in update_reports],
        label="fast-to-slow transfer",
    )
    axes[1, 1].set_title("Plasticity and fixed capacity")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("cycle" if axis in axes[0] else "micro-update")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_weight_native_checkpoint(
    path: Path,
    model: nn.Module,
    states: list[NativeWeightState],
    source_checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_by_name: dict[str, dict[str, torch.Tensor]] = {}
    for state in states:
        plasticity = state.parametrization
        state_by_name[state.name] = {
            "slow": state.slow.detach().cpu(),
            "fast": plasticity.fast.detach().cpu(),
            "row_gradient_mean": plasticity.row_gradient_mean.detach().cpu(),
            "column_gradient_mean": plasticity.column_gradient_mean.detach().cpu(),
            "row_gradient_square": plasticity.row_gradient_square.detach().cpu(),
            "column_gradient_square": plasticity.column_gradient_square.detach().cpu(),
            "row_conflict": plasticity.row_conflict.detach().cpu(),
            "column_conflict": plasticity.column_conflict.detach().cpu(),
            "row_importance": plasticity.row_importance.detach().cpu(),
            "column_importance": plasticity.column_importance.detach().cpu(),
            "gradient_mode_left": plasticity.gradient_mode_left.detach().cpu(),
            "gradient_mode_right": plasticity.gradient_mode_right.detach().cpu(),
            "gradient_mode_strength": plasticity.gradient_mode_strength.detach().cpu(),
            "last_mode_recurrence": plasticity.last_mode_recurrence.detach().cpu(),
            "last_mode_opposition": plasticity.last_mode_opposition.detach().cpu(),
            "observation_count": plasticity.observation_count.detach().cpu(),
        }
    torch.save(
        {
            "format": "weight_native_gradient_modes_v3",
            "model_config": source_checkpoint["model_config"],
            "native_gco_config": source_checkpoint["native_gco_config"],
            "weight_native_state": state_by_name,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model = instantiate_model(checkpoint, device)
    states = attach_weight_native_state(
        model,
        gradient_mode_rank=args.gradient_mode_rank,
        epsilon=args.epsilon,
    )
    trainable = sum(state.parametrization.fast.numel() for state in states)
    if not args.min_trainable_parameters <= trainable <= args.max_trainable_parameters:
        raise RuntimeError(
            f"Expected {args.min_trainable_parameters}..{args.max_trainable_parameters} native "
            f"weights, found {trainable}."
        )
    vocabulary_size = int(checkpoint["model_config"]["vocab_size"])
    base_candidates, guard, cycles = build_long_horizon_data(
        args, tokenizer, vocabulary_size
    )
    base_geometry = collect_geometry(model, guard, device=device)
    base_guard = evaluate_windows(model, guard, device=device)
    confirmation_windows: TextWindows | None = None
    confirmation_queries = []
    if args.confirmation_windows > 0:
        source_cycle = cycles[args.confirmation_source_cycle - 1]
        confirmation_windows = prefix_windows(
            source_cycle.corrected_eval, args.confirmation_windows
        )
        confirmation_queries = list(source_cycle.correction_queries)
    config = UpdateConfig(
        learning_rate=args.learning_rate,
        gradient_mean_decay=args.gradient_mean_decay,
        gradient_square_decay=args.gradient_square_decay,
        importance_decay=args.importance_decay,
        protection_strength=args.protection_strength,
        tangent_strength=args.tangent_strength,
        tangent_sweeps=args.tangent_sweeps,
        gradient_mode_decay=args.gradient_mode_decay,
        gradient_mode_power_steps=args.gradient_mode_power_steps,
        consolidation_rate=args.consolidation_rate,
        release_rate=args.release_rate,
        fast_decay=args.fast_decay,
        fast_capacity_ratio=args.fast_capacity_ratio,
        capacity_decay=args.capacity_decay,
        step_relative_radius=args.step_relative_radius,
        epsilon=args.epsilon,
    )
    calibration_reports: list[dict[str, float]] = []
    if args.calibrate_intrinsic_state:
        calibration_reports = calibrate_intrinsic_state(
            model,
            states,
            prefix_windows(base_candidates, args.calibration_windows),
            config=config,
            batch_size=args.micro_batch_windows,
            passes=args.calibration_passes,
            device=device,
        )
    del base_candidates
    print("1M WEIGHT-NATIVE GRADIENT-MODE CONTINUAL LEARNING")
    print("=" * 160)
    print(
        f"device={device} matrices={len(states)} native_weights={trainable} cycles={args.cycles} "
        f"micro_batch={args.micro_batch_windows} inner_steps={args.inner_steps} "
        f"fast_capacity={args.fast_capacity_ratio:.4f}"
    )
    if calibration_reports:
        print(
            f"calibration_batches={len(calibration_reports)} "
            f"final_coherence={calibration_reports[-1]['coherence']:.3f} "
            f"final_mode_strength={calibration_reports[-1]['mode_strength']:.3f}"
        )

    completed: list[CycleData] = []
    cycle_reports: list[dict[str, Any]] = []
    update_reports: list[dict[str, float]] = []
    update_index = 0
    for cycle_index, cycle in enumerate(cycles, start=1):
        current_before = evaluate_windows(model, cycle.train_without_noise, device=device)
        incoming_parts = [cycle.stream]
        if confirmation_windows is not None and cycle_index >= args.confirmation_start_cycle:
            incoming_parts.append(confirmation_windows)
        incoming_stream = (
            incoming_parts[0]
            if len(incoming_parts) == 1
            else combine_windows(incoming_parts)
        )
        stream = (
            permute_windows(incoming_stream, seed=args.seed + 10_007 * cycle_index)
            if args.shuffle_stream
            else incoming_stream
        )
        batches = split_windows(stream, batch_size=args.micro_batch_windows)
        started = time.perf_counter()
        for batch in batches:
            reports = train_stream_batch(
                model,
                states,
                batch,
                config=config,
                inner_steps=args.inner_steps,
                device=device,
            )
            update_index += 1
            final_inner = reports[-1]
            final_inner["update"] = float(update_index)
            final_inner["cycle"] = float(cycle_index)
            final_inner["batch_windows"] = float(batch.inputs.shape[0])
            update_reports.append(final_inner)
            if update_index == 1 or update_index % args.print_every == 0:
                print(
                    f"cycle={cycle_index:02d} update={update_index:03d} "
                    f"loss={final_inner['loss']:.4f} coherence={final_inner['coherence']:.3f} "
                    f"importance={final_inner['importance']:.3f} conflict={final_inner['conflict']:.3f} "
                    f"fast={final_inner['global_fast_capacity_ratio']:.4f} "
                    f"tangent={final_inner['tangent_removed']:.3f}"
                )
        completed.append(cycle)
        evaluation = evaluate_history(
            model,
            completed,
            guard,
            base_geometry,
            device=device,
        )
        evaluation["confirmed_correction_preference"] = (
            None
            if not confirmation_queries
            else evaluate_correction_queries(model, confirmation_queries, device=device)
        )
        current_after = evaluate_windows(model, cycle.train_without_noise, device=device)
        cycle_report = {
            "cycle": cycle_index,
            "seconds": time.perf_counter() - started,
            "micro_updates": len(batches),
            "current_before": current_before,
            "current_after": current_after,
            "relative_current_gain": (
                current_before["loss"] - current_after["loss"]
            )
            / max(current_before["loss"], args.epsilon),
            "evaluation": evaluation,
        }
        cycle_reports.append(cycle_report)
        print(
            f"cycle={cycle_index:02d} done gain={cycle_report['relative_current_gain']:.3f} "
            f"guard={evaluation['stable_guard']['loss']:.4f} "
            f"novel={evaluation['novel']['loss']:.4f} corrected={evaluation['corrected']['loss']:.4f} "
            f"obsolete={evaluation['obsolete']['loss']:.4f} cka={evaluation['minimum_cka']:.4f} "
            f"seconds={cycle_report['seconds']:.1f}"
        )

    intrinsic = intrinsic_state_report(states)
    final = cycle_reports[-1]["evaluation"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "weight_native_gradient_mode_cl.png"
    json_path = args.output_dir / "weight_native_gradient_mode_cl.json"
    checkpoint_path = args.output_dir / "weight_native_gradient_mode_cl.pt"
    plot_results(cycle_reports, update_reports, plot_path)
    save_weight_native_checkpoint(checkpoint_path, model, states, checkpoint, args)
    output = {
        "question": (
            "Can fixed fast/slow weights with factorized dependency-family state learn a continual text stream, "
            "release conflicting structure, and preserve useful geometry without replay or growing anchors?"
        ),
        "scope": (
            "Single-seed architecture experiment. Historical examples, semantic category names, "
            "correction labels, and geometry references are offline measurements only. Optional "
            "confirmation windows model recurrence in the incoming environment; the update operator "
            "receives only their ordinary language-model gradients."
        ),
        "mechanism": {
            "effective_weight": "slow + fast",
            "initial_importance": "factorized row/column slow-weight energy prior",
            "recurrence": "row/column EMA directional-gradient coherence",
            "gradient_modes": (
                "fixed-rank competitive left/right update directions; positive matches consolidate "
                "and negative matches release"
            ),
            "dependency": "factorized gradient sensitivity and intrinsic importance",
            "consequence": "first-order gain divided by gain plus protected structural damage",
            "geometry": "importance-weighted row/column energy tangent",
            "forgetting": "conflict releases importance; capacity removes weak fast state",
            "consolidation": "recurrent coherent fast state transfers into slow weights",
            "capacity": "one global fixed fast-weight norm budget allocated by recurrence utility",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": {
            "native_weights": trainable,
            "matrices": len(states),
            "layers": len(model.blocks),
            "width": int(model.token_embedding.W.shape[1]),
        },
        "memory": intrinsic,
        "intrinsic_calibration": calibration_reports,
        "base_guard": base_guard,
        "cycles": cycle_reports,
        "updates": update_reports,
        "final": final,
        "validation": {
            "guard_loss_ratio": final["stable_guard"]["loss"] / base_guard["loss"],
            "minimum_cka": final["minimum_cka"],
            "mean_cycle_relative_gain": sum(
                cycle["relative_current_gain"] for cycle in cycle_reports
            )
            / len(cycle_reports),
            "all_cycles_learned": all(
                cycle["relative_current_gain"] >= args.minimum_cycle_gain
                for cycle in cycle_reports
            ),
            "guard_within_limit": final["stable_guard"]["loss"]
            <= args.maximum_guard_loss_ratio * base_guard["loss"],
            "geometry_within_limit": final["minimum_cka"]
            >= args.minimum_acceptable_cka,
            "fast_capacity_respected": intrinsic["global_fast_capacity_ratio"]
            <= args.fast_capacity_ratio + 10.0 * args.epsilon,
            "correction_preference_positive": final["correction_preference"][
                "new_minus_old_margin"
            ]
            > 0.0,
            "misinformation_rejected": final["misinformation_false_preference"][
                "new_minus_old_margin"
            ]
            < 0.0,
            "confirmed_correction_preference_positive": (
                None
                if final["confirmed_correction_preference"] is None
                else final["confirmed_correction_preference"]["new_minus_old_margin"]
                > 0.0
            ),
        },
        "artifacts": {
            "plot": str(plot_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nFINAL WEIGHT-NATIVE GRADIENT-MODE CONTINUAL-LEARNING STATE")
    print("-" * 160)
    print(
        f"guard={base_guard['loss']:.5f}->{final['stable_guard']['loss']:.5f} "
        f"book={final['book']['loss']:.5f} novel={final['novel']['loss']:.5f} "
        f"corrected={final['corrected']['loss']:.5f} obsolete={final['obsolete']['loss']:.5f} "
        f"correction_margin={final['correction_preference']['new_minus_old_margin']:.4f} "
        f"false_margin={final['misinformation_false_preference']['new_minus_old_margin']:.4f} "
        f"min_cka={final['minimum_cka']:.4f}"
    )
    print(
        f"fixed_metaplastic_scalars={intrinsic['metaplastic_state_scalars']} "
        f"state_to_weight_ratio={intrinsic['state_to_weight_ratio']:.2f}"
    )
    print(
        f"gates=learn:{output['validation']['all_cycles_learned']} "
        f"guard:{output['validation']['guard_within_limit']} "
        f"geometry:{output['validation']['geometry_within_limit']} "
        f"capacity:{output['validation']['fast_capacity_respected']} "
        f"correction:{output['validation']['correction_preference_positive']} "
        f"confirmed:{output['validation']['confirmed_correction_preference_positive']} "
        f"misinformation:{output['validation']['misinformation_rejected']}"
    )
    print(f"wrote_json={json_path}")
    print(f"wrote_plot={plot_path}")
    print(f"wrote_checkpoint={checkpoint_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "model/checkpoints/gco-storage-capacity-1m/one_m-mixed-5000w-seed0.pt"
        ),
    )
    parser.add_argument(
        "--tokenizer-path", type=Path, default=Path("data/real_book/tokenizer.json")
    )
    parser.add_argument("--book-path", type=Path, default=Path("data/real_book/book.txt"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/analysis/gco-1m-weight-native-gradient-modes-seed0"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-trainable-parameters", type=int, default=950_000)
    parser.add_argument("--max-trainable-parameters", type=int, default=1_200_000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--base-book-words", type=int, default=2500)
    parser.add_argument("--base-fact-words", type=int, default=2500)
    parser.add_argument("--base-candidate-windows", type=int, default=64)
    parser.add_argument(
        "--calibrate-intrinsic-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--calibration-windows", type=int, default=64)
    parser.add_argument("--calibration-passes", type=int, default=1)
    parser.add_argument("--guard-windows", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--cycle-book-words", type=int, default=500)
    parser.add_argument("--cycle-book-windows", type=int, default=8)
    parser.add_argument("--cycle-fact-windows", type=int, default=8)
    parser.add_argument("--cycle-correction-count", type=int, default=4)
    parser.add_argument("--cycle-novel-start", type=int, default=500)
    parser.add_argument("--cycle-novel-count", type=int, default=4)
    parser.add_argument("--correction-source-start", type=int, default=0)
    parser.add_argument("--correction-donor-offset", type=int, default=101)
    parser.add_argument("--confirmation-windows", type=int, default=0)
    parser.add_argument("--confirmation-source-cycle", type=int, default=1)
    parser.add_argument("--confirmation-start-cycle", type=int, default=2)
    parser.add_argument("--cycle-eval-windows", type=int, default=8)
    parser.add_argument("--rare-fact-index", type=int, default=40)
    parser.add_argument("--rare-confirmation-period", type=int, default=8)
    parser.add_argument("--rare-confirmation-windows", type=int, default=1)
    parser.add_argument("--misinformation-source-index", type=int, default=60)
    parser.add_argument("--misinformation-donor-offset", type=int, default=211)
    parser.add_argument("--misinformation-variants", type=int, default=7)
    parser.add_argument("--misinformation-windows", type=int, default=1)
    parser.add_argument("--noise-windows", type=int, default=2)
    parser.add_argument("--micro-batch-windows", type=int, default=8)
    parser.add_argument(
        "--shuffle-stream", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--inner-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--gradient-mean-decay", type=float, default=0.9)
    parser.add_argument("--gradient-square-decay", type=float, default=0.99)
    parser.add_argument("--importance-decay", type=float, default=0.98)
    parser.add_argument("--protection-strength", type=float, default=8.0)
    parser.add_argument("--tangent-strength", type=float, default=1.0)
    parser.add_argument("--tangent-sweeps", type=int, default=3)
    parser.add_argument("--gradient-mode-rank", type=int, default=4)
    parser.add_argument("--gradient-mode-decay", type=float, default=0.95)
    parser.add_argument("--gradient-mode-power-steps", type=int, default=3)
    parser.add_argument("--consolidation-rate", type=float, default=0.5)
    parser.add_argument("--release-rate", type=float, default=0.2)
    parser.add_argument("--fast-decay", type=float, default=0.01)
    parser.add_argument("--fast-capacity-ratio", type=float, default=0.01)
    parser.add_argument("--capacity-decay", type=float, default=1.0)
    parser.add_argument("--step-relative-radius", type=float, default=0.005)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--minimum-acceptable-cka", type=float, default=0.9)
    parser.add_argument("--maximum-guard-loss-ratio", type=float, default=1.5)
    parser.add_argument("--minimum-cycle-gain", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=4)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
