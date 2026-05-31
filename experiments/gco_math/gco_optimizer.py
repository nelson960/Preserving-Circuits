"""Geometric Continual Optimizer for transformer MLP weights."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import torch


def merge_pathway_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, object]] = {}
    counts: dict[int, int] = {}
    for record in records:
        parameter = record["parameter"]
        pathway = record["pathway"]
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError("GCO pathway record parameter must be a torch.nn.Parameter.")
        if not isinstance(pathway, torch.Tensor):
            raise TypeError("GCO pathway record pathway must be a torch.Tensor.")
        key = id(parameter)
        if key not in grouped:
            grouped[key] = dict(record)
            grouped[key]["pathway"] = pathway.detach().clone()
            counts[key] = 1
        else:
            existing = grouped[key]["pathway"]
            if not isinstance(existing, torch.Tensor):
                raise TypeError("Internal GCO grouped pathway is not a tensor.")
            if existing.shape != pathway.shape:
                raise ValueError(f"GCO pathway shape changed for {record.get('name', '<unnamed>')}.")
            grouped[key]["pathway"] = existing + pathway.detach()
            counts[key] += 1
    merged: list[dict[str, object]] = []
    for key, record in grouped.items():
        pathway = record["pathway"]
        if not isinstance(pathway, torch.Tensor):
            raise TypeError("Internal GCO merged pathway is not a tensor.")
        record["pathway"] = pathway / float(counts[key])
        merged.append(record)
    return merged


class GeometricContinualOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        pressure_beta: float = 0.99,
        pressure_gamma_base: float = 20.0,
        pressure_mu_base: float = 0.01,
        pressure_warmup_steps: int = 20,
        interference_threshold: float = 0.05,
        projection_mode: str = "pre",
    ) -> None:
        if lr <= 0.0:
            raise ValueError("GCO lr must be positive.")
        if weight_decay < 0.0:
            raise ValueError("GCO weight_decay must be non-negative.")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("GCO betas must be in [0, 1).")
        if eps <= 0.0:
            raise ValueError("GCO eps must be positive.")
        if not 0.0 <= pressure_beta < 1.0:
            raise ValueError("GCO pressure_beta must be in [0, 1).")
        if pressure_gamma_base <= 0.0:
            raise ValueError("GCO pressure_gamma_base must be positive.")
        if pressure_mu_base < 0.0:
            raise ValueError("GCO pressure_mu_base must be non-negative.")
        if pressure_warmup_steps < 0:
            raise ValueError("GCO pressure_warmup_steps must be non-negative.")
        if interference_threshold < 0.0:
            raise ValueError("GCO interference_threshold must be non-negative.")
        if projection_mode not in {"pre", "post"}:
            raise ValueError("GCO projection_mode must be 'pre' or 'post'.")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "pressure_beta": pressure_beta,
            "pressure_gamma_base": pressure_gamma_base,
            "pressure_mu_base": pressure_mu_base,
            "pressure_warmup_steps": pressure_warmup_steps,
            "interference_threshold": interference_threshold,
            "projection_mode": projection_mode,
        }
        super().__init__(list(params), defaults)
        self._pathways: dict[int, dict[str, object]] = {}
        self._global_step = 0
        self.last_metrics: dict[str, float] = {
            "gco_projected_parameter_count": 0.0,
            "gco_pressure_mean": 0.0,
            "gco_pressure_max": 0.0,
            "gco_overlap_mean": 0.0,
            "gco_safe_update_ratio_mean": 1.0,
            "gco_projection_delta_ratio_mean": 0.0,
        }

    def set_pathways(self, records: Iterable[Mapping[str, object]]) -> None:
        self._pathways = {}
        for record in merge_pathway_records(records):
            parameter = record["parameter"]
            pathway = record["pathway"]
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError("GCO pathway record parameter must be a torch.nn.Parameter.")
            if not isinstance(pathway, torch.Tensor):
                raise TypeError("GCO pathway record pathway must be a torch.Tensor.")
            if parameter.shape != pathway.shape:
                raise ValueError(
                    f"GCO pathway shape {tuple(pathway.shape)} does not match parameter "
                    f"{record.get('name', '<unnamed>')} shape {tuple(parameter.shape)}."
                )
            self._pathways[id(parameter)] = record

    @staticmethod
    def _rowwise_project(update: torch.Tensor, weight: torch.Tensor, pressure: torch.Tensor, eps: float) -> torch.Tensor:
        if update.ndim != 2 or weight.ndim != 2 or pressure.ndim != 2:
            raise ValueError("GCO row-wise projection requires 2D update, weight, and pressure tensors.")
        denom = (weight * weight).sum(dim=1, keepdim=True).clamp_min(eps)
        scale = (update * weight).sum(dim=1, keepdim=True) / denom
        return update - pressure * scale * weight

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._global_step += 1
        projected_count = 0
        pressure_sum = 0.0
        pressure_max = 0.0
        overlap_sum = 0.0
        safe_ratio_sum = 0.0
        projection_delta_ratio_sum = 0.0
        ratio_count = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            projection_mode = str(group["projection_mode"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if not torch.isfinite(grad).all():
                    raise FloatingPointError("Non-finite gradient received by GCO.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
                    raise TypeError("GCO optimizer state is corrupted.")

                pathway_record = self._pathways.get(id(parameter))
                pressure = None
                overlap = None
                moment_grad = grad
                safe_ratio_value: float | None = None
                projection_delta_ratio_value: float | None = None
                if pathway_record is not None:
                    pathway = pathway_record["pathway"]
                    if not isinstance(pathway, torch.Tensor):
                        raise TypeError("GCO pathway record pathway must be a tensor.")
                    if parameter.ndim != 2:
                        raise ValueError(f"GCO protected parameter must be 2D: {pathway_record.get('name', '<unnamed>')}")
                    pathway = pathway.to(device=parameter.device, dtype=parameter.dtype)
                    if pathway.shape != parameter.shape:
                        raise ValueError(
                            f"GCO pathway shape {tuple(pathway.shape)} does not match protected parameter "
                            f"{pathway_record.get('name', '<unnamed>')} shape {tuple(parameter.shape)}."
                        )
                    if not torch.isfinite(pathway).all():
                        raise FloatingPointError("Non-finite activation pathway received by GCO.")
                    if "pressure_history" not in state:
                        state["pressure_history"] = torch.zeros_like(parameter)
                    pressure_history = state["pressure_history"]
                    if not isinstance(pressure_history, torch.Tensor):
                        raise TypeError("GCO pressure_history state is corrupted.")
                    pressure_history.mul_(float(group["pressure_beta"])).addcmul_(
                        grad.abs(),
                        pathway,
                        value=1.0 - float(group["pressure_beta"]),
                    )
                    layer_index = int(pathway_record["layer_index"])
                    layer_count = int(pathway_record["layer_count"])
                    if layer_count <= 0 or not 0 <= layer_index < layer_count:
                        raise ValueError("GCO pathway record has invalid layer index/count.")
                    mu = float(group["pressure_mu_base"]) * float(layer_count - layer_index) / float(layer_count)
                    gamma = float(group["pressure_gamma_base"]) * float(layer_index + 1) / float(layer_count)
                    warmup_steps = int(group["pressure_warmup_steps"])
                    warmup_scale = 1.0 if warmup_steps == 0 else min(1.0, float(self._global_step) / float(warmup_steps))
                    pressure = torch.sigmoid(gamma * (pressure_history - mu)) * warmup_scale
                    overlap = (pressure * pathway).sum() / pathway.sum().clamp_min(eps)
                    threshold = float(group["interference_threshold"])
                    if threshold > 0.0 and overlap.item() < threshold:
                        pressure = pressure * max(0.0, min(1.0, float(overlap.item()) / threshold))
                    if projection_mode == "pre":
                        projected_grad = self._rowwise_project(grad, parameter, pressure, eps)
                        raw_grad_norm = grad.norm()
                        if raw_grad_norm.item() > 0.0:
                            safe_ratio_value = float((projected_grad.norm() / raw_grad_norm.clamp_min(eps)).detach().cpu())
                            projection_delta_ratio_value = float(
                                ((grad - projected_grad).norm() / raw_grad_norm.clamp_min(eps)).detach().cpu()
                            )
                        moment_grad = projected_grad
                    projected_count += 1
                    pressure_sum += float(pressure.mean().detach().cpu())
                    pressure_max = max(pressure_max, float(pressure.max().detach().cpu()))
                    overlap_sum += float(overlap.detach().cpu())

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(moment_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(moment_grad, moment_grad, value=1.0 - beta2)
                step = int(state["step"])
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
                adam_direction = exp_avg / denom
                safe_direction = adam_direction
                if pathway_record is not None and projection_mode == "post":
                    assert pressure is not None
                    safe_direction = self._rowwise_project(adam_direction, parameter, pressure, eps)
                    raw_norm = adam_direction.norm()
                    if raw_norm.item() > 0.0:
                        safe_ratio_value = float((safe_direction.norm() / raw_norm.clamp_min(eps)).detach().cpu())
                        projection_delta_ratio_value = float(
                            ((adam_direction - safe_direction).norm() / raw_norm.clamp_min(eps)).detach().cpu()
                        )
                if safe_ratio_value is not None:
                    safe_ratio_sum += safe_ratio_value
                    assert projection_delta_ratio_value is not None
                    projection_delta_ratio_sum += projection_delta_ratio_value
                    ratio_count += 1
                parameter.add_(safe_direction, alpha=-lr / bias_correction1)

        if projected_count > 0:
            self.last_metrics = {
                "gco_projected_parameter_count": float(projected_count),
                "gco_pressure_mean": pressure_sum / float(projected_count),
                "gco_pressure_max": pressure_max,
                "gco_overlap_mean": overlap_sum / float(projected_count),
                "gco_safe_update_ratio_mean": safe_ratio_sum / float(max(1, ratio_count)),
                "gco_projection_delta_ratio_mean": projection_delta_ratio_sum / float(max(1, ratio_count)),
            }
        else:
            self.last_metrics = {
                "gco_projected_parameter_count": 0.0,
                "gco_pressure_mean": 0.0,
                "gco_pressure_max": 0.0,
                "gco_overlap_mean": 0.0,
                "gco_safe_update_ratio_mean": 1.0,
                "gco_projection_delta_ratio_mean": 0.0,
            }
        return loss
