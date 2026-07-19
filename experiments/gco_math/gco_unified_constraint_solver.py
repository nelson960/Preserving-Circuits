"""Low-rank constrained update primitives for continual-learning experiments.

The module operates on flattened parameter deltas.  It does not know about
tasks, semantic labels, or model architecture.  Callers provide:

* a new-learning gradient;
* a fixed-rank streaming sketch of soft functional constraints;
* hard linearized inequalities for measurements that must not cross a floor;
* a restore gradient defining the desired movement inside the soft sketch.

The returned value is a parameter delta, not a gradient.  Learning, soft
restoration, hard protection, and the trust region are therefore resolved in
one optimization step instead of projection followed by an unrelated restore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class StreamingConstraintSketch:
    rows: torch.Tensor | None
    updates: int
    observed_rows: int


@dataclass(frozen=True)
class UnifiedStep:
    delta: torch.Tensor
    report: dict[str, Any]


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}.")


def _finite_matrix(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape={tuple(value.shape)}.")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values.")


def _row_spectrum(rows: torch.Tensor) -> torch.Tensor:
    _finite_matrix("constraint rows", rows)
    gram = (rows @ rows.T).detach().to(device="cpu", dtype=torch.float32)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    singular = torch.sqrt(eigenvalues).sort(descending=True).values
    if not torch.isfinite(singular).all():
        raise FloatingPointError("Constraint spectrum contains non-finite values.")
    return singular


def constraint_rank_report(
    rows: torch.Tensor,
    *,
    parameter_count: int,
    rank_tolerance: float,
) -> dict[str, float | int]:
    if parameter_count <= 0 or rows.shape[1] != parameter_count:
        raise ValueError(
            f"Constraint width={rows.shape[1]} does not match parameter_count={parameter_count}."
        )
    _positive_finite("rank_tolerance", rank_tolerance)
    singular = _row_spectrum(rows)
    maximum = singular[0]
    numerical_rank = int((singular > maximum * rank_tolerance).sum().item())
    positive = singular[singular > 0.0]
    probabilities = positive / positive.sum().clamp_min(1e-12)
    effective_rank = float(
        torch.exp(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).item()
    )
    return {
        "rows": int(rows.shape[0]),
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "free_parameter_fraction": float(1.0 - numerical_rank / parameter_count),
        "largest_singular_value": float(singular[0]),
        "smallest_kept_singular_value": (
            float(singular[numerical_rank - 1]) if numerical_rank else 0.0
        ),
    }


def frequent_directions_compress(
    rows: torch.Tensor,
    *,
    rank: int,
    rank_tolerance: float,
) -> torch.Tensor:
    """Compress rows with the deterministic Frequent-Directions shrink step."""

    _finite_matrix("rows", rows)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    _positive_finite("rank_tolerance", rank_tolerance)
    gram = (rows @ rows.T).detach().to(device="cpu", dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    singular = torch.sqrt(eigenvalues[order].clamp_min(0.0))
    left = eigenvectors[:, order]
    if not torch.isfinite(singular).all() or not torch.isfinite(left).all():
        raise FloatingPointError("Frequent-Directions decomposition is non-finite.")
    numerical_rank = int(
        (singular > singular[0].clamp_min(1e-12) * rank_tolerance).sum().item()
    )
    kept = min(rank, numerical_rank)
    if kept <= 0:
        raise RuntimeError("Constraint rows have zero numerical rank.")
    right = (
        left[:, :kept].T.to(rows) @ rows
    ) / singular[:kept].to(rows).unsqueeze(1).clamp_min(1e-12)
    if numerical_rank > rank:
        shrinkage = singular[rank].square()
        shrunk = torch.sqrt(
            (singular[:kept].square() - shrinkage).clamp_min(0.0)
        ).to(rows)
    else:
        shrunk = singular[:kept].to(rows)
    compressed = shrunk.unsqueeze(1) * right
    nonzero = torch.linalg.vector_norm(compressed, dim=1) > 1e-12
    compressed = compressed[nonzero]
    if compressed.shape[0] == 0 or not torch.isfinite(compressed).all():
        raise FloatingPointError("Frequent-Directions compression produced no finite rows.")
    return compressed.detach()


def update_streaming_sketch(
    sketch: StreamingConstraintSketch,
    new_rows: torch.Tensor,
    *,
    rank: int,
    decay: float,
    rank_tolerance: float,
) -> StreamingConstraintSketch:
    _finite_matrix("new_rows", new_rows)
    if not math.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError(f"decay must be in (0, 1], got {decay}.")
    blocks = [new_rows.detach()]
    if sketch.rows is not None:
        if sketch.rows.shape[1] != new_rows.shape[1]:
            raise ValueError("Existing and incoming sketch rows have different widths.")
        blocks.insert(0, math.sqrt(decay) * sketch.rows.to(new_rows))
    combined = torch.cat(blocks, dim=0)
    compressed = frequent_directions_compress(
        combined,
        rank=rank,
        rank_tolerance=rank_tolerance,
    )
    return StreamingConstraintSketch(
        rows=compressed,
        updates=sketch.updates + 1,
        observed_rows=sketch.observed_rows + int(new_rows.shape[0]),
    )


def _orthonormal_row_basis(
    rows: torch.Tensor,
    *,
    rank_tolerance: float,
) -> torch.Tensor:
    """Return an orthonormal basis spanning the supplied parameter-space rows."""

    _finite_matrix("generator rows", rows)
    row_norms = torch.linalg.vector_norm(rows, dim=1)
    if not torch.isfinite(row_norms).all() or bool((row_norms <= 1e-12).any()):
        raise FloatingPointError("Unified-step generator rows contain a zero or invalid row.")
    normalized_rows = rows / row_norms.unsqueeze(1)
    gram = (normalized_rows @ normalized_rows.T).detach().cpu().to(dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    singular = torch.sqrt(eigenvalues[order].clamp_min(0.0))
    numerical_floor = math.sqrt(torch.finfo(rows.dtype).eps)
    effective_tolerance = max(rank_tolerance, numerical_floor)
    numerical_rank = int(
        (singular > singular[0].clamp_min(1e-12) * effective_tolerance).sum().item()
    )
    if numerical_rank <= 0:
        raise RuntimeError("Unified-step generator rows have zero numerical rank.")
    left = eigenvectors[:, order[:numerical_rank]].T.to(rows)
    basis = (left @ normalized_rows) / singular[:numerical_rank].to(rows).unsqueeze(1)
    if not torch.isfinite(basis).all():
        raise FloatingPointError("Reduced unified-step basis is non-finite.")

    # The first Gram decomposition is performed from float32 device products.
    # Re-diagonalize the resulting small basis Gram and whiten it explicitly so
    # weak retained directions do not accumulate MPS roundoff across updates.
    basis_gram = (basis @ basis.T).detach().cpu().to(dtype=torch.float64)
    basis_values, basis_vectors = torch.linalg.eigh(basis_gram)
    basis_order = torch.argsort(basis_values, descending=True)
    ordered_values = basis_values[basis_order].clamp_min(0.0)
    reorthogonal_rank = int(
        (
            ordered_values
            > ordered_values[0].clamp_min(1e-12) * effective_tolerance**2
        ).sum().item()
    )
    if reorthogonal_rank <= 0:
        raise RuntimeError("Reduced unified-step basis vanished during reorthogonalization.")
    rotation = basis_vectors[:, basis_order[:reorthogonal_rank]].T.to(rows)
    basis = (rotation @ basis) / torch.sqrt(
        ordered_values[:reorthogonal_rank]
    ).to(rows).unsqueeze(1)
    if not torch.isfinite(basis).all():
        raise FloatingPointError("Reorthogonalized unified-step basis is non-finite.")
    identity = torch.eye(
        reorthogonal_rank,
        device=basis.device,
        dtype=basis.dtype,
    )
    orthogonality_error = torch.linalg.matrix_norm(basis @ basis.T - identity)
    if float(orthogonality_error.detach().cpu()) > 100.0 * effective_tolerance:
        raise RuntimeError(
            "Reduced unified-step basis is not sufficiently orthonormal: "
            f"error={float(orthogonality_error.detach().cpu()):.6g}."
        )
    return basis.detach()


def _solve_reduced_hard_qp(
    *,
    hessian: torch.Tensor,
    rhs: torch.Tensor,
    hard_rows: torch.Tensor,
    hard_lower_bounds: torch.Tensor,
    trust_multiplier: float,
    damping: float,
    feasibility_tolerance: float,
    max_active_set_steps: int,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Solve the reduced hard-constrained quadratic for one trust multiplier."""

    dimension = int(rhs.numel())
    identity = torch.eye(dimension, device=rhs.device, dtype=rhs.dtype)
    metric = hessian + (trust_multiplier + damping) * identity
    base = torch.linalg.solve(metric, rhs)
    active: list[int] = []
    multipliers = rhs.new_zeros(0)
    value = base
    for _step in range(max_active_set_steps if hard_rows.shape[0] else 1):
        if hard_rows.shape[0] == 0:
            return value, active, multipliers
        violations = hard_lower_bounds - hard_rows @ value
        worst_value, worst_index_tensor = torch.max(violations, dim=0)
        if float(worst_value) <= feasibility_tolerance:
            return value, active, multipliers
        worst_index = int(worst_index_tensor.item())
        if worst_index not in active:
            active.append(worst_index)
        while active:
            active_rows = hard_rows[active]
            inverse_rows = torch.linalg.solve(metric, active_rows.T)
            active_gram = active_rows @ inverse_rows
            active_rhs = hard_lower_bounds[active] - active_rows @ base
            least_squares = torch.linalg.lstsq(active_gram, active_rhs)
            multipliers = least_squares.solution
            equality_residual = active_gram @ multipliers - active_rhs
            if float(torch.linalg.vector_norm(equality_residual)) > feasibility_tolerance:
                raise RuntimeError(
                    "Active hard constraints are mutually inconsistent in the reduced space: "
                    f"residual={float(torch.linalg.vector_norm(equality_residual)):.6g}."
                )
            negative = multipliers < -feasibility_tolerance
            if negative.any():
                remove_position = int(torch.argmin(multipliers).item())
                del active[remove_position]
                continue
            value = base + inverse_rows @ multipliers
            break
        if not active:
            value = base
            multipliers = rhs.new_zeros(0)
    raise RuntimeError(
        f"Hard-constraint active set did not converge in {max_active_set_steps} steps."
    )


def solve_unified_step(
    *,
    new_gradient: torch.Tensor,
    restore_gradient: torch.Tensor,
    soft_rows: torch.Tensor,
    hard_rows: torch.Tensor,
    hard_lower_bounds: torch.Tensor,
    learning_rate: float,
    soft_penalty: float,
    soft_restore_fraction: float,
    trust_radius: float,
    damping: float,
    feasibility_tolerance: float,
    max_active_set_steps: int,
    rank_tolerance: float,
) -> UnifiedStep:
    """Solve one soft-restoring, hard-protected trust-region parameter step.

    The parameter delta minimizes a local quadratic around ``-lr * g_new``.
    Soft rows pull the delta toward the projected restore direction.  Hard rows
    impose ``H delta >= lower_bounds`` through an active-set metric projection.
    """

    for name, value in (
        ("learning_rate", learning_rate),
        ("soft_penalty", soft_penalty),
        ("trust_radius", trust_radius),
        ("damping", damping),
        ("feasibility_tolerance", feasibility_tolerance),
        ("rank_tolerance", rank_tolerance),
    ):
        _positive_finite(name, value)
    if not math.isfinite(soft_restore_fraction) or soft_restore_fraction < 0.0:
        raise ValueError("soft_restore_fraction must be finite and non-negative.")
    if max_active_set_steps <= 0:
        raise ValueError("max_active_set_steps must be positive.")
    if new_gradient.ndim != 1 or restore_gradient.shape != new_gradient.shape:
        raise ValueError("Learning and restore gradients must be equally shaped vectors.")
    if not torch.isfinite(new_gradient).all() or not torch.isfinite(restore_gradient).all():
        raise FloatingPointError("Unified solver received a non-finite gradient.")
    _finite_matrix("soft_rows", soft_rows)
    if hard_rows.ndim != 2 or hard_rows.shape[1] != new_gradient.numel():
        raise ValueError("Hard constraints must be a matrix with parameter-width rows.")
    if hard_rows.numel() and not torch.isfinite(hard_rows).all():
        raise FloatingPointError("hard_rows contains non-finite values.")
    if soft_rows.shape[1] != new_gradient.numel():
        raise ValueError("Constraint rows and flattened gradients have different widths.")
    if hard_lower_bounds.shape != (hard_rows.shape[0],):
        raise ValueError("Hard lower bounds do not match hard constraint rows.")

    raw_delta = -learning_rate * new_gradient
    soft_target = -learning_rate * soft_restore_fraction * (soft_rows @ restore_gradient)
    generator_rows = torch.cat([raw_delta.unsqueeze(0), soft_rows, hard_rows], dim=0)
    reduced_basis = _orthonormal_row_basis(
        generator_rows,
        rank_tolerance=rank_tolerance,
    )
    raw_reduced = reduced_basis @ raw_delta
    soft_reduced = soft_rows @ reduced_basis.T
    hard_reduced = hard_rows @ reduced_basis.T

    # The reduced problem is small, so solve it in CPU float64. This avoids
    # unsupported low-dimensional eigensolvers on MPS and gives hard constraints
    # substantially tighter numerical accuracy than parameter-space clipping.
    raw_cpu = raw_reduced.detach().cpu().to(dtype=torch.float64)
    soft_cpu = soft_reduced.detach().cpu().to(dtype=torch.float64)
    hard_cpu = hard_reduced.detach().cpu().to(dtype=torch.float64)
    target_cpu = soft_target.detach().cpu().to(dtype=torch.float64)
    bounds_cpu = hard_lower_bounds.detach().cpu().to(dtype=torch.float64)
    reduced_dimension = int(raw_cpu.numel())
    identity_cpu = torch.eye(reduced_dimension, dtype=torch.float64)
    hessian_cpu = (
        (1.0 / learning_rate) * identity_cpu
        + soft_penalty * (soft_cpu.T @ soft_cpu)
    )
    rhs_cpu = (
        (1.0 / learning_rate) * raw_cpu
        + soft_penalty * (soft_cpu.T @ target_cpu)
    )

    unconstrained_trust, unconstrained_active, unconstrained_multipliers = (
        _solve_reduced_hard_qp(
            hessian=hessian_cpu,
            rhs=rhs_cpu,
            hard_rows=hard_cpu,
            hard_lower_bounds=bounds_cpu,
            trust_multiplier=0.0,
            damping=damping,
            feasibility_tolerance=feasibility_tolerance,
            max_active_set_steps=max_active_set_steps,
        )
    )
    pre_clip_norm = torch.linalg.vector_norm(unconstrained_trust)
    if not torch.isfinite(pre_clip_norm):
        raise FloatingPointError("Unified solver produced a non-finite reduced update norm.")

    trust_multiplier = 0.0
    reduced_delta = unconstrained_trust
    active = unconstrained_active
    multipliers = unconstrained_multipliers
    if float(pre_clip_norm) > trust_radius + feasibility_tolerance:
        lower_multiplier = 0.0
        upper_multiplier = 1.0
        bracketed = False
        minimum_bracket_norm = float(pre_clip_norm)
        for _step in range(max_active_set_steps):
            candidate, candidate_active, candidate_multipliers = _solve_reduced_hard_qp(
                hessian=hessian_cpu,
                rhs=rhs_cpu,
                hard_rows=hard_cpu,
                hard_lower_bounds=bounds_cpu,
                trust_multiplier=upper_multiplier,
                damping=damping,
                feasibility_tolerance=feasibility_tolerance,
                max_active_set_steps=max_active_set_steps,
            )
            candidate_norm = float(torch.linalg.vector_norm(candidate))
            minimum_bracket_norm = min(minimum_bracket_norm, candidate_norm)
            if candidate_norm <= trust_radius:
                reduced_delta = candidate
                active = candidate_active
                multipliers = candidate_multipliers
                bracketed = True
                break
            lower_multiplier = upper_multiplier
            upper_multiplier *= 2.0
        if not bracketed:
            raise RuntimeError(
                "Hard constraints and trust radius have no resolved feasible intersection: "
                f"radius={trust_radius:.6g}, minimum_bracket_norm={minimum_bracket_norm:.6g}."
            )
        for _step in range(max_active_set_steps):
            midpoint = 0.5 * (lower_multiplier + upper_multiplier)
            candidate, candidate_active, candidate_multipliers = _solve_reduced_hard_qp(
                hessian=hessian_cpu,
                rhs=rhs_cpu,
                hard_rows=hard_cpu,
                hard_lower_bounds=bounds_cpu,
                trust_multiplier=midpoint,
                damping=damping,
                feasibility_tolerance=feasibility_tolerance,
                max_active_set_steps=max_active_set_steps,
            )
            if float(torch.linalg.vector_norm(candidate)) > trust_radius:
                lower_multiplier = midpoint
            else:
                upper_multiplier = midpoint
                reduced_delta = candidate
                active = candidate_active
                multipliers = candidate_multipliers
        trust_multiplier = upper_multiplier

    delta = reduced_basis.T @ reduced_delta.to(reduced_basis)
    delta_norm = torch.linalg.vector_norm(delta)
    if float(delta_norm.detach().cpu()) > trust_radius + feasibility_tolerance:
        raise RuntimeError(
            "Joint trust-region solve exceeded its radius: "
            f"norm={float(delta_norm.detach().cpu()):.6g}, radius={trust_radius:.6g}."
        )
    clip_scale = torch.clamp(
        delta.new_tensor(float(delta_norm.detach().cpu()))
        / delta.new_tensor(float(pre_clip_norm)).clamp_min(1e-12),
        max=1.0,
    )
    hard_slack = hard_rows @ delta - hard_lower_bounds
    minimum_slack = (
        float(hard_slack.min().detach().cpu()) if hard_slack.numel() else math.inf
    )
    if minimum_slack < -feasibility_tolerance:
        raise RuntimeError(
            "Joint trust-region solve returned an infeasible hard constraint: "
            f"minimum_slack={minimum_slack:.6g}."
        )
    raw_norm = torch.linalg.vector_norm(raw_delta).clamp_min(1e-12)
    predicted_gain = -torch.dot(new_gradient, delta)
    if not torch.isfinite(delta).all() or not torch.isfinite(predicted_gain):
        raise FloatingPointError("Unified solver produced a non-finite result.")
    combined_rows = torch.cat([hard_rows, soft_rows], dim=0)
    rank_report = constraint_rank_report(
        combined_rows,
        parameter_count=new_gradient.numel(),
        rank_tolerance=rank_tolerance,
    )
    report: dict[str, Any] = {
        "raw_delta_norm": float(raw_norm.detach().cpu()),
        "delta_norm": float(delta_norm.detach().cpu()),
        "trust_clip_scale": float(clip_scale.detach().cpu()),
        "trust_multiplier": trust_multiplier,
        "trust_region_active": trust_multiplier > 0.0,
        "reduced_dimension": reduced_dimension,
        "predicted_gain": float(predicted_gain.detach().cpu()),
        "active_hard_constraints": len(active),
        "active_hard_indices": active,
        "hard_constraint_count": int(hard_rows.shape[0]),
        "soft_constraint_count": int(soft_rows.shape[0]),
        "minimum_hard_slack": minimum_slack,
        "soft_target_error_before": float(
            torch.linalg.vector_norm(soft_rows @ raw_delta - soft_target).detach().cpu()
        ),
        "soft_target_error_after": float(
            torch.linalg.vector_norm(soft_rows @ delta - soft_target).detach().cpu()
        ),
        "new_gradient_retained_fraction": float((delta_norm / raw_norm).detach().cpu()),
        "multipliers": [float(value) for value in multipliers.detach().cpu()],
        "capacity": rank_report,
    }
    return UnifiedStep(delta=delta.detach(), report=report)


def apply_flat_delta(
    parameters: list[torch.nn.Parameter],
    delta: torch.Tensor,
) -> None:
    if delta.ndim != 1 or not torch.isfinite(delta).all():
        raise ValueError("Parameter delta must be a finite vector.")
    expected = sum(parameter.numel() for parameter in parameters)
    if delta.numel() != expected:
        raise ValueError(f"Delta has {delta.numel()} entries; expected {expected}.")
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(delta[offset : offset + count].reshape_as(parameter))
            offset += count
    if offset != delta.numel():
        raise RuntimeError("Flat parameter delta was not consumed exactly.")
