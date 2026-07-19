"""Bounded recurrent trace summaries versus full-history trace organization.

This experiment removes the per-observation assignment state used by the first
autonomous trace-field test. Attention is now a deterministic function of an
incoming representation and a fixed set of trace keys. Three conditions use the
same objective and slot budget:

* full_history: refits from every accumulated observation;
* recurrent_summary: retains only per-trace mass, mean, and covariance;
* current_only: learns each stage without retained evidence.

Hidden stream groups are evaluation-only. The experiment asks whether bounded
trace sufficient statistics preserve merge, branch, composition, novelty,
stable memory, and capacity-driven forgetting without retaining raw history.
The final stage contains novelty and noise but no old-group rehearsal by
default, so a current-batch-only system cannot pass by seeing old evidence again.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.real_book_common import resolve_device
from experiments.gco_math.gco_tiny_autonomous_trace_field import (
    GROUP_COLORS,
    EvidencePoint,
    build_stream,
)


@dataclass
class EvidenceMoments:
    means: torch.Tensor
    covariances: torch.Tensor
    weights: torch.Tensor


@dataclass
class TraceSummary:
    mode: str
    means: torch.Tensor
    weights: torch.Tensor
    full_covariances: torch.Tensor | None
    diagonal: torch.Tensor | None
    factors: torch.Tensor | None
    variance_trace: torch.Tensor | None

    def validate(self) -> None:
        if self.mode not in {"full", "diagonal", "lowrank", "trace"}:
            raise ValueError(f"Unknown trace-summary covariance mode {self.mode!r}.")
        if self.means.ndim != 2 or self.weights.shape != (self.means.shape[0],):
            raise ValueError("Trace-summary means and weights have incompatible shapes.")
        slots, dimension = self.means.shape
        if self.mode == "full":
            if self.full_covariances is None or self.full_covariances.shape != (
                slots,
                dimension,
                dimension,
            ):
                raise ValueError("Full trace summary requires [slots, D, D] covariances.")
            if self.diagonal is not None or self.factors is not None or self.variance_trace is not None:
                raise ValueError("Full trace summary cannot also store compressed covariance fields.")
        elif self.mode == "trace":
            if self.full_covariances is not None or self.diagonal is not None or self.factors is not None:
                raise ValueError("Trace-only summary cannot store full, diagonal, or low-rank covariance.")
            if self.variance_trace is None or self.variance_trace.shape != (slots,):
                raise ValueError("Trace-only summary requires one total-variance scalar per slot.")
        else:
            if self.full_covariances is not None:
                raise ValueError("Compressed trace summary cannot store full covariances.")
            if self.variance_trace is not None:
                raise ValueError("Diagonal and low-rank summaries cannot store trace-only variance.")
            if self.diagonal is None or self.diagonal.shape != (slots, dimension):
                raise ValueError("Compressed trace summary requires [slots, D] diagonal variance.")
            if self.mode == "diagonal":
                if self.factors is not None:
                    raise ValueError("Diagonal trace summary cannot store low-rank factors.")
            elif self.factors is None or self.factors.ndim != 3 or self.factors.shape[:2] != (
                slots,
                dimension,
            ):
                raise ValueError("Low-rank trace summary requires [slots, D, rank] factors.")
        for name, value in (
            ("means", self.means),
            ("weights", self.weights),
            ("full_covariances", self.full_covariances),
            ("diagonal", self.diagonal),
            ("factors", self.factors),
            ("variance_trace", self.variance_trace),
        ):
            if value is not None and not torch.isfinite(value).all():
                raise FloatingPointError(f"Trace-summary {name} contains non-finite values.")

    def to_moments(self) -> EvidenceMoments:
        self.validate()
        if self.mode == "full":
            if self.full_covariances is None:
                raise RuntimeError("Validated full trace summary has no covariance tensor.")
            covariances = self.full_covariances
        elif self.mode in {"diagonal", "lowrank"}:
            if self.diagonal is None:
                raise RuntimeError("Validated compressed trace summary has no diagonal variance.")
            covariances = torch.diag_embed(self.diagonal)
            if self.mode == "lowrank":
                if self.factors is None:
                    raise RuntimeError("Validated low-rank trace summary has no factor tensor.")
                covariances = covariances + self.factors @ self.factors.transpose(1, 2)
        else:
            if self.variance_trace is None:
                raise RuntimeError("Validated trace-only summary has no total-variance tensor.")
            isotropic = self.variance_trace / self.means.shape[1]
            covariances = torch.diag_embed(
                isotropic.unsqueeze(1).expand(-1, self.means.shape[1])
            )
        return EvidenceMoments(
            means=self.means,
            covariances=covariances,
            weights=self.weights,
        )

    def stored_scalars(self) -> int:
        self.validate()
        total = self.means.numel() + self.weights.numel()
        if self.full_covariances is not None:
            total += self.full_covariances.numel()
        if self.diagonal is not None:
            total += self.diagonal.numel()
        if self.factors is not None:
            total += self.factors.numel()
        if self.variance_trace is not None:
            total += self.variance_trace.numel()
        return int(total)


@dataclass
class FunctionalTraceSolution:
    centers: torch.Tensor
    attention: torch.Tensor
    usage: torch.Tensor
    reconstruction: torch.Tensor
    point_error: torch.Tensor
    objective: float
    residual_bits: float
    assignment_bits: float
    ambiguity_bits: float
    structure_bits: float
    effective_slots: float
    expected_active_slots: float


def validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.num_slots <= 8:
        raise ValueError(f"--num-slots must be in [2, 8], got {args.num_slots}.")
    if args.d_model < 6:
        raise ValueError(f"--d-model must be >= 6, got {args.d_model}.")
    if args.trace_steps <= 0:
        raise ValueError(f"--trace-steps must be positive, got {args.trace_steps}.")
    if args.restarts <= 0:
        raise ValueError(f"--restarts must be positive, got {args.restarts}.")
    if args.trace_lr <= 0.0:
        raise ValueError(f"--trace-lr must be positive, got {args.trace_lr}.")
    if args.attention_scale <= 0.0:
        raise ValueError(f"--attention-scale must be positive, got {args.attention_scale}.")
    if args.observation_sigma <= 0.0:
        raise ValueError(f"--observation-sigma must be positive, got {args.observation_sigma}.")
    if args.encoding_span <= args.observation_sigma:
        raise ValueError("--encoding-span must exceed --observation-sigma.")
    if not 0.0 < args.concept_radius < args.encoding_span:
        raise ValueError("--concept-radius must be in (0, encoding_span).")
    if args.ambiguity_weight < 0.0:
        raise ValueError(f"--ambiguity-weight must be non-negative, got {args.ambiguity_weight}.")
    if args.stage3_old_points_per_group < 0:
        raise ValueError(
            "--stage3-old-points-per-group must be non-negative, got "
            f"{args.stage3_old_points_per_group}."
        )
    if args.summary_covariance not in {"full", "diagonal", "lowrank", "trace"}:
        raise ValueError(f"Unknown --summary-covariance {args.summary_covariance!r}.")
    if not 1 <= args.summary_rank <= args.d_model:
        raise ValueError(
            f"--summary-rank must be in [1, d_model], got {args.summary_rank}."
        )
    if args.power_iterations <= 0:
        raise ValueError(f"--power-iterations must be positive, got {args.power_iterations}.")
    for name in (
        "merge_points",
        "stable_points",
        "root_points",
        "obsolete_points",
        "branch_points",
        "novel_points",
        "noise_points",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}.")


def raw_moments(points: list[EvidencePoint], *, device: torch.device) -> EvidenceMoments:
    if not points:
        raise ValueError("Cannot construct evidence moments from zero points.")
    means = torch.tensor([point.vector for point in points], dtype=torch.float32, device=device)
    dimension = means.shape[1]
    return EvidenceMoments(
        means=means,
        covariances=torch.zeros(
            means.shape[0],
            dimension,
            dimension,
            dtype=means.dtype,
            device=device,
        ),
        weights=torch.ones(means.shape[0], dtype=means.dtype, device=device),
    )


def concatenate_moments(parts: list[EvidenceMoments]) -> EvidenceMoments:
    if not parts:
        raise ValueError("Cannot concatenate zero evidence-moment collections.")
    dimension = parts[0].means.shape[1]
    device = parts[0].means.device
    dtype = parts[0].means.dtype
    for part in parts:
        if part.means.ndim != 2 or part.means.shape[1] != dimension:
            raise ValueError("All evidence means must have the same feature dimension.")
        if part.covariances.shape != (part.means.shape[0], dimension, dimension):
            raise ValueError("Evidence covariance shape does not match means.")
        if part.weights.shape != (part.means.shape[0],):
            raise ValueError("Evidence weight shape does not match means.")
        if part.means.device != device or part.means.dtype != dtype:
            raise ValueError("All evidence moments must share device and dtype.")
    return EvidenceMoments(
        means=torch.cat([part.means for part in parts], dim=0),
        covariances=torch.cat([part.covariances for part in parts], dim=0),
        weights=torch.cat([part.weights for part in parts], dim=0),
    )


def validate_moments(evidence: EvidenceMoments) -> None:
    if evidence.means.ndim != 2 or evidence.means.shape[0] <= 0:
        raise ValueError(f"Evidence means must be non-empty [N, D], got {tuple(evidence.means.shape)}.")
    n, dimension = evidence.means.shape
    if evidence.covariances.shape != (n, dimension, dimension):
        raise ValueError(
            f"Evidence covariances must be {(n, dimension, dimension)}, "
            f"got {tuple(evidence.covariances.shape)}."
        )
    if evidence.weights.shape != (n,):
        raise ValueError(f"Evidence weights must be {(n,)}, got {tuple(evidence.weights.shape)}.")
    if torch.any(evidence.weights <= 0.0):
        raise ValueError("Every evidence item must have positive mass.")
    for name, value in (
        ("means", evidence.means),
        ("covariances", evidence.covariances),
        ("weights", evidence.weights),
    ):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Evidence {name} contains non-finite values.")


def functional_attention(
    means: torch.Tensor,
    centers: torch.Tensor,
    *,
    attention_scale: float,
) -> torch.Tensor:
    if means.ndim != 2 or centers.ndim != 2 or means.shape[1] != centers.shape[1]:
        raise ValueError(
            f"Attention expects means [N, D] and centers [S, D], got "
            f"{tuple(means.shape)} and {tuple(centers.shape)}."
        )
    squared_distance = ((means.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
    logits = -squared_distance / (2.0 * attention_scale**2)
    return torch.softmax(logits, dim=1)


def functional_objective(
    *,
    evidence: EvidenceMoments,
    centers: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    validate_moments(evidence)
    attention = functional_attention(
        evidence.means,
        centers,
        attention_scale=args.attention_scale,
    )
    reconstruction = attention @ centers
    covariance_energy = evidence.covariances.diagonal(dim1=1, dim2=2).sum(dim=1)
    squared_error = ((evidence.means - reconstruction) ** 2).sum(dim=1) + covariance_energy
    gaussian_bits = squared_error / (2.0 * args.observation_sigma**2 * math.log(2.0))
    outlier_bits = evidence.means.shape[1] * math.log2(
        (2.0 * args.encoding_span) / args.observation_sigma
    )
    point_residual_bits = outlier_bits * torch.log1p(gaussian_bits / outlier_bits)
    residual_bits = (evidence.weights * point_residual_bits).sum()

    total_mass = evidence.weights.sum()
    weighted_attention = evidence.weights.unsqueeze(1) * attention
    usage = weighted_attention.sum(dim=0) / total_mass
    eps = torch.finfo(evidence.means.dtype).eps
    assignment_bits = -(
        weighted_attention * torch.log2(usage.clamp_min(eps)).unsqueeze(0)
    ).sum()
    ambiguity_bits = -(
        evidence.weights.unsqueeze(1)
        * attention
        * torch.log2(attention.clamp_min(eps))
    ).sum()
    expected_active = (1.0 - (1.0 - usage.clamp(max=1.0 - eps)) ** total_mass).sum()
    center_bits = evidence.means.shape[1] * math.log2(
        (2.0 * args.encoding_span) / args.observation_sigma
    )
    structure_bits = center_bits * expected_active
    objective = (
        residual_bits
        + assignment_bits
        + args.ambiguity_weight * ambiguity_bits
        + structure_bits
    ) / total_mass
    return objective, {
        "attention": attention,
        "reconstruction": reconstruction,
        "squared_error": squared_error,
        "usage": usage,
        "residual_bits": residual_bits,
        "assignment_bits": assignment_bits,
        "ambiguity_bits": ambiguity_bits,
        "structure_bits": structure_bits,
        "expected_active": expected_active,
    }


def density_weighted_centers(
    evidence: EvidenceMoments,
    *,
    num_slots: int,
    first_index: int,
    observation_sigma: float,
) -> torch.Tensor:
    validate_moments(evidence)
    if evidence.means.shape[0] < num_slots:
        raise ValueError(
            f"Need at least {num_slots} evidence items to initialize slots, got {evidence.means.shape[0]}."
        )
    if not 0 <= first_index < evidence.means.shape[0]:
        raise ValueError("Trace initialization index is outside the evidence collection.")
    pairwise_distance = torch.cdist(evidence.means, evidence.means).square()
    kernel = torch.exp(-pairwise_distance / (2.0 * observation_sigma**2))
    density = (kernel * evidence.weights.unsqueeze(0)).sum(dim=1)
    selected = [first_index]
    min_distance = pairwise_distance[first_index].clone()
    for _ in range(1, num_slots):
        score = min_distance * density
        score[selected] = -torch.inf
        index = int(torch.argmax(score).item())
        selected.append(index)
        min_distance = torch.minimum(min_distance, pairwise_distance[index])
    return evidence.means[selected].clone()


def initialize_centers(
    evidence: EvidenceMoments,
    *,
    args: argparse.Namespace,
    restart: int,
    stage: int,
    previous_centers: torch.Tensor | None,
) -> torch.Tensor:
    if restart == 0 and previous_centers is not None:
        expected = (args.num_slots, evidence.means.shape[1])
        if previous_centers.shape != expected:
            raise ValueError(f"Previous centers must be {expected}, got {tuple(previous_centers.shape)}.")
        return previous_centers.detach().clone()
    rng = random.Random(args.seed + stage * args.restarts + restart)
    first_index = rng.randrange(evidence.means.shape[0])
    return density_weighted_centers(
        evidence,
        num_slots=args.num_slots,
        first_index=first_index,
        observation_sigma=args.observation_sigma,
    )


def fit_functional_trace_field(
    evidence: EvidenceMoments,
    *,
    args: argparse.Namespace,
    stage: int,
    previous_centers: torch.Tensor | None,
) -> FunctionalTraceSolution:
    validate_moments(evidence)
    best: FunctionalTraceSolution | None = None
    for restart in range(args.restarts):
        centers = initialize_centers(
            evidence,
            args=args,
            restart=restart,
            stage=stage,
            previous_centers=previous_centers,
        ).requires_grad_(True)
        optimizer = torch.optim.Adam([centers], lr=args.trace_lr)
        for step in range(1, args.trace_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            objective, parts = functional_objective(evidence=evidence, centers=centers, args=args)
            if not torch.isfinite(objective):
                diagnostics = {
                    name: bool(torch.isfinite(value).all().detach().cpu().item())
                    for name, value in parts.items()
                }
                raise FloatingPointError(
                    f"Non-finite functional trace objective at stage={stage}, restart={restart}, "
                    f"step={step}: {diagnostics}"
                )
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                centers.clamp_(-args.encoding_span, args.encoding_span)

        with torch.no_grad():
            objective, parts = functional_objective(evidence=evidence, centers=centers, args=args)
            usage = parts["usage"]
            entropy = -(usage * torch.log(usage.clamp_min(torch.finfo(usage.dtype).eps))).sum()
            solution = FunctionalTraceSolution(
                centers=centers.detach().clone(),
                attention=parts["attention"].detach().clone(),
                usage=usage.detach().clone(),
                reconstruction=parts["reconstruction"].detach().clone(),
                point_error=parts["squared_error"].detach().clone(),
                objective=float(objective.item()),
                residual_bits=float(parts["residual_bits"].item()),
                assignment_bits=float(parts["assignment_bits"].item()),
                ambiguity_bits=float(parts["ambiguity_bits"].item()),
                structure_bits=float(parts["structure_bits"].item()),
                effective_slots=float(torch.exp(entropy).item()),
                expected_active_slots=float(parts["expected_active"].item()),
            )
        if best is None or solution.objective < best.objective:
            best = solution
    if best is None:
        raise RuntimeError(f"No functional trace solution was produced at stage {stage}.")
    return best


def compress_to_trace_moments(
    evidence: EvidenceMoments,
    solution: FunctionalTraceSolution,
) -> EvidenceMoments:
    validate_moments(evidence)
    if solution.attention.shape != (evidence.means.shape[0], solution.centers.shape[0]):
        raise ValueError("Solution attention does not match evidence and slot counts.")
    responsibilities = evidence.weights.unsqueeze(1) * solution.attention
    masses = responsibilities.sum(dim=0)
    eps = torch.finfo(evidence.means.dtype).eps
    if torch.any(masses <= eps):
        raise RuntimeError(f"Cannot compress a trace with non-positive mass: {masses.tolist()}.")
    means = responsibilities.transpose(0, 1) @ evidence.means / masses.unsqueeze(1)
    input_second = evidence.covariances + evidence.means.unsqueeze(2) * evidence.means.unsqueeze(1)
    second = torch.einsum("ns,ndk->sdk", responsibilities, input_second) / masses[:, None, None]
    covariances = second - means.unsqueeze(2) * means.unsqueeze(1)
    covariances = 0.5 * (covariances + covariances.transpose(1, 2))
    covariance_diagonal = covariances.diagonal(dim1=1, dim2=2)
    if torch.any(covariance_diagonal < -1e-4):
        raise FloatingPointError(
            "Compressed covariance has a negative variance: "
            f"min={float(covariance_diagonal.min().item())}."
        )
    corrected_diagonal = covariance_diagonal.clamp_min(0.0)
    covariances = (
        covariances
        - torch.diag_embed(covariance_diagonal)
        + torch.diag_embed(corrected_diagonal)
    )
    return EvidenceMoments(means=means, covariances=covariances, weights=masses)


def orthonormalize_columns(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[1] <= 0:
        raise ValueError(f"Expected a non-empty matrix [D, rank], got {tuple(matrix.shape)}.")
    eps = torch.finfo(matrix.dtype).eps
    columns: list[torch.Tensor] = []
    for index in range(matrix.shape[1]):
        vector = matrix[:, index]
        for previous in columns:
            vector = vector - torch.dot(previous, vector) * previous
        norm = vector.norm()
        if not torch.isfinite(norm) or norm <= eps:
            raise FloatingPointError(
                f"Low-rank covariance basis became singular at column {index}."
            )
        columns.append(vector / norm)
    return torch.stack(columns, dim=1)


def lowrank_covariance_factors(
    covariances: torch.Tensor,
    *,
    rank: int,
    power_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if covariances.ndim != 3 or covariances.shape[1] != covariances.shape[2]:
        raise ValueError(f"Covariances must be [slots, D, D], got {tuple(covariances.shape)}.")
    slots, dimension, _ = covariances.shape
    if not 1 <= rank <= dimension:
        raise ValueError(f"rank must be in [1, {dimension}], got {rank}.")
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    initial = torch.randn(slots, dimension, rank, generator=cpu_generator, dtype=torch.float32)
    initial = initial.to(device=covariances.device, dtype=covariances.dtype)
    factors: list[torch.Tensor] = []
    diagonals: list[torch.Tensor] = []
    for slot in range(slots):
        basis = orthonormalize_columns(initial[slot])
        covariance = covariances[slot]
        for _ in range(power_iterations):
            basis = orthonormalize_columns(covariance @ basis)
        variances = torch.diagonal(basis.transpose(0, 1) @ covariance @ basis).clamp_min(0.0)
        factor = basis * torch.sqrt(variances).unsqueeze(0)
        residual_diagonal = (
            torch.diagonal(covariance) - (factor.square()).sum(dim=1)
        ).clamp_min(0.0)
        factors.append(factor)
        diagonals.append(residual_diagonal)
    return torch.stack(diagonals, dim=0), torch.stack(factors, dim=0)


def encode_trace_summary(
    moments: EvidenceMoments,
    *,
    mode: str,
    rank: int,
    power_iterations: int,
    seed: int,
) -> TraceSummary:
    validate_moments(moments)
    if mode == "full":
        summary = TraceSummary(
            mode=mode,
            means=moments.means,
            weights=moments.weights,
            full_covariances=moments.covariances,
            diagonal=None,
            factors=None,
            variance_trace=None,
        )
    elif mode == "diagonal":
        summary = TraceSummary(
            mode=mode,
            means=moments.means,
            weights=moments.weights,
            full_covariances=None,
            diagonal=moments.covariances.diagonal(dim1=1, dim2=2).clamp_min(0.0),
            factors=None,
            variance_trace=None,
        )
    elif mode == "lowrank":
        diagonal, factors = lowrank_covariance_factors(
            moments.covariances,
            rank=rank,
            power_iterations=power_iterations,
            seed=seed,
        )
        summary = TraceSummary(
            mode=mode,
            means=moments.means,
            weights=moments.weights,
            full_covariances=None,
            diagonal=diagonal,
            factors=factors,
            variance_trace=None,
        )
    elif mode == "trace":
        summary = TraceSummary(
            mode=mode,
            means=moments.means,
            weights=moments.weights,
            full_covariances=None,
            diagonal=None,
            factors=None,
            variance_trace=moments.covariances.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(0.0),
        )
    else:
        raise ValueError(f"Unknown trace-summary covariance mode {mode!r}.")
    summary.validate()
    return summary


def align_centers(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    if previous.shape != current.shape:
        raise ValueError("Cannot align center sets with different shapes.")
    num_slots = previous.shape[0]
    cost = torch.cdist(previous, current).square().detach().cpu()
    best_order: tuple[int, ...] | None = None
    best_cost: float | None = None
    for order in itertools.permutations(range(num_slots)):
        value = sum(float(cost[index, order[index]]) for index in range(num_slots))
        if best_cost is None or value < best_cost:
            best_cost = value
            best_order = order
    if best_order is None:
        raise RuntimeError("Center alignment did not produce an ordering.")
    order_tensor = torch.tensor(best_order, device=current.device, dtype=torch.long)
    return current[order_tensor]


def evaluate_centers(
    points: list[EvidencePoint],
    centers: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FunctionalTraceSolution, dict[str, dict[str, Any]]]:
    evidence = raw_moments(points, device=device)
    with torch.no_grad():
        objective, parts = functional_objective(evidence=evidence, centers=centers, args=args)
        usage = parts["usage"]
        entropy = -(usage * torch.log(usage.clamp_min(torch.finfo(usage.dtype).eps))).sum()
        solution = FunctionalTraceSolution(
            centers=centers.detach().clone(),
            attention=parts["attention"].detach().clone(),
            usage=usage.detach().clone(),
            reconstruction=parts["reconstruction"].detach().clone(),
            point_error=parts["squared_error"].detach().clone(),
            objective=float(objective.item()),
            residual_bits=float(parts["residual_bits"].item()),
            assignment_bits=float(parts["assignment_bits"].item()),
            ambiguity_bits=float(parts["ambiguity_bits"].item()),
            structure_bits=float(parts["structure_bits"].item()),
            effective_slots=float(torch.exp(entropy).item()),
            expected_active_slots=float(parts["expected_active"].item()),
        )
    groups: dict[str, dict[str, Any]] = {}
    for group in sorted({point.hidden_group for point in points}):
        indices = [index for index, point in enumerate(points) if point.hidden_group == group]
        index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_attention = solution.attention[index_tensor]
        mean_attention = group_attention.mean(dim=0)
        groups[group] = {
            "count": len(indices),
            "dominant_slot": int(mean_attention.argmax().item()),
            "dominant_share": float(mean_attention.max().item()),
            "mean_attention": [float(value) for value in mean_attention.detach().cpu()],
            "mean_attention_l1_dispersion": float(
                (group_attention - mean_attention).abs().sum(dim=1).mean().item()
            ),
            "mean_squared_error": float(solution.point_error[index_tensor].mean().item()),
        }
    return solution, groups


def centered_kernel_alignment(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must contain the same number of observations.")
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    numerator = (left.transpose(0, 1) @ right).square().sum()
    denominator = torch.sqrt(
        (left.transpose(0, 1) @ left).square().sum()
        * (right.transpose(0, 1) @ right).square().sum()
    )
    if denominator <= 0.0:
        raise FloatingPointError("Cannot compute CKA with zero representation norm.")
    return float((numerator / denominator).item())


def structure_checks(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merge_a = torch.tensor(groups["merge_a"]["mean_attention"])
    merge_b = torch.tensor(groups["merge_b"]["mean_attention"])
    branch_up = torch.tensor(groups["branch_up"]["mean_attention"])
    branch_down = torch.tensor(groups["branch_down"]["mean_attention"])
    branch_root = torch.tensor(groups["branch_root"]["mean_attention"])
    merge_distance = float((merge_a - merge_b).abs().sum().item())
    merge_scale = (
        groups["merge_a"]["mean_attention_l1_dispersion"]
        + groups["merge_b"]["mean_attention_l1_dispersion"]
    )
    branch_distance = float((branch_up - branch_down).abs().sum().item())
    branch_scale = (
        groups["branch_up"]["mean_attention_l1_dispersion"]
        + groups["branch_down"]["mean_attention_l1_dispersion"]
    )
    composition = 0.5 * (branch_up + branch_down)
    root_to_composition = float((branch_root - composition).abs().sum().item())
    root_to_single = min(
        float((branch_root - branch_up).abs().sum().item()),
        float((branch_root - branch_down).abs().sum().item()),
    )
    noise_slot = groups["noise"]["dominant_slot"]
    non_noise_slots = {
        row["dominant_slot"]
        for group, row in groups.items()
        if group != "noise"
    }
    return {
        "duplicate_code_l1_distance": merge_distance,
        "duplicate_within_group_l1_scale": merge_scale,
        "duplicate_sources_merge": merge_distance <= merge_scale,
        "branch_code_l1_distance": branch_distance,
        "branch_within_group_l1_scale": branch_scale,
        "context_branches_remain_distinct": branch_distance > branch_scale,
        "root_to_branch_composition_l1": root_to_composition,
        "root_to_nearest_single_branch_l1": root_to_single,
        "shared_branch_root_is_compositional": root_to_composition < root_to_single,
        "novel_replaces_obsolete": groups["novel"]["mean_squared_error"]
        < groups["obsolete"]["mean_squared_error"],
        "stable_survives": groups["stable"]["mean_squared_error"]
        < groups["obsolete"]["mean_squared_error"],
        "noise_has_no_exclusive_slot": noise_slot in non_noise_slots,
    }


def run_condition(
    *,
    mode: str,
    stages: list[list[EvidencePoint]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], TraceSummary | None]:
    if mode not in {"full_history", "recurrent_summary", "current_only"}:
        raise ValueError(f"Unknown condition {mode!r}.")
    accumulated: list[EvidencePoint] = []
    previous_centers: torch.Tensor | None = None
    summary: TraceSummary | None = None
    reports: list[dict[str, Any]] = []
    center_history: list[torch.Tensor] = []

    for stage_index, stage_points in enumerate(stages, start=1):
        accumulated.extend(stage_points)
        current = raw_moments(stage_points, device=device)
        if mode == "full_history":
            training_evidence = raw_moments(accumulated, device=device)
            persistent_items = len(accumulated)
        elif mode == "recurrent_summary":
            training_evidence = (
                current
                if summary is None
                else concatenate_moments([summary.to_moments(), current])
            )
            persistent_items = 0 if summary is None else summary.means.shape[0]
        else:
            training_evidence = current
            persistent_items = 0

        training_solution = fit_functional_trace_field(
            training_evidence,
            args=args,
            stage=stage_index,
            previous_centers=previous_centers,
        )
        centers = training_solution.centers
        if previous_centers is not None:
            centers = align_centers(previous_centers, centers)
        previous_centers = centers
        center_history.append(centers.detach().clone())
        evaluation, groups = evaluate_centers(
            accumulated,
            centers,
            args=args,
            device=device,
        )
        if mode == "recurrent_summary":
            with torch.no_grad():
                aligned_objective, aligned_parts = functional_objective(
                    evidence=training_evidence,
                    centers=centers,
                    args=args,
                )
                aligned_usage = aligned_parts["usage"]
                aligned_entropy = -(
                    aligned_usage
                    * torch.log(aligned_usage.clamp_min(torch.finfo(aligned_usage.dtype).eps))
                ).sum()
                summary_solution = FunctionalTraceSolution(
                    centers=centers,
                    attention=aligned_parts["attention"],
                    usage=aligned_usage,
                    reconstruction=aligned_parts["reconstruction"],
                    point_error=aligned_parts["squared_error"],
                    objective=float(aligned_objective.item()),
                    residual_bits=float(aligned_parts["residual_bits"].item()),
                    assignment_bits=float(aligned_parts["assignment_bits"].item()),
                    ambiguity_bits=float(aligned_parts["ambiguity_bits"].item()),
                    structure_bits=float(aligned_parts["structure_bits"].item()),
                    effective_slots=float(torch.exp(aligned_entropy).item()),
                    expected_active_slots=float(aligned_parts["expected_active"].item()),
                )
            compressed_moments = compress_to_trace_moments(training_evidence, summary_solution)
            summary = encode_trace_summary(
                compressed_moments,
                mode=args.summary_covariance,
                rank=args.summary_rank,
                power_iterations=args.power_iterations,
                seed=args.seed + stage_index,
            )

        reports.append(
            {
                "stage": stage_index,
                "training_items": int(training_evidence.means.shape[0]),
                "persistent_items_before_stage": int(persistent_items),
                "persistent_items_after_stage": (
                    int(summary.means.shape[0]) if mode == "recurrent_summary" and summary is not None
                    else (len(accumulated) if mode == "full_history" else 0)
                ),
                "persistent_scalars_after_stage": (
                    (
                        summary.stored_scalars()
                        if mode == "recurrent_summary" and summary is not None
                        else (
                            len(accumulated) * args.d_model
                            if mode == "full_history"
                            else 0
                        )
                    )
                ),
                "accumulated_raw_points": len(accumulated),
                "evaluation_objective": evaluation.objective,
                "evaluation_mean_squared_error": float(evaluation.point_error.mean().item()),
                "effective_slots": evaluation.effective_slots,
                "usage": [float(value) for value in evaluation.usage.detach().cpu()],
                "centers": centers.detach().cpu().tolist(),
                "groups": groups,
            }
        )
    return reports, center_history, summary


def plot_final_geometry(
    *,
    points: list[EvidencePoint],
    centers_by_mode: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    point_tensor = torch.tensor([point.vector for point in points], dtype=torch.float32)
    mean = point_tensor.mean(dim=0)
    _u, _s, vh = torch.linalg.svd(point_tensor - mean, full_matrices=False)
    basis = vh[:2].transpose(0, 1)

    def project(values: torch.Tensor) -> torch.Tensor:
        return (values - mean) @ basis

    modes = ["full_history", "recurrent_summary", "current_only"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    for axis, mode in zip(axes, modes):
        for group in sorted({point.hidden_group for point in points}):
            values = torch.tensor(
                [point.vector for point in points if point.hidden_group == group],
                dtype=torch.float32,
            )
            projected = project(values)
            axis.scatter(
                projected[:, 0],
                projected[:, 1],
                s=18,
                alpha=0.5,
                color=GROUP_COLORS[group],
                label=group,
            )
        centers = project(centers_by_mode[mode].detach().cpu())
        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="X",
            s=220,
            color="#111111",
            edgecolor="white",
            linewidth=1.0,
            label="trace key",
        )
        axis.set_title(mode.replace("_", " "))
        axis.set_xlabel("representation PC1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("representation PC2")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Final bounded trace geometry")
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_errors(
    *,
    final_reports: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    groups = sorted(final_reports["full_history"]["groups"])
    modes = ["full_history", "recurrent_summary", "current_only"]
    x = torch.arange(len(groups), dtype=torch.float32).numpy()
    width = 0.25
    fig, axis = plt.subplots(figsize=(12.5, 5.2))
    for offset, mode in enumerate(modes):
        values = [final_reports[mode]["groups"][group]["mean_squared_error"] for group in groups]
        axis.bar(x + (offset - 1) * width, values, width=width, label=mode.replace("_", " "))
    axis.set_xticks(x, groups, rotation=30, ha="right")
    axis.set_ylabel("mean reconstruction error")
    axis.set_title("Final retained and released evidence")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_memory_and_similarity(
    *,
    reports_by_mode: dict[str, list[dict[str, Any]]],
    cka_by_stage: list[float],
    output_path: Path,
) -> None:
    stages = [report["stage"] for report in reports_by_mode["full_history"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for mode in ["full_history", "recurrent_summary", "current_only"]:
        axes[0].plot(
            stages,
            [report["persistent_scalars_after_stage"] for report in reports_by_mode[mode]],
            marker="o",
            linewidth=2,
            label=mode.replace("_", " "),
        )
    axes[0].set_xlabel("stream stage")
    axes[0].set_ylabel("persistent stored scalars")
    axes[0].set_title("Persistent memory")
    axes[0].set_xticks(stages)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(stages, cka_by_stage, marker="o", linewidth=2, color="#d62728")
    axes[1].set_ylim(max(0.0, min(cka_by_stage) - 0.01), 1.001)
    axes[1].set_xlabel("stream stage")
    axes[1].set_ylabel("attention-code CKA")
    axes[1].set_title("Recurrent summary vs full history")
    axes[1].set_xticks(stages)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def prepare_stream(args: argparse.Namespace) -> list[list[EvidencePoint]]:
    stages = build_stream(args)
    stage3_old_groups = {"merge_b", "stable", "branch_up", "branch_down"}
    retained_stage3: list[EvidencePoint] = [
        point for point in stages[2] if point.hidden_group not in stage3_old_groups
    ]
    for group in sorted(stage3_old_groups):
        candidates = [point for point in stages[2] if point.hidden_group == group]
        if args.stage3_old_points_per_group > len(candidates):
            raise ValueError(
                f"Requested {args.stage3_old_points_per_group} stage-3 points for {group}, "
                f"but only {len(candidates)} were generated."
            )
        retained_stage3.extend(candidates[: args.stage3_old_points_per_group])
    stages[2] = retained_stage3
    return stages


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    stages = prepare_stream(args)
    modes = ["full_history", "recurrent_summary", "current_only"]
    reports_by_mode: dict[str, list[dict[str, Any]]] = {}
    centers_by_mode: dict[str, list[torch.Tensor]] = {}

    print("TINY RECURRENT TRACE FIELD")
    print("=" * 132)
    print(
        f"device={device} slots={args.num_slots} d_model={args.d_model} "
        f"steps={args.trace_steps} restarts={args.restarts}"
    )
    for mode in modes:
        reports, centers, _summary = run_condition(
            mode=mode,
            stages=stages,
            args=args,
            device=device,
        )
        reports_by_mode[mode] = reports
        centers_by_mode[mode] = centers

    accumulated: list[EvidencePoint] = []
    cka_by_stage: list[float] = []
    for stage_index, stage_points in enumerate(stages):
        accumulated.extend(stage_points)
        full_eval, _ = evaluate_centers(
            accumulated,
            centers_by_mode["full_history"][stage_index],
            args=args,
            device=device,
        )
        recurrent_eval, _ = evaluate_centers(
            accumulated,
            centers_by_mode["recurrent_summary"][stage_index],
            args=args,
            device=device,
        )
        cka_by_stage.append(centered_kernel_alignment(full_eval.attention, recurrent_eval.attention))

    print("\nFINAL METHOD SUMMARY")
    print("-" * 132)
    print(
        f"{'method':>20} {'items':>7} {'scalars':>9} {'mse':>10} "
        f"{'objective':>11} {'effSlots':>10} {'codeCKA':>10}"
    )
    final_reports: dict[str, dict[str, Any]] = {}
    checks: dict[str, dict[str, Any]] = {}
    for mode in modes:
        final = reports_by_mode[mode][-1]
        final_reports[mode] = final
        checks[mode] = structure_checks(final["groups"])
        code_cka = 1.0 if mode == "full_history" else (
            cka_by_stage[-1] if mode == "recurrent_summary" else float("nan")
        )
        print(
            f"{mode:>20} {final['persistent_items_after_stage']:7d} "
            f"{final['persistent_scalars_after_stage']:9d} "
            f"{final['evaluation_mean_squared_error']:10.4f} "
            f"{final['evaluation_objective']:11.4f} {final['effective_slots']:10.3f} "
            f"{code_cka:10.4f}"
        )

    print("\nRECURRENT SUMMARY STRUCTURE CHECKS")
    print("-" * 132)
    for name, value in checks["recurrent_summary"].items():
        print(f"{name:>48} = {value}")

    retained_groups = ("merge_a", "merge_b", "stable", "branch_root", "branch_up", "branch_down")
    retained_error = {
        mode: sum(
            final_reports[mode]["groups"][group]["mean_squared_error"]
            for group in retained_groups
        )
        / len(retained_groups)
        for mode in modes
    }
    recurrent_ratio = retained_error["recurrent_summary"] / retained_error["full_history"]
    current_ratio = retained_error["current_only"] / retained_error["full_history"]
    full_scalars = final_reports["full_history"]["persistent_scalars_after_stage"]
    recurrent_scalars = final_reports["recurrent_summary"]["persistent_scalars_after_stage"]
    comparison = {
        "retained_old_error": retained_error,
        "recurrent_old_error_ratio_vs_full": recurrent_ratio,
        "current_only_old_error_ratio_vs_full": current_ratio,
        "recurrent_vs_full_attention_cka": cka_by_stage[-1],
        "full_history_persistent_scalars": full_scalars,
        "recurrent_persistent_scalars": recurrent_scalars,
        "persistent_scalar_compression_ratio": full_scalars / recurrent_scalars,
        "recurrent_reduces_old_error_vs_current_only": retained_error["recurrent_summary"]
        < retained_error["current_only"],
    }
    print("\nMEMORY FIDELITY COMPARISON")
    print("-" * 132)
    for name, value in comparison.items():
        print(f"{name:>48} = {value}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = args.output_dir / "recurrent_trace_geometry.png"
    errors_path = args.output_dir / "recurrent_trace_group_errors.png"
    memory_path = args.output_dir / "recurrent_trace_memory.png"
    output_json = args.output_dir / "recurrent_trace_field.json"
    all_points = [point for stage in stages for point in stage]
    plot_final_geometry(
        points=all_points,
        centers_by_mode={mode: centers_by_mode[mode][-1] for mode in modes},
        output_path=geometry_path,
    )
    plot_group_errors(final_reports=final_reports, output_path=errors_path)
    plot_memory_and_similarity(
        reports_by_mode=reports_by_mode,
        cka_by_stage=cka_by_stage,
        output_path=memory_path,
    )

    output = {
        "question": (
            "Can five recurrent mass/mean/covariance summaries replace full raw history while preserving "
            "autonomous trace merging, branching, composition, novelty, and forgetting?"
        ),
        "scope": (
            "Attention is a function of representations and trace keys; no per-observation assignment state "
            "persists. This still operates in synthetic representation space and does not update model weights."
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stream": [[asdict(point) for point in stage] for stage in stages],
        "reports": reports_by_mode,
        "checks": checks,
        "comparison": comparison,
        "recurrent_vs_full_attention_cka_by_stage": cka_by_stage,
        "plots": {
            "geometry": str(geometry_path),
            "group_errors": str(errors_path),
            "memory": str(memory_path),
        },
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_plots={geometry_path},{errors_path},{memory_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/analysis/gco-tiny-recurrent-trace-field-seed0"),
    )
    parser.add_argument("--num-slots", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--trace-steps", type=int, default=700)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--trace-lr", type=float, default=0.03)
    parser.add_argument("--attention-scale", type=float, default=1.0)
    parser.add_argument("--ambiguity-weight", type=float, default=0.2)
    parser.add_argument(
        "--summary-covariance",
        choices=["full", "diagonal", "lowrank", "trace"],
        default="full",
    )
    parser.add_argument("--summary-rank", type=int, default=2)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument("--observation-sigma", type=float, default=0.24)
    parser.add_argument("--encoding-span", type=float, default=5.0)
    parser.add_argument("--concept-radius", type=float, default=3.2)
    parser.add_argument("--merge-points", type=int, default=24)
    parser.add_argument("--stable-points", type=int, default=30)
    parser.add_argument("--root-points", type=int, default=20)
    parser.add_argument("--obsolete-points", type=int, default=6)
    parser.add_argument("--branch-points", type=int, default=24)
    parser.add_argument("--novel-points", type=int, default=30)
    parser.add_argument("--noise-points", type=int, default=8)
    parser.add_argument("--stage3-old-points-per-group", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
