#!/usr/bin/env python3
"""Orthogonal-room write test for GCO.

This experiment isolates the first write question:

    Can a new key-value behavior be written into free geometric room while
    preserving old protected key-value behaviors?

The model is one linear associative layer:

    y = W k

Old protected behaviors are columns of K_old -> V_old. A new behavior is
k_new -> v_new. We measure how much of k_new lies outside the protected old-key
subspace, then compare three writes:

    raw           unconstrained rank-one write
    free          write using only the protected-nullspace component of k_new
    protected_ls  least-squares write penalized for changing old keys

No replay controller, no symbolic memory, no task labels. This is only the
geometry of writing.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True)
class WriteResult:
    method: str
    feasible: bool
    reason: str
    free_room_ratio: float
    protected_overlap_ratio: float
    new_mse_before: float
    new_mse_after: float | None
    new_error_reduction: float | None
    old_mse_before: float
    old_mse_after: float | None
    old_damage: float | None
    update_norm: float | None
    budget_scale: float | None
    update_to_gain_ratio: float | None
    max_old_key_error_after: float | None


@dataclass(frozen=True)
class SweepCase:
    seed: int
    key_dim: int
    value_dim: int
    old_count: int
    requested_overlap: float
    actual_free_room_ratio: float
    actual_protected_overlap_ratio: float
    condition_number_protected_system: float
    results: list[WriteResult]


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float in comma-separated list.")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer in comma-separated list.")
    return values


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
    raise ValueError(f"Unknown device: {name}")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        if device.type == "mps":
            raise RuntimeError("MPS does not support float64 for this experiment.")
        return torch.float64
    raise ValueError(f"Unknown dtype: {name}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def randn(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=dtype)


def normalize_columns(x: torch.Tensor, eps: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(x, dim=0, keepdim=True)
    if bool((norms <= eps).any().detach().cpu()):
        raise RuntimeError("Cannot normalize a zero-length column.")
    return x / norms


def orthonormal_columns(rows: int, cols: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if cols > rows:
        raise ValueError(f"Cannot build {cols} orthonormal columns in {rows} dimensions.")
    q, r = torch.linalg.qr(randn((rows, cols), device=device, dtype=dtype), mode="reduced")
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def projection_from_basis(basis: torch.Tensor) -> torch.Tensor:
    return basis @ basis.T


def mse_columns(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=0)


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def make_case(
    *,
    key_dim: int,
    value_dim: int,
    old_count: int,
    requested_overlap: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    if key_dim <= 1:
        raise ValueError("--key-dim must be greater than 1.")
    if value_dim <= 0:
        raise ValueError("--value-dim must be positive.")
    if old_count <= 0:
        raise ValueError("--old-count values must be positive.")
    if old_count > key_dim:
        raise ValueError("--old-count must be <= --key-dim for this first exact-protection test.")
    if not (0.0 <= requested_overlap <= 1.0):
        raise ValueError("--overlaps values must be in [0, 1].")

    set_seed(seed)
    old_basis = orthonormal_columns(key_dim, old_count, device=device, dtype=dtype)
    p_old = projection_from_basis(old_basis)
    p_free = torch.eye(key_dim, device=device, dtype=dtype) - p_old

    old_keys = old_basis
    old_values = randn((value_dim, old_count), device=device, dtype=dtype) / math.sqrt(value_dim)

    free_dim = key_dim - old_count
    if requested_overlap < 1.0 and free_dim <= 0:
        raise ValueError("Requested overlap < 1.0 but protected subspace has no free orthogonal complement.")

    old_component = old_basis @ normalize_columns(randn((old_count, 1), device=device, dtype=dtype), eps)
    if free_dim > 0:
        free_basis = orthonormal_columns(key_dim, free_dim, device=device, dtype=dtype)
        free_basis = p_free @ free_basis
        free_basis, _ = torch.linalg.qr(free_basis, mode="reduced")
        free_component = free_basis[:, :1]
    else:
        free_component = torch.zeros((key_dim, 1), device=device, dtype=dtype)

    k_new = math.sqrt(requested_overlap) * old_component
    if requested_overlap < 1.0:
        k_new = k_new + math.sqrt(1.0 - requested_overlap) * free_component
    k_new = normalize_columns(k_new, eps)
    v_new = randn((value_dim, 1), device=device, dtype=dtype) / math.sqrt(value_dim)

    actual_protected = scalar(torch.linalg.vector_norm(p_old @ k_new) ** 2 / (torch.linalg.vector_norm(k_new) ** 2 + eps))
    actual_free = scalar(torch.linalg.vector_norm(p_free @ k_new) ** 2 / (torch.linalg.vector_norm(k_new) ** 2 + eps))
    return old_keys, old_values, k_new, v_new, p_old, p_free, actual_protected, actual_free


def base_weight(old_keys: torch.Tensor, old_values: torch.Tensor) -> torch.Tensor:
    gram = old_keys.T @ old_keys
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    if not bool(torch.allclose(gram, identity, atol=1e-4, rtol=1e-4)):
        raise RuntimeError("Old keys are expected to be orthonormal in this first experiment.")
    return old_values @ old_keys.T


def evaluate_write(
    *,
    method: str,
    delta_w: torch.Tensor | None,
    infeasible_reason: str,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    actual_free: float,
    actual_protected: float,
) -> WriteResult:
    old_mse_before = scalar(mse_columns(w_base @ old_keys, old_values).mean())
    new_mse_before = scalar(mse_columns(w_base @ k_new, v_new).mean())

    if delta_w is None:
        return WriteResult(
            method=method,
            feasible=False,
            reason=infeasible_reason,
            free_room_ratio=actual_free,
            protected_overlap_ratio=actual_protected,
            new_mse_before=new_mse_before,
            new_mse_after=None,
            new_error_reduction=None,
            old_mse_before=old_mse_before,
            old_mse_after=None,
            old_damage=None,
            update_norm=None,
            budget_scale=None,
            update_to_gain_ratio=None,
            max_old_key_error_after=None,
        )

    w_after = w_base + delta_w
    old_errors_after = mse_columns(w_after @ old_keys, old_values)
    old_mse_after = scalar(old_errors_after.mean())
    new_mse_after = scalar(mse_columns(w_after @ k_new, v_new).mean())
    new_error_reduction = new_mse_before - new_mse_after
    old_damage = old_mse_after - old_mse_before
    update_norm = scalar(torch.linalg.matrix_norm(delta_w))
    if new_error_reduction <= 0:
        ratio = math.inf
    else:
        ratio = update_norm / (new_error_reduction + 1e-12)

    return WriteResult(
        method=method,
        feasible=True,
        reason="ok",
        free_room_ratio=actual_free,
        protected_overlap_ratio=actual_protected,
        new_mse_before=new_mse_before,
        new_mse_after=new_mse_after,
        new_error_reduction=new_error_reduction,
        old_mse_before=old_mse_before,
        old_mse_after=old_mse_after,
        old_damage=old_damage,
        update_norm=update_norm,
        budget_scale=None,
        update_to_gain_ratio=ratio,
        max_old_key_error_after=scalar(old_errors_after.max()),
    )


def apply_budget(delta_w: torch.Tensor, max_update_norm: float) -> tuple[torch.Tensor, float]:
    if max_update_norm <= 0:
        raise ValueError("--max-update-norm must be positive when provided.")
    norm = scalar(torch.linalg.matrix_norm(delta_w))
    if norm <= max_update_norm:
        return delta_w, 1.0
    scale = max_update_norm / (norm + 1e-12)
    return delta_w * scale, scale


def evaluate_budgeted_write(
    *,
    method: str,
    delta_w: torch.Tensor | None,
    infeasible_reason: str,
    max_update_norm: float | None,
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    old_values: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    actual_free: float,
    actual_protected: float,
) -> WriteResult:
    if delta_w is None or max_update_norm is None:
        return evaluate_write(
            method=method,
            delta_w=delta_w,
            infeasible_reason=infeasible_reason,
            w_base=w_base,
            old_keys=old_keys,
            old_values=old_values,
            k_new=k_new,
            v_new=v_new,
            actual_free=actual_free,
            actual_protected=actual_protected,
        )

    budgeted_delta, scale = apply_budget(delta_w, max_update_norm)
    result = evaluate_write(
        method=method,
        delta_w=budgeted_delta,
        infeasible_reason=infeasible_reason,
        w_base=w_base,
        old_keys=old_keys,
        old_values=old_values,
        k_new=k_new,
        v_new=v_new,
        actual_free=actual_free,
        actual_protected=actual_protected,
    )
    return WriteResult(
        method=result.method,
        feasible=result.feasible,
        reason=result.reason,
        free_room_ratio=result.free_room_ratio,
        protected_overlap_ratio=result.protected_overlap_ratio,
        new_mse_before=result.new_mse_before,
        new_mse_after=result.new_mse_after,
        new_error_reduction=result.new_error_reduction,
        old_mse_before=result.old_mse_before,
        old_mse_after=result.old_mse_after,
        old_damage=result.old_damage,
        update_norm=result.update_norm,
        budget_scale=scale,
        update_to_gain_ratio=result.update_to_gain_ratio,
        max_old_key_error_after=result.max_old_key_error_after,
    )


def raw_rank_one_write(w_base: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, eps: float) -> torch.Tensor:
    error = v_new - w_base @ k_new
    denom = scalar(k_new.T @ k_new)
    if denom <= eps:
        raise RuntimeError("New key has near-zero norm.")
    return error @ k_new.T / denom


def free_room_write(
    w_base: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    p_free: torch.Tensor,
    *,
    eps: float,
    min_free_room: float,
) -> tuple[torch.Tensor | None, str]:
    k_free = p_free @ k_new
    free_room = scalar(torch.linalg.vector_norm(k_free) ** 2 / (torch.linalg.vector_norm(k_new) ** 2 + eps))
    if free_room < min_free_room:
        return None, f"free_room_ratio {free_room:.6f} < --min-free-room {min_free_room:.6f}"
    denom = scalar(k_free.T @ k_new)
    if denom <= eps:
        return None, f"free write denominator {denom:.6e} <= eps {eps:.6e}"
    error = v_new - w_base @ k_new
    return error @ k_free.T / denom, "ok"


def protected_least_squares_write(
    w_base: torch.Tensor,
    old_keys: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    *,
    lambda_protect: float,
    lambda_ridge: float,
) -> tuple[torch.Tensor, float]:
    if lambda_protect < 0:
        raise ValueError("--lambda-protect must be non-negative.")
    if lambda_ridge <= 0:
        raise ValueError("--lambda-ridge must be positive.")

    key_dim = k_new.shape[0]
    identity = torch.eye(key_dim, device=k_new.device, dtype=k_new.dtype)
    system = k_new @ k_new.T + lambda_protect * (old_keys @ old_keys.T) + lambda_ridge * identity
    condition = scalar(torch.linalg.cond(system))
    right = torch.linalg.solve(system, k_new)
    error = v_new - w_base @ k_new
    return error @ right.T, condition


def run_case(
    *,
    key_dim: int,
    value_dim: int,
    old_count: int,
    requested_overlap: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    lambda_protect: float,
    lambda_ridge: float,
    min_free_room: float,
    max_update_norm: float | None,
    eps: float,
) -> SweepCase:
    old_keys, old_values, k_new, v_new, p_old, p_free, actual_protected, actual_free = make_case(
        key_dim=key_dim,
        value_dim=value_dim,
        old_count=old_count,
        requested_overlap=requested_overlap,
        seed=seed,
        device=device,
        dtype=dtype,
        eps=eps,
    )
    _ = p_old
    w_base = base_weight(old_keys, old_values)

    raw_delta = raw_rank_one_write(w_base, k_new, v_new, eps)
    free_delta, free_reason = free_room_write(
        w_base,
        k_new,
        v_new,
        p_free,
        eps=eps,
        min_free_room=min_free_room,
    )
    protected_delta, condition = protected_least_squares_write(
        w_base,
        old_keys,
        k_new,
        v_new,
        lambda_protect=lambda_protect,
        lambda_ridge=lambda_ridge,
    )

    return SweepCase(
        seed=seed,
        key_dim=key_dim,
        value_dim=value_dim,
        old_count=old_count,
        requested_overlap=requested_overlap,
        actual_free_room_ratio=actual_free,
        actual_protected_overlap_ratio=actual_protected,
        condition_number_protected_system=condition,
        results=[
            evaluate_write(
                method="raw",
                delta_w=raw_delta,
                infeasible_reason="ok",
                w_base=w_base,
                old_keys=old_keys,
                old_values=old_values,
                k_new=k_new,
                v_new=v_new,
                actual_free=actual_free,
                actual_protected=actual_protected,
            ),
            evaluate_write(
                method="free",
                delta_w=free_delta,
                infeasible_reason=free_reason,
                w_base=w_base,
                old_keys=old_keys,
                old_values=old_values,
                k_new=k_new,
                v_new=v_new,
                actual_free=actual_free,
                actual_protected=actual_protected,
            ),
            evaluate_write(
                method="protected_ls",
                delta_w=protected_delta,
                infeasible_reason="ok",
                w_base=w_base,
                old_keys=old_keys,
                old_values=old_values,
                k_new=k_new,
                v_new=v_new,
                actual_free=actual_free,
                actual_protected=actual_protected,
            ),
            evaluate_budgeted_write(
                method="raw_budget",
                delta_w=raw_delta,
                infeasible_reason="ok",
                max_update_norm=max_update_norm,
                w_base=w_base,
                old_keys=old_keys,
                old_values=old_values,
                k_new=k_new,
                v_new=v_new,
                actual_free=actual_free,
                actual_protected=actual_protected,
            ),
            evaluate_budgeted_write(
                method="free_budget",
                delta_w=free_delta,
                infeasible_reason=free_reason,
                max_update_norm=max_update_norm,
                w_base=w_base,
                old_keys=old_keys,
                old_values=old_values,
                k_new=k_new,
                v_new=v_new,
                actual_free=actual_free,
                actual_protected=actual_protected,
            ),
            evaluate_budgeted_write(
                method="protected_ls_budget",
                delta_w=protected_delta,
                infeasible_reason="ok",
                max_update_norm=max_update_norm,
                w_base=w_base,
                old_keys=old_keys,
                old_values=old_values,
                k_new=k_new,
                v_new=v_new,
                actual_free=actual_free,
                actual_protected=actual_protected,
            ),
        ],
    )


def fmt_float(value: float | None, width: int = 10) -> str:
    if value is None:
        return "None".rjust(width)
    if math.isinf(value):
        return "inf".rjust(width)
    return f"{value:{width}.4g}"


def print_explanation() -> None:
    print()
    print("GCO ORTHOGONAL-ROOM WRITE TEST")
    print("=" * 96)
    print("Question:")
    print("  Can a new key-value behavior be written through free geometry while old protected")
    print("  key-value behaviors stay unchanged?")
    print()
    print("Model:")
    print("  y = W k")
    print("  old protected behavior: W K_old = V_old")
    print("  new behavior:           W k_new -> v_new")
    print()
    print("Main geometry:")
    print("  P_free = I - K_old (K_old^T K_old)^-1 K_old^T")
    print("  free_room_ratio = ||P_free k_new||^2 / ||k_new||^2")
    print()
    print("Writes compared:")
    print("  raw           unconstrained rank-one write")
    print("  free          write only with P_free k_new")
    print("  protected_ls  least-squares write penalizing movement on K_old")
    print("  *_budget      same write clipped to --max-update-norm")
    print("=" * 96)


def print_cases(cases: list[SweepCase]) -> None:
    grouped: dict[tuple[int, float], list[SweepCase]] = {}
    for case in cases:
        grouped.setdefault((case.old_count, case.requested_overlap), []).append(case)

    print()
    print("Readable summary")
    print("-" * 96)
    print(
        "old  overlap  free    method               ok  new_before  new_after  new_gain   "
        "old_damage  upd_norm  scale  upd/gain"
    )
    print("-" * 96)
    for case in cases:
        for result in case.results:
            ok = "yes" if result.feasible else "no "
            print(
                f"{case.old_count:>3d}  "
                f"{case.requested_overlap:>7.2f}  "
                f"{case.actual_free_room_ratio:>5.2f}  "
                f"{result.method:<20s}  "
                f"{ok:<3s}  "
                f"{fmt_float(result.new_mse_before)}  "
                f"{fmt_float(result.new_mse_after)}  "
                f"{fmt_float(result.new_error_reduction)}  "
                f"{fmt_float(result.old_damage)}  "
                f"{fmt_float(result.update_norm)}  "
                f"{fmt_float(result.budget_scale, width=6)}  "
                f"{fmt_float(result.update_to_gain_ratio)}"
            )
            if not result.feasible:
                print(f"     infeasible: {result.reason}")
        print("-" * 96)

    print()
    print("How to read this:")
    print("  free close to 1.0 means the new key mostly lies outside protected old geometry.")
    print("  new_gain should be positive; larger means the write learned the new behavior.")
    print("  old_damage should stay near zero; large means the write overwrote old behavior.")
    print("  upd/gain shows how much update norm was needed for each unit of new error reduction.")
    print("  scale below 1.0 means the write exceeded --max-update-norm and was clipped.")


def cases_to_json(cases: list[SweepCase]) -> list[dict[str, object]]:
    return [
        {
            **{k: v for k, v in asdict(case).items() if k != "results"},
            "results": [asdict(result) for result in case.results],
        }
        for case in cases
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=16)
    parser.add_argument("--old-counts", type=str, default="8,24,48")
    parser.add_argument("--overlaps", type=str, default="0.0,0.25,0.5,0.75,0.9,0.97")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1, help="Number of consecutive seeds starting at --seed.")
    parser.add_argument("--lambda-protect", type=float, default=100.0)
    parser.add_argument("--lambda-ridge", type=float, default=1e-5)
    parser.add_argument("--min-free-room", type=float, default=1e-6)
    parser.add_argument(
        "--max-update-norm",
        type=float,
        default=None,
        help="If set, also reports budgeted variants clipped to this Frobenius norm.",
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/analysis/gco-write-orthogonal-room-seed0.json"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    old_counts = parse_int_list(args.old_counts)
    overlaps = parse_float_list(args.overlaps)

    cases: list[SweepCase] = []
    for seed_offset in range(args.seeds):
        case_seed = args.seed + seed_offset
        for old_count in old_counts:
            for overlap in overlaps:
                cases.append(
                    run_case(
                        key_dim=args.key_dim,
                        value_dim=args.value_dim,
                        old_count=old_count,
                        requested_overlap=overlap,
                        seed=case_seed,
                        device=device,
                        dtype=dtype,
                        lambda_protect=args.lambda_protect,
                        lambda_ridge=args.lambda_ridge,
                        min_free_room=args.min_free_room,
                        max_update_norm=args.max_update_norm,
                        eps=args.eps,
                    )
                )

    print_explanation()
    print_cases(cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "gco_write_orthogonal_room",
        "config": {
            "key_dim": args.key_dim,
            "value_dim": args.value_dim,
            "old_counts": old_counts,
            "overlaps": overlaps,
            "seed": args.seed,
            "seeds": args.seeds,
            "lambda_protect": args.lambda_protect,
            "lambda_ridge": args.lambda_ridge,
            "min_free_room": args.min_free_room,
            "max_update_norm": args.max_update_norm,
            "device": str(device),
            "dtype": str(dtype),
        },
        "cases": cases_to_json(cases),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote_json={args.output}")


if __name__ == "__main__":
    main()
