from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch


class SparseAutoencoder(torch.nn.Module):
    def __init__(self, hidden_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(hidden_dim, feature_dim)
        self.decoder = torch.nn.Linear(feature_dim, hidden_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        self.normalize_decoder_columns()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        norms = torch.linalg.vector_norm(self.decoder.weight, dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a one-layer ReLU SAE on captured residual activations.")
    parser.add_argument("--activations-pt", required=True, type=Path)
    parser.add_argument("--feature-dim", required=True, type=int)
    parser.add_argument("--l1-coeff", required=True, type=float)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-pt", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = device_from_name(args.device)
    payload = torch.load(args.activations_pt, map_location="cpu", weights_only=False)
    activations = require_activation_matrix(payload)
    input_mean = activations.mean(dim=0, keepdim=True)
    centered = activations - input_mean
    hidden_dim = int(centered.shape[1])

    model = SparseAutoencoder(hidden_dim=hidden_dim, feature_dim=args.feature_dim).to(device)
    data = centered.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses: list[dict[str, float | int]] = []
    for step in range(args.steps):
        indices = torch.randint(0, data.shape[0], (args.batch_size,), device=device)
        batch = data[indices]
        x_hat, z = model(batch)
        mse = torch.mean((x_hat - batch) ** 2)
        l1 = torch.mean(torch.abs(z))
        loss = mse + args.l1_coeff * l1
        if not torch.isfinite(loss):
            raise RuntimeError(f"SAE loss became non-finite at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.normalize_decoder_columns()
        if step == 0 or (step + 1) % max(1, args.steps // 20) == 0 or step + 1 == args.steps:
            losses.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu().item()),
                    "mse": float(mse.detach().cpu().item()),
                    "l1": float(l1.detach().cpu().item()),
                    "active_fraction": float((z.detach() > 0).float().mean().cpu().item()),
                }
            )
            print(json.dumps(losses[-1], sort_keys=True))

    model_cpu = model.cpu()
    sae_payload = {
        "state_dict": model_cpu.state_dict(),
        "input_mean": input_mean,
        "config": {
            "hidden_dim": hidden_dim,
            "feature_dim": args.feature_dim,
            "l1_coeff": args.l1_coeff,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "source_activations": str(args.activations_pt),
            "source_metadata": payload.get("metadata", {}),
        },
    }
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sae_payload, args.output_pt)
    with torch.no_grad():
        x_hat, z = model_cpu(centered)
        full_mse = float(torch.mean((x_hat - centered) ** 2).item())
        variance = float(torch.mean(centered**2).item())
        explained = 1.0 - full_mse / variance if variance > 0 else 0.0
        active_fraction = float((z > 0).float().mean().item())
    report = {
        "sae_path": str(args.output_pt),
        "activation_path": str(args.activations_pt),
        "hidden_dim": hidden_dim,
        "feature_dim": args.feature_dim,
        "l1_coeff": args.l1_coeff,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "full_mse": full_mse,
        "input_variance": variance,
        "variance_explained": explained,
        "active_fraction": active_fraction,
        "losses": losses,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_pt": str(args.output_pt), "report_json": str(args.report_json)}, sort_keys=True))


def require_activation_matrix(payload: Any) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise TypeError("activation payload must be a dict.")
    activations = payload.get("activations")
    if not isinstance(activations, torch.Tensor):
        raise TypeError("activation payload must contain tensor field 'activations'.")
    if activations.ndim != 2:
        raise ValueError(f"activations must be rank-2, got shape {tuple(activations.shape)}.")
    if not torch.is_floating_point(activations):
        raise TypeError(f"activations must be floating point, got {activations.dtype}.")
    if not torch.isfinite(activations).all():
        raise ValueError("activations contain non-finite values.")
    return activations.float()


def validate_args(args: argparse.Namespace) -> None:
    if not args.activations_pt.exists():
        raise FileNotFoundError(f"activations-pt does not exist: {args.activations_pt}")
    if args.feature_dim <= 0:
        raise ValueError(f"feature-dim must be positive, got {args.feature_dim}.")
    if args.l1_coeff < 0:
        raise ValueError(f"l1-coeff must be non-negative, got {args.l1_coeff}.")
    if args.steps <= 0:
        raise ValueError(f"steps must be positive, got {args.steps}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}.")
    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}.")
    if args.output_pt.exists():
        raise FileExistsError(f"output-pt already exists: {args.output_pt}")
    if args.report_json.exists():
        raise FileExistsError(f"report-json already exists: {args.report_json}")


def device_from_name(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false.")
    return torch.device(name)


if __name__ == "__main__":
    main()
