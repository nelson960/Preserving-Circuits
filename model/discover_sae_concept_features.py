from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from train_residual_sae import SparseAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank SAE features by concept selectivity on target-token activations."
    )
    parser.add_argument("--sae-pt", required=True, type=Path)
    parser.add_argument("--concept-activations-pt", required=True, type=Path)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--negative-labels", required=True, help="Comma-separated labels.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    negative_labels = parse_csv(args.negative_labels)
    if args.concept in negative_labels:
        raise ValueError("concept must not be included in negative-labels.")

    sae_payload = torch.load(args.sae_pt, map_location="cpu", weights_only=False)
    activation_payload = torch.load(args.concept_activations_pt, map_location="cpu", weights_only=False)
    activations = require_activation_matrix(activation_payload)
    rows = require_rows(activation_payload)
    sae = load_sae(sae_payload)
    input_mean = require_input_mean(sae_payload)
    centered = activations - input_mean
    with torch.no_grad():
        _x_hat, z = sae(centered)

    labels = [str(row["label"]) for row in rows]
    concept_indices = [idx for idx, label in enumerate(labels) if label == args.concept]
    negative_indices = [idx for idx, label in enumerate(labels) if label in negative_labels]
    if not concept_indices:
        raise RuntimeError(f"no rows found for concept {args.concept!r}.")
    if not negative_indices:
        raise RuntimeError(f"no rows found for negative labels {negative_labels}.")

    concept_z = z[concept_indices]
    negative_z = z[negative_indices]
    concept_mean = concept_z.mean(dim=0)
    negative_mean = negative_z.mean(dim=0)
    selectivity = concept_mean - negative_mean
    firing_thresholds = torch.quantile(z, 0.95, dim=0)
    concept_firing = (concept_z > firing_thresholds).float().mean(dim=0)
    negative_firing = (negative_z > firing_thresholds).float().mean(dim=0)
    firing_delta = concept_firing - negative_firing

    feature_reports: list[dict[str, Any]] = []
    for feature_index in range(z.shape[1]):
        values = z[:, feature_index]
        binary = torch.zeros(z.shape[0], dtype=torch.bool)
        binary[concept_indices] = True
        valid = torch.zeros(z.shape[0], dtype=torch.bool)
        valid[concept_indices] = True
        valid[negative_indices] = True
        auc = auroc(values[valid], binary[valid])
        top_rows = top_activating_rows(values, rows, top_k=min(8, len(rows)))
        feature_reports.append(
            {
                "feature_index": feature_index,
                "selectivity": float(selectivity[feature_index].item()),
                "concept_mean_activation": float(concept_mean[feature_index].item()),
                "negative_mean_activation": float(negative_mean[feature_index].item()),
                "firing_delta": float(firing_delta[feature_index].item()),
                "concept_firing_rate": float(concept_firing[feature_index].item()),
                "negative_firing_rate": float(negative_firing[feature_index].item()),
                "auroc": auc,
                "top_activating_rows": top_rows,
            }
        )
    ranked = sorted(
        feature_reports,
        key=lambda item: (float(item["auroc"]), float(item["selectivity"]), float(item["firing_delta"])),
        reverse=True,
    )
    report = {
        "sae_path": str(args.sae_pt),
        "activation_path": str(args.concept_activations_pt),
        "concept": args.concept,
        "negative_labels": negative_labels,
        "row_count": len(rows),
        "concept_count": len(concept_indices),
        "negative_count": len(negative_indices),
        "top_k": args.top_k,
        "ranked_features": ranked[: args.top_k],
        "method": {
            "feature_space": "SAE ReLU latent activations",
            "selectivity": "mean(z_j | concept) - mean(z_j | negative labels)",
            "auroc": "rank statistic over concept vs negative rows",
            "note": "SAE selectivity is not causal evidence; run ablation/patching next.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_json": str(args.output_json), "top_feature": ranked[0]}, indent=2, sort_keys=True))


def load_sae(payload: dict[str, Any]) -> SparseAutoencoder:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise TypeError("SAE payload must contain config dict.")
    hidden_dim = config.get("hidden_dim")
    feature_dim = config.get("feature_dim")
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("SAE config hidden_dim must be a positive int.")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("SAE config feature_dim must be a positive int.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("SAE payload must contain state_dict.")
    sae = SparseAutoencoder(hidden_dim=hidden_dim, feature_dim=feature_dim)
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae


def require_input_mean(payload: dict[str, Any]) -> torch.Tensor:
    input_mean = payload.get("input_mean")
    if not isinstance(input_mean, torch.Tensor):
        raise TypeError("SAE payload must contain tensor input_mean.")
    if input_mean.ndim != 2 or input_mean.shape[0] != 1:
        raise ValueError(f"input_mean must have shape [1, hidden_dim], got {tuple(input_mean.shape)}.")
    return input_mean.float()


def require_activation_matrix(payload: Any) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise TypeError("activation payload must be a dict.")
    activations = payload.get("activations")
    if not isinstance(activations, torch.Tensor):
        raise TypeError("activation payload must contain tensor activations.")
    if activations.ndim != 2:
        raise ValueError(f"activations must be rank-2, got shape {tuple(activations.shape)}.")
    return activations.float()


def require_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise TypeError("activation payload must be a dict.")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("activation payload must contain a non-empty rows list.")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"rows[{idx}] must be an object.")
        if "label" not in row:
            raise ValueError(f"rows[{idx}] is missing label.")
    return rows


def auroc(values: torch.Tensor, labels: torch.Tensor) -> float:
    if values.ndim != 1 or labels.ndim != 1 or values.shape[0] != labels.shape[0]:
        raise ValueError("values and labels must be rank-1 tensors with equal length.")
    pos = values[labels]
    neg = values[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        raise ValueError("AUROC requires at least one positive and one negative.")
    comparisons = (pos[:, None] > neg[None, :]).float()
    ties = (pos[:, None] == neg[None, :]).float() * 0.5
    return float((comparisons + ties).mean().item())


def top_activating_rows(values: torch.Tensor, rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    k = min(top_k, values.shape[0])
    top_values, top_indices = torch.topk(values, k=k, largest=True)
    output: list[dict[str, Any]] = []
    for value, index in zip(top_values, top_indices, strict=True):
        row = rows[int(index.item())]
        output.append(
            {
                "activation": float(value.item()),
                "row_index": int(index.item()),
                "prompt_id": str(row.get("prompt_id", "")),
                "label": str(row.get("label", "")),
                "target": str(row.get("target", "")),
                "text": str(row.get("text", "")),
            }
        )
    return output


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("CSV argument must contain at least one item.")
    if len(set(items)) != len(items):
        raise ValueError(f"CSV argument contains duplicate labels: {items}")
    return items


def validate_args(args: argparse.Namespace) -> None:
    if not args.sae_pt.exists():
        raise FileNotFoundError(f"sae-pt does not exist: {args.sae_pt}")
    if not args.concept_activations_pt.exists():
        raise FileNotFoundError(f"concept-activations-pt does not exist: {args.concept_activations_pt}")
    if args.top_k <= 0:
        raise ValueError(f"top-k must be positive, got {args.top_k}.")
    if args.output_json.exists():
        raise FileExistsError(f"output-json already exists: {args.output_json}")


if __name__ == "__main__":
    main()
