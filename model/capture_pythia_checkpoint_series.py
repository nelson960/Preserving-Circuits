from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture the same Pythia activation set for every checkpoint in a fine-tune report."
    )
    parser.add_argument("--train-report-json", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--layer-index", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=("all-tokens", "target-spans"))
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    report = load_train_report(args.train_report_json)
    checkpoints = require_checkpoints(report)
    args.output_root.mkdir(parents=True, exist_ok=False)

    captures: list[dict[str, Any]] = []
    capture_script = Path(__file__).resolve().parent / "capture_pythia_activations.py"
    if not capture_script.exists():
        raise FileNotFoundError(f"capture script does not exist: {capture_script}")
    for checkpoint in checkpoints:
        step = require_int(checkpoint, "step")
        checkpoint_path = Path(require_str(checkpoint, "path"))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")
        output_pt = args.output_root / f"step-{step:06d}-{args.mode}.pt"
        if output_pt.exists():
            raise FileExistsError(f"output activation file already exists: {output_pt}")
        command = [
            sys.executable,
            str(capture_script),
            "--model-dir",
            str(checkpoint_path),
            "--prompts",
            str(args.prompts),
            "--layer-index",
            str(args.layer_index),
            "--mode",
            args.mode,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--max-length",
            str(args.max_length),
            "--output-pt",
            str(output_pt),
        ]
        if args.hf_home is not None:
            command.extend(["--hf-home", str(args.hf_home)])
        subprocess.run(command, check=True)
        captures.append(
            {
                "step": step,
                "name": f"step{step}",
                "checkpoint_path": str(checkpoint_path),
                "activation_path": str(output_pt),
            }
        )

    manifest = {
        "train_report_json": str(args.train_report_json),
        "prompts": str(args.prompts),
        "layer_index": args.layer_index,
        "mode": args.mode,
        "device": args.device,
        "dtype": args.dtype,
        "captures": captures,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_json": str(args.manifest_json), "capture_count": len(captures)}, sort_keys=True))


def load_train_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("train report must be a JSON object.")
    return data


def require_checkpoints(report: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("train report must contain a non-empty checkpoints list.")
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise TypeError(f"checkpoints[{index}] must be an object.")
        require_int(checkpoint, "step")
        require_str(checkpoint, "path")
    return checkpoints


def require_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key!r} must be an int.")
    return value


def require_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} must be a non-empty string.")
    return value


def validate_args(args: argparse.Namespace) -> None:
    if not args.train_report_json.exists():
        raise FileNotFoundError(f"train-report-json does not exist: {args.train_report_json}")
    if not args.prompts.exists():
        raise FileNotFoundError(f"prompts does not exist: {args.prompts}")
    if args.layer_index < 0:
        raise ValueError(f"layer-index must be non-negative, got {args.layer_index}.")
    if args.max_length <= 0:
        raise ValueError(f"max-length must be positive, got {args.max_length}.")
    if args.hf_home is not None and args.hf_home.exists() and not args.hf_home.is_dir():
        raise NotADirectoryError(f"hf-home is not a directory: {args.hf_home}")
    if args.output_root.exists():
        raise FileExistsError(f"output-root already exists: {args.output_root}")
    if args.manifest_json.exists():
        raise FileExistsError(f"manifest-json already exists: {args.manifest_json}")


if __name__ == "__main__":
    main()
