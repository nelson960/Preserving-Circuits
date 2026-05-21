from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a decoder-only LM on a small sequential corpus and save checkpoints."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--train-param-regex", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--save-steps", required=True, help="Comma-separated step numbers, e.g. 0,20,80.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()

    validate_args(args)
    set_hf_home(args.hf_home)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_steps = parse_save_steps(args.save_steps, max_step=args.steps)
    train_set = load_train_set(args.train_json)
    device = device_from_name(args.device)
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if args.batch_size != 1 and tokenizer.pad_token_id is None:
        raise ValueError("batch-size > 1 requires tokenizer.pad_token_id; use batch-size 1 for this model.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    trainable_names = set_trainable_parameters(model, args.train_param_regex)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)

    args.output_root.mkdir(parents=True, exist_ok=False)
    saved_checkpoints: list[dict[str, Any]] = []
    losses: list[dict[str, float | int]] = []
    if 0 in save_steps:
        checkpoint_path = args.output_root / "checkpoint-step-000000"
        save_checkpoint(model, tokenizer, checkpoint_path)
        saved_checkpoints.append({"step": 0, "path": str(checkpoint_path)})

    examples = train_set["examples"]
    model.train()
    for step in range(1, args.steps + 1):
        batch = [examples[((step - 1) * args.batch_size + offset) % len(examples)] for offset in range(args.batch_size)]
        texts = [str(example["text"]) for example in batch]
        loss = compute_causal_loss(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=args.max_length,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became non-finite at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append({"step": step, "loss": loss_value})
        if step == 1 or step % max(1, args.steps // 10) == 0:
            print(json.dumps({"step": step, "loss": loss_value}, sort_keys=True))
        if step in save_steps:
            checkpoint_path = args.output_root / f"checkpoint-step-{step:06d}"
            save_checkpoint(model, tokenizer, checkpoint_path)
            saved_checkpoints.append({"step": step, "path": str(checkpoint_path)})

    report = {
        "model_dir": str(args.model_dir),
        "train_json": str(args.train_json),
        "train_name": train_set["name"],
        "train_param_regex": args.train_param_regex,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "loss_first": losses[0]["loss"],
        "loss_last": losses[-1]["loss"],
        "losses": losses,
        "checkpoints": saved_checkpoints,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report_json": str(args.report_json), "checkpoints": saved_checkpoints}, indent=2, sort_keys=True))


def set_trainable_parameters(model: torch.nn.Module, regex: str) -> list[str]:
    pattern = re.compile(regex)
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        trainable = pattern.search(name) is not None
        param.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError(f"train-param-regex matched no parameters: {regex!r}")
    return trainable_names


def compute_causal_loss(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    if len(texts) == 1:
        encoded = tokenizer(
            texts[0],
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
    else:
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
    if "input_ids" not in encoded:
        raise RuntimeError("tokenizer output missing input_ids.")
    input_ids = encoded["input_ids"]
    if input_ids.shape[-1] < 2:
        raise ValueError("causal training example must contain at least two tokens.")
    labels = input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = labels.to(device)
    outputs = model(**encoded, labels=labels, return_dict=True)
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError("model did not return loss.")
    return loss


def save_checkpoint(model: torch.nn.Module, tokenizer: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"checkpoint path already exists: {path}")
    path.mkdir(parents=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def load_train_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError("train JSON root must be an object.")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("train JSON must contain a non-empty name.")
    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("train JSON must contain a non-empty examples list.")
    seen_ids: set[str] = set()
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"examples[{index}] must be an object.")
        text = example.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"examples[{index}].text must be a non-empty string.")
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"examples[{index}].id must be a non-empty string.")
        if example_id in seen_ids:
            raise ValueError(f"duplicate example id: {example_id}")
        seen_ids.add(example_id)
    return data


def parse_save_steps(value: str, *, max_step: int) -> set[int]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("save-steps must contain at least one step.")
    steps: set[int] = set()
    for item in raw:
        try:
            step = int(item)
        except ValueError as error:
            raise ValueError(f"save step must be an integer, got {item!r}.") from error
        if step < 0 or step > max_step:
            raise ValueError(f"save step {step} out of range [0, {max_step}].")
        steps.add(step)
    return steps


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {args.model_dir}")
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"model-dir is not a directory: {args.model_dir}")
    if not args.train_json.exists():
        raise FileNotFoundError(f"train-json does not exist: {args.train_json}")
    if args.steps <= 0:
        raise ValueError(f"steps must be positive, got {args.steps}.")
    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}.")
    if args.max_length <= 0:
        raise ValueError(f"max-length must be positive, got {args.max_length}.")
    if args.output_root.exists():
        raise FileExistsError(f"output-root already exists: {args.output_root}")
    if args.report_json.exists():
        raise FileExistsError(f"report-json already exists: {args.report_json}")


def set_hf_home(hf_home: Path | None) -> None:
    if hf_home is None:
        return
    if hf_home.exists() and not hf_home.is_dir():
        raise NotADirectoryError(f"hf-home is not a directory: {hf_home}")
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home.resolve())


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def device_from_name(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false.")
    return torch.device(name)


if __name__ == "__main__":
    main()
