"""Shared helpers for the real-book continual learning experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown device {requested!r}. Expected one of: cpu, cuda, mps.")


def require_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not contain required token {token!r}.")
    return token_id


def format_qa_prompt(question: str) -> str:
    return question.strip() + " "


def make_lm_sequences(tokens: list[int], max_seq_len: int, pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    if max_seq_len < 2:
        raise ValueError(f"max_seq_len must be >= 2, got {max_seq_len}.")
    if not tokens:
        raise ValueError("Cannot create LM sequences from an empty token list.")

    input_seqs: list[list[int]] = []
    target_seqs: list[list[int]] = []

    if len(tokens) > max_seq_len:
        step = max_seq_len // 2
        starts = list(range(0, len(tokens) - max_seq_len + 1, step))
        tail_start = len(tokens) - max_seq_len
        if not starts or starts[-1] != tail_start:
            starts.append(tail_start)
        for start in starts:
            seq = tokens[start : start + max_seq_len]
            input_seqs.append(seq[:-1])
            target_seqs.append(seq[1:])
    else:
        seq = tokens + [pad_id] * (max_seq_len - len(tokens))
        input_seqs.append(seq[:-1])
        target_seqs.append(seq[1:])

    return input_seqs, target_seqs


def make_qa_supervision(
    prompts: list[dict[str, str]],
    tokenizer: Tokenizer,
    max_seq_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not prompts:
        return None

    input_seqs: list[list[int]] = []
    target_seqs: list[list[int]] = []
    answer_masks: list[list[float]] = []

    for prompt in prompts:
        prompt_ids = tokenizer.encode(format_qa_prompt(prompt["question"])).ids
        answer_ids = tokenizer.encode(prompt["answer"].strip()).ids
        if not prompt_ids:
            raise ValueError(f"Prompt encoded to zero tokens: {prompt['question']!r}")
        if not answer_ids:
            raise ValueError(f"Answer encoded to zero tokens: {prompt['answer']!r}")

        full_ids = prompt_ids + answer_ids
        if len(full_ids) > max_seq_len:
            raise ValueError(
                f"QA example exceeds max_seq_len={max_seq_len}: "
                f"question={prompt['question']!r}, answer={prompt['answer']!r}, tokens={len(full_ids)}"
            )

        padded = full_ids + [pad_id] * (max_seq_len - len(full_ids))
        input_seq = padded[:-1]
        target_seq = padded[1:]
        mask = [0.0] * (max_seq_len - 1)
        answer_start = len(prompt_ids) - 1
        for offset in range(len(answer_ids)):
            mask[answer_start + offset] = 1.0

        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
        answer_masks.append(mask)

    return (
        torch.tensor(input_seqs, dtype=torch.long),
        torch.tensor(target_seqs, dtype=torch.long),
        torch.tensor(answer_masks, dtype=torch.float32),
    )


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    denom = mask.sum()
    if denom.item() <= 0.0:
        raise ValueError("Answer supervision mask is empty.")
    return (losses * mask).sum() / denom
