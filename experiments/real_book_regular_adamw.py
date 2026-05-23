"""Sequential AdamW baseline fine-tuning script.

Fine-tunes the base transformer blocks sequentially on book chunks using standard AdamW
and evaluates prompt retention/forgetting after each chunk step.
"""

from __future__ import annotations
import argparse
import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

# Add parent directory to sys.path to import models
sys.path.append(str(Path(__file__).parent))
from models import DecoderTransformer
from real_book_common import (
    format_qa_prompt,
    make_lm_sequences,
    make_qa_supervision,
    masked_cross_entropy,
    require_token_id,
    resolve_device,
)

def format_prompt(question: str) -> str:
    # Ensure standard prompt format
    return format_qa_prompt(question)


def build_training_text(chunk: dict, include_local_prompts: bool) -> str:
    parts = [chunk["text"]]
    if include_local_prompts:
        for prompt in chunk["local_prompts"]:
            parts.append(f"{format_prompt(prompt['question'])}{prompt['answer']}")
    return "\n\n".join(parts)

def generate_greedy(model: DecoderTransformer, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 8, device: torch.device = torch.device("cpu")) -> str:
    model.eval()
    encoded = tokenizer.encode(prompt).ids
    input_ids = list(encoded)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = min(len(input_ids), model.max_seq_len)
            input_tensor = torch.tensor([input_ids[-seq_len:]], dtype=torch.long, device=device)
            logits, _ = model(input_tensor)
            next_token = logits[0, -1].argmax(dim=-1).item()
            input_ids.append(next_token)
            
            # Stop tokens
            eos_id = tokenizer.token_to_id("[EOS]")
            nl_id = tokenizer.token_to_id("\n")
            if next_token == eos_id or (nl_id is not None and next_token == nl_id):
                break
                
    generated_tokens = input_ids[len(encoded):]
    return tokenizer.decode(generated_tokens).strip()

def evaluate_qa_loss_and_acc(model: DecoderTransformer, tokenizer: Tokenizer, prompt_dict: dict[str, str], device: torch.device) -> dict[str, float | str]:
    question = prompt_dict["question"]
    expected_answer = prompt_dict["answer"].strip()
    
    prompt_str = format_prompt(question)
    prompt_ids = tokenizer.encode(prompt_str).ids
    answer_ids = tokenizer.encode(expected_answer).ids
    
    # 1. Compute loss of target answer tokens
    model.eval()
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > model.max_seq_len:
        raise ValueError(
            f"QA evaluation example exceeds max_seq_len={model.max_seq_len}: "
            f"question={question!r}, answer={expected_answer!r}, tokens={len(full_ids)}"
        )
        
    input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits, _ = model(input_tensor)
        # We only care about the loss of the answer tokens
        answer_start_idx = len(prompt_ids) - 1
        answer_logits = logits[0, answer_start_idx:answer_start_idx + len(answer_ids)]
        answer_targets = target_tensor[0, answer_start_idx:answer_start_idx + len(answer_ids)]
        loss = F.cross_entropy(answer_logits.reshape(-1, logits.shape[-1]), answer_targets.reshape(-1)).item()
        pred_ids = answer_logits.argmax(dim=-1)
        token_accuracy = (pred_ids == answer_targets).float().mean().item()
        exact_answer_match = 1.0 if torch.equal(pred_ids, answer_targets) else 0.0
        
    # 2. Greedy generation substring match
    output_str = generate_greedy(model, tokenizer, prompt_str, max_new_tokens=len(answer_ids) + 5, device=device)
    generation_match = 1.0 if expected_answer.lower() in output_str.lower() else 0.0
    
    return {
        "loss": float(loss),
        "accuracy": exact_answer_match,
        "token_accuracy": float(token_accuracy),
        "generation_match": generation_match,
        "generated": output_str,
        "expected": expected_answer,
        "question": question
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential AdamW baseline on real book chunks.")
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--output-model-path", type=Path, default=Path("checkpoints/real_book/regular_adamw_book_model.pt"))
    parser.add_argument("--output-results-json", type=Path, default=Path("model/analysis/real-book-regular-adamw.json"))
    parser.add_argument("--epochs-per-chunk", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-local-prompts-in-training", action="store_true")
    parser.add_argument("--qa-loss-weight", type=float, default=5.0)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_results_json.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load resources
    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSON not found at {args.chunks_path}. Run data preparation first.")
    with open(args.chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # 2. Instantiate Model and load weights
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    if not args.base_model_path.exists():
        raise FileNotFoundError(f"Base model checkpoint not found at {args.base_model_path}. Run train_base_model.py first.")
    
    model.load_state_dict(torch.load(args.base_model_path, map_location=device))
    print("Loaded base model weights.")

    # Freeze embeddings to preserve token manifold structure
    for name, param in model.named_parameters():
        if "token_embedding" in name or "position_embedding" in name or "lm_head" in name:
            param.requires_grad = False
            
    # Collect only parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    step_results = []

    # 3. Sequential training loop
    for chunk_idx, c_data in enumerate(chunks):
        chunk_id = c_data["chunk_id"]
        print(f"\n--- Training on Chunk {chunk_id} ({chunk_idx+1}/{len(chunks)}) ---")
        
        # Tokenize chunk training stream.
        training_text = build_training_text(c_data, args.include_local_prompts_in_training)
        tokens = tokenizer.encode(training_text).ids
        pad_id = require_token_id(tokenizer, "[PAD]")
        input_seqs, target_seqs = make_lm_sequences(tokens, args.max_seq_len, pad_id)
            
        inputs_t = torch.tensor(input_seqs, dtype=torch.long)
        targets_t = torch.tensor(target_seqs, dtype=torch.long)
        qa_supervision = None
        if args.include_local_prompts_in_training:
            qa_supervision = make_qa_supervision(c_data["local_prompts"], tokenizer, args.max_seq_len, pad_id)
            if qa_supervision is not None:
                qa_inputs_t, qa_targets_t, qa_mask_t = qa_supervision
                qa_inputs_t = qa_inputs_t.to(device)
                qa_targets_t = qa_targets_t.to(device)
                qa_mask_t = qa_mask_t.to(device)
        
        # Setup optimizer for chunk fine-tuning
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
        
        # Fine-tune model
        dataset_size = len(inputs_t)
        model.train()
        for epoch in range(args.epochs_per_chunk):
            permutation = torch.randperm(dataset_size)
            for i in range(0, dataset_size, args.batch_size):
                optimizer.zero_grad()
                indices = permutation[i : i + args.batch_size]
                b_ins = inputs_t[indices].to(device)
                b_targets = targets_t[indices].to(device)
                
                logits, _ = model(b_ins)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), b_targets.reshape(-1))
                if qa_supervision is not None:
                    qa_logits, _ = model(qa_inputs_t)
                    qa_loss = masked_cross_entropy(qa_logits, qa_targets_t, qa_mask_t)
                    loss = loss + args.qa_loss_weight * qa_loss
                loss.backward()
                optimizer.step()

        # 4. Evaluate Prompts
        print("Evaluating current, past, and composition prompts...")
        local_evals = []
        for p in c_data["local_prompts"]:
            local_evals.append(evaluate_qa_loss_and_acc(model, tokenizer, p, device))
            
        retention_evals = []
        for p in c_data["retention_prompts"]:
            retention_evals.append(evaluate_qa_loss_and_acc(model, tokenizer, p, device))
            
        composition_evals = []
        for p in c_data["composition_prompts"]:
            composition_evals.append(evaluate_qa_loss_and_acc(model, tokenizer, p, device))
            
        # Summary statistics
        local_acc = sum(r["accuracy"] for r in local_evals) / len(local_evals) if local_evals else 1.0
        retention_acc = sum(r["accuracy"] for r in retention_evals) / len(retention_evals) if retention_evals else 1.0
        comp_acc = sum(r["accuracy"] for r in composition_evals) / len(composition_evals) if composition_evals else 1.0
        local_gen_acc = sum(r["generation_match"] for r in local_evals) / len(local_evals) if local_evals else 1.0
        retention_gen_acc = sum(r["generation_match"] for r in retention_evals) / len(retention_evals) if retention_evals else 1.0
        comp_gen_acc = sum(r["generation_match"] for r in composition_evals) / len(composition_evals) if composition_evals else 1.0
        
        print(f"Local Prompt Accuracy: {local_acc:.4f}")
        print(f"Retention Prompt Accuracy: {retention_acc:.4f}")
        print(f"Local Generation Match: {local_gen_acc:.4f}")
        print(f"Retention Generation Match: {retention_gen_acc:.4f}")
        if composition_evals:
            print(f"Composition Prompt Accuracy: {comp_acc:.4f}")
            print(f"Composition Generation Match: {comp_gen_acc:.4f}")
            
        step_results.append({
            "chunk_id": chunk_id,
            "local_accuracy": local_acc,
            "retention_accuracy": retention_acc,
            "composition_accuracy": comp_acc,
            "local_generation_match": local_gen_acc,
            "retention_generation_match": retention_gen_acc,
            "composition_generation_match": comp_gen_acc,
            "local_evals": local_evals,
            "retention_evals": retention_evals,
            "composition_evals": composition_evals
        })

    # Save final model weights
    torch.save(model.state_dict(), args.output_model_path)
    print(f"Saved sequential AdamW model checkpoint to {args.output_model_path}")

    # Write evaluation metrics
    with open(args.output_results_json, "w", encoding="utf-8") as f:
        json.dump(step_results, f, indent=2)
    print(f"Saved evaluation results JSON to {args.output_results_json}")

if __name__ == "__main__":
    main()
