"""Pre-training script for the base causal language model.

Trains the DecoderTransformer on Alice's Adventures in Wonderland to establish
a stable representation manifold and basic grammar before continual book learning.
"""

from __future__ import annotations
import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

# Add the current experiments directory to sys.path to import models
sys.path.append(str(Path(__file__).parent))
from models import DecoderTransformer
from real_book_common import resolve_device

def load_data(text_path: Path, tokenizer: Tokenizer, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    text = text_path.read_text(encoding="utf-8")
    encoded = tokenizer.encode(text).ids
    
    # Chunk into sequences of length max_seq_len
    input_seqs = []
    target_seqs = []
    
    # Overlap slightly for better training coverage
    step = max_seq_len // 2
    for i in range(0, len(encoded) - max_seq_len, step):
        chunk = encoded[i : i + max_seq_len]
        input_seqs.append(chunk[:-1])
        target_seqs.append(chunk[1:])
        
    return (
        torch.tensor(input_seqs, dtype=torch.long),
        torch.tensor(target_seqs, dtype=torch.long)
    )

def train_base_model(args: argparse.Namespace) -> None:
    # 1. Load tokenizer
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}. Run prepare_real_book_benchmark.py first.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with vocab_size={vocab_size}")

    # 2. Load pre-training corpus
    if not args.train_text_path.exists():
        raise FileNotFoundError(f"Pre-training text not found at {args.train_text_path}.")
    inputs, targets = load_data(args.train_text_path, tokenizer, args.max_seq_len)
    print(f"Loaded {len(inputs)} training sequences of length {args.max_seq_len-1}")

    # 3. Initialize Model
    device = resolve_device(args.device)
    print(f"Training on device: {device}")

    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Initialized DecoderTransformer with {param_count/1e6:.3f}M parameters")

    # 4. Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    dataset_size = len(inputs)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        permutation = torch.randperm(dataset_size)
        
        pbar = tqdm(range(0, dataset_size, args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        for i in pbar:
            optimizer.zero_grad()
            indices = permutation[i : i + args.batch_size]
            batch_inputs = inputs[indices].to(device)
            batch_targets = targets[indices].to(device)
            
            logits, _ = model(batch_inputs)
            # Flatten for CE loss
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                batch_targets.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * len(indices)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Mean Loss: {epoch_loss / dataset_size:.4f}")

    # 5. Save model
    args.output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_model_path)
    print(f"Saved base model checkpoint to {args.output_model_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train the base DecoderTransformer model.")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--train-text_path", type=Path, default=Path("data/real_book/alice.txt"), help="Text file to pre-train on.")
    parser.add_argument("--output-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    train_base_model(args)

if __name__ == "__main__":
    main()
