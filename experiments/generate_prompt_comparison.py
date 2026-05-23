"""Evaluation script to compare model generations side-by-side.

Loads base, regular AdamW, and latent-geometry CL checkpoints, runs generation on
curated prompts, and writes model/analysis/real-book-generations.md.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

# Add experiments directory to path
sys.path.append(str(Path(__file__).parent))
from models import DecoderTransformer, ClosedOperator
from real_book_common import resolve_device
from real_book_regular_adamw import generate_greedy as regular_generate_greedy
from real_book_geometry_cl import eval_program, generate_greedy as cl_generate_greedy

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate side-by-side prompt comparisons.")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--eval-prompts-path", type=Path, default=Path("data/real_book/eval_prompts.json"))
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--regular-model-path", type=Path, default=Path("checkpoints/real_book/regular_adamw_book_model.pt"))
    parser.add_argument("--geometry-model-path", type=Path, default=Path("checkpoints/real_book/latent_geometry_book_model.pt"))
    parser.add_argument("--output-md-path", type=Path, default=Path("model/analysis/real-book-generations.md"))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--operator-hidden-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_md_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    # 2. Load evaluation prompts and chunks
    if not args.eval_prompts_path.exists() or not args.chunks_path.exists():
        raise FileNotFoundError("Evaluation data files not found. Prepare data first.")
    
    with open(args.eval_prompts_path, "r", encoding="utf-8") as f:
        eval_groups = json.load(f)
        
    with open(args.chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 3. Instantiate and load Base Model
    base_model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len
    ).to(device)
    base_model.load_state_dict(torch.load(args.base_model_path, map_location=device))
    base_model.eval()

    # 4. Instantiate and load Regular AdamW Model
    regular_model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len
    ).to(device)
    regular_model.load_state_dict(torch.load(args.regular_model_path, map_location=device))
    regular_model.eval()

    # 5. Load Latent Geometry CL State
    cl_state = torch.load(args.geometry_model_path, map_location=device)
    cl_base_model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len
    ).to(device)
    cl_base_model.load_state_dict(cl_state["base_model"])
    cl_base_model.eval()

    if "operator_hidden_dim" not in cl_state:
        raise KeyError("Geometry checkpoint is missing required key 'operator_hidden_dim'. Rerun real_book_geometry_cl.py.")
    operator_hidden_dim = int(cl_state["operator_hidden_dim"])

    # Reconstruct adaptation library
    library: dict[str, ClosedOperator] = {}
    for op_name, op_weights in cl_state["library"].items():
        op = ClosedOperator(args.d_model, operator_hidden_dim).to(device)
        op.load_state_dict(op_weights)
        op.eval()
        library[op_name] = op

    task_to_program = cl_state["task_to_program"]

    # 6. Generate outputs for each prompt group
    md_lines = [
        "# Real-Book Benchmark: Prompt Completion Generations",
        "",
        "This file compares the text generations of three model checkpoints side-by-side:",
        "1. **Base Model**: Pre-trained on grammar/syntax but naive to Wizard of Oz.",
        "2. **Regular AdamW Model**: Sequentially fine-tuned block-by-block; vulnerable to forgetting.",
        "3. **Latent-Geometry CL Model**: Dynamically adapted via closed operators.",
        ""
    ]

    for group_name, prompts in eval_groups.items():
        md_lines.append(f"## {group_name.replace('_', ' ').title()}")
        md_lines.append("")
        
        for p_idx, prompt in enumerate(prompts):
            question = prompt["question"]
            expected = prompt["answer"]
            
            prompt_str = question.strip() + " "
            
            # A. Base model generation
            base_out = regular_generate_greedy(base_model, tokenizer, prompt_str, max_new_tokens=10, device=device)
            
            # B. Regular AdamW generation
            adamw_out = regular_generate_greedy(regular_model, tokenizer, prompt_str, max_new_tokens=10, device=device)
            
            # C. Geometry CL generation
            # Determine appropriate program based on prompt's chunk origin
            matching_chunk_idx = None
            for idx, c in enumerate(chunks):
                if any(prompt["question"] == lp["question"] for lp in c["local_prompts"]):
                    matching_chunk_idx = idx
                    break
            
            if matching_chunk_idx is not None:
                chunk_id = chunks[matching_chunk_idx]["chunk_id"]
                if chunk_id not in task_to_program:
                    raise RuntimeError(f"No CL program found for prompt source chunk {chunk_id}.")
                prog = task_to_program[chunk_id]
            elif group_name == "cross_chunk_prompts":
                chunk_id = chunks[-1]["chunk_id"]
                if chunk_id not in task_to_program:
                    raise RuntimeError(f"No CL program found for final chunk {chunk_id}.")
                prog = task_to_program[chunk_id]
            else:
                raise RuntimeError(f"Could not determine CL program for prompt {question!r}.")
                
            cl_out = cl_generate_greedy(cl_base_model, library, prog, tokenizer, prompt_str, max_new_tokens=10, device=device)
            
            # Add to Markdown
            md_lines.extend([
                f"### Prompt {p_idx + 1}",
                f"**Prompt**: `{question}`",
                f"- **Expected Answer**: `{expected}`",
                f"- **Base Model Output**: `{base_out}`",
                f"- **Regular AdamW Output**: `{adamw_out}`",
                f"- **Geometry CL Output**: `{cl_out}`",
                ""
            ])

    args.output_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Generated side-by-side prompt report at: {args.output_md_path}")

if __name__ == "__main__":
    main()
