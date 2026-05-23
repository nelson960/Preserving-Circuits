"""Capacity Threshold Sweep script.

Runs the real book continual learning benchmark under different model sizes and loads,
identifying the capacity threshold for standard sequential AdamW versus Latent-Geometry CL.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Config mapping for different model sizes to achieve target parameters (~embedding + transformer blocks)
MODEL_CONFIGS = {
    "0.25M": {"d_model": 64, "n_layers": 2, "n_heads": 2, "d_ff": 128},
    "0.5M": {"d_model": 96, "n_layers": 3, "n_heads": 3, "d_ff": 256},
    "1.0M": {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512},
    "2.0M": {"d_model": 192, "n_layers": 4, "n_heads": 4, "d_ff": 768}
}

def make_text_probe(words: list[str], max_words: int = 80) -> str:
    if not words:
        raise ValueError("Cannot create a text probe from an empty word list.")
    return " ".join(words[:max_words])


def run_command(cmd: list[str]) -> None:
    print(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

def setup_load_data(load_name: str, output_dir: Path) -> None:
    # Set up datasets based on load settings
    prepare_script = Path(__file__).parent / "prepare_real_book_benchmark.py"
    
    if load_name == "Load_1":
        # Load 1: 1 book, 5 chunks
        run_command([
            sys.executable, str(prepare_script),
            "--num-chunks", "5",
            "--output-dir", str(output_dir)
        ])
    elif load_name == "Load_2":
        # Load 2: 1 book, 10 chunks
        run_command([
            sys.executable, str(prepare_script),
            "--num-chunks", "10",
            "--output-dir", str(output_dir)
        ])
    elif load_name in ["Load_3", "Load_4", "Load_5"]:
        # Setup standard 5 chunk dataset as basis
        run_command([
            sys.executable, str(prepare_script),
            "--num-chunks", "5",
            "--output-dir", str(output_dir)
        ])
        
        # Load multi-book text and append chunks
        chunks_json_path = output_dir / "chunks.json"
        fact_probes_path = output_dir / "fact_probes.json"
        with open(chunks_json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not fact_probes_path.exists():
            raise FileNotFoundError(f"Expected fact probes at {fact_probes_path}.")
        with open(fact_probes_path, "r", encoding="utf-8") as f:
            fact_probes = json.load(f)
            
        if load_name in ["Load_3", "Load_4", "Load_5"]:
            # Load 3: 2 books. Append chunks from Alice in Wonderland
            alice_path = output_dir / "alice.txt"
            if not alice_path.exists():
                raise FileNotFoundError(f"Expected Alice text at {alice_path}.")
            alice_words = alice_path.read_text(encoding="utf-8").split()
            if len(alice_words) < 5:
                raise RuntimeError(f"Alice text is too short to split into 5 chunks: {len(alice_words)} words.")
            sz = len(alice_words) // 5
            for i in range(5):
                chunk_words = alice_words[i*sz:(i+1)*sz]
                chunk_id = f"chunk_alice_{i+1:02d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": " ".join(chunk_words),
                    "local_prompts": [],
                    "retention_prompts": [],
                    "composition_prompts": []
                })
                fact_probes[chunk_id] = [make_text_probe(chunk_words)]
                    
        if load_name in ["Load_4", "Load_5"]:
            # Load 4: 3 books. Append chunks from The Time Machine
            tm_path = output_dir / "time_machine.txt"
            if not tm_path.exists():
                raise FileNotFoundError(f"Expected Time Machine text at {tm_path}.")
            tm_words = tm_path.read_text(encoding="utf-8").split()
            if len(tm_words) < 5:
                raise RuntimeError(f"Time Machine text is too short to split into 5 chunks: {len(tm_words)} words.")
            sz = len(tm_words) // 5
            for i in range(5):
                chunk_words = tm_words[i*sz:(i+1)*sz]
                chunk_id = f"chunk_tm_{i+1:02d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": " ".join(chunk_words),
                    "local_prompts": [],
                    "retention_prompts": [],
                    "composition_prompts": []
                })
                fact_probes[chunk_id] = [make_text_probe(chunk_words)]
                    
        # Update retention and composition prompts for all chunks dynamically to be cumulative
        eval_prompts_path = output_dir / "eval_prompts.json"
        cross_chunk_prompts = []
        if not eval_prompts_path.exists():
            raise FileNotFoundError(f"Expected evaluation prompts at {eval_prompts_path}.")
        with open(eval_prompts_path, "r", encoding="utf-8") as f:
            eval_prompts = json.load(f)
        cross_chunk_prompts = eval_prompts["cross_chunk_prompts"]

        all_local_prompts = []
        for idx, chunk in enumerate(chunks):
            chunk["retention_prompts"] = [p for p in all_local_prompts]
            all_local_prompts.extend(chunk["local_prompts"])
            if idx >= len(chunks) // 2:
                chunk["composition_prompts"] = cross_chunk_prompts

        # Write modified chunks
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        with open(fact_probes_path, "w", encoding="utf-8") as f:
            json.dump(fact_probes, f, indent=2)

def run_sweep(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_results = {}
    
    # Scripts paths
    base_train_script = Path(__file__).parent / "train_base_model.py"
    adamw_script = Path(__file__).parent / "real_book_regular_adamw.py"
    cl_script = Path(__file__).parent / "real_book_geometry_cl.py"

    temp_data_dir = args.output_dir / "temp_sweep_data"
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    for model_size in ["0.25M", "0.5M", "1.0M", "2.0M"]:
        print(f"\n==================================================")
        print(f"Sweeping Model Size: {model_size}")
        print(f"==================================================")
        
        cfg = MODEL_CONFIGS[model_size]
        sweep_results[model_size] = {}
        
        for load_name in ["Load_1", "Load_2", "Load_3", "Load_4", "Load_5"]:
            print(f"\n--- Running Load setting: {load_name} ---")
            
            # 1. Setup data files for this load
            setup_load_data(load_name, temp_data_dir)
            
            # For Load 5 bottleneck, we simulate it by setting d_ff lower or fewer epochs
            epochs_count = 15 if load_name == "Load_5" else 30
            
            # 2. Train base model for this size
            base_model_path = temp_data_dir / f"base_model_{model_size}.pt"
            run_command([
                sys.executable, str(base_train_script),
                "--tokenizer-path", str(temp_data_dir / "tokenizer.json"),
                "--train-text_path", str(temp_data_dir / "alice.txt"),
                "--output-model-path", str(base_model_path),
                "--epochs", "2",
                "--d-model", str(cfg["d_model"]),
                "--n-layers", str(cfg["n_layers"]),
                "--n-heads", str(cfg["n_heads"]),
                "--d-ff", str(cfg["d_ff"]),
                "--device", args.device
            ])
            
            # 3. Run sequential AdamW baseline
            adamw_results_path = temp_data_dir / "adamw_results.json"
            if adamw_results_path.exists():
                adamw_results_path.unlink()
                
            run_command([
                sys.executable, str(adamw_script),
                "--base-model-path", str(base_model_path),
                "--tokenizer-path", str(temp_data_dir / "tokenizer.json"),
                "--chunks-path", str(temp_data_dir / "chunks.json"),
                "--output-model-path", str(temp_data_dir / "adamw_model.pt"),
                "--output-results-json", str(adamw_results_path),
                "--epochs-per-chunk", str(epochs_count),
                "--include-local-prompts-in-training",
                "--qa-loss-weight", str(args.qa_loss_weight),
                "--d-model", str(cfg["d_model"]),
                "--n-layers", str(cfg["n_layers"]),
                "--n-heads", str(cfg["n_heads"]),
                "--d-ff", str(cfg["d_ff"]),
                "--device", args.device
            ])
            
            # Load baseline results
            with open(adamw_results_path, "r", encoding="utf-8") as f:
                adamw_res = json.load(f)
            
            # 4. Run Latent-Geometry CL
            cl_results_path = temp_data_dir / "cl_results.json"
            if cl_results_path.exists():
                cl_results_path.unlink()
                
            run_command([
                sys.executable, str(cl_script),
                "--base-model-path", str(base_model_path),
                "--tokenizer-path", str(temp_data_dir / "tokenizer.json"),
                "--chunks-path", str(temp_data_dir / "chunks.json"),
                "--fact-probes-path", str(temp_data_dir / "fact_probes.json"),
                "--output-model-path", str(temp_data_dir / "cl_model.pt"),
                "--output-results-json", str(cl_results_path),
                "--operator-epochs", str(epochs_count),
                "--include-local-prompts-in-training",
                "--qa-loss-weight", str(args.qa_loss_weight),
                "--d-model", str(cfg["d_model"]),
                "--n-layers", str(cfg["n_layers"]),
                "--n-heads", str(cfg["n_heads"]),
                "--d-ff", str(cfg["d_ff"]),
                "--device", args.device
            ])
            
            with open(cl_results_path, "r", encoding="utf-8") as f:
                cl_res = json.load(f)
                
            # Get final retention accuracy for both loops
            final_adamw_retention = adamw_res[-1]["retention_accuracy"] if adamw_res else 0.0
            final_cl_retention = cl_res[-1]["retention_accuracy"] if cl_res else 0.0
            
            sweep_results[model_size][load_name] = {
                "regular_adamw_retention": final_adamw_retention,
                "geometry_cl_retention": final_cl_retention,
                "capacity_threshold_exceeded_adamw": final_adamw_retention < 0.90,
                "capacity_threshold_exceeded_cl": final_cl_retention < 0.90,
                "operator_count": cl_res[-1]["operator_count"] if cl_res else 0
            }
            
            print(f"AdamW Final Retention: {final_adamw_retention:.4f} (Exceeded: {final_adamw_retention < 0.90})")
            print(f"Geometry CL Final Retention: {final_cl_retention:.4f} (Exceeded: {final_cl_retention < 0.90})")

    # Save sweep results
    sweep_path = args.output_dir / "capacity-threshold-sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep completed! Results saved to {sweep_path}")

    # Clean up temp sweep data directory
    if temp_data_dir.exists() and not args.keep_temp_data:
        import shutil
        shutil.rmtree(temp_data_dir)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run model capacity threshold sweep.")
    parser.add_argument("--output-dir", type=Path, default=Path("model/analysis"), help="Output results directory.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--qa-loss-weight", type=float, default=5.0)
    parser.add_argument("--keep-temp-data", action="store_true")
    args = parser.parse_args()
    
    run_sweep(args)

if __name__ == "__main__":
    main()
