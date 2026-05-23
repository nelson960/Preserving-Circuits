"""Geometry-gated Continual Learning (CL) loop for real-book chunks.

The default mode uses explicit latent-geometry diagnostics to select reuse,
compose, update, or allocate. Learned action-policy mode remains available via
--policy-mode learned, but updates are still opt-in because destructive updates
are the failure mode this benchmark is meant to expose.
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
from models import DecoderTransformer, ClosedOperator, LearnedActionPolicy
from real_book_common import (
    format_qa_prompt,
    make_lm_sequences,
    make_qa_supervision,
    masked_cross_entropy,
    require_token_id,
    resolve_device,
)

ACTION_NAMES = ["reuse", "compose", "update", "allocate"]

def format_prompt(question: str) -> str:
    return format_qa_prompt(question)


def build_training_text(chunk: dict, include_local_prompts: bool) -> str:
    parts = [chunk["text"]]
    if include_local_prompts:
        for prompt in chunk["local_prompts"]:
            parts.append(f"{format_prompt(prompt['question'])}{prompt['answer']}")
    return "\n\n".join(parts)

def generate_greedy(model: DecoderTransformer, library: dict[str, ClosedOperator], program: tuple, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 8, device: torch.device = torch.device("cpu")) -> str:
    model.eval()
    encoded = tokenizer.encode(prompt).ids
    input_ids = list(encoded)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = min(len(input_ids), model.max_seq_len)
            input_tensor = torch.tensor([input_ids[-seq_len:]], dtype=torch.long, device=device)
            
            # Forward pass through frozen base model
            _, h = model(input_tensor)
            
            # Apply adaptation program
            h_adapted = eval_program(program, h, library)
            logits = model.lm_head(h_adapted)
            
            next_token = logits[0, -1].argmax(dim=-1).item()
            input_ids.append(next_token)
            
            eos_id = tokenizer.token_to_id("[EOS]")
            nl_id = tokenizer.token_to_id("\n")
            if next_token == eos_id or (nl_id is not None and next_token == nl_id):
                break
                
    generated_tokens = input_ids[len(encoded):]
    return tokenizer.decode(generated_tokens).strip()

def eval_program(program: tuple, h: torch.Tensor, library: dict[str, ClosedOperator]) -> torch.Tensor:
    if program[0] == "var":
        return h
    elif program[0] == "op":
        op_name = program[1]
        sub_h = eval_program(program[2][0], h, library)
        return library[op_name](sub_h)
    raise ValueError(f"Unknown program node: {program[0]}")

def compute_program_loss(model: DecoderTransformer, library: dict[str, ClosedOperator], program: tuple, tokens_x: torch.Tensor, tokens_y: torch.Tensor) -> torch.Tensor:
    _, h = model(tokens_x)
    h_adapted = eval_program(program, h, library)
    logits = model.lm_head(h_adapted)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens_y.reshape(-1))

def compute_probe_loss(model: DecoderTransformer, library: dict[str, ClosedOperator], program: tuple, probes: list[str], tokenizer: Tokenizer, device: torch.device) -> float:
    if not probes:
        return 0.0
    pad_id = require_token_id(tokenizer, "[PAD]")
    total_loss = 0.0
    sequence_count = 0
    for probe in probes:
        ids = tokenizer.encode(probe).ids
        if len(ids) <= 1:
            raise ValueError(f"Probe encoded to {len(ids)} token(s): {probe!r}")
        input_seqs, target_seqs = make_lm_sequences(ids, model.max_seq_len, pad_id)
        ids_x = torch.tensor(input_seqs, dtype=torch.long, device=device)
        ids_y = torch.tensor(target_seqs, dtype=torch.long, device=device)
        with torch.no_grad():
            loss = compute_program_loss(model, library, program, ids_x, ids_y)
            total_loss += loss.item() * len(input_seqs)
        sequence_count += len(input_seqs)
    if sequence_count == 0:
        raise RuntimeError("No probe sequences were produced.")
    return total_loss / sequence_count

def compute_closure_error(model: DecoderTransformer, h_adapted: torch.Tensor) -> float:
    # Distance to closest token embeddings. Use matmul instead of torch.cdist:
    # cdist is a common unsupported/slow path on MPS.
    embeddings = model.token_embedding.weight.detach()  # [V, D]
    flat_h = h_adapted.reshape(-1, h_adapted.shape[-1]).detach()  # [B*T, D]
    h_norm = (flat_h * flat_h).sum(dim=1, keepdim=True)
    e_norm = (embeddings * embeddings).sum(dim=1).unsqueeze(0)
    dist_sq = (h_norm + e_norm - 2.0 * flat_h @ embeddings.T).clamp_min(0.0)
    min_dists = dist_sq.min(dim=-1).values.sqrt()
    return min_dists.mean().item()

def compute_one_step_gradient_conflict(
    model: DecoderTransformer,
    operator: ClosedOperator,
    tokens_x: torch.Tensor,
    tokens_y: torch.Tensor,
    probes: list[str],
    tokenizer: Tokenizer,
    device: torch.device
) -> float:
    # Compute gradients w.r.t operator parameters for current loss and old probes
    operator.zero_grad(set_to_none=True)
    
    # Current loss gradient
    _, h = model(tokens_x)
    h_adapted = operator(h)
    logits = model.lm_head(h_adapted)
    loss_curr = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens_y.reshape(-1))
    loss_curr.backward()
    
    g_curr = []
    for p in operator.parameters():
        if p.grad is not None:
            g_curr.append(p.grad.flatten().clone())
    if not g_curr:
        return 1.0
    g_curr = torch.cat(g_curr)
    
    operator.zero_grad(set_to_none=True)
    
    # Old probes loss gradient
    pad_id = require_token_id(tokenizer, "[PAD]")
    total_probe_loss = torch.tensor(0.0, device=device)
    count = 0
    for probe in probes:
        ids = tokenizer.encode(probe).ids
        if len(ids) <= 1:
            raise ValueError(f"Probe encoded to {len(ids)} token(s): {probe!r}")
        input_seqs, target_seqs = make_lm_sequences(ids, model.max_seq_len, pad_id)
        ids_x = torch.tensor(input_seqs, dtype=torch.long, device=device)
        ids_y = torch.tensor(target_seqs, dtype=torch.long, device=device)
        _, h_p = model(ids_x)
        h_p_adapted = operator(h_p)
        logits_p = model.lm_head(h_p_adapted)
        loss_p = F.cross_entropy(logits_p.reshape(-1, logits_p.shape[-1]), ids_y.reshape(-1))
        total_probe_loss += loss_p * len(input_seqs)
        count += len(input_seqs)
        
    if count == 0:
        return 1.0
        
    (total_probe_loss / count).backward()
    g_old = []
    for p in operator.parameters():
        if p.grad is not None:
            g_old.append(p.grad.flatten().clone())
    g_old = torch.cat(g_old)
    
    # Cosine similarity
    cos = F.cosine_similarity(g_curr, g_old, dim=0).item()
    operator.zero_grad(set_to_none=True)
    return cos

def structural_gated_update(
    model: DecoderTransformer,
    operator: ClosedOperator,
    tokens_x: torch.Tensor,
    tokens_y: torch.Tensor,
    lr: float,
    epochs: int,
    device: torch.device,
    qa_inputs: torch.Tensor | None = None,
    qa_targets: torch.Tensor | None = None,
    qa_mask: torch.Tensor | None = None,
    qa_loss_weight: float = 0.0,
) -> None:
    first = operator.net[0]
    second = operator.net[2]
    
    for epoch in range(epochs):
        operator.zero_grad(set_to_none=True)
        _, h = model(tokens_x)
        h_adapted = operator(h)
        logits = model.lm_head(h_adapted)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens_y.reshape(-1))
        if qa_inputs is not None:
            if qa_targets is None or qa_mask is None:
                raise ValueError("qa_targets and qa_mask are required when qa_inputs is provided.")
            _, h_qa = model(qa_inputs)
            h_qa_adapted = operator(h_qa)
            qa_logits = model.lm_head(h_qa_adapted)
            loss = loss + qa_loss_weight * masked_cross_entropy(qa_logits, qa_targets, qa_mask)
        loss.backward()
        
        # Calculate need & risk gating signals for each hidden neuron
        # Activation norm per hidden neuron (after first linear layer and ReLU)
        with torch.no_grad():
            h_hidden = operator.net[1](first(h.detach())) # [batch, seq_len, hidden_dim]
            activation = h_hidden.abs().mean(dim=(0, 1)) # [hidden_dim]
        
        # Norm of incoming gradient and outgoing weights/gradients
        incoming_grad = torch.zeros(first.bias.shape, device=device)
        if first.weight.grad is not None:
            incoming_grad = incoming_grad + first.weight.grad.detach().abs().mean(dim=1)
        if first.bias.grad is not None:
            incoming_grad = incoming_grad + first.bias.grad.detach().abs()
            
        downstream_weight = second.weight.detach().norm(dim=0)
        
        downstream_grad = torch.zeros(first.bias.shape, device=device)
        if second.weight.grad is not None:
            downstream_grad = downstream_grad + second.weight.grad.detach().norm(dim=0)
        
        need = activation * (incoming_grad + downstream_grad)
        risk = activation * downstream_weight
        
        # Normalize
        need = need / (need.max() + 1e-8)
        risk = risk / (risk.max() + 1e-8)
        
        # High need, low risk
        gates = need * (1.0 - risk)
        
        with torch.no_grad():
            for n_idx, gate in enumerate(gates):
                g = gate.item()
                if first.weight.grad is not None:
                    first.weight[n_idx].add_(first.weight.grad[n_idx], alpha=-lr * g)
                if first.bias.grad is not None:
                    first.bias[n_idx].add_(first.bias.grad[n_idx], alpha=-lr * g)
                if second.weight.grad is not None:
                    second.weight[:, n_idx].add_(second.weight.grad[:, n_idx], alpha=-lr * g)
            if second.bias.grad is not None:
                second.bias.add_(second.bias.grad, alpha=-lr * gates.mean().item())


def flatten_feature_rows(feature_rows: list[list[float]], operator_count: int) -> list[float]:
    values: list[float] = []
    for row in feature_rows:
        values.extend(row)
    values.append(float(operator_count))
    return values


def select_diagnostic_action(
    feature_rows: list[list[float]],
    operator_count: int,
    allow_updates: bool,
    reuse_loss_threshold: float,
    compose_loss_threshold: float,
    update_loss_threshold: float,
    update_old_loss_threshold: float,
    update_grad_cos_threshold: float,
) -> tuple[int, str]:
    if operator_count == 0:
        return 3, "empty library"

    reuse_loss = feature_rows[0][0]
    compose_loss = feature_rows[1][0]
    update_loss = feature_rows[2][0]
    update_old_loss = feature_rows[2][1]
    update_grad_cos = feature_rows[2][4]

    if reuse_loss <= reuse_loss_threshold:
        return 0, f"reuse_loss={reuse_loss:.4f} <= {reuse_loss_threshold:.4f}"
    if compose_loss <= compose_loss_threshold:
        return 1, f"compose_loss={compose_loss:.4f} <= {compose_loss_threshold:.4f}"
    if allow_updates:
        update_allowed = (
            update_loss <= update_loss_threshold
            and update_old_loss <= update_old_loss_threshold
            and update_grad_cos >= update_grad_cos_threshold
        )
        if update_allowed:
            return (
                2,
                "update diagnostics passed: "
                f"loss={update_loss:.4f}, old_loss={update_old_loss:.4f}, cos={update_grad_cos:.4f}",
            )
    return (
        3,
        "no existing program met thresholds"
        if allow_updates
        else "no existing program met thresholds; updates disabled unless --allow-updates is set",
    )

def evaluate_qa_loss_and_acc(
    model: DecoderTransformer,
    library: dict[str, ClosedOperator],
    program: tuple,
    tokenizer: Tokenizer,
    prompt_dict: dict[str, str],
    device: torch.device
) -> dict[str, float | str]:
    question = prompt_dict["question"]
    expected_answer = prompt_dict["answer"].strip()
    
    prompt_str = format_prompt(question)
    prompt_ids = tokenizer.encode(prompt_str).ids
    answer_ids = tokenizer.encode(expected_answer).ids
    
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > model.max_seq_len:
        raise ValueError(
            f"QA evaluation example exceeds max_seq_len={model.max_seq_len}: "
            f"question={question!r}, answer={expected_answer!r}, tokens={len(full_ids)}"
        )
        
    input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        _, h = model(input_tensor)
        h_adapted = eval_program(program, h, library)
        logits = model.lm_head(h_adapted)
        
        answer_start_idx = len(prompt_ids) - 1
        answer_logits = logits[0, answer_start_idx:answer_start_idx + len(answer_ids)]
        answer_targets = target_tensor[0, answer_start_idx:answer_start_idx + len(answer_ids)]
        loss = F.cross_entropy(answer_logits.reshape(-1, logits.shape[-1]), answer_targets.reshape(-1)).item()
        pred_ids = answer_logits.argmax(dim=-1)
        token_accuracy = (pred_ids == answer_targets).float().mean().item()
        exact_answer_match = 1.0 if torch.equal(pred_ids, answer_targets) else 0.0
        
    output_str = generate_greedy(model, library, program, tokenizer, prompt_str, max_new_tokens=len(answer_ids) + 5, device=device)
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

def build_cl_features(
    model: DecoderTransformer,
    library: dict[str, ClosedOperator],
    tokens_x: torch.Tensor,
    tokens_y: torch.Tensor,
    probes: list[str],
    tokenizer: Tokenizer,
    device: torch.device
) -> list[list[float]]:
    # Compute candidate action diagnostics
    features_list = []
    
    # Actions: 0=reuse, 1=compose, 2=update, 3=allocate
    # We evaluate the best representative program for each action
    
    # Representative 0: Reuse (best existing single operator)
    best_reuse_op = None
    best_reuse_loss = float("inf")
    for name in library:
        prog = ("op", name, (("var", 0),))
        with torch.no_grad():
            loss = compute_program_loss(model, library, prog, tokens_x, tokens_y).item()
        if loss < best_reuse_loss:
            best_reuse_loss = loss
            best_reuse_op = name
            
    if best_reuse_op:
        prog = ("op", best_reuse_op, (("var", 0),))
        with torch.no_grad():
            _, h = model(tokens_x)
            h_ad = eval_program(prog, h, library)
            closure = compute_closure_error(model, h_ad)
            drift = F.mse_loss(h_ad, h).item()
        old_loss = compute_probe_loss(model, library, prog, probes, tokenizer, device)
        features_list.append([best_reuse_loss, old_loss, closure, drift, 1.0, 1.0, 0.0]) # reuse features
    else:
        features_list.append([10.0, 10.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    # Representative 1: Compose (best combination of two operators)
    best_compose_prog = None
    best_compose_loss = float("inf")
    for op1 in library:
        for op2 in library:
            prog = ("op", op1, (("op", op2, (("var", 0),)),))
            with torch.no_grad():
                loss = compute_program_loss(model, library, prog, tokens_x, tokens_y).item()
            if loss < best_compose_loss:
                best_compose_loss = loss
                best_compose_prog = prog
                
    if best_compose_prog:
        with torch.no_grad():
            _, h = model(tokens_x)
            h_ad = eval_program(best_compose_prog, h, library)
            closure = compute_closure_error(model, h_ad)
            drift = F.mse_loss(h_ad, h).item()
        old_loss = compute_probe_loss(model, library, best_compose_prog, probes, tokenizer, device)
        features_list.append([best_compose_loss, old_loss, closure, drift, 1.0, 1.0, 0.0]) # compose features
    else:
        features_list.append([10.0, 10.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    # Representative 2: Update (best single operator evaluated with one-step gradient conflict)
    best_update_op = None
    best_update_loss = float("inf")
    for name in library:
        prog = ("op", name, (("var", 0),))
        with torch.no_grad():
            loss = compute_program_loss(model, library, prog, tokens_x, tokens_y).item()
        if loss < best_update_loss:
            best_update_loss = loss
            best_update_op = name
            
    if best_update_op:
        prog = ("op", best_update_op, (("var", 0),))
        with torch.no_grad():
            _, h = model(tokens_x)
            h_ad = eval_program(prog, h, library)
            closure = compute_closure_error(model, h_ad)
            drift = F.mse_loss(h_ad, h).item()
        old_loss = compute_probe_loss(model, library, prog, probes, tokenizer, device)
        # Gradient conflict
        cos = compute_one_step_gradient_conflict(model, library[best_update_op], tokens_x, tokens_y, probes, tokenizer, device)
        features_list.append([best_update_loss, old_loss, closure, drift, cos, 1.0, 0.0]) # update features
    else:
        features_list.append([10.0, 10.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    # Representative 3: Allocate (features representing standard initialization)
    features_list.append([3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]) # allocate features
    
    return features_list

def train_action_policy(
    base_model: DecoderTransformer,
    tokenizer: Tokenizer,
    alice_chunks_path: Path,
    device: torch.device,
    operator_hidden_dim: int,
) -> LearnedActionPolicy:
    print("\nPre-training learned action policy on Alice in Wonderland...")
    with open(alice_chunks_path, "r", encoding="utf-8") as f:
        alice_data = json.load(f)
        
    library: dict[str, ClosedOperator] = {}
    collected_features = []
    collected_labels = []
    
    # Run a teacher-guided CL loop to collect training tokens
    for chunk_idx, c_data in enumerate(alice_data[:3]): # Use first 3 chunks of Alice
        tokens = tokenizer.encode(c_data["text"]).ids
        
        pad_id = require_token_id(tokenizer, "[PAD]")
        input_seqs, target_seqs = make_lm_sequences(tokens, base_model.max_seq_len, pad_id)
        inputs_t = torch.tensor(input_seqs, dtype=torch.long, device=device)
        targets_t = torch.tensor(target_seqs, dtype=torch.long, device=device)
        
        # Current past probes
        past_probes = []
        for c in alice_data[:chunk_idx]:
            past_probes.append(c["text"][:100]) # simple prefix probes
            
        # Compute features
        feats = build_cl_features(base_model, library, inputs_t, targets_t, past_probes, tokenizer, device)
        
        z_t = flatten_feature_rows(feats, len(library))
        
        # Teacher decision rule:
        # If library is empty -> must allocate
        # Else if current loss is low under reuse -> reuse
        # Else -> allocate or update depending on conflict
        if len(library) == 0:
            action_label = 3 # allocate
        else:
            best_reuse_loss = feats[0][0]
            if best_reuse_loss < 1.5:
                action_label = 0 # reuse
            elif feats[2][4] > 0.0: # high cosine similarity -> safe to update
                action_label = 2 # update
            else:
                action_label = 3 # allocate
                
        collected_features.append(z_t)
        collected_labels.append(action_label)
        
        # Materialize chosen action in the dummy run to advance states
        if action_label == 3: # allocate
            op_name = f"OP_ALICE_{len(library)}"
            op = ClosedOperator(base_model.d_model, operator_hidden_dim).to(device)
            # Short fit
            optimizer = torch.optim.Adam(op.parameters(), lr=0.01)
            for _ in range(5):
                optimizer.zero_grad()
                _, h = base_model(inputs_t)
                h_ad = op(h)
                logits = base_model.lm_head(h_ad)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets_t.reshape(-1))
                loss.backward()
                optimizer.step()
            library[op_name] = op
            
    if not collected_features:
        raise RuntimeError("No action-policy training examples were collected from Alice chunks.")

    # Train MLP policy
    policy = LearnedActionPolicy(input_dim=len(collected_features[0]), hidden_dim=64, output_dim=4).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    x = torch.tensor(collected_features, dtype=torch.float32, device=device)
    y = torch.tensor(collected_labels, dtype=torch.long, device=device)
    
    for epoch in range(100):
        optimizer.zero_grad()
        logits = policy(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
    print("Action policy trained successfully.")
    return policy

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Latent-Geometry CL loop on real book chunks.")
    parser.add_argument("--base-model-path", type=Path, default=Path("checkpoints/real_book/base_model.pt"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/real_book/tokenizer.json"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/real_book/chunks.json"))
    parser.add_argument("--fact-probes-path", type=Path, default=Path("data/real_book/fact_probes.json"))
    parser.add_argument("--output-model-path", type=Path, default=Path("checkpoints/real_book/latent_geometry_book_model.pt"))
    parser.add_argument("--output-results-json", type=Path, default=Path("model/analysis/real-book-geometry-cl.json"))
    parser.add_argument("--operator-epochs", type=int, default=30)
    parser.add_argument("--update-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--include-local-prompts-in-training", action="store_true")
    parser.add_argument("--qa-loss-weight", type=float, default=5.0)
    parser.add_argument("--policy-mode", choices=["diagnostic", "learned"], default="diagnostic")
    parser.add_argument("--allow-updates", action="store_true")
    parser.add_argument("--reuse-loss-threshold", type=float, default=1.0)
    parser.add_argument("--compose-loss-threshold", type=float, default=1.0)
    parser.add_argument("--update-loss-threshold", type=float, default=1.0)
    parser.add_argument("--update-old-loss-threshold", type=float, default=1.0)
    parser.add_argument("--update-grad-cos-threshold", type=float, default=0.25)
    parser.add_argument("--operator-hidden-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_results_json.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load resources
    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSON not found. Run prepare_real_book_benchmark.py first.")
    with open(args.chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    if not args.fact_probes_path.exists():
        raise FileNotFoundError(f"Fact probes not found.")
    with open(args.fact_probes_path, "r", encoding="utf-8") as f:
        fact_probes = json.load(f)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # 2. Load frozen base model
    base_model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len
    ).to(device)
    base_model.load_state_dict(torch.load(args.base_model_path, map_location=device))
    base_model.freeze()
    print("Loaded frozen base model weights.")

    # 3. Optional learned action policy on background book splits.
    temp_alice_json: Path | None = None
    policy: LearnedActionPolicy | None = None
    if args.policy_mode == "learned":
        alice_chunks_path = args.chunks_path.parent / "alice.txt"
        if not alice_chunks_path.exists():
            raise FileNotFoundError(
                f"Policy mode 'learned' requires background text at {alice_chunks_path}."
            )
        temp_alice_json = args.chunks_path.parent / "alice_chunks.json"
        alice_text = alice_chunks_path.read_text(encoding="utf-8")
        alice_words = alice_text.split()
        if len(alice_words) < 3:
            raise RuntimeError(f"Not enough Alice words to train action policy: {len(alice_words)}")
        sz = len(alice_words) // 3
        alice_chunks_data = [
            {"chunk_id": f"alice_0{i+1}", "text": " ".join(alice_words[i*sz:(i+1)*sz])}
            for i in range(3)
        ]
        with open(temp_alice_json, "w", encoding="utf-8") as f:
            json.dump(alice_chunks_data, f)
        policy = train_action_policy(
            base_model,
            tokenizer,
            temp_alice_json,
            device,
            operator_hidden_dim=args.operator_hidden_dim,
        )

    # 4. Geometry CL Loop
    library: dict[str, ClosedOperator] = {}
    task_to_program = {}
    step_results = []
    
    all_past_probes = []

    for chunk_idx, c_data in enumerate(chunks):
        chunk_id = c_data["chunk_id"]
        print(f"\n--- Processing Chunk {chunk_id} ({chunk_idx+1}/{len(chunks)}) ---")
        
        # Tokenize current training stream.
        training_text = build_training_text(c_data, args.include_local_prompts_in_training)
        tokens = tokenizer.encode(training_text).ids
        pad_id = require_token_id(tokenizer, "[PAD]")
        input_seqs, target_seqs = make_lm_sequences(tokens, base_model.max_seq_len, pad_id)
        inputs_t = torch.tensor(input_seqs, dtype=torch.long, device=device)
        targets_t = torch.tensor(target_seqs, dtype=torch.long, device=device)
        qa_supervision = None
        if args.include_local_prompts_in_training:
            qa_supervision = make_qa_supervision(c_data["local_prompts"], tokenizer, base_model.max_seq_len, pad_id)
            if qa_supervision is not None:
                qa_inputs_t, qa_targets_t, qa_mask_t = qa_supervision
                qa_inputs_t = qa_inputs_t.to(device)
                qa_targets_t = qa_targets_t.to(device)
                qa_mask_t = qa_mask_t.to(device)
        
        # Compute latent geometry state z_t
        feats = build_cl_features(base_model, library, inputs_t, targets_t, all_past_probes, tokenizer, device)
        z_t = flatten_feature_rows(feats, len(library))

        if args.policy_mode == "learned":
            if policy is None:
                raise RuntimeError("Internal error: learned policy mode selected but no policy was trained.")
            with torch.no_grad():
                logits = policy(torch.tensor([z_t], dtype=torch.float32, device=device))
                action_idx = logits.squeeze(0).argmax().item()
            if len(library) == 0:
                action_idx = 3
                action_reason = "empty library action mask"
            elif action_idx == 2 and not args.allow_updates:
                action_idx = 3
                action_reason = "learned policy selected update, but updates are disabled"
            else:
                action_reason = "learned policy argmax"
        else:
            action_idx, action_reason = select_diagnostic_action(
                feats,
                len(library),
                allow_updates=args.allow_updates,
                reuse_loss_threshold=args.reuse_loss_threshold,
                compose_loss_threshold=args.compose_loss_threshold,
                update_loss_threshold=args.update_loss_threshold,
                update_old_loss_threshold=args.update_old_loss_threshold,
                update_grad_cos_threshold=args.update_grad_cos_threshold,
            )

        action = ACTION_NAMES[action_idx]
        print(f"Action choice: {action.upper()} ({action_reason})")
        
        # Execute Chosen Action
        if action == "reuse":
            # Map chunk to the best existing operator
            best_reuse_op = None
            best_loss = float("inf")
            for name in library:
                prog = ("op", name, (("var", 0),))
                loss = compute_program_loss(base_model, library, prog, inputs_t, targets_t).item()
                if loss < best_loss:
                    best_loss = loss
                    best_reuse_op = name
            task_to_program[chunk_id] = ("op", best_reuse_op, (("var", 0),))
            
        elif action == "compose":
            # Map chunk to best composed program
            best_comp_prog = None
            best_loss = float("inf")
            for op1 in library:
                for op2 in library:
                    prog = ("op", op1, (("op", op2, (("var", 0),)),))
                    loss = compute_program_loss(base_model, library, prog, inputs_t, targets_t).item()
                    if loss < best_loss:
                        best_loss = loss
                        best_comp_prog = prog
            task_to_program[chunk_id] = best_comp_prog
            
        elif action == "update":
            # Update the best existing operator to cover new knowledge
            best_op = None
            best_loss = float("inf")
            for name in library:
                prog = ("op", name, (("var", 0),))
                loss = compute_program_loss(base_model, library, prog, inputs_t, targets_t).item()
                if loss < best_loss:
                    best_loss = loss
                    best_op = name
                    
            print(f"Updating operator: {best_op}")
            structural_gated_update(
                base_model,
                library[best_op],
                inputs_t,
                targets_t,
                lr=args.lr,
                epochs=args.update_epochs,
                device=device,
                qa_inputs=qa_inputs_t if qa_supervision is not None else None,
                qa_targets=qa_targets_t if qa_supervision is not None else None,
                qa_mask=qa_mask_t if qa_supervision is not None else None,
                qa_loss_weight=args.qa_loss_weight,
            )
            task_to_program[chunk_id] = ("op", best_op, (("var", 0),))
            
        elif action == "allocate":
            # Train a new operator module
            op_name = f"OP_OZ_{len(library)}"
            print(f"Allocating new operator: {op_name}")
            op = ClosedOperator(base_model.d_model, args.operator_hidden_dim).to(device)
            
            # Fine-tune new operator weights
            optimizer = torch.optim.AdamW(op.parameters(), lr=args.lr)
            for epoch in range(args.operator_epochs):
                optimizer.zero_grad()
                _, h = base_model(inputs_t)
                h_ad = op(h)
                logits = base_model.lm_head(h_ad)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets_t.reshape(-1))
                if qa_supervision is not None:
                    _, h_qa = base_model(qa_inputs_t)
                    h_qa_ad = op(h_qa)
                    qa_logits = base_model.lm_head(h_qa_ad)
                    loss = loss + args.qa_loss_weight * masked_cross_entropy(qa_logits, qa_targets_t, qa_mask_t)
                loss.backward()
                optimizer.step()
                
            library[op_name] = op
            task_to_program[chunk_id] = ("op", op_name, (("var", 0),))

        # Evaluate prompts using the chosen adaptation program
        print("Evaluating prompt retention and QA completion...")
        current_program = task_to_program[chunk_id]
        
        local_evals = []
        for p in c_data["local_prompts"]:
            local_evals.append(evaluate_qa_loss_and_acc(base_model, library, current_program, tokenizer, p, device))
            
        retention_evals = []
        for p in c_data["retention_prompts"]:
            # Find the chunk that originally contained this prompt
            matching_chunk_idx = None
            for idx, c in enumerate(chunks):
                if any(p["question"] == lp["question"] for lp in c["local_prompts"]):
                    matching_chunk_idx = idx
                    break
            if matching_chunk_idx is not None:
                old_chunk_id = chunks[matching_chunk_idx]["chunk_id"]
                if old_chunk_id not in task_to_program:
                    raise RuntimeError(f"No stored program for retained prompt source chunk {old_chunk_id}.")
                old_program = task_to_program[old_chunk_id]
            else:
                raise RuntimeError(f"Could not find source chunk for retained prompt: {p['question']!r}")
            retention_evals.append(evaluate_qa_loss_and_acc(base_model, library, old_program, tokenizer, p, device))
            
        composition_evals = []
        for p in c_data["composition_prompts"]:
            composition_evals.append(evaluate_qa_loss_and_acc(base_model, library, current_program, tokenizer, p, device))
            
        # Summarize step performance
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
            "action_chosen": action,
            "action_reason": action_reason,
            "diagnostic_features": {
                "reuse": feats[0],
                "compose": feats[1],
                "update": feats[2],
                "allocate": feats[3],
            },
            "operator_count": len(library),
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
        
        # Append this chunk's facts to past probes list
        if chunk_id not in fact_probes:
            raise KeyError(f"Fact probes JSON has no entry for chunk {chunk_id}.")
        all_past_probes.extend(fact_probes[chunk_id])

    # Clean up temp file
    if temp_alice_json is not None and temp_alice_json.exists():
        temp_alice_json.unlink()

    # Save final model state dict
    # Save the base model weights along with the adaptation library weights
    state = {
        "base_model": base_model.state_dict(),
        "library": {name: op.state_dict() for name, op in library.items()},
        "task_to_program": task_to_program,
        "operator_hidden_dim": args.operator_hidden_dim,
    }
    torch.save(state, args.output_model_path)
    print(f"Saved Latent-Geometry CL model state to {args.output_model_path}")

    # Write evaluation metrics
    with open(args.output_results_json, "w", encoding="utf-8") as f:
        json.dump(step_results, f, indent=2)
    print(f"Saved evaluation results JSON to {args.output_results_json}")

if __name__ == "__main__":
    main()
