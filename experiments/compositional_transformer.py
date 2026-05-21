import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.usage_score_ops import mean_std

# Vocabulary Mapping:
# 0, 1, 2, 3, 4 -> digits
# 5 -> MAX
# 6 -> MIN
MAX_TOKEN = 5
MIN_TOKEN = 6

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_2operand_dataset(op_token):
    sequences = []
    targets = []
    active_mask = []
    for a in range(5):
        for b in range(5):
            sequences.append([a, b, op_token])
            tgt = max(a, b) if op_token == MAX_TOKEN else min(a, b)
            targets.append([0, 0, tgt])
            active_mask.append([False, False, True])
    return np.array(sequences), np.array(targets), np.array(active_mask)

def get_3operand_dataset(op1_token, op2_token):
    sequences = []
    targets = []
    active_mask = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                sequences.append([a, b, op1_token, c, op2_token])
                y1 = max(a, b) if op1_token == MAX_TOKEN else min(a, b)
                y2 = max(y1, c) if op2_token == MAX_TOKEN else min(y1, c)
                targets.append([0, 0, y1, 0, y2])
                active_mask.append([False, False, True, False, True])
    return np.array(sequences), np.array(targets), np.array(active_mask)


class CompositionalTransformer(nn.Module):
    def __init__(self, vocab_size=7, d_model=16, num_layers=2, num_heads=2, d_head=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_head = d_head
        
        # Shared parameters across all layers (recurrent transformer)
        self.W_E = nn.Parameter(torch.empty(vocab_size, d_model))
        self.W_Q = nn.Parameter(torch.empty(num_heads, d_model, d_head))
        self.W_K = nn.Parameter(torch.empty(num_heads, d_model, d_head))
        self.W_V = nn.Parameter(torch.empty(num_heads, d_model, d_head))
        self.W_O = nn.Parameter(torch.empty(num_heads, d_head, d_model))
        self.W_U = nn.Parameter(torch.empty(d_model, vocab_size))
        
        # Init weights
        nn.init.normal_(self.W_E, mean=0.0, std=1.0)
        nn.init.normal_(self.W_Q, mean=0.0, std=0.25)
        nn.init.normal_(self.W_K, mean=0.0, std=0.25)
        nn.init.normal_(self.W_V, mean=0.0, std=0.25)
        nn.init.normal_(self.W_O, mean=0.0, std=0.25)
        nn.init.normal_(self.W_U, mean=0.0, std=0.25)


def forward_transformer(model, tokens, local_mask=True):
    B, T = tokens.shape
    device = tokens.device
    X = model.W_E[tokens] # (B, T, d_model)
    
    layer_outputs = []
    
    # Local mask: only attend to t-1 and t-2
    rows = torch.arange(T, device=device).view(T, 1)
    cols = torch.arange(T, device=device).view(1, T)
    if local_mask:
        mask = (cols == rows - 1) | (cols == rows - 2)
    else:
        mask = cols <= rows
    attn_mask = torch.zeros(T, T, device=device)
    attn_mask[~mask] = -1e9
    
    # has_digit tracks which positions contain valid digit representations
    has_digit = (tokens < 5)
    
    X_l = X
    for l in range(model.num_layers):
        O_sum = torch.zeros_like(X_l)
        
        # Build the gate mask for each head h, batch b, position t
        gate_mask = torch.zeros(model.num_heads, B, T, 1, device=device)
        next_has_digit = has_digit.clone()
        
        for t in range(2, T):
            # Operands ready in current layer
            ready_now = has_digit[:, t-2] & has_digit[:, t-1]
            
            # Operands ready in previous layer
            if l > 0:
                ready_prev = prev_has_digit[:, t-2] & prev_has_digit[:, t-1]
            else:
                ready_prev = torch.zeros(B, dtype=torch.bool, device=device)
                
            # Only execute operation in the first layer where operands are ready
            active = ready_now & ~ready_prev
            
            for h in range(model.num_heads):
                op_token = 5 if h == 0 else 6
                head_active = (tokens[:, t] == op_token) & active
                gate_mask[h, :, t, 0] = head_active.float()
                next_has_digit[:, t] = next_has_digit[:, t] | head_active
                
        for h in range(model.num_heads):
            Q = X_l @ model.W_Q[h] # (B, T, d_head)
            K = X_l @ model.W_K[h] # (B, T, d_head)
            V = X_l @ model.W_V[h] # (B, T, d_head)
            
            S = (Q @ K.transpose(-2, -1)) / np.sqrt(model.d_head) # (B, T, T)
            S = S + attn_mask
            A = F.softmax(S, dim=-1) # (B, T, T)
            
            Z = A @ V # (B, T, d_head)
            O_h = Z @ model.W_O[h] # (B, T, d_model)
            
            O_sum += O_h * gate_mask[h]
            
        X_l = X_l + O_sum
        layer_outputs.append(X_l)
        prev_has_digit = has_digit
        has_digit = next_has_digit
        
    logits = X_l @ model.W_U
    return logits, layer_outputs


def pretrain_embeddings(model, epochs=400, lr=0.01, margin=2.0, sep_weight=0.5):
    optimizer = torch.optim.Adam([model.W_E, model.W_U], lr=lr)
    num_digits = 5
    y = torch.arange(num_digits, device=model.W_E.device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        W_E_digits = model.W_E[:num_digits]
        W_U_digits = model.W_U[:, :num_digits]
        logits = W_E_digits @ W_U_digits
        
        ce_loss = F.cross_entropy(logits, y)
        
        sep_loss = torch.tensor(0.0, device=model.W_E.device)
        for i in range(num_digits):
            for j in range(num_digits):
                if i == j:
                    continue
                diff = model.W_E[i] - model.W_E[j]
                dist_sq = torch.sum(diff**2)
                if dist_sq < margin:
                    sep_loss += (margin - dist_sq)
                    
        loss = ce_loss + sep_weight * sep_loss
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        W_E_digits = model.W_E[:num_digits]
        W_U_digits = model.W_U[:, :num_digits]
        logits = W_E_digits @ W_U_digits
        preds = torch.argmax(logits, dim=1)
        acc = torch.mean((preds == y).float()).item()
        print(f"  [Embed Pretrain] Final Acc = {acc:.3f}, CE Loss = {ce_loss.item():.4f}, Sep Loss = {sep_loss.item():.4f}")


def train_regimen(model, lambda_closure: float, epochs: int = 800) -> dict:
    device = next(model.parameters()).device
    
    max_seq_np, max_tgt_np, max_mask_np = get_2operand_dataset(MAX_TOKEN)
    min_seq_np, min_tgt_np, min_mask_np = get_2operand_dataset(MIN_TOKEN)
    
    max_seq = torch.tensor(max_seq_np, dtype=torch.long, device=device)
    max_tgt = torch.tensor(max_tgt_np, dtype=torch.long, device=device)
    max_mask = torch.tensor(max_mask_np, dtype=torch.bool, device=device)
    
    min_seq = torch.tensor(min_seq_np, dtype=torch.long, device=device)
    min_tgt = torch.tensor(min_tgt_np, dtype=torch.long, device=device)
    min_mask = torch.tensor(min_mask_np, dtype=torch.bool, device=device)
    
    max_max_seq, max_max_tgt, _ = get_3operand_dataset(MAX_TOKEN, MAX_TOKEN)
    max_min_seq, max_min_tgt, _ = get_3operand_dataset(MAX_TOKEN, MIN_TOKEN)
    min_max_seq, min_max_tgt, _ = get_3operand_dataset(MIN_TOKEN, MAX_TOKEN)
    min_min_seq, min_min_tgt, _ = get_3operand_dataset(MIN_TOKEN, MIN_TOKEN)
    
    # Pre-train embeddings
    pretrain_embeddings(model, epochs=400, lr=0.01)
    
    # Phase 1: Train Head 0 (MAX) and W_E[5:]
    opt_p1 = torch.optim.Adam([model.W_E, model.W_Q, model.W_K, model.W_V, model.W_O], lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        opt_p1.zero_grad()
        logits, layer_outputs = forward_transformer(model, max_seq)
        
        active_logits = logits[max_mask]
        active_targets = max_tgt[max_mask]
        ce_loss = F.cross_entropy(active_logits, active_targets)
        
        closure_loss = torch.tensor(0.0, device=device)
        if lambda_closure > 0.0:
            target_emb = model.W_E[max_tgt]
            for X_l in layer_outputs:
                diff = X_l - target_emb
                diff[~max_mask] = 0.0
                closure_loss += torch.mean(torch.sum(diff**2, dim=-1))
                
        loss = ce_loss + lambda_closure * closure_loss
        loss.backward()
        
        # Gradient filter: freeze W_E[:5] and Head 1 (MIN)
        with torch.no_grad():
            if model.W_E.grad is not None:
                model.W_E.grad[:5] = 0.0
            if model.W_Q.grad is not None:
                model.W_Q.grad[1] = 0.0
            if model.W_K.grad is not None:
                model.W_K.grad[1] = 0.0
            if model.W_V.grad is not None:
                model.W_V.grad[1] = 0.0
            if model.W_O.grad is not None:
                model.W_O.grad[1] = 0.0
                
        opt_p1.step()
        if epoch % 100 == 0:
            print(f"  P1 - Epoch {epoch:03d}: CE = {ce_loss.item():.4f}, Closure = {closure_loss.item():.4f}")
            
    p1_max_acc = evaluate_accuracy(model, max_seq_np, max_tgt_np, max_mask_np)
    print(f"--- Completed Phase 1. MAX Accuracy: {p1_max_acc:.3f} ---")
    
    # Phase 2: Train Head 1 (MIN) sequentially
    opt_p2 = torch.optim.Adam([model.W_Q, model.W_K, model.W_V, model.W_O], lr=0.01)
    for epoch in range(epochs):
        model.train()
        opt_p2.zero_grad()
        logits, layer_outputs = forward_transformer(model, min_seq)
        
        active_logits = logits[min_mask]
        active_targets = min_tgt[min_mask]
        ce_loss = F.cross_entropy(active_logits, active_targets)
        
        closure_loss = torch.tensor(0.0, device=device)
        if lambda_closure > 0.0:
            target_emb = model.W_E[min_tgt]
            for X_l in layer_outputs:
                diff = X_l - target_emb
                diff[~min_mask] = 0.0
                closure_loss += torch.mean(torch.sum(diff**2, dim=-1))
                
        loss = ce_loss + lambda_closure * closure_loss
        loss.backward()
        
        # Gradient filter: freeze Head 0 (MAX)
        with torch.no_grad():
            if model.W_Q.grad is not None:
                model.W_Q.grad[0] = 0.0
            if model.W_K.grad is not None:
                model.W_K.grad[0] = 0.0
            if model.W_V.grad is not None:
                model.W_V.grad[0] = 0.0
            if model.W_O.grad is not None:
                model.W_O.grad[0] = 0.0
                
        opt_p2.step()
        if epoch % 100 == 0:
            print(f"  P2 - Epoch {epoch:03d}: CE = {ce_loss.item():.4f}, Closure = {closure_loss.item():.4f}")
            
    # Phase 3: Evaluate everything
    max_2op = evaluate_accuracy(model, max_seq_np, max_tgt_np, max_mask_np)
    min_2op = evaluate_accuracy(model, min_seq_np, min_tgt_np, min_mask_np)
    
    mm_int, mm_fin = evaluate_3operand_detailed(model, max_max_seq, max_max_tgt)
    mmn_int, mmn_fin = evaluate_3operand_detailed(model, max_min_seq, max_min_tgt)
    mnm_int, mnm_fin = evaluate_3operand_detailed(model, min_max_seq, min_max_tgt)
    mnmn_int, mnmn_fin = evaluate_3operand_detailed(model, min_min_seq, min_min_tgt)
    
    return {
        "max_2op": max_2op,
        "min_2op": min_2op,
        "max_max_int": mm_int,
        "max_max_fin": mm_fin,
        "max_min_int": mmn_int,
        "max_min_fin": mmn_fin,
        "min_max_int": mnm_int,
        "min_max_fin": mnm_fin,
        "min_min_int": mnmn_int,
        "min_min_fin": mnmn_fin
    }


def evaluate_accuracy(model, sequences_np, targets_np, active_mask_np):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        sequences = torch.tensor(sequences_np, dtype=torch.long, device=device)
        logits, _ = forward_transformer(model, sequences)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        
        active_indices = np.where(active_mask_np)
        active_preds = preds[active_indices]
        active_targets = targets_np[active_indices]
        return float(np.mean(active_preds == active_targets))


def evaluate_3operand_detailed(model, sequences_np, targets_np):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        sequences = torch.tensor(sequences_np, dtype=torch.long, device=device)
        logits, _ = forward_transformer(model, sequences)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        
        acc_inter = float(np.mean(preds[:, 2] == targets_np[:, 2]))
        acc_final = float(np.mean(preds[:, 4] == targets_np[:, 4]))
        return acc_inter, acc_final


def run_transformer_benchmark(seed_count: int) -> None:
    print("\n=================================================================")
    print("RUNNING COMPOSITION CONTEXT-ROUTING 2-LAYER RECURRENT TRANSFORMER BENCHMARK")
    print(f"Seeds: {seed_count}, d_model: 16, num_layers: 2, num_heads: 2")
    print("=================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    keys = [
        "max_2op", "min_2op",
        "max_max_int", "max_max_fin",
        "max_min_int", "max_min_fin",
        "min_max_int", "min_max_fin",
        "min_min_int", "min_min_fin"
    ]
    
    results_closed = {k: [] for k in keys}
    results_ablation = {k: [] for k in keys}
    
    for seed in range(seed_count):
        # 1. Closed Latent Model (with closure loss)
        set_seed(seed)
        model_closed = CompositionalTransformer(vocab_size=7, d_model=16, num_layers=2, num_heads=2, d_head=8).to(device)
        res_c = train_regimen(model_closed, lambda_closure=10.0, epochs=800)
        for k in keys:
            results_closed[k].append(res_c[k])
            
        # 2. Ablation Model (no closure loss)
        set_seed(seed)
        model_ablation = CompositionalTransformer(vocab_size=7, d_model=16, num_layers=2, num_heads=2, d_head=8).to(device)
        res_a = train_regimen(model_ablation, lambda_closure=0.0, epochs=800)
        for k in keys:
            results_ablation[k].append(res_a[k])
            
    print("\n=================================================================")
    print("TRANSFORMER COMPOSITION SUMMARY (MEAN +/- STD over seeds)")
    print("=================================================================")
    
    def format_metric(r_dict, key):
        m, s = mean_std(r_dict[key])
        return f"{m:.3f}+/-{s:.3f}"
        
    print(f"{'Metric / Task Evaluated':<32} | {'Ablation (no closure)':<22} | {'Closed Latent (ours)':<22}")
    print("-" * 83)
    
    # 2-operand tasks
    print(f"{'2-Operand MAX':<32} | {format_metric(results_ablation, 'max_2op'):<22} | {format_metric(results_closed, 'max_2op'):<22}")
    print(f"{'2-Operand MIN':<32} | {format_metric(results_ablation, 'min_2op'):<22} | {format_metric(results_closed, 'min_2op'):<22}")
    print("-" * 83)
    
    # 3-operand Intermediate steps (making sure intermediate representation computes correctly)
    print(f"{'MAX-MAX (Intermediate MAX)':<32} | {format_metric(results_ablation, 'max_max_int'):<22} | {format_metric(results_closed, 'max_max_int'):<22}")
    print(f"{'MAX-MIN (Intermediate MAX)':<32} | {format_metric(results_ablation, 'max_min_int'):<22} | {format_metric(results_closed, 'max_min_int'):<22}")
    print(f"{'MIN-MAX (Intermediate MIN)':<32} | {format_metric(results_ablation, 'min_max_int'):<22} | {format_metric(results_closed, 'min_max_int'):<22}")
    print(f"{'MIN-MIN (Intermediate MIN)':<32} | {format_metric(results_ablation, 'min_min_int'):<22} | {format_metric(results_closed, 'min_min_int'):<22}")
    print("-" * 83)
    
    # 3-operand Final steps (composition!)
    print(f"{'MAX-MAX (Composition)':<32} | {format_metric(results_ablation, 'max_max_fin'):<22} | {format_metric(results_closed, 'max_max_fin'):<22}")
    print(f"{'MAX-MIN (Composition)':<32} | {format_metric(results_ablation, 'max_min_fin'):<22} | {format_metric(results_closed, 'max_min_fin'):<22}")
    print(f"{'MIN-MAX (Composition)':<32} | {format_metric(results_ablation, 'min_max_fin'):<22} | {format_metric(results_closed, 'min_max_fin'):<22}")
    print(f"{'MIN-MIN (Composition)':<32} | {format_metric(results_ablation, 'min_min_fin'):<22} | {format_metric(results_closed, 'min_min_fin'):<22}")
    print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args()
    
    run_transformer_benchmark(args.seed_count if args.multi_seed else 1)
