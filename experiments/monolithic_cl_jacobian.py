import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

num_digits = 5
task_names = ["ADD", "MAX", "COPY", "MIN", "SUB"]
task_to_token = {name: num_digits + idx for idx, name in enumerate(task_names)}
vocab_size = num_digits + len(task_names)

def op_add(x, y): return (x + y) % 5
def op_max(x, y): return np.maximum(x, y)
def op_copy(x): return x
def op_min(x, y): return np.minimum(x, y)
def op_sub(x, y): return (x - y) % 5

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MonolithicArithmeticOperator(nn.Module):
    def __init__(self, d_model=32, d_hidden=128):
        super().__init__()
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.W_U = nn.Linear(d_model, num_digits, bias=False)
        self.W_U.weight = nn.Parameter(self.W_E.weight[:num_digits])
        
        self.fc1 = nn.Linear(3 * d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)
        
    def forward(self, h_a, h_b, task_token_idx):
        task_emb = self.W_E(task_token_idx)
        x = torch.cat([h_a, h_b, task_emb], dim=-1)
        out = self.fc2(F.relu(self.fc1(x)))
        return out

def get_task_dataset(task_name):
    inputs = []
    targets = []
    task_tok = task_to_token[task_name]
    
    if task_name == "COPY":
        for x in range(5):
            inputs.append([x, 0, task_tok])
            targets.append(op_copy(x))
    else:
        for x in range(5):
            for y in range(5):
                inputs.append([x, y, task_tok])
                if task_name == "ADD": targets.append(op_add(x, y))
                elif task_name == "MAX": targets.append(op_max(x, y))
                elif task_name == "MIN": targets.append(op_min(x, y))
                elif task_name == "SUB": targets.append(op_sub(x, y))
                    
    return np.array(inputs), np.array(targets)

def evaluate_tasks(model, trained_tasks):
    device = next(model.parameters()).device
    model.eval()
    
    accuracies = {}
    for task in trained_tasks:
        ins, tgts = get_task_dataset(task)
        op_a_t = torch.tensor(ins[:, 0], dtype=torch.long, device=device)
        op_b_t = torch.tensor(ins[:, 1], dtype=torch.long, device=device)
        tasks_t = torch.tensor(ins[:, 2], dtype=torch.long, device=device)
        tgts_t = torch.tensor(tgts, dtype=torch.long, device=device)
        
        with torch.no_grad():
            h_a = model.W_E(op_a_t)
            h_b = model.W_E(op_b_t)
            h_b = torch.where((tasks_t == task_to_token["COPY"]).unsqueeze(1), torch.zeros_like(h_b), h_b)
            
            h_out = model(h_a, h_b, tasks_t)
            logits = model.W_U(h_out)
            preds = torch.argmax(logits, dim=-1)
            acc = torch.mean((preds == tgts_t).float()).item()
            accuracies[task] = acc
            
    return accuracies

def evaluate_composition_recursive(model, comp_name):
    device = next(model.parameters()).device
    model.eval()
    
    d0_list, d1_list, d2_list = [], [], []
    for d0 in range(5):
        for d1 in range(5):
            for d2 in range(5):
                d0_list.append(d0)
                d1_list.append(d1)
                d2_list.append(d2)
    comp_indices = np.stack([d0_list, d1_list, d2_list], axis=1)
    
    if comp_name == "max_of_sum":
        targets = np.maximum((comp_indices[:, 0] + comp_indices[:, 1]) % 5, comp_indices[:, 2])
    elif comp_name == "sum_of_max":
        targets = (np.maximum(comp_indices[:, 0], comp_indices[:, 1]) + comp_indices[:, 2]) % 5
    elif comp_name == "sub_of_sum":
        targets = ((comp_indices[:, 0] + comp_indices[:, 1]) % 5 - comp_indices[:, 2]) % 5
    elif comp_name == "max_of_min":
        targets = np.maximum(np.minimum(comp_indices[:, 0], comp_indices[:, 1]), comp_indices[:, 2])
    elif comp_name == "sum_of_copy":
        targets = (comp_indices[:, 2] + comp_indices[:, 0]) % 5
        
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    add_tok = torch.tensor([task_to_token["ADD"]] * 125, dtype=torch.long, device=device)
    max_tok = torch.tensor([task_to_token["MAX"]] * 125, dtype=torch.long, device=device)
    min_tok = torch.tensor([task_to_token["MIN"]] * 125, dtype=torch.long, device=device)
    sub_tok = torch.tensor([task_to_token["SUB"]] * 125, dtype=torch.long, device=device)
    copy_tok = torch.tensor([task_to_token["COPY"]] * 125, dtype=torch.long, device=device)
    
    d0_emb = model.W_E(torch.tensor(comp_indices[:, 0], dtype=torch.long, device=device))
    d1_emb = model.W_E(torch.tensor(comp_indices[:, 1], dtype=torch.long, device=device))
    d2_emb = model.W_E(torch.tensor(comp_indices[:, 2], dtype=torch.long, device=device))
    
    with torch.no_grad():
        if comp_name == "max_of_sum":
            h_inter = model(d0_emb, d1_emb, add_tok)
            h_final = model(h_inter, d2_emb, max_tok)
        elif comp_name == "sum_of_max":
            h_inter = model(d0_emb, d1_emb, max_tok)
            h_final = model(h_inter, d2_emb, add_tok)
        elif comp_name == "sub_of_sum":
            h_inter = model(d0_emb, d1_emb, add_tok)
            h_final = model(h_inter, d2_emb, sub_tok)
        elif comp_name == "max_of_min":
            h_inter = model(d0_emb, d1_emb, min_tok)
            h_final = model(h_inter, d2_emb, max_tok)
        elif comp_name == "sum_of_copy":
            h_inter = model(d2_emb, torch.zeros_like(d2_emb), copy_tok)
            h_final = model(h_inter, d0_emb, add_tok)
            
        logits = model.W_U(h_final)
        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((preds == targets_t).float()).item()
        
        target_emb = model.W_E(targets_t)
        drift = torch.mean(torch.sum((h_final - target_emb)**2, dim=-1)).item()
        
    return acc, drift

def get_task_jacobian(model, task_name):
    """
    Computes the Jacobian matrix of the network output with respect to the active parameters.
    Returns: Jacobian matrix of shape [N_outputs, N_weights]
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Identify parameters to protect
    params = list(model.fc1.parameters()) + list(model.fc2.parameters())
    for p in params:
        p.requires_grad = True
        if p.grad is not None:
            p.grad.zero_()
            
    ins, _ = get_task_dataset(task_name)
    op_a_t = torch.tensor(ins[:, 0], dtype=torch.long, device=device)
    op_b_t = torch.tensor(ins[:, 1], dtype=torch.long, device=device)
    tasks_t = torch.tensor(ins[:, 2], dtype=torch.long, device=device)
    
    h_a = model.W_E(op_a_t)
    h_b = model.W_E(op_b_t)
    h_b = torch.where((tasks_t == task_to_token["COPY"]).unsqueeze(1), torch.zeros_like(h_b), h_b)
    
    h_out = model(h_a, h_b, tasks_t) # [M, d_model]
    h_flat = h_out.flatten()
    
    n_outputs = h_flat.shape[0]
    jacobian_rows = []
    
    for i in range(n_outputs):
        model.zero_grad()
        h_flat[i].backward(retain_graph=True)
        grad_flat = torch.cat([p.grad.flatten() for p in params])
        jacobian_rows.append(grad_flat.clone())
        
    J = torch.stack(jacobian_rows) # [N_outputs, N_weights]
    return J

def compute_orthonormal_basis(J_cum, svd_threshold=1e-4):
    """
    Computes the orthonormal basis of the row space of J_cum using SVD.
    """
    if J_cum is None:
        return None
    try:
        # J_cum has shape [N_outputs, N_weights]
        # J_cum.T has shape [N_weights, N_outputs]
        U, S, Vh = torch.linalg.svd(J_cum.T, full_matrices=False)
        rank = torch.sum(S > svd_threshold).item()
        Q = U[:, :rank] # [N_weights, rank]
        return Q
    except RuntimeError:
        print("  [Warning] SVD failed to converge.")
        return None

def project_gradient_jacobian(model, Q, params):
    """
    Projects the current gradients onto the null space using a precomputed orthonormal basis Q.
    """
    if Q is None:
        return
    
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.flatten())
        else:
            grads.append(torch.zeros_like(p).flatten())
    g = torch.cat(grads)
    
    # Project: g_proj = g - Q (Q^T g)
    Q_g = Q.T @ g
    g_proj = g - Q @ Q_g
    
    # Write back gradients
    offset = 0
    for p in params:
        numel = p.numel()
        p.grad.copy_(g_proj[offset:offset+numel].view_as(p))
        offset += numel

def train_monolithic_step_jacobian(model, task_name, J_cum, mode="static", epochs=600, lambda_closure=2.0, lr=0.01):
    """
    Trains on current task, projecting gradients onto the null space of previous task Jacobians.
    """
    device = next(model.parameters()).device
    params = list(model.fc1.parameters()) + list(model.fc2.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)
    
    ins, tgts = get_task_dataset(task_name)
    op_a_t = torch.tensor(ins[:, 0], dtype=torch.long, device=device)
    op_b_t = torch.tensor(ins[:, 1], dtype=torch.long, device=device)
    tasks_t = torch.tensor(ins[:, 2], dtype=torch.long, device=device)
    targets_t = torch.tensor(tgts, dtype=torch.long, device=device)
    
    # Freeze representation manifold
    model.W_E.weight.requires_grad = False
    model.W_U.weight.requires_grad = False
    
    # Initialize Q (orthonormal basis)
    Q = None
    current_J_cum = None
    if J_cum is not None:
        if mode == "static":
            current_J_cum = J_cum
            Q = compute_orthonormal_basis(current_J_cum)
        elif mode == "dynamic":
            # J_cum is a list of prior task names; compute initial Jacobian tensor and Q
            dynamic_J_list = []
            for prev_task_name in J_cum:
                dynamic_J_list.append(get_task_jacobian(model, prev_task_name))
            current_J_cum = torch.cat(dynamic_J_list, dim=0)
            Q = compute_orthonormal_basis(current_J_cum)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        h_a = model.W_E(op_a_t)
        h_b = model.W_E(op_b_t)
        h_b = torch.where((tasks_t == task_to_token["COPY"]).unsqueeze(1), torch.zeros_like(h_b), h_b)
        
        h_out = model(h_a, h_b, tasks_t)
        logits = model.W_U(h_out)
        ce_loss = F.cross_entropy(logits, targets_t)
        
        closure_loss = torch.tensor(0.0, device=device)
        if lambda_closure > 0.0:
            target_emb = model.W_E(targets_t)
            closure_loss = torch.mean(torch.sum((h_out - target_emb)**2, dim=-1))
            
        loss = ce_loss + lambda_closure * closure_loss
        
        if torch.isnan(loss):
            print(f"  [Warning] NaN loss encountered during training task {task_name} at epoch {epoch}.")
            break
            
        loss.backward()
        
        # Calculate raw grad norm for diagnostics
        raw_norm = 0.0
        for p in params:
            if p.grad is not None:
                raw_norm += p.grad.norm().item() ** 2
        raw_norm = raw_norm ** 0.5
        
        # If dynamic projection is selected, recompute the Jacobian matrix and Q every 50 epochs.
        if mode == "dynamic" and J_cum is not None:
            if epoch > 0 and epoch % 50 == 0:
                dynamic_J_list = []
                for prev_task_name in J_cum:
                    dynamic_J_list.append(get_task_jacobian(model, prev_task_name))
                current_J_cum = torch.cat(dynamic_J_list, dim=0)
                Q = compute_orthonormal_basis(current_J_cum)
            
        # Apply Jacobian null-space projection
        project_gradient_jacobian(model, Q, params)
        
        # Clip gradient norm to prevent explosion and stabilize updates
        torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
        
        # Calculate projected grad norm for diagnostics
        proj_norm = 0.0
        for p in params:
            if p.grad is not None:
                proj_norm += p.grad.norm().item() ** 2
        proj_norm = proj_norm ** 0.5
        
        if epoch < 10:
            print(f"    Epoch {epoch:03d} | Loss: {loss.item():.4f} | Raw Grad Norm: {raw_norm:.4f} | Proj Grad Norm: {proj_norm:.4f}")
        
        optimizer.step()
        
        if epoch % 200 == 0:
            preds = torch.argmax(logits, dim=-1)
            acc = torch.mean((preds == targets_t).float()).item()
            print(f"    Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.2f}")

def run_experiment(mode="naive", seed_count=5, device="cpu"):
    stages = ["ADD", "MAX", "COPY", "MIN", "SUB"]
    comp_names = ["max_of_sum", "sum_of_max", "sub_of_sum", "max_of_min", "sum_of_copy"]
    
    all_accs = {task: [] for task in stages}
    all_comps = {comp: [] for comp in comp_names}
    all_drifts = []
    
    for seed in range(seed_count):
        print(f"  [Mode: {mode.upper()}] Running seed {seed+1}/{seed_count}...")
        set_seed(seed)
        model = MonolithicArithmeticOperator(d_model=32, d_hidden=128).to(device)
        
        # Pretrain digits embedding autoencoder and freeze
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            all_toks = torch.arange(vocab_size, device=device)
            ae_logits = model.W_E(all_toks) @ model.W_E.weight.T
            loss = F.cross_entropy(ae_logits, all_toks)
            loss.backward()
            optimizer.step()
            
        model.W_E.weight.requires_grad = False
        model.W_U.weight.requires_grad = False
        
        # Keep track of previous task info
        previous_tasks = []
        J_cum = None
        
        for task in stages:
            # Prepare Jacobian summary
            if mode == "static" and len(previous_tasks) > 0:
                J_list = []
                for prev_t in previous_tasks:
                    J_list.append(get_task_jacobian(model, prev_t))
                J_cum = torch.cat(J_list, dim=0)
            elif mode == "dynamic" and len(previous_tasks) > 0:
                # Store task names to recompute online
                J_cum = list(previous_tasks)
                
            train_monolithic_step_jacobian(
                model=model,
                task_name=task,
                J_cum=J_cum,
                mode=mode,
                epochs=800,
                lambda_closure=2.0,
                lr=0.02
            )
            
            previous_tasks.append(task)
            
        # Eval tasks
        accs = evaluate_tasks(model, stages)
        for t in stages:
            all_accs[t].append(accs[t])
            
        # Eval compositions
        for comp in comp_names:
            acc, drift = evaluate_composition_recursive(model, comp)
            all_comps[comp].append(acc)
            all_drifts.append(drift)
            
    # Aggregates
    avg_task_acc = np.mean([np.mean(all_accs[t]) for t in stages])
    avg_comp_acc = np.mean([np.mean(all_comps[c]) for c in comp_names])
    avg_drift = np.mean(all_drifts)
    
    return {
        "avg_task_acc": avg_task_acc,
        "avg_comp_acc": avg_comp_acc,
        "avg_drift": avg_drift,
        "add_retention": np.mean(all_accs["ADD"]),
        "max_retention": np.mean(all_accs["MAX"]),
        "sub_retention": np.mean(all_accs["SUB"])
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_count = 3
    
    print("\n" + "="*80)
    print(f"RUNNING MONOLITHIC JACOBIAN PROJECTION BENCHMARK (Seeds={seed_count})")
    print("Goal: Evaluate functional projection against gradient locking & catastrophic forgetting.")
    print("="*80)
    
    modes = ["naive", "static", "dynamic"]
    
    print(f"{'Method / Projection Mode':<28} | {'Avg Task Acc':<12} | {'Avg Comp Acc':<12} | {'ADD Retention':<13} | {'SUB Retention':<13}")
    print("-" * 90)
    
    for mode in modes:
        res = run_experiment(mode=mode, seed_count=seed_count, device=device)
        mode_desc = {
            "naive": "Naïve Sequential (No Replay)",
            "static": "Static Jacobian Projection",
            "dynamic": "Dynamic Jacobian Projection"
        }[mode]
        print(f"{mode_desc:<28} | {res['avg_task_acc']:.4f}       | {res['avg_comp_acc']:.4f}       | {res['add_retention']:.4f}        | {res['sub_retention']:.4f}")
        
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
