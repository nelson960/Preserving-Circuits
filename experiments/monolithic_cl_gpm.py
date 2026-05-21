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
        
        # GPM benefits from removing biases to prevent drift in unused directions
        self.fc1 = nn.Linear(3 * d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)
        
    def forward(self, h_a, h_b, task_token_idx):
        task_emb = self.W_E(task_token_idx)
        x = torch.cat([h_a, h_b, task_emb], dim=-1)
        # Capture activations for GPM subspace analysis if required
        self.last_fc1_input = x
        x_hid = F.relu(self.fc1(x))
        self.last_fc2_input = x_hid
        out = self.fc2(x_hid)
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

def compute_gpm_subspace(model, task_name, threshold=0.99):
    """
    Performs forward pass on the task dataset to collect activation inputs,
    computes SVD, and extracts basis vectors explaining `threshold` fraction of energy.
    """
    device = next(model.parameters()).device
    model.eval()
    
    ins, _ = get_task_dataset(task_name)
    op_a_t = torch.tensor(ins[:, 0], dtype=torch.long, device=device)
    op_b_t = torch.tensor(ins[:, 1], dtype=torch.long, device=device)
    tasks_t = torch.tensor(ins[:, 2], dtype=torch.long, device=device)
    
    with torch.no_grad():
        h_a = model.W_E(op_a_t)
        h_b = model.W_E(op_b_t)
        h_b = torch.where((tasks_t == task_to_token["COPY"]).unsqueeze(1), torch.zeros_like(h_b), h_b)
        _ = model(h_a, h_b, tasks_t)
        
        # Get captured activations
        act1 = model.last_fc1_input # [N, 3*d_model]
        act2 = model.last_fc2_input # [N, d_hidden]
        
    # Analyze act1
    U1, S1, V1 = torch.svd(act1)
    energy1 = torch.cumsum(S1**2, dim=0) / torch.sum(S1**2)
    k1 = torch.where(energy1 >= threshold)[0][0].item() + 1
    basis1 = V1[:, :k1] # [3*d_model, k1]
    
    # Analyze act2
    U2, S2, V2 = torch.svd(act2)
    energy2 = torch.cumsum(S2**2, dim=0) / torch.sum(S2**2)
    k2 = torch.where(energy2 >= threshold)[0][0].item() + 1
    basis2 = V2[:, :k2] # [d_hidden, k2]
    
    return basis1, basis2

def project_gradients(model, U_fc1, U_fc2):
    """
    Projects fc1 and fc2 weights gradients onto the orthogonal complement of the GPM spaces.
    """
    if U_fc1 is not None and model.fc1.weight.grad is not None:
        g = model.fc1.weight.grad
        # g is [d_hidden, 3*d_model]
        # U_fc1 is [3*d_model, K1]
        model.fc1.weight.grad.copy_(g - g @ (U_fc1 @ U_fc1.T))
        
    if U_fc2 is not None and model.fc2.weight.grad is not None:
        g = model.fc2.weight.grad
        # g is [d_model, d_hidden]
        # U_fc2 is [d_hidden, K2]
        model.fc2.weight.grad.copy_(g - g @ (U_fc2 @ U_fc2.T))

def train_monolithic_step_gpm(model, task_name, U_fc1, U_fc2, epochs=800, lambda_closure=2.0, lr=0.02):
    """
    Trains on the current task using GPM gradient projection to protect old tasks.
    W_E and W_U are frozen.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()), lr=lr)
    
    ins, tgts = get_task_dataset(task_name)
    op_a_t = torch.tensor(ins[:, 0], dtype=torch.long, device=device)
    op_b_t = torch.tensor(ins[:, 1], dtype=torch.long, device=device)
    tasks_t = torch.tensor(ins[:, 2], dtype=torch.long, device=device)
    targets_t = torch.tensor(tgts, dtype=torch.long, device=device)
    
    # Freeze representation manifold
    model.W_E.weight.requires_grad = False
    model.W_U.weight.requires_grad = False
    
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
            print(f"  [Error] Loss became NaN at epoch {epoch} for task {task_name}")
            break
            
        loss.backward()
        
        # GPM projection step: Modify gradients to be orthogonal to previous task activations
        project_gradients(model, U_fc1, U_fc2)
        
        optimizer.step()
        
        if epoch % 200 == 0:
            # Check accuracy during training to see if it's converging
            preds = torch.argmax(logits, dim=-1)
            acc = torch.mean((preds == targets_t).float()).item()
            print(f"    Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.2f}")

def update_gpm_bases(U_old, U_new):
    """
    Orthonormalizes the union of old and new subspace bases using QR decomposition.
    """
    if U_old is None:
        return U_new
    
    combined = torch.cat([U_old, U_new], dim=1)
    Q, _ = torch.linalg.qr(combined)
    return Q

def run_gpm_experiment(seed_count=5, threshold=0.99, lambda_closure=2.0, device="cpu"):
    stages = ["ADD", "MAX", "COPY", "MIN", "SUB"]
    comp_names = ["max_of_sum", "sum_of_max", "sub_of_sum", "max_of_min", "sum_of_copy"]
    
    all_accs = {task: [] for task in stages}
    all_comps = {comp: [] for comp in comp_names}
    all_drifts = []
    
    for seed in range(seed_count):
        set_seed(seed)
        model = MonolithicArithmeticOperator(d_model=32, d_hidden=128).to(device)
        
        # Pretrain digits embedding autoencoder and freeze them
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            all_toks = torch.arange(vocab_size, device=device)
            ae_logits = model.W_E(all_toks) @ model.W_E.weight.T
            loss = F.cross_entropy(ae_logits, all_toks)
            loss.backward()
            optimizer.step()
            
        # Freeze representation manifold
        model.W_E.weight.requires_grad = False
        model.W_U.weight.requires_grad = False
        
        # GPM memory projection spaces
        U_fc1 = None
        U_fc2 = None
        
        for idx, task in enumerate(stages):
            # Train task using current projected gradients
            train_monolithic_step_gpm(
                model=model,
                task_name=task,
                U_fc1=U_fc1,
                U_fc2=U_fc2,
                epochs=800,
                lambda_closure=lambda_closure,
                lr=0.02
            )
            
            # Compute new subspace activations and update projection basis
            new_b1, new_b2 = compute_gpm_subspace(model, task, threshold=threshold)
            U_fc1 = update_gpm_bases(U_fc1, new_b1)
            U_fc2 = update_gpm_bases(U_fc2, new_b2)
            
            if seed == 0:
                print(f"  [Seed 0] Post-Task {task}: U_fc1 rank = {U_fc1.shape[1]} / {3 * 32}, U_fc2 rank = {U_fc2.shape[1]} / 128")
            
        # Eval individual tasks
        accs = evaluate_tasks(model, stages)
        for t in stages:
            all_accs[t].append(accs[t])
            
        # Eval compositions
        comps_accs = []
        for comp in comp_names:
            acc, drift = evaluate_composition_recursive(model, comp)
            all_comps[comp].append(acc)
            comps_accs.append(acc)
            all_drifts.append(drift)
            
    # Compute aggregates
    avg_task_acc = np.mean([np.mean(all_accs[t]) for t in stages])
    avg_comp_acc = np.mean([np.mean(all_comps[c]) for c in comp_names])
    avg_drift = np.mean(all_drifts)
    
    print("\n" + "="*80)
    print(f"GPM BENCHMARK COMPLETED (Seeds={seed_count}, threshold={threshold})")
    print(f"Goal: Rehearsal-Free Continual Learning with Orthogonal Gradient Projection")
    print("="*80)
    
    print(f"{'Task / Composition':<26} | {'GPM Rehearsal-Free Accuracy':<36}")
    print("-" * 65)
    for t in stages:
        vals = all_accs[t]
        print(f"{t + ' accuracy':<26} | {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print("-" * 65)
    for comp in comp_names:
        vals = all_comps[comp]
        print(f"{comp + ' acc':<26} | {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print("-" * 65)
    print(f"{'Average Task Acc':<26} | {avg_task_acc:.4f}")
    print(f"{'Average Composition Acc':<26} | {avg_comp_acc:.4f}")
    print(f"{'manifold_drift':<26} | {avg_drift:.6f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_gpm_experiment(seed_count=5, threshold=0.99, lambda_closure=2.0, device=device)
