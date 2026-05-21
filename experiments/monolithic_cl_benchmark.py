import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary definition for digits (0-4) and tasks
num_digits = 5
task_names = ["ADD", "MAX", "COPY", "MIN", "SUB"]

# Task tokens mapping
task_to_token = {name: num_digits + idx for idx, name in enumerate(task_names)}
vocab_size = num_digits + len(task_names)

# Arithmetic functions
def op_add(x, y): return (x + y) % 5
def op_max(x, y): return np.maximum(x, y)
def op_copy(x): return x
def op_min(x, y): return np.minimum(x, y)
def op_sub(x, y): return (x - y) % 5

# Set seed function
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MonolithicArithmeticOperator(nn.Module):
    """
    A single monolithic neural network that processes all tasks.
    It takes two hidden states (h_a, h_b) and the task token, and computes the output.
    For unary operations, h_b is a zero vector.
    """
    def __init__(self, d_model=32, d_hidden=128):
        super().__init__()
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.W_U = nn.Linear(d_model, num_digits, bias=False)
        
        # Tie digit unembedding weight to digit embedding weight
        self.W_U.weight = nn.Parameter(self.W_E.weight[:num_digits])
        
        self.fc1 = nn.Linear(3 * d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        
    def forward(self, h_a, h_b, task_token_idx):
        task_emb = self.W_E(task_token_idx)
        x = torch.cat([h_a, h_b, task_emb], dim=-1)
        out = self.fc2(F.relu(self.fc1(x)))
        return out

def get_task_dataset(task_name):
    # Generates all combinations (5x5 for binary, 5 for unary)
    inputs = []
    targets = []
    task_tok = task_to_token[task_name]
    
    if task_name == "COPY":
        for x in range(5):
            inputs.append([x, 0, task_tok]) # 0 is dummy second operand
            targets.append(op_copy(x))
    else:
        for x in range(5):
            for y in range(5):
                inputs.append([x, y, task_tok])
                if task_name == "ADD":
                    targets.append(op_add(x, y))
                elif task_name == "MAX":
                    targets.append(op_max(x, y))
                elif task_name == "MIN":
                    targets.append(op_min(x, y))
                elif task_name == "SUB":
                    targets.append(op_sub(x, y))
                    
    return np.array(inputs), np.array(targets)

def train_monolithic_step(model, active_tasks, epochs=400, lambda_closure=10.0, lr=0.01):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    all_op_a = []
    all_op_b = []
    all_tasks = []
    all_targets = []
    
    for t in active_tasks:
        ins, tgts = get_task_dataset(t)
        all_op_a.append(ins[:, 0])
        all_op_b.append(ins[:, 1])
        all_tasks.append(ins[:, 2])
        all_targets.append(tgts)
        
    op_a_t = torch.tensor(np.concatenate(all_op_a), dtype=torch.long, device=device)
    op_b_t = torch.tensor(np.concatenate(all_op_b), dtype=torch.long, device=device)
    tasks_t = torch.tensor(np.concatenate(all_tasks), dtype=torch.long, device=device)
    targets_t = torch.tensor(np.concatenate(all_targets), dtype=torch.long, device=device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        h_a = model.W_E(op_a_t)
        # Handle dummy zero operand for COPY
        h_b = model.W_E(op_b_t)
        h_b = torch.where((tasks_t == task_to_token["COPY"]).unsqueeze(1), torch.zeros_like(h_b), h_b)
        
        h_out = model(h_a, h_b, tasks_t)
        
        # Unembedding output
        logits = model.W_U(h_out)
        ce_loss = F.cross_entropy(logits, targets_t)
        
        closure_loss = torch.tensor(0.0, device=device)
        if lambda_closure > 0.0:
            target_emb = model.W_E(targets_t)
            closure_loss = torch.mean(torch.sum((h_out - target_emb)**2, dim=-1))
            
        loss = ce_loss + lambda_closure * closure_loss
        loss.backward()
        optimizer.step()

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
    
    # Generate composition test indices (5x5x5 = 125 samples)
    d0_list, d1_list, d2_list = [], [], []
    for d0 in range(5):
        for d1 in range(5):
            for d2 in range(5):
                d0_list.append(d0)
                d1_list.append(d1)
                d2_list.append(d2)
    comp_indices = np.stack([d0_list, d1_list, d2_list], axis=1)
    
    # ground truth targets
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
    
    # Task tokens
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
            # ADD(d0, d1)
            h_inter = model(d0_emb, d1_emb, add_tok)
            # MAX(h_inter, d2)
            h_final = model(h_inter, d2_emb, max_tok)
        elif comp_name == "sum_of_max":
            # MAX(d0, d1)
            h_inter = model(d0_emb, d1_emb, max_tok)
            # ADD(h_inter, d2)
            h_final = model(h_inter, d2_emb, add_tok)
        elif comp_name == "sub_of_sum":
            # ADD(d0, d1)
            h_inter = model(d0_emb, d1_emb, add_tok)
            # SUB(h_inter, d2)
            h_final = model(h_inter, d2_emb, sub_tok)
        elif comp_name == "max_of_min":
            # MIN(d0, d1)
            h_inter = model(d0_emb, d1_emb, min_tok)
            # MAX(h_inter, d2)
            h_final = model(h_inter, d2_emb, max_tok)
        elif comp_name == "sum_of_copy":
            # COPY(d2)
            h_inter = model(d2_emb, torch.zeros_like(d2_emb), copy_tok)
            # ADD(h_inter, d0)
            h_final = model(h_inter, d0_emb, add_tok)
            
        logits = model.W_U(h_final)
        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((preds == targets_t).float()).item()
        
        target_emb = model.W_E(targets_t)
        drift = torch.mean(torch.sum((h_final - target_emb)**2, dim=-1)).item()
        
    return acc, drift

def run_monolithic_cl_benchmark(seed_count=10, device="cpu"):
    stages = ["ADD", "MAX", "COPY", "MIN", "SUB"]
    
    results = {
        "ADD_acc": [], "MAX_acc": [], "COPY_acc": [], "MIN_acc": [], "SUB_acc": [],
        "max_of_sum_acc": [], "sum_of_max_acc": [], "sub_of_sum_acc": [],
        "max_of_min_acc": [], "sum_of_copy_acc": [],
        "avg_comp": [], "drift": []
    }
    
    for seed in range(seed_count):
        set_seed(seed)
        model = MonolithicArithmeticOperator(d_model=32, d_hidden=128).to(device)
        
        # Pretrain digits embedding autoencoder
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            all_toks = torch.arange(vocab_size, device=device)
            ae_logits = model.W_E(all_toks) @ model.W_E.weight.T
            loss = F.cross_entropy(ae_logits, all_toks)
            loss.backward()
            optimizer.step()
            
        active_tasks = []
        for task in stages:
            active_tasks.append(task)
            train_monolithic_step(model, active_tasks, epochs=800, lambda_closure=2.0, lr=0.01)
            
        # Eval individual tasks
        accs = evaluate_tasks(model, stages)
        for t in stages:
            results[f"{t}_acc"].append(accs[t])
            
        # Eval compositions
        comp_names = ["max_of_sum", "sum_of_max", "sub_of_sum", "max_of_min", "sum_of_copy"]
        comps_accs = []
        comps_drifts = []
        for comp in comp_names:
            acc, drift = evaluate_composition_recursive(model, comp)
            results[f"{comp}_acc"].append(acc)
            comps_accs.append(acc)
            comps_drifts.append(drift)
            
        results["avg_comp"].append(np.mean(comps_accs))
        results["drift"].append(np.mean(comps_drifts))
        
    print("\n=================================================================")
    print("MONOLITHIC CL BENCHMARK COMPLETED")
    print(f"Seeds: {seed_count}, Hidden Dim: 32")
    print("=================================================================")
    
    def format_mean_std(field):
        vals = results[field]
        return f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
        
    print(f"{'Metric':<26} | {'Monolithic Weight-Evolution Operator (Ours)':<36}")
    print("-" * 70)
    for t in stages:
        print(f"{t + ' accuracy':<26} | {format_mean_std(f'{t}_acc'):<36}")
    print("-" * 70)
    for comp in comp_names:
        print(f"{comp + ' acc':<26} | {format_mean_std(f'{comp}_acc'):<36}")
    print("-" * 70)
    print(f"{'Average Composition Acc':<26} | {format_mean_std('avg_comp'):<36}")
    print(f"{'manifold_drift':<26} | {format_mean_std('drift'):<36}")
    print("=================================================================\n")
 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_monolithic_cl_benchmark(seed_count=10, device=device)
