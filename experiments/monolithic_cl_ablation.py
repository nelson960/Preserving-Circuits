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
        
        self.fc1 = nn.Linear(3 * d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        
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

def train_monolithic_step(model, current_task, previous_tasks, replay_ratio=0.0, epochs=600, lambda_closure=2.0, lr=0.005):
    """
    Trains the monolithic MLP on the current task, and replays a subset of previous tasks.
    W_E and W_U are frozen during this stage to prevent manifold drift.
    """
    device = next(model.parameters()).device
    # Only optimize MLP parameters; W_E and W_U are frozen!
    optimizer = torch.optim.Adam(list(model.fc1.parameters()) + list(model.fc2.parameters()), lr=lr)
    
    # 1. Get current task dataset
    curr_ins, curr_tgts = get_task_dataset(current_task)
    
    # 2. Get replay dataset
    replay_ins = []
    replay_tgts = []
    for prev_t in previous_tasks:
        ins, tgts = get_task_dataset(prev_t)
        n_samples = len(ins)
        if replay_ratio < 1.0:
            # Subsample randomly according to replay_ratio
            n_select = max(1, int(n_samples * replay_ratio))
            indices = np.random.choice(n_samples, n_select, replace=False)
            replay_ins.append(ins[indices])
            replay_tgts.append(tgts[indices])
        else:
            replay_ins.append(ins)
            replay_tgts.append(tgts)
            
    if len(replay_ins) > 0:
        replay_ins = np.concatenate(replay_ins)
        replay_tgts = np.concatenate(replay_tgts)
        all_ins = np.concatenate([curr_ins, replay_ins])
        all_tgts = np.concatenate([curr_tgts, replay_tgts])
    else:
        all_ins = curr_ins
        all_tgts = curr_tgts
        
    op_a_t = torch.tensor(all_ins[:, 0], dtype=torch.long, device=device)
    op_b_t = torch.tensor(all_ins[:, 1], dtype=torch.long, device=device)
    tasks_t = torch.tensor(all_ins[:, 2], dtype=torch.long, device=device)
    targets_t = torch.tensor(all_tgts, dtype=torch.long, device=device)
    
    # Freeze embeddings
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

def run_experiment(seed_count, replay_ratio, lambda_closure, device):
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
            
        # Freeze the representation manifold
        model.W_E.weight.requires_grad = False
        model.W_U.weight.requires_grad = False
        
        previous_tasks = []
        for task in stages:
            train_monolithic_step(
                model=model,
                current_task=task,
                previous_tasks=previous_tasks,
                replay_ratio=replay_ratio,
                epochs=600,
                lambda_closure=lambda_closure,
                lr=0.005
            )
            previous_tasks.append(task)
            
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
    
    # Return average accuracy of previous tasks right after learning new ones
    # (specifically looking at final task retention of early tasks)
    add_retention = np.mean(all_accs["ADD"])
    max_retention = np.mean(all_accs["MAX"])
    
    return {
        "avg_task_acc": avg_task_acc,
        "avg_comp_acc": avg_comp_acc,
        "avg_drift": avg_drift,
        "add_retention": add_retention,
        "max_retention": max_retention
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_count = 5
    
    print("\n" + "="*80)
    print(f"RUNNING CONTINUAL LEARNING ABLATION STUDY (Seeds={seed_count})")
    print("Goal: Evaluate catastrophic forgetting as replay is restricted or removed.")
    print("="*80)
    
    configs = [
        {"name": "1. Full Replay (Positive Control)", "replay_ratio": 1.0, "lambda_closure": 2.0},
        {"name": "2. No Replay, No Closure (Naïve OCL)", "replay_ratio": 0.0, "lambda_closure": 0.0},
        {"name": "3. No Replay, With Closure", "replay_ratio": 0.0, "lambda_closure": 2.0},
        {"name": "4. 20% Replay, No Closure", "replay_ratio": 0.2, "lambda_closure": 0.0},
        {"name": "5. 20% Replay, With Closure", "replay_ratio": 0.2, "lambda_closure": 2.0},
    ]
    
    print(f"{'Config Description':<38} | {'Avg Task Acc':<12} | {'Avg Comp Acc':<12} | {'ADD Retention':<13} | {'Manifold Drift':<14}")
    print("-" * 96)
    
    for config in configs:
        res = run_experiment(
            seed_count=seed_count,
            replay_ratio=config["replay_ratio"],
            lambda_closure=config["lambda_closure"],
            device=device
        )
        print(f"{config['name']:<38} | {res['avg_task_acc']:.4f}       | {res['avg_comp_acc']:.4f}       | {res['add_retention']:.4f}        | {res['avg_drift']:.6f}")
        
    print("="*96 + "\n")

if __name__ == "__main__":
    main()
