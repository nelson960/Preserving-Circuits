import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os

# Character set definitions
# Lowercase: 'a' (0) to 'j' (9)
# Uppercase: 'A' (10) to 'J' (19)
# Task tokens: [COPY] (20), [SHIFT] (21), [DOUBLE_SHIFT] (22), [CAPS] (23), [LOWER] (24)
vocab_size = 25

char_to_token = {}
token_to_char = {}
for i in range(10):
    c_low = chr(ord('a') + i)
    c_up = chr(ord('A') + i)
    char_to_token[c_low] = i
    char_to_token[c_up] = 10 + i
    token_to_char[i] = c_low
    token_to_char[10 + i] = c_up

task_names = ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]
for idx, name in enumerate(task_names):
    token = 20 + idx
    char_to_token[f"[{name}]"] = token
    token_to_char[token] = f"[{name}]"

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define operators mathematically on token indices
def op_copy(tok):
    if tok >= 20: return tok
    return tok

def op_shift(tok):
    if tok >= 20: return tok
    if tok < 10: # lowercase
        return (tok + 1) % 10
    else: # uppercase
        return 10 + ((tok - 10 + 1) % 10)

def op_double_shift(tok):
    if tok >= 20: return tok
    if tok < 10:
        return (tok + 2) % 10
    else:
        return 10 + ((tok - 10 + 2) % 10)

def op_caps(tok):
    if tok >= 20: return tok
    if tok < 10:
        return tok + 10
    return tok

def op_lower(tok):
    if tok >= 20: return tok
    if tok >= 10 and tok < 20:
        return tok - 10
    return tok

OP_MAP = {
    "COPY": op_copy,
    "SHIFT": op_shift,
    "DOUBLE_SHIFT": op_double_shift,
    "CAPS": op_caps,
    "LOWER": op_lower
}

class OperatorMLP(nn.Module):
    """
    A 2-layer MLP operating directly on code vectors in d_model.
    Acts as a closed latent operator F_h(c) -> c.
    """
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        
        # Tiny positive bias initialization to prevent dead ReLUs
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def get_task_dataset(task_name):
    """
    Generates all 20 character sequences for a given task.
    Sequence: [char, task_token] -> target: [target_char]
    """
    inputs = []
    targets = []
    task_token = char_to_token[f"[{task_name}]"]
    for c_idx in range(20):
        inputs.append([c_idx, task_token])
        targets.append(OP_MAP[task_name](c_idx))
    return np.array(inputs), np.array(targets)


def pretrain_embeddings(W_E, W_U, epochs=600, lr=0.01, margin=2.0, sep_weight=0.5):
    """
    Pretrains the embedding and unembedding matrices on character reconstruction
    with a separation constraint to establish a stable, discrete code space.
    """
    optimizer = torch.optim.Adam([W_E, W_U], lr=lr)
    y = torch.arange(20, device=W_E.device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = W_E[:20] @ W_U[:, :20]
        ce_loss = F.cross_entropy(logits, y)
        
        # Separation loss to prevent collapse
        sep_loss = torch.tensor(0.0, device=W_E.device)
        for i in range(20):
            for j in range(20):
                if i == j:
                    continue
                diff = W_E[i] - W_E[j]
                dist_sq = torch.sum(diff**2)
                if dist_sq < margin:
                    sep_loss += (margin - dist_sq)
                    
        loss = ce_loss + sep_weight * sep_loss
        loss.backward()
        optimizer.step()


def eval_program(program, init_embeddings, library):
    """
    Recursively executes a program tree on initial embeddings.
    program: ('var', 0) or ('op', op_name, args)
    """
    if program[0] == 'var':
        return init_embeddings
    elif program[0] == 'op':
        op_name = program[1]
        arg_embeddings = eval_program(program[2][0], init_embeddings, library)
        op_mlp = library[op_name]
        return op_mlp(arg_embeddings)
    raise ValueError("Invalid program structure")


def generate_programs(library, max_depth=2):
    """
    Generates all valid program trees up to a specified depth.
    """
    programs = [('var', 0)]
    
    def grow(depth):
        if depth >= max_depth:
            return
        current_ops = list(library.keys())
        for op in current_ops:
            # Arity 1 operators
            for sub in list(programs):
                if sub == ('var', 0) or sub[0] == 'op':
                    new_prog = ('op', op, (sub,))
                    if new_prog not in programs:
                        programs.append(new_prog)
        grow(depth + 1)
        
    grow(0)
    return [p for p in programs if p != ('var', 0)]


def search_best_program(task_name, library, W_E, W_U):
    """
    Searches the operator library for a program that solves the task.
    Returns (best_program, best_accuracy, output_embeddings)
    """
    inputs, targets = get_task_dataset(task_name)
    inputs_t = torch.tensor(inputs, dtype=torch.long, device=W_E.device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=W_E.device)
    
    init_emb = W_E[inputs_t[:, 0]]
    programs = generate_programs(library, max_depth=2)
    
    best_prog = None
    best_acc = -1.0
    best_embs = None
    
    for prog in programs:
        with torch.no_grad():
            out_embs = eval_program(prog, init_emb, library)
            logits = out_embs @ W_U
            preds = torch.argmax(logits, dim=-1)
            acc = torch.mean((preds == targets_t).float()).item()
            
            if acc > best_acc:
                best_acc = acc
                best_prog = prog
                best_embs = out_embs
                
    return best_prog, best_acc, best_embs


def train_operator(task_name, W_E, W_U, d_model, d_hidden, lambda_closure=10.0, epochs=800):
    """
    Trains a new OperatorMLP on the task with latent closure constraints.
    """
    device = W_E.device
    inputs, targets = get_task_dataset(task_name)
    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    mlp = OperatorMLP(d_model, d_hidden).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        mlp.train()
        optimizer.zero_grad()
        
        init_emb = W_E[inputs_t[:, 0]]
        out_emb = mlp(init_emb)
        logits = out_emb @ W_U
        
        ce_loss = F.cross_entropy(logits, targets_t)
        
        closure_loss = torch.tensor(0.0, device=device)
        if lambda_closure > 0.0:
            target_emb = W_E[targets_t]
            closure_loss = torch.mean(torch.sum((out_emb - target_emb)**2, dim=-1))
            
        loss = ce_loss + lambda_closure * closure_loss
        loss.backward()
        optimizer.step()
        
    return mlp


def evaluate_composition(program, comp_task_list, W_E, W_U, library):
    """
    Evaluates a program (or chain of task executions) on composition sequences.
    comp_task_list: list of task names to apply in sequence, e.g. ["SHIFT", "CAPS"]
    """
    device = W_E.device
    inputs = np.arange(20)
    
    # Compute ground truth
    targets = []
    for c in inputs:
        val = c
        for task in comp_task_list:
            val = OP_MAP[task](val)
        targets.append(val)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    # Evaluate using the program compilation
    init_emb = W_E[torch.tensor(inputs, dtype=torch.long, device=device)]
    with torch.no_grad():
        out_embs = eval_program(program, init_emb, library)
        logits = out_embs @ W_U
        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((preds == targets_t).float()).item()
        
        # Calculate manifold drift
        target_embs = W_E[targets_t]
        drift = torch.mean(torch.sum((out_embs - target_embs)**2, dim=-1)).item()
        
    return acc, drift


def run_continual_learning_lm(policy, d_model=16, d_hidden=32, lambda_closure=10.0, seed=0, device="cpu"):
    set_seed(seed)
    
    W_E = nn.Parameter(torch.empty(vocab_size, d_model, device=device))
    W_U = nn.Parameter(torch.empty(d_model, vocab_size, device=device))
    nn.init.normal_(W_E, mean=0.0, std=1.0)
    nn.init.normal_(W_U, mean=0.0, std=0.25)
    
    # Pre-train stable code space
    pretrain_embeddings(W_E, W_U)
    
    # Freeze vocabulary embeddings
    W_E.requires_grad = False
    W_U.requires_grad = False
    
    library = {}
    policy_programs = {}
    
    operator_count = 0
    params_added = 0
    
    # Sequential training tasks
    tasks = ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]
    task_accuracies = {}
    
    for stage_idx, task in enumerate(tasks):
        if policy == "always_new_operator":
            # Always train a new operator
            mlp = train_operator(task, W_E, W_U, d_model, d_hidden, lambda_closure)
            library[task] = mlp
            policy_programs[task] = ('op', task, (('var', 0),))
            operator_count += 1
            params_added += sum(p.numel() for p in mlp.parameters())
            
        elif policy == "always_try_reuse":
            # Force reuse of first trained operator if available
            if stage_idx == 0:
                mlp = train_operator(task, W_E, W_U, d_model, d_hidden, lambda_closure)
                library[task] = mlp
                policy_programs[task] = ('op', task, (('var', 0),))
                operator_count += 1
                params_added += sum(p.numel() for p in mlp.parameters())
            else:
                # Reuse the very first operator
                first_op = tasks[0]
                policy_programs[task] = ('op', first_op, (('var', 0),))
                
        elif policy == "admission_gated_reuse":
            # Gated reuse: search library first
            best_prog, best_acc, _ = search_best_program(task, library, W_E, W_U)
            
            if best_acc >= 0.98:
                # Reuse program!
                policy_programs[task] = best_prog
            else:
                # Train a new closed operator
                mlp = train_operator(task, W_E, W_U, d_model, d_hidden, lambda_closure)
                library[task] = mlp
                policy_programs[task] = ('op', task, (('var', 0),))
                operator_count += 1
                params_added += sum(p.numel() for p in mlp.parameters())
                
        # Evaluate current task accuracy
        inputs, targets = get_task_dataset(task)
        inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=device)
        init_emb = W_E[inputs_t[:, 0]]
        
        with torch.no_grad():
            out_embs = eval_program(policy_programs[task], init_emb, library)
            logits = out_embs @ W_U
            preds = torch.argmax(logits, dim=-1)
            acc = torch.mean((preds == targets_t).float()).item()
            task_accuracies[task] = acc

    # Zero-shot evaluation of compositions
    compositions = {
        "shift_then_caps": ["SHIFT", "CAPS"],
        "caps_then_shift": ["CAPS", "SHIFT"],
        "double_shift_then_caps": ["DOUBLE_SHIFT", "CAPS"],
        "shift_then_lower": ["SHIFT", "LOWER"]
    }
    
    comp_accuracies = {}
    comp_drifts = []
    
    for comp_name, task_list in compositions.items():
        # Compile program composition recursively
        def compile_prog(tasks_left):
            if not tasks_left:
                return ('var', 0)
            curr = tasks_left[-1]
            sub = compile_prog(tasks_left[:-1])
            # Retrieve policy's program for this task
            rule = policy_programs[curr]
            # Inline the subprogram into the variables of the rule
            def inline(p, replacement):
                if p[0] == 'var':
                    return replacement
                elif p[0] == 'op':
                    return ('op', p[1], (inline(p[2][0], replacement),))
                return p
            return inline(rule, sub)
            
        compiled_program = compile_prog(task_list)
        
        acc, drift = evaluate_composition(compiled_program, task_list, W_E, W_U, library)
        comp_accuracies[comp_name] = acc
        comp_drifts.append(drift)
        
    avg_comp_acc = np.mean(list(comp_accuracies.values()))
    avg_drift = np.mean(comp_drifts)
    
    # Calculate false reuse rate
    false_reused = 0
    for task in tasks[1:]:
        prog = policy_programs.get(task, ('var', 0))
        # If we reused something that wasn't perfect
        if prog[0] == 'op' and task_accuracies[task] < 0.90:
            false_reused += 1
    false_reuse_rate = false_reused / (len(tasks) - 1)
    
    return {
        "operator_count": operator_count,
        "new_parameters_added": params_added,
        "accuracies": task_accuracies,
        "compositions": comp_accuracies,
        "avg_comp_acc": avg_comp_acc,
        "manifold_drift": avg_drift,
        "false_reuse_rate": false_reuse_rate
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--multi-seed", action="store_true")
    args = parser.parse_args()
    
    seed_count = args.seed_count if args.multi_seed else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n=================================================================")
    print("RUNNING CHARACTER LANGUAGE MODEL CONTINUAL LEARNING BENCHMARK")
    print(f"Seeds: {seed_count}, Hidden Dim: 16, Code Dim: 4")
    print("=================================================================")
    
    policies = ["always_new_operator", "always_try_reuse", "admission_gated_reuse"]
    all_results = {p: [] for p in policies}
    
    for seed in range(seed_count):
        print(f"\n--- Seed {seed} ---")
        for policy in policies:
            res = run_continual_learning_lm(policy, d_model=16, d_hidden=32, lambda_closure=10.0, seed=seed, device=device)
            all_results[policy].append(res)
            
            # Print brief seed summary
            accs = res["accuracies"]
            print(f"  [{policy}] ops={res['operator_count']} params={res['new_parameters_added']} "
                  f"COPY_acc={accs['COPY']:.3f} SHIFT_acc={accs['SHIFT']:.3f} DOUBLE_SHIFT_acc={accs['DOUBLE_SHIFT']:.3f} "
                  f"avg_comp={res['avg_comp_acc']:.3f}")
            
    # Final comparative printing
    print("\n=================================================================")
    print("CHARACTER LANGUAGE MODEL CONTINUAL LEARNING FINAL SUMMARY")
    print("=================================================================")
    
    def format_mean_std(policy_res, field, subkey=None):
        vals = []
        for r in policy_res:
            if subkey:
                vals.append(r[field][subkey])
            else:
                vals.append(r[field])
        return f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
        
    print(f"{'Metric / Policy':<26} | {'always_new_operator':<22} | {'always_try_reuse':<22} | {'admission_gated_reuse':<22}")
    print("-" * 100)
    
    print(f"{'operator_count':<26} | {format_mean_std(all_results['always_new_operator'], 'operator_count'):<22} | {format_mean_std(all_results['always_try_reuse'], 'operator_count'):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'operator_count'):<22}")
    print(f"{'new_parameters_added':<26} | {format_mean_std(all_results['always_new_operator'], 'new_parameters_added'):<22} | {format_mean_std(all_results['always_try_reuse'], 'new_parameters_added'):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'new_parameters_added'):<22}")
    print("-" * 100)
    
    # Task accuracies
    for task in ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]:
        print(f"{task + ' accuracy':<26} | {format_mean_std(all_results['always_new_operator'], 'accuracies', task):<22} | {format_mean_std(all_results['always_try_reuse'], 'accuracies', task):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'accuracies', task):<22}")
        
    print("-" * 100)
    
    # Compositions
    for comp in ["shift_then_caps", "caps_then_shift", "double_shift_then_caps", "shift_then_lower"]:
        print(f"{comp + ' acc':<26} | {format_mean_std(all_results['always_new_operator'], 'compositions', comp):<22} | {format_mean_std(all_results['always_try_reuse'], 'compositions', comp):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'compositions', comp):<22}")
        
    print("-" * 100)
    print(f"{'Average Composition Acc':<26} | {format_mean_std(all_results['always_new_operator'], 'avg_comp_acc'):<22} | {format_mean_std(all_results['always_try_reuse'], 'avg_comp_acc'):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'avg_comp_acc'):<22}")
    print(f"{'manifold_drift':<26} | {format_mean_std(all_results['always_new_operator'], 'manifold_drift'):<22} | {format_mean_std(all_results['always_try_reuse'], 'manifold_drift'):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'manifold_drift'):<22}")
    print(f"{'false_reuse_rate':<26} | {format_mean_std(all_results['always_new_operator'], 'false_reuse_rate'):<22} | {format_mean_std(all_results['always_try_reuse'], 'false_reuse_rate'):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'false_reuse_rate'):<22}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
