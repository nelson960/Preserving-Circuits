import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os

# Vocabulary Definition
# 'a'-'z' (0-25), ' ' (26)
# Task tokens: [COPY] (27), [SHIFT] (28), [DOUBLE_SHIFT] (29), [CAPS] (30), [LOWER] (31)
# Special tokens: [QUERY] (32), [PAD] (33)
vocab_size = 34

char_to_token = {chr(ord('a') + i): i for i in range(26)}
char_to_token[' '] = 26

task_names = ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]
for idx, name in enumerate(task_names):
    char_to_token[f"[{name}]"] = 27 + idx

char_to_token['[QUERY]'] = 32
char_to_token['[PAD]'] = 33

token_to_char = {v: k for k, v in char_to_token.items()}

# Define operator math on lowercase token indices (0-25)
def op_copy(tok):
    return tok

def op_shift(tok):
    if tok >= 26: return tok
    return (tok + 1) % 26

def op_double_shift(tok):
    if tok >= 26: return tok
    return (tok + 2) % 26

def op_caps(tok):
    if tok >= 26: return tok
    return (tok + 13) % 26

def op_lower(tok):
    if tok >= 26: return tok
    return (tok + 13) % 26

OP_MAP = {
    "COPY": op_copy,
    "SHIFT": op_shift,
    "DOUBLE_SHIFT": op_double_shift,
    "CAPS": op_caps,
    "LOWER": op_lower
}

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        # Causal Attention Score
        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class CausalGPT(nn.Module):
    def __init__(self, vocab_size=vocab_size, d_model=32, num_layers=2, num_heads=2, d_hidden=64):
        super().__init__()
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_hidden) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.W_U = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights for embedding/unembedding space closure
        self.W_U.weight = self.W_E.weight

    def forward(self, x):
        h = self.W_E(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.W_U(h)
        return logits, h

class SkillHead(nn.Module):
    """
    A closed skill head operating on the frozen CausalGPT embedding code space.
    F_h(emb_t) -> emb_{t+1}
    """
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        
        # Default initialization to maintain gradient flow
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_pretraining_data(num_samples=200, seq_len=10):
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "cat", "run", "jump", "sit", "sleep"]
    sequences = []
    for _ in range(num_samples):
        sentence = ""
        while len(sentence) < seq_len:
            sentence += np.random.choice(words) + " "
        sentence = sentence[:seq_len]
        toks = [char_to_token[c] for c in sentence]
        sequences.append(toks)
    return torch.tensor(sequences, dtype=torch.long)


def pretrain_gpt(model, epochs=150, lr=0.005):
    device = next(model.parameters()).device
    data = generate_pretraining_data(num_samples=250, seq_len=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data[:, :-1])
        loss_lm = F.cross_entropy(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1))
        
        # Autoencoder / reconstruction constraint to shape the embedding manifold
        all_toks = torch.arange(vocab_size, device=device)
        ae_logits = model.W_E(all_toks) @ model.W_U.weight.T
        loss_ae = F.cross_entropy(ae_logits, all_toks)
        
        loss = loss_lm + 0.5 * loss_ae
        loss.backward()
        optimizer.step()


def get_task_dataset(task_name):
    inputs = []
    targets = []
    task_token = char_to_token[f"[{task_name}]"]
    for c in range(26):
        inputs.append([c, task_token])
        targets.append(OP_MAP[task_name](c))
    return np.array(inputs), np.array(targets)


def eval_program_gpt(program, init_hidden, library):
    if program[0] == 'var':
        return init_hidden
    elif program[0] == 'op':
        op_name = program[1]
        arg_hidden = eval_program_gpt(program[2][0], init_hidden, library)
        skill_head = library[op_name]
        return skill_head(arg_hidden)
    raise ValueError("Invalid program")


def generate_programs(library, max_depth=2):
    programs = [('var', 0)]
    
    def grow(depth):
        if depth >= max_depth:
            return
        current_ops = list(library.keys())
        for op in current_ops:
            for sub in list(programs):
                if sub == ('var', 0) or sub[0] == 'op':
                    new_prog = ('op', op, (sub,))
                    if new_prog not in programs:
                        programs.append(new_prog)
        grow(depth + 1)
        
    grow(0)
    return [p for p in programs if p != ('var', 0)]


def search_best_program_gpt(task_name, library, base_gpt):
    inputs, targets = get_task_dataset(task_name)
    device = next(base_gpt.parameters()).device
    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    init_hidden = base_gpt.W_E(inputs_t[:, 0])
    programs = generate_programs(library, max_depth=2)
    best_prog = None
    best_acc = -1.0
    
    for prog in programs:
        with torch.no_grad():
            out_hidden = eval_program_gpt(prog, init_hidden, library)
            logits = out_hidden @ base_gpt.W_U.weight.T
            preds = torch.argmax(logits, dim=-1)
            acc = torch.mean((preds == targets_t).float()).item()
            
            if acc > best_acc:
                best_acc = acc
                best_prog = prog
                
    return best_prog, best_acc


def train_skill_head(task_name, base_gpt, d_model, d_hidden, lambda_closure=10.0, epochs=300):
    device = next(base_gpt.parameters()).device
    inputs, targets = get_task_dataset(task_name)
    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    # Input is embedding code, target is target embedding code
    init_hidden = base_gpt.W_E(inputs_t[:, 0])
    target_hidden = base_gpt.W_E(targets_t)
        
    skill_head = SkillHead(d_model, d_hidden).to(device)
    optimizer = torch.optim.Adam(skill_head.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        skill_head.train()
        optimizer.zero_grad()
        
        out_hidden = skill_head(init_hidden)
        logits = out_hidden @ base_gpt.W_U.weight.T
        
        ce_loss = F.cross_entropy(logits, targets_t)
        
        closure_loss = torch.tensor(0.0, device=device)
        if lambda_closure > 0.0:
            closure_loss = torch.mean(torch.sum((out_hidden - target_hidden)**2, dim=-1))
            
        loss = ce_loss + lambda_closure * closure_loss
        loss.backward()
        optimizer.step()
        
    return skill_head


def evaluate_composition_gpt(program, comp_task_list, base_gpt, library):
    device = next(base_gpt.parameters()).device
    inputs = np.arange(26)
    
    targets = []
    for c in inputs:
        val = c
        for task in comp_task_list:
            val = OP_MAP[task](val)
        targets.append(val)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    init_hidden = base_gpt.W_E(torch.tensor(inputs, dtype=torch.long, device=device))
    with torch.no_grad():
        out_hidden = eval_program_gpt(program, init_hidden, library)
        logits = out_hidden @ base_gpt.W_U.weight.T
        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((preds == targets_t).float()).item()
        
        target_emb = base_gpt.W_E(targets_t)
        drift = torch.mean(torch.sum((out_hidden - target_emb)**2, dim=-1)).item()
        
    return acc, drift


def run_continual_learning_gpt(policy, d_model=32, d_hidden=64, lambda_closure=10.0, seed=0, device="cpu"):
    set_seed(seed)
    
    # Initialize autoregressive Causal GPT
    base_gpt = CausalGPT(vocab_size=vocab_size, d_model=d_model, num_layers=2, num_heads=2, d_hidden=d_hidden).to(device)
    
    # Pretrain base GPT
    pretrain_gpt(base_gpt)
    
    # Freeze the entire base GPT model
    for p in base_gpt.parameters():
        p.requires_grad = False
        
    library = {}
    policy_programs = {}
    
    operator_count = 0
    params_added = 0
    
    tasks = ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]
    task_accuracies = {}
    
    for stage_idx, task in enumerate(tasks):
        if policy == "always_new_operator":
            mlp = train_skill_head(task, base_gpt, d_model, d_hidden, lambda_closure)
            library[task] = mlp
            policy_programs[task] = ('op', task, (('var', 0),))
            operator_count += 1
            params_added += sum(p.numel() for p in mlp.parameters())
            
        elif policy == "always_try_reuse":
            if stage_idx == 0:
                mlp = train_skill_head(task, base_gpt, d_model, d_hidden, lambda_closure)
                library[task] = mlp
                policy_programs[task] = ('op', task, (('var', 0),))
                operator_count += 1
                params_added += sum(p.numel() for p in mlp.parameters())
            else:
                first_op = tasks[0]
                policy_programs[task] = ('op', first_op, (('var', 0),))
                
        elif policy == "admission_gated_reuse":
            best_prog, best_acc = search_best_program_gpt(task, library, base_gpt)
            
            if best_acc >= 0.98:
                policy_programs[task] = best_prog
            else:
                mlp = train_skill_head(task, base_gpt, d_model, d_hidden, lambda_closure)
                library[task] = mlp
                policy_programs[task] = ('op', task, (('var', 0),))
                operator_count += 1
                params_added += sum(p.numel() for p in mlp.parameters())
                
        # Eval current task accuracy
        inputs, targets = get_task_dataset(task)
        inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=device)
        
        with torch.no_grad():
            init_hidden = base_gpt.W_E(inputs_t[:, 0])
            out_hidden = eval_program_gpt(policy_programs[task], init_hidden, library)
            logits = out_hidden @ base_gpt.W_U.weight.T
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
        def compile_prog(tasks_left):
            if not tasks_left:
                return ('var', 0)
            curr = tasks_left[-1]
            sub = compile_prog(tasks_left[:-1])
            rule = policy_programs[curr]
            def inline(p, replacement):
                if p[0] == 'var':
                    return replacement
                elif p[0] == 'op':
                    return ('op', p[1], (inline(p[2][0], replacement),))
                return p
            return inline(rule, sub)
            
        compiled_program = compile_prog(task_list)
        
        acc, drift = evaluate_composition_gpt(compiled_program, task_list, base_gpt, library)
        comp_accuracies[comp_name] = acc
        comp_drifts.append(drift)
        
    avg_comp_acc = np.mean(list(comp_accuracies.values()))
    avg_drift = np.mean(comp_drifts)
    
    false_reused = 0
    for task in tasks[1:]:
        prog = policy_programs.get(task, ('var', 0))
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
    print("RUNNING AUTOREGRESSIVE GPT CONTINUAL LEARNING BENCHMARK")
    print(f"Seeds: {seed_count}, Hidden Dim: 32, Layers: 2, Heads: 2")
    print("=================================================================")
    
    policies = ["always_new_operator", "always_try_reuse", "admission_gated_reuse"]
    all_results = {p: [] for p in policies}
    
    for seed in range(seed_count):
        print(f"\n--- Seed {seed} ---")
        for policy in policies:
            res = run_continual_learning_gpt(policy, d_model=32, d_hidden=64, lambda_closure=10.0, seed=seed, device=device)
            all_results[policy].append(res)
            
            accs = res["accuracies"]
            print(f"  [{policy}] ops={res['operator_count']} params={res['new_parameters_added']} "
                  f"COPY_acc={accs['COPY']:.3f} SHIFT_acc={accs['SHIFT']:.3f} DOUBLE_SHIFT_acc={accs['DOUBLE_SHIFT']:.3f} "
                  f"avg_comp={res['avg_comp_acc']:.3f}")
            
    print("\n=================================================================")
    print("AUTOREGRESSIVE GPT CONTINUAL LEARNING FINAL SUMMARY")
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
    
    for task in ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]:
        print(f"{task + ' accuracy':<26} | {format_mean_std(all_results['always_new_operator'], 'accuracies', task):<22} | {format_mean_std(all_results['always_try_reuse'], 'accuracies', task):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'accuracies', task):<22}")
        
    print("-" * 100)
    
    for comp in ["shift_then_caps", "caps_then_shift", "double_shift_then_caps", "shift_then_lower"]:
        print(f"{comp + ' acc':<26} | {format_mean_std(all_results['always_new_operator'], 'compositions', comp):<22} | {format_mean_std(all_results['always_try_reuse'], 'compositions', comp):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'compositions', comp):<22}")
        
    print("-" * 100)
    print(f"{'Average Composition Acc':<26} | {format_mean_std(all_results['always_new_operator'], 'avg_comp_acc'):<22} | {format_mean_std(all_results['always_try_reuse'], 'avg_comp_acc'):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'avg_comp_acc'):<22}")
    print(f"{'manifold_drift':<26} | {format_mean_std(all_results['always_new_operator'], 'manifold_drift'):<22} | {format_mean_std(all_results['always_try_reuse'], 'manifold_drift'):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'manifold_drift'):<22}")
    print(f"{'false_reuse_rate':<26} | {format_mean_std(all_results['always_new_operator'], 'false_reuse_rate'):<22} | {format_mean_std(all_results['always_try_reuse'], 'false_reuse_rate'):<22} | {format_mean_std(all_results['admission_gated_reuse'], 'false_reuse_rate'):<22}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
