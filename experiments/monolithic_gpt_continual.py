import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Vocabulary Definition
vocab_size = 34
char_to_token = {chr(ord('a') + i): i for i in range(26)}
char_to_token[' '] = 26

task_names = ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]
for idx, name in enumerate(task_names):
    char_to_token[f"[{name}]"] = 27 + idx

char_to_token['[QUERY]'] = 32
char_to_token['[PAD]'] = 33

token_to_char = {v: k for k, v in char_to_token.items()}

# Task math
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

class MonolithicCausalGPT(nn.Module):
    def __init__(self, vocab_size=vocab_size, d_model=32, num_layers=2, num_heads=2, d_hidden=64):
        super().__init__()
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_hidden) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.W_U = nn.Linear(d_model, vocab_size, bias=False)
        self.W_U.weight = self.W_E.weight

    def forward(self, x):
        h = self.W_E(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.W_U(h)
        return logits, h

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_task_dataset(task_name):
    inputs = []
    targets = []
    task_token = char_to_token[f"[{task_name}]"]
    for c in range(26):
        inputs.append([c, task_token])
        targets.append(OP_MAP[task_name](c))
    return np.array(inputs), np.array(targets)

def train_monolithic_step(model, task_name, active_tasks, epochs=150, lambda_closure=10.0, lr=0.005):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # We construct the training dataset combining the active task and replay of older tasks
    all_inputs = []
    all_targets = []
    
    for t in active_tasks:
        ins, tgts = get_task_dataset(t)
        all_inputs.append(ins)
        all_targets.append(tgts)
        
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    inputs_t = torch.tensor(all_inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(all_targets, dtype=torch.long, device=device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits, H = model(inputs_t)
        
        # Logits at the task token position (index 1) predict target character
        logits_at_task = logits[:, 1]
        ce_loss = F.cross_entropy(logits_at_task, targets_t)
        
        closure_loss = torch.tensor(0.0, device=device)
        if lambda_closure > 0.0:
            target_emb = model.W_E(targets_t)
            hidden_at_task = H[:, 1]
            closure_loss = torch.mean(torch.sum((hidden_at_task - target_emb)**2, dim=-1))
            
        loss = ce_loss + lambda_closure * closure_loss
        loss.backward()
        optimizer.step()

def evaluate_tasks(model, trained_tasks):
    device = next(model.parameters()).device
    model.eval()
    
    accuracies = {}
    for task in trained_tasks:
        ins, tgts = get_task_dataset(task)
        ins_t = torch.tensor(ins, dtype=torch.long, device=device)
        tgts_t = torch.tensor(tgts, dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, _ = model(ins_t)
            preds = torch.argmax(logits[:, 1], dim=-1)
            acc = torch.mean((preds == tgts_t).float()).item()
            accuracies[task] = acc
            
    return accuracies

def evaluate_composition_monolithic(model, task_list):
    """
    Feeds a compositional prompt: [char, TASK_TOKEN_1, TASK_TOKEN_2]
    Verifies if output at index 2 correctly predicts T_2(T_1(char))
    """
    device = next(model.parameters()).device
    model.eval()
    
    inputs = []
    targets = []
    
    t1_tok = char_to_token[f"[{task_list[0]}]"]
    t2_tok = char_to_token[f"[{task_list[1]}]"]
    
    for c in range(26):
        inputs.append([c, t1_tok, t2_tok])
        val = OP_MAP[task_list[0]](c)
        val = OP_MAP[task_list[1]](val)
        targets.append(val)
        
    inputs_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits, H = model(inputs_t)
        # Logits at task 2 position (index 2)
        preds = torch.argmax(logits[:, 2], dim=-1)
        acc = torch.mean((preds == targets_t).float()).item()
        
        target_emb = model.W_E(targets_t)
        hidden_at_task2 = H[:, 2]
        drift = torch.mean(torch.sum((hidden_at_task2 - target_emb)**2, dim=-1)).item()
        
    return acc, drift

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Monolithic GPT (no frozen weights, all layers trainable!)
    model = MonolithicCausalGPT(vocab_size=vocab_size, d_model=32, num_layers=2, num_heads=2, d_hidden=64).to(device)
    
    # Pretrain the model to build the basic character representations
    print("Pretraining Monolithic GPT on autoencoder task...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        all_toks = torch.arange(vocab_size, device=device)
        ae_logits = model.W_E(all_toks) @ model.W_U.weight.T
        loss = F.cross_entropy(ae_logits, all_toks)
        loss.backward()
        optimizer.step()
    print("Pretraining complete.")
    
    # Train Tasks Sequentially
    tasks = ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]
    active_tasks = []
    
    for idx, task in enumerate(tasks):
        active_tasks.append(task)
        print(f"\n--- Training Task {idx+1}: {task} ---")
        
        # Train on current tasks (with exact replay of older active tasks)
        train_monolithic_step(model, task, active_tasks, epochs=150, lambda_closure=10.0, lr=0.005)
        
        # Evaluate accuracy on all tasks trained so far
        accs = evaluate_tasks(model, active_tasks)
        for t, acc in accs.items():
            print(f"  Accuracy {t}: {acc:.3f}")
            
    # Evaluate Compositionality
    print("\n=======================================================")
    print("EVALUATING ZERO-SHOT COMPOSITIONS IN MONOLITHIC MODEL")
    print("=======================================================")
    
    compositions = {
        "shift_then_caps": ["SHIFT", "CAPS"],
        "caps_then_shift": ["CAPS", "SHIFT"],
        "double_shift_then_caps": ["DOUBLE_SHIFT", "CAPS"],
        "shift_then_lower": ["SHIFT", "LOWER"]
    }
    
    for name, task_list in compositions.items():
        acc, drift = evaluate_composition_monolithic(model, task_list)
        print(f"{name:<24} | Accuracy: {acc:.4f} | Manifold Drift: {drift:.4f}")

if __name__ == "__main__":
    main()
