import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary Definition
vocab_size = 34
char_to_token = {chr(ord('a') + i): i for i in range(26)}
char_to_token[' '] = 26

task_names = ["COPY", "SHIFT", "DOUBLE_SHIFT", "CAPS", "LOWER"]
for idx, name in enumerate(task_names):
    char_to_token[f"[{name}]"] = 27 + idx

char_to_token['[QUERY]'] = 32
char_to_token['[PAD]'] = 33

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

class MonolithicOperator(nn.Module):
    """
    A single monolithic neural network that processes all tasks.
    It takes the hidden state of a character and the embedding of a task token,
    and maps it to the target character hidden state.
    """
    def __init__(self, d_model=32, d_hidden=64):
        super().__init__()
        # Shared Embedding and Unembedding
        self.W_E = nn.Embedding(vocab_size, d_model)
        self.W_U = nn.Linear(d_model, vocab_size, bias=False)
        self.W_U.weight = self.W_E.weight # Tie weights
        
        # Single shared MLP for ALL operations
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        
    def forward(self, h, task_token_idx):
        # Retrieve task embedding and condition the state
        task_emb = self.W_E(task_token_idx)
        x = h + task_emb
        
        # Pass through the single monolithic MLP
        out = self.fc2(F.relu(self.fc1(x)))
        return out

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

def train_monolithic_operator(model, active_tasks, epochs=400, lambda_closure=10.0, lr=0.01):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # We construct the training dataset combining the active task and replay of older tasks
    all_chars = []
    all_task_toks = []
    all_targets = []
    
    for t in active_tasks:
        ins, tgts = get_task_dataset(t)
        all_chars.append(ins[:, 0])
        all_task_toks.append(ins[:, 1])
        all_targets.append(tgts)
        
    chars_t = torch.tensor(np.concatenate(all_chars), dtype=torch.long, device=device)
    tasks_t = torch.tensor(np.concatenate(all_task_toks), dtype=torch.long, device=device)
    targets_t = torch.tensor(np.concatenate(all_targets), dtype=torch.long, device=device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Get character embedding
        h_in = model.W_E(chars_t)
        
        # Pass through monolithic operator
        h_out = model(h_in, tasks_t)
        
        # Predict target character
        logits = h_out @ model.W_U.weight.T
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
        chars_t = torch.tensor(ins[:, 0], dtype=torch.long, device=device)
        tasks_t = torch.tensor(ins[:, 1], dtype=torch.long, device=device)
        tgts_t = torch.tensor(tgts, dtype=torch.long, device=device)
        
        with torch.no_grad():
            h_in = model.W_E(chars_t)
            h_out = model(h_in, tasks_t)
            logits = h_out @ model.W_U.weight.T
            preds = torch.argmax(logits, dim=-1)
            acc = torch.mean((preds == tgts_t).float()).item()
            accuracies[task] = acc
            
    return accuracies

def evaluate_composition_recursive(model, task_list):
    device = next(model.parameters()).device
    model.eval()
    
    inputs = np.arange(26)
    targets = []
    
    for c in inputs:
        val = c
        for task in task_list:
            val = OP_MAP[task](val)
        targets.append(val)
        
    chars_t = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    
    # Task token indices
    t1_tok = torch.tensor([char_to_token[f"[{task_list[0]}]"]] * 26, dtype=torch.long, device=device)
    t2_tok = torch.tensor([char_to_token[f"[{task_list[1]}]"]] * 26, dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Step 0: embedding
        h0 = model.W_E(chars_t)
        # Step 1: apply task 1
        h1 = model(h0, t1_tok)
        # Step 2: apply task 2 recursively on top of h1
        h2 = model(h1, t2_tok)
        
        # Predict target character
        logits = h2 @ model.W_U.weight.T
        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((preds == targets_t).float()).item()
        
        target_emb = model.W_E(targets_t)
        drift = torch.mean(torch.sum((h2 - target_emb)**2, dim=-1)).item()
        
    return acc, drift

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pretrain the model to build the basic character representations
    model = MonolithicOperator(d_model=32, d_hidden=64).to(device)
    
    print("Pretraining Monolithic embeddings...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(150):
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
        
        # Train the SINGLE monolithic MLP (all weights update!)
        train_monolithic_operator(model, active_tasks, epochs=400, lambda_closure=10.0, lr=0.01)
        
        # Evaluate accuracy on all tasks trained so far
        accs = evaluate_tasks(model, active_tasks)
        for t, acc in accs.items():
            print(f"  Accuracy {t}: {acc:.3f}")
            
    # Evaluate Compositionality
    print("\n=======================================================")
    print("EVALUATING RECURSIVE MONOLITHIC COMPOSITIONS")
    print("=======================================================")
    
    compositions = {
        "shift_then_caps": ["SHIFT", "CAPS"],
        "caps_then_shift": ["CAPS", "SHIFT"],
        "double_shift_then_caps": ["DOUBLE_SHIFT", "CAPS"],
        "shift_then_lower": ["SHIFT", "LOWER"]
    }
    
    for name, task_list in compositions.items():
        acc, drift = evaluate_composition_recursive(model, task_list)
        print(f"{name:<24} | Accuracy: {acc:.4f} | Manifold Drift: {drift:.4f}")

if __name__ == "__main__":
    main()
