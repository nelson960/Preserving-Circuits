import numpy as np

class ZeroTransformer:
    """
    A pure mathematics (NumPy) implementation of a 1-Layer Attention-Only Transformer.
    No PyTorch, no black boxes. Just matrices and linear algebra.
    """
    def __init__(self, vocab_size, d_model, num_heads):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Initialize weights with small random values (Xavier/Glorot style)
        scale = 1.0 / np.sqrt(d_model)
        
        # 1. Embedding Matrix (Vocab -> Residual Stream)
        self.W_E = np.random.randn(vocab_size, d_model) * scale
        self.W_P = np.random.randn(50, d_model) * scale # Max 50 positions
        
        # 2. Attention Matrices (per head)
        # Using shape (num_heads, d_model, d_head) to keep heads explicit
        self.W_Q = np.random.randn(num_heads, d_model, self.d_head) * scale
        self.W_K = np.random.randn(num_heads, d_model, self.d_head) * scale
        self.W_V = np.random.randn(num_heads, d_model, self.d_head) * scale
        self.W_O = np.random.randn(num_heads, self.d_head, d_model) * scale
        
        # 3. Unembedding Matrix (Residual Stream -> Vocab)
        self.W_U = np.random.randn(d_model, vocab_size) * scale
        
    def softmax(self, x, axis=-1):
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
        
    def forward(self, tokens):
        """
        The Forward Pass: Pure Linear Algebra.
        tokens: list or 1D array of token integers of length T.
        """
        T = len(tokens)
        
        # ==========================================
        # 1. The Residual Stream (X)
        # ==========================================
        # X is shape (T, d_model)
        X = self.W_E[tokens] + self.W_P[:T]
        
        # ==========================================
        # 2. The Routing Circuit (Bilinear Form)
        # ==========================================
        # We will compute attention for all heads.
        # Q, K shape: (num_heads, T, d_head)
        Q = np.zeros((self.num_heads, T, self.d_head))
        K = np.zeros((self.num_heads, T, self.d_head))
        V = np.zeros((self.num_heads, T, self.d_head))
        
        for h in range(self.num_heads):
            Q[h] = X @ self.W_Q[h]
            K[h] = X @ self.W_K[h]
            V[h] = X @ self.W_V[h]
            
        # Attention Scores (S) = Q @ K.T
        # Shape: (num_heads, T, T)
        # In pure math: S_{i,j} = q_i \cdot k_j = x_i^T (W_Q W_K^T) x_j
        S = np.einsum('hti,hki->htk', Q, K) / np.sqrt(self.d_head)
        
        # Apply Causal Mask (can't look into the future)
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        S[:, mask] = -1e9
        
        # Attention Probabilities (A)
        A = self.softmax(S, axis=-1)
        
        # ==========================================
        # 3. The Information Movement Circuit
        # ==========================================
        # Move values across sequence based on A
        # Shape: (num_heads, T, d_head)
        head_outputs = np.einsum('htk,hki->hti', A, V)
        
        # Project back to residual stream
        # Shape: (num_heads, T, d_model)
        projected_outputs = np.zeros((self.num_heads, T, self.d_model))
        for h in range(self.num_heads):
            projected_outputs[h] = head_outputs[h] @ self.W_O[h]
            
        # Sum outputs from all heads and add to residual stream
        attention_out = np.sum(projected_outputs, axis=0)
        X_final = X + attention_out
        
        # ==========================================
        # 4. The Readout (Unembedding)
        # ==========================================
        # Project the final residual stream back to vocabulary logits
        logits = X_final @ self.W_U
        
        return logits, {
            "X_initial": X,
            "A": A,
            "X_final": X_final
        }

    def get_C_QK(self, head):
        """Returns the Bilinear Routing Matrix for a head: W_Q @ W_K^T"""
        return self.W_Q[head] @ self.W_K[head].T
        
    def get_C_OV(self, head):
        """Returns the Subspace Projection Matrix for a head: W_V @ W_O"""
        return self.W_V[head] @ self.W_O[head]

if __name__ == "__main__":
    # Test the Zero Transformer
    print("Initializing Zero Transformer...")
    model = ZeroTransformer(vocab_size=50, d_model=32, num_heads=2)
    
    # A dummy sequence: [K1, V1, K2, V2, Q] -> [5, 26, 12, 30, 5]
    sequence = [5, 26, 12, 30, 5]
    print(f"Input Sequence: {sequence}")
    
    logits, cache = model.forward(sequence)
    
    # The prediction for the final token
    final_token_logits = logits[-1]
    prediction = np.argmax(final_token_logits)
    
    print(f"Prediction (random weights): {prediction}")
    
    # Mathematical properties
    c_qk_0 = model.get_C_QK(head=0)
    c_ov_0 = model.get_C_OV(head=0)
    print(f"\nShape of C_QK (Routing matrix for Head 0): {c_qk_0.shape}")
    print(f"Shape of C_OV (Movement matrix for Head 0): {c_ov_0.shape}")
