import numpy as np

class LatentZeroTransformer:
    """
    A pure mathematics (NumPy) implementation inspired by DeepSeek MLA 
    (Multi-Head Latent Attention).
    Instead of separate W_K and W_V for each head, it projects the residual 
    stream into a tiny shared latent vector, and then decompresses it.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_latent):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_latent = d_latent # The compressed bottleneck!
        
        scale = 1.0 / np.sqrt(d_model)
        
        self.W_E = np.random.randn(vocab_size, d_model) * scale
        self.W_P = np.random.randn(50, d_model) * scale 
        
        # Q remains standard (though DeepSeek also compresses Q, we simplify here)
        self.W_Q = np.random.randn(num_heads, d_model, self.d_head) * scale
        
        # --- THE LATENT BOTTLENECK ---
        # 1. Compress Residual Stream into a tiny shared latent state C
        self.W_Compress_KV = np.random.randn(d_model, d_latent) * scale
        
        # 2. Decompress C into Keys and Values for each head
        self.W_Decompress_K = np.random.randn(num_heads, d_latent, self.d_head) * scale
        self.W_Decompress_V = np.random.randn(num_heads, d_latent, self.d_head) * scale
        
        # Output projection
        self.W_O = np.random.randn(num_heads, self.d_head, d_model) * scale
        
        self.W_U = np.random.randn(d_model, vocab_size) * scale
        
    def softmax(self, x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
        
    def forward(self, tokens):
        T = len(tokens)
        X = self.W_E[tokens] + self.W_P[:T]
        
        Q = np.zeros((self.num_heads, T, self.d_head))
        for h in range(self.num_heads):
            Q[h] = X @ self.W_Q[h]
            
        # ==========================================
        # LATENT COMPRESSION & DECOMPRESSION
        # ==========================================
        # Step 1: Compress the sequence into a shared latent space
        # Shape: (T, d_latent)
        C = X @ self.W_Compress_KV
        
        # Step 2: Decompress into K and V for all heads
        # Shape: (num_heads, T, d_head)
        K = np.zeros((self.num_heads, T, self.d_head))
        V = np.zeros((self.num_heads, T, self.d_head))
        for h in range(self.num_heads):
            K[h] = C @ self.W_Decompress_K[h]
            V[h] = C @ self.W_Decompress_V[h]
            
        # Standard Attention Mechanism
        S = np.einsum('hti,hki->htk', Q, K) / np.sqrt(self.d_head)
        
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        S[:, mask] = -1e9
        A = self.softmax(S, axis=-1)
        
        head_outputs = np.einsum('htk,hki->hti', A, V)
        
        projected_outputs = np.zeros((self.num_heads, T, self.d_model))
        for h in range(self.num_heads):
            projected_outputs[h] = head_outputs[h] @ self.W_O[h]
            
        X_final = X + np.sum(projected_outputs, axis=0)
        logits = X_final @ self.W_U
        
        return logits
        
    def get_C_QK(self, head):
        """
        The routing matrix is now factorized!
        It's W_Q @ (W_Compress_KV @ W_Decompress_K[head])^T
        """
        effective_W_K = self.W_Compress_KV @ self.W_Decompress_K[head]
        return self.W_Q[head] @ effective_W_K.T

if __name__ == "__main__":
    print("Initializing Latent Zero Transformer (d_model=32, d_latent=8)...")
    # Tiny latent space enforces massive bottlenecks
    model = LatentZeroTransformer(vocab_size=50, d_model=32, num_heads=2, d_latent=8)
    
    sequence = [5, 26, 12, 30, 5]
    logits = model.forward(sequence)
    print("Forward pass successful.")
    
    c_qk = model.get_C_QK(head=0)
    print(f"Shape of effective C_QK: {c_qk.shape}")
    print("Note: Because d_latent is 8, the effective W_K matrix is rank-deficient.")
    print("If Task B overwrites W_Compress_KV, ALL heads lose their circuits simultaneously.")
