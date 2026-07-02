import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.project_kv = nn.Linear(1, embed_dim)  # Project 1D features into embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, query, kv):
        """
        query: shape (B, 2048, 768)
        kv: shape (B, 2048)
        """
        B, seq_len, d_q = query.shape

        # Project kv from (B, 2048) -> (B, 2048, 768)
        kv = kv.unsqueeze(-1) # (B, 2048, 1)
        kv_proj = self.project_kv(kv)  # (B, 2048, 768)

        # Normalize inputs
        query = self.norm_q(query)
        kv_proj = self.norm_kv(kv_proj)

        # Perform multihead attention
        # MHA expects (B, seq, dim) if batch_first=True
        out, attn_weights = self.attn(query, kv_proj, kv_proj)  # Q, K, V all shape (B, 2048, 768)

        # Residual + dropout + norm (Post-LN)
        out = self.dropout(out)
        out = self.norm_out(out + query)
        
        return out, attn_weights

class EfficientCrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(2048, embed_dim)
        self.v_proj = nn.Linear(2048, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def reshape_heads(self, x):
        # (B, L, D) -> (B, num_heads, L, head_dim)
        B, L, D = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, num_heads, L, head_dim)

    def forward(self, query, kv):
        """
        query: shape (B, 2048, 768)
        kv: shape (B, 2048)
        """
        B, L, _ = query.shape

        query = self.norm_q(query)
        kv = self.k_proj(kv)  # (B, 2048, 768)
        kv = self.norm_kv(kv)

        # Project Q, K, V
        Q = self.q_proj(query)     # (B, 2048, 768)
        K = kv                     # K already projected and normalized
        V = self.v_proj(kv)        # (B, 2048, 768)

        # Reshape for multi-head
        Q = self.reshape_heads(Q)  # (B, num_heads, 2048, head_dim)
        K = self.reshape_heads(K)
        V = self.reshape_heads(V)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=False)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.embed_dim)

        # Final linear projection
        out = self.out_proj(attn_output)  # (B, 2048, 768)

        return out 