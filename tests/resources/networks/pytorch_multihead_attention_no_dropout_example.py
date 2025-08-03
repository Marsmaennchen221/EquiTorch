import torch
import torch.nn as nn


class MultiHeadAttentionModuleNoDropout(nn.Module):
    """A simple module with only MultiheadAttention (no dropout) for testing transpilation."""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, query, key, value):
        """Forward pass through multi-head attention."""
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        return attn_output, attn_weights
