import torch
import torch.nn as nn


class MultiHeadAttentionModule(nn.Module):
    """A simple module with only MultiheadAttention for testing transpilation."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, query, key, value, need_weights=True):
        """Forward pass through multi-head attention."""
        return self.multihead_attn(query, key, value, need_weights=need_weights)
    