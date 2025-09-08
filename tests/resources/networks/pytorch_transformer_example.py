import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerBlock(nn.Module):
    """Single transformer encoder block with self-attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class MinimalTransformer(nn.Module):
    """Minimal transformer for transpilation testing."""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_seq_length=100, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable positional encoding (simpler than sinusoidal)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_length, d_model))
        
        # Single transformer block
        self.transformer_block = SimpleTransformerBlock(d_model, num_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        seq_len, batch_size = src.shape
        
        # Embedding with scaling
        x = self.embedding(src) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = x + self.pos_embedding[:seq_len].unsqueeze(1)
        x = self.dropout(x)
        
        # Pass through transformer block
        x = self.transformer_block(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
