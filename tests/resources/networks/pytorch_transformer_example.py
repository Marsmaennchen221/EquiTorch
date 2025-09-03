"""
PyTorch Transformer implementation for transpilation testing.

This module contains a complete transformer architecture including:
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Positional encoding
- Residual connections

Designed to test the transpiler's ability to handle complex nested architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""
    
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_seq_length, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with self-attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # Use (seq_len, batch, embed_dim) format
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
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, x, mask=None):
        """Forward pass through all transformer layers.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SimpleTransformer(nn.Module):
    """Complete transformer model for sequence-to-sequence tasks.
    
    This is a simplified transformer that includes:
    - Input embedding
    - Positional encoding
    - Transformer encoder
    - Output projection
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_seq_length=5000, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Input embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask=None):
        """Forward pass through the complete transformer.
        
        Args:
            src: Source sequence tensor of shape (seq_len, batch_size)
            mask: Optional attention mask
            
        Returns:
            Output logits of shape (seq_len, batch_size, vocab_size)
        """
        # Embedding and scaling
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, mask)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class MiniTransformer(nn.Module):
    """Minimal transformer for testing - single layer, small dimensions."""
    
    def __init__(self, vocab_size=1000, d_model=64, num_heads=4, d_ff=128, dropout=0.1):
        super(MiniTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Simplified components for testing
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff, dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Simple positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(100, d_model))  # Max 100 positions
    
    def forward(self, src):
        """Forward pass through mini transformer.
        
        Args:
            src: Source sequence tensor of shape (seq_len, batch_size)
            
        Returns:
            Output logits of shape (seq_len, batch_size, vocab_size)
        """
        seq_len, batch_size = src.shape
        
        # Embedding
        x = self.embedding(src)  # (seq_len, batch_size, d_model)
        
        # Add positional encoding
        x = x + self.pos_embedding[:seq_len].unsqueeze(1)  # Broadcast over batch
        
        # Single transformer block
        x = self.transformer_block(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


# Example usage and testing functions
def create_sample_data(vocab_size=1000, seq_len=20, batch_size=2):
    """Create sample data for testing."""
    # Random token indices
    src = torch.randint(0, vocab_size, (seq_len, batch_size))
    return src


def test_transformer_components():
    """Test individual transformer components."""
    d_model, num_heads, d_ff = 64, 4, 128
    seq_len, batch_size = 10, 2
    
    # Test data
    x = torch.randn(seq_len, batch_size, d_model)
    
    # Test PositionalEncoding
    pos_enc = PositionalEncoding(d_model)
    x_with_pos = pos_enc(x)
    print(f"PositionalEncoding output shape: {x_with_pos.shape}")
    
    # Test TransformerBlock
    transformer_block = TransformerBlock(d_model, num_heads, d_ff)
    block_output = transformer_block(x_with_pos)
    print(f"TransformerBlock output shape: {block_output.shape}")
    
    # Test TransformerEncoder
    encoder = TransformerEncoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    encoder_output = encoder(x_with_pos)
    print(f"TransformerEncoder output shape: {encoder_output.shape}")


def test_complete_transformer():
    """Test complete transformer models."""
    vocab_size = 1000
    seq_len, batch_size = 15, 3
    
    # Create sample input
    src = create_sample_data(vocab_size, seq_len, batch_size)
    
    # Test SimpleTransformer
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128
    )
    
    output = model(src)
    print(f"SimpleTransformer output shape: {output.shape}")
    
    # Test MiniTransformer
    mini_model = MiniTransformer(vocab_size=vocab_size)
    mini_output = mini_model(src)
    print(f"MiniTransformer output shape: {mini_output.shape}")


if __name__ == "__main__":
    print("Testing Transformer Components...")
    test_transformer_components()
    
    print("\nTesting Complete Transformers...")
    test_complete_transformer()
    
    print("\nAll tests completed successfully!")
