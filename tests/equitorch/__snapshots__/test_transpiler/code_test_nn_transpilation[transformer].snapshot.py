import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equitorch.lib as eqt
"\nPyTorch Transformer implementation for transpilation testing.\n\nThis module contains a complete transformer architecture including:\n- Multi-head self-attention\n- Feed-forward networks\n- Layer normalization\n- Positional encoding\n- Residual connections\n\nDesigned to test the transpiler's ability to handle complex nested architectures.\n"
import math

class PositionalEncoding(eqx.Module):
    dropout: eqx.nn.Dropout
    'Sinusoidal positional encoding for transformer inputs.'

    def __init__(self, d_model, max_seq_length=5000, dropout=0.1, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, = jr.split(key, 1)
        key, subkey = jax.random.split(key)
        self.dropout = eqx.nn.Dropout(p=dropout, key=subkey)
        pe = jnp.zeros((max_seq_length, d_model))
        position = jnp.arange(0, max_seq_length, dtype=jnp.float32).unsqueeze(1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = jnp.sin(position * div_term)
        pe[:, 1::2] = jnp.cos(position * div_term)
        pe = jnp.expand_dims(pe, axis=0).transpose(0, 1)
        jax.vmap(self.register_buffer)('pe', pe)

    def __call__(self, x):
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TransformerBlock(eqx.Module):
    self_attention: eqt.nn.EquiTorchMultiHeadAttention
    feed_forward: Sequential
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    'Single transformer encoder block with self-attention and feed-forward.'

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, key_1, key_2, key_3, key_4 = jr.split(key, 5)
        key, subkey = jax.random.split(key)
        self.self_attention = eqt.nn.EquiTorchMultiHeadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, key=subkey)
        key, subkey = jax.random.split(key)
        self.feed_forward = Sequential(eqx.nn.Linear(d_model, d_ff, use_bias=True, key=subkey), jax.nn.relu(key=subkey), eqx.nn.Dropout(dropout, key=subkey), eqx.nn.Linear(d_ff, d_model, use_bias=True, key=subkey), eqx.nn.Dropout(dropout, key=subkey), key=subkey)
        key, subkey = jax.random.split(key)
        self.norm1 = eqx.nn.LayerNorm(d_model, key=subkey)
        key, subkey = jax.random.split(key)
        self.norm2 = eqx.nn.LayerNorm(d_model, key=subkey)
        key, subkey = jax.random.split(key)
        self.dropout = eqx.nn.Dropout(dropout, key=subkey)

    def __call__(self, x, mask=None):
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = jax.vmap(self.feed_forward)(x)
        x = self.norm2(x + ff_output)
        return x

class TransformerEncoder(eqx.Module):
    layers: ModuleList
    'Stack of transformer encoder blocks.'

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, = jr.split(key, 1)
        key, subkey = jax.random.split(key)
        self.layers = ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)], key=subkey)
        self.num_layers = num_layers

    def __call__(self, x, mask=None):
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

class SimpleTransformer(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_encoding: PositionalEncoding
    transformer_encoder: TransformerEncoder
    output_projection: eqx.nn.Linear
    'Complete transformer model for sequence-to-sequence tasks.\n    \n    This is a simplified transformer that includes:\n    - Input embedding\n    - Positional encoding\n    - Transformer encoder\n    - Output projection\n    '

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length=5000, dropout=0.1, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, key_1, key_2, key_3 = jr.split(key, 4)
        self.d_model = d_model
        self.vocab_size = vocab_size
        key, subkey = jax.random.split(key)
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=subkey)
        key, subkey = jax.random.split(key)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout, key=subkey)
        key, subkey = jax.random.split(key)
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, key=subkey)
        key, subkey = jax.random.split(key)
        self.output_projection = eqx.nn.Linear(d_model, vocab_size, use_bias=True, key=subkey)
        jax.vmap(self._init_parameters)()

    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __call__(self, src, mask=None):
        """Forward pass through the complete transformer.
        
        Args:
            src: Source sequence tensor of shape (seq_len, batch_size)
            mask: Optional attention mask
            
        Returns:
            Output logits of shape (seq_len, batch_size, vocab_size)
        """
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, mask)
        output = self.output_projection(x)
        return output

class MiniTransformer(eqx.Module):
    embedding: eqx.nn.Embedding
    transformer_block: TransformerBlock
    output_projection: eqx.nn.Linear
    pos_embedding: jnp.ndarray
    'Minimal transformer for testing - single layer, small dimensions.'

    def __init__(self, vocab_size=1000, d_model=64, num_heads=4, d_ff=128, dropout=0.1, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, key_1, key_2, key_3 = jr.split(key, 4)
        self.d_model = d_model
        key, subkey = jax.random.split(key)
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=subkey)
        key, subkey = jax.random.split(key)
        self.transformer_block = TransformerBlock(d_model, num_heads, d_ff, dropout, key=subkey)
        key, subkey = jax.random.split(key)
        self.output_projection = eqx.nn.Linear(d_model, vocab_size, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.pos_embedding = jax.random.normal(subkey, (100, d_model))

    def __call__(self, src):
        """Forward pass through mini transformer.
        
        Args:
            src: Source sequence tensor of shape (seq_len, batch_size)
            
        Returns:
            Output logits of shape (seq_len, batch_size, vocab_size)
        """
        seq_len, batch_size = src.shape
        x = self.embedding(src)
        x = x + self.pos_embedding[:seq_len].unsqueeze(1)
        x = self.transformer_block(x)
        output = self.output_projection(x)
        return output

def create_sample_data(vocab_size=1000, seq_len=20, batch_size=2):
    """Create sample data for testing."""
    src = torch.randint(0, vocab_size, (seq_len, batch_size))
    return src

def test_transformer_components():
    """Test individual transformer components."""
    d_model, num_heads, d_ff = (64, 4, 128)
    seq_len, batch_size = (10, 2)
    x = torch.randn(seq_len, batch_size, d_model)
    pos_enc = PositionalEncoding(d_model)
    x_with_pos = pos_enc(x)
    print(f'PositionalEncoding output shape: {x_with_pos.shape}')
    transformer_block = TransformerBlock(d_model, num_heads, d_ff)
    block_output = transformer_block(x_with_pos)
    print(f'TransformerBlock output shape: {block_output.shape}')
    encoder = TransformerEncoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    encoder_output = encoder(x_with_pos)
    print(f'TransformerEncoder output shape: {encoder_output.shape}')

def test_complete_transformer():
    """Test complete transformer models."""
    vocab_size = 1000
    seq_len, batch_size = (15, 3)
    src = create_sample_data(vocab_size, seq_len, batch_size)
    model = SimpleTransformer(vocab_size=vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=128)
    output = model(src)
    print(f'SimpleTransformer output shape: {output.shape}')
    mini_model = MiniTransformer(vocab_size=vocab_size)
    mini_output = mini_model(src)
    print(f'MiniTransformer output shape: {mini_output.shape}')
if __name__ == '__main__':
    print('Testing Transformer Components...')
    test_transformer_components()
    print('\nTesting Complete Transformers...')
    test_complete_transformer()
    print('\nAll tests completed successfully!')