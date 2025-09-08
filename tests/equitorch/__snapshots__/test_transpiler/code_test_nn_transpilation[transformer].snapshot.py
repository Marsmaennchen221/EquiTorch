import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equitorch.lib as eqt

class SimpleTransformerBlock(eqx.Module):
    self_attention: eqt.nn.EquiTorchMultiHeadAttention
    feed_forward: eqx.nn.Sequential
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    'Single transformer encoder block with self-attention and feed-forward.'

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, key_1 = jr.split(key, 2)
        key, subkey = jax.random.split(key)
        self.self_attention = eqt.nn.EquiTorchMultiHeadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, key=subkey)
        key, subkey = jax.random.split(key)
        self.feed_forward = eqx.nn.Sequential([eqx.nn.Linear(d_model, d_ff, use_bias=True, key=subkey), jax.nn.relu, eqx.nn.Dropout(dropout), eqx.nn.Linear(d_ff, d_model, use_bias=True, key=subkey), eqx.nn.Dropout(dropout)])
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = jax.vmap(self.feed_forward)(x)
        x = self.norm2(x + ff_output)
        return x

class MinimalTransformer(eqx.Module):
    d_model: int
    embedding: eqx.nn.Embedding
    pos_embedding: jnp.ndarray
    transformer_block: SimpleTransformerBlock
    output_projection: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    'Minimal transformer for transpilation testing.'

    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_seq_length=100, dropout=0.1, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, key_1, key_2, key_3 = jr.split(key, 4)
        self.d_model = d_model
        key, subkey = jax.random.split(key)
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=subkey)
        key, subkey = jax.random.split(key)
        self.pos_embedding = jax.random.normal(subkey, (max_seq_length, d_model))
        key, subkey = jax.random.split(key)
        self.transformer_block = SimpleTransformerBlock(d_model, num_heads, d_ff, dropout, key=subkey)
        key, subkey = jax.random.split(key)
        self.output_projection = eqx.nn.Linear(d_model, vocab_size, use_bias=True, key=subkey)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, src):
        seq_len, batch_size = src.shape
        x = self.embedding(src) * self.d_model ** 0.5
        x = x + jnp.expand_dims(self.pos_embedding[:seq_len], axis=1)
        x = self.dropout(x)
        x = self.transformer_block(x)
        output = self.output_projection(x)
        return output