import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equitorch.lib as eqt

class MultiHeadAttentionModuleNoDropout(eqx.Module):
    multihead_attn: eqt.nn.EquiTorchMultiHeadAttention
    'A simple module with only MultiheadAttention (no dropout) for testing transpilation.'

    def __init__(self, embed_dim, num_heads, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, = jr.split(key, 1)
        key, subkey = jax.random.split(key)
        self.multihead_attn = eqt.nn.EquiTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, key=subkey)

    def __call__(self, query, key, value):
        """Forward pass through multi-head attention."""
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        return (attn_output, attn_weights)