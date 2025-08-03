import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax
from typing import Optional, Tuple, Union

class EquiTorchMultiHeadAttention(eqx.Module):
    """Equinox implementation of MultiheadAttention with conditional weight output."""
    
    multihead_attn: eqx.nn.MultiheadAttention
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, key: jax.random.PRNGKey = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        
        self.multihead_attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout,
            key=key
        )
    
    def __call__(
        self, 
        query: jnp.ndarray, 
        key_: jnp.ndarray, 
        value: jnp.ndarray, 
        need_weights: bool = True,
        inference: bool = False,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Forward pass through multi-head attention with conditional weight output.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key_: Key tensor of shape (batch_size, seq_len, embed_dim)  
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            need_weights: Whether to return attention weights (static parameter)
            inference: Whether in inference mode (affects dropout, static parameter)
            key: PRNG key for dropout
            
        Returns:
            If need_weights=True: (output, attention_weights)
            If need_weights=False: output
        """
        
        # Create separate functions and JIT compile them with static arguments
        def _attention_with_weights(multihead_attn, query, key_, value, inference, key):
            # Get the attention output
            # Always ensure we have a key - create a dummy key if None
            if key is None:
                key = jax.random.PRNGKey(0)  # Dummy key for deterministic behavior
            output = multihead_attn(query, key_, value, inference=inference, key=key)
            
            # Compute attention weights
            seq_len, embed_dim = query.shape
            key_seq_len = key_.shape[0]
            num_heads = multihead_attn.num_heads
            head_dim = embed_dim // num_heads
            
            # Reshape for multi-head computation
            # query: (seq_len, embed_dim) -> (seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim)
            q = query.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
            # key: (key_seq_len, embed_dim) -> (key_seq_len, num_heads, head_dim) -> (num_heads, key_seq_len, head_dim)
            k = key_.reshape(key_seq_len, num_heads, head_dim).transpose(1, 0, 2)
            
            # Compute attention scores: (num_heads, seq_len, key_seq_len)
            scores = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(head_dim)
            weights = jax.nn.softmax(scores, axis=-1)
            
            # Average over heads for compatibility with PyTorch default behavior
            # Shape: (num_heads, seq_len, key_seq_len) -> (seq_len, key_seq_len)
            weights = jnp.mean(weights, axis=0)
            
            return output, weights
        
        def _attention_without_weights(multihead_attn, query, key_, value, inference, key):
            # Get the attention output only
            # Always ensure we have a key - create a dummy key if None
            if key is None:
                key = jax.random.PRNGKey(0)  # Dummy key for deterministic behavior
            output = multihead_attn(query, key_, value, inference=inference, key=key)
            return output
        
        # JIT compile the functions with static arguments
        jit_with_weights = jax.jit(_attention_with_weights, static_argnames=['inference'])
        jit_without_weights = jax.jit(_attention_without_weights, static_argnames=['inference'])
        
        # Use Python's native if statement to choose the function - this is resolved at call time
        if need_weights:
            return jit_with_weights(self.multihead_attn, query, key_, value, inference, key)
        else:
            return jit_without_weights(self.multihead_attn, query, key_, value, inference, key)
