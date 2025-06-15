import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features, out_features, bias=True, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        key, subkey = jax.random.split(key)
        self.weight = jax.random.normal(out_features, in_features, key=subkey)
        if bias:
            key, subkey = jax.random.split(key)
            self.bias = jax.random.normal(out_features, key=subkey)
        else:
            self.bias = None

    def __call__(self, x):
        """Forward pass using functional linear."""
        return jnp.dot(x, self.weight.T) + self.bias

class SimpleMLP(eqx.Module):
    layer1: CustomLinear
    layer2: CustomLinear
    'A simple MLP that uses custom linear layers to test recursive transpilation.'

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        key, subkey = jax.random.split(key)
        self.layer1 = CustomLinear(input_size, hidden_size, key=subkey)
        key, subkey = jax.random.split(key)
        self.layer2 = CustomLinear(hidden_size, output_size, key=subkey)

    def __call__(self, x):
        """Forward pass through the MLP."""
        x = jax.nn.relu(jax.vmap(self.layer1)(x))
        x = jax.vmap(self.layer2)(x)
        return x