import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super(MLP, self).__init__()
        self.fc1 = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=key_0)
        self.fc2 = eqx.nn.Linear(hidden_size, output_size, use_bias=True, key=key_1)

    def __call__(self, x):
        """Forward pass through the network."""
        x = jax.nn.relu(jax.vmap(self.fc1)(x))
        x = jax.vmap(self.fc2)(x)
        return x