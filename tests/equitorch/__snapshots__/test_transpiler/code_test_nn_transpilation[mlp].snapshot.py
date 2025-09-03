import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equitorch.lib as eqt

class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, key_1 = jr.split(key, 2)
        key, subkey = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.fc2 = eqx.nn.Linear(hidden_size, output_size, use_bias=True, key=subkey)

    def __call__(self, x):
        """Forward pass through the network."""
        x = jax.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x