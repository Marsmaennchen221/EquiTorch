import jax
import jax.numpy as jnp
import equinox as eqx

class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.fc2 = eqx.nn.Linear(hidden_size, output_size, use_bias=True, key=subkey)

    def __call__(self, x):
        """Forward pass through the network."""
        x = jax.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x