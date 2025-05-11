import jax
import jax.numpy as jnp
import equinox as eqx

class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    def __init__(self, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_0 = jax.random.split(key, 1)[0]
        self.fc1 = eqx.nn.Linear(input_size, hidden_size, use_bias=True, key=key_0)
        key_1 = jax.random.split(key, 1)[0]
        self.fc2 = eqx.nn.Linear(hidden_size, output_size, use_bias=True, key=key_1)

    def __call__(self, x):
                x = jax.nn.relu(jax.vmap(self.fc1)(x))
                x = jax.vmap(self.fc2)(x)
                return x