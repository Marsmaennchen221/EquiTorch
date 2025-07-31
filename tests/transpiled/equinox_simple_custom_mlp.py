import jax
import jax.numpy as jnp
import equinox as eqx
import equitorch.lib as eqt

class CustomLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features, out_features, bias=True, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        self.weight = jax.random.normal(subkey, (out_features, in_features))
        if bias:
            key, subkey = jax.random.split(key)
            self.bias = jax.random.normal(subkey, out_features)
        else:
            self.bias = None

    def __call__(self, x):
        """Forward pass using functional linear."""
        return jnp.dot(x, self.weight.T) + self.bias

class SimpleMLP(eqx.Module):
    """A simple MLP that uses custom linear layers to test recursive transpilation."""
    layer1: CustomLinear
    layer2: CustomLinear

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        self.layer1 = CustomLinear(input_size, hidden_size, key=subkey)
        key, subkey = jax.random.split(key)
        self.layer2 = CustomLinear(hidden_size, output_size, key=subkey)

    def __call__(self, x):
        """Forward pass through the MLP."""
        x = jax.nn.relu(self.layer1(x))
        x = self.layer2(x)
        return x
