import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equitorch.lib as eqt

class SimpleCNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc: eqx.nn.Linear

    def __init__(self, in_channels, num_classes, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, key_1, key_2 = jr.split(key, 3)
        key, subkey = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.conv2 = eqx.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.fc = eqx.nn.Linear(32 * 7 * 7, num_classes, use_bias=True, key=subkey)

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2), (1, 2, 2), 'VALID')
        x = jax.nn.relu(self.conv2(x))
        x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2), (1, 2, 2), 'VALID')
        x = jnp.ravel(x)
        x = self.fc(x)
        return x