import jax
import jax.numpy as jnp
import equinox as eqx

class SimpleCNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc: eqx.nn.Linear
    def __init__(self, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_0 = jax.random.split(key, 1)[0]
        self.conv1 = eqx.nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, use_bias=True, key=key_0)
        key_1 = jax.random.split(key, 1)[0]
        self.conv2 = eqx.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, use_bias=True, key=key_1)
        key_2 = jax.random.split(key, 1)[0]
        self.fc = eqx.nn.Linear(..., num_classes, use_bias=True, key=key_2)

    def __call__(self, x):
                x = jax.nn.relu(jax.vmap(self.fc1)(x))
                x = jax.vmap(self.fc2)(x)
                x = jax.nn.relu(jax.vmap(self.fc1)(x))
                x = jax.vmap(self.fc2)(x)
                x = jax.vmap(self.fc2)(x)
                x = jax.vmap(self.fc2)(x)
                return x