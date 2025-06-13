import jax
import jax.numpy as jnp
import equinox as eqx
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc: eqx.nn.Linear

    def __init__(self, in_channels, num_classes, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super(SimpleCNN, self).__init__()
        self.conv1 = eqx.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, use_bias=True, key=key_0)
        self.conv2 = eqx.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, use_bias=True, key=key_1)
        self.fc = eqx.nn.Linear(32 * 7 * 7, num_classes, use_bias=True, key=key_2)

    def __call__(self, x):
        x = jax.nn.relu(jax.vmap(self.conv1)(x))
        x = jax.nn.max_pool(x, 2)
        x = jax.nn.relu(jax.vmap(self.conv2)(x))
        x = jax.nn.max_pool(x, 2)
        x = jnp.reshape(x, (x.size(0), -1))
        x = jax.vmap(self.fc)(x)
        return x