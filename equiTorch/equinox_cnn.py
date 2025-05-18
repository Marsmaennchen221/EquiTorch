import jax
import jax.numpy as jnp
import equinox as eqx

class SimpleCNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc: eqx.nn.Linear
    def __init__(self, in_channels, num_classes, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        key_0 = jax.random.split(key, 1)[0]
        self.conv1 = eqx.nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, use_bias=True, key=key_0)
        key_1 = jax.random.split(key, 1)[0]
        self.conv2 = eqx.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, use_bias=True, key=key_1)
        key_2 = jax.random.split(key, 1)[0]
        self.fc = eqx.nn.Linear(32 * 7 * 7, num_classes, use_bias=True, key=key_2)

    def __call__(self, x):
                x = jax.nn.relu(jax.vmap(self.conv1)(x))
                x = jax.nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='0')
                x = jax.nn.relu(jax.vmap(self.conv2)(x))
                x = jax.nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='0')
                x = jnp.reshape(x, (x.shape[0], -1))
                x = jax.vmap(self.fc)(x)
                return x