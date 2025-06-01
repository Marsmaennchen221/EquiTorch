import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import torch.nn as nn

class ElmanRNNCell(eqx.Module):
    """Elman RNN Cell implementation matching PyTorch's nn.RNN behavior."""
    input_size: int
    hidden_size: int
    weight_ih: jnp.ndarray
    weight_hh: jnp.ndarray
    bias_ih: jnp.ndarray
    bias_hh: jnp.ndarray

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey):
        self.input_size = input_size
        self.hidden_size = hidden_size
        k1, k2, k3, k4 = jax.random.split(key, 4)
        std = 1.0 / hidden_size ** 0.5
        self.weight_ih = jax.random.uniform(k1, (hidden_size, input_size), minval=-std, maxval=std)
        self.weight_hh = jax.random.uniform(k2, (hidden_size, hidden_size), minval=-std, maxval=std)
        self.bias_ih = jax.random.uniform(k3, (hidden_size,), minval=-std, maxval=std)
        self.bias_hh = jax.random.uniform(k4, (hidden_size,), minval=-std, maxval=std)

    def __call__(self, input: jnp.ndarray, hidden: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh)"""
        return jnp.tanh(jnp.dot(self.weight_ih, input) + jnp.dot(self.weight_hh, hidden) + self.bias_ih + self.bias_hh)

class RNN(eqx.Module):
    rnn: ElmanRNNCell
    fc: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        self.hidden_size = hidden_size
        key, subkey = jax.random.split(key)
        self.rnn = ElmanRNNCell(input_size, hidden_size, key=subkey)
        key, subkey = jax.random.split(key)
        self.fc = eqx.nn.Linear(hidden_size, output_size, use_bias=True, key=subkey)

    def __call__(self, x, hidden=None):
        if hidden is None:
            hidden = jnp.zeros(1, x.shape[0], self.hidden_size, device=x.device)
        output, hidden = jax.lax.scan(lambda carry, x_t: (self.rnn(x_t, carry), self.rnn(x_t, carry)), hidden, x)
        output = jax.vmap(self.fc)(output[:, -1, :])
        return (output, hidden)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
