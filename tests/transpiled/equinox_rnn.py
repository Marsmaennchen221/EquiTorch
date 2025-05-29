import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import torch.nn as nn

class RNN(eqx.Module):
    fc: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = eqx.nn.Linear(hidden_size, output_size, use_bias=True, key=key_0)

    def __call__(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        output, hidden = jax.vmap(self.rnn)(x, hidden)
        output = jax.vmap(self.fc)(output[:, -1, :])
        return (output, hidden)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
