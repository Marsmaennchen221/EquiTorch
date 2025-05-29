import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(eqx.Module):
    attention: eqx.nn.Linear
    fc: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=key_0)
        self.context_vector = nn.Parameter(torch.randn(hidden_size))
        self.fc = eqx.nn.Linear(hidden_size, output_size, use_bias=True, key=key_1)

    def attention_net(self, lstm_output):
        attention_hidden = torch.tanh(self.attention(lstm_output))
        attention_score = torch.matmul(attention_hidden, self.context_vector)
        attention_weights = F.softmax(attention_score, dim=1).unsqueeze(2)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return (context, attention_weights)

    def __call__(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            hidden = (h0, c0)
        lstm_output, hidden = jax.vmap(self.lstm)(x, hidden)
        attn_output, attention_weights = jax.vmap(self.attention_net)(lstm_output)
        output = jax.vmap(self.fc)(attn_output)
        return (output, hidden, attention_weights)
