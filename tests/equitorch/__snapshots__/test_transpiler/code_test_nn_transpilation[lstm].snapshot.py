import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equitorch.lib as eqt

class AttentionLSTM(eqx.Module):
    lstm: eqx.nn.LSTMCell
    attention: eqx.nn.Linear
    context_vector: jnp.ndarray
    fc: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, key=None, **kwargs):
        if key is None:
            key = jr.PRNGKey(0)
        key_0, key_1, key_2, key_3 = jr.split(key, 4)
        self.hidden_size = hidden_size
        key, subkey = jax.random.split(key)
        self.lstm = eqx.nn.LSTMCell(input_size, hidden_size, batch_first=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.attention = eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.context_vector = jax.random.normal(subkey, hidden_size)
        key, subkey = jax.random.split(key)
        self.fc = eqx.nn.Linear(hidden_size, output_size, use_bias=True, key=subkey)

    def attention_net(self, lstm_output):
        attention_hidden = jnp.tanh(self.attention(lstm_output))
        attention_score = jnp.matmul(attention_hidden, self.context_vector)
        attention_weights = jnp.expand_dims(jax.nn.softmax(attention_score, dim=1), axis=2)
        context = jnp.sum(attention_weights * lstm_output, dim=1)
        return (context, attention_weights)

    def __call__(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            h0 = jnp.zeros((1, batch_size, self.hidden_size), device=x.device)
            c0 = jnp.zeros((1, batch_size, self.hidden_size), device=x.device)
            hidden = (h0, c0)
        lstm_output, hidden = (jax.lax.scan(lambda carry, x_t: (self.lstm(x_t, carry), self.lstm(x_t, carry)), hidden[0], x.transpose((1, 0, 2)))[1].transpose((1, 0, 2)), jax.lax.scan(lambda carry, x_t: (self.lstm(x_t, carry), self.lstm(x_t, carry)), hidden[0], x.transpose((1, 0, 2)))[0])
        attn_output, attention_weights = jax.vmap(self.attention_net)(lstm_output)
        output = self.fc(attn_output)
        return (output, hidden, attention_weights)