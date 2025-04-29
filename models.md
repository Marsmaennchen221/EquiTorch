# Neural Network Implementations: PyTorch vs Equinox

This document presents pairs of equivalent implementations in PyTorch and Equinox, arranged in increasing order of complexity. These examples highlight the paradigm differences that a transpiler would need to address.

## 1. Simple Multi-Layer Perceptron (MLP)

### PyTorch Implementation
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Usage
model = MLP(784, 128, 10)
x = torch.randn(32, 784)  # 32 samples, 784 features
output = model(x)
```

### Equinox Implementation
```python
import jax
import jax.numpy as jnp
import equinox as eqx

class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    
    def __init__(self, input_size, hidden_size, output_size, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        
        self.fc1 = eqx.nn.Linear(input_size, hidden_size, key=key1)
        self.fc2 = eqx.nn.Linear(hidden_size, output_size, key=key2)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        return x

# Usage
key = jax.random.PRNGKey(42)
model = MLP(784, 128, 10, key=key)
x = jax.random.normal(key, (32, 784))  # 32 samples, 784 features
output = model(x)
```

## 2. Convolutional Neural Network (CNN)

### PyTorch Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))  # [batch_size, 32, 28, 28]
        x = self.pool(x)           # [batch_size, 32, 14, 14]
        x = F.relu(self.conv2(x))  # [batch_size, 64, 14, 14]
        x = self.pool(x)           # [batch_size, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)  # [batch_size, 64 * 7 * 7]
        x = F.relu(self.fc1(x))    # [batch_size, 128]
        x = self.dropout(x)        # [batch_size, 128]
        x = self.fc2(x)            # [batch_size, 10]
        return x

# Usage
model = CNN()
x = torch.randn(32, 1, 28, 28)  # 32 MNIST images
output = model(x)
```

### Equinox Implementation
```python
import jax
import jax.numpy as jnp
import equinox as eqx

class CNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    
    def __init__(self, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 4)
        
        self.conv1 = eqx.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, key=keys[1])
        self.fc1 = eqx.nn.Linear(64 * 7 * 7, 128, key=keys[2])
        self.fc2 = eqx.nn.Linear(128, 10, key=keys[3])
    
    def __call__(self, x, key=None):
        # x shape: [batch_size, 1, 28, 28]
        x = jax.nn.relu(self.conv1(x))  # [batch_size, 32, 28, 28]
        x = jax.nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # [batch_size, 32, 14, 14]
        x = jax.nn.relu(self.conv2(x))  # [batch_size, 64, 14, 14]
        x = jax.nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # [batch_size, 64, 7, 7]
        x = x.reshape(x.shape[0], -1)  # [batch_size, 64 * 7 * 7]
        x = jax.nn.relu(self.fc1(x))    # [batch_size, 128]
        
        # Dropout in training mode only
        if key is not None:
            x = jax.random.bernoulli(key, 0.5, x.shape) * x / 0.5  # Scaled dropout
            
        x = self.fc2(x)            # [batch_size, 10]
        return x

# Usage - training mode
key = jax.random.PRNGKey(42)
model_key, dropout_key, data_key = jax.random.split(key, 3)
model = CNN(key=model_key)
x = jax.random.normal(data_key, (32, 1, 28, 28))  # 32 MNIST images
output = model(x, key=dropout_key)  # With dropout

# Usage - inference mode
inference_output = model(x, key=None)  # No dropout
```

## 3. Recurrent Neural Network (RNN)

### PyTorch Implementation
```python
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len, input_size]
        if hidden is None:
            # Initialize hidden state
            hidden = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
            
        # Run RNN
        output, hidden = self.rnn(x, hidden)
        
        # Use output of last time step for classification
        output = self.fc(output[:, -1, :])
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# Usage
model = RNN(10, 20, 2)
x = torch.randn(32, 15, 10)  # [batch_size, seq_len, input_size]
output, hidden = model(x)

# Using the hidden state for continuation
x2 = torch.randn(32, 5, 10)
output2, hidden2 = model(x2, hidden)
```

### Equinox Implementation
```python
import jax
import jax.numpy as jnp
import equinox as eqx

class RNN(eqx.Module):
    rnn_cell: eqx.nn.GRUCell
    fc: eqx.nn.Linear
    hidden_size: int
    
    def __init__(self, input_size, hidden_size, output_size, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        
        self.hidden_size = hidden_size
        self.rnn_cell = eqx.nn.GRUCell(input_size, hidden_size, key=key1)
        self.fc = eqx.nn.Linear(hidden_size, output_size, key=key2)
    
    def __call__(self, x, hidden=None):
        # x shape: [batch_size, seq_len, input_size]
        batch_size = x.shape[0]
        
        if hidden is None:
            # Initialize hidden state
            hidden = jnp.zeros((batch_size, self.hidden_size))
        
        # Process sequence using scan
        def scan_fn(h, x_t):
            h_next = self.rnn_cell(x_t, h)
            return h_next, h_next
        
        final_hidden, all_hidden = jax.lax.scan(
            scan_fn,
            hidden,
            jnp.transpose(x, (1, 0, 2))  # Convert to [seq_len, batch_size, input_size]
        )
        
        # Use output of last time step for classification
        output = self.fc(final_hidden)
        return output, final_hidden

# Usage
key = jax.random.PRNGKey(42)
model_key, data_key1, data_key2 = jax.random.split(key, 3)
model = RNN(10, 20, 2, key=model_key)
x = jax.random.normal(data_key1, (32, 15, 10))  # [batch_size, seq_len, input_size]
output, hidden = model(x)

# Using the hidden state for continuation
x2 = jax.random.normal(data_key2, (32, 5, 10))
output2, hidden2 = model(x2, hidden)
```

## 4. LSTM with Attention Mechanism

### PyTorch Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention layer
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Parameter(torch.randn(hidden_size))
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def attention_net(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        attention_hidden = torch.tanh(self.attention(lstm_output))
        # [batch_size, seq_len, hidden_size] @ [hidden_size] -> [batch_size, seq_len]
        attention_score = torch.matmul(attention_hidden, self.context_vector)
        attention_weights = F.softmax(attention_score, dim=1).unsqueeze(2)
        
        # Apply attention weights to LSTM outputs
        # [batch_size, seq_len, 1] * [batch_size, seq_len, hidden_size]
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights
        
    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len, input_size]
        batch_size = x.size(0)
        
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            hidden = (h0, c0)
        
        # Run LSTM
        lstm_output, hidden = self.lstm(x, hidden)
        
        # Apply attention
        attn_output, attention_weights = self.attention_net(lstm_output)
        
        # Final prediction
        output = self.fc(attn_output)
        
        return output, hidden, attention_weights

# Usage
model = AttentionLSTM(10, 20, 2)
x = torch.randn(32, 15, 10)  # [batch_size, seq_len, input_size]
output, hidden, attention = model(x)
```

### Equinox Implementation
```python
import jax
import jax.numpy as jnp
import equinox as eqx

class AttentionLSTM(eqx.Module):
    lstm_cell: eqx.nn.LSTMCell
    attention: eqx.nn.Linear
    context_vector: jnp.ndarray
    fc: eqx.nn.Linear
    hidden_size: int
    
    def __init__(self, input_size, hidden_size, output_size, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        lstm_key, attn_key, context_key, fc_key = jax.random.split(key, 4)
        
        self.hidden_size = hidden_size
        self.lstm_cell = eqx.nn.LSTMCell(input_size, hidden_size, key=lstm_key)
        self.attention = eqx.nn.Linear(hidden_size, hidden_size, key=attn_key)
        self.context_vector = jax.random.normal(context_key, (hidden_size,))
        self.fc = eqx.nn.Linear(hidden_size, output_size, key=fc_key)
    
    def attention_net(self, lstm_output):
        # lstm_output shape: [seq_len, batch_size, hidden_size]
        
        # Transpose to [batch_size, seq_len, hidden_size]
        lstm_output_t = jnp.transpose(lstm_output, (1, 0, 2))
        
        # Calculate attention scores
        attention_hidden = jnp.tanh(jax.vmap(self.attention)(lstm_output_t))
        # [batch_size, seq_len, hidden_size] @ [hidden_size] -> [batch_size, seq_len]
        attention_score = jnp.matmul(attention_hidden, self.context_vector)
        attention_weights = jax.nn.softmax(attention_score, axis=1)
        
        # Apply attention weights to LSTM outputs
        # [batch_size, seq_len, 1] * [batch_size, seq_len, hidden_size]
        attention_weights_expanded = jnp.expand_dims(attention_weights, axis=2)
        context = jnp.sum(attention_weights_expanded * lstm_output_t, axis=1)
        
        return context, attention_weights
    
    def __call__(self, x, hidden=None):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if hidden is None:
            h = jnp.zeros((batch_size, self.hidden_size))
            c = jnp.zeros((batch_size, self.hidden_size))
            hidden = (h, c)
        
        # Process sequence using scan
        def scan_fn(carry, x_t):
            h, c = carry
            h_next, c_next = self.lstm_cell(x_t, (h, c))
            return (h_next, c_next), h_next
        
        # Transpose to [seq_len, batch_size, input_size]
        x_t = jnp.transpose(x, (1, 0, 2))
        
        # Run LSTM
        (h_final, c_final), lstm_outputs = jax.lax.scan(
            scan_fn,
            hidden,
            x_t
        )
        
        # Apply attention
        attn_output, attention_weights = self.attention_net(lstm_outputs)
        
        # Final prediction
        output = self.fc(attn_output)
        
        return output, (h_final, c_final), attention_weights

# Usage
key = jax.random.PRNGKey(42)
model_key, data_key = jax.random.split(key)
model = AttentionLSTM(10, 20, 2, key=model_key)
x = jax.random.normal(data_key, (32, 15, 10))  # [batch_size, seq_len, input_size]
output, hidden, attention = model(x)
```

## 5. Transformer Model

### PyTorch Implementation
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_layer = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model
        
    def forward(self, src, src_mask=None):
        # src: [batch_size, seq_len]
        # Output: [batch_size, num_classes]
        
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        src = self.positional_encoding(src)  # [batch_size, seq_len, d_model]
        
        # Reshape for transformer: [seq_len, batch_size, d_model]
        src = src.transpose(0, 1)
        
        # Apply transformer layers
        output = self.transformer_encoder(src, src_mask)  # [seq_len, batch_size, d_model]
        
        # Use the representation of the first token for classification
        output = output[0, :, :]  # [batch_size, d_model]
        
        # Final prediction
        output = self.output_layer(output)  # [batch_size, num_classes]
        
        return output

# Usage
vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
max_seq_len = 100
num_classes = 10

model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len, num_classes)
input_ids = torch.randint(0, vocab_size, (32, 50))  # [batch_size, seq_len]
output = model(input_ids)
```

### Equinox Implementation
```python
import jax
import jax.numpy as jnp
import equinox as eqx
import math
from typing import Optional

class PositionalEncoding(eqx.Module):
    pe: jnp.ndarray
    
    def __init__(self, d_model, max_seq_len=100):
        super().__init__()
        position = jnp.arange(max_seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = jnp.zeros((max_seq_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[None, :, :]  # [1, max_seq_len, d_model]
    
    def __call__(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.shape[1], :]

class MultiHeadAttention(eqx.Module):
    d_model: int
    nhead: int
    head_dim: int
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    
    def __init__(self, d_model, nhead, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        q_key, k_key, v_key, out_key = jax.random.split(key, 4)
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = eqx.nn.Linear(d_model, d_model, key=q_key)
        self.k_proj = eqx.nn.Linear(d_model, d_model, key=k_key)
        self.v_proj = eqx.nn.Linear(d_model, d_model, key=v_key)
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=out_key)
    
    def __call__(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        # Linear projections
        q = self.q_proj(q)  # [batch_size, seq_len_q, d_model]
        k = self.k_proj(k)  # [batch_size, seq_len_k, d_model]
        v = self.v_proj(v)  # [batch_size, seq_len_v, d_model]
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, -1, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attention_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention weights
        context = jnp.matmul(attention_weights, v)
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.out_proj(context)
        
        return output

class TransformerEncoderLayer(eqx.Module):
    self_attn: MultiHeadAttention
    ff_linear1: eqx.nn.Linear
    ff_linear2: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    
    def __init__(self, d_model, nhead, dim_feedforward, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        attn_key, lin1_key, lin2_key = jax.random.split(key, 3)
        
        self.self_attn = MultiHeadAttention(d_model, nhead, key=attn_key)
        self.ff_linear1 = eqx.nn.Linear(d_model, dim_feedforward, key=lin1_key)
        self.ff_linear2 = eqx.nn.Linear(dim_feedforward, d_model, key=lin2_key)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)
    
    def __call__(self, src, src_mask=None):
        # Self-attention block
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = src + attn_output  # Residual connection
        src = self.norm1(src)  # Layer normalization
        
        # Feed-forward block
        ff_output = self.ff_linear2(jax.nn.relu(self.ff_linear1(src)))
        src = src + ff_output  # Residual connection
        src = self.norm2(src)  # Layer normalization
        
        return src

class TransformerEncoder(eqx.Module):
    layers: list
    
    def __init__(self, encoder_layer, num_layers):
        self.layers = [encoder_layer] * num_layers
    
    def __call__(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src

class TransformerModel(eqx.Module):
    embedding: eqx.nn.Embedding
    positional_encoding: PositionalEncoding
    transformer_encoder: TransformerEncoder
    output_layer: eqx.nn.Linear
    d_model: int
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len, num_classes, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        emb_key, encoder_key, out_key = jax.random.split(key, 3)
        
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=emb_key)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, key=encoder_key)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        self.output_layer = eqx.nn.Linear(d_model, num_classes, key=out_key)
        self.d_model = d_model
    
    def __call__(self, src, src_mask=None):
        # src: [batch_size, seq_len]
        # Output: [batch_size, num_classes]
        
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        src = self.positional_encoding(src)  # [batch_size, seq_len, d_model]
        
        # Apply transformer layers
        output = self.transformer_encoder(src, src_mask)  # [batch_size, seq_len, d_model]
        
        # Use the representation of the first token for classification
        output = output[:, 0, :]  # [batch_size, d_model]
        
        # Final prediction
        output = self.output_layer(output)  # [batch_size, num_classes]
        
        return output

# Usage
vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
max_seq_len = 100
num_classes = 10

key = jax.random.PRNGKey(42)
model_key, data_key = jax.random.split(key)
model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len, num_classes, key=model_key)
input_ids = jax.random.randint(data_key, (32, 50), 0, vocab_size)  # [batch_size, seq_len]
output = model(input_ids)
```

## 6. VAE (Variational Autoencoder)

### PyTorch Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        self.latent_dim = latent_dim
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(h))
        return reconstruction
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var):
        # Reconstruction loss (binary cross entropy)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return BCE + KLD

# Usage
input_dim = 784  # MNIST image size (28*28)
hidden_dim = 400
latent_dim = 20

model = VAE(input_dim, hidden_dim, latent_dim)
x = torch.rand(128, input_dim)  # 128 flattened MNIST images
recon_x, mu, log_var = model(x)
loss = model.loss_function(recon_x, x, mu, log_var)
```

### Equinox Implementation
```python
import jax
import jax.numpy as jnp
import equinox as eqx

class VAE(eqx.Module):
    fc1: eqx.nn.Linear
    fc_mu: eqx.nn.Linear
    fc_var: eqx.nn.Linear
    fc3: eqx.nn.Linear
    fc4: eqx.nn.Linear
    latent_dim: int
    
    def __init__(self, input_dim, hidden_dim, latent_dim, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 5)
        
        # Encoder
        self.fc1 = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])
        self.fc_mu = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[1])
        self.fc_var = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[2])
        
        # Decoder
        self.fc3 = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[3])
        self.fc4 = eqx.nn.Linear(hidden_dim, input_dim, key=keys[4])
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        h = jax.nn.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var, key):
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(key, shape=mu.shape)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        h = jax.nn.relu(self.fc3(z))
        reconstruction = jax.nn.sigmoid(self.fc4(h))
        return reconstruction
    
    def __call__(self, x, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
            
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, key)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var):
        # Reconstruction loss (binary cross entropy)
        BCE = -jnp.sum(x * jnp.log(recon_x + 1e-10) + (1 - x) * jnp.log(1 - recon_x + 1e-10))
        
        # KL divergence
        KLD = -0.5 * jnp.sum(1 + log_var - jnp.square(mu) - jnp.exp(log_var))
        
        return BCE + KLD

# Usage
input_dim = 784  # MNIST image size (28*28)
hidden_dim = 400
latent_dim = 20

key = jax.random.PRNGKey(42)
model_key, data_key, rng_key = jax.random.split(key, 3)
model = VAE(input_dim, hidden_dim, latent_dim, key=model_key)
x = jax.random.uniform(data_key, (128, input_dim))  # 128 flattened MNIST images
recon_x, mu, log_var = model(x, key=rng_key)
loss = model.loss_function(recon_x, x, mu, log_var)
```

## 7. GAN (Generative Adversarial Network)

### PyTorch Implementation
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(img_shape[0] * img_shape[1] * img_shape[2])),
            nn.Tanh()
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(img_shape[0] * img_shape[1] * img_shape[2]), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        self.latent_dim = latent_dim
        
    def generator_step(self, batch_size, device):
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Generate images
        generated_imgs = self.generator(z)
        
        # Assess generator performance with discriminator
        validity = self.discriminator(generated_imgs)
        
        # Generator wants to fool the discriminator (target all ones)
        g_loss = nn.BCELoss()(validity, torch.ones_like(validity))
        
        return g_loss, generated_imgs
    
    def discriminator_step(self, real_imgs, batch_size, device):
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Generate images
        generated_imgs = self.generator(z)
        
        # Real images should be classified as real (1)
        real_validity = self.discriminator(real_imgs)
        real_loss = nn.BCELoss()(real_validity, torch.ones_like(real_validity))
        
        # Generated images should be classified as fake (0)
        fake_validity = self.discriminator(generated_imgs.detach())
        fake_loss = nn.BCELoss()(fake_validity, torch.zeros_like(fake_validity))
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        return d_loss

# Usage
latent_dim = 100
img_shape = (1, 28, 28)  # Single channel 28x28 images (MNIST)
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GAN(latent_dim, img_shape).to(device)
real_imgs = torch.rand(batch_size, *img_shape, device=device)  # Placeholder for real images

# Discriminator step
d_loss = model.discriminator_step(real_imgs, batch_size, device)

# Generator step
g_loss, generated_imgs = model.generator_step(batch_size, device)
```

### Equinox Implementation
```python
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, List, Callable

class Generator(eqx.Module):
    layers: List[eqx.Module]
    img_shape: Tuple[int, int, int]
    
    def __init__(self, latent_dim, img_shape, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 5)
        
        self.img_shape = img_shape
        
        def make_block(in_feat, out_feat, key):
            linear = eqx.nn.Linear(in_feat, out_feat, key=key)
            norm = eqx.nn.BatchNorm(out_feat, momentum=0.8)
            return [linear, norm]
        
        self.layers = [
            *make_block(latent_dim, 128, keys[0]),
            *make_block(128, 256, keys[1]),
            *make_block(256, 512, keys[2]),
            *make_block(512, 1024, keys[3]),
            eqx.nn.Linear(1024, int(img_shape[0] * img_shape[1] * img_shape[2]), key=keys[4])
        ]
    
    def __call__(self, z, *, is_training=True):
        for i in range(0, len(self.layers) - 1, 2):
            z = self.layers[i](z)  # Linear
            z = self.layers[i+1](z, training=is_training)  # BatchNorm
            z = jax.nn.leaky_relu(z, 0.2)
        
        # Final layer with tanh activation
        z = self.layers[-1](z)
        img = jnp.tanh(z)
        
        # Reshape to image dimensions
        img = img.reshape(img.shape[0], *self.img_shape)
        return img

class Discriminator(eqx.Module):
    layers: List[eqx.Module]
    
    def __init__(self, img_shape, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)
        
        self.layers = [
            eqx.nn.Linear(int(img_shape[0] * img_shape[1] * img_shape[2]), 512, key=keys[0]),
            eqx.nn.Linear(512, 256, key=keys[1]),
            eqx.nn.Linear(256, 1, key=keys[2])
        ]
    
    def __call__(self, img):
        # Flatten image
        img_flat = img.reshape(img.shape[0], -1)
        
        x = self.layers[0](img_flat)
        x = jax.nn.leaky_relu(x, 0.2)
        
        x = self.layers[1](x)
        x = jax.nn.leaky_relu(x, 0.2)
        
        x = self.layers[2](x)
        validity = jax.nn.sigmoid(x)
        
        return validity

class GAN(eqx.Module):
    generator: Generator
    discriminator: Discriminator
    latent_dim: int
    
    def __init__(self, latent_dim, img_shape, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        gen_key, disc_key = jax.random.split(key)
        
        self.generator = Generator(latent_dim, img_shape, key=gen_key)
        self.discriminator = Discriminator(img_shape, key=disc_key)
        self.latent_dim = latent_dim
    
    def generator_loss(self, key):
        @jax.grad
        def loss_fn(generator):
            # Sample noise
            z = jax.random.normal(key, (batch_size, self.latent_dim))
            
            # Generate images
            generated_imgs = generator(z, is_training=True)
            
            # Assess generator performance with discriminator
            validity = self.discriminator(generated_imgs)
            
            # Generator wants to fool the discriminator (target all ones)
            g_loss = -jnp.mean(jnp.log(validity + 1e-10))
            
            return g_loss
        
        return loss_fn(self.generator)
    
    def discriminator_loss(self, real_imgs, key):
        z_key, _ = jax.random.split(key)
        
        @jax.grad
        def loss_fn(discriminator):
            # Sample noise
            z = jax.random.normal(z_key, (real_imgs.shape[0], self.latent_dim))
            
            # Generate images
            generated_imgs = self.generator(z, is_training=False)
            
            # Real images should be classified as real (1)
            real_validity = discriminator(real_imgs)
            real_loss = -jnp.mean(jnp.log(real_validity + 1e-10))
            
            # Generated images should be classified as fake (0)
            fake_validity = discriminator(generated_imgs)
            fake_loss = -jnp.mean(jnp.log(1 - fake_validity + 1e-10))
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            
            return d_loss
        
        return loss_fn(self.discriminator)
    
    def sample(self, batch_size, key):
        # Sample noise
        z = jax.random.normal(key, (batch_size, self.latent_dim))
        
        # Generate images
        return self.generator(z, is_training=False)

# Usage
latent_dim = 100
img_shape = (1, 28, 28)  # Single channel 28x28 images (MNIST)
batch_size = 64

key = jax.random.PRNGKey(42)
model_key, data_key, g_key, d_key = jax.random.split(key, 4)
model = GAN(latent_dim, img_shape, key=model_key)
real_imgs = jax.random.uniform(data_key, (batch_size, *img_shape))  # Placeholder for real images

# Discriminator step
d_grads = model.discriminator_loss(real_imgs, d_key)

# Generator step
g_grads = model.generator_loss(g_key)

# Generate samples
sample_key = jax.random.PRNGKey(10)
generated_imgs = model.sample(16, sample_key)
```

## 8. Neural ODE Model

### PyTorch Implementation
```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, integration_times=torch.tensor([0., 1.])):
        super().__init__()
        self.integration_times = integration_times
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        
        # ODE integration
        # Integrate from t=0 to t=1
        ode_solution = odeint(self.ode_func, h, self.integration_times)
        
        # Take state at final time step
        h_final = ode_solution[-1]
        
        # Decode
        y = self.decoder(h_final)
        
        return y

# Usage
input_dim = 10
hidden_dim = 32
output_dim = 2

model = NeuralODE(input_dim, hidden_dim, output_dim)
x = torch.randn(16, input_dim)
y = model(x)
```

### Equinox Implementation
```python
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

class ODEFunc(eqx.Module):
    net: eqx.nn.Sequential
    
    def __init__(self, hidden_dim, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        
        self.net = eqx.nn.Sequential([
            eqx.nn.Linear(hidden_dim, hidden_dim * 2, key=key1),
            jax.nn.tanh,
            eqx.nn.Linear(hidden_dim * 2, hidden_dim, key=key2)
        ])
    
    def __call__(self, t, y, args):
        return self.net(y)

class NeuralODE(eqx.Module):
    encoder: eqx.nn.Linear
    ode_func: ODEFunc
    decoder: eqx.nn.Linear
    t0: float
    t1: float
    
    def __init__(self, input_dim, hidden_dim, output_dim, t0=0.0, t1=1.0, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        enc_key, ode_key, dec_key = jax.random.split(key, 3)
        
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=enc_key)
        self.ode_func = ODEFunc(hidden_dim, key=ode_key)
        self.decoder = eqx.nn.Linear(hidden_dim, output_dim, key=dec_key)
        self.t0 = t0
        self.t1 = t1
    
    def __call__(self, x):
        # Encode
        h = self.encoder(x)
        
        # ODE integration
        # Set up the ODE solver
        term = diffrax.ODETerm(self.ode_func)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(t1=True)
        
        # Solve the ODE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=self.t0,
            t1=self.t1,
            dt0=0.1,
            y0=h,
            saveat=saveat,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
        )
        
        # Take state at final time step
        h_final = solution.ys
        
        # Decode
        y = self.decoder(h_final)
        
        return y
    
    def trajectory(self, x, times):
        # Encode the input
        h = self.encoder(x)
        
        # ODE integration with multiple save points
        term = diffrax.ODETerm(self.ode_func)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=times)
        
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=times[0],
            t1=times[-1],
            dt0=0.1,
            y0=h,
            saveat=saveat,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
        )
        
        # Return the entire trajectory
        return solution.ys

# Usage
input_dim = 10
hidden_dim = 32
output_dim = 2

key = jax.random.PRNGKey(42)
model_key, data_key = jax.random.split(key)
model = NeuralODE(input_dim, hidden_dim, output_dim, key=model_key)
x = jax.random.normal(data_key, (16, input_dim))
y = model(x)

# Generating a trajectory
times = jnp.linspace(0.0, 1.0, 100)
trajectory = model.trajectory(x, times)
```