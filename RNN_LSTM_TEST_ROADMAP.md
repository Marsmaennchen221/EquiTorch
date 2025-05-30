# RNN and LSTM Test Cases - Development Roadmap

## Summary

I have successfully integrated RNN and LSTM test cases into the EquiTorch transpiler test suite. These tests will initially fail, but help to provide a clear roadmap for further developement.

## Files Created

### 1. PyTorch Example Files
- **`tests/networks/pytorch_rnn_example.py`** - Simple RNN with hidden state management
- **`tests/networks/pytorch_lstm_example.py`** - LSTM with attention mechanism

### 2. Test Cases Added
- **`test_rnn_transpilation()`** - Tests RNN transpilation with 15 specific assertions
- **`test_lstm_transpilation()`** - Tests LSTM transpilation with 20 specific assertions

### 3. Generated Output Files
- **`tests/transpiled/equinox_rnn.py`** - Current transpiler output for RNN
- **`tests/transpiled/equinox_lstm.py`** - Current transpiler output for LSTM

## Current Transpiler State vs. Target

### RNN Transpilation Analysis

**Current Output Issues:**
```python
# ❌ CURRENT: Still has PyTorch imports and modules
import torch
import torch.nn as nn
self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

# ❌ CURRENT: Mixed PyTorch/JAX operations
hidden = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
output, hidden = jax.vmap(self.rnn)(x, hidden)

# ❌ CURRENT: Missing type annotations for RNN cell
class RNN(eqx.Module):
    fc: eqx.nn.Linear  # Missing rnn_cell annotation
```

**Target Output:**
```python
# ✅ TARGET: Pure JAX/Equinox
import jax
import jax.numpy as jnp
import equinox as eqx

class RNN(eqx.Module):
    rnn_cell: eqx.nn.GRUCell  # or eqx.nn.RNNCell
    fc: eqx.nn.Linear
    hidden_size: int
    
    def __call__(self, x, hidden=None):
        if hidden is None:
            hidden = jnp.zeros((batch_size, self.hidden_size))
        
        def scan_fn(h, x_t):
            h_next = self.rnn_cell(x_t, h)
            return h_next, h_next
        
        final_hidden, all_hidden = jax.lax.scan(
            scan_fn, hidden, jnp.transpose(x, (1, 0, 2))
        )
```

### LSTM Transpilation Analysis

**Current Output Issues:**
```python
# ❌ CURRENT: Still has PyTorch modules
self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
self.context_vector = nn.Parameter(torch.randn(hidden_size))

# ❌ CURRENT: Mixed PyTorch/JAX function calls
attention_hidden = torch.tanh(self.attention(lstm_output))
attention_weights = F.softmax(attention_score, dim=1).unsqueeze(2)
```

**Target Output:**
```python
# ✅ TARGET: Pure JAX/Equinox
class AttentionLSTM(eqx.Module):
    lstm_cell: eqx.nn.LSTMCell
    attention: eqx.nn.Linear
    context_vector: jnp.ndarray  # Not nn.Parameter
    fc: eqx.nn.Linear
    
    def __call__(self, x, hidden=None):
        # Use jax.lax.scan for LSTM processing
        # Use jnp.tanh, jax.nn.softmax, jnp.matmul, etc.
```

## Development Roadmap

### Phase 1: Core Module Conversions (HIGH PRIORITY)
1. **`nn.RNN` → `eqx.nn.GRUCell`** conversion in module_converters.py
2. **`nn.LSTM` → `eqx.nn.LSTMCell`** conversion in module_converters.py
3. **`nn.Parameter` → `jnp.ndarray`** conversion for learnable parameters

### Phase 2: Sequential Processing Patterns (HIGH PRIORITY)
4. **Add `jax.lax.scan` generation** for RNN/LSTM loops
5. **Tensor axis reordering** for batch_first semantics (`jnp.transpose`)
6. **Hidden state initialization** patterns

### Phase 3: Function Conversions (MEDIUM PRIORITY)
7. **`torch.zeros` → `jnp.zeros`** in functional_converters.py
8. **`torch.tanh` → `jnp.tanh`** in functional_converters.py
9. **`torch.matmul` → `jnp.matmul`** in functional_converters.py
10. **`torch.sum` → `jnp.sum`** in functional_converters.py
11. **`F.softmax` → `jax.nn.softmax`** in functional_converters.py
12. **`.unsqueeze()` → `jnp.expand_dims()`** in method_converters.py

### Phase 4: Advanced Features (MEDIUM PRIORITY)
13. **Device removal** - strip `.device` references
14. **Key management** - proper JAX PRNG key handling and splitting
15. **Tensor shape operations** - `.size()` → `.shape[]`

### Phase 5: Complex Patterns (LOW PRIORITY)
16. **Multiple return values** handling
17. **Optional parameter patterns** (hidden=None)
18. **Additional method preservation** (init_hidden)

## Test-Driven Development Approach

### Current Test Results
```bash
# RNN Test - 15 assertions, all failing
FAILED tests/equitorch/test_transpiler.py::test_rnn_transpilation

# LSTM Test - 20 assertions, all failing  
FAILED tests/equitorch/test_transpiler.py::test_lstm_transpilation
```

### Development Strategy
1. **Pick one failing assertion** from the roadmap above
2. **Implement the converter** in the appropriate file
3. **Run the test** to see progress
4. **Repeat** until all assertions pass

### Example Development Cycle
```bash
# 1. Implement nn.RNN → eqx.nn.GRUCell conversion
# 2. Run test to see progress
pytest tests/equitorch/test_transpiler.py::test_rnn_transpilation -v

# 3. Check which assertions now pass
# 4. Move to next failing assertion
```

## Key Benefits

1. **Clear Development Path** - Each failing assertion is a specific feature to implement
2. **Incremental Progress** - Can tackle one conversion at a time
3. **Regression Testing** - Ensures existing functionality doesn't break
4. **Real-world Complexity** - Tests handle actual challenging patterns from models.md
5. **Documentation** - Generated files show current vs. target state

## Next Steps

1. **Start with Phase 1** - Core module conversions (nn.RNN, nn.LSTM)
2. **Use failing tests** as your guide for what to implement next
3. **Run tests frequently** to track progress
4. **Update this roadmap** as features are completed

The aggressive test cases will drive systematic improvement of the transpiler's capabilities for handling complex sequential models.
