import pytest
from pathlib import Path
from equitorch.transpiler import transpile


@pytest.fixture
def script_dir():
    """Return the directory containing the test script."""
    return Path(__file__).parent


def test_mlp_transpilation(script_dir):
    """Test transpilation of a PyTorch MLP to Equinox."""
    # Read the PyTorch MLP example code
    pytorch_code = Path(script_dir / "../networks/pytorch_mlp_example.py").resolve().read_text()
    
    # Transpile the PyTorch code to Equinox code
    equinox_code = transpile(pytorch_code)
    
    # Verify the transpiled code contains expected elements
    assert "import jax" in equinox_code
    assert "import jax.numpy as jnp" in equinox_code
    assert "import equinox as eqx" in equinox_code
    assert "class MLP(eqx.Module):" in equinox_code
    assert "fc1: eqx.nn.Linear" in equinox_code
    assert "fc2: eqx.nn.Linear" in equinox_code
    assert "def __call__(self, x):" in equinox_code
    assert "jax.nn.relu" in equinox_code
    assert "jax.vmap(self.fc1)" in equinox_code
    assert "jax.vmap(self.fc2)" in equinox_code
    
    # Save the transpiled code to a file
    output_path = script_dir / "../transpiled/equinox_mlp.py"
    with open(output_path, "w") as f:
        f.write(equinox_code)
    
    # Verify the file was created
    assert output_path.exists()


def test_rnn_transpilation(script_dir):
    """Test transpilation of a PyTorch RNN to Equinox."""
    # Read the PyTorch RNN example code
    pytorch_code = Path(script_dir / "../networks/pytorch_rnn_example.py").resolve().read_text()
    
    # Transpile the PyTorch code to Equinox code
    equinox_code = transpile(pytorch_code)
    
    # Basic imports and structure
    assert "import jax" in equinox_code
    assert "import jax.numpy as jnp" in equinox_code
    assert "import equinox as eqx" in equinox_code
    assert "class RNN(eqx.Module):" in equinox_code
    
    # RNN-specific transformations that will likely fail initially:
    assert "rnn_cell: eqx.nn.GRUCell" in equinox_code or "rnn_cell: eqx.nn.RNNCell" in equinox_code
    assert "nn.RNN" not in equinox_code  # Should be converted away
    
    # JAX/Equinox patterns
    assert "def __call__(self, x" in equinox_code  # forward -> __call__
    assert "jax.lax.scan" in equinox_code  # Sequential processing
    assert "jnp.transpose" in equinox_code or "jnp.swapaxes" in equinox_code  # Axis reordering
    
    # Hidden state handling
    assert "jnp.zeros" in equinox_code  # torch.zeros -> jnp.zeros
    assert "hidden is None" in equinox_code  # Conditional logic preserved
    
    # Key management (advanced)
    assert "key=" in equinox_code  # JAX random key handling
    assert "jax.random.split" in equinox_code  # Key splitting
    
    # Tensor operations
    assert "x.shape[0]" in equinox_code  # batch_size extraction
    assert ".device" not in equinox_code  # Device concept should be removed
    
    # Multiple return values preserved
    assert "return output, " in equinox_code  # Tuple return
    
    # Method conversion challenges
    assert "def init_hidden" in equinox_code  # Additional methods preserved
    
    # Save the transpiled code to a file
    output_path = script_dir / "../transpiled/equinox_rnn.py"
    with open(output_path, "w") as f:
        f.write(equinox_code)
    
    # Verify the file was created
    assert output_path.exists()


def test_lstm_transpilation(script_dir):
    """Test transpilation of a PyTorch LSTM with Attention to Equinox."""
    # Read the PyTorch LSTM example code
    pytorch_code = Path(script_dir / "../networks/pytorch_lstm_example.py").resolve().read_text()
    
    # Transpile the PyTorch code to Equinox code
    equinox_code = transpile(pytorch_code)
    
    # Basic imports and structure
    assert "import jax" in equinox_code
    assert "import jax.numpy as jnp" in equinox_code
    assert "import equinox as eqx" in equinox_code
    assert "class AttentionLSTM(eqx.Module):" in equinox_code
    
    # LSTM-specific transformations that will likely fail initially:
    assert "lstm_cell: eqx.nn.LSTMCell" in equinox_code
    assert "nn.LSTM" not in equinox_code  # Should be converted away
    
    # JAX/Equinox patterns
    assert "def __call__(self, x" in equinox_code  # forward -> __call__
    assert "jax.lax.scan" in equinox_code  # Sequential processing
    
    # Attention mechanism transformations
    assert "attention: eqx.nn.Linear" in equinox_code
    assert "context_vector: jnp.ndarray" in equinox_code  # Parameter -> array
    assert "nn.Parameter" not in equinox_code  # Should be converted away
    assert "jax.nn.softmax" in equinox_code  # F.softmax -> jax.nn.softmax
    assert "jnp.tanh" in equinox_code  # torch.tanh -> jnp.tanh
    assert "jnp.matmul" in equinox_code  # torch.matmul -> jnp.matmul
    
    # Hidden state handling (LSTM has both h and c)
    assert "jnp.zeros" in equinox_code  # torch.zeros -> jnp.zeros
    assert "hidden is None" in equinox_code  # Conditional logic preserved
    
    # Key management (advanced)
    assert "key=" in equinox_code  # JAX random key handling
    assert "jax.random.split" in equinox_code  # Key splitting
    
    # Tensor operations
    assert "x.shape[0]" in equinox_code  # batch_size extraction
    assert ".device" not in equinox_code  # Device concept should be removed
    assert "jnp.sum" in equinox_code  # torch.sum -> jnp.sum
    assert "jnp.expand_dims" in equinox_code or ".unsqueeze" not in equinox_code  # unsqueeze conversion
    
    # Multiple return values preserved (output, hidden, attention_weights)
    assert "return output, " in equinox_code  # Tuple return
    assert "attention_weights" in equinox_code  # Attention weights preserved
    
    # Method conversion challenges
    assert "def attention_net" in equinox_code  # Additional methods preserved
    
    # Save the transpiled code to a file
    output_path = script_dir / "../transpiled/equinox_lstm.py"
    with open(output_path, "w") as f:
        f.write(equinox_code)
    
    # Verify the file was created
    assert output_path.exists()


def test_cnn_transpilation(script_dir):
    """Test transpilation of a PyTorch CNN to Equinox."""
    # Read the PyTorch CNN example code
    pytorch_cnn_code = Path(script_dir / "../networks/pytorch_cnn_example.py").resolve().read_text()
    
    # Transpile the PyTorch code to Equinox code
    equinox_cnn_code = transpile(pytorch_cnn_code)
    
    # Verify the transpiled code contains expected elements
    assert "import jax" in equinox_cnn_code
    assert "import jax.numpy as jnp" in equinox_cnn_code
    assert "import equinox as eqx" in equinox_cnn_code
    assert "class SimpleCNN(eqx.Module):" in equinox_cnn_code
    assert "conv1: eqx.nn.Conv2d" in equinox_cnn_code
    assert "conv2: eqx.nn.Conv2d" in equinox_cnn_code
    assert "fc: eqx.nn.Linear" in equinox_cnn_code
    assert "def __call__(self, x):" in equinox_cnn_code
    assert "jax.nn.relu" in equinox_cnn_code
    assert "jax.nn.max_pool" in equinox_cnn_code
    assert "jnp.reshape" in equinox_cnn_code
    
    # Save the transpiled code to a file
    output_path = script_dir / "../transpiled/equinox_cnn.py"
    with open(output_path, "w") as f:
        f.write(equinox_cnn_code)
    
    # Verify the file was created
    assert output_path.exists()
