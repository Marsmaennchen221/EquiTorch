from pathlib import Path

import pytest

from equitorch.transpiler import transpile


@pytest.fixture
def resources_dir():
    """Return the directory containing the test script."""
    return Path(__file__).parent / ".." / "resources"


def test_mlp_transpilation(resources_dir):
    """Test transpilation of a PyTorch MLP to Equinox."""
    # Read the PyTorch MLP example code
    pytorch_code = Path(resources_dir / "networks/pytorch_mlp_example.py").resolve().read_text()
    
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


def test_rnn_transpilation(resources_dir):
    """Test transpilation of a PyTorch RNN to Equinox."""
    # Read the PyTorch RNN example code
    pytorch_code = Path(resources_dir / "networks/pytorch_rnn_example.py").resolve().read_text()
    
    # Transpile the PyTorch code to Equinox code
    equinox_code = transpile(pytorch_code)
    
    # Basic imports and structure
    assert "import jax" in equinox_code
    assert "import jax.numpy as jnp" in equinox_code
    assert "import equinox as eqx" in equinox_code
    assert "class RNN(eqx.Module):" in equinox_code
    
    # RNN-specific transformations that will likely fail initially:
    assert "rnn_cell: ElmanRNNCell" in equinox_code or "ElmanRNNCell" in equinox_code
    # Check that nn.RNN is not used in actual code (excluding docstrings)
    code_lines = [line.strip() for line in equinox_code.split('\n') 
                  if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith('#')]
    code_without_docstrings = '\n'.join(code_lines)
    assert "nn.RNN(" not in code_without_docstrings  # Should be converted away from actual usage
    
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
    output_path = resources_dir / "../transpiled/equinox_rnn.py"
    with open(output_path, "w") as f:
        f.write(equinox_code)
    
    # Verify the file was created
    assert output_path.exists()


def test_lstm_transpilation(resources_dir):
    """Test transpilation of a PyTorch LSTM with Attention to Equinox."""
    # Read the PyTorch LSTM example code
    pytorch_code = Path(resources_dir / "networks/pytorch_lstm_example.py").resolve().read_text()
    
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
    output_path = resources_dir / "../transpiled/equinox_lstm.py"
    with open(output_path, "w") as f:
        f.write(equinox_code)
    
    # Verify the file was created
    assert output_path.exists()


def test_cnn_transpilation(resources_dir):
    """Test transpilation of a PyTorch CNN to Equinox."""
    # Read the PyTorch CNN example code
    pytorch_cnn_code = Path(resources_dir / "networks/pytorch_cnn_example.py").resolve().read_text()
    
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


def test_nested_custom_modules_transpilation(resources_dir):
    """Test transpilation of nested custom modules to verify recursive transpilation."""
    # Read the nested custom modules example
    pytorch_code = Path(resources_dir / "networks/pytorch_nested_custom_example.py").resolve().read_text()
    
    # Transpile the PyTorch code to Equinox code
    equinox_code = transpile(pytorch_code)
    
    # Basic imports and structure
    assert "import jax" in equinox_code
    assert "import jax.numpy as jnp" in equinox_code
    assert "import equinox as eqx" in equinox_code
    
    # Verify all classes are converted
    assert "class NestedCustomCNN(eqx.Module):" in equinox_code
    assert "class CustomResidualBlock(eqx.Module):" in equinox_code
    assert "class DeepNestedCNN(eqx.Module):" in equinox_code
    
    # Verify manual converters work (should use eqx.nn.*)
    assert "conv1: eqx.nn.Conv2d" in equinox_code
    assert "bn1: eqx.nn.BatchNorm2d" in equinox_code
    assert "eqx.nn.Linear" in equinox_code
    assert "eqx.nn.Dropout" in equinox_code
    
    # Verify auto-transpiled modules are present (should use class names directly)
    assert "gn2: GroupNorm" in equinox_code  # GroupNorm should be auto-transpiled
    assert "adaptive_pool: AdaptiveAvgPool2d" in equinox_code  # AdaptiveAvgPool2d should be auto-transpiled
    assert "LayerNorm" in equinox_code  # LayerNorm should be auto-transpiled
    
    # Verify auto-transpiled class definitions are injected
    assert "class GroupNorm(" in equinox_code
    assert "class AdaptiveAvgPool2d(" in equinox_code
    assert "class LayerNorm(" in equinox_code
    
    # Verify method transformations
    assert "def __call__(self, x" in equinox_code  # forward -> __call__
    
    # Verify tensor operations are converted
    assert "x.shape[0]" in equinox_code  # x.size(0) -> x.shape[0]
    assert "jnp.reshape" in equinox_code or "x.view" not in equinox_code  # view conversion
    
    # Verify functional calls are converted
    assert "jax.nn.relu" in equinox_code  # torch.relu -> jax.nn.relu
    assert "jax.nn.max_pool" in equinox_code  # torch.max_pool2d -> jax.nn.max_pool
    
    # Verify key management for manual converters
    assert "key=" in equinox_code
    assert "jax.random.PRNGKey" in equinox_code  # Key initialization present
    
    # Verify no PyTorch-specific constructs remain in our main classes
    # (Auto-transpiled classes may still contain PyTorch code, which is expected)
    main_classes = []
    lines = equinox_code.split('\n')
    in_main_class = False
    current_class = []
    
    for line in lines:
        if line.startswith('class NestedCustomCNN(') or line.startswith('class CustomResidualBlock(') or line.startswith('class DeepNestedCNN('):
            in_main_class = True
            current_class = [line]
        elif line.startswith('class ') and in_main_class:
            # End of main class, start of auto-transpiled class
            main_classes.append('\n'.join(current_class))
            in_main_class = False
            current_class = []
        elif in_main_class:
            current_class.append(line)
    
    if in_main_class and current_class:
        main_classes.append('\n'.join(current_class))
    
    main_classes_code = '\n'.join(main_classes)
    
    # These should be converted away from our main classes (but may appear in auto-transpiled classes)
    # Look for actual PyTorch patterns, not the converted eqx.nn patterns
    assert " nn.Conv2d(" not in main_classes_code  # Space before to avoid eqx.nn.Conv2d
    assert " nn.BatchNorm2d(" not in main_classes_code
    assert " nn.GroupNorm(" not in main_classes_code
    assert " nn.Linear(" not in main_classes_code
    assert " nn.Dropout(" not in main_classes_code
    assert " nn.LayerNorm(" not in main_classes_code
    assert " nn.AdaptiveAvgPool2d(" not in main_classes_code
    
    # Also check for assignment patterns
    assert "= nn.Conv2d(" not in main_classes_code
    assert "= nn.BatchNorm2d(" not in main_classes_code
    assert "= nn.GroupNorm(" not in main_classes_code
    assert "= nn.Linear(" not in main_classes_code
    assert "= nn.Dropout(" not in main_classes_code
    assert "= nn.LayerNorm(" not in main_classes_code
    assert "= nn.AdaptiveAvgPool2d(" not in main_classes_code
    
    # Verify recursive transpilation worked - nested classes should be properly handled
    # The CustomResidualBlock should be transpiled and its modules converted
    assert "CustomResidualBlock(" in equinox_code  # Should be called in DeepNestedCNN
    
    # Save the transpiled code for inspection
    output_path = resources_dir / "../transpiled/equinox_nested_custom.py"
    with open(output_path, "w") as f:
        f.write(equinox_code)
    
    # Verify the file was created
    assert output_path.exists()
    
    # Additional verification: check that the transpiled code is syntactically valid
    try:
        compile(equinox_code, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Transpiled code has syntax errors: {e}")
