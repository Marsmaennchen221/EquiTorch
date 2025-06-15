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
    
    # Save the transpiled code to a file FIRST (before any assertions)
    output_path = resources_dir / "../transpiled/equinox_mlp.py"
    with open(output_path, "w") as f:
        f.write(equinox_code)
    
    # Verify the file was created
    assert output_path.exists()
    
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


def test_simple_custom_mlp_transpilation(resources_dir):
    """Test transpilation of a simple custom MLP with custom linear layers."""
    # Read the PyTorch simple custom MLP example code
    pytorch_code = Path(resources_dir / "networks/pytorch_simple_custom_mlp_example.py").resolve().read_text()
    
    # Transpile the PyTorch code to Equinox code
    equinox_code = transpile(pytorch_code)
    
    # Save the transpiled code for inspection FIRST (before any assertions)
    output_path = resources_dir / "../transpiled/equinox_simple_custom_mlp.py"
    with open(output_path, "w") as f:
        f.write(equinox_code)
    
    # Verify the file was created
    assert output_path.exists()
    
    # Basic imports and structure
    assert "import jax" in equinox_code
    assert "import jax.numpy as jnp" in equinox_code
    assert "import equinox as eqx" in equinox_code
    
    # Verify both custom classes are converted
    assert "class CustomLinear(eqx.Module):" in equinox_code
    assert "class SimpleMLP(eqx.Module):" in equinox_code
    
    # Verify type annotations for parameters and custom modules
    assert "weight: jnp.ndarray" in equinox_code  # nn.Parameter -> jnp.ndarray
    assert "bias: jnp.ndarray" in equinox_code
    assert "layer1: CustomLinear" in equinox_code  # Custom class type annotation
    assert "layer2: CustomLinear" in equinox_code
    
    # Verify method transformations
    assert "def __call__(self, x" in equinox_code  # forward -> __call__
    
    # Verify functional calls are converted
    assert "jax.nn.relu" in equinox_code  # F.relu -> jax.nn.relu
    
    # Verify JAX patterns for module calls
    assert "jax.vmap(self.layer1)" in equinox_code
    assert "jax.vmap(self.layer2)" in equinox_code
    
    # Verify key management for random initialization
    assert "key=" in equinox_code  # JAX random key handling
    assert "jax.random.PRNGKey" in equinox_code  # Key initialization
    assert "jax.random.split" in equinox_code  # Key splitting
    
    # Verify no PyTorch-specific constructs remain in main classes
    # Extract only the main custom classes (not any auto-transpiled ones)
    main_classes = []
    lines = equinox_code.split('\n')
    in_main_class = False
    current_class = []
    
    for line in lines:
        if line.startswith('class CustomLinear(') or line.startswith('class SimpleMLP('):
            in_main_class = True
            current_class = [line]
        elif line.startswith('class ') and in_main_class:
            # End of main class, start of another class
            main_classes.append('\n'.join(current_class))
            in_main_class = False
            current_class = []
        elif in_main_class:
            current_class.append(line)
    
    if in_main_class and current_class:
        main_classes.append('\n'.join(current_class))
    
    main_classes_code = '\n'.join(main_classes)
    
    # Verify PyTorch constructs are converted away from main classes
    assert "nn.Parameter(" not in main_classes_code  # Should be converted
    assert "torch.randn" not in main_classes_code  # Should use JAX random
    assert "F.linear" not in main_classes_code  # Should be converted to JAX equivalent
    assert "F.relu" not in main_classes_code  # Should be jax.nn.relu
    
    # Verify custom module usage
    assert "CustomLinear(" in equinox_code  # Custom class should be instantiated
    
    # Additional verification: check that the transpiled code is syntactically valid
    try:
        compile(equinox_code, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Transpiled code has syntax errors: {e}")


def test_cnn_transpilation(resources_dir):
    """Test transpilation of a PyTorch CNN to Equinox."""
    # Read the PyTorch CNN example code
    pytorch_cnn_code = Path(resources_dir / "networks/pytorch_cnn_example.py").resolve().read_text()
    
    # Transpile the PyTorch code to Equinox code
    equinox_cnn_code = transpile(pytorch_cnn_code)
    
    # Save the transpiled code to a file FIRST (before any assertions)
    output_path = resources_dir / "../transpiled/equinox_cnn.py"
    with open(output_path, "w") as f:
        f.write(equinox_cnn_code)
    
    # Verify the file was created
    assert output_path.exists()
    
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
