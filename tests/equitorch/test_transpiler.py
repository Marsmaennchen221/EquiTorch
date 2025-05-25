from pathlib import Path

import pytest

from equitorch.transpiler import PyTorchToEquinoxTranspiler


@pytest.fixture
def script_dir():
    """Return the directory containing the test script."""
    return Path(__file__).parent


def test_mlp_transpilation(script_dir):
    """Test transpilation of a PyTorch MLP to Equinox."""
    # Read the PyTorch MLP example code
    pytorch_code = Path(script_dir / "../networks/pytorch_mlp_example.py").resolve().read_text()
    
    # Create a transpiler
    transpiler = PyTorchToEquinoxTranspiler()
    
    # Transpile the PyTorch code to Equinox code
    equinox_code = transpiler.transpile(pytorch_code)
    
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


def test_cnn_transpilation(script_dir):
    """Test transpilation of a PyTorch CNN to Equinox."""
    # Read the PyTorch CNN example code
    pytorch_cnn_code = Path(script_dir / "../networks/pytorch_cnn_example.py").resolve().read_text()
    
    # Create a transpiler
    transpiler = PyTorchToEquinoxTranspiler()
    
    # Transpile the PyTorch code to Equinox code
    equinox_cnn_code = transpiler.transpile(pytorch_cnn_code)
    
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
