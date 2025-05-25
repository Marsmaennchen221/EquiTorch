import pytest
from pathlib import Path
from equitorch.ast_transpiler import ASTBasedTranspiler
from equitorch.transpiler import PyTorchToEquinoxTranspiler


@pytest.fixture
def script_dir():
    """Return the directory containing the test script."""
    return Path(__file__).parent


def test_ast_transpiler_cnn(script_dir):
    """Test AST-based transpilation of a PyTorch CNN to Equinox."""
    # Read the PyTorch CNN example code
    pytorch_cnn_code = Path(script_dir / "../networks/pytorch_cnn_example.py").resolve().read_text()
    
    # Create AST-based transpiler
    ast_transpiler = ASTBasedTranspiler()
    
    # Transpile the PyTorch code to Equinox code
    equinox_cnn_code = ast_transpiler.transpile(pytorch_cnn_code)
    
    # Verify the transpiled code contains expected elements
    assert "import jax" in equinox_cnn_code
    assert "import jax.numpy as jnp" in equinox_cnn_code
    assert "import equinox as eqx" in equinox_cnn_code
    assert "class SimpleCNN(eqx.Module):" in equinox_cnn_code
    assert "conv1: eqx.nn.Conv2d" in equinox_cnn_code
    assert "conv2: eqx.nn.Conv2d" in equinox_cnn_code
    assert "fc: eqx.nn.Linear" in equinox_cnn_code
    assert "def __call__(self, x):" in equinox_cnn_code
    
    # Save the transpiled code to a file for inspection
    output_path = script_dir / "../transpiled/ast_equinox_cnn.py"
    with open(output_path, "w") as f:
        f.write(equinox_cnn_code)
    
    # Verify the file was created
    assert output_path.exists()
    
    print("AST-based transpiled code:")
    print(equinox_cnn_code)


def test_compare_transpilers(script_dir):
    """Compare output of AST-based transpiler with original string-based transpiler."""
    # Read the PyTorch CNN example code
    pytorch_cnn_code = Path(script_dir / "../networks/pytorch_cnn_example.py").resolve().read_text()
    
    # Create both transpilers
    ast_transpiler = ASTBasedTranspiler()
    string_transpiler = PyTorchToEquinoxTranspiler()
    
    # Transpile with both
    ast_result = ast_transpiler.transpile(pytorch_cnn_code)
    string_result = string_transpiler.transpile(pytorch_cnn_code)
    
    # Save both results for comparison
    ast_output_path = script_dir / "../transpiled/ast_comparison_cnn.py"
    string_output_path = script_dir / "../transpiled/string_comparison_cnn.py"
    
    with open(ast_output_path, "w") as f:
        f.write(ast_result)
    
    with open(string_output_path, "w") as f:
        f.write(string_result)
    
    # Both should contain the same key elements
    key_elements = [
        "import jax",
        "import jax.numpy as jnp", 
        "import equinox as eqx",
        "class SimpleCNN(eqx.Module):",
        "conv1: eqx.nn.Conv2d",
        "conv2: eqx.nn.Conv2d",
        "fc: eqx.nn.Linear",
        "def __call__(self, x):"
    ]
    
    for element in key_elements:
        assert element in ast_result, f"AST result missing: {element}"
        assert element in string_result, f"String result missing: {element}"
    
    print("\n=== AST-based result ===")
    print(ast_result)
    print("\n=== String-based result ===")
    print(string_result)


def test_ast_transpiler_mlp(script_dir):
    """Test AST-based transpilation of a PyTorch MLP to Equinox."""
    # Read the PyTorch MLP example code
    pytorch_mlp_code = Path(script_dir / "../networks/pytorch_mlp_example.py").resolve().read_text()
    
    # Create AST-based transpiler
    ast_transpiler = ASTBasedTranspiler()
    
    # Transpile the PyTorch code to Equinox code
    equinox_mlp_code = ast_transpiler.transpile(pytorch_mlp_code)
    
    # Verify the transpiled code contains expected elements
    assert "import jax" in equinox_mlp_code
    assert "import jax.numpy as jnp" in equinox_mlp_code
    assert "import equinox as eqx" in equinox_mlp_code
    assert "class MLP(eqx.Module):" in equinox_mlp_code
    assert "fc1: eqx.nn.Linear" in equinox_mlp_code
    assert "fc2: eqx.nn.Linear" in equinox_mlp_code
    assert "def __call__(self, x):" in equinox_mlp_code
    
    # Save the transpiled code to a file
    output_path = script_dir / "../transpiled/ast_equinox_mlp.py"
    with open(output_path, "w") as f:
        f.write(equinox_mlp_code)
    
    # Verify the file was created
    assert output_path.exists()
    
    print("AST-based MLP transpiled code:")
    print(equinox_mlp_code)
