import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
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
    # Note: In correct Equinox pattern, models process single examples
    # No vmap should be inside the model - batching is handled at the top level
    assert "jax.vmap(self.fc1)" not in equinox_code
    assert "jax.vmap(self.fc2)" not in equinox_code
    assert "self.fc1(x)" in equinox_code
    assert "self.fc2(x)" in equinox_code


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
    assert "jax.lax.reduce_window" in equinox_cnn_code
    assert "jnp.ravel" in equinox_cnn_code


# Helper functions for functional testing
def _import_transpiled_module(module_path):
    """Dynamically import a transpiled module."""
    spec = importlib.util.spec_from_file_location("transpiled_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["transpiled_module"] = module
    spec.loader.exec_module(module)
    return module


def _generate_random_input(shape, key):
    """Generate random input data with given shape."""
    return jax.random.normal(key, shape)


def _test_forward_pass(model, input_data):
    """Test forward pass and verify output properties."""
    original_batch_size = input_data.shape[0]
    
    # For Equinox models, use vmap to handle batches since models process single examples
    if input_data.ndim > 1 and input_data.shape[0] > 1:
        # Batched input - use vmap
        output = jax.vmap(model)(input_data)
        expected_batch_size = original_batch_size
    else:
        # Single example or already squeezed
        if input_data.ndim > 1 and input_data.shape[0] == 1:
            # Remove batch dimension for single example
            input_data = input_data[0]
        output = model(input_data)
        expected_batch_size = None  # No batch dimension expected
    
    # Verify output is a JAX array
    assert isinstance(output, jnp.ndarray), f"Output should be jnp.ndarray, got {type(output)}"
    
    # Verify output is finite
    assert jnp.all(jnp.isfinite(output)), "Output contains non-finite values"
    
    # Verify output has correct batch dimension
    if expected_batch_size is not None:
        assert output.shape[0] == expected_batch_size, f"Batch dimension mismatch: {output.shape[0]} vs {expected_batch_size}"
    
    return output


def _test_gradients(model, input_data, loss_fn=None):
    """Test gradient computation with respect to model parameters."""
    if loss_fn is None:
        # Simple MSE loss with zero target
        def loss_fn(model, x):
            # Use vmap for batched input, single call for single examples
            if x.ndim > 1 and x.shape[0] > 1:
                output = jax.vmap(model)(x)
            else:
                if x.ndim > 1 and x.shape[0] == 1:
                    x = x[0]  # Remove batch dimension for single example
                output = model(x)
            return jnp.mean(output**2)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(model, input_data)
    
    # Verify gradients exist and are finite
    def check_gradients(grad_tree):
        if isinstance(grad_tree, jnp.ndarray):
            assert jnp.all(jnp.isfinite(grad_tree)), "Gradients contain non-finite values"
            assert jnp.any(grad_tree != 0), "All gradients are zero"
        elif hasattr(grad_tree, '__dict__'):
            for attr_name, attr_value in grad_tree.__dict__.items():
                if attr_value is not None:
                    check_gradients(attr_value)
    
    check_gradients(gradients)
    return gradients


# Functional tests that actually run the transpiled modules
def test_mlp_instantiation_and_execution(resources_dir):
    """Test that the transpiled MLP can be instantiated and executed."""
    # Import the transpiled module
    module_path = resources_dir / "../transpiled/equinox_mlp.py"
    assert module_path.exists(), "Transpiled MLP module not found"
    
    module = _import_transpiled_module(module_path)
    
    # Test instantiation
    key = jax.random.PRNGKey(42)
    input_size, hidden_size, output_size = 10, 20, 5
    
    model = module.MLP(input_size, hidden_size, output_size, key=key)
    
    # Verify model structure
    assert hasattr(model, 'fc1'), "Model should have fc1 layer"
    assert hasattr(model, 'fc2'), "Model should have fc2 layer"
    
    # Test forward pass with different batch sizes
    for batch_size in [1, 5, 10]:
        key, subkey = jax.random.split(key)
        input_data = _generate_random_input((batch_size, input_size), subkey)
        
        output = _test_forward_pass(model, input_data)
        
        # Verify output shape
        if batch_size == 1:
            # Single example - no batch dimension expected
            expected_shape = (output_size,)
        else:
            # Multiple examples - batch dimension expected
            expected_shape = (batch_size, output_size)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test gradients
    key, subkey = jax.random.split(key)
    test_input = _generate_random_input((3, input_size), subkey)
    _test_gradients(model, test_input)


def test_simple_custom_mlp_instantiation_and_execution(resources_dir):
    """Test that the transpiled custom MLP can be instantiated and executed."""
    # Import the transpiled module
    module_path = resources_dir / "../transpiled/equinox_simple_custom_mlp.py"
    assert module_path.exists(), "Transpiled custom MLP module not found"
    
    module = _import_transpiled_module(module_path)
    
    # Test CustomLinear instantiation
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    custom_linear = module.CustomLinear(10, 5, key=subkey)
    assert hasattr(custom_linear, 'weight'), "CustomLinear should have weight"
    assert hasattr(custom_linear, 'bias'), "CustomLinear should have bias"
    assert custom_linear.weight.shape == (5, 10), f"Weight shape should be (5, 10), got {custom_linear.weight.shape}"
    assert custom_linear.bias.shape == (5,), f"Bias shape should be (5,), got {custom_linear.bias.shape}"
    
    # Test CustomLinear forward pass
    key, subkey = jax.random.split(key)
    test_input = _generate_random_input((3, 10), subkey)
    output = custom_linear(test_input)
    assert output.shape == (3, 5), f"CustomLinear output shape should be (3, 5), got {output.shape}"
    
    # Test SimpleMLP instantiation
    key, subkey = jax.random.split(key)
    input_size, hidden_size, output_size = 8, 16, 4
    
    model = module.SimpleMLP(input_size, hidden_size, output_size, key=subkey)
    
    # Verify model structure
    assert hasattr(model, 'layer1'), "Model should have layer1"
    assert hasattr(model, 'layer2'), "Model should have layer2"
    assert isinstance(model.layer1, module.CustomLinear), "layer1 should be CustomLinear"
    assert isinstance(model.layer2, module.CustomLinear), "layer2 should be CustomLinear"
    
    # Test forward pass with different batch sizes
    for batch_size in [1, 5, 10]:
        key, subkey = jax.random.split(key)
        input_data = _generate_random_input((batch_size, input_size), subkey)
        
        output = _test_forward_pass(model, input_data)
        
        # Verify output shape
        expected_shape = (batch_size, output_size)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test gradients
    key, subkey = jax.random.split(key)
    test_input = _generate_random_input((2, input_size), subkey)
    _test_gradients(model, test_input)


def test_cnn_instantiation_and_execution(resources_dir):
    """Test that the transpiled CNN can be instantiated and executed."""
    # Import the transpiled module
    module_path = resources_dir / "../transpiled/equinox_cnn.py"
    assert module_path.exists(), "Transpiled CNN module not found"
    
    module = _import_transpiled_module(module_path)
    
    # Test instantiation
    key = jax.random.PRNGKey(42)
    in_channels, num_classes = 3, 10
    
    model = module.SimpleCNN(in_channels, num_classes, key=key)
    
    # Verify model structure
    assert hasattr(model, 'conv1'), "Model should have conv1 layer"
    assert hasattr(model, 'conv2'), "Model should have conv2 layer"
    assert hasattr(model, 'fc'), "Model should have fc layer"
    
    # Test forward pass with different batch sizes
    # Note: CNN is designed for 28x28 images (hardcoded Linear layer size)
    test_cases = [
        (1, 3, 28, 28),   # Single image
        (5, 3, 28, 28),   # Small batch
        (2, 3, 28, 28),   # Another batch size
    ]
    
    for batch_size, channels, height, width in test_cases:
        key, subkey = jax.random.split(key)
        input_data = _generate_random_input((batch_size, channels, height, width), subkey)
        
        output = _test_forward_pass(model, input_data)
        
        # Verify output shape
        if batch_size == 1:
            # Single example - no batch dimension expected
            expected_shape = (num_classes,)
        else:
            # Multiple examples - batch dimension expected
            expected_shape = (batch_size, num_classes)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Test gradients
    key, subkey = jax.random.split(key)
    test_input = _generate_random_input((2, 3, 28, 28), subkey)
    _test_gradients(model, test_input)


def test_parameter_types_and_initialization():
    """Test that all model parameters are properly initialized as JAX arrays."""
    key = jax.random.PRNGKey(123)
    
    # Test MLP parameters
    module_path = Path(__file__).parent / "../transpiled/equinox_mlp.py"
    if module_path.exists():
        module = _import_transpiled_module(module_path)
        key, subkey = jax.random.split(key)
        model = module.MLP(10, 20, 5, key=subkey)
        
        # Check that all parameters are JAX arrays
        def check_params(obj):
            if isinstance(obj, jnp.ndarray):
                assert obj.dtype in [jnp.float32, jnp.float64], f"Parameter should be float, got {obj.dtype}"
                assert jnp.all(jnp.isfinite(obj)), "Parameter contains non-finite values"
            elif hasattr(obj, '__dict__'):
                for attr_value in obj.__dict__.values():
                    if attr_value is not None:
                        check_params(attr_value)
        
        check_params(model)
    
    # Test Custom MLP parameters
    module_path = Path(__file__).parent / "../transpiled/equinox_simple_custom_mlp.py"
    if module_path.exists():
        module = _import_transpiled_module(module_path)
        key, subkey = jax.random.split(key)
        model = module.SimpleMLP(8, 16, 4, key=subkey)
        
        # Verify custom linear layers have proper JAX arrays
        assert isinstance(model.layer1.weight, jnp.ndarray), "layer1.weight should be jnp.ndarray"
        assert isinstance(model.layer1.bias, jnp.ndarray), "layer1.bias should be jnp.ndarray"
        assert isinstance(model.layer2.weight, jnp.ndarray), "layer2.weight should be jnp.ndarray"
        assert isinstance(model.layer2.bias, jnp.ndarray), "layer2.bias should be jnp.ndarray"


def test_model_compilation_and_jit():
    """Test that transpiled models can be JIT compiled."""
    key = jax.random.PRNGKey(456)
    
    # Test MLP JIT compilation
    module_path = Path(__file__).parent / "../transpiled/equinox_mlp.py"
    if module_path.exists():
        module = _import_transpiled_module(module_path)
        key, subkey = jax.random.split(key)
        model = module.MLP(10, 20, 5, key=subkey)
        
        # Test JIT compiled forward pass
        key, subkey = jax.random.split(key)
        test_input = _generate_random_input((3, 10), subkey)
        
        # Compare JIT and non-JIT outputs using vmap for batched input
        regular_output = jax.vmap(model)(test_input)
        
        # JIT compile the vmapped function
        jit_vmapped_model = jax.jit(jax.vmap(model))
        jit_output = jit_vmapped_model(test_input)
        
        # Should be identical (within floating point precision)
        assert jnp.allclose(regular_output, jit_output, rtol=1e-6), "JIT and regular outputs should match"
