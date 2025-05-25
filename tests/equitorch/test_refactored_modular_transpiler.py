"""
Test suite for the refactored modular transpiler.

This test suite verifies that the modular converter approach works correctly
and produces the same results as the original simplified transpiler.
"""

import pytest
import ast
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from equitorch.simple_complete_transpiler_refactored import (
    PyTorchToEquinoxTransformer,
    simple_complete_transpile
)

# Also import the converters directly for testing
from equitorch.converters import (
    is_supported_module,
    get_module_converter,
    transform_functional_call,
    transform_method_call,
    is_tensor_method_call,
    get_method_name,
)


class TestModularConverters:
    """Test the individual modular converters."""
    
    def test_module_converter_support(self):
        """Test module converter support detection."""
        assert is_supported_module('Linear')
        assert is_supported_module('Conv2d')
        assert is_supported_module('BatchNorm2d')
        assert is_supported_module('Dropout')
        assert not is_supported_module('UnsupportedModule')
    
    def test_get_module_converter(self):
        """Test getting module converters."""
        linear_converter = get_module_converter('Linear')
        assert callable(linear_converter)
        
        conv_converter = get_module_converter('Conv2d')
        assert callable(conv_converter)
        
        with pytest.raises(KeyError):
            get_module_converter('UnsupportedModule')
    
    def test_functional_call_transformation(self):
        """Test functional call transformation."""
        # Create a simple F.relu call
        relu_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='F', ctx=ast.Load()),
                attr='relu',
                ctx=ast.Load()
            ),
            args=[ast.Name(id='x', ctx=ast.Load())],
            keywords=[]
        )
        
        result = transform_functional_call(relu_node, 'F', 'relu')
        
        # Should transform to jax.nn.relu
        assert isinstance(result.func, ast.Attribute)
        assert result.func.attr == 'relu'
        assert isinstance(result.func.value, ast.Attribute)
        assert result.func.value.attr == 'nn'
    
    def test_method_call_detection(self):
        """Test tensor method call detection."""
        # Create x.view() call
        view_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='x', ctx=ast.Load()),
                attr='view',
                ctx=ast.Load()
            ),
            args=[ast.Constant(value=-1), ast.Constant(value=784)],
            keywords=[]
        )
        
        assert is_tensor_method_call(view_node)
        assert get_method_name(view_node) == 'view'
        
        # Create self.conv() call (should not be tensor method)
        self_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr='conv',
                ctx=ast.Load()
            ),
            args=[ast.Name(id='x', ctx=ast.Load())],
            keywords=[]
        )
        
        assert not is_tensor_method_call(self_node)


class TestRefactoredTranspiler:
    """Test the refactored transpiler functionality."""
    
    def test_simple_linear_model(self):
        """Test transpilation of a simple linear model."""
        pytorch_code = """
import torch
import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)
"""
        
        result = simple_complete_transpile(pytorch_code)
        
        # Verify the result contains expected Equinox imports
        assert "import jax" in result
        assert "import jax.numpy as jnp" in result
        assert "import equinox as eqx" in result
        
        # Verify the class structure
        assert "class SimpleLinear(eqx.Module):" in result
        assert "linear: eqx.nn.Linear" in result
        assert "jax.vmap(self.linear)" in result
        assert "def __call__(self, x):" in result
    
    def test_complex_cnn_model(self):
        """Test transpilation of a complex CNN model with nested expressions."""
        pytorch_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Complex nested expressions that should be handled naturally
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(F.relu(self.conv2(x)), p=0.3)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""
        
        result = simple_complete_transpile(pytorch_code)
        
        # Verify complex expressions are handled naturally (no temp variables)
        assert "jax.nn.relu(jax.vmap(self.conv1)(x))" in result
        assert "jax.nn.dropout(jax.nn.relu(jax.vmap(self.conv2)(x))" in result
        assert "jnp.reshape" in result
        
        # Verify all modules are converted
        assert "conv1: eqx.nn.Conv2d" in result
        assert "conv2: eqx.nn.Conv2d" in result
        assert "fc1: eqx.nn.Linear" in result
        assert "fc2: eqx.nn.Linear" in result
        assert "dropout: eqx.nn.Dropout" in result
        
        # Verify no temporary variables are created
        assert "temp_" not in result
    
    def test_modular_converter_usage(self):
        """Test that the refactored transpiler uses modular converters correctly."""
        pytorch_code = """
import torch.nn as nn
import torch.nn.functional as F

class ModularTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, x):
        x = F.relu(x)
        x = x.view(-1, 784)
        return self.linear(x)
"""
        
        result = simple_complete_transpile(pytorch_code)
        
        # Verify modular converters are working
        assert "eqx.nn.Linear" in result  # Module converter
        assert "eqx.nn.Conv2d" in result  # Module converter
        assert "jax.nn.relu(x)" in result  # Functional converter
        assert "jnp.reshape(x, (-1, 784))" in result  # Method converter
        assert "jax.vmap(self.linear)" in result  # Module call converter
    
    def test_comparison_with_original(self):
        """Test that refactored version produces same results as original."""
        from equitorch.simple_complete_transpiler import simple_complete_transpile as original_transpile
        
        pytorch_code = """
import torch.nn as nn
import torch.nn.functional as F

class ComparisonTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)
"""
        
        original_result = original_transpile(pytorch_code)
        refactored_result = simple_complete_transpile(pytorch_code)
        
        # Both should produce equivalent transformations
        # (exact string comparison might differ due to formatting, so check key components)
        
        # Check that both have the same key transformations
        key_patterns = [
            "class ComparisonTest(eqx.Module):",
            "conv: eqx.nn.Conv2d",
            "fc: eqx.nn.Linear",
            "jax.nn.relu(jax.vmap(self.conv)(x))",
            "jnp.reshape(x, (x.shape[0], -1))",
            "jax.vmap(self.fc)(x)"
        ]
        
        for pattern in key_patterns:
            assert pattern in original_result, f"Original missing: {pattern}"
            assert pattern in refactored_result, f"Refactored missing: {pattern}"
    
    def test_ast_transformer_directly(self):
        """Test the AST transformer directly."""
        pytorch_code = """
import torch.nn as nn
import torch.nn.functional as F

class DirectTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        return x
"""
        
        # Parse and transform
        tree = ast.parse(pytorch_code)
        transformer = PyTorchToEquinoxTransformer()
        new_tree = transformer.visit(tree)
        
        # Verify AST structure
        assert isinstance(new_tree, ast.Module)
        
        # Find the class
        class_node = None
        for node in new_tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "DirectTest":
                class_node = node
                break
        
        assert class_node is not None
        assert len(class_node.bases) == 1
        assert isinstance(class_node.bases[0], ast.Attribute)
        assert class_node.bases[0].attr == "Module"


class TestModularArchitecture:
    """Test the modular architecture benefits."""
    
    def test_converter_separation(self):
        """Test that converters are properly separated."""
        # Test that we can import converters independently
        from equitorch.converters.module_converters import convert_linear
        from equitorch.converters.functional_converters import transform_torch_flatten
        from equitorch.converters.method_converters import transform_reshape
        
        # All should be callable
        assert callable(convert_linear)
        assert callable(transform_torch_flatten)
        assert callable(transform_reshape)
    
    def test_extensibility(self):
        """Test that the modular approach is extensible."""
        # We should be able to check what's supported
        from equitorch.converters import (
            get_supported_modules,
            get_supported_functionals,
            get_supported_methods
        )
        
        modules = get_supported_modules()
        functionals = get_supported_functionals()
        methods = get_supported_methods()
        
        assert 'Linear' in modules
        assert 'Conv2d' in modules
        assert ('F', 'relu') in functionals
        assert ('torch', 'flatten') in functionals
        assert 'view' in methods
        assert 'flatten' in methods
    
    def test_clean_main_transpiler(self):
        """Test that the main transpiler is now cleaner."""
        # The refactored transpiler should be significantly shorter
        with open('src/equitorch/simple_complete_transpiler_refactored.py', 'r') as f:
            refactored_lines = len(f.readlines())
        
        with open('src/equitorch/simple_complete_transpiler.py', 'r') as f:
            original_lines = len(f.readlines())
        
        # Refactored should be shorter (main logic is cleaner)
        assert refactored_lines < original_lines
        
        # But total functionality should be the same or better
        # (this is tested by the comparison tests above)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
