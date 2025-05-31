"""Module converters for PyTorch to Equinox transformations.

This module contains both the mappings and implementations for converting
PyTorch nn.Module instances to their Equinox equivalents.
"""

import ast
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
else:
    try:
        import torch.nn as nn
    except ImportError:
        nn = None

# Mapping from PyTorch module names to converter functions
MODULE_MAPPINGS: dict[str, str] = {
    "Linear": "convert_linear",
    "Conv1d": "convert_conv",
    "Conv2d": "convert_conv",
    "Conv3d": "convert_conv",
    "BatchNorm1d": "convert_batchnorm",
    "BatchNorm2d": "convert_batchnorm",
    "BatchNorm3d": "convert_batchnorm",
    "Dropout": "convert_dropout",
    "RNN": "convert_rnn",
    "LSTM": "convert_lstm",
    "Parameter": "convert_parameter",
}

# Cache for auto-transpiled modules to avoid re-transpiling
_AUTO_TRANSPILED_CACHE: dict[str, tuple[str, str]] = {}  # module_name -> (class_def, type_annotation)

# Registry for custom module implementations that can't be directly transpiled
CUSTOM_MODULE_IMPLEMENTATIONS: dict[str, str] = {
    "ElmanRNNCell": "get_elman_rnn_cell_definition",
    # Add more custom implementations here as needed
    # "CustomLSTMCell": "get_custom_lstm_cell_definition",
    # "CustomAttention": "get_custom_attention_definition",
}

# Mapping from PyTorch modules to their custom implementation names
PYTORCH_TO_CUSTOM_MAPPING: dict[str, str] = {
    "RNN": "ElmanRNNCell",
    # Add more mappings here as needed
    # "LSTM": "CustomLSTMCell",  # if we want custom LSTM instead of eqx.nn.LSTMCell
}


def auto_transpile_module(module_name: str) -> tuple[str, str]:
    """Auto-transpile a PyTorch module by recursively transpiling its source.
    
    Args:
        module_name: Name of the PyTorch module (e.g., 'GRU', 'MultiheadAttention')
        
    Returns:
        Tuple of (class_definition, type_annotation)
        
    Raises:
        ValueError: If the module cannot be auto-transpiled
    """
    if module_name in _AUTO_TRANSPILED_CACHE:
        return _AUTO_TRANSPILED_CACHE[module_name]
    
    if nn is None:
        raise ValueError("PyTorch not available for auto-transpilation")
    
    try:
        # Get the PyTorch module class
        pytorch_class = getattr(nn, module_name)
        
        # Get the source code
        source = inspect.getsource(pytorch_class)
        
        # Import the transpiler (avoid circular import)
        from ..transpiler import transpile
        
        # Recursively transpile the PyTorch source
        equinox_code = transpile(source)
        
        # Extract the class definition and type annotation
        class_def, type_annotation = extract_class_info(equinox_code, module_name)
        
        # Cache the result
        _AUTO_TRANSPILED_CACHE[module_name] = (class_def, type_annotation)
        
        return class_def, type_annotation
        
    except Exception as e:
        raise ValueError(f"Failed to auto-transpile {module_name}: {e}") from e


def extract_class_info(transpiled_code: str, original_name: str) -> tuple[str, str]:
    """Extract class definition and type annotation from transpiled code.
    
    Args:
        transpiled_code: The full transpiled module code
        original_name: Original PyTorch module name
        
    Returns:
        Tuple of (class_definition, type_annotation)
    """
    # Parse the transpiled code to extract the class
    tree = ast.parse(transpiled_code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Found a class definition
            class_name = node.name
            class_def = ast.unparse(node)
            type_annotation = class_name  # Use the class name as type annotation
            return class_def, type_annotation
    
    # Fallback: return the whole code and use original name
    return transpiled_code, original_name


def get_custom_module_definition(custom_name: str) -> str:
    """Get the definition for a custom module implementation.
    
    Args:
        custom_name: Name of the custom implementation (e.g., 'ElmanRNNCell')
        
    Returns:
        Class definition string
        
    Raises:
        ValueError: If the custom implementation is not found
    """
    if custom_name not in CUSTOM_MODULE_IMPLEMENTATIONS:
        raise ValueError(f"Custom implementation '{custom_name}' not found")
    
    function_name = CUSTOM_MODULE_IMPLEMENTATIONS[custom_name]
    function = globals()[function_name]
    return function()


def create_custom_module_converter(pytorch_module: str) -> Callable[[ast.Call, int], ast.Call]:
    """Create a converter function for a custom module implementation.
    
    Args:
        pytorch_module: Name of the PyTorch module (e.g., 'RNN')
        
    Returns:
        Converter function
        
    Raises:
        ValueError: If no custom implementation is mapped for this PyTorch module
    """
    if pytorch_module not in PYTORCH_TO_CUSTOM_MAPPING:
        raise ValueError(f"No custom implementation mapped for PyTorch module '{pytorch_module}'")
    
    custom_name = PYTORCH_TO_CUSTOM_MAPPING[pytorch_module]
    
    def converter(node: ast.Call, key_counter: int) -> ast.Call:
        key_name = f"key_{key_counter}"
        keywords = list(node.keywords)
        keywords.append(ast.keyword(arg="key", value=ast.Name(id=key_name, ctx=ast.Load())))
        
        return ast.Call(
            func=ast.Name(id=custom_name, ctx=ast.Load()),
            args=node.args[:2],  # Most RNN-like modules take input_size, hidden_size
            keywords=keywords,
        )
    
    return converter


def get_required_custom_implementations(pytorch_modules: set[str]) -> dict[str, str]:
    """Get all custom implementations required for the given PyTorch modules.
    
    Args:
        pytorch_modules: Set of PyTorch module names that were converted
        
    Returns:
        Dict mapping custom implementation names to their definitions
    """
    required_implementations = {}
    
    for pytorch_module in pytorch_modules:
        if pytorch_module in PYTORCH_TO_CUSTOM_MAPPING:
            custom_name = PYTORCH_TO_CUSTOM_MAPPING[pytorch_module]
            if custom_name not in required_implementations:
                try:
                    definition = get_custom_module_definition(custom_name)
                    required_implementations[custom_name] = definition
                except ValueError:
                    # Skip if definition not found
                    pass
    
    return required_implementations


def create_auto_transpiled_converter(module_name: str) -> Callable[[ast.Call, int], ast.Call]:
    """Create a converter function for an auto-transpiled module.
    
    Args:
        module_name: Name of the PyTorch module
        
    Returns:
        Converter function
    """
    def converter(node: ast.Call, key_counter: int) -> ast.Call:
        # Get the auto-transpiled class info
        class_def, type_annotation = auto_transpile_module(module_name)
        
        # Create a call to the auto-transpiled class
        key_name = f"key_{key_counter}"
        keywords = list(node.keywords)
        keywords.append(ast.keyword(arg="key", value=ast.Name(id=key_name, ctx=ast.Load())))
        
        return ast.Call(
            func=ast.Name(id=type_annotation, ctx=ast.Load()),
            args=node.args,
            keywords=keywords,
        )
    
    return converter


def convert_linear(node: ast.Call, key_counter: int) -> ast.Call:
    """Convert nn.Linear to eqx.nn.Linear."""
    key_name = f"key_{key_counter}"
    
    # Add key split before this (would need to be handled in parent context)
    keywords = list(node.keywords)
    keywords.append(ast.keyword(arg="use_bias", value=ast.Constant(value=True)))
    keywords.append(ast.keyword(arg="key", value=ast.Name(id=key_name, ctx=ast.Load())))
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="eqx", ctx=ast.Load()),
                attr="nn",
                ctx=ast.Load(),
            ),
            attr="Linear",
            ctx=ast.Load(),
        ),
        args=node.args,
        keywords=keywords,
    )


def convert_conv(node: ast.Call, key_counter: int) -> ast.Call:
    """Convert nn.Conv* to eqx.nn.Conv*."""
    conv_type = node.func.attr  # type: ignore[attr-defined]
    key_name = f"key_{key_counter}"
    
    keywords = list(node.keywords)
    keywords.append(ast.keyword(arg="use_bias", value=ast.Constant(value=True)))
    keywords.append(ast.keyword(arg="key", value=ast.Name(id=key_name, ctx=ast.Load())))
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="eqx", ctx=ast.Load()),
                attr="nn",
                ctx=ast.Load(),
            ),
            attr=conv_type,
            ctx=ast.Load(),
        ),
        args=node.args,
        keywords=keywords,
    )


def convert_batchnorm(node: ast.Call, key_counter: int) -> ast.Call:
    """Convert nn.BatchNorm* to eqx.nn.BatchNorm*."""
    bn_type = node.func.attr  # type: ignore[attr-defined]
    key_name = f"key_{key_counter}"
    
    keywords = list(node.keywords)
    keywords.append(ast.keyword(arg="key", value=ast.Name(id=key_name, ctx=ast.Load())))
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="eqx", ctx=ast.Load()),
                attr="nn",
                ctx=ast.Load(),
            ),
            attr=bn_type,
            ctx=ast.Load(),
        ),
        args=node.args,
        keywords=keywords,
    )


def convert_dropout(node: ast.Call, key_counter: int) -> ast.Call:
    """Convert nn.Dropout to eqx.nn.Dropout."""
    key_name = f"key_{key_counter}"
    
    keywords = list(node.keywords)
    keywords.append(ast.keyword(arg="key", value=ast.Name(id=key_name, ctx=ast.Load())))
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="eqx", ctx=ast.Load()),
                attr="nn",
                ctx=ast.Load(),
            ),
            attr="Dropout",
            ctx=ast.Load(),
        ),
        args=node.args,
        keywords=keywords,
    )


def get_module_converter(module_name: str) -> Callable[[ast.Call, int], ast.Call]:
    """Get the converter function for a given PyTorch module name.
    
    First tries manual converters, then falls back to auto-transpilation.
    
    Args:
        module_name: Name of the PyTorch module (e.g., 'Linear', 'Conv2d')
        
    Returns:
        Converter function that takes (node, key_counter) and returns transformed node
        
    Raises:
        KeyError: If no converter exists and auto-transpilation fails
    """
    # Try manual converter first
    if module_name in MODULE_MAPPINGS:
        converter_name = MODULE_MAPPINGS[module_name]
        return globals()[converter_name]
    
    # Fallback to auto-transpilation
    try:
        return create_auto_transpiled_converter(module_name)
    except Exception as e:
        raise KeyError(f"No converter available for module {module_name} and auto-transpilation failed: {e}") from e


def is_supported_module(module_name: str) -> bool:
    """Check if a PyTorch module is supported for conversion.
    
    Checks both manual converters and auto-transpilation capability.
    
    Args:
        module_name: Name of the PyTorch module
        
    Returns:
        True if the module can be converted, False otherwise
    """
    # Check manual converters first
    if module_name in MODULE_MAPPINGS:
        return True
    
    # Check if auto-transpilation is possible
    if nn is None:
        return False
    
    try:
        # Check if the module exists in PyTorch
        pytorch_class = getattr(nn, module_name, None)
        if pytorch_class is None:
            return False
        
        # Check if it's actually a class
        if not isinstance(pytorch_class, type):
            return False
        
        # Check if we can get source code
        inspect.getsource(pytorch_class)
        return True
    except (AttributeError, OSError, TypeError):
        return False


def convert_rnn(node: ast.Call, key_counter: int) -> ast.Call:
    """Convert nn.RNN using the generic custom module system."""
    return create_custom_module_converter("RNN")(node, key_counter)


def convert_lstm(node: ast.Call, key_counter: int) -> ast.Call:
    """Convert nn.LSTM to eqx.nn.LSTMCell."""
    key_name = f"key_{key_counter}"
    
    keywords = list(node.keywords)
    keywords.append(ast.keyword(arg="key", value=ast.Name(id=key_name, ctx=ast.Load())))
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="eqx", ctx=ast.Load()),
                attr="nn",
                ctx=ast.Load(),
            ),
            attr="LSTMCell",
            ctx=ast.Load(),
        ),
        args=node.args[:2],  # Only take input_size and hidden_size
        keywords=keywords,
    )


def convert_parameter(node: ast.Call, key_counter: int) -> ast.Call:  # noqa: ARG001
    """Convert nn.Parameter to the inner tensor creation.
    
    For nn.Parameter, we extract the inner tensor creation and convert it.
    """
    if node.args:
        return node.args[0]  # Return the tensor creation call (e.g., torch.randn)
    return node


def get_elman_rnn_cell_definition() -> str:
    """Return the Elman RNN cell class definition to inject into transpiled code."""
    return '''
class ElmanRNNCell(eqx.Module):
    """Elman RNN Cell implementation matching PyTorch's nn.RNN behavior."""
    input_size: int
    hidden_size: int
    weight_ih: jnp.ndarray  # input-to-hidden weights
    weight_hh: jnp.ndarray  # hidden-to-hidden weights
    bias_ih: jnp.ndarray    # input-to-hidden bias
    bias_hh: jnp.ndarray    # hidden-to-hidden bias
    
    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Initialize weights similar to PyTorch's default initialization
        std = 1.0 / (hidden_size) ** 0.5
        self.weight_ih = jax.random.uniform(k1, (hidden_size, input_size), minval=-std, maxval=std)
        self.weight_hh = jax.random.uniform(k2, (hidden_size, hidden_size), minval=-std, maxval=std)
        self.bias_ih = jax.random.uniform(k3, (hidden_size,), minval=-std, maxval=std)
        self.bias_hh = jax.random.uniform(k4, (hidden_size,), minval=-std, maxval=std)
    
    def __call__(self, input: jnp.ndarray, hidden: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh)"""
        return jnp.tanh(
            jnp.dot(self.weight_ih, input) + 
            jnp.dot(self.weight_hh, hidden) + 
            self.bias_ih + self.bias_hh
        )
'''


def get_supported_modules() -> list[str]:
    """Get a list of all supported PyTorch module names.
    
    Returns:
        List of supported module names
    """
    return list(MODULE_MAPPINGS.keys())
