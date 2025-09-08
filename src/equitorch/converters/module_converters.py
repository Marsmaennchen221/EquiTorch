"""Module converters for PyTorch to Equinox transformations.

This module contains both the mappings and implementations for converting
PyTorch nn.Module instances to their Equinox equivalents.

Refactored to follow the same modular architecture as functional_converters and method_converters.
"""

import ast
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch.nn as nn
else:
    try:
        import torch.nn as nn
    except ImportError:
        nn = None


# ============================================================================
# SIMPLE MODULE MAPPINGS - Direct 1:1 conversions
# ============================================================================

SIMPLE_MODULE_MAPPINGS: dict[str, str] = {
    "Linear": "eqx.nn.Linear",
    "Conv1d": "eqx.nn.Conv1d",
    "Conv2d": "eqx.nn.Conv2d",
    "Conv3d": "eqx.nn.Conv3d",
    "BatchNorm1d": "eqx.nn.BatchNorm1d",
    "BatchNorm2d": "eqx.nn.BatchNorm2d",
    "BatchNorm3d": "eqx.nn.BatchNorm3d",
    "Dropout": "eqx.nn.Dropout",
    "LayerNorm": "eqx.nn.LayerNorm",
    "Embedding": "eqx.nn.Embedding",
}


# ============================================================================
# SIMPLE MODULE CONFIGURATIONS - For modules needing basic parameter handling
# ============================================================================

@dataclass
class SimpleModuleConfig:
    """Configuration for simple module conversions."""
    target_module: str
    default_parameters: Optional[dict[str, Any]] = None
    parameter_filters: Optional[set[str]] = None
    args_slice: Optional[slice] = None


SIMPLE_MODULE_CONFIGS: dict[str, SimpleModuleConfig] = {
    "Linear": SimpleModuleConfig(
        target_module="eqx.nn.Linear",
        default_parameters={"use_bias": True}
    ),
    "Conv1d": SimpleModuleConfig(
        target_module="eqx.nn.Conv1d",
        default_parameters={"use_bias": True}
    ),
    "Conv2d": SimpleModuleConfig(
        target_module="eqx.nn.Conv2d",
        default_parameters={"use_bias": True}
    ),
    "Conv3d": SimpleModuleConfig(
        target_module="eqx.nn.Conv3d",
        default_parameters={"use_bias": True}
    ),
    "BatchNorm1d": SimpleModuleConfig(target_module="eqx.nn.BatchNorm1d"),
    "BatchNorm2d": SimpleModuleConfig(target_module="eqx.nn.BatchNorm2d"),
    "BatchNorm3d": SimpleModuleConfig(target_module="eqx.nn.BatchNorm3d"),
    "Dropout": SimpleModuleConfig(
        target_module="eqx.nn.Dropout",
        parameter_filters={"key"}  # Dropout doesn't take a key parameter
    ),
    "LayerNorm": SimpleModuleConfig(
        target_module="eqx.nn.LayerNorm",
        parameter_filters={"elementwise_affine", "eps", "key"}  # Filter PyTorch-specific params and key
    ),
    "Embedding": SimpleModuleConfig(
        target_module="eqx.nn.Embedding",
        parameter_filters={"padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse"}
    ),
    "LSTM": SimpleModuleConfig(
        target_module="eqx.nn.LSTMCell",
        args_slice=slice(0, 2)  # Only take input_size, hidden_size
    ),
}


# ============================================================================
# COMPLEX MODULE TRANSFORMERS - For modules requiring custom logic
# ============================================================================

CUSTOM_MODULE_TRANSFORMERS: dict[str, Callable[[ast.Call, int], tuple[ast.expr, ast.AST]]] = {
    "MultiheadAttention": lambda node, key_counter: transform_multihead_attention(node, key_counter),
    "RNN": lambda node, key_counter: transform_rnn_module(node, key_counter),
    "Parameter": lambda node, key_counter: transform_parameter(node, key_counter),
    "Sequential": lambda node, key_counter: transform_sequential(node, key_counter),
    "ModuleList": lambda node, key_counter: transform_module_list(node, key_counter),
    "ReLU": lambda node, key_counter: transform_relu(node, key_counter),
    "Dropout": lambda node, key_counter: transform_dropout(node, key_counter),
}


# ============================================================================
# CUSTOM IMPLEMENTATIONS - Code that needs to be injected
# ============================================================================

def _get_elman_rnn_cell_definition() -> str:
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


CUSTOM_IMPLEMENTATIONS: dict[str, str] = {
    "ElmanRNNCell": _get_elman_rnn_cell_definition(),
}


# ============================================================================
# TRANSFORMATION FUNCTIONS - Individual transformers for complex modules
# ============================================================================

def transform_multihead_attention(node: ast.Call, key_counter: int) -> tuple[ast.Call, ast.AST]:
    """Transform MultiheadAttention to use custom libEquitorch implementation."""
    
    # Extract arguments and keywords from PyTorch call
    args = list(node.args)
    newkeywords = list(node.keywords)
    
    # Add key parameter for JAX
    newkeywords.append(ast.keyword(arg="key", value=ast.Name(id="subkey", ctx=ast.Load())))
    ign = {"batch_first", "device", "dtype", "add_bias_kv", "add_zero_attn", "kdim", "vdim"}
    keywords = [k  for k in newkeywords if k.arg not in ign]
    # Create the call to eqt.nn.MultiheadAttention (NOT eqx.nn)
    transformed_node = ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="eqt", ctx=ast.Load()),  # Use eqt
                attr="nn",
                ctx=ast.Load()
            ),
            attr="EquiTorchMultiHeadAttention",  # Use the correct class name
            ctx=ast.Load()
        ),
        args=args,
        keywords=keywords
    )
    
    # Type annotation should also use eqt.nn.MultiheadAttention
    type_annotation = ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="eqt", ctx=ast.Load()),
            attr="nn",
            ctx=ast.Load()
        ),
        attr="EquiTorchMultiHeadAttention",
        ctx=ast.Load()
    )
    
    return transformed_node, type_annotation


def transform_rnn_module(node: ast.Call, key_counter: int) -> tuple[ast.Call, ast.AST]:
    """Transform RNN module to ElmanRNNCell with proper parameter filtering."""
    # Filter out PyTorch-specific parameters
    filtered_keywords = []
    pytorch_specific_params = {"batch_first", "dropout", "bidirectional", "num_layers"}
    
    for kw in node.keywords:
        if kw.arg not in pytorch_specific_params:
            filtered_keywords.append(kw)
    
    # Add key parameter
    filtered_keywords.append(ast.keyword(arg="key", value=ast.Name(id="subkey", ctx=ast.Load())))
    
    # Select only input_size and hidden_size arguments
    args = node.args[:2] if len(node.args) >= 2 else node.args
    
    # Create ElmanRNNCell call
    transformed_node = ast.Call(
        func=ast.Name(id="ElmanRNNCell", ctx=ast.Load()),
        args=args,
        keywords=filtered_keywords
    )
    
    type_annotation = ast.Name(id="ElmanRNNCell", ctx=ast.Load())
    
    return transformed_node, type_annotation


def transform_parameter(node: ast.Call, key_counter: int) -> tuple[ast.Call | ast.expr, ast.AST]:
    """Transform nn.Parameter to jnp.ndarray with proper tensor creation."""
    transformed_node: ast.Call | ast.expr
    if node.args and len(node.args) > 0:
        inner_call = node.args[0]  # Extract inner tensor creation (e.g., torch.randn(...))
        
        # Check if this is a torch.randn call that needs conversion
        if (isinstance(inner_call, ast.Call) and 
            isinstance(inner_call.func, ast.Attribute) and
            isinstance(inner_call.func.value, ast.Name) and
            inner_call.func.value.id == "torch" and
            inner_call.func.attr == "randn"):
            
            # Convert torch.randn(shape_args...) to jax.random.normal(key, (shape_args...))
            # Create the shape tuple from the original arguments
            if len(inner_call.args) == 1:
                # Single argument - use as-is
                shape_arg = inner_call.args[0]
            else:
                # Multiple arguments - wrap in tuple
                shape_arg = ast.Tuple(elts=inner_call.args, ctx=ast.Load())
            
            # Create jax.random.normal(subkey, shape)
            transformed_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="jax", ctx=ast.Load()),
                        attr="random",
                        ctx=ast.Load()
                    ),
                    attr="normal",
                    ctx=ast.Load()
                ),
                args=[
                    ast.Name(id="subkey", ctx=ast.Load()),  # key first
                    shape_arg  # shape second
                ],
                keywords=[]
            )
        else:
            # For other inner calls, just extract them
            transformed_node = inner_call
    else:
        # No arguments, use the node as-is
        transformed_node = node
        
    type_annotation = ast.Attribute(
        value=ast.Name(id="jnp", ctx=ast.Load()),
        attr="ndarray",
        ctx=ast.Load()
    )
    
    return transformed_node, type_annotation


def transform_sequential(node: ast.Call, key_counter: int) -> tuple[ast.Call, ast.AST]:
    """Transform nn.Sequential to eqx.nn.Sequential.
    
    Equinox Sequential expects a list of layers, not individual arguments.
    """
    # Filter out any PyTorch-specific keywords (including key - Sequential doesn't need it)
    keywords = [kw for kw in node.keywords if kw.arg != "key"]
    
    # Wrap all arguments in a list - eqx.nn.Sequential expects a list of layers
    layers_list = ast.List(elts=list(node.args), ctx=ast.Load())
    
    # Create call to eqx.nn.Sequential with list of layers
    transformed_node = ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="eqx", ctx=ast.Load()),
                attr="nn",
                ctx=ast.Load()
            ),
            attr="Sequential",
            ctx=ast.Load()
        ),
        args=[layers_list],  # Pass the list as a single argument
        keywords=keywords
    )
    
    # Type annotation should also use eqx.nn.Sequential
    type_annotation = ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="eqx", ctx=ast.Load()),
            attr="nn",
            ctx=ast.Load()
        ),
        attr="Sequential",
        ctx=ast.Load()
    )
    
    return transformed_node, type_annotation


def transform_module_list(node: ast.Call, key_counter: int) -> tuple[ast.Call, ast.AST]:
    """Transform nn.ModuleList to a simple list of modules."""
    # For now, just pass through the arguments and add key parameter
    keywords = list(node.keywords)
    keywords.append(ast.keyword(arg="key", value=ast.Name(id="subkey", ctx=ast.Load())))
    
    # Use auto-transpilation for ModuleList
    transformed_node = ast.Call(
        func=ast.Name(id="ModuleList", ctx=ast.Load()),
        args=node.args,
        keywords=keywords
    )
    
    type_annotation = ast.Name(id="ModuleList", ctx=ast.Load())
    
    return transformed_node, type_annotation


def transform_relu(node: ast.Call, key_counter: int) -> tuple[ast.Attribute, ast.AST]:
    """Transform nn.ReLU to jax.nn.relu function reference for Sequential.
    
    In Equinox Sequential, activation functions should be passed as functions,
    not instantiated modules. This returns jax.nn.relu directly.
    """
    # Return jax.nn.relu as a function reference (no call, no key parameter)
    transformed_node = ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="jax", ctx=ast.Load()),
            attr="nn",
            ctx=ast.Load()
        ),
        attr="relu",
        ctx=ast.Load()
    )
    
    # Type annotation is also jax.nn.relu (function type)
    type_annotation = ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="jax", ctx=ast.Load()),
            attr="nn",
            ctx=ast.Load()
        ),
        attr="relu",
        ctx=ast.Load()
    )
    
    return transformed_node, type_annotation


def transform_dropout(node: ast.Call, key_counter: int) -> tuple[ast.Call, ast.AST]:
    """Transform nn.Dropout to eqx.nn.Dropout without key parameter.
    
    Equinox Dropout doesn't take a key parameter.
    """
    # Filter out any existing key parameters and create clean call
    filtered_keywords = [kw for kw in node.keywords if kw.arg != "key"]
    
    transformed_node = ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="eqx", ctx=ast.Load()),
                attr="nn",
                ctx=ast.Load()
            ),
            attr="Dropout",
            ctx=ast.Load()
        ),
        args=node.args,
        keywords=filtered_keywords
    )
    
    # Type annotation
    type_annotation = ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="eqx", ctx=ast.Load()),
            attr="nn",
            ctx=ast.Load()
        ),
        attr="Dropout",
        ctx=ast.Load()
    )
    
    return transformed_node, type_annotation


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _create_module_reference_ast(module_path: str) -> ast.expr:
    """Create AST reference for module path like 'eqx.nn.Linear' or 'ElmanRNNCell'."""
    if "." not in module_path:
        return ast.Name(id=module_path, ctx=ast.Load())
    
    parts = module_path.split(".")
    result: ast.Name | ast.Attribute = ast.Name(id=parts[0], ctx=ast.Load())
    for part in parts[1:]:
        result = ast.Attribute(value=result, attr=part, ctx=ast.Load())
    return result


def transform_simple_module(node: ast.Call, module_name: str, key_counter: int) -> tuple[ast.Call, ast.AST]:
    """Transform a simple module using configuration."""
    config = SIMPLE_MODULE_CONFIGS.get(module_name)
    if not config:
        # Fallback to simple mapping
        target_module = SIMPLE_MODULE_MAPPINGS.get(module_name, module_name)
        config = SimpleModuleConfig(target_module=target_module)
    
    # Filter keywords
    keywords = list(node.keywords)
    if config.parameter_filters:
        keywords = [kw for kw in keywords if kw.arg not in config.parameter_filters]
    
    # Add default parameters
    if config.default_parameters:
        for param, value in config.default_parameters.items():
            keywords.append(ast.keyword(arg=param, value=ast.Constant(value=value)))
    
    # Add key parameter only if not filtered out
    if not config.parameter_filters or "key" not in config.parameter_filters:
        keywords.append(ast.keyword(arg="key", value=ast.Name(id="subkey", ctx=ast.Load())))
    
    # Select arguments
    args = node.args
    if config.args_slice:
        args = args[config.args_slice]
    
    # Create function reference and call
    func_ref = _create_module_reference_ast(config.target_module)
    transformed_node = ast.Call(func=func_ref, args=args, keywords=keywords)
    type_annotation = _create_module_reference_ast(config.target_module)
    
    return transformed_node, type_annotation


# ============================================================================
# AUTO-TRANSPILATION SYSTEM - Separate concern for unsupported modules
# ============================================================================

# Cache for auto-transpiled modules to avoid re-transpiling
_AUTO_TRANSPILED_CACHE: dict[str, tuple[str, str]] = {}  # module_name -> (class_def, type_annotation)


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


def transform_auto_transpiled_module(node: ast.Call, module_name: str, key_counter: int) -> tuple[ast.Call, ast.AST]:
    """Transform a module using auto-transpilation."""
    # Get the auto-transpiled class info
    class_def, type_annotation = auto_transpile_module(module_name)
    
    # Create a call to the auto-transpiled class
    keywords = list(node.keywords)
    keywords.append(ast.keyword(arg="key", value=ast.Name(id="subkey", ctx=ast.Load())))
    
    transformed_node = ast.Call(
        func=ast.Name(id=type_annotation, ctx=ast.Load()),
        args=node.args,
        keywords=keywords,
    )
    
    type_annotation_ast = ast.Name(id=type_annotation, ctx=ast.Load())
    
    return transformed_node, type_annotation_ast


def _can_auto_transpile(module_name: str) -> bool:
    """Check if a module can be auto-transpiled."""
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


# ============================================================================
# PUBLIC INTERFACE - Main entry points
# ============================================================================

def transform_module_call(node: ast.Call, module_name: str, key_counter: int) -> tuple[ast.Call, ast.AST]:
    """Transform a PyTorch module call to its Equinox equivalent.
    
    This is the main entry point that follows the same pattern as transform_functional_call.
    
    Args:
        node: The AST Call node to transform
        module_name: Name of the PyTorch module (e.g., 'Linear', 'Conv2d')
        key_counter: Counter for key generation (currently unused but kept for compatibility)
        
    Returns:
        Tuple of (transformed_node, type_annotation)
        
    Raises:
        KeyError: If no converter exists and auto-transpilation fails
    """
    # Check custom transformers first
    if module_name in CUSTOM_MODULE_TRANSFORMERS:
        transformer = CUSTOM_MODULE_TRANSFORMERS[module_name]
        return transformer(node, key_counter)
    
    # Check simple configurations
    if module_name in SIMPLE_MODULE_CONFIGS:
        return transform_simple_module(node, module_name, key_counter)
    
    # Check simple mappings
    if module_name in SIMPLE_MODULE_MAPPINGS:
        return transform_simple_module(node, module_name, key_counter)
    
    # Fallback to auto-transpilation
    try:
        return transform_auto_transpiled_module(node, module_name, key_counter)
    except Exception as e:
        raise KeyError(f"No converter available for module {module_name} and auto-transpilation failed: {e}") from e


def get_module_converter(module_name: str) -> Callable[[ast.Call, int], tuple[ast.Call, ast.AST]]:
    """Get the converter function for a given PyTorch module name.
    
    Maintained for backward compatibility with existing transpiler code.
    
    Args:
        module_name: Name of the PyTorch module (e.g., 'Linear', 'Conv2d')
        
    Returns:
        Converter function that takes (node, key_counter) and returns (transformed_node, type_annotation)
        
    Raises:
        KeyError: If no converter exists and auto-transpilation fails
    """
    def converter(node: ast.Call, key_counter: int) -> tuple[ast.Call, ast.AST]:
        return transform_module_call(node, module_name, key_counter)
    
    return converter


def is_supported_module(module_name: str) -> bool:
    """Check if a PyTorch module is supported for conversion.
    
    Checks custom transformers, simple configs, simple mappings, and auto-transpilation capability.
    
    Args:
        module_name: Name of the PyTorch module
        
    Returns:
        True if the module can be converted, False otherwise
    """
    # Check custom transformers
    if module_name in CUSTOM_MODULE_TRANSFORMERS:
        return True
    
    # Check simple configurations
    if module_name in SIMPLE_MODULE_CONFIGS:
        return True
    
    # Check simple mappings
    if module_name in SIMPLE_MODULE_MAPPINGS:
        return True
    
    # Check if auto-transpilation is possible
    return _can_auto_transpile(module_name)


def get_required_custom_implementations(pytorch_modules: set[str]) -> dict[str, str]:
    """Get all custom implementations required for the given PyTorch modules.
    
    Args:
        pytorch_modules: Set of PyTorch module names that were converted
        
    Returns:
        Dict mapping custom implementation names to their definitions
    """
    required_implementations = {}
    
    for pytorch_module in pytorch_modules:
        # Check if this module requires a custom implementation
        if pytorch_module == "RNN":
            required_implementations["ElmanRNNCell"] = CUSTOM_IMPLEMENTATIONS["ElmanRNNCell"]
    
    return required_implementations


def get_supported_modules() -> list[str]:
    """Get a list of all supported PyTorch module names.
    
    Returns:
        List of supported module names
    """
    supported: set[str] = set()
    supported.update(CUSTOM_MODULE_TRANSFORMERS.keys())
    supported.update(SIMPLE_MODULE_CONFIGS.keys())
    supported.update(SIMPLE_MODULE_MAPPINGS.keys())
    return list(supported)
