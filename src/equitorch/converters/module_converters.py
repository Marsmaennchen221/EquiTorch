"""Module converters for PyTorch to Equinox transformations.

This module contains both the mappings and implementations for converting
PyTorch nn.Module instances to their Equinox equivalents.
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


@dataclass
class ModuleConfig:
    """Configuration for module conversion."""
    target_module: str  # e.g., "eqx.nn.Linear", "ElmanRNNCell"
    custom_definition: Optional[str] = None  # For custom implementations
    parameter_filters: Optional[set[str]] = None  # Parameters to exclude
    default_parameters: Optional[dict[str, Any]] = None  # Parameters to add
    args_slice: Optional[slice] = None  # For modules that take subset of args


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


# Single unified mapping replacing all previous mapping dictionaries
MODULE_CONFIGS: dict[str, ModuleConfig] = {
    "Linear": ModuleConfig(
        target_module="eqx.nn.Linear",
        default_parameters={"use_bias": True}
    ),
    "Conv1d": ModuleConfig(
        target_module="eqx.nn.Conv1d",
        default_parameters={"use_bias": True}
    ),
    "Conv2d": ModuleConfig(
        target_module="eqx.nn.Conv2d",
        default_parameters={"use_bias": True}
    ),
    "Conv3d": ModuleConfig(
        target_module="eqx.nn.Conv3d",
        default_parameters={"use_bias": True}
    ),
    "BatchNorm1d": ModuleConfig(target_module="eqx.nn.BatchNorm1d"),
    "BatchNorm2d": ModuleConfig(target_module="eqx.nn.BatchNorm2d"),
    "BatchNorm3d": ModuleConfig(target_module="eqx.nn.BatchNorm3d"),
    "Dropout": ModuleConfig(target_module="eqx.nn.Dropout"),
    "LSTM": ModuleConfig(
        target_module="eqx.nn.LSTMCell",
        args_slice=slice(0, 2)  # Only take input_size, hidden_size
    ),
    "RNN": ModuleConfig(
        target_module="ElmanRNNCell",
        custom_definition=_get_elman_rnn_cell_definition(),
        parameter_filters={"batch_first", "dropout", "bidirectional", "num_layers"},
        args_slice=slice(0, 2)
    ),
    "Parameter": ModuleConfig(target_module="jnp.ndarray"),
}

# Cache for auto-transpiled modules to avoid re-transpiling
_AUTO_TRANSPILED_CACHE: dict[str, tuple[str, str]] = {}  # module_name -> (class_def, type_annotation)


def _create_module_reference_ast(module_path: str) -> ast.AST:
    """Create AST reference for module path like 'eqx.nn.Linear' or 'ElmanRNNCell'."""
    if "." not in module_path:
        return ast.Name(id=module_path, ctx=ast.Load())
    
    parts = module_path.split(".")
    result = ast.Name(id=parts[0], ctx=ast.Load())
    for part in parts[1:]:
        result = ast.Attribute(value=result, attr=part, ctx=ast.Load())
    return result


def _create_generic_converter(config: ModuleConfig) -> Callable[[ast.Call, int], tuple[ast.Call, ast.AST]]:
    """Create a converter function from configuration."""
    def converter(node: ast.Call, key_counter: int) -> tuple[ast.Call, ast.AST]:
        # Handle special case for Parameter
        if config.target_module == "jnp.ndarray":
            if node.args:
                transformed_node = node.args[0]  # Extract inner tensor creation
            else:
                transformed_node = node
            type_annotation = ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="ndarray",
                ctx=ast.Load()
            )
            return transformed_node, type_annotation
        
        # Filter keywords
        keywords = list(node.keywords)
        if config.parameter_filters:
            keywords = [kw for kw in keywords if kw.arg not in config.parameter_filters]
        
        # Add default parameters
        if config.default_parameters:
            for param, value in config.default_parameters.items():
                keywords.append(ast.keyword(arg=param, value=ast.Constant(value=value)))
        
        # Add key parameter
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
    
    return converter


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


def create_auto_transpiled_converter(module_name: str) -> Callable[[ast.Call, int], tuple[ast.Call, ast.AST]]:
    """Create a converter function for an auto-transpiled module.
    
    Args:
        module_name: Name of the PyTorch module
        
    Returns:
        Converter function
    """
    def converter(node: ast.Call, key_counter: int) -> tuple[ast.Call, ast.AST]:
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
    
    return converter


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


# Public interface functions - maintain existing signatures for compatibility
def get_module_converter(module_name: str) -> Callable[[ast.Call, int], tuple[ast.Call, ast.AST]]:
    """Get the converter function for a given PyTorch module name.
    
    First tries configured converters, then falls back to auto-transpilation.
    
    Args:
        module_name: Name of the PyTorch module (e.g., 'Linear', 'Conv2d')
        
    Returns:
        Converter function that takes (node, key_counter) and returns (transformed_node, type_annotation)
        
    Raises:
        KeyError: If no converter exists and auto-transpilation fails
    """
    # Try configured converter first
    if module_name in MODULE_CONFIGS:
        config = MODULE_CONFIGS[module_name]
        return _create_generic_converter(config)
    
    # Fallback to auto-transpilation
    try:
        return create_auto_transpiled_converter(module_name)
    except Exception as e:
        raise KeyError(f"No converter available for module {module_name} and auto-transpilation failed: {e}") from e


def is_supported_module(module_name: str) -> bool:
    """Check if a PyTorch module is supported for conversion.
    
    Checks both configured converters and auto-transpilation capability.
    
    Args:
        module_name: Name of the PyTorch module
        
    Returns:
        True if the module can be converted, False otherwise
    """
    # Check configured converters first
    if module_name in MODULE_CONFIGS:
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
        if pytorch_module in MODULE_CONFIGS:
            config = MODULE_CONFIGS[pytorch_module]
            if config.custom_definition:
                required_implementations[config.target_module] = config.custom_definition
    
    return required_implementations


def get_supported_modules() -> list[str]:
    """Get a list of all supported PyTorch module names.
    
    Returns:
        List of supported module names
    """
    return list(MODULE_CONFIGS.keys())
