"""Method converters for PyTorch functional methods to JAX/NumPy equivalents.

This module contains both the mappings and implementations for converting
PyTorch functional method calls (F.relu(), torch.flatten(), etc.) to their JAX/NumPy equivalents.
"""

import ast

# Mapping from PyTorch functional calls to JAX equivalents
# Format: (module, function) -> (target_module, target_function)
FUNCTIONAL_MAPPINGS: dict[tuple[str, str], tuple[str, str]] = {
    ("F", "relu"): ("jax.nn", "relu"),
    ("F", "dropout"): ("jax.nn", "dropout"),
    ("F", "max_pool1d"): ("jax.nn", "max_pool"),
    ("F", "max_pool2d"): ("jax.nn", "max_pool"),
    ("F", "max_pool3d"): ("jax.nn", "max_pool"),
    ("F", "softmax"): ("jax.nn", "softmax"),
    ("torch", "flatten"): ("jnp", "reshape"),
    ("torch", "zeros"): ("jnp", "zeros"),
    ("torch", "relu"): ("jax.nn", "relu"),
    ("torch", "max_pool2d"): ("jax.nn", "max_pool"),
    ("torch", "tanh"): ("jnp", "tanh"),
    ("torch", "matmul"): ("jnp", "matmul"),
    ("torch", "sum"): ("jnp", "sum"),
    ("torch", "randn"): ("jax.random", "normal"),
}


def transform_functional_call(node: ast.Call, module_name: str, func_name: str) -> ast.Call:
    """Transform a PyTorch functional call to its JAX equivalent.
    
    Args:
        node: The AST Call node to transform
        module_name: The module name (e.g., 'F', 'torch')
        func_name: The function name (e.g., 'relu', 'flatten')
        
    Returns:
        Transformed AST Call node
    """
    mapping_key = (module_name, func_name)
    
    if mapping_key not in FUNCTIONAL_MAPPINGS:
        return node  # Return unchanged if no mapping exists
    
    new_module, new_func = FUNCTIONAL_MAPPINGS[mapping_key]
    
    # Handle special cases that need custom transformation logic
    if func_name.startswith("max_pool"):
        return transform_max_pool(node, new_module, new_func, func_name)
    if func_name == "flatten" and module_name == "torch":
        return transform_torch_flatten(node)
    # Simple mapping - just change the function reference
    return transform_simple_functional(node, new_module, new_func)


def transform_simple_functional(node: ast.Call, new_module: str, new_func: str) -> ast.Call:
    """Transform a simple functional call with direct mapping.
    
    Args:
        node: The AST Call node
        new_module: Target module (e.g., 'jax.nn')
        new_func: Target function (e.g., 'relu')
        
    Returns:
        Transformed AST Call node
    """
    # Create the new function reference
    if "." in new_module:  # Handle jax.nn
        module_parts = new_module.split(".")
        new_func_ref = ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id=module_parts[0], ctx=ast.Load()),
                attr=module_parts[1],
                ctx=ast.Load(),
            ),
            attr=new_func,
            ctx=ast.Load(),
        )
    else:  # Handle simple module names like 'jnp'
        new_func_ref = ast.Attribute(
            value=ast.Name(id=new_module, ctx=ast.Load()),
            attr=new_func,
            ctx=ast.Load(),
        )
    
    return ast.Call(
        func=new_func_ref,
        args=node.args,
        keywords=node.keywords,
    )


def transform_max_pool(node: ast.Call, new_module: str, new_func: str, func_name: str) -> ast.Call:  # noqa: ARG001 TODO: remove temporary noqa
    """Transform max_pool with proper parameter mapping.
    
    Args:
        node: The AST Call node
        new_module: Target module ('jax.nn')
        new_func: Target function ('max_pool')
        func_name: Original function name for dimension info
        
    Returns:
        Transformed AST Call node
    """
    # Create jax.nn.max_pool function reference
    new_func_ref = ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="jax", ctx=ast.Load()),
            attr="nn",
            ctx=ast.Load(),
        ),
        attr="max_pool",
        ctx=ast.Load(),
    )
    
    # For now, use simplified parameter mapping
    # Full implementation would handle window_shape, strides, padding conversion
    return ast.Call(
        func=new_func_ref,
        args=node.args,
        keywords=node.keywords,
    )


def transform_torch_flatten(node: ast.Call) -> ast.Call:
    """Transform torch.flatten to jnp.reshape.
    
    Args:
        node: The AST Call node for torch.flatten
        
    Returns:
        Transformed AST Call node for jnp.reshape
    """
    if len(node.args) < 1:
        return node
    
    input_tensor = node.args[0]
    
    # Create shape tuple for flattening: (batch_size, -1)
    shape_elements = [
        ast.Subscript(
            value=ast.Attribute(value=input_tensor, attr="shape", ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        ),
        ast.Constant(value=-1),
    ]
    
    shape_tuple = ast.Tuple(elts=shape_elements, ctx=ast.Load())
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()),
            attr="reshape",
            ctx=ast.Load(),
        ),
        args=[input_tensor, shape_tuple],
        keywords=[],
    )


def is_supported_functional(module_name: str, func_name: str) -> bool:
    """Check if a PyTorch functional call is supported for conversion.
    
    Args:
        module_name: Module name (e.g., 'F', 'torch')
        func_name: Function name (e.g., 'relu', 'flatten')
        
    Returns:
        True if the functional call can be converted, False otherwise
    """
    return (module_name, func_name) in FUNCTIONAL_MAPPINGS


def get_functional_mapping(module_name: str, func_name: str) -> tuple[str, str] | None:
    """Get the JAX mapping for a PyTorch functional call.
    
    Args:
        module_name: Module name (e.g., 'F', 'torch')
        func_name: Function name (e.g., 'relu', 'flatten')
        
    Returns:
        Tuple of (target_module, target_function) or None if not supported
    """
    return FUNCTIONAL_MAPPINGS.get((module_name, func_name))


def get_supported_functionals() -> list[tuple[str, str]]:
    """Get a list of all supported PyTorch functional calls.
    
    Returns:
        List of (module_name, func_name) tuples
    """
    return list(FUNCTIONAL_MAPPINGS.keys())
