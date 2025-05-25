"""
Method converters for PyTorch tensor methods to JAX/NumPy equivalents.

This module contains both the mappings and implementations for converting
PyTorch tensor method calls (x.view(), x.flatten(), etc.) to their JAX/NumPy equivalents.
"""

import ast
from typing import Dict, Set, Optional


# Set of supported tensor methods that can be converted
SUPPORTED_METHODS: Set[str] = {
    'view',
    'reshape', 
    'flatten',
}

# Mapping from PyTorch tensor methods to target functions
METHOD_MAPPINGS: Dict[str, str] = {
    'view': 'jnp.reshape',
    'reshape': 'jnp.reshape',
    'flatten': 'jnp.reshape',
}


def transform_method_call(node: ast.Call, method_name: str) -> ast.Call:
    """
    Transform a PyTorch tensor method call to its JAX/NumPy equivalent.
    
    Args:
        node: The AST Call node to transform
        method_name: The method name (e.g., 'view', 'flatten')
        
    Returns:
        Transformed AST Call node
    """
    if method_name not in SUPPORTED_METHODS:
        return node  # Return unchanged if method not supported
    
    if method_name in ['view', 'reshape']:
        return transform_reshape(node)
    elif method_name == 'flatten':
        return transform_tensor_flatten(node)
    else:
        return node


def transform_reshape(node: ast.Call) -> ast.Call:
    """
    Transform x.view(...) or x.reshape(...) to jnp.reshape(x, ...).
    
    Args:
        node: The AST Call node for tensor.view() or tensor.reshape()
        
    Returns:
        Transformed AST Call node for jnp.reshape()
    """
    input_tensor = node.func.value
    
    # Create shape tuple from arguments
    if node.args:
        shape = ast.Tuple(elts=list(node.args), ctx=ast.Load())
    else:
        shape = ast.Tuple(elts=[], ctx=ast.Load())
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jnp', ctx=ast.Load()),
            attr='reshape',
            ctx=ast.Load()
        ),
        args=[input_tensor, shape],
        keywords=[]
    )


def transform_tensor_flatten(node: ast.Call) -> ast.Call:
    """
    Transform x.flatten() to jnp.reshape(x, (x.shape[0], -1)).
    
    Args:
        node: The AST Call node for tensor.flatten()
        
    Returns:
        Transformed AST Call node for jnp.reshape()
    """
    input_tensor = node.func.value
    
    # Handle flatten with start_dim parameter
    start_dim = 1  # Default start dimension
    if node.args and len(node.args) > 0:
        # If start_dim is provided as argument
        if isinstance(node.args[0], ast.Constant):
            start_dim = node.args[0].value
    
    # Create shape tuple for flattening
    if start_dim == 0:
        # Flatten everything: (-1,)
        shape_elements = [ast.Constant(value=-1)]
    else:
        # Keep first start_dim dimensions, flatten the rest: (x.shape[0], ..., x.shape[start_dim-1], -1)
        shape_elements = []
        for i in range(start_dim):
            shape_elements.append(
                ast.Subscript(
                    value=ast.Attribute(value=input_tensor, attr='shape', ctx=ast.Load()),
                    slice=ast.Constant(value=i),
                    ctx=ast.Load()
                )
            )
        shape_elements.append(ast.Constant(value=-1))
    
    shape_tuple = ast.Tuple(elts=shape_elements, ctx=ast.Load())
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jnp', ctx=ast.Load()),
            attr='reshape',
            ctx=ast.Load()
        ),
        args=[input_tensor, shape_tuple],
        keywords=[]
    )


def is_supported_method(method_name: str) -> bool:
    """
    Check if a PyTorch tensor method is supported for conversion.
    
    Args:
        method_name: Method name (e.g., 'view', 'flatten')
        
    Returns:
        True if the method can be converted, False otherwise
    """
    return method_name in SUPPORTED_METHODS


def get_method_mapping(method_name: str) -> Optional[str]:
    """
    Get the JAX/NumPy mapping for a PyTorch tensor method.
    
    Args:
        method_name: Method name (e.g., 'view', 'flatten')
        
    Returns:
        Target function name (e.g., 'jnp.reshape') or None if not supported
    """
    return METHOD_MAPPINGS.get(method_name)


def get_supported_methods() -> list[str]:
    """
    Get a list of all supported PyTorch tensor methods.
    
    Returns:
        List of supported method names
    """
    return list(SUPPORTED_METHODS)


def is_tensor_method_call(node: ast.Call) -> bool:
    """
    Check if an AST Call node represents a tensor method call.
    
    Args:
        node: AST Call node to check
        
    Returns:
        True if this is a tensor method call (x.method()), False otherwise
    """
    return (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id not in ['self', 'F', 'torch', 'nn'])


def get_method_name(node: ast.Call) -> Optional[str]:
    """
    Extract the method name from a tensor method call.
    
    Args:
        node: AST Call node representing a method call
        
    Returns:
        Method name or None if not a valid method call
    """
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None
