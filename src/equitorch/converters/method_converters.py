"""Method converters for PyTorch tensor methods to JAX/NumPy equivalents.

This module contains both the mappings and implementations for converting
PyTorch tensor method calls (x.view(), x.flatten(), etc.) to their JAX/NumPy equivalents.
"""

import ast

# Set of supported tensor methods that can be converted
SUPPORTED_METHODS: set[str] = {
    "view",
    "reshape", 
    "flatten",
    "size",
    "unsqueeze",
    "transpose",
    "float",
    "int",
    "bool",
    "long",
    "double",
}

# Mapping from PyTorch tensor methods to target functions
METHOD_MAPPINGS: dict[str, str] = {
    "view": "jnp.reshape",
    "reshape": "jnp.reshape",
    "flatten": "jnp.reshape",
}


def transform_method_call(node: ast.Call, method_name: str) -> ast.Call | ast.Attribute | ast.Subscript:
    """Transform a PyTorch tensor method call to its JAX/NumPy equivalent.
    
    Args:
        node: The AST Call node to transform
        method_name: The method name (e.g., 'view', 'flatten')
        
    Returns:
        Transformed AST Call node
    """
    if method_name not in SUPPORTED_METHODS:
        return node  # Return unchanged if method not supported

    result: ast.Call | ast.Attribute | ast.Subscript
    match method_name:
        case "view" | "reshape":
            result = transform_reshape(node)
        case "flatten":
            result = transform_tensor_flatten(node)
        case "size":
            result = transform_size(node)
        case "unsqueeze":
            result = transform_unsqueeze(node)
        case "transpose":
            result = transform_transpose(node)
        case "float":
            result = transform_dtype_cast(node, "jnp.float32")
        case "int":
            result = transform_dtype_cast(node, "jnp.int32")
        case "bool":
            result = transform_dtype_cast(node, "jnp.bool_")
        case "long":
            result = transform_dtype_cast(node, "jnp.int64")
        case "double":
            result = transform_dtype_cast(node, "jnp.float64")
        case _:
            result = node
    return result


def transform_reshape(node: ast.Call) -> ast.Call:
    """Transform x.view(...) or x.reshape(...) to appropriate JAX operation.
    
    For Equinox single-example processing, use jnp.ravel for flattening operations.
    
    Args:
        node: The AST Call node for tensor.view() or tensor.reshape()
        
    Returns:
        Transformed AST Call node
    """
    input_tensor = node.func.value  # type: ignore[attr-defined]
    
    # Check if this is a flattening operation like x.view(x.shape[0], -1) or x.view(batch_size, -1)
    # Note: -1 is represented as ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))
    is_negative_one = (isinstance(node.args[1], ast.UnaryOp) and
                       isinstance(node.args[1].op, ast.USub) and
                       isinstance(node.args[1].operand, ast.Constant) and
                       node.args[1].operand.value == 1)

    node_args_len = 2
    if (len(node.args) == node_args_len and is_negative_one):
        # This is a flattening operation - use jnp.ravel for single example processing
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="ravel",
                ctx=ast.Load(),
            ),
            args=[input_tensor],
            keywords=[],
        )
    # For other reshape operations, use jnp.reshape
    if node.args:
        shape = ast.Tuple(elts=list(node.args), ctx=ast.Load())
    else:
        shape = ast.Tuple(elts=[], ctx=ast.Load())
    
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()),
            attr="reshape",
            ctx=ast.Load(),
        ),
        args=[input_tensor, shape],
        keywords=[],
    )


def transform_tensor_flatten(node: ast.Call) -> ast.Call:
    """Transform x.flatten() to jnp.ravel(x) for single example processing.
    
    Args:
        node: The AST Call node for tensor.flatten()
        
    Returns:
        Transformed AST Call node for jnp.ravel()
    """
    input_tensor = node.func.value  # type: ignore[attr-defined]
    
    # For Equinox models that process single examples, use jnp.ravel
    # This flattens the tensor completely, which is appropriate for single examples
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()),
            attr="ravel",
            ctx=ast.Load(),
        ),
        args=[input_tensor],
        keywords=[],
    )


def transform_size(node: ast.Call) -> ast.Subscript | ast.Attribute:
    """Transform x.size(dim) to x.shape[dim].
    
    Args:
        node: The AST Call node for tensor.size()
        
    Returns:
        Transformed AST Subscript node for tensor.shape[dim]
    """
    input_tensor = node.func.value  # type: ignore[attr-defined]
    
    if node.args and len(node.args) > 0:
        # x.size(dim) -> x.shape[dim]
        dim = node.args[0]
        return ast.Subscript(
            value=ast.Attribute(value=input_tensor, attr="shape", ctx=ast.Load()),
            slice=dim,
            ctx=ast.Load(),
        )
    # x.size() -> x.shape (return the whole shape tuple)
    return ast.Attribute(value=input_tensor, attr="shape", ctx=ast.Load())  # type: ignore[return-value]


def transform_unsqueeze(node: ast.Call) -> ast.Call:
    """Transform x.unsqueeze(dim) to jnp.expand_dims(x, axis=dim).
    
    Args:
        node: The AST Call node for tensor.unsqueeze()
        
    Returns:
        Transformed AST Call node for jnp.expand_dims()
    """
    input_tensor = node.func.value  # type: ignore[attr-defined]
    
    if node.args and len(node.args) > 0:
        dim = node.args[0]
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="expand_dims",
                ctx=ast.Load(),
            ),
            args=[input_tensor],
            keywords=[ast.keyword(arg="axis", value=dim)],
        )
    # If no dimension specified, default to last dimension
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()),
            attr="expand_dims",
            ctx=ast.Load(),
        ),
        args=[input_tensor],
        keywords=[ast.keyword(arg="axis", value=ast.Constant(value=-1))],
    )


def transform_transpose(node: ast.Call) -> ast.Call:
    """Transform x.transpose(dim0, dim1) to jnp.transpose(x, axes=(dim0, dim1, ...)).
    
    Args:
        node: The AST Call node for tensor.transpose()
        
    Returns:
        Transformed AST Call node for jnp.transpose()
    """
    input_tensor = node.func.value  # type: ignore[attr-defined]
    node_args_len = 1

    if node.args and len(node.args) > node_args_len:
        # x.transpose(dim0, dim1) -> jnp.transpose(x, axes=(dim0, dim1))
        axes_tuple = ast.Tuple(elts=list(node.args), ctx=ast.Load())
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="transpose",
                ctx=ast.Load(),
            ),
            args=[input_tensor],
            keywords=[ast.keyword(arg="axes", value=axes_tuple)],
        )
    if node.args and len(node.args) == node_args_len:
        # x.transpose(dim) -> jnp.swapaxes(x, dim, -1) for single dimension
        dim = node.args[0]
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="swapaxes",
                ctx=ast.Load(),
            ),
            args=[input_tensor, dim, ast.Constant(value=-1)],
            keywords=[],
        )
    # x.transpose() -> jnp.transpose(x) (transpose all dimensions)
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()),
            attr="transpose",
            ctx=ast.Load(),
        ),
        args=[input_tensor],
        keywords=[],
    )


def transform_dtype_cast(node: ast.Call, target_dtype: str) -> ast.Call:
    """Transform x.float(), x.int(), etc. to x.astype(jnp.dtype).
    
    Args:
        node: The AST Call node for tensor.float(), tensor.int(), etc.
        target_dtype: Target JAX dtype (e.g., 'jnp.float32')
        
    Returns:
        Transformed AST Call node for tensor.astype(dtype)
    """
    input_tensor = node.func.value  # type: ignore[attr-defined]
    
    # Create the dtype reference (e.g., jnp.float32)
    dtype_parts = target_dtype.split(".")
    dtype_ref: ast.Attribute | ast.Name
    dtype_parts_len = 2
    if len(dtype_parts) == dtype_parts_len:
        dtype_ref = ast.Attribute(
            value=ast.Name(id=dtype_parts[0], ctx=ast.Load()),
            attr=dtype_parts[1],
            ctx=ast.Load(),
        )
    else:
        dtype_ref = ast.Name(id=target_dtype, ctx=ast.Load())
    
    return ast.Call(
        func=ast.Attribute(
            value=input_tensor,
            attr="astype",
            ctx=ast.Load(),
        ),
        args=[dtype_ref],
        keywords=[],
    )


def is_supported_method(method_name: str) -> bool:
    """Check if a PyTorch tensor method is supported for conversion.
    
    Args:
        method_name: Method name (e.g., 'view', 'flatten')
        
    Returns:
        True if the method can be converted, False otherwise
    """
    return method_name in SUPPORTED_METHODS


def get_method_mapping(method_name: str) -> str | None:
    """Get the JAX/NumPy mapping for a PyTorch tensor method.
    
    Args:
        method_name: Method name (e.g., 'view', 'flatten')
        
    Returns:
        Target function name (e.g., 'jnp.reshape') or None if not supported
    """
    return METHOD_MAPPINGS.get(method_name)


def get_supported_methods() -> list[str]:
    """Get a list of all supported PyTorch tensor methods.
    
    Returns:
        List of supported method names
    """
    return list(SUPPORTED_METHODS)


def is_tensor_method_call(node: ast.Call) -> bool:
    """Check if an AST Call node represents a tensor method call.
    
    Args:
        node: AST Call node to check
        
    Returns:
        True if this is a tensor method call (x.method()), False otherwise
    """
    if not isinstance(node.func, ast.Attribute):
        return False
    
    # Check if this is a method call on a tensor-like expression
    # This includes:
    # - Simple variables: x.method()
    # - Subscripts: tensor[:5].method()
    # - Attribute access: self.tensor.method()
    # - Function calls: func().method()
    
    # Exclude known non-tensor modules/objects
    if isinstance(node.func.value, ast.Name):
        return node.func.value.id not in ["self", "F", "torch", "nn", "jax", "jnp", "jr", "eqx", "eqt"]
    
    # Allow method calls on more complex expressions (subscripts, attributes, calls)
    return isinstance(node.func.value, ast.Subscript | ast.Attribute | ast.Call)


def get_method_name(node: ast.Call) -> str | None:
    """Extract the method name from a tensor method call.
    
    Args:
        node: AST Call node representing a method call
        
    Returns:
        Method name or None if not a valid method call
    """
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None
