"""Method converters for PyTorch functional methods to JAX/NumPy equivalents.

This module contains both the mappings and implementations for converting
PyTorch functional method calls (F.relu(), torch.flatten(), etc.) to their JAX/NumPy equivalents.
"""

import ast
from collections.abc import Callable

# Simple 1:1 functional mappings that only need function name changes
# Format: (module, function) -> (target_module, target_function)
FUNCTIONAL_MAPPINGS: dict[tuple[str, str], tuple[str, str]] = {
    ("F", "relu"): ("jax.nn", "relu"),
    ("F", "dropout"): ("jax.nn", "dropout"),
    ("F", "softmax"): ("jax.nn", "softmax"),
    ("torch", "relu"): ("jax.nn", "relu"),
    ("torch", "tanh"): ("jnp", "tanh"),
    ("torch", "matmul"): ("jnp", "matmul"),
    ("torch", "sum"): ("jnp", "sum"),
    ("torch", "arange"): ("jnp", "arange"),
    ("torch", "sin"): ("jnp", "sin"),
    ("torch", "cos"): ("jnp", "cos"),
    ("torch", "exp"): ("jnp", "exp"),
    ("torch", "float"): ("jnp", "float32"),
    ("torch", "int"): ("jnp", "int32"),
    ("torch", "bool"): ("jnp", "bool_"),
    ("torch", "float32"): ("jnp", "float32"),
    ("torch", "int32"): ("jnp", "int32"),
    ("torch", "int64"): ("jnp", "int64"),
    ("math", "sqrt"): ("jnp", "sqrt"),
    ("math", "log"): ("jnp", "log"),
}

# Complex transformations requiring custom logic
# Format: (module, function) -> transformer_function
CUSTOM_FUNCTIONAL_TRANSFORMERS: dict[tuple[str, str], Callable] = {
    ("F", "linear"): lambda node: transform_linear(node),
    ("F", "layer_norm"): lambda node: transform_layer_norm(node),
    ("F", "group_norm"): lambda node: transform_group_norm(node),
    ("F", "adaptive_avg_pool2d"): lambda node: transform_adaptive_avg_pool2d(node),
    ("F", "max_pool1d"): lambda node: transform_max_pool(node, "jax.nn", "max_pool", "max_pool1d"),
    ("F", "max_pool2d"): lambda node: transform_max_pool(node, "jax.nn", "max_pool", "max_pool2d"),
    ("F", "max_pool3d"): lambda node: transform_max_pool(node, "jax.nn", "max_pool", "max_pool3d"),
    ("torch", "flatten"): lambda node: transform_torch_flatten(node),
    ("torch", "zeros"): lambda node: transform_torch_zeros(node),
    ("torch", "randn"): lambda node: transform_torch_randn(node),
    ("torch", "randint"): lambda node: transform_torch_randint(node),
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

    # Check for custom transformers first
    if mapping_key in CUSTOM_FUNCTIONAL_TRANSFORMERS:
        transformer = CUSTOM_FUNCTIONAL_TRANSFORMERS[mapping_key]
        return transformer(node)

    # Check for simple mappings
    if mapping_key in FUNCTIONAL_MAPPINGS:
        new_module, new_func = FUNCTIONAL_MAPPINGS[mapping_key]
        return transform_simple_functional(node, new_module, new_func)

    # No mapping found - return unchanged
    return node


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


def transform_max_pool(
    node: ast.Call, new_module: str, new_func: str, func_name: str
) -> ast.Call | ast.expr:  # noqa: ARG001
    """Transform max_pool with proper parameter mapping using jax.scipy.signal.pool.

    Args:
        node: The AST Call node
        new_module: Target module (unused)
        new_func: Target function (unused)
        func_name: Original function name for dimension info

    Returns:
        Transformed AST Call node using jax.scipy.signal.pool
    """
    invalid_args_len = 2
    if len(node.args) < invalid_args_len:
        return node  # Invalid call

    try:
        # Extract arguments
        input_tensor = node.args[0]
        kernel_size = node.args[1]

        # Convert AST nodes to code strings
        input_str = ast.unparse(input_tensor)
        kernel_str = ast.unparse(kernel_size)

        # Extract stride if provided, otherwise default to kernel_size
        if len(node.args) > invalid_args_len:
            stride_str = ast.unparse(node.args[2])
        else:
            stride_str = kernel_str

    except RecursionError:
        # Fallback to original node if anything goes wrong
        return node

    else:
        # Build target code string using jax.lax.reduce_window
        # For Equinox single-example processing with 3D tensors (channels, height, width),
        # use 3D window dimensions: (1, kernel_size, kernel_size)
        target_code = f"""jax.lax.reduce_window(
            {input_str},
            -jnp.inf,
            jax.lax.max,
            (1, {kernel_str}, {kernel_str}),
            (1, {stride_str}, {stride_str}),
            'VALID'
        )"""

        # Parse back to AST and return the expression
        parsed = ast.parse(target_code, mode="eval")
        return parsed.body


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


def transform_torch_zeros(node: ast.Call) -> ast.Call:
    """Transform torch.zeros to jnp.zeros with proper shape tuple.

    Args:
        node: The AST Call node for torch.zeros

    Returns:
        Transformed AST Call node for jnp.zeros
    """
    # Convert arguments to a shape tuple
    # torch.zeros(1, 2, 3) -> jnp.zeros((1, 2, 3))
    if node.args:
        shape_tuple = ast.Tuple(elts=list(node.args), ctx=ast.Load())
    else:
        shape_tuple = ast.Tuple(elts=[], ctx=ast.Load())

    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()),
            attr="zeros",
            ctx=ast.Load(),
        ),
        args=[shape_tuple],
        keywords=node.keywords,
    )


def transform_torch_randn(node: ast.Call) -> ast.Call:
    """Transform torch.randn to jax.random.normal with proper argument order.

    PyTorch: torch.randn(shape_args...)
    JAX: jax.random.normal(key, (shape_args...))

    Args:
        node: The AST Call node for torch.randn

    Returns:
        Transformed AST Call node for jax.random.normal
    """
    # Create the shape tuple from the original arguments
    if len(node.args) == 1:
        # Single argument - use as-is
        shape_arg = node.args[0]
    else:
        # Multiple arguments - wrap in tuple
        shape_arg = ast.Tuple(elts=list(node.args), ctx=ast.Load())

    # Create jax.random.normal(subkey, shape)
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="jax", ctx=ast.Load()),
                attr="random",
                ctx=ast.Load(),
            ),
            attr="normal",
            ctx=ast.Load(),
        ),
        args=[
            ast.Name(id="subkey", ctx=ast.Load()),  # key first
            shape_arg,  # shape second
        ],
        keywords=[],
    )


def transform_torch_randint(node: ast.Call) -> ast.Call:
    """Transform torch.randint to jax.random.randint with proper argument order.

    PyTorch: torch.randint(low, high, size)
    JAX: jax.random.randint(key, shape, minval, maxval)

    Args:
        node: The AST Call node for torch.randint

    Returns:
        Transformed AST Call node for jax.random.randint
    """
    invalid_args_len = 3
    if len(node.args) < invalid_args_len:
        return node  # Invalid call

    low = node.args[0]
    high = node.args[1]
    shape = node.args[2]

    # Create jax.random.randint(subkey, shape, minval, maxval)
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="jax", ctx=ast.Load()),
                attr="random",
                ctx=ast.Load(),
            ),
            attr="randint",
            ctx=ast.Load(),
        ),
        args=[
            ast.Name(id="subkey", ctx=ast.Load()),  # key first
            shape,  # shape second
            low,  # minval third
            high,  # maxval fourth
        ],
        keywords=[],
    )


def transform_linear(node: ast.Call) -> ast.Call | ast.expr:
    """Transform F.linear(input, weight, bias) to JAX equivalent.

    PyTorch: F.linear(input, weight, bias)
    JAX: jnp.dot(input, weight.T) + bias (if bias is not None)

    Args:
        node: The AST Call node for F.linear

    Returns:
        Transformed AST Call node
    """
    invalid_args_len = 2
    if len(node.args) < invalid_args_len:
        return node  # Invalid call, return unchanged

    try:
        # Extract arguments
        input_tensor = node.args[0]
        weight = node.args[1]
        bias = node.args[2] if len(node.args) > invalid_args_len else None

        # Convert AST nodes to code strings
        input_str = ast.unparse(input_tensor)
        weight_str = ast.unparse(weight)

        # Build target code string
        if bias is not None:
            bias_str = ast.unparse(bias)
            target_code = f"jnp.dot({input_str}, {weight_str}.T) + {bias_str}"
        else:
            target_code = f"jnp.dot({input_str}, {weight_str}.T)"

    except RecursionError:
        # Fallback to original node if anything goes wrong
        return node

    else:
        # Parse back to AST and return the expression
        parsed = ast.parse(target_code, mode="eval")
        return parsed.body


def transform_layer_norm(node: ast.Call) -> ast.Call:
    """Transform F.layer_norm to JAX equivalent.

    TODO: Implement layer normalization transformation.
    """
    return node


def transform_group_norm(node: ast.Call) -> ast.Call:
    """Transform F.group_norm to JAX equivalent.

    TODO: Implement group normalization transformation.
    """
    return node


def transform_adaptive_avg_pool2d(node: ast.Call) -> ast.Call:
    """Transform F.adaptive_avg_pool2d to JAX equivalent.

    TODO: Implement adaptive average pooling transformation.
    """
    return node


def is_supported_functional(module_name: str, func_name: str) -> bool:
    """Check if a PyTorch functional call is supported for conversion.

    Args:
        module_name: Module name (e.g., 'F', 'torch')
        func_name: Function name (e.g., 'relu', 'flatten')

    Returns:
        True if the functional call can be converted, False otherwise
    """
    mapping_key = (module_name, func_name)
    return mapping_key in FUNCTIONAL_MAPPINGS or mapping_key in CUSTOM_FUNCTIONAL_TRANSFORMERS


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
    return list(FUNCTIONAL_MAPPINGS.keys()) + list(CUSTOM_FUNCTIONAL_TRANSFORMERS.keys())
