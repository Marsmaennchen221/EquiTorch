"""Module converters for PyTorch to Equinox transformations.

This module contains both the mappings and implementations for converting
PyTorch nn.Module instances to their Equinox equivalents.
"""

import ast
from collections.abc import Callable

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
}


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
    
    Args:
        module_name: Name of the PyTorch module (e.g., 'Linear', 'Conv2d')
        
    Returns:
        Converter function that takes (node, key_counter) and returns transformed node
        
    Raises:
        KeyError: If no converter exists for the given module name
    """
    if module_name not in MODULE_MAPPINGS:
        raise KeyError(f"No converter available for module: {module_name}")
    
    converter_name = MODULE_MAPPINGS[module_name]
    return globals()[converter_name]


def is_supported_module(module_name: str) -> bool:
    """Check if a PyTorch module is supported for conversion.
    
    Args:
        module_name: Name of the PyTorch module
        
    Returns:
        True if the module can be converted, False otherwise
    """
    return module_name in MODULE_MAPPINGS


def get_supported_modules() -> list[str]:
    """Get a list of all supported PyTorch module names.
    
    Returns:
        List of supported module names
    """
    return list(MODULE_MAPPINGS.keys())
