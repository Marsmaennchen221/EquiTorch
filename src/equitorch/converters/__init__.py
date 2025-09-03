"""Converters package for PyTorch to Equinox/JAX transformations.

This package contains modular converters for different types of PyTorch constructs:
- module_converters: PyTorch nn.Module instances to Equinox equivalents
- functional_converters: PyTorch functional calls (F.relu, torch.flatten) to JAX
- method_converters: PyTorch tensor methods (x.view(), x.flatten()) to JAX/NumPy
"""

from .functional_converters import (
    FUNCTIONAL_MAPPINGS,
    get_functional_mapping,
    get_supported_functionals,
    is_supported_functional,
    transform_functional_call,
    transform_max_pool,
    transform_torch_flatten,
)
from .method_converters import (
    METHOD_MAPPINGS,
    SUPPORTED_METHODS,
    get_method_mapping,
    get_method_name,
    get_supported_methods,
    is_supported_method,
    is_tensor_method_call,
    transform_method_call,
    transform_reshape,
    transform_tensor_flatten,
)
from .module_converters import (
    get_module_converter,
    get_supported_modules,
    is_supported_module,
)

__all__ = [  # noqa: RUF022
    # Module converters
    "get_module_converter",
    "get_supported_modules",
    "is_supported_module",

    # Functional converters
    "FUNCTIONAL_MAPPINGS",
    "get_functional_mapping",
    "get_supported_functionals",
    "is_supported_functional",
    "transform_functional_call",
    "transform_max_pool",
    "transform_torch_flatten",

    # Method converters
    "get_method_mapping",
    "get_method_name",
    "get_supported_methods",
    "is_supported_method",
    "is_tensor_method_call",
    "METHOD_MAPPINGS",
    "SUPPORTED_METHODS",
    "transform_method_call",
    "transform_reshape",
    "transform_tensor_flatten",
]
