"""
Converters package for PyTorch to Equinox/JAX transformations.

This package contains modular converters for different types of PyTorch constructs:
- module_converters: PyTorch nn.Module instances to Equinox equivalents
- functional_converters: PyTorch functional calls (F.relu, torch.flatten) to JAX
- method_converters: PyTorch tensor methods (x.view(), x.flatten()) to JAX/NumPy
"""

from .module_converters import (
    MODULE_MAPPINGS,
    get_module_converter,
    is_supported_module,
    get_supported_modules,
    convert_linear,
    convert_conv,
    convert_batchnorm,
    convert_dropout,
)

from .functional_converters import (
    FUNCTIONAL_MAPPINGS,
    transform_functional_call,
    is_supported_functional,
    get_functional_mapping,
    get_supported_functionals,
    transform_max_pool,
    transform_torch_flatten,
)

from .method_converters import (
    SUPPORTED_METHODS,
    METHOD_MAPPINGS,
    transform_method_call,
    is_supported_method,
    get_method_mapping,
    get_supported_methods,
    is_tensor_method_call,
    get_method_name,
    transform_reshape,
    transform_tensor_flatten,
)

__all__ = [  # noqa: RUF022
    # Module converters
    'convert_batchnorm',
    'convert_conv',
    'convert_dropout',
    'convert_linear',
    'get_module_converter',
    'get_supported_modules',
    'is_supported_module',
    'MODULE_MAPPINGS',

    # Functional converters
    'FUNCTIONAL_MAPPINGS',
    'get_functional_mapping',
    'get_supported_functionals',
    'is_supported_functional',
    'transform_functional_call',
    'transform_max_pool',
    'transform_torch_flatten',

    # Method converters
    'get_method_mapping',
    'get_method_name',
    'get_supported_methods',
    'is_supported_method',
    'is_tensor_method_call',
    'METHOD_MAPPINGS',
    'SUPPORTED_METHODS',
    'transform_method_call',
    'transform_reshape',
    'transform_tensor_flatten',
]
