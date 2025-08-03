"""
EquiTorch Custom Module Library

This library provides custom Equinox module implementations that maintain
compatibility with PyTorch APIs while leveraging JAX/Equinox functionality.
"""

from .attention import EquiTorchMultiHeadAttention

class nn:
    """Neural network modules namespace for EquiTorch library."""
    EquiTorchMultiHeadAttention = EquiTorchMultiHeadAttention


__all__ = [
    "EquiTorchMultiHeadAttention",
    "nn"
]