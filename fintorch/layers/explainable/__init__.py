"""
Explainable AI layers for FinTorch.

This module provides neural network layers with built-in explainability
through relevance propagation techniques.
"""

from .relevancepropagation import (
    LinearRelevancePropagation,
    safe_divide_where
)
from .ValueCaptureHook import ValueCaptureHook
from .linear import Linear
from .einsum import einsum
from .Dropout import Dropout
from .LeakyReLU import LeakyReLU
from .ReLU import ReLU
from .Layernorm import LayerNorm
from .Softmax import Softmax

__all__ = [
    # Base classes
    'LinearRelevancePropagation',

    # Layer implementations
    'Linear',
    'einsum',
    'Dropout',
    'LeakyReLU',
    'ReLU',
    'LayerNorm',
    'Softmax',

    # Utilities
    'ValueCaptureHook',
    'safe_divide_where',
]

# Version info
__version__ = '0.1.0'
