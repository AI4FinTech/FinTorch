import torch
import torch.nn as nn
from typing import Optional


class LayerNorm(nn.LayerNorm):
    """
    Explainable LayerNorm layer with relevance propagation capabilities.

    This class extends PyTorch's LayerNorm to support Layer-wise Relevance Propagation (LRP)
    for explainable AI applications.

    Args:
        normalized_shape: Input shape from an expected input of size
            [..., normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]
        eps: A value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: A boolean value that when set to True, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: True

    Methods:
        forward(input: torch.Tensor) -> torch.Tensor:
            Forward pass of LayerNorm.

        propagate(relevance: torch.Tensor) -> torch.Tensor:
            Propagates relevance backwards for explainable AI.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__(normalized_shape, eps, elementwise_affine, device=device, dtype=dtype)

        # Store input and output for relevance propagation
        self._input_tensor: Optional[torch.Tensor] = None
        self._output_tensor: Optional[torch.Tensor] = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LayerNorm with input/output caching for relevance propagation.

        Args:
            input: Input tensor

        Returns:
            torch.Tensor: Normalized output tensor
        """
        # Store input for relevance propagation
        self._input_tensor = input.detach().clone()

        # Perform standard layer normalization
        output = super().forward(input)

        # Store output for relevance propagation
        self._output_tensor = output.detach().clone()

        return output

    def propagate(self, relevance: torch.Tensor) -> torch.Tensor:
        """
        Propagates relevance backwards through the LayerNorm layer.

        For LayerNorm, we use the identity propagation rule as the normalization
        operation redistributes but doesn't fundamentally change the relative
        importance of features.

        Args:
            relevance: Relevance tensor to propagate backwards

        Returns:
            torch.Tensor: Propagated relevance tensor
        """
        if self._input_tensor is None or self._output_tensor is None:
            # If no cached input/output, return relevance as-is
            return relevance

        # For LayerNorm, we use a simple identity-like propagation
        # The normalization redistributes values but the relative importance
        # structure is preserved

        # Method 1: Simple identity propagation
        # return relevance

        # Method 2: Scale relevance by the ratio of input to output magnitude
        # This accounts for the normalization effect
        input_magnitude = torch.abs(self._input_tensor) + 1e-12
        output_magnitude = torch.abs(self._output_tensor) + 1e-12

        # Scale factor to reverse the normalization effect
        scale_factor = input_magnitude / output_magnitude

        # Apply the scale factor to relevance
        propagated_relevance = relevance * scale_factor

        return propagated_relevance

    def extra_repr(self) -> str:
        """String representation of the layer."""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
