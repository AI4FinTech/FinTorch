import torch
import torch.nn as nn
from fintorch.layers.explainable.relevancepropagation import safe_divide_where
from fintorch.layers.explainable.ValueCaptureHook import ValueCaptureHook

class einsum(nn.Module):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
        self.hook = ValueCaptureHook()
        self.register_forward_hook(self.hook.hook_fn)
        self._original_inputs = None

    def forward(self, *operands):
        # Store original inputs for gradient computation
        self._original_inputs = operands
        return torch.einsum(self.equation, *operands)

    def propagate(self, relevance):
        # Relevance formula: R_input = X * (∂Z/∂X) * (R_output / Z)

        # Use the original inputs for gradient computation
        X = self._original_inputs
        Z = self.hook.Y

        # Stabilize division to prevent numerical issues
        # scaling factor (R_output / Z)
        S = safe_divide_where(relevance, Z)

        # Calculate the gradient (∂Z/∂X) and multiply with the scaling factor
        # C = (∂Z/∂X) * (R_output / Z)
        C = torch.autograd.grad(Z, X, S, retain_graph=True, allow_unused=True)

        # Generalize the final multiplication for any number of operands
        if len(X) > 1:
            # Handle multiple inputs with a list comprehension
            # See equation (13) of Chefer et al. (2020) with the Hadamard product (element-wise multiplication)
            outputs = [x * c if c is not None else torch.zeros_like(x) for x, c in zip(X, C)]
        else:
            # Handle a single input
            if C[0] is not None:
                outputs = X[0] * C[0]
            else:
                outputs = torch.zeros_like(X[0])

        return outputs
