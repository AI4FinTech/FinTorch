import torch
import torch.nn as nn


class CausalConvolution(nn.Module):
    """
    Causal Convolution module for time series data.

    This module applies a causal convolution operation to time series data,
    allowing for the modeling of temporal dependencies in a causal manner.

    Attributes:
        number_of_series (int): The number of time series in the input data.
        length_input_window (int): The length of the input time window.
        number_of_heads (int): The number of attention heads.
        kernel (nn.Parameter): The learnable kernel for the causal convolution.
        base (torch.Tensor): A base tensor used for normalization.

    Methods:
        shift_kernel(kernel: torch.Tensor, shifts: int) -> torch.Tensor:
            Shifts the kernel along the time dimension.
        stack_shifted_kernel(kernel: torch.Tensor) -> torch.Tensor:
            Stacks the shifted kernels to create a lower triangular kernel.
        apply_kernel(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
            Applies the kernel to the input data.
        transform_x(x: torch.Tensor) -> torch.Tensor:
            Transforms the input data to remove self-information.
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the causal convolution module.
    """

    def __init__(
        self, number_of_series: int, length_input_window: int, number_of_heads: int
    ) -> None:
        super().__init__()

        self.number_of_series = number_of_series
        self.input_window = length_input_window
        self.number_of_heads = number_of_heads

        self.kernel = nn.Parameter(
            torch.ones(
                (
                    number_of_heads,
                    number_of_series,
                    number_of_series,
                    length_input_window,
                ),
                dtype=torch.float,
            )
        )
        self.register_parameter("kernel", self.kernel)

        # 6D tensor because the output of apply_kernel is a 6D tensor
        self.base = torch.tensor([i for i in range(1, self.input_window + 1)]).reshape(
            1, 1, 1, 1, -1, 1
        )

    def shift_kernel(self, kernel: torch.Tensor, shifts: int) -> torch.Tensor:
        # kernel: (number_of_heads, number_of_series, number_of_series, length_input_window)
        return torch.roll(kernel, shifts=shifts + 1, dims=3)

    def stack_shifted_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        # kernel: (number_of_heads, number_of_series, number_of_series, length_input_window)
        shifted_kernels = []
        for i in range(self.input_window):
            shifted_kernels.append(self.shift_kernel(kernel, i))
        kernel = torch.stack(shifted_kernels, dim=-2)

        # kernel: (number_of_heads, number_of_series, number_of_series, length_input_window, length_input_window)
        # Make the kernel lower triangular
        kernel = torch.tril(kernel, diagonal=0)
        return kernel

    def apply_kernel(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        number_of_heads, number_of_series, _, length_input_window, _ = kernel.shape
        batch_size, _, _, hidden_dimensionality = x.shape

        # Compute output using `einsum`
        # for verbose implementation (educational), see tests/models/causalformer/test_causalconv.py
        einsum_result = torch.einsum("hxyji,bxif->bhxyjf", kernel, x)

        # einsum_result:
        # (batch_size, number_of_heads, number_of_series, number_of_series, length_input_window, hidden_dimensionality)
        return einsum_result

    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.number_of_series):
            print(f"do something to number_of_series:{i} x[:, :, {i}, {i}, :, :]")
            # Select the the same series in from the (3) and (4) th dimension
            # (batch_size, number_of_heads, i, i, length_input_window, hidden_dimensionality)
            # dim 0 = batch_size
            # dim 1 = number_of_heads
            # dim 2 = number_of_series (roll dimension) -> shift right by 1
            # dim 3 = number_of_series
            # dim 4 = length_input_window
            # dim 5 = hidden_dimensionality
            # used for self-causation (samples in same sample time-window)
            x[:, :, i, i, :, :] = x[:, :, i, i, :, :].roll(1, dims=2)
            # exclude ground-truth value
            # (batch size, heads, hidden) => 0 for all self-relation (i,i) at time T values which after the shift/roll
            # operation is the first index (0) because it should not contain ground-truth X^T_{i,i}
            x[:, :, i, i, 0, :] *= torch.zeros_like(x[:, :, i, i, 0, :])

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, number_of_series, length_input_window, hidden_dimensionality)

        # Get stack shifted kernel
        kernel = self.stack_shifted_kernel(self.kernel)
        kernel.requires_grad_()  # makes this a learnable kernel

        # kernel: (number_of_heads, number_of_series, number_of_series, length_input_window, length_input_window)

        # x after:
        # (batch_size, number_of_heads, number_of_series, number_of_series, length_input_window, hidden_dimensionality)
        x = self.apply_kernel(x, kernel)
        x = x / self.base  # divide by 6D tensor

        # Remove self-information (Ground-truth)
        x = self.transform_x(x)

        return x
