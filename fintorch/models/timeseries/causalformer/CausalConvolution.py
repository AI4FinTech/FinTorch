import torch
import torch.nn as nn


class CausalConvolution(nn.Module):
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

        result = torch.zeros(
            (
                batch_size,
                number_of_heads,
                number_of_series,
                number_of_series,
                length_input_window,
                hidden_dimensionality,
            ),
            dtype=x.dtype,
            device=x.device,
        )

        # Einsum notation hxyji,bxif->bhxyjf
        for b in range(batch_size):
            for h in range(number_of_heads):
                for xx in range(number_of_series):
                    for y in range(number_of_series):
                        for j in range(length_input_window):
                            for f in range(hidden_dimensionality):
                                # End of loop over output results, now inner loop with results
                                for i in range(length_input_window):
                                    # Apply the learned kernel onto the input value x
                                    # sum over the input window length such that we obtain a single kernelized output
                                    # series causally convoluted with respect to all other series/input window length
                                    result[b, h, xx, y, j, f] += (
                                        kernel[h, y, xx, j, i] * x[b, xx, i, f]
                                    )
        # Result:
        # (batch_size, number_of_heads, number_of_series, number_of_series, length_input_window, hidden_dimensionality)
        return result

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
