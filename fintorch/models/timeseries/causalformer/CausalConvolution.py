import torch
import torch.nn as nn


class CausalConvolution(nn.Module):
    def __init__(self, number_of_series, length_input_window, number_of_heads):
        super().__init__()

        self.kernel = nn.Parameter(
            torch.ones(
                (number_of_heads, number_of_series, length_input_window),
                dtype=torch.float,
            )
        )
        self.register_parameter(self.kernel)

    def forward(self, x):
        # x : (batch_size, number_of_series, length_input_window, hidden_dimensionality)
        pass
