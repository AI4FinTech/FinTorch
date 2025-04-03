from typing import Optional
import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is a PyTorch module that implements a multi-head attention mechanism
    for time series data. It computes attention scores and applies them to the input data
    to produce context-aware representations.
    Attributes:
        number_of_heads (int): The number of attention heads.
        number_of_series (int): The number of time series in the input data.
        length_input_window (int): The length of the input time window.
        embedding_size (int): The size of the embedding dimension.
        tau (float): A scaling factor for the attention weights.
        hidden_dimensionality (int): The dimensionality of each attention head,
            calculated as embedding_size divided by number_of_heads.
        activation (nn.Softmax): The softmax activation function applied to the attention scores.
        dropout (nn.Dropout): Dropout layer applied to the attention weights for regularization.
    Methods:
        forward(x, mask=None):
            Computes the forward pass of the multi-head attention mechanism.
            Args:
                x (torch.Tensor): Input tensor of shape
                    (batch_size, number_of_series, length_input_window, hidden_dimensionality).
                mask (torch.Tensor, optional): A binary mask tensor of the same shape as the
                    attention scores, where 0 indicates positions to be masked.
            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, number_of_series, length_input_window, hidden_dimensionality).
    """

    def __init__(
        self,
        number_of_heads: int,
        number_of_series: int,
        length_input_window: int,
        embedding_size: int,
        tau: float,
    ) -> None:
        super().__init__()

        self.number_of_heads = number_of_heads
        self.number_of_series = number_of_series
        self.input_window = length_input_window
        self.tau = tau
        self.embedding_size = embedding_size
        self.hidden_dimensionality = embedding_size // number_of_heads

        self.activation = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Make Q, K, V
        Q, K, V = x, x, x
        # Q, K, V: (batch_size, number_of_series, length_input_window, hidden_dimensionality)

        qk = torch.matmul(Q, K.transpose(-2, -1))
        # qk: (batch_size, number_of_series, length_input_window, length_input_window)

        # Scale the dot product by the square root of the hidden dimensionality
        qk = qk / ((self.input_window * self.hidden_dimensionality) ** 0.5)

        # Apply masking if provided to ensure zero prediction for masked positions
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float("-inf"))

        # attention matrix R^{batch size, number_of_series, lenght_input_window, length_input_window}
        attention_weights = self.activation(qk / self.tau)
        attention_weights = self.dropout(attention_weights)

        # output R^{batch size, number_of_series, length_input_window, hidden}
        output = torch.matmul(attention_weights, V)

        return output
