from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for the CausalFormer model.

    This module implements the multi-head attention mechanism, which allows the model to
    attend to information from different representation subspaces at different positions.

    Attributes:
        number_of_heads (int): The number of attention heads.
        number_of_series (int): The number of time series in the input data.
        length_input_window (int): The length of the input time window.
        embedding_size (int): The size of the input embedding.
        tau (float): A scaling factor for the attention weights.
        hidden_dimensionality (int): The dimensionality of the hidden space for each head.
        activation (nn.Softmax): The softmax activation function.
        dropout (nn.Dropout): The dropout layer.

    Methods:
        forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            Forward pass of the multi-head attention module.

            Args:
                Q (torch.Tensor): Query tensor.
                K (torch.Tensor): Key tensor.
                V (torch.Tensor): Value tensor.
                mask (Optional[torch.Tensor]): Optional mask tensor.

            Returns:
                torch.Tensor: The output tensor.


    References:
    - Kong, Lingbai, Wengen Li, Hanchen Yang, Yichao Zhang, Jihong Guan, and Shuigeng Zhou. 2024. “CausalFormer:
      An Interpretable Transformer for Temporal Causal Discovery.” arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/2406.16708

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
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Q, K, (batch_size, number_of_heads, number_of_series, hidden_dimensionality)
        # V: (batch_size, number_of_heads, number_of_series, number_of_series, length_input_window, feature_dim)

        qk = torch.matmul(Q, K.transpose(-2, -1))
        # qk: (batch_size, number_of_heads, number_of_series, number_of_series)

        # Scale the dot product by the square root of the hidden dimensionality
        qk = qk / ((self.input_window * self.hidden_dimensionality) ** 0.5)

        # Apply masking if provided to ensure zero prediction for masked positions
        if mask is not None:
            # mask: (batch_size, number_of_heads, number_of_series, number_of_series)
            qk = qk.masked_fill(mask == 0, float("-inf"))

        # attention matrix R^{batch size, number_of_heads, number_of_series, number_of_series}
        attention_weights = self.activation(qk / self.tau)
        attention_weights = self.dropout(attention_weights)

        # output R^{batch size, number_of_heads, number_of_series, length_input_window, hidden}
        # einsum over dimension j = number_of_series
        output = torch.einsum("bhij,bhjitf->bhitf", attention_weights, V)

        return output
