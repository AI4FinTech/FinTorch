from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is a PyTorch module that implements a multi-head attention mechanism
    for time series data. It computes scaled dot-product attention across multiple heads
    and series, allowing the model to focus on different parts of the input sequence.

    Attributes:
        number_of_heads (int): The number of attention heads.
        number_of_series (int): The number of time series in the input.
        length_input_window (int): The length of the input time window.
        embedding_size (int): The size of the input embedding.
        tau (float): A scaling factor for the attention weights.
        hidden_dimensionality (int): The dimensionality of each attention head, computed as
            embedding_size divided by number_of_heads.
        activation (nn.Softmax): The softmax activation function applied to the attention weights.
        dropout (nn.Dropout): Dropout layer applied to the attention weights for regularization.

    Methods:
        forward(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            Computes the forward pass of the multi-head attention mechanism.

            Args:
                x (torch.Tensor): The input tensor of shape
                    (batch_size, number_of_heads, number_of_series, length_input_window, hidden_dimensionality).
                mask (Optional[torch.Tensor]): An optional mask tensor of shape
                    (batch_size, number_of_heads, number_of_series, length_input_window, length_input_window)
                    to prevent attention to certain positions.

            Returns:
                torch.Tensor: The output tensor of shape
                    (batch_size, number_of_heads, number_of_series, length_input_window, hidden_dimensionality).


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

        print(f"MultiheadAttention: shape of Q: {Q.shape} -- correct")
        print(f"MultiheadAttention: shape of K: {K.shape} -- correct")
        print(f"MultiheadAttention: shape of V: {V.shape} -- correct")

        qk = torch.matmul(Q, K.transpose(-2, -1))
        # qk: (batch_size, number_of_heads, number_of_series, number_of_series)
        print(f"MultiheadAttention: shape of qk: {qk.shape} -- correct")

        # Scale the dot product by the square root of the hidden dimensionality
        qk = qk / ((self.input_window * self.hidden_dimensionality) ** 0.5)

        # Apply masking if provided to ensure zero prediction for masked positions
        if mask is not None:
            # mask: (batch_size, number_of_heads, number_of_series, number_of_series)
            qk = qk.masked_fill(mask == 0, float("-inf"))

        # attention matrix R^{batch size, number_of_heads, number_of_series, number_of_series}
        attention_weights = self.activation(qk / self.tau)
        attention_weights = self.dropout(attention_weights)

        print(
            f"MultiheadAttention: shape of attention_weights: {attention_weights.shape} -- correct"
        )
        # output R^{batch size, number_of_heads, number_of_series, length_input_window, hidden}
        # einsum over dimension j = number_of_series
        output = torch.einsum("bhij,bhjitf->bhitf", attention_weights, V)

        print(f"MultiheadAttention: shape of output: {output.shape} -- correct")

        return output
