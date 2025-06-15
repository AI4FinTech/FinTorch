from typing import Optional

import torch
import torch.nn as nn
from fintorch.layers.explainable.einsum import einsum
from fintorch.layers.explainable import Dropout, Softmax


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
        activation (Softmax): The softmax activation function.
        dropout (Dropout): The dropout layer.

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

        propagate(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Propagates relevance backwards for explainable AI.

            Args:
                x (torch.Tensor): Relevance tensor to propagate backwards.

            Returns:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Relevance tensors for Q, K, V.


    References:
    - Kong, Lingbai, Wengen Li, Hanchen Yang, Yichao Zhang, Jihong Guan, and Shuigeng Zhou. 2024. "CausalFormer:
      An Interpretable Transformer for Temporal Causal Discovery." arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/2406.16708

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

        self.activation = Softmax(dim=-1)
        self.dropout = Dropout(0.1)

        # Initialize custom einsum operator for explainable AI
        self.qk_einsum = einsum('bhik,bhjk->bhij')
        self.attention_einsum = einsum('bhij,bhjitf->bhitf')
        self.mask_einsum = einsum('bhij,bhij->bhij')

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Q, K, (batch_size, number_of_heads, number_of_series, hidden_dimensionality)
        # V: (batch_size, number_of_heads, number_of_series, number_of_series, length_input_window, feature_dim)

        qk = self.qk_einsum(Q, K)
        # qk: (batch_size, number_of_heads, number_of_series, number_of_series)

        # Scale the dot product by the square root of the hidden dimensionality
        qk = qk / ((self.input_window * self.hidden_dimensionality) ** 0.5)

        # Apply masking if provided to ensure zero prediction for masked positions
        # we use the einsum operator, because that has explainable AI components in it
        if mask is not None:
            # mask: (batch_size, number_of_heads, number_of_series, number_of_series)
            # Create mask with -inf where mask == 0, otherwise 0
            mask_values = torch.where(mask == 0, torch.tensor(float("-inf"), device=qk.device), torch.tensor(0.0, device=qk.device))
            qk = qk + self.mask_einsum(torch.ones_like(qk), mask_values)

        # attention matrix R^{batch size, number_of_heads, number_of_series, number_of_series}
        attention_weights = self.activation(qk / self.tau)
        attention_weights = self.dropout(attention_weights)

        # output R^{batch size, number_of_heads, number_of_series, length_input_window, hidden}
        # einsum over dimension j = number_of_series
        output = self.attention_einsum(attention_weights, V)

        return output

    def propagate(
        self, x: torch.Tensor
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Propagates relevance backwards for explainable AI.

        Args:
            x: Relevance tensor to propagate backwards

        Returns:
            Tuple of relevance tensors for Q, K, V
        """

        # propagate relevance through attention einsum
        relevance_attention, relevance_value = self.attention_einsum.propagate(x)

        # propagate relevance through Dropout
        relevance_attention = self.dropout.propagate(relevance_attention)

        #propagate relevance through Softmax
        relevance_attention = self.activation.propagate(relevance_attention)

        # propagate relevance through masked einsum
        relevance_attention = self.mask_einsum.propagate(relevance_attention)

        # invert scaling
        relevance_attention *= ((self.input_window * self.hidden_dimensionality) ** 0.5)

        # Relevance propagation Q and K
        relevance_query, relevance_key = self.qk_einsum.propagate(relevance_attention)

        return relevance_query, relevance_key, relevance_value
