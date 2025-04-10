import torch
import torch.nn as nn
from fintorch.models.timeseries.causalformer.MultivariateCausalAttention import (
    MultivariateCausalAttention,
)
from fintorch.models.timeseries.causalformer.PositionwiseFeedForward import (
    PositionwiseFeedForward,
)


class EncoderLayer(nn.Module):
    """
    EncoderLayer module for the CausalFormer model.
    This module combines a multivariate causal attention mechanism with a position-wise feed-forward network.
    Attributes:
        number_of_heads (int): The number of attention heads.
        number_of_series (int): The number of time series in the input data.
        length_input_window (int): The length of the input time window.
        embedding_size (int): The size of the input embedding.
        feature_dimensionality (int): The dimensionality of features for each time step.
        tau (float): A scaling factor for the attention weights.
        dropout (float): The dropout rate applied after the embedding and normalization.


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
        feature_dimensionality: int,
        ffn_hidden_dimensionality: int,
        tau: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.number_of_heads = number_of_heads
        self.number_of_series = number_of_series
        self.length_input_window = length_input_window
        self.embedding_size = embedding_size
        self.feature_dimensionality = feature_dimensionality
        self.tau = tau
        self.dropout = dropout

        self.multivariatecausalattention = MultivariateCausalAttention(
            number_of_heads=self.number_of_heads,
            number_of_series=self.number_of_series,
            length_input_window=self.length_input_window,
            embedding_size=self.embedding_size,
            feature_dimensionality=self.feature_dimensionality,
            tau=self.tau,
            dropout=self.dropout,
        )

        self.normalization = nn.LayerNorm([length_input_window, feature_dimensionality])
        self.dropout_layer_1 = nn.Dropout(self.dropout)
        self.positionwisefeedforward = PositionwiseFeedForward(
            input_dim=self.feature_dimensionality,
            hidden_dimensionality=ffn_hidden_dimensionality,
            dropout_rate=self.dropout,
        )

        self.normalization_2 = nn.LayerNorm(
            [length_input_window, feature_dimensionality]
        )
        self.dropout_layer_2 = nn.Dropout(self.dropout)

    def forward(self, x_emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x_emb [batch_size, number_of_series, hidden_dim]
        # x [batch_size, number_of_series, length_input_window, feature_dimensionality]
        q, k = x_emb, x_emb
        x = self.multivariatecausalattention(q, k, x)
        # x [batch_size, number_of_series, length_input_window, feature_dimensionality]

        # Dropout + layernorm before feedforward
        x = self.dropout_layer_1(x)
        x = self.normalization(x)

        # Feedforward module
        x = self.positionwisefeedforward(x)

        # Dropout + layernorm after feedforward
        x = self.dropout_layer_2(x)
        x = self.normalization_2(x)

        return x
