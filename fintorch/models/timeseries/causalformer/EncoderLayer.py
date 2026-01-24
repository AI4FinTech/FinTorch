import torch
import torch.nn as nn
from fintorch.models.timeseries.causalformer.MultivariateCausalAttention import (
    MultivariateCausalAttention,
)
from fintorch.models.timeseries.causalformer.PositionwiseFeedForward import (
    PositionwiseFeedForward,
)
from fintorch.layers.explainable import LayerNorm, Dropout



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


    Methods:
        forward(x_emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            Forward pass of the encoder layer.

            Args:
                x_emb (torch.Tensor): Embedded input tensor.
                x (torch.Tensor): Raw input tensor.

            Returns:
                torch.Tensor: Output tensor after attention and feed-forward processing.

        propagate(x: torch.Tensor) -> torch.Tensor:
            Propagates relevance backwards for explainable AI.

            Args:
                x (torch.Tensor): Relevance tensor to propagate backwards.

            Returns:
                torch.Tensor: Propagated relevance tensor.

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

        self.normalization = LayerNorm((length_input_window, feature_dimensionality))
        self.dropout_layer_1 = Dropout(p=self.dropout)
        self.positionwisefeedforward = PositionwiseFeedForward(
            input_dim=self.feature_dimensionality,
            hidden_dimensionality=ffn_hidden_dimensionality,
            dropout_rate=self.dropout,
        )

        self.normalization_2 = LayerNorm(
            (length_input_window, feature_dimensionality)
        )
        self.dropout_layer_2 = Dropout(p=self.dropout)

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

    def propagate(self, x: torch.Tensor) -> torch.Tensor:
        # Reverse: normalization_2 -> dropout_layer_2 -> positionwisefeedforward -> normalization -> dropout_layer_1 -> multivariatecausalattention

        x = self.normalization_2.propagate(x)
        x = self.dropout_layer_2.propagate(x)
        x = self.positionwisefeedforward.propagate(x)
        x = self.normalization.propagate(x)
        x = self.dropout_layer_1.propagate(x)
        x = self.multivariatecausalattention.propagate(x)

        return x
