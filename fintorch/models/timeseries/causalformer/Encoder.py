import torch.nn as nn
import torch

from fintorch.models.timeseries.causalformer.Embedding import Embedding
from fintorch.models.timeseries.causalformer.EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    """
    Encoder module for time series modeling.
    This module is designed to process multivariate time series data using a stack of
    encoder layers. It first embeds the input data and then applies multiple encoder
    layers to extract meaningful representations.
    Args:
        number_of_layers (int): Number of encoder layers in the stack.
        number_of_heads (int): Number of attention heads in each encoder layer.
        number_of_series (int): Number of time series in the input data.
        length_input_window (int): Length of the input time window.
        embedding_size (int): Dimensionality of the embedding space.
        feature_dimensionality (int): Dimensionality of the input features.
        ffn_hidden_dimensionality (int): Dimensionality of the hidden layer in the feed-forward network.
        tau (float): Temperature parameter for scaling attention scores.
        dropout (float): Dropout rate applied in the embedding and encoder layers.
    Attributes:
        embedding (Embedding): Embedding layer to transform input data into the embedding space.
        layers (nn.ModuleList): List of encoder layers applied sequentially.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the encoder. Processes the input tensor through the embedding
            layer and the stack of encoder layers.
    Input Shape:
        x: torch.Tensor of shape [batch_size, number_of_series, length_input_window, feature_dimensionality]
    Output Shape:
        torch.Tensor of shape [batch_size, number_of_series, embedding_size]

    References:
    - Kong, Lingbai, Wengen Li, Hanchen Yang, Yichao Zhang, Jihong Guan, and Shuigeng Zhou. 2024. “CausalFormer:
      An Interpretable Transformer for Temporal Causal Discovery.” arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/2406.16708

    """

    def __init__(
        self,
        number_of_layers: int,
        number_of_heads: int,
        number_of_series: int,
        length_input_window: int,
        embedding_size: int,
        feature_dimensionality: int,
        ffn_hidden_dimensionality: int,
        tau: float,
        dropout: float,
    ):
        super().__init__()

        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.number_of_series = number_of_series
        self.length_input_window = length_input_window
        self.embedding_size = embedding_size
        self.feature_dimensionality = feature_dimensionality
        self.ffn_hidden_dimensionality = ffn_hidden_dimensionality
        self.tau = tau
        self.dropout = dropout

        self.embedding = Embedding(
            number_of_series=self.number_of_series,
            length_input_window=self.length_input_window,
            feature_dimensionality=self.feature_dimensionality,
            hidden_dimensionality=embedding_size,
            dropout=self.dropout,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    number_of_heads=self.number_of_heads,
                    number_of_series=self.number_of_series,
                    length_input_window=self.length_input_window,
                    embedding_size=self.embedding_size,
                    feature_dimensionality=self.feature_dimensionality,
                    ffn_hidden_dimensionality=self.ffn_hidden_dimensionality,
                    tau=self.tau,
                    dropout=self.dropout,
                )
                for _ in range(self.number_of_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, number_of_series, length_input_window, feature_dimensionality]
        # x_emb: [batch_size, number_of_series, embedding_size]
        x_emb = self.embedding(x)

        for layer in self.layers:
            x = layer(x_emb, x)

        return x
