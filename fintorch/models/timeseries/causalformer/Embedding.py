import torch.nn as nn
import torch


class Embedding(nn.Module):
    """
    Embedding module for time-series data.
    This module is designed to project input time-series data into a higher-dimensional
    space using a linear projection layer, followed by layer normalization and dropout.
    Attributes:
        number_of_series (int): The number of time-series in the input data.
        length_input_window (int): The length of the input time-series window.
        feature_dimensionality (int): The dimensionality of features for each time step.
        hidden_dimensionality (int): The dimensionality of the hidden embedding space (d_model).
        dropout (float): The dropout rate applied after the embedding and normalization.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the embedding module. Takes a batch of time-series data and
            returns the embedded representation.
            Args:
                x (torch.Tensor): Input tensor of shape
                    (batch_size, number_of_series, length_input_window, feature_dimensionality).
            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, number_of_series, hidden_dimensionality).

    References:
    - Kong, Lingbai, Wengen Li, Hanchen Yang, Yichao Zhang, Jihong Guan, and Shuigeng Zhou. 2024. “CausalFormer:
      An Interpretable Transformer for Temporal Causal Discovery.” arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/2406.16708
    """

    def __init__(
        self,
        number_of_series: int,
        length_input_window: int,
        feature_dimensionality: int,
        hidden_dimensionality: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.number_of_series = number_of_series
        self.length_input_window = length_input_window
        self.feature_dimensionality = feature_dimensionality
        self.hidden_dimensionality = hidden_dimensionality
        self.dropout = dropout
        self.d_model = hidden_dimensionality

        # A liinear projection layer to generate embeddingd for input time-series.
        # We project from R^{N x T} -> R^{N x d} with d as the dimensionality

        # Linear projection layer
        self.embedding = nn.Linear(
            in_features=self.length_input_window * self.feature_dimensionality,
            out_features=self.d_model,
            bias=True,
        )
        self.normalization = nn.LayerNorm(self.d_model)
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, number_of_series, length_input_window, feature_dimensionality)
        batch_size = x.shape[0]
        x = x.view(
            batch_size,
            self.number_of_series,
            self.length_input_window * self.feature_dimensionality,
        )
        # Apply the linear projection
        x = self.embedding(x)
        # Apply layer normalization
        x = self.normalization(x)
        # Apply dropout
        x = self.dropout_layer(x)
        return x
