import torch.nn as nn
import torch

from fintorch.models.timeseries.causalformer.Encoder import Encoder


class CausalFormer(nn.Module):
    """
    CausalFormer: A PyTorch module for time series forecasting using a transformer-based architecture.
    Args:
        number_of_layers (int): Number of layers in the encoder.
        number_of_heads (int): Number of attention heads in the encoder.
        number_of_series (int): Number of time series in the input data.
        length_input_window (int): Length of the input time window.
        length_output_window (int): Length of the output time window.
        embedding_size (int): Size of the embedding for each time series.
        feature_dimensionality (int): Dimensionality of the input features.
        ffn_hidden_dimensionality (int): Dimensionality of the feed-forward network's hidden layer.
        output_dimensionality (int): Dimensionality of the output features.
        tau (float): Temperature parameter for attention scaling.
        dropout (float): Dropout rate for regularization.
    Attributes:
        output_window (int): Length of the output time window.
        encoder (Encoder): Encoder module for processing the input time series.
        fully_connected (nn.Linear): Fully connected layer for projecting the encoder output to the desired output dimensionality.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor of shape
                    [batch_size, number_of_series, length_input_window, feature_dimensionality].
            Returns:
                torch.Tensor: Output tensor of shape
                    [batch_size, number_of_series, length_output_window, output_dimensionality].
    """

    def __init__(
        self,
        number_of_layers: int,
        number_of_heads: int,
        number_of_series: int,
        length_input_window: int,
        length_output_window: int,
        embedding_size: int,
        feature_dimensionality: int,
        ffn_hidden_dimensionality: int,
        output_dimensionality: int,
        tau: float,
        dropout: float,
    ):
        super().__init__()

        self.output_window = length_output_window

        self.encoder = Encoder(
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            number_of_series=number_of_series,
            length_input_window=length_input_window,
            embedding_size=embedding_size,
            feature_dimensionality=feature_dimensionality,
            ffn_hidden_dimensionality=ffn_hidden_dimensionality,
            tau=tau,
            dropout=dropout,
        )

        self.fully_connected = nn.Linear(
            in_features=feature_dimensionality, out_features=output_dimensionality
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, number_of_series, length_input_window, feature_dimensionality]

        x = self.encoder(x)
        # x: [batch_size, number_of_series, length_input_window, feature_dimensionality]
        x = self.fully_connected(x)
        # x: [batch_size, number_of_series, length_input_window, output_dimensionality]

        return x[
            :, :, -self.output_window :, :
        ]  # Return the output window time step(s) for each series
