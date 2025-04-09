import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Positionwise FeedForward module for the Transformer model.

    This module applies a position-wise feed-forward network to the input tensor.
    It consists of two linear layers with a ReLU activation in between and a dropout layer.

    Attributes:
        fc1 (nn.Linear): The first linear layer.
        fc2 (nn.Linear): The second linear layer.
        relu (nn.ReLU): The ReLU activation function.
        dropout (nn.Dropout): The dropout layer.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the position-wise feed-forward network.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor.
    """

    def __init__(
        self, input_dim: int, hidden_dimensionality: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dimensionality)
        self.fc2 = nn.Linear(hidden_dimensionality, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
