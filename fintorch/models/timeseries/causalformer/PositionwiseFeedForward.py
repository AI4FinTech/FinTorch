import torch
import torch.nn as nn
from fintorch.layers.explainable import Linear, ReLU, Dropout


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

        propagate(x: torch.Tensor) -> torch.Tensor:
            Propagates relevance backwards for explainable AI.

            Args:
                x (torch.Tensor): Relevance tensor to propagate backwards.

            Returns:
                torch.Tensor: Propagated relevance tensor.


    References:
    - Kong, Lingbai, Wengen Li, Hanchen Yang, Yichao Zhang, Jihong Guan, and Shuigeng Zhou. 2024. “CausalFormer:
      An Interpretable Transformer for Temporal Causal Discovery.” arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/2406.16708

    """

    def __init__(
        self, input_dim: int, hidden_dimensionality: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.fc1 = Linear(in_features=input_dim, out_features=hidden_dimensionality)
        self.fc2 = Linear(in_features=hidden_dimensionality, out_features=input_dim)
        self.relu = ReLU()
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def propagate(self, x: torch.Tensor) -> torch.Tensor:
        try:
            rel = self.fc2.propagate(x)
            if isinstance(rel, torch.Tensor):
                rel = self.dropout.propagate(rel)
                if isinstance(rel, torch.Tensor):
                    rel = self.relu.propagate(rel)
                    if isinstance(rel, torch.Tensor):
                        rel = self.fc1.propagate(rel)
                        if isinstance(rel, torch.Tensor):
                            return rel
            # Fallback to input if any step fails type check
            return x
        except Exception:
            # Fallback to input if propagation fails
            return x
