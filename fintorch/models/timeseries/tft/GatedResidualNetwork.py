from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddNorm(nn.Module):
    """
    Add & Normalize module.

    This module implements a combination of addition and layer normalization.
    It's designed to add a skip connection to the input and then apply layer
    normalization to the result.

    Args:
        dimension (int): The dimensionality of the input tensor.

    Attributes:
        norm (nn.LayerNorm): Layer normalization module.

    Methods:
        forward(x, skip):
            Applies the skip connection and layer normalization to the input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, ..., dimension).
                skip (torch.Tensor): Skip connection tensor of shape
                    (batch_size, ..., dimension).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, ..., dimension).
    """

    def __init__(self, dimension: int):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> Any:
        # Assumes x.shape == skip.shape
        return self.norm(x + skip)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) module.

    This module implements a Gated Residual Network, which is a type of neural
    network layer that combines residual connections with gating mechanisms.
    It's designed to process input data through a series of linear transformations
    and non-linear activations, with the ability to incorporate context information.

    Args:
        input_size (int): The dimensionality of the input tensor.
        hidden_size (int): The dimensionality of the hidden layer.
        output_size (int): The dimensionality of the output tensor.
        dropout (float): Dropout rate to apply to the input tensor.
        context_size (int): The dimensionality of the context tensor.

    Attributes:
        input_proj (nn.Linear): Linear layer for projecting the input to the hidden size.
        elu (nn.ELU): ELU activation function.
        context_proj (nn.Linear or None): Linear layer for projecting the context to the hidden size.
        fully_connected2 (nn.Linear): Linear layer for further processing the hidden representation.
        GLU (GatedLinearUnit): Gated Linear Unit for gating mechanism.
        add_norm (AddNorm): Add & Normalize layer for residual connections.

    Methods:
        forward(a, context=None):
            Applies the GRN to the input tensor.

            Args:
                a (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
                context (torch.Tensor, optional): Context tensor of shape (batch_size, context_size).
                    Defaults to None.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_size).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        context_size: int,
    ):
        super(GatedResidualNetwork, self).__init__()

        # input projection layer
        # [batch size, sequence length, input size] -> [batch size, sequence length, output size]
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()

        # Context projection layer
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)

        # eta 1
        self.fully_connected2 = nn.Linear(hidden_size, hidden_size)
        self.GLU = GatedLinearUnit(hidden_size, output_size, dropout=dropout)
        self.add_norm = AddNorm(hidden_size)

    def forward(
        self, a: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # a: [batch size, sequence length, input size]
        a = self.input_proj(a)

        residual = a

        if context is not None:
            # Equation (4)
            # project context to hidden dim
            context = self.context_proj(context)
            # expand over sequence length
            context = context.unsqueeze(1).expand(  # type: ignore[union-attr]
                -1, a.shape[1], -1
            )  # expand the context
            a = a + context
        a = self.elu(a)
        a = self.fully_connected2(a)

        a = self.GLU(a)
        a = self.add_norm(a, residual)

        return a  # [batch size, sequence length, output size]


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) module.

    This module applies a linear transformation to the input and then splits
    the result into two parts. It applies a sigmoid activation to one part
    and performs element-wise multiplication with the other part.

    Args:
        input_dimension (int): The dimensionality of the input tensor.
        output_dimension (int, optional): The dimensionality of the output tensor.
            If not provided, it defaults to the input_dimension.
        dropout (float, optional): Dropout rate to apply to the input tensor.
            If None, no dropout is applied. Defaults to None.

    Attributes:
        dropout (nn.Dropout or None): Dropout layer if dropout rate is provided.
        dense (nn.Linear): Linear layer for transforming the input.

    Methods:
        forward(upgamma):
            Applies the linear transformation and GLU operation to the input tensor.

            Args:
                upgamma (torch.Tensor): Input tensor of shape
                    (batch_size, ..., input_dimension).

            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, ..., output_dimension).
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super(GatedLinearUnit, self).__init__()

        # Create a valid dropout layer or set to None
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        # Linear projection of the eta
        output_dim = output_dimension or input_dimension
        self.dense = nn.Linear(input_dimension, output_dim * 2)

    def forward(self, upgamma: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            upgamma = self.dropout(upgamma)

        upgamma = self.dense(upgamma)
        return F.glu(
            upgamma, dim=-1
        )  # reduces the dimensionality to the original dimensionality


class GatedAddNorm(nn.Module):
    """
    Gated Add & Norm module.

    This module combines a Gated Linear Unit (GLU) with an Add & Norm operation.
    It's designed to process input data through a GLU and then apply a residual
    connection with layer normalization, similar to the AddNorm module but with
    an added gating mechanism.

    Args:
        input_dimension (int): The dimensionality of the input tensor.
        hidden_dimensions (int): The dimensionality of the hidden layer.
        output_dimension (int): The dimensionality of the output tensor.
        skip_dimension (int, optional): The dimensionality of the skip connection tensor.
            If not provided, it defaults to the output_dimension.
        dropout (float, optional): Dropout rate to apply to the input tensor.
            Defaults to 0.1.

    Attributes:
        skip_dimension (int): The dimensionality of the skip connection tensor.
        skip_layer_proj (nn.Linear or None): Linear layer for projecting the skip
            connection tensor to the output dimension if necessary.
        GLU (GatedLinearUnit): Gated Linear Unit for gating mechanism.
        add_norm (AddNorm): Add & Normalize layer for residual connections.

    Methods:
        forward(x, skip):
            Applies the skip connection projection (if necessary), GLU, and
            Add & Norm operation to the input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, ..., input_dimension).
                skip (torch.Tensor): Skip connection tensor of shape
                    (batch_size, ..., skip_dimension).

            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, ..., output_dimension).
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: int,
        output_dimension: int,
        skip_dimension: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super(GatedAddNorm, self).__init__()

        self.skip_dimension = skip_dimension

        if skip_dimension is not None and skip_dimension != output_dimension:
            self.skip_layer_proj = nn.Linear(skip_dimension, output_dimension)  # type: ignore
        else:
            self.skip_layer_proj = None  # type: ignore[assignment]

        self.GLU = GatedLinearUnit(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            dropout=dropout,
        )
        self.add_norm = AddNorm(hidden_dimensions)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> Any:
        if self.skip_layer_proj is not None:
            # skip is of a different dimensionality, project first
            skip = self.skip_layer_proj(skip)

        return self.add_norm(self.GLU(x), skip)
