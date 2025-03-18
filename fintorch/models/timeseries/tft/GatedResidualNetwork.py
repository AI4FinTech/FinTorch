import torch.nn as nn
import torch.nn.functional as F


class AddNorm(nn.Module):
    """
    AddNorm is a module that applies residual connection followed by layer normalization.

    This module takes two inputs: the main input tensor `x` and a skip connection tensor `skip`.
    It adds these two tensors element-wise and then applies layer normalization to the result.

    Args:
        dimension (int): The size of the input tensor's last dimension, which is used to initialize
                         the LayerNorm module.

    Forward Inputs:
        x (torch.Tensor): The main input tensor of shape (..., dimension).
        skip (torch.Tensor): The skip connection tensor of shape (..., dimension).

    Forward Returns:
        torch.Tensor: The result of applying the residual connection and layer normalization,
                      with the same shape as the input tensors.
    """

    def __init__(self, dimension):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x, skip):
        return self.norm(x + skip)


class GatedResidualNetwork(nn.Module):
    """
    A Gated Residual Network (GRN) module for processing input data with optional context.
    The GRN applies a series of transformations including input projection,
    gated linear unit (GLU), and residual connections with layer normalization.
    It optionally incorporates context information into the computation.
    Attributes:
        input_proj (nn.Linear): Linear layer for projecting the input to the hidden size.
        elu (nn.ELU): Exponential Linear Unit activation function.
        context_proj (nn.Linear, optional): Linear layer for projecting the context to the hidden size.
        fully_connected2 (nn.Linear): Fully connected layer for further processing.
        GLU (GatedLinearUnit): Gated Linear Unit for gating mechanism.
        add_norm (AddNorm): Add & Normalize layer for residual connections.
    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        output_size (int): Size of the output features.
        dropout (float): Dropout rate for regularization.
        context_size (int, optional): Size of the context features. If None, context is not used.
    Methods:
        forward(a, context=None):
            Forward pass of the GRN.
            Args:
                a (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
                context (torch.Tensor, optional): Context tensor of shape (batch_size, context_size).
                    If provided, it is incorporated into the computation.
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, output_size).
    """

    def __init__(self, input_size, hidden_size, output_size, dropout, context_size):
        super(GatedResidualNetwork, self).__init__()

        # input projection layer
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()

        # Context projection layer
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)

        # eta 1
        self.fully_connected2 = nn.Linear(hidden_size, hidden_size)
        self.GLU = GatedLinearUnit(hidden_size, output_size, dropout=dropout)
        self.add_norm = AddNorm(hidden_size)

    def forward(self, a, context=None):
        a = self.input_proj(a)

        residual = a

        if context is not None:
            # Equation (4)
            context = self.context_proj(context)
            context = context.unsqueeze(1).expand(
                -1, a.shape[1], -1
            )  # expand the context
            # TODO: check if this is the correct way
            a = a + context
        a = self.elu(a)
        a = self.fully_connected2(a)

        a = self.GLU(a)
        a = self.add_norm(a, residual)

        return a


class GatedLinearUnit(nn.Module):
    """
    A PyTorch implementation of the Gated Linear Unit (GLU).

    The Gated Linear Unit is a mechanism that applies a gating mechanism
    to the input tensor, reducing its dimensionality while preserving
    important features. It is commonly used in sequence modeling and
    time-series tasks.

    Args:
        input_dimension (int): The dimensionality of the input tensor.
        output_dimension (int, optional): The dimensionality of the output tensor.
            If not provided, it defaults to the input_dimension.
        dropout (float, optional): Dropout rate to apply to the input tensor.
            If None, no dropout is applied.

    Attributes:
        dropout (nn.Dropout or None): Dropout layer applied to the input tensor,
            or None if no dropout is specified.
        dense (nn.Linear): Linear layer that projects the input tensor to twice
            the output dimension for the gating mechanism.

    Methods:
        forward(upgamma):
            Applies the dropout (if specified), linear transformation, and
            gated linear unit (GLU) activation to the input tensor.

            Args:
                upgamma (torch.Tensor): Input tensor of shape
                    (batch_size, ..., input_dimension).

            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, ..., output_dimension).
    """


    def __init__(
        self, input_dimension: int, output_dimension: int = None, dropout: float = None
    ) -> None:
        super(GatedLinearUnit, self).__init__()

        # Create a valid dropout layer or set to None
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        # Linear projection of the eta
        output_dim = output_dimension or input_dimension
        self.dense = nn.Linear(input_dimension, output_dim * 2)

    def forward(self, upgamma):
        if self.dropout is not None:
            upgamma = self.dropout(upgamma)

        upgamma = self.dense(upgamma)
        return F.glu(
            upgamma, dim=-1
        )  # reduces the dimensionality to the original dimensionality


class GatedAddNorm(nn.Module):
    """
    A PyTorch module that combines a Gated Linear Unit (GLU) with an Add & Norm operation.
    This module is designed to process input data, apply gating mechanisms, and normalize
    the output while incorporating skip connections.

    Args:
        input_dimension (int): The dimensionality of the input features.
        hidden_dimensions (int): The dimensionality of the hidden features for the AddNorm layer.
        output_dimension (int): The dimensionality of the output features after the GLU.
        skip_dimension (int): The dimensionality of the skip connection input.
        dropout (float): The dropout rate applied within the GLU.

    Methods:
        forward(x, skip):
            Processes the input tensor `x` through the GLU, combines it with the skip connection
            tensor `skip`, and applies the Add & Norm operation.

            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, input_dimension).
                skip (torch.Tensor): The skip connection tensor of shape (batch_size, skip_dimension).

            Returns:
                torch.Tensor: The output tensor after applying the GLU and Add & Norm operations.
    """

    def __init__(
        self,
        input_dimension,
        hidden_dimensions,
        output_dimension,
        skip_dimension,
        dropout,
    ):
        super(GatedAddNorm, self).__init__()

        self.GLU = GatedLinearUnit(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            dropout=dropout,
        )
        self.add_norm = AddNorm(hidden_dimensions)

    def forward(self, x, skip):
        return self.add_norm(self.GLU(x), skip)
