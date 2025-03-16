import torch.nn as nn
import torch.nn.functional as F


class AddNorm(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(output_dimension)

    def forward(self, x, skip):
        return self.norm(x + skip)


class GatedResidualNetwork(nn.Module):
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
        self.add_norm = AddNorm(hidden_size, output_size)

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
        self.add_norm = AddNorm(hidden_dimensions, skip_dimension)

    def forward(self, x, skip):
        return self.add_norm(self.GLU(x), skip)
