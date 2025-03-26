from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fintorch.models.timeseries.tft.GatedResidualNetwork import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) module.

    This module implements a Variable Selection Network, which is a type of
    neural network layer designed to select relevant input variables from a
    set of multiple input features. It uses Gated Residual Networks (GRNs)
    to process each input feature and then combines them using a softmax-based
    attention mechanism.

    Args:
        inputs (Dict[str, int]): A dictionary where keys are input names and
            values are their respective dimensions.
        hidden_dimensions (int): The dimensionality of the hidden layers in the GRNs.
        dropout (float): Dropout rate to apply to the input tensor.
        context_size (int): The dimensionality of the context tensor.

    Attributes:
        inputs_length (int): The number of input features.
        input_grns (nn.ModuleDict): A dictionary of GRNs, one for each input feature.
        input_size_total (int): The total dimensionality of all input features.
        grn_input (GatedResidualNetwork): GRN for processing the concatenated input features.
        softmax (nn.Softmax): Softmax layer for computing attention weights.

    Methods:
        forward(x, context=None):
            Applies the VSN to the input features.

            Args:
                x (Dict[str, torch.Tensor]): A dictionary of input tensors, where keys
                    are input names and values are tensors of shape
                    (batch_size, sequence_length, input_dimension).
                context (torch.Tensor, optional): Context tensor of shape
                    (batch_size, context_size). Defaults to None.

            Returns:
                torch.Tensor: Output tensor of shape
                    (batch_size, sequence_length, hidden_dimensions).
    """

    def __init__(
        self,
        inputs: Dict[str, int],
        hidden_dimensions: int,
        dropout: float,
        context_size: int,
    ) -> None:
        super(VariableSelectionNetwork, self).__init__()

        self.inputs_length = len(inputs)

        # Loop over all inputs, create a GRN per input
        self.input_grns = nn.ModuleDict()
        self.input_size_total = 0
        for key, input_dim in inputs.items():
            self.input_size_total += input_dim
            # Equation (7)
            self.input_grns[key] = GatedResidualNetwork(
                input_size=input_dim,
                hidden_size=hidden_dimensions,
                output_size=hidden_dimensions,
                dropout=dropout,
                context_size=context_size,
            )

            # The GRN for the softmax, Equation (6)
        self.grn_input = GatedResidualNetwork(
            input_size=self.input_size_total,
            hidden_size=hidden_dimensions,
            output_size=hidden_dimensions,
            dropout=dropout,
            context_size=context_size,
        )

        # Equation (6)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: Dict[str, torch.Tensor], context: Optional[torch.Tensor] = None
    ) -> Any:
        # x: dictionary with tensors of different dimensionality
        # x[input]: Tensor => [batch size, sequence length, input feature dimension]
        # Context: Tensor => [batch size, context size]
        # Context size is linearly projected to hidden dimensionality, and expanded to the sequence length

        transformed_values_output = []
        values_output = []
        for key, grn in self.input_grns.items():
            # Equation (7)
            transformed_values_output.append(grn(x[key], context))
            values_output.append(x[key])

        # Create the XI matrix
        transformed_inputs = torch.stack(transformed_values_output, dim=-2)
        XI_embedding = torch.cat(values_output, dim=-1)

        # Equation (6)
        v_chi = self.softmax(self.grn_input(XI_embedding, context)).unsqueeze(-2)

        # Equation (8)
        output = transformed_inputs * v_chi
        # output => [batch size, sequence length, inputs, hidden]
        result = output.sum(dim=-2)  # sum over the input dimensions
        # result => [batch size, sequence length, hidden]
        return result
