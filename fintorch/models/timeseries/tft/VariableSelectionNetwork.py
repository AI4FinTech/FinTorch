from typing import Dict

import torch
import torch.nn as nn

from fintorch.models.timeseries.tft.GatedResidualNetwork import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):

    def __init__(
        self,
        inputs: Dict[str, int],
        hidden_dimensions,
        dropout,
        context_size,
    ):
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

    def forward(self, x, context=None):
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
        return output.sum(dim=-2)  # sum over the input dimensions
