from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import LSTM

from fintorch.models.timeseries.tft.GatedResidualNetwork import (
    GatedAddNorm,
    GatedResidualNetwork,
)
from fintorch.models.timeseries.tft.InterpretableMultiHeadAttention import (
    InterpretableMultiHeadAttention,
)
from fintorch.models.timeseries.tft.utils import attention_mask
from fintorch.models.timeseries.tft.VariableSelectionNetwork import (
    VariableSelectionNetwork,
)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) model.

    This class implements the Temporal Fusion Transformer model, which is a
    deep learning model for time series forecasting. It combines several
    advanced techniques, including variable selection networks, gated residual
    networks, and multi-head attention, to achieve state-of-the-art performance
    on a variety of time series forecasting tasks.

    Args:
        number_of_past_inputs (int): The number of past time steps to consider.
        horizon (int): The forecasting horizon (number of future time steps to predict).
        embedding_size_inputs (int): The dimensionality of the input embeddings.
        hidden_dimension (int): The dimensionality of the hidden layers.
        dropout (float): Dropout rate to apply to the input tensor.
        number_of_heads (int): The number of attention heads.
        past_inputs (Dict[str, int]): A dictionary mapping past input feature names to their dimensions.
        future_inputs (Dict[str, int]): A dictionary mapping future input feature names to their dimensions.
        static_inputs (Dict[str, int]): A dictionary mapping static input feature names to their dimensions.
        batch_size (int): The batch size.
        device (str): The device to use for computation (e.g., "cpu" or "cuda").
        quantiles (list[float]): List of quantiles to predict.

    Attributes:
        number_of_past_inputs (int): The number of past time steps to consider.
        horizon (int): The forecasting horizon.
        embedding_size_inputs (int): The dimensionality of the input embeddings.
        past_inputs (Dict[str, int]): A dictionary mapping past input feature names to their dimensions.
        future_inputs (Dict[str, int]): A dictionary mapping future input feature names to their dimensions.
        static_inputs (Dict[str, int]): A dictionary mapping static input feature names to their dimensions.
        batch_size (int): The batch size.
        quantiles (list[float]): List of quantiles to predict.
        number_of_quantiles (int): The number of quantiles to predict.
        device (str): The device to use for computation.
        number_of_targets (int): The number of targets to predict.
    """

    def __init__(
        self,
        number_of_past_inputs: int,
        horizon: int,
        embedding_size_inputs: int,
        hidden_dimension: int,
        dropout: float,
        number_of_heads: int,
        past_inputs: Dict[str, int],
        future_inputs: Dict[str, int],
        static_inputs: Dict[str, int],
        batch_size: int,
        device: str,
        quantiles: list[float],
    ) -> None:
        super(TemporalFusionTransformer, self).__init__()
        self.number_of_past_inputs = number_of_past_inputs
        self.horizon = horizon
        self.embedding_size_inputs = embedding_size_inputs
        self.past_inputs = past_inputs
        self.future_inputs = future_inputs
        self.static_inputs = static_inputs
        self.batch_size = batch_size
        self.quantiles = quantiles
        self.number_of_quantiles = len(quantiles)

        context_size = hidden_dimension
        self.device = device

        # Hyperparameters
        # TODO: make this a parameter of the TemporalFusionTransformer class to increase the flexiblility
        self.number_of_targets = number_of_target = 1
        lstm_layer = 1

        # Variable Selection Networks for each branch
        self.variable_selection_past = VariableSelectionNetwork(
            inputs=past_inputs,
            hidden_dimensions=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )

        if future_inputs is not None:
            self.variable_selection_future = VariableSelectionNetwork(
                inputs=future_inputs,
                hidden_dimensions=hidden_dimension,
                dropout=dropout,
                context_size=context_size,
            )
        if static_inputs is not None:
            self.variable_selection_static = VariableSelectionNetwork(
                inputs=static_inputs,
                hidden_dimensions=hidden_dimension,
                dropout=dropout,
                context_size=context_size,
            )

        # LSTM Encoder & Decoder
        self.lstm_encoder = LSTM(
            hidden_dimension,
            hidden_dimension,
            num_layers=lstm_layer,
            batch_first=True,
        )
        self.lstm_decoder = LSTM(
            hidden_dimension,
            hidden_dimension,
            num_layers=lstm_layer,
            batch_first=True,
        )

        # Gated AddNorm (skip connections)
        self.gated_add_norm = GatedAddNorm(
            input_dimension=hidden_dimension,
            hidden_dimensions=hidden_dimension,
            output_dimension=hidden_dimension,
            skip_dimension=hidden_dimension,
            dropout=dropout,
        )

        # Static covariate encoders (available if needed)
        self.lstm_cell = GatedResidualNetwork(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            output_size=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )
        self.lstm_hidden = GatedResidualNetwork(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            output_size=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )

        # Static Enrichment GRN
        self.context_static_enrichment = GatedResidualNetwork(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            output_size=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )

        # Temporal Self-Attention
        self.attention = InterpretableMultiHeadAttention(
            number_of_heads=number_of_heads,
            input_dimension=hidden_dimension,
            dropout=dropout,
        )

        # Position-wise Feed-forward GRN
        self.positionwise_grn = GatedResidualNetwork(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            output_size=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )
        # Dense layer for final output (single target)
        self.dense = nn.Linear(
            hidden_dimension, number_of_target * self.number_of_quantiles
        )

        # Static enrichment GRN
        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            output_size=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )

    def _init_lstm_states(
        self, static_inputs: torch.Tensor, device: str, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if static_inputs is None:
            h0 = torch.zeros(1, batch_size, self.embedding_size_inputs, device=device)
            c0 = torch.zeros(1, batch_size, self.embedding_size_inputs, device=device)
        else:
            h0 = static_inputs.unsqueeze(0)
            c0 = static_inputs.unsqueeze(0)
        return h0, c0

    def _post_process(
        self, attn_input: torch.Tensor, lstm_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask(
            past_length=self.number_of_past_inputs,
            future_length=self.horizon,
        )

        attention_output, attention_weights = self.attention(
            q=attn_input, k=attn_input, v=attn_input, mask=mask
        )

        # Focus on the future horizon outputs
        attention_output = attention_output[:, -self.horizon :, :]
        attn_input_slice = attn_input[:, -self.horizon :, :]
        lstm_output_slice = lstm_output[:, -self.horizon :, :]

        # Skip connection and feed-forward processing
        attention_output = self.gated_add_norm(attention_output, attn_input_slice)
        output_pos_ff = self.positionwise_grn(attention_output)
        output_pos_ff = self.gated_add_norm(output_pos_ff, lstm_output_slice)
        output = self.dense(output_pos_ff)
        # TODO: change when we want to support multiple targets
        output = output.reshape(
            output.shape[0],
            output.shape[1],
            self.number_of_targets,
            self.number_of_quantiles,
        )  # Reshape to (batch, horizon, 1, number_of_quantiles)
        return output, attention_weights

    def forward(
        self,
        past_inputs: Dict[str, torch.Tensor],
        future_inputs: Optional[Dict[str, torch.Tensor]] = None,
        static_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = past_inputs.get(list(past_inputs.keys())[0]).shape[0]  # type: ignore

        # Embed variables using variable selection networks
        past = self.variable_selection_past(past_inputs)
        future = (
            self.variable_selection_future(future_inputs)
            if future_inputs is not None
            else None
        )
        static = (
            self.variable_selection_static(static_inputs)
            if static_inputs is not None
            else None
        )

        h0, c0 = self._init_lstm_states(static, self.device, batch_size)
        encoder_output, (hidden, cell) = self.lstm_encoder(past, hx=(h0, c0))
        encoder_output = self.gated_add_norm(encoder_output, past)

        if future is not None:
            decoder_output, _ = self.lstm_decoder(future, hx=(hidden, cell))
            decoder_output = self.gated_add_norm(decoder_output, future)
            lstm_concat = torch.cat([encoder_output, decoder_output], dim=1)
            inputs_concat = torch.cat([past, future], dim=1)
        else:
            lstm_concat = encoder_output
            inputs_concat = past

        lstm_output = self.gated_add_norm(lstm_concat, inputs_concat)
        attn_input = self.static_enrichment_grn(lstm_output, context=static)
        return self._post_process(attn_input, lstm_output)
