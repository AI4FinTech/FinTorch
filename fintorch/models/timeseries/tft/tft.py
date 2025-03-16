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

    def __init__(
        self,
        number_of_past_inputs,
        number_of_future_inputs,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        past_inputs,
        future_inputs,
        static_inputs,
        batch_size,
        device,
    ):
        super(TemporalFusionTransformer, self).__init__()
        self.number_of_past_inputs = number_of_past_inputs
        self.number_of_future_inputs = number_of_future_inputs
        self.embedding_size_inputs = embedding_size_inputs
        self.past_inputs = past_inputs
        self.future_inputs = future_inputs
        self.static_inputs = static_inputs
        self.batch_size = batch_size

        context_size = hidden_dimension
        self.device = device

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
            num_layers=1,
            batch_first=True,
        )
        self.lstm_decoder = LSTM(
            hidden_dimension,
            hidden_dimension,
            num_layers=1,
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
        self.dense = nn.Linear(hidden_dimension, 1)

        # Static enrichment GRN
        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            output_size=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )

    def _init_lstm_states(self, static_inputs, device, batch_size):
        if static_inputs is None:
            h0 = torch.zeros(1, batch_size, self.embedding_size_inputs, device=device)
            c0 = torch.zeros(1, batch_size, self.embedding_size_inputs, device=device)
        else:
            h0 = static_inputs.unsqueeze(0)
            c0 = static_inputs.unsqueeze(0)
        return h0, c0

    def _post_process(self, attn_input, lstm_output):
        mask = attention_mask(
            past_length=self.number_of_past_inputs,
            future_length=self.number_of_future_inputs,
        )
        mask = None
        attention_output, attention_weights = self.attention(
            q=attn_input, k=attn_input, v=attn_input, mask=mask
        )

        # Focus on the future horizon outputs
        attention_output = attention_output[:, -self.number_of_future_inputs :, :]
        attn_input_slice = attn_input[:, -self.number_of_future_inputs :, :]
        lstm_output_slice = lstm_output[:, -self.number_of_future_inputs :, :]

        # Skip connection and feed-forward processing
        attention_output = self.gated_add_norm(attention_output, attn_input_slice)
        output_pos_ff = self.positionwise_grn(attention_output)
        output_pos_ff = self.gated_add_norm(output_pos_ff, lstm_output_slice)
        output = self.dense(output_pos_ff)
        return output, attention_weights

    def forward(self, past_inputs, future_inputs=None, static_inputs=None):

        batch_size = past_inputs["past_data"].shape[0]

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
