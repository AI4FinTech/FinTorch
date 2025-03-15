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
    ):
        super(TemporalFusionTransformer, self).__init__()
        self.number_of_past_inputs = number_of_past_inputs
        self.number_of_future_inputs = number_of_future_inputs
        self.embedding_size_inputs = embedding_size_inputs

        # TODO: all context is already transformed into a fixed size (hidden dimension)
        context_size = hidden_dimension

        # VSN for each color in Figure (2)
        self.variable_selection_past = VariableSelectionNetwork(
            inputs=past_inputs,
            hidden_dimensions=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )
        self.variable_selection_future = VariableSelectionNetwork(
            inputs=future_inputs,
            hidden_dimensions=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )
        self.variable_selection_static = VariableSelectionNetwork(
            inputs=static_inputs,
            hidden_dimensions=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )

        # LSTM
        self.lstm_encoder = LSTM(
            hidden_dimension,
            hidden_dimension,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            proj_size=0,
            device=None,
            dtype=None,
        )
        self.lstm_decoder = LSTM(
            hidden_dimension,
            hidden_dimension,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            proj_size=0,
            device=None,
            dtype=None,
        )

        # Gate and AddNorm (no trainable parameters. Therefore, initiated only once.)
        self.gated_add_norm = GatedAddNorm(
            input_dimension=hidden_dimension,
            hidden_dimensions=hidden_dimension,
            output_dimension=hidden_dimension,
            skip_dimension=hidden_dimension,
            dropout=dropout,
        )

        # static covariate encoders
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

        # Static Enrichment Layer GRN
        self.context_static_enrichment = GatedResidualNetwork(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            output_size=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )

        # Temporal Self-attention
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
        # Dense layer
        # TODO: we only support a single target
        self.dense = nn.Linear(hidden_dimension, 1)

        # TODO: Investigate whether this is the correct way to transform the features into embedding input
        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=hidden_dimension,
            hidden_size=hidden_dimension,
            output_size=hidden_dimension,
            dropout=dropout,
            context_size=context_size,
        )

    def forward(self, past_inputs, future_inputs, static_inputs):

        # Set attention masks
        # TODO: calculate attention masks

        # embed variables
        past_inputs = self.variable_selection_past(past_inputs)  # from dim to dim
        future_inputs = self.variable_selection_future(future_inputs)
        static_inputs = self.variable_selection_static(static_inputs)

        # LSTM: initial values based on context
        lstm_hidden_value = static_inputs.unsqueeze(0)  # self.grn_new(static_inputs)
        lstm_cell_value = static_inputs.unsqueeze(0)  # self.grn_new2(static_inputs)

        # print(f"lstm hidden shape: {lstm_hidden_value.shape}")
        # print(f"lstm cell shape: {lstm_cell_value.shape}")
        # print(f"past inputs shape: {past_inputs.shape}")

        # Run LSTM Encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            input=past_inputs, hx=(lstm_hidden_value, lstm_cell_value)
        )
        # Skip AddNorm
        encoder_output = self.gated_add_norm(encoder_output, past_inputs)

        # Run LSTM Decoder
        decoder_output, _ = self.lstm_decoder(input=future_inputs, hx=(hidden, cell))
        # Skip AddNorm
        decoder_output = self.gated_add_norm(decoder_output, future_inputs)

        # Concatenate output of lstm into a single layer in order to pass it to the static enrichment layer
        lstm_concatenate = torch.cat([encoder_output, decoder_output], dim=1)
        inputs_concatenated = torch.cat([past_inputs, future_inputs], dim=1)

        lstm_output = self.gated_add_norm(x=lstm_concatenate, skip=inputs_concatenated)

        # static enrichment
        self.context_static_enrichment(
            decoder_output,
        )
        # TODO: I don't understand this layer?
        attn_input = self.static_enrichment_grn(lstm_output, context=static_inputs)

        attention_output, attention_weights = self.attention(
            q=attn_input, k=attn_input, v=attn_input, mask=None
        )

        attention_output = attention_output[:, -self.number_of_future_inputs :, :]
        attn_input = attn_input[:, -self.number_of_future_inputs :, :]
        lstm_output = lstm_output[:, -self.number_of_future_inputs :, :]

        # skip over attention
        attention_output = self.gated_add_norm(x=attention_output, skip=attn_input)

        # psotition-wise feed forward
        output_pos_ff = self.positionwise_grn(attention_output)

        # skip over Temporal Fusion Decoder
        output_pos_ff = self.gated_add_norm(x=output_pos_ff, skip=lstm_output)

        # Dense
        output = self.dense(output_pos_ff)
        # TODO: it seems that this doesn't match, why do we have [batch size x 2 times squence length x # quantiles] as output

        return output, attention_weights
