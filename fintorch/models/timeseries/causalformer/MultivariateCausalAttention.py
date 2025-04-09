import torch
import torch.nn as nn
from fintorch.models.timeseries.causalformer.CausalConvolution import CausalConvolution
from fintorch.models.timeseries.causalformer.MultiHeadAttention import (
    MultiHeadAttention,
)


class MultivariateCausalAttention(nn.Module):
    def __init__(
        self,
        number_of_heads: int,
        number_of_series: int,
        length_input_window: int,
        embedding_size: int,
        feature_dimensionality: int,
        tau: float,
        dropout: float,
    ) -> None:
        super().__init__()

        # projection of embedding to Q
        self.Q_proj = nn.Linear(
            in_features=embedding_size,
            out_features=embedding_size,
            bias=False,
        )

        # projection of embedding to K
        self.K_proj = nn.Linear(
            in_features=embedding_size,
            out_features=embedding_size,
            bias=False,
        )

        # Causal convolution for V, use "raw" inputs
        self.V_proj = CausalConvolution(
            number_of_series=number_of_series,
            length_input_window=length_input_window,
            number_of_heads=number_of_heads,
        )

        self.attention = MultiHeadAttention(
            number_of_heads=number_of_heads,
            number_of_series=number_of_series,
            length_input_window=length_input_window,
            embedding_size=embedding_size,
            tau=tau,
        )

        self.tensor_head_dimensionality = embedding_size // number_of_heads
        self.number_of_heads = number_of_heads
        self.number_of_series = number_of_series
        self.input_window = length_input_window
        self.feature_dimensionality = feature_dimensionality
        self.concat_proj = nn.Linear(
            in_features=self.number_of_heads * self.feature_dimensionality,
            out_features=self.feature_dimensionality,
            bias=False,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # q,k: (batch_size, number_of_series, hidden)
        # v: (batch_size, number_of_heads, number_of_series, length_input_window, feature_dimensionality)

        # Q, K: (batch_size, number_of_series, hidden_dimensionality)
        Q = self.Q_proj(q)
        K = self.K_proj(k)
        # V: (batch_size, number_of_heads, number_of_series, series_num, input_window, feature_dimensionality)
        V = self.V_proj(v)

        # split into number of heads (add number of heads dimensionality)
        # Q, K (batch_size, number_of_series, hidden_dimensionality)
        # After view: (batch_size, number_of_series, number_of_heads, tensor_head_dimensionality)
        # transpose: (batch_size, number_of_heads, number_of_series, tensor_head_dimensionality)
        Q = Q.view(
            Q.shape[0],
            self.number_of_series,
            self.number_of_heads,
            self.tensor_head_dimensionality,
        ).transpose(1, 2)
        K = K.view(
            K.shape[0],
            self.number_of_series,
            self.number_of_heads,
            self.tensor_head_dimensionality,
        ).transpose(1, 2)

        # Apply attention
        output = self.attention(Q, K, V, mask)
        # output: (batch_size, number_of_heads, number_of_series, length_input_window, feature_dimensionality)

        # Concatenate
        # each head has attention weighted output features, we want to concatenate these features
        # and project them into the original feature dimension space

        # Step 1: Here we concatenate the number of series and input window per batch, number of heads.
        # Therefore, all individual time-series become one long time-series
        # number_of_series * length_input_window = one long timeseries
        batch_size = Q.shape[0]
        output = output.reshape(
            -1,
            self.number_of_heads,
            self.number_of_series * self.input_window,
            self.feature_dimensionality,
        )
        # We swap the number of heads and the (self.number_of_series * self.input_window) column
        # output: (batch_size, one long timeseries, number_of_heads, feature_dimensionality)
        output = output.permute(0, 2, 1, 3).contiguous()

        # Here we concatenate the features for each head into a single vector per batch, one long timeseries
        # head_concat_features = number_of_heads * feature_dimensionality
        output = output.view(
            batch_size,
            self.number_of_series * self.input_window,
            self.number_of_heads * self.feature_dimensionality,
        )
        # output: (batch_size, one long timeseries, head_concat_features)

        # Next, we split the one long timeseries into its components number_of_series * length_input_window
        # and we keep the concatenated feature dimension
        output = output.reshape(
            -1,
            self.number_of_series,
            self.input_window,
            self.number_of_heads * self.feature_dimensionality,
        )

        # linear projection
        # output: (batch_size, number_of_series, length_input_window, self.n_heads * feature_dimensionality)
        output = self.concat_proj(output)
        # output: (batch_size, number_of_series, length_input_window, feature_dimensionality)

        return output  # type: ignore
