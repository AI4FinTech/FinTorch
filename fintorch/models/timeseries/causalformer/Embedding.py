import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        number_of_series: int,
        length_input_window: int,
        feature_dimensionality: int,
        hidden_dimensionality: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.number_of_series = number_of_series
        self.length_input_window = length_input_window
        self.feature_dimensionality = feature_dimensionality
        self.hidden_dimensionality = hidden_dimensionality
        self.dropout = dropout
        self.d_model = hidden_dimensionality

        # A liinear projection layer to generate embeddingd for input time-series.
        # We project from R^{N x T} -> R^{N x d} with d as the dimensionality

        # Linear projection layer
        self.embedding = nn.Linear(
            in_features=self.length_input_window * self.feature_dimensionality,
            out_features=self.d_model,
            bias=True,
        )
        self.normalization = nn.LayerNorm(self.d_model)
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        # x: (batch_size, number_of_series, length_input_window, feature_dimensionality)
        batch_size = x.shape[0]
        x = x.view(
            batch_size,
            self.number_of_series,
            self.length_input_window * self.feature_dimensionality,
        )
        # Apply the linear projection
        x = self.embedding(x)
        # Apply layer normalization
        x = self.normalization(x)
        # Apply dropout
        x = self.dropout(x)
        return x
