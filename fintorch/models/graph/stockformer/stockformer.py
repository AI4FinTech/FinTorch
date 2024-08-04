import lightning as L
import torch
import torch.nn as nn

from fintorch.layers.attention import ActivationType, TransformerEncoder
from fintorch.layers.DCNN import DilatedConvolution


class temporalEmbedding(nn.Module):
    def __init__(self, D):
        super(self).__init__()
        self.linear_temporal_embedding_layer = nn.Linear()  # TODO: fix

    def forward(self, temporal_encoding):
        return self.linear_temporal_embedding_layer(temporal_encoding)


class dualEncoder(nn.Module):
    def __init__(self, features, **kwargs):
        super(dualEncoder, self).__init__()
        self.dialated_conv = DilatedConvolution(3, features, features, 1, 1)
        self.temporalAttention = TransformerEncoder(
            number_of_encoder_blocks=1, activation=ActivationType.SOFTMAX, **kwargs
        )

        self.sparseAttentionTrend = TransformerEncoder(
            number_of_encoder_blocks=1, activation=ActivationType.SPARSEMAX, **kwargs
        )
        self.sparseAttentionSeasonal = TransformerEncoder(
            number_of_encoder_blocks=1, activation=ActivationType.SPARSEMAX, **kwargs
        )

    def forward(self, trend_signal, seasonal_signal, temporal_emb, spatial_graph_emb):
        trend_signal_ = trend_signal + temporal_emb
        trend_signal = self.temporalAttention(trend_signal_)
        seasonal_signal = self.dialated_conv(seasonal_signal)

        trend_signal += spatial_graph_emb
        seasonal_signal += spatial_graph_emb
        sparse_attention_trend = self.sparseAttentionTrend(trend_signal)
        sparse_attention_seasonal = self.sparseAttentionSeasonal(seasonal_signal)
        trend_signal = sparse_attention_trend + trend_signal
        seasonal_signal = sparse_attention_seasonal + seasonal_signal

        return trend_signal, seasonal_signal


class Stockformer(nn.Module):
    def __init__(self, **kwargs):
        super(Stockformer, self).__init__()
        self.trend_projection = nn.Linear()  # TODO
        self.seasonal_projection = nn.Linear()  # TODO:
        self.temporal_projection = temporalEmbedding()

        self.dual_encoder = nn.ModuleList([dualEncoder(**kwargs) for i in range(L)])

        self.adaptive_fusion = TransformerEncoder(
            number_of_encoder_blocks=1, activation=ActivationType.SOFTMAX, **kwargs
        )

        self.conv_trend = nn.Conv2d(...(1, 1))
        self.conv_seasonal = nn.Conv2d(...(1, 1))

        self.fusion_projection = nn.Linear()  # TODO
        self.final_projection_trend = nn.Linear()  # TODO

    def forward(self, trend, seasonal, temporal_one_hot, spatial_graph_emb):
        trend = self.trend_projection(trend)
        seasonal = self.seasonal_projection(seasonal)
        temporal_embedding = self.temporal_projection(temporal_one_hot)

        for enc in self.dual_encoder:
            trend, seasonal = enc(
                trend, seasonal, temporal_embedding, spatial_graph_emb
            )

        hat_y_trend = self.conv_trend(trend) + temporal_embedding
        hat_y_seasonal = self.conv_seasonal(seasonal) + temporal_embedding

        hat_y = self.adaptive_fusion(hat_y_trend, hat_y_seasonal)
        hat_y = self.fusion_projection(hat_y)

        hat_y_trend = self.final_projection_trend(hat_y_trend)

        return hat_y, hat_y_trend


class StockFormerModule(L.LightningModule):
    def __init__(self, learning_rate: float = 0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = Stockformer(
            10,
            10,
            1,
            2,
            2,
            2,
            2,
            2,
            dim_input=10,
            dim_embedding=10,
            number_of_heads=3,
            dim_feedforward=3,
        )
        self.optimizers = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def forward(self, batch, edge):
        pass

    def loss(self, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        print("Forward passs... go go go!!!")
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return self.optimizers
