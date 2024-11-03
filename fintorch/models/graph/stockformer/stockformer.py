import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralforecast.common._base_multivariate import BaseMultivariate
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models.nbeats import NBEATSBlock, SeasonalityBasis, TrendBasis

from fintorch.layers.attention import ActivationType, TransformerEncoder
from fintorch.layers.DCNN import DilatedConvolution


class dualEncoder(nn.Module):
    def __init__(self, features, **kwargs):
        super().__init__()
        self.dialated_conv = DilatedConvolution(3, 10, 10, 1, 1)
        self.temporalAttention = TransformerEncoder(
            number_of_encoder_blocks=1,
            activation=ActivationType.SOFTMAX,
            dim_input=30,
            dim_embedding=30,
            number_of_heads=1,
            dim_feedforward=10,
        )

        self.sparseAttentionTrend = TransformerEncoder(
            number_of_encoder_blocks=1,
            activation=ActivationType.SPARSEMAX,
            dim_input=30,
            dim_embedding=30,
            number_of_heads=1,
            dim_feedforward=10,
        )
        self.sparseAttentionSeasonal = TransformerEncoder(
            number_of_encoder_blocks=1,
            activation=ActivationType.SPARSEMAX,
            dim_input=30,
            dim_embedding=30,
            number_of_heads=1,
            dim_feedforward=10,
        )

    def forward(self, trend_signal, seasonal_signal, spatial_graph_emb):
        logging.info(f"DualEncoder::Forward trend_signal:{trend_signal.shape}")
        trend_signal, _ = self.temporalAttention(trend_signal)
        logging.info(f"DualEncoder::Forward seasonal_signal:{seasonal_signal.shape}")
        seasonal_signal = self.dialated_conv(seasonal_signal)

        # TODO: add spatial graph encoding of the signal (shoudl be same dimensionality)
        # trend_signal += spatial_graph_emb
        # seasonal_signal += spatial_graph_emb
        sparse_attention_trend, _ = self.sparseAttentionTrend(trend_signal)
        sparse_attention_seasonal, _ = self.sparseAttentionSeasonal(seasonal_signal)

        # TODO: fix skip-connection
        trend_signal = sparse_attention_trend + trend_signal
        seasonal_signal = sparse_attention_seasonal + seasonal_signal
        logging.info("DualEncoder::forward end of forward")
        return trend_signal, seasonal_signal


class Stockformer(nn.Module):
    def __init__(self, dual_layers=1, **kwargs):
        super().__init__()
        self.trend_projection = nn.Linear(1, 30)  # TODO input size
        self.seasonal_projection = nn.Linear(1, 30)  # TODO:

        self.dual_encoder = nn.ModuleList(
            [dualEncoder(**kwargs) for i in range(dual_layers)]
        )

        kwargs.pop("features", None)
        self.adaptive_fusion = TransformerEncoder(
            number_of_encoder_blocks=1,
            activation=ActivationType.SOFTMAX,
            dim_input=30,
            dim_embedding=30,
            number_of_heads=1,
            dim_feedforward=10,
        )

        self.conv_trend = nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
        self.conv_seasonal = nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))

        self.fusion_projection = nn.Linear(30, 4)  # TODO
        self.final_projection_trend = nn.Linear(30, 4)  # TODO

    def forward(self, trend, seasonal, spatial_graph_emb):
        trend = self.trend_projection(trend)
        seasonal = self.seasonal_projection(seasonal)

        for enc in self.dual_encoder:
            trend, seasonal = enc(trend, seasonal, spatial_graph_emb)

        logging.info(f"Stockformer::forward trend:{trend.unsqueeze(1).shape}")
        hat_y_trend = self.conv_trend(trend.unsqueeze(1))
        hat_y_seasonal = self.conv_seasonal(seasonal.unsqueeze(1))

        logging.info(
            f"Stockformer::Forward hat_y_trend:{hat_y_trend.shape} hat_y_seasonal:{hat_y_seasonal.shape}"
        )

        hat_y, _ = self.adaptive_fusion(hat_y_trend.squeeze(1), hat_y_seasonal)
        hat_y = self.fusion_projection(hat_y)

        hat_y_trend = self.final_projection_trend(hat_y_trend)

        return hat_y, hat_y_trend


class LatentCorrelationLayer(nn.Module):

    def __init__(
        self, unit, input_size, leaky_rate, dropout_rate, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.unit = unit
        self.time_step = input_size
        self.alpha = leaky_rate

        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.GRU = nn.GRU(self.time_step, self.unit)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(
            diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat)
        )
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention
        return x

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )
        second_laplacian = laplacian
        third_laplacian = (
            2 * torch.matmul(laplacian, second_laplacian)
        ) - first_laplacian
        forth_laplacian = (
            2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        )
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0
        )
        return multi_order_laplacian

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)


class SimpleNBEATS(nn.Module):

    def __init__(
        self,
        h,
        input_size,
        dropout_prob_theta=0.0,
        n_polynomials=0,  # Lower frequencies can be captured by polynomials
        n_blocks=[1, 1],
        mlp_units=[[64, 64], [64, 64]],
        activation="ReLU",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        stack_types = ["trend", "seasonality"]
        n_harmonics = 2
        self.input_size = input_size
        self.h = h

        outputsize_multiplier = 1

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):
                # Shared weights
                if stack_types[i] == "seasonality":
                    n_theta = (
                        2
                        * (outputsize_multiplier + 1)
                        * int(np.ceil(n_harmonics / 2 * h) - (n_harmonics - 1))
                    )
                    basis = SeasonalityBasis(
                        harmonics=n_harmonics,
                        backcast_size=input_size,
                        forecast_size=h,
                        out_features=outputsize_multiplier,
                    )

                elif stack_types[i] == "trend":
                    n_theta = (outputsize_multiplier + 1) * (n_polynomials + 1)
                    basis = TrendBasis(
                        degree_of_polynomial=n_polynomials,
                        backcast_size=input_size,
                        forecast_size=h,
                        out_features=outputsize_multiplier,
                    )

                nbeats_block = NBEATSBlock(
                    input_size=input_size,
                    n_theta=n_theta,
                    mlp_units=mlp_units,
                    basis=basis,
                    dropout_prob=dropout_prob_theta,
                    activation=activation,
                )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

            self.blocks = nn.ModuleList(block_list)

    def forward(self, windows_batch):
        # Parse windows_batch
        insample_y = windows_batch["insample_y"][:, :, 0]
        insample_mask = windows_batch["insample_mask"][:, :, 0]

        # NBEATS' forward
        residuals = insample_y.flip(dims=(-1,))  # backcast init
        insample_mask = insample_mask.flip(dims=(-1,))

        # NOTE: I guess, we might need the back-cast only
        forecast = insample_y[:, -1:, None]  # Level with Naive1
        block_forecasts = [forecast.repeat(1, self.h, 1)]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, h, out_features)
        block_forecasts = torch.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2, 3)
        block_forecasts = block_forecasts.squeeze(-1)  # univariate output

        return block_forecasts


# NOTE: Extension of BaseMultivariate (lightning extenstion)
class StockFormerModule(BaseMultivariate):
    def __init__(
        self,
        h,
        input_size,
        n_series,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        dropout_rate: float = 0.5,
        leaky_rate: float = 0.2,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        step_size: int = 1,
        scaler_type: str = "robust",
        random_seed: int = 1,
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            batch_size=batch_size,
            loss=loss,
            n_series=4,
            valid_loss=valid_loss,
            learning_rate=learning_rate,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            step_size=step_size,
            **trainer_kwargs,
        )

        self.latent_correlation_layer = LatentCorrelationLayer(
            unit=n_series,
            input_size=input_size,
            leaky_rate=leaky_rate,
            dropout_rate=dropout_rate,
        )

        # NOTE: should I have one per time-series to learn the decomposition?
        self.decouple_flow_layer = SimpleNBEATS(h=h, input_size=input_size)

        self.stockformer = Stockformer(
            features=4,
            dim_input=input_size,
            dim_embedding=10,
            number_of_heads=10,
            dim_feedforward=4,
            dual_layers=1,
        )

        self.random = nn.Linear(n_series, out_features=1)

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        # insample_mask = windows_batch["insample_mask"]
        # Only predict for horizon

        La, att = self.latent_correlation_layer(
            insample_y
        )  # Batch, input_size, n_series => (n_series, n_series, n_series), (n_series, n_series)
        logging.info(f"L:{La.shape}, att:{att.shape}")
        # Concatenate the correlation layer, repeat representation

        # TODO: DFL per-series
        x = self.decouple_flow_layer(
            windows_batch
        )  # batch, values => batch, decomposition layer, values

        logging.info(
            f"Stockformer::forward: Feed to stockformer model x0:{x[:, 0, :].unsqueeze(-1).shape} x2:{x[:, 2, :].unsqueeze(-1).shape} netemb:{att[0, :]}"
        )
        hat_y, hat_y_trend = self.stockformer(
            x[:, 0, :].unsqueeze(-1), x[:, 2, :].unsqueeze(-1), att[0, :]
        )

        logging.info(
            f"hat_y:{hat_y.shape} hat_y_trend:{hat_y_trend.shape} horizon:{self.h}"
        )

        return self.random(insample_y)
