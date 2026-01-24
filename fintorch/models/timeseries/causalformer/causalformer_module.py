from typing import Any, Dict, Optional

import lightning as L
import torch
from fintorch.models.timeseries.causalformer.CausalFormer import CausalFormer
from torch.optim.lr_scheduler import StepLR


class CausalFormerModule(L.LightningModule):
    """
    CausalFormer Lightning Module that works with the new standardized dictionary format.

    This module processes time series data in the new standardized format and converts it
    to the format expected by the CausalFormer core model. CausalFormer is designed to
    handle multiple time series natively by calculating attention between different series
    to discover causal relationships between variables.

    Key Features:
    - Native multi-series processing without forced aggregation
    - Causal attention mechanism to discover relationships between time series
    - Designed for multivariate time series forecasting and causal discovery

    For multi-series data, configure number_of_series to match the actual number of series
    in your data to leverage CausalFormer's native multi-series processing capabilities.

    Note: Unlike TFT, CausalFormer does not require series selection methods as it is
    specifically designed to work with all series simultaneously to discover causal patterns.
    """

    def __init__(
        self,
        number_of_layers: int,
        number_of_heads: int,
        number_of_series: int,
        length_input_window: int,
        length_output_window: int,
        embedding_size: int,
        feature_dimensionality: int,
        ffn_hidden_dimensionality: int,
        output_dimensionality: int,
        tau: float,
        dropout: float,
        learning_rate: float = 0.01,
        lr_step_size: int = 30,
        lr_gamma: float = 0.1,
        weight_decay: float = 0,
        # Optional parameters for handling standardized format (not used in core functionality)
        series_selection_method: Optional[str] = None,
        series_index: Optional[int] = None,
        series_aggregation: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store series handling parameters (optional, for compatibility)
        self.series_selection_method = series_selection_method
        self.series_index = series_index
        self.series_aggregation = series_aggregation

        # Validate series selection parameters only if provided
        if series_selection_method is not None:
            valid_methods = ["first", "last", "index", "aggregate", "flatten"]
            if series_selection_method not in valid_methods:
                raise ValueError(f"series_selection_method must be one of {valid_methods}")

            if series_selection_method == "index" and series_index is None:
                raise ValueError(
                    "series_index must be provided when using 'index' selection method"
                )

        if series_aggregation is not None:
            valid_aggregations = ["mean", "sum", "max", "min"]
            if series_aggregation not in valid_aggregations:
                raise ValueError(f"series_aggregation must be one of {valid_aggregations}")

        self.causalformer = CausalFormer(
            number_of_layers=self.hparams["number_of_layers"],
            number_of_heads=self.hparams["number_of_heads"],
            number_of_series=self.hparams["number_of_series"],
            length_input_window=self.hparams["length_input_window"],
            length_output_window=self.hparams["length_output_window"],
            embedding_size=self.hparams["embedding_size"],
            feature_dimensionality=self.hparams["feature_dimensionality"],
            ffn_hidden_dimensionality=self.hparams["ffn_hidden_dimensionality"],
            output_dimensionality=self.hparams["output_dimensionality"],
            tau=self.hparams["tau"],
            dropout=self.hparams["dropout"],
        )

        self.loss = torch.nn.L1Loss()

    def forward(self, x: torch.Tensor) -> Any:
        return self.causalformer(x)

    def _process_multi_series_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process multi-series data based on the configured selection method.

        Args:
            data: Input tensor of shape [batch_size, time_length, num_series, features_dim]

        Returns:
            Processed tensor of shape [batch_size, time_length, processed_features_dim]
        """
        if data.shape[2] == 1:
            # Already single series, just squeeze the series dimension
            return data.squeeze(2)

        if self.series_selection_method == "first":
            return data[:, :, 0, :]
        elif self.series_selection_method == "last":
            return data[:, :, -1, :]
        elif self.series_selection_method == "index":
            if self.series_index >= data.shape[2]:
                raise IndexError(
                    f"series_index {self.series_index} out of range for {data.shape[2]} series"
                )
            return data[:, :, self.series_index, :]
        elif self.series_selection_method == "aggregate":
            if self.series_aggregation == "mean":
                return data.mean(dim=2)
            elif self.series_aggregation == "sum":
                return data.sum(dim=2)
            elif self.series_aggregation == "max":
                return data.max(dim=2)[0]
            elif self.series_aggregation == "min":
                return data.min(dim=2)[0]
        elif self.series_selection_method == "flatten":
            # Flatten series and features dimensions
            batch_size, time_length, num_series, features_dim = data.shape
            return data.view(batch_size, time_length, num_series * features_dim)

        raise ValueError(
            f"Unknown series_selection_method: {self.series_selection_method}"
        )

    def _prepare_data(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input and target tensors for the CausalFormer model.

        Input tensor transformations:
        - Input from new standardized format: combines past_target, past_covariates_known_future,
          and past_covariates_unknown_future into a single tensor
        - Input for CausalFormer: [batch_size, num_series, past_length, feature_dimensionality]

        Target tensor transformations:
        - Target from new format: [batch_size, future_length, num_series, num_target_features]
        - Target for CausalFormer: [batch_size, num_series, future_length, feature_dim]

        Args:
            batch: Dictionary containing the new standardized format tensors

        Returns:
            Tuple of (input_tensor, target_tensor) in CausalFormer expected format
        """
        # Combine all past features
        past_features = []

        if "past_target" in batch:
            past_features.append(batch["past_target"])

        if "past_covariates_known_future" in batch:
            past_features.append(batch["past_covariates_known_future"])

        if "past_covariates_unknown_future" in batch:
            past_features.append(batch["past_covariates_unknown_future"])

        # Concatenate all past features along the feature dimension
        if past_features:
            x = torch.cat(past_features, dim=-1)  # [batch_size, past_length, series_dim, total_features]
        else:
            raise ValueError("No past features found in batch")

        # Get target
        y = batch["output_target"]  # [batch_size, future_length, series_dim, num_target_features]

        # Process multi-series data if needed
        if x.ndim == 4 and x.shape[2] > 1:  # [batch_size, past_length, series_dim, total_features]
            # Check if we should use native multi-series processing (series_selection_method is None)
            if self.series_selection_method is None:
                # Keep all series for native multi-series processing - no reduction needed
                pass  # x remains [batch_size, past_length, series_dim, total_features]
                # y remains [batch_size, future_length, series_dim, num_target_features]
            else:
                # Use multi-series processing to reduce to single series
                x = self._process_multi_series_data(x)  # [batch_size, past_length, processed_features]

                # For target data, handle flatten method differently
                if self.series_selection_method == "flatten":
                    # For flatten method, don't flatten targets - use first series instead
                    y = y[:, :, 0, :]  # [batch_size, future_length, target_features]
                else:
                    # For other methods, process target data with same method
                    y = self._process_multi_series_data(y)  # [batch_size, future_length, processed_features]

        # Handle input tensor shape adaptively
        # Expected shape for CausalFormer: [batch_size, num_series, past_length, feature_dimensionality]
        if x.ndim == 3:  # [batch_size, past_length, total_features]
            # Add series dimension if it's missing
            x = x.unsqueeze(2)  # [batch_size, past_length, 1, total_features]
            x = x.permute(0, 2, 1, 3)  # [batch_size, 1, past_length, total_features]
        elif x.ndim == 4:  # [batch_size, past_length, series_dim, total_features]
            # Preserve all series and permute to expected format
            x = x.permute(0, 2, 1, 3)  # [batch_size, series_dim, past_length, total_features]

        # Handle target tensor shape - ensure it becomes 4D [batch_size, num_series, future_length, feature_dim]
        if y.ndim == 2:  # [batch_size, future_length]
            # Add both series and feature dimensions
            y = y.unsqueeze(2).unsqueeze(-1)  # [batch_size, future_length, 1, 1]
            y = y.permute(0, 2, 1, 3)  # [batch_size, 1, future_length, 1]
        elif y.ndim == 3:  # [batch_size, future_length, features] - from multi-series processing
            # Add series dimension and feature dimension if needed
            if y.shape[-1] == 1:  # Already has feature dimension
                y = y.unsqueeze(1)  # [batch_size, 1, future_length, 1]
            else:  # Multiple features, add series dimension
                y = y.unsqueeze(1)  # [batch_size, 1, future_length, features]
        elif y.ndim == 4:  # [batch_size, future_length, num_series, feature_dim]
            # Preserve all series and permute to expected format
            y = y.permute(0, 2, 1, 3)  # [batch_size, num_series, future_length, feature_dim]

        return x, y

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        x, y = self._prepare_data(batch)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        x, y = self._prepare_data(batch)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        x, y = self._prepare_data(batch)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        x, _ = self._prepare_data(batch)
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            amsgrad=True,
        )

        scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.hparams["lr_step_size"],
            gamma=self.hparams["lr_gamma"],
        )

        # Return the optimizer and scheduler in the format Lightning expects
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # How often to step the scheduler ('epoch' or 'step')
                "frequency": 1,  # How many intervals pass between steps
            },
        }
