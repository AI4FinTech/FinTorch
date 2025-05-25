from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch

from fintorch.models.timeseries.tft import TemporalFusionTransformer


class TemporalFusionTransformerModule(L.LightningModule):
    """
    Temporal Fusion Transformer (TFT) Lightning Module.

    This class implements the Temporal Fusion Transformer model as a
    LightningModule, which is a deep learning model for time series
    forecasting. It combines several advanced techniques, including variable
    selection networks, gated residual networks, and multi-head attention,
    to achieve state-of-the-art performance on a variety of time series
    forecasting tasks.

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
        series_selection_method (str): Method for handling multi-series data. Options:
            - "first": Use only the first series (default, backward compatible)
            - "last": Use only the last series
            - "index": Use series at specific index (requires series_index)
            - "aggregate": Aggregate across all series using series_aggregation method
            - "flatten": Flatten all series into additional features
        series_index (Optional[int]): Index of series to use when series_selection_method="index".
        series_aggregation (str): Aggregation method when series_selection_method="aggregate".
            Options: "mean", "sum", "max", "min" (default: "mean").

    Examples:
        # Use first series only (default behavior)
        model = TemporalFusionTransformerModule(
            ..., series_selection_method="first"
        )

        # Use third series (index 2)
        model = TemporalFusionTransformerModule(
            ..., series_selection_method="index", series_index=2
        )

        # Average all series together
        model = TemporalFusionTransformerModule(
            ..., series_selection_method="aggregate", series_aggregation="mean"
        )

        # Flatten all series as additional features
        model = TemporalFusionTransformerModule(
            ..., series_selection_method="flatten"
        )

    Attributes:
        tft_model (TemporalFusionTransformer): The underlying TFT model.

    Methods:
        _prepare_data(batch):
            Prepares the data from SimpleSyntheticDataset for use with the TFT model.
        forward(past_inputs, future_inputs, static_inputs):
            Computes the forward pass of the TFT model.
        quantile_loss(model_output, target):
            Computes the quantile loss between the model output and the target.
        training_step(batch, batch_idx):
            Performs a single training step.
        validation_step(batch, batch_idx):
            Performs a single validation step.
        test_step(batch, batch_idx):
            Performs a single test step.
        configure_optimizers():
            Configures the optimizer for training.
        predict_step(batch, batch_idx, dataloader_idx=0):
            Performs a single prediction step.

    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. "Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting." arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

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
        quantiles: List[float] = [0.1, 0.5, 0.9],
        series_selection_method: str = "first",
        series_index: Optional[int] = None,
        series_aggregation: str = "mean",
    ):
        super().__init__()
        self.save_hyperparameters()
        assert number_of_past_inputs > horizon, (
            "number_of_past_inputs must be larger than horizon"
        )

        # Multi-series handling configuration
        self.series_selection_method = series_selection_method
        self.series_index = series_index
        self.series_aggregation = series_aggregation

        # Validate series selection parameters
        valid_methods = ["first", "last", "index", "aggregate", "flatten"]
        if series_selection_method not in valid_methods:
            raise ValueError(f"series_selection_method must be one of {valid_methods}")

        if series_selection_method == "index" and series_index is None:
            raise ValueError(
                "series_index must be provided when using 'index' selection method"
            )

        valid_aggregations = ["mean", "sum", "max", "min"]
        if series_aggregation not in valid_aggregations:
            raise ValueError(f"series_aggregation must be one of {valid_aggregations}")

        self.tft_model = TemporalFusionTransformer(
            number_of_past_inputs,
            horizon,
            embedding_size_inputs,
            hidden_dimension,
            dropout,
            number_of_heads,
            past_inputs,
            future_inputs,
            static_inputs,
            batch_size,
            device,
            quantiles,
        )

    def forward(
        self,
        past_inputs: Dict[str, int],
        future_inputs: Dict[str, int],
        static_inputs: Dict[str, int],
    ) -> Any:
        return self.tft_model(past_inputs, future_inputs, static_inputs)

    def quantile_loss(
        self, model_output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target = target.unsqueeze(-1)

        # Check for correct shapes
        assert len(model_output.shape) == 4, (
            f"Model output shape incorrect: {model_output.shape}"
        )
        assert len(target.shape) == 3, f"Target shape incorrect: {target.shape}"
        assert model_output.shape[:2] == target.shape[:2], (
            f"Mismatch between predicted: {model_output.shape} and target shape:{target.shape}"
        )
        assert model_output.shape[3] == self.tft_model.number_of_quantiles, (
            "Mismatch between number of predicted quantiles and target quantiles"
        )

        dim_q = 3  # quantile dimension is the third dimension by definition
        device = model_output.device
        errors = target.unsqueeze(-1) - model_output
        quantiles_tensor = torch.tensor(self.tft_model.quantiles).to(device)
        losses = torch.max(
            (quantiles_tensor - 1) * errors, quantiles_tensor * errors
        ).sum(dim=dim_q)
        return losses.mean()

    def _unpack_batch(
        self,
        batch: Union[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                torch.Tensor,
            ],
        ],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
        Optional[Dict[str, torch.Tensor]],
        torch.Tensor,
    ]:
        """
        Unpacks the batch.
        Supports two formats:
         - Format 1 (4-tuple): (past_inputs, future_inputs, static_inputs, target)
         - Format 2 (2-tuple): (inputs, target) where inputs can be either:
              a) a tuple of (past_inputs, future_inputs, static_inputs), or
              b) a single tensor (only past_inputs)
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                past_inputs, future_inputs, static_inputs, target = batch
            elif len(batch) == 2:
                inputs, target = batch  # type: ignore[assignment]
                if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
                    past_inputs, future_inputs, static_inputs = inputs
                else:
                    # Assume only past_inputs are provided as a dict.
                    past_inputs = inputs  # type: ignore[assignment]
                    future_inputs, static_inputs = None, None
            else:  # type: ignore[unreachable]
                raise ValueError(
                    f"Unexpected batch format: expected 2 or 4 items, got {len(batch)}"
                )
        else:
            raise ValueError("Batch must be a tuple or list")
        return past_inputs, future_inputs, static_inputs, target

    def _prepare_data(
        self,
        batch: Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            torch.Tensor,
        ],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """
        Prepares the data from SimpleSyntheticDataset for use with the TFT model.

        The modified SimpleSyntheticDataset returns data with shape:
        - past_inputs["past_data"]: [batch_size, past_length, num_series, features_dim]
        - future_inputs["future_data"]: [batch_size, future_length, num_series, features_dim]
        - static_inputs["static_data"]: [batch_size, static_length]
        - target: [batch_size, future_length, num_series, features_dim]

        TFT expects:
        - past_inputs["past_data"]: [batch_size, past_length, features_dim]
        - future_inputs["future_data"]: [batch_size, future_length, features_dim]
        - static_inputs["static_data"]: [batch_size, static_length]
        - target: [batch_size, future_length]

        The method now supports multiple strategies for handling multi-series data based on
        the series_selection_method configuration.

        Args:
            batch: A tuple containing past_inputs, future_inputs, static_inputs, and target.

        Returns:
            Tuple of processed past_inputs, future_inputs, static_inputs, and target.
        """
        past_inputs, future_inputs, static_inputs, target = batch

        # Handle past_inputs
        if "past_data" in past_inputs:
            past_data = past_inputs["past_data"]
            if (
                past_data.ndim == 4
            ):  # [batch_size, past_length, num_series, features_dim]
                past_data = self._process_multi_series_data(past_data)
            past_inputs["past_data"] = past_data

        # Handle future_inputs
        if "future_data" in future_inputs:
            future_data = future_inputs["future_data"]
            if (
                future_data.ndim == 4
            ):  # [batch_size, future_length, num_series, features_dim]
                future_data = self._process_multi_series_data(future_data)
            future_inputs["future_data"] = future_data

        # Handle target
        target = self._process_multi_series_target(target)

        return past_inputs, future_inputs, static_inputs, target

    def _process_multi_series_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process multi-series data based on the configured selection method.

        Args:
            data: Input tensor of shape [batch_size, time_length, num_series, features_dim]

        Returns:
            Processed tensor of shape [batch_size, time_length, output_features_dim]
        """
        if self.series_selection_method == "first":
            return data[:, :, 0, :]
        elif self.series_selection_method == "last":
            return data[:, :, -1, :]
        elif self.series_selection_method == "index":
            if self.series_index is None:
                raise ValueError(
                    "series_index cannot be None when using 'index' selection method"
                )
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

    def _process_multi_series_target(self, target: torch.Tensor) -> torch.Tensor:
        """
        Process multi-series target data based on the configured selection method.

        Args:
            target: Target tensor of varying dimensions

        Returns:
            Processed tensor of shape [batch_size, future_length]
        """
        if target.ndim == 4:  # [batch_size, future_length, num_series, features_dim]
            if self.series_selection_method == "first":
                return target[:, :, 0, 0]
            elif self.series_selection_method == "last":
                return target[:, :, -1, 0]
            elif self.series_selection_method == "index":
                if self.series_index is None:
                    raise ValueError(
                        "series_index cannot be None when using 'index' selection method"
                    )
                if self.series_index >= target.shape[2]:
                    raise IndexError(
                        f"series_index {self.series_index} out of range for {target.shape[2]} series"
                    )
                return target[:, :, self.series_index, 0]
            elif self.series_selection_method == "aggregate":
                # Aggregate across series, take first feature
                target_series = target[
                    :, :, :, 0
                ]  # [batch_size, future_length, num_series]
                if self.series_aggregation == "mean":
                    return target_series.mean(dim=2)
                elif self.series_aggregation == "sum":
                    return target_series.sum(dim=2)
                elif self.series_aggregation == "max":
                    return target_series.max(dim=2)[0]
                elif self.series_aggregation == "min":
                    return target_series.min(dim=2)[0]
            elif self.series_selection_method == "flatten":
                # For target, we still need to return [batch_size, future_length]
                # so we aggregate across series and features
                return target.mean(dim=(2, 3))

        elif target.ndim == 3:  # [batch_size, future_length, num_series]
            if self.series_selection_method == "first":
                return target[:, :, 0]
            elif self.series_selection_method == "last":
                return target[:, :, -1]
            elif self.series_selection_method == "index":
                if self.series_index is None:
                    raise ValueError(
                        "series_index cannot be None when using 'index' selection method"
                    )
                if self.series_index >= target.shape[2]:
                    raise IndexError(
                        f"series_index {self.series_index} out of range for {target.shape[2]} series"
                    )
                return target[:, :, self.series_index]
            elif self.series_selection_method == "aggregate":
                if self.series_aggregation == "mean":
                    return target.mean(dim=2)
                elif self.series_aggregation == "sum":
                    return target.sum(dim=2)
                elif self.series_aggregation == "max":
                    return target.max(dim=2)[0]
                elif self.series_aggregation == "min":
                    return target.min(dim=2)[0]
            elif self.series_selection_method == "flatten":
                return target.mean(dim=2)

        # If target is already 2D or has unexpected dimensions, return as-is
        return target

    def training_step(
        self,
        batch: Union[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                torch.Tensor,
            ],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        # Prepare data from SimpleSyntheticDataset if needed
        if isinstance(batch, tuple) and len(batch) == 4:
            batch = self._prepare_data(batch)

        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)

        output, _ = self.forward(past_inputs, future_inputs, static_inputs)  # type: ignore

        # Calculate the loss
        loss = self.quantile_loss(output, target)

        # Log the loss
        self.log("train_loss_epoch", loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(
        self,
        batch: Union[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                torch.Tensor,
            ],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        # Prepare data from SimpleSyntheticDataset if needed
        if isinstance(batch, tuple) and len(batch) == 4:
            batch = self._prepare_data(batch)

        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)  # type: ignore

        output, _ = self.forward(past_inputs, future_inputs, static_inputs)  # type: ignore

        # Calculate the loss
        loss = self.quantile_loss(output, target)

        # Log the loss
        self.log("val_loss_epoch", loss, on_epoch=True, on_step=False)
        return loss

    def test_step(
        self,
        batch: Union[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                torch.Tensor,
            ],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        # Prepare data from SimpleSyntheticDataset if needed
        if isinstance(batch, tuple) and len(batch) == 4:
            batch = self._prepare_data(batch)

        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)
        output, _ = self.forward(past_inputs, future_inputs, static_inputs)  # type: ignore

        # Calculate the loss
        loss = self.quantile_loss(output, target)

        # Log the loss
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(
        self,
        batch: Union[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                torch.Tensor,
            ],
        ],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        # Prepare data from SimpleSyntheticDataset if needed
        if isinstance(batch, tuple) and len(batch) == 4:
            batch = self._prepare_data(batch)

        past_inputs, future_inputs, static_inputs, _ = self._unpack_batch(batch)  # type: ignore
        output, _ = self.forward(past_inputs, future_inputs, static_inputs)  # type: ignore

        return output
