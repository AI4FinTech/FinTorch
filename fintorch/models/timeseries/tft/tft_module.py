from typing import Any, Dict, List, Optional, Union

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
        num_past_target_features (int): Number of past target features.
        num_past_known_cov_features (int): Number of past known covariate features.
        num_past_unknown_cov_features (int): Number of past unknown covariate features.
        num_future_known_cov_features (int): Number of future known covariate features.
        num_static_real_features (int): Number of real-valued static features.
        num_static_categorical_features (int): Number of categorical static features.
        static_categorical_cardinalities (List[int]): List of cardinalities for each categorical feature.
        batch_size (int): The batch size.
        device (str): The device to use for computation (e.g., "cpu" or "cuda").
        quantiles (List[float]): List of quantiles to predict.
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
            Prepares the data from the new standardized format for use with the TFT model.
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
        # New granular feature parameters
        num_past_target_features: int,
        num_past_known_cov_features: int,
        num_past_unknown_cov_features: int,
        num_future_known_cov_features: int,
        num_static_real_features: int,
        num_static_categorical_features: int,
        static_categorical_cardinalities: List[int],
        batch_size: int,
        device: str,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        series_selection_method: str = "first",
        series_index: Optional[int] = None,
        series_aggregation: str = "mean",
        # Legacy parameters for backward compatibility
        past_inputs: Optional[Dict[str, int]] = None,
        future_inputs: Optional[Dict[str, int]] = None,
        static_inputs: Optional[Dict[str, int]] = None,
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

        # Store new feature dimensions
        self.num_past_target_features = num_past_target_features
        self.num_past_known_cov_features = num_past_known_cov_features
        self.num_past_unknown_cov_features = num_past_unknown_cov_features
        self.num_future_known_cov_features = num_future_known_cov_features
        self.num_static_real_features = num_static_real_features
        self.num_static_categorical_features = num_static_categorical_features
        self.static_categorical_cardinalities = static_categorical_cardinalities

        # Convert new format to legacy format for TFT core model
        # This maintains compatibility with the existing TFT implementation
        if past_inputs is None:
            past_inputs = {}
            # Always include past_target - it's required for TFT to work
            if num_past_target_features <= 0:
                raise ValueError("num_past_target_features must be > 0 - TFT requires at least past target data")
            past_inputs["past_target"] = num_past_target_features
            if num_past_known_cov_features > 0:
                past_inputs["past_known_cov"] = num_past_known_cov_features
            if num_past_unknown_cov_features > 0:
                past_inputs["past_unknown_cov"] = num_past_unknown_cov_features

        if future_inputs is None:
            future_inputs = {}
            if num_future_known_cov_features > 0:
                future_inputs["future_known_cov"] = num_future_known_cov_features

        if static_inputs is None:
            static_inputs = {}
            if num_static_real_features > 0:
                static_inputs["static_real"] = num_static_real_features
            if num_static_categorical_features > 0:
                static_inputs["static_categorical"] = num_static_categorical_features

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
        past_inputs: Dict[str, torch.Tensor],
        future_inputs: Dict[str, torch.Tensor],
        static_inputs: Dict[str, torch.Tensor],
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
        batch: Union[Dict[str, torch.Tensor], Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Unpacks the batch from the new standardized format.
        Supports the new single dictionary format.
        """
        if isinstance(batch, dict):
            return batch
        else:
            raise ValueError("Batch must be a dictionary in the new standardized format")

    def _prepare_data(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """
        Prepares the data from the new standardized format for use with the TFT model.

        The new standardized format provides data with shape:
        - past_target: [batch_size, past_time_steps, series_dim, num_target_features]
        - past_covariates_known_future: [batch_size, past_time_steps, series_dim, num_known_cov_features]
        - past_covariates_unknown_future: [batch_size, past_time_steps, series_dim, num_unknown_cov_features]
        - future_covariates_known: [batch_size, future_time_steps, series_dim, num_known_cov_features]
        - output_target: [batch_size, future_time_steps, series_dim, num_target_features]
        - static_features_real: [batch_size, series_dim, num_static_real_features]
        - static_features_categorical: [batch_size, series_dim, num_static_categorical_features]

        TFT expects:
        - past_inputs[key]: [batch_size, past_time_steps, features_dim]
        - future_inputs[key]: [batch_size, future_time_steps, features_dim]
        - static_inputs[key]: [batch_size, features_dim]
        - target: [batch_size, future_time_steps]

        The method supports multiple strategies for handling multi-series data based on
        the series_selection_method configuration.

        Args:
            batch: A dictionary containing the new standardized format tensors.

        Returns:
            Tuple of processed past_inputs, future_inputs, static_inputs, and target.
        """
        # Process past inputs
        past_inputs = {}
        if "past_target" in batch:
            past_target = batch["past_target"]
            if past_target.ndim == 4:  # [batch_size, time_steps, series_dim, features_dim]
                past_target = self._process_multi_series_data(past_target)
            past_inputs["past_target"] = past_target

        if "past_covariates_known_future" in batch:
            past_known_cov = batch["past_covariates_known_future"]
            if past_known_cov.ndim == 4:
                past_known_cov = self._process_multi_series_data(past_known_cov)
            past_inputs["past_known_cov"] = past_known_cov

        if "past_covariates_unknown_future" in batch:
            past_unknown_cov = batch["past_covariates_unknown_future"]
            if past_unknown_cov.ndim == 4:
                past_unknown_cov = self._process_multi_series_data(past_unknown_cov)
            past_inputs["past_unknown_cov"] = past_unknown_cov

        # Process future inputs
        future_inputs = {}
        if "future_covariates_known" in batch:
            future_known_cov = batch["future_covariates_known"]
            if future_known_cov.ndim == 4:
                future_known_cov = self._process_multi_series_data(future_known_cov)
            future_inputs["future_known_cov"] = future_known_cov

        # Process static inputs
        static_inputs = {}
        if "static_features_real" in batch:
            static_real = batch["static_features_real"]
            if static_real.ndim == 3:  # [batch_size, series_dim, features_dim]
                static_real = self._process_multi_series_static_data(static_real)
            static_inputs["static_real"] = static_real

        if "static_features_categorical" in batch:
            static_categorical = batch["static_features_categorical"]
            if static_categorical.ndim == 3:  # [batch_size, series_dim, features_dim]
                static_categorical = self._process_multi_series_static_data(static_categorical)
            elif static_categorical.dtype == torch.long:
                # Convert categorical to float for TFT compatibility
                static_categorical = static_categorical.float()
            static_inputs["static_categorical"] = static_categorical

        # Process target
        target = batch["output_target"]
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

    def _process_multi_series_static_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process multi-series static data based on the configured selection method.

        Args:
            data: Input tensor of shape [batch_size, num_series, features_dim]

        Returns:
            Processed tensor of shape [batch_size, output_features_dim]
        """
        # Convert to float if it's categorical (long) data
        if data.dtype == torch.long:
            data = data.float()

        if self.series_selection_method == "first":
            return data[:, 0, :]
        elif self.series_selection_method == "last":
            return data[:, -1, :]
        elif self.series_selection_method == "index":
            if self.series_index is None:
                raise ValueError(
                    "series_index cannot be None when using 'index' selection method"
                )
            if self.series_index >= data.shape[1]:
                raise IndexError(
                    f"series_index {self.series_index} out of range for {data.shape[1]} series"
                )
            return data[:, self.series_index, :]
        elif self.series_selection_method == "aggregate":
            if self.series_aggregation == "mean":
                return data.mean(dim=1)
            elif self.series_aggregation == "sum":
                return data.sum(dim=1)
            elif self.series_aggregation == "max":
                return data.max(dim=1)[0]
            elif self.series_aggregation == "min":
                return data.min(dim=1)[0]
        elif self.series_selection_method == "flatten":
            # Flatten series and features dimensions
            batch_size, num_series, features_dim = data.shape
            return data.view(batch_size, num_series * features_dim)

        raise ValueError(
            f"Unknown series_selection_method: {self.series_selection_method}"
        )

    def _process_multi_series_target(self, target: torch.Tensor) -> torch.Tensor:
        """
        Process multi-series target data based on the configured selection method.

        Args:
            target: Input tensor of shape [batch_size, time_length, num_series, features_dim]
                   or [batch_size, time_length, num_series]

        Returns:
            Processed tensor of shape [batch_size, time_length]
        """
        # Handle different target shapes
        if target.ndim == 4:  # [batch_size, time_length, num_series, features_dim]
            if self.series_selection_method == "first":
                result = target[:, :, 0, :]
            elif self.series_selection_method == "last":
                result = target[:, :, -1, :]
            elif self.series_selection_method == "index":
                if self.series_index is None:
                    raise ValueError(
                        "series_index cannot be None when using 'index' selection method"
                    )
                if self.series_index >= target.shape[2]:
                    raise IndexError(
                        f"series_index {self.series_index} out of range for {target.shape[2]} series"
                    )
                result = target[:, :, self.series_index, :]
            elif self.series_selection_method == "aggregate":
                if self.series_aggregation == "mean":
                    result = target.mean(dim=2)
                elif self.series_aggregation == "sum":
                    result = target.sum(dim=2)
                elif self.series_aggregation == "max":
                    result = target.max(dim=2)[0]
                elif self.series_aggregation == "min":
                    result = target.min(dim=2)[0]
                else:
                    raise ValueError(f"Unknown aggregation method: {self.series_aggregation}")
            elif self.series_selection_method == "flatten":
                # For target, we typically don't flatten - instead average across series
                result = target.mean(dim=2)
            else:
                raise ValueError(f"Unknown series_selection_method: {self.series_selection_method}")

            # If result still has features dimension, take the first feature or average
            if result.ndim == 3:  # [batch_size, time_length, features_dim]
                if result.shape[2] == 1:
                    result = result.squeeze(-1)  # [batch_size, time_length]
                else:
                    result = result.mean(dim=2)  # Average across features

        elif target.ndim == 3:  # [batch_size, time_length, num_series]
            if self.series_selection_method == "first":
                result = target[:, :, 0]
            elif self.series_selection_method == "last":
                result = target[:, :, -1]
            elif self.series_selection_method == "index":
                if self.series_index is None:
                    raise ValueError(
                        "series_index cannot be None when using 'index' selection method"
                    )
                if self.series_index >= target.shape[2]:
                    raise IndexError(
                        f"series_index {self.series_index} out of range for {target.shape[2]} series"
                    )
                result = target[:, :, self.series_index]
            elif self.series_selection_method == "aggregate":
                if self.series_aggregation == "mean":
                    result = target.mean(dim=2)
                elif self.series_aggregation == "sum":
                    result = target.sum(dim=2)
                elif self.series_aggregation == "max":
                    result = target.max(dim=2)[0]
                elif self.series_aggregation == "min":
                    result = target.min(dim=2)[0]
                else:
                    raise ValueError(f"Unknown aggregation method: {self.series_aggregation}")
            elif self.series_selection_method == "flatten":
                # For target, we typically don't flatten - instead average across series
                result = target.mean(dim=2)
            else:
                raise ValueError(f"Unknown series_selection_method: {self.series_selection_method}")
        else:
            # Target is already in the right shape [batch_size, time_length]
            result = target

        return result

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a single training step.

        Args:
            batch: A dictionary containing the batch data in the new standardized format.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the training loss.
        """
        batch = self._unpack_batch(batch)
        past_inputs, future_inputs, static_inputs, target = self._prepare_data(batch)

        model_output = self(past_inputs, future_inputs, static_inputs)
        loss = self.quantile_loss(model_output[0], target)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a single validation step.

        Args:
            batch: A dictionary containing the batch data in the new standardized format.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the validation loss.
        """
        batch = self._unpack_batch(batch)
        past_inputs, future_inputs, static_inputs, target = self._prepare_data(batch)

        model_output = self(past_inputs, future_inputs, static_inputs)
        loss = self.quantile_loss(model_output[0], target)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a single test step.

        Args:
            batch: A dictionary containing the batch data in the new standardized format.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the test loss.
        """
        batch = self._unpack_batch(batch)
        past_inputs, future_inputs, static_inputs, target = self._prepare_data(batch)

        model_output = self(past_inputs, future_inputs, static_inputs)
        loss = self.quantile_loss(model_output[0], target)

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        """
        Performs a single prediction step.

        Args:
            batch: A dictionary containing the batch data in the new standardized format.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            The model predictions.
        """
        batch = self._unpack_batch(batch)
        past_inputs, future_inputs, static_inputs, _ = self._prepare_data(batch)

        model_output = self(past_inputs, future_inputs, static_inputs)
        return model_output
