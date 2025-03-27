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

    Attributes:
        tft_model (TemporalFusionTransformer): The underlying TFT model.

    Methods:
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
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

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
    ):
        super().__init__()

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
        assert (
            len(model_output.shape) == 4
        ), f"Model output shape incorrect: {model_output.shape}"
        assert len(target.shape) == 3, f"Target shape incorrect: {target.shape}"
        assert (
            model_output.shape[:2] == target.shape[:2]
        ), "Mismatch between predicted and target shape"
        assert (
            model_output.shape[3] == self.tft_model.number_of_quantiles
        ), "Mismatch between number of predicted quantiles and target quantiles"

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
        past_inputs, future_inputs, static_inputs, _ = self._unpack_batch(batch)  # type: ignore
        output, _ = self.forward(past_inputs, future_inputs, static_inputs)  # type: ignore

        return output
