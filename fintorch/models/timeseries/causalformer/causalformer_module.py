from typing import Any, Dict, Tuple

import lightning as L
import torch
from fintorch.models.timeseries.causalformer.CausalFormer import CausalFormer
from torch.optim.lr_scheduler import StepLR


class CausalFormerModule(L.LightningModule):
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
    ):
        super().__init__()
        self.save_hyperparameters()

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

    def _prepare_data(
        self,
        batch: Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input and target tensors for the CausalFormer model.

        Input tensor transformations:
        - Input from SimpleSyntheticDataset: [batch_size, past_length, num_series, features_dim]
        - Input for CausalFormer: [batch_size, num_series, past_length, feature_dimensionality]

        Target tensor transformations:
        - Target 2D: [batch_size, future_length] -> [batch_size, num_series, future_length, feature_dim]
        - Target 3D: [batch_size, future_length, num_series] -> [batch_size, num_series, future_length, feature_dim]
        - Target 4D: [batch_size, future_length, num_series, feature_dim] -> [batch_size, num_series, future_length, feature_dim]

        Args:
            batch: Tuple containing (past_inputs, future_inputs, static_inputs, target)

        Returns:
            Tuple of (input_tensor, target_tensor) in CausalFormer expected format
        """
        past_inputs, _, _, target = batch
        x, y = past_inputs["past_data"], target

        # Handle input tensor shape adaptively
        # Expected input shape from SimpleSyntheticDataset: [batch_size, past_length, num_series, features_dim]
        # Expected shape for CausalFormer: [batch_size, num_series, past_length, feature_dimensionality]

        # Check input dimensions and reshape accordingly
        if x.ndim == 3:  # [batch_size, past_length, num_series]
            # Add feature dimension if it's missing
            x = x.unsqueeze(-1)  # [batch_size, past_length, num_series, 1]

        # Permute to the expected shape for the CausalFormer
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_series, past_length, feature_dim]

        # Handle target tensor shape - ensure it becomes 4D [batch_size, num_series, future_length, feature_dim]
        if y.ndim == 2:  # [batch_size, future_length]
            # Add both series and feature dimensions
            y = y.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, future_length, 1]
            # Expand to match the number of series from input tensor
            num_series = x.shape[1]
            y = y.expand(
                -1, num_series, -1, -1
            )  # [batch_size, num_series, future_length, 1]
        elif y.ndim == 3:  # [batch_size, future_length, num_series]
            # Add feature dimension and permute
            y = y.unsqueeze(-1)  # [batch_size, future_length, num_series, 1]
            y = y.permute(0, 2, 1, 3)  # [batch_size, num_series, future_length, 1]
        elif y.ndim == 4:  # [batch_size, future_length, num_series, feature_dim]
            # Permute to match the CausalFormer output format [batch_size, num_series, future_length, feature_dim]
            y = y.permute(0, 2, 1, 3)

        return x, y

    def training_step(
        self,
        batch: Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            torch.Tensor,
        ],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        x, y = self._prepare_data(batch)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def test_step(
        self,
        batch: Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            torch.Tensor,
        ],
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
        batch: Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            torch.Tensor,
        ],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:  # type: ignore
        x, _ = self._prepare_data(batch)
        y_hat = self(x)
        return y_hat

    def validation_step(
        self,
        batch: Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            torch.Tensor,
        ],
        batch_idx: int,
    ) -> Any:
        x, y = self._prepare_data(batch)  # type: ignore
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    # def loss(self, predicted, labels):

    #     print(f"labels:{labels}")
    #     mask = ~torch.ne(labels).float()
    #     num_valid = torch.sum(mask)

    #     if num_valid == 0:
    #         return torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)

    #     abs_error = torch.abs(predicted - labels)

    #     masked_abs_error = abs_error * mask

    #     # TODO: add model regularization to the loss
    #     # TODO: add LAM loss

    #     mae = torch.sum(masked_abs_error) / num_valid

    #     return mae

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
                # "monitor": "val_loss", # Optional: For schedulers like ReduceLROnPlateau
            },
        }
