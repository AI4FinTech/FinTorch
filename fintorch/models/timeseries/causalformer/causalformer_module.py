from typing import Any, Tuple

import lightning as L
import torch
from fintorch.models.timeseries.causalformer.CausalFormer import CausalFormer
from lightning.pytorch.utilities.types import STEP_OUTPUT
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
            number_of_layers=self.hparams.number_of_layers,
            number_of_heads=self.hparams.number_of_heads,
            number_of_series=self.hparams.number_of_series,
            length_input_window=self.hparams.length_input_window,
            length_output_window=self.hparams.length_output_window,
            embedding_size=self.hparams.embedding_size,
            feature_dimensionality=self.hparams.feature_dimensionality,
            ffn_hidden_dimensionality=self.hparams.ffn_hidden_dimensionality,
            output_dimensionality=self.hparams.output_dimensionality,
            tau=self.hparams.tau,
            dropout=self.hparams.dropout,
        )

        self.loss = torch.nn.L1Loss()
        self.loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.causalformer(x)

    def _prepare_data(self, batch):
        past_inputs, _, _, target = batch
        x, y = past_inputs["past_data"], target

        # Adds number of features dimensionality
        x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1, 3)
        # Adds number of series dimensionality
        y = y.unsqueeze(-1).unsqueeze(1)
        return x, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        x, y = self._prepare_data(batch)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        x, y = self._prepare_data(batch)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        x, _ = self._prepare_data(batch)
        y_hat = self(x)
        return y_hat

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = self._prepare_data(batch)
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
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True,
        )

        scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma,
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
