import lightning as L
import torch

from fintorch.models.timeseries.tft import TemporalFusionTransformer


class TemporalFusionTransformerModule(L.LightningModule):
    def __init__(
        self,
        number_of_past_inputs,
        number_of_future_inputs,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        past_inputs,
        future_inputs,
        static_inputs,
        batch_size,
        device,
    ):
        super().__init__()

        self.tft_model = TemporalFusionTransformer(
            number_of_past_inputs,
            number_of_future_inputs,
            embedding_size_inputs,
            hidden_dimension,
            dropout,
            number_of_heads,
            past_inputs,
            future_inputs,
            static_inputs,
            batch_size,
            device,
        )

    def forward(self, past_inputs, future_inputs, static_inputs):
        return self.tft_model(past_inputs, future_inputs, static_inputs)

    def _unpack_batch(self, batch):
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
                inputs, target = batch
                if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
                    past_inputs, future_inputs, static_inputs = inputs
                else:
                    # Assume only past_inputs are provided.
                    past_inputs = inputs
                    future_inputs, static_inputs = None, None
            else:
                raise ValueError(
                    f"Unexpected batch format: expected 2 or 4 items, got {len(batch)}"
                )
        else:
            raise ValueError("Batch must be a tuple or list")
        return past_inputs, future_inputs, static_inputs, target

    def training_step(self, batch, batch_idx):

        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)

        output, attention_weights = self.forward(
            past_inputs, future_inputs, static_inputs
        )

        # Calculate the loss
        # TODO: replace with loss reported in the paper
        loss = torch.nn.functional.mse_loss(output.squeeze(), target)

        # Log the loss
        # self.log("train_loss", loss)
        self.log("train_loss_epoch", loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)

        output, attention_weights = self.forward(
            past_inputs, future_inputs, static_inputs
        )

        # Calculate the loss
        # TODO: replace with loss reported in the paper
        loss = torch.nn.functional.mse_loss(output.squeeze(), target)

        # Log the loss
        # self.log("val_loss", loss)
        self.log("val_loss_epoch", loss, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)
        output, attention_weights = self.forward(
            past_inputs, future_inputs, static_inputs
        )

        # Calculate the loss
        # TODO: replace with loss reported in the paper
        loss = torch.nn.functional.mse_loss(output.squeeze(), target)

        # Log the loss
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        print("predict_step")
        print(f"batch {batch}")
        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)
        output, attention_weights = self.forward(
            past_inputs, future_inputs, static_inputs
        )

        return output
