
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
        )

    def forward(self, past_inputs, future_inputs, static_inputs):
        return self.tft_model(past_inputs, future_inputs, static_inputs)

    def training_step(self, batch, batch_idx):
        past_inputs, future_inputs, static_inputs, target = batch
        output, attention_weights = self.forward(
            past_inputs, future_inputs, static_inputs
        )

        # Calculate the loss
        # TODO: replace with loss reported in the paper
        loss = torch.nn.functional.mse_loss(output.squeeze(), target)

        # Log the loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        past_inputs, future_inputs, static_inputs, target = batch
        output, attention_weights = self.forward(
            past_inputs, future_inputs, static_inputs
        )

        # Calculate the loss
        # TODO: replace with loss reported in the paper
        loss = torch.nn.functional.mse_loss(output.squeeze(), target)

        # Log the loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        past_inputs, future_inputs, static_inputs, target = batch
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

    def predict_step(self, batch, batch_idx):
        past_inputs, future_inputs, static_inputs, target = batch
        output, attention_weights = self.forward(
            past_inputs, future_inputs, static_inputs
        )

        return output
