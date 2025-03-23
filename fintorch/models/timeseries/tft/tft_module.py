import os

import lightning as L
import torch

from fintorch.models.timeseries.tft import TemporalFusionTransformer


class TemporalFusionTransformerModule(L.LightningModule):
    """
    TemporalFusionTransformerModule is a PyTorch Lightning module that encapsulates the Temporal Fusion Transformer (TFT) model for time series forecasting.
    It provides methods for training, validation, testing, and prediction, along with utilities for handling input data batches.
    Attributes:
        tft_model (TemporalFusionTransformer): The underlying Temporal Fusion Transformer model.
    Methods:
        __init__(number_of_past_inputs, horizon, embedding_size_inputs, hidden_dimension, dropout, number_of_heads, past_inputs, future_inputs, static_inputs, batch_size, device):
            Initializes the TemporalFusionTransformerModule with the specified parameters.
        forward(past_inputs, future_inputs, static_inputs):
            Performs a forward pass through the TFT model.
        _unpack_batch(batch):
            Unpacks the input batch into past inputs, future inputs, static inputs, and target values. Supports multiple batch formats.
        training_step(batch, batch_idx):
            Defines the training step, including forward pass, loss computation, and logging.
        validation_step(batch, batch_idx):
            Defines the validation step, including forward pass, loss computation, and logging.
        test_step(batch, batch_idx):
            Defines the test step, including forward pass, loss computation, logging, and saving comparison plots of predictions vs targets.
        configure_optimizers():
            Configures the optimizer for training.
        predict_step(batch, batch_idx, dataloader_idx=0):
            Defines the prediction step, returning the model's output for the given batch.
    """

    def __init__(
        self,
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
        quantiles=[0.1, 0.5, 0.9],
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

    def forward(self, past_inputs, future_inputs, static_inputs):
        return self.tft_model(past_inputs, future_inputs, static_inputs)

    def quantile_loss(self, model_output, target):
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

        dim_q = 3
        device = model_output.device
        errors = target.unsqueeze(-1) - model_output
        quantiles_tensor = torch.tensor(self.tft_model.quantiles).to(device)
        losses = torch.max(
            (quantiles_tensor - 1) * errors, quantiles_tensor * errors
        ).sum(dim=dim_q)
        return losses.mean()

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

        output, _ = self.forward(past_inputs, future_inputs, static_inputs)

        # Calculate the loss
        # TODO: replace with loss reported in the paper
        # loss = torch.nn.functional.mse_loss(output.squeeze(), target.squeeze())
        loss = self.quantile_loss(output, target)

        # Log the loss
        # self.log("train_loss", loss)
        self.log("train_loss_epoch", loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)

        output, _ = self.forward(past_inputs, future_inputs, static_inputs)

        # Calculate the loss
        # TODO: replace with loss reported in the paper
        # loss = torch.nn.functional.mse_loss(output.squeeze(), target.squeeze())
        loss = self.quantile_loss(output, target)

        # Log the loss
        # self.log("val_loss", loss)
        self.log("val_loss_epoch", loss, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        past_inputs, future_inputs, static_inputs, target = self._unpack_batch(batch)
        output, _ = self.forward(past_inputs, future_inputs, static_inputs)

        # Calculate the loss
        # TODO: replace with loss reported in the paper
        # loss = torch.nn.functional.mse_loss(output.squeeze(), target.squeeze())
        loss = self.quantile_loss(output, target)

        # Plot and store the comparison between predicted outputs and targets
        import matplotlib.pyplot as plt

        # Create a directory to save the plots if it doesn't exist
        plot_dir = "/home/marcel/Documents/research/FinTorch/plots"
        os.makedirs(plot_dir, exist_ok=True)

        # Plot the predicted outputs vs targets
        plt.figure(figsize=(10, 6))
        plt.plot(
            target[0, :].squeeze().cpu().detach().numpy(), label="Target", color="black"
        )
        for i in range(output.shape[3]):
            plt.plot(
                output[0, :, :, i].squeeze().cpu().detach().numpy(),
                label=f"Predicted quantile {self.tft_model.quantiles[i]}",
            )

        plt.legend()
        plt.title(f"Comparison of Predicted Outputs and Targets - Batch {batch_idx}")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")

        # Save the plot
        plot_path = os.path.join(plot_dir, f"comparison_batch_{batch_idx}.png")
        plt.savefig(plot_path)
        plt.close()

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
