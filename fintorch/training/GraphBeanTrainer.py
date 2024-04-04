from typing import Any

import pytorch_lightning as pl


class GraphBEANModule(pl.LightningModule):

    def __init__(self, model, loss_fn, optimizer):
        super().__init__()
        print("Initialization of the GraphBEANModule")
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, x):
        print(f"Forward pass:{x}")
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return super().predict_step(*args, **kwargs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return self.optimizer
