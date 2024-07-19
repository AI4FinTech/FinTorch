# from fintorch.datasets import elliptic
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
import torchmetrics
from torch import Tensor
from torch_geometric.loader import DataLoader

from fintorch.datasets import elliptic

# Load the elliptic dataset
elliptic_dataset = elliptic.TransactionDataset("~/.fintorch_data", force_reload=True)


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        num_layers: int = 5,
        dp_rate: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the Elliptical class.

        Args:
            c_in (int): Number of input channels.
            c_hidden (int): Number of hidden channels.
            c_out (int): Number of output channels.
            num_layers (int, optional): Number of GNN layers. Defaults to 5.
            dp_rate (float, optional): Dropout rate. Defaults to 0.1.
            **kwargs: Additional keyword arguments to be passed to the GNN layers.

        Returns:
            None
        """

        super().__init__()
        gnn_layer = geom_nn.GCNConv

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.
            edge_index (Tensor): Edge index tensor.

        Returns:
            Tensor: Output tensor.
        """

        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                # In case of a geom layer, also pass the edge_index list
                x = layer(x, edge_index)
            else:
                x = layer(x)

        return x


class GNN(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=3
        )

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Get the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        # Calculate the loss for the mask
        loss = self.loss_module(x[mask], data.y[mask].long())
        pred = x[mask].argmax(dim=-1)

        return loss, pred, data.y[mask]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.forward(batch, mode="train")

        # log step metric
        self.accuracy(preds, y)
        self.log("train_acc_step", self.accuracy)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.forward(batch, mode="val")

        # log step metric
        self.accuracy(preds, y)
        self.log("val_acc_step", self.accuracy)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self.forward(batch, mode="test")

        # log step metric
        self.accuracy(preds, y)
        self.log("test_acc_step", self.accuracy)


def train_node_classifier(dataset, **model_kwargs):
    node_data_loader = DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=1000, enable_progress_bar=True
    )  # False because epoch size is 1

    # Note: the dimensions are specific for the Elliptic dataset
    model = GNN(**model_kwargs)
    trainer.fit(
        model, train_dataloaders=node_data_loader, val_dataloaders=node_data_loader
    )

    # Test best model on the test set
    trainer.test(model, node_data_loader, verbose=True)

    return model


node_gnn_model = train_node_classifier(
    dataset=elliptic_dataset, c_in=166, c_hidden=256, c_out=3, num_layers=4, dp_rate=0.1
)
