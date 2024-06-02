import lightning as L
import torch

from fintorch.datasets.ellipticpp import EllipticppDataModule
from fintorch.graph.layers.beanconv import BEANConvSimple
from fintorch.models.graph.graphBEAN import GraphBEANModule

torch.set_float32_matmul_precision("medium")

# We use an example data module from the elliptic dataset which is bipartite
data_module = EllipticppDataModule(("wallets", "to", "transactions"),
                                   force_reload=False)

# Create an instance of the GraphBEANModule
module = GraphBEANModule(
    ("wallets", "to", "transactions"),
    edge_types=[("wallets", "to", "transactions"),
                ("transactions", "to", "wallets")],
    learning_rate=0.0001,
    conv_type=BEANConvSimple,
    encoder_layers=5,
    decoder_layers=5,
    hidden_layers=512,
    classes=2,
    class_head_layers=100,
    classifier=True,
    predict="transactions",
)

# Create a PyTorch Lightning Trainer and train the module
trainer = L.Trainer(max_epochs=100, accelerator="gpu")

# Train the module using the dataloaders
trainer.fit(module, datamodule=data_module)
