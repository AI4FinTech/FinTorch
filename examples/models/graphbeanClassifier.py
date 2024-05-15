import lightning as L
from torch_geometric.nn.conv import TransformerConv

from fintorch.datasets.ellipticpp import EllipticDataModule
from fintorch.models.graph.graphBEAN import GraphBEANModule

# We use an example data module from the elliptic dataset which is bipartite
data_module = EllipticDataModule(("wallets", "to", "transactions"))

# Create an instance of the GraphBEANModule
module = GraphBEANModule(
    ("wallets", "to", "transactions"),
    learning_rate=0.001,
    classifier=True,
    conv_type=TransformerConv,
    encoder_layers=10,
    decoder_layers=10,
    hidden_layers=100,
    class_head_layers=4,
    predict="transactions",
)

# Create a PyTorch Lightning Trainer and train the module
trainer = L.Trainer(max_epochs=10, accelerator="gpu")

# Train the module using the dataloaders
trainer.fit(module, datamodule=data_module)
