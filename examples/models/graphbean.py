import lightning as L

from fintorch.datasets.ellipticpp import EllipticDataModule
from fintorch.graph.layers.beanconv import BEANConv
from fintorch.models.graph.graphBEAN import GraphBEANModule

# We use an example data module from the elliptic dataset which is bipartite
data_module = EllipticDataModule(("wallets", "to", "transactions"))

# Create an instance of the GraphBEANModule
module = GraphBEANModule(
    ("wallets", "to", "transactions"),
    edge_types=[("wallets", "to", "transactions"),
                ("transactions", "to", "wallets")],
    learning_rate=0.001,
    conv_type=BEANConv,
    encoder_layers=2,
    decoder_layers=2,
    hidden_layers=100,
    classifier=True,
)

# Create a PyTorch Lightning Trainer and train the module
trainer = L.Trainer(max_epochs=10,
                    accelerator="cpu",
                    limit_val_batches=0,
                    num_sanity_val_steps=0)

# Train the module using the dataloaders
trainer.fit(module, datamodule=data_module)
