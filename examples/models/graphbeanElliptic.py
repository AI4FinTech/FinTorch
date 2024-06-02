import lightning as L

from fintorch.datasets.ellipticpp import EllipticppDataModule
from fintorch.graph.layers.beanconv import BEANConvSimple
from fintorch.models.graph.graphbean.graphBEAN import GraphBEANModule

# We use an example data module from the elliptic dataset which is bipartite
data_module = EllipticppDataModule(("wallets", "to", "transactions"),
                                   force_reload=False)

# Create an instance of the GraphBEANModule
module = GraphBEANModule(
    ("wallets", "to", "transactions"),
    edge_types=[("wallets_to_transactions"), ("transactions_to_wallets")],
    learning_rate=0.001,
    conv_type=BEANConvSimple,
    encoder_layers=5,
    decoder_layers=5,
    hidden_layers=50,
)

# Create a PyTorch Lightning Trainer and train the module
trainer = L.Trainer(max_epochs=100, accelerator="gpu")

# Train the module using the dataloaders
trainer.fit(module, datamodule=data_module)
