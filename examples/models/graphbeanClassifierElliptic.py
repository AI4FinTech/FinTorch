import lightning as L
import torch

from fintorch.datasets.ellipticpp import EllipticppDataModule
from fintorch.graph.layers.beanconv import BEANConvSimple
from fintorch.models.graph.graphbean.graphBEAN import GraphBEANModule

torch.set_float32_matmul_precision("medium")


def main():
    # We use an example data module from the elliptic dataset which is bipartite
    data_module = EllipticppDataModule(
        ("transactions", "to", "wallets"), force_reload=False
    )

    try:
        data_module.setup()
        # Get the dimensionalities for the auto-encoder part
        mapping = dict()
        for key in data_module.dataset.metadata()[0]:
            mapping[key] = data_module.dataset[key].x.shape[1]
    except Exception as e:
        print(f"Error setting up data module or retrieving metadata: {e}")
        raise
    # Create an instance of the GraphBEANModule
    module = GraphBEANModule(
        ("transactions", "to", "wallets"),
        edge_types=[
            ("wallets", "to", "transactions"),
            ("transactions", "to", "wallets"),
        ],
        mapping=mapping,
        learning_rate=0.0001,
        conv_type=BEANConvSimple,
        encoder_layers=5,
        decoder_layers=5,
        hidden_layers=512,
        classes=2,
        class_head_layers=100,
        classifier=True,
        predict="wallets",
        node_types=list(mapping.keys()),
    )

    # Create a PyTorch Lightning Trainer and train the module
    trainer = L.Trainer(max_epochs=100, accelerator="auto")

    # Train the module using the dataloaders
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
