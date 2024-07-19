import lightning as L
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import LinkNeighborLoader

from fintorch.graph.layers.beanconv import BEANConvSimple
from fintorch.models.graph.graphbean.graphBEAN import GraphBEANModule


def main():
    class OGBDataModule(L.LightningDataModule):
        def __init__(self, data, edge):
            super().__init__()
            self.dataset = data
            self.edge = edge

        def train_dataloader(self):
            # Define neighbor sampling configuration
            src, to, dst = self.edge
            num_neighbors = {
                ("paper", "cites", "paper"): [10, 10],
                ("author", "writes", "paper"): [10, 10],
                ("paper", "has_topic", "field_of_study"): [10, 10],
                ("author", "affiliated_with", "institution"): [10, 10],
                ("institution", "rev_affiliated_with", "author"): [10, 10],
                ("paper", "rev_writes", "author"): [10, 10],
                ("field_of_study", "rev_has_topic", "paper"): [10, 10],
            }
            loader = LinkNeighborLoader(
                self.dataset,
                num_neighbors=num_neighbors,
                batch_size=512,
                shuffle=True,
                edge_label_index=(
                    (src, to, dst),
                    self.dataset[src, to, dst].edge_index,
                ),
                neg_sampling="binary",
                num_workers=14,
            )

            return loader

        def val_dataloader(self):
            # Define neighbor sampling configuration
            src, to, dst = self.edge
            num_neighbors = {
                ("paper", "cites", "paper"): [10, 10],
                ("author", "writes", "paper"): [10, 10],
                ("paper", "has_topic", "field_of_study"): [10, 10],
                ("author", "affiliated_with", "institution"): [10, 10],
                ("institution", "rev_affiliated_with", "author"): [10, 10],
                ("paper", "rev_writes", "author"): [10, 10],
                ("field_of_study", "rev_has_topic", "paper"): [10, 10],
            }
            loader = LinkNeighborLoader(
                self.dataset,
                num_neighbors=num_neighbors,
                batch_size=128,
                edge_label_index=(
                    (src, to, dst),
                    self.dataset[src, to, dst].edge_index,
                ),
                neg_sampling="binary",
                num_workers=14,
            )

            return loader

    dataset = OGB_MAG(root="./data", preprocess="metapath2vec")
    data = T.ToUndirected()(dataset[0])
    data_module = OGBDataModule(data, ("author", "writes", "paper"))

    try:
        # Get the dimensionalities for the auto-encoder part
        mapping = dict()
        for key in data_module.dataset.metadata()[0]:
            mapping[key] = data_module.dataset[key].x.shape[1]
    except Exception as e:
        print(f"Error retrieving metadata: {e}")
        raise
    # # Create an instance of the GraphBEANModule
    module = GraphBEANModule(
        ("author", "writes", "paper"),
        edge_types=[
            ("author", "writes", "paper"),
            ("paper", "rev_writes", "author"),
            ("author", "affiliated_with", "institution"),
            ("institution", "rev_affiliated_with", "author"),
        ],
        mapping=mapping,
        learning_rate=0.001,
        conv_type=BEANConvSimple,
        encoder_layers=5,
        decoder_layers=5,
        hidden_layers=50,
        predict="paper",
    )

    # Create a PyTorch Lightning Trainer and train the module
    trainer = L.Trainer(max_epochs=100, accelerator="auto")

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
