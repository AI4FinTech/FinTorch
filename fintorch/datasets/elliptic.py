import os

import numpy as np
import polars as pol
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from fintorch.datasets.kaggle.downloader import KaggleDownloader


class EllipticDataset(Dataset):
    # The elliptic dataset consist of three files
    # *elliptic_txs_classes.csv
    # *elliptic_txs_edgelist.csv
    # *elliptic_txs_features.csv

    def __init__(self, download=False):
        super().__init__()

        if download:
            # Force download
            self.download_data()
        else:
            # Download if it doesn't exist, otherwise use cache
            dataset_dir = os.path.expanduser(
                "~/.fintorch_data/elliptic_bitcoin_dataset"
            )

            features_file_path = os.path.join(dataset_dir, "elliptic_txs_features.csv")
            edgelist_file_path = os.path.join(dataset_dir, "elliptic_txs_edgelist.csv")
            classes_file_path = os.path.join(dataset_dir, "elliptic_txs_classes.csv")

            if (
                not os.path.exists(features_file_path)
                or not os.path.exists(edgelist_file_path)
                or not os.path.exists(classes_file_path)
            ):
                print("One or more files are missing. Starting download...")
                self.download_data()
            else:
                print("Using cached version of elliptic_bitcoin_dataset....")

        self.load_data()

    def load_data(self):
        features_file_path = (
            "~/.fintorch_data/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
        )
        edgelist_file_path = (
            "~/.fintorch_data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
        )
        classes_file_path = (
            "~/.fintorch_data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
        )

        # TODO: correctly implement the data preparation in polars

        features = pol.read_csv(features_file_path)
        edgelist = pol.read_csv(edgelist_file_path)
        classes = pol.read_csv(classes_file_path)

        # Mapping 'class' column for consistency
        classes = classes.with_columns(
            pol.col("class")
            .cast(pol.Utf8)
            .apply(lambda x: {"unknown": 2, "1": 1, "2": 0}.get(x))
        )

        classes = classes.with_columns(classes["txId"].cast(pol.Int32).alias("txId"))

        # Merging DataFrames efficiently with Polars
        df_merge = features.join(classes, how="left", left_on=0, right_on="txId")

        # Drop redundant column
        df_merge = df_merge.drop("txId")

        # Extracting the unique nodes
        nodes = df_merge[0].unique()

        # Creating the mapping dictionary (optimized for Polars)
        map_id = nodes.to_dict()

        # Mapping IDs efficiently using Polars
        edges = edgelist.with_columns(
            [
                pol.col("txId1").apply(lambda x: map_id.get(x)),
                pol.col("txId2").apply(lambda x: map_id.get(x)),
            ]
        )

        # Casting to integers
        edges = edges.select(pol.all().cast(pol.Int32))

        # Preparing edge_index for PyTorch
        edge_index = np.array(edges.to_numpy()).T
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        # Creating the weights tensor
        self.weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.float32)

        # Efficient in-place replacement
        df_merge = df_merge.with_columns(pol.col("class").replace(2, 0))

        # Extract labels
        self.labels = torch.tensor(df_merge["class"].to_numpy(), dtype=torch.float32)

        # Extract node features (with optimization)
        self.node_features = torch.tensor(
            df_merge.drop(["class"]).to_numpy(), dtype=torch.float32
        )

    def download_data(self):
        # Usage example:
        downloader = KaggleDownloader()
        downloader.download_dataset("ellipticco/elliptic-data-set")

    def __len__(self):
        return len(self.data_features)

    def __getitem__(self, idx):
        a_node_features = self.node_features[idx]
        a_edge_index = self.edge_index[idx]
        a_weight = self.weights[idx]  # Replace with the actual edge weights
        a_label = self.labels[idx]

        return {
            "x": a_node_features,
            "edge_index": a_edge_index,
            "edge_weights": a_weight,
            "y": a_label,
        }


class LightningEllipticDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = EllipticDataset(self.train_data, download=True)
        self.val_dataset = EllipticDataset(self.val_data)
        self.test_dataset = EllipticDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
