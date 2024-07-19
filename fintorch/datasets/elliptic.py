import os
from typing import Callable, List, Optional

import numpy as np
import polars as pol
import torch
from torch_geometric.data import Data, InMemoryDataset

from fintorch.datasets.kaggle.downloader import KaggleDownloader


class TransactionDataset(InMemoryDataset):
    """
    The Elliptic Data Set: Understanding Bitcoin Transactions

    The Elliptic Data Set provides valuable insights into Bitcoin transactions,
    allowing researchers to distinguish between licit and illicit activities.
    Here's what you need to know:

    Purpose:

    - Maps Bitcoin transactions to real-world entities.
    - Classifies entities as licit or illicit.

    Content:

    - Anonymized transaction graph from the Bitcoin blockchain.
    - Nodes represent transactions; edges represent Bitcoin flows.
    - 203,769 nodes and 234,355 edges.
    - Each node has 166 features.
    - Nodes are labeled "licit", "illicit", or "unknown".

    Features:

    - Include transaction time, inputs/outputs, fees, volume, and aggregated statistics.
    - Time steps group transactions into connected components.

    Citations:

    If you use the Elliptic Data Set, please cite:
    [1] Elliptic, www.elliptic.co.
    [2] M. Weber, et al., "Anti-Money Laundering in Bitcoin:
    Experimenting with Graph Convolutional Networks for Financial Forensics",
    KDD ’19 Workshop on Anomaly Detection in Finance, August 2019, Anchorage, AK, USA.

    The EllipticDataset consists of three files:

    - elliptic_txs_classes.csv
    - elliptic_txs_edgelist.csv
    - elliptic_txs_features.csv

    Parameters:
    - root (str): Root directory where the dataset exists or will be downloaded.
    - transform (Optional[Callable]): A function/transform that takes in an
      object and returns a transformed version. Default is None.
    - pre_transform (Optional[Callable]): A function/transform that takes in an
      object and returns a transformed version. Default is None.
    - pre_filter (Optional[Callable]): A function that takes in an object and
      returns a boolean value indicating whether the object should be included
      in the final dataset. Default is None.
    - force_reload (bool): Flag indicating whether to force reloading the dataset
      even if it already exists. Default is False.

    Returns:
    None
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

        self.load(self.processed_paths[0])
        data = self.get(0)
        self.data, self.slices = self.collate([data])

        assert isinstance(self._data, Data)

    @property
    def raw_file_names(self):
        """
        Returns a list of raw file names for the elliptic dataset.

        Returns:
            list: A list of raw file names.
        """
        return [
            "elliptic_bitcoin_dataset/elliptic_txs_features.csv",
            "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
            "elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns a list of processed file names.

        Returns:
            list: A list of processed file names.
        """
        return ["transaction_graph_v1.pt"]

    def download(self) -> None:
        """
        Downloads the elliptic dataset from Kaggle to `self.raw_dir`.
        """
        # Download from Kaggle to `self.raw_dir`.
        downloader = KaggleDownloader()
        downloader.download_dataset(
            "ellipticco/elliptic-data-set", download_dir=self.raw_dir
        )

    def process(self) -> None:
        """
        Process the dataset by reading the required files, performing data transformations,
        and saving the processed data to disk.

        Returns:
            None
        """

        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        features_file_path = os.path.join(
            self.raw_dir, "elliptic_bitcoin_dataset", "elliptic_txs_features.csv"
        )

        edgelist_file_path = os.path.join(
            self.raw_dir, "elliptic_bitcoin_dataset", "elliptic_txs_edgelist.csv"
        )
        classes_file_path = os.path.join(
            self.raw_dir, "elliptic_bitcoin_dataset", "elliptic_txs_classes.csv"
        )

        features = pol.read_csv(features_file_path, has_header=False)
        edgelist = pol.read_csv(edgelist_file_path)
        classes = pol.read_csv(classes_file_path)

        # Mapping 'class' column to numerics
        # The dataset has licit (0), illicit (1), and unknown (2) entities.
        classes = classes.with_columns(
            pol.col("class")
            .cast(pol.Utf8)
            .map_elements(
                lambda x: {"unknown": 2, "1": 1, "2": 0}.get(x), return_dtype=pol.Int64
            )
        )

        # Cast the classes to Int32 identifiers
        classes = classes.with_columns(classes["txId"].cast(pol.Int64).alias("txId"))

        # Merging DataFrames efficiently with Polars
        # The resulting DataFrame consists of the classes for each node
        df_merge = features.join(
            classes, how="left", left_on="column_1", right_on="txId"
        )

        nodes = df_merge.select("column_1").to_numpy().flatten()
        self.map_id = dict(zip(nodes, range(len(nodes))))

        edgelist = edgelist.with_columns(
            txId1=pol.col("txId1").replace(self.map_id, default=-1)
        )

        edgelist = edgelist.with_columns(
            txId2=pol.col("txId2").replace(self.map_id, default=-1)
        )

        # Preparing edge_index for PyTorch
        edge_index = np.array(edgelist.to_numpy()).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        # Extract labels
        labels = torch.tensor(df_merge["class"].to_numpy(), dtype=torch.float32).long()

        # Extract node features (with optimization)
        node_features = torch.tensor(
            df_merge.drop(["class", "column_1"]).to_numpy(), dtype=torch.float32
        )

        num_data = node_features.shape[0]
        num_train = int(0.8 * num_data)
        num_val = int(0.1 * num_data)
        num_test = num_data - num_train - num_val

        train_index = torch.arange(num_train, dtype=torch.long)
        val_index = torch.arange(num_train, num_train + num_val, dtype=torch.long)
        test_index = torch.arange(
            num_train + num_val, num_train + num_val + num_test, dtype=torch.long
        )

        train_mask = torch.zeros_like(labels, dtype=torch.bool)
        val_mask = torch.zeros_like(labels, dtype=torch.bool)
        test_mask = torch.zeros_like(labels, dtype=torch.bool)

        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        df = Data(
            x=node_features,
            edge_index=edge_index,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        self.save([df], self.processed_paths[0])
