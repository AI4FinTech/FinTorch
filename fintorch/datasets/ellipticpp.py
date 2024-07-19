import multiprocessing
from typing import Callable, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import polars as pol
import torch
import torch_geometric.transforms as T
from huggingface_hub import hf_hub_download
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm


class TransactionActorDataset(InMemoryDataset):
    """
    The Elliptic++ Data Set: Understanding Bitcoin Transactions and Actors/Wallets

    Extends the Elliptic Data Set by incorporating wallet addresses (actors) to
    enable the detection of illicit activity in the Bitcoin network.


    Purpose:

    * Maps Bitcoin transactions to real-world entities, including wallet addresses.
    * Classifies entities as licit, illicit, or unknown.
    * Facilitates the identification of illicit actors within the Bitcoin network.

    Content:

    * Inherits features and structure from the Elliptic Data Set:
        * Anonymized transaction graph from the Bitcoin blockchain.
        * 203,769 nodes (transactions) and 234,355 edges (Bitcoin flows).
        * Node features related to transactions, time, and aggregated statistics.
        * Node labels: "licit", "illicit", or "unknown".
    * Additional wallet address (actor) data:
        * 822k wallet addresses
        * Node features related to wallets, time, and other statistics.
        * Node labels: "licit", "illicit", or "unknown".

    Features:

    Here we discuss the wallet related features, for the transaction related
    features please see the elliptic.TransactionDataset.

    Note: we replaced all null values with 0 values.

    Class distribution for transactions:

    * Illicit (0)	4,545
    * Licit (1)	42,019
    * Unknown (2)	157,205

    Class distribution for wallets:

    * Illicit (0) 28,601
    * Licit (1) 338,871
    * Unknown (2) 900,788

    **Transaction related:**

    * **BTCtransacted:** Total BTC transacted (sent+received)
    * **BTCsent:** Total BTC sent
    * **BTCreceived:** Total BTC received
    * **Fees:** Total fees in BTC
    * **Feesshare:** Total fees as share of BTC transacted

    **Time related:**

    * **Blockstxs:** Number of blocks between transactions
    * **Blocksinput:** Number of blocks between being an input address
    * **Blocksoutput:** Number of blocks between being an output address
    * **Addr interactions:** Number of interactions among addresses (5 values: total, min, max, mean, median)
    * **Class:** Class label: {illicit, licit, unknown}

    **Transaction related:**

    * **Txstotal:** Total number of blockchain transactions
    * **TxSinput:** Total number of dataset transactions as input address
    * **TxSoutput:** Total number of dataset transactions as output address

    **Time related:**

    * **Timesteps:** Number of time steps transacting in
    * **Lifetime:** Lifetime in blocks
    * **Block first:** Block height first transacted in
    * **Blocklast:** Block height last transacted in
    * **Block first sent:** Block height first sent in
    * **Block first receive:** Block height first received in
    * **Repeat interactions:** Number of addresses transacted with multiple times (single value)



    Citations:

    [1] Elliptic, www.elliptic.co.
    [2] M. Weber, et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks
    for Financial Forensics", KDD ’19 Workshop on Anomaly Detection in Finance, August 2019, Anchorage, AK, USA.


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

        if force_reload:
            print("Forcing reload of the dataset")

        assert isinstance(self._data, HeteroData)

    @property
    def raw_file_names(self):
        return [
            "AddrAddr_edgelist.csv",
            "AddrTx_edgelist.csv",
            "TxAddr_edgelist.csv",
            "txs_classes.csv",
            "txs_edgelist.csv",
            "txs_features.csv",
            "wallets_features_classes_combined.csv",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns a list of processed file names.

        Returns:
            list: A list of processed file names.
        """
        return ["transaction_actor_graph_v1.pt"]

    def download(self):
        print("Start download from HuggingFace...")
        dataset_name = "AI4FinTech/ellipticpp"
        self.downloaded_files = []
        for file in tqdm(self.raw_file_names):
            a_downloaded_files = hf_hub_download(
                repo_id=dataset_name,
                filename=file,
                repo_type="dataset",
            )
            self.downloaded_files.append(a_downloaded_files)

    def map_classes(self, df):
        """
        Maps the classes in the DataFrame to numerical values.

        Args:
            df (DataFrame): The input DataFrame containing the 'class' column.

        Returns:
            DataFrame: The modified DataFrame with the 'class' column mapped to numerical values.
        """
        # Mapping 'class' column to numerics
        # The dataset has illicit (0), licit (1), and unknown (2) entities.
        return df.with_columns(
            pol.col("class")
            .cast(pol.Utf8)
            .map_elements(
                lambda x: {"3": 2, "2": 1, "1": 0}.get(x), return_dtype=pol.Int64
            )
        )

    def split_data(self, num_data, splits=None):
        """
        Splits the data into training, validation, and test sets based on the given splits.

        Args:
            num_data (int): The total number of data points.
            splits (list, optional): The split ratios for training, validation, and test sets.
                Defaults to [0.8, 0.1].

        Returns:
            tuple: A tuple containing the masks for the training, validation, and test sets.
        """

        if splits is None:
            splits = [0.8, 0.1]

        assert len(splits) == 2, "The length of splits should be 2"
        assert sum(splits) < 1, "The sum of splits should be less than 1"

        # Generate numbers
        num_train = int(splits[0] * num_data)
        num_val = int(splits[1] * num_data)
        num_test = num_data - num_train - num_val

        # Generate ranges
        train_index = torch.arange(num_train, dtype=torch.long)
        val_index = torch.arange(num_train, num_train + num_val, dtype=torch.long)
        test_index = torch.arange(
            num_train + num_val, num_train + num_val + num_test, dtype=torch.long
        )

        # Create masks
        train_mask = torch.zeros(num_data, dtype=torch.bool)
        val_mask = torch.zeros(num_data, dtype=torch.bool)
        test_mask = torch.zeros(num_data, dtype=torch.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        return train_mask, val_mask, test_mask

    def prepare_edge_index(self, edgelist, mapping_dict):
        """
        Prepares the edge index for PyTorch.

        Args:
            edgelist (pandas.DataFrame): The input edge list.
            mapping_dict (dict): A dictionary containing the mapping information.

        Returns:
            torch.Tensor: The prepared edge index as a PyTorch tensor.

        """
        for k, v in mapping_dict.items():
            edgelist = edgelist.with_columns(
                pol.col(k)
                .alias(k)
                .map_elements(lambda x: v.get(x, x), return_dtype=pol.Int64)
            )

        # Preparing edge_index for PyTorch
        edgelist = np.array(edgelist.to_numpy()).T
        return torch.tensor(edgelist, dtype=torch.long).contiguous()

    def process(self):
        """
        Process the dataset by preparing features and edge_index data, and constructing a HeteroData object.

        Returns:
            None
        """

        data_list = self.raw_file_names

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Prepare features

        # Load transaction features and classes into a Polars DataFrame
        transaction_classes = self.map_classes(pol.read_csv(self.downloaded_files[3]))
        features_transaction = pol.read_csv(self.downloaded_files[5])

        features_transaction = features_transaction.join(
            transaction_classes, how="left", left_on="txId", right_on="txId"
        )

        # Replace null values with 0
        features_transaction = features_transaction.fill_null(0)

        # Load wallets features and classes into Polars DataFrame
        features_wallets = self.map_classes(pol.read_csv(self.downloaded_files[6]))
        features_wallets = features_wallets.fill_null(0)

        # Prepare labels
        transaction_labels = torch.tensor(
            features_transaction["class"].to_numpy(), dtype=torch.float32
        ).long()
        wallet_labels = torch.tensor(
            features_wallets["class"].to_numpy(), dtype=torch.float32
        ).long()

        # Mask data
        transaction_train_mask, transaction_val_mask, transaction_test_mask = (
            self.split_data(features_transaction.shape[0])
        )

        wallet_train_mask, wallet_val_mask, wallet_test_mask = self.split_data(
            features_wallets.shape[0]
        )

        # PyTorch features
        tensor_features_wallets = torch.tensor(
            features_wallets.drop(["class", "Time step", "address"]).to_numpy(),
            dtype=torch.float32,
        )

        tensor_features_transaction = torch.tensor(
            features_transaction.drop(["class", "Time step", "txId"]).to_numpy(),
            dtype=torch.float32,
        )

        # Prepare edge_index data

        # Replace nodes with indices for transactions and wallets
        features_transaction = features_transaction.with_columns(
            mapped_id=pol.arange(0, features_transaction.shape[0])
        )

        features_wallets = features_wallets.with_columns(
            mapped_id=pol.arange(0, features_wallets.shape[0])
        )

        wallets_mapping = dict(
            zip(features_wallets["address"], features_wallets["mapped_id"])
        )
        transaction_mapping = dict(
            zip(features_transaction["txId"], features_transaction["mapped_id"])
        )

        edgelist_addr_addr = pol.read_csv(self.downloaded_files[0])
        edgelist_addr_tx = pol.read_csv(self.downloaded_files[1])
        edgelist_tx_addr = pol.read_csv(self.downloaded_files[2])
        edgelist_tx_tx = pol.read_csv(self.downloaded_files[4])

        tx_tx_dict = {"txId1": transaction_mapping, "txId2": transaction_mapping}
        addr_addr_dict = {
            "input_address": wallets_mapping,
            "output_address": wallets_mapping,
        }
        addr_tx_dict = {"input_address": wallets_mapping, "txId": transaction_mapping}
        tx_addr_dict = {"txId": transaction_mapping, "output_address": wallets_mapping}

        addr_tx_edge_index = self.prepare_edge_index(edgelist_addr_tx, addr_tx_dict)
        tx_addr_edge_index = self.prepare_edge_index(edgelist_tx_addr, tx_addr_dict)
        addr_addr_edge_index = self.prepare_edge_index(
            edgelist_addr_addr, addr_addr_dict
        )
        tx_tx_edge_index = self.prepare_edge_index(edgelist_tx_tx, tx_tx_dict)

        # Construct HeteroData object and store it

        df = HeteroData(
            wallets={
                "x": tensor_features_wallets,
                "y": wallet_labels,
                "train_mask": wallet_train_mask,
                "val_mask": wallet_val_mask,
                "test_mask": wallet_test_mask,
            },
            transactions={
                "x": tensor_features_transaction,
                "y": transaction_labels,
                "train_mask": transaction_train_mask,
                "val_mask": transaction_val_mask,
                "test_mask": transaction_test_mask,
            },
        )

        df["transactions", "transactions"].edge_index = tx_tx_edge_index
        df["transactions", "wallets"].edge_index = tx_addr_edge_index
        df["wallets", "transactions"].edge_index = addr_tx_edge_index
        df["wallets", "wallets"].edge_index = addr_addr_edge_index

        self.save([df], self.processed_paths[0])


class EllipticppDataModule(pl.LightningDataModule):
    """
    LightningDataModule for handling the Elliptic++ dataset. The EllipticDataModule class is responsible for loading,
    preprocessing, and preparing the Elliptic++ dataset for use in Lightning Model Modules.
    It handles tasks like splitting the data into training, validation, and testing sets, generating data loaders,
    and implementing neighbor sampling strategies for efficient training.


    Args:
        edge (Tuple[str, str, str]): A tuple representing the edge types in the dataset.
        num_val (float, optional): The ratio of validation data to the total dataset. Defaults to 0.1.
        num_test (float, optional): The ratio of test data to the total dataset. Defaults to 0.1.
        disjoint_train_ratio (float, optional): The ratio of disjoint training data to the total dataset.
                                                Defaults to 0.3.
        neg_sampling_ratio (float, optional): The ratio of negative samples to positive samples. Defaults to 2.0.
        num_neighbors (List[int], optional): A list of integers representing the number of neighbors to consider.
                                             Defaults to [10, 30].
        batch_size (int, optional): The batch size for data loading. Defaults to 128.
        neg_sampling (str, optional): The type of negative sampling. Defaults to "binary".
        num_workers (int, optional): The number of workers for data loading. Defaults to -1.

    """

    def __init__(
        self,
        edge: Tuple[str, str, str],
        num_val: float = 0.1,
        num_test: float = 0.1,
        disjoint_train_ratio: float = 0.3,
        neg_sampling_ratio: float = 2.0,
        num_neighbors: List[int] = None,
        batch_size: int = 128,
        neg_sampling: str = "binary",
        num_workers: int = -1,
        force_reload=False,
    ) -> None:
        super().__init__()

        assert (
            isinstance(edge, tuple) and len(edge) == 3
        ), "edge must be a tuple of length 3"
        assert (
            isinstance(num_val, float) and 0 <= num_val <= 1
        ), "num_val must be a float between 0 and 1"
        assert (
            isinstance(num_test, float) and 0 <= num_test <= 1
        ), "num_test must be a float between 0 and 1"
        assert (
            isinstance(disjoint_train_ratio, float) and 0 <= disjoint_train_ratio <= 1
        ), "disjoint_train_ratio must be a float between 0 and 1"
        assert (
            isinstance(neg_sampling_ratio, float) and neg_sampling_ratio > 0
        ), "neg_sampling_ratio must be a positive float"

        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), "batch_size must be a positive integer"
        assert isinstance(neg_sampling, str), "neg_sampling must be a string"

        if num_neighbors is None:
            num_neighbors = [10, 30]

        assert isinstance(num_neighbors, list) and all(
            isinstance(n, int) for n in num_neighbors
        ), "num_neighbors must be a list of integers"

        self.edge = edge
        self.num_val = num_val
        self.num_test = num_test
        self.disjoint_train_ratio = disjoint_train_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.neg_sampling = neg_sampling
        self.force_reload = force_reload

        if num_workers == -1:
            self.num_workers = multiprocessing.cpu_count() - 1
        else:
            self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = TransactionActorDataset(
            "~/.fintorch_data", force_reload=self.force_reload
        )
        self.dataset = dataset[0]
        self.split_dataset(dataset[0])

    def split_dataset(self, dataset):
        """
        Splits the given dataset into training, validation, and test sets and makes them available
        as self.train_data, self.val_data, and self.test_data

        Args:
            dataset: The input dataset to be split.

        Returns:
            None
        """

        transform = T.RandomLinkSplit(
            num_val=self.num_val,
            num_test=self.num_test,
            disjoint_train_ratio=self.disjoint_train_ratio,
            neg_sampling_ratio=self.neg_sampling_ratio,
            edge_types=[
                ("wallets", "to", "transactions"),
                ("transactions", "to", "wallets"),
                ("wallets", "to", "wallets"),
                ("transactions", "to", "transactions"),
            ],
        )
        self.train_data, self.val_data, self.test_data = transform(dataset)

    def train_dataloader(self):
        """
        Returns a DataLoader object for training.
        """
        src, to, dst = self.edge

        loader = LinkNeighborLoader(
            self.train_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=True,
            edge_label_index=(
                (src, to, dst),
                self.train_data[src, to, dst].edge_index,
            ),
            neg_sampling=self.neg_sampling,
            num_workers=self.num_workers,
        )

        return loader

    def val_dataloader(self):
        """
        Returns a data loader for the validation data.

        Returns:
            loader (LinkNeighborLoader): A data loader for the validation data.
        """
        src, to, dst = self.edge

        loader = LinkNeighborLoader(
            self.val_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            edge_label_index=(
                (src, to, dst),
                self.val_data[src, to, dst].edge_index,
            ),
            neg_sampling=self.neg_sampling,
            num_workers=self.num_workers,
        )

        return loader

    def test_dataloader(self):
        """
        Returns a DataLoader object for testing the dataset.

        Returns:
            DataLoader: A DataLoader object that loads test data for the dataset.
        """
        src, to, dst = self.edge
        loader = LinkNeighborLoader(
            self.test_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            edge_label_index=(
                (src, to, dst),
                self.test_data[src, to, dst].edge_index,
            ),
            neg_sampling=self.neg_sampling,
            num_workers=self.num_workers,
        )

        return loader
