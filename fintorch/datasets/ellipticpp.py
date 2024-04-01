from typing import Callable, List, Optional

import numpy as np
import polars as pol
import torch
from huggingface_hub import hf_hub_download
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm


class TransactionActorDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root,
                         transform,
                         pre_transform,
                         pre_filter,
                         force_reload=force_reload)

        self.load(self.processed_paths[0])
        data = self.get(0)
        self.data, self.slices = self.collate([data])

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
        return df.with_columns(
            pol.col("class").cast(pol.Utf8).map_elements(lambda x: {
                "unknown": 3,
                "1": 1,
                "2": 2
            }.get(x)))

    def split_data(self, num_data, splits=[0.8, 0.1]):
        """
        Splits the data into training, validation, and test sets based on the given splits.

        Args:
            num_data (int): The total number of data points.
            splits (list, optional): The split ratios for training, validation, and test sets.
                Defaults to [0.8, 0.1].

        Returns:
            tuple: A tuple containing the masks for the training, validation, and test sets.
        """

        assert len(splits) == 2, "The length of splits should be 2"
        assert sum(splits) < 1, "The sum of splits should be less than 1"

        # Generate numbers
        num_train = int(splits[0] * num_data)
        num_val = int(splits[1] * num_data)
        num_test = num_data - num_train - num_val

        # Generate ranges
        train_index = torch.arange(num_train, dtype=torch.long)
        val_index = torch.arange(num_train,
                                 num_train + num_val,
                                 dtype=torch.long)
        test_index = torch.arange(num_train + num_val,
                                  num_train + num_val + num_test,
                                  dtype=torch.long)

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
            edgelist = edgelist.replace(
                k,
                edgelist[k].apply(lambda x: v.get(x, x)),
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
        transaction_classes = self.map_classes(
            pol.read_csv(self.downloaded_files[3]))
        features_transaction = pol.read_csv(self.downloaded_files[5])

        features_transaction = features_transaction.join(transaction_classes,
                                                         how="left",
                                                         left_on="txId",
                                                         right_on="txId")

        # Load wallets features and classes into Polars DataFrame
        features_wallets = pol.read_csv(self.downloaded_files[6])

        # Prepare labels
        transaction_labels = torch.tensor(
            features_transaction["class"].to_numpy(),
            dtype=torch.float32).long()
        wallet_labels = torch.tensor(features_wallets["class"].to_numpy(),
                                     dtype=torch.float32).long()

        # Mask data
        transaction_train_mask, transaction_val_mask, transaction_test_mask = (
            self.split_data(features_transaction.shape[0]))

        wallet_train_mask, wallet_val_mask, wallet_test_mask = self.split_data(
            features_wallets.shape[0])

        # PyTorch features
        tensor_features_wallets = torch.tensor(
            features_wallets.drop(["class", "Time step",
                                   "address"]).to_numpy(),
            dtype=torch.float32,
        )

        tensor_features_transaction = torch.tensor(
            features_transaction.drop(["class", "Time step",
                                       "txId"]).to_numpy(),
            dtype=torch.float32,
        )

        # Prepare edge_index data

        # Replace nodes with indices for transactions and wallets
        features_transaction = features_transaction.with_columns(
            mapped_id=pol.arange(0, features_transaction.shape[0]))

        features_wallets = features_wallets.with_columns(
            mapped_id=pol.arange(0, features_wallets.shape[0]))

        wallets_mapping = dict(
            zip(features_wallets["address"], features_wallets["mapped_id"]))
        transaction_mapping = dict(
            zip(features_transaction["txId"],
                features_transaction["mapped_id"]))

        edgelist_addr_addr = pol.read_csv(self.downloaded_files[0])
        edgelist_addr_tx = pol.read_csv(self.downloaded_files[1])
        edgelist_tx_addr = pol.read_csv(self.downloaded_files[2])
        edgelist_tx_tx = pol.read_csv(self.downloaded_files[4])

        tx_tx_dict = {
            "txId1": transaction_mapping,
            "txId2": transaction_mapping
        }
        addr_addr_dict = {
            "input_address": wallets_mapping,
            "output_address": wallets_mapping,
        }
        addr_tx_dict = {
            "input_address": wallets_mapping,
            "txId": transaction_mapping
        }
        tx_addr_dict = {
            "txId": transaction_mapping,
            "output_address": wallets_mapping
        }

        addr_tx_edge_index = self.prepare_edge_index(edgelist_addr_tx,
                                                     addr_tx_dict)
        tx_addr_edge_index = self.prepare_edge_index(edgelist_tx_addr,
                                                     tx_addr_dict)
        addr_addr_edge_index = self.prepare_edge_index(edgelist_addr_addr,
                                                       addr_addr_dict)
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

        df["transaction", "transaction"].edge_index = tx_tx_edge_index
        df["transaction", "address"].edge_index = tx_addr_edge_index
        df["address", "transaction"].edge_index = addr_tx_edge_index
        df["address", "address"].edge_index = addr_addr_edge_index

        self.save([df], self.processed_paths[0])