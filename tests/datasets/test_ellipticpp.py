import pandas as pd
import torch

from fintorch.datasets.ellipticpp import TransactionActorDataset


def test_download():
    dataset = TransactionActorDataset(root="test_data", force_reload=True)
    dataset.download()
    assert len(dataset.downloaded_files) == len(dataset.raw_file_names)


def test_map_classes():
    dataset = TransactionActorDataset(root="test_data")
    df = pd.DataFrame({"class": ["unknown", "1", "2"]})
    mapped_df = dataset.map_classes(df)
    assert mapped_df["class"].tolist() == [3, 1, 2]


def test_split_data():
    dataset = TransactionActorDataset(root="test_data")
    num_data = 100
    train_mask, val_mask, test_mask = dataset.split_data(num_data,
                                                         splits=[0.8, 0.1])
    assert train_mask.sum() == 80
    assert val_mask.sum() == 10
    assert test_mask.sum() == 10


def test_prepare_edge_index():
    dataset = TransactionActorDataset(root="test_data")
    edgelist = pd.DataFrame({
        "input_address": ["a", "b", "c"],
        "output_address": ["x", "y", "z"]
    })
    mapping_dict = {"a": 0, "b": 1, "c": 2, "x": 3, "y": 4, "z": 5}
    edge_index = dataset.prepare_edge_index(edgelist, mapping_dict)
    assert torch.all(torch.eq(edge_index, torch.tensor([[0, 1, 2], [3, 4,
                                                                    5]])))


def test_process():
    dataset = TransactionActorDataset(root="test_data")
    dataset.process()
    assert dataset.data.wallets.x.shape[0] == dataset.data.wallets.y.shape[0]
    assert dataset.data.wallets.train_mask.shape[
        0] == dataset.data.wallets.y.shape[0]
    assert dataset.data.wallets.val_mask.shape[
        0] == dataset.data.wallets.y.shape[0]
    assert dataset.data.wallets.test_mask.shape[
        0] == dataset.data.wallets.y.shape[0]
    assert dataset.data.wallets.x.shape[1] == dataset.data.wallets.y.max() + 1
    assert (dataset.data.wallets.train_mask.sum() +
            dataset.data.wallets.val_mask.sum() +
            dataset.data.wallets.test_mask.sum() ==
            dataset.data.wallets.y.shape[0])

    assert dataset.data.transactions.x.shape[
        0] == dataset.data.transactions.y.shape[0]
    assert (dataset.data.transactions.train_mask.shape[0] ==
            dataset.data.transactions.y.shape[0])
    assert (dataset.data.transactions.val_mask.shape[0] ==
            dataset.data.transactions.y.shape[0])
    assert (dataset.data.transactions.test_mask.shape[0] ==
            dataset.data.transactions.y.shape[0])
    assert dataset.data.transactions.x.shape[
        1] == dataset.data.transactions.y.max() + 1
    assert (dataset.data.transactions.train_mask.sum() +
            dataset.data.transactions.val_mask.sum() +
            dataset.data.transactions.test_mask.sum() ==
            dataset.data.transactions.y.shape[0])

    assert dataset.data.edge_index_addr_tx.shape[0] == 2
    assert dataset.data.edge_index_tx_addr.shape[0] == 2
    assert dataset.data.edge_index_addr_addr.shape[0] == 2
    assert dataset.data.edge_index_tx_tx.shape[0] == 2


test_download()
test_map_classes()
test_split_data()
test_prepare_edge_index()
test_process()
