import polars as pl
import torch
from torch_geometric.loader import LinkNeighborLoader

from fintorch.datasets.ellipticpp import EllipticppDataModule, TransactionActorDataset


def test_download():
    dataset = TransactionActorDataset(root="test_data", force_reload=True)
    dataset.download()
    assert len(dataset.downloaded_files) == len(dataset.raw_file_names)


def test_map_classes():
    dataset = TransactionActorDataset(root="test_data")
    df = pl.DataFrame({"class": ["3", "2", "1"]})
    mapped_df = dataset.map_classes(df)
    assert mapped_df["class"].to_list() == [2, 1, 0]


def test_split_data():
    dataset = TransactionActorDataset(root="test_data")
    num_data = 100
    train_mask, val_mask, test_mask = dataset.split_data(num_data, splits=[0.8, 0.1])
    assert train_mask.sum() == 80
    assert val_mask.sum() == 10
    assert test_mask.sum() == 10


def test_prepare_edge_index():
    dataset = TransactionActorDataset(root="test_data")
    edgelist = pl.DataFrame(
        {"input_address": ["a", "b", "c"], "output_address": ["x", "y", "z"]}
    )
    mapping_dict = {
        "input_address": {"a": 0, "b": 1, "c": 2},
        "output_address": {"x": 3, "y": 4, "z": 5},
    }
    edge_index = dataset.prepare_edge_index(edgelist, mapping_dict)
    assert torch.all(torch.eq(edge_index, torch.tensor([[0, 1, 2], [3, 4, 5]])))


def test_process():
    dataset = TransactionActorDataset(root="test_data")
    dataset.process()
    dataset = dataset[0]

    assert dataset["wallets"].x.shape[0] == dataset["wallets"].y.shape[0]
    assert dataset["wallets"].train_mask.shape[0] == dataset["wallets"].y.shape[0]
    assert dataset["wallets"].val_mask.shape[0] == dataset["wallets"].y.shape[0]
    assert dataset["wallets"].test_mask.shape[0] == dataset["wallets"].y.shape[0]
    assert (
        dataset["wallets"].train_mask.sum()
        + dataset["wallets"].val_mask.sum()
        + dataset["wallets"].test_mask.sum()
        == dataset["wallets"].y.shape[0]
    )

    assert dataset["transactions"].x.shape[0] == dataset["transactions"].y.shape[0]
    assert (
        dataset["transactions"].train_mask.shape[0]
        == dataset["transactions"].y.shape[0]
    )
    assert (
        dataset["transactions"].val_mask.shape[0] == dataset["transactions"].y.shape[0]
    )
    assert (
        dataset["transactions"].test_mask.shape[0] == dataset["transactions"].y.shape[0]
    )
    assert (
        dataset["transactions"].train_mask.sum()
        + dataset["transactions"].val_mask.sum()
        + dataset["transactions"].test_mask.sum()
        == dataset["transactions"].y.shape[0]
    )

    assert dataset["wallets", "to", "transactions"]["edge_index"].shape[0] == 2
    assert dataset["transactions", "to", "wallets"]["edge_index"].shape[0] == 2
    assert dataset["wallets", "to", "wallets"]["edge_index"].shape[0] == 2
    assert dataset["transactions", "to", "transactions"]["edge_index"].shape[0] == 2


def test_setup():
    data_module = EllipticppDataModule(edge=("wallets", "to", "transactions"))
    data_module.setup()
    assert data_module.train_data is not None
    assert data_module.val_data is not None
    assert data_module.test_data is not None


def test_train_dataloader():
    data_module = EllipticppDataModule(edge=("wallets", "to", "transactions"))
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    assert train_dataloader is not None


def test_train_dataloader_type():
    data_module = EllipticppDataModule(edge=("wallets", "to", "transactions"))
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    assert isinstance(train_dataloader, LinkNeighborLoader)


def test_val_dataloader_type():
    data_module = EllipticppDataModule(edge=("wallets", "to", "transactions"))
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    assert isinstance(val_dataloader, LinkNeighborLoader)


def test_val_dataloader():
    data_module = EllipticppDataModule(edge=("wallets", "to", "transactions"))
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    assert val_dataloader is not None


def test_link_neighbor_loader_type():
    data_module = EllipticppDataModule(edge=("wallets", "to", "transactions"))
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    assert isinstance(train_dataloader, LinkNeighborLoader)


def test_test_dataloader():
    data_module = EllipticppDataModule(edge=("wallets", "to", "transactions"))
    data_module.setup()
    test_dataloader = data_module.test_dataloader()
    assert test_dataloader is not None


test_download()
test_map_classes()
test_split_data()
test_prepare_edge_index()
test_process()
test_setup()
test_train_dataloader()
test_val_dataloader()
test_test_dataloader()
