import os

import pytest

from fintorch.datasets.elliptic import TransactionDataset


@pytest.fixture
def dataset():
    root = "/tmp/data/fintorch/"
    return TransactionDataset(root, force_reload=True)


def test_raw_file_names(dataset):
    expected_files = [
        "elliptic_bitcoin_dataset/elliptic_txs_features.csv",
        "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
        "elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
    ]
    assert dataset.raw_file_names == expected_files


def test_processed_file_names(dataset):
    expected_files = ["transaction_graph_v1.pt"]
    assert dataset.processed_file_names == expected_files


def test_download(dataset):
    dataset.download()
    assert os.path.exists(dataset.raw_dir)


def test_process(dataset):
    dataset.process()
    assert os.path.exists(dataset.processed_paths[0])


def test_data_loading(dataset):
    data = dataset[0]
    assert "x" in data
    assert "edge_index" in data
    assert "y" in data
    assert "train_mask" in data
    assert "val_mask" in data
    assert "test_mask" in data
