import torch
from torch.utils.data import DataLoader

from fintorch.datasets.synthetic.simpleSynthetic import (
    SimpleSyntheticDataModule,
    SimpleSyntheticDataset,
)


def test_simple_synthetic_dataset_initialization():
    dataset = SimpleSyntheticDataset(
        length=100,
        trend_slope=0.2,
        seasonality_amplitude=2.0,
        seasonality_period=20,
        noise_level=0.2,
        past_length=12,
        future_length=6,
        static_length=3,
    )
    assert dataset.length == 100
    assert dataset.trend_slope == 0.2
    assert dataset.seasonality_amplitude == 2.0
    assert dataset.seasonality_period == 20
    assert dataset.noise_level == 0.2
    assert dataset.past_length == 12
    assert dataset.future_length == 6
    assert dataset.static_length == 3
    assert len(dataset.data) == 100
    assert len(dataset) == 100 - 12 - 6


def test_simple_synthetic_dataset_getitem():
    dataset = SimpleSyntheticDataset(
        length=100, past_length=12, future_length=6, static_length=3
    )
    past_inputs, future_inputs, static_inputs, target = dataset[0]

    assert isinstance(past_inputs, dict)
    assert "past_data" in past_inputs
    assert isinstance(past_inputs["past_data"], torch.Tensor)
    assert past_inputs["past_data"].shape == (12, 1)

    assert isinstance(future_inputs, dict)
    assert "future_data" in future_inputs
    assert isinstance(future_inputs["future_data"], torch.Tensor)
    assert future_inputs["future_data"].shape == (6, 1)

    assert isinstance(static_inputs, dict)
    assert "static_data" in static_inputs
    assert isinstance(static_inputs["static_data"], torch.Tensor)
    assert static_inputs["static_data"].shape == (3,)

    assert isinstance(target, torch.Tensor)
    assert target.shape == (6,)


def test_simple_synthetic_dataset_len():
    dataset = SimpleSyntheticDataset(
        length=100, past_length=12, future_length=6, static_length=3
    )
    assert len(dataset) == 100 - 12 - 6


def test_simple_synthetic_datamodule_initialization():
    datamodule = SimpleSyntheticDataModule(
        train_length=1000,
        val_length=100,
        test_length=100,
        batch_size=32,
        trend_slope=0.15,
        seasonality_amplitude=1.5,
        seasonality_period=15,
        noise_level=0.15,
        past_length=15,
        future_length=7,
        static_length=4,
        workers=4,
    )
    assert datamodule.train_length == 1000
    assert datamodule.val_length == 100
    assert datamodule.test_length == 100
    assert datamodule.batch_size == 32
    assert datamodule.trend_slope == 0.15
    assert datamodule.seasonality_amplitude == 1.5
    assert datamodule.seasonality_period == 15
    assert datamodule.noise_level == 0.15
    assert datamodule.past_length == 15
    assert datamodule.future_length == 7
    assert datamodule.static_length == 4
    assert datamodule.workers == 4


def test_simple_synthetic_datamodule_setup():
    datamodule = SimpleSyntheticDataModule(
        train_length=1000,
        val_length=100,
        test_length=100,
        batch_size=32,
        past_length=15,
        future_length=7,
        static_length=4,
    )
    datamodule.setup()
    assert isinstance(datamodule.train_dataset, SimpleSyntheticDataset)
    assert isinstance(datamodule.val_dataset, SimpleSyntheticDataset)
    assert isinstance(datamodule.test_dataset, SimpleSyntheticDataset)
    assert datamodule.train_dataset.length == 1000
    assert datamodule.val_dataset.length == 100
    assert datamodule.test_dataset.length == 100
    assert datamodule.train_dataset.past_length == 15
    assert datamodule.train_dataset.future_length == 7
    assert datamodule.train_dataset.static_length == 4


def test_simple_synthetic_datamodule_dataloaders():
    datamodule = SimpleSyntheticDataModule(
        train_length=1000,
        val_length=100,
        test_length=100,
        batch_size=32,
        past_length=15,
        future_length=7,
        static_length=4,
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    predict_dataloader = datamodule.predict_dataloader()

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(val_dataloader, DataLoader)
    assert isinstance(test_dataloader, DataLoader)
    assert isinstance(predict_dataloader, DataLoader)

    assert train_dataloader.batch_size == 32
    assert val_dataloader.batch_size == 32
    assert test_dataloader.batch_size == 32
    assert predict_dataloader.batch_size == 32
