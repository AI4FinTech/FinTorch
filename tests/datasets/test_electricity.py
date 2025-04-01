from torch.utils.data import DataLoader
from fintorch.datasets.electricity_simple import (
    ElectricityDataset,
    ElectricityDataModule,
)


def test_electricity_dataset_length():
    dataset = ElectricityDataset(past_length=10, future_length=5)
    expected_length = dataset.length - 10 - 5
    assert len(dataset) == expected_length, "Dataset length mismatch"


def test_electricity_dataset_getitem():
    dataset = ElectricityDataset(past_length=10, future_length=5)
    past_inputs, target = dataset[0]

    # Check past_inputs structure
    assert "past_data" in past_inputs, "Missing 'past_data' in past_inputs"
    assert past_inputs["past_data"].shape == (10, 1), "Past data shape mismatch"

    # Check target structure
    assert target.shape == (5,), "Target shape mismatch"


def test_electricity_dataset_slice():
    dataset = ElectricityDataset(past_length=10, future_length=5)
    sliced_data = dataset[:5]

    assert len(sliced_data) == 5, "Sliced dataset length mismatch"
    for past_inputs, target in sliced_data:
        assert "past_data" in past_inputs, "Missing 'past_data' in sliced past_inputs"
        assert past_inputs["past_data"].shape == (
            10,
            1,
        ), "Sliced past data shape mismatch"
        assert target.shape == (5,), "Sliced target shape mismatch"


def test_electricity_data_module_setup():
    data_module = ElectricityDataModule(batch_size=32, past_length=10, horizon=5)
    data_module.setup()

    assert len(data_module.train_dataset) > 0, "Train dataset is empty"
    assert len(data_module.val_dataset) > 0, "Validation dataset is empty"
    assert len(data_module.test_dataset) > 0, "Test dataset is empty"


def test_electricity_data_module_dataloaders():
    data_module = ElectricityDataModule(batch_size=32, past_length=10, horizon=5)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    assert isinstance(
        train_loader, DataLoader
    ), "Train dataloader is not a DataLoader instance"
    assert isinstance(
        val_loader, DataLoader
    ), "Validation dataloader is not a DataLoader instance"
    assert isinstance(
        test_loader, DataLoader
    ), "Test dataloader is not a DataLoader instance"

    # Check batch size
    for batch in train_loader:
        past_inputs, target = batch
        assert past_inputs["past_data"].shape[0] <= 32, "Train batch size mismatch"
        break
