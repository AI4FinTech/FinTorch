import pytest
from torch.utils.data import DataLoader

from fintorch.datasets.elliptic import EllipticDataset, LightningEllipticDataModule


@pytest.fixture
def sample_data():
    # Generate sample data for testing
    return [1, 2, 3, 4, 5]


def test_elliptic_dataset_length(sample_data):
    dataset = EllipticDataset(sample_data, download=False)
    assert len(dataset) == len(sample_data)


def test_elliptic_dataset_getitem(sample_data):
    dataset = EllipticDataset(sample_data, download=False)
    assert dataset[0] == sample_data[0]
    assert dataset[2] == sample_data[2]
    assert dataset[4] == sample_data[4]


def test_elliptic_dataset_download_data(mocker):
    dataset = EllipticDataset([], download=True)
    mocker.patch.object(dataset, "download_data")
    dataset.__init__([])
    dataset.download_data.assert_called_once()


@pytest.fixture
def sample_train_data():
    # Generate sample train data for testing
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_val_data():
    # Generate sample validation data for testing
    return [6, 7, 8, 9, 10]


@pytest.fixture
def sample_test_data():
    # Generate sample test data for testing
    return [11, 12, 13, 14, 15]


def test_lightning_elliptic_data_module_setup(
    sample_train_data, sample_val_data, sample_test_data
):
    data_module = LightningEllipticDataModule(
        sample_train_data, sample_val_data, sample_test_data
    )
    data_module.setup()

    assert isinstance(data_module.train_dataset, EllipticDataset)
    assert isinstance(data_module.val_dataset, EllipticDataset)
    assert isinstance(data_module.test_dataset, EllipticDataset)


def test_lightning_elliptic_data_module_train_dataloader(
    sample_train_data, sample_val_data, sample_test_data
):
    data_module = LightningEllipticDataModule(
        sample_train_data, sample_val_data, sample_test_data
    )
    data_module.setup()

    train_dataloader = data_module.train_dataloader()

    assert isinstance(train_dataloader, DataLoader)
    assert len(train_dataloader.dataset) == len(sample_train_data)
    assert train_dataloader.batch_size == data_module.batch_size
    assert train_dataloader.shuffle


def test_lightning_elliptic_data_module_val_dataloader(
    sample_train_data, sample_val_data, sample_test_data
):
    data_module = LightningEllipticDataModule(
        sample_train_data, sample_val_data, sample_test_data
    )
    data_module.setup()

    val_dataloader = data_module.val_dataloader()

    assert isinstance(val_dataloader, DataLoader)
    assert len(val_dataloader.dataset) == len(sample_val_data)
    assert val_dataloader.batch_size == data_module.batch_size
    assert not val_dataloader.shuffle


def test_lightning_elliptic_data_module_test_dataloader(
    sample_train_data, sample_val_data, sample_test_data
):
    data_module = LightningEllipticDataModule(
        sample_train_data, sample_val_data, sample_test_data
    )
    data_module.setup()

    test_dataloader = data_module.test_dataloader()

    assert isinstance(test_dataloader, DataLoader)
    assert len(test_dataloader.dataset) == len(sample_test_data)
    assert test_dataloader.batch_size == data_module.batch_size
    assert not test_dataloader.shuffle
