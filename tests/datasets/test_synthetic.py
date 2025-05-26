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
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
    )
    assert dataset.length == 100
    assert dataset.trend_slope == 0.2
    assert dataset.seasonality_amplitude == 2.0
    assert dataset.seasonality_period == 20
    assert dataset.noise_level == 0.2
    assert dataset.past_length == 12
    assert dataset.future_length == 6
    assert dataset.num_target_features == 1
    assert dataset.num_known_future_cov_features == 2
    assert dataset.num_unknown_future_cov_features == 1
    assert dataset.num_static_real_features == 2
    assert dataset.num_static_categorical_features == 1
    assert len(dataset.data['target']) == 100
    assert len(dataset) == 100 - 12 - 6


def test_simple_synthetic_dataset_getitem():
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=12,
        future_length=6,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
    )
    sample = dataset[0]

    # Check that we have the expected keys
    expected_keys = {
        "past_target",
        "past_covariates_known_future",
        "past_covariates_unknown_future",
        "future_covariates_known",
        "output_target",
        "static_features_real",
        "static_features_categorical"
    }
    assert set(sample.keys()) == expected_keys

    # Check tensor types
    for key, tensor in sample.items():
        assert isinstance(tensor, torch.Tensor)

    # Check shapes
    assert sample["past_target"].shape == (12, 1, 1)  # (past_length, series_dim, num_target_features)
    assert sample["past_covariates_known_future"].shape == (12, 1, 2)  # (past_length, series_dim, num_known_cov_features)
    assert sample["past_covariates_unknown_future"].shape == (12, 1, 1)  # (past_length, series_dim, num_unknown_cov_features)
    assert sample["future_covariates_known"].shape == (6, 1, 2)  # (future_length, series_dim, num_known_cov_features)
    assert sample["output_target"].shape == (6, 1, 1)  # (future_length, series_dim, num_target_features)
    assert sample["static_features_real"].shape == (1, 2)  # (series_dim, num_static_real_features)
    assert sample["static_features_categorical"].shape == (1, 1)  # (series_dim, num_static_categorical_features)

    # Check dtypes
    assert sample["past_target"].dtype == torch.float32
    assert sample["past_covariates_known_future"].dtype == torch.float32
    assert sample["past_covariates_unknown_future"].dtype == torch.float32
    assert sample["future_covariates_known"].dtype == torch.float32
    assert sample["output_target"].dtype == torch.float32
    assert sample["static_features_real"].dtype == torch.float32
    assert sample["static_features_categorical"].dtype == torch.long


def test_simple_synthetic_dataset_len():
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=12,
        future_length=6,
        num_target_features=1,
    )
    assert len(dataset) == 100 - 12 - 6


def test_simple_synthetic_dataset_multi_series():
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=10,
        future_length=5,
        num_series=3,
        num_target_features=2,
        num_known_cov_features=3,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=2,
        static_categorical_cardinalities=[5, 10]
    )
    sample = dataset[0]

    # Check shapes for multi-series case
    assert sample["past_target"].shape == (10, 3, 2)  # (past_length, series_dim=3, num_target_features=2)
    assert sample["past_covariates_known_future"].shape == (10, 3, 3)
    assert sample["past_covariates_unknown_future"].shape == (10, 3, 1)
    assert sample["future_covariates_known"].shape == (5, 3, 3)
    assert sample["output_target"].shape == (5, 3, 2)
    assert sample["static_features_real"].shape == (3, 2)  # (series_dim=3, num_static_real_features=2)
    assert sample["static_features_categorical"].shape == (3, 2)  # (series_dim=3, num_static_categorical_features=2)

    # Check cardinalities
    assert dataset.static_categorical_cardinalities == [5, 10]


def test_simple_synthetic_dataset_backward_compatibility():
    """Test that legacy parameters still work for backward compatibility."""
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=12,
        future_length=6,
        static_length=3,  # Legacy parameter
        features_dim=4,   # Legacy parameter
        num_series=1,
    )

    # Should still work and map to new parameters appropriately
    assert dataset.static_length == 3
    assert dataset.features_dim == 4
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
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
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
    assert datamodule.num_target_features == 1
    assert datamodule.num_known_cov_features == 2
    assert datamodule.num_unknown_cov_features == 1
    assert datamodule.num_static_real_features == 2
    assert datamodule.num_static_categorical_features == 1
    assert datamodule.workers == 4


def test_simple_synthetic_datamodule_setup():
    datamodule = SimpleSyntheticDataModule(
        train_length=1000,
        val_length=100,
        test_length=100,
        batch_size=32,
        past_length=15,
        future_length=7,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
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
    assert datamodule.train_dataset.num_target_features == 1


def test_simple_synthetic_datamodule_dataloaders():
    datamodule = SimpleSyntheticDataModule(
        train_length=1000,
        val_length=100,
        test_length=100,
        batch_size=32,
        past_length=15,
        future_length=7,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
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


def test_simple_synthetic_dataset_scalers():
    """Test that the scalers work correctly with the new feature organization."""
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=10,
        future_length=5,
        num_series=2,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
    )

    # Test getting scalers for different feature types
    target_scaler = dataset.get_scaler(series_idx=0, feature_type='target', feature_idx=0)
    known_cov_scaler = dataset.get_scaler(series_idx=0, feature_type='known_cov', feature_idx=0)
    unknown_cov_scaler = dataset.get_scaler(series_idx=0, feature_type='unknown_cov', feature_idx=0)

    # Test that scalers exist and are fitted
    assert target_scaler is not None
    assert known_cov_scaler is not None
    assert unknown_cov_scaler is not None

    # Test inverse transform
    sample = dataset[0]
    past_target = sample["past_target"][:, 0, 0]  # Get first series, first feature

    # This should work without errors
    original_scale = dataset.inverse_transform(
        past_target, series_idx=0, feature_type='target', feature_idx=0
    )
    assert original_scale.shape == past_target.shape


def test_dataloader_batching():
    """Test that DataLoader properly batches the new dictionary format."""
    dataset = SimpleSyntheticDataset(
        length=50,
        past_length=10,
        future_length=5,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))

    # Check that batch is a dictionary
    assert isinstance(batch, dict)

    # Check that each tensor has the correct batch dimension
    assert batch["past_target"].shape == (4, 10, 1, 1)  # (batch_size, past_length, series_dim, features)
    assert batch["past_covariates_known_future"].shape == (4, 10, 1, 2)
    assert batch["past_covariates_unknown_future"].shape == (4, 10, 1, 1)
    assert batch["future_covariates_known"].shape == (4, 5, 1, 2)
    assert batch["output_target"].shape == (4, 5, 1, 1)
    assert batch["static_features_real"].shape == (4, 1, 2)
    assert batch["static_features_categorical"].shape == (4, 1, 1)
