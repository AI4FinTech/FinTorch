import torch
from torch.utils.data import DataLoader

from fintorch.datasets.synthetic.simpleSynthetic import (
    SimpleSyntheticDataModule,
    SimpleSyntheticDataset,
    custom_collate_fn,
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
    """Test that the dataset works with granular parameters and legacy properties are accessible."""
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=12,
        future_length=6,
        num_static_real_features=2,
        num_static_categorical_features=1,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_series=1,
    )

    # Legacy properties should still be accessible and calculated correctly
    assert dataset.static_length == 3  # 2 real + 1 categorical
    assert dataset.features_dim == 4   # 1 target + 2 known + 1 unknown
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


def test_custom_collate_fn_basic():
    """Test basic functionality of custom_collate_fn."""
    # Create mock batch data
    batch = [
        {
            "past_target": torch.randn(10, 1, 1),
            "past_covariates_known_future": torch.randn(10, 1, 2),
            "future_covariates_known": torch.randn(5, 1, 2),
            "output_target": torch.randn(5, 1, 1),
            "static_features_real": torch.randn(1, 2),
        },
        {
            "past_target": torch.randn(10, 1, 1),
            "past_covariates_known_future": torch.randn(10, 1, 2),
            "future_covariates_known": torch.randn(5, 1, 2),
            "output_target": torch.randn(5, 1, 1),
            "static_features_real": torch.randn(1, 2),
        },
        {
            "past_target": torch.randn(10, 1, 1),
            "past_covariates_known_future": torch.randn(10, 1, 2),
            "future_covariates_known": torch.randn(5, 1, 2),
            "output_target": torch.randn(5, 1, 1),
            "static_features_real": torch.randn(1, 2),
        },
    ]

    collated = custom_collate_fn(batch)

    # Check that all keys are present
    expected_keys = {
        "past_target", "past_covariates_known_future",
        "future_covariates_known", "output_target", "static_features_real"
    }
    assert set(collated.keys()) == expected_keys

    # Check that tensors are properly stacked (batch dimension added)
    assert collated["past_target"].shape == (3, 10, 1, 1)  # (batch_size, time, series, features)
    assert collated["past_covariates_known_future"].shape == (3, 10, 1, 2)
    assert collated["future_covariates_known"].shape == (3, 5, 1, 2)
    assert collated["output_target"].shape == (3, 5, 1, 1)
    assert collated["static_features_real"].shape == (3, 1, 2)


def test_custom_collate_fn_with_none_values():
    """Test custom_collate_fn handles None values correctly."""
    batch = [
        {
            "past_target": torch.randn(10, 1, 1),
            "optional_feature": torch.randn(5, 1, 2),
            "static_features_real": torch.randn(1, 2),
        },
        {
            "past_target": torch.randn(10, 1, 1),
            "optional_feature": None,  # This item has None for optional_feature
            "static_features_real": torch.randn(1, 2),
        },
        {
            "past_target": torch.randn(10, 1, 1),
            "optional_feature": torch.randn(5, 1, 2),
            "static_features_real": torch.randn(1, 2),
        },
    ]

    collated = custom_collate_fn(batch)

    # past_target and static_features_real should be stacked normally
    assert collated["past_target"].shape == (3, 10, 1, 1)
    assert collated["static_features_real"].shape == (3, 1, 2)

    # optional_feature should stack only the non-None tensors
    assert collated["optional_feature"].shape == (2, 5, 1, 2)  # Only 2 items instead of 3


def test_custom_collate_fn_all_none_values():
    """Test custom_collate_fn when all values for a key are None."""
    batch = [
        {
            "past_target": torch.randn(10, 1, 1),
            "optional_feature": None,
            "static_features_real": torch.randn(1, 2),
        },
        {
            "past_target": torch.randn(10, 1, 1),
            "optional_feature": None,
            "static_features_real": torch.randn(1, 2),
        },
    ]

    collated = custom_collate_fn(batch)

    # past_target and static_features_real should be stacked normally
    assert collated["past_target"].shape == (2, 10, 1, 1)
    assert collated["static_features_real"].shape == (2, 1, 2)

    # optional_feature should be None since all values were None
    assert collated["optional_feature"] is None


def test_custom_collate_fn_empty_batch():
    """Test custom_collate_fn with empty batch (edge case)."""
    batch = []

    try:
        collated = custom_collate_fn(batch)
        # If it doesn't raise an error, the result should be an empty dict
        assert collated == {}
    except IndexError:
        # It's acceptable for the function to raise IndexError on empty batch
        pass


def test_custom_collate_fn_single_item():
    """Test custom_collate_fn with single item batch."""
    batch = [
        {
            "past_target": torch.randn(10, 1, 1),
            "past_covariates_known_future": torch.randn(10, 1, 2),
            "static_features_real": torch.randn(1, 2),
        }
    ]

    collated = custom_collate_fn(batch)

    # Should add batch dimension of 1
    assert collated["past_target"].shape == (1, 10, 1, 1)
    assert collated["past_covariates_known_future"].shape == (1, 10, 1, 2)
    assert collated["static_features_real"].shape == (1, 1, 2)


def test_custom_collate_fn_different_tensor_types():
    """Test custom_collate_fn with different tensor dtypes."""
    batch = [
        {
            "float_tensor": torch.randn(5, 2).float(),
            "long_tensor": torch.randint(0, 10, (3,)).long(),
            "bool_tensor": torch.tensor([True, False]).bool(),
        },
        {
            "float_tensor": torch.randn(5, 2).float(),
            "long_tensor": torch.randint(0, 10, (3,)).long(),
            "bool_tensor": torch.tensor([False, True]).bool(),
        },
    ]

    collated = custom_collate_fn(batch)

    # Check shapes
    assert collated["float_tensor"].shape == (2, 5, 2)
    assert collated["long_tensor"].shape == (2, 3)
    assert collated["bool_tensor"].shape == (2, 2)

    # Check dtypes are preserved
    assert collated["float_tensor"].dtype == torch.float32
    assert collated["long_tensor"].dtype == torch.int64
    assert collated["bool_tensor"].dtype == torch.bool


def test_custom_collate_fn_with_dataloader():
    """Test that custom_collate_fn works correctly with DataLoader."""
    dataset = SimpleSyntheticDataset(
        length=50,
        past_length=10,
        future_length=5,
        num_target_features=1,
        num_known_cov_features=2,
        num_static_real_features=2,
    )

    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    batch = next(iter(dataloader))

    # Check that batch is properly collated
    assert isinstance(batch, dict)
    assert batch["past_target"].shape == (4, 10, 1, 1)
    assert batch["past_covariates_known_future"].shape == (4, 10, 1, 2)
    assert batch["future_covariates_known"].shape == (4, 5, 1, 2)
    assert batch["output_target"].shape == (4, 5, 1, 1)
    assert batch["static_features_real"].shape == (4, 1, 2)


def test_custom_collate_fn_preserves_gradients():
    """Test that custom_collate_fn preserves gradient information."""
    batch = [
        {
            "tensor_with_grad": torch.randn(3, 2, requires_grad=True),
        },
        {
            "tensor_with_grad": torch.randn(3, 2, requires_grad=True),
        },
    ]

    collated = custom_collate_fn(batch)

    # Check that gradient requirement is preserved
    assert collated["tensor_with_grad"].requires_grad is True
    assert collated["tensor_with_grad"].shape == (2, 3, 2)
