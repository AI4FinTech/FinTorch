"""
Comprehensive test suite for the AirPassenger dataset module.

This test file focuses extensively on testing the `custom_collate_fn` function,
which is responsible for batching time-series data for the AirPassenger dataset.

Test Coverage:
--------------
1. custom_collate_fn tests:
   - Basic functionality with multiple batch items
   - Single item batching
   - Different tensor data types preservation
   - Gradient information preservation
   - Empty batch error handling
   - Variable sequence length error handling
   - Static data handling
   - List accumulation logic
   - Tensor stacking behavior
   - Dictionary structure maintenance
   - Type annotation compliance

2. AirPassengerDataset tests:
   - Dataset initialization
   - __getitem__ method functionality
   - Data format and structure validation

3. AirPassengerDataModule tests:
   - DataModule initialization
   - Setup method functionality
   - DataLoader creation and configuration
   - Integration with custom_collate_fn

The custom_collate_fn function performs the following key operations:
1. Takes a list of dictionaries (batch items) with standardized keys
2. Stacks tensors for each key across the batch dimension
3. Handles None values appropriately
4. Returns a single dictionary with batched tensors
"""

import torch
import pytest

from fintorch.datasets.airpassenger import (
    custom_collate_fn,
    AirPassengerDataset,
    AirPassengerDataModule,
)


def test_custom_collate_fn_basic():
    """Test basic functionality of custom_collate_fn with valid batch data."""
    # Create mock batch data matching the new dictionary format
    batch_item_1 = {
        "past_target": torch.randn(10, 1, 1),
        "past_covariates_known_future": torch.randn(10, 1, 2),
        "past_covariates_unknown_future": torch.randn(10, 1, 1),
        "future_covariates_known": torch.randn(5, 1, 2),
        "output_target": torch.randn(5, 1, 1),
        "static_features_real": torch.randn(1, 2),
        "static_features_categorical": torch.tensor([[0]], dtype=torch.long),
    }

    batch_item_2 = {
        "past_target": torch.randn(10, 1, 1),
        "past_covariates_known_future": torch.randn(10, 1, 2),
        "past_covariates_unknown_future": torch.randn(10, 1, 1),
        "future_covariates_known": torch.randn(5, 1, 2),
        "output_target": torch.randn(5, 1, 1),
        "static_features_real": torch.randn(1, 2),
        "static_features_categorical": torch.tensor([[1]], dtype=torch.long),
    }

    batch_item_3 = {
        "past_target": torch.randn(10, 1, 1),
        "past_covariates_known_future": torch.randn(10, 1, 2),
        "past_covariates_unknown_future": torch.randn(10, 1, 1),
        "future_covariates_known": torch.randn(5, 1, 2),
        "output_target": torch.randn(5, 1, 1),
        "static_features_real": torch.randn(1, 2),
        "static_features_categorical": torch.tensor([[0]], dtype=torch.long),
    }

    batch = [batch_item_1, batch_item_2, batch_item_3]

    collated_batch = custom_collate_fn(batch)

    # Check return type
    assert isinstance(collated_batch, dict)

    # Check dictionary keys
    expected_keys = {
        "past_target", "past_covariates_known_future", "past_covariates_unknown_future",
        "future_covariates_known", "output_target", "static_features_real", "static_features_categorical"
    }
    assert set(collated_batch.keys()) == expected_keys

    # Check tensor shapes - should have batch dimension added
    assert collated_batch["past_target"].shape == (3, 10, 1, 1)  # (batch, time, series, features)
    assert collated_batch["past_covariates_known_future"].shape == (3, 10, 1, 2)
    assert collated_batch["past_covariates_unknown_future"].shape == (3, 10, 1, 1)
    assert collated_batch["future_covariates_known"].shape == (3, 5, 1, 2)
    assert collated_batch["output_target"].shape == (3, 5, 1, 1)
    assert collated_batch["static_features_real"].shape == (3, 1, 2)
    assert collated_batch["static_features_categorical"].shape == (3, 1, 1)

    # Check that the data is correctly stacked
    assert torch.equal(collated_batch["past_target"][0], batch_item_1["past_target"])
    assert torch.equal(collated_batch["past_target"][1], batch_item_2["past_target"])
    assert torch.equal(collated_batch["past_target"][2], batch_item_3["past_target"])

    # Check data types are preserved
    assert collated_batch["past_target"].dtype == torch.float32
    assert collated_batch["static_features_categorical"].dtype == torch.long


def test_custom_collate_fn_single_item():
    """Test custom_collate_fn with a single batch item."""
    batch_item = {
        "past_target": torch.randn(8, 1, 1),
        "past_covariates_known_future": torch.randn(8, 1, 2),
        "past_covariates_unknown_future": torch.randn(8, 1, 1),
        "future_covariates_known": torch.randn(3, 1, 2),
        "output_target": torch.randn(3, 1, 1),
        "static_features_real": torch.randn(1, 2),
        "static_features_categorical": torch.tensor([[1]], dtype=torch.long),
    }

    batch = [batch_item]

    collated_batch = custom_collate_fn(batch)

    # Check return type
    assert isinstance(collated_batch, dict)

    # Check tensor shapes - should have batch dimension of 1
    assert collated_batch["past_target"].shape == (1, 8, 1, 1)
    assert collated_batch["past_covariates_known_future"].shape == (1, 8, 1, 2)
    assert collated_batch["output_target"].shape == (1, 3, 1, 1)
    assert collated_batch["static_features_real"].shape == (1, 1, 2)

    # Check that the data is correctly preserved
    assert torch.equal(collated_batch["past_target"][0], batch_item["past_target"])


def test_custom_collate_fn_different_tensor_types():
    """Test that custom_collate_fn preserves different tensor data types."""
    batch_item_1 = {
        "past_target": torch.randn(5, 1, 1).float(),
        "static_features_categorical": torch.tensor([[0]], dtype=torch.long),
        "output_target": torch.randn(3, 1, 1).double(),
    }

    batch_item_2 = {
        "past_target": torch.randn(5, 1, 1).float(),
        "static_features_categorical": torch.tensor([[1]], dtype=torch.long),
        "output_target": torch.randn(3, 1, 1).double(),
    }

    batch = [batch_item_1, batch_item_2]

    collated_batch = custom_collate_fn(batch)

    # Check that data types are preserved
    assert collated_batch["past_target"].dtype == torch.float32
    assert collated_batch["static_features_categorical"].dtype == torch.long
    assert collated_batch["output_target"].dtype == torch.float64


def test_custom_collate_fn_preserves_gradients():
    """Test that custom_collate_fn preserves gradient information."""
    tensor_with_grad = torch.randn(5, 1, 1, requires_grad=True)

    batch_item = {
        "past_target": tensor_with_grad,
        "output_target": torch.randn(3, 1, 1),
    }

    batch = [batch_item, batch_item]

    collated_batch = custom_collate_fn(batch)

    # Check that gradient information is preserved
    assert collated_batch["past_target"].requires_grad is True
    assert collated_batch["output_target"].requires_grad is False


def test_custom_collate_fn_empty_batch():
    """Test custom_collate_fn behavior with empty batch."""
    batch = []

    # Should handle empty batch gracefully
    with pytest.raises(IndexError):
        custom_collate_fn(batch)


def test_custom_collate_fn_variable_sequence_lengths():
    """Test custom_collate_fn with variable sequence lengths (should fail)."""
    batch_item_1 = {
        "past_target": torch.randn(10, 1, 1),
        "output_target": torch.randn(5, 1, 1),
    }

    batch_item_2 = {
        "past_target": torch.randn(8, 1, 1),  # Different length
        "output_target": torch.randn(5, 1, 1),
    }

    batch = [batch_item_1, batch_item_2]

    # Should raise an error when trying to stack tensors of different sizes
    with pytest.raises(RuntimeError):
        custom_collate_fn(batch)


def test_custom_collate_fn_static_data_handling():
    """Test custom_collate_fn handles static data correctly."""
    batch_item_1 = {
        "static_features_real": torch.randn(1, 2),
        "static_features_categorical": torch.tensor([[0]], dtype=torch.long),
        "past_target": torch.randn(5, 1, 1),
    }

    batch_item_2 = {
        "static_features_real": torch.randn(1, 2),
        "static_features_categorical": torch.tensor([[1]], dtype=torch.long),
        "past_target": torch.randn(5, 1, 1),
    }

    batch = [batch_item_1, batch_item_2]

    collated_batch = custom_collate_fn(batch)

    # Check that static features are properly batched
    assert collated_batch["static_features_real"].shape == (2, 1, 2)
    assert collated_batch["static_features_categorical"].shape == (2, 1, 1)


def test_air_passenger_dataset_initialization():
    """Test AirPassengerDataset initialization."""
    dataset = AirPassengerDataset(past_length=12, future_length=6)

    assert dataset.past_length == 12
    assert dataset.future_length == 6
    assert len(dataset) > 0


def test_air_passenger_dataset_getitem():
    """Test AirPassengerDataset __getitem__ method."""
    dataset = AirPassengerDataset(past_length=10, future_length=5)

    # Get a single item
    sample = dataset[0]

    # Check return type
    assert isinstance(sample, dict)

    # Check dictionary keys
    expected_keys = {
        "past_target", "past_covariates_known_future", "past_covariates_unknown_future",
        "future_covariates_known", "output_target", "static_features_real", "static_features_categorical"
    }
    assert set(sample.keys()) == expected_keys

    # Check tensor shapes
    assert sample["past_target"].shape == (10, 1, 1)  # (time, series, features)
    assert sample["past_covariates_known_future"].shape == (10, 1, 2)  # month + trend
    assert sample["past_covariates_unknown_future"].shape == (10, 1, 1)  # placeholder
    assert sample["future_covariates_known"].shape == (5, 1, 2)  # month + trend
    assert sample["output_target"].shape == (5, 1, 1)
    assert sample["static_features_real"].shape == (1, 2)
    assert sample["static_features_categorical"].shape == (1, 1)

    # Check data types
    assert sample["past_target"].dtype == torch.float32
    assert sample["past_covariates_known_future"].dtype == torch.float32
    assert sample["output_target"].dtype == torch.float32
    assert sample["static_features_categorical"].dtype == torch.long


def test_air_passenger_datamodule_initialization():
    """Test AirPassengerDataModule initialization."""
    datamodule = AirPassengerDataModule(
        batch_size=32,
        past_length=15,
        horizon=7,
        workers=2
    )

    assert datamodule.batch_size == 32
    assert datamodule.past_length == 15
    assert datamodule.future_length == 7
    assert datamodule.workers == 2


def test_air_passenger_datamodule_setup():
    """Test AirPassengerDataModule setup method."""
    datamodule = AirPassengerDataModule(batch_size=16, past_length=8, horizon=4)

    # Test setup for fit stage
    datamodule.setup(stage="fit")
    assert hasattr(datamodule, 'train_dataset')
    assert hasattr(datamodule, 'val_dataset')

    # Test setup for test stage
    datamodule.setup(stage="test")
    assert hasattr(datamodule, 'test_dataset')


def test_air_passenger_datamodule_dataloaders():
    """Test AirPassengerDataModule dataloader methods."""
    datamodule = AirPassengerDataModule(batch_size=4, past_length=6, horizon=3)
    datamodule.setup()

    # Test train dataloader
    train_loader = datamodule.train_dataloader()
    assert train_loader.batch_size == 4
    assert train_loader.collate_fn == custom_collate_fn

    # Test validation dataloader
    val_loader = datamodule.val_dataloader()
    assert val_loader.batch_size == 4
    assert val_loader.collate_fn == custom_collate_fn

    # Test test dataloader
    test_loader = datamodule.test_dataloader()
    assert test_loader.batch_size == 4
    assert val_loader.collate_fn == custom_collate_fn


def test_dataloader_with_custom_collate_fn():
    """Test that DataLoader works correctly with custom_collate_fn."""
    dataset = AirPassengerDataset(past_length=8, future_length=4, start_idx=0, end_idx=10)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Get a batch
    batch = next(iter(dataloader))

    # Check batch structure
    assert isinstance(batch, dict)

    # Check dictionary keys
    expected_keys = {
        "past_target", "past_covariates_known_future", "past_covariates_unknown_future",
        "future_covariates_known", "output_target", "static_features_real", "static_features_categorical"
    }
    assert set(batch.keys()) == expected_keys

    # Check batch dimensions
    assert batch["past_target"].shape[0] == 3  # batch size
    assert batch["future_covariates_known"].shape[0] == 3
    assert batch["output_target"].shape[0] == 3

    # Check sequence dimensions
    assert batch["past_target"].shape[1] == 8  # past_length
    assert batch["future_covariates_known"].shape[1] == 4  # future_length
    assert batch["output_target"].shape[1] == 4


def test_custom_collate_fn_type_annotations():
    """Test that custom_collate_fn handles type annotations correctly."""
    batch_item = {
        "past_target": torch.randn(5, 1, 1),
        "past_covariates_known_future": torch.randn(5, 1, 2),
        "output_target": torch.randn(3, 1, 1),
        "static_features_real": torch.randn(1, 2),
        "static_features_categorical": torch.tensor([[0]], dtype=torch.long),
    }

    batch = [batch_item, batch_item]

    # Function should work without type annotation issues
    collated_batch = custom_collate_fn(batch)

    assert isinstance(collated_batch, dict)
    assert len(collated_batch) == 5


def test_custom_collate_fn_list_accumulation():
    """Test custom_collate_fn list accumulation logic."""
    batch_items = []

    for i in range(4):
        batch_items.append({
            "past_target": torch.randn(6, 1, 1) + i,  # Add offset to distinguish
            "output_target": torch.randn(2, 1, 1) + i,
            "static_features_real": torch.randn(1, 2) + i,
        })

    collated_batch = custom_collate_fn(batch_items)

    # Check that all items are properly accumulated
    assert collated_batch["past_target"].shape[0] == 4
    assert collated_batch["output_target"].shape[0] == 4
    assert collated_batch["static_features_real"].shape[0] == 4

    # Check that order is preserved (approximate check due to random values)
    # The mean should increase with the offset we added
    means = [collated_batch["past_target"][i].mean().item() for i in range(4)]
    assert means[1] > means[0]
    assert means[2] > means[1]
    assert means[3] > means[2]


def test_custom_collate_fn_stacking_behavior():
    """Test custom_collate_fn tensor stacking behavior."""
    # Create distinguishable tensors
    tensor_1 = torch.ones(3, 1, 1)
    tensor_2 = torch.ones(3, 1, 1) * 2
    tensor_3 = torch.ones(3, 1, 1) * 3

    batch = [
        {"past_target": tensor_1, "output_target": torch.randn(2, 1, 1)},
        {"past_target": tensor_2, "output_target": torch.randn(2, 1, 1)},
        {"past_target": tensor_3, "output_target": torch.randn(2, 1, 1)},
    ]

    collated_batch = custom_collate_fn(batch)

    # Check that stacking preserves the individual tensors
    assert torch.equal(collated_batch["past_target"][0], tensor_1)
    assert torch.equal(collated_batch["past_target"][1], tensor_2)
    assert torch.equal(collated_batch["past_target"][2], tensor_3)

    # Check that the batch dimension is correctly added
    assert collated_batch["past_target"].dim() == 4  # batch + original 3 dims
    assert collated_batch["past_target"].shape[0] == 3  # batch size


def test_custom_collate_fn_dictionary_structure():
    """Test custom_collate_fn preserves dictionary structure."""
    batch_item = {
        "past_target": torch.randn(4, 1, 1),
        "past_covariates_known_future": torch.randn(4, 1, 2),
        "future_covariates_known": torch.randn(2, 1, 2),
        "output_target": torch.randn(2, 1, 1),
        "static_features_real": torch.randn(1, 2),
    }

    batch = [batch_item, batch_item]

    collated_batch = custom_collate_fn(batch)

    # Check that all keys are preserved
    assert set(collated_batch.keys()) == set(batch_item.keys())

    # Check that each value is a tensor (not None for available keys)
    for key, value in collated_batch.items():
        assert isinstance(value, torch.Tensor), f"Key {key} should be a tensor"


def test_custom_collate_fn_with_none_values():
    """Test custom_collate_fn handles None values correctly."""
    batch_item_1 = {
        "past_target": torch.randn(3, 1, 1),
        "static_features_real": None,  # Simulate None value
        "output_target": torch.randn(2, 1, 1),
    }

    batch_item_2 = {
        "past_target": torch.randn(3, 1, 1),
        "static_features_real": None,
        "output_target": torch.randn(2, 1, 1),
    }

    batch = [batch_item_1, batch_item_2]

    collated_batch = custom_collate_fn(batch)

    # Check that None values result in None in the collated batch
    assert collated_batch["static_features_real"] is None

    # Check that non-None values are properly stacked
    assert collated_batch["past_target"].shape == (2, 3, 1, 1)
    assert collated_batch["output_target"].shape == (2, 2, 1, 1)
