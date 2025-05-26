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
   - Static data handling (always None)
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
1. Accumulates batch items into separate lists for past_inputs, future_inputs, static_inputs, and targets
2. Stacks past_data tensors using torch.stack() to create batched tensors
3. Stacks future_data tensors using torch.stack() to create batched tensors
4. Handles static_data by setting it to None (as required by this dataset)
5. Stacks target tensors using torch.stack() to create batched targets
6. Returns properly formatted dictionaries and tensors for model consumption
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
    # Create mock batch data matching the expected format
    past_data_1 = torch.randn(10, 1)
    future_data_1 = torch.randn(5, 1)
    target_1 = torch.randn(5)

    past_data_2 = torch.randn(10, 1)
    future_data_2 = torch.randn(5, 1)
    target_2 = torch.randn(5)

    past_data_3 = torch.randn(10, 1)
    future_data_3 = torch.randn(5, 1)
    target_3 = torch.randn(5)

    batch = [
        (
            {"past_data": past_data_1},
            {"future_data": future_data_1},
            {"static_data": None},
            target_1
        ),
        (
            {"past_data": past_data_2},
            {"future_data": future_data_2},
            {"static_data": None},
            target_2
        ),
        (
            {"past_data": past_data_3},
            {"future_data": future_data_3},
            {"static_data": None},
            target_3
        ),
    ]

    collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

    # Check return types
    assert isinstance(collated_past, dict)
    assert isinstance(collated_future, dict)
    assert isinstance(collated_static, dict)
    assert isinstance(collated_targets, torch.Tensor)

    # Check dictionary keys
    assert "past_data" in collated_past
    assert "future_data" in collated_future
    assert "static_data" in collated_static

    # Check tensor shapes - should have batch dimension added
    assert collated_past["past_data"].shape == (3, 10, 1)  # (batch_size, time, features)
    assert collated_future["future_data"].shape == (3, 5, 1)
    assert collated_targets.shape == (3, 5)

    # Check static data is None
    assert collated_static["static_data"] is None

    # Check that the data is correctly stacked
    assert torch.equal(collated_past["past_data"][0], past_data_1)
    assert torch.equal(collated_past["past_data"][1], past_data_2)
    assert torch.equal(collated_past["past_data"][2], past_data_3)

    assert torch.equal(collated_future["future_data"][0], future_data_1)
    assert torch.equal(collated_future["future_data"][1], future_data_2)
    assert torch.equal(collated_future["future_data"][2], future_data_3)

    assert torch.equal(collated_targets[0], target_1)
    assert torch.equal(collated_targets[1], target_2)
    assert torch.equal(collated_targets[2], target_3)


def test_custom_collate_fn_single_item():
    """Test custom_collate_fn with a single item batch."""
    past_data = torch.randn(10, 1)
    future_data = torch.randn(5, 1)
    target = torch.randn(5)

    batch = [
        (
            {"past_data": past_data},
            {"future_data": future_data},
            {"static_data": None},
            target
        ),
    ]

    collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

    # Check shapes with batch size 1
    assert collated_past["past_data"].shape == (1, 10, 1)
    assert collated_future["future_data"].shape == (1, 5, 1)
    assert collated_targets.shape == (1, 5)
    assert collated_static["static_data"] is None

    # Check data integrity
    assert torch.equal(collated_past["past_data"][0], past_data)
    assert torch.equal(collated_future["future_data"][0], future_data)
    assert torch.equal(collated_targets[0], target)


def test_custom_collate_fn_different_tensor_types():
    """Test custom_collate_fn with different tensor dtypes."""
    # Create tensors with different dtypes
    past_data_float = torch.randn(10, 1).float()
    future_data_double = torch.randn(5, 1).double()
    target_int = torch.randint(0, 10, (5,)).int()

    past_data_float2 = torch.randn(10, 1).float()
    future_data_double2 = torch.randn(5, 1).double()
    target_int2 = torch.randint(0, 10, (5,)).int()

    batch = [
        (
            {"past_data": past_data_float},
            {"future_data": future_data_double},
            {"static_data": None},
            target_int
        ),
        (
            {"past_data": past_data_float2},
            {"future_data": future_data_double2},
            {"static_data": None},
            target_int2
        ),
    ]

    collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

    # Check that dtypes are preserved
    assert collated_past["past_data"].dtype == torch.float32
    assert collated_future["future_data"].dtype == torch.float64
    assert collated_targets.dtype == torch.int32

    # Check shapes
    assert collated_past["past_data"].shape == (2, 10, 1)
    assert collated_future["future_data"].shape == (2, 5, 1)
    assert collated_targets.shape == (2, 5)


def test_custom_collate_fn_preserves_gradients():
    """Test that custom_collate_fn preserves gradient information."""
    past_data = torch.randn(10, 1, requires_grad=True)
    future_data = torch.randn(5, 1, requires_grad=True)
    target = torch.randn(5, requires_grad=True)

    batch = [
        (
            {"past_data": past_data},
            {"future_data": future_data},
            {"static_data": None},
            target
        ),
    ]

    collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

    # Check that gradient information is preserved
    assert collated_past["past_data"].requires_grad
    assert collated_future["future_data"].requires_grad
    assert collated_targets.requires_grad


def test_custom_collate_fn_empty_batch():
    """Test custom_collate_fn with empty batch (edge case)."""
    batch = []

    # This should raise a RuntimeError when trying to stack empty tensor lists
    with pytest.raises(RuntimeError, match="stack expects a non-empty TensorList"):
        custom_collate_fn(batch)


def test_custom_collate_fn_variable_sequence_lengths():
    """Test custom_collate_fn with different sequence lengths (should fail)."""
    # Create data with different sequence lengths
    past_data_1 = torch.randn(10, 1)  # length 10
    past_data_2 = torch.randn(8, 1)   # length 8 - different!

    future_data_1 = torch.randn(5, 1)
    future_data_2 = torch.randn(5, 1)

    target_1 = torch.randn(5)
    target_2 = torch.randn(5)

    batch = [
        (
            {"past_data": past_data_1},
            {"future_data": future_data_1},
            {"static_data": None},
            target_1
        ),
        (
            {"past_data": past_data_2},
            {"future_data": future_data_2},
            {"static_data": None},
            target_2
        ),
    ]

    # This should raise a RuntimeError due to size mismatch in torch.stack
    with pytest.raises(RuntimeError):
        custom_collate_fn(batch)


def test_custom_collate_fn_static_data_always_none():
    """Test that static_data is always None regardless of input."""
    batch = [
        (
            {"past_data": torch.randn(10, 1)},
            {"future_data": torch.randn(5, 1)},
            {"static_data": None},
            torch.randn(5)
        ),
        (
            {"past_data": torch.randn(10, 1)},
            {"future_data": torch.randn(5, 1)},
            {"static_data": None},
            torch.randn(5)
        ),
    ]

    collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

    # Static data should always be None for this dataset
    assert collated_static["static_data"] is None


def test_air_passenger_dataset_initialization():
    """Test AirPassengerDataset initialization."""
    dataset = AirPassengerDataset(past_length=12, future_length=6)

    assert dataset.past_length == 12
    assert dataset.future_length == 6
    assert hasattr(dataset, 'scaler')
    assert hasattr(dataset, 'data')
    assert len(dataset) > 0


def test_air_passenger_dataset_getitem():
    """Test AirPassengerDataset __getitem__ method."""
    dataset = AirPassengerDataset(past_length=10, future_length=5)

    # Get a single item
    past_inputs, future_inputs, static_inputs, target = dataset[0]

    # Check return types and structure
    assert isinstance(past_inputs, dict)
    assert isinstance(future_inputs, dict)
    assert isinstance(static_inputs, dict)
    assert isinstance(target, torch.Tensor)

    # Check dictionary keys
    assert "past_data" in past_inputs
    assert "future_data" in future_inputs
    assert "static_data" in static_inputs

    # Check tensor shapes
    assert past_inputs["past_data"].shape == (10, 1)
    assert future_inputs["future_data"].shape == (5, 1)
    assert target.shape == (5,)

    # Check static data is None
    assert static_inputs["static_data"] is None

    # Check data types
    assert past_inputs["past_data"].dtype == torch.float32
    assert future_inputs["future_data"].dtype == torch.float32
    assert target.dtype == torch.float32


def test_air_passenger_datamodule_initialization():
    """Test AirPassengerDataModule initialization."""
    datamodule = AirPassengerDataModule(
        batch_size=32,
        past_length=12,
        horizon=6,
        workers=1
    )

    assert datamodule.batch_size == 32
    assert datamodule.past_length == 12
    assert datamodule.future_length == 6
    assert datamodule.workers == 1


def test_air_passenger_datamodule_setup():
    """Test AirPassengerDataModule setup method."""
    datamodule = AirPassengerDataModule(batch_size=16, past_length=10, horizon=5)
    datamodule.setup()

    # Check that datasets are created
    assert hasattr(datamodule, 'train_dataset')
    assert hasattr(datamodule, 'val_dataset')
    assert hasattr(datamodule, 'test_dataset')

    # Check that datasets have correct configurations
    assert datamodule.train_dataset.past_length == 10
    assert datamodule.train_dataset.future_length == 5
    assert datamodule.val_dataset.past_length == 10
    assert datamodule.val_dataset.future_length == 5
    assert datamodule.test_dataset.past_length == 10
    assert datamodule.test_dataset.future_length == 5


def test_air_passenger_datamodule_dataloaders():
    """Test AirPassengerDataModule dataloader methods."""
    datamodule = AirPassengerDataModule(batch_size=8, past_length=10, horizon=5)
    datamodule.setup()

    # Test train dataloader
    train_loader = datamodule.train_dataloader()
    assert train_loader.batch_size == 8
    assert train_loader.dataset == datamodule.train_dataset

    # Test val dataloader
    val_loader = datamodule.val_dataloader()
    assert val_loader.batch_size == 8
    assert val_loader.dataset == datamodule.val_dataset

    # Test test dataloader
    test_loader = datamodule.test_dataloader()
    assert test_loader.batch_size == 8
    assert test_loader.dataset == datamodule.test_dataset

    # Test predict dataloader
    predict_loader = datamodule.predict_dataloader()
    assert predict_loader.batch_size == 8
    assert predict_loader.dataset == datamodule.test_dataset


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
    past_inputs, future_inputs, static_inputs, targets = batch

    # Check batch structure
    assert isinstance(past_inputs, dict)
    assert isinstance(future_inputs, dict)
    assert isinstance(static_inputs, dict)
    assert isinstance(targets, torch.Tensor)

    # Check batch dimensions
    assert past_inputs["past_data"].shape[0] == 3  # batch size
    assert future_inputs["future_data"].shape[0] == 3
    assert targets.shape[0] == 3

    # Check sequence dimensions
    assert past_inputs["past_data"].shape[1] == 8  # past_length
    assert future_inputs["future_data"].shape[1] == 4  # future_length
    assert targets.shape[1] == 4


def test_custom_collate_fn_type_annotations():
    """Test that custom_collate_fn handles the expected input/output types correctly."""
    # Create a batch that matches the type annotation
    batch_item = (
        {"past_data": torch.randn(10, 1)},  # Dict[str, torch.Tensor]
        {"future_data": torch.randn(5, 1)},  # Dict[str, torch.Tensor]
        {"static_data": None},  # Dict[str, Optional[torch.Tensor]]
        torch.randn(5)  # torch.Tensor
    )

    batch = [batch_item, batch_item]

    result = custom_collate_fn(batch)

    # Check return type matches annotation
    assert isinstance(result, tuple)
    assert len(result) == 4

    past_inputs, future_inputs, static_inputs, targets = result

    # Check types match the annotation
    assert isinstance(past_inputs, dict)
    assert isinstance(future_inputs, dict)
    assert isinstance(static_inputs, dict)
    assert isinstance(targets, torch.Tensor)

    # Check dictionary values
    assert isinstance(past_inputs["past_data"], torch.Tensor)
    assert isinstance(future_inputs["future_data"], torch.Tensor)
    assert static_inputs["static_data"] is None


def test_custom_collate_fn_list_accumulation():
    """Test that the collation logic correctly accumulates items into lists."""
    # Create specific data to track through the process
    past_data_1 = torch.tensor([[1.0], [2.0], [3.0]])
    past_data_2 = torch.tensor([[4.0], [5.0], [6.0]])
    future_data_1 = torch.tensor([[7.0], [8.0]])
    future_data_2 = torch.tensor([[9.0], [10.0]])
    target_1 = torch.tensor([11.0, 12.0])
    target_2 = torch.tensor([13.0, 14.0])

    batch = [
        (
            {"past_data": past_data_1},
            {"future_data": future_data_1},
            {"static_data": None},
            target_1
        ),
        (
            {"past_data": past_data_2},
            {"future_data": future_data_2},
            {"static_data": None},
            target_2
        ),
    ]

    collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

    # Verify that the list accumulation step worked correctly
    # by checking that data is stacked in the correct order
    assert torch.equal(collated_past["past_data"][0], past_data_1)
    assert torch.equal(collated_past["past_data"][1], past_data_2)
    assert torch.equal(collated_future["future_data"][0], future_data_1)
    assert torch.equal(collated_future["future_data"][1], future_data_2)
    assert torch.equal(collated_targets[0], target_1)
    assert torch.equal(collated_targets[1], target_2)


def test_custom_collate_fn_stacking_behavior():
    """Test the torch.stack behavior in the collation function."""
    # Create tensors with specific values to verify stacking
    past_1 = torch.tensor([[1.0], [2.0]])
    past_2 = torch.tensor([[3.0], [4.0]])
    past_3 = torch.tensor([[5.0], [6.0]])

    future_1 = torch.tensor([[7.0]])
    future_2 = torch.tensor([[8.0]])
    future_3 = torch.tensor([[9.0]])

    target_1 = torch.tensor([10.0])
    target_2 = torch.tensor([11.0])
    target_3 = torch.tensor([12.0])

    batch = [
        ({"past_data": past_1}, {"future_data": future_1}, {"static_data": None}, target_1),
        ({"past_data": past_2}, {"future_data": future_2}, {"static_data": None}, target_2),
        ({"past_data": past_3}, {"future_data": future_3}, {"static_data": None}, target_3),
    ]

    collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

    # Test that torch.stack creates the expected tensor structure
    expected_past = torch.stack([past_1, past_2, past_3])
    expected_future = torch.stack([future_1, future_2, future_3])
    expected_targets = torch.stack([target_1, target_2, target_3])

    assert torch.equal(collated_past["past_data"], expected_past)
    assert torch.equal(collated_future["future_data"], expected_future)
    assert torch.equal(collated_targets, expected_targets)

    # Verify the stacking added the batch dimension correctly
    assert collated_past["past_data"].shape == (3, 2, 1)  # batch_size=3, seq_len=2, features=1
    assert collated_future["future_data"].shape == (3, 1, 1)  # batch_size=3, seq_len=1, features=1
    assert collated_targets.shape == (3, 1)  # batch_size=3, target_len=1


def test_custom_collate_fn_dictionary_structure():
    """Test that the collation function correctly handles dictionary structures."""
    batch = [
        (
            {"past_data": torch.randn(5, 1)},
            {"future_data": torch.randn(3, 1)},
            {"static_data": None},
            torch.randn(3)
        ),
        (
            {"past_data": torch.randn(5, 1)},
            {"future_data": torch.randn(3, 1)},
            {"static_data": None},
            torch.randn(3)
        ),
    ]

    collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

    # Test that dictionaries maintain their structure
    assert len(collated_past) == 1
    assert len(collated_future) == 1
    assert len(collated_static) == 1

    # Test that the correct keys are present
    assert "past_data" in collated_past
    assert "future_data" in collated_future
    assert "static_data" in collated_static

    # Test that no extra keys are added
    assert list(collated_past.keys()) == ["past_data"]
    assert list(collated_future.keys()) == ["future_data"]
    assert list(collated_static.keys()) == ["static_data"]


def test_custom_collate_fn_static_data_handling():
    """Test the specific handling of static_data (always None) in the collation function."""
    # Test various scenarios where static_data is None
    batch_scenarios = [
        # All None
        [
            ({"past_data": torch.randn(2, 1)}, {"future_data": torch.randn(1, 1)}, {"static_data": None}, torch.randn(1)),
            ({"past_data": torch.randn(2, 1)}, {"future_data": torch.randn(1, 1)}, {"static_data": None}, torch.randn(1)),
        ],
    ]

    for batch in batch_scenarios:
        collated_past, collated_future, collated_static, collated_targets = custom_collate_fn(batch)

        # The function explicitly sets static_data to None regardless of input
        assert collated_static["static_data"] is None

        # Verify this is consistent behavior across the batch
        assert isinstance(collated_static, dict)
        assert len(collated_static) == 1
