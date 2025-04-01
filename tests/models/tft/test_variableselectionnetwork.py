import torch

from fintorch.models.timeseries.tft.VariableSelectionNetwork import (
    VariableSelectionNetwork,
)


def test_variable_selection_network_forward():
    # Define input dimensions and parameters
    inputs = {"feature1": 4, "feature2": 3}
    hidden_dimensions = context_size = 16
    sequence_length = 3
    dropout = 0.1
    batch_size = 8

    # Create VariableSelectionNetwork instance
    vsn = VariableSelectionNetwork(inputs, hidden_dimensions, dropout, context_size)

    # Create input tensors
    x = {
        "feature1": torch.randn(
            batch_size, sequence_length, inputs["feature1"]
        ),  # Batch size of 8, sequence length, feature dimension 4
        "feature2": torch.randn(
            batch_size, sequence_length, inputs["feature2"]
        ),  # Batch size of 8, sequence length, feature dimension 3
    }
    context = torch.randn(batch_size, context_size)  # Context tensor

    # Forward pass
    output = vsn(x, context)

    # Check output shape
    assert output.shape == (
        batch_size,
        sequence_length,
        hidden_dimensions,
    ), "Output shape mismatch"

    # Check if the output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a tensor"


def test_variable_selection_network_no_context():
    # Define input dimensions and parameters
    inputs = {"feature1": 4, "feature2": 3}
    hidden_dimensions = 16
    batch_size = 8
    dropout = 0.1
    context_size = 6
    sequence_length = 5

    # Create VariableSelectionNetwork instance
    vsn = VariableSelectionNetwork(inputs, hidden_dimensions, dropout, context_size)

    # Create input tensors without context
    x = {
        "feature1": torch.randn(
            batch_size, sequence_length, inputs["feature1"]
        ),  # Batch size of 8, sequence length, feature dimension 4
        "feature2": torch.randn(
            batch_size, sequence_length, inputs["feature2"]
        ),  # Batch size of 8, sequence length, feature dimension 3
    }

    # Forward pass
    output = vsn(x)

    # Check output shape
    assert output.shape == (
        batch_size,
        sequence_length,
        hidden_dimensions,
    ), "Output shape mismatch without context"

    # Check if the output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a tensor without context"


def test_variable_selection_network_zero_input():
    # Define input dimensions and parameters
    inputs = {"feature1": 4, "feature2": 3}
    hidden_dimensions = context_size = 16
    sequence_length = 3
    dropout = 0.1
    batch_size = 8

    # Create VariableSelectionNetwork instance
    vsn = VariableSelectionNetwork(inputs, hidden_dimensions, dropout, context_size)

    # Create input tensors
    x = {
        "feature1": torch.randn(
            batch_size, sequence_length, inputs["feature1"]
        ),  # Batch size of 8, sequence length, feature dimension 4
        "feature2": torch.randn(
            batch_size, sequence_length, inputs["feature2"]
        ),  # Batch size of 8, sequence length, feature dimension 3
    }
    context = torch.randn(batch_size, context_size)  # Context tensor

    # Forward pass
    output = vsn(x, context)

    # Check if output is not NaN or Inf
    assert torch.all(
        torch.isfinite(output)
    ), "Output contains NaN or Inf for zero input"

    # Check output shape
    assert output.shape == (
        batch_size,
        sequence_length,
        hidden_dimensions,
    ), "Output shape mismatch"
