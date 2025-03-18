import torch
from fintorch.models.timeseries.tft.VariableSelectionNetwork import VariableSelectionNetwork

def test_variable_selection_network_forward():
    # Define input dimensions and parameters
    inputs = {"feature1": 4, "feature2": 3}
    hidden_dimensions = 8
    dropout = 0.1
    context_size = 6

    # Create VariableSelectionNetwork instance
    vsn = VariableSelectionNetwork(inputs, hidden_dimensions, dropout, context_size)

    # Create input tensors
    x = {
        "feature1": torch.randn(8, 4),  # Batch size of 8, feature dimension 4
        "feature2": torch.randn(8, 3),  # Batch size of 8, feature dimension 3
    }
    context = torch.randn(8, context_size)  # Context tensor

    # Forward pass
    output = vsn(x, context)

    # Check output shape
    assert output.shape == (8, hidden_dimensions), "Output shape mismatch"

    # Check if the output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a tensor"


def test_variable_selection_network_no_context():
    # Define input dimensions and parameters
    inputs = {"feature1": 4, "feature2": 3}
    hidden_dimensions = 8
    dropout = 0.1
    context_size = 6

    # Create VariableSelectionNetwork instance
    vsn = VariableSelectionNetwork(inputs, hidden_dimensions, dropout, context_size)

    # Create input tensors without context
    x = {
        "feature1": torch.randn(8, 4),  # Batch size of 8, feature dimension 4
        "feature2": torch.randn(8, 3),  # Batch size of 8, feature dimension 3
    }

    # Forward pass
    output = vsn(x)

    # Check output shape
    assert output.shape == (8, hidden_dimensions), "Output shape mismatch without context"

    # Check if the output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a tensor without context"


def test_variable_selection_network_zero_input():
    # Define input dimensions and parameters
    inputs = {"feature1": 4, "feature2": 3}
    hidden_dimensions = 8
    dropout = 0.1
    context_size = 6

    # Create VariableSelectionNetwork instance
    vsn = VariableSelectionNetwork(inputs, hidden_dimensions, dropout, context_size)

    # Create zero input tensors
    x = {
        "feature1": torch.zeros(8, 4),  # Batch size of 8, feature dimension 4
        "feature2": torch.zeros(8, 3),  # Batch size of 8, feature dimension 3
    }
    context = torch.zeros(8, context_size)  # Zero context tensor

    # Forward pass
    output = vsn(x, context)

    # Check if output is not NaN or Inf
    assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf for zero input"

    # Check output shape
    assert output.shape == (8, hidden_dimensions), "Output shape mismatch for zero input"
