import torch
from fintorch.models.timeseries.tft.GatedResidualNetwork import AddNorm


def test_add_norm_forward():
    # Define input dimensions
    dimension = 16

    # Create AddNorm instance
    add_norm = AddNorm(dimension)

    # Create input tensors
    x = torch.randn(8, dimension)  # Batch size of 8
    skip = torch.randn(8, dimension)

    # Forward pass
    output = add_norm(x, skip)

    # Check output shape
    assert output.shape == x.shape, "Output shape mismatch"

    # Check if the output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a tensor"


def test_add_norm_no_change_on_zero_input():
    # Define input dimensions
    dimension = 16

    # Create AddNorm instance
    add_norm = AddNorm(dimension)

    # Create zero input tensors
    x = torch.zeros(8, dimension)
    skip = torch.zeros(8, dimension)

    # Forward pass
    output = add_norm(x, skip)

    # Check if output is still zero
    assert torch.allclose(
        output, torch.zeros_like(output)
    ), "Output is not zero for zero input"


def test_add_norm_residual_connection():
    # Define input dimensions
    dimension = 16

    # Create AddNorm instance
    add_norm = AddNorm(dimension)

    # Create input tensors
    x = torch.randn(8, dimension)
    skip = torch.randn(8, dimension)

    # Forward pass
    output = add_norm(x, skip)

    # Check if residual connection is applied
    residual_sum = x + skip
    assert not torch.allclose(
        output, residual_sum
    ), "Residual connection not normalized"
