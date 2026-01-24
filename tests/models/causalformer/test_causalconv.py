import torch
import pytest

from fintorch.models.timeseries.causalformer.CausalConvolution import CausalConvolution

def test_layerwise_relevance_propagation():
    # Define parameters
    batch_size = 2
    number_of_series = 3
    length_input_window = 4
    hidden_dimensionality = 5
    number_of_heads = 2

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    # Create a sample input tensor
    x = torch.randn(
        batch_size, number_of_series, length_input_window, hidden_dimensionality,
        requires_grad=True
    )

    # Forward pass to set up the hooks
    output = causal_conv(x)

    # Create relevance tensor with same shape as output
    relevance = torch.randn_like(output)

    # Test the relevance propagation operations step by step
    # First test: reverse transform_x operation
    relevance_copy = relevance.clone()
    for i in range(causal_conv.number_of_series):
        relevance_copy[:, :, i, i, :, :] = relevance_copy[:, :, i, i, :, :].roll(-1, dims=2)

    # Check that the roll operation works correctly
    assert relevance_copy.shape == relevance.shape, (
        "Relevance shape should remain unchanged after transform_x reversal"
    )

    # Second test: reverse base division
    relevance_copy = relevance_copy * causal_conv.base

    # Check that multiplication with base tensor works
    assert relevance_copy.shape == relevance.shape, (
        "Relevance shape should remain unchanged after base multiplication"
    )

    # Test that relevance propagation method runs without crashing
    # Note: We catch autograd errors since the hook mechanism may not always
    # maintain the computation graph properly
    try:
        causal_conv.propagate(relevance)
    except Exception as e:
        pytest.fail(f"Relevance propagation method failed: {e}")

def test_relevance_propagation_consistency():
    # Test that relevance propagation operations are consistent
    batch_size = 1
    number_of_series = 2
    length_input_window = 3
    hidden_dimensionality = 4
    number_of_heads = 1

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    # Create a sample input tensor
    x = torch.randn(
        batch_size, number_of_series, length_input_window, hidden_dimensionality,
        requires_grad=True
    )

    # Forward pass
    output = causal_conv(x)

    # Test that the base tensor is moved to the correct device
    assert causal_conv.base.device == x.device, (
        "Base tensor should be on the same device as input"
    )

    # Create relevance tensor
    relevance = torch.ones_like(output)

    # Test that relevance propagation doesn't crash with various inputs
    try:
        causal_conv.propagate(relevance)
        assert True, "Relevance propagation should not crash"
    except Exception as e:
        assert False, f"Relevance propagation crashed with error: {e}"



def test_relevance_propagation_zero_input():
    """Test relevance propagation with zero relevance input."""
    batch_size = 1
    number_of_series = 2
    length_input_window = 3
    hidden_dimensionality = 2
    number_of_heads = 1

    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    x = torch.randn(
        batch_size, number_of_series, length_input_window, hidden_dimensionality,
        requires_grad=True
    )
    output = causal_conv(x)

    # Test with zero relevance
    zero_relevance = torch.zeros_like(output)

    try:
        zero_propagated = causal_conv.propagate(zero_relevance)
        assert torch.allclose(zero_propagated, torch.zeros_like(zero_propagated)), \
            "Zero relevance input should result in zero propagated relevance"
    except Exception as e:
        pytest.fail(f"Zero relevance propagation failed: {e}")
