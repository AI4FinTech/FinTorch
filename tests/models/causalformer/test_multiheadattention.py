import torch
from fintorch.models.timeseries.causalformer.MultiHeadAttention import (
    MultiHeadAttention,
)


def test_multheadattention_forward_shape():
    # Define parameters
    number_of_heads = 4
    number_of_series = 5
    length_input_window = 10
    hidden_dimensionality = 16

    # Create a sample input tensor
    batch_size = 2
    x = torch.randn(
        batch_size,
        number_of_heads,
        number_of_series,
        length_input_window,
        hidden_dimensionality,
    )

    # Initialize the MultiHeadAttention module
    mha_layer = MultiHeadAttention(
        number_of_heads=number_of_heads,
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        embedding_size=hidden_dimensionality,
        tau=1.0,
    )

    # Forward pass
    output = mha_layer(x)

    # Assertions
    assert (
        output.shape
        == (
            batch_size,
            number_of_heads,
            number_of_series,
            length_input_window,
            hidden_dimensionality,
        )
    ), f"Expected output shape {(batch_size, number_of_heads, number_of_series, length_input_window, hidden_dimensionality)}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"


def test_multheadattention_forward_with_mask():
    # Define parameters
    number_of_heads = 4
    number_of_series = 5
    length_input_window = 10
    hidden_dimensionality = 16

    # Create a sample input tensor
    batch_size = 2
    x = torch.randn(
        batch_size,
        number_of_heads,
        number_of_series,
        length_input_window,
        hidden_dimensionality,
    )

    # Create a mask tensor
    mask = torch.randint(
        0,
        2,
        (
            batch_size,
            number_of_heads,
            number_of_series,
            length_input_window,
            length_input_window,
        ),
    )

    # Initialize the MultiHeadAttention module
    mha_layer = MultiHeadAttention(
        number_of_heads=number_of_heads,
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        embedding_size=hidden_dimensionality,
        tau=1.0,
    )

    # Forward pass with mask
    output = mha_layer(x, mask=mask)

    # Assertions
    assert (
        output.shape
        == (
            batch_size,
            number_of_heads,
            number_of_series,
            length_input_window,
            hidden_dimensionality,
        )
    ), f"Expected output shape {(batch_size, number_of_heads, number_of_series, length_input_window, hidden_dimensionality)}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
