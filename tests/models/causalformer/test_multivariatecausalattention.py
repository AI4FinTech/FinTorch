import torch
from fintorch.models.timeseries.causalformer.MultivariateCausalAttention import (
    MultivariateCausalAttention,
)


def test_multivariatecausalattention_forward_shape():
    # Define parameters
    number_of_heads = 4
    number_of_series = 5
    length_input_window = 10
    dropout_rate = 0.1  # Define a dropout rate
    tau = 1.0
    embedding_size = 64
    feature_dim = 8

    # Create a sample input tensor
    batch_size = 2
    x_emb = torch.randn(batch_size, number_of_series, embedding_size)
    x = torch.randn(batch_size, number_of_series, length_input_window, feature_dim)

    q, k, v = x_emb, x_emb, x

    # Create a mask tensor
    mask = torch.randint(
        0,
        2,
        (
            batch_size,
            number_of_heads,
            number_of_series,
            number_of_series,
        ),
    )

    # Initialize the MultivariateCausalAttention module
    mca_layer = MultivariateCausalAttention(
        dropout=dropout_rate,
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
        embedding_size=embedding_size,
        feature_dimensionality=feature_dim,
        tau=tau,
    )

    # Forward pass
    output = mca_layer(q, k, v, mask)

    # Assertions
    assert output.shape == (
        batch_size,
        number_of_series,
        length_input_window,
        feature_dim,
    ), f"Expected output shape {(batch_size, number_of_series, length_input_window, feature_dim)}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"


# def test_multivariatecausalattention_forward_with_invalid_input():
#     # Define parameters
#     number_of_heads = 4
#     number_of_series = 5
#     length_input_window = 10
#     embedding_size = 16
#     dropout_rate = 0.1  # Define a dropout rate
#     tau = 1.0

#     # Create an invalid input tensor (wrong shape)
#     batch_size = 2
#     x = torch.randn(batch_size, number_of_series, length_input_window)

#     # Create a mask tensor
#     mask = torch.randint(
#         0,
#         2,
#         (
#             batch_size,
#             number_of_heads,
#             number_of_series,
#             length_input_window,
#             length_input_window,
#         ),
#     )

#     # Initialize the MultivariateCausalAttention module
#     mca_layer = MultivariateCausalAttention(
#         number_of_heads=number_of_heads,
#         number_of_series=number_of_series,
#         length_input_window=length_input_window,
#         embedding_size=embedding_size,
#         tau=tau,
#         dropout=dropout_rate,
#     )

#     # Forward pass with invalid input
#     try:
#         mca_layer(x, mask)
#         assert (
#             False
#         ), "Expected an error due to invalid input shape, but no error was raised."
#     except RuntimeError as e:
#         assert "size mismatch" in str(e), f"Unexpected error message: {e}"
