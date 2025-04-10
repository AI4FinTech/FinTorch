import torch
from fintorch.models.timeseries.causalformer.Encoder import Encoder


def test_encoder_forward():
    # Define parameters
    number_of_layers = 2
    number_of_heads = 4
    number_of_series = 5
    length_input_window = 10
    embedding_size = 16
    feature_dimensionality = 3
    ffn_hidden_dimensionality = 32
    tau = 0.1
    dropout = 0.1

    # Create a sample input tensor
    batch_size = 2
    x = torch.randn(
        batch_size, number_of_series, length_input_window, feature_dimensionality
    )

    # Initialize the Encoder module
    encoder = Encoder(
        number_of_layers=number_of_layers,
        number_of_heads=number_of_heads,
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        embedding_size=embedding_size,
        feature_dimensionality=feature_dimensionality,
        ffn_hidden_dimensionality=ffn_hidden_dimensionality,
        tau=tau,
        dropout=dropout,
    )

    # Forward pass
    try:
        output = encoder(x)

        # Assertions
        expected_shape = (batch_size, number_of_series, length_input_window, feature_dimensionality)
        assert (
            output.shape == expected_shape
        ), f"Expected output shape {expected_shape}, but got {output.shape}"
        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"

    except Exception as e:
        raise AssertionError(f"Encoder forward pass failed with exception: {e}")


def test_encoder_embedding_output_shape():
    # Define parameters
    number_of_layers = 2
    number_of_heads = 4
    number_of_series = 5
    length_input_window = 10
    embedding_size = 16
    feature_dimensionality = 3
    ffn_hidden_dimensionality = 32
    tau = 0.1
    dropout = 0.1

    # Create a sample input tensor
    batch_size = 2
    x = torch.randn(
        batch_size, number_of_series, length_input_window, feature_dimensionality
    )

    # Initialize the Encoder module
    encoder = Encoder(
        number_of_layers=number_of_layers,
        number_of_heads=number_of_heads,
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        embedding_size=embedding_size,
        feature_dimensionality=feature_dimensionality,
        ffn_hidden_dimensionality=ffn_hidden_dimensionality,
        tau=tau,
        dropout=dropout,
    )

    # Forward pass
    x_emb = encoder.embedding(x)

    # Assertions
    expected_shape = (batch_size, number_of_series, embedding_size)
    assert (
        x_emb.shape == expected_shape
    ), f"Expected embedding output shape {expected_shape}, but got {x_emb.shape}"
    assert isinstance(x_emb, torch.Tensor), "Embedding output should be a torch.Tensor"
