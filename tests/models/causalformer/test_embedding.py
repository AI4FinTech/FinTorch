import torch
from torch.testing import assert_close
from fintorch.models.timeseries.causalformer.Embedding import Embedding


def test_embedding_forward():
    # Define parameters
    number_of_series = 5
    length_input_window = 10
    feature_dimensionality = 3
    hidden_dimensionality = 16
    dropout = 0.1

    # Create a sample input tensor
    batch_size = 2
    x = torch.randn(
        batch_size, number_of_series, length_input_window, feature_dimensionality
    )

    # Initialize the Embedding module
    embedding_layer = Embedding(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        feature_dimensionality=feature_dimensionality,
        hidden_dimensionality=hidden_dimensionality,
        dropout=dropout,
    )

    # Forward pass
    output = embedding_layer(x)

    # Assertions
    assert (
        output.shape == (batch_size, number_of_series, hidden_dimensionality)
    ), f"Expected output shape {(batch_size, number_of_series, hidden_dimensionality)}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"


def test_embedding_dropout_effect():
    # Define parameters
    number_of_series = 5
    length_input_window = 10
    feature_dimensionality = 3
    hidden_dimensionality = 16
    dropout = 0.5

    # Create a sample input tensor
    batch_size = 2

    x = torch.randn(
        batch_size, number_of_series, length_input_window, feature_dimensionality
    )

    # Initialize the Embedding module
    embedding_layer = Embedding(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        feature_dimensionality=feature_dimensionality,
        hidden_dimensionality=hidden_dimensionality,
        dropout=dropout,
    )
    embedding_layer.eval()  # Set to evaluation mode to disable dropout
    output_eval = embedding_layer(x)

    embedding_layer.train()  # Set to training mode to enable dropout
    output_train = embedding_layer(x)

    # Assertions
    (
        assert_close(output_eval, output_eval),
        "Output in eval mode should be deterministic",
    )
    assert not torch.equal(
        output_eval, output_train
    ), "Output in train mode should differ due to dropout"
