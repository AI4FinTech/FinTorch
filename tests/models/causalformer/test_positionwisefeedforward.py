import torch
from fintorch.models.timeseries.causalformer.PositionwiseFeedForward import (
    PositionwiseFeedForward,
)
from torch.testing import assert_close


def test_positionwisefeedforward_output_shape():
    """
    Tests if the output shape of the PositionwiseFeedForward module is correct.
    """
    # Define parameters
    input_dim = 64
    hidden_dimensionality = 128
    dropout_rate = 0.1
    batch_size = 2
    number_of_series = 3
    length_input_window = 4
    feature_dimensionality = 64

    # Create a sample input tensor
    x = torch.randn(
        batch_size, number_of_series, length_input_window, feature_dimensionality
    )

    # Initialize the PositionwiseFeedForward module
    ff_layer = PositionwiseFeedForward(input_dim, hidden_dimensionality, dropout_rate)

    # Forward pass
    output = ff_layer(x)

    # Assertions (batch_size, number_of_series, length_input_window, feature_dimensionality)
    assert (
        output.shape == x.shape
    ), f"Expected output shape {x.shape}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"


def test_positionwisefeedforward_dropout_effect():
    """
    Tests if dropout is applied during training and disabled during evaluation.
    """
    # Define parameters
    input_dim = 32
    hidden_dimensionality = 64
    dropout_rate = 0.5  # Use a high dropout rate for a noticeable effect
    batch_size = 1
    number_of_series = 2
    length_input_window = 3
    feature_dimensionality = 32

    # Create a sample input tensor
    x = torch.randn(
        batch_size, number_of_series, length_input_window, feature_dimensionality
    )

    # Initialize the PositionwiseFeedForward module
    ff_layer = PositionwiseFeedForward(input_dim, hidden_dimensionality, dropout_rate)

    # --- Evaluation mode (dropout disabled) ---
    ff_layer.eval()
    with torch.no_grad():
        output_eval_1 = ff_layer(x)
        output_eval_2 = ff_layer(x)

    # Assertions for eval mode
    assert_close(
        output_eval_1, output_eval_2, msg="Outputs should be identical in eval mode"
    )

    # --- Training mode (dropout enabled) ---
    ff_layer.train()
    output_train_1 = ff_layer(x)
    # Need to run again as dropout is stochastic
    output_train_2 = ff_layer(x)

    # Assertions for train mode
    # Check that output in train mode is different from eval mode
    assert not torch.equal(
        output_eval_1, output_train_1
    ), "Output in train mode should differ from eval mode due to dropout"
    # Check that two forward passes in train mode are different (highly likely with dropout > 0)
    assert not torch.equal(
        output_train_1, output_train_2
    ), "Consecutive outputs in train mode should differ due to dropout"


def test_positionwisefeedforward_computation():
    """
    Tests the basic computation flow without dropout.
    """
    # Define parameters
    input_dim = 2
    hidden_dimensionality = 4
    dropout_rate = 0.0  # Disable dropout for deterministic check
    batch_size = 1

    number_of_series = 3
    length_input_window = 4
    feature_dimensionality = input_dim

    # Create a sample input tensor (batch_size, number_of_series, length_input_window, feature_dimensionality)
    x = torch.randn(
        batch_size, number_of_series, length_input_window, feature_dimensionality
    )

    # Initialize the PositionwiseFeedForward module
    ff_layer = PositionwiseFeedForward(input_dim, hidden_dimensionality, dropout_rate)

    # Manually set weights for predictable output (optional, but good for debugging)
    # For simplicity, we'll just check if it runs without error and returns the correct shape
    # If more rigorous checks are needed, set ff_layer.fc1.weight, ff_layer.fc1.bias, etc.

    # Forward pass
    ff_layer.eval()  # Ensure dropout is off
    with torch.no_grad():
        output = ff_layer(x)

    # Assertions
    expected_shape = (
        batch_size,
        number_of_series,
        length_input_window,
        feature_dimensionality,
    )

    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
    # Add more specific value checks here if weights were manually set
