import torch

# Assuming the following imports are correct based on your project structure
from fintorch.models.timeseries.causalformer.EncoderLayer import EncoderLayer
from torch.testing import assert_close

# --- Test Parameters ---
BATCH_SIZE = 4
NUM_SERIES = 3
INPUT_WINDOW = 10
EMBEDDING_SIZE = 64
FEATURE_DIM = 16  # Dimensionality of the raw time series features
FFN_HIDDEN_DIM = 128
NUM_HEADS = 8
TAU = 1.0
DROPOUT_RATE = 0.1


# --- Helper Function to Create Dummy Inputs ---
def _create_dummy_inputs():
    x_emb = torch.randn(BATCH_SIZE, NUM_SERIES, EMBEDDING_SIZE)
    # Note: The 'v' input to MultivariateCausalAttention is the raw feature tensor 'x'
    x = torch.randn(BATCH_SIZE, NUM_SERIES, INPUT_WINDOW, FEATURE_DIM)
    # Create a dummy mask (e.g., no masking) - adjust if your model uses specific masking
    # mask = torch.ones(BATCH_SIZE, NUM_HEADS, NUM_SERIES, NUM_SERIES, dtype=torch.bool) # Example if needed
    return x_emb, x


# --- Test Cases ---


def test_encoderlayer_output_shape():
    """
    Tests if the EncoderLayer returns an output tensor with the expected shape.
    """
    x_emb, x = _create_dummy_inputs()

    encoder_layer = EncoderLayer(
        number_of_heads=NUM_HEADS,
        number_of_series=NUM_SERIES,
        length_input_window=INPUT_WINDOW,
        embedding_size=EMBEDDING_SIZE,
        feature_dimensionality=FEATURE_DIM,
        ffn_hidden_dimensionality=FFN_HIDDEN_DIM,
        tau=TAU,
        dropout=DROPOUT_RATE,
    )

    # Forward pass
    output = encoder_layer(x_emb, x)

    # Assertions
    expected_shape = (BATCH_SIZE, NUM_SERIES, INPUT_WINDOW, FEATURE_DIM)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"


def test_encoderlayer_dropout_norm_effect():
    """
    Tests if dropout and normalization layers behave differently in train vs eval mode.
    """
    # Use a higher dropout rate for a more noticeable difference
    high_dropout_rate = 0.5
    x_emb, x = _create_dummy_inputs()

    encoder_layer = EncoderLayer(
        number_of_heads=NUM_HEADS,
        number_of_series=NUM_SERIES,
        length_input_window=INPUT_WINDOW,
        embedding_size=EMBEDDING_SIZE,
        feature_dimensionality=FEATURE_DIM,
        ffn_hidden_dimensionality=FFN_HIDDEN_DIM,
        tau=TAU,
        dropout=high_dropout_rate,  # Use high dropout
    )

    # --- Evaluation Mode ---
    encoder_layer.eval()
    with torch.no_grad():
        output_eval_1 = encoder_layer(x_emb, x)
        output_eval_2 = encoder_layer(x_emb, x)

    # Assertions for eval mode
    # Outputs should be identical because dropout is disabled
    assert_close(
        output_eval_1, output_eval_2, msg="Outputs should be identical in eval mode"
    )
    # Output should be different from the raw input 'x' due to attention, FFN, and norm
    # (unless layers happen to be identity, which is unlikely)
    assert not torch.equal(
        output_eval_1, x
    ), "Output in eval mode should differ from raw input 'x'"

    # --- Training Mode ---
    encoder_layer.train()
    # No torch.no_grad() here
    output_train_1 = encoder_layer(x_emb, x)
    # Run again - dropout should make it different
    output_train_2 = encoder_layer(x_emb, x)

    # Assertions for train mode
    # Check that output in train mode is different from eval mode (due to dropout)
    assert not torch.allclose(
        output_eval_1, output_train_1
    ), "Output in train mode should differ from eval mode due to dropout/norm"
    # Check that two forward passes in train mode are different (highly likely with dropout > 0)
    assert not torch.allclose(
        output_train_1, output_train_2
    ), "Consecutive outputs in train mode should differ due to dropout"


def test_encoderlayer_runs_with_mask_none():
    """
    Tests if the EncoderLayer runs correctly when the mask input to attention is None.
    """
    x_emb, x = _create_dummy_inputs()

    # Initialize the EncoderLayer
    encoder_layer = EncoderLayer(
        number_of_heads=NUM_HEADS,
        number_of_series=NUM_SERIES,
        length_input_window=INPUT_WINDOW,
        embedding_size=EMBEDDING_SIZE,
        feature_dimensionality=FEATURE_DIM,
        ffn_hidden_dimensionality=FFN_HIDDEN_DIM,  # Add missing argument
        tau=TAU,
        dropout=DROPOUT_RATE,
    )

    # Forward pass with mask=None
    output = encoder_layer(x_emb, x)

    # Assertions
    expected_shape = (BATCH_SIZE, NUM_SERIES, INPUT_WINDOW, FEATURE_DIM)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
