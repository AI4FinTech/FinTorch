import torch
from fintorch.models.timeseries.tft.InterpretableMultiHeadAttention import (
    InterpretableMultiHeadAttention,
)


def test_interpretable_multi_head_attention_output_shape():
    # Define parameters
    batch_size = 4
    seq_len = 10
    input_dimension = 16
    number_of_heads = 4
    dropout = 0.1

    # Create model instance
    model = InterpretableMultiHeadAttention(number_of_heads, input_dimension, dropout)

    # Create input tensors
    q = torch.randn(batch_size, seq_len, input_dimension)
    k = torch.randn(batch_size, seq_len, input_dimension)
    v = torch.randn(batch_size, seq_len, input_dimension)
    mask = torch.zeros(batch_size, seq_len, seq_len).bool()

    # Forward pass
    output, attentions = model(q, k, v, mask)

    # Check output shape
    assert output.shape == (
        batch_size,
        seq_len,
        input_dimension,
    ), "Output shape mismatch"
    assert attentions.shape == (
        batch_size,
        number_of_heads,
        seq_len,
        seq_len,
    ), "Attention shape mismatch"


def test_interpretable_multi_head_attention_no_mask():
    # Define parameters
    batch_size = 4
    seq_len = 10
    input_dimension = 16
    number_of_heads = 4
    dropout = 0.1

    # Create model instance
    model = InterpretableMultiHeadAttention(number_of_heads, input_dimension, dropout)

    # Create input tensors
    q = torch.randn(batch_size, seq_len, input_dimension)
    k = torch.randn(batch_size, seq_len, input_dimension)
    v = torch.randn(batch_size, seq_len, input_dimension)

    # Forward pass without mask
    output, attentions = model(q, k, v, None)

    # Check output shape
    assert output.shape == (
        batch_size,
        seq_len,
        input_dimension,
    ), "Output shape mismatch without mask"
    assert attentions.shape == (
        batch_size,
        number_of_heads,
        seq_len,
        seq_len,
    ), "Attention shape mismatch without mask"


def test_interpretable_multi_head_attention_with_mask():
    # Define parameters
    batch_size = 4
    seq_len = 10
    input_dimension = 16
    number_of_heads = 4
    dropout = 0.1

    # Create model instance
    model = InterpretableMultiHeadAttention(number_of_heads, input_dimension, dropout)

    # Create input tensors
    q = torch.randn(batch_size, seq_len, input_dimension)
    k = torch.randn(batch_size, seq_len, input_dimension)
    v = torch.randn(batch_size, seq_len, input_dimension)
    mask = torch.ones(batch_size, seq_len, seq_len).bool()

    # Forward pass with mask
    output, attentions = model(q, k, v, mask)

    # Check output shape
    assert output.shape == (
        batch_size,
        seq_len,
        input_dimension,
    ), "Output shape mismatch with mask"
    assert attentions.shape == (
        batch_size,
        number_of_heads,
        seq_len,
        seq_len,
    ), "Attention shape mismatch with mask"


def test_interpretable_multi_head_attention_dropout():
    # Define parameters
    batch_size = 4
    seq_len = 10
    input_dimension = 16
    number_of_heads = 4
    dropout = 0.5

    # Create model instance
    model = InterpretableMultiHeadAttention(number_of_heads, input_dimension, dropout)

    # Create input tensors
    q = torch.randn(batch_size, seq_len, input_dimension)
    k = torch.randn(batch_size, seq_len, input_dimension)
    v = torch.randn(batch_size, seq_len, input_dimension)
    mask = torch.zeros(batch_size, seq_len, seq_len).bool()

    # Forward pass
    model.eval()  # Disable dropout for deterministic testing
    output, attentions = model(q, k, v, mask)

    # Check output shape
    assert output.shape == (
        batch_size,
        seq_len,
        input_dimension,
    ), "Output shape mismatch with dropout"
    assert attentions.shape == (
        batch_size,
        number_of_heads,
        seq_len,
        seq_len,
    ), "Attention shape mismatch with dropout"
