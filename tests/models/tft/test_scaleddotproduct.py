import torch

from fintorch.models.timeseries.tft.InterpretableMultiHeadAttention import (
    InterpretableMultiHeadAttention,
    ScaledDotProductAttention,
)


# Tests for ScaledDotProductAttention
def test_scaled_dot_product_attention_no_mask():
    batch_size = 2
    seq_length = 3
    embedding_dim = 4
    query = torch.randn(batch_size, seq_length, embedding_dim)
    key = torch.randn(batch_size, seq_length, embedding_dim)
    value = torch.randn(batch_size, seq_length, embedding_dim)

    attention = ScaledDotProductAttention(dropout=0.0)  # Dropout 0 for simplicity
    output, attention_weights = attention(query, key, value, None)

    assert output.shape == (batch_size, seq_length, embedding_dim)
    assert attention_weights.shape == (batch_size, seq_length, seq_length)


def test_scaled_dot_product_attention_with_mask():
    batch_size = 2
    seq_length = 3
    embedding_dim = 4
    query = torch.randn(batch_size, seq_length, embedding_dim)
    key = torch.randn(batch_size, seq_length, embedding_dim)
    value = torch.randn(batch_size, seq_length, embedding_dim)
    # TODO: do we generate the masks correctly in the main body?
    mask = torch.tensor(
        [
            [[False, True, False], [True, False, True], [False, True, False]],
            [[False, True, False], [True, False, True], [False, True, False]],
        ]
    )

    attention = ScaledDotProductAttention(dropout=0.0)
    output, attention_weights = attention(query, key, value, mask)

    assert output.shape == (batch_size, seq_length, embedding_dim)
    assert attention_weights.shape == (batch_size, seq_length, seq_length)
    # Check that masked positions have very low attention weights (close to zero after softmax)
    threshold = 1e-5  # Adjust as needed
    assert torch.all(
        attention_weights[mask] < threshold
    ), f"Masked attention weights are not sufficiently close to zero:{attention_weights}."


# Tests for InterpretableMultiHeadAttention
def test_interpretable_multi_head_attention_no_mask():
    batch_size = 2
    seq_length = 3
    embedding_dim = 8  # Must be divisible by number_of_heads
    number_of_heads = 2
    q = torch.randn(batch_size, seq_length, embedding_dim)
    k = torch.randn(batch_size, seq_length, embedding_dim)
    v = torch.randn(batch_size, seq_length, embedding_dim)

    attention = InterpretableMultiHeadAttention(number_of_heads, embedding_dim, 0.0)
    output, attentions = attention(q, k, v, None)

    assert output.shape == (batch_size, seq_length, embedding_dim)
    assert attentions.shape == (batch_size, number_of_heads, seq_length, seq_length)


def test_interpretable_multi_head_attention_with_mask():
    batch_size = 2
    seq_length = 3
    embedding_dim = 8  # Must be divisible by number_of_heads
    number_of_heads = 2
    q = torch.randn(batch_size, seq_length, embedding_dim)
    k = torch.randn(batch_size, seq_length, embedding_dim)
    v = torch.randn(batch_size, seq_length, embedding_dim)
    mask = torch.tensor(
        [
            [[False, True, False], [True, False, True], [False, True, False]],
            [[False, True, False], [True, False, True], [False, True, False]],
        ]
    )

    attention = InterpretableMultiHeadAttention(number_of_heads, embedding_dim, 0.0)
    output, attentions = attention(q, k, v, mask)

    assert output.shape == (batch_size, seq_length, embedding_dim)
    assert attentions.shape == (batch_size, number_of_heads, seq_length, seq_length)
    # Check that masked positions have very low attention weights (close to zero after softmax)
    threshold = 1e-6
    for head in range(number_of_heads):
        assert torch.all(
            attentions[:, head, :, :][mask] < threshold
        ), "Attention weights should be close to zero for multi-head attention."
