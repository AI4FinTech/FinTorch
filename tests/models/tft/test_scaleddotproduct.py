import torch
from fintorch.models.timeseries.tft.InterpretableMultiHeadAttention import ScaledDotProductAttention

def test_scaled_dot_product_attention_shapes():
    # Define input dimensions
    batch_size = 4
    seq_length = 10
    embedding_dim = 16
    dropout = 0.1

    # Create ScaledDotProductAttention instance
    attention = ScaledDotProductAttention(dropout)

    # Create input tensors
    query = torch.randn(batch_size, seq_length, embedding_dim)
    key = torch.randn(batch_size, seq_length, embedding_dim)
    value = torch.randn(batch_size, seq_length, embedding_dim)
    mask = torch.zeros(batch_size, seq_length, seq_length, dtype=torch.bool)

    # Forward pass
    output, attention_weights = attention(query, key, value, mask)

    # Check output shapes
    assert output.shape == (batch_size, seq_length, embedding_dim), "Output shape mismatch"
    assert attention_weights.shape == (batch_size, seq_length, seq_length), "Attention weights shape mismatch"

def test_scaled_dot_product_attention_masking():
    # Define input dimensions
    batch_size = 2
    seq_length = 5
    embedding_dim = 8
    dropout = 0.1

    # Create ScaledDotProductAttention instance
    attention = ScaledDotProductAttention(dropout)

    # Create input tensors
    query = torch.randn(batch_size, seq_length, embedding_dim)
    key = torch.randn(batch_size, seq_length, embedding_dim)
    value = torch.randn(batch_size, seq_length, embedding_dim)
    mask = torch.ones(batch_size, seq_length, seq_length, dtype=torch.bool)

    # Forward pass
    output, attention_weights = attention(query, key, value, mask)

    # Check if masked positions have near-zero attention weights
    assert torch.all(attention_weights[mask] < 1e-8), "Masking failed"

def test_scaled_dot_product_attention_no_mask():
    # Define input dimensions
    batch_size = 3
    seq_length = 7
    embedding_dim = 12
    dropout = 0.1

    # Create ScaledDotProductAttention instance
    attention = ScaledDotProductAttention(dropout)

    # Create input tensors
    query = torch.randn(batch_size, seq_length, embedding_dim)
    key = torch.randn(batch_size, seq_length, embedding_dim)
    value = torch.randn(batch_size, seq_length, embedding_dim)

    # Forward pass without mask
    output, attention_weights = attention(query, key, value, mask=None)

    # Check if attention weights sum to 1 along the last dimension
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, seq_length)), \
        "Attention weights do not sum to 1"
