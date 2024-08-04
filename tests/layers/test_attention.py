import pytest
import torch

from fintorch.layers.attention import (
    EncoderBlock,
    MultiHeadAttention,
    TransformerEncoder,
)

# TODO: fix autogenerated by Co-pilot needs double checking


def test_multi_head_attention_forward():
    dim_input = 64
    dim_embedding = 128
    number_of_heads = 4
    sequence_length = 10
    batch_size = 32

    attention = MultiHeadAttention(dim_input, dim_embedding, number_of_heads)

    x = torch.randn(batch_size, sequence_length, dim_input)
    output, attention_probs = attention(x)

    assert output.shape == (batch_size, sequence_length, dim_embedding)
    assert attention_probs.shape == (
        batch_size,
        sequence_length,
        number_of_heads,
        sequence_length,
    )


def test_multi_head_attention_scaled_dot_product():
    dim_input = 64
    dim_embedding = 128
    number_of_heads = 4
    sequence_length = 10
    batch_size = 32

    attention = MultiHeadAttention(dim_input, dim_embedding, number_of_heads)

    q = torch.randn(batch_size, sequence_length, dim_embedding)
    k = torch.randn(batch_size, sequence_length, dim_embedding)
    v = torch.randn(batch_size, sequence_length, dim_embedding)
    mask = torch.ones(batch_size, sequence_length)

    values, attention_probs = attention.scaled_dot_product(q, k, v, mask)

    assert values.shape == (batch_size, sequence_length, dim_embedding)
    assert attention_probs.shape == (batch_size, sequence_length, sequence_length)


@pytest.mark.parametrize(
    "dim_input, dim_embedding, number_of_heads",
    [
        (64, 128, 4),
        (32, 64, 2),
        (128, 256, 8),
    ],
)
def test_multi_head_attention_init(dim_input, dim_embedding, number_of_heads):
    attention = MultiHeadAttention(dim_input, dim_embedding, number_of_heads)

    assert attention.dim_input == dim_input
    assert attention.dim_embedding == dim_embedding
    assert attention.number_of_heads == number_of_heads
    assert attention.head_dim == dim_embedding // number_of_heads


def test_encoder_block_forward():
    dim_input = 64
    dim_embedding = 128
    number_of_heads = 4
    dim_feedforward = 256
    dropout = 0.2
    number_of_layers_ff = 3
    encoder_block = EncoderBlock(
        dim_input,
        dim_embedding,
        number_of_heads,
        dim_feedforward,
        dropout,
        number_of_layers_ff,
    )

    batch_size = 10
    sequence_length = 10

    x = torch.randn(batch_size, sequence_length, dim_input)
    output, attention_map = encoder_block(x)
    assert output.shape == (batch_size, sequence_length, dim_input)
    assert attention_map.shape == (
        batch_size,
        number_of_heads,
        sequence_length,
        sequence_length,
    )


def test_encoder_block_forward_with_mask():
    dim_input = 64
    dim_embedding = 128
    number_of_heads = 4
    dim_feedforward = 256
    dropout = 0.2
    number_of_layers_ff = 3
    encoder_block = EncoderBlock(
        dim_input,
        dim_embedding,
        number_of_heads,
        dim_feedforward,
        dropout,
        number_of_layers_ff,
    )
    batch_size = 10
    sequence_length = 10
    x = torch.randn(batch_size, sequence_length, dim_input)
    mask = torch.ones(batch_size, sequence_length)
    output, attention_map = encoder_block(x, mask=mask)
    assert output.shape == (batch_size, sequence_length, dim_input)
    assert attention_map.shape == (
        batch_size,
        number_of_heads,
        sequence_length,
        sequence_length,
    )


@pytest.mark.parametrize(
    "dim_input, dim_embedding, number_of_heads, dim_feedforward, dropout, number_of_layers_ff",
    [
        (64, 128, 4, 256, 0.2, 3),
        (32, 64, 2, 128, 0.1, 2),
        (128, 256, 8, 512, 0.3, 4),
    ],
)
def test_encoder_block_init(
    dim_input,
    dim_embedding,
    number_of_heads,
    dim_feedforward,
    dropout,
    number_of_layers_ff,
):
    encoder_block = EncoderBlock(
        dim_input,
        dim_embedding,
        number_of_heads,
        dim_feedforward,
        dropout,
        number_of_layers_ff,
    )
    assert encoder_block.dim_input == dim_input
    assert encoder_block.dim_embedding == dim_embedding
    assert encoder_block.number_of_heads == number_of_heads
    assert encoder_block.dim_feedforward == dim_feedforward
    assert encoder_block.dropout == dropout
    assert len(encoder_block.feedforward) == number_of_layers_ff


def test_transformer_encoder_forward():
    dim_input = 64
    dim_embedding = 128
    number_of_heads = 4
    dim_feedforward = 256
    dropout = 0.2
    number_of_layers_ff = 3
    number_of_encoder_blocks = 2
    transformer_encoder = TransformerEncoder(
        number_of_encoder_blocks,
        dim_input=dim_input,
        dim_embedding=dim_embedding,
        number_of_heads=number_of_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        number_of_layers_ff=number_of_layers_ff,
    )
    batch_size = 32
    sequence_length = 10
    x = torch.randn(batch_size, sequence_length, dim_input)
    output, attention_maps = transformer_encoder(x)
    assert output.shape == (batch_size, sequence_length, dim_input)
    assert len(attention_maps) == number_of_encoder_blocks
    for attention_map in attention_maps:
        assert attention_map.shape == (
            batch_size,
            number_of_heads,
            sequence_length,
            sequence_length,
        )


@pytest.mark.parametrize(
    "dim_input, dim_embedding, number_of_heads, dim_feedforward, dropout, number_of_layers_ff, number_of_encoder_blocks",
    [
        (64, 128, 4, 256, 0.2, 3, 2),
        (32, 64, 2, 128, 0.1, 2, 3),
        (128, 256, 8, 512, 0.3, 4, 4),
    ],
)
def test_transformer_encoder_init(
    dim_input,
    dim_embedding,
    number_of_heads,
    dim_feedforward,
    dropout,
    number_of_layers_ff,
    number_of_encoder_blocks,
):
    transformer_encoder = TransformerEncoder(
        number_of_encoder_blocks,
        dim_input=dim_input,
        dim_embedding=dim_embedding,
        number_of_heads=number_of_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        number_of_layers_ff=number_of_layers_ff,
    )
    assert len(transformer_encoder.encoder_layers) == number_of_encoder_blocks
    for encoder_layer in transformer_encoder.encoder_layers:
        assert encoder_layer.dim_input == dim_input
        assert encoder_layer.dim_embedding == dim_embedding
        assert encoder_layer.number_of_heads == number_of_heads
        assert encoder_layer.dim_feedforward == dim_feedforward
        assert encoder_layer.dropout == dropout
        assert len(encoder_layer.feedforward) == number_of_layers_ff
