from typing import Optional
from torch import Tensor
import torch.nn as nn
import torch
import math
from torch.nn.functional import softmax
from entmax import sparsemax, entmax15

from enum import Enum


class ActivationType(Enum):
    SOFTMAX = softmax
    SPARSEMAX = sparsemax
    ENTMAX15 = entmax15


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_embedding: int,
        number_of_heads: int,
        activation: ActivationType = ActivationType.SOFTMAX,
    ) -> None:
        """
        Initializes the Attention module.

        Args:
            dim_input (int): The dimensionality of the input.
            dim_embedding (int): The dimensionality of the embedding.
            number_of_heads (int): The number of attention heads.
        """
        super().__init__()

        self.dim_input = dim_input
        self.number_of_heads = number_of_heads
        self.head_dim = dim_embedding // number_of_heads
        self.dim_embedding = dim_embedding

        # get the activation function
        self.activation_fn = activation

        # This should be flexible
        self.qkv = nn.Linear(dim_input, 3 * dim_embedding)
        self.ff = nn.Linear(dim_embedding, dim_embedding)

    def scaled_dot_product(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the scaled dot product attention.

        Args:
            q (Tensor): The query tensor.
            k (Tensor): The key tensor.
            v (Tensor): The value tensor.
            mask (any, optional): An optional mask tensor. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the output values tensor and the attention probabilities tensor.
        """
        # Q*K^T/sqrt(d_k)
        attention_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            q.size()[-1]
        )
        if mask is not None:
            attention_logits = attention_logits.masked_fill(mask == 0, -9e15)
        # softmax of the logits to get the probabilities
        attention = self.activation_fn(attention_logits, dim=-1)
        # Probabilities times the values
        values = torch.matmul(attention, v)
        return values, attention

    def forward(
        self, x: Tensor, x2: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the attention module. Supports cross-attention.

        Args:
            x (Tensor): Input tensor of shape [Batch size, sequence length, feature size].
            x2 (Tensor, optional): Input tensor of shape [Batch size, sequence length, feature size].
            mask (Tensor, optional): Mask tensor of shape [Batch size, sequence length] indicating which elements to attend to. Defaults to None.

        Returns:
            Tensor: Attention tensor of shape [Batch size, sequence length, head, dim_emb] and an attention tensor of dimensionality [sequence length, sequence length]
        """

        # For the cross-attention mechanism
        if x2 is None:
            x2 = x

        # We performed all multiplications in one go (q, k, v)
        qkv = self.qkv(
            x2
        )  # [Batch size, sequence length, feature size] -> [Batch size, sequence length, dim_embedding]

        batch_size, seq_len, _ = x.size()

        # TODO: implement masking such that it cannot look into the future!

        # untangle the calculations
        # [Batch size, sequence length, dim_embedding] -> [Batch size, sequence length, heads, dim_emb]
        qkv = qkv.reshape(
            batch_size, seq_len, self.number_of_heads, self.dim_embedding * 3
        )
        # Swap dimensionality of the heads and sequence length
        # [Batch size, sequence length, heads, dim_emb] -> [Batch size, heads, sequence length, dim_emb]
        qkv = qkv.permutate(0, 2, 1, 3)
        # Get individual matrices Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        # Invert the earlier transformation
        # [Batch size, heads, sequence length, dim_emb] -> [Batch size, sequence length, head, dim_emb]
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.dim_embedding)
        o = self.ff(values)  # projection layer

        return o, attention


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_embedding,
        number_of_heads,
        dim_feedforward,
        dropout=0.0,
        number_of_layers_ff=2,
        activation: ActivationType = ActivationType.SOFTMAX,
    ) -> None:
        super().__init__()
        assert number_of_layers_ff >= 2, "num_layers must be larger than 2"

        self.attention = MultiHeadAttention(dim_input, dim_embedding, number_of_heads)

        feedforward_layers = []
        feedforward_layers.append(nn.Linear(dim_embedding, dim_feedforward))
        feedforward_layers.append(nn.Dropout(dropout))
        feedforward_layers.append(nn.ReLU(inplace=True))

        for _ in range(number_of_layers_ff - 2):
            feedforward_layers.append(nn.Linear(dim_feedforward, dim_feedforward))
            feedforward_layers.append(nn.Dropout(dropout))
            feedforward_layers.append(nn.ReLU(inplace=True))

        feedforward_layers.append(nn.Linear(dim_feedforward, dim_input))

        self.feedforward = nn.Sequential(*feedforward_layers)

        self.norm1 = nn.LayerNorm(dim_input)
        self.norm2 = nn.LayerNorm(dim_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x2=None, mask=None):
        if x2 is None:  # self-attention
            x2 = x

        # Calculate attention scores
        attention, attention_map = self.attention(x, x2, mask=mask)
        # skip connection
        x = x + self.dropout(attention)
        # normalize layer
        x = self.norm1(x)

        # Last feed forward layer of the Encoder block
        linear_out = self.feedforward(x)
        # skip-connection
        x = x + self.dropout(linear_out)
        # normalize layer
        x = self.norm2(x)

        return x, attention_map


class TransformerEncoder(nn.Module):
    def __init__(self, number_of_encoder_blocks, **kwargs):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(**kwargs) for _ in range(number_of_encoder_blocks)]
        )

    def forward(self, x, x2=None, mask=None):
        if x2 is None:  # self-attention
            x2 = x

        attention_maps = []
        for encoder_layer in self.encoder_layers:
            x, attention_map = encoder_layer(x, x2, mask=mask)
            attention_maps.append(attention_map)

        return x, attention_maps
