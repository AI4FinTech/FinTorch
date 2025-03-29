from typing import Tuple

import torch
import torch.nn as nn


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention module.

    This module implements the interpretable multi-head attention mechanism,
    which is a variant of the standard multi-head attention that allows
    for the interpretation of the attention weights. It divides the input
    into multiple heads and computes attention independently for each head,
    then combines the results.

    Args:
        number_of_heads (int): The number of attention heads.
        input_dimension (int): The dimensionality of the input tensor.
        dropout (float): Dropout rate to apply to the attention weights.

    Attributes:
        number_of_heads (int): The number of attention heads.
        dropout (nn.Dropout): Dropout layer.
        value_projection (nn.Linear): Linear layer for projecting the value tensor.
        query_projections (nn.ModuleList): List of linear layers for projecting the query tensor for each head.
        key_projections (nn.ModuleList): List of linear layers for projecting the key tensor for each head.
        attention (ScaledDotProductAttention): Scaled dot-product attention module.
        final_projection (nn.Linear): Linear layer for projecting the concatenated attention outputs.

    Methods:
        forward(q, k, v, mask):
            Computes the forward pass of the interpretable multi-head attention.

            Args:
                q (torch.Tensor): Query tensor of shape (batch_size, seq_len, input_dimension).
                k (torch.Tensor): Key tensor of shape (batch_size, seq_len, input_dimension).
                v (torch.Tensor): Value tensor of shape (batch_size, seq_len, input_dimension).
                mask (torch.Tensor): Optional mask tensor of shape (batch_size, seq_len, seq_len),
                                     where masked positions are indicated by a value of 1.

            Returns:
                output (torch.Tensor): The output tensor of shape (batch_size, seq_len, input_dimension).
                attentions (torch.Tensor): The attention weights of shape (batch_size, num_heads, seq_len, seq_len).

    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(
        self, number_of_heads: int, input_dimension: int, dropout: float
    ) -> None:
        super(InterpretableMultiHeadAttention, self).__init__()
        # Creates the dimensions for the attentions,
        # we equally divide the input over the number of heads
        self.number_of_heads = number_of_heads
        dim_key = dim_query = dim_value = input_dimension // number_of_heads

        self.dropout = nn.Dropout(dropout)

        # Linear projection value
        self.value_projection = nn.Linear(input_dimension, dim_value)
        # Query projections for each head
        self.query_projections = nn.ModuleList(
            [nn.Linear(input_dimension, dim_query) for _ in range(number_of_heads)]
        )
        # Key projection for each head
        self.key_projections = nn.ModuleList(
            [nn.Linear(input_dimension, dim_key) for _ in range(number_of_heads)]
        )

        self.attention = ScaledDotProductAttention(dropout)
        self.final_projection = nn.Linear(dim_value, input_dimension, bias=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q = k = v = mask => [batch size, sequence length, hidden]
        heads_outputs = []
        attn_outputs = []

        v_proj = self.value_projection(v)
        # Generate a forward-pass for each head
        for head in range(self.number_of_heads):
            # Do a linear projection for all inputs
            q_head = self.query_projections[head](q)
            k_head = self.key_projections[head](k)

            # Feed the unique head projection, to the head attention mechanism
            # Note that we have a single attention mechanism for all heads
            # Equation (12)
            output, attention = self.attention(q_head, k_head, v_proj, mask)
            output_dropout = self.dropout(output)

            heads_outputs.append(output_dropout)
            attn_outputs.append(attention)

        # stack the output of all heads	along the embedding dimension
        heads = torch.stack(heads_outputs, dim=2)
        # stack the attentions of along the sequence dimension
        attentions_stacked = torch.stack(attn_outputs, dim=2)

        # In accordance with equations (14), (15), and (16)
        # take the mean head embedding
        mean_head = heads.mean(dim=2)
        output = self.final_projection(mean_head)
        output = self.dropout(output)

        # [batch size, sequence length, hidden]
        # [batch size, heads, sequence lenght, sequence length, sequence length]
        return output, attentions_stacked.permute(0, 2, 1, 3)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.
    This module implements the scaled dot-product attention mechanism,
    which computes the attention weights by taking the dot product of
    the query and key matrices, scaling the result, and then applying
    a softmax function.
    Args:
        dropout (float): Dropout probability applied to the attention weights.
    Attributes:
        dropout (nn.Dropout): Dropout layer.
        softmax (nn.Softmax): Softmax layer for normalizing attention weights.
    Methods:
        forward(query, key, value, mask):
            Computes the forward pass of the scaled dot-product attention.
            Args:
                query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embedding_dim).
                key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embedding_dim).
                value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embedding_dim).
                mask (torch.Tensor): Optional mask tensor of shape (batch_size, seq_len, seq_len),
                                     where masked positions are indicated by a value of 1.
            Returns:
                output (torch.Tensor): The output tensor of shape (batch_size, seq_len, embedding_dim).
                attention (torch.Tensor): The attention weights of shape (batch_size, seq_len, seq_len).
        mask_attention(attention, mask):
            Applies a mask to the attention weights.
            Args:
                attention (torch.Tensor): Attention weights tensor.
                mask (torch.Tensor): Mask tensor.
            Returns:
                torch.Tensor: Masked attention weights.

    Reference:
    Lim, Bryan, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister. 2019. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” arXiv [Stat.ML]. arXiv. http://arxiv.org/abs/1912.09363.

    """

    def __init__(self, dropout: float) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # perform a softmax on the last dimenion (transformed sequence output)
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # query:
        # k (batch_size, seq_length, embedding_dim)
        # q (batch_size, seq_length, embedding_dim)
        # v (batch_size, seq_length, embedding_dim)

        # First calculate QK^T, we have (batch_size, seq_length, embedding_dim) as inputs
        # using torch.bmm (batch matrix multiplication)
        # which expects (batch_size, n, m) and (batch_size, m, p) to return (batch_size, n, p)
        # transform k from (batch_size, seq_length, embedding_dim) => (batch_size, embedding_dim, seq_length)
        # such that we can use bmm with output (batch_size, seq_length, seq_length)
        attention = torch.bmm(query, key.permute(0, 2, 1))

        # scale factor of equation (10)
        embedding_dim = key.size(-1)
        d_attention = torch.as_tensor(
            embedding_dim, dtype=attention.dtype, device=attention.device
        ).sqrt()
        # Scaled attention
        attention = attention / d_attention
        attention = (
            self.mask_attention(attention, mask) if mask is not None else attention
        )

        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # attention: (batch_size, seq_length, seq_length)
        # value: (batch_size, seq_length, embedding_dim)
        # output bmm: (batch_size, seq_length, embedding_dim)
        # output attention: (batch_size, seq_length, seq_length)
        return torch.bmm(attention, value), attention

    def mask_attention(
        self, attention: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Attention (batch_size, seq_length, seq_length)
        # mask (batch_size, seq_length, seq_length)
        mask = mask.to(attention.device)
        return attention.masked_fill(mask, -1e9)
