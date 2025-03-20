import torch
import torch.nn as nn


class InterpretableMultiHeadAttention(nn.Module):
    """
    A PyTorch implementation of Interpretable Multi-Head Attention.
    This module implements a multi-head attention mechanism where each head
    has its own query and key projections, but shares a single value projection
    and attention mechanism. The outputs of all heads are averaged to produce
    the final output, making the attention mechanism interpretable.
    Attributes:
        number_of_heads (int): The number of attention heads.
        input_dimension (int): The dimensionality of the input embeddings.
        dropout (float): Dropout probability applied to the attention outputs.
    Methods:
        forward(q, k, v, mask):
            Computes the forward pass of the multi-head attention mechanism.
            Args:
                q (torch.Tensor): Query tensor of shape (batch_size, seq_len, input_dimension).
                k (torch.Tensor): Key tensor of shape (batch_size, seq_len, input_dimension).
                v (torch.Tensor): Value tensor of shape (batch_size, seq_len, input_dimension).
                mask (torch.Tensor): Optional mask tensor of shape (batch_size, seq_len, seq_len),
                                     where masked positions are indicated by a value of 1.
            Returns:
                output (torch.Tensor): The output tensor of shape (batch_size, seq_len, input_dimension).
                attentions_stacked (torch.Tensor): Stacked attention weights of shape
                                                   (batch_size, seq_len, number_of_heads, seq_len).
    """

    def __init__(self, number_of_heads, input_dimension, dropout):
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

    def forward(self, q, k, v, mask):
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

        return output, attentions_stacked.permute(0, 2, 1, 3)


class ScaledDotProductAttention(nn.Module):
    """
    Implements the Scaled Dot-Product Attention mechanism as described in the
    "Attention is All You Need" paper. This module computes attention scores
    and applies them to the input values.

    Args:
        dropout (float): Dropout probability to apply to the attention scores.

    Methods:
        forward(query, key, value, mask):
            Computes the scaled dot-product attention.

            Args:
                query (torch.Tensor): Query tensor of shape
                    (batch_size, seq_length, embedding_dim).
                key (torch.Tensor): Key tensor of shape
                    (batch_size, seq_length, embedding_dim).
                value (torch.Tensor): Value tensor of shape
                    (batch_size, seq_length, embedding_dim).
                mask (torch.Tensor or None): Optional mask tensor of shape
                    (batch_size, seq_length, seq_length). Positions with a
                    value of True are masked (set to a very large negative
                    value).

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - The output tensor of shape
                      (batch_size, seq_length, embedding_dim), which is the
                      result of applying the attention scores to the values.
                    - The attention scores tensor of shape
                      (batch_size, seq_length, seq_length).
    """

    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # perform a softmax on the last dimenion (transformed sequence output)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask):
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

    def mask_attention(self, attention, mask):
        mask = mask.to(attention.device)
        return attention.masked_fill(mask, -1e9)
