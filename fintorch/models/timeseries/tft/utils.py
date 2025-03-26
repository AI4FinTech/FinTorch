import torch


def attention_mask(past_length: int, future_length: int) -> torch.Tensor:
    """
    Generates an attention mask for the Temporal Fusion Transformer (TFT).

    Args:
        past_length (int): The length of the past sequence.
        future_length (int): The length of the future sequence.

    Returns:
        torch.Tensor: An attention mask of shape (total_length, total_length).
    """

    total_length = past_length + future_length
    mask = torch.tril(torch.ones(total_length, total_length))
    return mask == 0
