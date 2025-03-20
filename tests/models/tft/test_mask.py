from fintorch.models.timeseries.tft.utils import attention_mask


def test_attention_mask_shape():
    past_length = 5
    future_length = 3
    mask = attention_mask(past_length, future_length)

    # Check the shape of the mask
    assert mask.shape == (
        past_length + future_length,
        past_length + future_length,
    ), "Mask shape is incorrect"


def test_attention_mask_upper_triangle():
    past_length = 4
    future_length = 2
    mask = attention_mask(past_length, future_length).cpu().numpy()

    # Check that the upper triangle is masked (True)
    total_length = past_length + future_length
    for i in range(total_length):
        for j in range(i + 1, total_length):
            assert mask[i, j], f"Mask at position ({i}, {j}) should be True"


def test_attention_mask_lower_triangle():
    past_length = 3
    future_length = 3
    mask = attention_mask(past_length, future_length)

    # Check that the lower triangle is not masked (False)
    total_length = past_length + future_length
    for i in range(total_length):
        for j in range(i + 1):
            assert not mask[i, j], f"Mask at position ({i}, {j}) should be False"


def test_attention_mask_zero_lengths():
    past_length = 0
    future_length = 0
    mask = attention_mask(past_length, future_length)

    # Check that the mask is empty
    assert mask.numel() == 0, "Mask should be empty for zero lengths"
