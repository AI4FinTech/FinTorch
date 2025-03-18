import torch
from fintorch.models.timeseries.tft.tft import TemporalFusionTransformer

def test_tft_forward_with_past_inputs():
    # Define model parameters
    number_of_past_inputs = 10
    number_of_future_inputs = 5
    embedding_size_inputs = 16
    hidden_dimension = 32
    dropout = 0.1
    number_of_heads = 4
    past_inputs = ["past_feature_1", "past_feature_2"]
    future_inputs = ["future_feature_1"]
    static_inputs = ["static_feature_1"]
    batch_size = 8
    device = torch.device("cpu")

    # Create model instance
    tft = TemporalFusionTransformer(
        number_of_past_inputs=number_of_past_inputs,
        number_of_future_inputs=number_of_future_inputs,
        embedding_size_inputs=embedding_size_inputs,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
        number_of_heads=number_of_heads,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        static_inputs=static_inputs,
        batch_size=batch_size,
        device=device,
    )

    # Create input tensors
    past_data = torch.randn(batch_size, number_of_past_inputs, embedding_size_inputs)
    future_data = torch.randn(batch_size, number_of_future_inputs, embedding_size_inputs)
    static_data = torch.randn(batch_size, embedding_size_inputs)

    # Forward pass
    past_inputs_dict = {"past_data": past_data}
    future_inputs_dict = {"future_data": future_data}
    output, attention_weights = tft(
        past_inputs=past_inputs_dict,
        future_inputs=future_inputs_dict,
        static_inputs=static_data,
    )

    # Check output shape
    assert output.shape == (batch_size, number_of_future_inputs, 1), "Output shape mismatch"

    # Check attention weights shape
    assert attention_weights.shape == (
        batch_size,
        number_of_heads,
        number_of_future_inputs,
        number_of_past_inputs + number_of_future_inputs,
    ), "Attention weights shape mismatch"


def test_tft_forward_without_future_inputs():
    # Define model parameters
    number_of_past_inputs = 10
    number_of_future_inputs = 0
    embedding_size_inputs = 16
    hidden_dimension = 32
    dropout = 0.1
    number_of_heads = 4
    past_inputs = ["past_feature_1", "past_feature_2"]
    future_inputs = None
    static_inputs = ["static_feature_1"]
    batch_size = 8
    device = torch.device("cpu")

    # Create model instance
    tft = TemporalFusionTransformer(
        number_of_past_inputs=number_of_past_inputs,
        number_of_future_inputs=number_of_future_inputs,
        embedding_size_inputs=embedding_size_inputs,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
        number_of_heads=number_of_heads,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        static_inputs=static_inputs,
        batch_size=batch_size,
        device=device,
    )

    # Create input tensors
    past_data = torch.randn(batch_size, number_of_past_inputs, embedding_size_inputs)
    static_data = torch.randn(batch_size, embedding_size_inputs)

    # Forward pass
    past_inputs_dict = {"past_data": past_data}
    output, attention_weights = tft(
        past_inputs=past_inputs_dict,
        future_inputs=None,
        static_inputs=static_data,
    )

    # Check output shape
    assert output.shape == (batch_size, 0, 1), "Output shape mismatch"

    # Check attention weights shape
    assert attention_weights.shape == (
        batch_size,
        number_of_heads,
        0,
        number_of_past_inputs,
    ), "Attention weights shape mismatch"


def test_tft_forward_without_static_inputs():
    # Define model parameters
    number_of_past_inputs = 10
    number_of_future_inputs = 5
    embedding_size_inputs = 16
    hidden_dimension = 32
    dropout = 0.1
    number_of_heads = 4
    past_inputs = ["past_feature_1", "past_feature_2"]
    future_inputs = ["future_feature_1"]
    static_inputs = None
    batch_size = 8
    device = torch.device("cpu")

    # Create model instance
    tft = TemporalFusionTransformer(
        number_of_past_inputs=number_of_past_inputs,
        number_of_future_inputs=number_of_future_inputs,
        embedding_size_inputs=embedding_size_inputs,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
        number_of_heads=number_of_heads,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        static_inputs=static_inputs,
        batch_size=batch_size,
        device=device,
    )

    # Create input tensors
    past_data = torch.randn(batch_size, number_of_past_inputs, embedding_size_inputs)
    future_data = torch.randn(batch_size, number_of_future_inputs, embedding_size_inputs)

    # Forward pass
    past_inputs_dict = {"past_data": past_data}
    future_inputs_dict = {"future_data": future_data}
    output, attention_weights = tft(
        past_inputs=past_inputs_dict,
        future_inputs=future_inputs_dict,
        static_inputs=None,
    )

    # Check output shape
    assert output.shape == (batch_size, number_of_future_inputs, 1), "Output shape mismatch"

    # Check attention weights shape
    assert attention_weights.shape == (
        batch_size,
        number_of_heads,
        number_of_future_inputs,
        number_of_past_inputs + number_of_future_inputs,
    ), "Attention weights shape mismatch"
