import torch

from fintorch.models.timeseries.tft.tft import TemporalFusionTransformer


def test_tft_forward_only_past_inputs():
    # Define model parameters
    number_of_past_inputs = 10
    horizon = 2
    embedding_size_inputs = hidden_dimension = 32
    dropout = 0.1
    number_of_heads = 4
    past_inputs = {"feature1": 4, "feature2": 3}
    future_inputs = None
    static_inputs = None
    batch_size = 8
    quantiles = [0.9]
    device = torch.device("cpu")

    # Create model instance
    tft = TemporalFusionTransformer(
        number_of_past_inputs=number_of_past_inputs,
        horizon=horizon,
        embedding_size_inputs=embedding_size_inputs,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
        number_of_heads=number_of_heads,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        static_inputs=static_inputs,
        batch_size=batch_size,
        device=device,
        quantiles=quantiles
    )

    # Forward pass
    past_inputs_dict = {
        "feature1": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature1"]
        ),
        "feature2": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature2"]
        ),
    }

    output, attention_weights = tft(
        past_inputs=past_inputs_dict,
        future_inputs=None,
        static_inputs=None,
    )

    # Check output shape
    # (batch size, horizon, targets, quantiles)
    assert output.shape == (batch_size, 2, 1, 1), "Output shape mismatch"

    # Check attention weights shape
    assert attention_weights.shape == (
        batch_size,
        number_of_heads,
        number_of_past_inputs,
        number_of_past_inputs,
    ), "Attention weights shape mismatch"


def test_tft_forward_with_all_inputs():
    # Define model parameters
    number_of_past_inputs = 10
    horizon = 5
    embedding_size_inputs = hidden_dimension = (
        32  # TODO: must be equal otherwise we have an error
    )
    dropout = 0.1
    number_of_heads = 4
    past_inputs = {"feature1": 4, "feature2": 3}
    future_inputs = {"feature1": 4}
    static_inputs = {"feature3": 4, "feature4": 3}
    batch_size = 8
    device = torch.device("cpu")
    quantiles = [0.9]

    # Create model instance
    tft = TemporalFusionTransformer(
        number_of_past_inputs=number_of_past_inputs,
        horizon=horizon,
        embedding_size_inputs=embedding_size_inputs,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
        number_of_heads=number_of_heads,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        static_inputs=static_inputs,
        batch_size=batch_size,
        device=device,
        quantiles=quantiles,
    )

    # Forward pass
    past_inputs_dict = {
        "feature1": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature1"]
        ),
        "feature2": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature2"]
        ),
    }
    future_inputs_dict = {
        "feature1": torch.randn(batch_size, horizon, future_inputs["feature1"]),
    }
    static_data = {
        "feature3": torch.randn(batch_size, static_inputs["feature3"]),
        "feature4": torch.randn(batch_size, static_inputs["feature4"]),
    }
    output, attention_weights = tft(
        past_inputs=past_inputs_dict,
        future_inputs=future_inputs_dict,
        static_inputs=static_data,
    )

    # Check output shape
    # (batch size, horizon, targets, quantiles)
    assert output.shape == (
        batch_size,
        horizon,
        1,
        1
    ), "Output shape mismatch"

    # Check attention weights shape
    # TODO: check expected shape, check what we expect in the other test of the attention-head
    assert attention_weights.shape == (
        batch_size,
        number_of_heads,
        number_of_past_inputs + horizon,
        number_of_past_inputs + horizon,
    ), "Attention weights shape mismatch"


def test_tft_forward_without_future_inputs():
    # Define model parameters
    number_of_past_inputs = 10
    horizon = 2  # TODO: make this horizon
    embedding_size_inputs = hidden_dimension = (
        32  # TODO: must be equal otherwise we have an error
    )
    dropout = 0.1
    number_of_heads = 4
    past_inputs = {"feature1": 4, "feature2": 3}
    future_inputs = None
    static_inputs = {"feature3": 4, "feature4": 3}
    batch_size = 8
    device = torch.device("cpu")
    quantiles = [0.9]

    # Create model instance
    tft = TemporalFusionTransformer(
        number_of_past_inputs=number_of_past_inputs,
        horizon=horizon,
        embedding_size_inputs=embedding_size_inputs,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
        number_of_heads=number_of_heads,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        static_inputs=static_inputs,
        batch_size=batch_size,
        device=device,
        quantiles=quantiles,
    )

    past_inputs_dict = {
        "feature1": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature1"]
        ),
        "feature2": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature2"]
        ),
    }
    static_data = {
        "feature3": torch.randn(batch_size, static_inputs["feature3"]),
        "feature4": torch.randn(batch_size, static_inputs["feature4"]),
    }
    output, attention_weights = tft(
        past_inputs=past_inputs_dict,
        future_inputs=None,
        static_inputs=static_data,
    )

    # Check output shape
    # (batch size, horizon, targets, quantiles)
    assert output.shape == (batch_size, 2, 1, 1), "Output shape mismatch"

    # Check attention weights shape
    assert attention_weights.shape == (
        batch_size,
        number_of_heads,
        10,
        number_of_past_inputs,
    ), "Attention weights shape mismatch"


def test_tft_forward_without_static_inputs():
    # Define model parameters
    number_of_past_inputs = 10
    horizon = 5
    embedding_size_inputs = hidden_dimension = (
        32  # TODO: must be equal otherwise we have an error
    )
    dropout = 0.1
    number_of_heads = 4
    past_inputs = {"feature1": 4, "feature2": 3}
    future_inputs = {"feature1": 4}
    batch_size = 8
    device = torch.device("cpu")
    quantiles = [0.9]

    # Create model instance
    tft = TemporalFusionTransformer(
        number_of_past_inputs=number_of_past_inputs,
        horizon=horizon,
        embedding_size_inputs=embedding_size_inputs,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
        number_of_heads=number_of_heads,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        static_inputs=None,
        batch_size=batch_size,
        device=device,
        quantiles=quantiles,
    )

    # Forward pass
    past_inputs_dict = {
        "feature1": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature1"]
        ),
        "feature2": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature2"]
        ),
    }
    future_inputs_dict = {
        "feature1": torch.randn(batch_size, horizon, future_inputs["feature1"]),
    }

    output, attention_weights = tft(
        past_inputs=past_inputs_dict,
        future_inputs=future_inputs_dict,
        static_inputs=None,
    )

    # Check output shape
    # (batch size, horizon, targets, quantiles)
    assert output.shape == (
        batch_size,
        horizon,
        1,
        1
    ), "Output shape mismatch"

    # Check attention weights shape
    assert attention_weights.shape == (
        batch_size,
        number_of_heads,
        number_of_past_inputs + horizon,
        number_of_past_inputs + horizon,
    ), "Attention weights shape mismatch"
