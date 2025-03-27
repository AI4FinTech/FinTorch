import torch

from fintorch.models.timeseries.tft.tft_module import TemporalFusionTransformerModule


def test_forward_pass():
    # Define model parameters
    number_of_past_inputs = 4
    horizon = 2
    embedding_size_inputs = 8
    hidden_dimension = 16
    dropout = 0.1
    number_of_heads = 2
    past_inputs = {"feature1": 4, "feature2": 3}
    future_inputs = {"feature1": 4}
    static_inputs = {"feature3": 4, "feature4": 3}
    batch_size = 8
    device = "cpu"

    # Create model instance
    model = TemporalFusionTransformerModule(
        number_of_past_inputs,
        horizon,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        past_inputs,
        future_inputs,
        static_inputs,
        batch_size,
        device,
    )

    # Create dummy inputs
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

    # Forward pass
    output = model(past_inputs_dict, future_inputs_dict, static_data)

    # Check output shape
    assert output is not None, "Output is None"
    assert isinstance(output, tuple), "Output is not a tuple"
    assert len(output) == 2, "Output tuple does not contain two elements"
    assert output[0].shape[0] == batch_size, "Output batch size mismatch"


def test_training_step():
    # Define model parameters
    number_of_past_inputs = 4
    horizon = 2
    embedding_size_inputs = 8
    hidden_dimension = 16
    dropout = 0.1
    number_of_heads = 2
    past_inputs = {"feature1": 4, "feature2": 3}
    future_inputs = {"feature1": 4}
    static_inputs = {"feature3": 4, "feature4": 3}
    batch_size = 8
    device = "cpu"
    quantiles = [0.9]

    # Create model instance
    model = TemporalFusionTransformerModule(
        number_of_past_inputs,
        horizon,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        past_inputs,
        future_inputs,
        static_inputs,
        batch_size,
        device,
        quantiles = quantiles,
    )

    # Create dummy inputs
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

    # Generate target tensor
    target = torch.randn(batch_size, horizon)

    # Forward pass
    batch = (past_inputs_dict, future_inputs_dict, static_data, target)

    # Perform training step
    loss = model.training_step(batch, 0)

    # Check loss
    assert loss is not None, "Loss is None"
    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
    assert loss.item() > 0, "Loss is not positive"
