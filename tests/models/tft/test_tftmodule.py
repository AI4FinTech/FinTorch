import torch

from fintorch.models.timeseries.tft.tft_module import TemporalFusionTransformerModule


def test_forward_pass():
    # Define model parameters using new granular feature approach
    number_of_past_inputs = 4
    horizon = 2
    embedding_size_inputs = 8
    hidden_dimension = 16
    dropout = 0.1
    number_of_heads = 2
    num_past_target_features = 1
    num_past_known_cov_features = 2
    num_past_unknown_cov_features = 1
    num_future_known_cov_features = 2
    num_static_real_features = 2
    num_static_categorical_features = 1
    static_categorical_cardinalities = [5]
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
        num_past_target_features,
        num_past_known_cov_features,
        num_past_unknown_cov_features,
        num_future_known_cov_features,
        num_static_real_features,
        num_static_categorical_features,
        static_categorical_cardinalities,
        batch_size,
        device,
    )

    # Create dummy inputs in new standardized format
    past_inputs_dict = {
        "past_target": torch.randn(batch_size, number_of_past_inputs, num_past_target_features),
        "past_known_cov": torch.randn(batch_size, number_of_past_inputs, num_past_known_cov_features),
        "past_unknown_cov": torch.randn(batch_size, number_of_past_inputs, num_past_unknown_cov_features),
    }
    future_inputs_dict = {
        "future_known_cov": torch.randn(batch_size, horizon, num_future_known_cov_features),
    }
    static_data = {
        "static_real": torch.randn(batch_size, num_static_real_features),
        "static_categorical": torch.randint(0, 5, (batch_size, num_static_categorical_features)).float(),
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
    num_past_target_features = 1
    num_past_known_cov_features = 2
    num_past_unknown_cov_features = 1
    num_future_known_cov_features = 2
    num_static_real_features = 2
    num_static_categorical_features = 1
    static_categorical_cardinalities = [5]
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
        num_past_target_features,
        num_past_known_cov_features,
        num_past_unknown_cov_features,
        num_future_known_cov_features,
        num_static_real_features,
        num_static_categorical_features,
        static_categorical_cardinalities,
        batch_size,
        device,
        quantiles=quantiles,
    )

    # Create batch in new standardized format
    batch = {
        "past_target": torch.randn(batch_size, number_of_past_inputs, 1, num_past_target_features),
        "past_covariates_known_future": torch.randn(batch_size, number_of_past_inputs, 1, num_past_known_cov_features),
        "past_covariates_unknown_future": torch.randn(batch_size, number_of_past_inputs, 1, num_past_unknown_cov_features),
        "future_covariates_known": torch.randn(batch_size, horizon, 1, num_future_known_cov_features),
        "output_target": torch.randn(batch_size, horizon, 1, num_past_target_features),
        "static_features_real": torch.randn(batch_size, 1, num_static_real_features),
        "static_features_categorical": torch.randint(0, 5, (batch_size, 1, num_static_categorical_features)),
    }

    # Perform training step
    result = model.training_step(batch, 0)

    # Check loss
    assert result is not None, "Result is None"
    assert isinstance(result, dict), "Result is not a dictionary"
    assert "loss" in result, "Loss key not found in result"
    assert isinstance(result["loss"], torch.Tensor), "Loss is not a tensor"
    assert result["loss"].item() > 0, "Loss is not positive"


def test_series_selection_methods():
    # Test different series selection methods
    number_of_past_inputs = 4
    horizon = 2
    embedding_size_inputs = 8
    hidden_dimension = 16
    dropout = 0.1
    number_of_heads = 2
    num_past_target_features = 1
    num_past_known_cov_features = 2
    num_past_unknown_cov_features = 1
    num_future_known_cov_features = 2
    num_static_real_features = 2
    num_static_categorical_features = 1
    static_categorical_cardinalities = [5]
    batch_size = 4
    device = "cpu"

    # Test "first" method
    model_first = TemporalFusionTransformerModule(
        number_of_past_inputs,
        horizon,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        num_past_target_features,
        num_past_known_cov_features,
        num_past_unknown_cov_features,
        num_future_known_cov_features,
        num_static_real_features,
        num_static_categorical_features,
        static_categorical_cardinalities,
        batch_size,
        device,
        series_selection_method="first",
    )

    # Test "aggregate" method
    model_aggregate = TemporalFusionTransformerModule(
        number_of_past_inputs,
        horizon,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        num_past_target_features,
        num_past_known_cov_features,
        num_past_unknown_cov_features,
        num_future_known_cov_features,
        num_static_real_features,
        num_static_categorical_features,
        static_categorical_cardinalities,
        batch_size,
        device,
        series_selection_method="aggregate",
        series_aggregation="mean",
    )

    # Create multi-series batch
    num_series = 3
    batch = {
        "past_target": torch.randn(batch_size, number_of_past_inputs, num_series, num_past_target_features),
        "past_covariates_known_future": torch.randn(batch_size, number_of_past_inputs, num_series, num_past_known_cov_features),
        "past_covariates_unknown_future": torch.randn(batch_size, number_of_past_inputs, num_series, num_past_unknown_cov_features),
        "future_covariates_known": torch.randn(batch_size, horizon, num_series, num_future_known_cov_features),
        "output_target": torch.randn(batch_size, horizon, num_series, num_past_target_features),
        "static_features_real": torch.randn(batch_size, num_series, num_static_real_features),
        "static_features_categorical": torch.randint(0, 5, (batch_size, num_series, num_static_categorical_features)),
    }

    # Both should work without errors
    result_first = model_first.training_step(batch, 0)
    result_aggregate = model_aggregate.training_step(batch, 0)

    assert result_first["loss"].item() > 0
    assert result_aggregate["loss"].item() > 0


def test_validation_and_test_steps():
    # Define model parameters
    number_of_past_inputs = 4
    horizon = 2
    embedding_size_inputs = 8
    hidden_dimension = 16
    dropout = 0.1
    number_of_heads = 2
    num_past_target_features = 1
    num_past_known_cov_features = 2
    num_past_unknown_cov_features = 1
    num_future_known_cov_features = 2
    num_static_real_features = 2
    num_static_categorical_features = 1
    static_categorical_cardinalities = [5]
    batch_size = 4
    device = "cpu"

    # Create model instance
    model = TemporalFusionTransformerModule(
        number_of_past_inputs,
        horizon,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        num_past_target_features,
        num_past_known_cov_features,
        num_past_unknown_cov_features,
        num_future_known_cov_features,
        num_static_real_features,
        num_static_categorical_features,
        static_categorical_cardinalities,
        batch_size,
        device,
    )

    # Create batch
    batch = {
        "past_target": torch.randn(batch_size, number_of_past_inputs, 1, num_past_target_features),
        "past_covariates_known_future": torch.randn(batch_size, number_of_past_inputs, 1, num_past_known_cov_features),
        "past_covariates_unknown_future": torch.randn(batch_size, number_of_past_inputs, 1, num_past_unknown_cov_features),
        "future_covariates_known": torch.randn(batch_size, horizon, 1, num_future_known_cov_features),
        "output_target": torch.randn(batch_size, horizon, 1, num_past_target_features),
        "static_features_real": torch.randn(batch_size, 1, num_static_real_features),
        "static_features_categorical": torch.randint(0, 5, (batch_size, 1, num_static_categorical_features)),
    }

    # Test validation step
    val_result = model.validation_step(batch, 0)
    assert isinstance(val_result, dict)
    assert "loss" in val_result
    assert val_result["loss"].item() > 0

    # Test test step
    test_result = model.test_step(batch, 0)
    assert isinstance(test_result, dict)
    assert "loss" in test_result
    assert test_result["loss"].item() > 0

    # Test predict step
    pred_result = model.predict_step(batch, 0)
    assert pred_result is not None
    assert isinstance(pred_result, tuple)


def test_backward_compatibility():
    # Test that the model can still be initialized with legacy parameters
    number_of_past_inputs = 4
    horizon = 2
    embedding_size_inputs = 8
    hidden_dimension = 16
    dropout = 0.1
    number_of_heads = 2
    batch_size = 4
    device = "cpu"

    # Legacy format
    past_inputs = {"past_data": 3}
    future_inputs = {"future_data": 2}
    static_inputs = {"static_data": 2}

    # This should still work for backward compatibility
    model = TemporalFusionTransformerModule(
        number_of_past_inputs,
        horizon,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        num_past_target_features=1,
        num_past_known_cov_features=1,
        num_past_unknown_cov_features=1,
        num_future_known_cov_features=2,
        num_static_real_features=1,
        num_static_categorical_features=1,
        static_categorical_cardinalities=[5],
        batch_size=batch_size,
        device=device,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        static_inputs=static_inputs,
    )

    assert model is not None
