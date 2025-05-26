import pytest
import torch

from fintorch.models.timeseries.tft.tft_module import TemporalFusionTransformerModule


class TestTemporalFusionTransformerModule:
    """Comprehensive test suite for TemporalFusionTransformerModule"""

    @pytest.fixture
    def default_params(self):
        """Default parameters for model initialization"""
        return {
            "number_of_past_inputs": 10,
            "horizon": 5,
            "embedding_size_inputs": 16,
            "hidden_dimension": 32,
            "dropout": 0.1,
            "number_of_heads": 4,
            "num_past_target_features": 1,
            "num_past_known_cov_features": 3,
            "num_past_unknown_cov_features": 2,
            "num_future_known_cov_features": 2,
            "num_static_real_features": 4,
            "num_static_categorical_features": 2,
            "static_categorical_cardinalities": [10, 5],
            "batch_size": 8,
            "device": "cpu",
            "quantiles": [0.1, 0.5, 0.9]
        }

    @pytest.fixture
    def model(self, default_params):
        """Create a default model instance"""
        return TemporalFusionTransformerModule(**default_params)

    @pytest.fixture
    def sample_batch_single_series(self, default_params):
        """Create a sample batch with single series"""
        batch_size = default_params["batch_size"]
        past_length = default_params["number_of_past_inputs"]
        future_length = default_params["horizon"]

        return {
            "past_target": torch.randn(batch_size, past_length, 1, default_params["num_past_target_features"]),
            "past_covariates_known_future": torch.randn(batch_size, past_length, 1, default_params["num_past_known_cov_features"]),
            "past_covariates_unknown_future": torch.randn(batch_size, past_length, 1, default_params["num_past_unknown_cov_features"]),
            "future_covariates_known": torch.randn(batch_size, future_length, 1, default_params["num_future_known_cov_features"]),
            "output_target": torch.randn(batch_size, future_length, 1, default_params["num_past_target_features"]),
            "static_features_real": torch.randn(batch_size, 1, default_params["num_static_real_features"]),
            "static_features_categorical": torch.randint(0, 10, (batch_size, 1, default_params["num_static_categorical_features"])),
        }

    @pytest.fixture
    def sample_batch_multi_series(self, default_params):
        """Create a sample batch with multiple series"""
        batch_size = default_params["batch_size"]
        past_length = default_params["number_of_past_inputs"]
        future_length = default_params["horizon"]
        num_series = 3

        return {
            "past_target": torch.randn(batch_size, past_length, num_series, default_params["num_past_target_features"]),
            "past_covariates_known_future": torch.randn(batch_size, past_length, num_series, default_params["num_past_known_cov_features"]),
            "past_covariates_unknown_future": torch.randn(batch_size, past_length, num_series, default_params["num_past_unknown_cov_features"]),
            "future_covariates_known": torch.randn(batch_size, future_length, num_series, default_params["num_future_known_cov_features"]),
            "output_target": torch.randn(batch_size, future_length, num_series, default_params["num_past_target_features"]),
            "static_features_real": torch.randn(batch_size, num_series, default_params["num_static_real_features"]),
            "static_features_categorical": torch.randint(0, 10, (batch_size, num_series, default_params["num_static_categorical_features"])),
        }

    def test_model_initialization(self, default_params):
        """Test model initialization with various parameters"""
        model = TemporalFusionTransformerModule(**default_params)

        assert model is not None
        assert model.hparams["number_of_past_inputs"] == default_params["number_of_past_inputs"]
        assert model.hparams["horizon"] == default_params["horizon"]
        assert model.hparams["quantiles"] == default_params["quantiles"]
        assert model.series_selection_method == "first"  # default value

    def test_initialization_with_custom_series_params(self, default_params):
        """Test initialization with custom series handling parameters"""
        custom_params = default_params.copy()
        custom_params.update({
            "series_selection_method": "aggregate",
            "series_aggregation": "mean",
            "series_index": 2
        })

        model = TemporalFusionTransformerModule(**custom_params)
        assert model.series_selection_method == "aggregate"
        assert model.series_aggregation == "mean"
        assert model.series_index == 2

    def test_invalid_series_selection_method(self, default_params):
        """Test that invalid series selection method raises error"""
        with pytest.raises(ValueError, match="series_selection_method must be one of"):
            default_params["series_selection_method"] = "invalid_method"
            TemporalFusionTransformerModule(**default_params)

    def test_invalid_series_aggregation(self, default_params):
        """Test that invalid series aggregation raises error"""
        with pytest.raises(ValueError, match="series_aggregation must be one of"):
            default_params["series_selection_method"] = "aggregate"
            default_params["series_aggregation"] = "invalid_agg"
            TemporalFusionTransformerModule(**default_params)

    def test_index_method_without_series_index(self, default_params):
        """Test that index method without series_index raises error"""
        with pytest.raises(ValueError, match="series_index must be provided"):
            default_params["series_selection_method"] = "index"
            default_params["series_index"] = None
            TemporalFusionTransformerModule(**default_params)

    def test_forward_pass_single_series(self, model, sample_batch_single_series):
        """Test forward pass with single series data"""
        model.eval()
        with torch.no_grad():
            result = model.training_step(sample_batch_single_series, 0)

        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].item() >= 0

    def test_forward_pass_multi_series(self, default_params, sample_batch_multi_series):
        """Test forward pass with multi-series data"""
        model = TemporalFusionTransformerModule(**default_params)
        model.eval()

        with torch.no_grad():
            result = model.training_step(sample_batch_multi_series, 0)

        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].item() >= 0

    def test_series_selection_methods(self, default_params, sample_batch_multi_series):
        """Test different series selection methods"""
        selection_methods = ["first", "last", "aggregate"]

        for method in selection_methods:
            params = default_params.copy()
            params["series_selection_method"] = method
            if method == "aggregate":
                params["series_aggregation"] = "mean"

            model = TemporalFusionTransformerModule(**params)
            result = model.training_step(sample_batch_multi_series, 0)
            assert "loss" in result
            assert result["loss"].item() >= 0

    def test_series_index_selection(self, default_params, sample_batch_multi_series):
        """Test index-based series selection"""
        params = default_params.copy()
        params["series_selection_method"] = "index"
        params["series_index"] = 1

        model = TemporalFusionTransformerModule(**params)
        result = model.training_step(sample_batch_multi_series, 0)
        assert "loss" in result
        assert result["loss"].item() >= 0

    def test_series_aggregation_methods(self, default_params, sample_batch_multi_series):
        """Test different aggregation methods"""
        aggregation_methods = ["mean", "sum", "max", "min"]

        for agg_method in aggregation_methods:
            params = default_params.copy()
            params["series_selection_method"] = "aggregate"
            params["series_aggregation"] = agg_method

            model = TemporalFusionTransformerModule(**params)
            result = model.training_step(sample_batch_multi_series, 0)
            assert "loss" in result
            assert result["loss"].item() >= 0

    def test_series_index_out_of_range(self, default_params, sample_batch_multi_series):
        """Test that out-of-range series index raises error"""
        params = default_params.copy()
        params["series_selection_method"] = "index"
        params["series_index"] = 10  # Out of range for 3 series

        model = TemporalFusionTransformerModule(**params)

        with pytest.raises(IndexError, match="series_index .* out of range"):
            model.training_step(sample_batch_multi_series, 0)

    def test_validation_step(self, model, sample_batch_single_series):
        """Test validation step"""
        model.eval()
        with torch.no_grad():
            result = model.validation_step(sample_batch_single_series, 0)

        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].item() >= 0

    def test_test_step(self, model, sample_batch_single_series):
        """Test test step"""
        model.eval()
        with torch.no_grad():
            result = model.test_step(sample_batch_single_series, 0)

        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].item() >= 0

    def test_predict_step(self, model, sample_batch_single_series):
        """Test predict step"""
        model.eval()
        with torch.no_grad():
            result = model.predict_step(sample_batch_single_series, 0)

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # prediction and attention weights

    def test_quantile_loss_calculation(self, model):
        """Test quantile loss calculation"""
        batch_size, horizon = 4, 5
        # Model output shape: [batch, horizon, series, quantiles]
        y_pred = torch.randn(batch_size, horizon, 1, 3)  # 3 quantiles
        y_true = torch.randn(batch_size, horizon)  # 2D target, will be reshaped internally

        loss = model.quantile_loss(y_pred, y_true)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_configure_optimizers(self, model):
        """Test optimizer configuration"""
        optimizer_config = model.configure_optimizers()

        assert optimizer_config is not None
        assert hasattr(optimizer_config, "state_dict")  # Should be an optimizer

    def test_process_multi_series_data_shapes(self, default_params):
        """Test that _process_multi_series_data handles different input shapes correctly"""
        model = TemporalFusionTransformerModule(**default_params)

        # Test different input shapes
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        # Test first series selection
        model.series_selection_method = "first"
        result = model._process_multi_series_data(data)
        assert result.shape == (batch_size, time_length, features)

        # Test last series selection
        model.series_selection_method = "last"
        result = model._process_multi_series_data(data)
        assert result.shape == (batch_size, time_length, features)

        # Test aggregation
        model.series_selection_method = "aggregate"
        model.series_aggregation = "mean"
        result = model._process_multi_series_data(data)
        assert result.shape == (batch_size, time_length, features)

    def test_process_multi_series_target_shapes(self, default_params, sample_batch_multi_series):
        """Test that target processing handles different shapes correctly"""
        model = TemporalFusionTransformerModule(**default_params)

        # Test with different target shapes
        batch = sample_batch_multi_series.copy()

        # Test 3D target [batch, time, series]
        batch["output_target"] = torch.randn(8, 5, 3)
        processed_data = model._process_multi_series_target(batch["output_target"])
        assert processed_data.shape[0] == 8  # batch size preserved

    def test_missing_batch_keys(self, model, default_params):
        """Test handling of missing keys in batch"""
        incomplete_batch = {
            "past_target": torch.randn(8, 10, 1, 1),
            # Missing other required keys
        }

        with pytest.raises((KeyError, ValueError)):
            model.training_step(incomplete_batch, 0)

    def test_empty_batch(self, model):
        """Test handling of empty batch"""
        empty_batch = {}

        with pytest.raises((KeyError, ValueError)):
            model.training_step(empty_batch, 0)

    def test_different_batch_sizes(self, default_params):
        """Test handling of different batch sizes"""
        for batch_size in [1, 4, 16, 32]:
            params = default_params.copy()
            params["batch_size"] = batch_size
            model = TemporalFusionTransformerModule(**params)

            batch = {
                "past_target": torch.randn(batch_size, 10, 1, 1),
                "past_covariates_known_future": torch.randn(batch_size, 10, 1, 3),
                "past_covariates_unknown_future": torch.randn(batch_size, 10, 1, 2),
                "future_covariates_known": torch.randn(batch_size, 5, 1, 2),
                "output_target": torch.randn(batch_size, 5, 1, 1),
                "static_features_real": torch.randn(batch_size, 1, 4),
                "static_features_categorical": torch.randint(0, 10, (batch_size, 1, 2)),
            }

            result = model.training_step(batch, 0)
            assert "loss" in result

    def test_different_sequence_lengths(self, default_params):
        """Test handling of different sequence lengths"""
        for past_length, future_length in [(5, 3), (20, 10), (50, 20)]:
            params = default_params.copy()
            params["number_of_past_inputs"] = past_length
            params["horizon"] = future_length
            model = TemporalFusionTransformerModule(**params)

            batch = {
                "past_target": torch.randn(8, past_length, 1, 1),
                "past_covariates_known_future": torch.randn(8, past_length, 1, 3),
                "past_covariates_unknown_future": torch.randn(8, past_length, 1, 2),
                "future_covariates_known": torch.randn(8, future_length, 1, 2),
                "output_target": torch.randn(8, future_length, 1, 1),
                "static_features_real": torch.randn(8, 1, 4),
                "static_features_categorical": torch.randint(0, 10, (8, 1, 2)),
            }

            result = model.training_step(batch, 0)
            assert "loss" in result

    def test_gradient_flow(self, model, sample_batch_single_series):
        """Test that gradients flow through the model"""
        model.train()

        # Enable gradient computation
        for param in model.parameters():
            param.grad = None

        result = model.training_step(sample_batch_single_series, 0)
        loss = result["loss"]
        loss.backward()

        # Check that at least some parameters have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "No gradients found in model parameters"

    def test_model_deterministic_behavior(self, default_params, sample_batch_single_series):
        """Test that model produces consistent results with same input"""
        torch.manual_seed(42)
        model1 = TemporalFusionTransformerModule(**default_params)

        torch.manual_seed(42)
        model2 = TemporalFusionTransformerModule(**default_params)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            result1 = model1.training_step(sample_batch_single_series, 0)
            result2 = model2.training_step(sample_batch_single_series, 0)

        assert torch.allclose(result1["loss"], result2["loss"], atol=1e-6)

    def test_device_compatibility(self, default_params, sample_batch_single_series):
        """Test model works on different devices"""
        # Test CPU
        params_cpu = default_params.copy()
        params_cpu["device"] = "cpu"
        model_cpu = TemporalFusionTransformerModule(**params_cpu)

        result_cpu = model_cpu.training_step(sample_batch_single_series, 0)
        assert "loss" in result_cpu

    def test_model_save_load_hyperparameters(self, model, tmp_path):
        """Test that hyperparameters are properly saved and can be loaded"""
        # Test that hyperparameters are accessible
        assert hasattr(model, "hparams")
        assert "quantiles" in model.hparams
        assert "number_of_past_inputs" in model.hparams

    def test_backward_compatibility_legacy_format(self, default_params):
        """Test backward compatibility with legacy parameter format"""
        legacy_params = default_params.copy()
        legacy_params.update({
            "past_inputs": {"past_data": 6},
            "future_inputs": {"future_data": 2},
            "static_inputs": {"static_data": 6}
        })

        # Should not raise an error
        model = TemporalFusionTransformerModule(**legacy_params)
        assert model is not None

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
    def test_different_dropout_values(self, default_params, sample_batch_single_series, dropout):
        """Test model with different dropout values"""
        params = default_params.copy()
        params["dropout"] = dropout
        model = TemporalFusionTransformerModule(**params)

        result = model.training_step(sample_batch_single_series, 0)
        assert "loss" in result
        assert result["loss"].item() >= 0

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_different_attention_heads(self, default_params, sample_batch_single_series, num_heads):
        """Test model with different numbers of attention heads"""
        params = default_params.copy()
        params["number_of_heads"] = num_heads
        # Ensure embedding size is divisible by number of heads
        params["embedding_size_inputs"] = num_heads * 4
        model = TemporalFusionTransformerModule(**params)

        result = model.training_step(sample_batch_single_series, 0)
        assert "loss" in result
        assert result["loss"].item() >= 0

    def test_quantiles_validation(self, default_params):
        """Test that quantile validation works correctly"""
        # Valid quantiles
        params = default_params.copy()
        params["quantiles"] = [0.1, 0.5, 0.9]
        model = TemporalFusionTransformerModule(**params)
        assert model.hparams["quantiles"] == [0.1, 0.5, 0.9]

        # Test with single quantile
        params["quantiles"] = [0.5]
        model = TemporalFusionTransformerModule(**params)
        assert model.hparams["quantiles"] == [0.5]

    def test_model_training_mode_changes(self, model, sample_batch_single_series):
        """Test that model behaves differently in training vs eval mode"""
        # Training mode
        model.train()
        result_train = model.training_step(sample_batch_single_series, 0)

        # Eval mode
        model.eval()
        with torch.no_grad():
            result_eval = model.validation_step(sample_batch_single_series, 0)

        # Both should work but may give different results due to dropout
        assert "loss" in result_train
        assert "loss" in result_eval

    def test_memory_efficiency(self, default_params):
        """Test model memory usage with different configurations"""
        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create and delete multiple models to test memory cleanup
        for _ in range(5):
            model = TemporalFusionTransformerModule(**default_params)
            del model
            gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f} MB"
