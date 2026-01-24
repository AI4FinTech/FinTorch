import pytest
import torch
from unittest.mock import patch

from fintorch.models.timeseries.causalformer.causalformer_module import CausalFormerModule


class TestCausalFormerModule:
    """Comprehensive test suite for CausalFormerModule"""

    @pytest.fixture
    def default_params(self):
        """Default parameters for CausalFormer testing"""
        return {
            "number_of_layers": 2,
            "number_of_heads": 2,
            "number_of_series": 1,
            "length_input_window": 20,
            "length_output_window": 10,
            "embedding_size": 64,
            "feature_dimensionality": 6,  # 1 + 3 + 2 (past_target + past_known + past_unknown)
            "ffn_hidden_dimensionality": 128,
            "output_dimensionality": 1,
            "tau": 1.0,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "lr_step_size": 30,
            "lr_gamma": 0.1,
            "weight_decay": 0.0001,
            "series_selection_method": "first",
            "series_index": 0,
            "series_aggregation": "mean"
        }

    @pytest.fixture
    def model(self, default_params):
        """Create a default model instance"""
        return CausalFormerModule(**default_params)

    @pytest.fixture
    def sample_batch_single_series(self, default_params):
        """Create a sample batch with single series"""
        batch_size = 8
        input_length = default_params["length_input_window"]
        output_length = default_params["length_output_window"]

        return {
            "past_target": torch.randn(batch_size, input_length, 1, 1),
            "past_covariates_known_future": torch.randn(batch_size, input_length, 1, 3),
            "past_covariates_unknown_future": torch.randn(batch_size, input_length, 1, 2),
            "output_target": torch.randn(batch_size, output_length, 1, 1),
        }

    @pytest.fixture
    def sample_batch_multi_series(self, default_params):
        """Create a sample batch with multiple series"""
        batch_size = 8
        input_length = default_params["length_input_window"]
        output_length = default_params["length_output_window"]
        num_series = 5

        return {
            "past_target": torch.randn(batch_size, input_length, num_series, 1),
            "past_covariates_known_future": torch.randn(batch_size, input_length, num_series, 3),
            "past_covariates_unknown_future": torch.randn(batch_size, input_length, num_series, 2),
            "output_target": torch.randn(batch_size, output_length, num_series, 1),
        }

    @pytest.fixture
    def sample_batch_3d(self, default_params):
        """Create a sample batch with 3D tensors (no series dimension)"""
        batch_size = 8
        input_length = default_params["length_input_window"]
        output_length = default_params["length_output_window"]

        return {
            "past_target": torch.randn(batch_size, input_length, 6),
            "output_target": torch.randn(batch_size, output_length),
        }

    def test_model_initialization(self, default_params):
        """Test model initialization with default parameters"""
        model = CausalFormerModule(**default_params)

        assert model is not None
        assert model.hparams["number_of_layers"] == default_params["number_of_layers"]
        assert model.hparams["learning_rate"] == default_params["learning_rate"]
        assert model.series_selection_method == "first"
        assert model.series_index == 0
        assert model.series_aggregation == "mean"

    def test_initialization_with_custom_series_params(self, default_params):
        """Test initialization with custom series handling parameters"""
        custom_params = default_params.copy()
        custom_params.update({
            "series_selection_method": "aggregate",
            "series_aggregation": "sum",
            "series_index": 2
        })

        model = CausalFormerModule(**custom_params)
        assert model.series_selection_method == "aggregate"
        assert model.series_aggregation == "sum"
        assert model.series_index == 2

    def test_invalid_series_selection_method(self, default_params):
        """Test that invalid series selection method raises error"""
        with pytest.raises(ValueError, match="series_selection_method must be one of"):
            default_params["series_selection_method"] = "invalid_method"
            CausalFormerModule(**default_params)

    def test_invalid_series_aggregation(self, default_params):
        """Test that invalid series aggregation raises error"""
        with pytest.raises(ValueError, match="series_aggregation must be one of"):
            default_params["series_selection_method"] = "aggregate"
            default_params["series_aggregation"] = "invalid_agg"
            CausalFormerModule(**default_params)

    def test_index_method_without_series_index(self, default_params):
        """Test that index method without series_index raises error"""
        with pytest.raises(ValueError, match="series_index must be provided"):
            default_params["series_selection_method"] = "index"
            default_params["series_index"] = None
            CausalFormerModule(**default_params)

    def test_forward_pass(self, model):
        """Test basic forward pass"""
        batch_size, num_series, seq_len, features = 4, 1, 20, 6  # Match feature_dimensionality
        x = torch.randn(batch_size, num_series, seq_len, features)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_series, 10, 1)
        assert torch.isfinite(output).all()
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size

    def test_training_step_single_series(self, model, sample_batch_single_series):
        """Test training step with single series data"""
        result = model.training_step(sample_batch_single_series, 0)

        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].item() >= 0

    def test_training_step_multi_series(self, default_params, sample_batch_multi_series):
        """Test training step with multi-series data"""
        model = CausalFormerModule(**default_params)
        result = model.training_step(sample_batch_multi_series, 0)

        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].item() >= 0

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
        assert isinstance(result, torch.Tensor)

    def test_process_multi_series_data_first(self, model):
        """Test first series selection method"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "first"
        result = model._process_multi_series_data(data)

        assert result.shape == (batch_size, time_length, features)
        assert torch.allclose(result, data[:, :, 0, :])

    def test_process_multi_series_data_last(self, model):
        """Test last series selection method"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "last"
        result = model._process_multi_series_data(data)

        assert result.shape == (batch_size, time_length, features)
        assert torch.allclose(result, data[:, :, -1, :])

    def test_process_multi_series_data_index(self, model):
        """Test index-based series selection"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "index"
        model.series_index = 1
        result = model._process_multi_series_data(data)

        assert result.shape == (batch_size, time_length, features)
        assert torch.allclose(result, data[:, :, 1, :])

    def test_process_multi_series_data_index_out_of_range(self, model):
        """Test that out-of-range series index raises error"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "index"
        model.series_index = 5  # Out of range

        with pytest.raises(IndexError, match="series_index .* out of range"):
            model._process_multi_series_data(data)

    def test_process_multi_series_data_aggregate_mean(self, model):
        """Test mean aggregation method"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "aggregate"
        model.series_aggregation = "mean"
        result = model._process_multi_series_data(data)

        assert result.shape == (batch_size, time_length, features)
        expected = data.mean(dim=2)
        assert torch.allclose(result, expected)

    def test_process_multi_series_data_aggregate_sum(self, model):
        """Test sum aggregation method"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "aggregate"
        model.series_aggregation = "sum"
        result = model._process_multi_series_data(data)

        assert result.shape == (batch_size, time_length, features)
        expected = data.sum(dim=2)
        assert torch.allclose(result, expected)

    def test_process_multi_series_data_aggregate_max(self, model):
        """Test max aggregation method"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "aggregate"
        model.series_aggregation = "max"
        result = model._process_multi_series_data(data)

        assert result.shape == (batch_size, time_length, features)
        expected = data.max(dim=2)[0]
        assert torch.allclose(result, expected)

    def test_process_multi_series_data_aggregate_min(self, model):
        """Test min aggregation method"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "aggregate"
        model.series_aggregation = "min"
        result = model._process_multi_series_data(data)

        assert result.shape == (batch_size, time_length, features)
        expected = data.min(dim=2)[0]
        assert torch.allclose(result, expected)

    def test_process_multi_series_data_flatten(self, model):
        """Test flatten method"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "flatten"
        result = model._process_multi_series_data(data)

        expected_features = num_series * features
        assert result.shape == (batch_size, time_length, expected_features)
        expected = data.view(batch_size, time_length, expected_features)
        assert torch.allclose(result, expected)

    def test_process_multi_series_data_unknown_method(self, model):
        """Test that unknown series selection method raises error"""
        batch_size, time_length, num_series, features = 4, 10, 3, 5
        data = torch.randn(batch_size, time_length, num_series, features)

        model.series_selection_method = "unknown_method"

        with pytest.raises(ValueError, match="Unknown series_selection_method"):
            model._process_multi_series_data(data)

    def test_prepare_data_3d_input(self, model, sample_batch_3d):
        """Test data preparation with 3D input tensors"""
        x, y = model._prepare_data(sample_batch_3d)

        assert x.ndim == 4  # Should be [batch, series, time, features]
        assert y.ndim == 4  # Should be [batch, series, time, features]
        assert x.shape[1] == 1  # Single series dimension should be added

    def test_prepare_data_4d_input(self, model, sample_batch_single_series):
        """Test data preparation with 4D input tensors"""
        x, y = model._prepare_data(sample_batch_single_series)

        assert x.ndim == 4  # Should be [batch, series, time, features]
        assert y.ndim == 4  # Should be [batch, series, time, features]

    def test_prepare_data_concatenation(self, model, sample_batch_single_series):
        """Test that past features are properly concatenated"""
        batch = sample_batch_single_series.copy()

        # Check original shapes
        past_target_features = batch["past_target"].shape[-1]
        past_known_features = batch["past_covariates_known_future"].shape[-1] if "past_covariates_known_future" in batch else 0
        past_unknown_features = batch["past_covariates_unknown_future"].shape[-1] if "past_covariates_unknown_future" in batch else 0

        x, y = model._prepare_data(batch)

        expected_features = past_target_features + past_known_features + past_unknown_features
        assert x.shape[-1] == expected_features

    def test_prepare_data_missing_past_features(self, model):
        """Test that missing past features raises error"""
        empty_batch = {"output_target": torch.randn(4, 10, 1, 1)}

        with pytest.raises(ValueError, match="No past features found in batch"):
            model._prepare_data(empty_batch)

    def test_prepare_data_target_shapes(self, model):
        """Test target tensor shape handling"""
        batch_size = 4

        # Test 2D target [batch, time]
        batch_2d = {
            "past_target": torch.randn(batch_size, 20, 1, 1),
            "past_covariates_known_future": torch.randn(batch_size, 20, 1, 3),
            "past_covariates_unknown_future": torch.randn(batch_size, 20, 1, 2),
            "output_target": torch.randn(batch_size, 10)
        }
        x, y = model._prepare_data(batch_2d)
        assert y.shape == (batch_size, 1, 10, 1)

        # Test 3D target [batch, time, series]
        batch_3d = {
            "past_target": torch.randn(batch_size, 20, 1, 1),
            "past_covariates_known_future": torch.randn(batch_size, 20, 1, 3),
            "past_covariates_unknown_future": torch.randn(batch_size, 20, 1, 2),
            "output_target": torch.randn(batch_size, 10, 1)
        }
        x, y = model._prepare_data(batch_3d)
        assert y.shape == (batch_size, 1, 10, 1)

    def test_series_selection_methods_integration(self, default_params, sample_batch_multi_series):
        """Test different series selection methods with actual training steps"""
        selection_methods = ["first", "last", "aggregate", "index", "flatten"]

        for method in selection_methods:
            params = default_params.copy()
            params["series_selection_method"] = method

            if method == "aggregate":
                params["series_aggregation"] = "mean"
            elif method == "index":
                params["series_index"] = 1
            elif method == "flatten":
                # For flatten method, adjust feature_dimensionality to account for multiple series
                params["feature_dimensionality"] = 6 * sample_batch_multi_series["past_target"].shape[2]  # 6 features * num_series

            model = CausalFormerModule(**params)
            result = model.training_step(sample_batch_multi_series, 0)

            assert "loss" in result
            assert torch.isfinite(result["loss"])
            assert result["loss"].item() >= 0

    def test_configure_optimizers(self, model):
        """Test optimizer configuration"""
        optimizer_config = model.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        assert optimizer_config["lr_scheduler"]["interval"] == "epoch"
        assert optimizer_config["lr_scheduler"]["frequency"] == 1

    def test_loss_function(self, model):
        """Test that loss function is properly configured"""
        assert hasattr(model, "loss")
        assert isinstance(model.loss, torch.nn.L1Loss)

    def test_gradient_flow(self, model, sample_batch_single_series):
        """Test that gradients flow through the model"""
        model.train()

        # Clear gradients
        for param in model.parameters():
            param.grad = None

        result = model.training_step(sample_batch_single_series, 0)
        loss = result["loss"]
        loss.backward()

        # Check that at least some parameters have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "No gradients found in model parameters"

    def test_model_deterministic_behavior(self, default_params, sample_batch_single_series):
        """Test that model produces consistent results with same input and seed"""
        torch.manual_seed(42)
        model1 = CausalFormerModule(**default_params)

        torch.manual_seed(42)
        model2 = CausalFormerModule(**default_params)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            result1 = model1.training_step(sample_batch_single_series, 0)
            result2 = model2.training_step(sample_batch_single_series, 0)

        assert torch.allclose(result1["loss"], result2["loss"], atol=1e-6)

    def test_different_batch_sizes(self, default_params):
        """Test handling of different batch sizes"""
        for batch_size in [1, 4, 16, 32]:
            model = CausalFormerModule(**default_params)

            batch = {
                "past_target": torch.randn(batch_size, 20, 1, 1),
                "past_covariates_known_future": torch.randn(batch_size, 20, 1, 3),
                "past_covariates_unknown_future": torch.randn(batch_size, 20, 1, 2),
                "output_target": torch.randn(batch_size, 10, 1, 1),
            }

            result = model.training_step(batch, 0)

            assert "loss" in result
            assert torch.isfinite(result["loss"])
            assert result["loss"].item() >= 0

    def test_different_sequence_lengths(self, default_params):
        """Test handling of different sequence lengths"""
        for input_len, output_len in [(10, 5), (50, 20), (100, 30)]:
            params = default_params.copy()
            params["length_input_window"] = input_len
            params["length_output_window"] = output_len
            model = CausalFormerModule(**params)

            batch = {
                "past_target": torch.randn(8, input_len, 1, 1),
                "past_covariates_known_future": torch.randn(8, input_len, 1, 3),
                "past_covariates_unknown_future": torch.randn(8, input_len, 1, 2),
                "output_target": torch.randn(8, output_len, 1, 1),
            }

            result = model.training_step(batch, 0)

            assert "loss" in result
            assert torch.isfinite(result["loss"])
            assert result["loss"].item() >= 0

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
    def test_different_dropout_values(self, default_params, sample_batch_single_series, dropout):
        """Test model with different dropout values"""
        params = default_params.copy()
        params["dropout"] = dropout
        model = CausalFormerModule(**params)

        result = model.training_step(sample_batch_single_series, 0)
        assert "loss" in result
        assert result["loss"].item() >= 0

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_different_attention_heads(self, default_params, sample_batch_single_series, num_heads):
        """Test model with different numbers of attention heads"""
        params = default_params.copy()
        params["number_of_heads"] = num_heads
        # Ensure embedding size is compatible with number of heads
        params["embedding_size"] = num_heads * 16
        model = CausalFormerModule(**params)

        result = model.training_step(sample_batch_single_series, 0)
        assert "loss" in result
        assert result["loss"].item() >= 0

    @pytest.mark.parametrize("num_layers", [1, 2, 4, 6])
    def test_different_layer_counts(self, default_params, sample_batch_single_series, num_layers):
        """Test model with different numbers of layers"""
        params = default_params.copy()
        params["number_of_layers"] = num_layers
        model = CausalFormerModule(**params)

        result = model.training_step(sample_batch_single_series, 0)
        assert "loss" in result
        assert result["loss"].item() >= 0

    def test_model_training_mode_changes(self, model, sample_batch_single_series):
        """Test that model behaves appropriately in training vs eval mode"""
        # Training mode
        model.train()
        result_train = model.training_step(sample_batch_single_series, 0)

        # Eval mode
        model.eval()
        with torch.no_grad():
            result_eval = model.validation_step(sample_batch_single_series, 0)

        # Both should work
        assert "loss" in result_train
        assert "loss" in result_eval

    def test_hyperparameters_saving(self, model):
        """Test that hyperparameters are properly saved"""
        assert hasattr(model, "hparams")
        assert "learning_rate" in model.hparams
        assert "number_of_layers" in model.hparams
        assert "dropout" in model.hparams

    def test_missing_batch_keys(self, model):
        """Test handling of missing keys in batch"""
        incomplete_batch = {
            "past_target": torch.randn(4, 20, 1, 1),
            # Missing output_target
        }

        with pytest.raises(KeyError):
            model.training_step(incomplete_batch, 0)

    def test_empty_batch(self, model):
        """Test handling of empty batch"""
        empty_batch = {}

        with pytest.raises((KeyError, ValueError)):
            model.training_step(empty_batch, 0)

    def test_nan_input_handling(self, model, sample_batch_single_series):
        """Test handling of NaN inputs"""
        batch = sample_batch_single_series.copy()
        batch["past_target"][0, 0, 0, 0] = float('nan')

        result = model.training_step(batch, 0)
        # The model should handle NaN gracefully or produce NaN loss
        assert "loss" in result

    def test_infinite_input_handling(self, model, sample_batch_single_series):
        """Test handling of infinite inputs"""
        batch = sample_batch_single_series.copy()
        batch["past_target"][0, 0, 0, 0] = float('inf')

        result = model.training_step(batch, 0)
        # The model should handle inf gracefully or produce appropriate loss
        assert "loss" in result

    def test_zero_input_handling(self, model, sample_batch_single_series):
        """Test handling of zero inputs"""
        batch = sample_batch_single_series.copy()
        batch["past_target"].fill_(0.0)
        batch["output_target"].fill_(0.0)

        result = model.training_step(batch, 0)
        assert "loss" in result
        assert result["loss"].item() >= 0

    def test_model_state_dict(self, model):
        """Test that model state dict can be saved and loaded"""
        state_dict = model.state_dict()
        assert len(state_dict) > 0

        # Create new model and load state dict
        new_model = CausalFormerModule(**model.hparams)
        new_model.load_state_dict(state_dict)

        # Check that parameters are the same
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_logging_integration(self, model, sample_batch_single_series):
        """Test that logging calls work properly"""
        # Mock the log method to verify it's called
        with patch.object(model, 'log') as mock_log:
            model.training_step(sample_batch_single_series, 0)
            mock_log.assert_called()

    def test_memory_efficiency(self, default_params):
        """Test model memory usage"""
        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create and delete multiple models to test memory cleanup
        for _ in range(3):
            model = CausalFormerModule(**default_params)
            del model
            gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f} MB"
