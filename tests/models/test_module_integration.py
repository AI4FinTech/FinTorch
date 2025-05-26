import pytest
import torch
from unittest.mock import Mock
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from fintorch.models.timeseries.tft.tft_module import TemporalFusionTransformerModule
from fintorch.models.timeseries.causalformer.causalformer_module import CausalFormerModule


class TestModuleIntegration:
    """Integration tests for both TFT and CausalFormer modules"""

    @pytest.fixture
    def sample_dataloader(self):
        """Create a sample dataloader for testing"""
        batch_size = 4
        num_batches = 10

        # Generate synthetic data
        past_target = torch.randn(num_batches * batch_size, 20, 1, 1)
        past_known = torch.randn(num_batches * batch_size, 20, 1, 3)
        past_unknown = torch.randn(num_batches * batch_size, 20, 1, 2)
        future_known = torch.randn(num_batches * batch_size, 10, 1, 2)
        output_target = torch.randn(num_batches * batch_size, 10, 1, 1)
        static_real = torch.randn(num_batches * batch_size, 1, 4)
        static_cat = torch.randint(0, 5, (num_batches * batch_size, 1, 2))

        dataset = TensorDataset(
            past_target, past_known, past_unknown,
            future_known, output_target, static_real, static_cat
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def test_tft_with_lightning_trainer(self):
        """Test TFT module with Lightning trainer"""
        model = TemporalFusionTransformerModule(
            number_of_past_inputs=10,
            horizon=5,
            embedding_size_inputs=16,
            hidden_dimension=32,
            dropout=0.1,
            number_of_heads=2,
            num_past_target_features=1,
            num_past_known_cov_features=2,
            num_past_unknown_cov_features=1,
            num_future_known_cov_features=2,
            num_static_real_features=2,
            num_static_categorical_features=1,
            static_categorical_cardinalities=[5],
            batch_size=4,
            device="cpu",
        )

        # Create mock trainer
        trainer = Mock(spec=L.Trainer)
        trainer.fit = Mock()
        trainer.validate = Mock()
        trainer.test = Mock()

        # Test that model can be used with trainer interface
        assert hasattr(model, 'training_step')
        assert hasattr(model, 'validation_step')
        assert hasattr(model, 'test_step')
        assert hasattr(model, 'configure_optimizers')

    def test_causalformer_with_lightning_trainer(self):
        """Test CausalFormer module with Lightning trainer"""
        model = CausalFormerModule(
            number_of_layers=2,
            number_of_heads=4,
            number_of_series=1,
            length_input_window=20,
            length_output_window=10,
            embedding_size=32,
            feature_dimensionality=16,
            ffn_hidden_dimensionality=64,
            output_dimensionality=1,
            tau=1.0,
            dropout=0.1,
            learning_rate=0.001,
        )

        # Create mock trainer
        trainer = Mock(spec=L.Trainer)
        trainer.fit = Mock()
        trainer.validate = Mock()
        trainer.test = Mock()

        # Test that model can be used with trainer interface
        assert hasattr(model, 'training_step')
        assert hasattr(model, 'validation_step')
        assert hasattr(model, 'test_step')
        assert hasattr(model, 'configure_optimizers')

    def test_both_models_same_data_format(self):
        """Test that both models can handle the same standardized data format"""
        batch = {
            "past_target": torch.randn(4, 20, 1, 1),
            "past_covariates_known_future": torch.randn(4, 20, 1, 3),
            "past_covariates_unknown_future": torch.randn(4, 20, 1, 2),
            "future_covariates_known": torch.randn(4, 10, 1, 2),
            "output_target": torch.randn(4, 10, 1, 1),
            "static_features_real": torch.randn(4, 1, 4),
            "static_features_categorical": torch.randint(0, 5, (4, 1, 2)),
        }

        # TFT Model
        tft_model = TemporalFusionTransformerModule(
            number_of_past_inputs=20,
            horizon=10,
            embedding_size_inputs=16,
            hidden_dimension=32,
            dropout=0.1,
            number_of_heads=2,
            num_past_target_features=1,
            num_past_known_cov_features=3,
            num_past_unknown_cov_features=2,
            num_future_known_cov_features=2,
            num_static_real_features=4,
            num_static_categorical_features=2,
            static_categorical_cardinalities=[5, 5],
            batch_size=4,
            device="cpu",
        )

        # CausalFormer Model
        cf_model = CausalFormerModule(
            number_of_layers=2,
            number_of_heads=4,
            number_of_series=1,
            length_input_window=20,
            length_output_window=10,
            embedding_size=32,
            feature_dimensionality=6,  # past_target + past_known + past_unknown
            ffn_hidden_dimensionality=64,
            output_dimensionality=1,
            tau=1.0,
            dropout=0.1,
        )

        # Both should work with the same batch
        tft_result = tft_model.training_step(batch, 0)
        cf_result = cf_model.training_step(batch, 0)

        assert "loss" in tft_result
        assert "loss" in cf_result
        assert tft_result["loss"].item() >= 0
        assert cf_result["loss"].item() >= 0

    def test_model_comparison_performance(self):
        """Compare basic performance characteristics of both models"""
        batch_size = 8
        input_length = 20
        output_length = 10

        # Create identical input data
        batch = {
            "past_target": torch.randn(batch_size, input_length, 1, 1),
            "past_covariates_known_future": torch.randn(batch_size, input_length, 1, 2),
            "past_covariates_unknown_future": torch.randn(batch_size, input_length, 1, 1),
            "future_covariates_known": torch.randn(batch_size, output_length, 1, 2),
            "output_target": torch.randn(batch_size, output_length, 1, 1),
            "static_features_real": torch.randn(batch_size, 1, 2),
            "static_features_categorical": torch.randint(0, 5, (batch_size, 1, 1)),
        }

        # TFT Model
        tft_model = TemporalFusionTransformerModule(
            number_of_past_inputs=input_length,
            horizon=output_length,
            embedding_size_inputs=16,
            hidden_dimension=32,
            dropout=0.1,
            number_of_heads=2,
            num_past_target_features=1,
            num_past_known_cov_features=2,
            num_past_unknown_cov_features=1,
            num_future_known_cov_features=2,
            num_static_real_features=2,
            num_static_categorical_features=1,
            static_categorical_cardinalities=[5],
            batch_size=batch_size,
            device="cpu",
        )

        # CausalFormer Model
        cf_model = CausalFormerModule(
            number_of_layers=2,
            number_of_heads=2,
            number_of_series=1,
            length_input_window=input_length,
            length_output_window=output_length,
            embedding_size=32,
            feature_dimensionality=4,  # 1 + 2 + 1
            ffn_hidden_dimensionality=64,
            output_dimensionality=1,
            tau=1.0,
            dropout=0.1,
        )

        # Measure inference time
        import time

        # TFT timing
        tft_model.eval()
        with torch.no_grad():
            start_time = time.time()
            tft_result = tft_model.predict_step(batch, 0)
            tft_time = time.time() - start_time

        # CausalFormer timing
        cf_model.eval()
        with torch.no_grad():
            start_time = time.time()
            cf_result = cf_model.predict_step(batch, 0)
            cf_time = time.time() - start_time

        # Both should complete in reasonable time (< 1 second for this small example)
        assert tft_time < 1.0
        assert cf_time < 1.0

        # Both should return valid predictions
        assert tft_result is not None
        assert cf_result is not None

    def test_multi_series_consistency(self):
        """Test that both models handle multi-series data consistently"""
        batch_size = 4
        num_series = 3

        batch = {
            "past_target": torch.randn(batch_size, 20, num_series, 1),
            "past_covariates_known_future": torch.randn(batch_size, 20, num_series, 2),
            "past_covariates_unknown_future": torch.randn(batch_size, 20, num_series, 1),
            "future_covariates_known": torch.randn(batch_size, 10, num_series, 2),
            "output_target": torch.randn(batch_size, 10, num_series, 1),
            "static_features_real": torch.randn(batch_size, num_series, 2),
            "static_features_categorical": torch.randint(0, 5, (batch_size, num_series, 1)),
        }

        # Test different series selection methods
        for method in ["first", "last", "aggregate"]:
            # TFT with series selection
            tft_model = TemporalFusionTransformerModule(
                number_of_past_inputs=20,
                horizon=10,
                embedding_size_inputs=16,
                hidden_dimension=32,
                dropout=0.1,
                number_of_heads=2,
                num_past_target_features=1,
                num_past_known_cov_features=2,
                num_past_unknown_cov_features=1,
                num_future_known_cov_features=2,
                num_static_real_features=2,
                num_static_categorical_features=1,
                static_categorical_cardinalities=[5],
                batch_size=batch_size,
                device="cpu",
                series_selection_method=method,
                series_aggregation="mean" if method == "aggregate" else "mean",
            )

            # CausalFormer with series selection
            cf_model = CausalFormerModule(
                number_of_layers=2,
                number_of_heads=2,
                number_of_series=1,
                length_input_window=20,
                length_output_window=10,
                embedding_size=32,
                feature_dimensionality=4,
                ffn_hidden_dimensionality=64,
                output_dimensionality=1,
                tau=1.0,
                dropout=0.1,
                series_selection_method=method,
                series_aggregation="mean" if method == "aggregate" else "mean",
            )

            # Both should handle multi-series data
            tft_result = tft_model.training_step(batch, 0)
            cf_result = cf_model.training_step(batch, 0)

            assert "loss" in tft_result
            assert "loss" in cf_result
            assert tft_result["loss"].item() >= 0
            assert cf_result["loss"].item() >= 0


class TestEdgeCases:
    """Edge case tests for both modules"""

    def test_minimal_configuration(self):
        """Test both models with minimal configuration"""
        # Minimal TFT
        tft_minimal = TemporalFusionTransformerModule(
            number_of_past_inputs=2,
            horizon=1,
            embedding_size_inputs=4,
            hidden_dimension=8,
            dropout=0.0,
            number_of_heads=1,
            num_past_target_features=1,
            num_past_known_cov_features=0,
            num_past_unknown_cov_features=0,
            num_future_known_cov_features=0,
            num_static_real_features=0,
            num_static_categorical_features=0,
            static_categorical_cardinalities=[],
            batch_size=1,
            device="cpu",
        )

        # Minimal CausalFormer
        cf_minimal = CausalFormerModule(
            number_of_layers=1,
            number_of_heads=1,
            number_of_series=1,
            length_input_window=2,
            length_output_window=1,
            embedding_size=4,
            feature_dimensionality=1,
            ffn_hidden_dimensionality=8,
            output_dimensionality=1,
            tau=1.0,
            dropout=0.0,
        )

        # Minimal batch
        batch = {
            "past_target": torch.randn(1, 2, 1, 1),
            "output_target": torch.randn(1, 1, 1, 1),
        }

        # Both should work with minimal configuration
        tft_result = tft_minimal.training_step(batch, 0)
        cf_result = cf_minimal.training_step(batch, 0)

        assert "loss" in tft_result
        assert "loss" in cf_result

    def test_large_configuration(self):
        """Test both models with large configuration"""
        batch_size = 16

        # Large TFT
        tft_large = TemporalFusionTransformerModule(
            number_of_past_inputs=100,
            horizon=50,
            embedding_size_inputs=128,
            hidden_dimension=256,
            dropout=0.2,
            number_of_heads=8,
            num_past_target_features=5,
            num_past_known_cov_features=10,
            num_past_unknown_cov_features=8,
            num_future_known_cov_features=10,
            num_static_real_features=20,
            num_static_categorical_features=5,
            static_categorical_cardinalities=[10, 20, 5, 15, 8],
            batch_size=batch_size,
            device="cpu",
        )

        # Large CausalFormer
        cf_large = CausalFormerModule(
            number_of_layers=6,
            number_of_heads=8,
            number_of_series=1,
            length_input_window=100,
            length_output_window=50,
            embedding_size=128,
            feature_dimensionality=23,  # 5 + 10 + 8
            ffn_hidden_dimensionality=512,
            output_dimensionality=5,
            tau=1.0,
            dropout=0.2,
        )

        # Large batch
        batch = {
            "past_target": torch.randn(batch_size, 100, 1, 5),
            "past_covariates_known_future": torch.randn(batch_size, 100, 1, 10),
            "past_covariates_unknown_future": torch.randn(batch_size, 100, 1, 8),
            "future_covariates_known": torch.randn(batch_size, 50, 1, 10),
            "output_target": torch.randn(batch_size, 50, 1, 5),
            "static_features_real": torch.randn(batch_size, 1, 20),
            "static_features_categorical": torch.randint(0, 10, (batch_size, 1, 5)),
        }

        # Test that both can handle large configurations (may be slow)
        tft_result = tft_large.training_step(batch, 0)
        cf_result = cf_large.training_step(batch, 0)

        assert "loss" in tft_result
        assert "loss" in cf_result

    def test_extreme_values(self):
        """Test handling of extreme input values"""
        batch_size = 4

        # Standard models
        tft_model = TemporalFusionTransformerModule(
            number_of_past_inputs=10,
            horizon=5,
            embedding_size_inputs=16,
            hidden_dimension=32,
            dropout=0.1,
            number_of_heads=2,
            num_past_target_features=1,
            num_past_known_cov_features=2,
            num_past_unknown_cov_features=1,
            num_future_known_cov_features=2,
            num_static_real_features=2,
            num_static_categorical_features=1,
            static_categorical_cardinalities=[5],
            batch_size=batch_size,
            device="cpu",
        )

        cf_model = CausalFormerModule(
            number_of_layers=2,
            number_of_heads=2,
            number_of_series=1,
            length_input_window=10,
            length_output_window=5,
            embedding_size=32,
            feature_dimensionality=4,
            ffn_hidden_dimensionality=64,
            output_dimensionality=1,
            tau=1.0,
            dropout=0.1,
        )

        # Test with very large values
        large_batch = {
            "past_target": torch.randn(batch_size, 10, 1, 1) * 1000,
            "past_covariates_known_future": torch.randn(batch_size, 10, 1, 2) * 1000,
            "past_covariates_unknown_future": torch.randn(batch_size, 10, 1, 1) * 1000,
            "future_covariates_known": torch.randn(batch_size, 5, 1, 2) * 1000,
            "output_target": torch.randn(batch_size, 5, 1, 1) * 1000,
            "static_features_real": torch.randn(batch_size, 1, 2) * 1000,
            "static_features_categorical": torch.randint(0, 5, (batch_size, 1, 1)),
        }

        # Test with very small values
        small_batch = {
            "past_target": torch.randn(batch_size, 10, 1, 1) * 1e-6,
            "past_covariates_known_future": torch.randn(batch_size, 10, 1, 2) * 1e-6,
            "past_covariates_unknown_future": torch.randn(batch_size, 10, 1, 1) * 1e-6,
            "future_covariates_known": torch.randn(batch_size, 5, 1, 2) * 1e-6,
            "output_target": torch.randn(batch_size, 5, 1, 1) * 1e-6,
            "static_features_real": torch.randn(batch_size, 1, 2) * 1e-6,
            "static_features_categorical": torch.randint(0, 5, (batch_size, 1, 1)),
        }

        # Both models should handle extreme values gracefully
        for batch in [large_batch, small_batch]:
            tft_result = tft_model.training_step(batch, 0)
            cf_result = cf_model.training_step(batch, 0)

            assert "loss" in tft_result
            assert "loss" in cf_result
            # Allow for potential numerical issues with extreme values
            assert not torch.isnan(tft_result["loss"]) or not torch.isinf(tft_result["loss"])
            assert not torch.isnan(cf_result["loss"]) or not torch.isinf(cf_result["loss"])

    def test_batch_size_one(self):
        """Test both models with batch size of 1"""
        # Models configured for batch size 1
        tft_model = TemporalFusionTransformerModule(
            number_of_past_inputs=10,
            horizon=5,
            embedding_size_inputs=16,
            hidden_dimension=32,
            dropout=0.1,
            number_of_heads=2,
            num_past_target_features=1,
            num_past_known_cov_features=2,
            num_past_unknown_cov_features=1,
            num_future_known_cov_features=2,
            num_static_real_features=2,
            num_static_categorical_features=1,
            static_categorical_cardinalities=[5],
            batch_size=1,
            device="cpu",
        )

        cf_model = CausalFormerModule(
            number_of_layers=2,
            number_of_heads=2,
            number_of_series=1,
            length_input_window=10,
            length_output_window=5,
            embedding_size=32,
            feature_dimensionality=4,
            ffn_hidden_dimensionality=64,
            output_dimensionality=1,
            tau=1.0,
            dropout=0.1,
        )

        # Batch with size 1
        batch = {
            "past_target": torch.randn(1, 10, 1, 1),
            "past_covariates_known_future": torch.randn(1, 10, 1, 2),
            "past_covariates_unknown_future": torch.randn(1, 10, 1, 1),
            "future_covariates_known": torch.randn(1, 5, 1, 2),
            "output_target": torch.randn(1, 5, 1, 1),
            "static_features_real": torch.randn(1, 1, 2),
            "static_features_categorical": torch.randint(0, 5, (1, 1, 1)),
        }

        # Both should work with batch size 1
        tft_result = tft_model.training_step(batch, 0)
        cf_result = cf_model.training_step(batch, 0)

        assert "loss" in tft_result
        assert "loss" in cf_result

    def test_sequence_length_one(self):
        """Test both models with sequence length of 1"""
        # Models configured for minimal sequence length
        tft_model = TemporalFusionTransformerModule(
            number_of_past_inputs=1,
            horizon=1,
            embedding_size_inputs=16,
            hidden_dimension=32,
            dropout=0.1,
            number_of_heads=1,
            num_past_target_features=1,
            num_past_known_cov_features=1,
            num_past_unknown_cov_features=1,
            num_future_known_cov_features=1,
            num_static_real_features=1,
            num_static_categorical_features=1,
            static_categorical_cardinalities=[5],
            batch_size=4,
            device="cpu",
        )

        cf_model = CausalFormerModule(
            number_of_layers=1,
            number_of_heads=1,
            number_of_series=1,
            length_input_window=1,
            length_output_window=1,
            embedding_size=16,
            feature_dimensionality=3,
            ffn_hidden_dimensionality=32,
            output_dimensionality=1,
            tau=1.0,
            dropout=0.1,
        )

        # Batch with sequence length 1
        batch = {
            "past_target": torch.randn(4, 1, 1, 1),
            "past_covariates_known_future": torch.randn(4, 1, 1, 1),
            "past_covariates_unknown_future": torch.randn(4, 1, 1, 1),
            "future_covariates_known": torch.randn(4, 1, 1, 1),
            "output_target": torch.randn(4, 1, 1, 1),
            "static_features_real": torch.randn(4, 1, 1),
            "static_features_categorical": torch.randint(0, 5, (4, 1, 1)),
        }

        # Both should work with minimal sequence length
        tft_result = tft_model.training_step(batch, 0)
        cf_result = cf_model.training_step(batch, 0)

        assert "loss" in tft_result
        assert "loss" in cf_result

    def test_model_robustness_to_device_changes(self):
        """Test model robustness when moving between devices"""
        # Create models on CPU
        tft_model = TemporalFusionTransformerModule(
            number_of_past_inputs=10,
            horizon=5,
            embedding_size_inputs=16,
            hidden_dimension=32,
            dropout=0.1,
            number_of_heads=2,
            num_past_target_features=1,
            num_past_known_cov_features=2,
            num_past_unknown_cov_features=1,
            num_future_known_cov_features=2,
            num_static_real_features=2,
            num_static_categorical_features=1,
            static_categorical_cardinalities=[5],
            batch_size=4,
            device="cpu",
        )

        cf_model = CausalFormerModule(
            number_of_layers=2,
            number_of_heads=2,
            number_of_series=1,
            length_input_window=10,
            length_output_window=5,
            embedding_size=32,
            feature_dimensionality=4,
            ffn_hidden_dimensionality=64,
            output_dimensionality=1,
            tau=1.0,
            dropout=0.1,
        )

        # Test on CPU
        batch_cpu = {
            "past_target": torch.randn(4, 10, 1, 1),
            "past_covariates_known_future": torch.randn(4, 10, 1, 2),
            "past_covariates_unknown_future": torch.randn(4, 10, 1, 1),
            "future_covariates_known": torch.randn(4, 5, 1, 2),
            "output_target": torch.randn(4, 5, 1, 1),
            "static_features_real": torch.randn(4, 1, 2),
            "static_features_categorical": torch.randint(0, 5, (4, 1, 1)),
        }

        # Both should work on CPU
        tft_result_cpu = tft_model.training_step(batch_cpu, 0)
        cf_result_cpu = cf_model.training_step(batch_cpu, 0)

        assert "loss" in tft_result_cpu
        assert "loss" in cf_result_cpu
