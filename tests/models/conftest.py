import pytest
import torch


@pytest.fixture
def performance_params_tft():
    """Parameters for TFT performance testing"""
    return {
        "number_of_past_inputs": 50,
        "horizon": 20,
        "embedding_size_inputs": 64,
        "hidden_dimension": 128,
        "dropout": 0.1,
        "number_of_heads": 8,
        "num_past_target_features": 3,
        "num_past_known_cov_features": 5,
        "num_past_unknown_cov_features": 4,
        "num_future_known_cov_features": 5,
        "num_static_real_features": 10,
        "num_static_categorical_features": 3,
        "static_categorical_cardinalities": [10, 20, 15],
        "batch_size": 32,
        "device": "cpu",
    }


@pytest.fixture
def performance_params_cf():
    """Parameters for CausalFormer performance testing"""
    return {
        "number_of_layers": 4,
        "number_of_heads": 8,
        "number_of_series": 1,
        "length_input_window": 50,
        "length_output_window": 20,
        "embedding_size": 128,
        "feature_dimensionality": 12,  # 3 + 5 + 4
        "ffn_hidden_dimensionality": 256,
        "output_dimensionality": 3,
        "tau": 1.0,
        "dropout": 0.1,
        "learning_rate": 0.001,
    }


@pytest.fixture
def large_batch():
    """Create a large batch for performance testing"""
    batch_size = 64
    return {
        "past_target": torch.randn(batch_size, 50, 1, 3),
        "past_covariates_known_future": torch.randn(batch_size, 50, 1, 5),
        "past_covariates_unknown_future": torch.randn(batch_size, 50, 1, 4),
        "future_covariates_known": torch.randn(batch_size, 20, 1, 5),
        "output_target": torch.randn(batch_size, 20, 1, 3),
        "static_features_real": torch.randn(batch_size, 1, 10),
        "static_features_categorical": torch.randint(0, 10, (batch_size, 1, 3)),
    }
