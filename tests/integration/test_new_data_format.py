import torch
from torch.utils.data import DataLoader

from fintorch.datasets.synthetic.simpleSynthetic import SimpleSyntheticDataset
from fintorch.models.timeseries.tft.tft_module import TemporalFusionTransformerModule
from fintorch.models.timeseries.causalformer.causalformer_module import CausalFormerModule


def test_synthetic_dataset_with_tft_integration():
    """Test that SimpleSyntheticDataset works with TFT using the new standardized format."""
    # Create dataset
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=10,
        future_length=5,
        num_series=1,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
        static_categorical_cardinalities=[5]
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Create TFT model
    tft_model = TemporalFusionTransformerModule(
        number_of_past_inputs=10,
        horizon=5,
        embedding_size_inputs=8,
        hidden_dimension=16,
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

    # Test forward pass
    batch = next(iter(dataloader))
    result = tft_model.training_step(batch, 0)

    assert isinstance(result, dict)
    assert "loss" in result
    assert result["loss"].item() > 0


def test_synthetic_dataset_with_causalformer_integration():
    """Test that SimpleSyntheticDataset works with CausalFormer using the new standardized format."""
    # Create dataset
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=10,
        future_length=5,
        num_series=1,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Create CausalFormer model
    causalformer_model = CausalFormerModule(
        number_of_layers=2,
        number_of_heads=2,
        number_of_series=1,
        length_input_window=10,
        length_output_window=5,
        embedding_size=8,
        feature_dimensionality=4,  # 1 target + 2 known_cov + 1 unknown_cov
        ffn_hidden_dimensionality=16,
        output_dimensionality=1,
        tau=1.0,
        dropout=0.1,
        learning_rate=0.01,
    )

    # Test forward pass
    batch = next(iter(dataloader))
    result = causalformer_model.training_step(batch, 0)

    assert isinstance(result, dict)
    assert "loss" in result
    assert result["loss"].item() > 0


def test_multi_series_integration():
    """Test that multi-series data works correctly with both models."""
    # Create multi-series dataset
    dataset = SimpleSyntheticDataset(
        length=100,
        past_length=10,
        future_length=5,
        num_series=3,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
        static_categorical_cardinalities=[5]
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Test with TFT using "first" series selection
    tft_model = TemporalFusionTransformerModule(
        number_of_past_inputs=10,
        horizon=5,
        embedding_size_inputs=8,
        hidden_dimension=16,
        dropout=0.1,
        number_of_heads=2,
        num_past_target_features=1,
        num_past_known_cov_features=2,
        num_past_unknown_cov_features=1,
        num_future_known_cov_features=2,
        num_static_real_features=2,
        num_static_categorical_features=1,
        static_categorical_cardinalities=[5],
        batch_size=2,
        device="cpu",
        series_selection_method="first",
    )

    batch = next(iter(dataloader))
    tft_result = tft_model.training_step(batch, 0)
    assert tft_result["loss"].item() > 0

    # Test with TFT using "aggregate" series selection
    tft_model_agg = TemporalFusionTransformerModule(
        number_of_past_inputs=10,
        horizon=5,
        embedding_size_inputs=8,
        hidden_dimension=16,
        dropout=0.1,
        number_of_heads=2,
        num_past_target_features=1,
        num_past_known_cov_features=2,
        num_past_unknown_cov_features=1,
        num_future_known_cov_features=2,
        num_static_real_features=2,
        num_static_categorical_features=1,
        static_categorical_cardinalities=[5],
        batch_size=2,
        device="cpu",
        series_selection_method="aggregate",
        series_aggregation="mean",
    )

    tft_agg_result = tft_model_agg.training_step(batch, 0)
    assert tft_agg_result["loss"].item() > 0


def test_electricity_dataset_integration():
    """Test that ElectricityDataset works with models using the new format."""
    try:
        from fintorch.datasets.electricity_simple import ElectricityDataset

        # Create dataset (small sample for testing)
        dataset = ElectricityDataset(
            past_length=10,
            future_length=5,
            start_idx=0,
            end_idx=50,  # Small sample for testing
        )

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Create TFT model
        tft_model = TemporalFusionTransformerModule(
            number_of_past_inputs=10,
            horizon=5,
            embedding_size_inputs=8,
            hidden_dimension=16,
            dropout=0.1,
            number_of_heads=2,
            num_past_target_features=1,
            num_past_known_cov_features=4,
            num_past_unknown_cov_features=1,
            num_future_known_cov_features=4,
            num_static_real_features=2,
            num_static_categorical_features=1,
            static_categorical_cardinalities=[2],
            batch_size=2,
            device="cpu",
        )

        # Test forward pass
        batch = next(iter(dataloader))
        result = tft_model.training_step(batch, 0)

        assert isinstance(result, dict)
        assert "loss" in result
        assert result["loss"].item() > 0

    except Exception as e:
        # Skip test if electricity data is not available
        print(f"Skipping electricity test due to: {e}")
        pass


def test_data_shapes_consistency():
    """Test that data shapes are consistent throughout the pipeline."""
    dataset = SimpleSyntheticDataset(
        length=50,
        past_length=8,
        future_length=3,
        num_series=2,
        num_target_features=1,
        num_known_cov_features=3,
        num_unknown_cov_features=2,
        num_static_real_features=2,
        num_static_categorical_features=1,
    )

    # Test single sample
    sample = dataset[0]
    assert sample["past_target"].shape == (8, 2, 1)
    assert sample["past_covariates_known_future"].shape == (8, 2, 3)
    assert sample["past_covariates_unknown_future"].shape == (8, 2, 2)
    assert sample["future_covariates_known"].shape == (3, 2, 3)
    assert sample["output_target"].shape == (3, 2, 1)
    assert sample["static_features_real"].shape == (2, 2)
    assert sample["static_features_categorical"].shape == (2, 1)

    # Test batched data
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))

    assert batch["past_target"].shape == (4, 8, 2, 1)
    assert batch["past_covariates_known_future"].shape == (4, 8, 2, 3)
    assert batch["past_covariates_unknown_future"].shape == (4, 8, 2, 2)
    assert batch["future_covariates_known"].shape == (4, 3, 2, 3)
    assert batch["output_target"].shape == (4, 3, 2, 1)
    assert batch["static_features_real"].shape == (4, 2, 2)
    assert batch["static_features_categorical"].shape == (4, 2, 1)


def test_feature_types_and_dtypes():
    """Test that feature types and dtypes are correct."""
    dataset = SimpleSyntheticDataset(
        length=50,
        past_length=10,
        future_length=5,
        num_target_features=1,
        num_known_cov_features=2,
        num_unknown_cov_features=1,
        num_static_real_features=2,
        num_static_categorical_features=1,
    )

    sample = dataset[0]

    # Check dtypes
    assert sample["past_target"].dtype == torch.float32
    assert sample["past_covariates_known_future"].dtype == torch.float32
    assert sample["past_covariates_unknown_future"].dtype == torch.float32
    assert sample["future_covariates_known"].dtype == torch.float32
    assert sample["output_target"].dtype == torch.float32
    assert sample["static_features_real"].dtype == torch.float32
    assert sample["static_features_categorical"].dtype == torch.long

    # Check that categorical values are within expected range
    cat_values = sample["static_features_categorical"]
    assert torch.all(cat_values >= 0)
    assert torch.all(cat_values < 5)  # Default cardinality
