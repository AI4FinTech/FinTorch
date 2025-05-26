import unittest
import os
import tempfile
import sys

import torch
import numpy as np

# Find the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import the required modules directly
try:
    from fintorch.datasets.synthetic.simpleSynthetic import SimpleSyntheticDataset
    from fintorch.datasets.diamondata import DiamondDataset
    from fintorch.datasets.base.base_dataset import TimeSeriesDataset
except ImportError:
    # Alternative import path
    from fintorch.datasets.synthetic.simpleSynthetic import (
        SimpleSyntheticDataset,
    )
    from fintorch.datasets.diamondata import DiamondDataset
    from fintorch.datasets.base.base_dataset import TimeSeriesDataset


class TestDatasetFormat(unittest.TestCase):
    """Test the uniform format across different dataset implementations."""

    def setUp(self):
        # Create a temporary CSV file for DiamondDataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, "temp_data.csv")

        # Generate simple data for diamonddata
        # Using 3 series to match series_dim parameter
        data = np.random.randn(100, 3)
        np.savetxt(self.temp_file, data, delimiter=",")

        # Parameters for datasets
        self.time_steps = 10  # Number of past time steps
        self.future_steps = 5  # Number of future steps to predict
        self.static_length = 2  # Static feature dimension
        self.series_dim = 3  # Number of time series
        self.features_dim = 1  # Features per time series

        try:
            # Create the synthetic dataset
            self.synthetic_dataset = SimpleSyntheticDataset(
                length=100,  # Total length of time series
                past_length=self.time_steps,
                future_length=self.future_steps,
                static_length=self.static_length,
                num_series=self.series_dim,
                features_dim=self.features_dim,
            )

            # Create the diamond dataset
            self.diamond_dataset = DiamondDataset(
                local_path=self.temp_file,
                time_step=self.time_steps,
                output_window=self.future_steps,
                static_length=self.static_length,
            )
        except Exception as e:
            self.fail(f"Failed to create datasets: {str(e)}")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_is_instance_of_base(self):
        """Test that the datasets are instances of TimeSeriesDataset."""
        self.assertIsInstance(self.synthetic_dataset, TimeSeriesDataset)
        self.assertIsInstance(self.diamond_dataset, TimeSeriesDataset)

    def test_properties(self):
        """Test the properties of the datasets."""
        # Synthetic dataset properties
        self.assertEqual(self.synthetic_dataset.time_steps, self.time_steps)
        self.assertEqual(self.synthetic_dataset.future_steps, self.future_steps)
        self.assertEqual(self.synthetic_dataset.static_length, self.static_length)
        self.assertEqual(self.synthetic_dataset.series_dim, self.series_dim)
        self.assertEqual(self.synthetic_dataset.features_dim, self.features_dim)

        # Diamond dataset properties
        self.assertEqual(self.diamond_dataset.time_steps, self.time_steps)
        self.assertEqual(self.diamond_dataset.future_steps, self.future_steps)
        self.assertEqual(self.diamond_dataset.static_length, self.static_length)
        self.assertEqual(self.diamond_dataset.series_dim, self.series_dim)
        self.assertEqual(self.diamond_dataset.features_dim, 1)

    def test_getitem_format(self):
        """Test the format of the getitem method."""
        # Get an item from the synthetic dataset
        synthetic_item = self.synthetic_dataset[0]

        # Check required keys exist
        required_keys = [
            "past_target",
            "past_covariates_known_future",
            "past_covariates_unknown_future",
            "future_covariates_known",
            "output_target",
            "static_features_real",
            "static_features_categorical"
        ]
        for key in required_keys:
            self.assertIn(key, synthetic_item)

        # Check shapes for synthetic dataset
        self.assertEqual(
            synthetic_item["past_target"].shape,
            (self.time_steps, self.series_dim, self.features_dim),
        )
        self.assertEqual(
            synthetic_item["output_target"].shape,
            (self.future_steps, self.series_dim, self.features_dim),
        )

        # Get an item from the diamond dataset
        diamond_item = self.diamond_dataset[0]

        # Check required keys exist
        for key in required_keys:
            self.assertIn(key, diamond_item)

        # Check shapes for diamond dataset
        self.assertEqual(
            diamond_item["past_target"].shape,
            (self.time_steps, self.diamond_dataset.series_dim, 1),
        )
        self.assertEqual(
            diamond_item["output_target"].shape,
            (self.future_steps, self.diamond_dataset.series_dim, 1),
        )

    def test_types(self):
        """Test the types of the data returned by getitem."""
        # Get an item from the synthetic dataset
        synthetic_item = self.synthetic_dataset[0]

        # Check types for main tensors
        self.assertIsInstance(synthetic_item["past_target"], torch.Tensor)
        self.assertIsInstance(synthetic_item["output_target"], torch.Tensor)
        self.assertIsInstance(synthetic_item["static_features_real"], torch.Tensor)
        self.assertIsInstance(synthetic_item["static_features_categorical"], torch.Tensor)

        # Check dtypes
        self.assertEqual(synthetic_item["past_target"].dtype, torch.float32)
        self.assertEqual(synthetic_item["output_target"].dtype, torch.float32)
        self.assertEqual(synthetic_item["static_features_real"].dtype, torch.float32)
        self.assertEqual(synthetic_item["static_features_categorical"].dtype, torch.long)

        # Get an item from the diamond dataset
        diamond_item = self.diamond_dataset[0]

        # Check types for main tensors
        self.assertIsInstance(diamond_item["past_target"], torch.Tensor)
        self.assertIsInstance(diamond_item["output_target"], torch.Tensor)
        self.assertIsInstance(diamond_item["static_features_real"], torch.Tensor)
        self.assertIsInstance(diamond_item["static_features_categorical"], torch.Tensor)

        # Check dtypes
        self.assertEqual(diamond_item["past_target"].dtype, torch.float32)
        self.assertEqual(diamond_item["output_target"].dtype, torch.float32)
        self.assertEqual(diamond_item["static_features_real"].dtype, torch.float32)
        self.assertEqual(diamond_item["static_features_categorical"].dtype, torch.long)


if __name__ == "__main__":
    # Allow running the test directly with python tests/datasets/test_dataset_format.py
    print("Running dataset format tests...")
    unittest.main()
    print("Tests complete.")
