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
        past_inputs, future_inputs, static_inputs, target = self.synthetic_dataset[0]

        # Check keys exist
        self.assertIn("past_data", past_inputs)
        self.assertIn("future_data", future_inputs)
        self.assertIn("static_data", static_inputs)

        # Check shapes
        self.assertEqual(
            past_inputs["past_data"].shape,
            (self.time_steps, self.series_dim, self.features_dim),
        )
        self.assertEqual(
            future_inputs["future_data"].shape,
            (self.future_steps, self.series_dim, self.features_dim),
        )
        self.assertIsNone(static_inputs["static_data"])
        self.assertEqual(
            target.shape, (self.future_steps, self.series_dim, self.features_dim)
        )

        # Get an item from the diamond dataset
        past_inputs, future_inputs, static_inputs, target = self.diamond_dataset[0]

        # Check keys exist
        self.assertIn("past_data", past_inputs)
        self.assertIn("future_data", future_inputs)
        self.assertIn("static_data", static_inputs)

        # Check shapes
        self.assertEqual(
            past_inputs["past_data"].shape,
            (self.time_steps, self.diamond_dataset.series_dim, 1),
        )
        self.assertEqual(
            future_inputs["future_data"].shape,
            (self.future_steps, self.diamond_dataset.series_dim, 1),
        )
        self.assertIsNone(static_inputs["static_data"])
        self.assertEqual(
            target.shape, (self.future_steps, self.diamond_dataset.series_dim, 1)
        )

    def test_types(self):
        """Test the types of the data returned by getitem."""
        # Get an item from the synthetic dataset
        past_inputs, future_inputs, static_inputs, target = self.synthetic_dataset[0]

        # Check types
        self.assertIsInstance(past_inputs["past_data"], torch.Tensor)
        self.assertIsInstance(future_inputs["future_data"], torch.Tensor)
        self.assertIsNone(static_inputs["static_data"])
        self.assertIsInstance(target, torch.Tensor)

        # Check dtypes
        self.assertEqual(past_inputs["past_data"].dtype, torch.float32)
        self.assertEqual(future_inputs["future_data"].dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)

        # Get an item from the diamond dataset
        past_inputs, future_inputs, static_inputs, target = self.diamond_dataset[0]

        # Check types
        self.assertIsInstance(past_inputs["past_data"], torch.Tensor)
        self.assertIsInstance(future_inputs["future_data"], torch.Tensor)
        self.assertIsNone(static_inputs["static_data"])
        self.assertIsInstance(target, torch.Tensor)

        # Check dtypes
        self.assertEqual(past_inputs["past_data"].dtype, torch.float32)
        self.assertEqual(future_inputs["future_data"].dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)


if __name__ == "__main__":
    # Allow running the test directly with python tests/datasets/test_dataset_format.py
    print("Running dataset format tests...")
    unittest.main()
    print("Tests complete.")
