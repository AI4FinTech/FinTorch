import os
import tempfile
import unittest
from datetime import datetime

import polars as pol
import pytest

from fintorch.datasets.marketdata import MarketDataset


class TestMarketDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.split = "train"

        # Mock raw data for testing
        self.raw_dir = os.path.join(self.test_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        raw_data = pol.DataFrame(
            {
                "date_id": [0, 1],
                "time_id": [0, 1],
                "symbol_id": [101, 102],
                "responder_6": [0.5, 0.6],
                "feature_1": [1.0, 2.0],
                "feature_2": [3.0, 4.0],
            }
        )
        raw_data.write_parquet(os.path.join(self.raw_dir, f"{self.split}.parquet"))

    @pytest.mark.special
    def test_setup_directories(self):
        """Test that directories are set up correctly."""
        dataset = MarketDataset(root=self.test_dir, split=self.split)

        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "raw")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "processed")))

        self.assertTrue(hasattr(dataset, "data"))
        # Collect the data to check its content
        collected_data = dataset.data.collect()
        self.assertFalse(collected_data.is_empty())
        self.assertIn("unique_id", collected_data.columns)
        self.assertIn("ds", collected_data.columns)

        expected_paths = [
            os.path.join(self.test_dir, "raw", filename)
            for filename in [
                "features.csv",
                "responders.csv",
                "sample_submission.csv",
                "train.parquet",
                "test.parquet",
                "lags.parquet",
            ]
        ]

        # Check that raw_paths returns the correct file paths
        self.assertEqual(dataset.raw_paths(), expected_paths)

        # Check that each file exists
        for path in expected_paths:
            self.assertTrue(
                os.path.isfile(path), f"Expected file does not exist: {path}"
            )

        # Check that the train folder exists in the processed folder
        processed_train_folder = os.path.join(self.test_dir, "processed", "train")
        self.assertTrue(
            os.path.exists(processed_train_folder)
            and os.path.isdir(processed_train_folder),
            f"Processed train folder does not exist or is not a directory: {processed_train_folder}",
        )

    def test_map_to_datetime(self):
        """Test the map_to_datetime method."""
        batch = pol.DataFrame(
            {
                "date_id": [0],
                "time_id": [0],
            }
        )
        result = MarketDataset.map_to_datetime(batch)
        self.assertIn("ds", result.columns)
        expected_datetime = datetime(2023, 1, 1, 12, 0, 0)
        self.assertEqual(result["ds"][0], expected_datetime)

    @pytest.mark.special
    def test_setup_directories_failure(self):
        """Test that setup_directories raises an exception if it fails."""
        with unittest.mock.patch(
            "os.makedirs", side_effect=OSError("Permission denied")
        ):
            with self.assertRaises(RuntimeError):
                dataset = MarketDataset(root=self.test_dir)
                dataset.setup_directories()

    def test_download_without_credentials(self):
        """Test the download method handles missing Kaggle credentials."""
        # Temporarily rename the Kaggle config file if it exists
        kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kaggle_config):
            temp_config = kaggle_config + ".temp"
            os.rename(kaggle_config, temp_config)
        else:
            temp_config = None
        try:
            with self.assertRaises(Exception):
                MarketDataset(root=self.test_dir)
        finally:
            # Restore the Kaggle config file
            if temp_config:
                os.rename(temp_config, kaggle_config)
