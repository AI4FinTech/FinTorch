import os
import shutil
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

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_setup_directories(self):
        """Test that directories are set up correctly."""
        dataset = MarketDataset(root=self.test_dir, split=self.split)
        dataset.setup_directories()
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "raw")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "processed")))

    @pytest.mark.special
    def test_raw_paths(self):
        """Test that raw_paths returns the correct file paths."""
        dataset = MarketDataset(root=self.test_dir)
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
        self.assertEqual(dataset.raw_paths(), expected_paths)

    @pytest.mark.special
    def test_process(self):
        """Test the process method."""
        dataset = MarketDataset(root=self.test_dir, split=self.split)
        dataset.process()
        # Check that processed files are created
        processed_dir = os.path.join(self.test_dir, "processed", self.split)
        self.assertTrue(os.path.exists(processed_dir))
        files = os.listdir(processed_dir)
        self.assertTrue(len(files) > 0)

    @pytest.mark.special
    def test_load(self):
        """Test that data is loaded correctly."""
        dataset = MarketDataset(root=self.test_dir, split=self.split)
        # Ensure that data is processed
        dataset.process()
        dataset.load()
        self.assertTrue(hasattr(dataset, "data"))
        # Collect the data to check its content
        collected_data = dataset.data.collect()
        self.assertFalse(collected_data.is_empty())
        self.assertIn("unique_id", collected_data.columns)
        self.assertIn("ds", collected_data.columns)

    @pytest.mark.special
    def test_iter(self):
        """Test that the dataset can be iterated over correctly."""
        batch_size = 1
        dataset = MarketDataset(
            root=self.test_dir, batch_size=batch_size, split=self.split
        )
        # Ensure that data is processed and loaded
        dataset.process()
        dataset.load()
        iterator = iter(dataset)
        idx, batch_df = next(iterator)
        self.assertEqual(idx, 0)
        self.assertEqual(batch_df.shape, (batch_size, len(batch_df.columns)))
        # Check that the iterator stops after expected number of batches
        total_batches = (len(dataset.data.collect()) + batch_size - 1) // batch_size
        for _ in range(1, total_batches):
            next(iterator)
        with self.assertRaises(StopIteration):
            next(iterator)

    @pytest.mark.special
    def test_preprocess_batch(self):
        """Test the preprocess_batch method."""
        dataset = MarketDataset(root=self.test_dir, split=self.split)
        batch = pol.DataFrame(
            {
                "date_id": [0],
                "time_id": [0],
                "symbol_id": [101],
                "responder_6": [0.5],
                "feature_1": [1.0],
                "feature_2": [3.0],
            }
        )
        preprocessed = dataset.preprocess_batch(batch)
        self.assertIn("ds", preprocessed.columns)
        self.assertIn("unique_id", preprocessed.columns)
        self.assertIn("y", preprocessed.columns)
        self.assertEqual(preprocessed["unique_id"][0], 101)
        self.assertEqual(preprocessed["y"][0], 0.5)

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
        dataset = MarketDataset(root=self.test_dir)
        # Temporarily rename the Kaggle config file if it exists
        kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kaggle_config):
            temp_config = kaggle_config + ".temp"
            os.rename(kaggle_config, temp_config)
        else:
            temp_config = None
        try:
            with self.assertRaises(Exception):
                dataset.download()
        finally:
            # Restore the Kaggle config file
            if temp_config:
                os.rename(temp_config, kaggle_config)
