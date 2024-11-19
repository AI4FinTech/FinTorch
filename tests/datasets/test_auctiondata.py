import os
import shutil
import tempfile
import unittest

import polars as pol
import torch

from fintorch.datasets.auctiondata import AuctionDataset


class TestAuctionDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_setupDirectories(self):
        """Test that directories are set up correctly."""
        AuctionDataset(root=self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "raw")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "processed")))

    def test_processed_paths(self):
        """Test that processed_paths returns the correct file paths."""
        dataset = AuctionDataset(root=self.test_dir)
        expected_paths = [
            os.path.join(self.test_dir, "raw/train.csv"),
            os.path.join(self.test_dir, "raw/example_test_files/revealed_targets.csv"),
            os.path.join(self.test_dir, "raw/example_test_files/sample_submission.csv"),
            os.path.join(self.test_dir, "raw/example_test_files/test.csv"),
        ]
        self.assertEqual(dataset.processed_paths(), expected_paths)

    def test_load(self):
        dataset = AuctionDataset(root=self.test_dir)
        dataset.load()

        # Assertions for train DataFrame
        self.assertTrue(hasattr(dataset, "train"))
        self.assertIn("y", dataset.train.columns)
        self.assertIn("unique_id", dataset.train.columns)
        self.assertNotIn("row_id", dataset.train.columns)
        self.assertNotIn("near_price", dataset.train.columns)
        self.assertNotIn("far_price", dataset.train.columns)

        # Assertions for test DataFrame
        self.assertTrue(hasattr(dataset, "test"))
        self.assertIn("unique_id", dataset.test.columns)
        self.assertNotIn("near_price", dataset.test.columns)
        self.assertNotIn("far_price", dataset.test.columns)

    def test_len(self):
        """Test that __len__ returns the correct length."""
        dataset = AuctionDataset(root=self.test_dir)
        # Mock the train DataFrame
        dataset.train = pol.DataFrame({"col1": [1, 2, 3]})
        self.assertEqual(len(dataset), 3)

    def test_getitem(self):
        """Test that __getitem__ retrieves data correctly."""
        dataset = AuctionDataset(root=self.test_dir)
        # Mock the train DataFrame
        dataset.train = pol.DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "ds": [7, 8, 9]}
        )
        item = dataset[1]
        expected_tensor = torch.tensor([2, 5], dtype=torch.float32)
        self.assertTrue(torch.equal(item, expected_tensor))

    def test_process(self):
        """Test the process method (currently a placeholder)."""
        dataset = AuctionDataset(root=self.test_dir)
        try:
            dataset.process()
            self.assertTrue(True)  # Method executed without error
        except Exception as e:
            self.fail(f"Process method raised an exception: {e}")
