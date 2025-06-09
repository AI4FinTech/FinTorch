"""
Test Dataset Format Compliance

This test module validates that all FinTorch datasets conform to the standardized format
defined in the TimeSeriesDataset base class. The tests have been updated to match the
new data structure specification which includes:

MAJOR UPDATES:
1. Standardized dictionary format with 7 required keys in __getitem__
2. New granular property methods for feature counting
3. Covariate separation into known/unknown future categories
4. Static features split into real/categorical with cardinalities
5. Consistent tensor shapes and data types across all datasets

TESTED DATASETS:
- SimpleSyntheticDataset: Synthetic time series with configurable components
- CausalDataset: Real causal relationship data (diamond, fork, mediator, v types)

VALIDATION AREAS:
- Dictionary format compliance (all required keys present)
- Tensor shape validation (time_steps, series_dim, feature_dims)
- Data type consistency (float32 for features, long for categorical)
- Property method accuracy (feature counts, dimensions, cardinalities)
- Cross-sample consistency for batching compatibility
- Feature dimension arithmetic consistency

The tests ensure models can reliably depend on the standardized format for
architecture initialization and data processing pipelines.
"""

import unittest
import os
import sys

# Find the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: E402
from fintorch.datasets.synthetic.simpleSynthetic import SimpleSyntheticDataset  # noqa: E402
from fintorch.datasets.causal_data import CausalDataset  # noqa: E402
from fintorch.datasets.base.base_dataset import TimeSeriesDataset  # noqa: E402


class TestDatasetFormat(unittest.TestCase):
    """
    Test the uniform format across different dataset implementations.

    This test suite validates that all datasets in FinTorch conform to the
    standardized format defined in TimeSeriesDataset base class. It ensures:

    1. All required dictionary keys are present in __getitem__ output
    2. Tensor shapes match the expected format specification
    3. Data types are correct (float32 for features, long for categorical)
    4. Properties return consistent values
    5. New standardized properties are implemented correctly
    """

    def setUp(self):
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
                num_series=self.series_dim,
                num_target_features=self.features_dim,
                num_static_real_features=self.static_length,
                num_static_categorical_features=0,
            )

            # Create the causal dataset with diamond type
            self.diamond_dataset = CausalDataset(
                dataset_type='diamond',
                time_step=self.time_steps,
                output_window=self.future_steps,
                static_length=self.static_length,
            )
        except Exception as e:
            self.fail("Failed to create datasets: {}".format(str(e)))

    def tearDown(self):
        pass

    def test_dataset_is_instance_of_base(self):
        """Test that the datasets are instances of TimeSeriesDataset."""
        self.assertIsInstance(self.synthetic_dataset, TimeSeriesDataset)
        self.assertIsInstance(self.diamond_dataset, TimeSeriesDataset)

    def test_basic_properties(self):
        """Test the basic properties of the datasets."""
        # Synthetic dataset properties
        self.assertEqual(self.synthetic_dataset.time_steps, self.time_steps)
        self.assertEqual(self.synthetic_dataset.future_steps, self.future_steps)
        self.assertEqual(self.synthetic_dataset.static_length, self.static_length)
        self.assertEqual(self.synthetic_dataset.series_dim, self.series_dim)
        # features_dim = num_target_features + num_known_cov + num_unknown_cov = 1 + 2 + 1 = 4
        self.assertEqual(self.synthetic_dataset.features_dim, 4)

        # Diamond dataset properties
        self.assertEqual(self.diamond_dataset.time_steps, self.time_steps)
        self.assertEqual(self.diamond_dataset.future_steps, self.future_steps)
        self.assertEqual(self.diamond_dataset.static_length, self.static_length)
        # Diamond dataset has its own series dimension (determined by the actual data)
        self.assertGreater(self.diamond_dataset.series_dim, 0)
        self.assertEqual(self.diamond_dataset.features_dim, 1)

    def test_new_dataset_properties(self):
        """
        Test the new properties added to the base dataset class.

        Validates that all datasets implement the new standardized properties:
        - num_target_features: Number of target features per series
        - num_known_future_cov_features: Covariates with known future values
        - num_unknown_future_cov_features: Covariates without known future values
        - num_static_real_features: Real-valued static features count
        - num_static_categorical_features: Categorical static features count
        - static_categorical_cardinalities: List of category counts per categorical feature
        """
        # Test synthetic dataset
        self.assertIsInstance(self.synthetic_dataset.num_target_features, int)
        self.assertIsInstance(self.synthetic_dataset.num_known_future_cov_features, int)
        self.assertIsInstance(self.synthetic_dataset.num_unknown_future_cov_features, int)
        self.assertIsInstance(self.synthetic_dataset.num_static_real_features, int)
        self.assertIsInstance(self.synthetic_dataset.num_static_categorical_features, int)
        self.assertIsInstance(self.synthetic_dataset.static_categorical_cardinalities, list)

        # Test diamond dataset
        self.assertIsInstance(self.diamond_dataset.num_target_features, int)
        self.assertIsInstance(self.diamond_dataset.num_known_future_cov_features, int)
        self.assertIsInstance(self.diamond_dataset.num_unknown_future_cov_features, int)
        self.assertIsInstance(self.diamond_dataset.num_static_real_features, int)
        self.assertIsInstance(self.diamond_dataset.num_static_categorical_features, int)
        self.assertIsInstance(self.diamond_dataset.static_categorical_cardinalities, list)

        # Test specific values for synthetic dataset
        self.assertEqual(self.synthetic_dataset.num_target_features, 1)
        self.assertEqual(self.synthetic_dataset.num_known_future_cov_features, 2)
        self.assertEqual(self.synthetic_dataset.num_unknown_future_cov_features, 1)
        self.assertEqual(self.synthetic_dataset.num_static_real_features, 2)
        self.assertEqual(self.synthetic_dataset.num_static_categorical_features, 0)

        # Test specific values for diamond dataset
        self.assertEqual(self.diamond_dataset.num_target_features, 1)
        self.assertEqual(self.diamond_dataset.num_known_future_cov_features, 0)
        self.assertEqual(self.diamond_dataset.num_unknown_future_cov_features, 0)
        self.assertEqual(self.diamond_dataset.num_static_real_features, 2)
        self.assertEqual(self.diamond_dataset.num_static_categorical_features, 0)  # Diamond has no categorical features

    def test_getitem_format(self):
        """
        Test the format of the getitem method returns standardized dictionary.

        Validates that __getitem__ returns a dictionary with all required keys:
        - past_target: Historical target values
        - past_covariates_known_future: Historical covariates with known future
        - past_covariates_unknown_future: Historical covariates without known future
        - future_covariates_known: Future covariate values (known)
        - output_target: Target values to predict (labels)
        - static_features_real: Real-valued static features
        - static_features_categorical: Categorical static features
        """
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
            self.assertIn(key, synthetic_item, "Missing required key: {}".format(key))

        # Check shapes for synthetic dataset
        self.assertEqual(
            synthetic_item["past_target"].shape,
            (self.time_steps, self.series_dim, self.synthetic_dataset.num_target_features),
            "Incorrect past_target shape"
        )
        self.assertEqual(
            synthetic_item["output_target"].shape,
            (self.future_steps, self.series_dim, self.synthetic_dataset.num_target_features),
            "Incorrect output_target shape"
        )
        self.assertEqual(
            synthetic_item["static_features_real"].shape,
            (self.series_dim, self.synthetic_dataset.num_static_real_features),
            "Incorrect static_features_real shape"
        )
        self.assertEqual(
            synthetic_item["static_features_categorical"].shape,
            (self.series_dim, self.synthetic_dataset.num_static_categorical_features),
            "Incorrect static_features_categorical shape"
        )

        # Get an item from the diamond dataset
        diamond_item = self.diamond_dataset[0]

        # Check required keys exist
        for key in required_keys:
            self.assertIn(key, diamond_item, "Missing required key: {}".format(key))

        # Check shapes for diamond dataset
        self.assertEqual(
            diamond_item["past_target"].shape,
            (self.time_steps, self.diamond_dataset.series_dim, self.diamond_dataset.num_target_features),
            "Incorrect diamond past_target shape"
        )
        self.assertEqual(
            diamond_item["output_target"].shape,
            (self.future_steps, self.diamond_dataset.series_dim, self.diamond_dataset.num_target_features),
            "Incorrect diamond output_target shape"
        )
        self.assertEqual(
            diamond_item["static_features_real"].shape,
            (self.diamond_dataset.series_dim, self.diamond_dataset.num_static_real_features),
            "Incorrect diamond static_features_real shape"
        )
        self.assertEqual(
            diamond_item["static_features_categorical"].shape,
            (self.diamond_dataset.series_dim, self.diamond_dataset.num_static_categorical_features),
            "Incorrect diamond static_features_categorical shape"
        )

    def test_covariate_shapes(self):
        """
        Test the shapes of covariate tensors.

        Validates that covariate tensors have the correct shapes according to
        the standardized format:
        - past_covariates_*: (time_steps, series_dim, num_cov_features)
        - future_covariates_known: (future_steps, series_dim, num_known_cov_features)
        """
        # Test synthetic dataset
        synthetic_item = self.synthetic_dataset[0]

        self.assertEqual(
            synthetic_item["past_covariates_known_future"].shape,
            (self.time_steps, self.series_dim, self.synthetic_dataset.num_known_future_cov_features),
            "Incorrect past_covariates_known_future shape for synthetic"
        )
        self.assertEqual(
            synthetic_item["past_covariates_unknown_future"].shape,
            (self.time_steps, self.series_dim, self.synthetic_dataset.num_unknown_future_cov_features),
            "Incorrect past_covariates_unknown_future shape for synthetic"
        )
        self.assertEqual(
            synthetic_item["future_covariates_known"].shape,
            (self.future_steps, self.series_dim, self.synthetic_dataset.num_known_future_cov_features),
            "Incorrect future_covariates_known shape for synthetic"
        )

        # Test diamond dataset
        diamond_item = self.diamond_dataset[0]

        self.assertEqual(
            diamond_item["past_covariates_known_future"].shape,
            (self.time_steps, self.diamond_dataset.series_dim, self.diamond_dataset.num_known_future_cov_features),
            "Incorrect past_covariates_known_future shape for diamond"
        )
        self.assertEqual(
            diamond_item["past_covariates_unknown_future"].shape,
            (self.time_steps, self.diamond_dataset.series_dim, self.diamond_dataset.num_unknown_future_cov_features),
            "Incorrect past_covariates_unknown_future shape for diamond"
        )
        self.assertEqual(
            diamond_item["future_covariates_known"].shape,
            (self.future_steps, self.diamond_dataset.series_dim, self.diamond_dataset.num_known_future_cov_features),
            "Incorrect future_covariates_known shape for diamond"
        )

    def test_tensor_types(self):
        """Test the types of the tensors returned by getitem."""
        # Get an item from the synthetic dataset
        synthetic_item = self.synthetic_dataset[0]

        # Check types for main tensors
        self.assertIsInstance(synthetic_item["past_target"], torch.Tensor)
        self.assertIsInstance(synthetic_item["output_target"], torch.Tensor)
        self.assertIsInstance(synthetic_item["static_features_real"], torch.Tensor)
        self.assertIsInstance(synthetic_item["static_features_categorical"], torch.Tensor)
        self.assertIsInstance(synthetic_item["past_covariates_known_future"], torch.Tensor)
        self.assertIsInstance(synthetic_item["past_covariates_unknown_future"], torch.Tensor)
        self.assertIsInstance(synthetic_item["future_covariates_known"], torch.Tensor)

        # Check dtypes
        self.assertEqual(synthetic_item["past_target"].dtype, torch.float32)
        self.assertEqual(synthetic_item["output_target"].dtype, torch.float32)
        self.assertEqual(synthetic_item["static_features_real"].dtype, torch.float32)
        self.assertEqual(synthetic_item["static_features_categorical"].dtype, torch.long)
        self.assertEqual(synthetic_item["past_covariates_known_future"].dtype, torch.float32)
        self.assertEqual(synthetic_item["past_covariates_unknown_future"].dtype, torch.float32)
        self.assertEqual(synthetic_item["future_covariates_known"].dtype, torch.float32)

        # Get an item from the diamond dataset
        diamond_item = self.diamond_dataset[0]

        # Check types for main tensors
        self.assertIsInstance(diamond_item["past_target"], torch.Tensor)
        self.assertIsInstance(diamond_item["output_target"], torch.Tensor)
        self.assertIsInstance(diamond_item["static_features_real"], torch.Tensor)
        self.assertIsInstance(diamond_item["static_features_categorical"], torch.Tensor)
        self.assertIsInstance(diamond_item["past_covariates_known_future"], torch.Tensor)
        self.assertIsInstance(diamond_item["past_covariates_unknown_future"], torch.Tensor)
        self.assertIsInstance(diamond_item["future_covariates_known"], torch.Tensor)

        # Check dtypes
        self.assertEqual(diamond_item["past_target"].dtype, torch.float32)
        self.assertEqual(diamond_item["output_target"].dtype, torch.float32)
        self.assertEqual(diamond_item["static_features_real"].dtype, torch.float32)
        self.assertEqual(diamond_item["static_features_categorical"].dtype, torch.long)
        self.assertEqual(diamond_item["past_covariates_known_future"].dtype, torch.float32)
        self.assertEqual(diamond_item["past_covariates_unknown_future"].dtype, torch.float32)
        self.assertEqual(diamond_item["future_covariates_known"].dtype, torch.float32)

    def test_dataset_length(self):
        """Test that datasets have valid lengths."""
        self.assertGreater(len(self.synthetic_dataset), 0)
        self.assertGreater(len(self.diamond_dataset), 0)

        # Test that we can access all indices
        last_idx = len(self.synthetic_dataset) - 1
        try:
            _ = self.synthetic_dataset[last_idx]
        except IndexError:
            self.fail("Cannot access last index of synthetic dataset")

        last_idx = len(self.diamond_dataset) - 1
        try:
            _ = self.diamond_dataset[last_idx]
        except IndexError:
            self.fail("Cannot access last index of diamond dataset")

    def test_consistency_across_samples(self):
        """
        Test that shapes and types are consistent across different samples.

        Ensures that all samples from the same dataset have identical tensor
        shapes and data types, which is crucial for batching in DataLoaders.
        """
        if len(self.synthetic_dataset) < 2:
            self.skipTest("Not enough samples in synthetic dataset for consistency test")

        sample1 = self.synthetic_dataset[0]
        sample2 = self.synthetic_dataset[1]

        # Check that shapes are consistent
        for key in sample1.keys():
            self.assertEqual(
                sample1[key].shape, sample2[key].shape,
                "Inconsistent shape for key {} between samples".format(key)
            )
            self.assertEqual(
                sample1[key].dtype, sample2[key].dtype,
                "Inconsistent dtype for key {} between samples".format(key)
            )

        # Same test for diamond dataset
        if len(self.diamond_dataset) < 2:
            self.skipTest("Not enough samples in diamond dataset for consistency test")

        sample1 = self.diamond_dataset[0]
        sample2 = self.diamond_dataset[1]

        for key in sample1.keys():
            self.assertEqual(
                sample1[key].shape, sample2[key].shape,
                "Inconsistent shape for key {} between diamond samples".format(key)
            )
            self.assertEqual(
                sample1[key].dtype, sample2[key].dtype,
                "Inconsistent dtype for key {} between diamond samples".format(key)
            )

    def test_static_categorical_cardinalities_property(self):
        """
        Test that static_categorical_cardinalities returns valid values.

        Validates that the cardinalities list:
        - Is a list of integers
        - Has length equal to num_static_categorical_features
        - Contains positive values for each categorical feature

        This property is essential for setting up embedding layers in models.
        """
        # Test synthetic dataset
        cardinalities = self.synthetic_dataset.static_categorical_cardinalities
        self.assertIsInstance(cardinalities, list)
        self.assertEqual(
            len(cardinalities),
            self.synthetic_dataset.num_static_categorical_features,
            "Cardinalities list length should match number of categorical features"
        )

        # Test diamond dataset
        cardinalities = self.diamond_dataset.static_categorical_cardinalities
        self.assertIsInstance(cardinalities, list)
        self.assertEqual(
            len(cardinalities),
            self.diamond_dataset.num_static_categorical_features,
            "Diamond cardinalities list length should match number of categorical features"
        )

    def test_feature_dimension_consistency(self):
        """
        Test that feature dimensions are consistent with actual tensor shapes.

        Validates that the reported feature dimensions match the actual
        tensor shapes returned by __getitem__. This ensures that model
        initialization can rely on these properties for architecture setup.
        """
        # Test synthetic dataset
        total_static_synthetic = (self.synthetic_dataset.num_static_real_features +
                                 self.synthetic_dataset.num_static_categorical_features)
        self.assertEqual(total_static_synthetic, self.synthetic_dataset.static_length)

        # Test diamond dataset
        total_static_diamond = (self.diamond_dataset.num_static_real_features +
                               self.diamond_dataset.num_static_categorical_features)
        self.assertEqual(total_static_diamond, self.diamond_dataset.static_length)


if __name__ == "__main__":
    # Allow running the test directly with python tests/datasets/test_dataset_format.py
    print("Running dataset format tests...")
    unittest.main()
    print("Tests complete.")
