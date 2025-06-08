#!/usr/bin/env python3
"""
Comprehensive unit tests for causal data module classes and functions.

This test suite provides enhanced test coverage for CausalDataset, CausalDataModule,
and all utility functions, including edge cases, error handling, and property testing.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import torch
import polars as pl
import requests
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from fintorch.datasets.causal_data import (
    CausalDataset, 
    CausalDataModule, 
    create_causal_datamodule,
    custom_collate_fn,
    _download_causal_data,
    get_causal_data_dir,
    clear_causal_data,
    list_available_datasets,
    get_dataset_info,
    get_clean_adjacency_matrix
)


class TestCausalDataset:
    """Test suite for CausalDataset class."""

    @pytest.fixture
    def mock_data_dir(self):
        """Create a temporary directory with mock data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, 'diamond')
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Create mock data files
            for i in range(3):
                data_file = os.path.join(dataset_dir, f'data_{i}.csv')
                with open(data_file, 'w') as f:
                    f.write('node_0,node_1,node_2\n')
                    for j in range(100):
                        f.write(f'{np.random.rand():.4f},{np.random.rand():.4f},{np.random.rand():.4f}\n')
            
            # Create groundtruth file
            groundtruth_file = os.path.join(dataset_dir, 'groundtruth.csv')
            with open(groundtruth_file, 'w') as f:
                f.write('node_0,node_1,node_2\n')
                f.write('0,1,0\n')
                f.write('1,0,1\n')
                f.write('0,1,0\n')
            
            yield temp_dir

    def test_initialization_with_valid_params(self, mock_data_dir):
        """Test CausalDataset initialization with valid parameters."""
        with patch('fintorch.datasets.causal_data._download_causal_data'):
            dataset = CausalDataset(
                dataset_type='diamond',
                local_dir=mock_data_dir,
                time_step=10,
                output_window=5,
                static_length=3
            )
            
            assert dataset.dataset_type == 'diamond'
            assert dataset._time_step == 10
            assert dataset._output_window == 5
            assert dataset._static_length == 3
            assert len(dataset) > 0

    def test_initialization_with_default_params(self):
        """Test CausalDataset initialization with default parameters."""
        with patch('fintorch.datasets.causal_data._download_causal_data'), \
             patch.object(CausalDataset, '_load_data', return_value=(torch.randn(100, 5), None)), \
             patch.object(CausalDataset, '_generate_indices', return_value=list(range(10, 90))):
            
            dataset = CausalDataset(dataset_type='diamond')
            
            assert dataset.dataset_type == 'diamond'
            assert dataset._time_step == 10
            assert dataset._output_window == 5
            assert dataset._static_length == 2

    def test_load_data_with_valid_files(self, mock_data_dir):
        """Test _load_data method with valid data files."""
        with patch('fintorch.datasets.causal_data._download_causal_data'):
            dataset = CausalDataset(
                dataset_type='diamond',
                local_dir=mock_data_dir,
                time_step=10,
                output_window=5
            )
            
            assert dataset.data is not None
            assert dataset.groundtruth is not None
            assert dataset.data.shape[1] == 3  # 3 nodes
            assert dataset.groundtruth.shape == (3, 3)

    def test_load_data_no_files_found(self, mock_data_dir):
        """Test _load_data method when no data files are found."""
        # Remove data files but keep directory
        dataset_dir = os.path.join(mock_data_dir, 'diamond')
        for file in os.listdir(dataset_dir):
            if file.startswith('data_'):
                os.remove(os.path.join(dataset_dir, file))
        
        with patch('fintorch.datasets.causal_data._download_causal_data'):
            with pytest.raises(FileNotFoundError, match="No data files found"):
                CausalDataset(
                    dataset_type='diamond',
                    local_dir=mock_data_dir,
                    time_step=10,
                    output_window=5
                )

    def test_generate_indices_valid_range(self, mock_data_dir):
        """Test _generate_indices method with valid time step and output window."""
        with patch('fintorch.datasets.causal_data._download_causal_data'):
            dataset = CausalDataset(
                dataset_type='diamond',
                local_dir=mock_data_dir,
                time_step=10,
                output_window=5
            )
            
            indices = dataset._generate_indices()
            assert len(indices) > 0
            assert all(i >= 10 for i in indices)
            assert all(i + 5 <= len(dataset.data) for i in indices)

    def test_generate_indices_invalid_range(self):
        """Test _generate_indices method with invalid time step and output window."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, 'diamond')
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Create a very small data file (only 10 rows)
            data_file = os.path.join(dataset_dir, 'data_0.csv')
            with open(data_file, 'w') as f:
                f.write('node_0,node_1\n')
                for j in range(10):  # Only 10 data points
                    f.write(f'{j},{j+1}\n')
            
            with patch('fintorch.datasets.causal_data._download_causal_data'):
                # The dataset initialization itself should raise the error since _generate_indices is called in __init__
                with pytest.raises(ValueError, match="No valid indices found"):
                    CausalDataset(
                        dataset_type='diamond',
                        local_dir=temp_dir,
                        time_step=15,  # Too large for 10 data points
                        output_window=5
                    )

    def test_getitem_valid_index(self, mock_data_dir):
        """Test __getitem__ method with valid index."""
        with patch('fintorch.datasets.causal_data._download_causal_data'):
            dataset = CausalDataset(
                dataset_type='diamond',
                local_dir=mock_data_dir,
                time_step=10,
                output_window=5
            )
            
            sample = dataset[0]
            
            assert isinstance(sample, dict)
            assert 'past_target' in sample
            assert 'output_target' in sample
            assert 'static_features_real' in sample
            
            assert sample['past_target'].shape == (10, 3, 1)
            assert sample['output_target'].shape == (5, 3, 1)
            assert sample['static_features_real'].shape == (3, 2)  # (series_num, static_length)

    def test_getitem_invalid_index(self, mock_data_dir):
        """Test __getitem__ method with invalid index."""
        with patch('fintorch.datasets.causal_data._download_causal_data'):
            dataset = CausalDataset(
                dataset_type='diamond',
                local_dir=mock_data_dir,
                time_step=10,
                output_window=5
            )
            
            with pytest.raises(IndexError):
                dataset[len(dataset)]

    def test_properties(self, mock_data_dir):
        """Test all property methods of CausalDataset."""
        with patch('fintorch.datasets.causal_data._download_causal_data'):
            dataset = CausalDataset(
                dataset_type='diamond',
                local_dir=mock_data_dir,
                time_step=15,
                output_window=7,
                static_length=4
            )
            
            assert dataset.time_steps == 15
            assert dataset.future_steps == 7
            assert dataset.series_dim == 3
            assert dataset.features_dim == 1
            assert dataset.static_length == 4
            assert dataset.static_categorical_cardinalities == []
            assert dataset.num_target_features == 3
            assert dataset.num_known_future_cov_features == 0
            assert dataset.num_unknown_future_cov_features == 0
            assert dataset.num_static_real_features == 4
            assert dataset.num_static_categorical_features == 0

    def test_get_groundtruth(self, mock_data_dir):
        """Test get_groundtruth method."""
        with patch('fintorch.datasets.causal_data._download_causal_data'):
            dataset = CausalDataset(
                dataset_type='diamond',
                local_dir=mock_data_dir,
                time_step=10,
                output_window=5
            )
            
            groundtruth = dataset.get_groundtruth()
            assert groundtruth is not None
            assert isinstance(groundtruth, pl.DataFrame)
            assert groundtruth.shape == (3, 3)


class TestCausalDataModule:
    """Test suite for CausalDataModule class."""

    def test_initialization_valid_params(self):
        """Test CausalDataModule initialization with valid parameters."""
        datamodule = CausalDataModule(
            dataset_type='diamond',
            time_step=20,
            output_window=10,
            batch_size=64,
            num_workers=4,
            train_split=0.8,
            val_split=0.1
        )
        
        assert datamodule.dataset_type == 'diamond'
        assert datamodule.time_step == 20
        assert datamodule.output_window == 10
        assert datamodule.batch_size == 64
        assert datamodule.num_workers == 4
        assert datamodule.train_split == 0.8
        assert datamodule.val_split == 0.1
        assert abs(datamodule.test_split - 0.1) < 1e-10  # Handle floating point precision

    def test_initialization_invalid_dataset_type(self):
        """Test CausalDataModule initialization with invalid dataset type."""
        with pytest.raises(ValueError, match="dataset_type must be one of"):
            CausalDataModule(dataset_type='invalid_type')

    def test_initialization_invalid_splits(self):
        """Test CausalDataModule initialization with invalid split ratios."""
        with pytest.raises(ValueError, match="Train, validation, and test splits"):
            CausalDataModule(
                dataset_type='diamond',
                train_split=0.9,
                val_split=0.2  # Sum > 1
            )

    @patch.object(CausalDataset, '__init__', return_value=None)
    @patch.object(CausalDataset, '__len__', return_value=100)
    def test_setup(self, mock_len, mock_init):
        """Test setup method."""
        datamodule = CausalDataModule(
            dataset_type='diamond',
            train_split=0.7,
            val_split=0.2
        )
        
        datamodule.setup()
        
        assert datamodule.dataset is not None
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert datamodule.test_dataset is not None
        
        # Check dataset sizes
        assert len(datamodule.train_dataset) == 70
        assert len(datamodule.val_dataset) == 20
        assert len(datamodule.test_dataset) == 10

    def test_dataloaders(self):
        """Test dataloader methods."""
        with patch.object(CausalDataModule, 'setup'):
            datamodule = CausalDataModule(dataset_type='diamond', batch_size=32)
            
            # Mock datasets with proper length
            datamodule.train_dataset = MagicMock()
            datamodule.val_dataset = MagicMock()
            datamodule.test_dataset = MagicMock()
            
            # Mock __len__ method to return positive integers
            datamodule.train_dataset.__len__.return_value = 100
            datamodule.val_dataset.__len__.return_value = 50
            datamodule.test_dataset.__len__.return_value = 25
            
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            test_loader = datamodule.test_dataloader()
            predict_loader = datamodule.predict_dataloader()
            
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None
            assert predict_loader is not None

    def test_get_groundtruth_with_dataset(self):
        """Test get_groundtruth method when dataset is available."""
        datamodule = CausalDataModule(dataset_type='diamond')
        
        # Mock dataset with groundtruth
        mock_groundtruth = pl.DataFrame({'node_0': [0, 1], 'node_1': [1, 0]})
        datamodule.dataset = MagicMock()
        datamodule.dataset.get_groundtruth.return_value = mock_groundtruth
        
        result = datamodule.get_groundtruth()
        assert result is not None
        assert result.equals(mock_groundtruth)

    def test_get_groundtruth_without_dataset(self):
        """Test get_groundtruth method when dataset is not available."""
        datamodule = CausalDataModule(dataset_type='diamond')
        
        result = datamodule.get_groundtruth()
        assert result is None


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_custom_collate_fn_same_size_tensors(self):
        """Test custom_collate_fn with same-sized tensors."""
        batch = [
            {
                'past_data': torch.rand(10, 3, 1),
                'future_data': torch.rand(5, 3, 1),
                'static_data': torch.rand(2),
                'target': torch.rand(5, 3, 1)
            },
            {
                'past_data': torch.rand(10, 3, 1),
                'future_data': torch.rand(5, 3, 1),
                'static_data': torch.rand(2),
                'target': torch.rand(5, 3, 1)
            }
        ]
        
        collated_batch = custom_collate_fn(batch)
        
        assert isinstance(collated_batch, dict)
        assert collated_batch['past_data'].shape == (2, 10, 3, 1)
        assert collated_batch['future_data'].shape == (2, 5, 3, 1)
        assert collated_batch['static_data'].shape == (2, 2)
        assert collated_batch['target'].shape == (2, 5, 3, 1)

    def test_custom_collate_fn_different_size_tensors(self):
        """Test custom_collate_fn with different-sized tensors (should fail)."""
        batch = [
            {'time_series': torch.rand(5, 10)},
            {'time_series': torch.rand(7, 10)}  # Different size
        ]
        
        with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
            custom_collate_fn(batch)

    def test_download_causal_data_success(self):
        """Test _download_causal_data with successful download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('requests.get') as mock_get:
                
                # Mock successful response
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_response.content = b'mock,data\n1,2\n3,4'
                mock_get.return_value = mock_response
                
                _download_causal_data('diamond', temp_dir)
                
                # Verify directory was created and requests were made
                assert os.path.exists(os.path.join(temp_dir, 'diamond'))
                assert mock_get.called

    def test_download_causal_data_already_exists(self):
        """Test _download_causal_data when dataset already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, 'diamond')
            os.makedirs(dataset_dir)
            
            # Create a dummy file to make directory non-empty
            with open(os.path.join(dataset_dir, 'dummy.txt'), 'w') as f:
                f.write('dummy')
            
            with patch('requests.get') as mock_get:
                _download_causal_data('diamond', temp_dir)
                
                # Verify no requests were made
                mock_get.assert_not_called()

    def test_download_causal_data_request_failure(self):
        """Test _download_causal_data with request failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('requests.get') as mock_get, \
                 patch('os.path.exists', return_value=False):
                
                # Mock failed response
                mock_get.side_effect = requests.exceptions.RequestException("Network error")
                
                with pytest.raises(RuntimeError, match="Could not download any files"):
                    _download_causal_data('diamond', temp_dir)

    def test_create_causal_datamodule(self):
        """Test create_causal_datamodule function."""
        datamodule = create_causal_datamodule(
            dataset_type='diamond',
            time_step=15,
            batch_size=64
        )
        
        assert isinstance(datamodule, CausalDataModule)
        assert datamodule.dataset_type == 'diamond'
        assert datamodule.time_step == 15
        assert datamodule.batch_size == 64

    def test_get_clean_adjacency_matrix_with_index_column(self):
        """Test get_clean_adjacency_matrix with index column."""
        df = pl.DataFrame({
            'Unnamed: 0': ['node_0', 'node_1'],
            'node_0': [0.0, 1.0],
            'node_1': [1.0, 0.0]
        })
        
        matrix = get_clean_adjacency_matrix(df)
        
        assert matrix is not None
        assert matrix.shape == (2, 2)
        np.testing.assert_array_equal(matrix, [[0.0, 1.0], [1.0, 0.0]])

    def test_get_clean_adjacency_matrix_without_index_column(self):
        """Test get_clean_adjacency_matrix without index column."""
        df = pl.DataFrame({
            'node_0': [0.0, 1.0],
            'node_1': [1.0, 0.0]
        })
        
        matrix = get_clean_adjacency_matrix(df)
        
        assert matrix is not None
        assert matrix.shape == (2, 2)
        np.testing.assert_array_equal(matrix, [[0.0, 1.0], [1.0, 0.0]])

    def test_get_clean_adjacency_matrix_none_input(self):
        """Test get_clean_adjacency_matrix with None input."""
        result = get_clean_adjacency_matrix(None)
        assert result is None


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_causal_dataset_empty_data_files(self):
        """Test CausalDataset with empty data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, 'diamond')
            os.makedirs(dataset_dir)
            
            # Create empty data file
            with open(os.path.join(dataset_dir, 'data_0.csv'), 'w') as f:
                f.write('node_0,node_1\n')  # Header only
            
            with patch('fintorch.datasets.causal_data._download_causal_data'):
                with pytest.raises(ValueError, match="Could not load any valid data"):
                    CausalDataset(
                        dataset_type='diamond',
                        local_dir=temp_dir,
                        time_step=10,
                        output_window=5
                    )

    def test_causal_dataset_non_numeric_data(self):
        """Test CausalDataset with non-numeric data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, 'diamond')
            os.makedirs(dataset_dir)
            
            # Create data file with non-numeric data
            with open(os.path.join(dataset_dir, 'data_0.csv'), 'w') as f:
                f.write('text_col,another_text\n')
                f.write('hello,world\n')
                f.write('foo,bar\n')
            
            with patch('fintorch.datasets.causal_data._download_causal_data'):
                with pytest.raises(ValueError, match="Could not load any valid data"):
                    CausalDataset(
                        dataset_type='diamond',
                        local_dir=temp_dir,
                        time_step=10,
                        output_window=5
                    )

    def test_causal_dataset_corrupted_csv(self):
        """Test CausalDataset with corrupted CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, 'diamond')
            os.makedirs(dataset_dir)
            
            # Create corrupted CSV file
            with open(os.path.join(dataset_dir, 'data_0.csv'), 'w') as f:
                f.write('corrupted,csv,file\n')
                f.write('1,2\n')  # Missing column
                f.write('3,4,5,6\n')  # Extra column
            
            # Create valid CSV file
            with open(os.path.join(dataset_dir, 'data_1.csv'), 'w') as f:
                f.write('node_0,node_1\n')
                for i in range(50):
                    f.write(f'{i},{i+1}\n')
            
            with patch('fintorch.datasets.causal_data._download_causal_data'):
                # Should still work if at least one file is valid
                dataset = CausalDataset(
                    dataset_type='diamond',
                    local_dir=temp_dir,
                    time_step=10,
                    output_window=5
                )
                assert len(dataset) > 0

    def test_deterministic_static_features(self):
        """Test that static features are deterministic for the same index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, 'diamond')
            os.makedirs(dataset_dir)
            
            # Create minimal data file
            with open(os.path.join(dataset_dir, 'data_0.csv'), 'w') as f:
                f.write('node_0,node_1\n')
                for i in range(50):
                    f.write(f'{i},{i+1}\n')
            
            with patch('fintorch.datasets.causal_data._download_causal_data'):
                dataset = CausalDataset(
                    dataset_type='diamond',
                    local_dir=temp_dir,
                    time_step=10,
                    output_window=5,
                    static_length=5
                )
                
                # Get same sample multiple times
                sample1 = dataset[0]
                sample2 = dataset[0]
                
                # Static features should be identical
                torch.testing.assert_close(
                    sample1['static_features_real'],
                    sample2['static_features_real']
                )


def main():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == "__main__":
    main()