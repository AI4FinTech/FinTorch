#!/usr/bin/env python3
"""
Unit tests for causal data module utility functions.

This test suite verifies the functionality of utility functions in the causal data module,
including directory management, dataset listing, and information retrieval.
"""

import os
import tempfile
import shutil
import pytest
import polars as pl
import numpy as np
from unittest.mock import patch

from fintorch.datasets.causal_data import (
    get_causal_data_dir,
    clear_causal_data,
    list_available_datasets,
    get_dataset_info,
    get_clean_adjacency_matrix
)


def test_get_causal_data_dir():
    """Test the get_causal_data_dir function."""
    # Test default directory
    default_dir = get_causal_data_dir()
    assert default_dir.endswith('.fintorch_data/causal')
    assert os.path.isabs(default_dir)

    # Test specific dataset type directory
    diamond_dir = get_causal_data_dir('diamond')
    assert diamond_dir.endswith('.fintorch_data/causal/diamond')
    assert os.path.isabs(diamond_dir)


def test_clear_causal_data():
    """Test the clear_causal_data function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch the default directory to use temporary directory
        with patch('os.path.expanduser', return_value=temp_dir):
            # Create mock dataset directories
            os.makedirs(os.path.join(temp_dir, 'diamond'), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, 'fork'), exist_ok=True)
            
            # Verify directories exist before clearing
            assert os.path.exists(os.path.join(temp_dir, 'diamond'))
            assert os.path.exists(os.path.join(temp_dir, 'fork'))
            
            # Clear specific dataset
            clear_causal_data('diamond')
            assert not os.path.exists(os.path.join(temp_dir, 'diamond'))
            assert os.path.exists(os.path.join(temp_dir, 'fork'))
            
            # Recreate diamond directory
            os.makedirs(os.path.join(temp_dir, 'diamond'), exist_ok=True)
            
            # Clear all datasets
            clear_causal_data()
            assert not os.path.exists(os.path.join(temp_dir, 'diamond'))
            assert not os.path.exists(os.path.join(temp_dir, 'fork'))


def test_list_available_datasets():
    """Test the list_available_datasets function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch the default directory to use temporary directory
        with patch('os.path.expanduser', return_value=temp_dir):
            # Create mock dataset directories with data files
            for dataset in ['diamond', 'fork', 'mediator']:
                dataset_dir = os.path.join(temp_dir, dataset)
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Create a data file to simulate a valid dataset
                with open(os.path.join(dataset_dir, 'data_0.csv'), 'w') as f:
                    f.write('node_0,node_1\n1.0,2.0\n3.0,4.0')
            
            # Add an empty directory that should be ignored
            os.makedirs(os.path.join(temp_dir, 'empty_dir'), exist_ok=True)
            
            # Test list_available_datasets
            available = list_available_datasets()
            assert set(available) == {'diamond', 'fork', 'mediator'}


def test_get_dataset_info():
    """Test the get_dataset_info function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch the default directory to use temporary directory
        with patch('os.path.expanduser', return_value=temp_dir):
            # Create a mock dataset
            dataset_type = 'diamond'
            dataset_dir = os.path.join(temp_dir, dataset_type)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Create mock data files
            for i in range(2):
                with open(os.path.join(dataset_dir, f'data_{i}.csv'), 'w') as f:
                    # Create a sufficiently large file for size calculation
                    f.write('node_0,node_1\n' + '\n'.join([f'1.{i},2.{i}' for _ in range(1000)]))
            
            # Create groundtruth file
            with open(os.path.join(dataset_dir, 'groundtruth.csv'), 'w') as f:
                f.write('node_0,node_1\n0,1\n1,0')
            
            # Get dataset info
            info = get_dataset_info(dataset_type)
            
            assert info['dataset_type'] == dataset_type
            assert info['exists'] is True
            assert info['path'] == dataset_dir
            assert len(info['data_files']) == 2
            assert 'data_0.csv' in info['data_files']
            assert 'data_1.csv' in info['data_files']
            assert info['has_groundtruth'] is True
            assert info['total_size_mb'] > 0


def test_get_clean_adjacency_matrix():
    """Test the get_clean_adjacency_matrix function."""
    # Test with DataFrame containing an index column
    test_data = pl.DataFrame({
        'Unnamed: 0': ['node_0', 'node_1', 'node_2'],
        'node_0': [0.0, 1.0, 0.0],
        'node_1': [1.0, 0.0, 1.0],
        'node_2': [0.0, 1.0, 0.0]
    })
    
    # Get clean adjacency matrix
    matrix = get_clean_adjacency_matrix(test_data)
    
    # Verify matrix shape and content
    assert matrix.shape == (3, 3)
    np.testing.assert_array_almost_equal(matrix, [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ])
    
    # Test with DataFrame without index column
    test_data_no_index = pl.DataFrame({
        'node_0': [0.0, 1.0, 0.0],
        'node_1': [1.0, 0.0, 1.0],
        'node_2': [0.0, 1.0, 0.0]
    })
    
    matrix_no_index = get_clean_adjacency_matrix(test_data_no_index)
    assert matrix_no_index.shape == (3, 3)
    np.testing.assert_array_almost_equal(matrix_no_index, matrix)
    
    # Test with None input
    assert get_clean_adjacency_matrix(None) is None


def main():
    """Run all tests."""
    pytest.main([__file__])


if __name__ == "__main__":
    main()