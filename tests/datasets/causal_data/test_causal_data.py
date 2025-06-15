#!/usr/bin/env python3
"""
Unit tests for causal data module classes and functions.

This test suite verifies the functionality of CausalDataset and CausalDataModule,
as well as supplementary functions like custom_collate_fn and _download_causal_data.
"""

import os
import sys
import pytest
import torch
import requests
from unittest.mock import patch, MagicMock

# Add project root to Python path for CI environments
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from fintorch.datasets.causal_data import (
    CausalDataset,
    CausalDataModule,
    create_causal_datamodule,
    custom_collate_fn,
    _download_causal_data
)


@pytest.fixture
def sample_causal_dataset_config():
    """Fixture providing a basic configuration for CausalDataset."""
    return {
        'dataset_type': 'diamond',
        'time_step': 10,
        'output_window': 5,
    }


def test_causal_dataset_initialization(sample_causal_dataset_config):
    """Test initialization of CausalDataset."""
    with patch('fintorch.datasets.causal_data._download_causal_data', return_value=None), \
         patch.object(CausalDataset, '_load_data', return_value=(torch.randn(100, 5), None)), \
         patch.object(CausalDataset, '_generate_indices', return_value=list(range(10, 90))):

        dataset = CausalDataset(**sample_causal_dataset_config)

        assert dataset.dataset_type == 'diamond'
        assert dataset._time_step == 10
        assert dataset._output_window == 5


def test_causal_dataset_getitem(sample_causal_dataset_config, monkeypatch):
    """Test __getitem__ method of CausalDataset."""
    def mock_load_data(self, data_file_pattern):
        # Create mock data tensor and groundtruth
        self.data = torch.randn(100, 3)  # 100 time steps, 3 nodes
        self._series_num = 3
        self._features_dim = 1
        return self.data, None

    def mock_generate_indices(self):
        return list(range(self._time_step, 90))

    with patch('fintorch.datasets.causal_data._download_causal_data'):
        monkeypatch.setattr(CausalDataset, '_load_data', mock_load_data)
        monkeypatch.setattr(CausalDataset, '_generate_indices', mock_generate_indices)

        dataset = CausalDataset(**sample_causal_dataset_config)

        # Attempt to get an item
        item = dataset[0]

        # Basic checks
        assert isinstance(item, dict)
        assert 'past_target' in item
        assert 'output_target' in item

        # Check shapes
        assert item['past_target'].shape == (10, 3, 1)
        assert item['output_target'].shape == (5, 3, 1)
        assert item['static_features_real'].shape == (3, 2)  # (series_num, static_length)


def test_causal_datamodule_initialization():
    """Test initialization of CausalDataModule."""
    datamodule = CausalDataModule(
        dataset_type='diamond',
        batch_size=32,
        num_workers=2
    )

    assert datamodule.dataset_type == 'diamond'
    assert datamodule.batch_size == 32
    assert datamodule.num_workers == 2


def test_causal_datamodule_dataloaders():
    """Test dataloader generation in CausalDataModule."""
    with patch.object(CausalDataModule, 'setup'):
        datamodule = CausalDataModule(
            dataset_type='diamond',
            batch_size=32,
            num_workers=2
        )

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

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None


def test_custom_collate_fn():
    """Test custom_collate_fn for handling same-sized sequences."""
    # Create batch with same-sized tensors (as expected by the function)
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

    # Basic checks
    assert isinstance(collated_batch, dict)
    assert 'past_data' in collated_batch
    assert 'future_data' in collated_batch
    assert 'static_data' in collated_batch
    assert 'target' in collated_batch

    # Check batch dimension
    assert collated_batch['past_data'].shape[0] == len(batch)
    assert collated_batch['future_data'].shape[0] == len(batch)
    assert collated_batch['static_data'].shape[0] == len(batch)
    assert collated_batch['target'].shape[0] == len(batch)


def test_download_causal_data(tmp_path):
    """Test _download_causal_data function."""
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=False):

        # Create a mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'mock,data\n1.0,2.0\n3.0,4.0'
        mock_get.return_value = mock_response

        # Test successful download
        _download_causal_data('diamond', str(tmp_path))

        # Verify directory was created
        dataset_dir = tmp_path / 'diamond'
        assert dataset_dir.exists()


def test_download_causal_data_no_files():
    """Test _download_causal_data function when no files can be downloaded."""
    with patch('requests.get') as mock_get, \
         patch('os.path.exists', return_value=False):

        # Mock failed requests
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(RuntimeError, match="Could not download any files"):
            _download_causal_data('diamond', '/tmp')


def test_create_causal_datamodule():
    """Test create_causal_datamodule function."""
    datamodule = create_causal_datamodule(
        dataset_type='diamond',
        batch_size=32
    )

    assert isinstance(datamodule, CausalDataModule)
    assert datamodule.dataset_type == 'diamond'
    assert datamodule.batch_size == 32


def main():
    """Run all tests."""
    pytest.main([__file__])


if __name__ == "__main__":
    main()
