"""
Causal Data Loader for FinTorch

This module provides a comprehensive data loader for causal relationship datasets from the 
CausalFormer repository (https://github.com/lingbai-kong/CausalFormer). It supports loading 
multiple types of causal datasets for time series prediction and causal discovery tasks.

🚀 **Powered by Polars**: This module uses Polars DataFrames for high-performance data 
processing, providing faster CSV reading, lower memory usage, and better type safety 
compared to traditional Pandas-based approaches.

Supported Dataset Types:
- diamond: Diamond-shaped causal relationships
- fork: Fork-shaped causal relationships  
- mediator: Mediator causal relationships
- v: V-shaped causal relationships

Data is automatically downloaded and cached in ~/.fintorch_data/causal/ directory structure.

Basic Usage:
    from fintorch.datasets.causal_data import create_causal_datamodule
    
    # Create data module (uses ~/.fintorch_data/causal by default)
    data_module = create_causal_datamodule(
        dataset_type='diamond',
        time_step=20,
        output_window=10,
        batch_size=32
    )
    
    # Setup and use
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    # Access groundtruth causal relationships (Polars DataFrame)
    groundtruth = data_module.get_groundtruth()

Working with Polars DataFrames:
    from fintorch.datasets.causal_data import get_clean_adjacency_matrix
    
    # Get clean adjacency matrix (removes index columns automatically)
    adjacency_matrix = get_clean_adjacency_matrix(groundtruth)
    
    # Convert Polars DataFrame to numpy for visualization
    numpy_array = groundtruth.to_numpy()

Utility Functions:
- get_causal_data_dir(): Get data directory path
- list_available_datasets(): List cached datasets
- clear_causal_data(): Clear cached data
- get_dataset_info(): Get dataset information
- get_clean_adjacency_matrix(): Extract clean adjacency matrix from groundtruth

Performance Benefits of Polars:
- 2-10x faster CSV reading compared to Pandas
- Significantly lower memory usage for large datasets
- Better handling of data types and null values
- More intuitive and consistent API for data operations
"""

import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple
import glob

import lightning as L
import numpy as np
import polars as pl
import requests
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

from fintorch.datasets.base import TimeSeriesDataset


def get_causal_data_dir(dataset_type: str = None) -> str:
    """
    Get the standard FinTorch data directory for causal datasets.
    
    Args:
        dataset_type (str): Optional dataset type to get specific subdirectory
        
    Returns:
        Path to the causal data directory
    """
    base_dir = os.path.expanduser("~/.fintorch_data/causal")
    if dataset_type:
        return os.path.join(base_dir, dataset_type)
    return base_dir


def clear_causal_data(dataset_type: str = None) -> None:
    """
    Clear cached causal dataset files.
    
    Args:
        dataset_type (str): If specified, only clear this dataset type.
                           If None, clear all causal datasets.
    """
    if dataset_type:
        dataset_dir = get_causal_data_dir(dataset_type)
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
            print(f"Cleared cached data for {dataset_type} dataset")
        else:
            print(f"No cached data found for {dataset_type} dataset")
    else:
        base_dir = get_causal_data_dir()
        if os.path.exists(base_dir):
            import shutil
            shutil.rmtree(base_dir)
            print("Cleared all cached causal datasets")
        else:
            print("No cached causal datasets found")


def list_available_datasets() -> List[str]:
    """
    List all locally cached causal datasets.
    
    Returns:
        List of available dataset types
    """
    base_dir = get_causal_data_dir()
    if not os.path.exists(base_dir):
        return []
    
    datasets = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains data files
            data_files = glob.glob(os.path.join(item_path, "data_*.csv"))
            if data_files:
                datasets.append(item)
    
    return sorted(datasets)


def get_dataset_info(dataset_type: str) -> Dict[str, Any]:
    """
    Get information about a cached dataset.
    
    Args:
        dataset_type (str): Type of dataset to get info for
        
    Returns:
        Dictionary with dataset information
    """
    dataset_dir = get_causal_data_dir(dataset_type)
    
    info = {
        'dataset_type': dataset_type,
        'exists': os.path.exists(dataset_dir),
        'path': dataset_dir,
        'data_files': [],
        'has_groundtruth': False,
        'total_size_mb': 0
    }
    
    if info['exists']:
        # Find data files
        data_files = glob.glob(os.path.join(dataset_dir, "data_*.csv"))
        info['data_files'] = [os.path.basename(f) for f in sorted(data_files)]
        
        # Check for groundtruth
        groundtruth_path = os.path.join(dataset_dir, "groundtruth.csv")
        info['has_groundtruth'] = os.path.exists(groundtruth_path)
        
        # Calculate total size
        total_size = 0
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
    
    return info


def get_clean_adjacency_matrix(groundtruth: pl.DataFrame) -> np.ndarray:
    """
    Extract a clean adjacency matrix from groundtruth DataFrame.
    
    Args:
        groundtruth (pl.DataFrame): Groundtruth DataFrame from causal dataset
        
    Returns:
        numpy.ndarray: Clean adjacency matrix with index columns removed
    """
    if groundtruth is None:
        return None
    
    # Remove index-like columns
    index_cols = [col for col in groundtruth.columns 
                 if col.startswith('Unnamed:') or col.lower() in ['index', 'row_id']]
    
    if index_cols:
        clean_df = groundtruth.drop(index_cols)
    else:
        clean_df = groundtruth
    
    # Convert to numpy array
    adjacency_matrix = clean_df.to_numpy().astype(float)
    
    return adjacency_matrix


def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle the new standardized dictionary format.
    """
    # Get all keys from the first sample
    keys = batch[0].keys()

    # Stack tensors for each key
    collated_batch = {}
    for key in keys:
        # Stack all tensors for this key across the batch
        tensors = [sample[key] for sample in batch]
        collated_batch[key] = torch.stack(tensors)

    return collated_batch


def _download_causal_data(dataset_type: str, local_dir: str) -> None:
    """
    Downloads causal dataset from CausalFormer GitHub repository.
    
    Args:
        dataset_type (str): Type of dataset ('diamond', 'fork', 'mediator', 'v')
        local_dir (str): Local directory to store the downloaded data
    """
    base_url = "https://raw.githubusercontent.com/lingbai-kong/CausalFormer/main/data/basic"
    dataset_dir = os.path.join(local_dir, dataset_type)
    
    if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
        print(f"Dataset {dataset_type} already exists at {dataset_dir}. Skipping download.")
        return
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"Downloading {dataset_type} dataset from CausalFormer repository...")
    
    # Try to download data files (data_0.csv, data_1.csv, etc.) and groundtruth.csv
    files_downloaded = 0
    
    # Download groundtruth file
    groundtruth_url = f"{base_url}/{dataset_type}/groundtruth.csv"
    groundtruth_path = os.path.join(dataset_dir, "groundtruth.csv")
    
    try:
        response = requests.get(groundtruth_url)
        response.raise_for_status()
        with open(groundtruth_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded groundtruth.csv for {dataset_type}")
        files_downloaded += 1
    except requests.exceptions.RequestException as e:
        print(f"Could not download groundtruth.csv for {dataset_type}: {e}")
    
    # Try to download data files (data_0.csv, data_1.csv, etc.)
    data_file_index = 0
    max_attempts = 20  # Reasonable limit to avoid infinite loop
    
    while data_file_index < max_attempts:
        data_file_url = f"{base_url}/{dataset_type}/data_{data_file_index}.csv"
        data_file_path = os.path.join(dataset_dir, f"data_{data_file_index}.csv")
        
        try:
            response = requests.get(data_file_url)
            response.raise_for_status()
            with open(data_file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded data_{data_file_index}.csv for {dataset_type}")
            files_downloaded += 1
            data_file_index += 1
        except requests.exceptions.RequestException:
            # No more files to download
            break
    
    if files_downloaded == 0:
        raise RuntimeError(f"Could not download any files for dataset type '{dataset_type}'. Please check if the dataset exists.")
    
    print(f"Downloaded {files_downloaded} files for {dataset_type} dataset.")


class CausalDataset(TimeSeriesDataset):
    """
    PyTorch Dataset for loading and preprocessing causal relationship time-series data.
    Downloads data from CausalFormer GitHub repository if local files don't exist.
    
    Each dataset type (diamond, fork, mediator, v) contains:
    - Multiple data_x.csv files with nodes as columns and measurements as rows
    - A groundtruth.csv file showing true causal relationships between nodes
    """

    def __init__(
        self,
        dataset_type: str,
        local_dir: str = None,
        time_step: int = 10,
        output_window: int = 5,
        static_length: int = 2,
        data_file_pattern: str = "data_*.csv",
    ):
        """
        Initializes the CausalDataset.

        Args:
            dataset_type (str): Type of causal dataset ('diamond', 'fork', 'mediator', 'v')
            local_dir (str): Local directory to save/cache downloaded data. If None, uses ~/.fintorch_data/causal
            time_step (int): Length of the input sequence (past data)
            output_window (int): Length of the target sequence (future data)
            static_length (int): Dimension of dummy static features to generate. Default is 2.
            data_file_pattern (str): Pattern to match data files. Default is "data_*.csv"
        """
        super().__init__()
        self.dataset_type = dataset_type
        if local_dir is None:
            local_dir = os.path.expanduser("~/.fintorch_data/causal")
        self.local_dir = local_dir
        self.dataset_dir = os.path.join(local_dir, dataset_type)
        self._time_step = time_step
        self._output_window = output_window
        self._static_length = static_length

        # Download data if not present
        _download_causal_data(dataset_type, local_dir)

        # Load and combine all data files
        self.data, self.groundtruth = self._load_data(data_file_pattern)
        
        # Generate indices for sampling
        self.indices = self._generate_indices()

    def _load_data(self, data_file_pattern: str) -> Tuple[torch.Tensor, Optional[pl.DataFrame]]:
        """
        Load and combine all data files for the dataset type.
        
        Args:
            data_file_pattern (str): Pattern to match data files
            
        Returns:
            Tuple of (combined_data_tensor, groundtruth_dataframe)
        """
        # Find all data files
        data_files = glob.glob(os.path.join(self.dataset_dir, data_file_pattern))
        data_files.sort()  # Ensure consistent ordering
        
        if not data_files:
            raise FileNotFoundError(f"No data files found matching pattern {data_file_pattern} in {self.dataset_dir}")
        
        print(f"Found {len(data_files)} data files for {self.dataset_type} dataset")
        
        # Load and combine all data files
        all_data = []
        for data_file in data_files:
            try:
                df = pl.read_csv(data_file)
                # Remove any non-numeric columns (like index columns)
                numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
                if not numeric_cols:
                    print(f"Warning: No numeric columns found in {data_file}")
                    continue
                numeric_df = df.select(numeric_cols)
                all_data.append(numeric_df.to_numpy().astype('float32'))
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"Could not load any valid data from {len(data_files)} files")
        
        # Combine all data (concatenate along time dimension)
        combined_data = np.vstack(all_data)
        
        # Store dataset properties
        self._series_num = combined_data.shape[1]
        self._features_dim = 1  # Each node is treated as a single feature
        
        # Scale the data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(combined_data)
        
        # Load groundtruth if available
        groundtruth = None
        groundtruth_path = os.path.join(self.dataset_dir, "groundtruth.csv")
        if os.path.exists(groundtruth_path):
            try:
                groundtruth_raw = pl.read_csv(groundtruth_path)
                
                # Handle index columns (common in CSV exports)
                # Remove columns that look like index columns
                index_cols = [col for col in groundtruth_raw.columns 
                             if col.startswith('Unnamed:') or col.lower() in ['index', 'row_id']]
                
                if index_cols:
                    groundtruth = groundtruth_raw.drop(index_cols)
                    print(f"Removed index columns: {index_cols}")
                else:
                    groundtruth = groundtruth_raw
                
                print(f"Loaded groundtruth causal relationships for {self.dataset_type} (shape: {groundtruth.shape})")
            except Exception as e:
                print(f"Warning: Could not load groundtruth file: {e}")
        
        return torch.tensor(scaled_data, dtype=torch.float32), groundtruth

    def _generate_indices(self) -> List[int]:
        """
        Generate valid sample indices based on time step and output window constraints.
        
        Returns:
            List of valid end indices for sample windows
        """
        indices = []
        for i in range(self._time_step, len(self.data)):
            if i + self._output_window <= len(self.data):
                indices.append(i)
        
        if not indices:
            raise ValueError(
                f"No valid indices found. Data length: {len(self.data)}, "
                f"time_step: {self._time_step}, output_window: {self._output_window}"
            )
        
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def time_steps(self) -> int:
        """Returns the number of time steps in the input sequence."""
        return self._time_step

    @property
    def future_steps(self) -> int:
        """Returns the number of future steps to predict."""
        return self._output_window

    @property
    def series_dim(self) -> int:
        """Returns the number of time series (nodes) in the dataset."""
        return self._series_num

    @property
    def features_dim(self) -> int:
        """Returns the feature dimension per time series."""
        return self._features_dim

    @property
    def static_length(self) -> int:
        """Returns the length of static features."""
        return self._static_length

    @property
    def static_categorical_cardinalities(self) -> List[int]:
        """Returns cardinalities for categorical static features."""
        return []

    @property
    def num_target_features(self) -> int:
        """Returns the number of target features."""
        return self._series_num

    @property
    def num_known_future_cov_features(self) -> int:
        """Returns the number of known future covariate features."""
        return 0

    @property
    def num_unknown_future_cov_features(self) -> int:
        """Returns the number of unknown future covariate features."""
        return 0

    @property
    def num_static_real_features(self) -> int:
        """Returns the number of static real-valued features."""
        return self._static_length

    @property
    def num_static_categorical_features(self) -> int:
        """Returns the number of static categorical features."""
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Dictionary containing past_data, future_data, static_data, and target tensors
        """
        end_idx = self.indices[idx]
        start_idx = end_idx - self._time_step
        
        # Extract past and future data
        past_data = self.data[start_idx:end_idx]  # Shape: (time_step, series_num)
        future_data = self.data[end_idx:end_idx + self._output_window]  # Shape: (output_window, series_num)
        
        # Add feature dimension
        past_data = past_data.unsqueeze(-1)  # Shape: (time_step, series_num, 1)
        future_data = future_data.unsqueeze(-1)  # Shape: (output_window, series_num, 1)
        
        # Generate deterministic dummy static features based on sample index
        torch.manual_seed(idx)
        static_data = torch.randn(self._static_length)
        # Reset to a different seed to avoid affecting other random operations
        torch.manual_seed(torch.initial_seed())
        
        # Create target (same as future_data for this use case)
        target = future_data.clone()
        
        return {
            "past_data": past_data,
            "future_data": future_data,
            "static_data": static_data,
            "target": target,
        }

    def get_groundtruth(self) -> Optional[pl.DataFrame]:
        """
        Returns the groundtruth causal relationships if available.
        
        Returns:
            DataFrame containing causal relationships or None if not available
        """
        return self.groundtruth


class CausalDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for causal datasets from CausalFormer repository.
    Supports multiple dataset types: diamond, fork, mediator, v.
    """

    def __init__(
        self,
        dataset_type: str,
        local_dir: str = None,
        time_step: int = 10,
        output_window: int = 5,
        static_length: int = 2,
        batch_size: int = 128,
        num_workers: int = 0,
        train_split: float = 0.7,
        val_split: float = 0.15,
        data_file_pattern: str = "data_*.csv",
    ):
        """
        Initialize the CausalDataModule.
        
        Args:
            dataset_type (str): Type of causal dataset ('diamond', 'fork', 'mediator', 'v')
            local_dir (str): Local directory to save/cache downloaded data. If None, uses ~/.fintorch_data/causal
            time_step (int): Length of input sequence
            output_window (int): Length of target sequence
            static_length (int): Dimension of static features
            batch_size (int): Batch size for data loaders
            num_workers (int): Number of workers for data loading
            train_split (float): Fraction of data for training
            val_split (float): Fraction of data for validation
            data_file_pattern (str): Pattern to match data files
        """
        super().__init__()
        
        # Validate dataset type
        valid_types = ['diamond', 'fork', 'mediator', 'v']
        if dataset_type not in valid_types:
            raise ValueError(f"dataset_type must be one of {valid_types}, got {dataset_type}")
        
        # Store parameters
        self.dataset_type = dataset_type
        if local_dir is None:
            local_dir = os.path.expanduser("~/.fintorch_data/causal")
        self.local_dir = local_dir
        self.time_step = time_step
        self.output_window = output_window
        self.static_length = static_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.data_file_pattern = data_file_pattern

        # Save hyperparameters
        self.save_hyperparameters()

        # Validate splits
        self.test_split = 1.0 - self.train_split - self.val_split
        if not (0 < self.train_split < 1 and 0 < self.val_split < 1 and 0 < self.test_split < 1):
            raise ValueError("Train, validation, and test splits must be between 0 and 1 and sum to 1.")

        # Initialize dataset placeholders
        self.dataset: Optional[CausalDataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the datasets for training, validation, and testing."""
        if not self.dataset:
            self.dataset = CausalDataset(
                dataset_type=self.dataset_type,
                local_dir=self.local_dir,
                time_step=self.time_step,
                output_window=self.output_window,
                static_length=self.static_length,
                data_file_pattern=self.data_file_pattern,
            )

            # Split dataset
            total_size = len(self.dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns the prediction DataLoader (same as test)."""
        return self.test_dataloader()

    def get_groundtruth(self) -> Optional[pl.DataFrame]:
        """
        Returns the groundtruth causal relationships if available.
        
        Returns:
            DataFrame containing causal relationships or None if not available
        """
        if self.dataset:
            return self.dataset.get_groundtruth()
        return None


# Convenience function to create data modules for different dataset types
def create_causal_datamodule(
    dataset_type: str,
    local_dir: str = None,
    time_step: int = 10,
    output_window: int = 5,
    **kwargs
) -> CausalDataModule:
    """
    Convenience function to create a CausalDataModule.
    
    Args:
        dataset_type (str): Type of causal dataset ('diamond', 'fork', 'mediator', 'v')
        local_dir (str): Local directory to store data. If None, uses ~/.fintorch_data/causal
        time_step (int): Length of input sequence
        output_window (int): Length of target sequence
        **kwargs: Additional arguments for CausalDataModule
        
    Returns:
        CausalDataModule instance
    """
    if local_dir is None:
        local_dir = os.path.expanduser("~/.fintorch_data/causal")
    
    return CausalDataModule(
        dataset_type=dataset_type,
        local_dir=local_dir,
        time_step=time_step,
        output_window=output_window,
        **kwargs
    )