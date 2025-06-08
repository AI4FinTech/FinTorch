from typing import Any, Dict, List, Optional

import lightning as L
import polars as pl
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch.utils.data import DataLoader

from fintorch.datasets.base import TimeSeriesDataset


def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle the new dictionary format.
    """
    collated_batch = {}

    # Get all keys from the first item in the batch
    keys = batch[0].keys()

    for key in keys:
        # Stack tensors for each key across the batch
        tensors = [item[key] for item in batch if item[key] is not None]
        if tensors:
            collated_batch[key] = torch.stack(tensors)
        else:
            collated_batch[key] = None

    return collated_batch


class AirPassengerDataset(TimeSeriesDataset):
    """
    A PyTorch Dataset for the Air Passenger dataset that inherits from TimeSeriesDataset.
    This dataset provides time-series data for training and testing machine learning models.
    It uses the "Airline Passengers" dataset, which contains monthly totals of international
    airline passengers from 1949 to 1960, scaled using StandardScaler normalization.

    The dataset returns data in the new standardized dictionary format with proper
    feature separation into target, covariates, and static features.

    Args:
        past_length (int, optional): The number of past time steps to include in the input. Default is 10.
        future_length (int, optional): The number of future time steps to predict. Default is 5.
        start_idx (int, optional): The starting index of the dataset slice. Default is 0.
        end_idx (int, optional): The ending index of the dataset slice. If None, it is set to the length of the data
                                 minus `past_length` and `future_length`. Default is None.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing tensors with the following keys:
            - "past_target": Historical air passenger counts
            - "past_covariates_known_future": Historical time features (month, trend)
            - "past_covariates_unknown_future": Historical external features (placeholder)
            - "future_covariates_known": Future time features
            - "output_target": Target air passenger counts to predict
            - "static_features_real": Real-valued static features (placeholder)
            - "static_features_categorical": Categorical static features (placeholder)

    Example:
        dataset = AirPassengerDataset(past_length=12, future_length=6)
        sample = dataset[0]
        print(sample["past_target"].shape)  # torch.Size([12, 1, 1])
        print(sample["output_target"].shape)  # torch.Size([6, 1, 1])
    """
    def __init__(
        self,
        past_length: int = 10,
        future_length: int = 5,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.past_length = past_length
        self.future_length = future_length

        self.data = self._get_data()

        # Define dataset length based on indices
        self.start_idx = start_idx
        self.end_idx = (
            end_idx
            if end_idx is not None
            else len(self.data['target']) - past_length - future_length
        )
        self.length = self.end_idx - self.start_idx

    def _get_data(self) -> Dict[str, Any]:
        """
        Load and preprocess the air passenger data.

        Returns:
            Dict containing processed data arrays for different feature types
        """
        data = pl.read_csv(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        )

        print(f"Air Passenger dataset size: {data.shape}")

        # Extract passenger counts
        passenger_col = data.columns[1]  # Passenger count column

        # Scale the target (passenger counts)
        target_scaler = StandardScaler()
        target_data = data.select(pl.col(passenger_col)).to_numpy().reshape(-1, 1)
        target_scaled = target_scaler.fit_transform(target_data).flatten()

        # Create time features (month and trend)
        data_length = len(target_scaled)
        months = np.array([(i % 12) + 1 for i in range(data_length)])  # Month 1-12
        trend = np.arange(data_length).astype(float)  # Linear trend

        # Scale time features
        time_features = np.column_stack([months, trend])
        time_scaler = StandardScaler()
        time_features_scaled = time_scaler.fit_transform(time_features)

        # Store scalers for later use
        self.target_scaler = target_scaler
        self.time_scaler = time_scaler

        # For this dataset, we treat:
        # - Target: passenger counts (1 feature)
        # - Known future covariates: time features (2 features: month, trend)
        # - Unknown future covariates: placeholder (1 feature of zeros)
        # - Static real features: placeholder (2 features)
        # - Static categorical features: placeholder (1 feature)

        return {
            'target': target_scaled.reshape(-1, 1, 1),  # (time, series=1, features=1)
            'known_cov': time_features_scaled.reshape(-1, 1, 2),  # (time, series=1, features=2)
            'unknown_cov': np.zeros((data_length, 1, 1)),  # (time, series=1, features=1)
            'static_real': np.array([[0.0, 0.0]]),  # (series=1, features=2)
            'static_categorical': np.array([[0]], dtype=int)  # (series=1, features=1)
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset at the specified index in the new standardized format.

        Args:
            idx (int): The index of the sample to retrieve

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing tensors with standardized keys
        """
        if isinstance(idx, slice):
            return [
                self[i]
                for i in range(idx.start or 0, idx.stop or len(self), idx.step or 1)
            ]

        # Adjust index to be within the dataset slice
        idx = self.start_idx + idx

        # Extract time windows
        past_start = idx
        past_end = idx + self.past_length
        future_start = past_end
        future_end = future_start + self.future_length

        # Extract past data
        past_target = self.data['target'][past_start:past_end]
        past_known_cov = self.data['known_cov'][past_start:past_end]
        past_unknown_cov = self.data['unknown_cov'][past_start:past_end]

        # Extract future data
        future_known_cov = self.data['known_cov'][future_start:future_end]
        output_target = self.data['target'][future_start:future_end]

        # Static data (same for all time steps)
        static_real = self.data['static_real']
        static_categorical = self.data['static_categorical']

        # Convert to tensors
        result = {
            "past_target": torch.tensor(past_target, dtype=torch.float32),
            "past_covariates_known_future": torch.tensor(past_known_cov, dtype=torch.float32),
            "past_covariates_unknown_future": torch.tensor(past_unknown_cov, dtype=torch.float32),
            "future_covariates_known": torch.tensor(future_known_cov, dtype=torch.float32),
            "output_target": torch.tensor(output_target, dtype=torch.float32),
            "static_features_real": torch.tensor(static_real, dtype=torch.float32),
            "static_features_categorical": torch.tensor(static_categorical, dtype=torch.long),
        }

        return result

    # Abstract method implementations
    @property
    def time_steps(self) -> int:
        return self.past_length

    @property
    def future_steps(self) -> int:
        return self.future_length

    @property
    def series_dim(self) -> int:
        return 1  # Single time series

    @property
    def features_dim(self) -> int:
        """Legacy property for backward compatibility."""
        return 4  # 1 target + 2 known cov + 1 unknown cov

    @property
    def static_length(self) -> int:
        """Legacy property for backward compatibility."""
        return 3  # 2 real + 1 categorical

    @property
    def static_categorical_cardinalities(self) -> List[int]:
        return [2]  # Single categorical feature with 2 categories

    @property
    def num_target_features(self) -> int:
        return 1

    @property
    def num_known_future_cov_features(self) -> int:
        return 2  # month, trend

    @property
    def num_unknown_future_cov_features(self) -> int:
        return 1  # placeholder

    @property
    def num_static_real_features(self) -> int:
        return 2  # placeholder

    @property
    def num_static_categorical_features(self) -> int:
        return 1  # placeholder

    def get_scaler(self, feature_type: str = 'target') -> StandardScaler:
        """
        Get the scaler for a specific feature type.

        Args:
            feature_type: Type of feature ('target' or 'time')

        Returns:
            StandardScaler: The fitted scaler for the specified feature type
        """
        if feature_type == 'target':
            return self.target_scaler
        elif feature_type == 'time':
            return self.time_scaler
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def inverse_transform(
        self,
        data: torch.Tensor,
        feature_type: str = 'target'
    ) -> torch.Tensor:
        """
        Inverse transform the scaled data back to original scale.

        Args:
            data: Scaled tensor data
            feature_type: Type of feature ('target' or 'time')

        Returns:
            torch.Tensor: Data in original scale
        """
        scaler = self.get_scaler(feature_type)
        data_np = data.detach().cpu().numpy().reshape(-1, 1)
        original_data = scaler.inverse_transform(data_np)
        return torch.tensor(original_data).reshape(data.shape).to(data.device)


class AirPassengerDataModule(L.LightningDataModule):
    """
    Lightning DataModule for AirPassengerDataset.
    """
    def __init__(
        self,
        batch_size: int,
        past_length: int = 10,
        horizon: int = 5,
        workers: int = 1,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.past_length = past_length
        self.future_length = horizon
        self.workers = workers

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = AirPassengerDataset(
            past_length=self.past_length,
            future_length=self.future_length,
        )

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))

        # Create separate dataset instances to preserve time-series order
        if stage == "fit" or stage is None:
            self.train_dataset = AirPassengerDataset(
                past_length=self.past_length,
                future_length=self.future_length,
                start_idx=0,
                end_idx=train_size,
            )

            self.val_dataset = AirPassengerDataset(
                past_length=self.past_length,
                future_length=self.future_length,
                start_idx=train_size,
                end_idx=train_size + val_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = AirPassengerDataset(
                past_length=self.past_length,
                future_length=self.future_length,
                start_idx=train_size + val_size,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=custom_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=custom_collate_fn,
        )
