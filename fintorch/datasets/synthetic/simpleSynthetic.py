from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import numpy.typing as npt
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


class SimpleSyntheticDataset(TimeSeriesDataset):
    """
    SimpleSyntheticDataset is a PyTorch Dataset that generates synthetic time series data
    with configurable trend, seasonality, and noise components. It is designed for tasks
    involving time series forecasting and returns data in the new standardized dictionary format.

    Attributes:
        length (int): Total length of the generated time series data.
        trend_slope (float): Slope of the linear trend component. Default is 0.1.
        seasonality_amplitude (float): Amplitude of the sinusoidal seasonality component. Default is 1.0.
        seasonality_period (int): Period of the sinusoidal seasonality component. Default is 10.
        noise_level (float): Standard deviation of the Gaussian noise component. Default is 0.1.
        past_length (int): Length of the past data window. Default is 10.
        future_length (int): Length of the future data window. Default is 5.
        num_series (int): Number of time series to generate. Default is 1.
        num_target_features (int): Number of target features for each time series. Default is 1.
        num_known_cov_features (int): Number of known future covariate features. Default is 2.
        num_unknown_cov_features (int): Number of unknown future covariate features. Default is 1.
        num_static_real_features (int): Number of real-valued static features. Default is 2.
        num_static_categorical_features (int): Number of categorical static features. Default is 1.
        scalers (List[List[StandardScaler]]): 2D list of scalers, one per (series, feature) pair.
    """

    def __init__(
        self,
        length: int,
        trend_slope: float = 0.1,
        seasonality_amplitude: float = 1.0,
        seasonality_period: int = 10,
        noise_level: float = 0.1,
        past_length: int = 10,
        future_length: int = 5,
        num_series: int = 1,
        num_target_features: int = 1,
        num_known_cov_features: int = 2,
        num_unknown_cov_features: int = 1,
        num_static_real_features: int = 2,
        num_static_categorical_features: int = 1,
        static_categorical_cardinalities: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.length = length
        self.trend_slope = trend_slope
        self.seasonality_amplitude = seasonality_amplitude
        self.seasonality_period = seasonality_period
        self.noise_level = noise_level
        self._past_length = past_length
        self._future_length = future_length
        self._num_series = num_series

        # New granular feature dimensions
        self._num_target_features = num_target_features
        self._num_known_cov_features = num_known_cov_features
        self._num_unknown_cov_features = num_unknown_cov_features
        self._num_static_real_features = num_static_real_features
        self._num_static_categorical_features = num_static_categorical_features

        # Set default cardinalities if not provided
        if static_categorical_cardinalities is None:
            self._static_categorical_cardinalities = [5] * num_static_categorical_features
        else:
            self._static_categorical_cardinalities = static_categorical_cardinalities


        self._static_length = num_static_real_features + num_static_categorical_features


        self._features_dim = num_target_features + num_known_cov_features + num_unknown_cov_features

        self.data = self._generate_data()

    def _generate_data(self) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Generate synthetic time series data with trend, seasonality, and noise components.

        Returns:
            Dict containing generated data arrays for different feature types
        """
        # Initialize scalers for different feature types
        self.target_scalers = [
            [StandardScaler() for _ in range(self._num_target_features)]
            for _ in range(self._num_series)
        ]
        self.known_cov_scalers = [
            [StandardScaler() for _ in range(self._num_known_cov_features)]
            for _ in range(self._num_series)
        ]
        self.unknown_cov_scalers = [
            [StandardScaler() for _ in range(self._num_unknown_cov_features)]
            for _ in range(self._num_series)
        ]
        self.static_real_scalers = [
            [StandardScaler() for _ in range(self._num_static_real_features)]
            for _ in range(self._num_series)
        ]

        # Create empty arrays to store data
        target_data = np.zeros((self.length, self._num_series, self._num_target_features))
        known_cov_data = np.zeros((self.length, self._num_series, self._num_known_cov_features))
        unknown_cov_data = np.zeros((self.length, self._num_series, self._num_unknown_cov_features))
        static_real_data = np.zeros((self._num_series, self._num_static_real_features))
        static_categorical_data = np.zeros((self._num_series, self._num_static_categorical_features), dtype=int)

        # Generate each series
        for series_idx in range(self._num_series):
            # Generate target features
            for feat_idx in range(self._num_target_features):
                data = self._generate_single_series(series_idx, feat_idx, 'target')
                scaled_data = self.target_scalers[series_idx][feat_idx].fit_transform(
                    np.array(data).reshape(-1, 1)
                )
                target_data[:, series_idx, feat_idx] = scaled_data.flatten()

            # Generate known covariate features (e.g., day of week, hour)
            for feat_idx in range(self._num_known_cov_features):
                data = self._generate_single_series(series_idx, feat_idx, 'known_cov')
                scaled_data = self.known_cov_scalers[series_idx][feat_idx].fit_transform(
                    np.array(data).reshape(-1, 1)
                )
                known_cov_data[:, series_idx, feat_idx] = scaled_data.flatten()

            # Generate unknown covariate features (e.g., weather, external factors)
            for feat_idx in range(self._num_unknown_cov_features):
                data = self._generate_single_series(series_idx, feat_idx, 'unknown_cov')
                scaled_data = self.unknown_cov_scalers[series_idx][feat_idx].fit_transform(
                    np.array(data).reshape(-1, 1)
                )
                unknown_cov_data[:, series_idx, feat_idx] = scaled_data.flatten()

            # Generate static real features
            for feat_idx in range(self._num_static_real_features):
                # Static features don't change over time
                value = np.random.normal(series_idx * 0.5, 1.0)
                static_real_data[series_idx, feat_idx] = value

            # Generate static categorical features
            for feat_idx in range(self._num_static_categorical_features):
                cardinality = self._static_categorical_cardinalities[feat_idx]
                value = np.random.randint(0, cardinality)
                static_categorical_data[series_idx, feat_idx] = value

        return {
            'target': target_data,
            'known_cov': known_cov_data,
            'unknown_cov': unknown_cov_data,
            'static_real': static_real_data,
            'static_categorical': static_categorical_data
        }

    def _generate_single_series(self, series_idx: int, feat_idx: int, feature_type: str) -> List[float]:
        """
        Generate a single time series with different characteristics based on feature type.
        """
        data = []

        # Adjust parameters for diversity
        series_adjust = 1.0 + 0.1 * (series_idx / max(1, self._num_series))
        feat_adjust = 1.0 + 0.05 * (feat_idx / max(1, 3))  # Normalize by typical feature count

        # Different patterns for different feature types
        if feature_type == 'target':
            trend_factor = self.trend_slope * series_adjust
            seasonality_factor = self.seasonality_amplitude * feat_adjust
        elif feature_type == 'known_cov':
            # Known covariates might have more predictable patterns
            trend_factor = self.trend_slope * 0.5 * series_adjust
            seasonality_factor = self.seasonality_amplitude * 0.8 * feat_adjust
        else:  # unknown_cov
            # Unknown covariates might be more noisy
            trend_factor = self.trend_slope * 0.3 * series_adjust
            seasonality_factor = self.seasonality_amplitude * 0.6 * feat_adjust

        for i in range(self.length):
            # Trend component
            trend = trend_factor * i

            # Seasonality component with phase shift for diversity
            phase_shift = (
                0.5 * np.pi * (series_idx / max(1, self._num_series - 1))
                if self._num_series > 1
                else 0
            )
            seasonality = seasonality_factor * np.sin(
                2 * np.pi * i / self.seasonality_period + phase_shift
            )

            # Noise component
            noise = np.random.normal(0, self.noise_level)

            # Combine components
            value = trend + seasonality + noise
            data.append(value)

        return data

    def __len__(self) -> int:
        return self.length - self._past_length - self._future_length

    @property
    def time_steps(self) -> int:
        return self._past_length

    @property
    def future_steps(self) -> int:
        return self._future_length

    @property
    def series_dim(self) -> int:
        return self._num_series

    @property
    def features_dim(self) -> int:
        """Legacy property for backward compatibility."""
        return self._features_dim

    @property
    def past_length(self) -> int:
        return self._past_length

    @property
    def future_length(self) -> int:
        return self._future_length

    @property
    def static_length(self) -> int:
        """Legacy property for backward compatibility."""
        return self._static_length

    @property
    def static_categorical_cardinalities(self) -> List[int]:
        return self._static_categorical_cardinalities

    @property
    def num_target_features(self) -> int:
        return self._num_target_features

    @property
    def num_known_future_cov_features(self) -> int:
        return self._num_known_cov_features

    @property
    def num_unknown_future_cov_features(self) -> int:
        return self._num_unknown_cov_features

    @property
    def num_static_real_features(self) -> int:
        return self._num_static_real_features

    @property
    def num_static_categorical_features(self) -> int:
        return self._num_static_categorical_features

    def get_scaler(self, series_idx: int = 0, feature_type: str = 'target', feature_idx: int = 0) -> StandardScaler:
        """
        Get the scaler for a specific series and feature.

        Args:
            series_idx: Index of the series
            feature_type: Type of feature ('target', 'known_cov', 'unknown_cov')
            feature_idx: Index of the feature within the feature type

        Returns:
            StandardScaler: The fitted scaler for the specified feature
        """
        if feature_type == 'target':
            return self.target_scalers[series_idx][feature_idx]
        elif feature_type == 'known_cov':
            return self.known_cov_scalers[series_idx][feature_idx]
        elif feature_type == 'unknown_cov':
            return self.unknown_cov_scalers[series_idx][feature_idx]
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def inverse_transform(
        self,
        data: torch.Tensor,
        series_idx: int = 0,
        feature_type: str = 'target',
        feature_idx: int = 0
    ) -> torch.Tensor:
        """
        Inverse transform the scaled data back to original scale.

        Args:
            data: Scaled tensor data
            series_idx: Index of the series
            feature_type: Type of feature ('target', 'known_cov', 'unknown_cov')
            feature_idx: Index of the feature within the feature type

        Returns:
            torch.Tensor: Data in original scale
        """
        scaler = self.get_scaler(series_idx, feature_type, feature_idx)
        data_np = data.detach().cpu().numpy().reshape(-1, 1)
        original_data = scaler.inverse_transform(data_np)
        return torch.tensor(original_data).reshape(data.shape).to(data.device)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset at the specified index in the new standardized format.

        Args:
            idx (int): The index of the sample to retrieve

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing tensors with the following keys:
                - "past_target": Shape (past_length, series_dim, num_target_features)
                - "past_covariates_known_future": Shape (past_length, series_dim, num_known_cov_features)
                - "past_covariates_unknown_future": Shape (past_length, series_dim, num_unknown_cov_features)
                - "future_covariates_known": Shape (future_length, series_dim, num_known_cov_features)
                - "output_target": Shape (future_length, series_dim, num_target_features)
                - "static_features_real": Shape (series_dim, num_static_real_features)
                - "static_features_categorical": Shape (series_dim, num_static_categorical_features)
        """
        # Extract time windows
        past_start = idx
        past_end = idx + self._past_length
        future_start = past_end
        future_end = future_start + self._future_length

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


class SimpleSyntheticDataModule(L.LightningDataModule):
    """
    Lightning DataModule for SimpleSyntheticDataset.
    """

    def __init__(
        self,
        train_length: int,
        val_length: int,
        test_length: int,
        batch_size: int,
        trend_slope: float = 0.1,
        seasonality_amplitude: float = 1.0,
        seasonality_period: int = 10,
        noise_level: float = 0.1,
        past_length: int = 10,
        future_length: int = 5,
        num_series: int = 1,
        num_target_features: int = 1,
        num_known_cov_features: int = 2,
        num_unknown_cov_features: int = 1,
        num_static_real_features: int = 2,
        num_static_categorical_features: int = 1,
        static_categorical_cardinalities: Optional[List[int]] = None,
        workers: int = 1,
    ) -> None:
        super().__init__()
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.batch_size = batch_size
        self.trend_slope = trend_slope
        self.seasonality_amplitude = seasonality_amplitude
        self.seasonality_period = seasonality_period
        self.noise_level = noise_level
        self.past_length = past_length
        self.future_length = future_length
        self.num_series = num_series
        self.num_target_features = num_target_features
        self.num_known_cov_features = num_known_cov_features
        self.num_unknown_cov_features = num_unknown_cov_features
        self.num_static_real_features = num_static_real_features
        self.num_static_categorical_features = num_static_categorical_features
        self.static_categorical_cardinalities = static_categorical_cardinalities
        self.workers = workers

        # Calculate static_length from granular parameters
        self.static_length = num_static_real_features + num_static_categorical_features

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage."""
        common_params = {
            'trend_slope': self.trend_slope,
            'seasonality_amplitude': self.seasonality_amplitude,
            'seasonality_period': self.seasonality_period,
            'noise_level': self.noise_level,
            'past_length': self.past_length,
            'future_length': self.future_length,
            'num_series': self.num_series,
            'num_target_features': self.num_target_features,
            'num_known_cov_features': self.num_known_cov_features,
            'num_unknown_cov_features': self.num_unknown_cov_features,
            'num_static_real_features': self.num_static_real_features,
            'num_static_categorical_features': self.num_static_categorical_features,
            'static_categorical_cardinalities': self.static_categorical_cardinalities,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = SimpleSyntheticDataset(
                length=self.train_length,
                **common_params
            )
            self.val_dataset = SimpleSyntheticDataset(
                length=self.val_length,
                **common_params
            )

        if stage == "test" or stage is None:
            self.test_dataset = SimpleSyntheticDataset(
                length=self.test_length,
                **common_params
            )

        if stage == "predict" or stage is None:
            self.predict_dataset = SimpleSyntheticDataset(
                length=self.test_length,
                **common_params
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
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=custom_collate_fn,
        )
