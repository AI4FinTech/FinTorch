from typing import Any, Dict, Optional, Tuple

import lightning as L
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch.utils.data import DataLoader

from fintorch.datasets.base import TimeSeriesDataset


class SimpleSyntheticDataset(TimeSeriesDataset):
    """
    SimpleSyntheticDataset is a PyTorch Dataset that generates synthetic time series data
    with configurable trend, seasonality, and noise components. It is designed for tasks
    involving time series forecasting and includes past, future, and static data features.
    Note that the static data is randomly generated and does not contain relevant information for
    the model to learn from.

    Attributes:
        length (int): Total length of the generated time series data.
        trend_slope (float): Slope of the linear trend component. Default is 0.1.
        seasonality_amplitude (float): Amplitude of the sinusoidal seasonality component. Default is 1.0.
        seasonality_period (int): Period of the sinusoidal seasonality component. Default is 10.
        noise_level (float): Standard deviation of the Gaussian noise component. Default is 0.1.
        past_length (int): Length of the past data window. Default is 10.
        future_length (int): Length of the future data window. Default is 5.
        static_length (int): Length of the static feature vector. Default is 2.
        num_series (int): Number of time series to generate. Default is 1.
        features_dim (int): Number of features for each time series. Default is 1.
    Methods:
        __len__():
            Returns the number of samples in the dataset, accounting for the past and future window lengths.
        __getitem__(idx):
            Retrieves a single sample from the dataset at the specified index.
            Args:
                idx (int): Index of the sample to retrieve.
            Returns:
                tuple: A tuple containing:
                    - past_inputs (dict): Dictionary with past data tensor under the key "past_data".
                                         Shape: (past_length, num_series, features_dim)
                    - future_inputs (dict): Dictionary with future data tensor under the key "future_data".
                                           Shape: (future_length, num_series, features_dim)
                    - static_inputs (dict): Dictionary with static data tensor under the key "static_data".
                                           Shape: (static_length,)
                    - target (torch.Tensor): Target tensor representing the future data.
                                            Shape: (future_length, num_series, features_dim)
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
        static_length: int = 2,
        num_series: int = 1,
        features_dim: int = 1,
    ) -> None:
        super().__init__()
        self.length = length
        self.trend_slope = trend_slope
        self.seasonality_amplitude = seasonality_amplitude
        self.seasonality_period = seasonality_period
        self.noise_level = noise_level
        self._past_length = past_length
        self._future_length = future_length
        self._static_length = static_length
        self._num_series = num_series
        self._features_dim = features_dim

        self.data = self._generate_data()

    def _generate_data(self) -> np.ndarray:
        """
        Generate synthetic time series data with trend, seasonality, and noise components.

        Returns:
            np.ndarray: Generated data with shape (length, num_series, features_dim)
        """
        # Initialize the scaler
        scalers = [StandardScaler() for _ in range(self._num_series)]

        # Create empty array to store data
        all_series_data = np.zeros((self.length, self._num_series, self._features_dim))

        # Generate each series
        for series_idx in range(self._num_series):
            for feature_idx in range(self._features_dim):
                # Generate a single time series
                data = []
                # Adjust parameters slightly for each series to create diversity
                ts_adjust = 1.0 + 0.1 * (series_idx / max(1, self._num_series))
                feat_adjust = 1.0 + 0.05 * (feature_idx / max(1, self._features_dim))

                for i in range(self.length):
                    # Trend component
                    trend = self.trend_slope * ts_adjust * i

                    # Seasonality component with phase shift for diversity
                    phase_shift = (
                        0.5 * np.pi * (series_idx / max(1, self._num_series - 1))
                        if self._num_series > 1
                        else 0
                    )
                    seasonality = (
                        self.seasonality_amplitude
                        * feat_adjust
                        * np.sin(2 * np.pi * i / self.seasonality_period + phase_shift)
                    )

                    # Noise component
                    noise = np.random.normal(0, self.noise_level)

                    # Combine components
                    value = trend + seasonality + noise
                    data.append(value)

                # Scale the data
                data_array = np.array(data).reshape(-1, 1)
                scaled_data = scalers[series_idx].fit_transform(data_array)

                # Store in the main array
                all_series_data[:, series_idx, feature_idx] = scaled_data.flatten()

        # Store the scalers for later use
        self.scalers = scalers

        return all_series_data

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
        return self._features_dim

    @property
    def static_length(self) -> int:
        return self._static_length

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve

        Returns:
            tuple: A tuple containing:
                - past_inputs (dict): Dictionary with past data tensor under the key "past_data".
                                     Shape: (past_length, num_series, features_dim)
                - future_inputs (dict): Dictionary with future data tensor under the key "future_data".
                                       Shape: (future_length, num_series, features_dim)
                - static_inputs (dict): Dictionary with static data tensor under the key "static_data".
                                       Shape: (static_length,)
                - target (torch.Tensor): Target tensor representing the future data.
                                        Shape: (future_length, num_series, features_dim)
        """
        # Extract the past and future window data
        past_data = self.data[idx : idx + self._past_length]
        future_data = self.data[
            idx + self._past_length : idx + self._past_length + self._future_length
        ]
        target = future_data.copy()  # Create a copy to avoid reference issues

        # Generate static data
        static_data = np.random.rand(self._static_length)

        # Convert to tensors
        past_data = torch.tensor(
            past_data
        ).float()  # Shape: (past_length, num_series, features_dim)
        future_data = torch.tensor(
            future_data
        ).float()  # Shape: (future_length, num_series, features_dim)
        static_data = torch.tensor(static_data).float()  # Shape: (static_length,)
        target = torch.tensor(
            target
        ).float()  # Shape: (future_length, num_series, features_dim)

        # Create dictionaries for the inputs
        past_inputs = {"past_data": past_data}
        future_inputs = {"future_data": future_data}
        static_inputs = {"static_data": static_data}

        return past_inputs, future_inputs, static_inputs, target


class SimpleSyntheticDataModule(L.LightningDataModule):
    """
    SimpleSyntheticDataModule is a PyTorch Lightning DataModule designed to handle synthetic datasets
    for time series forecasting tasks. It provides train, validation, test, and prediction dataloaders
    with configurable dataset lengths, batch sizes, and data generation parameters.
    Attributes:
        train_length (int): Number of samples in the training dataset.
        val_length (int): Number of samples in the validation dataset.
        test_length (int): Number of samples in the test dataset.
        batch_size (int): Batch size for the dataloaders.
        trend_slope (float): Slope of the trend component in the synthetic data. Default is 0.1.
        seasonality_amplitude (float): Amplitude of the seasonality component in the synthetic data. Default is 1.0.
        seasonality_period (int): Period of the seasonality component in the synthetic data. Default is 10.
        noise_level (float): Standard deviation of the noise component in the synthetic data. Default is 0.1.
        past_length (int): Number of past time steps to include in the input sequence. Default is 10.
        future_length (int): Number of future time steps to predict. Default is 5.
        static_length (int): Number of static features to include in the dataset. Default is 2.
        num_series (int): Number of time series to generate. Default is 1.
        features_dim (int): Number of features for each time series. Default is 1.
        workers (int): Number of worker threads for data loading. Default is 1.
    Methods:
        setup(stage=None):
            Sets up the train, validation, and test datasets using the specified parameters.
        train_dataloader():
            Returns a DataLoader for the training dataset.
        val_dataloader():
            Returns a DataLoader for the validation dataset.
        test_dataloader():
            Returns a DataLoader for the test dataset.
        predict_dataloader():
            Returns a DataLoader for the prediction dataset (same as the test dataset).
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
        static_length: int = 2,
        num_series: int = 1,
        features_dim: int = 1,
        workers: int = 1,
    ):
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
        self.static_length = static_length
        self.num_series = num_series
        self.features_dim = features_dim
        self.workers = workers

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the datasets for training, validation, and testing.

        Args:
            stage: Optional stage parameter for different dataset splits
        """
        self.train_dataset = SimpleSyntheticDataset(
            length=self.train_length,
            trend_slope=self.trend_slope,
            seasonality_amplitude=self.seasonality_amplitude,
            seasonality_period=self.seasonality_period,
            noise_level=self.noise_level,
            past_length=self.past_length,
            future_length=self.future_length,
            static_length=self.static_length,
            num_series=self.num_series,
            features_dim=self.features_dim,
        )

        self.test_dataset = SimpleSyntheticDataset(
            length=self.test_length,
            trend_slope=self.trend_slope,
            seasonality_amplitude=self.seasonality_amplitude,
            seasonality_period=self.seasonality_period,
            noise_level=self.noise_level,
            past_length=self.past_length,
            future_length=self.future_length,
            static_length=self.static_length,
            num_series=self.num_series,
            features_dim=self.features_dim,
        )

        self.val_dataset = SimpleSyntheticDataset(
            length=self.val_length,
            trend_slope=self.trend_slope,
            seasonality_amplitude=self.seasonality_amplitude,
            seasonality_period=self.seasonality_period,
            noise_level=self.noise_level,
            past_length=self.past_length,
            future_length=self.future_length,
            static_length=self.static_length,
            num_series=self.num_series,
            features_dim=self.features_dim,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the DataLoader for training data"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the DataLoader for validation data"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the DataLoader for test data"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Returns the DataLoader for prediction data"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
