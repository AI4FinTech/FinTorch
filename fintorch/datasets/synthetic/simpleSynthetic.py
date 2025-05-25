from typing import Any, Dict, Optional, Tuple

import lightning as L
import numpy as np
import numpy.typing as npt
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
        scalers (List[List[StandardScaler]]): 2D list of scalers, one per (series, feature) pair.
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

    def _generate_data(self) -> npt.NDArray[np.float64]:
        """
        Generate synthetic time series data with trend, seasonality, and noise components.

        Returns:
            np.ndarray: Generated data with shape (length, num_series, features_dim)
        """
        # Initialize scalers as a 2D list: one scaler per (series, feature) pair
        scalers = [
            [StandardScaler() for _ in range(self._features_dim)]
            for _ in range(self._num_series)
        ]

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

                # Scale the data using the correct scaler for this (series, feature) pair
                data_array = np.array(data).reshape(-1, 1)
                scaled_data = scalers[series_idx][feature_idx].fit_transform(data_array)

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
    def past_length(self) -> int:
        return self._past_length

    @property
    def future_length(self) -> int:
        return self._future_length

    @property
    def static_length(self) -> int:
        return self._static_length

    def get_scaler(self, series_idx: int, feature_idx: int) -> StandardScaler:
        """
        Get the scaler for a specific (series, feature) pair.

        Args:
            series_idx (int): Index of the series
            feature_idx (int): Index of the feature

        Returns:
            StandardScaler: The scaler fitted for the specified series and feature
        """
        if series_idx >= self._num_series or feature_idx >= self._features_dim:
            raise IndexError(
                f"Invalid indices: series_idx={series_idx} (max: {self._num_series - 1}), "
                f"feature_idx={feature_idx} (max: {self._features_dim - 1})"
            )
        return self.scalers[series_idx][feature_idx]

    def inverse_transform(
        self, data: npt.NDArray[np.float64], series_idx: int, feature_idx: int
    ) -> npt.NDArray[np.float64]:
        """
        Apply inverse transformation to scaled data using the appropriate scaler.

        Args:
            data (np.ndarray): Scaled data to inverse transform
            series_idx (int): Index of the series
            feature_idx (int): Index of the feature

        Returns:
            npt.NDArray[np.float64]: Inverse transformed data
        """
        scaler = self.get_scaler(series_idx, feature_idx)
        result = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return result.astype(np.float64)  # type: ignore

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
        past_data_np = self.data[idx : idx + self._past_length]
        future_data_np = self.data[
            idx + self._past_length : idx + self._past_length + self._future_length
        ]
        target_np = future_data_np.copy()  # Create a copy to avoid reference issues

        # Generate static data
        static_data_np: npt.NDArray[np.float64] = np.random.rand(self._static_length)

        # Convert to tensors and squeeze to remove singleton dimensions when num_series=1 and features_dim=1
        past_data = torch.tensor(past_data_np).float()
        future_data = torch.tensor(future_data_np).float()
        static_data = torch.tensor(static_data_np).float()
        target = torch.tensor(target_np).float()

        # Squeeze singleton dimensions if num_series=1 and features_dim=1
        if self._num_series == 1 and self._features_dim == 1:
            past_data = past_data.squeeze(-1).squeeze(-1)  # Shape: (past_length,)
            future_data = future_data.squeeze(-1).squeeze(-1)  # Shape: (future_length,)
            target = target.squeeze(-1).squeeze(-1)  # Shape: (future_length,)

            # Reshape to expected format for tests
            past_data = past_data.unsqueeze(-1)  # Shape: (past_length, 1)
            future_data = future_data.unsqueeze(-1)  # Shape: (future_length, 1)
            target = target  # Shape: (future_length,)

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
