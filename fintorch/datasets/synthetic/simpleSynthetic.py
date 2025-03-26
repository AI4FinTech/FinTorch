from typing import Any, List, Optional

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SimpleSyntheticDataset(Dataset):  # type: ignore
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
                    - future_inputs (dict): Dictionary with future data tensor under the key "future_data".
                    - static_inputs (dict): Dictionary with static data tensor under the key "static_data".
                    - target (torch.Tensor): Target tensor representing the future data.
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
    ) -> None:
        self.length = length
        self.trend_slope = trend_slope
        self.seasonality_amplitude = seasonality_amplitude
        self.seasonality_period = seasonality_period
        self.noise_level = noise_level
        self.past_length = past_length
        self.future_length = future_length
        self.static_length = static_length

        self.data = self._generate_data()

    def _generate_data(self) -> List[float]:
        data = []
        for i in range(self.length):
            # Trend component
            trend = self.trend_slope * i

            # Seasonality component
            seasonality = self.seasonality_amplitude * np.sin(
                2 * np.pi * i / self.seasonality_period
            )

            # Noise component
            noise = np.random.normal(0, self.noise_level)

            # Combine components
            value = trend + seasonality + noise
            data.append(value)
        return data

    def __len__(self) -> int:
        return self.length - self.past_length - self.future_length

    def __getitem__(self, idx: int) -> Any:
        past_data = self.data[idx : idx + self.past_length]
        future_data = self.data[
            idx + self.past_length : idx + self.past_length + self.future_length
        ]
        target = future_data

        # Generate static data
        static_data = np.random.rand(self.static_length)

        # Convert to tensors
        past_data = torch.tensor(past_data).float().unsqueeze(-1)  # type: ignore
        future_data = torch.tensor(future_data).float()  # type: ignore

        static_data = torch.tensor(static_data).float()  # type: ignore
        target = torch.tensor(target).float()  # type: ignore

        # Create a dictionary for past, future, and static data
        past_inputs = {"past_data": past_data}
        future_inputs = {"future_data": future_data.unsqueeze(-1)}  # type: ignore
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
        self.workers = workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SimpleSyntheticDataset(
            length=self.train_length,
            trend_slope=self.trend_slope,
            seasonality_amplitude=self.seasonality_amplitude,
            seasonality_period=self.seasonality_period,
            noise_level=self.noise_level,
            past_length=self.past_length,
            future_length=self.future_length,
            static_length=self.static_length,
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
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
