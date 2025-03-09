import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SimpleSyntheticDataset(Dataset):
    def __init__(
        self,
        length,
        trend_slope=0.1,
        seasonality_amplitude=1.0,
        seasonality_period=10,
        noise_level=0.1,
        past_length=10,
        future_length=5,
        static_length=2,
    ):
        self.length = length
        self.trend_slope = trend_slope
        self.seasonality_amplitude = seasonality_amplitude
        self.seasonality_period = seasonality_period
        self.noise_level = noise_level
        self.past_length = past_length
        self.future_length = future_length
        self.static_length = static_length

        self.data = self._generate_data()

    def _generate_data(self):
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

    def __len__(self):
        return self.length - self.past_length - self.future_length

    def __getitem__(self, idx):
        past_data = self.data[idx : idx + self.past_length]
        future_data = self.data[
            idx + self.past_length : idx + self.past_length + self.future_length
        ]
        target = future_data

        # Generate static data
        static_data = np.random.rand(self.static_length)

        # Convert to tensors
        past_data = torch.tensor(past_data).float().unsqueeze(-1)
        future_data = torch.tensor(future_data).float()

        static_data = torch.tensor(static_data).float()
        target = torch.tensor(target).float()

        # Create a dictionary for past, future, and static data
        past_inputs = {"past_data": past_data}
        future_inputs = {"future_data": future_data.unsqueeze(-1)}
        static_inputs = {"static_data": static_data}

        return past_inputs, future_inputs, static_inputs, target


class SimpleSyntheticDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_length,
        val_length,
        test_length,
        batch_size,
        trend_slope=0.1,
        seasonality_amplitude=1.0,
        seasonality_period=10,
        noise_level=0.1,
        past_length=10,
        future_length=5,
        static_length=2,
        workers=1,
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

    def setup(self, stage=None):
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
