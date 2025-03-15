import lightning as L
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


class ElectricityDataset(Dataset):
    def __init__(
        self,
        past_length=10,
        future_length=5,
        static_length=2,
    ):
        self.past_length = past_length
        self.future_length = future_length
        self.static_length = static_length

        self.data = self._get_data()

    def _get_data(self):
        data = pl.read_csv(
            "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJM_Load_hourly.csv"
        )

        self.length = data.shape[0]

        print(
            f"Electricity data:{data.select(pl.col(data.columns[1])).to_numpy().squeeze()}"
        )

        return data.select(pl.col(data.columns[1])).to_numpy().squeeze()

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


class ElectricityDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_length,
        val_length,
        test_length,
        batch_size,
        past_length=10,
        future_length=5,
        static_length=2,
        workers=1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.past_length = past_length
        self.future_length = future_length
        self.static_length = static_length
        self.workers = workers

    def setup(self, stage=None):
        self.dataset = ElectricityDataset(
            past_length=self.past_length,
            future_length=self.future_length,
            static_length=self.static_length,
        )

        # TODO: properly split
        self.train_dataset = self.dataset
        self.test_dataset = self.dataset
        self.val_dataset = self.dataset

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
