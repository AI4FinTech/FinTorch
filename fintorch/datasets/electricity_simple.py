from typing import Any, Dict, Optional, Tuple

import lightning as L
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch.utils.data import DataLoader, Dataset


class ElectricityDataset(Dataset):  # type: ignore
    def __init__(
        self,
        past_length: int = 10,
        future_length: int = 5,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> None:
        self.past_length = past_length
        self.future_length = future_length

        self.data = self._get_data()

        # Define dataset length based on indices
        self.start_idx = start_idx
        self.end_idx = (
            end_idx
            if end_idx is not None
            else len(self.data) - past_length - future_length
        )
        self.length = self.end_idx - self.start_idx

    def _get_data(self) -> Any:
        data = pl.read_csv(
            "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/refs/heads/master/data/PJMW_hourly.csv"
        )

        self.length = data.shape[0]

        print(f"Electricity dataset size: {data.shape}")

        # Initialize the scaler
        scaler = StandardScaler()

        # Fit the scaler on the data and transform it
        data_scaled = scaler.fit_transform(
            data.select(pl.col(data.columns[1])).to_numpy().reshape(-1, 1)
        )

        # Store the scaler for later use
        self.scaler = scaler

        data_scaled = data_scaled.flatten()

        return data_scaled

    def __len__(self) -> int:
        return self.length - self.past_length - self.future_length

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(idx, slice):
            return [
                self[i]
                for i in range(idx.start or 0, idx.stop or len(self), idx.step or 1)
            ]

        # Adjust index to be within the dataset slice
        idx = self.start_idx + idx

        past_data = self.data[idx : idx + self.past_length]
        target = self.data[
            idx + self.past_length : idx + self.past_length + self.future_length
        ]

        # Convert to tensors
        past_data = torch.tensor(past_data).float().unsqueeze(-1)
        target = torch.tensor(target).float()

        # Create a dictionary for past, future, and static data
        past_inputs = {"past_data": past_data}

        return past_inputs, target


class ElectricityDataModule(L.LightningDataModule):
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
        dataset = ElectricityDataset(
            past_length=self.past_length,
            future_length=self.future_length,
        )

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))

        # Create separate dataset instances to preserve time-series order
        self.train_dataset = ElectricityDataset(
            past_length=self.past_length,
            future_length=self.future_length,
            start_idx=0,
            end_idx=train_size,
        )

        self.val_dataset = ElectricityDataset(
            past_length=self.past_length,
            future_length=self.future_length,
            start_idx=train_size,
            end_idx=train_size + val_size,
        )

        self.test_dataset = ElectricityDataset(
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
