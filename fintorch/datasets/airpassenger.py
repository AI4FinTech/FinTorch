from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch.utils.data import DataLoader, Dataset


def custom_collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]], torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]], torch.Tensor]:
    """
    Custom collate function to handle None values in static_data.
    """
    past_inputs_list = []
    future_inputs_list = []
    static_inputs_list = []
    targets_list = []

    for past_inputs, future_inputs, static_inputs, target in batch:
        past_inputs_list.append(past_inputs)
        future_inputs_list.append(future_inputs)
        static_inputs_list.append(static_inputs)
        targets_list.append(target)

    # Collate past_inputs
    past_data = torch.stack([item['past_data'] for item in past_inputs_list])
    collated_past_inputs = {'past_data': past_data}

    # Collate future_inputs
    future_data = torch.stack([item['future_data'] for item in future_inputs_list])
    collated_future_inputs = {'future_data': future_data}

    # Handle static_inputs (which contain None values)
    # Since static_data is None for all items, we keep it as None
    collated_static_inputs = {'static_data': None}

    # Collate targets
    targets = torch.stack(targets_list)

    return collated_past_inputs, collated_future_inputs, collated_static_inputs, targets


class AirPassengerDataset(Dataset):  # type: ignore
    """
    A PyTorch Dataset for the Air Passenger dataset.
    This dataset is designed to provide time-series data for training and testing
    machine learning models. It uses the "Airline Passengers" dataset, which contains
    monthly totals of international airline passengers from 1949 to 1960.
    The dataset is scaled using a `StandardScaler` from `sklearn` to normalize the data
    to have zero mean and unit variance.
    Attributes:
        past_length (int): The number of past time steps to include in the input.
        future_length (int): The number of future time steps to predict.
        start_idx (int): The starting index of the dataset slice.
        end_idx (int): The ending index of the dataset slice.
        length (int): The total number of samples in the dataset slice.
        data (numpy.ndarray): The scaled time-series data.
        scaler (StandardScaler): The scaler used to normalize the data.
    Methods:
        __len__(): Returns the total number of samples in the dataset slice.
        __getitem__(idx): Returns a single sample or a slice of samples from the dataset.
    Args:
        past_length (int, optional): The number of past time steps to include in the input. Default is 10.
        future_length (int, optional): The number of future time steps to predict. Default is 5.
        start_idx (int, optional): The starting index of the dataset slice. Default is 0.
        end_idx (int, optional): The ending index of the dataset slice. If None, it is set to the length of the data
                                 minus `past_length` and `future_length`. Default is None.
    Returns:
        tuple: A tuple containing:
            - past_inputs (dict): Dictionary with past data tensor under the key "past_data".
                                 Shape: (past_length, 1)
            - future_inputs (dict): Dictionary with future data tensor under the key "future_data".
                                   Shape: (future_length, 1)
            - static_inputs (dict): Dictionary with static data set to None under the key "static_data".
            - target (torch.Tensor): Target tensor representing the future data.
                                    Shape: (future_length,)

    Example:
        dataset = AirPassengerDataset(past_length=12, future_length=6)
        past_inputs, future_inputs, static_inputs, target = dataset[0]
        print(past_inputs["past_data"].shape)  # torch.Size([12, 1])
        print(future_inputs["future_data"].shape)  # torch.Size([6, 1])
        print(static_inputs["static_data"])  # None
        print(target.shape)  # torch.Size([6])
    """

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
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        )

        self.length = data.shape[0]

        print(f"Air Passenger dataset size: {data.shape}")

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
        return self.length

    def __getitem__(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Optional[torch.Tensor]],
        torch.Tensor
    ]:
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
        future_data = torch.tensor(target).float().unsqueeze(-1)
        target = torch.tensor(target).float()

        # Create dictionaries for past, future, and static data
        past_inputs = {"past_data": past_data}
        future_inputs = {"future_data": future_data}
        static_inputs = {"static_data": None}

        return past_inputs, future_inputs, static_inputs, target


class AirPassengerDataModule(L.LightningDataModule):
    """
    AirPassengerDataModule is a PyTorch Lightning DataModule designed for handling time-series data
    from the AirPassenger dataset. The dataset is assumed to be standard scaled, ensuring that
    features are normalized to have a mean of 0 and a standard deviation of 1.
    This DataModule splits the dataset into training, validation, and test sets while preserving
    the time-series order. It provides DataLoader objects for each split, which can be used
    directly in PyTorch Lightning training loops.
    Attributes:
        batch_size (int): The number of samples per batch for the DataLoader.
        past_length (int): The number of past time steps to consider as input.
        future_length (int): The number of future time steps to predict (horizon).
        workers (int): The number of worker threads for data loading.
    Methods:
        setup(stage=None):
            Prepares the dataset splits (train, validation, test) based on the AirPassengerDataset.
        train_dataloader():
            Returns a DataLoader for the training dataset.
        val_dataloader():
            Returns a DataLoader for the validation dataset.
        test_dataloader():
            Returns a DataLoader for the test dataset.
        predict_dataloader():
            Returns a DataLoader for the test dataset, used for predictions.
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
