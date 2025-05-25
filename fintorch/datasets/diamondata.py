import os
from typing import Any, Dict, Optional, Tuple

import lightning as L
import pandas as pd
import requests  # For downloading data
import torch
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch.utils.data import DataLoader, Dataset, random_split

from fintorch.datasets.base import TimeSeriesDataset


# --- Data Loading Classes ---
def _download_data(url: str, local_path: str) -> None:
    """Downloads data from URL to local_path if not already present."""
    if not os.path.exists(local_path):
        print(f"Downloading data from {url} to {local_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data from {url}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise
        except Exception as e:
            print(f"An error occurred during download: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise
    else:
        print(f"Data file already exists at {local_path}. Skipping download.")


class DiamondDataset(TimeSeriesDataset):
    """
    PyTorch Dataset for loading and preprocessing time-series data.
    Downloads data from a URL if the local file doesn't exist.
    Returns data in a dictionary format compatible with specific models.

    Returns data in the following format:
    - past_inputs (dict): Dictionary with past data tensor under the key "past_data".
                          Shape: (time_step, series_num, 1)
    - future_inputs (dict): Dictionary with future data tensor under the key "future_data".
                           Shape: (output_window, series_num, 1)
    - static_inputs (dict): Dictionary with static data set to None under the key "static_data".
    - target (torch.Tensor): Target tensor representing the future data.
                            Shape: (output_window, series_num, 1)
    """

    def __init__(
        self,
        local_path: str,
        time_step: int,
        output_window: int,
        static_length: int = 2,
    ):
        """
        Initializes the DiamondDataset.

        Args:
            local_path (str): The local path to save/cache the downloaded file.
            time_step (int): Length of the input sequence (past data).
            output_window (int): Length of the target sequence (future data).
            static_length (int): The dimension of the dummy static features to generate. Default is 2.
        """
        super().__init__()
        self.local_path = local_path
        self._time_step = time_step
        self._output_window = output_window
        self._static_length = static_length  # Store static feature length

        # --- Load Data ---
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(
                f"Data file not found at {self.local_path} after download attempt."
            )

        try:
            data_df = pd.read_csv(self.local_path)
            if pd.api.types.is_numeric_dtype(data_df.iloc[:, 0]):
                raw_data = data_df.values.astype("float32")
            else:
                raw_data = data_df.iloc[:, 1:].values.astype("float32")
        except pd.errors.EmptyDataError:
            print(f"Error: The downloaded file at {self.local_path} is empty.")
            raise
        except Exception as e:
            print(f"Error loading or processing data from {self.local_path}: {e}")
            raise

        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, 1)
        self._series_num = raw_data.shape[1]
        self._features_dim = 1  # Set features dimension to 1

        # --- Scale Data ---
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(raw_data)

        # --- Construct Input Sample Indices (data is retrieved in __getitem__) ---
        # Store indices instead of pre-processed samples to save memory
        # and allow generating static features per-item
        self.indices = []
        for i in range(self._time_step, len(self.data) + 1):
            if i >= self._output_window:
                # Check if both past and future windows fit within bounds
                if (i - self._time_step >= 0) and (i - self._output_window >= 0):
                    self.indices.append(i)  # Store the end index 'i'
            else:
                pass

        if not self.indices:
            print(
                f"Warning: No valid sample indices generated. Check time_step ({self._time_step}), "
                f"output_window ({self._output_window}), and data length ({len(self.data)})."
            )

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.indices)

    @property
    def time_steps(self) -> int:
        """
        Returns the number of past time steps in the input sequence.

        Returns:
            int: The number of past time steps.
        """
        return self._time_step

    @property
    def future_steps(self) -> int:
        """
        Returns the number of future time steps to predict.

        Returns:
            int: The number of future time steps.
        """
        return self._output_window

    @property
    def series_dim(self) -> int:
        """
        Returns the number of different time series in the dataset.

        Returns:
            int: The number of time series.
        """
        return self._series_num

    @property
    def features_dim(self) -> int:
        """
        Returns the number of features for each time series.

        Returns:
            int: The number of features.
        """
        return self._features_dim

    @property
    def static_length(self) -> int:
        """
        Returns the number of static features.

        Returns:
            int: The number of static features.
        """
        return self._static_length

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Optional[torch.Tensor]],
        torch.Tensor,
    ]:
        """
        Retrieves a single sample from the dataset in the specified dictionary format.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - past_inputs (dict): Dictionary with past data tensor under the key "past_data".
                                      Shape: (time_step, series_num, 1)
                - future_inputs (dict): Dictionary with future data tensor under the key "future_data".
                                        Shape: (output_window, series_num, 1)
                - static_inputs (dict): Dictionary with static data set to None under the key "static_data".
                - target (torch.Tensor): Target tensor representing the future data.
                                         Shape: (output_window, series_num, 1)
        """
        # Get the end index for the current sample
        end_idx = self.indices[idx]

        # Extract past and future (target) data sequences using the end index
        past_data_np = self.data[end_idx - self._time_step : end_idx]
        # Target is defined as the sequence ending at the current index, matching original snippet
        target_np = self.data[end_idx - self._output_window : end_idx]

        # Reshape numpy arrays to add feature dimension (time_steps, series_num, features=1)
        past_data_np = past_data_np.reshape(self._time_step, self._series_num, 1)
        target_np = target_np.reshape(self._output_window, self._series_num, 1)

        # Set static data to None
        static_data = None

        # Convert to tensors
        past_data = torch.from_numpy(
            past_data_np
        ).float()  # Shape: (time_step, series_num, 1)
        target = torch.from_numpy(
            target_np
        ).float()  # Shape: (output_window, series_num, 1)

        # Create dictionaries matching the required structure
        past_inputs = {"past_data": past_data}
        future_inputs = {"future_data": target}
        static_inputs = {"static_data": static_data}

        # Return in the specified tuple format - target already has shape (output_window, series_num, 1)
        return past_inputs, future_inputs, static_inputs, target


class DiamondDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for the DiamondDataset. Handles download and
    provides data in the specified dictionary format.

    Returns data in the following format:
    - past_inputs (dict): Dictionary with past data tensor under the key "past_data".
                          Shape: (time_step, series_num, 1)
    - future_inputs (dict): Dictionary with future data tensor under the key "future_data".
                           Shape: (output_window, series_num, 1)
    - static_inputs (dict): Dictionary with static data set to None under the key "static_data".
    - target (torch.Tensor): Target tensor representing the future data.
                            Shape: (output_window, series_num, 1)
    """

    def __init__(
        self,
        local_path: str,
        time_step: int,
        output_window: int,
        static_length: int = 2,  # Added static_length
        batch_size: int = 128,
        num_workers: int = 0,
        train_split: float = 0.7,
        val_split: float = 0.15,
    ):
        super().__init__()
        # Store hyperparameters as instance variables
        self.local_path = local_path
        self.time_step = time_step
        self.output_window = output_window
        self.static_length = static_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split

        # Save all hyperparameters
        self.save_hyperparameters()

        # Validate splits
        self.test_split = 1.0 - self.train_split - self.val_split
        if not (
            0 < self.train_split < 1
            and 0 < self.val_split < 1
            and 0 < self.test_split < 1
        ):
            raise ValueError(
                "Train, validation, and test splits must be between 0 and 1 and sum to 1."
            )
        if self.train_split + self.val_split >= 1.0:
            raise ValueError("Sum of train_split and val_split must be less than 1.")

        # Initialize dataset placeholders
        self.dataset: Optional[DiamondDataset] = None
        self.train_dataset: Optional[Dataset] = None  # type: ignore
        self.val_dataset: Optional[Dataset] = None  # type: ignore
        self.test_dataset: Optional[Dataset] = None  # type: ignore
        self.feature_dim: Optional[int] = None  # This might represent series_num now
        self.output_dim: Optional[int] = None  # This might represent series_num now
        self.series_num: Optional[int] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Loads data, creates dataset instance, and splits it."""
        if not self.dataset:
            try:
                if not os.path.exists(self.local_path):
                    print(
                        f"Data not found locally at {self.local_path}, attempting download in setup..."
                    )

                self.dataset = DiamondDataset(
                    local_path=self.local_path,
                    time_step=self.time_step,
                    output_window=self.output_window,
                    static_length=self.static_length,  # Pass static_length
                )
                # Store dimensions derived from the dataset
                self.series_num = self.dataset.series_dim
                # Feature/Output dim usually refers to the last dimension of the sequence data
                self.feature_dim = 1  # Based on reshape in dataset
                self.output_dim = 1  # Based on reshape in dataset

                total_len = len(self.dataset)
                train_len = int(total_len * self.train_split)
                val_len = int(total_len * self.val_split)
                test_len = total_len - train_len - val_len

                if train_len <= 0 or val_len <= 0 or test_len <= 0:
                    raise ValueError(
                        f"Calculated split lengths are too small or zero "
                        f"(train: {train_len}, val: {val_len}, test: {test_len}). "
                        f"Dataset size ({total_len}) might be too small for splits or data is incompatible."
                    )

                generator = torch.Generator().manual_seed(42)
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    self.dataset, [train_len, val_len, test_len], generator=generator
                )

            except FileNotFoundError as e:
                print(f"Error during setup: {e}. Cannot create DataModule.")
                self.train_dataset, self.val_dataset, self.test_dataset = (
                    None,
                    None,
                    None,
                )
                raise
            except ValueError as ve:
                print(f"Error during dataset splitting or processing: {ve}")
                self.train_dataset, self.val_dataset, self.test_dataset = (
                    None,
                    None,
                    None,
                )
                raise
            except Exception as e:
                print(f"An unexpected error occurred during setup: {e}")
                self.train_dataset, self.val_dataset, self.test_dataset = (
                    None,
                    None,
                    None,
                )
                raise

    # --- Dataloader methods ---
    def train_dataloader(self) -> DataLoader[Any]:
        if not self.train_dataset:
            raise RuntimeError(
                "Train dataset not available. Run setup() first or setup failed."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if not self.val_dataset:
            raise RuntimeError(
                "Validation dataset not available. Run setup() first or setup failed."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        if not self.test_dataset:
            raise RuntimeError(
                "Test dataset not available. Run setup() first or setup failed."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        if not self.test_dataset:
            raise RuntimeError(
                "Test dataset not available for prediction. Run setup() first or setup failed."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
