from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(
    Dataset[
        Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            torch.Tensor,
        ]
    ],
    ABC,
):
    """
    Base abstract class for time series datasets in FinTorch.

    This class defines a standard interface for time series datasets to ensure
    consistent shapes and return types across different dataset implementations.

    All time series datasets in FinTorch should inherit from this class and
    implement the __getitem__ and __len__ methods according to the specified format.

    The standard format for time series data in FinTorch is:
    - past_data: (time_steps, series_dim, features_dim)
    - future_data: (future_steps, series_dim, features_dim)
    - static_data: (static_length,)
    - target: (future_steps, series_dim, features_dim)

    Where:
    - time_steps: Number of past time steps in the input sequence
    - future_steps: Number of future time steps to predict
    - series_dim: Number of different time series in the dataset
    - features_dim: Number of features for each time series
    - static_length: Number of static features
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - past_inputs (dict): Dictionary with past data tensor under the key "past_data".
                                     Shape: (time_steps, series_dim, features_dim)
                - future_inputs (dict): Dictionary with future data tensor under the key "future_data".
                                       Shape: (future_steps, series_dim, features_dim)
                - static_inputs (dict): Dictionary with static data tensor under the key "static_data".
                                       Shape: (static_length,)
                - target (torch.Tensor): Target tensor representing the future data.
                                        Shape: (future_steps, series_dim, features_dim)
        """
        pass

    @property
    @abstractmethod
    def time_steps(self) -> int:
        """
        Returns the number of past time steps in the input sequence.

        Returns:
            int: The number of past time steps.
        """
        pass

    @property
    @abstractmethod
    def future_steps(self) -> int:
        """
        Returns the number of future time steps to predict.

        Returns:
            int: The number of future time steps.
        """
        pass

    @property
    @abstractmethod
    def series_dim(self) -> int:
        """
        Returns the number of different time series in the dataset.

        Returns:
            int: The number of time series.
        """
        pass

    @property
    @abstractmethod
    def features_dim(self) -> int:
        """
        Returns the number of features for each time series.

        Returns:
            int: The number of features.
        """
        pass

    @property
    @abstractmethod
    def static_length(self) -> int:
        """
        Returns the number of static features.

        Returns:
            int: The number of static features.
        """
        pass
