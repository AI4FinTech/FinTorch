from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset[Dict[str, torch.Tensor]], ABC):
    """
    Base abstract class for time series datasets in FinTorch.

    This class defines a standard interface for time series datasets to ensure
    consistent shapes and return types across different dataset implementations.

    All time series datasets in FinTorch should inherit from this class and
    implement the __getitem__ and __len__ methods according to the specified format.

    The standard format for time series data in FinTorch returns a single dictionary
    with the following structure per sample:

    {
        # --- Core Historical Data ---
        "past_target": torch.Tensor,  # Shape: (past_time_steps, series_dim, num_target_features)
        "past_covariates_known_future": torch.Tensor, # Shape: (past_time_steps, series_dim, num_known_future_cov_features)
        "past_covariates_unknown_future": torch.Tensor, # Shape: (past_time_steps, series_dim, num_unknown_future_cov_features)

        # --- Core Future Data (for decoders/targets) ---
        "future_covariates_known": torch.Tensor, # Shape: (future_time_steps, series_dim, num_known_future_cov_features)
        "output_target": torch.Tensor, # Shape: (future_time_steps, series_dim, num_target_features) # This is the label

        # --- Static Features (per series) ---
        "static_features_real": torch.Tensor, # Shape: (series_dim, num_static_real_features)
        "static_features_categorical": torch.Tensor, # Shape: (series_dim, num_static_categorical_features), dtype=torch.long

        # --- OPTIONAL ENHANCEMENTS ---
        "past_time_features": torch.Tensor, # Shape: (past_time_steps, num_time_features) OR (past_time_steps, series_dim, num_time_features)
        "future_time_features": torch.Tensor, # Shape: (future_time_steps, num_time_features) OR (future_time_steps, series_dim, num_time_features)
        "past_target_mask": torch.Tensor, # Shape: (past_time_steps, series_dim, num_target_features) or (past_time_steps, series_dim)
    }

    When batched by DataLoader, an initial batch_size dimension will be added to all tensors.
    For example, past_target will become (batch_size, past_time_steps, series_dim, num_target_features).

    Where:
    - past_time_steps: Number of past time steps in the input sequence
    - future_time_steps: Number of future time steps to predict
    - series_dim: Number of different time series in the dataset
    - num_target_features: Number of target features for each time series (often 1)
    - num_known_future_cov_features: Number of covariates with known future values
    - num_unknown_future_cov_features: Number of covariates without known future values
    - num_static_real_features: Number of real-valued static features
    - num_static_categorical_features: Number of categorical static features
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
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing tensors with the following keys:
                - "past_target": Historical target values
                - "past_covariates_known_future": Historical covariates with known future values
                - "past_covariates_unknown_future": Historical covariates without known future values
                - "future_covariates_known": Future covariate values (known)
                - "output_target": Target values to predict
                - "static_features_real": Real-valued static features
                - "static_features_categorical": Categorical static features

                Optional keys may include:
                - "past_time_features": Time-based features for historical period
                - "future_time_features": Time-based features for future period
                - "past_target_mask": Mask for missing values in past targets
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
        Returns the total number of features across all feature types.

        Note: This property is maintained for backward compatibility.
        For new implementations, consider using the more specific feature dimension properties.

        Returns:
            int: The total number of features.
        """
        pass

    @property
    @abstractmethod
    def static_length(self) -> int:
        """
        Returns the total number of static features (real + categorical).

        Returns:
            int: The total number of static features.
        """
        pass

    @property
    @abstractmethod
    def static_categorical_cardinalities(self) -> List[int]:
        """
        Returns a list of cardinalities for each static categorical feature.

        This is crucial for setting up embedding layers in models that use
        categorical features.

        Returns:
            List[int]: A list where each element represents the number of unique
                      categories for the corresponding categorical feature.
                      Empty list if no categorical features are present.
        """
        pass

    @property
    @abstractmethod
    def num_target_features(self) -> int:
        """
        Returns the number of target features.

        Returns:
            int: The number of target features (often 1 for univariate forecasting).
        """
        pass

    @property
    @abstractmethod
    def num_known_future_cov_features(self) -> int:
        """
        Returns the number of covariates with known future values.

        Returns:
            int: The number of known future covariate features.
        """
        pass

    @property
    @abstractmethod
    def num_unknown_future_cov_features(self) -> int:
        """
        Returns the number of covariates without known future values.

        Returns:
            int: The number of unknown future covariate features.
        """
        pass

    @property
    @abstractmethod
    def num_static_real_features(self) -> int:
        """
        Returns the number of real-valued static features.

        Returns:
            int: The number of real-valued static features.
        """
        pass

    @property
    @abstractmethod
    def num_static_categorical_features(self) -> int:
        """
        Returns the number of categorical static features.

        Returns:
            int: The number of categorical static features.
        """
        pass
