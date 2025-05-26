Time Series Datasets Tutorial
=============================

This tutorial explains how to work with and create custom time series datasets in FinTorch using the new standardized dictionary format.

Quick Start
-----------

If you want to get started immediately with FinTorch time series datasets, here's a minimal example:

.. code-block:: python

    from fintorch.datasets.synthetic.simpleSynthetic import SimpleSyntheticDataset
    from torch.utils.data import DataLoader

    # Create a simple synthetic dataset
    dataset = SimpleSyntheticDataset(
        length=1000,        # 1000 time steps total
        past_length=24,     # Use 24 past steps for prediction
        future_length=6,    # Predict 6 steps ahead
        num_series=1        # Single time series
    )

    # Create a DataLoader for training
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Examine the data structure
    sample = dataset[0]
    print("Available data keys:", list(sample.keys()))
    print("Past target shape:", sample["past_target"].shape)      # (24, 1, 1)
    print("Future target shape:", sample["output_target"].shape)  # (6, 1, 1)

    # Ready to use with FinTorch models!
    for batch in dataloader:
        # batch contains all the standardized keys
        # Pass directly to TFT, CausalFormer, or other models
        break

Overview
--------

FinTorch's time series datasets follow a standardized format that provides clear separation of different feature types and native support for multi-series data. This design enables better model architecture decisions and more robust data handling.

The Standardized Data Structure
-------------------------------

Each call to ``Dataset.__getitem__`` returns a single dictionary containing PyTorch tensors with the following keys:

Core Data Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    {
        # --- Core Historical Data ---
        "past_target": torch.Tensor,                    # Shape: (past_time_steps, series_dim, num_target_features)
        "past_covariates_known_future": torch.Tensor,   # Shape: (past_time_steps, series_dim, num_known_future_cov_features)
        "past_covariates_unknown_future": torch.Tensor, # Shape: (past_time_steps, series_dim, num_unknown_future_cov_features)

        # --- Core Future Data (for decoders/targets) ---
        "future_covariates_known": torch.Tensor,        # Shape: (future_time_steps, series_dim, num_known_future_cov_features)
        "output_target": torch.Tensor,                  # Shape: (future_time_steps, series_dim, num_target_features)

        # --- Static Features (per series) ---
        "static_features_real": torch.Tensor,           # Shape: (series_dim, num_static_real_features)
        "static_features_categorical": torch.Tensor,    # Shape: (series_dim, num_static_categorical_features), dtype=torch.long

        # --- OPTIONAL ENHANCEMENTS ---
        "past_time_features": torch.Tensor,             # Shape: (past_time_steps, num_time_features)
        "future_time_features": torch.Tensor,           # Shape: (future_time_steps, num_time_features)
        "past_target_mask": torch.Tensor,               # Shape: (past_time_steps, series_dim, num_target_features)
    }

Feature Types Explained
~~~~~~~~~~~~~~~~~~~~~~~

**Target Features (``past_target``, ``output_target``)**
    The main variables you want to predict. For example, stock prices, sales volumes, or electricity consumption.

**Known Future Covariates (``past_covariates_known_future``, ``future_covariates_known``)**
    Features whose future values are known at prediction time. Examples include:

    - Calendar features (day of week, month, holidays)
    - Scheduled events (promotions, maintenance windows)
    - External schedules (weather forecasts, market hours)

**Unknown Future Covariates (``past_covariates_unknown_future``)**
    Features that influence the target but whose future values are unknown. Examples include:

    - Past weather conditions
    - Historical market indicators
    - Previous customer behavior

**Static Features (``static_features_real``, ``static_features_categorical``)**
    Time-invariant properties of each series. Examples include:

    - Product categories
    - Store locations
    - Customer demographics

Using Existing Datasets
-----------------------

SimpleSyntheticDataset
~~~~~~~~~~~~~~~~~~~~~

The ``SimpleSyntheticDataset`` generates synthetic time series data with configurable patterns:

.. code-block:: python

    from fintorch.datasets.synthetic.simpleSynthetic import SimpleSyntheticDataset
    from torch.utils.data import DataLoader

    # Create a dataset with multiple feature types
    dataset = SimpleSyntheticDataset(
        length=1000,                            # Total time series length
        past_length=24,                         # Historical window size
        future_length=6,                        # Prediction horizon
        num_series=3,                           # Number of time series
        num_target_features=1,                  # Number of target variables
        num_known_cov_features=4,               # Known future covariates (e.g., time features)
        num_unknown_cov_features=2,             # Unknown future covariates
        num_static_real_features=3,             # Real-valued static features
        num_static_categorical_features=2,      # Categorical static features
        static_categorical_cardinalities=[5, 10], # Categories per categorical feature
        trend_slope=0.1,
        seasonality_amplitude=1.0,
        noise_level=0.1
    )

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Examine a sample
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("Past target shape:", sample["past_target"].shape)
    print("Output target shape:", sample["output_target"].shape)

ElectricityDataset
~~~~~~~~~~~~~~~~~

The ``ElectricityDataset`` provides real-world electricity consumption data:

.. code-block:: python

    from fintorch.datasets.electricity_simple import ElectricityDataset

    dataset = ElectricityDataset(
        past_length=168,    # 1 week of hourly data
        future_length=24    # Predict next 24 hours
    )

    sample = dataset[0]
    # Target: electricity consumption
    # Known covariates: time features (hour, day of week, month)
    # Static features: placeholder values

Creating Custom Datasets
------------------------

To create a custom time series dataset, inherit from ``TimeSeriesDataset`` and implement the required methods:

Basic Template
~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import numpy as np
    from typing import Dict, List
    from fintorch.datasets.base import TimeSeriesDataset

    class CustomTimeSeriesDataset(TimeSeriesDataset):
        def __init__(self, data_path: str, past_length: int = 24, future_length: int = 6):
            super().__init__()
            self.past_length = past_length
            self.future_length = future_length

            # Load and preprocess your data
            self.data = self._load_data(data_path)

        def _load_data(self, data_path: str) -> Dict:
            """Load and preprocess your time series data."""
            # Implementation depends on your data format
            # Return a dictionary with processed arrays for each feature type
            pass

        def __len__(self) -> int:
            # Return the number of valid samples
            return len(self.data['target']) - self.past_length - self.future_length

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            """Return a sample in the standardized format."""
            # Extract time windows
            past_start = idx
            past_end = idx + self.past_length
            future_start = past_end
            future_end = future_start + self.future_length

            # Extract data for each feature type
            past_target = self.data['target'][past_start:past_end]
            output_target = self.data['target'][future_start:future_end]

            # Convert to tensors and return standardized format
            return {
                "past_target": torch.tensor(past_target, dtype=torch.float32),
                "past_covariates_known_future": torch.tensor(
                    self.data['known_cov'][past_start:past_end], dtype=torch.float32
                ),
                "past_covariates_unknown_future": torch.tensor(
                    self.data['unknown_cov'][past_start:past_end], dtype=torch.float32
                ),
                "future_covariates_known": torch.tensor(
                    self.data['known_cov'][future_start:future_end], dtype=torch.float32
                ),
                "output_target": torch.tensor(output_target, dtype=torch.float32),
                "static_features_real": torch.tensor(
                    self.data['static_real'], dtype=torch.float32
                ),
                "static_features_categorical": torch.tensor(
                    self.data['static_categorical'], dtype=torch.long
                ),
            }

        # Implement required abstract properties
        @property
        def time_steps(self) -> int:
            return self.past_length

        @property
        def future_steps(self) -> int:
            return self.future_length

        @property
        def series_dim(self) -> int:
            return self.data['target'].shape[1]  # Number of series

        @property
        def features_dim(self) -> int:
            # Total features for backward compatibility
            return (self.num_target_features +
                   self.num_known_future_cov_features +
                   self.num_unknown_future_cov_features)

        @property
        def static_length(self) -> int:
            return self.num_static_real_features + self.num_static_categorical_features

        @property
        def static_categorical_cardinalities(self) -> List[int]:
            # Return cardinalities for each categorical feature
            return [5, 10]  # Example: 2 categorical features with 5 and 10 categories

        @property
        def num_target_features(self) -> int:
            return self.data['target'].shape[2]

        @property
        def num_known_future_cov_features(self) -> int:
            return self.data['known_cov'].shape[2]

        @property
        def num_unknown_future_cov_features(self) -> int:
            return self.data['unknown_cov'].shape[2]

        @property
        def num_static_real_features(self) -> int:
            return self.data['static_real'].shape[1]

        @property
        def num_static_categorical_features(self) -> int:
            return self.data['static_categorical'].shape[1]

Complete Example: Stock Price Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example creating a stock price dataset:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from typing import Dict, List
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    class StockPriceDataset(TimeSeriesDataset):
        def __init__(self, csv_path: str, past_length: int = 30, future_length: int = 5):
            super().__init__()
            self.past_length = past_length
            self.future_length = future_length
            self.scalers = {}

            # Load and process the data
            self.data = self._load_and_process_data(csv_path)

        def _load_and_process_data(self, csv_path: str) -> Dict:
            """Load stock data and organize by feature type."""
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['stock_symbol', 'date'])

            # Group by stock symbol
            stocks = df['stock_symbol'].unique()
            num_stocks = len(stocks)

            # Prepare data arrays
            time_length = len(df) // num_stocks

            # Target: closing prices (shape: [time, num_stocks, 1])
            target_data = np.zeros((time_length, num_stocks, 1))

            # Known covariates: time features (shape: [time, num_stocks, 4])
            known_cov_data = np.zeros((time_length, num_stocks, 4))

            # Unknown covariates: technical indicators (shape: [time, num_stocks, 3])
            unknown_cov_data = np.zeros((time_length, num_stocks, 3))

            # Static features
            static_real_data = np.zeros((num_stocks, 2))      # Market cap, sector rating
            static_categorical_data = np.zeros((num_stocks, 1), dtype=int)  # Sector

            # Process each stock
            for i, stock in enumerate(stocks):
                stock_df = df[df['stock_symbol'] == stock].copy()

                # Target: normalize closing prices
                scaler = StandardScaler()
                target_data[:, i, 0] = scaler.fit_transform(
                    stock_df[['close']].values
                ).flatten()
                self.scalers[f'target_{stock}'] = scaler

                # Known covariates: time features
                stock_df['day_of_week'] = stock_df['date'].dt.dayofweek
                stock_df['month'] = stock_df['date'].dt.month
                stock_df['quarter'] = stock_df['date'].dt.quarter
                stock_df['is_month_end'] = stock_df['date'].dt.is_month_end.astype(int)

                time_features = stock_df[['day_of_week', 'month', 'quarter', 'is_month_end']].values
                scaler = StandardScaler()
                known_cov_data[:, i, :] = scaler.fit_transform(time_features)
                self.scalers[f'time_{stock}'] = scaler

                # Unknown covariates: technical indicators
                stock_df['sma_10'] = stock_df['close'].rolling(10).mean()
                stock_df['rsi'] = self._calculate_rsi(stock_df['close'])
                stock_df['volatility'] = stock_df['close'].rolling(10).std()

                tech_features = stock_df[['sma_10', 'rsi', 'volatility']].fillna(0).values
                scaler = StandardScaler()
                unknown_cov_data[:, i, :] = scaler.fit_transform(tech_features)
                self.scalers[f'tech_{stock}'] = scaler

                # Static features
                static_real_data[i, 0] = stock_df['market_cap'].iloc[0]  # Market cap
                static_real_data[i, 1] = stock_df['sector_rating'].iloc[0]  # Sector rating

                # Encode sector as categorical
                le = LabelEncoder()
                static_categorical_data[i, 0] = le.fit_transform([stock_df['sector'].iloc[0]])[0]

            # Normalize static real features
            static_scaler = StandardScaler()
            static_real_data = static_scaler.fit_transform(static_real_data)
            self.scalers['static'] = static_scaler

            return {
                'target': target_data,
                'known_cov': known_cov_data,
                'unknown_cov': unknown_cov_data,
                'static_real': static_real_data,
                'static_categorical': static_categorical_data
            }

        def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
            """Calculate Relative Strength Index."""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        def __len__(self) -> int:
            return self.data['target'].shape[0] - self.past_length - self.future_length

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            # Extract time windows
            past_start = idx
            past_end = idx + self.past_length
            future_start = past_end
            future_end = future_start + self.future_length

            return {
                "past_target": torch.tensor(
                    self.data['target'][past_start:past_end], dtype=torch.float32
                ),
                "past_covariates_known_future": torch.tensor(
                    self.data['known_cov'][past_start:past_end], dtype=torch.float32
                ),
                "past_covariates_unknown_future": torch.tensor(
                    self.data['unknown_cov'][past_start:past_end], dtype=torch.float32
                ),
                "future_covariates_known": torch.tensor(
                    self.data['known_cov'][future_start:future_end], dtype=torch.float32
                ),
                "output_target": torch.tensor(
                    self.data['target'][future_start:future_end], dtype=torch.float32
                ),
                "static_features_real": torch.tensor(
                    self.data['static_real'], dtype=torch.float32
                ),
                "static_features_categorical": torch.tensor(
                    self.data['static_categorical'], dtype=torch.long
                ),
            }

        # Implement all required properties...
        @property
        def time_steps(self) -> int:
            return self.past_length

        @property
        def future_steps(self) -> int:
            return self.future_length

        @property
        def series_dim(self) -> int:
            return self.data['target'].shape[1]

        @property
        def features_dim(self) -> int:
            return 1 + 4 + 3  # target + known_cov + unknown_cov

        @property
        def static_length(self) -> int:
            return 3  # 2 real + 1 categorical

        @property
        def static_categorical_cardinalities(self) -> List[int]:
            return [10]  # Assuming 10 different sectors

        @property
        def num_target_features(self) -> int:
            return 1

        @property
        def num_known_future_cov_features(self) -> int:
            return 4

        @property
        def num_unknown_future_cov_features(self) -> int:
            return 3

        @property
        def num_static_real_features(self) -> int:
            return 2

        @property
        def num_static_categorical_features(self) -> int:
            return 1

Multi-Series Considerations
--------------------------

Handling Multiple Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with multiple time series, consider:

1. **Alignment**: Ensure all series have the same time steps
2. **Missing Data**: Handle gaps consistently across series
3. **Scaling**: Apply scaling per series or globally depending on your use case
4. **Static Features**: Each series can have different static properties

.. code-block:: python

    # Example: Different scaling approaches

    # Per-series scaling (recommended for heterogeneous series)
    for i in range(num_series):
        series_data = all_data[:, i, :]
        scaler = StandardScaler()
        scaled_data[:, i, :] = scaler.fit_transform(series_data)

    # Global scaling (for homogeneous series)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_data.reshape(-1, num_features))
    scaled_data = scaled_data.reshape(time_steps, num_series, num_features)

Best Practices
--------------

Data Organization
~~~~~~~~~~~~~~~~

1. **Separate by Feature Type**: Clearly distinguish between target, covariates, and static features
2. **Consistent Shapes**: Ensure tensor shapes follow the standard format
3. **Proper Dtypes**: Use ``float32`` for continuous features, ``long`` for categorical
4. **Handle Missing Data**: Use masking or imputation strategies

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

1. **Preprocess Once**: Do heavy preprocessing in ``__init__``, not ``__getitem__``
2. **Efficient Storage**: Use appropriate data types and consider memory mapping for large datasets
3. **Caching**: Cache processed data when possible

.. code-block:: python

    # Example: Efficient data loading
    class EfficientDataset(TimeSeriesDataset):
        def __init__(self, data_path: str):
            super().__init__()
            # Preprocess everything once
            self.processed_data = self._preprocess_all_data(data_path)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            # Just slice pre-processed tensors
            return {
                key: tensor[self._get_slice(idx, key)]
                for key, tensor in self.processed_data.items()
            }

Integration with Models
----------------------

The standardized format works seamlessly with FinTorch models:

.. code-block:: python

    from fintorch.models.timeseries.tft.tft_module import TemporalFusionTransformerModule
    from torch.utils.data import DataLoader

    # Create dataset and dataloader
    dataset = CustomTimeSeriesDataset("data.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model with matching dimensions
    model = TemporalFusionTransformerModule(
        number_of_past_inputs=dataset.time_steps,
        horizon=dataset.future_steps,
        embedding_size_inputs=64,
        hidden_dimension=128,
        dropout=0.1,
        number_of_heads=4,
        num_past_target_features=dataset.num_target_features,
        num_past_known_cov_features=dataset.num_known_future_cov_features,
        num_past_unknown_cov_features=dataset.num_unknown_future_cov_features,
        num_future_known_cov_features=dataset.num_known_future_cov_features,
        num_static_real_features=dataset.num_static_real_features,
        num_static_categorical_features=dataset.num_static_categorical_features,
        static_categorical_cardinalities=dataset.static_categorical_cardinalities,
        batch_size=32,
        device="cuda",
        series_selection_method="first"  # or "aggregate", "index", etc.
    )

    # Training loop
    for batch in dataloader:
        result = model.training_step(batch, 0)
        loss = result["loss"]
        # ... continue training

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Shape Mismatches**
    Ensure your tensors have the correct dimensions. Use ``.reshape()`` if needed.

**Dtype Errors**
    Categorical features must be ``torch.long``, continuous features should be ``torch.float32``.

**Memory Issues**
    For large datasets, consider lazy loading or data generators.

**Model Compatibility**
    Verify that your dataset's feature dimensions match the model's expected input dimensions.

.. code-block:: python

    # Debug shapes and types
    sample = dataset[0]
    for key, tensor in sample.items():
        print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")

Conclusion
----------

The standardized time series format in FinTorch provides a robust foundation for time series modeling. By following this structure, you can:

- Create interoperable datasets that work with all FinTorch models
- Leverage multi-series capabilities
- Clearly separate different types of features
- Build scalable and maintainable data pipelines

For more examples, see the existing dataset implementations in ``fintorch.datasets`` and the model tutorials showing integration with TFT and CausalFormer.
