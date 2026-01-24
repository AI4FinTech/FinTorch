# Time Series Data Handling in FinTorch

FinTorch provides a unified, modular architecture for time series forecasting that enables easy mixing and matching of datasets with different models. This document explains the core concepts, shared data structures, and how different components work together.

## Overview

The FinTorch time series framework is built around three key principles:

1. **Standardized Data Format**: All time series datasets follow a common interface and return data in a consistent format
1. **Model Agnostic Design**: Models can work with any dataset that implements the base interface
1. **Flexible Multi-Series Support**: Handle both single and multi-variate time series data seamlessly

## Core Data Structure

### Base Dataset Interface

All time series datasets in FinTorch inherit from `TimeSeriesDataset` (defined in `fintorch/datasets/base/base_dataset.py`), which establishes a standard interface:

```python
class TimeSeriesDataset(Dataset, ABC):
    """
    Base abstract class for time series datasets in FinTorch.

    Standard format:
    - past_data: (time_steps, series_dim, features_dim)
    - future_data: (future_steps, series_dim, features_dim)
    - static_data: (static_length,)
    - target: (future_steps, series_dim, features_dim)
    """
```

### Data Format Specification

The standardized return format from `__getitem__` ensures consistency across all datasets:

```python
def __getitem__(self, idx: int) -> Tuple[
    Dict[str, torch.Tensor],  # past_inputs
    Dict[str, torch.Tensor],  # future_inputs
    Dict[str, torch.Tensor],  # static_inputs
    torch.Tensor,             # target
]:
```

Where:

- **past_inputs**: Dictionary containing `"past_data"` tensor of shape `(time_steps, series_dim, features_dim)`
- **future_inputs**: Dictionary containing `"future_data"` tensor of shape `(future_steps, series_dim, features_dim)`
- **static_inputs**: Dictionary containing `"static_data"` tensor of shape `(static_length,)`
- **target**: Target tensor of shape `(future_steps, series_dim, features_dim)`

## Dataset Implementations

### SimpleSyntheticDataset

The `SimpleSyntheticDataset` (`fintorch/datasets/synthetic/simpleSynthetic.py`) demonstrates the full implementation of the base interface:

- **Inherits from**: `TimeSeriesDataset`
- **Data Generation**: Creates synthetic multi-series data with configurable parameters
- **Key Features**:
  - Generates multiple related time series with seasonal patterns
  - Includes static features for each series
  - Provides data scaling and inverse transformation capabilities
  - Full compliance with the standardized data format

**Usage Example**:

```python
dataset = SimpleSyntheticDataset(
    num_series=5,
    past_length=24,
    future_length=6,
    features_dim=3
)
```

### ElectricityDataset

The `ElectricityDataset` (`fintorch/datasets/electricity_simple.py`) shows real-world data integration:

- **Data Source**: PJM hourly electricity consumption data
- **Adaptation**: Transforms external data into the standardized format
- **Key Features**:
  - Downloads real electricity consumption data
  - Applies standardized scaling
  - Converts single-series data to the multi-dimensional format
  - Maintains temporal order for train/validation/test splits

**Note**: While this dataset doesn't fully inherit from `TimeSeriesDataset` yet, it demonstrates the data transformation patterns needed for compliance.

## Model Integration

### Temporal Fusion Transformer (TFT)

The TFT implementation (`fintorch/models/timeseries/tft/`) showcases advanced model integration:

**Key Components**:

- **TemporalFusionTransformer**: Core model architecture
- **TemporalFusionTransformerModule**: Lightning wrapper with data preprocessing
- **Multi-Series Handling**: Configurable strategies for processing multi-series data

**Data Processing Strategies**:

```python
# Available series selection methods
series_selection_method = "first"      # Use first series only
series_selection_method = "last"       # Use last series only
series_selection_method = "index"      # Use specific series index
series_selection_method = "aggregate"  # Aggregate across series (mean/sum/max/min)
series_selection_method = "flatten"    # Flatten all series as features
```

**Model Configuration**:

```python
tft_model = TemporalFusionTransformerModule(
    number_of_past_inputs=24,
    horizon=6,
    past_inputs={"past_data": 1},
    future_inputs={"future_data": 1},
    static_inputs={"static_data": 10},
    series_selection_method="aggregate",
    series_aggregation="mean"
)
```

### CausalFormer

The CausalFormer implementation (tested in `tests/models/causalformer/`) demonstrates another model architecture that can work with the standardized data format:

**Key Features**:

- **Causal Convolution**: Handles temporal dependencies with causal constraints
- **Multi-Head Attention**: Processes relationships between time series
- **Encoder-Decoder Architecture**: Flexible sequence-to-sequence modeling

## Benefits of the Unified Architecture

### 1. Dataset-Model Interoperability

Any dataset implementing `TimeSeriesDataset` can work with any model that accepts the standardized format:

```python
# Use synthetic data with TFT
synthetic_data = SimpleSyntheticDataset(...)
tft_model = TemporalFusionTransformerModule(...)
trainer.fit(tft_model, DataLoader(synthetic_data))

# Use same synthetic data with CausalFormer
causalformer_model = CausalFormer(...)
trainer.fit(causalformer_model, DataLoader(synthetic_data))
```

### 2. Consistent Data Preprocessing

The standardized format eliminates model-specific data preprocessing:

```python
# All models receive data in the same format
past_inputs, future_inputs, static_inputs, target = batch
```

### 3. Easy Experimentation

Researchers can quickly compare different models on the same dataset or test the same model on different datasets without code changes.

### 4. Multi-Series Flexibility

The architecture handles both single and multi-series scenarios transparently:

- **Single Series**: Use `series_dim=1`
- **Multi-Series**: Configure appropriate `series_selection_method`
- **Mixed Scenarios**: Models adapt automatically based on input dimensions

## Implementation Guidelines

### For New Datasets

1. **Inherit from TimeSeriesDataset**:

```python
class MyDataset(TimeSeriesDataset):
    def __getitem__(self, idx):
        # Return standardized format
        return past_inputs, future_inputs, static_inputs, target
```

2. **Implement Required Properties**:

```python
@property
def time_steps(self) -> int:
    return self._time_steps

@property
def future_steps(self) -> int:
    return self._future_steps

# ... other required properties
```

3. **Handle Multi-Series Data**:
   - Ensure tensors have correct dimensions: `(time_steps, series_dim, features_dim)`
   - Set `series_dim=1` for single series data
   - Include static features when available

### For New Models

1. **Accept Standardized Input Format**:

```python
def forward(self, past_inputs, future_inputs, static_inputs):
    past_data = past_inputs["past_data"]
    # ... model logic
```

2. **Handle Multi-Series Data**:

   - Implement appropriate strategies for multi-series processing
   - Consider providing configuration options for series handling
   - Document expected input dimensions

1. **Provide Lightning Module Wrapper**:

   - Include data preprocessing in the Lightning module
   - Handle different batch formats gracefully
   - Implement proper loss functions for multi-output scenarios

## Future Extensions

The standardized architecture enables several future enhancements:

1. **Additional Data Types**: Easy integration of new data sources
1. **Model Ensembles**: Combine multiple models with consistent interfaces
1. **Transfer Learning**: Pre-train off one dataset, fine-tune on another
1. **Automated Benchmarking**: Compare all models on all datasets systematically
1. **Real-time Inference**: Consistent serving interface across models

This unified approach makes FinTorch a powerful platform for time series research and production applications, enabling rapid experimentation while maintaining code clarity and reusability.
