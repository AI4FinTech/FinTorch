# Causal Data Module Test Suite

This directory contains comprehensive unit tests for the `fintorch.datasets.causal_data` module, providing extensive coverage for all classes, functions, and edge cases.

## Overview

The test suite validates the functionality of:
- `CausalDataset` class for loading and preprocessing causal time-series data
- `CausalDataModule` class for PyTorch Lightning data management
- Utility functions for data management and processing
- Edge cases and error handling scenarios

## Test Files

### 1. `test_causal_data.py`
**Purpose**: Core functionality tests for main classes and functions
**Coverage**:
- `CausalDataset` initialization and basic operations
- `CausalDataModule` initialization and dataloader creation
- `custom_collate_fn` for batch processing
- `_download_causal_data` function
- `create_causal_datamodule` factory function

### 2. `test_causal_utility_functions.py`
**Purpose**: Utility function tests
**Coverage**:
- `get_causal_data_dir()` - Data directory management
- `clear_causal_data()` - Data cleanup functionality
- `list_available_datasets()` - Dataset discovery
- `get_dataset_info()` - Dataset metadata retrieval
- `get_clean_adjacency_matrix()` - Matrix processing utilities

### 3. `test_causal_comprehensive.py`
**Purpose**: Comprehensive testing with extensive edge cases
**Coverage**:
- **CausalDataset Tests**:
  - Initialization with various parameter combinations
  - Data loading with valid and invalid files
  - Index generation for time series sampling
  - Property accessors and data retrieval
  - Groundtruth handling
  
- **CausalDataModule Tests**:
  - Parameter validation and initialization
  - Dataset splitting and setup
  - DataLoader creation and configuration
  - Groundtruth access methods
  
- **Utility Function Tests**:
  - Collate function with same and different sized tensors
  - Download functionality with success/failure scenarios
  - Matrix processing with various input formats
  
- **Edge Case Tests**:
  - Empty data files
  - Non-numeric data handling
  - Corrupted CSV files
  - Deterministic static feature generation

## Test Features

### 🧪 Comprehensive Coverage
- **43 total tests** covering all public methods and functions
- **100% function coverage** of the causal_data module
- **Edge case handling** for robust error conditions
- **Property testing** for all class properties

### 🔧 Mocking and Fixtures
- **Temporary directories** for isolated file system testing
- **Mock HTTP requests** for download functionality testing
- **Parameterized fixtures** for reusable test configurations
- **Proper cleanup** to prevent test interference

### 📊 Data Validation
- **Shape verification** for tensor outputs
- **Type checking** for returned objects
- **Content validation** for processed data
- **Deterministic testing** for reproducible results

## Running the Tests

### Run All Tests
```bash
python -m pytest tests/datasets/causal_data/ -v
```

### Run Specific Test Files
```bash
# Core functionality tests
python -m pytest tests/datasets/causal_data/test_causal_data.py -v

# Utility function tests
python -m pytest tests/datasets/causal_data/test_causal_utility_functions.py -v

# Comprehensive tests with edge cases
python -m pytest tests/datasets/causal_data/test_causal_comprehensive.py -v
```

### Run Specific Test Classes
```bash
# Test only CausalDataset functionality
python -m pytest tests/datasets/causal_data/test_causal_comprehensive.py::TestCausalDataset -v

# Test only CausalDataModule functionality
python -m pytest tests/datasets/causal_data/test_causal_comprehensive.py::TestCausalDataModule -v

# Test only edge cases
python -m pytest tests/datasets/causal_data/test_causal_comprehensive.py::TestEdgeCases -v
```

## Test Data

### Mock Data Structure
Tests use temporary directories with mock CSV files:
```
temp_dir/
└── diamond/
    ├── data_0.csv    # Time series data (nodes as columns)
    ├── data_1.csv    # Additional time series data
    ├── data_2.csv    # More time series data
    └── groundtruth.csv  # Causal relationship matrix
```

### Data Formats
- **Time Series Files**: CSV format with numeric node data
- **Groundtruth Files**: Adjacency matrix showing causal relationships
- **Various Sizes**: Tests with different data volumes and dimensions

## Key Test Scenarios

### ✅ Success Cases
- Valid dataset initialization and loading
- Proper data preprocessing and scaling
- Correct tensor shape generation
- Successful batch collation
- Working dataloader creation

### ❌ Error Cases
- Invalid dataset types
- Missing or corrupted data files
- Invalid time step/output window combinations
- Non-numeric data handling
- Network failures during downloads

### 🔄 Edge Cases
- Empty datasets
- Single data point scenarios
- Very large time windows
- Floating point precision issues
- File system permission errors

## Dependencies

The test suite requires:
- `pytest` - Testing framework
- `torch` - PyTorch tensors and operations
- `polars` - DataFrame operations
- `numpy` - Numerical computations
- `requests` - HTTP mocking
- `lightning` - PyTorch Lightning framework
- `sklearn` - Data preprocessing utilities

## Contributing

When adding new tests:

1. **Follow naming conventions**: Test functions should start with `test_`
2. **Use descriptive names**: Clearly indicate what functionality is being tested
3. **Include docstrings**: Explain the purpose and scope of each test
4. **Mock external dependencies**: Use patches for file I/O and network calls
5. **Clean up resources**: Use context managers and fixtures for proper cleanup
6. **Test edge cases**: Include both success and failure scenarios
7. **Verify shapes and types**: Always check tensor dimensions and data types

## Performance Considerations

- Tests use **temporary directories** to avoid file system conflicts
- **Mocked network calls** prevent actual downloads during testing
- **Small mock datasets** ensure fast test execution
- **Parallel execution** supported through pytest-xdist if needed

## Maintenance

The test suite is designed to be:
- **Self-contained**: No external dependencies or data files required
- **Deterministic**: Consistent results across different environments
- **Maintainable**: Clear structure and comprehensive documentation
- **Extensible**: Easy to add new tests for additional functionality

For issues or questions about the test suite, please refer to the main FinTorch documentation or create an issue in the project repository.