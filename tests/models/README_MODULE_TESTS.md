# FinTorch Module Tests Documentation

This document provides comprehensive documentation for the test suites covering the TemporalFusionTransformerModule (TFT) and CausalFormerModule components of FinTorch.

## Overview

The test suite has been significantly expanded to provide comprehensive coverage with over 100 test cases across multiple categories:

- **Unit Tests**: Core functionality and API testing
- **Integration Tests**: Cross-module compatibility and data format standardization
- **Performance Tests**: Speed, scalability, and resource usage
- **Stress Tests**: Stability under load and edge conditions
- **Edge Case Tests**: Boundary conditions and error handling
- **Memory Tests**: Memory usage patterns and leak detection

## Test Structure

```
tests/models/
├── tft/
│   └── test_tftmodule.py              # Comprehensive TFT module tests
├── causalformer/
│   └── test_causalformer_module.py    # Comprehensive CausalFormer tests
├── test_module_integration.py         # Cross-module integration tests
├── test_module_performance.py         # Performance and stress tests
└── README_MODULE_TESTS.md            # This documentation
```

## Test Categories

### 1. Unit Tests (`TestTemporalFusionTransformerModule`, `TestCausalFormerModule`)

#### Model Initialization Tests

- **Basic initialization**: Tests default parameter handling
- **Custom parameters**: Tests custom series handling configurations
- **Parameter validation**: Tests invalid parameter combinations
- **Hyperparameter saving**: Tests parameter persistence

#### Data Processing Tests

- **Input tensor shapes**: Tests 3D and 4D input handling
- **Multi-series processing**: Tests different series selection methods
- **Target tensor processing**: Tests various target tensor formats
- **Data concatenation**: Tests feature concatenation logic

#### Series Selection Methods

- **First series**: `series_selection_method="first"`
- **Last series**: `series_selection_method="last"`
- **Index-based**: `series_selection_method="index"` with `series_index`
- **Aggregation**: `series_selection_method="aggregate"` with various aggregation methods
- **Flatten**: `series_selection_method="flatten"`

#### Training Loop Tests

- **Training step**: Forward pass with loss computation
- **Validation step**: Evaluation mode testing
- **Test step**: Test phase evaluation
- **Predict step**: Inference without targets

#### Error Handling Tests

- **Missing batch keys**: Tests handling of incomplete data
- **Invalid series indices**: Tests out-of-range series selection
- **Empty batches**: Tests graceful error handling
- **Invalid parameters**: Tests parameter validation

### 2. Integration Tests (`TestModuleIntegration`)

#### Cross-Module Compatibility

- **Standardized data format**: Both modules handle identical input formats
- **Lightning integration**: Both modules work with PyTorch Lightning
- **Optimizer configuration**: Tests optimizer and scheduler setup
- **Multi-series consistency**: Consistent behavior across modules

#### Data Format Compatibility

- **New standardized format**: Tests with current batch structure
- **Backward compatibility**: Tests legacy format support
- **Series dimension handling**: Consistent multi-series processing

### 3. Edge Case Tests (`TestEdgeCases`)

#### Extreme Configurations

- **Minimal configuration**: Single layer, single head, minimal features
- **Large configuration**: Many layers, large hidden dimensions, long sequences
- **Batch size variations**: From 1 to 64+ samples
- **Sequence length variations**: From 1 to 100+ time steps

#### Extreme Values

- **Very large values**: Input magnitudes > 1000
- **Very small values**: Input magnitudes \< 1e-6
- **NaN handling**: Graceful NaN input processing
- **Infinite values**: Handling of infinite inputs
- **Zero inputs**: All-zero input handling

#### Boundary Conditions

- **Single sample batches**: Batch size = 1
- **Single time step**: Sequence length = 1
- **Device compatibility**: CPU/GPU transitions

### 4. Performance Tests (`TestPerformance`)

#### Timing Tests

- **Forward pass timing**: Inference speed measurement
- **Backward pass timing**: Training speed measurement
- **Warmup protocols**: Consistent timing methodology

#### Scalability Tests

- **Batch size scaling**: Performance vs batch size (1-64)
- **Sequence length scaling**: Performance vs sequence length (10-100)
- **Model size scaling**: Performance vs model parameters

#### Memory Tests

- **Memory usage patterns**: Peak memory consumption
- **Memory cleanup**: Proper memory deallocation
- **Memory leak detection**: Long-running stability

#### Optimization Tests

- **Gradient flow**: Proper gradient computation
- **Numerical stability**: Loss and gradient stability
- **Deterministic behavior**: Reproducible results with fixed seeds

### 5. Stress Tests (`TestStress`)

#### Long-Running Stability

- **Consecutive training steps**: 100+ training iterations
- **Memory leak detection**: Resource cleanup over time
- **Gradient accumulation**: Multi-batch gradient stability
- **Concurrent instances**: Multiple model instances

#### Resource Limits

- **Memory pressure**: High memory usage scenarios
- **Computational load**: Extended computation periods
- **Error recovery**: Handling of numerical instabilities

## Key Test Features

### Fixtures and Parameterization

- **Parameterized tests**: Multiple configurations tested automatically
- **Reusable fixtures**: Consistent test data and model configurations
- **Mock objects**: Safe testing of external dependencies

### Assertions and Validations

- **Shape validation**: Comprehensive tensor shape checking
- **Value range validation**: Reasonable output value ranges
- **Performance benchmarks**: Speed and memory thresholds
- **Numerical stability**: NaN and infinity detection

### Test Data Generation

- **Synthetic data**: Controlled test scenarios
- **Random seeds**: Reproducible test conditions
- **Realistic shapes**: Production-like data dimensions
- **Edge case data**: Boundary condition testing

## Running the Tests

### Quick Test Run

```bash
# Run all basic tests
python -m pytest tests/models/tft/test_tftmodule.py tests/models/causalformer/test_causalformer_module.py -v

# Run specific module tests
python -m pytest tests/models/tft/test_tftmodule.py -v
python -m pytest tests/models/causalformer/test_causalformer_module.py -v
```

### Comprehensive Test Suite

```bash
# Run all tests with coverage
python run_module_tests.py --mode all --verbose

# Run specific test categories
python run_module_tests.py --mode unit
python run_module_tests.py --mode performance
python run_module_tests.py --mode stress
```

### Test Categories by Markers

```bash
# Run only fast tests
python -m pytest tests/models/ -m "not slow and not performance and not stress"

# Run performance tests
python -m pytest tests/models/ -m "performance"

# Run TFT-specific tests
python -m pytest tests/models/ -m "tft"

# Run CausalFormer-specific tests
python -m pytest tests/models/ -m "causalformer"
```

### Test Configuration Options

```bash
# Run with coverage reporting
python -m pytest tests/models/ --cov=fintorch.models.timeseries --cov-report=html

# Run with timeout protection
python -m pytest tests/models/ --timeout=300

# Run with detailed output
python -m pytest tests/models/ -v --tb=long

# Run specific test patterns
python -m pytest tests/models/ -k "memory or gradient"
```

## Test Coverage Goals

### Functional Coverage

- ✅ **Model Initialization**: 100% of parameters and configurations
- ✅ **Data Processing**: All input formats and transformations
- ✅ **Training Loop**: All training/validation/test phases
- ✅ **Series Handling**: All selection and aggregation methods
- ✅ **Error Conditions**: Comprehensive error handling

### Performance Coverage

- ✅ **Speed Benchmarks**: Forward and backward pass timing
- ✅ **Memory Profiling**: Usage patterns and leak detection
- ✅ **Scalability**: Batch size and sequence length scaling
- ✅ **Resource Limits**: Stress testing under load

### Integration Coverage

- ✅ **Cross-Module**: TFT and CausalFormer compatibility
- ✅ **Lightning Integration**: PyTorch Lightning compatibility
- ✅ **Data Format**: Standardized batch format support
- ✅ **Device Compatibility**: CPU and GPU support

## Performance Benchmarks

### Expected Performance Targets

#### TFT Module

- **Forward Pass**: \< 1.0s for batch_size=64, sequence_length=50
- **Backward Pass**: \< 2.0s for batch_size=64, sequence_length=50
- **Memory Usage**: \< 500MB peak for typical configurations
- **Batch Scaling**: \< 3x time increase when doubling batch size

#### CausalFormer Module

- **Forward Pass**: \< 0.5s for batch_size=64, sequence_length=50
- **Backward Pass**: \< 1.0s for batch_size=64, sequence_length=50
- **Memory Usage**: \< 300MB peak for typical configurations
- **Batch Scaling**: \< 2.5x time increase when doubling batch size

### Memory Benchmarks

- **Memory Leaks**: \< 200MB increase over 20 iterations
- **Cleanup Efficiency**: > 80% memory recovery after model deletion
- **Concurrent Usage**: Support for 3+ model instances simultaneously

## Test Maintenance

### Adding New Tests

1. **Follow naming conventions**: `test_<functionality>_<variation>`
1. **Use appropriate fixtures**: Reuse existing data and model fixtures
1. **Add proper markers**: Mark tests with appropriate categories
1. **Include assertions**: Comprehensive validation of outputs
1. **Document purpose**: Clear test function docstrings

### Test Categories for New Features

```python
@pytest.mark.unit
def test_new_feature_basic():
    """Test basic functionality of new feature"""
    pass

@pytest.mark.performance
def test_new_feature_performance():
    """Test performance characteristics of new feature"""
    pass

@pytest.mark.edge_case
def test_new_feature_edge_cases():
    """Test edge cases and error conditions"""
    pass
```

### Updating Benchmarks

- Review performance targets quarterly
- Update memory limits based on infrastructure changes
- Adjust timeout values for slower environments
- Add new test scenarios for feature additions

## Troubleshooting

### Common Test Failures

#### Memory Issues

```
Error: Memory increased by X MB
```

- **Cause**: Potential memory leak
- **Solution**: Check model cleanup, add explicit `del` statements

#### Timeout Issues

```
Test timed out after X seconds
```

- **Cause**: Slow operations or infinite loops
- **Solution**: Increase timeout or optimize test conditions

#### Shape Mismatches

```
Expected shape [...] but got [...]
```

- **Cause**: Input/output tensor shape inconsistencies
- **Solution**: Verify data preparation and model configuration

#### Numerical Instabilities

```
NaN or Inf detected in loss/gradients
```

- **Cause**: Gradient explosion or numerical overflow
- **Solution**: Check learning rates, gradient clipping, input ranges

### Performance Debugging

- Use `--durations=0` to identify slow tests
- Profile memory usage with `memory_profiler`
- Monitor GPU utilization with `nvidia-smi`
- Check gradient norms for stability

## Contributing

### Test Development Guidelines

1. **Write failing tests first**: TDD approach
1. **Test edge cases**: Don't just test happy paths
1. **Use descriptive names**: Clear test intent
1. **Keep tests isolated**: No dependencies between tests
1. **Mock external dependencies**: Control test environment

### Code Quality Standards

- **Type hints**: Use proper type annotations
- **Documentation**: Comprehensive docstrings
- **Error handling**: Proper exception testing
- **Performance**: Consider test execution time
- **Maintainability**: Clean, readable test code

This comprehensive test suite ensures the reliability, performance, and maintainability of the FinTorch time series modules. Regular execution of these tests helps maintain code quality and catch regressions early in the development process.
