import torch
from fintorch.models.timeseries.causalformer.CausalConvolution import CausalConvolution
from torch.testing import assert_close


def test_stack_shifted_kernel_shape():
    # Define parameters
    number_of_series = 3
    length_input_window = 4
    number_of_heads = 2

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    # Get the stacked shifted kernel
    stacked_kernel = causal_conv.stack_shifted_kernel(causal_conv.kernel)

    # Expected shape: (number_of_heads, number_of_series, number_of_series, length_input_window, length_input_window)
    expected_shape = (
        number_of_heads,
        number_of_series,
        number_of_series,
        length_input_window,
        length_input_window,
    )

    print(f"Kernel shape: {causal_conv.kernel.shape}")
    print(f"Stacked kernel shape: {stacked_kernel.shape}")
    print(f"Expected shape: {expected_shape}")

    # Assertions
    assert (
        stacked_kernel.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {stacked_kernel.shape}"


def test_stack_shifted_kernel_lower_triangular():
    # Define parameters
    number_of_series = 3
    length_input_window = 4
    number_of_heads = 2

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    # Call stack_shifted_kernel
    stacked_kernel = causal_conv.stack_shifted_kernel(causal_conv.kernel)

    # Check if the kernel is lower triangular along the last two dimensions
    for i in range(length_input_window):
        for j in range(length_input_window):
            if j > i:
                assert torch.all(
                    stacked_kernel[..., i, j] == 0
                ), f"Kernel is not lower triangular at position ({i}, {j})"


def test_apply_kernel():
    # Define parameters
    batch_size = 2
    number_of_series = 3
    length_input_window = 4
    hidden_dimensionality = 5
    number_of_heads = 7

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    # Generate random input tensor `x` and kernel
    x = torch.randn(
        batch_size, number_of_series, length_input_window, hidden_dimensionality
    )
    kernel = causal_conv.stack_shifted_kernel(causal_conv.kernel)

    # Compute output using `einsum`
    einsum_result = torch.einsum("hxyji,bxif->bhxyjf", kernel, x)

    # Compute output using `stacked_output`
    stacked_output_result = causal_conv.apply_kernel(x, kernel)

    # Assert that the two results are close
    assert_close(
        stacked_output_result,
        einsum_result,
        rtol=1e-5,
        atol=1e-8,
        msg="stacked_output does not match einsum result",
    )


def test_transform_x():
    # Define parameters
    batch_size = 2
    number_of_series = 5
    length_input_window = 6
    hidden_dimensionality = 7
    number_of_heads = 4

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    # Create a sample input tensor
    x = torch.randn(
        batch_size,
        number_of_heads,
        number_of_series,
        number_of_series,
        length_input_window,
        hidden_dimensionality,
    )

    # Transform x
    transformed_x = causal_conv.transform_x(x.clone())

    # Check if the output shape is correct
    expected_shape = (
        batch_size,
        number_of_heads,
        number_of_series,
        number_of_series,
        length_input_window,
        hidden_dimensionality,
    )
    assert (
        transformed_x.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {transformed_x.shape}"

    # Check if the first element is zero
    for i in range(number_of_series):
        assert torch.all(
            transformed_x[:, :, i, i, 0, :] == 0
        ), f"Expected first element to be zero for series {i}"

    # Check if the other elements are shifted
    for i in range(number_of_series):
        for j in range(1, length_input_window):
            assert torch.all(
                transformed_x[:, :, i, i, j, :] == x[:, :, i, i, j - 1, :]
            ), f"Expected element {j} to be shifted for series {i}"


def test_forward_pass():
    # Define parameters
    batch_size = 2
    number_of_series = 3
    length_input_window = 4
    hidden_dimensionality = 5
    number_of_heads = 2

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    # Create a sample input tensor
    x = torch.randn(
        batch_size, number_of_series, length_input_window, hidden_dimensionality
    )

    # Forward pass
    output = causal_conv(x)

    # Check if the output shape is correct
    expected_shape = (
        batch_size,
        number_of_heads,
        number_of_series,
        number_of_series,
        length_input_window,
        hidden_dimensionality,
    )
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"

    # Check if the output is a tensor
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"


def test_kernel_requires_grad():
    # Define parameters
    number_of_series = 3
    length_input_window = 4
    number_of_heads = 2

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        number_of_heads=number_of_heads,
    )

    # Check if the kernel requires gradient
    assert causal_conv.kernel.requires_grad, "Kernel should require gradient"

    # Check if the stacked kernel requires gradient
    stacked_kernel = causal_conv.stack_shifted_kernel(causal_conv.kernel)
    assert stacked_kernel.requires_grad, "Stacked kernel should require gradient"


def test_base_shape():
    # Define parameters
    length_input_window = 4

    # Initialize the CausalConvolution module
    causal_conv = CausalConvolution(
        number_of_series=3,
        length_input_window=length_input_window,
        number_of_heads=2,
    )

    # Check if the base shape is correct
    expected_shape = (1, 1, 1, 1, length_input_window, 1)
    assert (
        causal_conv.base.shape == expected_shape
    ), f"Expected base shape {expected_shape}, but got {causal_conv.base.shape}"
