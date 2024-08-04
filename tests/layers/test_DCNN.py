import torch
from fintorch.layers.DCNN import DilatedConvolution


def test_dilated_convolution_forward():
    number_of_layers = 3
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    stride = 1
    sequence_length = 10
    batch_size = 32

    dilated_convolution = DilatedConvolution(
        number_of_layers, in_channels, out_channels, kernel_size, stride
    )

    x = torch.randn(batch_size, in_channels, sequence_length)
    output = dilated_convolution.forward(x)

    assert output.shape == (batch_size, out_channels, sequence_length)
