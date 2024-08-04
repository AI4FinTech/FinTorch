import torch.nn as nn


class DilatedConvolution(nn.Module):
    """

    References:
    - Borovykh, Anastasia, Sander Bohte, and Cornelis W. Oosterlee. 2018. “Dilated Convolutional Neural Networks for Time Series Forecasting.” Journal of Computational Finance. https://doi.org/10.21314/jcf.2019.358.

    """

    def __init__(
        self,
        number_of_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()

        # We have L dilation layers and the 2^{L-1} item is connected to the next 2^L items
        # For time-series analysis
        # input = (batch_size, in_channels, sequence_length)
        # weight = (out_channels, in_channels, iW)
        # - out_channels: The number of desired output channels (feature maps).
        # - iW: The filter size (kernel size) along the time dimension.
        # dilation size = 2^l
        # padding = (kernel_size-1) * dilation

        self.layers = nn.ModuleList()
        for i in range(number_of_layers):
            dilation_size = 2**i
            padding = (kernel_size - 1) * dilation_size
            layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation_size,
            )
            self.layers.append(layer)
            self.layers.append(nn.ReLU())

    def forward(self, x):
        # migth need a transpose to get the right shape
        for layer in self.layer:
            x = layer(x)

        return x
