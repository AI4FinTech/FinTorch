import torch.nn as nn


class DilatedConvolutionsLearnableSpacing(nn.Module):
    """

    References:
    Khalfaoui-Hassani, Ismail, Thomas Pellegrini, and Timothée Masquelier. "Dilated convolution with learnable spacings." arXiv preprint arXiv:2112.03740 (2021).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
