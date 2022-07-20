from typing import Tuple
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(
        self,
        kernel_a: Tuple[int, int],
        kernel_b: Tuple[int, int],
        dilation,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            padding="same",
            kernel_size=kernel_a,
            dilation=dilation,
        )
        self.leaky_relu_1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            padding="same",
            kernel_size=kernel_b,
            dilation=dilation,
        )
        self.leaky_relu_2 = nn.LeakyReLU()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.leaky_relu_1(x)
        x = self.conv2(x)

        x += identity

        x = self.leaky_relu_2(x)

        return x
