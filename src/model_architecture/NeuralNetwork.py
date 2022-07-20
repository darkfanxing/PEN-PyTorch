import torch
from torch import nn
from .blocks import hidden_block, ResidualBlock


class NeuralNetwork(nn.Module):
    min_density = 0.001

    def __init__(self):
        super().__init__()
        self.hidden_block_1 = hidden_block(325, 256)
        self.hidden_block_2 = hidden_block(256, 128)
        self.hidden_block_3 = hidden_block(128, 2048)
        self.up_sampling_block = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Upsample(scale_factor=4, mode="nearest"),
        )
        self.residual_block_1 = ResidualBlock(
            kernel_a=(5, 5),
            kernel_b=(6, 6),
            dilation=2,
        )
        self.residual_block_2 = ResidualBlock(
            kernel_a=(4, 4),
            kernel_b=(5, 5),
            dilation=4,
        )
        self.residual_block_3 = ResidualBlock(
            kernel_a=(3, 3),
            kernel_b=(4, 4),
            dilation=8,
        )
        self.average_pooling_1 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
        )
        self.average_pooling_2 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
        )
        self.average_pooling_3 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_block_1(x)
        x = self.hidden_block_2(x)
        x = self.hidden_block_3(x)
        x = torch.reshape(x, (-1, 8, 16, 16))
        x = self.up_sampling_block(x)
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        x_8 = self.residual_block_3(x)
        x_4 = self.average_pooling_1(x_8)
        x_2 = self.average_pooling_2(x_4)
        x = self.average_pooling_3(x_2)

        x_8 = nn.Sigmoid()(torch.mean(x_8, dim=1))
        x_4 = nn.Sigmoid()(torch.mean(x_4, dim=1))
        x_2 = nn.Sigmoid()(torch.mean(x_2, dim=1))
        x = nn.Sigmoid()(torch.mean(x, dim=1))

        x_8 = torch.where(x_8 < self.min_density,
                          self.min_density * torch.ones_like(x_8), x_8)
        x_4 = torch.where(x_4 < self.min_density,
                          self.min_density * torch.ones_like(x_4), x_4)
        x_2 = torch.where(x_2 < self.min_density,
                          self.min_density * torch.ones_like(x_2), x_2)
        x = torch.where(x < self.min_density,
                        self.min_density * torch.ones_like(x), x)

        return [x, x_2, x_4, x_8]
