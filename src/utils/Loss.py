import torch
from torch import nn


class Loss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, densities, compliances, volume_fraction, penalization,
                device):
        alpha = 1
        beta = 5
        gamma = 1
        delta = 1

        compliances = torch.sum(
            torch.div(
                compliances,
                torch.pow(
                    densities.transpose(1, 2).flatten(
                        start_dim=1,
                        end_dim=2,
                    ),
                    penalization,
                ),
            ),
            dim=1,
        )

        volume_fraction = volume_fraction.squeeze(dim=1)

        volume_error = torch.abs(volume_fraction - torch.mean(
            densities,
            axis=[1, 2],
        ))

        checkerboard_v = torch.abs(
            torch.nn.functional.conv2d(
                torch.unsqueeze(densities, dim=1),
                0.25 * torch.tensor(
                    [[[
                        [1, -1, 1],
                        [-1, 0, -1],
                        [1, -1, 1],
                    ]]],
                    dtype=torch.float32,
                ).to(device),
            )).mean()

        checkerboard_error = (torch.exp(checkerboard_v * 3) - 1) / (torch.exp(torch.tensor(3)) - 1)

        uncertainty = torch.mean(
            torch.exp(-6.5 * torch.pow(2 * densities - 1, 2)).to(device),
            axis=[1, 2],
        )

        return torch.mean(
            (alpha * compliances + 1) * (beta * volume_error + 1) *
            (gamma * checkerboard_error + 1) * (delta * uncertainty + 1))
