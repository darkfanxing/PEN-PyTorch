from torch import nn


def hidden_block(
    in_features: int,
    out_features: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        nn.BatchNorm1d(out_features),
        nn.PReLU(),
    )
