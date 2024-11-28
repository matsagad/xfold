from functools import partial
import torch
import torch.nn as nn


ROW_DIM = 0
COL_DIM = 1


class AxialDropout2D(nn.Module):
    def __init__(self, p: float, axis: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dropout_shape = [1, 1, 1]
        dropout_shape[self.axis] = x.shape[self.axis]
        return x * self.dropout(torch.ones(dropout_shape, device=x.device))


DropoutRowwise = partial(AxialDropout2D, axis=ROW_DIM)
DropoutColumnwise = partial(AxialDropout2D, axis=COL_DIM)

LinearNoBias = partial(nn.Linear, bias=False)


class LinearSigmoid(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))


class OneHotNearestBin(nn.Module):
    def __init__(self, bins: torch.Tensor) -> None:
        super().__init__()
        self.bins = bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bins = self.bins.shape[0]
        p = torch.zeros((x.numel(), n_bins), device=x.device)
        p[torch.argmin(torch.abs(x.view(-1, 1) - self.bins.unsqueeze(0)))] = 1
        return p.view(*x.shape, n_bins)


class OuterProductMean(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, proj_dim)
        self.proj2 = nn.Linear(input_dim, proj_dim)

        self.out_proj = nn.Linear(proj_dim * proj_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)

        a = self.proj1(x)
        b = self.proj2(x)
        o = (a.unsqueeze(-1) * b.unsqueeze(-2)).mean(dim=0).flatten(-2, -1)

        z = self.out_proj(o)

        return z
