import torch
import torch.nn as nn


class OneHotNearestBin(nn.Module):
    def __init__(self, bins: torch.Tensor) -> None:
        super().__init__()
        self.bins = bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bins = self.bins.shape[0]
        p = torch.zeros((x.numel(), n_bins), device=x.device)
        p[torch.argmin(torch.abs(x.view(-1, 1) - self.bins.unsqueeze(0)))] = 1
        return p.view(*x.shape, n_bins)
