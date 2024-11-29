import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from xfold.model.common.misc import OneHotNearestBin


class PLDDTHead(nn.Module):
    def __init__(self, single_dim: int, proj_dim: int, bins: torch.Tensor) -> None:
        super().__init__()
        n_bins = len(bins)
        self.bins = bins
        self.single_proj = nn.Sequential(
            nn.LayerNorm(single_dim),
            nn.Linear(single_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
        )
        self.to_p_plddt = nn.Sequential(nn.Linear(proj_dim, n_bins), nn.Softmax(dim=-1))
        self.to_p_lddt = OneHotNearestBin(bins)

    def forward(
        self, single_rep: torch.Tensor, true_lddt: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        a = self.single_proj(single_rep)
        p_plddt = self.to_p_plddt(a)
        plddt = p_plddt @ self.bins.unsqueeze(-1)

        if true_lddt is None:
            return plddt

        p_true_lddt = self.to_p_lddt(true_lddt)
        L_conf = F.cross_entropy(p_plddt, p_true_lddt).mean()
        return plddt, L_conf
