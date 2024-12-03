import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from xfold.model.common.misc import OneHotNearestBin
from xfold.protein.constants import MSAVocab, MSA_MASKED_SYMBOL


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


class TMScoreHead(nn.Module):
    def __init__(self, pair_dim: int, bins: torch.Tensor) -> None:
        super().__init__()
        self.bins = bins
        n_bins = len(bins)
        self.to_p_error_dist = nn.Sequential(
            nn.Linear(pair_dim, n_bins), nn.Softmax(dim=-1)
        )

    def tm_f(self, d: torch.Tensor) -> torch.Tensor:
        N_RES = len(d)
        d_0 = 1.24 * (max(N_RES, 19) - 15) ** (1 / 3) - 1.8
        return 1 / (1 + (d / d_0) ** 2)

    def forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
        # e_ij = || T_i^{-1} o x_j  -  T_i^{true, -1} o x_j^{true} ||
        p_error_dist = self.to_p_error_dist(pair_rep)
        p_tm = (
            (self.tm_f(self.bins).view(1, 1, -1) * p_error_dist)
            .sum(dim=-1)
            .mean(dim=1)
            .max()
        )
        return p_tm


class DistogramHead(nn.Module):
    def __init__(self, pair_dim: int, bins: torch.Tensor) -> None:
        super().__init__()
        self.bins = bins
        n_bins = len(bins)
        self.to_p_dist = nn.Sequential(nn.Linear(pair_dim, n_bins), nn.Softmax(dim=-1))
        self.cb_bin = OneHotNearestBin(bins)

    def forward(self, pair_rep: torch.Tensor, true_cb: torch.Tensor) -> torch.Tensor:
        N_RES = len(true_cb)
        p_dist = self.to_p_dist(pair_rep + pair_rep.transpose(0, 1))
        true_dist = self.cb_bin(torch.cdist(true_cb, true_cb, p=2))
        L_dist = -F.cross_entropy(p_dist, true_dist, reduction="sum") / (N_RES * N_RES)
        return L_dist


class MaskedMSAHead(nn.Module):
    def __init__(self, msa_dim: int) -> None:
        super().__init__()
        self.to_p_msa = nn.Sequential(
            nn.Linear(msa_dim, MSAVocab.vocab_size), nn.Softmax(dim=-1)
        )

    def forward(self, msa_rep: torch.Tensor, true_msa: torch.Tensor) -> torch.Tensor:
        is_masked = true_msa[:, :, MSAVocab.get_index(MSA_MASKED_SYMBOL)] == 1
        N_MASK = is_masked.sum()
        p_msa = self.to_p_msa(msa_rep)
        L_msa = (
            -F.cross_entropy(p_msa[is_masked], true_msa[is_masked], reduction="sum")
            / N_MASK
        )
        return L_msa
