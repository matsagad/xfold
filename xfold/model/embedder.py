import torch
import torch.nn as nn
from typing import Tuple
from xfold.model.common import OneHotNearestBin
from xfold.protein.sequence import AminoAcidVocab, MSA


class RelPos(nn.Module):
    def __init__(self, embed_dim: int, bins: torch.Tensor) -> None:
        super().__init__()
        self.one_hot = OneHotNearestBin(bins)
        self.linear = nn.Linear(bins.shape[0], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_pos = x.unsqueeze(1) - x.unsqueeze(0)
        p = self.linear(self.one_hot(d_pos))
        return p


class InputEmbedder(nn.Module):
    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        max_relpos_dist: int = 32,
    ) -> None:
        super().__init__()
        target_dim = AminoAcidVocab.vocab_size

        # Pair embedding layers
        self.pair_proj1 = nn.Linear(target_dim, pair_dim)
        self.pair_proj2 = nn.Linear(target_dim, pair_dim)

        dist_bins = torch.arange(-max_relpos_dist, max_relpos_dist + 1).float()
        self.relpos = RelPos(pair_dim, dist_bins)

        # MSA embedding layers
        self.msa_proj = nn.Linear(MSA.N_MSA_FEATS, msa_dim)
        self.target_proj = nn.Linear(target_dim, msa_dim)

    def forward(
        self,
        target_feat: torch.Tensor,
        residue_idx: torch.Tensor,
        msa_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed target sequence and relative residue positions
        pair1 = self.pair_proj1(target_feat)
        pair2 = self.pair_proj2(target_feat)
        pair_rep = pair1.unsqueeze(1) + pair2.unsqueeze(0)
        pair_rep = pair_rep + self.relpos(residue_idx)

        # Embed MSA and target sequence
        msa_rep = self.msa_proj(msa_feat) + self.target_proj(target_feat)

        return msa_rep, pair_rep


class RecyclingEmbedder(nn.Module):
    def __init__(self, msa_dim: int, pair_dim: int, bins: torch.Tensor) -> None:
        super().__init__()
        n_bins = len(bins)
        self.cb_bin = OneHotNearestBin(bins)
        self.cb_proj = nn.Linear(n_bins, pair_dim)
        self.layer_norm_pair = nn.LayerNorm((pair_dim,))
        self.layer_norm_msa = nn.LayerNorm((msa_dim,))

    def forward(
        self,
        msa_rep: torch.Tensor,
        pair_rep: torch.Tensor,
        prev_struct_cb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_msa_rep = msa_rep[0]

        # Embed CB atom distance information
        cb_dists = torch.cdist(prev_struct_cb, prev_struct_cb, p=2)
        pair_rep_cb = self.cb_proj(self.cb_bin(cb_dists))

        # Update representations with recycled information
        pair_update = pair_rep_cb + self.layer_norm_pair(pair_rep)
        target_msa_update = self.layer_norm_msa(target_msa_rep)

        pair_rep = pair_rep + pair_update
        msa_rep[0] = msa_rep[0] + target_msa_update

        return msa_rep, pair_rep
