import torch
import torch.nn as nn
from typing import Tuple
from xfold.model.common.misc import OneHotNearestBin


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
        target_msa_rep: torch.Tensor,
        pair_rep: torch.Tensor,
        prev_struct_cb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed CB atom distance information
        cb_dists = torch.cdist(prev_struct_cb, prev_struct_cb, p=2)
        pair_rep_cb = self.cb_proj(self.cb_bin(cb_dists))

        # Update representations with recycled information
        pair_update = pair_rep_cb + self.layer_norm_pair(pair_rep)
        target_msa_update = self.layer_norm_msa(target_msa_rep)

        return target_msa_update, pair_update
