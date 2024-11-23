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
        msa_embed_dim: int,
        pair_embed_dim: int,
        max_relative_residue_dist: int = 32,
    ) -> None:
        super().__init__()
        target_dim = AminoAcidVocab.vocab_size
        self.linear_pair_embed_target1 = nn.Linear(target_dim, pair_embed_dim)
        self.linear_pair_embed_target2 = nn.Linear(target_dim, pair_embed_dim)

        residue_dist_bins = torch.arange(
            -max_relative_residue_dist, max_relative_residue_dist + 1
        ).float()
        self.relpos = RelPos(pair_embed_dim, residue_dist_bins)

        self.linear_msa_embed_msa = nn.Linear(MSA.N_MSA_FEATS, msa_embed_dim)
        self.linear_msa_embed_target = nn.Linear(target_dim, msa_embed_dim)

    def forward(
        self,
        target_feat: torch.Tensor,
        residue_index: torch.Tensor,
        msa_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.linear_pair_embed_target1(target_feat)
        b = self.linear_pair_embed_target2(target_feat)
        pair_rep = a.unsqueeze(1) + b.unsqueeze(0)
        pair_rep = pair_rep + self.relpos(residue_index)

        msa_rep = self.linear_msa_embed_msa(msa_feat) + self.linear_msa_embed_target(
            target_feat
        )

        return msa_rep, pair_rep


class RecyclingEmbedder(nn.Module):
    def __init__(
        self, msa_embed_dim: int, pair_embed_dim: int, bins: torch.Tensor
    ) -> None:
        super().__init__()
        n_bins = len(bins)
        self.one_hot_cb_dist = OneHotNearestBin(bins)
        self.linear_pair_embed_bins = nn.Linear(n_bins, pair_embed_dim)
        self.layer_norm_pair = nn.LayerNorm((pair_embed_dim,))
        self.layer_norm_msa = nn.LayerNorm((msa_embed_dim,))

    def forward(
        self,
        target_msa_rep: torch.Tensor,
        pair_rep: torch.Tensor,
        prev_struct_rep_cb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed CB atom distance information
        cb_dist_matrix = torch.cdist(prev_struct_rep_cb, prev_struct_rep_cb, p=2)
        pair_embed = self.linear_pair_embed_bins(self.one_hot_cb_dist(cb_dist_matrix))

        # Embed Evoformer outputs
        new_pair_rep = pair_embed + self.layer_norm_pair(pair_rep)
        new_target_msa_rep = self.layer_norm_msa(target_msa_rep)

        return new_target_msa_rep, new_pair_rep
