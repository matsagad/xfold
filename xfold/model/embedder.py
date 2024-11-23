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
        self, msa_embed_dim: int, pair_embed_dim: int, max_residue_distance: int = 32
    ) -> None:
        super().__init__()
        residue_distance_bins = torch.arange(
            -max_residue_distance, max_residue_distance + 1
        ).float()

        self.linear_embed_target_to_pair1 = nn.Linear(
            AminoAcidVocab.vocab_size, pair_embed_dim
        )
        self.linear_embed_target_to_pair2 = nn.Linear(
            AminoAcidVocab.vocab_size, pair_embed_dim
        )
        self.relpos = RelPos(pair_embed_dim, residue_distance_bins)
        self.linear_embed_msa_to_msa = nn.Linear(MSA.N_MSA_FEATS, msa_embed_dim)
        self.linear_embed_target_to_msa = nn.Linear(
            AminoAcidVocab.vocab_size, msa_embed_dim
        )

    def forward(
        self,
        target_feat: torch.Tensor,
        residue_index: torch.Tensor,
        msa_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.linear_embed_target_to_pair1(target_feat)
        b = self.linear_embed_target_to_pair2(target_feat)
        pair_rep = a.unsqueeze(1) + b.unsqueeze(0)
        pair_rep = pair_rep + self.relpos(residue_index)

        msa_rep = self.linear_embed_msa_to_msa(
            msa_feat
        ) + self.linear_embed_target_to_msa(target_feat)

        return msa_rep, pair_rep
