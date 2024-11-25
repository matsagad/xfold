import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from xfold.model.common import (
    DropoutColumnwise,
    DropoutRowwise,
    LinearNoBias,
    OneHotNearestBin,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from xfold.protein.sequence import AminoAcidVocab, MSA
from xfold.protein.structure import TemplateProtein


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


class PairTransition(nn.Module):
    def __init__(self, input_dim: int, dim_scale_factor: int):
        super().__init__()
        expanded_dim = dim_scale_factor * input_dim
        self.layer_norm = nn.LayerNorm(input_dim)
        self.expand_dim = nn.Linear(input_dim, expanded_dim)
        self.out_proj = nn.Sequential(nn.ReLU(), nn.Linear(expanded_dim, input_dim))

    def forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
        pair_rep = self.layer_norm(pair_rep)
        pair_rep = self.out_proj(self.expand_dim(pair_rep))
        return pair_rep


class TemplatePairStack(nn.Module):
    def __init__(
        self,
        temp_dim: int,
        n_heads: int,
        n_blocks: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.update_pair_layers = nn.ModuleList()

        for _ in range(n_blocks):
            self.update_pair_layers.extend(
                [
                    nn.Sequential(
                        TriangleAttentionStartingNode(temp_dim, temp_dim, n_heads),
                        DropoutRowwise(p_dropout),
                    ),
                    nn.Sequential(
                        TriangleAttentionEndingNode(temp_dim, temp_dim, n_heads),
                        DropoutColumnwise(p_dropout),
                    ),
                    nn.Sequential(
                        TriangleMultiplicationOutgoing(temp_dim, temp_dim),
                        DropoutRowwise(p_dropout),
                    ),
                    nn.Sequential(
                        TriangleMultiplicationIncoming(temp_dim, temp_dim),
                        DropoutRowwise(p_dropout),
                    ),
                    PairTransition(temp_dim, 2),
                ]
            )

        self.layer_norm = nn.LayerNorm(temp_dim)

    def forward(self, temp_rep: torch.Tensor) -> torch.Tensor:
        for update_pair in self.update_pair_layers:
            temp_rep = temp_rep + update_pair(temp_rep)
        temp_rep = self.layer_norm(temp_rep)
        return temp_rep


class TemplatePointwiseAttention(nn.Module):
    def __init__(self, pair_dim: int, temp_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.temp_dim = temp_dim
        self.inv_sqrt_temp_dim = 1 / (temp_dim**0.5)

        self.to_q = LinearNoBias(pair_dim, n_heads * temp_dim)
        self.to_kv = LinearNoBias(temp_dim, 2 * n_heads * temp_dim)

        self.out_proj = nn.Linear(n_heads * temp_dim, pair_dim)

    def forward(self, pair_rep: torch.Tensor, temp_rep: torch.Tensor) -> torch.Tensor:
        pair_proj_shape = (*pair_rep.shape[:-1], self.temp_dim, self.n_heads)
        temp_proj_shape = (*temp_rep.shape[:-1], self.temp_dim, self.n_heads)

        q = self.to_q(pair_rep).view(pair_proj_shape)
        kv = self.to_kv(temp_rep).chunk(2, dim=-1)
        k, v = map(lambda u: u.view(temp_proj_shape), kv)

        a = F.softmax(
            self.inv_sqrt_temp_dim * torch.einsum("ijdh,sijdh->sijdh", q, k), dim=0
        )
        out = torch.einsum("sijdh,sijdh->ijdh", a, v)
        pair_update = self.out_proj(out.flatten(-2, -1))

        pair_rep = pair_rep + pair_update

        return pair_rep


class TemplateEmbedder(nn.Module):
    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        temp_dim: int,
        n_tri_attn_heads: int,
        pair_trans_dim_scale_factor: int,
        n_pw_attn_heads: int,
        p_dropout: float,
    ) -> None:
        super().__init__()

        self.angle_feat_proj = nn.Sequential(
            nn.Linear(TemplateProtein.N_TEMP_ANGLE_FEATS, msa_dim),
            nn.ReLU(),
            nn.Linear(msa_dim, msa_dim),
        )
        self.pair_feat_proj = nn.Linear(TemplateProtein.N_TEMP_PAIR_FEATS, temp_dim)
        self.temp_stack = TemplatePairStack(
            temp_dim, n_tri_attn_heads, pair_trans_dim_scale_factor, p_dropout
        )
        self.temp_pw_attn = TemplatePointwiseAttention(
            pair_dim, temp_dim, n_pw_attn_heads
        )

    def forward(
        self,
        msa_rep: torch.Tensor,
        pair_rep: torch.Tensor,
        temp_angle_feats: torch.Tensor,
        temp_pair_feats: torch.Tensor,
    ) -> torch.Tensor:
        msa_rep_angle = self.angle_feat_proj(temp_angle_feats)
        msa_rep = torch.cat([msa_rep, msa_rep_angle], dim=0)

        temp_rep = self.pair_feat_proj(temp_pair_feats)
        for i in range(len(temp_rep)):
            temp_rep[i] = temp_rep[i] + self.temp_stack(temp_rep[i])

        pair_rep = self.temp_pw_attn(pair_rep, temp_rep)
        return msa_rep, pair_rep
