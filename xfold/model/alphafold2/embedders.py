import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from xfold.model.common.misc import (
    DropoutColumnwise,
    DropoutRowwise,
    LinearNoBias,
    LinearSigmoid,
    OuterProductMean,
    RelPos,
)
from xfold.model.common.triangle_ops import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from xfold.model.alphafold2.evoformer import (
    PairTransition,
    MSATransition,
    MSARowAttentionWithPairBias,
)
from xfold.protein.sequence import AminoAcidVocab, MSA
from xfold.protein.structure import TemplateProtein


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
        res_index: torch.Tensor,
        msa_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed target sequence and relative residue positions
        pair1 = self.pair_proj1(target_feat)
        pair2 = self.pair_proj2(target_feat)
        pair_rep = pair1.unsqueeze(1) + pair2.unsqueeze(0)
        pair_rep = pair_rep + self.relpos(res_index)

        # Embed MSA and target sequence
        msa_rep = self.msa_proj(msa_feat) + self.target_proj(target_feat)

        return msa_rep, pair_rep


class TemplatePairBlock(nn.Module):
    def __init__(self, temp_dim: int, n_heads: int, pair_trans_dim_scale: int) -> None:
        super().__init__()
        self.update_pair_layers = nn.ModuleList(
            [
                nn.Sequential(
                    TriangleAttentionStartingNode(temp_dim, temp_dim, n_heads),
                    DropoutRowwise(0.25),
                ),
                nn.Sequential(
                    TriangleAttentionEndingNode(temp_dim, temp_dim, n_heads),
                    DropoutColumnwise(0.25),
                ),
                nn.Sequential(
                    TriangleMultiplicationOutgoing(temp_dim, temp_dim),
                    DropoutRowwise(0.25),
                ),
                nn.Sequential(
                    TriangleMultiplicationIncoming(temp_dim, temp_dim),
                    DropoutRowwise(0.25),
                ),
                PairTransition(temp_dim, pair_trans_dim_scale),
            ]
        )

    def forward(self, temp_rep: torch.Tensor) -> torch.Tensor:
        for update_pair in self.update_pair_layers:
            temp_rep = temp_rep + update_pair(temp_rep)
        return temp_rep


class TemplatePairStack(nn.Module):
    def __init__(
        self,
        temp_dim: int,
        n_heads: int,
        n_blocks: int,
        pair_trans_dim_scale: int,
    ) -> None:
        super().__init__()
        self.update_pair_layers = nn.ModuleList(
            [
                TemplatePairBlock(temp_dim, n_heads, pair_trans_dim_scale)
                for _ in range(n_blocks)
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
        self.inv_sqrt_dim = 1 / (temp_dim**0.5)

        self.to_q = LinearNoBias(pair_dim, n_heads * temp_dim)
        self.to_kv = LinearNoBias(temp_dim, 2 * n_heads * temp_dim)

        self.out_proj = nn.Linear(n_heads * temp_dim, pair_dim)

    def forward(self, pair_rep: torch.Tensor, temp_rep: torch.Tensor) -> torch.Tensor:
        pair_proj_shape = (*pair_rep.shape[:-1], self.temp_dim, self.n_heads)
        temp_proj_shape = (*temp_rep.shape[:-1], self.temp_dim, self.n_heads)

        q = self.to_q(pair_rep).view(pair_proj_shape)
        kv = self.to_kv(temp_rep).chunk(2, dim=-1)
        k, v = map(lambda u: u.view(temp_proj_shape), kv)

        a = F.softmax(self.inv_sqrt_dim * torch.einsum("ijdh,sijdh->sijh", q, k), dim=0)
        out = torch.einsum("sijh,sijdh->ijdh", a, v)
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
        n_pw_attn_heads: int,
        n_temp_pair_blocks: int,
        pair_trans_dim_scale: int,
    ) -> None:
        super().__init__()
        self.angle_feat_proj = nn.Sequential(
            nn.Linear(TemplateProtein.N_TEMP_ANGLE_FEATS, msa_dim),
            nn.ReLU(),
            nn.Linear(msa_dim, msa_dim),
        )
        self.pair_feat_proj = nn.Linear(TemplateProtein.N_TEMP_PAIR_FEATS, temp_dim)
        self.temp_stack = TemplatePairStack(
            temp_dim,
            n_tri_attn_heads,
            n_temp_pair_blocks,
            pair_trans_dim_scale,
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


class MSAColumnGlobalAttention(nn.Module):
    def __init__(self, msa_dim: int, proj_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.proj_dim = proj_dim
        self.inv_sqrt_dim = 1 / (proj_dim**0.5)

        self.layer_norm = nn.LayerNorm(msa_dim)
        self.to_qkv = LinearNoBias(msa_dim, (n_heads + 2) * proj_dim)
        self.to_g = LinearSigmoid(msa_dim, n_heads * proj_dim)

        self.out_proj = nn.Linear(n_heads * proj_dim, msa_dim)

    def forward(self, msa_rep: torch.Tensor) -> torch.Tensor:
        q_shape = (*msa_rep.shape[:-1], self.proj_dim, self.n_heads)
        msa_rep = self.layer_norm(msa_rep)

        q, k, v = self.to_qkv(msa_rep).split(
            (self.n_heads * self.proj_dim, self.proj_dim, self.proj_dim), dim=-1
        )
        q = q.view(q_shape).mean(dim=0)
        g = self.to_g(msa_rep).view(q_shape)

        a = F.softmax(self.inv_sqrt_dim * torch.einsum("idh,tid->tih", q, k), dim=0)

        out = g * torch.einsum("tih,tid->idh", a, v).unsqueeze(0)
        out = self.out_proj(out.flatten(-2, -1))

        return out


class ExtraMSABlock(nn.Module):
    def __init__(
        self,
        extra_msa_dim: int,
        pair_dim: int,
        msa_row_attn_dim: int,
        msa_col_global_attn_dim: int,
        outer_prod_msa_dim: int,
        n_tri_attn_heads: int,
        n_msa_attn_heads: int,
        msa_trans_dim_scale: int,
        pair_trans_dim_scale: int,
    ) -> None:
        super().__init__()
        # MSA stack
        self.msa_pair_row_attn = MSARowAttentionWithPairBias(
            extra_msa_dim, pair_dim, msa_row_attn_dim, n_msa_attn_heads
        )
        self.dropout_row = DropoutColumnwise(0.15)
        self.msa_global_attn = MSAColumnGlobalAttention(
            extra_msa_dim, msa_col_global_attn_dim, n_msa_attn_heads
        )
        self.msa_trans = MSATransition(extra_msa_dim, msa_trans_dim_scale)

        # Communication
        self.msa_to_pair = OuterProductMean(extra_msa_dim, outer_prod_msa_dim, pair_dim)

        # Pair stack
        self.update_pair_layers = nn.ModuleList(
            [
                nn.Sequential(
                    TriangleMultiplicationOutgoing(pair_dim, pair_dim),
                    DropoutRowwise(0.25),
                ),
                nn.Sequential(
                    TriangleMultiplicationIncoming(pair_dim, pair_dim),
                    DropoutRowwise(0.25),
                ),
                nn.Sequential(
                    TriangleAttentionStartingNode(pair_dim, pair_dim, n_tri_attn_heads),
                    DropoutRowwise(0.25),
                ),
                nn.Sequential(
                    TriangleAttentionEndingNode(pair_dim, pair_dim, n_tri_attn_heads),
                    DropoutColumnwise(0.25),
                ),
                PairTransition(pair_dim, pair_trans_dim_scale),
            ]
        )

    def forward(
        self, extra_msa_rep: torch.Tensor, pair_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        extra_msa_rep = extra_msa_rep + self.dropout_row(
            self.msa_pair_row_attn(extra_msa_rep, pair_rep)
        )
        extra_msa_rep = extra_msa_rep + self.msa_global_attn(extra_msa_rep)
        extra_msa_rep = extra_msa_rep + self.msa_trans(extra_msa_rep)

        pair_rep = pair_rep + self.msa_to_pair(extra_msa_rep)

        for update_pair in self.update_pair_layers:
            pair_rep = pair_rep + update_pair(pair_rep)

        return extra_msa_rep, pair_rep


class ExtraMSAStack(nn.Module):
    def __init__(
        self,
        extra_msa_dim: int,
        pair_dim: int,
        msa_row_attn_dim: int,
        msa_col_global_attn_dim: int,
        outer_prod_msa_dim: int,
        n_tri_attn_heads: int,
        n_msa_attn_heads: int,
        msa_trans_dim_scale: int,
        pair_trans_dim_scale: int,
        n_blocks: int,
    ) -> None:
        super().__init__()
        self.update_msa_pair_layers = nn.ModuleList(
            [
                ExtraMSABlock(
                    extra_msa_dim,
                    pair_dim,
                    msa_row_attn_dim,
                    msa_col_global_attn_dim,
                    outer_prod_msa_dim,
                    n_tri_attn_heads,
                    n_msa_attn_heads,
                    msa_trans_dim_scale,
                    pair_trans_dim_scale,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self, extra_msa_rep: torch.Tensor, pair_rep: torch.Tensor
    ) -> torch.Tensor:
        for update_msa_pair in self.update_msa_pair_layers:
            extra_msa_rep, pair_rep = update_msa_pair(extra_msa_rep, pair_rep)
        return pair_rep


class ExtraMSAEmbedder(nn.Module):
    def __init__(
        self,
        extra_msa_dim: int,
        pair_dim: int,
        msa_row_attn_dim: int,
        msa_col_global_attn_dim: int,
        outer_prod_msa_dim: int,
        n_tri_attn_heads: int,
        n_msa_attn_heads: int,
        msa_trans_dim_scale: int,
        pair_trans_dim_scale: int,
        n_blocks: int,
    ) -> None:
        super().__init__()
        self.extra_msa_proj = nn.Linear(MSA.N_EXTRA_MSA_FEATS, extra_msa_dim)
        self.extra_msa_stack = ExtraMSAStack(
            extra_msa_dim,
            pair_dim,
            msa_row_attn_dim,
            msa_col_global_attn_dim,
            outer_prod_msa_dim,
            n_tri_attn_heads,
            n_msa_attn_heads,
            msa_trans_dim_scale,
            pair_trans_dim_scale,
            n_blocks,
        )

    def forward(
        self, extra_msa_feat: torch.Tensor, pair_rep: torch.Tensor
    ) -> torch.Tensor:
        extra_msa_rep = self.extra_msa_proj(extra_msa_feat)
        pair_rep = self.extra_msa_stack(extra_msa_rep, pair_rep)
        return pair_rep
