import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from xfold.model.common.utils import (
    DropoutColumnwise,
    DropoutRowwise,
    LinearNoBias,
    LinearSigmoid,
    OuterProductMean,
)
from xfold.model.common.triangle_ops import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)


class TransitionBase(nn.Module):
    def __init__(self, input_dim: int, dim_scale: int):
        super().__init__()
        expanded_dim = dim_scale * input_dim
        self.layer_norm = nn.LayerNorm(input_dim)
        self.expand_dim = nn.Linear(input_dim, expanded_dim)
        self.out_proj = nn.Sequential(nn.ReLU(), nn.Linear(expanded_dim, input_dim))

    def forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
        pair_rep = self.layer_norm(pair_rep)
        pair_rep = self.out_proj(self.expand_dim(pair_rep))
        return pair_rep


PairTransition = TransitionBase
MSATransition = TransitionBase


class MSARowAttentionWithPairBias(nn.Module):
    def __init__(
        self, msa_dim: int, pair_dim: int, proj_dim: int, n_heads: int
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.proj_dim = proj_dim
        self.inv_sqrt_dim = 1 / (proj_dim**0.5)

        self.layer_norm = nn.LayerNorm(msa_dim)
        self.to_qkv = LinearNoBias(msa_dim, 3 * n_heads * proj_dim)
        self.to_b = nn.Sequential(
            nn.LayerNorm(pair_dim), LinearNoBias(pair_dim, n_heads)
        )
        self.to_g = LinearSigmoid(msa_dim, n_heads * proj_dim)

        self.out_proj = nn.Linear(n_heads * proj_dim, msa_dim)

    def forward(self, msa_rep: torch.Tensor, pair_rep: torch.Tensor) -> torch.Tensor:
        qkv_shape = (*msa_rep.shape[:-1], self.proj_dim, self.n_heads)
        msa_rep = self.layer_norm(msa_rep)

        qkv = self.to_qkv(msa_rep).chunk(3, dim=-1)
        q, k, v = map(lambda x: x.view(qkv_shape), qkv)
        b = self.to_b(pair_rep).unsqueeze(-2).unsqueeze(0)
        g = self.to_g(msa_rep).view(qkv_shape)

        a = F.softmax(
            self.inv_sqrt_dim * torch.einsum("sidh,sjdh->sijdh", q, k) + b, dim=2
        )

        out = g * torch.einsum("sijdh,sjdh->sidh", a, v)
        out = self.out_proj(out.flatten(-2, -1))

        return out


class MSAColumnAttention(nn.Module):
    def __init__(self, msa_dim: int, proj_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.proj_dim = proj_dim
        self.inv_sqrt_dim = 1 / (proj_dim**0.5)

        self.layer_norm = nn.LayerNorm(msa_dim)
        self.to_qkv = LinearNoBias(msa_dim, 3 * n_heads * proj_dim)
        self.to_g = LinearSigmoid(msa_dim, n_heads * proj_dim)

        self.out_proj = nn.Linear(n_heads * proj_dim, msa_dim)

    def forward(self, msa_rep: torch.Tensor) -> torch.Tensor:
        qkv_shape = (*msa_rep.shape[:-1], self.proj_dim, self.n_heads)
        msa_rep = self.layer_norm(msa_rep)

        qkv = self.to_qkv(msa_rep).chunk(3, dim=-1)
        q, k, v = map(lambda x: x.view(qkv_shape), qkv)
        g = self.to_g(msa_rep).view(qkv_shape)

        a = F.softmax(self.inv_sqrt_dim * torch.einsum("sidh,tidh->stidh", q, k), dim=1)

        # Typo in paper? should be a^h_{sti} v^h_{ti} instead of a^h_{sti} v^h_{st}
        out = g * torch.einsum("stidh,tidh->sidh", a, v)
        out = self.out_proj(out.flatten(-2, -1))

        return out


class EvoformerBlock(nn.Module):
    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        msa_row_attn_dim: int,
        msa_col_attn_dim: int,
        outer_prod_msa_dim: int,
        n_tri_attn_heads: int,
        n_msa_attn_heads: int,
        msa_trans_dim_scale: int,
        pair_trans_dim_scale: int,
    ) -> None:
        super().__init__()
        # MSA stack
        self.msa_pair_row_attn = MSARowAttentionWithPairBias(
            msa_dim, pair_dim, msa_row_attn_dim, n_msa_attn_heads
        )
        self.dropout_row = DropoutColumnwise(0.15)
        self.msa_global_attn = MSAColumnAttention(
            msa_dim, msa_col_attn_dim, n_msa_attn_heads
        )
        self.msa_trans = MSATransition(msa_dim, msa_trans_dim_scale)

        # Communication
        self.msa_to_pair = OuterProductMean(msa_dim, outer_prod_msa_dim, pair_dim)

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

    def forward(self, msa_rep: torch.Tensor, pair_rep: torch.Tensor) -> torch.Tensor:
        msa_rep = msa_rep + self.dropout_row(self.msa_pair_row_attn(msa_rep, pair_rep))
        msa_rep = msa_rep + self.msa_global_attn(msa_rep)
        msa_rep = msa_rep + self.msa_trans(msa_rep)

        pair_rep = pair_rep + self.msa_to_pair(msa_rep)

        for update_pair in self.update_pair_layers:
            pair_rep = pair_rep + update_pair(pair_rep)

        return msa_rep, pair_rep


class EvoformerStack(nn.Module):
    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        single_dim: int,
        msa_row_attn_dim: int,
        msa_col_attn_dim: int,
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
                EvoformerBlock(
                    msa_dim,
                    pair_dim,
                    msa_row_attn_dim,
                    msa_col_attn_dim,
                    outer_prod_msa_dim,
                    n_tri_attn_heads,
                    n_msa_attn_heads,
                    msa_trans_dim_scale,
                    pair_trans_dim_scale,
                )
                for _ in range(n_blocks)
            ]
        )
        self.output_layer = nn.Linear(msa_dim, single_dim)

    def forward(
        self, msa_rep: torch.Tensor, pair_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        TARGET_SEQ_INDEX = 0
        for update_msa_pair in self.update_msa_pair_layers:
            msa_rep, pair_rep = update_msa_pair(msa_rep, pair_rep)

        # Get single representation for target sequence
        single_rep = self.output_layer(msa_rep[TARGET_SEQ_INDEX])

        return msa_rep, pair_rep, single_rep
