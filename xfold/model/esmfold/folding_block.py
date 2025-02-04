import torch
import torch.nn as nn
from typing import Tuple
from xfold.model.common.triangle_ops import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from xfold.model.common.misc import DropoutColumnwise, DropoutRowwise
from xfold.model.esmfold.lm import SelfAttention


class FoldingBlock(nn.Module):
    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        single_head_dim: int,
        pair_head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        n_single_heads = single_dim // single_head_dim
        n_pair_heads = pair_dim // pair_head_dim
        self.n_single_heads = n_single_heads

        self.to_b = nn.Linear(pair_dim, n_single_heads)
        self.attn = SelfAttention(single_dim, single_head_dim, n_single_heads)
        self.single_proj = nn.Linear(single_dim, single_dim)

        self.outer_proj = nn.Linear(2 * single_dim, pair_dim)

        # Authors apparently didn't find dropout to provide any benefits
        # See https://github.com/facebookresearch/esm/issues/268
        # (Paper only talks about dropout in ESM language model not folding trunk)
        self.update_pair_layers = nn.ModuleList(
            [
                nn.Sequential(
                    TriangleMultiplicationOutgoing(pair_dim, pair_dim),
                    DropoutRowwise(dropout),
                ),
                nn.Sequential(
                    TriangleMultiplicationIncoming(pair_dim, pair_dim),
                    DropoutColumnwise(dropout),
                ),
                nn.Sequential(
                    TriangleAttentionStartingNode(
                        pair_dim, pair_head_dim, n_pair_heads
                    ),
                    DropoutRowwise(dropout),
                ),
                nn.Sequential(
                    TriangleAttentionEndingNode(pair_dim, pair_head_dim, n_pair_heads),
                    DropoutColumnwise(dropout),
                ),
            ]
        )

    def forward(
        self, single_rep: torch.Tensor, pair_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Single representation
        b = self.to_b(pair_rep).view(*pair_rep.shape[:-1], self.n_single_heads)

        single_rep = single_rep + self.attn(single_rep, bias=b)
        single_rep = single_rep + self.single_proj(single_rep)

        # Communication
        outer_prod = single_rep.unsqueeze(1) * single_rep.unsqueeze(2)
        outer_diff = single_rep.unsqueeze(1) - single_rep.unsqueeze(2)
        pair_rep = pair_rep + self.outer_proj(
            torch.cat((outer_prod, outer_diff), dim=-1)
        )

        # Pair representation
        for update_pair in self.update_pair_layers:
            pair_rep = pair_rep + update_pair(pair_rep[0]).unsqueeze(0)

        return single_rep, pair_rep
