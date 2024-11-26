import torch
import torch.nn as nn
import torch.nn.functional as F
from xfold.model.common.utils import LinearNoBias, LinearSigmoid


class TriangleAttentionBase(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, n_heads: int = 1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.proj_dim = proj_dim
        self.inv_sqrt_dim = 1 / (proj_dim**0.5)

        self.layer_norm = nn.LayerNorm(input_dim)
        self.to_qkv = LinearNoBias(input_dim, 3 * n_heads * proj_dim)
        self.to_b = LinearNoBias(input_dim, n_heads * proj_dim)
        self.to_g = LinearSigmoid(input_dim, n_heads * proj_dim)
        self.out_proj = nn.Linear(n_heads * proj_dim, input_dim)

    def compute_gated_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_RES, N_RES, N_FEATS]
        qkv_shape = (*x.shape[:2], self.proj_dim, self.n_heads)

        # Project inputs
        x = self.layer_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda u: u.view(qkv_shape), qkv)
        b = self.to_b(x).view(qkv_shape)
        g = self.to_g(x).view(qkv_shape)

        # Compute attention
        out = self.compute_gated_attention(q, k, v, b, g)
        z = self.out_proj(out.flatten(-2, -1))

        return z


class TriangleAttentionStartingNode(TriangleAttentionBase):
    def compute_gated_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        ## a_ijk := softmax_k( 1/sqrt(c) * q_ij^T k_ik + b_jk )
        a = F.softmax(
            self.inv_sqrt_dim * torch.einsum("ijdh,ikdh->ijkdh", q, k) + b.unsqueeze(0),
            dim=2,
        )
        ## o_ij := g_ij \odot \sum_k a_ijk * v_ik
        out = g * torch.einsum("ijkdh,ikdh->ijdh", a, v)
        return out


class TriangleAttentionEndingNode(TriangleAttentionBase):
    def compute_gated_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        ## a_ijk := softmax_k( 1/sqrt(c) * q_ij^T k_kj + b_ki )
        a = F.softmax(
            self.inv_sqrt_dim * torch.einsum("ijdh,kjdh->ijkdh", q, k)
            + torch.einsum("iokdh->koidh", b.unsqueeze(1)),
            dim=2,
        )
        ## o_ij := g_ij \odot \sum_k{ a_ijk * v_kj }
        out = g * torch.einsum("ijkdh,kjdh->ijdh", a, v)
        return out


class TriangleMultiplicationBase(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.weight_proj1 = LinearSigmoid(input_dim, proj_dim)
        self.edge_proj1 = nn.Linear(input_dim, proj_dim)
        self.weight_proj2 = LinearSigmoid(input_dim, proj_dim)
        self.edge_proj2 = nn.Linear(input_dim, proj_dim)

        self.to_g = LinearSigmoid(input_dim, proj_dim)
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(proj_dim), nn.Linear(proj_dim, input_dim)
        )

    def combine_edge_projections(
        self, edge1: torch.Tensor, edge2: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        edge1 = self.weight_proj1(x) * self.edge_proj1(x)
        edge2 = self.weight_proj2(x) * self.edge_proj2(x)

        g = self.to_g(x)
        z = g * self.pair_proj(self.combine_edge_projections(edge1, edge2))

        return z


class TriangleMultiplicationOutgoing(TriangleMultiplicationBase):
    def combine_edge_projections(
        self, edge1: torch.Tensor, edge2: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("ikd,jkd->ijd", edge1, edge2)


class TriangleMultiplicationIncoming(TriangleMultiplicationBase):
    def combine_edge_projections(
        self, edge1: torch.Tensor, edge2: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("kid,kjd->ijd", edge1, edge2)
