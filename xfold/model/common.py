from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


ROW_DIM = 0
COL_DIM = 1


class AxialDropout2D(nn.Module):
    def __init__(self, p: float, axis: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dropout_shape = [1, 1, 1]
        dropout_shape[self.axis] = x.shape[self.axis]
        return x * self.dropout(torch.ones(dropout_shape, device=x.device))


DropoutRowwise = partial(AxialDropout2D, axis=ROW_DIM)
DropoutColumnwise = partial(AxialDropout2D, axis=COL_DIM)

LinearNoBias = partial(nn.Linear, bias=False)


class LinearSigmoid(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))


class OneHotNearestBin(nn.Module):
    def __init__(self, bins: torch.Tensor) -> None:
        super().__init__()
        self.bins = bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bins = self.bins.shape[0]
        p = torch.zeros((x.numel(), n_bins), device=x.device)
        p[torch.argmin(torch.abs(x.view(-1, 1) - self.bins.unsqueeze(0)))] = 1
        return p.view(*x.shape, n_bins)


class AttentionAround(nn.Module):
    #     def forward(self, q, k, v) -> torch.Tensor:
    pass


class TriangleAttentionBase(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, n_heads: int = 1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.proj_dim = proj_dim
        self.inv_sqrt_proj_dim = 1 / (proj_dim**0.5)

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
        mat_shape = (*x.shape[:2], self.proj_dim, self.n_heads)

        # Project inputs
        x = self.layer_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda u: u.view(mat_shape), qkv)
        b = self.to_b(x).view(mat_shape)
        g = self.to_g(x).view(mat_shape)

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
            self.inv_sqrt_proj_dim * torch.einsum("ijdh,ikdh->ijkdh", q, k)
            + b.unsqueeze(0),
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
            self.inv_sqrt_proj_dim * torch.einsum("ijdh,kjdh->ijkdh", q, k)
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
