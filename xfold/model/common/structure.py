import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from xfold.protein.constants import (
    AA_ATOM14_MASK,
    AA_ATOM14_TO_RIGID_GROUP,
    AA_LIT_ATOM14_POS_4x1,
    AA_LIT_RIGID_TO_RIGID,
    AA_TORSION_NAMES,
)
from xfold.protein.structure import ProteinFrames, ProteinStructure
from xfold.model.common.utils import LinearNoBias


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        proj_dim: int,
        n_heads: int,
        n_query_points: int,
        n_point_values: int,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.proj_dim = proj_dim

        self.to_qkv = LinearNoBias(single_dim, 3 * n_heads * proj_dim)
        self.to_b = LinearNoBias(pair_dim, n_heads)

        self.to_qk_points = LinearNoBias(single_dim, 2 * n_heads * n_query_points * 3)
        self.to_v_points = LinearNoBias(single_dim, n_heads * n_point_values * 3)

        self.sqrt_inv_dim = 1 / (proj_dim**0.5)
        self.w_C = (2 / (9 * n_query_points)) ** 0.5
        self.w_L = (1 / 3) ** 0.5
        self.gamma = nn.Parameter(torch.zeros(n_heads))

        self.out_proj = nn.Linear(
            n_heads * (pair_dim + proj_dim + n_point_values * 3 + n_point_values),
            single_dim,
        )

    def forward(
        self, single_rep: torch.Tensor, pair_rep: torch.Tensor, frames: ProteinFrames
    ) -> torch.Tensor:
        qk_shape = (*single_rep.shape[:-1], self.proj_dim, self.n_heads)
        points_shape = (*single_rep.shape[:-1], -1, 3, self.n_heads)

        qkv = self.to_qkv(single_rep).chunk(3, dim=-1)
        q, k, v = map(lambda x: x.view(qk_shape), qkv)
        b = self.to_b(pair_rep)

        qk_points = self.to_qk_points(single_rep).chunk(2, dim=-1)
        q_points, k_points = map(lambda x: x.view(points_shape), qk_points)
        v_points = self.to_v_points(single_rep).view(points_shape)

        dp_attn_logits = self.sqrt_inv_dim * torch.einsum("idh,jdh->ijh", q, k) + b
        dist_logits = (
            (
                frames.apply_along_dim(q_points.movedim(-1, -2), dim=0).unsqueeze(1)
                - frames.apply_along_dim(k_points.movedim(-1, -2), dim=0).unsqueeze(0)
            )
            ** 2
        ).sum(dim=(-1, -3))
        dist_logits_scale = self.gamma * self.w_C / 2
        a = F.softmax(
            self.w_L * (dp_attn_logits - dist_logits_scale * dist_logits), dim=1
        )

        out_pair = torch.einsum("ijh,ijz->izh", a, pair_rep)
        out_value = torch.einsum("ijh,jdh->idh", a, v)
        out_points = frames.apply_inverse_along_dim(
            torch.einsum(
                "ijh,jpht->ipht",
                a,
                frames.apply_along_dim(v_points.movedim(-1, -2), dim=0),
            ),
            dim=0,
        )
        out_points_norm = torch.norm(out_points, dim=-1)

        out = torch.cat(
            [
                out_pair.flatten(-2, -1),
                out_value.flatten(-2, -1),
                out_points.flatten(-3, -1),
                out_points_norm.flatten(-2, -1),
            ],
            dim=1,
        )
        single_ipa = self.out_proj(out)

        return single_ipa


class BackboneUpdate(nn.Module):
    def __init__(self, single_dim: int) -> None:
        super().__init__()
        self.to_quaternion = nn.Linear(single_dim, 3)
        self.to_ts = nn.Linear(single_dim, 3)

    def forward(self, single_rep: torch.Tensor) -> ProteinFrames:
        N_RES = len(single_rep)

        bcd = self.to_quaternion(single_rep).transpose(0, 1)
        ts = self.to_ts(single_rep)

        quaternion_norm = torch.sqrt(1 + (bcd**2).sum(dim=0))
        a = 1 / quaternion_norm
        b, c, d = bcd / quaternion_norm

        Rs = torch.cat(
            [
                torch.stack(row, dim=1).unsqueeze(1)
                for row in [
                    [
                        a**2 + b**2 - c**2 - d**2,
                        2 * (b * c - a * d),
                        2 * (b * d + a * c),
                    ],
                    [
                        2 * (b * c + a * d),
                        a**2 - b**2 + c**2 - d**2,
                        2 * (c * d - a * b),
                    ],
                    [
                        2 * (b * d - a * c),
                        2 * (c * d + a * b),
                        a**2 - b**2 - c**2 + d**2,
                    ],
                ]
            ],
            dim=1,
        )
        bb_mask = torch.ones((N_RES,))

        return ProteinFrames(Rs, ts, bb_mask)


class AngleResNet(nn.Module):
    def __init__(self, single_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.single_proj1 = nn.Linear(single_dim, proj_dim)
        self.single_proj2 = nn.Linear(single_dim, proj_dim)
        self.angle_proj1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.angle_proj2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.to_pred_angles = nn.Sequential(
            nn.ReLU(), nn.Linear(proj_dim, 2 * len(AA_TORSION_NAMES))
        )

    def forward(
        self, single_rep: torch.Tensor, single_rep_initial: torch.Tensor
    ) -> torch.Tensor:
        a = self.single_proj1(single_rep) + self.single_proj2(single_rep_initial)
        a = a + self.angle_proj1(a)
        a = a + self.angle_proj2(a)
        pred_angles = self.to_pred_angles(a)
        return pred_angles.view(-1, len(AA_TORSION_NAMES), 2)


def frame_aligned_point_error(
    frames: ProteinFrames,
    ca_coords: torch.Tensor,
    true_frames: ProteinFrames,
    true_ca_coords: torch.Tensor,
    length_scale: float = 10,
    d_clamp: float = 10,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    aligned = frames.apply_inverse_along_dim(ca_coords.unsqueeze(0), dim=0)
    true_aligned = true_frames.apply_inverse_along_dim(
        true_ca_coords.unsqueeze(0), dim=0
    )
    d = torch.sqrt((aligned - true_aligned) ** 2 + epsilon)

    fape = 1 / length_scale * torch.clamp(d, max=d_clamp).mean()

    return fape


def torsion_angle_loss(
    pred_angles: torch.Tensor,
    true_angles: torch.Tensor,
    true_alt_angles: torch.Tensor,
    w_angle_norm: float = 0.02,
) -> torch.Tensor:
    pred_angle_norm = torch.norm(pred_angles, p=2, dim=-1)
    pred_angles = pred_angles / pred_angle_norm.unsqueeze(-1)

    L_torsion = torch.min(
        ((pred_angles - true_angles) ** 2).sum(dim=-1),
        ((pred_angles - true_alt_angles) ** 2).sum(dim=-1),
    ).mean()
    L_angle_norm = torch.abs(pred_angle_norm - 1).mean()

    return L_torsion + w_angle_norm * L_angle_norm


def rigid_4x4_rotx_from_angles(angles: torch.Tensor) -> torch.Tensor:
    angles = angles / torch.norm(angles, p=2, dim=-1, keepdim=True)

    out = torch.zeros((*angles.shape[:-1], 4, 4))

    out[..., 0, 0] = 1
    out[..., 0, 0] = 1

    out[..., 1, 1] = angles[..., 0]
    out[..., 1, 2] = -angles[..., 1]

    out[..., 2, 1] = angles[..., 1]
    out[..., 2, 2] = angles[..., 0]

    out[..., 3, 3] = 1

    return out


def compute_all_atom_coords(
    frames: ProteinFrames, angles: torch.Tensor, res_index: torch.Tensor
) -> Tuple[List[ProteinFrames], ProteinStructure]:
    angles = angles / torch.norm(angles, p=2, dim=-1, keepdim=True)

    T_omega, T_phi, T_psi, T_chi1, T_chi2, T_chi3, T_chi4 = rigid_4x4_rotx_from_angles(
        angles.swapdims(0, 1)
    )
    (
        _,
        T_omega_to_bb,
        T_phi_to_bb,
        T_psi_to_bb,
        T_chi1_to_bb,
        T_chi2_to_chi1,
        T_chi3_to_chi2,
        T_chi4_to_chi3,
    ) = AA_LIT_RIGID_TO_RIGID[res_index].swapdims(0, 1)

    T_bb_to_global = frames.as_4x4()

    # Make extra backbone frames
    T_1 = ProteinFrames.compose_4x4([T_bb_to_global, T_omega_to_bb, T_omega])
    T_2 = ProteinFrames.compose_4x4([T_bb_to_global, T_phi_to_bb, T_phi])
    T_3 = ProteinFrames.compose_4x4([T_bb_to_global, T_psi_to_bb, T_psi])

    # Make side chain frames
    T_4 = ProteinFrames.compose_4x4([T_bb_to_global, T_chi1_to_bb, T_chi1])
    T_5 = ProteinFrames.compose_4x4([T_4, T_chi2_to_chi1, T_chi2])
    T_6 = ProteinFrames.compose_4x4([T_5, T_chi3_to_chi2, T_chi3])
    T_7 = ProteinFrames.compose_4x4([T_6, T_chi4_to_chi3, T_chi4])

    T_all = torch.stack([T_bb_to_global, T_1, T_2, T_3, T_4, T_5, T_6, T_7], dim=0)

    # Map atom literature positions to global frame
    N_RES = len(res_index)
    N_ATOM14_TYPES = 14
    atom14_coords = torch.zeros((N_RES, N_ATOM14_TYPES, 3))
    atom14_mask = AA_ATOM14_MASK[res_index] == 1
    for i, res in enumerate(res_index):
        has_atom = atom14_mask[res]
        T_per_atom = T_all[AA_ATOM14_TO_RIGID_GROUP[res, has_atom], res]
        atom_pos = AA_LIT_ATOM14_POS_4x1[res, has_atom]
        atom14_coords[i, has_atom] = torch.einsum("ijk,ik->ij", T_per_atom, atom_pos)[
            :, :-1
        ]

    frames_all_atom = [ProteinFrames.from_4x4(frames_4x4) for frames_4x4 in T_all]
    struct = ProteinStructure.from_atom14(res_index, atom14_coords, atom14_mask)

    return frames_all_atom, struct


class StructureModule(nn.Module):
    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        proj_dim: int,
        ipa_dim: int,
        n_ipa_heads: int,
        n_ipa_query_points: int,
        n_ipa_point_values: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers

        self.layer_norm_single1 = nn.LayerNorm(single_dim)
        self.layer_norm_single2 = nn.Sequential(
            nn.Dropout(0.1), nn.LayerNorm(single_dim)
        )
        self.layer_norm_single3 = nn.Sequential(
            nn.Dropout(0.1), nn.LayerNorm(single_dim)
        )
        self.layer_norm_pair = nn.LayerNorm(pair_dim)

        self.single_proj1 = nn.Linear(single_dim, single_dim)
        self.single_proj2 = nn.Sequential(
            nn.Linear(single_dim, single_dim),
            nn.ReLU(),
            nn.Linear(single_dim, single_dim),
            nn.ReLU(),
            nn.Linear(single_dim, single_dim),
        )

        self.ipa = InvariantPointAttention(
            single_dim,
            pair_dim,
            ipa_dim,
            n_ipa_heads,
            n_ipa_query_points,
            n_ipa_point_values,
        )

        self.to_bb_update = BackboneUpdate(single_dim)

        self.to_pred_angles = AngleResNet(single_dim, proj_dim)

    def forward(
        self,
        single_rep: torch.Tensor,
        pair_rep: torch.Tensor,
        res_index: torch.Tensor,
        target_struct: ProteinStructure = None,
    ) -> Union[
        Tuple[ProteinStructure, float],
        Tuple[ProteinStructure, float, float, float, float],
    ]:
        N_RES = len(single_rep)

        compute_losses = target_struct is not None

        single_rep_initial = self.layer_norm_single1(single_rep)
        pair_rep = self.layer_norm_pair(pair_rep)

        single_rep = self.single_proj1(single_rep_initial)
        frames = ProteinFrames.zero_init(N_RES, requires_grad=self.training)

        if compute_losses:
            true_frames = ProteinFrames.from_structure(target_struct)
            _torsion_angles, _alt_torsion_angles, _ = target_struct.get_torsion_angles()
            true_angles = torch.stack(
                [torch.sin(_torsion_angles), torch.cos(_torsion_angles)],
                dim=2,
            )
            true_alt_angles = torch.stack(
                [torch.sin(_alt_torsion_angles), torch.cos(_alt_torsion_angles)],
                dim=2,
            )
            L_aux = 0

        for l in range(self.n_layers):
            single_rep = single_rep + self.ipa(single_rep, pair_rep, frames)
            single_rep = self.layer_norm_single2(single_rep)

            # Transition
            single_rep = single_rep + self.single_proj2(single_rep)
            single_rep = self.layer_norm_single3(single_rep)

            # Update backbone
            frames = frames.composed_with(self.to_bb_update(single_rep))

            # Predict torsion angles
            pred_angles = self.to_pred_angles(single_rep, single_rep_initial)

            # Auxiliary losses
            if compute_losses:
                ca_coords = frames.ts
                true_ca_coords = true_frames.ts

                L_aux_l = frame_aligned_point_error(
                    frames, ca_coords, true_frames, true_ca_coords
                ) + torsion_angle_loss(pred_angles, true_angles, true_alt_angles)
                L_aux = L_aux + L_aux_l

            ## Stop rotational gradients in intermediate layers
            if l < self.n_layers - 1 and self.training:
                frames.Rs = frames.Rs.detach()

        frames_per_group, pred_struct = compute_all_atom_coords(
            frames, pred_angles, res_index
        )
        plddt = torch.zeros((N_RES,))

        if compute_losses:
            L_aux = L_aux / self.n_layers

            # TODO: compute all-atom FAPE loss
            L_fape = 0
            L_conf = 0

            return pred_struct, plddt, L_fape, L_conf, L_aux

        return pred_struct, plddt
