import biotite
from biotite.structure import dihedral, dihedral_backbone
import biotite.structure
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from functools import reduce
from operator import mul
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from xfold.protein.constants import (
    AminoAcidVocab,
    AA_180_DEG_SYMMETRIC_CHI_ANGLES,
    AA_ATOM37_TYPES,
    AA_UNKNOWN,
    BB_ATOM_TYPES,
    CHI_ANGLE_ATOMS_BY_AA,
    HEAVY_ATOMS_BY_AA,
    MSA_GAP_SYMBOL,
)
from xfold.protein.sequence import Sequence, MSA


class ProteinStructure:
    def __init__(
        self,
        seq: str,
        atom_coords: Dict[str, torch.Tensor],
        atom_masks: Dict[str, torch.Tensor],
    ) -> None:
        self.seq = Sequence(seq.replace(MSA_GAP_SYMBOL, AA_UNKNOWN))
        self.msa_seq = MSA.one_hot_encode([seq])[0]
        self.atom_coords = atom_coords
        self.atom_masks = atom_masks

    def _from_atom_arr(atom_arr: biotite.structure.AtomArray) -> "ProteinStructure":
        # Sequence
        seq = "".join(
            map(
                AminoAcidVocab.contract_three_letter_code,
                atom_arr[atom_arr.atom_name == "CA"].res_name,
            )
        )
        N_RES = len(seq)

        # Structure
        N_COORDS_PER_RESIDUE = 3
        RESIDUE_INDEX_OFFSET = 1

        atom_coords = {}
        atom_masks = {}
        for atom in AA_ATOM37_TYPES:
            atom_coords[atom] = torch.zeros((N_RES, N_COORDS_PER_RESIDUE))
            atom_masks[atom] = torch.zeros(N_RES)

            residue_contains_atom = torch.full((N_RES,), False)
            residue_contains_atom[
                atom_arr[atom_arr.atom_name == atom].res_id - RESIDUE_INDEX_OFFSET
            ] = True
            atom_coords[atom][residue_contains_atom] = torch.from_numpy(
                atom_arr.coord[atom_arr.atom_name == atom]
            )
            atom_masks[atom][residue_contains_atom] = 1

        return ProteinStructure(seq, atom_coords, atom_masks)

    def from_pdb(f_pdb: str) -> "ProteinStructure":
        atom_arr = pdb.PDBFile.read(f_pdb).get_structure(model=1)

        n_chains = len(set(atom_arr.chain_id))
        if n_chains != 1:
            raise Exception(f"Protein in PDB file '{f_pdb}' is not monomeric.")
        return ProteinStructure._from_atom_arr(atom_arr)

    def from_mmcif(f_mmcif: str) -> "ProteinStructure":
        cif = pdbx.CIFFile.read(f_mmcif)
        names = list(cif.keys())
        if len(names) != 1:
            raise Exception(f"More than one protein in mmCIF file '{f_mmcif}'.")
        cif_block = cif[names[0]]

        atom_arr = pdbx.get_structure(cif_block, model=1)
        n_chains = len(set(atom_arr.chain_id))
        if n_chains != 1:
            raise Exception(f"Protein in mmCIF file '{f_mmcif}' is not monomeric.")
        return ProteinStructure._from_atom_arr(atom_arr)

    def _to_biotite_atom_array(self) -> biotite.structure.AtomArray:
        atoms = []
        for atom_type, coords in self.atom_coords.items():
            mask = self.atom_masks[atom_type]
            valid_res_ids = torch.argwhere(mask)[:, 0] + 1

            for res_id in valid_res_ids:
                atoms.append(
                    biotite.structure.Atom(
                        coord=coords[res_id - 1],
                        chain_id="A",
                        res_id=res_id,
                        res_name=AminoAcidVocab.get_three_letter_code(
                            self.seq.seq_str[res_id - 1]
                        ),
                        atom_name=atom_type,
                        element=atom_type[0],
                    )
                )
        atoms.sort(key=lambda x: x.res_id)
        return biotite.structure.array(atoms)

    def get_beta_carbon_coords(self) -> torch.Tensor:
        N_RES = self.seq.length()
        ALPHA_CARBON = "CA"
        BETA_CARBON = "CB"
        GLYCINE = "G"

        cb_atom_coords = torch.empty((N_RES, 3))
        is_glycine = self.seq.seq_one_hot[:, AminoAcidVocab.get_index(GLYCINE)] == 1
        cb_atom_coords[is_glycine] = self.atom_coords[BETA_CARBON][is_glycine]
        cb_atom_coords[~is_glycine] = self.atom_coords[ALPHA_CARBON][~is_glycine]
        return cb_atom_coords

    def get_torsion_angles(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N_RES = self.seq.length()
        N_CHI_ANGLES = 4

        atom_arr = self._to_biotite_atom_array()
        phi, psi, omega = tuple(map(torch.from_numpy, dihedral_backbone(atom_arr)))
        bb_angles = [omega, phi, psi]
        bb_angle_masks = torch.stack(
            [angle.isfinite().long() for angle in bb_angles], dim=1
        )
        for i, angle in enumerate(bb_angles):
            angle[bb_angle_masks[:, i] == 0] = 0

        chi_angles = torch.zeros((N_RES, N_CHI_ANGLES))
        alt_chi_angles = torch.zeros((N_RES, N_CHI_ANGLES))
        chi_angle_masks = torch.zeros((N_RES, N_CHI_ANGLES))
        for i in range(N_RES):
            res_id = i + 1
            res_atoms = atom_arr[atom_arr.res_id == res_id]
            res_code = res_atoms[0].res_name
            if res_atoms.array_length() == 0:
                continue
            atom_types_per_chi_angle = CHI_ANGLE_ATOMS_BY_AA[res_code]
            for chi_angle_no, atom_types in enumerate(atom_types_per_chi_angle):
                chi_angle_atom_matches = [
                    res_atoms[res_atoms.atom_name == atom_type]
                    for atom_type in atom_types
                ]
                if all(
                    len(atom_matches) == 1 for atom_matches in chi_angle_atom_matches
                ):
                    chi_angles[i][chi_angle_no] = dihedral(
                        *(atom_matches[0] for atom_matches in chi_angle_atom_matches)
                    ).item()
                    if (
                        res_code in AA_180_DEG_SYMMETRIC_CHI_ANGLES
                        and chi_angle_no + 1
                        == AA_180_DEG_SYMMETRIC_CHI_ANGLES[res_code]
                    ):
                        alt_chi_angles[i][chi_angle_no] = (
                            chi_angles[i][chi_angle_no] + torch.pi
                        )
                    else:
                        alt_chi_angles[i][chi_angle_no] = chi_angles[i][chi_angle_no]
                    chi_angle_masks[i][chi_angle_no] = 1

        bb_torsion_angles = torch.stack(bb_angles, dim=1)

        torsion_angles = torch.cat([bb_torsion_angles, chi_angles], dim=1)
        alt_torsion_angles = torch.cat([bb_torsion_angles, alt_chi_angles], dim=1)
        torsion_angle_masks = torch.cat([bb_angle_masks, chi_angle_masks], dim=1)

        return torsion_angles, alt_torsion_angles, torsion_angle_masks

    def from_atom14(
        res_index: torch.Tensor, atom14_coords: torch.Tensor, atom14_mask: torch.Tensor
    ) -> "ProteinStructure":
        seq = AminoAcidVocab.sequence_from_indices(res_index.tolist())
        atom_coords = {}
        atom_masks = {}
        for i_seq, aa in enumerate(seq):
            aa3 = AminoAcidVocab.get_three_letter_code(aa)
            for i_atom, atom_type in enumerate(HEAVY_ATOMS_BY_AA[aa3]):
                if atom_type not in atom_coords:
                    atom_coords[atom_type] = torch.zeros(len(seq), 3)
                if atom_type not in atom_masks:
                    atom_masks[atom_type] = torch.zeros(len(seq))
                atom_coords[atom_type][i_seq] = atom14_coords[i_seq, i_atom]
                atom_masks[atom_type][i_seq] = atom14_mask[i_seq, i_atom].float()
        return ProteinStructure(seq, atom_coords, atom_masks)


class ProteinFrames:
    def __init__(
        self, Rs: torch.Tensor, ts: torch.Tensor, bb_mask: torch.Tensor
    ) -> None:
        N_RES = Rs.shape[0]
        assert Rs.shape == (N_RES, 3, 3)
        assert ts.shape == (N_RES, 3)
        assert bb_mask.shape == (N_RES,)

        self.Rs = Rs
        self.ts = ts
        self.bb_mask = bb_mask

    def from_structure(structure: ProteinStructure) -> "ProteinFrames":
        Rs, ts = ProteinFrames._structure_to_frames(structure)
        bb_mask = reduce(
            mul, (structure.atom_masks[atom_type] for atom_type in BB_ATOM_TYPES)
        )
        return ProteinFrames(Rs, ts, bb_mask)

    def _structure_to_frames(
        structure: ProteinStructure,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return ProteinFrames.rigid_from_three_points(
            *(structure.atom_coords[atom_type] for atom_type in BB_ATOM_TYPES)
        )

    def from_4x4(frames_4x4: torch.Tensor) -> "ProteinFrames":
        Rs = frames_4x4[:, :3, :3]
        ts = frames_4x4[:, :3, 3]
        bb_mask = torch.ones((len(Rs),))
        return ProteinFrames(Rs, ts, bb_mask)

    def rigid_from_three_points(
        x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # AlphaFold Supplementary Info Algorithm 21
        v1 = x3 - x2
        v2 = x1 - x2

        e1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
        u2 = v2 - e1 * (e1 * v2).sum(dim=-1).unsqueeze(-1)
        e2 = u2 / torch.norm(u2, dim=-1, keepdim=True)
        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.cat([e1.unsqueeze(-2), e2.unsqueeze(-2), e3.unsqueeze(-2)], dim=-2)
        t = x2
        return R, t

    def zero_init(n_res: int, requires_grad: bool = False) -> "ProteinFrames":
        # "Black-hole initialisation"
        Rs = torch.eye(3).unsqueeze(0).repeat(n_res, 1, 1)
        ts = torch.zeros((n_res, 3))
        bb_mask = torch.ones((n_res,))
        if requires_grad:
            Rs.requires_grad = True
            ts.requires_grad = True
        return ProteinFrames(Rs, ts, bb_mask)

    def composed_with(self, other: "ProteinFrames") -> "ProteinFrames":
        return ProteinFrames(
            self.Rs @ other.Rs,
            (self.Rs @ other.ts.unsqueeze(-1)).squeeze(-1) + self.ts,
            self.bb_mask * other.bb_mask,
        )

    def compose(frames_list: List["ProteinFrames"]) -> "ProteinFrames":
        return reduce(lambda x2, x1: x1.composed_with(x2), reversed(frames_list))

    def compose_4x4(frames_4x4_list: List[torch.Tensor]) -> torch.Tensor:
        return reduce(lambda x2, x1: x1 @ x2, reversed(frames_4x4_list))

    def apply_along_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        assert x.shape[-1] == 3
        altered_shape = x.movedim(dim, 0).shape
        out = (
            (
                (
                    x.movedim(dim, 0).reshape(len(self.Rs), -1, 3)
                    @ self.Rs.transpose(-1, -2)
                )
                + self.ts.unsqueeze(1)
            )
            .view(altered_shape)
            .movedim(0, dim)
        )
        return out

    def apply_inverse_along_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        assert x.shape[-1] == 3
        altered_shape = x.movedim(dim, 0).shape
        out = (
            (
                (x.movedim(dim, 0).reshape(len(self.Rs), -1, 3) - self.ts.unsqueeze(1))
                @ self.Rs
            )
            .view(altered_shape)
            .movedim(0, dim)
        )
        return out

    def as_4x4(self) -> torch.Tensor:
        out = torch.zeros((len(self.Rs), 4, 4))
        out[:, :3, :3] = self.Rs
        out[:, :3, 3] = self.ts
        out[:, 3, 3] = 1
        return out


class TemplateProtein:
    N_TEMP_PAIR_FEATS = 88
    N_TEMP_ANGLE_FEATS = 57

    def __init__(self, structure: ProteinStructure) -> None:
        self.frames = ProteinFrames.from_structure(structure)
        self.template_pair_feat = self._build_pair_feature_matrix(structure)
        self.template_angle_feat = self._build_angle_feature_matrix(structure)

    def _build_pair_feature_matrix(self, structure: ProteinStructure) -> torch.Tensor:
        N_RES, N_TEMP_PAIR_FEATS = structure.seq.length(), self.N_TEMP_PAIR_FEATS
        ALPHA_CARBON = "CA"
        BETA_CARBON = "CB"
        GLYCINE = "G"

        # Distogram: one-hot encoding of binned interatomic CB distances
        N_BINS = 38
        MIN_INTERATOMIC_DISTANCE = 3.25
        MAX_INTERATOMIC_DISTANCE = 50.75
        BIN_EDGES = torch.linspace(
            MIN_INTERATOMIC_DISTANCE, MAX_INTERATOMIC_DISTANCE, N_BINS + 1
        )

        cb_atom_coords = structure.get_beta_carbon_coords()
        cb_dist_matrix = torch.cdist(cb_atom_coords, cb_atom_coords, p=2)
        bin_nos = torch.searchsorted(BIN_EDGES, cb_dist_matrix) - 1
        bin_nos[bin_nos == -1] = 0
        distogram = F.one_hot(bin_nos, N_BINS + 1)

        # Unit vectors: displacement of CA atoms transformed to be within local frame
        local_Rs, local_ts = self.frames.Rs, self.frames.ts
        diff_vector = (
            (
                structure.atom_coords[ALPHA_CARBON].unsqueeze(0)
                - structure.atom_coords[ALPHA_CARBON].unsqueeze(1)
            )
            - local_ts.unsqueeze(1)
        ) @ local_Rs
        unit_diff_vector = diff_vector / torch.norm(diff_vector, dim=-1, keepdim=True)

        # Amino acid one-hot vectors: including gap symbol and tiled both ways
        aatype_one_hot1 = structure.msa_seq[:, :-1].unsqueeze(1).expand(-1, N_RES, -1)
        aatype_one_hot2 = structure.msa_seq[:, :-1].unsqueeze(0).expand(N_RES, -1, -1)

        # Pseudo beta mask: CB (CA for glycine) atom coordinates exist
        pseudo_beta_mask = torch.empty(N_RES)
        is_glycine = (
            structure.seq.seq_one_hot[:, AminoAcidVocab.get_index(GLYCINE)] == 1
        )
        pseudo_beta_mask[is_glycine] = structure.atom_masks[ALPHA_CARBON][is_glycine]
        pseudo_beta_mask[~is_glycine] = structure.atom_masks[BETA_CARBON][~is_glycine]
        pseudo_beta_pair_mask = (
            pseudo_beta_mask.unsqueeze(1) * pseudo_beta_mask.unsqueeze(0)
        ).unsqueeze(-1)

        # Backbone frame mask: N, CA, C atom coordinates all exist
        bb_pair_mask = (
            self.frames.bb_mask.unsqueeze(1) * self.frames.bb_mask.unsqueeze(0)
        ).unsqueeze(-1)

        # fmt: off
        temp_pair_feat = torch.cat(
            [
                distogram,              # (N_res, N_res, 39)
                unit_diff_vector,       # (N_res, N_res,  3)
                aatype_one_hot1,        # (N_res, N_res, 22)
                aatype_one_hot2,        # (N_res, N_res, 22)
                pseudo_beta_pair_mask,  # (N_res, N_res,  1)
                bb_pair_mask,     # (N_res, N_res,  1)
            ],
            dim=-1,
        )
        # fmt: on
        assert temp_pair_feat.shape == (N_RES, N_RES, N_TEMP_PAIR_FEATS)

        return temp_pair_feat

    def _build_angle_feature_matrix(self, structure: ProteinStructure) -> torch.Tensor:
        N_RES, N_TEMP_ANGLE_FEATS = structure.seq.length(), self.N_TEMP_ANGLE_FEATS

        # Amino acid one-hot encoding
        aatype_one_hot = structure.msa_seq[:, :-1]

        # Torsion angles: 3 x backbone and 4 x side chain angles in sine and cosine
        torsion_angles, alt_torsion_angles, torsion_angle_masks = (
            structure.get_torsion_angles()
        )
        torsion_angles_sin_cos = torch.stack(
            [torch.sin(torsion_angles), torch.cos(torsion_angles)],
            dim=2,
        ).flatten(-2, -1)
        alt_torsion_angles_sin_cos = torch.stack(
            [torch.sin(alt_torsion_angles), torch.cos(alt_torsion_angles)],
            dim=2,
        ).flatten(-2, -1)

        # fmt: off
        temp_angle_feat = torch.cat(
            [
                aatype_one_hot,             # (N_res, 22)
                torsion_angles_sin_cos,     # (N_res, 14)
                alt_torsion_angles_sin_cos, # (N_res, 14)
                torsion_angle_masks,        # (N_res,  7)
            ],
            dim=1,
        )
        # fmt: on

        assert temp_angle_feat.shape == (N_RES, N_TEMP_ANGLE_FEATS)

        return temp_angle_feat
