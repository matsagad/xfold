import biotite
from biotite.structure import dihedral, dihedral_backbone
import biotite.structure
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from functools import reduce
from operator import mul
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from xfold.protein.constants import (
    AminoAcidVocab,
    AMINO_ACID_ATOM_TYPES,
    MSA_GAP_SYMBOL,
    AMINO_ACID_UNKNOWN,
    BACKBONE_ATOM_TYPES,
    AMINO_ACID_ATOMS_FOR_CHI_ANGLES,
    AMINO_ACID_180_DEG_SYMMETRIC_CHI_ANGLE,
)
from xfold.protein.sequence import Sequence, MSA


class ProteinStructure:
    def __init__(
        self,
        seq: str,
        atom_coords: Dict[str, torch.Tensor],
        atom_masks: Dict[str, torch.Tensor],
    ) -> None:
        self.seq = Sequence(seq.replace(MSA_GAP_SYMBOL, AMINO_ACID_UNKNOWN))
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
        n_residues = len(seq)

        # Structure
        N_COORDS_PER_RESIDUE = 3
        RESIDUE_INDEX_OFFSET = 1

        atom_coords = {}
        atom_masks = {}
        for atom in AMINO_ACID_ATOM_TYPES:
            atom_coords[atom] = torch.zeros((n_residues, N_COORDS_PER_RESIDUE))
            atom_masks[atom] = torch.zeros(n_residues)

            residue_contains_atom = torch.full((n_residues,), False)
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


class ProteinFrames:
    def __init__(
        self, Rs: torch.Tensor, ts: torch.Tensor, backbone_mask: torch.Tensor
    ) -> None:
        N_res = Rs.shape[0]
        assert Rs.shape == (N_res, 3, 3)
        assert ts.shape == (N_res, 3)
        assert backbone_mask.shape == (N_res,)

        self.Rs = Rs
        self.ts = ts
        self.backbone_mask = backbone_mask

    def from_structure(structure: ProteinStructure) -> "ProteinFrames":
        Rs, ts = ProteinFrames._structure_to_frames(structure)
        backbone_mask = reduce(
            mul, (structure.atom_masks[atom_type] for atom_type in BACKBONE_ATOM_TYPES)
        )
        return ProteinFrames(Rs, ts, backbone_mask)

    def _structure_to_frames(
        structure: ProteinStructure,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return ProteinFrames.rigid_from_three_points(
            *(structure.atom_coords[atom_type] for atom_type in BACKBONE_ATOM_TYPES)
        )

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


class TemplateProtein:
    def __init__(self, structure: ProteinStructure) -> None:
        self.frames = ProteinFrames.from_structure(structure)
        self.template_pair_feat = self._build_pair_feature_matrix(structure)
        self.template_angle_feat = self._build_angle_feature_matrix(structure)

    def _build_pair_feature_matrix(self, structure: ProteinStructure) -> torch.Tensor:
        N_res, N_temp_pair_feats = structure.seq.length(), 88

        # Distogram: one-hot encoding of binned interatomic CB distances
        ALPHA_CARBON = "CA"
        BETA_CARBON = "CB"
        GLYCINE = "G"
        N_COORDS_PER_RESIDUE = 3

        N_BINS = 38
        MIN_INTERATOMIC_DISTANCE = 3.25
        MAX_INTERATOMIC_DISTANCE = 50.75
        BIN_EDGES = torch.linspace(
            MIN_INTERATOMIC_DISTANCE, MAX_INTERATOMIC_DISTANCE, N_BINS + 1
        )

        cb_atom_coords = torch.empty((N_res, N_COORDS_PER_RESIDUE))
        is_glycine = (
            structure.seq.seq_one_hot[:, AminoAcidVocab.get_index(GLYCINE)] == 1
        )
        cb_atom_coords[is_glycine] = structure.atom_coords[BETA_CARBON][is_glycine]
        cb_atom_coords[~is_glycine] = structure.atom_coords[ALPHA_CARBON][~is_glycine]

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
        aatype_one_hot1 = structure.msa_seq[:, :-1].unsqueeze(1).expand(-1, N_res, -1)
        aatype_one_hot2 = structure.msa_seq[:, :-1].unsqueeze(0).expand(N_res, -1, -1)

        # Pseudo beta mask: CB (CA for glycine) atom coordinates exist
        pseudo_beta_mask = torch.empty(N_res)
        pseudo_beta_mask[is_glycine] = structure.atom_masks[ALPHA_CARBON][is_glycine]
        pseudo_beta_mask[~is_glycine] = structure.atom_masks[BETA_CARBON][~is_glycine]
        pseudo_beta_pair_mask = (
            pseudo_beta_mask.unsqueeze(1) * pseudo_beta_mask.unsqueeze(0)
        ).unsqueeze(-1)

        # Backbone frame mask: N, CA, C atom coordinates all exist
        backbone_pair_mask = (
            self.frames.backbone_mask.unsqueeze(1)
            * self.frames.backbone_mask.unsqueeze(0)
        ).unsqueeze(-1)

        # fmt: off
        temp_pair_feat = torch.cat(
            [
                distogram,              # (N_res, N_res, 39)
                unit_diff_vector,       # (N_res, N_res,  3)
                aatype_one_hot1,        # (N_res, N_res, 22)
                aatype_one_hot2,        # (N_res, N_res, 22)
                pseudo_beta_pair_mask,  # (N_res, N_res,  1)
                backbone_pair_mask,     # (N_res, N_res,  1)
            ],
            dim=-1,
        )
        # fmt: on
        assert temp_pair_feat.shape == (N_res, N_res, N_temp_pair_feats)

        return temp_pair_feat

    def _build_angle_feature_matrix(self, structure: ProteinStructure) -> torch.Tensor:
        N_res, N_temp_angle_feats = structure.seq.length(), 57

        # Amino acid one-hot encoding
        aatype_one_hot = structure.msa_seq[:, :-1]

        # Torsion angles: 3 x backbone and 4 x side chain angles in sine and cosine
        atom_arr = structure._to_biotite_atom_array()
        phi, psi, omega = tuple(map(torch.from_numpy, dihedral_backbone(atom_arr)))
        backbone_angles = [omega, phi, psi]
        backbone_angle_masks = torch.stack(
            [angle.isfinite().long() for angle in backbone_angles], dim=1
        )
        for i, angle in enumerate(backbone_angles):
            angle[backbone_angle_masks[:, i] == 0] = 0

        N_CHI_ANGLES = 4
        chi_angles = torch.zeros((N_res, N_CHI_ANGLES))
        alt_chi_angles = torch.zeros((N_res, N_CHI_ANGLES))
        chi_angle_masks = torch.zeros((N_res, N_CHI_ANGLES))
        for i in range(N_res):
            res_id = i + 1
            res_atoms = atom_arr[atom_arr.res_id == res_id]
            res_code = res_atoms[0].res_name
            if res_atoms.array_length() == 0:
                continue
            atom_types_per_chi_angle = AMINO_ACID_ATOMS_FOR_CHI_ANGLES[res_code]
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
                        res_code in AMINO_ACID_180_DEG_SYMMETRIC_CHI_ANGLE
                        and chi_angle_no + 1
                        == AMINO_ACID_180_DEG_SYMMETRIC_CHI_ANGLE[res_code]
                    ):
                        alt_chi_angles[i][chi_angle_no] = (
                            chi_angles[i][chi_angle_no] + torch.pi
                        )
                    else:
                        alt_chi_angles[i][chi_angle_no] = chi_angles[i][chi_angle_no]
                    chi_angle_masks[i][chi_angle_no] = 1

        backbone_torsion_angles = torch.stack(backbone_angles, dim=1)
        torsion_angles_sin_cos = torch.cat(
            [
                trig_angles
                for angles in (backbone_torsion_angles, chi_angles)
                for trig_angles in (torch.sin(angles), torch.cos(angles))
            ],
            dim=1,
        )
        alt_torsion_angles_sin_cos = torch.cat(
            [
                trig_angles
                for angles in (backbone_torsion_angles, alt_chi_angles)
                for trig_angles in (torch.sin(angles), torch.cos(angles))
            ],
            dim=1,
        )
        torsion_angle_masks = torch.cat([backbone_angle_masks, chi_angle_masks], dim=1)

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

        assert temp_angle_feat.shape == (N_res, N_temp_angle_feats)

        return temp_angle_feat
