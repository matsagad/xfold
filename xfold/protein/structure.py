import biotite
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from constants import (
    AminoAcidVocab,
    AMINO_ACID_ATOM_TYPES,
    MSA_GAP_SYMBOL,
    AMINO_ACID_UNKNOWN,
    BACKBONE_ATOM_TYPES,
)
from functools import reduce
from operator import mul
from sequence import Sequence, MSA
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


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


class ProteinFrames:
    def __init__(self, structure: ProteinStructure) -> None:
        self.Rs, self.ts = self._structure_to_frames(structure)
        self.backbone_mask = reduce(
            mul, (structure.atom_masks[atom_type] for atom_type in BACKBONE_ATOM_TYPES)
        )

    def _structure_to_frames(
        self, structure: ProteinStructure
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rigid_from_three_points(
            *(structure.atom_coords[atom_type] for atom_type in BACKBONE_ATOM_TYPES)
        )

    def rigid_from_three_points(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(x1.shape, x2.shape, x3.shape)
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
        self.frames = ProteinFrames(structure)
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
        is_glycine = structure.seq.seq[:, AminoAcidVocab.get_index(GLYCINE)] == 1
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
        # TODO: compute torsion angles
        raise NotImplementedError()
