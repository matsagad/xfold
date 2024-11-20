import biotite
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from constants import AminoAcidVocab, AMINO_ACID_ATOM_TYPES
from sequence import Sequence
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
        self.seq = Sequence(seq)
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

    def _structure_to_frames(
        self, structure: ProteinStructure
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rigid_from_three_points(
            *(structure.atom_coords[atom_type] for atom_type in ["N", "CA", "C"])
        )

    def rigid_from_three_points(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(x1.shape, x2.shape, x3.shape)
        # Supplementary Info Algorithm 21
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
        self.template_pair_feat = self._build_pair_feature_matrix(structure)
        self.template_angle_feat = self._build_angle_feature_matrix(structure)

    def _build_pair_feature_matrix(self, structure: ProteinStructure) -> torch.Tensor:
        N_res, N_temp_pair_feats = structure.seq.length(), 88
        temp_pair_feat = torch.empty((N_res, N_res, N_temp_pair_feats))

        # Distogram: one-hot encoding of binned interatomic CB distances
        ALPHA_CARBON = "CA"
        BETA_CARBON = "CB"
        N_COORDS_PER_RESIDUE = 3

        N_BINS = 38
        MIN_INTERATOMIC_DISTANCE = 3.25
        MAX_INTERATOMIC_DISTANCE = 50.75
        BIN_EDGES = torch.linspace(
            MIN_INTERATOMIC_DISTANCE, MAX_INTERATOMIC_DISTANCE, N_BINS + 1
        )

        carbon_atom_coords = torch.empty((N_res, N_COORDS_PER_RESIDUE))
        has_beta_carbon = structure.atom_masks[BETA_CARBON] == 1
        carbon_atom_coords[has_beta_carbon] = structure.atom_coords[BETA_CARBON][
            has_beta_carbon
        ]
        carbon_atom_coords[~has_beta_carbon] = structure.atom_coords[ALPHA_CARBON][
            ~has_beta_carbon
        ]

        dist = torch.cdist(carbon_atom_coords, carbon_atom_coords, p=2)
        bin_nos = torch.searchsorted(BIN_EDGES, dist) - 1
        bin_nos[bin_nos == -1] = 0
        temp_pair_feat[:, :, : N_BINS + 1] = F.one_hot(bin_nos, N_BINS + 1)

        # TODO: add unit vector and other features

    def _build_angle_feature_matrix(self, structure: ProteinStructure) -> torch.Tensor:
        # TODO: compute torsion angles
        raise NotImplementedError()
