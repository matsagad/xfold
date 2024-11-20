import biotite
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from constants import AminoAcidVocab, AMINO_ACID_ATOM_TYPES
from sequence import Sequence
import torch
from typing import Dict


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
