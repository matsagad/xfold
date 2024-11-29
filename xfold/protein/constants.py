import torch
from typing import Iterable, List

# Amino Acid Constants
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWXY"
AA_INDICES = {aa: i for i, aa in enumerate(AA_ALPHABET)}
AA_UNKNOWN = "X"

AA_CODE_EXPAND = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "X": "XAA",
    "Y": "TYR",
}
AA_CODE_CONTRACT = {aa3: aa1 for aa1, aa3 in AA_CODE_EXPAND.items()}
AA3_ALPHABET = [AA_CODE_EXPAND[aa] for aa in AA_ALPHABET]
AA3_INDICES = {aa3: i for i, aa3 in enumerate(AA3_ALPHABET)}


class AminoAcidVocab:
    vocab = AA_ALPHABET
    vocab_size = len(AA_ALPHABET)

    def get_index(aa: str) -> int:
        return AA_INDICES.get(aa, -1)

    def get_three_letter_code(aa: str) -> str:
        return AA_CODE_EXPAND.get(aa, "XXX")

    def contract_three_letter_code(aa3: str) -> str:
        return AA_CODE_CONTRACT.get(aa3, "X")

    def is_amino_acid(aa: str) -> bool:
        return aa in AA_CODE_EXPAND

    def index_sequence(seq: Iterable[str]) -> List[str]:
        return [AminoAcidVocab.get_index(aa) for aa in seq]

    def sequence_from_indices(indices: Iterable[int]) -> str:
        return "".join(AA_ALPHABET[index] for index in indices)


# fmt: off
# Amino acid atom constants
AA_ATOM37_TYPES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE",
    "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
AA_ATOM37_TYPES_INDICES = {atom_type: i for i, atom_type in enumerate(AA_ATOM37_TYPES)}
HEAVY_ATOMS_BY_AA = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O"],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}
## Don't include chi5 for ARG as largely insignificant
CHI_ANGLE_ATOMS_BY_AA = {
    "ALA": [],
    "ARG": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"]],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"]],
    "MET": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}
AA_180_DEG_SYMMETRIC_CHI_ANGLES = {
    "ASP": [2],
    "GLU": [3],
    "PHE": [2],
    "TYR": [2],
}
AA_AMBIG_ATOMS_RENAMING_SWAPS = {
    "ASP": {"OD1": "OD2", "OD2": "OD1"},
    "GLU": {"OE1": "OE2", "OE2": "OE1"},
    "PHE": {"CD1": "CD2", "CD2": "CD1", "CE1": "CE2", "CE2": "CE1"},
    "TYR": {"CD1": "CD2", "CD2": "CD1", "CE1": "CE2", "CE2": "CE1"},
}
## Taken from alphafold/common/residue_constants.py
## Legend: [atom, rigid group no, relative position]
##  where the rigid groups are:
##   0 - backbone group
##   1 - pre-omega group
##   2 - phi group
##   3 - psi group
##   4,5,6,7 - chi1,2,3,4 groups
AA_RIGID_GROUP_ATOM14_POS = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}
# fmt: on
AA_ATOM_TYPE_TO_RIGID_GROUP = {
    atom_type: rigid_group_no
    for atom_infos in AA_RIGID_GROUP_ATOM14_POS.values()
    for atom_type, rigid_group_no, _ in atom_infos
}
BB_ATOM_TYPES = ["N", "CA", "C"]
AA_TORSION_NAMES = ["omega", "phi", "psi", "chi1", "chi2", "chi3", "chi4"]

## Each amino acid can have at most 14 heavy atoms
_N_AA, _MAX_N_ATOM_PER_AA, _N_RIGID_GROUPS = len(AA_ALPHABET), 14, 8
AA_ATOM14_TO_RIGID_GROUP = torch.zeros(_N_AA, _MAX_N_ATOM_PER_AA, dtype=int)
AA_ATOM14_MASK = torch.zeros((_N_AA, _MAX_N_ATOM_PER_AA))
AA_LIT_ATOM14_POS = torch.zeros((_N_AA, _MAX_N_ATOM_PER_AA, 3))
AA_LIT_ATOM14_POS_4x1 = torch.zeros((_N_AA, _MAX_N_ATOM_PER_AA, 4))
AA_LIT_RIGID_TO_RIGID = torch.zeros((_N_AA, _N_RIGID_GROUPS, 4, 4))

AA_AMBIG_ATOMS_MASK = torch.zeros((_N_AA, _MAX_N_ATOM_PER_AA))
AA_AMBIG_ATOMS_PERMUTE = torch.arange(_MAX_N_ATOM_PER_AA).tile(_N_AA, 1)


def rigid_4x4_from_axes(
    e1: torch.Tensor, e2: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    e1 = e1 / torch.norm(e1, dim=-1, keepdim=True)
    e2 = e2 - e1 * (e1 * e2).sum(dim=-1).unsqueeze(-1)
    e2 = e2 / torch.norm(e2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2, dim=-1)

    out = torch.zeros((4, 4))
    out[:3, :3] = torch.stack([e1, e2, e3], dim=0)
    out[:3, 3] = t
    return out


def fill_rigid_group_constants() -> None:
    for aa3, atom_infos in AA_RIGID_GROUP_ATOM14_POS.items():
        res_index = AA3_INDICES[aa3]

        for atom_type, rigid_group_no, rel_pos in atom_infos:
            atom_index = HEAVY_ATOMS_BY_AA[aa3].index(atom_type)

            AA_ATOM14_TO_RIGID_GROUP[res_index, atom_index] = rigid_group_no
            AA_ATOM14_MASK[res_index, atom_index] = 1
            AA_LIT_ATOM14_POS[res_index, atom_index] = torch.tensor(rel_pos)
            AA_LIT_ATOM14_POS_4x1[res_index, atom_index] = torch.tensor((*rel_pos, 1))

    # bb to bb
    AA_LIT_RIGID_TO_RIGID[:, 0] = torch.eye(4)
    # pre-omega to bb
    AA_LIT_RIGID_TO_RIGID[:, 1] = torch.eye(4)

    for aa3, atom_infos in AA_RIGID_GROUP_ATOM14_POS.items():
        res_index = AA3_INDICES[aa3]
        atom_pos = {
            atom_type: torch.tensor(rel_pos) for atom_type, _, rel_pos in atom_infos
        }
        chi_angle_atoms = CHI_ANGLE_ATOMS_BY_AA[aa3]

        # phi to bb
        AA_LIT_RIGID_TO_RIGID[res_index, 2] = rigid_4x4_from_axes(
            e1=atom_pos["N"] - atom_pos["CA"],
            e2=torch.tensor([1, 0, 0]),
            t=atom_pos["N"],
        )
        # psi to bb
        AA_LIT_RIGID_TO_RIGID[res_index, 3] = rigid_4x4_from_axes(
            e1=atom_pos["C"] - atom_pos["CA"],
            e2=atom_pos["CA"] - atom_pos["N"],
            t=atom_pos["C"],
        )
        # chi1 to bb
        if chi_angle_atoms:
            atom1, atom2, atom3, _ = chi_angle_atoms[0]
            AA_LIT_RIGID_TO_RIGID[res_index, 4] = rigid_4x4_from_axes(
                e1=atom_pos[atom3] - atom_pos[atom2],
                e2=atom_pos[atom1] - atom_pos[atom2],
                t=atom_pos[atom3],
            )
        # chi2 to chi1, chi3 to chi2, chi4 to chi3
        for i in range(1, len(chi_angle_atoms)):
            axis_end_atom = chi_angle_atoms[i][2]
            AA_LIT_RIGID_TO_RIGID[res_index, i + 4] = rigid_4x4_from_axes(
                e1=atom_pos[axis_end_atom],
                e2=torch.tensor([-1, 0, 0]),
                t=atom_pos[axis_end_atom],
            )

def fill_ambig_constants() -> None:
    for aa3, swaps in AA_AMBIG_ATOMS_RENAMING_SWAPS.items():
        res_index = AA3_INDICES[aa3]
        heavy_atoms = HEAVY_ATOMS_BY_AA[aa3]
        for atom1, atom2 in swaps.items():
            AA_AMBIG_ATOMS_MASK[res_index, heavy_atoms.index(atom1)] = 1
            AA_AMBIG_ATOMS_PERMUTE[res_index, heavy_atoms.index(atom1)] = (
                heavy_atoms.index(atom2)
            )


fill_rigid_group_constants()
fill_ambig_constants()


# MSA Constants
MSA_GAP_SYMBOL = "-"
MSA_MASKED_SYMBOL = "#"
MSA_SYMBOL_INDICES = {
    **AA_INDICES,
    MSA_GAP_SYMBOL: AminoAcidVocab.vocab_size,
    MSA_MASKED_SYMBOL: AminoAcidVocab.vocab_size + 1,
}
AA_TEMP_INDICES = {**AA_INDICES, MSA_GAP_SYMBOL: AminoAcidVocab.vocab_size}


class MSAVocab:
    vocab = list(MSA_SYMBOL_INDICES.keys())
    vocab_size = len(MSA_SYMBOL_INDICES)

    def get_index(msa_char: str) -> int:
        return MSA_SYMBOL_INDICES.get(msa_char, -1)

    def is_msa_char(msa_char: str) -> bool:
        return msa_char in MSA_SYMBOL_INDICES

    def index_sequence(seq: Iterable[str]) -> List[str]:
        return [MSAVocab.get_index(msa_char) for msa_char in seq]
