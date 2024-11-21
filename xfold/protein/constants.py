from typing import Iterable, List

# Amino Acid Constants
AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWXY"
AMINO_ACID_INDICES = {aa: i for i, aa in enumerate(AMINO_ACID_ALPHABET)}
AMINO_ACID_UNKNOWN = "X"

AMINO_ACID_CODE_EXPAND = {
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
AMINO_ACID_CODE_CONTRACT = {aa3: aa1 for aa1, aa3 in AMINO_ACID_CODE_EXPAND.items()}


class AminoAcidVocab:
    vocab_size = len(AMINO_ACID_ALPHABET)

    def get_index(aa: str) -> int:
        return AMINO_ACID_INDICES.get(aa, -1)

    def get_three_letter_code(aa: str) -> str:
        return AMINO_ACID_CODE_EXPAND.get(aa, "XXX")

    def contract_three_letter_code(aa3: str) -> str:
        return AMINO_ACID_CODE_CONTRACT.get(aa3, "X")

    def is_amino_acid(aa: str) -> bool:
        return aa in AMINO_ACID_CODE_EXPAND

    def index_sequence(seq: Iterable[str]) -> List[str]:
        return [AminoAcidVocab.get_index(aa) for aa in seq]


# fmt: off
AMINO_ACID_ATOM_TYPES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE",
    "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
AMINO_ACID_ATOM_TYPES_INDICES = {atom_type: i for i, atom_type in enumerate(AMINO_ACID_ATOM_TYPES)}
ATOMS_PER_AMINO_ACID = {
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
AMINO_ACID_ATOMS_FOR_CHI_ANGLES = {
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
# fmt: on
BACKBONE_ATOM_TYPES = ["N", "CA", "C"]


# MSA Constants
MSA_GAP_SYMBOL = "-"
MSA_MASKED_SYMBOL = "#"
MSA_SYMBOL_INDICES = {
    **AMINO_ACID_INDICES,
    MSA_GAP_SYMBOL: AminoAcidVocab.vocab_size,
    MSA_MASKED_SYMBOL: AminoAcidVocab.vocab_size + 1,
}
AMINO_ACID_TEMP_INDICES = {
    **AMINO_ACID_INDICES,
    MSA_GAP_SYMBOL: AminoAcidVocab.vocab_size,
}


class MSAVocab:
    vocab_size = len(MSA_SYMBOL_INDICES)

    def get_index(msa_char: str) -> int:
        return MSA_SYMBOL_INDICES.get(msa_char, -1)

    def is_msa_char(msa_char: str) -> bool:
        return msa_char in MSA_SYMBOL_INDICES

    def index_sequence(seq: Iterable[str]) -> List[str]:
        return [MSAVocab.get_index(msa_char) for msa_char in seq]
