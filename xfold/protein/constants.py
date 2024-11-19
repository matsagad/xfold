AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWXY"
AMINO_ACID_INDICES = {aa: index for index, aa in enumerate(AMINO_ACID_ALPHABET)}

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


MSA_GAP_SYMBOL = "-"
MSA_MASKED_SYMBOL = "#"
MSA_SYMBOL_INDICES = {
    **AMINO_ACID_INDICES,
    MSA_GAP_SYMBOL: AminoAcidVocab.vocab_size,
    MSA_MASKED_SYMBOL: AminoAcidVocab.vocab_size + 1,
}


class MSAVocab:
    vocab_size = len(MSA_SYMBOL_INDICES)

    def get_index(msa_char: str) -> int:
        return MSA_SYMBOL_INDICES.get(msa_char, -1)

    def is_msa_char(msa_char: str) -> bool:
        return msa_char in MSA_SYMBOL_INDICES
