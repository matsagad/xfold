import torch
import torch.nn.functional as F
from typing import List
from constants import AminoAcidVocab, MSAVocab, MSA_GAP_SYMBOL


class Sequence:
    def __init__(self, seq: str) -> None:
        self._validate_sequence(seq)
        self.seq_str = seq.upper()

        self.seq = F.one_hot(
            torch.tensor([AminoAcidVocab.get_index(aa) for aa in self.seq_str]),
            AminoAcidVocab.vocab_size,
        )

    def __str__(self) -> str:
        return f'Sequence("{self.seq_str}")'

    def _validate_sequence(self, seq: str) -> None:
        for char in seq:
            if not char.isalpha():
                raise Exception(f"'{char}' is not an alpha character.")
            if not AminoAcidVocab.is_amino_acid(char.upper()):
                raise Exception(f"'{char}' is not a valid amino acid code.")


class MSA:
    def __init__(self, msa: List[str]) -> None:
        self._validate_msa(msa)
        self.msa_str = list(map(str.upper, msa))
        self.msa_feat = self._build_msa_feature_matrix(msa)

    def _validate_msa(self, msa: List[str]) -> None:
        if not msa:
            raise Exception("MSA is empty.")
        n_cols = len(msa[0])
        for row in msa:
            if len(row) != n_cols:
                raise Exception("MSA has inconsistent number of columns.")
            for char in row:
                if not MSAVocab.is_msa_char(char.upper()):
                    raise Exception(f"MSA has invalid character: '{char}'.")

    def _normalise_deletion_count(self, d: torch.Tensor) -> torch.Tensor:
        return 2 / torch.pi * torch.arctan(d / 3)

    def _build_msa_feature_matrix(self, msa: List[str]) -> torch.Tensor:
        N_clust, N_res, N_msa_feats = len(msa), len(msa[0]), 49
        msa_feat = torch.empty((N_clust, N_res, N_msa_feats))

        # One-hot representation
        N_one_hot = MSAVocab.vocab_size
        msa_feat[:, :, :N_one_hot] = F.one_hot(
            torch.tensor(
                [
                    [MSAVocab.get_index(msa_char) for msa_char in row]
                    for row in self.msa_str
                ]
            ),
            MSAVocab.vocab_size,
        )

        # Has deletion: is there a deletion to the left
        GAP_INDEX = MSAVocab.get_index(MSA_GAP_SYMBOL)
        msa_feat[:, 1:, N_one_hot] = msa_feat[:, :-1, GAP_INDEX] == 1
        msa_feat[:, 0, N_one_hot] = 0

        # Deletion value: normalised total deletions to the left
        cum_deletion_count = torch.cumsum(msa_feat[:, :, N_one_hot], dim=1)
        msa_feat[:, :, N_one_hot + 1] = self._normalise_deletion_count(
            cum_deletion_count
        )

        # Deletion mean: normalised mean, across all sequences, of deletions to the left
        msa_feat[:, :, N_one_hot + 2] = self._normalise_deletion_count(
            cum_deletion_count.mean(dim=0, keepdim=True)
        )

        # Cluster profile: amino acid and gap distribution for each residue position
        msa_feat[:, :, N_one_hot + 3 : 2 * N_one_hot + 3] = msa_feat[
            :, :, :N_one_hot
        ].mean(dim=0, keepdim=True)

        return msa_feat
