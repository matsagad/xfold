import torch
import torch.nn.functional as F
from typing import List
from xfold.protein.constants import AminoAcidVocab, MSAVocab, MSA_GAP_SYMBOL, Vocab


class Sequence:
    def __init__(self, seq: str, vocab: Vocab = AminoAcidVocab) -> None:
        self.vocab = vocab
        self._validate_sequence(seq, vocab)
        self.seq_str = seq.upper()
        self.seq_index = Sequence.residue_index(self.seq_str, vocab)
        self.seq_one_hot = Sequence.one_hot_encode(self.seq_str, vocab)

    def residue_index(seq: str, vocab: Vocab = AminoAcidVocab) -> torch.Tensor:
        return torch.tensor(vocab.index_sequence(seq)).long()

    def one_hot_encode(seq: str, vocab: Vocab = AminoAcidVocab) -> torch.Tensor:
        return F.one_hot(
            torch.tensor(vocab.index_sequence(seq)), vocab.vocab_size
        ).float()

    def length(self) -> int:
        return len(self.seq_index)

    def __str__(self) -> str:
        return f'Sequence("{self.seq_str}")'

    def _validate_sequence(self, seq: str, vocab: Vocab = AminoAcidVocab) -> None:
        for char in seq:
            if not char.isalpha():
                raise Exception(f"'{char}' is not an alpha character.")
            if not vocab.is_valid(char.upper()):
                raise Exception(f"'{char}' is not a valid amino acid code.")


class MSA:
    N_MSA_FEATS = 49
    N_EXTRA_MSA_FEATS = 25

    def __init__(self, msa: List[str], is_extra: bool = False) -> None:
        self._validate_msa(msa)
        self.is_extra = is_extra
        self.msa_str = list(map(str.upper, msa))
        self.msa_feat = self._build_msa_feature_matrix(msa, is_extra)

    def one_hot_encode(msa: List[str]) -> torch.Tensor:
        return F.one_hot(
            torch.tensor([MSAVocab.index_sequence(seq) for seq in msa]),
            MSAVocab.vocab_size,
        )

    def _validate_msa(self, msa: List[str]) -> None:
        if not msa:
            raise Exception("MSA is empty.")
        n_cols = len(msa[0])
        for seq in msa:
            if len(seq) != n_cols:
                raise Exception("MSA has inconsistent number of columns.")
            for char in seq:
                if not MSAVocab.is_msa_char(char.upper()):
                    raise Exception(f"MSA has invalid character: '{char}'.")

    def _normalise_deletion_count(self, d: torch.Tensor) -> torch.Tensor:
        return 2 / torch.pi * torch.arctan(d / 3)

    def _build_msa_feature_matrix(
        self, msa: List[str], is_extra: bool = False
    ) -> torch.Tensor:
        N_clust, N_res, N_msa_feats = len(msa), len(msa[0]), self.N_MSA_FEATS
        msa_feat = torch.empty((N_clust, N_res, N_msa_feats))

        # One-hot representation
        N_one_hot = MSAVocab.vocab_size
        msa_feat[:, :, :N_one_hot] = MSA.one_hot_encode(self.msa_str)

        # Has deletion: is there a deletion to the left
        GAP_INDEX = MSAVocab.get_index(MSA_GAP_SYMBOL)
        msa_feat[:, 1:, N_one_hot] = msa_feat[:, :-1, GAP_INDEX] == 1
        msa_feat[:, 0, N_one_hot] = 0

        # Deletion value: normalised total deletions to the left
        cum_deletion_count = torch.cumsum(msa_feat[:, :, N_one_hot], dim=1)
        msa_feat[:, :, N_one_hot + 1] = self._normalise_deletion_count(
            cum_deletion_count
        )

        ## No global/sequence-wide features for extra MSA
        if is_extra:
            return msa_feat[:, :, : N_one_hot + 2]

        # Deletion mean: normalised mean, across all sequences, of deletions to the left
        msa_feat[:, :, N_one_hot + 2] = self._normalise_deletion_count(
            cum_deletion_count.mean(dim=0, keepdim=True)
        )

        # Cluster profile: amino acid and gap distribution for each residue position
        msa_feat[:, :, N_one_hot + 3 : 2 * N_one_hot + 3] = msa_feat[
            :, :, :N_one_hot
        ].mean(dim=0, keepdim=True)

        return msa_feat

    def sample_msa(self, max_n_seqs: int = 512, is_extra: bool = None) -> "MSA":
        # Note: can't just slice through features since deletion mean
        # and cluster profile are MSA-wide/row-wise features
        N_seq = len(self.msa_str)
        indices = torch.randint(1, N_seq, (max_n_seqs - 1,))

        bootstrapped_msa = [self.msa_str[0]]
        for index in indices:
            bootstrapped_msa.append(self.msa_str[index])

        if is_extra is None:
            is_extra = self.is_extra
        return MSA(bootstrapped_msa, is_extra)
