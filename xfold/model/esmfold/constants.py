from typing import List
from xfold.protein.constants import AA_ALPHABET, Vocab


AA_UNCOMMON = "BOUZ"
AA_UNCOMMON_EXPAND = {
    "B": "ASX",
    "O": "PYL",
    "U": "SEC",
    "Z": "GLX",
}
SEQ_GAP = "-"
DUMMY_TOKEN = "."  # So that encoding is 32-dims. See https://github.com/facebookresearch/esm/issues/84"
START_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"
END_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"

ESM_ALPHABET = (
    list(AA_ALPHABET)
    + list(AA_UNCOMMON)
    + [SEQ_GAP, DUMMY_TOKEN, START_TOKEN, PAD_TOKEN, END_TOKEN, UNK_TOKEN, MASK_TOKEN]
)
assert len(ESM_ALPHABET) == 32
ESM_INDICES = {token: i for i, token in enumerate(ESM_ALPHABET)}


class ESMVocab(Vocab):
    vocab = ESM_ALPHABET
    vocab_size = len(ESM_ALPHABET)

    def get_index(token: str) -> int:
        return ESM_INDICES.get(token, ESM_INDICES[UNK_TOKEN])

    def is_valid(token: str) -> bool:
        return token in ESM_INDICES

    def tokenize_sequence(seq: str) -> List[str]:
        # We can manually tokenize since tokens are simple and a unique mapping exists.
        OPEN = "<"
        CLOSE = ">"

        tokens = []
        curr_token = []
        is_open = False
        is_invalid = False

        for char in seq:
            if (char == OPEN and is_open) or (char == CLOSE and not is_open):
                is_invalid = True
                break
            if char == OPEN:
                is_open = True
                curr_token.append(OPEN)
            elif char == CLOSE:
                is_open = False
                curr_token.append(CLOSE)
                tokens.append("".join(curr_token))
                curr_token = []
            elif is_open:
                curr_token.append(char)
            else:
                tokens.append(char)

        if curr_token:
            is_invalid = True
        if is_invalid:
            raise Exception(f"Invalid sequence. Can't tokenize '{seq}'.")

        return tokens

    def index_sequence(seq: str) -> List[int]:
        return [ESMVocab.get_index(token) for token in ESMVocab.tokenize_sequence(seq)]
