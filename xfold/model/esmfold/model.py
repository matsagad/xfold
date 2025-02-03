import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple
from xfold.configs.esmfold import ESMFoldConfig
from xfold.model import register_folding_model
from xfold.model.base import BaseFoldingModel
from xfold.model.esmfold.constants import ESMVocab, START_TOKEN, END_TOKEN, PAD_TOKEN
from xfold.model.esmfold.lm import ESM1, ESM2
from xfold.protein.sequence import Sequence
from xfold.protein.structure import ProteinStructure


@register_folding_model("esmfold", ESMFoldConfig)
class ESMFold(nn.Module, BaseFoldingModel):
    def __init__(
        self,
        seq_len: int = 1024,
        single_dim: int = 1024,
        pair_dim: int = 128,
        embed_dim: int = 2560,
        lm_attn_head_dim: int = 64,
        n_lm_attn_heads: int = 40,
        n_lm_encoder_blocks: int = 36,
        use_attn_map: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        # ESM2 language model
        self.lm = ESM2(
            seq_len,
            embed_dim,
            lm_attn_head_dim,
            n_lm_attn_heads,
            n_lm_encoder_blocks,
        )

        # Input embedder
        ## Single representation
        self.single_dim = single_dim
        self.single_rep_weights = nn.Parameter(torch.randn((n_lm_encoder_blocks + 1,)))
        self.single_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, single_dim),
            nn.ReLU(),
            nn.Linear(single_dim, single_dim),
        )

        ## Pair representation
        self.pair_dim = pair_dim
        self.use_attn_map = use_attn_map
        if use_attn_map:
            flat_attn_map_dim = n_lm_encoder_blocks * n_lm_attn_heads
            self.pair_proj = nn.Sequential(
                nn.LayerNorm(flat_attn_map_dim),
                nn.Linear(flat_attn_map_dim, pair_dim),
                nn.ReLU(),
                nn.Linear(pair_dim, pair_dim),
            )

        # Folding trunk
        self.folding_trunk = None

        # Structure module
        self.structure_module = None

    def fold(self, seq: str) -> Tuple[ProteinStructure, Dict[str, Any]]:
        seq = Sequence(seq, vocab=ESMVocab)

        struct, plddt, *_ = self.forward(seq)
        aux_preds = {"plddt": plddt}
        return struct, aux_preds

    def forward(
        self, target: Sequence, target_struct: ProteinStructure = None
    ) -> Tuple[ProteinStructure, float, list[float]]:
        target_seq_len = target.length()
        if target_seq_len + 2 > self.seq_len:
            raise Exception(
                f"Target sequence is too long ({target.length} > {self.seq_len - 2})"
            )
        token_ids = torch.full((1, self.seq_len), ESMVocab.get_index(PAD_TOKEN))
        token_ids[:, 0] = ESMVocab.get_index(START_TOKEN)
        token_ids[:, 1 : target_seq_len + 1] = target.seq_index
        token_ids[:, target_seq_len + 1] = ESMVocab.get_index(END_TOKEN)

        # Get initial single and pair representations
        res = self.lm(token_ids, self.use_attn_map)

        ## Single representation is weighted average of hidden states throughout encoder layers
        ## Hidden states shape is: n_layers + 1 x batch x seq_len x embed_dim
        single_rep = res["hidden_states"][:, :, 1 : target_seq_len + 1]
        single_rep = (
            F.softmax(self.single_rep_weights, dim=0).view(-1, 1, 1, 1) * single_rep
        ).sum(dim=0)
        single_rep = self.single_proj(single_rep)

        ## Pair representation is either:
        ##  - Pairwise relpos of single_rep
        ##  - MLP(attention maps throughout encoder layers)
        if self.use_attn_map:
            ## Attention maps shape is: n_layers x batch x seq_len x seq_len x n_heads
            pair_rep = self.pair_proj(res["attn_maps"].moveaxis(0, -1).flatten(-2, -1))
        else:
            ## It is zero here, but the folding trunk adds the relpos at each block
            pair_rep = torch.zeros((1, target_seq_len, target_seq_len, self.pair_dim))

        # TODO: Folding trunk

        return None, None, None

    def freeze_lm(self) -> None:
        for param in self.lm.parameters():
            param.requires_grad = False
