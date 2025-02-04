import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple
from xfold.configs.esmfold import ESMFoldConfig
from xfold.model import register_folding_model
from xfold.model.base import BaseFoldingModel
from xfold.model.common.misc import RelPos
from xfold.model.common.embedders import RecyclingEmbedder
from xfold.model.common.structure import StructureModule
from xfold.model.esmfold.constants import (
    esm_to_alphafold_indices,
    ESMVocab,
    START_TOKEN,
    END_TOKEN,
    PAD_TOKEN,
)
from xfold.model.esmfold.folding_block import FoldingBlock
from xfold.model.esmfold.lm import ESM2
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
        n_recycling_iters: int = 3,
        max_relpos_dist: int = 32,
        min_cb_dist: float = 3.375,
        max_cb_dist: float = 21.375,
        n_cb_bins: int = 15,
        fold_block_single_head_dim: int = 32,
        fold_block_pair_head_dim: int = 32,
        n_folding_blocks: int = 48,
        struct_proj_dim: int = 128,
        ipa_dim: int = 16,
        plddt_head_dim: int = 128,
        n_ipa_heads: int = 12,
        n_ipa_query_points: int = 4,
        n_ipa_point_values: int = 8,
        n_plddt_bins: int = 50,
        n_struct_module_layers: int = 8,
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

        dist_bins = torch.arange(-max_relpos_dist, max_relpos_dist + 1).float()
        self.res_index_to_pair = RelPos(pair_dim, dist_bins)

        # Folding trunk
        self.n_recycling_iters = n_recycling_iters

        ## Recycling embedder
        cb_bins = torch.linspace(min_cb_dist, max_cb_dist, n_cb_bins)
        self.recycling_embedder = RecyclingEmbedder(single_dim, pair_dim, cb_bins)

        ## Folding blocks
        self.folding_blocks = nn.ModuleList(
            [
                FoldingBlock(
                    single_dim,
                    pair_dim,
                    fold_block_single_head_dim,
                    fold_block_pair_head_dim,
                )
                for _ in range(n_folding_blocks)
            ]
        )

        ## Structure module
        plddt_bins = torch.linspace(1, 99, n_plddt_bins)
        self.structure_module = StructureModule(
            single_dim,
            pair_dim,
            struct_proj_dim,
            ipa_dim,
            plddt_head_dim,
            n_ipa_heads,
            n_ipa_query_points,
            n_ipa_point_values,
            plddt_bins,
            n_struct_module_layers,
        )

    def fold(self, seq: str) -> Tuple[ProteinStructure, Dict[str, Any]]:
        # While ESMVocab contains uncommon amino acids, leveraging AF2's
        # structure module entails conforming to the standard 20 amino acids.
        Sequence(seq)  # validates sequence
        seq = Sequence(seq, vocab=ESMVocab)

        struct, plddt, *_ = self.forward(seq)
        aux_preds = {"plddt": plddt}
        return struct, aux_preds

    def forward(
        self, target: Sequence, target_struct: ProteinStructure = None
    ) -> Tuple[ProteinStructure, float, list[float]]:
        N_RES = target.length()
        MAX_SEQ_LEN = self.seq_len
        if N_RES + 2 > MAX_SEQ_LEN:
            raise Exception(
                f"Target sequence is too long ({target.length} > {self.seq_len - 2})"
            )

        token_ids = torch.full((1, MAX_SEQ_LEN), ESMVocab.get_index(PAD_TOKEN))
        token_ids[:, 0] = ESMVocab.get_index(START_TOKEN)
        token_ids[:, 1 : N_RES + 1] = target.seq_index
        token_ids[:, N_RES + 1] = ESMVocab.get_index(END_TOKEN)

        mask = (token_ids == ESMVocab.get_index(PAD_TOKEN)).long()

        # Get initial single and pair representations
        reps = self.lm(token_ids, mask, self.use_attn_map)

        ## Single representation is weighted average of hidden states in encoder layers
        ## Hidden states shape is: n_layers + 1 x batch x seq_len x embed_dim
        single_rep = reps["hidden_states"][:, :, 1 : N_RES + 1, :]
        single_rep = (
            F.softmax(self.single_rep_weights, dim=0).view(-1, 1, 1, 1) * single_rep
        ).sum(dim=0)
        single_rep = self.single_proj(single_rep)

        ## Pair representation is either:
        ##  (1) Pairwise relpos of single_rep
        ##  (2) MLP(attention maps throughout encoder layers)
        if not self.use_attn_map:
            ## It's zero here, but the folding trunk adds the relpos at each cycling step
            pair_rep = torch.zeros((1, N_RES, N_RES, self.pair_dim))
        else:
            ## Attention maps shape is: n_layers x batch x seq_len x seq_len x n_heads
            pair_rep = self.pair_proj(
                reps["attn_maps"].moveaxis(0, -1).flatten(-2, -1)
            )[:, 1 : N_RES + 1, 1 : N_RES + 1]

        ## Note: seq_len (ESM context length) has been truncated to n_res
        ## (This differs from official implementation)
        single_rep_init = single_rep  # batch x n_res x single_dim
        pair_rep_init = pair_rep  # batch x n_res x n_res x pair_dim

        # Folding trunk
        prev_single_rep = torch.zeros((1, N_RES, self.single_dim))
        prev_pair_rep = torch.zeros((1, N_RES, N_RES, self.pair_dim))
        prev_struct_cb = torch.zeros((1, N_RES, 3))
        for _ in range(self.n_recycling_iters):
            ## Recycling embedder
            single_update, pair_update = self.recycling_embedder(
                prev_single_rep, prev_pair_rep, prev_struct_cb
            )
            single_rep = single_rep_init + single_update
            pair_rep = pair_rep_init + pair_update

            ## Folding blocks
            res_index = esm_to_alphafold_indices(target.seq_index)
            pair_rep = pair_rep + self.res_index_to_pair(res_index).unsqueeze(0)
            for folding_block in self.folding_blocks:
                single_rep, pair_rep = folding_block(single_rep, pair_rep)

            ## Structure module
            pred_struct, plddt, *losses = self.structure_module(
                single_rep[0], pair_rep[0], res_index, target_struct
            )

            prev_single_rep = single_rep
            prev_pair_rep = pair_rep
            prev_struct_cb = pred_struct.get_beta_carbon_coords()

        return pred_struct, plddt, losses

    def freeze_lm(self) -> None:
        for param in self.lm.parameters():
            param.requires_grad = False
