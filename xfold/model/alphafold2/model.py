import torch
import torch.nn as nn
from typing import List
from xfold.model.alphafold2.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateEmbedder,
    ExtraMSAEmbedder,
)
from xfold.model.alphafold2.evoformer import EvoformerStack
from xfold.model.common.structure import StructureModule
from xfold.protein.sequence import MSA, Sequence
from xfold.protein.structure import ProteinStructure, TemplateProtein


class AlphaFold2Config:
    # Inference
    n_recycling_iters: int = 3
    n_ensembles: int = 1

    # MSA bootstrapping
    max_n_msa_seqs: int = 512
    max_n_extra_msa_seqs: int = 1024

    # Embedding
    msa_dim: int = 256
    pair_dim: int = 128

    ## Input embedding
    max_relpos_dist: int = 32

    ## Recycling embedding
    min_cb_dist: float = 3.375
    max_cb_dist: float = 21.375
    n_cb_bins: int = 15

    ## Template embedding
    temp_dim: int = 64
    n_tri_attn_heads: int = 4
    n_pw_attn_heads: int = 4
    n_temp_pair_blocks: int = 2
    temp_pair_trans_dim_scale: int = 2

    ## Extra MSA embedding
    extra_msa_dim: int = 64
    extra_msa_row_attn_dim: int = 8
    extra_msa_col_global_attn_dim: int = 8
    outer_prod_msa_dim: int = 32
    n_msa_attn_heads: int = 8
    n_extra_msa_blocks: int = 4
    msa_trans_dim_scale: int = 4
    pair_trans_dim_scale: int = 4

    # Evoformer (mostly shared with extra MSA)
    single_dim: int = 384
    msa_row_attn_dim: int = 32
    msa_col_attn_dim: int = 32
    n_evoformer_blocks: int = 48

    # Structure module
    struct_proj_dim: int = 128
    ipa_dim: int = 16
    n_ipa_heads: int = 12
    n_ipa_query_points: int = 4
    n_ipa_point_values: int = 8
    n_struct_module_layers: int = 8


class AlphaFold2(nn.Module):
    def __init__(
        self,
        n_recycling_iters: int = 3,
        n_ensembles: int = 1,
        max_n_msa_seqs: int = 512,
        max_n_extra_msa_seqs: int = 1024,
        msa_dim: int = 256,
        pair_dim: int = 128,
        max_relpos_dist: int = 32,
        min_cb_dist: float = 3.375,
        max_cb_dist: float = 21.375,
        n_cb_bins: int = 15,
        temp_dim: int = 64,
        n_tri_attn_heads: int = 4,
        n_pw_attn_heads: int = 4,
        n_temp_pair_blocks: int = 2,
        temp_pair_trans_dim_scale: int = 2,
        extra_msa_dim: int = 64,
        extra_msa_row_attn_dim: int = 32,
        extra_msa_col_global_attn_dim: int = 8,
        outer_prod_msa_dim: int = 32,
        n_msa_attn_heads: int = 8,
        n_extra_msa_blocks: int = 4,
        msa_trans_dim_scale: int = 4,
        pair_trans_dim_scale: int = 4,
        single_dim: int = 384,
        msa_row_attn_dim: int = 32,
        msa_col_attn_dim: int = 32,
        n_evoformer_blocks: int = 48,
        struct_proj_dim: int = 128,
        ipa_dim: int = 16,
        n_ipa_heads: int = 12,
        n_ipa_query_points: int = 4,
        n_ipa_point_values: int = 8,
        n_struct_module_layers: int = 8,
    ):
        super().__init__()
        self.msa_dim = msa_dim
        self.pair_dim = pair_dim

        self.n_cycling_iters = n_recycling_iters
        self.n_ensembles = n_ensembles
        self.max_n_msa_seqs = max_n_msa_seqs
        self.max_n_extra_msa_seqs = max_n_extra_msa_seqs

        # Embedders of evolutionary information
        self.input_embedder = InputEmbedder(msa_dim, pair_dim, max_relpos_dist)

        cb_bins = torch.linspace(min_cb_dist, max_cb_dist, n_cb_bins)
        self.recycling_embedder = RecyclingEmbedder(msa_dim, pair_dim, cb_bins)

        self.template_embedder = TemplateEmbedder(
            msa_dim,
            pair_dim,
            temp_dim,
            n_tri_attn_heads,
            n_pw_attn_heads,
            n_temp_pair_blocks,
            temp_pair_trans_dim_scale,
        )

        self.extra_msa_embedder = ExtraMSAEmbedder(
            extra_msa_dim,
            pair_dim,
            extra_msa_row_attn_dim,
            extra_msa_col_global_attn_dim,
            outer_prod_msa_dim,
            n_tri_attn_heads,
            n_msa_attn_heads,
            msa_trans_dim_scale,
            pair_trans_dim_scale,
            n_extra_msa_blocks,
        )

        # Evoformer stack
        self.evoformer_stack = EvoformerStack(
            msa_dim,
            pair_dim,
            single_dim,
            msa_row_attn_dim,
            msa_col_attn_dim,
            outer_prod_msa_dim,
            n_tri_attn_heads,
            n_msa_attn_heads,
            msa_trans_dim_scale,
            pair_trans_dim_scale,
            n_evoformer_blocks,
        )

        # Structure module
        self.structure_module = StructureModule(
            single_dim,
            pair_dim,
            struct_proj_dim,
            ipa_dim,
            n_ipa_heads,
            n_ipa_query_points,
            n_ipa_point_values,
            n_struct_module_layers,
        )

    def forward(
        self,
        target: Sequence,
        msa: MSA,
        extra_msa: MSA,
        templates: List[TemplateProtein],
        target_struct: ProteinStructure = None,
    ) -> ProteinStructure:
        N_RES = target.length()

        n_ensembles = 1 if self.training else self.n_ensembles

        res_index = target.seq_index
        target_feat = target.seq_one_hot

        temp_angle_feats = torch.stack(
            [temp.template_angle_feat for temp in templates], dim=0
        )
        temp_pair_feats = torch.stack(
            [temp.template_pair_feat for temp in templates], dim=0
        )

        prev_avg_target_msa_rep = torch.zeros((self.msa_dim,))
        prev_avg_pair_rep = torch.zeros((N_RES, N_RES, self.pair_dim))
        prev_avg_struct_cb = torch.zeros((N_RES, 3))
        for _ in range(self.n_cycling_iters):
            avg_target_msa_rep = 0
            avg_pair_rep = 0
            avg_single_rep = 0
            for _ in range(n_ensembles):
                msa_cn = msa.sample_msa(self.max_n_msa_seqs)
                extra_msa_cn = extra_msa.sample_msa(self.max_n_extra_msa_seqs)

                msa_feat_cn = msa_cn.msa_feat
                extra_msa_feat_cn = extra_msa_cn.msa_feat

                # Input embedder
                msa_rep, pair_rep = self.input_embedder(
                    target_feat, res_index, msa_feat_cn
                )
                # Recycling embedder
                target_msa_rep, pair_rep = self.recycling_embedder(
                    prev_avg_target_msa_rep, prev_avg_pair_rep, prev_avg_struct_cb
                )
                # Templates embedder
                msa_rep, pair_rep = self.template_embedder(
                    msa_rep, pair_rep, temp_angle_feats, temp_pair_feats
                )
                # Extra MSAs embedder
                pair_rep = self.extra_msa_embedder(extra_msa_feat_cn, pair_rep)

                # Evoformer stack
                msa_rep, pair_rep, single_rep = self.evoformer_stack(msa_rep, pair_rep)

                avg_target_msa_rep = avg_target_msa_rep + target_msa_rep
                avg_pair_rep = avg_pair_rep + pair_rep
                avg_single_rep = avg_single_rep + single_rep

            avg_target_msa_rep = avg_target_msa_rep / n_ensembles
            avg_pair_rep = avg_pair_rep / n_ensembles
            avg_single_rep = avg_single_rep / n_ensembles

            # Structure module
            pred_struct, plddt, *losses = self.structure_module(
                avg_single_rep, avg_pair_rep, res_index, target_struct
            )

            prev_avg_target_msa_rep = avg_target_msa_rep
            prev_avg_pair_rep = avg_pair_rep
            prev_avg_struct_cb = pred_struct.get_beta_carbon_coords()

        return pred_struct, plddt, losses
