import torch
import torch.nn as nn
from typing import List
from xfold.model.alphafold2.embedder import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateEmbedder,
    ExtraMSAEmbedder,
)
from xfold.model.alphafold2.evoformer import EvoformerStack
from xfold.protein.sequence import MSA
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
    ):
        super().__init__()
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

    def forward(
        self,
        target: ProteinStructure,
        msa: MSA,
        extra_msa: MSA,
        templates: List[TemplateProtein],
    ) -> ProteinStructure:
        TARGET_SEQ_INDEX = 0
        n_ensembles = 1 if self.training else self.n_ensembles

        residue_index = target.seq.seq_index
        target_feat = target.seq.seq_one_hot
        N_res, N_coords = target.seq.length(), 3

        temp_angle_feats = torch.stack(
            [temp.template_angle_feat for temp in templates], dim=0
        )
        temp_pair_feats = torch.stack(
            [temp.template_pair_feat for temp in templates], dim=0
        )

        prev_avg_target_msa_rep = 0
        prev_avg_pair_rep = 0
        prev_avg_struct_cb = torch.zeros((N_res, N_coords))
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
                    target_feat, residue_index, msa_feat_cn
                )
                # Recycling embedder
                msa_rep, pair_rep = self.recycling_embedder(
                    msa_rep, pair_rep, prev_avg_struct_cb
                )
                # Templates embedder
                msa_rep, pair_rep = self.template_embedder(
                    msa_rep, pair_rep, temp_angle_feats, temp_pair_feats
                )
                # Extra MSAs embedder
                pair_rep = self.extra_msa_embedder(extra_msa_feat_cn, pair_rep)

                # Evoformer stack
                msa_rep, pair_rep, single_rep = self.evoformer_stack(msa_rep, pair_rep)
                target_msa_rep = msa_rep[TARGET_SEQ_INDEX]

                avg_target_msa_rep = avg_target_msa_rep + target_msa_rep
                avg_pair_rep = avg_pair_rep + pair_rep
                avg_single_rep = avg_single_rep + single_rep

            avg_target_msa_rep = avg_target_msa_rep / n_ensembles
            avg_pair_rep = avg_pair_rep / n_ensembles
            avg_single_rep = avg_single_rep / n_ensembles

            # Structure module
