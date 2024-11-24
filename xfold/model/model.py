import torch
import torch.nn as nn
from typing import List
from xfold.model.embedder import InputEmbedder, RecyclingEmbedder, TemplateEmbedder
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
    pair_trans_dim_scale_factor: int = 2
    n_pw_attn_heads: int = 4
    p_dropout_temp_pair_stack: float = 0.25


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
        pair_trans_dim_scale_factor: int = 2,
        n_pw_attn_heads: int = 4,
        p_dropout_temp_pair_stack: float = 0.25,
    ):
        super().__init__()
        self.n_cycling_iters = n_recycling_iters
        self.n_ensembles = n_ensembles
        self.max_n_msa_seqs = max_n_msa_seqs
        self.max_n_extra_msa_seqs = max_n_extra_msa_seqs

        self.input_embedder = InputEmbedder(msa_dim, pair_dim, max_relpos_dist)

        cb_bins = torch.linspace(min_cb_dist, max_cb_dist, n_cb_bins)
        self.recycling_embedder = RecyclingEmbedder(msa_dim, pair_dim, cb_bins)

        self.template_embedder = TemplateEmbedder(
            msa_dim,
            pair_dim,
            temp_dim,
            n_tri_attn_heads,
            pair_trans_dim_scale_factor,
            n_pw_attn_heads,
            p_dropout_temp_pair_stack,
        )

    def forward(
        self,
        target: ProteinStructure,
        msa: MSA,
        extra_msa: MSA,
        templates: List[TemplateProtein],
    ) -> ProteinStructure:
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

        prev_avg_msa_rep = 0
        prev_avg_pair_rep = 0
        prev_avg_struct_cb = torch.zeros((N_res, N_coords))
        for i_cycle in range(self.n_cycling_iters):
            avg_msa_rep = 0
            avg_pair_rep = 0
            avg_single_seq_rep = 0
            for i_ensemble in range(n_ensembles):
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

                # Evoformer stack

            # Structure module
