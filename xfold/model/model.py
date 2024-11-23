import torch.nn as nn
from typing import List
from xfold.model.embedder import InputEmbedder
from xfold.protein.sequence import MSA
from xfold.protein.structure import ProteinFrames, ProteinStructure, TemplateProtein


class AlphaFold2Config:
    # Inference configs
    n_recycling_iters: int = 3
    n_ensembles: int = 1

    # MSA bootstrapping configs
    max_n_msa_seqs: int = 512
    max_n_extra_msa_seqs: int = 1024

    # Embedding configs
    msa_embed_dim: int = 256
    pair_embed_dim: int = 128


class AlphaFold2(nn.Module):
    def __init__(
        self,
        n_recycling_iters: int = 3,
        n_ensembles: int = 1,
        max_n_msa_seqs: int = 512,
        max_n_extra_msa_seqs: int = 1024,
        msa_embed_dim: int = 256,
        pair_embed_dim: int = 128,
    ):
        super().__init__()
        self.n_cycling_iters = n_recycling_iters
        self.n_ensembles = n_ensembles
        self.max_n_msa_seqs = max_n_msa_seqs
        self.max_n_extra_msa_seqs = max_n_extra_msa_seqs

        self.input_embedder = InputEmbedder(msa_embed_dim, pair_embed_dim)

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
        n_res = target.seq.length()

        prev_avg_msa_rep = 0
        prev_avg_pair_rep = 0
        prev_avg_struct_rep = ProteinFrames.zero_init(n_res)
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

                # Templates embedder

                # Extra MSAs embedder

                # Evoformer stack

            # Structure module
