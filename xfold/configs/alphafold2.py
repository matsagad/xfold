from dataclasses import dataclass
from xfold.configs.base import BaseModelConfig


@dataclass
class AlphaFold2Config(BaseModelConfig):
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
    plddt_head_dim: int = 128
    n_ipa_heads: int = 12
    n_ipa_query_points: int = 4
    n_ipa_point_values: int = 8
    n_plddt_bins: int = 50
    n_struct_module_layers: int = 8
