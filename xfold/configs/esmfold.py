from dataclasses import dataclass
from xfold.configs.base import BaseModelConfig


@dataclass
class ESMFoldConfig(BaseModelConfig):
    seq_len: int = 1024
    single_dim: int = 1024
    pair_dim: int = 128

    # ESM2-model (following 3B model)
    embed_dim: int = 2560
    lm_attn_head_dim: int = 64
    n_lm_attn_heads: int = 40
    n_lm_encoder_blocks: int = 36
    use_attn_map: bool = False

    # Input embedding
    max_relpos_dist: int = 32

    # Folding trunk
    n_recycling_iters: int = 3

    ## Recycling embedding
    min_cb_dist: float = 3.375
    max_cb_dist: float = 21.375
    n_cb_bins: int = 15

    ## Folding blocks
    fold_block_single_head_dim: int = 32
    fold_block_pair_head_dim: int = 32
    n_folding_blocks: int = 48

    ## Structure module
    struct_proj_dim: int = 128
    ipa_dim: int = 16
    plddt_head_dim: int = 128
    n_ipa_heads: int = 12
    n_ipa_query_points: int = 4
    n_ipa_point_values: int = 8
    n_plddt_bins: int = 50
    n_struct_module_layers: int = 8
