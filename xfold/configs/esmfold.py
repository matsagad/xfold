from dataclasses import dataclass
from xfold.configs.base import BaseModelConfig


@dataclass
class ESMFoldConfig(BaseModelConfig):
    seq_len: int = 1024
    single_dim: int = 1024
    pair_dim: int = 128

    # ESM2-model
    embed_dim: int = 2560
    lm_attn_head_dim: int = 64
    n_lm_attn_heads: int = 40
    n_lm_encoder_blocks: int = 36
    use_attn_map: bool = False
