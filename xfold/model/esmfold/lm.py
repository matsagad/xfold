from functools import lru_cache, partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union
from xfold.model.esmfold.constants import ESMVocab, PAD_TOKEN


# Components for ESM (encoder-only) language models


class LearnablePE(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(seq_len, embed_dim)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        pos = torch.cumsum(mask, dim=0) * mask
        return self.emb(pos)


# Reuse module across all attention layers if same dimension
@lru_cache(maxsize=None)
class RoPE(nn.Module):
    def __init__(self, seq_len: int, head_dim: int) -> None:
        super().__init__()
        self.sin = None
        self.cos = None
        self.inv_freq = 1 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self._build_sinusoidal_tables(seq_len)

    # Based on gpt-neox with some tweaks
    def _build_sinusoidal_tables(self, seq_len: int) -> None:
        t = torch.arange(seq_len).float()
        freqs = t[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=1)
        self.cos = emb.cos()[None, :, :, None]  # 1 x seq_len x head_dim x 1
        self.sin = emb.sin()[None, :, :, None]  # 1 x seq_len x head_dim x 1

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qk_dim = q.shape[-3]  # sequence dim
        cos = self.cos[:, :qk_dim]
        sin = self.sin[:, :qk_dim]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class SelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        n_heads: int,
        use_rope: bool = False,
        rope_seq_len: int = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.inv_sqrt_dim = 1 / (self.head_dim**0.5)

        self.to_qkv = nn.Linear(input_dim, 3 * n_heads * head_dim)
        self.out_proj = nn.Linear(n_heads * head_dim, input_dim)

        self.use_rope = use_rope
        if use_rope and rope_seq_len is None:
            raise Exception(
                "Max sequence length must be specified for rotary embeddings."
            )
        self.embed_rotary = RoPE(rope_seq_len, head_dim) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        bias: torch.Tensor = None,
        output_attn_score: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        qkv_shape = (*x.shape[:-1], self.head_dim, self.n_heads)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda u: u.view(qkv_shape), qkv)

        if self.use_rope:
            # sequence length is at dim=-3
            q, k = self.embed_rotary(q, k)

        attn_score = self.inv_sqrt_dim * torch.einsum("ijdh,ikdh->ijkh", q, k)
        if bias is not None:
            attn_score = attn_score + bias
        if mask is not None:
            attn_score[mask == 0] = -1e8

        attn_logits = F.softmax(attn_score, dim=2)
        attn = torch.einsum("ijkh,ikdh->ijdh", attn_logits, v)
        out = self.out_proj(attn.flatten(-2, -1))

        if output_attn_score:
            return out, attn_score
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        proj_dim: int,
        attn_head_dim: int,
        n_attn_heads: int,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.attn = SelfAttention(
            input_dim, attn_head_dim, n_attn_heads, use_rope, seq_len
        )
        self.proj1 = nn.Linear(input_dim, proj_dim)
        self.proj2 = nn.Linear(proj_dim, input_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, output_attn_score: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn = self.attn(self.norm1(x), mask, output_attn_score=output_attn_score)
        if output_attn_score:
            attn, attn_score = attn

        x = x + attn
        x = x + self.proj2(F.gelu(self.proj1(self.norm2(x))))

        if output_attn_score:
            return x, attn_score
        return x


class BERT(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        attn_head_dim: int,
        n_attn_heads: int,
        n_encoder_blocks: int,
        use_rope: bool,
    ) -> None:
        super().__init__()
        self.embed_input = nn.Embedding(ESMVocab.vocab_size, hidden_dim)
        self.use_rope = use_rope  # ESM2 uses RoPE while ESM1 uses learned absolute PEs
        self.embed_pos = None if use_rope else LearnablePE(seq_len, hidden_dim)

        attn_proj_dim = 4 * hidden_dim
        self.encoder_layers = nn.ModuleList(
            [
                Encoder(
                    seq_len,
                    hidden_dim,
                    attn_proj_dim,
                    attn_head_dim,
                    n_attn_heads,
                    use_rope,
                )
                for _ in range(n_encoder_blocks)
            ]
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        output_attn_score: bool = False,
    ) -> Dict[str, Any]:
        hidden_states = []
        attn_maps = []

        x = self.embed_input(token_ids)
        if self.embed_pos is not None:
            x = x + self.embed_pos(mask)
        hidden_states.append(x)

        for encoder in self.encoder_layers:
            x = encoder(x, mask, output_attn_score)
            if output_attn_score:
                x, attn_score = x
                attn_maps.append(attn_score)
            hidden_states.append(x)

        res = {"hidden_states": torch.stack(hidden_states, dim=0)}
        if output_attn_score:
            res["attn_maps"] = torch.stack(attn_maps, dim=0)
        return res


ESM1 = partial(BERT, use_rope=False)
ESM2 = partial(BERT, use_rope=True)
