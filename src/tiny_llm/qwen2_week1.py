import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(hidden_size, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # B: batch size
        # L: sequence length
        # E: embedding dimension
        # x: (B, L, E)
        B, L, _ = x.shape

        # 1) Linear projection
        projection_q = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_dim) # (B, L, H_q, D)
        projection_k = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, self.head_dim) # (B, L, H, D)
        projection_v = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, self.head_dim) # (B, L, H, D)

        # 2) positional encoding
        projection_q = self.rope(projection_q, offset=slice(offset, offset + L))
        projection_k = self.rope(projection_k, offset=slice(offset, offset + L))

        # 3) transpose
        projection_q = projection_q.transpose(0, 2, 1, 3) # (B, H_q, L, D)
        projection_k = projection_k.transpose(0, 2, 1, 3) # (B, H, L, D)
        projection_v = projection_v.transpose(0, 2, 1, 3) # (B, H, L, D)

        # 4) attention
        x = scaled_dot_product_attention_grouped(
            projection_q.astype(mx.float32),
            projection_k.astype(mx.float32),
            projection_v.astype(mx.float32),
            mask=mask,
        ) # (B, H_q, L, D)

        # 5) merge heads
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size) # (B, L, E)

        # 6) linear projection
        return linear(x, self.wo) # (B, L, E)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
