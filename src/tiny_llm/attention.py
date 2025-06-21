import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # key: (N.., seq_len, head_dim)
    # value: (N.., seq_len, head_dim)
    # query: (N.., seq_len, head_dim)
    # mask: (N.., seq_len, seq_len)
    head_dim = query.shape[-1]
    factor = mx.rsqrt(head_dim) if scale is None else scale
    scores = query @ key.swapaxes(-2, -1) * factor # (N.., seq_len, seq_len)
    if mask is not None:
        scores += mask
    return softmax(scores, axis=-1) @ value # (N.., seq_len, head_dim)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads
        self.wq = wq # (hidden_size, num_heads * head_dim)
        self.wk = wk # (hidden_size, num_heads * head_dim)
        self.wv = wv # (hidden_size, num_heads * head_dim)
        self.wo = wo # (num_heads * head_dim, hidden_size)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # query: (N.., L, hidden_size)
        # key: (N.., L, hidden_size)
        # value: (N.., L, hidden_size)
        # mask: (N.., L, L)
        # output: (N.., L, hidden_size)
        Q = linear(query, self.wq) # (N.., L, num_heads * head_dim)
        K = linear(key, self.wk) # (N.., L, num_heads * head_dim)
        V = linear(value, self.wv) # (N.., L, num_heads * head_dim)
        query_heads = Q.reshape(*Q.shape[:-1], self.num_heads, self.head_dim).swapaxes(-2, -3) # (N.., num_heads, L, head_dim)
        key_heads = K.reshape(*K.shape[:-1], self.num_heads, self.head_dim).swapaxes(-2, -3) # (N.., num_heads, L, head_dim)
        value_heads = V.reshape(*V.shape[:-1], self.num_heads, self.head_dim).swapaxes(-2, -3) # (N.., num_heads, L, head_dim)
        output_heads = scaled_dot_product_attention_simple(query_heads, key_heads, value_heads, mask=mask) # (N.., num_heads, L, head_dim)
        output = output_heads.swapaxes(-2, -3).reshape(*Q.shape[:-1], -1) # (N.., L, num_heads, head_dim)
        return linear(output, self.wo) # (N.., L, hidden_size)

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
