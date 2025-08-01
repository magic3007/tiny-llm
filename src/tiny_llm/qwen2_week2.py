import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights, quantized_linear
from .kv_cache import TinyKvCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        self.head_dim = self.hidden_size // num_heads
        self.wq = wq # (num_heads * head_dim, E)
        self.wk = wk # (num_kv_heads * head_dim, E)
        self.wv = wv # (num_kv_heads * head_dim, E)
        self.wo = wo # (E, hidden_size)
        self.bq = bq 
        self.bk = bk 
        self.bv = bv 
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(self.head_dim, max_seq_len, theta)


    def __call__(
        self,
        x: mx.array,
        offsets: list[int] | int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # B: batch_size
        # L: seq_len
        # E: embedding dimension
        # X: (B, L, E)
        # offsets: (B, )

        B, L, _ = x.shape

        # 1) linear projection
        q_proj = quantized_linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_dim) # (B, L, H_q, D)
        k_proj = quantized_linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, self.head_dim) # (B, L, H, D)
        v_proj = quantized_linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, self.head_dim) # (B, L, H, D)

        # 2) position embedding
        if isinstance(offsets, int):
            offset_slices = [slice(offsets, offsets + L)]
        else:
            offset_slices = [slice(i, i + L) for i in offsets]

        q_proj = self.rope(x=q_proj, offset=offset_slices)
        k_proj = self.rope(x=k_proj, offset=offset_slices)

        # 3) group-query attension
        q_proj = q_proj.transpose(0, 2, 1, 3)
        k_proj = k_proj.transpose(0, 2, 1, 3)
        v_proj = v_proj.transpose(0, 2, 1, 3)

        k_proj, v_proj, _, kv_cache_mask = cache.update_and_fetch(
            k_proj, v_proj, q_L=L
        )
        if kv_cache_mask is not None:
            mask = kv_cache_mask

        x = scaled_dot_product_attention_grouped(
            q_proj.astype(mx.float32), 
            k_proj.astype(mx.float32), 
            v_proj.astype(mx.float32), 
            mask=mask
        ).astype(x.dtype) # (B, H_q, L, D)

        # 4) merge attension heads
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)

        # 5) ouput projection
        return quantized_linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # N.. is zero or more dimensions for batches
        # E is hidden_size (embedding dimension of the model)
        # I is intermediate_size (dimension of the hidden layer in MLP)
        # L is the sequence length

        # input: N.. x L x E
        # w_gate: I x E
        # w_up: I x E
        # w_down: E x I
        # output: N.. x L x E

        gate = quantized_linear(x, self.w_gate) # (N.., L, I)
        up = quantized_linear(x, self.w_up) # (N.., L, I)
        return quantized_linear(silu(gate) * up, self.w_down) # (N.., L, E)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.qwen2_multi_head_attention = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq,
            wk,
            wv,
            wo,
            bq,
            bk,
            bv,
            max_seq_len,
            theta
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)
        self.qwen2_mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.qwen2_multi_head_attention(self.input_layernorm(x), offset, cache=cache, mask=mask) # (B, L, E)
        h = x + r # (B, L, E)
        r = self.qwen2_mlp(self.post_attention_layernorm(h)) # (B, L, E)
        out = h + r
        return out


class Qwen2ModelWeek2:
    def __init__(self, mlx_model: Any):
        self.mlx_model = mlx_model
        self.hidden_size = mlx_model.args.hidden_size
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.intermediate_size = mlx_model.args.intermediate_size
        self.num_attention_heads = mlx_model.args.num_attention_heads
        self.max_position_embeddings = mlx_model.args.max_position_embeddings
        self.rms_norm_eps = mlx_model.args.rms_norm_eps
        self.vocab_size = mlx_model.args.vocab_size
        self.num_key_value_heads = mlx_model.args.num_key_value_heads
        self.rope_theta = mlx_model.args.rope_theta
        self.rope_traditional = mlx_model.args.rope_traditional
        self.precision = mx.float16

        precision = self.precision

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision),
        )
        self.layers_inner = []
        for i in range(mlx_model.args.num_hidden_layers):
            wq = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].self_attn.q_proj)
            wk = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].self_attn.k_proj)
            wv = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].self_attn.v_proj)
            wo = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].mlp.up_proj)
            w_down = QuantizedWeights.from_mlx_layer(mlx_model.model.layers[i].mlp.down_proj)

            precision = self.precision

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=mlx_model.model.layers[
                    i
                ].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = QuantizedWeights.from_mlx_layer(mlx_model.lm_head)
        else:
            self.w_lm_head = None


    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache]
    ) -> mx.array:
        x = self.embedding(inputs)
        mask = "causal" if x.shape[1] > 1 else None
        for i, layer in enumerate(self.layers_inner):
            x = layer(x, offset, cache[i], mask=mask)
        x = self.norm(x)
        if self.w_lm_head is not None:
            return quantized_linear(x, self.w_lm_head)
        else:
            return self.embedding.as_linear(x)