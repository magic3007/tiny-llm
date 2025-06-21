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
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

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
        ).astype(x.dtype) # (B, H_q, L, D)

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

        gate = linear(x, self.w_gate) # (N.., L, I)
        up = linear(x, self.w_up) # (N.., L, I)
        return linear(silu(gate) * up, self.w_down) # (N.., L, E)


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
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.qwen2_multi_head_attention(self.input_layernorm(x), offset, mask) # (B, L, E)
        h = x + r # (B, L, E)
        r = self.qwen2_mlp(self.post_attention_layernorm(h)) # (B, L, E)
        out = h + r
        return out


class Qwen2ModelWeek1:
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

        print(f"hidden_size: {self.hidden_size}")
        print(f"num_hidden_layers: {self.num_hidden_layers}")
        print(f"intermediate_size: {self.intermediate_size}")
        print(f"num_attention_heads: {self.num_attention_heads}")
        print(f"max_position_embeddings: {self.max_position_embeddings}")
        print(f"rms_norm_eps: {self.rms_norm_eps}")
        print(f"vocab_size: {self.vocab_size}")

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(self.precision),
        )
        self.layers_inner = []

        for i in range(self.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj).astype(self.precision)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj).astype(self.precision)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj).astype(self.precision)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj).astype(self.precision)
            bq = mlx_model.model.layers[i].self_attn.q_proj.bias.astype(self.precision)
            bk = mlx_model.model.layers[i].self_attn.k_proj.bias.astype(self.precision)
            bv = mlx_model.model.layers[i].self_attn.v_proj.bias.astype(self.precision)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj).astype(self.precision)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj).astype(self.precision)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj).astype(self.precision)
            w_input_layernorm = mlx_model.model.layers[i].input_layernorm.weight.astype(self.precision)
            w_post_attention_layernorm = mlx_model.model.layers[i].post_attention_layernorm.weight.astype(self.precision)

            layer = Qwen2TransformerBlock(
                num_attention_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                rms_norm_eps=self.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=bq,
                bk=bk,
                bv=bv,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=w_input_layernorm,
                w_post_attention_layernorm=w_post_attention_layernorm,
                max_seq_len=self.max_position_embeddings,
                theta=self.rope_theta
            )

            self.layers_inner.append(layer)

        self.norm = RMSNorm(
            self.hidden_size,
            weight=self.mlx_model.model.norm.weight.astype(self.precision),
            eps=self.rms_norm_eps
        )

        if not self.mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(self.mlx_model.lm_head)
        else:
            self.w_lm_head = None

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        x = self.embedding(inputs)
        for layer in self.layers_inner:
            x = layer(x, offset, mask="causal" if x.shape[1] > 1 else None)
        x = self.norm(x)
        if self.w_lm_head is not None:
            return linear(x, self.w_lm_head)
        else:
            return self.embedding.as_linear(x)

