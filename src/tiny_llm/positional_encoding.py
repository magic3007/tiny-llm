import mlx.core as mx
from numpy import arange


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        # dims: Head Embedding dimension. This is usually set to the dim of each head in the attention module computed as embed_dim // num_heads
        # base: The base for the geometric progression used to compute the rotation angles
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.half_dims = dims // 2

        freqs = mx.arange(0, self.dims, 2).astype(mx.float32) / self.dims
        freqs = mx.power(self.base, -freqs)
        t = mx.arange(self.seq_len, dtype=freqs.dtype) # shape: (seq_len)
        freqs = mx.outer(t, freqs) # shape: (seq_len, dims // 2)
        self.cos_freqs = mx.cos(freqs) # shape: (seq_len, dims // 2)
        self.sin_freqs = mx.sin(freqs) # shape: (seq_len, dims // 2)


    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        """Apply RoPE to input tensor.

        Args:
            x: Input tensor of shape (N, seq_len, num_heads, head_dim)
            offset: Position offset for the rotation frequencies

        Returns:
            Rotated tensor with same shape as input
        """
        N, S, H, D = x.shape
        assert D == self.dims, f"Expected head_dim={self.dims}, got {D}"
        assert S <= self.seq_len, f"Sequence length {S} exceeds maximum {self.seq_len}"

        # Select rotation frequencies based on offset
        if offset is None:
            cos_basis = self.cos_freqs[:S]
            sin_basis = self.sin_freqs[:S]
        else:
            cos_basis = self.cos_freqs[offset, :]
            sin_basis = self.sin_freqs[offset, :]

        # Split head dimension into pairs for rotation
        if self.traditional:
            x = x.reshape(N, S, H, self.half_dims, 2)
            x1, x2 = x[..., 0], x[..., 1] # shape: (N, S, H, self.half_dims)
        else:
            x1 = x[..., 0 : self.half_dims] # shape: (N, S, H, self.half_dims)
            x2 = x[..., self.half_dims : self.dims] # shape: (N, S, H, self.half_dims)

        # Broadcast rotation frequencies to match tensor dimensions
        cos_basis = cos_basis.reshape(-1, S, 1, self.half_dims)
        sin_basis = sin_basis.reshape(-1, S, 1, self.half_dims)

        # Apply 2D rotation
        real = x1 * cos_basis - x2 * sin_basis # shape: (N, S, H, self.half_dims)
        imag = x1 * sin_basis + x2 * cos_basis # shape: (N, S, H, self.half_dims)

        # Reconstruct original shape
        if self.traditional:
            y = mx.stack([real, imag], axis=-1).reshape(N, S, H, D)
        else:
            y = mx.concat([real, imag], axis=-1).reshape(N, S, H, D) # shape: (N, S, H, D)

        return y.astype(x.dtype)
