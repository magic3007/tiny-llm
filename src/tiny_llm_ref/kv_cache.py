from typing import Optional

from .attention import causal_mask
import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(
        self, key: mx.array, value: mx.array, q_L: int | None = None
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.key_values = [None] * max_active_requests
        self.real_seq_len = [0] * max_active_requests
        self.HD = None

    def update_and_fetch(
        self, key: mx.array, value: mx.array, q_L: int | None = None
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        B, H, S, D = key.shape
        assert key.shape == value.shape
        assert S <= self.max_seq_len
        assert self.HD == (H, D), f"expect {self.HD} but got {H, D}"
        assert B == self.max_active_requests
        # Step 1: append the result to the cache
        for b in range(B):
            if self.key_values[b] is None:
                continue
            cached_keys, cached_values = self.key_values[b]
            keys, values = key[b], value[b]
            keys = mx.concat([cached_keys, keys], axis=1)
            values = mx.concat([cached_values, values], axis=1)
            self.key_values[b] = (keys, values)
            self.real_seq_len[b] += S
        # Step 2: compute seq_len of this batch
        seq_len = max(self.real_seq_len)
        # Step 3: generate masks and a single array of keys and values
        masks = []
        keys = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=key.dtype)
        values = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=value.dtype)
        masks = mx.full(
            (self.max_active_requests, q_L, seq_len), -mx.inf, dtype=key.dtype
        )
        for b in range(B):
            if self.key_values[b] is None:
                # for some reasons we need to do this, otherwise it will cause wrong output?
                # maybe precision issues?
                masks[b, :, :] = causal_mask(q_L, seq_len, dtype=key.dtype)
                continue
            cached_keys, cached_values = self.key_values[b]
            S = self.real_seq_len[b]
            keys[b, :, seq_len - S : seq_len, :] = cached_keys
            values[b, :, seq_len - S : seq_len, :] = cached_values
            masks[b, :, seq_len - S : seq_len] = causal_mask(q_L, S, dtype=key.dtype)
        return keys, values, None, masks.reshape(B, 1, q_L, seq_len)

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        keys, values = prefilled.key_values
        B, H, L, D = keys.shape
        assert B == 1
        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D)
        self.real_seq_len[id] = L
        self.key_values[id] = (keys[0], values[0])

    def remove_request(self, id: int):
        if self.key_values is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.key_values[id] = None
        self.real_seq_len[id] = 0


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self, key: mx.array, value: mx.array, q_L: int | None = None
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if self.key_values is None:
            assert self.offset == 0
            self.key_values = (key, value)
            B, H, S, D = key.shape
            self.offset = S
            return key, value, 0, None
        else:
            B, H, S, D = key.shape
            assert key.shape == value.shape
            prev_keys, prev_values = self.key_values
            assert prev_keys.shape == (B, H, self.offset, D)
            assert prev_values.shape == (B, H, self.offset, D)
            new_keys = mx.concat([prev_keys, key], axis=2)
            new_values = mx.concat([prev_values, value], axis=2)
            self.key_values = (new_keys, new_values)
            start_offset = self.offset
            self.offset += S
            return new_keys, new_values, start_offset, None
