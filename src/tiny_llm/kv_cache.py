from typing import Optional

import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(
        self, key: mx.array, value: mx.array, q_L: int | None = None
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        pass

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.kv_cache = None
        self.offset = 0

    def update_and_fetch(
        self, key: mx.array, value: mx.array, q_L: int | None = None
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if self.kv_cache is None:
            B, H, S, D = key.shape # B: batch_size, H: num_heads, S: seq_len, D: head_dim
            self.kv_cache = (key, value)
            self.offset = S
            start_offset = 0
            return key, value, start_offset, None
        else:
            B, H, append_S, D = key.shape # append_S: appended seq_len
            prev_key, prev_value = self.kv_cache
            assert prev_key.shape == (B, H, self.offset, D)
            assert prev_value.shape == (B, H, self.offset, D)
            new_key = mx.concat([prev_key, key], axis=2)
            new_value = mx.concat([prev_value, value], axis=2)
            self.kv_cache = (new_key, new_value)
            start_offset = self.offset
            self.offset += append_S
            return new_key, new_value, start_offset, None

class TinyKvRotatingCache(TinyKvCache):
    def __init__(self, max_seq_len: int):
        pass

    def update_and_fetch(
        self, key: mx.array, value: mx.array, offset: int
    ) -> tuple[mx.array, mx.array]:
        pass
