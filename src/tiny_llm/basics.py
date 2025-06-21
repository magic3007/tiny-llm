import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    # x: (N.., I)
    # w: (O, I)
    # bias: (O,)
    output = x @ w.swapaxes(-2, -1) # (N.., O)
    if bias is not None:
        output += bias # (N.., O)
    return output


def silu(x: mx.array) -> mx.array:
    pass
