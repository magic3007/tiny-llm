import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y, offset):
        logits = model(y[None], offset) # (1, L, V)
        logits = logits[:, -1, :] # (1, V)
        logprobs = logits - mx.logsumexp(logits, keepdims=True) # for numerical stability
        if sampler is None:
            y = mx.argmax(logprobs, axis=-1)
        else:
            y = sampler(logprobs)
        return y

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    # generate/decode
    while True:
        token = _step(model, tokens, tokens.size)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        # 流式输出：在生成长文本时，可以实现逐字/逐词的流式输出效果，而不是等待所有 token 生成完毕。
        print(detokenizer.last_segment, end="", flush=True)

def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    pass


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    pass
