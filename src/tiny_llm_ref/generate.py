import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .kv_cache import *
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
        logits = model(y[None], offset)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(
            logits, keepdims=True
        )  # optional -- for numerical stability
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
        print(detokenizer.last_segment, end="", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    offset = 0
    prefill_max = 64
    total_tokens = tokens.size
    while tokens.size > prefill_max:
        token, _ = _step(model, tokens[:prefill_max], offset, kv_cache)
        for i in kv_cache:
            mx.eval(i.key_values[0])
            mx.eval(i.key_values[1])
        offset += prefill_max
        tokens = tokens[prefill_max:]
        print(f"Prefill progress: {offset}/{total_tokens}", flush=True)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    # generate/decode
    while True:
        token, _ = _step(model, tokens, offset, kv_cache)
        mx.eval(token)
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)
        if token.item() == tokenizer.eos_token_id:
            break
        offset += tokens.size
        tokens = token


def _step(model, y, offsets, kv_cache):
    logits = model(y, offsets, kv_cache)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    sampler = lambda x: mx.argmax(x, axis=-1)
    y = sampler(logprobs)
    return y


class _PrefillRequest:
    def __init__(
        self, model: any, tokenizer: TokenizerWrapper, prompt: str, max_step: int = 128
    ):
        self.prompt = prompt
        self.kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        self.model = model
        self.prefill_tokens = mx.array(
            tokenizer.encode(prompt, add_special_tokens=False)
        )
        self.offset = 0
        self.max_step = max_step

    def prefill(self):
        # returns None if prefill is not done
        tokens_to_prefill = min(self.max_step, self.prefill_tokens.size - self.offset)
        token = _step(
            self.model,
            self.prefill_tokens[self.offset : self.offset + tokens_to_prefill][None],
            [self.offset],
            self.kv_cache,
        )
        self.offset += tokens_to_prefill
        for i in self.kv_cache:
            mx.eval(i.key_values[0])
            mx.eval(i.key_values[1])
        if self.offset == self.prefill_tokens.size:
            mx.eval(token)
            return token, self.kv_cache, self.offset
        else:
            return None


def _print_progress(
    detokenizers: list[TokenizerWrapper],
    prompt_idx: list[int],
    is_idle: list[bool],
    pending_prefill_requests: _PrefillRequest | None,
):
    for i in range(len(detokenizers)):
        if is_idle[i]:
            print(f"Decode {i}: idle", flush=True)
        else:
            print(f"Decode {i}[{prompt_idx[i]}]: {detokenizers[i].text}", flush=True)
    if pending_prefill_requests is not None:
        print(
            f"Prefill {pending_prefill_requests.offset}/{pending_prefill_requests.prefill_tokens.size}",
            flush=True,
        )
    else:
        print("Prefill: idle", flush=True)


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    is_idle = [True] * batch_size
    prompt_idx = [0] * batch_size
    next_tokens = mx.array([0] * batch_size)
    offsets = mx.array([0] * batch_size)
    detokenizers = [None] * batch_size
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    pending_prefill_requests = None

    print(f"Processing {len(prompts)} prompts")
    prompts = enumerate(prompts)
    more_prompts = True
    while True:
        if not more_prompts and all(is_idle):
            break
        # prefill until no idle slots
        while any(is_idle) and more_prompts and pending_prefill_requests is None:
            try:
                idx, prompt = next(prompts)
            except StopIteration:
                more_prompts = False
                break
            pending_prefill_requests = _PrefillRequest(
                model, tokenizer, prompt, prefill_step
            )
            break

        if pending_prefill_requests is not None:
            res = pending_prefill_requests.prefill()
            if res is not None:
                pending_prefill_requests = None
                token, prefill_kv_cache, offset = res

                if token.item() == tokenizer.eos_token_id:
                    # if the first token is eos, we skip this prompt
                    continue

                for i in range(batch_size):
                    if is_idle[i]:
                        detokenizers[i] = tokenizer.detokenizer.__class__(
                            tokenizer._tokenizer
                        )
                        detokenizers[i].add_token(token.item())
                        prompt_idx[i] = idx
                        is_idle[i] = False
                        for prefill_cache, batch_cache in zip(
                            prefill_kv_cache, kv_cache
                        ):
                            batch_cache.add_request(prefill_cache, i)
                        next_tokens[i] = token
                        offsets[i] = offset
                        break

        if not all(is_idle):
            next_tokens = mx.array(next_tokens)
            # decode
            next_tokens = _step(model, next_tokens.reshape(-1, 1), offsets, kv_cache)
            offsets += 1
            for i in range(batch_size):
                if not is_idle[i]:
                    detokenizers[i].add_token(next_tokens[i].item())
                    remove_due_to_eos = next_tokens[i].item() == tokenizer.eos_token_id
                    remove_due_to_max_seq_len = offsets[i] >= max_seq_len
                    if remove_due_to_eos or remove_due_to_max_seq_len:
                        reason = "EOS" if remove_due_to_eos else "Max Seq Len"
                        result.append((prompt_idx[i], detokenizers[i].text))
                        print(f"Removing request {i} due to {reason}", flush=True)
                        batch_cache.remove_request(i)
                        is_idle[i] = True
                        continue
        _print_progress(detokenizers, prompt_idx, is_idle, pending_prefill_requests)
    return result
