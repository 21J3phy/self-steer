"""A small prefill/decode loop on a persistent cache.

It mirrors mlx_lm 0.31.3's generate_step closely enough to reproduce
lazy-genius-qwen generations token for token:
- prefill all but the last pending token in chunks (default 512);
- logits of the last position -> logits processors (generated tokens only)
  -> logprobs = logits - logsumexp(logits) -> sampler;
- the same HF-order sampler (temperature, top-k, top-p) ported from
  lazy-genius-qwen code/mlx_sampled.py.

Unlike stream_generate it has no one-token lookahead, so an edit changed after
token n applies to the prediction of token n+1. It also keeps the cache and the
penalty history across segments (tool calls) of one assistant turn.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import time

import mlx.core as mx
from mlx_lm import sample_utils
from mlx_lm.generate import generation_stream
from mlx_lm.models import cache as cache_lib


@dataclass
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_tokens: int = 1024
    prefill_step_size: int = 512


def matched_seed(base_seed, question_id):
    """Same derivation as lazy-genius-qwen (stable across processes)."""
    payload = json.dumps(['qwen-neuron-sampling-v1', base_seed, question_id],
                         ensure_ascii=False, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def filtered_logprobs(logprobs, temperature, top_p, top_k):
    scores = logprobs / temperature
    scores = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    if 0 < top_k < scores.shape[-1]:
        scores = sample_utils.apply_top_k(scores, top_k)
        scores = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    if top_p < 1.0:
        scores = sample_utils.apply_top_p(scores, top_p)
    return scores


def make_sampler(s: Sampling):
    if s.temperature == 0:
        return lambda logprobs: mx.argmax(logprobs, axis=-1)

    def sample(logprobs):
        return sample_utils.categorical_sampling(
            filtered_logprobs(logprobs, s.temperature, s.top_p, s.top_k), 1.0)
    return sample


def make_processors(s: Sampling):
    return sample_utils.make_logits_processors(
        presence_penalty=s.presence_penalty, presence_context_size=s.max_tokens + 1,
        repetition_penalty=s.repetition_penalty, repetition_context_size=s.max_tokens + 1)


class Runner:
    """Owns a cache and the count of tokens already inside it."""
    def __init__(self, engine):
        self.engine = engine
        self.reset()

    def reset(self):
        self.cache = cache_lib.make_prompt_cache(self.engine.model)
        self.length = 0

    def snapshot(self):
        return copy.deepcopy(self.cache), self.length

    def restore(self, snap):
        cache, length = snap
        self.cache, self.length = copy.deepcopy(cache), length

    def _call(self, tokens):
        self.engine.stack.set_span(self.length, len(tokens))
        logits = self.engine.model(mx.array(tokens)[None], cache=self.cache)
        self.length += len(tokens)
        return logits

    def prefill(self, tokens, step=512):
        """Feeds tokens without sampling (all of them)."""
        with mx.stream(generation_stream):
            for i in range(0, len(tokens), step):
                self._call(tokens[i:i + step])
                mx.eval([c.state for c in self.cache])
                mx.clear_cache()

    def step(self, tokens):
        """Feeds tokens and returns last-position logits, shape (1, vocab)."""
        with mx.stream(generation_stream):
            return self._call(tokens)[:, -1, :]

    def feed_and_logits(self, pending, step=512):
        """generate_step's prefill: all but the last token in chunks, then the last."""
        if not pending:
            raise ValueError('Need at least one pending token')
        self.prefill(pending[:-1], step)
        return self.step(pending[-1:])


def sample_loop(runner, pending, *, sampler, processors, history, max_tokens, stop_ids,
                prefill_step=512, on_token=None):
    """Generates until a stop token or max_tokens.

    `history` holds this turn's previously generated tokens (for penalties) and
    is extended in place. The final sampled token (stop or last) is returned
    but NOT fed into the cache; the caller puts it at the head of the next
    pending list. Returns (tokens, reason, timings).
    """
    started = time.perf_counter()
    logits = runner.feed_and_logits(pending, prefill_step)
    prefill_done = None
    out = []
    reason = 'length'
    while True:
        if processors:
            hist = mx.array(history, dtype=mx.int32) if history else mx.array([], dtype=mx.int32)
            with mx.stream(generation_stream):
                for p in processors:
                    logits = p(hist, logits)
        with mx.stream(generation_stream):
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = sampler(logprobs)
        token = int(y.item())
        if prefill_done is None:
            prefill_done = time.perf_counter()
        # The position that predicted this token is runner.length - 1.
        out.append(token)
        history.append(token)
        if on_token:
            on_token(token, runner.length - 1)
        if token in stop_ids:
            reason = 'stop'
            break
        if len(out) >= max_tokens:
            break
        logits = runner.step([token])
    end = time.perf_counter()
    timings = {'first_token_s': (prefill_done or end) - started, 'total_s': end - started,
               'decode_tok_per_s': (len(out) - 1) / max(end - (prefill_done or end), 1e-9)}
    return out, reason, timings
