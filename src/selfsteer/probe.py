"""Build new modes in seconds from contrastive examples (CAA).

A controller writes a few (prompt, positive answer, negative answer) triples.
For each one we run the chat-formatted prompt + answer through the frozen model
and record the mean residual stream over the answer tokens after every layer.
The mode's direction at layer L is mean(positive) - mean(negative), averaged
over examples. As an `add` edit it is scaled into Lazy Genius dose units:
strength x median residual norm at L x unit direction. As an `ablate` edit the
unit direction is projected out of that layer's MLP write (a rank-1 weight edit).
"""
from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn

from .edits import Edit, unit
from .model import text_layers


class _Recorder(nn.Module):
    def __init__(self, layer, index, store, start):
        super().__init__()
        self.inner = layer
        self.is_linear = layer.is_linear
        self._index = index
        self._store = store
        self._start = start

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask=mask, cache=cache)
        self._store[self._index] = out[0, self._start:].astype(mx.float32).mean(axis=0)
        return out


def chat_with_answer(tokenizer, prompt, answer, system=None):
    """Tokens for [system] + user prompt + assistant answer, and where the answer starts."""
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False)
    prefix = list(prefix.tolist() if hasattr(prefix, 'tolist') else prefix)
    answer_tokens = list(tokenizer.encode(answer, add_special_tokens=False))
    if not answer_tokens:
        raise ValueError('Empty example answer')
    return prefix + answer_tokens, len(prefix)


def answer_means(engine, tokens, start):
    """Mean residual after each layer over positions [start:], with no edits live."""
    layers = text_layers(engine.model)
    store = {}
    originals = list(layers)
    saved = engine.stack.actives
    engine.stack.actives = []
    try:
        for i, layer in enumerate(originals):
            layers[i] = _Recorder(layer, i, store, start)
        engine.stack.set_span(0, len(tokens))
        engine.model(mx.array(tokens)[None])
        mx.eval(store)
    finally:
        for i, layer in enumerate(originals):
            layers[i] = layer
        engine.stack.actives = saved
    return store


def contrast(engine, examples, system=None):
    """Average (positive - negative) answer-mean difference per layer."""
    if len(examples) < 2:
        raise ValueError('Give at least 2 contrastive examples')
    diffs = {}
    for ex in examples:
        pos, ps = chat_with_answer(engine.tokenizer, ex['prompt'], ex['positive'], system)
        neg, ns = chat_with_answer(engine.tokenizer, ex['prompt'], ex['negative'], system)
        mp, mn = answer_means(engine, pos, ps), answer_means(engine, neg, ns)
        for layer in mp:
            d = mp[layer] - mn[layer]
            diffs[layer] = diffs.get(layer, 0) + d / len(examples)
    mx.eval(diffs)
    return diffs


def build_mode(engine, library, *, name, description, examples, layers=None, strength=0.5,
               kind='add', window=None, author='claude', system=None):
    """Creates a mode from examples and adds it to the library. Returns (edit, stats)."""
    started = time.perf_counter()
    diffs = contrast(engine, examples, system)
    if layers is None:
        n = engine.num_layers
        layers = list(range(n // 4, n // 2 + 1))  # a mid-depth band, e.g. 8-16 of 32
    layers = sorted(set(int(l) for l in layers))
    bad = [l for l in layers if not 0 <= l < engine.num_layers]
    if bad:
        raise ValueError(f'Layers out of range: {bad}')
    if kind == 'add':
        if not library.resid_median:
            raise ValueError('Library has no resid_median calibration (load the lazy preset first)')
        vectors = {l: float(strength) * library.resid_median[l] * unit(diffs[l]) for l in layers}
    elif kind == 'ablate':
        vectors = {l: unit(diffs[l]) for l in layers}
    else:
        raise ValueError(f'Unknown kind {kind!r}')
    # How separable were the examples? Norm of the raw difference vs. the residual scale.
    separation = {l: round(float(mx.linalg.norm(diffs[l]).item()) / library.resid_median.get(l, 1.0), 4)
                  for l in layers}
    edit = Edit(name=name, kind=kind, vectors=vectors, window=window, description=description,
                author=author, meta={'strength': strength, 'examples': len(examples),
                                     'method': 'CAA mean difference over answer tokens'})
    library.add(edit)
    overlap = {}
    for other in library.names():
        if other == name:
            continue
        o = library.get(other)
        shared = [l for l in layers if l in o.vectors]
        if shared:
            cos = sum(float(mx.sum(unit(edit.vectors[l]) * unit(o.vectors[l])).item()) for l in shared) / len(shared)
            overlap[other] = round(cos, 3)
    stats = {'seconds': round(time.perf_counter() - started, 2), 'layers': layers,
             'separation_by_layer': separation, 'mean_cosine_with_existing_modes': overlap,
             **edit.summary()}
    return edit, stats
