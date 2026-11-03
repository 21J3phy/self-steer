"""Reading Qwen's neurons and weights directly (the Weight-Space Programming way).

These are the controller's instruments, not behavior tests:
- read_activations: a teacher-forced pass over texts that records every MLP
  neuron's activation (the input to down_proj) and the residual stream,
  averaged over the answer tokens. No edits are live.
- neuron_info: what a neuron writes: its W_down column's norm and the tokens
  that column promotes or suppresses through the final norm and unembedding.
- top_writers: which neurons' columns write a given direction most.
- export_down: a layer's dequantized W_down (or some columns) for offline analysis.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from .edits import dequantized, unit
from .model import text_layers
from .probe import chat_with_answer


def _inner(model):
    lm = getattr(model, 'language_model', model)
    return getattr(lm, 'model', lm)


def lens(engine, vector, k=12):
    """Tokens a residual-space vector promotes and suppresses (final norm + tied unembedding)."""
    inner = _inner(engine.model)
    v = mx.array(vector, dtype=mx.float32)[None, None, :]
    logits = inner.embed_tokens.as_linear(inner.norm(v.astype(mx.bfloat16)))[0, 0].astype(mx.float32)
    order = mx.argsort(logits)
    mx.eval(order, logits)
    top = order[-k:][::-1].tolist()
    bottom = order[:k].tolist()
    dec = lambda ids: [engine.tokenizer.decode([i]) for i in ids]
    return {'promotes': dec(top), 'suppresses': dec(bottom),
            'promote_logits': [round(float(logits[i].item()), 2) for i in top]}


def neuron_info(engine, layer, neurons, k=10):
    w = dequantized(engine.projections[layer])
    out = []
    for n in neurons:
        col = w[:, int(n)]
        out.append({'layer': layer, 'neuron': int(n), 'column_norm': round(float(mx.linalg.norm(col).item()), 4),
                    **lens(engine, col, k)})
    return out


def token_direction(engine, tokens):
    """Mean unembedding row of the given token strings (each must encode to one token)."""
    inner = _inner(engine.model)
    ids = []
    for t in tokens:
        enc = engine.tokenizer.encode(t, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    if not ids:
        raise ValueError('None of the tokens encode to a single token')
    rows = inner.embed_tokens(mx.array(ids)).astype(mx.float32)
    return unit(mx.mean(rows, axis=0)), ids


def top_writers(engine, layer, direction, k=64):
    """Neurons whose W_down columns have the largest signed projection on `direction`."""
    u = unit(mx.array(direction, dtype=mx.float32))
    w = dequantized(engine.projections[layer])
    scores = u @ w
    order = mx.argsort(scores)
    mx.eval(order)
    top = order[-k:][::-1].tolist()
    return [{'neuron': int(n), 'writes': round(float(scores[n].item()), 4)} for n in top]


def read_activations(engine, items, system=None):
    """Teacher-forced read, no edits live. Each item is {"prompt", "answer"}.

    Returns float16 arrays: neurons (items, layers, mlp) and resid (items, layers, hidden),
    each the mean over the answer tokens, plus the mean log-prob of the answer tokens.
    """
    layers = text_layers(engine.model)
    n_layers = len(layers)
    neurons = np.zeros((len(items), n_layers, engine.num_neurons), dtype=np.float16)
    resid = np.zeros((len(items), n_layers, engine.hidden_size), dtype=np.float16)
    logprob = np.zeros(len(items), dtype=np.float32)
    lengths = np.zeros(len(items), dtype=np.int32)
    saved = engine.stack.actives
    engine.stack.actives = []
    originals = list(layers)
    try:
        for i, item in enumerate(items):
            tokens, start = chat_with_answer(engine.tokenizer, item['prompt'], item['answer'], system)
            store = {}
            for j, layer in enumerate(originals):
                layers[j] = _ResidRecorder(layer, j, store, start)
            engine.stack.capture = {}
            engine.stack.set_span(0, len(tokens))
            logits = engine.model(mx.array(tokens)[None])[0].astype(mx.float32)
            capture = engine.stack.capture
            engine.stack.capture = None
            for j, layer in enumerate(originals):
                layers[j] = layer
            lp = logits[start - 1:-1] - mx.logsumexp(logits[start - 1:-1], axis=-1, keepdims=True)
            tgt = mx.array(tokens[start:])
            ans_lp = mx.take_along_axis(lp, tgt[:, None], axis=-1).mean()
            act = mx.stack([capture[j][0, start:].astype(mx.float32).mean(axis=0) for j in range(n_layers)])
            res = mx.stack([store[j] for j in range(n_layers)])
            mx.eval(act, res, ans_lp)
            neurons[i] = np.array(act, dtype=np.float16)
            resid[i] = np.array(res, dtype=np.float16)
            logprob[i] = float(ans_lp.item())
            lengths[i] = len(tokens) - start
    finally:
        engine.stack.capture = None
        for j, layer in enumerate(originals):
            layers[j] = layer
        engine.stack.actives = saved
    return {'neurons': neurons, 'resid': resid, 'answer_logprob': logprob, 'answer_tokens': lengths}


class _ResidRecorder:
    """Records the mean residual after a decoder layer over positions [start:]."""
    def __init__(self, layer, index, store, start):
        self.inner, self.is_linear = layer, layer.is_linear
        self.index, self.store, self.start = index, store, start

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask=mask, cache=cache)
        self.store[self.index] = out[0, self.start:].astype(mx.float32).mean(axis=0)
        return out


def export_down(engine, layer, neurons=None):
    w = dequantized(engine.projections[layer])
    if neurons is not None:
        w = w[:, mx.array([int(n) for n in neurons])]
    mx.eval(w)
    return np.array(w, dtype=np.float32)
