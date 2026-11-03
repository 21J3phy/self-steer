"""The edit stack: which edits are live, and at which absolute token positions.

Every layer's down_proj is wrapped once by EditableDownProj. The decode loop
calls `stack.set_span(start, length)` before each forward pass, so the wrappers
know the absolute positions being computed. MLX builds its graph when the model
is called from Python, so edits changed between two forward passes apply from
the very next token, with no model reload.

The arithmetic for a single add-edit matches lazy-genius-qwen's
WindowedResponseScope exactly: sum in float32, cast once to the output dtype.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools

import mlx.core as mx
import mlx.nn as nn

from .edits import Edit

_ids = itertools.count(1)


@dataclass
class Active:
    edit: Edit
    start: int  # first absolute position (inclusive)
    stop: int | None  # last absolute position (exclusive); None = open-ended
    tag: str = ''
    id: int = field(default_factory=lambda: next(_ids))
    steered_positions: int = 0

    def covers(self, position):
        return position >= self.start and (self.stop is None or position < self.stop)

    def overlap(self, start, end):
        hi = end if self.stop is None else min(end, self.stop)
        return max(0, hi - max(start, self.start))


class EditStack:
    def __init__(self):
        self.actives: list[Active] = []
        self.span_start = 0
        self.span_len = 0
        self.wrapped_layers = 0
        self.capture = None  # {layer: neuron activations} while reading activations

    # -- span bookkeeping (called by the decode loop) --
    def set_span(self, start, length):
        if start < 0 or length < 1:
            raise ValueError('Bad span')
        self.span_start, self.span_len = start, length

    # -- edit management --
    def activate(self, edit, start, stop=None, tag=''):
        if stop is not None and stop <= start:
            raise ValueError('Empty edit window')
        active = Active(edit, start, stop, tag)
        self.actives.append(active)
        return active

    def deactivate(self, active_or_tag):
        before = len(self.actives)
        if isinstance(active_or_tag, Active):
            self.actives = [a for a in self.actives if a.id != active_or_tag.id]
        else:
            self.actives = [a for a in self.actives if a.tag != active_or_tag]
        return before - len(self.actives)

    def clear(self):
        self.actives = []

    def names_at(self, position):
        return [a.edit.name for a in self.actives if a.covers(position)]

    # -- the hook --
    def apply(self, layer, out, x=None):
        if self.capture is not None and x is not None:
            self.capture[layer] = x
        if not self.actives:
            return out
        if out.ndim != 3 or out.shape[0] != 1 or out.shape[1] != self.span_len:
            raise ValueError('Edits support batch size one with a span set by the decode loop')
        start = self.span_start
        end = start + out.shape[1]
        live = [a for a in self.actives if a.edit.touches(layer) and a.overlap(start, end)]
        if not live:
            return out
        positions = mx.arange(start, end)
        result = out.astype(mx.float32)
        # Ablations act on the layer's own write (a weight edit), then adds (a bias patch).
        for a in sorted(live, key=lambda a: a.edit.kind != 'ablate'):
            stop = end if a.stop is None else a.stop
            mask = (positions >= a.start) & (positions < stop)
            if a.edit.kind == 'columns':
                # W[:, S] += dW  =>  out += x[S] @ dW^T, from this layer's own neuron activations.
                idx, dw = a.edit.columns[layer]
                delta = x[..., idx].astype(mx.float32) @ dw.T
                result = result + delta * mask[None, :, None]
                if layer == min(a.edit.columns):
                    a.steered_positions += a.overlap(start, end)
                continue
            v = a.edit.vectors[layer].astype(mx.float32)
            if a.edit.kind == 'ablate':
                proj = (result @ v)[..., None] * v[None, None, :]
                result = result - proj * mask[None, :, None]
            else:
                result = result + v[None, None, :] * mask[None, :, None]
            if layer == min(a.edit.vectors):
                a.steered_positions += a.overlap(start, end)
        return result.astype(out.dtype)


class EditableDownProj(nn.Module):
    """Wraps mlp.down_proj; the frozen projection is untouched."""
    def __init__(self, projection, layer, stack):
        super().__init__()
        self.projection = projection
        self.layer = layer
        self._stack = stack  # underscore: not a child module or parameter

    def __call__(self, x):
        return self._stack.apply(self.layer, self.projection(x), x)


def install(layers, stack):
    """Wraps every layer's down_proj. Returns the original projections."""
    projections = {}
    for i, layer in enumerate(layers):
        current = layer.mlp.down_proj
        if isinstance(current, EditableDownProj):
            projections[i] = current.projection
            continue
        projections[i] = current
        layer.mlp.down_proj = EditableDownProj(current, i, stack)
    stack.wrapped_layers = len(layers)
    return projections


def uninstall(layers):
    for layer in layers:
        current = layer.mlp.down_proj
        if isinstance(current, EditableDownProj):
            layer.mlp.down_proj = current.projection
