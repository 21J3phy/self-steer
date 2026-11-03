"""Edits: small, reversible changes to Qwen's MLP outputs.

Every edit acts on the output of a layer's `mlp.down_proj`, which is written
straight into the residual stream.

- `add`: adds a fixed vector at that layer. This is exactly a bias patch on
  down_proj. Lazy Genius neuron doses and CAA vectors are both this kind.
- `ablate`: removes a unit direction u from the output, out -> out - (out.u)u.
  This equals the rank-1 weight edit W_down <- (I - u u^T) W_down, applied
  next to the frozen 8-bit weights instead of rewriting them.
- `columns`: changes chosen columns (neurons) of W_down, W[:, S] <- W[:, S] + dW.
  It is applied as out += dW @ x[S], using the layer's own neuron activations x.
  Ops: `scale` (dW = (f - 1) W[:, S]; f = 0 silences the neurons) and `project`
  (dW = -alpha u u^T W[:, S]; removes direction u from just those columns, the
  edit Weight-Space Programming applied to Qwen3.5-0.8B).

Neuron doses follow the Lazy Genius convention:
dose x median residual norm at that layer x normalized W_down column.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import mlx.core as mx

KINDS = ('add', 'ablate', 'columns')


@dataclass
class Edit:
    """A resolved edit: one vector per layer, in residual-stream units."""
    name: str
    kind: str
    vectors: dict[int, mx.array]
    # Number of answer positions to steer, counting the position that predicts
    # the first answer token. None means "until the answer ends".
    window: int | None = None
    description: str = ''
    author: str = 'builtin'
    meta: dict = field(default_factory=dict)
    # kind == 'columns': {layer: (neuron indices as int32 array, dW of shape hidden x k)}
    columns: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'Unknown edit kind {self.kind!r}')
        if self.window is not None and (type(self.window) is not int or self.window < 1):
            raise ValueError('window must be a positive integer or None')
        if self.kind == 'columns':
            if not self.columns or self.vectors:
                raise ValueError('A columns edit needs columns and no vectors')
            return
        if not self.vectors:
            raise ValueError('An edit needs at least one layer')
        if self.kind == 'ablate':
            self.vectors = {layer: unit(v) for layer, v in self.vectors.items()}

    @property
    def layers(self):
        return sorted(self.columns if self.kind == 'columns' else self.vectors)

    def touches(self, layer):
        return layer in (self.columns if self.kind == 'columns' else self.vectors)

    def scaled(self, gain):
        """A copy with its change multiplied by gain (ablations ignore gain)."""
        if self.kind == 'ablate' or gain == 1:
            return self
        if self.kind == 'columns':
            cols = {l: (idx, dw * gain) for l, (idx, dw) in self.columns.items()}
            return Edit(self.name, self.kind, {}, self.window, self.description, self.author,
                        dict(self.meta, gain=gain), columns=cols)
        return Edit(self.name, self.kind, {l: v * gain for l, v in self.vectors.items()},
                    self.window, self.description, self.author, dict(self.meta, gain=gain))

    def summary(self):
        if self.kind == 'columns':
            detail = {'neurons_by_layer': {l: int(idx.size) for l, (idx, _) in self.columns.items()},
                      'weights_changed': int(sum(dw.size for _, dw in self.columns.values()))}
        else:
            detail = {'norm_by_layer': {l: round(float(mx.linalg.norm(v.astype(mx.float32)).item()), 3)
                                        for l, v in self.vectors.items()}}
        return {'name': self.name, 'kind': self.kind, 'layers': self.layers,
                'window': self.window, 'description': self.description,
                'author': self.author, **detail, **self.meta}


def unit(v):
    v = v.astype(mx.float32)
    norm = float(mx.linalg.norm(v).item())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Cannot normalize a zero or nonfinite direction')
    return v / norm


def down_column(projection, neuron):
    """Column `neuron` of W_down as float32, dequantizing if needed.

    Ported from lazy-genius-qwen code/mlx_confirm.py.
    """
    if hasattr(projection, 'scales'):
        weight = mx.dequantize(projection.weight, projection.scales, projection.biases,
                               group_size=projection.group_size, bits=projection.bits,
                               mode=projection.mode)
    else:
        weight = projection.weight
    column = weight[:, neuron].astype(mx.float32)
    mx.eval(column)
    return column


def validate_neuron_preset(preset, num_layers, num_neurons):
    """Checks the Lazy Genius preset format and returns {layer: resid_median}."""
    if preset.get('scope', 'response') != 'response' or preset.get('system') is not None:
        raise ValueError('Preset must be response-scoped with no system prompt')
    rows = preset.get('neurons')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Preset needs a nonempty neurons list')
    medians = {int(k): float(v) for k, v in preset.get('resid_median', {}).items()}
    seen = set()
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f'Bad neuron row: {row}')
        layer, neuron, dose = row
        if type(layer) is not int or type(neuron) is not int:
            raise ValueError(f'Layer and neuron must be integers: {row}')
        if not (0 <= layer < num_layers and 0 <= neuron < num_neurons):
            raise ValueError(f'Neuron out of range: {row}')
        if (layer, neuron) in seen or not math.isfinite(float(dose)):
            raise ValueError(f'Duplicate or nonfinite neuron: {row}')
        seen.add((layer, neuron))
        if not math.isfinite(medians.get(layer, 0)) or medians.get(layer, 0) <= 0:
            raise ValueError(f'Missing positive resid_median for layer {layer}')
    return medians


def vectors_from_neurons(rows, medians, projections):
    """Sum of dose x median x unit W_down column per layer (Lazy Genius math)."""
    additions = {}
    for layer, neuron, dose in rows:
        column = down_column(projections[layer], neuron)
        norm = float(mx.linalg.norm(column).item())
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f'Invalid write-column norm: {layer}:{neuron}')
        delta = float(dose) * medians[layer] * column / max(norm, 1e-8)
        additions[layer] = additions.get(layer, mx.zeros_like(column)) + delta
    mx.eval(additions)
    for layer, delta in additions.items():
        if not bool(mx.all(mx.isfinite(delta)).item()):
            raise ValueError(f'Nonfinite addition at layer {layer}')
    return additions


def dequantized(projection):
    """The full W_down of a layer as float32 (hidden x neurons)."""
    if hasattr(projection, 'scales'):
        return mx.dequantize(projection.weight, projection.scales, projection.biases,
                             group_size=projection.group_size, bits=projection.bits,
                             mode=projection.mode).astype(mx.float32)
    return projection.weight.astype(mx.float32)


def columns_from_ops(ops, projections, num_layers, num_neurons):
    """Turns column ops into {layer: (indices, dW)}; several ops on a layer are merged."""
    merged = {}
    for op in ops:
        layer = int(op['layer'])
        neurons = sorted(set(int(n) for n in op['neurons']))
        if not 0 <= layer < num_layers or not neurons or not all(0 <= n < num_neurons for n in neurons):
            raise ValueError(f'Bad column op: {op}')
        w = dequantized(projections[layer])[:, mx.array(neurons)]
        if op['op'] == 'scale':
            dw = (float(op['factor']) - 1.0) * w
        elif op['op'] == 'project':
            u = unit(mx.array(op['direction'], dtype=mx.float32))
            dw = -float(op.get('alpha', 1.0)) * mx.outer(u, u @ w)
        else:
            raise ValueError(f"Unknown column op {op['op']!r}")
        for n, col in zip(neurons, dw.T):
            merged.setdefault(layer, {})
            merged[layer][n] = merged[layer].get(n, 0) + col
    out = {}
    for layer, cols in merged.items():
        idx = sorted(cols)
        out[layer] = (mx.array(idx, dtype=mx.int32), mx.stack([cols[n] for n in idx], axis=1))
    mx.eval([v for pair in out.values() for v in pair])
    return out


def edit_from_preset(preset, projections, *, num_layers, num_neurons, sign=1.0):
    """Builds an Edit from a preset JSON dict.

    Two preset kinds are accepted:
    - "neurons": the Lazy Genius format ([layer, neuron, dose] rows + resid_median).
    - "vectors": {"kind": "add"|"ablate", "vectors": {layer: [floats]}}.
    """
    common = dict(name=preset['name'], window=preset.get('window'),
                  description=preset.get('description', ''), author=preset.get('author', 'builtin'),
                  meta=dict(preset.get('meta', {})))
    if preset.get('kind') == 'columns':
        cols = columns_from_ops(preset['ops'], projections, num_layers, num_neurons)
        common['meta'].update(ops=[{k: v for k, v in op.items() if k != 'direction'} for op in preset['ops']])
        edit = Edit(kind='columns', vectors={}, columns=cols, **common)
        edit.source_ops = preset['ops']
        return edit
    if 'neurons' in preset:
        medians = validate_neuron_preset(preset, num_layers, num_neurons)
        vectors = vectors_from_neurons(preset['neurons'], medians, projections)
        if sign != 1.0:
            vectors = {l: v * sign for l, v in vectors.items()}
        common['meta'].update(neurons=len(preset['neurons']), source=preset.get('source', ''))
        return Edit(kind='add', vectors=vectors, **common)
    vectors = {int(l): mx.array(v, dtype=mx.float32) for l, v in preset['vectors'].items()}
    kind = preset.get('kind', 'add')
    if kind == 'add' and sign != 1.0:
        vectors = {l: v * sign for l, v in vectors.items()}
    return Edit(kind=kind, vectors=vectors, **common)
