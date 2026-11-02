"""Loads Qwen3.5 with every MLP down_proj wrapped for live edits."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .stack import EditStack, install

DEFAULT_MODEL = Path('~/models/Qwen3.5-4B-8bit').expanduser()
EXPECTED_REVISION = '1704583a86670ba531b7f2c92530a2f725cfd45b'


def text_layers(model):
    lm = getattr(model, 'language_model', model)
    inner = getattr(lm, 'model', lm)
    return inner.layers


@dataclass
class Engine:
    model: object
    tokenizer: object
    stack: EditStack
    layers: list
    projections: dict
    num_layers: int
    num_neurons: int
    hidden_size: int
    path: str = ''

    @classmethod
    def wrap(cls, model, tokenizer, path=''):
        stack = EditStack()
        layers = text_layers(model)
        projections = install(layers, stack)
        first = projections[0]
        if hasattr(first, 'scales'):
            hidden, packed = first.weight.shape
            neurons = packed * 32 // first.bits
        else:
            hidden, neurons = first.weight.shape
        return cls(model, tokenizer, stack, layers, projections, len(layers), neurons, hidden, str(path))

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        from mlx_lm import load
        path = Path(path).expanduser()
        model, tokenizer = load(str(path))
        model.eval()
        return cls.wrap(model, tokenizer, path)

    def eos_ids(self):
        return set(int(t) for t in self.tokenizer.eos_token_ids)

    def describe(self):
        info = {'path': self.path, 'layers': self.num_layers, 'mlp_neurons_per_layer': self.num_neurons,
                'hidden_size': self.hidden_size, 'wrapped_layers': self.stack.wrapped_layers}
        config = Path(self.path) / 'config.json' if self.path else None
        if config and config.exists():
            info['quantization'] = json.loads(config.read_text()).get('quantization')
        return info
