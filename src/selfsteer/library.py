"""The mode library: named edits Qwen (or a controller) can switch on."""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading

import mlx.core as mx

from .edits import Edit, edit_from_preset

PRESET_DIR = Path(__file__).resolve().parents[2] / 'presets'
NORMAL = ('normal', 'none', 'off', 'default')
NAME_RE = re.compile(r'^[a-z][a-z0-9_-]{0,31}$')


class Library:
    def __init__(self, engine):
        self.engine = engine
        self.modes: dict[str, Edit] = {}
        self.resid_median: dict[int, float] = {}
        self.lock = threading.Lock()
        self.version = 0

    def load_dir(self, directory=PRESET_DIR):
        directory = Path(directory)
        for path in sorted(directory.glob('*.json')):
            self.add(self.load_preset(path))
        return self

    def load_preset(self, path):
        path = Path(path)
        preset = json.loads(path.read_text())
        sign = 1.0
        if 'derive' in preset:
            base = json.loads((path.parent / preset['derive']['from']).read_text())
            sign = float(preset['derive'].get('sign', 1.0))
            merged = dict(base)
            merged.update({k: v for k, v in preset.items() if k != 'derive'})
            merged.setdefault('meta', {})
            merged['meta'] = dict(base.get('meta', {}), derived_from=preset['derive']['from'], sign=sign)
            preset = merged
        if 'resid_median' in preset and not self.resid_median:
            self.resid_median = {int(k): float(v) for k, v in preset['resid_median'].items()}
        return edit_from_preset(preset, self.engine.projections, num_layers=self.engine.num_layers,
                                num_neurons=self.engine.num_neurons, sign=sign)

    def add(self, edit: Edit, replace=True):
        if not NAME_RE.match(edit.name) or edit.name in NORMAL:
            raise ValueError(f'Bad mode name {edit.name!r}: use lowercase letters, digits, - or _')
        with self.lock:
            if edit.name in self.modes and not replace:
                raise ValueError(f'Mode {edit.name!r} already exists')
            self.modes[edit.name] = edit
            self.version += 1
        return edit

    def remove(self, name):
        with self.lock:
            self.modes.pop(name)
            self.version += 1

    def get(self, name):
        if name not in self.modes:
            raise KeyError(f"Unknown mode {name!r}. Available: {', '.join(self.names())}")
        return self.modes[name]

    def names(self):
        return sorted(self.modes)

    def guide(self):
        """Mode list shown to Qwen."""
        lines = [f'- {name}: {self.modes[name].description}' for name in self.names()]
        return '\n'.join(lines)

    def save(self, name, directory=PRESET_DIR):
        edit = self.get(name)
        if 'neurons' in edit.meta and edit.meta.get('source'):
            raise ValueError('Neuron presets are already files')
        if edit.kind == 'columns':
            data = {'name': edit.name, 'kind': 'columns', 'window': edit.window, 'description': edit.description,
                    'author': edit.author, 'ops': edit.source_ops}
            path = Path(directory) / f'{name}.json'
            path.write_text(json.dumps(data))
            return path
        data = {'name': edit.name, 'kind': edit.kind, 'window': edit.window,
                'description': edit.description, 'author': edit.author,
                'meta': {k: v for k, v in edit.meta.items() if _jsonable(v)},
                'vectors': {str(l): [round(float(x), 7) for x in v.astype(mx.float32).tolist()]
                            for l, v in edit.vectors.items()}}
        path = Path(directory) / f'{name}.json'
        path.write_text(json.dumps(data))
        return path


def _jsonable(value):
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False
