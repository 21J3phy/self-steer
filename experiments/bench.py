"""E2: overhead of live edits, CAA mode building, re-prefill and snapshots.

  uv run python experiments/bench.py --out runs/bench.json
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx

from selfsteer.decode import Runner, Sampling
from selfsteer.edits import Edit, unit
from selfsteer.library import Library
from selfsteer.model import Engine
from selfsteer.probe import build_mode
from selfsteer.session import Armed, Session

PROMPT = 'Explain how a refrigerator keeps food cold, step by step.'

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--repeats', type=int, default=3)
args = ap.parse_args()

engine = Engine.load()
lib = Library(engine).load_dir()
lazy = lib.get('lazy')
mx.random.seed(7)
direction = {l: unit(mx.random.normal((engine.hidden_size,))) for l in range(8, 17)}
lib.add(Edit('full', 'add', dict(lazy.vectors), window=None, description='lazy vector, whole answer'))
lib.add(Edit('ablate9', 'ablate', direction, description='random 9-layer rank-1 ablation'))
results = {'model': engine.describe(), 'decode': {}, 'build': {}, 'reprefill': {}, 'snapshot': {}}


def decode_speed(modes):
    rates = []
    for r in range(args.repeats):
        s = Session(engine, lib, use_tools=False, sampling=Sampling(max_tokens=256, temperature=0.0))
        s.armed = [Armed(m, 'answer', 1.0, 'bench') for m in modes]
        turn = s.send(PROMPT, seed=r)
        seg = turn.segments[-1]
        rates.append(seg['decode_tok_per_s'])
    return {'tok_per_s_median': statistics.median(rates), 'runs': rates}


for label, modes in [('none', []), ('lazy_window6', ['lazy']), ('lazy_whole_answer', ['full']),
                     ('ablate_9_layers', ['ablate9']), ('whole_plus_ablate', ['full', 'ablate9'])]:
    results['decode'][label] = decode_speed(modes)
    print(label, results['decode'][label]['tok_per_s_median'], flush=True)

examples = [
    {'prompt': 'What is photosynthesis?', 'positive': 'Plants are like little chefs. They take sunlight, water and air and cook their own food!',
     'negative': 'Photosynthesis is the process by which plants convert light energy, water and carbon dioxide into glucose and oxygen.'},
    {'prompt': 'Why is the sky blue?', 'positive': 'Sunlight has all the colors mixed up. The air bounces the blue color around the most, so we see blue everywhere!',
     'negative': 'Rayleigh scattering by atmospheric molecules scatters shorter wavelengths more strongly, so diffuse skylight is dominated by blue.'},
] * 4
for n in (2, 8):
    t = []
    for r in range(args.repeats):
        started = time.perf_counter()
        build_mode(engine, lib, name=f'bench{n}', description='bench', examples=examples[:n], strength=0.5)
        t.append(time.perf_counter() - started)
    results['build'][f'{n}_examples'] = {'seconds_median': statistics.median(t), 'runs': t}
    print('build', n, statistics.median(t), flush=True)

ids = engine.tokenizer.encode(('The quick brown fox jumps over the lazy dog. ' * 600), add_special_tokens=False)
for length in (512, 1024, 2048, 4096):
    t = []
    for r in range(args.repeats):
        runner = Runner(engine)
        started = time.perf_counter()
        runner.prefill(ids[:length])
        t.append(time.perf_counter() - started)
    snap_t = []
    for r in range(args.repeats):
        started = time.perf_counter()
        snap = runner.snapshot()
        mx.eval([c.state for c in snap[0]])
        snap_t.append(time.perf_counter() - started)
    results['reprefill'][str(length)] = {'seconds_median': statistics.median(t), 'tok_per_s': length / statistics.median(t)}
    results['snapshot'][str(length)] = {'seconds_median': statistics.median(snap_t)}
    print('reprefill', length, statistics.median(t), 'snapshot', statistics.median(snap_t), flush=True)

t = []
for r in range(1000):
    started = time.perf_counter()
    a = engine.stack.activate(lazy, 100, 106, tag='bench')
    engine.stack.deactivate(a)
    t.append(time.perf_counter() - started)
results['switch_seconds_median'] = statistics.median(t)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(results, indent=1))
print(json.dumps({k: v for k, v in results.items() if k != 'model'}, indent=1))
