"""Does self-steer reproduce published Lazy Genius generations token for token?

Replays (question, seed, condition) cells from lazy-genius-qwen's test-2
generations with the new engine: baseline = no edits, selected = the `lazy`
mode armed for the answer (first 6 positions). Compares generated token ids.

  uv run python scripts/parity_lazy.py --published PATH/answers.json --n 8 --out runs/parity.json
"""
import argparse
import json
from pathlib import Path
import time

from selfsteer.library import Library
from selfsteer.model import Engine
from selfsteer.session import Session

ap = argparse.ArgumentParser()
ap.add_argument('--published', type=Path, required=True)
ap.add_argument('--ids', nargs='*', default=None, help='base question ids (default: first N)')
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--seed', type=int, default=17)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()

rows = json.loads(args.published.read_text())
cells = {}
for r in rows:
    if r['base_seed'] == args.seed:
        cells.setdefault(r['base_question_id'], {})[r['condition']] = r
ids = args.ids or sorted(cells)[:args.n]

engine = Engine.load()
lib = Library(engine).load_dir()
results = []
for qid in ids:
    for cond in ('baseline', 'selected'):
        ref = cells[qid][cond]
        s = Session(engine, lib, use_tools=False)
        if cond == 'selected':
            s.set_mode('lazy', 'answer', by='controller')
        t = time.perf_counter()
        turn = s.send(ref['question'], seed=ref['sampling_seed'])
        got = [tok for seg in [s.ledger[len(ref['prompt_token_ids']):]] for tok in seg] + [s.pending[0]]
        prompt_ok = s.ledger[:len(ref['prompt_token_ids'])] == ref['prompt_token_ids']
        want = ref['generated_token_ids']
        same = got == want
        prefix = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        results.append({'question_id': qid, 'condition': cond, 'prompt_identical': prompt_ok,
                        'tokens_identical': same, 'matching_prefix': prefix, 'ours': len(got),
                        'published': len(want), 'seconds': round(time.perf_counter() - t, 2),
                        'steered_positions': sum(seg['steered_positions'] for seg in turn.segments),
                        'text_identical': turn.reply == ref['output']})
        print(json.dumps(results[-1]), flush=True)

summary = {'cells': len(results), 'tokens_identical': sum(r['tokens_identical'] for r in results),
           'text_identical': sum(r['text_identical'] for r in results), 'results': results}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(summary, indent=1))
print('SUMMARY', summary['tokens_identical'], '/', summary['cells'], 'token-identical')
