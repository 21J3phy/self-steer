"""E4 generation: `self` (Qwen decides whether to go lazy) and `prompt` (be-concise prompt).

  uv run python experiments/smart_lazy.py --questions PATH/fair_final96.json \
      --seeds 17 29 --conditions self prompt --out runs/smart/answers.jsonl

Append-only JSONL; rerunning skips finished cells.
"""
import argparse
import json
from pathlib import Path

from selfsteer.decode import matched_seed
from selfsteer.library import PRESET_DIR, Library
from selfsteer.model import Engine
from selfsteer.session import Session


class PlaceboSession(Session):
    """Same tools, prompt and tool responses as `self`, but set_mode arms nothing."""
    def _arm_segment(self, start):
        return []

CONCISE = ('Be concise. Give the answer directly, with no preamble, no restating the question, '
           'and no extra explanation unless asked.')

ap = argparse.ArgumentParser()
ap.add_argument('--questions', type=Path, required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[17, 29])
ap.add_argument('--conditions', nargs='+', default=['self', 'prompt'])
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()

questions = json.loads(args.questions.read_text())
done = set()
if args.out.exists():
    for line in args.out.read_text().splitlines():
        r = json.loads(line)
        done.add((r['condition'], r['question_id']))
args.out.parent.mkdir(parents=True, exist_ok=True)

engine = Engine.load()
lazy_only = Library(engine)
lazy_only.add(lazy_only.load_preset(PRESET_DIR / 'lazy.json'))

with args.out.open('a') as f:
    for seed in args.seeds:
        for q in questions:
            for cond in args.conditions:
                qid = f"{q['id']}::seed{seed}"
                if (cond, qid) in done:
                    continue
                if cond == 'self':
                    s = Session(engine, lazy_only, use_tools=True)
                elif cond == 'placebo':
                    s = PlaceboSession(engine, lazy_only, use_tools=True)
                elif cond == 'prompt':
                    s = Session(engine, lazy_only, use_tools=False, system=CONCISE)
                else:
                    raise SystemExit(f'unknown condition {cond}')
                sampling_seed = matched_seed(seed, q['id'])
                turn = s.send(q['question'], seed=sampling_seed)
                row = {'condition': cond, 'question_id': qid, 'base_question_id': q['id'], 'base_seed': seed,
                       'category': q.get('category'), 'question': q['question'], 'output': turn.reply,
                       'word_count': len(turn.reply.split()), 'generated_tokens': turn.generated_tokens,
                       'at_token_cap': turn.finish == 'length', 'finish_reason': turn.finish,
                       'modes_used': turn.modes_used, 'tool_calls': turn.tool_calls,
                       'segments': [{k: v for k, v in seg.items() if k != 'text'} for seg in turn.segments],
                       'seconds': turn.seconds, 'sampling_seed': sampling_seed}
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(cond, qid, row['word_count'], row['modes_used'], flush=True)
