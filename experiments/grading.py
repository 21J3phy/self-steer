"""E4 blind grading: prepare chunks, then merge judgments into scored rows.

  prepare: uv run python experiments/grading.py prepare --new runs/smart/answers.jsonl \
              --questions LG/data/fair_final96.json --old LG/results/test1/review/scored-answers.json \
              --out runs/smart/review
  merge:   uv run python experiments/grading.py merge --review runs/smart/review

Follows lazy-genius-qwen code/prepare_sampled_blind.py: shuffled, condition-blind,
per-item rubrics. 48 previously graded test-1 answers are mixed in as calibration.
"""
import argparse
import json
from pathlib import Path
import random

CAL_N = 48


def prepare(a):
    questions = {q['id']: q for q in json.loads(a.questions.read_text())}
    new = [json.loads(l) for l in a.new.read_text().splitlines() if l.strip()]
    old = json.loads(a.old.read_text())
    pool = [r for r in old if str(r['base_seed']) in ('17', '29')]
    pool.sort(key=lambda r: (r['condition'], r['question_id']))
    calibration = random.Random(0).sample(pool, CAL_N)
    rows = [dict(r, source='new') for r in new] + [
        {'condition': f"old-{r['condition']}", 'question_id': r['question_id'], 'base_question_id': r['base_question_id'],
         'base_seed': int(r['base_seed']), 'question': r['question'], 'output': r['output'],
         'at_token_cap': r['at_token_cap'] in (True, 'True'), 'source': 'calibration',
         'old_core': r['core_correct'] in (True, 'True'), 'old_whole': r['correct'] in (True, 'True'),
         'old_completion_failure': r['completion_failure'] in (True, 'True')} for r in calibration]
    random.Random(a.seed).shuffle(rows)
    blind, key = [], []
    for i, row in enumerate(rows, 1):
        q = questions[row['base_question_id']]
        if q['question'] != row['question']:
            raise ValueError('question mismatch')
        label = f'S{i:04d}'
        blind.append({'id': label, 'question': q['question'], 'reference': q.get('reference', q.get('answer')),
                      'rubric': q.get('rubric'), 'core_rubric': q.get('core_rubric'),
                      'whole_answer_rubric': q.get('whole_answer_rubric'), 'output': row['output'],
                      'at_token_cap': bool(row['at_token_cap'])})
        key.append({'id': label, **{k: v for k, v in row.items() if k not in ('output', 'question', 'tool_calls', 'segments')}})
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'blind-key.json').write_text(json.dumps(key, indent=1, ensure_ascii=False))
    for i in range(a.chunks):
        (a.out / f'blind-{i + 1}.json').write_text(json.dumps(blind[i::a.chunks], indent=1, ensure_ascii=False))
    print(json.dumps({'answers': len(rows), 'new': len(new), 'calibration': len(calibration), 'chunks': a.chunks}))


def merge(a):
    key = {k['id']: k for k in json.loads((a.review / 'blind-key.json').read_text())}
    judged = {}
    for path in sorted(a.review.glob('judgments-*.json')):
        for j in json.loads(path.read_text()):
            if j['id'] in judged:
                raise ValueError(f"duplicate judgment {j['id']}")
            judged[j['id']] = j
    missing = sorted(set(key) - set(judged))
    if missing:
        raise SystemExit(f'missing judgments: {missing[:10]} ({len(missing)})')
    adjudications = {}
    adj = a.review / 'adjudications.json'
    if adj.exists():
        adjudications = {x['id']: x for x in json.loads(adj.read_text())}
    scored = []
    for sid, k in key.items():
        j = dict(judged[sid])
        if sid in adjudications:
            j.update({f: adjudications[sid][f] for f in ('core_correct', 'correct', 'completion_failure') if f in adjudications[sid]})
            j['adjudicated'] = adjudications[sid].get('reason', True)
        scored.append({**k, 'core_correct': bool(j['core_correct']), 'correct': bool(j['correct']),
                       'completion_failure': bool(j['completion_failure']), 'reason': j.get('reason', ''),
                       'adjudicated': j.get('adjudicated')})
    (a.review / 'scored.json').write_text(json.dumps(scored, indent=1, ensure_ascii=False))
    cal = [s for s in scored if s['source'] == 'calibration']
    agree = {f: sum(s[f] == s[f'old_{n}'] for s in cal) for f, n in (('core_correct', 'core'), ('correct', 'whole'))}
    print(json.dumps({'scored': len(scored), 'calibration_items': len(cal), 'calibration_agreement': agree}))


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest='cmd', required=True)
p = sub.add_parser('prepare')
p.add_argument('--new', type=Path, required=True)
p.add_argument('--questions', type=Path, required=True)
p.add_argument('--old', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--chunks', type=int, default=9)
p.add_argument('--seed', type=int, default=90217)
m = sub.add_parser('merge')
m.add_argument('--review', type=Path, required=True)
a = ap.parse_args()
prepare(a) if a.cmd == 'prepare' else merge(a)
