"""E7: turn the Kaggle self-steering runs into blind grading chunks (same brief and format as E4).

  python3 experiments/e7/prepare_e7.py   ->  runs/e7/review/{blind-*.json, blind-key.json, GRADING_SAMPLED.md}
"""
import json
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs' / 'remote'
OUT = ROOT / 'runs' / 'e7' / 'review'
questions = {q['id']: q for q in json.loads((ROOT / 'remote' / 'fair_final96.json').read_text())}
rows = []
for cond in ('placebo', 'self', 'selffixed'):
    for r in json.loads((RUNS / f'e7_eval_{cond}_out.json').read_text())['result']['rows']:
        q = questions[r['id']]
        called = [c['call'].get('arguments', {}).get('name') for c in r['tool_calls'] if c.get('call') and c['call'].get('name') == 'set_mode']
        rows.append({'condition': cond, 'question_id': f"{r['id']}::seed{r['seed']}", 'base_question_id': r['id'],
                     'base_seed': r['seed'], 'category': q['category'], 'question': q['question'], 'output': r['reply'],
                     'at_token_cap': r['finish'] == 'length', 'words': r['words'], 'called': called})
random.Random(70217).shuffle(rows)
OUT.mkdir(parents=True, exist_ok=False)
blind, key = [], []
for i, row in enumerate(rows, 1):
    q = questions[row['base_question_id']]
    label = f'S{i:04d}'
    blind.append({'id': label, 'question': q['question'], 'reference': q.get('reference', q.get('answer')), 'rubric': q.get('rubric'),
                  'core_rubric': q.get('core_rubric'), 'whole_answer_rubric': q.get('whole_answer_rubric'),
                  'output': row['output'], 'at_token_cap': row['at_token_cap']})
    key.append({'id': label, 'source': 'new', **{k: v for k, v in row.items() if k not in ('output', 'question')}})
(OUT / 'blind-key.json').write_text(json.dumps(key, indent=1, ensure_ascii=False))
CHUNKS = 4
for i in range(CHUNKS):
    (OUT / f'blind-{i + 1}.json').write_text(json.dumps(blind[i::CHUNKS], indent=1, ensure_ascii=False))
shutil.copy(ROOT / 'runs' / 'smart' / 'review_placebo' / 'GRADING_SAMPLED.md', OUT / 'GRADING_SAMPLED.md')
print(json.dumps({'answers': len(rows), 'chunks': CHUNKS, 'by_condition': {c: sum(r['condition'] == c for r in rows) for c in ('placebo', 'self', 'selffixed')}}))
