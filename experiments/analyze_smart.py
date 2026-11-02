"""E4 analysis: every condition against baseline on the test-1 set, seeds 17 and 29.

  uv run python experiments/analyze_smart.py --old LG/results/test1/review/scored-answers.json \
      --new runs/smart/review/scored.json --routed runs/smart/routed.json \
      --questions LG/data/fair_final96.json --out results/smart_lazy.json
"""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--old', type=Path, required=True)
ap.add_argument('--new', type=Path, required=True)
ap.add_argument('--routed', type=Path, required=True)
ap.add_argument('--questions', type=Path, required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[17, 29])
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()

truth = lambda v: v in (True, 'True', 'true', 1)
questions = {q['id']: q for q in json.loads(a.questions.read_text())}
qids = sorted(questions)
cells = {}  # (condition, qid, seed) -> row


def put(cond, row, extra=None):
    seed = int(row['base_seed'])
    if seed not in a.seeds:
        return
    cells[(cond, row['base_question_id'], seed)] = {
        'words': max(1, len(row['output'].split())), 'core': truth(row['core_correct']), 'whole': truth(row['correct']),
        'fail': truth(row['completion_failure']), 'output': row['output'], **(extra or {})}


for r in json.loads(a.old.read_text()):
    put({'baseline': 'baseline', 'selected': 'lazy'}[r['condition']], r)
for r in json.loads(a.new.read_text()):
    if r['source'] == 'new':
        put(r['condition'], r, {'lazy_chosen': 'lazy' in (r.get('modes_used') or [])})
routed = json.loads(a.routed.read_text())['decisions']
for q in qids:
    for s in a.seeds:
        src = 'lazy' if routed[q] == 'lazy' else 'baseline'
        cells[('routed', q, s)] = dict(cells[(src, q, s)], lazy_chosen=routed[q] == 'lazy')
        b, l = cells[('baseline', q, s)], cells[('lazy', q, s)]
        pick = l if (l['core'], l['whole']) >= (b['core'], b['whole']) else b
        cells[('oracle', q, s)] = dict(pick, lazy_chosen=pick is l)
conditions = ['baseline', 'lazy', 'self', 'prompt', 'routed', 'oracle']
for c in conditions:
    missing = [(q, s) for q in qids for s in a.seeds if (c, q, s) not in cells]
    if missing:
        raise SystemExit(f'{c} missing {len(missing)} cells, e.g. {missing[:3]}')


def boot(diffs, n=10000):
    diffs = np.array(diffs)
    rng = np.random.default_rng(0)
    means = diffs[rng.integers(0, len(diffs), (n, len(diffs)))].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(cond, subset=None):
    qs = subset or qids
    rows = [(cells[(cond, q, s)], cells[('baseline', q, s)]) for q in qs for s in a.seeds]
    out = {'n': len(rows),
           'words_mean': statistics.mean(r['words'] for r, _ in rows),
           'total_word_reduction': 1 - sum(r['words'] for r, _ in rows) / sum(b['words'] for _, b in rows),
           'median_per_answer_reduction': statistics.median(1 - r['words'] / b['words'] for r, b in rows),
           'core': sum(r['core'] for r, _ in rows), 'whole': sum(r['whole'] for r, _ in rows),
           'completion_failures': sum(r['fail'] for r, _ in rows)}
    if 'lazy_chosen' in rows[0][0]:
        out['lazy_chosen'] = sum(bool(r.get('lazy_chosen')) for r, _ in rows)
    if cond != 'baseline':
        for metric in ('core', 'whole'):
            d = [statistics.mean(cells[(cond, q, s)][metric] - cells[('baseline', q, s)][metric] for s in a.seeds) for q in qs]
            mean, lo, hi = boot(d)
            out[f'{metric}_diff_pts'] = {'mean': 100 * mean, 'ci95': [100 * lo, 100 * hi]}
        new_err = sum(1 for q in qs for s in a.seeds if cells[('baseline', q, s)]['core'] and not cells[(cond, q, s)]['core'])
        fixed = sum(1 for q in qs for s in a.seeds if not cells[('baseline', q, s)]['core'] and cells[(cond, q, s)]['core'])
        out['new_core_errors'], out['fixed_core_errors'] = new_err, fixed
    return out


cats = sorted({q['category'] for q in questions.values()})
report = {'seeds': a.seeds, 'questions': len(qids), 'conditions': {}, 'by_category': {}}
for c in conditions:
    report['conditions'][c] = summarize(c)
for cat in cats:
    qs = [q for q in qids if questions[q]['category'] == cat]
    report['by_category'][cat] = {c: {k: v for k, v in summarize(c, qs).items()
                                      if k in ('n', 'core', 'whole', 'total_word_reduction', 'lazy_chosen', 'words_mean')}
                                  for c in conditions}
lazy_red = report['conditions']['lazy']['total_word_reduction']
self_c = report['conditions']['self']
plan = report['by_category'].get('planning', {})
report['hypotheses'] = {
    'H4a_self_keeps_half_of_lazy_reduction': self_c['total_word_reduction'] >= 0.5 * lazy_red,
    'H4b_self_planning_core_ge_lazy': plan.get('self', {}).get('core', 0) >= plan.get('lazy', {}).get('core', 0),
    'H4c_self_core_lower_bound_gt_minus5': self_c['core_diff_pts']['ci95'][0] > -5,
}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(report, indent=1))
print(json.dumps({c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
                  for c, s in report['conditions'].items()}, indent=1))
print(json.dumps(report['hypotheses']))
