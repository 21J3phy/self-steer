"""Writes paper/numbers.tex (LaTeX macros) from results/*.json, so every number in the paper is generated.

  uv run python paper/make_numbers.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
macros = {}


def pct(x, d=0):
    return f'{100 * x:.{d}f}'


def pts(x, d=1):
    return f'{x:+.{d}f}'


def ci(c, d=1):
    return f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'


parity = json.loads((R / 'parity.json').read_text())
macros['FidelityCells'] = str(parity['cells'])
macros['FidelityIdentical'] = str(parity['tokens_identical'])

if (R / 'bench.json').exists():
    b = json.loads((R / 'bench.json').read_text())
    for k, name in [('none', 'None'), ('lazy_window6', 'LazySix'), ('lazy_whole_answer', 'Whole'),
                    ('ablate_9_layers', 'Ablate'), ('whole_plus_ablate', 'Both')]:
        macros[f'Decode{name}'] = f"{b['decode'][k]['tok_per_s_median']:.1f}"
    macros['BuildTwo'] = f"{b['build']['2_examples']['seconds_median']:.1f}"
    macros['BuildEight'] = f"{b['build']['8_examples']['seconds_median']:.1f}"
    for n, letter in (('512', 'A'), ('1024', 'B'), ('2048', 'C'), ('4096', 'D')):
        macros[f'Reprefill{letter}'] = f"{b['reprefill'][n]['seconds_median']:.1f}"
    macros['ReprefillTps'] = f"{b['reprefill']['2048']['tok_per_s']:.0f}"
    macros['SnapshotMs'] = f"{1000 * b['snapshot']['2048']['seconds_median']:.1f}"
    macros['SwitchUs'] = f"{1e6 * b['switch_seconds_median']:.0f}"

s = json.loads((R / 'smart_lazy.json').read_text())
C = s['conditions']
for cond, name in [('lazy', 'Lazy'), ('self', 'Self'), ('prompt', 'Prompt'), ('routed', 'Routed'),
                   ('oracle', 'Oracle'), ('placebo', 'Placebo'), ('baseline', 'Base')]:
    if cond not in C:
        continue
    c = C[cond]
    macros[f'{name}Words'] = pct(c['total_word_reduction'])
    macros[f'{name}Median'] = pct(c['median_per_answer_reduction'])
    macros[f'{name}WordsMean'] = f"{c['words_mean']:.0f}"
    macros[f'{name}CoreN'] = str(c['core'])
    macros[f'{name}WholeN'] = str(c['whole'])
    macros[f'{name}Fail'] = str(c['completion_failures'])
    if cond != 'baseline':
        macros[f'{name}Core'] = pts(c['core_diff_pts']['mean'])
        macros[f'{name}CoreAbs'] = f"{abs(c['core_diff_pts']['mean']):.0f}"
        macros[f'{name}CoreCI'] = ci(c['core_diff_pts']['ci95'])
        macros[f'{name}Whole'] = pts(c['whole_diff_pts']['mean'])
        macros[f'{name}WholeCI'] = ci(c['whole_diff_pts']['ci95'])
        macros[f'{name}New'] = str(c['new_core_errors'])
        macros[f'{name}Fixed'] = str(c['fixed_core_errors'])
    if 'lazy_chosen' in c:
        macros[f'{name}Chosen'] = str(c['lazy_chosen'])
macros['Cells'] = str(C['baseline']['n'])
if 'calibration' in s:
    macros['CalCore'] = str(s['calibration']['core_correct'])
    macros['CalWhole'] = str(s['calibration']['correct'])
    macros['CalN'] = str(s['calibration']['n'])
for k, v in s.get('extra', {}).items():
    macros[k] = str(v)

if (R / 'context.json').exists():
    ctx = json.loads((R / 'context.json').read_text())['summary']
    for key, name in [('none', 'None'), ('lazy+keep', 'LazyKeep'), ('lazy+reprefill', 'LazyRe'),
                      ('full+keep', 'FullKeep'), ('full+reprefill', 'FullRe')]:
        macros[f'Recall{name}'] = str(ctx[key]['recalled'])
        if 'kl_mean' in ctx[key]:
            macros[f'KL{name}'] = f"{ctx[key]['kl_mean']:.4f}"
            macros[f'KLmax{name}'] = f"{ctx[key]['kl_max']:.3f}"
            macros[f'Agree{name}'] = pct(ctx[key]['argmax_agree'], 1)
        if 'reprefill_seconds_mean' in ctx[key]:
            macros[f'RepSec{name}'] = f"{ctx[key]['reprefill_seconds_mean']:.1f}"
    macros['CtxN'] = str(ctx['none']['n'])

if (R / 'modes_eval.json').exists():
    m = json.loads((R / 'modes_eval.json').read_text())
    for mode, name in [('eli5', 'Eli'), ('bullets', 'Bul'), ('plain-math', 'Math'), ('plain-math-add', 'MathAdd')]:
        if mode not in m:
            continue
        for cond, cn in [('normal', 'Normal'), ('mode', 'Mode'), ('instruction', 'Instr')]:
            sm = m[mode]['summary'][cond]
            macros[f'{name}{cn}Words'] = f"{sm['words_mean']:.0f}"
            if sm.get('fk_grade_mean') is not None:
                macros[f'{name}{cn}FK'] = f"{sm['fk_grade_mean']:.1f}"
            macros[f'{name}{cn}Bul'] = str(sm['mostly_bullets'])
            macros[f'{name}{cn}Tex'] = str(sm['has_latex'])
            if 'numeric_correct' in sm:
                macros[f'{name}{cn}Num'] = str(sm['numeric_correct'])

if (R / 'demo.json').exists():
    d = json.loads((R / 'demo.json').read_text())
    for k, v in d.get('macros', {}).items():
        macros[k] = str(v)

lines = [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())]
(ROOT / 'paper' / 'numbers.tex').write_text('\n'.join(lines) + '\n')
print(len(lines), 'macros')
