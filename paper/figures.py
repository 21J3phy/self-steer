"""Figures for the paper, from the result JSON files in results/.

  uv run --with matplotlib python paper/figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
OUT = ROOT / 'paper'
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})

COLORS = {'lazy': '#2b6cb0', 'self': '#d97706', 'placebo': '#a16207', 'prompt': '#b91c1c', 'routed': '#15803d',
          'oracle': '#6b7280'}
LABELS = {'lazy': 'always lazy', 'self': 'Qwen decides (self)', 'placebo': 'self-placebo (no edit)',
          'prompt': '"Be concise" prompt', 'routed': 'Claude routes', 'oracle': 'hindsight oracle'}


def tradeoff():
    a = json.loads((R / 'smart_lazy.json').read_text())
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.axhline(0, color='#999', lw=0.8)
    ax.axhline(-5, color='#999', lw=0.8, ls=':')
    ax.text(1, -6.0, 'noninferiority margin (-5 pts)', fontsize=7, color='#666', va='top')
    ax.scatter([0], [0], color='black', zorder=3)
    ax.annotate('normal', (0, 0), xytext=(-4, 5), textcoords='offset points', fontsize=7.5)
    for c in ['lazy', 'self', 'placebo', 'prompt', 'routed', 'oracle']:
        if c not in a['conditions']:
            continue
        s = a['conditions'][c]
        x = 100 * s['total_word_reduction']
        y, (lo, hi) = s['core_diff_pts']['mean'], s['core_diff_pts']['ci95']
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt='o', color=COLORS[c], capsize=2.5, ms=5,
                    mfc='white' if c == 'oracle' else COLORS[c], zorder=3)
        offsets = {'routed': (-36, 6), 'oracle': (6, 3), 'lazy': (5, -11), 'placebo': (6, 2), 'self': (6, -3),
                   'prompt': (-20, -14)}
        ax.annotate(LABELS[c], (x, y), xytext=offsets.get(c, (6, -3)), textcoords='offset points',
                    fontsize=7.5, color=COLORS[c])
    ax.set_xlabel('fewer words than normal Qwen (%)')
    ax.set_ylabel('core accuracy vs. normal (pts, 95% CI)')
    ax.set_xlim(-5, 100)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_tradeoff.pdf')


def choices():
    a = json.loads((R / 'smart_lazy.json').read_text())
    cats = ['python', 'shell_git', 'arithmetic_time', 'writing_reading', 'concept', 'planning']
    names = ['Python', 'Shell/Git', 'Arith./time', 'Writing', 'Concept', 'Planning']
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    width = 0.38
    for k, c in enumerate(['self', 'routed']):
        vals = [100 * a['by_category'][cat][c]['lazy_chosen'] / a['by_category'][cat][c]['n'] for cat in cats]
        ax.bar([i + (k - 0.5) * width for i in range(len(cats))], vals, width, color=COLORS[c], label=LABELS[c])
    ax.set_xticks(range(len(cats)), names, fontsize=7.5)
    ax.set_ylabel('chose lazy (%)')
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_choices.pdf')


if __name__ == '__main__':
    tradeoff()
    choices()
    print('ok')
