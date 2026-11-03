"""A big, readable chart for the X post (PNG).

  uv run --with matplotlib python paper/post_chart.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
a = json.loads((ROOT / 'results' / 'smart_lazy.json').read_text())['conditions']
rows = [('Claude decides', 'routed', '#15803d'), ('Always lazy', 'lazy', '#2b6cb0'),
        ('Qwen decides itself', 'self', '#d97706'), ('"Be concise" prompt', 'prompt', '#b91c1c')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, gridspec_kw={'wspace': 0.08})
names = [r[0] for r in rows]
y = list(range(len(rows)))[::-1]
words = [100 * a[k]['total_word_reduction'] for _, k, _ in rows]
acc = [a[k]['core_diff_pts']['mean'] for _, k, _ in rows]
cols = [c for _, _, c in rows]
ax1.barh(y, words, color=cols, height=0.62)
for yi, w in zip(y, words):
    ax1.text(w + 1.5, yi, f'{w:.0f}%', va='center', fontsize=15, fontweight='bold')
ax1.set_xlim(0, 112)
ax1.set_title('shorter answers', fontsize=16, loc='left', color='#333')
ax1.set_yticks(y, names, fontsize=16)
ax2.barh(y, acc, color=cols, height=0.62)
for yi, v in zip(y, acc):
    label = f'{v:+.0f} pts' if abs(v) >= 1 else ('±0 pts' if abs(v) < 0.05 else f'{v:+.1f} pts')
    ax2.text(v + (0.8 if v >= 0 else -0.8), yi, label, va='center',
             ha='left' if v >= 0 else 'right', fontsize=15, fontweight='bold')
ax2.axvline(0, color='#888', lw=1)
ax2.set_xlim(-44, 14)
ax2.spines['left'].set_visible(False)
ax2.set_title('accuracy vs. normal Qwen', fontsize=16, loc='left', color='#333')
for ax in (ax1, ax2):
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', length=0)
fig.suptitle('Who should decide when Qwen3.5-4B switches on its "lazy" neurons?', fontsize=19, x=0.02, ha='left', y=0.99)
fig.text(0.02, 0.01, '96 sealed questions × 2 seeds, blind-graded. Lazy = 344-neuron edit on the first 6 tokens.  github.com/21J3phy/self-steer',
         fontsize=11.5, color='#555')
fig.subplots_adjust(left=0.2, right=0.98, top=0.83, bottom=0.08)
out = ROOT / 'results' / 'post_chart.png'
fig.savefig(out, dpi=200)
print(out)
