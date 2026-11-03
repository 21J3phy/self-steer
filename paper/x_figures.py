"""Figures for the X thread (1600x900 PNGs) -> results/x/.

  uv run --with matplotlib python paper/x_figures.py

Numbers come from results/ where they are stored there; the rest are kernel outputs logged in
results/remote_calls.jsonl (each is marked with the call that printed it).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'x'
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
BLUE, ORANGE, AQUA, RED, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#b4b3ad'
plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Helvetica', 'DejaVu Sans'], 'font.size': 11,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'axes.titlesize': 11.5, 'axes.titleweight': 'medium',
    'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'axes.titlepad': 10,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
})


def frame(title, subtitle, ncols=1, widths=None, top=0.74, bottom=0.14, wspace=0.32, left=0.07):
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    fig.text(0.04, 0.93, title, fontsize=15, fontweight='bold', color=INK, va='top')
    fig.text(0.04, 0.855, subtitle, fontsize=10.5, color=INK2, va='top')
    gs = fig.add_gridspec(1, ncols, left=left, right=0.97, top=top, bottom=bottom, wspace=wspace,
                          width_ratios=widths)
    return fig, gs


def m(x, fmt='{:+.1f}'):
    return fmt.format(x).replace('-', '\u2212')


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print('wrote', OUT / name)


# ---------------------------------------------------------------- 1. the system
def fig_system():
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    fig.text(0.04, 0.93, 'Edits to W_down, switched between tokens', fontsize=15, fontweight='bold', color=INK, va='top')
    fig.text(0.04, 0.855, 'Qwen can switch an edit on itself; Claude reads neurons and weights and writes new edits',
             fontsize=10.5, color=INK2, va='top')

    def box(x, y, w, h, fc, ec, lw=1.0, r=0.18):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))

    # Qwen: a layer stack with down_proj patches
    box(0.7, 0.9, 5.6, 5.8, '#f3f2ef', GRID)
    ax.text(0.95, 6.35, 'Qwen3.5-4B', fontsize=12, fontweight='bold', color=INK)
    ax.text(0.95, 5.95, '32 layers · edits sit on each MLP down_proj', fontsize=9, color=INK2)
    for i in range(12):
        y = 1.2 + i * 0.37
        ax.add_patch(FancyBboxPatch((1.0, y), 3.0, 0.24, boxstyle='round,pad=0,rounding_size=0.06', fc='#e4e3df', ec='none'))
        if i in (3, 5, 6, 8, 9, 10):
            ax.add_patch(FancyBboxPatch((4.15, y), 0.9, 0.24, boxstyle='round,pad=0,rounding_size=0.06', fc=BLUE, ec='none'))
    ax.text(5.2, 3.25, 'W_down\ncolumn\nedits', fontsize=8.5, color=BLUE, va='center')
    # token stream with the switch
    toks = ['…', 'set_mode', '(lazy)', '→', 'The', 'total', 'is', '…']
    x = 7.0
    ax.text(7.0, 6.35, 'One answer, edits gated by token position', fontsize=10, fontweight='medium', color=INK)
    for j, t in enumerate(toks):
        w = 0.32 + 0.105 * len(t)
        on = 4 <= j <= 6
        ax.add_patch(FancyBboxPatch((x, 5.35), w, 0.5, boxstyle='round,pad=0,rounding_size=0.1',
                                    fc=BLUE if on else '#ecebe7', ec='none'))
        ax.text(x + w / 2, 5.6, t, ha='center', va='center', fontsize=8.5, color='white' if on else INK2,
                family='monospace')
        x += w + 0.12
    ax.text(7.0, 4.95, 'Qwen calls a tool; the edit turns on for the next 6 tokens. ~1 µs to switch.', fontsize=8.5, color=INK2)
    # Claude controller
    box(7.0, 0.9, 8.3, 3.55, SURFACE, BLUE, lw=1.2)
    ax.text(7.3, 4.0, 'Claude (controller, over MCP)', fontsize=11, fontweight='bold', color=INK)
    rows = [('kernel', 'Python over raw weights and saved reads'),
            ('reads', 'activations, residuals, direct logit attribution'),
            ('budget', '10 forward passes per task'),
            ('edit', 'save_edit: scale / project specific W_down columns')]
    for k, (a, b) in enumerate(rows):
        yy = 3.35 - k * 0.6
        ax.add_patch(FancyBboxPatch((7.3, yy - 0.2), 1.2, 0.42, boxstyle='round,pad=0,rounding_size=0.08', fc='#e8f0fb', ec='none'))
        ax.text(7.9, yy, a, ha='center', va='center', fontsize=8.5, color=BLUE, fontweight='medium')
        ax.text(8.7, yy, b, va='center', fontsize=9, color=INK2)
    ax.add_patch(FancyArrowPatch((7.0, 2.3), (6.35, 2.3), arrowstyle='-|>', mutation_scale=12, color=BLUE, lw=1.3))
    ax.text(0.7, 0.35, 'self-steer · github.com/21J3phy/self-steer', fontsize=8.5, color=MUTED)
    save(fig, '1_system.png')


# ---------------------------------------------------------------- 2. who holds the switch
def fig_deciders():
    c = json.loads((ROOT / 'results' / 'smart_lazy.json').read_text())['conditions']
    rows = [('Claude routes from question text', 'routed', BLUE), ('Always on', 'lazy', GRAY),
            ('Qwen decides, via tool call', 'self', ORANGE), ('Placebo tool call, no edit', 'placebo', GRAY),
            ('"Be concise" system prompt', 'prompt', GRAY)]
    fig, gs = frame('Who holds the switch decides what the edit costs',
                    'Length-reducing edit · 96 sealed questions × 2 seeds · blind-graded · 95% CI',
                    ncols=2, widths=[1, 1.25], wspace=0.08, left=0.30, bottom=0.12)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    y = list(range(len(rows)))[::-1]
    for yi, (name, k, col) in zip(y, rows):
        w = 100 * c[k]['total_word_reduction']
        a1.barh(yi, w, height=0.5, color=col)
        a1.text(w + 2, yi, f'{w:.0f}%', va='center', fontsize=10, color=INK)
        d = c[k]['core_diff_pts']; lo, hi = d['ci95']
        a2.plot([lo, hi], [yi, yi], color=col if col != GRAY else MUTED, lw=2, solid_capstyle='round')
        a2.plot(d['mean'], yi, 'o', ms=8, color=col if col != GRAY else MUTED, mec=SURFACE, mew=2)
        a2.text(d['mean'], yi + 0.22, m(d['mean']), ha='center', va='bottom', fontsize=10, color=INK)
    a1.set_yticks(y, [r[0] for r in rows], fontsize=10, color=INK)
    a1.tick_params(axis='y', length=0)
    a1.set_xlim(0, 118); a1.set_xticks([0, 50, 100], ['0', '50', '100%']); a1.set_ylim(-0.6, 4.7)
    a1.set_title('Answers shorter'); grid(a1, 'x'); a1.spines['left'].set_visible(False)
    a2.axvline(0, color=INK2, lw=0.9)
    a2.set_yticks([]); a2.spines['left'].set_visible(False)
    a2.set_xlim(-42, 12); a2.set_ylim(-0.6, 4.7); a2.set_title('Core accuracy vs. normal (points)'); grid(a2, 'x')
    save(fig, '2_deciders.png')


# ---------------------------------------------------------------- 3. precision
def fig_precision():
    n = json.loads((ROOT / 'paper' / 'numbers.json').read_text()) if (ROOT / 'paper' / 'numbers.json').exists() else None
    # 8-bit (E4, local MLX): ArithSelf 24, ArithPlacebo 37 (paper/numbers.tex); 35/48 lazy calls (results/smart_lazy/answers.jsonl)
    # FP32 (E7, Kaggle): results/e7/scored.json
    s = json.loads((ROOT / 'results' / 'e7' / 'scored.json').read_text())
    fp_core = {c: sum(r['core_correct'] for r in s if r['condition'] == c) for c in ('placebo', 'self')}
    fp_calls = sum('lazy' in r['called'] for r in s if r['condition'] == 'self')
    rows = [l for l in (ROOT / 'results' / 'smart_lazy' / 'answers.jsonl').read_text().splitlines() if l.strip()]
    e4 = [json.loads(l) for l in rows]
    e4 = [r for r in e4 if r.get('category') == 'arithmetic_time' and r['condition'] == 'self']
    q8_calls = sum(any((t.get('call') or {}).get('name') == 'set_mode' and str(((t.get('call') or {}).get('arguments') or {}).get('name', '')).lower() == 'lazy' for t in r.get('tool_calls', [])) for r in e4)
    fig, gs = frame('The self-selection failure moved with precision',
                    'Same 24 arithmetic & time questions × 2 seeds, same template, tool and prompt',
                    ncols=2, widths=[1, 1.4], wspace=0.35)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    a1.bar([0, 1], [q8_calls, fp_calls], width=0.55, color=ORANGE)
    for x, v in zip([0, 1], [q8_calls, fp_calls]):
        a1.text(x, v + 1, f'{v}/48', ha='center', fontsize=10.5, color=INK)
    a1.set_xticks([0, 1], ['8-bit MLX', 'FP32'], color=INK); a1.set_ylim(0, 52); a1.set_yticks([0, 24, 48])
    a1.set_title('Answers where Qwen switched lazy on'); grid(a1)
    xs = [0, 1]; w = 0.34
    pl = [37, fp_core['placebo']]; sf = [24, fp_core['self']]
    a2.bar([x - w / 2 - 0.02 for x in xs], pl, width=w, color=GRAY, label='placebo tool call')
    a2.bar([x + w / 2 + 0.02 for x in xs], sf, width=w, color=ORANGE, label='Qwen switches lazy on')
    for x, p, q in zip(xs, pl, sf):
        a2.text(x - w / 2 - 0.02, p + 1, str(p), ha='center', fontsize=10, color=INK)
        a2.text(x + w / 2 + 0.02, q + 1, str(q), ha='center', fontsize=10, color=INK)
    a2.set_xticks(xs, ['8-bit MLX', 'FP32'], color=INK); a2.set_ylim(0, 52); a2.set_yticks([0, 24, 48])
    a2.set_title('Core correct, of 48'); grid(a2)
    a2.legend(frameon=False, fontsize=9.5, loc='upper right', ncol=1, labelcolor=INK2)
    save(fig, '3_precision.png')


# ---------------------------------------------------------------- 4. LaTeX
def fig_latex():
    fig, gs = frame('DLA finds the LaTeX writers. Silencing them thins LaTeX, never removes it',
                    'Plain-text math · column edits chosen by direct logit attribution · 10-pass budget',
                    ncols=3, widths=[1, 1, 1.1], wspace=0.42, top=0.68, bottom=0.2)
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))
    # cumulative share of positive attribution (call j11 in remote_calls.jsonl; top-1 = 1.197 of 74.91)
    k = ['1', '64', '256', '512', '1024']; share = [100 * 1.197 / 74.91, 15, 22, 26, 31]
    a1.plot(range(5), share, '-o', color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5)
    a1.set_ylim(0, 100); a1.set_yticks([0, 50, 100], ['0', '50', '100%'])
    a1.set_xticks(range(5), k); a1.set_xlabel('top-k neurons')
    a1.set_title('Share of attribution\nin the top-k neurons'); grid(a1)
    a1.text(3, 26 + 6, '26%', ha='center', fontsize=9.5, color=INK)
    # practice dose-response (passes 1, 5, 6, 8; LaTeX tokens per 100 words)
    dens = [21.2, 20.9, 18.2, 10.9]
    a2.plot(range(4), dens, '-o', color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5)
    for x, d in enumerate(dens):
        a2.text(x, d + 1.6, f'{d:.1f}', ha='center', fontsize=9, color=INK)
    a2.set_xticks(range(4), ['0', '1', '256', '512']); a2.set_xlabel('neurons silenced (by DLA rank)')
    a2.set_ylim(0, 28); a2.set_title('Practice: LaTeX tokens\nper 100 words'); grid(a2)
    # held-out
    names = ['normal', '512 off', 'prompt']; vals = [25.6, 10.1, 9.7]; any_tex = ['20/20', '20/20', '14/20']
    a3.bar(range(3), vals, width=0.55, color=[GRAY, BLUE, AQUA])
    for x, v, t in zip(range(3), vals, any_tex):
        a3.text(x, v + 0.8, f'{v:.1f}', ha='center', fontsize=9.5, color=INK)
    a3.set_xticks(range(3), [f'{n}\n{t}' for n, t in zip(names, any_tex)], fontsize=9.2, color=INK)
    a3.set_xlabel('answers with any LaTeX')
    a3.set_ylim(0, 30); a3.set_title('Held-out, 20 prompts:\nLaTeX tokens per 100 words'); grid(a3)
    save(fig, '4_latex.png')


# ---------------------------------------------------------------- 5. eli5 read
def fig_eli5_read():
    fig, gs = frame('A matched-context read isolates "explain like I\'m 5"',
                    'Same answer texts, read with vs. without the instruction · logit lens of the difference at L31',
                    ncols=2, widths=[1, 1.15], wspace=0.3, top=0.72)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    a1.axis('off'); a1.set_xlim(0, 2.25); a1.set_ylim(0, 9)
    a1.set_title('Tokens the instruction effect pushes', pad=10)
    up = [' yummy', ' big', ' mommy', ' silly', ' sleepy', ' pretend', ' helpers', ' comfy']
    dn = [' fundamentally', ' essentially', ' predominantly', ' primarily', ' extensively', ' subsequently', ' physiological', ' comprehensive']
    a1.text(0.0, 8.55, 'up', fontsize=9.5, color=BLUE, fontweight='medium')
    a1.text(1.05, 8.55, 'down', fontsize=9.5, color=RED, fontweight='medium')
    for i, (u, d) in enumerate(zip(up, dn)):
        yy = 7.7 - i * 1.0
        for x0, t, col in ((0.0, u, BLUE), (1.05, d, RED)):
            a1.add_patch(FancyBboxPatch((x0, yy - 0.32), 0.1 + 0.068 * len(t.strip()), 0.64,
                                        boxstyle='round,pad=0,rounding_size=0.12', fc=col, ec='none', alpha=0.12))
            a1.text(x0 + 0.06, yy, t.strip(), va='center', fontsize=9.5, color=INK, family='monospace')
    # MLP direct share (call k13 output: fraction of the effect at layer l written by MLPs L0..l)
    L = [4, 8, 12, 16, 20, 24, 28, 31]; f = [0.02, -0.18, -0.19, -0.11, -0.01, 0.11, 0.23, 0.19]
    a2.axhspan(-0.3, 0, color=RED, alpha=0.06, lw=0)
    a2.axhline(0, color=INK2, lw=0.9)
    a2.plot(L, [100 * v for v in f], '-o', color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5)
    a2.text(31, 19 + 6, '19%', ha='center', fontsize=9.5, color=INK)
    a2.text(5, -26, 'MLPs work against it', fontsize=9, color=RED)
    a2.set_ylim(-30, 100); a2.set_yticks([0, 50, 100], ['0', '50', '100%'])
    a2.set_xticks([0, 8, 16, 24, 31]); a2.set_xlabel('layer')
    a2.set_title('Share written directly by MLPs (layers ≤ ℓ)'); grid(a2)
    a2.text(16, 70, 'the rest is attention', fontsize=9.5, color=INK2, ha='center')
    save(fig, '5_eli5_read.png')


# ---------------------------------------------------------------- 6. eli5 edit
def fig_eli5_edit():
    fig, gs = frame('Amplifying aligned neurons: \u22123.5 grade levels. The prompt: \u22127.4',
                    'Scaling edits on chosen neurons · Flesch\u2013Kincaid grade · practice passes, then 20 held-out prompts',
                    ncols=2, widths=[1.35, 1], wspace=0.3, bottom=0.2)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    what = ['p4: 1024 set\nto kid level', 'p6: 2048\n\u00d74', 'p7: 4096\naligned \u00d74', 'p8: 3584 aligned\n\u00d74, no L30\u201331']
    delta = [10.4 - 9.3, 6.9 - 9.3, 5.9 - 10.0, 6.5 - 10.0]
    loops = ['0/8', '1/8', '2/16', '0/16']
    a1.bar(range(4), delta, width=0.55, color=[GRAY, GRAY, GRAY, BLUE], zorder=2)
    a1.axhline(0, color=INK2, lw=0.9)
    a1.axhline(-3, color=INK2, lw=0.9, ls=(0, (3, 3)), zorder=1)
    a1.text(-0.32, -3.12, 'pre-registered\nbar: \u22123', fontsize=8.3, color=INK2, va='top')
    for x, d, l in zip(range(4), delta, loops):
        if d > 0:
            a1.text(x, d + 0.15, m(d), ha='center', va='bottom', fontsize=9.5, color=INK)
        else:
            a1.text(x, d + 0.12, m(d), ha='center', va='bottom', fontsize=9.5, zorder=3,
                    color='white' if x == 3 else INK, fontweight='medium')
        a1.text(x, 2.2, f'loops {l}', ha='center', fontsize=8.5, color=RED if l[0] != '0' else INK2)
    a1.set_xticks(range(4), what, fontsize=8.3)
    a1.set_ylim(-5.4, 2.7); a1.set_title('Practice: grade change vs. normal'); grid(a1)
    vals = [10.7, 7.2, 3.3]
    a2.bar(range(3), vals, width=0.55, color=[GRAY, BLUE, AQUA])
    for x, v in enumerate(vals):
        a2.text(x, v + 0.3, f'{v:.1f}', ha='center', fontsize=10, color=INK)
    a2.set_xticks(range(3), ['normal', 'edit\n1/20 loops', 'prompt'], color=INK, fontsize=9.5)
    a2.set_ylim(0, 12.5); a2.set_title('Held-out grade, 20 prompts'); grid(a2)
    save(fig, '6_eli5_edit.png')


# ---------------------------------------------------------------- 7. digits
def fig_digits():
    fig, gs = frame('After the tool call, lazy\'s extra digit push lives in the last layer',
                    'Teacher-forced reads of 9 good answers (first 6 tokens) · projection on the digit readout, recovered from W31',
                    ncols=2, widths=[1, 1.4], wspace=0.28, bottom=0.2)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # tool-only component (tool effect − bare effect), pre-norm residual (kernel k_e7d)
    L = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    v = [-0.02, -0.01, 0.0, 0.01, 0.0, 0.02, -0.02, -0.02, -0.02, -0.06, -0.04, -0.09]
    a1.axhline(0, color=INK2, lw=0.9)
    a1.plot(L, v, '-o', color=BLUE, lw=2, ms=5, mec=SURFACE, mew=1.5)
    a1.set_ylim(-1, 1); a1.set_yticks([-1, 0, 1], ['\u22121', '0', '1']); a1.set_xticks([8, 16, 24, 30]); a1.set_xlabel('layer')
    a1.set_title('Tool-only digit push, layers 8–30'); grid(a1)
    a1.text(19, 0.45, 'flat until layer 31', ha='center', fontsize=9.5, color=INK2)
    # final normalized residual on the digit direction (kernels k_e7g, fix reads)
    names = ['bare', 'bare\n+ lazy', 'tool', 'tool\n+ lazy', 'fix 1\nproject', 'fix 2\nrestore']
    vals = [5.92, 6.44, 5.53, 7.62, 4.93, 6.72]
    cols = [GRAY, GRAY, GRAY, ORANGE, BLUE, BLUE]
    a2.bar(range(6), vals, width=0.6, color=cols)
    for x, val in enumerate(vals):
        a2.text(x, val + 0.12, f'{val:.2f}', ha='center', fontsize=9, color=INK)
    a2.set_xticks(range(6), names, fontsize=8.6)
    a2.set_ylim(0, 8.6); a2.set_title('Digit level at the final layer (normalized)'); grid(a2)
    fig.subplots_adjust(bottom=0.23)
    save(fig, '7_digits.png')


if __name__ == '__main__':
    fig_system(); fig_deciders(); fig_precision(); fig_latex(); fig_eli5_read(); fig_eli5_edit(); fig_digits()
