"""E5: held-out evaluation of the modes Claude built over MCP.

For each mode: 20 held-out prompts x {normal, mode, instruction}, where
`instruction` asks for the same style in the system prompt (no edit).

  uv run python experiments/modes_eval.py --modes eli5 bullets plain-math --out runs/modes_eval.json
"""
import argparse
import json
from pathlib import Path
import random
import re
import statistics

from selfsteer.decode import Sampling
from selfsteer.library import Library
from selfsteer.model import Engine
from selfsteer.session import Session

ELI5_PROMPTS = [
    'How do vaccines work?', 'Why do we have seasons?', 'How does the internet work?', 'What is electricity?',
    'Why do cats purr?', 'How do airplanes stay in the air?', 'What causes earthquakes?', 'How does a battery work?',
    'Why do leaves change color in the fall?', 'What is DNA?', 'How does a computer remember things?',
    'Why does the moon change shape?', 'What is inflation in the economy?', 'How do magnets work?',
    'Why do we need sleep?', 'How does GPS know where I am?', 'What is a black hole?', 'How do bees make honey?',
    'Why is the ocean salty?', 'How does a microwave heat food?']
BULLET_PROMPTS = [
    'How can I sleep better?', 'What should I pack for a weekend camping trip?', 'How do I prepare for a job interview?',
    'What are good ways to save money on groceries?', 'How do I start learning to code?', 'How can I make my laptop battery last longer?',
    'What should I check before buying a used car?', 'How do I keep houseplants alive?', 'What are the benefits of regular exercise?',
    'How do I write a good cover letter?', 'What makes a strong password?', 'How do I reduce stress during exams?',
    'What should I do in my first week at a new job?', 'How can I be more productive working from home?',
    'What are tips for public speaking?', 'How do I plan a budget for a trip abroad?', 'How do I clean a cast iron pan?',
    'What should a beginner know about investing?', 'How do I make my resume stand out?', 'How do I train for a 5K run?']


def math_prompts(n=20, seed=5):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            price, pct = rng.randrange(20, 400), rng.choice([10, 15, 20, 25, 30, 40])
            out.append((f'A jacket costs ${price} and is {pct}% off. What is the sale price?', round(price * (100 - pct) / 100, 2)))
        elif kind == 1:
            speed, hours = rng.randrange(40, 90), rng.choice([1.5, 2, 2.5, 3, 4])
            out.append((f'A car drives at {speed} km/h for {hours} hours. How far does it go?', round(speed * hours, 2)))
        elif kind == 2:
            a, b = rng.randrange(12, 99), rng.randrange(12, 99)
            out.append((f'A box holds {a} pencils. How many pencils are in {b} boxes?', a * b))
        else:
            total, people = rng.randrange(60, 300), rng.choice([3, 4, 5, 6, 8])
            out.append((f'A dinner bill of ${total} is split equally among {people} friends. How much does each pay?',
                        round(total / people, 2)))
    return out


def syllables(word):
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    groups = re.findall(r'[aeiouy]+', word)
    n = len(groups) - (1 if word.endswith('e') and len(groups) > 1 and not word.endswith('le') else 0)
    return max(1, n)


def fk_grade(text):
    words = re.findall(r"[A-Za-z']+", text)
    sentences = max(1, len(re.findall(r'[.!?]+', text)))
    if not words:
        return None
    return 0.39 * len(words) / sentences + 11.8 * sum(syllables(w) for w in words) / len(words) - 15.59


def mostly_bullets(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    bullets = sum(bool(re.match(r'^([-*•]|\d+[.)])\s', l)) for l in lines)
    return bullets / len(lines) >= 0.5


LATEX = re.compile(r'\\\(|\\\[|\\frac|\\boxed|\\times|\\text|\\cdot|\$[^$\n]+\$')


def has_number(text, value):
    nums = [float(x.replace(',', '')) for x in re.findall(r'\d[\d,]*\.?\d*', text)]
    return any(abs(n - float(value)) < 0.011 for n in nums)


TASKS = {
    'eli5': {'prompts': [(p, None) for p in ELI5_PROMPTS],
             'instruction': 'Explain it like I am five years old: very simple words and short sentences.'},
    'bullets': {'prompts': [(p, None) for p in BULLET_PROMPTS], 'instruction': 'Answer as a short bulleted list.'},
    'plain-math': {'prompts': math_prompts(), 'instruction': 'Do not use LaTeX. Write all math in plain text.'},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-tokens', type=int, default=700)
    args = ap.parse_args()
    engine = Engine.load()
    lib = Library(engine).load_dir()
    report = {}
    for mode in args.modes:
        task = TASKS[mode]
        rows = []
        for i, (prompt, answer) in enumerate(task['prompts']):
            row = {'prompt': prompt, 'answer': answer}
            for cond in ('normal', 'mode', 'instruction'):
                s = Session(engine, lib, use_tools=False, sampling=Sampling(max_tokens=args.max_tokens),
                            system=task['instruction'] if cond == 'instruction' else None)
                if cond == 'mode':
                    s.set_mode(mode, 'answer', by='controller')
                turn = s.send(prompt, seed=2000 + i)
                text = turn.reply
                row[cond] = {'text': text, 'words': len(text.split()), 'finish': turn.finish,
                             'fk_grade': fk_grade(text), 'bullets': mostly_bullets(text),
                             'latex': bool(LATEX.search(text)),
                             'correct': has_number(text, answer) if answer is not None else None}
            rows.append(row)
            print(mode, i, {c: (row[c]['words'], round(row[c]['fk_grade'] or 0, 1), row[c]['bullets'], row[c]['latex'], row[c]['correct'])
                            for c in ('normal', 'mode', 'instruction')}, flush=True)
        summary = {}
        for cond in ('normal', 'mode', 'instruction'):
            vals = [r[cond] for r in rows]
            fk = [v['fk_grade'] for v in vals if v['fk_grade'] is not None]
            summary[cond] = {'n': len(vals), 'words_mean': statistics.mean(v['words'] for v in vals),
                             'fk_grade_mean': statistics.mean(fk) if fk else None,
                             'mostly_bullets': sum(v['bullets'] for v in vals), 'has_latex': sum(v['latex'] for v in vals),
                             'hit_cap': sum(v['finish'] == 'length' for v in vals)}
            if vals[0]['correct'] is not None:
                summary[cond]['numeric_correct'] = sum(bool(v['correct']) for v in vals)
        report[mode] = {'summary': summary, 'edit': lib.get(mode).summary(), 'rows': rows}
        print(mode, json.dumps(summary), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
