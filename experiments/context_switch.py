"""E3: does switching an edit mid-conversation break Qwen's memory of earlier turns?

24 three-turn conversations. Turn 1 gives four facts; turn 2 is an unrelated
question answered under an edit (armed by the controller for that answer only);
turn 3 asks to recall two facts with no edit. After turn 2 the conversation is
forked: `keep` leaves the steered cache, `reprefill` re-encodes it with no edits.

  uv run python experiments/context_switch.py --out runs/context.json
"""
import argparse
import json
from pathlib import Path
import random

import mlx.core as mx

from selfsteer.decode import Runner
from selfsteer.edits import Edit
from selfsteer.library import Library
from selfsteer.model import Engine
from selfsteer.session import ASSISTANT_OPEN, Session

MANAGERS = ['Dana Whitfield', 'Rohan Mehta', 'Priya Natarajan', 'Tomás Ortega', 'Mei Lin Zhao', 'Kwame Asante',
            'Sofia Lindqvist', 'Hiroshi Tanaka']
DOGS = ['Biscuit', 'Juniper', 'Pickles', 'Moose', 'Saffron', 'Waffles', 'Tango', 'Clover']
TURN2 = [
    'Write a Python function that returns the second largest number in a list.',
    'How do I undo the last git commit but keep my changes?',
    'Explain what a hash table is.',
    'What is 15% of 240?',
    'Give me a shell command to count lines in all .py files in a folder.',
    'What does HTTP status 404 mean?',
    'Write a SQL query that returns the 5 most recent orders.',
    'How do I center a div with CSS?',
    'What is the difference between a list and a tuple in Python?',
    'Convert 72 degrees Fahrenheit to Celsius.',
    'Explain recursion in simple terms.',
    'How do I make a Python virtual environment?',
    'What is a race condition?',
    'Write a regex that matches a US ZIP code.',
    'What does the chmod 755 command do?',
    'How many minutes are in 3.75 hours?',
    'Explain what an API is.',
    'Write a JavaScript function that debounces another function.',
    'What is the time complexity of binary search?',
    'How do I rename a git branch?',
    'What is the difference between TCP and UDP?',
    'Summarize what a mortgage is in two sentences.',
    'Write a Python one-liner to reverse a string.',
    'What is 37 times 24?',
]


def conversations(n=24):
    out = []
    for i in range(n):
        rng = random.Random(1000 + i)
        facts = {'locker code': str(rng.randrange(1000, 10000)), 'manager': rng.choice(MANAGERS),
                 'gate': f"{rng.choice('ABCDE')}{rng.randrange(2, 40)}", "dog's name": rng.choice(DOGS)}
        asked = rng.sample(sorted(facts), 2)
        turn1 = (f"Please remember these details for later: my locker code is {facts['locker code']}, my manager is "
                 f"{facts['manager']}, my flight leaves from gate {facts['gate']}, and my dog's name is "
                 f"{facts[chr(100) + 'og' + chr(39) + 's name']}. Just reply 'Noted.'")
        question = {'locker code': "what's my locker code", 'manager': "who's my manager",
                    'gate': 'which gate does my flight leave from', "dog's name": "what's my dog's name"}
        turn3 = f'Quick check: {question[asked[0]]}, and {question[asked[1]]}?'
        out.append({'i': i, 'facts': facts, 'asked': asked, 'turns': [turn1, TURN2[i], turn3], 'seed': 1000 + i})
    return out


def recalled(reply, conv):
    text = reply.lower()
    return all(conv['facts'][k].lower() in text for k in conv['asked'])


def kl_rows(engine, keep_session, clean_session, suffix, answer_tokens):
    """Mean KL(clean || keep) over the turn-3 prediction positions."""
    seq = suffix + answer_tokens
    out = {}
    logits = {}
    for label, sess in (('keep', keep_session), ('clean', clean_session)):
        runner = Runner(engine)
        runner.restore(sess.runner.snapshot())
        engine.stack.set_span(runner.length, len(seq))
        logits[label] = engine.model(mx.array(seq)[None], cache=runner.cache)[0].astype(mx.float32)
    start = len(suffix) - 1
    lp_c = logits['clean'][start:] - mx.logsumexp(logits['clean'][start:], axis=-1, keepdims=True)
    lp_k = logits['keep'][start:] - mx.logsumexp(logits['keep'][start:], axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(lp_c) * (lp_c - lp_k), axis=-1)
    mx.eval(kl)
    vals = kl.tolist()
    out['kl_first'] = vals[0]
    out['kl_mean'] = sum(vals) / len(vals)
    out['kl_max'] = max(vals)
    out['positions'] = len(vals)
    out['argmax_agree'] = float(mx.mean(mx.argmax(lp_c, -1) == mx.argmax(lp_k, -1)).item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=24)
    args = ap.parse_args()
    engine = Engine.load()
    lib = Library(engine).load_dir()
    lazy = lib.get('lazy')
    lib.add(Edit('full', 'add', dict(lazy.vectors), window=None, description='lazy vector, whole answer'))
    results = []
    for conv in conversations(args.n):
        t1, t2, t3 = conv['turns']
        row = {'i': conv['i'], 'facts': conv['facts'], 'asked': conv['asked'], 'turn2': t2, 'runs': {}}
        for edit in ('none', 'lazy', 'full'):
            s = Session(engine, lib, use_tools=False)
            s.send(t1, seed=conv['seed'])
            if edit != 'none':
                s.set_mode(edit, 'answer', by='controller')
            r2 = s.send(t2, seed=conv['seed'] + 1)
            branches = [('keep', s)] if edit == 'none' else [('keep', s.fork('keep')), ('reprefill', s.fork('reprefill'))]
            done = {}
            for policy, b in branches:
                r3 = b.send(t3, seed=conv['seed'] + 2)
                done[policy] = (b, r3)
                key = 'none' if edit == 'none' else f'{edit}+{policy}'
                row['runs'][key] = {'turn2_words': len(r2.reply.split()),
                                    'turn2_steered_positions': sum(x['steered_positions'] for x in r2.segments),
                                    'turn3_reply': r3.reply, 'recalled': recalled(r3.reply, conv),
                                    'reprefill_seconds': r3.reprefill_seconds}
            if edit != 'none':
                # KL between the steered (keep) cache and a clean cache, on the clean branch's answer.
                clean_b, clean_r3 = done['reprefill']
                keep_state = s.fork('keep')
                clean_state = s.fork('keep')
                clean_state.reprefill()
                suffix = [keep_state.pending[0]] if keep_state.pending else []
                suffix += keep_state._encode(f'\n<|im_start|>user\n{t3}<|im_end|>\n{ASSISTANT_OPEN}')
                answer = clean_b.ledger[len(clean_b.ledger) - (clean_r3.generated_tokens - 1):][:32]
                row['runs'][f'{edit}+keep']['kl'] = kl_rows(engine, keep_state, clean_state, suffix, answer)
        results.append(row)
        print(json.dumps({'i': row['i'], **{k: (v['recalled'], round(v.get('kl', {}).get('kl_mean', 0), 4))
                                            for k, v in row['runs'].items()}}), flush=True)
    summary = {}
    for key in results[0]['runs']:
        runs = [r['runs'][key] for r in results]
        summary[key] = {'recalled': sum(x['recalled'] for x in runs), 'n': len(runs),
                        'turn2_words_mean': sum(x['turn2_words'] for x in runs) / len(runs)}
        kls = [x['kl'] for x in runs if 'kl' in x]
        if kls:
            summary[key].update({m: sum(k[m] for k in kls) / len(kls) for m in ('kl_first', 'kl_mean', 'kl_max', 'argmax_agree')})
        rp = [x['reprefill_seconds'] for x in runs if x.get('reprefill_seconds')]
        if rp:
            summary[key]['reprefill_seconds_mean'] = sum(rp) / len(rp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'summary': summary, 'conversations': results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
