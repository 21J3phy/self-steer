"""Practice arithmetic prompts for the controller's forward passes (never the E5 held-out set).

  uv run python experiments/practice_math.py > runs/work/practice_math.json
"""
import json
import random

rng = random.Random(99)
out = []
for i in range(48):
    k = i % 6
    if k == 0:
        a, b = rng.randrange(3, 40), rng.randrange(3, 40)
        out.append({'prompt': f'A garden is {a} m long and {b} m wide. What is its area?', 'answer': a * b})
    elif k == 1:
        p, n = rng.randrange(2, 30), rng.randrange(3, 12)
        out.append({'prompt': f'Notebooks cost ${p} each. How much do {n} notebooks cost?', 'answer': p * n})
    elif k == 2:
        x = [rng.randrange(10, 99) for _ in range(4)]
        out.append({'prompt': f'What is the average of {x[0]}, {x[1]}, {x[2]} and {x[3]}?', 'answer': sum(x) / 4})
    elif k == 3:
        base, pct = rng.randrange(50, 500), rng.choice([5, 10, 12, 15, 20])
        out.append({'prompt': f'A rent of ${base} goes up by {pct}%. What is the new rent?', 'answer': round(base * (1 + pct / 100), 2)})
    elif k == 4:
        rate, t = rng.randrange(4, 25), rng.randrange(3, 15)
        out.append({'prompt': f'A tank fills at {rate} liters per minute. How many liters after {t} minutes?', 'answer': rate * t})
    else:
        m = rng.randrange(65, 400)
        out.append({'prompt': f'How many hours and minutes is {m} minutes?', 'answer': m // 60})
print(json.dumps(out, indent=1))
