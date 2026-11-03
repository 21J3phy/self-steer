"""Build the Kaggle notebook that runs the remote worker behind a token-protected tunnel.

  uv run python remote/build_kaggle.py            # writes kaggle/ (notebook + metadata) and runs/remote.json (token)
  kaggle kernels push -p remote/kaggle --accelerator NvidiaTeslaT4 -t 21600
  kaggle kernels logs -f niravsurabhi/self-steer-remote     # prints TUNNEL_URL when ready
"""
import json
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'kaggle'
OUT.mkdir(exist_ok=True)
token = secrets.token_hex(24)
budget = 10
lazy = json.loads((REPO / 'presets' / 'lazy.json').read_text())
payload = {
    'worker.py': (ROOT / 'worker.py').read_text(),
    'presets/lazy.json': json.dumps(lazy),
    'data/heldout_e5.json': (ROOT / 'heldout_e5.json').read_text(),
    'data/practice_math.json': (ROOT / 'practice_math.json').read_text(),
    'data/fair_final96.json': (ROOT / 'fair_final96.json').read_text(),
    'data/fair2_final96.json': (ROOT / 'fair2_final96.json').read_text(),
}


def cell(kind, src):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


setup = '''import os, sys, subprocess, json, time, re, shutil
from pathlib import Path
os.environ.update(HF_HOME='/tmp/hf', HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                  TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'transformers==5.5.4', 'accelerate==1.15.0'], check=True)
subprocess.run(['nvidia-smi'], check=True)
code = Path('/kaggle/working/selfsteer'); code.mkdir(exist_ok=True)
'''
materialize = ('payload = json.loads(' + repr(json.dumps(payload)) + ')\n'
               'for name, content in payload.items():\n'
               '    p = code / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)\n'
               'print(sorted(payload))\n')
launch = f'''env = dict(os.environ, SELFSTEER_TOKEN={token!r}, SELFSTEER_BUDGET='{budget}', SELFSTEER_WORK='/kaggle/working/work')
log = open('/kaggle/working/worker.log', 'w')
wp = subprocess.Popen([sys.executable, '-u', str(code / 'worker.py'), str(code / 'presets/lazy.json')], cwd=code,
                      env=env, stdout=log, stderr=subprocess.STDOUT)
t0 = time.time()
while True:
    text = Path('/kaggle/working/worker.log').read_text()
    if 'WORKER READY' in text:
        print(text[-2000:], flush=True); break
    if wp.poll() is not None:
        print(text[-5000:], flush=True); raise RuntimeError('worker died')
    time.sleep(10)
    print(f'loading model... {{time.time()-t0:.0f}}s', flush=True)
subprocess.run(['wget', '-q', '-O', '/tmp/cloudflared',
                'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'], check=True)
os.chmod('/tmp/cloudflared', 0o755)
tp = subprocess.Popen(['/tmp/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8000'],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in tp.stdout:
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', line)
    if m:
        url = m.group(0); break
print('TUNNEL_URL', url, flush=True)
'''
wait = '''last = time.time()
while not Path('/kaggle/working/work/SHUTDOWN').exists():
    if wp.poll() is not None:
        print('worker exited', wp.returncode, flush=True); break
    if time.time() - last > 120:
        jobs = Path('/kaggle/working/work/trace.jsonl')
        n = len(jobs.read_text().splitlines()) if jobs.exists() else 0
        print(f'alive {time.strftime("%H:%M:%S")} jobs_done={n}', flush=True); last = time.time()
    time.sleep(5)
print('shutting down', flush=True)
tp.terminate(); wp.terminate()
'''
nb = {'cells': [cell('markdown', '# self-steer remote worker\\n\\nQwen3.5-4B with live neuron/weight edits and weight-space instruments, behind a token-protected tunnel.'),
                cell('code', setup), cell('code', materialize), cell('code', launch), cell('code', wait)],
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
      'nbformat': 4, 'nbformat_minor': 5}
(OUT / 'self-steer-remote.ipynb').write_text(json.dumps(nb, indent=1))
(OUT / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'niravsurabhi/self-steer-remote', 'title': 'self-steer remote', 'code_file': 'self-steer-remote.ipynb',
    'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
    'machine_shape': 'NvidiaTeslaT4', 'kernel_sources': [], 'dataset_sources': [], 'competition_sources': [],
    'model_sources': []}, indent=2))
(REPO / 'runs').mkdir(exist_ok=True)
(REPO / 'runs' / 'remote.json').write_text(json.dumps({'token': token, 'kernel': 'niravsurabhi/self-steer-remote'}))
print('built', OUT, 'budget', budget)
