"""Controller client for the remote worker: submit a tool call, poll until done, print or save the result.

  python3 remote/call.py status
  python3 remote/call.py py '{"code": "print(W(3).shape)"}'
  python3 remote/call.py generate @args.json --out runs/remote/gen1.json
"""
import json
from pathlib import Path
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
cfg = json.loads((REPO / 'runs' / 'remote.json').read_text())
H = {'Authorization': f"Bearer {cfg['token']}", 'Content-Type': 'application/json',
     'User-Agent': 'selfsteer-controller/0.1'}


def req(path, data=None):
    r = urllib.request.Request(cfg['url'] + path, data=json.dumps(data).encode() if data is not None else None,
                               headers=H, method='POST' if data is not None else 'GET')
    return json.loads(urllib.request.urlopen(r, timeout=90).read())


def submit(tool, args, tries=30, wait=10):
    # Retry only when the request cannot have reached the worker (name lookup or connection failed),
    # so a budgeted call is never submitted twice.
    import socket
    import urllib.error
    for i in range(tries):
        try:
            return req('/call', {'tool': tool, 'args': args})['job']
        except urllib.error.URLError as exc:
            if not isinstance(exc.reason, (socket.gaierror, ConnectionRefusedError)) or i == tries - 1:
                raise
            print('submit retry', exc, file=sys.stderr)
            time.sleep(wait)


def call(tool, args, poll=5):
    jid = submit(tool, args)
    while True:
        try:
            job = req(f'/job/{jid}')
        except Exception as exc:  # transient tunnel hiccup
            print('poll error', exc, file=sys.stderr)
            time.sleep(poll)
            continue
        if job['status'] in ('done', 'error'):
            return job
        time.sleep(poll)


if __name__ == '__main__':
    tool = sys.argv[1]
    raw = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else '{}'
    args = json.loads(Path(raw[1:]).read_text()) if raw.startswith('@') else json.loads(raw)
    out = None
    if '--out' in sys.argv:
        out = Path(sys.argv[sys.argv.index('--out') + 1])
    t = time.time()
    job = call(tool, args)
    job['client_seconds'] = round(time.time() - t, 1)
    log = REPO / 'runs' / 'remote' / 'calls.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write(json.dumps(job) + '\n')
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(job, indent=1))
    if job['status'] == 'error':
        print(job['error'])
        sys.exit(1)
    text = json.dumps(job['result'], indent=1, ensure_ascii=False)
    print(text if len(text) < 6000 or not out else text[:6000] + '\n...[truncated; full result in --out]')
