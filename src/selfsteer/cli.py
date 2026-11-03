"""selfsteer serve | chat"""
from __future__ import annotations

import argparse
import random
import sys


def chat(args):
    from .library import Library
    from .model import Engine
    from .session import Session

    colors = {'lazy': '\033[96m', 'detailed': '\033[95m'}
    reset = '\033[0m'

    def on_event(e):
        if e['type'] == 'token':
            c = colors.get(e['modes'][0], '\033[93m') if e['modes'] else ''
            sys.stdout.write(f"{c}{e['text']}{reset if c else ''}")
            sys.stdout.flush()
        elif e['type'] == 'tool_call':
            print(f"\n  [qwen -> {e['name']}({e.get('arguments')})]")
        elif e['type'] == 'steer':
            print(f"  [edit live: {', '.join(e['modes'])}]")
        elif e['type'] == 'reprefill':
            print(f"  [re-encoded {e['tokens']} tokens without edits in {e['seconds']}s]")

    engine = Engine.load(args.model) if args.model else Engine.load()
    library = Library(engine).load_dir()
    session = Session(engine, library, use_tools=not args.no_tools, cache_policy=args.cache_policy,
                      on_event=on_event)
    print(f"Modes: {', '.join(library.names())}. Commands: /mode NAME [answer|conversation], /reset, /quit")
    while True:
        try:
            line = input('\nyou> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == '/quit':
            break
        if line == '/reset':
            session = Session(engine, library, use_tools=not args.no_tools,
                              cache_policy=args.cache_policy, on_event=on_event)
            continue
        if line.startswith('/mode'):
            parts = line.split()
            try:
                print(session.set_mode(parts[1] if len(parts) > 1 else 'normal',
                                       parts[2] if len(parts) > 2 else 'answer', by='user'))
            except KeyError as exc:
                print(exc.args[0])
            continue
        print('qwen> ', end='')
        turn = session.send(line, seed=random.randrange(2**31))
        print(f"\n  [{len(turn.reply.split())} words, {turn.generated_tokens} tokens, {turn.seconds}s"
              f"{', modes: ' + ', '.join(turn.modes_used) if turn.modes_used else ''}]")


def main(argv=None):
    ap = argparse.ArgumentParser(prog='selfsteer')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve', help='model + MCP endpoint + web chat')
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=8765)
    s.add_argument('--model', default=None)
    s.add_argument('--cache-policy', default='keep', choices=['keep', 'reprefill'])
    s.add_argument('--budget', type=int, default=10, help='forward passes for weight-space work')
    c = sub.add_parser('chat', help='terminal chat')
    c.add_argument('--model', default=None)
    c.add_argument('--no-tools', action='store_true')
    c.add_argument('--cache-policy', default='keep', choices=['keep', 'reprefill'])
    args = ap.parse_args(argv)
    if args.cmd == 'serve':
        from .server import serve
        serve(args.host, args.port, args.model, args.cache_policy, args.budget)
    else:
        chat(args)


if __name__ == '__main__':
    main()
