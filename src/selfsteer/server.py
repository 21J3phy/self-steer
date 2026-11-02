"""One process: the model, an MCP endpoint for the controller, and a web chat.

  selfsteer serve            # http://127.0.0.1:8765 (chat UI), /mcp (MCP)

Claude Code connects with:
  claude mcp add --transport http self-steer http://127.0.0.1:8765/mcp
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import random
import time

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

from .controller import RequestBoard
from .decode import Sampling
from .library import Library
from .model import Engine
from .probe import build_mode
from .session import Session
from .worker import ModelWorker

WEB = Path(__file__).parent / 'web' / 'index.html'

INSTRUCTIONS = """You are the controller for a local Qwen3.5-4B whose neurons can be edited live.
Qwen can switch its own modes (set_mode) and can ask you for new ones (request_mode).
Loop: wait_for_request -> design 6-12 contrastive examples (same prompt, positive answer in the
requested style, negative plain answer) -> create_mode -> try_mode on 2-3 prompts and adjust
strength (typical add strength 0.3-1.5) -> fulfill_request with the mode name.
You can also set_mode on the live chat yourself, or chat with Qwen."""


class Hub:
    def __init__(self, engine, library, cache_policy='keep'):
        self.engine, self.library = engine, library
        self.worker = ModelWorker()
        self.board = RequestBoard()
        self.subscribers: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.cache_policy = cache_policy
        self.busy = False
        self.session = self.new_session(cache_policy)

    def new_session(self, cache_policy=None):
        self.cache_policy = cache_policy or self.cache_policy
        self.session = Session(self.engine, self.library, use_tools=True, allow_requests=True,
                               board=self.board, pump=self.worker.pump, on_event=self.broadcast,
                               cache_policy=self.cache_policy)
        self.broadcast({'type': 'reset', 'cache_policy': self.cache_policy, 't': time.time()})
        return self.session

    def broadcast(self, event):
        if self.loop is None:
            return
        for q in list(self.subscribers):
            self.loop.call_soon_threadsafe(q.put_nowait, event)

    def state(self):
        s = self.session
        return {'model': self.engine.describe(), 'modes': [self.library.get(n).summary() for n in self.library.names()],
                'session': {'name': s.name, 'turns': len(s.turns), 'cache_policy': s.cache_policy,
                            'armed': [a.__dict__ for a in s.armed], 'ledger_tokens': len(s.ledger), 'busy': self.busy},
                'requests': [r.public() for r in self.board.requests.values()]}

    def chat(self, message):
        self.busy = True
        try:
            turn = self.session.send(message, seed=random.randrange(2**31))
            return turn.public()
        finally:
            self.busy = False

    def try_mode(self, name, prompts, gain, max_tokens, seed):
        out = []
        for prompt in prompts:
            row = {'prompt': prompt}
            for label, mode in (('normal', None), (name, name)):
                s = Session(self.engine, self.library, use_tools=False,
                            sampling=Sampling(max_tokens=max_tokens))
                if mode:
                    s.set_mode(mode, 'answer', gain=gain)
                turn = s.send(prompt, seed=seed)
                row[label] = {'text': turn.reply, 'words': len(turn.reply.split()), 'finish': turn.finish}
            out.append(row)
        return out


def build_server(hub: Hub) -> MCPServer:
    mcp = MCPServer('self-steer', instructions=INSTRUCTIONS)

    async def on_worker(fn, *args, **kwargs):
        hub.loop = hub.loop or asyncio.get_running_loop()
        return await asyncio.to_thread(hub.worker.call, fn, *args, **kwargs)

    @mcp.tool()
    async def status() -> dict:
        """Model, live session, modes and open requests."""
        return await on_worker(hub.state)

    @mcp.tool()
    async def list_modes() -> list[dict]:
        """Every mode in the library with its layers, norms, window and author."""
        return [hub.library.get(n).summary() for n in hub.library.names()]

    @mcp.tool()
    async def create_mode(name: str, description: str, examples: list[dict], layers: list[int] | None = None,
                          strength: float = 0.6, kind: str = 'add', window: int | None = None) -> dict:
        """Build a mode from contrastive examples, each {"prompt", "positive", "negative"}.
        positive = an answer in the wanted style, negative = a plain answer to the same prompt.
        kind "add" steers toward the style; kind "ablate" removes the direction from the chosen
        layers' MLP weights (a rank-1 weight edit). Default layers: 8-16. window: steer only the
        first N answer positions (None = whole answer)."""
        _, stats = await on_worker(build_mode, hub.engine, hub.library, name=name, description=description,
                                   examples=examples, layers=layers, strength=strength, kind=kind,
                                   window=window, author='claude')
        hub.broadcast({'type': 'library', 'action': 'created', 'mode': stats, 't': time.time()})
        return stats

    @mcp.tool()
    async def try_mode(name: str, prompts: list[str], gain: float = 1.0, max_tokens: int = 200,
                       seed: int = 0) -> list[dict]:
        """Side-by-side answers, normal vs. the mode, on fresh conversations (same seed)."""
        return await on_worker(hub.try_mode, name, prompts, gain, max_tokens, seed)

    @mcp.tool()
    async def set_mode(name: str, scope: str = 'answer', gain: float = 1.0) -> str:
        """Switch a mode on the live chat as the controller ("normal" clears it)."""
        return await on_worker(hub.session.set_mode, name, scope, gain, 'controller')

    @mcp.tool()
    async def chat(message: str) -> dict:
        """Send a user message to Qwen in the live chat and get its reply, tool calls and modes used."""
        return await on_worker(hub.chat, message)

    @mcp.tool()
    async def new_conversation(cache_policy: str = 'keep') -> dict:
        """Start a fresh chat. cache_policy: keep | reprefill."""
        await on_worker(hub.new_session, cache_policy)
        return await on_worker(hub.state)

    @mcp.tool()
    async def wait_for_request(timeout_s: float = 50.0) -> dict | None:
        """Block until Qwen asks for a new mode (request_mode), or the timeout passes."""
        hub.loop = hub.loop or asyncio.get_running_loop()
        req = await asyncio.to_thread(hub.board.claim_next, timeout_s)
        if req:
            hub.broadcast({'type': 'request_claimed', 'id': req.id, 't': time.time()})
        return req.public() if req else None

    @mcp.tool()
    async def fulfill_request(request_id: int, mode: str | None = None, note: str = '') -> dict:
        """Answer Qwen's request with the name of the mode you built (or no mode, with a note)."""
        if mode is not None:
            hub.library.get(mode)
        return hub.board.fulfill(request_id, mode, note).public()

    @mcp.tool()
    async def save_mode(name: str) -> str:
        """Persist a runtime mode to presets/<name>.json."""
        return str(await on_worker(hub.library.save, name))

    @mcp.tool()
    async def transcript(last_n: int = 5) -> list[dict]:
        """The last turns of the live chat."""
        return hub.session.transcript()[-last_n:]

    # ------------------------------------------------------------------ web
    @mcp.custom_route('/', methods=['GET'])
    async def index(request: Request):
        return HTMLResponse(WEB.read_text())

    @mcp.custom_route('/api/state', methods=['GET'])
    async def api_state(request: Request):
        hub.loop = hub.loop or asyncio.get_running_loop()
        return JSONResponse(hub.state())

    @mcp.custom_route('/api/events', methods=['GET'])
    async def api_events(request: Request):
        hub.loop = hub.loop or asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        hub.subscribers.add(q)

        async def stream():
            try:
                yield 'retry: 1000\n\n'
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15)
                        yield f'data: {json.dumps(event, default=str)}\n\n'
                    except asyncio.TimeoutError:
                        yield ': keepalive\n\n'
            finally:
                hub.subscribers.discard(q)
        return StreamingResponse(stream(), media_type='text/event-stream')

    @mcp.custom_route('/api/chat', methods=['POST'])
    async def api_chat(request: Request):
        hub.loop = hub.loop or asyncio.get_running_loop()
        body = await request.json()
        message = str(body.get('message', '')).strip()
        if not message:
            return JSONResponse({'error': 'empty message'}, status_code=400)
        if hub.busy:
            return JSONResponse({'error': 'Qwen is still answering'}, status_code=409)
        hub.busy = True  # set before the job runs so double clicks are rejected
        hub.worker.submit(hub.chat, message)
        return JSONResponse({'ok': True})

    @mcp.custom_route('/api/mode', methods=['POST'])
    async def api_mode(request: Request):
        hub.loop = hub.loop or asyncio.get_running_loop()
        body = await request.json()
        try:
            text = await on_worker(hub.session.set_mode, body.get('name', 'normal'),
                                   body.get('scope', 'answer'), float(body.get('gain', 1.0)), 'user')
        except KeyError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        return JSONResponse({'ok': True, 'text': text})

    @mcp.custom_route('/api/reset', methods=['POST'])
    async def api_reset(request: Request):
        hub.loop = hub.loop or asyncio.get_running_loop()
        body = await request.json()
        await on_worker(hub.new_session, body.get('cache_policy'))
        return JSONResponse(hub.state())

    return mcp


def serve(host='127.0.0.1', port=8765, model_path=None, cache_policy='keep'):
    import uvicorn
    engine = Engine.load(model_path) if model_path else Engine.load()
    library = Library(engine).load_dir()
    hub = Hub(engine, library, cache_policy)
    app = build_server(hub).streamable_http_app(host=host)
    print(f'self-steer: chat at http://{host}:{port}/  MCP at http://{host}:{port}/mcp', flush=True)
    uvicorn.run(app, host=host, port=port, log_level='warning')
