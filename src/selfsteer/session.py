"""A conversation in which Qwen (or a controller) can switch edits between tokens.

The session keeps an append-only token ledger: exactly the tokens Qwen has
seen or written, including its own tool calls. The cache always holds
ledger[:runner.length]. New turns append rendered suffixes rather than
re-rendering the whole chat, so the cache stays valid across turns.

Cache policies after an edit has written steered states into the cache:
- keep: leave them (cheap; the past stays as it was computed).
- reprefill: before the next user turn, recompute the whole ledger with no
  edits, so the model's memory of the conversation is the unedited model's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
import time

import mlx.core as mx

from . import tools as tool_lib
from .decode import Runner, Sampling, make_processors, make_sampler, sample_loop
from .library import NORMAL

_session_ids = itertools.count(1)

ASSISTANT_OPEN = '<|im_start|>assistant\n<think>\n\n</think>\n\n'


@dataclass
class Armed:
    name: str
    scope: str = 'answer'  # 'answer' | 'conversation'
    gain: float = 1.0
    by: str = 'qwen'  # 'qwen' | 'controller' | 'user'


@dataclass
class TurnResult:
    user: str
    reply: str
    segments: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    modes_used: list = field(default_factory=list)
    generated_tokens: int = 0
    finish: str = ''
    seconds: float = 0.0
    reprefill_seconds: float | None = None
    seed: int | None = None

    def public(self):
        return {k: v for k, v in self.__dict__.items()}


class Session:
    def __init__(self, engine, library, *, use_tools=True, allow_requests=False, system=None,
                 cache_policy='keep', sampling=None, board=None, pump=None, on_event=None,
                 max_tool_calls=3, request_timeout=120.0, name=None):
        if cache_policy not in ('keep', 'reprefill'):
            raise ValueError('cache_policy must be keep or reprefill')
        self.engine, self.library = engine, library
        self.tok = engine.tokenizer
        self.use_tools = use_tools
        self.allow_requests = allow_requests and board is not None
        self.extra_system = system
        self.cache_policy = cache_policy
        self.sampling = sampling or Sampling()
        self.board, self.pump = board, pump
        self.on_event = on_event or (lambda e: None)
        self.max_tool_calls = max_tool_calls
        self.request_timeout = request_timeout
        self.name = name or f's{next(_session_ids)}'
        self.runner = Runner(engine)
        self.ledger: list[int] = []
        self.pending: list[int] = []
        self.armed: list[Armed] = []
        self.dirty = False
        self.turns: list[TurnResult] = []
        self.log: list[dict] = []
        self.eos = engine.eos_ids()
        self.im_end = self.tok.convert_tokens_to_ids('<|im_end|>')
        self.tool_end = self.tok.convert_tokens_to_ids('</tool_call>')
        self.tool_start = self.tok.convert_tokens_to_ids('<tool_call>')

    # -------------------------------------------------------------- helpers
    def _encode(self, text):
        return list(self.tok.encode(text, add_special_tokens=False))

    def _emit(self, kind, **data):
        event = {'type': kind, 'session': self.name, 't': time.time(), **data}
        self.log.append(event)
        self.on_event(event)

    def _first_prompt(self, user):
        messages = []
        system = None
        if self.use_tools:
            system = tool_lib.system_prompt(self.library, self.allow_requests, self.extra_system)
        elif self.extra_system:
            system = self.extra_system
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': user})
        kwargs = dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if self.use_tools:
            kwargs['tools'] = tool_lib.tool_schemas(self.allow_requests)
        tokens = self.tok.apply_chat_template(messages, **kwargs)
        return list(tokens.tolist() if hasattr(tokens, 'tolist') else tokens)

    def _feed_pending_to_ledger(self, pending):
        self.ledger.extend(pending)

    # -------------------------------------------------------------- modes
    def set_mode(self, name, scope='answer', gain=1.0, by='controller'):
        if name in NORMAL:
            self.armed = []
            self._emit('mode', action='clear', by=by)
            return 'Back to normal: no edit.'
        edit = self.library.get(name)
        self.armed = [Armed(name, scope, float(gain), by)]
        self._emit('mode', action='set', name=name, scope=scope, gain=gain, by=by, edit=edit.summary())
        return (f"Mode '{name}' is on for your {'next answer' if scope == 'answer' else 'answers until changed'}. "
                'Now answer the user.')

    def _arm_segment(self, start):
        """Activates armed modes from absolute position `start` (predicts the next token)."""
        actives = []
        for armed in self.armed:
            edit = self.library.get(armed.name).scaled(armed.gain)
            stop = start + edit.window if edit.window else None
            actives.append(self.engine.stack.activate(edit, start, stop, tag='segment'))
        return actives

    # -------------------------------------------------------------- tools
    def _run_tool(self, call):
        name, args = call.get('name'), call.get('arguments') or {}
        if name == 'set_mode':
            mode = str(args.get('name', '')).strip().lower()
            scope = args.get('scope') or 'answer'
            if scope not in ('answer', 'conversation'):
                scope = 'answer'
            try:
                return self.set_mode(mode, scope, by='qwen')
            except KeyError as exc:
                return str(exc.args[0])
        if name == 'request_mode' and self.allow_requests:
            description = str(args.get('description', '')).strip()
            req = self.board.post(description, self.name)
            self._emit('request', id=req.id, description=description)
            req = self.board.wait(req, self.request_timeout, self.pump)
            self._emit('request_done', **req.public())
            if req.status == 'done':
                edit = self.library.get(req.mode)
                return (f"Claude built mode '{req.mode}': {edit.description} "
                        f"Call set_mode with name '{req.mode}' to use it.")
            return 'No new mode could be built in time. Answer as well as you can.'
        return f'Unknown tool {name!r}.'

    # -------------------------------------------------------------- turns
    def reprefill(self):
        started = time.perf_counter()
        saved = self.engine.stack.actives
        self.engine.stack.actives = []
        try:
            self.runner.reset()
            self.runner.prefill(self.ledger, self.sampling.prefill_step_size)
        finally:
            self.engine.stack.actives = saved
        self.dirty = False
        return time.perf_counter() - started

    def send(self, user, seed=None):
        s = self.sampling
        started = time.perf_counter()
        result = TurnResult(user=user, reply='', seed=seed)
        if self.cache_policy == 'reprefill' and self.dirty:
            # Everything except the pending tail is recomputed with no edits.
            result.reprefill_seconds = round(self.reprefill(), 3)
            self._emit('reprefill', seconds=result.reprefill_seconds, tokens=len(self.ledger))
        if not self.ledger and not self.pending:
            self.pending = self._first_prompt(user)
        else:
            if not self.pending or self.pending[-1] != self.im_end:
                self.pending.append(self.im_end)
            self.pending += self._encode(f'\n<|im_start|>user\n{user}<|im_end|>\n{ASSISTANT_OPEN}')
        self._emit('user', text=user)
        if seed is not None:
            mx.random.seed(seed)
        sampler, processors = make_sampler(s), make_processors(s)
        history: list[int] = []
        stop_ids = set(self.eos) | ({self.tool_end} if self.use_tools else set())
        visible = []
        tool_calls = 0
        detok = self.tok.detokenizer
        while True:
            start = self.runner.length + len(self.pending) - 1
            actives = self._arm_segment(start)
            names = [a.edit.name for a in actives]
            if names:
                self._emit('steer', modes=names, start=start,
                           windows={a.edit.name: a.edit.window for a in actives})
            detok.reset()
            hidden = {'tool': False}

            def on_token(token, position, hidden=hidden):
                # Tool-call markup and stop tokens are shown as events, not text.
                if token == self.tool_start:
                    hidden['tool'] = True
                if hidden['tool'] or token in stop_ids:
                    return
                detok.add_token(token)
                steered = self.engine.stack.names_at(position)
                self._emit('token', text=detok.last_segment, modes=steered)

            fed = list(self.pending)
            budget = s.max_tokens - len(history)
            tokens, reason, timings = sample_loop(
                self.runner, self.pending, sampler=sampler, processors=processors, history=history,
                max_tokens=budget, stop_ids=stop_ids, prefill_step=s.prefill_step_size, on_token=on_token)
            self._feed_pending_to_ledger(fed + tokens[:-1])
            steered = sum(a.steered_positions for a in actives)
            if steered:
                self.dirty = True
            self.engine.stack.deactivate('segment')
            text = self.tok.decode(tokens, skip_special_tokens=True)
            seg = {'tokens': len(tokens), 'reason': reason, 'modes': names,
                   'steered_positions': steered, 'text': text, **{k: round(v, 3) for k, v in timings.items()}}
            result.segments.append(seg)
            result.modes_used += [n for n in names if n not in result.modes_used]
            last = tokens[-1] if tokens else None
            if reason == 'stop' and last == self.tool_end and tool_calls < self.max_tool_calls:
                tool_calls += 1
                body = self.tok.decode(tokens, skip_special_tokens=False)
                pre = body[:body.find('<tool_call>')].strip() if '<tool_call>' in body else ''
                if pre:
                    visible.append(pre)
                try:
                    call = tool_lib.parse(body, tool_lib.tool_schemas(self.allow_requests))
                    self._emit('tool_call', name=call.get('name'), arguments=call.get('arguments'))
                    output = self._run_tool(call)
                except Exception as exc:  # malformed call: tell Qwen, keep going
                    call, output = {'error': str(exc)}, f'Tool call failed: {exc}'
                result.tool_calls.append({'call': call, 'result': output})
                self._emit('tool_result', text=output)
                self.pending = [last] + self._encode(
                    f'<|im_end|>\n<|im_start|>user\n<tool_response>\n{output}\n</tool_response>'
                    f'<|im_end|>\n{ASSISTANT_OPEN}')
                continue
            visible.append(text.strip() if reason == 'stop' and last == self.tool_end else text)
            # Close the assistant turn. The stop token (or a synthetic <|im_end|>) is pending.
            if last in self.eos:
                self.pending = [last]
            else:
                self.pending = ([last] if last is not None else []) + [self.im_end]
            result.finish = 'stop' if reason == 'stop' else 'length'
            break
        result.reply = '\n\n'.join(v for v in visible if v).strip()
        result.generated_tokens = len(history)
        result.seconds = round(time.perf_counter() - started, 3)
        # 'answer'-scoped modes end with the turn.
        self.armed = [a for a in self.armed if a.scope == 'conversation']
        self.turns.append(result)
        self._emit('done', reply=result.reply, modes_used=result.modes_used,
                   generated_tokens=result.generated_tokens, seconds=result.seconds)
        return result

    def fork(self, cache_policy=None, on_event=None):
        """An independent copy of this conversation (cache included)."""
        twin = Session(self.engine, self.library, use_tools=self.use_tools, allow_requests=self.allow_requests,
                       system=self.extra_system, cache_policy=cache_policy or self.cache_policy,
                       sampling=self.sampling, board=self.board, pump=self.pump, on_event=on_event,
                       max_tool_calls=self.max_tool_calls, request_timeout=self.request_timeout)
        twin.runner.restore(self.runner.snapshot())
        twin.ledger, twin.pending = list(self.ledger), list(self.pending)
        twin.armed = [Armed(**a.__dict__) for a in self.armed]
        twin.dirty, twin.turns = self.dirty, list(self.turns)
        return twin

    def transcript(self):
        return [t.public() for t in self.turns]

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump({'session': self.name, 'cache_policy': self.cache_policy,
                       'turns': self.transcript(), 'ledger_tokens': len(self.ledger)}, f, indent=1)
