"""Requests from Qwen to an outside controller (Claude via MCP).

Qwen's request_mode tool posts a request here and its turn waits. Claude
long-polls with wait_for_request, builds a mode, and calls fulfill_request.
While Qwen waits, the model thread keeps running other jobs (so Claude's
create_mode / try_mode calls can use the model), via the `pump` callback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import threading
import time

_ids = itertools.count(1)


@dataclass
class Request:
    description: str
    session: str
    id: int = field(default_factory=lambda: next(_ids))
    created: float = field(default_factory=time.time)
    status: str = 'open'  # open -> claimed -> done | failed | expired
    mode: str | None = None
    note: str = ''
    done: threading.Event = field(default_factory=threading.Event)

    def public(self):
        return {'id': self.id, 'description': self.description, 'session': self.session,
                'status': self.status, 'mode': self.mode, 'note': self.note,
                'age_s': round(time.time() - self.created, 1)}


class RequestBoard:
    def __init__(self):
        self.requests: dict[int, Request] = {}
        self.cond = threading.Condition()

    def post(self, description, session):
        with self.cond:
            req = Request(description, session)
            self.requests[req.id] = req
            self.cond.notify_all()
            return req

    def claim_next(self, timeout):
        """Blocks up to timeout for an open request; marks it claimed."""
        deadline = time.time() + timeout
        with self.cond:
            while True:
                for req in self.requests.values():
                    if req.status == 'open':
                        req.status = 'claimed'
                        return req
                left = deadline - time.time()
                if left <= 0:
                    return None
                self.cond.wait(left)

    def fulfill(self, request_id, mode=None, note=''):
        with self.cond:
            req = self.requests[request_id]
            req.mode, req.note = mode, note
            req.status = 'done' if mode else 'failed'
            req.done.set()
            return req

    def wait(self, req, timeout, pump=None):
        """Waits for a request to be fulfilled, running `pump()` between checks."""
        deadline = time.time() + timeout
        while not req.done.is_set():
            if time.time() >= deadline:
                req.status = 'expired'
                return req
            if pump:
                pump(0.05)
            else:
                req.done.wait(0.05)
        return req
