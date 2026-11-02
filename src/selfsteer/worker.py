"""One thread owns the model. Everything that touches MLX runs here."""
from __future__ import annotations

from concurrent.futures import Future
import queue
import threading


class ModelWorker:
    def __init__(self):
        self.jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name='model-worker', daemon=True)
        self.thread.start()

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        self.jobs.put((future, fn, args, kwargs))
        return future

    def call(self, fn, *args, **kwargs):
        """Runs fn on the worker and waits; runs inline if already on the worker."""
        if threading.current_thread() is self.thread:
            return fn(*args, **kwargs)
        return self.submit(fn, *args, **kwargs).result()

    def pump(self, timeout=0.05):
        """Runs at most one queued job. Only call from the worker thread
        (used while a Qwen turn waits on its controller)."""
        try:
            job = self.jobs.get(timeout=timeout)
        except queue.Empty:
            return False
        self._run(job)
        return True

    def _run(self, job):
        future, fn, args, kwargs = job
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # report to the caller, keep the worker alive
            future.set_exception(exc)

    def _loop(self):
        while True:
            self._run(self.jobs.get())
