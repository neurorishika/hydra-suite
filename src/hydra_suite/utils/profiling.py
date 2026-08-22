"""Hierarchical span profiler — the one timing primitive for the suite.

Ambient by design: call sites write ``with span(NAME):`` and the active
recorder is found through a :class:`~contextvars.ContextVar`. When no recorder
is armed, ``span()`` returns a shared frozen no-op object (~152 ns per enter
against a 7.7 ns unwrapped baseline), so instrumentation can stay in the hot
path with Debug Mode off.

Aggregate-only: totals are kept per distinct span *path*, never as sample
lists, so memory is O(distinct paths) regardless of run length.

Stacks are thread-local (they live in the ContextVar value); only the node
table is shared, under a lock. Threads that never armed the recorder record
nothing — bind them with :func:`bind_target`.
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

PRIORITY_PROCESS = 0
PRIORITY_SESSION = 1

_ACTIVE: contextvars.ContextVar["SpanRecorder | None"] = contextvars.ContextVar(
    "hydra_span_recorder", default=None
)


def deep_gpu_enabled() -> bool:
    """``HYDRA_PROFILE_GPU=1`` — the opt-in deep-GPU diagnostic pass.

    Deep mode syncs on ``gpu=True`` spans AND forces ``pipeline_depth=1`` (see
    ``pipeline._effective_depth``), so there is no producer thread whose queue
    a device-wide sync could drain. The run is explicitly NOT the production
    schedule; that is the price of device attribution.
    """
    return bool(os.environ.get("HYDRA_PROFILE_GPU"))


def _synchronize() -> None:
    """Device-wide sync. Only ever called from a gpu_sync recorder."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001 — profiling must never break a run
        logger.debug("Span profiler: device synchronize failed", exc_info=True)


class _NullSpan:
    """Shared frozen no-op returned when nothing is armed."""

    __slots__ = ()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def add_units(self, _n: float) -> None:
        pass


_NULL_SPAN = _NullSpan()


class _Node:
    """One span path's running totals."""

    __slots__ = (
        "name",
        "total_s",
        "self_s",
        "n_calls",
        "units",
        "max_s",
        "first_call_s",
        "thread",
        "children",
    )

    def __init__(self, name: str, thread: str) -> None:
        self.name = name
        self.total_s = 0.0
        self.self_s = 0.0
        self.n_calls = 0
        self.units = 0.0
        self.max_s = 0.0
        self.first_call_s: float | None = None
        self.thread = thread
        self.children: dict[str, _Node] = {}


class _Span:
    """Live span. Pops by identity, never by name, and never swallows."""

    __slots__ = ("_rec", "_node", "_stack", "_t0", "_child_s", "_sync")

    def __init__(
        self, rec: "SpanRecorder", node: _Node, stack: list, sync: bool = False
    ) -> None:
        self._rec = rec
        self._node = node
        self._stack = stack
        self._t0 = 0.0
        self._child_s = 0.0
        self._sync = sync

    def __enter__(self) -> "_Span":
        self._stack.append(self)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> bool:
        if self._sync:
            _synchronize()
        dt = time.perf_counter() - self._t0
        # Identity pop: an exception may have skipped inner __exit__ calls, so
        # pop through them. Guarded by a membership check — if `self` is NOT on
        # the stack (a leaked span already cleared at disarm, or a late __exit__
        # from a GC'd generator) an unguarded loop would drain the thread's LIVE
        # stack and orphan every open span.
        stack = self._stack
        if any(s is self for s in stack):
            while stack:
                top = stack.pop()
                if top is self:
                    break
        else:
            return False
        if stack:
            stack[-1]._child_s += dt
        node = self._node
        with self._rec._lock:
            node.total_s += dt
            node.self_s += dt - self._child_s
            node.n_calls += 1
            if dt > node.max_s:
                node.max_s = dt
            if node.first_call_s is None:
                node.first_call_s = dt
        return False

    def add_units(self, n: float) -> None:
        with self._rec._lock:
            self._node.units += float(n)


class SpanRecorder:
    """Collects a span tree. One per profiled scope."""

    def __init__(
        self, priority: int = PRIORITY_SESSION, gpu_sync: bool | None = None
    ) -> None:
        self.priority = int(priority)
        self.gpu_sync = deep_gpu_enabled() if gpu_sync is None else bool(gpu_sync)
        self._lock = threading.Lock()
        self._roots: dict[str, _Node] = {}
        # Stacks are thread-local: a single shared stack would interleave
        # pushes from concurrent threads and corrupt parentage, and the lock
        # protects counters, not stack coherence.
        self._thread_stacks: dict[int, list] = {}
        self._start = time.perf_counter()
        self._armed_once = False
        self._end: float | None = None
        # Name of the thread that armed first — the renderer's reference for
        # deciding which subtrees are "concurrent". Hardcoding "MainThread" is
        # wrong: TrackingWorker is a QThread, so on the GUI path EVERY node
        # would be flagged concurrent while headless gate runs looked fine.
        self.root_thread: str = threading.current_thread().name

    # -- lifecycle ------------------------------------------------------

    @contextmanager
    def armed(self) -> Iterator["SpanRecorder"]:
        """Arm this recorder for the duration of the block.

        Defers (yields without arming) when an equal-or-higher-priority
        recorder is already active, so a nested consumer's spans land in the
        outer tree instead of being split out of it.
        """
        active = _ACTIVE.get()
        if active is not None and active.priority >= self.priority:
            yield self
            return
        token = _ACTIVE.set(self)
        # Reset the clock at the FIRST arm, not at construction: the profiler is
        # built early (worker.py:699, before model loading) but armed much
        # later, and the root total_s is the denominator for every depth-1
        # percentage. Without this, minutes of un-spanned setup dilute them all.
        if not self._armed_once:
            self._armed_once = True
            self._start = time.perf_counter()
            self.root_thread = threading.current_thread().name
        try:
            yield self
        finally:
            stack = self._stack_for_thread()
            if stack:
                logger.warning(
                    "Span profiler: unbalanced stack at disarm (%d span(s) still "
                    "open: %s)",
                    len(stack),
                    ", ".join(s._node.name for s in stack),
                )
                stack.clear()
            self._end = time.perf_counter()
            _ACTIVE.reset(token)

    @contextmanager
    def bind_thread(self) -> Iterator["SpanRecorder"]:
        """Arm inside a thread this recorder did not spawn.

        A new thread starts with a fresh context, so ``_ACTIVE`` reads its
        default and spans would vanish silently. Call this INSIDE the thread.
        """
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    # -- recording ------------------------------------------------------

    def _stack_for_thread(self) -> list:
        tid = threading.get_ident()
        stack = self._thread_stacks.get(tid)
        if stack is None:
            # The per-thread table itself is shared, so its insert takes the
            # lock; the returned list is touched by one thread only.
            with self._lock:
                stack = self._thread_stacks.setdefault(tid, [])
        return stack

    def span(self, name: str, units: float | None = None, gpu: bool = False) -> _Span:
        stack = self._stack_for_thread()
        thread = threading.current_thread().name
        with self._lock:
            table = self._roots if not stack else stack[-1]._node.children
            node = table.get(name)
            if node is None:
                node = _Node(name, thread)
                table[name] = node
            if units is not None:
                node.units += float(units)
        return _Span(self, node, stack, gpu and self.gpu_sync)

    # -- output ---------------------------------------------------------

    @property
    def wall_clock_s(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    def snapshot(self) -> dict:
        """Nested tree under a synthetic root. Safe to call while armed."""
        with self._lock:
            children = [_node_to_dict(n) for n in self._roots.values()]
        total = sum(c["total_s"] for c in children)
        return {
            "name": "root",
            "total_s": round(self.wall_clock_s, 6),
            "self_s": round(max(0.0, self.wall_clock_s - total), 6),
            "n_calls": 1,
            "units": 0.0,
            "max_s": round(self.wall_clock_s, 6),
            "first_call_s": round(self.wall_clock_s, 6),
            "thread": self.root_thread,
            "children": children,
        }


def _node_to_dict(node: _Node) -> dict:
    return {
        "name": node.name,
        "total_s": round(node.total_s, 6),
        "self_s": round(node.self_s, 6),
        "n_calls": node.n_calls,
        "units": node.units,
        "max_s": round(node.max_s, 6),
        "first_call_s": round(node.first_call_s or 0.0, 6),
        "thread": node.thread,
        "children": [_node_to_dict(c) for c in node.children.values()],
    }


# -- module-level API used by call sites ---------------------------------


def current() -> "SpanRecorder | None":
    """The armed recorder for this context, or None."""
    return _ACTIVE.get()


def span(name: str, units: float | None = None, gpu: bool = False):
    """Open a span on the armed recorder, or a no-op when nothing is armed."""
    rec = _ACTIVE.get()
    if rec is None:
        return _NULL_SPAN
    return rec.span(name, units=units, gpu=gpu)


def bind_target(fn):
    """Wrap a thread target so it records into the recorder armed *here*.

    Captured at wrap time on the spawning thread; armed inside the new thread.
    Returns ``fn`` unchanged when nothing is armed, so the unprofiled path
    gains no wrapper at all.
    """
    rec = _ACTIVE.get()
    if rec is None:
        return fn

    def _wrapped(*args, **kwargs):
        with rec.bind_thread():
            return fn(*args, **kwargs)

    return _wrapped
