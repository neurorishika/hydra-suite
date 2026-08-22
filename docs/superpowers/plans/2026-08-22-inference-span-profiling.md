# Inference Span Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hierarchical span profiler that, when Debug Mode is on, localizes a constant-cost performance defect to a single function in one tracking run — with byte-identical output and no measurable cost when Debug Mode is off.

**Architecture:** A dependency-free timing primitive (`SpanRecorder`) lives in `utils/profiling.py` and is reached ambiently through a `ContextVar`, so call sites write `with span(NAME):` without threading a profiler argument through every signature. `TrackingProfiler` owns its lifecycle (constructs it when `enabled`, arms it, renders it into the existing profile JSON and log). Span names are module constants in `utils/profiling_names.py`, enforced by a static coverage test. Threads that the recorder never armed are bound explicitly at six sites.

**Tech Stack:** Python 3.13, `contextvars`, `threading`, `time.perf_counter`, PyTorch (MPS/CUDA sync for the opt-in deep mode), pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-inference-span-profiling-design.md`

## Global Constraints

- Tracking output must stay **byte-identical with Debug Mode on and off**. Profiling that perturbs results is worse than none.
- **No measurable cost when Debug Mode is off** — pass criterion in Task 14: on/off median delta ≤2% and smaller than the within-condition IQR.
- `core/` must not import from any app layer (`trackerkit`, `posekit`, …). The gate arrives as the `ENABLE_PROFILING` param, which already exists.
- `utils/` must not import from `core/`, `data/`, or any app layer. `utils/profiling.py` imports only the standard library.
- CLAUDE.md design principles: no god objects, no copy-pasted boilerplate. Span names are constants, never duplicated string literals.
- **Implementation rule 1:** take the timers from `instrumentation.patch`, take **none** of the memoization (`_CHW_MEMO`, `reset_chw_memo`, `HYDRA_CHW_MEMO`). That is a functional change riding in the same diff and would break byte-identity.
- **Implementation rule 2:** every hunk in the **instrumentation** commits (Tasks 9-11) is a `with span(...)` / `@spanned` wrapper, an import, or an indentation change from one of those. No logic edits. Two sanctioned exceptions live in earlier, separately-reviewable commits: Task 6's `_effective_depth` depth clamp (behavior change, but only under `HYDRA_PROFILE_GPU=1`) and Task 7's deletion of the `HYDRA_RT_PROFILE` machinery.
- **Implementation rule 3:** spans wrap loops, never loop bodies. The "no measurable cost" claim depends on it.
- Commit as the configured git user (Rishika Mohanta). Do **not** add a `Co-Authored-By: Claude` trailer.
- All work happens on branch `feat/inference-span-profiling` in worktree `.worktrees/span-profiling`.

## Revision history

**Revision 2** (this document). Three adversarial reviews ran against revision 1 and found the plan substantially defective. The corrections, so a reviewer can check them rather than take them on trust:

| Defect in revision 1 | Correction |
|---|---|
| Self-proving run used `ant_obb_sleap`, which has `enable_pose_extractor: false` and `cnn_classifiers: []` — 2 of its 3 compared nodes do not exist | Moved to `ant_cnn_identity`, the only fixture with pose + CNN + head-tail; spec amended to rev 3; Task 14 Step 8 now machine-asserts non-vacuousness |
| Debug-OFF gate injected `debug_mode: false`, which fires the User-mode cleanup (`session.py:619-637`) and **deletes the two CSVs the gate compares** — it would have reported success having compared nothing | Clears `enable_profiling` instead, leaving `DEBUG_MODE` at its `True` default |
| Every `runner.py` invocation used `--config` / `--out` and omitted the required `--video`; the real args are `--orig-config`, `--video`, `--outdir` | All five commands corrected |
| `BACKEND_FORWARD` decorated `run_headtail_batch` / `run_cnn_batch`, nesting crop cost **under** model cost — the 24.0 s majority share of the originating defect would have blended into one `self_s` | Spans placed inside both functions as siblings, on the calls those functions actually make |
| Realtime tree never armed in Debug Mode (`run_batch_pass` is gated on `not effective_realtime_tracking_mode`) — a capability regression versus the deleted `HYDRA_RT_PROFILE` | Task 11 Step 6 arms the realtime loop too, which also activates the `worker.py:448` prefetcher binding |
| The spec's golden span-path test was absent; its substitute matched constant names as raw substrings (`CNN` ⊂ `CNNModel` in 79 files) | Golden test added (Task 11 Steps 7-8); registry test matches `N.<CONST>` tokens |
| `main_thread="MainThread"` hardcoded — `TrackingWorker` is a QThread, so every GUI node would read `concurrent` while headless gates looked clean | Recorder records its own arming thread |
| `_start` stamped at construction, diluting every depth-1 percentage by however long setup ran | Reset at first arm |
| Ten constants declared with no placement; `READ` wrapped `cap.read()` but not the `cap.set()` seek that *is* the 12 s cost | Placements added, `TRACK_FORWARD`/`TRACK_BACKWARD`/`WARP` deleted with reasons, `READ` wraps seek+read |
| `runner.py` used `span`/`N` with no import step; identity-pop drained the live stack when `self` was absent; process recorder armed only the first calling thread | All fixed in place |

Also corrected: the `caplog` reconstruction crashed on `%`-bearing lines, the rule-3 grep heuristic flagged the plan's own compliant code, and Task 16 wrote a merge SHA before the merge and ran `git checkout main` from a worktree where it cannot succeed.

## Two spec corrections this plan makes

Both were found by reading the code the spec cites. They are resolved here rather than by amending the spec, and are called out so a reviewer sees them.

1. **There is no `SessionRunner`.** The spec's `session/` tree says "armed at `SessionRunner.run` (session.py:530+)". The real class is `TrackingSessionCore` and the method is `run_post_tracking` (`session.py:528`). More importantly, **`session.py` constructs no `TrackingProfiler` at all today** — the three that exist are `worker.py:699` (per tracking pass), `post/merge.py:122`, and `post/interpolated_crops.py:1421`. Task 12 therefore creates a session-scoped profiler, and Task 4 adds an explicit **priority rule** so the two profilers nested inside it (`merge`, `interpolated_crops`) defer instead of stealing their subtrees out of the session tree.

2. **The warp-pool binding row is dropped, deliberately.** The spec's threading table lists `crops.py:164 _get_warp_pool` ThreadPoolExecutor via `initializer=`. Binding it is possible, but the only work available to instrument inside those workers is a *per-item* `apply_fit` body running once per detection — a span there violates implementation rule 3 (5M calls at 50 detections × 100k frames). The parent `apply_fit` span already bounds that cost **inclusively**, because the caller blocks on the pool. So the pool is left unbound and no spans are taken inside it. Task 11 documents this so the absence reads as a decision, not an oversight.

## File Structure

**New files:**

| File | Responsibility |
|---|---|
| `src/hydra_suite/utils/profiling.py` | The timing primitive: `SpanRecorder`, `_Span`, `_NullSpan`, the `ContextVar`, module-level `span()` / `current()` / `bind_target()`. Stdlib-only leaf. |
| `src/hydra_suite/utils/profiling_names.py` | Every span name as a module constant, plus the `@spanned` decorator. No logic. |
| `src/hydra_suite/utils/profiling_report.py` | Snapshot → JSON-ready dict and snapshot → indented text tree. Pure rendering, so `TrackingProfiler` does not grow a renderer. |
| `src/hydra_suite/utils/profiling_process.py` | The `HYDRA_PROFILE=1` process-level recorder and its `atexit` dump. Kept out of `profiling.py` so the primitive stays free of env/IO concerns. |
| `tests/utils/test_profiling.py` | Recorder semantics: nesting, `self_s`, `max_s`/`first_call_s`, units, disarmed cost, exceptions. |
| `tests/utils/test_profiling_threads.py` | `bind_target`, two-thread concurrent nesting, per-thread percentages. |
| `tests/utils/test_profiling_report.py` | Rendering: JSON shape, tree ordering, `ms/unit`. |
| `tests/utils/test_profiling_registry.py` | Static complement: every name constant is referenced; every span call site uses a constant. |
| `tests/core/inference/test_span_golden_paths.py` | The golden span-path set — the runtime guard against a silently-dropped span. |
| `tests/utils/test_profiling_process.py` | `HYDRA_PROFILE=1` arming, precedence, dump location. |
| `tests/core/tracking/test_profiler_spans.py` | `TrackingProfiler.spans` / `armed()` / priority deferral / JSON `spans` key. |

**Modified files:**

| File | Change |
|---|---|
| `src/hydra_suite/core/tracking/profiler.py` | `.spans` recorder, `armed()`, priority, `spans` + `gpu_mode` JSON keys, `SPAN TREE` log block. |
| `src/hydra_suite/core/inference/pipeline.py` | Batch-tree spans; producer-thread binding; deep-mode depth forcing. |
| `src/hydra_suite/core/inference/runner.py` | Realtime-tree spans; **delete** `HYDRA_RT_PROFILE` machinery and its seven call sites. |
| `src/hydra_suite/core/inference/stages/crops.py` | `crop_extract` / `affine_loop` / `warp_batch` / `apply_fit` spans. |
| `src/hydra_suite/core/canonicalization/resample.py` | `frame_to_chw` span around the sub-slice conversion loop. |
| `src/hydra_suite/core/inference/stages/{obb,headtail,cnn,pose,filtering}.py` | Stage-boundary spans. |
| `src/hydra_suite/core/inference/cache/writer.py` | Worker-thread binding + `cache_write` spans. |
| `src/hydra_suite/utils/frame_prefetcher.py` | Decode-thread binding in all three prefetcher classes. |
| `src/hydra_suite/core/tracking/session.py` | Session profiler + `session/` tree. |
| `src/hydra_suite/core/post/{merge,interpolated_crops,media_export}.py` | `post/` and `interp_crops/` trees; media writer-thread binding. |
| `src/hydra_suite/core/tracking/worker.py` | Arm around the runner pass; realtime prefetcher binding. |
| `docs/developer-guide/` | New profiling page. |

---

### Task 1: The span recorder primitive

**Files:**
- Create: `src/hydra_suite/utils/profiling.py`
- Test: `tests/utils/test_profiling.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class SpanRecorder(priority: int = 1)` with `.span(name, units=None, gpu=False) -> ContextManager[_Span]`, `.snapshot() -> dict`, `.armed() -> ContextManager[SpanRecorder]`, `.priority: int`, `.wall_clock_s: float`
  - `class _Span` with `.add_units(n: float) -> None`
  - `span(name, units=None, gpu=False)` — module level, what all call sites use
  - `current() -> SpanRecorder | None`
  - `PRIORITY_PROCESS = 0`, `PRIORITY_SESSION = 1`
  - Snapshot node dict keys: `name`, `total_s`, `self_s`, `n_calls`, `units`, `max_s`, `first_call_s`, `thread`, `children` (list)

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_profiling.py`:

```python
import time

import pytest

from hydra_suite.utils import profiling
from hydra_suite.utils.profiling import SpanRecorder, current, span


def _child(node, name):
    for c in node["children"]:
        if c["name"] == name:
            return c
    raise AssertionError(f"{name} not in {[c['name'] for c in node['children']]}")


def test_disarmed_span_is_a_noop():
    assert current() is None
    with span("nothing") as sp:
        sp.add_units(5)
    assert current() is None


def test_disarmed_span_returns_the_shared_singleton():
    with span("a") as one:
        pass
    with span("b") as two:
        pass
    assert one is two


def test_nesting_and_self_time():
    rec = SpanRecorder()
    with rec.armed():
        with rec.span("parent"):
            with rec.span("child"):
                time.sleep(0.02)
    snap = rec.snapshot()
    parent = _child(snap, "parent")
    child = _child(parent, "child")
    assert child["total_s"] >= 0.02
    assert parent["total_s"] >= child["total_s"]
    # self_s is inclusive minus direct children
    assert parent["self_s"] == pytest.approx(
        parent["total_s"] - child["total_s"], abs=1e-6
    )
    assert child["self_s"] == pytest.approx(child["total_s"], abs=1e-6)


def test_same_name_under_different_parents_stays_distinct():
    rec = SpanRecorder()
    with rec.armed():
        with rec.span("cnn"):
            with rec.span("crop_extract"):
                pass
        with rec.span("pose"):
            with rec.span("crop_extract"):
                pass
    snap = rec.snapshot()
    assert _child(_child(snap, "cnn"), "crop_extract")["n_calls"] == 1
    assert _child(_child(snap, "pose"), "crop_extract")["n_calls"] == 1


def test_max_s_and_first_call_s():
    rec = SpanRecorder()
    with rec.armed():
        with rec.span("s"):
            time.sleep(0.03)
        with rec.span("s"):
            pass
        with rec.span("s"):
            pass
    node = _child(rec.snapshot(), "s")
    assert node["n_calls"] == 3
    assert node["first_call_s"] >= 0.03
    assert node["max_s"] >= 0.03
    # the mean alone would hide that the cost was all in call 1
    assert node["total_s"] / 3 < node["max_s"]


def test_units_are_summed_per_node_and_never_from_children():
    rec = SpanRecorder()
    with rec.armed():
        with rec.span("window", units=1):
            with rec.span("pose", units=40):
                pass
        with rec.span("window", units=1):
            with rec.span("pose", units=35):
                pass
    snap = rec.snapshot()
    window = _child(snap, "window")
    assert window["units"] == 2
    assert _child(window, "pose")["units"] == 75


def test_add_units_after_enter():
    rec = SpanRecorder()
    with rec.armed():
        with rec.span("detect") as sp:
            sp.add_units(12)
    assert _child(rec.snapshot(), "detect")["units"] == 12


def test_exception_mid_span_leaves_a_balanced_stack():
    rec = SpanRecorder()
    with rec.armed():
        with pytest.raises(ValueError):
            with rec.span("outer"):
                with rec.span("inner"):
                    raise ValueError("boom")
        # a phantom parent would nest this under "outer"
        with rec.span("after"):
            pass
    snap = rec.snapshot()
    assert {c["name"] for c in snap["children"]} == {"outer", "after"}


def test_armed_restores_the_previous_recorder():
    outer = SpanRecorder()
    inner = SpanRecorder()
    with outer.armed():
        assert current() is outer
        with inner.armed():
            assert current() is inner
        assert current() is outer
    assert current() is None


def test_armed_defers_to_an_equal_or_higher_priority_recorder():
    session = SpanRecorder(priority=profiling.PRIORITY_SESSION)
    nested = SpanRecorder(priority=profiling.PRIORITY_SESSION)
    with session.armed():
        with nested.armed():
            with span("work"):
                pass
    # the nested recorder deferred, so the span landed in the session tree
    assert _child(session.snapshot(), "work")["n_calls"] == 1
    assert nested.snapshot()["children"] == []


def test_armed_wins_over_a_lower_priority_recorder():
    process = SpanRecorder(priority=profiling.PRIORITY_PROCESS)
    session = SpanRecorder(priority=profiling.PRIORITY_SESSION)
    with process.armed():
        with session.armed():
            with span("inside"):
                pass
        with span("outside"):
            pass
    assert _child(session.snapshot(), "inside")["n_calls"] == 1
    assert {c["name"] for c in process.snapshot()["children"]} == {"outside"}


def test_unbalanced_stack_warns_at_disarm(caplog):
    rec = SpanRecorder()
    cm = rec.armed()
    cm.__enter__()
    sp = rec.span("leaky")
    sp.__enter__()
    cm.__exit__(None, None, None)
    assert any("unbalanced" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/utils/test_profiling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.utils.profiling'`

- [ ] **Step 3: Write the implementation**

Create `src/hydra_suite/utils/profiling.py`:

```python
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

    __slots__ = ("_rec", "_node", "_stack", "_t0", "_child_s")

    def __init__(self, rec: "SpanRecorder", node: _Node, stack: list) -> None:
        self._rec = rec
        self._node = node
        self._stack = stack
        self._t0 = 0.0
        self._child_s = 0.0

    def __enter__(self) -> "_Span":
        self._stack.append(self)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> bool:
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

    def __init__(self, priority: int = PRIORITY_SESSION) -> None:
        self.priority = int(priority)
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
        return _Span(self, node, stack)

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/utils/test_profiling.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/utils/profiling.py tests/utils/test_profiling.py
git commit -m "feat(profiling): span recorder primitive"
```

---

### Task 2: Thread binding semantics

**Files:**
- Test: `tests/utils/test_profiling_threads.py`

**Interfaces:**
- Consumes: `SpanRecorder`, `bind_target`, `span` from Task 1.
- Produces: no new API — this task proves the Task 1 threading contract and fixes it if the tests expose a defect. The spec calls the missing-binding failure "the single most likely silent bug in the design", so it gets its own gate.

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_profiling_threads.py`:

```python
import threading
import time

from hydra_suite.utils.profiling import SpanRecorder, bind_target, span


def _child(node, name):
    for c in node["children"]:
        if c["name"] == name:
            return c
    raise AssertionError(f"{name} not in {[c['name'] for c in node['children']]}")


def test_unbound_thread_records_nothing():
    """The failure this design exists to prevent: work that costs zero."""
    rec = SpanRecorder()

    def worker():
        with span("orphan"):
            time.sleep(0.01)

    with rec.armed():
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert rec.snapshot()["children"] == []


def test_bind_target_captures_the_armed_recorder():
    rec = SpanRecorder()

    def worker():
        with span("producer_work"):
            time.sleep(0.01)

    with rec.armed():
        t = threading.Thread(target=bind_target(worker))
        t.start()
        t.join()
    node = _child(rec.snapshot(), "producer_work")
    assert node["n_calls"] == 1
    assert node["total_s"] >= 0.01


def test_bind_target_is_identity_when_nothing_is_armed():
    def worker():
        pass

    assert bind_target(worker) is worker


def test_two_threads_nest_without_corrupting_parentage():
    """A shared stack would interleave pushes and cross-parent the trees."""
    rec = SpanRecorder()
    ready = threading.Barrier(2)

    def side():
        with span("side"):
            ready.wait()
            with span("side_child"):
                time.sleep(0.02)

    with rec.armed():
        t = threading.Thread(target=bind_target(side), name="side-thread")
        t.start()
        with span("main"):
            ready.wait()
            with span("main_child"):
                time.sleep(0.02)
        t.join()

    snap = rec.snapshot()
    main = _child(snap, "main")
    side_node = _child(snap, "side")
    assert [c["name"] for c in main["children"]] == ["main_child"]
    assert [c["name"] for c in side_node["children"]] == ["side_child"]


def test_nodes_are_stamped_with_their_thread():
    rec = SpanRecorder()

    def worker():
        with span("off_thread"):
            pass

    with rec.armed():
        t = threading.Thread(target=bind_target(worker), name="pipeline-obb-producer")
        t.start()
        t.join()
    assert _child(rec.snapshot(), "off_thread")["thread"] == "pipeline-obb-producer"


def test_concurrent_totals_may_exceed_wall_clock():
    """Documented distortion: percentages are per-thread, never across."""
    rec = SpanRecorder()

    def worker():
        with span("b"):
            time.sleep(0.05)

    with rec.armed():
        t = threading.Thread(target=bind_target(worker))
        t.start()
        with span("a"):
            time.sleep(0.05)
        t.join()
    snap = rec.snapshot()
    summed = sum(c["total_s"] for c in snap["children"])
    assert summed > 0.08  # ~0.10 of span time inside ~0.05 of wall-clock
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/utils/test_profiling_threads.py -v`
Expected: all PASS (6 tests). If `test_two_threads_nest_without_corrupting_parentage` or `test_nodes_are_stamped_with_their_thread` fails, `_stack_for_thread` is inserting into `_thread_stacks` without the lock — re-check that Task 1's `with self._lock: setdefault(...)` guard is present, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/utils/test_profiling_threads.py
git commit -m "test(profiling): thread binding and concurrent nesting"
```

---

### Task 3: Span-name registry and the `@spanned` decorator

**Files:**
- Create: `src/hydra_suite/utils/profiling_names.py`
- Test: `tests/utils/test_profiling_registry.py`

**Interfaces:**
- Consumes: `span` from Task 1.
- Produces: one `str` constant per span (names listed below, used verbatim by Tasks 7–12), plus `spanned(name, units=None, gpu=False)` — a decorator wrapping a function in a span.

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_profiling_registry.py`:

```python
import re
from pathlib import Path

import hydra_suite
from hydra_suite.utils import profiling_names
from hydra_suite.utils.profiling import SpanRecorder
from hydra_suite.utils.profiling_names import spanned

SRC = Path(hydra_suite.__file__).parent


def _constants() -> dict[str, str]:
    return {
        k: v
        for k, v in vars(profiling_names).items()
        if k.isupper() and isinstance(v, str)
    }


def test_decorator_records_a_span():
    rec = SpanRecorder()

    @spanned("decorated")
    def work():
        return 42

    with rec.armed():
        assert work() == 42
    assert rec.snapshot()["children"][0]["name"] == "decorated"


def test_decorator_preserves_name_and_doc():
    @spanned("x")
    def work():
        """docstring."""

    assert work.__name__ == "work"
    assert work.__doc__ == "docstring."


def test_decorator_is_transparent_when_disarmed():
    @spanned("y")
    def work(a, b=2):
        return a + b

    assert work(1, b=3) == 4


def test_every_constant_is_used_somewhere_in_src():
    """A refactor that drops a span must fail a test, not go silent.

    Matches the ATTRIBUTE REFERENCE (``N.CNN`` / ``profiling_names.CNN``), not
    the bare name. A raw substring check is vacuous: ``CNN`` occurs in 79 files
    via ``CNNModel``, ``POSE`` in 29 via ``ENABLE_POSE_EXTRACTOR``, ``WARP``
    via ``WARP_BATCH``, ``WRITE`` via ``CACHE_WRITE``. Every one of those spans
    could be deleted wholesale and a substring test would stay green.
    """
    sources = [
        p.read_text()
        for p in SRC.rglob("*.py")
        if p.name != "profiling_names.py"
    ]
    blob = "\n".join(sources)
    unused = [
        k
        for k in _constants()
        if not re.search(rf"(?:\bN|profiling_names)\.{k}\b", blob)
    ]
    assert not unused, f"span names declared but never placed: {unused}"


def test_span_call_sites_use_constants_not_literals():
    """Enforces the registry rule: no duplicated string literals."""
    bad: list[str] = []
    pattern = re.compile(r"(?:^|[^\w.])(?:span|spanned)\(\s*([\"'])")
    for p in SRC.rglob("*.py"):
        if p.name in {"profiling.py", "profiling_names.py", "profiling_process.py"}:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pattern.search(line):
                bad.append(f"{p.relative_to(SRC)}:{i}: {line.strip()}")
    assert not bad, "span() called with a string literal; use a NAMES constant:\n" + "\n".join(bad)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/utils/test_profiling_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.utils.profiling_names'`

- [ ] **Step 3: Write the implementation**

Create `src/hydra_suite/utils/profiling_names.py`:

```python
"""Every span name, as a constant.

Bare string literals at call sites would be copy-pasted boilerplate in string
form, and a refactor that moved a function would silently drop its row with
nothing failing. ``tests/utils/test_profiling_registry.py`` enforces both
halves: every constant here is used, and no call site passes a literal.

Names are LOCAL to their parent — the tree supplies the prefix, so
``CROP_EXTRACT`` under ``cnn`` and under ``pose`` stay distinct without
callers hand-prefixing strings. Dynamic / label-keyed names are prohibited:
they would make memory O(labels).
"""

from __future__ import annotations

import functools

from .profiling import span

# -- session tree ---------------------------------------------------------
SESSION = "session"
# NOTE: the spec map lists track_forward / track_backward here. They are
# omitted deliberately: the session profiler arms inside run_post_tracking
# (session.py:528), which runs AFTER both tracking passes complete, so there is
# no scope in which those spans could be opened. The passes are already
# profiled separately by worker.py's own profiler.
POSTPROCESS = "postprocess"
BACKWARD_POSTPROCESS = "backward_postprocess"
POSE_QUALITY = "pose_quality"
TEMPORAL_POSE = "temporal_pose"
TRAJECTORY_POSTPROC = "trajectory_postproc"
INTERPOLATE_AND_SCALE = "interpolate_and_scale"
MERGE = "merge"
RICH_EXPORT = "rich_export"
BUILD_DATAFRAME = "build_dataframe"
RELINK = "relink"
WRITE = "write"
DATASET_GENERATION = "dataset_generation"
MEDIA_EXPORT = "media_export"
ANNOTATED_VIDEO = "annotated_video"

# -- inference tree -------------------------------------------------------
INFERENCE = "inference"
BATCH_PASS = "batch_pass"
OPEN_CACHES = "open_caches"
WINDOW = "window"
DECODE = "decode"
DETECT = "detect"
RUN_OBB = "run_obb"
MODEL_EXECUTE = "model_execute"
EXTRACT_RAW = "extract_raw"
MATERIALIZE = "materialize"
RUN_BGSUB_BATCH = "run_bgsub_batch"
FILTER = "filter"
HEADTAIL = "headtail"
CNN = "cnn"
POSE = "pose"
CROP_EXTRACT = "crop_extract"
FRAME_TO_CHW = "frame_to_chw"
AFFINE_LOOP = "affine_loop"
WARP_BATCH = "warp_batch"
FOREIGN_MASK = "foreign_mask"
APPLY_FIT = "apply_fit"
BACKEND_FORWARD = "backend_forward"
PREP_LOOP = "prep_loop"
TRANSPORT = "transport"
APRILTAG = "apriltag"
CACHE_WRITE = "cache_write"
ENQUEUE = "enqueue"
FLUSH = "flush"
ASSEMBLE_SCATTER = "assemble_scatter"

# -- realtime tree --------------------------------------------------------
REALTIME = "realtime"
RT_OBB = "obb"
RT_CROPS = "crops"
RT_INDIVIDUAL = "individual"
RT_CACHE = "cache"
RT_FINALIZE = "finalize"

# -- post tree ------------------------------------------------------------
POST = "post"
PREPARE = "prepare"
RESOLVE = "resolve"
INTERPOLATE = "interpolate"
TAG_IDENTITY = "tag_identity"
RESCALE = "rescale"

# -- interpolated-crops tree ---------------------------------------------
INTERP_CROPS = "interp_crops"
SETUP = "setup"
GAP_DETECTION = "gap_detection"
CROP_EXTRACTION = "crop_extraction"
READ = "read"
WARP = "warp"
POSE_INFERENCE = "pose_inference"
CNN_INFERENCE = "cnn_inference"
FINALIZE = "finalize"


def spanned(name: str, units: float | None = None, gpu: bool = False):
    """Wrap a function body in a span. Use for function boundaries.

    ``with span(...)`` is reserved for sub-function regions such as
    ``FRAME_TO_CHW`` and ``AFFINE_LOOP``.
    """

    def _decorate(fn):
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            with span(name, units=units, gpu=gpu):
                return fn(*args, **kwargs)

        return _wrapper

    return _decorate
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/utils/test_profiling_registry.py -v`
Expected: `test_every_constant_is_used_somewhere_in_src` FAILS (nothing uses them yet); the other four PASS.

- [ ] **Step 5: Make the coverage test pass by deferring it until instrumentation lands**

Mark it so the suite is green now and the guard arms in Task 12:

```python
import pytest

@pytest.mark.xfail(
    reason="arms once instrumentation lands (Task 12 removes this marker)",
    strict=False,
)
def test_every_constant_is_used_somewhere_in_src():
```

Run: `python -m pytest tests/utils/test_profiling_registry.py -v`
Expected: 4 PASS, 1 XFAIL

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/utils/profiling_names.py tests/utils/test_profiling_registry.py
git commit -m "feat(profiling): span-name registry and spanned decorator"
```

---

### Task 4: Rendering — JSON dict and text tree

**Files:**
- Create: `src/hydra_suite/utils/profiling_report.py`
- Test: `tests/utils/test_profiling_report.py`

**Interfaces:**
- Consumes: `SpanRecorder.snapshot()` from Task 1.
- Produces:
  - `render_tree_lines(snapshot: dict, main_thread: str) -> list[str]`
  - `SPAN_TREE_HEADER: str`

- [ ] **Step 1: Write the failing test**

Create `tests/utils/test_profiling_report.py`:

```python
from hydra_suite.utils.profiling_report import render_tree_lines

SNAP = {
    "name": "root",
    "total_s": 10.0,
    "self_s": 1.0,
    "n_calls": 1,
    "units": 0.0,
    "max_s": 10.0,
    "first_call_s": 10.0,
    "thread": "MainThread",
    "children": [
        {
            "name": "window",
            "total_s": 9.0,
            "self_s": 1.0,
            "n_calls": 100,
            "units": 100.0,
            "max_s": 0.5,
            "first_call_s": 0.5,
            "thread": "MainThread",
            "children": [
                {
                    "name": "cnn",
                    "total_s": 6.0,
                    "self_s": 6.0,
                    "n_calls": 100,
                    "units": 4000.0,
                    "max_s": 0.1,
                    "first_call_s": 0.1,
                    "thread": "MainThread",
                    "children": [],
                },
                {
                    "name": "decode",
                    "total_s": 2.0,
                    "self_s": 2.0,
                    "n_calls": 100,
                    "units": 0.0,
                    "max_s": 0.1,
                    "first_call_s": 0.1,
                    "thread": "pipeline-obb-producer",
                    "children": [],
                },
            ],
        }
    ],
}


def test_children_sort_by_total_descending():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    body = [ln for ln in lines if "cnn" in ln or "decode" in ln]
    assert "cnn" in body[0] and "decode" in body[1]


def test_percentages_are_of_the_parent():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    cnn = next(ln for ln in lines if "cnn" in ln)
    assert "66.7%" in cnn  # 6.0 / 9.0


def test_ms_per_unit_is_shown_when_units_present():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    cnn = next(ln for ln in lines if "cnn" in ln)
    assert "1.50 ms/u" in cnn  # 6.0 s / 4000 units


def test_ms_per_unit_omitted_without_units():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    decode = next(ln for ln in lines if "decode" in ln)
    assert "ms/u" not in decode


def test_off_thread_nodes_are_marked_concurrent():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    decode = next(ln for ln in lines if "decode" in ln)
    assert "concurrent" in decode
    cnn = next(ln for ln in lines if "cnn" in ln)
    assert "concurrent" not in cnn


def test_depth_is_indented():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    window = next(ln for ln in lines if "window" in ln)
    cnn = next(ln for ln in lines if "cnn" in ln)
    assert len(cnn) - len(cnn.lstrip()) > len(window) - len(window.lstrip())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/utils/test_profiling_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.utils.profiling_report'`

- [ ] **Step 3: Write the implementation**

Create `src/hydra_suite/utils/profiling_report.py`:

```python
"""Render a span snapshot as log lines.

Kept out of both ``profiling.py`` (which stays a stdlib-only primitive) and
``TrackingProfiler`` (which would otherwise grow a renderer alongside its
lifecycle role).

Percentages are OF THE PARENT, never of a global total: at depth>=2 summed
span time legitimately exceeds wall-clock when threads overlap, so a
"% of run" column would be a lie. Off-thread subtrees carry a ``concurrent``
marker so a subtree that is 43% of its thread but 4% of the pass reads as both.
"""

from __future__ import annotations

SPAN_TREE_HEADER = (
    "  {:<38} {:>9} {:>7} {:>8} {:>9} {:>9}".format(
        "SPAN", "total", "% par", "n", "ms/call", "max ms"
    )
)


def render_tree_lines(snapshot: dict, main_thread: str) -> list[str]:
    """Indented tree, children sorted by ``total_s`` descending."""
    lines: list[str] = []
    _render(snapshot, snapshot["total_s"], 0, main_thread, lines)
    return lines


def _render(
    node: dict, parent_total: float, depth: int, main_thread: str, out: list[str]
) -> None:
    if depth > 0:
        pct = (node["total_s"] / parent_total * 100.0) if parent_total > 0 else 0.0
        n = max(node["n_calls"], 1)
        label = ("  " * depth) + node["name"]
        suffix = ""
        if node.get("units"):
            suffix += f" | {node['total_s'] / node['units'] * 1000:.2f} ms/u"
        if node.get("thread") and node["thread"] != main_thread:
            suffix += f" | concurrent ({node['thread']})"
        out.append(
            "  {:<38} {:>8.2f}s {:>6.1f}% {:>8d} {:>8.2f} {:>8.2f}{}".format(
                label[:38],
                node["total_s"],
                pct,
                node["n_calls"],
                node["total_s"] / n * 1000,
                node["max_s"] * 1000,
                suffix,
            )
        )
    for child in sorted(
        node.get("children", []), key=lambda c: c["total_s"], reverse=True
    ):
        _render(child, node["total_s"], depth + 1, main_thread, out)
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/utils/test_profiling_report.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/utils/profiling_report.py tests/utils/test_profiling_report.py
git commit -m "feat(profiling): span tree renderer"
```

---

### Task 5: TrackingProfiler integration

**Files:**
- Modify: `src/hydra_suite/core/tracking/profiler.py:141-183` (`__init__`), `:495-509` (summary dict), `:604` (end of `log_final_summary`)
- Test: `tests/core/tracking/test_profiler_spans.py`

**Interfaces:**
- Consumes: `SpanRecorder`, `PRIORITY_SESSION` (Task 1); `render_tree_lines`, `SPAN_TREE_HEADER` (Task 4).
- Produces:
  - `TrackingProfiler.spans: SpanRecorder | None`
  - `TrackingProfiler.armed() -> ContextManager` — the context manager every consumer wraps its work in
  - `get_summary()` gains keys `"spans"` (nested tree) and `"gpu_mode"` (`"off"` / `"deep"`)

- [ ] **Step 1: Write the failing tests**

Create `tests/core/tracking/test_profiler_spans.py`:

```python
import json

from hydra_suite.core.tracking.profiler import TrackingProfiler
from hydra_suite.utils.profiling import current, span


def test_disabled_profiler_builds_no_recorder():
    prof = TrackingProfiler(enabled=False)
    assert prof.spans is None


def test_disabled_profiler_armed_is_a_noop():
    prof = TrackingProfiler(enabled=False)
    with prof.armed():
        assert current() is None
        with span("nothing"):
            pass


def test_enabled_profiler_collects_spans():
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        with span("stage"):
            with span("substage"):
                pass
    tree = prof.spans.snapshot()
    stage = tree["children"][0]
    assert stage["name"] == "stage"
    assert stage["children"][0]["name"] == "substage"


def test_summary_carries_spans_and_gpu_mode(monkeypatch):
    # A lingering exported HYDRA_PROFILE_GPU would fail this AND silently force
    # pipeline_depth=1 on every run in that shell.
    monkeypatch.delenv("HYDRA_PROFILE_GPU", raising=False)
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        with span("stage"):
            pass
    prof.end_frame()
    summary = prof.get_summary()
    assert summary["gpu_mode"] == "off"
    assert summary["spans"]["children"][0]["name"] == "stage"


def test_existing_summary_keys_are_untouched():
    prof = TrackingProfiler(enabled=True)
    prof.phase_start("batched_detection")
    prof.phase_end("batched_detection")
    prof.end_frame()
    summary = prof.get_summary()
    for key in ("enabled", "total_frames", "wall_clock_s", "phases", "categories"):
        assert key in summary
    assert "batched_detection" in summary["phases"]


def test_summary_is_json_serialisable(tmp_path):
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        with span("stage") as sp:
            sp.add_units(3)
    prof.end_frame()
    out = tmp_path / "p.json"
    assert prof.export_summary(out) is not None
    loaded = json.loads(out.read_text())
    assert loaded["spans"]["children"][0]["units"] == 3


def test_nested_profiler_defers_to_the_outer_one():
    """merge / interpolated_crops run inside the session profiler's scope."""
    outer = TrackingProfiler(enabled=True)
    inner = TrackingProfiler(enabled=True)
    with outer.armed():
        with inner.armed():
            with span("nested_work"):
                pass
    assert outer.spans.snapshot()["children"][0]["name"] == "nested_work"
    assert inner.spans.snapshot()["children"] == []


def test_log_final_summary_emits_a_span_tree(caplog):
    import logging

    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        with span("stage"):
            pass
    prof.end_frame()
    with caplog.at_level(logging.INFO):
        prof.log_final_summary()
    # getMessage(), NOT `r.message % r.args`: pytest's capture handler already
    # formats the record, so re-applying args raises TypeError on the
    # "(gpu_mode=%s)" line and ValueError on any line containing a literal `%`
    # (the renderer's "% par" column).
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "SPAN TREE" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/core/tracking/test_profiler_spans.py -v`
Expected: FAIL — `AttributeError: 'TrackingProfiler' object has no attribute 'spans'`

- [ ] **Step 3: Add the recorder to `__init__`**

In `src/hydra_suite/core/tracking/profiler.py`, add to the imports at the top (after `import numpy as np`):

```python
from hydra_suite.utils.profiling import PRIORITY_SESSION, SpanRecorder
from hydra_suite.utils.profiling_report import SPAN_TREE_HEADER, render_tree_lines
```

Then change `__init__` (line 141-145) from:

```python
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if not enabled:
            return
```

to:

```python
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        # The span recorder is the ONLY thing built before the early return, so
        # `.spans` is always a valid attribute (None when disabled).
        self.spans: SpanRecorder | None = (
            SpanRecorder(priority=PRIORITY_SESSION) if enabled else None
        )
        if not enabled:
            return
```

- [ ] **Step 4: Add `armed()`**

Insert immediately after `set_config` (which ends around line 190) in `profiler.py`:

```python
    # ------------------------------------------------------------------
    # Span profiling
    # ------------------------------------------------------------------
    @contextmanager
    def armed(self):
        """Arm the span recorder for the duration of the block.

        Every consumer (the runner pass, ``TrackingSessionCore``, ``merge``,
        ``interpolated_crops``) wraps its work in this. Nested arms defer to
        the outermost profiler of equal priority, so a session-scoped tree
        keeps its ``merge`` and ``interp_crops`` subtrees instead of having
        them split into a separate recorder.

        A no-op when disabled — ``span()`` then returns the shared null
        singleton and nothing is recorded.
        """
        if self.spans is None:
            yield self
            return
        with self.spans.armed():
            yield self
```

and add `from contextlib import contextmanager` to the imports.

- [ ] **Step 5: Add the summary keys**

In `get_summary` (profiler.py:495), change the `summary = {...}` literal to append two keys after `"categories": categories,`:

```python
            "categories": categories,
            "gpu_mode": _gpu_mode(),
            "spans": self.spans.snapshot() if self.spans is not None else None,
        }
```

and add this module-level helper near the top of `profiler.py`, after `logger = logging.getLogger(__name__)`:

```python
def _gpu_mode() -> str:
    """``"deep"`` when HYDRA_PROFILE_GPU=1 (see Task 6), else ``"off"``.

    Stamped into the JSON so nobody compares a serialized depth=1 tree
    against a production depth=2 one.
    """
    import os

    return "deep" if os.environ.get("HYDRA_PROFILE_GPU") else "off"
```

- [ ] **Step 6: Add the SPAN TREE log block**

In `log_final_summary`, immediately before the final `logger.info("=" * 60)` (profiler.py:604):

```python
        spans = summary.get("spans")
        if spans and spans.get("children"):
            logger.info("-" * 60)
            logger.info("  SPAN TREE  (gpu_mode=%s)", summary.get("gpu_mode", "off"))
            logger.info(SPAN_TREE_HEADER)
            logger.info("-" * 60)
            for line in render_tree_lines(spans, main_thread=spans["thread"]):
                logger.info("%s", line)
```

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/core/tracking/test_profiler_spans.py tests/test_tracking_profiler.py tests/core/tracking/test_profile_output_path.py -v`
Expected: PASS — including the pre-existing profiler tests, which have no strict key-set assertions.

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/core/tracking/profiler.py tests/core/tracking/test_profiler_spans.py
git commit -m "feat(profiling): wire span recorder into TrackingProfiler"
```

---

### Task 6: Opt-in deep-GPU mode

**Files:**
- Modify: `src/hydra_suite/utils/profiling.py` (`_Span.__exit__`, `SpanRecorder.__init__`)
- Modify: `src/hydra_suite/core/inference/pipeline.py:139-190` (`__init__` / `window_size` region — force depth 1)
- Test: `tests/utils/test_profiling_gpu.py`

**Interfaces:**
- Consumes: `SpanRecorder` (Task 1).
- Produces:
  - `hydra_suite.utils.profiling.deep_gpu_enabled() -> bool`
  - `SpanRecorder(..., gpu_sync=False)`; `gpu=True` spans call `torch.{cuda,mps}.synchronize()` on exit **only** when `gpu_sync` is on.

**Why this is opt-in:** `torch.cuda.synchronize()` / `torch.mps.synchronize()` are **device-wide**, not stream-scoped, and `pipeline_depth` defaults to 2, so a sync on the consumer thread drains the *producer's* in-flight OBB kernels and bills OBB's device time to CNN. Measured on `hydra-mps`: `torch.mps.synchronize()` costs 318 µs with one pending op versus 8.5 µs unsynced — a ~37x penalty, ~1.8 ms/frame at ~6 gpu spans/frame, which alone breaks the ≤2% target. The default profiled path therefore does **not** sync.

- [ ] **Step 1: Write the failing test**

Create `tests/utils/test_profiling_gpu.py`:

```python
import pytest

from hydra_suite.utils import profiling
from hydra_suite.utils.profiling import SpanRecorder, deep_gpu_enabled


def test_deep_gpu_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HYDRA_PROFILE_GPU", raising=False)
    assert deep_gpu_enabled() is False


def test_deep_gpu_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("HYDRA_PROFILE_GPU", "1")
    assert deep_gpu_enabled() is True


def test_default_recorder_never_syncs(monkeypatch):
    calls = []
    monkeypatch.setattr(profiling, "_synchronize", lambda: calls.append(1))
    rec = SpanRecorder(gpu_sync=False)
    with rec.armed():
        with rec.span("forward", gpu=True):
            pass
    assert calls == []


def test_gpu_sync_recorder_syncs_only_gpu_spans(monkeypatch):
    calls = []
    monkeypatch.setattr(profiling, "_synchronize", lambda: calls.append(1))
    rec = SpanRecorder(gpu_sync=True)
    with rec.armed():
        with rec.span("host_only"):
            pass
        with rec.span("forward", gpu=True):
            pass
    assert len(calls) == 1


def test_pipeline_depth_is_forced_to_one_in_deep_mode(monkeypatch):
    from hydra_suite.core.inference.pipeline import _effective_depth

    monkeypatch.delenv("HYDRA_PROFILE_GPU", raising=False)
    assert _effective_depth(2) == 2
    monkeypatch.setenv("HYDRA_PROFILE_GPU", "1")
    assert _effective_depth(2) == 1
    assert _effective_depth(4) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/utils/test_profiling_gpu.py -v`
Expected: FAIL — `ImportError: cannot import name 'deep_gpu_enabled'`

- [ ] **Step 3: Add the sync path to `profiling.py`**

Add near the top of `src/hydra_suite/utils/profiling.py`, after the `_ACTIVE` definition:

```python
import os


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
```

Change `SpanRecorder.__init__` to accept the flag:

```python
    def __init__(self, priority: int = PRIORITY_SESSION, gpu_sync: bool | None = None) -> None:
        self.priority = int(priority)
        self.gpu_sync = deep_gpu_enabled() if gpu_sync is None else bool(gpu_sync)
```

(keep the remaining body unchanged).

Pass the flag into the span and honour it on exit. In `SpanRecorder.span`, change the return to:

```python
        return _Span(self, node, stack, gpu and self.gpu_sync)
```

In `_Span`, add `_sync` to `__slots__`, accept it in `__init__`, and sync as the **first** statement of `__exit__` so the queued device work is inside the measured interval:

```python
    __slots__ = ("_rec", "_node", "_stack", "_t0", "_child_s", "_sync")

    def __init__(self, rec, node, stack, sync: bool = False) -> None:
        self._rec = rec
        self._node = node
        self._stack = stack
        self._t0 = 0.0
        self._child_s = 0.0
        self._sync = sync

    def __exit__(self, *_exc) -> bool:
        if self._sync:
            _synchronize()
        dt = time.perf_counter() - self._t0
        ...  # rest unchanged
```

- [ ] **Step 4: Force depth 1 in deep mode**

In `src/hydra_suite/core/inference/pipeline.py`, add a module-level helper next to the other module functions:

```python
def _effective_depth(depth: int) -> int:
    """``pipeline_depth``, clamped to 1 under the deep-GPU profiling pass.

    A device-wide sync taken on the consumer thread drains the producer's
    in-flight OBB kernels, so deep-GPU mode removes the producer rather than
    reporting cross-stage misattribution as fact.
    """
    from hydra_suite.utils.profiling import deep_gpu_enabled

    return 1 if deep_gpu_enabled() else int(depth)
```

and in `Pipeline.__init__` (line 139), wrap the stored depth:

```python
        self.depth = _effective_depth(depth)
```

(replace whatever the current `self.depth = ...` assignment is with this, leaving the parameter name unchanged).

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/utils/test_profiling_gpu.py tests/test_inference_pipeline_depth1.py tests/test_inference_pipeline_stop.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/utils/profiling.py src/hydra_suite/core/inference/pipeline.py tests/utils/test_profiling_gpu.py
git commit -m "feat(profiling): opt-in deep-GPU mode (HYDRA_PROFILE_GPU)"
```

---

### Task 7: Process-level recorder; retire `HYDRA_RT_PROFILE`

**Files:**
- Create: `src/hydra_suite/utils/profiling_process.py`
- Modify: `src/hydra_suite/core/inference/runner.py:64-87` (delete), `:777, :833, :840-841, :914, :972, :1053-1055` (delete call sites)
- Test: `tests/utils/test_profiling_process.py`

**Interfaces:**
- Consumes: `SpanRecorder`, `PRIORITY_PROCESS` (Task 1); `render_tree_lines` (Task 4).
- Produces:
  - `hydra_suite.utils.profiling_process.maybe_arm_process_recorder() -> SpanRecorder | None`
  - `hydra_suite.utils.profiling_process.dump_path() -> Path`

**Why:** naive deletion of `HYDRA_RT_PROFILE` is a capability regression — `core/inference` is also driven by DetectKit and PoseKit, which have no `TrackingProfiler`, and the env var was the only way to profile those paths. `HYDRA_PROFILE=1` replaces it with the same recorder and renderer. It is also the supported way to profile **without changing what the run does**: Debug Mode is not observation-only (it changes intermediate cleanup and CSV outputs at `session.py:614-621`), so "turn on Debug and re-run" profiles a different run than the one that was slow.

- [ ] **Step 1: Write the failing test**

Create `tests/utils/test_profiling_process.py`:

```python
import json

import pytest

from hydra_suite.utils import profiling_process
from hydra_suite.utils.profiling import PRIORITY_PROCESS, current, span
from hydra_suite.utils.profiling_process import maybe_arm_process_recorder


@pytest.fixture(autouse=True)
def _reset():
    profiling_process.reset_for_test()
    yield
    profiling_process.reset_for_test()


def test_not_armed_without_the_env_var(monkeypatch):
    monkeypatch.delenv("HYDRA_PROFILE", raising=False)
    monkeypatch.delenv("HYDRA_RT_PROFILE", raising=False)
    assert maybe_arm_process_recorder() is None
    assert current() is None


def test_armed_by_hydra_profile(monkeypatch):
    monkeypatch.setenv("HYDRA_PROFILE", "1")
    rec = maybe_arm_process_recorder()
    assert rec is not None
    assert rec.priority == PRIORITY_PROCESS
    with span("detectkit_work"):
        pass
    assert rec.snapshot()["children"][0]["name"] == "detectkit_work"


def test_legacy_alias_still_works(monkeypatch):
    monkeypatch.delenv("HYDRA_PROFILE", raising=False)
    monkeypatch.setenv("HYDRA_RT_PROFILE", "1")
    assert maybe_arm_process_recorder() is not None


def test_arming_is_idempotent(monkeypatch):
    monkeypatch.setenv("HYDRA_PROFILE", "1")
    assert maybe_arm_process_recorder() is maybe_arm_process_recorder()


def test_tracking_profiler_wins_while_armed(monkeypatch):
    from hydra_suite.core.tracking.profiler import TrackingProfiler

    monkeypatch.setenv("HYDRA_PROFILE", "1")
    process = maybe_arm_process_recorder()
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        with span("inside_session"):
            pass
    with span("outside_session"):
        pass
    assert prof.spans.snapshot()["children"][0]["name"] == "inside_session"
    assert [c["name"] for c in process.snapshot()["children"]] == ["outside_session"]


def test_dump_writes_json(monkeypatch, tmp_path):
    monkeypatch.setenv("HYDRA_PROFILE", "1")
    monkeypatch.setattr(profiling_process, "dump_path", lambda: tmp_path / "p.json")
    maybe_arm_process_recorder()
    with span("work"):
        pass
    profiling_process.dump()
    loaded = json.loads((tmp_path / "p.json").read_text())
    assert loaded["spans"]["children"][0]["name"] == "work"


def test_rt_prof_machinery_is_gone():
    import hydra_suite.core.inference.runner as runner

    for name in ("_RT_PROF_ACC", "_rt_prof_on", "_rt_prof_add", "_rt_prof_flush"):
        assert not hasattr(runner, name), f"{name} should have been deleted"


def test_detectkit_and_posekit_paths_arm_the_recorder(monkeypatch):
    """Risk 4's mitigation must not ship untested.

    Both kits drive ``core/inference`` with no ``TrackingProfiler``, so
    ``InferenceRunner.__init__`` is the arming point they share. Assert on the
    arming seam rather than constructing a kit — the kits need Qt and models,
    which no unit test should require.
    """
    import hydra_suite.core.inference.runner as runner

    monkeypatch.setenv("HYDRA_PROFILE", "1")
    src = __import__("inspect").getsource(runner.InferenceRunner.__init__)
    assert "maybe_arm_process_recorder" in src

    rec = maybe_arm_process_recorder()
    with span("kit_driven_work"):
        pass
    assert rec.snapshot()["children"][0]["name"] == "kit_driven_work"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/utils/test_profiling_process.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.utils.profiling_process'`

- [ ] **Step 3: Write the process recorder**

Create `src/hydra_suite/utils/profiling_process.py`:

```python
"""``HYDRA_PROFILE=1`` — a process-level span recorder.

Replaces the retired ``HYDRA_RT_PROFILE`` machinery in
``core/inference/runner.py``. Two reasons it is not just an alias:

* ``core/inference`` is also driven by DetectKit and PoseKit, which build no
  ``TrackingProfiler``. Without this, retiring the env var would blind them.
* Debug Mode is not observation-only — it changes intermediate cleanup and CSV
  outputs — so "turn on Debug and re-run" profiles a DIFFERENT run than the
  one that was slow. This is the User-mode route to profile the real run.

Precedence: a ``TrackingProfiler`` recorder (``PRIORITY_SESSION``) wins while
armed and the process recorder resumes outside it. Spans go to exactly one
recorder, never both.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
from pathlib import Path

from .profiling import PRIORITY_PROCESS, SpanRecorder, _ACTIVE
from .profiling_report import render_tree_lines

logger = logging.getLogger(__name__)

_recorder: SpanRecorder | None = None
_lock = threading.Lock()
_video_log_dir: Path | None = None


def enabled() -> bool:
    return bool(os.environ.get("HYDRA_PROFILE") or os.environ.get("HYDRA_RT_PROFILE"))


def set_log_dir(path) -> None:
    """Tell the dump where a session's ``<video>_logs/`` directory is."""
    global _video_log_dir
    _video_log_dir = Path(path) if path else None


def dump_path() -> Path:
    if _video_log_dir is not None:
        return _video_log_dir / f"span_profile_{os.getpid()}.json"
    from hydra_suite.paths import get_data_dir

    return Path(get_data_dir()) / "profiles" / f"span_profile_{os.getpid()}.json"


def maybe_arm_process_recorder() -> SpanRecorder | None:
    """Arm (once) when the env var is set. Returns the recorder, or None.

    Armed for the process lifetime — no ``reset()`` — so any thread that
    inherits or binds this context records into it.
    """
    global _recorder
    if not enabled():
        return None
    with _lock:
        if _recorder is None:
            _recorder = SpanRecorder(priority=PRIORITY_PROCESS)
            atexit.register(dump)
    # Arm on EVERY call, not just at creation: contextvars do not cross
    # threads, so a runner constructed on the GUI thread and driven from a
    # worker thread would otherwise record nothing. `is None` preserves the
    # spec's precedence — a TrackingProfiler armed here keeps the context.
    if _ACTIVE.get() is None:
        _ACTIVE.set(_recorder)
    return _recorder


def dump() -> None:
    """Write the tree to JSON and log it. Never raises."""
    if _recorder is None:
        return
    snap = _recorder.snapshot()
    try:
        path = dump_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"spans": snap}, indent=2))
        logger.info("Span profile written to %s", path)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to write span profile", exc_info=True)
    for line in render_tree_lines(snap, main_thread=snap["thread"]):
        logger.info("%s", line)


def reset_for_test() -> None:
    """Test hook only."""
    global _recorder, _video_log_dir
    _recorder = None
    _video_log_dir = None
    _ACTIVE.set(None)
```

- [ ] **Step 4: Delete the `HYDRA_RT_PROFILE` machinery**

In `src/hydra_suite/core/inference/runner.py`, delete lines 64-87 in their entirety (the comment block, `_RT_PROF_ACC`, `_rt_prof_on`, `_rt_prof_add`, `_rt_prof_flush`). Then delete these statements:

- `:777-778` — `_prof = _rt_prof_on()` and `_ts = time.perf_counter() if _prof else 0.0`
- `:831-835` — the `if _prof:` block computing `_now` and calling `_rt_prof_add("obb", ...)`
- `:839-841` — the `if _prof:` block inside the zero-detection early return
- `:912-915` — the `if _prof:` block calling `_rt_prof_add("crops", ...)`
- `:970-973` — the `if _prof:` block calling `_rt_prof_add("individual", ...)`
- `:1052-1055` — the `if _prof:` block calling `_rt_prof_add("finalize", ...)` and `_rt_prof_flush()`

Task 9 replaces them with spans at the same boundaries, so verify with `grep -n "_rt_prof\|_prof\b" src/hydra_suite/core/inference/runner.py` returning nothing.

- [ ] **Step 5: Arm the process recorder at the two entry points**

In `src/hydra_suite/core/inference/runner.py`, at the top of `InferenceRunner.__init__`, add:

```python
        from hydra_suite.utils.profiling_process import maybe_arm_process_recorder

        maybe_arm_process_recorder()
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/utils/test_profiling_process.py -v`
Expected: PASS (8 tests)

Run: `python -m pytest tests/ -k "inference or runner" -x -q`
Expected: no new failures versus the pre-change baseline.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/utils/profiling_process.py src/hydra_suite/core/inference/runner.py tests/utils/test_profiling_process.py
git commit -m "feat(profiling): HYDRA_PROFILE process recorder; retire HYDRA_RT_PROFILE"
```

---

### Task 8: Bind the off-thread sites

**Files:**
- Modify: `src/hydra_suite/core/inference/pipeline.py:541-543` (producer thread)
- Modify: `src/hydra_suite/core/inference/cache/writer.py:63` (async worker)
- Modify: `src/hydra_suite/utils/frame_prefetcher.py:104, :265, :357` (three prefetcher classes)
- Modify: `src/hydra_suite/core/post/media_export.py:605` (writer thread)

**Interfaces:**
- Consumes: `bind_target` (Task 1).
- Produces: no new API. Every thread that will carry a span in Tasks 9-12 records into the armed recorder instead of vanishing.

**Why this task precedes instrumentation:** spans on an unbound thread read the ContextVar default and record nothing — the report would confidently show that work costing zero. `bind_target` returns the function unchanged when nothing is armed, so these five edits are no-ops on the unprofiled path.

The spec's sixth row — the `crops.py:164` warp `ThreadPoolExecutor` — is **deliberately not bound**. The only work inside those workers is a per-detection `apply_fit` body; a span there would violate implementation rule 3 (5M calls at 50 detections × 100k frames), and the parent `apply_fit` span already bounds the pool's cost inclusively because the caller blocks on it.

- [ ] **Step 1: Bind the pipeline producer**

In `src/hydra_suite/core/inference/pipeline.py`, add to the imports:

```python
from hydra_suite.utils.profiling import bind_target
```

and change lines 541-543 from:

```python
        producer_thread = threading.Thread(
            target=producer, name="pipeline-obb-producer", daemon=True
        )
```

to:

```python
        producer_thread = threading.Thread(
            # A new thread starts with a fresh context, so an unbound producer
            # would report zero OBB time at depth>=2.
            target=bind_target(producer),
            name="pipeline-obb-producer",
            daemon=True,
        )
```

- [ ] **Step 2: Bind the cache writer worker and span its loop**

Also add `ENQUEUE` and `FLUSH` spans, or the bound writer thread carries no spans at all: wrap the body of `_enqueue_or_write` in `with span(N.ENQUEUE):` and the actual disk write inside `_worker_loop` in `with span(N.FLUSH):`. Both are per-item, so they are the one sanctioned rule-3 exception on this thread — the writer's whole purpose is per-item I/O and there is no enclosing loop to hoist to. Note the exception in the commit message.

- [ ] **Step 2b: Bind the cache writer worker**

In `src/hydra_suite/core/inference/cache/writer.py`, add `from hydra_suite.utils.profiling import bind_target` to the imports and change line 63 from:

```python
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
```

to:

```python
            self._worker = threading.Thread(
                target=bind_target(self._worker_loop), daemon=True
            )
```

- [ ] **Step 3: Bind the three frame prefetchers**

In `src/hydra_suite/utils/frame_prefetcher.py`, add `from .profiling import bind_target` to the imports, then change all three thread constructions:

- line 104 (`FramePrefetcher.start`): `target=bind_target(self._prefetch_loop)`
- line 265 (`SparseFramePrefetcher.start`): `target=bind_target(self._prefetch_loop)`
- line 357 (`SequentialScanPrefetcher.start`): `target=bind_target(self._scan_loop)`

This is the most consequential binding: `interp_crops/crop_extraction/read` sits directly over these threads, so an unbound version would report near-zero frame-read cost against ~12 s of measured video seek. Note the span must go **inside** the decode loop — a span wrapped around `.read()` on the consumer side measures queue-wait, not decode.

- [ ] **Step 4: Bind the media-export writer**

In `src/hydra_suite/core/post/media_export.py`, add `from hydra_suite.utils.profiling import bind_target` to the imports and change line 605 from:

```python
    _writer = _threading.Thread(target=_writer_thread, daemon=True)
```

to:

```python
    _writer = _threading.Thread(target=bind_target(_writer_thread), daemon=True)
```

- [ ] **Step 5: Verify no thread site was missed**

Run:

```bash
grep -rn "threading.Thread(\|_threading.Thread(" src/hydra_suite/core src/hydra_suite/utils src/hydra_suite/data
```

Expected: every hit either wraps its target in `bind_target(...)` or is a Qt/GUI thread outside the profiled scope. Record any newly-found site in the commit message rather than leaving it silent.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/utils/test_profiling_threads.py tests/test_inference_pipeline_depth1.py tests/test_inference_pipeline_stop.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/core/inference/pipeline.py src/hydra_suite/core/inference/cache/writer.py src/hydra_suite/utils/frame_prefetcher.py src/hydra_suite/core/post/media_export.py
git commit -m "feat(profiling): bind off-thread sites to the span recorder"
```

---

### Task 9: Instrument the inference batch tree

**Files:**
- Modify: `src/hydra_suite/core/inference/pipeline.py:209-433` (`_run_detection_for_window`, `_process_obb_results`)
- Modify: `src/hydra_suite/core/inference/runner.py:1273+` (`run_batch_pass`)
- Modify: `src/hydra_suite/core/inference/stages/crops.py:51-112, :180-231, :339-354`
- Modify: `src/hydra_suite/core/canonicalization/resample.py:194-209`
- Modify: `src/hydra_suite/core/inference/stages/{obb,headtail,cnn,pose,filtering}.py`

**Interfaces:**
- Consumes: `span` (Task 1); the name constants from Task 3.
- Produces: the `inference/batch_pass/` subtree. Every span name used here is already declared in `profiling_names.py`.

**This is the task the whole feature exists for.** The 34.4 s defect was **24.0 s in the head-tail + CNN consumers and 10.4 s in pose** — so head-tail and CNN get the same `crop_extract` children as pose, not pose alone. The tell is that the frame→CHW conversion is O(frame area) and independent of detection count while the warp scales with `units`; that signature is only visible when the two are separate spans.

- [ ] **Step 1: Instrument the crop seam (the defect's location)**

In `src/hydra_suite/core/inference/stages/crops.py`, add:

```python
from hydra_suite.utils.profiling import span
from hydra_suite.utils import profiling_names as N
```

In `extract_canonical_crops` (line 51), wrap the affine loop and the warp call. Replace lines 92-108 with:

```python
    m_aligns: list[np.ndarray] = []
    with span(N.AFFINE_LOOP):
        for i in range(n):
            try:
                m_align, _theta, _clipped = canonical_affine(
                    obb_result.corners[i], geometry
                )
            except ValueError:
                m_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            m_aligns.append(m_align)

    with span(N.WARP_BATCH, units=n, gpu=True):
        crops = canonical_warp_batch_from_frame(
            frame, m_aligns, geometry, lambda sub: _frame_to_chw_float(sub, device)
        )

    if suppress_foreign and n > 1:
        with span(N.FOREIGN_MASK, units=n):
            crops = _apply_foreign_mask_canonical_batch(
                crops, obb_result, geometry, background_color
            )
    return crops
```

In `extract_classifier_crops` (line 180), apply the identical treatment to lines 207-219:

```python
    m_aligns: list[np.ndarray] = []
    with span(N.AFFINE_LOOP):
        for i in range(n):
            try:
                m_align, _theta, _clipped = canonical_affine(
                    obb_result.corners[i], geometry
                )
            except ValueError:
                m_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            m_aligns.append(m_align)

    with span(N.WARP_BATCH, units=n, gpu=True):
        crops_t = canonical_warp_batch_from_frame(
            frame, m_aligns, geometry, lambda sub: _frame_to_chw_float(sub, device)
        )
```

Wrap `apply_fit_batch` (line 339) with the decorator — it blocks on the warp pool, so its span bounds the pool's cost inclusively:

```python
@N.spanned(N.APPLY_FIT)
def apply_fit_batch(crops: list, fit: FitResult) -> list:
```

- [ ] **Step 2: Split out the frame→CHW conversion**

In `src/hydra_suite/core/canonicalization/resample.py`, add:

```python
from hydra_suite.utils.profiling import span
from hydra_suite.utils import profiling_names as N
```

Wrap the sub-slice conversion loop (lines 194-200) — the loop, not its body:

```python
    boxes = [_canvas_footprint_aabb(m, geometry, (h_in, w_in)) for m in m_aligns]
    subs: List[torch.Tensor | None] = []
    # Separated from WARP_BATCH because this cost is O(sum of crop footprints)
    # and the warp scales with n — the signature that identifies an O(frame
    # area) conversion regression.
    with span(N.FRAME_TO_CHW, units=len(boxes)):
        for x0, y0, x1, y1 in boxes:
            if x1 > x0 and y1 > y0:
                subs.append(to_chw_float(_slice_frame_view(view, layout, x0, y0, x1, y1)))
            else:
                subs.append(None)
```

- [ ] **Step 3: Instrument the stage internals — crop_extract and backend_forward as SIBLINGS**

Add `from hydra_suite.utils import profiling_names as N` and `from hydra_suite.utils.profiling import span` to each stage module.

**Do not decorate `run_headtail_batch` / `run_cnn_batch` with `BACKEND_FORWARD`.** Those functions extract crops *and* run the model, so a decorator makes `crop_extract` a child of `backend_forward` and blends the two costs into one `self_s` — which is how the 24.0 s head-tail + CNN majority share of the originating defect would stay unlocalized. The spec's map draws them as siblings; place them as siblings.

In `stages/headtail.py:280-313` and `stages/cnn.py:170-196` (the two have identical structure, a CUDA branch and a CPU/MPS branch):

```python
    if frames_on_cuda(runtime, frames):
        from hydra_suite.core.canonicalization.resample import letterbox_fit

        from .crops import extract_canonical_crops_batch

        with span(N.CROP_EXTRACT):
            batch = extract_canonical_crops_batch(frames, obb_results, geometry, runtime)
        n_total = batch.crops.shape[0]
        if n_total:
            with span(N.APPLY_FIT, units=n_total):
                fitted = letterbox_fit(batch.crops, fit.model_wh)
                cuda_crops = [
                    (fitted[i] * 255.0).floor().clamp(0, 255) for i in range(n_total)
                ]
            with span(N.BACKEND_FORWARD, units=n_total, gpu=True):
                all_probs = model.backend.predict_batch_cuda(cuda_crops, input_is_bgr=False)
        else:
            all_probs = []
    else:
        from .crops import apply_fit_batch, extract_classifier_crops_batch_np

        with span(N.CROP_EXTRACT):
            batch = extract_classifier_crops_batch_np(frames, obb_results, geometry)
        if batch.crops:
            np_crops: list[np.ndarray] = apply_fit_batch(batch.crops, fit)
            with span(N.BACKEND_FORWARD, units=len(np_crops), gpu=True):
                all_probs = model.backend.predict_batch(np_crops)
        else:
            all_probs = []
```

(`apply_fit_batch` carries its own `APPLY_FIT` span from Step 1, so the CPU branch needs no extra wrapper.)

In `stages/pose.py:385`, do **not** decorate `run_pose_batch` either. Wrap its per-crop preparation loop (the `for i in range(n_total):` at ~:429) in `with span(N.PREP_LOOP, units=n_total):`, the host↔device crop transfer inside it in `with span(N.TRANSPORT):` **hoisted above the loop** if the transfer is batched — otherwise omit `TRANSPORT` and delete the constant rather than putting a span in a per-crop body — and the `model.backend.predict_batch*` call in `with span(N.BACKEND_FORWARD, units=n_total, gpu=True):`.

In `stages/obb.py`, wrap the model call inside `run_obb` in `with span(N.MODEL_EXECUTE, gpu=True):` and the raw-tensor extraction that follows it in `with span(N.EXTRACT_RAW):` — decorating the whole of `run_obb` would collapse the spec map's `run_obb/{model_execute, extract_raw}` into one node. Read `obb.py:466+` and place both inside.

`materialize_tensors` (`obb.py:1476`) is called from `_process_obb_results`' per-frame loop on the **consumer** thread, so it is NOT decorated — Step 4 wraps the loop instead. `filter_for_source` (`filtering.py:318`) is likewise not decorated; both call sites (batch and realtime) wrap it, so a decorator would produce `filter/filter`.

`BACKEND_FORWARD` repeats deliberately: names are local to their parent, so the three land under `headtail/`, `cnn/` and `pose/` as distinct nodes.

- [ ] **Step 4: Instrument the pipeline window**

In `src/hydra_suite/core/inference/pipeline.py`, add `from hydra_suite.utils import profiling_names as N` and `from hydra_suite.utils.profiling import span`. **Add the same two imports to `src/hydra_suite/core/inference/runner.py`** — Step 5 and all of Task 10 use `span` and `N` there, and Task 7 added only a function-local import of `maybe_arm_process_recorder`.

In `_run_detection_for_window` (line 209), wrap the body:

```python
        cfg = self.stages.config
        with span(N.DETECT, units=len(window.frames)):
            if cfg.detection_source == "bgsub":
                with span(N.RUN_BGSUB_BATCH, units=len(window.frames)):
                    return run_bgsub_batch(
                        window.frames,
                        window.frame_indices,
                        self.stages.bgsub_model,
                        cfg.bgsub,
                        self.runtime,
                    )
            with span(N.RUN_OBB, units=len(window.frames)):
                # DECODE goes on the producer side: `_stream_windows` is the
                # generator that reads frames, and at depth>=2 it runs on the
                # bound producer thread. Wrap its per-window `next()` in
                # `_stream_windows` itself with `with span(N.DECODE):`.
                raw_list = run_obb(
                    window.frames,
                    self.stages.obb_models,
                    cfg.obb,
                    self.runtime,
                    roi_mask=self.stages.roi_mask,
                )
                for raw in raw_list:
                    if isinstance(raw, _RawOBBTensors):
                        self.runtime.handoff(raw.xywhr)
                        self.runtime.handoff(raw.corners)
                        self.runtime.handoff(raw.conf)
                return raw_list
```

In `_process_obb_results` (line 258), wrap the four downstream stage blocks. The `headtail` block becomes:

```python
        headtail: dict[int, Any] | None = None
        if self.stages.headtail_model is not None:
            with span(N.HEADTAIL, units=sum(o.num_detections for o in nonempty_obbs)):
                headtail = run_headtail_batch(
                    nonempty_frames,
                    nonempty_obbs,
                    self.stages.headtail_model,
                    cfg.headtail,
                    self.runtime,
                    geometry,
                )
```

Wrap the per-frame materialize/filter loop (`for frame, frame_idx, raw in zip(...)`, ~:293) in `with span(N.MATERIALIZE, units=len(frames)):` — the loop, so `materialize_tensors` and `filter_for_source` are counted once per window rather than once per frame. Wrap the CNN phase loop (the loop, not its body) in `with span(N.CNN, units=sum(o.num_detections for o in nonempty_obbs)):`, the pose block in `with span(N.POSE, units=sum(o.num_detections for o in nonempty_obbs)):`, the AprilTag loop in `with span(N.APRILTAG):`, the per-type cache-write loop in `with span(N.CACHE_WRITE):`, and the closing `return scatter(...)` in `with span(N.ASSEMBLE_SCATTER):`.

Inside the pose block, wrap `extract_canonical_crops_batch` in `with span(N.CROP_EXTRACT):` so `AFFINE_LOOP` / `WARP_BATCH` / `FRAME_TO_CHW` from Steps 1-2 nest under it. Do the same inside `run_headtail_batch` and `run_cnn_batch` — each calls `extract_classifier_crops_batch`; wrap that call in `with span(N.CROP_EXTRACT):`.

- [ ] **Step 5: Open the batch-pass root**

In `src/hydra_suite/core/inference/runner.py`, in `run_batch_pass` (line 1273), wrap the body after the `cache_dir` guard:

```python
        with span(N.INFERENCE), span(N.BATCH_PASS):
            with span(N.OPEN_CACHES):
                caches = _open_caches(
                    self.config, self.cache_dir, self._video_sig, self._roi_mask
                )
            self._caches = caches
            ...  # remainder of the existing body, indented one level
```

and in `Pipeline._run_sync` / `_run_double_buffer`, wrap each per-window consumer call in `with span(N.WINDOW, units=len(window)):` — the loop iteration boundary, one span per window, which is what makes `detection_batch_size=1` versus 25 readable as `ms/unit`.

- [ ] **Step 6: Verify no span sits inside a per-detection loop body**

A grep heuristic does not work here — it flags the legitimate wrap-the-loop pattern's neighbors and misses violations whose loop header is more than a few lines up. Do a targeted read instead. List every span call site and check each one's enclosing scope by hand:

```bash
git diff main -- src/hydra_suite | grep -n "span(N\.\|@N.spanned" | sed 's/^/  /'
```

For each, open the file and answer: *what is the cadence of this span — per run, per pass, per window, per frame, or per detection?* Per-detection is a violation; hoist it above the loop and use `units` instead.

Three per-frame spans are **sanctioned exceptions**, and they are the complete list — anything else per-frame or finer is a defect:

| Span | Site | Why |
|---|---|---|
| `READ` | prefetcher decode loops | the decode *is* the unit of work; there is no enclosing loop to hoist to |
| `ENQUEUE` / `FLUSH` | cache-writer loop | the writer's whole purpose is per-item I/O |

At 100k frames these run ~100k times each, not 5M. Record the count you expect in the commit message.

- [ ] **Step 7: Verify the diff is wrappers only**

Run: `git diff main --stat -- src/hydra_suite/core`
Read the full diff and confirm every hunk is a `with span(...)` / `@spanned` wrapper, an import, or an indentation change from one of those. Any logic edit is implementation-rule-2 violation. Confirm no `_CHW_MEMO`, `reset_chw_memo`, or `HYDRA_CHW_MEMO` appears anywhere:

```bash
git diff main | grep -c "CHW_MEMO"
```

Expected: `0`

- [ ] **Step 8: Run the tests**

Run: `python -m pytest tests/ -k "inference or crop or canonical or pipeline" -q`
Expected: no new failures versus the pre-change baseline.

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/core
git commit -m "feat(profiling): instrument the inference batch tree"
```

---

### Task 10: Instrument the realtime tree

**Files:**
- Modify: `src/hydra_suite/core/inference/runner.py:760-1273` (`run_realtime`)

**Interfaces:**
- Consumes: `span` (Task 1); `REALTIME`, `RT_OBB`, `RT_CROPS`, `RT_INDIVIDUAL`, `RT_CACHE`, `RT_FINALIZE` (Task 3).
- Produces: the `inference/realtime/` subtree.

The realtime tree mirrors `run_realtime`'s own structure, **not** the batch child names — the two paths do different work and a shared vocabulary would imply a comparability that does not hold. These spans replace the `_rt_prof_add` calls deleted in Task 7, at the same four boundaries plus `finalize`.

- [ ] **Step 1: Wrap the four sections**

**Wrap, do not restructure.** The real function has an `if self.config.detection_source == "bgsub": ... else: ...` at `runner.py:779-813`, with the `detection_ids` re-stamp and the `caches.detection.write_frame` call **outside both branches**. Open the spans around the existing statements and re-indent; do not move any statement into or out of a branch. Identify the `_prof` blocks to delete **by content** (`if _prof:` … `_rt_prof_add(...)`), never by the line numbers in Task 7 — those drift as soon as the first edit lands.

Shape:

```python
        with span(N.REALTIME):
            with span(N.RT_OBB, units=1):
                # ENTIRE existing block verbatim, one indent level deeper:
                #   if detection_source == "bgsub": ... else: ...
                #   raw_obb = OBBResult(... re-stamped detection_ids ...)
                #   if caches is not None and caches.detection is not None:
                #       caches.detection.write_frame(frame_idx, result=raw_obb)
                ...

            with span(N.RT_FILTER):
                filtered_obb, det_indices = filter_for_source(
                    self.config, raw_obb, roi_mask
                )
            ...
```

Use `RT_FILTER = "filter"` (a realtime-tree constant), not the batch tree's `N.FILTER` — the spec is explicit that the realtime tree mirrors `run_realtime`'s structure and does not share the batch child names. Add `RT_FILTER` to `profiling_names.py`.

Wrap the crop-extraction block that ended at former line 914 in `with span(N.RT_CROPS, units=filtered_obb.num_detections):`, the `_do_ht()/_do_cnn()/_do_pose()/_do_at()` block that ended at former line 972 in `with span(N.RT_INDIVIDUAL, units=filtered_obb.num_detections):`, the cache-persist block that follows it in `with span(N.RT_CACHE):`, and the streaming-payload block that ended at former line 1053 in `with span(N.RT_FINALIZE):`.

The zero-detection early return at former line 839 sits inside the `REALTIME` span and needs no special handling — `_Span.__exit__` runs on the `return`.

- [ ] **Step 2: Verify the deleted machinery left nothing behind**

Run: `grep -n "_rt_prof\|_RT_PROF\|_prof\b" src/hydra_suite/core/inference/runner.py`
Expected: no output.

- [ ] **Step 3: Run the tests**

Run: `python -m pytest tests/ -k "realtime or runner" -q`
Expected: no new failures.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/inference/runner.py
git commit -m "feat(profiling): instrument the realtime tree"
```

---

### Task 11: Instrument the session, post and interp_crops trees

**Files:**
- Modify: `src/hydra_suite/core/tracking/session.py:153-172` (`__init__`), `:528-586` (`run_post_tracking`)
- Modify: `src/hydra_suite/core/post/merge.py:122-175`
- Modify: `src/hydra_suite/core/post/interpolated_crops.py:1421+`
- Modify: `src/hydra_suite/core/tracking/worker.py:1239`

**Interfaces:**
- Consumes: `TrackingProfiler.armed()` (Task 5); `span` (Task 1); the session/post/interp names (Task 3).
- Produces: the `session/`, `post/` and `interp_crops/` trees.

**Why this task exists:** `SessionRunner` does not exist — the class is `TrackingSessionCore` and `session.py` builds no profiler at all today. Its stages (`postprocess`, `rich_export`, `interpolated_crops`, `dataset_generation`, `media_export`, `annotated_video`) are ~28% of wall in pandas code plus dataset-generation seeks, and no armed consumer wrapped any of it. Without this task the design reproduces, at session scope, the "one opaque bucket" failure the whole feature exists to fix.

- [ ] **Step 1: Give `TrackingSessionCore` a profiler**

In `src/hydra_suite/core/tracking/session.py`, add the imports:

```python
from hydra_suite.core.tracking.profiler import TrackingProfiler
from hydra_suite.utils.profiling import span
from hydra_suite.utils import profiling_names as N
```

and at the end of `__init__` (line 156-172):

```python
        # Session-scoped span profiler. The merge / interpolated_crops
        # profilers nested below defer to this one (equal priority), so their
        # subtrees stay in the session tree instead of being split out.
        self._profiler = TrackingProfiler(
            enabled=bool(self.params.get("ENABLE_PROFILING", False))
        )
```

- [ ] **Step 2: Arm and span `run_post_tracking`**

Wrap the body of `run_post_tracking` (line 528) in `with self._profiler.armed(), span(N.SESSION):` and wrap each stage call in its span:

- `self._postprocess_csv(forward_csv)` → `with span(N.POSTPROCESS):`, and inside `_postprocess_csv` (session.py:176-189) wrap its stages in `N.POSE_QUALITY`, `N.TEMPORAL_POSE` and `N.TRAJECTORY_POSTPROC`. Read the method and place them on whatever it actually calls — without children this span is a single stage-level bucket over the ~28%-of-wall pandas hotspot, i.e. the session-scope reprise of the `batched_detection` failure this feature exists to fix.
- `self._interpolate_and_scale(forward_processed)` (the `else` branch, taken on every non-backward run) → `with span(N.INTERPOLATE_AND_SCALE):` — omitted from the first draft, leaving single-pass runs with an unmeasured stage
- `self._postprocess_csv(f"{base}_backward{ext}")` → `with span(N.BACKWARD_POSTPROCESS):`
- `self._merge(forward_processed, backward_processed)` → `with span(N.MERGE):`
- `self._export_rich(final_csv)` → `with span(N.RICH_EXPORT):`
- `self._run_interp_crops(final_csv)` → `with span(N.INTERP_CROPS):`
- `self._relink_export_rich(final_csv)` → `with span(N.RELINK):`
- `self._run_dataset_generation(final_csv)` → `with span(N.DATASET_GENERATION):`
- `self._run_final_media_export(final_csv)` → `with span(N.MEDIA_EXPORT):`
- `self._run_annotated_video(final_csv)` → `with span(N.ANNOTATED_VIDEO):`
- `_save_trajectories_to_csv(final_df, final_csv)` → `with span(N.WRITE):`

- [ ] **Step 3: Export the session tree**

At the end of `run_post_tracking`, before building `SessionResult`:

```python
            self._profiler.end_frame()
            self._profiler.log_final_summary()
            # Wire the HYDRA_PROFILE dump location — without this call
            # `profiling_process.set_log_dir` is dead code and the spec's
            # "<video>_logs/ when a session supplies one" clause never holds.
            from hydra_suite.utils.profiling_process import set_log_dir

            set_log_dir(build_video_log_dir(self.video_path, create=True))
            from hydra_suite.utils.video_artifacts import build_video_log_dir

            self._profiler.export_summary(
                build_video_log_dir(self.video_path, create=True)
                / "tracking_profile_session.json"
            )
```

There is **no** `"log_dir"` key in `self.paths` (`grep -rn '"log_dir"' src/` is empty), so `build_video_log_dir` — confirmed at `src/hydra_suite/utils/video_artifacts.py:112` — is the only route, the same one `worker._resolve_profile_path` uses.

The export is skipped on every `_stopped_result()` early return and on `TrackingSessionError` (`session.py:641-642`). That is correct for a cancelled run, but state it so a missing file does not read as a bug.

- [ ] **Step 4: Instrument the post tree**

In `src/hydra_suite/core/post/merge.py`, the profiler at line 122 keeps its existing `phase_start`/`phase_end` calls untouched. Add `from hydra_suite.utils.profiling import span` and `from hydra_suite.utils import profiling_names as N`, wrap the function body in `with profiler.armed(), span(N.POST):`, and add a span beside each existing phase pair:

- `prepare_trajs_for_merge` calls → `with span(N.PREPARE):`
- `resolve_trajectories(...)` → `with span(N.RESOLVE):`
- `interpolate_trajectories(...)` → `with span(N.INTERPOLATE):`
- `resolve_tag_identities(...)` → `with span(N.TAG_IDENTITY):`
- `rescale_coordinates(...)` → `with span(N.RESCALE):`

The `armed()` here defers when the session profiler is already armed (Task 5's priority rule), which is the normal path; it still arms when `merge` is called standalone.

- [ ] **Step 5: Instrument the interp_crops tree**

In `src/hydra_suite/core/post/interpolated_crops.py`, wrap the body from line 1421 in `with profiler.armed(), span(N.INTERP_CROPS):` and add spans at:

- `_validate_and_setup(...)` → `with span(N.SETUP):`
- the gap-detection block → `with span(N.GAP_DETECTION):`
- the per-gap crop-extraction loop (the loop, not the body) → `with span(N.CROP_EXTRACTION):`
- the pose-inference call → `with span(N.POSE_INFERENCE):`
- the CNN-inference call → `with span(N.CNN_INFERENCE):`
- the finalize/save block → `with span(N.FINALIZE):`

The `READ` span goes **inside** the prefetcher decode loops bound in Task 8 (`frame_prefetcher.py` `_prefetch_loop` / `_scan_loop`). A span on the consumer side of the queue would measure queue-wait, not decode.

**It must wrap the seek together with the read.** `SparseFramePrefetcher._prefetch_loop` (`frame_prefetcher.py:269-277`) does `self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)` **then** `cap.read()`, and the file's own comment at `:336` puts the per-frame seek at ~5-50 ms — the seek *is* the ~12 s measured in `project_sleap_roundtrip_audit`. Wrapping `cap.read()` alone reports near-zero exactly where the cost lives, the same class of lie the spec's Threading section warns about, one line lower down:

```python
        with span(N.READ):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = self.cap.read()
```

Apply the same pairing in all three prefetcher classes (`:109-113`, `:269-277`, `:361-370`), matching whatever each one's seek/read pair actually is.

`WARP` would land in the interpolated-crop warp call inside a per-crop loop body — a rule-3 violation. **Delete the `WARP` constant** and rely on `crop_extraction`'s inclusive total plus its `units`; note the deletion in the commit message.

- [ ] **Step 6: Arm around the runner pass AND the realtime loop**

In `src/hydra_suite/core/tracking/worker.py`, wrap the `inference_runner.run_batch_pass(...)` call at line 1239 in `with profiler.armed():` so the `inference/` tree lands in the forward/backward pass profile that `worker.py:4158` already exports.

**This is not sufficient on its own.** `run_batch_pass` is guarded at `worker.py:1229-1232` by `and not effective_realtime_tracking_mode`, so in realtime mode that arm never executes and every span from Task 10 hits the null singleton. Since Task 7 deleted `HYDRA_RT_PROFILE` — which *did* work in a GUI realtime run — leaving it there would be exactly the capability regression Task 7's rationale promises to avoid.

So also wrap the **per-frame tracking loop** (not each frame — one `armed()` at the phase boundary enclosing the loop) in `with profiler.armed():`. Find it by locating the `run_realtime` call sites:

```bash
grep -n "run_realtime(" src/hydra_suite/core/tracking/worker.py
```

Arm at the nearest enclosing phase boundary above them. This also activates the `worker.py:448-452` `FramePrefetcher` binding from Task 8 — `bind_target` returns the target unchanged when nothing is armed, so without this arm that binding is a no-op and the spec's sixth threading row stays uncovered.

- [ ] **Step 6b: Verify both trees actually populate**

```bash
grep -n "profiler.armed()" src/hydra_suite/core/tracking/worker.py
```

Expected: two occurrences — one around `run_batch_pass`, one around the realtime loop. One occurrence means the realtime tree is dead.

- [ ] **Step 7: Add the golden span-path test**

The spec names this "the **only** test that catches 'a span silently disappeared in a refactor' — the failure mode the whole feature exists to prevent". The registry test in Task 3 is a weaker, static complement: it proves a constant is *referenced*, not that a span is *reached at runtime* under the right parent.

Create `tests/core/inference/test_span_golden_paths.py`:

```python
"""Golden span-path set: the durable guard against a silently-dropped span."""

import json
from pathlib import Path

from hydra_suite.core.tracking.profiler import TrackingProfiler
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span

GOLDEN = Path(__file__).parent / "span_golden_paths.json"


def _paths(node, prefix=()):
    """Flatten a snapshot to the set of slash-joined span paths."""
    out = set()
    for child in node["children"]:
        path = prefix + (child["name"],)
        out.add("/".join(path))
        out |= _paths(child, path)
    return out


def _synthetic_batch_window():
    """Mirror the real nesting of one batch window without loading models.

    Every `with span(...)` here must correspond to a real placement in
    pipeline.py / crops.py / the stage modules. When the two drift, this test
    fails -- which is the entire point.
    """
    with span(N.INFERENCE), span(N.BATCH_PASS):
        with span(N.OPEN_CACHES):
            pass
        with span(N.WINDOW, units=1):
            with span(N.DETECT, units=1):
                with span(N.RUN_OBB, units=1):
                    with span(N.MODEL_EXECUTE, gpu=True):
                        pass
                    with span(N.EXTRACT_RAW):
                        pass
            with span(N.MATERIALIZE, units=1):
                pass
            for stage in (N.HEADTAIL, N.CNN, N.POSE):
                with span(stage, units=4):
                    with span(N.CROP_EXTRACT):
                        with span(N.AFFINE_LOOP):
                            pass
                        with span(N.WARP_BATCH, units=4, gpu=True):
                            with span(N.FRAME_TO_CHW, units=4):
                                pass
                    with span(N.APPLY_FIT, units=4):
                        pass
                    with span(N.BACKEND_FORWARD, units=4, gpu=True):
                        pass
            with span(N.CACHE_WRITE):
                with span(N.ENQUEUE):
                    pass
            with span(N.ASSEMBLE_SCATTER):
                pass


def test_golden_span_paths():
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        _synthetic_batch_window()
    actual = _paths(prof.spans.snapshot())
    expected = set(json.loads(GOLDEN.read_text()))

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    assert not missing, (
        f"span paths disappeared: {missing}\n"
        "A refactor dropped a span. Restore it, or update the golden set "
        "DELIBERATELY with a note in the commit message."
    )
    assert not extra, f"new span paths not in the golden set: {extra}"


def test_crop_extract_and_backend_forward_are_siblings():
    """Regression guard for the head-tail/CNN blending defect.

    24.0s of the 34.4s originating defect lived in the head-tail + CNN crop
    path. If crop_extract nests UNDER backend_forward, that cost blends with
    model time in one self_s and the tree indicts the wrong function.
    """
    prof = TrackingProfiler(enabled=True)
    with prof.armed():
        _synthetic_batch_window()
    paths = _paths(prof.spans.snapshot())
    for stage in ("headtail", "cnn", "pose"):
        base = f"inference/batch_pass/window/{stage}"
        assert f"{base}/crop_extract" in paths
        assert f"{base}/backend_forward" in paths
        assert f"{base}/backend_forward/crop_extract" not in paths
```

- [ ] **Step 8: Generate the golden file and check it in**

```bash
python - <<'EOF'
import json, sys
sys.path.insert(0, "tests/core/inference")
from test_span_golden_paths import _paths, _synthetic_batch_window, GOLDEN
from hydra_suite.core.tracking.profiler import TrackingProfiler

prof = TrackingProfiler(enabled=True)
with prof.armed():
    _synthetic_batch_window()
GOLDEN.write_text(json.dumps(sorted(_paths(prof.spans.snapshot())), indent=2))
print(GOLDEN.read_text())
EOF
```

Read the generated set and confirm by eye that it matches the spec's span map before committing it. A golden file generated from a wrong implementation locks in the wrong tree.

Run: `python -m pytest tests/core/inference/test_span_golden_paths.py -v`
Expected: PASS (2 tests)

- [ ] **Step 9: Arm the registry coverage test**

Remove the `@pytest.mark.xfail` marker added in Task 3 Step 5 from `test_every_constant_is_used_somewhere_in_src`.

Run: `python -m pytest tests/utils/test_profiling_registry.py -v`
Expected: PASS. Any constant reported as unused is a span the plan declared but never placed — either place it or delete the constant; do not silence the test.

- [ ] **Step 10: Run the tests**

Run: `python -m pytest tests/ -k "session or merge or post or interp or profil or span" -q`
Expected: no new failures.

- [ ] **Step 11: Commit**

```bash
git add src/hydra_suite tests/utils/test_profiling_registry.py tests/core/inference/
git commit -m "feat(profiling): instrument the session, post and interp_crops trees"
```

---

### Task 12: Documentation

**Files:**
- Create: `docs/developer-guide/profiling.md`
- Modify: `mkdocs.yml` (nav entry)

**Interfaces:**
- Consumes: everything above.
- Produces: the page a future investigator reads instead of writing ad-hoc timers.

- [ ] **Step 1: Write the page**

Create `docs/developer-guide/profiling.md`:

```markdown
# Profiling a tracking run

## Which instrument

| Symptom | Instrument |
|---|---|
| A stage got slower and you want to know which function | Span profiler (below) |
| A cross-cutting tax smeared over thousands of small calls | `cProfile` — see "What the span profiler cannot find" |
| You need device time, not host time | `HYDRA_PROFILE_GPU=1` |

## Turning it on

**Debug Mode.** The span profiler is on whenever `ENABLE_PROFILING` is —
which the Debug Mode toggle already derives. The tree is written to
`<video>_logs/tracking_profile_{forward,backward,session}.json` under a
`"spans"` key, and logged as a `SPAN TREE` block.

**`HYDRA_PROFILE=1`.** Arms a process-level recorder with no `TrackingProfiler`
required. Use it for two cases:

1. DetectKit / PoseKit, which build no `TrackingProfiler` at all.
2. Profiling a **User-mode** run without changing what it does. Debug Mode is
   not observation-only — it changes intermediate cleanup and CSV outputs — so
   "turn on Debug and re-run" profiles a different run than the one that was
   slow.

The dump lands in `<video>_logs/` when a session supplies one, otherwise
`$HYDRA_DATA_DIR/profiles/span_profile_<pid>.json`.

`HYDRA_RT_PROFILE` is kept as an alias for `HYDRA_PROFILE`.

## Reading the tree

Each node reports `total_s` (inclusive), `self_s` (inclusive minus direct
children), `n_calls`, `units`, `max_s` and `first_call_s`.

- **`self_s` localizes the defect.** High inclusive time with near-zero self
  time exonerates a stage and indicts its child.
- **`ms/unit` answers batch-size questions.** At `detection_batch_size=1` each
  window is one frame; at 25 the same work arrives in 1/25th the calls, so
  per-call overhead falls straight out of comparing the two runs.
- **`max_s` and `first_call_s` catch warmup.** A 5 s TensorRT engine build
  inside `backend_forward` (n=500) inflates the mean by 10 ms/call and is
  otherwise indistinguishable from a uniform 10 ms slowdown.
- **Percentages are of the parent, within a thread.** At depth≥2 summed span
  time legitimately exceeds wall-clock when threads overlap. Nodes on another
  thread are marked `concurrent`. A subtree that is 43% of its thread but 4% of
  the pass is both — do not read the first number as a speedup ceiling. This is
  the distortion behind the refuted SLEAP-batching premise: pose measured 4.6%
  of wall and batching returned ~0 end-to-end gain.

## Device time

The default profiled path does **not** synchronize. Spans are host wall-clock.
`torch.{cuda,mps}.synchronize()` is device-wide, and `pipeline_depth` defaults
to 2, so a sync on the consumer thread would drain the producer's in-flight OBB
kernels and bill OBB's device time to CNN.

`HYDRA_PROFILE_GPU=1` opts into a deep pass that syncs on GPU spans **and**
forces `pipeline_depth=1`, so there is no producer to contaminate. That run is
explicitly not the production schedule. The JSON stamps `"gpu_mode"` so nobody
compares the two.

| Mode | `total_s` means | GPU attribution |
|---|---|---|
| default | host cost under the production schedule | device work smears into whichever span later blocks |
| deep | host cost under a serialized depth=1 schedule | per-span device time, uncontaminated |

Per-span CUDA events would give device time without serializing, and would let
the deep pass keep depth=2. They are CUDA-only (no MPS equivalent) and are a
named future slice, not an oversight.

## What the span profiler cannot find

**Diffuse self-time defects.** A cross-cutting tax executed inside many small
operations — grad-mode toggling, measured at 26.5 s over 20 k calls — smears as
slightly-elevated `self_s` across dozens of spans and never aggregates into one
line. No span layout catches it. Use `cProfile`:

```bash
python -m cProfile -o /tmp/prof.out -m hydra_suite.trackerkit.cli track <args>
python -c "import pstats; pstats.Stats('/tmp/prof.out').sort_stats('tottime').print_stats(30)"
```

## Adding a span

1. Add the name to `src/hydra_suite/utils/profiling_names.py`. Never pass a
   string literal — `tests/utils/test_profiling_registry.py` fails on it, and a
   refactor that moved the function would otherwise silently drop the row.
2. Use `@spanned(NAME)` for a function boundary, `with span(NAME):` for a
   sub-function region.
3. **Wrap loops, never loop bodies.** A span inside a per-detection body runs
   5M times at 50 detections × 100k frames, measuring what the aggregate
   already reports. Per-detection cost comes from `units`.
4. If the code runs on its own thread, wrap the thread target in
   `bind_target(...)`. An unbound thread records nothing and the report shows
   that work costing zero.

The warp `ThreadPoolExecutor` (`stages/crops.py`) is deliberately **not**
bound: the only work in those workers is a per-detection body, and the parent
`apply_fit` span already bounds the pool's cost inclusively because the caller
blocks on it.
```

- [ ] **Step 2: Add the nav entry**

In `mkdocs.yml`, add `- Profiling: developer-guide/profiling.md` under the Developer Guide section, alongside the existing `runtime-integration.md` entry.

- [ ] **Step 3: Build the docs**

Run: `make docs-check`
Expected: strict build passes; terminology check clean.

- [ ] **Step 4: Commit**

```bash
git add docs/developer-guide/profiling.md mkdocs.yml
git commit -m "docs: span profiler guide"
```

---

### Task 13: Formatting, lint, and full-suite delta

**Files:** all touched above.

- [ ] **Step 1: Format**

Run: `make commit-prep`

- [ ] **Step 2: Lint**

Run: `make lint-moderate`
Expected: no new findings versus `main`. Compare with `git stash`-free method — run the same target on a clean `main` checkout if the count is ambiguous.

- [ ] **Step 3: Run the test suite per-file**

`pytest tests/` never finishes on this repo — a ClassKit modal-dialog hang plus a SIGABRT. Batch instead:

```bash
for f in $(git diff --name-only main -- tests/); do
  python -m pytest "$f" -q --timeout=300 || echo "FAILED: $f"
done
python -m pytest tests/test_tracking_profiler.py tests/core/tracking/ tests/utils/ -q --timeout=300
```

Expected: all new test files pass; pre-existing failures unchanged. Record the delta, not the absolute count — `main` carries ~24 known pre-existing failures.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "style: format span profiler changes"
```

---

### Task 14: MPS gates

**Files:** none modified — this task produces evidence.

**Prerequisite:** kill dead/stale **sleap/hydra** processes before any heavy run. Never interfere with a process that is not sleap/hydra.

```bash
pgrep -af "sleap|hydra" | grep -v "$$" | grep -v pgrep
```

(the `grep -v "$$"` matters — a bare `pgrep` self-matches and hangs a poll loop.)

- [ ] **Step 1: Establish the baseline BEFORE the change**

**Do not stash.** By this point Tasks 1-13 are all committed, so there is nothing to stash — and the stash stack is shared with the main checkout and every other worktree. The matrix's baseline is supplied by `MAIN_SRC` (the legacy worktree); nothing needs un-applying.

If a pre-change run is wanted for attribution, it should have been run from `main` (`09a14c33`) **before Task 1**, per CLAUDE.md's "run this BEFORE and AFTER a risky slice with the same baseline". If that was skipped, run it now from a detached checkout of `09a14c33` in a separate worktree rather than perturbing this one.

Setup, once per machine, with `conda activate hydra-mps` active:

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/span-profiling
bash tools/equivalence/fixtures/fetch_fixtures.sh
git fetch origin --tags
git worktree add --detach .worktrees/equiv-legacy legacy/main
```

- [ ] **Step 2: Kill stale processes**

```bash
pgrep -af "sleap|hydra" | grep -v pgrep
```

Kill only dead/stale **sleap/hydra** processes. Never interfere with a process that is not sleap/hydra. The `grep -v pgrep` matters — a bare `pgrep` self-matches and hangs a poll loop.

- [ ] **Step 3: Run the equivalence matrix with profiling ON (the default)**

All nine fixture configs set `"enable_profiling": true` and the default matrix runs eight of them (`ant_cnn_identity_marked` is excluded at `run_matrix.sh:56-75`), so the instrumented path is hot without any config edit:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_spans_on RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```

- [ ] **Step 4: Verify the CSVs are not empty before trusting any verdict**

```bash
find /tmp/equiv_spans_on -name "*.csv" -exec sh -c 'echo "$(wc -l < "$1") $1"' _ {} \;
```

Expected: every row count `> 1`. A bare shell (conda not active) yields EMPTY CSVs that falsely compare "EQUIVALENT". If conda was not active, re-run.

Acceptance: every clip's EQUIVALENCE at or near its DETERMINISM floor — positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched, for **both** `_forward.csv` and `_tracking_final.csv`. Known baseline noise: bistable head/tail π-flips on head/tail clips.

- [ ] **Step 5: Run the matrix with profiling OFF**

**Clear `enable_profiling`, NOT `debug_mode`.** Setting `debug_mode: false` derives `DEBUG_MODE=False`, which fires the User-mode cleanup at `session.py:619-637` and **deletes `_forward.csv` and `_tracking_final.csv`** via `_user_mode_intermediate_paths` (`session.py:102-113`) — the exact two files this gate compares. The code comment says so: *"NO-OP in debug mode (and thus a no-op for the equivalence gate)."* The comparison would then find nothing and report success. Clearing `enable_profiling` while leaving `debug_mode` absent keeps `DEBUG_MODE` at its `True` default, so the debug CSVs are still written, and disarms every span — which isolates exactly the variable this change introduces.

`run_matrix.sh` hardcodes config paths inside its `VIDEOS` array (`$FX/configs/*.json`, `FX="$WT/tools/equivalence/fixtures"`), so there is no config-directory knob. Edit the fixtures in place and revert afterwards:

```bash
python - <<'EOF'
import json, glob
for f in glob.glob("tools/equivalence/fixtures/configs/*.json"):
    d = json.load(open(f))
    d["enable_profiling"] = False
    json.dump(d, open(f, "w"), indent=2)
EOF

REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_spans_off RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh

git checkout -- tools/equivalence/fixtures/configs/
```

Verify row counts `> 1` again, then record the verdict. This is the constraint that matters most — byte-identical with profiling disarmed.

- [ ] **Step 6: Overhead measurement — N=5 alternating**

**Current src vs current src, `enable_profiling` true vs false.** One variable, same tree, same models. A single on/off pair cannot resolve the ≤2% target: this box has a measured ~30% wall-clock swing under load that once produced a bogus `1.65x SLOWER` verdict on a code path the change never touched.

`runner.py` has no `--enable-profiling` flag, so build two config copies. Its real arguments are `--orig-config` (required), `--video` (required), `--outdir` (required), plus `--runtime`, `--label`, `--skeleton`, `--detection-batch-size` (`runner.py:211-241`):

```bash
python - <<'EOF'
import json
src = "tools/equivalence/fixtures/configs/fly_obb.json"
for state in (True, False):
    d = json.load(open(src))
    d["enable_profiling"] = state
    json.dump(d, open(f"/tmp/fly_obb_{'on' if state else 'off'}.json", "w"), indent=2)
EOF

for i in 1 2 3 4 5; do
  for mode in on off; do
    /usr/bin/time -p python tools/equivalence/runner.py \
      --orig-config /tmp/fly_obb_${mode}.json \
      --video tools/equivalence/fixtures/clips/fly_obb.mp4 \
      --outdir /tmp/ovh_${mode}_${i} \
      --runtime mps --label ovh_${mode}_${i} 2>> /tmp/ovh.log
  done
done
```

Report **median and IQR per condition**. Pass criterion: the on/off median delta is ≤2% **and** smaller than the within-condition IQR. If the noise floor exceeds the effect, report "below noise floor" — that is a valid outcome, not a fake pass.

- [ ] **Step 7: The self-proving run**

Use **`ant_cnn_identity`**, the only fixture with all three consumers enabled. Verified against the configs:

| fixture | `enable_pose_extractor` | `pose_model_dir` | `cnn_classifiers` | headtail |
|---|---|---|---|---|
| `ant_obb_sleap` | `false` | — | `[]` | yes |
| `ant_cnn_identity` | `true` | SLEAP unet | 1 model | yes |

`is_pose_inference_enabled` (`session_policy.py:29`) requires both `enable_pose_extractor` and a non-empty `pose_model_dir`, so **`ant_obb_sleap` runs neither pose nor CNN** — two of the three `backend_forward` nodes would not exist there. Spec revision 2 named `ant_obb_sleap` on an inverted reading of the fixtures; revision 3 corrects it.

```bash
python tools/equivalence/runner.py \
  --orig-config tools/equivalence/fixtures/configs/ant_cnn_identity.json \
  --video tools/equivalence/fixtures/clips/ant_cnn_identity.mp4 \
  --outdir /tmp/sp_b1 --runtime mps \
  --skeleton tools/equivalence/fixtures/ooceraea_biroi.json

python tools/equivalence/runner.py \
  --orig-config tools/equivalence/fixtures/configs/ant_cnn_identity.json \
  --video tools/equivalence/fixtures/clips/ant_cnn_identity.mp4 \
  --outdir /tmp/sp_b25 --runtime mps --detection-batch-size 25 \
  --skeleton tools/equivalence/fixtures/ooceraea_biroi.json
```

Confirm the clip and skeleton filenames against `run_matrix.sh:56-75` before running; correct them if they differ.

- [ ] **Step 8: Assert the criterion is not vacuous, then evaluate it**

Machine-check that all three nodes exist with `n_calls > 0` before comparing — the vacuousness bug has now been shipped twice by hand:

```bash
python - <<'EOF'
import json, glob, sys

def find(node, path):
    for name in path:
        node = next((c for c in node["children"] if c["name"] == name), None)
        if node is None:
            return None
    return node

for out in ("/tmp/sp_b1", "/tmp/sp_b25"):
    f = glob.glob(f"{out}/**/tracking_profile_forward.json", recursive=True)
    assert f, f"no profile JSON under {out}"
    spans = json.load(open(f[0]))["spans"]
    for stage in ("headtail", "cnn", "pose"):
        node = find(spans, ["inference", "batch_pass", "window", stage, "backend_forward"])
        if node is None or node["n_calls"] == 0:
            sys.exit(f"VACUOUS: {stage}/backend_forward absent or n_calls=0 in {out}")
        print(out, stage, "n_calls=", node["n_calls"],
              "ms/unit=", round(node["total_s"] / max(node["units"], 1) * 1000, 3))
EOF
```

Adjust the span path if the tree shape differs (at `pipeline_depth=2` the detect-side spans root on the producer thread — see Task 9 Step 5).

**If the per-call overhead of a 1-frame window against 25-frame windows is not readable from those two JSON files alone, the profiler has not earned its keep. Report that as a failure, not a caveat.**

This is a **profiling experiment, not a byte-identity gate** — changing the window size changes decode and crop batching, so a tracking diff is expected and is not evidence of a profiler bug.

- [ ] **Step 9: Clean up the baseline worktree**

```bash
git worktree remove --force .worktrees/equiv-legacy && git worktree prune
git status --porcelain tools/equivalence/fixtures/configs/
```

The second command must print nothing — if Step 5's `git checkout --` was skipped, the fixture configs are still modified.

- [ ] **Step 10: Commit the evidence**

Write the four verdicts (profiling-ON equivalence, profiling-OFF equivalence, overhead median/IQR, self-proving comparison) into the spec under a new `## Gate results` section, and commit:

```bash
git add docs/superpowers/specs/2026-08-21-inference-span-profiling-design.md
git commit -m "docs: MPS gate results for the span profiler"
```

---

### Task 15: CUDA gate on mehek

**Files:** none modified — evidence only.

- [ ] **Step 1: Get the branch onto mehek**

```bash
git bundle create /tmp/span-profiling.bundle main..feat/inference-span-profiling
scp /tmp/span-profiling.bundle rutalab@mehek.taild08eb9.ts.net:/tmp/
```

- [ ] **Step 2: Set up on the box**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite
git fetch /tmp/span-profiling.bundle feat/inference-span-profiling:feat/inference-span-profiling
git checkout feat/inference-span-profiling
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh    # once
git fetch origin --tags
git worktree add --detach .worktrees/equiv-legacy legacy/main
```

- [ ] **Step 3: Kill stale processes, then run**

```bash
pgrep -af "sleap|hydra" | grep -v pgrep
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_spans_cuda RUNTIME=cuda \
  nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda.log 2>&1 &
```

Pose/SLEAP clips **require** the `sleap` conda env on the box and conda on PATH.

- [ ] **Step 4: Verify row counts, then read the verdict**

```bash
find /tmp/equiv_spans_cuda -name "*.csv" -exec sh -c 'echo "$(wc -l < "$1") $1"' _ {} \;
grep -E "EQUIVALENT|DIFFERENT|DETERMINISM|PERFORMANCE" /tmp/equiv_cuda.log
```

Expected: byte-identical on every clip, at the determinism floor.

- [ ] **Step 5: Confirm the deep-GPU path runs on CUDA**

Not a gate — the deep pass deliberately changes the schedule — but it must not crash:

```bash
HYDRA_PROFILE_GPU=1 python tools/equivalence/runner.py \
  --orig-config tools/equivalence/fixtures/configs/fly_obb.json \
  --video tools/equivalence/fixtures/clips/fly_obb.mp4 \
  --outdir /tmp/deepgpu --runtime cuda
python -c "
import json,glob
d=json.load(open(glob.glob('/tmp/deepgpu/**/tracking_profile_forward.json', recursive=True)[0]))
print('gpu_mode =', d['gpu_mode'])
assert d['gpu_mode'] == 'deep'
"
```

- [ ] **Step 6: Clean up and record**

```bash
git worktree remove --force .worktrees/equiv-legacy && git worktree prune
```

Add the CUDA verdict to the `## Gate results` section of the spec and commit.

---

### Task 16: Merge and docs lifecycle

**Files:**
- Move: `docs/superpowers/specs/2026-08-21-inference-span-profiling-design.md` → `docs/superpowers/specs/done/`
- Move: `docs/superpowers/plans/2026-08-22-inference-span-profiling.md` → `docs/superpowers/plans/done/`

- [ ] **Step 1: Verify every checkbox in this plan is checked**

If any step is unchecked, the docs stay active — CLAUDE.md's rule is explicit that an incomplete checklist keeps a doc out of `done/`.

- [ ] **Step 2: Move both docs, then merge, then stamp the SHA**

The `Shipped — merged to main (<sha>)` header cannot be written before the merge that creates that SHA exists — the `de7ed06e` convention it follows was a post-merge amendment. Order: move the docs, merge, then amend the header with the real SHA.

- [ ] **Step 3: Move both docs**

```bash
git mv docs/superpowers/specs/2026-08-21-inference-span-profiling-design.md docs/superpowers/specs/done/
git mv docs/superpowers/plans/2026-08-22-inference-span-profiling.md docs/superpowers/plans/done/
git commit -m "docs: move span profiler spec and plan to done/"
```

- [ ] **Step 4: Merge to local main — from the PRIMARY repo, not this worktree**

`main` is checked out in the primary repo, so `git checkout main` fails inside this worktree. Change directory first:

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git merge --no-ff feat/inference-span-profiling -m "Merge branch 'feat/inference-span-profiling'"
```

- [ ] **Step 4b: Stamp the merge SHA into the spec header**

```bash
SHA=$(git rev-parse --short HEAD)
sed -i '' "s/\*\*Status:\*\* pending implementation plan/**Status:** Shipped — merged to main ($SHA)/" \
  docs/superpowers/specs/done/2026-08-21-inference-span-profiling-design.md
git add -A && git commit --amend --no-edit
```

- [ ] **Step 5: Write the memory file**

Create `~/.claude/projects/-Users-neurorishika-Projects-Rockefeller-Kronauer-multi-animal-tracker/memory/project_span_profiling_done.md` recording: the merge SHA, both gate verdicts, the overhead median/IQR, the self-proving result, the two spec corrections (no `SessionRunner`; warp pool deliberately unbound), and that `HYDRA_PROFILE=1` is the User-mode profiling route. Add the one-line pointer to `MEMORY.md`.

- [ ] **Step 6: Clean up the worktree — from the primary repo**

You must already be in the primary repo (Step 4). Removing the worktree while sitting inside it fails:

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git worktree remove .worktrees/span-profiling
git branch -d feat/inference-span-profiling
```
