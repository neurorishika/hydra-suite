"""Regression tests for the SLEAP inference service's per-request GC cadence.

Root cause (2026-07-23): the embedded HTTP service ran ``gc.collect()`` in the
``finally`` of every ``/infer`` request. The server is single-threaded, so under
per-frame streaming pose (~1 call/frame) each request blocked on the previous
request's full-heap collection -- ~46 ms/call, ~24 s over a 500-frame clip, i.e.
4x the actual inference cost. The fix makes the sweep periodic (env-tunable via
``HYDRA_SLEAP_GC_EVERY``, default 50) and byte-identical in output.

These tests lock in the periodic behaviour so a future edit cannot silently
reintroduce a per-request collection.
"""

import re

from hydra_suite.integrations.sleap import service


def _embedded_code() -> str:
    code = service._SLEAP_SERVICE_CODE
    assert (
        "def do_POST" in code
    ), "embedded service code must define the request handler"
    return code


def test_embedded_service_code_is_valid_python():
    import ast

    ast.parse(_embedded_code())  # raises SyntaxError on regression


def test_do_post_does_not_collect_every_request():
    """The do_POST finally must delegate to _maybe_gc(), never a bare gc.collect()."""
    code = _embedded_code()
    # Isolate the do_POST method body (up to the next method definition).
    start = code.index("def do_POST")
    end = code.index("def log_message", start)
    do_post = code[start:end]
    assert "_maybe_gc()" in do_post, "do_POST should call the periodic _maybe_gc()"
    assert "gc.collect()" not in do_post, (
        "do_POST must not call gc.collect() on every request -- that reintroduces "
        "the per-request GC stall (~24s/clip). Use _maybe_gc()."
    )
    assert "HYDRA_SLEAP_GC_EVERY" in code, "GC cadence must remain env-tunable"


def _extract_maybe_gc(code: str):
    """Exec just the cadence init + _maybe_gc def in an isolated namespace."""
    m = re.search(
        r"(try:\s*\n\s+_GC_EVERY.*?def _maybe_gc\(\):.*?gc\.collect\(\)\n)", code, re.S
    )
    assert m, "could not locate the _GC_EVERY/_maybe_gc block in embedded code"
    return m.group(1)


class _FakeGC:
    def __init__(self):
        self.collects = 0

    def collect(self):
        self.collects += 1


def _run_cadence(every: int, calls: int):
    import os

    snippet = _extract_maybe_gc(_embedded_code())
    fake = _FakeGC()
    ns = {"os": os, "gc": fake}
    os.environ["HYDRA_SLEAP_GC_EVERY"] = str(every)
    try:
        exec(snippet, ns)  # defines _GC_EVERY, _req_count, _maybe_gc
    finally:
        os.environ.pop("HYDRA_SLEAP_GC_EVERY", None)
    for _ in range(calls):
        ns["_maybe_gc"]()
    return fake.collects, ns


def test_maybe_gc_collects_periodically():
    collects, ns = _run_cadence(every=50, calls=500)
    assert ns["_GC_EVERY"] == 50
    assert (
        collects == 10
    ), f"expected 10 collections over 500 calls at cadence 50, got {collects}"


def test_maybe_gc_cadence_is_env_tunable_and_never_zero():
    # Explicit small cadence honoured.
    assert _run_cadence(every=1, calls=7)[0] == 7  # every-request still available
    # Guard against a pathological 0 -> clamped to >=1 (no ZeroDivisionError).
    collects, ns = _run_cadence(every=0, calls=5)
    assert ns["_GC_EVERY"] == 1
    assert collects == 5
