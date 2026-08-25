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
    outer = SpanRecorder(priority=profiling.PRIORITY_PROCESS)
    inner = SpanRecorder(priority=profiling.PRIORITY_SESSION)
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
