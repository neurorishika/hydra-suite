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
