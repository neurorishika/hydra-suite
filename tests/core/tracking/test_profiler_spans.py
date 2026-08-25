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
