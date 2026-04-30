from __future__ import annotations

import logging

from hydra_suite.core.tracking.profiler import TrackingProfiler


def test_log_periodic_includes_interval_phase_breakdown(caplog) -> None:
    profiler = TrackingProfiler(enabled=True)

    for _frame_idx in range(2):
        profiler.add_sample("frame_resize", 0.002)
        profiler.add_sample("roi_prepare", 0.001)
        profiler.add_sample("live_pose_transport", 0.002, work_units=2)
        profiler.add_sample("live_pose_inference", 0.003, work_units=2)
        profiler.add_sample("live_pose_postprocess", 0.001, work_units=2)
        profiler.add_phase_time("pose_transport", 0.002, work_units=2)
        profiler.add_phase_time("pose_inference", 0.003, work_units=2)
        profiler.add_phase_time("pose_postprocess", 0.001, work_units=2)
        profiler.end_frame()

    with caplog.at_level(logging.INFO, logger="hydra_suite.core.tracking.profiler"):
        profiler.log_periodic(interval=2)

    assert "=== PROFILING SUMMARY (last 2 frames) ===" in caplog.text
    assert "PHASE TIMING" in caplog.text
    assert "frame_resize" in caplog.text
    assert "roi_prepare" in caplog.text
    assert "pose_transport" in caplog.text
    assert "pose_inference" in caplog.text
    assert "pose_postprocess" in caplog.text
    assert "ms/individual" in caplog.text
