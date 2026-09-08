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

    assert "--- PROFILING [last 2 frames] ---" in caplog.text
    assert "PHASE BREAKDOWN" in caplog.text
    assert "frame_resize" in caplog.text
    assert "roi_prepare" in caplog.text
    assert "pose_transport" in caplog.text
    assert "pose_inference" in caplog.text
    assert "pose_postprocess" in caplog.text
    assert "ms/individual" in caplog.text


def test_initialization_phase_closes_after_model_load_and_autotune_preflight():
    """The `initialization` phase must end after setup, not before it.

    `steady = wall - initialization - cleanup` is consumed by the inference
    autotuner both inside the calibration child
    (`sidecar_child._profile_times`) and for production throughput regression
    detection.  When `phase_end("initialization")` fired before the
    InferenceRunner constructions and the autotune preflight, every measured
    window's steady time contained a full model load, and a production run's
    steady time absorbed the whole calibration budget -- which tripped the
    0.85 regression rule and demoted the profile that had just been validated,
    causing a re-tune loop.

    There is no unit seam for this ordering, so assert it on the source: the
    boundary must appear after the last `InferenceRunner(` construction and
    after the autotune preflight call.
    """

    from pathlib import Path

    import hydra_suite.core.tracking.worker as worker_module

    whole = Path(worker_module.__file__).read_text(encoding="utf-8")
    # Scope to run_tracking: `InferenceRunner(` also appears in docstrings and
    # comments further down the file, and a future helper added below this
    # method must not break the assertion.
    body_start = whole.index('profiler.phase_start("initialization")')
    body_end = whole.index('profiler.phase_start("tracking_loop")')
    source = whole[body_start:body_end]

    boundary = source.index('profiler.phase_end("initialization")')

    constructions = [
        source.rindex("inference_runner = InferenceRunner("),
        source.rindex("bgsub_runner = InferenceRunner("),
    ]
    assert max(constructions) < boundary
    assert source.index(") = _autotune_session.lookup(") < boundary
    # ... and still before the first measured phase.
    assert boundary < source.index('profiler.phase_start("batched_detection")')
