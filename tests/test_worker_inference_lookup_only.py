"""A tracking run applies profiles. It never measures them."""

import pathlib


def test_worker_never_constructs_a_trial_executor():
    src = pathlib.Path("src/hydra_suite/core/tracking/worker.py").read_text(
        encoding="utf-8"
    )
    assert "ContainedTrialExecutor" not in src
    assert "SidecarTrialSpec" not in src


def test_worker_calls_session_lookup_not_a_local_resolver():
    src = pathlib.Path("src/hydra_suite/core/tracking/worker.py").read_text(
        encoding="utf-8"
    )
    assert "_resolve_inference_autotune_before_load" not in src
    assert "session.lookup" in src or "autotune_session.lookup" in src


def test_profiler_enabling_no_longer_depends_on_a_mode_string():
    """observe_production_throughput is gated on profiler.enabled and writes the
    measured-key record the S2 bridge needs. If the gate reads a key that no
    longer exists, the bridge dies silently."""
    src = pathlib.Path("src/hydra_suite/core/tracking/worker.py").read_text(
        encoding="utf-8"
    )
    assert "INFERENCE_AUTOTUNE_MODE" not in src
    assert "APPLY_TUNED_INFERENCE" in src


def test_cancel_inference_autotune_is_gone():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    assert not hasattr(TrackingEngineCore, "cancel_inference_autotune")
