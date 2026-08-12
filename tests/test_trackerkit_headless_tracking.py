"""Slice 4: headless_tracking drives TrackingEngineCore + TrackingSessionCore, no Qt."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

import hydra_suite.trackerkit.headless_tracking as ht
from hydra_suite.core.tracking.session import SessionResult
from hydra_suite.trackerkit.cli_config import TrackerCliSession, TrackerCliVideoProbe


def test_headless_tracking_module_imports_no_qt():
    """The CLI runtime path must not import PySide6/QtCore at any depth."""
    src = Path(ht.__file__).read_text()
    tree = ast.parse(src, filename=ht.__file__)
    offenders = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            mod = node.module
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        if mod and any(q in mod for q in ("PySide6", "QtCore", "QtWidgets", "QtGui")):
            offenders.append(f"{mod}:{node.lineno}")
    assert not offenders, "headless_tracking must be Qt-free: " + "; ".join(offenders)


def _make_session(tmp_path, **overrides) -> TrackerCliSession:
    base = dict(
        video_path=str(tmp_path / "video.mp4"),
        config_path=None,
        video_probe=TrackerCliVideoProbe(
            fps=30.0, total_frames=10, width=64, height=64
        ),
        config={},
        raw_csv_path=str(tmp_path / "video_tracking.csv"),
        final_csv_path=str(tmp_path / "video_tracking_forward_processed.csv"),
        params={"FPS": 30.0},
        save_confidence_metrics=False,
        use_cached_detections=False,
        enable_backward_tracking=False,
        enable_postprocessing=True,
        interpolation_method="None",
        interpolation_max_gap_seconds=0.0,
        heading_flip_max_burst=5,
        identity_method="none_disabled",
        enable_pose_extractor=False,
    )
    base.update(overrides)
    return TrackerCliSession(**base)


def test_forward_only_drives_engine_then_session(monkeypatch, tmp_path):
    class _CachePlan:
        inference_model_id = "bgsub_test"
        engine_model_id = None
        detection_cache_path = str(tmp_path / "cache.npz")

    monkeypatch.setattr(ht, "plan_tracking_cache", lambda *a, **k: _CachePlan())

    calls = {"engine_passes": [], "post": None}

    def _fake_engine_pass(
        session,
        *,
        params,
        raw_csv_path,
        backward_mode,
        detection_cache_path,
        use_cached_detections,
        should_stop,
    ):
        calls["engine_passes"].append(backward_mode)
        assert params["INFERENCE_MODEL_ID"] == "bgsub_test"
        return True, [30.0, 30.0], pd.DataFrame({"TrajectoryID": [0], "X": [1]}), {}

    monkeypatch.setattr(ht, "_run_engine_pass", _fake_engine_pass)

    def _fake_run_post(self, forward_trajectories, backward_trajectories=None):
        calls["post"] = (forward_trajectories, backward_trajectories)
        return SessionResult(
            success=True,
            final_csv_path=str(tmp_path / "out_final.csv"),
            rich_export_path=None,
            media_paths=[],
            dataset_result=None,
            summary_lines=["final_csv=out_final.csv"],
            error=None,
        )

    monkeypatch.setattr(
        "hydra_suite.core.tracking.session.TrackingSessionCore.run_post_tracking",
        _fake_run_post,
    )

    session = _make_session(tmp_path)
    result = ht.run_headless_tracking_session(session)

    assert result["success"] is True
    assert calls["engine_passes"] == [False]  # forward only
    assert calls["post"][1] is None  # no backward df
    assert any("avg_fps=30.0" in line for line in result["lines"])


def test_backward_enabled_runs_two_passes(monkeypatch, tmp_path):
    class _CachePlan:
        inference_model_id = "m"
        engine_model_id = None
        detection_cache_path = str(tmp_path / "cache.npz")

    monkeypatch.setattr(ht, "plan_tracking_cache", lambda *a, **k: _CachePlan())

    seen = {"passes": []}

    def _fake_engine_pass(
        session,
        *,
        params,
        raw_csv_path,
        backward_mode,
        detection_cache_path,
        use_cached_detections,
        should_stop,
    ):
        seen["passes"].append((backward_mode, use_cached_detections))
        return True, [30.0], pd.DataFrame({"TrajectoryID": [0]}), {}

    monkeypatch.setattr(ht, "_run_engine_pass", _fake_engine_pass)

    def _fake_run_post(self, forward_trajectories, backward_trajectories=None):
        assert backward_trajectories is not None  # backward df threaded through
        return SessionResult(True, str(tmp_path / "f.csv"), None, [], None, [], None)

    monkeypatch.setattr(
        "hydra_suite.core.tracking.session.TrackingSessionCore.run_post_tracking",
        _fake_run_post,
    )

    result = ht.run_headless_tracking_session(
        _make_session(tmp_path, enable_backward_tracking=True)
    )
    assert result["success"] is True
    # forward pass first (uses cache flag from session), backward pass forces no-cache.
    assert seen["passes"] == [(False, False), (True, False)]


def test_forward_failure_short_circuits_before_session(monkeypatch, tmp_path):
    class _CachePlan:
        inference_model_id = "m"
        engine_model_id = None
        detection_cache_path = str(tmp_path / "cache.npz")

    monkeypatch.setattr(ht, "plan_tracking_cache", lambda *a, **k: _CachePlan())
    monkeypatch.setattr(
        ht,
        "_run_engine_pass",
        lambda *a, **k: (False, [], None, {}),
    )

    def _must_not_run(*a, **k):
        raise AssertionError("session must not run after forward failure")

    monkeypatch.setattr(
        "hydra_suite.core.tracking.session.TrackingSessionCore.run_post_tracking",
        _must_not_run,
    )

    result = ht.run_headless_tracking_session(_make_session(tmp_path))
    assert result["success"] is False
    assert "forward" in result["error"].lower()
