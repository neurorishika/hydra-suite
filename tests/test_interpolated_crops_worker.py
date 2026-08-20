"""Interpolated-crops pipeline tests.

The implementation moved out of ``trackerkit/gui/workers/crops_worker.py`` into
the Qt-free ``hydra_suite.core.post.interpolated_crops`` module; the worker is
now a thin wrapper around ``run_interpolated_crops``. These tests therefore
target the core module's functions directly.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry, ClippingStats
from hydra_suite.core.post import interpolated_crops as ic


def test_interpolated_worker_skips_backend_init_when_no_eligible_gaps(
    monkeypatch,
) -> None:
    finalized = {"called": False}

    class FakeCap:
        def release(self) -> None:
            return None

    class FakeGenerator:
        def finalize(self) -> None:
            finalized["called"] = True

    monkeypatch.setattr(
        ic,
        "_validate_and_setup",
        lambda csv_path, video_path, cache_path, params, profiler, should_stop: (
            pd.DataFrame(
                [
                    {
                        "TrajectoryID": 1,
                        "FrameID": 0,
                        "State": "active",
                        "X": 0.0,
                        "Y": 0.0,
                        "Theta": 0.0,
                    }
                ]
            ),
            FakeCap(),
            None,
            FakeGenerator(),
            "unused-output-dir",
            False,
            True,
            1.0,
            1.0,
            CanonicalGeometry.from_reference(20.0, 2.0, 1.3),
        ),
    )
    monkeypatch.setattr(
        ic,
        "_detect_interpolation_gaps",
        lambda params, should_stop, df, detection_cache, position_scale, size_scale: (
            {},
            4,
            0,
            0,
        ),
    )
    monkeypatch.setattr(
        ic,
        "_init_interpolation_backends",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("backend initialization should be skipped")
        ),
    )
    monkeypatch.setattr(ic, "_cleanup_backends", lambda *args, **kwargs: None)

    result = ic.run_interpolated_crops("tracks.csv", "source.mp4", "cache.npz", {})

    assert finalized["called"] is True
    assert result["no_work_reason"] == "no_eligible_gaps"
    assert result["occluded_rows"] == 4
    assert result["eligible_frames"] == 0
    assert result["eligible_rows"] == 0
    assert result["pose_rows_produced"] == 0
    assert result["cnn_rows_produced"] == 0


def test_init_interpolation_backends_delegates_to_shared_stage_loaders(
    monkeypatch,
    tmp_path,
) -> None:
    """Golden rule (Task 9): pose/CNN/head-tail backends route through the
    SAME per-stage loaders ``Pipeline`` uses (``load_pose_model``/
    ``load_cnn_model``/``load_headtail_model``), threaded from the single
    resolved ``RuntimeContext`` -- not a divergent hand-rolled runtime-flavor
    ladder (``_init_cnn_backends``/``_init_headtail_analyzer``/
    ``_init_pose_backend``, all retired by Task 9/12; superseded by this
    test). Params -> ``PoseConfig``/``CNNConfig``/``HeadTailConfig`` field
    translation (e.g. ``POSE_SLEAP_ENV``) is covered separately by
    ``test_inference_config_from_params.py``; this test only asserts
    ``_init_interpolation_backends`` calls the shared loaders with those
    configs and the shared runtime, and returns what they return."""
    pose_module = importlib.import_module("hydra_suite.core.inference.stages.pose")
    cnn_module = importlib.import_module("hydra_suite.core.inference.stages.cnn")
    headtail_module = importlib.import_module(
        "hydra_suite.core.inference.stages.headtail"
    )

    cnn_model = tmp_path / "cnn_model.pth"
    cnn_model.write_text("cnn", encoding="utf-8")
    headtail_model = tmp_path / "headtail_model.pt"
    headtail_model.write_text("ht", encoding="utf-8")
    pose_model_path = tmp_path / "yolo_pose.pt"
    pose_model_path.write_text("pose", encoding="utf-8")

    captured: dict[str, object] = {}
    fake_pose_model = object()
    fake_cnn_model = object()
    fake_headtail_model = object()

    def _fake_load_pose_model(config, runtime, **kwargs):
        captured["pose_config"] = config
        captured["pose_runtime"] = runtime
        captured["pose_out_root"] = kwargs.get("out_root")
        return fake_pose_model

    def _fake_load_cnn_model(config, runtime):
        captured["cnn_config"] = config
        captured["cnn_runtime"] = runtime
        return fake_cnn_model

    def _fake_load_headtail_model(config, runtime):
        captured["headtail_config"] = config
        captured["headtail_runtime"] = runtime
        return fake_headtail_model

    monkeypatch.setattr(pose_module, "load_pose_model", _fake_load_pose_model)
    monkeypatch.setattr(cnn_module, "load_cnn_model", _fake_load_cnn_model)
    monkeypatch.setattr(
        headtail_module, "load_headtail_model", _fake_load_headtail_model
    )

    # Gen-2: the pipeline resolves the single RUNTIME_TIER through
    # RuntimeResolver and threads ONE RuntimeContext to every stage loader.
    # The "cpu" tier resolves to native torch/CPU on every host, independent
    # of platform accelerators.
    params = {
        "ENABLE_POSE_EXTRACTOR": True,
        "POSE_MODEL_TYPE": "yolo",
        "POSE_MODEL_DIR": str(pose_model_path),
        "CNN_CLASSIFIERS": [
            {"label": "cnn_identity", "model_path": str(cnn_model), "batch_size": 4}
        ],
        "RUNTIME_TIER": "cpu",
        "YOLO_HEADTAIL_MODEL_PATH": str(headtail_model),
        "USE_APRILTAGS": False,
    }
    geometry = ic.canonical_geometry_from_params(params)

    (
        cfg,
        runtime,
        pose_model,
        apriltag_model,
        cnn_models,
        cnn_labels,
        headtail_model_result,
    ) = ic._init_interpolation_backends(params, str(tmp_path), geometry)

    assert pose_model is fake_pose_model
    assert cnn_models == [fake_cnn_model]
    assert cnn_labels == ["cnn_identity"]
    assert headtail_model_result is fake_headtail_model
    assert apriltag_model is None

    # Every stage loader received the SAME resolved RuntimeContext -- the one
    # source of tier -> backend/device truth, not a per-stage re-derivation.
    assert captured["pose_runtime"] is runtime
    assert captured["cnn_runtime"] is runtime
    assert captured["headtail_runtime"] is runtime
    assert captured["pose_out_root"] == str(tmp_path)
    assert captured["pose_config"] is cfg.pose
    assert captured["cnn_config"] is cfg.cnn_phases[0]
    assert captured["headtail_config"] is cfg.headtail


def test_crops_worker_has_no_divergent_flavor_ladder() -> None:
    """Source guard: the deleted runtime-flavor ladder must not reappear."""
    from pathlib import Path as _Path

    for module_name in (
        "hydra_suite.trackerkit.gui.workers.crops_worker",
        "hydra_suite.core.post.interpolated_crops",
    ):
        src = _Path(importlib.import_module(module_name).__file__).read_text(
            encoding="utf-8"
        )
        for banned in (
            "is_cuda_like",
            "onnx_cuda",
            "create_pose_backend_from_config",
            "YoloNativeBackend",
        ):
            assert (
                banned not in src
            ), f"divergent pose ladder token still present in {module_name}: {banned}"


def test_filter_degenerate_and_get_corners_accumulates_clipping_stats() -> None:
    """GAP 1: the interpolated-crops path shares the same run-scoped
    ClippingStats accumulator/reporting pattern as InferenceRunner/Pipeline
    (core/tracking/worker.py's end-of-run summary), instead of discarding the
    ``clipped`` flag returned by ``canonical_affine``.
    """
    geometry = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)

    clipping_stats = ClippingStats()
    assert clipping_stats.clipped_count == 0
    assert clipping_stats.total_count == 0

    # A task whose OBB is far larger than the canonical canvas overflows it.
    clipped_task = {"cx": 0.0, "cy": 0.0, "w": 500.0, "h": 250.0, "theta": 0.0}
    # A task sized to the reference body fits comfortably within the canvas.
    fitting_task = {"cx": 0.0, "cy": 0.0, "w": 20.0, "h": 10.0, "theta": 0.0}

    ic._filter_degenerate_and_get_corners(
        [clipped_task, fitting_task], geometry, clipping_stats
    )

    assert clipping_stats.total_count == 2
    assert clipping_stats.clipped_count == 1
    assert clipping_stats.worst_overflow_ratio > 1.0
    summary = clipping_stats.summary()
    assert summary is not None
    assert "1/2" in summary


def test_filter_degenerate_and_get_corners_counts_degenerate_drops() -> None:
    """A degenerate OBB is DROPPED here -- filtered out of ``kept_tasks``
    entirely (Task 12 delegates to ``synthetic_detections.filter_degenerate_
    tasks``, which returns kept tasks only rather than a ``None`` placeholder
    affine) -- so it must be counted on the shared ClippingStats rather than
    making the run quietly shorter (spec 2026-08-18).
    """
    geometry = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    clipping_stats = ClippingStats()

    # w=h=0 -> every OBB corner coincides -> canonical_affine raises.
    degenerate_task = {
        "cx": 5.0,
        "cy": 5.0,
        "w": 0.0,
        "h": 0.0,
        "theta": 0.0,
        "frame_id": 0,
        "traj_id": 1,
    }
    fitting_task = {
        "cx": 0.0,
        "cy": 0.0,
        "w": 20.0,
        "h": 10.0,
        "theta": 0.0,
        "frame_id": 0,
        "traj_id": 2,
    }

    kept_tasks, corners = ic._filter_degenerate_and_get_corners(
        [degenerate_task, fitting_task], geometry, clipping_stats
    )

    assert kept_tasks == [fitting_task]
    assert len(corners) == 1
    assert clipping_stats.degenerate_skipped_count == 1
    # The dropped task is not recorded as a normal (clipped/total) detection.
    assert clipping_stats.total_count == 1
    assert clipping_stats.clipped_count == 0

    summary = clipping_stats.summary()
    assert summary is not None and "DEGENERATE" in summary


def _run_with_no_eligible_gaps(monkeypatch) -> None:
    """Drive ``run_interpolated_crops`` through the early-return
    ("no eligible gaps") path used by
    ``test_interpolated_worker_skips_backend_init_when_no_eligible_gaps``
    above. The reset of the degenerate-padding warning happens before that
    early return, so this still exercises the real reset call inside the
    real entry point without needing a full pipeline run."""

    class FakeCap:
        def release(self) -> None:
            return None

    class FakeGenerator:
        def finalize(self) -> None:
            return None

    monkeypatch.setattr(
        ic,
        "_validate_and_setup",
        lambda csv_path, video_path, cache_path, params, profiler, should_stop: (
            pd.DataFrame(
                [
                    {
                        "TrajectoryID": 1,
                        "FrameID": 0,
                        "State": "active",
                        "X": 0.0,
                        "Y": 0.0,
                        "Theta": 0.0,
                    }
                ]
            ),
            FakeCap(),
            None,
            FakeGenerator(),
            "unused-output-dir",
            False,
            True,
            1.0,
            1.0,
            CanonicalGeometry.from_reference(20.0, 2.0, 1.3),
        ),
    )
    monkeypatch.setattr(
        ic,
        "_detect_interpolation_gaps",
        lambda params, should_stop, df, detection_cache, position_scale, size_scale: (
            {},
            4,
            0,
            0,
        ),
    )
    monkeypatch.setattr(
        ic,
        "_init_interpolation_backends",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("backend initialization should be skipped")
        ),
    )
    monkeypatch.setattr(ic, "_cleanup_backends", lambda *args, **kwargs: None)

    ic.run_interpolated_crops("tracks.csv", "source.mp4", "cache.npz", {})


def test_run_interpolated_crops_rearms_degenerate_padding_warning_each_run(
    monkeypatch, caplog
) -> None:
    """The decisive regression test for the once-per-process -> once-per-run
    fix: in a long-lived process that calls ``run_interpolated_crops``
    repeatedly (e.g. the GUI running tracking/interpolation more than once),
    the degenerate-padding warning must be able to fire again on EVERY run,
    not just the first one ever. Driven entirely through the real entry
    point -- never resets the module flag by hand."""
    from hydra_suite.core.tracking.pose import pose_pipeline as pp

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    degenerate_corners = np.array(
        [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float32
    )

    import logging

    with caplog.at_level(logging.WARNING, logger=pp.logger.name):
        # Run 1: real entry point resets the guard, then a degenerate crop
        # attempt fires the warning.
        _run_with_no_eligible_gaps(monkeypatch)
        result1 = pp.extract_one_crop(
            frame, degenerate_corners, 0, -0.9, [degenerate_corners], False, (0, 0, 0)
        )
        assert result1 is None
        warnings_after_run1 = [
            r for r in caplog.records if "AprilTag crop padding" in r.message
        ]
        assert len(warnings_after_run1) == 1

        # Same run, another degenerate detection: must NOT warn again.
        result1b = pp.extract_one_crop(
            frame, degenerate_corners, 1, -0.9, [degenerate_corners], False, (0, 0, 0)
        )
        assert result1b is None
        assert (
            len([r for r in caplog.records if "AprilTag crop padding" in r.message])
            == 1
        )

        # Run 2 (same process): the real entry point must re-arm the guard so
        # the warning can fire again -- this is the behavior that was broken.
        _run_with_no_eligible_gaps(monkeypatch)
        result2 = pp.extract_one_crop(
            frame, degenerate_corners, 0, -0.9, [degenerate_corners], False, (0, 0, 0)
        )
        assert result2 is None
        warnings_after_run2 = [
            r for r in caplog.records if "AprilTag crop padding" in r.message
        ]
        assert len(warnings_after_run2) == 2


def test_filter_degenerate_and_get_corners_tolerates_no_stats() -> None:
    """The accumulator is optional; a degenerate task must not blow up when
    the caller passed no ClippingStats, and is simply absent from the kept
    tasks (Task 12's ``filter_degenerate_tasks`` delegation)."""
    geometry = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    kept_tasks, corners = ic._filter_degenerate_and_get_corners(
        [
            {
                "cx": 0.0,
                "cy": 0.0,
                "w": 0.0,
                "h": 0.0,
                "theta": 0.0,
                "frame_id": 0,
                "traj_id": 1,
            }
        ],
        geometry,
        None,
    )
    assert kept_tasks == []
    assert corners == []


# NOTE: the old ``test_pose_fit_failure_skips_detection`` (F6: a Layer 2
# ``apply_fit`` failure must drop the detection rather than feed the backend
# an un-fit crop) tested ``interpolated_crops.py``'s own hand-rolled
# ``_flush_pose_batch``, which Task 10 deleted -- pose inference now runs
# through ``inference/stages/pose.py::run_pose_batch``, the SAME function
# ``Pipeline`` calls for real detections, so this module has no equivalent
# function left to test. Coverage of ``run_pose_batch``'s own Layer 2
# fit-failure handling belongs in ``tests/test_inference_stages_pose.py``,
# not here; it does not currently exist there, and observed there today,
# ``run_pose_batch`` catches an ``apply_fit`` exception and still feeds the
# backend the un-fit crop (``except Exception: pass`` then appends anyway) --
# a change from this module's old drop-the-detection behaviour. That is a
# pre-existing Task 10 behaviour, out of Task 12's wiring-only scope; flagged
# in the Task 12 report as a candidate follow-up, not fixed here.
