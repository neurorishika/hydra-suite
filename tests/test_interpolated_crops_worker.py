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
from hydra_suite.runtime.resolver import ResolvedBackend


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


def test_interpolated_worker_uses_split_cnn_and_headtail_runtimes(
    monkeypatch,
    tmp_path,
) -> None:
    cnn_module = importlib.import_module(
        "hydra_suite.core.individual.classification.cnn"
    )
    headtail_module = importlib.import_module(
        "hydra_suite.core.individual.classification.headtail"
    )

    cnn_model = tmp_path / "cnn_model.pth"
    cnn_model.write_text("cnn", encoding="utf-8")
    headtail_model = tmp_path / "headtail_model.pt"
    headtail_model.write_text("ht", encoding="utf-8")

    observed: dict[str, object] = {}

    class FakeCNNConfig:
        def __init__(
            self,
            model_path: str,
            confidence: float,
            batch_size: int,
            scoring_mode: str = "atomic",
        ) -> None:
            self.model_path = model_path
            self.confidence = confidence
            self.batch_size = batch_size
            self.scoring_mode = scoring_mode

    class FakeCNNBackend:
        def __init__(
            self, config, model_path: str | None = None, resolved=None
        ) -> None:
            observed["cnn_resolved"] = resolved

    class FakeHeadTailAnalyzer:
        def __init__(self, model_path: str, resolved=None, **kwargs) -> None:
            observed["headtail_resolved"] = resolved
            observed["headtail_geometry"] = kwargs.get("geometry")
            self.is_available = True

        def close(self) -> None:
            return None

    monkeypatch.setattr(cnn_module, "CNNIdentityConfig", FakeCNNConfig)
    monkeypatch.setattr(cnn_module, "CNNIdentityBackend", FakeCNNBackend)
    monkeypatch.setattr(headtail_module, "HeadTailAnalyzer", FakeHeadTailAnalyzer)

    # Gen-2: the pipeline resolves the single RUNTIME_TIER through
    # RuntimeResolver and threads a ResolvedBackend to both the CNN and
    # head-tail stages. The "cpu" tier resolves to native torch/CPU on every
    # host, independent of platform accelerators.
    params = {
        "CNN_CLASSIFIERS": [
            {"label": "cnn_identity", "model_path": str(cnn_model), "batch_size": 4}
        ],
        "RUNTIME_TIER": "cpu",
        "YOLO_HEADTAIL_MODEL_PATH": str(headtail_model),
    }
    geometry = ic.canonical_geometry_from_params(params)

    ic._init_cnn_backends(params)
    ic._init_headtail_analyzer(params, geometry)

    assert observed["cnn_resolved"] == ResolvedBackend("torch", "cpu", False)
    assert observed["headtail_resolved"] == ResolvedBackend("torch", "cpu", False)
    # The head-tail analyzer must share the ONE project-wide canvas, not build
    # its own from a bare aspect ratio.
    assert observed["headtail_geometry"] == geometry


def test_init_pose_backend_yolo_delegates_to_load_pose_backend(
    monkeypatch, tmp_path
) -> None:
    """Golden rule: the YOLO pose branch routes through the shared
    ``core/inference/api.load_pose_backend`` shim (patched here in the
    pipeline's module) instead of duplicating the runtime-flavor ladder."""
    pose_utils = importlib.import_module("hydra_suite.core.individual.pose.utils")

    monkeypatch.setattr(
        pose_utils, "load_skeleton_from_json", lambda _p: (["kpt0", "kpt1"], [])
    )

    captured: dict[str, object] = {}

    class FakeBackend:
        output_keypoint_names = ["kpt0", "kpt1"]

        def __init__(self) -> None:
            self.warmup_calls = 0

        def warmup(self) -> None:
            self.warmup_calls += 1
            captured["warmed_up"] = True

    fake_backend = FakeBackend()

    def _fake_load_pose_backend(**kwargs):
        captured.update(kwargs)
        return fake_backend

    monkeypatch.setattr(ic, "load_pose_backend", _fake_load_pose_backend)

    backend, kpt_source_names, kpt_labels = ic._init_pose_backend(
        {
            "ENABLE_POSE_EXTRACTOR": True,
            "POSE_MODEL_TYPE": "yolo",
            "POSE_MODEL_DIR": "/models/yolo_pose.pt",
            "POSE_MIN_KPT_CONF_VALID": 0.3,
            "POSE_BATCH_SIZE": 8,
            "RUNTIME_TIER": "cpu",
        },
        str(tmp_path),
    )

    assert backend is fake_backend
    assert captured["backend_family"] == "yolo"
    assert captured["model_path"] == "/models/yolo_pose.pt"
    # Runtime is derived from RUNTIME_TIER (Gen-2 FT1); the "cpu" tier resolves
    # to native torch/CPU on every host, so the threaded string is host-stable.
    assert captured["compute_runtime"] == "cpu"
    # Regression guard: load_pose_backend (-> load_pose_model) already warms
    # the backend it returns; a second caller-side warmup() call is redundant
    # and, for the SLEAP service backend, breaks _service_started_here
    # ownership tracking, leaking the service subprocess past close().
    assert fake_backend.warmup_calls == 0
    assert "warmed_up" not in captured
    assert kpt_source_names == ["kpt0", "kpt1"]
    assert kpt_labels


def test_init_pose_backend_sleap_delegates_to_load_pose_backend(
    monkeypatch, tmp_path
) -> None:
    """Golden rule: the SLEAP pose branch routes through the shared
    ``load_pose_backend`` shim and threads SLEAP settings (env, max_instances)
    through -- the tier -> flavor decision lives in ``load_pose_model``, not
    here, so this asserts delegation + settings, not the resolved flavor."""
    pose_utils = importlib.import_module("hydra_suite.core.individual.pose.utils")

    monkeypatch.setattr(
        pose_utils, "load_skeleton_from_json", lambda _p: (["kpt0"], [(0, 1)])
    )

    captured: dict[str, object] = {}

    class FakeBackend:
        output_keypoint_names = ["kpt0"]

        def __init__(self) -> None:
            self.warmup_calls = 0

        def warmup(self) -> None:
            self.warmup_calls += 1
            captured["warmed_up"] = True

    fake_backend = FakeBackend()

    def _fake_load_pose_backend(**kwargs):
        captured.update(kwargs)
        return fake_backend

    monkeypatch.setattr(ic, "load_pose_backend", _fake_load_pose_backend)

    backend, kpt_source_names, kpt_labels = ic._init_pose_backend(
        {
            "ENABLE_POSE_EXTRACTOR": True,
            "POSE_MODEL_TYPE": "sleap",
            "POSE_MODEL_DIR": "/models/sleap_model",
            "POSE_MIN_KPT_CONF_VALID": 0.25,
            "POSE_BATCH_SIZE": 4,
            "POSE_SLEAP_ENV": "sleap_env_x",
            "POSE_SLEAP_MAX_INSTANCES": 2,
            "RUNTIME_TIER": "cpu",
        },
        str(tmp_path),
    )

    assert backend is fake_backend
    # Regression guard: same double-warmup leak as the YOLO case above, but
    # more severe for SLEAP -- the service backend's warmup() ownership
    # bookkeeping (_service_started_here) is not idempotent across calls, so
    # a second warmup() here leaves the SLEAP service process orphaned after
    # close().
    assert fake_backend.warmup_calls == 0
    assert "warmed_up" not in captured
    assert kpt_source_names == ["kpt0"]

    assert captured["backend_family"] == "sleap"
    assert captured["compute_runtime"] == "cpu"
    assert captured["sleap_env"] == "sleap_env_x"
    assert captured["sleap_max_instances"] == 2
    assert captured["model_path"] == "/models/sleap_model"
    assert captured["out_root"] == str(tmp_path)


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


def test_compute_frame_corners_and_affines_accumulates_clipping_stats() -> None:
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

    ic._compute_frame_corners_and_affines(
        [clipped_task, fitting_task], geometry, clipping_stats
    )

    assert clipping_stats.total_count == 2
    assert clipping_stats.clipped_count == 1
    assert clipping_stats.worst_overflow_ratio > 1.0
    summary = clipping_stats.summary()
    assert summary is not None
    assert "1/2" in summary


def test_compute_frame_corners_and_affines_counts_degenerate_drops() -> None:
    """A degenerate OBB is DROPPED here (affine None, skipped downstream by
    ``_extract_pose_crop``), so it must be counted on the shared ClippingStats
    rather than making the run quietly shorter (spec 2026-08-18).
    """
    geometry = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    clipping_stats = ClippingStats()

    # w=h=0 -> every OBB corner coincides -> canonical_affine raises.
    degenerate_task = {"cx": 5.0, "cy": 5.0, "w": 0.0, "h": 0.0, "theta": 0.0}
    fitting_task = {"cx": 0.0, "cy": 0.0, "w": 20.0, "h": 10.0, "theta": 0.0}

    _corners, affines = ic._compute_frame_corners_and_affines(
        [degenerate_task, fitting_task], geometry, clipping_stats
    )

    assert affines[0] is None and affines[1] is not None
    assert clipping_stats.degenerate_skipped_count == 1
    # The dropped task is not recorded as a normal (clipped/total) detection.
    assert clipping_stats.total_count == 1
    assert clipping_stats.clipped_count == 0

    summary = clipping_stats.summary()
    assert summary is not None and "DEGENERATE" in summary


def test_compute_frame_corners_and_affines_tolerates_no_stats() -> None:
    """The accumulator is optional; a degenerate task must not blow up when
    the caller passed no ClippingStats."""
    geometry = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    _corners, affines = ic._compute_frame_corners_and_affines(
        [{"cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0, "theta": 0.0}], geometry, None
    )
    assert affines == [None]


def test_pose_fit_failure_skips_detection(monkeypatch) -> None:
    """F6: an ``apply_fit`` failure on a canonical crop must drop the
    detection, not fall back to feeding the backend a raw canvas crop while
    still composing/inverting a fit that was never applied.
    """

    class _FakeProfiler:
        def tick(self, *_a, **_k) -> None:
            return None

        def tock(self, *_a, **_k) -> None:
            return None

    class _FakePoseBackend:
        def __init__(self) -> None:
            self.calls: list[list[np.ndarray]] = []

        def predict_batch(self, crops):
            self.calls.append(list(crops))
            return [None] * len(crops)

    monkeypatch.setattr(
        ic,
        "apply_fit",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("fit failed")),
    )

    geometry = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    canvas_w, canvas_h = geometry.canvas_wh
    raw_crop = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    pose_backend = _FakePoseBackend()
    interp_pose_rows: list[dict] = []
    pending_crops = [raw_crop]
    pending_entries = [
        {
            "task": {"frame_id": 7, "traj_id": 3},
            "filename": "frame_7_traj_3.png",
            "crop_info": {
                "canonical": True,
                "M_forward": np.eye(2, 3, dtype=np.float32),
            },
        }
    ]

    ic._flush_pose_batch(
        pose_backend,
        pending_crops,
        pending_entries,
        interp_pose_rows,
        [],
        [],
        _FakeProfiler(),
        geometry,
    )

    # The backend must never see the raw canvas crop for the failed fit.
    assert len(pose_backend.calls) == 1
    called_crops = pose_backend.calls[0]
    assert len(called_crops) == 0
    assert not any(np.array_equal(c, raw_crop) for c in called_crops)

    # The detection is dropped entirely -- no pose row produced for it.
    assert interp_pose_rows == []
