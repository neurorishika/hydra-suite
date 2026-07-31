"""Pose runtime golden rule: PoseKit's ``_build_pose_backend`` delegates.

``posekit/gui/workers.py``'s ``_build_pose_backend`` no longer carries its own
runtime-flavor ladder (``is_cuda_like`` / ``onnx_cuda`` /
``create_pose_backend_from_config`` / ``HYDRA_SLEAP_FLAVOR``). It now routes
every backend build through ``core/inference/api.load_pose_backend`` (the shared
shim over ``stages/pose.load_pose_model``), so the tier -> flavor decision lives
in exactly one place (the pose runtime golden rule).

These tests assert the delegation (call args) rather than the resolved
runtime_flavor/device -- the latter is covered by
``tests/test_pose_golden_rule.py`` / ``tests/test_inference_stages_pose.py``
against ``load_pose_model`` directly.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def _workers_module():
    return importlib.import_module("hydra_suite.posekit.gui.workers")


def test_build_pose_backend_delegates_to_load_pose_backend(monkeypatch) -> None:
    workers = _workers_module()

    captured: dict[str, object] = {}
    sentinel = object()

    def _fake_load_pose_backend(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(workers, "load_pose_backend", _fake_load_pose_backend)

    backend = workers._build_pose_backend(
        backend_family="yolo",
        model_path="/models/yolo_pose.pt",
        exported_model_path="",
        compute_runtime="coreml",
        min_valid_conf=0.0,
        batch_size=4,
        conf=0.25,
        keypoint_names=["a", "b"],
        skeleton_edges=[],
        out_root="/tmp/out",
        sleap_env=None,
        sleap_max_instances=1,
    )

    assert backend is sentinel
    assert captured["backend_family"] == "yolo"
    assert captured["model_path"] == "/models/yolo_pose.pt"
    assert captured["compute_runtime"] == "coreml"
    assert captured["keypoint_names"] == ["a", "b"]
    assert captured["min_valid_confidence"] == 0.0
    assert captured["confidence_threshold"] == 0.25


def test_build_pose_backend_sleap_threads_sleap_settings(monkeypatch) -> None:
    """SLEAP settings (env, batch, max_instances, exported/out_root) must be
    threaded through to the shim, not silently dropped."""
    workers = _workers_module()

    captured: dict[str, object] = {}

    def _fake_load_pose_backend(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(workers, "load_pose_backend", _fake_load_pose_backend)

    workers._build_pose_backend(
        backend_family="sleap",
        model_path="/models/sleap_model",
        exported_model_path="/exports/model.onnx",
        compute_runtime="cuda",
        min_valid_conf=0.2,
        batch_size=2,
        conf=0.25,
        keypoint_names=["a", "b"],
        skeleton_edges=[(0, 1)],
        out_root="/tmp/out",
        sleap_env="sleap_env_x",
        sleap_batch=8,
        sleap_max_instances=3,
    )

    assert captured["backend_family"] == "sleap"
    assert captured["model_path"] == "/models/sleap_model"
    assert captured["compute_runtime"] == "cuda"
    assert captured["exported_model_path"] == "/exports/model.onnx"
    assert captured["out_root"] == "/tmp/out"
    assert captured["sleap_env"] == "sleap_env_x"
    assert captured["sleap_batch"] == 8
    assert captured["sleap_max_instances"] == 3
    assert captured["skeleton_edges"] == [(0, 1)]


def test_posekit_workers_has_no_divergent_flavor_ladder() -> None:
    """Source guard: the deleted runtime-flavor ladder must not reappear."""
    src = Path(
        importlib.import_module("hydra_suite.posekit.gui.workers").__file__
    ).read_text(encoding="utf-8")
    for banned in (
        "is_cuda_like",
        "onnx_cuda",
        "create_pose_backend_from_config",
        "YoloNativeBackend",
    ):
        assert banned not in src, f"divergent pose ladder token still present: {banned}"


class _FakePose:
    def __init__(self, keypoints) -> None:
        self.keypoints = keypoints


class _FakeWarmupCountingBackend:
    """Backend stub that tracks warmup()/close() calls without doing real work."""

    def __init__(self, n_kpts: int) -> None:
        self.warmup_calls = 0
        self.closed = False
        self._n_kpts = n_kpts

    def warmup(self) -> None:
        self.warmup_calls += 1

    def predict_batch(self, images):
        return [_FakePose([(0.0, 0.0, 1.0)] * self._n_kpts) for _ in images]

    def close(self) -> None:
        self.closed = True


def test_pose_predict_worker_does_not_double_warmup(monkeypatch, tmp_path) -> None:
    """Regression: PosePredictWorker.run() must not call backend.warmup() --
    load_pose_backend (-> stages/pose.load_pose_model) already warms the
    backend it returns. A redundant second warmup() breaks the SLEAP service
    backend's ``_service_started_here`` ownership tracking and leaks the
    service subprocess past ``close()``."""
    workers = _workers_module()

    fake_backend = _FakeWarmupCountingBackend(n_kpts=2)

    def _fake_build_pose_backend(**kwargs):
        return fake_backend

    monkeypatch.setattr(workers, "_build_pose_backend", _fake_build_pose_backend)
    monkeypatch.setattr(workers.cv2, "imread", lambda _p: np.zeros((4, 4, 3)))

    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"fake")

    results: dict[str, object] = {}
    worker = workers.PosePredictWorker(
        model_path=tmp_path / "model.pt",
        image_path=image_path,
        out_root=tmp_path,
        keypoint_names=["a", "b"],
        skeleton_edges=[],
        backend="yolo",
        runtime_flavor="cpu",
    )
    worker.finished.connect(lambda preds: results.update(finished=preds))
    worker.failed.connect(lambda msg: results.update(failed=msg))

    worker.run()

    assert results.get("failed") is None, results.get("failed")
    assert fake_backend.warmup_calls == 0
    assert fake_backend.closed is True


def test_bulk_pose_predict_worker_does_not_double_warmup(monkeypatch, tmp_path) -> None:
    """Same double-warmup regression guard as
    ``test_pose_predict_worker_does_not_double_warmup``, for the bulk path."""
    workers = _workers_module()

    fake_backend = _FakeWarmupCountingBackend(n_kpts=1)

    def _fake_build_pose_backend(**kwargs):
        return fake_backend

    monkeypatch.setattr(workers, "_build_pose_backend", _fake_build_pose_backend)
    monkeypatch.setattr(
        workers.cv2, "imread", lambda _p: __import__("numpy").zeros((4, 4, 3))
    )

    image_paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for p in image_paths:
        p.write_bytes(b"fake")

    results: dict[str, object] = {}
    worker = workers.BulkPosePredictWorker(
        model_path=tmp_path / "model.pt",
        image_paths=image_paths,
        out_root=tmp_path,
        keypoint_names=["a"],
        skeleton_edges=[],
        backend="yolo",
        runtime_flavor="cpu",
    )
    worker.finished.connect(lambda preds: results.update(finished=preds))
    worker.failed.connect(lambda msg: results.update(failed=msg))

    worker.run()

    assert results.get("failed") is None, results.get("failed")
    assert fake_backend.warmup_calls == 0
    assert fake_backend.closed is True


def test_pred_runtime_flavor_returns_coreml_not_onnx_mps(monkeypatch) -> None:
    """Critical-finding regression test: ``MainWindow._pred_runtime_flavor``
    must map the native Apple GPU-Fast backend to ``"coreml"`` (not the
    ONNX-CoreML-EP ``"onnx_mps"`` flavor), or the exact ONNX-CoreML-EP bug
    reappears one function away for cache-key construction and exported-model-
    path browsing.

    ``_pred_runtime_flavor`` resolves the selected tier via ``RuntimeResolver``
    (Runtime Gen-2, FT7b): on an MPS host the ``gpu_fast`` tier resolves to the
    ``coreml`` backend. Exercises the unbound method against a minimal fake
    ``self`` (mirroring ``tests/test_posekit_main_window.py``'s
    ``SimpleNamespace`` pattern) rather than constructing a full ``MainWindow``.
    """
    from types import SimpleNamespace

    from hydra_suite.posekit.gui import main_window as mw
    from hydra_suite.posekit.gui.main_window import MainWindow
    from hydra_suite.runtime.resolver import PlatformInfo

    # Force an Apple-Silicon platform so gpu_fast -> coreml regardless of host.
    monkeypatch.setattr(
        mw, "detect_platform", lambda: PlatformInfo(has_cuda=False, has_mps=True)
    )

    fake_self = SimpleNamespace(
        _selected_tier=lambda *a, **k: "gpu_fast",
        _pred_backend=lambda: "yolo",
    )
    assert MainWindow._pred_runtime_flavor(fake_self) == "coreml"

    fake_self._pred_backend = lambda: "sleap"
    assert MainWindow._pred_runtime_flavor(fake_self) == "coreml"
