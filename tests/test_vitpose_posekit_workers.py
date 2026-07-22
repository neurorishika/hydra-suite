"""ViTPose worker plumbing: _build_pose_backend threads vitpose_batch, and a
vitpose backend re-raises (no legacy PoseInferenceService fallback)."""

from __future__ import annotations

import importlib


def _workers():
    return importlib.import_module("hydra_suite.posekit.gui.workers")


def test_build_pose_backend_threads_vitpose_batch(monkeypatch):
    workers = _workers()
    captured: dict[str, object] = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(workers, "load_pose_backend", _fake)
    workers._build_pose_backend(
        backend_family="vitpose",
        model_path="/models/vit.pt",
        exported_model_path="",
        compute_runtime="cpu",
        min_valid_conf=0.0,
        batch_size=4,
        conf=0.25,
        keypoint_names=["a", "b"],
        skeleton_edges=[],
        out_root="/tmp/out",
        sleap_env=None,
        vitpose_batch=6,
    )
    assert captured["backend_family"] == "vitpose"
    assert captured["model_path"] == "/models/vit.pt"
    assert captured["vitpose_batch"] == 6


def test_vitpose_worker_reraises_no_legacy_fallback(monkeypatch, tmp_path):
    workers = _workers()

    # Force the shared build path to fail.
    def _boom(**kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(workers, "_build_pose_backend", _boom)

    # Guard: the legacy fallback path must NOT be reached for vitpose.
    called = {"legacy": False}

    class _FakeInfer:
        def __init__(self, *a, **k):
            pass

        def get_cached_pred(self, *a, **k):
            return None

        def predict(self, *a, **k):
            called["legacy"] = True
            return None, "legacy reached"

    monkeypatch.setattr(workers, "PoseInferenceService", _FakeInfer)

    img = tmp_path / "f.png"
    import cv2
    import numpy as np

    cv2.imwrite(str(img), np.zeros((16, 16, 3), np.uint8))

    errors: list[str] = []
    w = workers.PosePredictWorker(
        model_path=tmp_path / "vit.pt",
        image_path=img,
        out_root=tmp_path,
        keypoint_names=["a", "b"],
        skeleton_edges=[],
        backend="vitpose",
        runtime_flavor="cpu",
        vitpose_batch=4,
    )
    w.failed.connect(lambda msg: errors.append(msg))
    w.run()

    assert called["legacy"] is False
    assert errors and "vitpose" in errors[0].lower()
