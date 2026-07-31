"""Tests for DetectKit's prediction-preview helpers.

These verify the preview path is wired through the inference pipeline's public
``load_obb_executor`` + ``extract_obb_result`` API (not raw ultralytics), and
that per-detection ``class_id`` now reflects the model's real class ids rather
than a hardcoded ``0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import hydra_suite.detectkit.gui.prediction_preview as pp

# --------------------------------------------------------------------------
# Fakes mimicking the shape of an ultralytics Results.obb that
# ``extract_obb_result`` consumes (xywhr / conf / cls torch tensors).
# --------------------------------------------------------------------------


class _FakeOBB:
    def __init__(self, xywhr, conf, cls):
        self.xywhr = torch.tensor(xywhr, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self):
        return self.xywhr.shape[0]


class _FakeResult:
    def __init__(self, obb):
        self.obb = obb
        self.boxes = None


class _FakeExecutor:
    """Stand-in for a loaded OBB executor: records predict() calls."""

    def __init__(self, obb: _FakeOBB):
        self._obb = obb
        self.calls: list[dict] = []

    def predict(self, frames, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(self._obb)]


def _two_detection_obb() -> _FakeOBB:
    # Two valid detections with distinct, non-zero class ids (2 and 5).
    return _FakeOBB(
        xywhr=[[100.0, 100.0, 40.0, 20.0, 0.1], [200.0, 150.0, 30.0, 30.0, 0.2]],
        conf=[0.9, 0.8],
        cls=[2, 5],
    )


def _write_dummy_image(path: Path) -> str:
    import cv2

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_no_raw_ultralytics_import_in_source():
    """The raw ``from ultralytics import YOLO`` path must be gone."""
    source = Path(pp.__file__).read_text()
    assert "from ultralytics import YOLO" not in source
    assert "import ultralytics" not in source


def test_preview_for_image_returns_real_class_ids(tmp_path, monkeypatch):
    pp._get_torch_model.cache_clear()
    executor = _FakeExecutor(_two_detection_obb())
    monkeypatch.setattr(pp, "load_obb_executor", lambda *a, **k: executor)

    img_path = _write_dummy_image(tmp_path / "frame.png")

    dets = pp.predict_preview_detections_for_image(
        executor, img_path, device="cpu", confidence_threshold=0.5
    )

    assert len(dets) == 2
    for d in dets:
        assert set(d.keys()) == {"class_id", "polygon_px", "confidence"}
        assert isinstance(d["class_id"], int)
        assert len(d["polygon_px"]) == 4
        for pt in d["polygon_px"]:
            assert len(pt) == 2
            assert all(isinstance(v, float) for v in pt)
        assert isinstance(d["confidence"], float)

    # class_id must reflect the OBBResult.class_ids (2, 5), NOT a hardcoded 0.
    class_ids = sorted(d["class_id"] for d in dets)
    assert class_ids == [2, 5]
    # Confidences round-trip from the fake result.
    confs = sorted(round(d["confidence"], 3) for d in dets)
    assert confs == [0.8, 0.9]


def test_predict_preview_detections_loads_via_executor(tmp_path, monkeypatch):
    pp._get_torch_model.cache_clear()
    executor = _FakeExecutor(_two_detection_obb())
    captured: dict = {}

    def _fake_loader(model_path, compute_runtime, **kwargs):
        captured["model_path"] = model_path
        captured["compute_runtime"] = compute_runtime
        captured["kwargs"] = kwargs
        return executor

    monkeypatch.setattr(pp, "load_obb_executor", _fake_loader)

    img_path = _write_dummy_image(tmp_path / "frame.png")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")

    dets = pp.predict_preview_detections(
        img_path, str(model_path), device_preference="cpu", confidence_threshold=0.4
    )

    assert len(dets) == 2
    assert sorted(d["class_id"] for d in dets) == [2, 5]
    # cpu preference must resolve to the "cpu" compute runtime for the executor.
    assert captured["compute_runtime"] == "cpu"
    # Preview must never trigger a slow TRT/CoreML export on a click.
    assert captured["kwargs"].get("auto_export") is False


def test_load_torch_model_returns_executor_and_runtime(tmp_path, monkeypatch):
    pp._get_torch_model.cache_clear()
    executor = _FakeExecutor(_two_detection_obb())
    monkeypatch.setattr(pp, "load_obb_executor", lambda *a, **k: executor)

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")

    handle, runtime = pp.load_torch_model(str(model_path), "cpu")
    assert handle is executor
    assert runtime == "cpu"


def test_sequential_preview_resolves_real_class_id(tmp_path, monkeypatch):
    """Sequential preview must surface the merged OBBResult's real class id."""
    pp._get_torch_model.cache_clear()

    # Stage-1 detect executor: return one axis-aligned box covering part of frame.
    class _Boxes:
        def __init__(self, xyxy):
            self._xyxy = torch.tensor(xyxy, dtype=torch.float32)

        def __len__(self):
            return self._xyxy.shape[0]

        @property
        def xyxy(self):
            return self._xyxy

    class _DetectResult:
        def __init__(self, boxes):
            self.boxes = boxes
            self.obb = None

    class _DetectExecutor:
        def predict(self, frames, **kwargs):
            return [_DetectResult(_Boxes([[10.0, 10.0, 50.0, 50.0]]))]

    # Stage-2 OBB executor: return one oriented box with class id 7.
    obb = _FakeOBB(xywhr=[[20.0, 20.0, 16.0, 10.0, 0.0]], conf=[0.77], cls=[7])
    obb_executor = _FakeExecutor(obb)
    detect_executor = _DetectExecutor()

    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    dets = pp.predict_obb_for_frame_sequential(
        detect_executor,
        obb_executor,
        frame,
        detect_device="cpu",
        obb_device="cpu",
        conf=0.25,
        iou=0.7,
    )
    assert len(dets) == 1  # (cx, cy, w, h, theta, conf) tuple
    assert len(dets[0]) == 6

    img_path = _write_dummy_image(tmp_path / "frame.png")
    # Reuse the same fake executors through the loader seam.
    loaders = iter([detect_executor, obb_executor])
    monkeypatch.setattr(pp, "load_obb_executor", lambda *a, **k: next(loaders))

    dict_dets = pp.predict_preview_detections_sequential(
        img_path,
        str(tmp_path / "detect.pt"),
        str(tmp_path / "obb.pt"),
        device_preference="cpu",
        confidence_threshold=0.25,
    )
    assert len(dict_dets) == 1
    assert dict_dets[0]["class_id"] == 7  # real class id, not hardcoded 0
    assert len(dict_dets[0]["polygon_px"]) == 4
