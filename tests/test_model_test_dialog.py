"""Tests for ModelTestDialog parameter construction and inference execution.

Covers the port off the legacy ``YOLOOBBDetector`` onto the production
``load_obb_executor`` factory (mirrors Plan A Task 5's
``test_bench_obb_uses_load_obb_executor_not_legacy_detector`` pattern).
"""

from __future__ import annotations

import numpy as np

from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import build_test_params


def test_build_test_params_direct_mode():
    params = build_test_params(
        model_path="/models/best.pt",
        role="obb_direct",
        compute_runtime="cpu",
        imgsz=640,
    )
    assert params == {
        "model_path": "/models/best.pt",
        "compute_runtime": "cpu",
        "imgsz": 640,
        "task": "obb",
    }


def test_build_test_params_seq_crop_obb():
    params = build_test_params(
        model_path="/models/crop.pt",
        role="seq_crop_obb",
        compute_runtime="cuda",
        imgsz=160,
        crop_pad_ratio=0.2,
        min_crop_size_px=32,
        enforce_square=False,
        detect_model_path="/models/detect.pt",
    )
    assert params["task"] == "obb"
    assert params["model_path"] == "/models/crop.pt"
    assert params["compute_runtime"] == "cuda"
    assert params["imgsz"] == 160
    assert params["detect_model_path"] == "/models/detect.pt"
    assert params["crop_pad_ratio"] == 0.2
    assert params["min_crop_size_px"] == 32
    assert params["enforce_square"] is False


def test_build_test_params_seq_crop_obb_defaults_detect_model_to_model_path():
    params = build_test_params(
        model_path="/models/crop.pt",
        role="seq_crop_obb",
        compute_runtime="cpu",
        imgsz=160,
    )
    assert params["detect_model_path"] == "/models/crop.pt"


def test_build_test_params_seq_detect():
    params = build_test_params(
        model_path="/models/detect.pt",
        role="seq_detect",
        compute_runtime="cuda",
        imgsz=640,
    )
    assert params == {
        "model_path": "/models/detect.pt",
        "compute_runtime": "cuda",
        "imgsz": 640,
        "task": "detect",
    }
    assert "detect_model_path" not in params


def test_build_test_params_no_legacy_keys():
    """The new dict must not carry any legacy YOLOOBBDetector-shaped keys."""
    params = build_test_params(
        model_path="/models/best.pt",
        role="obb_direct",
        compute_runtime="cpu",
        imgsz=640,
    )
    legacy_keys = {
        "YOLO_MODEL_PATH",
        "YOLO_DEVICE",
        "YOLO_IMGSZ",
        "YOLO_CONFIDENCE_THRESHOLD",
        "YOLO_IOU_THRESHOLD",
        "YOLO_MAX_TARGETS",
        "USE_TENSORRT",
        "USE_ONNX",
        "YOLO_OBB_MODE",
        "YOLO_OBB_DIRECT_MODEL_PATH",
        "YOLO_DETECT_MODEL_PATH",
        "YOLO_CROP_OBB_MODEL_PATH",
    }
    assert not legacy_keys & set(params)


class _FakeOBB:
    def __init__(self, xywhr, conf):
        self._xywhr = xywhr
        self.conf = _Tensor(conf)

    @property
    def xywhr(self):
        return _Tensor(self._xywhr)

    def __len__(self):
        return len(self._xywhr)


class _Tensor:
    """Minimal stand-in for a torch tensor exposing ``.cpu().numpy()``."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr.copy()


class _FakeResult:
    def __init__(self, obb=None, boxes=None):
        self.obb = obb
        self.boxes = boxes


class _FakeExecutor:
    """Fake executor recording predict() calls, matching load_obb_executor's shape."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def predict(self, frames, **kwargs):
        self.calls.append((frames, kwargs))
        return self._results


def test_test_worker_execute_uses_load_obb_executor_not_legacy_detector(monkeypatch):
    """``_TestWorker.execute`` must call ``load_obb_executor``, never ``YOLOOBBDetector``."""
    import hydra_suite.trackerkit.gui.dialogs.model_test_dialog as dialog_module

    load_calls = []
    fake_result = _FakeResult(
        obb=_FakeOBB(
            xywhr=[[10.0, 10.0, 4.0, 2.0, 0.0]],
            conf=[0.9],
        )
    )
    fake_executor = _FakeExecutor([fake_result])

    def fake_load_obb_executor(model_path, compute_runtime, **kwargs):
        load_calls.append((model_path, compute_runtime, kwargs))
        return fake_executor

    monkeypatch.setattr(
        "hydra_suite.core.inference.runtime_artifacts.load_obb_executor",
        fake_load_obb_executor,
    )

    def _fail_if_called(*_a, **_k):
        raise AssertionError("_TestWorker.execute must not construct YOLOOBBDetector")

    monkeypatch.setattr(
        "hydra_suite.core.detectors.YOLOOBBDetector", _fail_if_called, raising=False
    )

    # cv2.imread is monkeypatched so this test doesn't depend on real image files.
    fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(dialog_module.cv2, "imread", lambda path: fake_frame.copy())

    params = build_test_params(
        model_path="/models/best.pt",
        role="obb_direct",
        compute_runtime="cpu",
        imgsz=640,
    )
    worker = dialog_module._TestWorker(params, ["/fake/sample.jpg"])

    emitted = []
    worker.image_ready.connect(lambda frame: emitted.append(frame))
    finished = []
    worker.finished_all.connect(lambda: finished.append(True))

    worker.execute()

    assert load_calls, "load_obb_executor was never called"
    assert load_calls[0][0] == "/models/best.pt"
    assert load_calls[0][1] == "cpu"
    assert fake_executor.calls, "executor.predict was never called"
    assert emitted, "no annotated frame was emitted"
    assert finished == [True]


def test_test_worker_run_detect_only_draws_axis_aligned_boxes():
    """seq_detect role draws plain rectangles (no OBB rotation) from stage-1 boxes."""
    import hydra_suite.trackerkit.gui.dialogs.model_test_dialog as dialog_module

    boxes = _Tensor([[5.0, 5.0, 15.0, 25.0]])

    class _Boxes:
        xyxy = boxes

        def __len__(self):
            return 1

    fake_result = _FakeResult(boxes=_Boxes())
    fake_executor = _FakeExecutor([fake_result])

    corners = dialog_module._TestWorker._run_detect_only(
        np.zeros((32, 32, 3), dtype=np.uint8), fake_executor
    )

    assert len(corners) == 1
    quad = corners[0]
    assert quad.shape == (4, 2)
    assert np.allclose(quad, [[5.0, 5.0], [15.0, 5.0], [15.0, 25.0], [5.0, 25.0]])
