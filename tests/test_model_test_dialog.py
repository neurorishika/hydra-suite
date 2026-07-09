"""Tests for ModelTestDialog parameter construction and inference execution.

Covers the port off the legacy ``YOLOOBBDetector`` onto the production
``load_obb_executor`` factory (mirrors Plan A Task 5's
``test_bench_obb_uses_load_obb_executor_not_legacy_detector`` pattern).
"""

from __future__ import annotations

import numpy as np

from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import (
    build_test_params,
    training_device_to_compute_runtime,
)


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


def test_training_device_auto_resolves_to_cuda_when_available(monkeypatch):
    """``"auto"`` must autodetect (CUDA > MPS > CPU), not regress to CPU."""
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.TORCH_CUDA_AVAILABLE", True, raising=False
    )
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.MPS_AVAILABLE", False, raising=False
    )

    assert training_device_to_compute_runtime("auto") == "cuda"


def test_training_device_auto_resolves_to_mps_when_no_cuda(monkeypatch):
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.TORCH_CUDA_AVAILABLE", False, raising=False
    )
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.MPS_AVAILABLE", True, raising=False
    )

    assert training_device_to_compute_runtime("auto") == "mps"


def test_training_device_auto_resolves_to_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.TORCH_CUDA_AVAILABLE", False, raising=False
    )
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.MPS_AVAILABLE", False, raising=False
    )

    assert training_device_to_compute_runtime("auto") == "cpu"


def test_training_device_explicit_values_unaffected_by_availability(monkeypatch):
    """Explicit ``cpu``/``cuda``/``mps``/``cuda:N`` must not be autodetected."""
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.TORCH_CUDA_AVAILABLE", True, raising=False
    )
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.MPS_AVAILABLE", True, raising=False
    )

    assert training_device_to_compute_runtime("cpu") == "cpu"
    assert training_device_to_compute_runtime("cuda:0") == "cuda"
    assert training_device_to_compute_runtime("mps") == "mps"


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
    assert load_calls[0][2]["max_det"] == dialog_module._TEST_MAX_DET
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


def test_test_worker_execute_seq_crop_obb_threads_max_det_to_both_executors(
    monkeypatch,
):
    """Both stage-1 (detect) and stage-2 (crop-OBB) executor loads get max_det."""
    import hydra_suite.trackerkit.gui.dialogs.model_test_dialog as dialog_module

    load_calls = []
    fake_executor = _FakeExecutor([_FakeResult(obb=_FakeOBB(xywhr=[], conf=[]))])

    def fake_load_obb_executor(model_path, compute_runtime, **kwargs):
        load_calls.append((model_path, compute_runtime, kwargs))
        return fake_executor

    monkeypatch.setattr(
        "hydra_suite.core.inference.runtime_artifacts.load_obb_executor",
        fake_load_obb_executor,
    )
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(dialog_module.cv2, "imread", lambda path: fake_frame.copy())

    params = build_test_params(
        model_path="/models/crop.pt",
        role="seq_crop_obb",
        compute_runtime="cpu",
        imgsz=160,
        detect_model_path="/models/detect.pt",
    )
    worker = dialog_module._TestWorker(params, ["/fake/sample.jpg"])
    worker.execute()

    assert len(load_calls) == 2
    for _model_path, _runtime, kwargs in load_calls:
        assert kwargs["max_det"] == dialog_module._TEST_MAX_DET


def test_run_sequential_end_to_end_offset_and_scale_bookkeeping():
    """``_run_sequential`` (detect -> crop -> stage2-OBB -> merge) must produce
    final corners in *frame* coordinates, correctly undoing the stage-2 resize
    and the crop offset -- the path most likely to carry an offset/scale bug.
    """
    import hydra_suite.trackerkit.gui.dialogs.model_test_dialog as dialog_module

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Stage-1 detect: a single axis-aligned box centered at (50, 50), 20x20.
    detect_boxes = _Tensor([[40.0, 40.0, 60.0, 60.0]])

    class _Boxes:
        xyxy = detect_boxes

        def __len__(self):
            return 1

    detect_result = _FakeResult(boxes=_Boxes())
    detect_executor = _FakeExecutor([detect_result])

    # crop_spec: crop_pad_ratio=0.15, min_crop_size_px=64, enforce_square=True
    # half = max(20,20)/2 + 0.15*20 = 13 -> enforced to max(13, 32) = 32
    # crop region: cx=50, cy=50 -> [18:82, 18:82] (64x64), offset=(18, 18)
    crop_spec = dialog_module._SeqCropSpec(
        crop_pad_ratio=0.15, min_crop_size_px=64, enforce_square_crop=True
    )

    # Stage-2 obb result predicted on the 32x32-resized crop (stage2_imgsz=32):
    # center (16, 16), w=8, h=4, angle=0 -> in the *64x64 crop's own space*
    # (scale = 64/32 = 2) this becomes center (32, 32), w=16, h=8.
    stage2_result = _FakeResult(
        obb=_FakeOBB(xywhr=[[16.0, 16.0, 8.0, 4.0, 0.0]], conf=[0.9])
    )
    obb_executor = _FakeExecutor([stage2_result])

    corners = dialog_module._TestWorker._run_sequential(
        frame,
        0,
        detect_executor,
        obb_executor,
        crop_spec,
        32,  # stage2_imgsz
    )

    assert len(corners) == 1
    quad = np.asarray(corners[0])
    assert quad.shape == (4, 2)
    # Final centroid must land back at (50, 50) in frame coordinates:
    # crop-space center (32, 32) + offset (18, 18) = (50, 50).
    centroid = quad.mean(axis=0)
    assert np.allclose(centroid, [50.0, 50.0], atol=1e-3)
    # major=w=16, minor=h=8, angle=0 -> corners at cx +/- 8, cy +/- 4.
    expected = np.array(
        [[42.0, 46.0], [58.0, 46.0], [58.0, 54.0], [42.0, 54.0]], dtype=np.float32
    )
    assert np.allclose(quad, expected, atol=1e-3)

    # Both stages must have been invoked exactly once with the expected inputs.
    assert len(detect_executor.calls) == 1
    assert len(obb_executor.calls) == 1
    obb_call_frames = obb_executor.calls[0][0]
    assert len(obb_call_frames) == 1
    # Crops fed to stage-2 must be resized to the configured stage2 size.
    assert obb_call_frames[0].shape[:2] == (32, 32)
