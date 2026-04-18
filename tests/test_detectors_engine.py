from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.module_loader import load_src_module, make_cv2_stub


class _FakeContour(dict):
    def __len__(self):
        return int(self.get("n_points", 6))


def _load_engine_module():
    """Load the detectors package components with a cv2 stub.

    Now that engine.py is split into multiple modules, we load each
    submodule and assemble a combined namespace for backward compatibility
    with existing tests.
    """
    cv2_stub = make_cv2_stub()
    stubs = {"cv2": cv2_stub}

    # Load submodules in dependency order
    utils_mod = load_src_module(
        "hydra_suite/core/detectors/_utils.py",
        "hydra_suite.core.detectors._utils",
        stubs=stubs,
    )
    # Ensure the _utils module is importable by the geometry/artifact mixins
    sys.modules["hydra_suite.core.detectors._utils"] = utils_mod

    geom_mod = load_src_module(
        "hydra_suite/core/detectors/_obb_geometry.py",
        "hydra_suite.core.detectors._obb_geometry",
        stubs=stubs,
    )
    sys.modules["hydra_suite.core.detectors._obb_geometry"] = geom_mod

    art_mod = load_src_module(
        "hydra_suite/core/detectors/_runtime_artifacts.py",
        "hydra_suite.core.detectors._runtime_artifacts",
        stubs=stubs,
    )
    sys.modules["hydra_suite.core.detectors._runtime_artifacts"] = art_mod

    bg_mod = load_src_module(
        "hydra_suite/core/detectors/bg_detector.py",
        "hydra_suite.core.detectors.bg_detector",
        stubs=stubs,
    )
    sys.modules["hydra_suite.core.detectors.bg_detector"] = bg_mod

    yolo_mod = load_src_module(
        "hydra_suite/core/detectors/yolo_detector.py",
        "hydra_suite.core.detectors.yolo_detector",
        stubs=stubs,
    )
    sys.modules["hydra_suite.core.detectors.yolo_detector"] = yolo_mod

    filter_mod = load_src_module(
        "hydra_suite/core/detectors/detection_filter.py",
        "hydra_suite.core.detectors.detection_filter",
        stubs=stubs,
    )
    sys.modules["hydra_suite.core.detectors.detection_filter"] = filter_mod

    factory_mod = load_src_module(
        "hydra_suite/core/detectors/factory.py",
        "hydra_suite.core.detectors.factory",
        stubs=stubs,
    )
    sys.modules["hydra_suite.core.detectors.factory"] = factory_mod

    # Assemble combined namespace matching what the old engine.py exported
    mod = types.ModuleType("detectors_engine_under_test")
    mod.ObjectDetector = bg_mod.ObjectDetector
    mod.YOLOOBBDetector = yolo_mod.YOLOOBBDetector
    mod.create_detector = factory_mod.create_detector
    mod.DetectionFilter = filter_mod.DetectionFilter
    mod._normalize_detection_ids = utils_mod._normalize_detection_ids
    mod.Path = Path
    return mod


def test_object_detector_detect_objects_filters_and_limits_count() -> None:
    mod = _load_engine_module()

    params = {
        "MAX_TARGETS": 2,
        "MAX_CONTOUR_MULTIPLIER": 20,
        "MIN_CONTOUR_AREA": 10.0,
        "ENABLE_SIZE_FILTERING": True,
        "MIN_OBJECT_SIZE": 15.0,
        "MAX_OBJECT_SIZE": 200.0,
        "MERGE_AREA_THRESHOLD": 1000.0,
        "CONSERVATIVE_KERNEL_SIZE": 3,
        "CONSERVATIVE_ERODE_ITER": 1,
    }
    detector = mod.ObjectDetector(params)

    contours = [
        _FakeContour(
            area=20.0,
            ellipse=((10.0, 10.0), (8.0, 4.0), 15.0),
            rect=(0, 0, 5, 5),
        ),
        _FakeContour(
            area=80.0,
            ellipse=((30.0, 30.0), (20.0, 8.0), 40.0),
            rect=(20, 20, 10, 10),
        ),
        _FakeContour(
            area=140.0,
            ellipse=((50.0, 50.0), (24.0, 10.0), 55.0),
            rect=(40, 40, 12, 12),
        ),
        _FakeContour(  # filtered by min size
            area=5.0,
            ellipse=((70.0, 70.0), (4.0, 2.0), 0.0),
            rect=(65, 65, 4, 4),
        ),
    ]

    meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
        contours, frame_count=1
    )
    assert yolo_results is None
    assert len(meas) == 2  # limited by MAX_TARGETS
    assert all(m.shape == (3,) for m in meas)
    assert len(shapes) == 2
    assert len(confidences) == 2
    assert all(np.isnan(c) for c in confidences)
    assert len(sizes) >= 2  # current implementation keeps original filtered size list


def test_create_detector_defaults_to_background_subtraction() -> None:
    mod = _load_engine_module()
    detector = mod.create_detector({"DETECTION_METHOD": "background_subtraction"})
    assert isinstance(detector, mod.ObjectDetector)


def test_tensorrt_engine_path_is_model_adjacent_and_stable_across_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.engine"
            out.write_bytes(b"fake-engine")
            return str(out)

    fake_ultra = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    model_dir_a = tmp_path / "a"
    model_dir_b = tmp_path / "b"
    model_dir_a.mkdir(parents=True, exist_ok=True)
    model_dir_b.mkdir(parents=True, exist_ok=True)
    model_a = model_dir_a / "best.pt"
    model_b = model_dir_b / "best.pt"
    model_a.write_bytes(b"model-a")
    model_b.write_bytes(b"model-b")

    def build_engine_path(model_path: Path, model_id: str):
        det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
        det.params = {
            "TENSORRT_MAX_BATCH_SIZE": 8,
            "INFERENCE_MODEL_ID": model_id,
        }
        det.device = "cuda:0"
        det.use_tensorrt = False
        det.tensorrt_model_path = None
        det._try_load_tensorrt_model(str(model_path))
        assert det.use_tensorrt
        assert det.tensorrt_model_path is not None
        return det.tensorrt_model_path

    path_a_id1 = build_engine_path(model_a, "id-A")
    path_b_id1 = build_engine_path(model_b, "id-A")
    path_a_id2 = build_engine_path(model_a, "id-B")

    # TensorRT artifacts are co-located with source model paths.
    # Different model locations should map to different engine paths.
    assert path_a_id1 != path_b_id1
    # Same model location should keep the same engine path even if inference id changes.
    assert path_a_id1 == path_a_id2


def test_tensorrt_engine_path_is_batch_specific(tmp_path: Path, monkeypatch) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.engine"
            out.write_bytes(b"fake-engine")
            return str(out)

    fake_ultra = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    def build_engine_path(batch_size: int):
        det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
        det.params = {
            "TENSORRT_MAX_BATCH_SIZE": int(batch_size),
            "INFERENCE_MODEL_ID": "id-A",
        }
        det.device = "cuda:0"
        det.use_tensorrt = False
        det.tensorrt_model_path = None
        det.tensorrt_batch_size = 1
        det._try_load_tensorrt_model(str(model_path))
        assert det.use_tensorrt
        assert det.tensorrt_model_path is not None
        return det.tensorrt_model_path, int(det.tensorrt_batch_size)

    path_b8, b8 = build_engine_path(8)
    path_b4, b4 = build_engine_path(4)
    assert path_b8 != path_b4
    assert path_b8.endswith("_b8.engine")
    assert path_b4.endswith("_b4.engine")
    assert b8 == 8
    assert b4 == 4


def test_onnx_artifact_path_is_batch_specific_and_model_adjacent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.overrides = {}
            self.model = types.SimpleNamespace(args={})

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    fake_ultra = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    def build_onnx_path(batch_size: int):
        det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
        det.params = {
            "TENSORRT_MAX_BATCH_SIZE": int(batch_size),
            "INFERENCE_MODEL_ID": "id-A",
        }
        det.device = "cpu"
        det.use_onnx = False
        det.onnx_model_path = None
        det.onnx_imgsz = None
        det.onnx_batch_size = 1
        det._try_load_onnx_model(str(model_path))
        assert det.use_onnx
        assert det.onnx_model_path is not None
        return det.onnx_model_path, int(det.onnx_batch_size)

    path_b8, b8 = build_onnx_path(8)
    path_b4, b4 = build_onnx_path(4)

    assert path_b8 != path_b4
    assert path_b8.endswith("_b8.onnx")
    assert path_b4.endswith("_b4.onnx")
    assert b8 == 8
    assert b4 == 4


def test_onnx_export_disables_end2end_postprocess_for_obb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        instances = []
        export_end2end = []

        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.overrides = {}
            self.model = types.SimpleNamespace(
                args={},
                model=[types.SimpleNamespace(end2end=True)],
            )
            FakeYOLO.instances.append(self)

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            FakeYOLO.export_end2end.append(self.model.model[-1].end2end)
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "TENSORRT_MAX_BATCH_SIZE": 1,
        "INFERENCE_MODEL_ID": "id-A",
        "YOLO_EXPORT_RAW_HEAD": True,
    }
    det.device = "cpu"
    det.use_onnx = False
    det.onnx_model_path = None
    det.onnx_imgsz = None
    det.onnx_batch_size = 1

    det._try_load_onnx_model(str(model_path))

    assert FakeYOLO.export_end2end == [False]
    assert FakeYOLO.instances[0].model.model[-1].end2end is True
    assert det.use_onnx


def test_realtime_onnx_artifact_forces_batch1(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.overrides = {}
            self.model = types.SimpleNamespace(args={})

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "TENSORRT_MAX_BATCH_SIZE": 8,
        "INFERENCE_MODEL_ID": "id-A",
        "TRACKING_REALTIME_MODE": True,
        "YOLO_EXPORT_RAW_HEAD": True,
    }
    det.device = "mps"
    det.use_onnx = False
    det.onnx_model_path = None
    det.onnx_imgsz = None
    det.onnx_batch_size = 1

    det._try_load_onnx_model(str(model_path))

    assert det.onnx_batch_size == 1
    assert det.onnx_model_path is not None
    assert det.onnx_model_path.endswith("_b1.onnx")


def test_non_realtime_mps_onnx_artifact_keeps_configured_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.overrides = {}
            self.model = types.SimpleNamespace(args={})

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "TENSORRT_MAX_BATCH_SIZE": 8,
        "INFERENCE_MODEL_ID": "id-A",
        "TRACKING_REALTIME_MODE": False,
        "YOLO_EXPORT_RAW_HEAD": True,
    }
    det.device = "mps"
    det.use_onnx = False
    det.onnx_model_path = None
    det.onnx_imgsz = None
    det.onnx_batch_size = 1

    det._try_load_onnx_model(str(model_path))

    assert det.onnx_batch_size == 8
    assert det.onnx_model_path is not None
    assert det.onnx_model_path.endswith("_b8.onnx")


def test_detect_aux_onnx_export_uses_rawhead_profile_suffix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        instances = []
        export_end2end = []

        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.model = types.SimpleNamespace(
                args={},
                model=[types.SimpleNamespace(end2end=True)],
            )
            FakeYOLO.instances.append(self)

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            FakeYOLO.export_end2end.append(self.model.model[-1].end2end)
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    model_path = tmp_path / "detect.pt"
    model_path.write_bytes(b"model")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "ENABLE_ONNX_RUNTIME": True,
        "YOLO_EXPORT_RAW_HEAD": True,
    }
    det.device = "cpu"

    onnx_path = det._prepare_runtime_artifact_for_task(str(model_path), task="detect")

    assert onnx_path.endswith("_detect_rawheadv1_b1.onnx")
    assert FakeYOLO.export_end2end == [False]
    assert FakeYOLO.instances[0].model.model[-1].end2end is True


def test_detect_aux_onnx_export_uses_configured_batch_outside_realtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        instances = []

        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.model = types.SimpleNamespace(
                args={},
                model=[types.SimpleNamespace(end2end=True)],
            )
            FakeYOLO.instances.append(self)

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    model_path = tmp_path / "detect.pt"
    model_path.write_bytes(b"model")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "ENABLE_ONNX_RUNTIME": True,
        "TENSORRT_MAX_BATCH_SIZE": 8,
        "TRACKING_REALTIME_MODE": False,
        "YOLO_EXPORT_RAW_HEAD": True,
    }
    det.device = "cpu"

    onnx_path = det._prepare_runtime_artifact_for_task(str(model_path), task="detect")

    assert onnx_path.endswith("_detect_rawheadv1_b8.onnx")


def test_sequential_crop_onnx_artifact_uses_stage2_build_batch_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.overrides = {}
            self.model = types.SimpleNamespace(args={})

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    model_path = tmp_path / "crop.pt"
    model_path.write_bytes(b"model")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "TENSORRT_MAX_BATCH_SIZE": 1,
        "INFERENCE_MODEL_ID": "id-A",
        "YOLO_OBB_MODE": "sequential",
        "YOLO_SEQ_STAGE2_RUNTIME_BUILD_BATCH_SIZE": 25,
        "YOLO_EXPORT_RAW_HEAD": True,
    }
    det.device = "mps"
    det.use_onnx = False
    det.onnx_model_path = None
    det.onnx_imgsz = None
    det.onnx_batch_size = 1

    det._try_load_onnx_model(str(model_path))

    assert det.onnx_batch_size == 25
    assert det.onnx_model_path is not None
    assert det.onnx_model_path.endswith("_b25.onnx")


def test_detect_aux_onnx_artifact_can_override_detect_build_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        instances = []

        def __init__(self, path, task=None):
            self.path = path
            self.task = task
            self.model = types.SimpleNamespace(
                args={},
                model=[types.SimpleNamespace(end2end=True)],
            )
            FakeYOLO.instances.append(self)

        def to(self, _device):
            return self

        def export(self, **_kwargs):
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    model_path = tmp_path / "detect.pt"
    model_path.write_bytes(b"model")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "ENABLE_ONNX_RUNTIME": True,
        "TENSORRT_MAX_BATCH_SIZE": 25,
        "YOLO_DETECT_RUNTIME_BUILD_BATCH_SIZE": 1,
        "YOLO_EXPORT_RAW_HEAD": True,
    }
    det.device = "cpu"

    onnx_path = det._prepare_runtime_artifact_for_task(str(model_path), task="detect")

    assert onnx_path.endswith("_detect_rawheadv1_b1.onnx")


def test_coreml_failed_onnx_artifact_reuses_cached_model_on_cpu_in_same_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()
    mod.YOLOOBBDetector._SESSION_ONNX_CPU_FALLBACK_ARTIFACTS.clear()

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    class FakeYOLO:
        export_calls = []

        def __init__(self, path, task=None):
            self.path = str(path)
            self.task = task
            self.overrides = {}
            self.model = types.SimpleNamespace(args={})

        def to(self, _device):
            return self

        def export(self, **kwargs):
            FakeYOLO.export_calls.append(kwargs)
            out = export_root / f"{uuid.uuid4().hex}.onnx"
            out.write_bytes(b"fake-onnx")
            return str(out)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    det1 = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det1.params = {
        "TENSORRT_MAX_BATCH_SIZE": 1,
        "INFERENCE_MODEL_ID": "id-A",
    }
    det1.device = "mps"
    det1.use_onnx = False
    det1.onnx_model_path = None
    det1.onnx_imgsz = None
    det1.onnx_batch_size = 1
    det1._onnx_predict_device = None
    det1._try_load_onnx_model(str(model_path))

    onnx_path = Path(det1.onnx_model_path)
    meta_path = onnx_path.with_suffix(f"{onnx_path.suffix}.runtime_meta.json")
    meta_path.unlink()

    class FailingPredictor:
        def __init__(self, artifact_path: Path):
            self.predictor = object()
            self._hydra_runtime_artifact_path = str(artifact_path)
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs.get("device"))
            if kwargs.get("device") == "mps":
                raise RuntimeError(
                    "CoreMLExecutionProvider failure: Unable to compute the prediction using a neural network model"
                )
            return ["ok"]

    failing = FailingPredictor(onnx_path)
    result = det1._predict_with_coreml_fallback(
        failing,
        {"device": "mps"},
        context="OBB inference",
    )

    assert result == ["ok"]
    assert failing.calls == ["mps", "cpu"]
    assert det1.obb_predict_device == "cpu"

    det2 = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det2.params = {
        "TENSORRT_MAX_BATCH_SIZE": 1,
        "INFERENCE_MODEL_ID": "id-A",
    }
    det2.device = "mps"
    det2.use_onnx = False
    det2.onnx_model_path = None
    det2.onnx_imgsz = None
    det2.onnx_batch_size = 1
    det2._onnx_predict_device = None
    det2._try_load_onnx_model(str(model_path))

    assert len(FakeYOLO.export_calls) == 1
    assert det2.use_onnx is True
    assert det2.onnx_model_path == str(onnx_path)
    assert det2._onnx_predict_device == "cpu"


def test_load_model_for_task_uses_cpu_for_blacklisted_onnx_artifact_on_mps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_engine_module()
    mod.YOLOOBBDetector._SESSION_ONNX_CPU_FALLBACK_ARTIFACTS.clear()

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = str(path)
            self.task = task

        def to(self, _device):
            return self

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )

    onnx_path = tmp_path / "detect.onnx"
    onnx_path.write_bytes(b"onnx")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {}
    det.device = "mps"
    det._mark_onnx_artifact_for_cpu_fallback(onnx_path)

    _model, predict_device = det._load_model_for_task(str(onnx_path), task="detect")

    assert predict_device == "cpu"


def test_yolo_raw_detection_cap_is_two_x_max_targets() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {"MAX_TARGETS": 6}
    assert det._raw_detection_cap() == 12


def test_resolve_onnx_imgsz_prefers_model_metadata(tmp_path: Path, monkeypatch) -> None:
    mod = _load_engine_module()

    class FakeYOLO:
        def __init__(self, _path, task=None):
            self.task = task
            self.overrides = {"imgsz": 1504}
            self.model = types.SimpleNamespace(args={"imgsz": 1504})

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"x")

    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {}
    imgsz = det._resolve_onnx_imgsz(model_path=model_path)
    assert imgsz == 1504


def test_filter_raw_detections_applies_conf_size_and_target_limit() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "YOLO_CONFIDENCE_THRESHOLD": 0.5,
        "YOLO_IOU_THRESHOLD": 0.7,
        "MAX_TARGETS": 2,
        "ENABLE_SIZE_FILTERING": True,
        "MIN_OBJECT_SIZE": 40.0,
        "MAX_OBJECT_SIZE": 200.0,
    }

    meas = [
        np.array([10.0, 10.0, 0.0], dtype=np.float32),
        np.array([40.0, 40.0, 0.0], dtype=np.float32),
        np.array([70.0, 70.0, 0.0], dtype=np.float32),
        np.array([100.0, 100.0, 0.0], dtype=np.float32),
    ]
    sizes = [120.0, 80.0, 60.0, 20.0]
    shapes = [(120.0, 1.2), (80.0, 1.1), (60.0, 1.0), (20.0, 1.0)]
    confidences = [0.95, 0.8, 0.2, 0.9]
    obb = [
        np.array([[8, 8], [12, 8], [12, 12], [8, 12]], dtype=np.float32),
        np.array([[38, 38], [42, 38], [42, 42], [38, 42]], dtype=np.float32),
        np.array([[68, 68], [72, 68], [72, 72], [68, 72]], dtype=np.float32),
        np.array([[98, 98], [102, 98], [102, 102], [98, 102]], dtype=np.float32),
    ]
    ids = [101.0, 102.0, 103.0, 104.0]

    out = det.filter_raw_detections(
        meas, sizes, shapes, confidences, obb, roi_mask=None, detection_ids=ids
    )
    out_meas, out_sizes, _, out_conf, _, out_ids = out

    assert len(out_meas) == 2
    assert out_sizes == [120.0, 80.0]
    assert np.allclose(out_conf, [0.95, 0.8], rtol=1e-6, atol=1e-6)
    assert out_ids == [101, 102]


def test_filter_raw_detections_applies_roi_mask() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "YOLO_CONFIDENCE_THRESHOLD": 0.0,
        "YOLO_IOU_THRESHOLD": 0.7,
        "MAX_TARGETS": 4,
        "ENABLE_SIZE_FILTERING": False,
    }

    roi = np.zeros((20, 20), dtype=np.uint8)
    roi[:, :10] = 255

    meas = [
        np.array([5.0, 10.0, 0.0], dtype=np.float32),
        np.array([15.0, 10.0, 0.0], dtype=np.float32),
    ]
    sizes = [50.0, 60.0]
    shapes = [(50.0, 1.0), (60.0, 1.0)]
    confidences = [0.6, 0.7]
    obb = [
        np.array([[4, 9], [6, 9], [6, 11], [4, 11]], dtype=np.float32),
        np.array([[14, 9], [16, 9], [16, 11], [14, 11]], dtype=np.float32),
    ]
    ids = [1.0, 2.0]

    out = det.filter_raw_detections(
        meas, sizes, shapes, confidences, obb, roi_mask=roi, detection_ids=ids
    )
    _, out_sizes, _, out_conf, _, out_ids = out
    assert out_sizes == [50.0]
    assert np.allclose(out_conf, [0.6], rtol=1e-6, atol=1e-6)
    assert out_ids == [1]


def test_filter_raw_detections_filters_heading_hints_consistently() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {
        "YOLO_CONFIDENCE_THRESHOLD": 0.5,
        "YOLO_IOU_THRESHOLD": 0.7,
        "MAX_TARGETS": 4,
        "ENABLE_SIZE_FILTERING": False,
    }

    meas = [
        np.array([5.0, 10.0, 0.0], dtype=np.float32),
        np.array([15.0, 10.0, 0.0], dtype=np.float32),
    ]
    sizes = [50.0, 60.0]
    shapes = [(50.0, 1.0), (60.0, 1.0)]
    confidences = [0.6, 0.4]  # second one filtered by confidence
    obb = [
        np.array([[4, 9], [6, 9], [6, 11], [4, 11]], dtype=np.float32),
        np.array([[14, 9], [16, 9], [16, 11], [14, 11]], dtype=np.float32),
    ]
    ids = [1.0, 2.0]
    heading_hints = [0.25, 1.25]
    heading_confidences = [0.9, 0.2]
    directed_mask = [1, 0]

    out = det.filter_raw_detections(
        meas,
        sizes,
        shapes,
        confidences,
        obb,
        roi_mask=None,
        detection_ids=ids,
        heading_hints=heading_hints,
        heading_confidences=heading_confidences,
        directed_mask=directed_mask,
    )
    (
        _,
        out_sizes,
        _,
        out_conf,
        _,
        out_ids,
        out_heading,
        out_heading_conf,
        out_directed,
    ) = out
    assert out_sizes == [50.0]
    assert np.allclose(out_conf, [0.6], rtol=1e-6, atol=1e-6)
    assert out_ids == [1]
    assert np.allclose(out_heading, [0.25], rtol=1e-6, atol=1e-6)
    assert np.allclose(out_heading_conf, [0.9], rtol=1e-6, atol=1e-6)
    assert out_directed == [1]


def test_sequential_stage2_obb_runs_in_batched_crop_call() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {}
    det.device = "cpu"

    class _ArrayWrap:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class _Boxes:
        def __init__(self):
            self.xyxy = _ArrayWrap([[10, 20, 40, 60], [60, 70, 90, 100]])
            self.conf = _ArrayWrap([0.5, 0.9])

        def __len__(self):
            return 2

    boxes = _Boxes()
    det.detect_model = types.SimpleNamespace(
        predict=lambda **_kwargs: [types.SimpleNamespace(boxes=boxes)]
    )
    det._build_sequential_crop = lambda _frame, _bbox: (
        np.zeros((8, 8, 3), dtype=np.uint8),
        (1.0, 2.0),
    )

    calls = {"count": 0, "source_len": 0}

    def _fake_predict(source, target_classes, raw_conf_floor, max_det):
        calls["count"] += 1
        calls["source_len"] = len(source) if isinstance(source, list) else -1
        return [
            types.SimpleNamespace(obb=f"obb_{i}") for i in range(calls["source_len"])
        ]

    def _fake_extract(_obb):
        meas = [np.array([5.0, 6.0, 0.1], dtype=np.float32)]
        sizes = [20.0]
        shapes = [(10.0, 1.0)]
        conf = [0.9]
        corners = [np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=np.float32)]
        return meas, sizes, shapes, conf, corners

    det._predict_obb_results = _fake_predict
    det._extract_raw_detections = _fake_extract

    out = det._run_sequential_raw_detection(
        np.zeros((128, 128, 3), dtype=np.uint8),
        target_classes=None,
        raw_conf_floor=0.01,
        max_det=4,
    )
    raw_meas, _, _, _, _, _ = out

    assert calls["count"] == 1
    assert calls["source_len"] == 2
    assert len(raw_meas) == 2


def test_sequential_stage2_obb_chunks_by_individual_batch_size() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {"YOLO_SEQ_INDIVIDUAL_BATCH_SIZE": 3}

    calls = []

    def _fake_stage2(chunk, _target_classes, _raw_conf_floor, _max_det, _predict_imgsz):
        calls.append(len(chunk))
        return [
            types.SimpleNamespace(obb=f"obb_{index}") for index in range(len(chunk))
        ]

    det._seq_run_stage2_obb = _fake_stage2

    crops = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(5)]
    results = det._seq_run_stage2_obb_batched(
        crops,
        target_classes=None,
        raw_conf_floor=0.01,
        max_det=8,
        predict_imgsz=16,
    )

    assert calls == [3, 3]
    assert len(results) == 5


def test_headtail_hint_uses_batched_classify_call() -> None:
    """Verify _compute_headtail_hints delegates to analyzer.analyze_crops."""
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {"YOLO_HEADTAIL_CONF_THRESHOLD": 0.6, "INDIVIDUAL_CROP_PADDING": 0.1}

    calls = {"count": 0}

    def _mock_analyze(frames, per_frame_obb_corners, profiler=None):
        calls["count"] += 1
        # Return (heading=0.5, conf=0.95, directed=1) for each detection
        return [[(0.5, 0.95, 1) for _ in corners] for corners in per_frame_obb_corners]

    analyzer = HeadTailAnalyzer.__new__(HeadTailAnalyzer)
    analyzer._backend = "yolo"
    analyzer._model = object()
    analyzer.analyze_crops = _mock_analyze
    det._headtail_analyzer = analyzer

    obb_corners = [
        np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=np.float32),
        np.array([[3, 3], [5, 3], [5, 4], [3, 4]], dtype=np.float32),
    ]
    heading_hints, heading_confidences, directed_mask, _affines = (
        det._compute_headtail_hints(np.zeros((64, 64, 3), dtype=np.uint8), obb_corners)
    )

    assert calls["count"] == 1
    assert np.allclose(heading_confidences, [0.95, 0.95], rtol=1e-6, atol=1e-6)
    assert directed_mask == [1, 1]
    assert np.allclose(heading_hints, [0.5, 0.5], rtol=1e-6, atol=1e-6)


def test_validate_headtail_class_names_accepts_five_class_schema() -> None:
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    normalized = HeadTailAnalyzer._validate_class_names(
        ["head_up", "head_down", "head_left", "head_right", "head_unknown"],
        strict=True,
        source="test model",
    )

    assert normalized == ["up", "down", "left", "right", "unknown"]


def test_validate_headtail_class_names_rejects_partial_schema() -> None:
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    with pytest.raises(ValueError, match="Expected exactly"):
        HeadTailAnalyzer._validate_class_names(
            ["left", "right", "unknown"], strict=True, source="test model"
        )


def test_load_headtail_yolo_model_requires_supported_schema() -> None:
    """Loading a YOLO head-tail model populates _headtail_analyzer."""
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.device = "cpu"
    det.params = {}

    # Mock _load_model_for_task to return a fake YOLO model
    det._load_model_for_task = lambda _path, task: (
        types.SimpleNamespace(
            names={0: "up", 1: "down", 2: "left", 3: "right", 4: "unknown"}
        ),
        "cpu",
    )

    # Monkey-patch HeadTailAnalyzer constructor to simulate tiny-load failure
    original_init = HeadTailAnalyzer.__init__

    def _skip_init(self, *args, **kwargs):
        self._backend = "none"
        self._model = None
        self._class_names = None
        self._input_size = None
        self._device = "cpu"
        self._conf_threshold = 0.5
        self._ref_ar = 2.0
        self._canonical_margin = 1.3
        self._padding_fraction = 0.3
        self._predict_device = None
        self._backend_obj = None
        self._canonical_labels = ()

    HeadTailAnalyzer.__init__ = _skip_init
    try:
        det._load_headtail_model("headtail.pt")
    finally:
        HeadTailAnalyzer.__init__ = original_init

    assert det._headtail_analyzer is not None
    assert det._headtail_analyzer.backend == "yolo"
    assert det._headtail_analyzer.class_names == [
        "up",
        "down",
        "left",
        "right",
        "unknown",
    ]


def test_load_headtail_model_rejects_invalid_named_schema() -> None:
    """Strict validation rejects unsupported 3-class schema."""
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.device = "cpu"
    det.params = {}

    # Monkey-patch HeadTailAnalyzer constructor to load with bad class names
    original_init = HeadTailAnalyzer.__init__

    def _load_with_bad_names(self, *args, **kwargs):
        self._backend = "classkit_tiny"
        self._model = object()
        self._class_names = ["left", "right", "unknown"]
        self._input_size = (128, 64)
        self._device = "cpu"
        self._conf_threshold = 0.5
        self._ref_ar = 2.0
        self._canonical_margin = 1.3
        self._padding_fraction = 0.3
        self._predict_device = None
        self._backend_obj = None
        self._canonical_labels = ("left", "right", "unknown")

    HeadTailAnalyzer.__init__ = _load_with_bad_names
    try:
        with pytest.raises(ValueError, match="Expected exactly"):
            det._load_headtail_model("bad_headtail.pth")
    finally:
        HeadTailAnalyzer.__init__ = original_init


def test_classkit_headtail_hints_abstain_on_up_down_unknown() -> None:
    """classkit_tiny backend abstains on up/down/unknown directions."""
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)
    det.params = {"YOLO_HEADTAIL_CONF_THRESHOLD": 0.6, "INDIVIDUAL_CROP_PADDING": 0.1}

    # Build a mock analyzer that returns canned classkit_tiny-style results
    # simulating up/unknown/left/right predictions
    def _mock_analyze(frames, per_frame_obb_corners, profiler=None):
        results = []
        for corners in per_frame_obb_corners:
            frame_results = []
            # Canned: up(abstain), unknown(abstain), left(directed), right(directed)
            canned = [
                (float("nan"), 0.95, 0),  # up -> abstain
                (float("nan"), 0.99, 0),  # unknown -> abstain
                ((0.5 + np.pi) % (2 * np.pi), 0.92, 1),  # left -> directed
                (0.5, 0.91, 1),  # right -> directed
            ]
            for di in range(len(corners)):
                if di < len(canned):
                    frame_results.append(canned[di])
                else:
                    frame_results.append((float("nan"), 0.0, 0))
            results.append(frame_results)
        return results

    analyzer = HeadTailAnalyzer.__new__(HeadTailAnalyzer)
    analyzer._backend = "classkit_tiny"
    analyzer._model = object()
    analyzer.analyze_crops = _mock_analyze
    det._headtail_analyzer = analyzer

    obb_corners = [
        np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=np.float32),
        np.array([[3, 3], [5, 3], [5, 4], [3, 4]], dtype=np.float32),
        np.array([[6, 6], [8, 6], [8, 7], [6, 7]], dtype=np.float32),
        np.array([[9, 9], [11, 9], [11, 10], [9, 10]], dtype=np.float32),
    ]

    heading_hints, heading_confidences, directed_mask, _affines = (
        det._compute_headtail_hints(np.zeros((64, 64, 3), dtype=np.uint8), obb_corners)
    )

    assert np.isnan(heading_hints[0])
    assert np.isnan(heading_hints[1])
    assert heading_confidences[2] == pytest.approx(0.92, abs=1e-6)
    assert heading_confidences[3] == pytest.approx(0.91, abs=1e-6)
    assert directed_mask == [0, 0, 1, 1]
    assert heading_hints[2] == pytest.approx((0.5 + np.pi) % (2 * np.pi), abs=1e-6)
    assert heading_hints[3] == pytest.approx(0.5, abs=1e-6)


def test_filter_overlapping_uses_precise_iou_for_all_overlaps() -> None:
    mod = _load_engine_module()
    det = mod.YOLOOBBDetector.__new__(mod.YOLOOBBDetector)

    calls = {"indices": []}

    def _fake_precise(_corners1, _corners_list, indices):
        calls["indices"].append(list(indices))
        # Suppress only the first overlapping candidate.
        vals = [0.9 if idx == 1 else 0.2 for idx in indices]
        return np.asarray(vals, dtype=np.float32)

    det._compute_obb_iou_batch = _fake_precise

    meas = [
        np.array([10.0, 10.0, 0.0], dtype=np.float32),
        np.array([11.0, 10.0, 0.0], dtype=np.float32),
        np.array([9.0, 10.0, 0.0], dtype=np.float32),
    ]
    sizes = [100.0, 90.0, 80.0]
    shapes = [(100.0, 1.0), (90.0, 1.0), (80.0, 1.0)]
    confidences = [0.95, 0.9, 0.85]
    corners = [
        np.array([[8, 8], [12, 8], [12, 12], [8, 12]], dtype=np.float32),
        np.array([[9, 8], [13, 8], [13, 12], [9, 12]], dtype=np.float32),
        np.array([[7, 8], [11, 8], [11, 12], [7, 12]], dtype=np.float32),
    ]

    out = det._filter_overlapping_detections(
        meas,
        sizes,
        shapes,
        confidences,
        corners,
        iou_threshold=0.5,
    )
    out_meas, out_sizes, _, _, _ = out

    assert calls["indices"] and calls["indices"][0] == [1, 2]
    assert len(out_meas) == 2
    assert out_sizes == [100.0, 80.0]


def test_loads_notebook_tiny_headtail_state_dict(tmp_path: Path) -> None:
    """HeadTailAnalyzer loads a notebook-style raw state_dict checkpoint."""
    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    # Build and save a notebook-style raw state_dict checkpoint
    tiny = HeadTailAnalyzer._build_tiny_classifier(input_size=(128, 64))
    ckpt_path = tmp_path / "tiny_headtail.pth"
    import torch

    torch.save(tiny.state_dict(), ckpt_path)

    analyzer = HeadTailAnalyzer(
        model_path=str(ckpt_path), device="cpu", conf_threshold=0.5
    )

    assert analyzer.backend == "tiny"
    assert analyzer.is_available
    assert analyzer.model is not None


def test_tiny_headtail_inference_converts_bgr_to_rgb() -> None:
    """HeadTailAnalyzer._predict converts BGR crops to RGB for tiny backend."""
    import torch.nn as nn

    from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

    class _Probe(nn.Module):
        # Logit uses channel0 - channel2. If RGB conversion is correct, BGR-red
        # input becomes RGB-red and yields a strongly positive logit.
        def forward(self, x):
            v = x[:, 0, 0, 0] - x[:, 2, 0, 0]
            return v.unsqueeze(1) * 10.0

    analyzer = HeadTailAnalyzer.from_components(
        model=_Probe().eval(),
        backend="tiny",
        class_names=None,
        input_size=None,
        device="cpu",
    )

    bgr_red = np.array([[[0, 0, 255]]], dtype=np.uint8)
    probs = analyzer._predict([bgr_red])

    assert len(probs) == 1
    assert float(probs[0]) > 0.9
