from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _gpu_stub(**overrides):
    def _identity_decorator(*_args, **_kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    base = {
        "CUDA_AVAILABLE": False,
        "MPS_AVAILABLE": False,
        "ONNXRUNTIME_AVAILABLE": True,
        "ONNXRUNTIME_PROVIDERS": ["CPUExecutionProvider"],
        "ONNXRUNTIME_CPU_AVAILABLE": True,
        "ONNXRUNTIME_COREML_AVAILABLE": False,
        "ONNXRUNTIME_CUDA_AVAILABLE": False,
        "TENSORRT_AVAILABLE": False,
        "TORCH_CUDA_AVAILABLE": False,
        "SLEAP_RUNTIME_ONNX_AVAILABLE": True,
        "SLEAP_RUNTIME_TENSORRT_AVAILABLE": False,
        "NUMBA_AVAILABLE": False,
        "GPU_AVAILABLE": False,
        "ANY_ACCELERATION": False,
        "CUPY_AVAILABLE": False,
        "TORCH_AVAILABLE": False,
        "F": None,
        "cp": None,
        "cupy_ndimage": None,
        "njit": _identity_decorator,
        "prange": range,
        "torch": None,
        "get_device_info": lambda: {},
        "log_device_info": lambda: None,
    }
    base.update(overrides)
    mod = types.ModuleType("hydra_suite.utils.gpu_utils")
    for key, value in base.items():
        setattr(mod, key, value)
    return mod


@contextmanager
def _patched_modules(stubs: dict):
    sentinel = object()
    original = {}
    try:
        for name, stub in stubs.items():
            original[name] = sys.modules.get(name, sentinel)
            sys.modules[name] = stub
        yield
    finally:
        for name, old in original.items():
            if old is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_runtime_api_module(stubs: dict):
    module_name = "runtime_api_sleap_export_test"
    module_path = SRC_ROOT / "hydra_suite" / "core" / "identity" / "pose" / "api.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _patched_modules(stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def _load_sleap_backend_module(stubs: dict):
    module_name = "sleap_backend_under_test"
    module_path = (
        SRC_ROOT
        / "hydra_suite"
        / "core"
        / "identity"
        / "pose"
        / "backends"
        / "sleap.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _patched_modules(stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def test_sleap_export_backend_onnx_predicts_canonical_output(tmp_path: Path) -> None:
    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(
            MPS_AVAILABLE=True,
            ONNXRUNTIME_COREML_AVAILABLE=True,
        ),
    }
    mod = _load_sleap_backend_module(stubs)

    export_dir = tmp_path / "sleap_exported"
    export_dir.mkdir()
    (export_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.onnx").write_bytes(b"fake-onnx")

    providers_seen = []

    class _FakeInput:
        name = "image"
        shape = ["batch", 3, 64, 64]
        type = "tensor(uint8)"

    class _FakeOutput:
        def __init__(self, name):
            self.name = name

    class _FakeSession:
        def __init__(self, path, providers):
            providers_seen.append((str(path), list(providers)))
            self._providers = list(providers)

        def get_inputs(self):
            return [_FakeInput()]

        def get_outputs(self):
            return [_FakeOutput("peaks"), _FakeOutput("peak_vals")]

        def run(self, _unused, feed_dict):
            batch = feed_dict["image"]
            count = int(batch.shape[0])
            peaks = np.zeros((count, 3, 2), dtype=np.float32)
            peak_vals = np.zeros((count, 3), dtype=np.float32)
            for i in range(count):
                peaks[i, 0] = (10.0 + i, 20.0)
                peaks[i, 1] = (30.0 + i, 40.0)
                peaks[i, 2] = (50.0 + i, 60.0)
                peak_vals[i] = (0.9, 0.7, 0.1)
            return [peaks, peak_vals]

    sys.modules["onnxruntime"] = types.SimpleNamespace(InferenceSession=_FakeSession)

    with _patched_modules(stubs):
        backend = mod.SleapExportedBackend(
            exported_model_path=str(export_dir),
            runtime_flavor="onnx",
            runtime_request="onnx_coreml",
            device="mps",
            keypoint_names=["k1", "k2", "k3"],
            min_valid_conf=0.2,
            batch_size=4,
        )

        crops = [
            np.zeros((24, 24, 3), dtype=np.uint8),
            np.zeros((26, 26, 3), dtype=np.uint8),
        ]
        out = backend.predict_batch(crops)
        assert providers_seen
        first_provider = providers_seen[0][1][0]
        if isinstance(first_provider, tuple):
            first_provider = first_provider[0]
        assert first_provider == "CoreMLExecutionProvider"
        assert len(out) == 2
        assert out[0].num_keypoints == 3
        assert out[0].num_valid == 2
        assert np.isclose(out[0].mean_conf, (0.9 + 0.7 + 0.1) / 3.0)
        assert out[0].keypoints.shape == (3, 3)
        backend.close()


def test_prepare_export_crop_uses_rgb_letterbox_and_inverse_mapping() -> None:
    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(),
    }
    mod = _load_sleap_backend_module(stubs)

    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    crop[:, :, 0] = 11
    crop[:, :, 1] = 22
    crop[:, :, 2] = 33

    prepared, transform = mod._prepare_export_crop(crop, (64, 64), 3)

    assert prepared.shape == (64, 64, 3)
    assert np.all(prepared[:16, :, :] == 0)
    assert np.all(prepared[48:, :, :] == 0)
    assert np.all(prepared[16:48, :, 0] == 33)
    assert np.all(prepared[16:48, :, 1] == 22)
    assert np.all(prepared[16:48, :, 2] == 11)

    model_space = np.array(
        [
            [0.0, 16.0, 0.9],
            [60.8, 44.8, 0.7],
        ],
        dtype=np.float32,
    )
    restored = mod._restore_export_keypoints(model_space, transform)
    assert np.allclose(restored[:, :2], np.array([[0.0, 0.0], [19.0, 9.0]]), atol=1e-3)
    assert np.allclose(restored[:, 2], np.array([0.9, 0.7]), atol=1e-6)


def test_sleap_native_backend_uses_in_memory_transport_when_supported(
    tmp_path: Path,
) -> None:
    calls = []

    class _FakePoseInferenceService:
        def __init__(self, out_root, keypoint_names, skeleton_edges=None):
            self.out_root = Path(out_root)
            self.keypoint_names = list(keypoint_names)
            self.skeleton_edges = list(skeleton_edges or [])

        @classmethod
        def sleap_service_running(cls):
            return False

        @classmethod
        def start_sleap_service(cls, env_name, out_root):
            return True, "", Path(out_root) / "log.txt"

        @classmethod
        def shutdown_sleap_service(cls):
            return None

        @classmethod
        def sleap_native_array_video_supported(cls):
            return True

        def predict(self, model_path, image_paths, **kwargs):
            calls.append(
                {
                    "model_path": str(model_path),
                    "image_paths": list(image_paths),
                    "kwargs": dict(kwargs),
                }
            )
            preds = {}
            for i, p in enumerate(image_paths):
                preds[str(p)] = [
                    (10.0 + i, 20.0, 0.9),
                    (30.0 + i, 40.0, 0.7),
                ]
            return preds, ""

    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(),
        "hydra_suite.integrations.sleap.service": types.SimpleNamespace(
            PoseInferenceService=_FakePoseInferenceService
        ),
    }
    mod = _load_sleap_backend_module(stubs)

    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir()

    with _patched_modules(stubs):
        backend = mod.SleapServiceBackend(
            model_dir=str(model_dir),
            out_root=str(tmp_path),
            keypoint_names=["k1", "k2"],
            min_valid_conf=0.2,
            sleap_env="sleap",
            sleap_device="cpu",
            sleap_batch=4,
            sleap_max_instances=1,
            runtime_flavor="native",
        )

        crops = [
            np.zeros((24, 24, 3), dtype=np.uint8),
            np.zeros((26, 26, 3), dtype=np.uint8),
        ]
        out = backend.predict_batch(crops)

        assert calls
        assert calls[0]["kwargs"]["sleap_runtime_flavor"] == "native"
        assert calls[0]["kwargs"]["image_payloads"] is not None
        assert len(calls[0]["kwargs"]["image_payloads"]) == 2
        assert calls[0]["kwargs"]["image_payloads"][0]["shm_name"]
        assert calls[0]["kwargs"]["cache_predictions"] is False
        assert len(out) == 2
        assert out[0].num_valid == 2
        backend.close()


def test_sleap_export_backend_falls_back_to_service_when_unavailable(
    tmp_path: Path,
) -> None:
    class _FakePoseInferenceService:
        def __init__(self, out_root, keypoint_names, skeleton_edges=None):
            self.out_root = Path(out_root)
            self.keypoint_names = list(keypoint_names)
            self.skeleton_edges = list(skeleton_edges or [])

        @classmethod
        def sleap_service_running(cls):
            return False

        @classmethod
        def start_sleap_service(cls, env_name, out_root):
            return True, "", out_root / "log.txt"

        @classmethod
        def shutdown_sleap_service(cls):
            return None

        def predict(self, model_path, image_paths, **_kwargs):
            preds = {}
            for p in image_paths:
                preds[str(p)] = [(1.0, 2.0, 0.9) for _ in self.keypoint_names]
            return preds, ""

    def _raise_load_exported_model(*_args, **_kwargs):
        raise RuntimeError("export backend unavailable")

    predictors_mod = types.ModuleType("sleap_nn.export.predictors")
    predictors_mod.load_exported_model = _raise_load_exported_model
    metadata_mod = types.ModuleType("sleap_nn.export.metadata")
    metadata_mod.load_metadata = lambda _p: {}
    export_mod = types.ModuleType("sleap_nn.export")
    export_mod.predictors = predictors_mod
    export_mod.metadata = metadata_mod
    sleap_nn_mod = types.ModuleType("sleap_nn")
    sleap_nn_mod.export = export_mod

    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(),
        "sleap_nn": sleap_nn_mod,
        "sleap_nn.export": export_mod,
        "sleap_nn.export.predictors": predictors_mod,
        "sleap_nn.export.metadata": metadata_mod,
        "hydra_suite.integrations.sleap.service": types.SimpleNamespace(
            PoseInferenceService=_FakePoseInferenceService
        ),
    }
    mod = _load_runtime_api_module(stubs)

    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir()
    export_dir = tmp_path / "sleap_exported"
    export_dir.mkdir()
    cfg = mod.PoseRuntimeConfig(
        backend_family="sleap",
        runtime_flavor="onnx",
        device="cpu",
        model_path=str(model_dir),
        exported_model_path=str(export_dir),
        min_valid_conf=0.2,
        keypoint_names=["head", "thorax", "abdomen"],
        sleap_batch=2,
        sleap_max_instances=1,
    )

    with _patched_modules(stubs):
        backend = mod.create_pose_backend_from_config(cfg)
        assert isinstance(backend, mod.SleapServiceBackend)
        backend.close()


def test_create_pose_backend_uses_direct_exported_backend_for_sleap_onnx(
    tmp_path: Path,
) -> None:
    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(MPS_AVAILABLE=True),
    }
    mod = _load_runtime_api_module(stubs)

    export_dir = tmp_path / "sleap_exported"
    export_dir.mkdir()
    (export_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (export_dir / "model.onnx").write_bytes(b"fake-onnx")

    class _FakeInput:
        name = "image"
        shape = ["batch", 3, 64, 64]
        type = "tensor(uint8)"

    class _FakeOutput:
        def __init__(self, name):
            self.name = name

    class _FakeSession:
        def __init__(self, _path, providers=None, **_kwargs):
            self._providers = list(providers or [])

        def get_inputs(self):
            return [_FakeInput()]

        def get_outputs(self):
            return [_FakeOutput("peaks"), _FakeOutput("peak_vals")]

        def run(self, _unused, feed_dict):
            batch = feed_dict["image"]
            count = int(batch.shape[0])
            peaks = np.zeros((count, 1, 2), dtype=np.float32)
            peak_vals = np.ones((count, 1), dtype=np.float32)
            return [peaks, peak_vals]

    sys.modules["onnxruntime"] = types.SimpleNamespace(InferenceSession=_FakeSession)

    cfg = mod.PoseRuntimeConfig(
        backend_family="sleap",
        runtime_flavor="onnx_mps",
        device="mps",
        model_path=str(tmp_path / "sleap_model"),
        exported_model_path=str(export_dir),
        min_valid_conf=0.2,
        keypoint_names=["head"],
        sleap_batch=2,
        sleap_max_instances=1,
    )

    with _patched_modules(stubs):
        backend = mod.create_pose_backend_from_config(cfg)
        assert isinstance(backend, mod.SleapExportedBackend)
        backend.close()


def test_yolo_auto_runtime_exports_onnx_and_reuses_cached_artifact(
    tmp_path: Path,
) -> None:
    export_calls = []

    class _FakeYOLO:
        def __init__(self, path):
            self.path = str(path)

        def to(self, _device):
            return self

        def export(self, **kwargs):
            export_calls.append(kwargs)
            out = tmp_path / "tmp_export.onnx"
            out.write_bytes(b"onnx-bytes")
            return str(out)

    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(SLEAP_RUNTIME_ONNX_AVAILABLE=False),
        "ultralytics": types.SimpleNamespace(YOLO=_FakeYOLO),
    }
    mod = _load_runtime_api_module(stubs)

    model_path = tmp_path / "pose.pt"
    model_path.write_bytes(b"pt-bytes")
    cfg = mod.PoseRuntimeConfig(
        backend_family="yolo",
        runtime_flavor="auto",
        device="cpu",
        model_path=str(model_path),
        out_root=str(tmp_path),
        yolo_batch=4,
        keypoint_names=["head", "tail"],
    )

    with _patched_modules(stubs):
        b1 = mod.create_pose_backend_from_config(cfg)
        p1 = Path(b1.model_path)
        assert p1.suffix == ".onnx"
        assert p1.exists()
        b1.close()

        b2 = mod.create_pose_backend_from_config(cfg)
        p2 = Path(b2.model_path)
        b2.close()

    assert p1 == p2
    assert len(export_calls) == 1


def test_yolo_auto_runtime_reexports_when_model_changes(tmp_path: Path) -> None:
    export_calls = []

    class _FakeYOLO:
        def __init__(self, path):
            self.path = str(path)

        def to(self, _device):
            return self

        def export(self, **kwargs):
            export_calls.append(kwargs)
            out = tmp_path / f"tmp_export_{len(export_calls)}.onnx"
            out.write_bytes(f"onnx-{len(export_calls)}".encode("utf-8"))
            return str(out)

    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(SLEAP_RUNTIME_ONNX_AVAILABLE=False),
        "ultralytics": types.SimpleNamespace(YOLO=_FakeYOLO),
    }
    mod = _load_runtime_api_module(stubs)

    model_path = tmp_path / "pose.pt"
    model_path.write_bytes(b"pt-v1")
    cfg = mod.PoseRuntimeConfig(
        backend_family="yolo",
        runtime_flavor="auto",
        device="cpu",
        model_path=str(model_path),
        out_root=str(tmp_path),
        yolo_batch=4,
        keypoint_names=["head", "tail"],
    )

    with _patched_modules(stubs):
        b1 = mod.create_pose_backend_from_config(cfg)
        p1 = Path(b1.model_path)
        b1.close()

        # Change source model fingerprint.
        model_path.write_bytes(b"pt-v2-with-different-size")

        b2 = mod.create_pose_backend_from_config(cfg)
        p2 = Path(b2.model_path)
        b2.close()

    assert p1 == p2
    assert len(export_calls) >= 2
    assert p2.read_bytes() == b"onnx-2"


def test_yolo_backend_rejects_directory_model_path_with_clear_error(
    tmp_path: Path,
) -> None:
    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(SLEAP_RUNTIME_ONNX_AVAILABLE=False),
        "ultralytics": types.SimpleNamespace(YOLO=lambda *_a, **_k: None),
    }
    mod = _load_runtime_api_module(stubs)

    sleap_like_dir = tmp_path / "sleap_model_dir"
    sleap_like_dir.mkdir()
    cfg = mod.PoseRuntimeConfig(
        backend_family="yolo",
        runtime_flavor="native",
        device="cpu",
        model_path=str(sleap_like_dir),
        out_root=str(tmp_path),
        keypoint_names=["head", "tail"],
    )

    with _patched_modules(stubs):
        try:
            mod.create_pose_backend_from_config(cfg)
            raise AssertionError(
                "Expected RuntimeError for directory model path under yolo backend."
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            assert "pose_model_type" in msg
            assert "sleap" in msg


def test_attempt_sleap_cli_export_prefers_size_aware_commands_first(
    tmp_path: Path,
) -> None:
    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(),
    }
    mod = _load_sleap_backend_module(stubs)

    model_dir = tmp_path / "sleap_model"
    export_dir = tmp_path / "sleap_export"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    seen_cmds = []

    def _fake_run(cmd, timeout_sec=1800):
        seen_cmds.append(list(cmd))
        return False, "fail"

    mod.run_cli_command = _fake_run
    ok, _err = mod._attempt_sleap_cli_export(
        model_dir=model_dir,
        export_dir=export_dir,
        runtime_flavor="onnx",
        sleap_env="",
        input_hw=(320, 352),
    )
    assert ok is False
    assert seen_cmds
    first = seen_cmds[0]
    assert "--input-height" in first
    assert "--input-width" in first
    assert "320" in first
    assert "352" in first


def test_attempt_sleap_cli_export_includes_batch_profile(
    tmp_path: Path,
) -> None:
    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(),
    }
    mod = _load_sleap_backend_module(stubs)

    model_dir = tmp_path / "sleap_model"
    export_dir = tmp_path / "sleap_export"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    seen_cmds = []

    def _fake_run(cmd, timeout_sec=1800):
        _ = timeout_sec
        seen_cmds.append(list(cmd))
        return False, "fail"

    mod.run_cli_command = _fake_run
    ok, _err = mod._attempt_sleap_cli_export(
        model_dir=model_dir,
        export_dir=export_dir,
        runtime_flavor="tensorrt",
        sleap_env="",
        input_hw=(224, 224),
        batch_size=8,
    )
    assert ok is False
    assert seen_cmds
    first = seen_cmds[0]
    assert "--batch-size" in first or "--batch" in first
    assert "8" in first


def test_coerce_prediction_batch_normalizes_out_of_range_confidences() -> None:
    stubs = {
        "hydra_suite.utils.gpu_utils": _gpu_stub(),
    }
    pose_utils = importlib.util.spec_from_file_location(
        "pose_utils_under_test",
        SRC_ROOT / "hydra_suite" / "core" / "identity" / "pose" / "utils.py",
    )
    utils_module = importlib.util.module_from_spec(pose_utils)
    with _patched_modules(stubs):
        sys.modules["pose_utils_under_test"] = utils_module
        try:
            pose_utils.loader.exec_module(utils_module)
        finally:
            sys.modules.pop("pose_utils_under_test", None)

    # Mimic exported ONNX outputs that provide logits instead of probabilities.
    pred_out = {
        "instance_peaks": np.array([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]),
        "instance_peak_vals": np.array([[2.0, 0.0, -2.0]], dtype=np.float32),
    }
    kpts = utils_module.coerce_prediction_batch(pred_out, batch_size=1)[0]
    assert kpts is not None
    assert np.all((kpts[:, 2] >= 0.0) & (kpts[:, 2] <= 1.0))
    assert np.isclose(kpts[0, 2], 0.880797, atol=1e-5)
    assert np.isclose(kpts[1, 2], 0.5, atol=1e-6)
    assert np.isclose(kpts[2, 2], 0.119203, atol=1e-5)
