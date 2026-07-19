"""Pose golden rule: SLEAP inference must never do a disk round-trip.

These tests lock in that the disk-touching fallbacks have been removed from the
client backend (temp-file PNG transport) and the service (materialize-to-disk
video construction), and that a shared-memory failure fails loud instead of
silently retrying via disk.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _gpu_stub():
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


def _load_sleap_backend_module(stubs: dict):
    module_name = "sleap_backend_no_disk_under_test"
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
        preds = {}
        for i, p in enumerate(image_paths):
            preds[str(p)] = [(10.0 + i, 20.0, 0.9), (30.0 + i, 40.0, 0.7)]
        return preds, ""


def _make_backend(tmp_path: Path, mod):
    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir()
    return mod.SleapServiceBackend(
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


def _stubs():
    return {
        "hydra_suite.utils.gpu_utils": _gpu_stub(),
        "hydra_suite.integrations.sleap.service": types.SimpleNamespace(
            PoseInferenceService=_FakePoseInferenceService
        ),
    }


def test_backend_has_no_temp_file_transport() -> None:
    """The disk-touching temp-file fallback method must be gone entirely."""
    stubs = _stubs()
    mod = _load_sleap_backend_module(stubs)
    assert not hasattr(mod.SleapServiceBackend, "_predict_batch_via_temp_files")


def test_backend_instance_has_no_tmp_dir(tmp_path: Path) -> None:
    """No TemporaryDirectory is created on the backend anymore."""
    stubs = _stubs()
    mod = _load_sleap_backend_module(stubs)
    with _patched_modules(stubs):
        backend = _make_backend(tmp_path, mod)
        try:
            assert not hasattr(backend, "_tmp_dir")
            assert not hasattr(backend, "_tmp_root")
        finally:
            backend.close()


def test_predict_batch_fails_loud_when_shared_memory_raises(tmp_path: Path) -> None:
    """A shared-memory failure must propagate, not silently retry via disk."""
    stubs = _stubs()
    mod = _load_sleap_backend_module(stubs)
    with _patched_modules(stubs):
        backend = _make_backend(tmp_path, mod)
        try:
            sentinel = RuntimeError("shm boom")

            def _boom(_crops):
                raise sentinel

            backend._predict_batch_via_shared_memory = _boom

            crops = [np.zeros((24, 24, 3), dtype=np.uint8)]
            with pytest.raises(RuntimeError) as excinfo:
                backend.predict_batch(crops)
            assert excinfo.value is sentinel
        finally:
            backend.close()


def test_service_handler_no_longer_references_materialize_to_disk() -> None:
    """The service must not materialize image arrays to disk in the handler."""
    service_src = (
        SRC_ROOT / "hydra_suite" / "integrations" / "sleap" / "service.py"
    ).read_text(encoding="utf-8")
    assert "_materialize_image_arrays" not in service_src
    # The disk fallback log message must be gone too.
    assert "falling back to temporary image files" not in service_src


def test_backend_source_has_no_imwrite_in_predict_path() -> None:
    """The client backend must not write PNGs anywhere (no cv2.imwrite)."""
    backend_src = (
        SRC_ROOT
        / "hydra_suite"
        / "core"
        / "identity"
        / "pose"
        / "backends"
        / "sleap.py"
    ).read_text(encoding="utf-8")
    assert not re.search(r"cv2\.imwrite", backend_src)
    assert "TemporaryDirectory" not in backend_src
