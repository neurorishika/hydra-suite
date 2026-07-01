"""Selection-logic tests for OBB runtime-artifact loading (Task 15 / H4).

These tests assert the *selection logic* of ``load_obb_executor`` on CPU using a
FAKE exporter + FAKE executor factory injected via the module's hooks. They do
NOT require a real CUDA / TensorRT installation.

The H4 bug being guarded against: when ``COMPUTE_RUNTIME`` is ``tensorrt`` the
old ``_load_yolo`` silently loaded a PyTorch ``.pt`` model and ran it on PyTorch
— never using the requested TRT runtime. The correct behaviour is to auto-export
the artifact (when ``auto_export=True``) and run a direct executor, or raise a
CLEAR error (when ``auto_export=False`` and no artifact exists) — never a silent
PyTorch fallback.

onnx_* runtimes raise ArtifactExportError immediately — OBB no longer supports
the ONNX path; the production pipeline (runtime_to_compute_runtime) never emits
onnx_* for OBB.

Real-export tests (which need ultralytics + a CUDA box) are guarded so they SKIP
locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hydra_suite.core.inference import runtime_artifacts as ra
from hydra_suite.core.inference.runtime_artifacts import (
    ArtifactExportError,
    load_obb_executor,
)


class _FakeTorchModel:
    """Stand-in for an ultralytics YOLO ``.pt`` model."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.to_calls: list[str] = []
        self.names = {0: "ant"}

    def to(self, device: str):
        self.to_calls.append(device)
        return self


class _FakeExecutor:
    """Stand-in for a direct ONNX/TRT executor."""

    def __init__(self, runtime: str, artifact_path: str, imgsz: int) -> None:
        self.runtime = runtime
        self.artifact_path = artifact_path
        self.imgsz = imgsz


@pytest.fixture
def fake_loader(monkeypatch):
    """Inject a fake torch-model loader + fake direct-executor factory.

    Returns a dict of counters the test can assert against.
    """
    counters = {"export": 0, "executor": 0, "torch_load": 0}

    def fake_load_torch(model_path: str):
        counters["torch_load"] += 1
        return _FakeTorchModel(model_path)

    def fake_export(*, pt_path, artifact_path, runtime, imgsz, batch_size):
        # Simulate a real export by creating the artifact file on disk.
        counters["export"] += 1
        Path(artifact_path).write_bytes(b"fake-artifact")
        return Path(artifact_path)

    def fake_executor_factory(*, runtime, artifact_path, imgsz, class_names=None):
        counters["executor"] += 1
        return _FakeExecutor(runtime, str(artifact_path), int(imgsz))

    monkeypatch.setattr(ra, "_load_torch_model", fake_load_torch)
    monkeypatch.setattr(ra, "_export_artifact", fake_export)
    monkeypatch.setattr(ra, "_create_direct_executor", fake_executor_factory)
    return counters


# ---------------------------------------------------------------------------
# Selection logic (CPU, fake exporter) — the core Task 15 assertions
# ---------------------------------------------------------------------------


def test_cuda_runtime_returns_torch_model_no_export(fake_loader, tmp_path):
    """compute_runtime="cuda" → plain torch model, no export attempted."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    executor = load_obb_executor(str(pt), "cuda", auto_export=True)
    assert isinstance(executor, _FakeTorchModel)
    assert fake_loader["export"] == 0
    assert fake_loader["executor"] == 0
    # cuda routes torch model onto the device.
    assert executor.to_calls == ["cuda:0"]


def test_cpu_runtime_returns_torch_model_no_export(fake_loader, tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    executor = load_obb_executor(str(pt), "cpu", auto_export=True)
    assert isinstance(executor, _FakeTorchModel)
    assert fake_loader["export"] == 0
    assert executor.to_calls == []  # cpu does not call .to()


def test_mps_runtime_returns_torch_model_no_export(fake_loader, tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    executor = load_obb_executor(str(pt), "mps", auto_export=True)
    assert isinstance(executor, _FakeTorchModel)
    assert fake_loader["export"] == 0
    assert executor.to_calls == ["mps"]


def test_tensorrt_auto_export_triggers_export_exactly_once(fake_loader, tmp_path):
    """tensorrt + auto_export=True + missing .engine → export ONCE, return executor."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    adapter = load_obb_executor(str(pt), "tensorrt", auto_export=True)
    assert isinstance(adapter, ra.DirectExecutorAdapter)
    assert isinstance(adapter._executor, _FakeExecutor)
    assert adapter._executor.runtime == "tensorrt"
    assert fake_loader["export"] == 1
    assert fake_loader["executor"] == 1


def test_tensorrt_existing_engine_skips_export(fake_loader, tmp_path):
    """If a fresh .engine already exists, no re-export — load it directly."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    # Pre-create the engine artifact + metadata so it is considered fresh.
    engine = ra._artifact_path_for(pt, "tensorrt")
    engine.write_bytes(b"prebuilt")
    ra._write_fresh_marker(engine, pt)

    adapter = load_obb_executor(str(pt), "tensorrt", auto_export=True)
    assert isinstance(adapter._executor, _FakeExecutor)
    assert fake_loader["export"] == 0  # NOT rebuilt
    assert fake_loader["executor"] == 1


def test_tensorrt_no_auto_export_missing_engine_raises_clear_error(
    fake_loader, tmp_path
):
    """tensorrt + auto_export=False + missing .engine → CLEAR error, NOT silent
    PyTorch fallback (this is the H4 bug)."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    with pytest.raises(ArtifactExportError) as exc:
        load_obb_executor(str(pt), "tensorrt", auto_export=False)
    msg = str(exc.value).lower()
    assert "auto_export" in msg or "auto-export" in msg
    assert "tensorrt" in msg
    # Crucially: no export, no torch fallback executor.
    assert fake_loader["export"] == 0
    assert fake_loader["executor"] == 0


def test_onnx_cuda_raises_unsupported(fake_loader, tmp_path):
    """onnx_cuda is not supported for OBB — production pipeline never emits it."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    with pytest.raises(ArtifactExportError, match="Unsupported compute_runtime"):
        load_obb_executor(str(pt), "onnx_cuda", auto_export=True)
    assert fake_loader["export"] == 0
    assert fake_loader["executor"] == 0


def test_explicit_engine_path_used_directly(fake_loader, tmp_path):
    """A user-supplied .engine path is used as-is (no export, no .pt loading)."""
    engine = tmp_path / "prebuilt.engine"
    engine.write_bytes(b"prebuilt")
    adapter = load_obb_executor(str(engine), "tensorrt", auto_export=False)
    assert isinstance(adapter._executor, _FakeExecutor)
    assert fake_loader["export"] == 0
    assert fake_loader["torch_load"] == 0
    assert adapter._executor.runtime == "tensorrt"


def test_explicit_onnx_path_with_onnx_cpu_raises_unsupported(fake_loader, tmp_path):
    """onnx_cpu is not a supported OBB runtime — raises immediately."""
    onnx = tmp_path / "prebuilt.onnx"
    onnx.write_bytes(b"prebuilt")
    with pytest.raises(ArtifactExportError, match="Unsupported compute_runtime"):
        load_obb_executor(str(onnx), "onnx_cpu", auto_export=False)
    assert fake_loader["export"] == 0
    assert fake_loader["executor"] == 0


# ---------------------------------------------------------------------------
# Adapter wrapping: the direct executor must be exposed via a YOLO-compatible
# .predict() so stages/obb.py geometry extraction is unchanged.
# ---------------------------------------------------------------------------


def test_executor_adapter_translates_predict_kwargs():
    """The adapter wrapping a direct executor must translate the YOLO-style
    predict(conf=, iou=, classes=, verbose=, device=) call into the direct
    executor's predict(conf_thres=, classes=, max_det=) call."""
    captured = {}

    class _DirectExec:
        def predict(self, frames, *, conf_thres, classes, max_det):
            captured["conf_thres"] = conf_thres
            captured["classes"] = classes
            captured["max_det"] = max_det
            captured["n_frames"] = len(frames)
            return ["result"]

    adapter = ra.DirectExecutorAdapter(_DirectExec(), max_det=20)
    out = adapter.predict(
        ["frame"], conf=0.01, iou=1.0, classes=[2], verbose=False, device="cuda:0"
    )
    assert out == ["result"]
    assert captured["conf_thres"] == 0.01
    assert captured["classes"] == [2]
    assert captured["max_det"] == 20
    assert captured["n_frames"] == 1


# ---------------------------------------------------------------------------
# Real-export tests — guarded so they SKIP locally (no CUDA/ONNX/TRT here).
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="real TRT export needs a CUDA device")
def test_real_tensorrt_export_roundtrip(tmp_path):  # pragma: no cover - CUDA only
    pytest.importorskip("tensorrt")
    pytest.importorskip("ultralytics")
    # A real .pt model + CUDA device would be required here. This test exists to
    # document the real-export path; it skips on non-CUDA machines.
    pytest.skip("requires a real .pt OBB model + CUDA device")
