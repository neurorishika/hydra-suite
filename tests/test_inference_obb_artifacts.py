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
the ONNX path; the production pipeline's tier→compute-runtime-string resolution
never emits onnx_* for OBB.

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
    counters = {"export": 0, "executor": 0, "torch_load": 0, "tasks": []}

    def fake_load_torch(model_path: str):
        counters["torch_load"] += 1
        return _FakeTorchModel(model_path)

    def fake_export(*, pt_path, artifact_path, runtime, imgsz, batch_size):
        # Simulate a real export by creating the artifact file on disk.
        counters["export"] += 1
        Path(artifact_path).write_bytes(b"fake-artifact")
        return Path(artifact_path)

    def fake_executor_factory(
        *, runtime, artifact_path, imgsz, class_names=None, task="obb"
    ):
        counters["executor"] += 1
        counters["tasks"].append(task)
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
    ra._write_fresh_marker(engine, pt, ra._DEFAULT_IMGSZ)

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
    adapter = load_obb_executor(
        str(engine), "tensorrt", auto_export=False, imgsz_override=1024
    )
    assert isinstance(adapter._executor, _FakeExecutor)
    assert fake_loader["export"] == 0
    assert fake_loader["torch_load"] == 0
    assert adapter._executor.runtime == "tensorrt"
    assert adapter._executor.imgsz == 1024


def test_explicit_engine_without_imgsz_or_meta_raises(fake_loader, tmp_path):
    """No imgsz_override + no freshness sidecar must fail loudly, never silently 640."""
    engine = tmp_path / "prebuilt.engine"
    engine.write_bytes(b"prebuilt")
    with pytest.raises(ArtifactExportError) as exc:
        load_obb_executor(str(engine), "tensorrt", auto_export=False)
    assert "imgsz" in str(exc.value).lower()
    assert fake_loader["executor"] == 0


def test_explicit_engine_reads_imgsz_from_meta_sidecar(fake_loader, tmp_path):
    """Falls back to the JSON freshness sidecar recorded at export time."""
    engine = tmp_path / "prebuilt.engine"
    engine.write_bytes(b"prebuilt")
    ra._meta_path(engine).write_text(
        '{"source_mtime_ns": 0, "imgsz": 1024}', encoding="utf-8"
    )
    adapter = load_obb_executor(str(engine), "tensorrt", auto_export=False)
    assert adapter._executor.imgsz == 1024


def test_explicit_onnx_path_with_onnx_cpu_raises_unsupported(fake_loader, tmp_path):
    """onnx_cpu is not a supported OBB runtime — raises immediately."""
    onnx = tmp_path / "prebuilt.onnx"
    onnx.write_bytes(b"prebuilt")
    with pytest.raises(ArtifactExportError, match="Unsupported compute_runtime"):
        load_obb_executor(str(onnx), "onnx_cpu", auto_export=False)
    assert fake_loader["export"] == 0
    assert fake_loader["executor"] == 0


def test_artifact_path_embeds_requested_batch_size(tmp_path):
    """Different batch sizes must produce different cached artifact filenames,
    so a workflow requesting batch=8 never reuses a batch=1 (or batch=16)
    cached engine."""
    pt = tmp_path / "model.pt"
    assert ra._artifact_path_for(pt, "tensorrt", batch_size=1).name == "model_b1.engine"
    assert ra._artifact_path_for(pt, "tensorrt", batch_size=8).name == "model_b8.engine"
    # Default (no batch_size passed) preserves today's behaviour exactly.
    assert ra._artifact_path_for(pt, "tensorrt").name == "model_b1.engine"


def test_tensorrt_export_uses_dynamic_profile_when_batch_size_gt_one(
    tmp_path, monkeypatch
):
    """batch_size > 1 must export with dynamic=True (a real optimization
    profile covering 1..batch_size); batch_size == 1 must stay dynamic=False
    (today's static engine) -- this is the routing rule from Spec 1's
    Phase A/B decision (2026-07-04): realtime/batch=1 uses the un-taxed
    static engine, batch>=2 uses the dynamic engine."""
    import sys
    import types

    captured: dict = {}

    class _FakeExportYOLO:
        def __init__(self, path):
            self.path = path
            self.model = types.SimpleNamespace(
                model=[types.SimpleNamespace(end2end=False)]
            )

        def export(self, **kwargs):
            captured.update(kwargs)
            out = tmp_path / "exported.engine"
            out.write_bytes(b"fake-engine")
            return str(out)

    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.YOLO = _FakeExportYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    monkeypatch.setattr(ra, "_create_direct_executor", lambda **kw: object())

    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    load_obb_executor(str(pt), "tensorrt", auto_export=True, batch_size=8)
    assert captured["dynamic"] is True
    assert captured["batch"] == 8

    captured.clear()
    pt2 = tmp_path / "model2.pt"
    pt2.write_bytes(b"x")
    load_obb_executor(str(pt2), "tensorrt", auto_export=True, batch_size=1)
    assert captured["dynamic"] is False
    assert captured["batch"] == 1


def test_tensorrt_batch_size_two_and_eight_export_separate_cached_artifacts(
    fake_loader, tmp_path
):
    """Requesting batch_size=8 then batch_size=1 for the same .pt must export
    TWICE (two distinct cached files), not reuse/clobber one artifact."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")

    load_obb_executor(str(pt), "tensorrt", auto_export=True, batch_size=8)
    assert fake_loader["export"] == 1
    load_obb_executor(str(pt), "tensorrt", auto_export=True, batch_size=1)
    assert fake_loader["export"] == 2  # different artifact path -> re-export, not reuse

    # Requesting batch_size=8 again reuses the now-cached batch=8 artifact.
    load_obb_executor(str(pt), "tensorrt", auto_export=True, batch_size=8)
    assert fake_loader["export"] == 2


def test_load_obb_executor_segment_task_routes_to_create_direct_executor(
    fake_loader, tmp_path
):
    """``task="segment"`` must be forwarded verbatim to ``_create_direct_executor``,
    same as the existing "obb"/"detect" plumbing."""
    pt_path = tmp_path / "model.pt"
    pt_path.write_bytes(b"fake")

    load_obb_executor(str(pt_path), "tensorrt", auto_export=True, task="segment")

    assert fake_loader["tasks"][-1] == "segment"


def test_create_direct_executor_segment_task_routes_to_segment_factory(
    monkeypatch, tmp_path
):
    """``_create_direct_executor(task="segment")`` must dispatch to
    ``create_direct_segment_executor`` -- not silently fall through to the
    default OBB (or plain-detect) factory."""
    from hydra_suite.core.inference import direct_executors as dor

    calls: list[str] = []

    def fake_segment_factory(*, runtime, artifact_path, imgsz, class_names=None, **_kw):
        calls.append("segment")
        return object()

    def fail_factory(*_args, **_kwargs):
        raise AssertionError("wrong direct-executor factory was called")

    monkeypatch.setattr(dor, "create_direct_segment_executor", fake_segment_factory)
    monkeypatch.setattr(dor, "create_direct_obb_executor", fail_factory)
    monkeypatch.setattr(dor, "create_direct_detect_executor", fail_factory)

    artifact_path = tmp_path / "model.engine"
    artifact_path.write_bytes(b"fake")
    result = ra._create_direct_executor(
        runtime="tensorrt", artifact_path=artifact_path, imgsz=640, task="segment"
    )

    assert calls == ["segment"]
    assert result is not None


# ---------------------------------------------------------------------------
# Adapter wrapping: the direct executor must be exposed via a YOLO-compatible
# .predict() so stages/obb.py geometry extraction is unchanged.
# ---------------------------------------------------------------------------


def test_executor_adapter_translates_predict_kwargs():
    """The adapter wrapping a direct executor must translate the YOLO-style
    predict(conf=, iou=, classes=, verbose=, device=) call into the direct
    executor's predict(conf_thres=, classes=, max_det=) call.

    The adapter must NOT forward the caller's ``iou`` (obb.py always passes
    ``iou=1.0`` because filtering.py applies config.iou_threshold's NMS
    later) into the executor's own NMS knob — the direct executor owns its
    protective iou_thres default (0.5 for raw-CBC, 1.0 for end2end) and
    must not have it overridden by the caller.
    """
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


# ---------------------------------------------------------------------------
# Task 8: TRT engine batch sized from the REAL tile count when slicing is on.
# ---------------------------------------------------------------------------

from hydra_suite.core.inference.config import (  # noqa: E402
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
    SliceConfig,
)


def test_load_obb_models_receives_tile_batch_when_sliced(monkeypatch):
    import hydra_suite.core.inference.stages.obb as obbmod

    captured = {}

    def _fake_load(config, runtime, *, batch_size=1):
        captured["batch_size"] = batch_size
        from hydra_suite.core.inference.stages.obb import OBBModels

        return OBBModels(mode="direct", direct_model=object())

    monkeypatch.setattr(obbmod, "load_obb_models", _fake_load)

    import hydra_suite.core.inference.runner as runnermod
    from hydra_suite.core.inference.runner import _load_obb_for_config

    # 2160x3840 (4K) frame at imgsz 640, overlap 0.2 -> real tile grid is well
    # above any small hardcoded guess (verified against plan_slices directly).
    monkeypatch.setattr(runnermod, "_probe_frame_hw", lambda p: (2160, 3840))
    monkeypatch.setattr(runnermod, "_probe_model_imgsz", lambda p: 640)

    cfg = InferenceConfig(
        detection_batch_size=2,
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path="m.pt",
                slice=SliceConfig(enabled=True, geometry_mode="auto_model"),
            ),
        ),
    )

    class _RT:
        device = "cpu"
        tensor_on_cuda = False

    _load_obb_for_config(cfg, _RT(), video_path="v.mp4")

    from hydra_suite.core.inference.runner import _sliced_tile_batch

    expected = _sliced_tile_batch(cfg, (2160, 3840), 640)
    assert expected > 16, f"test must exercise a >16-tile case, got {expected}"
    assert captured["batch_size"] == expected


def test_load_obb_models_unchanged_when_slicing_disabled(monkeypatch):
    import hydra_suite.core.inference.stages.obb as obbmod

    captured = {}

    def _fake_load(config, runtime, *, batch_size=1):
        captured["batch_size"] = batch_size
        from hydra_suite.core.inference.stages.obb import OBBModels

        return OBBModels(mode="direct", direct_model=object())

    monkeypatch.setattr(obbmod, "load_obb_models", _fake_load)
    from hydra_suite.core.inference.runner import _load_obb_for_config

    cfg = InferenceConfig(
        detection_batch_size=2,
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="m.pt")),
    )

    class _RT:
        device = "cpu"
        tensor_on_cuda = False

    _load_obb_for_config(cfg, _RT(), video_path="v.mp4")
    assert captured["batch_size"] == 2  # window depth, exactly as before
