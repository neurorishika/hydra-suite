"""Tests for the gpu_fast + CoreML-OBB-batch=1 UI-notice helper.

CoreML's OBB export cannot use a dynamic batch axis (Spec 1 Phase A/B,
2026-07-04: ultralytics' CoreML export hard-crashes at compile time for OBB
models when both the batch and spatial dims are dynamic together), so OBB
detection under gpu_fast on Apple Silicon is permanently batch=1 even though
CoreML classification (identity/head-tail/CNN) batches normally. These tests
verify the GUI helper that identifies this case so the batch-policy notice
can say so explicitly, distinct from the TensorRT/ONNX "fixed batch" message.
"""

from __future__ import annotations

import types

from hydra_suite.trackerkit.gui.orchestrators import session as session_mod


class _FakeMainWindow:
    """Minimal stand-in exposing only what SessionOrchestrator needs here."""


def _make_orchestrator(tier: str) -> session_mod.SessionOrchestrator:
    orch = session_mod.SessionOrchestrator.__new__(session_mod.SessionOrchestrator)
    orch._mw = _FakeMainWindow()
    orch._current_runtime_tier = lambda: tier  # type: ignore[method-assign]
    return orch


def test_gpu_fast_obb_is_coreml_only_true_on_apple_silicon(monkeypatch):
    orch = _make_orchestrator("gpu_fast")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=True, has_cuda=False),
    )
    assert orch._gpu_fast_obb_is_coreml_only() is True


def test_gpu_fast_obb_is_coreml_only_false_on_cuda(monkeypatch):
    orch = _make_orchestrator("gpu_fast")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=False, has_cuda=True),
    )
    assert orch._gpu_fast_obb_is_coreml_only() is False


def test_gpu_fast_obb_is_coreml_only_false_when_tier_is_not_gpu_fast(monkeypatch):
    orch = _make_orchestrator("gpu")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=True, has_cuda=False),
    )
    assert orch._gpu_fast_obb_is_coreml_only() is False


def test_runtime_requires_fixed_yolo_batch_true_for_apple_silicon_gpu_fast(
    monkeypatch,
):
    """The existing 'fixed batch' UI gate must also fire for Apple-Silicon
    gpu_fast (CoreML OBB), not just tensorrt/onnx, so the frame-batch
    controls get disabled there too."""
    orch = _make_orchestrator("gpu_fast")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=True, has_cuda=False),
    )
    orch._selected_compute_runtime = lambda: "mps"  # type: ignore[method-assign]
    assert orch._runtime_requires_fixed_yolo_batch() is True
