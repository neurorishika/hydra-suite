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


def test_selected_compute_runtime_reports_native_coreml_on_apple_gpu_fast(
    monkeypatch,
):
    """_selected_compute_runtime must report the concrete "coreml" backend
    (not the legacy-detector-vocabulary "mps") on Apple-Silicon gpu_fast, now
    that it derives from resolve_compute_runtime() instead of the deleted
    duplicate _tier_to_compute_runtime() tier->runtime mapping."""
    orch = _make_orchestrator("gpu_fast")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=True, has_cuda=False),
    )
    assert orch._selected_compute_runtime() == "coreml"


def test_selected_compute_runtime_reports_cuda_tensorrt_on_cuda_gpu_fast(
    monkeypatch,
):
    """Sanity check the CUDA side of the same derivation still reports
    "tensorrt" for gpu_fast, matching the old _tier_to_compute_runtime
    behavior (which was correct for CUDA, only wrong for Apple Silicon)."""
    orch = _make_orchestrator("gpu_fast")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=False, has_cuda=True),
    )
    assert orch._selected_compute_runtime() == "tensorrt"


def test_main_window_preview_safe_runtime_downgrades_coreml():
    """Regression test for the LIVE call path: ``MainWindow._preview_safe_runtime``
    (called from ``detection_panel.py::_collect_preview_detection_context`` and
    ``tracking.py`` preview helpers) must downgrade the new "coreml" value that
    ``_selected_compute_runtime()`` can now report on Apple-Silicon gpu_fast to
    "mps", exactly like ``SessionOrchestrator._preview_safe_runtime`` already
    does. Before this fix, ``MainWindow``'s independent copy of this mapping
    had no "coreml" branch and returned "coreml" unchanged, which would have
    sent preview/Test-Detection/head-tail/CNN-preview through the exported
    CoreML backend instead of falling back to native MPS."""
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    assert MainWindow._preview_safe_runtime("coreml") == "mps"
    # Existing mappings must be unaffected by the fix.
    assert MainWindow._preview_safe_runtime("onnx_cpu") == "cpu"
    assert MainWindow._preview_safe_runtime("onnx_coreml") == "mps"
    assert MainWindow._preview_safe_runtime("onnx_cuda") == "cuda"
    assert MainWindow._preview_safe_runtime("tensorrt") == "cuda"
    assert MainWindow._preview_safe_runtime("mps") == "mps"
    assert MainWindow._preview_safe_runtime("cpu") == "cpu"


def test_main_window_preview_safe_runtime_delegates_to_session_orchestrator():
    """``MainWindow._preview_safe_runtime`` should delegate to
    ``SessionOrchestrator._preview_safe_runtime`` (single source of truth) so
    the two copies cannot drift out of sync again."""
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    for value in (
        "coreml",
        "onnx_coreml",
        "onnx_cpu",
        "onnx_cuda",
        "tensorrt",
        "mps",
        "cpu",
    ):
        assert MainWindow._preview_safe_runtime(
            value
        ) == session_mod.SessionOrchestrator._preview_safe_runtime(value)
