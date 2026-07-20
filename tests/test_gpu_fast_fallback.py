"""Tests for gpu_fast tier best-effort TRT→native-CUDA fallback (Task 4).

These tests run on any host (Apple Silicon, CPU-only) by monkeypatching the
TRT/ONNX load path to raise, then asserting the fallback logic engages.
No real CUDA device is required.
"""

from __future__ import annotations

import logging


def test_classifier_gpu_fast_falls_back_to_native_when_onnx_load_fails(
    monkeypatch, caplog
):
    """Classifier falls back to native when ONNX/TRT peer load raises."""
    from hydra_suite.core.identity.classification import backend as bmod
    from hydra_suite.runtime.resolver import ResolvedBackend

    be = bmod.ClassifierBackend.__new__(bmod.ClassifierBackend)
    be._resolved = ResolvedBackend("tensorrt", "cuda", False)
    be._model_path = "/fake/model.pth"
    be._loaded = False
    be._active_execution_backend = "unloaded"
    be._trt_profile_max_batch = None

    # Simulate a loader that returns a sentinel when called natively.
    class _FakeLoader:
        @staticmethod
        def parse_metadata(path):
            raise AssertionError("parse_metadata should not be called here")

        @staticmethod
        def load(path, device):
            return f"NATIVE_MODEL_ON_{device}"

    be._loader = _FakeLoader()

    # Fake metadata so _uses_factor_backends() and _uses_imagenet_normalization()
    # return sensible defaults.
    from hydra_suite.core.identity.classification.backend import ClassifierMetadata

    be._metadata = ClassifierMetadata(
        arch="tinyclassifier",
        input_size=(64, 64),
        is_multihead=False,
        factor_names=["flat"],
        class_names_per_factor=[["a", "b"]],
        monochrome=False,
        recommended_confidence_threshold=None,
        source_path="/fake/model.pth",
    )

    # Force _load_onnx to raise so the best-effort fallback triggers.
    monkeypatch.setattr(
        bmod.ClassifierBackend,
        "_load_onnx",
        lambda self: (_ for _ in ()).throw(RuntimeError("no TRT engine")),
    )

    # _should_fallback_to_native_runtime normally checks ORT providers; make it
    # return False so _ensure_loaded() takes the _load_onnx() branch and raises.
    monkeypatch.setattr(
        bmod.ClassifierBackend,
        "_should_fallback_to_native_runtime",
        lambda self: False,
    )

    with caplog.at_level(logging.WARNING):
        be._ensure_loaded_best_effort()

    assert be._active_execution_backend == "native"
    assert be._loaded is True
    # The model was loaded natively on the cuda device (tensorrt→cuda mapping).
    assert be._model == "NATIVE_MODEL_ON_cuda"
    # A WARNING mentioning "fall" was emitted.
    assert any(
        "fall" in r.message.lower() for r in caplog.records
    ), f"Expected a fallback WARNING; got: {[r.message for r in caplog.records]}"


def test_obb_gpu_fast_falls_back_to_cuda_when_trt_artifact_missing(
    monkeypatch, caplog, tmp_path
):
    """OBB loader falls back to native cuda when TRT raises ArtifactExportError."""
    import hydra_suite.core.inference.stages.obb as obb_mod
    from hydra_suite.core.inference.runtime_artifacts import ArtifactExportError

    # Track which compute_runtimes load_obb_executor was called with.
    calls: list[str] = []

    def fake_load_obb_executor(
        model_path, compute_runtime, *, auto_export, max_det, **kwargs
    ):
        calls.append(str(compute_runtime))
        if compute_runtime == "tensorrt":
            raise ArtifactExportError("no .engine artifact and auto_export=False")
        # Success on cuda — return a sentinel.
        return "CUDA_MODEL"

    monkeypatch.setattr(obb_mod, "load_obb_executor", fake_load_obb_executor)

    # Use a tmp_path model path so Path operations don't crash.
    fake_pt = tmp_path / "model.pt"
    fake_pt.write_text("fake")

    with caplog.at_level(logging.WARNING):
        result = obb_mod._load_yolo(
            str(fake_pt),
            "tensorrt",
            auto_export=False,
            max_det=20,
        )

    assert result == "CUDA_MODEL", f"Expected CUDA_MODEL sentinel, got {result!r}"
    assert "tensorrt" in calls, "Expected tensorrt attempt first"
    assert "cuda" in calls, "Expected cuda fallback attempt"
    assert any(
        "fall" in r.message.lower() for r in caplog.records
    ), f"Expected a fallback WARNING; got: {[r.message for r in caplog.records]}"


def test_obb_gpu_fast_falls_back_to_cuda_when_trt_build_crashes(
    monkeypatch, caplog, tmp_path
):
    """OBB loader falls back to native cuda when TRT raises a plain RuntimeError.

    This covers the auto_export=True path where ultralytics raises RuntimeError
    (not ArtifactExportError) mid-build.  Before the fix only ArtifactExportError
    was caught, so a RuntimeError propagated as a hard crash.
    """
    import hydra_suite.core.inference.stages.obb as obb_mod

    calls: list[str] = []

    def fake_load_obb_executor(
        model_path, compute_runtime, *, auto_export, max_det, **kwargs
    ):
        calls.append(str(compute_runtime))
        if compute_runtime == "tensorrt":
            raise RuntimeError("TRT build failed")
        # Native-CUDA attempt succeeds — return a sentinel.
        return "CUDA_MODEL_AFTER_BUILD_CRASH"

    monkeypatch.setattr(obb_mod, "load_obb_executor", fake_load_obb_executor)

    fake_pt = tmp_path / "model.pt"
    fake_pt.write_text("fake")

    with caplog.at_level(logging.WARNING):
        result = obb_mod._load_yolo(
            str(fake_pt),
            "tensorrt",
            auto_export=True,
            max_det=20,
        )

    assert (
        result == "CUDA_MODEL_AFTER_BUILD_CRASH"
    ), f"Expected CUDA_MODEL_AFTER_BUILD_CRASH sentinel, got {result!r}"
    assert calls == ["tensorrt", "cuda"], f"Unexpected call sequence: {calls}"
    assert any(
        "fall" in r.message.lower() for r in caplog.records
    ), f"Expected a fallback WARNING; got: {[r.message for r in caplog.records]}"
