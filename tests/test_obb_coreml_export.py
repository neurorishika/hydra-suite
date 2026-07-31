"""CoreML export path tests for OBB runtime artifacts (Task 3 / Phase 3).

Unit tests assert path logic only and do not require coremltools or Apple hardware.
The real-export smoke test is guarded by ``pytest.importorskip("coremltools")``
and ``sys.platform == "darwin"`` so it only runs on Mac with coremltools installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_coreml_artifact_path_suffix():
    from hydra_suite.core.inference.runtime_artifacts import _artifact_path_for

    p = _artifact_path_for("model.pt", "coreml")
    assert str(p).endswith(".mlpackage")


def test_coreml_artifact_path_stem():
    from hydra_suite.core.inference.runtime_artifacts import _artifact_path_for

    p = _artifact_path_for("/some/dir/yolov8n-obb.pt", "coreml")
    assert p.stem == "yolov8n-obb"
    assert p.suffix == ".mlpackage"


def test_coreml_artifact_suffix_helper():
    from hydra_suite.core.inference.runtime_artifacts import _artifact_suffix

    assert _artifact_suffix("coreml") == ".mlpackage"


def test_coreml_runtimes_set():
    from hydra_suite.core.inference.runtime_artifacts import _COREML_RUNTIMES

    assert "coreml" in _COREML_RUNTIMES


def test_load_obb_executor_coreml_missing_no_autoexport(tmp_path):
    """coreml + auto_export=False + no .mlpackage → ArtifactExportError."""
    import hydra_suite.core.inference.runtime_artifacts as ra
    from hydra_suite.core.inference.runtime_artifacts import (
        ArtifactExportError,
        load_obb_executor,
    )

    pt_file = tmp_path / "model.pt"
    pt_file.write_bytes(b"fake")

    def fake_load_torch(model_path: str):
        class _M:
            names = {0: "ant"}
            overrides = {}

        return _M()

    monkeypatch_orig_load = ra._load_torch_model
    ra._load_torch_model = fake_load_torch
    try:
        with pytest.raises(ArtifactExportError, match="auto_export=False"):
            load_obb_executor(str(pt_file), "coreml", auto_export=False)
    finally:
        ra._load_torch_model = monkeypatch_orig_load


def test_load_obb_executor_coreml_auto_export(tmp_path, monkeypatch):
    """coreml + auto_export=True → export called, then model loaded from .mlpackage."""
    import hydra_suite.core.inference.runtime_artifacts as ra
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor

    pt_file = tmp_path / "model.pt"
    pt_file.write_bytes(b"fake")
    pt_file.touch()

    calls = {"export": 0, "load": []}

    class _FakeModel:
        names = {0: "ant"}
        overrides = {}

    def fake_load_torch(model_path: str):
        calls["load"].append(model_path)
        return _FakeModel()

    def fake_export(*, pt_path, artifact_path, runtime, imgsz, batch_size):
        calls["export"] += 1
        assert runtime == "coreml"
        # Simulate directory artifact creation.
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / "model.mlmodel").write_bytes(b"fake-mlpackage")

    monkeypatch.setattr(ra, "_load_torch_model", fake_load_torch)
    monkeypatch.setattr(ra, "_export_artifact", fake_export)

    result = load_obb_executor(str(pt_file), "coreml", auto_export=True)
    assert calls["export"] == 1
    # The CoreML executor is now wrapped for batch-safety; it delegates to the
    # loaded model, so identity checks go through the wrapper.
    assert isinstance(result, ra._CoreMLBatchExecutor)
    assert isinstance(result._model, _FakeModel)
    assert result.names == {0: "ant"}  # attribute delegation
    mlpackage = tmp_path / "model.mlpackage"
    assert mlpackage.exists()


def test_load_obb_executor_coreml_imgsz_override(tmp_path, monkeypatch):
    """coreml + imgsz_override → export uses the override, not the checkpoint's own imgsz."""
    import hydra_suite.core.inference.runtime_artifacts as ra
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor

    pt_file = tmp_path / "model.pt"
    pt_file.write_bytes(b"fake")
    pt_file.touch()

    calls = {"export_imgsz": []}

    class _FakeModel:
        names = {0: "ant"}
        overrides = {}

    def fake_load_torch(model_path: str):
        return _FakeModel()

    def fake_export(*, pt_path, artifact_path, runtime, imgsz, batch_size):
        calls["export_imgsz"].append(imgsz)
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / "model.mlmodel").write_bytes(b"fake-mlpackage")

    monkeypatch.setattr(ra, "_load_torch_model", fake_load_torch)
    monkeypatch.setattr(ra, "_export_artifact", fake_export)
    monkeypatch.setattr(ra, "_resolve_imgsz", lambda pt_path: 160)

    load_obb_executor(str(pt_file), "coreml", auto_export=True, imgsz_override=128)
    assert calls["export_imgsz"] == [128]


def test_coreml_batch_executor_dispatches_per_image():
    """A CoreML ``YOLO(mlpackage)`` is static batch=1 (``AutoBackend`` ->
    ``CoreMLBackend`` infers only ``im[0]``): a multi-image ``predict`` returns
    one ``Results`` and then ``IndexError``s inside ultralytics'
    ``stream_inference``. ``_CoreMLBatchExecutor`` must dispatch a batch one
    image at a time and concatenate the results, while a single-image call
    passes straight through (byte-identical to the working batch=1 path) and
    all other attributes delegate to the wrapped model."""
    from hydra_suite.core.inference.runtime_artifacts import _CoreMLBatchExecutor

    class _FakeCoreMLYOLO:
        task = "obb"
        names = {0: "ant"}

        def __init__(self):
            self.batch_sizes = []

        def predict(self, source, **kw):
            assert isinstance(source, (list, tuple))
            self.batch_sizes.append(len(source))
            # Model the CoreML contract the wrapper exists to satisfy: the
            # underlying executor must never be handed more than one image.
            assert len(source) == 1, "CoreML executor only supports batch=1"
            return [("R", source[0])]

    fake = _FakeCoreMLYOLO()
    ex = _CoreMLBatchExecutor(fake)

    # Multi-image batch -> per-image dispatch, one Result per image, in order.
    out = ex.predict(["a", "b", "c"], imgsz=128, verbose=False)
    assert out == [("R", "a"), ("R", "b"), ("R", "c")]
    assert fake.batch_sizes == [1, 1, 1]

    # Attribute delegation (`.task` is read by _assert_task_matches_checkpoint).
    assert ex.task == "obb"
    assert ex.names == {0: "ant"}

    # Single-image list passes straight through (one call, unchanged).
    fake.batch_sizes.clear()
    assert ex.predict(["z"]) == [("R", "z")]
    assert fake.batch_sizes == [1]


# ---------------------------------------------------------------------------
# Real-export smoke test — Apple Silicon + coremltools only.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML only on macOS")
def test_coreml_real_export_smoke(tmp_path):
    """Real export smoke: yolov8n-obb → .mlpackage → predict on random frame."""
    coremltools = pytest.importorskip("coremltools")  # noqa: F841
    import numpy as np

    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("ultralytics not installed")

    # Use yolov8n-obb.pt (ultralytics auto-downloads on first access).
    pt_name = "yolov8n-obb.pt"
    # Export to a temp dir so we don't pollute the cwd.
    import os

    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        base_model = YOLO(pt_name)
        export_path = base_model.export(
            format="coreml",
            imgsz=640,
            nms=False,
        )
        # Resolve while still in tmp_path so relative paths work.
        mlpackage_path = Path(export_path).expanduser().resolve()
    finally:
        os.chdir(orig_dir)

    assert mlpackage_path.exists(), f".mlpackage not produced at {mlpackage_path}"
    assert mlpackage_path.suffix == ".mlpackage"

    # Load back and run predict on a random frame.
    loaded = YOLO(str(mlpackage_path))
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = loaded.predict(frame, conf=0.01, verbose=False)
    assert results is not None, "predict returned None"
    # Results may be empty (no detections on noise) but must not crash.
