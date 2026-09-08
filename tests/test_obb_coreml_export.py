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


def test_load_obb_executor_coreml_missing_no_autoexport(tmp_path, monkeypatch):
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

    monkeypatch.setattr(ra, "_load_torch_model", fake_load_torch)
    monkeypatch.setattr(
        ra,
        "_runtime_artifact_store_for",
        lambda source: ra.RuntimeArtifactStore(
            Path(source).parent / ".test-runtime-artifacts"
        ),
    )
    with pytest.raises(ArtifactExportError, match="auto_export=False"):
        load_obb_executor(str(pt_file), "coreml", auto_export=False)


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
    monkeypatch.setattr(
        ra,
        "_runtime_artifact_store_for",
        lambda source: ra.RuntimeArtifactStore(
            Path(source).parent / ".test-runtime-artifacts"
        ),
    )

    result = load_obb_executor(str(pt_file), "coreml", auto_export=True)
    assert calls["export"] == 1
    # The CoreML executor is now wrapped for batch-safety; it delegates to the
    # loaded model, so identity checks go through the wrapper.
    assert isinstance(result, ra._CoreMLBatchExecutor)
    assert isinstance(result._model, _FakeModel)
    assert result.names == {0: "ant"}  # attribute delegation
    assert calls["load"][-1].endswith(".mlpackage")
    assert ".test-runtime-artifacts" in calls["load"][-1]


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
    monkeypatch.setattr(
        ra,
        "_runtime_artifact_store_for",
        lambda source: ra.RuntimeArtifactStore(
            Path(source).parent / ".test-runtime-artifacts"
        ),
    )

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


# ---------------------------------------------------------------------------
# First-run export: locked, out of place, atomically published.
#
# These mock ``ultralytics`` itself rather than ``_export_artifact``: what is
# under test is precisely what happens at ``artifact_path`` WHILE the exporter
# runs, and an ``_export_artifact`` stub would be the thing deciding that.
# ---------------------------------------------------------------------------


class _FakeLoadedModel:
    names = {0: "ant"}
    overrides = {}

    def __init__(self, path=None):
        self.path = path


def _install_fake_ultralytics(monkeypatch, *, on_export=None, exports=None):
    """Fake ``ultralytics.YOLO`` whose ``export`` behaves like the real one.

    The real CoreML exporter writes ``<stem>.mlpackage`` BESIDE the ``.pt`` it
    was handed, and builds it incrementally. This fake does both, so a test can
    observe the artifact path at the one moment that matters.
    """
    import sys
    from types import SimpleNamespace

    class _FakeYOLO:
        def __init__(self, path):
            self.path = Path(path)
            self.names = {0: "ant"}
            self.overrides = {}
            self.model = SimpleNamespace(model=[SimpleNamespace(end2end=False)])

        def export(self, **kwargs):
            out = self.path.with_suffix(".mlpackage")
            out.mkdir(parents=True, exist_ok=True)
            (out / "PARTIAL").write_bytes(b"half-written")
            if exports is not None:
                exports.append(str(out))
            if on_export is not None:
                on_export(out)
            (out / "Manifest.json").write_text("{}", encoding="utf-8")
            (out / "PARTIAL").unlink()
            return str(out)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))


def test_concurrent_coreml_loads_export_once_and_share_the_artifact(
    tmp_path, monkeypatch
):
    """Two loaders that miss the cache at once must not both export.

    On an Apple host the ``gpu_fast`` tier resolves the OBB stage to CoreML, so
    every child of a parallel batch takes this path against the SAME artifact.
    Without ``artifact_build_lock`` both export, and the loser reads a
    directory the winner is still replacing.
    """
    import threading
    import time

    import hydra_suite.core.inference.runtime_artifacts as ra

    pt = tmp_path / "model.pt"
    pt.write_bytes(b"fake-checkpoint")
    monkeypatch.setattr(ra, "_load_torch_model", lambda p: _FakeLoadedModel(p))
    monkeypatch.setattr(ra, "_resolve_imgsz", lambda p: 128)

    exporting = threading.Event()
    exports: list = []

    def _slow(out):
        exporting.set()
        time.sleep(0.8)

    _install_fake_ultralytics(monkeypatch, on_export=_slow, exports=exports)

    results: dict = {}

    def _load(tag):
        try:
            results[tag] = ra.load_obb_executor(str(pt), "coreml", auto_export=True)
        except Exception as exc:  # noqa: BLE001 - reported by the assertions
            results[tag] = exc

    first = threading.Thread(target=_load, args=("first",))
    first.start()
    assert exporting.wait(10), "the first loader never reached the exporter"
    second = threading.Thread(target=_load, args=("second",))
    second.start()
    first.join(60)
    second.join(60)

    assert len(exports) == 1, f"the artifact was exported {len(exports)}x: {exports}"
    for tag in ("first", "second"):
        assert isinstance(
            results.get(tag), ra._CoreMLBatchExecutor
        ), f"{tag}: {results.get(tag)!r}"
    artifact = tmp_path / "model.mlpackage"
    assert (artifact / "Manifest.json").exists()
    assert ra._artifact_is_fresh(artifact, pt, 128)


def test_coreml_export_never_exposes_a_partial_artifact(tmp_path, monkeypatch):
    """The .mlpackage must appear at ``artifact_path`` whole or not at all.

    Exporting from the checkpoint itself put ultralytics' incremental output
    directly at ``artifact_path`` -- ``out_path == artifact_path``, so
    ``_install_artifact_atomically`` was skipped and a sibling process could
    load a directory that was still being written.
    """
    import hydra_suite.core.inference.runtime_artifacts as ra

    pt = tmp_path / "model.pt"
    pt.write_bytes(b"fake-checkpoint")
    artifact = tmp_path / "model.mlpackage"
    monkeypatch.setattr(ra, "_load_torch_model", lambda p: _FakeLoadedModel(p))
    monkeypatch.setattr(ra, "_resolve_imgsz", lambda p: 128)

    seen: dict = {}

    def _probe(out):
        seen["out_path"] = out
        seen["artifact_exists"] = artifact.exists()
        seen["marker_exists"] = ra._meta_path(artifact).exists()

    _install_fake_ultralytics(monkeypatch, on_export=_probe)

    ra.load_obb_executor(str(pt), "coreml", auto_export=True)

    assert seen["out_path"] != artifact, "the export still ran in place"
    assert seen["artifact_exists"] is False, "a partial .mlpackage was visible"
    assert seen["marker_exists"] is False, "the freshness marker outlived the rebuild"
    assert (artifact / "Manifest.json").exists()
    assert not (artifact / "PARTIAL").exists()
    leftovers = [p.name for p in tmp_path.glob(".model.mlpackage.*")]
    assert leftovers == [], leftovers


def test_stale_coreml_artifact_is_replaced_only_once_the_new_one_is_whole(
    tmp_path, monkeypatch
):
    """A rebuild over a STALE artifact must clear the marker first and leave
    the old directory intact until the finished one swaps in."""
    import os

    import hydra_suite.core.inference.runtime_artifacts as ra

    pt = tmp_path / "model.pt"
    pt.write_bytes(b"fake-checkpoint")
    artifact = tmp_path / "model.mlpackage"
    artifact.mkdir()
    (artifact / "OLD").write_text("previous artifact", encoding="utf-8")
    ra._write_fresh_marker(artifact, pt, 128)
    assert ra._artifact_is_fresh(artifact, pt, 128)
    # Invalidate it the way a re-trained checkpoint would.
    stat = pt.stat()
    os.utime(pt, (stat.st_atime + 10, stat.st_mtime + 10))
    assert not ra._artifact_is_fresh(artifact, pt, 128)

    monkeypatch.setattr(ra, "_load_torch_model", lambda p: _FakeLoadedModel(p))
    monkeypatch.setattr(ra, "_resolve_imgsz", lambda p: 128)

    during: dict = {}

    def _probe(out):
        during["marker"] = ra._meta_path(artifact).exists()
        during["old_intact"] = (artifact / "OLD").exists()
        during["no_partial_at_artifact"] = not (artifact / "PARTIAL").exists()

    _install_fake_ultralytics(monkeypatch, on_export=_probe)

    ra.load_obb_executor(str(pt), "coreml", auto_export=True)

    assert during["marker"] is False, "the stale marker survived into the rebuild"
    assert during["old_intact"] is True, "the old artifact was destroyed mid-build"
    assert during["no_partial_at_artifact"] is True
    assert (artifact / "Manifest.json").exists()
    assert not (artifact / "OLD").exists(), "the new artifact never replaced the old"
    assert ra._artifact_is_fresh(artifact, pt, 128)
    leftovers = [p.name for p in tmp_path.glob(".model.mlpackage.*")]
    assert leftovers == [], leftovers
