"""CoreML determinism + .mlpackage freshness tests (Task 6, Phase 3).

Tests:
1. ClassifierBackend with compute_runtime="coreml" produces byte-identical
   outputs for the same input on two consecutive predict_batch calls.
2. A .mlpackage artifact is still considered fresh after a simulated
   load-induced Manifest.json touch (i.e., freshness is sidecar-based, not
   based on the package directory mtime).

Both tests require coremltools and Apple MPS; they are skipped elsewhere.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

coremltools = pytest.importorskip("coremltools")

_MODEL_PATH = Path(
    os.path.expanduser(
        "~/Library/Application Support/hydra-suite/models/classification"
        "/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth"
    )
)

_MPS_AVAILABLE = __import__(
    "hydra_suite.utils.gpu_utils", fromlist=["MPS_AVAILABLE"]
).MPS_AVAILABLE


def _model_available() -> bool:
    return _MODEL_PATH.exists()


# ---------------------------------------------------------------------------
# Freshness: .mlpackage stays fresh after a Manifest.json touch.
# ---------------------------------------------------------------------------


def test_mlpackage_freshness_survives_manifest_touch(tmp_path):
    """Freshness marker survives a simulated load-induced Manifest.json rewrite.

    coremltools rewrites Manifest.json inside the .mlpackage on every MLModel
    load. Freshness must be judged by the sidecar .runtime_meta.json (keyed to
    the source .pt mtime), NOT by the package directory mtime. This test
    creates a synthetic scenario and asserts _artifact_is_fresh still returns
    True after Manifest.json is touched inside the package.
    """
    from hydra_suite.core.inference.runtime_artifacts import (
        _artifact_is_fresh,
        _write_fresh_marker,
    )

    # Create a fake .pt source file.
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"fake-checkpoint")

    # Create a fake .mlpackage directory + Manifest.json (simulating an export).
    pkg = tmp_path / "model.mlpackage"
    pkg.mkdir()
    manifest = pkg / "Manifest.json"
    manifest.write_text('{"version": 1}', encoding="utf-8")

    # Write the freshness sidecar keyed to the source .pt mtime.
    _write_fresh_marker(pkg, pt, 640)

    # Confirm fresh before any mutation.
    assert _artifact_is_fresh(
        pkg, pt, 640
    ), "should be fresh right after marker is written"

    # Simulate coremltools rewriting Manifest.json on MLModel load: advance its
    # mtime so it is newer than the sidecar.
    time.sleep(0.01)  # ensure measurable mtime difference
    manifest.write_text(
        '{"version": 2, "rewritten_by_coremltools": true}', encoding="utf-8"
    )
    manifest.touch()  # force mtime update

    # The sidecar still records the original .pt mtime; freshness must hold.
    assert _artifact_is_fresh(pkg, pt, 640), (
        "freshness should survive Manifest.json touch because the sidecar "
        "records source .pt mtime, not the package directory mtime"
    )


# ---------------------------------------------------------------------------
# Determinism: two predict_batch calls on identical input produce equal output.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _MPS_AVAILABLE, reason="Apple MPS required for CoreML")
@pytest.mark.skipif(not _model_available(), reason="orientation model not found")
def test_coreml_classifier_is_deterministic_run_to_run():
    """CoreML predict_batch is byte-identical for identical input on two calls.

    Constructs a ClassifierBackend on the real orientation .pth with
    compute_runtime="coreml", runs predict_batch twice on the same 8 fixed
    crops, and asserts np.array_equal on all per-factor probability vectors.
    """
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    rng = np.random.default_rng(42)
    crops = [rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(8)]

    backend = ClassifierBackend(str(_MODEL_PATH), compute_runtime="coreml")
    try:
        out1 = backend.predict_batch(crops)
        out2 = backend.predict_batch(crops)
    finally:
        backend.close()

    assert len(out1) == len(out2) == 8, "both calls must return 8 crop results"

    for i, (row1, row2) in enumerate(zip(out1, out2)):
        assert len(row1) == len(row2), f"crop {i}: factor count mismatch"
        for k, (v1, v2) in enumerate(zip(row1, row2)):
            assert np.array_equal(v1, v2), (
                f"crop {i}, factor {k}: CoreML output not byte-identical "
                f"between two runs.\nrun1={v1}\nrun2={v2}"
            )
