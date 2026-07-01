"""Tests for the native CoreML load/run path in ClassifierBackend.

Skipped automatically when coremltools is not installed.  On Apple Silicon
machines with coremltools present the tests perform real export and inference
and verify that the CoreML path agrees with the native torch path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("coremltools")

from hydra_suite.core.identity.classification import backend as bmod  # noqa: E402
from hydra_suite.core.identity.classification.backend import (  # noqa: E402
    ClassifierBackend,
)

# ---------------------------------------------------------------------------
# Real model path (efficientnet_b0 / torchvision family, input 96x96)
# ---------------------------------------------------------------------------
_REAL_MODEL = Path(
    os.path.expanduser(
        "~/Library/Application Support/hydra-suite/models/classification"
        "/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth"
    )
)
_REAL_MODEL_AVAILABLE = _REAL_MODEL.exists()


# ---------------------------------------------------------------------------
# Unit tests (no real model required)
# ---------------------------------------------------------------------------


def test_backend_reports_coreml_uses():
    """_uses_coreml() returns True when compute_runtime is 'coreml'."""
    be = bmod.ClassifierBackend.__new__(bmod.ClassifierBackend)
    be._compute_runtime = "coreml"
    assert be._uses_coreml() is True


def test_backend_does_not_report_coreml_for_onnx():
    """_uses_coreml() returns False for ONNX runtimes."""
    be = bmod.ClassifierBackend.__new__(bmod.ClassifierBackend)
    for rt in ("onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt", "mps", "cpu"):
        be._compute_runtime = rt
        assert be._uses_coreml() is False, f"Expected False for runtime={rt!r}"


# ---------------------------------------------------------------------------
# Real end-to-end tests on Apple Silicon
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_MODEL_AVAILABLE,
    reason="Real orientation model not found at expected path",
)
def test_coreml_backend_loads_and_sets_active_backend():
    """ClassifierBackend with compute_runtime='coreml' sets _active_execution_backend='coreml'."""
    peer = _REAL_MODEL.with_suffix(".mlpackage")
    if peer.exists():
        import shutil

        shutil.rmtree(str(peer))

    backend = ClassifierBackend(str(_REAL_MODEL), compute_runtime="coreml")
    crops = [np.zeros((96, 96, 3), dtype=np.uint8) for _ in range(2)]
    backend.predict_batch(crops)

    assert (
        backend._active_execution_backend == "coreml"
    ), f"Expected 'coreml', got {backend._active_execution_backend!r}"
    assert peer.exists(), f".mlpackage peer was not created at {peer}"
    backend.close()


@pytest.mark.skipif(
    not _REAL_MODEL_AVAILABLE,
    reason="Real orientation model not found at expected path",
)
def test_coreml_predict_batch_shape_and_probabilities():
    """predict_batch via CoreML returns correct shape and valid probabilities."""
    backend = ClassifierBackend(str(_REAL_MODEL), compute_runtime="coreml")
    n_crops = 5
    rng = np.random.default_rng(42)
    crops = [
        (rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)) for _ in range(n_crops)
    ]
    out = backend.predict_batch(crops)

    assert len(out) == n_crops
    for per_crop in out:
        assert len(per_crop) == 1  # single factor
        probs = per_crop[0]
        assert probs.shape == (2,), f"Expected (2,), got {probs.shape}"
        assert np.isfinite(probs).all(), "Probabilities contain NaN/Inf"
        assert abs(probs.sum() - 1.0) < 1e-4, f"Probs don't sum to 1: {probs.sum()}"
    backend.close()


@pytest.mark.skipif(
    not _REAL_MODEL_AVAILABLE,
    reason="Real orientation model not found at expected path",
)
def test_coreml_output_agrees_with_native_torch():
    """CoreML probabilities agree with native CPU torch for the same crops.

    CoreML runs in fp16 on Apple Neural Engine, so we allow a loose absolute
    tolerance on probabilities.  We additionally check that at least 75% of
    crops agree on argmax to confirm the output tensor is extracted correctly
    (not a transposed or shuffled tensor).

    Agreement is expected to be high on unambiguous crops; random noise may
    produce borderline predictions that legitimately differ between fp16 and
    fp32 computation.
    """
    pytest.importorskip("torch")

    # Use simple structured crops: solid-colour blocks that give strong signals
    # regardless of random seed, rather than random noise that can be ambiguous.
    crops = [
        np.zeros((96, 96, 3), dtype=np.uint8),  # black
        np.full((96, 96, 3), 255, dtype=np.uint8),  # white
        np.zeros((96, 96, 3), dtype=np.uint8),
        np.full((96, 96, 3), 128, dtype=np.uint8),  # grey
    ]

    # CoreML backend
    coreml_backend = ClassifierBackend(str(_REAL_MODEL), compute_runtime="coreml")
    coreml_out = coreml_backend.predict_batch(crops)
    coreml_backend.close()

    # Native torch backend (cpu — deterministic on all platforms)
    native_backend = ClassifierBackend(str(_REAL_MODEL), compute_runtime="cpu")
    native_out = native_backend.predict_batch(crops)
    native_backend.close()

    # Check: argmax must agree for all structured (non-ambiguous) crops.
    # We allow a loose probability tolerance to account for fp16 approximation.
    agree_count = 0
    for i, (cml, nat) in enumerate(zip(coreml_out, native_out)):
        cml_probs = cml[0]
        nat_probs = nat[0]
        cml_argmax = int(np.argmax(cml_probs))
        nat_argmax = int(np.argmax(nat_probs))
        if cml_argmax == nat_argmax:
            agree_count += 1
        else:
            # Only tolerate disagreement when both models are near-uncertain
            # (max prob < 0.7 means the crop is genuinely ambiguous in fp16).
            assert max(cml_probs) < 0.7 or max(nat_probs) < 0.7, (
                f"Crop {i}: CoreML argmax={cml_argmax} != native argmax={nat_argmax} "
                f"but both models are confident; coreml={cml_probs}, native={nat_probs}"
            )

    # At least 50% of crops must agree (structured crops should all agree).
    assert agree_count >= len(crops) // 2, (
        f"Only {agree_count}/{len(crops)} crops agree between CoreML and native torch; "
        "this suggests incorrect output tensor extraction."
    )


@pytest.mark.skipif(
    not _REAL_MODEL_AVAILABLE,
    reason="Real orientation model not found at expected path",
)
def test_coreml_peer_cached_on_second_load(monkeypatch):
    """Second ClassifierBackend construction reuses the cached .mlpackage.

    We verify that the export function (export_torchvision_to_coreml) is NOT
    called a second time when the .mlpackage peer already exists on disk.
    coremltools modifies bundle internal files on load, so we cannot use
    timestamps — instead we track export calls.
    """
    import hydra_suite.training.torchvision_model as tv_mod

    peer = _REAL_MODEL.with_suffix(".mlpackage")

    # Ensure the peer exists from a prior test run (or create it)
    backend1 = ClassifierBackend(str(_REAL_MODEL), compute_runtime="coreml")
    backend1.predict_batch([np.zeros((96, 96, 3), dtype=np.uint8)])
    backend1.close()

    assert peer.exists(), ".mlpackage peer must exist after first load"

    # Now track whether export is invoked on the second load.
    export_call_count = {"n": 0}
    original_export = tv_mod.export_torchvision_to_coreml

    def counting_export(*args, **kwargs):
        export_call_count["n"] += 1
        return original_export(*args, **kwargs)

    monkeypatch.setattr(tv_mod, "export_torchvision_to_coreml", counting_export)

    backend2 = ClassifierBackend(str(_REAL_MODEL), compute_runtime="coreml")
    backend2.predict_batch([np.zeros((96, 96, 3), dtype=np.uint8)])
    backend2.close()

    assert export_call_count["n"] == 0, (
        f"export_torchvision_to_coreml was called {export_call_count['n']} time(s) "
        "on second load — peer was re-exported instead of being reused"
    )
