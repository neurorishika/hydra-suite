"""YOLO-classify must see the whole animal, not ultralytics' centre crop.

YOLO-classify is the one classifier family that escapes the byte-identical
train/inference geometry guard: ultralytics applies its own
``Resize(shortest_edge)`` + ``CenterCrop(size)`` at inference and
``RandomResizedCrop`` at training, on top of whatever we hand it.
``_forward_yolo`` now pre-fits every crop to ``ClassifierMetadata.input_size``
via Layer 2 before calling the model, so the image ultralytics sees is
already square -- its own centre crop becomes a no-op. The vendor transform
is still free to do whatever it wants on top of a square input; that
residual is measured and accepted, not eliminated (operator decision,
2026-08-05).
"""

import numpy as np
import pytest


def test_prefit_makes_centre_crop_a_noop():
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input

    # A 128x64 canonical crop pre-fitted to a square is already square, so
    # ultralytics' Resize(shortest_edge) + CenterCrop(size) cannot remove
    # anything: shortest edge == longest edge.
    fit = fit_to_model_input((128, 64), (224, 224))
    out = apply_fit(np.full((64, 128, 3), 200, dtype=np.uint8), fit)
    assert out.shape[0] == out.shape[1]


def test_forward_yolo_prefits(monkeypatch):
    from hydra_suite.core.individual.classification import backend as backend_mod

    seen = []

    class _StubYolo:
        def __call__(self, crops, **kwargs):
            seen.extend(np.asarray(c) for c in crops)
            return []

    obj = object.__new__(backend_mod.ClassifierBackend)
    obj._model = _StubYolo()
    obj._metadata = type("M", (), {"input_size": (224, 224), "monochrome": False})()
    backend_mod.ClassifierBackend._forward_yolo(
        obj, [np.full((64, 128, 3), 200, dtype=np.uint8)]
    )
    assert seen[0].shape[:2] == (224, 224)


@pytest.mark.xfail(
    reason="YOLO-classify runs ultralytics' own Resize+CenterCrop (inference) and "
    "RandomResizedCrop (training); the canonical byte-identity guarantee does not "
    "extend to it. Known-lossy by operator decision, 2026-08-05. Follow-up: "
    "replace or bypass the vendor transform.",
    strict=False,
)
def test_yolo_classify_train_matches_inference():
    """Pre-fitted train and inference paths are NOT guaranteed byte-identical.

    Unlike every other classifier family (see
    test_train_inference_fit_identity.py), YOLO-classify still routes through
    ultralytics' own transform stack on both ends. The pre-fit gets both ends
    as close as the vendor pipeline allows -- measured to be exactly
    byte-identical when training uses ``scale=0.0`` (RandomResizedCrop's
    internal scale collapses to (1.0, 1.0), a full-area, no-op crop on an
    already-square source) -- but this test intentionally still fails because
    ultralytics is free to change that internal behavior out from under us at
    any time, and no test in this repo pins ultralytics' internals.
    """
    raise AssertionError(
        "byte-identity across the ultralytics vendor transform is not guaranteed"
    )
