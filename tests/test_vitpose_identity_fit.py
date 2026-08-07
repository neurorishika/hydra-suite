"""F3: ViTPose takes the identity Layer-2 fit on all devices.

Bug: ViTPose exposed `preferred_input_wh == (192, 256)`, so `model_input_wh`
returned that fixed size and `run_pose`/`run_pose_batch` applied a SECOND
Layer-2 fit (canonical canvas -> 192x256) on the non-CUDA branch, even though
ViTPose's own `preprocess_crop` (box2cs/top_down_affine) already performs the
canvas -> model-input fit internally. The redundant second resample made
MPS/CPU ViTPose diverge from both the CUDA branch (which fed the raw canvas,
no `apply_fit`) and from training (which also uses the crop's full extent,
see test_vitpose_train_infer_box_parity.py).

Fix: a `does_own_letterbox` backend flag (default False) tells
`model_input_wh` to hand back an IDENTITY fit -- `geometry.canvas_wh` --
instead of the backend's `preferred_input_wh`. `ViTPoseBackend` sets this to
True. With `model_wh == geometry.canvas_wh`, the fit computed in
`run_pose`/`run_pose_batch` is the identity affine (scale 1, zero offset), so
`apply_fit` becomes a no-op and both the CUDA and non-CUDA branches feed
(and back-project against) the same canonical canvas.
"""

from __future__ import annotations

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.stages.pose import model_input_wh


class _DoesOwnLetterboxBackend:
    """Stand-in for ViTPoseBackend: owns its own Layer-2 fit internally."""

    does_own_letterbox = True
    preferred_input_wh = (192, 256)
    preferred_input_size = 256


class _Model:
    def __init__(self, backend):
        self.backend = backend


def test_vitpose_gets_identity_fit():
    geometry = CanonicalGeometry.from_reference(60, 2.0, 1.3)

    wh = model_input_wh(_Model(_DoesOwnLetterboxBackend()), geometry)

    assert wh == geometry.canvas_wh
    assert wh != (192, 256)


def test_does_own_letterbox_defaults_to_false_when_absent():
    """A backend with no `does_own_letterbox` attribute falls back to the
    pre-existing `preferred_input_wh` behaviour (e.g. SLEAP-exported)."""

    class _NoFlagBackend:
        preferred_input_wh = (384, 256)
        preferred_input_size = 384

    geometry = CanonicalGeometry.from_reference(60, 2.0, 1.3)
    wh = model_input_wh(_Model(_NoFlagBackend()), geometry)

    assert wh == (384, 256)


def test_vitpose_backend_class_declares_does_own_letterbox():
    from hydra_suite.core.individual.pose.backends.vitpose import ViTPoseBackend

    assert getattr(ViTPoseBackend, "does_own_letterbox", False) is True
