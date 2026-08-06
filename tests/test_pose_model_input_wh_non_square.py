"""Regression guard (Deviation C): `model_input_wh` must use a backend's true
(W, H) input via `preferred_input_wh`, not collapse it to a square via the
scalar `preferred_input_size` (== max(W, H)).

Bug: `model_input_wh` in `core/inference/stages/pose.py` only read the scalar
`preferred_input_size` and returned `(dim, dim)`. For a 192x256 ViTPose model
or a fixed-HxW SLEAP-exported model, that fits the canonical crop into a
SQUARE, which then gets letterboxed AGAIN inside the backend's own
preprocessing -- a wholly redundant resample that also loses resolution.

Fix: backends with a fixed input expose `preferred_input_wh` (true (W, H));
`model_input_wh` uses it directly when present. Backends with no fixed input
(SLEAP service, `preferred_input_size == 0`) must keep getting the identity
fit -- `model_input_wh` returns `geometry.canvas_wh` for them, unchanged.
"""

from __future__ import annotations

import torch

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.identity.pose.backends.sleap import SleapExportedBackend
from hydra_suite.core.identity.pose.backends.vitpose import ViTPoseBackend
from hydra_suite.core.identity.pose.vitpose.geometry import PoseGeometry
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.inference.stages.pose import PoseModel, model_input_wh

_GEOMETRY = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)


def _write_vitpose_ckpt(tmp_path, wh):
    geom = PoseGeometry(wh)
    model = build_vitpose("B", "classic", num_keypoints=9, geom=geom)
    path = tmp_path / "m.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "variant": "B",
            "num_keypoints": 9,
            "input_size": geom.to_hw(),
        },
        path,
    )
    return path


def test_vitposemodel_input_wh_is_true_non_square(tmp_path):
    ckpt = _write_vitpose_ckpt(tmp_path, (192, 256))
    backend = ViTPoseBackend(str(ckpt), device="cpu")
    model = PoseModel(
        backend=backend, n_keypoints=9, keypoint_names=[f"k{i}" for i in range(9)]
    )

    wh = model_input_wh(model, _GEOMETRY)

    assert wh == (192, 256)
    assert wh[0] != wh[1]  # NOT collapsed to a square


def test_sleap_exportedmodel_input_wh_is_true_non_square():
    # Build the backend object without a real ONNX file/session -- only the
    # `preferred_input_wh` property (derived purely from `_input_hw`) is
    # under test here.
    backend = object.__new__(SleapExportedBackend)
    backend._input_hw = (256, 384)  # (H, W): a 384x256 (W,H) model input

    assert backend.preferred_input_wh == (384, 256)

    model = PoseModel(
        backend=backend, n_keypoints=5, keypoint_names=[f"k{i}" for i in range(5)]
    )
    wh = model_input_wh(model, _GEOMETRY)

    assert wh == (384, 256)
    assert wh[0] != wh[1]  # NOT collapsed to a square


def test_backend_with_no_fixed_input_keeps_identity_fit_to_geometry_canvas():
    """SLEAP service (`preferred_input_size == 0`, no `preferred_input_wh`)
    must be unaffected: `model_input_wh` falls back to the canonical
    geometry's own canvas, exactly as before this fix.
    """

    class _NoFixedInputBackend:
        @property
        def preferred_input_size(self) -> int:
            return 0

    model = PoseModel(
        backend=_NoFixedInputBackend(), n_keypoints=3, keypoint_names=["a", "b", "c"]
    )
    wh = model_input_wh(model, _GEOMETRY)
    assert wh == _GEOMETRY.canvas_wh
