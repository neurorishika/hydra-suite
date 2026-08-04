"""Backend geometry threading and the artifact recipe tag."""

from __future__ import annotations

import torch

from hydra_suite.core.identity.pose.backends.vitpose import (
    _VITPOSE_RECIPE_TAG,
    ViTPoseBackend,
    _vitpose_artifact_signature,
)
from hydra_suite.core.identity.pose.vitpose.geometry import PoseGeometry
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose

SQUARE = PoseGeometry((256, 256))


def test_recipe_tag_is_bumped_so_old_artifacts_rebuild_once(tmp_path):
    # Geometry changes the exported graph; every v1 artifact must be invalidated.
    # Assert this against an actually generated signature, not just the raw
    # constant -- a signature builder that stopped consuming the tag would
    # otherwise still pass a bare `== "vitpose-v2"` check on the constant.
    assert _VITPOSE_RECIPE_TAG == "vitpose-v2"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    sig = _vitpose_artifact_signature(str(ckpt), "onnx")
    assert sig.startswith(f"{_VITPOSE_RECIPE_TAG}|")


def test_signature_carries_the_recipe_tag(tmp_path):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    sig = _vitpose_artifact_signature(str(ckpt), "coreml")
    assert sig.startswith("vitpose-v2|coreml|")


def _write_square_ckpt(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    path = tmp_path / "square.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "variant": "B",
            "num_keypoints": 9,
            "input_size": SQUARE.to_hw(),
        },
        path,
    )
    return path


def test_backend_adopts_the_checkpoint_geometry(tmp_path):
    backend = ViTPoseBackend(str(_write_square_ckpt(tmp_path)), device="cpu")
    assert backend._geom == SQUARE
    assert backend.preferred_input_size == 256


def test_backend_predicts_end_to_end_at_a_square_geometry(tmp_path):
    import numpy as np

    backend = ViTPoseBackend(str(_write_square_ckpt(tmp_path)), device="cpu")
    results = backend.predict_batch([np.zeros((80, 80, 3), dtype=np.uint8)])
    assert len(results) == 1
    assert results[0].keypoints.shape == (9, 3)
    assert results[0].num_keypoints == 9
