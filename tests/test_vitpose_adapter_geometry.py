"""Geometry recovery when loading a fine-tuned or external checkpoint."""

from __future__ import annotations

import pytest
import torch

from hydra_suite.core.individual.pose.vitpose.adapter import load_finetuned_checkpoint
from hydra_suite.core.individual.pose.vitpose.geometry import (
    DEFAULT_GEOMETRY,
    PoseGeometry,
)
from hydra_suite.core.individual.pose.vitpose.vitpose import build_vitpose

SQUARE = PoseGeometry((256, 256))


def _save(tmp_path, payload, name="ckpt.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


def test_stored_input_size_is_authoritative(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    path = _save(
        tmp_path,
        {
            "model_state": model.state_dict(),
            "variant": "B",
            "num_keypoints": 9,
            "input_size": SQUARE.to_hw(),
        },
    )
    loaded, meta = load_finetuned_checkpoint(path)
    assert meta.geometry == SQUARE
    assert loaded.backbone.pos_embed.shape == (1, 257, 768)


def test_square_geometry_is_inferred_when_not_stored(tmp_path):
    # This is the external-checkpoint case: no input_size key at all.
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    path = _save(tmp_path, {"state_dict": model.state_dict()})
    loaded, meta = load_finetuned_checkpoint(path)
    assert meta.geometry == SQUARE
    assert meta.num_keypoints == 9
    with torch.no_grad():
        assert loaded(torch.zeros(1, 3, 256, 256)).shape == (1, 9, 64, 64)


def test_default_geometry_is_inferred_for_an_upstream_shaped_checkpoint(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=17)
    path = _save(tmp_path, {"state_dict": model.state_dict()})
    _, meta = load_finetuned_checkpoint(path)
    assert meta.geometry == DEFAULT_GEOMETRY


def test_stored_geometry_contradicting_the_weights_raises(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=9)  # 193 tokens
    path = _save(
        tmp_path,
        {
            "model_state": model.state_dict(),
            "variant": "B",
            "num_keypoints": 9,
            "input_size": SQUARE.to_hw(),  # claims 257 tokens
        },
    )
    with pytest.raises(ValueError, match="does not match"):
        load_finetuned_checkpoint(path)


def test_meta_still_carries_variant_head_and_keypoints(tmp_path):
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    path = _save(tmp_path, {"state_dict": model.state_dict()})
    _, meta = load_finetuned_checkpoint(path)
    assert (meta.variant, meta.head, meta.num_keypoints) == ("B", "classic", 9)


def test_malformed_input_size_raises_value_error_not_type_error(tmp_path):
    # A third-party checkpoint can carry a scalar (non-iterable) input_size.
    # That must surface as a ValueError naming the offending field, not an
    # opaque TypeError from PoseGeometry.from_hw trying to iterate an int.
    model = build_vitpose("B", "classic", num_keypoints=9)
    path = _save(
        tmp_path,
        {
            "model_state": model.state_dict(),
            "variant": "B",
            "num_keypoints": 9,
            "input_size": 5,
        },
    )
    with pytest.raises(ValueError, match="5"):
        load_finetuned_checkpoint(path)
