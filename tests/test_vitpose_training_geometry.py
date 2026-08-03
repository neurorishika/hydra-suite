"""Training-side geometry: config validation, pos_embed resize, checkpoint stamp."""

from __future__ import annotations

import pytest
import torch

from hydra_suite.core.identity.pose.vitpose.geometry import PoseGeometry
from hydra_suite.core.identity.pose.vitpose.training.config import validate_run_config
from hydra_suite.core.identity.pose.vitpose.training.model_setup import (
    build_finetune_model,
    load_finetune_init,
)
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose

SQUARE = PoseGeometry((256, 256))


def _base_cfg(**over):
    cfg = {
        "init_checkpoint": "x.pth",
        "variant": "B",
        "num_keypoints": 9,
        "dataset_dir": "d",
        "output_dir": "o",
    }
    cfg.update(over)
    return cfg


def test_input_size_defaults_to_none():
    assert validate_run_config(_base_cfg()).input_size is None


def test_input_size_is_accepted_as_height_width():
    assert validate_run_config(_base_cfg(input_size=[256, 256])).input_size == [
        256,
        256,
    ]


@pytest.mark.parametrize("bad", [[256], [256, 250], [0, 256], "256x256"])
def test_malformed_input_size_is_rejected(bad):
    with pytest.raises(ValueError, match="input_size"):
        validate_run_config(_base_cfg(input_size=bad))


def test_build_finetune_model_honours_geometry():
    model = build_finetune_model("B", 9, 0.1, geom=SQUARE)
    assert model.backbone.pos_embed.shape == (1, 257, 768)


def test_build_finetune_model_default_is_unchanged():
    model = build_finetune_model("B", 9, 0.1)
    assert model.backbone.pos_embed.shape == (1, 193, 768)


def test_finetune_init_resizes_pos_embed_across_geometries(tmp_path):
    # THE point of this slice: initialise a 256x256 model from 192x256 weights.
    pretrained = build_vitpose("B", "classic", num_keypoints=17)
    ckpt = tmp_path / "pre.pth"
    torch.save({"state_dict": pretrained.state_dict()}, ckpt)

    model = build_finetune_model("B", 9, 0.1, geom=SQUARE)
    load_finetune_init(model, ckpt, geom=SQUARE)  # must not raise

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 9, 64, 64)


def test_finetune_init_same_geometry_still_works(tmp_path):
    pretrained = build_vitpose("B", "classic", num_keypoints=17)
    ckpt = tmp_path / "pre.pth"
    torch.save({"state_dict": pretrained.state_dict()}, ckpt)
    model = build_finetune_model("B", 9, 0.1)
    load_finetune_init(model, ckpt)
    assert model.backbone.pos_embed.shape == (1, 193, 768)


def test_finetune_init_leaves_final_layer_fresh_across_geometries(tmp_path):
    pretrained = build_vitpose("B", "classic", num_keypoints=17)
    ckpt = tmp_path / "pre.pth"
    torch.save({"state_dict": pretrained.state_dict()}, ckpt)
    model = build_finetune_model("B", 9, 0.1, geom=SQUARE)
    load_finetune_init(model, ckpt, geom=SQUARE)
    # K differs, so the final layer must NOT have been loaded.
    assert model.keypoint_head.final_layer.weight.shape[0] == 9
