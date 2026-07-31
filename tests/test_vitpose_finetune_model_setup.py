import pytest
import torch

from hydra_suite.core.identity.pose.vitpose.training.model_setup import (
    build_finetune_model,
    load_finetune_init,
)
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
from hydra_suite.core.identity.pose.vitpose.weights import CheckpointKeyError


def test_build_shapes_and_droppath():
    m = build_finetune_model("B", num_keypoints=6, drop_path=0.1)
    out = m(torch.zeros(1, 3, 256, 192))
    assert out.shape == (1, 6, 64, 48)


def test_load_reinits_only_final_layer(tmp_path):
    # a "pretrained" classic K=17 checkpoint, saved as {"state_dict": ...}
    pre = build_vitpose("B", "classic", num_keypoints=17)
    ckpt = tmp_path / "pre.pth"
    torch.save({"state_dict": pre.state_dict()}, ckpt)

    model = build_finetune_model("B", num_keypoints=6, drop_path=0.1)
    fresh_final = model.keypoint_head.final_layer.weight.clone()
    # a backbone param must actually change to the pretrained value
    load_finetune_init(model, ckpt)
    assert torch.equal(
        model.backbone.blocks[0].attn.qkv.weight, pre.backbone.blocks[0].attn.qkv.weight
    )
    # final layer stays the freshly-initialised K=6 conv (shape 6, not 17)
    assert model.keypoint_head.final_layer.weight.shape[0] == 6
    assert torch.equal(model.keypoint_head.final_layer.weight, fresh_final)


def test_load_rejects_variant_mismatch(tmp_path):
    pre = build_vitpose("S", "classic", num_keypoints=17)  # wrong variant
    ckpt = tmp_path / "s.pth"
    torch.save(pre.state_dict(), ckpt)
    model = build_finetune_model("B", num_keypoints=6, drop_path=0.1)
    with pytest.raises(CheckpointKeyError):
        load_finetune_init(model, ckpt)
