import torch
import torch.nn as nn

from hydra_suite.core.identity.pose.vitpose.heads import (
    ClassicHead,
    SimpleHead,
    build_head,
)


def test_classic_head_shape():
    h = ClassicHead(embed_dim=768, num_keypoints=17).eval()
    with torch.no_grad():
        out = h(torch.zeros(2, 768, 16, 12))
    assert out.shape == (2, 17, 64, 48)


def test_classic_deconv_indices_match_checkpoint_keys():
    """Checkpoint keys are keypoint_head.deconv_layers.{0,1,3,4}.*
    -> Sequential(ConvT, BN, ReLU, ConvT, BN, ReLU): params at 0,1,3,4 only.
    ReLU carries no params, which is what creates the 2->3 gap."""
    h = ClassicHead(embed_dim=768, num_keypoints=17)
    layers = h.deconv_layers
    assert isinstance(layers[0], nn.ConvTranspose2d)
    assert isinstance(layers[1], nn.BatchNorm2d)
    assert isinstance(layers[2], nn.ReLU)
    assert isinstance(layers[3], nn.ConvTranspose2d)
    assert isinstance(layers[4], nn.BatchNorm2d)
    assert isinstance(layers[5], nn.ReLU)
    keys = {k.split(".")[1] for k in h.state_dict() if k.startswith("deconv")}
    assert keys == {"0", "1", "3", "4"}


def test_classic_deconv_config():
    h = ClassicHead(embed_dim=768, num_keypoints=17)
    d0 = h.deconv_layers[0]
    assert d0.kernel_size == (4, 4)
    assert d0.stride == (2, 2)
    assert d0.padding == (1, 1)
    assert d0.output_padding == (0, 0)
    assert d0.bias is None  # bias=False
    assert h.final_layer.kernel_size == (1, 1)


def test_simple_head_shape_and_final_conv():
    h = SimpleHead(embed_dim=768, num_keypoints=17).eval()
    with torch.no_grad():
        out = h(torch.zeros(2, 768, 16, 12))
    assert out.shape == (2, 17, 64, 48)
    assert h.final_layer.kernel_size == (3, 3)
    assert h.final_layer.padding == (1, 1)


def test_simple_head_has_no_deconv_params():
    """vitpose-b-simple.pth carries only keypoint_head.final_layer.*"""
    h = SimpleHead(embed_dim=768, num_keypoints=17)
    assert not [k for k in h.state_dict() if k.startswith("deconv")]


def test_simple_head_applies_relu_before_upsample():
    """The ReLU lives in upstream's _transform_inputs, BEFORE F.interpolate.
    With a strictly-negative input, a pre-upsample ReLU zeroes everything, so
    the final conv sees only its bias."""
    h = SimpleHead(embed_dim=8, num_keypoints=2).eval()
    with torch.no_grad():
        h.final_layer.weight.fill_(1.0)
        h.final_layer.bias.zero_()
        out = h(torch.full((1, 8, 16, 12), -5.0))
    assert torch.count_nonzero(out) == 0, "ReLU is not applied before upsampling"


def test_build_head_dispatch():
    assert isinstance(build_head("classic", 768, 17), ClassicHead)
    assert isinstance(build_head("simple", 768, 17), SimpleHead)
