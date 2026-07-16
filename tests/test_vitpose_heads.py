import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra_suite.core.identity.pose.vitpose.config import HEATMAP_SIZE_WH
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
    """SimpleHead must apply ReLU BEFORE the upsample (upstream puts it in
    _transform_inputs, ahead of the interpolation). Applying it after is a
    silent bug: same shapes, different numbers.

    A spatially CONSTANT input cannot test this -- bilinear interpolation of a
    constant field is constant, so relu(interp(x)) == interp(relu(x)) and the
    test would pass on the broken implementation. Use a varying, mixed-sign
    input so the two orderings genuinely diverge.
    """
    h = SimpleHead(embed_dim=8, num_keypoints=2).eval()
    with torch.no_grad():
        h.final_layer.weight.fill_(1.0)
        h.final_layer.bias.zero_()

        x = torch.zeros(1, 8, 16, 12)
        x[:, :, ::2, :] = 1.0
        x[:, :, 1::2, :] = -1.0

        got = h(x)

        w, hh = HEATMAP_SIZE_WH
        relu_first = h.final_layer(
            F.interpolate(F.relu(x), size=(hh, w), mode="bilinear", align_corners=False)
        )
        relu_after = h.final_layer(
            F.relu(F.interpolate(x, size=(hh, w), mode="bilinear", align_corners=False))
        )

    assert torch.allclose(
        got, relu_first
    ), "SimpleHead does not match relu-before-upsample"
    assert not torch.allclose(relu_first, relu_after), (
        "test input fails to discriminate the two ReLU orderings — fix the "
        "input, not the assertion"
    )


def test_simple_head_uses_align_corners_false():
    """SimpleHead's F.interpolate must use align_corners=False (upstream's
    setting). Flipping it is a silent bug: it shifts keypoints by a fraction
    of a heatmap cell, which downstream bbox scaling amplifies into several
    image pixels of error -- same output shape, different numbers.
    """
    h = SimpleHead(embed_dim=8, num_keypoints=2).eval()
    with torch.no_grad():
        h.final_layer.weight.fill_(1.0)
        h.final_layer.bias.zero_()

        torch.manual_seed(0)
        x = torch.randn(1, 8, 16, 12)

        got = h(x)

        w, hh = HEATMAP_SIZE_WH
        false_ref = h.final_layer(
            F.interpolate(F.relu(x), size=(hh, w), mode="bilinear", align_corners=False)
        )
        true_ref = h.final_layer(
            F.interpolate(F.relu(x), size=(hh, w), mode="bilinear", align_corners=True)
        )

    assert torch.allclose(
        got, false_ref
    ), "SimpleHead does not match align_corners=False"
    assert not torch.allclose(false_ref, true_ref), (
        "test input fails to discriminate align_corners=False vs True — fix "
        "the input, not the assertion"
    )


def test_build_head_dispatch():
    assert isinstance(build_head("classic", 768, 17), ClassicHead)
    assert isinstance(build_head("simple", 768, 17), SimpleHead)
