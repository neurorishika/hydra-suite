import numpy as np
import pytest

from hydra_suite.core.identity.pose.vitpose.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PADDING_FACTOR,
    PIXEL_STD,
)
from hydra_suite.core.identity.pose.vitpose.transforms import (
    box2cs,
    get_warp_matrix,
    normalize,
    top_down_affine,
    transform_preds,
)


def test_box2cs_center():
    c, s = box2cs(np.array([10.0, 20.0, 40.0, 80.0]))
    assert np.allclose(c, [30.0, 60.0])


def test_box2cs_scale_uses_pixel_std_and_padding():
    _, s = box2cs(np.array([0.0, 0.0, 192.0, 256.0]))
    assert np.allclose(s, np.array([192.0, 256.0]) / PIXEL_STD * PADDING_FACTOR)


def test_box2cs_fixes_aspect_ratio():
    """A wide box must be grown in height to reach the 192:256 aspect."""
    _, s = box2cs(np.array([0.0, 0.0, 400.0, 100.0]))
    assert s[0] / s[1] == pytest.approx(192 / 256, rel=1e-6)


def test_warp_matrix_shape():
    m = get_warp_matrix(
        0.0,
        np.array([100.0, 100.0]),
        np.array([191.0, 255.0]),
        np.array([200.0, 200.0]),
    )
    assert m.shape == (2, 3)


def test_warp_uses_size_minus_one_for_udp():
    """UDP defines unit length as pixel SPACING (size-1), not pixel count.
    A centred square box must map its centre to the destination centre and its
    edges to exactly 0 and size-1."""
    img = np.zeros((400, 400, 3), np.uint8)
    center = np.array([200.0, 200.0])
    scale = np.array([400.0, 400.0]) / PIXEL_STD
    out = top_down_affine(img, center, scale)
    assert out.shape == (256, 192, 3)


def test_affine_maps_marker_to_expected_pixel():
    """Put a white marker at the box centre; after warping it must land at the
    destination centre (191/2, 255/2), i.e. the UDP (size-1) convention."""
    img = np.zeros((400, 400, 3), np.uint8)
    img[198:203, 198:203] = 255
    center = np.array([200.0, 200.0])
    scale = np.array([400.0, 400.0]) / PIXEL_STD
    out = top_down_affine(img, center, scale)
    ys, xs = np.nonzero(out[:, :, 0])
    assert abs(xs.mean() - 191 / 2) < 1.5
    assert abs(ys.mean() - 255 / 2) < 1.5


def test_normalize_is_rgb_chw_and_imagenet():
    img = np.zeros((256, 192, 3), np.uint8)
    img[:, :, 2] = 255  # BGR red
    out = normalize(img)
    assert out.shape == (3, 256, 192)
    assert out.dtype == np.float32
    # channel 0 is R after BGR->RGB, so it should be the (1 - mean)/std value
    assert np.allclose(out[0], (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], atol=1e-5)
    assert np.allclose(out[2], (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], atol=1e-5)


def test_warp_matrix_udp_corner_correspondence():
    """Directly verify the UDP (size-1) semantics of get_warp_matrix.

    Unlike the marker-at-centre tests above (which cancel the -1 correction
    for a box exactly matching the image, and only distinguish UDP from
    non-UDP by ~0.5-1px via quantized pixel centroids), this test applies
    the raw matrix to the analytically known far corner of the ROI and
    checks it lands at exactly (w-1, h-1), not (w, h). A missing "-1" would
    fail this by a full pixel with no quantization noise to hide behind.
    """
    from hydra_suite.core.identity.pose.vitpose.config import IMAGE_SIZE_WH

    center = np.array([200.0, 200.0])
    scale = np.array([400.0, 400.0]) / PIXEL_STD
    w, h = IMAGE_SIZE_WH
    size_target = scale * PIXEL_STD
    size_dst = np.array([w, h], dtype=np.float32) - 1.0

    trans = get_warp_matrix(0.0, center * 2.0, size_dst, size_target)

    corner_lo = center - size_target / 2.0
    corner_hi = center + size_target / 2.0

    def apply(pt):
        return trans @ np.array([pt[0], pt[1], 1.0])

    assert np.allclose(apply(corner_lo), [0.0, 0.0], atol=1e-3)
    assert np.allclose(apply(corner_hi), [w - 1, h - 1], atol=1e-3)


def test_transform_preds_roundtrip_center():
    """A prediction at the heatmap centre must map back to the box centre."""
    center = np.array([200.0, 200.0])
    scale = np.array([400.0, 400.0]) / PIXEL_STD
    coords = np.array([[47 / 2, 63 / 2]])
    out = transform_preds(coords, center, scale, (48, 64))
    assert np.allclose(out[0], center, atol=1.0)
