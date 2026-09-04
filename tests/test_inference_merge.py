import math

import numpy as np
import pytest

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.merge import (
    band_membership,
    merge_obb_detections,
)


def _obb(cx, cy, w, h, angle=0.0, conf=0.9, cls=0):
    from hydra_suite.core.inference.stages.obb import (
        _corners_from_xywhr,
        _normalize_obb_geometry,
    )

    cx_a = np.array([cx], np.float32)
    cy_a = np.array([cy], np.float32)
    w_a = np.array([w], np.float32)
    h_a = np.array([h], np.float32)
    ang, sizes, aspect = _normalize_obb_geometry(
        w_a, h_a, np.array([angle], np.float32)
    )
    corners = _corners_from_xywhr(cx_a, cy_a, w_a, h_a, ang)
    return OBBResult(
        frame_idx=0,
        centroids=np.stack([cx_a, cy_a], axis=1),
        angles=ang,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1),
        confidences=np.array([conf], np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, 1),
        class_ids=np.array([cls], np.int64),
    )


def _concat(*results):
    from hydra_suite.core.inference.stages.obb import merge_obb_results

    return merge_obb_results(0, list(results))


def test_nms_suppresses_duplicate_keeps_one():
    dup = _concat(_obb(100, 100, 40, 40, conf=0.9), _obb(102, 101, 40, 40, conf=0.5))
    out = merge_obb_detections(
        dup, policy="nms", metric="iou", threshold=0.5, backend="cv2"
    )
    assert out.num_detections == 1
    assert out.confidences[0] == 0.9  # higher-conf survivor


def test_merge_preserves_native_polygon_for_nms_survivor():
    """A sliced segment prediction must not become a four-corner OBB at NMS."""
    survivor = _obb(100, 100, 40, 40, conf=0.9)
    duplicate = _obb(102, 101, 40, 40, conf=0.5)
    survivor.polygons = [
        np.array([[80, 80], [120, 80], [118, 95], [120, 120], [80, 120]], np.float32)
    ]
    duplicate.polygons = [
        np.array([[82, 81], [122, 81], [122, 121], [82, 121]], np.float32)
    ]

    out = merge_obb_detections(
        _concat(survivor, duplicate),
        policy="nms",
        metric="iou",
        threshold=0.5,
        backend="cv2",
    )

    assert out.polygons is not None
    assert len(out.polygons) == 1
    assert len(out.polygons[0]) == 5
    np.testing.assert_array_equal(out.polygons[0], survivor.polygons[0])


def test_nmm_unions_truncated_pair_into_one_larger_box():
    # Realistic straddling case: one tile catches the whole animal, the
    # neighbouring tile catches only a clipped sliver of it.
    #   big   = x[70,130], area 2400
    #   small = x[62,82],  area 800
    #   intersection = 12 x 40 = 480
    #   IoS = 480 / min(2400, 800) = 0.600  -> >= 0.5, MERGES
    #   IoU = 480 / (2400 + 800 - 480) = 0.176 -> < 0.5, would NOT merge
    big = _obb(100, 100, 60, 40, conf=0.8)
    small = _obb(72, 100, 20, 40, conf=0.7)
    out = merge_obb_detections(
        _concat(big, small),
        policy="greedy_nmm",
        metric="ios",
        threshold=0.5,
        backend="cv2",
    )
    assert out.num_detections == 1
    # union box strictly larger than the largest member -> proves union
    # semantics, not mere suppression.
    assert out.sizes[0] > big.sizes[0]
    assert out.confidences[0] == 0.8  # max conf


def test_nmm_preserves_highest_confidence_native_polygon():
    """A merged tile duplicate retains a mask contour, never a box contour."""
    big = _obb(100, 100, 60, 40, conf=0.8)
    small = _obb(72, 100, 20, 40, conf=0.7)
    polygon = np.array(
        [[70, 80], [130, 80], [127, 96], [130, 120], [70, 120]], np.float32
    )
    big.polygons = [polygon]
    small.polygons = [np.array([[62, 80], [82, 80], [82, 120], [62, 120]], np.float32)]

    out = merge_obb_detections(
        _concat(big, small),
        policy="greedy_nmm",
        metric="ios",
        threshold=0.5,
        backend="cv2",
    )

    assert out.polygons is not None
    assert len(out.polygons) == 1
    np.testing.assert_array_equal(out.polygons[0], polygon)


def test_native_polygons_route_gpu_merge_through_contour_safe_oracle():
    """The tensor-only backend must not drop segment contours."""
    first = _obb(100, 100, 40, 40, conf=0.9)
    second = _obb(102, 101, 40, 40, conf=0.5)
    polygon = np.array(
        [[80, 80], [120, 80], [118, 96], [120, 120], [80, 120]], np.float32
    )
    first.polygons = [polygon]
    second.polygons = [
        np.array([[82, 81], [122, 81], [122, 121], [82, 121]], np.float32)
    ]

    out = merge_obb_detections(
        _concat(first, second),
        policy="nms",
        metric="iou",
        threshold=0.5,
        backend="gpu",
    )

    assert out.polygons is not None
    np.testing.assert_array_equal(out.polygons[0], polygon)


def test_cv2_union_corners_match_expected_rotated_rectangle():
    """cv2-backend analogue of ``test_gpu_backend_union_corners_match_expected...``.

    Exactly the ``test_nmm_unions_truncated_pair...`` construction, asserted at
    the CORNERS level rather than only on ``sizes``. That distinction is the
    whole point: pairing a ``(w, h)`` pair with the wrong angle convention
    rotates a non-square box 90 degrees while leaving its area -- and therefore
    ``sizes`` -- exactly invariant, so a sizes-only assertion is blind to it.

    For this union point set ``cv2.minAreaRect`` reports ``(w, h) = (40, 68)``
    at -90 degrees, i.e. the ``w < h`` regime. Applying
    ``_normalize_obb_geometry`` twice to that same raw pair (once inside
    ``_union_obb``, once again in ``_assemble``) adds 90 degrees twice, landing
    the reported angle on the MINOR axis and emitting a 68x40 box rotated 90
    degrees from the true footprint.

    Union of big (x[70,130], y[80,120]) and small (x[62,82], y[80,120]) is
    exactly x[62,130], y[80,120]: center (96, 100), major 68 horizontal,
    minor 40, angle 0.
    """
    from hydra_suite.core.inference.stages.obb import _corners_from_xywhr

    big = _obb(100, 100, 60, 40, conf=0.8)
    small = _obb(72, 100, 20, 40, conf=0.7)
    out = merge_obb_detections(
        _concat(big, small),
        policy="greedy_nmm",
        metric="ios",
        threshold=0.5,
        backend="cv2",
    )
    assert out.num_detections == 1
    exp_corners = _corners_from_xywhr(
        np.array([96.0], np.float32),
        np.array([100.0], np.float32),
        np.array([68.0], np.float32),
        np.array([40.0], np.float32),
        np.array([0.0], np.float32),
    )
    np.testing.assert_allclose(out.corners[0], exp_corners[0], atol=0.5)
    np.testing.assert_allclose(float(out.angles[0]), 0.0, atol=1e-3)


def test_iou_metric_does_not_merge_what_ios_merges():
    """Same straddling pair: IoS=0.60 merges, IoU=0.176 does not.

    This is why ios is the default metric for cross-tile merging.
    """
    big = _obb(100, 100, 60, 40, conf=0.8)
    small = _obb(72, 100, 20, 40, conf=0.7)
    out = merge_obb_detections(
        _concat(big, small),
        policy="greedy_nmm",
        metric="iou",
        threshold=0.5,
        backend="cv2",
    )
    assert out.num_detections == 2


def test_ios_vs_iou_threshold_behavior():
    a = _obb(100, 100, 60, 20, conf=0.9)  # small box fully inside big one
    b = _obb(100, 100, 60, 60, conf=0.6)
    iou_out = merge_obb_detections(
        _concat(a, b), policy="nms", metric="iou", threshold=0.6, backend="cv2"
    )
    ios_out = merge_obb_detections(
        _concat(a, b), policy="nms", metric="ios", threshold=0.6, backend="cv2"
    )
    # IoU of nested boxes is low (< 0.6) -> both kept; IoS is 1.0 -> one kept.
    assert iou_out.num_detections == 2
    assert ios_out.num_detections == 1


def test_overlap_zero_returns_input_unchanged():
    r = _concat(_obb(10, 10, 5, 5), _obb(500, 500, 5, 5))
    # merge with threshold 1.0 (no pair can meet it) is a no-op count-wise.
    out = merge_obb_detections(
        r, policy="greedy_nmm", metric="ios", threshold=1.01, backend="cv2"
    )
    assert out.num_detections == 2


def test_nms_kept_survivor_preserves_geometry_of_non_square_box():
    """Regression for the 90-degree-rotation bug: a kept-single survivor's
    corners/angle must pass through byte-for-byte, not be reconstructed via
    cv2.minAreaRect (which can swap w/h and rotate the box 90 degrees for a
    non-square OBB).
    """
    survivor = _obb(100, 100, 60, 20, angle=0.0, conf=0.9)
    far_away = _obb(500, 500, 5, 5, conf=0.5)  # no overlap -> both are "kept"
    out = merge_obb_detections(
        _concat(survivor, far_away),
        policy="nms",
        metric="iou",
        threshold=0.5,
        backend="cv2",
    )
    assert out.num_detections == 2
    idx = int(np.argmin(np.abs(out.centroids[:, 0] - 100)))
    np.testing.assert_allclose(out.corners[idx], survivor.corners[0], atol=1e-3)
    np.testing.assert_allclose(out.angles[idx], survivor.angles[0], atol=1e-5)


def test_passthrough_detection_preserves_geometry_of_non_square_box():
    """Regression for the 90-degree-rotation bug on the passthrough path:
    a detection outside the overlap band must pass through untouched.

    Needs >= 2 band members so the merge stage actually runs the quadratic
    loop and reaches ``_assemble`` (with a single band member it early-returns
    the input unchanged, which would make this test vacuous).
    """
    band_member_a = _obb(10, 10, 5, 5, conf=0.9)
    band_member_b = _obb(500, 500, 5, 5, conf=0.5)  # no overlap w/ band_member_a
    passthrough = _obb(100, 100, 60, 20, angle=0.0, conf=0.7)
    r = _concat(band_member_a, band_member_b, passthrough)
    overlap_bands = np.array([True, True, False])
    out = merge_obb_detections(
        r,
        policy="greedy_nmm",
        metric="ios",
        threshold=0.5,
        backend="cv2",
        overlap_bands=overlap_bands,
    )
    assert out.num_detections == 3
    idx = int(np.argmin(np.abs(out.centroids[:, 0] - 100)))
    np.testing.assert_allclose(out.corners[idx], passthrough.corners[0], atol=1e-3)
    np.testing.assert_allclose(out.angles[idx], passthrough.angles[0], atol=1e-5)


def test_band_membership_flags_only_overlap_region():
    tiles = [(0, 0, 100, 100), (80, 0, 180, 100)]  # overlap band x in [80,100]
    corners = np.array(
        [
            [[10, 10], [20, 10], [20, 20], [10, 20]],  # exclusive to tile 0
            [[85, 40], [95, 40], [95, 50], [85, 50]],  # in the band
        ],
        dtype=np.float32,
    )
    band = band_membership(corners, tiles)
    assert band.tolist() == [False, True]


@pytest.mark.parametrize("policy", ["nms", "greedy_nmm"])
def test_gpu_backend_matches_cv2_within_tolerance(policy):
    rng = np.random.default_rng(1)
    parts = []
    for _ in range(8):
        parts.append(
            _obb(
                *rng.uniform([60, 60, 30, 30], [200, 200, 60, 60]),
                angle=float(rng.uniform(0, np.pi)),
                conf=float(rng.uniform(0.3, 0.99)),
            )
        )
    r = _concat(*parts)
    cv2_out = merge_obb_detections(
        r, policy=policy, metric="ios", threshold=0.5, backend="cv2"
    )
    gpu_out = merge_obb_detections(
        r, policy=policy, metric="ios", threshold=0.5, backend="gpu"
    )
    # same count (grouping decisions agree) within tolerance.
    assert gpu_out.num_detections == cv2_out.num_detections
    # centroids of survivors match within a few px after sorting.
    cc = np.sort(cv2_out.centroids.sum(axis=1))
    gc = np.sort(gpu_out.centroids.sum(axis=1))
    assert np.allclose(cc, gc, atol=3.0)


def test_gpu_backend_honours_overlap_bands_like_cv2():
    """Finding I1: the gpu backend must apply ``overlap_bands`` exactly like the
    cv2 oracle, not merge every detection in the frame.

    Two genuinely DISTINCT touching animals inside ONE tile's exclusive region
    can exceed ``ios >= 0.5`` (here the small one is fully inside the big one's
    hull, ios = 1.0) -- but they can have no cross-tile duplicate, so the cv2
    oracle never considers them. Before this fix the gpu backend ignored the
    band mask entirely and unioned them into one detection: a correctness
    divergence from the declared oracle in exactly the crowded-scene case this
    feature exists for.

    The frame also contains a REAL cross-tile duplicate pair straddling the
    x=256 tile boundary, so the test proves the band mask is being applied
    (exclusive pair survives) and not merely that merging was disabled
    (straddling pair still collapses).
    """
    tiles = [(0, 0, 256, 256), (240, 0, 496, 256)]
    exclusive_big = _obb(100, 100, 60, 60, conf=0.9)
    exclusive_small = _obb(105, 100, 20, 20, conf=0.6)  # ios == 1.0 vs big
    straddle_a = _obb(250, 200, 40, 40, conf=0.8)
    straddle_b = _obb(254, 200, 40, 40, conf=0.5)
    r = _concat(exclusive_big, exclusive_small, straddle_a, straddle_b)

    bands = band_membership(r.corners, tiles)
    # Non-trivial mask: the exclusive pair is out of band, the straddlers in.
    assert bands.tolist() == [False, False, True, True]

    kw = dict(policy="greedy_nmm", metric="ios", threshold=0.5, overlap_bands=bands)
    cv2_out = merge_obb_detections(r, backend="cv2", **kw)
    gpu_out = merge_obb_detections(r, backend="gpu", **kw)

    # 2 untouched exclusive-region detections + 1 unioned straddling pair.
    assert cv2_out.num_detections == 3
    assert gpu_out.num_detections == cv2_out.num_detections
    cc = np.sort(cv2_out.centroids.sum(axis=1))
    gc = np.sort(gpu_out.centroids.sum(axis=1))
    assert np.allclose(cc, gc, atol=3.0)


def test_gpu_backend_all_bands_false_is_a_no_op():
    """No band member can have a cross-tile duplicate -> nothing may merge."""
    r = _concat(_obb(100, 100, 60, 60, conf=0.9), _obb(105, 100, 20, 20, conf=0.6))
    bands = np.zeros(r.num_detections, dtype=bool)
    out = merge_obb_detections(
        r,
        policy="greedy_nmm",
        metric="ios",
        threshold=0.5,
        backend="gpu",
        overlap_bands=bands,
    )
    assert out.num_detections == 2
    np.testing.assert_allclose(out.corners, r.corners, atol=1e-5)


def test_gpu_backend_single_member_passthrough():
    r = _obb(100, 100, 40, 40, conf=0.9)
    out = merge_obb_detections(
        r, policy="greedy_nmm", metric="ios", threshold=0.5, backend="gpu"
    )
    assert out.num_detections == 1


def test_gpu_backend_nms_keeps_survivor_geometry_verbatim():
    """Regression for the 90-degree-rotation bug on the gpu NMS path: the
    kept survivor's corners/angle must pass through untouched by array
    indexing, not be reconstructed from any geometry kernel.
    """
    survivor = _obb(100, 100, 60, 20, angle=0.3, conf=0.9)
    far_away = _obb(500, 500, 5, 5, conf=0.5)  # no overlap -> both kept
    out = merge_obb_detections(
        _concat(survivor, far_away),
        policy="nms",
        metric="iou",
        threshold=0.5,
        backend="gpu",
    )
    assert out.num_detections == 2
    idx = int(np.argmin(np.abs(out.centroids[:, 0] - 100)))
    np.testing.assert_allclose(out.corners[idx], survivor.corners[0], atol=1e-3)
    np.testing.assert_allclose(out.angles[idx], survivor.angles[0], atol=1e-5)


def test_gpu_backend_union_corners_match_expected_rotated_rectangle():
    """Corners-level regression for the exact bug class that hit Task 3's
    cv2 union path: pairing a (w, h, angle) triple with the WRONG angle
    convention silently rotates a non-square box 90 degrees while leaving its
    *area* invariant -- an area/sizes-only assertion cannot catch this.

    Construction: two axis-aligned (in a rotated local frame) rectangles that
    share the exact same extent along one axis, so their union is *exactly* a
    single rectangle of known (w, h) -- no ambiguity from convex-hull
    corner-rounding. The whole configuration is then rigidly rotated by
    ``theta`` (chosen to land exactly on one of the kernel's 64 candidate
    angles, eliminating quantization error) so the box is non-axis-aligned.
    If the union kernel swapped the (w, h, angle) convention, the recovered
    corners would describe a 40x68 box rotated ~90 degrees from the expected
    68x40 box -- a completely different footprint that this assertion catches
    directly, unlike an area-only check.
    """
    from hydra_suite.core.inference.stages.obb import _corners_from_xywhr

    k = 5
    theta = k * math.pi / 64  # exact grid point of _union_via_kernel's search
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rot(x, y):
        return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    # Un-rotated construction (theta=0): big spans u in [70,130], v in
    # [80,120]; small spans u in [62,82], v in [80,120] -- identical v-range,
    # so their union is exactly the rectangle u in [62,130], v in [80,120]
    # (w=68, h=40, center u=96, v=100).
    c1 = rot(100.0, 100.0)
    c2 = rot(72.0, 100.0)
    big = _obb(c1[0], c1[1], 60, 40, angle=theta, conf=0.8)
    small = _obb(c2[0], c2[1], 20, 40, angle=theta, conf=0.7)
    r = _concat(big, small)
    out = merge_obb_detections(
        r, policy="greedy_nmm", metric="ios", threshold=0.5, backend="gpu"
    )
    assert out.num_detections == 1

    exp_cx, exp_cy = rot(96.0, 100.0)
    exp_corners = _corners_from_xywhr(
        np.array([exp_cx], np.float32),
        np.array([exp_cy], np.float32),
        np.array([68.0], np.float32),
        np.array([40.0], np.float32),
        np.array([theta], np.float32),
    )
    np.testing.assert_allclose(out.corners[0], exp_corners[0], atol=0.5)
    np.testing.assert_allclose(float(out.angles[0]), theta, atol=1e-3)
