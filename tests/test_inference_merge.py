import numpy as np

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
