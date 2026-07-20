import numpy as np
import pytest
import torch

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.filtering import (
    filter_detections,
    filter_from_tensors,
    filter_raw,
    filter_with_indices,
)
from hydra_suite.core.inference.stages.obb import _empty_obb_result, _RawOBBTensors


def _cpu_rt() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )


def _cuda_rt() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        tensor_on_cuda=True,
        requested_gpu=True,
    )


def _make_obb(
    centroids, confidences, sizes=None, corners=None, frame_idx=0
) -> OBBResult:
    n = len(confidences)
    sizes_arr = np.array(sizes or [500.0] * n, dtype=np.float32)
    if corners is None:
        # Build a square box around each centroid so the corner-based OBB NMS
        # (cv2 convex-hull IoU, matching the legacy detector) sees geometrically
        # consistent boxes. A unit square at the origin for every detection would
        # make all detections fully overlap and be suppressed to one.
        corners_list = []
        for i in range(n):
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            half = max(1.0, float(np.sqrt(max(sizes_arr[i], 1.0))) / 2.0)
            corners_list.append(
                [
                    [cx - half, cy - half],
                    [cx + half, cy - half],
                    [cx + half, cy + half],
                    [cx - half, cy + half],
                ]
            )
        corners_arr = np.array(corners_list, dtype=np.float32)
    else:
        corners_arr = np.array(corners, dtype=np.float32)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.array(centroids, dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=sizes_arr,
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.array(confidences, dtype=np.float32),
        corners=corners_arr,
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _make_raw_tensors(
    centroids, confidences, sizes=None, frame_idx=0
) -> _RawOBBTensors:
    """Build _RawOBBTensors using CPU tensors (no CUDA required for unit tests)."""
    n = len(confidences)
    ws = [s**0.5 for s in (sizes or [500.0] * n)]
    xywhr = torch.tensor(
        [[centroids[i][0], centroids[i][1], ws[i], ws[i], 0.0] for i in range(n)],
        dtype=torch.float32,
    )
    corners = torch.zeros(n, 4, 2, dtype=torch.float32)
    conf = torch.tensor(confidences, dtype=torch.float32)
    return _RawOBBTensors(frame_idx=frame_idx, xywhr=xywhr, corners=corners, conf=conf)


def _cpu_config(**kwargs) -> OBBConfig:
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        **kwargs,
    )


def test_filter_confidence_gate():
    raw = _make_obb([[100, 100], [200, 200]], [0.3, 0.8])
    result = filter_detections(raw, _cpu_config(confidence_threshold=0.5))
    assert result.num_detections == 1
    assert result.confidences[0] == pytest.approx(0.8)


def test_filter_min_size_gate():
    raw = _make_obb([[100, 100], [200, 200]], [0.9, 0.9], sizes=[50.0, 500.0])
    result = filter_detections(raw, _cpu_config(min_object_size=100.0))
    assert result.num_detections == 1
    assert result.sizes[0] == pytest.approx(500.0)


def test_filter_max_size_gate():
    raw = _make_obb([[100, 100], [200, 200]], [0.9, 0.9], sizes=[50.0, 5000.0])
    result = filter_detections(raw, _cpu_config(max_object_size=1000.0))
    assert result.num_detections == 1
    assert result.sizes[0] == pytest.approx(50.0)


def test_filter_size_gate_uses_ellipse_area_not_rect():
    """Regression: MIN/MAX_OBJECT_SIZE thresholds are calibrated for ellipse area
    (pi/4 * w * h), matching the legacy detector (_obb_geometry.py). A box whose
    RECTANGLE area (sizes = w*h) exceeds the max but whose ELLIPSE area is under it
    must be KEPT — previously it was wrongly dropped, removing the largest ~19% of
    detections in crowded scenes."""
    # rect area 1100 > 1000, but ellipse area 1100 * pi/4 ~= 864 < 1000 -> keep
    raw = _make_obb([[100, 100]], [0.9], sizes=[1100.0])
    assert (
        filter_detections(raw, _cpu_config(max_object_size=1000.0)).num_detections == 1
    )
    # ellipse area 2000 * pi/4 ~= 1571 > 1000 -> dropped
    raw_big = _make_obb([[100, 100]], [0.9], sizes=[2000.0])
    assert (
        filter_detections(raw_big, _cpu_config(max_object_size=1000.0)).num_detections
        == 0
    )


def test_filter_with_indices_size_gate_uses_ellipse_area():
    """filter_with_indices is the InferenceRunner hot path (batch + realtime +
    load_frame). It must apply the SAME ellipse-area size gate as filter_detections
    so crowded scenes keep the same detections as the legacy detector."""
    # rect area 1100 > 1000 max, but ellipse area ~864 < 1000 -> keep
    raw = _make_obb([[100, 100]], [0.9], sizes=[1100.0])
    out, idx = filter_with_indices(raw, _cpu_config(max_object_size=1000.0))
    assert out.num_detections == 1 and len(idx) == 1
    raw_big = _make_obb([[100, 100]], [0.9], sizes=[2000.0])  # ellipse ~1571 > 1000
    out_big, _ = filter_with_indices(raw_big, _cpu_config(max_object_size=1000.0))
    assert out_big.num_detections == 0


def test_filter_roi_mask():
    raw = _make_obb([[50, 50], [300, 300]], [0.9, 0.9])
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[0:100, 0:100] = 255
    result = filter_detections(raw, _cpu_config(), roi_mask=mask)
    assert result.num_detections == 1
    assert result.centroids[0, 0] == pytest.approx(50.0)


def test_filter_max_detections():
    raw = _make_obb([[i * 50, 0] for i in range(10)], [0.9] * 10)
    result = filter_detections(raw, _cpu_config(max_detections=3))
    assert result.num_detections == 3


def test_filter_empty_input():
    raw = _empty_obb_result(0)
    result = filter_detections(raw, _cpu_config())
    assert result.num_detections == 0


def test_filter_all_pass():
    raw = _make_obb([[100, 100], [200, 200]], [0.9, 0.8])
    result = filter_detections(raw, _cpu_config(confidence_threshold=0.5))
    assert result.num_detections == 2


def test_filter_preserves_detection_ids_through_subset():
    """Per Correction 14: filter_detections must subset raw.detection_ids,
    NOT regenerate them — survivors keep their original IDs."""
    raw = _make_obb([[100, 100], [200, 200], [300, 300]], [0.3, 0.8, 0.9], frame_idx=5)
    expected_ids = raw.detection_ids.copy()
    result = filter_detections(raw, _cpu_config(confidence_threshold=0.5))
    assert result.num_detections == 2
    # Surviving IDs are raw IDs at indices 1 and 2 (the high-conf ones).
    # NMS may reorder by confidence; compare as sorted sets.
    np.testing.assert_array_equal(
        np.sort(result.detection_ids), np.sort(expected_ids[[1, 2]])
    )


def test_filter_suppresses_overlapping_detection_keeping_highest_conf():
    """Ported from the deleted legacy ``_filter_overlapping_detections`` test:
    two heavily-overlapping OBBs (IoU >= iou_threshold) must collapse to one,
    and the surviving detection is the higher-confidence one. Exercises the
    real corner-polygon IoU NMS in ``_obb_nms`` (legacy parity)."""
    # Box A centred ~(100,100) and box B centred ~(98,100), both 20x20:
    # overlap x-extent 18 -> IoU 360/440 ~= 0.82 >= default 0.7 threshold.
    corners = [
        [[90, 90], [110, 90], [110, 110], [90, 110]],
        [[88, 90], [108, 90], [108, 110], [88, 110]],
    ]
    raw = _make_obb([[100, 100], [98, 100]], [0.80, 0.95], corners=corners)
    result = filter_detections(raw, _cpu_config(confidence_threshold=0.0))
    assert result.num_detections == 1
    assert result.confidences[0] == pytest.approx(0.95)


def test_filter_from_tensors_confidence_gate():
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.3, 0.8])
    result = filter_from_tensors(
        raw, _cpu_config(confidence_threshold=0.5), None, _cuda_rt()
    )
    assert result.num_detections == 1
    assert result.confidences[0] == pytest.approx(0.8)


def test_filter_from_tensors_min_size_gate():
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.9, 0.9], sizes=[50.0, 500.0])
    result = filter_from_tensors(
        raw, _cpu_config(min_object_size=100.0), None, _cuda_rt()
    )
    assert result.num_detections == 1
    assert result.sizes[0] == pytest.approx(500.0)


def test_filter_from_tensors_max_size_gate():
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.9, 0.9], sizes=[50.0, 5000.0])
    result = filter_from_tensors(
        raw, _cpu_config(max_object_size=1000.0), None, _cuda_rt()
    )
    assert result.num_detections == 1
    assert result.sizes[0] == pytest.approx(50.0)


def test_filter_from_tensors_size_gate_uses_ellipse_area():
    """GPU path mirrors the CPU path: size gates compare ellipse area, not rect."""
    raw = _make_raw_tensors([[100, 100]], [0.9], sizes=[1100.0])  # ellipse ~864
    out = filter_from_tensors(
        raw, _cpu_config(max_object_size=1000.0), None, _cuda_rt()
    )
    assert out.num_detections == 1
    raw_big = _make_raw_tensors([[100, 100]], [0.9], sizes=[2000.0])  # ellipse ~1571
    out_big = filter_from_tensors(
        raw_big, _cpu_config(max_object_size=1000.0), None, _cuda_rt()
    )
    assert out_big.num_detections == 0


def test_filter_from_tensors_roi_mask():
    raw = _make_raw_tensors([[50, 50], [300, 300]], [0.9, 0.9])
    mask = torch.zeros(400, 400, dtype=torch.uint8)
    mask[0:100, 0:100] = 1
    result = filter_from_tensors(raw, _cpu_config(), mask, _cuda_rt())
    assert result.num_detections == 1
    assert result.centroids[0, 0] == pytest.approx(50.0)


def test_filter_from_tensors_empty_input():
    raw = _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.zeros((0, 5)),
        corners=torch.zeros((0, 4, 2)),
        conf=torch.zeros(0),
    )
    result = filter_from_tensors(raw, _cpu_config(), None, _cuda_rt())
    assert result.num_detections == 0


def test_filter_from_tensors_assigns_detection_ids():
    """CUDA path constructs detection_ids on the post-filter subset."""
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.9, 0.8], frame_idx=3)
    result = filter_from_tensors(
        raw, _cpu_config(confidence_threshold=0.5), None, _cuda_rt()
    )
    assert result.num_detections == 2
    assert result.detection_ids.shape == (2,)
    assert result.detection_ids.dtype == np.int64
    # IDs follow the standard frame_idx * STRIDE + slot convention
    assert result.detection_ids[0] == 3 * 10000


def test_filter_raw_dispatches_to_cpu_path():
    raw = _make_obb([[100, 100], [200, 200]], [0.3, 0.8])
    result = filter_raw(
        raw, _cpu_config(confidence_threshold=0.5), None, None, _cpu_rt()
    )
    assert isinstance(result, OBBResult)
    assert result.num_detections == 1


def test_filter_raw_dispatches_to_gpu_path():
    raw = _make_raw_tensors([[100, 100], [200, 200]], [0.3, 0.8])
    result = filter_raw(
        raw, _cpu_config(confidence_threshold=0.5), None, None, _cuda_rt()
    )
    assert isinstance(result, OBBResult)
    assert result.num_detections == 1


def test_final_cap_keeps_largest_by_size_not_confidence():
    """H5 parity: when detections exceed max_detections, legacy keeps the
    LARGEST (sort by size), not the most confident (_obb_geometry:587-588)."""
    # 3 well-separated detections; cap to 2. The most confident is the SMALLEST,
    # so a confidence-based cap would keep it while a size-based cap drops it.
    centroids = [(50.0, 50.0), (200.0, 200.0), (350.0, 350.0)]
    confidences = [0.99, 0.80, 0.70]  # smallest box is most confident
    sizes = [100.0, 900.0, 1600.0]
    raw = _make_obb(centroids, confidences, sizes=sizes)
    # iou=1.0 disables NMS so only the size cap decides survivors.
    cfg = _cpu_config(confidence_threshold=0.0, iou_threshold=1.0, max_detections=2)

    out = filter_detections(raw, cfg)
    assert out.num_detections == 2
    kept = sorted(out.sizes.tolist())
    assert kept == [900.0, 1600.0]  # the two LARGEST, not the most confident

    # filter_with_indices must agree (used for cache keying).
    out2, idx = filter_with_indices(raw, cfg)
    assert sorted(out2.sizes.tolist()) == [900.0, 1600.0]
    assert len(idx) == 2


def test_final_cap_size_based_on_cuda_tensor_path():
    """H5 parity on the CUDA tensor path: cap keeps the largest by size."""
    centroids = [(50.0, 50.0), (200.0, 200.0), (350.0, 350.0)]
    confidences = [0.99, 0.80, 0.70]
    sizes = [100.0, 900.0, 1600.0]
    raw = _make_raw_tensors(centroids, confidences, sizes=sizes)
    cfg = _cpu_config(confidence_threshold=0.0, iou_threshold=1.0, max_detections=2)

    out = filter_from_tensors(raw, cfg, None, _cuda_rt())
    assert out.num_detections == 2
    # sizes on the tensor path are w*h = (sqrt(size))^2 ≈ size.
    kept = sorted(round(s) for s in out.sizes.tolist())
    assert kept == [900, 1600]
