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
)
from hydra_suite.core.inference.stages.obb import _empty_obb_result, _RawOBBTensors


def _cpu_rt() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _cuda_rt() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        default_runtime="cuda",
        tensor_on_cuda=True,
    )


def _make_obb(
    centroids, confidences, sizes=None, corners=None, frame_idx=0
) -> OBBResult:
    n = len(confidences)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.array(centroids, dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.array(sizes or [500.0] * n, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.array(confidences, dtype=np.float32),
        corners=np.array(
            corners or [[[0, 0], [1, 0], [1, 1], [0, 1]]] * n, dtype=np.float32
        ),
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
