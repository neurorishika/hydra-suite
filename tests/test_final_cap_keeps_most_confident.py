"""The final MAX_TARGETS cut keeps the most CONFIDENT detections.

It used to keep the LARGEST (`np.argsort(raw.sizes)[::-1]`), carried over from
the legacy detector as "H5 parity". On a frame with more detections than
targets that let a large low-confidence blob displace a small high-confidence
animal -- the opposite of what the cap is for.

Background subtraction is deliberately excluded: its confidences are NaN, so
confidence ordering is meaningless there and size remains the only signal.

This is an intentional behaviour change to tracking output; goldens recorded
under the size ordering must be regenerated.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.inference.config import OBBConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import (
    filter_for_source,
    filter_from_tensors,
    filter_with_indices,
)
from hydra_suite.core.inference.stages.obb import _RawOBBTensors


def _result(sizes, confs, frame_idx=0):
    n = len(sizes)
    sizes = np.asarray(sizes, dtype=np.float32)
    # Square boxes of the given area, spread far apart so NMS never fires.
    side = np.sqrt(sizes)
    centroids = np.stack(
        [np.arange(n, dtype=np.float32) * 10_000.0, np.zeros(n, dtype=np.float32)],
        axis=1,
    )
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    for i in range(n):
        cx, cy = centroids[i]
        h = side[i] / 2.0
        corners[i] = [
            [cx - h, cy - h],
            [cx + h, cy - h],
            [cx + h, cy + h],
            [cx - h, cy + h],
        ]
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=np.zeros(n, dtype=np.float32),
        sizes=sizes,
        shapes=np.stack([sizes, np.ones(n, dtype=np.float32)], axis=1),
        confidences=np.asarray(confs, dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
        class_ids=np.zeros(n, dtype=np.int64),
    )


def _cfg(max_detections):
    return OBBConfig(
        confidence_threshold=0.0,
        iou_threshold=1.0,  # disable NMS; isolate the cap
        max_detections=max_detections,
        raw_detection_cap=0,
    )


# detection 0 is HUGE but barely confident; 1 and 2 are small but confident.
_SIZES = [10_000.0, 100.0, 100.0]
_CONFS = [0.10, 0.90, 0.80]


def test_numpy_path_final_cap_keeps_most_confident():
    out, _ = filter_with_indices(_result(_SIZES, _CONFS), _cfg(2), None)
    assert out.num_detections == 2
    assert sorted(out.confidences.tolist()) == pytest.approx([0.8, 0.9])
    assert 0.10 not in out.confidences.tolist(), "kept the big low-confidence blob"


def test_tensor_path_final_cap_keeps_most_confident():
    n = len(_SIZES)
    side = np.sqrt(_SIZES)
    raw = _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.tensor(
            [[i * 10_000.0, 0.0, float(side[i]), float(side[i]), 0.0] for i in range(n)]
        ),
        corners=torch.zeros((n, 4, 2)),
        conf=torch.tensor(_CONFS),
        cls=torch.zeros(n),
    )

    class _Rt:
        tensor_on_cuda = True
        device = "cpu"

    out = filter_from_tensors(raw, _cfg(2), None, _Rt())
    assert out.num_detections == 2
    assert sorted(out.confidences.tolist()) == pytest.approx([0.8, 0.9])


def test_both_paths_agree_on_the_cap():
    numpy_out, _ = filter_with_indices(_result(_SIZES, _CONFS), _cfg(2), None)
    n = len(_SIZES)
    side = np.sqrt(_SIZES)
    raw = _RawOBBTensors(
        frame_idx=0,
        xywhr=torch.tensor(
            [[i * 10_000.0, 0.0, float(side[i]), float(side[i]), 0.0] for i in range(n)]
        ),
        corners=torch.zeros((n, 4, 2)),
        conf=torch.tensor(_CONFS),
        cls=torch.zeros(n),
    )

    class _Rt:
        tensor_on_cuda = True
        device = "cpu"

    tensor_out = filter_from_tensors(raw, _cfg(2), None, _Rt())
    assert sorted(numpy_out.confidences.tolist()) == pytest.approx(
        sorted(tensor_out.confidences.tolist())
    )


def test_equal_confidences_break_ties_deterministically():
    """Ties must not depend on sort instability -- repeated runs must agree."""
    res = _result([100.0, 200.0, 300.0, 400.0], [0.5, 0.5, 0.5, 0.5])
    first, _ = filter_with_indices(res, _cfg(2), None)
    for _ in range(5):
        again, _ = filter_with_indices(
            _result([100.0, 200.0, 300.0, 400.0], [0.5] * 4), _cfg(2), None
        )
        assert again.centroids.tolist() == first.centroids.tolist()


def test_bgsub_still_uses_size_because_confidences_are_nan():
    """bg-sub confidences are NaN; size is the only usable ordering there."""
    from hydra_suite.core.inference.stages.filtering import (
        MAX_DOWNSTREAM_CROPS_PER_FRAME,
    )

    n = MAX_DOWNSTREAM_CROPS_PER_FRAME + 3
    sizes = np.arange(n, dtype=np.float32) + 1.0
    res = _result(sizes.tolist(), [float("nan")] * n)

    class _BgCfg:
        detection_source = "bgsub"

    out, _ = filter_for_source(_BgCfg(), res, None)
    assert out.num_detections == MAX_DOWNSTREAM_CROPS_PER_FRAME
    # The largest survive.
    assert out.sizes.min() == pytest.approx(
        float(n - MAX_DOWNSTREAM_CROPS_PER_FRAME + 1)
    )
