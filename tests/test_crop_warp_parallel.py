"""The parallel pose crop-warp must be byte-identical to the serial loop.

``_warp_crops_for_obb`` runs independent cv2.warpAffine calls across a shared
thread pool when the detection count is large enough. cv2 releases the GIL and
each warp writes its own buffer, and ``pool.map`` preserves order, so the output
must match the serial path exactly. This locks that invariant in.
"""

import os

import numpy as np

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages import crops as crops_mod


def _make_obb(n: int, frame_idx: int = 0) -> OBBResult:
    corners = []
    for i in range(n):
        cx, cy = 40 + i * 30, 60 + (i % 3) * 15
        w, h = 22, 13
        corners.append(
            np.array(
                [
                    [cx - w, cy - h],
                    [cx + w, cy - h],
                    [cx + w, cy + h],
                    [cx - w, cy + h],
                ],
                dtype=np.float32,
            )
        )
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32),
        sizes=np.ones(n, np.float32),
        shapes=np.zeros((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=np.stack(corners, 0),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _warp(arr, obb, threads):
    os.environ["HYDRA_CROP_WARP_THREADS"] = str(threads)
    try:
        return crops_mod._warp_crops_for_obb(
            arr, obb, aspect_ratio=2.0, padding_fraction=0.1
        )
    finally:
        os.environ.pop("HYDRA_CROP_WARP_THREADS", None)


def test_parallel_warp_is_byte_identical_to_serial():
    arr = np.random.default_rng(7).integers(0, 256, (260, 640, 3), dtype=np.uint8)
    obb = _make_obb(12)  # >= _WARP_MIN_PARALLEL, so the pool engages
    serial = _warp(arr, obb, threads=1)
    parallel = _warp(arr, obb, threads=4)
    assert len(serial) == len(parallel) == 12
    for i, (a, b) in enumerate(zip(serial, parallel)):
        assert a.shape == b.shape, f"crop {i} shape mismatch"
        assert np.array_equal(a, b), f"crop {i} differs serial vs parallel"


def test_small_batch_stays_serial_and_matches():
    # n below the threshold must not use the pool, and still be correct.
    arr = np.random.default_rng(3).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    obb = _make_obb(2)
    serial = _warp(arr, obb, threads=1)
    parallel = _warp(arr, obb, threads=8)  # still serial: n=2 < _WARP_MIN_PARALLEL
    for a, b in zip(serial, parallel):
        assert np.array_equal(a, b)


def test_threads_env_of_one_disables_pool():
    os.environ["HYDRA_CROP_WARP_THREADS"] = "1"
    try:
        assert crops_mod._crop_warp_threads() == 1
        assert crops_mod._get_warp_pool(1) is None
    finally:
        os.environ.pop("HYDRA_CROP_WARP_THREADS", None)
