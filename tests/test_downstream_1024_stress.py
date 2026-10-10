"""Memory stress: one 4512x4512 frame with MAX_DETECTIONS_PER_FRAME OBBs.

Per-animal crops are materialised in ``DOWNSTREAM_CHUNK_SIZE``-row chunks
(``split_rows``), so a frame at the 1024-detection limit must never build all
of its crops at once. Excluded by default (``benchmark``); run with::

    python -m pytest tests/test_downstream_1024_stress.py -m benchmark -q -s

``tracemalloc`` sees numpy (and cv2-through-numpy) allocations only, not the
torch CPU allocator, so the process max-RSS growth is printed alongside it
(torch allocator and first-call library initialisation; no whole-frame float
copy is made -- ``canonical_warp_batch_from_frame`` converts only each crop's
own footprint). Each test also prints the frame's wall-clock time, so a
quadratic per-frame path (the pre-cull pose foreign mask cost ~7 s/frame at
1024 detections) is visible.
"""

from __future__ import annotations

import sys
import time
import tracemalloc

import numpy as np
import pytest
import torch

try:  # POSIX only; the RSS figure is informational
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.downstream_select import split_rows
from hydra_suite.core.inference.limits import (
    DOWNSTREAM_CHUNK_SIZE,
    MAX_DETECTIONS_PER_FRAME,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import (
    ForeignSet,
    extract_canonical_crops_batch,
)

pytestmark = pytest.mark.benchmark

_FRAME_SIDE = 4512
_GRID = 32  # 32 x 32 = 1024 detections
_BOX_W, _BOX_H = 60.0, 30.0
_GEOM = CanonicalGeometry(canvas_wh=(128, 64), margin=1.3, aspect_ratio=2.0)
_GIB = 1024**3


def _runtime(device: str) -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device=device,
        use_nvdec=False,
        tensor_on_cuda=False,
        requested_gpu=device != "cpu",
    )


def _grid_obb() -> OBBResult:
    """1024 axis-aligned, non-overlapping OBBs on a regular grid."""
    n = _GRID * _GRID
    assert n == MAX_DETECTIONS_PER_FRAME
    pitch = _FRAME_SIDE / _GRID  # 141 px >> box size: no overlap
    gy, gx = np.meshgrid(np.arange(_GRID), np.arange(_GRID), indexing="ij")
    cx = ((gx.ravel() + 0.5) * pitch).astype(np.float32)
    cy = ((gy.ravel() + 0.5) * pitch).astype(np.float32)
    hw, hh = _BOX_W / 2, _BOX_H / 2
    corners = np.stack(
        [
            np.stack([cx - hw, cy - hh], -1),
            np.stack([cx + hw, cy - hh], -1),
            np.stack([cx + hw, cy + hh], -1),
            np.stack([cx - hw, cy + hh], -1),
        ],
        axis=1,
    ).astype(np.float32)
    return OBBResult(
        frame_idx=0,
        centroids=np.stack([cx, cy], -1),
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, _BOX_W * _BOX_H, np.float32),
        shapes=np.tile(np.array([[_BOX_W, _BOX_H]], np.float32), (n, 1)),
        confidences=np.ones(n, np.float32),
        corners=corners,
        detection_ids=np.arange(n, dtype=np.int64),
    )


def _max_rss_bytes() -> int:
    if resource is None:
        return 0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # Linux reports KiB


def _run_chunks(frame, obb: OBBResult, device: str) -> list[int]:
    """Crop the superset chunk by chunk, as ``run_superset_chunked`` does."""
    rows: list[int] = []
    for offset, chunk in split_rows(obb, DOWNSTREAM_CHUNK_SIZE):
        foreign = ForeignSet(
            corners=obb.corners,
            self_rows=np.arange(offset, offset + chunk.num_detections),
        )
        batch = extract_canonical_crops_batch(
            [frame],
            [chunk],
            _GEOM,
            _runtime(device),
            suppress_foreign=True,
            foreign_by_frame={obb.frame_idx: foreign},
        )
        assert batch.crops.shape[1:] == (3, _GEOM.canvas_h, _GEOM.canvas_w)
        rows.append(int(batch.crops.shape[0]))
        del batch  # downstream keeps per-animal results, never the crops
    return rows


def test_1024_detection_frame_crops_stay_chunk_bounded_on_cpu():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (_FRAME_SIDE, _FRAME_SIDE, 3), dtype=np.uint8)
    obb = _grid_obb()

    rss_before = _max_rss_bytes()
    t0 = time.perf_counter()
    rows = _run_chunks(frame, obb, "cpu")
    wall = time.perf_counter() - t0  # untraced: tracemalloc slows numpy
    tracemalloc.start()
    try:
        rows = _run_chunks(frame, obb, "cpu")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_growth = _max_rss_bytes() - rss_before

    print(
        f"\n[1024 stress cpu] chunks={rows} wall={wall:.2f} s/frame "
        f"tracemalloc_peak={peak / 2**20:.1f} MiB "
        f"max_rss_growth={rss_growth / 2**20:.1f} MiB"
    )
    assert sum(rows) == MAX_DETECTIONS_PER_FRAME
    assert all(r <= DOWNSTREAM_CHUNK_SIZE for r in rows)
    assert peak < _GIB


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_1024_detection_frame_crops_stay_chunk_bounded_on_mps():
    rng = np.random.default_rng(0)
    host = rng.integers(0, 256, (_FRAME_SIDE, _FRAME_SIDE, 3), dtype=np.uint8)
    frame = torch.from_numpy(host).to("mps")  # HWC uint8 on device
    obb = _grid_obb()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    before = torch.mps.driver_allocated_memory()

    t0 = time.perf_counter()
    rows = _run_chunks(frame, obb, "mps")
    torch.mps.synchronize()
    wall = time.perf_counter() - t0
    after = torch.mps.driver_allocated_memory()

    print(
        f"\n[1024 stress mps] chunks={rows} wall={wall:.2f} s/frame "
        "driver_allocated before="
        f"{before / 2**20:.1f} MiB after={after / 2**20:.1f} MiB "
        f"delta={(after - before) / 2**20:.1f} MiB"
    )
    assert sum(rows) == MAX_DETECTIONS_PER_FRAME
    assert all(r <= DOWNSTREAM_CHUNK_SIZE for r in rows)
