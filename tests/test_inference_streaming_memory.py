"""Isolated memory regressions for inference tile streaming.

These tests deliberately use a no-op model.  They measure orchestration memory,
not framework/model allocations, and never request enough memory to pressure the
host.  Running in a spawned process prevents allocator history from hiding a
regression that materializes the complete tile grid.
"""

from __future__ import annotations

import gc
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

psutil = pytest.importorskip("psutil")


def _measure_grid_peak_rss(overlap: float, sender) -> None:
    from hydra_suite.core.inference.config import SliceConfig
    from hydra_suite.core.inference.stages.regions import Grid

    process = psutil.Process()
    frame = np.zeros((2048, 2048, 3), dtype=np.uint8)
    slice_config = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_width=128,
        slice_height=128,
        overlap_width_ratio=overlap,
        overlap_height_ratio=overlap,
        tile_batch_size=4,
    )
    config = SimpleNamespace(
        direct=SimpleNamespace(
            slice=slice_config,
            confidence_floor=0.01,
            model_task="obb",
        ),
        target_classes=[],
    )
    peak_rss = process.memory_info().rss

    class _NoOpModel:
        imgsz = 128

        def predict(self, images, **kwargs):
            nonlocal peak_rss
            peak_rss = max(peak_rss, process.memory_info().rss)
            return [object() for _ in images]

    gc.collect()
    baseline_rss = process.memory_info().rss
    tile_count = 0
    for chunk in Grid().iter_region_results(
        [frame],
        SimpleNamespace(direct_model=_NoOpModel()),
        config,
        SimpleNamespace(tensor_on_cuda=False, device="cpu"),
    ):
        tile_count += len(chunk)
        peak_rss = max(peak_rss, process.memory_info().rss)
        del chunk
    sender.send((tile_count, max(0, peak_rss - baseline_rss)))
    sender.close()


def _run_isolated_measurement(overlap: float) -> tuple[int, int]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_measure_grid_peak_rss, args=(overlap, sender))
    process.start()
    sender.close()
    assert receiver.poll(30), "tile memory probe did not complete"
    result = receiver.recv()
    process.join(timeout=10)
    assert process.exitcode == 0
    return result


def test_tile_grid_count_does_not_scale_peak_pixel_memory():
    sparse_count, sparse_growth = _run_isolated_measurement(0.0)
    dense_count, dense_growth = _run_isolated_measurement(0.75)

    assert dense_count > sparse_count * 10
    # The old eager Grid.plan path retained roughly 183 MiB for the dense grid.
    # Allow broad platform/allocator noise while proving peak growth is tied to
    # the four-tile admitted chunk rather than total grid cardinality.
    assert sparse_growth < 64 * 1024 * 1024
    assert dense_growth < 64 * 1024 * 1024
    assert dense_growth - sparse_growth < 32 * 1024 * 1024
