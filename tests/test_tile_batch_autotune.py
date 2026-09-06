from __future__ import annotations

from hydra_suite.core.inference.stages.tile_batch_autotune import (
    TileBatchAutotuneKey,
    clear_tile_batch_autotune_cache,
    select_tile_batch_size,
    tile_batch_candidates,
)


def _key(**changes) -> TileBatchAutotuneKey:
    values = dict(
        artifact="sha256:model-a",
        backend="torch",
        device="cuda:0",
        imgsz=640,
        tile_wh=(640, 640),
        full_frame=False,
        task="obb",
        per_job_bytes=10,
        byte_budget=1000,
        admitted_max=8,
    )
    values.update(changes)
    return TileBatchAutotuneKey(**values)


def test_candidates_are_bounded_and_include_admitted_limit():
    assert tile_batch_candidates(7) == (1, 2, 4, 7)


def test_selects_fastest_admitted_batch_after_warmup():
    calls = []

    def timer(batch):
        calls.append(batch)
        return {1: 1.0, 2: 1.0, 4: 0.5, 8: 4.0}[batch]

    assert select_tile_batch_size(_key(), benchmark=timer) == 4
    assert calls == [1, 1, 2, 4, 8]


def test_cached_selection_avoids_repeat_probe():
    clear_tile_batch_autotune_cache()
    calls = []
    assert (
        select_tile_batch_size(_key(), benchmark=lambda n: calls.append(n) or float(n))
        == 1
    )
    assert (
        select_tile_batch_size(
            _key(), benchmark=lambda n: (_ for _ in ()).throw(AssertionError(n))
        )
        == 1
    )
    assert calls


def test_keys_separate_device_geometry_and_artifact():
    clear_tile_batch_autotune_cache()
    assert select_tile_batch_size(_key(), benchmark=lambda n: 1.0 / n) == 8
    assert (
        select_tile_batch_size(_key(device="cuda:1"), benchmark=lambda n: float(n)) == 1
    )
    assert (
        select_tile_batch_size(_key(tile_wh=(320, 320)), benchmark=lambda n: float(n))
        == 1
    )
    assert (
        select_tile_batch_size(
            _key(artifact="sha256:model-b"), benchmark=lambda n: float(n)
        )
        == 1
    )


def test_failure_or_unusable_clock_falls_back_to_one():
    clear_tile_batch_autotune_cache()
    assert (
        select_tile_batch_size(
            _key(), benchmark=lambda n: (_ for _ in ()).throw(RuntimeError())
        )
        == 1
    )
    assert (
        select_tile_batch_size(_key(artifact="sha256:clock"), benchmark=lambda n: 0.0)
        == 1
    )


def test_probe_output_is_not_an_input_to_selection_or_normal_predictions():
    """The tuner handles timings only; detector result objects stay untouched."""
    output = [object()]
    seen = []

    def benchmark(batch):
        seen.append(output)
        return 1.0 / batch

    assert (
        select_tile_batch_size(_key(artifact="sha256:output"), benchmark=benchmark) == 8
    )
    assert all(item is output for item in seen)
