"""I4 safety net: an out-of-memory predict halves its chunk and retries.

The dense-mask estimate assumes <= DENSE_MASK_ESTIMATE_CANDIDATES masks per
item; a crowded segment tile can exceed it. Sliced and sequential predict
calls therefore catch device OOM (``torch.OutOfMemoryError`` on CUDA; a
plain ``RuntimeError`` "MPS backend out of memory" on MPS), empty the device
cache, halve the chunk and retry, down to 1 (then re-raise). Without an OOM
nothing changes (one call, the original chunk), so fresh runs stay
byte-identical.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from hydra_suite.core.inference.stages import slicing
from hydra_suite.core.inference.stages.slicing import predict_with_oom_halving


@pytest.fixture(autouse=True)
def _fresh_run_scope():
    slicing.reset_oversize_warnings()
    yield
    slicing.reset_oversize_warnings()


def _fake(limit, exc=None):
    calls: list[int] = []

    def predict(items):
        calls.append(len(items))
        if len(items) > limit:
            raise (
                exc or torch.OutOfMemoryError("CUDA out of memory. Tried to allocate")
            )
        return [f"r{i}" for i in items]

    return predict, calls


def test_no_oom_is_one_call_at_the_original_chunk(caplog):
    predict, calls = _fake(limit=100)
    with caplog.at_level(logging.WARNING):
        out = predict_with_oom_halving(list(range(13)), predict, "tiles")
    assert out == [f"r{i}" for i in range(13)]
    assert calls == [13]
    assert not caplog.records


def test_oom_halves_until_it_fits_and_keeps_order(caplog):
    predict, calls = _fake(limit=3)
    with caplog.at_level(logging.WARNING):
        out = predict_with_oom_halving(list(range(13)), predict, "tiles")
    assert out == [f"r{i}" for i in range(13)]
    assert max(c for c in calls[1:]) <= 7 and calls[0] == 13
    assert all(c <= 3 for c in calls if c not in (13, 7, 4))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "13" in warnings[0] and "7" in warnings[0]


def test_reduced_chunk_sticks_for_the_rest_of_the_run(caplog):
    predict, calls = _fake(limit=3)
    predict_with_oom_halving(list(range(13)), predict, "tiles")
    calls.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        out = predict_with_oom_halving(list(range(6)), predict, "tiles")
    assert out == [f"r{i}" for i in range(6)]
    assert calls and max(calls) <= 3  # no repeated OOM probing at full size
    assert not caplog.records  # one WARNING per run


def test_mps_out_of_memory_runtime_error_also_halves():
    exc = RuntimeError("MPS backend out of memory (MPS allocated: 1 GiB ...)")
    predict, calls = _fake(limit=2, exc=exc)
    out = predict_with_oom_halving(list(range(5)), predict, "crops")
    assert out == [f"r{i}" for i in range(5)]


def test_batch_one_oom_and_unrelated_errors_reraise():
    predict, _ = _fake(limit=0)
    with pytest.raises(torch.OutOfMemoryError):
        predict_with_oom_halving([1, 2, 3, 4], predict, "tiles")
    other, calls = _fake(limit=0, exc=RuntimeError("shape mismatch"))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        predict_with_oom_halving([1, 2, 3, 4], other, "tiles2")
    assert calls == [4]  # not retried


class _Model:
    def __init__(self, limit):
        self.limit, self.sizes = limit, []

    def predict(self, images, **kwargs):
        self.sizes.append(len(images))
        if len(images) > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory.")
        return [("res", id(img)) for img in images]


def _cfg():
    return SimpleNamespace(
        direct=SimpleNamespace(confidence_floor=0.01),
        target_classes=[],
        raw_detection_cap=0,
        max_detections=10,
        mode="direct",
    )


def test_sliced_predict_tiles_survives_oom(monkeypatch):
    import numpy as np

    from hydra_suite.core.inference.stages import obb

    monkeypatch.setattr(obb, "effective_raw_detection_cap", lambda cfg: 1025)
    images = [np.zeros((4, 4, 3), np.uint8) for _ in range(8)]
    model = _Model(limit=2)
    out = slicing._predict_tiles(
        images,
        model,
        _cfg(),
        SimpleNamespace(device="cpu"),
        64,
        letterbox=False,
        chunk_size=8,
    )
    assert [r[1] for r in out] == [id(img) for img in images]
    assert model.sizes[0] == 8 and max(model.sizes[1:]) <= 4


def test_sequential_stage2_chunk_survives_oom():
    import numpy as np

    from hydra_suite.core.inference.stages.regions import (
        Affine,
        Region,
        Stage1Proposals,
    )

    regions = [
        Region(image=np.zeros((4, 4, 3), np.uint8), affine=Affine(), frame_idx=0)
        for _ in range(6)
    ]
    model = _Model(limit=1)
    seq = SimpleNamespace(obb_confidence_threshold=0.01, stage2_image_size=32)
    out = Stage1Proposals._predict_stage2_chunk(
        0,
        regions,
        SimpleNamespace(obb_model=model),
        seq,
        SimpleNamespace(device="cpu"),
        1025,
    )
    assert [r for _, r, _res in out] == regions
    assert [res[1] for _, _r, res in out] == [id(r.image) for r in regions]
