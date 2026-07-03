"""Regression tests for the NVDEC trial-decode / cv2-fallback fix.

Background: ``PyNvVideoCodec`` only validates a decoded stream's macroblock
count against the GPU's hardware limit *lazily*, on the first real
``get_batch_frames()`` call — not when ``CreateSimpleDecoder`` succeeds. A
clip whose resolution exceeds that limit therefore passed
``_try_open_nvdec``'s try/except cleanly and only crashed deep inside the
main detection loop, uncaught (see
docs/superpowers/specs/2026-07-02-sequential-obb-gpu-fast-known-issues.md
section 1.2).

The fix adds a trial ``get_batch_frames(1)`` call inside ``_try_open_nvdec``
itself, in the same try/except as decoder setup, so this failure surfaces at
open time and the caller falls back to cv2 decode instead of crashing
mid-pass. Because NVDEC decode is forward-only, the trial-decoded frame's
content is cached and returned so it can be replayed as the first frame
instead of being silently discarded.

These tests exercise ``_try_open_nvdec`` / ``_read_nvdec_batch`` in isolation
with a fully mocked ``PyNvVideoCodec``/``cupy`` (no real GPU/NVDEC hardware
required).
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from hydra_suite.core.tracking.ingest.detection_phase import (
    _read_nvdec_batch,
    _try_open_nvdec,
)


def _install_fake_nvdec_modules(monkeypatch, decoder):
    """Inject fake ``PyNvVideoCodec``/``cupy`` modules into sys.modules so the
    function-local ``import cupy as cp`` / ``import PyNvVideoCodec as nvc``
    succeed and return our mocks."""
    fake_nvc = types.ModuleType("PyNvVideoCodec")
    fake_nvc.CreateSimpleDecoder = MagicMock(return_value=decoder)
    fake_nvc.OutputColorType = types.SimpleNamespace(RGB="RGB")

    fake_cp = types.ModuleType("cupy")

    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_nvc)
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    return fake_nvc, fake_cp


def _make_meta(num_frames=100, width=640, height=480):
    return types.SimpleNamespace(num_frames=num_frames, width=width, height=height)


def test_try_open_nvdec_falls_back_when_trial_decode_raises(monkeypatch):
    """A clip that exceeds the GPU's macroblock limit raises on the FIRST
    real get_batch_frames() call (not at CreateSimpleDecoder time). The
    trial-decode inside _try_open_nvdec must catch this and return None so
    the caller falls back to cv2 — not let it propagate into the main loop.
    """
    decoder = MagicMock()
    decoder.get_stream_metadata.return_value = _make_meta()
    decoder.get_batch_frames.side_effect = RuntimeError(
        "HandleVideoSequence: Error code 801, Error Type: MBCount not supported"
    )
    _install_fake_nvdec_modules(monkeypatch, decoder)

    result = _try_open_nvdec("fake_high_res_clip.mp4", start_frame=0)

    assert result is None
    decoder.get_batch_frames.assert_called_once_with(1)


def test_try_open_nvdec_succeeds_and_primes_first_frame(monkeypatch):
    """A clip NVDEC can actually decode: _try_open_nvdec must succeed and
    return the trial-decoded frame's content (not discard it) so the caller
    can replay it as frame 0 without a wasted/duplicate decode."""
    sentinel_frame = MagicMock()
    decoder = MagicMock()
    decoder.get_stream_metadata.return_value = _make_meta()
    decoder.get_batch_frames.return_value = [sentinel_frame]
    _install_fake_nvdec_modules(monkeypatch, decoder)

    fake_cuda_tensor = MagicMock()
    fake_cuda_tensor.clone.return_value = "PRIMED_TENSOR"
    monkeypatch.setattr(
        "hydra_suite.core.tracking.ingest.detection_phase._nvdec_frame_to_cuda_tensor",
        lambda frame, cp: fake_cuda_tensor,
    )

    result = _try_open_nvdec("fake_ok_clip.mp4", start_frame=0)

    assert result is not None
    dec, meta, cp, primed_frame = result
    assert dec is decoder
    assert primed_frame == "PRIMED_TENSOR"
    decoder.get_batch_frames.assert_called_once_with(1)


def test_try_open_nvdec_returns_none_on_missing_libs(monkeypatch):
    """No PyNvVideoCodec/cupy installed at all -> None, same as before."""
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)
    monkeypatch.setitem(sys.modules, "cupy", None)

    result = _try_open_nvdec("whatever.mp4", start_frame=0)

    assert result is None


def test_read_nvdec_batch_consumes_primed_frame_without_extra_decode(monkeypatch):
    """When a primed_frame is supplied, _read_nvdec_batch must use it as the
    first batch element and NOT call get_batch_frames() for that slot — the
    trial-decode already consumed frame 0 from the stream."""
    sdec = MagicMock()
    sdec.get_batch_frames.return_value = ["frame_1"]
    cp = MagicMock()
    monkeypatch.setattr(
        "hydra_suite.core.tracking.ingest.detection_phase._nvdec_frame_to_cuda_tensor",
        lambda frame, cp: MagicMock(clone=lambda: f"CLONED_{frame}"),
    )

    batch, consumed = _read_nvdec_batch(
        sdec,
        cp,
        batch_size=2,
        start_frame=0,
        end_frame=99,
        frame_idx=0,
        is_stop_requested=lambda: False,
        primed_frame="PRIMED_TENSOR",
    )

    assert consumed == 2
    assert batch[0] == "PRIMED_TENSOR"
    # Only ONE real decode call for the second batch slot — the primed frame
    # did not trigger a redundant get_batch_frames() call.
    assert sdec.get_batch_frames.call_count == 1


def test_read_nvdec_batch_without_primed_frame_behaves_as_before(monkeypatch):
    sdec = MagicMock()
    sdec.get_batch_frames.side_effect = [["f0"], ["f1"]]
    cp = MagicMock()
    monkeypatch.setattr(
        "hydra_suite.core.tracking.ingest.detection_phase._nvdec_frame_to_cuda_tensor",
        lambda frame, cp: MagicMock(clone=lambda: f"CLONED_{frame}"),
    )

    batch, consumed = _read_nvdec_batch(
        sdec,
        cp,
        batch_size=2,
        start_frame=0,
        end_frame=99,
        frame_idx=0,
        is_stop_requested=lambda: False,
        primed_frame=None,
    )

    assert consumed == 2
    assert sdec.get_batch_frames.call_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
