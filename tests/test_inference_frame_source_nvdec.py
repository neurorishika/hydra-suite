"""Tests for NvdecFrameReader and make_frame_source.

NVDEC tests are skipped on machines without CUDA + PyNvVideoCodec (e.g. this
MPS/CPU dev machine).  The fallback tests (make_frame_source with use_nvdec=False,
or with use_nvdec=True but PyNvVideoCodec absent) run on all platforms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixture: tiny synthetic AVI (shared with cpu tests)
# ---------------------------------------------------------------------------


def _write_tiny_video(path: Path, n_frames: int = 5, width: int = 16, height: int = 12):
    """Write a tiny BGR video with ``n_frames`` solid-colour frames.

    NOTE: The MJPEG codec is used for portability on all platforms (including MPS
    dev machines that lack x264).  MJPEG may not engage hardware NVDEC on all
    NVIDIA GPUs — only a subset of Turing/Ampere/Ada cards support MJPEG NVDEC.
    For the real ``mehek`` hardware-decode run, prefer an H.264 or H.265 clip
    so NVDEC engagement is guaranteed.
    """
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    assert writer.isOpened(), f"VideoWriter failed for {path}"
    for i in range(n_frames):
        color = (i * 40 % 256, i * 20 % 256, i * 10 % 256)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def tiny_video(tmp_path):
    p = tmp_path / "tiny.avi"
    _write_tiny_video(p, n_frames=5)
    return p


# ---------------------------------------------------------------------------
# Minimal RuntimeContext stub so tests don't need the real config machinery
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Minimal stand-in for RuntimeContext."""

    def __init__(self, use_nvdec: bool, device: str = "cpu"):
        self.use_nvdec = use_nvdec
        self.device = device


# ---------------------------------------------------------------------------
# NVDEC tests — SKIPPED unless CUDA + PyNvVideoCodec are both available
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def nvdec_skip():
    """Session-scoped skip marker; evaluated once."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — NvdecFrameReader cannot run")
    pytest.importorskip("PyNvVideoCodec", reason="PyNvVideoCodec not installed")
    pytest.importorskip("cupy", reason="cupy not installed")


def test_nvdec_reader_yields_cuda_tensors(nvdec_skip, tiny_video):
    """NvdecFrameReader yields CUDA tensors with the correct count and index range.

    This test only executes on machines that have CUDA + PyNvVideoCodec + cupy
    (e.g. 'mehek').  It is SKIPPED locally (MPS/CPU dev machine).
    """
    import torch

    from hydra_suite.core.inference.sources import NvdecFrameReader

    with NvdecFrameReader(
        tiny_video, device="cuda:0", start_frame=1, end_frame=3
    ) as src:
        assert src.start_frame == 1
        assert src.end_frame == 3
        assert src.frame_count == 3

        pairs = list(src)

    assert len(pairs) == 3
    for expected_idx, (actual_idx, tensor) in enumerate(pairs, start=1):
        assert (
            actual_idx == expected_idx
        ), f"expected idx {expected_idx}, got {actual_idx}"
        assert isinstance(tensor, torch.Tensor), "frame must be a torch.Tensor"
        assert tensor.device.type == "cuda", "tensor must be on CUDA device"
        assert tensor.dtype == torch.uint8, "tensor must be uint8"
        assert tensor.ndim == 3, "tensor must be HWC (3D)"
        assert tensor.shape[2] == 3, "tensor must have 3 channels (RGB)"


def test_nvdec_reader_full_range(nvdec_skip, tiny_video):
    """NvdecFrameReader with default range yields all 5 frames."""
    from hydra_suite.core.inference.sources import NvdecFrameReader

    with NvdecFrameReader(tiny_video, device="cuda:0") as src:
        assert src.frame_count == 5
        pairs = list(src)

    assert len(pairs) == 5
    indices = [idx for idx, _ in pairs]
    assert indices == list(range(5))


def test_nvdec_reader_rejects_non_cuda_device(tiny_video):
    """NvdecFrameReader raises ValueError when device is not CUDA.

    This test does NOT require CUDA to be available — the ValueError is raised
    before any CUDA call is made.
    """
    from hydra_suite.core.inference.sources import NvdecFrameReader

    with pytest.raises(ValueError, match="CUDA device"):
        NvdecFrameReader(tiny_video, device="cpu")


# ---------------------------------------------------------------------------
# make_frame_source — fallback tests (run on all platforms, including MPS)
# ---------------------------------------------------------------------------


def test_make_frame_source_returns_cpu_when_use_nvdec_false(tiny_video):
    """make_frame_source returns CpuFrameReader when runtime.use_nvdec is False."""
    from hydra_suite.core.inference.sources import CpuFrameReader, make_frame_source

    runtime = _FakeRuntime(use_nvdec=False, device="cpu")
    src = make_frame_source(tiny_video, runtime, start_frame=0)
    try:
        assert isinstance(
            src, CpuFrameReader
        ), f"expected CpuFrameReader, got {type(src).__name__}"
        assert src.frame_count == 5
    finally:
        src.close()


def test_make_frame_source_falls_back_to_cpu_when_nvdec_import_fails(tiny_video):
    """make_frame_source gracefully falls back to CpuFrameReader when PyNvVideoCodec
    is unavailable (the case on this MPS machine) — must never raise.
    """
    import sys

    from hydra_suite.core.inference.sources import CpuFrameReader, make_frame_source

    # Simulate use_nvdec=True but on a non-CUDA machine where the import fails.
    # On this dev machine PyNvVideoCodec is absent, so the fallback is exercised
    # without any monkey-patching.
    runtime = _FakeRuntime(use_nvdec=True, device="cuda:0")
    # This must NOT raise regardless of whether PyNvVideoCodec is installed.
    src = make_frame_source(tiny_video, runtime, start_frame=0)
    try:
        assert isinstance(
            src, CpuFrameReader
        ), f"expected CpuFrameReader fallback, got {type(src).__name__}"
        pairs = list(src)
        assert len(pairs) == 5
    finally:
        src.close()


def test_make_frame_source_respects_frame_range(tiny_video):
    """make_frame_source passes start_frame/end_frame through to the reader."""
    from hydra_suite.core.inference.sources import make_frame_source

    runtime = _FakeRuntime(use_nvdec=False, device="cpu")
    src = make_frame_source(tiny_video, runtime, start_frame=2, end_frame=4)
    try:
        assert src.start_frame == 2
        assert src.end_frame == 4
        assert src.frame_count == 3
    finally:
        src.close()


# ---------------------------------------------------------------------------
# cv2-probe frame count helper — CPU-testable (no CUDA needed)
# ---------------------------------------------------------------------------


def _nvdec_effective_end_frame(
    total_frames_from_metadata: int,
    end_frame_arg: "int | None",
    cv2_total: int,
    start_frame: int = 0,
) -> tuple[int, int]:
    """Pure helper that replicates the NvdecFrameReader end_frame resolution logic.

    Returns ``(effective_end_frame, effective_frame_count)`` given:

    * ``total_frames_from_metadata``: what ``meta.num_frames`` reported (0 = absent).
    * ``end_frame_arg``: the caller-supplied ``end_frame`` (``None`` = whole video).
    * ``cv2_total``: what ``cv2.CAP_PROP_FRAME_COUNT`` would report for the same file.
    * ``start_frame``: start of the requested range (default 0).

    This mirrors the branching in ``NvdecFrameReader.__init__`` so it can be
    unit-tested on CPU without constructing a real decoder.
    """
    import sys

    total_frames = total_frames_from_metadata

    if end_frame_arg is None and total_frames <= 0:
        # cv2 probe branch
        if cv2_total > 0:
            total_frames = cv2_total
        else:
            total_frames = sys.maxsize  # open-ended sentinel

    if end_frame_arg is None:
        eff_end = total_frames - 1
    else:
        if total_frames > 0 and total_frames < sys.maxsize:
            eff_end = min(total_frames - 1, int(end_frame_arg))
        else:
            eff_end = int(end_frame_arg)

    eff_count = max(0, eff_end - start_frame + 1)
    return eff_end, eff_count


def test_nvdec_frame_count_probe_metadata_present():
    """When metadata reports num_frames > 0, cv2 probe is not used."""
    end, count = _nvdec_effective_end_frame(
        total_frames_from_metadata=10, end_frame_arg=None, cv2_total=999
    )
    assert end == 9, f"expected 9, got {end}"
    assert count == 10, f"expected 10, got {count}"


def test_nvdec_frame_count_probe_metadata_absent_cv2_succeeds():
    """When metadata is missing (0) and end_frame=None, cv2 probe gives the count."""
    end, count = _nvdec_effective_end_frame(
        total_frames_from_metadata=0, end_frame_arg=None, cv2_total=5
    )
    assert end == 4, f"expected 4, got {end}"
    assert count == 5, f"expected 5, got {count}"


def test_nvdec_frame_count_probe_metadata_absent_cv2_also_fails():
    """When both metadata and cv2 return 0, open-ended sentinel is used."""
    import sys

    end, count = _nvdec_effective_end_frame(
        total_frames_from_metadata=0, end_frame_arg=None, cv2_total=0
    )
    # open-ended: end_frame = sys.maxsize - 1, count = sys.maxsize
    assert end == sys.maxsize - 1
    assert count == sys.maxsize


def test_nvdec_frame_count_explicit_end_frame_with_metadata():
    """Explicit end_frame is clamped to metadata total when metadata is present."""
    end, count = _nvdec_effective_end_frame(
        total_frames_from_metadata=5, end_frame_arg=10, cv2_total=999
    )
    assert end == 4, f"expected 4 (clamped), got {end}"
    assert count == 5, f"expected 5, got {count}"


def test_nvdec_frame_count_no_silent_single_frame_on_missing_metadata(tiny_video):
    """Regression guard: end_frame=None + missing metadata must NOT yield frame_count=1.

    The old code set end_frame = start_frame when total_frames=0 and end_frame=None,
    silently truncating whole-video passes to a single frame.  This test uses the
    pure helper to confirm the cv2-probe branch gives the real count instead.
    """
    import cv2

    # Determine real frame count via cv2 (our fixture video has 5 frames).
    cap = cv2.VideoCapture(str(tiny_video))
    cv2_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    end, count = _nvdec_effective_end_frame(
        total_frames_from_metadata=0,  # simulate absent NVDEC metadata
        end_frame_arg=None,
        cv2_total=cv2_total,
    )
    assert count > 1, (
        f"frame_count must not be 1 when metadata is absent and end_frame=None; "
        f"cv2 reported {cv2_total} frames, got count={count}"
    )
