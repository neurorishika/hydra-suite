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
    """Write a tiny BGR video with ``n_frames`` solid-colour frames."""
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
