"""Hardware-accelerated video encoder with automatic backend selection.

Backend priority: nvenc (NVIDIA) → videotoolbox (macOS) → pyav_software (libx264)
→ opencv (mp4v software fallback).

PyAV is an optional dependency — on CPU-only systems the module falls back to
cv2.VideoWriter automatically.  Install PyAV via the [cuda] or [mps] extras:

    pip install "hydra-suite[mps]"   # Apple Silicon
    pip install "hydra-suite[cuda]"  # NVIDIA
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# ── Global state ──────────────────────────────────────────────────────────────

_BACKEND_CACHE: Optional[str] = None

# Consumer NVIDIA GPUs (pre-Ampere) allow ≤ 3 concurrent NVENC sessions.
# RTX 30-series and newer have no limit; 4 is a safe default across generations.
_NVENC_ACTIVE: int = 0
_NVENC_MAX: int = 4

# ── Backend detection ─────────────────────────────────────────────────────────

_AV_CODEC_NAME = {
    "nvenc": "h264_nvenc",
    "videotoolbox": "h264_videotoolbox",
    "pyav_software": "libx264",
}

_AV_CODEC_OPTS: dict[str, dict[str, str]] = {
    "nvenc": {"preset": "p4", "tune": "ll"},
    "videotoolbox": {},
    "pyav_software": {"preset": "fast", "crf": "23"},
}


def _try_encode(codec_name: str) -> bool:
    """Return True if codec_name successfully encodes a 2x2 test clip."""
    try:
        import av

        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            container = av.open(tmp, mode="w")
            stream = container.add_stream(codec_name, rate=1)
            stream.width = 2
            stream.height = 2
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame(2, 2, "yuv420p")
            for pkt in stream.encode(frame):
                container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            return True
        except Exception:
            return False
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception:
        return False


def _probe_backend() -> str:
    """Detect the best available video encoding backend."""
    try:
        import av
    except ImportError:
        return "opencv"

    available: set[str] = av.codecs_available  # type: ignore[attr-defined]

    if "h264_nvenc" in available and _try_encode("h264_nvenc"):
        return "nvenc"

    if sys.platform == "darwin" and "h264_videotoolbox" in available:
        if _try_encode("h264_videotoolbox"):
            return "videotoolbox"

    if "libx264" in available and _try_encode("libx264"):
        return "pyav_software"

    return "opencv"


def probe_video_backend() -> str:
    """Return the best available video encoding backend (result is cached)."""
    global _BACKEND_CACHE
    if _BACKEND_CACHE is None:
        _BACKEND_CACHE = _probe_backend()
    return _BACKEND_CACHE


# ── VideoEncoder ──────────────────────────────────────────────────────────────


class VideoEncoder:
    """Frame-by-frame video encoder with automatic hardware-acceleration selection.

    Accepts BGR uint8 frames (same as cv2.VideoWriter).  The backend is
    auto-detected once per process and cached; pass ``backend`` explicitly to
    override (useful for testing or when the caller knows the environment).

    NVENC concurrent sessions are tracked globally; when the cap is reached new
    encoders silently fall back to libx264 software encoding.

    Usage::

        with VideoEncoder(path, fps=30, width=1920, height=1080) as enc:
            for frame_bgr in frames:
                enc.write(frame_bgr)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float,
        width: int,
        height: int,
        backend: Optional[str] = None,
    ) -> None:
        self._path = Path(path)
        self._fps = float(fps)
        self._width = int(width)
        self._height = int(height)
        self._backend: str = backend or probe_video_backend()
        self._used_nvenc: bool = False
        self._container = None
        self._stream = None
        self._cv_writer = None
        self._open()

    # ── internal ──────────────────────────────────────────────────────────────

    def _open(self) -> None:
        global _NVENC_ACTIVE
        if self._backend == "nvenc":
            if _NVENC_ACTIVE >= _NVENC_MAX:
                self._backend = "pyav_software"
            else:
                _NVENC_ACTIVE += 1
                self._used_nvenc = True

        if self._backend in _AV_CODEC_NAME:
            try:
                import av
            except ImportError:
                # PyAV not available; fall back to opencv
                self._backend = "opencv"

        if self._backend in _AV_CODEC_NAME:
            import av

            codec = _AV_CODEC_NAME[self._backend]
            opts = _AV_CODEC_OPTS[self._backend]
            self._container = av.open(str(self._path), mode="w")
            self._stream = self._container.add_stream(codec, rate=int(self._fps))
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = "yuv420p"
            if opts:
                self._stream.options = opts
        else:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._cv_writer = cv2.VideoWriter(
                str(self._path), fourcc, self._fps, (self._width, self._height)
            )

    # ── public API ────────────────────────────────────────────────────────────

    def write(self, frame_bgr: np.ndarray) -> None:
        """Write one BGR uint8 frame (same interface as cv2.VideoWriter.write)."""
        if self._stream is not None:
            import av

            frame_rgb = frame_bgr[:, :, ::-1].copy()
            vf = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            vf = vf.reformat(format="yuv420p")
            for pkt in self._stream.encode(vf):
                self._container.mux(pkt)
        elif self._cv_writer is not None:
            self._cv_writer.write(frame_bgr)

    def release(self) -> None:
        """Flush encoder buffers and close the output file."""
        global _NVENC_ACTIVE
        if self._container is not None:
            try:
                for pkt in self._stream.encode():
                    self._container.mux(pkt)
                self._container.close()
            except Exception:
                pass
            self._container = None
            self._stream = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
        if self._used_nvenc:
            _NVENC_ACTIVE = max(0, _NVENC_ACTIVE - 1)
            self._used_nvenc = False

    def __enter__(self) -> "VideoEncoder":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
