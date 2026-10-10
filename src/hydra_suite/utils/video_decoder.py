"""Decode stage for the final annotated video: a decoder ladder on a thread.

Frames come out strictly in order, already at the output size, as C-contiguous
BGR uint8 -- the exact shape the overlay draw and ``VideoEncoder`` want.

Ladder (first candidate whose probe decodes the first frame wins):

1. ``cuvid``   -- PyAV + NVDEC (``h264_cuvid``/``hevc_cuvid``) with the
   decoder's ``resize`` option, so the downscale happens on the GPU. Linux only.
2. ``hwaccel`` -- PyAV with a hardware device (``videotoolbox`` on macOS,
   ``cuda`` on Linux), frames transferred to system memory.
3. ``pyav``    -- PyAV software decode with frame threading.
4. ``opencv``  -- ``cv2.VideoCapture`` (the historical path).

A mid-stream decode error re-opens the NEXT candidate at the current frame
index, so a failing hardware path never drops or duplicates a frame; with no
candidate left the error is raised.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Optional, Protocol, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameReader(Protocol):
    """One opened decoder, positioned at its start frame."""

    def read(self) -> Optional[np.ndarray]:
        """The next frame (output size, BGR uint8, contiguous) or None at EOF."""

    def close(self) -> None:
        """Release the decoder."""


@dataclass(frozen=True)
class DecoderCandidate:
    """A named way to open the video at a given frame index."""

    name: str
    open: Callable[[int], FrameReader]


# ── conversion ────────────────────────────────────────────────────────────────

_LIMITED_RANGE_420 = ("yuv420p", "nv12")


def _resize_plane(plane: np.ndarray, w: int, h: int) -> np.ndarray:
    if plane.shape[1] == w and plane.shape[0] == h:
        return plane
    return cv2.resize(plane, (w, h), interpolation=cv2.INTER_AREA)


def frame_to_bgr(frame, out_w: int, out_h: int) -> np.ndarray:
    """A decoded ``av.VideoFrame`` as contiguous BGR uint8 at ``out_w x out_h``.

    Even-sized limited-range yuv420p / nv12 (the common case) is resized per
    plane with INTER_AREA and converted by OpenCV (BT.601 limited range, the
    matrix swscale uses for these streams) -- several times faster than
    swscale's single-threaded bgr24 path on 4K+ frames. Everything else (odd
    sizes, full range, 10-bit, other layouts) goes through swscale with
    area interpolation.
    """
    fmt = frame.format.name
    w, h = frame.width, frame.height
    full_range = int(getattr(frame, "color_range", 0) or 0) == 2
    if fmt in _LIMITED_RANGE_420 and not full_range and w % 2 == 0 and h % 2 == 0:
        arr = frame.to_ndarray()
        if arr.shape == (h * 3 // 2, w):
            y = _resize_plane(arr[:h], out_w, out_h)
            if fmt == "yuv420p":
                q = (h // 2) * (w // 2)
                chroma = arr[h:].reshape(-1)
                u = _resize_plane(
                    chroma[:q].reshape(h // 2, w // 2), out_w // 2, out_h // 2
                )
                v = _resize_plane(
                    chroma[q:].reshape(h // 2, w // 2), out_w // 2, out_h // 2
                )
                packed = np.concatenate(
                    (y.reshape(-1), u.reshape(-1), v.reshape(-1))
                ).reshape(out_h * 3 // 2, out_w)
                return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_I420)
            uv = arr[h:].reshape(h // 2, w // 2, 2)
            uv = _resize_plane(uv, out_w // 2, out_h // 2)
            packed = np.concatenate((y.reshape(-1), uv.reshape(-1))).reshape(
                out_h * 3 // 2, out_w
            )
            return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_NV12)
    out = frame.reformat(
        width=out_w, height=out_h, format="bgr24", interpolation="AREA"
    ).to_ndarray()
    return np.ascontiguousarray(out)


# ── PyAV readers ──────────────────────────────────────────────────────────────


class _PyAVReader:
    """PyAV decode positioned at ``start_frame``.

    Indexing: from frame 0 we just count decoded frames (robust to VFR and odd
    pts, and what cv2's sequential read does). For ``start_frame > 0`` we seek
    to the preceding keyframe and map pts -> index relative to the FIRST
    frame's pts (cv2's own convention), discarding until ``start_frame``; if
    pts are missing, non-monotonic, or skip over ``start_frame`` we restart and
    count from the beginning of the stream instead of guessing.
    """

    def __init__(
        self,
        path: str,
        start_frame: int,
        out_w: int,
        out_h: int,
        *,
        hwaccel_device: Optional[str] = None,
        cuvid_decoder: Optional[str] = None,
    ) -> None:
        import av

        self._av = av
        self._path = path
        self._start = int(start_frame)
        self._out_w, self._out_h = int(out_w), int(out_h)
        self._hwaccel_device = hwaccel_device
        self._cuvid_decoder = cuvid_decoder
        self._container = None
        self._cc = None
        self._open()
        self._iter = self._frames()

    # container / decoder plumbing

    def _open(self) -> None:
        av = self._av
        kwargs = {}
        if self._hwaccel_device:
            from av.codec.hwaccel import HWAccel

            kwargs["hwaccel"] = HWAccel(
                device_type=self._hwaccel_device, allow_software_fallback=False
            )
        self._container = av.open(self._path, **kwargs)
        self._stream = self._container.streams.video[0]
        if self._cuvid_decoder:
            cc = av.CodecContext.create(self._cuvid_decoder, "r")
            src = self._stream.codec_context
            if src.extradata:
                cc.extradata = src.extradata
            opts = {}
            if (self._out_w, self._out_h) != (self._stream.width, self._stream.height):
                opts["resize"] = f"{self._out_w}x{self._out_h}"
            if opts:
                cc.options = opts
            cc.open()
            self._cc = cc
        else:
            self._stream.thread_type = "AUTO"

    def _close_container(self) -> None:
        if self._cc is not None:
            try:
                self._cc.close()
            except Exception:
                pass
            self._cc = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    def _decoded(self):
        if self._cc is None:
            yield from self._container.decode(self._stream)
            return
        for packet in self._container.demux(self._stream):
            # The flush packet (``packet.size == 0``) drains the decoder.
            yield from self._cc.decode(None if packet.size == 0 else packet)
        yield from self._cc.decode(None)

    def _first_pts(self):
        for frame in self._decoded():
            return frame.pts
        return None

    def _rate(self) -> Optional[Fraction]:
        rate = self._stream.average_rate or self._stream.guessed_rate
        return Fraction(rate) if rate else None

    def _seek_frames(self):
        """Frames from ``start_frame`` via keyframe seek, or None if unreliable."""
        rate = self._rate()
        tb = self._stream.time_base
        pts0 = self._first_pts()
        if rate is None or tb is None or pts0 is None:
            return None
        target = pts0 + int(Fraction(self._start) / rate / Fraction(tb))
        self._container.seek(target, stream=self._stream, backward=True)
        if self._cc is not None:
            self._cc.flush_buffers()
        prev = None
        decoded = self._decoded()
        for frame in decoded:
            if frame.pts is None:
                return None
            idx = round((frame.pts - pts0) * Fraction(tb) * rate)
            if prev is not None and idx != prev + 1:
                return None
            if prev is None and idx > self._start:
                return None  # seek overshot the start
            prev = idx
            if idx == self._start:
                return frame, decoded
        return None

    def _frames(self):
        if self._start > 0:
            found = self._seek_frames()
            if found is None:
                logger.debug("pts unreliable for %s; counting from 0", self._path)
                self._close_container()
                self._open()
                for idx, frame in enumerate(self._decoded()):
                    if idx >= self._start:
                        yield frame
                return
            first, rest = found
            yield first
            yield from rest
            return
        yield from self._decoded()

    # public

    def read(self) -> Optional[np.ndarray]:
        frame = next(self._iter, None)
        if frame is None:
            return None
        return frame_to_bgr(frame, self._out_w, self._out_h)

    def close(self) -> None:
        self._close_container()


class _OpenCVReader:
    """``cv2.VideoCapture`` -- the historical decode path -- plus INTER_AREA."""

    def __init__(self, path: str, start_frame: int, out_w: int, out_h: int):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cv2 could not open {path}")
        if start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self._size = (int(out_w), int(out_h))

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            return None
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        return frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)

    def close(self) -> None:
        self._cap.release()


# ── ladder ────────────────────────────────────────────────────────────────────


def _stream_codec(path: str) -> Optional[str]:
    try:
        import av

        with av.open(path) as container:
            return container.streams.video[0].codec_context.name
    except Exception:
        return None


def default_decoder_candidates(
    video_path: str, out_w: int, out_h: int
) -> list[DecoderCandidate]:
    """The decoder ladder for this host, best first; ``opencv`` is always last."""
    cands: list[DecoderCandidate] = []
    try:
        import av
        from av.codec.hwaccel import hwdevices_available

        hw_devices = set(hwdevices_available())
        codecs = set(av.codecs_available)
    except Exception:
        av = None
    if av is not None:
        if sys.platform.startswith("linux") and "cuda" in hw_devices:
            codec = _stream_codec(video_path)
            cuvid = f"{codec}_cuvid" if codec else None
            if cuvid in codecs:
                cands.append(
                    DecoderCandidate(
                        "cuvid",
                        lambda s, c=cuvid: _PyAVReader(
                            video_path, s, out_w, out_h, cuvid_decoder=c
                        ),
                    )
                )
        device = {"darwin": "videotoolbox"}.get(sys.platform)
        if sys.platform.startswith("linux"):
            device = "cuda"
        if device and device in hw_devices:
            cands.append(
                DecoderCandidate(
                    "hwaccel",
                    lambda s, d=device: _PyAVReader(
                        video_path, s, out_w, out_h, hwaccel_device=d
                    ),
                )
            )
        cands.append(
            DecoderCandidate("pyav", lambda s: _PyAVReader(video_path, s, out_w, out_h))
        )
    cands.append(
        DecoderCandidate("opencv", lambda s: _OpenCVReader(video_path, s, out_w, out_h))
    )
    return cands


class FrameSource:
    """Opens the first working candidate and reads through mid-stream failures."""

    def __init__(
        self,
        candidates: Sequence[DecoderCandidate],
        *,
        start_frame: int,
        out_size: tuple[int, int],
    ) -> None:
        self._candidates = list(candidates)
        self._next_index = int(start_frame)
        self._out_size = out_size
        self._reader: Optional[FrameReader] = None
        self._pending: Optional[np.ndarray] = None
        self._pos = 0  # index into _candidates of the active reader
        self.name = ""
        self._open_from(0, first=True)

    def _open_from(self, pos: int, *, first: bool = False) -> None:
        errors = []
        for i in range(pos, len(self._candidates)):
            cand = self._candidates[i]
            reader = None
            try:
                reader = cand.open(self._next_index)
                frame = reader.read()
                if frame is None and i < len(self._candidates) - 1:
                    raise RuntimeError("probe decoded no frame")
            except Exception as exc:  # probe failed -> next candidate
                errors.append(f"{cand.name}: {exc}")
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
                continue
            self._reader, self._pending, self._pos = reader, frame, i
            self.name = cand.name
            w, h = self._out_size
            if first:
                logger.info(
                    "Annotated video decode: %s -> %dx%d%s",
                    cand.name,
                    w,
                    h,
                    f" (skipped: {'; '.join(errors)})" if errors else "",
                )
            else:
                logger.warning(
                    "Annotated video decode switched to %s at frame %d",
                    cand.name,
                    self._next_index,
                )
            return
        raise RuntimeError(
            "No video decoder could open the video: " + "; ".join(errors)
        )

    def read(self) -> Optional[np.ndarray]:
        if self._pending is not None:
            frame, self._pending = self._pending, None
        else:
            try:
                frame = self._reader.read()
            except Exception as exc:
                if self._pos >= len(self._candidates) - 1:
                    raise
                logger.warning(
                    "Decoder %s failed at frame %d (%s); falling back.",
                    self.name,
                    self._next_index,
                    exc,
                )
                self._reader.close()
                self._open_from(self._pos + 1)
                frame, self._pending = self._pending, None
        if frame is not None:
            self._next_index += 1
        return frame

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None


# ── threaded stage ────────────────────────────────────────────────────────────


class _Failure:
    def __init__(self, exc: BaseException):
        self.exc = exc


_EOF = object()


class ThreadedFrameReader:
    """Runs a ``FrameSource`` on its own thread behind a bounded queue.

    ``get()`` returns frames in order, ``None`` at the end, and re-raises any
    exception from the decode thread. ``close()`` stops the thread promptly
    even if it is blocked on a full queue, and closes the decoder.
    """

    def __init__(
        self,
        source_factory: Callable[[], FrameSource],
        *,
        max_frames: int,
        maxsize: int = 8,
        thread_wrapper: Callable[[Callable], Callable] = lambda fn: fn,
    ) -> None:
        self._factory = source_factory
        self._max = int(max_frames)
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=thread_wrapper(self._run), name="video-decode", daemon=True
        )
        self._done = False
        self.decoder_name = ""

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        source = None
        try:
            source = self._factory()
            self.decoder_name = source.name
            for _ in range(self._max):
                if self._stop.is_set():
                    return
                frame = source.read()
                if frame is None or not self._put(frame):
                    break
            self._put(_EOF)
        except Exception as exc:  # propagate to the consumer
            self._put(_Failure(exc))
        finally:
            if source is not None:
                source.close()

    def start(self) -> "ThreadedFrameReader":
        self._thread.start()
        return self

    def get(self) -> Optional[np.ndarray]:
        if self._done:
            return None
        item = self._q.get()
        if item is _EOF:
            self._done = True
            return None
        if isinstance(item, _Failure):
            self._done = True
            raise item.exc
        return item

    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    def close(self, timeout: float = 10.0) -> None:
        self._stop.set()
        while True:  # unblock a producer waiting on put()
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout)
