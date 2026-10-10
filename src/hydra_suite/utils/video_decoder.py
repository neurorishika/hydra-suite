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
from typing import Callable, NamedTuple, Optional, Protocol, Sequence

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


def _plane_view(plane, width: int, height: int, channels: int = 1) -> np.ndarray:
    """Zero-copy (height, width[, channels]) uint8 view of one ``av`` plane.

    Rows keep their ``line_size`` stride; OpenCV reads strided rows directly,
    so no plane is ever copied just to drop the padding.
    """
    rows = np.frombuffer(plane, np.uint8)[: plane.line_size * height].reshape(
        height, plane.line_size
    )
    view = rows[:, : width * channels]
    return view if channels == 1 else view.reshape(height, width, channels)


def _resize_into(src: np.ndarray, dst: np.ndarray) -> None:
    if src.shape[:2] == dst.shape[:2]:
        np.copyto(dst, src)
    else:
        cv2.resize(
            src, (dst.shape[1], dst.shape[0]), dst=dst, interpolation=cv2.INTER_AREA
        )


def frame_to_bgr(frame, out_w: int, out_h: int) -> np.ndarray:
    """A decoded ``av.VideoFrame`` as contiguous BGR uint8 at ``out_w x out_h``.

    Even-sized limited-range yuv420p / nv12 (the common case) is read straight
    from the frame's planes (no ``to_ndarray`` copies), resized per plane with
    INTER_AREA and converted by OpenCV (BT.601 limited range, the matrix
    swscale uses for these streams) -- several times faster than swscale's
    single-threaded bgr24 path on 4K+ frames. Everything else (odd sizes,
    full range, 10-bit, other layouts) goes through swscale with area
    interpolation.
    """
    fmt = frame.format.name
    w, h = frame.width, frame.height
    full_range = int(getattr(frame, "color_range", 0) or 0) == 2
    if (
        fmt in _LIMITED_RANGE_420
        and not full_range
        and w % 2 == 0
        and h % 2 == 0
        and out_w % 2 == 0
        and out_h % 2 == 0
    ):
        planes = frame.planes
        y = _plane_view(planes[0], w, h)
        if fmt == "nv12":
            uv = _plane_view(planes[1], w // 2, h // 2, 2)
            if (out_w, out_h) != (w, h):
                y = cv2.resize(y, (out_w, out_h), interpolation=cv2.INTER_AREA)
                uv = cv2.resize(
                    uv, (out_w // 2, out_h // 2), interpolation=cv2.INTER_AREA
                )
            return cv2.cvtColorTwoPlane(y, uv, cv2.COLOR_YUV2BGR_NV12)
        packed = np.empty((out_h * 3 // 2, out_w), np.uint8)
        q = (out_h // 2) * (out_w // 2)
        flat = packed.reshape(-1)
        _resize_into(y, packed[:out_h])
        _resize_into(
            _plane_view(planes[1], w // 2, h // 2),
            flat[out_h * out_w : out_h * out_w + q].reshape(out_h // 2, out_w // 2),
        )
        _resize_into(
            _plane_view(planes[2], w // 2, h // 2),
            flat[out_h * out_w + q :].reshape(out_h // 2, out_w // 2),
        )
        return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_I420)
    out = frame.reformat(
        width=out_w, height=out_h, format="bgr24", interpolation="AREA"
    ).to_ndarray()
    return np.ascontiguousarray(out)


# ── PyAV readers ──────────────────────────────────────────────────────────────


_PAST_END = object()


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
        self.past_end = False  # stream ended before start_frame (not an error)
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
        drained = False
        for packet in self._container.demux(self._stream):
            if packet.size == 0:  # demux's own flush packet drains the decoder
                drained = True
                yield from self._cc.decode(None)
            else:
                yield from self._cc.decode(packet)
        if not drained:
            # A second drain raises EOFError ("End of file: avcodec_send_packet").
            yield from self._cc.decode(None)

    def _first_pts(self):
        for frame in self._decoded():
            return frame.pts
        return None

    def _provable_cfr_rate(self) -> Optional[Fraction]:
        """The nominal rate IF pts -> index is provably exact, else None.

        Exact means: the container's duration is exactly ``frames / rate``
        (every frame interval nominal, so no dropped frame / VFR anywhere in
        the file) -- checked again per decoded frame in ``_seek_frames``.
        ``average_rate`` is NOT used: it is a measured mean, and on a stream
        with one dropped frame it made the index drift past 0.5 mid-file.
        """
        s = self._stream
        rate, tb, frames, duration = s.guessed_rate, s.time_base, s.frames, s.duration
        if not (rate and tb and frames and duration):
            return None
        rate = Fraction(rate)
        if Fraction(duration) * Fraction(tb) * rate != frames:
            return None
        return rate

    def _seek_frames(self):
        """Frames from ``start_frame`` via keyframe seek.

        Returns ``(first, rest)``, ``_PAST_END`` when the stream ends before
        ``start_frame``, or None when the pts cannot be trusted (caller then
        counts decoded frames from the start of the stream).
        """
        rate = self._provable_cfr_rate()
        tb = self._stream.time_base
        if rate is None:
            return None
        pts0 = self._first_pts()
        if pts0 is None:
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
            exact = (frame.pts - pts0) * Fraction(tb) * rate
            if exact.denominator != 1:
                return None  # off-grid pts: not provably CFR
            idx = int(exact)
            if prev is not None and idx != prev + 1:
                return None
            if prev is None and idx > self._start:
                return None  # seek overshot the start
            prev = idx
            if idx == self._start:
                return frame, decoded
        return _PAST_END if prev is not None else None

    def _frames(self):
        if self._start > 0:
            found = self._seek_frames()
            if found is _PAST_END:
                self.past_end = True
                return
            if found is None:
                logger.warning(
                    "Annotated video: frame timing of %s is not provably constant;"
                    " decoding from frame 0 to reach start frame %d.",
                    self._path,
                    self._start,
                )
                self._close_container()
                self._open()
                n = 0
                for idx, frame in enumerate(self._decoded()):
                    n = idx + 1
                    if idx >= self._start:
                        yield frame
                if 0 < n <= self._start:
                    self.past_end = True
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
        self._start = int(start_frame)
        self._delivered = 0
        self.past_end = False

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            # A seek past the end: cv2 opened the file but has nothing there.
            self.past_end = self._start > 0 and self._delivered == 0
            return None
        self._delivered += 1
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        return frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)

    def close(self) -> None:
        self._cap.release()


# ── ladder ────────────────────────────────────────────────────────────────────


class StreamFacts(NamedTuple):
    codec: Optional[str]
    width: int
    height: int
    rotated: bool  # display-rotation metadata that cv2 applies and PyAV doesn't


def _stream_facts(path: str) -> StreamFacts:
    codec, width, height = None, 0, 0
    try:
        import av

        with av.open(path) as container:
            stream = container.streams.video[0]
            codec, width, height = (
                stream.codec_context.name,
                stream.width,
                stream.height,
            )
    except Exception:
        pass
    rotated = False
    cap = cv2.VideoCapture(path)
    try:
        if cap.isOpened():
            rotated = bool(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    finally:
        cap.release()
    return StreamFacts(codec, width, height, rotated)


def _av_caps() -> tuple[set, set]:
    """(hw device types FFmpeg was built with, available codec names)."""
    import av
    from av.codec.hwaccel import hwdevices_available

    return set(hwdevices_available()), set(av.codecs_available)


def _cuda_usable() -> bool:
    """A CUDA device this process can actually use (respects CUDA_VISIBLE_DEVICES).

    Same check as ``batch_fanout.host_has_cuda``; a compiled-in ``cuda`` hw
    device type alone (the PyPI PyAV wheel on a CPU box) is not enough.
    """
    try:
        from hydra_suite.utils.gpu_utils import CUDA_AVAILABLE, TORCH_CUDA_AVAILABLE

        return bool(CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE)
    except Exception:  # noqa: BLE001 - no torch/cupy just means no CUDA
        return False


def default_decoder_candidates(
    video_path: str, out_w: int, out_h: int
) -> list[DecoderCandidate]:
    """The decoder ladder for this host and video, best first.

    * Rotation metadata -> ``opencv`` only: cv2 applies the display rotation
      (as tracking's ``CpuFrameReader`` did), PyAV/cuvid do not.
    * ``cuvid`` (NVDEC + on-GPU resize) first wherever a CUDA device is usable
      (Linux, Windows).
    * Then, measured (fix round 1): PyAV software beats VideoToolbox and the
      CUDA ``hwaccel`` (which download full-size frames) and beats cv2 when
      downscaling; at the source size cv2 -- today's path -- is as fast or
      faster, so it leads there.
    """
    opencv = DecoderCandidate(
        "opencv", lambda s: _OpenCVReader(video_path, s, out_w, out_h)
    )
    facts = _stream_facts(video_path)
    if facts.rotated:
        return [opencv]
    try:
        hw_devices, codecs = _av_caps()
    except Exception:
        return [opencv]
    cuda_platform = sys.platform.startswith("linux") or sys.platform == "win32"
    cuda_ok = cuda_platform and "cuda" in hw_devices and _cuda_usable()
    cands: list[DecoderCandidate] = []
    cuvid = f"{facts.codec}_cuvid" if facts.codec else None
    if cuda_ok and cuvid in codecs:
        cands.append(
            DecoderCandidate(
                "cuvid",
                lambda s, c=cuvid: _PyAVReader(
                    video_path, s, out_w, out_h, cuvid_decoder=c
                ),
            )
        )
    pyav = DecoderCandidate("pyav", lambda s: _PyAVReader(video_path, s, out_w, out_h))
    device = None
    if sys.platform == "darwin" and "videotoolbox" in hw_devices:
        device = "videotoolbox"
    elif cuda_ok:
        device = "cuda"
    hwaccel = (
        DecoderCandidate(
            "hwaccel",
            lambda s, d=device: _PyAVReader(
                video_path, s, out_w, out_h, hwaccel_device=d
            ),
        )
        if device
        else None
    )
    same_size = (out_w, out_h) == (facts.width, facts.height)
    tail = [opencv, pyav, hwaccel] if same_size else [pyav, hwaccel, opencv]
    cands.extend(c for c in tail if c is not None)
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
                if (
                    frame is None
                    and i < len(self._candidates) - 1
                    and not getattr(reader, "past_end", False)
                ):
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
