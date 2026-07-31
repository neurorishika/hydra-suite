"""FrameSource abstraction for the inference pipeline.

A :class:`FrameSource` is an iterator over ``(frame_index, frame)`` pairs in
strict ascending frame-index order.  The CPU implementation wraps cv2 and
mirrors *exactly* the inline ``_frame_source`` generator that was previously
embedded in :meth:`~hydra_suite.core.inference.runner.InferenceRunner.run_batch_pass`.

The abstraction exists so Task 14 can drop in an NVDEC-backed reader without
touching ``run_batch_pass`` or ``Pipeline.run``.

Contract
--------
* Iteration yields ``(int, numpy.ndarray)`` pairs where the int is the
  absolute video frame index and the ndarray is an HWC uint8 BGR image (on
  the CPU path).  NVDEC implementations may yield a CUDA tensor instead of
  a numpy array.
* Frame indices are ascending and contiguous within the requested range.
  The range ``[start_frame, end_frame]`` is *inclusive* on both ends.
* ``frame_count`` returns the number of frames in the requested range
  (``end_frame - start_frame + 1`` after clamping to the video length).
  It is available immediately after construction, before iteration begins.
* ``close()`` releases underlying resources.  It is safe to call more than
  once.  Using the source as a context manager (``with CpuFrameReader(...) as
  src:``) calls ``close()`` on exit.

NVDEC parity note
-----------------
:class:`NvdecFrameReader` output is **NOT** bit-identical to
:class:`CpuFrameReader` output.  NVDEC uses NVIDIA's hardware chroma
sampling pipeline (YUV→RGB in hardware, then stored as RGB planes) whereas
cv2 uses a different YUV→BGR software conversion.  Downstream stages that
depend on exact per-pixel values must account for this; stages that work on
normalised float tensors are unaffected in practice.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from .runtime import RuntimeContext

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract base for frame providers consumed by :class:`~hydra_suite.core.inference.pipeline.Pipeline`.

    Subclasses yield ``(frame_index, frame)`` pairs in ascending index order
    for the range ``[start_frame, end_frame]`` supplied at construction time.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` pairs in ascending order."""
        ...

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total number of frames in the requested range."""
        ...

    @property
    @abstractmethod
    def start_frame(self) -> int:
        """First frame index (inclusive, 0-based, after clamping to video bounds)."""
        ...

    @property
    @abstractmethod
    def end_frame(self) -> int:
        """Last frame index (inclusive, 0-based, after clamping to video bounds)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""
        ...

    # Context-manager support so callers can write ``with CpuFrameReader(...) as s``.
    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class CpuFrameReader(FrameSource):
    """cv2.VideoCapture-backed :class:`FrameSource`.

    Mirrors the semantics of the inline ``_frame_source`` generator that was
    previously embedded in :meth:`~hydra_suite.core.inference.runner.InferenceRunner.run_batch_pass`:

    * Seeks to ``start_frame`` before iteration (``cap.set(CAP_PROP_POS_FRAMES)``).
    * Yields ``(idx, frame)`` for ``idx`` in ``[start_frame, end_frame]`` in
      ascending order, stopping early if ``cap.read()`` returns ``False``.
    * ``end_frame`` is clamped to ``video_total - 1`` so callers may pass an
      arbitrarily large sentinel (e.g. ``sys.maxsize``) without overshooting.
    * ``start_frame`` is clamped to ``max(0, start_frame)``.
    * Frames are ``numpy.ndarray`` HWC uint8 BGR (raw cv2 output, no colour
      conversion) — identical to what ``cap.read()`` returns.

    Parameters
    ----------
    video_path:
        Path to the video file.
    start_frame:
        First frame index (inclusive, 0-based).  Defaults to 0.
    end_frame:
        Last frame index (inclusive, 0-based).  ``None`` means the last frame
        of the video (``video_total - 1``).
    """

    def __init__(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        import cv2

        self._video_path = Path(video_path)
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self._video_path}")
        self._cap = cap
        self._cv2 = cv2

        video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = video_total - 1
        self._start_frame = max(0, int(start_frame))
        self._end_frame = min(video_total - 1, int(end_frame))
        self._frame_count = max(0, self._end_frame - self._start_frame + 1)

        # Seek to the start position upfront (mirrors run_batch_pass).
        if self._start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def end_frame(self) -> int:
        return self._end_frame

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` pairs from ``start_frame`` to ``end_frame``.

        Stops early on a failed ``cap.read()`` (e.g. corrupted/truncated video),
        exactly as the legacy inline generator did.
        """
        cap = self._cap
        if cap is None:
            return
        idx = self._start_frame
        end = self._end_frame
        while idx <= end:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def close(self) -> None:
        """Release the cv2 VideoCapture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore[assignment]


class NvdecFrameReader(FrameSource):
    """PyNvVideoCodec hardware-decode :class:`FrameSource` (CUDA only).

    Frames are decoded directly into CUDA device memory and returned as
    ``torch.Tensor`` objects on ``"cuda"`` with dtype ``torch.uint8`` and
    shape ``(H, W, 3)`` in **RGB** channel order.

    .. warning::
        Output is **NOT** bit-identical to :class:`CpuFrameReader`.  NVDEC
        decodes via the hardware YUV→RGB path; cv2 uses a different software
        YUV→BGR pipeline.  See module docstring for details.

    Parameters
    ----------
    video_path:
        Path to the video file.
    device:
        CUDA device string, e.g. ``"cuda"`` or ``"cuda:0"``.  Must be a CUDA
        device; any other value raises ``ValueError`` at construction time.
    start_frame:
        First frame index (inclusive, 0-based).  Defaults to 0.
    end_frame:
        Last frame index (inclusive, 0-based).  ``None`` means the last frame
        of the video (inferred from stream metadata).

    Raises
    ------
    ImportError
        If ``PyNvVideoCodec`` or ``cupy`` are not installed.
    ValueError
        If *device* is not a CUDA device.
    RuntimeError
        If the NVDEC decoder fails to open the video.
    """

    def __init__(
        self,
        video_path: str | Path,
        device: str = "cuda:0",
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        # Validate device before touching the decoder.
        if not str(device).startswith("cuda"):
            raise ValueError(
                f"NvdecFrameReader requires a CUDA device, got: {device!r}"
            )

        import cupy as cp  # noqa: F401 — validate importable
        import PyNvVideoCodec as nvc

        self._video_path = Path(video_path)
        self._device = device
        self._cp = cp

        # TODO: multi-GPU — parse device index from runtime.device instead of gpuid=0
        # Decode to NATIVE (NV12) rather than the decoder's built-in RGB: its
        # hardware YUV->RGB uses a color matrix/range (BT.709 / full range for a
        # large untagged frame) that differs from cv2's hardcoded BT.601 limited
        # range by up to 16/255 per channel -- fatal for a colortag identity
        # classifier. We do the YUV->RGB ourselves (``_nv12_to_rgb_bt601``) with
        # cv2-matching BT.601 limited-range coefficients so NVDEC frames match the
        # CPU-decode path the classifier was trained on.
        dec = nvc.CreateSimpleDecoder(
            encSource=str(self._video_path),
            gpuid=0,
            useDeviceMemory=True,
            outputColorType=nvc.OutputColorType.NATIVE,
        )

        try:
            meta = dec.get_stream_metadata()
            total_frames: int = (
                int(meta.num_frames) if hasattr(meta, "num_frames") else 0
            )

            self._start_frame: int = max(0, int(start_frame))

            if end_frame is None and total_frames <= 0:
                # MJPEG / variable-length / metadata-absent stream: decoder did not
                # report a frame count.  Probe with cv2 (same approach as
                # CpuFrameReader) so we have a concrete upper bound for progress
                # reporting and the __iter__ loop.  If cv2 also returns 0, we log a
                # WARNING and fall back to decode-until-exhausted (end_frame=None
                # sentinel stored as sys.maxsize so __iter__ runs until the decoder
                # returns an empty batch).
                import cv2

                probe = cv2.VideoCapture(str(self._video_path))
                cv2_total = (
                    int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) if probe.isOpened() else 0
                )
                probe.release()

                if cv2_total > 0:
                    total_frames = cv2_total
                    logger.debug(
                        "NvdecFrameReader: stream metadata missing frame count for %s; "
                        "cv2 probe reports %d frames",
                        self._video_path,
                        total_frames,
                    )
                else:
                    import sys

                    logger.warning(
                        "NvdecFrameReader: frame count unknown for %s (neither NVDEC "
                        "metadata nor cv2 CAP_PROP_FRAME_COUNT could determine it). "
                        "Reader will iterate until the decoder returns no frames.",
                        self._video_path,
                    )
                    # Use maxsize as an open-ended sentinel; __iter__ stops on empty batch.
                    total_frames = sys.maxsize

            if end_frame is None:
                self._end_frame = total_frames - 1
            else:
                if total_frames > 0:
                    import sys

                    if total_frames < sys.maxsize:
                        self._end_frame = min(total_frames - 1, int(end_frame))
                    else:
                        self._end_frame = int(end_frame)
                else:
                    self._end_frame = int(end_frame)

            self._frame_count = max(0, self._end_frame - self._start_frame + 1)

            if self._start_frame > 0:
                dec.seek_to_index(self._start_frame)

            # Trial-decode the first frame to validate NVDEC can actually decode
            # this stream on THIS GPU before committing to the NVDEC path. Some
            # GPUs reject clips above a per-frame macroblock limit ("MBCount not
            # supported", error 801) — that failure surfaces HERE at the first
            # get_batch_frames, not at decoder creation. Raising now lets
            # make_frame_source fall back to CpuFrameReader gracefully instead of
            # crashing mid-pass. The decoded frame is cached and yielded first by
            # __iter__ so the trial does not consume a frame.
            self._primed_frame = None
            if self._frame_count > 0:
                first_batch = dec.get_batch_frames(1)
                if first_batch:
                    self._primed_frame = self._nvdec_frame_to_cuda_tensor(
                        first_batch[0]
                    ).clone()

        except Exception:
            # Decoder was created but setup failed — release the hardware context
            # before re-raising so no decode slot is leaked.
            dec = None  # PyNvVideoCodec decoder GC'd on dereference
            raise

        self._dec = dec
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def end_frame(self) -> int:
        return self._end_frame

    def _plane_to_tensor(self, plane, frame):
        """Wrap one NVDEC CUDA plane as a torch.Tensor (zero-copy)."""
        import torch

        cp = self._cp
        cai = plane.__cuda_array_interface__
        shape = cai["shape"]
        byte_size = 1
        for d in shape:
            byte_size *= int(d)
        mem = cp.cuda.UnownedMemory(cai["data"][0], byte_size, frame)
        ptr = cp.cuda.MemoryPointer(mem, 0)
        cp_arr = cp.ndarray(
            shape=shape, dtype=cp.uint8, memptr=ptr, strides=cai.get("strides") or None
        )
        return torch.as_tensor(cp_arr, device="cuda")

    def _nv12_to_rgb_bt601(self, y, uv):
        """Convert an NV12 (Y + interleaved UV) frame to ``(H, W, 3)`` uint8 RGB
        on-GPU with cv2-matching BT.601 limited-range coefficients.

        Mirrors OpenCV ``COLOR_YUV2RGB_NV12`` (ITU-R BT.601, limited/studio range,
        nearest 2x2 chroma upsampling), so a hardware-decoded frame matches the
        cv2 CPU-decode path the identity/pose models were trained on.
        """
        import torch

        y = y.reshape(y.shape[0], y.shape[1]).float()  # (H, W)
        H, W = y.shape
        u = uv[..., 0].float()
        v = uv[..., 1].float()
        # nearest chroma upsampling: duplicate each 2x2 block (matches cv2 NV12).
        u = u.repeat_interleave(2, 0).repeat_interleave(2, 1)[:H, :W]
        v = v.repeat_interleave(2, 0).repeat_interleave(2, 1)[:H, :W]
        yf = (y - 16.0) * 1.164
        uf = u - 128.0
        vf = v - 128.0
        r = yf + 1.596 * vf
        g = yf - 0.813 * vf - 0.391 * uf
        b = yf + 2.018 * uf
        return torch.stack([r, g, b], dim=-1).clamp_(0.0, 255.0).to(torch.uint8)

    def _nvdec_frame_to_cuda_tensor(self, frame):
        """Convert a PyNvVideoCodec NATIVE (NV12) DecodedFrame to a ``(H, W, 3)``
        uint8 RGB CUDA torch.Tensor with cv2-matching BT.601 conversion.

        The returned tensor is a fresh allocation (the conversion allocates), so
        it does not alias the decoder buffer -- but the caller still clones for
        parity with the historical contract.
        """
        planes = frame.cuda()
        if not planes or len(planes) < 2:
            raise ValueError(
                f"NVDec NATIVE frame expected Y+UV planes, got {len(planes) if planes else 0}"
            )
        y = self._plane_to_tensor(planes[0], frame)  # (H, W, 1)
        uv = self._plane_to_tensor(planes[1], frame)  # (H/2, W/2, 2)
        return self._nv12_to_rgb_bt601(y, uv)

    def __iter__(self) -> Iterator[tuple[int, "np.ndarray"]]:
        """Yield ``(frame_index, cuda_tensor)`` pairs from ``start_frame`` to ``end_frame``.

        Each frame tensor is immediately cloned so the decoder buffer can be
        safely reused for the next ``get_batch_frames()`` call.  Stops early
        if the decoder returns no frames (end of stream).
        """
        if self._closed or self._dec is None:
            return

        idx = self._start_frame
        end = self._end_frame
        # Yield the frame primed during construction (the trial-decode that
        # validated NVDEC works on this GPU) first, so the validation decode is
        # not wasted. It was already cloned, so it is safe to yield directly.
        if self._primed_frame is not None:
            yield idx, self._primed_frame
            self._primed_frame = None
            idx += 1
        while idx <= end:
            batch = self._dec.get_batch_frames(1)
            if not batch:
                break
            cuda_tensor = self._nvdec_frame_to_cuda_tensor(batch[0])
            # Clone immediately — NVDec decoder buffer is reused on next get_batch_frames().
            yield idx, cuda_tensor.clone()
            idx += 1

    def close(self) -> None:
        """Release the NVDEC decoder handle."""
        if not self._closed:
            self._dec = None  # PyNvVideoCodec decoder GC'd on dereference
            self._closed = True


def make_frame_source(
    video_path: str | Path,
    runtime: "RuntimeContext",
    start_frame: int = 0,
    end_frame: int | None = None,
) -> FrameSource:
    """Factory: return the best available :class:`FrameSource` for *video_path*.

    Selects :class:`NvdecFrameReader` when ``runtime.use_nvdec`` is ``True``
    **and** the NVDEC decoder can be imported and opened successfully.
    Falls back to :class:`CpuFrameReader` in all other cases (missing library,
    decoder-open failure, non-CUDA runtime) and logs a notice.

    Parameters
    ----------
    video_path:
        Path to the video file.
    runtime:
        Active :class:`~hydra_suite.core.inference.runtime.RuntimeContext`.
    start_frame:
        First frame index (inclusive, 0-based).  Defaults to 0.
    end_frame:
        Last frame index (inclusive, 0-based).  ``None`` means the last frame.

    Returns
    -------
    FrameSource
        Either a :class:`NvdecFrameReader` or :class:`CpuFrameReader`.
    """
    if runtime.use_nvdec:
        try:
            reader = NvdecFrameReader(
                video_path,
                device=runtime.device,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            logger.info("make_frame_source: using NvdecFrameReader for %s", video_path)
            return reader
        except Exception as exc:
            logger.warning(
                "make_frame_source: NVDEC unavailable (%s), falling back to CpuFrameReader "
                "(use_nvdec=True was requested but could not be satisfied)",
                exc,
            )

    logger.info("make_frame_source: using CpuFrameReader for %s", video_path)
    return CpuFrameReader(video_path, start_frame=start_frame, end_frame=end_frame)
