"""Scale and aspect-ratio jitter of the crop window (training-only).

Classifiers are trained on canonical crops whose window size and aspect ratio
are fixed by the canonicalization contract. At inference the detector box, the
per-frame scale estimate and the crop padding all wobble, so the same animal
arrives in a slightly larger, smaller, wider or narrower window.
``ScaleCropAug`` teaches the model to tolerate that.

It re-crops the image about its CENTRE (the centre is never moved):

* ``scale_jitter`` -- the window's overall size, factor ``k ~ U(1-j, 1+j)``
  (``k > 1`` is a wider view, so the animal ends up smaller after the model's
  fit-to-input step; ``k < 1`` is a tighter crop).
* ``aspect_jitter`` -- the window's width:height ratio, log-uniform in
  ``[1/(1+a), 1+a]`` while keeping the window area fixed.

The window is cut out by a pure translation (no resampling, no stretching of
the animal) and edge-replicated where it extends past the image. The output
therefore has a DIFFERENT shape/aspect than the input on purpose: every
training pipeline letterboxes it to the model input afterwards
(``CanonicalFitTransform``), which is what makes the animal look
larger/smaller/with more or less context.

Determinism: all randomness comes from a seeded ``numpy.random.Generator`` held
on the instance (same contract as ``CanonicalAug``); DataLoader workers are
decorrelated on first call.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["ScaleCropAug"]

_MIN_EDGE = 8


class ScaleCropAug:
    """Random centred window re-crop. Callable ``uint8 HWC -> uint8 HWC'``.

    Args:
        scale_jitter: window-size range; factor drawn from ``U(1-j, 1+j)``.
            ``0`` disables.
        aspect_jitter: window aspect-ratio range; ratio drawn log-uniformly from
            ``[1/(1+a), 1+a]`` at constant window area. ``0`` disables.
        seed: seed for the instance RNG.
    """

    def __init__(
        self,
        scale_jitter: float = 0.0,
        aspect_jitter: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.scale_jitter = float(min(max(scale_jitter, 0.0), 0.9))
        self.aspect_jitter = float(min(max(aspect_jitter, 0.0), 1.0))
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._worker_checked = False

    @property
    def active(self) -> bool:
        """True when at least one of size / aspect jitter is enabled."""
        return self.scale_jitter > 0.0 or self.aspect_jitter > 0.0

    def _maybe_decorrelate_worker(self) -> None:
        if self._worker_checked:
            return
        self._worker_checked = True
        try:
            import torch.utils.data as _tud

            info = _tud.get_worker_info()
        except Exception:
            info = None
        if info is None:
            return
        if self.seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng([int(self.seed), int(info.id)])

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import cv2

        arr = np.asarray(img)
        if not self.active:
            return arr
        self._maybe_decorrelate_worker()
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        h, w = arr.shape[:2]
        k = 1.0
        if self.scale_jitter > 0.0:
            k = float(
                self._rng.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter)
            )
        log_r = 0.0
        if self.aspect_jitter > 0.0:
            lim = math.log1p(self.aspect_jitter)
            log_r = float(self._rng.uniform(-lim, lim))
        sqrt_r = math.exp(0.5 * log_r)
        out_w = max(_MIN_EDGE, int(round(w * k * sqrt_r)))
        out_h = max(_MIN_EDGE, int(round(h * k / sqrt_r)))
        # Pure translation that puts the source centre on the window centre.
        tx = (out_w - w) / 2.0
        ty = (out_h - h) / 2.0
        m = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
        out = cv2.warpAffine(
            arr,
            m,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return np.ascontiguousarray(out)
