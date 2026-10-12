"""Centred window scale / aspect-ratio jitter for ViTPose fine-tuning.

Same augmentation as ``hydra_suite.training.scale_crop_aug.ScaleCropAug``
(classifiers) and ``ultralytics_window_aug`` (YOLO-pose): re-crop the image
about its fixed centre with a random window size (``scale_jitter``) and a
random width:height ratio at constant area (``aspect_jitter``). The animal is
never stretched and the crop centre never moves.

This package is a standalone leaf (it must not import ``hydra_suite``), so the
sampling math is deliberately duplicated; ``tests/test_vitpose_window_jitter.py``
pins it to ``ScaleCropAug`` draw-for-draw so the two cannot drift apart.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

_MIN_EDGE = 8


class WindowJitter:
    """Draw a centred window and re-crop an image plus its keypoints."""

    def __init__(
        self, scale_jitter: float = 0.0, aspect_jitter: float = 0.0, seed: int = 0
    ) -> None:
        self.scale_jitter = float(min(max(scale_jitter, 0.0), 0.9))
        self.aspect_jitter = float(min(max(aspect_jitter, 0.0), 1.0))
        self._rng = np.random.default_rng(seed)

    @property
    def active(self) -> bool:
        return self.scale_jitter > 0.0 or self.aspect_jitter > 0.0

    def sample_window(self, h: int, w: int) -> tuple[int, int]:
        """Return ``(out_w, out_h)``; consumes RNG in the ScaleCropAug order."""
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
        return (
            max(_MIN_EDGE, int(round(w * k * sqrt_r))),
            max(_MIN_EDGE, int(round(h * k / sqrt_r))),
        )

    def recrop(self, img: np.ndarray, kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Re-crop ``img`` about its centre; shift ``kp`` ``(K, 3)`` to match.

        Pixels outside the original image are black (as in ``top_down_affine``).
        Keypoints that fall outside the new window get visibility 0.
        """
        if not self.active:
            return img, kp
        h, w = img.shape[:2]
        out_w, out_h = self.sample_window(h, w)
        tx, ty = (out_w - w) / 2.0, (out_h - h) / 2.0
        m = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
        out = cv2.warpAffine(
            img,
            m,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        kp2 = kp.copy()
        kp2[:, 0] += tx
        kp2[:, 1] += ty
        outside = (
            (kp2[:, 0] < 0)
            | (kp2[:, 0] > out_w - 1)
            | (kp2[:, 1] < 0)
            | (kp2[:, 1] > out_h - 1)
        )
        kp2[outside, 2] = 0.0
        return np.ascontiguousarray(out), kp2
