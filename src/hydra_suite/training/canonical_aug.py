"""Moderate robustness augmentation for canonical crops (training-only).

``CanonicalAug`` is applied to the canonical crop *before* the Layer-2
isotropic letterbox (``CanonicalFitTransform``). Its purpose is to make the
classifier/pose models robust to the small, benign variations in how a crop
can be resampled and mildly degraded across different compute paths
(different resize kernels, sub-pixel jitter from geometry rounding, JPEG
compression in a data pipeline, etc.) -- variation that is deliberately
*absent* from the deterministic canonicalization contract used at inference,
but that is worth teaching the model to tolerate during training.

Moderate profile (exact spec, see
``.superpowers/sdd/2026-08-06-canonicalization-divergence-fixes/task-13-brief.md``):

- Resample-kernel swap: randomly choose among {torch bilinear, torch
  bilinear+antialias, cv2 INTER_AREA, cv2 INTER_LINEAR} and round-trip the
  crop through a mild random rescale using that kernel.
- Sub-pixel warp jitter: ``dx, dy ~ U(-0.5, 0.5)`` px, ``dtheta ~ U(-1, 1)``
  deg, applied as a small affine perturbation.
- Mild degradation: with probability ``p_degrade`` (default 0.3), apply
  EITHER a small Gaussian blur OR a JPEG round-trip at quality
  ``q ~ U[85, 100]``.

Determinism: all randomness is drawn from a seeded ``np.random.Generator``
stored on the instance (``np.random.default_rng(seed)``). The global
``numpy.random`` state and Python's ``random`` module are never touched, so
the same seed always reproduces the same output for the same input -- this
is required for reproducible training runs and is covered by
``tests/test_canonical_aug.py``.

This transform is training-only: it must never be wired into the inference
path, and it is OFF by default (opt-in via the augmentation profile).
"""

from __future__ import annotations

import numpy as np

__all__ = ["CanonicalAug"]

_RESAMPLE_KERNELS = (
    "torch_bilinear",
    "torch_bilinear_antialias",
    "cv2_area",
    "cv2_linear",
)


class CanonicalAug:
    """Training-only "Moderate" robustness augmentation for canonical crops.

    Callable ``np.ndarray -> np.ndarray`` (uint8 HWC in/out, shape-preserving).
    Apply to the canonical crop *before* ``CanonicalFitTransform`` (Layer 2).
    """

    def __init__(self, seed: int | None = None, p_degrade: float = 0.3) -> None:
        self.seed = seed
        self.p_degrade = float(p_degrade)
        self._rng = np.random.default_rng(seed)
        self._worker_checked = False

    def _maybe_decorrelate_worker(self) -> None:
        """Give each DataLoader worker an independent augmentation stream.

        When a ``DataLoader`` uses ``num_workers > 0`` it forks/pickles this
        instance into each worker, so every worker inherits the SAME seeded
        ``_rng`` state and would draw identical augmentations -- reducing
        effective augmentation diversity. On the first call inside a worker we
        re-seed once, mixing the worker id in: a seeded ``CanonicalAug`` stays
        fully reproducible but decorrelated across workers, while an unseeded
        one draws fresh OS entropy per worker. A directly constructed instance
        (no worker context, e.g. tests) is left untouched, preserving the
        same-seed determinism contract.
        """
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
        self._maybe_decorrelate_worker()
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        orig_shape = arr.shape

        out = self._resample_kernel_swap(arr)
        out = self._subpixel_warp(out)
        out = self._mild_degrade(out)

        if out.shape != orig_shape:
            out = self._force_shape(out, orig_shape)
        return np.ascontiguousarray(out.astype(np.uint8))

    # -- resample-kernel swap -------------------------------------------------

    def _resample_kernel_swap(self, img: np.ndarray) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        kernel_idx = int(self._rng.integers(0, len(_RESAMPLE_KERNELS)))
        kernel = _RESAMPLE_KERNELS[kernel_idx]
        scale = float(self._rng.uniform(0.85, 1.15))
        mid_h = max(1, int(round(h * scale)))
        mid_w = max(1, int(round(w * scale)))

        if kernel == "torch_bilinear":
            mid = self._torch_resize(img, mid_h, mid_w, antialias=False)
            out = self._torch_resize(mid, h, w, antialias=False)
        elif kernel == "torch_bilinear_antialias":
            mid = self._torch_resize(img, mid_h, mid_w, antialias=True)
            out = self._torch_resize(mid, h, w, antialias=True)
        elif kernel == "cv2_area":
            mid = cv2.resize(img, (mid_w, mid_h), interpolation=cv2.INTER_AREA)
            out = cv2.resize(mid, (w, h), interpolation=cv2.INTER_AREA)
        else:  # cv2_linear
            mid = cv2.resize(img, (mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
            out = cv2.resize(mid, (w, h), interpolation=cv2.INTER_LINEAR)
        return out

    @staticmethod
    def _torch_resize(
        img: np.ndarray, out_h: int, out_w: int, antialias: bool
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        squeeze_channel = img.ndim == 2
        arr = img[:, :, None] if squeeze_channel else img
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        t = t.unsqueeze(0).float()
        resized = F.interpolate(
            t,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
            antialias=antialias,
        )
        resized = resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).round()
        out = resized.numpy().astype(np.uint8)
        if squeeze_channel:
            out = out[:, :, 0]
        return out

    # -- sub-pixel warp jitter -------------------------------------------------

    def _subpixel_warp(self, img: np.ndarray) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        dx = float(self._rng.uniform(-0.5, 0.5))
        dy = float(self._rng.uniform(-0.5, 0.5))
        dtheta = float(self._rng.uniform(-1.0, 1.0))
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, dtheta, 1.0)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        warped = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    # -- mild degradation --------------------------------------------------

    def _mild_degrade(self, img: np.ndarray) -> np.ndarray:
        import cv2

        if self._rng.random() >= self.p_degrade:
            return img

        if self._rng.random() < 0.5:
            sigma = float(self._rng.uniform(0.3, 0.8))
            return cv2.GaussianBlur(img, (3, 3), sigma)

        quality = int(self._rng.integers(85, 101))
        ok, encoded = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return img
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            return img
        if decoded.ndim != img.ndim:
            if decoded.ndim == 2 and img.ndim == 3:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
                if img.shape[2] != decoded.shape[2]:
                    decoded = self._force_shape(decoded, img.shape)
            else:
                decoded = self._force_shape(decoded, img.shape)
        return decoded

    @staticmethod
    def _force_shape(img: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        import cv2

        h, w = shape[0], shape[1]
        out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if out.ndim != len(shape):
            out = out.reshape(shape)
        elif out.shape != shape:
            out = np.resize(out, shape)
        return out
