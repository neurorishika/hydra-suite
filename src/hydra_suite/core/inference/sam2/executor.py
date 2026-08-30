"""Standalone torch-only SAM2 prompt-in/mask-out executor (lazy sam2 import)."""

from __future__ import annotations

import cv2
import numpy as np

from hydra_suite.core.inference.torch_device import resolve_torch_device

from .checkpoints import SAM2_VARIANTS, ensure_checkpoint

# Kept as a name for existing callers; the logic lives in torch_device.py.
resolve_sam2_device = resolve_torch_device


class Sam2SegmentExecutor:
    """Wraps a SAM2 image predictor: set_image once, segment per prompt."""

    def __init__(self, predictor) -> None:
        self._predictor = predictor

    @classmethod
    def from_variant(
        cls, variant: str, device: str | None = None, *, allow_download: bool = True
    ) -> "Sam2SegmentExecutor":
        # Lazy import: only paid when escalation actually runs.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        entry = SAM2_VARIANTS[variant]
        ckpt = ensure_checkpoint(variant, allow_download=allow_download)
        dev = device or resolve_sam2_device()
        model = build_sam2(entry.config_name, str(ckpt), device=dev)
        return cls(SAM2ImagePredictor(model))

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def segment(self, box_xyxy, positive_points, negative_points):
        pts = list(positive_points) + list(negative_points)
        labels = [1] * len(positive_points) + [0] * len(negative_points)
        masks, ious, _ = self._predictor.predict(
            box=np.array(box_xyxy, dtype=np.float32),
            point_coords=np.array(pts, dtype=np.float32) if pts else None,
            point_labels=np.array(labels, dtype=np.int32) if pts else None,
            multimask_output=True,
        )
        best = int(np.argmax(ious))
        return masks[best].astype(bool), float(ious[best])
