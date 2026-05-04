from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import AprilTagConfig
from ..result import AprilTagResult, OBBResult


@dataclass
class AprilTagModel:
    detector: Any | None  # apriltag.Detector or None when disabled
    config: AprilTagConfig

    def close(self) -> None:
        pass


def load_apriltag_model(config: AprilTagConfig) -> AprilTagModel:
    if not config.enabled:
        return AprilTagModel(detector=None, config=config)
    try:
        import apriltag

        options = apriltag.DetectorOptions(
            families=config.tag_family,
            nthreads=config.threads,
            quad_decimate=config.decimate,
            quad_sigma=config.blur,
            refine_edges=int(config.refine_edges),
            decode_sharpening=config.decode_sharpening,
            max_hamming=config.max_hamming,
        )
        detector = apriltag.Detector(options)
    except ImportError:
        detector = None
    return AprilTagModel(detector=detector, config=config)


def run_apriltag(
    cpu_crops: list[np.ndarray],
    obb_result: OBBResult,
    model: AprilTagModel,
    config: AprilTagConfig,
) -> AprilTagResult:
    """Detect AprilTags in each AABB crop. No I/O, no mode branching.

    AprilTag always runs on CPU regardless of runtime path.
    """
    empty = AprilTagResult(
        tag_ids=[],
        det_indices=[],
        centers=np.zeros((0, 2), dtype=np.float32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )

    if not config.enabled or model.detector is None or not cpu_crops:
        return empty

    tag_ids: list[int] = []
    det_indices: list[int] = []
    centers: list[np.ndarray] = []
    corners_list: list[np.ndarray] = []

    for det_idx, crop in enumerate(cpu_crops):
        if crop.size == 0:
            continue
        preprocessed = _preprocess_crop(crop, config)
        if preprocessed.ndim == 3:
            gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        else:
            gray = preprocessed
        detections = model.detector.detect(gray)
        for det in detections:
            tag_id = int(det.tag_id)
            if config.max_tag_id is not None and tag_id > config.max_tag_id:
                continue
            tag_ids.append(tag_id)
            det_indices.append(det_idx)
            centers.append(np.array(det.center, dtype=np.float32))
            corners_list.append(np.array(det.corners, dtype=np.float32))

    if not tag_ids:
        return empty

    return AprilTagResult(
        tag_ids=tag_ids,
        det_indices=det_indices,
        centers=np.stack(centers, axis=0),
        corners=np.stack(corners_list, axis=0),
    )


def _preprocess_crop(crop: np.ndarray, config: AprilTagConfig) -> np.ndarray:
    """Apply unsharp mask and contrast enhancement before detection."""
    ksize = (config.unsharp_kernel[0] | 1, config.unsharp_kernel[1] | 1)
    blurred = cv2.GaussianBlur(crop.astype(np.float32), ksize, config.unsharp_sigma)
    sharpened = crop.astype(np.float32) + config.unsharp_amount * (
        crop.astype(np.float32) - blurred
    )
    sharpened = np.clip(sharpened * config.contrast_factor, 0, 255).astype(np.uint8)
    return sharpened
