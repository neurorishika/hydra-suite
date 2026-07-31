from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import AprilTagConfig
from ..result import AprilTagResult, OBBResult


@dataclass
class AprilTagModel:
    detector: Any | None  # lab-fork apriltag detector, or None when disabled
    config: AprilTagConfig

    def close(self) -> None:
        pass


def load_apriltag_model(config: AprilTagConfig) -> AprilTagModel:
    if not config.enabled:
        return AprilTagModel(detector=None, config=config)

    # The lab apriltag fork (Social-Evolution-and-Behavior/apriltag) is mandatory
    # for AprilTag identity: only it exposes the tag36ARTag family used by the
    # lab tags. ``_get_apriltag`` imports the module and validates that the
    # required family is present, raising a clear, actionable ImportError
    # otherwise. We intentionally do NOT swallow that error — AprilTag enabled
    # without the fork must fail loudly rather than silently disable detection
    # (matching legacy ``AprilTagDetector`` behaviour).
    from hydra_suite.core.identity.classification.apriltag import _get_apriltag

    at = _get_apriltag()
    detector = at.apriltag(
        family=config.tag_family,
        threads=config.threads,
        maxhamming=config.max_hamming,
        decimate=config.decimate,
        blur=config.blur,
        refine_edges=1 if config.refine_edges else 0,
        decode_sharpening=config.decode_sharpening,
    )
    return AprilTagModel(detector=detector, config=config)


def run_apriltag(
    cpu_crops: list[np.ndarray],
    obb_result: OBBResult,
    model: AprilTagModel,
    config: AprilTagConfig,
) -> AprilTagResult:
    """Detect AprilTags in each AABB crop. No I/O, no mode branching.

    AprilTag always runs on CPU regardless of runtime path. Uses the lab
    apriltag fork detector, whose ``detect`` returns a list of dicts keyed by
    ``id`` / ``center`` / ``hamming`` and corners under ``lb-rb-rt-lt`` (or
    ``corners`` for some bindings).
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
        # Grayscale first, then unsharp + contrast (matches legacy preprocessing).
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        enhanced = _preprocess_crop(gray, config)
        detections = model.detector.detect(enhanced)
        for det in detections:
            tag_id = int(det["id"])
            if config.max_tag_id is not None and tag_id > config.max_tag_id:
                continue
            # Corner key varies between bindings: the lab fork uses
            # ``lb-rb-rt-lt``; some builds use ``corners``.
            raw_corners = det.get("lb-rb-rt-lt")
            if raw_corners is None:
                raw_corners = det.get("corners")
            if raw_corners is None:
                continue
            crn = np.asarray(raw_corners, dtype=np.float32)
            if crn.ndim != 2 or crn.shape[0] != 4:
                continue
            tag_ids.append(tag_id)
            det_indices.append(det_idx)
            centers.append(np.asarray(det["center"], dtype=np.float32))
            corners_list.append(crn)

    if not tag_ids:
        return empty

    return AprilTagResult(
        tag_ids=tag_ids,
        det_indices=det_indices,
        centers=np.stack(centers, axis=0),
        corners=np.stack(corners_list, axis=0),
    )


def _preprocess_crop(gray: np.ndarray, config: AprilTagConfig) -> np.ndarray:
    """Apply unsharp mask and contrast enhancement to a grayscale crop.

    Mirrors legacy ``_unsharp_mask`` (``addWeighted(img, 1+amount, blur,
    -amount, 0)``) followed by ``_contrast_enhance``
    (``mean + factor * (x - mean)``).
    """
    ksize = (config.unsharp_kernel[0] | 1, config.unsharp_kernel[1] | 1)
    blurred = cv2.GaussianBlur(gray, ksize, config.unsharp_sigma)
    sharpened = cv2.addWeighted(
        gray, 1.0 + config.unsharp_amount, blurred, -config.unsharp_amount, 0
    )
    mean = float(np.mean(sharpened))
    enhanced = np.clip(
        mean + config.contrast_factor * (sharpened.astype(np.float32) - mean), 0, 255
    )
    return enhanced.astype(np.uint8)
