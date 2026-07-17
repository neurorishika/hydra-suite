"""Background-subtraction detection stage.

Mirrors the load/run/run_batch shape of the other stages. Unlike OBB there is
no model file: the "model" is a BackgroundModel primed from the video itself.

bg-sub is strictly sequential -- BackgroundModel carries cross-frame state
(running-max lightest background, EMA adaptive background, convergence latch).
Pipeline drives a single in-order consumer, so this is safe, but random access
is NOT: load_frame must be served from cache only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from hydra_suite.core.background.measure import BackgroundMeasurer, corners_from_ellipse
from hydra_suite.core.background.model import BackgroundModel
from hydra_suite.utils.image_processing import apply_image_adjustments

from ..config import BgSubConfig
from ..result import DETECTION_ID_STRIDE, OBBResult
from ..runtime import RuntimeContext

logger = logging.getLogger(__name__)


@dataclass
class BgSubModel:
    bg_model: BackgroundModel
    measurer: BackgroundMeasurer

    def close(self) -> None:
        # BackgroundModel holds only numpy/CuPy arrays; nothing to release.
        pass


def load_bgsub_model(
    config: BgSubConfig,
    runtime: RuntimeContext,
    video_path: str | None = None,
) -> BgSubModel:
    """Construct and prime the background model.

    Priming is this stage's equivalent of loading model weights.
    """
    bg_model = BackgroundModel(config.params)
    bg_model.configure_runtime(runtime)
    if video_path:
        cap = cv2.VideoCapture(video_path)
        try:
            bg_model.prime_background(cap)
        finally:
            cap.release()
    return BgSubModel(bg_model=bg_model, measurer=BackgroundMeasurer(config.params))


def _empty_result(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((0, 2), np.float32),
        angles=np.zeros((0,), np.float32),
        sizes=np.zeros((0,), np.float32),
        shapes=np.zeros((0, 2), np.float32),
        confidences=np.zeros((0,), np.float32),
        corners=np.zeros((0, 4, 2), np.float32),
        detection_ids=np.zeros((0,), np.int64),
    )


def _to_gray(frame: np.ndarray, config: BgSubConfig, use_gpu: bool) -> np.ndarray:
    p = config.params
    resize_f = float(p.get("RESIZE_FACTOR", 1.0) or 1.0)
    if resize_f < 1.0:
        frame = cv2.resize(
            frame, (0, 0), fx=resize_f, fy=resize_f, interpolation=cv2.INTER_AREA
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return apply_image_adjustments(
        gray, p["BRIGHTNESS"], p["CONTRAST"], p["GAMMA"], use_gpu
    )


def run_bgsub(
    frame: np.ndarray,
    frame_idx: int,
    model: BgSubModel,
    config: BgSubConfig,
    runtime: RuntimeContext,
    roi_mask: np.ndarray | None = None,
) -> OBBResult:
    """Detect on one frame. Frames MUST arrive in order."""
    gray = _to_gray(frame, config, model.bg_model.use_gpu)

    background = model.bg_model.update_and_get_background(gray, roi_mask)
    if background is None:
        return _empty_result(frame_idx)  # first frame: model has no history yet

    fg_mask = model.bg_model.generate_foreground_mask(gray, background)
    if roi_mask is not None:
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

    if config.enable_conservative_split:
        fg_mask = model.measurer.apply_conservative_split(fg_mask, gray, background)

    meas, sizes, shapes, confidences = model.measurer.detect_objects(fg_mask, frame_idx)
    if not meas:
        return _empty_result(frame_idx)

    centroids = np.array([[m[0], m[1]] for m in meas], np.float32)
    angles = np.array([m[2] for m in meas], np.float32)
    sizes_arr = np.array(sizes, np.float32)
    shapes_arr = np.array(shapes, np.float32)

    corners = np.stack(
        [
            corners_from_ellipse(
                float(centroids[i][0]),
                float(centroids[i][1]),
                float(_major_from_shape(shapes_arr[i])),
                float(_minor_from_shape(shapes_arr[i])),
                float(angles[i]),
            )
            for i in range(len(meas))
        ]
    ).astype(np.float32)

    detection_ids = np.array(
        [frame_idx * DETECTION_ID_STRIDE + i for i in range(len(meas))], np.int64
    )

    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=angles,
        sizes=sizes_arr,
        shapes=shapes_arr,
        confidences=np.array(confidences, np.float32),
        corners=corners,
        detection_ids=detection_ids,
    )


def _major_from_shape(shape: np.ndarray) -> float:
    """Recover the ellipse major axis from (ellipse_area, aspect_ratio).

    detect_objects stores area = pi * (major/2) * (minor/2) and
    aspect = major/minor, so major = sqrt(4 * area * aspect / pi).
    """
    area, aspect = float(shape[0]), float(shape[1])
    if aspect <= 0 or area <= 0:
        return 0.0
    return float(np.sqrt(4.0 * area * aspect / np.pi))


def _minor_from_shape(shape: np.ndarray) -> float:
    area, aspect = float(shape[0]), float(shape[1])
    if aspect <= 0 or area <= 0:
        return 0.0
    return float(np.sqrt(4.0 * area / (np.pi * aspect)))


def run_bgsub_batch(
    frames: list[np.ndarray],
    frame_indices: list[int],
    model: BgSubModel,
    config: BgSubConfig,
    runtime: RuntimeContext,
    roi_mask: np.ndarray | None = None,
) -> list[OBBResult]:
    """Detect a window of frames.

    Sequential by necessity: each frame mutates the background model. The
    "batch" is a window boundary for the cache/pipeline, not vectorisation.
    Frames MUST be in ascending order; random access must be served from
    cache only, never routed through this function out of order.
    """
    return [
        run_bgsub(frame, idx, model, config, runtime, roi_mask=roi_mask)
        for frame, idx in zip(frames, frame_indices)
    ]
