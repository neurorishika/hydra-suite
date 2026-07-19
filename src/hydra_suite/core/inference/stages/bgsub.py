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
from ..result import OBBResult
from ..runtime import RuntimeContext

logger = logging.getLogger(__name__)


@dataclass
class BgSubModel:
    bg_model: BackgroundModel
    measurer: BackgroundMeasurer
    roi_cache_key: tuple | None = None
    roi_cache: np.ndarray | None = None
    # Last computed mask/background, for the SHOW_FG / SHOW_BG preview overlays.
    # Safe to stash here: BgSubModel is already stateful and strictly sequential,
    # so "last" is unambiguously the frame the caller just submitted. Realtime
    # only -- run_realtime reads these into FrameResult; batch ignores them.
    last_fg_mask: np.ndarray | None = None
    last_bg_u8: np.ndarray | None = None

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
        detection_ids=OBBResult.make_detection_ids(frame_idx, 0),
        class_ids=np.zeros(0, dtype=np.int64),
    )


def _to_gray(frame: np.ndarray, config: BgSubConfig, use_gpu: bool) -> np.ndarray:
    """Grayscale + brightness/contrast/gamma. Deliberately does NOT resize.

    See run_bgsub's docstring: RESIZE_FACTOR is applied by the *caller* on the
    realtime path, so applying it here too would scale every centroid by
    resize_f**2 with no error.
    """
    p = config.params
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Defaults are the identity transform and match cli_config.py:783-785, so an
    # under-specified param dict is a no-op here rather than a KeyError.
    return apply_image_adjustments(
        gray,
        p.get("BRIGHTNESS", 0),
        p.get("CONTRAST", 1.0),
        p.get("GAMMA", 1.0),
        use_gpu,
    )


def _resolve_roi_mask(
    model: BgSubModel, roi_mask: np.ndarray | None, target_shape: tuple
) -> np.ndarray | None:
    """Resize the ROI mask to match whatever shape the frame actually is.

    Still load-bearing even though the stage no longer resizes the frame itself
    (Task 10b), because the two entry points differ: realtime callers hand in a
    mask already in resized space (worker.py:2088-2090 sizes it off the
    post-resize frame), making this a shape-equality no-op, while run_bgsub_batch
    scales the frame but receives a full-resolution mask, where this does the
    real work. Keying off `gray.shape` rather than RESIZE_FACTOR is what makes
    one resolver serve both.

    Cached because ROI geometry is static for a run — resampling the same binary
    mask every frame is pure waste. Mirrors worker.py::_resolve_resized_roi_mask.
    """
    if roi_mask is None:
        return None
    h, w = int(target_shape[0]), int(target_shape[1])
    if roi_mask.shape[:2] == (h, w):
        return roi_mask
    key = (id(roi_mask), h, w)
    if model.roi_cache_key == key and model.roi_cache is not None:
        return model.roi_cache
    resized = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    model.roi_cache_key = key
    model.roi_cache = resized
    return resized


def run_bgsub(
    frame: np.ndarray,
    frame_idx: int,
    model: BgSubModel,
    config: BgSubConfig,
    runtime: RuntimeContext,
    roi_mask: np.ndarray | None = None,
) -> OBBResult:
    """Detect on one frame. Frames MUST arrive in order.

    RESIZE CONTRACT -- the asymmetry with run_bgsub_batch is DELIBERATE, please
    do not "fix" it:

        `frame` MUST ALREADY be scaled by RESIZE_FACTOR. This stage does not
        resize it.

    This is the realtime entry point, and the realtime caller (worker.py:2072)
    already scales the frame for *every* detection method before dispatching,
    then works in that resized coordinate space for the rest of its loop --
    worker.py:2088-2090 sizes the ROI mask from the post-resize `frame.shape`.
    So the worker cannot stop pre-resizing, and this stage must not resize
    again: doing both would scale every centroid by resize_f**2, silently.

    run_bgsub_batch takes raw frames (straight off FrameSource, which applies no
    resize) and therefore scales them itself before delegating here. The two
    paths MUST agree, because RESIZE_FACTOR is in `_BGSUB_KEY_PARAMS` -- if they
    disagree, the detection cache is a lie.
    """
    gray = _to_gray(frame, config, model.bg_model.use_gpu)
    roi_resized = _resolve_roi_mask(model, roi_mask, gray.shape)
    # Before background update, matching worker.py:2286: stabilization feeds the
    # background model, it does not correct its output.
    gray = model.bg_model.apply_lighting_stabilization(gray, roi_resized)

    background = model.bg_model.update_and_get_background(gray, roi_resized)
    if background is None:
        model.last_fg_mask = None
        model.last_bg_u8 = None
        return _empty_result(frame_idx)  # first frame: model has no history yet

    fg_mask = model.bg_model.generate_foreground_mask(gray, background)
    if roi_resized is not None:
        fg_mask = cv2.bitwise_and(fg_mask, roi_resized)

    if config.enable_conservative_split:
        fg_mask = model.measurer.apply_conservative_split(fg_mask, gray, background)

    # Stash the exact mask detection ran on, so the SHOW_FG overlay shows what
    # the detector saw rather than an approximation of it.
    model.last_fg_mask = fg_mask
    model.last_bg_u8 = background

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

    detection_ids = OBBResult.make_detection_ids(frame_idx, len(meas))

    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=angles,
        sizes=sizes_arr,
        shapes=shapes_arr,
        confidences=np.array(confidences, np.float32),
        corners=corners,
        detection_ids=detection_ids,
        # bg-sub has no class head -- all detections are the single generic
        # "object" class (0).
        class_ids=np.zeros(len(meas), dtype=np.int64),
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

    RESIZE CONTRACT -- the asymmetry with run_bgsub is DELIBERATE, please do not
    "fix" it:

        `frames` arrive RAW and this function scales them by RESIZE_FACTOR.

    Batch frames come straight off FrameSource, which applies no resize (neither
    do pipeline.py or runner.py). run_bgsub's realtime caller pre-scales instead,
    so the scaling has to happen here to make the two paths agree -- mandatory,
    since RESIZE_FACTOR is in `_BGSUB_KEY_PARAMS` and a batch/realtime
    disagreement would make the detection cache a lie. INTER_AREA matches
    worker.py::_resize_tracking_frame's policy for non-YOLO methods.
    """
    resize_f = float(config.params.get("RESIZE_FACTOR", 1.0) or 1.0)
    if resize_f < 1.0:
        frames = [
            cv2.resize(
                f, (0, 0), fx=resize_f, fy=resize_f, interpolation=cv2.INTER_AREA
            )
            for f in frames
        ]
    return [
        run_bgsub(frame, idx, model, config, runtime, roi_mask=roi_mask)
        for frame, idx in zip(frames, frame_indices)
    ]
