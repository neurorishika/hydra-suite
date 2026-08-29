"""
Dataset generation utilities for active learning.
Identifies challenging frames and exports them for annotation.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.core.inference.cache.reuse import (
    get_or_compute_raw,
    open_raw_detection_cache_reader,
)
from hydra_suite.core.inference.stages.filtering import filter_for_source
from hydra_suite.data.al.escalation import (
    LabelRecord,
    achievable_levels,
    records_from_obb_result,
)
from hydra_suite.data.al.export import ExportedFrame, export_al_dataset
from hydra_suite.utils.geometry_levels import GeometryLevel

logger = logging.getLogger(__name__)

_TASK_LEVELS = {
    "segment": GeometryLevel.POLYGON,
    "obb": GeometryLevel.OBB,
    "detect": GeometryLevel.AABB,
}


def resolve_native_level(params) -> GeometryLevel:
    """The geometry level the configured detection source can actually produce.

    Never claims a level the model did not compute: a rotated quad is OBB, not
    a polygon. bg-sub produces true foreground contours, so it reaches POLYGON.
    """
    method = str(params.get("DETECTION_METHOD", "background_subtraction")).lower()
    if method == "background_subtraction":
        return GeometryLevel.POLYGON
    if method != "yolo_obb":
        return GeometryLevel.OBB

    return _TASK_LEVELS.get(resolve_detection_task(params), GeometryLevel.OBB)


def resolve_detection_task(params) -> str:
    """The YOLO head the export detection pass will actually run.

    Single authority so `resolve_native_level`, `_init_detection_runner` and
    the exported provenance can never disagree about which task ran. The
    sequential stage-2 key is ``YOLO_SEQ_STAGE2_TASK`` -- the one
    ``core/inference/config.build_inference_config_from_params`` reads;
    ``YOLO_OBB_STAGE2_TASK`` was a key nothing ever wrote, so sequential rounds
    silently resolved to the direct-mode default.
    """
    mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    if mode == "sequential":
        raw = params.get(
            "YOLO_SEQ_STAGE2_TASK", params.get("YOLO_OBB_STAGE2_TASK", "obb")
        )
    else:
        raw = params.get("YOLO_OBB_DIRECT_TASK", "obb")
    task = str(raw or "obb").strip().lower()
    return task if task in _TASK_LEVELS else "obb"


def resolve_detection_model_path(params) -> str:
    """The checkpoint the export detection pass will actually load.

    Sequential mode runs the *crop* OBB model as its geometry stage, not the
    direct model -- stamping the direct path into provenance made a sequential
    round claim a model it never ran.
    """
    mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    if mode == "sequential":
        return str(
            params.get("YOLO_CROP_OBB_MODEL_PATH", "")
            or params.get("YOLO_MODEL_PATH", "")
            or ""
        )
    return str(
        params.get(
            "YOLO_OBB_DIRECT_MODEL_PATH",
            params.get("YOLO_MODEL_PATH", ""),
        )
        or ""
    )


def effective_acquisition_weights(params):
    """The acquisition weights that actually rank frames for `params`.

    The preset is only the starting point: every channel whose ``METRIC_*``
    toggle is off is zeroed, the uncertainty channel is zeroed for bg-sub
    (which reports NaN confidences), and the survivors are renormalized. This
    is the single derivation -- `FrameQualityScorer` consumes it, and the
    exporter stamps it into provenance, so a round's recorded weights are by
    construction the ones that produced it.
    """
    from hydra_suite.data.al.acquisition import PRESETS, AcquisitionWeights

    method = str(params.get("DETECTION_METHOD", "")).strip().lower()
    confidence_available = method != "background_subtraction"

    enabled = {
        "uncertainty": bool(params.get("METRIC_LOW_CONFIDENCE", True))
        and confidence_available,
        "count": bool(params.get("METRIC_COUNT_MISMATCH", True)),
        "assignment": bool(params.get("METRIC_HIGH_ASSIGNMENT_COST", True)),
        "track_loss": bool(params.get("METRIC_TRACK_LOSS", True)),
        "position_uncertainty": bool(params.get("METRIC_HIGH_UNCERTAINTY", False)),
        "crowd": bool(params.get("METRIC_CROWDING", True)),
        "fragmentation": bool(params.get("METRIC_FRAGMENTED_DETECTIONS", True)),
    }

    base = PRESETS.get(
        params.get("DATASET_AL_PRESET", "tracker_default"), PRESETS["tracker_default"]
    )
    weights = AcquisitionWeights(
        uncertainty=base.uncertainty if enabled["uncertainty"] else 0.0,
        nms_instability=0.0,
        count=base.count if enabled["count"] else 0.0,
        crowd=base.crowd if enabled["crowd"] else 0.0,
        edge=base.edge,
        fragmentation=base.fragmentation if enabled["fragmentation"] else 0.0,
        assignment=base.assignment if enabled["assignment"] else 0.0,
        track_loss=base.track_loss if enabled["track_loss"] else 0.0,
        position_uncertainty=(
            base.position_uncertainty if enabled["position_uncertainty"] else 0.0
        ),
    ).normalized()
    return weights, enabled


def _effective_acquisition_weights(params) -> dict:
    """`effective_acquisition_weights` as a JSON-serializable dict."""
    from dataclasses import asdict

    weights, _enabled = effective_acquisition_weights(params)
    return {k: float(v) for k, v in asdict(weights).items()}


class FrameQualityScorer:
    """Tracker-side adapter that produces ALSignals and selects worst frames.

    Public API (`score_frame`, `get_worst_frames`) is preserved for callers; the
    underlying ranking now lives in `hydra_suite.data.al.acquisition`.
    """

    def __init__(self, params, frame_shape: tuple[int, int] | None = None):
        self.params = params
        self.frame_signals: dict = {}
        self.max_targets = params.get("MAX_TARGETS", 4)
        self.conf_threshold = params.get("DATASET_CONF_THRESHOLD", 0.5)

        # -------------------------------------------------------------------
        # COORDINATE SPACE: this scorer operates entirely in RESIZE_FACTOR
        # WORKING space, because that is the space `detection_data["obb_corners"]`
        # arrives in -- the detection cache is written from the resized
        # detection frame (see core/inference), never from the original frame.
        #
        # Callers hand us ORIGINAL-space quantities: `frame_shape` comes from
        # cv2.CAP_PROP_FRAME_{WIDTH,HEIGHT}, and REFERENCE_BODY_SIZE is an
        # original-space length (core/canonicalization/geometry.py:83,
        # core/tracking/worker.py, core/assigners/hungarian.py all multiply it
        # by RESIZE_FACTOR to reach working space). Both are converted ONCE
        # here. Comparing an original-space reference length against
        # working-space corners made `fragmentation` -- the largest weight in
        # `tracker_default` -- a function of the resize knob rather than of the
        # scene, and `edge` had the mirror defect. This is the second time this
        # class of bug has appeared; do not "simplify" these two lines away.
        # -------------------------------------------------------------------
        resize_factor = float(params.get("RESIZE_FACTOR", 1.0) or 1.0)
        if resize_factor <= 0:
            resize_factor = 1.0
        self.resize_factor = resize_factor

        # Original-space value, kept under its historical public name.
        self.reference_body_size = max(
            float(params.get("REFERENCE_BODY_SIZE", 20.0)), 1.0
        )
        # Working-space value: what every signal below is actually scored with.
        self.reference_body_size_working = max(
            self.reference_body_size * resize_factor, 1.0
        )
        # (H, W) of the coordinate space `obb_corners` live in. Required for a
        # meaningful edge score: passing (1, 1) with pixel-space corners made
        # score_crowd return values in the hundreds.
        self.frame_shape = (
            (
                max(int(round(float(frame_shape[0]) * resize_factor)), 1),
                max(int(round(float(frame_shape[1]) * resize_factor)), 1),
            )
            if frame_shape
            else None
        )

        # Public legacy boolean flags (kept for backward compat).
        self.use_confidence = bool(params.get("METRIC_LOW_CONFIDENCE", True))
        self.use_count_mismatch = bool(params.get("METRIC_COUNT_MISMATCH", True))
        self.use_assignment_cost = bool(params.get("METRIC_HIGH_ASSIGNMENT_COST", True))
        self.use_track_loss = bool(params.get("METRIC_TRACK_LOSS", True))
        self.use_uncertainty = bool(params.get("METRIC_HIGH_UNCERTAINTY", False))
        self.use_fragmented_detections = bool(
            params.get("METRIC_FRAGMENTED_DETECTIONS", True)
        )
        self.use_crowd = bool(params.get("METRIC_CROWDING", True))

        # Background subtraction sets every detection confidence to NaN
        # (core/background/measure.py), so score_uncertainty always returns
        # 0.0 on that path. Its weight must be zeroed explicitly and the
        # remaining weights renormalized -- otherwise the dead uncertainty
        # weight still counts in the denominator and dilutes every other
        # channel by its share for no reason.
        method = str(params.get("DETECTION_METHOD", "")).strip().lower()
        self._confidence_available = method != "background_subtraction"

        self._weights, self._enabled = effective_acquisition_weights(params)

    def score_frame(self, frame_id, detection_data=None, tracking_data=None):
        from hydra_suite.data.al.signals import (
            ALSignals,
            score_count_deviation,
            score_crowd,
            score_fragmentation,
            score_uncertainty,
        )

        detection_data = detection_data or {}
        tracking_data = tracking_data or {}

        # ------------------------------------------------------------------
        # New pipeline: build ALSignals and store for get_worst_frames.
        # ------------------------------------------------------------------
        confidences = detection_data.get("confidences") or []
        mean_conf = float(np.mean(confidences)) if confidences else float("nan")
        uncertainty = score_uncertainty(confidences, conf_floor=self.conf_threshold)

        n_dets = int(detection_data.get("count", len(confidences)))
        count_dev = score_count_deviation(n_dets, self.max_targets)

        obb_corners = self._extract_obb_corners(detection_data)
        if obb_corners and self.frame_shape is not None:
            crowd, edge = score_crowd(obb_corners, frame_shape=self.frame_shape)
        elif obb_corners:
            # No frame shape available: crowd is shape-independent, edge is not
            # -- computing it against a fake (1, 1) shape is the bug this fixes.
            crowd, _ = score_crowd(obb_corners, frame_shape=(1, 1))
            edge = 0.0
        else:
            crowd, edge = 0.0, 0.0
        fragmentation = score_fragmentation(
            obb_corners,
            reference_major_axis=self.reference_body_size_working * 2.2,
        )

        extras: dict[str, float] = {}
        ac = tracking_data.get("assignment_confidences") or []
        if ac:
            extras["assignment"] = max(0.0, 1.0 - float(np.mean(ac)))
        elif tracking_data.get("assignment_costs"):
            costs = tracking_data["assignment_costs"]
            extras["assignment"] = float(min(np.mean(costs) / 50.0, 1.0))

        lost = int(tracking_data.get("lost_tracks", 0))
        if lost > 0:
            extras["track_loss"] = float(min(lost / max(self.max_targets, 1), 1.0))

        unc = tracking_data.get("uncertainties") or []
        if unc:
            extras["position_uncertainty"] = float(min(np.mean(unc) / 50.0, 1.0))

        signal = ALSignals(
            frame_id=int(frame_id),
            n_detections=n_dets,
            mean_confidence=mean_conf,
            uncertainty_score=uncertainty,
            count_deviation=count_dev,
            crowd_score=crowd,
            fragmentation_score=fragmentation,
            edge_score=edge,
            extras=extras,
        )
        self.frame_signals[int(frame_id)] = signal

        from hydra_suite.data.al.acquisition import _composite_score

        return float(_composite_score([signal], self._weights)[0])

    def explain_scores(self) -> dict:
        """Per-channel maxima, for reporting why a selection came back empty."""
        from hydra_suite.data.al.acquisition import explain

        return explain(list(self.frame_signals.values()), self._weights)

    def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
        from hydra_suite.data.al.acquisition import select

        signals = list(self.frame_signals.values())
        rng = np.random.default_rng() if probabilistic else None
        return select(
            signals,
            weights=self._weights,
            k=int(max_frames),
            diversity_window=int(diversity_window),
            probabilistic=bool(probabilistic),
            rng=rng,
            min_score=float(self.params.get("DATASET_MIN_SELECTION_SCORE", 0.0)),
        )

    def _extract_obb_corners(self, detection_data):
        corners = detection_data.get("obb_corners") or []
        out: list[np.ndarray] = []
        for c in corners:
            if c is None:
                continue
            arr = np.asarray(c, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] >= 3:
                out.append(arr)
        return out


def _init_detection_runner(params, video_path):
    """Build a detection-only InferenceRunner for dataset label extraction.

    Unlike the legacy version this supports every detection source, not just
    `yolo_obb`: returning None for bg-sub meant every exported label was a
    fabricated reference-size box.

    `video_path` wires the runner's `cache_dir` to the SAME on-disk
    `.inference_cache_<stem>/` folder tracking already populated, so
    `_detect_records_for_frames` can read the existing detection cache
    instead of always rerunning inference at export time.
    """
    method = str(params.get("DETECTION_METHOD", "background_subtraction")).lower()
    native_level = resolve_native_level(params)
    try:
        from ..core.inference.runner import InferenceRunner
        from ..utils.video_artifacts import build_inference_cache_dir

        cache_dir = build_inference_cache_dir(video_path)

        if method == "background_subtraction":
            # build_inference_config_from_params never builds a BgSubConfig
            # (it only ever wires up `obb=`), so a bgsub-backed InferenceConfig
            # has to be constructed explicitly here, mirroring the live
            # tracking path in core/tracking/worker.py (bgsub branch).
            from ..core.inference.config import (
                BgSubConfig,
                InferenceConfig,
                migrate_runtime_to_tier,
            )

            _compute_runtime = str(params.get("COMPUTE_RUNTIME", "cpu"))
            _raw_tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
            _runtime_tier = (
                _raw_tier
                if _raw_tier in {"cpu", "gpu", "gpu_fast"}
                else migrate_runtime_to_tier({_compute_runtime})
            )
            bgsub_cfg = BgSubConfig.from_params(params)
            if native_level is GeometryLevel.POLYGON:
                bgsub_cfg.emit_native_geometry = True
            cfg = InferenceConfig(
                obb=None,
                bgsub=bgsub_cfg,
                runtime_tier=_runtime_tier,
                detection_batch_size=int(params.get("DETECTION_BATCH_SIZE", 1) or 1),
            )
        else:
            from ..core.inference.config import build_obb_only_config

            mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
            task = resolve_detection_task(params)
            model_path = resolve_detection_model_path(params) or "yolo26s-obb.pt"
            # Sequential mode's geometry stage is the CROP model, and its
            # stage-1 detector plus every YOLO_SEQ_* knob live in keys
            # `build_obb_only_config` does not take. Without these the export
            # pass silently built a sequential config with an empty stage-1
            # model path and dataclass-default stage-2 knobs -- a different
            # detector from the one that produced the tracking being reviewed.
            extra_params = None
            if mode == "sequential":
                extra_params = {
                    k: v for k, v in params.items() if str(k).startswith("YOLO_SEQ_")
                }
                extra_params["YOLO_DETECT_MODEL_PATH"] = params.get(
                    "YOLO_DETECT_MODEL_PATH", ""
                )
                extra_params["YOLO_CROP_OBB_MODEL_PATH"] = model_path
                extra_params["YOLO_SEQ_STAGE2_TASK"] = task
            cfg = build_obb_only_config(
                model_path,
                runtime_tier=str(params.get("RUNTIME_TIER", "") or "") or None,
                confidence_threshold=float(
                    params.get("DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05)
                ),
                iou_threshold=float(params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5)),
                max_targets=max(1, int(params.get("MAX_TARGETS", 8))),
                mode=mode,
                model_task=task,
                emit_native_geometry=(native_level is GeometryLevel.POLYGON),
                extra_params=extra_params,
            )

        runner = InferenceRunner(cfg, cache_dir=cache_dir, video_path=video_path)
        logger.info(
            "Detection runner initialized for dataset export (method=%s, level=%s)",
            method,
            native_level.label,
        )
        return runner
    except Exception as e:
        # Whole-run failure => loud. This used to return None and let the
        # export continue, which was survivable only while a fabricated
        # reference-size box existed as a fallback. That fallback is gone, so
        # `None` now means "write an empty label file for every frame" --
        # fabricated *negative* ground truth, strictly worse than the failure
        # it was hiding.
        raise RuntimeError(
            "Could not initialize the export detection runner "
            f"(method={method}, level={native_level.label}): {e}. "
            "No labels can be produced without it; fix the detection "
            "configuration (model path, runtime) and export again."
        ) from e


def _expand_frame_ids(frame_ids, include_context, total_frames):
    """Expand frame list with +/-1 context frames if requested."""
    frames_to_export = set()
    for frame_id in frame_ids:
        frames_to_export.add(frame_id)
        if include_context:
            if frame_id > 0:
                frames_to_export.add(frame_id - 1)
            if frame_id < total_frames - 1:
                frames_to_export.add(frame_id + 1)
    return sorted(frames_to_export)


def _read_and_resize_frame(cap, frame_id, params, first_frame_shape):
    """Read a video frame, resize for detection, and validate shape.

    Returns (original_frame, detection_frame, updated_first_shape) or None if
    the frame could not be read.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not (ret and frame is not None and frame.size > 0):
        logger.warning(f"Could not read frame {frame_id}, skipping")
        return None

    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    if resize_factor != 1.0 and resize_factor > 0:
        h, w = frame.shape[:2]
        new_w, new_h = int(w * resize_factor), int(h * resize_factor)
        if new_w > 0 and new_h > 0:
            frame_for_detection = cv2.resize(frame, (new_w, new_h))
        else:
            logger.warning(
                f"Invalid resize dimensions for frame {frame_id}, using original"
            )
            frame_for_detection = frame
    else:
        frame_for_detection = frame

    if frame_for_detection.size == 0:
        logger.warning(f"Frame {frame_id} has zero size, skipping")
        return None

    if first_frame_shape is None:
        first_frame_shape = frame_for_detection.shape
    elif frame_for_detection.shape != first_frame_shape:
        h, w = first_frame_shape[:2]
        frame_for_detection = cv2.resize(frame_for_detection, (w, h))
        logger.debug(
            f"Resized frame {frame_id} to match batch dimensions: {first_frame_shape}"
        )

    return frame, frame_for_detection, first_frame_shape


def _csv_scale_back(df, resize_factor, frame_width, frame_height):
    """Determine scale factor to map CSV coordinates back to original space."""
    if not (resize_factor and resize_factor < 1.0):
        return 1.0
    try:
        max_x = df["X"].max()
        max_y = df["Y"].max()
        if (
            max_x <= frame_width * resize_factor * 1.05
            and max_y <= frame_height * resize_factor * 1.05
        ):
            return 1.0 / resize_factor
    except Exception:
        pass
    return 1.0


def _open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return cap


def _frame_is_readable(cap, frame_id) -> bool:
    """Cheap readability probe: seeks + reads, but never retains the frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    return bool(ret and frame is not None and frame.size > 0)


class _LazyFrameImages(Mapping):
    """Original (full-resolution) frames, decoded from `cap` on `__getitem__`.

    Only the frame currently being accessed is ever resident in memory.
    `export_al_dataset`'s authoritative root reads each key exactly once
    (derived roots hardlink instead), so this keeps the export's image
    footprint at O(1) frames instead of O(frame count) -- at default panel
    settings (100 frames x3 for context, on 4K video) the eager dict this
    replaces was ~15 GB resident.
    """

    def __init__(self, cap, params, frame_ids):
        self._cap = cap
        self._params = params
        self._frame_ids = list(frame_ids)

    def __getitem__(self, frame_id):
        read = _read_and_resize_frame(self._cap, frame_id, self._params, None)
        if read is None:
            raise KeyError(frame_id)
        original, _for_detection, _shape = read
        return original

    def __iter__(self):
        return iter(self._frame_ids)

    def __len__(self):
        return len(self._frame_ids)


class _LazyDetectionFrames(Mapping):
    """Detection-resized frames, decoded from `cap` on `__getitem__`.

    `_detect_records_for_frames` reads one batch at a time and discards it
    once detection on that batch completes, so nothing beyond the batch
    currently in flight is resident.
    """

    def __init__(self, cap, params, frame_ids):
        self._cap = cap
        self._params = params
        self._frame_ids = list(frame_ids)

    def __getitem__(self, frame_id):
        read = _read_and_resize_frame(self._cap, frame_id, self._params, None)
        if read is None:
            raise KeyError(frame_id)
        _original, for_detection, _shape = read
        return for_detection

    def __iter__(self):
        return iter(self._frame_ids)

    def __len__(self):
        return len(self._frame_ids)


def _detect_records_for_frames(runner, frames, params, native_level):
    """Run detection over `frames`.

    Returns ``({frame_id: [LabelRecord]}, stats)`` where ``stats`` counts the
    per-frame failures under named keys (currently ``detection_failed``).

    `frames` is a Mapping[frame_id, image] consulted in batches -- only the
    current batch's images are held in memory at once, whether `frames` is a
    plain dict (tests) or a lazy, decode-on-access mapping (production).

    Per-frame failure semantics: a frame whose detection (or whose geometry
    extraction) raises is dropped and *counted*, never silently swallowed.
    Swallowing it would have handed the exporter a frame with zero records,
    which becomes an empty YOLO label file -- i.e. "I could not compute
    geometry" written to disk as "there is no geometry here".
    """
    stats = {"detection_failed": 0}
    if runner is None:
        return {}, stats
    batch_size = _get_detector_batch_size(runner)
    # Detection runs on the RESIZE_FACTOR-scaled frame (`_LazyDetectionFrames`
    # yields `frame_for_detection`, never the original), so raw obb corners
    # come back in resized-frame space. Every downstream consumer -- the
    # strict-label matcher against original-space CSV rows, and label
    # normalization against the original frame's `.shape` -- needs original
    # pixel space. Scale once, here, so nobody downstream has to remember to.
    # Do not remove this: it replaces the scale-back the deleted legacy
    # `_measurements_to_detections` used to do; dropping it silently
    # re-introduces a coordinate-space mismatch whenever RESIZE_FACTOR < 1.
    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    detection_scale_back = (
        1.0 / resize_factor if resize_factor and resize_factor < 1.0 else 1.0
    )
    out: dict[int, list[LabelRecord]] = {}
    frame_ids = sorted(frames)
    # Keep one cache handle for the full export.  Each chunk otherwise creates
    # a new handle, whose first read decompresses every array in detection.npz.
    # This preserves the current cache-miss inference batch shape: only cache
    # reads are coalesced in memory.
    cache_reader = open_raw_detection_cache_reader(runner, runner.cache_dir)
    for start in range(0, len(frame_ids), batch_size):
        chunk = frame_ids[start : start + batch_size]
        images = []
        valid_chunk = []
        for fid in chunk:
            try:
                images.append(frames[fid])
            except KeyError:
                continue
            valid_chunk.append(fid)
        if not images:
            continue
        try:
            # Reuse the on-disk detection cache the tracking pass already
            # built (`runner.cache_dir`), instead of always rerunning
            # inference at export time -- the overwhelming common case is
            # exporting AFTER tracking has covered the whole video, which
            # makes this a pure cache read (zero `detect_batch_raw` calls).
            # `get_or_compute_raw` returns RAW (pre-filter) results, so this
            # must re-apply the SAME `filter_for_source` gate `detect_batch`
            # used to apply, mirroring `InferenceRunner.load_frame` -- an
            # unfiltered raw result includes below-threshold/duplicate/
            # over-max-targets detections `detect_batch` always excluded.
            #
            # `write=False` is mandatory here, not an optimization: this
            # cache_dir is tracking's OWN `.inference_cache_<stem>/`, and a
            # cache MISS (a different model mtime, a SAHI-sliced tracking run
            # whose key this export config cannot reproduce, a partial or
            # subrange tracking pass) would otherwise have each chunk's
            # `DetectionCacheHandle.close()` rewrite that file from this
            # chunk's buffer alone -- destroying the complete detection cache
            # backward/replay tracking depends on. Export borrows the cache; it
            # never owns it. A hit is unaffected (already a pure read), so this
            # keeps 100% of the reuse win and recomputes in memory on a miss.
            results_by_idx = get_or_compute_raw(
                runner,
                runner.cache_dir,
                images,
                list(valid_chunk),
                write=False,
                cache_reader=cache_reader,
            )
            results = [
                filter_for_source(
                    runner.config,
                    results_by_idx[fid],
                    getattr(runner, "_roi_mask", None),
                )[0]
                for fid in valid_chunk
            ]
        except Exception as e:
            logger.warning(
                "Detection failed for batch starting at %s: %s", valid_chunk[0], e
            )
            stats["detection_failed"] += len(valid_chunk)
            continue
        for fid, obb in zip(valid_chunk, results):
            # Geometry extraction is inside the per-frame guard: a missing
            # native contour on one frame must cost that frame, not the round.
            try:
                records = records_from_obb_result(obb, native_level)
            except Exception as e:
                logger.warning("Geometry extraction failed for frame %s: %s", fid, e)
                stats["detection_failed"] += 1
                continue
            if detection_scale_back != 1.0:
                for rec in records:
                    rec.points = (rec.points * detection_scale_back).astype(np.float32)
            out[fid] = records
    return out, stats


_LOST_STATES = {"lost", "interpolated", "predicted"}


def _select_records_for_frame(rows, frame_records, params, scale_back):
    """Pair tracked CSV rows with detector geometry; export only real matches.

    Strict by design. The legacy exporter wrote a fabricated
    `ref*2.2 x ref*0.8` box whenever a row had no nearby detection, and wrote
    labels for `lost`/interpolated rows the detector never saw. Since AL
    selects frames precisely where tracking struggled, both behaviours injected
    wrong boxes exactly where the model was weakest. Do not restore a
    fabricated-geometry fallback here: a row with no real detection must be
    dropped and counted, never invented.

    Matching is mutual-exclusion (one row <-> one detection) via the Hungarian
    algorithm, gated by a radius scaled to REFERENCE_BODY_SIZE rather than the
    legacy hardcoded 50 px, so it neither reaches a neighbouring animal for
    small species nor fails to reach the correct one for large species.
    """
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    drops = {"lost": 0, "unmatched": 0}
    if rows is None or len(rows) == 0 or not frame_records:
        if rows is not None:
            for _, row in rows.iterrows():
                state = str(row.get("State", "")).strip().lower()
                if state in _LOST_STATES:
                    drops["lost"] += 1
                else:
                    drops["unmatched"] += 1
        return [], drops

    live_rows = []
    for _, row in rows.iterrows():
        state = str(row.get("State", "")).strip().lower()
        if state in _LOST_STATES:
            drops["lost"] += 1
            continue
        if pd.isna(row["X"]) or pd.isna(row["Y"]):
            drops["unmatched"] += 1
            continue
        live_rows.append((float(row["X"]) * scale_back, float(row["Y"]) * scale_back))

    if not live_rows:
        return [], drops

    reference = max(float(params.get("REFERENCE_BODY_SIZE", 20.0)), 1.0)
    max_distance = reference * 2.2

    centers = np.array(
        [rec.points.mean(axis=0) for rec in frame_records], dtype=np.float64
    )
    targets = np.array(live_rows, dtype=np.float64)
    cost = np.linalg.norm(targets[:, None, :] - centers[None, :, :], axis=2)

    # linear_sum_assignment minimizes TOTAL cost, not "prefer in-radius pairs".
    # Solving on the raw distance matrix and filtering afterwards can leave a
    # row unmatched even though a fully in-radius assignment existed, because
    # the optimizer chose a globally-cheaper arrangement that used a different
    # (also in-radius) detection for that row's rightful match, stranding it
    # on an out-of-radius leftover. Penalize out-of-radius pairs with a large
    # sentinel *before* solving so the optimizer maximizes in-radius pairings
    # first, then apply the real `max_distance` gate against the ORIGINAL
    # (unpenalized) cost matrix. Do not use np.inf: scipy raises on infeasible
    # (all-inf-row/col) matrices.
    solve_cost = np.where(cost <= max_distance, cost, max_distance * 1e6)
    row_idx, col_idx = linear_sum_assignment(solve_cost)
    matched_detections: list[int] = []
    matched_rows = set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= max_distance:
            matched_detections.append(int(c))
            matched_rows.add(int(r))

    drops["unmatched"] += len(live_rows) - len(matched_rows)
    return [frame_records[i] for i in sorted(matched_detections)], drops


def export_dataset(
    video_path,
    csv_path,
    frame_ids,
    output_dir,
    dataset_name,
    class_name,
    params,
    include_context: bool = True,
    export_levels=None,
    class_names=None,
):
    """Export selected frames and labels as an escalated AL dataset.

    Returns the manifest dict from `export_al_dataset` (previously a directory
    path string).
    """
    from datetime import datetime

    import pandas as pd

    native_level = resolve_native_level(params)
    allowed = achievable_levels(native_level)
    levels = list(export_levels) if export_levels else list(allowed)
    unsupported = [lvl for lvl in levels if lvl not in allowed]
    if unsupported:
        raise ValueError(
            f"requested levels {[lvl.label for lvl in unsupported]} exceed the "
            f"native level {native_level.label!r} of the configured detector"
        )

    resolved_class_names = (
        list(class_names) if class_names else [class_name or "object"]
    )

    cap = _open_video(video_path)
    # Bound before the try so the `finally` can always read it: the runner is
    # built INSIDE the try because `_init_detection_runner` raises on failure
    # (it used to return None), and building it above leaked `cap` on every
    # bad model path or runtime.
    runner = None
    try:
        runner = _init_detection_runner(params, video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        df = pd.read_csv(csv_path)
        selected = {int(f) for f in frame_ids}
        frames_to_export = _expand_frame_ids(frame_ids, include_context, total_frames)

        # Readability is probed and immediately discarded -- no frame data is
        # retained here. The two lazy mappings below re-decode each valid
        # frame on demand, so at most one batch of detection frames (or one
        # authoritative-root image) is ever resident at once, rather than
        # every export frame at full resolution simultaneously.
        valid_frame_ids = [
            fid for fid in frames_to_export if _frame_is_readable(cap, fid)
        ]

        images = _LazyFrameImages(cap, params, valid_frame_ids)
        detection_frames = _LazyDetectionFrames(cap, params, valid_frame_ids)

        records_by_frame, detection_stats = _detect_records_for_frames(
            runner, detection_frames, params, native_level
        )

        resize_factor = params.get("RESIZE_FACTOR", 1.0)
        scale_back = _csv_scale_back(df, resize_factor, frame_width, frame_height)
        rows_by_frame = {int(fid): sub for fid, sub in df.groupby("FrameID")}

        exported: list[ExportedFrame] = []
        for fid in valid_frame_ids:
            records, drops = _select_records_for_frame(
                rows_by_frame.get(fid),
                records_by_frame.get(fid, []),
                params,
                scale_back,
            )
            exported.append(
                ExportedFrame(
                    frame_id=fid,
                    image_name=f"f{fid:06d}.jpg",
                    records=records,
                    is_context=fid not in selected,
                    drops=drops,
                )
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{dataset_name}_{timestamp}" if str(dataset_name).strip() else timestamp
        round_dir = Path(output_dir).resolve() / name

        provenance = {
            "source_video": str(video_path),
            "source_csv": str(csv_path),
            "detection_method": params.get("DETECTION_METHOD"),
            # The model/task actually run by the export pass. Sequential mode
            # runs the crop-OBB checkpoint under its stage-2 task, so reading
            # the direct-mode keys unconditionally made a sequential round
            # claim a model and a task it never used.
            "yolo_obb_mode": str(params.get("YOLO_OBB_MODE", "direct")).strip().lower(),
            "model_path": resolve_detection_model_path(params) or None,
            "model_task": resolve_detection_task(params),
            "export_confidence_threshold": params.get(
                "DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05
            ),
            "export_iou_threshold": params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5),
            "acquisition_preset": params.get("DATASET_AL_PRESET"),
            # The preset NAME alone does not reproduce a round: presets change,
            # and the scorer zeroes channels whose METRIC_* toggle is off and
            # renormalizes. Record the effective weights that actually ranked
            # these frames.
            "acquisition_weights": _effective_acquisition_weights(params),
            "resize_factor": float(params.get("RESIZE_FACTOR", 1.0) or 1.0),
            "reference_body_size": float(params.get("REFERENCE_BODY_SIZE", 20.0)),
            "image_width": frame_width,
            "image_height": frame_height,
            "note": (
                "Labels come from a dedicated export detection pass at lower "
                "confidence than tracking, so they may differ from the tracked "
                "detections. Every exported label is a detection that bound "
                "one-to-one to a tracked CSV row. KNOWN LIMITATION: a "
                "detection that binds to no row is dropped silently and is "
                "NOT counted anywhere (`dropped_unmatched` counts the mirror "
                "case, rows with no detection), so an animal the tracker "
                "missed can be visible in an exported image yet carry no "
                "label -- review every image before training. Frames where no "
                "detection survived are not exported at all, rather than "
                "written as empty (background) labels."
            ),
        }

        # `images` is only actually read here, while `cap` is still open --
        # export_al_dataset's authoritative root decodes each frame lazily.
        manifest = export_al_dataset(
            round_dir=round_dir,
            frames=exported,
            images=images,
            native_level=native_level,
            levels=levels,
            class_names=resolved_class_names,
            provenance=provenance,
            extra_totals=detection_stats,
        )
    finally:
        cap.release()
        if runner is not None:
            runner.close()

    logger.info("Dataset exported to %s (%d frames)", round_dir, len(exported))
    return manifest


def _get_detector_batch_size(runner):
    """Return the batch size to use for detection."""
    if runner is not None and getattr(runner, "config", None) is not None:
        return max(1, int(getattr(runner.config, "detection_batch_size", 1)))
    return 1
