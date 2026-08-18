"""
Dataset generation utilities for active learning.
Identifies challenging frames and exports them for annotation.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

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

    mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    if mode == "sequential":
        task = str(params.get("YOLO_OBB_STAGE2_TASK", "obb")).strip().lower()
    else:
        task = str(params.get("YOLO_OBB_DIRECT_TASK", "obb")).strip().lower()
    return _TASK_LEVELS.get(task, GeometryLevel.OBB)


class FrameQualityScorer:
    """Tracker-side adapter that produces ALSignals and selects worst frames.

    Public API (`score_frame`, `get_worst_frames`) is preserved for callers; the
    underlying ranking now lives in `hydra_suite.data.al.acquisition`.
    """

    def __init__(self, params, frame_shape: tuple[int, int] | None = None):
        from hydra_suite.data.al.acquisition import PRESETS, AcquisitionWeights

        self.params = params
        self.frame_signals: dict = {}
        self.max_targets = params.get("MAX_TARGETS", 4)
        self.conf_threshold = params.get("DATASET_CONF_THRESHOLD", 0.5)
        self.reference_body_size = max(
            float(params.get("REFERENCE_BODY_SIZE", 20.0)), 1.0
        )
        # (H, W) of the coordinate space `obb_corners` live in. Required for a
        # meaningful edge score: passing (1, 1) with pixel-space corners made
        # score_crowd return values in the hundreds.
        self.frame_shape = tuple(frame_shape) if frame_shape else None

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

        self._enabled = {
            "uncertainty": self.use_confidence and self._confidence_available,
            "count": self.use_count_mismatch,
            "assignment": self.use_assignment_cost,
            "track_loss": self.use_track_loss,
            "position_uncertainty": self.use_uncertainty,
            "crowd": self.use_crowd,
            "fragmentation": self.use_fragmented_detections,
        }

        preset_name = params.get("DATASET_AL_PRESET", "tracker_default")
        base = PRESETS.get(preset_name, PRESETS["tracker_default"])
        self._weights = AcquisitionWeights(
            uncertainty=base.uncertainty if self._enabled["uncertainty"] else 0.0,
            nms_instability=0.0,
            count=base.count if self._enabled["count"] else 0.0,
            crowd=base.crowd if self._enabled["crowd"] else 0.0,
            edge=base.edge,
            fragmentation=(
                base.fragmentation if self._enabled["fragmentation"] else 0.0
            ),
            assignment=base.assignment if self._enabled["assignment"] else 0.0,
            track_loss=base.track_loss if self._enabled["track_loss"] else 0.0,
            position_uncertainty=(
                base.position_uncertainty
                if self._enabled["position_uncertainty"]
                else 0.0
            ),
        ).normalized()

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
            obb_corners, reference_major_axis=self.reference_body_size * 2.2
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


def _init_detection_runner(params):
    """Build a detection-only InferenceRunner for dataset label extraction.

    Unlike the legacy version this supports every detection source, not just
    `yolo_obb`: returning None for bg-sub meant every exported label was a
    fabricated reference-size box.
    """
    method = str(params.get("DETECTION_METHOD", "background_subtraction")).lower()
    native_level = resolve_native_level(params)
    try:
        from ..core.inference.runner import InferenceRunner

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
            task = (
                (
                    str(params.get("YOLO_OBB_STAGE2_TASK", "obb"))
                    if mode == "sequential"
                    else str(params.get("YOLO_OBB_DIRECT_TASK", "obb"))
                )
                .strip()
                .lower()
            )
            model_path = str(
                params.get(
                    "YOLO_OBB_DIRECT_MODEL_PATH",
                    params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
                )
                or "yolo26s-obb.pt"
            )
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
            )

        runner = InferenceRunner(cfg)
        logger.info(
            "Detection runner initialized for dataset export (method=%s, level=%s)",
            method,
            native_level.label,
        )
        return runner
    except Exception as e:
        logger.warning(
            "Could not initialize detection runner: %s. "
            "Labels will fall back to reference-size approximation.",
            e,
        )
        return None


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
    """Run detection over `frames` and return {frame_id: [LabelRecord]}.

    `frames` is a Mapping[frame_id, image] consulted in batches -- only the
    current batch's images are held in memory at once, whether `frames` is a
    plain dict (tests) or a lazy, decode-on-access mapping (production).
    """
    if runner is None:
        return {}
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
            results = runner.detect_batch(images, frame_indices=list(valid_chunk))
        except Exception as e:
            logger.warning(
                "Detection failed for batch starting at %s: %s", valid_chunk[0], e
            )
            continue
        for fid, obb in zip(valid_chunk, results):
            records = records_from_obb_result(obb, native_level)
            if detection_scale_back != 1.0:
                for rec in records:
                    rec.points = (rec.points * detection_scale_back).astype(np.float32)
            out[fid] = records
    return out


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
    runner = _init_detection_runner(params)
    try:
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

        records_by_frame = _detect_records_for_frames(
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
            "model_path": params.get("YOLO_OBB_DIRECT_MODEL_PATH"),
            "model_task": params.get("YOLO_OBB_DIRECT_TASK"),
            "export_confidence_threshold": params.get(
                "DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05
            ),
            "export_iou_threshold": params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5),
            "acquisition_preset": params.get("DATASET_AL_PRESET"),
            "image_width": frame_width,
            "image_height": frame_height,
            "note": (
                "Labels come from a dedicated export detection pass at lower "
                "confidence than tracking, so they may differ from the tracked "
                "detections. Review before training."
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
