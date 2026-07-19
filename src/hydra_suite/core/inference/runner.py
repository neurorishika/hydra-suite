from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cache.keys import (
    apriltag_cache_key,
    bgsub_detection_cache_key,
    cnn_cache_key,
    detection_cache_key,
    headtail_cache_key,
    pose_cache_key,
    video_signature,
    with_video_signature,
)
from .cache.store import (
    AprilTagCacheHandle,
    CacheHandle,
    CNNCacheHandle,
    DetectionCacheHandle,
    HeadTailCacheHandle,
    PoseCacheHandle,
)
from .cache.writer import CacheWriter
from .config import InferenceConfig
from .pipeline import Pipeline, PipelineStages
from .result import (
    AprilTagResult,
    CNNResult,
    FrameResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
    assemble_resolved_headings,
)
from .runtime import RuntimeContext
from .stages.apriltag import AprilTagModel, run_apriltag
from .stages.bgsub import BgSubModel, run_bgsub
from .stages.cnn import CNNModel, run_cnn
from .stages.crops import extract_aabb_crops, extract_canonical_crops
from .stages.filtering import filter_for_source
from .stages.headtail import HeadTailModel, run_headtail
from .stages.obb import OBBModels, _RawOBBTensors, materialize_tensors, run_obb
from .stages.pose import PoseModel, run_pose

logger = logging.getLogger(__name__)

# Opt-in realtime per-stage profiler (HYDRA_RT_PROFILE=1). Accumulates wall-clock
# per stage across run_realtime calls and logs a steady-state breakdown every
# 100 frames. Zero overhead when the env var is unset.
_RT_PROF_ACC: dict[str, float] = {}


def _rt_prof_on() -> bool:
    return bool(os.environ.get("HYDRA_RT_PROFILE"))


def _rt_prof_add(section: str, dt: float) -> None:
    _RT_PROF_ACC[section] = _RT_PROF_ACC.get(section, 0.0) + dt


def _rt_prof_flush() -> None:
    n = _RT_PROF_ACC.get("frames", 0.0)
    if n <= 0 or n % 100 != 0:
        return
    parts = " ".join(
        f"{k}={1000 * v / n:.1f}ms/f"
        for k, v in sorted(_RT_PROF_ACC.items())
        if k != "frames"
    )
    logger.warning("RT_PROFILE after %d frames: %s", int(n), parts)


@dataclass
class _AllModels:
    # Exactly one of obb/bgsub is set, mirroring InferenceConfig.detection_source
    # (bgsub is last with a default so existing keyword constructions still work).
    obb: OBBModels | None
    headtail: HeadTailModel | None
    cnn: list[CNNModel]
    pose: PoseModel | None
    apriltag: AprilTagModel | None
    bgsub: BgSubModel | None = None


@dataclass
class _CacheSet:
    detection: DetectionCacheHandle | None = None
    headtail: HeadTailCacheHandle | None = None
    cnn: list[CNNCacheHandle] = field(default_factory=list)
    pose: PoseCacheHandle | None = None
    apriltag: AprilTagCacheHandle | None = None

    def all_handles(self) -> list[CacheHandle]:
        handles: list[CacheHandle] = []
        if self.detection is not None:
            handles.append(self.detection)
        if self.headtail is not None:
            handles.append(self.headtail)
        handles.extend(self.cnn)
        if self.pose is not None:
            handles.append(self.pose)
        if self.apriltag is not None:
            handles.append(self.apriltag)
        return handles


def _load_all_models(
    config: InferenceConfig,
    runtime: RuntimeContext,
    *,
    cache_only: bool = False,
    video_path: str | None = None,
) -> _AllModels:
    """Load all inference models.

    When *cache_only* is True the runner will only be used for cache replay
    (``load_frame``/``caches_all_valid``/``detection_cache_covers_range``).
    In that mode every model besides the OBB detector is skipped: OBB is
    required to look up cache-key validity; HeadTail, CNN, Pose, and AprilTag
    models are never invoked during replay so we avoid the expensive backend
    initialisation (notably the ~8 s per-session SLEAP/ORT-TRT-EP init).

    bg-sub is the exception to the OBB rule: its cache key hashes params only
    (there is no model file — the "model" is a BackgroundModel primed from the
    video), so under *cache_only* it is skipped entirely rather than loaded.
    Priming reads ~BACKGROUND_PRIME_FRAMES frames off the video, so this is a
    real saving, and a replay pass never calls the stage anyway.
    """
    from .stages.apriltag import load_apriltag_model
    from .stages.bgsub import load_bgsub_model
    from .stages.cnn import load_cnn_model
    from .stages.headtail import load_headtail_model
    from .stages.obb import load_obb_models
    from .stages.pose import load_pose_model

    obb = None
    bgsub = None
    if config.detection_source == "obb":
        obb = load_obb_models(
            config.obb, runtime, batch_size=config.detection_batch_size
        )
    elif not cache_only:
        bgsub = load_bgsub_model(config.bgsub, runtime, video_path=video_path)

    if cache_only:
        logger.debug(
            "InferenceRunner cache_only=True: skipping HeadTail/CNN/Pose/AprilTag "
            "model init (backward/replay pass reads from cache only)."
        )
        return _AllModels(
            obb=obb, headtail=None, cnn=[], pose=None, apriltag=None, bgsub=bgsub
        )

    headtail = (
        load_headtail_model(config.headtail, runtime)
        if config.headtail is not None
        else None
    )
    cnn = [load_cnn_model(c, runtime) for c in config.cnn_phases]
    pose = load_pose_model(config.pose, runtime) if config.pose is not None else None
    apriltag = load_apriltag_model(config.apriltag) if config.apriltag.enabled else None
    return _AllModels(
        obb=obb,
        headtail=headtail,
        cnn=cnn,
        pose=pose,
        apriltag=apriltag,
        bgsub=bgsub,
    )


def _open_caches(
    config: InferenceConfig, cache_dir: Path, video_sig: str = ""
) -> _CacheSet:
    # Bind every per-video cache to the exact source file so a changed video
    # (e.g. a clip regenerated under the same name with a different frame count)
    # invalidates the cache instead of serving stale, truncated detections.
    def _k(key):
        return with_video_signature(key, video_sig)

    detection_key = (
        detection_cache_key(config.obb)
        if config.detection_source == "obb"
        else bgsub_detection_cache_key(config.bgsub)
    )

    return _CacheSet(
        detection=DetectionCacheHandle(
            path=cache_dir / "detection.npz",
            key=_k(detection_key),
        ),
        headtail=(
            HeadTailCacheHandle(
                path=cache_dir / "headtail.npz",
                key=_k(headtail_cache_key(config.headtail)),
            )
            if config.headtail is not None
            else None
        ),
        cnn=[
            CNNCacheHandle(
                path=cache_dir / f"cnn_{c.label}.npz",
                key=_k(cnn_cache_key(c)),
                label=c.label,
            )
            for c in config.cnn_phases
        ],
        pose=(
            PoseCacheHandle(
                path=cache_dir / "pose.npz",
                key=_k(pose_cache_key(config.pose)),
            )
            if config.pose is not None
            else None
        ),
        apriltag=(
            AprilTagCacheHandle(
                path=cache_dir / "apriltag.npz",
                key=_k(apriltag_cache_key(config.apriltag)),
            )
            if config.apriltag.enabled
            else None
        ),
    )


def _build_frame_result(
    frame_idx: int,
    filtered_obb: OBBResult,
    det_indices: np.ndarray,
    ht: HeadTailResult | None,
    cnn_results: list[CNNResult],
    pose_result: PoseResult | None,
    at_result: AprilTagResult | None,
    overrides_headtail: bool = True,
) -> FrameResult:
    pose_headings: np.ndarray | None = None
    pose_valid: np.ndarray | None = None
    if pose_result is not None:
        pose_headings = getattr(pose_result, "heading_overrides", None)
        pose_valid = pose_result.valid_mask
    resolved = assemble_resolved_headings(
        filtered_obb,
        ht,
        pose_headings,
        pose_valid,
        overrides_headtail=overrides_headtail,
    )
    return FrameResult(
        frame_idx=frame_idx,
        obb=filtered_obb,
        filtered_indices=[int(i) for i in det_indices],
        headtail=ht,
        cnn=cnn_results,
        pose=pose_result,
        apriltag=at_result,
        resolved_headings=resolved,
    )


def _load_headtail_for_indices(
    cache: HeadTailCacheHandle | None,
    frame_idx: int,
    det_indices: np.ndarray,
    filtered_obb: OBBResult,
) -> HeadTailResult | None:
    if cache is None or len(det_indices) == 0:
        return None
    data = cache.read_frame(frame_idx)
    if data is None:
        return None
    cached_det_indices, hints, confs, directed = data
    idx_map = {int(v): i for i, v in enumerate(cached_det_indices)}
    n = len(det_indices)
    out_hints = np.full(n, float("nan"), dtype=np.float32)
    out_confs = np.zeros(n, dtype=np.float32)
    out_directed = np.zeros(n, dtype=np.uint8)
    for i, di in enumerate(det_indices):
        j = idx_map.get(int(di))
        if j is not None:
            out_hints[i] = hints[j]
            out_confs[i] = confs[j]
            out_directed[i] = 1 if bool(directed[j]) else 0
    return HeadTailResult(
        heading_hints=out_hints,
        heading_confidences=out_confs,
        directed_mask=out_directed,
        canonical_affines=None,
    )


def _load_cnn_for_indices(
    caches: list[CNNCacheHandle],
    cnn_configs: list,
    frame_idx: int,
    det_indices: np.ndarray,
) -> list[CNNResult]:
    results: list[CNNResult] = []
    det_set = {int(di) for di in det_indices}
    for cache, cfg in zip(caches, cnn_configs):
        preds = cache.read_frame(frame_idx)
        if preds is None:
            results.append(CNNResult(label=cfg.label, predictions=[]))
            continue
        aligned = [p for p in preds if p.det_index in det_set]
        results.append(CNNResult(label=cfg.label, predictions=aligned))
    return results


def _load_pose_for_indices(
    cache: PoseCacheHandle | None,
    frame_idx: int,
    det_indices: np.ndarray,
    filtered_obb: OBBResult,
) -> PoseResult | None:
    if cache is None or len(det_indices) == 0:
        return None
    data = cache.read_frame(frame_idx)
    if data is None:
        return None
    cached_keypoints, cached_det_indices, cached_valid = data
    idx_map = {int(v): i for i, v in enumerate(cached_det_indices)}
    n = len(det_indices)
    if cached_keypoints.ndim < 2:
        return None
    kp_shape = cached_keypoints.shape[1:]
    out_kp = np.zeros((n, *kp_shape), dtype=np.float32)
    out_valid = np.zeros(n, dtype=bool)
    for i, di in enumerate(det_indices):
        j = idx_map.get(int(di))
        if j is not None:
            out_kp[i] = cached_keypoints[j]
            out_valid[i] = bool(cached_valid[j])
    return PoseResult(keypoints=out_kp, valid_mask=out_valid)


def _load_apriltag(
    cache: AprilTagCacheHandle | None,
    frame_idx: int,
) -> AprilTagResult | None:
    if cache is None:
        return None
    return cache.read_frame(frame_idx)


class InferenceRunner:
    """Orchestrates model lifecycle, real-time inference, and batch-pass caching.

    `caches_all_valid()` returns True only when every enabled cache file exists
    and matches its key. Real-time path runs all stages on a single frame, no I/O.
    Batch-pass path runs OBB on batched frames natively, then iterates per frame
    for HeadTail/CNN/Pose/AprilTag (no cross-frame crop batching) so each crop's
    aspect ratio is preserved when stages internally resize to model input size.

    Pass ``cache_only=True`` when the runner will only be used for cache replay
    (backward/replay passes that call ``load_frame``, ``caches_all_valid``, or
    ``detection_cache_covers_range``).  In that mode the expensive HeadTail, CNN,
    Pose (including SLEAP), and AprilTag backends are never initialised — only the
    lightweight OBB model wrapper is loaded so cache-key validation still works.
    This eliminates the ~8 s per-session SLEAP/ORT-TRT-EP init on backward passes.
    """

    def __init__(
        self,
        config: InferenceConfig,
        cache_dir: Path | None = None,
        video_path: str | Path | None = None,
        cache_only: bool = False,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.cache_only = cache_only
        # Fingerprint of the source video; folded into every cache key so caches
        # are only reused for the exact file they were computed from.
        self._video_path = str(video_path) if video_path else None
        self._video_sig = video_signature(self._video_path)
        self.runtime = RuntimeContext.from_config(config)
        # bg-sub's "model" is a BackgroundModel primed from the video itself, so
        # the loader needs the path; the OBB loader ignores it.
        self._models = _load_all_models(
            config,
            self.runtime,
            cache_only=cache_only,
            video_path=self._video_path,
        )
        self._caches: _CacheSet | None = None
        # True when self._caches was opened for WRITING (realtime persistence);
        # False when opened read-only by load_frame. close() only flushes when
        # writable, so a backward (read) pass never overwrites the forward cache.
        self._caches_writable = False

    def caches_all_valid(self) -> bool:
        if self.cache_dir is None:
            return False
        caches = _open_caches(self.config, self.cache_dir, self._video_sig)
        return all(h.is_valid() for h in caches.all_handles())

    def detection_cache_covers_range(self, start_frame: int, end_frame: int) -> bool:
        """Return True iff the detection cache spans every frame in the range.

        Key validity alone (``caches_all_valid``) does not guarantee a cache
        produced by a full forward pass: an interrupted or shorter run yields a
        valid-keyed cache covering fewer frames. Backward/replay passes must
        additionally confirm frame-range coverage (legacy parity, H9).
        """
        if self.cache_dir is None:
            return False
        caches = _open_caches(self.config, self.cache_dir, self._video_sig)
        if caches.detection is None:
            return False
        return caches.detection.covers_frame_range(start_frame, end_frame)

    def detection_cache_missing_frames(
        self, start_frame: int, end_frame: int, max_report: int = 10
    ) -> list[int]:
        """Report up to ``max_report`` frames missing from the detection cache."""
        if self.cache_dir is None:
            return []
        caches = _open_caches(self.config, self.cache_dir, self._video_sig)
        if caches.detection is None:
            return []
        return caches.detection.get_missing_frames(start_frame, end_frame, max_report)

    def run_realtime(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        roi_mask: np.ndarray | None = None,
        roi_mask_cuda: Any = None,
    ) -> FrameResult:
        # Lazily open the caches for WRITING so the realtime forward pass persists
        # detections + downstream results. Backward tracking replays them via
        # load_frame; without this, realtime + backward gets an empty backward pass.
        if self._caches is None and self.cache_dir is not None:
            self._caches = _open_caches(self.config, self.cache_dir, self._video_sig)
            self._caches_writable = True
        caches = self._caches if self._caches_writable else None

        _prof = _rt_prof_on()
        _ts = time.perf_counter() if _prof else 0.0
        if self.config.detection_source == "bgsub":
            if self._models.bgsub is None:
                raise RuntimeError(
                    "run_realtime() requires a loaded bg-sub model, but this runner "
                    "was constructed with cache_only=True (replay only). Construct "
                    "without cache_only to run detection."
                )
            # bg-sub is CPU numpy end to end: it never produces _RawOBBTensors, so
            # the materialize / raw-cap step does not apply. It is also strictly
            # sequential — safe here because run_realtime is driven in frame order.
            raw_obb = run_bgsub(
                frame,
                frame_idx,
                self._models.bgsub,
                self.config.bgsub,
                self.runtime,
                roi_mask=roi_mask,
            )
        else:
            raw_list = run_obb([frame], self._models.obb, self.config.obb, self.runtime)
            raw = raw_list[0]
            if isinstance(raw, _RawOBBTensors):
                raw_obb = materialize_tensors(raw, self.config.obb.raw_detection_cap)
            else:
                raw_obb = raw
        # Re-stamp detection_ids with the real frame_idx (materialize_tensors / the
        # CPU OBB path generate them at frame 0) so cached ids are unique per frame.
        raw_obb = OBBResult(
            frame_idx=frame_idx,
            centroids=raw_obb.centroids,
            angles=raw_obb.angles,
            sizes=raw_obb.sizes,
            shapes=raw_obb.shapes,
            confidences=raw_obb.confidences,
            corners=raw_obb.corners,
            detection_ids=OBBResult.make_detection_ids(
                frame_idx, raw_obb.num_detections
            ),
            class_ids=raw_obb.class_ids,
        )
        if caches is not None and caches.detection is not None:
            caches.detection.write_frame(frame_idx, result=raw_obb)

        if _prof:
            _now = time.perf_counter()
            _rt_prof_add("obb", _now - _ts)
            _ts = _now

        filtered_obb, det_indices = filter_for_source(self.config, raw_obb, roi_mask)

        if filtered_obb.num_detections == 0:
            if _prof:
                _rt_prof_add("frames", 1)
                _rt_prof_flush()
            empty_result = _build_frame_result(
                frame_idx, filtered_obb, np.zeros(0, np.int32), None, [], None, None
            )
            # Task 11 fix: surface the bg-sub masks here too, exactly like the
            # non-empty path below (646-648). last_bg_u8 is the source of
            # truth for "was the background established" -- it is None ONLY
            # during the true first-frame warmup (see bgsub.py:167-170) and a
            # real array on every frame after, even with zero detections.
            # worker.py:2314 uses `bg_u8 is None` as its warmup sentinel; if
            # this early return skipped the assignment, a post-warmup
            # zero-detection frame (occlusion, animal left, threshold blip)
            # would be misread as still-warming-up and silently drop Kalman
            # aging + the CSV row for that frame.
            if (
                self.config.detection_source == "bgsub"
                and self._models.bgsub is not None
            ):
                empty_result.fg_mask = self._models.bgsub.last_fg_mask
                empty_result.bg_u8 = self._models.bgsub.last_bg_u8
            return empty_result

        ar = (
            self.config.headtail.canonical_aspect_ratio if self.config.headtail else 2.0
        )
        mg = self.config.headtail.canonical_margin if self.config.headtail else 1.3
        # Canonical (native-extent) crops are now only consumed by the pose stage;
        # head-tail / CNN warp directly from the frame. Skip the extraction
        # entirely when there is no pose model (e.g. OBB-only / identity clips).
        # Foreign-ant masking (suppress_foreign_regions) mirrors legacy's
        # unconditional suppress_foreign_obb: legacy has no realtime/batch
        # split and always masks, so the realtime path must too.
        pose_cfg = self.config.pose
        suppress_foreign = (
            pose_cfg.suppress_foreign_regions if pose_cfg is not None else False
        )
        background_color = (
            pose_cfg.background_color if pose_cfg is not None else (0, 0, 0)
        )
        canonical_crops = (
            extract_canonical_crops(
                frame,
                filtered_obb,
                ar,
                mg,
                self.runtime,
                suppress_foreign=suppress_foreign,
                background_color=background_color,
            )
            if self._models.pose is not None
            else None
        )
        aabb_crops = (
            extract_aabb_crops(
                frame, filtered_obb, padding=self.config.apriltag.crop_padding
            )
            if self._models.apriltag
            else []
        )

        if _prof:
            _now = time.perf_counter()
            _rt_prof_add("crops", _now - _ts)
            _ts = _now

        def _do_ht() -> HeadTailResult | None:
            if not self._models.headtail:
                return None
            return run_headtail(
                frame,
                filtered_obb,
                self._models.headtail,
                self.config.headtail,
                self.runtime,
                ar,
                mg,
            )

        def _do_cnn() -> list[CNNResult]:
            return [
                run_cnn(frame, filtered_obb, mdl, cfg, self.runtime, ar, mg)
                for cfg, mdl in zip(self.config.cnn_phases, self._models.cnn)
            ]

        def _do_pose() -> PoseResult | None:
            if not self._models.pose:
                return None
            return run_pose(
                canonical_crops,
                filtered_obb,
                self._models.pose,
                self.config.pose,
                self.runtime,
                ar,
                mg,
            )

        def _do_at() -> AprilTagResult | None:
            if not self._models.apriltag:
                return None
            return run_apriltag(
                aabb_crops,
                filtered_obb,
                self._models.apriltag,
                self.config.apriltag,
            )

        # Run the individual-analysis stages SEQUENTIALLY, not in a per-frame
        # ThreadPoolExecutor. Profiling on CUDA (RT_PROFILE) showed the per-frame
        # pool cost ~834 ms/frame vs ~37 ms/frame sequential (a 22x regression):
        # spinning up a fresh 4-thread pool every frame and driving CUDA / the
        # onnxruntime SLEAP backend from short-lived worker threads serialises on
        # the GIL and the default CUDA stream while paying thread + context setup
        # each frame, with no real parallelism on a single GPU. Sequential brings
        # realtime back to legacy parity (~137 ms/frame total incl. frame read).
        ht_result = _do_ht()
        cnn_results = _do_cnn()
        pose_result = _do_pose()
        at_result = _do_at()

        if _prof:
            _now = time.perf_counter()
            _rt_prof_add("individual", _now - _ts)
            _ts = _now

        # Persist downstream results (keyed by det_indices) so the backward pass
        # can replay them via load_frame -- mirrors _run_batch's cache writes.
        if caches is not None:
            if caches.headtail is not None and ht_result is not None:
                caches.headtail.write_frame(
                    frame_idx,
                    det_indices=det_indices,
                    heading_hints=ht_result.heading_hints,
                    heading_confidences=ht_result.heading_confidences,
                    directed_mask=ht_result.directed_mask,
                )
            for cache, cnn_result in zip(caches.cnn, cnn_results):
                if cnn_result is not None:
                    cache.write_frame(frame_idx, predictions=cnn_result.predictions)
            if caches.pose is not None and pose_result is not None:
                caches.pose.write_frame(
                    frame_idx,
                    det_indices=det_indices,
                    keypoints=pose_result.keypoints,
                    valid_mask=pose_result.valid_mask,
                )
            if caches.apriltag is not None and at_result is not None:
                caches.apriltag.write_frame(frame_idx, result=at_result)

        frame_result = _build_frame_result(
            frame_idx,
            filtered_obb,
            det_indices,
            ht_result,
            cnn_results,
            pose_result,
            at_result,
        )

        # Task 10b: surface the bg-sub masks for the SHOW_FG / SHOW_BG preview
        # overlays. Realtime-only, like streaming_payload below: run_bgsub just
        # stashed these on the (strictly sequential) model, so "last" is this
        # frame's. Left None on the OBB path, which has no such masks.
        if self.config.detection_source == "bgsub" and self._models.bgsub is not None:
            frame_result.fg_mask = self._models.bgsub.last_fg_mask
            frame_result.bg_u8 = self._models.bgsub.last_bg_u8

        # Task 17g: build StreamingAnalysisPayload for legacy identity consumers.
        try:
            from hydra_suite.core.tracking.ingest.streaming_payload import (
                StreamingAnalysisPayload,
            )

            frame_result.streaming_payload = StreamingAnalysisPayload.from_frame_result(
                frame_result,
                runtime_family=str(self.runtime.default_runtime),
                input_is_bgr=True,
            )
        except Exception:
            pass  # streaming_payload is optional; failures are non-fatal

        if _prof:
            _rt_prof_add("finalize", time.perf_counter() - _ts)
            _rt_prof_add("frames", 1)
            _rt_prof_flush()

        return frame_result

    def detect_batch(
        self,
        frames: "list[np.ndarray]",
        frame_indices: "list[int] | None" = None,
        roi_mask: "np.ndarray | None" = None,
    ) -> "list[OBBResult]":
        """Run OBB detection over a list of frames, returning filtered results
        in memory. No cache is read or written. Mirrors run_realtime's
        detect+filter prefix; for the dataset-generation batched path.
        """
        if self._models.obb is None:
            raise RuntimeError(
                "detect_batch requires an OBB detection config (config.obb)"
            )
        frames = list(frames)
        if frame_indices is None:
            frame_indices = list(range(len(frames)))

        raw_list = run_obb(frames, self._models.obb, self.config.obb, self.runtime)
        results: list[OBBResult] = []
        for raw, f_idx in zip(raw_list, frame_indices):
            if isinstance(raw, _RawOBBTensors):
                raw_obb = materialize_tensors(raw, self.config.obb.raw_detection_cap)
            else:
                raw_obb = raw
            raw_obb = OBBResult(
                frame_idx=f_idx,
                centroids=raw_obb.centroids,
                angles=raw_obb.angles,
                sizes=raw_obb.sizes,
                shapes=raw_obb.shapes,
                confidences=raw_obb.confidences,
                corners=raw_obb.corners,
                detection_ids=OBBResult.make_detection_ids(
                    f_idx, raw_obb.num_detections
                ),
                class_ids=raw_obb.class_ids,
            )
            filtered_obb, _ = filter_for_source(self.config, raw_obb, roi_mask)
            results.append(filtered_obb)
        return results

    def _build_pipeline(self, caches: _CacheSet) -> Pipeline:
        """Construct the depth=1 Pipeline that drives the batch stage layer.

        The Pipeline owns the per-window stage sequence (OBB → crops → HT/CNN/pose
        → AprilTag → scatter); cache writes go through a ``CacheWriter`` (sync mode)
        that reproduces ``_run_batch``'s exact raw-result side effects.
        """
        stages = PipelineStages(
            config=self.config,
            obb_models=self._models.obb,
            bgsub_model=self._models.bgsub,
            headtail_model=self._models.headtail,
            cnn_models=self._models.cnn,
            pose_model=self._models.pose,
            apriltag_model=self._models.apriltag,
        )
        handles: dict[str, CacheHandle] = {}
        if caches.detection is not None:
            handles["detection"] = caches.detection
        if caches.headtail is not None:
            handles["headtail"] = caches.headtail
        for cnn_cfg, cnn_handle in zip(self.config.cnn_phases, caches.cnn):
            handles[f"cnn_{cnn_cfg.label}"] = cnn_handle
        if caches.pose is not None:
            handles["pose"] = caches.pose
        if caches.apriltag is not None:
            handles["apriltag"] = caches.apriltag
        # depth>=2 uses an async CacheWriter so cache writes never stall the
        # compute path; the consumer thread still calls the direct write helpers
        # (write_detection/write_downstream) in strict window order, so the cache
        # layout is byte-identical to the synchronous depth=1 writer.
        async_mode = self.config.pipeline_depth >= 2
        writer = CacheWriter(handles, self.config.cnn_phases, async_mode=async_mode)
        return Pipeline(stages, self.runtime, writer, depth=self.config.pipeline_depth)

    def run_batch_pass(
        self,
        video_path: Path,
        progress_cb=None,
        start_frame: int = 0,
        end_frame: int | None = None,
        should_stop=None,
    ) -> None:
        from .sources import make_frame_source

        if self.cache_dir is None:
            raise RuntimeError("cache_dir must be set before calling run_batch_pass")

        # make_frame_source selects NvdecFrameReader when runtime.use_nvdec is True
        # and the decoder is available; otherwise falls back to CpuFrameReader.
        # Clamping and seeking are handled inside each reader implementation.
        frame_source = make_frame_source(
            video_path, self.runtime, start_frame, end_frame
        )

        caches = _open_caches(self.config, self.cache_dir, self._video_sig)
        self._caches = caches

        # Recover the clamped bounds from the reader so range_total matches.
        start_frame = frame_source.start_frame
        end_frame = frame_source.end_frame
        range_total = frame_source.frame_count

        # The whole pass is now driven by Pipeline.run: it owns the windowing and
        # (at depth>=2) the producer/consumer double buffer. The video decode is
        # the producer's first stage and is fed in as a lazy (frame_idx, frame)
        # generator so frames are never all buffered at once. Range clamping,
        # progress cadence, signature binding, and the final cache close are
        # preserved; only the orchestration moved into the Pipeline.
        pipeline = self._build_pipeline(caches)
        try:
            pipeline.run(
                frame_source,
                range(start_frame, end_frame + 1),
                progress_cb=progress_cb,
                range_total=range_total,
                should_stop=should_stop,
            )
        finally:
            frame_source.close()
            # depth>=2 uses an async CacheWriter; flush/close it before closing the
            # handles so all queued writes land (Pipeline.run already does this on
            # its own teardown path, but a pre-run failure may skip it).
            try:
                pipeline.cache_writer.close()
            except Exception:
                pass
            for h in caches.all_handles():
                h.close()

    def _run_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        caches: _CacheSet,
    ) -> None:
        """Process a single window through the Pipeline (test/legacy seam).

        No longer used by ``run_batch_pass`` (which now drives the whole pass via
        ``Pipeline.run``), but retained as a single-window entry point for tests
        that exercise the per-window stage sequence + cache writes directly.
        Cache side effects are identical to the full-pass path.
        """
        from .pipeline import BatchWindow

        pipeline = self._build_pipeline(caches)
        pipeline._process_window(
            BatchWindow(frames=list(frames), frame_indices=list(frame_indices))
        )
        # _process_window enqueues writes; an async (depth>=2) writer offloads
        # them to its worker thread. The full pass flushes via Pipeline.run's
        # teardown; this direct-seam path must flush so the writes land before
        # the caller inspects the handles.
        pipeline.cache_writer.flush()
        pipeline.cache_writer.close()

    def load_frame(self, frame_idx: int) -> FrameResult:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir not set — cannot load cached frames")
        if self._caches is None:
            self._caches = _open_caches(self.config, self.cache_dir, self._video_sig)

        raw_obb = (
            self._caches.detection.read_frame(frame_idx)
            if self._caches.detection is not None
            else None
        )
        if raw_obb is None:
            raise KeyError(f"Frame {frame_idx} not found in detection cache")

        # Cache-only by construction: bg-sub carries cross-frame state and must
        # never be re-run for random access — filter_for_source is the identity
        # on the bg-sub branch, so this stays a pure cache read.
        filtered_obb, det_indices = filter_for_source(self.config, raw_obb)

        ht_result = _load_headtail_for_indices(
            self._caches.headtail, frame_idx, det_indices, filtered_obb
        )
        cnn_results = _load_cnn_for_indices(
            self._caches.cnn, self.config.cnn_phases, frame_idx, det_indices
        )
        pose_result = _load_pose_for_indices(
            self._caches.pose, frame_idx, det_indices, filtered_obb
        )
        at_result = _load_apriltag(self._caches.apriltag, frame_idx)

        return _build_frame_result(
            frame_idx,
            filtered_obb,
            det_indices,
            ht_result,
            cnn_results,
            pose_result,
            at_result,
        )

    def close(self) -> None:
        # Flush realtime-written caches to disk so a later backward pass can
        # replay them. Only when writable: a read-only (load_frame/backward)
        # handle has an empty buffer and close() would overwrite the cache.
        if self._caches is not None and self._caches_writable:
            for h in self._caches.all_handles():
                h.close()
            self._caches = None
            self._caches_writable = False
        if self._models.obb is not None:
            self._models.obb.close()
        if self._models.bgsub is not None:
            self._models.bgsub.close()
        if self._models.headtail is not None:
            self._models.headtail.close()
        for mdl in self._models.cnn:
            mdl.close()
        if self._models.pose is not None:
            self._models.pose.close()
        if self._models.apriltag is not None:
            self._models.apriltag.close()
