from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cache.keys import (
    apriltag_cache_key,
    cnn_cache_key,
    detection_cache_key,
    headtail_cache_key,
    pose_cache_key,
)
from .cache.store import (
    AprilTagCacheHandle,
    CacheHandle,
    CNNCacheHandle,
    DetectionCacheHandle,
    HeadTailCacheHandle,
    PoseCacheHandle,
)
from .config import InferenceConfig
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
from .stages.cnn import CNNModel, run_cnn
from .stages.crops import extract_aabb_crops, extract_canonical_crops
from .stages.filtering import filter_with_indices
from .stages.headtail import HeadTailModel, run_headtail
from .stages.obb import OBBModels, _RawOBBTensors, materialize_tensors, run_obb
from .stages.pose import PoseModel, run_pose


@dataclass
class _AllModels:
    obb: OBBModels
    headtail: HeadTailModel | None
    cnn: list[CNNModel]
    pose: PoseModel | None
    apriltag: AprilTagModel | None


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


def _load_all_models(config: InferenceConfig, runtime: RuntimeContext) -> _AllModels:
    from .stages.apriltag import load_apriltag_model
    from .stages.cnn import load_cnn_model
    from .stages.headtail import load_headtail_model
    from .stages.obb import load_obb_models
    from .stages.pose import load_pose_model

    obb = load_obb_models(config.obb, runtime)
    headtail = (
        load_headtail_model(config.headtail, runtime)
        if config.headtail is not None
        else None
    )
    cnn = [load_cnn_model(c, runtime) for c in config.cnn_phases]
    pose = load_pose_model(config.pose, runtime) if config.pose is not None else None
    apriltag = load_apriltag_model(config.apriltag) if config.apriltag.enabled else None
    return _AllModels(obb=obb, headtail=headtail, cnn=cnn, pose=pose, apriltag=apriltag)


def _open_caches(config: InferenceConfig, cache_dir: Path) -> _CacheSet:
    return _CacheSet(
        detection=DetectionCacheHandle(
            path=cache_dir / "detection.npz",
            key=detection_cache_key(config.obb),
        ),
        headtail=(
            HeadTailCacheHandle(
                path=cache_dir / "headtail.npz",
                key=headtail_cache_key(config.headtail),
            )
            if config.headtail is not None
            else None
        ),
        cnn=[
            CNNCacheHandle(
                path=cache_dir / f"cnn_{c.label}.npz",
                key=cnn_cache_key(c),
                label=c.label,
            )
            for c in config.cnn_phases
        ],
        pose=(
            PoseCacheHandle(
                path=cache_dir / "pose.npz",
                key=pose_cache_key(config.pose),
            )
            if config.pose is not None
            else None
        ),
        apriltag=(
            AprilTagCacheHandle(
                path=cache_dir / "apriltag.npz",
                key=apriltag_cache_key(config.apriltag),
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
    """

    def __init__(self, config: InferenceConfig, cache_dir: Path | None = None) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.runtime = RuntimeContext.from_config(config)
        self._models = _load_all_models(config, self.runtime)
        self._caches: _CacheSet | None = None

    def caches_all_valid(self) -> bool:
        if self.cache_dir is None:
            return False
        caches = _open_caches(self.config, self.cache_dir)
        return all(h.is_valid() for h in caches.all_handles())

    def run_realtime(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray | None = None,
        roi_mask_cuda: Any = None,
    ) -> FrameResult:
        raw_list = run_obb([frame], self._models.obb, self.config.obb, self.runtime)
        raw = raw_list[0]
        if isinstance(raw, _RawOBBTensors):
            raw_obb = materialize_tensors(raw)
        else:
            raw_obb = raw
        # Re-stamp detection_ids since materialize_tensors and the CPU OBB path may
        # generate them with frame_idx=0; ensure consistency.
        raw_obb = OBBResult(
            frame_idx=0,
            centroids=raw_obb.centroids,
            angles=raw_obb.angles,
            sizes=raw_obb.sizes,
            shapes=raw_obb.shapes,
            confidences=raw_obb.confidences,
            corners=raw_obb.corners,
            detection_ids=OBBResult.make_detection_ids(0, raw_obb.num_detections),
        )
        filtered_obb, det_indices = filter_with_indices(
            raw_obb, self.config.obb, roi_mask
        )

        if filtered_obb.num_detections == 0:
            return _build_frame_result(
                0, filtered_obb, np.zeros(0, np.int32), None, [], None, None
            )

        ar = (
            self.config.headtail.canonical_aspect_ratio if self.config.headtail else 2.0
        )
        mg = self.config.headtail.canonical_margin if self.config.headtail else 1.3
        canonical_crops = extract_canonical_crops(
            frame, filtered_obb, ar, mg, self.runtime
        )
        aabb_crops = (
            extract_aabb_crops(
                frame, filtered_obb, padding=self.config.apriltag.crop_padding
            )
            if self._models.apriltag
            else []
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            ht_fut = (
                pool.submit(
                    run_headtail,
                    canonical_crops,
                    filtered_obb,
                    self._models.headtail,
                    self.config.headtail,
                    self.runtime,
                )
                if self._models.headtail
                else None
            )
            cnn_futs = [
                pool.submit(
                    run_cnn,
                    canonical_crops,
                    filtered_obb,
                    mdl,
                    cfg,
                    self.runtime,
                )
                for cfg, mdl in zip(self.config.cnn_phases, self._models.cnn)
            ]
            pose_fut = (
                pool.submit(
                    run_pose,
                    canonical_crops,
                    filtered_obb,
                    self._models.pose,
                    self.config.pose,
                    self.runtime,
                )
                if self._models.pose
                else None
            )
            at_fut = (
                pool.submit(
                    run_apriltag,
                    aabb_crops,
                    filtered_obb,
                    self._models.apriltag,
                    self.config.apriltag,
                )
                if self._models.apriltag
                else None
            )
            ht_result = ht_fut.result() if ht_fut else None
            cnn_results = [f.result() for f in cnn_futs]
            pose_result = pose_fut.result() if pose_fut else None
            at_result = at_fut.result() if at_fut else None

        frame_result = _build_frame_result(
            0,
            filtered_obb,
            det_indices,
            ht_result,
            cnn_results,
            pose_result,
            at_result,
        )

        # Task 17g: build StreamingAnalysisPayload for legacy identity consumers.
        try:
            from hydra_suite.core.tracking.streaming_payload import (
                StreamingAnalysisPayload,
            )

            frame_result.streaming_payload = StreamingAnalysisPayload.from_frame_result(
                frame_result,
                runtime_family=str(self.runtime.default_runtime),
                input_is_bgr=True,
            )
        except Exception:
            pass  # streaming_payload is optional; failures are non-fatal

        return frame_result

    def run_batch_pass(self, video_path: Path, progress_cb=None) -> None:
        import cv2

        if self.cache_dir is None:
            raise RuntimeError("cache_dir must be set before calling run_batch_pass")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        caches = _open_caches(self.config, self.cache_dir)
        self._caches = caches
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_size = self.config.detection_batch_size

        frames_buf: list[np.ndarray] = []
        indices_buf: list[int] = []
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_buf.append(frame)
                indices_buf.append(processed)
                processed += 1
                if len(frames_buf) == batch_size:
                    self._run_batch(frames_buf, indices_buf, caches)
                    frames_buf.clear()
                    indices_buf.clear()
                if (
                    progress_cb
                    and total_frames > 0
                    and processed % max(1, total_frames // 100) == 0
                ):
                    progress_cb(processed, total_frames)
            if frames_buf:
                self._run_batch(frames_buf, indices_buf, caches)
            if progress_cb:
                progress_cb(processed, total_frames)
        finally:
            cap.release()
            for h in caches.all_handles():
                h.close()

    def _run_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        caches: _CacheSet,
    ) -> None:
        ar = (
            self.config.headtail.canonical_aspect_ratio if self.config.headtail else 2.0
        )
        mg = self.config.headtail.canonical_margin if self.config.headtail else 1.3

        # OBB still runs cross-frame natively
        raw_list = run_obb(frames, self._models.obb, self.config.obb, self.runtime)

        for frame, frame_idx, raw in zip(frames, frame_indices, raw_list):
            obb_result = (
                materialize_tensors(raw) if isinstance(raw, _RawOBBTensors) else raw
            )
            # Re-stamp detection_ids for this frame
            obb_result = OBBResult(
                frame_idx=frame_idx,
                centroids=obb_result.centroids,
                angles=obb_result.angles,
                sizes=obb_result.sizes,
                shapes=obb_result.shapes,
                confidences=obb_result.confidences,
                corners=obb_result.corners,
                detection_ids=OBBResult.make_detection_ids(
                    frame_idx, obb_result.num_detections
                ),
            )
            if caches.detection is not None:
                caches.detection.write_frame(frame_idx, result=obb_result)

            filtered_obb, det_indices = filter_with_indices(obb_result, self.config.obb)
            if filtered_obb.num_detections == 0:
                continue

            canonical_crops = extract_canonical_crops(
                frame, filtered_obb, ar, mg, self.runtime
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                ht_fut = (
                    pool.submit(
                        run_headtail,
                        canonical_crops,
                        filtered_obb,
                        self._models.headtail,
                        self.config.headtail,
                        self.runtime,
                    )
                    if self._models.headtail is not None
                    else None
                )
                cnn_futs = [
                    pool.submit(
                        run_cnn, canonical_crops, filtered_obb, mdl, cfg, self.runtime
                    )
                    for cfg, mdl in zip(self.config.cnn_phases, self._models.cnn)
                ]
                pose_fut = (
                    pool.submit(
                        run_pose,
                        canonical_crops,
                        filtered_obb,
                        self._models.pose,
                        self.config.pose,
                        self.runtime,
                    )
                    if self._models.pose is not None
                    else None
                )
                ht_result = ht_fut.result() if ht_fut else None
                cnn_results = [f.result() for f in cnn_futs]
                pose_result = pose_fut.result() if pose_fut else None

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

            if self._models.apriltag is not None:
                aabb_crops = extract_aabb_crops(
                    frame, filtered_obb, padding=self.config.apriltag.crop_padding
                )
                at_result = run_apriltag(
                    aabb_crops,
                    filtered_obb,
                    self._models.apriltag,
                    self.config.apriltag,
                )
                if caches.apriltag is not None and at_result is not None:
                    caches.apriltag.write_frame(frame_idx, result=at_result)

    def load_frame(self, frame_idx: int) -> FrameResult:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir not set — cannot load cached frames")
        if self._caches is None:
            self._caches = _open_caches(self.config, self.cache_dir)

        raw_obb = (
            self._caches.detection.read_frame(frame_idx)
            if self._caches.detection is not None
            else None
        )
        if raw_obb is None:
            raise KeyError(f"Frame {frame_idx} not found in detection cache")

        filtered_obb, det_indices = filter_with_indices(raw_obb, self.config.obb)

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
        self._models.obb.close()
        if self._models.headtail is not None:
            self._models.headtail.close()
        for mdl in self._models.cnn:
            mdl.close()
        if self._models.pose is not None:
            self._models.pose.close()
        if self._models.apriltag is not None:
            self._models.apriltag.close()
