from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hydra_suite.core.canonicalization.geometry import ClippingStats
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span

if TYPE_CHECKING:
    from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
    from hydra_suite.core.individual.identity.catalog import IdentityCatalog

    from .config import PoseConfig
    from .identity_evidence_config import IdentityEvidenceRunConfig
    from .stages.identity_evidence import IdentityEvidenceStage

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
from .runtime import RuntimeContext, resolved_backend_for
from .stages.apriltag import AprilTagModel, run_apriltag
from .stages.bgsub import BgSubModel, run_bgsub
from .stages.cnn import CNNModel, run_cnn
from .stages.crops import extract_aabb_crops, extract_canonical_crops
from .stages.filtering import filter_for_source
from .stages.headtail import HeadTailModel, run_headtail
from .stages.obb import OBBModels, _RawOBBTensors, materialize_tensors, run_obb
from .stages.pose import PoseModel, run_pose

logger = logging.getLogger(__name__)


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


def _sliced_tile_batch(
    config: InferenceConfig, frame_hw: tuple[int, int], imgsz: int
) -> int:
    """Exact tiles-per-frame for the configured slice plan (+1 for full-frame pass).

    Bounded by ``slicing.MAX_TILE_CHUNK`` — the SAME cap the sliced path chunks
    its predict calls with, so the exported engine profile always covers the
    largest batch that will actually be issued.
    """
    from .stages.slicing import MAX_TILE_CHUNK, plan_slices

    slice_cfg = config.obb.direct.slice
    plan = plan_slices(
        frame_hw,
        slice_cfg,
        imgsz,
        # Deliberately UNGATED (roi_mask=None): this sizes the TensorRT dynamic-
        # batch engine profile and must cover the largest tile chunk that can
        # ever be issued. ROI gating only ever reduces the tile count (and falls
        # back to the full grid on an empty ROI), so the ungated count is the
        # correct upper bound; gating here could undersize the engine profile.
        None,
        ref_object_px=slice_cfg.reference_body_px,
    )
    return max(1, min(plan.jobs_per_frame, MAX_TILE_CHUNK))


def _probe_frame_hw(video_path: str | None) -> tuple[int, int] | None:
    """Read (height, width) off the video's first frame metadata, or None."""
    if not video_path:
        return None
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if h <= 0 or w <= 0:
            return None
        return (h, w)
    except Exception:
        return None
    finally:
        cap.release()


def _probe_model_imgsz(model_path: str | None) -> int | None:
    """Resolve the model's square input size, or None on failure."""
    if not model_path:
        return None
    from .runtime_artifacts import _resolve_imgsz

    try:
        return _resolve_imgsz(Path(model_path))
    except Exception:
        return None


def _load_obb_for_config(
    config: InferenceConfig,
    runtime: RuntimeContext,
    video_path: str | None = None,
) -> OBBModels:
    """Load OBB models, sizing the TRT engine batch from the real tile count.

    With slicing on, the model is fed TILE batches, not frame batches, so the
    engine's dynamic profile must cover tiles-per-chunk (spec 5c) — otherwise
    TensorRT fails ``setInputShape`` at runtime. When slicing is disabled this
    is a no-op passthrough: ``config.detection_batch_size`` unchanged, exactly
    as before this helper existed.
    """
    from .stages.obb import load_obb_models

    batch_size = config.detection_batch_size
    direct = config.obb.direct if config.obb is not None else None
    slice_cfg = getattr(direct, "slice", None) if direct is not None else None
    if slice_cfg is not None and slice_cfg.enabled:
        frame_hw = _probe_frame_hw(video_path)
        imgsz = _probe_model_imgsz(direct.model_path)
        if frame_hw is not None and imgsz:
            batch_size = max(batch_size, _sliced_tile_batch(config, frame_hw, imgsz))
        else:
            logger.warning(
                "Sliced OBB inference enabled but frame size (%s) and/or model "
                "imgsz (%s) could not be probed at load time; falling back to "
                "detection_batch_size=%d for the TRT engine profile. If the "
                "runtime tier is gpu_fast, the exported engine's dynamic-batch "
                "profile may not cover the real tile-chunk size and inference "
                "may fail at setInputShape — a manually sized engine export "
                "may be required.",
                frame_hw,
                imgsz,
                batch_size,
            )
    return load_obb_models(config.obb, runtime, batch_size=batch_size)


def _pose_config_model_path(pose_config: PoseConfig) -> str:
    """The active backend's checkpoint path for ``pose_config``, or "" if unset."""
    if pose_config.backend == "yolo" and pose_config.yolo is not None:
        return pose_config.yolo.model_path
    if pose_config.backend == "sleap" and pose_config.sleap is not None:
        return pose_config.sleap.model_path
    if pose_config.backend == "vitpose" and pose_config.vitpose is not None:
        return pose_config.vitpose.model_path
    return ""


def _warn_geometry_mismatch(model_path: str, session_geometry) -> None:
    """F2 guard: log ``warn_on_geometry_mismatch``'s message, if any.

    ``warn_on_geometry_mismatch`` (core/inference/canonical_meta.py) had no
    production caller before this -- the model-side provenance stamp existed
    but nothing ever consulted it, so a model trained under a different
    canonical geometry than the current session loaded silently. Called once
    per model at load time, here, the single place every stage's model path
    and the session's ``CanonicalGeometry`` are both already in scope.
    """
    if not model_path:
        return
    from .canonical_meta import warn_on_geometry_mismatch

    message = warn_on_geometry_mismatch(model_path, session_geometry)
    if message:
        logger.warning(message)


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
    from .stages.pose import load_pose_model

    obb = None
    bgsub = None
    if config.detection_source == "obb":
        obb = _load_obb_for_config(config, runtime, video_path=video_path)
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
    if config.headtail is not None:
        _warn_geometry_mismatch(config.headtail.model_path, config.canonical)

    cnn = [load_cnn_model(c, runtime) for c in config.cnn_phases]
    for _cnn_cfg in config.cnn_phases:
        _warn_geometry_mismatch(_cnn_cfg.model_path, config.canonical)

    pose = load_pose_model(config.pose, runtime) if config.pose is not None else None
    if config.pose is not None:
        _pose_model_path = _pose_config_model_path(config.pose)
        if _pose_model_path:
            _warn_geometry_mismatch(_pose_model_path, config.canonical)

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
    config: InferenceConfig,
    cache_dir: Path,
    video_sig: str = "",
    roi_mask: "np.ndarray | None" = None,
) -> _CacheSet:
    # Bind every per-video cache to the exact source file so a changed video
    # (e.g. a clip regenerated under the same name with a different frame count)
    # invalidates the cache instead of serving stale, truncated detections.
    def _k(key):
        return with_video_signature(key, video_sig)

    detection_key = (
        # roi_mask is folded into the OBB key ONLY when slicing is enabled (see
        # detection_cache_key); None / disabled slicing => byte-identical key.
        detection_cache_key(config.obb, roi_mask)
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
                key=_k(headtail_cache_key(config.headtail, config.canonical)),
            )
            if config.headtail is not None
            else None
        ),
        cnn=[
            CNNCacheHandle(
                path=cache_dir / f"cnn_{c.label}.npz",
                key=_k(cnn_cache_key(c, config.canonical)),
                label=c.label,
            )
            for c in config.cnn_phases
        ],
        pose=(
            PoseCacheHandle(
                path=cache_dir / "pose.npz",
                key=_k(pose_cache_key(config.pose, config.canonical)),
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


def _build_identity_evidence_stage(
    identity_evidence: "IdentityEvidenceRunConfig",
) -> tuple["IdentityCatalog", "IdentityEvidenceStage"]:
    """Build the (catalog, stage) pair for one resolved identity-evidence config.

    One ``EvidenceBuilder`` per CNN phase, keyed by the same phase label used
    for ``_CacheSet.cnn`` / the ``cnn_reads`` dict passed to
    ``IdentityEvidenceStage.evidences_for_frame`` -- Task 3's "unmatched key"
    skip only ever triggers on a genuinely absent phase, never a naming
    mismatch introduced here.

    Catalog basis (final-fix wave, CRITICAL): each CNN phase's
    ``EvidenceBuilder`` is built against that phase's OWN phase-local
    cartesian catalog (``build_phase_catalog_labels``), not the shared
    global catalog. This reproduces the old tracking-time
    ``IdentityEvidenceEmitter``'s per-source catalog basis: a phase with
    fewer reachable labels than the global catalog (CNN+AprilTag configs, or
    multi-CNN-phase configs where each phase only covers its own labels)
    never sees "phase-unreachable" global entries floored to the builder's
    internal ``1e-6`` -- those entries simply do not exist in the phase
    catalog. The global catalog is still built here (and used for AprilTag
    evidence, which is already global-basis). The tracking worker's
    ``_remap_source_log_probs_to_catalog`` (``core/tracking/worker.py``)
    then remaps each phase's evidence from its phase basis to the global
    catalog with the SAME ``1e-300`` floor + renormalize the old emitter
    path relied on, via ``IdentityEvidenceStage.catalog_labels_by_source``
    persisted alongside the sidecar.
    """
    from hydra_suite.core.individual.identity.catalog import IdentityCatalog
    from hydra_suite.core.individual.identity.evidence_builder import (
        EvidenceBuilder,
        build_phase_catalog_labels,
    )

    from .stages.identity_evidence import IdentityEvidenceStage

    catalog = IdentityCatalog.from_spec(identity_evidence.catalog_spec)
    cnn_builders = {}
    for phase in identity_evidence.cnn_phases:
        phase_catalog = IdentityCatalog(
            labels=build_phase_catalog_labels(phase.class_names_per_factor)
        )
        cnn_builders[phase.label] = EvidenceBuilder(
            phase_catalog,
            phase.label,
            phase.class_names_per_factor,
            calibration=phase.calibration,
            calibration_signature=phase.calibration_signature,
            runtime_signature=identity_evidence.runtime_signature,
        )
    stage = IdentityEvidenceStage(catalog, cnn_builders, identity_evidence.tag_to_label)
    return catalog, stage


def write_identity_evidence_sidecar(
    caches: "_CacheSet",
    config: InferenceConfig,
    stage: "IdentityEvidenceStage",
    frame_range: "range",
    out_path: Path,
    catalog_labels: "tuple[str, ...]",
) -> None:
    """Read back raw per-frame caches over `frame_range` and write the evidence sidecar.

    The batch seam Task 4 wires into ``run_batch_pass``: for each frame, reads
    the raw (pre-filter) detection cache, re-derives the SAME filtered
    detection set + order the pipeline used when it ran HeadTail/CNN/AprilTag
    for that frame (``filter_for_source(config, raw_obb)`` -- deterministic,
    no ``roi_mask``, exactly mirroring ``Pipeline._process_window``'s
    ``filter_for_source(cfg, obb_result)`` call and ``InferenceRunner.load_frame``'s
    own re-filter), and uses the resulting ``filtered_obb.detection_ids`` as the
    stable ``det_ids`` list `IdentityEvidenceStage` expects -- aligned by
    position with the CNN/AprilTag stages' own ``det_index`` (both are
    sequential 0..N-1 over that SAME filtered set, since CNN/AprilTag ran
    against the pipeline's filtered_obb, not the raw one).

    ``caches`` must already be flushed to disk (``read_frame`` is disk-backed
    only -- it never sees an unflushed in-memory write buffer), so this is
    called with a cache set whose handles were opened AFTER the raw caches
    were closed (either freshly reopened via ``_open_caches``, or the same
    handles post-``close()``).
    """
    from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache

    evidence_cache = IdentityEvidenceCache(
        out_path,
        catalog_labels=catalog_labels,
        mode="w",
        catalog_labels_by_source=stage.catalog_labels_by_source,
    )
    if caches.detection is None:
        evidence_cache.flush()
        return

    cnn_caches = list(caches.cnn)
    for frame_idx in frame_range:
        raw_obb = caches.detection.read_frame(frame_idx)
        if raw_obb is None:
            continue
        filtered_obb, _ = filter_for_source(config, raw_obb)
        if filtered_obb.num_detections == 0:
            continue
        det_ids = [int(d) for d in filtered_obb.detection_ids]

        cnn_reads: dict[str, list] = {}
        for cnn_cache in cnn_caches:
            preds = cnn_cache.read_frame(frame_idx)
            if preds:
                cnn_reads[cnn_cache.label] = preds

        tag_read = (
            caches.apriltag.read_frame(frame_idx)
            if caches.apriltag is not None
            else None
        )

        evidences = stage.evidences_for_frame(frame_idx, det_ids, cnn_reads, tag_read)
        if evidences:
            evidence_cache.save_frame(frame_idx, evidences)

    evidence_cache.flush()


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
        roi_mask: "np.ndarray | None" = None,
        identity_evidence: "IdentityEvidenceRunConfig | None" = None,
    ) -> None:
        from hydra_suite.utils.profiling_process import maybe_arm_process_recorder

        maybe_arm_process_recorder()

        self.config = config
        self.cache_dir = cache_dir
        self.cache_only = cache_only
        # Arena ROI mask for sliced-inference tile gating. It is the single
        # source of truth for BOTH (a) the detection cache key (folded in only
        # when slicing is enabled AND this is non-None -- see detection_cache_key)
        # and (b) frame-space tile gating during the batch pass. Setting it at
        # construction (rather than only per-pass) is what lets a SEPARATE
        # backward/replay run reproduce the exact same cache key via
        # caches_all_valid() and read the forward run's cache.
        self._roi_mask = roi_mask
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
        # Identity Phase 3, Task 4: when set, both run_realtime and
        # run_batch_pass write an IdentityEvidence sidecar during the
        # inference pass, ahead of tracking (Task 5 flips the tracker to read
        # it). None (no identity configured) is a strict no-op -- neither pass
        # touches identity evidence at all.
        self._identity_evidence = identity_evidence
        self._identity_catalog: "IdentityCatalog | None" = None
        self._identity_stage: "IdentityEvidenceStage | None" = None
        if identity_evidence is not None:
            self._identity_catalog, self._identity_stage = (
                _build_identity_evidence_stage(identity_evidence)
            )
        # Realtime-only: the evidence sidecar for the live/streaming pass,
        # opened lazily on the first frame that has caches to write into
        # (mirrors self._caches' own lazy-open-for-writing pattern) and
        # flushed once in close().
        self._identity_evidence_cache = None
        # Run-scoped: counts detections clipped by the fixed canonical canvas
        # and the worst overflow_ratio seen, across the life of this runner
        # (one tracking pass). See ClippingStats; surfaced by the caller (e.g.
        # TrackingWorker) in its end-of-run summary alongside the other
        # tracking-loop counters.
        self.clipping_stats = ClippingStats()

    @property
    def obb_class_names(self) -> "dict[int, str] | None":
        """id->name map from the loaded OBB model, for label display. None if no OBB model."""
        if self._models.obb is None:
            return None
        # direct mode uses direct_model; sequential's classes come from the stage-2 obb_model
        model = self._models.obb.direct_model or self._models.obb.obb_model
        names = getattr(model, "names", None)
        if names is None:
            return None
        # normalize to dict[int,str] (ultralytics .names may be a dict or list)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {int(i): str(v) for i, v in enumerate(names)}

    def caches_all_valid(self) -> bool:
        if self.cache_dir is None:
            return False
        caches = _open_caches(
            self.config, self.cache_dir, self._video_sig, self._roi_mask
        )
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
        caches = _open_caches(
            self.config, self.cache_dir, self._video_sig, self._roi_mask
        )
        if caches.detection is None:
            return False
        return caches.detection.covers_frame_range(start_frame, end_frame)

    def detection_cache_missing_frames(
        self, start_frame: int, end_frame: int, max_report: int = 10
    ) -> list[int]:
        """Report up to ``max_report`` frames missing from the detection cache."""
        if self.cache_dir is None:
            return []
        caches = _open_caches(
            self.config, self.cache_dir, self._video_sig, self._roi_mask
        )
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
        with span(N.REALTIME):
            # Lazily open the caches for WRITING so the realtime forward pass persists
            # detections + downstream results. Backward tracking replays them via
            # load_frame; without this, realtime + backward gets an empty backward pass.
            if self._caches is None and self.cache_dir is not None:
                self._caches = _open_caches(
                    self.config, self.cache_dir, self._video_sig, self._roi_mask
                )
                self._caches_writable = True
            caches = self._caches if self._caches_writable else None

            with span(N.RT_OBB, units=1):
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
                    # roi_mask is frame-space (the caller passes the mask matching this
                    # exact frame's geometry); it enables ROI tile gating on the sliced
                    # path and is ignored on the non-sliced path (see run_obb).
                    raw_list = run_obb(
                        [frame],
                        self._models.obb,
                        self.config.obb,
                        self.runtime,
                        roi_mask=roi_mask,
                    )
                    raw = raw_list[0]
                    if isinstance(raw, _RawOBBTensors):
                        raw_obb = materialize_tensors(
                            raw, self.config.obb.raw_detection_cap
                        )
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

            with span(N.RT_FILTER):
                filtered_obb, det_indices = filter_for_source(
                    self.config, raw_obb, roi_mask
                )

            if filtered_obb.num_detections == 0:
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

            with span(N.RT_CROPS, units=filtered_obb.num_detections):
                geometry = self.config.canonical
                # F1 guard: every detection that will be canonicalized by ANY consumer
                # (headtail, cnn, pose all warp through this one geometry -- see the
                # comment below) gets its overflow_ratio recorded here, once, in the
                # single place that already has both `filtered_obb` and `geometry` in
                # scope -- rather than duplicating this at each of the many internal
                # canonical_affine call sites (which would double-count a detection
                # once per consumer stage).
                if (
                    self._models.headtail is not None
                    or self._models.cnn
                    or self._models.pose is not None
                ):
                    for _corners in filtered_obb.corners:
                        self.clipping_stats.record(_corners, geometry)
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
                # PoseConfig.background_color was deleted: it was never populated by
                # from_parameters (always (0, 0, 0)), a dead second home for the fill
                # colour. Zero is now the one honest fill value everywhere.
                background_color = (0, 0, 0)
                canonical_crops = (
                    extract_canonical_crops(
                        frame,
                        filtered_obb,
                        geometry,
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

            with span(N.RT_INDIVIDUAL, units=filtered_obb.num_detections):

                def _do_ht() -> HeadTailResult | None:
                    if not self._models.headtail:
                        return None
                    return run_headtail(
                        frame,
                        filtered_obb,
                        self._models.headtail,
                        self.config.headtail,
                        self.runtime,
                        geometry,
                    )

                def _do_cnn() -> list[CNNResult]:
                    return [
                        run_cnn(frame, filtered_obb, mdl, cfg, self.runtime, geometry)
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
                        geometry,
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

            with span(N.RT_CACHE):
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
                            cache.write_frame(
                                frame_idx, predictions=cnn_result.predictions
                            )
                    if caches.pose is not None and pose_result is not None:
                        caches.pose.write_frame(
                            frame_idx,
                            det_indices=det_indices,
                            keypoints=pose_result.keypoints,
                            valid_mask=pose_result.valid_mask,
                        )
                    if caches.apriltag is not None and at_result is not None:
                        caches.apriltag.write_frame(frame_idx, result=at_result)

                # Identity Phase 3, Task 4 (realtime seam): build + persist this
                # frame's identity evidence inline, from the SAME in-hand
                # filtered_obb/cnn_results/at_result -- no read-back needed (unlike
                # the batch seam, which re-derives det_ids from a disk read-back
                # after the pass). Identical evidence contract to the batch path:
                # det_ids come from filtered_obb.detection_ids (stable ids, aligned
                # by position with CNN/AprilTag det_index, both 0..N-1 over this same
                # filtered_obb). Only runs when caches are open for writing -- a pure
                # in-memory/preview realtime call (cache_dir=None) writes nothing.
                if caches is not None and self._identity_stage is not None:
                    self._write_identity_evidence_realtime(
                        frame_idx, filtered_obb, cnn_results, at_result
                    )

            with span(N.RT_FINALIZE):
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
                if (
                    self.config.detection_source == "bgsub"
                    and self._models.bgsub is not None
                ):
                    frame_result.fg_mask = self._models.bgsub.last_fg_mask
                    frame_result.bg_u8 = self._models.bgsub.last_bg_u8

                # Task 17g: build StreamingAnalysisPayload for legacy identity consumers.
                try:
                    from hydra_suite.core.tracking.ingest.streaming_payload import (
                        StreamingAnalysisPayload,
                    )

                    resolved = resolved_backend_for(self.runtime)
                    if resolved.backend == "tensorrt":
                        runtime_family = "tensorrt"
                    elif resolved.backend == "coreml":
                        runtime_family = "coreml"
                    else:
                        runtime_family = resolved.device
                    frame_result.streaming_payload = (
                        StreamingAnalysisPayload.from_frame_result(
                            frame_result,
                            runtime_family=runtime_family,
                            input_is_bgr=True,
                        )
                    )
                except Exception:
                    pass  # streaming_payload is optional; failures are non-fatal

                return frame_result

    def _identity_evidence_sidecar_path(self, source_name: str) -> Path:
        """`<cache_dir>/detection.npz`-based sidecar path for `source_name` ("batch"/"live").

        The signature slot is the Task 1 content hash (catalog + per-phase
        calibration temps + this run's video signature) rather than a bare
        pass-name string, so a catalog or calibration change invalidates only
        the sidecar -- never the raw detection/CNN/AprilTag caches, whose keys
        do not carry identity information at all.
        """
        from hydra_suite.core.tracking.identity.evidence_emitter import (
            build_evidence_cache_path,
        )

        from .identity_evidence_key import identity_evidence_cache_key

        assert self._identity_evidence is not None  # caller-guaranteed
        key = identity_evidence_cache_key(
            self._identity_evidence.catalog_spec,
            self._identity_evidence.per_factor_temps(),
            self._video_sig,
        )
        return build_evidence_cache_path(
            str(self.cache_dir / "detection.npz"), source_name, key
        )

    def identity_evidence_sidecar_path(self, source_name: str) -> "Path | None":
        """Public accessor: where the ``source_name`` ("batch"/"live") identity
        evidence sidecar is/will be written, or ``None`` when this runner has no
        identity-evidence config. Lets read-side consumers (the tracking worker)
        locate the sidecar this runner writes without recomputing the Task-1
        content-hash key themselves.
        """
        if self._identity_evidence is None:
            return None
        return self._identity_evidence_sidecar_path(source_name)

    @property
    def identity_evidence_cache(self) -> "IdentityEvidenceCache | None":
        """The realtime (write-mode) identity evidence cache, or ``None``.

        ``IdentityEvidenceCache.load_frame`` reads straight from its in-memory
        buffer regardless of read/write mode, so this lets the tracking loop
        read a just-written frame's evidence back before ``close()``/flush.
        """
        return self._identity_evidence_cache

    def _write_identity_evidence_realtime(
        self,
        frame_idx: int,
        filtered_obb: OBBResult,
        cnn_results: list[CNNResult],
        at_result: AprilTagResult | None,
    ) -> None:
        from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache

        if self._identity_evidence_cache is None:
            self._identity_evidence_cache = IdentityEvidenceCache(
                self._identity_evidence_sidecar_path("live"),
                catalog_labels=self._identity_catalog.labels,
                mode="w",
                catalog_labels_by_source=self._identity_stage.catalog_labels_by_source,
            )

        det_ids = [int(d) for d in filtered_obb.detection_ids]
        cnn_reads = {
            cnn_result.label: cnn_result.predictions
            for cnn_result in cnn_results
            if cnn_result is not None and cnn_result.predictions
        }
        evidences = self._identity_stage.evidences_for_frame(
            frame_idx, det_ids, cnn_reads, at_result
        )
        if evidences:
            self._identity_evidence_cache.save_frame(frame_idx, evidences)

    def _write_identity_evidence_batch(self, start_frame: int, end_frame: int) -> None:
        """Batch seam: read back the just-flushed raw caches, write the sidecar.

        Called AFTER `run_batch_pass`'s caches are closed (flushed to disk) --
        `CacheHandle.read_frame` is disk-backed only, so a read-back against
        still-buffered (unflushed) writes would see nothing. Opens a fresh
        `_CacheSet` (read-only use; no key/`is_valid()` surprises from reusing
        already-closed write handles).
        """
        if self._identity_evidence is None or self.cache_dir is None:
            return
        read_caches = _open_caches(
            self.config, self.cache_dir, self._video_sig, self._roi_mask
        )
        out_path = self._identity_evidence_sidecar_path("batch")
        write_identity_evidence_sidecar(
            read_caches,
            self.config,
            self._identity_stage,
            range(start_frame, end_frame + 1),
            out_path,
            self._identity_catalog.labels,
        )

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

    def _frame_space_roi_mask(
        self, video_path: str | Path | None
    ) -> "np.ndarray | None":
        """Resample ``self._roi_mask`` to the video's native frame geometry.

        Batch-pass frames are decoded at native video resolution (no resize), so
        the mask handed to ``plan_slices`` must be in that exact H x W space. When
        the frame size cannot be probed, or the mask already matches, the mask is
        returned unchanged -- and ``plan_slices``' own coordinate-space guard is
        the final safety net (a shape mismatch degrades to no gating, never a
        mis-gate).
        """
        mask = self._roi_mask
        if mask is None:
            return None
        frame_hw = _probe_frame_hw(str(video_path) if video_path else None)
        if frame_hw is None or mask.shape[:2] == frame_hw:
            return mask
        import cv2

        return cv2.resize(
            mask,
            (frame_hw[1], frame_hw[0]),  # cv2 wants (w, h)
            interpolation=cv2.INTER_NEAREST,
        )

    def _build_pipeline(
        self, caches: _CacheSet, roi_mask: "np.ndarray | None" = None
    ) -> Pipeline:
        """Construct the depth=1 Pipeline that drives the batch stage layer.

        The Pipeline owns the per-window stage sequence (OBB → crops → HT/CNN/pose
        → AprilTag → scatter); cache writes go through a ``CacheWriter`` (sync mode)
        that reproduces ``_run_batch``'s exact raw-result side effects.

        ``roi_mask`` (frame-space) is threaded onto ``PipelineStages`` so the OBB
        stage can ROI-gate slice tiles; ``None`` keeps the full tile grid.
        """
        stages = PipelineStages(
            config=self.config,
            obb_models=self._models.obb,
            bgsub_model=self._models.bgsub,
            headtail_model=self._models.headtail,
            cnn_models=self._models.cnn,
            pose_model=self._models.pose,
            apriltag_model=self._models.apriltag,
            roi_mask=roi_mask,
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
        return Pipeline(
            stages,
            self.runtime,
            writer,
            depth=self.config.pipeline_depth,
            clipping_stats=self.clipping_stats,
        )

    def run_batch_pass(
        self,
        video_path: Path,
        progress_cb=None,
        start_frame: int = 0,
        end_frame: int | None = None,
        should_stop=None,
        roi_mask: "np.ndarray | None" = None,
    ) -> None:
        from .sources import make_frame_source

        if self.cache_dir is None:
            raise RuntimeError("cache_dir must be set before calling run_batch_pass")
        with span(N.INFERENCE), span(N.BATCH_PASS):

            # An explicit roi_mask overrides the construction-time one so the cache
            # key (opened below) and the tile gating both use the same mask -- and so
            # a separate backward run built with the same construction-time mask
            # reproduces the identical key. Passing it only at construction is the
            # recommended path; this override keeps the two in lockstep either way.
            if roi_mask is not None:
                self._roi_mask = roi_mask

            # make_frame_source selects NvdecFrameReader when runtime.use_nvdec is True
            # and the decoder is available; otherwise falls back to CpuFrameReader.
            # Clamping and seeking are handled inside each reader implementation.
            frame_source = make_frame_source(
                video_path, self.runtime, start_frame, end_frame
            )

            with span(N.OPEN_CACHES):
                caches = _open_caches(
                    self.config, self.cache_dir, self._video_sig, self._roi_mask
                )
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
            # Resample the ROI mask to the native frame geometry for tile gating
            # (the cache key above already folded the mask by content, independent
            # of this resample).
            pipeline = self._build_pipeline(
                caches, roi_mask=self._frame_space_roi_mask(video_path)
            )
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

            # Identity Phase 3, Task 4 (batch seam): write the evidence sidecar
            # AFTER the raw caches above are flushed to disk -- and only on a
            # successful pass (an exception in `pipeline.run` propagates out of
            # the `try/finally` above and this line is never reached, matching
            # the raw caches' own "no sidecar from a failed pass" behavior).
            # `_write_identity_evidence_batch` is itself a no-op when no identity
            # config was passed to this runner.
            self._write_identity_evidence_batch(start_frame, end_frame)

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
            self._caches = _open_caches(
                self.config, self.cache_dir, self._video_sig, self._roi_mask
            )

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
        # Flush the realtime identity-evidence sidecar (Task 4), if any frame
        # ever wrote to it. Mirrors the raw caches' write-mode-only flush
        # above: this cache is only ever opened in mode="w" by
        # _write_identity_evidence_realtime.
        if self._identity_evidence_cache is not None:
            self._identity_evidence_cache.flush()
            self._identity_evidence_cache = None
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
