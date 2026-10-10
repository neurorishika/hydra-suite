from __future__ import annotations

import logging
import os
import shutil
import uuid
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
    from hydra_suite.core.inference.autotune.models import InferenceRuntimeOverlay

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
    replay_filter_hash,
    video_signature,
    with_replay_filters,
    with_video_signature,
)
from .cache.set_manifest import (
    CACHE_SET_FILENAME,
    load_cache_set,
    publish_cache_set,
    publish_compatibility_links,
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
from .cancellation import InferenceCancelled
from .config import InferenceConfig
from .downstream_select import (
    DownstreamCacheError,
    apriltag_raw_to_positions,
    cnn_raw_to_positions,
    positions_in,
)
from .limits import MAX_DETECTIONS_PER_FRAME, DetectionLimitStats
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
from .stages.crops import ForeignSet, extract_aabb_crops, extract_canonical_crops
from .stages.filtering import filter_for_source
from .stages.headtail import HeadTailModel, run_headtail
from .stages.obb import (
    OBBModels,
    _RawOBBTensors,
    effective_raw_detection_cap,
    materialize_tensors,
    rank_and_bound,
    run_obb,
)
from .stages.pose import PoseModel, run_pose

logger = logging.getLogger(__name__)


def _clone_cache_revision(source: Path, destination: Path) -> None:
    """Hard-link one revision using O(directory-depth) Python memory."""
    destination.mkdir(parents=True, exist_ok=False)
    with os.scandir(source) as entries:
        for entry in entries:
            source_entry = Path(entry.path)
            destination_entry = destination / entry.name
            if entry.is_symlink():
                raise ValueError("cache revision must not contain symbolic links")
            if entry.is_dir(follow_symlinks=False):
                _clone_cache_revision(source_entry, destination_entry)
            elif entry.is_file(follow_symlinks=False):
                os.link(source_entry, destination_entry)
            else:
                raise ValueError("cache revision contains an unsupported entry")


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
    cache_dir: Path | None = None
    generation_id: str | None = None
    revision_id: str | None = None
    member_names: list[str] = field(default_factory=list)
    pending_promotion: bool = False
    discard_if_unchanged: bool = False
    staging_root: Path | None = None
    set_manifest_valid: bool = True

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

    def close(self, *, complete: bool = True) -> None:
        """Close members, then expose a prepared generation with one rename."""
        handles = self.all_handles()
        for handle in handles:
            handle.close(commit_generation=complete)
        if not complete or not self.pending_promotion:
            return
        if self.discard_if_unchanged and not any(
            getattr(handle, "_write_started", False) for handle in handles
        ):
            if self.staging_root is not None:
                shutil.rmtree(self.staging_root)
            self.pending_promotion = False
            return
        if (
            self.cache_dir is None
            or self.generation_id is None
            or self.revision_id is None
        ):
            raise RuntimeError(
                "cache set cannot be promoted without generation metadata"
            )
        if not handles or not all(handle.is_reusable() for handle in handles):
            raise RuntimeError("refusing to promote an incomplete cache generation")
        reference = handles[0].coverage_ranges()
        if any(handle.coverage_ranges() != reference for handle in handles):
            raise RuntimeError("refusing to promote a mixed-coverage cache generation")
        if any(handle._store.generation_id != self.generation_id for handle in handles):
            raise RuntimeError("refusing to promote mixed cache generations")
        manifest = publish_cache_set(
            self.cache_dir,
            self.generation_id,
            self.revision_id,
            self.member_names,
        )
        publish_compatibility_links(self.cache_dir, manifest)
        self.pending_promotion = False


def _admitted_sliced_tile_batch(
    *,
    slice_cfg: Any,
    frame_hw: tuple[int, int],
    imgsz: int,
    task: str,
    max_detections: int,
    device_tiles: bool = False,
) -> int:
    """Largest resource-admissible tile chunk for a sliced direct OBB run.

    The TensorRT engine profile must cover the largest *predict call*, not all
    jobs in a frame.  The streaming sliced path applies the same request cap,
    memory estimator, budget, and hard limit below.  Keeping the calculation
    identical prevents an engine from being optimized for an impossible 36+
    tile batch when execution will only ever submit a smaller admitted chunk.

    ``device_tiles`` describes source residency when it is known: CPU/numpy
    tiles include their contiguous crop allocation, whereas CUDA views do not.
    """
    from .stages.slicing import MAX_TILE_CHUNK, admitted_tile_chunk_size, plan_slices

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
    requested = min(
        plan.jobs_per_frame,
        int(getattr(slice_cfg, "tile_batch_size", MAX_TILE_CHUNK)),
    )
    return admitted_tile_chunk_size(
        plan,
        imgsz=imgsz,
        device_tiles=device_tiles,
        requested=requested,
        byte_budget=int(
            getattr(slice_cfg, "tile_memory_budget_bytes", 256 * 1024 * 1024)
        ),
        task=task,
        max_detections=max_detections,
    )


def _sliced_tile_batch(
    config: InferenceConfig,
    frame_hw: tuple[int, int],
    imgsz: int,
    *,
    device_tiles: bool = False,
) -> int:
    """Return the admissible direct-OBB tile chunk used for its TRT profile."""
    from .stages.obb import effective_raw_detection_cap

    direct = config.obb.direct
    return _admitted_sliced_tile_batch(
        slice_cfg=direct.slice,
        frame_hw=frame_hw,
        imgsz=imgsz,
        task=direct.model_task,
        max_detections=effective_raw_detection_cap(config),
        device_tiles=device_tiles,
    )


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


def frame_space_roi_mask(
    roi_mask: np.ndarray | None, video_path: str | Path | None
) -> np.ndarray | None:
    """Return an ROI mask in the native coordinate space of cached frames.

    Detection caches store native video-frame coordinates even when tracking
    parameters originated from a resized display.  Cache readers and auxiliary
    output diagnostics must therefore use this same nearest-neighbor transform
    before applying source filtering.
    """

    if roi_mask is None:
        return None
    frame_hw = _probe_frame_hw(str(video_path) if video_path else None)
    if frame_hw is None or roi_mask.shape[:2] == frame_hw:
        return roi_mask
    import cv2

    return cv2.resize(
        roi_mask,
        (frame_hw[1], frame_hw[0]),  # cv2 wants (w, h)
        interpolation=cv2.INTER_NEAREST,
    )


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
    artifact_batch_size = config.runtime_artifact_batch_size
    direct = config.obb.direct if config.obb is not None else None
    sequential = config.obb.sequential if config.obb is not None else None
    slice_cfg = getattr(direct, "slice", None) if direct is not None else None
    stage1_slice_cfg = (
        getattr(sequential, "stage1_slice", None) if sequential is not None else None
    )
    if slice_cfg is not None and slice_cfg.enabled:
        frame_hw = _probe_frame_hw(video_path)
        imgsz = _probe_model_imgsz(direct.model_path)
        if frame_hw is not None and imgsz:
            # Sliced inference submits tile chunks, never a frame window.  Do
            # not retain a larger detection_batch_size here: it would make TRT
            # optimize an inadmissible profile that the sliced path cannot use.
            batch_size = _sliced_tile_batch(
                config,
                frame_hw,
                imgsz,
                device_tiles=bool(getattr(runtime, "tensor_on_cuda", False)),
            )
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
        return load_obb_models(
            config.obb,
            runtime,
            batch_size=max(batch_size, artifact_batch_size or batch_size),
        )

    if stage1_slice_cfg is not None and stage1_slice_cfg.enabled:
        frame_hw = _probe_frame_hw(video_path)
        imgsz = (
            int(sequential.detect_image_size)
            if sequential.detect_image_size > 0
            else _probe_model_imgsz(sequential.detect_model_path)
        )
        if frame_hw is not None and imgsz:
            from .stages.obb import effective_raw_detection_cap

            stage1_batch_size = _admitted_sliced_tile_batch(
                slice_cfg=stage1_slice_cfg,
                frame_hw=frame_hw,
                imgsz=imgsz,
                task="detect",
                max_detections=effective_raw_detection_cap(config),
                device_tiles=bool(getattr(runtime, "tensor_on_cuda", False)),
            )
            return load_obb_models(
                config.obb,
                runtime,
                batch_size=max(batch_size, artifact_batch_size or batch_size),
                stage1_batch_size=max(
                    stage1_batch_size, artifact_batch_size or stage1_batch_size
                ),
            )
        logger.warning(
            "Sliced sequential stage-1 enabled but frame size (%s) and/or "
            "detect imgsz (%s) could not be probed at load time; falling back "
            "to detection_batch_size=%d for its TensorRT engine profile.",
            frame_hw,
            imgsz,
            batch_size,
        )

    return load_obb_models(
        config.obb,
        runtime,
        batch_size=max(batch_size, artifact_batch_size or batch_size),
    )


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


def cache_set_is_fully_reusable(caches: _CacheSet) -> bool:
    """Whether every configured cache member is key-valid and coextensive.

    This is deliberately independent of :class:`InferenceRunner` construction:
    callers that only decide whether to prepare/reuse replay evidence must not
    initialize an OBB backend merely to inspect cache metadata.  It is the
    pure cache-set portion of ``InferenceRunner.caches_all_valid()``.
    """

    handles = caches.all_handles()
    if (
        not caches.set_manifest_valid
        or not handles
        or not all(handle.is_reusable() for handle in handles)
    ):
        return False
    if caches.generation_id is not None and any(
        handle._store.generation_id != caches.generation_id for handle in handles
    ):
        return False
    # A child crash can publish detection chunks before a downstream stage
    # finishes. Matching keys alone must not turn that honest partial cache
    # into a reusable complete pass.
    reference = caches.detection.coverage_ranges() if caches.detection else ()
    return all(handle.coverage_ranges() == reference for handle in handles)


@dataclass(frozen=True)
class _PartialReusePlan:
    """Which per-animal stages a detection-reusing batch pass must recompute.

    ``True`` = the stage's cache is stale for the current config (typically a
    replay-filter change re-keyed it) and is recomputed + rewritten; ``False``
    = it is reused untouched. Detection is always reused under a plan.
    """

    headtail: bool
    cnn: tuple[bool, ...]
    pose: bool
    apriltag: bool

    @property
    def any_recompute(self) -> bool:
        return self.headtail or any(self.cnn) or self.pose or self.apriltag


def partial_reuse_plan(
    caches: _CacheSet, start_frame: int, end_frame: int
) -> _PartialReusePlan | None:
    """Plan a batch pass that reuses the detection cache, or ``None``.

    Engages only when the detection member of the active cache generation is
    key-valid and covers EXACTLY ``[start_frame, end_frame]``: the recomputed
    per-animal members are written over that range, and the cache set is only
    promotable when every member has the same coverage. A per-animal member is
    reused when it is reusable, of the same generation and coextensive with
    detection; anything else is recomputed.
    """

    if not caches.set_manifest_valid or caches.detection is None:
        return None
    generation = caches.generation_id
    detection = caches.detection
    expected = ((int(start_frame), int(end_frame)),)
    if (
        generation is None
        or not detection.is_reusable()
        or detection._store.generation_id != generation
        or detection.coverage_ranges() != expected
    ):
        return None

    def _stale(handle) -> bool:
        if handle is None:
            return False
        return not (
            handle.is_reusable()
            and handle._store.generation_id == generation
            and handle.coverage_ranges() == expected
        )

    return _PartialReusePlan(
        headtail=_stale(caches.headtail),
        cnn=tuple(_stale(handle) for handle in caches.cnn),
        pose=_stale(caches.pose),
        apriltag=_stale(caches.apriltag),
    )


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
    *,
    read_only: bool = False,
    write_mode: str = "auto",
    filter_hash: "str | None" = None,
) -> _CacheSet:
    # Bind every per-video cache to the exact source file so a changed video
    # (e.g. a clip regenerated under the same name with a different frame count)
    # invalidates the cache instead of serving stale, truncated detections.
    def _k(key):
        return with_video_signature(key, video_sig)

    # Per-animal (downstream) caches hold every N-free filter survivor, so
    # their keys follow the replay filters (never N). The detection key is NOT
    # wrapped: it stores raw, pre-filter results. ``filter_hash`` overrides the
    # hash derived from ``config`` -- a read-only replay at candidate filters
    # opens the caches under the filters they were WRITTEN with (their
    # provenance) and lets the raw-index loaders raise for any detection the
    # candidate admits that the stored superset lacks.
    if filter_hash is None:
        filter_hash = replay_filter_hash(config, roi_mask)

    def _dk(key):
        return _k(with_replay_filters(key, filter_hash))

    detection_key = (
        # roi_mask is folded into the OBB key ONLY when slicing is enabled (see
        # detection_cache_key); None / disabled slicing => byte-identical key.
        # detection_batch_size is folded in ONLY when it is not 1 (see
        # _batch_term), so every cache written at the default batch keeps its
        # key. A hand-set batch used to write its detections under the SAME
        # key as a batch-1 run, and the next "Use cached detections" run
        # replayed them (B2).
        detection_cache_key(config.obb, roi_mask, config.detection_batch_size)
        if config.detection_source == "obb"
        else bgsub_detection_cache_key(config.bgsub)
    )

    member_names = ["detection.npz"]
    if config.headtail is not None:
        member_names.append("headtail.npz")
    member_names.extend(f"cnn_{c.label}.npz" for c in config.cnn_phases)
    if config.pose is not None:
        member_names.append("pose.npz")
    if config.apriltag.enabled:
        member_names.append("apriltag.npz")

    active = load_cache_set(cache_dir)
    active_matches = active is not None and set(active.members) == set(member_names)
    if read_only:
        generation_id = active.generation_id if active_matches else None
        revision_id = active.revision_id if active_matches else None
        root = (
            (cache_dir / next(iter(active.members.values()))).parent
            if active_matches
            else cache_dir
        )
        pending_promotion = False
        discard_if_unchanged = False
        set_manifest_valid = (
            active_matches or not (cache_dir / CACHE_SET_FILENAME).exists()
        )
    elif write_mode == "resume" and active_matches:
        generation_id = active.generation_id
        revision_id = uuid.uuid4().hex
        source_root = (cache_dir / next(iter(active.members.values()))).parent
        root = cache_dir / ".cache-generations" / generation_id / revision_id
        try:
            _clone_cache_revision(source_root, root)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        pending_promotion = True
        discard_if_unchanged = True
        set_manifest_valid = True
    else:
        generation_id = uuid.uuid4().hex
        revision_id = uuid.uuid4().hex
        root = cache_dir / ".cache-generations" / generation_id / revision_id
        pending_promotion = True
        discard_if_unchanged = False
        set_manifest_valid = True

    caches = _CacheSet(
        detection=DetectionCacheHandle(
            path=root / "detection.npz",
            key=_k(detection_key),
            read_only=read_only,
            write_mode=write_mode,
            generation_id=generation_id,
        ),
        headtail=(
            HeadTailCacheHandle(
                path=root / "headtail.npz",
                key=_dk(headtail_cache_key(config.headtail, config.canonical)),
                read_only=read_only,
                write_mode=write_mode,
                generation_id=generation_id,
            )
            if config.headtail is not None
            else None
        ),
        cnn=[
            CNNCacheHandle(
                path=root / f"cnn_{c.label}.npz",
                key=_dk(cnn_cache_key(c, config.canonical)),
                label=c.label,
                read_only=read_only,
                write_mode=write_mode,
                generation_id=generation_id,
            )
            for c in config.cnn_phases
        ],
        pose=(
            PoseCacheHandle(
                path=root / "pose.npz",
                key=_dk(pose_cache_key(config.pose, config.canonical)),
                read_only=read_only,
                write_mode=write_mode,
                generation_id=generation_id,
            )
            if config.pose is not None
            else None
        ),
        apriltag=(
            AprilTagCacheHandle(
                path=root / "apriltag.npz",
                key=_dk(apriltag_cache_key(config.apriltag)),
                read_only=read_only,
                write_mode=write_mode,
                generation_id=generation_id,
            )
            if config.apriltag.enabled
            else None
        ),
        cache_dir=cache_dir,
        generation_id=generation_id,
        revision_id=revision_id,
        member_names=member_names,
        pending_promotion=pending_promotion,
        discard_if_unchanged=discard_if_unchanged,
        staging_root=root if pending_promotion else None,
        set_manifest_valid=set_manifest_valid,
    )
    return caches


def _build_identity_evidence_stage(
    identity_evidence: "IdentityEvidenceRunConfig",
    unknown_prior: float = 0.0,
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

    ``unknown_prior`` (spec R6, Task 5): forwarded to every per-phase
    ``EvidenceBuilder`` so the composite/flat fused posterior redistributes
    mass onto the catalog's "unknown" slot instead of leaving it pinned at
    the per-factor floor product. 0.0 (default) is a strict no-op.
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
            unknown_prior=unknown_prior,
        )
    stage = IdentityEvidenceStage(catalog, cnn_builders, identity_evidence.tag_to_label)
    return catalog, stage


def _require_within_written_superset(
    written_config: "InferenceConfig | None",
    raw_obb: OBBResult,
    roi_mask: "np.ndarray | None",
    det_indices: np.ndarray,
    frame_idx: int,
) -> None:
    """Raise unless the replayed final set lies inside the WRITTEN superset.

    No-op when ``written_config`` is None (replay filters == write filters, so
    the final set is a subset by construction). Otherwise the superset the
    per-animal caches were written for is re-derived from the raw detections
    with the written filters; any final-N index outside it means those caches
    hold no result for it.
    """
    if written_config is None or len(det_indices) == 0:
        return
    _, superset = filter_for_source(
        written_config, raw_obb, roi_mask, apply_max_detections=False
    )
    missing = sorted(set(np.asarray(det_indices).tolist()) - set(superset.tolist()))
    if missing:
        raise DownstreamCacheError(
            f"frame {frame_idx}: the current detection filters admit "
            f"detection(s) {missing[:8]} that the per-animal caches were not "
            "built for -- per-animal results need recomputing for these filter "
            "settings (rerun inference without cache reuse)."
        )


def write_identity_evidence_sidecar(
    caches: "_CacheSet",
    config: InferenceConfig,
    stage: "IdentityEvidenceStage",
    frame_range: "range",
    out_path: Path,
    catalog_labels: "tuple[str, ...]",
    roi_mask: "np.ndarray | None" = None,
    written_config: "InferenceConfig | None" = None,
) -> None:
    """Read back raw per-frame caches over `frame_range` and write the evidence sidecar.

    ``written_config`` (read-only candidate replay only): the config whose
    filters the per-animal caches were written with; each frame's final set
    must lie inside that superset or ``DownstreamCacheError`` is raised.

    The batch seam Task 4 wires into ``run_batch_pass``: for each frame, reads
    the raw (pre-filter) detection cache and re-derives the final-N filtered
    detection set via ``filter_for_source(config, raw_obb, roi_mask)`` -- the
    same deterministic call ``InferenceRunner.load_frame`` makes -- giving
    ``filtered_obb`` plus its RAW detection-cache indices. The per-animal
    CNN/AprilTag caches hold the N-free superset keyed by raw index, so they
    are read through the replay loaders (``_load_cnn_for_indices`` /
    ``_load_apriltag``), which look those raw indices up and return results
    positionally aligned with ``filtered_obb`` -- and so with the stable
    ``det_ids`` (``filtered_obb.detection_ids``) `IdentityEvidenceStage`
    expects. A final-N index missing from a CNN cache raises
    ``DownstreamCacheError``. ``roi_mask`` must be the SAME frame-space mask
    the batch pass used (see ``InferenceRunner._write_identity_evidence_batch``),
    or this sidecar's final set silently diverges from the replayed one.

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
        filtered_obb, det_idx = filter_for_source(config, raw_obb, roi_mask)
        if filtered_obb.num_detections == 0:
            continue
        _require_within_written_superset(
            written_config, raw_obb, roi_mask, det_idx, frame_idx
        )
        det_ids = [int(d) for d in filtered_obb.detection_ids]

        # Read through the replay loaders: they look up the final-N RAW
        # indices in the superset caches and return results positionally
        # aligned with filtered_obb (so with det_ids). A phase with no
        # predictions is omitted, not passed as an empty list.
        try:
            cnn_results = _load_cnn_for_indices(
                cnn_caches, config.cnn_phases, frame_idx, det_idx
            )
        except DownstreamCacheError as exc:
            raise DownstreamCacheError(
                f"identity evidence, frame {frame_idx}: the current detection "
                "filters admit a detection the CNN caches hold no result for -- "
                "per-animal results need recomputing for these filter settings. "
                f"{exc}"
            ) from exc
        cnn_reads: dict[str, list] = {
            r.label: r.predictions for r in cnn_results if r.predictions
        }
        tag_read = _load_apriltag(caches.apriltag, frame_idx, det_idx)

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
    """Head-tail rows for the RAW ``det_indices``, positionally aligned.

    The cache holds the N-free superset keyed by raw detection-cache index; a
    requested index absent from it raises :class:`DownstreamCacheError`.
    """
    if cache is None or len(det_indices) == 0:
        return None
    data = cache.read_frame(frame_idx)
    if data is None:
        return None
    cached_det_indices, hints, confs, directed = data
    pos = positions_in(cached_det_indices, det_indices)
    return HeadTailResult(
        heading_hints=np.asarray(hints, dtype=np.float32)[pos],
        heading_confidences=np.asarray(confs, dtype=np.float32)[pos],
        directed_mask=np.asarray(directed)[pos].astype(bool).astype(np.uint8),
        canonical_affines=None,
    )


def _load_cnn_for_indices(
    caches: list[CNNCacheHandle],
    cnn_configs: list,
    frame_idx: int,
    det_indices: np.ndarray,
) -> list[CNNResult]:
    """One CNNResult per phase for the RAW ``det_indices``, positionally aligned.

    Cached ``det_index`` values are raw detection-cache indices; the returned
    predictions carry positions 0..K-1 over ``det_indices``. A phase with no
    cached frame, or written as explicit empty coverage (``predictions=[]``,
    the "phase produced no result" marker), yields an empty result. Otherwise
    a requested index with no prediction raises :class:`DownstreamCacheError`.
    """
    results: list[CNNResult] = []
    for cache, cfg in zip(caches, cnn_configs):
        preds = cache.read_frame(frame_idx)
        if not preds or len(det_indices) == 0:
            results.append(CNNResult(label=cfg.label, predictions=[]))
            continue
        results.append(cnn_raw_to_positions(preds, det_indices, cfg.label))
    return results


def _load_pose_for_indices(
    cache: PoseCacheHandle | None,
    frame_idx: int,
    det_indices: np.ndarray,
    filtered_obb: OBBResult,
) -> PoseResult | None:
    """Pose rows for the RAW ``det_indices``, positionally aligned.

    A requested index absent from the cached superset raises
    :class:`DownstreamCacheError`.
    """
    if cache is None or len(det_indices) == 0:
        return None
    data = cache.read_frame(frame_idx)
    if data is None:
        return None
    cached_keypoints, cached_det_indices, cached_valid = data
    if cached_keypoints.ndim < 2:
        return None
    pos = positions_in(cached_det_indices, det_indices)
    return PoseResult(
        keypoints=np.asarray(cached_keypoints, dtype=np.float32)[pos],
        valid_mask=np.asarray(cached_valid).astype(bool)[pos],
    )


def _load_apriltag(
    cache: AprilTagCacheHandle | None,
    frame_idx: int,
    det_indices: np.ndarray,
) -> AprilTagResult | None:
    """AprilTag detections on the RAW ``det_indices``, re-indexed positionally.

    Tags are sparse: a tag whose raw detection is outside ``det_indices`` is
    dropped (absence means "no tag", not an incoherent cache).
    """
    if cache is None:
        return None
    return apriltag_raw_to_positions(cache.read_frame(frame_idx), det_indices)


def _identity_evidence_base_signature(
    config: InferenceConfig, video_sig: str, roi_mask: "np.ndarray | None"
) -> str:
    """Base signature of the identity-evidence sidecar key.

    The sidecar holds evidence for the FINAL-N filtered set, so -- unlike the
    N-free per-animal caches -- it depends on N as well as on the replay
    filters. Both are folded in alongside the video signature.
    """
    parts = [video_sig]
    filter_hash = replay_filter_hash(config, roi_mask)
    if filter_hash:
        parts.append(f"filters={filter_hash}")
    if config.detection_source == "bgsub":
        bg = config.bgsub
        if bg is not None:
            parts.append(f"n={int(bg.max_targets)}x{int(bg.max_contour_multiplier)}")
    elif config.obb is not None:
        parts.append(f"n={int(config.obb.max_detections)}")
    return "|".join(parts)


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
        runtime_overlay: "InferenceRuntimeOverlay | None" = None,
        cache_filter_config: "InferenceConfig | None" = None,
    ) -> None:
        from hydra_suite.utils.profiling_process import maybe_arm_process_recorder

        from .stages.slicing import reset_oversize_warnings

        maybe_arm_process_recorder()
        # One run == one runner: the once-per-run oversize-admission WARNING
        # scope restarts here (before any model load sizes a tile batch).
        reset_oversize_warnings()

        self.config = config
        # Immutable per-run evidence of requested/admitted/effective execution
        # values. The runner consumes the already-resolved config and never
        # mutates or persists this overlay.
        self.runtime_overlay = runtime_overlay
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
        # The config whose replay filters the per-animal caches were WRITTEN
        # with. None => ``config`` itself (the normal case: write and replay
        # share filters). A read-only replay of candidate filters passes the
        # written (provenance) config: the per-animal caches open under its
        # filter hash (see _open_caches), and every replayed frame checks the
        # candidate's final set lies inside the written superset -- the only
        # check that also covers CNN/AprilTag, whose caches cannot tell "no
        # detection was stored" from "no result". The arena ROI is shared by
        # both (it is not a candidate parameter). Only read paths use it: a
        # writing pass keys what it computes under its own filters, so it is
        # refused outside cache-only mode.
        if cache_filter_config is not None and not cache_only:
            raise ValueError("cache_filter_config is only valid with cache_only=True")
        self._cache_filter_config = cache_filter_config
        self._cache_filter_hash = (
            replay_filter_hash(cache_filter_config, roi_mask)
            if cache_filter_config is not None
            else None
        )
        # Memoizes _frame_space_roi_mask's result, keyed by (video_path,
        # id(self._roi_mask)) so a later `self._roi_mask` reassignment (see
        # run_batch_pass's optional roi_mask override) naturally invalidates
        # the cache instead of silently reusing a stale resample.
        self._frame_space_roi_mask_cache: dict = {}
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
        self.runtime_artifact_ids = (
            self._models.obb.runtime_artifact_ids
            if self._models.obb is not None
            else ()
        )
        self.runtime_artifact_prepare_seconds = (
            self._models.obb.runtime_artifact_prepare_seconds
            if self._models.obb is not None
            else 0.0
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
                _build_identity_evidence_stage(
                    identity_evidence,
                    unknown_prior=float(config.identity_unknown_prior),
                )
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
        # Run-scoped: frames whose candidate count exceeded
        # MAX_DETECTIONS_PER_FRAME (each logged as a WARNING when recorded);
        # shared with the batch Pipeline like ``clipping_stats``.
        self.detection_limit_stats = DetectionLimitStats()

    def _rank_and_bound(self, raw_obb: OBBResult, frame_idx: int) -> OBBResult:
        """Confidence-rank + bound one OBB frame; record a limit hit loudly."""
        raw_obb, candidate_count = rank_and_bound(raw_obb)
        if candidate_count > MAX_DETECTIONS_PER_FRAME:
            self.detection_limit_stats.record(frame_idx, candidate_count)
        return raw_obb

    @property
    def obb_class_names(self) -> "dict[int, str] | None":
        """id->name map from the loaded OBB model, for label display. None if no OBB model."""
        if self._models.obb is None:
            return None
        # direct mode uses direct_model; sequential's classes come from the
        # stage-2 obb_model
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
            self.config,
            self.cache_dir,
            self._video_sig,
            self._roi_mask,
            read_only=True,
            filter_hash=self._cache_filter_hash,
        )
        return cache_set_is_fully_reusable(caches)

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
            self.config,
            self.cache_dir,
            self._video_sig,
            self._roi_mask,
            read_only=True,
            filter_hash=self._cache_filter_hash,
        )
        if not caches.set_manifest_valid or caches.detection is None:
            return False
        return caches.detection.covers_frame_range(start_frame, end_frame)

    def detection_cache_missing_frames(
        self, start_frame: int, end_frame: int, max_report: int = 10
    ) -> list[int]:
        """Report up to ``max_report`` frames missing from the detection cache."""
        if self.cache_dir is None:
            return []
        caches = _open_caches(
            self.config,
            self.cache_dir,
            self._video_sig,
            self._roi_mask,
            read_only=True,
            filter_hash=self._cache_filter_hash,
        )
        if not caches.set_manifest_valid or caches.detection is None:
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
                    self.config,
                    self.cache_dir,
                    self._video_sig,
                    self._roi_mask,
                    write_mode="resume",
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
                    # sequential — safe here because run_realtime is driven in frame
                    # order.
                    raw_obb = run_bgsub(
                        frame,
                        frame_idx,
                        self._models.bgsub,
                        self.config.bgsub,
                        self.runtime,
                        roi_mask=roi_mask,
                        limit_stats=self.detection_limit_stats,
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
                            raw, effective_raw_detection_cap(self.config.obb)
                        )
                    else:
                        raw_obb = raw
                    raw_obb = self._rank_and_bound(raw_obb, frame_idx)
                # Re-stamp detection_ids with the real frame_idx (materialize_tensors / the
                # CPU OBB path generate them at frame 0) so cached ids are unique per
                # frame.
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
                    polygons=raw_obb.polygons,
                )
                if caches is not None and caches.detection is not None:
                    caches.detection.write_frame(frame_idx, result=raw_obb)

            with span(N.RT_FILTER):
                # Per-animal stages run on the N-free SUPERSET (every filter
                # survivor, no 2N window / final N cut) so the downstream caches
                # serve any N at replay -- the same contract as the batch
                # Pipeline. The returned FrameResult (and the identity evidence)
                # is the final-N set, a subset of the superset by construction.
                superset_obb, superset_idx = filter_for_source(
                    self.config, raw_obb, roi_mask, apply_max_detections=False
                )
                final_obb, final_idx = filter_for_source(self.config, raw_obb, roi_mask)

            def _empty_frame_result() -> FrameResult:
                empty_result = _build_frame_result(
                    frame_idx, final_obb, np.zeros(0, np.int32), None, [], None, None
                )
                # Task 11 fix: surface the bg-sub masks here too, exactly like the
                # non-empty path below. last_bg_u8 is the source of truth for
                # "was the background established" -- it is None ONLY during the
                # true first-frame warmup (see bgsub.py:167-170) and a real array
                # on every frame after, even with zero detections. worker.py uses
                # `bg_u8 is None` as its warmup sentinel; if an empty-frame return
                # skipped the assignment, a post-warmup zero-detection frame
                # (occlusion, animal left, threshold blip) would be misread as
                # still-warming-up and silently drop Kalman aging + the CSV row
                # for that frame.
                if (
                    self.config.detection_source == "bgsub"
                    and self._models.bgsub is not None
                ):
                    empty_result.fg_mask = self._models.bgsub.last_fg_mask
                    empty_result.bg_u8 = self._models.bgsub.last_bg_u8
                return empty_result

            if superset_obb.num_detections == 0:
                if caches is not None:
                    empty_det_indices = np.zeros(0, np.int32)
                    if caches.headtail is not None:
                        caches.headtail.write_frame(
                            frame_idx,
                            det_indices=empty_det_indices,
                            heading_hints=np.zeros(0, np.float32),
                            heading_confidences=np.zeros(0, np.float32),
                            directed_mask=np.zeros(0, np.uint8),
                        )
                    for cache in caches.cnn:
                        cache.write_frame(frame_idx, predictions=[])
                    if caches.pose is not None:
                        caches.pose.write_frame(
                            frame_idx,
                            det_indices=empty_det_indices,
                            keypoints=np.zeros((0, 0, 3), np.float32),
                            valid_mask=np.zeros(0, bool),
                        )
                    if caches.apriltag is not None:
                        caches.apriltag.write_frame(
                            frame_idx,
                            result=AprilTagResult(
                                tag_ids=[],
                                det_indices=[],
                                centers=np.zeros((0, 2), np.float32),
                                corners=np.zeros((0, 4, 2), np.float32),
                            ),
                        )
                return _empty_frame_result()

            from . import limits
            from .downstream_select import (
                apriltag_positions_to_raw,
                cnn_positions_to_raw,
                narrow_to_final,
                run_superset_chunked,
            )

            geometry = self.config.canonical
            with span(N.RT_CROPS, units=superset_obb.num_detections):
                # F1 guard: every detection that will be canonicalized by ANY
                # consumer (headtail, cnn, pose all warp through this one
                # geometry) gets its overflow_ratio recorded here, once, rather
                # than at each of the many internal canonical_affine call sites
                # (which would double-count a detection once per consumer
                # stage). The whole superset is canonicalized, so the whole
                # superset is recorded (mirrors the batch Pipeline).
                if (
                    self._models.headtail is not None
                    or self._models.cnn
                    or self._models.pose is not None
                ):
                    for _corners in superset_obb.corners:
                        self.clipping_stats.record(_corners, geometry)

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

            def _do_ht(chunk: OBBResult) -> HeadTailResult | None:
                if not self._models.headtail:
                    return None
                return run_headtail(
                    frame,
                    chunk,
                    self._models.headtail,
                    self.config.headtail,
                    self.runtime,
                    geometry,
                )

            def _do_cnn(chunk: OBBResult) -> list[CNNResult]:
                return [
                    run_cnn(frame, chunk, mdl, cfg, self.runtime, geometry)
                    for cfg, mdl in zip(self.config.cnn_phases, self._models.cnn)
                ]

            def _do_pose(chunk: OBBResult, canonical_crops) -> PoseResult | None:
                if not self._models.pose:
                    return None
                return run_pose(
                    canonical_crops,
                    chunk,
                    self._models.pose,
                    self.config.pose,
                    self.runtime,
                    geometry,
                )

            def _do_at(chunk: OBBResult, aabb_crops) -> AprilTagResult | None:
                if not self._models.apriltag:
                    return None
                return run_apriltag(
                    aabb_crops,
                    chunk,
                    self._models.apriltag,
                    self.config.apriltag,
                )

            def _run_chunk(chunk: OBBResult, foreign: ForeignSet):
                with span(N.RT_CROPS, units=chunk.num_detections):
                    # Canonical (native-extent) crops are only consumed by the
                    # pose stage; head-tail / CNN warp directly from the frame.
                    # Pose masks each crop against the FULL superset (``foreign``),
                    # not the chunk, so a detection's pose never depends on N or
                    # its chunk (R7).
                    canonical_crops = (
                        extract_canonical_crops(
                            frame,
                            chunk,
                            geometry,
                            self.runtime,
                            suppress_foreign=suppress_foreign,
                            background_color=background_color,
                            foreign_set=foreign,
                        )
                        if self._models.pose is not None
                        else None
                    )
                    aabb_crops = (
                        extract_aabb_crops(
                            frame, chunk, padding=self.config.apriltag.crop_padding
                        )
                        if self._models.apriltag
                        else []
                    )

                with span(N.RT_INDIVIDUAL, units=chunk.num_detections):
                    # Run the individual-analysis stages SEQUENTIALLY, not in a
                    # per-frame ThreadPoolExecutor. Profiling on CUDA (RT_PROFILE)
                    # showed the per-frame pool cost ~834 ms/frame vs ~37 ms/frame
                    # sequential (a 22x regression): spinning up a fresh 4-thread
                    # pool every frame and driving CUDA / the onnxruntime SLEAP
                    # backend from short-lived worker threads serialises on the
                    # GIL and the default CUDA stream while paying thread +
                    # context setup each frame, with no real parallelism on a
                    # single GPU. Sequential brings realtime back to legacy parity
                    # (~137 ms/frame total incl. frame read).
                    return (
                        _do_ht(chunk),
                        _do_cnn(chunk),
                        _do_pose(chunk, canonical_crops),
                        _do_at(chunk, aabb_crops),
                    )

            # The superset is processed in DOWNSTREAM_CHUNK_SIZE-row chunks so a
            # frame with up to MAX_DETECTIONS_PER_FRAME detections never
            # materialises all its crops at once. Results are whole-superset,
            # positionally aligned with superset_obb; cnn_all is phase-aligned
            # (None for a phase with no result).
            ht_all, cnn_all, pose_all, at_all = run_superset_chunked(
                superset_obb,
                _run_chunk,
                len(self._models.cnn),
                limits.DOWNSTREAM_CHUNK_SIZE,
            )

            with span(N.RT_CACHE):
                # Persist RAW downstream results for the whole superset, keyed by
                # RAW detection-cache index (CNN det_index and AprilTag
                # det_indices converted to raw) so the backward pass can replay
                # any N via load_frame -- mirrors the batch Pipeline's writes.
                if caches is not None:
                    if caches.headtail is not None and ht_all is not None:
                        caches.headtail.write_frame(
                            frame_idx,
                            det_indices=superset_idx,
                            heading_hints=ht_all.heading_hints,
                            heading_confidences=ht_all.heading_confidences,
                            directed_mask=ht_all.directed_mask,
                        )
                    # A phase with no result writes explicit empty coverage,
                    # exactly like CacheWriter (batch), so a cache written by
                    # either path replays identically.
                    for cache, cnn_result in zip(caches.cnn, cnn_all):
                        cache.write_frame(
                            frame_idx,
                            predictions=(
                                []
                                if cnn_result is None
                                else cnn_positions_to_raw(
                                    cnn_result, superset_idx
                                ).predictions
                            ),
                        )
                    if caches.pose is not None and pose_all is not None:
                        caches.pose.write_frame(
                            frame_idx,
                            det_indices=superset_idx,
                            keypoints=pose_all.keypoints,
                            valid_mask=pose_all.valid_mask,
                        )
                    if caches.apriltag is not None and at_all is not None:
                        caches.apriltag.write_frame(
                            frame_idx,
                            result=apriltag_positions_to_raw(at_all, superset_idx),
                        )

            if final_obb.num_detections == 0:
                # The superset survived the N-free filters but the final N cut
                # left nothing: the superset caches are written above; the frame
                # itself is empty (no identity evidence, like the empty path).
                return _empty_frame_result()

            # Final-N rows inside the superset; raises if final is not a subset.
            ht_result, cnn_results, pose_result, at_result = narrow_to_final(
                superset_idx, final_idx, ht_all, cnn_all, pose_all, at_all
            )

            with span(N.RT_CACHE):
                # Identity Phase 3, Task 4 (realtime seam): build + persist this
                # frame's identity evidence inline, from the in-hand FINAL-N
                # final_obb/cnn_results/at_result -- no read-back needed (unlike
                # the batch seam, which re-derives det_ids from a disk read-back
                # after the pass). det_ids come from final_obb.detection_ids
                # (stable ids, aligned by position with the narrowed CNN/AprilTag
                # det_index, both 0..K-1 over this same final_obb). Only runs when
                # caches are open for writing -- a pure in-memory/preview realtime
                # call (cache_dir=None) writes nothing.
                if caches is not None and self._identity_stage is not None:
                    self._write_identity_evidence_realtime(
                        frame_idx, final_obb, cnn_results, at_result
                    )

            with span(N.RT_FINALIZE):
                frame_result = _build_frame_result(
                    frame_idx,
                    final_obb,
                    final_idx,
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

                # Task 17g: build StreamingAnalysisPayload for legacy identity
                # consumers.
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
            _identity_evidence_base_signature(
                self.config, self._video_sig, self._roi_mask
            ),
            unknown_prior=float(self.config.identity_unknown_prior),
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

    def ensure_identity_evidence_sidecar(
        self,
        start_frame: int,
        end_frame: int,
        *,
        out_path: "Path | None" = None,
    ) -> "Path | None":
        """Return the "batch" identity-evidence sidecar, rebuilding it if absent.

        On cache reuse no batch pass runs, so the sidecar for the CURRENT key
        (filters + N) may not exist yet -- e.g. a replay at a different N than
        the caches were written with. It is rebuilt here from the per-animal
        caches through the raw-index replay loaders; no stage model runs.
        ``out_path`` overrides the destination (a read-only replay must not
        write into the cache directory). Returns ``None`` when this runner has
        no identity-evidence config.
        """
        if self._identity_evidence is None or self.cache_dir is None:
            return None
        path = (
            Path(out_path)
            if out_path is not None
            else self._identity_evidence_sidecar_path("batch")
        )
        if not path.exists():
            self._write_identity_evidence_batch(start_frame, end_frame, path)
        return path

    def _write_identity_evidence_batch(
        self, start_frame: int, end_frame: int, out_path: "Path | None" = None
    ) -> None:
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
            self.config,
            self.cache_dir,
            self._video_sig,
            self._roi_mask,
            read_only=True,
            filter_hash=self._cache_filter_hash,
        )
        if out_path is None:
            out_path = self._identity_evidence_sidecar_path("batch")
        write_identity_evidence_sidecar(
            read_caches,
            self.config,
            self._identity_stage,
            range(start_frame, end_frame + 1),
            out_path,
            self._identity_catalog.labels,
            roi_mask=self._frame_space_roi_mask(self._video_path),
            written_config=self._cache_filter_config,
        )

    def detect_batch_raw(
        self,
        frames: "list[np.ndarray]",
        frame_indices: "list[int] | None" = None,
        roi_mask: "np.ndarray | None" = None,
    ) -> "list[OBBResult]":
        """Run OBB detection over a batch of frames, returning UNFILTERED raw
        results (pre-``filter_for_source``). No cache is read or written.
        """
        if self._models.obb is None:
            raise RuntimeError(
                "detect_batch_raw requires an OBB detection config (config.obb)"
            )
        frames = list(frames)
        if frame_indices is None:
            frame_indices = list(range(len(frames)))

        raw_list = run_obb(
            frames,
            self._models.obb,
            self.config.obb,
            self.runtime,
            roi_mask=roi_mask,
        )
        raw_results: list[OBBResult] = []
        for raw, f_idx in zip(raw_list, frame_indices):
            if isinstance(raw, _RawOBBTensors):
                raw_obb = materialize_tensors(
                    raw, effective_raw_detection_cap(self.config.obb)
                )
            else:
                raw_obb = raw
            raw_obb = self._rank_and_bound(raw_obb, f_idx)
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
                polygons=raw_obb.polygons,
            )
            raw_results.append(raw_obb)
        return raw_results

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
        raw_results = self.detect_batch_raw(frames, frame_indices, roi_mask)
        results: list[OBBResult] = []
        for raw_obb in raw_results:
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

        Memoized per (video_path, id(self._roi_mask)): ``load_frame`` calls
        this once per frame during cached-replay (forward reuse and the
        whole backward pass), and re-probing the video file's dimensions via
        a fresh VideoCapture open on every single frame would be a real,
        avoidable per-frame cost over a long session. The mask and the
        video's geometry are both fixed for the life of one pass, so this is
        safe to cache; a later `self._roi_mask` reassignment (a different
        object) changes the cache key and forces a fresh resample rather
        than reusing a stale one.
        """
        mask = self._roi_mask
        if mask is None:
            return None
        cache_key = (str(video_path) if video_path else None, id(mask))
        if cache_key in self._frame_space_roi_mask_cache:
            return self._frame_space_roi_mask_cache[cache_key]
        resolved = frame_space_roi_mask(mask, video_path)
        self._frame_space_roi_mask_cache[cache_key] = resolved
        return resolved

    def _build_pipeline(
        self,
        caches: _CacheSet,
        roi_mask: "np.ndarray | None" = None,
        reuse_plan: _PartialReusePlan | None = None,
    ) -> Pipeline:
        """Construct the depth=1 Pipeline that drives the batch stage layer.

        The Pipeline owns the per-window stage sequence (OBB → crops → HT/CNN/pose
        → AprilTag → scatter); cache writes go through a ``CacheWriter`` (sync mode)
        that reproduces ``_run_batch``'s exact raw-result side effects.

        ``roi_mask`` (frame-space) is threaded onto ``PipelineStages`` so the OBB
        stage can ROI-gate slice tiles; ``None`` keeps the full tile grid.

        ``reuse_plan`` (partial reuse): detections are read from the detection
        cache instead of running the detector, and only the stale per-animal
        stages run and are written. Head-tail still runs (unwritten) when a
        stale CNN phase or pose needs it, so their inputs match a fresh run.
        """
        headtail_model = self._models.headtail
        cnn_models = list(self._models.cnn)
        pose_model = self._models.pose
        apriltag_model = self._models.apriltag
        detection_reader = None
        if reuse_plan is not None:
            detection_reader = caches.detection.read_frame
            cnn_models = [
                model if stale else None
                for model, stale in zip(cnn_models, reuse_plan.cnn)
            ]
            if not reuse_plan.pose:
                pose_model = None
            if not reuse_plan.apriltag:
                apriltag_model = None
            if not (reuse_plan.headtail or any(reuse_plan.cnn) or reuse_plan.pose):
                headtail_model = None
        stages = PipelineStages(
            config=self.config,
            obb_models=self._models.obb,
            bgsub_model=self._models.bgsub,
            headtail_model=headtail_model,
            cnn_models=cnn_models,
            pose_model=pose_model,
            apriltag_model=apriltag_model,
            roi_mask=roi_mask,
        )
        writes = reuse_plan or _PartialReusePlan(
            headtail=True,
            cnn=tuple(True for _ in caches.cnn),
            pose=True,
            apriltag=True,
        )
        handles: dict[str, CacheHandle] = {}
        if caches.detection is not None and reuse_plan is None:
            handles["detection"] = caches.detection
        if caches.headtail is not None and writes.headtail:
            handles["headtail"] = caches.headtail
        for cnn_cfg, cnn_handle, stale in zip(
            self.config.cnn_phases, caches.cnn, writes.cnn
        ):
            if stale:
                handles[f"cnn_{cnn_cfg.label}"] = cnn_handle
        if caches.pose is not None and writes.pose:
            handles["pose"] = caches.pose
        if caches.apriltag is not None and writes.apriltag:
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
            detection_limit_stats=self.detection_limit_stats,
            detection_reader=detection_reader,
        )

    def run_batch_pass(
        self,
        video_path: Path,
        progress_cb=None,
        start_frame: int = 0,
        end_frame: int | None = None,
        should_stop=None,
        roi_mask: "np.ndarray | None" = None,
        reuse_detection_cache: bool = False,
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

            # Recover the clamped bounds from the reader so range_total matches.
            start_frame = frame_source.start_frame
            end_frame = frame_source.end_frame
            range_total = frame_source.frame_count

            # Partial reuse (opt-in: the caller allows cache reuse): when the
            # detection cache already covers this exact range under the
            # current detection key, a changed replay filter only re-keys the
            # per-animal caches -- read detections back instead of running the
            # detector, and recompute only the stale per-animal stages.
            reuse_plan = None
            if reuse_detection_cache:
                probe = _open_caches(
                    self.config,
                    self.cache_dir,
                    self._video_sig,
                    self._roi_mask,
                    read_only=True,
                )
                reuse_plan = partial_reuse_plan(probe, start_frame, end_frame)
                probe.close()
                if reuse_plan is None:
                    logger.info(
                        "Detection cache does not cover frames %d-%d under the "
                        "current detection settings; running the detector.",
                        start_frame,
                        end_frame,
                    )
                elif not reuse_plan.any_recompute:
                    logger.info(
                        "Every inference cache is reusable for frames %d-%d; "
                        "nothing to recompute.",
                        start_frame,
                        end_frame,
                    )
                    frame_source.close()
                    self._write_identity_evidence_batch(start_frame, end_frame)
                    return
                else:
                    logger.info(
                        "Reusing cached detections for frames %d-%d; "
                        "recomputing stale per-animal stages only (%s).",
                        start_frame,
                        end_frame,
                        reuse_plan,
                    )

            with span(N.OPEN_CACHES):
                caches = _open_caches(
                    self.config,
                    self.cache_dir,
                    self._video_sig,
                    self._roi_mask,
                    write_mode=(
                        "fresh" if start_frame == 0 and reuse_plan is None else "resume"
                    ),
                )
            self._caches = caches

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
                caches,
                roi_mask=self._frame_space_roi_mask(video_path),
                reuse_plan=reuse_plan,
            )
            complete_pass = False
            try:
                pass_result = pipeline.run(
                    frame_source,
                    range(start_frame, end_frame + 1),
                    progress_cb=progress_cb,
                    range_total=range_total,
                    should_stop=should_stop,
                )
                if pass_result is not None and pass_result.cancelled:
                    raise InferenceCancelled(
                        "inference cancelled before the active frame window completed"
                    )
                complete_pass = pass_result is None or (
                    pass_result.frames_processed == range_total
                )
            finally:
                frame_source.close()
                # depth>=2 uses an async CacheWriter; flush/close it before closing the
                # handles so all queued writes land (Pipeline.run already does this on
                # its own teardown path, but a pre-run failure may skip it).
                pipeline.cache_writer.close()
                # Never close a handle while a timed-out writer can still be
                # inside it. CacheWriter.close raises before this point when
                # its worker fails to stop by the deadline.
                caches.close(complete=complete_pass)

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

    def load_filtered_obb(self, frame_idx: int):
        """Load one cached OBB frame through the production filter contract.

        This is the cache-replay prefix shared by full ``load_frame`` calls and
        consumers such as confidence-density construction.  Keeping it here
        makes every consumer use the candidate's source-aware filtering and
        native-frame ROI transform rather than reimplementing one or the other.
        """

        filtered_obb, det_indices, _raw, _roi = self._load_filtered_with_raw(frame_idx)
        return filtered_obb, det_indices

    def _load_filtered_with_raw(self, frame_idx: int):
        """``load_filtered_obb`` plus the raw frame and frame-space ROI used."""

        if self.cache_dir is None:
            raise RuntimeError("cache_dir not set — cannot load cached frames")
        if self._caches is None:
            self._caches = _open_caches(
                self.config,
                self.cache_dir,
                self._video_sig,
                self._roi_mask,
                read_only=True,
                filter_hash=self._cache_filter_hash,
            )
        if not self._caches.set_manifest_valid:
            raise RuntimeError("inference cache set manifest is invalid or incomplete")

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
        #
        # roi_mask: cached frames are read back at native video-frame geometry
        # (the batch pass never resizes), so the mask must be resampled into
        # that same space -- exactly the transform run_batch_pass already
        # applies via _frame_space_roi_mask() before handing it to the
        # pipeline. Without this, ROI filtering silently never applied to any
        # cached/replayed YOLO-OBB read (forward cache reuse AND the backward
        # pass), regardless of the ROI configured at construction.
        roi = self._frame_space_roi_mask(self._video_path)
        filtered_obb, det_indices = filter_for_source(self.config, raw_obb, roi)
        return filtered_obb, det_indices, raw_obb, roi

    def load_frame(self, frame_idx: int) -> FrameResult:
        """Load one cached frame with its production filtering and evidence."""

        filtered_obb, det_indices, raw_obb, roi = self._load_filtered_with_raw(
            frame_idx
        )
        assert self._caches is not None  # established by _load_filtered_with_raw
        _require_within_written_superset(
            self._cache_filter_config, raw_obb, roi, det_indices, frame_idx
        )
        try:
            ht_result = _load_headtail_for_indices(
                self._caches.headtail, frame_idx, det_indices, filtered_obb
            )
            cnn_results = _load_cnn_for_indices(
                self._caches.cnn, self.config.cnn_phases, frame_idx, det_indices
            )
            pose_result = _load_pose_for_indices(
                self._caches.pose, frame_idx, det_indices, filtered_obb
            )
        except DownstreamCacheError as exc:
            raise DownstreamCacheError(
                f"frame {frame_idx}: the current detection filters admit a "
                "detection the per-animal (head-tail/CNN/pose) caches hold no "
                "result for -- per-animal results need recomputing for these "
                f"filter settings (rerun inference without cache reuse). {exc}"
            ) from exc
        at_result = _load_apriltag(self._caches.apriltag, frame_idx, det_indices)

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
            self._caches.close()
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
