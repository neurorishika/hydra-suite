"""Active learning worker for DetectKit projects."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Literal, Sequence

import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.cache.reuse import get_or_compute_raw
from hydra_suite.core.inference.config import OBBConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import filter_with_indices
from hydra_suite.data.al.acquisition import PRESETS, AcquisitionWeights, select
from hydra_suite.data.al.candidate_pool import CandidatePoolConfig, build_candidate_pool
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.export import ExportedFrame, export_al_dataset
from hydra_suite.data.al.frame_source import (
    DetectKitProjectSource,
    FrameRef,
    FrameSource,
    ImageFolderFrameSource,
    VideoFrameSource,
)
from hydra_suite.data.al.inference_adapter import build_obb_config_for_al
from hydra_suite.data.al.signals import (
    ALSignals,
    detections_from_obb_result,
    score_count_deviation,
    score_crowd,
    score_fragmentation,
    score_nms_instability,
    score_uncertainty,
)
from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
from hydra_suite.utils.geometry import obb_corners_from_dims as _detection_corners
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.utils.video_artifacts import build_inference_cache_dir
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)

# `Detection`/`DetectorFn` (the per-frame detector-closure aliases) are gone
# with the closure itself: nothing outside this module ever referenced either,
# and the AL path now describes its detector declaratively (`ALDetectorSpec`)
# rather than passing an opaque callable.


@dataclass
class ALDetectorSpec:
    """How to build the round's detector -- a declarative spec, not a closure.

    This replaces the old ``ALRequest.detector_fn`` per-frame closure. The AL
    pipeline no longer calls an opaque `detector(frame, conf, iou)` once (in
    fact four times) per frame; it builds one `InferenceRunner` from these
    fields and runs a single batched, cached detection pass over the whole
    candidate list. The fields are exactly the tuple
    ``detectkit_resolve_inference_models`` returns (`kind`, primary, secondary)
    plus the sequential-mode crop padding, i.e. exactly what
    ``build_obb_config_for_al`` consumes.
    """

    kind: Literal["obb_direct", "sequential"]
    model_path: str
    secondary_model_path: str | None = None
    crop_pad_ratio: float = 0.15
    runtime_tier: str | None = None


@dataclass
class ALRequest:
    """User input for one active-learning round."""

    input_kind: Literal["video", "folder", "project"]
    input_path: str
    project: DetectKitProject
    budget: int
    preset: str = "balanced"
    weights_override: AcquisitionWeights | None = None
    expected_count: int = 0
    detector: ALDetectorSpec | None = None
    diversity_window: int = 30
    probabilistic: bool = True
    candidate_pool: CandidatePoolConfig = field(default_factory=CandidatePoolConfig)
    base_conf: float = 0.25
    base_iou: float = 0.7
    export_level: str = "obb"
    export_levels: list[str] = field(default_factory=lambda: ["obb"])
    # The model's actual geometry ceiling (GeometryLevel.label string), set
    # independently of what the caller *requests* -- this is the honesty
    # guard's ground truth. It must never be derived from `export_levels`
    # itself (that makes the "don't claim more than the model can produce"
    # check tautological: a caller-chosen request can never exceed a limit
    # computed from that same request). Mirrors `resolve_native_level` in
    # `data/dataset_generation.py`, which derives from the actual configured
    # detection method/task, independent of what any dialog selected.
    native_level: str = "obb"


@dataclass
class ALResult:
    """Outcome of one AL round."""

    source_path: str
    n_picked: int
    selected_frames: list[int]


class _LazyALImages(Mapping):
    """Picked-frame images, decoded from `source` on `__getitem__`.

    `export_al_dataset`'s authoritative root reads each key exactly once, so
    only the frame currently being written is ever resident -- there is no
    eager dict of every picked frame's pixels.
    """

    def __init__(
        self,
        source: FrameSource,
        frame_refs_by_id: dict[int, "FrameRef"],
        frame_ids: Sequence[int],
    ) -> None:
        self._source = source
        self._frame_refs_by_id = frame_refs_by_id
        self._frame_ids = list(frame_ids)

    def __getitem__(self, frame_id: int):
        img = self._source.read(self._frame_refs_by_id[frame_id])
        if img is None:
            raise KeyError(frame_id)
        return img

    def __iter__(self):
        return iter(self._frame_ids)

    def __len__(self) -> int:
        return len(self._frame_ids)


def _build_frame_source(req: ALRequest) -> FrameSource:
    if req.input_kind == "video":
        return VideoFrameSource(req.input_path)
    if req.input_kind == "folder":
        return ImageFolderFrameSource(req.input_path)
    if req.input_kind == "project":
        return DetectKitProjectSource(req.project, only_unlabeled=True)
    raise ValueError(f"unknown input_kind: {req.input_kind}")


def _build_detection_context(req: ALRequest) -> tuple[object, OBBConfig]:
    """Return ``(runner, obb_config)`` for this round's batched detection pass.

    The single construction seam for the AL detector -- the counterpart of
    `_build_frame_source` for the model side, and the one place tests patch to
    inject a runner double (anything implementing
    ``detect_batch_raw(frames, frame_indices=...) -> list[OBBResult]``) instead
    of loading real weights.

    ``InferenceRunner`` is imported lazily: it pulls in torch/ultralytics, and
    importing this module (the AL dialog does, to build an `ALRequest`) must
    not pay that cost.
    """
    spec = req.detector
    if spec is None:
        raise ValueError("ALRequest.detector is required to build a detection context")
    config = build_obb_config_for_al(
        spec.kind,
        spec.model_path,
        spec.secondary_model_path,
        crop_pad_ratio=spec.crop_pad_ratio,
        confidence_threshold=req.base_conf,
        iou_threshold=req.base_iou,
        runtime_tier=spec.runtime_tier,
    )

    from hydra_suite.core.inference.runner import InferenceRunner

    # Only a video input has a stable file fingerprint to bind the detection
    # cache key to (see `with_video_signature`); folder/project inputs pass
    # None, which is the documented "non-video context" case.
    video_path = req.input_path if req.input_kind == "video" else None
    runner = InferenceRunner(config, video_path=video_path)
    return runner, config.obb


@contextlib.contextmanager
def _al_detection_cache_dir(req: ALRequest) -> Iterator[Path]:
    """Yield the directory the round's raw-detection cache lives in.

    For a video input this is a DEDICATED ``al/`` subdirectory of the same
    ``.inference_cache_<stem>/`` folder tracking uses -- never the folder
    itself. `get_or_compute_raw` always writes ``<cache_dir>/detection.npz``,
    and `DetectionCacheHandle.close()` rewrites that file from its own buffer
    alone (this repo's documented no-merge convention). Since an AL round's
    cache key is derived from its own `build_obb_only_config` and will rarely
    match a tracking run's key, pointing AL at the shared folder would mean
    every AL round silently overwrote a complete, expensive tracking detection
    cache with a sparse candidate-only one. AL therefore keeps its own file:
    repeat AL rounds on the same video with the same detector settings still
    hit a pure cache read, and tracking's cache is untouchable from here.

    Folder/project inputs deliberately get a throwaway directory instead: their
    frame ids are positions in a sorted file listing, which shift whenever an
    image is added or removed, so a persisted cache could later serve one
    image's detections for another's. Those inputs still get the batched
    single-pass detection; only cross-run reuse is given up.
    """
    if req.input_kind == "video":
        cache_dir = build_inference_cache_dir(req.input_path, create=True) / "al"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yield cache_dir
        return
    with tempfile.TemporaryDirectory(prefix="hydra_al_cache_") as tmp_dir:
        yield Path(tmp_dir)


def _frame_signals(
    frame_id: int,
    raw_obb_result: OBBResult,
    obb_config: OBBConfig,
    expected_count: int,
    base_conf: float,
    base_iou: float,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[ALSignals, list]:
    # Base detections come from the same cheap re-filter of the cached raw
    # OBBResult that `score_nms_instability` uses internally -- no detector
    # call happens here. Detections may be 6-tuples (cx,cy,w,h,theta,conf);
    # d[5] and d[:5] stay valid if a 7th (native polygon) element is ever
    # added upstream, so no branching is needed here.
    base_config = dataclasses.replace(
        obb_config, confidence_threshold=base_conf, iou_threshold=base_iou
    )
    filtered, _ = filter_with_indices(raw_obb_result, base_config, roi_mask=None)
    detections = detections_from_obb_result(filtered)
    confidences = [d[5] for d in detections]
    uncertainty = score_uncertainty(confidences, conf_floor=base_conf)
    count_dev = score_count_deviation(len(detections), expected_count)

    obb_corners = [_detection_corners(*d[:5]) for d in detections]
    # `frame_shape` is the (h, w) of the decoded frame this raw result came
    # from, when the caller still has it in hand -- `run_active_learning` does,
    # from the reads that fed the batched detection pass. Edge-proximity is
    # meaningless without a real frame extent, so when no shape is supplied
    # (any caller scoring purely from a cached raw result) it is honestly
    # zeroed rather than scored against a fake extent -- mirroring
    # `data/dataset_generation.py::FrameQualityScorer.score_frame`. The crowd
    # component is shape-independent (pairwise polygon overlap only) either
    # way.
    if frame_shape is None:
        crowd, _ = score_crowd(obb_corners, frame_shape=(1, 1))
        edge = 0.0
    else:
        crowd, edge = score_crowd(obb_corners, frame_shape=frame_shape)

    nms = score_nms_instability(
        raw_obb_result, obb_config, base_conf=base_conf, base_iou=base_iou
    )

    signal = ALSignals(
        frame_id=frame_id,
        n_detections=len(detections),
        mean_confidence=float(np.mean(confidences)) if confidences else float("nan"),
        uncertainty_score=uncertainty,
        nms_instability=nms,
        count_deviation=count_dev,
        crowd_score=crowd,
        fragmentation_score=score_fragmentation(obb_corners),
        edge_score=edge,
    )
    return signal, detections


def _records_from_detections(detections: list) -> list[LabelRecord]:
    """Convert detector tuples into LabelRecords, polygon-first."""
    records: list[LabelRecord] = []
    for rec in detections:
        cx, cy, ww, hh, theta, conf = rec[:6]
        polygon = rec[6] if len(rec) > 6 else None
        if polygon is not None:
            pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            level = GeometryLevel.POLYGON
        else:
            pts = _detection_corners(cx, cy, ww, hh, theta)
            level = GeometryLevel.OBB
        records.append(
            LabelRecord(
                class_id=0,
                confidence=float(conf),
                points=pts,
                level=level,
            )
        )
    return records


def run_active_learning(
    req: ALRequest,
    progress: Callable[[int, str], None] | None = None,
) -> ALResult:
    """Execute one AL round end-to-end. Pure function for testability.

    Three phases, in place of the old strictly-sequential
    decode -> detect -> score loop (which cost four uncached, unbatched model
    calls per candidate frame):

    1. Candidate selection -- one sequential decode of the source, with the
       motion prefilter and windowed dedup running inline, before any model
       call (`build_candidate_pool`).
    2. Detection -- the whole candidate list goes through ONE batched,
       cached pass (`get_or_compute_raw`), so the GPU sees many frames per
       call and a detection cache shared with tracking is read/written.
    3. Scoring -- every signal, including NMS instability, is derived from
       each frame's cached raw `OBBResult` by re-running the cheap NumPy
       filtering gate; no further model calls happen.

    Everything from selection (`select`) onward is unchanged from the
    pre-restructure implementation.
    """
    if req.detector is None or not str(req.detector.model_path or "").strip():
        raise ValueError(
            "ALRequest.detector must be set to an ALDetectorSpec carrying the "
            "active model path (the caller resolves the project's model)"
        )

    weights = req.weights_override or PRESETS.get(req.preset, PRESETS["balanced"])

    source = _build_frame_source(req)
    try:
        return _run_active_learning_with_source(req, source, weights, progress)
    finally:
        # Ruling: the round owns the source's OS handles. `VideoFrameSource`
        # holds one `cv2.VideoCapture` open across all its reads, and this
        # function reads from the source in all three phases AND again during
        # export (the readability probe and `_LazyALImages`), so the earliest
        # correct release point is the end of the round -- not the end of
        # phase 2. Sources without a `close()` (folder/project, test doubles)
        # simply have nothing to release.
        close = getattr(source, "close", None)
        if callable(close):
            close()


def _run_active_learning_with_source(
    req: ALRequest,
    source: FrameSource,
    weights: AcquisitionWeights,
    progress: Callable[[int, str], None] | None,
) -> ALResult:
    # --- Phase 1: candidate selection (one sequential decode, no model) -----
    if progress:
        progress(5, "Building candidate pool...")
    candidates = build_candidate_pool(source, req.candidate_pool)
    if not candidates:
        raise RuntimeError(
            "0 candidates after FilterKit dedup; relax threshold or stride."
        )
    cap = req.candidate_pool.max_candidates
    if cap is not None and len(candidates) >= cap:
        # The cap truncates from the start of the source, so say so rather
        # than silently scoring only the opening of a long video.
        logger.info(
            "Candidate pool hit its %d-frame cap; only the first %d distinct "
            "frames of the source are scored this round. Raise "
            "CandidatePoolConfig.max_candidates (watch memory) or read the "
            "source with a stride to spread coverage.",
            cap,
            cap,
        )

    # --- Phase 2: one batched, cached detection pass over all candidates ----
    if progress:
        progress(25, f"Detecting on {len(candidates)} candidates...")
    runner, obb_config = _build_detection_context(req)

    readable: list[FrameRef] = []
    frames: list[np.ndarray] = []
    frame_shapes: dict[int, tuple[int, int]] = {}
    for ref in candidates:
        img = source.read(ref)
        if img is None:
            # Same log-and-skip contract the old per-frame scoring loop had
            # for an unreadable candidate: it simply never gets scored.
            continue
        readable.append(ref)
        frames.append(img)
        frame_shapes[ref.frame_id] = (int(img.shape[0]), int(img.shape[1]))
    if not readable:
        raise RuntimeError("No candidate frame could be decoded; nothing to score.")

    frame_indices = [ref.frame_id for ref in readable]
    with _al_detection_cache_dir(req) as cache_dir:
        raw_by_idx = get_or_compute_raw(runner, cache_dir, frames, frame_indices)
    # The pixels are not needed past detection -- only each frame's extent,
    # already captured in `frame_shapes` for the edge-proximity signal.
    del frames

    # --- Phase 3: score every candidate from its cached raw result ----------
    if progress:
        progress(60, f"Scoring {len(readable)} candidates...")
    signals: list[ALSignals] = []
    detections_by_id: dict[int, list] = {}
    frame_refs_by_id: dict[int, object] = {}
    for i, ref in enumerate(readable):
        sig, dets = _frame_signals(
            ref.frame_id,
            raw_by_idx[ref.frame_id],
            obb_config,
            req.expected_count,
            req.base_conf,
            req.base_iou,
            frame_shape=frame_shapes[ref.frame_id],
        )
        signals.append(sig)
        detections_by_id[ref.frame_id] = dets
        frame_refs_by_id[ref.frame_id] = ref
        if progress and i % 10 == 0:
            pct = 60 + int(20 * i / max(len(readable), 1))
            progress(pct, f"Scoring {i}/{len(readable)}")

    if progress:
        progress(85, "Selecting top-K frames...")
    rng = np.random.default_rng()
    picked_ids = select(
        signals,
        weights=weights,
        k=req.budget,
        diversity_window=req.diversity_window,
        probabilistic=req.probabilistic,
        rng=rng if req.probabilistic else None,
    )

    if progress:
        progress(95, "Writing dataset...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    proj_dir = Path(req.project.project_dir)
    source_root = proj_dir / "sources" / f"al_round_{timestamp}"

    # Readability is probed (and the frame discarded) up front, same as a
    # picked frame that fails re-read did under the old hand-rolled writer:
    # log-and-skip rather than aborting the whole round. Frames that pass the
    # probe are read a second time by `export_al_dataset`'s lazy `images`
    # mapping below -- an extra decode, but it keeps this function from
    # holding every picked frame resident at once (see `_LazyALImages`).
    written_ids: list[int] = []
    for fid in picked_ids:
        ref = frame_refs_by_id[fid]
        if source.read(ref) is None:
            logger.warning("Could not re-read picked frame %s; skipping.", fid)
            continue
        written_ids.append(fid)

    requested_levels = [
        GeometryLevel.from_str(lbl) for lbl in (req.export_levels or [req.export_level])
    ]
    # `native_level` is the model's real ceiling, independent of what was
    # requested -- `export_al_dataset` raises if any requested level exceeds
    # it. Deriving this from `requested_levels` itself would make that guard
    # tautological (a request can never exceed a limit computed from that
    # same request), which is exactly the bug this task closes: a
    # zero-detection round (every picked frame has 0 detections -- plausible
    # for uncertainty-driven top-K picks) never reaches `derive_down`'s
    # per-record check either, so this is the only real gate in that case.
    native_level = GeometryLevel.from_str(req.native_level)

    exported = [
        ExportedFrame(
            frame_id=fid,
            image_name=f"f_{fid:06d}.jpg",
            records=_records_from_detections(detections_by_id[fid]),
        )
        for fid in written_ids
    ]

    provenance = {
        "input_kind": req.input_kind,
        "input_path": req.input_path,
        "preset": req.preset,
        "budget": req.budget,
        "expected_count": req.expected_count,
        "base_conf": req.base_conf,
        "base_iou": req.base_iou,
    }

    manifest = export_al_dataset(
        round_dir=source_root,
        frames=exported,
        images=_LazyALImages(source, frame_refs_by_id, written_ids),
        native_level=native_level,
        levels=requested_levels,
        class_names=[req.project.class_name],
        provenance=provenance,
    )

    # ONE OBBSource for the round's authoritative root only -- the root
    # export_al_dataset marks derived_from=None (the highest level actually
    # requested, which equals native_level whenever native_level itself was
    # among the requested levels -- see data/al/export.py's _write_root).
    # The exporter still writes every requested level's sibling folder to
    # disk (data/al/export.py is unchanged) -- those siblings are simply not
    # registered as separate project sources; training derives lower levels
    # from the registered source on demand, same as any other source.
    authoritative_root = next(
        (
            root_meta
            for root_meta in manifest["roots"]
            if root_meta["derived_from"] is None
        ),
        None,
    )
    if authoritative_root is None:
        raise RuntimeError(
            "AL round manifest has no authoritative root (derived_from=None "
            "entry) -- this indicates a corrupt or incompatible manifest."
        )
    req.project.sources.append(
        OBBSource(
            path=authoritative_root["path"],
            name=f"al_round_{timestamp}",
            validated=False,
            original_path=req.input_path,
            source_kind="detectkit_al",
            imported=True,
            level=authoritative_root["level"],
            reviewed=bool(authoritative_root["reviewed"]),
            derived_from=None,
        )
    )

    source_path = authoritative_root["path"]

    if progress:
        progress(100, "Active learning complete")

    # `written_ids` is the pre-export list: it counts every frame that passed
    # the readability probe. The exporter then drops frames whose records did
    # not survive (see `frames_skipped_no_records`), so reporting the probe
    # count claimed more images than exist on disk. Take the truth from the
    # manifest, and report the frames actually written.
    # From the manifest, not recomputed from `exported`: the exporter also
    # drops degenerate records, which can empty a frame that still looks
    # non-empty in the list handed to it.
    exported_ids = [int(fid) for fid in manifest["selected_frame_ids"]]
    skipped = int(manifest["totals"].get("frames_skipped_no_records", 0))
    if skipped:
        logger.warning(
            "%d of %d picked frame(s) carried no label geometry and were not "
            "exported.",
            skipped,
            len(written_ids),
        )

    return ALResult(
        source_path=source_path,
        n_picked=int(manifest["totals"]["frames_exported"]),
        selected_frames=exported_ids,
    )


class ALWorker(BaseWorker):
    """QThread wrapper around run_active_learning.

    Uses the inherited BaseWorker signals (`progress`, `status`, `error`)
    plus an AL-specific `result_ready(source_path, n_picked, selected_frames)`
    signal carrying the structured result. `QThread.finished` (parameterless)
    is inherited and emitted automatically by Qt when run() returns.
    """

    result_ready = Signal(str, int, list)

    def __init__(self, request: ALRequest):
        super().__init__()
        self._request = request

    def execute(self):
        def cb(pct, msg):
            if self._should_stop():
                return
            self.progress.emit(int(pct))
            self.status.emit(str(msg))

        result = run_active_learning(self._request, progress=cb)
        if not self._should_stop():
            self.result_ready.emit(
                result.source_path,
                result.n_picked,
                list(result.selected_frames),
            )

    def _should_stop(self) -> bool:
        return bool(self.isInterruptionRequested())
