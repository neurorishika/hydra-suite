"""Active learning worker for DetectKit projects."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from PySide6.QtCore import Signal

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
from hydra_suite.data.al.signals import (
    ALSignals,
    score_count_deviation,
    score_crowd,
    score_fragmentation,
    score_nms_instability,
    score_uncertainty,
)
from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
from hydra_suite.utils.geometry import obb_corners_from_dims as _detection_corners
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)

Detection = tuple  # (cx, cy, w, h, theta, conf)
DetectorFn = Callable[[np.ndarray, float, float], Sequence[Detection]]


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
    detector_fn: DetectorFn | None = None
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


def _frame_signals(
    frame: np.ndarray,
    frame_id: int,
    detector_fn: DetectorFn,
    expected_count: int,
    base_conf: float,
    base_iou: float,
) -> tuple[ALSignals, list]:
    # Detections may be 6-tuples (cx,cy,w,h,theta,conf) or 7-tuples with a
    # trailing native polygon (Task 14 export path); d[5] and d[:5] are valid
    # for both, so no branching is needed here.
    detections = list(detector_fn(frame, base_conf, base_iou))
    confidences = [d[5] for d in detections]
    uncertainty = score_uncertainty(confidences, conf_floor=base_conf)
    count_dev = score_count_deviation(len(detections), expected_count)

    h, w = frame.shape[:2]
    obb_corners = [_detection_corners(*d[:5]) for d in detections]
    crowd, edge = score_crowd(obb_corners, frame_shape=(h, w))

    nms = score_nms_instability(
        frame, detector_fn, base_conf=base_conf, base_iou=base_iou
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
    """Execute one AL round end-to-end. Pure function for testability."""
    if req.detector_fn is None:
        raise ValueError(
            "ALRequest.detector_fn must be set (model must be loaded by caller)"
        )

    weights = req.weights_override or PRESETS.get(req.preset, PRESETS["balanced"])

    if progress:
        progress(5, "Building candidate pool...")
    source = _build_frame_source(req)
    candidates = build_candidate_pool(source, req.candidate_pool)
    if not candidates:
        raise RuntimeError(
            "0 candidates after FilterKit dedup; relax threshold or stride."
        )

    if progress:
        progress(20, f"Scoring {len(candidates)} candidates...")
    signals: list[ALSignals] = []
    detections_by_id: dict[int, list] = {}
    frame_refs_by_id: dict[int, object] = {}
    for i, ref in enumerate(candidates):
        img = source.read(ref)
        if img is None:
            continue
        sig, dets = _frame_signals(
            img,
            ref.frame_id,
            req.detector_fn,
            req.expected_count,
            req.base_conf,
            req.base_iou,
        )
        signals.append(sig)
        detections_by_id[ref.frame_id] = dets
        frame_refs_by_id[ref.frame_id] = ref
        if progress and i % 10 == 0:
            pct = 20 + int(60 * i / max(len(candidates), 1))
            progress(pct, f"Scoring {i}/{len(candidates)}")

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
