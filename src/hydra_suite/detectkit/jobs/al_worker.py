"""Active learning worker for DetectKit projects."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Sequence

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.data.al.acquisition import PRESETS, AcquisitionWeights, select
from hydra_suite.data.al.candidate_pool import CandidatePoolConfig, build_candidate_pool
from hydra_suite.data.al.frame_source import (
    DetectKitProjectSource,
    FrameSource,
    ImageFolderFrameSource,
    VideoFrameSource,
)
from hydra_suite.data.al.signals import (
    ALSignals,
    score_count_deviation,
    score_crowd,
    score_nms_instability,
    score_uncertainty,
)
from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
from hydra_suite.utils.geometry import obb_corners_from_dims as _detection_corners
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


@dataclass
class ALResult:
    """Outcome of one AL round."""

    source_path: str
    n_picked: int
    selected_frames: list[int]


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
    mean_conf, margin = score_uncertainty(confidences, conf_floor=base_conf)
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
        mean_confidence=mean_conf,
        margin=margin,
        nms_instability=nms,
        count_deviation=count_dev,
        crowd_score=crowd,
        edge_score=edge,
    )
    return signal, detections


def _write_geometry_label(
    path: Path, records: list, frame_size: tuple[int, int]
) -> None:
    """Write YOLO labels: a native polygon when present, else OBB corners.

    Each record is a 6-tuple ``(cx, cy, w, h, theta, conf)`` or a 7-tuple with
    a trailing native polygon (an ``(P, 2)`` pixel-space array, or ``None``).
    When the polygon is absent, output is byte-identical to the legacy
    OBB-corner writer.
    """
    h, w = frame_size
    with path.open("w") as fp:
        for rec in records:
            cx, cy, ww, hh, theta, _conf = rec[:6]
            polygon = rec[6] if len(rec) > 6 else None
            if polygon is not None:
                pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
            else:
                pts = _detection_corners(cx, cy, ww, hh, theta)
            pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
            pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
            line = "0 " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)) + "\n"
            fp.write(line)


def _write_yolo_obb_label(
    path: Path, detections: list, frame_size: tuple[int, int]
) -> None:
    """Back-compat alias for :func:`_write_geometry_label`."""
    _write_geometry_label(path, detections, frame_size)


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
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written_ids: list[int] = []
    for fid in picked_ids:
        ref = frame_refs_by_id[fid]
        img = source.read(ref)
        if img is None:
            logger.warning("Could not re-read picked frame %s; skipping.", fid)
            continue
        dets = detections_by_id[fid]
        img_path = images_dir / f"f_{fid:06d}.jpg"
        cv2.imwrite(str(img_path), img)
        _write_geometry_label(
            labels_dir / f"f_{fid:06d}.txt",
            dets,
            frame_size=img.shape[:2],
        )
        written_ids.append(fid)

    (source_root / "classes.txt").write_text(req.project.class_name + "\n")

    new_source = OBBSource(
        path=str(source_root),
        name=f"al_round_{timestamp}",
        validated=False,
        original_path=req.input_path,
        source_kind="detectkit_al",
        imported=True,
        level=req.export_level,
    )
    req.project.sources.append(new_source)

    if progress:
        progress(100, "Active learning complete")

    return ALResult(
        source_path=str(source_root),
        n_picked=len(written_ids),
        selected_frames=written_ids,
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
