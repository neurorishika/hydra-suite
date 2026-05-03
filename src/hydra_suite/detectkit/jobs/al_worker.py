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


def _detection_corners(cx, cy, ww, hh, theta) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    local = np.array(
        [
            [-ww / 2, -hh / 2],
            [ww / 2, -hh / 2],
            [ww / 2, hh / 2],
            [-ww / 2, hh / 2],
        ],
        dtype=np.float32,
    )
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return local @ rot.T + np.array([cx, cy], dtype=np.float32)


def _frame_signals(
    frame: np.ndarray,
    frame_id: int,
    detector_fn: DetectorFn,
    expected_count: int,
    base_conf: float,
    base_iou: float,
) -> tuple[ALSignals, list]:
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


def _write_yolo_obb_label(
    path: Path, detections: list, frame_size: tuple[int, int]
) -> None:
    h, w = frame_size
    with path.open("w") as fp:
        for cx, cy, ww, hh, theta, _ in detections:
            corners = _detection_corners(cx, cy, ww, hh, theta)
            corners[:, 0] = np.clip(corners[:, 0] / w, 0.0, 1.0)
            corners[:, 1] = np.clip(corners[:, 1] / h, 0.0, 1.0)
            line = "0 " + " ".join(f"{v:.6f}" for v in corners.flatten()) + "\n"
            fp.write(line)


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
    detections_by_id: dict[int, tuple[np.ndarray, list]] = {}
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
        detections_by_id[ref.frame_id] = (img, dets)
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

    for fid in picked_ids:
        img, dets = detections_by_id[fid]
        img_path = images_dir / f"f_{fid:06d}.jpg"
        cv2.imwrite(str(img_path), img)
        _write_yolo_obb_label(
            labels_dir / f"f_{fid:06d}.txt",
            dets,
            frame_size=img.shape[:2],
        )

    (source_root / "classes.txt").write_text(req.project.class_name + "\n")

    new_source = OBBSource(
        path=str(source_root),
        name=f"al_round_{timestamp}",
        validated=False,
        original_path=req.input_path,
        source_kind="detectkit_al",
        imported=True,
    )
    req.project.sources.append(new_source)

    if progress:
        progress(100, "Active learning complete")

    return ALResult(
        source_path=str(source_root),
        n_picked=len(picked_ids),
        selected_frames=picked_ids,
    )


class ALWorker(BaseWorker):
    """QThread wrapper around run_active_learning."""

    progress_signal = Signal(int, str)
    finished_signal = Signal(str, int, list)
    error_signal = Signal(str)

    def __init__(self, request: ALRequest):
        super().__init__()
        self._request = request

    def execute(self):
        try:

            def cb(pct, msg):
                if not self._should_stop():
                    self.progress_signal.emit(int(pct), str(msg))

            result = run_active_learning(self._request, progress=cb)
            if not self._should_stop():
                self.finished_signal.emit(
                    result.source_path,
                    result.n_picked,
                    list(result.selected_frames),
                )
        except Exception as e:
            logger.exception("AL worker failed")
            self.error_signal.emit(str(e))

    def _should_stop(self) -> bool:
        return bool(self.isInterruptionRequested())
