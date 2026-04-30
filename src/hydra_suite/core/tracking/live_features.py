"""Lightweight in-memory feature stores for realtime tracking mode.

These stores mimic the minimal read APIs of the existing pose/tag/CNN caches so
the tracking loop can consume live per-frame analysis outputs before the backing
artifacts are finalized to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hydra_suite.core.identity.classification.cnn import ClassPrediction
from hydra_suite.core.identity.evidence import IdentityEvidence


def _compute_pose_statistics(
    keypoints: Optional[np.ndarray], min_valid_conf: float = 0.2
) -> tuple[float, float, int, int]:
    """Compute pose summary statistics from raw keypoints."""
    if keypoints is None:
        return 0.0, 0.0, 0, 0
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return 0.0, 0.0, 0, 0

    conf_values = np.clip(arr[:, 2], 0.0, 1.0)
    mean_conf = float(np.mean(conf_values))
    valid_mask = conf_values >= float(min_valid_conf)
    valid_fraction = float(np.mean(valid_mask))
    num_valid = int(np.sum(valid_mask))
    num_keypoints = int(len(arr))
    return mean_conf, valid_fraction, num_valid, num_keypoints


class LivePosePropertiesStore:
    """In-memory pose-properties view compatible with the tracking loop."""

    def __init__(self) -> None:
        self._frames: Dict[int, dict[str, Any]] = {}

    def update_frame(
        self,
        frame_idx: int,
        detection_ids: Sequence[int],
        pose_keypoints: Sequence[Optional[np.ndarray]],
    ) -> None:
        ids = [int(det_id) for det_id in detection_ids]
        keypoints = []
        for value in pose_keypoints:
            if value is None:
                keypoints.append(None)
            else:
                keypoints.append(np.asarray(value, dtype=np.float32).copy())
        self._frames[int(frame_idx)] = {
            "detection_ids": ids,
            "pose_keypoints": keypoints,
        }

    def get_frame(self, frame_idx: int, min_valid_conf: float = 0.2) -> Dict[str, Any]:
        frame = self._frames.get(int(frame_idx))
        if frame is None:
            return {
                "detection_ids": [],
                "pose_mean_conf": [],
                "pose_valid_fraction": [],
                "pose_num_valid": [],
                "pose_num_keypoints": [],
                "pose_keypoints": [],
            }

        ids = list(frame.get("detection_ids", []))
        keypoints = list(frame.get("pose_keypoints", []))
        mean_conf = []
        valid_fraction = []
        num_valid = []
        num_keypoints = []
        for kpts in keypoints:
            stats = _compute_pose_statistics(kpts, min_valid_conf)
            mean_conf.append(stats[0])
            valid_fraction.append(stats[1])
            num_valid.append(stats[2])
            num_keypoints.append(stats[3])
        return {
            "detection_ids": ids,
            "pose_mean_conf": mean_conf,
            "pose_valid_fraction": valid_fraction,
            "pose_num_valid": num_valid,
            "pose_num_keypoints": num_keypoints,
            "pose_keypoints": keypoints,
        }

    def close(self) -> None:
        """Match cache interface; nothing to release for in-memory mode."""


class LiveTagObservationStore:
    """In-memory tag-observation view compatible with build_tag_detection_map()."""

    def __init__(self) -> None:
        self._frames: Dict[int, dict[str, np.ndarray]] = {}

    def update_frame(
        self,
        frame_idx: int,
        tag_ids: Sequence[int],
        centers_xy: Sequence[Tuple[float, float]],
        corners: Sequence[np.ndarray],
        det_indices: Sequence[int],
        hammings: Optional[Sequence[int]] = None,
    ) -> None:
        n = len(tag_ids)
        self._frames[int(frame_idx)] = {
            "tag_ids": np.asarray(tag_ids, dtype=np.int32),
            "centers_xy": (
                np.asarray(centers_xy, dtype=np.float32).reshape(n, 2)
                if n > 0
                else np.zeros((0, 2), dtype=np.float32)
            ),
            "corners": (
                np.asarray(corners, dtype=np.float32).reshape(n, 4, 2)
                if n > 0
                else np.zeros((0, 4, 2), dtype=np.float32)
            ),
            "det_indices": np.asarray(det_indices, dtype=np.int32),
            "hammings": np.asarray(
                hammings if hammings is not None else ([0] * n),
                dtype=np.int32,
            ),
        }

    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        frame = self._frames.get(int(frame_idx))
        if frame is None:
            return {
                "tag_ids": np.array([], dtype=np.int32),
                "centers_xy": np.zeros((0, 2), dtype=np.float32),
                "corners": np.zeros((0, 4, 2), dtype=np.float32),
                "det_indices": np.array([], dtype=np.int32),
                "hammings": np.array([], dtype=np.int32),
            }
        return frame

    def close(self) -> None:
        """Match cache interface; nothing to release for in-memory mode."""


class LiveCNNIdentityStore:
    """In-memory CNN prediction store compatible with cnn_build_association_entries()."""

    def __init__(self, catalog_labels: Optional[Sequence[str]] = None) -> None:
        self._frames: Dict[int, List[ClassPrediction]] = {}
        self._evidences: Dict[int, List[IdentityEvidence]] = {}
        self._catalog_labels: Optional[tuple[str, ...]] = (
            tuple(str(label) for label in catalog_labels)
            if catalog_labels is not None
            else None
        )

    @property
    def catalog_labels(self) -> Optional[tuple[str, ...]]:
        return self._catalog_labels

    def set_catalog_labels(self, catalog_labels: Sequence[str]) -> None:
        self._catalog_labels = tuple(str(label) for label in catalog_labels)

    def update_frame(
        self,
        frame_idx: int,
        predictions: Sequence[ClassPrediction],
        posteriors: Optional[Sequence[Optional[Sequence[np.ndarray]]]] = None,
        evidences: Optional[Sequence[IdentityEvidence]] = None,
    ) -> None:
        copied = [
            ClassPrediction(
                det_index=int(pred.det_index),
                factor_names=pred.factor_names,
                class_names=pred.class_names,
                confidences=tuple(float(c) for c in pred.confidences),
            )
            for pred in predictions
        ]
        self._frames[int(frame_idx)] = copied
        if evidences is not None:
            self._evidences[int(frame_idx)] = list(evidences)

    def load(self, frame_idx: int) -> list[ClassPrediction]:
        return list(self._frames.get(int(frame_idx), []))

    def load_evidences(self, frame_idx: int) -> list[IdentityEvidence]:
        return list(self._evidences.get(int(frame_idx), []))
