"""Map per-animal stage results between the N-free superset and the final N set.

Per-animal stages run on the superset (every filter survivor) and are cached
by RAW detection-cache index; consumers see results positionally aligned with
the final filtered OBB. These pure helpers do that translation in one place.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
)


class DownstreamCacheError(RuntimeError):
    """A replayed detection has no per-animal result (incoherent cache)."""


def positions_in(superset_idx: np.ndarray, final_idx: np.ndarray) -> np.ndarray:
    where = {int(v): i for i, v in enumerate(np.asarray(superset_idx).tolist())}
    out = []
    for v in np.asarray(final_idx).tolist():
        if int(v) not in where:
            raise DownstreamCacheError(
                f"detection index {int(v)} has no per-animal result in the "
                "superset; the inference cache set is incoherent -- rebuild it."
            )
        out.append(where[int(v)])
    return np.asarray(out, dtype=np.int64)


def select_headtail(
    ht: HeadTailResult | None, pos: np.ndarray
) -> HeadTailResult | None:
    if ht is None:
        return None
    return HeadTailResult(
        heading_hints=ht.heading_hints[pos],
        heading_confidences=ht.heading_confidences[pos],
        directed_mask=ht.directed_mask[pos],
        canonical_affines=(
            None if ht.canonical_affines is None else ht.canonical_affines[pos]
        ),
    )


def select_pose(p: PoseResult | None, pos: np.ndarray) -> PoseResult | None:
    if p is None:
        return None
    out = PoseResult(keypoints=p.keypoints[pos], valid_mask=p.valid_mask[pos])
    overrides = getattr(p, "heading_overrides", None)
    if overrides is not None:
        out.heading_overrides = np.asarray(overrides)[pos]
    return out


def select_cnn(r: CNNResult, pos: np.ndarray) -> CNNResult:
    by_index = {p.det_index: p for p in r.predictions}
    preds = [
        replace(by_index[int(src)], det_index=dst)
        for dst, src in enumerate(np.asarray(pos).tolist())
        if int(src) in by_index
    ]
    return CNNResult(label=r.label, predictions=preds)


def cnn_positions_to_raw(r: CNNResult, superset_idx: np.ndarray) -> CNNResult:
    sup = np.asarray(superset_idx)
    return CNNResult(
        label=r.label,
        predictions=[
            replace(p, det_index=int(sup[p.det_index])) for p in r.predictions
        ],
    )


def cnn_raw_to_positions(
    preds: list[CNNDetectionPrediction], final_idx: np.ndarray, label: str
) -> CNNResult:
    by_raw = {p.det_index: p for p in preds}
    out = []
    for dst, raw in enumerate(np.asarray(final_idx).tolist()):
        if int(raw) not in by_raw:
            raise DownstreamCacheError(
                f"CNN '{label}' has no prediction for detection index {int(raw)}."
            )
        out.append(replace(by_raw[int(raw)], det_index=dst))
    return CNNResult(label=label, predictions=out)


def _at_subset(
    at: AprilTagResult, rows: list[int], det_indices: list[int]
) -> AprilTagResult:
    return AprilTagResult(
        tag_ids=[at.tag_ids[i] for i in rows],
        det_indices=det_indices,
        centers=at.centers[rows] if rows else np.zeros((0, 2), np.float32),
        corners=at.corners[rows] if rows else np.zeros((0, 4, 2), np.float32),
    )


def select_apriltag(
    at: AprilTagResult | None, pos: np.ndarray
) -> AprilTagResult | None:
    if at is None:
        return None
    new_of = {int(src): dst for dst, src in enumerate(np.asarray(pos).tolist())}
    rows = [i for i, d in enumerate(at.det_indices) if int(d) in new_of]
    return _at_subset(at, rows, [new_of[int(at.det_indices[i])] for i in rows])


def apriltag_positions_to_raw(at: AprilTagResult | None, superset_idx: np.ndarray):
    if at is None:
        return None
    sup = np.asarray(superset_idx)
    rows = list(range(len(at.tag_ids)))
    return _at_subset(at, rows, [int(sup[d]) for d in at.det_indices])


def apriltag_raw_to_positions(at: AprilTagResult | None, final_idx: np.ndarray):
    if at is None:
        return None
    new_of = {int(raw): dst for dst, raw in enumerate(np.asarray(final_idx).tolist())}
    rows = [i for i, d in enumerate(at.det_indices) if int(d) in new_of]
    return _at_subset(at, rows, [new_of[int(at.det_indices[i])] for i in rows])


def split_rows(obb: OBBResult, size: int) -> list[tuple[int, OBBResult]]:
    from .stages.filtering import _select

    n = obb.num_detections
    if n <= size:
        return [(0, obb)]
    return [
        (start, _select(obb, np.arange(start, min(n, start + size))))
        for start in range(0, n, size)
    ]


def concat_headtail(parts: list[HeadTailResult | None]) -> HeadTailResult | None:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    affines = [p.canonical_affines for p in parts]
    return HeadTailResult(
        heading_hints=np.concatenate([p.heading_hints for p in parts]),
        heading_confidences=np.concatenate([p.heading_confidences for p in parts]),
        directed_mask=np.concatenate([p.directed_mask for p in parts]),
        canonical_affines=(
            None if any(a is None for a in affines) else np.concatenate(affines)
        ),
    )


def concat_pose(parts: list[PoseResult | None]) -> PoseResult | None:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    out = PoseResult(
        keypoints=np.concatenate([p.keypoints for p in parts]),
        valid_mask=np.concatenate([p.valid_mask for p in parts]),
    )
    overrides = [getattr(p, "heading_overrides", None) for p in parts]
    if all(o is not None for o in overrides):
        out.heading_overrides = np.concatenate(overrides)
    return out


def concat_cnn(parts: list[tuple[int, CNNResult]]) -> CNNResult:
    label = parts[0][1].label
    preds = [
        replace(p, det_index=offset + p.det_index)
        for offset, r in parts
        for p in r.predictions
    ]
    return CNNResult(label=label, predictions=preds)


def concat_apriltag(
    parts: list[tuple[int, AprilTagResult | None]],
) -> AprilTagResult | None:
    present = [(o, a) for o, a in parts if a is not None]
    if not present:
        return None
    tag_ids, det, centers, corners = [], [], [], []
    for offset, a in present:
        tag_ids += list(a.tag_ids)
        det += [offset + int(d) for d in a.det_indices]
        centers.append(a.centers)
        corners.append(a.corners)
    return AprilTagResult(
        tag_ids, det, np.concatenate(centers), np.concatenate(corners)
    )
