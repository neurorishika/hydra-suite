"""Utilities for filtering and validating pose keypoints before rendering."""

from __future__ import annotations

import math


def normalize_pose_render_min_conf(value: object, default: float = 0.2) -> float:
    """Return a finite, non-negative confidence threshold for pose rendering."""
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float(default)
    if not math.isfinite(threshold):
        threshold = float(default)
    return max(0.0, threshold)


def is_renderable_pose_keypoint(
    x_value: object,
    y_value: object,
    confidence_value: object,
    min_confidence: object = 0.0,
) -> bool:
    """Return True only for finite coordinates with strictly positive confidence."""
    try:
        x_coord = float(x_value)
        y_coord = float(y_value)
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        return False

    threshold = normalize_pose_render_min_conf(min_confidence, default=0.0)
    return (
        math.isfinite(x_coord)
        and math.isfinite(y_coord)
        and math.isfinite(confidence)
        and confidence > 0.0
        and confidence >= threshold
    )


def draw_pose_skeleton(
    frame,
    kpts_arr,
    pose_edges,
    color,
    *,
    min_conf: float,
    line_thickness: int,
    point_radius: int,
    point_thickness: int,
) -> None:
    """Draw one animal's skeleton edges, then keypoints, onto ``frame`` in place.

    ``kpts_arr`` is ``(K, 3)`` of ``(x, y, conf)`` already in ``frame`` pixels.
    Shared by the annotated full-frame video and the per-track canonical videos
    so both overlays apply the same visibility rule and draw order.
    """
    import cv2

    if pose_edges:
        for e0, e1 in pose_edges:
            if e0 < 0 or e1 < 0 or e0 >= len(kpts_arr) or e1 >= len(kpts_arr):
                continue
            if not is_renderable_pose_keypoint(
                kpts_arr[e0, 0],
                kpts_arr[e0, 1],
                kpts_arr[e0, 2],
                min_conf,
            ) or not is_renderable_pose_keypoint(
                kpts_arr[e1, 0],
                kpts_arr[e1, 1],
                kpts_arr[e1, 2],
                min_conf,
            ):
                continue
            cv2.line(
                frame,
                (
                    int(round(float(kpts_arr[e0, 0]))),
                    int(round(float(kpts_arr[e0, 1]))),
                ),
                (
                    int(round(float(kpts_arr[e1, 0]))),
                    int(round(float(kpts_arr[e1, 1]))),
                ),
                color,
                line_thickness,
            )
    for kpt in kpts_arr:
        if not is_renderable_pose_keypoint(kpt[0], kpt[1], kpt[2], min_conf):
            continue
        cv2.circle(
            frame,
            (int(round(float(kpt[0]))), int(round(float(kpt[1])))),
            point_radius,
            color,
            point_thickness,
        )


__all__ = [
    "draw_pose_skeleton",
    "is_renderable_pose_keypoint",
    "normalize_pose_render_min_conf",
]
