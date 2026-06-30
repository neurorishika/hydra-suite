from __future__ import annotations

from typing import Any

import numpy as np

from ..result import (
    AprilTagResult,
    CNNResult,
    FrameResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
    assemble_resolved_headings,
)


def scatter(
    obb_by_frame: "dict[int, OBBResult]",
    headtail: "dict[int, HeadTailResult] | None",
    cnns: "dict[int, list[CNNResult]] | None",
    pose: "dict[int, PoseResult] | None",
    apriltag: "dict[int, AprilTagResult | None] | None",
    config: Any,
    overrides_headtail: bool = True,
) -> "list[FrameResult]":
    """Build one FrameResult per frame from per-frame stage results.

    Heading resolution priority: pose -> headtail -> OBB axis.
    This mirrors the logic in runner._build_frame_result / assemble_resolved_headings.

    Args:
        obb_by_frame: frame_idx -> OBBResult (filtered).
        headtail: frame_idx -> HeadTailResult, or None if stage not run.
        cnns: frame_idx -> list[CNNResult] (one per CNN phase), or None.
        pose: frame_idx -> PoseResult, or None if stage not run.
        apriltag: frame_idx -> AprilTagResult or None, or None if not run.
        config: InferenceConfig (unused currently; reserved for future flags).
        overrides_headtail: if True, pose heading takes priority over headtail.
    """
    frame_results: list[FrameResult] = []
    for frame_idx in sorted(obb_by_frame):
        obb = obb_by_frame[frame_idx]
        n = obb.num_detections

        ht = headtail.get(frame_idx) if headtail is not None else None
        cnn_list: list[CNNResult] = cnns.get(frame_idx, []) if cnns is not None else []
        pose_result = pose.get(frame_idx) if pose is not None else None
        at_result = apriltag.get(frame_idx) if apriltag is not None else None

        pose_headings: np.ndarray | None = None
        pose_valid: np.ndarray | None = None
        if pose_result is not None:
            pose_headings = getattr(pose_result, "heading_overrides", None)
            pose_valid = pose_result.valid_mask

        resolved = assemble_resolved_headings(
            obb,
            ht,
            pose_headings,
            pose_valid,
            overrides_headtail=overrides_headtail,
        )

        frame_results.append(
            FrameResult(
                frame_idx=frame_idx,
                obb=obb,
                filtered_indices=list(range(n)),
                headtail=ht,
                cnn=cnn_list,
                pose=pose_result,
                apriltag=at_result,
                resolved_headings=resolved,
            )
        )
    return frame_results
