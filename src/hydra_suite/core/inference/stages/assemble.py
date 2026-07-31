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


def suppress_foreign_keypoints(
    keypoints,
    target_corners,
    foreign_corners_list,
) -> np.ndarray:
    """Zero confidence of keypoints landing inside any foreign (other-animal) OBB.

    Delegates to ``hydra_suite.utils.geometry.filter_keypoints_by_foreign_obbs``.
    Operates on frame-space coordinates.

    Args:
        keypoints: ``[K, 3]`` float32 array of ``(x, y, conf)`` in frame coords,
            or ``None``.
        target_corners: ``(4, 2)`` float32 OBB corner array for the current
            detection — present for API symmetry; its own OBB is never suppressed.
        foreign_corners_list: List of ``(4, 2)`` float32 OBB corner arrays for
            all *other* detections in the frame.

    Returns:
        Modified copy of *keypoints* with contaminated entries having ``conf=0``.
        ``None`` is passed through unchanged.
    """
    if keypoints is None:
        return keypoints

    from hydra_suite.utils.geometry import filter_keypoints_by_foreign_obbs

    # Build a unified corners list: target at index 0, foreigners after it.
    # filter_keypoints_by_foreign_obbs skips target_idx (=0) automatically.
    all_corners = [target_corners] + list(foreign_corners_list)
    return filter_keypoints_by_foreign_obbs(keypoints, all_corners, target_idx=0)


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

        # Apply foreign-OBB keypoint suppression when the pose config enables it.
        # PoseResult.keypoints is (D, K, 3) in frame (image) coordinates — the
        # correct space for OBB corner comparison.
        suppress = (
            pose_result is not None
            and config is not None
            and getattr(config, "pose", None) is not None
            and getattr(config.pose, "suppress_foreign_regions", False)
            and n > 1
        )
        if suppress:
            all_corners = [obb.corners[i] for i in range(n)]
            suppressed_kpts = pose_result.keypoints.copy()
            for i in range(n):
                target_corners = all_corners[i]
                foreign = [all_corners[j] for j in range(n) if j != i]
                suppressed_kpts[i] = suppress_foreign_keypoints(
                    pose_result.keypoints[i], target_corners, foreign
                )
            pose_result = PoseResult(
                keypoints=suppressed_kpts,
                valid_mask=pose_result.valid_mask,
            )

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
