"""Pure end-of-session summary builder (Qt-free), shared by GUI and CLI."""

from __future__ import annotations

import os
from typing import Any, Mapping

from hydra_suite.core.tracking import session_policy


def build_session_summary_lines(
    config: Mapping[str, Any], result: Mapping[str, Any]
) -> list[str]:
    """Build end-of-session summary lines from a config dict + a runtime result dict."""
    lines: list[str] = []

    wall = result.get("wall_seconds")
    if wall is not None:
        h = int(wall // 3600)
        m = int((wall % 3600) // 60)
        s = int(wall % 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
        lines.append(f"Duration: {elapsed_str}")

    frames = int(result.get("frames_processed") or 0)
    if frames > 0:
        lines.append(f"Frames processed: {frames}")
    fps_vals = [f for f in (result.get("fps_list") or []) if f and f > 0]
    if fps_vals:
        avg_fps = sum(fps_vals) / len(fps_vals)
        lines.append(f"Average FPS: {avg_fps:.1f}")

    video_path = result.get("video_path")
    if video_path:
        lines.append(f"Video: {os.path.basename(video_path)}")
    csv_path = result.get("csv_path")
    if csv_path:
        lines.append(f"Output CSV: {os.path.basename(csv_path)}")

    traj_count = result.get("trajectory_count")
    if traj_count is not None:
        lines.append(f"Trajectories: {int(traj_count)}")

    pipelines = []
    if bool(config.get("enable_postprocessing")):
        pipelines.append("Post-processing")
    if bool(config.get("enable_backward_tracking")):
        pipelines.append("Backward tracking")
    if session_policy.is_individual_pipeline_enabled(config):
        pipelines.append("Individual analysis")
        if bool(config.get("enable_pose_extractor")):
            pipelines.append("Pose extraction")
    if pipelines:
        lines.append("Pipelines: " + ", ".join(pipelines))

    lines.append("")

    dataset = result.get("dataset")
    if dataset is not None:
        if dataset.get("success"):
            lines.append(
                f"✓ Dataset generated: {dataset['num_frames']} frame(s)"
                f"\n  Location: {dataset['dir']}"
            )
        else:
            lines.append(
                f"✗ Dataset generation failed: {dataset.get('error', 'unknown error')}"
            )

    return lines
