"""Tracking overlay drawing utilities.

Standalone functions for rendering tracking visualizations (trajectory lines,
orientation arrows, Kalman uncertainty ellipses, OBB boxes, labels, etc.)
onto video frames.
"""

import math
import zlib

import cv2
import numpy as np


def draw_uncertainty_ellipses(overlay, kf_manager, params, track_states):
    """Draw Kalman filter uncertainty ellipses for debugging.

    Args:
        overlay: BGR frame to draw on (modified in-place).
        kf_manager: KalmanFilterManager with .P and .X attributes.
        params: Parameter dict (needs ``TRAJECTORY_COLORS``).
        track_states: List of track state strings.
    """
    if kf_manager is None:
        return

    P = kf_manager.P  # Shape: (N, 5, 5)
    X = kf_manager.X  # Shape: (N, 5)
    colors = params.get("TRAJECTORY_COLORS", [(255, 0, 0)] * len(X))

    for i in range(len(X)):
        if track_states[i] == "lost":
            continue

        x, y = X[i, 0], X[i, 1]
        P_pos = P[i, :2, :2]

        eigenvalues, eigenvectors = np.linalg.eig(P_pos)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # 95% confidence ellipse (chi-square for 2D)
        scale = np.sqrt(5.991)
        width = 2 * scale * np.sqrt(max(0, eigenvalues[0]))
        height = 2 * scale * np.sqrt(max(0, eigenvalues[1]))
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        center = (int(x), int(y))
        axes = (int(width), int(height))
        color = tuple(int(c) for c in colors[i])
        cv2.ellipse(overlay, center, axes, angle, 0, 360, color, 2)


def _draw_obb_from_corners(overlay, obb_corners):
    """Draw OBB boxes from cached corner arrays."""
    for corners in obb_corners:
        if corners is not None:
            corners_int = corners.astype(np.int32)
            cv2.polylines(
                overlay,
                [corners_int],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )


def _draw_obb_from_yolo_results(overlay, yolo_results):
    """Draw OBB boxes from a live YOLO results object."""
    if (
        not hasattr(yolo_results, "obb")
        or yolo_results.obb is None
        or len(yolo_results.obb) == 0
    ):
        return
    obb_data = yolo_results.obb
    for i in range(len(obb_data)):
        corners = obb_data.xyxyxyxy[i].cpu().numpy().astype(np.int32)
        cv2.polylines(
            overlay,
            [corners],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
        )
        if hasattr(obb_data, "conf"):
            conf = obb_data.conf[i].cpu().item()
            cx = int(corners[:, 0].mean())
            cy = int(corners[:, 1].mean())
            cv2.putText(
                overlay,
                f"{conf:.2f}",
                (cx - 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )


def _draw_yolo_obb(overlay, obb_corners, yolo_results):
    """Draw YOLO OBB boxes from cached corners or live results."""
    if obb_corners is not None and len(obb_corners) > 0:
        _draw_obb_from_corners(overlay, obb_corners)
    elif yolo_results is not None:
        _draw_obb_from_yolo_results(overlay, yolo_results)


def _draw_track_overlays(
    overlay, p, trajectories, track_states, ids, continuity, identity_labels=None
):
    """Draw per-track circles, orientation arrows, trajectories, and labels."""
    colors = p["TRAJECTORY_COLORS"]
    for i, tr in enumerate(trajectories):
        if not tr or track_states[i] == "lost":
            continue
        x, y, th, _ = tr[-1]
        if math.isnan(x):
            continue
        pt = (int(x), int(y))
        # Use identity label for a stable color when available; fall back to slot index.
        identity_lbl = (identity_labels[i] if identity_labels and i < len(identity_labels) else None) or ""
        if identity_lbl:
            col_idx = zlib.crc32(identity_lbl.encode()) % len(colors)
        else:
            col_idx = i % len(colors)
        col = tuple(int(c) for c in colors[col_idx])
        if p.get("SHOW_CIRCLES"):
            cv2.circle(overlay, pt, 8, col, -1)
        if p.get("SHOW_ORIENTATION"):
            ex, ey = int(x + 20 * math.cos(th)), int(y + 20 * math.sin(th))
            cv2.line(overlay, pt, (ex, ey), col, 2)
        if p.get("SHOW_TRAJECTORIES"):
            pts = np.array(
                [(pt[0], pt[1]) for pt in tr if not math.isnan(pt[0])],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            if len(pts) > 1:
                cv2.polylines(overlay, [pts], isClosed=False, color=col, thickness=2)
        if p.get("SHOW_LABELS") or p.get("SHOW_STATE"):
            if p.get("SHOW_LABELS"):
                label = f"{identity_lbl} C:{continuity[i]}" if identity_lbl else f"T{ids[i]} C:{continuity[i]}"
            else:
                label = ""
            state = f" [{track_states[i]}]" if p.get("SHOW_STATE") else ""
            cv2.putText(
                overlay,
                f"{label}{state}",
                (pt[0] + 15, pt[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                2,
            )


def draw_overlays(
    overlay,
    p,
    trajectories,
    track_states,
    ids,
    continuity,
    fg,
    bg,
    kf_manager=None,
    yolo_results=None,
    obb_corners=None,
    identity_labels=None,
):
    """Draw all tracking overlays on a frame.

    Args:
        overlay: BGR frame to draw on (modified in-place).
        p: Parameter dict.
        trajectories: Per-track trajectory lists.
        track_states: Per-track state strings.
        ids: Per-track ID labels.
        continuity: Per-track continuity counts.
        fg: Foreground mask (or None).
        bg: Background model (or None).
        kf_manager: KalmanFilterManager (for uncertainty ellipses).
        yolo_results: YOLO results object (direct detection mode).
        obb_corners: OBB corners list (cached detection mode).
        identity_labels: Per-slot identity label strings (or None). When a
            non-empty string is provided for a slot, it is used as the display
            label and drives a stable color assignment so the same identity
            always renders with the same color.
    """
    if p.get("SHOW_YOLO_OBB", False):
        _draw_yolo_obb(overlay, obb_corners, yolo_results)

    if p.get("SHOW_KALMAN_UNCERTAINTY", False):
        draw_uncertainty_ellipses(overlay, kf_manager, p, track_states)

    if any(
        p.get(k)
        for k in [
            "SHOW_CIRCLES",
            "SHOW_ORIENTATION",
            "SHOW_TRAJECTORIES",
            "SHOW_LABELS",
            "SHOW_STATE",
        ]
    ):
        _draw_track_overlays(
            overlay, p, trajectories, track_states, ids, continuity, identity_labels
        )

    if p.get("SHOW_FG") and fg is not None:
        small_fg = cv2.resize(fg, (0, 0), fx=0.3, fy=0.3)
        overlay[0 : small_fg.shape[0], 0 : small_fg.shape[1]] = cv2.cvtColor(
            small_fg, cv2.COLOR_GRAY2BGR
        )
    if p.get("SHOW_BG") and bg is not None:
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        small_bg = cv2.resize(bg_bgr, (0, 0), fx=0.3, fy=0.3)
        overlay[0 : small_bg.shape[0], -small_bg.shape[1] :] = small_bg
