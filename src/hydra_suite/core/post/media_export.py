"""Qt-free media export: trajectory persistence, coordinate scaling, and the
annotated-video overlay chain. Extracted from trackerkit/gui/orchestrators/tracking.py
(Slice 3 of the headless-session-service program)."""

from __future__ import annotations

import json
import logging
import os
import re

import cv2
import numpy as np
import pandas as pd

from hydra_suite.core.individual.dataset.oriented_video import (
    OrientedTrackVideoExporter,
)
from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.properties.export import build_pose_keypoint_labels
from hydra_suite.utils.pose_visualization import (
    is_renderable_pose_keypoint,
    normalize_pose_render_min_conf,
)
from hydra_suite.utils.profiling import bind_target
from hydra_suite.utils.video_encoder import VideoEncoder

logger = logging.getLogger(__name__)


def scale_trajectories_to_original_space(trajectories_df, resize_factor):
    """Scale trajectory coordinates from resized space back to original video space."""
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if resize_factor == 1.0:
        return trajectories_df
    scale_factor = 1.0 / resize_factor
    logger.info(
        f"Scaling trajectories to original video space (resize_factor={resize_factor:.3f}, scale_factor={scale_factor:.3f})"
    )
    result_df = trajectories_df.copy()
    result_df["X"] = result_df["X"] * scale_factor
    result_df["Y"] = result_df["Y"] * scale_factor
    logger.info(
        f"Scaled {len(result_df)} trajectory points to original video coordinates"
    )
    return result_df


def format_video_track_label(track_id, unique_identity_key=None) -> str:
    """Return the overlay label for one rendered track row."""
    token = str(unique_identity_key).strip() if unique_identity_key is not None else ""
    if token and token.lower() != "nan":
        try:
            from hydra_suite.core.post.identity_postprocess import parse_identity_key

            parsed = parse_identity_key(token)
        except Exception:
            parsed = {}
        if parsed:
            compact_parts = []
            cnn_parts_by_label: dict[str, list[str]] = {}
            for source in sorted(parsed):
                value = str(parsed[source]).strip()
                if not value:
                    continue
                if source == "apriltag":
                    compact_parts.append(f"Tag {value}")
                    continue
                if source.startswith("cnn:"):
                    parts = source.split(":")
                    label = parts[1] if len(parts) >= 2 else source
                    compact_value = value
                    if len(parts) >= 3:
                        compact_value = value
                    elif "+" in value:
                        pieces = []
                        for item in value.split("+"):
                            item = str(item).strip()
                            if not item:
                                continue
                            if ":" in item:
                                item = str(item.split(":", 1)[1]).strip()
                            if item:
                                pieces.append(item)
                        if pieces:
                            compact_value = " / ".join(pieces)
                    if compact_value:
                        cnn_parts_by_label.setdefault(label, []).append(compact_value)
                    continue
                compact_parts.append(f"{source}={value}")
            for label in sorted(cnn_parts_by_label):
                values = [value for value in cnn_parts_by_label[label] if value]
                if not values:
                    continue
                compact_parts.append(
                    values[0] if len(values) == 1 else " / ".join(values)
                )
            if compact_parts:
                return " | ".join(compact_parts)
        return token
    return f"ID{track_id}"


def _is_explicit_unknown_label(value) -> bool:
    """True when *value* is the literal, resolved ``"unknown"`` label token.

    Distinguishes "this identity tier resolved to unknown" from "this
    identity tier has no value at all" (missing/blank/NaN) -- the two must
    be handled differently by the resolved-identity priority chain below
    (I6): a genuinely resolved ``"unknown"`` must stop the fallthrough,
    while a blank/missing value must still let the chain continue to the
    next tier.
    """
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() == "unknown"


def normalize_video_identity_color_key(value):
    """Return a stable identity color key token or an empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    token = str(value).strip()
    if not token or token.lower() == "nan":
        return ""
    if token.lower() == "unknown":
        return ""
    try:
        from hydra_suite.core.post.identity_postprocess import parse_identity_key

        parsed = parse_identity_key(token)
    except Exception:
        parsed = {}
    if parsed:
        informative_values = [
            str(v).strip()
            for v in parsed.values()
            if str(v).strip() and str(v).strip().lower() != "unknown"
        ]
        if not informative_values:
            return ""
    return token


def build_video_track_label_array(trajectories_df):
    """Precompute one overlay label per row using stable identity when available."""
    if trajectories_df is None or len(trajectories_df) == 0:
        return np.asarray([], dtype=object)
    # Prioritize resolved final identity over raw per-frame classifier evidence (audit S8).
    identity_columns = [
        C.FINAL_LABEL,
        C.FINAL_SMOOTHED_LABEL,
        C.UNIQUE_IDENTITY_KEY,
    ]
    track_ids = trajectories_df["TrajectoryID"].tolist()
    labels = []
    for row_index, track_id in enumerate(track_ids):
        chosen_token = None
        for column in identity_columns:
            if column not in trajectories_df.columns:
                continue
            raw_value = trajectories_df.iloc[row_index][column]
            if column in (
                C.FINAL_LABEL,
                C.FINAL_SMOOTHED_LABEL,
            ) and _is_explicit_unknown_label(raw_value):
                # I6: "unknown" IS the resolved answer for this track (not a
                # missing value) -- stop here rather than falling through to
                # UniqueIdentityKey's raw per-frame classifier evidence.
                break
            token = normalize_video_identity_color_key(raw_value)
            if token:
                chosen_token = token
                break
        labels.append(format_video_track_label(track_id, chosen_token))
    return np.asarray(labels, dtype=object)


def build_video_track_color_key_array(trajectories_df):
    """Precompute one color key per row, preferring identity evidence over TrajectoryID."""
    if trajectories_df is None or len(trajectories_df) == 0:
        return np.asarray([], dtype=object)
    # Prioritize resolved final identity over raw per-frame classifier evidence (audit S8).
    identity_columns = [
        C.FINAL_LABEL,
        C.FINAL_SMOOTHED_LABEL,
        C.UNIQUE_IDENTITY_KEY,
    ]
    track_ids = trajectories_df["TrajectoryID"].tolist()
    color_keys = []
    for row_index, track_id in enumerate(track_ids):
        chosen_key = ""
        for column in identity_columns:
            if column not in trajectories_df.columns:
                continue
            raw_value = trajectories_df.iloc[row_index][column]
            if column in (
                C.FINAL_LABEL,
                C.FINAL_SMOOTHED_LABEL,
            ) and _is_explicit_unknown_label(raw_value):
                # I6: "unknown" IS the resolved answer for this track (not a
                # missing value) -- stop here rather than falling through to
                # UniqueIdentityKey's raw per-frame classifier evidence.
                break
            token = normalize_video_identity_color_key(raw_value)
            if token:
                chosen_key = f"identity:{token}"
                break
        if not chosen_key:
            chosen_key = f"trajectory:{int(track_id)}"
        color_keys.append(chosen_key)
    return np.asarray(color_keys, dtype=object)


def build_precomputed_color_palette(colors, _track_ids, color_keys):
    """Build per-row colors, reusing one color for rows with the same identity key."""
    _category20_colors = [
        (127, 127, 31),
        (188, 189, 34),
        (140, 86, 75),
        (255, 127, 14),
        (214, 39, 40),
        (255, 152, 150),
        (197, 176, 213),
        (148, 103, 189),
        (196, 156, 148),
        (227, 119, 194),
        (199, 199, 199),
        (140, 140, 140),
        (23, 190, 207),
        (158, 218, 229),
        (57, 59, 121),
        (82, 84, 163),
        (107, 110, 207),
        (156, 158, 222),
        (99, 121, 57),
        (140, 162, 82),
    ]
    _n_cat = len(_category20_colors)

    def _fallback_color(_track_id):
        _tid = int(_track_id)
        return (
            tuple(colors[_tid])
            if colors and _tid < len(colors)
            else _category20_colors[_tid % _n_cat]
        )

    _identity_palette = {}
    _next_identity_color_idx = 0
    _row_colors = []
    for _tid, _key in zip(_track_ids.tolist(), color_keys.tolist()):
        _key_token = str(_key)
        if _key_token.startswith("identity:"):
            if _key_token not in _identity_palette:
                _identity_palette[_key_token] = (
                    tuple(colors[_next_identity_color_idx])
                    if colors and _next_identity_color_idx < len(colors)
                    else _category20_colors[_next_identity_color_idx % _n_cat]
                )
                _next_identity_color_idx += 1
            _row_colors.append(_identity_palette[_key_token])
            continue
        _row_colors.append(_fallback_color(_tid))
    return _row_colors


def build_video_draw_params(params, config, fps, trajectories_df):
    """Return drawing parameters derived from params, config dict, and body size."""
    colors = params.get("TRAJECTORY_COLORS", [])
    reference_body_size = params.get("REFERENCE_BODY_SIZE", 30.0)
    show_labels = bool(config.get("video_show_labels", True))
    show_orientation = bool(config.get("video_show_orientation", True))
    show_trails = bool(config.get("video_show_trails", False))
    trail_duration_sec = float(config.get("video_trail_duration", 1.0))
    trail_duration_frames = int(trail_duration_sec * fps)
    marker_size = float(config.get("video_marker_size", 0.3))
    text_scale = float(config.get("video_text_scale", 0.5))
    arrow_length = float(config.get("video_arrow_length", 0.7))
    advanced_config = params.get("ADVANCED_CONFIG", {})
    marker_radius = int(marker_size * reference_body_size)
    arrow_len = int(arrow_length * reference_body_size)
    text_size = 0.5 * text_scale
    marker_thickness = max(2, int(0.15 * reference_body_size))
    pose_point_radius = int(
        max(
            1,
            advanced_config.get("video_pose_point_radius", max(2, marker_radius // 3)),
        )
    )
    pose_point_thickness = int(advanced_config.get("video_pose_point_thickness", -1))
    pose_line_thickness = int(
        max(1, advanced_config.get("video_pose_line_thickness", 2))
    )
    pose_color_mode = (
        str(advanced_config.get("video_pose_color_mode", "track")).strip().lower()
    )
    pose_fixed_color_raw = advanced_config.get("video_pose_color", [255, 255, 255])
    if (
        isinstance(pose_fixed_color_raw, (list, tuple))
        and len(pose_fixed_color_raw) == 3
    ):
        try:
            pose_fixed_color = tuple(
                int(max(0, min(255, float(v)))) for v in pose_fixed_color_raw
            )
        except Exception:
            pose_fixed_color = (255, 255, 255)
    else:
        pose_fixed_color = (255, 255, 255)
    pose_min_conf = normalize_pose_render_min_conf(
        params.get("POSE_MIN_KPT_CONF_VALID", 0.2)
    )
    return dict(
        colors=colors,
        show_labels=show_labels,
        show_orientation=show_orientation,
        show_trails=show_trails,
        trail_duration_frames=trail_duration_frames,
        marker_radius=marker_radius,
        arrow_len=arrow_len,
        text_size=text_size,
        text_scale=text_scale,
        marker_thickness=marker_thickness,
        pose_point_radius=pose_point_radius,
        pose_point_thickness=pose_point_thickness,
        pose_line_thickness=pose_line_thickness,
        pose_color_mode=pose_color_mode,
        pose_fixed_color=pose_fixed_color,
        pose_min_conf=pose_min_conf,
        advanced_config=advanced_config,
    )


def get_pose_column_info(params, advanced_config, trajectories_df):
    """Return (pose_edges, pose_column_triplets, show_pose) for video rendering."""
    pose_edges = []
    pose_column_triplets = []
    show_pose = bool(advanced_config.get("video_show_pose", True))
    pose_col_pattern = re.compile(r"^PoseKpt_(.+)_(X|Y|Conf)$")
    pose_labels_available = {}
    for col in trajectories_df.columns:
        m = pose_col_pattern.match(str(col))
        if m is None:
            continue
        label = m.group(1)
        axis = m.group(2)
        pose_labels_available.setdefault(label, set()).add(axis)
    if not pose_labels_available:
        show_pose = False
    if show_pose:
        skeleton_names = []
        skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
        if skeleton_file and os.path.exists(skeleton_file):
            try:
                with open(skeleton_file, "r", encoding="utf-8") as f:
                    skeleton_data = json.load(f)
                names_raw = skeleton_data.get(
                    "keypoint_names", skeleton_data.get("keypoints", [])
                )
                skeleton_names = [str(n) for n in names_raw]
                raw_edges = skeleton_data.get(
                    "skeleton_edges", skeleton_data.get("edges", [])
                )
                for edge in raw_edges:
                    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                        try:
                            pose_edges.append((int(edge[0]), int(edge[1])))
                        except Exception:
                            continue
            except Exception:
                pose_edges = []
        ordered_labels = build_pose_keypoint_labels(skeleton_names, len(skeleton_names))
        extras = sorted(
            [lbl for lbl in pose_labels_available.keys() if lbl not in ordered_labels]
        )
        ordered_labels.extend(extras)
        for label in ordered_labels:
            axes = pose_labels_available.get(label, set())
            if {"X", "Y", "Conf"}.issubset(axes):
                pose_column_triplets.append(
                    (
                        f"PoseKpt_{label}_X",
                        f"PoseKpt_{label}_Y",
                        f"PoseKpt_{label}_Conf",
                    )
                )
        if not pose_column_triplets:
            show_pose = False
    return pose_edges, pose_column_triplets, show_pose


def preextract_traj_arrays(
    trajectories_df, show_pose, pose_column_triplets, show_trails
):
    """Pre-extract trajectory arrays and index structures for O(1)/O(log N) lookups."""
    _frame_ids = trajectories_df["FrameID"].to_numpy(dtype=np.int32)
    _track_ids = trajectories_df["TrajectoryID"].to_numpy(dtype=np.int32)
    _xs = trajectories_df["X"].to_numpy(dtype=np.float64)
    _ys = trajectories_df["Y"].to_numpy(dtype=np.float64)
    _label_texts = build_video_track_label_array(trajectories_df)
    _thetas = (
        trajectories_df["Theta"].to_numpy(dtype=np.float64)
        if "Theta" in trajectories_df.columns
        else np.full(len(trajectories_df), np.nan)
    )
    _pose_kpts = None
    if show_pose and pose_column_triplets:
        _K = len(pose_column_triplets)
        _N = len(trajectories_df)
        _pose_kpts = np.full((_K, _N, 3), np.nan, dtype=np.float32)
        for _k, (_x_col, _y_col, _c_col) in enumerate(pose_column_triplets):
            if _x_col in trajectories_df.columns:
                _pose_kpts[_k, :, 0] = trajectories_df[_x_col].to_numpy(
                    dtype=np.float32
                )
            if _y_col in trajectories_df.columns:
                _pose_kpts[_k, :, 1] = trajectories_df[_y_col].to_numpy(
                    dtype=np.float32
                )
            if _c_col in trajectories_df.columns:
                _pose_kpts[_k, :, 2] = trajectories_df[_c_col].to_numpy(
                    dtype=np.float32
                )
    traj_indices_by_frame: dict = {}
    for _i in range(len(_frame_ids)):
        _fid = int(_frame_ids[_i])
        if _fid not in traj_indices_by_frame:
            traj_indices_by_frame[_fid] = []
        traj_indices_by_frame[_fid].append(_i)
    _track_sorted_row_indices: dict = {}
    _track_sorted_frame_vals: dict = {}
    if show_trails:
        _tmp_track: dict = {}
        for _i in range(len(_track_ids)):
            _tid = int(_track_ids[_i])
            if _tid not in _tmp_track:
                _tmp_track[_tid] = []
            _tmp_track[_tid].append(_i)
        for _tid, _idxs in _tmp_track.items():
            _idx_arr = np.asarray(_idxs, dtype=np.int32)
            _order = np.argsort(_frame_ids[_idx_arr])
            _track_sorted_row_indices[_tid] = _idx_arr[_order]
            _track_sorted_frame_vals[_tid] = _frame_ids[_idx_arr[_order]]
    return (
        _frame_ids,
        _track_ids,
        _xs,
        _ys,
        _label_texts,
        _thetas,
        _pose_kpts,
        traj_indices_by_frame,
        _track_sorted_row_indices,
        _track_sorted_frame_vals,
    )


def draw_trail_for_track(
    frame,
    track_id,
    frame_idx,
    color,
    _xs,
    _ys,
    _track_sorted_frame_vals,
    _track_sorted_row_indices,
    trail_duration_frames,
    marker_thickness,
):
    """Draw the fading trail for a single track on the given frame."""
    if track_id not in _track_sorted_frame_vals:
        return
    _sfv = _track_sorted_frame_vals[track_id]
    _sri = _track_sorted_row_indices[track_id]
    _lo = int(np.searchsorted(_sfv, frame_idx - trail_duration_frames, side="left"))
    _hi = int(np.searchsorted(_sfv, frame_idx, side="left"))
    if _hi - _lo < 2:
        return
    _trail_xs = _xs[_sri[_lo:_hi]]
    _trail_ys = _ys[_sri[_lo:_hi]]
    _trail_fs = _sfv[_lo:_hi]
    _trail_lw = max(1, marker_thickness // 2)
    for _seg in range(_hi - _lo - 1):
        _px1, _py1 = _trail_xs[_seg], _trail_ys[_seg]
        _px2, _py2 = _trail_xs[_seg + 1], _trail_ys[_seg + 1]
        if np.isnan(_px1) or np.isnan(_py1) or np.isnan(_px2) or np.isnan(_py2):
            continue
        _age = frame_idx - int(_trail_fs[_seg])
        _alpha = 1.0 - (_age / trail_duration_frames)
        cv2.line(
            frame,
            (int(_px1), int(_py1)),
            (int(_px2), int(_py2)),
            (int(color[0] * _alpha), int(color[1] * _alpha), int(color[2] * _alpha)),
            _trail_lw,
        )


def draw_single_track_on_frame(
    frame,
    row_i,
    track_id,
    cx,
    cy,
    color,
    draw_p,
    _thetas,
    _pose_kpts,
    _label_texts,
    pose_edges,
):
    """Draw circle, label, orientation arrow, and pose for a single track."""
    marker_radius = draw_p["marker_radius"]
    marker_thickness = draw_p["marker_thickness"]
    cv2.circle(frame, (cx, cy), marker_radius, color, marker_thickness)
    if draw_p["show_labels"]:
        label_offset = int(marker_radius + 5)
        cv2.putText(
            frame,
            str(_label_texts[row_i]),
            (cx + label_offset, cy - label_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            draw_p["text_size"],
            color,
            max(1, int(draw_p["text_scale"] * 2)),
        )
    if draw_p["show_orientation"]:
        _theta = _thetas[row_i]
        if not np.isnan(_theta):
            cv2.arrowedLine(
                frame,
                (cx, cy),
                (
                    int(cx + draw_p["arrow_len"] * np.cos(_theta)),
                    int(cy + draw_p["arrow_len"] * np.sin(_theta)),
                ),
                color,
                marker_thickness,
                tipLength=0.3,
            )
    if _pose_kpts is not None:
        kpts_arr = _pose_kpts[:, row_i, :]
        if np.any(np.isfinite(kpts_arr[:, 2])):
            pose_color = (
                color
                if draw_p["pose_color_mode"] == "track"
                else draw_p["pose_fixed_color"]
            )
            if pose_edges:
                for e0, e1 in pose_edges:
                    if e0 < 0 or e1 < 0 or e0 >= len(kpts_arr) or e1 >= len(kpts_arr):
                        continue
                    if not is_renderable_pose_keypoint(
                        kpts_arr[e0, 0],
                        kpts_arr[e0, 1],
                        kpts_arr[e0, 2],
                        draw_p["pose_min_conf"],
                    ) or not is_renderable_pose_keypoint(
                        kpts_arr[e1, 0],
                        kpts_arr[e1, 1],
                        kpts_arr[e1, 2],
                        draw_p["pose_min_conf"],
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
                        pose_color,
                        draw_p["pose_line_thickness"],
                    )
            for kpt in kpts_arr:
                if not is_renderable_pose_keypoint(
                    kpt[0], kpt[1], kpt[2], draw_p["pose_min_conf"]
                ):
                    continue
                cv2.circle(
                    frame,
                    (int(round(float(kpt[0]))), int(round(float(kpt[1])))),
                    draw_p["pose_point_radius"],
                    pose_color,
                    draw_p["pose_point_thickness"],
                )


def render_annotated_video_frames(
    cap,
    out,
    start_frame,
    total_frames,
    draw_p,
    pose_edges,
    show_pose,
    arrays,
    progress=None,
    should_stop=None,
):
    """Write annotated frames from cap into out. Return True if completed, False if cancelled."""
    import queue as _queue
    import threading as _threading

    (
        _frame_ids,
        _track_ids,
        _xs,
        _ys,
        _label_texts,
        _thetas,
        _pose_kpts,
        traj_indices_by_frame,
        _track_sorted_row_indices,
        _track_sorted_frame_vals,
        _row_colors,
    ) = arrays
    _write_q: _queue.Queue = _queue.Queue(maxsize=4)

    def _writer_thread():
        while True:
            _item = _write_q.get()
            if _item is None:
                break
            out.write(_item)

    _writer = _threading.Thread(target=bind_target(_writer_thread), daemon=True)
    _writer.start()
    cancelled = False

    for rel_idx in range(total_frames):
        if should_stop is not None and should_stop():
            cancelled = True
            break
        frame_idx = start_frame + rel_idx
        ret, frame = cap.read()
        if not ret:
            break

        frame_row_indices = traj_indices_by_frame.get(frame_idx, [])

        if draw_p["show_trails"]:
            for row_i in frame_row_indices:
                track_id = int(_track_ids[row_i])
                color = tuple(_row_colors[row_i])
                draw_trail_for_track(
                    frame,
                    track_id,
                    frame_idx,
                    color,
                    _xs,
                    _ys,
                    _track_sorted_frame_vals,
                    _track_sorted_row_indices,
                    draw_p["trail_duration_frames"],
                    draw_p["marker_thickness"],
                )

        for row_i in frame_row_indices:
            track_id = int(_track_ids[row_i])
            cx_f, cy_f = _xs[row_i], _ys[row_i]
            if np.isnan(cx_f) or np.isnan(cy_f):
                continue
            cx, cy = int(cx_f), int(cy_f)
            color = tuple(_row_colors[row_i])
            draw_single_track_on_frame(
                frame,
                row_i,
                track_id,
                cx,
                cy,
                color,
                draw_p,
                _thetas,
                _pose_kpts if show_pose else None,
                _label_texts,
                pose_edges,
            )

        _write_q.put(frame)

        if progress is not None and rel_idx % 30 == 0:
            pct = int(((rel_idx + 1) / total_frames) * 100)
            progress(pct, "Generating video...")

    _write_q.put(None)
    _writer.join()
    return not cancelled


def open_video_cap_and_writer(video_path, output_path):
    """Open video capture and writer; return (cap, out, fps, total_video_frames) or None on error."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        out = VideoEncoder(output_path, fps=fps, width=frame_width, height=frame_height)
    except Exception:
        logger.error(f"Failed to create output video: {output_path}")
        cap.release()
        return None
    logger.info(f"Writing video: {frame_width}x{frame_height} @ {fps} FPS")
    return cap, out, fps, total_video_frames


def compute_video_frame_range(params, total_video_frames):
    """Return (start_frame, end_frame, total_frames) clamped to video bounds."""
    start_frame = int(params.get("START_FRAME", 0) or 0)
    end_frame = params.get("END_FRAME", None)
    if end_frame is None:
        end_frame = total_video_frames - 1 if total_video_frames > 0 else 0
    end_frame = int(end_frame)
    if total_video_frames > 0:
        start_frame = max(0, min(start_frame, total_video_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_video_frames - 1))
    total_frames = max(0, end_frame - start_frame + 1)
    logger.info(
        f"Exporting tracked frame range: {start_frame}-{end_frame} ({total_frames} frames)"
    )
    return start_frame, end_frame, total_frames


def load_video_trajectories(final_csv_path):
    """Load best available trajectories for video generation (prefers rich export CSV)."""
    from hydra_suite.core.post.rich_export import rich_export_path

    if not final_csv_path:
        return None, None
    candidates = [
        rich_export_path(final_csv_path),
        rich_export_path(final_csv_path, legacy=True),
        final_csv_path,
    ]
    candidate = next((path for path in candidates if os.path.exists(path)), None)
    if not candidate:
        return None, None
    try:
        return pd.read_csv(candidate), candidate
    except Exception:
        logger.exception("Failed to load video trajectories from: %s", candidate)
        return None, None


def render_annotated_video(
    *,
    trajectories_df,
    video_path,
    output_path,
    params,
    config,
    progress=None,
    should_stop=None,
):
    """Generate an annotated overlay video from post-processed trajectories.

    Returns the output path on success, or None on failure/cancellation (partial
    output file is deleted on cancellation so no half-written video survives)."""
    logger.info("=" * 80)
    logger.info("Generating video from post-processed trajectories...")
    logger.info("=" * 80)

    if trajectories_df is None or trajectories_df.empty:
        return None
    if not video_path or not output_path:
        logger.error("Video input or output path not specified")
        return None

    opened = open_video_cap_and_writer(video_path, output_path)
    if opened is None:
        return None
    cap, out, fps, total_video_frames = opened

    start_frame, end_frame, total_frames = compute_video_frame_range(
        params, total_video_frames
    )
    if total_frames <= 0:
        logger.error("Invalid frame range for video generation.")
        cap.release()
        out.release()
        return None

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    draw_p = build_video_draw_params(params, config, fps, trajectories_df)
    pose_edges, pose_column_triplets, show_pose = get_pose_column_info(
        params, draw_p["advanced_config"], trajectories_df
    )
    (
        _frame_ids,
        _track_ids,
        _xs,
        _ys,
        _label_texts,
        _thetas,
        _pose_kpts,
        traj_indices_by_frame,
        _track_sorted_row_indices,
        _track_sorted_frame_vals,
    ) = preextract_traj_arrays(
        trajectories_df, show_pose, pose_column_triplets, draw_p["show_trails"]
    )
    _color_keys = build_video_track_color_key_array(trajectories_df)
    _row_colors = build_precomputed_color_palette(
        draw_p["colors"], _track_ids, _color_keys
    )

    arrays = (
        _frame_ids,
        _track_ids,
        _xs,
        _ys,
        _label_texts,
        _thetas,
        _pose_kpts,
        traj_indices_by_frame,
        _track_sorted_row_indices,
        _track_sorted_frame_vals,
        _row_colors,
    )
    completed = render_annotated_video_frames(
        cap,
        out,
        start_frame,
        total_frames,
        draw_p,
        pose_edges,
        show_pose,
        arrays,
        progress=progress,
        should_stop=should_stop,
    )

    cap.release()
    out.release()

    if not completed:
        logger.info("Annotated video generation cancelled; removing partial output.")
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            logger.warning("Could not delete partial video: %s", output_path)
        return None

    logger.info(f"✓ Video saved to: {output_path}")
    logger.info("=" * 80)
    return output_path


def export_final_media(
    *,
    final_csv_path,
    config,
    video_path,
    detection_cache_path,
    interpolated_roi_npz_path,
    fps,
    image_root,
    video_root,
    export_images,
    export_videos,
    background_color,
    geometry=None,
    progress=None,
    should_stop=None,
):
    """Export final canonical stills and/or orientation-fixed per-track videos.

    ``geometry`` is the session's project-wide Layer 1
    :class:`~hydra_suite.core.canonicalization.geometry.CanonicalGeometry`.
    It MUST be threaded through: without it the exporter falls back to its own
    DEFAULT geometry, which silently diverges from the canvas every other
    crop-consuming stage in the same session used.

    Returns the exporter result dict, or None if nothing is requested or the
    final CSV / detection cache is missing."""
    if not export_images and not export_videos:
        return None
    if not final_csv_path or not os.path.exists(final_csv_path):
        return None
    if not detection_cache_path or not os.path.exists(detection_cache_path):
        logger.warning(
            "Skipping final canonical media export: no compatible detection cache is available."
        )
        return None
    if export_images and image_root is None:
        logger.warning(
            "Skipping final canonical image export: no image output directory found."
        )
        export_images = False
    if export_videos and video_root is None:
        logger.warning(
            "Skipping final media video export: no video output directory found."
        )
        export_videos = False
    if not export_images and not export_videos:
        return None

    from pathlib import Path

    export_root = video_root or image_root
    image_output_dir = (
        str((Path(image_root) / "images").expanduser()) if image_root else None
    )

    suppress_dataset = bool(
        config.get("suppress_foreign_obb_individual_dataset", False)
    )
    suppress_videos = bool(config.get("suppress_foreign_obb_oriented_videos", False))
    suppress_foreign_obb = suppress_videos if export_videos else suppress_dataset

    exporter = OrientedTrackVideoExporter(
        str(export_root),
        final_csv_path,
        video_path=video_path,
        detection_cache_path=detection_cache_path,
        interpolated_roi_npz_path=interpolated_roi_npz_path,
        fps=fps,
        background_color=tuple(int(c) for c in background_color),
        suppress_foreign_obb=suppress_foreign_obb,
        suppress_foreign_obb_images=suppress_dataset,
        suppress_foreign_obb_videos=suppress_videos,
        export_images=export_images,
        image_output_dir=image_output_dir,
        image_interval=int(config.get("individual_save_interval", 1)),
        image_format=str(config.get("individual_output_format", "png")),
        export_videos=export_videos,
        fix_direction_flips=bool(
            config.get("final_media_export_fix_direction_flips", False)
        ),
        heading_flip_max_burst=int(
            config.get("final_media_export_heading_flip_burst", 5)
        ),
        enable_affine_stabilization=bool(
            config.get("final_media_export_enable_affine_stabilization", False)
        ),
        stabilization_window=int(
            config.get("final_media_export_stabilization_window", 5)
        ),
        output_subdir="",
        geometry=geometry,
    )
    result = exporter.export(progress_callback=progress, should_stop=should_stop)
    return result.to_dict()
