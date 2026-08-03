"""Qt-free media export: trajectory persistence, coordinate scaling, and the
annotated-video overlay chain. Extracted from trackerkit/gui/orchestrators/tracking.py
(Slice 3 of the headless-session-service program)."""

from __future__ import annotations

import csv
import json
import logging
import os
import re

import numpy as np
import pandas as pd

from hydra_suite.core.identity.properties.export import build_pose_keypoint_labels
from hydra_suite.utils.pose_visualization import normalize_pose_render_min_conf

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


def save_trajectories_to_csv(trajectories, output_path):
    """Save processed trajectories to CSV. Accepts a DataFrame or list-of-tuples."""
    if trajectories is None:
        logger.warning("No post-processed trajectories to save (None).")
        return False
    if isinstance(trajectories, pd.DataFrame):
        if trajectories.empty:
            logger.warning("No post-processed trajectories to save (empty DataFrame).")
            return False
        try:
            df_to_save = trajectories.copy()
            for col in ["X", "Y", "FrameID"]:
                if col in df_to_save.columns:
                    df_to_save[col] = pd.to_numeric(df_to_save[col], errors="coerce")
                    df_to_save[col] = df_to_save[col].round().astype("Int64")
            unwanted_cols = ["TrackID", "Index"]
            df_to_save = df_to_save.drop(
                columns=[col for col in unwanted_cols if col in df_to_save.columns],
                errors="ignore",
            )
            base_cols = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
            other_cols = [col for col in df_to_save.columns if col not in base_cols]
            ordered_cols = base_cols + other_cols
            df_to_save[ordered_cols].to_csv(output_path, index=False)
            logger.info(
                f"Successfully saved {df_to_save['TrajectoryID'].nunique()} post-processed trajectories "
                f"({len(df_to_save)} rows) with {len(ordered_cols)} columns to {output_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save processed trajectories to {output_path}: {e}")
            return False

    if not trajectories:
        logger.warning("No post-processed trajectories to save.")
        return False
    header = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for trajectory_id, segment in enumerate(trajectories):
                for x, y, theta, frame_id in segment:
                    x_val = int(x) if not np.isnan(x) else ""
                    y_val = int(y) if not np.isnan(y) else ""
                    frame_val = int(frame_id) if not np.isnan(frame_id) else ""
                    writer.writerow([trajectory_id, x_val, y_val, theta, frame_val])
        logger.info(f"Saved trajectories to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save trajectories to {output_path}: {e}")
        return False


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
    identity_columns = [
        "UniqueIdentityKey",
        "IdentityAssignedLabel",
        "IdentityOfflineLabel",
        "IdentitySmoothedLabel",
    ]
    track_ids = trajectories_df["TrajectoryID"].tolist()
    labels = []
    for row_index, track_id in enumerate(track_ids):
        chosen_token = None
        for column in identity_columns:
            if column not in trajectories_df.columns:
                continue
            token = normalize_video_identity_color_key(
                trajectories_df.iloc[row_index][column]
            )
            if token:
                chosen_token = token
                break
        labels.append(format_video_track_label(track_id, chosen_token))
    return np.asarray(labels, dtype=object)


def build_video_track_color_key_array(trajectories_df):
    """Precompute one color key per row, preferring identity evidence over TrajectoryID."""
    if trajectories_df is None or len(trajectories_df) == 0:
        return np.asarray([], dtype=object)
    identity_columns = [
        "UniqueIdentityKey",
        "IdentityAssignedLabel",
        "IdentityOfflineLabel",
        "IdentitySmoothedLabel",
    ]
    track_ids = trajectories_df["TrajectoryID"].tolist()
    color_keys = []
    for row_index, track_id in enumerate(track_ids):
        chosen_key = ""
        for column in identity_columns:
            if column not in trajectories_df.columns:
                continue
            token = normalize_video_identity_color_key(
                trajectories_df.iloc[row_index][column]
            )
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
