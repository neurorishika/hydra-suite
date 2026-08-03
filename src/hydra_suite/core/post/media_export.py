"""Qt-free media export: trajectory persistence, coordinate scaling, and the
annotated-video overlay chain. Extracted from trackerkit/gui/orchestrators/tracking.py
(Slice 3 of the headless-session-service program)."""

from __future__ import annotations

import csv
import logging

import numpy as np
import pandas as pd

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
