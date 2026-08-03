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
