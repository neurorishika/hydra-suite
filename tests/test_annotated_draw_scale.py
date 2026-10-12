"""Drawing the annotated video at output size (``video_output_scale``).

The golden ``tests/goldens/annotated_draw/scale_1_0.npz`` was produced by the
PRE-scale ``media_export`` drawing code (commit d4f7ae6c) on the synthetic case
below: trails, labels, orientation arrows, pose skeleton + points, a NaN theta,
a NaN position and a low-confidence keypoint. Scale 1.0 must reproduce it
pixel for pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.post import media_export as mx

GOLDEN = Path(__file__).parent / "goldens" / "annotated_draw" / "scale_1_0.npz"


def synthetic_case(skeleton_path):
    n_frames, h, w = 10, 96, 128
    Path(skeleton_path).write_text(
        json.dumps({"keypoint_names": ["head", "tail"], "skeleton_edges": [[0, 1]]})
    )
    rows = []
    for f in range(n_frames):
        for tid, (x0, y0) in enumerate([(20.3, 30.7), (70.9, 50.2), (100.5, 20.5)]):
            x = x0 + 1.7 * f
            y = y0 + 0.9 * f * (1 if tid % 2 else -1)
            if tid == 2 and f == 4:
                x = np.nan  # a gap
            theta = np.nan if (tid == 1 and f == 3) else 0.3 * f + tid
            rows.append(
                {
                    "TrajectoryID": tid,
                    "FrameID": f,
                    "X": x,
                    "Y": y,
                    "Theta": theta,
                    "PoseKpt_head_X": x + 4.4,
                    "PoseKpt_head_Y": y - 3.6,
                    "PoseKpt_head_Conf": 0.9,
                    "PoseKpt_tail_X": x - 5.5,
                    "PoseKpt_tail_Y": y + 2.5,
                    "PoseKpt_tail_Conf": 0.05 if f == 6 else 0.8,
                }
            )
    df = pd.DataFrame(rows)
    params = {
        "TRAJECTORY_COLORS": [(0, 255, 0), (255, 0, 0), (0, 0, 255)],
        "REFERENCE_BODY_SIZE": 20.0,
        "ADVANCED_CONFIG": {"video_show_pose": True},
        "POSE_MIN_KPT_CONF_VALID": 0.2,
        "POSE_SKELETON_FILE": str(skeleton_path),
        "START_FRAME": 0,
        "END_FRAME": None,
    }
    config = {
        "video_show_labels": True,
        "video_show_orientation": True,
        "video_show_trails": True,
        "video_trail_duration": 0.5,
        "video_marker_size": 0.3,
        "video_text_scale": 0.6,
        "video_arrow_length": 0.7,
    }
    return df, params, config, n_frames, h, w


def _background(h, w):
    frame = np.full((h, w, 3), 40, np.uint8)
    frame[:, :, 1] = (np.arange(w) % 255).astype(np.uint8)[None, :]
    return frame


def _render(tmp_path, scale):
    df, params, config, n_frames, h, w = synthetic_case(tmp_path / "skel.json")
    out_w, out_h = int(round(w * scale)), int(round(h * scale))
    state = mx.prepare_annotated_draw_state(
        df, params, config, 10.0, scale=scale, scale_xy=(out_w / w, out_h / h)
    )
    frames = []
    for idx in range(n_frames):
        frame = _background(out_h, out_w)
        mx.draw_annotations_on_frame(frame, idx, state)
        frames.append(frame)
    return np.stack(frames), state


def test_scale_one_is_pixel_identical_to_pre_scale_drawing(tmp_path):
    golden = np.load(GOLDEN)["frames"]
    frames, _ = _render(tmp_path, 1.0)
    assert frames.shape == golden.shape
    assert np.array_equal(frames, golden)


def test_draw_params_scale_sizes_and_keep_thickness_at_least_one():
    params = {
        "TRAJECTORY_COLORS": [],
        "REFERENCE_BODY_SIZE": 40.0,
        "ADVANCED_CONFIG": {},
        "POSE_MIN_KPT_CONF_VALID": 0.2,
    }
    config = {
        "video_marker_size": 0.3,
        "video_text_scale": 0.5,
        "video_arrow_length": 0.7,
    }
    df = pd.DataFrame({"TrajectoryID": [0], "X": [1.0], "Y": [2.0]})
    full = mx.build_video_draw_params(params, config, 30.0, df)
    half = mx.build_video_draw_params(params, config, 30.0, df, scale=0.5)
    tiny = mx.build_video_draw_params(params, config, 30.0, df, scale=0.1)
    assert full["marker_radius"] == 12 and half["marker_radius"] == 6
    assert full["arrow_len"] == 28 and half["arrow_len"] == 14
    assert half["text_size"] == pytest.approx(full["text_size"] * 0.5)
    assert full["marker_thickness"] == 6 and half["marker_thickness"] == 3
    assert half["pose_point_radius"] == 2  # round(4 * 0.5)
    assert half["pose_line_thickness"] == 1
    assert half["label_pad"] == pytest.approx(2.5)
    # Thickness never drops below 1; -1 (filled) is a sentinel, not a size.
    for key in ("marker_thickness", "pose_line_thickness", "text_thickness"):
        assert tiny[key] >= 1
    assert tiny["pose_point_thickness"] == -1
    assert tiny["pose_point_radius"] >= 1


def test_coordinates_are_scaled_per_axis(tmp_path):
    _, state = _render(tmp_path, 0.5)
    _, full = _render(tmp_path, 1.0)
    xs_full, ys_full = full.arrays[2], full.arrays[3]
    xs, ys = state.arrays[2], state.arrays[3]
    np.testing.assert_allclose(xs, xs_full * 0.5)
    np.testing.assert_allclose(ys, ys_full * 0.5)
    kp, kp_full = state.arrays[6], full.arrays[6]
    np.testing.assert_allclose(kp[:, :, 0], kp_full[:, :, 0] * 0.5, rtol=1e-6)
    np.testing.assert_allclose(kp[:, :, 1], kp_full[:, :, 1] * 0.5, rtol=1e-6)
    np.testing.assert_array_equal(kp[:, :, 2], kp_full[:, :, 2])  # conf untouched


def test_half_scale_draws_marker_at_half_position(tmp_path):
    frames, _ = _render(tmp_path, 0.5)
    base = _background(48, 64)
    changed = np.any(frames[0] != base, axis=2)
    ys, xs = np.nonzero(changed)
    # Track 0 at frame 0 sits at (20.3, 30.7) in source px -> ~(10, 15).
    assert changed[:, :].any()
    near = (np.abs(xs - 10) <= 6) & (np.abs(ys - 15) <= 6)
    assert near.any()
    # Nothing is drawn past the output canvas (no source-pixel coordinates).
    assert frames.shape[1:3] == (48, 64)
