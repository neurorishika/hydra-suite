"""Focused merge-wizard overlay regressions for RefineKit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydra_suite.refinekit.gui.dialogs.merge_wizard import (
    MAIN_TRACK_BGR,
    _FrameDetections,
    _make_overlay_fn,
    _make_swap_overlay_fn,
)


def test_merge_wizard_frame_detections_reads_current_cache_tuple_shape() -> None:
    class _FakeCache:
        def get_frame(self, _frame_idx: int):
            return (
                [[10.0, 20.0, 0.25]],
                [],
                [[100.0, 2.0]],
                [],
                [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])],
                [],
                [],
                [],
                [],
                [],
                None,
                None,
            )

    dets = _FrameDetections(_FakeCache(), inv_resize=2.0)
    result = dets.get(12)

    assert result is not None
    meas_arr, _semi_axes, obb_corners = result
    assert tuple(meas_arr[0]) == pytest.approx((20.0, 40.0, 0.25))
    assert np.allclose(
        obb_corners[0],
        np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]]),
    )


def test_merge_overlay_draws_full_target_trajectory_in_pink() -> None:
    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2, 5, 6, 7],
            "TrajectoryID": [1, 1, 1, 2, 2, 2],
            "X": [10.0, 20.0, 30.0, 70.0, 80.0, 90.0],
            "Y": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        }
    )

    overlay = _make_overlay_fn(
        df,
        source_id=1,
        target_id=2,
        crop_box=(0, 0, 100, 100),
        merged_color=MAIN_TRACK_BGR,
        frame_start=0,
        frame_end=7,
        frame_dets=None,
    )

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = overlay(img, frame_idx=1)
    target_region = out[47:54, 77:84]
    color = np.array(MAIN_TRACK_BGR, dtype=np.int16)
    diff = np.abs(target_region.astype(np.int16) - color).sum(axis=2)

    assert np.any(diff <= 15)


def test_merge_overlay_does_not_draw_dashed_bridge_in_gap() -> None:
    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2, 5, 6, 7],
            "TrajectoryID": [1, 1, 1, 2, 2, 2],
            "X": [10.0, 20.0, 30.0, 70.0, 80.0, 90.0],
            "Y": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        }
    )

    overlay = _make_overlay_fn(
        df,
        source_id=1,
        target_id=2,
        crop_box=(0, 0, 100, 100),
        merged_color=MAIN_TRACK_BGR,
        frame_start=0,
        frame_end=7,
        frame_dets=None,
    )

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = overlay(img, frame_idx=4)
    bridge_region = out[47:54, 42:48]
    color = np.array(MAIN_TRACK_BGR, dtype=np.int16)
    diff = np.abs(bridge_region.astype(np.int16) - color).sum(axis=2)

    assert not np.any(diff <= 15)


def test_swap_overlay_shows_only_final_completed_result_in_pink() -> None:
    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2, 0, 1, 2, 5, 6, 7],
            "TrajectoryID": [1, 1, 1, 2, 2, 2, 2, 2, 2],
            "X": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 60.0, 70.0, 80.0],
            "Y": [30.0, 30.0, 30.0, 70.0, 70.0, 70.0, 30.0, 30.0, 30.0],
        }
    )

    overlay = _make_swap_overlay_fn(
        df,
        source_id=1,
        target_id=2,
        swap_frame=5,
        crop_box=(0, 0, 100, 100),
        merged_color=MAIN_TRACK_BGR,
        frame_start=0,
        frame_end=7,
        frame_dets=None,
    )

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out_pre = overlay(img.copy(), frame_idx=1)
    out_post = overlay(img.copy(), frame_idx=6)
    color = np.array(MAIN_TRACK_BGR, dtype=np.int16)

    source_region = out_pre[27:34, 17:24]
    target_pre_region = out_pre[67:74, 17:24]
    target_future_region = out_pre[27:34, 67:74]
    target_post_region = out_post[27:34, 67:74]
    orphan_region = out_post[67:74, 17:24]

    source_diff = np.abs(source_region.astype(np.int16) - color).sum(axis=2)
    target_pre_diff = np.abs(target_pre_region.astype(np.int16) - color).sum(axis=2)
    target_future_diff = np.abs(target_future_region.astype(np.int16) - color).sum(
        axis=2
    )
    target_post_diff = np.abs(target_post_region.astype(np.int16) - color).sum(axis=2)
    orphan_diff = np.abs(orphan_region.astype(np.int16) - color).sum(axis=2)

    assert np.any(source_diff <= 15)
    assert not np.any(target_pre_diff <= 15)
    assert not np.any(target_future_diff <= 15)
    assert np.any(target_post_diff <= 15)
    assert not np.any(orphan_diff <= 15)
