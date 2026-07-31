from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.trackerkit.gui.orchestrators.tracking import (
    PREVIEW_MAX_DURATION_SECONDS,
    compute_capped_preview_range,
)


def test_range_within_limit_is_unchanged():
    # 100 frames at 30 fps (~3.3s) is well under the 300s cap.
    clamped_end, was_clamped = compute_capped_preview_range(0, 99, fps=30.0)
    assert clamped_end == 99
    assert was_clamped is False


def test_range_exceeding_limit_is_clamped_to_max_duration():
    # 30 fps * 300s = 9000 frames. Selecting 0..19999 (20000 frames) must clamp
    # to a 9000-frame window starting at start_frame.
    clamped_end, was_clamped = compute_capped_preview_range(0, 19_999, fps=30.0)
    assert was_clamped is True
    assert clamped_end == 0 + (30 * PREVIEW_MAX_DURATION_SECONDS) - 1
    assert clamped_end == 8999


def test_clamp_is_relative_to_start_frame_not_zero():
    clamped_end, was_clamped = compute_capped_preview_range(5_000, 30_000, fps=30.0)
    assert was_clamped is True
    assert clamped_end == 5_000 + (30 * PREVIEW_MAX_DURATION_SECONDS) - 1


def test_exact_boundary_is_not_clamped():
    # Exactly 9000 frames (300s at 30fps) must NOT be reported as clamped.
    max_frames = int(round(30.0 * PREVIEW_MAX_DURATION_SECONDS))
    clamped_end, was_clamped = compute_capped_preview_range(0, max_frames - 1, fps=30.0)
    assert was_clamped is False
    assert clamped_end == max_frames - 1


def test_custom_max_duration_is_respected():
    clamped_end, was_clamped = compute_capped_preview_range(
        0, 999, fps=10.0, max_duration_seconds=60
    )
    assert was_clamped is True
    assert clamped_end == 599  # 10 fps * 60s - 1
