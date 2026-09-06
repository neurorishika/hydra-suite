"""Regression tests for the live preview observation-only overlay policy."""

from collections import deque

import numpy as np

from hydra_suite.core.tracking import visualization
from hydra_suite.core.tracking.optimization import optimizer_workers as ow


def _record_draw_calls(monkeypatch, module):
    calls = []
    for name in ("polylines", "circle", "line", "arrowedLine", "putText"):
        monkeypatch.setattr(
            module.cv2,
            name,
            lambda *args, _name=name, **kwargs: calls.append(_name),
        )
    return calls


def _overlay_params():
    return {
        "TRAJECTORY_COLORS": [(1, 2, 3)],
        "SHOW_CIRCLES": True,
        "SHOW_ORIENTATION": True,
        "SHOW_TRAJECTORIES": True,
        "SHOW_LABELS": True,
        "SHOW_STATE": True,
    }


def test_normal_preview_only_marks_tracks_observed_on_current_frame(monkeypatch):
    calls = _record_draw_calls(monkeypatch, visualization)
    overlay = np.zeros((32, 32, 3), dtype=np.uint8)
    trajectories = [[(3.0, 4.0, 0.0, 0), (5.0, 6.0, 0.0, 1)]]

    visualization.draw_overlays(
        overlay,
        _overlay_params(),
        trajectories,
        ["occluded"],
        [7],
        [1],
        None,
        None,
        current_frame_matches=set(),
    )

    assert calls == ["polylines"]

    calls.clear()
    visualization.draw_overlays(
        overlay,
        _overlay_params(),
        trajectories,
        ["active"],
        [7],
        [1],
        None,
        None,
        current_frame_matches={0},
    )
    assert calls == ["polylines", "circle", "line", "putText"]

    calls.clear()
    visualization.draw_overlays(
        overlay,
        _overlay_params(),
        trajectories,
        ["lost"],
        [7],
        [1],
        None,
        None,
        current_frame_matches={0},
    )
    assert calls == []


def test_autotuner_preview_marks_only_real_observations_and_preserves_history(
    monkeypatch,
):
    calls = _record_draw_calls(monkeypatch, ow)
    display = np.zeros((32, 32, 3), dtype=np.uint8)
    trail = [deque([(3, 4), (5, 6)])]

    ow._preview_render_tracks(
        display,
        1,
        ["occluded"],
        trail,
        [7],
        [(1, 2, 3)],
        True,
        True,
        True,
        True,
        {},
    )
    assert calls == ["polylines"]

    calls.clear()
    ow._preview_render_tracks(
        display,
        1,
        ["active"],
        trail,
        [7],
        [(1, 2, 3)],
        True,
        True,
        True,
        True,
        {0: (12.0, 13.0, 0.0)},
    )
    assert calls == ["polylines", "circle", "arrowedLine", "putText"]

    calls.clear()
    ow._preview_render_tracks(
        display,
        1,
        ["lost"],
        trail,
        [7],
        [(1, 2, 3)],
        True,
        True,
        True,
        True,
        {0: (12.0, 13.0, 0.0)},
    )
    assert calls == []


def test_autotuner_trails_never_append_unobserved_kalman_positions():
    trail = [deque([(3, 4)]), deque([(8, 9)])]

    ow._preview_update_trails(trail, ["occluded", "lost"], {})
    assert list(trail[0]) == [(3, 4)]
    assert list(trail[1]) == []

    ow._preview_update_trails(
        trail,
        ["active", "occluded"],
        {0: (12.7, 13.9, 0.0)},
    )
    assert list(trail[0]) == [(3, 4), (12, 13)]
    assert list(trail[1]) == []
