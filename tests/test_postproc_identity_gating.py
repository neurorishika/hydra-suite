"""Identity-influences-trajectory-structure gate.

The post-processing tab exposes a checkbox ("Let identity drive splits and
block stitches") that sets ``IDENTITY_GATES_TRAJECTORY_STRUCTURE``.  When
unchecked, identity labels still flow through the output but no longer
cause forward/backward-disagreement splits or block stitches between
consecutive fragments.
"""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.core.post.processing import (
    _compute_identity_disagree_frames,
    _stitch_broken_trajectory_fragments,
)
from hydra_suite.trackerkit import cli_config
from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


def _row(frame: int, x: float, y: float, label: str = "", committed: int = 0) -> dict:
    return {
        "FrameID": frame,
        "X": x,
        "Y": y,
        "IdentityFinalSource": "offline" if committed else "",
        "IdentityFinalLabel": label,
    }


def _two_overlapping_lookups() -> tuple[dict, dict]:
    """Two trajectories that occupy the same physical track but commit to
    different identity labels for ten consecutive frames.  Mimics the
    forward/backward-pass disagreement that triggers conservative-merge
    splits when identity gating is on."""
    t1 = {f: _row(f, 100.0 + f, 100.0, "mouse_A", 1) for f in range(0, 20)}
    t2 = {f: _row(f, 100.0 + f, 100.0, "mouse_B", 1) for f in range(0, 20)}
    return t1, t2


def test_disagree_frames_empty_when_identity_drives_splits_false() -> None:
    """When the user has unchecked the master toggle, identity-driven splits
    must short-circuit to an empty set even with sustained disagreement."""
    t1, t2 = _two_overlapping_lookups()
    frames = _compute_identity_disagree_frames(
        t1,
        t2,
        agreement_distance=10.0,
        min_run=5,
        identity_drives_splits=False,
    )
    assert frames == frozenset()


def test_disagree_frames_register_when_identity_drives_splits_true() -> None:
    """Sanity check — with the default toggle, sustained disagreement
    registers, so the off-state above is real (not vacuously empty)."""
    t1, t2 = _two_overlapping_lookups()
    frames = _compute_identity_disagree_frames(
        t1,
        t2,
        agreement_distance=10.0,
        min_run=5,
        identity_drives_splits=True,
    )
    assert len(frames) >= 5


def _two_consecutive_fragments_diff_labels() -> list[pd.DataFrame]:
    """Trajectory broken into two consecutive temporal fragments at the same
    spatial location, each committed to a different identity label.  Realistic
    case: a brief occlusion split a track and the decoder committed to
    different identities on either side."""
    a = pd.DataFrame([_row(f, 100.0 + f, 100.0, "mouse_A", 1) for f in range(0, 10)])
    b = pd.DataFrame(
        [_row(f, 100.0 + (f - 11) + 11, 100.0, "mouse_B", 1) for f in range(11, 20)]
    )
    return [a, b]


def test_stitch_blocks_when_identity_gates_stitching_true() -> None:
    """Default behaviour: conflicting committed labels block stitching."""
    fragments = _two_consecutive_fragments_diff_labels()
    out = _stitch_broken_trajectory_fragments(
        fragments,
        agreement_distance=20.0,
        max_gap=5,
        identity_gates_stitching=True,
    )
    assert len(out) == 2, "identity gate should keep fragments separate"


def test_stitch_allows_when_identity_gates_stitching_false() -> None:
    """Zero-weight contract: identity labels must NOT block stitching."""
    fragments = _two_consecutive_fragments_diff_labels()
    out = _stitch_broken_trajectory_fragments(
        fragments,
        agreement_distance=20.0,
        max_gap=5,
        identity_gates_stitching=False,
    )
    assert len(out) == 1, (
        "with identity gating disabled, geometry alone should stitch the "
        f"fragments — got {len(out)} trajectories"
    )


# ---------------------------------------------------------------------------
# Phase 6 Task 7: post-hoc identity is a first-class, realtime-independent
# toggle, and build_engine_params threads the new smoothing knob.
# ---------------------------------------------------------------------------

_MINIMAL_PROBE_KWARGS = dict(fps=30.0, total_frames=100, width=640, height=480)


def _build_params(cfg: dict) -> dict:
    probe = cli_config.TrackerCliVideoProbe(**_MINIMAL_PROBE_KWARGS)
    rt = RuntimeContext(
        fps=probe.fps,
        total_frames=probe.total_frames,
        frame_width=probe.width,
        frame_height=probe.height,
    )
    return build_engine_params(cfg, runtime=rt)


def test_posthoc_enabled_independent_of_realtime_toggle():
    """IDENTITY_POSTHOC_ENABLED must not move when the realtime toggle does."""
    base = {"detection_method": "background_subtraction", "max_targets": 2}

    realtime_on = _build_params({**base, "enable_identity_in_tracking": True})
    realtime_off = _build_params({**base, "enable_identity_in_tracking": False})

    assert realtime_on["IDENTITY_POSTHOC_ENABLED"] is True
    assert realtime_off["IDENTITY_POSTHOC_ENABLED"] is True
    assert realtime_on["ENABLE_IDENTITY_IN_TRACKING"] is True
    assert realtime_off["ENABLE_IDENTITY_IN_TRACKING"] is False


def test_build_engine_params_emits_identity_enable_smoothing():
    base = {"detection_method": "background_subtraction", "max_targets": 2}

    default_params = _build_params(base)
    assert default_params["IDENTITY_ENABLE_SMOOTHING"] is True

    off_params = _build_params({**base, "enable_identity_smoothing": False})
    assert off_params["IDENTITY_ENABLE_SMOOTHING"] is False

    on_params = _build_params({**base, "enable_identity_smoothing": True})
    assert on_params["IDENTITY_ENABLE_SMOOTHING"] is True


def test_tracking_panel_realtime_tooltip_makes_no_posthoc_claims():
    """The realtime master-toggle tooltip (~:538) must describe only the
    realtime/online effect on tracking and say nothing about post-hoc
    (which now runs as an independent stage -- see postprocess_panel.py).
    """
    import inspect

    from hydra_suite.trackerkit.gui.panels import tracking_panel

    source = inspect.getsource(tracking_panel)
    idx = source.index("self.chk_enable_identity_in_tracking.setToolTip(")
    tooltip_src = source[idx : idx + 700]
    lowered = tooltip_src.lower()
    assert "post-hoc" not in lowered
    assert "post hoc" not in lowered
    assert "post process" not in lowered
    assert "post-process" not in lowered
    assert "realtime" in lowered or "tracking" in lowered
