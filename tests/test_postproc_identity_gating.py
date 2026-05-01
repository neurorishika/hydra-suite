"""Identity-influences-trajectory-structure gate.

The post-processing tab exposes a checkbox ("Let identity drive splits and
block stitches") that sets ``IDENTITY_GATES_TRAJECTORY_STRUCTURE``.  When
unchecked, identity labels still flow through the output but no longer
cause forward/backward-disagreement splits or block stitches between
consecutive fragments.
"""

from __future__ import annotations

import pandas as pd

from hydra_suite.core.post.processing import (
    _compute_identity_disagree_frames,
    _stitch_broken_trajectory_fragments,
)


def _row(frame: int, x: float, y: float, label: str = "", committed: int = 0) -> dict:
    return {
        "FrameID": frame,
        "X": x,
        "Y": y,
        "IdentityCommitted": committed,
        "IdentityAssignedLabel": label,
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
