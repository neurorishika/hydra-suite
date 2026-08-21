"""Tests for post-processing tag identity utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "hydra_suite/core/post/tag_identity.py",
    "tag_identity_under_test",
)
resolve_tag_identities = mod.resolve_tag_identities
detect_tag_swaps = mod.detect_tag_swaps


class FakeTagCache:
    """Minimal stand-in for TagObservationCache."""

    def __init__(self, data, total_frames=100):
        self._data = data
        self._total = total_frames

    def get_frame(self, frame_idx):
        empty = {
            "tag_ids": np.array([], dtype=np.int32),
            "centers_xy": np.zeros((0, 2), dtype=np.float32),
        }
        return self._data.get(frame_idx, empty)

    def get_total_frames(self):
        return self._total


# ---------------------------------------------------------------------------
# resolve_tag_identities
# ---------------------------------------------------------------------------


def test_resolve_tag_identities_majority_vote():
    """Tag that appears most often near a trajectory wins."""
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0, 0],
            "FrameID": [1, 2, 3, 4],
            "X": [10.0, 10.0, 10.0, 10.0],
            "Y": [20.0, 20.0, 20.0, 20.0],
        }
    )
    cache = FakeTagCache(
        {
            1: {
                "tag_ids": np.array([5], dtype=np.int32),
                "centers_xy": np.array([[10.0, 20.0]], dtype=np.float32),
            },
            2: {
                "tag_ids": np.array([5], dtype=np.int32),
                "centers_xy": np.array([[10.0, 20.0]], dtype=np.float32),
            },
            3: {
                "tag_ids": np.array([7], dtype=np.int32),
                "centers_xy": np.array([[10.0, 20.0]], dtype=np.float32),
            },
            4: {
                "tag_ids": np.array([5], dtype=np.int32),
                "centers_xy": np.array([[10.0, 20.0]], dtype=np.float32),
            },
        }
    )
    result = resolve_tag_identities(df, cache, {})
    assert "TagID" in result.columns
    assert "TagVotes" in result.columns
    assert (result["TagID"] == 5).all()
    assert (result["TagVotes"] == 3).all()


def test_resolve_tag_identities_no_cache():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0],
            "FrameID": [1],
            "X": [10.0],
            "Y": [20.0],
        }
    )
    result = resolve_tag_identities(df, None, {})
    assert (result["TagID"] == -1).all()


def test_resolve_tag_identities_out_of_radius():
    """Tags too far from the trajectory should not be associated."""
    df = pd.DataFrame(
        {
            "TrajectoryID": [0],
            "FrameID": [1],
            "X": [10.0],
            "Y": [20.0],
        }
    )
    cache = FakeTagCache(
        {
            1: {
                "tag_ids": np.array([5], dtype=np.int32),
                "centers_xy": np.array([[999.0, 999.0]], dtype=np.float32),
            },
        }
    )
    result = resolve_tag_identities(df, cache, {"TAG_ASSOCIATION_RADIUS": 50.0})
    assert (result["TagID"] == -1).all()


# ---------------------------------------------------------------------------
# detect_tag_swaps
# ---------------------------------------------------------------------------


def test_detect_tag_swaps_simple():
    """A tag switching between two trajectories should trigger a swap."""
    # Trajectory 0 at (10,10), Trajectory 1 at (100,100)
    rows = []
    for fid in range(1, 11):
        rows.append({"TrajectoryID": 0, "FrameID": fid, "X": 10.0, "Y": 10.0})
        rows.append({"TrajectoryID": 1, "FrameID": fid, "X": 100.0, "Y": 100.0})
    df = pd.DataFrame(rows)

    # Tag 5 near traj 0 for frames 1-5, then near traj 1 for frames 6-10
    cache_data = {}
    for fid in range(1, 6):
        cache_data[fid] = {
            "tag_ids": np.array([5], dtype=np.int32),
            "centers_xy": np.array([[10.0, 10.0]], dtype=np.float32),
        }
    for fid in range(6, 11):
        cache_data[fid] = {
            "tag_ids": np.array([5], dtype=np.int32),
            "centers_xy": np.array([[100.0, 100.0]], dtype=np.float32),
        }
    cache = FakeTagCache(cache_data)

    swaps = detect_tag_swaps(df, cache, {"TAG_SWAP_MIN_STREAK": 3})
    assert len(swaps) >= 1
    assert swaps[0]["tag_id"] == 5
    assert swaps[0]["from_traj"] == 0
    assert swaps[0]["to_traj"] == 1


def test_detect_tag_swaps_no_data():
    result = detect_tag_swaps(None, None, {})
    assert result == []
