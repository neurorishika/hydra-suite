"""Windowed relink candidate search returns exactly the all-pairs result."""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.post.processing import _relink_candidates, _score_relink_candidate

ARGS = (6, 60.0, 15.0, 0.45, 0.4, float(np.pi / 3), 1e-3)


def _fragments(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n):
        s = int(rng.integers(0, 400))
        e = s + int(rng.integers(0, 40))
        out.append(
            {
                "traj_id": t,
                "start_frame": s,
                "end_frame": e,
                "identity_sources": {} if rng.uniform() < 0.5 else {"cnn": str(t % 3)},
                "start_pos": rng.uniform(0, 100, 2).astype(np.float32),
                "end_pos": rng.uniform(0, 100, 2).astype(np.float32),
                "start_velocity": rng.normal(0, 2, 2).astype(np.float32),
                "end_velocity": rng.normal(0, 2, 2).astype(np.float32),
                "start_speed": float(rng.uniform(0, 3)),
                "end_speed": float(rng.uniform(0, 3)),
                "start_theta": float(rng.uniform(0, 6.28)),
                "end_theta": float(rng.uniform(0, 6.28)),
                "start_pose": None,
                "end_pose": None,
                "start_pose_quality": 0.0,
                "end_pose_quality": 0.0,
                "start_vis": 0.0,
                "end_vis": 0.0,
            }
        )
    return out


def _all_pairs(fragments):
    out = []
    for src in fragments:
        for dst in fragments:
            if (
                src["traj_id"] == dst["traj_id"]
                or dst["start_frame"] <= src["end_frame"]
            ):
                continue
            r = _score_relink_candidate(src, dst, *ARGS)
            if r is not None:
                out.append(r)
    return out


@pytest.mark.parametrize("seed", range(20))
def test_windowed_candidates_equal_all_pairs(seed):
    frags = _fragments(seed, 200)
    assert _relink_candidates(frags, *ARGS) == _all_pairs(frags)
