"""The resolve stage must report progress, monotonically, inside 30..60%.

`merge_trajectories` emitted 30% ("Resolving trajectory conflicts...") and
then nothing until 60%. On a full-length video that stage is ~3/4 of
post-processing -- three hours on the run that prompted this work -- so a
healthy run was indistinguishable from a hang.
"""

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.post import processing as P
from tests.test_merge_overlap_interval_index import _make_population


def _split(trajs):
    """Split a population into a forward and a backward half that overlap."""
    fwd = [df.copy() for df in trajs[: len(trajs) // 2 + 10]]
    bwd = [df.copy() for df in trajs[len(trajs) // 2 - 10 :]]
    return fwd, bwd


def test_resolve_reports_intermediate_progress():
    fwd, bwd = _split(_make_population(300, seed=2))
    seen = []
    P.resolve_trajectories(
        fwd, bwd, {}, progress=lambda frac, msg: seen.append((frac, msg))
    )

    assert seen, "resolve_trajectories reported no progress at all"
    fractions = [f for f, _ in seen]
    assert all(0.0 <= f <= 1.0 for f in fractions), fractions
    assert fractions == sorted(fractions), "progress went backwards"
    assert len({m for _, m in seen}) > 1, "only one distinct message was reported"


def test_progress_is_optional():
    """The CLI path passes no callback; the resolver must not require one."""
    fwd, bwd = _split(_make_population(120, seed=8))
    out = P.resolve_trajectories(fwd, bwd, {})
    assert isinstance(out, list)


def test_progress_does_not_change_the_result():
    fwd, bwd = _split(_make_population(300, seed=2))
    quiet = P.resolve_trajectories([d.copy() for d in fwd], [d.copy() for d in bwd], {})
    loud = P.resolve_trajectories(
        [d.copy() for d in fwd],
        [d.copy() for d in bwd],
        {},
        progress=lambda frac, msg: None,
    )
    assert len(quiet) == len(loud)
    for k, (a, b) in enumerate(zip(quiet, loud)):
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=f"trajectory[{k}]")


def test_sub_progress_scales_and_clamps():
    seen = []
    scaled = P._sub_progress(lambda f, m: seen.append(f), 0.25, 0.75)
    for value in (-1.0, 0.0, 0.5, 1.0, 2.0):
        scaled(value, "x")
    assert seen == [0.25, 0.25, 0.5, 0.75, 0.75]
    assert P._sub_progress(None, 0.0, 1.0) is None


def test_merge_trajectories_fills_the_30_to_60_band():
    """End-to-end through the caller that owns the band."""
    from hydra_suite.core.post.merge import merge_trajectories

    trajs = _make_population(300, seed=2)
    fwd, bwd = _split(trajs)
    seen = []
    merge_trajectories(
        fwd,
        bwd,
        total_frames=4000,
        params={},
        resize_factor=1.0,
        interp_method="none",
        max_gap=5,
        progress=lambda pct, msg: seen.append((pct, msg)),
    )

    band = [(p, m) for p, m in seen if 30 < p < 60]
    assert band, f"nothing reported between 30% and 60%; got {seen}"
    percents = [p for p, _ in seen]
    assert percents == sorted(percents), f"progress went backwards: {percents}"


@pytest.mark.parametrize("n_arenas", [2, 3])
def test_each_arena_gets_a_slice_of_the_band(n_arenas):
    trajs = _make_population(120, seed=1)
    for k, df in enumerate(trajs):
        df["arena_id"] = k % n_arenas
    fwd, bwd = _split(trajs)

    seen = []
    P.resolve_trajectories(
        fwd, bwd, {}, progress=lambda frac, msg: seen.append((frac, msg))
    )
    arena_msgs = [m for _, m in seen if m.startswith("Resolving arena ")]
    assert len(arena_msgs) == n_arenas, arena_msgs
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert max(fractions) <= 1.0 and min(fractions) >= 0.0


def test_stop_during_resolve_still_reports_nothing_out_of_band():
    fwd, bwd = _split(_make_population(200, seed=7))
    seen = []
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 40

    P.resolve_trajectories(
        fwd,
        bwd,
        {},
        should_stop=should_stop,
        progress=lambda frac, msg: seen.append(frac),
    )
    assert all(0.0 <= f <= 1.0 for f in seen)


def test_damped_pass_fraction_is_monotone_and_bounded():
    """The overlap-merge loop runs until convergence, so its fraction must rise
    monotonically without ever claiming completion."""
    trajs = _make_population(200, seed=2)
    seen = []
    P._merge_overlapping_agreeing_trajectories(
        [d.copy() for d in trajs],
        15.0,
        5,
        5,
        progress=lambda frac, msg: seen.append(frac),
    )
    assert seen
    assert seen == sorted(seen)
    assert max(seen) < 1.0
    assert np.isclose(seen[0], 0.5)
