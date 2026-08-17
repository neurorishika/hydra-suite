import dataclasses

import pytest

from hydra_suite.data.al.acquisition import (
    AcquisitionWeights,
    _composite_score,
    explain,
    select,
)
from hydra_suite.data.al.signals import (
    ALSignals,
    score_count_deviation,
    score_uncertainty,
)
from hydra_suite.utils.geometry import obb_corners_from_dims


def _box(cx, cy, w=40.0, h=16.0, theta=0.0):
    return obb_corners_from_dims(cx, cy, w, h, theta)


def test_fragmentation_is_zero_for_well_separated_normal_boxes():
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [_box(100, 100), _box(300, 300), _box(500, 100)]
    assert score_fragmentation(boxes) == 0.0


def test_fragmentation_fires_on_two_small_overlapping_boxes():
    """One animal split into two half-size detections sitting on top of it."""
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [
        _box(300, 300),  # normal-size animal
        _box(500, 300),  # normal-size animal
        _box(100, 100, w=18.0, h=8.0),  # fragment
        _box(108, 102, w=18.0, h=8.0),  # its twin, close + small
    ]
    assert score_fragmentation(boxes) > 0.45


def test_fragmentation_needs_at_least_two_boxes():
    from hydra_suite.data.al.signals import score_fragmentation

    assert score_fragmentation([]) == 0.0
    assert score_fragmentation([_box(100, 100)]) == 0.0


def test_fragmentation_is_bounded():
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [_box(100, 100, w=5, h=3), _box(100, 100, w=5, h=3)]
    assert 0.0 <= score_fragmentation(boxes) <= 1.0


def test_tracker_default_preset_weights_fragmentation():
    from hydra_suite.data.al.acquisition import PRESETS

    assert PRESETS["tracker_default"].fragmentation == 0.30


def test_al_signals_carries_fragmentation_field():
    from hydra_suite.data.al.signals import ALSignals

    assert ALSignals(frame_id=0).fragmentation_score == 0.0


def test_uncertainty_is_zero_above_the_floor():
    assert score_uncertainty([0.9, 0.8], conf_floor=0.5) == 0.0


def test_uncertainty_rises_as_confidence_falls():
    low = score_uncertainty([0.1], conf_floor=0.5)
    mid = score_uncertainty([0.3], conf_floor=0.5)
    assert 0.0 < mid < low <= 1.0


def test_uncertainty_of_all_nan_confidences_is_zero():
    """bg-sub emits all-NaN confidences; that must not read as 'uncertain'."""
    assert score_uncertainty([float("nan"), float("nan")]) == 0.0


def test_count_deviation_is_zero_on_exact_match():
    assert score_count_deviation(4, 4) == 0.0


def test_count_deviation_penalizes_undercount_twice_as_hard():
    under = score_count_deviation(2, 4)  # missed two animals
    over = score_count_deviation(6, 4)  # two spurious boxes
    assert under == pytest.approx(2 * over)


def test_composite_score_is_zero_for_a_clean_frame():
    """A frame with nothing wrong must score exactly 0, not 'least bad'."""
    clean = ALSignals(frame_id=0, n_detections=4, mean_confidence=0.95)
    weights = AcquisitionWeights(uncertainty=1.0)
    assert select([clean], weights=weights, k=1, min_score=0.01) == []


def test_min_score_gate_is_comparable_across_runs():
    weights = AcquisitionWeights(uncertainty=1.0)
    # ALSignals no longer derives uncertainty_score from mean_confidence (that
    # hardcoded-floor derivation was removed -- it silently disagreed with
    # real per-caller floors like al_worker.py's base_conf=0.25). Every
    # constructor, including this fixture, must compute it itself via
    # score_uncertainty(confidences, conf_floor=...), same as production code.
    mild = ALSignals(
        frame_id=0,
        mean_confidence=0.45,
        uncertainty_score=score_uncertainty([0.45], conf_floor=0.5),
    )
    severe = ALSignals(
        frame_id=9999,
        mean_confidence=0.05,
        uncertainty_score=score_uncertainty([0.05], conf_floor=0.5),
    )

    # Severe alone, and severe alongside mild, must both clear a 0.5 gate --
    # under min-max normalization the lone frame would have normalized to 0.
    assert select([severe], weights=weights, k=5, min_score=0.5, probabilistic=False)
    picked = select(
        [mild, severe], weights=weights, k=5, min_score=0.5, probabilistic=False
    )
    assert picked == [9999]


def test_explain_reports_per_channel_maxima():
    weights = AcquisitionWeights(uncertainty=0.5, count=0.5)
    signals = [
        ALSignals(
            frame_id=0,
            mean_confidence=0.4,
            uncertainty_score=score_uncertainty([0.4], conf_floor=0.5),
            count_deviation=0.25,
        ),
        ALSignals(
            frame_id=1,
            mean_confidence=0.2,
            uncertainty_score=score_uncertainty([0.2], conf_floor=0.5),
            count_deviation=0.10,
        ),
    ]
    report = explain(signals, weights)
    assert report["uncertainty"] == pytest.approx(0.6)
    assert report["count"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# End-to-end wiring guard: every channel in AcquisitionWeights must actually
# reach `_composite_score`. Nothing above exercises this directly -- every
# existing test scores the signal functions in isolation, so a future edit
# that silently drops a channel from `_composite_score`'s `channels` dict or
# from `_channel_array`'s `attr_map` would leave every other test green while
# that channel contributes zero forever. This test builds, for each weighted
# channel, an ALSignals whose only nonzero severity is that channel, weights
# only that channel, and asserts the composite score is nonzero.
# ---------------------------------------------------------------------------

_CHANNEL_SIGNAL_BUILDERS = {
    "uncertainty": lambda: ALSignals(frame_id=0, uncertainty_score=0.5),
    "nms_instability": lambda: ALSignals(frame_id=0, nms_instability=0.5),
    "count": lambda: ALSignals(frame_id=0, count_deviation=0.5),
    "crowd": lambda: ALSignals(frame_id=0, crowd_score=0.5),
    "fragmentation": lambda: ALSignals(frame_id=0, fragmentation_score=0.5),
    "edge": lambda: ALSignals(frame_id=0, edge_score=0.5),
    "assignment": lambda: ALSignals(frame_id=0, extras={"assignment": 0.5}),
    "track_loss": lambda: ALSignals(frame_id=0, extras={"track_loss": 0.5}),
    "position_uncertainty": lambda: ALSignals(
        frame_id=0, extras={"position_uncertainty": 0.5}
    ),
}

# The builder map above is hand-maintained, so it does not self-extend: a new
# AcquisitionWeights field added but never wired into a builder here (or into
# _composite_score) would slip through silently -- the same failure mode this
# guard exists to catch, one generation later. Assert the two field sets stay
# in lockstep.
assert set(_CHANNEL_SIGNAL_BUILDERS) == {
    f.name for f in dataclasses.fields(AcquisitionWeights)
}


@pytest.mark.parametrize("channel", sorted(_CHANNEL_SIGNAL_BUILDERS))
def test_composite_score_wires_every_weighted_channel(channel):
    signal = _CHANNEL_SIGNAL_BUILDERS[channel]()
    zeroed = {f.name: 0.0 for f in dataclasses.fields(AcquisitionWeights)}
    zeroed[channel] = 1.0
    weights = AcquisitionWeights(**zeroed)

    score = _composite_score([signal], weights)

    assert score[0] > 0.0, f"channel {channel!r} did not reach _composite_score"
