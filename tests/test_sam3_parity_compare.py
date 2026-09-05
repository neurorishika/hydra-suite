"""Unit tests for the Task 0 (SAM3 spike-parity) pure statistics functions.

These run entirely on synthetic data -- no sam3, no calibration harness,
no GPU -- because they are the real gate for `tools/sam3_parity/compare_models.py`
(see that module's docstring: Step 6, the live run, is deferred).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "sam3_parity"))

from compare_models import (  # noqa: E402
    OperatingPoint,
    PairedComparison,
    average_precision,
    bootstrap_ci,
    extras_unique_to_a,
    interpolate_extras_at_recall,
    paired_comparison,
    paired_frame_diffs,
    precision_recall_curve,
    sign_test,
    unmatched_predictions,
)

# ---------------------------------------------------------------------------
# Step 1: paired per-frame diffs
# ---------------------------------------------------------------------------


def test_paired_frame_diffs_elementwise():
    diffs = paired_frame_diffs([5.0, 3.0, 8.0], [4.0, 4.0, 6.0])
    np.testing.assert_allclose(diffs, [1.0, -1.0, 2.0])


def test_paired_frame_diffs_rejects_length_mismatch():
    with pytest.raises(ValueError):
        paired_frame_diffs([1.0, 2.0], [1.0])


def test_paired_frame_diffs_rejects_empty():
    with pytest.raises(ValueError):
        paired_frame_diffs([], [])


# ---------------------------------------------------------------------------
# Sign test
# ---------------------------------------------------------------------------


def test_sign_test_all_positive_is_significant():
    # 10 of 10 frames favour A: this should be a small p-value regardless of
    # magnitude (sign test only reads the sign).
    diffs = [0.1] * 10
    p = sign_test(diffs)
    assert p < 0.01


def test_sign_test_balanced_is_not_significant():
    diffs = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    p = sign_test(diffs)
    assert p == pytest.approx(1.0)


def test_sign_test_ties_are_dropped():
    # Two exact zeros contribute nothing; among the rest, split evenly.
    diffs = [0.0, 0.0, 1.0, -1.0]
    p = sign_test(diffs)
    assert p == pytest.approx(1.0)


def test_sign_test_all_ties_returns_one():
    assert sign_test([0.0, 0.0, 0.0]) == 1.0


def test_sign_test_matches_hand_rolled_fallback():
    diffs = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
    p_scipy = sign_test(diffs)

    import compare_models as cm

    saved = cm._scipy_stats
    try:
        cm._scipy_stats = None
        p_manual = sign_test(diffs)
    finally:
        cm._scipy_stats = saved
    assert p_scipy == pytest.approx(p_manual, abs=1e-9)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_excludes_zero_for_clear_effect():
    rng = np.random.default_rng(42)
    diffs = rng.normal(loc=2.0, scale=0.3, size=200)
    lo, hi = bootstrap_ci(diffs, n_resamples=2000, seed=1)
    assert lo > 0.0
    assert lo < hi

    # Sanity: the CI should bracket the true mean closely at this n.
    assert lo < 2.0 < hi


def test_bootstrap_ci_includes_zero_for_null_effect():
    # Construct diffs whose sample mean is exactly zero, so the null-effect
    # assertion does not depend on a particular RNG draw landing close
    # enough to zero for the bootstrap to bracket it.
    diffs = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5] * 4)
    lo, hi = bootstrap_ci(diffs, n_resamples=2000, seed=2)
    assert lo <= 0.0 <= hi


def test_bootstrap_ci_is_deterministic_given_seed():
    diffs = [1.0, 2.0, -1.0, 0.5, 3.0]
    a = bootstrap_ci(diffs, n_resamples=500, seed=99)
    b = bootstrap_ci(diffs, n_resamples=500, seed=99)
    assert a == b


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


# ---------------------------------------------------------------------------
# paired_comparison end to end
# ---------------------------------------------------------------------------


def test_paired_comparison_significant_effect():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=8.0, scale=1.0, size=64)
    b = a - rng.normal(loc=1.5, scale=0.2, size=64)
    result = paired_comparison(a, b, n_resamples=2000, seed=3)
    assert isinstance(result, PairedComparison)
    assert result.n_frames == 64
    assert result.significant
    assert result.mean_diff == pytest.approx(1.5, abs=0.3)


def test_paired_comparison_null_effect_is_not_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=6.0, scale=1.0, size=16)
    b = a + rng.normal(loc=0.0, scale=0.05, size=16)
    result = paired_comparison(a, b, n_resamples=2000, seed=4)
    # At n=16 with genuinely tiny noise around zero difference, the CI
    # should not reliably exclude zero.
    assert result.ci_low < 0.0 < result.ci_high or not result.significant


def test_paired_comparison_matches_the_documented_noise_floor_case():
    # The exact scenario the plan's Task 0 rationale describes: ~5-8
    # extras/frame, a 1.2-2.0 target delta, 16 frames, sd ~0.6-0.7 on
    # INDEPENDENT means. Paired, the same per-frame noise should let a
    # ~1.5 delta resolve when frame difficulty is shared.
    rng = np.random.default_rng(5)
    frame_difficulty = rng.normal(loc=6.5, scale=2.0, size=16)
    extras_ours = frame_difficulty + rng.normal(loc=0.0, scale=0.4, size=16)
    extras_spike = frame_difficulty - 1.5 + rng.normal(loc=0.0, scale=0.4, size=16)
    result = paired_comparison(extras_ours, extras_spike, n_resamples=4000, seed=6)
    assert result.significant
    assert result.mean_diff > 0


# ---------------------------------------------------------------------------
# Step 3: matched-recall interpolation
# ---------------------------------------------------------------------------


def _sweep(points):
    return [
        OperatingPoint(confidence=c, recall=r, extra_per_frame=e) for c, r, e in points
    ]


def test_interpolate_extras_at_recall_linear_midpoint():
    points = _sweep([(0.9, 0.5, 2.0), (0.5, 0.7, 6.0), (0.1, 0.9, 12.0)])
    # Recall 0.6 sits halfway between 0.5 (extra=2.0) and 0.7 (extra=6.0).
    value = interpolate_extras_at_recall(points, 0.6)
    assert value == pytest.approx(4.0)


def test_interpolate_extras_at_recall_exact_grid_point():
    points = _sweep([(0.9, 0.5, 2.0), (0.5, 0.7, 6.0)])
    assert interpolate_extras_at_recall(points, 0.7) == pytest.approx(6.0)


def test_interpolate_extras_at_recall_out_of_range_returns_none():
    points = _sweep([(0.9, 0.5, 2.0), (0.5, 0.7, 6.0)])
    assert interpolate_extras_at_recall(points, 0.95) is None
    assert interpolate_extras_at_recall(points, 0.1) is None


def test_interpolate_extras_at_recall_empty_points():
    assert interpolate_extras_at_recall([], 0.5) is None


# ---------------------------------------------------------------------------
# Step 4: AP / PR curve
# ---------------------------------------------------------------------------


def test_average_precision_perfect_detector_is_one():
    # recall sweeps 0->1 with precision always 1.0.
    recalls = [0.0, 0.25, 0.5, 0.75, 1.0]
    precisions = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert average_precision(recalls, precisions) == pytest.approx(1.0)


def test_average_precision_worse_detector_scores_lower():
    recalls = [0.0, 0.5, 1.0]
    good = [1.0, 0.9, 0.8]
    bad = [1.0, 0.5, 0.2]
    assert average_precision(recalls, good) > average_precision(recalls, bad)


def test_average_precision_monotone_envelope():
    # A precision dip at an intermediate recall should be papered over by
    # the envelope (precision can only be non-increasing with recall).
    recalls = [0.0, 0.3, 0.6, 1.0]
    precisions = [0.9, 0.5, 0.95, 0.4]
    ap = average_precision(recalls, precisions)
    # Envelope becomes [0.95, 0.95, 0.95, 0.4]; trapz over recall.
    expected_envelope = [0.95, 0.95, 0.95, 0.4]
    expected = np.trapezoid(expected_envelope, recalls)
    assert ap == pytest.approx(expected)


def test_average_precision_empty_is_zero():
    assert average_precision([], []) == 0.0


def test_precision_recall_curve_sorted_and_bounded():
    points = _sweep([(0.9, 0.5, 2.0), (0.5, 0.8, 6.0), (0.1, 0.95, 20.0)])
    recalls, precisions = precision_recall_curve(points)
    assert list(recalls) == sorted(recalls)
    assert np.all(precisions >= 0.0)
    assert np.all(precisions <= 1.0)


def test_precision_recall_curve_with_missed_per_frame_recovers_scale():
    # matched/frame = recall * total, total = matched + missed.
    # recall=0.5, missed=5 -> total=10, matched=5; extra=5 -> precision=0.5.
    points = _sweep([(0.5, 0.5, 5.0)])
    recalls, precisions = precision_recall_curve(points, missed_per_frame=[5.0])
    assert precisions[0] == pytest.approx(0.5)


def test_precision_recall_curve_rejects_misaligned_missed():
    points = _sweep([(0.5, 0.5, 5.0)])
    with pytest.raises(ValueError):
        precision_recall_curve(points, missed_per_frame=[1.0, 2.0])


# ---------------------------------------------------------------------------
# Step 5: adjudication helpers
# ---------------------------------------------------------------------------


def _square(cx, cy, half=1.0):
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float32,
    )


def _stub_match_fn(preds, labels, **_kwargs):
    """Minimal one-to-one match: pair by containment of centroid, greedy."""
    pairs = []
    used_labels = set()
    for pi, pred in enumerate(preds):
        pc = pred.mean(axis=0)
        for gi, label in enumerate(labels):
            if gi in used_labels:
                continue
            lo = label.min(axis=0)
            hi = label.max(axis=0)
            if np.all(pc >= lo) and np.all(pc <= hi):
                pairs.append((pi, gi))
                used_labels.add(gi)
                break
    return pairs


def _iou(a, b):
    # Axis-aligned box IoU sufficient for these synthetic squares.
    a_lo, a_hi = a.min(axis=0), a.max(axis=0)
    b_lo, b_hi = b.min(axis=0), b.max(axis=0)
    inter_lo = np.maximum(a_lo, b_lo)
    inter_hi = np.minimum(a_hi, b_hi)
    inter_wh = np.clip(inter_hi - inter_lo, 0, None)
    inter = inter_wh[0] * inter_wh[1]
    area_a = (a_hi[0] - a_lo[0]) * (a_hi[1] - a_lo[1])
    area_b = (b_hi[0] - b_lo[0]) * (b_hi[1] - b_lo[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def test_unmatched_predictions_finds_extras():
    labels = [_square(0, 0), _square(10, 10)]
    preds = [_square(0, 0), _square(10, 10), _square(50, 50)]
    unmatched = unmatched_predictions(preds, labels, match_fn=_stub_match_fn)
    assert unmatched == [2]


def test_unmatched_predictions_none_when_all_matched():
    labels = [_square(0, 0)]
    preds = [_square(0, 0)]
    assert unmatched_predictions(preds, labels, match_fn=_stub_match_fn) == []


def test_extras_unique_to_a_excludes_agreeing_extras():
    labels = [_square(0, 0)]
    # Both models independently emit an extra at (50, 50): likely a real,
    # unlabelled ant, not a lucky coincidence of clutter -- should NOT be
    # flagged as "unique to A".
    preds_a = [_square(0, 0), _square(50, 50)]
    preds_b = [_square(0, 0), _square(50.2, 50.2)]
    unique = extras_unique_to_a(
        preds_a, preds_b, labels, match_fn=_stub_match_fn, iou_fn=_iou
    )
    assert unique == []


def test_extras_unique_to_a_keeps_disagreeing_extras():
    labels = [_square(0, 0)]
    preds_a = [_square(0, 0), _square(50, 50)]
    preds_b = [_square(0, 0)]  # B has no counterpart near (50, 50).
    unique = extras_unique_to_a(
        preds_a, preds_b, labels, match_fn=_stub_match_fn, iou_fn=_iou
    )
    assert unique == [1]


def test_extras_unique_to_a_empty_b_flags_all_unmatched():
    labels = []
    preds_a = [_square(0, 0), _square(5, 5)]
    preds_b = []
    unique = extras_unique_to_a(
        preds_a, preds_b, labels, match_fn=_stub_match_fn, iou_fn=_iou
    )
    assert unique == [0, 1]
