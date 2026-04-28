"""Tests for heading consistency improvements.

Covers:
1. Kalman filter theta normalization after predict/correct.
2. Post-processing heading flip correction (_fix_heading_flips).
3. Worker _smooth_orientation hysteresis (unit-level).
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Kalman theta normalization
# ---------------------------------------------------------------------------


class TestKalmanThetaNormalization:
    """Verify that Kalman state theta stays within [0, 2*pi)."""

    def _make_kf(self, n=1):
        from hydra_suite.core.filters.kalman import KalmanFilterManager

        params = {
            "REFERENCE_BODY_SIZE": 20.0,
            "RESIZE_FACTOR": 1.0,
            "KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0,
            "KALMAN_NOISE_COVARIANCE": 0.03,
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": 0.1,
            "KALMAN_DAMPING": 0.95,
        }
        return KalmanFilterManager(n, params)

    def test_theta_stays_normalized_after_predict(self):
        kf = self._make_kf()
        kf.initialize_filter(0, np.array([100, 100, 5.5, 5, 3], dtype=np.float32))
        for _ in range(50):
            kf.predict()
            theta = float(kf.X[0, 2])
            assert 0.0 <= theta < 2 * np.pi, f"theta={theta} out of [0, 2pi)"

    def test_theta_stays_normalized_after_correct(self):
        kf = self._make_kf()
        kf.initialize_filter(0, np.array([100, 100, 0.1, 0, 0], dtype=np.float32))
        # Feed measurements spanning full circle
        for meas_theta in np.linspace(0, 2 * np.pi, 40, endpoint=False):
            kf.predict()
            meas = np.array([100, 100, meas_theta], dtype=np.float32)
            kf.correct(0, meas)
            theta = float(kf.X[0, 2])
            assert 0.0 <= theta < 2 * np.pi, f"theta={theta} out of [0, 2pi)"

    def test_theta_does_not_drift_with_repeated_cycles(self):
        """Simulate 500 predict/correct cycles with near-constant measurement."""
        kf = self._make_kf()
        kf.initialize_filter(0, np.array([50, 50, 3.0, 0, 0], dtype=np.float32))
        for i in range(500):
            kf.predict()
            # Small oscillation around 3.0 radians
            meas_theta = 3.0 + 0.05 * np.sin(i * 0.1)
            kf.correct(0, np.array([50, 50, meas_theta], dtype=np.float32))
        theta = float(kf.X[0, 2])
        assert 0.0 <= theta < 2 * np.pi

    def test_theta_r_scale_increases_noise(self):
        """theta_r_scale > 1 should yield bigger theta variance after correction."""
        kf1 = self._make_kf()
        kf2 = self._make_kf()
        init = np.array([50, 50, 1.0, 0, 0], dtype=np.float32)
        kf1.initialize_filter(0, init.copy())
        kf2.initialize_filter(0, init.copy())

        kf1.predict()
        kf2.predict()

        meas = np.array([55, 55, 2.0], dtype=np.float32)
        kf1.correct(0, meas, theta_r_scale=1.0)
        kf2.correct(0, meas, theta_r_scale=5.0)

        # With higher theta_r_scale, the filter trusts the measurement less,
        # so the state theta should stay closer to the prediction (1.0).
        err1 = abs(float(kf1.X[0, 2]) - 1.0)
        err2 = abs(float(kf2.X[0, 2]) - 1.0)
        assert err2 < err1, (
            f"Higher R scaling should keep theta closer to prediction: "
            f"err_normal={err1:.4f}, err_scaled={err2:.4f}"
        )


# ---------------------------------------------------------------------------
# Post-processing _fix_heading_flips
# ---------------------------------------------------------------------------


class TestFixHeadingFlips:
    """Test the post-processing heading flip correction."""

    def _fix(self, theta, max_burst=5):
        from hydra_suite.core.post.processing import _fix_heading_flips

        return _fix_heading_flips(np.asarray(theta, dtype=np.float64), max_burst)

    def test_single_frame_flip_corrected(self):
        """A single-frame 180° flip should be corrected."""
        base = 1.0  # ~57 degrees
        theta = [base, base, base + np.pi, base, base]
        result = self._fix(theta)
        for i, val in enumerate(result):
            diff = abs(val - base) % (2 * np.pi)
            diff = min(diff, 2 * np.pi - diff)
            assert diff < 0.5, f"Frame {i}: expected ~{base}, got {val}"

    def test_multi_frame_burst_corrected(self):
        """A 3-frame flip burst should be corrected (within max_burst=5)."""
        base = 2.0
        theta = [base] * 5 + [base + np.pi] * 3 + [base] * 5
        result = self._fix(theta)
        for i, val in enumerate(result):
            diff = abs(val - base) % (2 * np.pi)
            diff = min(diff, 2 * np.pi - diff)
            assert diff < 0.5, f"Frame {i}: expected ~{base}, got {val}"

    def test_long_flip_not_corrected(self):
        """A flip lasting > max_burst frames should be left alone."""
        base = 1.5
        flipped = (base + np.pi) % (2 * np.pi)
        theta = [base] * 5 + [flipped] * 10 + [base] * 5
        result = self._fix(theta, max_burst=5)
        # The long segment should remain flipped
        for i in range(5, 15):
            diff = abs(result[i] - flipped) % (2 * np.pi)
            diff = min(diff, 2 * np.pi - diff)
            assert diff < 0.5, f"Frame {i}: long flip should be preserved"

    def test_nan_handling(self):
        """NaN values in theta should be preserved."""
        base = 1.0
        theta = [base, np.nan, base + np.pi, np.nan, base]
        result = self._fix(theta)
        assert np.isnan(result[1])
        assert np.isnan(result[3])

    def test_no_flips_untouched(self):
        """A smooth trajectory should not be modified."""
        theta = np.linspace(0.5, 1.5, 20)
        result = self._fix(theta)
        np.testing.assert_allclose(result, theta, atol=1e-10)

    def test_empty_and_short(self):
        """Edge cases: empty, 1-element, 2-element arrays."""
        assert len(self._fix([])) == 0
        assert len(self._fix([1.0])) == 1
        np.testing.assert_allclose(self._fix([1.0, 1.0]), [1.0, 1.0])

    def test_wraparound_flip(self):
        """Flip near the 0/2pi boundary (e.g., 0.1 → 3.24)."""
        base = 0.1
        flipped = base + np.pi  # ~3.24
        theta = [base, base, flipped, base, base]
        result = self._fix(theta)
        for val in result:
            if not np.isnan(val):
                diff = abs(val - base) % (2 * np.pi)
                diff = min(diff, 2 * np.pi - diff)
                assert diff < 0.5


# ---------------------------------------------------------------------------
# Worker _smooth_orientation hysteresis
# ---------------------------------------------------------------------------


class TestSmoothOrientationHysteresis:
    """Test that the heading flip hysteresis gate works."""

    def _make_worker_stub(self):
        """Create a minimal mock for _smooth_orientation testing."""
        from unittest.mock import MagicMock

        from hydra_suite.core.tracking.worker import TrackingWorker

        worker = MagicMock(spec=TrackingWorker)
        worker.backward_mode = False
        worker._smooth_orientation = TrackingWorker._smooth_orientation.__get__(
            worker, TrackingWorker
        )
        worker._normalize_theta = TrackingWorker._normalize_theta
        worker._circular_abs_diff_rad = TrackingWorker._circular_abs_diff_rad
        worker._collapse_obb_axis_theta = (
            TrackingWorker._collapse_obb_axis_theta.__get__(worker, TrackingWorker)
        )
        return worker

    def test_hysteresis_blocks_single_frame_flip(self):
        """A single directed stationary 180° flip should be blocked."""
        from collections import deque

        worker = self._make_worker_stub()
        base = 1.0
        flipped = base + math.pi
        orientation_last = [base]
        pos_deques = [deque(maxlen=2)]
        flip_counters = [0]
        params = {
            "DIRECTED_ORIENT_SMOOTHING": True,
            "DIRECTED_ORIENT_FLIP_CONFIDENCE": 0.5,
            "DIRECTED_ORIENT_FLIP_PERSISTENCE": 3,
            "VELOCITY_THRESHOLD": 2.0,
        }

        # Single frame requesting a flip — should be rejected
        result = worker._smooth_orientation(
            0,
            flipped,
            0.0,
            params,
            orientation_last,
            pos_deques,
            directed_heading=True,
            orient_confidence=0.9,
            heading_flip_counters=flip_counters,
        )
        # Should remain close to base (flip rejected)
        diff = abs(result - base) % (2 * math.pi)
        diff = min(diff, 2 * math.pi - diff)
        assert diff < 0.5, f"Single-frame flip should be rejected, got {result}"
        assert flip_counters[0] == 0

    def test_hysteresis_rejects_stationary_flip_after_persistence(self):
        """Persistent stationary 180° swaps should remain rejected."""
        from collections import deque

        worker = self._make_worker_stub()
        base = 1.0
        flipped = base + math.pi
        orientation_last = [base]
        pos_deques = [deque(maxlen=2)]
        flip_counters = [0]
        params = {
            "DIRECTED_ORIENT_SMOOTHING": True,
            "DIRECTED_ORIENT_FLIP_CONFIDENCE": 0.5,
            "DIRECTED_ORIENT_FLIP_PERSISTENCE": 3,
            "VELOCITY_THRESHOLD": 2.0,
        }

        for _ in range(5):
            result = worker._smooth_orientation(
                0,
                flipped,
                0.0,
                params,
                orientation_last,
                pos_deques,
                directed_heading=True,
                orient_confidence=0.9,
                heading_flip_counters=flip_counters,
            )
            orientation_last[0] = result

        diff = abs(result - base) % (2 * math.pi)
        diff = min(diff, 2 * math.pi - diff)
        assert (
            diff < 0.5
        ), f"Stationary persistent flip should remain rejected, got {result}"
        assert flip_counters[0] == 0

    def test_hysteresis_accepts_motion_supported_flip_after_persistence(self):
        """A persistent flip should be accepted once motion corroborates it."""
        from collections import deque

        worker = self._make_worker_stub()
        base = 0.0
        flipped = math.pi
        orientation_last = [base]
        pos_deques = [deque([(0.0, 0.0, 0), (-5.0, 0.0, 1)], maxlen=2)]
        flip_counters = [0]
        params = {
            "DIRECTED_ORIENT_SMOOTHING": True,
            "DIRECTED_ORIENT_FLIP_CONFIDENCE": 0.5,
            "DIRECTED_ORIENT_FLIP_PERSISTENCE": 3,
            "VELOCITY_THRESHOLD": 2.0,
        }

        for _ in range(3):
            result = worker._smooth_orientation(
                0,
                flipped,
                5.0,
                params,
                orientation_last,
                pos_deques,
                directed_heading=True,
                orient_confidence=0.9,
                heading_flip_counters=flip_counters,
            )
            orientation_last[0] = result

        diff = abs(result - flipped) % (2 * math.pi)
        diff = min(diff, 2 * math.pi - diff)
        assert (
            diff < 0.5
        ), f"Motion-supported flip should be accepted after persistence, got {result}"
        assert flip_counters[0] == 0
