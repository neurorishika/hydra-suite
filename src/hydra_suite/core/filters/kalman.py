"""
Biologically-Constrained Vectorized Kalman Filter.
Features:
1. Anisotropic Process Noise (Longitudinal vs. Lateral uncertainty)
2. Velocity Damping (Friction) for stop-and-go behavior
3. Joseph-Form Numerical Stability
4. Circular Angle Wrap-around
"""

import logging

import numpy as np

from hydra_suite.utils.gpu_utils import NUMBA_AVAILABLE, njit

logger = logging.getLogger(__name__)


# --- Numba Kernels (Optimized for Large N) ---
@njit(cache=True)
def _predict_kernel(X, P, F, Q_base, q_long, q_lat):
    """
    Predicts next state and rotates process noise to align with animal heading.
    """
    for i in range(len(X)):
        # 1. Rotate Process Noise based on current orientation
        theta = X[i, 2]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Calculate rotated velocity noise components
        # (Rotates a diagonal [q_long, q_lat] matrix by theta)
        r11 = cos_t**2 * q_long + sin_t**2 * q_lat
        r12 = cos_t * sin_t * (q_long - q_lat)
        r22 = sin_t**2 * q_long + cos_t**2 * q_lat

        # Apply specific noise to this animal
        Qi = Q_base.copy()
        Qi[3, 3] = r11
        Qi[3, 4] = r12
        Qi[4, 3] = r12
        Qi[4, 4] = r22

        # 2. State Prediction: X = F @ X
        X[i] = F @ X[i]

        # 3. Covariance Prediction: P = FPF^T + Qi
        P[i] = F @ P[i] @ F.T + Qi

        # 4. Stability: Variance Floor
        # Prevents the filter from 'collapsing' during long pauses
        for j in range(5):
            if P[i, j, j] < 0.1:
                P[i, j, j] = 0.1

        # 4b. Symmetrize P after predict: float32 matrix multiply F@P@F.T
        # accumulates asymmetry over time, which can make S indefinite.
        for _pi in range(5):
            for _pj in range(_pi + 1, 5):
                _pavg = (P[i, _pi, _pj] + P[i, _pj, _pi]) * 0.5
                P[i, _pi, _pj] = _pavg
                P[i, _pj, _pi] = _pavg

        # 4c. Cauchy-Schwarz clamp: |P[r,c]| <= sqrt(P[r,r]*P[c,c]).
        # Necessary condition for a valid covariance matrix; prevents
        # off-diagonal drift from making S near-singular.
        for _pi in range(5):
            for _pj in range(_pi + 1, 5):
                _lim = (P[i, _pi, _pi] * P[i, _pj, _pj]) ** 0.5
                if P[i, _pi, _pj] > _lim:
                    P[i, _pi, _pj] = _lim
                    P[i, _pj, _pi] = _lim
                elif P[i, _pi, _pj] < -_lim:
                    P[i, _pi, _pj] = -_lim
                    P[i, _pj, _pi] = -_lim

        # 5. Normalize theta to [0, 2*pi) to prevent state drift
        two_pi = 2.0 * np.pi
        X[i, 2] = X[i, 2] - np.floor(X[i, 2] / two_pi) * two_pi

    return X, P


@njit(cache=True)
def _correct_kernel(X, P, H, R, identity_mat, track_idx, measurement, max_velocity):
    """
    Corrects state using Joseph Form for stability and circular angle logic.
    Uses K_eff (effective gain) when innovation is clipped so that the
    covariance update remains consistent with the actually-applied correction.
    """
    x = X[track_idx].reshape(5, 1)
    p = P[track_idx]
    z = measurement.reshape(3, 1)

    # Guard: if P is already corrupted (NaN/Inf from a prior predict step that
    # slipped through sanitize), skip this correction entirely.  The caller's
    # post-correct finiteness check will detect the unchanged non-finite P and
    # call _reset_corrupted_track.
    for _gi in range(5):
        for _gj in range(5):
            if not np.isfinite(p[_gi, _gj]):
                return X, P

    # Innovation
    y = z - (H @ x)

    # --- Circular Angle Wrap ---
    if y[2, 0] > np.pi:
        y[2, 0] -= 2 * np.pi
    elif y[2, 0] < -np.pi:
        y[2, 0] += 2 * np.pi

    # Symmetrize P before use to eliminate float32 off-diagonal drift that
    # accumulates through repeated Joseph-form updates and can make S
    # indefinite even when R provides a positive diagonal floor.
    for _i in range(5):
        for _j in range(_i + 1, 5):
            _avg = (p[_i, _j] + p[_j, _i]) * 0.5
            p[_i, _j] = _avg
            p[_j, _i] = _avg

    # Innovation Covariance & Kalman Gain (computed before any clipping)
    S = (H @ p @ H.T) + R
    # Regularise S before inversion: with extreme anisotropic process noise
    # (ratio up to 260:1) and float32 arithmetic, accumulated P asymmetry can
    # make S near-singular.  A 0.1 diagonal jitter is ~1.7× R_diag (≈0.06)
    # and keeps the condition number bounded without meaningfully biasing the
    # filter.  fastmath=True removed from this kernel for the same reason.
    S[0, 0] += 0.1
    S[1, 1] += 0.1
    S[2, 2] += 0.1
    K = p @ H.T @ np.linalg.inv(S)

    # --- Innovation Clipping ---
    # Cap position innovation to max_velocity.  Build K_eff by scaling the
    # position-measurement columns of K so the Joseph-form covariance update
    # stays consistent with the correction actually applied.
    pos_innov_sq = y[0, 0] ** 2 + y[1, 0] ** 2
    clip_scale = 1.0
    max_v = max(max_velocity, 0.0)
    if pos_innov_sq > max_v**2:
        clip_scale = max_v / max(np.sqrt(pos_innov_sq), 1e-9)
        y[0, 0] *= clip_scale
        y[1, 0] *= clip_scale

    K_eff = K.copy()
    if clip_scale < 1.0:
        for row_i in range(5):
            K_eff[row_i, 0] *= clip_scale
            K_eff[row_i, 1] *= clip_scale

    # Update State (K @ y_clipped == K_eff @ y_orig)
    X[track_idx] = (x + (K @ y)).flatten()

    # Normalize theta to [0, 2*pi) after correction
    two_pi = 2.0 * np.pi
    X[track_idx, 2] = X[track_idx, 2] - np.floor(X[track_idx, 2] / two_pi) * two_pi

    # Apply velocity constraint after correction
    vx, vy = X[track_idx, 3], X[track_idx, 4]
    speed = np.sqrt(vx**2 + vy**2)
    vel_scale = 1.0
    if speed > max_v:
        vel_scale = max_v / max(speed, 1e-9)
        X[track_idx, 3] *= vel_scale
        X[track_idx, 4] *= vel_scale

    # Joseph Form Covariance Update using K_eff: consistent with clipped correction
    IKH = identity_mat - (K_eff @ H)
    P[track_idx] = (IKH @ p @ IKH.T) + (K_eff @ R @ K_eff.T)

    # Symmetrize P after update to prevent off-diagonal float32 drift from
    # compounding across frames into negative eigenvalues.
    for _i in range(5):
        for _j in range(_i + 1, 5):
            _avg = (P[track_idx, _i, _j] + P[track_idx, _j, _i]) * 0.5
            P[track_idx, _i, _j] = _avg
            P[track_idx, _j, _i] = _avg

    # Propagate velocity cap through P: D P D^T where D=diag(1,1,1,vs,vs).
    # Row scaling  (P[3,:] *= vs, P[4,:] *= vs) applies D on the left → D P.
    # Column scaling (P[:,3] *= vs, P[:,4] *= vs) applies D on the right → D P D^T.
    # Since D is diagonal and symmetric (D = D^T), this is exactly the
    # congruence transform that keeps P consistent with the capped velocity state.
    if vel_scale < 1.0:
        P[track_idx, 3, :] *= vel_scale
        P[track_idx, 4, :] *= vel_scale
        P[track_idx, :, 3] *= vel_scale
        P[track_idx, :, 4] *= vel_scale

    return X, P


class KalmanFilterManager:
    """
    Manages a batch of biologically-constrained Kalman Filters.
    """

    def __init__(self, num_targets, params):
        self.num_targets = num_targets
        self.params = params
        self.dim_s = 5  # [x, y, theta, vx, vy]
        self.dim_m = 3  # [x, y, theta]

        # Track ages (number of updates since initialization)
        self.track_ages = np.zeros(num_targets, dtype=np.int32)
        self.age_threshold = max(
            1, int(params.get("KALMAN_MATURITY_AGE", 5))
        )  # Frames to reach full dynamics
        self.initial_velocity_retention = params.get(
            "KALMAN_INITIAL_VELOCITY_RETENTION", 0.2
        )

        # Maximum velocity constraint (pixels/frame, as multiplier of body size)
        # Uses the scaled body size because the KF state lives in resized pixel space.
        reference_body_size = params.get("REFERENCE_BODY_SIZE", 20.0)
        resize_factor = params.get("RESIZE_FACTOR", 1.0)
        max_velocity_multiplier = params.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)
        self.max_velocity = max(
            0.0,
            float(max_velocity_multiplier)
            * float(reference_body_size)
            * float(resize_factor),
        )

        # 1. Initialize State (N, 5)
        self.X = np.zeros((self.num_targets, self.dim_s), dtype=np.float32)

        # 2. Initialize Covariance (N, 5, 5)
        # Moderate uncertainty allows the filter to adapt quickly to initial motion
        self.init_P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        self.P = np.stack([self.init_P.copy() for _ in range(num_targets)])

        # 3. Process Noise Parameters
        q_sigma = float(params.get("KALMAN_NOISE_COVARIANCE", 0.03))
        # High noise forward, low noise sideways (anisotropic).
        # If KALMAN_ANISOTROPY_RATIO is set the lateral multiplier is derived as
        # long / ratio — user specifies the *shape* of the noise ellipse (biology),
        # and the optimizer tunes the *scale* via KALMAN_LONGITUDINAL_NOISE_MULTIPLIER.
        q_long_multiplier = float(
            params.get("KALMAN_LONGITUDINAL_NOISE_MULTIPLIER", 5.0)
        )
        if "KALMAN_ANISOTROPY_RATIO" in params:
            ratio = max(float(params["KALMAN_ANISOTROPY_RATIO"]), 1.0)
            q_lat_multiplier = q_long_multiplier / ratio
        else:
            q_lat_multiplier = float(params.get("KALMAN_LATERAL_NOISE_MULTIPLIER", 0.1))
        self.q_long = q_sigma * q_long_multiplier
        self.q_lat = q_sigma * q_lat_multiplier

        # Base jitter for position and theta
        self.Q_base = np.diag([q_sigma, q_sigma, q_sigma, 0.0, 0.0]).astype(np.float32)

        # 4. Measurement Noise
        r_val = params.get("KALMAN_MEASUREMENT_NOISE_COVARIANCE", 0.1)
        self.R = (np.eye(self.dim_m, dtype=np.float32) * float(r_val)).astype(
            np.float32
        )

        # 5. Transition Matrix F with Friction (Damping)
        # Prevents overshoot when an animal stops suddenly
        damp = float(params.get("KALMAN_DAMPING", 0.95))
        # x_new = x + vx (full step), vx_new = damp * vx (friction only on velocity)
        self.F = np.array(
            [
                [1, 0, 0, 1.0, 0],
                [0, 1, 0, 0, 1.0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, damp, 0],
                [0, 0, 0, 0, damp],
            ],
            dtype=np.float32,
        )

        # 6. Measurement Matrix H
        self.H = np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32
        )

        self.identity_mat = np.eye(self.dim_s, dtype=np.float32)

    def initialize_filter(self, track_idx: int, initial_state: np.ndarray) -> None:
        """Reset one track slot with a new initial state estimate."""
        self.X[track_idx] = initial_state.flatten()
        self.P[track_idx] = self.init_P.copy()
        self.track_ages[track_idx] = 0  # Reset age counter

    def _reset_corrupted_track(self, track_idx: int, reason: str) -> None:
        """Reset one corrupted KF slot while preserving finite position hints."""
        old_x = np.asarray(self.X[track_idx], dtype=np.float32)
        x0 = float(old_x[0]) if np.isfinite(old_x[0]) else 0.0
        y0 = float(old_x[1]) if np.isfinite(old_x[1]) else 0.0
        theta0 = float(old_x[2]) if np.isfinite(old_x[2]) else 0.0
        # Preserve finite velocity so the track can still follow a fast-moving
        # animal on the next frame, clamped to max_velocity for safety.
        vx0 = float(old_x[3]) if np.isfinite(old_x[3]) else 0.0
        vy0 = float(old_x[4]) if np.isfinite(old_x[4]) else 0.0
        if self.max_velocity > 0.0:
            speed = float(np.sqrt(vx0**2 + vy0**2))
            if speed > self.max_velocity:
                scale = self.max_velocity / max(speed, 1e-9)
                vx0 *= scale
                vy0 *= scale
        self.X[track_idx] = np.array([x0, y0, theta0, vx0, vy0], dtype=np.float32)
        self.P[track_idx] = self.init_P.copy()
        self.track_ages[track_idx] = 0
        logger.warning(
            "Kalman track %d reset due to numerical instability: %s",
            int(track_idx),
            reason,
        )

    def _sanitize_all_tracks(self, reason: str) -> None:
        """Reset any track slots with non-finite state or covariance."""
        for i in range(self.num_targets):
            if not (np.isfinite(self.X[i]).all() and np.isfinite(self.P[i]).all()):
                self._reset_corrupted_track(i, reason)

    def predict(self) -> np.ndarray:
        """Predict next measurement-space states for all active track slots."""
        # Standard batch prediction
        if NUMBA_AVAILABLE:
            self.X, self.P = _predict_kernel(
                self.X, self.P, self.F, self.Q_base, self.q_long, self.q_lat
            )
            # Cap per-diagonal covariance to prevent Mahalanobis distances from
            # collapsing toward zero for tracks that have been coasting many frames.
            # Position variance growing to 10,000+ makes essentially every detection
            # in the frame look like a valid match, corrupting assignment.
            _p_max = float(self.params.get("KALMAN_MAX_COVARIANCE_DIAGONAL", 1000.0))
            _diag = np.arange(self.dim_s)
            np.clip(self.P[:, _diag, _diag], 0.1, _p_max, out=self.P[:, _diag, _diag])
            # Cauchy-Schwarz re-enforcement after diagonal cap: the cap may have
            # lowered a diagonal entry, making an existing off-diagonal exceed
            # its bound.  Clamp now so S = H@P[:3,:3]@H.T + R stays PSD.
            _d_sqrt = np.sqrt(np.maximum(self.P[:, _diag, _diag], 0.0))  # (N, dim_s)
            for _r in range(self.dim_s):
                for _c in range(_r + 1, self.dim_s):
                    _lim_rc = _d_sqrt[:, _r] * _d_sqrt[:, _c]
                    np.clip(self.P[:, _r, _c], -_lim_rc, _lim_rc, out=self.P[:, _r, _c])
                    self.P[:, _c, _r] = self.P[:, _r, _c]
        else:
            # Basic NumPy Fallback (Standard isotropic prediction)
            for i in range(self.num_targets):
                theta = self.X[i, 2]
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                r11 = cos_t**2 * self.q_long + sin_t**2 * self.q_lat
                r12 = cos_t * sin_t * (self.q_long - self.q_lat)
                r22 = sin_t**2 * self.q_long + cos_t**2 * self.q_lat

                Qi = self.Q_base.copy()
                Qi[3, 3] = r11
                Qi[3, 4] = r12
                Qi[4, 3] = r12
                Qi[4, 4] = r22

                self.X[i] = self.F @ self.X[i]
                self.P[i] = self.F @ self.P[i] @ self.F.T + Qi

                # Variance floor
                for j in range(5):
                    if self.P[i, j, j] < 0.1:
                        self.P[i, j, j] = 0.1

                # Symmetrize and Cauchy-Schwarz clamp (mirrors Numba kernel)
                self.P[i] = (self.P[i] + self.P[i].T) * 0.5
                _d5 = np.sqrt(np.maximum(np.diag(self.P[i]), 0.0))
                for _r in range(5):
                    for _c in range(_r + 1, 5):
                        _lim = _d5[_r] * _d5[_c]
                        self.P[i, _r, _c] = np.clip(self.P[i, _r, _c], -_lim, _lim)
                        self.P[i, _c, _r] = self.P[i, _r, _c]

                # Normalize theta to [0, 2*pi)
                self.X[i, 2] = self.X[i, 2] % (2.0 * np.pi)

            # Cap diagonal covariance (mirrors the Numba-path cap above)
            _p_max = float(self.params.get("KALMAN_MAX_COVARIANCE_DIAGONAL", 1000.0))
            _diag = np.arange(self.dim_s)
            np.clip(self.P[:, _diag, _diag], 0.1, _p_max, out=self.P[:, _diag, _diag])

            # Re-apply Cauchy-Schwarz after the diagonal cap (cap may have lowered
            # a diagonal, making an existing off-diagonal exceed the new bound).
            _d_sqrt = np.sqrt(np.maximum(self.P[:, _diag, _diag], 0.0))  # (N, dim_s)
            for _r in range(self.dim_s):
                for _c in range(_r + 1, self.dim_s):
                    _lim_rc = _d_sqrt[:, _r] * _d_sqrt[:, _c]
                    np.clip(self.P[:, _r, _c], -_lim_rc, _lim_rc, out=self.P[:, _r, _c])
                    self.P[:, _c, _r] = self.P[:, _r, _c]

        # Guard against numerical blow-ups before downstream gating/assignment.
        self._sanitize_all_tracks("post-predict non-finite state/covariance")

        # Apply age-dependent velocity damping AFTER prediction (vectorized).
        # Young tracks have their velocity heavily damped toward zero.
        #
        # NOTE — double-damping for very young tracks:
        # The F-matrix already applies a per-step friction factor ``damp``
        # (KALMAN_DAMPING, default 0.95).  Young tracks then receive an
        # *additional* age-based retention factor ``vr`` here (ranging from
        # KALMAN_INITIAL_VELOCITY_RETENTION at age=0 up to 1.0 at maturity).
        # For a brand-new track the effective per-step velocity retention is
        #   damp × initial_velocity_retention   (e.g. 0.95 × 0.2 = 0.19).
        # This is intentional: strong damping prevents absurd velocity
        # extrapolation on the first few frames when the KF has no reliable
        # motion history.
        ages = self.track_ages[: self.num_targets].astype(np.float32)
        young_mask = ages < self.age_threshold
        if np.any(young_mask):
            age_ratio = ages[young_mask] / float(self.age_threshold)
            vr = (
                np.float32(self.initial_velocity_retention)
                + (np.float32(1.0) - np.float32(self.initial_velocity_retention))
                * age_ratio
            )  # shape (n_young,)
            # Damp velocity state
            self.X[young_mask, 3] *= vr
            self.X[young_mask, 4] *= vr
            # Propagate through covariance: rows/cols 3,4
            vr2d = vr[:, None]  # (n_young, 1) for broadcasting
            self.P[young_mask, 3, :] *= vr2d
            self.P[young_mask, 4, :] *= vr2d
            self.P[young_mask, :, 3] *= vr2d
            self.P[young_mask, :, 4] *= vr2d

        # Vectorized maximum velocity constraint.
        vx = self.X[: self.num_targets, 3]
        vy = self.X[: self.num_targets, 4]
        speed = np.sqrt(vx**2 + vy**2)
        over_mask = speed > self.max_velocity
        if np.any(over_mask):
            scale = np.where(
                over_mask, self.max_velocity / np.maximum(speed, 1e-9), 1.0
            ).astype(np.float32)
            self.X[: self.num_targets, 3] *= scale
            self.X[: self.num_targets, 4] *= scale
            scale2d = scale[:, None]
            self.P[: self.num_targets, 3, :] *= scale2d
            self.P[: self.num_targets, 4, :] *= scale2d
            self.P[: self.num_targets, :, 3] *= scale2d
            self.P[: self.num_targets, :, 4] *= scale2d

        return self.X[:, :3].copy()

    def get_predictions(self) -> np.ndarray:
        """Compatibility wrapper returning `predict()` output."""
        return self.predict()

    def correct(
        self, track_idx: int, measurement: np.ndarray, theta_r_scale: float = 1.0
    ) -> None:
        """Correct a track with one measurement update.

        Parameters
        ----------
        theta_r_scale : float
            Multiplier applied to R[2,2] (theta measurement noise) for this
            correction.  Values > 1 make the filter trust its own heading
            prediction more than the measurement; used when heading confidence
            is low.
        """
        R_eff = self.R
        if theta_r_scale != 1.0:
            R_eff = self.R.copy()
            R_eff[2, 2] *= theta_r_scale
        if NUMBA_AVAILABLE:
            try:
                self.X, self.P = _correct_kernel(
                    self.X,
                    self.P,
                    self.H,
                    R_eff,
                    self.identity_mat,
                    track_idx,
                    measurement,
                    self.max_velocity,
                )
            except np.linalg.LinAlgError:
                # Numba's linalg.inv raises LinAlgError (not just silent NaN) when
                # fastmath=False and S contains non-finite values.  Reset and continue.
                self._reset_corrupted_track(
                    track_idx,
                    "non-finite matrix in S during correction (numba path)",
                )
                return
            if not (
                np.isfinite(self.X[track_idx]).all()
                and np.isfinite(self.P[track_idx]).all()
            ):
                self._reset_corrupted_track(
                    track_idx,
                    "post-correct non-finite state/covariance (numba path)",
                )
        else:
            # Manual fallback with Theta-Wrap logic
            z = measurement.reshape(3, 1)
            x = self.X[track_idx].reshape(5, 1)
            p = self.P[track_idx]
            y = z - (self.H @ x)
            if y[2, 0] > np.pi:
                y[2, 0] -= 2 * np.pi
            elif y[2, 0] < -np.pi:
                y[2, 0] += 2 * np.pi
            p = (p + p.T) * 0.5  # symmetrize before use
            S = self.H @ p @ self.H.T + R_eff
            S += np.eye(self.dim_m, dtype=np.float32) * 0.1
            K = p @ self.H.T @ np.linalg.inv(S)
            # Innovation clipping with K_eff for consistent covariance update
            pos_innov_sq = float(y[0, 0] ** 2 + y[1, 0] ** 2)
            clip_scale = 1.0
            max_v = max(float(self.max_velocity), 0.0)
            if pos_innov_sq > max_v**2:
                clip_scale = max_v / max(np.sqrt(pos_innov_sq), 1e-9)
                y[0, 0] *= clip_scale
                y[1, 0] *= clip_scale
            K_eff = K.copy()
            if clip_scale < 1.0:
                K_eff[:, 0] *= clip_scale
                K_eff[:, 1] *= clip_scale
            self.X[track_idx] = (x + (K @ y)).flatten()
            # Normalize theta to [0, 2*pi) after correction
            self.X[track_idx, 2] = self.X[track_idx, 2] % (2.0 * np.pi)
            IKH = self.identity_mat - (K_eff @ self.H)
            self.P[track_idx] = IKH @ p @ IKH.T + (K_eff @ R_eff @ K_eff.T)
            self.P[track_idx] = (self.P[track_idx] + self.P[track_idx].T) * 0.5
            # Velocity constraint + covariance propagation
            vx, vy = self.X[track_idx, 3], self.X[track_idx, 4]
            speed = np.sqrt(vx**2 + vy**2)
            if speed > max_v:
                vel_scale = max_v / max(speed, 1e-9)
                self.X[track_idx, 3] *= vel_scale
                self.X[track_idx, 4] *= vel_scale
                self.P[track_idx, 3, :] *= vel_scale
                self.P[track_idx, 4, :] *= vel_scale
                self.P[track_idx, :, 3] *= vel_scale
                self.P[track_idx, :, 4] *= vel_scale

            if not (
                np.isfinite(self.X[track_idx]).all()
                and np.isfinite(self.P[track_idx]).all()
            ):
                self._reset_corrupted_track(
                    track_idx,
                    "post-correct non-finite state/covariance (python path)",
                )

        # Increment track age after successful update
        self.track_ages[track_idx] += 1

    def get_mahalanobis_matrices(self) -> np.ndarray:
        """Return inverse innovation covariance matrices used by assignment."""
        # Use a robust Python path to avoid propagating invalid inverses.
        self._sanitize_all_tracks("pre-mahal non-finite state/covariance")
        s_inv = np.zeros((self.num_targets, self.dim_m, self.dim_m), dtype=np.float32)
        default_s = self.H @ self.init_P @ self.H.T + self.R
        default_s += np.eye(self.dim_m, dtype=np.float32) * 1e-6
        default_inv = np.linalg.inv(default_s).astype(np.float32)
        eye_m = np.eye(self.dim_m, dtype=np.float32)
        for i in range(self.num_targets):
            S = (self.H @ self.P[i] @ self.H.T + self.R).astype(np.float32)
            if not np.isfinite(S).all():
                self._reset_corrupted_track(i, "non-finite innovation covariance S")
                s_inv[i] = default_inv
                continue
            try:
                inv_i = np.linalg.inv(S)
                if not np.isfinite(inv_i).all():
                    raise np.linalg.LinAlgError("non-finite inverse")
                s_inv[i] = inv_i.astype(np.float32)
            except np.linalg.LinAlgError:
                # Near-singular S: regularize and retry, then fall back.
                S_reg = S + eye_m * np.float32(1e-6)
                try:
                    inv_i = np.linalg.inv(S_reg)
                    if not np.isfinite(inv_i).all():
                        raise np.linalg.LinAlgError("non-finite inverse after jitter")
                    s_inv[i] = inv_i.astype(np.float32)
                except np.linalg.LinAlgError:
                    self._reset_corrupted_track(
                        i,
                        "singular innovation covariance S during mahal inversion",
                    )
                    s_inv[i] = default_inv
        return s_inv

    def get_position_uncertainties(self) -> list[float]:
        """Return per-track positional uncertainty summary values."""
        return np.trace(self.P[:, :2, :2], axis1=1, axis2=2).tolist()
