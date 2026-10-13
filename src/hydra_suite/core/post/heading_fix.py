"""Tuning for the directed-heading fix in post-processing.

Kept free of pandas/scipy so config builders can read the parameter keys
without importing the post-processing module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadingFixSettings:
    """Tuning for :func:`_fix_heading_globally`, in seconds and body sizes.

    Durations convert to frames with ``fps``; with no frame rate the per-frame
    fallbacks in brackets apply. The defaults describe walking animals, not any
    one dataset; override them per job with the ``heading_fix_*`` config keys.

    * ``override_seconds`` [12 frames]: a run of head calls that disagrees with
      its surroundings and is shorter than this is corrected; longer runs are
      treated as real evidence.
    * ``reversal_fold_seconds`` [50 frames]: an instant reversal (see below)
      shorter than this is folded into its surroundings even when its own head
      calls agree, because head-tail and pose can falter together on curled,
      grooming or climbing animals.
    * ``max_turn_deg_s``: a heading change between adjacent frames larger than
      max(120 deg, max_turn_deg_s / fps) is an instant reversal. The rule is
      off once that reaches 180 deg (very low frame rates).
    * ``reversal_gap_seconds`` [2 frames]: frames this close still count as
      adjacent across missing head calls.
    * ``join_speed_body_per_s`` [1 body size per frame]: a step faster than
      this is a track join onto another animal, where a reversal is allowed.
    """

    body_size_px: float | None = None
    fps: float | None = None
    override_seconds: float = 1.2
    reversal_fold_seconds: float = 5.0
    max_turn_deg_s: float = 720.0
    reversal_gap_seconds: float = 0.2
    join_speed_body_per_s: float = 10.0

    # Motion counts as evidence once the animal covers a quarter body size over
    # a +-0.2 s window (+-2 frames without a frame rate), i.e. it is walking.
    MOTION_MIN_BODY_FRAC = 0.25
    MOTION_HALF_WINDOW_S = 0.2
    # Motion costs this fraction of the head-call cost: it never overrules
    # sustained head calls (animals do walk backwards), it only tips short runs.
    MOTION_COST_RATIO = 0.6
    REVERSAL_MIN_RAD = 2.0 * np.pi / 3.0

    _PARAM_KEYS = {
        "override_seconds": "HEADING_FIX_OVERRIDE_SECONDS",
        "reversal_fold_seconds": "HEADING_FIX_REVERSAL_FOLD_SECONDS",
        "max_turn_deg_s": "HEADING_FIX_MAX_TURN_DEG_S",
        "reversal_gap_seconds": "HEADING_FIX_REVERSAL_GAP_SECONDS",
        "join_speed_body_per_s": "HEADING_FIX_JOIN_SPEED_BODY_PER_S",
    }

    @classmethod
    def from_params(cls, params, resize_factor: float = 1.0) -> "HeadingFixSettings":
        """Build from a tracking params dict.

        ``REFERENCE_BODY_SIZE`` is in source-video pixels; trajectories before
        the final rescale live in the resized frame, hence ``resize_factor``.
        """
        params = params or {}

        def _positive(key):
            try:
                value = float(params.get(key) or 0.0)
            except (TypeError, ValueError):
                return None
            return value if value > 0 else None

        body = _positive("REFERENCE_BODY_SIZE")
        overrides = {
            field: value
            for field, key in cls._PARAM_KEYS.items()
            if (value := _positive(key)) is not None
        }
        return cls(
            body_size_px=body * float(resize_factor or 1.0) if body else None,
            fps=_positive("FPS"),
            **overrides,
        )

    def _frames(self, seconds: float, fallback: int) -> int:
        if not self.fps:
            return fallback
        return max(1, int(round(seconds * self.fps)))

    @property
    def obs_cost(self) -> float:
        """Per-frame cost of overriding a frame's own head call."""
        return 2.0 * np.pi / self._frames(self.override_seconds, 12)

    @property
    def motion_cost(self) -> float:
        return self.MOTION_COST_RATIO * self.obs_cost

    @property
    def reversal_cost(self) -> float:
        # Folding a reversed run of L frames costs L * obs_cost; keeping it
        # costs two reversals of ~pi + reversal_cost each.
        fold = self._frames(self.reversal_fold_seconds, 50)
        return max(0.0, (fold * self.obs_cost - 2.0 * np.pi) / 2.0)

    @property
    def reversal_rad(self) -> float:
        if not self.fps:
            return self.REVERSAL_MIN_RAD
        return max(self.REVERSAL_MIN_RAD, np.radians(self.max_turn_deg_s) / self.fps)

    @property
    def reversal_max_gap(self) -> int:
        return self._frames(self.reversal_gap_seconds, 2)

    @property
    def join_step_px(self) -> float | None:
        if not self.body_size_px:
            return None
        per_frame = self.join_speed_body_per_s / self.fps if self.fps else 1.0
        return per_frame * self.body_size_px

    @property
    def motion_half_window(self) -> int:
        return self._frames(self.MOTION_HALF_WINDOW_S, 2)
