"""Optuna-based optimizer for background-subtraction detection parameters.

Samples frames from the video, primes the background model once, then
runs Bayesian optimisation over threshold / morphology / split parameters
with three self-contained objectives that avoid the circular dependency
between detection params and REFERENCE_BODY_SIZE:

1. **Count accuracy** – fraction of sampled frames where the number of
   detections equals ``MAX_TARGETS``.
2. **Size consistency** – within each frame that has the correct count,
   how uniform are the detection areas? (1 − coefficient of variation).
3. **Size stability** – across frames, how stable is the median detection
   area? (1 − CoV of per-frame medians).

None of these require REFERENCE_BODY_SIZE as input, so the optimizer
can run *before* body-size calibration.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _expected_body_area(params: Dict[str, Any]) -> float:
    """Return the one-animal area implied by the current body-size settings."""
    reference_body_size = float(params.get("REFERENCE_BODY_SIZE", 0.0) or 0.0)
    resize_factor = float(params.get("RESIZE_FACTOR", 1.0) or 1.0)
    if reference_body_size <= 0.0 or resize_factor <= 0.0:
        return 0.0
    return float(np.pi * (reference_body_size / 2.0) ** 2 * (resize_factor**2))


def _prime_frame_search_upper_bound(params: Dict[str, Any], total_frames: int) -> int:
    """Choose a practical upper bound when tuning background priming frames."""
    if total_frames <= 0:
        return 0
    current = max(0, int(params.get("BACKGROUND_PRIME_FRAMES", 30) or 0))
    return min(total_frames, max(current * 4, current + 24, 120))


def _build_prime_frame_indices(total_frames: int, max_prime_frames: int) -> np.ndarray:
    """Build deterministic prime-frame indices for repeatable background priming."""
    if total_frames <= 0 or max_prime_frames <= 0:
        return np.array([], dtype=int)
    if max_prime_frames >= total_frames:
        return np.arange(total_frames, dtype=int)
    return np.unique(np.linspace(0, total_frames - 1, max_prime_frames, dtype=int))


def _suggest_object_size_params(trial, tune, params):
    """Suggest size-filter bounds while preserving the GUI's multiplier semantics."""
    body_area = _expected_body_area(params)
    current_min_px = float(
        params.get("MIN_OBJECT_SIZE", max(body_area * 0.3, 1.0))
        or max(body_area * 0.3, 1.0)
    )
    current_max_px = float(
        params.get("MAX_OBJECT_SIZE", max(body_area * 3.0, current_min_px + 1.0))
        or max(body_area * 3.0, current_min_px + 1.0)
    )

    if body_area > 0.0:
        current_min_mult = max(0.05, current_min_px / body_area)
        current_max_mult = max(current_min_mult + 0.05, current_max_px / body_area)

        if "MIN_OBJECT_SIZE" in tune:
            min_mult = trial.suggest_float("MIN_OBJECT_SIZE_MULTIPLIER", 0.05, 2.0)
        else:
            min_mult = current_min_mult

        if "MAX_OBJECT_SIZE" in tune:
            max_mult = trial.suggest_float(
                "MAX_OBJECT_SIZE_MULTIPLIER",
                max(min_mult + 0.05, 0.5),
                10.0,
            )
        else:
            max_mult = max(current_max_mult, min_mult + 0.05)

        return int(round(min_mult * body_area)), int(round(max_mult * body_area))

    min_low = 1
    min_high = max(min_low + 1, int(round(max(current_min_px * 3.0, 250.0))))
    if "MIN_OBJECT_SIZE" in tune:
        min_px = trial.suggest_int("MIN_OBJECT_SIZE", min_low, min_high)
    else:
        min_px = int(round(current_min_px))

    max_low = max(min_px + 1, int(round(max(current_max_px * 0.5, min_px + 1))))
    max_high = max(max_low + 1, int(round(max(current_max_px * 3.0, 500.0))))
    if "MAX_OBJECT_SIZE" in tune:
        max_px = trial.suggest_int("MAX_OBJECT_SIZE", max_low, max_high)
    else:
        max_px = int(round(max(current_max_px, min_px + 1)))
    return min_px, max_px


def _suggest_trial_params(trial, tune, params, total_frames):
    """Suggest or inherit each BG-sub detection parameter for an Optuna trial."""
    trial_params: Dict[str, Any] = {}

    # Image adjustments / polarity
    if "BRIGHTNESS" in tune:
        trial_params["BRIGHTNESS"] = trial.suggest_int("BRIGHTNESS", -255, 255)
    else:
        trial_params["BRIGHTNESS"] = int(params.get("BRIGHTNESS", 0))

    if "CONTRAST" in tune:
        trial_params["CONTRAST"] = trial.suggest_float("CONTRAST", 0.1, 3.0)
    else:
        trial_params["CONTRAST"] = float(params.get("CONTRAST", 1.0))

    if "GAMMA" in tune:
        trial_params["GAMMA"] = trial.suggest_float("GAMMA", 0.1, 3.0)
    else:
        trial_params["GAMMA"] = float(params.get("GAMMA", 1.0))

    if "DARK_ON_LIGHT_BACKGROUND" in tune:
        trial_params["DARK_ON_LIGHT_BACKGROUND"] = trial.suggest_categorical(
            "DARK_ON_LIGHT_BACKGROUND",
            [True, False],
        )
    else:
        trial_params["DARK_ON_LIGHT_BACKGROUND"] = bool(
            params.get("DARK_ON_LIGHT_BACKGROUND", True)
        )

    # Background model
    if "BACKGROUND_PRIME_FRAMES" in tune:
        trial_params["BACKGROUND_PRIME_FRAMES"] = trial.suggest_int(
            "BACKGROUND_PRIME_FRAMES",
            0,
            _prime_frame_search_upper_bound(params, total_frames),
        )
    else:
        trial_params["BACKGROUND_PRIME_FRAMES"] = int(
            params.get("BACKGROUND_PRIME_FRAMES", 30)
        )

    if "ENABLE_ADAPTIVE_BACKGROUND" in tune:
        enable_adaptive = trial.suggest_categorical(
            "ENABLE_ADAPTIVE_BACKGROUND",
            [True, False],
        )
    else:
        enable_adaptive = bool(params.get("ENABLE_ADAPTIVE_BACKGROUND", True))
    trial_params["ENABLE_ADAPTIVE_BACKGROUND"] = enable_adaptive

    if "BACKGROUND_LEARNING_RATE" in tune:
        trial_params["BACKGROUND_LEARNING_RATE"] = trial.suggest_float(
            "BACKGROUND_LEARNING_RATE",
            1e-5,
            0.1,
            log=True,
        )
    else:
        trial_params["BACKGROUND_LEARNING_RATE"] = float(
            params.get("BACKGROUND_LEARNING_RATE", 0.001)
        )

    # Lighting stabilization
    if "ENABLE_LIGHTING_STABILIZATION" in tune:
        enable_lighting = trial.suggest_categorical(
            "ENABLE_LIGHTING_STABILIZATION",
            [True, False],
        )
    else:
        enable_lighting = bool(params.get("ENABLE_LIGHTING_STABILIZATION", True))
    trial_params["ENABLE_LIGHTING_STABILIZATION"] = enable_lighting

    if "LIGHTING_SMOOTH_FACTOR" in tune:
        trial_params["LIGHTING_SMOOTH_FACTOR"] = trial.suggest_float(
            "LIGHTING_SMOOTH_FACTOR",
            0.8,
            0.999,
        )
    else:
        trial_params["LIGHTING_SMOOTH_FACTOR"] = float(
            params.get("LIGHTING_SMOOTH_FACTOR", 0.95)
        )

    if "LIGHTING_MEDIAN_WINDOW" in tune:
        median_half = trial.suggest_int("LIGHTING_MEDIAN_HALF", 1, 7)
        trial_params["LIGHTING_MEDIAN_WINDOW"] = median_half * 2 + 1
    else:
        trial_params["LIGHTING_MEDIAN_WINDOW"] = int(
            params.get("LIGHTING_MEDIAN_WINDOW", 5)
        )

    # THRESHOLD_VALUE
    if "THRESHOLD_VALUE" in tune:
        trial_params["THRESHOLD_VALUE"] = trial.suggest_int("THRESHOLD_VALUE", 0, 255)
    else:
        trial_params["THRESHOLD_VALUE"] = params["THRESHOLD_VALUE"]

    # MORPH_KERNEL_SIZE (odd)
    if "MORPH_KERNEL_SIZE" in tune:
        morph_half = trial.suggest_int("MORPH_KERNEL_HALF", 0, 12)
        trial_params["MORPH_KERNEL_SIZE"] = morph_half * 2 + 1
    else:
        trial_params["MORPH_KERNEL_SIZE"] = params["MORPH_KERNEL_SIZE"]

    # MIN_CONTOUR_AREA
    if "MIN_CONTOUR_AREA" in tune:
        trial_params["MIN_CONTOUR_AREA"] = trial.suggest_int(
            "MIN_CONTOUR_AREA", 10, 500
        )
    else:
        trial_params["MIN_CONTOUR_AREA"] = params["MIN_CONTOUR_AREA"]

    if "MAX_CONTOUR_MULTIPLIER" in tune:
        trial_params["MAX_CONTOUR_MULTIPLIER"] = trial.suggest_int(
            "MAX_CONTOUR_MULTIPLIER",
            5,
            100,
        )
    else:
        trial_params["MAX_CONTOUR_MULTIPLIER"] = int(
            params.get("MAX_CONTOUR_MULTIPLIER", 20)
        )

    # ENABLE_SIZE_FILTERING group
    if "ENABLE_SIZE_FILTERING" in tune:
        enable_size_filter = trial.suggest_categorical(
            "ENABLE_SIZE_FILTERING",
            [True, False],
        )
    else:
        enable_size_filter = bool(params.get("ENABLE_SIZE_FILTERING", False))
    trial_params["ENABLE_SIZE_FILTERING"] = enable_size_filter

    min_object_size, max_object_size = _suggest_object_size_params(
        trial,
        tune,
        params,
    )
    trial_params["MIN_OBJECT_SIZE"] = min_object_size
    trial_params["MAX_OBJECT_SIZE"] = max_object_size

    # ENABLE_ADDITIONAL_DILATION group
    if "ENABLE_ADDITIONAL_DILATION" in tune:
        enable_dil = trial.suggest_categorical(
            "ENABLE_ADDITIONAL_DILATION",
            [True, False],
        )
    else:
        enable_dil = params.get("ENABLE_ADDITIONAL_DILATION", False)
    trial_params["ENABLE_ADDITIONAL_DILATION"] = enable_dil

    if "DILATION_KERNEL_SIZE" in tune:
        dil_half = trial.suggest_int("DILATION_KERNEL_HALF", 0, 7)
        trial_params["DILATION_KERNEL_SIZE"] = dil_half * 2 + 1
    else:
        trial_params["DILATION_KERNEL_SIZE"] = params.get("DILATION_KERNEL_SIZE", 3)
    if "DILATION_ITERATIONS" in tune:
        trial_params["DILATION_ITERATIONS"] = trial.suggest_int(
            "DILATION_ITERATIONS",
            1,
            5,
        )
    else:
        trial_params["DILATION_ITERATIONS"] = params.get("DILATION_ITERATIONS", 1)

    # ENABLE_CONSERVATIVE_SPLIT group
    if "ENABLE_CONSERVATIVE_SPLIT" in tune:
        enable_split = trial.suggest_categorical(
            "ENABLE_CONSERVATIVE_SPLIT",
            [True, False],
        )
    else:
        enable_split = params.get("ENABLE_CONSERVATIVE_SPLIT", False)
    trial_params["ENABLE_CONSERVATIVE_SPLIT"] = enable_split

    if "CONSERVATIVE_KERNEL_SIZE" in tune:
        split_half = trial.suggest_int("CONSERVATIVE_KERNEL_HALF", 0, 7)
        trial_params["CONSERVATIVE_KERNEL_SIZE"] = split_half * 2 + 1
    else:
        trial_params["CONSERVATIVE_KERNEL_SIZE"] = params.get(
            "CONSERVATIVE_KERNEL_SIZE",
            3,
        )
    if "CONSERVATIVE_ERODE_ITER" in tune:
        trial_params["CONSERVATIVE_ERODE_ITER"] = trial.suggest_int(
            "CONSERVATIVE_ERODE_ITER",
            1,
            5,
        )
    else:
        trial_params["CONSERVATIVE_ERODE_ITER"] = params.get(
            "CONSERVATIVE_ERODE_ITER",
            1,
        )

    return trial_params


@dataclass
class _BgFrameCache:
    """Video frames cached once so trials can rerun the full BG pipeline cheaply."""

    prime_frames: List[np.ndarray]
    sample_frames: List[np.ndarray]
    sample_indices: List[int]
    roi_mask: Optional[np.ndarray]


def _read_gray_frames(cap, frame_indices, resize_f, stop_check):
    """Read raw grayscale frames once; trial-specific adjustments happen later."""
    frames: List[np.ndarray] = []
    for idx in frame_indices:
        if stop_check():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if resize_f < 1.0:
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=resize_f,
                fy=resize_f,
                interpolation=cv2.INTER_AREA,
            )
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return frames


def _resize_roi_mask(roi_mask, frame_shape):
    """Resize ROI mask once to match the cached frame dimensions."""
    if roi_mask is None or frame_shape is None:
        return roi_mask
    if roi_mask.shape[:2] == frame_shape:
        return roi_mask
    return cv2.resize(
        roi_mask,
        (frame_shape[1], frame_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def _prime_background_from_frames(prime_frames, trial_params, roi_mask):
    """Prime the lightest-pixel background model from cached raw frames."""
    from ...utils.image_processing import apply_image_adjustments
    from .model import BackgroundModel

    if not prime_frames:
        return None, None, None

    brightness = trial_params.get("BRIGHTNESS", 0)
    contrast = trial_params.get("CONTRAST", 1.0)
    gamma = trial_params.get("GAMMA", 1.0)

    bg_temp = None
    intensity_samples: List[float] = []
    for raw_gray in prime_frames:
        gray = apply_image_adjustments(
            raw_gray,
            brightness,
            contrast,
            gamma,
            use_gpu=False,
        )
        pixels = gray[roi_mask > 0] if roi_mask is not None else gray.ravel()
        sample = BackgroundModel._iqr_mean_intensity(pixels)
        if sample is not None:
            intensity_samples.append(sample)

        gray_f32 = gray.astype(np.float32)
        if bg_temp is None:
            bg_temp = gray_f32
        else:
            bg_temp = np.maximum(bg_temp, gray_f32)

    if bg_temp is None:
        return None, None, None

    if intensity_samples:
        reference_intensity = float(np.median(intensity_samples))
    elif roi_mask is not None:
        roi_pixels = bg_temp[roi_mask > 0]
        reference_intensity = (
            float(np.mean(roi_pixels))
            if len(roi_pixels) > 0
            else float(np.mean(bg_temp))
        )
    else:
        reference_intensity = float(np.mean(bg_temp))

    return bg_temp, bg_temp.copy(), reference_intensity


def _init_trial_pipeline(trial_params, frame_cache):
    """Create a fresh BG-sub pipeline state for one trial evaluation."""
    from .measure import BackgroundMeasurer
    from .model import BackgroundModel

    bg_model = BackgroundModel(trial_params)
    prime_count = max(0, int(trial_params.get("BACKGROUND_PRIME_FRAMES", 0) or 0))
    lightest_bg, adaptive_bg, reference_intensity = _prime_background_from_frames(
        frame_cache.prime_frames[:prime_count],
        trial_params,
        frame_cache.roi_mask,
    )
    if lightest_bg is not None:
        bg_model.lightest_background = lightest_bg
        bg_model.adaptive_background = adaptive_bg
        bg_model.reference_intensity = reference_intensity

    return (
        bg_model,
        BackgroundMeasurer(trial_params),
        deque(maxlen=50),
        {},
    )


def _run_bg_trial_frame(
    raw_gray,
    frame_index,
    trial_params,
    bg_model,
    detector,
    roi_mask,
    intensity_history,
    lighting_state,
):
    """Run one cached frame through the full BG-subtraction detection pipeline."""
    from ...utils.image_processing import apply_image_adjustments, stabilize_lighting

    gray = apply_image_adjustments(
        raw_gray,
        trial_params.get("BRIGHTNESS", 0),
        trial_params.get("CONTRAST", 1.0),
        trial_params.get("GAMMA", 1.0),
        use_gpu=False,
    )

    if trial_params.get("ENABLE_LIGHTING_STABILIZATION", True):
        gray, intensity_history, _ = stabilize_lighting(
            gray,
            bg_model.reference_intensity,
            intensity_history,
            trial_params.get("LIGHTING_SMOOTH_FACTOR", 0.95),
            roi_mask,
            trial_params.get("LIGHTING_MEDIAN_WINDOW", 5),
            lighting_state,
            use_gpu=False,
        )

    bg_u8 = bg_model.update_and_get_background(
        gray,
        roi_mask,
    )
    if bg_u8 is None:
        return gray, None, None, [], [], []

    fg_mask = bg_model.generate_foreground_mask(gray, bg_u8)
    if roi_mask is not None:
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)
    if trial_params.get("ENABLE_CONSERVATIVE_SPLIT", True):
        fg_mask = detector.apply_conservative_split(fg_mask, gray, bg_u8)

    meas, sizes, shapes, _conf = detector.detect_objects(fg_mask, frame_index)
    return gray, bg_u8, fg_mask, meas, sizes, shapes


def _aggregate_trial_scores(count_scores, consistency_scores, frame_medians):
    """Compute count/consistency/stability sub-scores from per-frame data."""
    s_count = float(np.mean(count_scores)) if count_scores else 0.0
    s_consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0
    if len(frame_medians) >= 2:
        med_arr = np.array(frame_medians)
        med_mean = med_arr.mean()
        if med_mean > 1e-6:
            s_stability = max(0.0, 1.0 - med_arr.std() / med_mean)
        else:
            s_stability = 0.0
    elif frame_medians:
        s_stability = 1.0
    else:
        s_stability = 0.0
    return s_count, s_consistency, s_stability


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BgOptimizationResult:
    """One completed trial from the BG optimiser."""

    params: Dict[str, Any]
    score: float
    trial_number: int
    sub_scores: Dict[str, float] = field(default_factory=dict)
    median_area: float = 0.0  # median detection area in pixels²


@dataclass
class BgOptimizationRun:
    """Result of one full `run_bg_optimization` call.

    Bundles the completed trial results together with the cached raw
    frames (frame cache) so a preview step can reuse the exact same
    sample set without re-reading the video.
    """

    results: List[BgOptimizationResult]
    prime_frames: List[np.ndarray]
    sample_frames: List[np.ndarray]
    sample_indices: List[int]
    roi_mask: Optional[np.ndarray]


def _build_sampler(sampler_type: str, n_active: int):
    """Construct the Optuna sampler based on *sampler_type*.

    "auto" — OptunaHub AutoSampler (GP early, TPE later).
    "gp"   — GPSampler (Matérn-2.5, ARD, log-EI).
    "tpe"  — Multivariate TPE (robust fallback).
    """
    stype = sampler_type

    if stype == "auto":
        try:
            import optunahub  # type: ignore[import-untyped]

            return optunahub.load_module(
                "samplers/auto_sampler",
            ).AutoSampler()
        except Exception as e:
            logger.warning(
                "AutoSampler unavailable (%s), falling back to GP.",
                e,
            )
            stype = "gp"

    if stype == "gp":
        try:
            qmc = optuna.samplers.QMCSampler(
                qmc_type="sobol",
                seed=42,
                independent_sampler=optuna.samplers.RandomSampler(seed=42),
            )
            return optuna.samplers.GPSampler(
                seed=42,
                n_startup_trials=max(10, n_active),
                deterministic_objective=True,
                independent_sampler=qmc,
            )
        except Exception as e:
            logger.warning(
                "GPSampler unavailable (%s), falling back to TPE.",
                e,
            )

    # "tpe" (default / fallback)
    return optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=max(20, n_active * 2),
        seed=42,
    )


def run_bg_optimization(
    video_path: str,
    base_params: Dict[str, Any],
    tuning_config: Dict[str, bool],
    scoring_weights: Dict[str, float],
    n_trials: int,
    n_sample_frames: int,
    sampler_type: str,
    *,
    progress_cb: "Callable[[int, str], None] | None" = None,
    stop_check: "Callable[[], bool] | None" = None,
) -> BgOptimizationRun:
    """Run the Optuna BG-subtraction parameter search (Qt-free, pure core).

    Mirrors what ``BgSubtractionOptimizer._run_optimization`` used to do as a
    QThread method, but reports progress via ``progress_cb`` and honors
    ``stop_check`` instead of emitting Qt signals / reading ``self._stop_requested``.
    """

    def _stopped() -> bool:
        return stop_check() if stop_check is not None else False

    def _report(pct, msg):
        if progress_cb is not None:
            progress_cb(int(pct), msg)

    empty_run = BgOptimizationRun(
        results=[],
        prime_frames=[],
        sample_frames=[],
        sample_indices=[],
        roi_mask=None,
    )

    if optuna is None:
        _report(0, "Error: optuna is not installed")
        return empty_run

    params = base_params

    # --- 1. open video -------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _report(0, "Error: cannot open video")
        return empty_run

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = params.get("START_FRAME", 0)
    end = min(params.get("END_FRAME", total_frames - 1), total_frames - 1)
    max_prime_frames = _prime_frame_search_upper_bound(params, total_frames)

    # --- 2. choose sample frames --------------------------------------
    base_prime_frames = int(params.get("BACKGROUND_PRIME_FRAMES", 30) or 0)
    first_valid = start + base_prime_frames
    if first_valid >= end:
        first_valid = start
    n_available = end - first_valid + 1
    n_sample = min(n_sample_frames, max(n_available, 1))
    sample_indices = np.linspace(first_valid, end, n_sample, dtype=int).tolist()
    prime_indices = _build_prime_frame_indices(total_frames, max_prime_frames).tolist()

    resize_f = params.get("RESIZE_FACTOR", 1.0)

    # --- 3. cache raw frames once --------------------------------------
    _report(5, "Caching optimization frames …")
    prime_frames = _read_gray_frames(
        cap,
        prime_indices,
        resize_f,
        _stopped,
    )
    if prime_frames is None:
        return empty_run
    sample_frames = _read_gray_frames(
        cap,
        sample_indices,
        resize_f,
        _stopped,
    )
    cap.release()

    if not sample_frames:
        _report(0, "Error: no frames could be read")
        return empty_run

    # --- 4. scoring setup -----------------------------------------------
    max_targets = params.get("MAX_TARGETS", 5)

    # Pre-resize ROI mask
    roi_mask = _resize_roi_mask(
        params.get("ROI_MASK"),
        sample_frames[0].shape if sample_frames else None,
    )
    frame_cache = _BgFrameCache(
        prime_frames=prime_frames,
        sample_frames=sample_frames,
        sample_indices=list(sample_indices),
        roi_mask=roi_mask,
    )

    # --- 5. run Optuna -------------------------------------------------
    _report(10, "Starting optimisation …")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_active = sum(1 for v in tuning_config.values() if v)
    n_frames = len(sample_frames)

    # Pruning: report intermediate count scores and let Optuna kill
    # obviously-bad trials early (e.g. wrong threshold → 0 detections).
    # Trials report every PRUNE_INTERVAL frames; the pruner kicks in
    # after n_warmup_steps reports and n_startup_trials completed trials.
    _PRUNE_INTERVAL = max(1, n_frames // 6)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=_build_sampler(sampler_type, n_active),
        pruner=pruner,
    )
    results: List[BgOptimizationResult] = []

    # Normalise scoring weights
    sw = scoring_weights
    w_cnt_r = sw.get("SCORE_WEIGHT_COUNT", 0.50)
    w_con_r = sw.get("SCORE_WEIGHT_CONSISTENCY", 0.30)
    w_stb_r = sw.get("SCORE_WEIGHT_STABILITY", 0.20)
    w_sum = w_cnt_r + w_con_r + w_stb_r
    if w_sum < 1e-9:
        w_cnt, w_con, w_stb = 1 / 3, 1 / 3, 1 / 3
    else:
        w_cnt = w_cnt_r / w_sum
        w_con = w_con_r / w_sum
        w_stb = w_stb_r / w_sum

    # Filter tuning_config to only enabled params
    tune = {k: v for k, v in tuning_config.items() if v}

    def objective(trial):
        if _stopped():
            raise optuna.TrialPruned()

        trial_params = _suggest_trial_params(trial, tune, params, total_frames)

        det_params = dict(params)
        det_params.update(trial_params)
        bg_model, detector, intensity_history, lighting_state = _init_trial_pipeline(
            det_params,
            frame_cache,
        )

        count_scores: List[float] = []
        consistency_scores: List[float] = []
        frame_medians: List[float] = []
        prune_step = 0

        for fi, raw_gray in enumerate(frame_cache.sample_frames):
            if _stopped():
                raise optuna.TrialPruned()

            _gray, _bg_u8, _fg_mask, meas, sizes, _shapes = _run_bg_trial_frame(
                raw_gray,
                frame_cache.sample_indices[fi],
                det_params,
                bg_model,
                detector,
                frame_cache.roi_mask,
                intensity_history,
                lighting_state,
            )
            n_det = len(meas)

            if max_targets > 0:
                err = abs(n_det - max_targets) / max_targets
                count_scores.append(max(0.0, 1.0 - err))

            if sizes and n_det == max_targets and n_det >= 2:
                areas = np.array(sizes, dtype=float)
                mean_a = areas.mean()
                if mean_a > 1e-6:
                    cov = areas.std() / mean_a
                    consistency_scores.append(max(0.0, 1.0 - cov))
                else:
                    consistency_scores.append(0.0)
                frame_medians.append(float(np.median(areas)))
            elif sizes:
                frame_medians.append(float(np.median(sizes)))

            if (fi + 1) % _PRUNE_INTERVAL == 0 or fi == n_frames - 1:
                running = float(np.mean(count_scores)) if count_scores else 0.0
                trial.report(running, step=prune_step)
                prune_step += 1
                if trial.should_prune():
                    raise optuna.TrialPruned()

        s_count, s_consistency, s_stability = _aggregate_trial_scores(
            count_scores,
            consistency_scores,
            frame_medians,
        )
        score = w_cnt * s_count + w_con * s_consistency + w_stb * s_stability

        trial_median_area = float(np.median(frame_medians)) if frame_medians else 0.0

        results.append(
            BgOptimizationResult(
                params=trial_params,
                score=score,
                trial_number=trial.number,
                sub_scores={
                    "count": s_count,
                    "consistency": s_consistency,
                    "stability": s_stability,
                },
                median_area=trial_median_area,
            )
        )

        pct = int(10 + 85 * (trial.number + 1) / n_trials)
        _report(
            pct,
            f"Trial {trial.number + 1}/{n_trials} — "
            f"score {score:.3f}  "
            f"(cnt {s_count:.2f}, con {s_consistency:.2f}, "
            f"stb {s_stability:.2f})",
        )
        return score

    study.optimize(objective, n_trials=n_trials)

    results.sort(key=lambda r: -r.score)
    _report(100, "Optimisation complete!")

    return BgOptimizationRun(
        results=results,
        prime_frames=list(frame_cache.prime_frames),
        sample_frames=list(frame_cache.sample_frames),
        sample_indices=list(frame_cache.sample_indices),
        roi_mask=(
            frame_cache.roi_mask.copy() if frame_cache.roi_mask is not None else None
        ),
    )


def generate_bg_previews(
    video_path: str,
    base_params: Dict[str, Any],
    trial_params: Dict[str, Any],
    n_sample_frames: int,
    *,
    prime_frames: Optional[List[np.ndarray]] = None,
    sample_frames: Optional[List[np.ndarray]] = None,
    sample_indices: Optional[List[int]] = None,
    roi_mask: Optional[np.ndarray] = None,
    frame_cb: "Callable[[int, np.ndarray], None] | None" = None,
    stop_check: "Callable[[], bool] | None" = None,
) -> None:
    """Generate annotated preview frames for a given detection parameter set (Qt-free, pure core).

    Reads the same sample frames as the optimiser (or the ones passed in via
    ``prime_frames``/``sample_frames``/``sample_indices``/``roi_mask``), runs the
    full BG-sub detection pipeline for the chosen parameters, and reports
    (index, RGB) pairs via ``frame_cb`` instead of emitting Qt signals.
    """

    def _stopped() -> bool:
        return stop_check() if stop_check is not None else False

    params = base_params
    det_params = dict(params)
    det_params.update(trial_params)

    if sample_frames is None or sample_indices is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = params.get("START_FRAME", 0)
        end = min(params.get("END_FRAME", total_frames - 1), total_frames - 1)
        prime_n = int(params.get("BACKGROUND_PRIME_FRAMES", 30) or 0)
        first_valid = start + prime_n
        if first_valid >= end:
            first_valid = start
        n_available = end - first_valid + 1
        n_sample = min(n_sample_frames, max(n_available, 1))
        sample_indices = np.linspace(first_valid, end, n_sample, dtype=int).tolist()
        resize_f = params.get("RESIZE_FACTOR", 1.0)
        sample_frames = _read_gray_frames(
            cap,
            sample_indices,
            resize_f,
            _stopped,
        )
        max_prime_frames = _prime_frame_search_upper_bound(params, total_frames)
        prime_indices = _build_prime_frame_indices(
            total_frames, max_prime_frames
        ).tolist()
        prime_frames = _read_gray_frames(
            cap,
            prime_indices,
            resize_f,
            _stopped,
        )
        cap.release()
        if sample_frames is None or not sample_frames:
            return
        roi_mask = _resize_roi_mask(
            params.get("ROI_MASK"),
            sample_frames[0].shape if sample_frames else None,
        )

    frame_cache = _BgFrameCache(
        prime_frames=prime_frames or [],
        sample_frames=sample_frames,
        sample_indices=sample_indices,
        roi_mask=roi_mask,
    )
    bg_model, detector, intensity_history, lighting_state = _init_trial_pipeline(
        det_params,
        frame_cache,
    )

    for fi, idx in enumerate(frame_cache.sample_indices):
        if _stopped():
            break
        raw_gray = frame_cache.sample_frames[fi]
        gray, _bg_u8, _fg, meas, sizes, shapes = _run_bg_trial_frame(
            raw_gray,
            idx,
            det_params,
            bg_model,
            detector,
            frame_cache.roi_mask,
            intensity_history,
            lighting_state,
        )
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # draw detections on frame
        for j, (m, sz) in enumerate(zip(meas, sizes)):
            cx, cy = int(m[0]), int(m[1])
            radius = max(int((sz / 3.14159) ** 0.5), 3)
            cv2.circle(display, (cx, cy), radius, (0, 255, 0), 2)
            cv2.putText(
                display,
                str(j + 1),
                (cx + radius + 2, cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        # count overlay
        max_t = params.get("MAX_TARGETS", 5)
        n_det = len(meas)
        clr = (
            (0, 255, 0)
            if n_det == max_t
            else ((0, 200, 255) if n_det < max_t else (0, 0, 255))
        )
        cv2.putText(
            display,
            f"Frame {int(idx)}  |  {n_det}/{max_t} detections",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            clr,
            1,
            cv2.LINE_AA,
        )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        if frame_cb is not None:
            frame_cb(int(fi), rgb)
