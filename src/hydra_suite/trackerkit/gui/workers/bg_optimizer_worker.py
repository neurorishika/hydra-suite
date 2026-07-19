"""Qt wrappers for the pure bg-sub optimizer (which lives Qt-free in
core/background/optimizer). Translates the pure functions' progress/frame/stop
callbacks into Qt signals. Keeps the exact interface the bg_parameter_helper
dialog depends on."""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from hydra_suite.core.background.optimizer import (
    generate_bg_previews,
    run_bg_optimization,
)

logger = logging.getLogger(__name__)


class BgSubtractionOptimizer(QThread):
    progress_signal = Signal(int, str)
    result_signal = Signal(list)
    finished_signal = Signal()

    def __init__(
        self,
        video_path,
        base_params,
        tuning_config,
        scoring_weights,
        n_trials,
        n_sample_frames,
        sampler_type,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.base_params = base_params
        self.tuning_config = tuning_config
        self.scoring_weights = scoring_weights
        self.n_trials = n_trials
        self.n_sample_frames = n_sample_frames
        self.sampler_type = sampler_type
        self._stop_requested = False
        # Frame cache the dialog reads (via _preview_cache_kwargs) to hand off to
        # the preview worker. Populated on completion.
        self._cached_prime_frames = None
        self._cached_sample_frames = None
        self._cached_sample_indices = None
        self._cached_roi_mask = None

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            run = run_bg_optimization(
                self.video_path,
                self.base_params,
                self.tuning_config,
                self.scoring_weights,
                self.n_trials,
                self.n_sample_frames,
                self.sampler_type,
                progress_cb=lambda pct, msg="": self.progress_signal.emit(
                    int(pct), msg
                ),
                stop_check=lambda: self._stop_requested,
            )
            self._cached_prime_frames = run.prime_frames
            self._cached_sample_frames = run.sample_frames
            self._cached_sample_indices = run.sample_indices
            self._cached_roi_mask = run.roi_mask
            self.result_signal.emit(run.results)
        except Exception:  # noqa: BLE001 - surface via progress, mirror old behavior
            logger.exception("BgSubtractionOptimizer failed")
            self.progress_signal.emit(0, "Optimization failed.")
        finally:
            self.finished_signal.emit()


class BgDetectionPreviewWorker(QThread):
    frame_signal = Signal(int, object)
    finished_signal = Signal()

    def __init__(
        self,
        video_path,
        base_params,
        trial_params,
        n_sample_frames,
        cached_prime_frames=None,
        cached_sample_frames=None,
        cached_sample_indices=None,
        roi_mask=None,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.base_params = base_params
        self.trial_params = trial_params
        self.n_sample_frames = n_sample_frames
        self._cached_prime_frames = cached_prime_frames
        self._cached_sample_frames = cached_sample_frames
        self._cached_sample_indices = cached_sample_indices
        self._roi_mask = roi_mask
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            generate_bg_previews(
                self.video_path,
                self.base_params,
                self.trial_params,
                self.n_sample_frames,
                prime_frames=self._cached_prime_frames,
                sample_frames=self._cached_sample_frames,
                sample_indices=self._cached_sample_indices,
                roi_mask=self._roi_mask,
                frame_cb=lambda idx, rgb: self.frame_signal.emit(int(idx), rgb),
                stop_check=lambda: self._stop_requested,
            )
        except Exception:  # noqa: BLE001
            logger.exception("BgDetectionPreviewWorker failed")
        finally:
            self.finished_signal.emit()
