"""
Tracking pipeline profiler — collects per-frame timing data and exports a
complete summary at the end of a tracking run.

Covers every pipeline phase:

* Initialization (video, detector, background model, Kalman, assigner)
* Batched detection
* Confidence density map
* Unified precompute (pose, AprilTag, CNN identity, crop extraction)
* Main tracking loop (preprocessing, detection, features, assignment, …)
* Cleanup
* Post-processing (resolve, interpolate, tag identity, rescale)

The profiler is **disabled by default** (zero overhead).  Enable it via the
"Enable performance profiling" checkbox in Get Started → Debug, or by passing
``enabled=True`` to the constructor.

Usage inside ``TrackingWorker.run()``:

    profiler = TrackingProfiler(enabled=params.get("ENABLE_PROFILING", False))

    profiler.phase_start("initialization")
    ...
    profiler.phase_end("initialization")

    for frame in frames:
        profiler.tick("preprocessing")
        ...
        profiler.tock("preprocessing")
        profiler.end_frame()
        profiler.log_periodic(100)

    profiler.export_summary("/path/to/profile.json")
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Canonical ordering for display / export.
CATEGORY_ORDER = [
    # Per-frame tracking loop categories
    "frame_read",
    "preprocessing",
    "frame_resize",
    "roi_prepare",
    "detection",
    "live_filter",
    "live_crop_extraction",
    "live_pose_transport",
    "live_pose_inference",
    "live_pose_postprocess",
    "live_apriltag",
    "live_cnn_identity",
    "features",
    "kf_predict",
    "cost_matrix",
    "hungarian",
    "state_update",
    "kf_update",
    "csv_write",
    "individual_dataset",
    "visualization",
    "video_write",
    "gui_emit",
    "other",
    # InterpolatedCropsWorker categories
    "interp_pose_inference",
    # Batched detection diagnostic categories
    "batched_frame_read",
    "batched_infer",          # detector.detect_objects_batched() wall time
    "batched_cache_write",   # _cache_batch_results() wall time
]


# Canonical ordering for pipeline phases (coarse timing).
PHASE_ORDER = [
    "initialization",
    "batched_detection",
    # Sub-phases within batched_detection (engine-level)
    "yolo_obb_inference",
    "yolo_obb_model_execute",
    "yolo_obb_extract_raw",
    "sequential_obb_crop",
    "sequential_obb_inference",
    "headtail_crop",
    "headtail_inference",
    "confidence_density",
    "precompute",
    "precompute_frame_read",
    "precompute_cache_load",
    "precompute_filter",
    "precompute_crop_extraction",
    "precompute_pose",
    "precompute_apriltag",
    "precompute_cnn_identity",
    # Streaming individual-analysis phases (Streaming Phases 3-4)
    "streaming_payload_build",
    "streaming_pose_dispatch",
    "streaming_cnn_dispatch",
    "streaming_evidence_emit",
    "pose_transport",
    "pose_inference",
    "pose_postprocess",
    "tracking_loop",
    "cleanup",
    # Post-processing phases (separate profiler instance in MergeWorker)
    "post_prepare",
    "post_resolve",
    "post_interpolate",
    "post_tag_identity",
    "post_rescale",
    # Interpolated crops phases (separate profiler in InterpolatedCropsWorker)
    "interp_setup",
    "interp_gap_detection",
    "interp_crop_extraction",
    "interp_finalize",
]


class TrackingProfiler:
    """Lightweight wall-clock profiler for the tracking pipeline.

    * ``tick`` / ``tock`` bracket each per-frame section.
    * ``phase_start`` / ``phase_end`` bracket coarse pipeline phases.
    * ``end_frame`` records the total frame time and keeps running totals.
    * ``log_periodic`` prints a summary to the logger every *n* frames.
    * ``export_summary`` writes a JSON report.

    When ``enabled=False`` (the default), every method is a no-op with
    negligible overhead.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if not enabled:
            return

        # Running totals
        self._totals: dict[str, float] = defaultdict(float)
        self._unit_totals: dict[str, float] = defaultdict(float)

        # Per-frame samples (lists of per-frame durations in seconds)
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._unit_samples: dict[str, list[float]] = defaultdict(list)

        # Current frame accumulators
        self._frame_totals: dict[str, float] = defaultdict(float)
        self._frame_units: dict[str, float] = defaultdict(float)

        # Interval accumulators (reset after each periodic log)
        self._interval_totals: dict[str, float] = defaultdict(float)
        self._interval_phase_totals: dict[str, float] = defaultdict(float)
        self._interval_phase_units: dict[str, float] = defaultdict(float)
        self._interval_frames: int = 0
        self._interval_first_frame_idx: int | None = None
        self._current_frame_idx: int | None = None

        # Tick tracking
        self._pending: dict[str, float] = {}

        # Frame-level bookkeeping
        self._frame_start: float | None = None
        self._total_frames: int = 0

        # Whole-run wall-clock
        self._run_start: float = time.time()
        self._run_end: float | None = None

        # Phase-level timing (batched detection, post-processing, etc.)
        self._phase_times: dict[str, float] = {}
        self._phase_units: dict[str, float] = defaultdict(float)
        self._phase_pending: dict[str, float] = {}

        # Optional run configuration metadata (set by caller)
        self._config: dict = {}

    # ------------------------------------------------------------------
    # Configuration metadata
    # ------------------------------------------------------------------
    def set_config(self, **kwargs) -> None:
        """Store run configuration metadata to include in the exported summary."""
        if not self.enabled:
            return
        self._config.update(kwargs)

    # ------------------------------------------------------------------
    # Phase / interval boundary helpers
    # ------------------------------------------------------------------
    def discard_frame_state(self) -> None:
        """Discard any pending per-frame tick/tock state without recording it.

        Call this at a phase boundary where tick/tock was used during a prior
        phase that never called ``end_frame()`` (e.g. the batched-detection
        phase uses ``tick/tock("batched_frame_read")`` across hundreds of
        batches but never flushes via ``end_frame``).  Without this call, the
        accumulated ``_frame_totals`` and the long-running ``_frame_start``
        timestamp would be flushed on the *first* Phase-2 ``end_frame()``,
        injecting the entire Phase-1 decode time into the per-tracking-frame
        statistics.
        """
        if not self.enabled:
            return
        self._frame_totals = defaultdict(float)
        self._frame_units = defaultdict(float)
        self._frame_start = None
        self._pending.clear()

    def reset_interval(self) -> None:
        """Reset the per-interval accumulators.

        Call this just before the main tracking loop starts so that coarse
        phase timings accumulated during Phase 1 (batched detection, confidence
        density map, precompute, etc.) do not appear in the first periodic
        per-frame summary window.
        """
        if not self.enabled:
            return
        self._interval_totals = defaultdict(float)
        self._interval_phase_totals = defaultdict(float)
        self._interval_phase_units = defaultdict(float)
        self._interval_frames = 0
        self._interval_first_frame_idx = None

    def notify_frame_index(self, frame_idx: int) -> None:
        """Record the current absolute frame index for periodic-log context.

        Call once per tracking-loop iteration, just after *frame_idx* is
        known, so that the periodic summary can report the exact frame range
        it covers (e.g. "frames 8661–8760").
        """
        if not self.enabled:
            return
        if self._interval_first_frame_idx is None:
            self._interval_first_frame_idx = frame_idx
        self._current_frame_idx = frame_idx

    # ------------------------------------------------------------------
    # Section timing
    # ------------------------------------------------------------------
    def tick(self, category: str) -> None:
        """Start timing *category* for the current frame."""
        if not self.enabled:
            return
        self._pending[category] = time.time()
        if self._frame_start is None:
            self._frame_start = time.time()

    def tock(self, category: str, work_units: float | int | None = None) -> None:
        """Stop timing *category*, accumulate into frame totals."""
        if not self.enabled:
            return
        start = self._pending.pop(category, None)
        if start is None:
            return
        elapsed = time.time() - start
        self._frame_totals[category] += elapsed
        self._record_frame_units(category, work_units)

    def add_sample(
        self,
        category: str,
        elapsed_s: float,
        work_units: float | int | None = None,
    ) -> None:
        """Accumulate a pre-measured duration into the current frame."""
        if not self.enabled:
            return
        elapsed = float(elapsed_s)
        if elapsed <= 0.0:
            return
        self._frame_totals[category] += elapsed
        self._record_frame_units(category, work_units)
        if self._frame_start is None:
            self._frame_start = time.time()

    # ------------------------------------------------------------------
    # Phase timing (for coarse phases: batched detection, post-processing)
    # ------------------------------------------------------------------
    def phase_start(self, name: str) -> None:
        """Begin timing a coarse pipeline phase (e.g. ``batched_detection``)."""
        if not self.enabled:
            return
        self._phase_pending[name] = time.time()

    def phase_end(self, name: str, work_units: float | int | None = None) -> None:
        """End timing a coarse pipeline phase."""
        if not self.enabled:
            return
        start = self._phase_pending.pop(name, None)
        if start is None:
            return
        elapsed = time.time() - start
        self._phase_times[name] = self._phase_times.get(name, 0.0) + elapsed
        self._record_phase_units(name, work_units)
        self._record_interval_phase(name, elapsed, work_units)

    def add_phase_time(
        self,
        name: str,
        elapsed_s: float,
        work_units: float | int | None = None,
    ) -> None:
        """Accumulate a pre-measured duration into a coarse phase total."""
        if not self.enabled:
            return
        elapsed = float(elapsed_s)
        if elapsed <= 0.0:
            return
        self._phase_times[name] = self._phase_times.get(name, 0.0) + elapsed
        self._record_phase_units(name, work_units)
        self._record_interval_phase(name, elapsed, work_units)

    # ------------------------------------------------------------------
    # Frame boundary
    # ------------------------------------------------------------------
    def end_frame(self) -> None:
        """Finalise timing for the current frame.

        * Computes ``frame_read`` as residual if not explicitly ticked.
        * Stores per-frame samples for percentile stats.
        """
        if not self.enabled:
            return
        # Compute frame_read as residual wall-clock gap if not explicitly timed.
        frame_wall = (
            (time.time() - self._frame_start) if self._frame_start is not None else 0.0
        )
        measured = sum(self._frame_totals.values())
        residual = max(0.0, frame_wall - measured)
        if "frame_read" not in self._frame_totals:
            self._frame_totals["frame_read"] = residual
        else:
            # If frame_read was explicitly timed, add any unaccounted gap as "other"
            if residual > 1e-6:
                self._frame_totals["other"] += residual

        # Flush per-frame data into running totals
        for cat, dur in self._frame_totals.items():
            self._totals[cat] += dur
            self._samples[cat].append(dur)
            self._interval_totals[cat] += dur
            units = self._frame_units.get(cat, 0.0)
            if units > 0.0:
                self._unit_totals[cat] += units
                self._unit_samples[cat].append(units)

        self._total_frames += 1
        self._interval_frames += 1

        # Reset frame accumulators
        self._frame_totals = defaultdict(float)
        self._frame_units = defaultdict(float)
        self._frame_start = None

    # ------------------------------------------------------------------
    # Periodic console summary (replaces the old 100-frame logger block)
    # ------------------------------------------------------------------
    def log_periodic(self, interval: int = 100) -> None:
        """Log a per-frame breakdown every *interval* frames."""
        if not self.enabled:
            return
        if self._interval_frames == 0 or self._interval_frames % interval != 0:
            return
        total = sum(self._interval_totals.values())
        if total <= 0:
            return

        # Build a context-rich header: pass name + absolute frame range.
        pass_name = self._config.get("pass_name", "")
        prefix = f"{pass_name.upper()} · " if pass_name else ""
        start_f = self._interval_first_frame_idx
        end_f = self._current_frame_idx
        if start_f is not None and end_f is not None:
            header = f"{prefix}frames {start_f}\u2013{end_f}  (+{self._interval_frames} frames)"
        else:
            header = f"{prefix}last {self._interval_frames} frames"
        logger.info("--- PROFILING [%s] ---", header)

        for cat in self._ordered_categories(self._interval_totals):
            dur = self._interval_totals[cat]
            pct = (dur / total) * 100
            avg_ms = (dur / self._interval_frames) * 1000
            unit_suffix = ""
            interval_units = self._interval_units_for_category(cat)
            if interval_units > 0.0:
                unit_suffix = f" | {(dur / interval_units) * 1000:.2f} ms/individual"
            logger.info(
                "  %-22s %5.1f%%  %7.2f ms/frame%s",
                cat,
                pct,
                avg_ms,
                unit_suffix,
            )

        if self._interval_phase_totals:
            logger.info("  PHASE BREAKDOWN")
            for name in self._ordered_phases(self._interval_phase_totals):
                dur = self._interval_phase_totals[name]
                pct = (dur / total) * 100 if total > 0 else 0.0
                avg_ms = (dur / self._interval_frames) * 1000
                unit_suffix = ""
                interval_units = self._interval_phase_units.get(name, 0.0)
                if interval_units > 0.0:
                    unit_suffix = (
                        f" | {(dur / interval_units) * 1000:.2f} ms/individual"
                    )
                logger.info(
                    "    %-30s %5.1f%%  %7.2f ms/frame%s",
                    name,
                    pct,
                    avg_ms,
                    unit_suffix,
                )
        logger.info(
            "  %-22s         %7.2f ms/frame",
            "TOTAL",
            (total / self._interval_frames) * 1000,
        )
        logger.info("-" * 50)

        # Reset interval accumulators for the next window.
        self._interval_totals = defaultdict(float)
        self._interval_phase_totals = defaultdict(float)
        self._interval_phase_units = defaultdict(float)
        self._interval_frames = 0
        self._interval_first_frame_idx = None

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        """Return the complete profiling summary as a dictionary."""
        if not self.enabled:
            return {"enabled": False}

        self._run_end = time.time()
        wall_clock = (self._run_end - self._run_start) if self._run_start else 0.0
        n = max(self._total_frames, 1)
        total_measured = sum(self._totals.values())

        # Per-category statistics
        categories = {}
        for cat in self._ordered_categories(self._totals):
            dur = self._totals[cat]
            samples = self._samples.get(cat, [])
            arr = np.array(samples, dtype=np.float64) if samples else np.zeros(1)
            categories[cat] = {
                "total_s": round(dur, 4),
                "percent": (
                    round((dur / total_measured) * 100, 2)
                    if total_measured > 0
                    else 0.0
                ),
                "mean_ms": round((dur / n) * 1000, 3),
                "std_ms": round(float(np.std(arr)) * 1000, 3),
                "min_ms": round(float(np.min(arr)) * 1000, 3),
                "max_ms": round(float(np.max(arr)) * 1000, 3),
                "p50_ms": round(float(np.median(arr)) * 1000, 3),
                "p95_ms": round(float(np.percentile(arr, 95)) * 1000, 3),
                "p99_ms": round(float(np.percentile(arr, 99)) * 1000, 3),
            }
            total_units = self._unit_totals.get(cat, 0.0)
            if total_units > 0.0:
                categories[cat]["total_units"] = self._format_units(total_units)
                categories[cat]["mean_units_per_frame"] = round(total_units / n, 3)
                categories[cat]["ms_per_unit"] = round((dur / total_units) * 1000, 3)

        # Phase-level timing (ordered by PHASE_ORDER, then any extras)
        phases = {}
        ordered_phase_names = [n for n in PHASE_ORDER if n in self._phase_times] + [
            n for n in self._phase_times if n not in PHASE_ORDER
        ]
        for name in ordered_phase_names:
            dur = self._phase_times[name]
            phases[name] = {
                "total_s": round(dur, 4),
                "avg_ms_per_frame": round((dur / n) * 1000, 3),
                "percent_of_wall": (
                    round((dur / wall_clock) * 100, 2) if wall_clock > 0 else 0.0
                ),
            }
            total_units = self._phase_units.get(name, 0.0)
            if total_units > 0.0:
                phases[name]["total_units"] = self._format_units(total_units)
                phases[name]["ms_per_unit"] = round((dur / total_units) * 1000, 3)

        summary = {
            "enabled": True,
            "total_frames": self._total_frames,
            "wall_clock_s": round(wall_clock, 3),
            "measured_s": round(total_measured, 4),
            "avg_fps": round(n / wall_clock, 2) if wall_clock > 0 else 0.0,
            "avg_frame_ms": (
                round((total_measured / n) * 1000, 3) if total_measured > 0 else 0.0
            ),
            "phases": phases,
            "categories": categories,
        }
        if self._config:
            summary["config"] = self._config
        return summary

    def export_summary(self, path: str | Path) -> str | None:
        """Write the profiling summary as a JSON file.

        Returns the path written, or ``None`` on failure.
        """
        if not self.enabled:
            return None
        summary = self.get_summary()
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(summary, fh, indent=2)
            logger.info("Profiling summary exported to %s", path)
            return str(path)
        except Exception:
            logger.warning("Failed to export profiling summary", exc_info=True)
            return None

    def log_final_summary(self) -> None:
        """Log a comprehensive final summary to the logger."""
        if not self.enabled:
            return
        summary = self.get_summary()
        n = summary["total_frames"]
        if n == 0:
            return

        pass_name = self._config.get("pass_name", "")
        pass_label = f"  {pass_name.upper()} PASS — " if pass_name else "  "

        logger.info("=" * 60)
        logger.info("%sTRACKING PROFILING — FINAL SUMMARY", pass_label)
        logger.info("=" * 60)
        logger.info(
            "  Frames: %d | Wall-clock: %.1fs | Throughput: %.1f fps",
            n,
            summary["wall_clock_s"],
            summary["avg_fps"],
        )

        if summary["phases"]:
            logger.info("-" * 60)
            logger.info("  PHASE TIMING  (% of wall-clock)")
            for name, info in summary["phases"].items():
                unit_suffix = ""
                if "ms_per_unit" in info:
                    unit_suffix = f" | {info['ms_per_unit']:.2f} ms/unit"
                logger.info(
                    "    %-30s %8.2f s  (%5.1f%%)%s",
                    name,
                    info["total_s"],
                    info.get("percent_of_wall", 0.0),
                    unit_suffix,
                )

        if summary["categories"]:
            logger.info("-" * 60)
            logger.info(
                "  TRACKING-LOOP CATEGORIES  (per-frame, %d frames, avg %.2f ms/frame)",
                n,
                summary["avg_frame_ms"],
            )
            logger.info(
                "  %-22s %6s %8s %8s %8s %8s",
                "CATEGORY",
                "%",
                "mean",
                "p50",
                "p95",
                "max",
            )
            logger.info("-" * 60)
            for cat, info in summary["categories"].items():
                unit_suffix = ""
                if "ms_per_unit" in info:
                    unit_suffix = f" | {info['ms_per_unit']:.2f} ms/individual"
                logger.info(
                    "  %-22s %5.1f%% %7.2fms %7.2fms %7.2fms %7.2fms%s",
                    cat,
                    info["percent"],
                    info["mean_ms"],
                    info["p50_ms"],
                    info["p95_ms"],
                    info["max_ms"],
                    unit_suffix,
                )
            logger.info("-" * 60)
            logger.info(
                "  %-22s        %7.2fms",
                "TOTAL",
                summary["avg_frame_ms"],
            )
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ordered_categories(self, d: dict) -> list[str]:
        """Return keys of *d* in canonical order, extras appended alphabetically."""
        present = [c for c in CATEGORY_ORDER if c in d and d[c] > 0]
        extras = sorted(c for c in d if c not in CATEGORY_ORDER and d[c] > 0)
        return present + extras

    def _ordered_phases(self, d: dict) -> list[str]:
        """Return phase names in canonical order, extras appended alphabetically."""
        present = [p for p in PHASE_ORDER if p in d and d[p] > 0]
        extras = sorted(p for p in d if p not in PHASE_ORDER and d[p] > 0)
        return present + extras

    def _interval_units_for_category(self, category: str) -> float:
        """Best-effort interval work-unit count for a category."""
        phase_name = category.removeprefix("live_")
        if phase_name in self._interval_phase_units:
            return self._interval_phase_units.get(phase_name, 0.0)
        return 0.0

    def _record_interval_phase(
        self,
        name: str,
        elapsed_s: float,
        work_units: float | int | None = None,
    ) -> None:
        """Accumulate phase timing into the current periodic interval."""
        elapsed = float(elapsed_s)
        if elapsed <= 0.0:
            return
        self._interval_phase_totals[name] += elapsed
        units = self._coerce_units(work_units)
        if units > 0.0:
            self._interval_phase_units[name] += units

    def _record_frame_units(
        self, category: str, work_units: float | int | None = None
    ) -> None:
        units = self._coerce_units(work_units)
        if units > 0.0:
            self._frame_units[category] += units

    def _record_phase_units(
        self, name: str, work_units: float | int | None = None
    ) -> None:
        units = self._coerce_units(work_units)
        if units > 0.0:
            self._phase_units[name] += units

    @staticmethod
    def _coerce_units(work_units: float | int | None) -> float:
        try:
            units = float(work_units)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(units) or units <= 0.0:
            return 0.0
        return units

    @staticmethod
    def _format_units(units: float) -> int | float:
        rounded = round(units)
        if abs(units - rounded) < 1e-6:
            return int(rounded)
        return round(units, 3)
