"""Minimal TrackerKit CLI runner for config-driven tracking sessions (Qt-free)."""

from __future__ import annotations

import json
import logging
import signal
import tempfile
import threading
from pathlib import Path
from typing import Callable, Sequence

from hydra_suite.runtime.cuda_devices import parse_gpu_selectors, resolve_gpu_selectors
from hydra_suite.trackerkit.batch_fanout import (
    FanoutOptions,
    FanoutResult,
    run_batch_fanout,
)
from hydra_suite.trackerkit.batch_plan import BatchJobSpec, plan_batch_jobs
from hydra_suite.trackerkit.cli_config import load_tracker_cli_session
from hydra_suite.trackerkit.headless_tracking import run_headless_tracking_session

logger = logging.getLogger(__name__)


def fanout_requested(gpus: str | None, jobs: int | None) -> bool:
    """The gating rule: fan-out iff --gpus was given or --jobs > 1.

    ``jobs is None`` means "not specified" and is equivalent to 1 here: on its
    own it never engages fan-out, so the plain ``trackerkit track a.mp4 b.mp4``
    invocation keeps the in-process sequential path.
    """
    return bool(str(gpus or "").strip()) or int(jobs or 1) > 1


def run_tracking_cli(
    video_paths: Sequence[str],
    *,
    config_path: str | None = None,
    keystone_override: bool = False,
    sahi_profile: str | None = None,
    gpus: str | None = None,
    jobs: int | None = None,
    threads_per_job: int | None = None,
    log_level: str = "INFO",
) -> int:
    """Run one or more TrackerKit sessions from the CLI (direct Qt-free path)."""

    videos = [str(path).strip() for path in video_paths if str(path).strip()]
    if not videos:
        raise ValueError("At least one video path is required.")

    for video_path in videos:
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
    if config_path and not Path(config_path).is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    specs = plan_batch_jobs(
        videos,
        explicit_config_path=config_path,
        keystone_override=keystone_override,
        sahi_profile=sahi_profile,
    )
    if not specs:
        raise ValueError("No videos were resolved for tracking.")
    if not fanout_requested(gpus, jobs):
        return _run_sequential(specs)
    devices = resolve_gpu_selectors(parse_gpu_selectors(gpus)) if gpus else []
    # An unspecified --jobs means "one slot per selected GPU" (and 1 with no
    # GPUs); an explicit --jobs is honoured but never exceeds the GPU count,
    # because a second job on a GPU would contend for that GPU's memory.
    if devices and jobs is None:
        effective_jobs = len(devices)
    else:
        effective_jobs = max(1, int(jobs or 1))
        if devices:
            effective_jobs = min(effective_jobs, len(devices))
    options = FanoutOptions(
        gpus=devices,
        jobs=effective_jobs,
        threads_per_job=threads_per_job,
        log_level=log_level,
    )
    return _run_fanout(specs, options)


def _run_sequential(specs: Sequence[BatchJobSpec]) -> int:
    """The in-process path: one session after another on this process."""
    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="trackerkit-cli-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for spec in specs:
            logger.info(
                "Tracker CLI: preparing video %s/%s: %s",
                spec.index,
                len(specs),
                spec.video_path,
            )
            session = load_tracker_cli_session(
                spec.video_path,
                config_path=spec.config_path,
                config_data=spec.config,
            )
            # Persist the resolved keystone baseline for provenance/debugging;
            # the direct path consumes ``session`` directly and needs no config
            # file. Dump the OVERRIDDEN config (``session.config``), not the
            # pre-override baseline, so the provenance file names the profile
            # that actually ran. ``config_path is None`` narrows this to the
            # videos that truly inherited the baseline dict -- a later video
            # that loads the keystone's own file is keystone provenance too,
            # but it has a file of its own to point at.
            if spec.provenance == "keystone-baseline" and spec.config_path is None:
                keystone_dump = tmpdir_path / f"keystone_config_{spec.index}.json"
                with open(keystone_dump, "w", encoding="utf-8") as handle:
                    json.dump(session.config or {}, handle, indent=2)

            result = run_headless_tracking_session(session)

            if result.get("success"):
                summary = " | ".join(result.get("lines", []))
                logger.info("Tracker CLI completed: %s", summary)
            else:
                error_message = result.get("error") or "Tracker session failed."
                logger.error(
                    "Tracker CLI failed for %s: %s",
                    spec.video_path,
                    error_message,
                )
                exit_code = 1
                break

    return exit_code


class _CliEvents:
    """Log-only event sink for the terminal."""

    def job_started(self, spec, gpu, log_path) -> None:
        logger.info(
            "[job %d] started %s on %s (log: %s)",
            spec.index,
            Path(spec.video_path).name,
            gpu.uuid if gpu else "inherited device",
            log_path,
        )

    def job_progress(self, spec, percent, message) -> None:
        logger.info("[job %d] %3d%% %s", spec.index, percent, message)

    def job_log(self, spec, line) -> None:
        # Child lines already land in the per-job log file; the terminal only
        # shows the progress/lifecycle summary.
        pass

    def job_finished(self, result) -> None:
        logger.info(
            "[job %d] %s (%.1fs)",
            result.spec.index,
            "OK" if result.success else f"FAIL: {result.error}",
            result.wall_s,
        )


def _install_stop_flag() -> tuple[Callable[[], bool], Callable[[], None]]:
    """Turn SIGINT into a cooperative stop flag for the fan-out scheduler."""
    stop = threading.Event()
    try:
        previous = signal.getsignal(signal.SIGINT)

        def _handler(_signum, _frame):
            logger.warning("SIGINT received - stopping all fan-out children.")
            stop.set()

        signal.signal(signal.SIGINT, _handler)

        def _restore() -> None:
            try:
                signal.signal(signal.SIGINT, previous)
            except (ValueError, OSError):
                pass

    except (ValueError, OSError):
        # Not the main thread (or no signal support): run without a handler.
        def _restore() -> None:  # noqa: E704
            return None

    return stop.is_set, _restore


def _print_fanout_table(result: FanoutResult) -> None:
    rows = [("#", "video", "gpu", "status", "wall", "log")]
    for job in result.jobs:
        rows.append(
            (
                str(job.spec.index),
                Path(job.spec.video_path).name,
                (job.gpu.uuid[:12] if job.gpu else "-"),
                "OK" if job.success else f"FAIL ({job.error})",
                f"{job.wall_s:.0f}s",
                str(job.log_path),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    ok = sum(1 for job in result.jobs if job.success)
    print(
        f"\n{ok}/{len(result.jobs)} videos succeeded"
        + ("  (CANCELLED)" if result.cancelled else "")
    )


def _run_fanout(specs: Sequence[BatchJobSpec], options: FanoutOptions) -> int:
    """The process-per-slot path: one child per video, N at a time."""
    should_stop, restore = _install_stop_flag()
    try:
        logger.info(
            "Tracker CLI fan-out: %d videos, %d slot(s)%s",
            len(specs),
            len(options.gpus) if options.gpus else options.jobs,
            f" on GPUs {[g.index for g in options.gpus]}" if options.gpus else "",
        )
        result = run_batch_fanout(
            specs, options, should_stop=should_stop, events=_CliEvents()
        )
    finally:
        restore()
    _print_fanout_table(result)
    return 0 if result.success else 1
