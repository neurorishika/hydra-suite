"""Minimal TrackerKit CLI runner for config-driven tracking sessions (Qt-free)."""

from __future__ import annotations

import json
import logging
import signal
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from hydra_suite.runtime.cpu_budget import (
    ENV_CPU_THREADS,
    bind_thread_cpu_cap,
    inherited_cpu_threads,
    resolve_cpu_budget,
    set_process_cpu_cap,
    single_run_cpu_cap,
)
from hydra_suite.runtime.cuda_devices import list_cuda_devices, parse_gpu_selectors
from hydra_suite.trackerkit.batch_fanout import (
    FanoutOptions,
    FanoutResult,
    build_gpu_slots,
    decide_gpu_slots,
    host_has_cuda,
    is_auto_max_jobs,
    parse_max_jobs,
    plan_fanout_cpu,
    resolve_auto_max_jobs,
    resolve_slot_count,
    run_batch_fanout,
)
from hydra_suite.trackerkit.batch_plan import BatchJobSpec, plan_batch_jobs
from hydra_suite.trackerkit.cli_config import load_tracker_cli_session
from hydra_suite.trackerkit.headless_tracking import (
    install_stop_signal_handlers,
    restore_signal_handlers,
    run_headless_tracking_session,
)

logger = logging.getLogger(__name__)


def fanout_requested(gpus: str | None, jobs: int | None) -> bool:
    """The gating rule: fan-out iff --gpus was given or --jobs > 1.

    ``jobs is None`` means "not specified" and is equivalent to 1 here: on its
    own it never engages fan-out, so the plain ``trackerkit track a.mp4 b.mp4``
    invocation keeps the in-process sequential path.
    """
    return bool(str(gpus or "").strip()) or int(jobs or 1) > 1


def _inherited_cpu_threads() -> Optional[int]:
    """A valid ``HYDRA_CPU_THREADS`` from this process's env, else None."""
    return inherited_cpu_threads(logger=logger)


def _single_run_cpu_cap(cpu_budget: Optional[int]) -> Optional[int]:
    """The thread cap for an in-process (non-fan-out) run, or None."""
    return single_run_cpu_cap(cpu_budget, logger=logger)


def _resolve_auto_max_jobs(
    *,
    n_gpus: int,
    jobs_per_gpu: int,
    cpu_budget: Optional[int],
    n_videos: int,
    threads_per_job: Optional[int],
) -> Optional[int]:
    """``--max-jobs auto`` -> M (logged), or None when M <= GPUs x K.

    ``cpu_budget`` is the same figure :func:`plan_fanout_cpu` splits (flag /
    HYDRA_CPU_BUDGET / usable cores, already bounded by an inherited cap). At
    or below GPUs x K no child would wait on the GPU phase lock, so the run
    takes the default (unset) path unchanged.
    """
    budget = resolve_cpu_budget(cpu_budget)
    m, reason = resolve_auto_max_jobs(
        n_gpus=n_gpus,
        jobs_per_gpu=jobs_per_gpu,
        budget_cores=budget.cores,
        n_videos=n_videos,
        threads_per_job=threads_per_job,
    )
    logger.info("--max-jobs auto -> %d (%s)", m, reason)
    return m if m > n_gpus * jobs_per_gpu else None


def run_tracking_cli(
    video_paths: Sequence[str],
    *,
    config_path: str | None = None,
    keystone_override: bool = False,
    sahi_profile: str | None = None,
    video_scale: float | None = None,
    gpus: str | None = None,
    jobs: int | None = None,
    jobs_per_gpu: int = 1,
    longest_first: bool = False,
    max_jobs: int | str | None = None,
    keep_going: bool = False,
    threads_per_job: int | None = None,
    log_level: str = "INFO",
    apply_tuned_inference: bool | None = None,
    inference_autotune_manual: Sequence[str] | None = None,
    cpu_budget: int | None = None,
) -> int:
    """Run one or more TrackerKit sessions from the CLI (direct Qt-free path)."""

    if int(jobs_per_gpu) < 1:
        raise ValueError(f"jobs_per_gpu must be >= 1, got {jobs_per_gpu}")
    if int(jobs_per_gpu) > 1 and not str(gpus or "").strip():
        raise ValueError("--jobs-per-gpu needs --gpus")
    if longest_first and not fanout_requested(gpus, jobs):
        raise ValueError("--longest-first needs fan-out (--gpus or --jobs > 1)")
    if max_jobs is not None:
        # An integer >= 1 or "auto" (resolved below, once the GPUs and the
        # CPU budget are known).
        max_jobs = parse_max_jobs(max_jobs)
        if not str(gpus or "").strip():
            raise ValueError("--max-jobs needs --gpus")
        if jobs is not None:
            raise ValueError("use either --jobs or --max-jobs, not both")
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
        video_scale=video_scale,
        apply_tuned_inference=apply_tuned_inference,
        inference_autotune_manual=inference_autotune_manual,
    )
    if not specs:
        raise ValueError("No videos were resolved for tracking.")
    if not fanout_requested(gpus, jobs):
        # Opt-in single-run cap (--cpu-budget / HYDRA_CPU_BUDGET), or the
        # per-child cap a fan-out parent hands down via HYDRA_CPU_THREADS.
        # Applied before _run_sequential, which is where every session (and
        # its inference / AprilTag config) is built. No budget source -> None
        # -> nothing is touched (plain-path invariant).
        cap = _single_run_cpu_cap(cpu_budget)
        if cap is not None:
            set_process_cpu_cap(cap)
            bind_thread_cpu_cap()
        if keep_going:
            return _run_sequential(specs, keep_going=True)
        return _run_sequential(specs)
    devices = (
        decide_gpu_slots(
            parse_gpu_selectors(gpus), list_cuda_devices(), host_has_cuda()
        )
        if gpus
        else []
    )
    # An unspecified --jobs means "every GPU slot" (GPUs x --jobs-per-gpu, and
    # 1 with no GPUs); an explicit --jobs is honoured but never exceeds the
    # slot count, so a GPU is shared only when --jobs-per-gpu asks for it.
    effective_jobs = len(build_gpu_slots(devices, jobs=jobs, jobs_per_gpu=jobs_per_gpu))
    if max_jobs is not None and not devices:
        logger.warning(
            "--max-jobs %s ignored: --gpus %s resolved to no CUDA device, so the "
            "batch runs unpinned with %d job(s).",
            max_jobs,
            gpus,
            effective_jobs,
        )
    if int(jobs_per_gpu) > 1 and not devices:
        logger.warning(
            "--jobs-per-gpu %d ignored: --gpus %s resolved to no CUDA device, "
            "so the batch runs unpinned with %d job(s).",
            int(jobs_per_gpu),
            gpus,
            effective_jobs,
        )
    # The parent itself stays uncapped (it only schedules); each child gets
    # budget // jobs threads via HYDRA_CPU_THREADS, and the slot count
    # (GPUs x --jobs-per-gpu, from build_gpu_slots) is clamped to what the
    # budget can feed (unless --threads-per-job is explicit).
    inherited = _inherited_cpu_threads()
    if inherited is not None:
        # This process already runs under a thread cap (a shell export or an
        # outer fan-out): the children together must stay inside it, so it is
        # a ceiling on the budget, never something they multiply.
        cpu_budget = inherited if cpu_budget is None else min(cpu_budget, inherited)
        logger.info(
            "CPU budget: %s=%d bounds this fan-out; budget is %d core(s).",
            ENV_CPU_THREADS,
            inherited,
            cpu_budget,
        )
    if is_auto_max_jobs(max_jobs) and devices:
        max_jobs = _resolve_auto_max_jobs(
            n_gpus=len(devices),
            jobs_per_gpu=int(jobs_per_gpu),
            cpu_budget=cpu_budget,
            n_videos=len(specs),
            threads_per_job=threads_per_job,
        )
    options, cpu_plan = plan_fanout_cpu(
        FanoutOptions(
            gpus=devices,
            jobs=effective_jobs,
            jobs_per_gpu=int(jobs_per_gpu) if devices else 1,
            longest_first=bool(longest_first),
            max_jobs=int(max_jobs) if (max_jobs is not None and devices) else None,
            keep_going=bool(keep_going),
            threads_per_job=threads_per_job,
            log_level=log_level,
        ),
        n_videos=len(specs),
        cpu_budget=cpu_budget,
    )
    for warning in cpu_plan.warnings:
        logger.warning("CPU budget: %s", warning)
    logger.info(
        "Fan-out: %d concurrent job(s) x %d CPU thread(s)",
        resolve_slot_count(options),
        options.threads_per_job,
    )
    return _run_fanout(specs, options)


def _run_sequential(specs: Sequence[BatchJobSpec], *, keep_going: bool = False) -> int:
    """The in-process path: one session after another on this process.

    A failed video stops the batch unless ``keep_going``.
    """
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
                if not keep_going:
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


def _fanout_stop_signals() -> list[int]:
    """Signals that must stop a fan-out run.

    SIGINT alone is not enough: a batch scheduler or ``kill`` sends SIGTERM,
    and a closed terminal sends SIGHUP. Either would kill the parent outright
    and strand every GPU child, so both are trapped and turned into the same
    cooperative stop flag. SIGHUP does not exist on Windows.
    """
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    # Windows: the batch fan-out stops a child with CTRL_BREAK_EVENT (its own
    # process group), which arrives as SIGBREAK.
    if hasattr(signal, "SIGBREAK"):
        signals.append(signal.SIGBREAK)
    return signals


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
    stop_event, previous_handlers, installed = install_stop_signal_handlers(
        _fanout_stop_signals(),
        log_message="%s received - stopping all fan-out children.",
    )
    try:
        logger.info(
            "Tracker CLI fan-out: %d videos, %d slot(s)%s",
            len(specs),
            resolve_slot_count(options),
            (
                f" on GPUs {[g.index for g in options.gpus]}"
                + (
                    f" x {options.jobs_per_gpu} jobs per GPU"
                    if resolve_slot_count(options) > len(options.gpus)
                    and options.max_jobs is None
                    else ""
                )
                + (
                    f", at most {options.jobs_per_gpu} GPU phase(s) per GPU"
                    if options.max_jobs is not None
                    and resolve_slot_count(options)
                    > len(options.gpus) * int(options.jobs_per_gpu)
                    else ""
                )
                if options.gpus
                else ""
            ),
        )
        result = run_batch_fanout(
            specs, options, should_stop=stop_event.is_set, events=_CliEvents()
        )
    finally:
        restore_signal_handlers(previous_handlers, installed)
    _print_fanout_table(result)
    return 0 if result.success else 1
