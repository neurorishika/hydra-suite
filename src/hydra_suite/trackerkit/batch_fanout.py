"""Process-per-video fan-out for TrackerKit batches (Qt-free).

Each job is the ordinary CLI child ``python -m hydra_suite.trackerkit.app track
<video> --config <json>`` -- the exact sequential path -- pinned to one GPU
with ``CUDA_VISIBLE_DEVICES=<uuid>``. The SLEAP service the child spawns
inherits that mask, so pose runs on the same GPU. This module owns
scheduling, log capture, progress parsing, failure policy and cancellation;
it changes nothing about how a video is tracked.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Sequence, TextIO

from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit.batch_plan import BatchJobSpec
from hydra_suite.utils.video_artifacts import (
    build_video_log_dir,
    choose_writable_artifact_base_dir,
)

logger = logging.getLogger(__name__)

_PROGRESS_RE = re.compile(
    r"\[(?:track forward|track backward|post)\] (\d{1,3})% ?(.*)$"
)
_SUMMARY_RE = re.compile(r"Tracker CLI completed: (.*)$")
_THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass
class FanoutOptions:
    gpus: list[CudaDevice] = field(default_factory=list)
    jobs: int = 1
    threads_per_job: Optional[int] = None
    log_level: str = "INFO"
    run_dir: Optional[Path] = None
    child_command: Optional[Callable[[BatchJobSpec, Path], list[str]]] = None
    poll_s: float = 0.2
    sigint_grace_s: float = 10.0
    term_grace_s: float = 5.0


@dataclass
class FanoutJobResult:
    spec: BatchJobSpec
    gpu: Optional[CudaDevice]
    returncode: Optional[int]
    success: bool
    log_path: Path
    summary_lines: list[str]
    error: Optional[str]
    wall_s: float


@dataclass
class FanoutResult:
    jobs: list[FanoutJobResult]
    cancelled: bool

    @property
    def success(self) -> bool:
        return (
            not self.cancelled and bool(self.jobs) and all(j.success for j in self.jobs)
        )


class FanoutEvents(Protocol):
    def job_started(  # noqa: E704
        self, spec: BatchJobSpec, gpu: Optional[CudaDevice], log_path: Path
    ) -> None: ...

    def job_progress(  # noqa: E704
        self, spec: BatchJobSpec, percent: int, message: str
    ) -> None: ...

    def job_log(self, spec: BatchJobSpec, line: str) -> None: ...  # noqa: E704

    def job_finished(self, result: FanoutJobResult) -> None: ...  # noqa: E704


class NullEvents:
    def job_started(self, spec, gpu, log_path) -> None: ...  # noqa: E704

    def job_progress(self, spec, percent, message) -> None: ...  # noqa: E704

    def job_log(self, spec, line) -> None: ...  # noqa: E704

    def job_finished(self, result) -> None: ...  # noqa: E704


def parse_progress_line(line: str) -> Optional[tuple[int, str]]:
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip()


def parse_summary_line(line: str) -> Optional[list[str]]:
    m = _SUMMARY_RE.search(line)
    if not m:
        return None
    return [part.strip() for part in m.group(1).split("|") if part.strip()]


def build_child_env(
    base: Mapping[str, str],
    *,
    gpu: Optional[CudaDevice],
    threads_per_job: Optional[int],
) -> dict[str, str]:
    """Inherit everything (conda, HYDRA_*), pin the GPU, force unbuffered logs."""
    env = dict(base)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if threads_per_job is not None and int(threads_per_job) > 0:
        for var in _THREAD_CAP_VARS:
            env.setdefault(var, str(int(threads_per_job)))
    return env


def default_child_command(
    spec: BatchJobSpec, config_json: Path, *, log_level: str = "INFO"
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "hydra_suite.trackerkit.app",
        "--log-level",
        str(log_level),
        "track",
        spec.video_path,
        "--config",
        str(config_json),
    ]


def _job_log_path(spec: BatchJobSpec, timestamp: str) -> Path:
    base = choose_writable_artifact_base_dir(spec.video_path)
    log_dir = build_video_log_dir(spec.video_path, artifact_base_dir=base, create=True)
    return log_dir / f"{Path(spec.video_path).stem}_fanout_{timestamp}.log"


def _command_for_log(command: Sequence[str]) -> str:
    """One-line rendering of the child command for the log header.

    Multi-line arguments (an inline ``python -c`` script, as the tests use)
    are elided so the header stays a single line and never mimics child
    output.
    """
    return " ".join("<inline-script>" if "\n" in arg else arg for arg in command)


@dataclass
class _Live:
    spec: BatchJobSpec
    gpu: Optional[CudaDevice]
    proc: subprocess.Popen
    log_path: Path
    log_handle: TextIO
    started_at: float
    reader: Optional[threading.Thread] = None
    summary_lines: list[str] = field(default_factory=list)
    last_error: Optional[str] = None


def _pump(live: _Live, events: FanoutEvents) -> None:
    assert live.proc.stdout is not None
    try:
        for raw in live.proc.stdout:
            line = raw.rstrip("\r\n")
            try:
                live.log_handle.write(line + "\n")
                live.log_handle.flush()
            except OSError:
                pass
            progress = parse_progress_line(line)
            if progress is not None:
                events.job_progress(live.spec, *progress)
            summary = parse_summary_line(line)
            if summary is not None:
                live.summary_lines = summary
            if " - ERROR - " in line or line.startswith("Error:"):
                live.last_error = line
            events.job_log(live.spec, line)
    except Exception as exc:  # noqa: BLE001 - reader must never kill the scheduler
        live.last_error = f"log reader failed: {exc}"


def _launch(
    spec: BatchJobSpec,
    gpu: Optional[CudaDevice],
    options: FanoutOptions,
    run_dir: Path,
    timestamp: str,
    events: FanoutEvents,
) -> _Live:
    config_json = run_dir / f"job_{spec.index}_config.json"
    config_json.write_text(json.dumps(spec.config, indent=2), encoding="utf-8")
    command = (
        options.child_command
        or (lambda s, c: default_child_command(s, c, log_level=options.log_level))
    )(spec, config_json)
    env = build_child_env(os.environ, gpu=gpu, threads_per_job=options.threads_per_job)
    log_path = _job_log_path(spec, timestamp)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"# trackerkit fan-out job {spec.index}: {spec.video_path}\n"
        f"# gpu={gpu.uuid if gpu else '<inherited>'} "
        f"command={_command_for_log(command)}\n"
    )
    log_handle.flush()
    popen_kwargs: dict = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(command, **popen_kwargs)
    live = _Live(
        spec=spec,
        gpu=gpu,
        proc=proc,
        log_path=log_path,
        log_handle=log_handle,
        started_at=time.monotonic(),
    )
    live.reader = threading.Thread(
        target=_pump, args=(live, events), name=f"fanout-log-{spec.index}", daemon=True
    )
    live.reader.start()
    events.job_started(spec, gpu, log_path)
    logger.info(
        "Fan-out: launched job %d (%s) on %s -> %s",
        spec.index,
        Path(spec.video_path).name,
        gpu.uuid if gpu else "inherited device",
        log_path,
    )
    return live


def _finish(live: _Live, *, cancelled: bool) -> FanoutJobResult:
    if live.reader is not None:
        live.reader.join(timeout=5)
    try:
        live.log_handle.close()
    except Exception:
        pass
    rc = live.proc.returncode
    ok = rc == 0 and not cancelled
    error = None
    if not ok:
        error = "cancelled" if cancelled else (live.last_error or f"exit code {rc}")
    return FanoutJobResult(
        spec=live.spec,
        gpu=live.gpu,
        returncode=rc,
        success=ok,
        log_path=live.log_path,
        summary_lines=list(live.summary_lines),
        error=error,
        wall_s=time.monotonic() - live.started_at,
    )


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole session group, falling back to the child alone.

    The child spawns grandchildren -- notably the SLEAP service via ``conda run
    -n sleap``. Signalling only the child's pid leaves those orphaned on the
    escalation path, so SIGTERM/SIGKILL go to the process group. This is safe
    because ``start_new_session=True`` in :func:`_launch` makes that group ours
    and ours alone; we can never signal the scheduler or an unrelated process.
    """
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass  # group already gone or not ours: fall back to the child
    try:
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        pass


def _stop_children(running: list[_Live], options: FanoutOptions) -> None:
    """SIGINT (clean engine stop) -> SIGTERM -> SIGKILL, with grace periods."""

    def _alive() -> list[_Live]:
        return [job for job in running if job.proc.poll() is None]

    # SIGINT goes to the child pid ONLY: its handler performs a clean engine
    # stop and shuts down its own SLEAP service. Broadcasting it to the group
    # would race that orderly teardown.
    for live in _alive():
        try:
            # NOTE (Windows): CTRL_BREAK_EVENT is only valid for a child started
            # with creationflags=CREATE_NEW_PROCESS_GROUP, which _launch does not
            # set. On nt this would hit the whole console group, scheduler
            # included. Deployment is macOS/Linux; a Windows port must fix this.
            live.proc.send_signal(
                signal.SIGINT if os.name != "nt" else signal.CTRL_BREAK_EVENT
            )
        except Exception:
            pass
    deadline = time.monotonic() + options.sigint_grace_s
    while _alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    # Escalation: the child forfeited its clean exit, so take the whole group
    # down with it rather than leaking grandchildren.
    for live in _alive():
        _signal_group(live.proc, signal.SIGTERM)
    deadline = time.monotonic() + options.term_grace_s
    while _alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    for live in _alive():
        _signal_group(live.proc, signal.SIGKILL)
    for live in running:
        try:
            live.proc.wait(timeout=5)
        except Exception:
            pass


def run_batch_fanout(
    specs: Sequence[BatchJobSpec],
    options: FanoutOptions,
    *,
    events: Optional[FanoutEvents] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> FanoutResult:
    """Run every spec as a child process across the configured slots."""
    events = events or NullEvents()
    should_stop = should_stop or (lambda: False)
    specs = list(specs)
    if not specs:
        return FanoutResult(jobs=[], cancelled=False)
    # Indices are 1-based and must be unique: they name the per-job effective
    # config file. Callers that leave them unset (or collide) get reindexed.
    if any(s.index < 1 for s in specs) or len({s.index for s in specs}) != len(specs):
        specs = [replace(s, index=i) for i, s in enumerate(specs, 1)]

    slots: list[Optional[CudaDevice]]
    if options.gpus:
        n = max(
            1,
            min(
                int(options.jobs) if options.jobs else len(options.gpus),
                len(options.gpus),
            ),
        )
        slots = list(options.gpus[:n])
    else:
        slots = [None] * max(1, int(options.jobs))

    run_dir = (
        Path(options.run_dir)
        if options.run_dir
        else Path(tempfile.mkdtemp(prefix="trackerkit-fanout-"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pending = list(specs)
    running: list[_Live] = []
    finished: dict[int, FanoutJobResult] = {}
    free_slots: list[Optional[CudaDevice]] = list(slots)
    halted = False
    cancelled = False

    while pending or running:
        if should_stop():
            cancelled = True
            _stop_children(running, options)
            for live in running:
                res = _finish(live, cancelled=True)
                finished[live.spec.index] = res
                events.job_finished(res)
            running.clear()
            break

        # reap
        for live in list(running):
            if live.proc.poll() is not None:
                running.remove(live)
                free_slots.append(live.gpu)
                res = _finish(live, cancelled=False)
                finished[live.spec.index] = res
                events.job_finished(res)
                if not res.success:
                    halted = True
                    logger.error(
                        "Fan-out: job %d failed (%s); no further jobs will launch",
                        live.spec.index,
                        res.error,
                    )

        # launch
        while pending and free_slots and not halted:
            spec = pending.pop(0)
            gpu = free_slots.pop(0)
            running.append(_launch(spec, gpu, options, run_dir, timestamp, events))

        if not running and (halted or not pending):
            break
        time.sleep(options.poll_s)

    results: list[FanoutJobResult] = []
    for spec in specs:
        if spec.index in finished:
            results.append(finished[spec.index])
        else:
            results.append(
                FanoutJobResult(
                    spec=spec,
                    gpu=None,
                    returncode=None,
                    success=False,
                    log_path=run_dir / f"job_{spec.index}_not_started.log",
                    summary_lines=[],
                    error="not started" if not cancelled else "cancelled",
                    wall_s=0.0,
                )
            )
    return FanoutResult(jobs=results, cancelled=cancelled)
