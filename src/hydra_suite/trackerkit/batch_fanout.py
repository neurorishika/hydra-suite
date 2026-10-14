"""Process-per-video fan-out for TrackerKit batches (Qt-free).

Each job is the ordinary CLI child ``python -m hydra_suite.trackerkit.app track
<video> --config <json>`` -- the exact sequential path -- pinned to one GPU
with ``CUDA_VISIBLE_DEVICES=<uuid>``. A GPU may host ``jobs_per_gpu`` children
at once; each is still its own process and session group. The SLEAP service the child spawns
inherits that mask, so pose runs on the same GPU. With ``max_jobs`` above
GPUs x ``jobs_per_gpu`` more children run than there are GPU slots; each child
then takes a slot only around its GPU phases (see
:mod:`hydra_suite.runtime.gpu_phase_lock`). This module owns
scheduling, log capture, progress parsing, failure policy and cancellation;
it changes nothing about how a video is tracked.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
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

from hydra_suite.runtime.cpu_budget import (
    CpuBudget,
    CpuPlan,
    capped_child_env,
    plan_cpu,
    resolve_cpu_budget,
)
from hydra_suite.runtime.cuda_devices import CudaDevice, resolve_gpu_selectors
from hydra_suite.runtime.gpu_phase_lock import ENV_VARS as _GPU_PHASE_ENV_VARS
from hydra_suite.runtime.gpu_phase_lock import phase_lock_env
from hydra_suite.trackerkit.batch_plan import BatchJobSpec
from hydra_suite.utils.video_artifacts import (
    build_video_log_dir,
    choose_writable_artifact_base_dir,
)

#: Windows has no SIGKILL; TerminateProcess (what Popen.kill/terminate both use
#: there) is already a hard kill, so SIGTERM is the faithful stand-in.
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _is_windows() -> bool:
    return os.name == "nt"


def _kill_tree_windows(pid: int) -> bool:
    """Windows: hard-kill ``pid`` and every live descendant (no process groups).

    POSIX reaches grandchildren (the SLEAP service under ``conda run``) through
    the child's session group; Windows has no equivalent, so the tree is walked
    while the leader is still alive. Returns False if ``pid`` is already gone --
    its orphaned descendants are then unreachable by ancestry (a documented
    Windows limitation of this best-effort cleanup).
    """
    import psutil

    try:
        leader = psutil.Process(pid)
        victims = leader.children(recursive=True) + [leader]
    except psutil.Error:
        return False
    for proc in victims:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(victims, timeout=5)
    return True


logger = logging.getLogger(__name__)

_PROGRESS_RE = re.compile(
    r"\[(?:track forward|track backward|post)\] (\d{1,3})% ?(.*)$"
)
_SUMMARY_RE = re.compile(r"Tracker CLI completed: (.*)$")


@dataclass
class FanoutOptions:
    gpus: list[CudaDevice] = field(default_factory=list)
    # ``None`` means "unspecified": every GPU slot when ``gpus`` is set,
    # otherwise a single slot.
    jobs: Optional[int] = None
    # Concurrent children per selected GPU; ignored without ``gpus``. With
    # ``max_jobs`` it is the number of concurrent GPU *phases* per GPU instead.
    jobs_per_gpu: int = 1
    # Total concurrent children (needs ``gpus``). ``None`` keeps the slot
    # layout above. Above GPUs x jobs_per_gpu, children share a GPU's
    # jobs_per_gpu phase slots through a file-lock semaphore.
    max_jobs: Optional[int] = None
    # Keep launching queued videos after a child fails (default: halt).
    keep_going: bool = False
    # Launch the longest videos first (see :func:`order_longest_first`);
    # ``video_frames`` overrides the frame-count probe (tests).
    longest_first: bool = False
    video_frames: Optional[Callable[[str], Optional[int]]] = None
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
    """Scheduler callbacks. NOT all delivered on the same thread.

    ``job_progress`` and ``job_log`` fire on that job's own log-reader thread
    (one per running child), while ``job_started`` and ``job_finished`` fire on
    the thread that called :func:`run_batch_fanout`. Implementations must
    therefore never touch UI objects directly -- marshal to the GUI thread
    (e.g. a queued Qt signal) instead.
    """

    def job_started(  # noqa: E704
        self, spec: BatchJobSpec, gpu: Optional[CudaDevice], log_path: Path
    ) -> None: ...

    def job_progress(  # noqa: E704
        self, spec: BatchJobSpec, percent: int, message: str
    ) -> None: ...

    def job_log(self, spec: BatchJobSpec, line: str) -> None: ...  # noqa: E704

    def job_finished(self, result: FanoutJobResult) -> None: ...  # noqa: E704


class LiveChildRegistry:
    """The pids of the children currently running, readable from any thread.

    ``run_batch_fanout`` owns its ``running`` list on the scheduler thread, so
    nobody else can see it. That is fine for the cooperative stop, which the
    scheduler performs itself -- but a caller that has ALREADY waited out the
    stop budget (the GUI's close path) has no way to reach the survivors, and
    proceeding without one orphans children that still hold their GPUs.

    Registration is bounded by the child's lifetime: a pid is added when it is
    launched and dropped the moment it is reaped, so ``kill_all`` can never
    signal a recycled pid belonging to somebody else.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pids: set[int] = set()

    def add(self, pid: int) -> None:
        with self._lock:
            self._pids.add(int(pid))

    def discard(self, pid: int) -> None:
        with self._lock:
            self._pids.discard(int(pid))

    def pids(self) -> list[int]:
        with self._lock:
            return sorted(self._pids)

    def kill_all(self) -> list[int]:
        """SIGKILL every live child's whole session group; return the pids hit.

        No grace period and no escalation ladder: this is the last resort, for
        a caller that has already spent the cooperative budget. The group --
        not the pid -- because the child spawns grandchildren (the SLEAP
        service via ``conda run``), and killing only the leader would leave
        those running.
        """
        killed: list[int] = []
        for pid in self.pids():
            if _is_windows():
                if _kill_tree_windows(pid):
                    killed.append(pid)
                else:
                    self.discard(pid)
                continue
            try:
                os.killpg(pid, signal.SIGKILL)
                killed.append(pid)
            except (ProcessLookupError, PermissionError, OSError):
                # Already gone: the normal case for a child that exited while
                # the caller was deciding.
                self.discard(pid)
        return killed


class NullEvents:
    def job_started(self, spec, gpu, log_path) -> None: ...  # noqa: E704

    def job_progress(self, spec, percent, message) -> None: ...  # noqa: E704

    def job_log(self, spec, line) -> None: ...  # noqa: E704

    def job_finished(self, result) -> None: ...  # noqa: E704


def host_has_cuda() -> bool:
    """True when THIS host can run CUDA, independently of ``nvidia-smi``.

    Imported lazily: ``gpu_utils`` pulls in torch, and this module is otherwise
    import-cheap. Used only to tell "no GPUs because there are none" apart from
    "no GPUs because nvidia-smi is broken".
    """
    try:
        from hydra_suite.utils.gpu_utils import CUDA_AVAILABLE, TORCH_CUDA_AVAILABLE

        return bool(CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE)
    except Exception:  # noqa: BLE001 - a missing torch just means "no CUDA"
        return False


def decide_gpu_slots(
    selectors: Sequence[str],
    devices: Sequence[CudaDevice],
    host_has_cuda: bool,
) -> list[CudaDevice]:
    """Turn a user GPU selection into the devices to pin children to.

    Pure (availability is injected) and shared by the CLI and the GUI so the two
    cannot drift: the GUI used to guard the whole resolution with ``if
    available:`` and silently run UNPINNED whenever ``nvidia-smi`` returned
    nothing -- on a CUDA host every child then falls back to ``cuda:0`` and
    contends for one device, which is exactly what the pinning exists to
    prevent.

    - no selectors: no request, run unpinned;
    - devices visible: resolve the selectors against them (raises on an unknown
      or duplicated selector);
    - nothing visible on a CUDA-capable host: ``nvidia-smi`` is missing, masked
      or timing out -- refuse;
    - nothing visible on a host with no CUDA at all: ``auto`` is best-effort and
      runs unpinned, while an explicitly named device is still an error.

    Raises ``ValueError``; callers surface it (CLI: propagate, GUI: warn+abort).
    """
    wanted = [str(sel).strip() for sel in selectors if str(sel).strip()]
    if not wanted:
        return []
    if devices:
        return resolve_gpu_selectors(wanted, devices)
    if host_has_cuda:
        raise ValueError(
            "nvidia-smi listed no GPUs, but this host has CUDA available to "
            "torch -- nvidia-smi is missing, masked or timing out. Refusing to "
            "run unpinned: every child would fall back to cuda:0 and contend "
            "for the same device. Fix nvidia-smi or clear the GPU selection."
        )
    if wanted == ["auto"]:
        return []
    raise ValueError(
        "no CUDA devices visible to nvidia-smi; --gpus needs an NVIDIA host"
    )


def build_gpu_slots(
    gpus: Sequence[CudaDevice],
    *,
    jobs: Optional[int],
    jobs_per_gpu: int = 1,
) -> list[Optional[CudaDevice]]:
    """The scheduler's slots: each GPU ``jobs_per_gpu`` times, capped by ``jobs``.

    The single source of the slot arithmetic for the scheduler, the CLI and the
    GUI. Slots go round-robin over the GPUs (``g0, g1, g0, g1``) so a ``jobs``
    cap below ``GPUs x K`` spreads across every GPU before doubling up on one.
    At ``jobs_per_gpu=1`` this is exactly ``gpus[:jobs]``. ``jobs=None`` means
    every slot; without GPUs it is ``jobs`` unpinned slots (default 1).
    """
    k = int(jobs_per_gpu)
    if k < 1:
        raise ValueError(f"jobs_per_gpu must be >= 1, got {jobs_per_gpu}")
    if not gpus:
        return [None] * max(1, int(jobs or 1))
    slots: list[Optional[CudaDevice]] = list(gpus) * k
    requested = len(slots) if jobs is None else int(jobs)
    return slots[: max(1, min(requested, len(slots)))]


def scheduler_slots(options: FanoutOptions) -> list[Optional[CudaDevice]]:
    """The child slots :func:`run_batch_fanout` opens for ``options``.

    Without ``max_jobs`` (or without GPUs) this is exactly
    :func:`build_gpu_slots`. With it, ``max_jobs`` slots go round-robin over
    the GPUs, so each GPU hosts an even share of the children; past GPUs x
    ``jobs_per_gpu`` that share exceeds ``jobs_per_gpu`` and the GPU phase
    semaphore (see :func:`phase_lock_slots`) keeps the GPU work bounded.
    """
    k = int(options.jobs_per_gpu)
    if options.gpus and options.max_jobs is not None:
        m = max(1, int(options.max_jobs))
        stride = max(k, -(-m // len(options.gpus)))
        return build_gpu_slots(options.gpus, jobs=m, jobs_per_gpu=stride)
    return build_gpu_slots(options.gpus, jobs=options.jobs, jobs_per_gpu=k)


def phase_lock_slots(
    options: FanoutOptions, slots: Sequence[Optional[CudaDevice]]
) -> Optional[int]:
    """Per-GPU phase slots K when children must share them, else None.

    Only when more children are open than GPUs x ``jobs_per_gpu``: at or
    below that every child already has a slot of its own, and the children
    get no lock env at all (the default path stays byte-identical).
    """
    k = max(1, int(options.jobs_per_gpu))
    if options.gpus and len(slots) > len(options.gpus) * k:
        return k
    return None


def video_frame_count(video_path: str) -> Optional[int]:
    """The container's frame count, or ``None`` when it cannot be read."""
    try:
        # Lazy: cli_config pulls in cv2, and this module is otherwise cheap.
        from hydra_suite.trackerkit.cli_config import probe_video

        return probe_video(video_path).total_frames
    except Exception:  # noqa: BLE001 - an unreadable video just has no length
        return None


def _frames_to_track(
    spec: BatchJobSpec, frame_count: Callable[[str], Optional[int]]
) -> Optional[int]:
    """Frames the child will track: the video clipped to the config's range."""
    try:
        total = frame_count(spec.video_path)
    except Exception:  # noqa: BLE001 - treat a failing probe as "unknown"
        total = None
    if not total or int(total) <= 0:
        return None
    last = int(total) - 1
    cfg = spec.config or {}
    try:
        start = max(0, int(cfg.get("start_frame") or 0))
        end = cfg.get("end_frame")
        end = last if end is None else min(int(end), last)
    except (TypeError, ValueError):
        return int(total)
    return max(0, end - start + 1)


def order_longest_first(
    specs: Sequence[BatchJobSpec],
    frame_count: Callable[[str], Optional[int]] = video_frame_count,
) -> list[BatchJobSpec]:
    """Specs reordered so the most frames launch first (largest-job-first).

    A long video started last becomes the batch's tail, with every other
    slot idle; starting it first lets the short ones fill in around it.
    Videos whose length cannot be read go first: they may be long, and if
    they are unreadable the child fails fast instead of after hours of work.
    Ties keep input order. ``index`` travels with each spec unchanged.
    """
    lengths = [_frames_to_track(spec, frame_count) for spec in specs]
    order = sorted(
        range(len(specs)),
        key=lambda i: (lengths[i] is not None, -(lengths[i] or 0), i),
    )
    return [specs[i] for i in order]


def suggested_threads_per_job(n_slots: int, cpu_count: Optional[int]) -> int:
    """An even split of the host's cores across concurrent children (>= 1)."""
    return max(1, int(cpu_count or 1) // max(1, int(n_slots)))


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
        # The child sizes itself from its own cap, never the box-wide budget.
        env = capped_child_env(env, int(threads_per_job))
    return env


def resolve_slot_count(options: FanoutOptions) -> int:
    """Concurrent child slots the scheduler opens for ``options``.

    Delegates to :func:`scheduler_slots` (the single slot authority), so
    ``jobs_per_gpu`` is counted: unspecified jobs -> GPUs x K, an explicit
    jobs is clamped to it; without GPUs, ``jobs`` unpinned slots; with
    ``max_jobs`` and GPUs, ``max_jobs``.
    """
    return len(scheduler_slots(options))


def plan_fanout_cpu(
    options: FanoutOptions,
    *,
    n_videos: int,
    cpu_budget: Optional[int],
    env: Optional[Mapping[str, str]] = None,
    usable: Optional[int] = None,
) -> tuple[FanoutOptions, CpuPlan]:
    """Fit the slot count into the CPU budget and size each child's threads."""
    budget = resolve_cpu_budget(cpu_budget, env=env, usable=usable)
    plan = plan_cpu(
        jobs=resolve_slot_count(options),
        n_videos=n_videos,
        budget=budget,
        gpu_bound=bool(options.gpus),
        threads_per_job=options.threads_per_job,
    )
    if options.gpus and options.max_jobs is not None:
        # max_jobs is the slot authority here; the CPU clamp applies to it.
        return (
            replace(
                options,
                max_jobs=plan.jobs,
                jobs=min(plan.jobs, int(options.jobs or plan.jobs)),
                threads_per_job=plan.threads_per_job,
            ),
            plan,
        )
    return replace(options, jobs=plan.jobs, threads_per_job=plan.threads_per_job), plan


def describe_cpu_plan(
    plan: CpuPlan, budget: CpuBudget, requested_jobs: int, *, n_gpus: int = 0
) -> str:
    """One-line human summary of a fan-out CPU plan (for the GUI label).

    With ``n_gpus`` and more jobs than GPUs (``jobs_per_gpu`` > 1 after the
    clamp), it says how many children share each GPU.
    """
    text = (
        f"{plan.jobs} job(s) x {plan.threads_per_job} thread(s) "
        f"of {budget.cores} core(s)"
    )
    if n_gpus and plan.jobs > int(n_gpus):
        per_gpu = -(-int(plan.jobs) // int(n_gpus))
        text += f" (up to {per_gpu} per GPU on {int(n_gpus)} GPU(s))"
    if plan.clamped:
        text += f" — reduced from {int(requested_jobs)} to fit the CPU budget"
    return text


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
    # The timestamp has 1 s resolution, so it alone lets two fan-outs of the
    # same video append into one file; the pid + job index make it unique.
    return (
        log_dir / f"{Path(spec.video_path).stem}_fanout_{timestamp}"
        f"_p{os.getpid()}_job{spec.index}.log"
    )


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
    child_registry: Optional["LiveChildRegistry"] = None,
    phase_lock: Optional[tuple[Path, int]] = None,
) -> _Live:
    config_json = run_dir / f"job_{spec.index}_config.json"
    config_json.write_text(json.dumps(spec.config, indent=2), encoding="utf-8")
    command = (
        options.child_command
        or (lambda s, c: default_child_command(s, c, log_level=options.log_level))
    )(spec, config_json)
    env = build_child_env(os.environ, gpu=gpu, threads_per_job=options.threads_per_job)
    if phase_lock is not None and gpu is not None:
        # Children outnumber this GPU's slots: they take a slot only around
        # their GPU phases. Never inherit a stale semaphore from our own env.
        for var in _GPU_PHASE_ENV_VARS:
            env.pop(var, None)
        env.update(phase_lock_env(phase_lock[0], gpu.uuid, phase_lock[1]))
    log_path = _job_log_path(spec, timestamp)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"# trackerkit fan-out job {spec.index}: {spec.video_path}\n"
        f"# gpu={gpu.uuid if gpu else '<inherited>'} "
        + (f"gpu_phase_slots={phase_lock[1]} " if phase_lock and gpu else "")
        + f"command={_command_for_log(command)}\n"
    )
    log_handle.flush()
    popen_kwargs: dict = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if _is_windows():
        # Its own console process group, so CTRL_BREAK_EVENT in _stop_children
        # reaches this child alone -- not the scheduler and every sibling.
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    # Popen is LAST: everything that can fail has already succeeded, so a raise
    # before this point cannot leak a process. Anything after it must tear the
    # process back down rather than abandon it.
    try:
        proc = subprocess.Popen(command, **popen_kwargs)
    except BaseException:
        log_handle.close()
        raise
    # Register IMMEDIATELY: from the instant Popen returns there is a process
    # to kill, and anything between here and the caller's bookkeeping is a
    # window in which a hard kill from another thread would miss it.
    if child_registry is not None:
        child_registry.add(proc.pid)
    try:
        live = _Live(
            spec=spec,
            gpu=gpu,
            proc=proc,
            log_path=log_path,
            log_handle=log_handle,
            started_at=time.monotonic(),
        )
        live.reader = threading.Thread(
            target=_pump,
            args=(live, events),
            name=f"fanout-log-{spec.index}",
            daemon=True,
        )
        live.reader.start()
        events.job_started(spec, gpu, log_path)
    except BaseException:
        # The process is already running; never abandon it.
        _signal_group(proc, _SIGKILL)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        try:
            log_handle.close()
        except Exception:
            pass
        # This pid is dead and never became a job the caller knows about, so it
        # must leave the registry with it.
        if child_registry is not None:
            child_registry.discard(proc.pid)
        raise
    logger.info(
        "Fan-out: launched job %d (%s) on %s -> %s",
        spec.index,
        Path(spec.video_path).name,
        gpu.uuid if gpu else "inherited device",
        log_path,
    )
    return live


def _reap_process_group(proc: subprocess.Popen) -> None:
    """Take down whatever is left of a terminal child's session group.

    ``start_new_session=True`` in :func:`_launch` makes the child a session
    leader, so its pgid IS its pid -- and ``killpg`` still reaches the group
    after ``poll()`` has reaped the leader itself. Without this, a child that
    dies on its own (crash, OOM-kill, ``os._exit``) leaves its grandchildren --
    notably the SLEAP service spawned via ``conda run -n sleap`` -- running
    forever, because ``_stop_children`` only ever signals children that are
    still ``_alive()``.

    ``ProcessLookupError`` on the first signal means the group is already empty,
    which is the normal, orderly case.
    """
    if os.name == "nt":  # pragma: no cover - POSIX-only process groups
        return
    pid = proc.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return  # nothing left in the group (the usual clean exit)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return
        time.sleep(0.02)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _finish(live: _Live, *, cancelled: bool) -> FanoutJobResult:
    # BEFORE the reader join: a surviving grandchild holds the inherited stdout
    # pipe open, so the reader thread cannot see EOF until the group is gone.
    _reap_process_group(live.proc)
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

    Windows has no process groups to signal: the live tree is walked and
    hard-killed instead (TerminateProcess is the only "signal" there anyway).
    """
    if _is_windows():
        if not _kill_tree_windows(proc.pid):
            try:
                proc.kill()
            except Exception:
                pass
        return
    try:
        # ``proc.pid`` IS the pgid (start_new_session=True), and unlike
        # ``os.getpgid(pid)`` it still works once poll() has reaped the
        # leader -- so an already-exited child's group is still reachable.
        os.killpg(proc.pid, sig)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass  # group already gone or not ours: fall back to the child
    try:
        if sig == _SIGKILL:
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
            # Windows: _launch starts each child in its own process group
            # (CREATE_NEW_PROCESS_GROUP), so CTRL_BREAK_EVENT reaches that child
            # only; trackerkit's CLI traps the resulting SIGBREAK as a stop.
            live.proc.send_signal(
                signal.CTRL_BREAK_EVENT if _is_windows() else signal.SIGINT
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
        _signal_group(live.proc, _SIGKILL)
    for live in running:
        try:
            live.proc.wait(timeout=5)
        except Exception:
            pass
        # Same guarantee as the terminal transition in _finish: on the
        # scheduler-raise path _finish is never called, so reap the group here
        # too rather than leave grandchildren behind.
        _reap_process_group(live.proc)


def run_batch_fanout(
    specs: Sequence[BatchJobSpec],
    options: FanoutOptions,
    *,
    events: Optional[FanoutEvents] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    child_registry: Optional[LiveChildRegistry] = None,
) -> FanoutResult:
    """Run every spec as a child process across the configured slots.

    ``child_registry``, when given, is kept in step with the live children so a
    caller on another thread can hard-kill them (see :class:`LiveChildRegistry`).
    """
    events = events or NullEvents()
    should_stop = should_stop or (lambda: False)
    specs = list(specs)
    if not specs:
        return FanoutResult(jobs=[], cancelled=False)
    # Indices are 1-based and must be unique: they name the per-job effective
    # config file. Callers that leave them unset (or collide) get reindexed.
    if any(s.index < 1 for s in specs) or len({s.index for s in specs}) != len(specs):
        specs = [replace(s, index=i) for i, s in enumerate(specs, 1)]

    # Unspecified jobs -> every GPU slot; an explicit jobs is clamped to
    # GPUs x jobs_per_gpu, so a device is shared only when asked for.
    slots = scheduler_slots(options)
    lock_k = phase_lock_slots(options, slots)
    # Keyed on the slots actually opened: a --jobs cap can leave one child
    # per GPU even at jobs_per_gpu > 1, and then nothing is shared.
    if options.gpus and len(slots) > len(options.gpus) and not options.threads_per_job:
        logger.warning(
            "Fan-out: %d children across %d GPU(s) without --threads-per-job "
            "risks CPU oversubscription -- every child starts one "
            "OpenMP/BLAS/torch thread per core. Consider --threads-per-job %d "
            "(cpu_count / concurrent children).",
            len(slots),
            len(options.gpus),
            suggested_threads_per_job(len(slots), os.cpu_count()),
        )

    owns_run_dir = not options.run_dir
    run_dir = (
        Path(options.run_dir)
        if options.run_dir
        else Path(tempfile.mkdtemp(prefix="trackerkit-fanout-"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Microseconds, not seconds: two fan-outs launched from the same GUI within
    # one second would otherwise share (and append to) one per-job log file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    phase_lock: Optional[tuple[Path, int]] = None
    if lock_k is not None:
        lock_dir = run_dir / "gpu_phase_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        phase_lock = (lock_dir, lock_k)
        logger.info(
            "Fan-out: %d children on %d GPU(s); each GPU runs at most %d GPU "
            "phase(s) at once (locks in %s)",
            len(slots),
            len(options.gpus),
            lock_k,
            lock_dir,
        )

    pending = list(specs)
    if options.longest_first:
        logger.info(
            "Fan-out: reading frame counts of %d video(s) for longest-first",
            len(specs),
        )
        pending = order_longest_first(specs, options.video_frames or video_frame_count)
        logger.info(
            "Fan-out: longest first -> %s",
            ", ".join(Path(s.video_path).name for s in pending),
        )
    running: list[_Live] = []
    finished: dict[int, FanoutJobResult] = {}
    free_slots: list[Optional[CudaDevice]] = list(slots)
    halted = False
    cancelled = False

    def _forget(live: _Live) -> None:
        """Drop a reaped child from the registry (pid reuse safety)."""
        if child_registry is not None:
            child_registry.discard(live.proc.pid)

    # A raise from anywhere in the scheduling loop (an unexpected bug, a
    # should_stop callback that throws, KeyboardInterrupt from the parent's
    # own SIGINT) must not leave GPU children running: they hold whole
    # devices and would outlive the process that owns them.
    try:
        while pending or running:
            if should_stop():
                cancelled = True
                # Snapshot BEFORE signalling: a child that had already exited on its
                # own is reported on its own merits, not as a casualty of the stop.
                already_exited = {id(live): live.proc.poll() for live in running}
                _stop_children(running, options)
                for live in running:
                    finished_on_its_own = already_exited.get(id(live)) is not None
                    res = _finish(live, cancelled=not finished_on_its_own)
                    _forget(live)
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
                    _forget(live)
                    finished[live.spec.index] = res
                    events.job_finished(res)
                    if not res.success:
                        if options.keep_going:
                            logger.error(
                                "Fan-out: job %d failed (%s); continuing with the "
                                "queue (--keep-going)",
                                live.spec.index,
                                res.error,
                            )
                        else:
                            halted = True
                            logger.error(
                                "Fan-out: job %d failed (%s); no further jobs "
                                "will launch",
                                live.spec.index,
                                res.error,
                            )

            # launch
            while pending and free_slots and not halted:
                spec = pending.pop(0)
                gpu = free_slots.pop(0)
                try:
                    running.append(
                        _launch(
                            spec,
                            gpu,
                            options,
                            run_dir,
                            timestamp,
                            events,
                            child_registry=child_registry,
                            phase_lock=phase_lock,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    # A launch that never got off the ground must not abandon the
                    # children already running: record it, halt new launches, and
                    # let the loop drain the survivors normally.
                    logger.exception(
                        "Fan-out: could not launch job %d (%s)",
                        spec.index,
                        spec.video_path,
                    )
                    free_slots.append(gpu)
                    try:
                        failed_log = _job_log_path(spec, timestamp)
                    except Exception:
                        failed_log = run_dir / f"job_{spec.index}_not_started.log"
                    res = FanoutJobResult(
                        spec=spec,
                        gpu=gpu,
                        returncode=None,
                        success=False,
                        log_path=failed_log,
                        summary_lines=[],
                        error=f"launch failed: {exc}",
                        wall_s=0.0,
                    )
                    finished[spec.index] = res
                    events.job_finished(res)
                    if not options.keep_going:
                        halted = True

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
        outcome = FanoutResult(jobs=results, cancelled=cancelled)
        if owns_run_dir:
            # The effective per-job configs are provenance for a run that went
            # wrong; a clean run has nothing to explain, so don't leak a temp
            # dir per fan-out.
            if outcome.success:
                shutil.rmtree(run_dir, ignore_errors=True)
            else:
                logger.info("Fan-out: effective job configs kept at %s", run_dir)
        return outcome
    except BaseException:
        _stop_children(running, options)
        for live in running:
            _forget(live)
        raise
