# TrackerKit Batch GPU Fan-out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a TrackerKit batch of N videos as N child processes, one per GPU, from both `trackerkit track` and the GUI Batch panel, with output byte-identical to the sequential path.

**Architecture:** A Qt-free planner (`batch_plan.py`) produces one effective config per video and is consumed by both the existing sequential loop and a new Qt-free subprocess scheduler (`batch_fanout.py`). Each child is `python -m hydra_suite.trackerkit.app track <video> --config <json>` pinned via `CUDA_VISIBLE_DEVICES=<uuid>`. A blocking file lock guards every first-run artifact build so concurrent children cannot corrupt shared engines. The GUI adds a `BaseWorker` that maps scheduler callbacks to signals and a `BaseDialog` job table.

**Tech Stack:** Python 3.13, `subprocess`, `fcntl`/`msvcrt`, `nvidia-smi`, PySide6 (GUI only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-batch-gpu-fanout-design.md`

## Global Constraints

- Work in the worktree `.worktrees/batch-fanout` on branch `feat/batch-gpu-fanout`. Run tests with `conda activate hydra-mps` and `PYTHONPATH=$PWD/src` from the worktree root (the editable install points at the main checkout, not the worktree).
- Run a single test file at a time: `python -m pytest tests/test_<name>.py -q`. Never run the whole suite (known hangs).
- No PySide6 import in `trackerkit/batch_plan.py`, `trackerkit/batch_fanout.py`, `trackerkit/cli.py`, `runtime/artifact_lock.py`, `runtime/cuda_devices.py`. `tests/test_trackerkit_cli_cutover.py::test_cli_module_imports_no_qt` guards `cli.py`; add the same AST check for the new modules.
- Fan-out settings (`batch_parallel*`) are session state on `TrackerConfig` only and must never appear in `ConfigOrchestrator.build_config_dict()` output or in the per-child config JSON.
- The child command never carries `--gpus`, `--jobs`, or `--threads-per-job`.
- Gating rule: fan-out engages iff `--gpus` is given or `--jobs > 1`; otherwise `run_tracking_cli` is byte-for-byte the existing in-process sequential path.
- Thread-cap env vars are opt-in (`--threads-per-job`), never set by default.
- Commit after every task with a conventional-commit message. Do not add a `Co-Authored-By` trailer (user preference).
- Before any equivalence run: kill stale `sleap`/`hydra` processes only, clear `find . -name __pycache__ -path '*hydra_suite*' -exec rm -rf {} +` in the worktree, and `export KMP_DUPLICATE_LIB_OK=TRUE`.

---

### Task 1: Blocking artifact build lock

**Files:**
- Create: `src/hydra_suite/runtime/artifact_lock.py`
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py:719-752` (the `_artifact_is_fresh` / `_export_artifact` block inside `_load_direct_executor`)
- Modify: `src/hydra_suite/core/individual/pose/backends/sleap.py:947-1010` (`auto_export_sleap_model`)
- Modify: `src/hydra_suite/core/individual/pose/backends/sleap.py:415-440` (`_init_tensorrt_runner`, both `_build_trt_engine_from_onnx` calls)
- Modify: `src/hydra_suite/core/individual/pose/runtime/onnx_session.py:112`
- Modify: `src/hydra_suite/core/individual/classification/backend.py:821`
- Test: `tests/test_artifact_lock.py`

**Interfaces:**
- Produces: `hydra_suite.runtime.artifact_lock.artifact_build_lock(target: Path | str, *, timeout_s: float | None = None) -> ContextManager[None]`; `hydra_suite.runtime.artifact_lock.ArtifactLockTimeout(RuntimeError)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_artifact_lock.py
"""Cross-process blocking lock used around first-run artifact builds."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from hydra_suite.runtime.artifact_lock import ArtifactLockTimeout, artifact_build_lock


def test_lock_file_lives_beside_target(tmp_path):
    target = tmp_path / "model_b1.engine"
    with artifact_build_lock(target):
        assert (tmp_path / "model_b1.engine.lock").exists()


def test_lock_is_reentrant_across_sequential_uses(tmp_path):
    target = tmp_path / "x.engine"
    with artifact_build_lock(target):
        pass
    with artifact_build_lock(target):
        pass  # second acquisition must not block or raise


def test_second_process_blocks_until_first_releases(tmp_path):
    target = tmp_path / "shared.engine"
    holder = (
        "import sys, time\n"
        "from hydra_suite.runtime.artifact_lock import artifact_build_lock\n"
        "with artifact_build_lock(sys.argv[1]):\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(1.5)\n"
        "print('RELEASED', flush=True)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(target)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "LOCKED"
    t0 = time.monotonic()
    with artifact_build_lock(target):
        waited = time.monotonic() - t0
    proc.wait(timeout=10)
    assert waited >= 1.0, f"second holder did not block (waited {waited:.2f}s)"


def test_timeout_raises_when_held_elsewhere(tmp_path):
    target = tmp_path / "held.engine"
    holder = (
        "import sys, time\n"
        "from hydra_suite.runtime.artifact_lock import artifact_build_lock\n"
        "with artifact_build_lock(sys.argv[1]):\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(3)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(target)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "LOCKED"
    with pytest.raises(ArtifactLockTimeout):
        with artifact_build_lock(target, timeout_s=0.3):
            pass
    proc.wait(timeout=10)


def test_lock_survives_missing_parent_dir(tmp_path):
    target = tmp_path / "nested" / "deeper" / "model.engine"
    with artifact_build_lock(target):
        assert target.parent.is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_artifact_lock.py -q`
Expected: FAIL with `ModuleNotFoundError: hydra_suite.runtime.artifact_lock`

- [ ] **Step 3: Write the implementation**

```python
# src/hydra_suite/runtime/artifact_lock.py
"""Blocking cross-process lock for first-run artifact builds.

``HeavyJobLease`` (resource_lease.py) is deliberately non-blocking: it exists
to REFUSE a second heavy job. Artifact builds need the opposite -- the second
process must WAIT for the first to finish exporting, then re-check whether the
artifact now exists. This module provides that blocking primitive and nothing
else. It imports no torch/onnx so it is safe on every host.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator

fcntl: Any
try:
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

msvcrt: Any
try:
    import msvcrt as _msvcrt

    msvcrt = _msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

_POLL_SECONDS = 0.05


class ArtifactLockTimeout(RuntimeError):
    """Raised when ``timeout_s`` elapses before the lock is acquired."""


def lock_path_for(target: Path | str) -> Path:
    """Return ``<target>.lock`` beside the artifact (never inside it)."""
    target_path = Path(target)
    return target_path.parent / f"{target_path.name}.lock"


def _try_lock_nonblocking(handle: IO[str]) -> bool:
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    if msvcrt is not None:  # pragma: no cover - Windows
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    raise RuntimeError("this platform has no supported inter-process file lock")


def _unlock(handle: IO[str]) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def artifact_build_lock(
    target: Path | str, *, timeout_s: float | None = None
) -> Iterator[None]:
    """Hold an exclusive lock on ``<target>.lock`` for the ``with`` body.

    Blocks (polling) until acquired. ``timeout_s=None`` waits forever; a
    positive value raises :class:`ArtifactLockTimeout` on expiry. The lock
    inode is never deleted, so PID reuse or a crashed holder cannot leave a
    stale lock: the OS releases ``flock`` when the holder dies.

    Callers must use double-checked locking: check the artifact, acquire,
    RE-CHECK, then build. The second process that blocked here will find the
    first process's finished artifact on its re-check and skip the build.
    """
    path = lock_path_for(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    if msvcrt is not None and fcntl is None:  # pragma: no cover - Windows
        handle.seek(0)
        if not handle.read(1):
            handle.write("\0")
            handle.flush()
    deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
    locked = False
    try:
        while True:
            if _try_lock_nonblocking(handle):
                locked = True
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise ArtifactLockTimeout(
                    f"timed out after {timeout_s}s waiting for {path}"
                )
            time.sleep(_POLL_SECONDS)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
        except OSError:
            pass
        yield
    finally:
        try:
            if locked:
                _unlock(handle)
        finally:
            handle.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_artifact_lock.py -q`
Expected: 5 passed

- [ ] **Step 5: Apply the lock at the YOLO OBB export site**

In `src/hydra_suite/core/inference/runtime_artifacts.py`, add `from hydra_suite.runtime.artifact_lock import artifact_build_lock` to the imports, then replace the block in `_load_direct_executor` that starts with `if _artifact_is_fresh(` and ends with `logger.info("Exported %s OBB artifact: %s", runtime, artifact_path)` with:

```python
    def _fresh() -> bool:
        return _artifact_is_fresh(
            artifact_path,
            resolved,
            imgsz,
            batch_size=batch_size,
            enforce_trt_profile=True,
        )

    if _fresh():
        logger.info("Reusing cached %s OBB artifact: %s", runtime, artifact_path.name)
    else:
        if not auto_export:
            raise ArtifactExportError(
                f"compute_runtime={compute_runtime!r} requested but no fresh "
                f"{_artifact_suffix(runtime)} artifact exists for {resolved.name} "
                f"and auto_export=False. Provide a prebuilt "
                f"{_artifact_suffix(runtime)} (point model_path at it) or enable "
                f"auto_export (CUDA box) — refusing to silently fall back to "
                f"PyTorch (H4)."
            )
        # Concurrent fan-out children may all miss the cache at once;
        # serialize the build and re-check after the wait (double-checked).
        with artifact_build_lock(artifact_path):
            if _fresh():
                logger.info(
                    "Reusing %s OBB artifact built by another process: %s",
                    runtime,
                    artifact_path.name,
                )
            else:
                _export_artifact(
                    pt_path=resolved,
                    artifact_path=artifact_path,
                    runtime=runtime,
                    imgsz=imgsz,
                    batch_size=batch_size,
                )
                _write_fresh_marker(
                    artifact_path,
                    resolved,
                    imgsz,
                    batch_size=batch_size,
                    enforce_trt_profile=True,
                )
                logger.info("Exported %s OBB artifact: %s", runtime, artifact_path)
```

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_runtime_artifacts*.py tests/test_direct_obb*.py -q` (whichever exist; `ls tests | grep -i 'runtime_artifact\|direct_obb'`)
Expected: same pass count as before the edit.

- [ ] **Step 6: Apply the lock at the SLEAP export site**

In `src/hydra_suite/core/individual/pose/backends/sleap.py`, import `artifact_build_lock` and wrap the body of `auto_export_sleap_model` from the line `export_dir = model_path.parent / f"{model_path.name}.{runtime}"` down to the final `return str(export_dir.resolve())` in `with artifact_build_lock(export_dir):` (the lock file is `<export_dir>.lock` beside `model_path.parent`, which survives the `rmtree(export_dir)`). Keep the `artifact_meta_matches` early return INSIDE the lock so a waiting process re-checks before rebuilding. Indent the existing code one level; change nothing else.

- [ ] **Step 7: Apply the lock at the native SLEAP TRT build sites**

In `_init_tensorrt_runner` (same file), wrap each `_build_trt_engine_from_onnx(...)` call:

```python
                with artifact_build_lock(rebuilt):
                    built = rebuilt.exists() or _build_trt_engine_from_onnx(
                        onnx_path, rebuilt, fixed_hw=self._input_hw
                    )
                if built:
```

and

```python
            with artifact_build_lock(engine_path):
                built = engine_path.exists() or _build_trt_engine_from_onnx(
                    model_path, engine_path, fixed_hw=self._input_hw
                )
            if built:
```

(For the stale-engine rebuild branch `rebuilt` already exists and failed to deserialize, so use `built = _build_trt_engine_from_onnx(...)` without the `.exists()` short-circuit there.)

- [ ] **Step 8: Apply the lock around TensorRT-EP ONNX session creation**

In `src/hydra_suite/core/individual/pose/runtime/onnx_session.py`:

```python
from hydra_suite.runtime.artifact_lock import artifact_build_lock

def _has_trt_provider(providers) -> bool:
    return any(
        (p if isinstance(p, str) else p[0]) == "TensorrtExecutionProvider"
        for p in providers
    )
...
        providers = execution_providers_for(resolved)
        if _has_trt_provider(providers):
            # ORT builds its TRT engine into the shared per-machine cache dir
            # on first session creation; serialize concurrent builders.
            with artifact_build_lock(Path(str(model_path)).with_suffix(".trt_ep")):
                self._session = ort.InferenceSession(str(model_path), providers=providers)
        else:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
```

In `src/hydra_suite/core/individual/classification/backend.py` around line 821, apply the same shape to the first `ort.InferenceSession(str(peer), providers=providers)` call (the CPU fallback call stays unlocked). Reuse the `_has_trt_provider` helper by importing it from `onnx_session`.

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_classifier_backend*.py tests/test_sleap*.py tests/test_pose_runtime*.py -q` (use `ls tests | grep -i 'classifier_backend\|sleap\|pose_runtime'` to pick the files that exist)
Expected: same pass/fail set as on the branch base (record it first with `git stash` NOT allowed — instead run the same files once in the main checkout at `/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker` to get the baseline).

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/runtime/artifact_lock.py tests/test_artifact_lock.py \
  src/hydra_suite/core/inference/runtime_artifacts.py \
  src/hydra_suite/core/individual/pose/backends/sleap.py \
  src/hydra_suite/core/individual/pose/runtime/onnx_session.py \
  src/hydra_suite/core/individual/classification/backend.py
git commit -m "feat(runtime): blocking artifact build lock around every first-run engine export"
```

---

### Task 2: Physical CUDA device enumeration and selector parsing

**Files:**
- Create: `src/hydra_suite/runtime/cuda_devices.py`
- Test: `tests/test_cuda_devices.py`

**Interfaces:**
- Produces:
  - `CudaDevice(index: int, uuid: str, name: str)` frozen dataclass
  - `list_cuda_devices(*, runner=None) -> list[CudaDevice]` (runs `nvidia-smi --query-gpu=index,uuid,name,mig.mode.current --format=csv,noheader,nounits`; `runner` is an injectable `Callable[[list[str]], str | None]` returning stdout or `None`; MIG-enabled rows are skipped; returns `[]` when nvidia-smi is missing or fails)
  - `parse_gpu_selectors(text: str) -> list[str]` (splits on commas, expands `a-b` ranges to ordinals, keeps other tokens verbatim; `"auto"` → `["auto"]`)
  - `resolve_gpu_selectors(selectors: Sequence[str], devices: Sequence[CudaDevice] | None = None) -> list[CudaDevice]` (raises `ValueError` on unknown ordinal, ambiguous or unknown UUID prefix, duplicate physical device, or when no devices exist)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cuda_devices.py
from __future__ import annotations

import pytest

from hydra_suite.runtime.cuda_devices import (
    CudaDevice,
    list_cuda_devices,
    parse_gpu_selectors,
    resolve_gpu_selectors,
)

_SMI = (
    "0, GPU-aaaa1111-0000-0000-0000-000000000000, NVIDIA RTX 6000 Ada Generation, Disabled\n"
    "1, GPU-bbbb2222-0000-0000-0000-000000000000, NVIDIA RTX 6000 Ada Generation, Disabled\n"
    "2, GPU-cccc3333-0000-0000-0000-000000000000, NVIDIA A100, Enabled\n"
)


def _fake_runner(_cmd):
    return _SMI


def test_list_devices_parses_and_skips_mig():
    devices = list_cuda_devices(runner=_fake_runner)
    assert [d.index for d in devices] == [0, 1]
    assert devices[0].uuid.startswith("GPU-aaaa1111")
    assert devices[1].name == "NVIDIA RTX 6000 Ada Generation"


def test_list_devices_empty_when_smi_missing():
    assert list_cuda_devices(runner=lambda _cmd: None) == []


def test_parse_selectors_expands_ranges_and_keeps_uuids():
    assert parse_gpu_selectors("0,2-4,GPU-abcd") == ["0", "2", "3", "4", "GPU-abcd"]
    assert parse_gpu_selectors(" auto ") == ["auto"]
    assert parse_gpu_selectors("") == []


def test_resolve_ordinals_and_uuid_prefix():
    devices = list_cuda_devices(runner=_fake_runner)
    picked = resolve_gpu_selectors(["1", "GPU-aaaa"], devices=devices)
    assert [d.index for d in picked] == [1, 0]


def test_resolve_auto_returns_all():
    devices = list_cuda_devices(runner=_fake_runner)
    assert resolve_gpu_selectors(["auto"], devices=devices) == devices


@pytest.mark.parametrize("bad", [["7"], ["GPU-zzzz"], ["0", "GPU-aaaa"], ["GPU-"]])
def test_resolve_rejects_unknown_ambiguous_and_duplicates(bad):
    devices = list_cuda_devices(runner=_fake_runner)
    with pytest.raises(ValueError):
        resolve_gpu_selectors(bad, devices=devices)


def test_resolve_with_no_devices_is_an_error():
    with pytest.raises(ValueError):
        resolve_gpu_selectors(["0"], devices=[])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_cuda_devices.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/hydra_suite/runtime/cuda_devices.py
"""Physical CUDA device enumeration for process-per-GPU fan-out.

Children are pinned with ``CUDA_VISIBLE_DEVICES=<uuid>``. UUIDs are used
because an ordinal in the child's mask names a PHYSICAL device regardless of
the parent's own mask, so an ordinal list computed under a parent mask would
be wrong. This module talks only to ``nvidia-smi`` (no torch/cupy import) so
it is cheap and safe on hosts without CUDA, where it simply returns ``[]``.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

_SMI_TIMEOUT_S = 5.0
_QUERY = "index,uuid,name,mig.mode.current"


@dataclass(frozen=True)
class CudaDevice:
    index: int
    uuid: str
    name: str


def _run_nvidia_smi(command: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=_SMI_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def list_cuda_devices(
    *, runner: Callable[[list[str]], Optional[str]] | None = None
) -> list[CudaDevice]:
    """Return every non-MIG physical GPU reported by nvidia-smi, index order."""
    run = runner or _run_nvidia_smi
    stdout = run(
        ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"]
    )
    if not stdout:
        return []
    devices: list[CudaDevice] = []
    for row in csv.reader(stdout.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            continue
        index, uuid, name, mig = (item.strip() for item in row)
        if mig.lower() not in {"disabled", "n/a", "[n/a]", "not supported", ""}:
            continue
        try:
            devices.append(CudaDevice(index=int(index), uuid=uuid, name=name))
        except ValueError:
            continue
    devices.sort(key=lambda d: d.index)
    return devices


def parse_gpu_selectors(text: str) -> list[str]:
    """Split ``"0,2-4,GPU-abc"`` into ``["0","2","3","4","GPU-abc"]``."""
    out: list[str] = []
    for raw in str(text or "").split(","):
        token = raw.strip()
        if not token:
            continue
        if token.lower() == "auto":
            return ["auto"]
        lo, sep, hi = token.partition("-")
        if sep and lo.isdigit() and hi.isdigit():
            a, b = int(lo), int(hi)
            if b < a:
                raise ValueError(f"bad GPU range {token!r}")
            out.extend(str(i) for i in range(a, b + 1))
        else:
            out.append(token)
    return out


def resolve_gpu_selectors(
    selectors: Sequence[str], devices: Sequence[CudaDevice] | None = None
) -> list[CudaDevice]:
    """Map ordinals / UUID prefixes / ``auto`` onto physical devices.

    Raises ``ValueError`` when nothing is available, a selector is unknown or
    ambiguous, or the same physical device is named twice.
    """
    available = list(list_cuda_devices() if devices is None else devices)
    if not available:
        raise ValueError(
            "no CUDA devices visible to nvidia-smi; --gpus needs an NVIDIA host"
        )
    wanted = list(selectors)
    if wanted == ["auto"]:
        return available
    picked: list[CudaDevice] = []
    for selector in wanted:
        sel = str(selector).strip()
        if sel.isdigit():
            matches = [d for d in available if d.index == int(sel)]
        elif sel.upper().startswith("GPU-") and len(sel) > 4:
            matches = [d for d in available if d.uuid.upper().startswith(sel.upper())]
        else:
            raise ValueError(
                f"GPU selector {sel!r} is neither an ordinal nor a GPU- UUID prefix"
            )
        if len(matches) != 1:
            raise ValueError(
                f"GPU selector {sel!r} matched {len(matches)} devices "
                f"(available: {[d.index for d in available]})"
            )
        if matches[0] in picked:
            raise ValueError(f"GPU {matches[0].index} selected more than once")
        picked.append(matches[0])
    return picked
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_cuda_devices.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/runtime/cuda_devices.py tests/test_cuda_devices.py
git commit -m "feat(runtime): nvidia-smi based physical CUDA device enumeration and selector parsing"
```

---

### Task 3: One batch planner for the sequential loop and the fan-out

**Files:**
- Create: `src/hydra_suite/trackerkit/batch_plan.py`
- Modify: `src/hydra_suite/trackerkit/cli.py:23-114` (`run_tracking_cli`)
- Test: `tests/test_trackerkit_batch_plan.py`

**Interfaces:**
- Consumes: `hydra_suite.trackerkit.session_plan.build_batch_video_plan`, `hydra_suite.trackerkit.cli_config.{load_tracker_cli_config, apply_sahi_profile_override, _default_output_paths}`
- Produces:
  - `BatchJobSpec(index: int, video_path: str, config_path: str | None, config: dict, provenance: str)` frozen dataclass (`provenance ∈ {"own-sidecar","explicit","keystone-baseline"}`)
  - `plan_batch_jobs(video_paths: Sequence[str], *, explicit_config_path: str | None = None, keystone_override: bool = False, sahi_profile: str | None = None) -> list[BatchJobSpec]`
  - `BatchPlanError(ValueError)` for duplicate videos / colliding outputs

- [ ] **Step 1: Write the failing tests (including the frozen reference loop)**

```python
# tests/test_trackerkit_batch_plan.py
"""plan_batch_jobs must hand every video the SAME config dict the pre-refactor
sequential loop in cli.py passed to load_tracker_cli_session."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from hydra_suite.trackerkit import batch_plan
from hydra_suite.trackerkit.batch_plan import BatchPlanError, plan_batch_jobs
from hydra_suite.trackerkit.cli_config import load_tracker_cli_config
from hydra_suite.trackerkit.session_plan import build_batch_video_plan


def _reference_loop(videos, *, config_path=None, keystone_override=False, sahi_profile=None):
    """Verbatim copy of the pre-refactor cli.run_tracking_cli config resolution."""
    from hydra_suite.trackerkit.cli_config import apply_sahi_profile_override

    plan = build_batch_video_plan(
        videos, explicit_config_path=config_path, keystone_override=keystone_override
    )
    out = []
    baseline = None
    for index, item in enumerate(plan, start=1):
        effective = None
        if item.use_keystone_baseline and item.config_path is None:
            effective = baseline or {}
        if sahi_profile:
            base = effective if effective is not None else load_tracker_cli_config(item.config_path)
            effective = apply_sahi_profile_override(base, sahi_profile)
        # what load_tracker_cli_session would deepcopy into session.config
        cfg = deepcopy(dict(effective)) if effective is not None else load_tracker_cli_config(item.config_path)
        if index == 1:
            baseline = (
                deepcopy(load_tracker_cli_config(item.config_path))
                if item.config_path
                else deepcopy(cfg)
            )
        out.append((item.video_path, cfg))
    return out


def _mk_video(tmp_path: Path, name: str, cfg: dict | None = None) -> str:
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"\x00")
    if cfg is not None:
        (tmp_path / f"{name}_config.json").write_text(json.dumps(cfg))
    return str(video)


@pytest.fixture(autouse=True)
def _no_sahi(monkeypatch):
    # apply_sahi_profile_override needs a real model sidecar; stub it to a
    # deterministic transform so the parity test exercises the plumbing.
    def _fake(cfg, profile):
        out = deepcopy(dict(cfg))
        out["SAHI_PROFILE"] = profile
        return out

    monkeypatch.setattr(batch_plan, "apply_sahi_profile_override", _fake)
    import hydra_suite.trackerkit.cli_config as cc

    monkeypatch.setattr(cc, "apply_sahi_profile_override", _fake)


@pytest.mark.parametrize("keystone_override", [False, True])
@pytest.mark.parametrize("sahi_profile", [None, "prof_a"])
def test_planner_matches_reference_loop_mixed_sidecars(tmp_path, keystone_override, sahi_profile):
    a = _mk_video(tmp_path, "a", {"k": "keystone"})
    b = _mk_video(tmp_path, "b", {"k": "own_b"})
    c = _mk_video(tmp_path, "c")  # no sidecar
    videos = [a, b, c]
    ref = _reference_loop(videos, keystone_override=keystone_override, sahi_profile=sahi_profile)
    specs = plan_batch_jobs(videos, keystone_override=keystone_override, sahi_profile=sahi_profile)
    assert [(s.video_path, s.config) for s in specs] == ref
    assert [s.index for s in specs] == [1, 2, 3]


def test_planner_matches_reference_loop_explicit_config(tmp_path):
    a = _mk_video(tmp_path, "a")
    b = _mk_video(tmp_path, "b", {"k": "own_b"})
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"k": "explicit"}))
    ref = _reference_loop([a, b], config_path=str(explicit))
    specs = plan_batch_jobs([a, b], explicit_config_path=str(explicit))
    assert [(s.video_path, s.config) for s in specs] == ref
    assert specs[0].provenance == "explicit"
    assert specs[1].provenance == "keystone-baseline"  # explicit implies override


def test_planner_no_sidecars_uses_empty_baseline(tmp_path):
    a = _mk_video(tmp_path, "a")
    b = _mk_video(tmp_path, "b")
    specs = plan_batch_jobs([a, b])
    assert specs[0].config == {} and specs[1].config == {}
    assert specs[0].config_path is None


def test_planner_rejects_duplicate_video(tmp_path):
    a = _mk_video(tmp_path, "a")
    with pytest.raises(BatchPlanError):
        plan_batch_jobs([a, a])


def test_planner_rejects_same_stem_when_outputs_collide(tmp_path, monkeypatch):
    a = _mk_video(tmp_path / "x", "same") if (tmp_path / "x").mkdir() is None else None
    b = _mk_video(tmp_path / "y", "same") if (tmp_path / "y").mkdir() is None else None
    # Force both raw CSVs to the same path (read-only dir redirect scenario).
    monkeypatch.setattr(
        batch_plan, "_default_output_paths", lambda v: (str(tmp_path / "same_tracking.csv"), "")
    )
    with pytest.raises(BatchPlanError):
        plan_batch_jobs([a, b])


def test_planner_does_not_mutate_returned_configs_across_jobs(tmp_path):
    a = _mk_video(tmp_path, "a", {"k": "keystone", "nested": {"v": 1}})
    b = _mk_video(tmp_path, "b")
    specs = plan_batch_jobs([a, b])
    specs[0].config["nested"]["v"] = 99
    assert specs[1].config["nested"]["v"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_batch_plan.py -q`
Expected: FAIL with `ModuleNotFoundError: hydra_suite.trackerkit.batch_plan`

- [ ] **Step 3: Write the planner**

```python
# src/hydra_suite/trackerkit/batch_plan.py
"""One planner that resolves the effective config for every video in a batch.

Both the in-process sequential loop (``cli.run_tracking_cli``) and the
process-per-GPU fan-out (``batch_fanout``) consume this, so the config a child
receives is -- by construction -- the config the sequential loop would have
used. Qt-free.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from hydra_suite.trackerkit.cli_config import (
    _default_output_paths,
    apply_sahi_profile_override,
    load_tracker_cli_config,
)
from hydra_suite.trackerkit.session_plan import build_batch_video_plan


class BatchPlanError(ValueError):
    """A batch cannot run: duplicate inputs or colliding outputs."""


@dataclass(frozen=True)
class BatchJobSpec:
    index: int  # 1-based; 1 is the keystone
    video_path: str
    config_path: str | None  # the sidecar/explicit file this config came from
    config: dict[str, Any]  # effective config; deep-copied per job
    provenance: str  # "own-sidecar" | "explicit" | "keystone-baseline"


def _reject_collisions(video_paths: Sequence[str]) -> None:
    seen_real: dict[str, str] = {}
    seen_raw_csv: dict[str, str] = {}
    for video in video_paths:
        real = os.path.realpath(video)
        if real in seen_real:
            raise BatchPlanError(
                f"video listed twice: {video} and {seen_real[real]}"
            )
        seen_real[real] = video
        raw_csv, _ = _default_output_paths(video)
        raw_key = os.path.realpath(raw_csv)
        if raw_key in seen_raw_csv:
            raise BatchPlanError(
                f"two videos would write the same output {raw_csv}: "
                f"{video} and {seen_raw_csv[raw_key]}"
            )
        seen_raw_csv[raw_key] = video


def plan_batch_jobs(
    video_paths: Sequence[str],
    *,
    explicit_config_path: str | None = None,
    keystone_override: bool = False,
    sahi_profile: str | None = None,
) -> list[BatchJobSpec]:
    """Resolve one ``BatchJobSpec`` per video with today's keystone rules.

    Semantics are exactly the pre-refactor ``cli.run_tracking_cli`` loop:
    video 1's resolved config (pre-SAHI-override when it came from a file)
    becomes the keystone baseline; a video with no config of its own inherits
    the baseline; ``sahi_profile`` is applied to every video.
    """
    videos = [str(v).strip() for v in video_paths if str(v).strip()]
    if not videos:
        raise BatchPlanError("at least one video path is required")
    _reject_collisions(videos)

    plan = build_batch_video_plan(
        videos,
        explicit_config_path=explicit_config_path,
        keystone_override=keystone_override,
    )
    specs: list[BatchJobSpec] = []
    baseline: dict[str, Any] | None = None
    for index, item in enumerate(plan, start=1):
        inherits = item.use_keystone_baseline and item.config_path is None
        if inherits:
            cfg: dict[str, Any] = deepcopy(baseline or {})
            provenance = "keystone-baseline"
        else:
            cfg = load_tracker_cli_config(item.config_path)
            provenance = (
                "explicit"
                if explicit_config_path
                and item.config_path
                and os.path.realpath(item.config_path)
                == os.path.realpath(explicit_config_path)
                else "own-sidecar"
            )
        if sahi_profile:
            cfg = apply_sahi_profile_override(cfg, sahi_profile)
        cfg = deepcopy(dict(cfg))
        if index == 1:
            baseline = (
                deepcopy(load_tracker_cli_config(item.config_path))
                if item.config_path
                else deepcopy(cfg)
            )
        specs.append(
            BatchJobSpec(
                index=index,
                video_path=item.video_path,
                config_path=item.config_path if not inherits else None,
                config=cfg,
                provenance=provenance,
            )
        )
    return specs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_batch_plan.py -q`
Expected: all passed. If `test_planner_rejects_same_stem_when_outputs_collide` fails on the fixture construction, simplify it to create the two directories explicitly before calling `_mk_video`.

- [ ] **Step 5: Rewrite the sequential loop in `cli.py` on top of the planner**

Replace the body of `run_tracking_cli` in `src/hydra_suite/trackerkit/cli.py` (keep the signature for now; Task 5 extends it):

```python
def run_tracking_cli(
    video_paths: Sequence[str],
    *,
    config_path: str | None = None,
    keystone_override: bool = False,
    sahi_profile: str | None = None,
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
    return _run_sequential(specs)


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
            # Persist the resolved keystone baseline for provenance/debugging.
            if spec.provenance == "keystone-baseline":
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
                    "Tracker CLI failed for %s: %s", spec.video_path, error_message
                )
                exit_code = 1
                break
    return exit_code
```

Update imports: add `from hydra_suite.trackerkit.batch_plan import BatchJobSpec, plan_batch_jobs`, keep `load_tracker_cli_session`, drop `load_tracker_cli_config`, `apply_sahi_profile_override`, `build_batch_video_plan`, and `deepcopy` if now unused.

- [ ] **Step 6: Run the existing CLI tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_cli_cutover.py tests/test_trackerkit_cli_sahi_profile.py tests/test_trackerkit_cli_config.py tests/test_trackerkit_session_plan.py tests/test_gui_cli_profile_parity.py -q`
Expected: same results as on the branch base. `test_trackerkit_cli_sahi_profile.py` may patch `hydra_suite.trackerkit.cli.apply_sahi_profile_override`; if so, change the patch target to `hydra_suite.trackerkit.batch_plan.apply_sahi_profile_override` and nothing else.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/batch_plan.py src/hydra_suite/trackerkit/cli.py tests/test_trackerkit_batch_plan.py tests/test_trackerkit_cli_sahi_profile.py
git commit -m "refactor(trackerkit): extract batch planner; sequential CLI loop consumes it"
```

---

### Task 4: Qt-free fan-out scheduler

**Files:**
- Create: `src/hydra_suite/trackerkit/batch_fanout.py`
- Test: `tests/test_trackerkit_batch_fanout.py`

**Interfaces:**
- Consumes: `BatchJobSpec` (Task 3), `CudaDevice` (Task 2), `hydra_suite.utils.video_artifacts.{build_video_log_dir, choose_writable_artifact_base_dir}`
- Produces:
  - `FanoutOptions(gpus: list[CudaDevice] = [], jobs: int = 1, threads_per_job: int | None = None, log_level: str = "INFO", run_dir: Path | None = None, child_command: Callable[[BatchJobSpec, Path], list[str]] | None = None, poll_s: float = 0.2, sigint_grace_s: float = 10.0, term_grace_s: float = 5.0)`
  - `FanoutJobResult(spec, gpu: CudaDevice | None, returncode: int | None, success: bool, log_path: Path, summary_lines: list[str], error: str | None, wall_s: float)`
  - `FanoutResult(jobs: list[FanoutJobResult], cancelled: bool)` with `.success` property (all jobs succeeded and not cancelled)
  - `FanoutEvents` Protocol: `job_started(spec, gpu, log_path)`, `job_progress(spec, percent: int, message: str)`, `job_log(spec, line: str)`, `job_finished(result)`
  - `NullEvents` (no-op implementation)
  - `build_child_env(base: Mapping[str,str], *, gpu: CudaDevice | None, threads_per_job: int | None) -> dict[str,str]`
  - `default_child_command(spec, config_json: Path, *, log_level: str) -> list[str]`
  - `parse_progress_line(line: str) -> tuple[int, str] | None`, `parse_summary_line(line: str) -> list[str] | None`
  - `run_batch_fanout(specs, options, *, events=None, should_stop=None) -> FanoutResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trackerkit_batch_fanout.py
"""Scheduler tests drive a FAKE child (a tiny python script) so they need no
models, videos, or GPUs. The fake prints the same progress/summary lines the
real child's logging emits."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit.batch_fanout import (
    FanoutOptions,
    build_child_env,
    default_child_command,
    parse_progress_line,
    parse_summary_line,
    run_batch_fanout,
)
from hydra_suite.trackerkit.batch_plan import BatchJobSpec

FAKE_CHILD = r'''
import json, os, signal, sys, time
cfg = json.load(open(sys.argv[1]))
mode = cfg.get("mode", "ok")
sleep = float(cfg.get("sleep", 0.05))
if mode == "trap_sigint":
    def _h(*_):
        print("SIGINT received - requesting clean stop", flush=True); sys.exit(130)
    signal.signal(signal.SIGINT, _h)
if mode == "ignore_signals":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("2026-01-01 - x - INFO - [track forward] 10% starting", flush=True)
print("GPU=" + os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), flush=True)
print("OMP=" + os.environ.get("OMP_NUM_THREADS", "<unset>"), flush=True)
time.sleep(sleep)
print("2026-01-01 - x - INFO - [post] 90% merging", flush=True)
if mode == "fail":
    print("2026-01-01 - x - ERROR - Tracker CLI failed for v: boom", flush=True)
    sys.exit(3)
if mode in ("trap_sigint", "ignore_signals"):
    time.sleep(30)
print("2026-01-01 - x - INFO - Tracker CLI completed: video=%s | rows=5 | avg_fps=9.0" % cfg["name"], flush=True)
'''


def _spec(tmp_path: Path, name: str, **cfg) -> BatchJobSpec:
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"\x00")
    return BatchJobSpec(
        index=0,
        video_path=str(video),
        config_path=None,
        config={"name": name, **cfg},
        provenance="own-sidecar",
    )


def _fake_command(spec: BatchJobSpec, config_json: Path) -> list[str]:
    return [sys.executable, "-c", FAKE_CHILD, str(config_json)]


class _Recorder:
    def __init__(self):
        self.started, self.progress, self.logs, self.finished = [], [], [], []

    def job_started(self, spec, gpu, log_path):
        self.started.append((spec.video_path, gpu, log_path))

    def job_progress(self, spec, percent, message):
        self.progress.append((spec.video_path, percent, message))

    def job_log(self, spec, line):
        self.logs.append((spec.video_path, line))

    def job_finished(self, result):
        self.finished.append(result)


def test_parse_progress_and_summary_lines():
    assert parse_progress_line("t - n - INFO - [track forward] 45% detecting") == (45, "detecting")
    assert parse_progress_line("t - n - INFO - [track backward] 7% x") == (7, "x")
    assert parse_progress_line("t - n - INFO - [post] 100% done") == (100, "done")
    assert parse_progress_line("random text") is None
    assert parse_summary_line("t - n - INFO - Tracker CLI completed: video=a | rows=5") == ["video=a", "rows=5"]
    assert parse_summary_line("t - n - INFO - other") is None


def test_build_child_env_pins_gpu_and_unbuffered_and_caps_only_when_unset():
    gpu = CudaDevice(index=3, uuid="GPU-1234", name="x")
    env = build_child_env({"PATH": "/bin", "OMP_NUM_THREADS": "7"}, gpu=gpu, threads_per_job=4)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-1234"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert env["OMP_NUM_THREADS"] == "7"          # parent's value wins
    assert env["NUMBA_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "4"
    assert env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["PATH"] == "/bin"


def test_build_child_env_without_gpu_or_caps_sets_nothing_extra():
    env = build_child_env({"PATH": "/bin"}, gpu=None, threads_per_job=None)
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "OMP_NUM_THREADS" not in env
    assert env["PYTHONUNBUFFERED"] == "1"


def test_default_child_command_shape(tmp_path):
    spec = _spec(tmp_path, "a")
    cmd = default_child_command(spec, tmp_path / "cfg.json", log_level="DEBUG")
    assert cmd[:3] == [sys.executable, "-m", "hydra_suite.trackerkit.app"]
    assert "track" in cmd and spec.video_path in cmd and "--config" in cmd
    assert cmd[cmd.index("--log-level") + 1] == "DEBUG"
    for forbidden in ("--gpus", "--jobs", "--threads-per-job"):
        assert forbidden not in cmd


def test_runs_all_jobs_and_reports_success(tmp_path):
    specs = [_spec(tmp_path, n) for n in ("a", "b", "c")]
    rec = _Recorder()
    result = run_batch_fanout(
        specs, FanoutOptions(jobs=2, run_dir=tmp_path / "run", child_command=_fake_command), events=rec
    )
    assert result.success and not result.cancelled
    assert [r.returncode for r in result.jobs] == [0, 0, 0]
    assert sorted(r.summary_lines[0] for r in result.jobs) == ["video=a", "video=b", "video=c"]
    assert len(rec.started) == 3 and len(rec.finished) == 3
    assert any(p == 90 for _, p, _ in rec.progress)
    for r in result.jobs:
        assert r.log_path.exists()
        assert "GPU=<unset>" in r.log_path.read_text()
    # per-job effective config was written for provenance
    assert sorted(p.name for p in (tmp_path / "run").glob("job_*_config.json")) == [
        "job_1_config.json", "job_2_config.json", "job_3_config.json"
    ]


def test_gpu_slots_pin_each_child(tmp_path):
    gpus = [CudaDevice(0, "GPU-aaaa", "x"), CudaDevice(1, "GPU-bbbb", "x")]
    specs = [_spec(tmp_path, n, sleep=0.3) for n in ("a", "b", "c", "d")]
    result = run_batch_fanout(
        specs, FanoutOptions(gpus=gpus, jobs=99, run_dir=tmp_path / "run", child_command=_fake_command)
    )
    assert result.success
    seen = sorted(r.log_path.read_text().split("GPU=")[1].split("\n")[0] for r in result.jobs)
    assert seen == ["GPU-aaaa", "GPU-aaaa", "GPU-bbbb", "GPU-bbbb"]  # jobs clamped to 2 slots, reused


def test_concurrency_respects_jobs(tmp_path):
    specs = [_spec(tmp_path, n, sleep=0.6) for n in ("a", "b", "c", "d")]
    t0 = time.monotonic()
    result = run_batch_fanout(specs, FanoutOptions(jobs=4, run_dir=tmp_path / "run", child_command=_fake_command))
    wall = time.monotonic() - t0
    assert result.success
    assert wall < 2.0, f"4 jobs at jobs=4 should overlap; took {wall:.1f}s"


def test_failure_stops_new_launches_but_finishes_running(tmp_path):
    specs = [
        _spec(tmp_path, "a", mode="fail"),
        _spec(tmp_path, "b", sleep=0.5),
        _spec(tmp_path, "c"),
        _spec(tmp_path, "d"),
    ]
    result = run_batch_fanout(specs, FanoutOptions(jobs=2, run_dir=tmp_path / "run", child_command=_fake_command))
    assert not result.success
    by_name = {r.spec.config["name"]: r for r in result.jobs}
    assert by_name["a"].returncode == 3 and by_name["a"].error
    assert by_name["b"].success                      # already running: finished
    assert by_name["c"].returncode is None and not by_name["c"].success   # never launched
    assert by_name["d"].returncode is None


def test_cancel_sends_sigint_and_child_exits_cleanly(tmp_path):
    specs = [_spec(tmp_path, "a", mode="trap_sigint")]
    stop = {"flag": False}

    def _should_stop():
        return stop["flag"]

    import threading

    threading.Timer(0.6, lambda: stop.__setitem__("flag", True)).start()
    t0 = time.monotonic()
    result = run_batch_fanout(
        specs,
        FanoutOptions(jobs=1, run_dir=tmp_path / "run", child_command=_fake_command, sigint_grace_s=5),
        should_stop=_should_stop,
    )
    assert result.cancelled and not result.success
    assert result.jobs[0].returncode == 130
    assert time.monotonic() - t0 < 4.0
    assert "SIGINT received" in result.jobs[0].log_path.read_text()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_cancel_escalates_to_kill_when_child_ignores_signals(tmp_path):
    specs = [_spec(tmp_path, "a", mode="ignore_signals")]
    stop = {"flag": False}
    import threading

    threading.Timer(0.5, lambda: stop.__setitem__("flag", True)).start()
    t0 = time.monotonic()
    result = run_batch_fanout(
        specs,
        FanoutOptions(jobs=1, run_dir=tmp_path / "run", child_command=_fake_command,
                      sigint_grace_s=0.3, term_grace_s=0.3),
        should_stop=lambda: stop["flag"],
    )
    assert result.cancelled
    assert result.jobs[0].returncode not in (0, None)
    assert time.monotonic() - t0 < 5.0


def test_module_imports_no_qt():
    import ast

    import hydra_suite.trackerkit.batch_fanout as mod
    import hydra_suite.trackerkit.batch_plan as plan_mod

    for module in (mod, plan_mod):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.startswith(("PySide6", "PyQt")) for n in names), module.__file__
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_batch_fanout.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the scheduler**

```python
# src/hydra_suite/trackerkit/batch_fanout.py
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Sequence

from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit.batch_plan import BatchJobSpec
from hydra_suite.utils.video_artifacts import (
    build_video_log_dir,
    choose_writable_artifact_base_dir,
)

logger = logging.getLogger(__name__)

_PROGRESS_RE = re.compile(r"\[(?:track forward|track backward|post)\] (\d{1,3})% ?(.*)$")
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
        return not self.cancelled and bool(self.jobs) and all(j.success for j in self.jobs)


class FanoutEvents(Protocol):
    def job_started(self, spec: BatchJobSpec, gpu: Optional[CudaDevice], log_path: Path) -> None: ...
    def job_progress(self, spec: BatchJobSpec, percent: int, message: str) -> None: ...
    def job_log(self, spec: BatchJobSpec, line: str) -> None: ...
    def job_finished(self, result: FanoutJobResult) -> None: ...


class NullEvents:
    def job_started(self, spec, gpu, log_path) -> None: ...
    def job_progress(self, spec, percent, message) -> None: ...
    def job_log(self, spec, line) -> None: ...
    def job_finished(self, result) -> None: ...


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


@dataclass
class _Live:
    spec: BatchJobSpec
    gpu: Optional[CudaDevice]
    proc: subprocess.Popen
    log_path: Path
    log_handle: object
    reader: threading.Thread
    started_at: float
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
    command = (options.child_command or (
        lambda s, c: default_child_command(s, c, log_level=options.log_level)
    ))(spec, config_json)
    env = build_child_env(os.environ, gpu=gpu, threads_per_job=options.threads_per_job)
    log_path = _job_log_path(spec, timestamp)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"# trackerkit fan-out job {spec.index}: {spec.video_path}\n"
        f"# gpu={gpu.uuid if gpu else '<inherited>'} command={' '.join(command)}\n"
    )
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
        spec=spec, gpu=gpu, proc=proc, log_path=log_path, log_handle=log_handle,
        reader=threading.Thread(target=lambda: None), started_at=time.monotonic(),
    )
    live.reader = threading.Thread(
        target=_pump, args=(live, events), name=f"fanout-log-{spec.index}", daemon=True
    )
    live.reader.start()
    events.job_started(spec, gpu, log_path)
    logger.info(
        "Fan-out: launched job %d (%s) on %s -> %s",
        spec.index, Path(spec.video_path).name, gpu.uuid if gpu else "inherited device", log_path,
    )
    return live


def _finish(live: _Live, *, cancelled: bool) -> FanoutJobResult:
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
        spec=live.spec, gpu=live.gpu, returncode=rc, success=ok, log_path=live.log_path,
        summary_lines=list(live.summary_lines), error=error,
        wall_s=time.monotonic() - live.started_at,
    )


def _stop_children(running: list[_Live], options: FanoutOptions) -> None:
    """SIGINT (clean engine stop) -> SIGTERM -> SIGKILL, with grace periods."""
    def _alive() -> list[_Live]:
        return [l for l in running if l.proc.poll() is None]

    for live in _alive():
        try:
            live.proc.send_signal(signal.SIGINT if os.name != "nt" else signal.CTRL_BREAK_EVENT)
        except Exception:
            pass
    deadline = time.monotonic() + options.sigint_grace_s
    while _alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    for live in _alive():
        try:
            live.proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + options.term_grace_s
    while _alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    for live in _alive():
        try:
            live.proc.kill()
        except Exception:
            pass
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

    slots: list[Optional[CudaDevice]]
    if options.gpus:
        n = max(1, min(int(options.jobs) if options.jobs else len(options.gpus), len(options.gpus)))
        slots = list(options.gpus[:n])
    else:
        slots = [None] * max(1, int(options.jobs))

    run_dir = Path(options.run_dir) if options.run_dir else Path(
        tempfile.mkdtemp(prefix="trackerkit-fanout-")
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
                        live.spec.index, res.error,
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
                    spec=spec, gpu=None, returncode=None, success=False,
                    log_path=run_dir / f"job_{spec.index}_not_started.log",
                    summary_lines=[], error="not started" if not cancelled else "cancelled",
                    wall_s=0.0,
                )
            )
    return FanoutResult(jobs=results, cancelled=cancelled)
```

Note for the implementer: the tests construct `BatchJobSpec(index=0, ...)` for every spec, so `job_<index>_config.json` would collide. Assign indices in `run_batch_fanout` when a spec's index is 0 or duplicated: build `specs = [replace(s, index=i) for i, s in enumerate(specs, 1)]` if `len({s.index for s in specs}) != len(specs)` (use `dataclasses.replace`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_batch_fanout.py -q`
Expected: all passed. `test_failure_stops_new_launches_but_finishes_running` depends on job `a` failing before `c` is launched; with `jobs=2`, `a` and `b` launch together, `a` exits in ~50 ms, and the reap happens before the launch loop, so `c` never launches. If it flakes, give `a` `sleep=0.01` and `b` `sleep=0.8`.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/batch_fanout.py tests/test_trackerkit_batch_fanout.py
git commit -m "feat(trackerkit): Qt-free process-per-GPU batch fan-out scheduler"
```

---

### Task 5: CLI flags `--gpus`, `--jobs`, `--threads-per-job`

**Files:**
- Modify: `src/hydra_suite/trackerkit/app.py:84-135` (track subparser + validation), `:222-258` (`main` dispatch)
- Modify: `src/hydra_suite/trackerkit/cli.py` (`run_tracking_cli` signature + dispatch, new `_run_fanout`, `_print_fanout_table`)
- Test: `tests/test_trackerkit_cli_fanout.py`

**Interfaces:**
- Consumes: `plan_batch_jobs`, `run_batch_fanout`, `FanoutOptions`, `parse_gpu_selectors`, `resolve_gpu_selectors`
- Produces: `run_tracking_cli(video_paths, *, config_path=None, keystone_override=False, sahi_profile=None, gpus: str | None = None, jobs: int = 1, threads_per_job: int | None = None, log_level: str = "INFO") -> int`; `fanout_requested(gpus, jobs) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trackerkit_cli_fanout.py
from __future__ import annotations

from pathlib import Path

import pytest

from hydra_suite.trackerkit import cli
from hydra_suite.trackerkit.app import parse_arguments
from hydra_suite.trackerkit.batch_fanout import FanoutJobResult, FanoutResult
from hydra_suite.trackerkit.batch_plan import BatchJobSpec


def _videos(tmp_path, n=2):
    out = []
    for i in range(n):
        p = tmp_path / f"v{i}.mp4"
        p.write_bytes(b"\x00")
        out.append(str(p))
    return out


def test_parse_track_flags(tmp_path):
    v = _videos(tmp_path, 1)
    args = parse_arguments(["track", v[0], "--gpus", "0-1", "--jobs", "3", "--threads-per-job", "8"])
    assert args.gpus == "0-1" and args.jobs == 3 and args.threads_per_job == 8


def test_parse_track_defaults(tmp_path):
    v = _videos(tmp_path, 1)
    args = parse_arguments(["track", v[0]])
    assert args.gpus is None and args.jobs == 1 and args.threads_per_job is None


def test_fanout_gating_rule():
    assert not cli.fanout_requested(None, 1)
    assert cli.fanout_requested("0", 1)
    assert cli.fanout_requested(None, 2)


def test_sequential_path_untouched_when_not_requested(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    calls = []
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: calls.append(len(specs)) or 0)
    monkeypatch.setattr(cli, "_run_fanout", lambda *a, **k: pytest.fail("fan-out must not run"))
    assert cli.run_tracking_cli(v) == 0
    assert calls == [2]


def test_fanout_path_used_when_jobs_gt_1(tmp_path, monkeypatch):
    v = _videos(tmp_path, 2)
    seen = {}

    def _fake_fanout(specs, options, *, should_stop):
        seen["n"] = len(specs)
        seen["jobs"] = options.jobs
        seen["gpus"] = options.gpus
        seen["threads"] = options.threads_per_job
        return FanoutResult(
            jobs=[FanoutJobResult(s, None, 0, True, tmp_path / "l", ["video=x"], None, 1.0) for s in specs],
            cancelled=False,
        )

    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    monkeypatch.setattr(cli, "_run_sequential", lambda specs: pytest.fail("sequential must not run"))
    assert cli.run_tracking_cli(v, jobs=2, threads_per_job=4) == 0
    assert seen == {"n": 2, "jobs": 2, "gpus": [], "threads": 4}


def test_fanout_exit_code_1_when_any_job_fails(tmp_path, monkeypatch, capsys):
    v = _videos(tmp_path, 2)

    def _fake_fanout(specs, options, *, should_stop):
        jobs = [
            FanoutJobResult(specs[0], None, 0, True, tmp_path / "a.log", ["video=a"], None, 1.0),
            FanoutJobResult(specs[1], None, 3, False, tmp_path / "b.log", [], "exit code 3", 1.0),
        ]
        return FanoutResult(jobs=jobs, cancelled=False)

    monkeypatch.setattr(cli, "run_batch_fanout", _fake_fanout)
    assert cli.run_tracking_cli(v, jobs=2) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "OK" in out and "b.log" in out


def test_gpus_on_host_without_cuda_is_an_error(tmp_path, monkeypatch):
    v = _videos(tmp_path, 1)
    monkeypatch.setattr(cli, "resolve_gpu_selectors", lambda sel: (_ for _ in ()).throw(ValueError("no CUDA")))
    with pytest.raises(ValueError):
        cli.run_tracking_cli(v, gpus="0")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_cli_fanout.py -q`
Expected: FAIL (`AttributeError: fanout_requested`, `parse_arguments` unrecognized `--gpus`)

- [ ] **Step 3: Add the argparse flags in `app.py`**

After the `--sahi-profile` argument in `parse_arguments`:

```python
    track_parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Run videos as parallel child processes, one per listed GPU. "
            "Accepts ordinals (0,1,2), ranges (0-8), GPU- UUID prefixes, or "
            "'auto' for every GPU nvidia-smi reports. Each child sees exactly "
            "one GPU via CUDA_VISIBLE_DEVICES. NVIDIA hosts only."
        ),
    )
    track_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Maximum concurrent videos. Defaults to 1 (in-process sequential). "
            "With --gpus it is clamped to the number of GPUs; without --gpus, "
            "N>1 runs N children that share the current device visibility."
        ),
    )
    track_parser.add_argument(
        "--threads-per-job",
        type=int,
        default=None,
        help=(
            "Opt-in CPU thread cap per child (sets OMP/MKL/OPENBLAS/NUMBA "
            "*_NUM_THREADS when not already set). Off by default."
        ),
    )
```

Add validation after the existing `videos`/`video_list` checks:

```python
        if int(getattr(args, "jobs", 1) or 1) < 1:
            track_parser.error("--jobs must be >= 1")
        tpj = getattr(args, "threads_per_job", None)
        if tpj is not None and int(tpj) < 1:
            track_parser.error("--threads-per-job must be >= 1")
```

In `main`, extend the `run_tracking_cli(...)` call:

```python
            exit_code = run_tracking_cli(
                resolved_videos,
                config_path=args.config,
                keystone_override=bool(args.keystone_override),
                sahi_profile=getattr(args, "sahi_profile", None),
                gpus=getattr(args, "gpus", None),
                jobs=int(getattr(args, "jobs", 1) or 1),
                threads_per_job=getattr(args, "threads_per_job", None),
                log_level=str(args.log_level),
            )
```

- [ ] **Step 4: Add the dispatch in `cli.py`**

Imports to add:

```python
import signal
import threading
import time

from hydra_suite.runtime.cuda_devices import parse_gpu_selectors, resolve_gpu_selectors
from hydra_suite.trackerkit.batch_fanout import (
    FanoutJobResult,
    FanoutOptions,
    FanoutResult,
    run_batch_fanout,
)
```

Change the signature and tail of `run_tracking_cli`:

```python
def fanout_requested(gpus: str | None, jobs: int) -> bool:
    """The gating rule: fan-out iff --gpus was given or --jobs > 1."""
    return bool(str(gpus or "").strip()) or int(jobs or 1) > 1


def run_tracking_cli(
    video_paths: Sequence[str],
    *,
    config_path: str | None = None,
    keystone_override: bool = False,
    sahi_profile: str | None = None,
    gpus: str | None = None,
    jobs: int = 1,
    threads_per_job: int | None = None,
    log_level: str = "INFO",
) -> int:
    ...  # validation + plan_batch_jobs as in Task 3 ...
    if not fanout_requested(gpus, jobs):
        return _run_sequential(specs)
    devices = resolve_gpu_selectors(parse_gpu_selectors(gpus)) if gpus else []
    options = FanoutOptions(
        gpus=devices,
        jobs=int(jobs or 1) if not devices else max(1, min(int(jobs or len(devices)), len(devices))),
        threads_per_job=threads_per_job,
        log_level=log_level,
    )
    return _run_fanout(specs, options)
```

Add:

```python
class _CliEvents:
    """Log-only event sink for the terminal."""

    def job_started(self, spec, gpu, log_path) -> None:
        logger.info("[job %d] started %s on %s (log: %s)", spec.index,
                    Path(spec.video_path).name, gpu.uuid if gpu else "inherited device", log_path)

    def job_progress(self, spec, percent, message) -> None:
        logger.info("[job %d] %3d%% %s", spec.index, percent, message)

    def job_log(self, spec, line) -> None:  # child lines already land in the per-job log file
        pass

    def job_finished(self, result) -> None:
        logger.info("[job %d] %s (%.1fs)", result.spec.index,
                    "OK" if result.success else f"FAIL: {result.error}", result.wall_s)


def _install_stop_flag() -> tuple[Callable[[], bool], Callable[[], None]]:
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
        def _restore() -> None: ...
    return stop.is_set, _restore


def _print_fanout_table(result: FanoutResult) -> None:
    rows = [("#", "video", "gpu", "status", "wall", "log")]
    for job in result.jobs:
        rows.append((
            str(job.spec.index),
            Path(job.spec.video_path).name,
            (job.gpu.uuid[:12] if job.gpu else "-"),
            "OK" if job.success else f"FAIL ({job.error})",
            f"{job.wall_s:.0f}s",
            str(job.log_path),
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)))
    ok = sum(1 for j in result.jobs if j.success)
    print(f"\n{ok}/{len(result.jobs)} videos succeeded" + ("  (CANCELLED)" if result.cancelled else ""))


def _run_fanout(specs: Sequence[BatchJobSpec], options: FanoutOptions) -> int:
    should_stop, restore = _install_stop_flag()
    try:
        logger.info(
            "Tracker CLI fan-out: %d videos, %d slot(s)%s",
            len(specs),
            len(options.gpus) if options.gpus else options.jobs,
            f" on GPUs {[g.index for g in options.gpus]}" if options.gpus else "",
        )
        result = run_batch_fanout(specs, options, events=_CliEvents(), should_stop=should_stop)
    finally:
        restore()
    _print_fanout_table(result)
    return 0 if result.success else 1
```

Note the test monkeypatches `cli.run_batch_fanout` and calls it with `(specs, options, should_stop=...)` only; pass `events` as a keyword AFTER `should_stop` is fine, but the fake accepts only `should_stop`. Make `_run_fanout` call `run_batch_fanout(specs, options, should_stop=should_stop, events=_CliEvents())` and let the test fake accept `**kwargs`: update the two fakes in the test to `def _fake_fanout(specs, options, *, should_stop, events=None)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_cli_fanout.py tests/test_trackerkit_cli_cutover.py tests/test_trackerkit_cli_sahi_profile.py -q`
Expected: all passed (cutover's `test_cli_module_imports_no_qt` must still pass)

- [ ] **Step 6: Smoke the real child command on this host (no GPU)**

Run from the worktree with `hydra-mps` active and fixtures present:

```bash
export PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE
python -m hydra_suite.trackerkit.app track \
  tools/equivalence/fixtures/clips/fly_obb.mp4 tools/equivalence/fixtures/clips/worm_bgsub.mp4 \
  --config tools/equivalence/fixtures/configs/fly_obb.json --jobs 2 2>&1 | tail -20
```

(Check `ls tools/equivalence/fixtures/` for the real config/clip layout first; adapt paths.) Expected: two `[job N] started` lines, progress lines, a final table with 2 OK rows and exit code 0. If a job fails, open its `<stem>_logs/<stem>_fanout_<ts>.log`.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/app.py src/hydra_suite/trackerkit/cli.py tests/test_trackerkit_cli_fanout.py
git commit -m "feat(trackerkit): --gpus/--jobs/--threads-per-job fan-out flags on trackerkit track"
```

---

### Task 6: Session schema fields and SetupPanel controls

**Files:**
- Modify: `src/hydra_suite/trackerkit/config/schemas.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py:327-340` (after `chk_batch_keystone_override`)
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py` (handlers `_on_batch_parallel_changed`)
- Test: `tests/test_trackerkit_config_schema_fanout.py`

**Interfaces:**
- Produces: `TrackerConfig.batch_parallel: bool = False`, `TrackerConfig.batch_parallel_jobs: int = 0` (0 = auto), `TrackerConfig.batch_parallel_gpus: str = "auto"`; SetupPanel widgets `chk_batch_parallel: QCheckBox`, `spin_batch_parallel_jobs: QSpinBox`, `edit_batch_parallel_gpus: QLineEdit`, `container_batch_parallel: QWidget`; `MainWindow._on_batch_parallel_changed()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trackerkit_config_schema_fanout.py
from hydra_suite.trackerkit.config.schemas import TrackerConfig


def test_defaults():
    c = TrackerConfig()
    assert c.batch_parallel is False
    assert c.batch_parallel_jobs == 0
    assert c.batch_parallel_gpus == "auto"


def test_round_trip():
    c = TrackerConfig(batch_parallel=True, batch_parallel_jobs=4, batch_parallel_gpus="0-3")
    d = c.to_dict()
    assert d["batch_parallel"] is True and d["batch_parallel_jobs"] == 4 and d["batch_parallel_gpus"] == "0-3"
    back = TrackerConfig.from_dict(d)
    assert (back.batch_parallel, back.batch_parallel_jobs, back.batch_parallel_gpus) == (True, 4, "0-3")


def test_from_dict_tolerates_missing_keys():
    back = TrackerConfig.from_dict({})
    assert back.batch_parallel is False and back.batch_parallel_jobs == 0 and back.batch_parallel_gpus == "auto"


def test_fanout_fields_never_reach_engine_config():
    """The GUI's sidecar config builder must not emit these keys."""
    import ast
    from pathlib import Path

    import hydra_suite.trackerkit.gui.orchestrators.config as mod

    src = Path(mod.__file__).read_text()
    assert "batch_parallel" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_config_schema_fanout.py -q`
Expected: FAIL (`TypeError: unexpected keyword batch_parallel`)

- [ ] **Step 3: Add the schema fields**

In `TrackerConfig` after `batch_videos`:

```python
    # --- Batch fan-out (session state only; NEVER emitted into the per-video
    # engine config, so cache keys and child invocations are unchanged) ---
    batch_parallel: bool = False
    batch_parallel_jobs: int = 0  # 0 = one per selected GPU (or 1 without GPUs)
    batch_parallel_gpus: str = "auto"
```

In `to_dict` after `"batch_videos"`:

```python
            "batch_parallel": bool(self.batch_parallel),
            "batch_parallel_jobs": int(self.batch_parallel_jobs),
            "batch_parallel_gpus": str(self.batch_parallel_gpus),
```

In `from_dict` after `batch_videos=`:

```python
            batch_parallel=bool(data.get("batch_parallel", False)),
            batch_parallel_jobs=int(data.get("batch_parallel_jobs", 0)),
            batch_parallel_gpus=str(data.get("batch_parallel_gpus", "auto")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_config_schema_fanout.py -q`
Expected: 4 passed

- [ ] **Step 5: Add the panel controls**

In `setup_panel.py`, directly after `v_container.addWidget(self.chk_batch_keystone_override)`:

```python
        # --- Parallel fan-out (one child process per GPU) ---
        from hydra_suite.runtime.cuda_devices import list_cuda_devices

        cuda_devices = list_cuda_devices()
        self.chk_batch_parallel = QCheckBox("Run videos in parallel (one process per GPU)")
        self.chk_batch_parallel.setToolTip(
            "Each video runs as its own headless child process pinned to one\n"
            "GPU (CUDA_VISIBLE_DEVICES). Output is identical to sequential\n"
            "batch tracking. No live preview while running."
        )
        self.chk_batch_parallel.setChecked(bool(self._main_window.config.batch_parallel))
        self.chk_batch_parallel.toggled.connect(self._main_window._on_batch_parallel_changed)
        v_container.addWidget(self.chk_batch_parallel)

        self.container_batch_parallel = QWidget()
        h_par = QHBoxLayout(self.container_batch_parallel)
        h_par.setContentsMargins(0, 0, 0, 0)
        h_par.addWidget(QLabel("Jobs:"))
        self.spin_batch_parallel_jobs = QSpinBox()
        self.spin_batch_parallel_jobs.setRange(1, 64)
        default_jobs = int(self._main_window.config.batch_parallel_jobs) or max(1, len(cuda_devices))
        self.spin_batch_parallel_jobs.setValue(default_jobs)
        self.spin_batch_parallel_jobs.setToolTip("Maximum videos running at once.")
        self.spin_batch_parallel_jobs.valueChanged.connect(
            self._main_window._on_batch_parallel_changed
        )
        h_par.addWidget(self.spin_batch_parallel_jobs)
        self.lbl_batch_parallel_gpus = QLabel("GPUs:")
        h_par.addWidget(self.lbl_batch_parallel_gpus)
        self.edit_batch_parallel_gpus = QLineEdit(
            str(self._main_window.config.batch_parallel_gpus or "auto")
        )
        self.edit_batch_parallel_gpus.setPlaceholderText("auto, 0-3, 0,2, or GPU-uuid")
        self.edit_batch_parallel_gpus.setToolTip(
            f"{len(cuda_devices)} CUDA device(s) detected by nvidia-smi.\n"
            "'auto' = all of them."
        )
        self.edit_batch_parallel_gpus.editingFinished.connect(
            self._main_window._on_batch_parallel_changed
        )
        h_par.addWidget(self.edit_batch_parallel_gpus)
        has_cuda = bool(cuda_devices)
        self.lbl_batch_parallel_gpus.setVisible(has_cuda)
        self.edit_batch_parallel_gpus.setVisible(has_cuda)
        self.container_batch_parallel.setVisible(self.chk_batch_parallel.isChecked())
        v_container.addWidget(self.container_batch_parallel)
```

Add `QLineEdit`, `QSpinBox`, `QLabel`, `QHBoxLayout`, `QWidget` to the panel's PySide6 imports if missing.

In `main_window.py` next to `_on_batch_mode_toggled`:

```python
    def _on_batch_parallel_changed(self, *_args) -> None:
        """Mirror the Batch › Parallel controls into session state."""
        panel = self._setup_panel
        self.config.batch_parallel = bool(panel.chk_batch_parallel.isChecked())
        self.config.batch_parallel_jobs = int(panel.spin_batch_parallel_jobs.value())
        self.config.batch_parallel_gpus = (
            panel.edit_batch_parallel_gpus.text().strip() or "auto"
        )
        panel.container_batch_parallel.setVisible(self.config.batch_parallel)
```

- [ ] **Step 6: Launch the GUI once to verify the controls render**

Run: `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE timeout 25 python -m hydra_suite.trackerkit.app` and tick Batch → the Parallel row appears; ticking it reveals Jobs (and GPUs only on a CUDA host). Close the window. Expected: no traceback on stdout.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/config/schemas.py src/hydra_suite/trackerkit/gui/panels/setup_panel.py src/hydra_suite/trackerkit/gui/main_window.py tests/test_trackerkit_config_schema_fanout.py
git commit -m "feat(trackerkit-gui): batch parallel session fields and Setup panel controls"
```

---

### Task 7: GUI worker, job-table dialog, and orchestrator branch

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/batch_fanout_worker.py`
- Create: `src/hydra_suite/trackerkit/gui/dialogs/batch_fanout_dialog.py`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py:1039-1083` (`start_tracking` branch), `:206-311` (`stop_tracking`)
- Test: `tests/test_trackerkit_batch_fanout_worker.py`

**Interfaces:**
- Consumes: `plan_batch_jobs`, `run_batch_fanout`, `FanoutOptions`, `FanoutResult`, `parse_gpu_selectors`, `resolve_gpu_selectors`, `list_cuda_devices`, `BaseWorker`, `BaseDialog`
- Produces:
  - `BatchFanoutWorker(BaseWorker)` with `__init__(specs, options, parent=None)`, signals `job_started(int, str, str)`, `job_progress(int, int, str)`, `job_log(int, str)`, `job_finished(int, bool, str)`, `fanout_finished(object)` (a `FanoutResult`), method `cancel()`
  - `BatchFanoutDialog(BaseDialog)` with `__init__(specs, parent)`, slots `on_job_started(int, str, str)`, `on_job_progress(int, int, str)`, `on_job_finished(int, bool, str)`, `on_fanout_finished(object)`, signal `cancel_requested()`
  - `TrackingOrchestrator.start_batch_fanout() -> bool`, `TrackingOrchestrator._on_batch_fanout_finished(result)`

- [ ] **Step 1: Write the failing worker test (no QApplication needed for the mapping)**

```python
# tests/test_trackerkit_batch_fanout_worker.py
"""The worker's FanoutEvents implementation must map callbacks to signals with
bounded payloads. We call the event methods directly and capture emits via
a lightweight signal spy, so no event loop is required."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication  # noqa: E402

from hydra_suite.runtime.cuda_devices import CudaDevice  # noqa: E402
from hydra_suite.trackerkit.batch_fanout import FanoutJobResult, FanoutOptions  # noqa: E402
from hydra_suite.trackerkit.batch_plan import BatchJobSpec  # noqa: E402
from hydra_suite.trackerkit.gui.workers.batch_fanout_worker import BatchFanoutWorker  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QCoreApplication.instance() or QCoreApplication([])


def _spec(i):
    return BatchJobSpec(index=i, video_path=f"/tmp/v{i}.mp4", config_path=None, config={}, provenance="own-sidecar")


def test_events_map_to_signals(app):
    worker = BatchFanoutWorker([_spec(1)], FanoutOptions())
    got = {}
    worker.job_started.connect(lambda i, v, l: got.__setitem__("started", (i, v, l)))
    worker.job_progress.connect(lambda i, p, m: got.__setitem__("progress", (i, p, m)))
    worker.job_finished.connect(lambda i, ok, msg: got.__setitem__("finished", (i, ok, msg)))
    gpu = CudaDevice(0, "GPU-abc", "x")
    worker.job_started_cb(_spec(1), gpu, Path("/tmp/l.log"))
    worker.job_progress_cb(_spec(1), 42, "detecting")
    worker.job_finished_cb(FanoutJobResult(_spec(1), gpu, 0, True, Path("/tmp/l.log"), ["video=v1"], None, 2.0))
    assert got["started"] == (1, "/tmp/v1.mp4", "/tmp/l.log")
    assert got["progress"] == (1, 42, "detecting")
    assert got["finished"][0] == 1 and got["finished"][1] is True


def test_cancel_sets_stop_flag(app):
    worker = BatchFanoutWorker([_spec(1)], FanoutOptions())
    assert worker.should_stop() is False
    worker.cancel()
    assert worker.should_stop() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_batch_fanout_worker.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the worker**

```python
# src/hydra_suite/trackerkit/gui/workers/batch_fanout_worker.py
"""Qt bridge for the Qt-free batch fan-out scheduler."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Signal

from hydra_suite.trackerkit.batch_fanout import (
    FanoutJobResult,
    FanoutOptions,
    run_batch_fanout,
)
from hydra_suite.trackerkit.batch_plan import BatchJobSpec
from hydra_suite.widgets.workers import BaseWorker, bounded_worker_message


class BatchFanoutWorker(BaseWorker):
    """Runs ``run_batch_fanout`` on a QThread and re-emits its events."""

    job_started = Signal(int, str, str)      # index, video_path, log_path
    job_progress = Signal(int, int, str)     # index, percent, message
    job_log = Signal(int, str)               # index, line
    job_finished = Signal(int, bool, str)    # index, success, summary-or-error
    fanout_finished = Signal(object)         # FanoutResult

    def __init__(self, specs: Sequence[BatchJobSpec], options: FanoutOptions, parent=None) -> None:
        super().__init__(parent)
        self._specs = list(specs)
        self._options = options
        self._stop = threading.Event()
        self.result = None

    # --- cancellation -----------------------------------------------------
    def cancel(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set() or self.isInterruptionRequested()

    # --- FanoutEvents (called from the worker thread) ----------------------
    def job_started_cb(self, spec: BatchJobSpec, gpu, log_path: Path) -> None:
        self.job_started.emit(int(spec.index), str(spec.video_path), str(log_path))

    def job_progress_cb(self, spec: BatchJobSpec, percent: int, message: str) -> None:
        self.job_progress.emit(int(spec.index), int(percent), bounded_worker_message(message))

    def job_log_cb(self, spec: BatchJobSpec, line: str) -> None:
        self.job_log.emit(int(spec.index), bounded_worker_message(line))

    def job_finished_cb(self, result: FanoutJobResult) -> None:
        text = " | ".join(result.summary_lines) if result.success else (result.error or "failed")
        self.job_finished.emit(int(result.spec.index), bool(result.success), bounded_worker_message(text))

    # --- BaseWorker ---------------------------------------------------------
    def execute(self) -> None:
        events = _EventAdapter(self)
        self.result = run_batch_fanout(
            self._specs, self._options, events=events, should_stop=self.should_stop
        )
        self.fanout_finished.emit(self.result)


class _EventAdapter:
    def __init__(self, worker: BatchFanoutWorker) -> None:
        self._w = worker

    def job_started(self, spec, gpu, log_path) -> None:
        self._w.job_started_cb(spec, gpu, log_path)

    def job_progress(self, spec, percent, message) -> None:
        self._w.job_progress_cb(spec, percent, message)

    def job_log(self, spec, line) -> None:
        self._w.job_log_cb(spec, line)

    def job_finished(self, result) -> None:
        self._w.job_finished_cb(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_trackerkit_batch_fanout_worker.py -q`
Expected: 2 passed

- [ ] **Step 5: Write the dialog**

```python
# src/hydra_suite/trackerkit/gui/dialogs/batch_fanout_dialog.py
"""Per-job progress table for a parallel batch run."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
)

from hydra_suite.trackerkit.batch_plan import BatchJobSpec
from hydra_suite.widgets.dialogs import BaseDialog

_COLS = ("#", "Video", "GPU", "Status", "Progress", "Last message")


class BatchFanoutDialog(BaseDialog):
    cancel_requested = Signal()

    def __init__(self, specs: Sequence[BatchJobSpec], parent=None) -> None:
        super().__init__("Parallel Batch Tracking", parent, buttons=QDialogButtonBox.Cancel)
        self.setModal(False)
        self.resize(900, 420)
        self._rows: dict[int, int] = {}
        self._log_paths: dict[int, str] = {}
        self._header = QLabel(f"{len(specs)} videos queued.")
        self.add_content(self._header)
        self._table = QTableWidget(len(specs), len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row, spec in enumerate(specs):
            self._rows[int(spec.index)] = row
            self._table.setItem(row, 0, QTableWidgetItem(str(spec.index)))
            self._table.setItem(row, 1, QTableWidgetItem(Path(spec.video_path).name))
            self._table.setItem(row, 2, QTableWidgetItem("-"))
            self._table.setItem(row, 3, QTableWidgetItem("queued"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            self._table.setCellWidget(row, 4, bar)
            self._table.setItem(row, 5, QTableWidgetItem(""))
        self.add_content(self._table)
        self._buttons.rejected.disconnect(self.reject)
        self._buttons.rejected.connect(self._on_cancel_clicked)
        self._finished = False

    def _on_cancel_clicked(self) -> None:
        if self._finished:
            self.reject()
            return
        self._header.setText("Cancelling… waiting for children to stop.")
        self.cancel_requested.emit()

    def _set(self, index: int, col: int, text: str) -> None:
        row = self._rows.get(int(index))
        if row is None:
            return
        item = self._table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(row, col, item)
        item.setText(text)

    def on_job_started(self, index: int, video: str, log_path: str) -> None:
        self._log_paths[index] = log_path
        self._set(index, 3, "running")

    def on_job_progress(self, index: int, percent: int, message: str) -> None:
        row = self._rows.get(int(index))
        if row is None:
            return
        bar = self._table.cellWidget(row, 4)
        if bar is not None:
            bar.setValue(max(0, min(100, int(percent))))
        self._set(index, 5, message)

    def on_job_finished(self, index: int, ok: bool, text: str) -> None:
        self._set(index, 3, "OK" if ok else "FAILED")
        self._set(index, 5, text)

    def set_gpu(self, index: int, label: str) -> None:
        self._set(index, 2, label)

    def on_fanout_finished(self, result) -> None:
        self._finished = True
        ok = sum(1 for j in result.jobs if j.success)
        suffix = " (cancelled)" if result.cancelled else ""
        self._header.setText(f"Done: {ok}/{len(result.jobs)} videos succeeded{suffix}. Logs are in each video's <stem>_logs/ folder.")
        self._buttons.clear()
        self._buttons.addButton(QDialogButtonBox.Close)
        self._buttons.rejected.connect(self.reject)
```

- [ ] **Step 6: Add the orchestrator branch**

In `tracking.py`, at the top of `start_tracking` **inside** `if not preview_mode:` and **inside** the `if self._panels.setup.g_batch.isChecked():` block, replace the confirmation message text so it reads `"in parallel across GPUs"` when `chk_batch_parallel` is checked and `"sequentially"` otherwise (compute `mode_word` before the `QMessageBox.question`). Then, immediately after the `save_config(...)` block (still inside `if not preview_mode:`), add:

```python
            if (
                self._panels.setup.g_batch.isChecked()
                and getattr(self._panels.setup, "chk_batch_parallel", None) is not None
                and self._panels.setup.chk_batch_parallel.isChecked()
            ):
                # Parallel fan-out replaces the sequential re-entrant batch loop
                # entirely: children are headless CLI processes.
                self._mw.current_batch_index = -1
                self.start_batch_fanout()
                return
```

Add the methods to `TrackingOrchestrator`:

```python
    def start_batch_fanout(self) -> bool:
        from hydra_suite.runtime.cuda_devices import (
            list_cuda_devices,
            parse_gpu_selectors,
            resolve_gpu_selectors,
        )
        from hydra_suite.trackerkit.batch_fanout import FanoutOptions
        from hydra_suite.trackerkit.batch_plan import BatchPlanError, plan_batch_jobs
        from hydra_suite.trackerkit.gui.dialogs.batch_fanout_dialog import BatchFanoutDialog
        from hydra_suite.trackerkit.gui.workers.batch_fanout_worker import BatchFanoutWorker

        cfg = self._mw.config
        setup = self._panels.setup
        videos = list(self._mw.batch_videos)
        try:
            # The keystone's sidecar was just written by save_config(), so the
            # planner sees exactly what `trackerkit track --video-list` would.
            specs = plan_batch_jobs(
                videos, keystone_override=setup.chk_batch_keystone_override.isChecked()
            )
        except BatchPlanError as exc:
            QMessageBox.warning(self._mw, "Batch cannot start", str(exc))
            return False

        devices = []
        if list_cuda_devices():
            try:
                devices = resolve_gpu_selectors(parse_gpu_selectors(cfg.batch_parallel_gpus or "auto"))
            except ValueError as exc:
                QMessageBox.warning(self._mw, "GPU selection", str(exc))
                return False
        jobs = int(cfg.batch_parallel_jobs) or (len(devices) if devices else 1)
        if devices:
            jobs = max(1, min(jobs, len(devices)))
        options = FanoutOptions(gpus=devices, jobs=jobs, log_level="INFO")

        worker = BatchFanoutWorker(specs, options, parent=self._mw)
        dialog = BatchFanoutDialog(specs, parent=self._mw)
        worker.job_started.connect(dialog.on_job_started)
        worker.job_progress.connect(dialog.on_job_progress)
        worker.job_finished.connect(dialog.on_job_finished)
        worker.fanout_finished.connect(dialog.on_fanout_finished)
        worker.fanout_finished.connect(self._on_batch_fanout_finished)
        worker.error.connect(lambda msg: self._on_batch_fanout_error(msg))
        dialog.cancel_requested.connect(worker.cancel)
        self._mw.batch_fanout_worker = worker
        self._mw.batch_fanout_dialog = dialog

        self._mw._stop_all_requested = False
        self._mw.btn_start.setText("Stop Tracking")
        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        self._mw.progress_bar.setRange(0, 0)  # busy indicator; per-job bars live in the dialog
        self._mw.progress_label.setText(
            f"Parallel batch: {len(specs)} videos on {len(devices) or jobs} slot(s)…"
        )
        self._mw._apply_ui_state("tracking")
        dialog.show()
        worker.start()
        logger.info("Parallel batch fan-out started: %d videos, %d slot(s)", len(specs), len(devices) or jobs)
        return True

    def _on_batch_fanout_error(self, message: str) -> None:
        logger.error("Batch fan-out worker error: %s", message)
        QMessageBox.critical(self._mw, "Parallel batch failed", message)
        self._restore_after_fanout()

    def _on_batch_fanout_finished(self, result) -> None:
        ok = sum(1 for j in result.jobs if j.success)
        logger.info("Parallel batch finished: %d/%d succeeded (cancelled=%s)", ok, len(result.jobs), result.cancelled)
        self._restore_after_fanout()
        if not result.cancelled:
            QMessageBox.information(
                self._mw, "Batch Complete",
                f"{ok}/{len(result.jobs)} videos succeeded.\nPer-video logs are in each video's <stem>_logs/ folder.",
            )

    def _restore_after_fanout(self) -> None:
        self._mw.progress_bar.setRange(0, 100)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        self._mw.progress_label.setText("Ready")
        self._mw._cleanup_session_logging()
        self._mw._set_ui_controls_enabled(True)
        self._mw.btn_start.blockSignals(True)
        self._mw.btn_start.setChecked(False)
        self._mw.btn_start.blockSignals(False)
        self._mw.btn_start.setText("Start Full Tracking")
        self._mw._apply_ui_state("idle" if self._mw.current_video_path else "no_video")
        self._mw.current_batch_index = -1
        self._cleanup_thread_reference("batch_fanout_worker")
```

Check how `_cleanup_thread_reference` is called elsewhere in `stop_tracking` (it takes the attribute name or the worker; match the existing call style). In `stop_tracking`, right after `self._mw._stop_all_requested = True`, add:

```python
        fanout_worker = getattr(self._mw, "batch_fanout_worker", None)
        if fanout_worker is not None and fanout_worker.isRunning():
            fanout_worker.cancel()
            self._request_qthread_stop(
                fanout_worker, "BatchFanoutWorker", timeout_ms=25000, force_terminate=False
            )
```

Initialise `self.batch_fanout_worker = None` and `self.batch_fanout_dialog = None` in `MainWindow.__init__` next to the other worker attributes (search for `self.session_worker = None`).

- [ ] **Step 7: Verify the GUI end to end on this host**

Run: `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE python -m hydra_suite.trackerkit.app`. Load `fly_obb.mp4` from the fixtures, tick Batch, add `worm_bgsub.mp4`, tick "Run videos in parallel", Jobs=2, Start Full Tracking, confirm. Expected: the job table shows both running, both reach OK, "Batch Complete 2/2". Click Stop mid-run on a second attempt: both rows show FAILED/cancelled within ~15 s and the UI returns to idle. Record what you observed in the commit body.

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/batch_fanout_worker.py \
  src/hydra_suite/trackerkit/gui/dialogs/batch_fanout_dialog.py \
  src/hydra_suite/trackerkit/gui/orchestrators/tracking.py \
  src/hydra_suite/trackerkit/gui/main_window.py \
  tests/test_trackerkit_batch_fanout_worker.py
git commit -m "feat(trackerkit-gui): parallel batch fan-out worker, job table dialog, orchestrator branch"
```

---

### Task 8: Documentation

**Files:**
- Create: `docs/user-guide/trackerkit-cli.md`
- Modify: `docs/user-guide/configuration-reference.md` (add the three session fields under whatever section lists `batch_videos`/session state; if none exists, add a short "Batch session state" subsection)
- Modify: `mkdocs.yml` (add the new page under the User Guide nav next to `workflow.md`)
- Modify: `CLAUDE.md` (one line under "Launching the Applications")

- [ ] **Step 1: Write the CLI page**

```markdown
# TrackerKit command line

`trackerkit track` runs the same tracking pipeline as the GUI, headless.

## One video

```bash
trackerkit track video.mp4                     # uses video_config.json beside the video
trackerkit track video.mp4 --config my.json    # explicit config
```

## A batch

```bash
trackerkit track a.mp4 b.mp4 c.mp4             # a.mp4 is the keystone
trackerkit track --video-list batch.txt        # one absolute path per line, keystone first (the GUI's Export List format)
trackerkit track --video-list batch.txt --keystone-override
```

Keystone rules: the first video's config is the baseline. Later videos use
their own `<stem>_config.json` when present, otherwise the baseline.
`--keystone-override` (or an explicit `--config` on a multi-video batch)
forces the baseline onto every video. `--sahi-profile` applies to all.

Videos run one after another in this process. Output per video:
`<stem>_tracking.csv` (raw) and `<stem>_tracking_forward_processed.csv`.

## Parallel across GPUs

```bash
trackerkit track --video-list batch.txt --gpus auto          # one child per GPU nvidia-smi reports
trackerkit track --video-list batch.txt --gpus 0-3           # four GPUs
trackerkit track --video-list batch.txt --gpus 0,2,GPU-8f1a  # ordinals or UUID prefixes
trackerkit track --video-list batch.txt --jobs 3             # three children sharing the current device (CPU / Apple Silicon)
```

Each video becomes a child process `trackerkit track <video> --config <effective.json>`
pinned with `CUDA_VISIBLE_DEVICES=<uuid>`, so every child (and the SLEAP
service it starts) sees exactly one GPU. Output is identical to the sequential
run. Rules:

- Fan-out engages only when `--gpus` is given or `--jobs > 1`.
- `--jobs` is clamped to the number of selected GPUs.
- A failed video stops new launches; running videos finish. Exit code is 1
  if any video failed. A per-video table with log paths is printed at the end.
- Ctrl-C asks every child to stop cleanly, then terminates stragglers.
- Per-child logs: `<video dir>/<stem>_logs/<stem>_fanout_<timestamp>.log`.
- `--threads-per-job N` (opt-in) caps OMP/MKL/OpenBLAS/Numba threads per child.
  Leave it off unless the host is oversubscribed; it can change floating-point
  reduction order in threaded kernels.
- Requirements in each child: `conda` on `PATH` for SLEAP pose, and the same
  `HYDRA_DATA_DIR`/`HYDRA_CONFIG_DIR` as the parent (inherited automatically).

The GUI exposes the same feature under **Batch › Run videos in parallel**.

## Example: nine-GPU host

```bash
conda activate hydra-cuda
trackerkit track --video-list /data/session_2026-09/batch.txt --gpus auto
```
```

- [ ] **Step 2: Add the config-reference entries, nav entry, and CLAUDE.md line**

configuration-reference: a table with `batch_parallel` (bool, default false), `batch_parallel_jobs` (int, 0 = one per GPU), `batch_parallel_gpus` (string, `auto`), each described as session state that never enters the per-video engine config.

CLAUDE.md under "Launching the Applications", after the `detectkit` line's code block:

```
`trackerkit track --video-list x.txt --gpus auto` fans a batch out one child process per GPU (see `docs/user-guide/trackerkit-cli.md`); fan-out engages only with `--gpus` or `--jobs > 1`, so the plain CLI path stays byte-identical.
```

- [ ] **Step 3: Build the docs**

Run: `make docs-build 2>&1 | tail -5`
Expected: build succeeds with no warnings about the new page.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/trackerkit-cli.md docs/user-guide/configuration-reference.md mkdocs.yml CLAUDE.md
git commit -m "docs: TrackerKit CLI page with GPU fan-out; session fields in configuration reference"
```

---

### Task 9: Verification gates

**Files:** none new in `src/`. Create `tools/equivalence/fanout_gate.sh`.

- [x] **Step 1: Write the fan-out vs sequential gate script**

```bash
#!/usr/bin/env bash
# Fan-out vs sequential byte-identity gate. Run from the worktree root with
# the platform conda env ACTIVE (hydra-mps here, hydra-cuda on mehek).
#   FIXTURES=tools/equivalence/fixtures OUT=/tmp/fanout_gate bash tools/equivalence/fanout_gate.sh fly_obb worm_bgsub ant_obb_sleap ant_cnn_identity
set -euo pipefail
export PYTHONPATH="$PWD/src" KMP_DUPLICATE_LIB_OK=TRUE
FIXTURES="${FIXTURES:-tools/equivalence/fixtures}"
OUT="${OUT:-/tmp/fanout_gate}"
EXTRA="${EXTRA:-}"            # e.g. "--threads-per-job 4" or "--gpus 0"
CLIPS=("$@")
rm -rf "$OUT"; mkdir -p "$OUT/seq" "$OUT/par"
find "$PWD/src" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

stage() {  # stage <mode>: symlink clips+configs into a fresh dir so outputs are isolated
  local dir="$OUT/$1"
  for clip in "${CLIPS[@]}"; do
    ln -sf "$(realpath "$FIXTURES/clips/$clip.mp4")" "$dir/$clip.mp4"
    cp "$FIXTURES/configs/$clip.json" "$dir/${clip}_config.json"
  done
}
stage seq; stage par
list_seq="$OUT/seq/batch.txt"; list_par="$OUT/par/batch.txt"
for clip in "${CLIPS[@]}"; do echo "$OUT/seq/$clip.mp4" >> "$list_seq"; echo "$OUT/par/$clip.mp4" >> "$list_par"; done

echo "== sequential =="; python -m hydra_suite.trackerkit.app track --video-list "$list_seq" 2>&1 | tail -3
echo "== fan-out (--jobs 2 $EXTRA) =="; python -m hydra_suite.trackerkit.app track --video-list "$list_par" --jobs 2 $EXTRA 2>&1 | tail -8

status=0
for clip in "${CLIPS[@]}"; do
  for suffix in _tracking.csv _tracking_forward_processed.csv; do
    a="$OUT/seq/$clip$suffix"; b="$OUT/par/$clip$suffix"
    rows=$(wc -l < "$a" 2>/dev/null || echo 0)
    if [ "$rows" -le 1 ]; then echo "❌ $clip$suffix: sequential CSV has $rows rows (empty run = fake pass)"; status=1; continue; fi
    if cmp -s "$a" "$b"; then echo "✅ $clip$suffix byte-identical ($rows rows)"; else echo "❌ $clip$suffix DIFFERS"; status=1; fi
  done
done
exit $status
```

Adjust the fixture layout (`clips/`, `configs/`) to whatever `ls tools/equivalence/fixtures` actually shows, and check how `tools/equivalence/runner.py` names its config sidecar so the staged `${clip}_config.json` matches what the CLI expects.

- [x] **Step 2: Run the gate on MPS (this box)**

```bash
pkill -f "sleap" || true; pkill -f "hydra_suite.trackerkit.app" || true   # stale sleap/hydra only
conda activate hydra-mps
OUT=/tmp/fanout_gate bash tools/equivalence/fanout_gate.sh fly_obb worm_bgsub ant_obb_sleap ant_cnn_identity
```

Expected: 8 ✅ lines, exit 0. `ant_obb_sleap` proves two concurrent SLEAP service children on one host. If SLEAP CSVs are empty, conda was not active in the shell that launched the parent.

- [x] **Step 3: Run the gate with thread caps engaged**

```bash
EXTRA="--threads-per-job 4" OUT=/tmp/fanout_gate_caps bash tools/equivalence/fanout_gate.sh fly_obb worm_bgsub ant_obb_sleap ant_cnn_identity
```

Expected: either 8 ✅ (record in the plan that caps are identity-safe on MPS) or a ❌ on some clip (record which; caps stay opt-in per spec §6). Either outcome is a valid result; write it into `docs/superpowers/specs/2026-09-06-batch-gpu-fanout-design.md` §6 as a dated note.

- [x] **Step 4: Run the standard MPS equivalence matrix for the lock touches**

```bash
git -C /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker worktree add --detach .worktrees/equiv-base 5977e705 2>/dev/null || true
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
REPO=$PWD WT=$PWD/.worktrees/batch-fanout MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/.worktrees/batch-fanout/src \
  OUT=/tmp/equiv_fanout RUNTIME=mps bash tools/equivalence/run_matrix.sh
```

Expected: every clip EQUIVALENT at its determinism floor with row counts > 1 (baseline = branch base `5977e705`, so the only delta is this branch).

- [x] **Step 5: CUDA on mehek**

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/batch-fanout
git bundle create /tmp/fanout.bundle main..feat/batch-gpu-fanout   # or push the branch to origin
scp /tmp/fanout.bundle rutalab@mehek.taild08eb9.ts.net:/tmp/
ssh rutalab@mehek.taild08eb9.ts.net 'cd ~/hydra-suite && git fetch /tmp/fanout.bundle feat/batch-gpu-fanout:feat/batch-gpu-fanout && git checkout feat/batch-gpu-fanout && source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda && pkill -f sleap || true; OUT=/tmp/fanout_gate bash tools/equivalence/fanout_gate.sh fly_obb worm_bgsub ant_obb_sleap ant_cnn_identity && EXTRA="--gpus 0" OUT=/tmp/fanout_gate_gpu bash tools/equivalence/fanout_gate.sh fly_obb ant_obb_sleap'
```

Expected: all ✅. The `--gpus 0` run exercises UUID pinning end to end on one physical GPU (the child log's `# gpu=GPU-…` header line proves the pin). Then run the standard matrix with `RUNTIME=cuda` per CLAUDE.md.

- [x] **Step 6: Record results and commit the gate script**

Append a "Verification results (2026-09-XX)" section to the plan with the ✅/❌ lines from steps 2-5 verbatim, then:

```bash
git add tools/equivalence/fanout_gate.sh docs/superpowers/plans/2026-09-06-batch-gpu-fanout.md docs/superpowers/specs/2026-09-06-batch-gpu-fanout-design.md
git commit -m "test(equivalence): fan-out vs sequential byte-identity gate + recorded results"
```

---

## Verification results (2026-09-06)

Branch `feat/batch-gpu-fanout` @ `7da50e2d`; baseline for the standard matrix is
the branch base `5977e705`. Gate script: `tools/equivalence/fanout_gate.sh`
(committed in this task). All runs used the fixture clips at 500 frames
(`ant_cnn_identity` 489) staged through `tools/equivalence/runner.py`'s own
`build_config`, so each sidecar is exactly what the equivalence harness would
hand that clip. Both boxes printed `(none set)` for inherited `*_NUM_THREADS`,
and every gate verified `hydra_suite` resolved inside the tree under test.

### Step 2 — MPS fan-out vs sequential (`--jobs 2`), this box, `hydra-mps`

`OUT=/tmp/fanout_gate bash tools/equivalence/fanout_gate.sh fly_obb worm_bgsub ant_obb_sleap ant_cnn_identity`

```
4/4 videos succeeded
✅ 753 child [job N] lines in par.log (children really ran)
✅ fly_obb_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ fly_obb_tracking_final_with_individual.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_final.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_backward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ worm_bgsub_tracking_final_with_individual.csv byte-identical (seq=2707 rows, par=2707 rows)
✅ worm_bgsub_tracking_final.csv byte-identical (seq=2707 rows, par=2707 rows)
✅ worm_bgsub_tracking_forward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ ant_obb_sleap_tracking_backward.csv byte-identical (seq=12501 rows, par=12501 rows)
✅ ant_obb_sleap_tracking_final_with_individual.csv byte-identical (seq=11882 rows, par=11882 rows)
✅ ant_obb_sleap_tracking_final.csv byte-identical (seq=11882 rows, par=11882 rows)
✅ ant_obb_sleap_tracking_forward.csv byte-identical (seq=12501 rows, par=12501 rows)
✅ ant_cnn_identity_tracking_backward.csv byte-identical (seq=12226 rows, par=12226 rows)
✅ ant_cnn_identity_tracking_final_with_individual.csv byte-identical (seq=10202 rows, par=10202 rows)
✅ ant_cnn_identity_tracking_final.csv byte-identical (seq=10202 rows, par=10202 rows)
✅ ant_cnn_identity_tracking_forward.csv byte-identical (seq=12226 rows, par=12226 rows)
### GATE PASSED -- fan-out output is byte-identical to sequential.
GATE_MPS_EXIT=0
```

`ant_obb_sleap` and `ant_cnn_identity` both drive SLEAP, and with two slots they
overlapped, so this is also the two-concurrent-SLEAP-service-children test the
spec's §5 asks for. The gate compares **four** CSVs per clip, not the two the
brief listed: the CLI emits `_tracking_forward`, `_tracking_backward`,
`_tracking_final` and `_tracking_final_with_individual` (the last is the only
file carrying identity/pose columns, so it is what makes the identity claim
non-vacuous). `_tracking.csv` / `_tracking_forward_processed.csv` are the
session's internal raw/final path names and are not what lands on disk here.

### Step 3 — MPS with thread caps (`--threads-per-job 4`)

Identical 16 ✅ lines and `GATE_CAPS_EXIT=0`, i.e. **caps are identity-safe on
MPS**. Non-vacuous by construction: `build_child_env` uses `setdefault`, and the
gate printed `(none set)` for the parent's `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB`
thread variables, so the children genuinely ran capped at 4 threads while the
sequential run they were compared against was uncapped. Recorded as a dated note
in spec §6; caps nevertheless **stay opt-in** (one platform only).

### Step 4 — Standard MPS equivalence matrix (`5977e705` vs `7da50e2d`)

`OUT=/tmp/equiv_fanout RUNTIME=mps bash tools/equivalence/run_matrix.sh` — all
8 fixture clips.

- **48/48 `VERDICT: EQUIVALENT ✅`, 0 `DIFFERENCES`**, across
  `DETERMINISM new_a vs new_b` and `EQUIVALENCE legacy vs new_a` for
  `_forward`, `_final` and `_final_with_individual` (the last with
  `--strict-columns`).
- Every single comparison printed literally
  `pos |Δ| (px): max=0.000e+00 mean=0.000e+00 p99=0.000e+00` and
  `theta |Δ| (rad): max=0.000e+00 mean=0.000e+00`. No θ π-flips appeared: with a
  modern baseline the orientation anchor makes θ deterministic, so this is exact
  equality rather than "at the noise floor".
- `### all clips produced comparable output.` / `MATRIX_EXIT=0`.
- Provenance (the check that matters): 8 runner lines
  `branch=HEAD commit=5977e705bd` (legacy) and 16
  `branch=feat/batch-gpu-fanout commit=7da50e2dca` (new_a + new_b).

PERFORMANCE: 7 of 8 clips within tolerance
(`ant_pose_headtail` 0.85x, `ant_obb_sequential` 0.93x, `worm_bgsub` 1.01x,
`worm_bgsub_scaled` 1.02x, `ant_cnn_identity` 1.14x,
`ant_cnn_identity_relink` 0.97x, `fly_obb` 0.99x); **`ant_obb_sleap` printed
`new/legacy time ratio = 1.30x -> PERFORMANCE: SLOWER ❌`** (legacy 106.07s vs
new 138.04s). Not attributed to this branch, on this box's own evidence: the
two *identical-code* runs of that clip in the same matrix were
`new_a=138.04s` and `new_b=173.27s`, a **1.26x same-code spread**, and
`ant_pose_headtail` likewise spread `139.76s`/`170.31s` (1.22x). SLEAP-service
clips on MPS therefore have a run-to-run noise floor at or above the 1.25x
tolerance, so a single 1.30x sample is not a signal. Corroborating: the same
clip on CUDA measured **0.99x**, and all 48 correctness comparisons are exactly
zero.
**Confirmation re-run (same box, same baseline, `ONLY=ant_obb_sleap`):**
`legacy: 124.924s (4.0 fps) new: 113.797s (4.39 fps) -> new/legacy time ratio =
0.91x -> PERFORMANCE: EQUIVALENT ✅`, with correctness again exactly zero
(`pos |Δ| max=0.000e+00`, `theta |Δ| max=0.000e+00`) and a same-code
`new_a=113.797s` / `new_b=137.554s` spread of 1.21x. Two independent samples of
the same clip therefore give 1.30x and 0.91x, straddling the tolerance in both
directions -- the 1.30x was measurement jitter, not a regression.

### Step 5 — CUDA on mehek (`hydra-cuda`, RTX 6000 Ada, single GPU)

Run twice. The **first** pair of gates ran from `~/hydra-suite` checked out to
the branch; a concurrent agent then switched that checkout to branch `m2` at
16:19:42 (git reflog), so the gates were re-run from a **dedicated detached
worktree** `.worktrees/fanout-cur` @ `7da50e2d` for a result that cannot have
been contaminated. Both pairs agree.

`--jobs 2`, four clips (isolated worktree):

```
4/4 videos succeeded
✅ 753 child [job N] lines in par.log (children really ran)
✅ fly_obb_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ fly_obb_tracking_final.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_final_with_individual.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_backward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ worm_bgsub_tracking_final.csv byte-identical (seq=2725 rows, par=2725 rows)
✅ worm_bgsub_tracking_final_with_individual.csv byte-identical (seq=2725 rows, par=2725 rows)
✅ worm_bgsub_tracking_forward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ ant_obb_sleap_tracking_backward.csv byte-identical (seq=12501 rows, par=12501 rows)
✅ ant_obb_sleap_tracking_final.csv byte-identical (seq=11843 rows, par=11843 rows)
✅ ant_obb_sleap_tracking_final_with_individual.csv byte-identical (seq=11843 rows, par=11843 rows)
✅ ant_obb_sleap_tracking_forward.csv byte-identical (seq=12501 rows, par=12501 rows)
✅ ant_cnn_identity_tracking_backward.csv byte-identical (seq=12226 rows, par=12226 rows)
✅ ant_cnn_identity_tracking_final.csv byte-identical (seq=7957 rows, par=7957 rows)
✅ ant_cnn_identity_tracking_final_with_individual.csv byte-identical (seq=7957 rows, par=7957 rows)
✅ ant_cnn_identity_tracking_forward.csv byte-identical (seq=12226 rows, par=12226 rows)
### GATE PASSED -- fan-out output is byte-identical to sequential.
GATE1_EXIT=0
```

`EXTRA="--gpus 0"` on `fly_obb` + `ant_obb_sleap`: 8 ✅, `GATE2_EXIT=0`, and the
**UUID pin is proven end to end** by the child log headers and the summary
table:

```
#  video              gpu           status  wall  log
1  fly_obb.mp4        GPU-088a4fff  OK      22s   ...
2  ant_obb_sleap.mp4  GPU-088a4fff  OK      47s   ...
# gpu=GPU-088a4fff-9dff-dcce-5c6e-b7b29fc28177 command=.../python -m hydra_suite.trackerkit.app ...
```

`CUDA_VISIBLE_DEVICES` is set to the device **UUID**, not the ordinal. Note the
gate passes `--jobs 2` while `--gpus 0` names one device, so slots clamp to
`min(2, 1) = 1` — the table shows the two videos running one at a time, which is
the documented behaviour, not a scheduling failure.

**CUDA standard matrix** (`MAIN_SRC=.worktrees/fanout-base` @ `5977e705`,
`WT_SRC=.worktrees/fanout-cur` @ `7da50e2d`, `RUNTIME=cuda`), all 8 clips:

- **48/48 `VERDICT: EQUIVALENT ✅`, 0 `DIFFERENCES`, 0 ❌ of any kind**; every
  comparison `pos |Δ| max=0.000e+00`, `theta |Δ| max=0.000e+00`.
- PERFORMANCE **8/8 EQUIVALENT ✅**: 0.97x, 0.99x, 0.98x, 1.01x, 1.01x, 1.01x,
  1.01x, 0.98x.
- `### all clips produced comparable output.` / `MATRIX_EXIT=0`.
- Provenance: legacy runs `commit=5977e705bd`, new runs `commit=7da50e2dca`.

A first CUDA matrix attempt was **discarded as invalid** and none of its numbers
are quoted: mehek already had a `.worktrees/equiv-base` pinned at an unrelated
old commit (`81e3736b`) so `git worktree add` failed and `MAIN_SRC` was the wrong
baseline, and the concurrent `m2` checkout swapped `WT_SRC` mid-run. `grep -E
"branch=.*commit=" <matrix log>` catches both failures in seconds and should be
the first thing checked on any matrix log.

### Conclusion

Fan-out output is byte-identical to sequential output on both platforms, with and
without thread caps, including two concurrent SLEAP service children and a
UUID-pinned GPU child; and the artifact-lock touches in `runtime_artifacts.py`,
`pose/backends/sleap.py`, `pose/runtime/onnx_session.py` and
`classification/backend.py` changed nothing on the standard matrix — 96 of 96
comparisons across the two platforms are exactly zero.

---

## Fix wave (2026-09-07)

Adversarial review of `4661e0b6` (findings in `/tmp/batch-fanout-adversarial/adversarial-findings.md`;
full write-up in `/tmp/batch-fanout-adversarial/fix-wave-report.md`). Nine commits,
`4661e0b6` → `HEAD`: C1 stranded grandchildren, C2 shared `video_output_path`,
I2 GUI silently unpinned, I3 in-place artifact rebuild, I4 lock-test PYTHONPATH,
I5 window-close budget, plus the minor folds, this gate leg, and a self-review
follow-up that fixes a defect introduced by the I3 fix itself (a `finally` that
deleted the displaced good artifact when the swap-in rename failed). No
inference numerics changed.

### Why the original gates missed C2

`fanout_gate.sh` wrote a per-video sidecar for EVERY clip, so every job was
"own-sidecar" and the planner's keystone-baseline branch — the default GUI batch
flow — was never executed; `runner.py`'s `DISABLE` block also forces
`video_output_enabled` off, so the annotated video (the one output actually taken
from the inherited config) was invisible. `INHERIT=1` fixes both: a sidecar for
the FIRST clip only (with `video_output_enabled: true`), the rest inherit, and
the gate additionally asserts that every clip rendered its own
`<stem>_tracking.mp4` beside its own video in BOTH legs.

### RED evidence for C2 (MPS, `INHERIT=1`, `fly_obb worm_bgsub`, planner reverted to pre-C2)

```
== side outputs: every clip must render its OWN annotated video ==
✅ seq/fly_obb_tracking.mp4 (53680063 bytes)
❌ seq/worm_bgsub_tracking.mp4 missing or empty -- no overlay of its own
❌ seq: 2 clips -> 1 annotated videos (paths collided)
✅ par/fly_obb_tracking.mp4 (6922411 bytes)
❌ par/worm_bgsub_tracking.mp4 missing or empty -- no overlay of its own
❌ par: 2 clips -> 1 annotated videos (paths collided)
### GATE FAILED
```

Note the byte counts: in the sequential leg `fly_obb_tracking.mp4` is **53.7 MB**
— that is the *worm* render, which overwrote the keystone's own overlay (the fly
render is 6.9 MB, as the fan-out leg shows, where the race went the other way).
The 8 CSV comparisons were ✅ throughout, which is exactly why this shipped.

### MPS — default leg (`fly_obb worm_bgsub ant_obb_sleap`, `--jobs 2`, `hydra-mps`)

`rc=0` both legs, `3/3 videos succeeded`, 526 child `[job N]` lines.

```
✅ fly_obb_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ fly_obb_tracking_final_with_individual.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_final.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_backward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ worm_bgsub_tracking_final_with_individual.csv byte-identical (seq=2707 rows, par=2707 rows)
✅ worm_bgsub_tracking_final.csv byte-identical (seq=2707 rows, par=2707 rows)
✅ worm_bgsub_tracking_forward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ ant_obb_sleap_tracking_backward.csv byte-identical (seq=12501 rows, par=12501 rows)
✅ ant_obb_sleap_tracking_final_with_individual.csv byte-identical (seq=11882 rows, par=11882 rows)
✅ ant_obb_sleap_tracking_final.csv byte-identical (seq=11882 rows, par=11882 rows)
✅ ant_obb_sleap_tracking_forward.csv byte-identical (seq=12501 rows, par=12501 rows)
### GATE PASSED -- fan-out output is byte-identical to sequential.
```

### MPS — INHERIT leg (keystone-only sidecar + annotated video)

`rc=0` both legs, `3/3 videos succeeded`, 678 child `[job N]` lines. The
inheriting clips run the fly-OBB keystone config, so their row counts differ from
the default leg by construction — what matters is seq == par, and that all three
overlays exist.

```
✅ fly_obb_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ fly_obb_tracking_final_with_individual.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_final.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_final_with_individual.csv byte-identical (seq=480 rows, par=480 rows)
✅ worm_bgsub_tracking_final.csv byte-identical (seq=480 rows, par=480 rows)
✅ worm_bgsub_tracking_forward.csv byte-identical (seq=1474 rows, par=1474 rows)
✅ ant_obb_sleap_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ ant_obb_sleap_tracking_final_with_individual.csv byte-identical (seq=1089 rows, par=1089 rows)
✅ ant_obb_sleap_tracking_final.csv byte-identical (seq=1089 rows, par=1089 rows)
✅ ant_obb_sleap_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)

== side outputs: every clip must render its OWN annotated video ==
✅ seq/fly_obb_tracking.mp4 (6922411 bytes)
✅ seq/worm_bgsub_tracking.mp4 (53680063 bytes)
✅ seq/ant_obb_sleap_tracking.mp4 (172692960 bytes)
✅ seq: 3 clips -> 3 distinct annotated videos
✅ par/fly_obb_tracking.mp4 (6922411 bytes)
✅ par/worm_bgsub_tracking.mp4 (53680063 bytes)
✅ par/ant_obb_sleap_tracking.mp4 (172692960 bytes)
✅ par: 3 clips -> 3 distinct annotated videos
### GATE PASSED -- fan-out output is byte-identical to sequential.
```

Each clip's overlay is byte-for-byte the same size in `seq` and `par`.

### CUDA (mehek, `hydra-cuda`, RTX 6000 Ada) — default leg, `--jobs 2`

Dedicated detached worktree `~/hydra-suite/.worktrees/fanout-fixwave` @ `4c635bdf`
(transported as a git bundle over scp); `FIXTURES` pointed at the main checkout's
fixtures (clips and models are gitignored). `rc=0` both legs, `3/3 videos
succeeded`, 526 child `[job N]` lines.

```
✅ fly_obb_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ fly_obb_tracking_final.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_final_with_individual.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_backward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ worm_bgsub_tracking_final.csv byte-identical (seq=2725 rows, par=2725 rows)
✅ worm_bgsub_tracking_final_with_individual.csv byte-identical (seq=2725 rows, par=2725 rows)
✅ worm_bgsub_tracking_forward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ ant_obb_sleap_tracking_backward.csv byte-identical (seq=12501 rows, par=12501 rows)
✅ ant_obb_sleap_tracking_final.csv byte-identical (seq=11843 rows, par=11843 rows)
✅ ant_obb_sleap_tracking_final_with_individual.csv byte-identical (seq=11843 rows, par=11843 rows)
✅ ant_obb_sleap_tracking_forward.csv byte-identical (seq=12501 rows, par=12501 rows)
### GATE PASSED -- fan-out output is byte-identical to sequential.
```

### CUDA (mehek) — INHERIT leg, `--jobs 2 --gpus 0`

`rc=0` both legs, `3/3 videos succeeded`, 678 child `[job N]` lines, every child
header showing `# gpu=GPU-088a4fff-9dff-dcce-5c6e-b7b29fc28177` (the UUID pin,
not an ordinal).

```
✅ fly_obb_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ fly_obb_tracking_final.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_final_with_individual.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_backward.csv byte-identical (seq=1480 rows, par=1480 rows)
✅ worm_bgsub_tracking_final.csv byte-identical (seq=503 rows, par=503 rows)
✅ worm_bgsub_tracking_final_with_individual.csv byte-identical (seq=503 rows, par=503 rows)
✅ worm_bgsub_tracking_forward.csv byte-identical (seq=1492 rows, par=1492 rows)
✅ ant_obb_sleap_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ ant_obb_sleap_tracking_final.csv byte-identical (seq=1089 rows, par=1089 rows)
✅ ant_obb_sleap_tracking_final_with_individual.csv byte-identical (seq=1089 rows, par=1089 rows)
✅ ant_obb_sleap_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)

== side outputs: every clip must render its OWN annotated video ==
✅ seq/fly_obb_tracking.mp4 (4311711 bytes)
✅ seq/worm_bgsub_tracking.mp4 (6383494 bytes)
✅ seq/ant_obb_sleap_tracking.mp4 (52679380 bytes)
✅ seq: 3 clips -> 3 distinct annotated videos
✅ par/fly_obb_tracking.mp4 (4311711 bytes)
✅ par/worm_bgsub_tracking.mp4 (6383494 bytes)
✅ par/ant_obb_sleap_tracking.mp4 (52679380 bytes)
✅ par: 3 clips -> 3 distinct annotated videos
### GATE PASSED -- fan-out output is byte-identical to sequential.
```

**Honesty note:** mehek was concurrently running an unrelated SAM3 LoRA training
job (~20 GB of 49 GB VRAM, GPU at 100%) throughout both CUDA legs. It was left
running (never a sleap/hydra process of ours). The gate asserts byte-identity
only — it makes no timing claim — so contention does not affect these verdicts,
but no perf number should be read off these runs.

### Fix-wave test summary (all `hydra-mps`, one file at a time)

| file | result |
| --- | --- |
| `tests/test_trackerkit_batch_fanout.py` | 25 passed |
| `tests/test_trackerkit_batch_plan.py` | 16 passed |
| `tests/test_trackerkit_batch_fanout_worker.py` | 16 passed |
| `tests/test_trackerkit_cli_fanout.py` | 14 passed |
| `tests/test_artifact_lock.py` | 8 passed (also with `env -u PYTHONPATH`) |
| `tests/test_inference_obb_artifacts.py` | 28 passed, 1 skipped |
| `tests/test_sleap_export_crop_normalization.py` | 8 passed |
| `tests/test_runtime_api_sleap_export.py` | 11 passed |
| `tests/test_sleap_export_predict_worker.py` | 1 passed |
| `tests/test_cuda_devices.py` | 10 passed |
| `tests/test_trackerkit_config_schema_fanout.py` | 4 passed |

## Fix wave 3 (2026-09-07)

Adversarial re-review of `4661e0b6 -> da4dd3ea` raised N1 (`_retarget_side_outputs`
also fired on `explicit` provenance), N2 (`_load_coreml_executor` exported
unlocked and in place), gate hardening, and the I5 escape hatch. Commits
`04169ca8`, `e8d5ab22`, `409ff8fb`, `4ad86454`, `7dca1c86`.

**mehek is NOT required for this wave.** The CUDA CSV path is unchanged: N1 only
moves side-output paths (`file_path`/`csv_path`/`video_output_path` — none of
which the engine reads), and for the fan-out child the re-plan has
`len(plan) == 1`, so the branch it now takes is the one that leaves the config
alone. N2 touches the CoreML loader only (Apple-only; the TRT loader already had
the lock). I5 is GUI-only. Nothing here can change a CUDA number.

### Why the previous INHERIT leg was blind to N1

It staged the keystone at the DEFAULT `<stage>/<clip>_tracking.mp4` — exactly
where a retargeting planner sends a borrowed config — so "honoured the user's
path" and "overwrote it with the default" produced the same filename. The
keystone now renders to `<stage>/renders/<clip>_CUSTOM.mp4`.

### RED evidence for N1 (MPS, `INHERIT=1`, `fly_obb worm_bgsub`, planner at `da4dd3ea`)

All eight CSVs byte-identical — the CSVs cannot see this bug — and then:

```
== side outputs: the keystone keeps ITS path, borrowers get their own ==
✅ seq/renders/fly_obb_CUSTOM.mp4 (6922411 bytes) -- keystone path honoured
✅ seq: keystone did not render to the default fly_obb_tracking.mp4
✅ seq/worm_bgsub_tracking.mp4 (53680063 bytes) -- borrower renders beside its own video
✅ seq: 1 borrowers -> 1 distinct annotated videos
❌ par/renders/fly_obb_CUSTOM.mp4 missing or empty -- the keystone's own render path was discarded
❌ par/fly_obb_tracking.mp4 exists -- the keystone rendered to the DEFAULT path, not the one it names
✅ par/worm_bgsub_tracking.mp4 (53680063 bytes) -- borrower renders beside its own video
❌ par: 1 borrowers -> 2 annotated videos beside the clips (paths collided)
❌ keystone custom render: seq='6922411' par='<missing>' bytes -- the legs rendered differently
### GATE FAILED -- see the ❌ lines above.
```

### Render byte size is NOT an invariant on macOS (measured)

The first `gpu_fast` leg failed only on `❌ keystone custom render: seq='6922411'
par='6260151' bytes`, with every CSV byte-identical. Direct measurement of
`h264_videotoolbox` on the same 300 frames of `fly_obb.mp4`:

| condition | bytes | md5 |
| --- | --- | --- |
| serial run 1 | 4005825 | `b7344dd0…` |
| serial run 2 | 4005825 | `e67608b7…` |
| 2-way concurrent, run 1 | 3794061 | `ec804f37…` |
| 2-way concurrent, run 2 | 3794061 | `a9551180…` |

The encoder is byte-nondeterministic even for identical input, and its bitrate is
load-sensitive. The fan-out leg has encoder contention and the sequential leg does
not, so size equality measures the encoder, not the pipeline. Size identity was
requested for this gate; it was **replaced** with frame-count/geometry/fps
identity — what a lossy encoder does preserve — for the keystone AND every
borrower (a borrower's size difference previously slipped through a bare
non-empty check). On the failing run those matched exactly
(`500 frames, 1200x1200 @ 100.0 fps` and `500 frames, 1920x1200 @ 5.0 fps`),
confirming the ❌ was the encoder.

### MPS — default leg (`fly_obb worm_bgsub ant_obb_sleap`, `--jobs 2`, `hydra-mps`)

```
✅ fly_obb_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ fly_obb_tracking_final_with_individual.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_final.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_backward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ worm_bgsub_tracking_final_with_individual.csv byte-identical (seq=2707 rows, par=2707 rows)
✅ worm_bgsub_tracking_final.csv byte-identical (seq=2707 rows, par=2707 rows)
✅ worm_bgsub_tracking_forward.csv byte-identical (seq=5001 rows, par=5001 rows)
✅ ant_obb_sleap_tracking_backward.csv byte-identical (seq=12501 rows, par=12501 rows)
✅ ant_obb_sleap_tracking_final_with_individual.csv byte-identical (seq=11882 rows, par=11882 rows)
✅ ant_obb_sleap_tracking_final.csv byte-identical (seq=11882 rows, par=11882 rows)
✅ ant_obb_sleap_tracking_forward.csv byte-identical (seq=12501 rows, par=12501 rows)
### GATE PASSED -- fan-out output is byte-identical to sequential.
```

### MPS — INHERIT leg with a custom keystone render (`fly_obb worm_bgsub ant_obb_sleap`)

Twelve CSVs byte-identical (as above, INHERIT row counts), then:

```
== side outputs: the keystone keeps ITS path, borrowers get their own ==
✅ seq/renders/fly_obb_CUSTOM.mp4 (6922411 bytes) -- keystone path honoured
✅ seq: keystone did not render to the default fly_obb_tracking.mp4
✅ seq/worm_bgsub_tracking.mp4 (53680063 bytes) -- borrower renders beside its own video
✅ seq/ant_obb_sleap_tracking.mp4 (172692960 bytes) -- borrower renders beside its own video
✅ seq: 2 borrowers -> 2 distinct annotated videos
✅ par/renders/fly_obb_CUSTOM.mp4 (6922411 bytes) -- keystone path honoured
✅ par: keystone did not render to the default fly_obb_tracking.mp4
✅ par/worm_bgsub_tracking.mp4 (53680063 bytes) -- borrower renders beside its own video
✅ par/ant_obb_sleap_tracking.mp4 (172692960 bytes) -- borrower renders beside its own video
✅ par: 2 borrowers -> 2 distinct annotated videos

-- every render: same content on both legs --
✅ renders/fly_obb_CUSTOM.mp4: 500 frames, 1200x1200 @ 100.0 fps on both legs
✅ worm_bgsub_tracking.mp4: 500 frames, 1920x1200 @ 5.0 fps on both legs
✅ ant_obb_sleap_tracking.mp4: 500 frames, 4512x4512 @ 25.0 fps on both legs
### GATE PASSED -- fan-out output is byte-identical to sequential.
```

### MPS — `gpu_fast` (CoreML) INHERIT leg from a fresh artifact state (`--jobs 2`)

`STAGE_MODELS=1` copies the checkpoint into each leg's own staging dir and
repoints the sidecar, so `par/` starts with NO `.mlpackage` and both children
miss the cache at once. Under INHERIT, `worm_bgsub` borrows `fly_obb`'s config,
so both children load the same OBB model — which is what makes them race.

```
✅ fly_obb_tracking_backward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ fly_obb_tracking_final_with_individual.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_final.csv byte-identical (seq=1501 rows, par=1501 rows)
✅ fly_obb_tracking_forward.csv byte-identical (seq=1495 rows, par=1495 rows)
✅ worm_bgsub_tracking_backward.csv byte-identical (seq=1480 rows, par=1480 rows)
✅ worm_bgsub_tracking_final_with_individual.csv byte-identical (seq=489 rows, par=489 rows)
✅ worm_bgsub_tracking_final.csv byte-identical (seq=489 rows, par=489 rows)
✅ worm_bgsub_tracking_forward.csv byte-identical (seq=1474 rows, par=1474 rows)

-- every render: same content on both legs --
✅ renders/fly_obb_CUSTOM.mp4: 500 frames, 1200x1200 @ 100.0 fps on both legs
✅ worm_bgsub_tracking.mp4: 500 frames, 1920x1200 @ 5.0 fps on both legs

== first-run artifact build: the racing children must publish ONE artifact ==
✅ seq: 1 derived artifact(s) under models/
✅ seq: freshness marker present for 20260503-171130_26x_fly_train7.mlpackage
✅ seq: no staging/displaced artifact leftovers
✅ par: 1 derived artifact(s) under models/
✅ par: freshness marker present for 20260503-171130_26x_fly_train7.mlpackage
✅ par: no staging/displaced artifact leftovers
✅ par: exactly 1 child exported the artifact (the other waited on the lock)
✅ par: 1 child(ren) reused the artifact built by another process
### GATE PASSED -- fan-out output is byte-identical to sequential.
```

The last two lines are the only direct end-to-end evidence for N2: "one artifact
survives" is also what two racing exporters leave behind, because the atomic
installer tidies up after the loser. In the children's own logs, job 2 logged
`Exported CoreML artifact` once and job 1 logged
`Reusing CoreML artifact built by another process` once — the double-check
inside the lock did its job.

### Fix-wave-3 test summary (all `hydra-mps`, one file at a time)

| file | result |
| --- | --- |
| `tests/test_trackerkit_batch_plan.py` | 18 passed (+2 new, both RED at `da4dd3ea`) |
| `tests/test_obb_coreml_export.py` | 12 passed (+3 new, all RED pre-N2) |
| `tests/test_trackerkit_batch_fanout_worker.py` | 20 passed (+4 new, 3 RED pre-I5) |
| `tests/test_trackerkit_batch_fanout.py` | 25 passed |
| `tests/test_trackerkit_cli_fanout.py` | 14 passed |
| `tests/test_trackerkit_cli_config.py` | 10 passed |
| `tests/test_inference_obb_artifacts.py` | 28 passed, 1 skipped |
| `tests/test_artifact_lock.py` | 8 passed |
| `tests/test_coreml_determinism.py` | 2 passed |
| `tests/test_sleap_export_crop_normalization.py` | 8 passed |

No existing test pinned the in-place CoreML export, so none had to be relaxed.
`test_execute_runs_the_scheduler_and_emits_the_result`'s fake scheduler was
updated for the new `child_registry` kwarg (and now asserts the worker passes
its own registry).

## Self-review notes

- Spec §4.1 planner → Task 3; §4.2 devices → Task 2; §4.3 scheduler → Task 4; §4.4 locks → Task 1; §4.5 CLI → Task 5; §4.6 schema → Task 6; §4.7 GUI → Tasks 6-7; §4.8 docs → Task 8; §5 SLEAP → covered by inheritance (no code) and gated in Task 9 step 2; §6 thread caps → Task 9 step 3; §7 verification → Task 9.
- The spec's `explicit_config_data` planner parameter was dropped: the GUI saves the keystone sidecar before starting, so `plan_batch_jobs(batch_videos, keystone_override=…)` sees exactly what the CLI `--video-list` path sees. The spec is amended in the same commit as this plan.
- Names used across tasks: `BatchJobSpec`, `plan_batch_jobs`, `BatchPlanError`, `CudaDevice`, `list_cuda_devices`, `parse_gpu_selectors`, `resolve_gpu_selectors`, `FanoutOptions`, `FanoutJobResult`, `FanoutResult`, `run_batch_fanout`, `build_child_env`, `default_child_command`, `parse_progress_line`, `parse_summary_line`, `artifact_build_lock`, `fanout_requested`, `_run_sequential`, `_run_fanout`, `BatchFanoutWorker`, `BatchFanoutDialog`, `start_batch_fanout`.
