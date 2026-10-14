"""The per-GPU phase semaphore (runtime.gpu_phase_lock), with real processes.

No GPU: the semaphore is pure file locking. Each "child" is a tiny python
process that takes a phase and reports through marker files, so the tests see
exactly what two fan-out children sharing one GPU would.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hydra_suite
from hydra_suite.runtime import gpu_phase_lock as gpl

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX flock semantics (Windows uses msvcrt)"
)

_SRC = str(Path(hydra_suite.__file__).resolve().parents[1])

# argv: <marker_dir> <name> [hold_until_file]
_HOLDER = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[4])
from hydra_suite.runtime.gpu_phase_lock import gpu_phase
marker_dir, name, release = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with gpu_phase(name, poll_s=0.02):
    (marker_dir / (name + ".acquired")).write_text(str(time.monotonic()))
    while not (release and Path(release).exists()):
        time.sleep(0.02)
(marker_dir / (name + ".released")).write_text(str(time.monotonic()))
"""


def _env(lock_dir: Path, slots: int = 1, gpu: str = "GPU-aaaa") -> dict:
    env = dict(os.environ)
    env.update(gpl.phase_lock_env(lock_dir, gpu, slots))
    return env


def _spawn(tmp_path: Path, name: str, env: dict, release: str = "") -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(tmp_path), name, release, _SRC],
        env=env,
    )


def _wait_for(path: Path, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def _reap(*procs: subprocess.Popen) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


def test_second_child_waits_until_the_first_releases(tmp_path):
    lock_dir = tmp_path / "locks"
    release_a = tmp_path / "release_a"
    a = _spawn(tmp_path, "a", _env(lock_dir), str(release_a))
    b = None
    try:
        assert _wait_for(tmp_path / "a.acquired")
        b = _spawn(tmp_path, "b", _env(lock_dir), str(tmp_path / "release_b"))
        time.sleep(0.6)
        assert not (tmp_path / "b.acquired").exists(), "K=1: b must wait for a"
        release_a.write_text("go")
        assert _wait_for(tmp_path / "b.acquired"), "b never got the freed slot"
        assert a.wait(timeout=10) == 0
        (tmp_path / "release_b").write_text("go")
        assert b.wait(timeout=10) == 0
    finally:
        _reap(a, *([b] if b else []))


def test_a_killed_holder_frees_its_slot(tmp_path):
    lock_dir = tmp_path / "locks"
    a = _spawn(tmp_path, "a", _env(lock_dir))  # never releases on its own
    b = None
    try:
        assert _wait_for(tmp_path / "a.acquired")
        b = _spawn(tmp_path, "b", _env(lock_dir), str(tmp_path / "release_b"))
        time.sleep(0.4)
        assert not (tmp_path / "b.acquired").exists()
        os.kill(a.pid, signal.SIGKILL)
        a.wait(timeout=10)
        assert _wait_for(tmp_path / "b.acquired"), "a dead holder kept the slot"
        (tmp_path / "release_b").write_text("go")
        assert b.wait(timeout=10) == 0
    finally:
        _reap(a, *([b] if b else []))


def test_k2_admits_two_at_once_and_queues_the_third(tmp_path):
    lock_dir = tmp_path / "locks"
    procs = [
        _spawn(tmp_path, n, _env(lock_dir, slots=2), str(tmp_path / f"release_{n}"))
        for n in ("a", "b")
    ]
    try:
        assert _wait_for(tmp_path / "a.acquired")
        assert _wait_for(tmp_path / "b.acquired")
        procs.append(
            _spawn(tmp_path, "c", _env(lock_dir, slots=2), str(tmp_path / "release_c"))
        )
        time.sleep(0.5)
        assert not (tmp_path / "c.acquired").exists()
        (tmp_path / "release_b").write_text("go")
        assert _wait_for(tmp_path / "c.acquired")
        for n in ("a", "c"):
            (tmp_path / f"release_{n}").write_text("go")
        assert [p.wait(timeout=10) for p in procs] == [0, 0, 0]
    finally:
        _reap(*procs)


def test_different_gpus_do_not_share_slots(tmp_path):
    lock_dir = tmp_path / "locks"
    a = _spawn(tmp_path, "a", _env(lock_dir, gpu="GPU-aaaa"), str(tmp_path / "ra"))
    b = _spawn(tmp_path, "b", _env(lock_dir, gpu="GPU-bbbb"), str(tmp_path / "rb"))
    try:
        assert _wait_for(tmp_path / "a.acquired")
        assert _wait_for(tmp_path / "b.acquired")
        (tmp_path / "ra").write_text("go")
        (tmp_path / "rb").write_text("go")
        assert (a.wait(timeout=10), b.wait(timeout=10)) == (0, 0)
    finally:
        _reap(a, b)


def test_without_env_it_is_a_no_op(tmp_path, monkeypatch):
    for var in gpl.ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not gpl.gpu_phase_enabled()
    with gpl.gpu_phase("x") as lease:
        assert not lease.active and not lease.acquired
    # A no-op phase must not pull torch in.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from hydra_suite.runtime.gpu_phase_lock import gpu_phase\n"
        "with gpu_phase('x'): pass\n"
        "print('torch' in sys.modules)"
    )
    env = {k: v for k, v in os.environ.items() if k not in gpl.ENV_VARS}
    out = subprocess.run(
        [sys.executable, "-c", code, _SRC], env=env, capture_output=True, text=True
    )
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.parametrize(
    "env",
    [
        {gpl.ENV_LOCK_DIR: "/tmp/x", gpl.ENV_GPU: "GPU-a"},
        {gpl.ENV_LOCK_DIR: "/tmp/x", gpl.ENV_GPU: "GPU-a", gpl.ENV_SLOTS: "0"},
        {gpl.ENV_LOCK_DIR: "/tmp/x", gpl.ENV_GPU: "GPU-a", gpl.ENV_SLOTS: "two"},
        {gpl.ENV_GPU: "GPU-a", gpl.ENV_SLOTS: "1"},
    ],
)
def test_incomplete_or_invalid_env_disables_the_lock(env):
    assert gpl.config_from_env(env) is None


def test_nested_phases_share_one_slot(tmp_path, monkeypatch):
    for key, value in gpl.phase_lock_env(tmp_path / "locks", "GPU-a", 1).items():
        monkeypatch.setenv(key, value)
    with gpl.gpu_phase("outer", poll_s=0.01) as outer:
        assert outer.acquired
        # K=1: a second acquisition would deadlock if it were not re-entrant.
        with gpl.gpu_phase("inner", poll_s=0.01) as inner:
            assert inner.acquired
        # The inner exit must not have freed the outer's slot.
        assert gpl._held["depth"] == 1
    assert gpl._held["depth"] == 0


def test_stop_while_waiting_returns_an_unacquired_lease(tmp_path, monkeypatch):
    lock_dir = tmp_path / "locks"
    a = _spawn(tmp_path, "a", _env(lock_dir), str(tmp_path / "ra"))
    try:
        assert _wait_for(tmp_path / "a.acquired")
        for key, value in gpl.phase_lock_env(lock_dir, "GPU-aaaa", 1).items():
            monkeypatch.setenv(key, value)
        calls = []

        def _stop():
            calls.append(1)
            return len(calls) >= 3

        lease = gpl.acquire_gpu_phase("waiter", should_stop=_stop, poll_s=0.01)
        assert lease.active and not lease.acquired
        lease.release()  # harmless
        assert gpl._held["depth"] == 0
        (tmp_path / "ra").write_text("go")
        assert a.wait(timeout=10) == 0
    finally:
        _reap(a)


def test_release_frees_torch_cache_only_when_cuda_is_initialised(monkeypatch, tmp_path):
    calls = []

    class _Cuda:
        def __init__(self, initialised):
            self._initialised = initialised

        def is_initialized(self):
            return self._initialised

        def empty_cache(self):
            calls.append("empty")

    class _Torch:
        cuda = _Cuda(True)

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    for key, value in gpl.phase_lock_env(tmp_path / "locks", "GPU-a", 1).items():
        monkeypatch.setenv(key, value)
    with gpl.gpu_phase("x"):
        assert calls == []
    assert calls == ["empty"]
    _Torch.cuda = _Cuda(False)
    with gpl.gpu_phase("y"):
        pass
    assert calls == ["empty"]
