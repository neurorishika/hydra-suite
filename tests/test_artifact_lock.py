"""Cross-process blocking lock used around first-run artifact builds."""

from __future__ import annotations

import subprocess
import sys
import time

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
