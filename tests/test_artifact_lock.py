"""Cross-process blocking lock used around first-run artifact builds."""

from __future__ import annotations

import logging
import os
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


def test_lock_degrades_when_directory_is_read_only(tmp_path, caplog):
    """A read-only model directory must not make the build fail.

    The lock is an optimisation for CONCURRENT builders; a directory nobody can
    write to is one nobody can build into either. Degrading to unlocked
    operation restores exactly the single-process behaviour that shipped before
    the lock existed.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    target = ro_dir / "model.engine"
    os.chmod(ro_dir, 0o555)
    try:
        with caplog.at_level(
            logging.WARNING, logger="hydra_suite.runtime.artifact_lock"
        ):
            entered = False
            with artifact_build_lock(target):
                entered = True
        assert entered, "degraded lock must still run the body exactly once"
        assert not (ro_dir / "model.engine.lock").exists()
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            str(target) in msg or "model.engine.lock" in msg for msg in warnings
        ), warnings
    finally:
        os.chmod(ro_dir, 0o755)


def test_degraded_lock_still_propagates_body_exceptions(tmp_path):
    """Degrading must not turn the ``with`` body into a swallowed exception."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    ro_dir = tmp_path / "readonly2"
    ro_dir.mkdir()
    os.chmod(ro_dir, 0o555)
    try:
        with pytest.raises(ValueError):
            with artifact_build_lock(ro_dir / "m.engine"):
                raise ValueError("boom")
    finally:
        os.chmod(ro_dir, 0o755)


def test_tensorrt_ep_lock_target_lives_in_the_engine_cache(tmp_path, monkeypatch):
    """The TRT-EP lock belongs beside the resource it guards: the ORT engine
    cache dir, which is writable by construction -- NOT beside the model, which
    may sit on a read-only share."""
    import hydra_suite.paths as paths_mod
    from hydra_suite.runtime.onnx_providers import tensorrt_ep_lock_target

    monkeypatch.setattr(paths_mod, "get_data_dir", lambda: tmp_path)
    model = tmp_path / "models" / "clf.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"\x00")

    target = tensorrt_ep_lock_target(model)
    cache_dir = tmp_path / "trt_engine_cache"
    assert cache_dir.is_dir()
    assert target.parent == cache_dir
    assert target.suffix == ".trt_ep"
    # Stable for the same model, distinct for a different one.
    assert target == tensorrt_ep_lock_target(str(model))
    other = tmp_path / "models" / "pose.onnx"
    other.write_bytes(b"\x00")
    assert tensorrt_ep_lock_target(other) != target
