"""Review fixes for the fan-out GPU phase lock (no GPU, no models).

- replay (cache_only) runners load no model at all;
- a forward runner can defer its models until a pass needs them;
- holds_gpu_models() decides whether a pass needs a GPU slot;
- two threads of one process never hold two slots;
- a child never inherits a stale semaphore env;
- a stop while waiting skips the session's GPU body;
- the wait log counts other waiters.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hydra_suite.runtime import gpu_phase_lock as gpl

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX flock semantics")


def _obb_cfg():
    from hydra_suite.core.inference.config import (
        InferenceConfig,
        OBBConfig,
        OBBDirectConfig,
    )

    return InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/fake.pt"))
    )


@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    for key, value in gpl.phase_lock_env(tmp_path / "locks", "GPU-a", 1).items():
        monkeypatch.setenv(key, value)
    return tmp_path / "locks"


@pytest.fixture(autouse=True)
def _no_lock_env(monkeypatch):
    for var in gpl.ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --- 1. replay runners load no detector ---------------------------------------


def test_cache_only_loads_no_detector_model():
    from hydra_suite.core.inference.runner import _load_all_models

    with patch(
        "hydra_suite.core.inference.stages.obb.load_obb_models",
        side_effect=AssertionError("cache_only must not load the detector"),
    ):
        models = _load_all_models(_obb_cfg(), MagicMock(), cache_only=True)
    assert models.obb is None and models.bgsub is None


def test_cache_only_runner_holds_no_gpu_models(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    with (
        patch(
            "hydra_suite.core.inference.runner.RuntimeContext.from_config",
            return_value=MagicMock(),
        ),
        patch(
            "hydra_suite.core.inference.stages.obb.load_obb_models",
            side_effect=AssertionError("cache_only must not load the detector"),
        ),
    ):
        runner = InferenceRunner(_obb_cfg(), cache_dir=tmp_path, cache_only=True)
    assert not runner.holds_gpu_models()
    assert runner.runtime_artifact_ids == ()


# --- 3. deferred model loading ------------------------------------------------


def _loaded_models():
    from hydra_suite.core.inference.runner import _AllModels

    obb = MagicMock(name="obb")
    obb.runtime_artifact_ids = ("engine-1",)
    obb.runtime_artifact_prepare_seconds = 1.5
    return _AllModels(obb=obb, headtail=None, cnn=[], pose=None, apriltag=None)


def test_deferred_runner_loads_models_only_on_first_use(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    calls = []

    def _load(config, runtime, *, cache_only=False, video_path=None):
        calls.append(cache_only)
        return _loaded_models()

    with (
        patch(
            "hydra_suite.core.inference.runner.RuntimeContext.from_config",
            return_value=MagicMock(),
        ),
        patch("hydra_suite.core.inference.runner._load_all_models", side_effect=_load),
    ):
        runner = InferenceRunner(_obb_cfg(), cache_dir=tmp_path, load_models=False)
        assert calls == [] and not runner.holds_gpu_models()
        runner._ensure_models()
        assert calls == [False] and runner.holds_gpu_models()
        assert runner.runtime_artifact_ids == ("engine-1",)
        runner._ensure_models()
        assert calls == [False]  # once


def test_deferred_models_load_before_a_batch_pass(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    with (
        patch(
            "hydra_suite.core.inference.runner.RuntimeContext.from_config",
            return_value=MagicMock(),
        ),
        patch(
            "hydra_suite.core.inference.runner._load_all_models",
            side_effect=lambda *a, **k: _loaded_models(),
        ),
    ):
        runner = InferenceRunner(_obb_cfg(), cache_dir=tmp_path, load_models=False)
        with patch("hydra_suite.core.inference.sources.make_frame_source") as src:
            src.side_effect = RuntimeError("stop after the model load")
            with pytest.raises(RuntimeError, match="stop after the model load"):
                runner.run_batch_pass(tmp_path / "v.mp4")
        assert runner.holds_gpu_models()


def test_default_runner_still_loads_eagerly(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    with (
        patch(
            "hydra_suite.core.inference.runner.RuntimeContext.from_config",
            return_value=MagicMock(),
        ),
        patch(
            "hydra_suite.core.inference.runner._load_all_models",
            side_effect=lambda *a, **k: _loaded_models(),
        ) as load,
    ):
        runner = InferenceRunner(_obb_cfg(), cache_dir=tmp_path)
    load.assert_called_once()
    assert runner.holds_gpu_models()


def test_release_models_leaves_nothing_on_the_gpu(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    models = _loaded_models()
    with (
        patch(
            "hydra_suite.core.inference.runner.RuntimeContext.from_config",
            return_value=MagicMock(),
        ),
        patch(
            "hydra_suite.core.inference.runner._load_all_models",
            return_value=models,
        ),
    ):
        runner = InferenceRunner(_obb_cfg(), cache_dir=tmp_path)
    obb = models.obb
    runner.release_models()
    assert not runner.holds_gpu_models()
    obb.close.assert_called_once()
    runner.close()
    obb.close.assert_called_once()  # not closed twice


# --- 2. bg-sub needs no slot -------------------------------------------------


def test_bgsub_runner_holds_no_gpu_models(tmp_path):
    from hydra_suite.core.inference.config import BgSubConfig, InferenceConfig
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = InferenceConfig(obb=None, bgsub=BgSubConfig.from_params({}))
    with (
        patch(
            "hydra_suite.core.inference.runner.RuntimeContext.from_config",
            return_value=MagicMock(device="cuda:0"),
        ),
        patch(
            "hydra_suite.core.inference.stages.bgsub.load_bgsub_model",
            return_value=MagicMock(name="bgsub"),
        ),
    ):
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
    # A gpu-tier bg-sub runner (device cuda:0) still has no network to guard.
    assert runner._models.bgsub is not None
    assert not runner.holds_gpu_models()


# --- worker helper: stop while waiting ----------------------------------------


@posix_only
def test_worker_acquire_returns_false_when_stopped_while_waiting(lock_env):
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    holder = gpl.GpuPhaseConfig(lock_env, "GPU-a", 1)
    got = gpl._try_acquire_any(holder)  # another "child" holds the only slot
    assert got is not None
    try:
        engine = TrackingEngineCore.__new__(TrackingEngineCore)
        engine._gpu_phase_lease = None
        engine._is_stop_requested = lambda: True
        assert engine._acquire_gpu_phase_or_stop("inference") is False
        assert engine._gpu_phase_lease is None
        assert gpl._held["depth"] == 0
    finally:
        gpl._unlock(got[1])
        os.close(got[1])


def test_worker_acquire_is_a_noop_without_env():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    engine = TrackingEngineCore.__new__(TrackingEngineCore)
    engine._gpu_phase_lease = None
    engine._is_stop_requested = lambda: False
    assert engine._acquire_gpu_phase_or_stop("inference") is True
    assert not engine._gpu_phase_lease.active


# --- 4. two threads never take two slots ---------------------------------------


@posix_only
def test_two_threads_share_one_slot_even_with_k2(tmp_path, monkeypatch):
    for key, value in gpl.phase_lock_env(tmp_path / "locks", "GPU-a", 2).items():
        monkeypatch.setenv(key, value)
    barrier = threading.Barrier(2)
    leases = []

    def _take():
        barrier.wait()
        leases.append(gpl.acquire_gpu_phase("t", poll_s=0.01))

    threads = [threading.Thread(target=_take) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(leases) == 2 and all(lease.acquired for lease in leases)
    assert gpl._held["depth"] == 2
    # Exactly one slot is held: the other is still free to another process.
    probe = gpl.GpuPhaseConfig(tmp_path / "locks", "GPU-a", 2)
    got = gpl._try_acquire_any(probe)
    assert got is not None
    gpl._unlock(got[1])
    os.close(got[1])
    for lease in leases:
        lease.release()
    assert gpl._held["depth"] == 0 and gpl._held["fd"] is None


@posix_only
def test_a_thread_waiting_joins_a_phase_its_process_takes(lock_env):
    outer = gpl.acquire_gpu_phase("outer")
    try:
        lease = gpl.acquire_gpu_phase("inner", poll_s=0.01)
        assert lease.acquired
        lease.release()
    finally:
        outer.release()
    assert gpl._held["depth"] == 0


# --- 5. no inherited semaphore env -----------------------------------------------


def test_child_env_never_inherits_a_stale_semaphore(tmp_path, monkeypatch):
    import subprocess

    from hydra_suite.runtime.cuda_devices import CudaDevice
    from hydra_suite.trackerkit.batch_fanout import FanoutOptions, run_batch_fanout
    from hydra_suite.trackerkit.batch_plan import BatchJobSpec

    for key, value in gpl.phase_lock_env("/stale", "GPU-old", 3).items():
        monkeypatch.setenv(key, value)
    real_popen = subprocess.Popen
    envs = []

    def _popen(command, **kwargs):
        envs.append(kwargs["env"])
        return real_popen([sys.executable, "-c", "print('ok')"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", _popen)
    video = tmp_path / "a.mp4"
    video.write_bytes(b"\x00")
    spec = BatchJobSpec(
        index=1, video_path=str(video), config_path=None, config={}, provenance="x"
    )
    result = run_batch_fanout(
        [spec],
        FanoutOptions(
            gpus=[CudaDevice(0, "GPU-aaaa", "x")],
            threads_per_job=1,
            run_dir=tmp_path / "run",
        ),
    )
    assert result.success
    assert not set(envs[0]) & set(gpl.ENV_VARS)


# --- 6. stop while waiting skips the session body ----------------------------


@posix_only
@pytest.mark.parametrize("held", [True, False])
@pytest.mark.parametrize("method", ["_run_interp_crops", "_run_dataset_generation"])
def test_session_skips_gpu_work_when_stopped_while_waiting(
    lock_env, monkeypatch, tmp_path, method, held
):
    from hydra_suite.core.tracking import session as session_mod

    holder = gpl.GpuPhaseConfig(lock_env, "GPU-a", 1)
    got = gpl._try_acquire_any(holder) if held else None
    ran = []
    monkeypatch.setattr(
        session_mod, "run_interpolated_crops", lambda *a, **k: ran.append(1) or {}
    )
    monkeypatch.setattr(
        session_mod.TrackingSessionCore,
        "_generate_active_learning_dataset",
        lambda self, *a: ran.append(1) or {},
    )
    monkeypatch.setattr(
        "hydra_suite.core.tracking.session_policy.should_run_interpolated_postpass",
        lambda config: True,
    )
    sess = session_mod.TrackingSessionCore.__new__(session_mod.TrackingSessionCore)
    sess.callbacks = MagicMock()
    # Stop is requested only while waiting for the slot (when it is held).
    # The early "already stopped?" checks in the session must pass, so this
    # only answers True to the semaphore's wait loop.
    sess.callbacks.should_stop = lambda: (
        held and sys._getframe(1).f_code.co_name == "acquire_gpu_phase"
    )
    sess.config = {"enable_dataset_generation": True}
    sess.params = {"DATASET_EXPORT_LEVELS": None, "DETECTION_METHOD": "yolo_obb"}
    sess.paths = {}
    sess.pose_state = MagicMock()
    sess.video_path = str(tmp_path / "v.mp4")
    Path(sess.video_path).write_bytes(b"\x00")
    csv = tmp_path / "f.csv"
    csv.write_text("a\n")
    try:
        result = getattr(sess, method)(str(csv))
    finally:
        if got is not None:
            gpl._unlock(got[1])
            os.close(got[1])
    if held:
        assert ran == [] and result is None
    else:  # control: the same call does the work when the slot is free
        assert ran == [1]


# --- 7. the wait log counts other waiters --------------------------------------


@posix_only
def test_wait_log_reports_other_waiters(lock_env, caplog):
    import subprocess

    import hydra_suite

    src = str(Path(hydra_suite.__file__).resolve().parents[1])
    holder = gpl.GpuPhaseConfig(lock_env, "GPU-a", 1)
    got = gpl._try_acquire_any(holder)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from hydra_suite.runtime.gpu_phase_lock import acquire_gpu_phase;"
        "acquire_gpu_phase('other', poll_s=0.02).release()"
    )
    other = subprocess.Popen([sys.executable, "-c", code, src], env=dict(os.environ))
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if list(lock_env.glob("GPU-a.waiter.*")):
                break
            time.sleep(0.02)
        stop_after = []

        def _stop():
            stop_after.append(1)
            return len(stop_after) > 2

        with caplog.at_level(logging.INFO, logger=gpl.__name__):
            lease = gpl.acquire_gpu_phase("me", should_stop=_stop, poll_s=0.01)
        assert not lease.acquired
        assert any(
            "1 other child(ren) also waiting" in r.getMessage() for r in caplog.records
        )
    finally:
        gpl._unlock(got[1])
        os.close(got[1])
        other.wait(timeout=10)
    # Every waiter cleaned its mark up.
    assert list(lock_env.glob("GPU-a.waiter.*")) == []
