from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import hydra_suite.runtime.resource_lease as lease_module
from hydra_suite.runtime.resource_lease import (
    HeavyJobLease,
    ResourceBusyError,
    canonical_heavy_job_lease,
    canonical_resource_key,
    owner_is_live,
)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_conflicting_process_is_refused_but_another_resource_can_proceed(tmp_path):
    script = """
import sys, time
from pathlib import Path
from hydra_suite.runtime.resource_lease import HeavyJobLease
lease = HeavyJobLease('host:cuda:0', 'child', Path(sys.argv[1])).acquire()
print('acquired', flush=True)
time.sleep(30)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "acquired"
        with pytest.raises(ResourceBusyError) as error:
            HeavyJobLease("host:cuda:0", "parent", tmp_path).acquire()
        assert error.value.owner is not None
        assert owner_is_live(error.value.owner)

        with HeavyJobLease("host:cuda:1", "other", tmp_path) as lease:
            assert lease.owner is not None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_stale_metadata_is_replaced_only_after_lock_acquisition(tmp_path):
    lease = HeavyJobLease("host:mps:unified", "new job", tmp_path)
    lease.path.parent.mkdir(parents=True, exist_ok=True)
    lease.path.write_text(
        json.dumps(
            {
                "resource_key": "host:mps:unified",
                "job_name": "dead job",
                "lease_id": "old",
                "pid": 99999999,
                "process_start_time": 1.0,
                "hostname": "not-this-host",
                "acquired_at": 1.0,
            }
        ),
        encoding="utf-8",
    )

    with lease:
        assert lease.owner is not None
        assert lease.owner.job_name == "new job"
        assert owner_is_live(lease.owner)

    persisted = json.loads(lease.path.read_text(encoding="utf-8"))
    assert persisted["job_name"] == "new job"


def test_canonical_keys_use_physical_cuda_uuid_and_one_mps_pool():
    assert canonical_resource_key("cuda", index=0, device_uuid="GPU-ABC") == (
        canonical_resource_key("cuda", index=7, device_uuid="gpu-abc")
    )
    assert canonical_resource_key("mps", index=0) == canonical_resource_key(
        "mps", index=99, device_uuid="ignored"
    )
    assert canonical_resource_key("mps", index=0).endswith(":mps:unified")
    assert canonical_resource_key(
        "cuda",
        index=1,
        environ={"CUDA_VISIBLE_DEVICES": "GPU-ONE,GPU-TWO"},
    ).endswith(":cuda:uuid:gpu-two")


def test_canonical_lease_factory_uses_shared_hydra_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))
    lease = canonical_heavy_job_lease("training", "mps", index=3)

    assert lease.path.parent == tmp_path / "runtime" / "heavy-job-leases"
    assert lease.resource_key.endswith(":mps:unified")


def test_failed_owner_metadata_setup_unlocks_and_closes_handle(tmp_path, monkeypatch):
    unlocked_while_open = []
    monkeypatch.setattr(lease_module, "_try_lock", lambda _handle: None)
    monkeypatch.setattr(
        lease_module,
        "_unlock",
        lambda handle: unlocked_while_open.append(not handle.closed),
    )
    monkeypatch.setattr(
        lease_module.psutil,
        "Process",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("metadata probe failed")),
    )
    lease = HeavyJobLease("host:cpu:memory", "broken", tmp_path)

    with pytest.raises(RuntimeError, match="metadata probe failed"):
        lease.acquire()

    assert unlocked_while_open == [True]
    assert lease.owner is None
