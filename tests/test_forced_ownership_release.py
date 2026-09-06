"""Defect B: an explicit, audited way out of retained (owned) containment.

``WorkloadStillOwnedError`` deliberately keeps the sidecar and its leases owned
when teardown cannot prove the child exited. That is correct, but it left no
forced-release mechanism on any path, so a stuck run blocked every later run.
The way out must stay explicit, audited, and refused while the child is alive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hydra_suite.runtime.process_supervisor import (
    ContainmentPlan,
    ForcedReleaseRefusedError,
    SupervisedSidecar,
    WorkloadStillOwnedError,
)
from hydra_suite.runtime.resource_lease import (
    HeavyJobLease,
    ResourceBusyError,
    canonical_heavy_job_lease_set,
)
from hydra_suite.runtime.resource_limits import (
    LimitBackend,
    ProcessMemoryLimits,
    build_limited_launch,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process-group tests")


@pytest.fixture(autouse=True)
def _isolate_canonical_lease_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))


def _require_process_table_scan() -> None:
    try:
        next(iter(psutil.process_iter(["pid"])), None)
    except (psutil.Error, OSError):
        pytest.skip("sandbox denies process-table enumeration")


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _owned_sidecar(tmp_path, monkeypatch):
    """Return a sidecar stuck in the retained-ownership state, plus its leases."""

    _require_process_table_scan()
    launch = build_limited_launch(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        ProcessMemoryLimits(soft_host_bytes=1024**3, hard_host_bytes=2 * 1024**3),
        backend=LimitBackend.WATCHDOG_ONLY,
        environment=_child_env(),
    )
    leases = canonical_heavy_job_lease_set(
        "supervised", "cpu", lease_dir=tmp_path / "runtime" / "heavy-job-leases"
    )
    plan = ContainmentPlan(
        launch=launch,
        job_name="supervised",
        minimum_system_available_bytes=0,
        terminate_grace_seconds=0,
    )
    sidecar = SupervisedSidecar(plan)
    real_teardown = sidecar._terminate_and_reap
    monkeypatch.setattr(sidecar, "_terminate_and_reap", lambda _grace: False)
    with pytest.raises(WorkloadStillOwnedError):
        sidecar.wait(timeout=0)
    lease_dir = leases.leases[0].path.parent
    # The lease is genuinely still held by the owning sidecar.
    with pytest.raises(ResourceBusyError):
        HeavyJobLease(leases.resource_keys[0], "competitor", lease_dir).acquire()
    return sidecar, leases.resource_keys[0], lease_dir, real_teardown


def _reap(sidecar, real_teardown, monkeypatch):
    monkeypatch.setattr(sidecar, "_terminate_and_reap", real_teardown)
    if sidecar.tree is not None:
        sidecar.tree.kill()
    if sidecar.process is not None:
        try:
            sidecar.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pytest.skip("child could not be reaped in this sandbox")
    deadline = time.monotonic() + 5
    while sidecar.tree is not None and sidecar.tree.is_alive():
        if time.monotonic() > deadline:  # pragma: no cover - defensive
            pytest.skip("owned tree stayed alive in this sandbox")
        time.sleep(0.02)


def test_forced_release_is_refused_while_the_child_is_alive(tmp_path, monkeypatch):
    sidecar, key, lease_dir, real_teardown = _owned_sidecar(tmp_path, monkeypatch)
    try:
        with pytest.raises(ForcedReleaseRefusedError) as refusal:
            sidecar.force_release_ownership(
                operator="tester", reason="I want my lease back"
            )
        assert sidecar.process.pid in refusal.value.live_pids
        # Refused means refused: the lease is still owned.
        with pytest.raises(ResourceBusyError):
            HeavyJobLease(key, "competitor", lease_dir).acquire()
        assert not (lease_dir / "forced-releases.jsonl").exists()
    finally:
        _reap(sidecar, real_teardown, monkeypatch)
        sidecar._release_leases()


def test_forced_release_requires_an_operator_and_a_reason(tmp_path, monkeypatch):
    sidecar, _key, _lease_dir, real_teardown = _owned_sidecar(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="operator"):
            sidecar.force_release_ownership(operator="  ", reason="anything")
        with pytest.raises(ValueError, match="reason"):
            sidecar.force_release_ownership(operator="tester", reason="")
    finally:
        _reap(sidecar, real_teardown, monkeypatch)
        sidecar._release_leases()


def test_forced_release_frees_the_lease_and_records_who_forced_it(
    tmp_path, monkeypatch
):
    sidecar, key, lease_dir, real_teardown = _owned_sidecar(tmp_path, monkeypatch)
    _reap(sidecar, real_teardown, monkeypatch)

    record = sidecar.force_release_ownership(
        operator="rishika", reason="orphan confirmed gone by hand"
    )
    assert record.operator == "rishika"
    assert record.reason == "orphan confirmed gone by hand"
    assert key in record.resource_keys

    # The whole point: a later run can now take the lease.
    with HeavyJobLease(key, "after forced release", lease_dir):
        pass

    audit = lease_dir / "forced-releases.jsonl"
    entries = [json.loads(line) for line in audit.read_text().splitlines() if line]
    assert entries[-1]["operator"] == "rishika"
    assert entries[-1]["reason"] == "orphan confirmed gone by hand"
    assert entries[-1]["releasing_pid"] == os.getpid()
    assert entries[-1]["job_name"] == "supervised"


class _StubSidecar:
    def __init__(self, *, refuse: bool) -> None:
        self.refuse = refuse
        self.calls: list[dict] = []

    def cancel(self, *_args, **_kwargs):
        raise WorkloadStillOwnedError("still owned", self)

    def force_release_ownership(self, *, operator, reason):
        self.calls.append({"operator": operator, "reason": reason})
        if self.refuse:
            raise ForcedReleaseRefusedError("still alive", (4321,))
        from hydra_suite.runtime.process_supervisor import ForcedReleaseRecord

        return ForcedReleaseRecord(
            released_at=0.0,
            operator=operator,
            reason=reason,
            job_name="job",
            resource_keys=("k",),
            hostname="host",
            releasing_pid=1,
            ownership_was_uncertain=True,
            audit_path="/tmp/audit.jsonl",
        )


def _run_cli_with_owned_error(monkeypatch, argv, *, refuse):
    import hydra_suite.detectkit.cli as cli

    stub = _StubSidecar(refuse=refuse)
    owned = WorkloadStillOwnedError("owned", stub)
    owned.run_id = "run-1"
    cleaned: list[bool] = []
    owned.recovery_cleanup = lambda: cleaned.append(True)
    finalized: list[dict] = []
    monkeypatch.setattr(cli, "run", lambda _args: (_ for _ in ()).throw(owned))
    monkeypatch.setattr(
        cli,
        "finalize_run_record",
        lambda run_id, **kwargs: finalized.append({"run_id": run_id, **kwargs}),
    )
    return cli.main(argv), stub, finalized, cleaned


def test_cli_without_the_flag_still_preserves_the_owning_exception(monkeypatch):
    with pytest.raises(WorkloadStillOwnedError):
        _run_cli_with_owned_error(monkeypatch, ["--config", "plan.json"], refuse=False)


def test_cli_force_release_flag_releases_and_finalizes(monkeypatch):
    code, stub, finalized, cleaned = _run_cli_with_owned_error(
        monkeypatch,
        [
            "--config",
            "plan.json",
            "--force-release-ownership",
            "orphan gone",
            "--force-release-operator",
            "rishika",
        ],
        refuse=False,
    )
    assert code == 1
    assert stub.calls == [{"operator": "rishika", "reason": "orphan gone"}]
    assert finalized[0]["run_id"] == "run-1"
    assert finalized[0]["failure_details"]["containment"]["ownership"] == (
        "force-released"
    )
    # The partial publish artifact must go, or the retry hits FileExistsError.
    assert cleaned == [True]


def test_cli_force_release_refusal_keeps_ownership(monkeypatch):
    with pytest.raises(WorkloadStillOwnedError):
        _run_cli_with_owned_error(
            monkeypatch,
            [
                "--config",
                "plan.json",
                "--force-release-ownership",
                "orphan gone",
                "--force-release-operator",
                "rishika",
            ],
            refuse=True,
        )
