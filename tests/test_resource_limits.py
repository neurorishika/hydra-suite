from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hydra_suite.runtime.resource_limits as limits_module
from hydra_suite.runtime.child_bootstrap import install_linux_parent_death_signal
from hydra_suite.runtime.process_supervisor import ExitEvidence, ExitKind, classify_exit
from hydra_suite.runtime.resource_limits import (
    CgroupEvidence,
    LimitBackend,
    ProcessMemoryLimits,
    build_limited_launch,
    cgroup_path_contains_unit,
    probe_systemd_cgroup_evidence,
    select_limit_backend,
    systemd_scope_is_quiescent,
    systemd_user_scope_available,
)


def _test_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_systemd_command_places_kernel_limits_outside_bootstrap():
    limits = ProcessMemoryLimits(soft_host_bytes=100, hard_host_bytes=200)
    with pytest.raises(ValueError, match="generated internally"):
        build_limited_launch(
            ["python", "work.py"],
            limits,
            backend=LimitBackend.SYSTEMD_CGROUP,
            systemd_unit="colliding.scope",
        )
    launch = build_limited_launch(
        ["python", "work.py"],
        limits,
        backend=LimitBackend.SYSTEMD_CGROUP,
        environment={},
        python_executable="/usr/bin/python3",
    )

    assert launch.command[:4] == (
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
    )
    assert "--wait" not in launch.command
    assert "--property=MemoryHigh=100" in launch.command
    assert "--property=MemoryMax=200" in launch.command
    assert "--property=MemorySwapMax=0" in launch.command
    assert "--property=TasksMax=512" in launch.command
    assert "--address-space-bytes" not in launch.command
    assert launch.systemd_unit is not None
    assert launch.systemd_unit.startswith("hydra-job-")
    assert launch.systemd_unit.endswith(".scope")
    assert launch.command[launch.command.index("--unit") + 1] == launch.systemd_unit
    assert launch.limits is limits
    assert dict(launch.environment) == {}


def test_process_limit_must_be_positive():
    with pytest.raises(ValueError, match="process count"):
        ProcessMemoryLimits(soft_host_bytes=100, hard_host_bytes=200, max_processes=0)


def test_cuda_launch_requires_one_resolver_supplied_physical_identity():
    limits = ProcessMemoryLimits(soft_host_bytes=100, hard_host_bytes=200)
    with pytest.raises(ValueError, match="exactly one"):
        build_limited_launch(["python"], limits, accelerator_kind="cuda")
    launch = build_limited_launch(
        ["python"],
        limits,
        accelerator_kind="cuda",
        accelerator_device_uuid=" GPU-REAL ",
    )

    assert launch.accelerator_device_uuid == "GPU-REAL"
    assert launch.accelerator_pci_bus_id is None


def test_rlimit_launch_documents_virtual_memory_and_sets_mps_before_exec():
    launch = build_limited_launch(
        ["python", "work.py"],
        ProcessMemoryLimits(
            soft_host_bytes=100,
            hard_host_bytes=200,
            mps_high_watermark_ratio=0.7,
        ),
        backend=LimitBackend.RLIMIT_AS,
        environment={},
        python_executable="/usr/bin/python3",
    )

    assert "--address-space-bytes" in launch.command
    assert "--mps-high-watermark-ratio" in launch.command
    assert any("virtual address space" in item for item in launch.limitations)
    assert any("does not cap discrete GPU VRAM" in item for item in launch.limitations)


def test_bootstrap_sets_mps_guard_before_replacing_itself_with_workload():
    command = [
        sys.executable,
        "-c",
        "import os; print(os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'])",
    ]
    launch = build_limited_launch(
        command,
        ProcessMemoryLimits(
            soft_host_bytes=100,
            hard_host_bytes=200,
            mps_high_watermark_ratio=0.65,
        ),
        backend=LimitBackend.WATCHDOG_ONLY,
        accelerator_kind="mps",
        environment=_test_env(),
    )

    result = subprocess.run(
        launch.command,
        env=launch.environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert result.stdout.strip() == "0.65"


def test_mps_launch_requires_an_explicit_allocator_high_watermark():
    with pytest.raises(ValueError, match="MPS.*high-watermark"):
        build_limited_launch(
            ["python", "work.py"],
            ProcessMemoryLimits(soft_host_bytes=100, hard_host_bytes=200),
            backend=LimitBackend.WATCHDOG_ONLY,
            accelerator_kind="mps",
        )


def test_none_environment_inherits_but_empty_environment_stays_empty(monkeypatch):
    monkeypatch.setenv("HYDRA_ENV_SENTINEL", "present")
    limits = ProcessMemoryLimits(soft_host_bytes=100, hard_host_bytes=200)

    inherited = build_limited_launch(
        ["python", "work.py"], limits, backend=LimitBackend.WATCHDOG_ONLY
    )
    empty = build_limited_launch(
        ["python", "work.py"],
        limits,
        backend=LimitBackend.WATCHDOG_ONLY,
        environment={},
    )

    assert inherited.environment["HYDRA_ENV_SENTINEL"] == "present"
    assert dict(empty.environment) == {}
    with pytest.raises(TypeError):
        empty.environment["late-mutation"] = "forbidden"  # type: ignore[index]

    monkeypatch.setattr(Path, "exists", lambda _path: True)
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/systemd-run")
    assert not systemd_user_scope_available(system="Linux", environ={})


def test_systemd_evidence_timeout_and_disappearing_unit_are_explicit(monkeypatch):
    calls = []

    def disappeared(*args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(args[0], 1, "", "Unit vanished")

    monkeypatch.setattr(subprocess, "run", disappeared)
    evidence = probe_systemd_cgroup_evidence("gone.scope", timeout_seconds=0.25)
    assert isinstance(evidence, CgroupEvidence)
    assert not evidence.available
    assert "vanished" in (evidence.error or "").lower()
    assert calls[0]["timeout"] == 0.25

    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["systemctl"], 0.25)

    monkeypatch.setattr(subprocess, "run", timed_out)
    evidence = probe_systemd_cgroup_evidence("slow.scope", timeout_seconds=0.25)
    assert not evidence.available
    assert "timed out" in (evidence.error or "").lower()


def test_systemd_scope_signal_success_does_not_prove_live_cgroup_empty(
    tmp_path, monkeypatch
):
    completed = subprocess.CompletedProcess(
        [],
        0,
        stdout="ActiveState=inactive\nControlGroup=/user.slice/hydra-owned.scope\n",
        stderr="",
    )
    monkeypatch.setattr(limits_module.subprocess, "run", lambda *_a, **_k: completed)
    cgroup = tmp_path / "user.slice" / "hydra-owned.scope"
    cgroup.mkdir(parents=True)
    (cgroup / "cgroup.procs").write_text("123\n", encoding="utf-8")

    assert (
        systemd_scope_is_quiescent("hydra-owned.scope", cgroup_root=tmp_path) is False
    )

    (cgroup / "cgroup.procs").write_text("", encoding="utf-8")
    assert systemd_scope_is_quiescent("hydra-owned.scope", cgroup_root=tmp_path) is True

    missing_membership = subprocess.CompletedProcess(
        [], 0, stdout="ActiveState=inactive\n", stderr=""
    )
    monkeypatch.setattr(
        limits_module.subprocess, "run", lambda *_a, **_k: missing_membership
    )
    assert systemd_scope_is_quiescent("hydra-owned.scope", cgroup_root=tmp_path) is None


def test_unloaded_systemd_scope_is_quiescent_but_bus_error_is_unknown(monkeypatch):
    absent = subprocess.CompletedProcess(
        [], 1, stdout="", stderr="Unit hydra-gone.scope could not be found."
    )
    monkeypatch.setattr(limits_module.subprocess, "run", lambda *_a, **_k: absent)
    assert systemd_scope_is_quiescent("hydra-gone.scope") is True

    bus_error = subprocess.CompletedProcess(
        [], 1, stdout="", stderr="Failed to connect to bus: connection refused"
    )
    monkeypatch.setattr(limits_module.subprocess, "run", lambda *_a, **_k: bus_error)
    assert systemd_scope_is_quiescent("hydra-gone.scope") is None


def test_cgroup_unit_matching_uses_an_exact_path_component():
    cgroup = "0::/user.slice/hydra-owned.scope-extra/child\n"

    assert not cgroup_path_contains_unit(cgroup, "hydra-owned.scope")
    assert cgroup_path_contains_unit(cgroup, "hydra-owned.scope-extra")


def test_linux_parent_death_signal_is_installed_before_race_check():
    calls = []

    def fake_prctl(option, signal_number):
        calls.append((option, signal_number))
        return 0

    assert install_linux_parent_death_signal(
        123,
        system="Linux",
        prctl_call=fake_prctl,
        get_parent_pid=lambda: 123,
    )
    assert calls

    with pytest.raises(RuntimeError, match="parent exited"):
        install_linux_parent_death_signal(
            123,
            system="Linux",
            prctl_call=fake_prctl,
            get_parent_pid=lambda: 999,
        )


def test_runtime_package_import_does_not_load_accelerator_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, hydra_suite.runtime; "
            "print('hydra_suite.utils.gpu_utils' in sys.modules)",
        ],
        env=_test_env(),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert result.stdout.strip() == "False"


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS hard cap requires Linux")
def test_tiny_isolated_rlimit_rejects_child_allocation_without_harming_parent():
    # The subprocess is capped at 128 MiB and intentionally asks for 512 MiB.
    # Physical pages are not consumed: RLIMIT_AS rejects the allocation first.
    command = [
        sys.executable,
        "-S",
        "-c",
        "x = bytearray(512 * 1024 * 1024); print(len(x))",
    ]
    launch = build_limited_launch(
        command,
        ProcessMemoryLimits(
            soft_host_bytes=64 * 1024**2, hard_host_bytes=128 * 1024**2
        ),
        backend=LimitBackend.RLIMIT_AS,
        environment=_test_env(),
    )

    result = subprocess.run(
        launch.command,
        env=launch.environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "MemoryError" in result.stderr
    classified = classify_exit(
        ExitEvidence(
            returncode=result.returncode,
            output_tail=result.stderr,
            limit_backend=launch.backend,
        )
    )
    assert classified.kind is ExitKind.HOST_HARD_LIMIT
    # Reaching this assertion in the uncapped pytest parent is the safety
    # property; no dangerous machine-scale allocation was attempted.
    assert bytearray(1024) == bytes(1024)


def test_backend_selection_prefers_systemd_then_posix_fallback():
    assert (
        select_limit_backend(system="Linux", systemd_available=True)
        is LimitBackend.SYSTEMD_CGROUP
    )
    assert (
        select_limit_backend(system="Linux", systemd_available=False)
        is LimitBackend.RLIMIT_AS
    )
    assert (
        select_limit_backend(system="Windows", systemd_available=False)
        is LimitBackend.WATCHDOG_ONLY
    )
    assert (
        select_limit_backend(system="Darwin", systemd_available=False)
        is LimitBackend.WATCHDOG_ONLY
    )
