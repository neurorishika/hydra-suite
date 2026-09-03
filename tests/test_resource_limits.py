from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hydra_suite.runtime.process_supervisor import ExitEvidence, ExitKind, classify_exit
from hydra_suite.runtime.resource_limits import (
    LimitBackend,
    ProcessMemoryLimits,
    build_limited_launch,
    select_limit_backend,
)


def _test_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_systemd_command_places_kernel_limits_outside_bootstrap():
    launch = build_limited_launch(
        ["python", "work.py"],
        ProcessMemoryLimits(soft_host_bytes=100, hard_host_bytes=200),
        backend=LimitBackend.SYSTEMD_CGROUP,
        environment={},
        python_executable="/usr/bin/python3",
        systemd_unit="hydra-test.scope",
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
    assert "--address-space-bytes" not in launch.command
    assert launch.systemd_unit == "hydra-test.scope"


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
