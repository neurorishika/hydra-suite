import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hydra_suite.runtime.process_supervisor import ExitKind


@pytest.mark.parametrize(
    "kind",
    [
        ExitKind.HOST_SOFT_LIMIT,
        ExitKind.HOST_HARD_LIMIT,
        ExitKind.ACCELERATOR_OOM,
        ExitKind.CANCELED,
        ExitKind.ORDINARY_FAILURE,
    ],
)
def test_protected_operation_preserves_distinct_exit_classification(kind):
    from hydra_suite.detectkit.sidecars.supervisor import _failed_outcome

    result = _failed_outcome(
        SimpleNamespace(
            classified_exit=SimpleNamespace(kind=kind, message="classified"),
            peak_tree_rss_bytes=2 * 1024**3,
            peak_accelerator_bytes=3,
            dropped_output_lines=7,
        ),
        4 * 1024**3,
    )

    assert result.failure_kind == kind.value
    assert result.canceled is (kind is ExitKind.CANCELED)
    assert "4.0 GiB" in result.message
    assert "2.0 GiB" in result.message
    assert result.dropped_output_lines == 7


def test_progress_protocol_is_typed_and_bounded():
    from hydra_suite.detectkit.sidecars.supervisor import _parse_progress

    valid = json.dumps(
        {
            "detectkit_sidecar": 1,
            "type": "progress",
            "percent": 150,
            "message": "working",
        }
    )
    assert _parse_progress(valid) == (100, "working")
    assert _parse_progress(json.dumps({"detectkit_sidecar": 1})) is None
    assert _parse_progress("{" + "x" * 20_000) is None
    oversized = json.dumps(
        {
            "detectkit_sidecar": 1,
            "type": "progress",
            "percent": 2,
            "message": "x" * 4097,
        }
    )
    assert _parse_progress(oversized) is None


def test_model_operation_containment_uses_admitted_capacity_not_estimate():
    from hydra_suite.detectkit.sidecars.supervisor import _containment_limits
    from hydra_suite.runtime.resource_budget import AcceleratorKind

    budget = SimpleNamespace(usable_host_bytes=32 * 1024**3)
    observation = SimpleNamespace(total_host_bytes=64 * 1024**3)

    soft, hard, mps_ratio = _containment_limits(
        budget, observation, AcceleratorKind.MPS
    )

    assert hard == 32 * 1024**3
    assert soft == int(32 * 1024**3 * 0.9)
    assert mps_ratio == 0.5


def test_protected_operation_surfaces_a_valid_child_failure_report(
    monkeypatch, tmp_path
):
    from hydra_suite.detectkit.sidecars import supervisor
    from hydra_suite.detectkit.sidecars.protocol import (
        Operation,
        SidecarRequest,
        SidecarResult,
        SidecarStatus,
        read_request,
        write_result,
    )
    from hydra_suite.runtime.resource_budget import AcceleratorKind

    observation = SimpleNamespace(
        total_host_bytes=64 * supervisor.GiB,
        available_host_bytes=48 * supervisor.GiB,
    )
    budget = SimpleNamespace(
        admitted=True,
        refusals=(),
        usable_host_bytes=32 * supervisor.GiB,
        reserved_host_bytes=8 * supervisor.GiB,
        accelerator_peak_bytes=0,
    )
    supervised = SimpleNamespace(
        classified_exit=SimpleNamespace(
            kind=ExitKind.ORDINARY_FAILURE,
            message="Worker exited with code 1",
        ),
        peak_tree_rss_bytes=2 * supervisor.GiB,
        peak_accelerator_bytes=None,
        dropped_output_lines=0,
    )

    def fake_sidecar(plan, **_kwargs):
        request_path = Path(plan.launch.command[-3])
        result_path = Path(plan.launch.command[-1])
        request = read_request(request_path)
        write_result(
            result_path,
            SidecarResult(
                request.request_id,
                request.operation,
                SidecarStatus.FAILED,
                "MPS backend initialization failed",
            ),
        )
        return SimpleNamespace(process=None, wait=lambda: supervised)

    monkeypatch.setattr(
        supervisor,
        "_accelerator_for",
        lambda _device: (AcceleratorKind.CPU, None, None, None),
    )
    monkeypatch.setattr(supervisor, "probe_resources", lambda *_a, **_k: observation)
    monkeypatch.setattr(
        supervisor, "evaluate_resource_request", lambda *_a, **_k: budget
    )
    monkeypatch.setattr(supervisor, "resource_telemetry", lambda *_a, **_k: {})
    monkeypatch.setattr(supervisor, "SupervisedSidecar", fake_sidecar)

    outcome = supervisor.ProtectedOperation(
        SidecarRequest("request", Operation.DATASET_INFERENCE, {}),
        device="cpu",
        cleanup_paths=(tmp_path / "output.npz",),
    ).run()

    assert not outcome.success
    assert outcome.failure_kind == ExitKind.ORDINARY_FAILURE.value
    assert outcome.message == "MPS backend initialization failed"
    assert "Memory cap" not in outcome.message


def test_auto_device_binds_visible_cuda_on_linux(monkeypatch):
    from hydra_suite.detectkit.sidecars import supervisor
    from hydra_suite.runtime.resource_budget import AcceleratorKind
    from hydra_suite.training.sam3_lora import preflight

    observed = SimpleNamespace(
        uuid="GPU-exact", name="GPU", free_bytes=8, total_bytes=16
    )
    monkeypatch.setattr(supervisor.sys, "platform", "linux")
    monkeypatch.setattr(preflight, "_probe_cuda_device", lambda value: observed)

    kind, uuid, pci, device = supervisor._accelerator_for("auto")

    assert kind is AcceleratorKind.CUDA
    assert uuid == "GPU-exact"
    assert pci is None
    assert device is observed


def test_uncertain_sidecar_retains_control_and_outputs_until_recovery(
    monkeypatch, tmp_path
):
    from hydra_suite.detectkit.sidecars import supervisor
    from hydra_suite.detectkit.sidecars.protocol import Operation, SidecarRequest
    from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError
    from hydra_suite.runtime.resource_budget import AcceleratorKind

    output = tmp_path / "private-output.npz"
    output.write_bytes(b"partial")
    owner = SimpleNamespace(cancel=lambda *_args, **_kwargs: None)
    captured_control_dirs: list[Path] = []

    def fail_after_launch(plan, **_kwargs):
        request_path = Path(plan.launch.command[-3])
        captured_control_dirs.append(request_path.parent)
        assert request_path.is_file()
        raise WorkloadStillOwnedError("ownership uncertain", owner)

    observation = SimpleNamespace(
        total_host_bytes=64 * supervisor.GiB,
        available_host_bytes=48 * supervisor.GiB,
    )
    budget = SimpleNamespace(
        admitted=True,
        refusals=(),
        usable_host_bytes=32 * supervisor.GiB,
        reserved_host_bytes=8 * supervisor.GiB,
        accelerator_peak_bytes=0,
    )
    monkeypatch.setattr(
        supervisor,
        "_accelerator_for",
        lambda _device: (AcceleratorKind.CPU, None, None, None),
    )
    monkeypatch.setattr(supervisor, "probe_resources", lambda *_a, **_k: observation)
    monkeypatch.setattr(
        supervisor, "evaluate_resource_request", lambda *_a, **_k: budget
    )
    monkeypatch.setattr(supervisor, "SupervisedSidecar", fail_after_launch)

    operation = supervisor.ProtectedOperation(
        SidecarRequest("request", Operation.DATASET_INFERENCE, {}),
        device="cpu",
        cleanup_paths=(output,),
    )
    with pytest.raises(WorkloadStillOwnedError) as caught:
        operation.run()

    control_dir = captured_control_dirs[0]
    assert control_dir.is_dir()
    assert output.is_file()
    assert caught.value.sidecar is owner
    assert caught.value.recovery_cleanup is not None

    caught.value.recovery_cleanup()

    assert not control_dir.exists()
    assert not output.exists()


def _cuda_protected_run(monkeypatch, tmp_path, *, uuid_by_call, device="cuda:0"):
    """One ProtectedOperation on CUDA; returns (asked, outcome).

    The probe answers by DEVICE STRING, like `nvidia-smi`: a UUID argument
    finds nothing, because only the CHILD's CUDA_VISIBLE_DEVICES carries the
    pin.
    """

    from hydra_suite.detectkit.sidecars import supervisor
    from hydra_suite.detectkit.sidecars.protocol import (
        Operation,
        SidecarRequest,
        SidecarResult,
        SidecarStatus,
        read_request,
        write_result,
    )
    from hydra_suite.runtime.resource_budget import AcceleratorKind
    from hydra_suite.training.sam3_lora import preflight

    asked: list[str] = []

    def probe(dev):
        asked.append(dev)
        uuid = uuid_by_call(len(asked))
        if uuid is None:
            return None
        return SimpleNamespace(
            uuid=uuid,
            name="GPU",
            free_bytes=8 * supervisor.GiB,
            total_bytes=16 * supervisor.GiB,
        )

    monkeypatch.setattr(preflight, "_probe_cuda_device", probe)
    monkeypatch.setattr(
        supervisor,
        "_accelerator_for",
        lambda _device: (
            AcceleratorKind.CUDA,
            "GPU-pinned",
            None,
            SimpleNamespace(name="GPU"),
        ),
    )
    observation = SimpleNamespace(
        total_host_bytes=64 * supervisor.GiB,
        available_host_bytes=48 * supervisor.GiB,
    )
    budget = SimpleNamespace(
        admitted=True,
        refusals=(),
        usable_host_bytes=32 * supervisor.GiB,
        reserved_host_bytes=8 * supervisor.GiB,
        accelerator_peak_bytes=0,
    )
    supervised = SimpleNamespace(
        classified_exit=SimpleNamespace(kind=ExitKind.SUCCESS, message="ok"),
        peak_tree_rss_bytes=2 * supervisor.GiB,
        peak_accelerator_bytes=None,
        dropped_output_lines=0,
    )

    def fake_sidecar(plan, *, prelaunch_check=None, accelerator_probe=None, **_kwargs):
        if prelaunch_check is not None:
            prelaunch_check()
        if accelerator_probe is not None:
            accelerator_probe()
        request_path = Path(plan.launch.command[-3])
        result_path = Path(plan.launch.command[-1])
        request = read_request(request_path)
        write_result(
            result_path,
            SidecarResult(
                request.request_id, request.operation, SidecarStatus.SUCCESS, "ok"
            ),
        )
        return SimpleNamespace(process=None, wait=lambda: supervised)

    monkeypatch.setattr(supervisor, "probe_resources", lambda *_a, **_k: observation)
    monkeypatch.setattr(
        supervisor, "evaluate_resource_request", lambda *_a, **_k: budget
    )
    monkeypatch.setattr(supervisor, "resource_telemetry", lambda *_a, **_k: {})
    monkeypatch.setattr(supervisor, "SupervisedSidecar", fake_sidecar)

    outcome = supervisor.ProtectedOperation(
        SidecarRequest("request", Operation.DATASET_INFERENCE, {}),
        device=device,
        cleanup_paths=(tmp_path / "output.npz",),
    ).run()
    return asked, outcome


def test_the_parent_reprobes_by_device_string_not_by_the_pinned_uuid(
    monkeypatch, tmp_path
):
    """Same defect as the Ultralytics supervisor, third instance.

    `_probe_cuda_device` resolves a UUID only out of CUDA_VISIBLE_DEVICES, and
    the parent never sets it -- the pin goes on the CHILD. Asking for the UUID
    returned None and made both the prelaunch check and the accelerator probe
    raise unconditionally, so every DetectKit GPU job failed at launch.
    """

    asked, outcome = _cuda_protected_run(
        monkeypatch, tmp_path, uuid_by_call=lambda _n: "GPU-pinned"
    )

    assert outcome.success
    assert asked, "the parent must re-probe before launching"
    assert not any(str(item).startswith("GPU-") for item in asked)
    assert set(asked) == {"cuda:0"}


def test_a_device_swap_before_launch_is_still_detected(monkeypatch, tmp_path):
    """The `.uuid` equality is what detects a swap; probing by string keeps it."""

    asked, outcome = _cuda_protected_run(
        monkeypatch, tmp_path, uuid_by_call=lambda _n: "GPU-other"
    )

    assert not outcome.success
    assert "changed" in outcome.message
    # It must fail for the RIGHT reason -- a real swap, not a UUID re-probe
    # that can never resolve.
    assert not any(str(item).startswith("GPU-") for item in asked)


def test_missing_device_telemetry_before_launch_is_still_refused(monkeypatch, tmp_path):
    asked, outcome = _cuda_protected_run(
        monkeypatch, tmp_path, uuid_by_call=lambda _n: None
    )

    assert not outcome.success
    assert not any(str(item).startswith("GPU-") for item in asked)
