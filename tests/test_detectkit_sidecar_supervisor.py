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
