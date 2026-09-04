"""Protected SAM3 publish launcher and parent registry transaction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hydra_suite.runtime.process_supervisor import (
    ClassifiedExit,
    ExitKind,
    SupervisedResult,
    WorkloadStillOwnedError,
)
from hydra_suite.runtime.resource_budget import ResourceObservation
from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora import publish as pub
from hydra_suite.training.sam3_lora import publish_cli


class _FinishedProcess:
    def poll(self):
        return 0


class _WaitingProcess:
    def poll(self):
        return None


class _EmptyOutput:
    def drain(self, _timeout):
        return [], True, None


def _result(kind: ExitKind, *, returncode: int = 0):
    return SupervisedResult(
        returncode=returncode,
        classified_exit=ClassifiedExit(kind, f"classified {kind.value}"),
        output_tail=("bounded tail\n",),
        dropped_output_lines=0,
        watchdog=None,
        cgroup=None,
        output_error=None,
        peak_tree_rss_bytes=1234,
        minimum_system_available_bytes=5678,
    )


class _SuccessfulSidecar:
    seen_plan = None

    def __init__(self, plan, *, prelaunch_check, **_kwargs):
        type(self).seen_plan = plan
        prelaunch_check()
        self.plan = plan
        self.process = _FinishedProcess()
        self.output = _EmptyOutput()

    def wait(self, *, post_exit_check):
        command = list(self.plan.launch.command)
        request_path = Path(command[command.index("--request") + 1])
        result_path = Path(command[command.index("--result") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        artifact = (
            Path(request["models_root"]) / "sam3_finetuned" / f"{request['run_id']}.pt"
        )
        sidecar = artifact.with_name(artifact.name + ".sam3_meta.json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        # The registry transaction must not happen until after this validated
        # child receipt exists.
        assert not (Path(request["models_root"]) / "model_registry.json").exists()
        artifact.write_bytes(b"validated-checkpoint")
        sidecar.write_text(
            json.dumps(
                {
                    "imgsz": 1008,
                    "publish_attempt_id": request["publish_attempt_id"],
                }
            ),
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps({"artifact_path": str(artifact), "sidecar_path": str(sidecar)}),
            encoding="utf-8",
        )
        result = _result(ExitKind.SUCCESS)
        post_exit_check(result)
        return result

    def cancel(self, _grace):
        raise AssertionError("successful sidecar must not be canceled")


@pytest.fixture(autouse=True)
def _ample_resources(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(
        pub,
        "probe_resources",
        lambda: ResourceObservation(
            total_host_bytes=64 * gib, available_host_bytes=48 * gib
        ),
    )


def _inputs(tmp_path: Path):
    base = tmp_path / "base.pt"
    adapters = tmp_path / "adapters.pt"
    base.write_bytes(b"base")
    adapters.write_bytes(b"adapters")
    return base, adapters


def _publish(tmp_path: Path, **kwargs):
    base, adapters = _inputs(tmp_path)
    return pub.publish_sam3_model(
        run_id="run-1",
        adapters_path=adapters,
        base_checkpoint=base,
        build_manifest={"tile_px": 1007},
        params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
        source_fingerprint="fp1",
        models_root=tmp_path / "models",
        **kwargs,
    )


def test_importing_parent_publish_module_does_not_import_torch():
    code = (
        "import sys; "
        "import hydra_suite.training.sam3_lora.publish; "
        "assert 'torch' not in sys.modules"
    )
    environment = {**os.environ, "PYTHONPATH": "src"}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_publish_child_refuses_oversized_request_before_heavy_import(tmp_path):
    request = tmp_path / "request.json"
    request.write_bytes(b"{" + b" " * publish_cli.MAX_REQUEST_BYTES + b"}")

    with pytest.raises(RuntimeError, match="safe size bound"):
        publish_cli._read_request(request)

    assert "torch" not in publish_cli.__dict__


def test_publish_uses_host_only_containment_then_registers(monkeypatch, tmp_path):
    monkeypatch.setattr(pub, "SupervisedSidecar", _SuccessfulSidecar)

    key, artifact = _publish(tmp_path)

    assert key == "sam3_finetuned/run-1.pt"
    assert Path(artifact).read_bytes() == b"validated-checkpoint"
    registry = json.loads(
        (tmp_path / "models" / "model_registry.json").read_text(encoding="utf-8")
    )
    assert registry["entries"][key]["usage_role"] == "semantic_sam3"
    plan = _SuccessfulSidecar.seen_plan
    assert plan.launch.accelerator_kind.value == "cpu"
    assert len(plan.expected_resource_keys) == 1
    assert plan.expected_resource_keys[0].endswith(":host-memory")
    assert plan.launch.limits.hard_host_bytes > plan.launch.limits.soft_host_bytes


def test_sidecar_hard_limit_classification_is_preserved(monkeypatch, tmp_path):
    class HardLimitSidecar(_SuccessfulSidecar):
        def wait(self, *, post_exit_check):
            return _result(ExitKind.HOST_HARD_LIMIT, returncode=-9)

    monkeypatch.setattr(pub, "SupervisedSidecar", HardLimitSidecar)

    with pytest.raises(pub.Sam3PublishError) as raised:
        _publish(tmp_path)

    assert raised.value.failure_kind == "host-hard-limit"
    assert not (tmp_path / "models" / "model_registry.json").exists()
    assert not (tmp_path / "models" / "sam3_finetuned" / "run-1.pt").exists()


def test_registry_replace_failure_preserves_registry_and_removes_artifact(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pub, "SupervisedSidecar", _SuccessfulSidecar)
    registry_path = tmp_path / "models" / "model_registry.json"
    registry_path.parent.mkdir(parents=True)
    original = b'{"schema_version": 2, "entries": {"old": {}}}'
    registry_path.write_bytes(original)
    monkeypatch.setattr(
        pub,
        "_atomic_replace_registry",
        lambda *_args: (_ for _ in ()).throw(OSError("registry replace")),
    )

    # The fake's ordering assertion applies to a new registry; retain the same
    # child behavior here without that assertion.
    class ExistingRegistrySidecar(_SuccessfulSidecar):
        def wait(self, *, post_exit_check):
            command = list(self.plan.launch.command)
            request_path = Path(command[command.index("--request") + 1])
            result_path = Path(command[command.index("--result") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            artifact = (
                Path(request["models_root"])
                / "sam3_finetuned"
                / f"{request['run_id']}.pt"
            )
            sidecar = artifact.with_name(artifact.name + ".sam3_meta.json")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"validated-checkpoint")
            sidecar.write_text(
                json.dumps(
                    {
                        "imgsz": 1008,
                        "publish_attempt_id": request["publish_attempt_id"],
                    }
                ),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {"artifact_path": str(artifact), "sidecar_path": str(sidecar)}
                ),
                encoding="utf-8",
            )
            result = _result(ExitKind.SUCCESS)
            post_exit_check(result)
            return result

    monkeypatch.setattr(pub, "SupervisedSidecar", ExistingRegistrySidecar)
    with pytest.raises(OSError, match="registry replace"):
        _publish(tmp_path)

    assert registry_path.read_bytes() == original
    assert not (tmp_path / "models" / "sam3_finetuned" / "run-1.pt").exists()
    assert not list(registry_path.parent.glob(".model_registry.json.*.tmp"))


def test_cancellation_cleans_outputs_without_registration(monkeypatch, tmp_path):
    class CancelSidecar:
        def __init__(self, plan, *, prelaunch_check, **_kwargs):
            prelaunch_check()
            self.process = _WaitingProcess()
            self.output = _EmptyOutput()

        def cancel(self, _grace):
            return None

    monkeypatch.setattr(pub, "SupervisedSidecar", CancelSidecar)

    with pytest.raises(pub.Sam3PublishError) as raised:
        _publish(tmp_path, should_cancel=lambda: True)

    assert raised.value.canceled is True
    assert not (tmp_path / "models" / "model_registry.json").exists()


def test_uncertain_constructor_retains_recovery_cleanup(monkeypatch, tmp_path):
    class UncertainSidecar:
        attempt_id = ""

        def __init__(self, plan, **_kwargs):
            command = list(plan.launch.command)
            request_path = Path(command[command.index("--request") + 1])
            type(self).attempt_id = json.loads(
                request_path.read_text(encoding="utf-8")
            )["publish_attempt_id"]
            raise WorkloadStillOwnedError("ownership retained", self)

    monkeypatch.setattr(pub, "SupervisedSidecar", UncertainSidecar)

    with pytest.raises(WorkloadStillOwnedError) as raised:
        _publish(tmp_path)

    artifact = tmp_path / "models" / "sam3_finetuned" / "run-1.pt"
    sidecar = artifact.with_name(artifact.name + ".sam3_meta.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"late child output")
    sidecar.write_text(
        json.dumps({"publish_attempt_id": UncertainSidecar.attempt_id}),
        encoding="utf-8",
    )
    assert raised.value.recovery_cleanup is not None
    raised.value.recovery_cleanup()
    assert not artifact.exists()
    assert not sidecar.exists()


def test_failed_child_does_not_delete_raced_final_pair(monkeypatch, tmp_path):
    class RacedFinalSidecar(_SuccessfulSidecar):
        def wait(self, *, post_exit_check):
            command = list(self.plan.launch.command)
            request_path = Path(command[command.index("--request") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            artifact = (
                Path(request["models_root"])
                / "sam3_finetuned"
                / f"{request['run_id']}.pt"
            )
            sidecar = artifact.with_name(artifact.name + ".sam3_meta.json")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"concurrent-checkpoint")
            sidecar.write_text(
                json.dumps({"publish_attempt_id": "0" * 32}), encoding="utf-8"
            )
            return _result(ExitKind.ORDINARY_FAILURE, returncode=1)

    monkeypatch.setattr(pub, "SupervisedSidecar", RacedFinalSidecar)

    with pytest.raises(pub.Sam3PublishError):
        _publish(tmp_path)

    artifact = tmp_path / "models" / "sam3_finetuned" / "run-1.pt"
    sidecar = artifact.with_name(artifact.name + ".sam3_meta.json")
    assert artifact.read_bytes() == b"concurrent-checkpoint"
    assert (
        json.loads(sidecar.read_text(encoding="utf-8"))["publish_attempt_id"]
        == "0" * 32
    )


def test_live_profile_growth_refuses_before_child_work(monkeypatch, tmp_path):
    real_assess = pub._assess_publish
    initial = None

    def drifting_assessment(**kwargs):
        nonlocal initial
        decision = real_assess(**kwargs)
        if initial is None:
            initial = decision
            return decision
        return pub._PublishDecision(
            **{
                **decision.__dict__,
                "hard_host_bytes": initial.hard_host_bytes + 1,
            }
        )

    monkeypatch.setattr(pub, "_assess_publish", drifting_assessment)
    monkeypatch.setattr(pub, "SupervisedSidecar", _SuccessfulSidecar)

    with pytest.raises(pub.Sam3PublishError) as raised:
        _publish(tmp_path)

    assert raised.value.failure_kind == "host-admission-refusal"
    assert not (tmp_path / "models" / "model_registry.json").exists()
