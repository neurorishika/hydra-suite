"""SAM3 launcher integration with the bounded resource supervisor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hydra_suite.runtime.process_supervisor import (
    ClassifiedExit,
    ExitKind,
    SupervisedResult,
)
from hydra_suite.runtime.resource_lease import ResourceBusyError
from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.sam3_lora import train as tr


def _spec(tmp_path, **overrides):
    params = Sam3LoraParams(prompt="ant", label_quality_acknowledged=True, **overrides)
    return TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=params,
    )


class _Decision:
    def __init__(self, *, admitted=True, refusals=(), uuid="GPU-physical-0"):
        self.admitted = admitted
        self.refusals = tuple(refusals)
        self.warnings = ()
        self.cuda_device = (
            SimpleNamespace(uuid=uuid, total_bytes=48 << 30, free_bytes=47 << 30)
            if uuid
            else None
        )
        self.dataset = SimpleNamespace(marker="metadata-only")
        self.budget = SimpleNamespace(
            host_peak_bytes=10 << 30,
            reserved_host_bytes=8 << 30,
        )

    def to_dict(self):
        return {
            "admitted": self.admitted,
            "refusals": list(self.refusals),
            "cuda_device": (
                {"uuid": self.cuda_device.uuid} if self.cuda_device else None
            ),
        }


class _Output:
    def __init__(self, lines=()):
        self.lines = list(lines)

    def drain(self, timeout=0):
        lines, self.lines = self.lines, []
        return lines, True, None


def _supervised(kind=ExitKind.SUCCESS, *, returncode=0, tail=()):
    return SupervisedResult(
        returncode=returncode,
        classified_exit=ClassifiedExit(kind, f"classified {kind.value}"),
        output_tail=tuple(tail),
        dropped_output_lines=0,
        watchdog=None,
        cgroup=None,
        output_error=None,
        peak_accelerator_bytes=2 << 30,
        accelerator_observation_error=None,
    )


def _install(
    monkeypatch,
    tmp_path,
    *,
    result=None,
    lines=(),
    write_artifact=True,
    decisions=None,
):
    decisions = list(decisions or [_Decision(), _Decision()])

    def assess(_spec, **_kwargs):
        return decisions.pop(0) if len(decisions) > 1 else decisions[0]

    monkeypatch.setattr(tr.preflight_module, "assess_preflight", assess)
    monkeypatch.setattr(
        tr.preflight_module,
        "_probe_cuda_device",
        lambda _device: SimpleNamespace(
            uuid="GPU-physical-0", total_bytes=48 << 30, free_bytes=46 << 30
        ),
    )
    calls = []
    supervised = result or _supervised()

    class FakeSidecar:
        def __init__(self, plan, *, prelaunch_check, **kwargs):
            self.plan = plan
            self.output = _Output(lines)
            self.process = SimpleNamespace(poll=lambda: supervised.returncode)
            self.canceled = False
            calls.append(self)
            prelaunch_check()
            if write_artifact:
                artifact = tmp_path / "run" / "adapters.pt"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"adapter")

        def wait(self, *, post_exit_check=None, **_kwargs):
            if post_exit_check is not None:
                post_exit_check(supervised)
            return supervised

        def cancel(self, _grace):
            self.canceled = True

    monkeypatch.setattr(tr, "SupervisedSidecar", FakeSidecar)
    return calls


def test_preflight_refuses_before_sidecar_or_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tr.preflight_module,
        "assess_preflight",
        lambda _spec: _Decision(admitted=False, refusals=("host memory unsafe",)),
    )
    monkeypatch.setattr(
        tr,
        "SupervisedSidecar",
        lambda *args, **kwargs: pytest.fail("must not launch"),
    )

    result = tr.train_sam3_lora(_spec(tmp_path), str(tmp_path / "run"))

    assert not result["success"]
    assert result["failure_kind"] == "host-admission-refusal"
    assert not (tmp_path / "run").exists()


def test_final_live_probe_runs_inside_constructor_before_launch(tmp_path, monkeypatch):
    calls = _install(
        monkeypatch,
        tmp_path,
        decisions=[
            _Decision(),
            _Decision(admitted=False, refusals=("VRAM changed",)),
        ],
    )

    result = tr.train_sam3_lora(_spec(tmp_path), str(tmp_path / "run"))

    assert not result["success"]
    assert "VRAM changed" in result["error_message"]
    assert len(calls) == 1
    assert not (tmp_path / "run" / "adapters.pt").exists()


def test_progress_plain_logs_and_bounded_diagnostics_are_propagated(
    tmp_path, monkeypatch
):
    lines = [
        "plain startup\n",
        '@@HYDRA_SAM3_PROGRESS@@{"type":"log","message":"loss 1.0"}\n',
        '@@HYDRA_SAM3_PROGRESS@@{"type":"progress","epoch":1,"total":2}\n',
    ]
    _install(monkeypatch, tmp_path, lines=lines)
    logs, progress = [], []

    result = tr.train_sam3_lora(
        _spec(tmp_path),
        str(tmp_path / "run"),
        log_cb=logs.append,
        progress_cb=lambda epoch, total: progress.append((epoch, total)),
    )

    assert result["success"]
    assert logs == ["plain startup", "loss 1.0"]
    assert progress == [(1, 2)]
    assert result["containment"]["peak_observed_device_used_bytes"] == 2 << 30
    assert "not kernel-capped" in result["containment"]["cuda_vram_enforcement"]


def test_silent_nonzero_and_cuda_oom_exits_are_structured(tmp_path, monkeypatch):
    _install(
        monkeypatch,
        tmp_path,
        result=_supervised(
            ExitKind.ACCELERATOR_OOM,
            returncode=1,
            tail=("torch.cuda.OutOfMemoryError: CUDA out of memory\n",),
        ),
        lines=(),
        write_artifact=False,
    )

    result = tr.train_sam3_lora(_spec(tmp_path), str(tmp_path / "run"))

    assert not result["success"]
    assert result["exit_code"] == 1
    assert result["failure_kind"] == "accelerator-oom"
    assert "CUDA out of memory" in result["error_message"]


def test_exit_zero_without_nonempty_artifact_is_failure(tmp_path, monkeypatch):
    _install(monkeypatch, tmp_path, write_artifact=False)

    result = tr.train_sam3_lora(_spec(tmp_path), str(tmp_path / "run"))

    assert not result["success"]
    assert "did not write" in result["error_message"]


def test_cancel_is_step_and_output_independent(tmp_path, monkeypatch):
    calls = _install(monkeypatch, tmp_path, lines=(), write_artifact=True)

    result = tr.train_sam3_lora(
        _spec(tmp_path), str(tmp_path / "run"), should_cancel=lambda: True
    )

    assert result["canceled"]
    assert result["failure_kind"] == "canceled"
    assert calls[0].canceled
    assert not (tmp_path / "run" / "adapters.pt").exists()


def test_conflicting_canonical_lease_is_reported(tmp_path, monkeypatch):
    decisions = [_Decision(), _Decision()]
    monkeypatch.setattr(
        tr.preflight_module,
        "assess_preflight",
        lambda _spec, **_kwargs: decisions.pop(0),
    )

    class BusySidecar:
        def __init__(self, *_args, **_kwargs):
            raise ResourceBusyError("real-host:host-memory", None)

    monkeypatch.setattr(tr, "SupervisedSidecar", BusySidecar)

    result = tr.train_sam3_lora(_spec(tmp_path), str(tmp_path / "run"))

    assert not result["success"]
    assert "already leased" in result["error_message"]


def test_plan_uses_physical_cuda_and_one_immutable_limit_source(tmp_path, monkeypatch):
    calls = _install(monkeypatch, tmp_path)

    result = tr.train_sam3_lora(_spec(tmp_path), str(tmp_path / "run"))

    assert result["success"]
    plan = calls[0].plan
    assert plan.launch.accelerator_device_uuid == "GPU-physical-0"
    assert plan.launch.limits is plan.launch.limits
    assert (
        plan.watchdog_policy.soft_tree_rss_bytes == plan.launch.limits.soft_host_bytes
    )
    assert len(plan.expected_resource_keys) == 2


def test_callback_failure_cancels_and_reaps_sidecar(tmp_path, monkeypatch):
    calls = _install(monkeypatch, tmp_path, lines=("plain\n",))

    with pytest.raises(ValueError, match="callback failed"):
        tr.train_sam3_lora(
            _spec(tmp_path),
            str(tmp_path / "run"),
            log_cb=lambda _line: (_ for _ in ()).throw(ValueError("callback failed")),
        )

    assert calls[0].canceled
    assert not (tmp_path / "run" / "adapters.pt").exists()
