"""SAM3 LoRA launcher tests (train.py) -- the sidecar-subprocess boundary.

Training now runs in a dedicated `hydra-sam3` conda env, launched as a
subprocess via `popen_conda` (see
`docs/superpowers/specs/2026-09-01-sam3-training-sidecar-env-design.md`,
section 3). These tests fake `popen_conda` entirely -- no real subprocess,
no conda, no sam3, no GPU -- and assert the launcher's streaming, exit-code,
and artifact-verification discipline.

THE CRITICAL RULE under test throughout: a child that exits 0 without having
written `adapters.pt` must still report `success: False`. Earlier in this
branch an in-process version returned success after training zero batches,
which would have published a checkpoint identical to stock -- see the
original of this file, task-8 fix round 1, finding 2. That discipline now
has to hold across a process boundary instead of inside one function.
"""

from __future__ import annotations

from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.sam3_lora import preflight as pf
from hydra_suite.training.sam3_lora import train as tr


def _healthy_spec(tmp_path):
    p = Sam3LoraParams(prompt="ant", label_quality_acknowledged=True)
    return TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=p,
    )


def _pass_preflight(monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 48.0)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 100)
    monkeypatch.setattr(pf, "_free_disk_gb", lambda p: 100.0)


class _FakeProcess:
    """Stands in for `subprocess.Popen` as returned by `popen_conda`."""

    def __init__(self, lines, returncode=0, terminate_hangs=False):
        self._lines = list(lines)
        self.returncode = returncode
        self.stdout = iter(self._lines)
        self.terminated = False
        self.killed = False
        self._terminate_hangs = terminate_hangs
        self._waited = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self._terminate_hangs and self.terminated and not self.killed and timeout:
            import subprocess

            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        self._waited = True
        return self.returncode


def _fake_popen_conda(calls, process):
    def _fake(*args, **kwargs):
        calls.append((args, kwargs))
        return process

    return _fake


def test_preflight_refuses_before_any_subprocess(tmp_path, monkeypatch):
    """A spec missing label_quality_acknowledged must never launch a child."""
    calls = []
    monkeypatch.setattr(tr, "popen_conda", _fake_popen_conda(calls, None))

    p = Sam3LoraParams(prompt="ant", label_quality_acknowledged=False)
    spec = TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=p,
    )

    result = tr.train_sam3_lora(spec, str(tmp_path / "run"))

    assert result["success"] is False
    assert calls == []


def test_progress_and_log_records_forwarded(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    run_dir = tmp_path / "run"

    def _write_artifact_and_lines(*a, **k):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "adapters.pt").write_bytes(b"fake")
        lines = [
            "plain startup line\n",
            '@@HYDRA_SAM3_PROGRESS@@{"type": "log", "message": "epoch 0 step 10 loss 1.0"}\n',
            '@@HYDRA_SAM3_PROGRESS@@{"type": "progress", "epoch": 1, "total": 2}\n',
        ]
        return _FakeProcess(lines, returncode=0)

    calls = []

    def _fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return _write_artifact_and_lines()

    monkeypatch.setattr(tr, "popen_conda", _fake_popen)

    logs = []
    progresses = []
    result = tr.train_sam3_lora(
        _healthy_spec(tmp_path),
        str(run_dir),
        log_cb=logs.append,
        progress_cb=lambda e, t: progresses.append((e, t)),
    )

    assert result["success"] is True
    assert result["artifact_path"] == str(run_dir / "adapters.pt")
    assert "plain startup line" in logs
    assert "epoch 0 step 10 loss 1.0" in logs
    assert progresses == [(1, 2)]
    assert len(calls) == 1


def test_malformed_json_record_treated_as_plain_text(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "adapters.pt").write_bytes(b"fake")

    bad_line = "@@HYDRA_SAM3_PROGRESS@@{not valid json\n"
    process = _FakeProcess([bad_line], returncode=0)
    monkeypatch.setattr(tr, "popen_conda", lambda *a, **k: process)

    logs = []
    result = tr.train_sam3_lora(
        _healthy_spec(tmp_path), str(run_dir), log_cb=logs.append
    )

    assert result["success"] is True
    assert any("not valid json" in line for line in logs)


def test_nonzero_exit_reports_failure(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    run_dir = tmp_path / "run"
    process = _FakeProcess(["something went wrong\n"], returncode=1)
    monkeypatch.setattr(tr, "popen_conda", lambda *a, **k: process)

    result = tr.train_sam3_lora(_healthy_spec(tmp_path), str(run_dir))

    assert result["success"] is False
    assert "code 1" in result["error_message"]
    assert "something went wrong" in result["error_message"]
    assert not (run_dir / "adapters.pt").exists()


def test_exit_zero_without_artifact_reports_failure(tmp_path, monkeypatch):
    """THE critical rule: exit 0 with no adapters.pt written is still failure."""
    _pass_preflight(monkeypatch)
    run_dir = tmp_path / "run"
    process = _FakeProcess(["trained nothing but exited clean\n"], returncode=0)
    monkeypatch.setattr(tr, "popen_conda", lambda *a, **k: process)

    result = tr.train_sam3_lora(_healthy_spec(tmp_path), str(run_dir))

    assert result["success"] is False
    assert not (run_dir / "adapters.pt").exists()
    assert "did not write" in result["error_message"]


def test_cancellation_terminates_child(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    run_dir = tmp_path / "run"

    # Enough lines that should_cancel() can fire before the generator is
    # exhausted.
    process = _FakeProcess(["line one\n", "line two\n", "line three\n"], returncode=0)
    monkeypatch.setattr(tr, "popen_conda", lambda *a, **k: process)

    calls = {"n": 0}

    def _should_cancel():
        calls["n"] += 1
        return calls["n"] >= 2

    result = tr.train_sam3_lora(
        _healthy_spec(tmp_path), str(run_dir), should_cancel=_should_cancel
    )

    assert result["canceled"] is True
    assert result["success"] is False
    assert process.terminated is True


def test_cancellation_kills_child_if_terminate_hangs(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    run_dir = tmp_path / "run"
    process = _FakeProcess(["line one\n"], returncode=0, terminate_hangs=True)
    monkeypatch.setattr(tr, "popen_conda", lambda *a, **k: process)

    result = tr.train_sam3_lora(
        _healthy_spec(tmp_path), str(run_dir), should_cancel=lambda: True
    )

    assert result["canceled"] is True
    assert process.terminated is True
    assert process.killed is True


def test_spec_serialised_to_run_dir(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    run_dir = tmp_path / "run"

    def _fake_popen(*args, **kwargs):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "adapters.pt").write_bytes(b"fake")
        return _FakeProcess([], returncode=0)

    monkeypatch.setattr(tr, "popen_conda", _fake_popen)

    tr.train_sam3_lora(_healthy_spec(tmp_path), str(run_dir))

    spec_path = run_dir / "spec.json"
    assert spec_path.exists()
    import json

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    assert payload["sam3_params"]["prompt"] == "ant"
