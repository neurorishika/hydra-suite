from pathlib import Path
from types import SimpleNamespace

from hydra_suite.runtime.process_supervisor import ExitKind
from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)


def _spec(tmp_path):
    return TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path),
        base_model="yolo.pt",
        hyperparams=TrainingHyperParams(batch=1, imgsz=64, workers=0),
        device="cpu",
    )


def _budget():
    return SimpleNamespace(
        admitted=True,
        estimator_version="test-v1",
        host_peak_bytes=4 * 1024**3,
        accelerator_peak_bytes=0,
        usable_host_bytes=8 * 1024**3,
        usable_accelerator_bytes=None,
        reserved_host_bytes=1024**3,
        dominant_phase="training",
        limits=SimpleNamespace(batch_size=1, workers=0, prefetch_batches=2),
        refusals=(),
    )


def test_generic_training_uses_shared_bounded_supervisor(monkeypatch, tmp_path):
    import hydra_suite.training.ultralytics_supervisor as mod

    seen = {}

    class Process:
        returncode = 0

        def poll(self):
            return 0

    class Output:
        def drain(self, timeout=None):
            return (["Epoch 2/5\n"], True, None)

    class Sidecar:
        def __init__(self, plan, **kwargs):
            seen["plan"] = plan
            seen["kwargs"] = kwargs
            self.process = Process()
            self.output = Output()

        def wait(self):
            return SimpleNamespace(
                returncode=0,
                classified_exit=SimpleNamespace(kind=ExitKind.SUCCESS, message="ok"),
                peak_tree_rss_bytes=123,
                peak_accelerator_bytes=None,
                dropped_output_lines=0,
            )

    monkeypatch.setattr(mod, "evaluate_resource_request", lambda *a, **k: _budget())
    monkeypatch.setattr(mod, "SupervisedSidecar", Sidecar)
    progress = []
    result = mod.run_ultralytics_supervised(
        ["trainer"], _spec(tmp_path), progress_cb=lambda *args: progress.append(args)
    )

    assert result["success"] is True
    assert result["peak_tree_rss_bytes"] == 123
    assert result["resource_telemetry"]["observed"]["peak_tree_rss_bytes"] == 123
    assert seen["kwargs"]["output_max_lines"] == mod.OUTPUT_MAX_LINES
    assert seen["kwargs"]["output_max_chars"] == mod.OUTPUT_MAX_CHARS
    assert seen["plan"].launch.limits.hard_host_bytes <= _budget().usable_host_bytes


def test_generic_training_cancellation_terminates_sidecar(monkeypatch, tmp_path):
    import hydra_suite.training.ultralytics_supervisor as mod

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

    class Sidecar:
        def __init__(self, *args, **kwargs):
            self.process = Process()
            self.output = SimpleNamespace(drain=lambda timeout=None: ([], False, None))
            self.cancelled = False

        def cancel(self, grace):
            self.cancelled = True
            self.process.returncode = -15

    installed = Sidecar
    monkeypatch.setattr(mod, "evaluate_resource_request", lambda *a, **k: _budget())
    monkeypatch.setattr(mod, "SupervisedSidecar", installed)
    result = mod.run_ultralytics_supervised(
        ["trainer"], _spec(tmp_path), should_cancel=lambda: True
    )

    assert result["canceled"] is True
    assert result["failure_kind"] == ExitKind.CANCELED.value


def test_generic_training_preserves_uncertain_ownership_recovery(monkeypatch, tmp_path):
    import pytest

    import hydra_suite.training.ultralytics_supervisor as mod
    from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError

    owner = object()

    def fail(*args, **kwargs):
        raise WorkloadStillOwnedError("still owned", owner)

    monkeypatch.setattr(mod, "evaluate_resource_request", lambda *a, **k: _budget())
    monkeypatch.setattr(mod, "SupervisedSidecar", fail)

    with pytest.raises(WorkloadStillOwnedError) as caught:
        mod.run_ultralytics_supervised(["trainer"], _spec(tmp_path))
    assert caught.value.sidecar is owner


def test_generic_training_retries_recognized_oom_in_a_fresh_reduced_child(
    monkeypatch, tmp_path
):
    import hydra_suite.training.ultralytics_supervisor as mod

    calls = []

    def run_once(command, spec, **_kwargs):
        calls.append((tuple(command), spec, spec.hyperparams))
        success = len(calls) == 2
        return {
            "success": success,
            "failure_kind": (
                ExitKind.SUCCESS.value if success else ExitKind.ACCELERATOR_OOM.value
            ),
            "hard_host_bytes": 100,
            "resource_telemetry": {"observed": {"peak_tree_rss_bytes": 90}},
        }

    monkeypatch.setattr(mod, "_run_ultralytics_once", run_once)
    spec = _spec(tmp_path)
    spec.hyperparams.batch = 8

    result = mod.run_ultralytics_supervised(["trainer", "batch=8", "workers=0"], spec)

    assert result["success"] is True
    assert calls[0][0] != calls[1][0]
    assert calls[0][1] is not calls[1][1]
    assert calls[0][2] is not calls[1][2]
    assert "batch=4" in calls[1][0]
    assert result["retry_history"] == [
        {"attempt": 1, "field": "batch_size", "from": 8, "to": 4}
    ]


def _auto_spec(tmp_path, batch, imgsz=64):
    return TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path),
        base_model="yolo.pt",
        hyperparams=TrainingHyperParams(batch=batch, imgsz=imgsz, workers=0),
        device="cpu",
    )


def _capture_commands(
    monkeypatch, tmp_path, spec, *, child_writes=None, oom_on_attempt=None, **kwargs
):
    """Drive the supervisor with a fake `_run_ultralytics_once`.

    The resolution child goes through the SAME entry point, so the fake writes
    `batch_resolution.json` for it -- reflecting that the child writes a file
    rather than printing a marker line.
    """
    import json

    import hydra_suite.training.ultralytics_supervisor as mod

    commands: list[tuple[str, ...]] = []

    def run_once(command, run_spec, **_kwargs):
        command = [str(item) for item in command]
        if "hydra_suite.training.yolo_autobatch" in command:
            if child_writes is not None:
                out = command[command.index("--out") + 1]
                Path(out).write_text(json.dumps({"resolved": child_writes}))
            return {
                "success": child_writes is not None,
                "canceled": False,
                "failure_kind": ExitKind.SUCCESS.value,
            }
        commands.append(tuple(command))
        failed = oom_on_attempt is not None and len(commands) - 1 == oom_on_attempt
        return {
            "success": not failed,
            "failure_kind": (
                ExitKind.ACCELERATOR_OOM.value if failed else ExitKind.SUCCESS.value
            ),
            "hard_host_bytes": 100,
            "resource_telemetry": {"observed": {"peak_tree_rss_bytes": 90}},
        }

    monkeypatch.setattr(mod, "_run_ultralytics_once", run_once)
    monkeypatch.setattr(
        mod, "_accelerator", lambda device: (mod.AcceleratorKind.CUDA, None)
    )
    result = mod.run_ultralytics_supervised(
        ["trainer", f"batch={spec.hyperparams.batch}", "workers=0"],
        spec,
        run_dir=tmp_path,
        **kwargs,
    )
    return commands, result


def test_resolution_runs_before_launch_and_the_command_has_a_positive_batch(
    monkeypatch, tmp_path
):
    commands, _ = _capture_commands(
        monkeypatch, tmp_path, _auto_spec(tmp_path, -1), child_writes=24
    )
    assert "batch=-1" not in commands[0]
    assert "batch=24" in commands[0]


def test_the_resolved_batch_is_persisted_with_its_provenance(monkeypatch, tmp_path):
    import json

    _capture_commands(monkeypatch, tmp_path, _auto_spec(tmp_path, -1), child_writes=24)
    payload = json.loads((tmp_path / "batch_resolution.json").read_text())
    assert payload["requested"] == -1
    assert payload["resolved"] == 24
    assert payload["provenance"] == "ultralytics_autobatch"
    assert payload["resolved_at_unix_ns"] > 0


def test_an_explicit_batch_is_never_rewritten_and_runs_no_child(monkeypatch, tmp_path):
    commands, _ = _capture_commands(monkeypatch, tmp_path, _auto_spec(tmp_path, 16))
    assert "batch=16" in commands[0]


def test_a_memory_pressure_retry_halves_from_the_resolved_batch(monkeypatch, tmp_path):
    commands, _ = _capture_commands(
        monkeypatch,
        tmp_path,
        _auto_spec(tmp_path, -1),
        child_writes=24,
        oom_on_attempt=0,
    )
    assert "batch=24" in commands[0]
    assert "batch=12" in commands[1]


def test_a_resolution_failure_still_launches_with_a_positive_batch(
    monkeypatch, tmp_path
):
    commands, _ = _capture_commands(
        monkeypatch, tmp_path, _auto_spec(tmp_path, -1), child_writes=None
    )
    assert "batch=16" in commands[0]


def test_cancellation_during_resolution_launches_no_training(monkeypatch, tmp_path):
    commands, result = _capture_commands(
        monkeypatch,
        tmp_path,
        _auto_spec(tmp_path, -1),
        child_writes=24,
        should_cancel=lambda: True,
    )
    assert commands == []
    assert result["canceled"] is True
    assert not (tmp_path / "batch_resolution.json").exists()


def test_host_estimate_uses_the_resolved_batch(tmp_path):
    import pytest

    import hydra_suite.training.ultralytics_supervisor as mod

    big = mod._estimate_host_bytes(_auto_spec(tmp_path, 24, imgsz=1280))
    small = mod._estimate_host_bytes(_auto_spec(tmp_path, 1, imgsz=1280))
    assert big > small
    with pytest.raises(ValueError, match="resolved"):
        mod._estimate_host_bytes(_auto_spec(tmp_path, -1, imgsz=1280))


def test_a_bare_ordinal_device_reaches_the_cuda_resolution_path(monkeypatch, tmp_path):
    """`device: "0"` is the documented DetectKit config. Before normalisation
    it classified as CPU, so `batch: -1` silently ran the default."""
    import hydra_suite.training.ultralytics_supervisor as mod

    asked: list[str] = []

    def recorder(device):
        asked.append(device)
        return mod.AcceleratorKind.CUDA, None

    monkeypatch.setattr(mod, "_accelerator", recorder)
    spec = _auto_spec(tmp_path, -1)
    spec.device = "0"

    def run_once(command, run_spec, **_kwargs):
        command = [str(item) for item in command]
        if "hydra_suite.training.yolo_autobatch" in command:
            Path(command[command.index("--out") + 1]).write_text('{"resolved": 24}')
            return {"success": True, "canceled": False}
        return {
            "success": True,
            "failure_kind": ExitKind.SUCCESS.value,
            "hard_host_bytes": 100,
            "resource_telemetry": {"observed": {"peak_tree_rss_bytes": 90}},
        }

    monkeypatch.setattr(mod, "_run_ultralytics_once", run_once)
    mod.run_ultralytics_supervised(
        ["trainer", "batch=-1", "workers=0", "device=0"], spec, run_dir=tmp_path
    )
    assert asked == ["cuda:0"]


def test_normalisation_never_changes_the_launch_command_device(monkeypatch, tmp_path):
    """Ultralytics expects its OWN convention in the command. Normalisation is
    for our classification, not for what we pass through."""
    import hydra_suite.training.ultralytics_supervisor as mod

    monkeypatch.setattr(
        mod, "_accelerator", lambda device: (mod.AcceleratorKind.CUDA, None)
    )
    spec = _auto_spec(tmp_path, -1)
    spec.device = "0"
    commands: list[tuple[str, ...]] = []

    def run_once(command, run_spec, **_kwargs):
        command = [str(item) for item in command]
        if "hydra_suite.training.yolo_autobatch" in command:
            Path(command[command.index("--out") + 1]).write_text('{"resolved": 24}')
            return {"success": True, "canceled": False}
        commands.append(tuple(command))
        return {
            "success": True,
            "failure_kind": ExitKind.SUCCESS.value,
            "hard_host_bytes": 100,
            "resource_telemetry": {"observed": {"peak_tree_rss_bytes": 90}},
        }

    monkeypatch.setattr(mod, "_run_ultralytics_once", run_once)
    mod.run_ultralytics_supervised(
        ["trainer", "batch=-1", "workers=0", "device=0"], spec, run_dir=tmp_path
    )
    assert "device=0" in commands[0]
    assert "device=cuda:0" not in commands[0]
    assert spec.device == "0"
    assert "batch=24" in commands[0]


def test_an_unavailable_normalised_device_falls_back_instead_of_refusing(
    monkeypatch, tmp_path
):
    """Normalising must not turn a run that used to launch on CPU into a
    refusal on a box with no CUDA."""
    import hydra_suite.training.ultralytics_supervisor as mod

    def unavailable(device):
        if device.startswith("cuda"):
            raise RuntimeError("the requested CUDA device is unavailable")
        return mod.AcceleratorKind.CPU, None

    monkeypatch.setattr(mod, "_accelerator", unavailable)
    spec = _auto_spec(tmp_path, -1)
    spec.device = "0"
    commands: list[tuple[str, ...]] = []

    def run_once(command, run_spec, **_kwargs):
        commands.append(tuple(str(item) for item in command))
        return {
            "success": True,
            "failure_kind": ExitKind.SUCCESS.value,
            "hard_host_bytes": 100,
            "resource_telemetry": {"observed": {"peak_tree_rss_bytes": 90}},
        }

    monkeypatch.setattr(mod, "_run_ultralytics_once", run_once)
    result = mod.run_ultralytics_supervised(
        ["trainer", "batch=-1", "workers=0", "device=0"], spec, run_dir=tmp_path
    )
    assert result["success"] is True
    assert "batch=16" in commands[0]
