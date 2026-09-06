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
    # `commands` filters the resolution argv out, so assert on the artifact a
    # child would have to leave and on the provenance it would have to claim.
    import json

    payload = json.loads((tmp_path / "batch_resolution.json").read_text())
    assert payload["provenance"] == "explicit"
    assert payload["requested"] == payload["resolved"] == 16


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


def test_the_effective_batch_records_what_the_oom_ladder_settled_on(
    monkeypatch, tmp_path
):
    """`resolved` is what resolution chose; `effective_batch` is what ran.

    The ladder halves the batch in a fresh child on a classified OOM, so a
    durable artifact carrying only `resolved` would claim a batch twice the
    size of the one that actually trained.
    """

    import json

    _capture_commands(
        monkeypatch,
        tmp_path,
        _auto_spec(tmp_path, -1),
        child_writes=24,
        oom_on_attempt=0,
    )
    payload = json.loads((tmp_path / "batch_resolution.json").read_text())
    assert payload["resolved"] == 24
    assert payload["effective_batch"] == 12


def test_the_effective_batch_equals_the_resolved_batch_without_a_retry(
    monkeypatch, tmp_path
):
    import json

    _capture_commands(monkeypatch, tmp_path, _auto_spec(tmp_path, -1), child_writes=24)
    payload = json.loads((tmp_path / "batch_resolution.json").read_text())
    assert payload["resolved"] == payload["effective_batch"] == 24


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


def test_a_bare_ordinal_on_a_box_without_cuda_launches_rather_than_refusing(
    monkeypatch, tmp_path
):
    """The same case, driven end to end through the REAL `_accelerator`."""
    import hydra_suite.training.ultralytics_supervisor as mod

    try:
        real_kind, _cuda = mod._accelerator("cuda:0")
    except RuntimeError:
        real_kind = None
    if real_kind is mod.AcceleratorKind.CUDA:
        import pytest

        pytest.skip("this box has CUDA; the absent-device path cannot be taken")

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
    # No resolution child ran, and the run launched with a positive batch.
    assert len(commands) == 1
    assert "batch=16" in commands[0]
    assert "device=0" in commands[0]

    import json

    assert (
        json.loads((tmp_path / "batch_resolution.json").read_text())["provenance"]
        == "default_non_cuda"
    )


def test_a_raising_resolution_child_does_not_abort_the_training_run(
    monkeypatch, tmp_path
):
    """`_run_ultralytics_once` raises on real paths -- notably `raise
    output_error` in its drain loop, the exact dropped/erroring-output
    condition that made us read a file rather than a stdout marker. runner.py
    has no try around `run_ultralytics_supervised`, so an escape would kill
    the run."""
    import hydra_suite.training.ultralytics_supervisor as mod

    monkeypatch.setattr(
        mod, "_accelerator", lambda device: (mod.AcceleratorKind.CUDA, None)
    )
    commands: list[tuple[str, ...]] = []

    def run_once(command, run_spec, **_kwargs):
        text = [str(item) for item in command]
        if "hydra_suite.training.yolo_autobatch" in text:
            raise OSError("the child output channel failed")
        commands.append(tuple(text))
        return {
            "success": True,
            "failure_kind": ExitKind.SUCCESS.value,
            "hard_host_bytes": 100,
            "resource_telemetry": {"observed": {"peak_tree_rss_bytes": 90}},
        }

    monkeypatch.setattr(mod, "_run_ultralytics_once", run_once)
    result = mod.run_ultralytics_supervised(
        ["trainer", "batch=-1", "workers=0"], _auto_spec(tmp_path, -1), run_dir=tmp_path
    )
    assert result["success"] is True
    assert "batch=16" in commands[0]


def test_a_still_owned_resolution_child_stops_the_run(monkeypatch, tmp_path):
    """The one deliberate exception: never launch training on top of a child
    that may still hold the lease."""
    import pytest

    import hydra_suite.training.ultralytics_supervisor as mod
    from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError

    owner = object()
    monkeypatch.setattr(
        mod, "_accelerator", lambda device: (mod.AcceleratorKind.CUDA, None)
    )
    launched: list = []

    def run_once(command, run_spec, **_kwargs):
        text = [str(item) for item in command]
        if "hydra_suite.training.yolo_autobatch" in text:
            raise WorkloadStillOwnedError("still owned", owner)
        launched.append(text)
        return {"success": True, "failure_kind": ExitKind.SUCCESS.value}

    monkeypatch.setattr(mod, "_run_ultralytics_once", run_once)
    with pytest.raises(WorkloadStillOwnedError) as caught:
        mod.run_ultralytics_supervised(
            ["trainer", "batch=-1", "workers=0"],
            _auto_spec(tmp_path, -1),
            run_dir=tmp_path,
        )
    assert caught.value.sidecar is owner
    assert launched == []


# --- bare-ordinal accelerator classification + its warning period -----------


def _cuda_device(free_bytes, total_bytes=64 * 1024**3):
    return SimpleNamespace(
        name="NVIDIA Test",
        uuid="GPU-1111",
        free_bytes=free_bytes,
        total_bytes=total_bytes,
    )


def _fake_probe_resources(host, gpu_free=None, gpu_total=None):
    """A `probe_resources` double accepting both call shapes.

    ``host`` is ``(total_host_bytes, available_host_bytes)``; ``gpu_free`` is
    what a CUDA-kind observation reports as available device memory.
    """

    from hydra_suite.runtime.resource_budget import AcceleratorKind, ResourceObservation

    def probe(accelerator_kind, accelerator_name=None, accelerator_probe=None):
        if accelerator_kind is AcceleratorKind.CUDA:
            return ResourceObservation(
                total_host_bytes=total_host,
                available_host_bytes=available_host,
                accelerator_kind=AcceleratorKind.CUDA,
                accelerator_name=accelerator_name or "NVIDIA Test",
                total_accelerator_bytes=gpu_total or 64 * 1024**3,
                available_accelerator_bytes=gpu_free,
            )
        return ResourceObservation(
            total_host_bytes=total_host,
            available_host_bytes=available_host,
            accelerator_kind=AcceleratorKind.CPU,
        )

    total_host, available_host = host
    return probe


def _launch_sidecar(monkeypatch, mod, launched):
    class Process:
        returncode = 0

        def poll(self):
            return 0

    class Output:
        def drain(self, timeout=None):
            return ([], True, None)

    class Sidecar:
        def __init__(self, plan, **kwargs):
            # The real SupervisedSidecar runs this before launching. Swallowing
            # it would leave the downgrade's `budget = legacy` untested: the
            # prelaunch accelerator gate is the second place the widening could
            # newly refuse.
            check = kwargs.get("prelaunch_check")
            if check is not None:
                check()
            # The real SupervisedSidecar also polls `accelerator_probe` while
            # the child runs. Dropping it left the accelerator re-probe site
            # completely uncovered while three sibling sites were tested --
            # exactly how a gap survives review.
            probe = kwargs.get("accelerator_probe")
            if probe is not None:
                probe()
            launched.append(plan)
            self.process = Process()
            self.output = Output()

        def wait(self):
            return SimpleNamespace(
                returncode=0,
                classified_exit=SimpleNamespace(kind=ExitKind.SUCCESS, message="ok"),
                peak_tree_rss_bytes=1,
                peak_accelerator_bytes=None,
                dropped_output_lines=0,
            )

    monkeypatch.setattr(mod, "SupervisedSidecar", Sidecar)


def _bare_ordinal_run(monkeypatch, tmp_path, *, host, gpu_free, device="0"):
    """Drive one real-gate run for a bare-ordinal device; return (result, log)."""

    import hydra_suite.training.ultralytics_supervisor as mod
    from hydra_suite.training.sam3_lora import preflight

    monkeypatch.setattr(
        preflight, "_probe_cuda_device", lambda dev: _cuda_device(gpu_free)
    )
    monkeypatch.setattr(
        mod, "probe_resources", _fake_probe_resources(host, gpu_free=gpu_free)
    )
    launched = []
    _launch_sidecar(monkeypatch, mod, launched)
    spec = _spec(tmp_path)
    spec.device = device
    spec.hyperparams = TrainingHyperParams(batch=2, imgsz=64, workers=0)
    log = []
    result = mod.run_ultralytics_supervised(
        ["trainer", "device=0"], spec, run_dir=tmp_path, log_cb=log.append
    )
    return result, log, launched


def test_a_bare_ordinal_is_classified_cuda_and_gets_a_uuid_pin(monkeypatch, tmp_path):
    import hydra_suite.training.ultralytics_supervisor as mod
    from hydra_suite.training.sam3_lora import preflight

    monkeypatch.setattr(
        preflight, "_probe_cuda_device", lambda dev: _cuda_device(40 * 1024**3)
    )
    kind, observed = mod._accelerator("0")
    assert kind is mod.AcceleratorKind.CUDA
    assert observed is not None and observed.uuid == "GPU-1111"

    result, _log, launched = _bare_ordinal_run(
        monkeypatch,
        tmp_path,
        host=(256 * 1024**3, 200 * 1024**3),
        gpu_free=40 * 1024**3,
    )
    assert result["success"] is True
    assert launched[0].launch.environment["CUDA_VISIBLE_DEVICES"] == "GPU-1111"


def test_a_multi_gpu_ordinal_resolves_against_the_first_device(monkeypatch, tmp_path):
    import hydra_suite.training.ultralytics_supervisor as mod
    from hydra_suite.training.sam3_lora import preflight

    asked = []

    def probe(device):
        asked.append(device)
        return _cuda_device(40 * 1024**3)

    monkeypatch.setattr(preflight, "_probe_cuda_device", probe)
    kind, _observed = mod._accelerator("0,1")
    assert kind is mod.AcceleratorKind.CUDA
    assert asked == ["cuda:0"]

    _result, log, _launched = _bare_ordinal_run(
        monkeypatch,
        tmp_path,
        host=(256 * 1024**3, 200 * 1024**3),
        gpu_free=40 * 1024**3,
        device="0,1",
    )
    assert any("cuda:0" in line for line in log)


def test_the_new_accelerator_gate_warns_instead_of_refusing(monkeypatch, tmp_path):
    """The whole point of the warning period: a run the widened classification
    would newly refuse still launches, loudly."""

    result, log, launched = _bare_ordinal_run(
        monkeypatch,
        tmp_path,
        host=(256 * 1024**3, 200 * 1024**3),
        gpu_free=1 * 1024**3,
    )

    assert result["success"] is True
    assert launched, "the run must still launch during the warning period"
    warning = "\n".join(result["admission_warnings"])
    assert warning and warning in "\n".join(log)
    assert "device=0" in warning
    assert "classified as CPU" in warning and "classified as CUDA" in warning
    assert "REFUSED IN A FUTURE RELEASE" in warning
    assert "GiB" in warning


def test_a_pre_existing_refusal_is_still_refused_during_the_warning_period(
    monkeypatch, tmp_path
):
    """Scoping proof: the host gate refused this run BEFORE the widening, so
    the warning period must not rescue it."""

    result, log, launched = _bare_ordinal_run(
        monkeypatch,
        tmp_path,
        host=(16 * 1024**3, 1 * 1024**3),
        gpu_free=1 * 1024**3,
    )

    assert result["success"] is False
    assert result["failure_kind"] == ExitKind.HOST_ADMISSION_REFUSAL.value
    assert not launched, "a pre-existing refusal must never launch"
    assert result["admission_warnings"] == []
    assert not any("FUTURE RELEASE" in line for line in log)


def test_the_widened_classification_keeps_device_zero_in_the_launch_command(
    monkeypatch, tmp_path
):
    import hydra_suite.training.ultralytics_supervisor as mod
    from hydra_suite.training.sam3_lora import preflight

    monkeypatch.setattr(
        preflight, "_probe_cuda_device", lambda dev: _cuda_device(40 * 1024**3)
    )
    monkeypatch.setattr(
        mod,
        "probe_resources",
        _fake_probe_resources((256 * 1024**3, 200 * 1024**3), gpu_free=40 * 1024**3),
    )
    launched = []
    _launch_sidecar(monkeypatch, mod, launched)
    spec = _spec(tmp_path)
    spec.device = "0"
    spec.hyperparams = TrainingHyperParams(batch=2, imgsz=64, workers=0)
    mod.run_ultralytics_supervised(
        ["trainer", "batch=2", "device=0"], spec, run_dir=tmp_path
    )
    command = list(launched[0].launch.command)
    assert "device=0" in command
    assert "device=cuda:0" not in command
    assert spec.device == "0"


def test_a_bare_ordinal_without_cuda_still_returns_cpu_rather_than_raising(
    monkeypatch,
):
    """The hazard an earlier round hit: normalising "0" must not convert a
    working CPU run into a hard failure on a box with no CUDA."""

    import hydra_suite.training.ultralytics_supervisor as mod
    from hydra_suite.training.sam3_lora import preflight

    asked = []

    def absent(device):
        asked.append(device)
        return None

    monkeypatch.setattr(preflight, "_probe_cuda_device", absent)
    assert mod._accelerator("0") == (mod.AcceleratorKind.CPU, None)
    assert asked == ["cuda:0"]


def test_ending_the_warning_period_is_a_one_line_change(monkeypatch, tmp_path):
    """Requirement 4: flipping the named constant must actually restore the
    refusal, so the flag can never quietly become inert."""

    import hydra_suite.training.ultralytics_supervisor as mod

    monkeypatch.setattr(mod, "BARE_ORDINAL_ACCELERATOR_GATE_WARNING_PERIOD", False)
    result, log, launched = _bare_ordinal_run(
        monkeypatch,
        tmp_path,
        host=(256 * 1024**3, 200 * 1024**3),
        gpu_free=1 * 1024**3,
    )

    assert result["success"] is False
    assert result["failure_kind"] == ExitKind.HOST_ADMISSION_REFUSAL.value
    assert launched == []
    assert result["admission_warnings"] == []
    assert not any("FUTURE RELEASE" in line for line in log)


def test_a_downgraded_run_is_recorded_in_the_durable_telemetry(monkeypatch, tmp_path):
    """`result["admission_warnings"]` has no consumer in src/; the telemetry
    dict is what gets persisted, so the downgrade must be stamped there."""

    result, _log, _launched = _bare_ordinal_run(
        monkeypatch,
        tmp_path,
        host=(256 * 1024**3, 200 * 1024**3),
        gpu_free=1 * 1024**3,
    )
    telemetry = result["resource_telemetry"]
    assert telemetry["admission_downgraded"] is True
    assert any("FUTURE RELEASE" in item for item in telemetry["admission_warnings"])

    clean, _log2, _launched2 = _bare_ordinal_run(
        monkeypatch,
        tmp_path,
        host=(256 * 1024**3, 200 * 1024**3),
        gpu_free=40 * 1024**3,
    )
    assert clean["resource_telemetry"]["admission_downgraded"] is False


def _probe_recording_run(monkeypatch, tmp_path, *, uuid_by_call, device="0"):
    """One real-gate run whose re-probes are recorded; returns (asked, result).

    The probe answers by DEVICE STRING, exactly like `nvidia-smi` does: a UUID
    argument would find nothing, because the parent's CUDA_VISIBLE_DEVICES is
    not set to the pin (only the CHILD's environment is).
    """

    import hydra_suite.training.ultralytics_supervisor as mod
    from hydra_suite.training.sam3_lora import preflight

    asked = []

    def probe(dev):
        asked.append(dev)
        uuid = uuid_by_call(len(asked))
        if uuid is None:
            return None
        observed = _cuda_device(40 * 1024**3)
        observed.uuid = uuid
        return observed

    monkeypatch.setattr(preflight, "_probe_cuda_device", probe)
    monkeypatch.setattr(
        mod,
        "probe_resources",
        _fake_probe_resources((256 * 1024**3, 200 * 1024**3), gpu_free=40 * 1024**3),
    )
    launched = []
    _launch_sidecar(monkeypatch, mod, launched)
    spec = _spec(tmp_path)
    spec.device = device
    spec.hyperparams = TrainingHyperParams(batch=2, imgsz=64, workers=0)
    result = mod.run_ultralytics_supervised(
        ["trainer", "device=0"], spec, run_dir=tmp_path
    )
    return asked, result


def test_the_parent_reprobes_by_device_string_not_by_the_pinned_uuid(
    monkeypatch, tmp_path
):
    """The prelaunch check and the accelerator probe must not ask for a UUID.

    `_probe_cuda_device` resolves a UUID only out of CUDA_VISIBLE_DEVICES, and
    the parent never sets that -- the pin goes on the CHILD's environment. So
    a UUID argument returned None and BOTH sites raised unconditionally
    ("the selected physical CUDA device changed" / "telemetry became
    unavailable"), no matter how much memory was free. Verified against a real
    device on the CUDA box.
    """

    asked, result = _probe_recording_run(
        monkeypatch, tmp_path, uuid_by_call=lambda _n: "GPU-1111"
    )

    assert result["success"] is True
    assert asked, "the parent must re-probe before launching"
    assert not any(str(item).startswith("GPU-") for item in asked)
    assert set(asked) == {"cuda:0"}


def test_a_device_swap_before_launch_is_still_detected(monkeypatch, tmp_path):
    """The `.uuid` equality check is what detects a swap; probing by string
    must not weaken it."""

    _asked, result = _probe_recording_run(
        monkeypatch,
        tmp_path,
        # First call classifies the accelerator and sets the pin; every later
        # call is a re-probe, and reports a DIFFERENT physical device.
        uuid_by_call=lambda n: "GPU-1111" if n == 1 else "GPU-2222",
    )

    assert result["success"] is False
    assert "the selected physical CUDA device changed" in result["error_message"]
    # It must fail for the RIGHT reason: a real swap, not a UUID re-probe that
    # can never resolve. Without this the broken version passes trivially.
    assert not any(str(item).startswith("GPU-") for item in _asked)


def test_missing_device_telemetry_before_launch_is_still_refused(monkeypatch, tmp_path):
    _asked, result = _probe_recording_run(
        monkeypatch,
        tmp_path,
        uuid_by_call=lambda n: "GPU-1111" if n == 1 else None,
    )

    assert result["success"] is False
    assert "CUDA device" in result["error_message"]
    assert not any(str(item).startswith("GPU-") for item in _asked)
