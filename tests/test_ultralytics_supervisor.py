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
        usable_host_bytes=8 * 1024**3,
        reserved_host_bytes=1024**3,
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
