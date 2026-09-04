"""Cancellation coverage for protected DetectKit dataset inference."""

from __future__ import annotations


class _FakeOperation:
    def __init__(self, *_args, **_kwargs):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _RecoverySidecar:
    def __init__(self):
        self.fail_cleanup = True
        self.cancel_calls = 0

    def cancel(self):
        self.cancel_calls += 1
        if self.fail_cleanup:
            from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError

            raise WorkloadStillOwnedError("still owned", self)


class _RecoveryOperation(_FakeOperation):
    def __init__(self, sidecar):
        super().__init__()
        self.sidecar = sidecar

    def run(self, **_kwargs):
        from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError

        error = WorkloadStillOwnedError(
            "guardian could not prove quiescence", self.sidecar
        )
        error.recovery_cleanup = lambda: setattr(self, "recovered", True)
        raise error


def _worker(tmp_path, monkeypatch):
    from hydra_suite.detectkit.jobs import dataset_inference

    monkeypatch.setattr(dataset_inference, "ProtectedOperation", _FakeOperation)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    return dataset_inference.DatasetInferenceWorker(
        project_dir=tmp_path / "project",
        source_path=source,
        model_path=str(model),
        device_preference="cpu",
        confidence_threshold=0.01,
    )


def test_dataset_inference_cancel_delegates_to_process_group_owner(
    tmp_path, monkeypatch
):
    worker = _worker(tmp_path, monkeypatch)
    worker.cancel()
    assert worker._operation.cancelled
    assert worker.is_cancelled()


def test_dataset_inference_cancel_is_idempotent(tmp_path, monkeypatch):
    worker = _worker(tmp_path, monkeypatch)
    worker.cancel()
    worker.cancel()
    assert worker.is_cancelled()


def test_dataset_inference_retains_and_retries_uncertain_containment(
    tmp_path, monkeypatch
):
    from hydra_suite.detectkit.jobs import dataset_inference

    sidecar = _RecoverySidecar()
    operation = _RecoveryOperation(sidecar)
    monkeypatch.setattr(
        dataset_inference, "ProtectedOperation", lambda *_a, **_k: operation
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    worker = dataset_inference.DatasetInferenceWorker(
        project_dir=tmp_path / "project",
        source_path=source,
        model_path=str(model),
        device_preference="cpu",
        confidence_threshold=0.01,
    )

    worker.run()

    assert worker.containment_recovery_required
    assert not worker.retry_containment_cleanup()
    assert worker.containment_recovery_required
    assert not getattr(operation, "recovered", False)

    sidecar.fail_cleanup = False

    assert worker.retry_containment_cleanup()
    assert worker.failure_exception is None
    assert operation.recovered


def test_main_window_has_no_dataset_model_execution_seam():
    import inspect

    from hydra_suite.detectkit.gui import main_window

    source = inspect.getsource(main_window)
    assert "load_torch_model" not in source
    assert "predict_preview_detections_for_image" not in source
    assert "predict_sliced_obb_result" not in source
