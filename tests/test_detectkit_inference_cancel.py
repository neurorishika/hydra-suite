"""Cancellation coverage for protected DetectKit dataset inference."""

from __future__ import annotations


class _FakeOperation:
    def __init__(self, *_args, **_kwargs):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


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


def test_main_window_has_no_dataset_model_execution_seam():
    import inspect

    from hydra_suite.detectkit.gui import main_window

    source = inspect.getsource(main_window)
    assert "load_torch_model" not in source
    assert "predict_preview_detections_for_image" not in source
    assert "predict_sliced_obb_result" not in source
