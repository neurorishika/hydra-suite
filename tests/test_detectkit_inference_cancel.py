"""Cancellation coverage for DetectKit dataset inference."""

from __future__ import annotations

from hydra_suite.detectkit.gui import main_window


def test_dataset_inference_cancel_discards_current_image_and_stops(monkeypatch):
    worker = main_window._DetectKitDatasetInferenceWorker(
        ["first.png", "second.png"],
        "model.pt",
        "cpu",
        0.01,
    )
    calls: list[str] = []
    successes: list[dict] = []
    statuses: list[str] = []

    monkeypatch.setattr(
        main_window, "load_torch_model", lambda *_a, **_k: (object(), "cpu")
    )

    def _predict(_model, image_path, *, should_stop, **_kwargs):
        calls.append(image_path)
        worker.cancel()
        assert should_stop()
        return [{"class_id": 0, "confidence": 0.9}]

    monkeypatch.setattr(main_window, "predict_preview_detections_for_image", _predict)
    worker.success.connect(successes.append)
    worker.status.connect(statuses.append)

    worker.execute()

    assert calls == ["first.png"]
    assert successes == []
    assert statuses[-1] == "Inference cancelled."
    assert worker.is_cancelled()


def test_dataset_inference_cancel_is_thread_safe_and_idempotent():
    worker = main_window._DetectKitDatasetInferenceWorker([], "model.pt", "cpu", 0.01)

    worker.cancel()
    worker.cancel()

    assert worker.is_cancelled()
