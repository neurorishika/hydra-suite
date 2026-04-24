from __future__ import annotations

import importlib

import pandas as pd

from hydra_suite.trackerkit.gui.workers.crops_worker import InterpolatedCropsWorker


def test_interpolated_worker_skips_backend_init_when_no_eligible_gaps(
    monkeypatch,
) -> None:
    worker = InterpolatedCropsWorker("tracks.csv", "source.mp4", "cache.npz", {})
    emitted: list[dict[str, object]] = []
    finalized = {"called": False}

    class FakeCap:
        def release(self) -> None:
            return None

    class FakeGenerator:
        def finalize(self) -> None:
            finalized["called"] = True

    worker.finished_signal.connect(lambda result: emitted.append(result))

    monkeypatch.setattr(
        InterpolatedCropsWorker,
        "_validate_and_setup",
        lambda self, profiler: (
            pd.DataFrame(
                [
                    {
                        "TrajectoryID": 1,
                        "FrameID": 0,
                        "State": "active",
                        "X": 0.0,
                        "Y": 0.0,
                        "Theta": 0.0,
                    }
                ]
            ),
            FakeCap(),
            None,
            FakeGenerator(),
            "unused-output-dir",
            False,
            True,
            1.0,
            1.0,
            2.0,
            0.1,
        ),
    )
    monkeypatch.setattr(
        InterpolatedCropsWorker,
        "_detect_interpolation_gaps",
        lambda self, df, detection_cache, position_scale, size_scale: ({}, 4, 0, 0),
    )
    monkeypatch.setattr(
        InterpolatedCropsWorker,
        "_init_interpolation_backends",
        lambda self, output_dir: (_ for _ in ()).throw(
            AssertionError("backend initialization should be skipped")
        ),
    )
    monkeypatch.setattr(
        InterpolatedCropsWorker,
        "_cleanup_backends",
        lambda self, *args, **kwargs: None,
    )

    worker.execute()

    assert finalized["called"] is True
    assert len(emitted) == 1
    assert emitted[0]["no_work_reason"] == "no_eligible_gaps"
    assert emitted[0]["occluded_rows"] == 4
    assert emitted[0]["eligible_frames"] == 0
    assert emitted[0]["eligible_rows"] == 0
    assert emitted[0]["pose_rows_produced"] == 0
    assert emitted[0]["cnn_rows_produced"] == 0


def test_interpolated_worker_uses_split_cnn_and_headtail_runtimes(
    monkeypatch,
    tmp_path,
) -> None:
    cnn_module = importlib.import_module("hydra_suite.core.identity.classification.cnn")
    headtail_module = importlib.import_module(
        "hydra_suite.core.identity.classification.headtail"
    )

    cnn_model = tmp_path / "cnn_model.pth"
    cnn_model.write_text("cnn", encoding="utf-8")
    headtail_model = tmp_path / "headtail_model.pt"
    headtail_model.write_text("ht", encoding="utf-8")

    observed: dict[str, object] = {}

    class FakeCNNConfig:
        def __init__(self, model_path: str, confidence: float, batch_size: int) -> None:
            self.model_path = model_path
            self.confidence = confidence
            self.batch_size = batch_size

    class FakeCNNBackend:
        def __init__(
            self, config, model_path: str | None = None, compute_runtime: str = "cpu"
        ) -> None:
            observed["cnn_runtime"] = compute_runtime

    class FakeHeadTailAnalyzer:
        def __init__(self, model_path: str, device: str = "cpu", **kwargs) -> None:
            observed["headtail_device"] = device
            self.is_available = True

        def close(self) -> None:
            return None

    monkeypatch.setattr(cnn_module, "CNNIdentityConfig", FakeCNNConfig)
    monkeypatch.setattr(cnn_module, "CNNIdentityBackend", FakeCNNBackend)
    monkeypatch.setattr(headtail_module, "HeadTailAnalyzer", FakeHeadTailAnalyzer)

    worker = InterpolatedCropsWorker(
        "tracks.csv",
        "source.mp4",
        "cache.npz",
        {
            "CNN_CLASSIFIERS": [
                {"label": "cnn_identity", "model_path": str(cnn_model), "batch_size": 4}
            ],
            "CNN_COMPUTE_RUNTIME": "onnx_cpu",
            "COMPUTE_RUNTIME": "mps",
            "YOLO_HEADTAIL_MODEL_PATH": str(headtail_model),
            "HEADTAIL_COMPUTE_RUNTIME": "cuda",
        },
    )

    worker._init_cnn_backends()
    worker._init_headtail_analyzer()

    assert observed["cnn_runtime"] == "onnx_cpu"
    assert observed["headtail_device"] == "cuda"
