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


def test_init_pose_backend_yolo_no_build_runtime_config(monkeypatch, tmp_path) -> None:
    """Task 5: the YOLO pose branch must construct ``YoloNativeBackend``
    directly (mirroring ``core/inference/stages/pose.py::load_pose_model`` and
    Task 3's ``_preview_run_pose_overlay``) instead of calling the legacy
    ``build_runtime_config`` translation step.

    Task 8: ``build_runtime_config`` was since deleted from
    ``core/identity/pose/api.py`` entirely (zero real callers remained), so
    it cannot be called here."""
    pose_api = importlib.import_module("hydra_suite.core.identity.pose.api")
    yolo_module = importlib.import_module(
        "hydra_suite.core.identity.pose.backends.yolo"
    )

    assert not hasattr(pose_api, "build_runtime_config")

    captured: dict[str, object] = {}

    class FakeYoloBackend:
        def __init__(
            self, model_path, device, min_valid_conf, keypoint_names, batch_size
        ):
            captured["model_path"] = model_path
            captured["device"] = device
            captured["min_valid_conf"] = min_valid_conf
            captured["batch_size"] = batch_size
            self.output_keypoint_names = ["kpt0", "kpt1"]

        def warmup(self) -> None:
            captured["warmed_up"] = True

    monkeypatch.setattr(yolo_module, "YoloNativeBackend", FakeYoloBackend)

    worker = InterpolatedCropsWorker(
        "tracks.csv",
        "source.mp4",
        "cache.npz",
        {
            "ENABLE_POSE_EXTRACTOR": True,
            "POSE_MODEL_TYPE": "yolo",
            "POSE_MODEL_DIR": "/models/yolo_pose.pt",
            "POSE_MIN_KPT_CONF_VALID": 0.3,
            "POSE_BATCH_SIZE": 8,
            "COMPUTE_RUNTIME": "cuda",
        },
    )

    backend, kpt_source_names, kpt_labels = worker._init_pose_backend(str(tmp_path))

    assert isinstance(backend, FakeYoloBackend)
    assert captured["device"] == "cuda:0"
    assert captured["model_path"] == "/models/yolo_pose.pt"
    assert captured["warmed_up"] is True
    assert kpt_source_names == ["kpt0", "kpt1"]
    assert kpt_labels


def test_init_pose_backend_sleap_no_build_runtime_config(monkeypatch, tmp_path) -> None:
    """Task 5: the SLEAP pose branch must construct ``PoseRuntimeConfig``
    directly instead of calling the legacy ``build_runtime_config`` translation
    step.

    Task 8: ``build_runtime_config`` was since deleted from
    ``core/identity/pose/api.py`` entirely (zero real callers remained), so
    it cannot be called here."""
    pose_api = importlib.import_module("hydra_suite.core.identity.pose.api")
    pose_types = importlib.import_module("hydra_suite.core.identity.pose.types")

    assert not hasattr(pose_api, "build_runtime_config")

    captured: dict[str, object] = {}

    class FakeBackend:
        output_keypoint_names = ["kpt0"]

        def warmup(self) -> None:
            captured["warmed_up"] = True

    def _fake_create_pose_backend_from_config(config):
        captured["pose_config"] = config
        return FakeBackend()

    monkeypatch.setattr(
        pose_api,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    worker = InterpolatedCropsWorker(
        "tracks.csv",
        "source.mp4",
        "cache.npz",
        {
            "ENABLE_POSE_EXTRACTOR": True,
            "POSE_MODEL_TYPE": "sleap",
            "POSE_MODEL_DIR": "/models/sleap_model",
            "POSE_MIN_KPT_CONF_VALID": 0.25,
            "POSE_BATCH_SIZE": 4,
            "POSE_SLEAP_ENV": "sleap_env_x",
            "POSE_SLEAP_MAX_INSTANCES": 2,
            "COMPUTE_RUNTIME": "cuda",
        },
    )

    backend, kpt_source_names, kpt_labels = worker._init_pose_backend(str(tmp_path))

    assert isinstance(backend, FakeBackend)
    assert captured["warmed_up"] is True
    assert kpt_source_names == ["kpt0"]

    pose_config = captured["pose_config"]
    assert isinstance(pose_config, pose_types.PoseRuntimeConfig)
    assert pose_config.backend_family == "sleap"
    assert pose_config.runtime_flavor == "onnx_cuda"
    assert pose_config.device == "cuda"
    assert pose_config.sleap_device == "cuda"
    assert pose_config.sleap_env == "sleap_env_x"
    assert pose_config.sleap_max_instances == 2
    assert pose_config.model_path == "/models/sleap_model"
