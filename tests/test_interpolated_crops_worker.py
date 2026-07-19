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


def test_init_pose_backend_yolo_delegates_to_load_pose_backend(
    monkeypatch, tmp_path
) -> None:
    """Golden rule: the YOLO pose branch routes through the shared
    ``core/inference/api.load_pose_backend`` shim (patched here in the worker's
    module) instead of duplicating the runtime-flavor ladder."""
    crops_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.crops_worker"
    )
    pose_utils = importlib.import_module("hydra_suite.core.identity.pose.utils")

    monkeypatch.setattr(
        pose_utils, "load_skeleton_from_json", lambda _p: (["kpt0", "kpt1"], [])
    )

    captured: dict[str, object] = {}

    class FakeBackend:
        output_keypoint_names = ["kpt0", "kpt1"]

        def __init__(self) -> None:
            self.warmup_calls = 0

        def warmup(self) -> None:
            self.warmup_calls += 1
            captured["warmed_up"] = True

    fake_backend = FakeBackend()

    def _fake_load_pose_backend(**kwargs):
        captured.update(kwargs)
        return fake_backend

    monkeypatch.setattr(crops_worker, "load_pose_backend", _fake_load_pose_backend)

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

    assert backend is fake_backend
    assert captured["backend_family"] == "yolo"
    assert captured["model_path"] == "/models/yolo_pose.pt"
    assert captured["compute_runtime"] == "cuda"
    # Regression guard: load_pose_backend (-> load_pose_model) already warms
    # the backend it returns; a second GUI-side warmup() call is redundant
    # and, for the SLEAP service backend, breaks _service_started_here
    # ownership tracking, leaking the service subprocess past close().
    assert fake_backend.warmup_calls == 0
    assert "warmed_up" not in captured
    assert kpt_source_names == ["kpt0", "kpt1"]
    assert kpt_labels


def test_init_pose_backend_sleap_delegates_to_load_pose_backend(
    monkeypatch, tmp_path
) -> None:
    """Golden rule: the SLEAP pose branch routes through the shared
    ``load_pose_backend`` shim and threads SLEAP settings (env, max_instances)
    through -- the tier -> flavor decision lives in ``load_pose_model``, not
    here, so this asserts delegation + settings, not the resolved flavor."""
    crops_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.crops_worker"
    )
    pose_utils = importlib.import_module("hydra_suite.core.identity.pose.utils")

    monkeypatch.setattr(
        pose_utils, "load_skeleton_from_json", lambda _p: (["kpt0"], [(0, 1)])
    )

    captured: dict[str, object] = {}

    class FakeBackend:
        output_keypoint_names = ["kpt0"]

        def __init__(self) -> None:
            self.warmup_calls = 0

        def warmup(self) -> None:
            self.warmup_calls += 1
            captured["warmed_up"] = True

    fake_backend = FakeBackend()

    def _fake_load_pose_backend(**kwargs):
        captured.update(kwargs)
        return fake_backend

    monkeypatch.setattr(crops_worker, "load_pose_backend", _fake_load_pose_backend)

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

    assert backend is fake_backend
    # Regression guard: same double-warmup leak as the YOLO case above, but
    # more severe for SLEAP -- the service backend's warmup() ownership
    # bookkeeping (_service_started_here) is not idempotent across calls, so
    # a second warmup() here leaves the SLEAP service process orphaned after
    # close().
    assert fake_backend.warmup_calls == 0
    assert "warmed_up" not in captured
    assert kpt_source_names == ["kpt0"]

    assert captured["backend_family"] == "sleap"
    assert captured["compute_runtime"] == "cuda"
    assert captured["sleap_env"] == "sleap_env_x"
    assert captured["sleap_max_instances"] == 2
    assert captured["model_path"] == "/models/sleap_model"
    assert captured["out_root"] == str(tmp_path)


def test_crops_worker_has_no_divergent_flavor_ladder() -> None:
    """Source guard: the deleted runtime-flavor ladder must not reappear."""
    from pathlib import Path as _Path

    src = _Path(
        importlib.import_module(
            "hydra_suite.trackerkit.gui.workers.crops_worker"
        ).__file__
    ).read_text(encoding="utf-8")
    for banned in (
        "is_cuda_like",
        "onnx_cuda",
        "create_pose_backend_from_config",
        "YoloNativeBackend",
    ):
        assert banned not in src, f"divergent pose ladder token still present: {banned}"
