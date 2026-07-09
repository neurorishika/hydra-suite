"""Task 6 regression tests: PoseKit's direct pose-backend construction.

``posekit/gui/workers.py``'s ``_build_pose_backend`` replaces the legacy
``build_runtime_config``/``create_pose_backend_from_config`` translation path
(Step 5 of the Task 6 brief), mirroring
``core/inference/stages/pose.py::load_pose_model`` and the pattern already
applied to ``trackerkit/gui/workers/preview_worker.py`` (Task 3) and
``trackerkit/gui/workers/crops_worker.py`` (Task 5).

The critical case under test: Apple GPU-Fast resolves to the canonical
runtime ``"coreml"`` (Task 6's core fix, in ``posekit/gui/runtimes.py``).
These tests confirm that value flows all the way through to backend
construction without being silently collapsed back into an ONNX CoreML-EP
path (the bug this task fixes) for either backend family.
"""

from __future__ import annotations

import importlib


def _workers_module():
    return importlib.import_module("hydra_suite.posekit.gui.workers")


def test_build_pose_backend_yolo_coreml_uses_mps_device(monkeypatch) -> None:
    workers = _workers_module()
    yolo_mod = importlib.import_module("hydra_suite.core.identity.pose.backends.yolo")

    captured: dict[str, object] = {}

    class FakeYoloBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(yolo_mod, "YoloNativeBackend", FakeYoloBackend)

    backend = workers._build_pose_backend(
        backend_family="yolo",
        model_path="/models/yolo_pose.pt",
        exported_model_path="",
        compute_runtime="coreml",
        min_valid_conf=0.0,
        batch_size=4,
        conf=0.25,
        keypoint_names=["a", "b"],
        skeleton_edges=[],
        out_root="/tmp/out",
        sleap_env=None,
        sleap_max_instances=1,
    )

    assert isinstance(backend, FakeYoloBackend)
    assert captured["device"] == "mps"
    assert captured["model_path"] == "/models/yolo_pose.pt"


def test_build_pose_backend_yolo_mps_and_coreml_use_same_device(monkeypatch) -> None:
    """The "gpu" and "gpu_fast" tiers on Apple Silicon both resolve YOLO to
    the mps device — YoloNativeBackend has no separate native-CoreML export
    path, matching ``core/inference/stages/pose.py::load_pose_model``."""
    workers = _workers_module()
    yolo_mod = importlib.import_module("hydra_suite.core.identity.pose.backends.yolo")

    devices: list[str] = []

    class FakeYoloBackend:
        def __init__(self, **kwargs):
            devices.append(kwargs["device"])

    monkeypatch.setattr(yolo_mod, "YoloNativeBackend", FakeYoloBackend)

    for compute_runtime in ("mps", "coreml"):
        workers._build_pose_backend(
            backend_family="yolo",
            model_path="/models/yolo_pose.pt",
            exported_model_path="",
            compute_runtime=compute_runtime,
            min_valid_conf=0.0,
            batch_size=4,
            conf=0.25,
            keypoint_names=[],
            skeleton_edges=[],
            out_root="/tmp/out",
            sleap_env=None,
            sleap_max_instances=1,
        )

    assert devices == ["mps", "mps"]


def test_build_pose_backend_yolo_tensorrt_cuda_uses_cuda_device(monkeypatch) -> None:
    """``_pred_runtime_flavor`` can hand this helper the legacy-shaped
    "tensorrt_cuda" string (from ``derive_pose_runtime_settings``); it must
    still resolve to the CUDA device like the plain "tensorrt"/"cuda" case."""
    workers = _workers_module()
    yolo_mod = importlib.import_module("hydra_suite.core.identity.pose.backends.yolo")

    captured: dict[str, object] = {}

    class FakeYoloBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(yolo_mod, "YoloNativeBackend", FakeYoloBackend)

    workers._build_pose_backend(
        backend_family="yolo",
        model_path="/models/yolo_pose.pt",
        exported_model_path="",
        compute_runtime="tensorrt_cuda",
        min_valid_conf=0.0,
        batch_size=4,
        conf=0.25,
        keypoint_names=[],
        skeleton_edges=[],
        out_root="/tmp/out",
        sleap_env=None,
        sleap_max_instances=1,
    )

    assert captured["device"] == "cuda:0"


def test_build_pose_backend_sleap_coreml_uses_native_flavor(monkeypatch) -> None:
    """SLEAP has no ONNX-CoreML EP support (dynamic-shape failures on its
    UNet), so both "mps" and "coreml" compute_runtime must resolve to the
    native SLEAP (TensorFlow/Metal) runtime flavor, matching
    ``core/inference/stages/pose.py::load_pose_model``'s SLEAP branch."""
    workers = _workers_module()
    pose_api = importlib.import_module("hydra_suite.core.identity.pose.api")
    pose_types = importlib.import_module("hydra_suite.core.identity.pose.types")

    captured: dict[str, object] = {}

    def _fake_create_pose_backend_from_config(config):
        captured["config"] = config
        return object()

    monkeypatch.setattr(
        pose_api,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )
    # workers._build_pose_backend imports create_pose_backend_from_config
    # locally from hydra_suite.core.identity.pose.api — patch the module
    # attribute it will resolve at call time.
    import hydra_suite.core.identity.pose.api as _pose_api_mod

    monkeypatch.setattr(
        _pose_api_mod,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    for compute_runtime in ("mps", "coreml"):
        workers._build_pose_backend(
            backend_family="sleap",
            model_path="/models/sleap_model",
            exported_model_path="",
            compute_runtime=compute_runtime,
            min_valid_conf=0.2,
            batch_size=2,
            conf=0.25,
            keypoint_names=["a", "b"],
            skeleton_edges=[],
            out_root="/tmp/out",
            sleap_env="sleap_env_x",
            sleap_max_instances=1,
        )
        cfg = captured["config"]
        assert isinstance(cfg, pose_types.PoseRuntimeConfig)
        assert cfg.runtime_flavor == "native"
        assert cfg.device == "mps"
        assert cfg.sleap_env == "sleap_env_x"


def test_pred_runtime_flavor_returns_coreml_not_onnx_mps() -> None:
    """Critical-finding regression test: ``MainWindow._pred_runtime_flavor``
    must not route ``"coreml"`` through ``derive_pose_runtime_settings``
    (whose ``_normalize_runtime`` collapses "coreml" -> "onnx_coreml" ->
    "onnx_mps" flavor), or the exact ONNX-CoreML-EP bug Task 6 fixes in
    ``runtimes.py`` reappears one function away for cache-key construction
    and exported-model-path browsing.

    Exercises the unbound method against a minimal fake ``self`` (mirroring
    ``tests/test_posekit_main_window.py``'s ``SimpleNamespace`` pattern)
    rather than constructing a full ``MainWindow``, which requires a live
    project/image-path context this test doesn't need.
    """
    from types import SimpleNamespace

    from hydra_suite.posekit.gui.main_window import MainWindow

    fake_self = SimpleNamespace(
        _selected_compute_runtime=lambda *a, **k: "coreml",
        _pred_backend=lambda: "yolo",
    )
    assert MainWindow._pred_runtime_flavor(fake_self) == "coreml"

    fake_self._pred_backend = lambda: "sleap"
    assert MainWindow._pred_runtime_flavor(fake_self) == "coreml"
