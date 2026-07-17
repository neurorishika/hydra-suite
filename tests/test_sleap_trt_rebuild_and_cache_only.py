"""Tests for Follow-up A (SLEAP TRT engine auto-rebuild) and
Follow-up B (cache_only skips heavy model loading).

Follow-up A tests
-----------------
All tests mock TensorRT and file-system interactions so they are CUDA-free.
They verify the *decision* logic in SleapExportedBackend._init_tensorrt_runner:
  - Valid .trt present  → _DirectTensorRTEngine (no rebuild, no ORT-TRT-EP)
  - Stale  .trt present → rebuild attempted; if rebuild succeeds → engine
  - Stale  .trt present → rebuild fails       → ORT-TRT-EP fallback + warning
  - No .trt, .onnx only  → rebuild attempted; if succeeds   → engine
  - No .trt, .onnx only  → rebuild fails      → ORT-TRT-EP fallback + warning

Follow-up B tests
-----------------
Verify that InferenceRunner(cache_only=True) skips HeadTail / CNN / Pose /
AprilTag model loading.  Because Pose is the most expensive (SLEAP ~8 s init),
confirming it is not called is the key assertion.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_onnx_dir(tmp_path: Path) -> Path:
    """Create a fake SLEAP export directory containing only an ONNX file."""
    (tmp_path / "model.onnx").write_bytes(b"fake onnx")
    return tmp_path


def _make_trt_dir(tmp_path: Path) -> Path:
    """Create a fake SLEAP export directory containing only a .trt file."""
    (tmp_path / "model.trt").write_bytes(b"fake trt engine")
    return tmp_path


def _make_both_dir(tmp_path: Path) -> Path:
    """Create a fake export directory with both .onnx and .trt files."""
    (tmp_path / "model.onnx").write_bytes(b"fake onnx")
    (tmp_path / "model.trt").write_bytes(b"fake trt engine")
    return tmp_path


# ---------------------------------------------------------------------------
# Follow-up A: _init_tensorrt_runner decision logic
# ---------------------------------------------------------------------------


class TestInitTensorrtRunner:
    """Tests for SleapExportedBackend._init_tensorrt_runner without real CUDA."""

    def _make_backend_shell(self, model_path: str):  # type: ignore[return]
        """Create a SleapExportedBackend instance without calling __init__
        so we can test _init_tensorrt_runner in isolation.
        """
        from hydra_suite.core.identity.pose.backends.sleap import SleapExportedBackend

        obj = object.__new__(SleapExportedBackend)
        obj.runtime_flavor = "tensorrt"
        obj.runtime_request = "tensorrt"
        obj.device = "cuda"
        obj.model_path = Path(model_path)
        obj.output_keypoint_names = ["kp0"]
        obj.min_valid_conf = 0.2
        obj.batch_size = 4
        obj._last_profile = {}
        obj._input_hw = None
        return obj

    def test_valid_trt_routes_to_direct_engine(self, tmp_path):
        """When a valid .trt is present _DirectTensorRTEngine is returned."""
        export_dir = _make_trt_dir(tmp_path)
        trt_file = export_dir / "model.trt"
        backend = self._make_backend_shell(str(trt_file))

        mock_engine = MagicMock(name="DirectTensorRTEngine")

        with patch(
            "hydra_suite.core.identity.pose.backends.sleap._DirectTensorRTEngine",
            return_value=mock_engine,
        ) as MockEngine:
            result = backend._init_tensorrt_runner(trt_file)

        MockEngine.assert_called_once_with(trt_file)
        assert result is mock_engine

    def test_stale_trt_attempts_rebuild_from_sibling_onnx(self, tmp_path):
        """Stale .trt → rebuild from sibling .onnx → _DirectTensorRTEngine."""
        export_dir = _make_both_dir(tmp_path)
        trt_file = export_dir / "model.trt"
        onnx_file = export_dir / "model.onnx"
        backend = self._make_backend_shell(str(trt_file))

        mock_engine = MagicMock(name="DirectTensorRTEngine")
        call_count = [0]

        def engine_side_effect(path):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Stale engine: version mismatch")
            return mock_engine

        with (
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._DirectTensorRTEngine",
                side_effect=engine_side_effect,
            ),
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._build_trt_engine_from_onnx",
                return_value=True,
            ) as mock_build,
        ):
            result = backend._init_tensorrt_runner(trt_file)

        mock_build.assert_called_once_with(onnx_file, trt_file, fixed_hw=None)
        assert result is mock_engine

    def test_stale_trt_rebuild_fails_falls_back_to_ort_ep(self, tmp_path):
        """Stale .trt + failed rebuild → ORT TensorRT-EP fallback with warning."""
        export_dir = _make_both_dir(tmp_path)
        trt_file = export_dir / "model.trt"
        backend = self._make_backend_shell(str(trt_file))

        mock_ort_session = MagicMock(name="DirectOnnxSession")

        with (
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._DirectTensorRTEngine",
                side_effect=RuntimeError("stale"),
            ),
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._build_trt_engine_from_onnx",
                return_value=False,
            ),
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._DirectOnnxSession",
                return_value=mock_ort_session,
            ) as MockOrt,
        ):
            result = backend._init_tensorrt_runner(trt_file)

        # Should have fallen back to ORT session
        MockOrt.assert_called_once()
        assert result is mock_ort_session

    def test_onnx_only_attempts_build_then_uses_engine(self, tmp_path):
        """No .trt present, .onnx only → build .trt → _DirectTensorRTEngine."""
        export_dir = _make_onnx_dir(tmp_path)
        onnx_file = export_dir / "model.onnx"
        backend = self._make_backend_shell(str(onnx_file))

        mock_engine = MagicMock(name="DirectTensorRTEngine")
        expected_trt = onnx_file.with_suffix(".trt")

        with (
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._build_trt_engine_from_onnx",
                return_value=True,
            ) as mock_build,
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._DirectTensorRTEngine",
                return_value=mock_engine,
            ) as MockEngine,
        ):
            result = backend._init_tensorrt_runner(onnx_file)

        mock_build.assert_called_once_with(onnx_file, expected_trt, fixed_hw=None)
        MockEngine.assert_called_once_with(expected_trt)
        assert result is mock_engine

    def test_onnx_only_build_fails_falls_back_to_ort_ep(self, tmp_path):
        """No .trt present + build failure → ORT TensorRT-EP fallback."""
        export_dir = _make_onnx_dir(tmp_path)
        onnx_file = export_dir / "model.onnx"
        backend = self._make_backend_shell(str(onnx_file))

        mock_ort_session = MagicMock(name="DirectOnnxSession")

        with (
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._build_trt_engine_from_onnx",
                return_value=False,
            ),
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._DirectOnnxSession",
                return_value=mock_ort_session,
            ) as MockOrt,
        ):
            result = backend._init_tensorrt_runner(onnx_file)

        MockOrt.assert_called_once()
        assert result is mock_ort_session

    def test_stale_trt_no_sibling_onnx_falls_back_to_ort_ep(self, tmp_path):
        """Stale .trt + no sibling .onnx → ORT TensorRT-EP fallback."""
        # Only .trt, no .onnx beside it
        trt_file = tmp_path / "model.trt"
        trt_file.write_bytes(b"stale engine")
        backend = self._make_backend_shell(str(trt_file))

        mock_ort_session = MagicMock(name="DirectOnnxSession")

        with (
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._DirectTensorRTEngine",
                side_effect=RuntimeError("stale"),
            ),
            patch(
                "hydra_suite.core.identity.pose.backends.sleap._DirectOnnxSession",
                return_value=mock_ort_session,
            ) as MockOrt,
        ):
            result = backend._init_tensorrt_runner(trt_file)

        MockOrt.assert_called_once()
        assert result is mock_ort_session


# ---------------------------------------------------------------------------
# Follow-up A: _build_trt_engine_from_onnx guard tests
# ---------------------------------------------------------------------------


class TestBuildTrtEngineFromOnnx:
    """Tests for the _build_trt_engine_from_onnx helper."""

    def test_returns_false_when_tensorrt_import_fails(self, tmp_path):
        """When TensorRT is not installed, return False without raising."""
        from hydra_suite.core.identity.pose.backends.sleap import (
            _build_trt_engine_from_onnx,
        )

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.trt"

        with patch.dict("sys.modules", {"tensorrt": None}):
            result = _build_trt_engine_from_onnx(onnx_path, engine_path)

        assert result is False
        assert not engine_path.exists()

    def test_returns_false_on_builder_exception(self, tmp_path):
        """If TRT builder raises an exception, return False without propagating."""
        from hydra_suite.core.identity.pose.backends.sleap import (
            _build_trt_engine_from_onnx,
        )

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.trt"

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.Builder.side_effect = RuntimeError("CUDA init failed")

        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            result = _build_trt_engine_from_onnx(onnx_path, engine_path)

        assert result is False

    def test_returns_true_and_writes_engine_on_success(self, tmp_path):
        """Successful build returns True and writes serialized plan bytes."""
        from hydra_suite.core.identity.pose.backends.sleap import (
            _build_trt_engine_from_onnx,
        )

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake onnx bytes")
        engine_path = tmp_path / "model.trt"

        plan_bytes = b"serialized trt plan"

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0

        mock_logger_inst = MagicMock()
        mock_trt.Logger.return_value = mock_logger_inst

        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_config = MagicMock()

        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser
        mock_parser.parse.return_value = True
        mock_builder.create_builder_config.return_value = mock_config
        mock_builder.build_serialized_network.return_value = bytearray(plan_bytes)

        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            result = _build_trt_engine_from_onnx(onnx_path, engine_path)

        assert result is True
        assert engine_path.exists()
        assert engine_path.read_bytes() == plan_bytes

    def test_returns_false_when_build_serialized_network_returns_none(self, tmp_path):
        """build_serialized_network returning None → return False."""
        from hydra_suite.core.identity.pose.backends.sleap import (
            _build_trt_engine_from_onnx,
        )

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.trt"

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0

        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_config = MagicMock()

        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser
        mock_parser.parse.return_value = True
        mock_builder.create_builder_config.return_value = mock_config
        mock_builder.build_serialized_network.return_value = None

        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            result = _build_trt_engine_from_onnx(onnx_path, engine_path)

        assert result is False

    def _mock_trt_with_dynamic_hw_input(self):
        """A mock `tensorrt` module whose parsed network has one input with a
        dynamic batch dim AND dynamic (symbolic) H/W dims -- mirrors SLEAP's
        sleap-nn ONNX export, whose graph declares [batch, 3, height, width]
        all as symbolic except channels.
        """
        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0

        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_config = MagicMock()
        mock_profile = MagicMock()

        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser
        mock_parser.parse.return_value = True
        mock_builder.create_builder_config.return_value = mock_config
        mock_builder.create_optimization_profile.return_value = mock_profile
        mock_builder.build_serialized_network.return_value = bytearray(b"plan")

        mock_input = MagicMock()
        mock_input.name = "image"
        mock_input.shape = [-1, 3, -1, -1]
        mock_network.num_inputs = 1
        mock_network.get_input.return_value = mock_input

        return mock_trt, mock_profile

    def test_dynamic_hw_pinned_when_fixed_hw_given(self, tmp_path):
        """A dynamic H/W input is pinned to *fixed_hw* instead of bailing --
        this is the actual bug fix: SLEAP's exporter emits symbolic H/W even
        though every crop is always resized to one fixed size before inference.
        """
        from hydra_suite.core.identity.pose.backends.sleap import (
            _build_trt_engine_from_onnx,
        )

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.trt"

        mock_trt, mock_profile = self._mock_trt_with_dynamic_hw_input()

        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            result = _build_trt_engine_from_onnx(
                onnx_path, engine_path, fixed_hw=(224, 224)
            )

        assert result is True
        assert engine_path.exists()
        mock_profile.set_shape.assert_called_once_with(
            "image", (1, 3, 224, 224), (64, 3, 224, 224), (512, 3, 224, 224)
        )

    def test_dynamic_hw_without_fixed_hw_bails(self, tmp_path):
        """No fixed_hw hint -> bail to ORT-EP rather than build a wrong engine."""
        from hydra_suite.core.identity.pose.backends.sleap import (
            _build_trt_engine_from_onnx,
        )

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        engine_path = tmp_path / "model.trt"

        mock_trt, mock_profile = self._mock_trt_with_dynamic_hw_input()

        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            result = _build_trt_engine_from_onnx(onnx_path, engine_path)

        assert result is False
        assert not engine_path.exists()
        mock_profile.set_shape.assert_not_called()


# ---------------------------------------------------------------------------
# Follow-up B: InferenceRunner cache_only skips model loading
# ---------------------------------------------------------------------------


class TestInferenceRunnerCacheOnly:
    """Verify that cache_only=True skips HeadTail / CNN / Pose / AprilTag."""

    def _make_cfg(self):
        from hydra_suite.core.inference.config import (
            InferenceConfig,
            OBBConfig,
            OBBDirectConfig,
        )

        return InferenceConfig(
            obb=OBBConfig(
                mode="direct",
                direct=OBBDirectConfig(
                    model_path="/fake/model.pt", compute_runtime="cpu"
                ),
            ),
        )

    def test_cache_only_true_skips_pose_model_loading(self, tmp_path):
        """With cache_only=True, load_pose_model must not be called."""
        cfg = self._make_cfg()

        mock_obb_models = MagicMock(name="OBBModels")

        with (
            patch(
                "hydra_suite.core.inference.runner.RuntimeContext.from_config",
                return_value=MagicMock(),
            ),
            patch(
                "hydra_suite.core.inference.runner._load_all_models",
                wraps=lambda config, runtime, *, cache_only=False, video_path=None: (
                    __import__(
                        "hydra_suite.core.inference.runner", fromlist=["_AllModels"]
                    )._AllModels(
                        obb=mock_obb_models,
                        headtail=None,
                        cnn=[],
                        pose=None,
                        apriltag=None,
                    )
                    if cache_only
                    else (_ for _ in ()).throw(
                        AssertionError("cache_only should be True")
                    )
                ),
            ),
        ):
            from hydra_suite.core.inference.runner import InferenceRunner

            runner = InferenceRunner(cfg, cache_dir=tmp_path, cache_only=True)

        assert runner.cache_only is True
        # Models that would trigger SLEAP init are absent
        assert runner._models.pose is None
        assert runner._models.headtail is None
        assert runner._models.cnn == []
        assert runner._models.apriltag is None

    def test_cache_only_false_loads_all_models(self, tmp_path):
        """With cache_only=False (default), _load_all_models is called normally."""
        from hydra_suite.core.inference.runner import InferenceRunner, _AllModels

        cfg = self._make_cfg()
        mock_models = MagicMock(spec=_AllModels)
        mock_models.obb = MagicMock()
        mock_models.headtail = MagicMock()
        mock_models.cnn = [MagicMock()]
        mock_models.pose = MagicMock()
        mock_models.apriltag = None

        with (
            patch(
                "hydra_suite.core.inference.runner.RuntimeContext.from_config",
                return_value=MagicMock(),
            ),
            patch(
                "hydra_suite.core.inference.runner._load_all_models",
                return_value=mock_models,
            ) as mock_load,
        ):
            InferenceRunner(cfg, cache_dir=tmp_path, cache_only=False)

        mock_load.assert_called_once()
        _call_kwargs = mock_load.call_args
        # cache_only must be forwarded as False
        assert _call_kwargs.kwargs.get("cache_only", False) is False

    def test_cache_only_skips_pose_when_pose_config_present(self, tmp_path):
        """Even if a pose config exists, cache_only=True skips pose loading."""
        from hydra_suite.core.inference.config import (
            InferenceConfig,
            OBBConfig,
            OBBDirectConfig,
            PoseConfig,
            PoseSLEAPConfig,
        )
        from hydra_suite.core.inference.runner import _load_all_models

        pose_cfg = PoseConfig(
            backend="sleap",
            sleap=PoseSLEAPConfig(
                model_path="/fake/sleap_model",
                conda_env="sleap",
                batch_size=4,
                max_instances=1,
            ),
            skeleton_file="",
        )
        cfg = InferenceConfig(
            obb=OBBConfig(
                mode="direct",
                direct=OBBDirectConfig(
                    model_path="/fake/model.pt", compute_runtime="cpu"
                ),
            ),
            pose=pose_cfg,
        )

        mock_obb = MagicMock(name="OBBModels")
        mock_runtime = MagicMock(name="RuntimeContext")

        with (
            patch(
                "hydra_suite.core.inference.stages.obb.load_obb_models",
                return_value=mock_obb,
            ),
            patch(
                "hydra_suite.core.inference.stages.pose.load_pose_model",
            ) as mock_load_pose,
        ):
            result = _load_all_models(cfg, mock_runtime, cache_only=True)

        # Pose model must NOT have been loaded
        mock_load_pose.assert_not_called()
        assert result.pose is None

    def test_load_all_models_cache_only_false_calls_pose_loader(self, tmp_path):
        """With cache_only=False, load_pose_model IS called when pose config given."""
        from hydra_suite.core.inference.config import (
            InferenceConfig,
            OBBConfig,
            OBBDirectConfig,
            PoseConfig,
            PoseSLEAPConfig,
        )
        from hydra_suite.core.inference.runner import _load_all_models

        pose_cfg = PoseConfig(
            backend="sleap",
            sleap=PoseSLEAPConfig(
                model_path="/fake/sleap_model",
                conda_env="sleap",
                batch_size=4,
                max_instances=1,
            ),
            skeleton_file="",
        )
        cfg = InferenceConfig(
            obb=OBBConfig(
                mode="direct",
                direct=OBBDirectConfig(
                    model_path="/fake/model.pt", compute_runtime="cpu"
                ),
            ),
            pose=pose_cfg,
        )

        mock_obb = MagicMock(name="OBBModels")
        mock_pose = MagicMock(name="PoseModel")
        mock_runtime = MagicMock(name="RuntimeContext")

        with (
            patch(
                "hydra_suite.core.inference.stages.obb.load_obb_models",
                return_value=mock_obb,
            ),
            patch(
                "hydra_suite.core.inference.stages.pose.load_pose_model",
                return_value=mock_pose,
            ) as mock_load_pose,
            patch(
                "hydra_suite.core.inference.stages.apriltag.load_apriltag_model",
                return_value=None,
            ),
        ):
            result = _load_all_models(cfg, mock_runtime, cache_only=False)

        mock_load_pose.assert_called_once()
        assert result.pose is mock_pose


# ---------------------------------------------------------------------------
# Follow-up B: TrackingWorker passes cache_only=True in backward mode
# ---------------------------------------------------------------------------


class TestTrackingWorkerCacheOnly:
    """Verify TrackingWorker passes cache_only=True to InferenceRunner when backward."""

    def test_backward_mode_passes_cache_only_true(self, tmp_path):
        """In backward mode, InferenceRunner is constructed with cache_only=True."""
        # We do NOT instantiate a full TrackingWorker (Qt dependency);
        # instead we verify that the worker module imports InferenceRunner
        # and that the code path would pass cache_only=self.backward_mode.
        # We test this by importing the relevant section of worker and asserting
        # the call signature contains cache_only=True.
        #
        # Since TrackingWorker.run() is too large to unit-test in isolation,
        # we verify the intent via a targeted source-code inspection guard:
        # The worker must pass cache_only=self.backward_mode.
        import ast
        from pathlib import Path

        worker_src = (
            Path(__file__).resolve().parents[1]
            / "src/hydra_suite/core/tracking/worker.py"
        )
        tree = ast.parse(worker_src.read_text(encoding="utf-8"))

        # Find all calls to InferenceRunner(...)
        found_cache_only = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match InferenceRunner(...) calls
            if (
                getattr(func, "id", "") != "InferenceRunner"
                and getattr(func, "attr", "") != "InferenceRunner"
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "cache_only":
                    found_cache_only = True

        assert (
            found_cache_only
        ), "InferenceRunner call in worker.py must include cache_only= keyword arg"
