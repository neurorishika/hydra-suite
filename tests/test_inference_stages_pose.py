import json
from unittest.mock import MagicMock

import numpy as np
import torch

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.config import (
    PoseConfig,
    PoseSLEAPConfig,
    PoseYOLOConfig,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.pose import (
    PoseModel,
    load_pose_model,
    run_pose,
    run_pose_batch,
)

_TEST_GEOMETRY = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)


def _cpu_rt():
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )


def _cuda_gpu_rt():
    """gpu tier on a CUDA host: native torch, tensors stay on-device."""
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=True,
        tensor_on_cuda=True,
        requested_gpu=True,
    )


def _cuda_gpu_fast_rt():
    """gpu_fast tier on a CUDA host: TensorRT engines, CPU numpy outputs."""
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=True,
        tensor_on_cuda=False,
        requested_gpu=True,
    )


def _obb(n: int) -> OBBResult:
    return OBBResult(
        frame_idx=0,
        centroids=np.array([[100.0, 100.0]] * n, dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.ones(n, dtype=np.float32) * 400.0,
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32),
        corners=np.array(
            [[[80, 90], [120, 90], [120, 110], [80, 110]]] * n, dtype=np.float32
        ),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _mock_pose_result(n_kpts: int = 4, conf: float = 0.8):
    """Real backend result: pose.types.PoseResult with `.keypoints` as (K, 3) numpy.

    Both the YOLO and SLEAP backends return this canonical type from
    predict_batch (NOT an ultralytics Results object).
    """
    from hydra_suite.core.individual.pose.types import PoseResult as BackendPoseResult

    kpts = np.zeros((n_kpts, 3), dtype=np.float32)
    kpts[:, 2] = conf
    return BackendPoseResult(
        keypoints=kpts,
        mean_conf=conf,
        valid_fraction=1.0,
        num_valid=n_kpts,
        num_keypoints=n_kpts,
    )


def test_run_pose_empty_crops():
    config = PoseConfig(yolo=PoseYOLOConfig(model_path="/p.pt"))
    model = PoseModel(
        backend=MagicMock(), n_keypoints=4, keypoint_names=["a", "b", "c", "d"]
    )
    crops = torch.zeros((0, 3, 64, 64))
    result = run_pose(crops, _obb(0), model, config, _cpu_rt(), geometry=_TEST_GEOMETRY)
    assert result.keypoints.shape == (0, 4, 3)
    assert result.valid_mask.shape == (0,)


def test_run_pose_shape():
    config = PoseConfig(
        yolo=PoseYOLOConfig(model_path="/p.pt"),
        min_keypoint_confidence=0.5,
    )
    mock_backend = MagicMock()
    mock_backend.predict_batch.return_value = [
        _mock_pose_result(4, conf=0.8),
        _mock_pose_result(4, conf=0.8),
    ]
    model = PoseModel(
        backend=mock_backend, n_keypoints=4, keypoint_names=["a", "b", "c", "d"]
    )
    crops = torch.zeros((2, 3, 64, 64))
    result = run_pose(crops, _obb(2), model, config, _cpu_rt(), geometry=_TEST_GEOMETRY)
    assert result.keypoints.shape == (2, 4, 3)
    assert result.valid_mask.shape == (2,)


def test_load_pose_model_reads_canonical_skeleton_keys(tmp_path, monkeypatch):
    """Regression: load_pose_model must read 'keypoint_names'/'skeleton_edges'
    (canonical skeleton JSON keys), not the wrong 'keypoints'/'edges' keys.
    Uses the YOLO branch with a stub backend to avoid loading a real model.
    """
    skel = tmp_path / "skel.json"
    skel.write_text(
        json.dumps(
            {
                "keypoint_names": ["head", "thorax", "abdomen"],
                "skeleton_edges": [[0, 1], [1, 2]],
            }
        )
    )

    import hydra_suite.core.individual.pose.backends.yolo as yolo_mod

    captured = {}

    class _StubBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(yolo_mod, "YoloNativeBackend", _StubBackend)

    config = PoseConfig(
        backend="yolo",
        yolo=PoseYOLOConfig(model_path="/p.pt"),
        skeleton_file=str(skel),
    )
    model = load_pose_model(config, _cpu_rt())

    assert model.keypoint_names == ["head", "thorax", "abdomen"]
    assert model.n_keypoints == 3
    assert captured["keypoint_names"] == ["head", "thorax", "abdomen"]


def test_load_pose_model_sleap_gpu_tier_uses_native_cuda(tmp_path, monkeypatch):
    """gpu tier must run SLEAP's native (non-exported) model on CUDA via the
    service backend -- matching every other stage's "gpu = native torch"
    semantics -- instead of silently reaching for the gpu_fast (onnx) path.
    """
    import hydra_suite.core.individual.pose.api as api_mod

    captured = {}

    def _fake_create_pose_backend_from_config(config):
        captured["runtime_flavor"] = config.runtime_flavor
        captured["device"] = config.device
        return MagicMock()

    monkeypatch.setattr(
        api_mod,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    config = PoseConfig(
        backend="sleap",
        sleap=PoseSLEAPConfig(model_path="/fake/sleap_model_dir"),
    )
    load_pose_model(config, _cuda_gpu_rt())

    assert captured["runtime_flavor"] == "native"
    assert captured["device"] == "cuda"


def test_load_pose_model_sleap_gpu_fast_tier_uses_tensorrt(tmp_path, monkeypatch):
    """gpu_fast tier keeps using the exported (TensorRT) SLEAP backend."""
    import hydra_suite.core.individual.pose.api as api_mod

    captured = {}

    def _fake_create_pose_backend_from_config(config):
        captured["runtime_flavor"] = config.runtime_flavor
        captured["device"] = config.device
        return MagicMock()

    monkeypatch.setattr(
        api_mod,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    config = PoseConfig(
        backend="sleap",
        sleap=PoseSLEAPConfig(model_path="/fake/sleap_model_dir"),
    )
    load_pose_model(config, _cuda_gpu_fast_rt())

    assert captured["runtime_flavor"] == "tensorrt"
    assert captured["device"] == "cuda"


def test_run_pose_batch_composes_layer2_fit_with_layer1_affine_for_backprojection():
    """Numeric back-projection composition check.

    Ported from the deleted `_flush_pose_batch`-era test in
    `test_interpolated_crops_worker_layer2_fit.py` (removed when Task 12
    inlined interpolated-crops pose inference onto this module's
    `run_pose_batch`, which now owns the composition being verified): a
    keypoint at the exact Layer-2-fitted location of the canonical canvas's
    centre must round-trip back to that centre through `run_pose_batch`'s
    composed Layer 1 (OBB -> canvas) . Layer 2 (canvas -> model input)
    inverse affine, proving the two transforms are composed correctly before
    inverting rather than applied/ignored inconsistently.
    """
    from hydra_suite.core.canonicalization.fit import fit_affine, fit_to_model_input
    from hydra_suite.core.individual.pose.types import PoseResult as BackendPoseResult
    from hydra_suite.core.inference.result import CropBatch

    geometry = _TEST_GEOMETRY
    canvas_w, canvas_h = geometry.canvas_wh
    model_wh = (96, 64)  # non-square, and != the canonical canvas size

    fit = fit_to_model_input(geometry.canvas_wh, model_wh)
    fit_m = fit_affine(fit)
    canvas_center = np.array([canvas_w / 2.0, canvas_h / 2.0, 1.0], dtype=np.float64)
    model_space_center = fit_m @ canvas_center  # where Layer 2 puts the canvas centre

    # OBB corners centred on the canvas centre with angle 0 -> canonical_affine
    # for this detection is the identity 2x3 (Layer 1 is a no-op), isolating
    # the Layer-2 (fit_m) composition under test.
    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    major, minor = canvas_w * 0.5, canvas_h * 0.5
    corners = np.array(
        [
            [cx - major / 2, cy - minor / 2],
            [cx + major / 2, cy - minor / 2],
            [cx + major / 2, cy + minor / 2],
            [cx - major / 2, cy + minor / 2],
        ],
        dtype=np.float32,
    )
    obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[cx, cy]], dtype=np.float32),
        angles=np.zeros(1, dtype=np.float32),
        sizes=np.array([major * minor], dtype=np.float32),
        shapes=np.array([[major, minor]], dtype=np.float32),
        confidences=np.ones(1, dtype=np.float32),
        corners=corners[None, ...],
        detection_ids=OBBResult.make_detection_ids(0, 1),
    )

    class FakePoseBackend:
        preferred_input_size = 0  # signals "has a preferred_input_wh instead"

        @property
        def preferred_input_wh(self):
            return model_wh

        def predict_batch(self, crops):
            kpts = np.array(
                [[model_space_center[0], model_space_center[1], 1.0]],
                dtype=np.float32,
            )
            return [
                BackendPoseResult(
                    keypoints=kpts,
                    mean_conf=1.0,
                    valid_fraction=1.0,
                    num_valid=1,
                    num_keypoints=1,
                )
            ]

    model = PoseModel(backend=FakePoseBackend(), n_keypoints=1, keypoint_names=["k0"])
    config = PoseConfig(yolo=PoseYOLOConfig(model_path="/p.pt"))
    batch = CropBatch(
        crops=torch.zeros((1, 3, canvas_h, canvas_w)),
        detection_ids=OBBResult.make_detection_ids(0, 1),
        frame_index=np.array([0], dtype=np.int64),
        obb_by_frame={0: obb},
        native_sizes=np.array([[canvas_h, canvas_w]], dtype=np.int64),
    )

    results = run_pose_batch(batch, model, config, _cpu_rt(), geometry=geometry)

    kpt = results[0].keypoints[0, 0]
    assert abs(float(kpt[0]) - cx) < 1e-3
    assert abs(float(kpt[1]) - cy) < 1e-3


def test_run_pose_valid_mask_high_conf():
    config = PoseConfig(
        yolo=PoseYOLOConfig(model_path="/p.pt"),
        min_keypoint_confidence=0.5,
        min_valid_keypoints=2,
    )
    mock_backend = MagicMock()
    r0 = _mock_pose_result(4, conf=0.9)
    r1 = _mock_pose_result(4, conf=0.1)
    mock_backend.predict_batch.return_value = [r0, r1]
    model = PoseModel(backend=mock_backend, n_keypoints=4, keypoint_names=list("abcd"))
    crops = torch.zeros((2, 3, 64, 64))
    result = run_pose(crops, _obb(2), model, config, _cpu_rt(), geometry=_TEST_GEOMETRY)
    assert bool(result.valid_mask[0]) is True
    assert bool(result.valid_mask[1]) is False


def test_pose_model_close_closes_underlying_backend():
    """Regression: PoseModel.close() used to be a no-op; for the SLEAP
    service backend this is what actually reaches shutdown_sleap_service()."""
    # spec=["close"]: real backends here (YoloNativeBackend/SleapExportedBackend)
    # expose close() only, not release() -- pin the fake to match so the test
    # actually exercises the close() branch of close_backend_resource().
    mock_backend = MagicMock(spec=["close"])
    model = PoseModel(backend=mock_backend, n_keypoints=4, keypoint_names=list("abcd"))
    model.close()
    mock_backend.close.assert_called_once()
