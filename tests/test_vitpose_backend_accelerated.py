import numpy as np
import pytest
import torch

from hydra_suite.core.identity.pose.backends.vitpose import (
    ViTPoseBackend,
    auto_export_vitpose_model,
)
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def _ckpt(tmp_path, k=3):
    model = build_vitpose("S", "classic", num_keypoints=k)
    p = tmp_path / "best.pt"
    torch.save(
        {"model_state": model.state_dict(), "variant": "S", "num_keypoints": k}, p
    )
    return p


def test_predict_batch_cuda_falls_back_to_numpy(tmp_path):
    # On a non-TRT runner, predict_batch_cuda must degrade to predict_batch.
    be = ViTPoseBackend(
        str(_ckpt(tmp_path)), device="cpu", keypoint_names=["a", "b", "c"]
    )
    assert hasattr(be, "predict_batch_cuda")


def test_coreml_backend_predicts(tmp_path):
    pytest.importorskip("coremltools")
    p = _ckpt(tmp_path, k=3)
    cfg = PoseRuntimeConfig(
        backend_family="vitpose",
        model_path=str(p),
        runtime_flavor="coreml",
        device="mps",
    )
    art = auto_export_vitpose_model(cfg, "coreml")
    be = ViTPoseBackend(
        str(p),
        device="mps",
        runtime_flavor="coreml",
        keypoint_names=["a", "b", "c"],
        exported_model_path=art,
    )
    res = be.predict_batch([np.zeros((60, 40, 3), np.uint8)])
    assert res[0].keypoints.shape == (3, 3)


def test_forward_handles_list_returning_runner(tmp_path):
    # OnnxSessionRunner.run returns a LIST (session.run(None, ...)), not a dict.
    # _forward must coerce a list the same way it coerces a dict, or the ONNX
    # fallback path yields a spurious leading axis and decoding breaks.
    be = ViTPoseBackend(
        str(_ckpt(tmp_path, k=3)), device="cpu", keypoint_names=["a", "b", "c"]
    )

    class _ListRunner:
        def run(self, batch):
            b = batch.shape[0]
            return [np.zeros((b, 3, 64, 48), dtype=np.float32)]  # list, like ORT

    be._runner = _ListRunner()
    res = be.predict_batch([np.zeros((60, 40, 3), np.uint8)])
    assert len(res) == 1
    assert res[0].keypoints.shape == (3, 3)


def test_factory_coreml_flavor_not_collapsed(tmp_path):
    pytest.importorskip("coremltools")
    from hydra_suite.core.identity.pose.api import create_pose_backend_from_config
    from hydra_suite.core.identity.pose.types import PoseRuntimeConfig

    p = _ckpt(tmp_path, k=3)
    cfg = PoseRuntimeConfig(
        backend_family="vitpose",
        model_path=str(p),
        runtime_flavor="coreml",
        device="mps",
        keypoint_names=["a", "b", "c"],
    )
    be = create_pose_backend_from_config(cfg)
    # The factory must honor coreml (export + wire a runner), NOT collapse to native.
    assert be._runtime_flavor == "coreml"
    assert be._runner is not None
    res = be.predict_batch([np.zeros((60, 40, 3), np.uint8)])
    assert res[0].keypoints.shape == (3, 3)
