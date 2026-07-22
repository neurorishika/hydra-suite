import torch

from hydra_suite.core.identity.pose.api import create_pose_backend_from_config
from hydra_suite.core.identity.pose.backends.vitpose import ViTPoseBackend
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig
from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose


def test_factory_builds_vitpose(tmp_path):
    model = build_vitpose("S", "classic", num_keypoints=3)
    p = tmp_path / "m.pt"
    torch.save(
        {"model_state": model.state_dict(), "variant": "S", "num_keypoints": 3}, p
    )
    cfg = PoseRuntimeConfig(
        backend_family="vitpose",
        runtime_flavor="native",
        device="cpu",
        model_path=str(p),
        keypoint_names=["a", "b", "c"],
        vitpose_batch=2,
    )
    be = create_pose_backend_from_config(cfg)
    assert isinstance(be, ViTPoseBackend)
    assert be.output_keypoint_names == ["a", "b", "c"]
