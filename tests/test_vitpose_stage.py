from hydra_suite.runtime.resolver import STAGES, PlatformInfo, RuntimeResolver


def test_vitpose_pose_stage_registered():
    assert "vitpose_pose" in STAGES


def test_resolver_vitpose_pose_native_on_cpu():
    r = RuntimeResolver("cpu", PlatformInfo(has_cuda=False, has_mps=False))
    resolved = r.resolve("vitpose_pose")
    assert resolved.backend == "torch"
    assert resolved.device == "cpu"


def test_resolver_vitpose_pose_gpu_fast_cuda():
    r = RuntimeResolver("gpu_fast", PlatformInfo(has_cuda=True, has_mps=False))
    resolved = r.resolve("vitpose_pose", artifact_available=lambda: True)
    assert resolved.backend == "tensorrt"
    assert resolved.device == "cuda"


def test_load_pose_model_vitpose_native_cpu(tmp_path):
    import torch

    from hydra_suite.core.identity.pose.backends.vitpose import ViTPoseBackend
    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose
    from hydra_suite.core.inference.config import PoseConfig, PoseViTPoseConfig
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.pose import load_pose_model
    from hydra_suite.runtime.resolver import ResolvedBackend

    model = build_vitpose("S", "classic", num_keypoints=3)
    p = tmp_path / "best.pt"
    torch.save(
        {"model_state": model.state_dict(), "variant": "S", "num_keypoints": 3}, p
    )
    cfg = PoseConfig(backend="vitpose", vitpose=PoseViTPoseConfig(model_path=str(p)))
    runtime = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        resolved=ResolvedBackend("torch", "cpu", False),
    )
    pm = load_pose_model(cfg, runtime, keypoint_names=["a", "b", "c"])
    assert isinstance(pm.backend, ViTPoseBackend)
    assert pm.n_keypoints == 3
