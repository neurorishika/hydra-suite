from hydra_suite.core.inference.cache.keys import pose_cache_key
from hydra_suite.core.inference.config import PoseConfig, PoseViTPoseConfig


def test_posevitposeconfig_roundtrip():
    cfg = PoseConfig(
        backend="vitpose",
        vitpose=PoseViTPoseConfig(
            model_path="/tmp/best.pt", variant="B", num_keypoints=6, batch_size=8
        ),
    )
    assert cfg.backend == "vitpose"
    assert cfg.vitpose.model_path == "/tmp/best.pt"


def test_cache_key_vitpose_branch(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    cfg = PoseConfig(
        backend="vitpose",
        vitpose=PoseViTPoseConfig(model_path=str(p)),
    )
    key = pose_cache_key(cfg)
    assert key.model_path == str(p)


def test_build_from_params_vitpose(tmp_path):
    from hydra_suite.core.inference.config import build_inference_config_from_params

    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    cfg = build_inference_config_from_params(
        {
            "ENABLE_POSE_EXTRACTOR": True,
            "POSE_MODEL_TYPE": "vitpose",
            "POSE_VITPOSE_MODEL_PATH": str(p),
            "POSE_BATCH_SIZE": 8,
        }
    )
    assert cfg.pose is not None
    assert cfg.pose.backend == "vitpose"
    assert cfg.pose.vitpose.model_path == str(p)
    assert cfg.pose.vitpose.batch_size == 8
