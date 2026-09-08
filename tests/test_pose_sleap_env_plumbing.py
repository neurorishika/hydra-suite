"""POSE_SLEAP_ENV must select the conda env the SLEAP service is spawned under.

Regression guard for a bug where ``build_inference_config_from_params`` built
``PoseSLEAPConfig`` without ``conda_env``, so the dataclass default ``"sleap"``
always won and a user-configured env was silently ignored.

Asserting the key is present in the params dict proves nothing; these tests
follow the value all the way to the ``conda run -n <env>`` argv.
"""

import pytest

from hydra_suite.core.inference.config import build_inference_config_from_params


def _params(tmp_path, env_value):
    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir(exist_ok=True)
    return {
        "RUNTIME_TIER": "cpu",
        "ENABLE_POSE_EXTRACTOR": True,
        "POSE_MODEL_TYPE": "sleap",
        "POSE_SLEAP_MODEL_DIR": str(model_dir),
        "POSE_SLEAP_ENV": env_value,
        "POSE_BATCH_SIZE": 4,
    }


@pytest.mark.parametrize("env_value", ["sleap-nn", "sleap"])
def test_config_carries_configured_sleap_env(tmp_path, env_value):
    cfg = build_inference_config_from_params(_params(tmp_path, env_value))
    assert cfg.pose is not None and cfg.pose.sleap is not None
    assert cfg.pose.sleap.conda_env == env_value


def test_blank_sleap_env_falls_back_to_default(tmp_path):
    cfg = build_inference_config_from_params(_params(tmp_path, "  "))
    assert cfg.pose.sleap.conda_env == "sleap"


@pytest.mark.parametrize("env_value", ["sleap-nn", "sleap"])
def test_configured_env_reaches_the_conda_run_spawn(tmp_path, monkeypatch, env_value):
    """End-to-end: params -> InferenceConfig -> backend -> `conda run -n <env>`."""
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages import pose as pose_stage
    from hydra_suite.integrations.sleap import service as sleap_service

    spawned = []

    def _fake_popen(cmd, **kwargs):
        spawned.append(list(cmd))
        raise RuntimeError("spawn intercepted by test")

    monkeypatch.setattr(sleap_service, "popen_conda", _fake_popen)
    monkeypatch.setattr(sleap_service.shutil, "which", lambda name: "/usr/bin/conda")
    monkeypatch.setattr(sleap_service, "_sleap_env_preflight", lambda env: (True, ""))
    # Fresh service singleton so a previous test's state can't short-circuit start().
    monkeypatch.setattr(
        sleap_service, "_SLEAP_SERVICE", sleap_service._SleapHttpService()
    )

    cfg = build_inference_config_from_params(_params(tmp_path, env_value))
    runtime = RuntimeContext.from_config(cfg)

    # warmup() swallows the intercepted spawn; the argv is what we assert on.
    pose_stage.load_pose_model(
        cfg.pose,
        runtime,
        keypoint_names=["head", "tail"],
        skeleton_edges=[(0, 1)],
        out_root=str(tmp_path),
    )

    assert spawned, "SLEAP service was never spawned"
    argv = spawned[0]
    assert argv[:3] == ["conda", "run", "-n"], argv
    assert argv[3] == env_value, argv
