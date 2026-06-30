import tempfile

import pytest

from hydra_suite.core.inference.config import (
    InferenceConfig,
    InferenceConfigError,
    OBBConfig,
    OBBDirectConfig,
)


def _min_cfg(**kw):
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
        ),
        **kw,
    )


def test_pipeline_depth_defaults_to_2():
    cfg = _min_cfg()
    assert cfg.pipeline_depth == 2


def test_pipeline_depth_roundtrips_via_json():
    cfg = _min_cfg(pipeline_depth=4)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        cfg.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.pipeline_depth == 4


def test_pipeline_depth_validation_rejects_zero():
    with pytest.raises(InferenceConfigError, match="pipeline_depth"):
        _min_cfg(pipeline_depth=0)


def test_pipeline_depth_validation_rejects_negative():
    with pytest.raises(InferenceConfigError, match="pipeline_depth"):
        _min_cfg(pipeline_depth=-1)


def test_pipeline_depth_from_json_validates():
    import json

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "obb": {
                        "mode": "direct",
                        "direct": {
                            "model_path": "/m.pt",
                            "compute_runtime": "cpu",
                            "confidence_floor": 0.001,
                            "confidence_threshold": 0.25,
                        },
                    },
                    "pipeline_depth": 0,
                }
            )
        )
        path = f.name
    with pytest.raises(InferenceConfigError, match="pipeline_depth"):
        InferenceConfig.from_json(path)
