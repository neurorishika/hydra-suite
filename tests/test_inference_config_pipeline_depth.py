import tempfile

import pytest

from hydra_suite.core.inference.config import (
    MAX_DETECTION_BATCH_SIZE,
    MAX_PIPELINE_DEPTH,
    InferenceConfig,
    InferenceConfigError,
    OBBConfig,
    OBBDirectConfig,
    build_inference_config_from_params,
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


def test_pipeline_depth_is_built_from_tracker_params():
    cfg = build_inference_config_from_params(
        {
            "YOLO_OBB_DIRECT_MODEL_PATH": "/m.pt",
            "PIPELINE_DEPTH": 1,
        }
    )
    assert cfg.pipeline_depth == 1


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pipeline_depth", MAX_PIPELINE_DEPTH + 1),
        ("detection_batch_size", 0),
        ("detection_batch_size", MAX_DETECTION_BATCH_SIZE + 1),
    ],
)
def test_pipeline_retention_bounds_reject_adversarial_direct_configs(field, value):
    with pytest.raises(InferenceConfigError, match=field):
        _min_cfg(**{field: value})


def test_loaded_config_cannot_bypass_detection_batch_bound(tmp_path):
    cfg = _min_cfg()
    path = tmp_path / "oversized.json"
    cfg.to_json(path)
    payload = path.read_text().replace(
        '"detection_batch_size": 1',
        f'"detection_batch_size": {MAX_DETECTION_BATCH_SIZE + 1}',
    )
    path.write_text(payload)

    with pytest.raises(InferenceConfigError, match="detection_batch_size"):
        InferenceConfig.from_json(path)


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
                            "confidence_floor": 0.001,
                            "confidence_threshold": 0.25,
                        },
                    },
                    "runtime_tier": "cpu",
                    "pipeline_depth": 0,
                }
            )
        )
        path = f.name
    with pytest.raises(InferenceConfigError, match="pipeline_depth"):
        InferenceConfig.from_json(path)
