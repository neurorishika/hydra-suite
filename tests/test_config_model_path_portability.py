"""Model paths in saved configs are models-root-relative and resolve back.

A packed job's per-video sidecar must contain no staging-machine absolute
paths, and `verify_job` enforces that. Two keys leaked absolute paths on save:
`color_tag_model_path` and `cnn_classifiers[].model_path`. Relativizing on save
is only safe if the load side resolves symmetrically -- `cnn_classifiers` was
already resolved, `color_tag_model_path` was NOT.
"""

import pytest

from hydra_suite.core.inference.model_paths import make_model_path_relative
from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    for rel in ("classification/colortag/tags.pth", "classification/identity/ids.pth"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(root))
    return root


def _runtime():
    return RuntimeContext(fps=30.0, total_frames=10, frame_width=64, frame_height=64)


def test_color_tag_absolute_path_is_relativized(models_root):
    absolute = str(models_root / "classification" / "colortag" / "tags.pth")
    assert make_model_path_relative(absolute) == "classification/colortag/tags.pth"


def test_color_tag_relative_path_resolves_in_engine_params(models_root):
    """The load-side asymmetry: this key was emitted verbatim, unresolved."""
    cfg = {
        "color_tag_model_path": "classification/colortag/tags.pth",
        "identity_method": "color_tag",
        "enable_identity_analysis": True,
    }
    params = build_engine_params(cfg, runtime=_runtime())
    expected = str(models_root / "classification" / "colortag" / "tags.pth")
    assert params["COLOR_TAG_MODEL_PATH"] == expected
    # The legacy singular bridge must carry the RESOLVED value too.
    assert params["CNN_CLASSIFIER_MODEL_PATH"] == expected


def test_empty_color_tag_stays_empty(models_root):
    params = build_engine_params({"color_tag_model_path": ""}, runtime=_runtime())
    assert params["COLOR_TAG_MODEL_PATH"] == ""
    assert params["CNN_CLASSIFIER_MODEL_PATH"] == ""


def test_color_tag_outside_models_root_is_left_absolute(models_root, tmp_path):
    outside = tmp_path / "elsewhere" / "tags.pth"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"z")
    assert make_model_path_relative(str(outside)) == str(outside)
    params = build_engine_params(
        {"color_tag_model_path": str(outside)}, runtime=_runtime()
    )
    assert params["COLOR_TAG_MODEL_PATH"] == str(outside)


def test_absolute_color_tag_still_resolves_unchanged(models_root):
    """Existing configs holding absolute paths must keep working."""
    absolute = str(models_root / "classification" / "colortag" / "tags.pth")
    params = build_engine_params({"color_tag_model_path": absolute}, runtime=_runtime())
    assert params["COLOR_TAG_MODEL_PATH"] == absolute


def test_cnn_classifier_round_trip_is_identity_for_engine_params(models_root):
    absolute = str(models_root / "classification" / "identity" / "ids.pth")
    a = build_engine_params(
        {"cnn_classifiers": [{"model_path": absolute, "batch_size": 8}]},
        runtime=_runtime(),
    )
    b = build_engine_params(
        {
            "cnn_classifiers": [
                {"model_path": make_model_path_relative(absolute), "batch_size": 8}
            ]
        },
        runtime=_runtime(),
    )
    assert a["CNN_CLASSIFIERS"] == b["CNN_CLASSIFIERS"]
    assert a["CNN_CLASSIFIERS"][0]["model_path"] == absolute
