import json
from pathlib import Path

from hydra_suite.trackerkit.engine_params import (
    SLICE_MERGE_DEFAULTS,
    RuntimeContext,
    build_engine_params,
)

RUNTIME = RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)

# Explicit advanced baseline: every SAHI key at a value distinguishable from
# both the profile's and the module defaults, so "the overlay did nothing" and
# "the overlay wrote defaults" are different observations.
ADVANCED = {
    "slice_overlap": 0.2,
    "slice_object_tile_fraction": 0.15,
    "slice_width": 0,
    "slice_height": 0,
    "slice_trained_body_px": 0.0,
    "slice_merge_policy": "greedy_nmm",
    "slice_merge_metric": "ios",
    "slice_merge_threshold": 0.5,
    "slice_merge_backend": "cv2",
}

SETTINGS = {
    "enabled": True,
    "geometry_mode": "auto_object",
    "slice_width": 704,
    "slice_height": 512,
    "overlap": 0.31,
    "object_tile_fraction": 0.11,
    "trained_body_px": 560.0,
    "confidence_threshold": 0.42,
    "merge_policy": "nmm",
    "merge_metric": "iou",
    "merge_threshold": 0.6,
    "merge_backend": "cv2",
}


def _sidecar(tmp_path: Path, payload: dict, name: str = "model.pt") -> str:
    model = tmp_path / name
    model.write_text("stub", encoding="utf-8")
    (tmp_path / (name + ".slice_meta.json")).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return str(model)


def _profiled(tmp_path: Path) -> str:
    return _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {"geometry_mode": "auto_model", "imgsz": 640},
            "primary_profile_id": "balanced",
            "profiles": [{"id": "balanced", "name": "Balanced", "settings": SETTINGS}],
        },
    )


def _cfg(model_path: str, **extra) -> dict:
    base = {
        "yolo_obb_mode": "direct",
        "yolo_obb_direct_model_path": model_path,
        "slice_enabled": False,
        "slice_geometry_mode": "auto_model",
        "yolo_confidence_threshold": 0.25,
    }
    base.update(extra)
    return base


def _build(cfg):
    return build_engine_params(cfg, runtime=RUNTIME, advanced_config=dict(ADVANCED))


def test_named_profile_supplies_the_advanced_only_keys(tmp_path):
    params = _build(_cfg(_profiled(tmp_path), slice_profile_id="balanced"))
    assert params["SLICE_OVERLAP"] == 0.31
    assert params["SLICE_OBJECT_TILE_FRACTION"] == 0.11
    assert params["SLICE_WIDTH"] == 704
    assert params["SLICE_HEIGHT"] == 512
    assert params["SLICE_TRAINED_BODY_PX"] == 560.0
    assert params["SLICE_MERGE_POLICY"] == "nmm"
    assert params["SLICE_MERGE_METRIC"] == "iou"
    assert params["SLICE_MERGE_THRESHOLD"] == 0.6
    assert params["SLICE_MERGE_BACKEND"] == "cv2"


def test_config_owns_enabled_geometry_and_confidence(tmp_path):
    """Ruling R1/R2: the three keys build_config_dict persists are authoritative.

    The profile claims enabled=True, auto_object and 0.42; the config says
    False, auto_model and 0.25. The config must win -- it already records what
    the user saw when the session was saved.
    """
    params = _build(_cfg(_profiled(tmp_path), slice_profile_id="balanced"))
    assert params["SLICE_ENABLED"] is False
    assert params["SLICE_GEOMETRY_MODE"] == "auto_model"
    assert params["YOLO_CONFIDENCE_THRESHOLD"] == 0.25


def test_confidence_disagreement_warns(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _build(_cfg(_profiled(tmp_path), slice_profile_id="balanced"))
    assert any(
        "Balanced" in record.message and "0.42" in record.message
        for record in caplog.records
    )


def test_matching_confidence_does_not_warn(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _build(
            _cfg(
                _profiled(tmp_path),
                slice_profile_id="balanced",
                yolo_confidence_threshold=0.42,
            )
        )
    assert not [r for r in caplog.records if "confidence" in r.message.lower()]


def test_primary_applies_when_config_names_nothing(tmp_path):
    assert _build(_cfg(_profiled(tmp_path)))["SLICE_OVERLAP"] == 0.31


def test_custom_id_uses_saved_snapshot_not_primary(tmp_path):
    snapshot = dict(SETTINGS, overlap=0.07, merge_policy="nms")
    params = _build(
        _cfg(
            _profiled(tmp_path),
            slice_profile_id="__custom__",
            slice_profile_settings=snapshot,
        )
    )
    assert params["SLICE_OVERLAP"] == 0.07
    assert params["SLICE_MERGE_POLICY"] == "nms"


def test_training_geometry_is_applied(tmp_path):
    """Ruling R2: the commonest sidecar in the wild has no profiles at all."""
    model = _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {
                "geometry_mode": "auto_object",
                "imgsz": 640,
                "overlap": 0.27,
                "reference_body_px": 480.0,
            },
            "primary_profile_id": "",
            "profiles": [],
        },
    )
    params = _build(_cfg(model, slice_enabled=True))
    assert params["SLICE_OVERLAP"] == 0.27
    assert params["SLICE_TRAINED_BODY_PX"] == 480.0
    # _training_values returns enabled=True; the config must still own it.
    assert params["SLICE_ENABLED"] is True
    params_off = _build(_cfg(model, slice_enabled=False))
    assert params_off["SLICE_ENABLED"] is False


def test_legacy_v1_flat_sidecar_is_applied(tmp_path):
    model = _sidecar(
        tmp_path,
        {"geometry_mode": "auto_object", "imgsz": 640, "overlap": 0.33},
    )
    assert _build(_cfg(model))["SLICE_OVERLAP"] == 0.33


def test_unclaimed_merge_keys_become_defaults(tmp_path):
    """Ruling R3: unclaimed means the DEFAULT, never 'whatever was there'."""
    settings = {k: v for k, v in SETTINGS.items() if not k.startswith("merge_")}
    model = _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {"geometry_mode": "auto_model"},
            "primary_profile_id": "p",
            "profiles": [{"id": "p", "name": "P", "settings": settings}],
        },
    )
    advanced = dict(ADVANCED, slice_merge_policy="nmm", slice_merge_threshold=0.9)
    params = build_engine_params(_cfg(model), runtime=RUNTIME, advanced_config=advanced)
    assert params["SLICE_MERGE_POLICY"] == SLICE_MERGE_DEFAULTS["merge_policy"]
    assert params["SLICE_MERGE_THRESHOLD"] == SLICE_MERGE_DEFAULTS["merge_threshold"]


def test_missing_sidecar_leaves_advanced_untouched(tmp_path):
    model = tmp_path / "bare.pt"
    model.write_text("stub", encoding="utf-8")
    advanced = dict(ADVANCED, slice_overlap=0.44)
    params = build_engine_params(
        _cfg(str(model), slice_profile_id="balanced"),
        runtime=RUNTIME,
        advanced_config=advanced,
    )
    assert params["SLICE_OVERLAP"] == 0.44


def test_nonexistent_model_path_does_not_raise(tmp_path):
    assert _build(_cfg(str(tmp_path / "nope.pt")))["SLICE_OVERLAP"] == 0.2


def test_corrupt_sidecar_is_a_no_op(tmp_path):
    model = tmp_path / "c.pt"
    model.write_text("stub", encoding="utf-8")
    (tmp_path / "c.pt.slice_meta.json").write_text("{not json", encoding="utf-8")
    assert _build(_cfg(str(model)))["SLICE_OVERLAP"] == 0.2


def test_sequential_mode_gets_no_overlay(tmp_path):
    params = _build(
        _cfg(
            _profiled(tmp_path),
            yolo_obb_mode="sequential",
            slice_profile_id="balanced",
        )
    )
    assert params["SLICE_OVERLAP"] == 0.2


def test_non_numeric_confidence_threshold_does_not_raise(tmp_path):
    """A malformed sidecar confidence_threshold must never take down the run."""
    settings = dict(SETTINGS, confidence_threshold="bogus")
    model = _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {"geometry_mode": "auto_model"},
            "primary_profile_id": "balanced",
            "profiles": [{"id": "balanced", "name": "Balanced", "settings": settings}],
        },
    )
    params = _build(_cfg(model, slice_profile_id="balanced"))
    # Still applies the nine geometry/merge keys despite the bad confidence.
    assert params["SLICE_OVERLAP"] == 0.31
    assert params["SLICE_MERGE_POLICY"] == "nmm"


def test_numeric_string_merge_threshold_is_coerced_to_float(tmp_path):
    settings = dict(SETTINGS, merge_threshold="0.6")
    model = _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {"geometry_mode": "auto_model"},
            "primary_profile_id": "balanced",
            "profiles": [{"id": "balanced", "name": "Balanced", "settings": settings}],
        },
    )
    params = _build(_cfg(model, slice_profile_id="balanced"))
    assert params["SLICE_MERGE_THRESHOLD"] == 0.6
    assert isinstance(params["SLICE_MERGE_THRESHOLD"], float)


def test_non_numeric_merge_threshold_falls_back_to_default(tmp_path):
    settings = dict(SETTINGS, merge_threshold="bogus")
    model = _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {"geometry_mode": "auto_model"},
            "primary_profile_id": "balanced",
            "profiles": [{"id": "balanced", "name": "Balanced", "settings": settings}],
        },
    )
    params = _build(_cfg(model, slice_profile_id="balanced"))
    assert params["SLICE_MERGE_THRESHOLD"] == SLICE_MERGE_DEFAULTS["merge_threshold"]
