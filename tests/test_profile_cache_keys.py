"""Profiles must split the detection cache exactly where they differ.

Confidence is DELIBERATELY excluded from the key (cache/keys.py:100 -- it is
re-applied at tracking time over raw detections), so two profiles differing
only in confidence correctly share detections. Geometry, overlap and merge
settings change which raw detections exist and must not.

The ``_hash_for`` helper below runs through the REAL production translation
chain -- panel values -> ``advanced_config`` -> ``build_engine_params`` ->
``_slice_config_from_params`` -> ``_slice_config_hash`` -- via
``hydra_suite.trackerkit.engine_params.build_engine_params``, the shared
Qt-free param builder that also feeds the CLI and (eventually) the GUI
bridge. This is a deliberate, documented exception to the "core must not
depend on app layers" rule: TESTS may import an app-layer module even though
``src/`` may not.
"""

from hydra_suite.core.inference.cache.keys import _slice_config_hash
from hydra_suite.core.inference.config import _slice_config_from_params
from hydra_suite.core.inference.slice_meta import (
    slice_meta_to_panel_values,
    upsert_slice_profile,
)
from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params

TRAINING = {"geometry_mode": "auto_object", "imgsz": 640, "overlap": 0.2}
BASE_SETTINGS = {
    "enabled": True,
    "geometry_mode": "auto_object",
    "slice_width": 0,
    "slice_height": 0,
    "overlap": 0.2,
    "object_tile_fraction": 0.4,
    "trained_body_px": 560.0,
    "confidence_threshold": 0.35,
    "merge_policy": "greedy_nmm",
    "merge_metric": "ios",
    "merge_threshold": 0.5,
    "merge_backend": "cv2",
}

_RUNTIME = RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)


def _hash_for(**overrides) -> str:
    """Panel values -> advanced_config -> build_engine_params -> SliceConfig -> hash.

    Goes through ``hydra_suite.trackerkit.engine_params.build_engine_params``
    (the real, shared Qt-free translation used by both the CLI and the GUI
    bridge -- see engine_params.py:349-914) rather than hand-building the
    ``SLICE_*`` params dict, so this test covers the production panel ->
    advanced_config -> params -> SliceConfig chain, not a local copy of it.
    """
    meta = upsert_slice_profile(
        TRAINING, name="P", settings=dict(BASE_SETTINGS, **overrides)
    )
    values = slice_meta_to_panel_values(meta, meta["profiles"][0]["id"])
    cfg = {
        "detection_method": "yolo_obb",
        "slice_enabled": values["enabled"],
        "slice_geometry_mode": values["geometry_mode"],
    }
    advanced_config = {
        "slice_overlap": values["overlap"],
        "slice_height": values["slice_height"],
        "slice_width": values["slice_width"],
        "slice_object_tile_fraction": values["object_tile_fraction"],
        "slice_trained_body_px": values["trained_body_px"],
        "slice_merge_policy": values["merge_policy"] or "greedy_nmm",
        "slice_merge_metric": values["merge_metric"] or "ios",
        "slice_merge_threshold": values["merge_threshold"] or 0.5,
        "slice_merge_backend": values["merge_backend"] or "cv2",
    }
    params = build_engine_params(cfg, runtime=_RUNTIME, advanced_config=advanced_config)
    slice_cfg = _slice_config_from_params(
        params, "SLICE_", reference_body_px=values["trained_body_px"]
    )
    return _slice_config_hash(slice_cfg)


def test_geometry_difference_splits_the_detection_cache():
    assert _hash_for() != _hash_for(object_tile_fraction=0.7, overlap=0.1)


def test_merge_difference_splits_the_detection_cache():
    assert _hash_for() != _hash_for(merge_threshold=0.8)
    assert _hash_for() != _hash_for(merge_policy="nmm")


def test_confidence_only_difference_shares_the_cache_by_design():
    assert _hash_for() == _hash_for(confidence_threshold=0.15)


def test_every_profile_field_survives_the_panel_translation():
    meta = upsert_slice_profile(TRAINING, name="P", settings=BASE_SETTINGS)
    values = slice_meta_to_panel_values(meta, meta["profiles"][0]["id"])
    for key in (
        "merge_policy",
        "merge_metric",
        "merge_threshold",
        "merge_backend",
        "confidence_threshold",
    ):
        assert values[key] is not None, f"{key} dropped in translation"


def test_two_profiles_live_on_one_artifact():
    meta = upsert_slice_profile(TRAINING, name="Balanced", settings=BASE_SETTINGS)
    meta = upsert_slice_profile(
        meta, name="Fast scan", settings=dict(BASE_SETTINGS, object_tile_fraction=0.7)
    )
    assert len(meta["profiles"]) == 2 and meta["schema_version"] == 2
