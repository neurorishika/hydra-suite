from hydra_suite.core.inference.slice_meta import resolve_slice_profile_values

TRAINING = {"geometry_mode": "auto_object", "imgsz": 640, "overlap": 0.2}
SETTINGS = {
    "enabled": True,
    "geometry_mode": "auto_object",
    "slice_width": 0,
    "slice_height": 0,
    "overlap": 0.31,
    "object_tile_fraction": 0.11,
    "trained_body_px": 560.0,
    "confidence_threshold": 0.35,
    "merge_policy": "nmm",
    "merge_metric": "iou",
    "merge_threshold": 0.6,
    "merge_backend": "cv2",
}
META = {
    "schema_version": 2,
    "training_geometry": TRAINING,
    "primary_profile_id": "balanced",
    "profiles": [
        {"id": "balanced", "name": "Balanced", "settings": SETTINGS},
        {
            "id": "recall",
            "name": "High recall",
            "settings": dict(SETTINGS, overlap=0.45),
        },
    ],
}
SNAPSHOT = dict(SETTINGS, overlap=0.07, base_profile_name="Balanced")


def test_live_id_wins_over_snapshot():
    v = resolve_slice_profile_values(META, "recall", SNAPSHOT)
    assert v["resolution"] == "requested"
    assert v["overlap"] == 0.45


def test_custom_with_snapshot_uses_snapshot_not_primary():
    v = resolve_slice_profile_values(META, "__custom__", SNAPSHOT)
    assert v["resolution"] == "saved_settings"
    assert v["overlap"] == 0.07


def test_custom_without_snapshot_falls_back_to_primary():
    v = resolve_slice_profile_values(META, "__custom__", None)
    assert v["resolution"] == "primary"
    assert v["profile_id"] == "balanced"


def test_missing_id_with_snapshot_uses_snapshot():
    v = resolve_slice_profile_values(META, "deleted-profile", SNAPSHOT)
    assert v["resolution"] == "saved_settings"
    assert v["overlap"] == 0.07


def test_missing_id_without_snapshot_uses_primary():
    v = resolve_slice_profile_values(META, "deleted-profile", None)
    assert v["resolution"] == "primary"


def test_training_request_ignores_snapshot():
    v = resolve_slice_profile_values(META, "__training__", SNAPSHOT)
    assert v["resolution"] == "training"
    assert v["profile_id"] is None


def test_empty_id_uses_primary():
    assert resolve_slice_profile_values(META, "", None)["resolution"] == "primary"


def test_no_primary_and_no_id_is_training():
    meta = dict(META, primary_profile_id="")
    assert resolve_slice_profile_values(meta, "", None)["resolution"] == "training"
