"""F2: the DetectKit preview tiles like TrackerKit, from the model's sidecar."""

from hydra_suite.core.inference.slice_meta import upsert_slice_profile, write_slice_meta
from hydra_suite.detectkit.gui.models import SliceTrainingSettings
from hydra_suite.detectkit.jobs.preview_tiling import resolve_preview_tiling

GEOM = {
    "geometry_mode": "auto_object",
    "imgsz": 1024,
    "object_tile_fraction": 0.1,
    "target_sizes": [51.2, 102.4, 153.6, 204.8],
    "overlap": 0.25,
    "reference_body_px": 44.0,
}


def _model(tmp_path, meta=None):
    model = tmp_path / "det.pt"
    model.write_bytes(b"x")
    if meta is not None:
        write_slice_meta(model, meta)
    return model


def test_no_sidecar_uses_project(tmp_path):
    project = SliceTrainingSettings(enabled=True, overlap=0.3)
    got = resolve_preview_tiling(
        _model(tmp_path), project, project_imgsz=640, override=None
    )
    assert got.source == "project" and got.slice_settings == project
    assert got.imgsz == 640


def test_training_geometry_like_trackerkit(tmp_path):
    project = SliceTrainingSettings(enabled=True, reference_body_px=10.0)
    got = resolve_preview_tiling(
        _model(tmp_path, {"training_geometry": GEOM}),
        project,
        project_imgsz=640,
        override=None,
    )
    assert got.source == "training"
    assert got.imgsz == 640  # the project's knob
    assert got.slice_settings.enabled is True
    assert got.slice_settings.overlap == 0.25
    # median(target_sizes)/imgsz = 128/1024, TrackerKit's value
    assert got.slice_settings.object_tile_fraction == 0.125
    assert got.slice_settings.reference_body_px == 10.0  # dataset (project) first


def test_stamped_body_when_project_has_none(tmp_path):
    got = resolve_preview_tiling(
        _model(tmp_path, {"training_geometry": GEOM}),
        SliceTrainingSettings(enabled=True),
        project_imgsz=640,
        override=None,
    )
    assert got.slice_settings.reference_body_px == 44.0
    assert (got.merge_policy, got.merge_metric) == ("greedy_nmm", "ios")


def test_primary_profile_wins(tmp_path):
    """Review Focus 1."""
    meta = upsert_slice_profile(
        {"training_geometry": GEOM},
        name="High recall",
        settings={
            "object_tile_fraction": 0.08,
            "overlap": 0.35,
            "merge_policy": "nms",
            "merge_metric": "iou",
            "merge_threshold": 0.6,
        },
        primary=True,
    )
    got = resolve_preview_tiling(
        _model(tmp_path, meta),
        SliceTrainingSettings(enabled=True),
        project_imgsz=640,
        override=None,
    )
    assert got.source == "profile:High recall"
    assert got.slice_settings.object_tile_fraction == 0.08
    assert got.slice_settings.target_size_fractions == [0.08]
    assert got.slice_settings.overlap == 0.35
    assert got.slice_settings.merge_threshold == 0.6
    assert (got.merge_policy, got.merge_metric) == ("nms", "iou")


def test_profile_merge_policy_nmm_passes_through_raw(tmp_path):
    """F6: ``nmm`` is a raw pass-through value, never rewritten."""
    meta = upsert_slice_profile(
        {"training_geometry": GEOM},
        name="Legacy",
        settings={"merge_policy": "nmm"},
        primary=True,
    )
    got = resolve_preview_tiling(
        _model(tmp_path, meta),
        SliceTrainingSettings(enabled=True),
        project_imgsz=640,
        override=None,
    )
    assert got.merge_policy == "nmm"


def test_project_toggle_owns_enabled(tmp_path):
    got = resolve_preview_tiling(
        _model(tmp_path, {"training_geometry": GEOM}),
        SliceTrainingSettings(enabled=False),
        project_imgsz=640,
        override=None,
    )
    assert got.slice_settings.enabled is False


def test_disabled_profile_does_not_disable_an_enabled_project(tmp_path):
    """Deviation 16: the project owns ``enabled``, not the profile."""
    meta = upsert_slice_profile(
        {"training_geometry": GEOM},
        name="Off",
        settings={"enabled": False, "overlap": 0.3},
        primary=True,
    )
    got = resolve_preview_tiling(
        _model(tmp_path, meta),
        SliceTrainingSettings(enabled=True),
        project_imgsz=640,
        override=None,
    )
    assert got.slice_settings.enabled is True


def test_override_wins(tmp_path):
    override = SliceTrainingSettings(enabled=True, overlap=0.4)
    got = resolve_preview_tiling(
        _model(tmp_path, {"training_geometry": GEOM}),
        SliceTrainingSettings(),
        project_imgsz=640,
        override=override,
    )
    assert got.source == "override" and got.slice_settings == override


def test_corrupt_sidecar_falls_back_to_project(tmp_path):
    model = _model(tmp_path)
    model.with_name(model.name + ".slice_meta.json").write_text("{nope")
    got = resolve_preview_tiling(
        model, SliceTrainingSettings(enabled=True), project_imgsz=640, override=None
    )
    assert got.source == "project"


def test_no_model_path_uses_project():
    got = resolve_preview_tiling(
        None, SliceTrainingSettings(enabled=True), project_imgsz=800, override=None
    )
    assert got.source == "project" and got.imgsz == 800


def test_cache_key_changes_when_sidecar_changes(tmp_path):
    """Review Focus 4: recalibrating must invalidate cached predictions."""
    from hydra_suite.detectkit.jobs.dataset_inference import preview_settings_dict

    model = _model(tmp_path, {"training_geometry": GEOM})
    a = preview_settings_dict(
        resolve_preview_tiling(
            model, SliceTrainingSettings(enabled=True), project_imgsz=640, override=None
        )
    )
    write_slice_meta(
        model,
        upsert_slice_profile(
            {"training_geometry": GEOM},
            name="P",
            settings={"overlap": 0.4},
            primary=True,
        ),
    )
    b = preview_settings_dict(
        resolve_preview_tiling(
            model, SliceTrainingSettings(enabled=True), project_imgsz=640, override=None
        )
    )
    assert a != b
    assert b["slice_imgsz"] == 640
    assert (b["slice_merge_policy"], b["slice_merge_metric"]) == ("greedy_nmm", "ios")


def test_gui_signature_changes_when_primary_profile_changes(tmp_path):
    """Plan review M3: the in-session reuse check must see a recalibration."""
    from hydra_suite.detectkit.jobs.preview_tiling import dataset_signature

    model = _model(tmp_path, {"training_geometry": GEOM})
    project = SliceTrainingSettings(enabled=True)

    def signature():
        tiling = resolve_preview_tiling(
            model, project, project_imgsz=640, override=None
        )
        return dataset_signature("/src", str(model), "auto", tiling)

    before = signature()
    assert signature() == before  # stable when nothing changed
    write_slice_meta(
        model,
        upsert_slice_profile(
            {"training_geometry": GEOM},
            name="Recalibrated",
            settings={"object_tile_fraction": 0.08, "merge_policy": "nms"},
            primary=True,
        ),
    )
    assert signature() != before


def test_preview_runs_at_the_profile_fraction_exactly(tmp_path):
    """Review Focus 1: no px round trip (0.055*640/640 is 1 ulp off 0.055)."""
    from hydra_suite.detectkit.gui.prediction_preview import sliced_preview_fraction

    meta = upsert_slice_profile(
        {"training_geometry": GEOM},
        name="Fine",
        settings={"object_tile_fraction": 0.055},
        primary=True,
    )
    got = resolve_preview_tiling(
        _model(tmp_path, meta),
        SliceTrainingSettings(enabled=True),
        project_imgsz=640,
        override=None,
    )
    assert sliced_preview_fraction(got.slice_settings) == 0.055
    # A legacy pixel project keeps its median(px)/640 scale.
    legacy = SliceTrainingSettings.from_dict({"target_sizes": [64.0, 96.0, 128.0]})
    assert sliced_preview_fraction(legacy) == 0.15


def test_disabled_preview_key_ignores_the_sidecar(tmp_path):
    """m1: recalibrating must not invalidate a NON-sliced preview cache."""
    from hydra_suite.detectkit.jobs.dataset_inference import preview_settings_dict

    disabled = SliceTrainingSettings(enabled=False)
    plain = tmp_path / "a"
    plain.mkdir()
    profiled = tmp_path / "b"
    profiled.mkdir()
    a = preview_settings_dict(
        resolve_preview_tiling(
            _model(plain, {"training_geometry": GEOM}),
            disabled,
            project_imgsz=640,
            override=None,
        )
    )
    b = preview_settings_dict(
        resolve_preview_tiling(
            _model(
                profiled,
                upsert_slice_profile(
                    {"training_geometry": GEOM},
                    name="P",
                    settings={"overlap": 0.4, "merge_policy": "nms"},
                    primary=True,
                ),
            ),
            disabled,
            project_imgsz=640,
            override=None,
        )
    )
    assert a == b
    assert a["slice_settings"] == {"enabled": False}
    assert not any(key.startswith("slice_") and key != "slice_settings" for key in a)
