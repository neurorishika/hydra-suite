from hydra_suite.detectkit.gui.models import DetectKitProject, SliceTrainingSettings


def test_slice_settings_defaults_off():
    s = SliceTrainingSettings()
    assert s.enabled is False
    assert s.geometry_mode == "auto_object"
    assert s.target_sizes == [200.0, 300.0, 400.0]


def test_project_slice_settings_round_trip(tmp_path):
    proj = DetectKitProject(project_dir=tmp_path)
    proj.slice_settings = SliceTrainingSettings(
        enabled=True,
        geometry_mode="custom",
        slice_width=512,
        slice_height=512,
        target_sizes=[150.0, 350.0],
        negative_tile_fraction=0.2,
    )
    out = tmp_path / "state.json"
    proj.save(out)
    loaded = DetectKitProject.load(out)
    assert loaded.slice_settings.enabled is True
    assert loaded.slice_settings.geometry_mode == "custom"
    assert loaded.slice_settings.slice_width == 512
    assert loaded.slice_settings.target_sizes == [150.0, 350.0]
    assert abs(loaded.slice_settings.negative_tile_fraction - 0.2) < 1e-9


def test_legacy_project_without_slice_settings_loads(tmp_path):
    out = tmp_path / "legacy.json"
    out.write_text('{"version": 1, "class_names": ["ant"]}', encoding="utf-8")
    loaded = DetectKitProject.load(out)
    assert loaded.slice_settings.enabled is False  # default when absent
