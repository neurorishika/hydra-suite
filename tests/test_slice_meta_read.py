import json

from hydra_suite.core.inference.slice_meta import (
    read_slice_meta,
    slice_meta_to_panel_values,
)


def test_read_absent_returns_none(tmp_path):
    assert read_slice_meta(tmp_path / "model.pt") is None


def test_read_malformed_returns_none(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    (tmp_path / "model.pt.slice_meta.json").write_text("{not json", encoding="utf-8")
    assert read_slice_meta(model) is None


def test_read_present_returns_dict(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    (tmp_path / "model.pt.slice_meta.json").write_text(
        json.dumps({"geometry_mode": "auto_object", "reference_body_px": 560.0}),
        encoding="utf-8",
    )
    meta = read_slice_meta(model)
    assert meta["reference_body_px"] == 560.0


def test_map_full():
    meta = {
        "geometry_mode": "auto_object",
        "overlap": 0.2,
        "reference_body_px": 560.0,
        "target_sizes": [200.0, 300.0, 400.0],
        "imgsz": 640,
    }
    v = slice_meta_to_panel_values(meta)
    assert v["enabled"] is True
    assert v["geometry_mode"] == "auto_object"
    assert v["overlap"] == 0.2
    assert v["trained_body_px"] == 560.0
    assert abs(v["object_tile_fraction"] - 300.0 / 640.0) < 1e-6


def test_map_empty_targets_falls_back_to_object_tile_fraction():
    meta = {
        "geometry_mode": "auto_object",
        "target_sizes": [],
        "imgsz": 640,
        "object_tile_fraction": 0.17,
        "reference_body_px": 100.0,
    }
    assert slice_meta_to_panel_values(meta)["object_tile_fraction"] == 0.17


def test_map_missing_imgsz_falls_back():
    meta = {
        "target_sizes": [300.0],
        "object_tile_fraction": 0.18,
        "reference_body_px": 50.0,
    }
    assert slice_meta_to_panel_values(meta)["object_tile_fraction"] == 0.18


def test_map_missing_keys_use_defaults():
    v = slice_meta_to_panel_values({})
    assert v["geometry_mode"] == "auto_object"
    assert v["overlap"] == 0.2
    assert v["trained_body_px"] == 0.0
    assert v["object_tile_fraction"] == 0.15
