import json

import pytest

from hydra_suite.core.inference.slice_meta import (
    _training_values,
    merge_training_geometry,
    resolve_slice_profile_values,
    sidecar_path,
    slice_meta_to_panel_values,
    write_slice_meta,
)
from hydra_suite.core.inference.tiling_meta import (
    read_tiling_meta,
    sam3_meta_path,
    training_geometry_from_sam3_manifest,
    training_geometry_from_yolo_manifest,
)
from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "det.pt"
    p.write_bytes(b"x")
    return p


V1_YOLO = {
    "geometry_mode": "auto_object",
    "imgsz": 640,
    "object_tile_fraction": 0.1,
    "overlap": 0.2,
    "target_sizes": [32, 64, 96, 128],
    "reference_body_px": 40.0,
    "slice_width": 0,
    "slice_height": 0,
}


def test_absent_returns_none(model):
    assert read_tiling_meta(model) is None


def test_corrupt_returns_none(model):
    sidecar_path(model).write_text("{not json", encoding="utf-8")
    assert read_tiling_meta(model) is None


def test_v1_flat_yolo(model):
    _write(sidecar_path(model), V1_YOLO)
    meta = read_tiling_meta(model)
    assert meta.model_family == "yolo" and meta.source == "slice_meta"
    assert meta.training.object_tile_fractions == (0.05, 0.1, 0.15, 0.2)
    assert meta.training.reference_body_px == 40.0
    assert meta.training.fragment_policy == "drop"
    assert meta.operating_fraction == _training_values(V1_YOLO)["object_tile_fraction"]
    assert meta.imgsz == 640


def test_v1_target_sizes_without_imgsz(model):
    """Review Focus 2: 640 anchor for YOLO; operating matches TrackerKit."""
    doc = {k: v for k, v in V1_YOLO.items() if k != "imgsz"}
    _write(sidecar_path(model), doc)
    meta = read_tiling_meta(model)
    assert meta.training.object_tile_fractions == (0.05, 0.1, 0.15, 0.2)
    assert (
        meta.operating_fraction == _training_values(doc)["object_tile_fraction"] == 0.1
    )


def test_v2_profiles_preserved(model):
    profile = {
        "id": "bal-1",
        "name": "Balanced",
        "note": "",
        "settings": {"overlap": 0.3},
        "measurement": {},
    }
    _write(
        sidecar_path(model),
        {
            "schema_version": 2,
            "training_geometry": V1_YOLO,
            "primary_profile_id": "bal-1",
            "profiles": [profile],
        },
    )
    meta = read_tiling_meta(model)
    assert meta.primary_profile_id == "bal-1"
    assert meta.profiles == (profile,)


def test_v2_profiles_only_no_geometry(model):
    profile = {"id": "a-1", "name": "A", "note": "", "settings": {}, "measurement": {}}
    _write(
        sidecar_path(model),
        {"schema_version": 2, "training_geometry": {}, "profiles": [profile]},
    )
    meta = read_tiling_meta(model)
    assert meta.training is None
    assert meta.profiles == (profile,)


def test_v3_yolo_round_trip(model):
    geometry = training_geometry_from_yolo_manifest(
        dict(V1_YOLO, target_sizes=[33, 70, 101])
    )
    write_slice_meta(
        model, merge_training_geometry(None, geometry, model_family="yolo")
    )
    meta = read_tiling_meta(model)
    assert meta.operating_fraction == geometry["prefill_object_tile_fraction"]
    assert (
        list(meta.training.object_tile_fractions) == geometry["object_tile_fractions"]
    )


def test_v3_sam3_round_trip(model):
    geometry = training_geometry_from_sam3_manifest(
        {
            "tile_px_set": [[1940, 1940], [970, 970]],
            "object_tile_fractions": [0.0275, 0.055],
            "prefill_object_tile_fraction": 0.04125,
            "reference_body_px": 53.4,
            "tile_overlap": 0.25,
            "geometry_mode": "auto_object",
        },
        imgsz=1008,
    )
    write_slice_meta(
        model, merge_training_geometry(None, geometry, model_family="sam3")
    )
    meta = read_tiling_meta(model)
    assert meta.model_family == "sam3"
    assert meta.tile_px_set == ((1940, 1940), (970, 970))
    assert meta.operating_fraction == 0.04125
    assert meta.training.fragment_policy == "crowd"
    assert meta.training.overlap == 0.25


def test_legacy_sam3_meta_single_scale(model):
    _write(
        sam3_meta_path(model),
        {
            "base_variant": "sam3",
            "prompt": "ant",
            "train_tile_px": 971,
            "object_tile_fraction": 0.055,
            "reference_body_px": 53.4,
            "imgsz": 1008,
        },
    )
    meta = read_tiling_meta(model)
    assert meta.model_family == "sam3" and meta.source == "sam3_meta"
    assert meta.tile_px_set == ((971, 971),)
    assert meta.training.object_tile_fractions == (0.055,)
    assert meta.operating_fraction == 0.055
    assert meta.training.fragment_policy == "crowd"
    assert "prompt" not in meta.extras


def test_legacy_sam3_meta_multiscale(model):
    _write(
        sam3_meta_path(model),
        {
            "train_tile_px_set": [[1940, 1940], [970, 970]],
            "object_tile_fractions": [0.0275, 0.055],
            "prefill_object_tile_fraction": 0.04125,
            "reference_body_px": 53.4,
            "imgsz": 1008,
        },
    )
    meta = read_tiling_meta(model)
    assert meta.tile_px_set == ((1940, 1940), (970, 970))
    assert meta.operating_fraction == 0.04125


def test_slice_meta_geometry_wins_over_sam3_meta(model):
    _write(
        sam3_meta_path(model),
        {"train_tile_px": 500, "object_tile_fraction": 0.1, "imgsz": 1008},
    )
    geometry = training_geometry_from_sam3_manifest(
        {
            "tile_px": [971, 971],
            "object_tile_fraction": 0.055,
            "reference_body_px": 53.4,
        },
        imgsz=1008,
    )
    write_slice_meta(
        model, merge_training_geometry(None, geometry, model_family="sam3")
    )
    meta = read_tiling_meta(model)
    assert meta.source == "slice_meta"
    assert meta.tile_px_set == ((971, 971),)


def test_lenient_on_bad_values(model):
    _write(
        sidecar_path(model),
        {"overlap": 1.5, "geometry_mode": "weird", "object_tile_fraction": 0},
    )
    meta = read_tiling_meta(model)
    assert meta.training.overlap == 0.9
    # Review M3: an invalid YOLO mode reads as _training_values' auto_object.
    assert meta.training.geometry_mode == "auto_object"


@pytest.mark.parametrize(
    "doc",
    [
        {"tile_px_set": 971},
        {"tile_px_set": [["a", "b"]]},
        {"tile_px_set": [[float("nan"), 1]]},
        {"tile_px_set": [[float("inf"), 1]]},
        {"imgsz": float("inf")},
        {"imgsz": float("nan")},
        {"imgsz": 0.5, "target_sizes": [64]},
        {"train_tile_px": 971, "imgsz": float("nan")},
        {"training_geometry": [1, 2]},
        {"profiles": {"a": 1}},
        {"model_family": 7, "overlap": 0.2},
        {"object_tile_fractions": {"a": 1}},
        {"reference_body_px": "1e999"},
        {"imgsz": 10**400},
        {"train_tile_px": 10**400},
        {"train_tile_px_set": [[10**400, 1]]},
        {"model_family": [1], "overlap": 0.2},
    ],
)
def test_hostile_documents_never_raise(model, doc):
    """Adversarial M4."""
    sidecar_path(model).write_text(json.dumps(doc, allow_nan=True), encoding="utf-8")
    read_tiling_meta(model)


def test_profiles_survive_hostile_geometry(model):
    """A malformed geometry must not cost the user their calibration profiles."""
    profile = {"id": "p-1", "name": "P", "note": "", "settings": {}, "measurement": {}}
    _write(
        sidecar_path(model),
        {
            "schema_version": 3,
            "model_family": "yolo",
            "training_geometry": {"imgsz": 10**400, "train_tile_px": 10**400},
            "primary_profile_id": "p-1",
            "profiles": [profile],
        },
    )
    meta = read_tiling_meta(model)
    assert meta is not None and meta.profiles == (profile,)


GOOD_PROFILE = {
    "id": "p1",
    "name": "P1",
    "note": "",
    "settings": {"object_tile_fraction": 0.1},
    "measurement": {},
}
GOOD_GEOM = {
    "geometry_mode": "auto_object",
    "imgsz": 640,
    "object_tile_fraction": 0.1,
    "target_sizes": [64],
    "reference_body_px": 40.0,
}


@pytest.mark.parametrize("bad_measurement", ["oops", [1, 2], 5])
def test_malformed_profile_measurement_keeps_geometry_and_good_profile(
    model, bad_measurement
):
    """Review M1: one malformed profile measurement must not cost the whole document."""
    bad = {
        "id": "p2",
        "name": "P2",
        "settings": {"object_tile_fraction": 0.2},
        "measurement": bad_measurement,
    }
    doc = {
        "schema_version": 3,
        "model_family": "yolo",
        "training_geometry": GOOD_GEOM,
        "primary_profile_id": "p1",
        "profiles": [GOOD_PROFILE, bad],
    }
    _write(sidecar_path(model), doc)
    meta = read_tiling_meta(model)
    assert meta is not None and meta.training is not None
    assert meta.training.reference_body_px == 40.0
    assert meta.profiles[0] == GOOD_PROFILE
    assert meta.profiles[1]["measurement"] == {}
    # TrackerKit's readers on the same file
    assert slice_meta_to_panel_values(doc, None)["profile_id"] == "p1"
    assert resolve_slice_profile_values(doc, "p2", None)["profile_id"] == "p2"
    cfg = {
        "yolo_obb_mode": "direct",
        "yolo_obb_direct_model_path": str(model),
        "slice_enabled": False,
        "slice_geometry_mode": "auto_model",
        "yolo_confidence_threshold": 0.25,
        "slice_profile_id": "p1",
    }
    build_engine_params(
        cfg,
        runtime=RuntimeContext(
            fps=30.0, total_frames=100, frame_width=640, frame_height=480
        ),
        advanced_config={},
    )


@pytest.mark.parametrize("bad_fraction", [None, 0, 1.5])
def test_legacy_sam3_meta_never_invents_a_fraction(model, bad_fraction):
    """Review M2: canonicalize's bare-scalar fallback is not a SAM3 measurement."""
    _write(
        sam3_meta_path(model),
        {
            "train_tile_px": 971,
            "object_tile_fraction": bad_fraction,
            "reference_body_px": 55.4,
            "imgsz": 1008,
        },
    )
    meta = read_tiling_meta(model)
    assert meta.operating_fraction is None
    assert meta.training.object_tile_fractions == ()
    assert meta.tile_px_set == ((971, 971),)


def test_legacy_sam3_meta_family_defaults(model):
    """Review M3: SAM3 defaults, not the YOLO ones, and no default fraction claimed."""
    _write(
        sam3_meta_path(model),
        {"train_tile_px": 971, "reference_body_px": 55.4, "imgsz": 1008},
    )
    training = read_tiling_meta(model).training
    assert training.geometry_mode == "auto_object"
    assert training.merge_policy == "nms" and training.merge_metric == "polygon_iou"
    assert training.fragment_policy == "crowd"
    assert training.object_tile_fractions == ()


def test_yolo_family_defaults_claim_no_fraction(model):
    _write(sidecar_path(model), {"overlap": 0.2, "reference_body_px": 40.0})
    training = read_tiling_meta(model).training
    assert training.geometry_mode == "auto_object"
    assert training.merge_policy == "greedy_nmm" and training.merge_metric == "ios"
    assert training.fragment_policy == "drop"
    assert training.object_tile_fractions == ()
