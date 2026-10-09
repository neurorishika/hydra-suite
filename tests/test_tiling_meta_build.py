import pytest

from hydra_suite.core.inference.geometry_drift import stamped_object_tile_fraction
from hydra_suite.core.inference.slice_meta import (
    SLICE_META_SCHEMA_VERSION,
    _training_values,
    merge_training_geometry,
    normalized_slice_meta,
    training_geometry,
)
from hydra_suite.core.inference.tiling_meta import (
    training_geometry_from_sam3_manifest,
    training_geometry_from_yolo_manifest,
)


def _yolo_manifest(**over):
    base = {
        "geometry_mode": "auto_object",
        "imgsz": 640,
        "object_tile_fraction": 0.1,
        "slice_width": 0,
        "slice_height": 0,
        "overlap": 0.2,
        "min_area_ratio": 0.25,
        "negative_tile_fraction": 0.15,
        "target_sizes": [32, 64, 96, 128],
        "full_frame_mix": True,
        "reference_body_px": 41.5,
        "multiscale_loss_balance": {"enabled": True, "power": 0.5},
    }
    base.update(over)
    return base


YOLO_MANIFESTS = [
    _yolo_manifest(),
    _yolo_manifest(target_sizes=[33, 70, 101]),
    _yolo_manifest(target_sizes=[], object_tile_fraction=0.137),
    _yolo_manifest(target_sizes=[96], imgsz=1024),
    _yolo_manifest(geometry_mode="custom", slice_width=800, slice_height=600),
    _yolo_manifest(target_sizes=[4, 8], imgsz=640),
    {
        "geometry_mode": "auto_object",
        "target_sizes": [200.0, 300.0],
        "reference_body_px": 42.0,
    },
    {"object_tile_fraction": 0, "overlap": 0.95},
    # No bare scalar: a derived prefill would change stamped_object_tile_fraction.
    {"target_sizes": [64], "imgsz": 640},
    {},
]


@pytest.mark.parametrize("manifest", YOLO_MANIFESTS)
def test_yolo_v3_is_additive(manifest):
    """Review Focus 1 + adversarial M3: every v2 consumer sees identical input."""
    v3 = training_geometry_from_yolo_manifest(manifest)
    assert {k: v3[k] for k in manifest} == manifest
    assert _training_values(v3) == _training_values(manifest)
    assert stamped_object_tile_fraction(v3) == stamped_object_tile_fraction(manifest)


def test_yolo_v3_added_keys():
    v3 = training_geometry_from_yolo_manifest(_yolo_manifest())
    assert v3["object_tile_fractions"] == [0.05, 0.1, 0.15, 0.2]
    assert v3["prefill_object_tile_fraction"] == pytest.approx(0.125)
    assert v3["trained_body_px"] == 41.5
    assert v3["fragment_policy"] == "drop"
    assert "tile_px_set" not in v3  # YOLO measures body per frame: no single set


def test_yolo_v3_never_invents():
    """Adversarial B1/m2: an empty manifest gains no geometry."""
    v3 = training_geometry_from_yolo_manifest({})
    assert v3 == {"fragment_policy": "drop"}


def test_yolo_v3_does_not_mutate_input():
    manifest = _yolo_manifest()
    snapshot = dict(manifest)
    training_geometry_from_yolo_manifest(manifest)
    assert manifest == snapshot


SAM3_MULTI = {
    "geometry_mode": "auto_object",
    "tile_px_set": [[1940, 1940], [970, 970]],
    "object_tile_fractions": [0.0275, 0.055],
    "prefill_object_tile_fraction": 0.04125,
    "prefill_tile_px": [970, 970],
    "full_frame_mix": False,
    "scale_range_px": [970, 1940],
    "reference_body_px": 53.4,
    "tile_overlap": 0.25,
    "min_retained_area_frac": 0.3,
    "fragment_counts": {"x": 1},
    "scale_counts": {"tile:970x970": 10},
}


def test_sam3_multiscale_shape():
    v3 = training_geometry_from_sam3_manifest(SAM3_MULTI, imgsz=1008)
    assert v3["object_tile_fractions"] == [0.0275, 0.055]
    assert v3["prefill_object_tile_fraction"] == 0.04125
    assert "object_tile_fraction" not in v3  # SAM3 multi-scale convention
    assert v3["tile_px_set"] == [[1940, 1940], [970, 970]]
    assert v3["overlap"] == 0.25
    assert v3["min_area_ratio"] == 0.3
    assert v3["geometry_mode"] == "auto_object"
    assert v3["trained_body_px"] == v3["reference_body_px"] == 53.4
    assert v3["fragment_policy"] == "crowd"
    assert v3["imgsz"] == 1008
    assert v3["full_frame_mix"] is False and v3["scale_range_px"] == [970, 1940]
    for legacy in (
        "tile_overlap",
        "min_retained_area_frac",
        "prefill_tile_px",
        "fragment_counts",
        "scale_counts",
    ):
        assert legacy not in v3


def test_sam3_single_scale_keeps_bare_scalar():
    v3 = training_geometry_from_sam3_manifest(
        {
            "tile_px": [971, 971],
            "object_tile_fraction": 0.055,
            "reference_body_px": 53.4,
        },
        imgsz=1008,
    )
    assert v3["object_tile_fraction"] == v3["prefill_object_tile_fraction"] == 0.055
    assert v3["tile_px_set"] == [[971, 971]]


def test_sam3_never_invents():
    """Adversarial B1: absent mode/overlap/min-area stay absent."""
    v3 = training_geometry_from_sam3_manifest({"tile_px": 971}, imgsz=1008)
    assert v3 == {
        "fragment_policy": "crowd",
        "imgsz": 1008,
        "tile_px_set": [[971, 971]],
    }


@pytest.mark.parametrize(
    "hostile",
    [
        {"tile_px_set": 971},
        {"tile_px_set": [["a", "b"]]},
        {"tile_px_set": [[float("nan"), 1]]},
        {"tile_px_set": [[float("inf"), 1]]},
        {"tile_px": True},
        {"object_tile_fractions": "x"},
        {"tile_px_set": [[10**400, 1]]},
        {"tile_px": 10**400},
    ],
)
def test_sam3_builder_never_raises(hostile):
    training_geometry_from_sam3_manifest(hostile, imgsz=1008)


def test_schema_v3_and_family():
    assert SLICE_META_SCHEMA_VERSION == 3
    doc = merge_training_geometry(None, {"overlap": 0.2}, model_family="sam3")
    assert doc["schema_version"] == 3 and doc["model_family"] == "sam3"
    assert normalized_slice_meta(doc)["model_family"] == "sam3"
    assert (
        normalized_slice_meta({"overlap": 0.2})["model_family"] == "yolo"
    )  # v1/v2 were YOLO-only


def test_flat_doc_geometry_excludes_envelope():
    assert training_geometry(
        {"overlap": 0.2, "schema_version": 1, "model_family": "yolo"}
    ) == {"overlap": 0.2}


def test_merge_preserves_profiles():
    existing = {
        "schema_version": 2,
        "training_geometry": {"overlap": 0.2},
        "primary_profile_id": "p-1",
        "profiles": [
            {
                "id": "p-1",
                "name": "Balanced",
                "note": "",
                "settings": {"overlap": 0.3},
                "measurement": {"f1": 0.9},
            }
        ],
    }
    doc = merge_training_geometry(existing, {"overlap": 0.25}, model_family="yolo")
    assert doc["profiles"] == existing["profiles"]
    assert doc["primary_profile_id"] == "p-1"
    assert doc["training_geometry"] == {"overlap": 0.25}


@pytest.mark.parametrize("bad_fraction", [None, 0, 1.5])
def test_sam3_never_invents_a_fraction(bad_fraction):
    """Review M2: no fraction stamped from canonicalize's fallback ladder."""
    v3 = training_geometry_from_sam3_manifest(
        {
            "tile_px": [971, 971],
            "object_tile_fraction": bad_fraction,
            "reference_body_px": 55.4,
        },
        imgsz=1008,
    )
    assert "object_tile_fraction" not in v3
    assert "prefill_object_tile_fraction" not in v3
    assert "object_tile_fractions" not in v3
    assert v3["tile_px_set"] == [[971, 971]]


def test_sam3_stated_prefill_alone_is_stamped():
    v3 = training_geometry_from_sam3_manifest(
        {"prefill_object_tile_fraction": 0.04}, imgsz=1008
    )
    assert v3["prefill_object_tile_fraction"] == 0.04


@pytest.mark.parametrize("body", [0, 0.0, -1.0])
def test_yolo_trained_body_px_only_when_positive(body):
    """Review m4: a 0 body is 'not measured', not a trained body size."""
    v3 = training_geometry_from_yolo_manifest(_yolo_manifest(reference_body_px=body))
    assert "trained_body_px" not in v3


def test_sam3_does_not_stamp_keep_empty_tiles():
    """Review m5: publish never forwards it, so it is not part of the stamp."""
    v3 = training_geometry_from_sam3_manifest({"keep_empty_tiles": True}, imgsz=1008)
    assert "keep_empty_tiles" not in v3
