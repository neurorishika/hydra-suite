import logging

import numpy as np
import pytest

from hydra_suite.utils import tiling_spec
from hydra_suite.utils.tiling_spec import TilingSpec, canonicalize


@pytest.fixture(autouse=True)
def _fresh_warn_once_registry():
    """Warnings fire once per (field, value) per PROCESS; reset so assertions
    on caplog do not depend on test order."""
    tiling_spec._WARNED.clear()
    yield


def test_trackerkit_profile_settings():
    canonical, extras = canonicalize(
        {
            "enabled": True,
            "geometry_mode": "auto_object",
            "overlap": 0.3,
            "object_tile_fraction": 0.12,
            "trained_body_px": 40.0,
            "slice_width": 0,
            "slice_height": 0,
            "confidence_threshold": 0.4,
            "merge_policy": "nms",
            "merge_metric": "iou",
            "merge_threshold": 0.6,
            "merge_backend": "cv2",
        }
    )
    assert canonical["object_tile_fractions"] == (0.12,)
    assert canonical["reference_body_px"] == 40.0
    assert canonical["overlap"] == 0.3
    assert extras == {"confidence_threshold": 0.4, "merge_backend": "cv2"}


def test_advanced_config_prefixed_keys():
    canonical, _ = canonicalize(
        {
            "slice_overlap": 0.25,
            "slice_object_tile_fraction": 0.2,
            "slice_trained_body_px": 33.0,
            "slice_merge_threshold": 0.4,
            "slice_width": 512,
        }
    )
    assert canonical["overlap"] == 0.25
    assert canonical["object_tile_fractions"] == (0.2,)
    assert canonical["reference_body_px"] == 33.0
    assert canonical["merge_threshold"] == 0.4
    assert canonical["slice_width"] == 512  # slice_width is itself canonical


def test_target_sizes_with_imgsz_is_bit_exact_with_trackerkit():
    geometry = {
        "target_sizes": [32, 64, 96, 128],
        "imgsz": 640,
        "object_tile_fraction": 0.1,
    }
    canonical, extras = canonicalize(geometry)
    assert canonical["object_tile_fractions"] == (0.05, 0.1, 0.15, 0.2)
    # core/inference/slice_meta._training_values computes exactly this:
    assert canonical["operating_fraction"] == max(
        0.01, min(0.9, float(np.median(np.asarray([32.0, 64.0, 96.0, 128.0]))) / 640)
    )
    assert extras == {"imgsz": 640}
    assert "target_sizes" not in extras


def test_target_sizes_without_imgsz_needs_explicit_denominator():
    with pytest.raises(ValueError, match="imgsz"):
        canonicalize({"target_sizes": [64, 128]})
    canonical, _ = canonicalize({"target_sizes": [64, 128]}, legacy_px_imgsz=640.0)
    assert canonical["object_tile_fractions"] == (0.1, 0.2)
    # Without a stated imgsz TrackerKit ignores target_sizes for the operating
    # value and falls back to object_tile_fraction, else its 0.15 literal.
    assert canonical["operating_fraction"] == 0.15
    canonical, _ = canonicalize(
        {"target_sizes": [64, 128], "object_tile_fraction": 0.12}, legacy_px_imgsz=640.0
    )
    assert canonical["operating_fraction"] == 0.12


def test_operating_mirrors_legacy_reader_on_bare_scalar_edge_cases():
    # slice_meta._training_values clamps 0 to 0.01 and maps unparseable to 0.15.
    assert canonicalize({"object_tile_fraction": 0})[0]["operating_fraction"] == 0.01
    assert canonicalize({"object_tile_fraction": "x"})[0]["operating_fraction"] == 0.15
    assert canonicalize({"object_tile_fraction": 2.0})[0]["operating_fraction"] == 0.9


def test_fraction_set_precedence():
    canonical, _ = canonicalize(
        {
            "object_tile_fractions": [0.03, 0.06],
            "target_size_fractions": [0.5],
            "target_sizes": [64],
            "object_tile_fraction": 0.2,
            "imgsz": 640,
        }
    )
    assert canonical["object_tile_fractions"] == (0.03, 0.06)


def test_empty_fraction_set_falls_through():
    canonical, _ = canonicalize(
        {"target_size_fractions": [], "object_tile_fraction": 0.1}
    )
    assert canonical["object_tile_fractions"] == (0.1,)


def test_sam3_sidecar_multiscale():
    canonical, extras = canonicalize(
        {
            "object_tile_fractions": [0.0275, 0.055],
            "prefill_object_tile_fraction": 0.055,
            "train_tile_px_set": [[1940, 1940], [970, 970]],
            "reference_body_px": 53.4,
            "imgsz": 1008,
        }
    )
    assert canonical["object_tile_fractions"] == (0.0275, 0.055)
    assert canonical["operating_fraction"] == 0.055
    assert "train_tile_px_set" in extras


def test_sam3_build_manifest_names():
    canonical, _ = canonicalize({"tile_overlap": 0.25, "min_retained_area_frac": 0.3})
    assert canonical["overlap"] == 0.25
    assert canonical["min_area_ratio"] == 0.3


def test_escalation_request_names():
    canonical, _ = canonicalize(
        {"tile_fraction": 0.05, "overlap": 0.5, "merge_iou": 0.4}
    )
    assert canonical["object_tile_fractions"] == (0.05,)
    assert canonical["merge_threshold"] == 0.4
    assert canonical["merge_metric"] == "polygon_iou"


@pytest.mark.parametrize("value", [None, 0, 0.0])
def test_escalation_full_frame_means_disabled(value):
    canonical, _ = canonicalize({"tile_fraction": value})
    assert canonical["enabled"] is False
    assert "object_tile_fractions" not in canonical


def test_explicit_merge_metric_beats_merge_iou_implication():
    canonical, _ = canonicalize({"merge_iou": 0.4, "merge_metric": "iou"})
    assert canonical["merge_metric"] == "iou"


def test_lenient_read_clamps_and_warns(caplog):
    caplog.set_level(logging.WARNING)
    canonical, _ = canonicalize(
        {
            "overlap": 0.95,
            "geometry_mode": "bogus",
            "merge_policy": "nmm",
            "object_tile_fraction": 1.5,
            "reference_body_px": float("nan"),
            "min_area_ratio": 2.0,
            "merge_threshold": -1,
        }
    )
    assert canonical["overlap"] == 0.9
    assert "geometry_mode" not in canonical
    assert canonical["merge_policy"] == "greedy_nmm"
    assert "object_tile_fractions" not in canonical
    assert "reference_body_px" not in canonical
    assert canonical["min_area_ratio"] == 1.0
    assert canonical["merge_threshold"] == 0.0
    assert "overlap" in caplog.text and "geometry_mode" in caplog.text


def test_overlap_axes_disagreement_takes_width(caplog):
    caplog.set_level(logging.WARNING)
    canonical, _ = canonicalize(
        {"overlap_width_ratio": 0.2, "overlap_height_ratio": 0.3}
    )
    assert canonical["overlap"] == 0.2
    assert "overlap_height_ratio" in caplog.text


def test_from_mapping_lenient_never_raises_on_legacy_values():
    spec = TilingSpec.from_mapping(
        {"overlap": 0.95, "merge_policy": "nmm"}, backend="yolo_infer"
    )
    assert spec.overlap == 0.9
    assert spec.merge_policy == "greedy_nmm"
    assert spec.object_tile_fractions == (0.15,)  # backend default filled


def test_from_mapping_without_backend_does_not_invent_fractions():
    spec = TilingSpec.from_mapping({"geometry_mode": "auto_model"})
    assert spec.object_tile_fractions == ()


@pytest.mark.parametrize("imgsz", [0.5, float("inf"), float("nan"), "x", -640, True])
def test_unusable_imgsz_never_divides(imgsz):
    """Adversarial M4: no ZeroDivision/Overflow on hostile imgsz."""
    canonical, _ = canonicalize(
        {"target_sizes": [64], "imgsz": imgsz}, legacy_px_imgsz=640.0
    )
    assert canonical["object_tile_fractions"] == (0.1,)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("false", False),
        ("0", False),
        ("true", True),
        (0, False),
        (1, True),
        (True, True),
    ],
)
def test_enabled_parses_strings(raw, expected):
    assert canonicalize({"enabled": raw})[0]["enabled"] is expected


def test_repeated_legacy_warning_logged_once(caplog):
    caplog.set_level(logging.WARNING)
    for _ in range(5):
        canonicalize({"overlap": 0.97})
    assert caplog.text.count("overlap=0.97") == 1


def test_none_and_empty_mapping():
    assert canonicalize(None) == ({}, {})
    assert canonicalize({}) == ({}, {})
