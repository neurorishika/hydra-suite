import logging

import numpy as np
import pytest

from hydra_suite.utils import tiling_spec
from hydra_suite.utils.tiling_spec import TilingSpec, canonicalize


@pytest.fixture(autouse=True)
def _fresh_warn_once_registry():
    """Warnings fire once per (field, value) per PROCESS; reset so assertions
    on caplog do not depend on test order."""
    tiling_spec.reset_warnings()
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


def test_target_sizes_without_imgsz_needs_explicit_denominator(caplog):
    # D1: never raise; never rescale by an unstated size -> fractions absent.
    caplog.set_level(logging.WARNING)
    canonical, _ = canonicalize({"target_sizes": [64, 128]})
    assert "object_tile_fractions" not in canonical
    assert "target_sizes" in caplog.text
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
    # Unusable imgsz -> no median/imgsz; nothing bare -> legacy 0.15 literal.
    assert canonical["operating_fraction"] == 0.15


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


# --- S1 fix wave -----------------------------------------------------------


def test_detectkit_default_settings_payload_never_raises():
    """D1: the real SliceTrainingSettings default carries target_sizes, no imgsz."""
    from hydra_suite.detectkit.gui.models import SliceTrainingSettings

    payload = SliceTrainingSettings().to_dict()
    assert payload["target_sizes"] and "imgsz" not in payload
    canonical, extras = canonicalize(payload)
    assert "object_tile_fractions" not in canonical  # never rescaled by a guess
    assert canonical["operating_fraction"] == 0.1  # bare scalar, like TrackerKit
    assert "negative_tile_fraction" in extras
    TilingSpec.from_canonical(canonical, backend="yolo_train")  # constructs


@pytest.mark.parametrize(
    "mapping",
    [
        {"imgsz": 10**400},
        {"imgsz": 10**400, "target_sizes": [64]},
        {"measured_reference_body_px": 10**400},
        {"overlap": 10**400},
        {"object_tile_fraction": 10**400},
        {"target_sizes": [10**400], "imgsz": 640},
    ],
)
def test_huge_ints_never_raise(mapping):
    """D2: float(10**400) raises OverflowError; reads stay lenient."""
    canonicalize(mapping)


def test_from_mapping_huge_int_geometry_mode():
    spec = TilingSpec.from_mapping({"slice_geometry_mode": 10**400})
    assert spec.geometry_mode == "auto_model"


def test_operating_from_sam3_params_set():
    """D3(c): a SET key beats the bare scalar for the operating scale."""
    canonical, _ = canonicalize(
        {"object_tile_fraction": 0.055, "object_tile_fractions": [0.05, 0.1]}
    )
    assert canonical["object_tile_fractions"] == (0.05, 0.1)
    assert canonical["operating_fraction"] == pytest.approx(0.075)
    assert canonical["operating_fraction"] == float(np.median([0.05, 0.1]))


def test_operating_from_slice_training_settings_shape():
    canonical, _ = canonicalize(
        {
            "object_tile_fraction": 0.1,
            "target_size_fractions": [0.05, 0.1, 0.15, 0.2],
            "target_sizes": [32, 64, 96, 128],
        }
    )
    assert canonical["object_tile_fractions"] == (0.05, 0.1, 0.15, 0.2)
    assert canonical["operating_fraction"] == pytest.approx(0.125)


def test_operating_from_advanced_config_scalar():
    """m8: the prefixed scalar also feeds the operating scale."""
    canonical, _ = canonicalize({"slice_object_tile_fraction": 0.2})
    assert canonical["operating_fraction"] == 0.2


def test_escalation_tile_px_is_custom_square():
    """D4: SemanticEscalationRequest-shaped tile_px."""
    canonical, extras = canonicalize({"tile_fraction": 0.05, "tile_px": 512})
    assert canonical["slice_width"] == canonical["slice_height"] == 512
    assert canonical["geometry_mode"] == "custom"
    assert "tile_px" not in extras


def test_tile_px_square_pair_and_explicit_mode():
    canonical, _ = canonicalize({"tile_px": [640, 640], "geometry_mode": "auto_object"})
    assert canonical["slice_width"] == canonical["slice_height"] == 640
    assert canonical["geometry_mode"] == "auto_object"


def test_tile_px_never_overrides_explicit_slice_size():
    canonical, _ = canonicalize({"tile_px": 512, "slice_width": 256})
    assert canonical["slice_width"] == 256
    assert "slice_height" not in canonical
    assert "geometry_mode" not in canonical


@pytest.mark.parametrize("value", [None, 0, "x", [640, 320], -5])
def test_tile_px_unusable_is_consumed_and_ignored(value, caplog):
    caplog.set_level(logging.WARNING)
    canonical, extras = canonicalize({"tile_px": value})
    assert "slice_width" not in canonical and "geometry_mode" not in canonical
    assert "tile_px" not in extras
    if value not in (None, 0):
        assert "tile_px" in caplog.text


def test_positive_tile_fraction_enables():
    """D6: an escalation asking for tiles is enabled; explicit enabled wins."""
    assert canonicalize({"tile_fraction": 0.05})[0]["enabled"] is True
    assert (
        canonicalize({"tile_fraction": 0.05, "enabled": False})[0]["enabled"] is False
    )


@pytest.mark.parametrize("mapping", [["a", "b"], "overlap", 42, ("x",)])
def test_non_mapping_inputs_return_empty(mapping):
    """m6: non-dict inputs are unreadable, not an exception."""
    assert canonicalize(mapping) == ({}, {})


def test_none_values_never_become_canonical():
    canonical, _ = canonicalize({"overlap": None, "geometry_mode": None})
    assert canonical == {}


def test_both_fraction_set_keys_always_consumed():
    """m7"""
    _, extras = canonicalize(
        {"object_tile_fractions": [0.1], "target_size_fractions": [0.2]}
    )
    assert extras == {}


def test_warning_registry_is_bounded():
    """m9: cleared when full, never stops warning."""
    for i in range(tiling_spec._WARN_CAP + 10):
        canonicalize({"overlap": 1.0 + i})
    assert len(tiling_spec._WARNED) <= tiling_spec._WARN_CAP


def test_warning_key_is_truncated():
    canonicalize({"geometry_mode": "x" * 5000})
    assert all(len(value) <= 200 for _, value in tiling_spec._WARNED)


def _legacy_reader_fraction(geometry):
    from hydra_suite.core.inference.slice_meta import _training_values

    return _training_values(geometry)["object_tile_fraction"]


@pytest.mark.parametrize(
    "geometry",
    [
        {"target_sizes": [32, 64, 96, 128], "imgsz": 640, "object_tile_fraction": 0.1},
        {"target_sizes": [32, 64, 96, 128], "imgsz": 1024, "object_tile_fraction": 0.1},
        {"target_sizes": [48, 96], "imgsz": 640},
        {"target_sizes": [32, 64, 96, 128], "object_tile_fraction": 0.12},
        {"object_tile_fraction": 0.12},
        {"object_tile_fraction": 0},
        {"object_tile_fraction": "x"},
        {"object_tile_fraction": 2.0},
        {"target_sizes": [64, 128], "imgsz": 0, "object_tile_fraction": 0.12},
        {"target_sizes": [64, 128], "imgsz": -640, "object_tile_fraction": 0.12},
    ],
)
def test_operating_fraction_parity_with_trackerkit_reader(geometry):
    """Code-review 3: bit-exact with slice_meta._training_values."""
    canonical, _ = canonicalize(geometry)
    assert canonical["operating_fraction"] == _legacy_reader_fraction(geometry)


def test_array_valued_legacy_fields_never_raise():
    canonicalize(
        {
            "tile_px": np.array([512, 512]),
            "overlap": 0.2,
            "overlap_width_ratio": np.array([0.2]),
            "overlap_height_ratio": np.array([0.3, 0.1]),
        }
    )
