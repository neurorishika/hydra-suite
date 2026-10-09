import json
import math

import numpy as np
import pytest

from hydra_suite.utils.slice_geometry import resolve_scales
from hydra_suite.utils.tiling_spec import (
    BACKEND_DEFAULTS,
    TilingSpec,
    operating_fraction,
)


def test_default_spec_is_valid_and_disabled():
    spec = TilingSpec()
    assert spec.enabled is False
    assert spec.geometry_mode == "auto_model"
    assert spec.object_tile_fractions == ()
    assert spec.overlap is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"geometry_mode": "bogus"},
        {"object_tile_fractions": (0.0,)},
        {"object_tile_fractions": (1.5,)},
        {"object_tile_fractions": (math.nan,)},
        {"overlap": 0.95},
        {"overlap": -0.1},
        {"reference_body_px": -1.0},
        {"slice_width": -1},
        {"min_area_ratio": 1.5},
        {"fragment_policy": "keep"},
        {"merge_policy": "nmm"},
        {"merge_metric": "dice"},
        {"merge_threshold": 2.0},
    ],
)
def test_direct_construction_is_strict(kwargs):
    with pytest.raises(ValueError):
        TilingSpec(**kwargs)


def test_fractions_coerced_to_float_tuple():
    spec = TilingSpec(object_tile_fractions=[0.1, 0.2])
    assert spec.object_tile_fractions == (0.1, 0.2)
    assert all(isinstance(f, float) for f in spec.object_tile_fractions)


def test_backend_defaults_table():
    assert BACKEND_DEFAULTS["yolo_train"].object_tile_fractions == (
        0.05,
        0.10,
        0.15,
        0.20,
    )
    assert BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions == (0.15,)
    assert BACKEND_DEFAULTS["yolo_infer"].geometry_mode == "auto_model"
    assert BACKEND_DEFAULTS["sam3"].object_tile_fractions == (0.055,)
    assert BACKEND_DEFAULTS["sam3"].fragment_policy == "crowd"
    assert BACKEND_DEFAULTS["sam3"].merge_metric == "polygon_iou"
    assert BACKEND_DEFAULTS["sam2"].object_tile_fractions == ()


def test_defaults_constructor_uses_table():
    spec = TilingSpec.defaults("sam3")
    assert spec.object_tile_fractions == (0.055,)
    assert spec.fragment_policy == "crowd"
    assert spec.enabled is False


def test_operating_fraction_is_np_median():
    # TrackerKit today: median(target_sizes)/imgsz -> 80/640 for [32,64,96,128];
    # SAM3 dataset_build stamps prefill_object_tile_fraction with the same rule.
    assert operating_fraction((0.05, 0.10, 0.15, 0.20)) == pytest.approx(0.125)
    assert operating_fraction((0.0275, 0.055)) == pytest.approx(0.04125)
    assert operating_fraction((0.055,)) == 0.055


def test_operating_fraction_clamps_and_handles_empty():
    assert operating_fraction(()) is None
    assert operating_fraction((0.005,)) == 0.01
    assert operating_fraction((0.95,)) == 0.9


def test_training_tile_sizes_delegates_to_resolve_scales():
    spec = TilingSpec(
        enabled=True,
        geometry_mode="auto_object",
        object_tile_fractions=(0.1, 0.2),
        reference_body_px=50.0,
    )
    expected = resolve_scales(
        geometry_mode="auto_object",
        imgsz=640,
        reference_body_px=50.0,
        fractions=(0.1, 0.2),
        object_tile_fraction=0.15000000000000002,
        slice_width=0,
        slice_height=0,
    )
    assert spec.training_tile_sizes(640) == expected == [(500, 500), (250, 250)]


def test_to_mapping_is_canonical_and_json_safe():
    mapping = TilingSpec(object_tile_fractions=(0.1,), overlap=0.2).to_mapping()
    assert mapping["object_tile_fractions"] == [0.1]
    assert mapping["overlap"] == 0.2
    assert set(mapping) == {
        "enabled",
        "geometry_mode",
        "object_tile_fractions",
        "reference_body_px",
        "slice_width",
        "slice_height",
        "overlap",
        "min_area_ratio",
        "fragment_policy",
        "merge_policy",
        "merge_metric",
        "merge_threshold",
    }


# --- S1 fix wave: strict typing (D5) and from_canonical -----------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_body_px": "50"},
        {"reference_body_px": True},
        {"overlap": "0.2"},
        {"overlap": False},
        {"min_area_ratio": "0.3"},
        {"merge_threshold": True},
        {"slice_width": 512.0},
        {"slice_width": "512"},
        {"slice_height": True},
        {"enabled": "false"},
        {"enabled": 1},
        {"object_tile_fractions": 0.1},
        {"object_tile_fractions": "0.1"},
        {"object_tile_fractions": (0.1, "0.2")},
        {"object_tile_fractions": (True,)},
        {"object_tile_fractions": (None,)},
        {"geometry_mode": ["auto_model"]},
    ],
)
def test_strict_types_raise_value_error_not_type_error(kwargs):
    with pytest.raises(ValueError):
        TilingSpec(**kwargs)


def test_numpy_inputs_coerced_to_python_and_json_safe():
    spec = TilingSpec(
        enabled=np.bool_(True),
        object_tile_fractions=np.array([0.1, 0.2]),
        reference_body_px=np.float32(50.0),
        slice_width=np.int64(512),
        slice_height=np.int32(384),
        overlap=np.float64(0.25),
        min_area_ratio=np.float64(0.3),
        merge_threshold=np.float32(0.5),
    )
    assert spec.enabled is True
    assert type(spec.reference_body_px) is float
    assert type(spec.overlap) is float
    assert type(spec.slice_width) is int and spec.slice_width == 512
    assert all(type(f) is float for f in spec.object_tile_fractions)
    json.dumps(spec.to_mapping())


def test_int_floats_stored_as_float():
    spec = TilingSpec(reference_body_px=50, overlap=0, merge_threshold=1)
    assert type(spec.reference_body_px) is float
    assert type(spec.overlap) is float
    assert type(spec.merge_threshold) is float


def test_from_canonical_ignores_operating_fraction():
    spec = TilingSpec.from_canonical(
        {"operating_fraction": 0.125, "object_tile_fractions": (0.1, 0.15)}
    )
    assert spec.object_tile_fractions == (0.1, 0.15)
    assert "operating_fraction" not in spec.to_mapping()


def test_from_canonical_backend_overlay():
    spec = TilingSpec.from_canonical({"overlap": 0.3}, backend="sam3")
    assert spec.overlap == 0.3
    assert spec.object_tile_fractions == (0.055,)
    assert spec.fragment_policy == "crowd"


def test_from_canonical_full_frame_escalation_keeps_backend_fractions():
    from hydra_suite.utils.tiling_spec import canonicalize

    canonical, _ = canonicalize({"tile_fraction": 0})
    spec = TilingSpec.from_canonical(canonical, backend="sam3")
    assert spec.enabled is False
    assert spec.object_tile_fractions == (0.055,)
