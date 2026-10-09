import logging

import pytest

from hydra_suite.utils.slice_geometry import tile_size_for_mode
from hydra_suite.utils.tiling_spec import (
    DEFAULT_OVERLAP,
    FRACTION_MAX,
    FRACTION_MIN,
    OVERLAP_MAX,
    Sourced,
    TilingSpec,
    resolve_object_tile_fractions,
    resolve_operating_fraction,
    resolve_overlap,
    resolve_reference_body_px,
    resolve_tile_size,
)


def test_body_px_priority_chain():
    assert resolve_reference_body_px(
        override=50, dataset_median=40, stamped=30, tracker_reference=20
    ) == Sourced(50.0, "override")
    assert resolve_reference_body_px(
        dataset_median=40, stamped=30, tracker_reference=20
    ) == Sourced(40.0, "dataset")
    assert resolve_reference_body_px(stamped=30, tracker_reference=20) == Sourced(
        30.0, "stamped"
    )
    assert resolve_reference_body_px(tracker_reference=20) == Sourced(20.0, "user")
    assert resolve_reference_body_px() == Sourced(0.0, "default")


@pytest.mark.parametrize("bad", [0, -5, float("nan"), "x", None, True])
def test_body_px_skips_unusable_values(bad):
    assert resolve_reference_body_px(override=bad, stamped=30) == Sourced(
        30.0, "stamped"
    )


def test_operating_fraction_chain():
    assert resolve_operating_fraction(
        backend="yolo_infer", profile=0.12, stamped_operating=0.2
    ) == Sourced(0.12, "profile")
    assert resolve_operating_fraction(
        backend="yolo_infer", stamped_operating=0.2
    ) == Sourced(0.2, "stamped")
    assert resolve_operating_fraction(
        backend="yolo_infer", stamped_fractions=(0.05, 0.1, 0.15, 0.2)
    ) == Sourced(pytest.approx(0.125), "stamped")
    assert resolve_operating_fraction(
        backend="sam3", stamped_fractions=(0.05, 0.1, 0.15, 0.2)
    ) == Sourced(pytest.approx(0.125), "stamped")
    assert resolve_operating_fraction(backend="yolo_infer") == Sourced(0.15, "default")
    assert resolve_operating_fraction(backend="sam2") == Sourced(None, "default")


def test_fraction_set_chain():
    assert resolve_object_tile_fractions(
        backend="yolo_train", user=[0.1, 0.2], profile=(0.3,), stamped=(0.4,)
    ) == Sourced((0.1, 0.2), "user")
    assert resolve_object_tile_fractions(
        backend="yolo_train", profile=(0.3,), stamped=(0.4,)
    ) == Sourced((0.3,), "profile")
    assert resolve_object_tile_fractions(
        backend="yolo_train", stamped=(0.4,)
    ) == Sourced((0.4,), "stamped")
    assert resolve_object_tile_fractions(backend="yolo_train") == Sourced(
        (0.05, 0.10, 0.15, 0.20), "default"
    )
    assert resolve_object_tile_fractions(
        backend="yolo_train", user=[], stamped=["x", None]
    ) == Sourced((0.05, 0.10, 0.15, 0.20), "default")
    assert resolve_object_tile_fractions(backend="sam2") == Sourced((), "default")


def test_overlap_chain():
    assert resolve_overlap(override=0.4, saved=0.3, fractions=(0.1,)) == Sourced(
        0.4, "override"
    )
    assert resolve_overlap(saved=0.3, fractions=(0.1,)) == Sourced(0.3, "user")
    assert resolve_overlap(saved=0.2, fractions=(0.5,)) == Sourced(
        0.2, "user"
    )  # saved is never re-derived
    assert resolve_overlap(fractions=(0.05, 0.15)) == Sourced(
        0.2, "derived"
    )  # 0.15 + 0.05
    assert resolve_overlap(fractions=(0.88,)) == Sourced(0.9, "derived")  # ceiling
    assert resolve_overlap() == Sourced(DEFAULT_OVERLAP, "default")


def test_derived_overlap_reproduces_trackerkit_default_exactly():
    assert resolve_overlap(fractions=(0.15,)).value == 0.2


def test_derived_overlap_guarantees_whole_animal_in_some_tile():
    """m1: against the REALIZED tile (rounded, inside the planner's [64, 4096]
    clamp), the overlap strip is at least one body wide."""
    checked = 0
    for body in (8.0, 20.0, 53.4, 100.0, 300.0):
        for frac in (0.03, 0.055, 0.1, 0.15, 0.3, 0.5):
            if not 64 <= body / frac <= 4096:
                continue  # outside the window the guarantee is stated for
            overlap = resolve_overlap(fractions=(frac,)).value
            tile_w, _ = tile_size_for_mode(
                geometry_mode="auto_object",
                imgsz=640,
                reference_body_px=body,
                object_tile_fraction=frac,
                slice_width=0,
                slice_height=0,
            )
            assert overlap * tile_w >= body, (body, frac, tile_w, overlap)
            checked += 1
    assert checked >= 20


def test_tile_size_matches_planner():
    spec = TilingSpec(enabled=True, geometry_mode="auto_object", reference_body_px=60.0)
    expected = tile_size_for_mode(
        geometry_mode="auto_object",
        imgsz=640,
        reference_body_px=60.0,
        object_tile_fraction=0.15,
        slice_width=0,
        slice_height=0,
    )
    assert resolve_tile_size(spec, imgsz=640, fraction=0.15) == Sourced(
        expected, "derived"
    )


def test_tile_size_custom_is_user():
    spec = TilingSpec(geometry_mode="custom", slice_width=512, slice_height=384)
    assert resolve_tile_size(spec, imgsz=640, fraction=None) == Sourced(
        (512, 384), "user"
    )
    spec0 = TilingSpec(geometry_mode="custom")
    assert resolve_tile_size(spec0, imgsz=640, fraction=None) == Sourced(
        (640, 640), "derived"
    )


# --- S1 fix wave: clamp-with-warning (m2) and drift pins ----------------------


@pytest.fixture(autouse=True)
def _fresh_warn_once_registry():
    from hydra_suite.utils import tiling_spec

    tiling_spec.reset_warnings()
    yield


def test_overlap_out_of_range_is_clamped_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    assert resolve_overlap(saved=1.0) == Sourced(OVERLAP_MAX, "user")
    assert resolve_overlap(override=-0.1, saved=0.3) == Sourced(0.0, "override")
    assert "overlap" in caplog.text
    assert resolve_overlap(override="x", saved=0.3) == Sourced(0.3, "user")


def test_operating_fraction_out_of_range_is_clamped_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    assert resolve_operating_fraction(backend="yolo_infer", profile=1.5) == Sourced(
        FRACTION_MAX, "profile"
    )
    assert resolve_operating_fraction(
        backend="yolo_infer", stamped_operating=0.0
    ) == Sourced(FRACTION_MIN, "stamped")
    assert resolve_operating_fraction(
        backend="yolo_infer", stamped_fractions=(0.0, 2.0)
    ) == Sourced(pytest.approx((FRACTION_MIN + FRACTION_MAX) / 2), "stamped")
    assert "fraction" in caplog.text
    assert resolve_operating_fraction(backend="yolo_infer", profile="x") == Sourced(
        0.15, "default"
    )


def test_fraction_set_out_of_range_is_clamped_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    assert resolve_object_tile_fractions(
        backend="yolo_train", stamped=[0, 2.0]
    ) == Sourced((FRACTION_MIN, FRACTION_MAX), "stamped")
    assert resolve_object_tile_fractions(
        backend="yolo_train", user=["x", 0.1]
    ) == Sourced((0.1,), "user")
    assert "fraction" in caplog.text


def _auto_object_size(body, frac):
    return tile_size_for_mode(
        geometry_mode="auto_object",
        imgsz=640,
        reference_body_px=body,
        object_tile_fraction=frac,
        slice_width=0,
        slice_height=0,
    )


def test_fraction_bounds_pinned_to_planner_clamp():
    """Code-review 2: FRACTION_MIN/MAX are tile_size_for_mode's clamp."""
    body = 30.0  # body / FRACTION_MIN = 3000 px, inside the [64, 4096] tile clamp
    assert _auto_object_size(body, FRACTION_MIN) == _auto_object_size(
        body, FRACTION_MIN / 2
    )
    assert _auto_object_size(body, FRACTION_MIN) != _auto_object_size(
        body, FRACTION_MIN * 1.1
    )
    body = 100.0  # body / FRACTION_MAX = 111 px
    assert _auto_object_size(body, FRACTION_MAX) == _auto_object_size(
        body, (FRACTION_MAX + 1.0) / 2
    )
    assert _auto_object_size(body, FRACTION_MAX) != _auto_object_size(
        body, FRACTION_MAX * 0.9
    )


@pytest.mark.parametrize(
    "first",
    ["hydra_suite.utils.tiling_resolve", "hydra_suite.utils.tiling_spec"],
)
def test_split_modules_import_in_either_order(first):
    """resolve_* live in tiling_resolve and are re-exported by tiling_spec."""
    import subprocess
    import sys

    code = (
        f"import {first}\n"
        "from hydra_suite.utils import tiling_resolve, tiling_spec\n"
        "from hydra_suite.utils.tiling_spec import resolve_overlap\n"
        "assert resolve_overlap is tiling_resolve.resolve_overlap\n"
        "assert resolve_overlap(fractions=(0.15,)).value == 0.2\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
