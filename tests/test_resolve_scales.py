"""Task 2: one shared scale-set resolver, no behaviour change.

``resolve_scales`` replaces the body of
``training.sliced_dataset._tile_sizes_for_params``. After that refactor the
live function IS the shared one, so comparing them would be tautological --
``_frozen_oracle`` below is a verbatim copy of the pre-refactor body, taken at
`0d4d4cae`, and it is the actual oracle. Do not "simplify" it to call the
shared code.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from hydra_suite.detectkit.config.training import SliceTrainingConfig
from hydra_suite.detectkit.gui.models import SliceTrainingSettings
from hydra_suite.utils.slice_geometry import (
    resolve_scales,
    target_fractions_from,
    tile_size_for_mode,
)


def _frozen_oracle(params, reference_body_px):
    """Verbatim pre-refactor ``_tile_sizes_for_params`` (sliced_dataset.py)."""
    if (
        params.geometry_mode == "auto_object"
        and reference_body_px > 0
        and params.target_sizes
    ):
        sizes: list[tuple[int, int]] = []
        for target in params.target_sizes:
            frac = max(0.01, min(0.9, float(target) / max(1, params.imgsz)))
            w, h = tile_size_for_mode(
                geometry_mode="auto_object",
                imgsz=params.imgsz,
                reference_body_px=reference_body_px,
                object_tile_fraction=frac,
                slice_width=0,
                slice_height=0,
            )
            if (w, h) not in sizes:
                sizes.append((w, h))
        if sizes:
            return sizes
    w, h = tile_size_for_mode(
        geometry_mode=params.geometry_mode,
        imgsz=params.imgsz,
        reference_body_px=reference_body_px,
        object_tile_fraction=params.object_tile_fraction,
        slice_width=params.slice_width,
        slice_height=params.slice_height,
    )
    return [(w, h)]


_TARGET_SETS = [
    [],
    [32.0, 64.0, 96.0, 128.0],
    [64.0],
    [64.0, 64.0],  # exact duplicate
    [64.0, 64.4],  # collapses only after rounding
    [1.0, 4096.0],  # both outside the [0.01, 0.9] fraction clamp
    [-8.0, 64.0],  # malformed negative -> clamped, not dropped
]
_REFS = [0.0, 0.4, 12.0, 40.0, 512.0]
_IMGSZ = [0, 1, 640, 1008]
_MODES = ["auto_object", "auto_model", "custom"]


@pytest.mark.parametrize(
    "targets,ref,imgsz,mode",
    list(itertools.product(_TARGET_SETS, _REFS, _IMGSZ, _MODES)),
)
def test_resolve_scales_matches_frozen_oracle(targets, ref, imgsz, mode):
    params = SimpleNamespace(
        geometry_mode=mode,
        imgsz=imgsz,
        object_tile_fraction=0.10,
        slice_width=0,
        slice_height=0,
        target_sizes=list(targets),
    )
    fractions = [max(0.01, min(0.9, float(t) / max(1, imgsz))) for t in targets]
    assert resolve_scales(
        geometry_mode=mode,
        imgsz=imgsz,
        reference_body_px=ref,
        fractions=fractions,
        object_tile_fraction=0.10,
        slice_width=0,
        slice_height=0,
    ) == _frozen_oracle(params, ref)


def test_resolve_scales_dedupes_preserving_first_seen_order():
    sizes = resolve_scales(
        geometry_mode="auto_object",
        imgsz=640,
        reference_body_px=40.0,
        fractions=[0.20, 0.05, 0.20, 0.10],
        object_tile_fraction=0.10,
        slice_width=0,
        slice_height=0,
    )
    assert sizes == [(200, 200), (800, 800), (400, 400)]


def test_resolve_scales_takes_no_pixel_list():
    """SAM3 must be structurally unable to route a legacy pixel set in."""
    import inspect

    names = set(inspect.signature(resolve_scales).parameters)
    assert "target_sizes" not in names and "fractions" in names


def test_legacy_pixel_denominator_is_required_and_shifts_by_1_575x():
    """The 640-vs-1008 trap, made explicit and tested.

    Legacy absolute ``target_sizes`` are anchored to a 640px input. Dividing
    them by SAM3's 1008px input instead silently rescales every tile by
    1008/640 = 1.575x, so the denominator has no default.
    """
    import inspect

    denominator = inspect.signature(target_fractions_from).parameters[
        "legacy_pixel_denominator"
    ]
    assert denominator.default is inspect.Parameter.empty
    assert denominator.kind is inspect.Parameter.KEYWORD_ONLY

    at_640 = target_fractions_from(
        fractions=(), legacy_pixel_sizes=(64.0,), legacy_pixel_denominator=640.0
    )
    at_1008 = target_fractions_from(
        fractions=(), legacy_pixel_sizes=(64.0,), legacy_pixel_denominator=1008.0
    )
    assert at_640 == [0.1]
    assert at_640[0] / at_1008[0] == pytest.approx(1008.0 / 640.0)


def test_shared_target_fractions_filters_malformed_before_falling_back():
    """ONE behaviour: filter to (0, 1], then fall back if nothing survives."""
    assert target_fractions_from(
        fractions=(0.05, 0.0, 1.5, -0.2, 0.20),
        legacy_pixel_sizes=(64.0,),
        legacy_pixel_denominator=640.0,
    ) == [0.05, 0.20]
    assert target_fractions_from(
        fractions=(0.0, 2.0),
        legacy_pixel_sizes=(64.0, -8.0),
        legacy_pixel_denominator=640.0,
    ) == [0.1]
    assert (
        target_fractions_from(
            fractions=(), legacy_pixel_sizes=(), legacy_pixel_denominator=640.0
        )
        == []
    )


def test_config_and_gui_target_fractions_agree_on_malformed_input():
    config = SliceTrainingConfig(target_size_fractions=(0.0, 2.0), target_sizes=(64.0,))
    gui = SliceTrainingSettings(target_size_fractions=[0.0, 2.0], target_sizes=[64.0])
    assert config.target_fractions() == gui.target_fractions() == [0.1]
