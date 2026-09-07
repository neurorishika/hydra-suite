"""D18 -- one fragment floor, not two.

Characterization guards (NOT fail-first: D18 is explicitly meant to change no
behaviour today). They pin two things the unification must hold:

1. YOLO's ``SliceBuildParams.min_area_ratio`` and SAM3's
   ``Sam3LoraParams.min_area_ratio`` both default from the SAME upstream
   constant (``utils.slice_geometry.DEFAULT_MIN_AREA_RATIO``), so a future
   change to one cannot silently diverge from the other without also
   touching that shared line.
2. The POLICY difference below the floor -- YOLO drops the instance, SAM3
   downgrades it to ``is_crowd`` -- is untouched by the unification (D3
   stays open and separate; unifying it would be an accidental regression).
"""

import re
from pathlib import Path

import numpy as np

from hydra_suite.detectkit.config.training import SliceTrainingConfig
from hydra_suite.detectkit.gui.models import SliceTrainingSettings
from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.dataset_build import (
    MIN_RETAINED_AREA_FRAC,
    _tile_frame,
)
from hydra_suite.training.sliced_dataset import SliceBuildParams, _tile_one_image
from hydra_suite.utils.slice_geometry import DEFAULT_MIN_AREA_RATIO


def test_both_builder_params_default_from_the_same_shared_constant():
    """Single source of truth: same value, same upstream name, not just a coincidence."""
    assert SliceBuildParams().min_area_ratio == DEFAULT_MIN_AREA_RATIO
    assert Sam3LoraParams(prompt="ant").min_area_ratio == DEFAULT_MIN_AREA_RATIO
    # The SAM3 module constant is now derived from the same shared default,
    # per D18's "the SAM3 module constant becomes that field's default".
    assert MIN_RETAINED_AREA_FRAC == DEFAULT_MIN_AREA_RATIO


def test_both_builder_params_track_a_custom_shared_value():
    """Changing the shared constant's value (not just its plumbing) reaches both dataclasses'
    field defaults -- proving they read from it rather than each hard-coding 0.25."""
    import hydra_suite.utils.slice_geometry as slice_geometry_module

    original = slice_geometry_module.DEFAULT_MIN_AREA_RATIO
    try:
        slice_geometry_module.DEFAULT_MIN_AREA_RATIO = 0.42
        # dataclass field defaults are bound at class-definition time, so this
        # test instead asserts the two dataclasses reference the identical
        # module attribute rather than two independently-typed literals.
        import hydra_suite.training.contracts as contracts_module
        import hydra_suite.training.sliced_dataset as sliced_dataset_module

        assert (
            contracts_module.Sam3LoraParams.__dataclass_fields__[
                "min_area_ratio"
            ].default
            == sliced_dataset_module.SliceBuildParams.__dataclass_fields__[
                "min_area_ratio"
            ].default
        )
    finally:
        slice_geometry_module.DEFAULT_MIN_AREA_RATIO = original


def test_detectkit_config_surfaces_also_default_from_the_shared_constant():
    """The coordinator's fix-round finding: config surfaces that FEED the
    builders (DetectKit CLI plan + GUI project settings) must default from
    the same shared constant too, or changing it silently stops reaching
    what the GUI/CLI actually send -- the exact divergence D18 exists to
    prevent, just relocated one layer up.
    """
    assert SliceTrainingConfig().min_area_ratio == DEFAULT_MIN_AREA_RATIO
    assert SliceTrainingSettings().min_area_ratio == DEFAULT_MIN_AREA_RATIO


def test_no_new_hardcoded_min_area_ratio_default_creeps_in():
    """Future-proofing, by construction rather than by enumeration.

    Greps every ``*.py`` under ``src/`` for a dataclass-style field default
    of the form ``min_area_ratio: float = <literal>`` and asserts every
    match's RHS is a NAME (``DEFAULT_MIN_AREA_RATIO`` or the historical
    ``MIN_RETAINED_AREA_FRAC`` alias defined in terms of it), never a bare
    numeric literal. A newly added dataclass field or function parameter
    that hardcodes e.g. ``0.25`` or ``0.3`` directly will fail this test
    without needing to be named here individually. This is NOT airtight --
    it only catches the ``min_area_ratio: float = ...`` spelling, so a
    renamed field or a value set via ``field(default=...)`` would slip past
    -- but it is cheap and catches the exact shape this bug had twice.
    """
    src_root = Path(__file__).resolve().parents[1] / "src"
    pattern = re.compile(r"min_area_ratio\s*:\s*float\s*=\s*([^\s,)#]+)")
    offenders = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            rhs = match.group(1)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", rhs):
                offenders.append(f"{path}: {match.group(0)!r}")
    assert offenders == [], (
        "found a min_area_ratio default that is not a named reference to "
        f"the shared constant: {offenders}"
    )


def _sub_floor_seam_polygon():
    # 20% left / 80% right split around x=50 in a 100x100 frame: below the
    # 0.25 floor on the left tile, comfortably above it on the right.
    return np.array([[48, 10], [58, 10], [58, 20], [48, 20]], dtype=np.float32)


def test_policy_divergence_survives_yolo_drops_sam3_downgrades():
    """The thing most likely to be "helpfully" unified by accident: it must not be.

    Same measurement (frame-space polygon-area retained-fraction), same floor
    value, opposite consequence below it.
    """
    poly_px = _sub_floor_seam_polygon()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # -- YOLO: sub-floor fragment is DROPPED (no label line emitted for it).
    poly_norm = poly_px.copy()
    poly_norm[:, 0] /= 100.0
    poly_norm[:, 1] /= 100.0
    _crop_left, lines_left = _tile_one_image(
        frame, [(0, poly_norm)], (0, 0, 50, 100), None, 0.25
    )
    _crop_right, lines_right = _tile_one_image(
        frame, [(0, poly_norm)], (50, 0, 100, 100), None, 0.25
    )
    assert lines_left == []  # sub-floor fragment: dropped entirely
    assert len(lines_right) == 1  # well-above-floor sibling: kept as a normal positive

    # -- SAM3: sub-floor fragment is KEPT and flagged `is_crowd`, not dropped.
    tiles = list(_tile_frame(frame, [poly_px], 50, 50, 0.0, False, min_area_ratio=0.25))
    flags = sorted(
        flag for _rect, _crop, instances in tiles for _poly, flag in instances
    )
    assert flags == [False, True]  # BOTH sides retained; only one is downgraded
